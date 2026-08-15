"""Infer request-shaping and execution owners for Ollmo."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from flask import jsonify

from helpers.model_capabilities import CAPABILITY_SPEECH_TO_TEXT


def _normalize_structured_text_artifact_extension(value: Any) -> str:
    token = str(value or '').strip().lower().lstrip('.')
    aliases = {
        'markdown': 'md',
        'plain text': 'txt',
        'text': 'txt',
    }
    return aliases.get(token, token)


def _structured_text_artifact_requests_from_payload(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, Mapping):
        return []
    raw_requests: list[Any] = []
    if isinstance(payload.get('text_artifact_requests'), list):
        raw_requests.extend(payload.get('text_artifact_requests') or [])
    if isinstance(payload.get('artifact_request'), Mapping):
        raw_requests.append(payload.get('artifact_request'))
    if str(payload.get('text_artifact_extension') or '').strip():
        raw_requests.append(
            {
                'extension': payload.get('text_artifact_extension'),
                'source_name': payload.get('text_artifact_source_name'),
                'source': payload.get('text_artifact_source') or 'structured_text_artifact_contract',
                'target_path': payload.get('text_artifact_target_path'),
            }
        )

    requests: list[dict[str, str]] = []
    request_index_by_name: dict[tuple[str, str], int] = {}
    request_index_by_target: dict[str, int] = {}
    for raw_request in raw_requests:
        if not isinstance(raw_request, Mapping):
            continue
        extension = _normalize_structured_text_artifact_extension(raw_request.get('extension'))
        source_name = str(raw_request.get('source_name') or '').strip() or f'generated-{extension}'
        source = str(raw_request.get('source') or '').strip() or 'structured_text_artifact_contract'
        target_path = str(raw_request.get('target_path') or '').strip()
        if not extension:
            continue
        name_identity = (extension, source_name.casefold())
        if target_path and target_path in request_index_by_target:
            continue
        existing_index = request_index_by_name.get(name_identity)
        if existing_index is not None:
            existing_target = str(
                requests[existing_index].get('target_path') or ''
            ).strip()
            if not existing_target and target_path:
                requests[existing_index]['target_path'] = target_path
                request_index_by_target[target_path] = existing_index
                continue
            if not target_path or existing_target == target_path:
                continue
        request_index = len(requests)
        requests.append(
            {
                'extension': extension,
                'source_name': source_name,
                'source': source,
                **(
                    {
                        'target_path': target_path
                    }
                    if target_path
                    else {}
                ),
            }
        )
        request_index_by_name.setdefault(name_identity, request_index)
        if target_path:
            request_index_by_target[target_path] = request_index
    return requests


@dataclass
class InferRuntimeOwner:
    hooks: dict[str, Any]
    capability_embedding: str
    capability_image_generation: str
    capability_vision_analysis: str
    runtime_status_path_getter: Any
    request_timeout_error: type[Exception]
    request_connection_error: type[Exception]
    request_exception_error: type[Exception]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def apply_required_session_control_defaults(
        self,
        data: dict[str, Any],
        *,
        instance: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        updated = dict(data or {})
        instance_payload = instance if isinstance(instance, dict) else {}
        session_controls = instance_payload.get('session_controls') if isinstance(instance_payload.get('session_controls'), dict) else {}
        fields = session_controls.get('fields') if isinstance(session_controls.get('fields'), dict) else {}
        applied: dict[str, Any] = {}
        request_keys = self._hook('session_control_request_keys')
        for field_key, field in fields.items():
            if not isinstance(field, dict):
                continue
            if field.get('visible') is False or not field.get('required'):
                continue
            request_key = request_keys.get(str(field_key))
            if not request_key:
                continue
            raw_value = updated.get(request_key)
            if raw_value is not None and str(raw_value).strip() != '':
                continue
            default_value = field.get('default_value')
            if default_value not in (None, ''):
                updated[request_key] = default_value
                applied[request_key] = default_value
                continue
            options = [
                str(item or '').strip()
                for item in (field.get('options') or [])
                if str(item or '').strip()
            ]
            if field.get('default_first_option') and options:
                updated[request_key] = options[0]
                applied[request_key] = options[0]
        return updated, applied

    def build_missing_required_session_controls(self, instance: dict[str, Any], data: Any) -> list[dict[str, Any]]:
        schema = instance.get('session_controls') if isinstance(instance.get('session_controls'), dict) else {}
        fields = schema.get('fields') if isinstance(schema.get('fields'), dict) else {}
        missing: list[dict[str, Any]] = []
        request_keys = self._hook('session_control_request_keys')
        for field_key, field in fields.items():
            if not isinstance(field, dict):
                continue
            if field.get('visible') is False or not field.get('required'):
                continue
            request_key = request_keys.get(str(field_key))
            if not request_key:
                continue
            raw_value = data.get(request_key)
            if raw_value is None or str(raw_value).strip() == '':
                missing.append(
                    {
                        'field_key': str(field_key),
                        'request_key': request_key,
                        'label': str(field.get('label') or field_key).strip(),
                        'message': str(field.get('required_message') or '').strip()
                        or f"{str(field.get('label') or field_key).strip()} is required.",
                    }
                )
        return missing

    def prepare_effective_request_data(
        self,
        data: Any,
        *,
        route_info: Optional[dict[str, Any]] = None,
        instance: Optional[dict[str, Any]] = None,
        compute_semantics: bool = True,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        normalize_request_payload = self._hook('normalize_request_payload')
        extract_responses_prompt = self._hook('extract_responses_prompt')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')
        normalize_capability = self._hook('normalize_capability')
        plan_compound_execution_payload = self._hook('plan_compound_execution_payload')
        apply_prompt_control_hints = self._hook('apply_prompt_control_hints')
        extract_ghost_route_messages = self._hook('extract_ghost_route_messages')
        extract_semantic_materializer_prompt = self._hook('extract_semantic_materializer_prompt')
        merge_request_meta_runtime_truth = self._hook('merge_request_meta_runtime_truth')
        build_working_frame = self._hook('build_working_frame')

        effective_data = normalize_request_payload(data)
        effective_data_for_hints = dict(effective_data)
        current_turn_prompt = (
            extract_responses_current_turn_prompt(effective_data_for_hints)
            or extract_responses_prompt(effective_data_for_hints)
        )
        effective_data_for_hints['_current_turn_prompt'] = current_turn_prompt
        effective_data_for_hints['_prompt_hint'] = current_turn_prompt
        effective_capability = normalize_capability(
            (route_info or {}).get('capability')
            or (instance or {}).get('capability')
            or effective_data.get('capability')
        )
        planner_meta: dict[str, Any] = {}
        if route_info and compute_semantics:
            effective_data_for_hints, planner_meta = plan_compound_execution_payload(
                effective_data_for_hints,
                route_info=route_info,
            )
        effective_data_for_hints, control_hints = apply_prompt_control_hints(
            effective_data_for_hints,
            capability=effective_capability,
            instance=instance if isinstance(instance, dict) else None,
            context_messages=extract_ghost_route_messages(data),
        )
        effective_data_for_hints, required_control_defaults = self.apply_required_session_control_defaults(
            effective_data_for_hints,
            instance=instance if isinstance(instance, dict) else None,
        )
        if required_control_defaults:
            control_hints = {
                **required_control_defaults,
                **control_hints,
            }
        semantic_materializer_prompt = extract_semantic_materializer_prompt(
            effective_data_for_hints,
            capability=effective_capability,
        )
        if semantic_materializer_prompt:
            effective_data_for_hints['prompt'] = semantic_materializer_prompt
        effective_data = {
            key: value
            for key, value in effective_data_for_hints.items()
            if key not in {'_prompt_hint', '_current_turn_prompt'}
        }

        enriched_route_info = route_info
        if route_info and (planner_meta.get('attempted') or control_hints):
            enriched_route_info = dict(route_info)
            route_runtime = dict(enriched_route_info.get('route_runtime') or {})
            if planner_meta.get('attempted'):
                route_runtime['execution_planner'] = planner_meta
                semantic_compute = (
                    dict(route_runtime.get('semantic_compute'))
                    if isinstance(route_runtime.get('semantic_compute'), dict)
                    else None
                )
                if semantic_compute is not None:
                    semantic_compute['performed'] = True
                    semantic_compute['evidence_role'] = 'preview_computed_non_learnable'
                    semantic_compute['learnable'] = False
                    route_runtime['semantic_compute'] = semantic_compute
            if control_hints:
                route_runtime['control_hints'] = control_hints
            route_runtime = merge_request_meta_runtime_truth(
                route_runtime,
                effective_data,
                planner_meta=planner_meta,
                route_payload=enriched_route_info,
            )
            enriched_route_info['route_runtime'] = route_runtime
            request_meta = route_runtime.get('request_meta') if isinstance(route_runtime.get('request_meta'), dict) else {}
            if request_meta:
                enriched_route_info['request_meta'] = request_meta
        elif route_info:
            enriched_route_info = dict(route_info)
            route_runtime = merge_request_meta_runtime_truth(
                enriched_route_info.get('route_runtime') if isinstance(enriched_route_info.get('route_runtime'), dict) else {},
                effective_data,
                planner_meta=planner_meta,
                route_payload=enriched_route_info,
            )
            enriched_route_info['route_runtime'] = route_runtime
            request_meta = route_runtime.get('request_meta') if isinstance(route_runtime.get('request_meta'), dict) else {}
            if request_meta:
                enriched_route_info['request_meta'] = request_meta

        if enriched_route_info:
            route_runtime = dict(enriched_route_info.get('route_runtime') or {})
            route_runtime['working_frame'] = build_working_frame(
                request_payload=effective_data,
                route_payload=enriched_route_info,
                freeze=False,
            )
            enriched_route_info['route_runtime'] = route_runtime

        return effective_data, enriched_route_info, planner_meta, control_hints

    def build_responses_infer_execution_payload(
        self,
        data: Any,
        *,
        route_info: Optional[dict[str, Any]],
        instance: dict[str, Any],
        instance_id: str,
        backend: str,
        capability: str,
        request_model_override: Optional[str],
        upload_present: bool = False,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], bool, bool]:
        extract_selected_reference_artifacts = self._hook('extract_selected_reference_artifacts')
        select_matching_selected_reference_artifact = self._hook('select_matching_selected_reference_artifact')
        extract_responses_prompt = self._hook('extract_responses_prompt')
        should_attach_selected_reference_file_context = self._hook('should_attach_selected_reference_file_context')
        translate_responses_payload_to_infer_payload = self._hook('translate_responses_payload_to_infer_payload')
        apply_selected_reference_prompt_prefix = self._hook('apply_selected_reference_prompt_prefix')
        coerce_seed = self._hook('coerce_seed')
        find_image_artifact_seed = self._hook('find_image_artifact_seed')
        extract_ghost_route_messages = self._hook('extract_ghost_route_messages')
        choose_context_strategy = self._hook('choose_context_strategy')
        normalize_capability = self._hook('normalize_capability')
        sanitize_artifact_records = self._hook('sanitize_artifact_records')
        parse_bool = self._hook('parse_bool')

        normalized_payload = data if isinstance(data, dict) else dict(data)
        explicit_file_path = str(normalized_payload.get('file_path') or '').strip()
        suppress_reference_file_context = parse_bool(
            normalized_payload.get('suppress_reference_file_context'),
            default=False,
        )
        if (
            suppress_reference_file_context
            and isinstance(route_info, dict)
            and (
                route_info.get('route_reuse_last_artifact')
                or route_info.get('route_artifact_ref')
                or route_info.get('route_artifact_path')
            )
        ):
            route_info = dict(route_info)
            route_info['route_reuse_last_artifact'] = False
            route_info['route_artifact_ref'] = None
            route_info['route_artifact_path'] = None
            if isinstance(route_info.get('route_runtime'), dict):
                route_runtime = dict(route_info['route_runtime'])
                route_runtime.pop('route_reuse_last_artifact', None)
                route_runtime.pop('route_artifact_ref', None)
                route_runtime.pop('route_artifact_path', None)
                route_info['route_runtime'] = route_runtime
        expose_input_artifacts = bool(upload_present or explicit_file_path)
        input_artifacts = sanitize_artifact_records(normalized_payload.get('input_artifacts'))
        selected_reference_artifacts = extract_selected_reference_artifacts(normalized_payload)
        matched_selected_reference = select_matching_selected_reference_artifact(
            selected_reference_artifacts,
            capability,
            instance=instance,
        )
        responses_prompt = extract_responses_prompt(normalized_payload)
        raw_file_path = explicit_file_path
        raw_file_path_source = 'explicit_file_path' if explicit_file_path else ''
        if (
            not raw_file_path
            and not suppress_reference_file_context
            and route_info
            and route_info.get('route_reuse_last_artifact')
        ):
            raw_file_path = str(route_info.get('route_artifact_path') or '').strip()
            if raw_file_path:
                raw_file_path_source = 'route_reuse'
        if (
            not raw_file_path
            and not upload_present
            and not suppress_reference_file_context
            and should_attach_selected_reference_file_context(
                prompt=responses_prompt,
                capability=capability,
                selected_reference_artifact=matched_selected_reference,
            )
        ):
            raw_file_path = str(matched_selected_reference.get('path') or '').strip()
            if raw_file_path:
                raw_file_path_source = 'selected_reference'
        if not raw_file_path and normalize_capability(capability) == CAPABILITY_SPEECH_TO_TEXT:
            for artifact in input_artifacts:
                artifact_type = str(artifact.get('type') or artifact.get('kind') or '').strip().lower()
                artifact_path = str(artifact.get('path') or artifact.get('source_path') or '').strip()
                if artifact_path and artifact_type in {'audio', 'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus'}:
                    raw_file_path = artifact_path
                    raw_file_path_source = (
                        'input_artifact'
                        if self.artifact_is_external_input(artifact)
                        else 'artifact_reference'
                    )
                    expose_input_artifacts = self.artifact_is_external_input(artifact)
                    break
        has_file_context = bool(upload_present or raw_file_path)

        infer_payload = translate_responses_payload_to_infer_payload(normalized_payload)
        structured_text_artifact_requests = _structured_text_artifact_requests_from_payload(normalized_payload)
        if structured_text_artifact_requests:
            infer_payload['text_artifact_requests'] = structured_text_artifact_requests
            infer_payload.setdefault('artifact_request', structured_text_artifact_requests[0])
        for key in (
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'branch_id',
            'phase_id',
            'obligation_id',
            'task_id',
            'workload_task_id',
            'output_type',
            'depends_on',
            'input_refs',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'requires_artifact',
            'suppress_reference_file_context',
            'suppress_image_state_enrichment',
            'suppress_generated_image_enrichment',
            'image_state_enrichment_suppression_reason',
        ):
            value = normalized_payload.get(key)
            if value not in (None, '', [], {}):
                infer_payload[key] = value
        infer_payload['instance_id'] = instance_id
        if backend:
            infer_payload['backend'] = backend
        route_capability = normalize_capability(capability)
        if route_capability:
            infer_payload['capability'] = route_capability
        route_model = str((instance or {}).get('model') or '').strip()
        if route_model:
            infer_payload['model'] = route_model
        if request_model_override:
            infer_payload['request_model'] = request_model_override
        selected_reference_prompt = None
        if not suppress_reference_file_context:
            selected_reference_prompt = apply_selected_reference_prompt_prefix(
                infer_payload.get('prompt'),
                selected_reference_artifacts,
                capability,
            )
        if selected_reference_prompt:
            infer_payload['prompt'] = selected_reference_prompt
        if raw_file_path:
            infer_payload['file_path'] = raw_file_path
        bindings: list[dict[str, Any]] = []
        if selected_reference_artifacts and not suppress_reference_file_context:
            infer_payload['reference_artifacts'] = [dict(item) for item in selected_reference_artifacts if isinstance(item, dict)]
            for artifact in selected_reference_artifacts:
                if not isinstance(artifact, dict):
                    continue
                resolved_path = str(artifact.get('path') or '').strip() or None
                if not resolved_path:
                    continue
                bindings.append(
                    {
                        'binding_kind': 'direct_reference',
                        'artifact_ref': str(artifact.get('artifact_ref') or artifact.get('ref') or '').strip() or None,
                        'resolved_path': resolved_path,
                    }
                )
        route_artifact_ref = str((route_info or {}).get('route_artifact_ref') or '').strip() or None
        if (
            route_info
            and not suppress_reference_file_context
            and route_info.get('route_reuse_last_artifact')
            and route_artifact_ref
            and raw_file_path
            and not any(str(item.get('artifact_ref') or '').strip() == route_artifact_ref for item in bindings)
        ):
            bindings.append(
                {
                    'binding_kind': 'route_reuse',
                    'artifact_ref': route_artifact_ref,
                    'resolved_path': raw_file_path,
                }
            )
        if bindings:
            infer_payload['artifact_bindings'] = bindings
        if raw_file_path_source in {'route_reuse', 'selected_reference', 'artifact_reference'}:
            expose_input_artifacts = False
        response_id = str(normalized_payload.get('response_id') or normalized_payload.get('responseId') or '').strip()
        if response_id:
            infer_payload['response_id'] = response_id
        conversation_id = str(normalized_payload.get('conversation_id') or normalized_payload.get('conversationId') or '').strip()
        if conversation_id:
            infer_payload['conversation_id'] = conversation_id
        request_id = str(normalized_payload.get('request_id') or normalized_payload.get('requestId') or '').strip()
        if request_id:
            infer_payload['request_id'] = request_id
        if route_info:
            route_source = str(route_info.get('route_source') or '').strip()
            route_reason = str(route_info.get('route_reason') or '').strip()
            infer_payload['provenance_origin'] = 'responses_late_fill' if route_source in {'late_fill', 'phase_continuation'} else 'responses'
            if route_source:
                infer_payload['provenance_route_source'] = route_source
            if route_reason:
                infer_payload['provenance_route_reason'] = route_reason
        if normalize_capability(capability) == self.capability_image_generation and infer_payload.get('seed') in (None, '') and raw_file_path:
            inferred_seed = None
            if matched_selected_reference and str(matched_selected_reference.get('type') or '').strip().lower() != 'message':
                selected_path = str((matched_selected_reference or {}).get('path') or '').strip()
                if selected_path == raw_file_path:
                    inferred_seed = coerce_seed((matched_selected_reference or {}).get('seed'))
            if inferred_seed is None:
                inferred_seed = find_image_artifact_seed(
                    extract_ghost_route_messages(normalized_payload, include_selected_reference=True),
                    raw_file_path,
                )
            if inferred_seed is not None:
                infer_payload['seed'] = inferred_seed

        updated_route_info = route_info
        if route_info and has_file_context:
            updated_route_info = dict(route_info)
            route_runtime = dict(updated_route_info.get('route_runtime') or {})
            route_runtime['context_strategy'] = choose_context_strategy(
                instance=instance,
                messages=[],
                prompt=extract_responses_prompt(normalized_payload),
                has_file_context=True,
            )
            updated_route_info['route_runtime'] = route_runtime

        return infer_payload, updated_route_info, has_file_context, expose_input_artifacts

    def artifact_is_external_input(self, value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        origin = str(value.get('origin') or value.get('source') or '').strip().lower()
        if origin in {'upload', 'user_upload', 'external_input', 'external_file'}:
            return True
        if origin == 'local_path':
            internal_tokens = (
                value.get('artifact_ref'),
                value.get('artifactRef'),
                value.get('source_response_id'),
                value.get('response_id'),
                value.get('response_instance_id'),
                value.get('provenance_id'),
                value.get('derived_from'),
            )
            return not any(str(token or '').strip() for token in internal_tokens)
        return False

    def artifact_is_reference_reuse(self, value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        if self.artifact_is_external_input(value):
            return False
        origin = str(value.get('origin') or value.get('source') or '').strip().lower()
        if origin in {
            'assistant_output',
            'generated_output',
            'ollmo_output',
            'conversation_reference',
            'selected_reference',
            'registry_reference',
            'late_fill_output',
            'response_output',
        }:
            return True
        if str(value.get('binding_kind') or '').strip():
            return True
        for key in (
            'source_response_id',
            'response_instance_id',
            'provenance_id',
            'derived_from',
        ):
            if str(value.get(key) or '').strip():
                return True
        return False

    def artifact_identity_tokens(self, value: Any) -> set[str]:
        extract_artifact_ref = self._hook('extract_artifact_ref')
        if not isinstance(value, Mapping):
            return set()
        tokens: set[str] = set()
        for candidate in (
            value.get('artifact_ref'),
            value.get('artifactRef'),
            value.get('ref'),
            value.get('path'),
            value.get('source_path'),
            value.get('sourcePath'),
            value.get('resolved_path'),
            value.get('resolvedPath'),
            value.get('artifact_path'),
            value.get('artifactPath'),
        ):
            token = str(candidate or '').strip()
            if token:
                tokens.add(token)
        extracted_ref = extract_artifact_ref(value)
        if extracted_ref:
            tokens.add(str(extracted_ref).strip())
        return tokens

    def normalize_reference_mirror_input_artifacts(self, payload: Any) -> dict[str, Any]:
        sanitize_artifact_records = self._hook('sanitize_artifact_records')
        merge_unique_artifact_records = self._hook('merge_unique_artifact_records')
        if not isinstance(payload, dict):
            return {}
        filtered = dict(payload)
        input_artifacts = sanitize_artifact_records(filtered.get('input_artifacts'))
        if not input_artifacts:
            filtered.pop('input_artifacts', None)
            return filtered
        reference_artifacts = sanitize_artifact_records(filtered.get('reference_artifacts'))
        artifact_bindings = filtered.get('artifact_bindings') if isinstance(filtered.get('artifact_bindings'), list) else []
        reference_tokens: set[str] = set()
        for artifact in reference_artifacts:
            reference_tokens.update(self.artifact_identity_tokens(artifact))
        for binding in artifact_bindings:
            if isinstance(binding, Mapping):
                reference_tokens.update(self.artifact_identity_tokens(binding))
        kept_inputs: list[dict[str, Any]] = []
        mirrored_references: list[dict[str, Any]] = []
        for artifact in input_artifacts:
            artifact_tokens = self.artifact_identity_tokens(artifact)
            if (
                (artifact_tokens and artifact_tokens.intersection(reference_tokens))
                or self.artifact_is_reference_reuse(artifact)
            ):
                mirrored_references.append(dict(artifact))
                continue
            kept_inputs.append(dict(artifact))
        if mirrored_references:
            merged_references = merge_unique_artifact_records(reference_artifacts, mirrored_references)
            if merged_references:
                filtered['reference_artifacts'] = merged_references
        if kept_inputs:
            filtered['input_artifacts'] = kept_inputs
        else:
            filtered.pop('input_artifacts', None)
        return filtered

    def filter_responses_infer_result(
        self,
        payload: Any,
        *,
        expose_input_artifacts: bool,
    ) -> dict[str, Any]:
        filtered = dict(payload or {}) if isinstance(payload, dict) else {}
        filtered = self.normalize_reference_mirror_input_artifacts(filtered)
        if not expose_input_artifacts:
            filtered.pop('input_artifacts', None)
        return filtered

    def execute_infer_request(self, data: Any, *, upload=None):
        rewind_upload_stream = self._hook('rewind_upload_stream')
        normalize_external_identifier = self._hook('normalize_external_identifier')
        lookup_instance = self._hook('lookup_instance')
        normalize_backend = self._hook('normalize_backend')
        select_backend_request_model = self._hook('select_backend_request_model')
        normalize_capability = self._hook('normalize_capability')
        infer_capability = self._hook('infer_capability')
        extract_semantic_materializer_prompt = self._hook('extract_semantic_materializer_prompt')
        extract_selected_reference_artifacts = self._hook('extract_selected_reference_artifacts')
        select_matching_selected_reference_artifact = self._hook('select_matching_selected_reference_artifact')
        should_attach_selected_reference_file_context = self._hook('should_attach_selected_reference_file_context')
        apply_selected_reference_prompt_prefix = self._hook('apply_selected_reference_prompt_prefix')
        sanitize_artifact_records = self._hook('sanitize_artifact_records')
        parse_float_with_bounds = self._hook('parse_float_with_bounds')
        parse_int_with_bounds = self._hook('parse_int_with_bounds')
        parse_bool = self._hook('parse_bool')
        build_runtime_status_stub = self._hook('build_runtime_status_stub')
        build_infer_dedupe_key = self._hook('build_infer_dedupe_key')
        acquire_infer_slot = self._hook('acquire_infer_slot')
        release_infer_slot = self._hook('release_infer_slot')
        log_unified_event = self._hook('log_unified_event')
        file_kind_from_name = self._hook('file_kind_from_name')
        save_upload_to_temp = self._hook('save_upload_to_temp')
        save_local_path_to_temp = self._hook('save_local_path_to_temp')
        persist_request_input_artifacts = self._hook('persist_request_input_artifacts')
        persist_input_artifact_registry_records = self.hooks.get('persist_input_artifact_registry_records')
        find_artifact_registry_record = self.hooks.get('find_artifact_registry_record')
        to_base64 = self._hook('to_base64')
        read_text_file = self._hook('read_text_file')
        hash_file_sha256 = self._hook('hash_file_sha256')
        find_cached_pdf_insight = self._hook('find_cached_pdf_insight')
        extract_pdf_text_content = self._hook('extract_pdf_text_content')
        render_pdf_pages_to_base64 = self._hook('render_pdf_pages_to_base64')
        log_pdf_infer_event = self._hook('log_pdf_infer_event')
        record_instance_activity = self._hook('record_instance_activity')
        record_instance_success = self._hook('record_instance_success')
        record_instance_failure = self._hook('record_instance_failure')
        log_runtime_status_transition = self._hook('log_runtime_status_transition')
        InferContext = self._hook('InferContext')
        InferArtifacts = self._hook('InferArtifacts')
        dispatch_infer_request = self._hook('dispatch_infer_request')
        whisper_transcribe = self._hook('whisper_transcribe')
        mlx_audio_speech = self._hook('mlx_audio_speech')
        ollama_generate = self._hook('ollama_generate')
        extract_saved_image_path_from_generate_output = self._hook('extract_saved_image_path_from_generate_output')
        extract_image_data_url_from_generate_output = self._hook('extract_image_data_url_from_generate_output')
        extract_generate_seed = self._hook('extract_generate_seed')
        ollama_openai_image_generation = self._hook('ollama_openai_image_generation')
        persist_audio_bytes_locally = self._hook('persist_audio_bytes_locally')
        persist_image_data_url_locally = self._hook('persist_image_data_url_locally')
        extract_generate_content = self._hook('extract_generate_content')
        persist_text_artifact_locally = self._hook('persist_text_artifact_locally')
        persist_text_markdown_locally = self._hook('persist_text_markdown_locally')
        persist_transcript_text_locally = self._hook('persist_transcript_text_locally')
        ocr_pdf_page_with_ollama = self._hook('ocr_pdf_page_with_ollama')
        render_single_pdf_page_to_base64 = self._hook('render_single_pdf_page_to_base64')
        ocr_image_with_deepseek = self._hook('ocr_image_with_deepseek')
        is_generic_ocr_instruction_prompt = self._hook('is_generic_ocr_instruction_prompt')
        clean_ocr_output_text = self._hook('clean_ocr_output_text')
        looks_like_ocr_prompt_echo = self._hook('looks_like_ocr_prompt_echo')
        ollama_chat = self._hook('ollama_chat')
        openai_chat_completions = self._hook('openai_chat_completions')
        mlx_chat_completions = self._hook('mlx_chat_completions')
        enrich_generated_image_payload = self._hook('enrich_generated_image_payload')
        persist_generated_image_provenance_for_infer_result = self._hook('persist_generated_image_provenance_for_infer_result')
        coerce_seed = self._hook('coerce_seed')

        request_timeout_error = self.request_timeout_error
        request_connection_error = self.request_connection_error
        request_exception_error = self.request_exception_error

        rewind_upload_stream(upload)

        try:
            instance_id = normalize_external_identifier(data.get("instance_id"), field_name='instance_id')
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        instance = lookup_instance(instance_id)
        if not instance:
            return jsonify({"error": f"Instance '{instance_id}' was not found."}), 404

        backend = normalize_backend(data.get("backend") or instance.get("backend"))
        model_name = str(data.get("model") or instance.get("model") or "").strip()
        request_model_name = select_backend_request_model(
            instance,
            data.get("request_model") or instance.get("request_model"),
            model_name,
        )
        capability = normalize_capability(data.get("capability")) or normalize_capability(instance.get("capability"))
        if not capability:
            capability = infer_capability(model_name, backend)
        if capability == self.capability_embedding:
            return jsonify(
                {
                    "error": (
                        "Embedding instances are reserved for internal Ghost pre-ranking and "
                        "are not executed through /api/infer."
                    )
                }
            ), 400

        if backend not in {"ollama", "mlx", "llama_cpp"}:
            return jsonify({"error": f"Unknown backend type '{backend}'."}), 400

        port = instance.get("port")
        if not port:
            return jsonify({"error": "Instance has no target port."}), 400
        try:
            port = int(port)
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid target port '{port}'."}), 400

        prompt = str(data.get("prompt") or "").strip()
        semantic_materializer_prompt = extract_semantic_materializer_prompt(
            data,
            capability=capability,
        )
        if semantic_materializer_prompt:
            prompt = semantic_materializer_prompt
        user_prompt = prompt
        text_artifact_requests = _structured_text_artifact_requests_from_payload(data)
        raw_file_path = str(data.get("file_path") or "").strip()
        suppress_reference_file_context = parse_bool(
            data.get('suppress_reference_file_context'),
            default=False,
        )
        reference_artifacts = extract_selected_reference_artifacts(data)
        request_input_artifacts = sanitize_artifact_records(data.get('input_artifacts'))
        matched_selected_reference = select_matching_selected_reference_artifact(
            reference_artifacts,
            capability,
            instance=instance,
        )
        if (
            not raw_file_path
            and not (upload and getattr(upload, "filename", None))
            and not suppress_reference_file_context
            and should_attach_selected_reference_file_context(
                prompt=prompt,
                capability=capability,
                selected_reference_artifact=matched_selected_reference,
            )
        ):
            raw_file_path = str(matched_selected_reference.get("path") or "").strip()
        reference_backed_file_path = False
        normalized_raw_file_path = str(raw_file_path or '').strip()
        artifact_bindings = data.get('artifact_bindings') if isinstance(data.get('artifact_bindings'), list) else []
        if normalized_raw_file_path and isinstance(reference_artifacts, list):
            for artifact in [*reference_artifacts, *request_input_artifacts, *artifact_bindings]:
                if not isinstance(artifact, dict):
                    continue
                artifact_path = str(artifact.get('path') or artifact.get('resolved_path') or '').strip()
                artifact_tokens = self.artifact_identity_tokens(artifact)
                if artifact_path and artifact_path == normalized_raw_file_path:
                    reference_backed_file_path = True
                    break
                if normalized_raw_file_path in artifact_tokens:
                    reference_backed_file_path = True
                    break
            if not reference_backed_file_path and callable(find_artifact_registry_record):
                try:
                    registry_record = find_artifact_registry_record(normalized_raw_file_path)
                except Exception:
                    registry_record = None
                if isinstance(registry_record, Mapping):
                    reference_backed_file_path = True
        if not suppress_reference_file_context:
            prompt = apply_selected_reference_prompt_prefix(
                prompt,
                reference_artifacts,
                capability,
            )
        task = str(data.get("task") or "transcribe").strip().lower() or "transcribe"
        if task not in {"transcribe", "translate"}:
            return jsonify({"error": "Invalid value for 'task'. Allowed: transcribe, translate."}), 400
        language = str(data.get("language") or "").strip() or None
        voice = str(data.get("voice") or "").strip() or None
        instruct = str(data.get("instruct") or "").strip() or None
        ocr_mode = str(data.get("ocr_mode") or "").strip().lower() or None
        raw_width = data.get("width")
        raw_height = data.get("height")
        raw_seed = data.get("seed")
        lang_code = str(data.get("lang_code") or "").strip() or None
        response_format = str(data.get("response_format") or "").strip().lower() or None
        if response_format and not re.fullmatch(r"[a-z0-9]+", response_format):
            return jsonify({"error": "Invalid value for 'response_format'."}), 400
        try:
            speed = parse_float_with_bounds(data.get("speed"), default=1.0, minimum=0.5, maximum=2.0)
            pitch = parse_float_with_bounds(data.get("pitch"), default=1.0, minimum=0.5, maximum=2.0)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        has_width = raw_width not in (None, '')
        has_height = raw_height not in (None, '')
        if has_width != has_height:
            return jsonify({"error": "Please provide Width and Height together, or leave both empty."}), 400
        image_width: Optional[int] = None
        image_height: Optional[int] = None
        image_seed: Optional[int] = None
        if has_width and has_height:
            try:
                image_width = parse_int_with_bounds(raw_width, default=1024, minimum=64, maximum=4096)
                image_height = parse_int_with_bounds(raw_height, default=1024, minimum=64, maximum=4096)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            if image_width % 16 != 0 or image_height % 16 != 0:
                return jsonify({"error": "Width and Height must be multiples of 16."}), 400
            if image_width * image_height > 4_194_304:
                return jsonify({"error": "Width × Height must not exceed 4 MP."}), 400
        if raw_seed not in (None, ''):
            try:
                image_seed = parse_int_with_bounds(raw_seed, default=0, minimum=0, maximum=2_147_483_647)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        reuse_cached_pdf = parse_bool(data.get("reuse_cached"), default=True)
        pdf_prefer_text = parse_bool(data.get("pdf_prefer_text"), default=False)
        infer_timeout_min = 5 if parse_bool(data.get("internal_fast_timeout"), default=False) else 60
        infer_timeout_sec = parse_int_with_bounds(
            data.get("infer_timeout_sec"),
            default=1200,
            minimum=infer_timeout_min,
            maximum=7200,
        )
        pdf_synthesize = parse_bool(data.get("pdf_synthesize"), default=False)
        pdf_page_timeout_sec = parse_int_with_bounds(
            data.get("pdf_page_timeout_sec"),
            default=min(240, infer_timeout_sec),
            minimum=45,
            maximum=1800,
        )
        pdf_max_image_side = parse_int_with_bounds(
            data.get("pdf_max_image_side"),
            default=2400,
            minimum=1200,
            maximum=6000,
        )
        pdf_page_retry_dpi_raw = data.get("pdf_page_retry_dpi")

        temp_path: Optional[Path] = None
        resolved_source: Optional[Path] = None
        infer_slot_key: Optional[str] = None
        file_kind = ""
        file_name = ""
        input_artifacts: list[dict[str, Any]] = []
        file_sha256 = ""
        image_b64: Optional[str] = None
        text_from_file = ""
        text_from_file_truncated = False
        text_from_file_inline_bytes = 0
        text_from_file_total_bytes = 0
        pdf_page_images: list[str] = []
        pdf_warnings: list[str] = []
        pdf_total_pages = 0
        pdf_render_dpi = 180
        pdf_page_retry_dpi = 120
        runtime_status_instance = build_runtime_status_stub(
            instance_id=instance_id,
            model=model_name,
            backend=backend,
            capability=capability,
            port=port,
            pid=instance.get("pid"),
        ) or instance
        try:
            infer_slot_key = build_infer_dedupe_key(
                instance_id=instance_id,
                backend=backend,
                capability=capability,
                model_name=model_name,
                prompt=prompt,
                upload=upload,
                local_file_path=raw_file_path,
            )
            if not acquire_infer_slot(infer_slot_key):
                return jsonify(
                    {
                        "error": (
                            "An identical request is already running. "
                            "Please wait for the in-flight result."
                        )
                    }
                ), 409
            logging.info(
                "/api/infer request accepted: instance=%s capability=%s backend=%s file=%s",
                instance_id,
                capability,
                backend,
                str(getattr(upload, "filename", "") or raw_file_path or ""),
            )
            log_unified_event(
                category="infer",
                action="request",
                status="started",
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                file_kind=file_kind,
                file_name=file_name or raw_file_path,
                message="Infer request accepted.",
            )

            if upload and upload.filename and raw_file_path:
                return jsonify({"error": "Please send either 'file' or 'file_path', not both at the same time."}), 400

            if upload and upload.filename:
                file_name = str(upload.filename or "").strip()
                file_kind = file_kind_from_name(upload.filename)
                temp_path = save_upload_to_temp(upload)
            elif raw_file_path:
                try:
                    resolved_source, temp_path = save_local_path_to_temp(raw_file_path)
                except (ValueError, FileNotFoundError, IsADirectoryError, OSError) as exc:
                    return jsonify({"error": str(exc)}), 400
                file_name = resolved_source.name
                file_kind = file_kind_from_name(file_name)
                logging.info(
                    "Using local file path input: source=%s kind=%s size_mb=%.2f",
                    resolved_source,
                    file_kind,
                    resolved_source.stat().st_size / (1024 * 1024),
                )

            if temp_path and file_name:
                if not (raw_file_path and reference_backed_file_path):
                    input_artifacts = persist_request_input_artifacts(
                        temp_path=temp_path,
                        file_name=file_name,
                        file_kind=file_kind,
                        upload=upload,
                        source_path=str(resolved_source or raw_file_path or '').strip(),
                    )
                    if input_artifacts and callable(persist_input_artifact_registry_records):
                        persist_input_artifact_registry_records(
                            input_artifacts,
                            request_payload=data if isinstance(data, Mapping) else {},
                        )

            if temp_path and file_name:
                if file_kind == "image":
                    image_b64 = to_base64(temp_path)
                elif file_kind == "text":
                    (
                        text_from_file,
                        text_from_file_truncated,
                        text_from_file_inline_bytes,
                        text_from_file_total_bytes,
                    ) = read_text_file(temp_path)
                elif file_kind == "pdf":
                    file_sha256 = hash_file_sha256(temp_path)
                    file_size_mb = round(temp_path.stat().st_size / (1024 * 1024), 2)
                    logging.info(
                        "PDF upload received: file=%s size_mb=%s capability=%s",
                        file_name,
                        file_size_mb,
                        capability,
                    )
                    if reuse_cached_pdf:
                        cached_entry = find_cached_pdf_insight(
                            file_sha256=file_sha256,
                            model_name=model_name,
                            backend=backend,
                            capability=capability,
                            prompt=user_prompt,
                        )
                        if cached_entry:
                            return jsonify(
                                {
                                    "instance_id": instance_id,
                                    "capability": capability,
                                    "mode": cached_entry.get("mode") or "vision_analysis_pdf_cached",
                                    "content": cached_entry.get("content") or "",
                                    "warnings": cached_entry.get("warnings") or [],
                                    "pdf_source": cached_entry.get("pdf_source"),
                                    "pdf_total_pages": cached_entry.get("pdf_total_pages"),
                                    "pdf_processed_pages": cached_entry.get("pdf_processed_pages"),
                                    "saved_text_path": cached_entry.get("artifact_path"),
                                    "cached": True,
                                    "cache_id": cached_entry.get("id"),
                                    "input_artifacts": input_artifacts,
                                }
                            ), 200
                    pdf_max_pages_raw = data.get("pdf_max_pages")
                    if str(pdf_max_pages_raw or "").strip():
                        pdf_max_pages = parse_int_with_bounds(
                            pdf_max_pages_raw,
                            default=500,
                            minimum=1,
                            maximum=500,
                        )
                    else:
                        pdf_max_pages = None
                    pdf_default_dpi = 260 if capability == self.capability_vision_analysis else 180
                    pdf_render_dpi = parse_int_with_bounds(
                        data.get("pdf_dpi"),
                        default=pdf_default_dpi,
                        minimum=96,
                        maximum=600,
                    )
                    pdf_page_retry_dpi = parse_int_with_bounds(
                        pdf_page_retry_dpi_raw,
                        default=max(120, int(pdf_render_dpi * 0.65)),
                        minimum=96,
                        maximum=480,
                    )
                    if pdf_page_retry_dpi >= pdf_render_dpi:
                        pdf_page_retry_dpi = max(96, pdf_render_dpi - 40)
                    use_text_first = bool(pdf_prefer_text) or capability != self.capability_vision_analysis
                    if use_text_first:
                        text_from_file = extract_pdf_text_content(temp_path)
                        if text_from_file:
                            logging.info("PDF text layer extracted: chars=%s", len(text_from_file))
                    if not text_from_file:
                        pdf_page_images, pdf_total_pages, pdf_warnings = render_pdf_pages_to_base64(
                            temp_path,
                            max_pages=pdf_max_pages,
                            dpi=pdf_render_dpi,
                            max_image_side_px=pdf_max_image_side,
                        )
                        logging.info(
                            "PDF rendered for OCR: pages_rendered=%s total_pages=%s dpi=%s max_side=%s warnings=%s",
                            len(pdf_page_images),
                            pdf_total_pages,
                            pdf_render_dpi,
                            pdf_max_image_side,
                            len(pdf_warnings),
                        )
                    if not use_text_first and not pdf_page_images:
                        text_from_file = extract_pdf_text_content(temp_path)
                        if text_from_file:
                            logging.info("PDF text fallback extracted: chars=%s", len(text_from_file))
                    if not text_from_file and not pdf_page_images:
                        base_error = (
                            "PDF could not be analyzed. Install 'pypdf' for text-based PDFs. "
                            "Scanned PDFs can use optional 'PyMuPDF' after its separate "
                            "AGPL-3.0 or commercial upstream terms are reviewed."
                        )
                        if pdf_warnings:
                            base_error = f"{base_error} Notes: {' '.join(pdf_warnings)}"
                        log_pdf_infer_event(
                            instance_id=instance_id,
                            model_name=model_name,
                            backend=backend,
                            capability=capability,
                            prompt=user_prompt,
                            file_name=file_name,
                            file_sha256=file_sha256,
                            status="error",
                            error=base_error,
                            warnings=pdf_warnings,
                        )
                        return jsonify({"error": base_error}), 400

            record_instance_activity(
                instance_id,
                path=self.runtime_status_path_getter(),
                instance=runtime_status_instance,
                activity='busy',
            )

            infer_context = InferContext(
                instance_id=instance_id,
                backend=backend,
                capability=capability,
                model_name=model_name,
                port=port,
                prompt=prompt,
                user_prompt=user_prompt,
                infer_timeout_sec=infer_timeout_sec,
                pdf_page_timeout_sec=pdf_page_timeout_sec,
                pdf_max_image_side=pdf_max_image_side,
                pdf_synthesize=pdf_synthesize,
                task=task,
                language=language,
                voice=voice,
                instruct=instruct,
                response_format=response_format,
                speed=speed,
                pitch=pitch,
                lang_code=lang_code,
                tts_model_type=str(instance.get("tts_model_type") or "").strip() or None,
                tts_speakers=list(instance.get("tts_speakers") or []) if isinstance(instance.get("tts_speakers"), list) else [],
                tts_languages=list(instance.get("tts_languages") or []) if isinstance(instance.get("tts_languages"), list) else [],
                ocr_mode=ocr_mode,
                image_width=image_width,
                image_height=image_height,
                image_seed=image_seed,
                text_artifact_requests=text_artifact_requests,
                prompt_is_semantic_materializer_payload=bool(
                    semantic_materializer_prompt
                ),
            )
            infer_artifacts = InferArtifacts(
                temp_path=temp_path,
                file_kind=file_kind,
                file_name=file_name,
                file_sha256=file_sha256,
                image_b64=image_b64,
                text_from_file=text_from_file,
                text_from_file_truncated=text_from_file_truncated,
                text_from_file_inline_bytes=text_from_file_inline_bytes,
                text_from_file_total_bytes=text_from_file_total_bytes,
                pdf_page_images=pdf_page_images,
                pdf_warnings=pdf_warnings,
                pdf_total_pages=pdf_total_pages,
                pdf_render_dpi=pdf_render_dpi,
                pdf_page_retry_dpi=pdf_page_retry_dpi,
            )
            infer_ops = {
                "whisper_transcribe": whisper_transcribe,
                "mlx_audio_speech": (lambda port, _model_name, prompt, instruct=None, voice=None, response_format=None, speed=1.0, pitch=1.0, lang_code=None, max_tokens=None, temperature=None, top_p=None, top_k=None, repetition_penalty=None, timeout_sec=600: mlx_audio_speech(
                    port,
                    request_model_name,
                    prompt,
                    instruct=instruct,
                    voice=voice,
                    response_format=response_format,
                    speed=speed,
                    pitch=pitch,
                    lang_code=lang_code,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    timeout_sec=timeout_sec,
                )),
                "ollama_generate": ollama_generate,
                "extract_saved_image_path_from_generate_output": extract_saved_image_path_from_generate_output,
                "extract_image_data_url_from_generate_output": extract_image_data_url_from_generate_output,
                "extract_generate_seed": extract_generate_seed,
                "ollama_openai_image_generation": ollama_openai_image_generation,
                "persist_audio_bytes_locally": persist_audio_bytes_locally,
                "persist_image_data_url_locally": persist_image_data_url_locally,
                "extract_generate_content": extract_generate_content,
                "persist_text_artifact_locally": persist_text_artifact_locally,
                "persist_text_markdown_locally": persist_text_markdown_locally,
                "persist_transcript_text_locally": persist_transcript_text_locally,
                "log_pdf_infer_event": log_pdf_infer_event,
                "ocr_pdf_page_with_ollama": ocr_pdf_page_with_ollama,
                "render_single_pdf_page_to_base64": render_single_pdf_page_to_base64,
                "ocr_image_with_deepseek": ocr_image_with_deepseek,
                "is_generic_ocr_instruction_prompt": is_generic_ocr_instruction_prompt,
                "clean_ocr_output_text": clean_ocr_output_text,
                "looks_like_ocr_prompt_echo": looks_like_ocr_prompt_echo,
                "ollama_chat": ollama_chat,
                "openai_chat_completions": (lambda port, _model_name, messages, timeout_sec=600: openai_chat_completions(
                    backend,
                    port,
                    request_model_name,
                    messages,
                    timeout_sec=timeout_sec,
                )),
                "mlx_chat_completions": (lambda port, _model_name, messages, timeout_sec=600: mlx_chat_completions(
                    port,
                    request_model_name,
                    messages,
                    timeout_sec=timeout_sec,
                )),
                "request_timeout_error": request_timeout_error,
                "request_connection_error": request_connection_error,
                "request_exception_error": request_exception_error,
                "max_pdf_inline_response_chars": self._hook('max_pdf_inline_response_chars')(),
            }
            payload, status_code = dispatch_infer_request(
                infer_context,
                infer_artifacts,
                infer_ops,
            )
            if status_code < 400:
                payload = dict(payload or {})
                if reference_artifacts:
                    payload['reference_artifacts'] = [dict(item) for item in reference_artifacts if isinstance(item, dict)]
            if input_artifacts:
                payload = dict(payload or {})
                payload['input_artifacts'] = [dict(item) for item in input_artifacts if isinstance(item, dict)]
            if status_code < 400:
                if parse_bool(data.get('suppress_image_state_enrichment') or data.get('suppress_generated_image_enrichment'), default=False):
                    payload = dict(payload or {})
                    payload['suppress_image_state_enrichment'] = True
                    suppression_reason = str(data.get('image_state_enrichment_suppression_reason') or '').strip()
                    if suppression_reason:
                        payload['image_state_enrichment_suppression_reason'] = suppression_reason
                payload = enrich_generated_image_payload(payload)
                provenance_record = persist_generated_image_provenance_for_infer_result(
                    payload,
                    request_payload=data if isinstance(data, dict) else None,
                    instance_id=instance_id,
                    model_name=model_name,
                    backend=backend,
                    capability=capability,
                    user_prompt=user_prompt,
                    prompt=prompt,
                    raw_file_path=raw_file_path,
                    input_artifacts=input_artifacts,
                    reference_artifacts=reference_artifacts,
                    image_width=image_width,
                    image_height=image_height,
                    image_seed=image_seed,
                )
                if isinstance(provenance_record, dict):
                    provenance_id = str(provenance_record.get('provenance_id') or '').strip()
                    if provenance_id:
                        payload['provenance_id'] = provenance_id
                    output_artifact = provenance_record.get('output')
                    if isinstance(output_artifact, dict):
                        artifact_id = str(output_artifact.get('artifact_id') or '').strip()
                        artifact_ref = str(output_artifact.get('artifact_ref') or '').strip()
                        if artifact_id:
                            payload['artifact_id'] = artifact_id
                        if artifact_ref:
                            payload['artifact_ref'] = artifact_ref
                    derived_from = provenance_record.get('derived_from')
                    if isinstance(derived_from, list) and derived_from:
                        payload['derived_from'] = list(derived_from)
            if status_code < 400:
                previous_status, current_status = record_instance_success(
                    instance_id,
                    path=self.runtime_status_path_getter(),
                    instance=runtime_status_instance,
                )
            elif 400 <= status_code < 500:
                previous_status, current_status = record_instance_activity(
                    instance_id,
                    path=self.runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    activity='idle',
                )
            else:
                previous_status, current_status = record_instance_failure(
                    instance_id,
                    path=self.runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    message=payload.get("error") or payload.get("mode") or "Infer request failed.",
                )
            log_runtime_status_transition(previous_status, current_status)
            log_unified_event(
                category="infer",
                action="request",
                status="ok" if status_code < 400 else "failed",
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                mode=payload.get("mode"),
                file_kind=file_kind,
                file_name=file_name or raw_file_path,
                http_status=status_code,
                warnings=payload.get("warnings"),
                message=payload.get("error") or payload.get("mode") or "Infer request completed.",
            )
            return jsonify(payload), status_code
        except request_timeout_error:
            previous_status, current_status = record_instance_failure(
                instance_id,
                path=self.runtime_status_path_getter(),
                instance=runtime_status_instance,
                message="Infer request timed out.",
            )
            log_runtime_status_transition(previous_status, current_status)
            log_unified_event(
                category="infer",
                action="request",
                status="failed",
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                file_kind=file_kind,
                file_name=file_name or raw_file_path,
                message="Infer request timed out.",
            )
            return jsonify(
                {
                    "error": (
                        "Timed out while waiting for the model request. The model process may still be running locally. "
                        "Use smaller PDF chunks or increase infer_timeout_sec/pdf_page_timeout_sec."
                    )
                }
            ), 504
        except request_connection_error as exc:
            previous_status, current_status = record_instance_failure(
                instance_id,
                path=self.runtime_status_path_getter(),
                instance=runtime_status_instance,
                message=f"Infer connection lost: {exc}",
            )
            log_runtime_status_transition(previous_status, current_status)
            log_unified_event(
                category="infer",
                action="request",
                status="failed",
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                file_kind=file_kind,
                file_name=file_name or raw_file_path,
                message=f"Infer connection lost: {exc}",
            )
            return jsonify(
                {
                    "error": (
                        "The connection to the model instance was interrupted "
                        f"({exc}). Restart the instance and send the request again."
                    )
                }
            ), 503
        except request_exception_error as exc:
            details = str(exc)
            if getattr(exc, "response", None) is not None:
                try:
                    payload = exc.response.json()
                    details = payload.get("error") or payload.get("message") or details
                except Exception:  # noqa: BLE001
                    details = getattr(exc.response, "text", details)[:300]
            previous_status, current_status = record_instance_failure(
                instance_id,
                path=self.runtime_status_path_getter(),
                instance=runtime_status_instance,
                message=f"Infer request failed: {details}",
            )
            log_runtime_status_transition(previous_status, current_status)
            log_unified_event(
                category="infer",
                action="request",
                status="failed",
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                file_kind=file_kind,
                file_name=file_name or raw_file_path,
                message=f"Infer request failed: {details}",
            )
            return jsonify({"error": f"Model request failed: {details}"}), 502
        except Exception as exc:  # noqa: BLE001
            previous_status, current_status = record_instance_failure(
                instance_id,
                path=self.runtime_status_path_getter(),
                instance=runtime_status_instance,
                message=str(exc),
            )
            log_runtime_status_transition(previous_status, current_status)
            log_unified_event(
                category="infer",
                action="request",
                status="failed",
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                file_kind=file_kind,
                file_name=file_name or raw_file_path,
                message=str(exc),
            )
            logging.exception("Error in /api/infer for %s: %s", instance_id, exc)
            return jsonify({"error": str(exc)}), 500
        finally:
            release_infer_slot(infer_slot_key)
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
