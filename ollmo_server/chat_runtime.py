"""Chat execution and streaming owners for Ollmo."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from flask import Response, jsonify, stream_with_context

from ollmo_server.response_semantics_runtime import (
    attach_phase_output_acceptance,
    classify_phase_output_text,
    control_json_envelope_suspected,
    phase_output_is_graph_preparation,
    phase_output_repair_notice,
    phase_output_repair_system_message,
    request_explicitly_allows_control_diagnostics,
)


@dataclass
class ChatRuntimeOwner:
    hooks: dict[str, Any]
    capability_embedding: str
    capability_image_generation: str
    capability_speech_to_text: str
    capability_text_to_speech: str
    request_timeout_error: type[Exception]
    request_connection_error: type[Exception]
    request_exception_error: type[Exception]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def stream_chat_backend_as_responses(
        self,
        *,
        instance_id: str,
        target_port: int,
        model_name: str,
        backend: str,
        capability: str,
        messages: list[dict],
        request_model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        route_payload: Optional[dict[str, Any]] = None,
        response_id: Optional[str] = None,
        request_payload: Optional[dict[str, Any]] = None,
        artifact_prompt: Optional[str] = None,
    ):
        normalize_backend = self._hook('normalize_backend')
        build_runtime_status_stub = self._hook('build_runtime_status_stub')
        build_canonical_response_payload = self._hook('build_canonical_response_payload')
        attach_response_semantic_phase_payload = self._hook('attach_response_semantic_phase_payload')
        normalize_response_lookup_id = self._hook('normalize_response_lookup_id')
        register_response_lookup = self._hook('register_response_lookup')
        touch_response_lookup = self._hook('touch_response_lookup')
        register_response_stream = self._hook('register_response_stream')
        append_response_stream_events = self._hook('append_response_stream_events')
        wait_for_response_stream_events = self._hook('wait_for_response_stream_events')
        close_response_stream = self._hook('close_response_stream')
        attach_pre_freeze_closure_review = self._hook('attach_pre_freeze_closure_review')
        finalize_response_frame_payload = self._hook('finalize_response_frame_payload')
        schedule_response_late_fill = self._hook('schedule_response_late_fill')
        late_fill_stream_waits_for_terminal = self.hooks.get(
            'late_fill_stream_waits_for_terminal',
            lambda: True,
        )
        apply_direct_artifact_materialization_closure = self.hooks.get(
            'apply_direct_artifact_materialization_closure'
        )
        schedule_post_response_substrate_hygiene = self.hooks.get('schedule_post_response_substrate_hygiene')
        log_unified_event = self._hook('log_unified_event')
        record_instance_activity = self._hook('record_instance_activity')
        record_instance_success = self._hook('record_instance_success')
        record_instance_failure = self._hook('record_instance_failure')
        log_runtime_status_transition = self._hook('log_runtime_status_transition')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        open_openai_chat_stream = self._hook('open_openai_chat_stream')
        open_ollama_chat_stream = self._hook('open_ollama_chat_stream')
        iter_openai_stream_deltas = self._hook('iter_openai_stream_deltas')
        iter_ollama_stream_deltas = self._hook('iter_ollama_stream_deltas')
        execute_chat_backend_request = self._hook('execute_chat_backend_request')
        persist_generated_text_artifact_if_requested = self._hook('persist_generated_text_artifact_if_requested')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')
        request_exception_details = self._hook('request_exception_details')
        chat_timeout_seconds = self._hook('chat_timeout_seconds')

        normalized_backend = normalize_backend(backend)
        chat_timeout_sec = chat_timeout_seconds(model_name, normalized_backend, capability)
        runtime_status_instance = build_runtime_status_stub(
            instance_id=instance_id,
            model=model_name,
            backend=normalized_backend,
            capability=capability,
            port=target_port,
        )
        response_id = normalize_response_lookup_id(response_id)
        message_id = f'msg_{uuid.uuid4().hex}'
        preparation_stream = phase_output_is_graph_preparation(
            route_payload=route_payload,
            request_payload=request_payload,
            capability=capability,
        )
        explicit_control_diagnostics = request_explicitly_allows_control_diagnostics(
            request_payload=request_payload,
            route_payload=route_payload,
            capability=capability,
        )
        buffer_phase_output = preparation_stream and not explicit_control_diagnostics

        def format_sse_event(event_name: str, payload: dict[str, Any]) -> str:
            return (
                f"event: {event_name}\n"
                f"data: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"
            )

        created_payload = build_canonical_response_payload(
            instance_id=instance_id,
            model_name=model_name,
            backend=normalized_backend,
            capability=capability,
            mode='chat',
            output_text='',
            source_payload=attach_response_semantic_phase_payload(
                {},
                output_text='',
                route_payload=route_payload,
                request_payload=request_payload,
                capability=capability,
            ),
            route_payload=route_payload,
            response_id=response_id,
            message_id=message_id,
        )
        register_response_lookup(
            response_id=response_id,
            message_id=message_id,
            instance_id=instance_id,
            model_name=model_name,
            backend=normalized_backend,
            capability=capability,
            mode='chat',
            route_payload=route_payload,
        )

        record_instance_activity(
            instance_id,
            path=runtime_status_path_getter(),
            instance=runtime_status_instance,
            activity='busy',
        )

        upstream_response = None
        effective_port = int(target_port)
        started_at = time.time()

        try:
            if normalized_backend in {'mlx', 'llama_cpp'}:
                upstream_response = open_openai_chat_stream(
                    backend=normalized_backend,
                    target_port=target_port,
                    request_model_override=request_model_override,
                    model_name=model_name,
                    messages=messages,
                    timeout_sec=chat_timeout_sec,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                delta_iter = iter_openai_stream_deltas(upstream_response)
            else:
                upstream_response, effective_port = open_ollama_chat_stream(
                    target_port=target_port,
                    model_name=model_name,
                    messages=messages,
                    timeout_sec=chat_timeout_sec,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                delta_iter = iter_ollama_stream_deltas(upstream_response)
        except self.request_timeout_error:
            touch_response_lookup(
                response_id,
                status='failed',
                error_message=f'Timeout Port {target_port}',
            )
            if upstream_response is not None:
                try:
                    upstream_response.close()
                except Exception:  # noqa: BLE001
                    pass
            previous_status, current_status = record_instance_failure(
                instance_id,
                path=runtime_status_path_getter(),
                instance=runtime_status_instance,
                message='Chat stream timed out.',
            )
            log_runtime_status_transition(previous_status, current_status)
            return jsonify({'error': f'Timeout Port {target_port}'}), 504
        except self.request_exception_error as exc:
            details = request_exception_details(exc)
            touch_response_lookup(
                response_id,
                status='failed',
                error_message=f'Request to port {target_port} failed: {details}',
            )
            if upstream_response is not None:
                try:
                    upstream_response.close()
                except Exception:  # noqa: BLE001
                    pass
            previous_status, current_status = record_instance_failure(
                instance_id,
                path=runtime_status_path_getter(),
                instance=runtime_status_instance,
                message=f'Chat stream failed: {details}',
            )
            log_runtime_status_transition(previous_status, current_status)
            return jsonify({'error': f'Request to port {target_port} failed: {details}'}), 500
        except Exception:
            touch_response_lookup(
                response_id,
                status='failed',
                error_message='Chat stream failed before streaming could begin.',
            )
            if upstream_response is not None:
                try:
                    upstream_response.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

        register_response_stream(response_id)
        in_progress_payload = {**created_payload, 'status': 'in_progress', 'output': []}
        target_payload = {
            'instance_id': instance_id,
            'model': model_name,
            'backend': normalized_backend,
            'capability': capability,
            'port': effective_port,
        }
        route_event_payload = {}
        if isinstance(route_payload, dict):
            route_event_payload = {
                key: route_payload.get(key)
                for key in (
                    'route_source',
                    'route_reason',
                    'route_router_instance_id',
                    'route_router_model',
                    'route_artifact_ref',
                    'route_artifact_path',
                    'context_mode',
                    'context_reason',
                )
                if route_payload.get(key) not in (None, '', [], {})
            }
        append_response_stream_events(
            response_id,
            [
                f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': in_progress_payload}, ensure_ascii=False)}\n\n",
                f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': in_progress_payload}, ensure_ascii=False)}\n\n",
                format_sse_event(
                    'response.route.resolved',
                    {
                        'type': 'response.route.resolved',
                        'response_id': response_id,
                        'route': route_event_payload,
                        'target': target_payload,
                    },
                ),
                format_sse_event(
                    'response.backend.started',
                    {
                        'type': 'response.backend.started',
                        'response_id': response_id,
                        'target': target_payload,
                    },
                ),
                f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': {'id': message_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []}}, ensure_ascii=False)}\n\n",
                f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}}, ensure_ascii=False)}\n\n",
            ],
        )

        def stream_worker():
            accumulated = ''
            try:
                for delta in delta_iter:
                    if not delta:
                        continue
                    accumulated += delta
                    if not buffer_phase_output:
                        touch_response_lookup(response_id, output_text=accumulated)
                        append_response_stream_events(
                            response_id,
                            [
                                f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'delta': delta}, ensure_ascii=False)}\n\n"
                            ],
                        )

                if not accumulated:
                    accumulated = execute_chat_backend_request(
                        target_port=effective_port,
                        model_name=model_name,
                        backend=normalized_backend,
                        capability=capability,
                        messages=messages,
                        request_model_override=request_model_override,
                    )

                phase_acceptance_attempts: list[dict[str, Any]] = []
                if buffer_phase_output or (
                    control_json_envelope_suspected(accumulated)
                    and not explicit_control_diagnostics
                ):
                    phase_acceptance_attempts.append(
                        classify_phase_output_text(accumulated)
                    )
                    if (
                        preparation_stream
                        and phase_acceptance_attempts[-1].get('status') == 'repair_required'
                    ):
                        try:
                            retried_text = execute_chat_backend_request(
                                target_port=effective_port,
                                model_name=model_name,
                                backend=normalized_backend,
                                capability=capability,
                                messages=[*messages, phase_output_repair_system_message()],
                                request_model_override=request_model_override,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                            )
                            phase_acceptance_attempts.append(
                                classify_phase_output_text(retried_text)
                            )
                        except Exception as exc:  # noqa: BLE001
                            phase_acceptance_attempts.append(
                                {
                                    'status': 'repair_required',
                                    'accepted_text': '',
                                    'source_sha256': '',
                                    'source_bytes': 0,
                                    'reason': (
                                        'bounded same-phase repair call failed before producing '
                                        f'content ({type(exc).__name__})'
                                    ),
                                }
                            )
                    accepted_text = str(
                        phase_acceptance_attempts[-1].get('accepted_text') or ''
                    ).strip()
                    accumulated = (
                        phase_output_repair_notice()
                        if phase_acceptance_attempts[-1].get('status') == 'repair_required'
                        else accepted_text
                    )
                phase_output_repair_required = bool(
                    phase_acceptance_attempts
                    and phase_acceptance_attempts[-1].get('status') == 'repair_required'
                )

                if buffer_phase_output:
                    touch_response_lookup(response_id, output_text=accumulated)
                    if accumulated:
                        append_response_stream_events(
                            response_id,
                            [
                                f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'delta': accumulated}, ensure_ascii=False)}\n\n"
                            ],
                        )

                request_info = request_payload if isinstance(request_payload, dict) else {}
                current_artifact_prompt = str(artifact_prompt or '').strip()
                if not current_artifact_prompt:
                    current_artifact_prompt = str(
                        extract_responses_current_turn_prompt(request_info)
                        or request_info.get('prompt')
                        or ''
                    ).strip()
                text_artifact_payload = (
                    {}
                    if phase_output_repair_required
                    else persist_generated_text_artifact_if_requested(
                        accumulated,
                        prompt=current_artifact_prompt,
                        model_name=model_name,
                        mode='responses_stream_chat_text_artifact',
                        request_payload=request_info,
                    )
                )
                final_payload = build_canonical_response_payload(
                    instance_id=instance_id,
                    model_name=model_name,
                    backend=normalized_backend,
                    capability=capability,
                    mode='chat',
                    output_text=accumulated,
                    source_payload=attach_response_semantic_phase_payload(
                        text_artifact_payload,
                        output_text=accumulated,
                        route_payload=route_payload,
                        request_payload=request_payload,
                        capability=capability,
                    ),
                    route_payload=route_payload,
                    response_id=response_id,
                    message_id=message_id,
                )
                if phase_acceptance_attempts:
                    final_payload = attach_phase_output_acceptance(
                        final_payload,
                        phase_acceptance_attempts,
                    )
                final_payload, artifact_completion_gap = attach_pre_freeze_closure_review(
                    final_payload,
                    output_text=accumulated,
                    route_payload=route_payload,
                    request_payload=request_payload,
                )
                _direct_closure_status = 'completed'
                if not artifact_completion_gap and callable(apply_direct_artifact_materialization_closure):
                    final_payload, _direct_closure_status = apply_direct_artifact_materialization_closure(
                        final_payload,
                        request_payload=dict(request_payload or {}),
                        route_payload=route_payload,
                        artifact_gap=artifact_completion_gap,
                        terminal_status='completed',
                    )
                final_payload = finalize_response_frame_payload(
                    final_payload,
                    request_payload=request_payload,
                    persist=True,
                )
                touch_response_lookup(
                    response_id,
                    status='completed',
                    output_text=accumulated,
                    response_payload=final_payload,
                    error_message='',
                )
                final_item = final_payload['output'][0]
                final_part = final_item['content'][0]
                wait_for_late_fill_terminal = bool(artifact_completion_gap) and bool(
                    late_fill_stream_waits_for_terminal()
                )
                completion_events = [
                    f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': accumulated}, ensure_ascii=False)}\n\n",
                    f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': final_part}, ensure_ascii=False)}\n\n",
                    f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': final_item}, ensure_ascii=False)}\n\n",
                ]
                if not wait_for_late_fill_terminal:
                    completion_events.append(
                        f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': final_payload}, ensure_ascii=False)}\n\n"
                    )
                log_unified_event(
                    category='chat',
                    action='request',
                    status='ok',
                    instance_id=instance_id,
                    model=model_name,
                    backend=normalized_backend,
                    capability=capability,
                    latency_sec=round(time.time() - started_at, 3),
                    message='Chat response streamed.',
                )
                previous_status, current_status = record_instance_success(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance={**runtime_status_instance, 'port': effective_port},
                    latency_sec=round(time.time() - started_at, 3),
                )
                log_runtime_status_transition(previous_status, current_status)
                if (
                    not artifact_completion_gap
                    and callable(schedule_post_response_substrate_hygiene)
                    and str(final_payload.get('lifecycle_state') or final_payload.get('status') or '').strip().lower()
                    == 'completed'
                ):
                    try:
                        schedule_post_response_substrate_hygiene(
                            final_payload,
                            route_payload=route_payload,
                            reason='stream_chat_completed',
                        )
                    except Exception:  # noqa: BLE001
                        logging.exception('Could not schedule post-response substrate hygiene.')
                append_response_stream_events(
                    response_id,
                    completion_events,
                    done=not wait_for_late_fill_terminal,
                )
                if artifact_completion_gap:
                    schedule_response_late_fill(
                        response_payload=final_payload,
                        request_payload=dict(request_payload or {}),
                        assistant_message=accumulated,
                        artifact_gap=artifact_completion_gap,
                        source_route_payload=route_payload,
                    )
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                touch_response_lookup(
                    response_id,
                    status='failed',
                    output_text=accumulated,
                    error_message=error_message,
                )
                log_unified_event(
                    category='chat',
                    action='request',
                    status='failed',
                    instance_id=instance_id,
                    model=model_name,
                    backend=normalized_backend,
                    capability=capability,
                    latency_sec=round(time.time() - started_at, 3),
                    message=error_message,
                )
                previous_status, current_status = record_instance_failure(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance={**runtime_status_instance, 'port': effective_port},
                    message=error_message,
                    latency_sec=round(time.time() - started_at, 3),
                )
                log_runtime_status_transition(previous_status, current_status)
                append_response_stream_events(
                    response_id,
                    [
                        f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'error': {'message': error_message}}, ensure_ascii=False)}\n\n"
                    ],
                    done=True,
                )

        threading.Thread(target=stream_worker, daemon=True).start()

        def generate():
            cursor = 0
            try:
                while True:
                    events, done = wait_for_response_stream_events(response_id, cursor)
                    if events:
                        for event in events:
                            yield event
                        cursor += len(events)
                    if done and not events:
                        break
            finally:
                close_response_stream(response_id)

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    def execute_chat_request(self, data: Any):
        normalize_capability = self._hook('normalize_capability')
        parse_float_with_bounds = self._hook('parse_float_with_bounds')
        parse_int_with_bounds = self._hook('parse_int_with_bounds')
        normalize_external_identifier = self._hook('normalize_external_identifier')
        load_running_instances = self._hook('load_running_instances')
        normalize_backend = self._hook('normalize_backend')
        infer_capability = self._hook('infer_capability')
        build_runtime_status_stub = self._hook('build_runtime_status_stub')
        is_port_listening = self._hook('is_port_listening')
        record_instance_activity = self._hook('record_instance_activity')
        record_instance_success = self._hook('record_instance_success')
        record_instance_failure = self._hook('record_instance_failure')
        log_runtime_status_transition = self._hook('log_runtime_status_transition')
        log_unified_event = self._hook('log_unified_event')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        execute_chat_backend_request = self._hook('execute_chat_backend_request')
        request_exception_details = self._hook('request_exception_details')

        logging.info('API call: /api/chat for instance: %s', data.get('instance_id'))

        target_port = data.get('port')
        model_name = data.get('model')
        messages = data.get('messages')
        instance_id = data.get('instance_id')
        backend = data.get('backend')
        capability = normalize_capability(data.get('capability'))
        request_model_override = data.get('request_model')
        try:
            temperature = parse_float_with_bounds(data.get('temperature'), default=0.7, minimum=0.0, maximum=2.0) if data.get('temperature') not in (None, '') else None
            top_p = parse_float_with_bounds(data.get('top_p') or data.get('topP'), default=0.9, minimum=0.0, maximum=1.0) if data.get('top_p') not in (None, '') or data.get('topP') not in (None, '') else None
            max_tokens = parse_int_with_bounds(
                data.get('max_tokens') or data.get('maxTokens'),
                default=1_000_000,
                minimum=1,
                maximum=1_000_000,
            ) if data.get('max_tokens') not in (None, '') or data.get('maxTokens') not in (None, '') else None
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        if str(instance_id or '').strip():
            try:
                instance_id = normalize_external_identifier(instance_id, field_name='instance_id')
            except ValueError as exc:
                return jsonify({'error': str(exc)}), 400

        instance_info = None
        if instance_id:
            instances = load_running_instances()
            instance_info = next((inst for inst in instances if inst.get('instance_id') == instance_id), None)
            if instance_info:
                target_port = target_port or instance_info.get('port')
                model_name = model_name or instance_info.get('model')
                backend = backend or instance_info.get('backend', 'ollama')
                capability = capability or normalize_capability(instance_info.get('capability'))
                request_model_override = request_model_override or instance_info.get('request_model')

        backend = normalize_backend(backend)
        capability = capability or infer_capability(model_name, backend)

        if backend not in ('ollama', 'mlx', 'llama_cpp'):
            logging.error('Unknown backend type: %s', backend)
            return jsonify({'error': f"Unknown backend type '{backend}'"}), 400

        if capability in {
            self.capability_embedding,
            self.capability_image_generation,
            self.capability_speech_to_text,
            self.capability_text_to_speech,
        }:
            logging.warning(
                '/api/chat is not supported for capability=%s (model=%s, backend=%s)',
                capability,
                model_name,
                backend,
            )
            return jsonify(
                {
                    'error': (
                        f"/api/chat is not supported for capability '{capability}'. "
                        'Use a capability-specific endpoint.'
                    )
                }
            ), 400

        if not all([target_port, model_name, messages]):
            logging.error('Missing data in chat request')
            return jsonify({'error': 'Missing data'}), 400

        runtime_status_instance = instance_info or build_runtime_status_stub(
            instance_id=instance_id,
            model=model_name,
            backend=backend,
            capability=capability,
            port=target_port,
        )

        if not is_port_listening(int(target_port)):
            logging.error("Target port %s for '%s' is not active.", target_port, instance_id)
            if instance_id:
                previous_status, current_status = record_instance_failure(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    message=f'Target port {target_port} is not active.',
                )
                log_runtime_status_transition(previous_status, current_status)
            return jsonify({'error': f'Target port {target_port} is not active.'}), 404

        started_at = time.time()
        if instance_id:
            record_instance_activity(
                instance_id,
                path=runtime_status_path_getter(),
                instance=runtime_status_instance,
                activity='busy',
            )
        try:
            assistant_message = execute_chat_backend_request(
                target_port=int(target_port),
                model_name=model_name,
                backend=backend,
                capability=capability,
                messages=messages,
                request_model_override=request_model_override,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            logging.info('Received response from backend %s (port %s).', backend, target_port)
            log_unified_event(
                category='chat',
                action='request',
                status='ok',
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                latency_sec=round(time.time() - started_at, 3),
                message='Chat response returned.',
            )
            if instance_id:
                previous_status, current_status = record_instance_success(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    latency_sec=round(time.time() - started_at, 3),
                )
                log_runtime_status_transition(previous_status, current_status)
            return jsonify({'role': 'assistant', 'content': assistant_message})

        except self.request_timeout_error:
            logging.error('Timeout Port %s', target_port)
            log_unified_event(
                category='chat',
                action='request',
                status='failed',
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                latency_sec=round(time.time() - started_at, 3),
                message=f'Timeout Port {target_port}',
            )
            if instance_id:
                previous_status, current_status = record_instance_failure(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    message=f'Timeout Port {target_port}',
                    latency_sec=round(time.time() - started_at, 3),
                )
                log_runtime_status_transition(previous_status, current_status)
            return jsonify({'error': f'Timeout Port {target_port}'}), 504
        except self.request_exception_error as exc:
            logging.error('Port %s error: %s', target_port, exc)
            error_details = request_exception_details(exc)
            log_unified_event(
                category='chat',
                action='request',
                status='failed',
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                latency_sec=round(time.time() - started_at, 3),
                message=error_details,
            )
            if instance_id:
                previous_status, current_status = record_instance_failure(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    message=error_details,
                    latency_sec=round(time.time() - started_at, 3),
                )
                log_runtime_status_transition(previous_status, current_status)
            return jsonify({'error': f'Request to port {target_port} failed: {error_details}'}), 500
        except (json.JSONDecodeError, ValueError) as exc:
            logging.error('JSON error on port %s: %s', target_port, exc)
            log_unified_event(
                category='chat',
                action='request',
                status='failed',
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=capability,
                latency_sec=round(time.time() - started_at, 3),
                message=str(exc),
            )
            if instance_id:
                previous_status, current_status = record_instance_failure(
                    instance_id,
                    path=runtime_status_path_getter(),
                    instance=runtime_status_instance,
                    message=str(exc),
                    latency_sec=round(time.time() - started_at, 3),
                )
                log_runtime_status_transition(previous_status, current_status)
            return jsonify({'error': f'JSON error on port {target_port}: {exc}'}), 500
