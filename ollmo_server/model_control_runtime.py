"""Runtime/model control owners for Ollmo."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ollmo_core.runtime_liveness import (
    runtime_instance_is_selectable,
    runtime_instance_score,
)
from ollmo_core.start_policy import (
    StartSourcePolicyError,
    attach_start_audit,
    build_start_audit,
    validate_start_source,
)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or '').strip().lower()
    return token in {'1', 'true', 'yes', 'on', 'force', 'forced'}


def _model_identity(value: Any) -> str:
    token = str(value or '').strip().lower()
    if token.endswith(':latest'):
        token = token[:-7]
    return token


def _existing_instance_is_usable(instance: dict[str, Any], *, capability: Any = None) -> bool:
    return runtime_instance_is_selectable(instance, capability=capability)


@dataclass
class ModelControlRuntimeOwner:
    hooks: dict[str, Any]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def build_backend_fabric_response(
        self,
        *,
        include_catalog: bool,
        refresh_runtime_status: bool = False,
    ) -> tuple[dict[str, Any], int]:
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        load_running_instances = self._hook('load_running_instances')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        list_available_models = self._hook('list_available_models')
        build_backend_fabric_snapshot = self._hook('build_backend_fabric_snapshot')

        instances = merge_instances_with_runtime_status(
            load_running_instances(),
            path=runtime_status_path_getter(),
            refresh=refresh_runtime_status,
        )
        available_models = list_available_models(include_limits=False) if include_catalog else None
        return build_backend_fabric_snapshot(
            instances=instances,
            available_models=available_models,
        ), 200

    def build_available_models_response(self, *, include_limits: bool) -> tuple[dict[str, Any], int]:
        list_available_models = self._hook('list_available_models')
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')
        infer_capability = self._hook('infer_capability')
        build_registry_metadata = self._hook('build_registry_metadata')
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        load_running_instances = self._hook('load_running_instances')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        build_backend_fabric_snapshot = self._hook('build_backend_fabric_snapshot')

        try:
            aggregated = list_available_models(include_limits=include_limits)
            normalized_models = []
            for entry in aggregated:
                if not isinstance(entry, dict):
                    normalized_models.append(entry)
                    continue
                normalized = dict(entry)
                model_name = str(
                    normalized.get('model')
                    or normalized.get('modelName')
                    or normalized.get('name')
                    or ''
                ).strip()
                backend = normalize_backend(normalized.get('backend'))
                capability = normalize_capability(normalized.get('capability')) or infer_capability(
                    model_name,
                    backend,
                )
                normalized.setdefault('model', model_name)
                normalized.setdefault('modelName', model_name)
                normalized['backend'] = backend
                normalized['capability'] = capability
                normalized.update(
                    build_registry_metadata(
                        model_name,
                        backend,
                        capability,
                        metadata=normalized,
                    )
                )
                normalized_models.append(normalized)
            instances = merge_instances_with_runtime_status(
                load_running_instances(),
                path=runtime_status_path_getter(),
                refresh=False,
            )
            backend_fabric = build_backend_fabric_snapshot(
                instances=instances,
                available_models=normalized_models,
            )
            return {'models': normalized_models, 'backend_fabric': backend_fabric}, 200
        except Exception as exc:  # noqa: BLE001
            logging.exception('Error while fetching available models: %s', exc)
            return {'error': str(exc)}, 500

    def pull_model_response(self, data: Any) -> tuple[dict[str, Any], int]:
        normalize_backend = self._hook('normalize_backend')
        pull_model = self._hook('pull_model')
        log_unified_event = self._hook('log_unified_event')

        payload = data if isinstance(data, dict) else {}
        model_name = payload.get('model')
        if not model_name:
            return {'error': "Parameter 'model' is required."}, 400
        backend = normalize_backend(payload.get('backend') or 'ollama')
        logging.info('Pull model via API: %s (backend=%s)', model_name, backend)
        success, message = pull_model(model_name, backend)
        status = 200 if success else 500
        log_unified_event(
            category='runtime',
            action='pull_model',
            status='ok' if success else 'failed',
            model=model_name,
            backend=backend,
            message=message,
        )
        return {'success': success, 'message': message}, status

    def remove_model_response(self, data: Any) -> tuple[dict[str, Any], int]:
        normalize_backend = self._hook('normalize_backend')
        remove_model = self._hook('remove_model')
        log_unified_event = self._hook('log_unified_event')

        payload = data if isinstance(data, dict) else {}
        model_name = payload.get('model')
        if not model_name:
            return {'error': "Parameter 'model' is required."}, 400
        backend = normalize_backend(payload.get('backend') or 'ollama')
        remove_kwargs: dict[str, Any] = {}
        for source_key, target_key in (
            ('model_source', 'model_source'),
            ('modelSource', 'model_source'),
            ('model_path', 'model_path'),
            ('modelPath', 'model_path'),
            ('hf_repo', 'hf_repo'),
            ('hfRepo', 'hf_repo'),
            ('hf_file', 'hf_file'),
            ('hfFile', 'hf_file'),
        ):
            value = str(payload.get(source_key) or '').strip()
            if value and target_key not in remove_kwargs:
                remove_kwargs[target_key] = value
        logging.info('Remove model via API: %s (backend=%s)', model_name, backend)
        success, message = remove_model(model_name, backend, **remove_kwargs)
        status = 200 if success else 500
        log_unified_event(
            category='runtime',
            action='remove_model',
            status='ok' if success else 'failed',
            model=model_name,
            backend=backend,
            message=message,
        )
        return {'success': success, 'message': message}, status

    def start_model_response(
        self,
        data: Any,
        *,
        default_start_source: Any = None,
    ) -> tuple[dict[str, Any], int]:
        canonical_model_name = self._hook('canonical_model_name')
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')
        start_model_request_error = self._hook('start_model_request_error')
        start_instance = self._hook('start_instance')
        infer_capability = self._hook('infer_capability')
        record_instance_started = self._hook('record_instance_started')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        log_unified_event = self._hook('log_unified_event')
        log_runtime_status_transition = self._hook('log_runtime_status_transition')
        load_running_instances = self._hook('load_running_instances')
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        instance_supports_capability = self._hook('instance_supports_capability')

        model_name = 'unknown'
        payload = data if isinstance(data, dict) else {}
        try:
            start_source = validate_start_source(
                payload.get('start_source') or payload.get('startSource'),
                default=default_start_source,
                context='start_model_response',
            )
        except StartSourcePolicyError as exc:
            policy_violation = dict(exc.policy_violation)
            logging.warning('Model start policy violation: %s', policy_violation)
            log_unified_event(
                category='runtime',
                action='start_model',
                status='policy_violation',
                start_source=policy_violation.get('start_source'),
                policy_violation=policy_violation,
                message=str(exc),
            )
            return {
                'error': str(exc),
                'start_source': policy_violation.get('start_source'),
                'policy_violation': policy_violation,
            }, 400
        start_audit = build_start_audit(
            start_source,
            context='start_model_response',
            extra={'endpoint': '/api/start_model'},
        )
        try:
            model_name = canonical_model_name(payload)
            raw_backend = str(payload.get('backend') or 'ollama').strip()
            backend = normalize_backend(raw_backend)
            if backend not in {'ollama', 'mlx', 'llama_cpp'}:
                raise start_model_request_error(
                    f"Unknown backend type '{raw_backend}'.",
                    status_code=400,
                )

            requested_capability = normalize_capability(payload.get('capability')) or None
            preferred_port = payload.get('preferred_port')
            if preferred_port is not None:
                try:
                    preferred_port = int(preferred_port)
                except (TypeError, ValueError) as exc:
                    raise start_model_request_error(
                        "Invalid value for 'preferred_port'.",
                        status_code=400,
                    ) from exc

            logging.info(
                'Start model via API: %s (backend=%s, requested_capability=%s)',
                model_name,
                backend,
                requested_capability,
            )
            force_start = _coerce_bool(payload.get('force_start') or payload.get('forceStart'))
            capability_for_match = requested_capability or infer_capability(model_name, backend)
            if not force_start:
                instances = merge_instances_with_runtime_status(
                    load_running_instances(),
                    path=runtime_status_path_getter(),
                    refresh=True,
                )
                matching_instances = [
                    entry
                    for entry in instances
                    if isinstance(entry, dict)
                    and normalize_backend(entry.get('backend')) == backend
                    and _model_identity(entry.get('model')) == _model_identity(model_name)
                    and (
                        not capability_for_match
                        or instance_supports_capability(entry, capability_for_match)
                    )
                    and _existing_instance_is_usable(entry, capability=capability_for_match)
                ]
                if matching_instances:
                    instance = sorted(
                        matching_instances,
                        key=lambda entry: runtime_instance_score(
                            entry,
                            capability=capability_for_match,
                        ),
                        reverse=True,
                    )[0]
                    capability = normalize_capability(instance.get('capability')) or capability_for_match
                    log_unified_event(
                        category='runtime',
                        action='start_model',
                        status='ok',
                        model=model_name,
                        backend=backend,
                        capability=capability,
                        instance_id=instance.get('instance_id'),
                        port=instance.get('port'),
                        start_source=start_source,
                        start_audit={**start_audit, 'status': 'reused'},
                        message=f'Reused existing {model_name}',
                    )
                    return {
                        'instance': instance,
                        'status': 'reused',
                        'reused': True,
                        'start_source': start_source,
                        'start_audit': {**start_audit, 'status': 'reused'},
                    }, 200
            instance = start_instance(
                model_name,
                backend,
                requested_capability,
                model_path=payload.get('model_path'),
                preferred_port=preferred_port,
                hf_file=payload.get('hf_file'),
                launch_defaults=payload.get('launch_defaults')
                if isinstance(payload.get('launch_defaults'), dict)
                else None,
                start_source=start_source,
            )
            instance = attach_start_audit(
                instance,
                start_source=start_source,
                context='start_model_response',
                extra={'endpoint': '/api/start_model'},
            )
            capability = normalize_capability(instance.get('capability')) or requested_capability or infer_capability(
                model_name,
                backend,
            )
            previous_status, current_status = record_instance_started(
                instance,
                path=runtime_status_path_getter(),
            )
            log_unified_event(
                category='runtime',
                action='start_model',
                status='ok',
                model=model_name,
                backend=backend,
                capability=capability,
                instance_id=instance.get('instance_id'),
                port=instance.get('port'),
                start_source=start_source,
                start_audit=instance.get('start_audit') or start_audit,
                message=f'Started {model_name}',
            )
            log_runtime_status_transition(previous_status, current_status)
            return {
                'instance': instance,
                'status': 'started',
                'start_source': start_source,
                'start_audit': instance.get('start_audit') or start_audit,
            }, 200
        except Exception as exc:  # noqa: BLE001
            status_code = int(getattr(exc, 'status_code', 500) or 500)
            details = getattr(exc, 'details', None)
            if status_code == 500:
                logging.exception('Error while starting model %s: %s', model_name, exc)
            else:
                logging.warning('Start model request failed for %s: %s', model_name, exc)
            log_unified_event(
                category='runtime',
                action='start_model',
                status='failed',
                model=model_name,
                start_source=start_source,
                start_audit=start_audit,
                message=str(exc),
            )
            response_payload = {'error': str(exc)}
            if details:
                response_payload['details'] = details
            return response_payload, status_code

    def stop_model_response(self, data: Any) -> tuple[dict[str, Any], int]:
        normalize_external_identifier = self._hook('normalize_external_identifier')
        stop_instance = self._hook('stop_instance')
        remove_instance_status = self._hook('remove_instance_status')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        cleanup_runtime_hygiene = self._hook('cleanup_runtime_hygiene')
        config_file_name = self._hook('config_file_name')
        flask_log_path = self._hook('flask_log_path')
        active_global_log_paths = self._hook('active_global_log_paths')
        log_unified_event = self._hook('log_unified_event')
        log_runtime_status_transition = self._hook('log_runtime_status_transition')
        build_stop_payload = self._hook('build_stop_payload')

        payload = data if isinstance(data, dict) else {}
        try:
            instance_id = normalize_external_identifier(
                payload.get('instance_id'),
                field_name='instance_id',
            )
        except ValueError as exc:
            return {'error': str(exc)}, 400

        logging.info('Stop instance via API: %s', instance_id)
        try:
            result, instance = stop_instance(instance_id)
            previous_status = None
            if result.state in {'stopped', 'stopping'}:
                previous_status = remove_instance_status(
                    instance_id,
                    path=runtime_status_path_getter(),
                )
                try:
                    cleanup_runtime_hygiene(
                        registry_path=config_file_name(),
                        status_path=runtime_status_path_getter(),
                        log_dir=flask_log_path().parent,
                        sync_external=False,
                        active_global_log_paths=active_global_log_paths(include_webserver=True),
                    )
                except Exception as cleanup_exc:  # noqa: BLE001
                    logging.warning(
                        'Runtime hygiene after stopping %s failed: %s',
                        instance_id,
                        cleanup_exc,
                    )
            log_unified_event(
                category='runtime',
                action='stop_model',
                status='ok' if result.state in {'stopped', 'stopping'} else 'failed',
                instance_id=instance_id,
                message=result.message,
                details=result.details,
            )
            if previous_status:
                log_runtime_status_transition(
                    previous_status,
                    {
                        **previous_status,
                        'readiness': 'stopped',
                    },
                )
            response_payload, status_code = build_stop_payload(result, instance)
            if status_code >= 400:
                logging.warning(
                    'Stop for %s ended with status %s: %s',
                    instance_id,
                    result.state,
                    result.message,
                )
            else:
                logging.info('Stop for %s status %s', instance_id, result.state)
            return response_payload, status_code
        except Exception as exc:  # noqa: BLE001
            status_code = int(getattr(exc, 'status_code', 500) or 500)
            details = getattr(exc, 'details', None)
            if status_code == 500:
                logging.exception('Error while stopping instance %s: %s', instance_id, exc)
            response_payload = {'error': str(exc)}
            if details:
                response_payload['details'] = details
            return response_payload, status_code
