"""Post-response cleanup for volatile model substrate.

This module only targets loaded model memory exposed by backend control
surfaces. It must not edit Ollmo's durable truth files, response frames, or
artifact registries.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib import error as urlerror
from urllib import request as urlrequest


_INSTANCE_ID_KEYS = {
    'instance_id',
    'response_instance_id',
    'fill_instance_id',
    'target_instance_id',
    'selected_instance_id',
    'resolved_instance_id',
    'execution_instance_id',
}
_INSTANCE_ID_LIST_KEYS = {'instance_ids', 'response_instance_ids', 'fill_instance_ids'}
_BUSY_ACTIVITY_STATES = {'busy', 'executing', 'generating', 'streaming', 'working', 'loading', 'starting', 'stopping'}
_UNLOADABLE_MLX_BACKENDS = {'mlx', 'mlx_vlm'}


class _StdlibPostResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = int(status_code)
        self._body = body

    @property
    def text(self) -> str:
        return self._body.decode('utf-8', errors='replace')

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 400):
            raise RuntimeError(f'HTTP {self.status_code}: {self.text[:200]}')


def _stdlib_post_json(url: str, *, json: Optional[dict[str, Any]] = None, timeout: Optional[float] = None) -> _StdlibPostResponse:
    data = None
    headers = {}
    if json is not None:
        data = json_module_dumps(json).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urlrequest.Request(url, data=data, method='POST', headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            body = response.read()
            status_code = getattr(response, 'status', response.getcode())
            return _StdlibPostResponse(status_code, body)
    except urlerror.HTTPError as exc:
        body = exc.read()
        return _StdlibPostResponse(exc.code, body)


def json_module_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def normalize_post_response_substrate_unload_policy(value: Any) -> str:
    token = str(value or '').strip().lower()
    if token in {'off', 'false', '0', 'disabled', 'disable', 'none', 'no'}:
        return 'off'
    if token in {'', 'conservative', 'true', '1', 'final', 'on', 'enabled', 'enable', 'yes'}:
        return 'conservative'
    return 'conservative'


def _append_unique(values: list[str], seen: set[str], value: Any) -> None:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_unique(values, seen, item)
        return
    text = str(value or '').strip()
    if not text or text in seen:
        return
    seen.add(text)
    values.append(text)


def collect_response_substrate_instance_ids(*payloads: Any) -> list[str]:
    """Collect instance ids from response truth without inferring from prose."""

    collected: list[str] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, raw_value in value.items():
                key_text = str(key)
                if key_text in _INSTANCE_ID_KEYS:
                    _append_unique(collected, seen, raw_value)
                elif key_text in _INSTANCE_ID_LIST_KEYS:
                    _append_unique(collected, seen, raw_value)
                walk(raw_value)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)

    for payload in payloads:
        walk(payload)
    return collected


def _instance_identity_tokens(instance: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()

    def add(value: Any) -> None:
        text = str(value or '').strip()
        if text:
            tokens.add(text)

    for key in ('instance_id', 'id', 'name'):
        add(instance.get(key))
    metadata = instance.get('metadata') if isinstance(instance.get('metadata'), Mapping) else {}
    for key in ('instance_id', 'id', 'name'):
        add(metadata.get(key))
    backend_runtime = (
        instance.get('backend_runtime')
        if isinstance(instance.get('backend_runtime'), Mapping)
        else {}
    )
    add(backend_runtime.get('instance_id'))
    return tokens


def _runtime_payload(instance: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = instance.get('backend_runtime')
    if isinstance(runtime, Mapping):
        return runtime
    runtime = instance.get('runtime')
    return runtime if isinstance(runtime, Mapping) else {}


def _backend_name(instance: Mapping[str, Any]) -> str:
    runtime = _runtime_payload(instance)
    return str(
        instance.get('backend')
        or instance.get('backend_package')
        or runtime.get('backend')
        or runtime.get('backend_package')
        or ''
    ).strip().lower()


def _instance_model_name(instance: Mapping[str, Any]) -> str:
    runtime = _runtime_payload(instance)
    return str(
        instance.get('model')
        or instance.get('model_name')
        or instance.get('modelName')
        or runtime.get('model')
        or runtime.get('model_name')
        or ''
    ).strip()


def _instance_port(instance: Mapping[str, Any]) -> Optional[int]:
    runtime = _runtime_payload(instance)
    for value in (instance.get('port'), runtime.get('port')):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_busy_instance(instance: Mapping[str, Any]) -> bool:
    runtime = _runtime_payload(instance)
    candidates = [
        instance.get('activity'),
        instance.get('readiness'),
        runtime.get('activity'),
        runtime.get('readiness'),
    ]
    return any(str(value or '').strip().lower() in _BUSY_ACTIVITY_STATES for value in candidates)


@dataclass
class PostResponseSubstrateHygieneRuntimeOwner:
    policy: str
    load_running_instances: Callable[[], list[dict[str, Any]]]
    merge_instances_with_runtime_status: Callable[..., list[dict[str, Any]]]
    log_unified_event: Callable[..., Any]
    requests_post: Callable[..., Any] = _stdlib_post_json
    runtime_status_path_getter: Optional[Callable[[], Any]] = None
    timeout_sec: float = 8.0
    run_async: bool = True

    def schedule_post_response_substrate_hygiene(
        self,
        response_payload: Mapping[str, Any],
        *,
        route_payload: Optional[Mapping[str, Any]] = None,
        reason: str = 'response_terminal',
    ) -> dict[str, Any]:
        if normalize_post_response_substrate_unload_policy(self.policy) == 'off':
            return {'status': 'disabled', 'policy': 'off'}

        instance_ids = collect_response_substrate_instance_ids(response_payload, route_payload)
        if not instance_ids:
            return {'status': 'skipped', 'skip_reason': 'no_response_instances'}

        response_id = str(response_payload.get('id') or '').strip()
        instances = self._load_instances()
        matched_instances = self._match_instances(instance_ids, instances)
        unmatched_results = []
        for instance_id in instance_ids:
            if instance_id in matched_instances:
                continue
            result = {
                'status': 'skipped',
                'instance_id': instance_id,
                'skip_reason': 'instance_not_running',
            }
            self._log_event(result, response_id=response_id, reason=reason)
            unmatched_results.append(result)
        matched_instance_ids = [instance_id for instance_id in instance_ids if instance_id in matched_instances]
        if not matched_instance_ids:
            return {
                'status': 'skipped',
                'skip_reason': 'no_running_response_instances',
                'results': unmatched_results,
            }

        if self.run_async:
            threading.Thread(
                target=self._run_post_response_substrate_hygiene_safely,
                kwargs={
                    'instance_ids': matched_instance_ids,
                    'response_id': response_id,
                    'reason': reason,
                    'instances': list(matched_instances.values()),
                },
                daemon=True,
            ).start()
            return {'status': 'scheduled', 'instance_ids': matched_instance_ids, 'results': unmatched_results}

        result = self.run_post_response_substrate_hygiene(
            instance_ids=matched_instance_ids,
            response_id=response_id,
            reason=reason,
            instances=list(matched_instances.values()),
        )
        if unmatched_results:
            result['results'] = unmatched_results + list(result.get('results') or [])
        return result

    def _run_post_response_substrate_hygiene_safely(
        self,
        *,
        instance_ids: list[str],
        response_id: str,
        reason: str,
        instances: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        try:
            self.run_post_response_substrate_hygiene(
                instance_ids=instance_ids,
                response_id=response_id,
                reason=reason,
                instances=instances,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception('Post-response substrate hygiene failed: %s', exc)

    def run_post_response_substrate_hygiene(
        self,
        *,
        instance_ids: list[str],
        response_id: str = '',
        reason: str = 'response_terminal',
        instances: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        runtime_instances = instances if isinstance(instances, list) else self._load_instances()
        matched_instances = self._match_instances(instance_ids, runtime_instances)
        results: list[dict[str, Any]] = []
        for instance_id in instance_ids:
            instance = matched_instances.get(instance_id)
            if not instance:
                result = {
                    'status': 'skipped',
                    'instance_id': instance_id,
                    'skip_reason': 'instance_not_running',
                }
                self._log_event(result, response_id=response_id, reason=reason)
                results.append(result)
                continue
            result = self._unload_instance(instance_id, instance, response_id=response_id, reason=reason)
            results.append(result)
        return {'status': 'completed', 'results': results}

    def _load_instances(self) -> list[dict[str, Any]]:
        instances = self.load_running_instances()
        if not isinstance(instances, list):
            instances = []
        kwargs: dict[str, Any] = {}
        if self.runtime_status_path_getter is not None:
            kwargs['path'] = self.runtime_status_path_getter()
        kwargs['refresh'] = True
        try:
            merged = self.merge_instances_with_runtime_status(instances, **kwargs)
        except TypeError:
            merged = self.merge_instances_with_runtime_status(instances)
        return [item for item in merged if isinstance(item, dict)]

    def _match_instances(
        self,
        instance_ids: list[str],
        instances: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        matches: dict[str, dict[str, Any]] = {}
        for instance in instances:
            tokens = _instance_identity_tokens(instance)
            for instance_id in instance_ids:
                if instance_id in tokens and instance_id not in matches:
                    matches[instance_id] = instance
        return matches

    def _unload_instance(
        self,
        instance_id: str,
        instance: Mapping[str, Any],
        *,
        response_id: str,
        reason: str,
    ) -> dict[str, Any]:
        backend = _backend_name(instance)
        model_name = _instance_model_name(instance)
        capability = str(instance.get('capability') or '').strip()
        if _is_busy_instance(instance):
            result = {
                'status': 'skipped',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'capability': capability,
                'skip_reason': 'instance_busy',
            }
            self._log_event(result, response_id=response_id, reason=reason)
            return result

        if backend == 'ollama':
            result = self._unload_ollama_instance(instance_id, instance, backend=backend, model_name=model_name)
            self._log_event(result, response_id=response_id, reason=reason)
            return result

        if backend in _UNLOADABLE_MLX_BACKENDS:
            result = self._unload_mlx_instance(instance_id, instance, backend=backend, model_name=model_name)
            self._log_event(result, response_id=response_id, reason=reason)
            return result

        result = {
            'status': 'skipped',
            'instance_id': instance_id,
            'backend': backend,
            'model': model_name,
            'capability': capability,
            'skip_reason': 'unsupported_backend_unload',
        }
        self._log_event(result, response_id=response_id, reason=reason)
        return result

    def _unload_ollama_instance(
        self,
        instance_id: str,
        instance: Mapping[str, Any],
        *,
        backend: str,
        model_name: str,
    ) -> dict[str, Any]:
        port = _instance_port(instance)
        if not port or not model_name:
            return {
                'status': 'skipped',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'skip_reason': 'missing_ollama_unload_target',
            }
        url = f'http://127.0.0.1:{port}/api/generate'
        try:
            response = self.requests_post(
                url,
                json={'model': model_name, 'prompt': '', 'stream': False, 'keep_alive': 0},
                timeout=self.timeout_sec,
            )
            if hasattr(response, 'raise_for_status'):
                response.raise_for_status()
            return {
                'status': 'ok',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'unload_url': url,
                'unload_kind': 'ollama_keep_alive_zero',
            }
        except Exception as exc:  # noqa: BLE001
            return {
                'status': 'failed',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'unload_url': url,
                'message': str(exc),
            }

    def _unload_mlx_instance(
        self,
        instance_id: str,
        instance: Mapping[str, Any],
        *,
        backend: str,
        model_name: str,
    ) -> dict[str, Any]:
        runtime = _runtime_payload(instance)
        unload_url = str(runtime.get('unload_url') or instance.get('unload_url') or '').strip()
        supports_unload = bool(runtime.get('supports_unload') or instance.get('supports_unload'))
        if not supports_unload or not unload_url:
            return {
                'status': 'skipped',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'skip_reason': 'unsupported_backend_unload',
            }
        try:
            response = self.requests_post(unload_url, json={}, timeout=self.timeout_sec)
            if hasattr(response, 'raise_for_status'):
                response.raise_for_status()
            return {
                'status': 'ok',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'unload_url': unload_url,
                'unload_kind': 'mlx_unload_url',
            }
        except Exception as exc:  # noqa: BLE001
            return {
                'status': 'failed',
                'instance_id': instance_id,
                'backend': backend,
                'model': model_name,
                'unload_url': unload_url,
                'message': str(exc),
            }

    def _log_event(self, result: Mapping[str, Any], *, response_id: str, reason: str) -> None:
        try:
            self.log_unified_event(
                category='responses',
                action='post_response_substrate_hygiene',
                status=str(result.get('status') or 'unknown'),
                response_id=response_id,
                instance_id=str(result.get('instance_id') or '').strip(),
                backend=str(result.get('backend') or '').strip(),
                model=str(result.get('model') or '').strip(),
                capability=str(result.get('capability') or '').strip(),
                reason=reason,
                skip_reason=result.get('skip_reason'),
                unload_kind=result.get('unload_kind'),
                unload_url=result.get('unload_url'),
                message=str(result.get('message') or '').strip(),
            )
        except Exception:  # noqa: BLE001
            logging.exception('Could not log post-response substrate hygiene event.')
