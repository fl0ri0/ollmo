"""Shared model lifecycle service layer for Ollmo."""

from __future__ import annotations

import difflib
import hashlib
import logging
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    SUPPORTED_CAPABILITIES,
    build_registry_metadata,
    infer_capability,
    normalize_backend,
    normalize_capability,
)
from helpers.session_controls import build_session_controls
from helpers.tts_model_metadata import read_snapshot_model_metadata
from ollmo_core.registry import list_runtime_entries, read_registry_entries
from ollmo_core.start_policy import (
    StartSourcePolicyError,
    attach_start_audit,
    validate_start_source,
)
from ollmo_runtime.ollama_model_manager import (
    StopResult,
    get_available_models as manager_get_available_models,
    pull_model as manager_pull_model,
    remove_model as manager_remove_model,
    start_model_instance as manager_start_model_instance,
    stop_model_instance as manager_stop_model_instance,
)
from ollmo_runtime.llama_cpp_model_manager import (
    list_available_llama_cpp_models,
    pull_llama_cpp_model,
    remove_llama_cpp_model,
    start_llama_cpp_instance,
    stop_llama_cpp_instance,
)

try:
    from ollmo_runtime.mlx_model_manager import (
        list_cached_models as mlx_list_cached_models,
        pull_hf_model as mlx_pull_hf_model,
        remove_hf_model as mlx_remove_hf_model,
        start_mlx_model,
        stop_mlx_instance,
    )
except ImportError:  # pragma: no cover - optional dependency
    mlx_list_cached_models = None
    mlx_pull_hf_model = None
    mlx_remove_hf_model = None
    start_mlx_model = None
    stop_mlx_instance = None


class RuntimeRequestError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


StartModelRequestError = RuntimeRequestError

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


REPO_ROOT = Path(__file__).resolve().parent.parent
START_INSTANCE_LOCK_PATH = Path(tempfile.gettempdir()) / (
    f"ollmo-start-instance-{hashlib.sha256(str(REPO_ROOT).encode('utf-8')).hexdigest()[:16]}.lock"
)
_START_INSTANCE_THREAD_LOCK = threading.Lock()


@contextmanager
def start_instance_lock():
    START_INSTANCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    wait_started = time.monotonic()
    with _START_INSTANCE_THREAD_LOCK:
        waited = time.monotonic() - wait_started
        if waited > 0.05:
            logging.info('Waited %.2fs for in-process start lock.', waited)

        if fcntl is None:
            yield
            return

        with START_INSTANCE_LOCK_PATH.open('a+', encoding='utf-8') as handle:
            file_wait_started = time.monotonic()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            file_waited = time.monotonic() - file_wait_started
            if file_waited > 0.05:
                logging.info('Waited %.2fs for filesystem start lock.', file_waited)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical_model_name(data: dict) -> str:
    model_name = (data.get('modelName') or data.get('model') or '').strip()
    if not model_name:
        raise RuntimeRequestError("Parameter 'model' fehlt.")
    return model_name


def with_model_metadata(
    model_name: str,
    backend: str,
    capability: Optional[str],
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        'name': model_name,
        'model': model_name,
    }
    if extra:
        payload.update(extra)
    payload.update(build_registry_metadata(model_name, backend, capability, metadata=payload))
    return payload


def list_running_instances(config_path: str = 'model_ports.json', *, prune: bool = False) -> List[dict]:
    """Return configured runtime instances without pruning the stable registry by default.

    Read/status/routing callers should preserve `model_ports.json` and let
    runtime status report unreachable instances. Explicit lifecycle or hygiene
    callers can pass `prune=True` when they intentionally want to rewrite the
    registry.
    """
    instances = list_runtime_entries(prune=prune, path=config_path, sync_external=False)
    formatted = []
    for inst in instances:
        entry = dict(inst)
        model_name = entry.get('modelName') or entry.get('model') or ''
        metadata = build_registry_metadata(
            model_name,
            entry.get('backend'),
            entry.get('capability'),
            metadata=entry,
        )
        entry.update(metadata)
        if model_name:
            entry['model'] = model_name
        entry.update(
            read_snapshot_model_metadata(
                model_name,
                entry.get('request_model') or entry.get('model_path'),
            )
        )
        entry['session_controls'] = build_session_controls(entry)
        formatted.append(entry)
    return formatted


def lookup_instance(instance_id: str, config_path: str = 'model_ports.json') -> Optional[dict]:
    instances = list_running_instances(config_path=config_path)
    return next((inst for inst in instances if inst.get('instance_id') == instance_id), None)


def list_available_models(include_limits: bool = False) -> List[dict]:
    aggregated = []

    ollama_models = manager_get_available_models(include_limits=include_limits)
    for entry in ollama_models:
        if isinstance(entry, dict):
            model_name = entry.get('name') or entry.get('modelName') or entry.get('model')
            extra = dict(entry)
            extra.pop('model', None)
            extra.pop('modelName', None)
        else:
            model_name = str(entry)
            extra = {}
        model_name = (model_name or '').strip()
        if not model_name:
            continue
        aggregated.append(
            with_model_metadata(
                model_name,
                'ollama',
                extra.get('capability'),
                extra=extra,
            )
        )

    if mlx_list_cached_models:
        try:
            for item in mlx_list_cached_models():
                model_name = (item.get('repo') or '').strip()
                if not model_name:
                    continue
                aggregated.append(
                    with_model_metadata(
                        model_name,
                        'mlx',
                        item.get('capability'),
                        extra={
                            'model_path': item.get('path'),
                            'size_gb': item.get('size_gb'),
                            'mtime': item.get('mtime').isoformat() if item.get('mtime') else None,
                            'model_source': 'huggingface',
                            'server_kind': item.get('mlx_server'),
                            'backend_package': item.get('backend_package'),
                            'backend_contract': item.get('backend_contract'),
                            'provider_capabilities': item.get('provider_capabilities'),
                            'backend_metadata': item.get('backend_metadata'),
                            'launch_defaults': (item.get('backend_metadata') or {}).get('launch_defaults'),
                            'runnable': item.get('runnable', True),
                            'disabled_reason': item.get('disabled_reason'),
                            **({'limits': item.get('limits')} if item.get('limits') else {}),
                            **read_snapshot_model_metadata(model_name, item.get('path')),
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logging.warning('Could not list MLX models: %s', exc)

    try:
        for item in list_available_llama_cpp_models():
            model_name = (item.get('name') or item.get('modelName') or item.get('model') or '').strip()
            if not model_name:
                continue
            extra = dict(item)
            extra.pop('model', None)
            extra.pop('modelName', None)
            aggregated.append(
                with_model_metadata(
                    model_name,
                    'llama_cpp',
                    extra.get('capability'),
                    extra=extra,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning('Could not list llama.cpp models: %s', exc)

    return aggregated


def pull_model(model_name: str, backend: str = 'ollama') -> Tuple[bool, str]:
    normalized_backend = normalize_backend(backend)
    if normalized_backend == 'mlx':
        if not mlx_pull_hf_model:
            return False, 'MLX/Hugging Face pull is not available.'
        return mlx_pull_hf_model(model_name)
    if normalized_backend == 'llama_cpp':
        return pull_llama_cpp_model(model_name)
    return manager_pull_model(model_name)


def remove_model(
    model_name: str,
    backend: str = 'ollama',
    *,
    model_source: Optional[str] = None,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
) -> Tuple[bool, str]:
    normalized_backend = normalize_backend(backend)
    if normalized_backend == 'mlx':
        if not mlx_remove_hf_model:
            return False, 'MLX/Hugging Face remove is not available.'
        return mlx_remove_hf_model(model_name)
    if normalized_backend == 'llama_cpp':
        return remove_llama_cpp_model(
            model_name,
            model_source=model_source,
            model_path=model_path,
            hf_repo=hf_repo,
            hf_file=hf_file,
        )
    return manager_remove_model(model_name)


def _extract_ollama_model_names() -> List[str]:
    names: List[str] = []
    raw = manager_get_available_models(include_limits=False)
    for entry in raw:
        if isinstance(entry, dict):
            candidate = entry.get('name') or entry.get('modelName') or entry.get('model')
        else:
            candidate = str(entry)
        candidate = (candidate or '').strip()
        if candidate:
            names.append(candidate)
    return sorted(set(names))


def _available_ollama_model_capability(model_name: str) -> Optional[str]:
    target_variants = _ollama_model_variants(model_name)
    raw = manager_get_available_models(include_limits=False)
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        candidate = str(entry.get('name') or entry.get('modelName') or entry.get('model') or '').strip()
        if not candidate:
            continue
        if _ollama_model_variants(candidate).isdisjoint(target_variants):
            continue
        capability = normalize_capability(entry.get('capability'))
        if capability in SUPPORTED_CAPABILITIES:
            return capability
    return None


def _ollama_model_variants(model_name: str) -> set[str]:
    token = (model_name or '').strip().lower()
    if not token:
        return set()
    variants: set[str] = {token}
    base = token.split(':', 1)[0]
    variants.add(base)
    if '/' in base:
        variants.add(base.split('/', 1)[1])
    for item in list(variants):
        variants.add(f'{item}:latest')
    return variants


def validate_ollama_model_name(model_name: str) -> str:
    available = _extract_ollama_model_names()
    if model_name in available:
        return model_name

    lower_map = {item.lower(): item for item in available}
    lowered = model_name.lower()
    if lowered in lower_map:
        return lower_map[lowered]

    requested_variants = _ollama_model_variants(model_name)
    matches = []
    for candidate in available:
        candidate_variants = _ollama_model_variants(candidate)
        if requested_variants & candidate_variants:
            matches.append(candidate)
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeRequestError(
            f"Model name '{model_name}' is ambiguous. Use an exact name from 'ollama list'.",
            status_code=400,
            details={'candidates': matches},
        )

    suggestions = difflib.get_close_matches(model_name, available, n=3, cutoff=0.4)
    message = f"Model '{model_name}' is not available in 'ollama list'."
    if suggestions:
        message += f" Did you mean: {', '.join(suggestions)}?"
    raise RuntimeRequestError(
        message,
        status_code=400,
        details={'suggestions': suggestions, 'available_count': len(available)},
    )


def _normalize_instance_response(
    instance: Optional[dict],
    model_name: str,
    backend: str,
    capability: str,
    *,
    start_source: str,
) -> dict:
    if not instance:
        raise RuntimeError(f"Failed to start '{model_name}'.")
    normalized = dict(instance)
    normalized.update(build_registry_metadata(model_name, backend, capability, metadata=normalized))
    normalized['model'] = model_name
    return attach_start_audit(
        normalized,
        start_source=start_source,
        context='start_instance',
        extra={'backend': backend, 'capability': capability},
    )


def _start_ollama_instance(model_name: str, capability: str, *, start_source: str) -> dict:
    resolved_model_name = validate_ollama_model_name(model_name)
    instance = manager_start_model_instance(
        resolved_model_name,
        capability=capability,
        start_source=start_source,
    )
    return _normalize_instance_response(
        instance,
        resolved_model_name,
        'ollama',
        capability,
        start_source=start_source,
    )


def _start_mlx_speech_instance(
    model_name: str,
    capability: str,
    model_path: Optional[str],
    preferred_port: Optional[int],
    launch_defaults: Optional[dict[str, Any]],
    start_source: str,
) -> dict:
    if not start_mlx_model:
        raise RuntimeRequestError('MLX support is not available.', status_code=501)
    try:
        instance = start_mlx_model(
            model_name,
            model_path,
            preferred_port,
            capability=capability,
            launch_defaults=launch_defaults,
            start_source=start_source,
        )
    except ValueError as exc:
        raise RuntimeRequestError(str(exc), status_code=400) from exc
    except RuntimeError as exc:
        raise RuntimeRequestError(str(exc), status_code=400) from exc
    return _normalize_instance_response(
        instance,
        model_name,
        'mlx',
        capability,
        start_source=start_source,
    )


def _start_mlx_instance(
    model_name: str,
    capability: str,
    model_path: Optional[str],
    preferred_port: Optional[int],
    launch_defaults: Optional[dict[str, Any]],
    start_source: str,
) -> dict:
    if not start_mlx_model:
        raise RuntimeRequestError('MLX support is not available.', status_code=501)
    try:
        instance = start_mlx_model(
            model_name,
            model_path,
            preferred_port,
            capability=capability,
            launch_defaults=launch_defaults,
            start_source=start_source,
        )
    except ValueError as exc:
        raise RuntimeRequestError(str(exc), status_code=400) from exc
    except RuntimeError as exc:
        raise RuntimeRequestError(str(exc), status_code=400) from exc
    return _normalize_instance_response(
        instance,
        model_name,
        'mlx',
        capability,
        start_source=start_source,
    )


def _start_llama_cpp_instance(
    model_name: str,
    capability: str,
    model_path: Optional[str],
    preferred_port: Optional[int],
    start_source: str,
    hf_file: Optional[str] = None,
) -> dict:
    try:
        instance = start_llama_cpp_instance(
            model_name,
            model_path=model_path,
            preferred_port=preferred_port,
            capability=capability,
            hf_file=hf_file,
            start_source=start_source,
        )
    except ValueError as exc:
        raise RuntimeRequestError(str(exc), status_code=400) from exc
    except RuntimeError as exc:
        raise RuntimeRequestError(str(exc), status_code=400) from exc
    return _normalize_instance_response(
        instance,
        str(instance.get('model') or model_name),
        'llama_cpp',
        capability,
        start_source=start_source,
    )


def start_instance(
    model_name: str,
    backend: str,
    capability: Optional[str],
    *,
    model_path: Optional[str] = None,
    preferred_port: Optional[int] = None,
    hf_file: Optional[str] = None,
    launch_defaults: Optional[dict[str, Any]] = None,
    start_source: Optional[str] = None,
) -> dict:
    try:
        normalized_start_source = validate_start_source(
            start_source,
            context='start_instance',
        )
    except StartSourcePolicyError as exc:
        raise RuntimeRequestError(
            str(exc),
            status_code=400,
            details={'policy_violation': exc.policy_violation},
        ) from exc
    normalized_backend = normalize_backend(backend)
    snapshot_metadata = read_snapshot_model_metadata(model_name, model_path)
    normalized_capability = (
        normalize_capability(capability)
        or (_available_ollama_model_capability(model_name) if normalized_backend == 'ollama' else None)
        or infer_capability(model_name, normalized_backend, metadata=snapshot_metadata)
    )
    if normalized_capability not in SUPPORTED_CAPABILITIES:
        raise RuntimeRequestError(
            f"Unknown capability '{normalized_capability}' for model '{model_name}'.",
            status_code=400,
        )
    with start_instance_lock():
        if normalized_backend == 'ollama':
            if normalized_capability in {
                CAPABILITY_CHAT,
                CAPABILITY_EMBEDDING,
                CAPABILITY_VISION_ANALYSIS,
                CAPABILITY_IMAGE_GENERATION,
            }:
                return _start_ollama_instance(
                    model_name,
                    normalized_capability,
                    start_source=normalized_start_source,
                )
            raise RuntimeRequestError(
                f"Unsupported start type: backend='{normalized_backend}', capability='{normalized_capability}' "
                f"for model '{model_name}'.",
                status_code=400,
            )

        if normalized_backend == 'mlx':
            if normalized_capability in {
                CAPABILITY_CHAT,
                CAPABILITY_SPEECH_TO_TEXT,
                CAPABILITY_TEXT_TO_SPEECH,
                CAPABILITY_VISION_ANALYSIS,
            }:
                return _start_mlx_instance(
                    model_name,
                    normalized_capability,
                    model_path,
                    preferred_port,
                    launch_defaults,
                    normalized_start_source,
                )
            raise RuntimeRequestError(
                f"Unsupported start type: backend='{normalized_backend}', capability='{normalized_capability}' "
                f"for model '{model_name}'.",
                status_code=400,
            )

        if normalized_backend == 'llama_cpp':
            if normalized_capability in {CAPABILITY_CHAT, CAPABILITY_EMBEDDING}:
                return _start_llama_cpp_instance(
                    model_name,
                    normalized_capability,
                    model_path,
                    preferred_port,
                    start_source=normalized_start_source,
                    hf_file=hf_file,
                )
            raise RuntimeRequestError(
                f"Unsupported start type: backend='{normalized_backend}', capability='{normalized_capability}' "
                f"for model '{model_name}'.",
                status_code=400,
            )

        raise RuntimeRequestError(f"Unknown backend type '{backend}'.", status_code=400)


def stop_instance(instance_id: str, config_path: str = 'model_ports.json') -> Tuple[StopResult, Optional[dict]]:
    backend = None
    for inst in read_registry_entries(config_path):
        if inst.get('instance_id') == instance_id:
            backend = inst.get('backend') or 'ollama'
            break

    if backend == 'mlx':
        if not stop_mlx_instance:
            raise RuntimeRequestError('MLX support is not available.', status_code=501)
        success, instance = stop_mlx_instance(instance_id)
        state = 'stopped' if success else 'failed'
        message = (
            f"MLX instance '{instance_id}' was stopped."
            if success
            else f"MLX instance '{instance_id}' could not be stopped."
        )
        return StopResult(state=state, message=message, details={'backend': 'mlx'}), instance

    if backend == 'llama_cpp':
        success, instance = stop_llama_cpp_instance(instance_id)
        state = 'stopped' if success else 'failed'
        message = (
            f"llama.cpp instance '{instance_id}' was stopped."
            if success
            else f"llama.cpp instance '{instance_id}' could not be stopped."
        )
        return StopResult(state=state, message=message, details={'backend': 'llama_cpp'}), instance

    return manager_stop_model_instance(instance_id)
