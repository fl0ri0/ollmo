"""Normalized backend discovery and lifecycle contract for local Ollmo backends."""

from __future__ import annotations

import datetime as dt
from typing import Any, Iterable

from helpers.model_capabilities import normalize_backend
from ollmo_runtime.llama_cpp_model_manager import describe_llama_cpp_runtime_probe
from ollmo_runtime.mlx_model_manager import describe_mlx_runtime_variants
from ollmo_runtime.ollama_model_manager import describe_ollama_runtime_probe

BACKEND_VARIANT_ORDER = (
    'ollama',
    'llama_cpp',
    'mlx_lm',
    'mlx_vlm',
    'mlx_audio',
    'mlx_whisper',
)

BACKEND_VARIANT_DEFAULTS = {
    'ollama': {
        'family': 'ollama',
        'variant': 'ollama',
        'label': 'Ollama',
    },
    'llama_cpp': {
        'family': 'llama.cpp',
        'variant': 'llama_cpp',
        'label': 'llama.cpp',
    },
    'mlx_lm': {
        'family': 'mlx',
        'variant': 'mlx_lm',
        'label': 'MLX LM',
    },
    'mlx_vlm': {
        'family': 'mlx',
        'variant': 'mlx_vlm',
        'label': 'MLX VLM',
    },
    'mlx_audio': {
        'family': 'mlx',
        'variant': 'mlx_audio',
        'label': 'MLX Audio',
    },
    'mlx_whisper': {
        'family': 'mlx',
        'variant': 'mlx_whisper',
        'label': 'MLX Whisper',
    },
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _variant_from_entry(entry: dict[str, Any]) -> str | None:
    backend = normalize_backend(entry.get('backend'))
    if backend == 'ollama':
        return 'ollama'
    if backend == 'llama_cpp':
        return 'llama_cpp'
    if backend != 'mlx':
        return None

    backend_package = str(entry.get('backend_package') or '').strip().lower()
    if backend_package == 'mlx_lm':
        return 'mlx_lm'
    if backend_package == 'mlx_vlm':
        return 'mlx_vlm'
    if backend_package == 'mlx_audio':
        return 'mlx_audio'
    if backend_package == 'mlx_whisper_shim':
        return 'mlx_whisper'

    server_kind = str(entry.get('mlx_server') or '').strip().lower()
    if server_kind == 'mlx_vlm':
        return 'mlx_vlm'
    if server_kind == 'mlx_audio':
        return 'mlx_audio'
    if server_kind == 'mlx_whisper':
        return 'mlx_whisper'

    backend_metadata = entry.get('backend_metadata') if isinstance(entry.get('backend_metadata'), dict) else {}
    metadata_package = str(backend_metadata.get('backend_package') or '').strip().lower()
    if metadata_package == 'mlx_vlm':
        return 'mlx_vlm'
    if metadata_package == 'mlx_audio':
        return 'mlx_audio'
    if metadata_package == 'mlx_whisper_shim':
        return 'mlx_whisper'
    return 'mlx_lm'


def _base_catalog_counts() -> dict[str, int]:
    return {
        'available_model_count': 0,
        'runnable_model_count': 0,
        'cached_only_model_count': 0,
        'running_instance_count': 0,
    }


def _compute_auto_wiring_state(
    runtime_state: str,
    *,
    running_instance_count: int,
    discoverable_model_count: int,
) -> str:
    if running_instance_count > 0:
        return 'active'
    if runtime_state == 'missing':
        return 'missing'
    if runtime_state == 'degraded':
        return 'degraded'
    if discoverable_model_count > 0:
        return 'discoverable'
    return 'unwired'


def _build_probe_map() -> dict[str, dict[str, Any]]:
    payload = {
        'ollama': describe_ollama_runtime_probe(),
        'llama_cpp': describe_llama_cpp_runtime_probe(),
    }
    payload.update(describe_mlx_runtime_variants())
    return payload


def build_backend_fabric_snapshot(
    instances: Iterable[dict[str, Any]] | None = None,
    available_models: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probe_map = _build_probe_map()
    instance_groups: dict[str, list[dict[str, Any]]] = {key: [] for key in BACKEND_VARIANT_ORDER}
    model_groups: dict[str, dict[str, int]] = {key: _base_catalog_counts() for key in BACKEND_VARIANT_ORDER}
    issues_by_variant: dict[str, list[str]] = {
        key: list((probe_map.get(key) or {}).get('issues') or [])
        for key in BACKEND_VARIANT_ORDER
    }

    for raw_entry in instances or []:
        if not isinstance(raw_entry, dict):
            continue
        variant = _variant_from_entry(raw_entry)
        if not variant or variant not in instance_groups:
            continue
        instance_groups[variant].append(dict(raw_entry))
        model_groups[variant]['running_instance_count'] += 1

    for raw_entry in available_models or []:
        if not isinstance(raw_entry, dict):
            continue
        variant = _variant_from_entry(raw_entry)
        if not variant or variant not in model_groups:
            continue
        counts = model_groups[variant]
        counts['available_model_count'] += 1
        runnable = raw_entry.get('runnable')
        if runnable is False:
            counts['cached_only_model_count'] += 1
            disabled_reason = str(raw_entry.get('disabled_reason') or '').strip()
            if disabled_reason and disabled_reason not in issues_by_variant[variant]:
                issues_by_variant[variant].append(disabled_reason)
        else:
            counts['runnable_model_count'] += 1

    backend_items: list[dict[str, Any]] = []
    summary = {
        'backend_count': 0,
        'runtime_runnable_backend_count': 0,
        'runtime_degraded_backend_count': 0,
        'runtime_missing_backend_count': 0,
        'wiring_active_backend_count': 0,
        'wiring_discoverable_backend_count': 0,
        'wiring_unwired_backend_count': 0,
        'wiring_degraded_backend_count': 0,
        'wiring_missing_backend_count': 0,
    }

    for variant in BACKEND_VARIANT_ORDER:
        defaults = BACKEND_VARIANT_DEFAULTS[variant]
        probe = dict(probe_map.get(variant) or {})
        runtime_state = str(probe.get('runtime_state') or 'missing').strip().lower() or 'missing'
        counts = model_groups[variant]
        auto_wiring_state = _compute_auto_wiring_state(
            runtime_state,
            running_instance_count=counts['running_instance_count'],
            discoverable_model_count=counts['runnable_model_count'],
        )
        record = {
            'backend_id': variant,
            'family': defaults['family'],
            'variant': defaults['variant'],
            'label': defaults['label'],
            'runtime_state': runtime_state,
            'auto_wiring_state': auto_wiring_state,
            'auto_detected': runtime_state != 'missing',
            'auto_wireable': runtime_state == 'runnable' and (
                counts['running_instance_count'] > 0 or counts['runnable_model_count'] > 0
            ),
            'catalog': counts,
            'instance_ids': [
                str(item.get('instance_id') or '').strip()
                for item in instance_groups[variant]
                if str(item.get('instance_id') or '').strip()
            ],
            'operations': probe.get('operations') if isinstance(probe.get('operations'), dict) else {},
            'detection': probe.get('detection') if isinstance(probe.get('detection'), dict) else {},
            'issues': issues_by_variant[variant],
        }
        backend_items.append(record)
        summary['backend_count'] += 1
        if runtime_state == 'runnable':
            summary['runtime_runnable_backend_count'] += 1
        elif runtime_state == 'degraded':
            summary['runtime_degraded_backend_count'] += 1
        else:
            summary['runtime_missing_backend_count'] += 1

        if auto_wiring_state == 'active':
            summary['wiring_active_backend_count'] += 1
        elif auto_wiring_state == 'discoverable':
            summary['wiring_discoverable_backend_count'] += 1
        elif auto_wiring_state == 'unwired':
            summary['wiring_unwired_backend_count'] += 1
        elif auto_wiring_state == 'degraded':
            summary['wiring_degraded_backend_count'] += 1
        elif auto_wiring_state == 'missing':
            summary['wiring_missing_backend_count'] += 1

    return {
        'schema_version': 1,
        'generated_at': _now_iso(),
        'backends': backend_items,
        'summary': summary,
    }
