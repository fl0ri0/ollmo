#!/usr/bin/env python3
"""Unified startup launcher for Ollama + MLX model instances."""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_runtime.ollama_model_manager import (
    CONFIG_FILE as OLLAMA_CONFIG_FILE,
    DEFAULT_SERVER_PORT,
    LOG_DIR as OLLAMA_LOG_DIR,
    ensure_default_server_running,
    format_capability_badge,
    format_model_label,
    initialize_instance_counters,
    is_port_listening,
    list_local_model_entries,
    start_model,
)
from ollmo_runtime.mlx_model_manager import (
    PORT_MAX as MLX_PORT_MAX,
    START_PORT as MLX_START_PORT,
    find_mlx_snapshots,
    format_repo_label,
    next_free_port as next_free_mlx_port,
    resolve_mlx_python,
    start_mlx_model,
)
from ollmo_runtime.llama_cpp_model_manager import (
    LLAMA_CPP_PORT_MAX,
    LLAMA_CPP_START_PORT,
    describe_llama_cpp_runtime_probe,
    list_available_llama_cpp_models,
    start_llama_cpp_instance,
)
from ollmo_runtime.runtime_hygiene import cleanup_runtime_hygiene
from ollmo_core.registry import (
    filter_active_registry_entries,
    read_registry_entries,
    write_registry_entries,
)

BACKEND_LABELS = {
    'ollama': 'Ollama',
    'mlx': 'MLX',
    'llama_cpp': 'llama.cpp',
}
CAPABILITY_LABELS = {
    'chat': 'Chat',
    'embedding': 'Embedding',
    'vision_analysis': 'OCR/Vision',
    'image_generation': 'Image',
    'text_to_speech': 'TTS',
    'speech_to_text': 'STT',
}
TOKEN_RANGE_PATTERN = re.compile(r'^\d+(?:-\d+)?$')
CONFIG_PATH = Path(OLLAMA_CONFIG_FILE)


@dataclass
class CatalogEntry:
    backend: str
    model_name: str
    display_label: str
    capability: str
    capability_badge: str
    size: str
    backend_label: str = ''
    type_label: str = ''
    details: str = ''
    model_path: str = ''


def _backend_label(backend: str) -> str:
    return BACKEND_LABELS.get(str(backend or '').strip().lower(), str(backend or '').strip())


def _capability_label(capability: str) -> str:
    return CAPABILITY_LABELS.get(str(capability or '').strip().lower(), str(capability or '').strip() or 'Unknown')


def _capability_cell(capability: str) -> str:
    return _capability_label(capability)


def _compact_languages(values: list[str]) -> str:
    if not values:
        return ''
    filtered = [token for token in values if token and token.lower() != 'auto']
    if not filtered:
        return ''
    return f'{len(filtered)} langs'


def _compact_speakers(values: list[str]) -> str:
    if not values:
        return ''
    return f'{len(values)} voices'


def _rich_type_label(entry: Dict[str, Any]) -> str:
    capability = str(entry.get('capability') or '').strip().lower()
    if capability == 'speech_to_text':
        if entry.get('stt_realtime'):
            return 'Realtime STT'
        return 'STT'
    if capability == 'text_to_speech':
        model_type = str(entry.get('tts_model_type') or '').strip().lower()
        if model_type == 'voice_design':
            return 'Voice TTS'
        if model_type == 'custom_voice':
            return 'Speaker TTS'
        if model_type == 'kitten_tts':
            return 'Speaker TTS'
        if model_type == 'voxtral_tts':
            return 'Multilang TTS'
        return 'TTS'
    if capability == 'vision_analysis':
        pipeline_tag = str(entry.get('snapshot_pipeline_tag') or '').strip().lower()
        if 'ocr' in str(entry.get('model_name') or entry.get('name') or '').lower():
            return 'OCR/Vision'
        if pipeline_tag in {'image-to-text', 'image-text-to-text'}:
            return 'Vision'
        return 'OCR/Vision'
    if capability == 'image_generation':
        return 'Image'
    if capability == 'embedding':
        return 'Embedding'
    return 'Chat'


def _rich_backend_label(entry: Dict[str, Any]) -> str:
    backend = str(entry.get('backend') or '').strip().lower()
    if backend != 'mlx':
        return _backend_label(backend)
    backend_package = str(entry.get('backend_package') or '').strip().lower()
    if backend_package == 'mlx_audio':
        return 'MLX Audio'
    if backend_package == 'mlx_whisper_shim':
        return 'MLX Whisper'
    if backend_package == 'mlx_vlm':
        return 'MLX VLM'
    if backend_package == 'mlx_lm':
        return 'MLX LM'
    server_kind = str((entry.get('backend_metadata') or {}).get('server_kind') or '').strip().lower()
    if server_kind == 'mlx_audio':
        return 'MLX Audio'
    if server_kind == 'mlx_whisper':
        return 'MLX Whisper'
    if server_kind == 'mlx_vlm':
        return 'MLX VLM'
    return 'MLX LM'


def _rich_details(entry: Dict[str, Any]) -> str:
    parts: list[str] = []
    capability = str(entry.get('capability') or '').strip().lower()
    if capability == 'text_to_speech':
        speakers = entry.get('tts_speakers') if isinstance(entry.get('tts_speakers'), list) else []
        languages = entry.get('tts_languages') if isinstance(entry.get('tts_languages'), list) else []
        response_formats = entry.get('tts_response_formats') if isinstance(entry.get('tts_response_formats'), list) else []
        speaker_part = _compact_speakers([str(item) for item in speakers])
        if speaker_part:
            parts.append(speaker_part)
        language_part = _compact_languages([str(item) for item in languages])
        if language_part:
            parts.append(language_part)
        if response_formats:
            parts.append('/'.join(str(item) for item in response_formats[:3]))
    elif capability == 'speech_to_text':
        if entry.get('stt_realtime'):
            parts.append('realtime')
        languages = entry.get('snapshot_languages') if isinstance(entry.get('snapshot_languages'), list) else []
        language_part = _compact_languages([str(item) for item in languages])
        if language_part:
            parts.append(language_part)
    elif capability == 'vision_analysis':
        languages = entry.get('snapshot_languages') if isinstance(entry.get('snapshot_languages'), list) else []
        language_part = _compact_languages([str(item) for item in languages])
        if language_part:
            parts.append(language_part)
    return ' · '.join(parts)


def _render_catalog_option(
    index: int,
    entry: CatalogEntry,
    *,
    index_width: int,
    label_width: int,
    type_width: int,
    size_width: int,
    backend_width: int,
    details_width: int,
) -> str:
    return (
        f'{index:>{index_width}}   '
        f'{entry.display_label:<{label_width}}   '
        f'{entry.type_label:<{type_width}}   '
        f'{entry.size:>{size_width}}   '
        f'{entry.backend_label:<{backend_width}}   '
        f'{entry.details:<{details_width}}'
    )


def _expand_selection_token(token: str) -> List[int]:
    normalized = re.sub(r'\s+', '', str(token or '').replace('–', '-').replace('—', '-'))
    if not normalized or not TOKEN_RANGE_PATTERN.fullmatch(normalized):
        raise ValueError(f'Invalid selection value: {token}')
    if '-' not in normalized:
        return [int(normalized)]
    start_str, end_str = normalized.split('-', 1)
    start, end = int(start_str), int(end_str)
    if start <= end:
        return list(range(start, end + 1))
    return list(range(end, start + 1))


def _read_active_runtime_entries() -> tuple[List[Dict[str, Any]], bool]:
    raw_entries = read_registry_entries(CONFIG_PATH)
    filtered_entries = filter_active_registry_entries(raw_entries)
    changed = filtered_entries != raw_entries
    runtime_entries = [entry for entry in filtered_entries if not entry.get('agent')]
    return runtime_entries, changed


def _discover_ollama_entries() -> List[CatalogEntry]:
    if not ensure_default_server_running():
        print('⚠️  Ollama discovery is not available. Continuing with MLX-only.')
        return []

    entries: List[CatalogEntry] = []
    for item in list_local_model_entries():
        item = {'backend': 'ollama', **dict(item)}
        model_name = str(item.get('name') or '').strip()
        if not model_name:
            continue
        capability = str(item.get('capability') or '').strip()
        entries.append(
            CatalogEntry(
                backend='ollama',
                model_name=model_name,
                display_label=format_model_label(model_name),
                capability=capability,
                capability_badge=format_capability_badge(capability),
                size=str(item.get('size') or 'unknown'),
                backend_label=_rich_backend_label(item),
                type_label=_rich_type_label(item),
                details=_rich_details(item),
            )
        )
    return entries


def _discover_mlx_entries() -> List[CatalogEntry]:
    try:
        mlx_python = resolve_mlx_python()
        print(f'ℹ️  Using MLX Python: {mlx_python}')
    except RuntimeError as err:
        print(f'⚠️  {err}')
        return []

    entries: List[CatalogEntry] = []
    for item in find_mlx_snapshots():
        item = {'backend': 'mlx', **dict(item)}
        repo_id = str(item.get('repo') or '').strip()
        if not repo_id:
            continue
        capability = str(item.get('capability') or '').strip()
        entries.append(
            CatalogEntry(
                backend='mlx',
                model_name=repo_id,
                display_label=format_repo_label(repo_id),
                capability=capability,
                capability_badge=format_capability_badge(capability),
                size=f"{float(item.get('size_gb') or 0.0):.2f} GB",
                backend_label=_rich_backend_label(item),
                type_label=_rich_type_label(item),
                details=_rich_details(item),
                model_path=str(item.get('path') or ''),
            )
        )
    return entries


def _discover_catalog() -> List[CatalogEntry]:
    catalog = _discover_ollama_entries() + _discover_mlx_entries() + _discover_llama_cpp_entries()
    backend_rank = {'ollama': 0, 'mlx': 1, 'llama_cpp': 2}
    catalog.sort(key=lambda entry: (backend_rank.get(entry.backend, 9), entry.display_label.lower()))
    return catalog


def _discover_llama_cpp_entries() -> List[CatalogEntry]:
    _print_llama_cpp_runtime_hint()
    entries: List[CatalogEntry] = []
    for item in list_available_llama_cpp_models():
        item = {'backend': 'llama_cpp', **dict(item)}
        if item.get('runnable') is False:
            continue
        model_name = str(item.get('name') or item.get('model') or '').strip()
        if not model_name:
            continue
        capability = str(item.get('capability') or '').strip() or 'chat'
        model_path = str(item.get('model_path') or '').strip()
        hf_repo = str(item.get('hf_repo') or '').strip()
        size = 'unknown'
        if isinstance(item.get('size_gb'), (int, float)) and float(item.get('size_gb') or 0.0) > 0:
            size = f"{float(item.get('size_gb') or 0.0):.2f} GB"
        elif model_path:
            try:
                size = f"{Path(model_path).stat().st_size / (1024 ** 3):.2f} GB"
            except OSError:
                pass
        entries.append(
            CatalogEntry(
                backend='llama_cpp',
                model_name=model_name,
                display_label=format_model_label(model_name),
                capability=capability,
                capability_badge=format_capability_badge(capability),
                size=size,
                backend_label=_rich_backend_label(item),
                type_label=_rich_type_label(item),
                details='GGUF local file' if model_path else (hf_repo or 'GGUF Hugging Face repo'),
                model_path=model_path,
            )
        )
    return entries


def _print_llama_cpp_runtime_hint() -> None:
    probe = describe_llama_cpp_runtime_probe()
    if probe.get('runtime_state') == 'runnable':
        server_bin = str((probe.get('detection') or {}).get('server_bin') or '').strip()
        if server_bin:
            print(f'ℹ️  Using llama.cpp server: {server_bin}')
        else:
            print('ℹ️  Using llama.cpp server.')
        return
    issues = probe.get('issues') if isinstance(probe.get('issues'), list) else []
    reason = '; '.join(str(item) for item in issues if item) or 'llama.cpp runtime unavailable.'
    print(f'⚠️  llama.cpp startup unavailable: {reason}')


def _prompt_selection(catalog: List[CatalogEntry]) -> List[CatalogEntry]:
    print('\nWhich models should be started as their own instance?')
    print("Allowed: '1 2', '1,2', '1-5', '1 3-5', 'a' for all.")
    print("You can choose a model more than once (for example: '1 1 2').")
    print('Enter = cancel.\n')

    index_width = max(2, len(str(len(catalog))))
    label_width = min(34, max(len(entry.display_label) for entry in catalog) + 3)
    type_width = max(len('Type'), max(len(entry.type_label or _capability_cell(entry.capability)) for entry in catalog))
    size_width = max(len('Size'), max(len(entry.size) for entry in catalog))
    backend_width = max(len('Backend'), max(len(entry.backend_label or _backend_label(entry.backend)) for entry in catalog))
    details_width = max(len('Details'), max(len(entry.details or '') for entry in catalog))
    header = (
        f'{"No.":>{index_width}}   '
        f'{"Model":<{label_width}}   '
        f'{"Type":<{type_width}}   '
        f'{"Size":>{size_width}}   '
        f'{"Backend":<{backend_width}}   '
        f'{"Details":<{details_width}}'
    )
    separator = (
        f'{"-" * index_width}   '
        f'{"-" * label_width}   '
        f'{"-" * type_width}   '
        f'{"-" * size_width}   '
        f'{"-" * backend_width}   '
        f'{"-" * details_width}'
    )
    print(header)
    print(separator)
    for idx, entry in enumerate(catalog, start=1):
        print(
            _render_catalog_option(
                idx,
                entry,
                index_width=index_width,
                label_width=label_width,
                type_width=type_width,
                size_width=size_width,
                backend_width=backend_width,
                details_width=details_width,
            )
        )

    try:
        choice = input('▶️  Selection: ').strip().lower()
    except EOFError:
        print('ℹ️  No interactive input is available; continuing without model selection.')
        return []
    if not choice:
        return []

    selection_indices: List[int] = []
    if choice in {'a', 'all', '*'}:
        selection_indices = list(range(1, len(catalog) + 1))
    else:
        for token in re.split(r'[,\s]+', choice):
            token = token.strip()
            if not token:
                continue
            try:
                selection_indices.extend(_expand_selection_token(token))
            except ValueError:
                print(f'⚠️  Ignoring invalid input: {token}')

    if not selection_indices:
        return []

    invalid = [idx for idx in selection_indices if idx < 1 or idx > len(catalog)]
    if invalid:
        unique_invalid = sorted(set(invalid))
        printable = ', '.join(str(idx) for idx in unique_invalid[:6])
        if len(unique_invalid) > 6:
            printable += ', …'
        print(f'⚠️  Skipping invalid numbers ({printable}); there are only {len(catalog)} model(s).')

    valid_indices = [idx for idx in selection_indices if 1 <= idx <= len(catalog)]
    if not valid_indices:
        return []

    grouped = OrderedDict()
    for idx in valid_indices:
        grouped.setdefault(idx, 0)
        grouped[idx] += 1
    summary_parts = []
    for idx, count in grouped.items():
        entry = catalog[idx - 1]
        token = f'#{idx} {entry.display_label} [{_backend_label(entry.backend)}]'
        if count > 1:
            token += f' ×{count}'
        summary_parts.append(token)
    print(f"➡️  Selection: {', '.join(summary_parts)}")
    return [catalog[idx - 1] for idx in valid_indices]


def _write_registry_once(runtime_entries: List[Dict[str, Any]]) -> None:
    write_registry_entries(
        runtime_entries,
        path=CONFIG_PATH,
        preserve_agents=True,
        sync_external=False,
    )


def main() -> int:
    current_instances, pruned_changed = _read_active_runtime_entries()
    initialize_instance_counters(current_instances)
    used_ports = {entry.get('port') for entry in current_instances if entry.get('port')}

    catalog = _discover_catalog()
    if not catalog:
        print('⚠️  No models are available to start. Continuing with the Ollmo control plane only.')
        if pruned_changed:
            _write_registry_once(current_instances)
        return 0

    selected_entries = _prompt_selection(catalog)
    if not selected_entries:
        print('ℹ️  No models were selected.')
        if pruned_changed:
            _write_registry_once(current_instances)
        return 0

    started_servers: List[Dict[str, Any]] = []
    current_mlx_port = MLX_START_PORT
    current_llama_cpp_port = LLAMA_CPP_START_PORT

    for entry in selected_entries:
        if entry.backend == 'ollama':
            server_info = start_model(
                entry.model_name,
                used_ports,
                current_instances,
                capability=entry.capability,
                start_source='startup_policy',
            )
            if server_info:
                started_servers.append(server_info)
                current_instances.append(server_info)
                continue
            else:
                print(f"⚠️  Error while starting an instance of '{entry.display_label}'.")
                continue

        if entry.backend == 'mlx':
            try:
                preferred_port = next_free_mlx_port(current_mlx_port, MLX_PORT_MAX)
            except RuntimeError as exc:
                print(f'❌ {exc}')
                break
            current_mlx_port = preferred_port + 1
            print(f"\n🚀 Starting instance '{entry.display_label}' ({entry.capability_badge}) on port {preferred_port}...")
            try:
                instance_record = start_mlx_model(
                    entry.model_name,
                    model_path=entry.model_path,
                    preferred_port=preferred_port,
                    capability=entry.capability,
                    register_instance=False,
                    prune_registry=False,
                    start_source='startup_policy',
                )
            except Exception as exc:  # noqa: BLE001
                print(f'❌ Start failed: {exc}')
                continue
            print(f"✅ Running: {entry.display_label} at http://127.0.0.1:{preferred_port}")
            print(
                f"   {entry.capability_badge} | Instance: {instance_record['instance_id']} | Logs: {instance_record.get('log')}"
            )
            started_servers.append(instance_record)
            current_instances.append(instance_record)
            continue

        try:
            preferred_port = next_free_mlx_port(current_llama_cpp_port, LLAMA_CPP_PORT_MAX)
        except RuntimeError as exc:
            print(f'❌ {exc}')
            break
        current_llama_cpp_port = preferred_port + 1
        print(f"\n🚀 Starting instance '{entry.display_label}' ({entry.capability_badge}) on port {preferred_port}...")
        try:
            instance_record = start_llama_cpp_instance(
                entry.model_name,
                model_path=entry.model_path,
                preferred_port=preferred_port,
                capability=entry.capability,
                register_instance=False,
                start_source='startup_policy',
            )
        except Exception as exc:  # noqa: BLE001
            print(f'❌ Start failed: {exc}')
            continue
        print(f"✅ Running: {entry.display_label} at http://127.0.0.1:{preferred_port}")
        print(
            f"   {entry.capability_badge} | Instance: {instance_record['instance_id']} | Logs: {instance_record.get('log')}"
        )
        started_servers.append(instance_record)
        current_instances.append(instance_record)

    if started_servers or pruned_changed:
        _write_registry_once(current_instances)

    if started_servers:
        ollama_count = sum(1 for entry in started_servers if str(entry.get('backend') or '').lower() == 'ollama')
        mlx_count = sum(1 for entry in started_servers if str(entry.get('backend') or '').lower() == 'mlx')
        llama_cpp_count = sum(1 for entry in started_servers if str(entry.get('backend') or '').lower() == 'llama_cpp')
        print(
            f"\n✅ Successfully started {len(started_servers)} model server(s) and loaded their models. "
            f"Details in '{CONFIG_PATH.name}'."
        )
        print(f'   Started: {ollama_count} Ollama, {mlx_count} MLX, {llama_cpp_count} llama.cpp.')
    else:
        print('\n⚠️  No model servers could be started.')

    active_global_logs = []
    if is_port_listening(DEFAULT_SERVER_PORT):
        active_global_logs.append(OLLAMA_LOG_DIR / f'ollama_default_server_{DEFAULT_SERVER_PORT}.log')

    hygiene_summary = cleanup_runtime_hygiene(
        registry_path=CONFIG_PATH,
        status_path=REPO_ROOT / 'state' / 'runtime_status.json',
        log_dir=REPO_ROOT / 'logs',
        sync_external=False,
        active_global_log_paths=active_global_logs,
    )
    if hygiene_summary.get('archived_count'):
        print(
            "🧹 Runtime hygiene after startup: "
            f"{hygiene_summary.get('archived_count', 0)} stale logs archived."
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
