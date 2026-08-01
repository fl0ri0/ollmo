"""Runtime manager for local llama.cpp server instances."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:  # pragma: no cover - optional in lightweight test environments
    class _RequestsFallback:
        class exceptions:  # noqa: D106 - compatibility shim
            RequestException = Exception

        @staticmethod
        def get(*args, **kwargs):
            raise RuntimeError('requests is required for llama.cpp HTTP probes.')

    requests = _RequestsFallback()

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_VISION_ANALYSIS,
    build_registry_metadata,
    normalize_capability,
)
from ollmo_core.start_policy import attach_start_audit, validate_start_source
from ollmo_core.status import remove_instance_status
from ollmo_runtime.child_process_env import sanitized_child_process_env
from ollmo_runtime.ollama_model_manager import (
    kill_processes_on_port,
    safe_log_fragment,
)
from ollmo_runtime.registry import (
    is_port_listening as registry_is_port_listening,
    pid_is_running as registry_pid_is_running,
    read_registry_entries,
    write_registry_entries,
)
from ollmo_runtime.runtime_log_hygiene import prepare_clean_runtime_log

CONFIG_FILE = Path('model_ports.json')
LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = Path('state')
STATE_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_PATH = STATE_DIR / 'llama_cpp_catalog.json'

DEFAULT_LLAMA_SERVER = '/opt/homebrew/bin/llama-server'
DEFAULT_LLAMA_CLI = '/opt/homebrew/bin/llama-cli'
DEFAULT_HF_CLI = '/opt/homebrew/bin/hf'
DEFAULT_MLX_HF_CLI = '/opt/mlx/venv/bin/hf'
ENV_LLAMA_SERVER = os.environ.get('LLAMA_CPP_SERVER_BIN')
ENV_LLAMA_CLI = os.environ.get('LLAMA_CPP_CLI_BIN')
ENV_HF_CLI = os.environ.get('LLAMA_CPP_HF_CLI_BIN') or os.environ.get('HF_CLI_BIN')
LLAMA_CPP_START_PORT = int(os.environ.get('LLAMA_CPP_START_PORT', '11551'))
LLAMA_CPP_PORT_MAX = int(os.environ.get('LLAMA_CPP_PORT_MAX', '11600'))
LLAMA_CPP_START_TIMEOUT_SEC = int(os.environ.get('LLAMA_CPP_START_TIMEOUT_SEC', '180'))
LLAMA_CPP_MODELS_DIR = os.environ.get('LLAMA_CPP_MODELS_DIR')

SAFE_INSTANCE_RE = re.compile(r'[^A-Za-z0-9_.:-]+')
GGUF_PREFERENCE_MARKERS = (
    'q4_k_m',
    'q4_k_s',
    'q4_0',
    'q4',
    'q5_k_m',
    'q5_k_s',
    'q5_0',
    'q5',
    'q6_k',
    'q6',
    'q8_0',
    'q8',
    'bf16',
    'f16',
    'fp16',
)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.environ.get(name, '')).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_int_with_override(name: str, default: int, *, minimum: int = 1) -> tuple[int, bool]:
    raw = str(os.environ.get(name, '')).strip()
    if not raw:
        return default, False
    try:
        value = int(raw)
    except ValueError:
        return default, False
    if value < minimum:
        return default, False
    return value, True


def _resolved_binary(configured: Optional[str], fallback_name: str, default_path: str) -> Optional[str]:
    configured_token = str(configured or '').strip()
    if configured_token:
        configured_path = Path(configured_token).expanduser()
        if configured_path.exists() and os.access(configured_path, os.X_OK):
            return str(configured_path)
    fallback = shutil.which(fallback_name)
    if fallback:
        return fallback
    default = Path(default_path).expanduser()
    if default.exists() and os.access(default, os.X_OK):
        return str(default)
    return None


def resolve_llama_server_bin() -> Optional[str]:
    return _resolved_binary(ENV_LLAMA_SERVER, 'llama-server', DEFAULT_LLAMA_SERVER)


def resolve_llama_cli_bin() -> Optional[str]:
    return _resolved_binary(ENV_LLAMA_CLI, 'llama-cli', DEFAULT_LLAMA_CLI)


def resolve_hf_cli_bin() -> Optional[str]:
    configured = _resolved_binary(ENV_HF_CLI, 'hf', DEFAULT_HF_CLI)
    if configured:
        return configured
    sibling_candidates = [
        Path(DEFAULT_MLX_HF_CLI),
        Path(sys.executable).expanduser().parent / 'hf',
    ]
    seen: set[str] = set()
    for candidate in sibling_candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, '')).strip().lower()
    if not raw:
        return default
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return default


@lru_cache(maxsize=4)
def _llama_server_help_text(server_bin: str) -> str:
    try:
        completed = subprocess.run(
            [server_bin, '--help'],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=sanitized_child_process_env(),
        )
    except Exception:
        return ''
    return '\n'.join(
        fragment
        for fragment in (completed.stdout, completed.stderr)
        if isinstance(fragment, str) and fragment.strip()
    )


def _server_supports_flag(server_bin: str, *flags: str) -> bool:
    help_text = _llama_server_help_text(server_bin)
    if not help_text:
        return False
    return all(flag in help_text for flag in flags if str(flag or '').strip())


def _llama_cpp_launch_defaults(
    server_bin: Optional[str],
    *,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
) -> dict[str, Any]:
    batch_size, _batch_size_explicit = _env_int_with_override('LLAMA_CPP_BATCH_SIZE', 512, minimum=32)
    ubatch_size, ubatch_size_explicit = _env_int_with_override('LLAMA_CPP_UBATCH_SIZE', 128, minimum=16)
    defaults = {
        'prompt_cache': _env_flag('LLAMA_CPP_CACHE_PROMPT', True),
        'kv_offload': _env_flag('LLAMA_CPP_KV_OFFLOAD', True),
        'flash_attention': str(os.environ.get('LLAMA_CPP_FLASH_ATTN', 'auto') or 'auto').strip().lower() or 'auto',
        'ctx_size': _env_int('LLAMA_CPP_CTX_SIZE', 32768, minimum=512),
        'batch_size': batch_size,
        'ubatch_size': ubatch_size,
    }
    if (
        _looks_like_multimodal_chat_family(model_name, hf_repo, hf_file, model_path)
        and not ubatch_size_explicit
    ):
        defaults['ubatch_size'] = defaults['batch_size']
    if defaults['flash_attention'] not in {'on', 'off', 'auto'}:
        defaults['flash_attention'] = 'auto'
    if defaults['ubatch_size'] > defaults['batch_size']:
        defaults['ubatch_size'] = defaults['batch_size']
    if not server_bin:
        defaults['supported_flags'] = {}
        return defaults
    defaults['supported_flags'] = {
        'cache_prompt': _server_supports_flag(server_bin, '--cache-prompt'),
        'kv_offload': _server_supports_flag(server_bin, '--kv-offload'),
        'flash_attention': _server_supports_flag(server_bin, '--flash-attn'),
        'ctx_size': _server_supports_flag(server_bin, '--ctx-size'),
        'batch_size': _server_supports_flag(server_bin, '--batch-size'),
        'ubatch_size': _server_supports_flag(server_bin, '--ubatch-size'),
        'mmproj': _server_supports_flag(server_bin, '--mmproj'),
    }
    return defaults


def _common_models_roots() -> list[Path]:
    roots: list[Path] = []
    if LLAMA_CPP_MODELS_DIR:
        roots.append(Path(LLAMA_CPP_MODELS_DIR).expanduser())
    roots.extend(
        [
            Path.home() / 'Models' / 'llama.cpp',
            Path.home() / 'Models' / 'GGUF',
            Path.home() / 'models' / 'llama.cpp',
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _huggingface_hub_cache_root() -> Path:
    explicit_cache = str(os.environ.get('HUGGINGFACE_HUB_CACHE') or '').strip()
    if explicit_cache:
        return Path(explicit_cache).expanduser()
    hf_home = str(os.environ.get('HF_HOME') or '').strip()
    if hf_home:
        return Path(hf_home).expanduser() / 'hub'
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def _hf_repo_cache_dir(repo_id: Optional[str]) -> Optional[Path]:
    token = str(repo_id or '').strip()
    if not token:
        return None
    return _huggingface_hub_cache_root() / f"models--{token.replace('/', '--')}"


def _hf_repo_id_from_cache_dir(cache_dir: Path | str) -> Optional[str]:
    name = Path(cache_dir).name
    if not name.startswith('models--'):
        return None
    token = name[len('models--'):]
    if not token:
        return None
    repo_id = token.replace('--', '/').strip()
    if '/' not in repo_id:
        return None
    return repo_id


def _latest_hf_snapshot_dir(repo_id: Optional[str]) -> Optional[Path]:
    repo_dir = _hf_repo_cache_dir(repo_id)
    if not repo_dir or not repo_dir.exists():
        return None
    snapshots_dir = repo_dir / 'snapshots'
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return None
    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    refs_dir = repo_dir / 'refs'
    if refs_dir.exists() and refs_dir.is_dir():
        for ref_name in ('main',):
            ref_path = refs_dir / ref_name
            try:
                ref_target = ref_path.read_text(encoding='utf-8').strip()
            except Exception:
                ref_target = ''
            if not ref_target:
                continue
            candidate = snapshots_dir / ref_target
            if candidate.exists() and candidate.is_dir():
                return candidate
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _estimate_directory_size_bytes(path: Path, *, suffixes: Optional[tuple[str, ...]] = None) -> int:
    total = 0
    for candidate in path.rglob('*'):
        try:
            if not candidate.is_file():
                continue
            if suffixes and candidate.suffix.lower() not in suffixes:
                continue
            total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _estimate_llama_cpp_source_size_gb(
    *,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
) -> Optional[float]:
    resolved_model_path = str(model_path or '').strip()
    if resolved_model_path:
        path = Path(resolved_model_path).expanduser()
        try:
            if path.exists() and path.is_file():
                return round(path.stat().st_size / (1024 ** 3), 2)
        except OSError:
            return None

    snapshot_dir = _latest_hf_snapshot_dir(hf_repo)
    if not snapshot_dir:
        return None

    resolved_hf_file = str(hf_file or '').strip()
    if resolved_hf_file:
        target = snapshot_dir / resolved_hf_file
        try:
            if target.exists() and target.is_file():
                return round(target.stat().st_size / (1024 ** 3), 2)
        except OSError:
            return None

    gguf_bytes = _estimate_directory_size_bytes(snapshot_dir, suffixes=('.gguf',))
    if gguf_bytes > 0:
        return round(gguf_bytes / (1024 ** 3), 2)

    total_bytes = _estimate_directory_size_bytes(snapshot_dir)
    if total_bytes > 0:
        return round(total_bytes / (1024 ** 3), 2)
    return None


def _hf_repo_has_cached_gguf(repo_id: Optional[str]) -> bool:
    snapshot_dir = _latest_hf_snapshot_dir(repo_id)
    if not snapshot_dir:
        return False
    return _estimate_directory_size_bytes(snapshot_dir, suffixes=('.gguf',)) > 0


def _snapshot_gguf_files(snapshot_dir: Optional[Path]) -> list[Path]:
    if not snapshot_dir or not snapshot_dir.exists() or not snapshot_dir.is_dir():
        return []
    discovered: list[Path] = []
    for candidate in snapshot_dir.rglob('*.gguf'):
        try:
            if candidate.is_file():
                discovered.append(candidate)
        except OSError:
            continue
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in discovered:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _cached_hf_repo_dirs() -> list[Path]:
    hub_root = _huggingface_hub_cache_root()
    if not hub_root.exists() or not hub_root.is_dir():
        return []
    discovered: list[Path] = []
    try:
        candidates = sorted(hub_root.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return []
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
        except OSError:
            continue
        if not candidate.name.startswith('models--'):
            continue
        discovered.append(candidate)
    return discovered


def _is_mmproj_gguf(path: Path | str) -> bool:
    name = Path(path).name.lower()
    return 'mmproj' in name or 'mm-proj' in name or 'projector' in name


def _select_mmproj_candidate(gguf_files: list[Path]) -> Optional[Path]:
    mmproj_candidates = [candidate for candidate in gguf_files if _is_mmproj_gguf(candidate)]
    if not mmproj_candidates:
        return None
    return min(mmproj_candidates, key=lambda candidate: candidate.name.lower())


def _gguf_preference_rank(path: Path | str) -> int:
    name = Path(path).name.lower()
    for index, marker in enumerate(GGUF_PREFERENCE_MARKERS):
        if marker in name:
            return index
    return len(GGUF_PREFERENCE_MARKERS) + 1


def _resolve_cached_hf_launch_artifacts(
    repo_id: Optional[str],
    *,
    hf_file: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    snapshot_dir = _latest_hf_snapshot_dir(repo_id)
    if not snapshot_dir:
        return None

    gguf_files = _snapshot_gguf_files(snapshot_dir)
    if not gguf_files:
        return None

    main_candidates = [candidate for candidate in gguf_files if not _is_mmproj_gguf(candidate)]
    if not main_candidates:
        return None

    requested_file = str(hf_file or '').strip()
    selected_model: Optional[Path] = None
    if requested_file:
        exact_match = snapshot_dir / requested_file
        if exact_match.exists() and exact_match.is_file() and not _is_mmproj_gguf(exact_match):
            selected_model = exact_match
        else:
            for candidate in main_candidates:
                if candidate.name == requested_file:
                    selected_model = candidate
                    break

    if selected_model is None:
        selected_model = min(
            main_candidates,
            key=lambda candidate: (
                _gguf_preference_rank(candidate),
                candidate.name.lower(),
            ),
        )

    selected_mmproj = _select_mmproj_candidate(gguf_files)

    return {
        'model_path': str(selected_model),
        'hf_file': selected_model.name,
        'mmproj_path': str(selected_mmproj) if selected_mmproj else None,
    }


def _resolve_local_mmproj_for_model_path(
    model_name: Optional[str],
    model_path: Optional[str],
) -> Optional[str]:
    resolved_model_path = str(model_path or '').strip()
    if not resolved_model_path:
        return None
    path = Path(resolved_model_path).expanduser()
    try:
        if not path.exists() or not path.is_file() or _is_mmproj_gguf(path):
            return None
    except OSError:
        return None
    if not _looks_like_multimodal_chat_family(
        str(model_name or path.stem).strip() or path.stem,
        None,
        None,
        str(path),
    ):
        return None
    selected_mmproj = _select_mmproj_candidate(_snapshot_gguf_files(path.parent))
    return str(selected_mmproj) if selected_mmproj else None


def _missing_cached_hf_repo_message(repo_id: Optional[str], *, hf_file: Optional[str] = None) -> str:
    repo_token = str(repo_id or '').strip() or 'unknown HF repo'
    requested_file = str(hf_file or '').strip()
    if requested_file:
        return (
            f"llama.cpp repo '{repo_token}' cannot be started locally: "
            f"GGUF file '{requested_file}' is not present in the local Hugging Face cache. "
            "Pull the model first."
        )
    return (
        f"llama.cpp repo '{repo_token}' cannot be started locally: "
        "No local GGUF snapshot was found in the Hugging Face cache. "
        "Pull the model first."
    )


def _read_catalog_entries() -> list[dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return []
    try:
        payload = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _write_catalog_entries(entries: list[dict[str, Any]]) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def _catalog_source_key(source_kind: str, *, model_path: Optional[str] = None, hf_repo: Optional[str] = None, hf_file: Optional[str] = None) -> str:
    if source_kind == 'local_gguf':
        return f"local::{str(model_path or '').strip()}"
    return f"hf::{str(hf_repo or '').strip()}::{str(hf_file or '').strip()}"


def _upsert_catalog_entry(entry: dict[str, Any]) -> None:
    entries = _read_catalog_entries()
    source_key = str(entry.get('source_key') or '').strip()
    if not source_key:
        return
    kept = [item for item in entries if str(item.get('source_key') or '').strip() != source_key]
    kept.append(entry)
    _write_catalog_entries(kept)


def _remove_catalog_entries(
    *,
    source_kind: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
) -> list[dict[str, Any]]:
    target_source_kind = str(source_kind or '').strip()
    target_model_name = str(model_name or '').strip()
    target_model_path = str(model_path or '').strip()
    target_hf_repo = str(hf_repo or '').strip()
    target_hf_file = str(hf_file or '').strip()

    def _resolved_text_path(value: str) -> str:
        token = str(value or '').strip()
        if not token:
            return ''
        try:
            return str(Path(token).expanduser().resolve(strict=False))
        except OSError:
            return str(Path(token).expanduser())

    target_model_path_resolved = _resolved_text_path(target_model_path)

    def _matches(entry: dict[str, Any]) -> bool:
        if not isinstance(entry, dict):
            return False
        entry_source_kind = str(entry.get('source_kind') or '').strip()
        entry_model_name = str(entry.get('model_name') or entry.get('display_name') or '').strip()
        entry_model_path_resolved = _resolved_text_path(str(entry.get('model_path') or '').strip())
        entry_hf_repo = str(entry.get('hf_repo') or '').strip()
        entry_hf_file = str(entry.get('hf_file') or '').strip()
        if target_source_kind:
            if entry_source_kind != target_source_kind:
                return False
            if target_source_kind == 'local_gguf' and target_model_path_resolved:
                return entry_model_path_resolved == target_model_path_resolved
            if target_source_kind == 'hf_repo' and target_hf_repo:
                if entry_hf_repo != target_hf_repo:
                    return False
                return not target_hf_file or entry_hf_file == target_hf_file
        if target_model_path_resolved and entry_model_path_resolved == target_model_path_resolved:
            return True
        if target_hf_repo and entry_hf_repo == target_hf_repo:
            return not target_hf_file or entry_hf_file == target_hf_file
        if target_model_name:
            path_stem = Path(entry_model_path_resolved).stem if entry_model_path_resolved else ''
            return target_model_name in {entry_model_name, entry_hf_repo, path_stem}
        return False

    entries = _read_catalog_entries()
    removed = [entry for entry in entries if _matches(entry)]
    if removed:
        remaining = [entry for entry in entries if entry not in removed]
        _write_catalog_entries(remaining)
    return removed


def _looks_like_multimodal_chat_family(*tokens: Optional[str]) -> bool:
    combined = ' '.join(str(token or '').strip().lower() for token in tokens if str(token or '').strip())
    if not combined:
        return False
    return any(
        marker in combined
        for marker in (
            'gemma-4',
            'gemma4',
            'gemma-3',
            'gemma3',
            'qwen2.5-vl',
            'qwen2-vl',
            'qwen-vl',
            'pixtral',
            'smolvlm',
            'molmo',
            'paligemma',
            'idefics',
            'llava',
        )
    )


def _build_catalog_model_payload(
    model_name: str,
    *,
    source_kind: str,
    runnable: bool,
    disabled_reason: Optional[str],
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    size_gb: Optional[float] = None,
    startup_source: str,
    discovery_source: str,
) -> dict[str, Any]:
    provider_capabilities = [CAPABILITY_CHAT]
    inputs = ['text']
    outputs = ['text']
    backend_capabilities = ['completion']
    resolved_mmproj_path = str(mmproj_path or '').strip() or None
    requires_local_mmproj = source_kind == 'local_gguf' or startup_source == 'local_gguf'
    if _looks_like_multimodal_chat_family(model_name, hf_repo, hf_file, model_path) and (
        not requires_local_mmproj or resolved_mmproj_path
    ):
        provider_capabilities.append(CAPABILITY_VISION_ANALYSIS)
        inputs = ['text', 'image']
        backend_capabilities.append('vision')
    backend_metadata = {
        'source': discovery_source,
        'backend_package': 'llama_cpp',
        'backend_contract': 'llama.cpp.server',
        'startup_source': startup_source,
        'native_endpoint_paths': ['/v1/models', '/v1/chat/completions'],
        'request_model_strategy': 'model_bound_at_launch',
        'capabilities': backend_capabilities,
    }
    if hf_repo:
        backend_metadata['hf_repo'] = hf_repo
    if hf_file:
        backend_metadata['hf_file'] = hf_file
    metadata = {
        'model_path': model_path,
        'hf_repo': hf_repo,
        'hf_file': hf_file,
        'backend_package': 'llama_cpp',
        'backend_contract': 'llama.cpp.server',
        'provider_capabilities': provider_capabilities,
        'inputs': inputs,
        'outputs': outputs,
        'backend_metadata': backend_metadata,
    }
    payload = build_registry_metadata(model_name, 'llama_cpp', CAPABILITY_CHAT, metadata=metadata)
    payload.update(
        {
            'name': model_name,
            'model': model_name,
            'model_path': model_path,
            'hf_repo': hf_repo,
            'hf_file': hf_file,
            'mmproj_path': resolved_mmproj_path,
            'model_source': source_kind,
            'size_gb': size_gb,
            'runnable': runnable,
            'disabled_reason': disabled_reason,
        }
    )
    return payload


def _source_contract_metadata(
    model_name: str,
    *,
    source_kind: str,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
    mmproj_path: Optional[str] = None,
) -> dict[str, Any]:
    payload = _build_catalog_model_payload(
        model_name,
        source_kind=source_kind,
        runnable=True,
        disabled_reason=None,
        model_path=model_path,
        hf_repo=hf_repo,
        hf_file=hf_file,
        mmproj_path=mmproj_path,
        startup_source=source_kind,
        discovery_source='llama_cpp_source_contract',
    )
    backend_metadata = payload.get('backend_metadata') if isinstance(payload.get('backend_metadata'), dict) else {}
    return {
        'provider_capabilities': list(payload.get('provider_capabilities') or []),
        'inputs': list(payload.get('inputs') or []),
        'outputs': list(payload.get('outputs') or []),
        'backend_capabilities': list(backend_metadata.get('capabilities') or []),
    }


def _safe_instance_slug(value: str) -> str:
    token = value.replace('/', '__')
    token = SAFE_INSTANCE_RE.sub('-', token)
    return token.strip('-._') or 'llama-cpp'


def sanitize_instance_id(model_name: str, port: int) -> str:
    return f'{_safe_instance_slug(model_name)}-llama_cpp-{port}'


def _load_config_entries() -> list[dict[str, Any]]:
    return read_registry_entries(CONFIG_FILE)


def _write_config_entries(entries: list[dict[str, Any]]) -> None:
    write_registry_entries(entries, path=CONFIG_FILE, preserve_agents=True, sync_external=False)


def _prune_stale_llama_cpp_entries() -> None:
    entries = _load_config_entries()
    kept: list[dict[str, Any]] = []
    changed = False
    for entry in entries:
        if str(entry.get('backend') or '').strip().lower() != 'llama_cpp':
            kept.append(entry)
            continue
        port = entry.get('port')
        pid = entry.get('pid')
        port_alive = False
        if isinstance(port, int):
            port_alive = registry_is_port_listening(port)
        elif str(port or '').isdigit():
            port_alive = registry_is_port_listening(int(str(port)))
        pid_alive = bool(pid) and registry_pid_is_running(int(pid))
        if port_alive or pid_alive:
            kept.append(entry)
        else:
            changed = True
    if changed:
        _write_config_entries(kept)


def list_llama_cpp_instances() -> list[dict[str, Any]]:
    _prune_stale_llama_cpp_entries()
    return [
        entry
        for entry in _load_config_entries()
        if str(entry.get('backend') or '').strip().lower() == 'llama_cpp'
    ]


def describe_llama_cpp_runtime_probe() -> dict[str, Any]:
    server_bin = resolve_llama_server_bin()
    cli_bin = resolve_llama_cli_bin()
    hf_cli_bin = resolve_hf_cli_bin()
    server_detected = bool(server_bin)
    cli_detected = bool(cli_bin)
    runtime_state = 'runnable' if server_detected else 'missing'
    launch_defaults = _llama_cpp_launch_defaults(server_bin)
    issues: list[str] = []
    if server_detected and not cli_detected:
        issues.append('llama-cli not found; CLI-side cache inspection is unavailable.')
    if not server_detected:
        issues.append('llama-server not found on PATH or configured default paths.')
    return {
        'backend_id': 'llama_cpp',
        'family': 'llama.cpp',
        'variant': 'llama_cpp',
        'label': 'llama.cpp',
        'runtime_state': runtime_state,
        'operations': {
            'discover': True,
            'list_models': True,
            'pull_model': bool(cli_detected or hf_cli_bin or server_detected),
            'remove_model': True,
            'start_instance': server_detected,
            'stop_instance': True,
        },
        'detection': {
            'server_bin': server_bin,
            'server_detected': server_detected,
            'cli_bin': cli_bin,
            'cli_detected': cli_detected,
            'hf_cli_bin': hf_cli_bin,
            'hf_cli_detected': bool(hf_cli_bin),
            'models_roots': [str(path) for path in _common_models_roots()],
            'launch_defaults': launch_defaults,
        },
        'issues': issues,
    }


def _local_gguf_entries() -> list[Path]:
    entries: list[Path] = []
    for root in _common_models_roots():
        if not root.exists() or not root.is_dir():
            continue
        try:
            entries.extend(sorted(root.rglob('*.gguf')))
        except Exception:
            continue
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in entries:
        if _is_mmproj_gguf(candidate):
            continue
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def list_local_gguf_models() -> list[dict[str, Any]]:
    probe = describe_llama_cpp_runtime_probe()
    runnable = probe.get('runtime_state') == 'runnable'
    disabled_reason = None if runnable else '; '.join(probe.get('issues') or []) or 'llama.cpp runtime unavailable.'
    items: list[dict[str, Any]] = []
    for path in _local_gguf_entries():
        model_name = path.stem
        mmproj_path = _resolve_local_mmproj_for_model_path(model_name, str(path))
        payload = _build_catalog_model_payload(
            model_name,
            source_kind='local_gguf',
            runnable=runnable,
            disabled_reason=disabled_reason,
            model_path=str(path),
            mmproj_path=mmproj_path,
            size_gb=_estimate_llama_cpp_source_size_gb(model_path=str(path)),
            startup_source='local_gguf',
            discovery_source='llama_cpp_local_scan',
        )
        items.append(payload)
    return items


def list_llama_cpp_catalog_models() -> list[dict[str, Any]]:
    probe = describe_llama_cpp_runtime_probe()
    runnable = probe.get('runtime_state') == 'runnable'
    disabled_reason = None if runnable else '; '.join(probe.get('issues') or []) or 'llama.cpp runtime unavailable.'
    local_paths = {str(path.resolve()) for path in _local_gguf_entries() if path.exists()}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in _read_catalog_entries():
        source_kind = str(entry.get('source_kind') or '').strip()
        model_name = str(entry.get('model_name') or entry.get('display_name') or entry.get('hf_repo') or entry.get('model_path') or '').strip()
        model_path = str(entry.get('model_path') or '').strip() or None
        hf_repo = str(entry.get('hf_repo') or '').strip() or None
        hf_file = str(entry.get('hf_file') or '').strip() or None
        effective_mmproj_path: Optional[str] = None
        if not model_name or source_kind not in {'local_gguf', 'hf_repo'}:
            continue
        if source_kind == 'local_gguf' and model_path:
            path = Path(model_path).expanduser()
            if not path.exists() or not path.is_file():
                continue
            resolved = str(path.resolve())
            effective_mmproj_path = _resolve_local_mmproj_for_model_path(model_name, resolved)
            if resolved in local_paths:
                continue
            dedupe_key = f'local::{resolved}'
        else:
            dedupe_key = f'hf::{hf_repo or model_name}::{hf_file or ""}'
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        item_runnable = runnable
        item_disabled_reason = disabled_reason
        effective_model_path = model_path
        effective_hf_file = hf_file
        startup_source = source_kind
        if source_kind == 'hf_repo':
            cached_artifacts = _resolve_cached_hf_launch_artifacts(hf_repo, hf_file=hf_file)
            if cached_artifacts:
                effective_model_path = str(cached_artifacts.get('model_path') or '').strip() or None
                effective_hf_file = str(cached_artifacts.get('hf_file') or '').strip() or effective_hf_file
                effective_mmproj_path = str(cached_artifacts.get('mmproj_path') or '').strip() or None
                startup_source = 'local_gguf'
            else:
                item_runnable = False
                cache_reason = _missing_cached_hf_repo_message(hf_repo, hf_file=hf_file)
                item_disabled_reason = f'{disabled_reason}; {cache_reason}' if disabled_reason else cache_reason
        items.append(
            _build_catalog_model_payload(
                model_name,
                source_kind=source_kind,
                runnable=item_runnable,
                disabled_reason=item_disabled_reason,
                model_path=effective_model_path,
                hf_repo=hf_repo,
                hf_file=effective_hf_file,
                mmproj_path=effective_mmproj_path,
                size_gb=(
                    entry.get('size_gb')
                    if isinstance(entry.get('size_gb'), (int, float))
                    else _estimate_llama_cpp_source_size_gb(
                        model_path=effective_model_path,
                        hf_repo=hf_repo,
                        hf_file=effective_hf_file,
                    )
                ),
                startup_source=startup_source,
                discovery_source='llama_cpp_catalog_state',
            )
        )
    return items


def list_cached_hf_gguf_models() -> list[dict[str, Any]]:
    probe = describe_llama_cpp_runtime_probe()
    runnable = probe.get('runtime_state') == 'runnable'
    disabled_reason = None if runnable else '; '.join(probe.get('issues') or []) or 'llama.cpp runtime unavailable.'
    local_paths = {str(path.resolve()) for path in _local_gguf_entries() if path.exists()}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for repo_dir in _cached_hf_repo_dirs():
        repo_id = _hf_repo_id_from_cache_dir(repo_dir)
        if not repo_id:
            continue
        cached_artifacts = _resolve_cached_hf_launch_artifacts(repo_id)
        if not cached_artifacts:
            continue
        model_path = str(cached_artifacts.get('model_path') or '').strip() or None
        if not model_path:
            continue
        try:
            resolved_model_path = str(Path(model_path).expanduser().resolve())
        except OSError:
            resolved_model_path = str(Path(model_path).expanduser())
        if resolved_model_path in local_paths:
            continue
        effective_hf_file = str(cached_artifacts.get('hf_file') or '').strip() or None
        dedupe_key = f'hf::{repo_id}::{effective_hf_file or ""}'
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        items.append(
            _build_catalog_model_payload(
                repo_id,
                source_kind='hf_repo',
                runnable=runnable,
                disabled_reason=disabled_reason,
                model_path=model_path,
                hf_repo=repo_id,
                hf_file=effective_hf_file,
                mmproj_path=str(cached_artifacts.get('mmproj_path') or '').strip() or None,
                size_gb=_estimate_llama_cpp_source_size_gb(
                    model_path=model_path,
                    hf_repo=repo_id,
                    hf_file=effective_hf_file,
                ),
                startup_source='local_gguf',
                discovery_source='llama_cpp_hf_cache_scan',
            )
        )
    return items


def list_available_llama_cpp_models() -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list_local_gguf_models() + list_cached_hf_gguf_models() + list_llama_cpp_catalog_models():
        source_kind = str(item.get('model_source') or '').strip()
        model_path = str(item.get('model_path') or '').strip()
        hf_repo = str(item.get('hf_repo') or '').strip()
        hf_file = str(item.get('hf_file') or '').strip()
        if source_kind == 'local_gguf' and model_path:
            dedupe_key = f'local::{model_path}'
        else:
            dedupe_key = f'hf::{hf_repo or item.get("model") or ""}::{hf_file}'
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append(item)
    return merged


def _looks_like_hf_repo(value: str) -> bool:
    token = str(value or '').strip()
    if not token or os.path.isabs(token):
        return False
    if token.endswith('.gguf'):
        return False
    return '/' in token


def _pull_hf_repo_with_hf_cli(hf_cli_bin: str, repo_id: str, *, hf_file: Optional[str] = None) -> tuple[bool, str]:
    cmd = [hf_cli_bin, 'download', repo_id]
    if hf_file:
        cmd.extend(['--include', hf_file])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=sanitized_child_process_env(),
    )
    if result.returncode == 0:
        message = result.stdout.strip() or f"llama.cpp HF repo {repo_id} was loaded into the cache successfully."
        return True, message
    error = result.stderr.strip() or result.stdout.strip() or f'Download failed for {repo_id}.'
    return False, error


def _pull_hf_repo_with_llama_cli(llama_cli_bin: str, repo_id: str, *, hf_file: Optional[str] = None) -> tuple[bool, str]:
    cmd = [
        llama_cli_bin,
        '--hf-repo',
        repo_id,
        '--no-warmup',
        '--no-conversation',
        '--ctx-size',
        '32',
        '--n-predict',
        '1',
        '--prompt',
        '.',
    ]
    if hf_file:
        cmd.extend(['--hf-file', hf_file])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=sanitized_child_process_env(),
    )
    if result.returncode == 0:
        message = result.stdout.strip() or f"llama.cpp HF repo {repo_id} was prepared successfully."
        return True, message
    error = result.stderr.strip() or result.stdout.strip() or f'Download failed for {repo_id}.'
    return False, error


def pull_llama_cpp_model(model_name: str, *, hf_file: Optional[str] = None) -> tuple[bool, str]:
    source = _resolve_model_source(
        model_name,
        None,
        hf_file=hf_file,
        prefer_cached_local=False,
        allow_remote_hf_repo=True,
    )
    source_kind = str(source.get('source_kind') or '').strip()
    display_name = str(source.get('display_name') or model_name).strip()
    model_path = str(source.get('model_path') or '').strip() or None
    hf_repo = str(source.get('hf_repo') or '').strip() or None
    resolved_hf_file = str(source.get('hf_file') or '').strip() or None

    if source_kind == 'local_gguf':
        source_key = _catalog_source_key('local_gguf', model_path=model_path)
        _upsert_catalog_entry(
            {
                'source_key': source_key,
                'source_kind': 'local_gguf',
                'display_name': display_name,
                'model_name': display_name,
                'model_path': model_path,
                'hf_repo': None,
                'hf_file': None,
                'size_gb': _estimate_llama_cpp_source_size_gb(model_path=model_path),
                'added_ts': int(time.time()),
            }
        )
        return True, f'Local GGUF registered for llama.cpp: {model_path}'

    if not hf_repo:
        return False, 'No llama.cpp Hugging Face repo was provided.'

    hf_cli_bin = resolve_hf_cli_bin()
    success = False
    message = ''
    if hf_cli_bin and ':' not in hf_repo and not resolved_hf_file:
        success, message = _pull_hf_repo_with_hf_cli(hf_cli_bin, hf_repo)
    elif hf_cli_bin:
        success, message = _pull_hf_repo_with_hf_cli(hf_cli_bin, hf_repo, hf_file=resolved_hf_file)
    else:
        return False, "No non-interactive 'hf' CLI was found; install or link 'hf' for llama.cpp pulls."

    if not success:
        return False, message

    source_key = _catalog_source_key('hf_repo', hf_repo=hf_repo, hf_file=resolved_hf_file)
    _upsert_catalog_entry(
        {
            'source_key': source_key,
            'source_kind': 'hf_repo',
            'display_name': display_name,
            'model_name': display_name,
            'model_path': None,
            'hf_repo': hf_repo,
            'hf_file': resolved_hf_file,
            'size_gb': _estimate_llama_cpp_source_size_gb(hf_repo=hf_repo, hf_file=resolved_hf_file),
            'added_ts': int(time.time()),
        }
    )
    return True, message


def _resolve_model_source(
    model_name: str,
    model_path: Optional[str],
    hf_file: Optional[str] = None,
    *,
    prefer_cached_local: bool = True,
    allow_remote_hf_repo: bool = False,
) -> dict[str, Any]:
    explicit_path = str(model_path or '').strip()
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists() and path.is_file():
            display_name = str(model_name or '').strip() or path.stem
            return {
                'source_kind': 'local_gguf',
                'display_name': display_name,
                'model_path': str(path),
                'hf_repo': None,
                'hf_file': None,
                'mmproj_path': _resolve_local_mmproj_for_model_path(display_name, str(path)),
            }
        if _looks_like_hf_repo(explicit_path):
            display_name = str(model_name or '').strip() or explicit_path
            cached_artifacts = _resolve_cached_hf_launch_artifacts(explicit_path, hf_file=hf_file) if prefer_cached_local else None
            if cached_artifacts:
                return {
                    'source_kind': 'local_gguf',
                    'display_name': display_name,
                    'model_path': str(cached_artifacts['model_path']),
                    'hf_repo': explicit_path,
                    'hf_file': str(cached_artifacts.get('hf_file') or '').strip() or None,
                    'mmproj_path': str(cached_artifacts.get('mmproj_path') or '').strip() or None,
                }
            if not allow_remote_hf_repo:
                raise ValueError(_missing_cached_hf_repo_message(explicit_path, hf_file=hf_file))
            return {
                'source_kind': 'hf_repo',
                'display_name': display_name,
                'model_path': None,
                'hf_repo': explicit_path,
                'hf_file': str(hf_file or '').strip() or None,
                'mmproj_path': None,
            }
        raise ValueError(f"GGUF model path was not found: {explicit_path}")

    model_token = str(model_name or '').strip()
    if not model_token:
        raise ValueError('No llama.cpp model name or model path was provided.')
    path = Path(model_token).expanduser()
    if path.exists() and path.is_file():
        return {
            'source_kind': 'local_gguf',
            'display_name': path.stem,
            'model_path': str(path),
            'hf_repo': None,
            'hf_file': None,
            'mmproj_path': _resolve_local_mmproj_for_model_path(path.stem, str(path)),
        }
    if _looks_like_hf_repo(model_token):
        cached_artifacts = _resolve_cached_hf_launch_artifacts(model_token, hf_file=hf_file) if prefer_cached_local else None
        if cached_artifacts:
            return {
                'source_kind': 'local_gguf',
                'display_name': model_token,
                'model_path': str(cached_artifacts['model_path']),
                'hf_repo': model_token,
                'hf_file': str(cached_artifacts.get('hf_file') or '').strip() or None,
                'mmproj_path': str(cached_artifacts.get('mmproj_path') or '').strip() or None,
            }
        if not allow_remote_hf_repo:
            raise ValueError(_missing_cached_hf_repo_message(model_token, hf_file=hf_file))
        return {
            'source_kind': 'hf_repo',
            'display_name': model_token,
            'model_path': None,
            'hf_repo': model_token,
            'hf_file': str(hf_file or '').strip() or None,
            'mmproj_path': None,
        }
    raise ValueError(
        "For llama.cpp, provide either a local GGUF path via 'model_path' "
        "or a Hugging Face repo such as 'org/model-GGUF'."
    )


def _next_free_port(preferred_port: Optional[int] = None) -> int:
    used_ports = {int(entry.get('port')) for entry in _load_config_entries() if str(entry.get('port') or '').isdigit()}
    if preferred_port is not None:
        if preferred_port in used_ports or registry_is_port_listening(preferred_port):
            raise RuntimeError(f'Bevorzugter Port {preferred_port} ist bereits belegt.')
        return preferred_port
    for port in range(LLAMA_CPP_START_PORT, LLAMA_CPP_PORT_MAX + 1):
        if port in used_ports:
            continue
        if registry_is_port_listening(port):
            continue
        return port
    raise RuntimeError('No free llama.cpp port is available.')


def _wait_for_server_ready(port: int, *, timeout_sec: int = LLAMA_CPP_START_TIMEOUT_SEC) -> bool:
    deadline = time.time() + max(5, int(timeout_sec))
    last_exception: Optional[Exception] = None
    while time.time() < deadline:
        try:
            response = requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5)
            if response.ok:
                payload = response.json()
                if isinstance(payload, dict):
                    return True
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
        time.sleep(1.0)
    if last_exception:
        return False
    return registry_is_port_listening(port)


def build_instance_record(
    model_name: str,
    *,
    source_kind: str,
    port: int,
    log_path: Path,
    launch_defaults: Optional[dict[str, Any]] = None,
    capability: Optional[str] = None,
    pid: Optional[int] = None,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    request_model: Optional[str] = None,
) -> dict[str, Any]:
    resolved_capability = normalize_capability(capability) or CAPABILITY_CHAT
    resolved_mmproj_path = str(mmproj_path or '').strip() or None
    if not resolved_mmproj_path and source_kind == 'local_gguf':
        resolved_mmproj_path = _resolve_local_mmproj_for_model_path(model_name, model_path)
    source_contract = _source_contract_metadata(
        model_name,
        source_kind=source_kind,
        model_path=model_path,
        hf_repo=hf_repo,
        hf_file=hf_file,
        mmproj_path=resolved_mmproj_path,
    )
    provider_capabilities = [
        token
        for token in source_contract.get('provider_capabilities', [])
        if normalize_capability(token)
    ] or [resolved_capability]
    if resolved_capability not in provider_capabilities:
        provider_capabilities.insert(0, resolved_capability)
    native_endpoint_paths = ['/v1/models', '/v1/chat/completions']
    if resolved_capability == CAPABILITY_EMBEDDING:
        native_endpoint_paths.append('/v1/embeddings')
    normalized_launch_defaults = launch_defaults if isinstance(launch_defaults, dict) else {}
    metadata = {
        'backend_package': 'llama_cpp',
        'backend_contract': 'llama.cpp.server',
        'provider_capabilities': provider_capabilities,
        'inputs': source_contract.get('inputs') or ['text'],
        'outputs': source_contract.get('outputs') or ['text'],
        'backend_metadata': {
            'source': 'llama_cpp_runtime_contract',
            'backend_package': 'llama_cpp',
            'backend_contract': 'llama.cpp.server',
            'startup_source': source_kind,
            'native_endpoint_paths': native_endpoint_paths,
            'request_model_strategy': 'model_bound_at_launch',
            'supports_hf_repo': False,
            'supports_local_gguf': True,
            'launch_defaults': normalized_launch_defaults,
            'capabilities': source_contract.get('backend_capabilities') or ['completion'],
            **({'hf_repo': hf_repo} if hf_repo else {}),
            **({'hf_file': hf_file} if hf_file else {}),
            **({'mmproj_path': resolved_mmproj_path} if resolved_mmproj_path else {}),
        },
    }
    registry_metadata = build_registry_metadata(model_name, 'llama_cpp', resolved_capability, metadata=metadata)
    return {
        'instance_id': sanitize_instance_id(model_name, port),
        'model': model_name,
        **registry_metadata,
        'backend_package': 'llama_cpp',
        'backend_contract': 'llama.cpp.server',
        'provider_capabilities': provider_capabilities,
        'port': port,
        'pid': pid,
        'log': str(log_path),
        'request_model': request_model or model_name,
        'model_path': model_path,
        'hf_repo': hf_repo,
        'hf_file': hf_file,
        'mmproj_path': resolved_mmproj_path,
        'ts': int(time.time()),
    }


def _register_instance(instance: dict[str, Any]) -> None:
    entries = _load_config_entries()
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get('backend') or '').strip().lower() == 'llama_cpp':
            if entry.get('instance_id') == instance.get('instance_id'):
                continue
            if entry.get('port') == instance.get('port'):
                continue
        filtered.append(entry)
    filtered.append(instance)
    _write_config_entries(filtered)


def _remove_instance(instance_id: str) -> None:
    entries = _load_config_entries()
    filtered = [entry for entry in entries if entry.get('instance_id') != instance_id]
    if len(filtered) != len(entries):
        _write_config_entries(filtered)


def start_llama_cpp_instance(
    model_name: str,
    model_path: Optional[str] = None,
    preferred_port: Optional[int] = None,
    capability: Optional[str] = None,
    *,
    hf_file: Optional[str] = None,
    register_instance: bool = True,
    start_source: Optional[str] = None,
) -> dict[str, Any]:
    normalized_start_source = validate_start_source(
        start_source,
        context='llama_cpp_start_model',
    )
    server_bin = resolve_llama_server_bin()
    if not server_bin:
        raise RuntimeError('llama-server was not found.')

    _prune_stale_llama_cpp_entries()
    source = _resolve_model_source(model_name, model_path, hf_file=hf_file)
    port = _next_free_port(preferred_port)
    display_name = str(source['display_name'])
    request_model = display_name
    instance_id = sanitize_instance_id(display_name, port)
    log_name = f"llama_cpp_server_{safe_log_fragment(display_name)}_{port}.log"
    log_path = LOG_DIR / log_name
    resolved_capability = normalize_capability(capability) or CAPABILITY_CHAT
    launch_defaults = _llama_cpp_launch_defaults(
        server_bin,
        model_name=display_name,
        model_path=source.get('model_path'),
        hf_repo=source.get('hf_repo'),
        hf_file=source.get('hf_file'),
    )

    cmd = [
        server_bin,
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        '--alias',
        request_model,
        '--no-webui',
        '--jinja',
    ]
    supported_flags = launch_defaults.get('supported_flags') if isinstance(launch_defaults.get('supported_flags'), dict) else {}
    if source['source_kind'] == 'local_gguf':
        cmd.extend(['-m', str(source['model_path'])])
        mmproj_path = str(source.get('mmproj_path') or '').strip()
        if mmproj_path and supported_flags.get('mmproj'):
            cmd.extend(['--mmproj', mmproj_path])
    else:
        cmd.extend(['-hf', str(source['hf_repo'])])
        if source.get('hf_file'):
            cmd.extend(['--hf-file', str(source['hf_file'])])
    if resolved_capability == CAPABILITY_EMBEDDING:
        cmd.append('--embedding')
    ctx_size = launch_defaults.get('ctx_size')
    if supported_flags.get('ctx_size') and isinstance(ctx_size, int) and ctx_size > 0:
        cmd.extend(['--ctx-size', str(ctx_size)])
    batch_size = launch_defaults.get('batch_size')
    if supported_flags.get('batch_size') and isinstance(batch_size, int) and batch_size > 0:
        cmd.extend(['--batch-size', str(batch_size)])
    ubatch_size = launch_defaults.get('ubatch_size')
    if supported_flags.get('ubatch_size') and isinstance(ubatch_size, int) and ubatch_size > 0:
        cmd.extend(['--ubatch-size', str(ubatch_size)])
    if launch_defaults.get('prompt_cache') and supported_flags.get('cache_prompt'):
        cmd.append('--cache-prompt')
    if launch_defaults.get('kv_offload') and supported_flags.get('kv_offload'):
        cmd.append('--kv-offload')
    flash_attention = str(launch_defaults.get('flash_attention') or '').strip().lower()
    if flash_attention in {'on', 'off', 'auto'} and supported_flags.get('flash_attention'):
        cmd.extend(['--flash-attn', flash_attention])

    prepare_clean_runtime_log(
        log_path,
        metadata={
            'backend': 'llama_cpp',
            'instance_id': instance_id,
            'port': port,
            'model': display_name,
        },
    )
    with log_path.open('wb') as handle:
        handle.write(f'\n\n---- LAUNCH {time.strftime("%Y-%m-%d %H:%M:%S")} ----\n'.encode('utf-8'))
        handle.write((' '.join(cmd) + '\n').encode('utf-8'))
        process = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
            env=sanitized_child_process_env(),
        )

    if not _wait_for_server_ready(port):
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                process.kill()
            except Exception:
                pass
        raise RuntimeError(f"llama.cpp model '{display_name}' did not open port {port}. See {log_path}.")

    instance = build_instance_record(
        display_name,
        source_kind=str(source['source_kind']),
        port=port,
        log_path=log_path,
        launch_defaults=launch_defaults,
        capability=resolved_capability,
        pid=process.pid,
        model_path=source.get('model_path'),
        hf_repo=source.get('hf_repo'),
        hf_file=source.get('hf_file'),
        mmproj_path=source.get('mmproj_path'),
        request_model=request_model,
    )
    instance = attach_start_audit(
        instance,
        start_source=normalized_start_source,
        context='llama_cpp_start_model',
        extra={'backend': 'llama_cpp', 'capability': resolved_capability},
    )
    if register_instance:
        _register_instance(instance)
    return instance


def stop_llama_cpp_instance(instance_id: str) -> tuple[bool, Optional[dict[str, Any]]]:
    entries = _load_config_entries()
    target = next(
        (
            entry
            for entry in entries
            if entry.get('instance_id') == instance_id and str(entry.get('backend') or '').strip().lower() == 'llama_cpp'
        ),
        None,
    )
    if not target:
        return False, None

    port = target.get('port')
    pid = target.get('pid')
    success = True
    if pid and registry_pid_is_running(int(pid)):
        try:
            os.kill(int(pid), 15)
        except ProcessLookupError:
            pass
        except Exception:
            success = False
    if port and str(port).isdigit():
        success = kill_processes_on_port(int(port)) and success
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not registry_is_port_listening(int(port)):
                break
            time.sleep(0.2)
        success = success and (not registry_is_port_listening(int(port)))

    if success:
        _remove_instance(instance_id)
        remove_instance_status(instance_id)
    return success, target


def _matching_llama_cpp_instances(
    *,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
) -> list[dict[str, Any]]:
    target_model_name = str(model_name or '').strip()
    target_hf_repo = str(hf_repo or '').strip()
    target_model_path = str(model_path or '').strip()
    try:
        target_model_path_resolved = str(Path(target_model_path).expanduser().resolve(strict=False)) if target_model_path else ''
    except OSError:
        target_model_path_resolved = str(Path(target_model_path).expanduser()) if target_model_path else ''
    matches: list[dict[str, Any]] = []
    for entry in list_llama_cpp_instances():
        if not isinstance(entry, dict):
            continue
        entry_model_name = str(entry.get('model') or '').strip()
        entry_hf_repo = str(entry.get('hf_repo') or '').strip()
        entry_model_path = str(entry.get('model_path') or '').strip()
        try:
            entry_model_path_resolved = str(Path(entry_model_path).expanduser().resolve(strict=False)) if entry_model_path else ''
        except OSError:
            entry_model_path_resolved = str(Path(entry_model_path).expanduser()) if entry_model_path else ''
        if target_hf_repo and entry_hf_repo == target_hf_repo:
            matches.append(entry)
            continue
        if target_model_path_resolved and entry_model_path_resolved == target_model_path_resolved:
            matches.append(entry)
            continue
        if target_model_name and target_model_name == entry_model_name:
            matches.append(entry)
    return matches


def _remove_local_gguf_file(model_name: str, model_path: str) -> list[str]:
    removed_paths: list[str] = []
    path = Path(str(model_path or '').strip()).expanduser()
    if not str(path).strip():
        return removed_paths
    mmproj_path = str(_resolve_local_mmproj_for_model_path(model_name, str(path)) or '').strip()
    if path.exists() and path.is_file():
        if path.suffix.lower() != '.gguf':
            raise ValueError(f'llama.cpp local removal expects a .gguf file, got: {path}')
        path.unlink()
        removed_paths.append(str(path))
    if mmproj_path:
        mmproj = Path(mmproj_path).expanduser()
        if mmproj.exists() and mmproj.is_file():
            mmproj.unlink()
            removed_paths.append(str(mmproj))
    return removed_paths


def _remove_hf_repo_cache(hf_repo: str) -> bool:
    repo_dir = _hf_repo_cache_dir(hf_repo)
    if not repo_dir or not repo_dir.exists():
        return False
    shutil.rmtree(repo_dir)
    return True


def remove_llama_cpp_model(
    model_name: str,
    *,
    model_source: Optional[str] = None,
    model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_file: Optional[str] = None,
) -> tuple[bool, str]:
    target_model_name = str(model_name or '').strip()
    target_source = str(model_source or '').strip() or None
    target_model_path = str(model_path or '').strip() or None
    target_hf_repo = str(hf_repo or '').strip() or None
    target_hf_file = str(hf_file or '').strip() or None

    if not target_hf_repo and target_source == 'hf_repo' and _looks_like_hf_repo(target_model_name):
        target_hf_repo = target_model_name
    if not target_model_path and target_source == 'local_gguf':
        candidate_paths = [
            str(item.get('model_path') or '').strip()
            for item in list_available_llama_cpp_models()
            if str(item.get('model_source') or '').strip() == 'local_gguf'
            and str(item.get('name') or item.get('model') or '').strip() == target_model_name
            and str(item.get('model_path') or '').strip()
        ]
        if len(candidate_paths) == 1:
            target_model_path = candidate_paths[0]

    matching_instances = _matching_llama_cpp_instances(
        model_name=target_model_name,
        model_path=target_model_path,
        hf_repo=target_hf_repo,
    )
    for entry in matching_instances:
        instance_id = str(entry.get('instance_id') or '').strip()
        if not instance_id:
            continue
        success, _record = stop_llama_cpp_instance(instance_id)
        if not success:
            return False, f'llama.cpp instance {instance_id} could not be stopped before removal.'

    cache_removed = False
    local_removed_paths: list[str] = []
    if target_source == 'hf_repo' or target_hf_repo:
        resolved_hf_repo = str(target_hf_repo or '').strip() or target_model_name
        cache_removed = _remove_hf_repo_cache(resolved_hf_repo)
    if target_source == 'local_gguf' or target_model_path:
        if target_model_path:
            local_removed_paths = _remove_local_gguf_file(target_model_name, target_model_path)

    removed_catalog_entries = _remove_catalog_entries(
        source_kind=target_source,
        model_name=target_model_name,
        model_path=target_model_path,
        hf_repo=target_hf_repo,
        hf_file=target_hf_file,
    )

    if not cache_removed and not local_removed_paths and not removed_catalog_entries:
        return False, f"llama.cpp source '{target_model_name}' was not found."

    details: list[str] = []
    if cache_removed:
        details.append('removed HF cache')
    if local_removed_paths:
        details.append(
            'removed local file'
            if len(local_removed_paths) == 1
            else f'removed {len(local_removed_paths)} local files'
        )
    if removed_catalog_entries:
        details.append(
            'removed catalog entry'
            if len(removed_catalog_entries) == 1
            else f'removed {len(removed_catalog_entries)} catalog entries'
        )
    if matching_instances:
        details.append(
            'stopped matching instance'
            if len(matching_instances) == 1
            else f'stopped {len(matching_instances)} matching instances'
        )
    if not details:
        details.append('cleaned stale source')
    return True, f"Removed llama.cpp source '{target_model_name}': {', '.join(details)}."
