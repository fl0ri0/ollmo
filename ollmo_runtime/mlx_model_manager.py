#!/usr/bin/env python3
import os, sys, re, shutil, subprocess, threading, time
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    build_registry_metadata,
    infer_capability,
    normalize_capability,
)
from helpers.ocr_modes import get_ocr_model_family
from helpers.tts_model_metadata import read_snapshot_model_metadata
from ollmo_core.start_policy import attach_start_audit, validate_start_source
from ollmo_core.status import remove_instance_status
from ollmo_runtime.child_process_env import sanitized_child_process_env
from ollmo_runtime.ollama_model_manager import kill_processes_on_port  # type: ignore
from ollmo_runtime.registry import read_registry_entries, write_registry_entries
from ollmo_runtime.runtime_log_hygiene import prepare_clean_runtime_log

ENV_MLX_PYTHON = os.environ.get("MLX_PYTHON")
ENV_MLX_AUDIO_PYTHON = os.environ.get("MLX_AUDIO_PYTHON")
DEFAULT_MLX_PYTHON = Path("/opt/mlx/venv/bin/python")
DEFAULT_MLX_AUDIO_PYTHON = Path("/opt/mlx-audio/venv/bin/python")
_RESOLVED_MLX_PYTHON = None
CONFIG_FILE = Path("model_ports.json")

START_PORT = int(os.environ.get("MLX_START_PORT", "11501"))
PORT_MAX   = int(os.environ.get("MLX_PORT_MAX", "11550"))
CACHE_DIR  = Path(os.environ.get("HF_HOME", Path.home()/".cache"/"huggingface")) / "hub"
LOG_DIR    = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEECH_SERVER_SCRIPT = REPO_ROOT / "scripts" / "mlx_whisper_server.py"
MLX_VLM_MODULE = "mlx_vlm.server"
MLX_AUDIO_MODULE = "mlx_audio.server"
MLX_POST_START_MONITOR_SEC = float(os.environ.get("MLX_POST_START_MONITOR_SEC", "120"))
MLX_POST_START_MONITOR_INTERVAL_SEC = float(os.environ.get("MLX_POST_START_MONITOR_INTERVAL_SEC", "2"))

SAFE_INSTANCE_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
HF_REPO_PREFIX = "model/"
CAPABILITY_BADGES = {
    CAPABILITY_CHAT: "💬 Chat",
    CAPABILITY_VISION_ANALYSIS: "👁 OCR/Vision",
    CAPABILITY_TEXT_TO_SPEECH: "🔊 TTS",
    CAPABILITY_SPEECH_TO_TEXT: "🎙 STT",
}
MLX_PACKAGE_METADATA_SOURCE = "mlx_package_contract"


def _env_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_text(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    return raw


def _mlx_lm_launch_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "prefill_step_size": _env_optional_int("OLLMO_MLX_LM_PREFILL_STEP_SIZE", 2048),
        "prompt_cache_size": _env_optional_int("OLLMO_MLX_LM_PROMPT_CACHE_SIZE", 10),
        "prompt_cache_bytes": _env_optional_text("OLLMO_MLX_LM_PROMPT_CACHE_BYTES"),
    }
    return {key: value for key, value in defaults.items() if value not in (None, "")}


def _mlx_vlm_requires_conservative_defaults(
    repo_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    model_type = str((metadata or {}).get("snapshot_model_type") or "").strip().lower()
    repo_name = str(repo_id or "").strip().lower()
    if model_type == "gemma4":
        return True
    return "gemma-4" in repo_name


def _mlx_vlm_launch_defaults(
    repo_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    conservative_defaults = _mlx_vlm_requires_conservative_defaults(repo_id, metadata)
    default_scheme = None if conservative_defaults else "uniform"
    scheme = str(os.environ.get("OLLMO_MLX_VLM_KV_QUANT_SCHEME", default_scheme or "") or "").strip().lower()
    if scheme not in {"", "uniform", "turboquant"}:
        scheme = default_scheme or ""
    defaults: dict[str, Any] = {
        "prefill_step_size": _env_optional_int(
            "OLLMO_MLX_VLM_PREFILL_STEP_SIZE",
            None if conservative_defaults else 2048,
        ),
        "kv_bits": _env_optional_float("OLLMO_MLX_VLM_KV_BITS"),
        "kv_quant_scheme": scheme or None,
        "kv_group_size": _env_optional_int(
            "OLLMO_MLX_VLM_KV_GROUP_SIZE",
            None if conservative_defaults else 64,
        ),
        "max_kv_size": _env_optional_int("OLLMO_MLX_VLM_MAX_KV_SIZE"),
        "quantized_kv_start": _env_optional_int(
            "OLLMO_MLX_VLM_QUANTIZED_KV_START",
            None if conservative_defaults else 5000,
        ),
    }
    return {key: value for key, value in defaults.items() if value not in (None, "")}


def _mlx_vlm_provider_capabilities(
    repo_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> list[str]:
    if get_ocr_model_family(repo_id):
        return [CAPABILITY_VISION_ANALYSIS]
    return [CAPABILITY_CHAT, CAPABILITY_VISION_ANALYSIS]


def _version_parts(version: Optional[str]) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(version or ""))
    return tuple(int(part) for part in parts[:3])


def _version_at_least(version: Optional[str], minimum: tuple[int, ...]) -> bool:
    current = _version_parts(version)
    if not current:
        return False
    padded_current = current + (0,) * max(0, len(minimum) - len(current))
    padded_minimum = minimum + (0,) * max(0, len(current) - len(minimum))
    return padded_current >= padded_minimum


def _version_before(version: Optional[str], minimum: tuple[int, ...]) -> bool:
    current = _version_parts(version)
    if not current:
        return False
    padded_current = current + (0,) * max(0, len(minimum) - len(current))
    padded_minimum = minimum + (0,) * max(0, len(current) - len(minimum))
    return padded_current < padded_minimum


def _mlx_distribution_name(server_kind: str) -> str:
    if server_kind == "mlx_vlm":
        return "mlx-vlm"
    if server_kind == "mlx_audio":
        return "mlx-audio"
    if server_kind == "mlx_whisper":
        return "mlx-whisper"
    return "mlx-lm"


def _resolved_mlx_capability(repo_id: str, capability: Optional[str], metadata: Optional[dict] = None) -> str:
    resolved = normalize_capability(capability)
    if resolved:
        return resolved
    return infer_capability(repo_id, "mlx", metadata=metadata)

def _mlx_contract_details(repo_id: str, capability: Optional[str], metadata: Optional[dict] = None) -> dict:
    resolved = _resolved_mlx_capability(repo_id, capability, metadata=metadata)
    server_kind = _server_kind_for_capability(resolved, repo_id=repo_id, metadata=metadata)

    if server_kind == "mlx_whisper":
        return {
            "backend_package": "mlx_whisper_shim",
            "backend_contract": "ollmo.scripts.mlx_whisper_server",
            "provider_capabilities": [CAPABILITY_SPEECH_TO_TEXT],
            "backend_metadata": {
                "source": MLX_PACKAGE_METADATA_SOURCE,
                "backend_package": "mlx_whisper_shim",
                "backend_contract": "ollmo.scripts.mlx_whisper_server",
                "package_label": "Local MLX Whisper shim",
                "capabilities": [CAPABILITY_SPEECH_TO_TEXT],
                "server_kind": server_kind,
                "instance_capabilities": [CAPABILITY_SPEECH_TO_TEXT],
                "package_capabilities": [CAPABILITY_SPEECH_TO_TEXT],
                "shim_kind": "local_http_compatibility_shim",
                "native_endpoint_paths": ["/healthz", "/v1/audio/transcriptions", "/api/transcribe"],
                "runtime_constraints": ["local_shim", "model_path_required_at_launch"],
            },
        }

    if server_kind == "mlx_audio":
        return {
            "backend_package": "mlx_audio",
            "backend_contract": "mlx_audio.server",
            "provider_capabilities": [resolved],
            "backend_metadata": {
                "source": MLX_PACKAGE_METADATA_SOURCE,
                "backend_package": "mlx_audio",
                "backend_contract": "mlx_audio.server",
                "package_label": "mlx-audio",
                "capabilities": [CAPABILITY_TEXT_TO_SPEECH, CAPABILITY_SPEECH_TO_TEXT, "speech_to_speech"],
                "server_kind": server_kind,
                "instance_capabilities": [resolved],
                "package_capabilities": [
                    CAPABILITY_TEXT_TO_SPEECH,
                    CAPABILITY_SPEECH_TO_TEXT,
                    "speech_to_speech",
                ],
                "native_endpoint_paths": ["/health", "/v1/audio/speech", "/v1/audio/transcriptions"],
                "lazy_loads_model": True,
                "runtime_constraints": ["shared_server_process", "model_selected_per_request"],
                "tts_response_formats": ["mp3", "wav", "flac"],
            },
        }

    if server_kind == "mlx_vlm":
        launch_defaults = _mlx_vlm_launch_defaults(repo_id=repo_id, metadata=metadata)
        provider_capabilities = _mlx_vlm_provider_capabilities(repo_id=repo_id, metadata=metadata)
        package_checks = _mlx_package_runtime_check(server_kind)
        runtime_version = str(package_checks.get("package_version") or "").strip() or None
        snapshot_conversion_version = str((metadata or {}).get("snapshot_mlx_vlm_conversion_version") or "").strip() or None
        compatibility_warnings: list[str] = []
        if (
            _mlx_vlm_requires_conservative_defaults(repo_id, metadata)
            and snapshot_conversion_version
            and _version_before(snapshot_conversion_version, (0, 6))
            and _version_at_least(runtime_version, (0, 6))
        ):
            compatibility_warnings.append(
                "gemma4_snapshot_converted_with_pre_0_6_mlx_vlm_runtime_0_6_or_newer"
            )
        runtime_constraints = ["single_loaded_model"]
        if compatibility_warnings:
            runtime_constraints.append("snapshot_runtime_compatibility_advisory")
        return {
            "backend_package": "mlx_vlm",
            "backend_contract": "mlx_vlm.server",
            "provider_capabilities": provider_capabilities,
            "backend_metadata": {
                "source": MLX_PACKAGE_METADATA_SOURCE,
                "backend_package": "mlx_vlm",
                "backend_contract": "mlx_vlm.server",
                "package_label": "mlx-vlm",
                "package_version": runtime_version,
                "snapshot_mlx_vlm_conversion_version": snapshot_conversion_version,
                "compatibility_warnings": compatibility_warnings,
                "capabilities": provider_capabilities + ["vision"],
                "server_kind": server_kind,
                "instance_capabilities": provider_capabilities,
                "package_capabilities": provider_capabilities,
                "native_endpoint_paths": [
                    "/models",
                    "/v1/models",
                    "/chat/completions",
                    "/v1/chat/completions",
                    "/responses",
                    "/v1/responses",
                    "/health",
                    "/unload",
                ],
                "lazy_loads_model": True,
                "supports_unload": True,
                "single_loaded_model": True,
                "runtime_knobs": [
                    "prefill_step_size",
                    "kv_bits",
                    "kv_quant_scheme",
                    "kv_group_size",
                    "max_kv_size",
                    "quantized_kv_start",
                ],
                "launch_defaults": launch_defaults,
                "runtime_constraints": runtime_constraints,
            },
        }

    launch_defaults = _mlx_lm_launch_defaults()
    return {
        "backend_package": "mlx_lm",
        "backend_contract": "mlx_lm.server",
        "provider_capabilities": [CAPABILITY_CHAT],
        "backend_metadata": {
            "source": MLX_PACKAGE_METADATA_SOURCE,
            "backend_package": "mlx_lm",
            "backend_contract": "mlx_lm.server",
            "package_label": "mlx-lm",
            "capabilities": [CAPABILITY_CHAT],
            "server_kind": server_kind,
            "instance_capabilities": [CAPABILITY_CHAT],
            "package_capabilities": [CAPABILITY_CHAT],
            "runtime_knobs": ["prefill_step_size", "prompt_cache_size", "prompt_cache_bytes"],
            "launch_defaults": launch_defaults,
            "cache_features": ["rotating_kv_cache", "prompt_cache"],
            "runtime_constraints": ["model_path_required_at_launch", "per_process_model_binding"],
        },
    }


@lru_cache(maxsize=16)
def _python_module_available(python_bin: str, module_name: str) -> bool:
    probe_script = r"""
import sys
from pathlib import Path

parts = sys.argv[1].split(".")

def module_exists(base: Path, names) -> bool:
    cursor = base
    for index, name in enumerate(names):
        package_dir = cursor / name
        module_file = cursor / f"{name}.py"
        is_last = index == len(names) - 1

        if is_last:
            return module_file.is_file() or package_dir.is_dir()

        if package_dir.is_dir():
            cursor = package_dir
            continue
        return False
    return False

for entry in sys.path:
    root = Path(entry or ".")
    try:
        if module_exists(root, parts):
            raise SystemExit(0)
    except OSError:
        continue

raise SystemExit(1)
"""
    try:
        result = subprocess.run(
            [python_bin, "-c", probe_script, module_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            env=sanitized_child_process_env(),
        )
    except Exception:
        return False
    return result.returncode == 0


@lru_cache(maxsize=16)
def _python_package_version(python_bin: str, package_name: str) -> Optional[str]:
    probe_script = r"""
import importlib.metadata as metadata
import sys

try:
    print(metadata.version(sys.argv[1]))
except Exception:
    raise SystemExit(1)
"""
    try:
        result = subprocess.run(
            [python_bin, "-c", probe_script, package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            env=sanitized_child_process_env(),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    version = str(result.stdout or "").strip()
    return version or None


@lru_cache(maxsize=16)
def _python_runtime_dependency_import_check(
    python_bin: str,
    module_name: str,
) -> dict[str, Any]:
    """Import one non-MLX dependency without loading a model runtime."""

    probe_script = r"""
import importlib
import sys

importlib.import_module(sys.argv[1])
"""
    try:
        result = subprocess.run(
            [python_bin, "-c", probe_script, module_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=20,
            env=sanitized_child_process_env(),
        )
    except subprocess.TimeoutExpired:
        return {
            "module": module_name,
            "ready": False,
            "error": f"Timed out while importing runtime dependency '{module_name}'.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "module": module_name,
            "ready": False,
            "error": str(exc),
        }

    if result.returncode == 0:
        return {
            "module": module_name,
            "ready": True,
            "error": None,
        }

    raw_error = str(result.stderr or result.stdout or "").strip()
    error_lines = [line.strip() for line in raw_error.splitlines() if line.strip()]
    error = error_lines[-1] if error_lines else (
        f"Runtime dependency '{module_name}' exited with status {result.returncode}."
    )
    return {
        "module": module_name,
        "ready": False,
        "error": error,
    }


def _mlx_model_runtime_module(server_kind: str, model_type: str) -> Optional[str]:
    normalized_kind = str(server_kind or "").strip().lower()
    normalized_type = str(model_type or "").strip().lower()
    if not normalized_type or normalized_kind not in {"mlx_lm", "mlx_vlm"}:
        return None
    package = "mlx_vlm" if normalized_kind == "mlx_vlm" else "mlx_lm"
    return f"{package}.models.{normalized_type}"


def _mlx_model_runtime_support(
    server_kind: str,
    model_type: str,
    python_bin: Optional[str],
) -> dict[str, Any]:
    module_name = _mlx_model_runtime_module(server_kind, model_type)
    if not module_name:
        return {
            "model_type": str(model_type or "").strip().lower(),
            "model_runtime_module": None,
            "model_runtime_module_available": True,
        }

    available = bool(python_bin) and _python_module_available(python_bin, module_name)
    return {
        "model_type": str(model_type or "").strip().lower(),
        "model_runtime_module": module_name,
        "model_runtime_module_available": available,
    }


@lru_cache(maxsize=8)
def _mlx_package_runtime_check(server_kind: str) -> dict:
    required_module = "mlx_lm"
    runtime_module = "mlx_lm"
    script_present = None

    if server_kind == "mlx_vlm":
        required_module = "mlx_vlm"
        runtime_module = "mlx_vlm"
    elif server_kind == "mlx_audio":
        required_module = "mlx_audio.server"
        runtime_module = "mlx_audio.server"
    elif server_kind == "mlx_whisper":
        required_module = "mlx_whisper"
        runtime_module = "mlx_whisper"
        script_present = SPEECH_SERVER_SCRIPT.exists()

    try:
        python_bin = resolve_mlx_python(required_module)
        python_error = None
    except Exception as exc:  # noqa: BLE001
        python_bin = None
        python_error = str(exc)

    runtime_module_available = bool(python_bin) and _python_module_available(python_bin, runtime_module)
    package_distribution = _mlx_distribution_name(server_kind)
    package_version = (
        _python_package_version(str(python_bin), package_distribution)
        if python_bin
        else None
    )
    dependency_checks: list[dict[str, Any]] = []
    if server_kind == "mlx_whisper" and python_bin:
        dependency_checks.append(
            _python_runtime_dependency_import_check(str(python_bin), "numba")
        )
    runtime_dependencies_ready = all(
        bool(item.get("ready"))
        for item in dependency_checks
    )
    runtime_dependency_error = next(
        (
            str(item.get("error") or "").strip()
            for item in dependency_checks
            if not item.get("ready") and str(item.get("error") or "").strip()
        ),
        None,
    )
    checks = {
        "required_python_module": required_module,
        "required_runtime_module": runtime_module,
        "package_distribution": package_distribution,
        "package_version": package_version,
        "python_path": python_bin,
        "python_resolved": bool(python_bin),
        "runtime_module_available": runtime_module_available,
        "runtime_dependency_checks": dependency_checks,
        "runtime_dependencies_ready": runtime_dependencies_ready,
        "runtime_dependency_error": runtime_dependency_error,
    }
    if script_present is not None:
        checks["server_script_path"] = str(SPEECH_SERVER_SCRIPT)
        checks["server_script_present"] = script_present
    if python_error:
        checks["python_error"] = python_error
    return checks


def _mlx_discovery_details(
    repo_id: str,
    capability: Optional[str],
    *,
    config_present: bool,
    metadata: Optional[dict] = None,
) -> dict:
    resolved = _resolved_mlx_capability(repo_id, capability, metadata=metadata)
    server_kind = _server_kind_for_capability(resolved, repo_id=repo_id, metadata=metadata)
    package_checks = _mlx_package_runtime_check(server_kind)
    model_support = _mlx_model_runtime_support(
        server_kind,
        str((metadata or {}).get("snapshot_model_type") or ""),
        package_checks.get("python_path") if isinstance(package_checks, dict) else None,
    )

    runnable = bool(
        config_present
        and package_checks.get("python_resolved")
        and package_checks.get("runtime_module_available")
        and package_checks.get("runtime_dependencies_ready", True)
        and model_support.get("model_runtime_module_available", True)
    )
    disabled_reason = None
    if not config_present:
        disabled_reason = "Snapshot is missing config.json, so Ollmo cannot treat it as a runnable MLX model."
    elif not package_checks.get("python_resolved"):
        disabled_reason = (
            f"Required MLX runtime is unavailable for {server_kind}: "
            f"{package_checks.get('python_error') or 'python resolver failed'}"
        )
    elif not package_checks.get("runtime_module_available"):
        disabled_reason = (
            f"Required runtime module '{package_checks.get('required_runtime_module')}' is not importable "
            f"from {package_checks.get('python_path') or 'the resolved MLX python'}."
        )
    elif not package_checks.get("runtime_dependencies_ready", True):
        disabled_reason = (
            f"Required runtime dependency is incompatible in "
            f"{package_checks.get('python_path') or 'the resolved MLX python'}: "
            f"{package_checks.get('runtime_dependency_error') or 'dependency import failed'}"
        )
    elif not model_support.get("model_runtime_module_available", True):
        disabled_reason = (
            f"Model type '{model_support.get('model_type') or 'unknown'}' is not supported by "
            f"{server_kind} in {package_checks.get('python_path') or 'the resolved MLX python'}."
        )
    elif server_kind == "mlx_whisper" and not package_checks.get("server_script_present"):
        runnable = False
        disabled_reason = f"Required Whisper shim script is missing: {package_checks.get('server_script_path')}"

    discovery_checks = {
        "config_json_present": config_present,
        **package_checks,
        **model_support,
    }
    return {
        "discovery_source": "huggingface_cache_snapshot",
        "discovery_state": "runnable" if runnable else "cached_only",
        "runnable_checks": discovery_checks,
        "runnable": runnable,
        "disabled_reason": disabled_reason,
    }


def describe_mlx_runtime_variants() -> dict[str, dict]:
    variants = {
        'mlx_lm': {
            'server_kind': 'mlx_lm',
            'label': 'MLX LM',
        },
        'mlx_vlm': {
            'server_kind': 'mlx_vlm',
            'label': 'MLX VLM',
        },
        'mlx_audio': {
            'server_kind': 'mlx_audio',
            'label': 'MLX Audio',
        },
        'mlx_whisper': {
            'server_kind': 'mlx_whisper',
            'label': 'MLX Whisper',
        },
    }
    payload: dict[str, dict] = {}
    for variant_id, details in variants.items():
        checks = _mlx_package_runtime_check(details['server_kind'])
        python_resolved = bool(checks.get('python_resolved'))
        module_available = bool(checks.get('runtime_module_available'))
        dependencies_ready = bool(checks.get('runtime_dependencies_ready', True))
        script_required = 'server_script_present' in checks
        script_present = bool(checks.get('server_script_present')) if script_required else True
        if python_resolved and module_available and dependencies_ready and script_present:
            runtime_state = 'runnable'
        elif python_resolved or module_available or script_required:
            runtime_state = 'degraded'
        else:
            runtime_state = 'missing'
        issues: list[str] = []
        if checks.get('python_error'):
            issues.append(str(checks.get('python_error')))
        if python_resolved and not module_available:
            issues.append(
                f"Required runtime module '{checks.get('required_runtime_module')}' is not importable."
            )
        if not dependencies_ready:
            issues.append(
                str(checks.get('runtime_dependency_error') or 'Required runtime dependency import failed.')
            )
        if script_required and not script_present:
            issues.append(f"Required server script is missing: {checks.get('server_script_path')}")
        payload[variant_id] = {
            'backend_id': variant_id,
            'family': 'mlx',
            'variant': variant_id,
            'label': details['label'],
            'runtime_state': runtime_state,
            'operations': {
                'discover': True,
                'list_models': True,
                'pull_model': True,
                'remove_model': True,
                'start_instance': runtime_state == 'runnable',
                'stop_instance': True,
            },
            'detection': checks,
            'issues': issues,
        }
    return payload


def sanitize_instance_id(repo_id: str, port: int) -> str:
    # Preserve familiar characters while keeping it JSON/UI friendly
    slug = repo_id.replace("/", "__")
    slug = SAFE_INSTANCE_RE.sub("-", slug)
    return f"{slug}-mlx-{port}"


def load_config_entries():
    return read_registry_entries(CONFIG_FILE)


def write_config_entries(entries):
    write_registry_entries(entries, path=CONFIG_FILE, preserve_agents=True, sync_external=False)


def prune_stale_mlx_entries() -> None:
    entries = load_config_entries()
    changed = False
    kept = []
    for inst in entries:
        if inst.get("backend") == "mlx":
            port = inst.get("port")
            if port and port_in_use(int(port)):
                kept.append(inst)
            else:
                changed = True
        else:
            kept.append(inst)
    if changed:
        write_config_entries(kept)


def register_mlx_instance(instance: dict) -> None:
    entries = load_config_entries()
    filtered = []
    for inst in entries:
        if inst.get("backend") == "mlx":
            same_id = inst.get("instance_id") == instance["instance_id"]
            same_port = inst.get("port") == instance["port"]
            if same_id or same_port:
                continue
        filtered.append(inst)
    filtered.append(instance)
    write_config_entries(filtered)


def remove_mlx_instance(instance_id: str) -> None:
    entries = load_config_entries()
    filtered = [inst for inst in entries if inst.get("instance_id") != instance_id]
    if len(filtered) != len(entries):
        write_config_entries(filtered)


def list_mlx_instances() -> list:
    prune_stale_mlx_entries()
    return [inst for inst in load_config_entries() if inst.get("backend") == "mlx"]


def wait_for_port_shutdown(port: int, timeout: float = 5.0, interval: float = 0.2) -> bool:
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if not port_in_use(port):
            return True
        time.sleep(interval)
    return not port_in_use(port)


def stop_mlx_instance(instance_id: str):
    entries = load_config_entries()
    target = next((inst for inst in entries if inst.get("instance_id") == instance_id and inst.get("backend") == "mlx"), None)
    if not target:
        return False, None

    port = target.get("port")
    success = True
    if port:
        success = kill_processes_on_port(int(port))
        if success:
            success = wait_for_port_shutdown(int(port))

    if success:
        remove_mlx_instance(instance_id)
    return success, target


def build_instance_record(
    repo_id: str,
    model_path: str,
    port: int,
    log_path: Path,
    capability: Optional[str] = None,
    pid: Optional[int] = None,
    launch_defaults: Optional[dict[str, Any]] = None,
) -> dict:
    snapshot_metadata = read_snapshot_model_metadata(repo_id, model_path)
    registry_metadata = build_registry_metadata(repo_id, "mlx", capability, metadata=snapshot_metadata)
    resolved_capability = str(registry_metadata.get("capability") or capability or "").strip() or None
    contract_details = _mlx_contract_details(repo_id, resolved_capability, metadata=snapshot_metadata)
    normalized_launch_defaults = launch_defaults if isinstance(launch_defaults, dict) else {}
    if normalized_launch_defaults:
        contract_details["backend_metadata"] = {
            **contract_details["backend_metadata"],
            "launch_defaults": normalized_launch_defaults,
        }
    request_model = repo_id if contract_details["backend_package"] == "mlx_audio" else model_path
    snapshot_extras = dict(snapshot_metadata)
    return {
        "instance_id": sanitize_instance_id(repo_id, port),
        "model": repo_id,
        **snapshot_extras,
        **registry_metadata,
        "backend_package": contract_details["backend_package"],
        "backend_contract": contract_details["backend_contract"],
        "provider_capabilities": contract_details["provider_capabilities"],
        "backend_metadata": contract_details["backend_metadata"],
        "port": port,
        "pid": pid,
        "model_path": model_path,
        "request_model": request_model,
        "launch_defaults": normalized_launch_defaults,
        "mlx_server": _server_kind_for_capability(
            resolved_capability,
            repo_id=repo_id,
            metadata=snapshot_metadata,
        ),
        "log": str(log_path),
        "ts": int(time.time()),
    }


def reconcile_recent_mlx_instance(
    instance: dict,
    *,
    monitor_sec: float = MLX_POST_START_MONITOR_SEC,
    poll_sec: float = MLX_POST_START_MONITOR_INTERVAL_SEC,
) -> bool:
    instance_id = str(instance.get("instance_id") or "").strip()
    port = instance.get("port")
    if not instance_id or port is None:
        return False

    try:
        port = int(port)
    except (TypeError, ValueError):
        return False

    deadline = time.time() + max(0.0, float(monitor_sec))
    sleep_interval = max(0.1, float(poll_sec))

    while True:
        if not port_in_use(port):
            remove_mlx_instance(instance_id)
            remove_instance_status(instance_id)
            return False
        if time.time() >= deadline:
            return True
        time.sleep(sleep_interval)


def schedule_recent_mlx_instance_reconciliation(
    instance: dict,
    *,
    monitor_sec: float = MLX_POST_START_MONITOR_SEC,
    poll_sec: float = MLX_POST_START_MONITOR_INTERVAL_SEC,
) -> None:
    thread = threading.Thread(
        target=reconcile_recent_mlx_instance,
        kwargs={
            "instance": dict(instance),
            "monitor_sec": monitor_sec,
            "poll_sec": poll_sec,
        },
        daemon=True,
        name=f"mlx-reconcile-{instance.get('instance_id') or 'unknown'}",
    )
    thread.start()


def list_cached_models() -> list:
    return find_hf_cached_models()


def resolve_hf_cli() -> str:
    candidates = []
    def add_python_sibling_candidates(python_path: Path) -> None:
        raw = python_path.expanduser()
        candidates.append(raw.parent / "hf")
        try:
            resolved = raw.resolve()
        except FileNotFoundError:
            return
        resolved_candidate = resolved.parent / "hf"
        if resolved_candidate != candidates[-1]:
            candidates.append(resolved_candidate)

    if ENV_MLX_PYTHON:
        add_python_sibling_candidates(Path(ENV_MLX_PYTHON))
    add_python_sibling_candidates(DEFAULT_MLX_PYTHON)
    try:
        add_python_sibling_candidates(Path(resolve_mlx_python("mlx_lm")))
    except Exception:
        pass

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    if shutil.which("hf"):
        return shutil.which("hf")

    searched = ", ".join(str(path) for path in candidates if path)
    raise RuntimeError(f"No executable 'hf' CLI was found. Checked: {searched}")


def _hf_model_target(repo_id: str) -> str:
    token = str(repo_id or "").strip()
    if token.startswith(HF_REPO_PREFIX):
        return token
    return f"{HF_REPO_PREFIX}{token}"


def pull_hf_model(repo_id: str) -> tuple[bool, str]:
    if not repo_id:
        return False, "No Hugging Face repo was provided."
    try:
        hf_cli = resolve_hf_cli()
        result = subprocess.run(
            [hf_cli, "download", repo_id],
            capture_output=True,
            text=True,
            check=False,
            env=sanitized_child_process_env(),
        )
        if result.returncode == 0:
            message = result.stdout.strip() or f"HF model {repo_id} was loaded into the cache successfully."
            return True, message
        error = result.stderr.strip() or result.stdout.strip() or f"Download failed for {repo_id}."
        return False, error
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def remove_hf_model(repo_id: str) -> tuple[bool, str]:
    if not repo_id:
        return False, "No Hugging Face repo was provided."
    try:
        hf_cli = resolve_hf_cli()
        result = subprocess.run(
            [hf_cli, "cache", "rm", _hf_model_target(repo_id), "--yes"],
            capture_output=True,
            text=True,
            check=False,
            env=sanitized_child_process_env(),
        )
        if result.returncode == 0:
            message = result.stdout.strip() or f"HF model {repo_id} was removed from the cache."
            return True, message
        error = result.stderr.strip() or result.stdout.strip() or f"Removal failed for {repo_id}."
        return False, error
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _server_kind_for_capability(
    capability: Optional[str],
    *,
    repo_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    resolved = normalize_capability(capability)
    model_type = str((metadata or {}).get("snapshot_model_type") or "").strip().lower()
    name = str(repo_id or "").strip().lower()
    if resolved == CAPABILITY_SPEECH_TO_TEXT:
        if "whisper" not in model_type and "whisper" not in name:
            return "mlx_audio"
        return "mlx_whisper"
    if resolved == CAPABILITY_TEXT_TO_SPEECH:
        return "mlx_audio"
    if resolved == CAPABILITY_VISION_ANALYSIS:
        return "mlx_vlm"
    return "mlx_lm"


def start_mlx_model(
    repo_id: str,
    model_path: Optional[str] = None,
    preferred_port: Optional[int] = None,
    capability: Optional[str] = None,
    *,
    launch_defaults: Optional[dict[str, Any]] = None,
    register_instance: bool = True,
    prune_registry: bool = True,
    start_source: Optional[str] = None,
) -> dict:
    normalized_start_source = validate_start_source(
        start_source,
        context='mlx_start_model',
    )
    if prune_registry:
        prune_stale_mlx_entries()
    snapshots = find_mlx_snapshots()
    if not snapshots:
        raise RuntimeError("No MLX models were found in the cache.")

    selected = None
    normalized_path = Path(model_path).resolve() if model_path else None
    for candidate in snapshots:
        if normalized_path and Path(candidate["path"]).resolve() == normalized_path:
            selected = candidate
            break
        if candidate["repo"] == repo_id:
            selected = candidate
            if not model_path:
                break

    if not selected:
        available = ", ".join(sorted(entry["repo"] for entry in snapshots))
        raise ValueError(f"MLX model '{repo_id}' was not found. Available models: {available}")

    if preferred_port is not None:
        if port_in_use(preferred_port):
            raise RuntimeError(f"Bevorzugter Port {preferred_port} ist bereits belegt.")
        port = preferred_port
    else:
        port = next_free_port()

    snapshot_metadata = read_snapshot_model_metadata(selected["repo"], selected["path"])
    registry_metadata = build_registry_metadata(
        selected["repo"],
        "mlx",
        capability,
        metadata=snapshot_metadata,
    )
    resolved_capability = normalize_capability(registry_metadata.get("capability")) or infer_capability(
        selected["repo"],
        "mlx",
        metadata=snapshot_metadata,
    )
    server_kind = str(selected.get("mlx_server") or "").strip() or _server_kind_for_capability(
        resolved_capability,
        repo_id=selected["repo"],
        metadata=snapshot_metadata,
    )
    effective_launch_defaults: dict[str, Any]
    if server_kind == "mlx_vlm":
        effective_launch_defaults = _mlx_vlm_launch_defaults(
            repo_id=selected["repo"],
            metadata=snapshot_metadata,
        )
    elif server_kind == "mlx_lm":
        effective_launch_defaults = _mlx_lm_launch_defaults()
    else:
        effective_launch_defaults = {}
    if isinstance(launch_defaults, dict):
        effective_launch_defaults.update(
            {
                key: value
                for key, value in launch_defaults.items()
                if value not in (None, "")
            }
        )
    if server_kind == "mlx_whisper":
        log, pid = launch_whisper_server(selected["path"], port)
    elif server_kind == "mlx_audio":
        log, pid = launch_audio_server(port)
    elif server_kind == "mlx_vlm":
        log, pid = launch_vlm_server(port, launch_defaults=effective_launch_defaults)
    elif server_kind == "mlx_lm":
        log, pid = launch(selected["path"], port, launch_defaults=effective_launch_defaults)
    else:
        raise RuntimeError(
            f"MLX server kind '{server_kind}' is not currently supported for '{repo_id}'."
        )
    if not wait_for_port(port):
        kill_processes_on_port(port)
        raise RuntimeError(f"MLX model '{repo_id}' did not open port {port}. See {log}.")

    instance_record = build_instance_record(
        selected["repo"],
        selected["path"],
        port,
        log,
        capability=capability,
        pid=pid,
        launch_defaults=effective_launch_defaults,
    )
    instance_record = attach_start_audit(
        instance_record,
        start_source=normalized_start_source,
        context='mlx_start_model',
        extra={'backend': 'mlx', 'capability': resolved_capability},
    )
    if register_instance:
        register_mlx_instance(instance_record)
        schedule_recent_mlx_instance_reconciliation(instance_record)
    return instance_record

def resolve_mlx_python(required_module: str = "mlx_lm") -> str:
    global _RESOLVED_MLX_PYTHON
    if _RESOLVED_MLX_PYTHON is not None and required_module == "mlx_lm":
        return _RESOLVED_MLX_PYTHON

    candidates = []
    if required_module.startswith("mlx_audio") and ENV_MLX_AUDIO_PYTHON:
        candidates.append(Path(ENV_MLX_AUDIO_PYTHON))
    if required_module.startswith("mlx_audio"):
        candidates.append(DEFAULT_MLX_AUDIO_PYTHON)
    if ENV_MLX_PYTHON:
        candidates.append(Path(ENV_MLX_PYTHON))
    candidates.append(DEFAULT_MLX_PYTHON)
    candidates.append(Path(sys.executable))

    seen = set()
    ordered = []
    for cand in candidates:
        if not cand:
            continue
        cand = Path(cand).expanduser()
        try:
            key = str(cand.resolve(strict=True))
        except FileNotFoundError:
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cand)

    attempts = []
    for cand in ordered:
        if not cand.exists():
            attempts.append(f"{cand} (not found)")
            continue
        if not os.access(cand, os.X_OK):
            attempts.append(f"{cand} (not executable)")
            continue
        if _python_module_available(str(cand), required_module):
            if required_module == "mlx_lm":
                _RESOLVED_MLX_PYTHON = str(cand)
            return str(cand)
        attempts.append(f"{cand} (without {required_module})")

    detail = "; ".join(attempts) if attempts else "no candidates available"
    raise RuntimeError(f"No Python interpreter with '{required_module}' was found ({detail}).")

def launch(model_path: str, port: int, launch_defaults: Optional[dict[str, Any]] = None):
    python_bin = resolve_mlx_python("mlx_lm")
    log = LOG_DIR / f"mlx_{port}.log"
    cmd = [python_bin, "-m", "mlx_lm.server", "--model", model_path, "--port", str(port)]
    normalized_launch_defaults = launch_defaults if isinstance(launch_defaults, dict) else {}
    if normalized_launch_defaults.get("prefill_step_size") is not None:
        cmd.extend(["--prefill-step-size", str(normalized_launch_defaults["prefill_step_size"])])
    if normalized_launch_defaults.get("prompt_cache_size") is not None:
        cmd.extend(["--prompt-cache-size", str(normalized_launch_defaults["prompt_cache_size"])])
    if normalized_launch_defaults.get("prompt_cache_bytes") not in (None, ""):
        cmd.extend(["--prompt-cache-bytes", str(normalized_launch_defaults["prompt_cache_bytes"])])
    prepare_clean_runtime_log(
        log,
        metadata={
            "backend": "mlx",
            "port": port,
            "mlx_server": "mlx_lm",
            "model_path": model_path,
        },
    )
    with open(log, "wb") as lf:
        lf.write(f"\n\n---- LAUNCH {datetime.now()} ----\n".encode())
        lf.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
            env=sanitized_child_process_env(),
        )
    return log, proc.pid


def launch_vlm_server(port: int, launch_defaults: Optional[dict[str, Any]] = None):
    python_bin = resolve_mlx_python("mlx_vlm")
    log = LOG_DIR / f"mlx_vlm_{port}.log"
    cmd = [python_bin, "-m", MLX_VLM_MODULE, "--host", "127.0.0.1", "--port", str(port)]
    normalized_launch_defaults = launch_defaults if isinstance(launch_defaults, dict) else {}
    if normalized_launch_defaults.get("prefill_step_size") is not None:
        cmd.extend(["--prefill-step-size", str(normalized_launch_defaults["prefill_step_size"])])
    if normalized_launch_defaults.get("kv_bits") is not None:
        cmd.extend(["--kv-bits", str(normalized_launch_defaults["kv_bits"])])
    scheme = str(normalized_launch_defaults.get("kv_quant_scheme") or "").strip().lower()
    if scheme in {"uniform", "turboquant"}:
        cmd.extend(["--kv-quant-scheme", scheme])
    if normalized_launch_defaults.get("kv_group_size") is not None:
        cmd.extend(["--kv-group-size", str(normalized_launch_defaults["kv_group_size"])])
    if normalized_launch_defaults.get("max_kv_size") is not None:
        cmd.extend(["--max-kv-size", str(normalized_launch_defaults["max_kv_size"])])
    if normalized_launch_defaults.get("quantized_kv_start") is not None:
        cmd.extend(["--quantized-kv-start", str(normalized_launch_defaults["quantized_kv_start"])])
    prepare_clean_runtime_log(
        log,
        metadata={
            "backend": "mlx",
            "port": port,
            "mlx_server": "mlx_vlm",
        },
    )
    with open(log, "wb") as lf:
        lf.write(f"\n\n---- LAUNCH {datetime.now()} ----\n".encode())
        lf.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
            env=sanitized_child_process_env(),
        )
    return log, proc.pid


def launch_audio_server(port: int):
    python_bin = resolve_mlx_python("mlx_audio.server")
    log = LOG_DIR / f"mlx_audio_{port}.log"
    cmd = [
        python_bin,
        "-m",
        MLX_AUDIO_MODULE,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    prepare_clean_runtime_log(
        log,
        metadata={
            "backend": "mlx",
            "port": port,
            "mlx_server": "mlx_audio",
        },
    )
    with open(log, "wb") as lf:
        lf.write(f"\n\n---- LAUNCH {datetime.now()} ----\n".encode())
        lf.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
            env=sanitized_child_process_env(),
        )
    return log, proc.pid


def launch_whisper_server(model_path: str, port: int):
    if not SPEECH_SERVER_SCRIPT.exists():
        raise RuntimeError(f"Whisper-Server Skript fehlt: {SPEECH_SERVER_SCRIPT}")
    python_bin = resolve_mlx_python()
    log = LOG_DIR / f"mlx_whisper_{port}.log"
    cmd = [
        python_bin,
        str(SPEECH_SERVER_SCRIPT),
        "--model-path",
        model_path,
        "--port",
        str(port),
    ]
    prepare_clean_runtime_log(
        log,
        metadata={
            "backend": "mlx",
            "port": port,
            "mlx_server": "mlx_whisper",
            "model_path": model_path,
        },
    )
    with open(log, "wb") as lf:
        lf.write(f"\n\n---- LAUNCH {datetime.now()} ----\n".encode())
        lf.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
            env=sanitized_child_process_env(),
        )
    return log, proc.pid

def decode_repo(models_dir_name: str) -> str:
    # models--org--repo  ->  org/repo
    m = re.match(r"models--(.+?)--(.+)$", models_dir_name)
    if not m: return models_dir_name
    org = m.group(1)
    repo = m.group(2)
    return f"{org}/{repo}"


def format_repo_label(repo_id: str) -> str:
    token = str(repo_id or "").strip()
    if "/" in token:
        token = token.split("/", 1)[1]
    token = re.sub(r"(?i)^mlx-community[-_/]*", "", token)
    token = re.sub(r"(?i)-bf16$", "", token)
    token = re.sub(r"(?i)-mlx$", "", token)
    token = token.replace("_", " ").replace("-", " ")
    token = re.sub(r"\s+", " ", token).strip()
    return token or str(repo_id or "").strip()


def format_capability_badge(capability: Optional[str]) -> str:
    resolved = normalize_capability(capability)
    return CAPABILITY_BADGES.get(resolved, f"🧩 {resolved or 'Unknown'}")


def render_model_option(index: int, model: dict, label_width: int = 34) -> str:
    label = format_repo_label(model.get("repo", ""))
    capability = format_capability_badge(model.get("capability"))
    size = f"{float(model.get('size_gb') or 0.0):.2f} GB"
    return f"[{index}] {label:<{label_width}} {capability:<14} {size:>8}"


def _snapshot_is_llama_cpp_managed(repo_id: str, snapshot_dir: Path) -> bool:
    # GGUF snapshots belong to Ollmo's llama.cpp surface and should not show up
    # as generic MLX/HF-cache models.
    try:
        next(snapshot_dir.rglob("*.gguf"))
        return True
    except StopIteration:
        return False
    except OSError:
        return False

def find_hf_cached_models():
    base = CACHE_DIR
    out = []
    if not base.exists():
        return out
    for mdir in sorted(base.glob("models--*--*")):
        repo_id = decode_repo(mdir.name)
        snaps = (mdir / "snapshots")
        if not snaps.is_dir(): 
            continue
        # Take the newest snapshot first.
        cand = sorted(snaps.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in cand:
            if _snapshot_is_llama_cpp_managed(repo_id, snap):
                break
            cfg = snap / "config.json"
            size_gb = sum((f.stat().st_size for f in snap.rglob("*") if f.is_file())) / (1024**3)
            snapshot_metadata = read_snapshot_model_metadata(repo_id, str(snap))
            capability = infer_capability(repo_id, "mlx", metadata=snapshot_metadata)
            server_kind = _server_kind_for_capability(
                capability,
                repo_id=repo_id,
                metadata=snapshot_metadata,
            )
            config_present = cfg.is_file()
            out.append({
                "repo": repo_id,
                "path": str(snap),
                "mtime": datetime.fromtimestamp(snap.stat().st_mtime),
                "size_gb": size_gb,
                "capability": capability,
                "mlx_server": server_kind,
                **_mlx_contract_details(repo_id, capability, metadata=snapshot_metadata),
                **_mlx_discovery_details(
                    repo_id,
                    capability,
                    config_present=config_present,
                    metadata=snapshot_metadata,
                ),
                **snapshot_metadata,
            })
            break  # Show only one snapshot per repo.
    # dedupe by repo (keep freshest)
    seen = {}
    for item in out:
        if item["repo"] not in seen or item["mtime"] > seen[item["repo"]]["mtime"]:
            seen[item["repo"]] = item
    return sorted(seen.values(), key=lambda x: x["repo"].lower())


def find_mlx_snapshots():
    return [item for item in find_hf_cached_models() if item.get("runnable")]

def port_in_use(port:int)->bool:
    try:
        res = subprocess.run(
            ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=sanitized_child_process_env(),
        )
        return res.returncode == 0
    except FileNotFoundError:
        # macOS normally ships with lsof; if it is missing, treat ports as not in use.
        return False

def next_free_port(start=START_PORT, pmax=PORT_MAX):
    p = start
    while p <= pmax:
        if not port_in_use(p):
            return p
        p += 1
    raise RuntimeError("Kein freier Port im Bereich %d..%d" % (start, pmax))

def wait_for_port(port:int, timeout:float=15.0, interval:float=0.5)->bool:
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if port_in_use(port):
            return True
        time.sleep(interval)
    return False

TOKEN_RANGE_PATTERN = re.compile(r"^\d+(?:-\d+)?$")


def expand_selection_token(token: str):
    token = token.strip()
    if not token:
        return []
    normalized = token.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", "", normalized)
    if not TOKEN_RANGE_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid selection value: {token}")
    if "-" in normalized:
        a_str, b_str = normalized.split("-", 1)
        a, b = int(a_str), int(b_str)
        if a <= b:
            return list(range(a, b + 1))
        return list(range(b, a + 1))
    return [int(normalized)]

def main():
    prune_stale_mlx_entries()
    models = find_mlx_snapshots()
    if not models:
        print("No MLX models were found in the HF cache:", CACHE_DIR, file=sys.stderr)
        print("Tip: try running once, for example: mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --prompt 'hi'")
        sys.exit(1)

    try:
        mlx_python = resolve_mlx_python()
    except RuntimeError as err:
        print(f"❌ {err}", file=sys.stderr)
        print("Set MLX_PYTHON to the interpreter with 'mlx_lm' installed (for example /opt/mlx/venv/bin/python).", file=sys.stderr)
        return

    print(f"ℹ️  Using MLX Python: {mlx_python}")

    print("\nWhich MLX models should be started as their own instance?")
    print("Enter numbers or ranges separated by spaces.")
    print("You can choose a model more than once (for example: '1 1 2').")
    print("Enter = cancel.\n")
    label_width = min(40, max(len(format_repo_label(m["repo"])) for m in models) + 2)
    for i, m in enumerate(models, 1):
        print(render_model_option(i, m, label_width=label_width))

    choice = input("> ").strip().lower()
    if not choice:
        print("Canceled."); return
    selection_indices = []
    if choice in ("a","all","*"):
        selection_indices = list(range(1, len(models) + 1))
    else:
        tokens = re.split(r'[,\s]+', choice)
        for part in tokens:
            part = part.strip()
            if not part:
                continue
            try:
                selection_indices.extend(expand_selection_token(part))
            except ValueError:
                print(f"⚠️  Ignoring invalid input: {part}")
    if not selection_indices:
        print("No valid selection."); return
    sorted_indices = selection_indices
    invalid = [i for i in sorted_indices if i < 1 or i > len(models)]
    if invalid:
        max_idx = len(models)
        unique_invalid = sorted(set(invalid))
        if len(unique_invalid) > 6:
            printable = ", ".join(str(i) for i in unique_invalid[:6]) + ", …"
        else:
            printable = ", ".join(str(i) for i in unique_invalid)
        print(f"⚠️  Skipping invalid numbers ({printable}); there are only {max_idx} model(s).")
    valid_indices = [i for i in sorted_indices if 1 <= i <= len(models)]
    if not valid_indices:
        print("No valid selection."); return
    grouped = OrderedDict()
    for idx in valid_indices:
        grouped.setdefault(idx, 0)
        grouped[idx] += 1
    summary_parts = []
    for idx, count in grouped.items():
        name = format_repo_label(models[idx - 1]["repo"])
        if count > 1:
            summary_parts.append(f"#{idx} {name} ×{count}")
        else:
            summary_parts.append(f"#{idx} {name}")
    summary = ", ".join(summary_parts)
    print(f"➡️  Selection: {summary}")
    selected = [models[i-1] for i in valid_indices]

    # Start selected models.
    current_port = START_PORT
    for m in selected:
        try:
            p = next_free_port(current_port, PORT_MAX)
        except RuntimeError as e:
            print("❌", e); break
        current_port = p+1
        capability = infer_capability(m["repo"], "mlx")
        capability_badge = format_capability_badge(capability)
        display_label = format_repo_label(m["repo"])
        print(f"\n🚀 Starting instance '{display_label}' ({capability_badge}) on port {p}...")
        try:
            instance_record = start_mlx_model(
                m["repo"],
                model_path=m["path"],
                preferred_port=p,
                capability=capability,
                start_source='startup_policy',
            )
        except Exception as exc:  # noqa: BLE001
            if capability == CAPABILITY_SPEECH_TO_TEXT:
                print(f"❌ Whisper start failed: {exc}")
            elif capability == CAPABILITY_TEXT_TO_SPEECH:
                print(f"❌ TTS start failed: {exc}")
            else:
                print(f"❌ Start failed: {exc}")
            continue
        print(
            f"✅ Running: {display_label} at http://127.0.0.1:{p}"
        )
        print(
            f"   {capability_badge} | Instance: {instance_record['instance_id']} | Logs: {instance_record.get('log')}"
        )

if __name__ == "__main__":
    main()
