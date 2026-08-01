# ollama_model_manager.py
import json
import os
import signal
import shutil
import socket
import subprocess
import time
import re
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # Still required for /api/ps.
except ImportError:  # pragma: no cover - optional in lightweight test environments
    class _RequestsFallback:
        class exceptions:  # noqa: D106 - compatibility shim
            RequestException = Exception

        @staticmethod
        def get(*args, **kwargs):
            raise RuntimeError("requests is required for Ollama HTTP probes.")

        @staticmethod
        def post(*args, **kwargs):
            raise RuntimeError("requests is required for Ollama HTTP probes.")

    requests = _RequestsFallback()
from helpers.ollama_client import fetch_ollama_show
from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    build_ollama_show_summary,
    build_registry_metadata,
    infer_capability,
    normalize_capability,
)
from ollmo_core.start_policy import attach_start_audit, validate_start_source
from ollmo_runtime.child_process_env import sanitized_child_process_env
from ollmo_runtime.runtime_log_hygiene import prepare_clean_global_log
from ollmo_runtime.registry import (
    enrich_registry_entry as core_enrich_registry_entry,
    filter_active_registry_entries,
    is_port_listening as registry_is_port_listening,
    list_runtime_entries,
    load_registry_entries,
    pid_is_running as registry_pid_is_running,
    read_registry_entries,
    write_registry_entries,
)

OLLAMA_CLI = "/opt/homebrew/bin/ollama"
CONFIG_FILE = "model_ports.json"
LOG_DIR = Path("logs")
DEFAULT_SERVER_PORT = 11434
START_PORT = 11435
MAX_PORT = 11500
instance_counters = Counter()
STOP_WAIT_SECONDS = 20.0
STOP_WAIT_EXTENSION = 15.0
MODEL_SHOW_TIMEOUT_SECONDS = 10
MODEL_EMBED_WARMUP_TIMEOUT_SECONDS = 20
SAFE_LOG_FRAGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
OLLAMA_HOMEBREW_PREFIX = Path(OLLAMA_CLI).expanduser().parent.parent
OLLAMA_LIBRARY_DIR_CANDIDATES = [
    OLLAMA_HOMEBREW_PREFIX / "lib",
    OLLAMA_HOMEBREW_PREFIX / "opt" / "mlx-c" / "lib",
]
CAPABILITY_BADGES = {
    CAPABILITY_CHAT: "💬 Chat",
    CAPABILITY_EMBEDDING: "🧩 Embedding",
    CAPABILITY_VISION_ANALYSIS: "👁 OCR/Vision",
    CAPABILITY_IMAGE_GENERATION: "🖼️  Image",
    CAPABILITY_TEXT_TO_SPEECH: "🔊 TTS",
    CAPABILITY_SPEECH_TO_TEXT: "🎙 STT",
}


@dataclass
class StopResult:
    state: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        return self.state in {"stopped", "stopping"}


def enrich_registry_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return core_enrich_registry_entry(entry)


def safe_log_fragment(value: str) -> str:
    fragment = SAFE_LOG_FRAGMENT_RE.sub("_", (value or "").strip())
    return fragment.strip("._") or "model"


def _prepend_env_path(existing: Optional[str], value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return str(existing or "")
    parts = [token]
    if existing:
        parts.append(existing)
    return ":".join(part for part in parts if part)


def build_ollama_env(*, port: Optional[int] = None) -> Dict[str, str]:
    env = sanitized_child_process_env()
    for candidate in OLLAMA_LIBRARY_DIR_CANDIDATES:
        if not candidate.exists():
            continue
        current = str(candidate)
        env["OLLAMA_LIBRARY_PATH"] = _prepend_env_path(env.get("OLLAMA_LIBRARY_PATH"), current)
        env["DYLD_LIBRARY_PATH"] = _prepend_env_path(env.get("DYLD_LIBRARY_PATH"), current)
        env["DYLD_FALLBACK_LIBRARY_PATH"] = _prepend_env_path(env.get("DYLD_FALLBACK_LIBRARY_PATH"), current)
        break
    if port is not None:
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    return env


def format_model_label(model_name: str) -> str:
    token = str(model_name or "").strip()
    if "/" in token:
        token = token.split("/", 1)[1]
    token = token.replace(":latest", "")
    token = token.replace(":", " ")
    token = token.replace("_", " ").replace("-", " ")
    token = re.sub(r"\s+", " ", token).strip()
    return token or str(model_name or "").strip()


def format_capability_badge(capability: Optional[str]) -> str:
    resolved = normalize_capability(capability)
    return CAPABILITY_BADGES.get(resolved, f"🧩 {resolved or 'Unknown'}")


def render_model_option(index: int, entry: Dict[str, Any], label_width: int = 28) -> str:
    label = format_model_label(str(entry.get("name") or ""))
    capability = format_capability_badge(entry.get("capability"))
    size = str(entry.get("size") or "unknown")
    return f"[{index}] {label:<{label_width}} {capability:<14} {size:>8}"


def _normalize_provider_capabilities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for raw in value:
        token = normalize_capability(raw)
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _capability_from_provider_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    provider_capabilities = _normalize_provider_capabilities((metadata or {}).get("capabilities"))
    if CAPABILITY_EMBEDDING in provider_capabilities:
        return CAPABILITY_EMBEDDING
    return None


def _fetch_model_metadata(model_name: str, *, port: Optional[int] = None) -> dict[str, Any]:
    model = str(model_name or "").strip()
    if not model:
        return {}
    try:
        target_port = int(port) if port is not None else DEFAULT_SERVER_PORT
        return fetch_ollama_show(
            target_port,
            model,
            timeout=MODEL_SHOW_TIMEOUT_SECONDS,
        )
    except Exception:
        return {}


def list_local_model_entries() -> List[Dict[str, Any]]:
    try:
        print("ℹ️  Querying the model list from the default server (11434)...")
        result = subprocess.run(
            [OLLAMA_CLI, "list"],
            capture_output=True,
            text=True,
            check=True,
            env=build_ollama_env(),
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) <= 1:
            print("   No models found.")
            return []

        entries: List[Dict[str, Any]] = []
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            model_name = parts[0]
            size = "unknown"
            if len(parts) >= 4:
                size = f"{parts[2]} {parts[3]}"
            provider_metadata = _fetch_model_metadata(model_name)
            capability = (
                _capability_from_provider_metadata(provider_metadata)
                or infer_capability(model_name, "ollama", metadata=provider_metadata)
            )
            entry = {
                "name": model_name,
                "capability": capability,
                "size": size,
            }
            show_summary = build_ollama_show_summary(provider_metadata)
            if show_summary:
                entry["backend_metadata"] = show_summary
            provider_capabilities = _normalize_provider_capabilities(provider_metadata.get("capabilities"))
            if provider_capabilities:
                entry["provider_capabilities"] = provider_capabilities
            entries.append(entry)
        print(f"   Found models: {', '.join(entry['name'] for entry in entries)}")
        return entries
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error while fetching the model list: {exc}")
        return []


def is_port_listening(port: int, host: str = "localhost") -> bool:
    return registry_is_port_listening(port, host)


def pid_is_running(pid: Optional[int]) -> bool:
    return registry_pid_is_running(pid)


def read_config() -> List[Dict]:
    return read_registry_entries(CONFIG_FILE)


def write_config(instances: List[Dict]) -> None:
    write_registry_entries(
        instances,
        path=CONFIG_FILE,
        preserve_agents=True,
        sync_external=False,
    )


def filter_running_instances(instances: List[Dict]) -> List[Dict]:
    return filter_active_registry_entries(instances)


def load_instances(prune: bool = True) -> List[Dict]:
    return load_registry_entries(
        prune=prune,
        path=CONFIG_FILE,
        sync_external=False,
    )


def initialize_instance_counters(instances: List[Dict]) -> None:
    instance_counters.clear()
    for inst in instances:
        model_name = inst.get("model")
        instance_id = inst.get("instance_id")
        if not model_name or not instance_id:
            continue
        prefix = f"{model_name}-"
        if instance_id.startswith(prefix):
            suffix = instance_id[len(prefix) :]
            try:
                value = int(suffix)
            except ValueError:
                continue
            if value > instance_counters[model_name]:
                instance_counters[model_name] = value


def allocate_instance_id(model_name: str, existing_instances: List[Dict]) -> str:
    while True:
        instance_counters[model_name] += 1
        candidate = f"{model_name}-{instance_counters[model_name]}"
        if all(inst.get("instance_id") != candidate for inst in existing_instances):
            return candidate


def ensure_default_server_running() -> bool:
    if is_port_listening(DEFAULT_SERVER_PORT):
        print(f"✅ Default server (port {DEFAULT_SERVER_PORT}) is running.")
        return True

    print(f"🧠 Default server (port {DEFAULT_SERVER_PORT}) is not running. Starting it now...")
    try:
        default_env = build_ollama_env()
        log_file = LOG_DIR / f"ollama_default_server_{DEFAULT_SERVER_PORT}.log"
        default_env.pop("OLLAMA_HOST", None)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        prepare_clean_global_log(
            log_file,
            metadata={
                'service': 'ollama_default',
                'port': DEFAULT_SERVER_PORT,
            },
        )
        with open(log_file, "w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [OLLAMA_CLI, "serve"],
                env=default_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        time.sleep(3)
        if is_port_listening(DEFAULT_SERVER_PORT):
            print(f"✅ Default server started (PID: {process.pid}).")
            return True
        print(f"❌ Could not start the default server. Check '{log_file}'.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error while starting the default server: {exc}")
        return False


def describe_ollama_runtime_probe() -> dict[str, Any]:
    configured_cli = Path(OLLAMA_CLI).expanduser()
    configured_cli_exists = configured_cli.exists() and os.access(configured_cli, os.X_OK)
    resolved_cli = str(configured_cli) if configured_cli_exists else (shutil.which('ollama') or '')
    cli_detected = bool(resolved_cli)
    issues: list[str] = []
    if not cli_detected:
        issues.append('Ollama CLI not found.')
    elif not configured_cli_exists and resolved_cli:
        issues.append(
            f'Configured Ollama CLI path {configured_cli} is unavailable; using detected binary {resolved_cli}.'
        )
    return {
        'backend_id': 'ollama',
        'family': 'ollama',
        'variant': 'ollama',
        'label': 'Ollama',
        'runtime_state': 'runnable' if cli_detected else 'missing',
        'operations': {
            'discover': cli_detected,
            'list_models': cli_detected,
            'pull_model': cli_detected,
            'remove_model': cli_detected,
            'start_instance': cli_detected,
            'stop_instance': True,
        },
        'detection': {
            'configured_cli_path': str(configured_cli),
            'resolved_cli_path': resolved_cli or None,
            'cli_detected': cli_detected,
            'default_server_port': DEFAULT_SERVER_PORT,
            'default_server_listening': is_port_listening(DEFAULT_SERVER_PORT),
        },
        'issues': issues,
    }


def list_local_models() -> List[str]:
    return [entry["name"] for entry in list_local_model_entries()]


def prompt_model_selection(models: List[Dict[str, Any]]) -> List[str]:
    print("\nWhich models should be started as their own instance?")
    print("Enter numbers separated by spaces.")
    print("You can choose a model more than once (for example: '1 1 2').")
    label_width = min(34, max(len(format_model_label(str(model.get("name") or ""))) for model in models) + 2)
    for idx, model in enumerate(models, start=1):
        print(render_model_option(idx, model, label_width=label_width))
    choice = input("▶️  Selection: ").strip().split()

    selected_names: List[str] = []
    for token in choice:
        if token.isdigit():
            num = int(token)
            if 1 <= num <= len(models):
                selected_names.append(str(models[num - 1].get("name") or ""))
            else:
                print(f"⚠️  Invalid number: {num}")
        else:
            print(f"⚠️  Invalid input: {token}")
    return selected_names


def find_free_port(
    start: int = START_PORT,
    end: int = MAX_PORT,
    used_ports: Optional[set] = None,
) -> int:
    used = used_ports or set()
    for port in range(start, end):
        if port == DEFAULT_SERVER_PORT or port in used:
            continue
        if not is_port_listening(port):
            return port
    raise RuntimeError(f"No free ports found in the range {start}-{end}.")


def wait_for_model_loaded(port: int, model_name: str, timeout: int = 60, display_name: Optional[str] = None) -> bool:
    """Wait until the model is listed on the specific server via /api/ps."""
    shown_name = display_name or model_name
    print(f"⏳ Waiting for '{shown_name}' to load on port {port} (max {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/api/ps", timeout=2)
            if response.status_code == 200:
                running_data = response.json()
                if "models" in running_data:
                    for running_model in running_data["models"]:
                        if running_model.get("name", "").startswith(model_name):
                            print(f"✅ Model '{shown_name}' is now loaded on port {port}.")
                            return True
        except requests.exceptions.RequestException:
            pass
        except json.JSONDecodeError:
            pass
        time.sleep(2)
    print(f"❌ Model '{shown_name}' did not load on port {port} after {timeout}s (timeout).")
    return False


def wait_for_embedding_model_loaded(
    port: int,
    model_name: str,
    timeout: int = 60,
    display_name: Optional[str] = None,
) -> bool:
    shown_name = display_name or model_name
    print(f"⏳ Waiting for '{shown_name}' to load on port {port} via /api/embed (max {timeout}s)...")
    start_time = time.time()
    last_error = ''
    while time.time() - start_time < timeout:
        try:
            response = requests.post(
                f"http://127.0.0.1:{port}/api/embed",
                json={"model": model_name, "input": "ollmo embedding warmup"},
                timeout=MODEL_EMBED_WARMUP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                print(f"✅ Embedding model '{shown_name}' is now loaded on port {port}.")
                return True
            last_error = "Response did not include embeddings."
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
        except json.JSONDecodeError:
            last_error = "Invalid JSON response from /api/embed."
        time.sleep(2)

    if last_error:
        print(f"❌ Model '{shown_name}' did not load after {timeout}s: {last_error}")
    else:
        print(f"❌ Model '{shown_name}' did not load on port {port} after {timeout}s (timeout).")
    return False


def terminate_pid(pid: int, timeout: float = 3.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        print(f"❌ No permission to terminate PID {pid}: {exc}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_is_running(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        print(f"❌ No permission for kill -9 on PID {pid}: {exc}")
        return False

    time.sleep(0.2)
    return not pid_is_running(pid)


def wait_for_shutdown(pid: Optional[int], port: Optional[int], timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        port_closed = not (port and is_port_listening(port))
        pid_stopped = not (pid and pid_is_running(pid))
        if port_closed and pid_stopped:
            return True
        time.sleep(0.2)
    port_closed = not (port and is_port_listening(port))
    pid_stopped = not (pid and pid_is_running(pid))
    return port_closed and pid_stopped


def list_listening_pids(port: int) -> Tuple[List[int], Optional[str]]:
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
            env=sanitized_child_process_env(),
        )
    except FileNotFoundError:
        return [], "'lsof' was not found. Install 'lsof' to inspect ports."

    if result.returncode not in (0, 1):  # 1 == nothing found
        error = result.stderr.strip() or result.stdout.strip() or f"lsof exit code {result.returncode}"
        return [], error

    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids, None


def kill_processes_on_port(port: int) -> bool:
    """
    Terminates any process that is currently listening on `port`. Mirrors the
    checks performed inside stop_multi_models.sh so the API gets the same resiliency.
    """
    pids, error = list_listening_pids(port)
    if error:
        print(f"⚠️  {error}")
        return False

    if not pids:
        return True

    success = True
    for pid in pids:
        if not terminate_pid(pid):
            success = False
    return success
def start_model(
    model_name: str,
    used_ports: set,
    existing_instances: List[Dict],
    capability: Optional[str] = None,
    *,
    start_source: Optional[str] = None,
) -> Optional[Dict]:
    global instance_counters
    normalized_start_source = validate_start_source(
        start_source,
        context='ollama_start_model',
    )
    try:
        port = find_free_port(used_ports=used_ports)
    except RuntimeError as exc:
        print(f"❌ Error while finding a free port: {exc}")
        return None

    used_ports.add(port)
    instance_id = allocate_instance_id(model_name, existing_instances)
    display_label = format_model_label(model_name)
    resolved_capability = normalize_capability(capability) or infer_capability(model_name, "ollama")
    capability_badge = format_capability_badge(resolved_capability)

    print(f"\n🚀 Starting instance '{display_label}' ({capability_badge}) on port {port}...")
    env = build_ollama_env(port=port)
    safe_instance = safe_log_fragment(instance_id)
    safe_model = safe_log_fragment(model_name)
    log_file = LOG_DIR / f"ollama_server_{safe_instance}_{port}.log"
    serve_process = None
    load_process = None
    log_handle = None

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "w", encoding="utf-8")
        serve_process = subprocess.Popen(
            [OLLAMA_CLI, "serve"],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        time.sleep(2)
        if not is_port_listening(port):
            print(f"❌ The server for instance '{instance_id}' could not open port {port}.")
            raise RuntimeError("Server port is not active")

        show_metadata = _fetch_model_metadata(model_name, port=port)
        registry_metadata = build_registry_metadata(
            model_name,
            "ollama",
            resolved_capability,
            metadata={"backend_metadata": build_ollama_show_summary(show_metadata)},
        )

        print(f"   Server for '{display_label}' is listening on port {port}.")
        if resolved_capability == CAPABILITY_IMAGE_GENERATION:
            instance_info = {
                "instance_id": instance_id,
                "model": model_name,
                "port": port,
                "pid": serve_process.pid if serve_process else None,
                "log": str(log_file),
                **registry_metadata,
            }
            instance_info = attach_start_audit(
                instance_info,
                start_source=normalized_start_source,
                context='ollama_start_model',
                extra={'backend': 'ollama', 'capability': resolved_capability},
            )
            if log_handle and not log_handle.closed:
                try:
                    log_handle.close()
                except OSError:
                    pass
            print(f"✅ Running: {display_label} at http://127.0.0.1:{port}")
            print(f"   {capability_badge} | Instance: {instance_id} | Logs: {log_file}")
            return instance_info

        if resolved_capability == CAPABILITY_EMBEDDING:
            if wait_for_embedding_model_loaded(port, model_name, display_name=display_label):
                instance_info = {
                    "instance_id": instance_id,
                    "model": model_name,
                    "port": port,
                    "pid": serve_process.pid if serve_process else None,
                    "log": str(log_file),
                    **registry_metadata,
                }
                instance_info = attach_start_audit(
                    instance_info,
                    start_source=normalized_start_source,
                    context='ollama_start_model',
                    extra={'backend': 'ollama', 'capability': resolved_capability},
                )
                if log_handle and not log_handle.closed:
                    try:
                        log_handle.close()
                    except OSError:
                        pass
                print(f"✅ Running: {display_label} at http://127.0.0.1:{port}")
                print(f"   {capability_badge} | Instance: {instance_id} | Logs: {log_file}")
                return instance_info

            raise RuntimeError(f"Embedding model {model_name} did not load")

        print(f"   Loading '{display_label}' on this server...")
        load_log = LOG_DIR / f"{safe_model}_load_on_{port}_{safe_instance}.log"
        with open(load_log, "w", encoding="utf-8") as load_handle:
            load_process = subprocess.Popen(
                [OLLAMA_CLI, "run", model_name],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=load_handle,
                stderr=subprocess.STDOUT,
            )

        if wait_for_model_loaded(port, model_name, display_name=display_label):
            if load_process:
                try:
                    load_process.terminate()
                except ProcessLookupError:
                    pass
            instance_info = {
                "instance_id": instance_id,
                "model": model_name,
                "port": port,
                "pid": serve_process.pid if serve_process else None,
                "log": str(log_file),
                **registry_metadata,
            }
            instance_info = attach_start_audit(
                instance_info,
                start_source=normalized_start_source,
                context='ollama_start_model',
                extra={'backend': 'ollama', 'capability': resolved_capability},
            )
            if log_handle and not log_handle.closed:
                try:
                    log_handle.close()
                except OSError:
                    pass
            print(f"✅ Running: {display_label} at http://127.0.0.1:{port}")
            print(f"   {capability_badge} | Instance: {instance_id} | Logs: {log_file}")
            return instance_info

        raise RuntimeError(f"Model {model_name} did not load")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error while starting instance '{instance_id}': {exc}")
        if serve_process:
            serve_process.terminate()
            try:
                serve_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                serve_process.kill()
        if load_process:
            try:
                load_process.terminate()
            except ProcessLookupError:
                pass
        used_ports.discard(port)
        if log_handle:
            try:
                log_handle.close()
            except OSError:
                pass
        return None


def start_model_instance(
    model_name: str,
    capability: Optional[str] = None,
    *,
    start_source: Optional[str] = None,
) -> Optional[Dict]:
    normalized_start_source = validate_start_source(
        start_source,
        context='ollama_start_model_instance',
    )
    if not ensure_default_server_running():
        raise RuntimeError("Default Ollama server could not be started.")

    existing_instances = load_instances(prune=True)
    initialize_instance_counters(existing_instances)
    used_ports = {inst.get("port") for inst in existing_instances if inst.get("port")}
    current_instances = list(existing_instances)

    result = start_model(
        model_name,
        used_ports,
        current_instances,
        capability=capability,
        start_source=normalized_start_source,
    )
    if result:
        current_instances.append(result)
        write_config(current_instances)
    return result


def _stop_instance_process(instance: Dict) -> StopResult:
    instance_id = instance.get("instance_id")
    pid = instance.get("pid")
    port = instance.get("port")
    details: Dict[str, Any] = {
        "instance_id": instance_id,
        "port": port,
        "pid": pid,
        "attempts": [],
    }

    if pid:
        primary_success = terminate_pid(pid)
        details["attempts"].append(
            {"type": "pid", "value": pid, "success": primary_success}
        )
    else:
        details["attempts"].append({"type": "pid", "value": None, "success": None})

    port_attempt = {"type": "port", "value": port, "success": None}
    if port:
        port_attempt["was_listening"] = is_port_listening(port)
        port_attempt["success"] = kill_processes_on_port(port)
    details["attempts"].append(port_attempt)

    if wait_for_shutdown(pid, port, timeout=STOP_WAIT_SECONDS):
        return StopResult(
            state="stopped",
            message=f"Instance '{instance_id}' was stopped.",
            details=details,
        )

    # Fallback diagnostics mirroring stop_multi_models.sh
    remaining: Dict[str, Any] = {}
    if port:
        remaining["port_listening"] = is_port_listening(port)
        pids, error = list_listening_pids(port)
        remaining["pids"] = pids
        if error:
            remaining["error"] = error
    if pid:
        remaining["pid_running"] = pid_is_running(pid)
    details["remaining_after_wait"] = remaining

    fallback_success = False
    if port:
        fallback_success = kill_processes_on_port(port)
        details["fallback"] = {
            "port": port,
            "success": fallback_success,
            "timestamp": time.time(),
        }
    extended_wait = wait_for_shutdown(None, port, timeout=STOP_WAIT_EXTENSION)
    if extended_wait:
        return StopResult(
            state="stopped",
            message=f"Instance '{instance_id}' was stopped after a retry.",
            details=details,
        )

    if fallback_success:
        details["recommendation"] = "Wait a few seconds and check /api/running_instances again."
        return StopResult(
            state="stopping",
            message=f"Stop signal sent for '{instance_id}'; port {port} is still closing.",
            details=details,
        )

    details["recommendation"] = "Run './stop_multi_models.sh' to terminate stuck processes."
    return StopResult(
        state="failed",
        message=f"Instance '{instance_id}' could not be stopped.",
        details=details,
    )


def stop_model_instance(instance_id: str) -> Tuple[StopResult, Optional[Dict]]:
    instances = read_config()
    target = next((inst for inst in instances if inst.get("instance_id") == instance_id), None)
    if not target:
        print(f"ℹ️  Instance '{instance_id}' was not found or has already been stopped.")
        return StopResult(
            state="stopped",
            message=f"Instance '{instance_id}' was already removed.",
            details={"instance_id": instance_id, "action": "noop"},
        ), None

    print(f"\n🛑 Stopping instance '{instance_id}'...")
    result = _stop_instance_process(target)
    if result.state == "stopped":
        remaining = [inst for inst in instances if inst.get("instance_id") != instance_id]
        remaining = filter_running_instances(remaining)
        write_config(remaining)
    elif result.state == "failed":
        print(f"❌ Instance '{instance_id}' could not be stopped completely.")
    return result, target


def get_running_instances() -> List[Dict]:
    return list_runtime_entries(
        prune=True,
        path=CONFIG_FILE,
        sync_external=False,
    )


def get_available_models(include_limits: bool = False) -> List:
    if not ensure_default_server_running():
        return []
    models = list_local_model_entries()
    if not include_limits:
        return models
    enriched = []
    for entry in models:
        if not isinstance(entry, dict):
            enriched.append({"name": str(entry)})
            continue
        payload = {
            "name": entry.get("name"),
            "capability": entry.get("capability"),
        }
        if entry.get("provider_capabilities"):
            payload["provider_capabilities"] = entry.get("provider_capabilities")
        enriched.append(payload)
    return enriched


def pull_model(model_name: str) -> Tuple[bool, str]:
    if not model_name:
        return False, "No model name was provided."
    try:
        result = subprocess.run(
            [OLLAMA_CLI, "pull", model_name],
            capture_output=True,
            text=True,
            check=False,
            env=build_ollama_env(),
        )
        if result.returncode == 0:
            message = result.stdout.strip() or f"Model {model_name} was loaded successfully."
            return True, message
        error = result.stderr.strip() or result.stdout.strip() or f"Pull failed for {model_name}."
        return False, error
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def remove_model(model_name: str) -> Tuple[bool, str]:
    if not model_name:
        return False, "No model name was provided."
    try:
        result = subprocess.run(
            [OLLAMA_CLI, "rm", model_name],
            capture_output=True,
            text=True,
            check=False,
            env=build_ollama_env(),
        )
        if result.returncode == 0:
            message = result.stdout.strip() or f"Model {model_name} was removed."
            return True, message
        error = result.stderr.strip() or result.stdout.strip() or f"Removal failed for {model_name}."
        return False, error
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not ensure_default_server_running():
        print("❌ Aborting.")
        return

    existing_instances = load_instances(prune=True)
    initialize_instance_counters(existing_instances)
    used_ports = {inst.get("port") for inst in existing_instances if inst.get("port")}
    current_instances = list(existing_instances)

    available_model_entries = list_local_model_entries()
    if not available_model_entries:
        print("❌ No models are available to start.")
        return

    selected_model_names = prompt_model_selection(available_model_entries)
    if not selected_model_names:
        print("❌ No models were selected.")
        return

    started_servers: List[Dict] = []
    for model_to_start in selected_model_names:
        capability = next(
            (
                normalize_capability(item.get("capability"))
                for item in available_model_entries
                if str(item.get("name") or "").strip() == model_to_start
                and normalize_capability(item.get("capability"))
            ),
            None,
        ) or infer_capability(model_to_start, "ollama")
        server_info = start_model(
            model_to_start,
            used_ports,
            current_instances,
            capability=capability,
            start_source='startup_policy',
        )
        if server_info:
            started_servers.append(server_info)
            current_instances.append(server_info)
        else:
            print(f"⚠️  Error while starting an instance of '{model_to_start}'.")

    if started_servers:
        write_config(current_instances)
        print(
            f"\n✅ Successfully started {len(started_servers)} model server(s) and loaded the models. "
            f"Details are in '{CONFIG_FILE}'."
        )
        print("➡️  Continue in the UI: start or verify the Ollmo webserver on port 5001.")
    else:
        print("\n⚠️  No model servers could be started.")
        if not existing_instances:
            write_config([])


if __name__ == "__main__":
    main()
