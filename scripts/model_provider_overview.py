#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from typing import List, Dict, Any, Optional
import requests

# ------------ CONFIGURATION -------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PORTS_FILE = os.path.join(BASE_DIR, "model_ports.json")

# Enable/disable certain providers
ENABLE_OPENAI = True           # List OpenAI API models
ENABLE_OLLAMA_DEFAULT = True   # List models from the default Ollama server (11434)
ENABLE_MLX_SCAN = True         # Scan mlx/other OpenAI-compatible servers

# OpenAI API Key (read from environment)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Port configuration
OLLAMA_DEFAULT_PORT = 11434             # Default Ollama server
OLLAMA_INSTANCE_PORT_START = 11435      # Additional Ollama instances
OLLAMA_INSTANCE_PORT_END   = 11500

MLX_PORT_START = 11501                  # mlx-lm/OpenAI-compatible servers
MLX_PORT_END   = 11550

LOCAL_TIMEOUT = 0.5  # seconds for local HTTP checks
# ----------------------------------------

def safe_get(entry: Dict[str, Any], *keys, default=None):
    """Return the first non-None value for these keys from the dict, or default."""
    for key in keys:
        value = entry.get(key)
        if value is not None:
            return value
    return default

# ---------- OpenAI API Models ----------
def list_openai_models() -> List[Dict[str, Any]]:
    if not ENABLE_OPENAI:
        return []
    if not OPENAI_API_KEY:
        print("[OpenAI] Skipping: OPENAI_API_KEY not set.", file=sys.stderr)
        return []
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[OpenAI] Failed to list models: {e}", file=sys.stderr)
        return []
    models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        models.append({
            "provider": "openai",
            "endpoint": "https://api.openai.com",
            "port": None,
            "kind": "openai",
            "model": model_id,
            "source": "openai_api",
        })
    return models

# ----- Ollama default server on 11434 -----
def list_ollama_default() -> List[Dict[str, Any]]:
    if not ENABLE_OLLAMA_DEFAULT:
        return []
    base = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}"
    url = f"{base}/api/tags"
    try:
        resp = requests.get(url, timeout=LOCAL_TIMEOUT)
        if not resp.ok:
            return []
        data = resp.json()
    except Exception:
        return []
    entries = []
    for m in data.get("models", []):
        name = safe_get(m, "name", "model", default="unknown")
        entries.append({
            "provider": f"ollama:{OLLAMA_DEFAULT_PORT}",
            "endpoint": base,
            "port": OLLAMA_DEFAULT_PORT,
            "kind": "ollama_default",
            "model": name,
            "source": "ollama_default",
        })
    return entries

# ----- Additional Ollama instances from model_ports.json -----
def list_ollama_from_json() -> List[Dict[str, Any]]:
    if not os.path.isfile(MODEL_PORTS_FILE):
        print(f"[Ollama] {MODEL_PORTS_FILE} not found, skipping.", file=sys.stderr)
        return []
    try:
        with open(MODEL_PORTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Ollama] Failed to read {MODEL_PORTS_FILE}: {e}", file=sys.stderr)
        return []
    if isinstance(data, list):
        instances = data
    elif isinstance(data, dict):
        instances = data.get("instances") or data.get("servers") or data.get("models") or []
    else:
        print(f"[Ollama] Unexpected JSON structure in {MODEL_PORTS_FILE}", file=sys.stderr)
        return []
    out: List[Dict[str, Any]] = []
    for inst in instances:
        name = safe_get(inst, "instance_name", "name", "key", "id", default="unknown-instance")
        model = safe_get(inst, "model", "base_model", "model_name", default=name)
        port = safe_get(inst, "port", "listen_port", "server_port")
        if port is None:
            continue
        endpoint = f"http://127.0.0.1:{port}"
        out.append({
            "provider": f"ollama:{port}",
            "endpoint": endpoint,
            "port": port,
            "kind": "ollama",
            "model": model,
            "instance": name,
            "source": "model_ports.json",
        })
    return out

# ----- Scan MLX/OpenAI-compatible local servers -----
def scan_mlx_port(port: int) -> Optional[List[Dict[str, Any]]]:
    base = f"http://127.0.0.1:{port}"
    url = f"{base}/v1/models"
    try:
        resp = requests.get(url, timeout=LOCAL_TIMEOUT)
        if not resp.ok:
            return None
        data = resp.json()
    except Exception:
        return None
    entries = []
    for m in data.get("data", []):
        mid = m.get("id", "unknown-model")
        entries.append({
            "provider": f"mlx:{port}",
            "endpoint": base,
            "port": port,
            "kind": "openai_compatible",
            "model": mid,
            "source": "mlx_scan",
        })
    return entries or None

def list_mlx_models() -> List[Dict[str, Any]]:
    if not ENABLE_MLX_SCAN:
        return []
    all_mlx: List[Dict[str, Any]] = []
    for port in range(MLX_PORT_START, MLX_PORT_END + 1):
        entries = scan_mlx_port(port)
        if entries:
            all_mlx.extend(entries)
    return all_mlx

# ----- Utilities -----
def print_overview(entries: List[Dict[str, Any]]):
    if not entries:
        print("No models found from any provider.")
        return
    entries.sort(key=lambda e: (str(e["provider"]), str(e["model"])))
    provider_w = max(len(str(e["provider"])) for e in entries)
    endpoint_w = max(len(str(e["endpoint"])) for e in entries)
    kind_w = max(len(str(e["kind"])) for e in entries)
    print("\n=== Model Provider Overview ===\n")
    header = f"{'#':>3}  {'Provider':<{provider_w}}  {'Kind':<{kind_w}}  {'Endpoint':<{endpoint_w}}  Model"
    print(header)
    print("-" * len(header))
    for idx, e in enumerate(entries, start=1):
        line = (
            f"{idx:>3}  "
            f"{e['provider']:<{provider_w}}  "
            f"{e['kind']:<{kind_w}}  "
            f"{e['endpoint']:<{endpoint_w}}  "
            f"{e['model']}"
        )
        print(line)
    print()

def copy_to_clipboard(text: str):
    try:
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(input=text.encode("utf-8"))
    except Exception as e:
        print(f"Failed to copy to clipboard: {e}", file=sys.stderr)

def main():
    all_entries: List[Dict[str, Any]] = []
    # 1) OpenAI API models
    all_entries.extend(list_openai_models())
    # 2) Default Ollama server (11434)
    all_entries.extend(list_ollama_default())
    # 3) Additional Ollama instances from model_ports.json (11435-11500)
    all_entries.extend(list_ollama_from_json())
    # 4) mlx/OpenAI-compatible servers (11501-11550)
    all_entries.extend(list_mlx_models())
    print_overview(all_entries)
    if not all_entries:
        return
    try:
        choice = input("Select a model number to copy (or press Enter to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not choice:
        return
    try:
        idx = int(choice)
    except ValueError:
        print("Not a valid number.")
        return
    if idx < 1 or idx > len(all_entries):
        print("Number out of range.")
        return
    entry = all_entries[idx - 1]
    config = {
        "provider": entry["provider"],
        "endpoint": entry["endpoint"],
        "model": entry["model"],
        "kind": entry["kind"],
    }
    text = json.dumps(config, ensure_ascii=False)
    copy_to_clipboard(text)
    print("\nCopied to clipboard:")
    print(text)

if __name__ == "__main__":
    main()
