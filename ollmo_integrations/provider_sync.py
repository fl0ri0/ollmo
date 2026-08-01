#!/usr/bin/env python3
"""
Manual external-client provider sync based on model_ports.json.

This implementation stays under ollmo_integrations so external-client config
projection remains separate from general startup. The integration registry
defines which client adapters are currently supported.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_integrations.downstream_sync import sync_downstream_integrations

OLLMO_WEB_BASE = os.environ.get("OLLMO_WEB_BASE", "http://127.0.0.1:5001")
MODEL_PORTS_PATH = REPO_ROOT / "model_ports.json"


def _fetch_json(endpoint: str):
    try:
        req = Request(endpoint, headers={"Accept": "application/json"})
        with urlopen(req, timeout=1.5) as resp:
            payload = resp.read()
            if not payload:
                return None
            return json.loads(payload.decode("utf-8"))
    except (URLError, ValueError, OSError):
        return None


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_model_ports():
    if not MODEL_PORTS_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PORTS_PATH} was not found.")
    with MODEL_PORTS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if isinstance(data, list):
            return data
        return data.get("models") or data.get("instances") or []


def load_instances():
    # Authoritative: model_ports.json
    base_instances = _load_model_ports()

    # Optional enrichment from live webserver (available models + running metadata)
    running = _fetch_json(f"{OLLMO_WEB_BASE}/api/running_instances")
    available = _fetch_json(f"{OLLMO_WEB_BASE}/api/available_models")

    available_names: set[str] = set()
    if isinstance(available, dict):
        for item in available.get("models") or []:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = item
            if isinstance(name, str) and name:
                available_names.add(name)
    elif isinstance(available, list):
        for item in available:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = item
            if isinstance(name, str) and name:
                available_names.add(name)

    running_by_port: dict[int, dict] = {}
    if isinstance(running, list):
        for inst in running:
            if not isinstance(inst, dict):
                continue
            port = _coerce_int(inst.get("port"))
            if port:
                running_by_port[port] = inst

    merged: list[dict] = []
    for inst in base_instances:
        if not isinstance(inst, dict):
            continue
        merged_inst = dict(inst)
        port = _coerce_int(inst.get("port"))
        overlay = running_by_port.get(port)
        if isinstance(overlay, dict):
            for key, value in overlay.items():
                if key not in merged_inst or merged_inst[key] in (None, ""):
                    merged_inst[key] = value
        model_name = merged_inst.get("model")
        if available_names and isinstance(model_name, str):
            merged_inst["available"] = model_name in available_names
        merged.append(merged_inst)

    if not merged and isinstance(running, list) and running:
        merged = [inst for inst in running if isinstance(inst, dict)]
        print("⚠️ model_ports.json is empty; using running_instances as a fallback.")

    if running_by_port or available_names:
        print(
            f"ℹ️ Sync sources: model_ports.json ({len(merged)}), "
            f"running_instances={len(running_by_port)}, "
            f"available_models={len(available_names)}."
        )
    return merged


def main():
    instances = load_instances()
    summary = sync_downstream_integrations(instances)
    if summary.codex_changed:
        print(f"✅ Updated Codex config.toml from {MODEL_PORTS_PATH}.")
    else:
        print("ℹ️ Codex config.toml is already up to date or unchanged.")


if __name__ == "__main__":
    main()
