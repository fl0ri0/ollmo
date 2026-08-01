#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = Path(os.path.expanduser("~/.codex/config.toml"))
MODEL_PORTS_PATH = REPO_ROOT / "model_ports.json"
EXPLICIT_STOPPED_PORTS_ENV = "OLLMO_STOPPED_PORTS"
SECTION_PATTERN = re.compile(
    r'^\s*\[model_providers\.(?P<label>"[^"]+"|\'[^\']+\'|[^\]]+)\]\s*$'
)


def load_ports() -> Set[int]:
    """Return ports from model_ports.json, tolerating both list and dict formats."""
    if not MODEL_PORTS_PATH.exists():
        return set()
    try:
        data = json.loads(MODEL_PORTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Konnte {MODEL_PORTS_PATH} nicht lesen: {exc}")
        return set()

    if isinstance(data, list):
        entries: Iterable = data
    elif isinstance(data, dict):
        entries = data.get("models") or data.get("instances") or []
        if not isinstance(entries, list):
            entries = []
    else:
        entries = []

    ports: Set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("agent"):
            continue
        port_value = entry.get("port")
        try:
            port = int(port_value)
        except (TypeError, ValueError):
            continue
        ports.add(port)
    return ports


def parse_provider_name(label: str) -> str:
    label = label.strip()
    if label.startswith(("'", '"')) and label.endswith(("'", '"')):
        return label[1:-1]
    return label


def extract_port(provider_name: str):
    match = re.match(r"local-(\d+)", provider_name)
    if match:
        return int(match.group(1))
    return None


def parse_ports_csv(raw: Optional[str]) -> Set[int]:
    ports: Set[int] = set()
    for chunk in str(raw or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            ports.add(int(token))
        except ValueError:
            continue
    return ports


def purge_inactive_sections(
    text: str,
    active_ports: Set[int],
    *,
    explicit_remove_ports: Optional[Set[int]] = None,
) -> Tuple[str, List[str]]:
    removed: List[str] = []
    output: List[str] = []
    skip_section = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        header_match = None
        if stripped.startswith("["):
            header_match = SECTION_PATTERN.match(stripped)

        if header_match:
            provider = parse_provider_name(header_match.group("label"))
            port = extract_port(provider)
            should_remove = False
            if port is not None:
                if explicit_remove_ports is not None:
                    should_remove = port in explicit_remove_ports
                else:
                    should_remove = port not in active_ports
            if should_remove:
                skip_section = True
                removed.append(provider)
                continue  # drop the header line and its block
            skip_section = False
            output.append(line)
        else:
            if skip_section:
                continue
            output.append(line)

    return "".join(output), removed


def cleanup_providers():
    if not CONFIG_PATH.exists():
        print(f"❌ config.toml was not found at {CONFIG_PATH}")
        return

    explicit_remove_ports = None
    if EXPLICIT_STOPPED_PORTS_ENV in os.environ:
        explicit_remove_ports = parse_ports_csv(os.environ.get(EXPLICIT_STOPPED_PORTS_ENV))
        print(f"Explicit stopped ports for cleanup: {sorted(explicit_remove_ports)}")
        ports_alive = set()
    else:
        ports_alive = load_ports()
        if ports_alive:
            print(f"Active ports according to model_ports.json: {sorted(ports_alive)}")
        else:
            print("Active ports according to model_ports.json: None detected")

    original = CONFIG_PATH.read_text(encoding="utf-8")
    updated, removed = purge_inactive_sections(
        original,
        ports_alive,
        explicit_remove_ports=explicit_remove_ports,
    )

    if removed:
        CONFIG_PATH.write_text(updated, encoding="utf-8")
        print("🧹 Removed inactive providers:", ", ".join(removed))
    else:
        print("No inactive providers to remove — config is already clean.")


if __name__ == "__main__":
    cleanup_providers()
