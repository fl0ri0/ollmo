"""
Utility helpers to mirror running model ports into ~/.codex/config.toml.

Keeps [model_providers.local-*] sections in sync with the instances stored in
model_ports.json so UI-driven start/stop actions require no extra scripts.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import quote

DEFAULT_CONFIG_PATH = Path(os.path.expanduser("~/.codex/config.toml"))
MODEL_PROVIDER_HEADER = re.compile(
    r'^\s*\[model_providers\.(?P<label>"[^"]+"|\'[^\']+\'|[^\]]+)\]\s*$'
)
ANY_HEADER = re.compile(r'^\s*\[[^\]]+\]\s*$')


def normalize_backend(value: str | None) -> str:
    return (value or "ollama").lower()


def escape_toml(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )


def parse_provider_label(label: str) -> str:
    text = label.strip()
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]
    return text


def control_plane_base_url() -> str:
    return os.environ.get("OLLMO_WEB_BASE", "http://127.0.0.1:5001").rstrip("/")


def build_provider_base_url(instance: dict, port: int) -> str:
    instance_id = str(instance.get("instance_id") or "").strip()
    if instance_id:
        encoded = quote(instance_id, safe="")
        return f"{control_plane_base_url()}/api/local_provider/{encoded}/v1"
    return f"http://127.0.0.1:{port}/v1"


def strip_local_sections(text: str) -> str:
    if not text:
        return ""
    output: List[str] = []
    skip = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            match = MODEL_PROVIDER_HEADER.match(stripped)
            if match:
                provider = parse_provider_label(match.group("label"))
                if provider.startswith("local-"):
                    skip = True
                    continue
                skip = False
            else:
                skip = False
        if skip:
            continue
        output.append(line)
    return "".join(output).rstrip()


def build_provider_blocks(instances: Iterable[dict]) -> Tuple[str, int]:
    blocks: List[str] = []
    seen = set()
    count = 0
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("agent"):
            continue
        port = inst.get("port")
        try:
            port = int(port)
        except (TypeError, ValueError):
            continue
        provider_id = f"local-{port}"
        if provider_id in seen:
            continue
        seen.add(provider_id)
        backend = normalize_backend(inst.get("backend"))
        backend_label = "MLX" if backend == "mlx" else "Ollama"
        model_name = inst.get("model") or inst.get("instance_id") or provider_id
        provider_name = escape_toml(f"{model_name} ({backend_label} {port})")
        base_url = build_provider_base_url(inst, port)
        block_lines = [
            f'[model_providers.{provider_id}]',
            f'name = "{provider_name}"',
            f'type = "{backend}"',
            f'base_url = "{base_url}"',
            'wire_api = "responses"',
            ""
        ]
        blocks.append("\n".join(block_lines))
        count += 1
    return ("\n\n".join(blocks).rstrip(), count)


def sync_codex_config(instances: Iterable[dict], *, config_path: Path | str | None = None) -> bool:
    """
    Rewrite ~/.codex/config.toml so [model_providers.local-*] mirrors the provided instances.
    Never raises—logs warnings instead so lifecycle operations continue even if config writes fail.
    Returns True when the target file changed.
    """
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        original = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        original = ""
    except OSError as exc:
        print(f"⚠️ Codex config sync skipped ({exc}).")
        return False

    trimmed = strip_local_sections(original)
    blocks_text, provider_count = build_provider_blocks(instances)

    new_content = trimmed.rstrip()
    if provider_count:
        separator = "\n\n" if new_content else ""
        new_content = f"{new_content}{separator}{blocks_text}".rstrip()

    if new_content:
        new_content = f"{new_content}\n"
    normalized_original = original if not original or original.endswith("\n") else f"{original}\n"
    if new_content == normalized_original:
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        print(f"🔄 Synced {provider_count} local providers to {target}")
        return True
    except OSError as exc:
        print(f"⚠️ Unable to update {target}: {exc}")
        return False
