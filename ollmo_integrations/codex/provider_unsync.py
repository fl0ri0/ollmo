"""Explicit Codex provider unsync helpers for Ollmo-managed local providers."""

from __future__ import annotations

from pathlib import Path

from ollmo_integrations.codex.config_sync import DEFAULT_CONFIG_PATH, strip_local_sections


def unsync_codex_config(*, config_path: Path | str | None = None) -> bool:
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        original = target.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f'ℹ️ Codex config was not found at {target}; nothing to unsync.')
        return False
    except OSError as exc:
        print(f'⚠️ Codex unsync skipped ({exc}).')
        return False

    trimmed = strip_local_sections(original).rstrip()
    payload = f'{trimmed}\n' if trimmed else ''
    normalized_original = original if not original or original.endswith('\n') else f'{original}\n'
    if payload == normalized_original:
        print(f'ℹ️ Codex bereits unsynced: {target}')
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding='utf-8')
        print(f'🔄 Unsynced Ollmo-managed Codex providers from {target}')
        return True
    except OSError as exc:
        print(f'⚠️ Unable to unsync {target}: {exc}')
        return False
