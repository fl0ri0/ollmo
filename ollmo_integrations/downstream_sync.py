"""Shared downstream config sync helper for registered external clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ollmo_integrations.registry import integration_module

@dataclass
class DownstreamSyncSummary:
    codex_changed: bool = False


def sync_downstream_integrations(
    instances: Iterable[dict],
) -> DownstreamSyncSummary:
    summary = DownstreamSyncSummary()
    codex_sync = integration_module('codex', 'sync')

    try:
        summary.codex_changed = bool(codex_sync.sync_codex_config(instances))
    except Exception as exc:  # noqa: BLE001
        print(f'⚠️ Codex config sync skipped: {exc}')

    return summary
