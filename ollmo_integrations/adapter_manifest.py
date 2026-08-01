"""Declarative adapter manifest metadata for external-client integration docks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class IntegrationAdapterManifest:
    integration_id: str
    manifest_id: str
    label: str
    status: str
    summary: str
    prerequisites: tuple[str, ...] = ()
    setup_steps: tuple[str, ...] = ()
    sync_commands: tuple[tuple[str, ...], ...] = ()
    unsync_commands: tuple[tuple[str, ...], ...] = ()
    managed_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'integration_id': self.integration_id,
            'manifest_id': self.manifest_id,
            'label': self.label,
            'status': self.status,
            'summary': self.summary,
            'prerequisites': list(self.prerequisites),
            'setup_steps': list(self.setup_steps),
            'sync_commands': [list(command) for command in self.sync_commands],
            'unsync_commands': [list(command) for command in self.unsync_commands],
            'managed_paths': list(self.managed_paths),
        }


_ADAPTER_MANIFESTS: dict[str, IntegrationAdapterManifest] = {
    'codex': IntegrationAdapterManifest(
        integration_id='codex',
        manifest_id='codex.local_provider_projection',
        label='Codex local provider projection',
        status='available',
        summary='Project Ollmo model registry entries into the local Codex config.',
        prerequisites=(
            'Codex is installed and configured on this machine.',
            'The active Ollmo checkout owns model_ports.json.',
        ),
        setup_steps=(
            'Start or inspect Ollmo through ./ollmo or scripts/ollmoctl.py.',
            'Run the provider sync command after model_ports.json changes.',
        ),
        sync_commands=(('python3', 'scripts/sync_model_providers.py'),),
        unsync_commands=(('python3', 'scripts/unsync_model_providers.py', '--integration', 'codex'),),
        managed_paths=('~/.codex/config.toml',),
    ),
}


def list_adapter_manifests() -> list[IntegrationAdapterManifest]:
    return [_ADAPTER_MANIFESTS[key] for key in sorted(_ADAPTER_MANIFESTS)]


def get_adapter_manifest(integration_id: str) -> IntegrationAdapterManifest:
    normalized = str(integration_id or '').strip().lower()
    if not normalized:
        raise ValueError('integration_id is required')
    manifest = _ADAPTER_MANIFESTS.get(normalized)
    if manifest is None:
        raise KeyError(f'Unknown integration adapter manifest: {integration_id}')
    return manifest


def build_adapter_manifest(integration_ids: Optional[Iterable[str]] = None) -> dict[str, Any]:
    if integration_ids is None:
        manifests = list_adapter_manifests()
    else:
        manifests = [get_adapter_manifest(integration_id) for integration_id in integration_ids]
    return {
        'kind': 'ollmo.integration_adapter_manifest',
        'adapters': [manifest.to_dict() for manifest in manifests],
    }
