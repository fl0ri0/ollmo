"""Integration registry for external-client adapter modules."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class IntegrationAdapter:
    integration_id: str
    sync_module: str | None = None
    cleanup_module: str | None = None
    unsync_module: str | None = None
    execution_module: str | None = None


INTEGRATION_ADAPTERS: dict[str, IntegrationAdapter] = {
    'codex': IntegrationAdapter(
        integration_id='codex',
        sync_module='ollmo_integrations.codex.config_sync',
        cleanup_module='ollmo_integrations.codex.provider_cleanup',
        unsync_module='ollmo_integrations.codex.provider_unsync',
        execution_module='ollmo_integrations.codex.execution',
    ),
}

NAMED_MODULES: dict[str, str] = {
    'provider_sync': 'ollmo_integrations.provider_sync',
    'provider_unsync': 'ollmo_integrations.provider_unsync',
    'downstream_sync': 'ollmo_integrations.downstream_sync',
    'adapter_manifest': 'ollmo_integrations.adapter_manifest',
}


def list_integration_adapters() -> list[IntegrationAdapter]:
    return [INTEGRATION_ADAPTERS[key] for key in sorted(INTEGRATION_ADAPTERS)]


def get_integration_adapter(integration_id: str) -> IntegrationAdapter:
    normalized = str(integration_id or '').strip().lower()
    if not normalized:
        raise ValueError('integration_id is required')
    adapter = INTEGRATION_ADAPTERS.get(normalized)
    if not adapter:
        raise KeyError(f'Unknown integration adapter: {integration_id}')
    return adapter


def integration_module(integration_id: str, module_role: str) -> ModuleType:
    adapter = get_integration_adapter(integration_id)
    normalized_role = str(module_role or '').strip().lower()
    module_path = {
        'sync': adapter.sync_module,
        'cleanup': adapter.cleanup_module,
        'unsync': adapter.unsync_module,
        'execution': adapter.execution_module,
    }.get(normalized_role)
    if not module_path:
        raise KeyError(f"Integration '{adapter.integration_id}' has no '{normalized_role}' module.")
    return import_module(module_path)


def named_module(module_name: str) -> ModuleType:
    normalized = str(module_name or '').strip().lower()
    module_path = NAMED_MODULES.get(normalized)
    if not module_path:
        raise KeyError(f'Unknown integration module: {module_name}')
    return import_module(module_path)
