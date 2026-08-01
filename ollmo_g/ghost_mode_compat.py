"""API-edge compatibility for legacy `ghost_mode` hints.

This module intentionally lives outside `ollmo_g.semantic_roles.registry` so the
role library stays declarative and clean. Legacy modes select an initial role
set only; they do not route, execute, alter planner budgets, waive, supersede,
or freeze.
"""

from __future__ import annotations

from typing import Any

from ollmo_g.semantic_roles import semantic_role

DEFAULT_LEGACY_MODE = 'worker'
SUPPORTED_LEGACY_MODES = ('repair', 'worker', 'explorer', 'improviser')

LEGACY_MODE_TO_ROLE_IDS: dict[str, tuple[str, ...]] = {
    'repair': ('repairer', 'evidence_reasoner', 'doubt_challenger'),
    'worker': ('materializer', 'quality_reviewer', 'transition_committer'),
    'explorer': ('possibility_expander', 'structural_planner', 'integrator'),
    'improviser': ('possibility_expander', 'materializer', 'quality_reviewer'),
}

_LEGACY_MODE_ALIASES = {
    'build': 'worker',
    'builder': 'worker',
    'creative': 'improviser',
    'execute': 'worker',
    'fix': 'repair',
    'healer': 'repair',
    'improvise': 'improviser',
    'investigate': 'explorer',
    'research': 'explorer',
}


def _clean_id(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_').replace(' ', '_')


def normalize_legacy_mode(value: Any) -> str:
    return normalize_legacy_mode_hint(value) or DEFAULT_LEGACY_MODE


def normalize_legacy_mode_hint(value: Any) -> str | None:
    cleaned = _clean_id(value)
    if not cleaned:
        return None
    if cleaned in SUPPORTED_LEGACY_MODES:
        return cleaned
    return _LEGACY_MODE_ALIASES.get(cleaned)


def semantic_roles_for_legacy_mode(mode: Any) -> list[dict[str, Any]]:
    normalized_mode = normalize_legacy_mode(mode)
    roles: list[dict[str, Any]] = []
    for role_id in LEGACY_MODE_TO_ROLE_IDS[normalized_mode]:
        role = semantic_role(role_id)
        if role:
            roles.append(role)
    return roles


def build_legacy_mode_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for mode in SUPPORTED_LEGACY_MODES:
        roles = semantic_roles_for_legacy_mode(mode)
        catalog.append(
            {
                'mode': mode,
                'label': mode.title(),
                'summary': 'Compatibility API alias translated into semantic role orientation.',
                'compatibility_only': True,
                'semantic_role_ids': [role['role_id'] for role in roles],
                'semantic_roles': roles,
            }
        )
    return catalog
