"""Declarative semantic role registry.

Every `*.md` file in this package that contains a valid first fenced JSON block
with `kind: "ollmo.semantic_role"` becomes a role. This file does not define
legacy Ghost modes, compatibility aliases, or route policy.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROLE_DIR = Path(__file__).resolve().parent
_ROLE_BLOCK_RE = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)


def _clean_id(value: Any) -> str:
    text = str(value or '').strip().lower()
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss',
        '-': '_',
        ' ': '_',
        '/': '_',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return '_'.join(part for part in text.split('_') if part)


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return []
    result: list[str] = []
    for item in values:
        text = str(item or '').strip()
        if text and text not in result:
            result.append(text)
    return result


def _role_file_payload(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None
    match = _ROLE_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('kind') != 'ollmo.semantic_role':
        return None
    role_id = _clean_id(payload.get('role_id'))
    if not role_id:
        return None
    normalized = dict(payload)
    normalized['role_id'] = role_id
    normalized.setdefault('source_kind', 'semantic_role')
    normalized.setdefault('source_file', path.name)
    for key in (
        'activation_terms',
        'related_lenses',
        'movement_axes',
        'allowed_advisory_actions',
        'failure_modes',
        'evidence_requirements',
        'focus_questions',
    ):
        normalized[key] = _clean_string_list(normalized.get(key))
    return normalized


@lru_cache(maxsize=1)
def _role_definitions() -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for path in sorted(_ROLE_DIR.glob('*.md')):
        payload = _role_file_payload(path)
        if not payload:
            continue
        definitions[payload['role_id']] = payload
    return definitions


@lru_cache(maxsize=1)
def _lookup() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for role_id, definition in _role_definitions().items():
        aliases[_clean_id(role_id)] = role_id
        aliases[_clean_id(definition.get('name'))] = role_id
        for term in definition.get('activation_terms') or ():
            aliases[_clean_id(term)] = role_id
    return aliases


def normalize_semantic_role_id(value: Any) -> str | None:
    cleaned = _clean_id(value)
    if not cleaned:
        return None
    return _lookup().get(cleaned)


def semantic_role(role_id: Any) -> dict[str, Any] | None:
    normalized = normalize_semantic_role_id(role_id)
    if normalized is None:
        return None
    return deepcopy(_role_definitions()[normalized])


def semantic_role_for_lens(lens: Any) -> dict[str, Any] | None:
    return semantic_role(lens)


def build_semantic_role_catalog() -> list[dict[str, Any]]:
    definitions = _role_definitions()
    return [deepcopy(definitions[role_id]) for role_id in sorted(definitions)]
