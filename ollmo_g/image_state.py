"""Compact image-state parsing and reuse helpers for Ghost."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCED_JSON_RE = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r'(\{[\s\S]*\})')


def _clip(value: Any, *, max_chars: int = 180) -> str:
    text = str(value or '').strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + '...[truncated]'


def _extract_json_payload(raw: str) -> Optional[dict[str, Any]]:
    text = str(raw or '').strip()
    if not text:
        return None
    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    generic = _JSON_OBJECT_RE.search(text)
    if generic:
        try:
            parsed = json.loads(generic.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_image_state(
    raw_state: Any,
    *,
    fallback_summary: Optional[str] = None,
    describer_instance_id: Optional[str] = None,
    describer_model: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    source = raw_state if isinstance(raw_state, dict) else {}
    summary = _clip(source.get('summary') or fallback_summary or '', max_chars=180)
    subject = _clip(source.get('subject') or '', max_chars=100)
    scene = _clip(source.get('scene') or '', max_chars=120)
    style = _clip(source.get('style') or '', max_chars=120)
    key_elements = []
    for item in source.get('key_elements') or []:
        token = _clip(item, max_chars=80)
        if token and token not in key_elements:
            key_elements.append(token)
        if len(key_elements) >= 6:
            break
    if not summary:
        parts = [token for token in (subject, scene, style) if token]
        if key_elements:
            parts.append('Elements: ' + ', '.join(key_elements[:4]))
        summary = _clip('. '.join(parts), max_chars=180)
    if not any((summary, subject, scene, style, key_elements)):
        return None
    payload: dict[str, Any] = {
        'summary': summary or None,
        'subject': subject or None,
        'scene': scene or None,
        'style': style or None,
        'key_elements': key_elements,
    }
    if describer_instance_id:
        payload['describer_instance_id'] = str(describer_instance_id).strip()
    if describer_model:
        payload['describer_model'] = str(describer_model).strip()
    return payload


def parse_image_state_response(
    raw_text: str,
    *,
    describer_instance_id: Optional[str] = None,
    describer_model: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    text = str(raw_text or '').strip()
    if not text:
        return None
    parsed = _extract_json_payload(text)
    if isinstance(parsed, dict):
        return normalize_image_state(
            parsed,
            describer_instance_id=describer_instance_id,
            describer_model=describer_model,
        )
    return normalize_image_state(
        {},
        fallback_summary=text,
        describer_instance_id=describer_instance_id,
        describer_model=describer_model,
    )


def image_state_anchor(image_state: Any) -> Optional[str]:
    if not isinstance(image_state, dict):
        return None
    for key in ('summary', 'subject', 'scene'):
        token = _clip(image_state.get(key) or '', max_chars=180).rstrip(' ,.;:')
        if token:
            return token
    elements = [
        str(item or '').strip().rstrip(' ,.;:')
        for item in (image_state.get('key_elements') or [])
        if str(item or '').strip()
    ]
    if elements:
        return _clip(', '.join(elements[:4]), max_chars=180)
    return None
