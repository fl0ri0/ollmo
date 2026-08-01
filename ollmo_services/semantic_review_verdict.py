"""Structured semantic review verdict normalization."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


SEMANTIC_REVIEW_VERDICT_KIND = 'ollmo.semantic_review_verdict'
SEMANTIC_REVIEW_VERDICT_VERSION = 1

VERDICT_PASSED = 'passed'
VERDICT_FAILED = 'failed'
VERDICT_UNCERTAIN = 'uncertain'

TRANSITION_TRUTHFUL_FREEZE = 'truthful_freeze'
TRANSITION_SEMANTIC_REVIEW = 'semantic_review'
TRANSITION_REPAIR_DEPENDENCY_CHAIN = 'repair_dependency_chain'
TRANSITION_REPAIR_BRANCH_CONTRACT = 'repair_branch_contract'
TRANSITION_REBUILD_FROM_PROMOTED_OBLIGATIONS = 'rebuild_from_promoted_obligations'
TRANSITION_WAIVE_WITH_EVIDENCE = 'waive_with_evidence'
TRANSITION_SUPERSEDE_WITH_REPLACEMENT_TRUTH = 'supersede_with_replacement_truth'
TRANSITION_CLARIFY = 'clarify'
TRANSITION_MANUAL_REVIEW = 'manual_review'

_STATUS_BY_VERDICT = {
    VERDICT_PASSED: 'fulfilled',
    VERDICT_FAILED: 'blocked',
    VERDICT_UNCERTAIN: 'pending',
}
_ALLOWED_TRANSITIONS = {
    TRANSITION_TRUTHFUL_FREEZE,
    TRANSITION_SEMANTIC_REVIEW,
    TRANSITION_REPAIR_DEPENDENCY_CHAIN,
    TRANSITION_REPAIR_BRANCH_CONTRACT,
    TRANSITION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
    TRANSITION_WAIVE_WITH_EVIDENCE,
    TRANSITION_SUPERSEDE_WITH_REPLACEMENT_TRUTH,
    TRANSITION_CLARIFY,
    TRANSITION_MANUAL_REVIEW,
}
_FENCED_JSON_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.IGNORECASE | re.DOTALL)
_HEADING_RE = re.compile(r'(?im)^\s*(?P<key>[a-z][a-z0-9_ -]{1,80})\s*:\s*(?P<value>.*)$')


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _compact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in dict(value or {}).items()
        if item not in (None, '', [], {})
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {'none', 'n/a', 'no', 'nothing'}:
            return []
        lines = [line.strip(' \t-*•') for line in text.splitlines()]
        cleaned = [line for line in lines if line and line.lower() not in {'none', 'n/a'}]
        return cleaned or [text]
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for raw_item in value:
            if isinstance(raw_item, Mapping):
                text = _clean_text(raw_item.get('ref') or raw_item.get('id') or raw_item.get('name') or raw_item)
            else:
                text = _clean_text(raw_item)
            if text and text.lower() not in {'none', 'n/a'} and text not in items:
                items.append(text)
        return items
    return []


def _normalize_token(value: Any) -> str:
    return re.sub(r'[^a-z0-9_]+', '_', _clean_text(value).lower()).strip('_')


def _normalize_verdict(value: Any) -> str:
    token = _normalize_token(value)
    if token in {
        'pass',
        'passed',
        'fulfilled',
        'complete',
        'completed',
        'ok',
        'satisfied',
        'truthful_freeze',
    }:
        return VERDICT_PASSED
    if token in {
        'fail',
        'failed',
        'blocked',
        'needs_repair',
        'repair_required',
        'wrong',
        'incorrect',
        'missing',
        'insufficient',
        'not_fulfilled',
        'not_satisfied',
    }:
        return VERDICT_FAILED
    if token in {
        'pending',
        'uncertain',
        'unknown',
        'needs_review',
        'needs_semantic_review',
        'needs_waiver_or_supersession',
        'clarify',
        'needs_clarification',
    }:
        return VERDICT_UNCERTAIN
    return ''


def _default_transition_for_verdict(verdict: str, *, parse_status: str) -> str:
    if verdict == VERDICT_PASSED:
        return TRANSITION_TRUTHFUL_FREEZE
    if parse_status == 'missing_structured_verdict':
        return TRANSITION_MANUAL_REVIEW
    return TRANSITION_SEMANTIC_REVIEW


def _normalize_transition(value: Any, *, verdict: str, parse_status: str) -> str:
    token = _normalize_token(value)
    if token in _ALLOWED_TRANSITIONS:
        return token
    if token in {'truthful_freeze_after_review', 'truthful_freeze_after_repair', 'freeze'}:
        return TRANSITION_TRUTHFUL_FREEZE
    if token in {'repair', 'needs_repair'}:
        return TRANSITION_SEMANTIC_REVIEW
    if token in {'waive', 'waiver'}:
        return TRANSITION_WAIVE_WITH_EVIDENCE
    if token in {'supersede', 'supersession'}:
        return TRANSITION_SUPERSEDE_WITH_REPLACEMENT_TRUTH
    return _default_transition_for_verdict(verdict, parse_status=parse_status)


def _normalize_criterion_status(value: Any) -> str:
    verdict = _normalize_verdict(value)
    if verdict == VERDICT_PASSED:
        return 'passed'
    if verdict == VERDICT_FAILED:
        return 'failed'
    return 'uncertain'


def _normalize_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _json_mapping_from_candidate(candidate: str) -> dict[str, Any]:
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _balanced_json_object(text: str) -> str:
    start = text.find('{')
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find('{', start + 1)
    return ''


def _json_mapping_from_text(text: str) -> tuple[dict[str, Any], str]:
    for match in _FENCED_JSON_RE.finditer(text):
        payload = _json_mapping_from_candidate(match.group(1))
        if payload:
            return payload, 'fenced_json'
    balanced = _balanced_json_object(text)
    if balanced:
        payload = _json_mapping_from_candidate(balanced)
        if payload:
            return payload, 'json'
    return {}, ''


def _legacy_heading_mapping_from_text(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for match in _HEADING_RE.finditer(text):
        key = _normalize_token(match.group('key'))
        value = match.group('value').strip()
        if key:
            values[key] = value
    if not values:
        return {}
    return {
        'verdict': values.get('verdict') or values.get('overall_status') or values.get('status'),
        'status': values.get('overall_status') or values.get('status'),
        'whole_intent_fit': values.get('whole_intent_fit'),
        'evidence_refs': values.get('evidence_used') or values.get('evidence_refs'),
        'defects': values.get('missing_or_wrong_work') or values.get('defects'),
        'recommended_transition': values.get('recommended_transition'),
    }


def _normalize_criterion_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    results: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            result = _compact_mapping(
                {
                    'criterion': _clean_text(item.get('criterion') or item.get('name')),
                    'status': _normalize_criterion_status(item.get('status') or item.get('verdict')),
                    'reason': _clean_text(item.get('reason') or item.get('rationale')),
                    'evidence_refs': _string_list(item.get('evidence_refs') or item.get('evidence')),
                }
            )
            if result.get('criterion'):
                results.append(result)
        else:
            text = _clean_text(item)
            if text:
                results.append({'criterion': text, 'status': 'uncertain'})
    return results


def normalize_semantic_review_verdict(
    value: Any,
    *,
    source_text: str = '',
    review_id: str = '',
    branch_id: str = '',
    phase_id: str = '',
) -> dict[str, Any]:
    """Normalize model semantic review output into a bounded advisory verdict."""

    raw: dict[str, Any]
    source_format = 'mapping'
    text = source_text if source_text else (value if isinstance(value, str) else '')
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw, source_format = _json_mapping_from_text(str(text or ''))
        if not raw:
            raw = _legacy_heading_mapping_from_text(str(text or ''))
            source_format = 'legacy_headings' if raw else ''

    parse_status = 'parsed' if raw else 'missing_structured_verdict'
    verdict = _normalize_verdict(
        raw.get('verdict')
        or raw.get('overall_status')
        or raw.get('status')
    ) if raw else VERDICT_UNCERTAIN
    if not verdict:
        verdict = VERDICT_UNCERTAIN
    transition = _normalize_transition(
        raw.get('recommended_transition')
        or raw.get('decision_action')
        or raw.get('action'),
        verdict=verdict,
        parse_status=parse_status,
    )
    confidence = _normalize_confidence(raw.get('confidence')) if raw else None
    evidence_refs = _string_list(
        raw.get('evidence_refs')
        or raw.get('evidence_used')
        or raw.get('runtime_evidence_refs')
    ) if raw else []
    defects = _string_list(
        raw.get('defects')
        or raw.get('missing_or_wrong_work')
        or raw.get('issues')
    ) if raw else []
    criterion_results = _normalize_criterion_results(
        raw.get('criterion_results')
        or raw.get('criteria_results')
        or raw.get('criteria')
    ) if raw else []
    payload = {
        'kind': SEMANTIC_REVIEW_VERDICT_KIND,
        'verdict_version': SEMANTIC_REVIEW_VERDICT_VERSION,
        'status': _STATUS_BY_VERDICT[verdict],
        'verdict': verdict,
        'passed': verdict == VERDICT_PASSED,
        'blocking': verdict != VERDICT_PASSED,
        'parse_status': parse_status,
        'source_format': source_format or None,
        'review_id': _clean_text(review_id),
        'branch_id': _clean_text(branch_id),
        'phase_id': _clean_text(phase_id),
        'confidence': confidence,
        'recommended_transition': transition,
        'whole_intent_fit': _clean_text(raw.get('whole_intent_fit') or raw.get('reason')) if raw else None,
        'criterion_results': criterion_results,
        'evidence_refs': evidence_refs,
        'defects': defects,
        'authority_boundary': (
            _clean_text(raw.get('authority_boundary')) if raw else ''
        ) or 'advisory_review_only_runtime_contracts_closure_decide_truth',
    }
    if parse_status == 'missing_structured_verdict':
        payload['reason'] = 'semantic review branch completed without a parseable verdict'
    return _compact_mapping(payload)


def semantic_review_verdict_from_text(
    text: Any,
    *,
    review_id: str = '',
    branch_id: str = '',
    phase_id: str = '',
) -> dict[str, Any]:
    return normalize_semantic_review_verdict(
        text,
        source_text=_clean_text(text),
        review_id=review_id,
        branch_id=branch_id,
        phase_id=phase_id,
    )
