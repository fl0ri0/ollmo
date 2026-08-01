"""Canonical context IR helpers for history, memory, and reference promotion."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Optional

from ollmo_g.candidate_contracts import (
    build_candidate_graph,
    review_candidate_promotions,
)
from ollmo_services.artifact_contracts import sanitize_artifact_record

CONTEXT_IR_VERSION = 1

_CANDIDATE_STATUSES = {
    'candidate',
    'reserved',
    'possible',
    'draft',
    'optional',
    'not_promoted',
    'not-promoted',
    'unpromoted',
}
_DISCARDED_STATUSES = {'discarded', 'rejected', 'waived', 'not_needed', 'not-needed'}
_PROMOTED_STATUSES = {'promoted', 'active', 'claimed', 'selected', 'used'}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _safe_id_token(value: Any) -> str:
    token = _clean_text(value)
    if not token:
        return ''
    cleaned = ''.join(ch if ch.isalnum() else '-' for ch in token.lower())
    cleaned = '-'.join(part for part in cleaned.split('-') if part)
    return cleaned[:96]


def _digest_id(value: Mapping[str, Any]) -> str:
    text = repr(sorted((str(key), repr(item)) for key, item in value.items()))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _candidate_id(raw: Mapping[str, Any], *, source_kind: str, index: int) -> str:
    for key in (
        'candidate_id',
        'context_candidate_id',
        'reference_candidate_id',
        'memory_candidate_id',
    ):
        token = _clean_text(raw.get(key))
        if token:
            return token
    for key in (
        'artifact_ref',
        'ref',
        'message_id',
        'source_message_id',
        'memory_id',
        'id',
    ):
        token = _safe_id_token(raw.get(key))
        if token:
            return f'context-{source_kind}-{token}'
    path_token = _safe_id_token(raw.get('path') or raw.get('source_path') or raw.get('name'))
    if path_token:
        return f'context-{source_kind}-{path_token}'
    return f'context-{source_kind}-{index}-{_digest_id(raw)}'


def _source_kind(raw: Mapping[str, Any], default_kind: str) -> str:
    token = _clean_text(
        raw.get('source_kind')
        or raw.get('context_kind')
        or raw.get('candidate_kind')
        or raw.get('type')
        or raw.get('kind')
        or default_kind
    ).lower().replace('-', '_')
    if token in {'image', 'audio', 'document', 'pdf', 'text', 'message'}:
        return 'artifact' if token != 'message' else 'message'
    if token in {'memory', 'history', 'history_scan', 'reference', 'artifact', 'message', 'context'}:
        return token
    return default_kind


def _candidate_status(raw: Mapping[str, Any], *, promoted: bool = False) -> str:
    if promoted:
        return 'promoted'
    for key in ('status', 'candidate_status', 'context_state', 'context_status', 'promotion_status'):
        token = _clean_text(raw.get(key)).lower()
        if not token:
            continue
        if token in _PROMOTED_STATUSES:
            return 'promoted'
        if token in {'reserved'}:
            return 'reserved'
        if token in {'not_promoted', 'not-promoted', 'unpromoted'}:
            return 'not_promoted'
        if token in _DISCARDED_STATUSES:
            return 'discarded'
        if token in _CANDIDATE_STATUSES:
            return 'candidate'
    return 'candidate'


def _promotion_target(source_kind: str, raw: Mapping[str, Any]) -> str:
    explicit = _clean_text(raw.get('promotion_target') or raw.get('target')).lower().replace('-', '_')
    if explicit in {'active_context', 'active_reference', 'active_memory', 'history_scan'}:
        return explicit
    if source_kind == 'history_scan':
        return 'history_scan'
    if source_kind in {'artifact', 'message', 'reference'}:
        return 'active_reference'
    if source_kind == 'memory':
        return 'active_memory'
    return 'active_context'


def _is_promoted(raw: Mapping[str, Any], *, forced: bool = False) -> bool:
    if forced:
        return True
    for key in ('promoted', 'active', 'selected', 'used'):
        value = raw.get(key)
        if isinstance(value, bool) and value:
            return True
    status = _candidate_status(raw)
    if status == 'promoted':
        return True
    target = _clean_text(raw.get('promotion_target') or raw.get('target')).lower().replace('-', '_')
    return target in {'active_context', 'active_reference', 'active_memory', 'history_scan'}


def _normalized_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    artifact_kind = _clean_text(raw.get('type') or raw.get('kind')).lower() or None
    return sanitize_artifact_record(
        raw,
        default_kind=artifact_kind,
        default_origin='conversation_reference',
        include_content=False,
    ) or {}


def _normalize_candidate(
    raw: Mapping[str, Any],
    *,
    source: str,
    default_kind: str,
    index: int,
    promoted: bool = False,
) -> Optional[dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    source_kind = _source_kind(raw, default_kind)
    candidate = {
        'kind': 'ollmo.context_candidate',
        'candidate_id': _candidate_id(raw, source_kind=source_kind, index=index),
        'source_kind': source_kind,
        'status': _candidate_status(raw, promoted=promoted),
        'source': source,
        'promotion_policy': _clean_text(raw.get('promotion_policy')) or 'requires_current_turn_relevance',
    }
    for key in (
        'summary',
        'reason',
        'relevance',
        'memory_id',
        'message_id',
        'source_message_id',
        'source_response_id',
        'conversation_id',
        'role',
        'scan_scope',
        'scan_status',
        'scan_execution_status',
        'scan_policy',
        'source_surface',
        'history_instance_id',
        'source_updated_at',
    ):
        value = _clean_text(raw.get(key))
        if value:
            candidate[key] = value
    if isinstance(raw.get('score'), (int, float)):
        candidate['score'] = raw.get('score')
    for key in ('rank', 'relevance_score'):
        if isinstance(raw.get(key), (int, float)):
            candidate[key] = raw.get(key)
    scan_targets = [
        _clean_text(item)
        for item in (raw.get('scan_targets') or [])
        if _clean_text(item)
    ] if isinstance(raw.get('scan_targets'), list) else []
    if scan_targets:
        candidate['scan_targets'] = scan_targets
    for key in ('artifact_refs', 'match_terms'):
        values = [
            _clean_text(item)
            for item in (raw.get(key) or [])
            if _clean_text(item)
        ] if isinstance(raw.get(key), list) else []
        if values:
            candidate[key] = values[:12]
    if isinstance(raw.get('scan_result_count'), int):
        candidate['scan_result_count'] = raw.get('scan_result_count')
    artifact = _normalized_artifact(raw) if source_kind in {'artifact', 'message', 'reference'} else {}
    if artifact:
        candidate['artifact_ref'] = artifact.get('artifact_ref')
        candidate['artifact_type'] = artifact.get('type')
        for key in ('path', 'source_path', 'name', 'mime_type', 'origin'):
            value = artifact.get(key)
            if value not in (None, '', [], {}):
                candidate[key] = value
    return {key: value for key, value in candidate.items() if value not in (None, '', [], {})}


def _promotion_from_candidate(candidate: Mapping[str, Any], raw: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    candidate_id = _clean_text(candidate.get('candidate_id'))
    if not candidate_id:
        return None
    target = _promotion_target(_clean_text(candidate.get('source_kind')) or 'context', raw)
    promotion = {
        'kind': 'ollmo.context_candidate_promotion',
        'candidate_id': candidate_id,
        'target': target,
        'source': _clean_text(raw.get('promotion_source')) or _clean_text(candidate.get('source')),
        'reason': _clean_text(raw.get('promotion_reason') or raw.get('reason')),
        'artifact_ref': _clean_text(candidate.get('artifact_ref')),
        'memory_id': _clean_text(candidate.get('memory_id')),
        'message_id': _clean_text(candidate.get('message_id') or candidate.get('source_message_id')),
    }
    return {key: value for key, value in promotion.items() if value not in (None, '', [], {})}


def _candidate_values(payload: Mapping[str, Any], *, payload_label: str) -> list[tuple[Mapping[str, Any], str, str]]:
    values: list[tuple[Mapping[str, Any], str, str]] = []
    for key, default_kind in (
        ('context_candidates', 'context'),
        ('memory_candidates', 'memory'),
        ('history_candidates', 'history'),
        ('reference_candidates', 'reference'),
    ):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping):
                values.append((item, f'{payload_label}.{key}', default_kind))
    meta = payload.get('request_meta') if isinstance(payload.get('request_meta'), Mapping) else {}
    for key, default_kind in (
        ('context_candidates', 'context'),
        ('memory_candidates', 'memory'),
        ('history_candidates', 'history'),
    ):
        items = meta.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping):
                values.append((item, f'{payload_label}.request_meta.{key}', default_kind))
    return values


def _reference_values(payload: Mapping[str, Any], *, payload_label: str) -> list[tuple[Mapping[str, Any], str]]:
    values: list[tuple[Mapping[str, Any], str]] = []
    for key in ('reference_artifacts', 'selected_reference_artifacts'):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping):
                values.append((item, f'{payload_label}.{key}'))
    return values


def _context_strategy_candidate(payload: Mapping[str, Any], *, payload_label: str) -> Optional[dict[str, Any]]:
    strategy = payload.get('context_strategy') if isinstance(payload.get('context_strategy'), Mapping) else {}
    if not strategy:
        return None
    mode = _clean_text(strategy.get('mode')).lower()
    if not mode:
        return None
    reason = _clean_text(strategy.get('reason'))
    promoted_modes = {'recent_history', 'compressed_history', 'bounded_file_context'}
    is_promoted = mode in promoted_modes
    raw = {
        'candidate_id': f'context-strategy-{mode}',
        'source_kind': 'history' if mode != 'bounded_file_context' else 'context',
        'status': 'promoted' if is_promoted else 'not_promoted',
        'summary': reason or f'context strategy: {mode}',
        'reason': reason,
        'promotion_source': f'{payload_label}.context_strategy',
        'promotion_reason': reason,
    }
    if is_promoted:
        raw['promotion_target'] = 'active_context'
    return raw


def build_context_ir(
    *,
    request_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
    runtime_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return canonical context candidate and promotion state for a request."""

    candidates: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    context_gate_reviews: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    seen_promotions: set[tuple[str, str]] = set()

    def add_candidate(raw: Mapping[str, Any], *, source: str, default_kind: str, index: int, promoted: bool = False) -> None:
        candidate = _normalize_candidate(
            raw,
            source=source,
            default_kind=default_kind,
            index=index,
            promoted=promoted,
        )
        if not candidate:
            return
        candidate_id = _clean_text(candidate.get('candidate_id'))
        if not candidate_id:
            return
        if candidate_id in seen_candidate_ids:
            return
        seen_candidate_ids.add(candidate_id)
        candidates.append(candidate)
        if _is_promoted(raw, forced=promoted) or candidate.get('status') == 'promoted':
            promotion = _promotion_from_candidate(candidate, raw)
            if not promotion:
                return
            key = (_clean_text(promotion.get('candidate_id')), _clean_text(promotion.get('target')))
            if key in seen_promotions:
                return
            seen_promotions.add(key)
            promotions.append(promotion)

    payloads = [
        (request_payload or {}, 'request'),
        (response_payload or {}, 'response'),
        (runtime_payload or {}, 'runtime'),
    ]
    index = 0
    for payload, label in payloads:
        if not isinstance(payload, Mapping):
            continue
        existing_ir = payload.get('context_ir') if isinstance(payload.get('context_ir'), Mapping) else {}
        if existing_ir:
            for item in existing_ir.get('context_candidates') or []:
                if isinstance(item, Mapping):
                    index += 1
                    add_candidate(item, source=f'{label}.context_ir', default_kind='context', index=index)
        strategy_candidate = _context_strategy_candidate(payload, payload_label=label)
        if strategy_candidate:
            index += 1
            add_candidate(
                strategy_candidate,
                source=f'{label}.context_strategy',
                default_kind='context',
                index=index,
            )
        strategy = payload.get('context_strategy') if isinstance(payload.get('context_strategy'), Mapping) else {}
        context_gate_review = (
            strategy.get('context_gate_review')
            if isinstance(strategy.get('context_gate_review'), Mapping)
            else {}
        )
        if context_gate_review:
            context_gate_reviews.append(
                {
                    'source': f'{label}.context_strategy.context_gate_review',
                    **dict(context_gate_review),
                }
            )
        strategy_candidates = strategy.get('context_candidates') if isinstance(strategy.get('context_candidates'), list) else []
        for item in strategy_candidates:
            if isinstance(item, Mapping):
                index += 1
                add_candidate(
                    item,
                    source=f'{label}.context_strategy.context_candidates',
                    default_kind='context',
                    index=index,
                )
        for raw, source, default_kind in _candidate_values(payload, payload_label=label):
            index += 1
            add_candidate(raw, source=source, default_kind=default_kind, index=index)
        for raw, source in _reference_values(payload, payload_label=label):
            index += 1
            promoted_raw = {
                **dict(raw),
                'promotion_source': source,
                'promotion_reason': _clean_text(raw.get('promotion_reason')) or 'explicit_current_turn_reference',
            }
            add_candidate(promoted_raw, source=source, default_kind='reference', index=index, promoted=True)

    if not candidates and not promotions:
        return {}

    promoted_ids = [
        _clean_text(item.get('candidate_id'))
        for item in promotions
        if _clean_text(item.get('candidate_id'))
    ]
    active_reference_refs = [
        _clean_text(item.get('artifact_ref'))
        for item in promotions
        if _clean_text(item.get('target')) == 'active_reference' and _clean_text(item.get('artifact_ref'))
    ]
    active_memory_ids = [
        _clean_text(item.get('memory_id'))
        for item in promotions
        if _clean_text(item.get('target')) == 'active_memory' and _clean_text(item.get('memory_id'))
    ]
    promoted_history_scan_candidate_ids = [
        _clean_text(item.get('candidate_id'))
        for item in promotions
        if _clean_text(item.get('target')) == 'history_scan' and _clean_text(item.get('candidate_id'))
    ]
    candidate_graph = build_candidate_graph(
        context_candidates=candidates,
        promotions=promotions,
        intent_ref='context_intent',
        source='context_ir',
    )
    promotion_review = review_candidate_promotions(
        candidate_graph,
        existing_contracts={'source': 'context_ir.promotions'},
    )
    result = {
        'kind': 'ollmo.context_ir',
        'ir_version': CONTEXT_IR_VERSION,
        'context_candidates': candidates,
        'promotions': promotions,
        'candidate_graph': candidate_graph,
        'promotion_review': promotion_review,
        'candidate_count': len(candidates),
        'promotion_count': len(promotions),
        'promoted_candidate_ids': promoted_ids,
        'active_reference_artifact_refs': active_reference_refs,
        'active_memory_ids': active_memory_ids,
        'promoted_history_scan_candidate_ids': promoted_history_scan_candidate_ids,
    }
    if context_gate_reviews:
        result['context_gate_reviews'] = context_gate_reviews
        result['context_gate_review'] = context_gate_reviews[-1]
    return result
