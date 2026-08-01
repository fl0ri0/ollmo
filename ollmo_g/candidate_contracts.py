"""General candidate and promotion contract helpers.

Candidates are visible possibilities. Promotion review is the only transition
that turns them into owed work, active context, or another runtime contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from helpers.model_capabilities import normalize_capability

CANDIDATE_CONTRACTS_VERSION = 1

_PROMOTED_STATUSES = {
    'active',
    'accepted',
    'claimed',
    'fulfilled',
    'completed',
    'promoted',
    'promoted_to_obligation',
    'promotion_accepted',
    'selected',
    'used',
}
_WAIVED_STATUSES = {
    'not-needed',
    'not_needed',
    'not_needed_verified',
    'skipped_verified',
    'unnecessary_verified',
    'waived',
}
_SUPERSEDED_STATUSES = {
    'obsolete',
    'replaced',
    'superseded',
    'no-longer-relevant',
    'no_longer_relevant',
}
_REJECTED_STATUSES = {
    'discarded',
    'error',
    'failed',
    'invalid',
    'rejected',
}
_OMITTED_STATUSES = {
    'not-promoted',
    'not_promoted',
    'omitted',
    'unpromoted',
}
_RESERVED_STATUSES = {
    'candidate',
    'draft',
    'optional',
    'possible',
    'reserved',
}
_RECONSIDERABLE_STATUSES = {
    'candidate',
    'omitted',
    'reserved',
    'stale',
}
_VALID_CANDIDATE_TYPES = {
    'context',
    'continuation',
    'evidence',
    'learning',
    'memory',
    'output',
    'reference',
    'repair',
    'workload_task',
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _clip_text(value: Any, *, max_chars: int = 240) -> str:
    text = _clean_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + '...'


def _safe_token(value: Any, *, fallback: str = 'item') -> str:
    text = _clean_text(value)
    if not text:
        text = fallback
    token = ''.join(ch if ch.isalnum() else '-' for ch in text.lower())
    token = '-'.join(part for part in token.split('-') if part)
    return token[:96] or fallback


def _digest(value: Mapping[str, Any]) -> str:
    text = repr(sorted((str(key), repr(item)) for key, item in value.items()))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _clean_string_list(value: Any, *, limit: int = 16, max_chars: int = 160) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        text = _clip_text(raw_item, max_chars=max_chars)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _clean_text(raw_key)
            if not key or raw_value in (None, '', [], {}):
                continue
            payload[key] = _json_safe(raw_value)
        return payload
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value if item not in (None, '', [], {})]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _candidate_type(raw: Mapping[str, Any], explicit: Optional[str]) -> str:
    token = _clean_text(
        explicit
        or raw.get('candidate_type')
        or raw.get('contract_type')
        or raw.get('source_kind')
        or raw.get('promotion_target')
    ).lower().replace('-', '_')
    if token in {'active_context', 'history', 'history_scan', 'message'}:
        return 'context'
    if token in {'active_reference', 'artifact', 'selected_reference'}:
        return 'reference'
    if token in {'active_memory'}:
        return 'memory'
    if token in _VALID_CANDIDATE_TYPES:
        return token
    if _clean_text(raw.get('obligation_id') or raw.get('output_type')):
        return 'output'
    if _clean_text(raw.get('task_id') or raw.get('workload_task_id')):
        return 'workload_task'
    if _clean_text(raw.get('artifact_ref') or raw.get('path')):
        return 'reference'
    return 'context'


def _candidate_status(raw: Mapping[str, Any], *, promoted: bool = False) -> str:
    for key in (
        'status',
        'candidate_status',
        'contract_state',
        'contract_status',
        'obligation_state',
        'intent_state',
        'promotion_status',
    ):
        token = _clean_text(raw.get(key)).lower()
        if not token:
            continue
        if token in _SUPERSEDED_STATUSES:
            return 'superseded'
        if token in _PROMOTED_STATUSES:
            return 'promoted'
        if token in _WAIVED_STATUSES:
            return 'waived'
        if token in _REJECTED_STATUSES:
            return 'rejected'
        if token in _OMITTED_STATUSES:
            return 'omitted'
        if token in {'stale'}:
            return 'stale'
        if token in _RESERVED_STATUSES:
            return 'candidate' if token != 'reserved' else 'reserved'
    if promoted:
        return 'promoted'
    if _clean_text(raw.get('contract_ref') or raw.get('promoted_obligation_id')):
        return 'promoted'
    return 'candidate'


def _candidate_id(
    raw: Mapping[str, Any],
    *,
    candidate_type: str,
    source: str,
    index: int,
) -> str:
    for key in (
        'candidate_id',
        'context_candidate_id',
        'reference_candidate_id',
        'memory_candidate_id',
        'repair_candidate_id',
    ):
        token = _clean_text(raw.get(key))
        if token:
            return token
    if candidate_type == 'output':
        for key in ('promoted_from_candidate_id', 'obligation_id', 'phase_id', 'branch_id'):
            token = _clean_text(raw.get(key))
            if token:
                return f'candidate-output-{_safe_token(token)}'
    if candidate_type == 'workload_task':
        token = _clean_text(raw.get('task_id') or raw.get('workload_task_id') or raw.get('phase_id'))
        if token:
            return f'candidate-workload-{_safe_token(token)}'
    for key in (
        'artifact_ref',
        'ref',
        'message_id',
        'source_message_id',
        'memory_id',
        'id',
        'phase_id',
        'branch_id',
    ):
        token = _clean_text(raw.get(key))
        if token:
            return f'candidate-{candidate_type}-{_safe_token(token)}'
    return f'candidate-{candidate_type}-{_safe_token(source)}-{index}-{_digest(raw)}'


def _contract_ref(raw: Mapping[str, Any], *, candidate_type: str) -> str:
    for key in (
        'contract_ref',
        'promoted_contract_ref',
        'promoted_obligation_id',
        'obligation_id',
    ):
        token = _clean_text(raw.get(key))
        if token:
            return token
    if candidate_type == 'workload_task':
        task_id = _clean_text(raw.get('task_id') or raw.get('workload_task_id'))
        if task_id and _candidate_status(raw) == 'promoted':
            return task_id
    target = _clean_text(raw.get('promotion_target') or raw.get('target'))
    if target and _candidate_status(raw) == 'promoted':
        return target
    return ''


def normalize_candidate(
    raw: Mapping[str, Any],
    *,
    source: str,
    intent_ref: Optional[str] = None,
    candidate_type: Optional[str] = None,
    index: int = 1,
    promoted: bool = False,
) -> dict[str, Any]:
    """Normalize one current Ollmo candidate-like record."""

    if not isinstance(raw, Mapping):
        return {}
    normalized_type = _candidate_type(raw, candidate_type)
    status = _candidate_status(raw, promoted=promoted)
    payload: dict[str, Any] = {
        'kind': 'ollmo.candidate',
        'candidate_id': _candidate_id(
            raw,
            candidate_type=normalized_type,
            source=source,
            index=index,
        ),
        'candidate_type': normalized_type,
        'source': _clean_text(raw.get('source')) or source,
        'intent_ref': _clean_text(raw.get('intent_ref')) or _clean_text(intent_ref),
        'status': status,
        'promotion_policy': (
            _clean_text(raw.get('promotion_policy'))
            or (
                'runtime_required'
                if status == 'promoted'
                else 'closed_superseded'
                if status == 'superseded'
                else 'closed_by_review'
                if status in {'rejected', 'waived'}
                else 'requires_review'
            )
        ),
    }
    if status in _RECONSIDERABLE_STATUSES:
        payload['reconsiderable'] = True
        payload['execution_policy'] = (
            _clean_text(raw.get('execution_policy'))
            or 'non_executable_until_promoted'
        )
        payload['reconsideration_policy'] = (
            _clean_text(raw.get('reconsideration_policy'))
            or 'review_again_when_context_or_intent_changes'
        )
    elif status == 'promoted':
        payload['execution_policy'] = (
            _clean_text(raw.get('execution_policy'))
            or 'executable_obligation'
        )
    elif status in {'rejected', 'superseded', 'waived'}:
        payload['terminal'] = True
        if status == 'superseded':
            payload['supersession_policy'] = (
                _clean_text(raw.get('supersession_policy'))
                or 'closed_by_current_runtime_truth'
            )
    for key in (
        'phase_id',
        'branch_id',
        'task_id',
        'obligation_id',
        'artifact_ref',
        'memory_id',
        'message_id',
        'source_message_id',
        'source_response_id',
        'promotion_source',
        'superseded_by',
        'superseded_by_candidate_id',
        'superseded_by_obligation_id',
    ):
        value = _clean_text(raw.get(key))
        if value:
            payload[key] = value
    target_ref = _clean_text(
        raw.get('target_ref')
        or raw.get('promotion_target')
        or raw.get('target')
        or raw.get('artifact_ref')
        or raw.get('memory_id')
    )
    if target_ref:
        payload['target_ref'] = target_ref
    capability = normalize_capability(raw.get('capability'))
    if capability:
        payload['capability'] = capability
    output_type = _clean_text(raw.get('output_type') or raw.get('type')).lower()
    if output_type:
        payload['output_type'] = output_type
    contract_ref = _contract_ref(raw, candidate_type=normalized_type)
    if contract_ref:
        payload['contract_ref'] = contract_ref
    for key in (
        'summary',
        'reason',
        'promotion_reason',
        'waiver_reason',
        'supersession_reason',
        'relevance',
        'objective',
        'deliverable',
        'rationale',
        'advisory_role',
        'decision_notes',
    ):
        value = _clip_text(raw.get(key), max_chars=320)
        if value:
            payload[key] = value
    for key in ('priority', 'relevance_score', 'score', 'rank'):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            payload[key] = value
    for key in (
        'depends_on',
        'evidence_refs',
        'input_refs',
        'match_terms',
        'artifact_refs',
        'evidence_requirements',
        'reconsideration_triggers',
        'semantic_review_criteria',
        'learning_hint_refs',
    ):
        values = _clean_string_list(raw.get(key), limit=16, max_chars=160)
        if values:
            payload[key] = values
    for key in ('promotion_suggestions', 'waiver_candidates', 'repair_candidates', 'supersession_candidates'):
        value = raw.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            payload[key] = _json_safe(value)
    related_candidate_id = _clean_text(raw.get('related_candidate_id'))
    if related_candidate_id:
        payload['related_candidate_id'] = related_candidate_id
    return _json_safe(payload)


def _merge_candidate(existing: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    status_rank = {
        'superseded': 6,
        'rejected': 5,
        'waived': 4,
        'promoted': 3,
        'reserved': 2,
        'candidate': 1,
        'omitted': 1,
        'stale': 0,
    }
    merged = dict(existing)
    incoming_status = _clean_text(incoming.get('status')).lower()
    existing_status = _clean_text(existing.get('status')).lower()
    if status_rank.get(incoming_status, 0) >= status_rank.get(existing_status, 0):
        merged['status'] = incoming_status or existing_status or 'candidate'
        if merged['status'] == 'promoted':
            merged.pop('reconsiderable', None)
            merged.pop('reconsideration_policy', None)
            merged['execution_policy'] = 'executable_obligation'
        elif merged['status'] in _RECONSIDERABLE_STATUSES:
            merged['reconsiderable'] = True
            merged.setdefault('execution_policy', 'non_executable_until_promoted')
            merged.setdefault('reconsideration_policy', 'review_again_when_context_or_intent_changes')
        elif merged['status'] in {'rejected', 'superseded', 'waived'}:
            merged.pop('reconsiderable', None)
            merged.pop('reconsideration_policy', None)
            merged['terminal'] = True
            if merged['status'] == 'superseded':
                merged.setdefault('supersession_policy', 'closed_by_current_runtime_truth')
    for key, value in incoming.items():
        if value in (None, '', [], {}) or key in {'kind', 'candidate_id', 'candidate_type', 'status'}:
            continue
        if key not in merged or merged.get(key) in (None, '', [], {}):
            merged[key] = _json_safe(value)
            continue
        if key in {
            'depends_on',
            'evidence_refs',
            'input_refs',
            'match_terms',
            'artifact_refs',
            'evidence_requirements',
            'reconsideration_triggers',
            'semantic_review_criteria',
            'learning_hint_refs',
        }:
            merged[key] = _clean_string_list([*(merged.get(key) or []), *(value or [])], limit=24)
        elif key in {'promotion_suggestions', 'waiver_candidates', 'repair_candidates', 'supersession_candidates'}:
            existing_items = [
                _json_safe(item)
                for item in (merged.get(key) or [])
                if item not in (None, '', [], {})
            ]
            incoming_items = [
                _json_safe(item)
                for item in (value or [])
                if item not in (None, '', [], {})
            ]
            merged_items: list[Any] = []
            for item in [*existing_items, *incoming_items]:
                if item not in merged_items:
                    merged_items.append(item)
            merged[key] = merged_items
    return _json_safe(merged)


def _push_candidate(
    values: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    raw: Mapping[str, Any],
    *,
    source: str,
    intent_ref: str,
    candidate_type: str,
    index: int,
    promoted: bool = False,
    candidate_id: Optional[str] = None,
) -> None:
    raw_payload = dict(raw)
    if candidate_id:
        raw_payload['candidate_id'] = candidate_id
    candidate = normalize_candidate(
        raw_payload,
        source=source,
        intent_ref=intent_ref,
        candidate_type=candidate_type,
        index=index,
        promoted=promoted,
    )
    candidate_id_value = _clean_text(candidate.get('candidate_id'))
    if not candidate_id_value:
        return
    if candidate_id_value in by_id:
        by_id[candidate_id_value] = _merge_candidate(by_id[candidate_id_value], candidate)
        for offset, item in enumerate(values):
            if _clean_text(item.get('candidate_id')) == candidate_id_value:
                values[offset] = by_id[candidate_id_value]
                break
        return
    by_id[candidate_id_value] = candidate
    values.append(candidate)


def build_candidate_graph(
    *,
    candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    output_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    output_obligations: Optional[Sequence[Mapping[str, Any]]] = None,
    workload_tasks: Optional[Sequence[Mapping[str, Any]]] = None,
    context_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    promotions: Optional[Sequence[Mapping[str, Any]]] = None,
    workload_proposal_review: Optional[Mapping[str, Any]] = None,
    intent_ref: Optional[str] = None,
    source: str = 'candidate_contracts',
) -> dict[str, Any]:
    """Build a normalized graph from current candidate-like runtime surfaces."""

    values: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    intent = _clean_text(intent_ref) or 'intent_anchor'
    index = 0

    for item in output_candidates or []:
        if isinstance(item, Mapping):
            index += 1
            _push_candidate(
                values,
                by_id,
                item,
                source='request_ir.output_candidates',
                intent_ref=intent,
                candidate_type='output',
                index=index,
            )
    for item in output_obligations or []:
        if not isinstance(item, Mapping):
            continue
        index += 1
        candidate_id = _clean_text(item.get('promoted_from_candidate_id'))
        _push_candidate(
            values,
            by_id,
            item,
            source='request_ir.output_obligations',
            intent_ref=intent,
            candidate_type='output',
            index=index,
            promoted=True,
            candidate_id=candidate_id or None,
        )
    for item in workload_tasks or []:
        if not isinstance(item, Mapping):
            continue
        index += 1
        task_id = _clean_text(item.get('task_id') or item.get('workload_task_id') or item.get('phase_id'))
        task_payload = dict(item)
        if _clean_text(item.get('candidate_id')):
            task_payload['related_candidate_id'] = _clean_text(item.get('candidate_id'))
        if task_id:
            task_payload['candidate_id'] = f'candidate-workload-{_safe_token(task_id)}'
        _push_candidate(
            values,
            by_id,
            task_payload,
            source='workload_graph.tasks',
            intent_ref=intent,
            candidate_type='workload_task',
            index=index,
        )
    for item in context_candidates or []:
        if isinstance(item, Mapping):
            index += 1
            _push_candidate(
                values,
                by_id,
                item,
                source='context_ir.context_candidates',
                intent_ref=intent,
                candidate_type=None,
                index=index,
            )
    for item in candidates or []:
        if isinstance(item, Mapping):
            index += 1
            _push_candidate(
                values,
                by_id,
                item,
                source=source,
                intent_ref=intent,
                candidate_type=None,
                index=index,
            )
    proposal_review = workload_proposal_review if isinstance(workload_proposal_review, Mapping) else {}
    for item in proposal_review.get('rejections') or []:
        if not isinstance(item, Mapping):
            continue
        index += 1
        proposal_id = _clean_text(item.get('proposal_id') or item.get('target')) or f'proposal-{index}'
        _push_candidate(
            values,
            by_id,
            {
                'candidate_id': f'candidate-rejected-workload-proposal-{_safe_token(proposal_id)}',
                'candidate_type': 'workload_task',
                'status': 'rejected',
                'reason': item.get('reason'),
                'target_ref': item.get('target'),
                'promotion_policy': 'never_auto_promote',
            },
            source='workload_proposal_review.rejections',
            intent_ref=intent,
            candidate_type='workload_task',
            index=index,
        )

    promotion_edges: list[dict[str, Any]] = []
    for item in promotions or []:
        if not isinstance(item, Mapping):
            continue
        candidate_id = _clean_text(item.get('candidate_id'))
        contract_ref = _clean_text(item.get('contract_ref') or item.get('obligation_id') or item.get('target'))
        if not candidate_id or not contract_ref:
            continue
        promotion_edges.append(
            _json_safe(
                {
                    'candidate_id': candidate_id,
                    'contract_ref': contract_ref,
                    'source': _clean_text(item.get('source')) or 'promotion_records',
                    'reason': _clean_text(item.get('reason')),
                }
            )
        )
        if candidate_id in by_id:
            by_id[candidate_id] = _merge_candidate(
                by_id[candidate_id],
                {'status': 'promoted', 'contract_ref': contract_ref},
            )
            for offset, existing in enumerate(values):
                if _clean_text(existing.get('candidate_id')) == candidate_id:
                    values[offset] = by_id[candidate_id]
                    break

    type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for candidate in values:
        candidate_type = _clean_text(candidate.get('candidate_type')) or 'unknown'
        status = _clean_text(candidate.get('status')).lower() or 'candidate'
        type_counts[candidate_type] = type_counts.get(candidate_type, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

    return _json_safe(
        {
            'kind': 'ollmo.candidate_graph',
            'candidate_graph_version': CANDIDATE_CONTRACTS_VERSION,
            'intent_ref': intent,
            'candidate_count': len(values),
            'type_counts': type_counts,
            'status_counts': status_counts,
            'promotion_edges': promotion_edges,
            'candidates': values,
        }
    )


def review_candidate_promotions(
    candidate_graph: Mapping[str, Any],
    *,
    existing_contracts: Optional[Mapping[str, Any]] = None,
    controls: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Review normalized candidates and make promotion/omission explicit."""

    candidates = candidate_graph.get('candidates') if isinstance(candidate_graph, Mapping) else []
    if not isinstance(candidates, list):
        candidates = []
    policy = controls if isinstance(controls, Mapping) else {}
    authority_default = _clean_text(policy.get('promotion_authority')) or 'runtime_review'
    decisions: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate_id = _clean_text(raw_candidate.get('candidate_id'))
        if not candidate_id:
            continue
        status = _clean_text(raw_candidate.get('status')).lower() or 'candidate'
        contract_ref = _clean_text(raw_candidate.get('contract_ref'))
        if status == 'superseded':
            decision = 'superseded'
        elif status == 'waived':
            decision = 'waived'
        elif status == 'rejected':
            decision = 'rejected'
        elif status == 'promoted' or contract_ref:
            decision = 'promoted'
        elif status in {'omitted', 'not_promoted', 'not-promoted'}:
            decision = 'omitted'
        elif status == 'stale':
            decision = 'stale'
        else:
            decision = 'reserved'
        reason = _clean_text(raw_candidate.get('promotion_reason') or raw_candidate.get('reason'))
        if not reason:
            reason = {
                'promoted': 'candidate has an explicit promoted contract reference',
                'superseded': 'candidate contract was superseded by newer runtime truth',
                'waived': 'candidate was explicitly waived by review or policy',
                'rejected': 'candidate was rejected before promotion',
                'omitted': 'candidate did not pass promotion review',
                'stale': 'candidate no longer matches current runtime truth',
                'reserved': 'candidate remains possible but not owed work',
            }.get(decision, 'candidate review completed')
        authority = _clean_text(raw_candidate.get('promotion_source')) or authority_default
        decision_payload = {
            'candidate_id': candidate_id,
            'candidate_type': _clean_text(raw_candidate.get('candidate_type')) or None,
            'decision': decision,
            'contract_ref': contract_ref or None,
            'authority': authority,
            'reason': reason,
            'evidence_refs': raw_candidate.get('evidence_refs') if isinstance(raw_candidate.get('evidence_refs'), list) else None,
        }
        if decision in {'omitted', 'reserved', 'stale'}:
            decision_payload['reconsiderable'] = True
            decision_payload['execution_policy'] = (
                _clean_text(raw_candidate.get('execution_policy'))
                or 'non_executable_until_promoted'
            )
            decision_payload['reconsideration_policy'] = (
                _clean_text(raw_candidate.get('reconsideration_policy'))
                or 'review_again_when_context_or_intent_changes'
            )
        elif decision == 'promoted':
            decision_payload['execution_policy'] = (
                _clean_text(raw_candidate.get('execution_policy'))
                or 'executable_obligation'
            )
        elif decision in {'rejected', 'superseded', 'waived'}:
            decision_payload['terminal'] = True
            if decision == 'superseded':
                decision_payload['supersession_policy'] = (
                    _clean_text(raw_candidate.get('supersession_policy'))
                    or 'closed_by_current_runtime_truth'
                )
                for key in ('superseded_by', 'superseded_by_candidate_id', 'superseded_by_obligation_id'):
                    value = _clean_text(raw_candidate.get(key))
                    if value:
                        decision_payload[key] = value
        decisions.append(_json_safe(decision_payload))

    counts = {
        'promoted': sum(1 for item in decisions if item.get('decision') == 'promoted'),
        'reserved': sum(1 for item in decisions if item.get('decision') == 'reserved'),
        'omitted': sum(1 for item in decisions if item.get('decision') == 'omitted'),
        'superseded': sum(1 for item in decisions if item.get('decision') == 'superseded'),
        'waived': sum(1 for item in decisions if item.get('decision') == 'waived'),
        'rejected': sum(1 for item in decisions if item.get('decision') == 'rejected'),
        'stale': sum(1 for item in decisions if item.get('decision') == 'stale'),
    }
    status = 'empty'
    if decisions:
        status = 'reviewed'
    if counts['rejected']:
        status = 'reviewed_with_rejections'
    if counts['waived']:
        status = 'reviewed_with_waivers'
    if counts['superseded']:
        status = 'reviewed_with_supersessions'
    if existing_contracts:
        contract_source = _clean_text(existing_contracts.get('source')) or 'existing_contracts'
    else:
        contract_source = 'candidate_graph'
    return _json_safe(
        {
            'kind': 'ollmo.promotion_review',
            'promotion_review_version': CANDIDATE_CONTRACTS_VERSION,
            'status': status,
            'contract_source': contract_source,
            'candidate_count': len(decisions),
            'promoted_count': counts['promoted'],
            'reserved_count': counts['reserved'],
            'omitted_count': counts['omitted'],
            'superseded_count': counts['superseded'],
            'waived_count': counts['waived'],
            'rejected_count': counts['rejected'],
            'stale_count': counts['stale'],
            'counts': counts,
            'decisions': decisions,
        }
    )
