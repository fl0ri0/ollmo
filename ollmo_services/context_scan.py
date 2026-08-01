"""Small promoted history-scan adapter over existing Ollmo ledgers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ollmo_services.artifact_registry import DEFAULT_ARTIFACT_REGISTRY_LEDGER
from ollmo_services.chat_history import (
    DEFAULT_RESPONSE_FRAME_LEDGER,
    DEFAULT_RESPONSE_FRAMES_DIR,
    list_chat_history_index,
    read_chat_history,
)

CONTEXT_SCAN_VERSION = 1

_STOPWORDS = {
    'about',
    'after',
    'again',
    'alle',
    'allen',
    'alles',
    'also',
    'and',
    'aber',
    'bitte',
    'can',
    'chat',
    'conversation',
    'der',
    'die',
    'das',
    'den',
    'dem',
    'des',
    'ein',
    'eine',
    'einen',
    'einer',
    'entire',
    'find',
    'for',
    'from',
    'ganze',
    'ganzen',
    'gesamte',
    'gesamten',
    'history',
    'ich',
    'ist',
    'mit',
    'mir',
    'nach',
    'noch',
    'oder',
    'please',
    'scan',
    'search',
    'that',
    'the',
    'this',
    'und',
    'verlauf',
    'was',
    'what',
    'where',
    'with',
    'you',
}


def _clean_text(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '').strip())


def _preview(value: Any, *, limit: int = 220) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + '...'


def _tokenize(value: Any, *, limit: int = 24) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r'[\wäöüÄÖÜß]+', str(value or '').lower()):
        token = token.strip('_')
        if len(token) < 4 or token in _STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _safe_id(value: Any, *, fallback: str = 'item') -> str:
    text = _clean_text(value)
    if not text:
        text = fallback
    token = ''.join(ch if ch.isalnum() else '-' for ch in text.lower())
    token = '-'.join(part for part in token.split('-') if part)
    return token[:96] or fallback


def _digest(value: Any) -> str:
    text = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _score_text(value: Any, query_terms: list[str]) -> tuple[int, list[str]]:
    text = str(value or '').lower()
    matched = [term for term in query_terms if term and term in text]
    return len(matched), matched


def _artifact_refs(value: Any) -> list[str]:
    refs: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            ref = _clean_text(item.get('artifact_ref') or item.get('artifactRef') or item.get('ref'))
            if ref and ref not in refs:
                refs.append(ref)
            for child in item.values():
                if len(refs) >= 12:
                    return
                visit(child)
        elif isinstance(item, list):
            for child in item:
                if len(refs) >= 12:
                    return
                visit(child)

    visit(value)
    return refs


def _read_jsonl(path: Path, *, limit: int = 800) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    scanned = 0
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line:
            continue
        scanned += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
        if scanned >= max(1, int(limit)):
            break
    return records


def _response_frame_ledger_path(history_dir: Path | str | None = None) -> Path:
    if history_dir:
        return Path(history_dir).parent / 'response_frames' / DEFAULT_RESPONSE_FRAME_LEDGER
    return DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get('content')
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(_clean_text(item.get('text') or item.get('content') or item.get('type')))
            else:
                parts.append(_clean_text(item))
        content = ' '.join(part for part in parts if part)
    return _clean_text(content)


def _record_text(value: Any, *, depth: int = 0) -> str:
    if depth > 3:
        return ''
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in (
            'prompt',
            'input',
            'output_text',
            'summary',
            'name',
            'path',
            'artifact_ref',
            'response_id',
            'model',
            'capability',
            'mode',
        ):
            text = _clean_text(value.get(key))
            if text:
                parts.append(text)
        for key in ('request', 'target', 'route', 'artifact', 'metadata', 'provenance', 'output', 'artifacts'):
            nested = value.get(key)
            if isinstance(nested, (Mapping, list)):
                nested_text = _record_text(nested, depth=depth + 1)
                if nested_text:
                    parts.append(nested_text)
        return ' '.join(parts)
    if isinstance(value, list):
        return ' '.join(_record_text(item, depth=depth + 1) for item in value[:12])
    return _clean_text(value)


def _push_candidate(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    seen: set[str],
) -> None:
    candidate_id = _clean_text(candidate.get('candidate_id'))
    if not candidate_id or candidate_id in seen:
        return
    seen.add(candidate_id)
    candidates.append({key: value for key, value in candidate.items() if value not in (None, '', [], {})})


def _history_message_candidates(
    *,
    query_terms: list[str],
    history_dir: Path | str | None,
    max_histories: int,
    max_messages_per_history: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    for index_item in list_chat_history_index(history_dir=history_dir)[:max_histories]:
        instance_id = _clean_text(index_item.get('instance_id'))
        if not instance_id:
            continue
        history = read_chat_history(instance_id, history_dir=history_dir)
        messages = history.get('messages') if isinstance(history.get('messages'), list) else []
        for message_index, message in enumerate(reversed(messages[-max_messages_per_history:])):
            if not isinstance(message, Mapping):
                continue
            scanned += 1
            text = _message_text(message)
            score, matched = _score_text(text, query_terms)
            if score <= 0:
                continue
            role = _clean_text(message.get('role')) or 'message'
            message_id = _clean_text(message.get('message_id') or message.get('messageId') or message.get('id'))
            source_id = message_id or f'{len(messages) - message_index}'
            refs = _artifact_refs(message)
            _push_candidate(
                candidates,
                {
                    'candidate_id': f'history-message-{_safe_id(instance_id)}-{_safe_id(source_id)}',
                    'source_kind': 'message',
                    'status': 'promoted',
                    'promotion_target': 'active_context',
                    'promotion_source': 'history_scan',
                    'promotion_policy': 'requires_promoted_history_scan_and_lexical_relevance',
                    'promotion_reason': f'matched history terms: {", ".join(matched[:6])}',
                    'summary': f'{instance_id} {role}: {_preview(text)}',
                    'reason': 'matched promoted history scan',
                    'relevance': f'matched terms: {", ".join(matched[:8])}',
                    'conversation_id': instance_id,
                    'history_instance_id': instance_id,
                    'source_message_id': message_id,
                    'role': role,
                    'source_surface': 'chat_history',
                    'score': score,
                    'match_terms': matched[:8],
                    'artifact_refs': refs[:8],
                },
                seen=seen,
            )
    return candidates, scanned


def _response_frame_candidates(
    *,
    query_terms: list[str],
    history_dir: Path | str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    records = _read_jsonl(_response_frame_ledger_path(history_dir), limit=limit)
    for record in records:
        text = _record_text(record)
        score, matched = _score_text(text, query_terms)
        if score <= 0:
            continue
        response_id = _clean_text(record.get('response_id') or record.get('id')) or _digest(record)
        refs = _artifact_refs(record)
        _push_candidate(
            candidates,
            {
                'candidate_id': f'history-response-frame-{_safe_id(response_id)}',
                'source_kind': 'history',
                'status': 'promoted',
                'promotion_target': 'active_context',
                'promotion_source': 'history_scan',
                'promotion_policy': 'requires_promoted_history_scan_and_lexical_relevance',
                'promotion_reason': f'matched response-frame terms: {", ".join(matched[:6])}',
                'summary': f'response {response_id}: {_preview(text)}',
                'reason': 'matched promoted response-frame ledger scan',
                'relevance': f'matched terms: {", ".join(matched[:8])}',
                'source_response_id': response_id,
                'source_surface': 'response_frame_ledger',
                'score': score,
                'match_terms': matched[:8],
                'artifact_refs': refs[:8],
            },
            seen=seen,
        )
    return candidates, len(records)


def _artifact_registry_candidates(
    *,
    query_terms: list[str],
    artifact_registry_ledger: Path | str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    ledger = Path(artifact_registry_ledger) if artifact_registry_ledger else DEFAULT_ARTIFACT_REGISTRY_LEDGER
    records = _read_jsonl(ledger, limit=limit)
    for record in records:
        text = _record_text(record)
        score, matched = _score_text(text, query_terms)
        if score <= 0:
            continue
        artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
        metadata = record.get('metadata') if isinstance(record.get('metadata'), Mapping) else {}
        artifact_ref = _clean_text(record.get('artifact_ref') or artifact.get('artifact_ref') or artifact.get('ref'))
        artifact_path = _clean_text(
            record.get('image_path')
            or artifact.get('path')
            or metadata.get('path')
            or metadata.get('file_path')
        )
        artifact_type = _clean_text(artifact.get('type') or record.get('type') or metadata.get('type')) or 'artifact'
        candidate_id = artifact_ref or artifact_path or _digest(record)
        _push_candidate(
            candidates,
            {
                'candidate_id': f'history-artifact-{_safe_id(candidate_id)}',
                'source_kind': 'artifact',
                'status': 'promoted',
                'promotion_target': 'active_reference',
                'promotion_source': 'history_scan',
                'promotion_policy': 'requires_promoted_history_scan_and_lexical_relevance',
                'promotion_reason': f'matched artifact-registry terms: {", ".join(matched[:6])}',
                'summary': f'{artifact_type} {artifact_ref or artifact_path}: {_preview(text)}',
                'reason': 'matched promoted artifact-registry scan',
                'relevance': f'matched terms: {", ".join(matched[:8])}',
                'artifact_ref': artifact_ref,
                'type': artifact_type,
                'path': artifact_path,
                'source_surface': 'artifact_registry',
                'score': score,
                'match_terms': matched[:8],
            },
            seen=seen,
        )
    return candidates, len(records)


def _rank_candidates(candidates: Iterable[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    source_order = {
        'chat_history': 0,
        'response_frame_ledger': 1,
        'artifact_registry': 2,
    }
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item.get('score') or 0),
            -source_order.get(_clean_text(item.get('source_surface')), 9),
            _clean_text(item.get('candidate_id')),
        ),
        reverse=True,
    )
    promoted = ranked[: max(1, int(limit))]
    for rank, candidate in enumerate(promoted, start=1):
        candidate['rank'] = rank
        candidate['relevance_score'] = int(candidate.get('score') or 0)
    return promoted


def build_history_scan_context_candidates(
    *,
    prompt: str,
    history_dir: Path | str | None = None,
    artifact_registry_ledger: Path | str | None = None,
    max_candidates: int = 8,
    max_histories: int = 16,
    max_messages_per_history: int = 60,
    ledger_limit: int = 800,
) -> dict[str, Any]:
    """Return promoted context candidates from existing history/storage surfaces."""

    query_terms = _tokenize(prompt)
    if not query_terms:
        return {
            'kind': 'ollmo.history_scan_result',
            'scan_version': CONTEXT_SCAN_VERSION,
            'status': 'skipped',
            'reason': 'no searchable query terms',
            'ranking_policy': 'lexical_term_overlap_then_source_order',
            'query_terms': [],
            'context_candidates': [],
            'candidate_count': 0,
            'matched_candidate_count': 0,
            'promoted_candidate_count': 0,
            'omitted_candidate_count': 0,
            'matched': {
                'chat_history': 0,
                'response_frame_ledger': 0,
                'artifact_registry': 0,
            },
            'scanned': {
                'chat_history': 0,
                'response_frame_ledger': 0,
                'artifact_registry': 0,
            },
        }

    history_candidates, scanned_messages = _history_message_candidates(
        query_terms=query_terms,
        history_dir=history_dir,
        max_histories=max_histories,
        max_messages_per_history=max_messages_per_history,
    )
    frame_candidates, scanned_frames = _response_frame_candidates(
        query_terms=query_terms,
        history_dir=history_dir,
        limit=ledger_limit,
    )
    artifact_candidates, scanned_artifacts = _artifact_registry_candidates(
        query_terms=query_terms,
        artifact_registry_ledger=artifact_registry_ledger,
        limit=ledger_limit,
    )
    all_candidates = [*history_candidates, *frame_candidates, *artifact_candidates]
    ranked_candidates = _rank_candidates(
        all_candidates,
        limit=max_candidates,
    )
    matched_counts = {
        'chat_history': len(history_candidates),
        'response_frame_ledger': len(frame_candidates),
        'artifact_registry': len(artifact_candidates),
    }
    matched_candidate_count = len(all_candidates)
    promoted_candidate_count = len(ranked_candidates)
    return {
        'kind': 'ollmo.history_scan_result',
        'scan_version': CONTEXT_SCAN_VERSION,
        'status': 'completed',
        'ranking_policy': 'lexical_term_overlap_then_source_order',
        'query_terms': query_terms,
        'context_candidates': ranked_candidates,
        'candidate_count': promoted_candidate_count,
        'matched_candidate_count': matched_candidate_count,
        'promoted_candidate_count': promoted_candidate_count,
        'omitted_candidate_count': max(0, matched_candidate_count - promoted_candidate_count),
        'matched': matched_counts,
        'scanned': {
            'chat_history': scanned_messages,
            'response_frame_ledger': scanned_frames,
            'artifact_registry': scanned_artifacts,
        },
    }
