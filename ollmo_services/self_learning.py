"""Offline self-learning eval-case extraction from frozen Ollmo frames."""

from __future__ import annotations

import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from ollmo_services.graph_repair import classify_surface_repair_actionability
from ollmo_services.self_learning_retention import (
    DEFAULT_RETENTION_MANIFEST,
    collect_self_learning_retention_roots,
    retention_summary,
)

SELF_LEARNING_VERSION = 1
DEFAULT_RESPONSE_FRAMES_DIR = Path('state/response_frames')
DEFAULT_RESPONSE_FRAME_LEDGER = 'responses.jsonl'
DEFAULT_SELF_LEARNING_DIR = Path('state/self_learning')
DEFAULT_EVAL_CASE_LEDGER = 'eval_cases.jsonl'
DEFAULT_SELF_LEARNING_REPORT = 'report.json'
DEFAULT_ACCEPTED_POLICY_SNAPSHOT = 'accepted_policy_snapshot.json'
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
GRAPH_REBASE_CORPUS_MANIFEST_KIND = 'ollmo.graph_rebase_shadow_corpus_manifest'
GRAPH_REBASE_CORPUS_SCHEMA_VERSION = 1
_GRAPH_REBASE_CORPUS_MAX_MANIFESTS = 256
_GRAPH_REBASE_CORPUS_MAX_CASES = 4096
_GRAPH_REBASE_CORPUS_MAX_GRAPH_RECORD_GROUPS = 16
_GRAPH_REBASE_CORPUS_MAX_GRAPH_RECORDS_PER_GROUP = 64
_JSONL_REVERSE_READ_CHUNK_BYTES = 1024 * 1024
CHAT_ROUTE_HEALTH_CASE_KIND = 'chat_completion_route_health_signal'
LEGACY_CHAT_ROUTE_HEALTH_CASE_KIND = 'fra' + 'gile_chat_' + 'provider_' + 'family'
ALLOWED_ACCEPTED_LEARNING_TARGET_AREAS = {
    'artifact_fulfillment_policy',
    'aspiration_policy',
    'closure_review_policy',
    'commitment_policy',
    'context_gate_policy',
    'controlled_attention_policy',
    'graph_repair_policy',
    'graph_rebase_policy',
    'redraw_scope_policy',
    'ghost_decision_contract_policy',
    'ghost_intake_graph_policy',
    'reconsideration_policy',
    'semantic_decision_policy',
    'semantic_review_policy',
    'semantic_verdict_policy',
    'supersession_policy',
    'workload_decision_policy',
}
ACCEPTED_LEARNING_AUTHORITY_LEVELS = {
    'soft_hint',
    'advisory',
    'preferred',
    'enforced',
}
DEFAULT_ACCEPTED_LEARNING_AUTHORITY = 'soft_hint'
REPLACE_EXISTING_EVAL_CASE_POLICY = 'replace_existing'
MERGE_EXISTING_EVAL_CASE_POLICY = 'union_by_case_id_new_wins_preserve_existing'


def _now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _iter_text_fragments(value: Any, *, max_depth: int = 6) -> Iterable[str]:
    if max_depth <= 0:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, Mapping):
        for key in ('text', 'content', 'input', 'prompt', 'instructions', 'output_text', 'summary', 'preview', 'message', 'error'):
            if key in value:
                yield from _iter_text_fragments(value.get(key), max_depth=max_depth - 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_text_fragments(item, max_depth=max_depth - 1)


def _normalize_accepted_learning_authority(value: Any) -> str:
    token = _clean_text(value).lower().replace('-', '_')
    if token in ACCEPTED_LEARNING_AUTHORITY_LEVELS:
        return token
    return DEFAULT_ACCEPTED_LEARNING_AUTHORITY


def _canonical_case_kind(value: Any) -> str:
    token = _clean_text(value)
    if token == LEGACY_CHAT_ROUTE_HEALTH_CASE_KIND:
        return CHAT_ROUTE_HEALTH_CASE_KIND
    return token


def _canonical_case_kinds(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: Counter[str] = Counter()
    for key, count in value.items():
        kind = _canonical_case_kind(key)
        if not kind:
            continue
        try:
            amount = int(count or 0)
        except (TypeError, ValueError):
            amount = 1
        counts[kind] += max(0, amount)
    return dict(sorted(counts.items()))


def _canonical_evidence_case_id(value: Any) -> str:
    return _clean_text(value).replace(LEGACY_CHAT_ROUTE_HEALTH_CASE_KIND, CHAT_ROUTE_HEALTH_CASE_KIND)


def _canonical_accepted_learning_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _json_safe(dict(value))
    record['case_kinds'] = _canonical_case_kinds(record.get('case_kinds'))
    evidence_ids = record.get('evidence_case_ids') if isinstance(record.get('evidence_case_ids'), list) else []
    record['evidence_case_ids'] = [
        evidence_id
        for evidence_id in (_canonical_evidence_case_id(item) for item in evidence_ids)
        if evidence_id
    ]
    return _json_safe(record)


def _canonical_eval_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    metadata = _json_safe(dict(value))
    legacy_backend_key = 'provider_' + 'family_hint'
    if legacy_backend_key in metadata and 'backend_family_hint' not in metadata:
        metadata['backend_family_hint'] = metadata.pop(legacy_backend_key)
    return _json_safe(metadata)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if item not in (None, '', [], {})}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value if item not in (None, '', [], {})]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_safe_preserving_empty(value: Any) -> Any:
    """Convert values to JSON-safe shapes without changing stored case content."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe_preserving_empty(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_preserving_empty(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(value: Any) -> str:
    text = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def _assemble_reverse_jsonl_line(
    oldest_fragment: bytes,
    newer_fragments: list[bytes],
) -> bytes:
    """Assemble one reverse-read line while releasing its buffered chunks."""

    if not newer_fragments:
        return oldest_fragment
    fragments = [oldest_fragment, *reversed(newer_fragments)]
    newer_fragments.clear()
    return b''.join(fragments)


def _iter_reverse_jsonl_lines(path: Path) -> Iterable[bytes]:
    """Yield raw JSONL lines newest-first with at most one line buffered."""

    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            cursor = handle.tell()
            newer_fragments: list[bytes] = []
            chunk_size = max(1, int(_JSONL_REVERSE_READ_CHUNK_BYTES))

            while cursor > 0:
                read_size = min(chunk_size, cursor)
                cursor -= read_size
                handle.seek(cursor)
                chunk = handle.read(read_size)
                parts = chunk.split(b'\n')
                if len(parts) == 1:
                    newer_fragments.append(parts[0])
                    continue

                yield _assemble_reverse_jsonl_line(parts[-1], newer_fragments)
                for part in reversed(parts[1:-1]):
                    yield part
                newer_fragments.append(parts[0])

            if newer_fragments:
                yield _assemble_reverse_jsonl_line(b'', newer_fragments)
    except OSError:
        return


def _iter_jsonl(path: Path, *, limit: int) -> Iterable[dict[str, Any]]:
    """Yield recent JSON-object records newest-first without loading the ledger."""

    scanned = 0
    scan_limit = max(1, int(limit))
    for raw_line in _iter_reverse_jsonl_lines(path):
        line = raw_line.strip()
        if not line:
            continue
        scanned += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        is_mapping = isinstance(payload, dict)
        del raw_line
        del line
        if is_mapping:
            yield payload
        del payload
        if scanned >= scan_limit:
            break


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Compatibility list wrapper for bounded JSONL consumers."""

    return list(_iter_jsonl(path, limit=limit))


def load_eval_cases(
    input_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load a complete eval-case ledger without dropping historical rows.

    Merge mode promises to preserve every existing case. Invalid or ambiguous
    row identity therefore fails closed instead of being skipped like bounded
    diagnostic JSONL inputs.
    """

    target = Path(input_path) if input_path else DEFAULT_SELF_LEARNING_DIR / DEFAULT_EVAL_CASE_LEDGER
    if not target.exists():
        return []

    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    try:
        handle = target.open('r', encoding='utf-8')
    except OSError as exc:
        raise ValueError(f'could not read existing eval-case ledger {target}: {exc}') from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'existing eval-case ledger {target} has malformed JSON on line {line_number}'
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f'existing eval-case ledger {target} line {line_number} must be a JSON object'
                )
            case = dict(payload)
            case_id = _clean_text(case.get('case_id'))
            if not case_id:
                raise ValueError(
                    f'existing eval-case ledger {target} line {line_number} is missing case_id'
                )
            if case_id in seen_case_ids:
                raise ValueError(
                    f'existing eval-case ledger {target} contains duplicate case_id {case_id!r}'
                )
            seen_case_ids.add(case_id)
            cases.append(case)
    return cases


def _eval_cases_by_case_id(
    cases: Iterable[Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for ordinal, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f'{source} eval case {ordinal} must be a mapping')
        case_id = _clean_text(case.get('case_id'))
        if not case_id:
            raise ValueError(f'{source} eval case {ordinal} is missing case_id')
        if case_id in indexed:
            raise ValueError(f'{source} eval cases contain duplicate case_id {case_id!r}')
        indexed[case_id] = dict(case)
    return indexed


def merge_eval_cases_by_case_id(
    existing_cases: Iterable[Mapping[str, Any]],
    new_cases: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a deterministic non-destructive union in which fresh cases win."""

    previous_by_id = _eval_cases_by_case_id(existing_cases, source='existing')
    new_by_id = _eval_cases_by_case_id(new_cases, source='new')
    previous_ids = set(previous_by_id)
    new_ids = set(new_by_id)
    replaced_ids = previous_ids & new_ids
    preserved_ids = previous_ids - new_ids

    merged_by_id = dict(previous_by_id)
    merged_by_id.update(new_by_id)
    merged_cases = [merged_by_id[case_id] for case_id in sorted(merged_by_id)]
    counts = {
        'previous_case_count': len(previous_by_id),
        'new_case_count': len(new_by_id),
        'preserved_case_count': len(preserved_ids),
        'replaced_case_count': len(replaced_ids),
        'removed_case_count': 0,
    }
    return merged_cases, counts


def _eval_case_merge_policy(*, merge_existing: bool) -> dict[str, Any]:
    if merge_existing:
        return {
            'name': MERGE_EXISTING_EVAL_CASE_POLICY,
            'mode': 'merge_existing',
            'identity_key': 'case_id',
            'conflict_resolution': 'new_case_wins',
            'existing_case_disposition': 'preserve_old_only_cases',
            'ordering': 'case_id_ascending',
            'max_cases_scope': 'newly_generated_cases_only',
            'preserved_case_truth_role': 'historical_eval_evidence_only_not_current_runtime_truth',
            'optimization_policy': 'proposal_only_reviewed_patch_required',
            'accepted_learning_authority': 'soft_hint_only',
            'automatic_policy_promotion': False,
            'automatic_policy_enablement': False,
            'runtime_effect': 'none',
        }
    return {
        'name': REPLACE_EXISTING_EVAL_CASE_POLICY,
        'mode': 'replace_existing',
        'identity_key': 'case_id',
        'existing_case_disposition': 'not_read_replacement_behavior',
        'ordering': 'fresh_extraction_order',
        'max_cases_scope': 'newly_generated_cases',
        'optimization_policy': 'proposal_only_reviewed_patch_required',
        'accepted_learning_authority': 'soft_hint_only',
        'automatic_policy_promotion': False,
        'automatic_policy_enablement': False,
        'runtime_effect': 'none',
    }


def _nested_mapping(payload: Mapping[str, Any], *path: str) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, Mapping) else {}


def _nested_list(payload: Mapping[str, Any], *path: str) -> list[Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return []
        current = current.get(key)
    return list(current) if isinstance(current, list) else []


def _frame_response_id(frame: Mapping[str, Any]) -> str:
    return _clean_text(frame.get('response_id') or frame.get('id')) or f'frame-{_digest(frame)}'


def _frame_prompt(frame: Mapping[str, Any]) -> str:
    request = frame.get('request') if isinstance(frame.get('request'), Mapping) else {}
    text = '\n'.join(_iter_text_fragments(request.get('prompt') or request.get('input') or request.get('instructions')))
    return text.strip()


def _frame_target(frame: Mapping[str, Any]) -> dict[str, Any]:
    target = frame.get('target') if isinstance(frame.get('target'), Mapping) else {}
    return {
        key: value
        for key, value in {
            'instance_id': _clean_text(target.get('instance_id')),
            'model': _clean_text(target.get('model')),
            'backend': _clean_text(target.get('backend')),
            'capability': _clean_text(target.get('capability')),
            'mode': _clean_text(target.get('mode')),
        }.items()
        if value
    }


def _snapshot_path(snapshot_root: Path, ref: Mapping[str, Any]) -> Path | None:
    raw_path = _clean_text(ref.get('path'))
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return snapshot_root / path


def _read_snapshot_ref(snapshot_root: Path | None, ref: Any) -> dict[str, Any]:
    if snapshot_root is None or not isinstance(ref, Mapping):
        return {}
    path = _snapshot_path(snapshot_root, ref)
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _hydrate_frame_for_learning(
    frame: Mapping[str, Any],
    *,
    snapshot_root: Path | None,
) -> dict[str, Any]:
    hydrated = dict(frame)
    if snapshot_root is None:
        return hydrated

    current_state = dict(hydrated.get('current_state')) if isinstance(hydrated.get('current_state'), Mapping) else {}
    runtime = dict(hydrated.get('runtime')) if isinstance(hydrated.get('runtime'), Mapping) else {}
    runtime_snapshot = _read_snapshot_ref(snapshot_root, hydrated.get('runtime_snapshot_ref'))
    if not runtime_snapshot:
        runtime_snapshot = _read_snapshot_ref(snapshot_root, current_state.get('runtime_snapshot_ref'))
    if runtime_snapshot:
        merged_runtime = {**runtime_snapshot, **runtime}
        graph_review = _read_snapshot_ref(snapshot_root, merged_runtime.get('graph_closure_review_snapshot_ref'))
        if graph_review and not isinstance(merged_runtime.get('graph_closure_review'), Mapping):
            merged_runtime['graph_closure_review'] = graph_review
        hydrated['runtime'] = merged_runtime
        runtime = merged_runtime

    late_fill = dict(hydrated.get('late_fill')) if isinstance(hydrated.get('late_fill'), Mapping) else {}
    for ref in (
        late_fill.get('full_snapshot_ref'),
        late_fill.get('review_snapshot_ref'),
        hydrated.get('late_fill_snapshot_ref'),
        current_state.get('late_fill_snapshot_ref'),
        runtime.get('late_fill_snapshot_ref'),
    ):
        late_fill_snapshot = _read_snapshot_ref(snapshot_root, ref)
        if late_fill_snapshot:
            late_fill = {**late_fill_snapshot, **late_fill}
            break
    if late_fill:
        hydrated['late_fill'] = late_fill

    planning = dict(hydrated.get('planning')) if isinstance(hydrated.get('planning'), Mapping) else {}
    artifact_flow = dict(planning.get('artifact_flow')) if isinstance(planning.get('artifact_flow'), Mapping) else {}
    artifact_flow_snapshot = _read_snapshot_ref(snapshot_root, planning.get('artifact_flow_snapshot_ref'))
    if artifact_flow_snapshot:
        planning['artifact_flow'] = {**artifact_flow_snapshot, **artifact_flow}
        hydrated['planning'] = planning
    return hydrated


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounded_corpus_graph_records(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        return {}
    records: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(value, key=lambda item: str(item))[:_GRAPH_REBASE_CORPUS_MAX_GRAPH_RECORD_GROUPS]:
        raw_items = value.get(key)
        if not isinstance(raw_items, list):
            continue
        items = [
            dict(item)
            for item in raw_items[:_GRAPH_REBASE_CORPUS_MAX_GRAPH_RECORDS_PER_GROUP]
            if isinstance(item, Mapping)
        ]
        if items:
            records[str(key)] = items
    return records


def _corpus_record_identity(value: Mapping[str, Any]) -> str:
    for key in (
        'review_id',
        'execution_key',
        'successor_key',
        'idempotency_key',
        'patch_id',
        'rebase_id',
        'frame_id',
        'id',
        'proposal_id',
    ):
        identity = _clean_text(value.get(key))
        if identity:
            return f'{key}:{identity}'
    return f'digest:{_digest(value)}'


def _corpus_case_provenance(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            'corpus_id': binding.get('corpus_id'),
            'corpus_digest': binding.get('corpus_digest'),
            'manifest_file': binding.get('manifest_file'),
            'case_id': binding.get('case_id'),
            'category': binding.get('category'),
            'workload_family': binding.get('workload_family'),
            'state': binding.get('state'),
            'settled_outcome': binding.get('settled_outcome'),
            'response_id': binding.get('response_id'),
            'frame_id': binding.get('frame_id'),
            'frame_sequence': binding.get('frame_sequence'),
        }.items()
        if value not in (None, '', [], {})
    }


def _load_graph_rebase_corpus(
    corpus_dir: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage: dict[str, Any] = {
        'kind': 'ollmo.graph_rebase_corpus_coverage',
        'status': 'not_configured' if corpus_dir is None else 'completed',
        'corpus_dir': str(corpus_dir) if corpus_dir is not None else None,
        'manifest_file_count': 0,
        'manifest_count': 0,
        'malformed_manifest_count': 0,
        'unsupported_manifest_count': 0,
        'case_count': 0,
        'malformed_case_count': 0,
        'binding_candidate_count': 0,
        'state_counts': {},
        'diagnostics': [],
    }
    if corpus_dir is None:
        return _json_safe(coverage), []
    if not corpus_dir.is_dir():
        coverage['status'] = 'missing_directory'
        coverage['diagnostics'] = [
            {
                'status': 'missing_directory',
                'path': str(corpus_dir),
            }
        ]
        return _json_safe(coverage), []

    manifest_paths = [
        path
        for path in sorted(corpus_dir.glob('*.json'))
        if path.is_file()
    ]
    coverage['manifest_file_count'] = len(manifest_paths)
    if len(manifest_paths) > _GRAPH_REBASE_CORPUS_MAX_MANIFESTS:
        coverage['status'] = 'partial'
        coverage['manifest_file_limit'] = _GRAPH_REBASE_CORPUS_MAX_MANIFESTS
        coverage['manifest_file_truncated_count'] = len(manifest_paths) - _GRAPH_REBASE_CORPUS_MAX_MANIFESTS
        manifest_paths = manifest_paths[:_GRAPH_REBASE_CORPUS_MAX_MANIFESTS]

    bindings: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []
    case_budget_remaining = _GRAPH_REBASE_CORPUS_MAX_CASES
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            coverage['malformed_manifest_count'] += 1
            diagnostics.append(
                {
                    'manifest_file': manifest_path.name,
                    'status': 'malformed_manifest',
                    'error': str(exc)[:300],
                }
            )
            continue
        if not isinstance(manifest, Mapping):
            coverage['malformed_manifest_count'] += 1
            diagnostics.append(
                {
                    'manifest_file': manifest_path.name,
                    'status': 'malformed_manifest',
                    'error': 'manifest root must be a JSON object',
                }
            )
            continue
        if (
            _clean_text(manifest.get('kind')) != GRAPH_REBASE_CORPUS_MANIFEST_KIND
            or _integer_or_none(manifest.get('schema_version')) != GRAPH_REBASE_CORPUS_SCHEMA_VERSION
        ):
            coverage['unsupported_manifest_count'] += 1
            diagnostics.append(
                {
                    'manifest_file': manifest_path.name,
                    'status': 'unsupported_manifest',
                    'kind': manifest.get('kind'),
                    'schema_version': manifest.get('schema_version'),
                }
            )
            continue
        raw_cases = manifest.get('cases')
        if not isinstance(raw_cases, list):
            coverage['malformed_manifest_count'] += 1
            diagnostics.append(
                {
                    'manifest_file': manifest_path.name,
                    'status': 'malformed_manifest',
                    'error': 'cases must be a JSON array',
                }
            )
            continue

        coverage['manifest_count'] += 1
        corpus_id = _clean_text(manifest.get('corpus_id'))
        corpus_digest = _clean_text(manifest.get('corpus_digest'))
        for ordinal, raw_case in enumerate(raw_cases):
            if case_budget_remaining <= 0:
                coverage['status'] = 'partial'
                coverage['case_limit'] = _GRAPH_REBASE_CORPUS_MAX_CASES
                coverage['case_truncated'] = True
                break
            case_budget_remaining -= 1
            coverage['case_count'] += 1
            if not isinstance(raw_case, Mapping):
                coverage['malformed_case_count'] += 1
                state_counts['malformed'] += 1
                diagnostics.append(
                    {
                        'manifest_file': manifest_path.name,
                        'case_ordinal': ordinal,
                        'status': 'malformed_case',
                        'error': 'case must be a JSON object',
                    }
                )
                continue
            case_id = _clean_text(raw_case.get('case_id'))
            state = _clean_text(raw_case.get('state')).lower()
            if not case_id or not state:
                coverage['malformed_case_count'] += 1
                state_counts['malformed'] += 1
                diagnostics.append(
                    {
                        'manifest_file': manifest_path.name,
                        'case_id': case_id or None,
                        'case_ordinal': ordinal,
                        'status': 'malformed_case',
                        'error': 'case_id and state are required',
                    }
                )
                continue
            state_counts[state] += 1
            final_debug = raw_case.get('final_debug') if isinstance(raw_case.get('final_debug'), Mapping) else {}
            summary = final_debug.get('summary') if isinstance(final_debug.get('summary'), Mapping) else {}
            response_frame = (
                summary.get('response_frame')
                if isinstance(summary.get('response_frame'), Mapping)
                else {}
            )
            response_id = _clean_text(summary.get('id'))
            frame_id = _clean_text(response_frame.get('frame_id'))
            frame_sequence = _integer_or_none(response_frame.get('frame_sequence'))
            binding = {
                'corpus_id': corpus_id,
                'corpus_digest': corpus_digest,
                'manifest_file': manifest_path.name,
                'case_id': case_id,
                'category': _clean_text(raw_case.get('category')),
                'workload_family': _clean_text(raw_case.get('workload_family')),
                'state': state,
                'settled_outcome': _clean_text(raw_case.get('settled_outcome')),
                'response_id': response_id,
                'frame_id': frame_id,
                'frame_sequence': frame_sequence,
                'declared_response_id': _clean_text(raw_case.get('response_id')),
                'declared_last_frame_id': _clean_text(raw_case.get('last_frame_id')),
                'declared_last_frame_sequence': _integer_or_none(raw_case.get('last_frame_sequence')),
                '_graph_records': _bounded_corpus_graph_records(summary.get('graph_records')),
                '_diagnostic_records': _bounded_corpus_graph_records(summary.get('diagnostic_records')),
                '_redraw_scope': dict(summary.get('redraw_scope'))
                if isinstance(summary.get('redraw_scope'), Mapping)
                else {},
            }
            if state in {'planned', 'dependency_blocked'}:
                binding['binding_status'] = state
                bindings.append(binding)
                continue
            if state not in {'settled_terminal', 'settled_repair_needed'}:
                binding['binding_status'] = 'malformed_case'
                binding['binding_error'] = 'unsupported case state for settled evidence'
                coverage['malformed_case_count'] += 1
                bindings.append(binding)
                continue
            declared_response_id = _clean_text(raw_case.get('response_id'))
            if _clean_text(final_debug.get('status')).lower() != 'captured':
                binding['binding_status'] = 'malformed_binding'
                binding['binding_error'] = 'settled evidence requires final_debug.status=captured'
                coverage['malformed_case_count'] += 1
                bindings.append(binding)
                continue
            if not declared_response_id or declared_response_id != response_id:
                binding['binding_status'] = 'malformed_binding'
                binding['binding_error'] = 'case.response_id must exactly match final_debug.summary.id'
                coverage['malformed_case_count'] += 1
                bindings.append(binding)
                continue
            if not response_id or not frame_id or frame_sequence is None:
                binding['binding_status'] = 'malformed_binding'
                binding['binding_error'] = 'final_debug.summary.id and response_frame frame identity are required'
                coverage['malformed_case_count'] += 1
                bindings.append(binding)
                continue
            binding['binding_status'] = 'pending_ledger_binding'
            coverage['binding_candidate_count'] += 1
            bindings.append(binding)
        if case_budget_remaining <= 0:
            break

    coverage['state_counts'] = dict(sorted(state_counts.items()))
    coverage['diagnostics'] = diagnostics
    if not manifest_paths:
        coverage['status'] = 'empty'
    elif coverage['status'] == 'completed' and (
        coverage['malformed_manifest_count']
        or coverage['unsupported_manifest_count']
        or coverage['malformed_case_count']
    ):
        coverage['status'] = 'partial'
    return _json_safe(coverage), bindings


def _overlay_corpus_eval_evidence(
    frame: Mapping[str, Any],
    bindings: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    overlaid = dict(frame)
    runtime = dict(overlaid.get('runtime')) if isinstance(overlaid.get('runtime'), Mapping) else {}
    graph = (
        dict(runtime.get('request_phase_graph'))
        if isinstance(runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    diagnostics = (
        dict(runtime.get('developer_diagnostics'))
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    redraw_reviews: list[dict[str, Any]] = []
    existing_review = graph.get('redraw_scope_ladder_review')
    if isinstance(existing_review, Mapping):
        redraw_reviews.append(dict(existing_review))
    for item in graph.get('redraw_scope_ladder_reviews') or []:
        if isinstance(item, Mapping):
            redraw_reviews.append(dict(item))

    for binding in bindings:
        graph_records = binding.get('_graph_records')
        if isinstance(graph_records, Mapping):
            for key, raw_items in graph_records.items():
                if not isinstance(raw_items, list):
                    continue
                existing_value = graph.get(key)
                if existing_value not in (None, []) and not isinstance(existing_value, list):
                    continue
                existing_items = [
                    dict(item)
                    for item in existing_value or []
                    if isinstance(item, Mapping)
                ]
                existing_identities = {_corpus_record_identity(item) for item in existing_items}
                for item in raw_items:
                    if not isinstance(item, Mapping):
                        continue
                    item_identity = _corpus_record_identity(item)
                    if item_identity not in existing_identities:
                        existing_items.append(dict(item))
                        existing_identities.add(item_identity)
                if existing_items:
                    graph[str(key)] = existing_items
        diagnostic_records = binding.get('_diagnostic_records')
        if isinstance(diagnostic_records, Mapping):
            for key, raw_items in diagnostic_records.items():
                if not isinstance(raw_items, list):
                    continue
                existing_value = diagnostics.get(key)
                if existing_value not in (None, []) and not isinstance(existing_value, list):
                    continue
                existing_items = [
                    dict(item)
                    for item in existing_value or []
                    if isinstance(item, Mapping)
                ]
                existing_identities = {_corpus_record_identity(item) for item in existing_items}
                for item in raw_items:
                    if not isinstance(item, Mapping):
                        continue
                    item_identity = _corpus_record_identity(item)
                    if item_identity not in existing_identities:
                        existing_items.append(dict(item))
                        existing_identities.add(item_identity)
                if existing_items:
                    diagnostics[str(key)] = existing_items
        redraw_scope = binding.get('_redraw_scope')
        if isinstance(redraw_scope, Mapping) and redraw_scope:
            redraw_reviews.append(dict(redraw_scope))

    if redraw_reviews:
        by_identity: dict[str, dict[str, Any]] = {}
        for review in redraw_reviews:
            identity = _clean_text(review.get('review_id')) or _digest(review)
            if identity not in by_identity:
                by_identity[identity] = review
        merged_reviews = list(by_identity.values())
        if not isinstance(existing_review, Mapping):
            graph['redraw_scope_ladder_review'] = merged_reviews[0]
        graph['redraw_scope_ladder_reviews'] = merged_reviews
    if graph:
        runtime['request_phase_graph'] = graph
    if diagnostics:
        runtime['developer_diagnostics'] = diagnostics
    if runtime:
        overlaid['runtime'] = runtime
    return overlaid


def _attach_corpus_provenance(
    cases: Iterable[dict[str, Any]],
    bindings: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provenance = [_corpus_case_provenance(binding) for binding in bindings]
    provenance = [item for item in provenance if item]
    if not provenance:
        return [dict(case) for case in cases]
    attached: list[dict[str, Any]] = []
    for case in cases:
        item = dict(case)
        metadata = dict(item.get('metadata')) if isinstance(item.get('metadata'), Mapping) else {}
        metadata['graph_rebase_corpus'] = {'bindings': provenance}
        item['metadata'] = _canonical_eval_metadata(metadata)
        attached.append(item)
    return attached


def _visible_output_text(frame: Mapping[str, Any]) -> str:
    fragments: list[str] = []
    for container in (
        frame.get('output'),
        frame.get('current_state'),
        frame.get('response_frame'),
    ):
        if isinstance(container, Mapping):
            for key in ('text', 'output_text', 'output', 'content'):
                fragments.extend(_iter_text_fragments(container.get(key)))
            outputs = container.get('outputs')
            if isinstance(outputs, list):
                fragments.extend(_iter_text_fragments(outputs))
    fragments.extend(_iter_text_fragments(frame.get('output_text')))
    return '\n'.join(dict.fromkeys(fragment for fragment in fragments if fragment))


def _artifact_records_from(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if any(key in value for key in ('artifact_ref', 'path', 'mime_type', 'kind', 'type')):
            kind = _clean_text(value.get('kind') or value.get('type')).lower()
            path = _clean_text(value.get('path'))
            artifact_ref = _clean_text(value.get('artifact_ref') or value.get('ref'))
            mime_type = _clean_text(value.get('mime_type'))
            if path or artifact_ref or mime_type or kind in {'image', 'audio', 'text', 'file'}:
                records.append(dict(value))
        for key in ('output', 'outputs', 'artifacts', 'saved_text_artifacts', 'saved_image_path', 'saved_audio_path'):
            if key in value:
                records.extend(_artifact_records_from(value.get(key)))
    elif isinstance(value, list):
        for item in value:
            records.extend(_artifact_records_from(item))
    elif isinstance(value, str) and value:
        records.append({'path': value})
    return records


def _saved_artifact_records(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ('artifacts', 'output', 'current_state'):
        records.extend(_artifact_records_from(frame.get(key)))
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        identity = _clean_text(record.get('artifact_ref') or record.get('ref') or record.get('path') or record.get('name'))
        if not identity:
            identity = _digest(record)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(record)
    return unique


def _artifact_request_signals(prompt: str) -> dict[str, bool]:
    low = prompt.lower()
    return {
        'requests_image_artifacts': bool(
            re.search(r'\b(generated image|image artifact|bilder|images?|image files?|bild(er)?)\b', low)
        ),
        'requests_index_html': 'index.html' in low,
        'requests_styles_css': 'styles.css' in low,
        'requests_file_artifacts': bool(
            re.search(r'\b(save|saved|local artifact|local artifacts|files?|dateien|artifact|artifacts|export|bundle)\b', low)
        ),
    }


def _is_explicit_artifact_request(prompt: str) -> bool:
    signals = _artifact_request_signals(prompt)
    return (
        signals['requests_index_html']
        or signals['requests_styles_css']
        or signals['requests_file_artifacts']
        or signals['requests_image_artifacts']
    )


def _contains_control_json_leak(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            '"request_ir"',
            "'request_ir'",
            'request_ir',
            '"output_obligations"',
            'output_obligations',
            '"candidate_graph"',
            'candidate_graph',
            '"route"',
            'ghost_route',
        )
    )


def _frame_lifecycle_state(frame: Mapping[str, Any]) -> str:
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), Mapping) else {}
    late_fill = frame.get('late_fill') if isinstance(frame.get('late_fill'), Mapping) else {}
    return _clean_text(
        current_state.get('lifecycle_state')
        or frame.get('lifecycle_state')
        or late_fill.get('lifecycle_state')
        or frame.get('status')
    ).lower()


def _is_terminal_state(state: str) -> bool:
    return state in {
        'completed',
        'failed',
        'cancelled',
        'waived',
        'superseded',
        'frozen',
        'late_fill_completed',
        'late_fill_failed',
        'repair_needed',
        'blocked',
    }


def _late_fill_truth(frame: Mapping[str, Any]) -> dict[str, Any]:
    late_fill = frame.get('late_fill') if isinstance(frame.get('late_fill'), Mapping) else {}
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), Mapping) else {}
    current_late_fill = current_state.get('late_fill') if isinstance(current_state.get('late_fill'), Mapping) else {}
    return {**dict(current_late_fill), **dict(late_fill)}


def _text_from_payload(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=str)


def _collect_late_fill_error_text(late_fill: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in ('error', 'error_summary', 'blocked_reason', 'final_materialization_contract_reason'):
        pieces.extend(_iter_text_fragments(late_fill.get(key)))
    for key in ('failed_branches', 'recovery_candidates', 'materialization_contract_open_checks', 'fill_results'):
        value = late_fill.get(key)
        if isinstance(value, list):
            pieces.extend(_iter_text_fragments(value))
            if value:
                pieces.append(_text_from_payload(value[:5]))
    return '\n'.join(piece for piece in pieces if piece)


def _extract_backend_family_hint(text: str, frame: Mapping[str, Any], late_fill: Mapping[str, Any]) -> str:
    haystack = '\n'.join(
        [
            text,
            _text_from_payload(_frame_target(frame)),
            _text_from_payload(late_fill.get('failed_branches') or []),
            _text_from_payload(late_fill.get('fill_results') or []),
        ]
    ).lower()
    for token in ('mlx', 'vlm', 'llama.cpp', 'ollama'):
        if token in haystack:
            return token
    return 'chat_backend_family'


def _case(
    *,
    frame: Mapping[str, Any],
    layer: str,
    kind: str,
    severity: str,
    summary: str,
    evidence: str,
    target_area: str,
    target_surfaces: list[str],
    suggested_action: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    response_id = _frame_response_id(frame)
    canonical_kind = _canonical_case_kind(kind)
    payload = {
        'kind': 'ollmo.self_learning_eval_case',
        'case_version': SELF_LEARNING_VERSION,
        'case_id': f'eval-{layer}-{canonical_kind}-{response_id}-{_digest(metadata or summary)}',
        'response_id': response_id,
        'frame_status': _clean_text(frame.get('status')),
        'layer': layer,
        'case_kind': canonical_kind,
        'severity': severity,
        'summary': summary,
        'evidence': evidence,
        'target_area': target_area,
        'target_surfaces': target_surfaces,
        'suggested_action': suggested_action,
        'prompt': _frame_prompt(frame),
        'target': _frame_target(frame),
        'metadata': _canonical_eval_metadata(metadata or {}),
        'optimization_policy': 'proposal_only_reviewed_patch_required',
    }
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, '', [], {})
    }


def _extract_intake_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    adequacy = graph_review.get('intent_graph_adequacy') if isinstance(graph_review.get('intent_graph_adequacy'), Mapping) else {}
    if _clean_text(adequacy.get('status')).lower() != 'pending':
        return []
    checks = adequacy.get('checks') if isinstance(adequacy.get('checks'), list) else []
    cases: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        output_type = _clean_text(check.get('output_type')) or 'output'
        cases.append(
            _case(
                frame=frame,
                layer='intake',
                kind='intent_graph_inadequacy',
                severity='high',
                summary=f'Current user intent expected more {output_type} obligations than the graph represented.',
                evidence=_clean_text(check.get('evidence')) or 'intent_graph_adequacy_missing_output_obligation',
                target_area='ghost_intake_graph_policy',
                target_surfaces=[
                    'GHOST.md',
                    'ollmo_g/intent.py',
                    'ollmo_g/request_phase_graph.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Review intent parsing and graph promotion rules for this request shape.',
                metadata={
                    'adequacy_status': adequacy.get('status'),
                    'expected_output_counts': adequacy.get('expected_output_counts'),
                    'graph_output_counts': adequacy.get('graph_output_counts'),
                    'check': dict(check),
                },
            )
        )
    return cases


def _extract_closure_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = _clean_text(graph_review.get('status')).lower()
    if status not in {'pending', 'blocked'}:
        return []
    checks = graph_review.get('checks') if isinstance(graph_review.get('checks'), list) else []
    cases: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if _clean_text(check.get('check_kind')) == 'intent_graph_adequacy':
            continue
        check_status = _clean_text(check.get('status')).lower()
        if check_status not in {'pending', 'planned', 'active', 'deferred', 'blocked'}:
            continue
        cases.append(
            _case(
                frame=frame,
                layer='closure',
                kind='open_graph_obligation',
                severity='high' if check_status == 'blocked' else 'medium',
                summary='A promoted graph/IR obligation remained open at closure review.',
                evidence=_clean_text(check.get('evidence')) or f'graph_closure_review:{check_status}',
                target_area='closure_review_policy',
                target_surfaces=[
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_services/frame_planning.py',
                    'ollmo_orchestration/working_frame.py',
                ],
                suggested_action='Review closure, late-fill, fulfillment, or waiver handling for this obligation pattern.',
                metadata={
                    'graph_review_status': status,
                    'graph_review_counts': graph_review.get('counts'),
                    'check': dict(check),
                },
            )
        )
    return cases


def _extract_context_cases(frame: Mapping[str, Any], context_contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    context_gate_review = (
        context_contract.get('context_gate_review')
        if isinstance(context_contract.get('context_gate_review'), Mapping)
        else {}
    )
    history_scan = (
        context_gate_review.get('history_scan')
        if isinstance(context_gate_review.get('history_scan'), Mapping)
        else {}
    )
    if not history_scan:
        return []
    executed = bool(history_scan.get('executed'))
    matched = int(history_scan.get('matched_candidate_count') or 0)
    promoted = int(history_scan.get('promoted_candidate_count') or 0)
    cases: list[dict[str, Any]] = []
    if executed and matched <= 0:
        cases.append(
            _case(
                frame=frame,
                layer='context',
                kind='history_scan_no_matches',
                severity='low',
                summary='A broader history scan was promoted but found no relevant matches.',
                evidence='context_gate_review.history_scan.no_matches',
                target_area='context_gate_policy',
                target_surfaces=[
                    'GHOST.md',
                    'ollmo_server/ghost_route_runtime.py',
                    'ollmo_services/context_scan.py',
                ],
                suggested_action='If frequent, review deep-history trigger precision and scan query extraction.',
                metadata={
                    'context_gate_review': dict(context_gate_review),
                    'history_scan': dict(history_scan),
                },
            )
        )
    elif executed and promoted > 0:
        cases.append(
            _case(
                frame=frame,
                layer='context',
                kind='history_scan_promoted_matches',
                severity='positive',
                summary='A promoted history scan found compact relevant matches and preserved ids/refs.',
                evidence='context_gate_review.history_scan.promoted_matches',
                target_area='context_gate_policy',
                target_surfaces=[
                    'GHOST.md',
                    'ollmo_server/ghost_route_runtime.py',
                    'ollmo_services/context_scan.py',
                ],
                suggested_action='Keep as a positive eval trace for context-gate behavior.',
                metadata={
                    'context_gate_review': dict(context_gate_review),
                    'history_scan': dict(history_scan),
                },
            )
        )
    return cases


def _extract_artifact_cases(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    review = _nested_mapping(frame, 'planning', 'artifact_flow', 'review')
    pending = _nested_list(review, 'pending_output_slot_ids')
    blocked = _nested_list(review, 'blocked_output_slot_ids')
    if not pending and not blocked:
        return []
    cases = [
        _case(
            frame=frame,
            layer='artifact',
            kind='open_output_slots',
            severity='high' if blocked else 'medium',
            summary='Artifact flow still had open output slots at freeze/review time.',
            evidence='artifact_flow.review.open_output_slots',
            target_area='artifact_fulfillment_policy',
            target_surfaces=[
                'ollmo_services/frame_planning.py',
                'ollmo_services/response_frames.py',
                'ollmo_server/response_semantics_runtime.py',
            ],
            suggested_action='Review output materialization, artifact detection, and slot fulfillment rules.',
            metadata={
                'pending_output_slot_ids': pending,
                'blocked_output_slot_ids': blocked,
                'artifact_review_status': review.get('status'),
            },
        )
    ]
    lifecycle_state = _frame_lifecycle_state(frame)
    if _is_terminal_state(lifecycle_state):
        cases.append(
            _case(
                frame=frame,
                layer='artifact',
                kind='open_output_slots_after_terminal_state',
                severity='high' if blocked else 'medium',
                summary='Artifact flow still had open output slots after a terminal lifecycle state.',
                evidence='artifact_flow.review.open_output_slots_after_terminal_state',
                target_area='artifact_fulfillment_policy',
                target_surfaces=[
                    'ollmo_services/frame_planning.py',
                    'ollmo_services/response_frames.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Review terminal slot reconciliation so stale pending or blocked slots do not survive truthful freeze.',
                metadata={
                    'lifecycle_state': lifecycle_state,
                    'pending_output_slot_ids': pending,
                    'blocked_output_slot_ids': blocked,
                    'artifact_review_status': review.get('status'),
                },
            )
        )
    return cases


def _extract_artifact_request_cases(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt = _frame_prompt(frame)
    if not _is_explicit_artifact_request(prompt):
        return []
    visible_text = _visible_output_text(frame)
    artifacts = _saved_artifact_records(frame)
    control_json_leaked = _contains_control_json_leak(visible_text)
    cases: list[dict[str, Any]] = []
    signals = _artifact_request_signals(prompt)
    if not artifacts and (control_json_leaked or visible_text):
        cases.append(
            _case(
                frame=frame,
                layer='intake',
                kind='artifact_request_collapsed_to_plain_chat',
                severity='high',
                summary='An explicit artifact request closed as plain chat without saved artifact truth.',
                evidence='explicit_artifact_request_without_saved_artifacts',
                target_area='ghost_intake_graph_policy',
                target_surfaces=[
                    'GHOST.md',
                    'ollmo_g/request_phase_graph.py',
                    'ollmo_g/candidate_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Review artifact-request intake and promotion so explicit file/image obligations cannot close as chat-only prose without waiver.',
                metadata={
                    'artifact_request_signals': signals,
                    'saved_artifact_count': len(artifacts),
                    'control_json_leaked': control_json_leaked,
                    'visible_output_preview': visible_text[:500],
                },
            )
        )
    if control_json_leaked:
        cases.append(
            _case(
                frame=frame,
                layer='ghost_decision_contract',
                kind='artifact_control_json_leaked_to_user',
                severity='high',
                summary='Control-plane JSON or request IR leaked into visible user output for an artifact/content request.',
                evidence='visible_output_contains_request_ir_or_output_obligations',
                target_area='ghost_decision_contract_policy',
                target_surfaces=[
                    'GHOST.md',
                    'ollmo_g/payload.py',
                    'ollmo_g/router.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Review no-prose-only/router-output boundaries and text artifact payload extraction.',
                metadata={
                    'artifact_request_signals': signals,
                    'visible_output_preview': visible_text[:800],
                },
            )
        )
    return cases


def _extract_late_fill_contract_cases(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    late_fill = _late_fill_truth(frame)
    if not late_fill:
        return []
    cases: list[dict[str, Any]] = []
    late_fill_status = _clean_text(late_fill.get('status')).lower()
    final_contract = _clean_text(late_fill.get('final_materialization_contract_status')).lower()
    materialization_unmet = late_fill.get('materialization_contract_unmet') is True or final_contract == 'unmet'
    error_text = _collect_late_fill_error_text(late_fill)
    if materialization_unmet:
        cases.append(
            _case(
                frame=frame,
                layer='closure',
                kind='materialization_contract_unmet',
                severity='high',
                summary='Final materialization contract remained unmet according to late-fill or closure truth.',
                evidence='late_fill.materialization_contract_unmet',
                target_area='closure_review_policy',
                target_surfaces=[
                    'ollmo_server/late_fill_runtime.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_services/response_frames.py',
                ],
                suggested_action='Review final materialization contract reconciliation, repair promotion, and truthful freeze rules.',
                metadata={
                    'late_fill_status': late_fill_status,
                    'final_materialization_contract_status': final_contract,
                    'materialization_contract_open_checks': late_fill.get('materialization_contract_open_checks'),
                    'failed_branch_count': late_fill.get('failed_branch_count'),
                },
            )
        )
    if (
        'TEXT_ARTIFACT_TARGET_BINDING_VIOLATION' in error_text
        or 'text_artifact_target_path_mismatch' in error_text
        or 'Target-bound repairs must update the requested target file' in error_text
    ):
        cases.append(
            _case(
                frame=frame,
                layer='artifact',
                kind='target_text_artifact_binding_violation',
                severity='high' if materialization_unmet else 'medium',
                summary='A target-bound text artifact repair saved or returned a sibling file instead of the requested target path.',
                evidence='late_fill.text_artifact_target_path_mismatch',
                target_area='artifact_fulfillment_policy',
                target_surfaces=[
                    'ollmo_server/late_fill_runtime.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_core/transports.py',
                ],
                suggested_action=(
                    'Preserve exact target-path binding for text artifact repairs: patch the requested target file, '
                    'keep wrong sibling artifacts non-fulfilling, and retry the same branch with stronger target evidence.'
                ),
                metadata={
                    'late_fill_status': late_fill_status,
                    'final_materialization_contract_status': final_contract,
                    'target_binding_evidence_preview': error_text[:800],
                },
            )
        )
    if re.search(r'unsupported href element\s*<\s*can\s*>?', error_text, flags=re.I) or '<can' in error_text:
        cases.append(
            _case(
                frame=frame,
                layer='artifact',
                kind='html_navigation_tag_typo',
                severity='high' if materialization_unmet else 'medium',
                summary='HTML syntax review found an unsupported navigation element such as <can> where <a> was required.',
                evidence='text_artifact_syntax_sanity.unsupported_href_element',
                target_area='artifact_fulfillment_policy',
                target_surfaces=[
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_server/late_fill_runtime.py',
                    'ollmo_services/scoped_file_tools.py',
                ],
                suggested_action='Preserve this as a deterministic HTML repair eval case for anchor-tag syntax fixes.',
                metadata={
                    'late_fill_status': late_fill_status,
                    'syntax_evidence_preview': error_text[:800],
                },
            )
        )
    link_failure_tokens = ('missing image', 'missing css', 'placeholder', 'unresolved', 'broken link', 'does not resolve')
    if any(token in error_text.lower() for token in link_failure_tokens):
        cases.append(
            _case(
                frame=frame,
                layer='artifact',
                kind='broken_artifact_dependency_link',
                severity='high',
                summary='A saved artifact dependency link was missing, unresolved, or placeholder-like.',
                evidence='linked_artifact_dependency_unresolved',
                target_area='artifact_fulfillment_policy',
                target_surfaces=[
                    'ollmo_services/artifact_registry.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_services/response_artifact_bundles.py',
                ],
                suggested_action='Review deterministic link rebind and linked artifact closure before final freeze.',
                metadata={
                    'late_fill_status': late_fill_status,
                    'link_evidence_preview': error_text[:800],
                },
            )
        )
    if '500' in error_text and '/v1/chat/completions' in error_text:
        backend_family_hint = _extract_backend_family_hint(error_text, frame, late_fill)
        cases.append(
            _case(
                frame=frame,
                layer='workload',
                kind=CHAT_ROUTE_HEALTH_CASE_KIND,
                severity='medium',
                summary='A chat or text-repair branch hit a backend 500 on a chat-completions endpoint; this is route-health preference evidence only.',
                evidence='late_fill.chat_completions_500',
                target_area='workload_decision_policy',
                target_surfaces=[
                    'ollmo_server/late_fill_runtime.py',
                    'ollmo_server/backend_transport_runtime.py',
                    'ollmo_server/ghost_route_runtime.py',
                ],
                suggested_action='Record a non-authoritative route-health preference for chat/coalesced text repair; do not infer provider bans, offline state, hard degraded truth, or graph repair proof from this hint.',
                metadata={
                    'backend_family_hint': backend_family_hint,
                    'late_fill_status': late_fill_status,
                    'error_preview': error_text[:800],
                },
            )
        )
    fulfilled_contract = late_fill_status == 'completed' or final_contract == 'fulfilled'
    nonterminal_error = bool(_clean_text(late_fill.get('error_summary')) or _clean_text(late_fill.get('error')))
    failed_branch_count = int(late_fill.get('failed_branch_count') or 0)
    if fulfilled_contract and (nonterminal_error or failed_branch_count > 0):
        cases.append(
            _case(
                frame=frame,
                layer='closure',
                kind='nonterminal_failed_branch_with_fulfilled_contract',
                severity='low',
                summary='A final contract was fulfilled while superseded or canonical-evidence-satisfied branch failure evidence remained.',
                evidence='late_fill.fulfilled_contract_with_nonterminal_failure_evidence',
                target_area='closure_review_policy',
                target_surfaces=[
                    'ollmo_server/late_fill_runtime.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'docs/RESPONSES_CONTRACT.md',
                ],
                suggested_action='Treat this as status reconciliation learning, not a hard fulfillment failure.',
                metadata={
                    'late_fill_status': late_fill_status,
                    'final_materialization_contract_status': final_contract,
                    'failed_branch_count': failed_branch_count,
                    'error_summary': late_fill.get('error_summary'),
                },
            )
        )
    return cases


def _extract_positive_fulfillment_case(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if _clean_text(graph_review.get('status')).lower() != 'fulfilled':
        return []
    counts = graph_review.get('counts') if isinstance(graph_review.get('counts'), Mapping) else {}
    if int(counts.get('fulfilled') or 0) <= 0:
        return []
    return [
        _case(
            frame=frame,
            layer='closure',
            kind='fulfilled_graph_contract',
            severity='positive',
            summary='Graph/IR obligations were fulfilled by runtime truth.',
            evidence='graph_closure_review.fulfilled',
            target_area='closure_review_policy',
            target_surfaces=[
                'ollmo_server/response_semantics_runtime.py',
                'ollmo_services/frame_planning.py',
            ],
            suggested_action='Keep as a positive eval trace for closure behavior.',
            metadata={
                'graph_review_counts': counts,
                'contract_source': graph_review.get('contract_source'),
            },
        )
    ]


def _extract_semantic_decision_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    decision_contract_review = (
        graph_review.get('decision_contract_review')
        if isinstance(graph_review.get('decision_contract_review'), Mapping)
        else {}
    )
    semantic_decision = (
        decision_contract_review.get('semantic_decision_review')
        if isinstance(decision_contract_review.get('semantic_decision_review'), Mapping)
        else {}
    )
    surface_state = (
        graph_review.get('surface_state')
        if isinstance(graph_review.get('surface_state'), Mapping)
        else {}
    )
    graph_status = _clean_text(graph_review.get('status')).lower()
    cases: list[dict[str, Any]] = []
    proposal_count = int(semantic_decision.get('proposal_count') or 0) if semantic_decision else 0
    if proposal_count and graph_status in {'pending', 'blocked'}:
        cases.append(
            _case(
                frame=frame,
                layer='semantic_decision',
                kind='semantic_decision_proposals_unresolved',
                severity='high' if graph_status == 'blocked' else 'medium',
                summary='Semantic decision proposals remained unresolved at closure review.',
                evidence='decision_contract_review.semantic_decision_review.unresolved',
                target_area='semantic_decision_policy',
                target_surfaces=[
                    'ollmo_g/decision_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'GHOST.md',
                ],
                suggested_action='Review whether semantic decision proposals should guide repair, review, waiver, supersession, or continuation more clearly.',
                metadata={
                    'graph_review_status': graph_status,
                    'semantic_decision_review': dict(semantic_decision),
                    'surface_state': dict(surface_state) if surface_state else {},
                },
            )
        )
    elif proposal_count and graph_status == 'fulfilled':
        cases.append(
            _case(
                frame=frame,
                layer='semantic_decision',
                kind='semantic_decision_review_resolved',
                severity='positive',
                summary='Semantic decision proposals existed and the closure review reached fulfilled truth.',
                evidence='decision_contract_review.semantic_decision_review.resolved',
                target_area='semantic_decision_policy',
                target_surfaces=[
                    'ollmo_g/decision_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'GHOST.md',
                ],
                suggested_action='Keep as a positive eval trace for semantic decision review behavior.',
                metadata={
                    'graph_review_status': graph_status,
                    'semantic_decision_review': dict(semantic_decision),
                },
            )
        )

    category_counts = (
        surface_state.get('category_counts')
        if isinstance(surface_state.get('category_counts'), Mapping)
        else {}
    )
    unresolved_surface_count = sum(
        int(category_counts.get(key) or 0)
        for key in ('open', 'blocked', 'repair_pending', 'semantic_review_pending')
    )
    if graph_status == 'fulfilled' and unresolved_surface_count > 0:
        cases.append(
            _case(
                frame=frame,
                layer='surface',
                kind='surface_state_unresolved_at_freeze',
                severity='high',
                summary='Surface state still showed unresolved work even though closure review was fulfilled.',
                evidence='graph_closure_review.surface_state.unresolved_at_fulfilled_freeze',
                target_area='semantic_decision_policy',
                target_surfaces=[
                    'ollmo_server/response_semantics_runtime.py',
                    'static/ui/message-state.js',
                    'docs/PRINCIPLES.md',
                ],
                suggested_action='Review surface-state projection and closure status consistency.',
                metadata={
                    'graph_review_status': graph_status,
                    'surface_state': dict(surface_state),
                },
            )
        )
    return cases


def _extract_controlled_attention_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    decision_contract_review = (
        graph_review.get('decision_contract_review')
        if isinstance(graph_review.get('decision_contract_review'), Mapping)
        else {}
    )
    attention_review = (
        decision_contract_review.get('controlled_attention_review')
        if isinstance(decision_contract_review.get('controlled_attention_review'), Mapping)
        else {}
    )
    if not attention_review:
        return []
    graph_status = _clean_text(graph_review.get('status')).lower()
    frame_count = int(attention_review.get('frame_count') or 0)
    if frame_count and graph_status in {'pending', 'blocked'}:
        return [
            _case(
                frame=frame,
                layer='controlled_attention',
                kind='controlled_attention_unresolved',
                severity='high' if graph_status == 'blocked' else 'medium',
                summary='Controlled attention frames remained unresolved at closure review.',
                evidence='decision_contract_review.controlled_attention_review.unresolved',
                target_area='controlled_attention_policy',
                target_surfaces=[
                    'ollmo_g/decision_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_g/router.py',
                    'GHOST.md',
                ],
                suggested_action='Review whether scoped attention frames are guiding Ghost toward bounded transition proposals instead of root-prompt replay or blind retry.',
                metadata={
                    'graph_review_status': graph_status,
                    'controlled_attention_review': dict(attention_review),
                },
            )
        ]
    if frame_count and graph_status == 'fulfilled':
        return [
            _case(
                frame=frame,
                layer='controlled_attention',
                kind='controlled_attention_resolved',
                severity='positive',
                summary='Controlled attention frames existed and closure reached fulfilled truth.',
                evidence='decision_contract_review.controlled_attention_review.resolved',
                target_area='controlled_attention_policy',
                target_surfaces=[
                    'ollmo_g/decision_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_g/router.py',
                ],
                suggested_action='Keep as a positive eval trace for controlled attention behavior.',
                metadata={
                    'graph_review_status': graph_status,
                    'controlled_attention_review': dict(attention_review),
                },
            )
        ]
    return []


def _extract_orientation_review_cases(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
    *,
    review_key: str,
    frame_count_key: str,
    layer: str,
    unresolved_kind: str,
    resolved_kind: str,
    target_area: str,
    summary_name: str,
) -> list[dict[str, Any]]:
    decision_contract_review = (
        graph_review.get('decision_contract_review')
        if isinstance(graph_review.get('decision_contract_review'), Mapping)
        else {}
    )
    review = (
        decision_contract_review.get(review_key)
        if isinstance(decision_contract_review.get(review_key), Mapping)
        else {}
    )
    if not review:
        return []
    graph_status = _clean_text(graph_review.get('status')).lower()
    frame_count = int(review.get('frame_count') or decision_contract_review.get(frame_count_key) or 0)
    if frame_count <= 0:
        return []
    if graph_status in {'pending', 'blocked'}:
        return [
            _case(
                frame=frame,
                layer=layer,
                kind=unresolved_kind,
                severity='high' if graph_status == 'blocked' else 'medium',
                summary=f'{summary_name} frames remained unresolved at closure review.',
                evidence=f'decision_contract_review.{review_key}.unresolved',
                target_area=target_area,
                target_surfaces=[
                    'ollmo_g/decision_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'GHOST.md',
                ],
                suggested_action=f'Review whether {summary_name.lower()} guidance is helping Ghost choose the right advisory transition.',
                metadata={
                    'graph_review_status': graph_status,
                    review_key: dict(review),
                },
            )
        ]
    if graph_status == 'fulfilled':
        return [
            _case(
                frame=frame,
                layer=layer,
                kind=resolved_kind,
                severity='positive',
                summary=f'{summary_name} frames existed and closure reached fulfilled truth.',
                evidence=f'decision_contract_review.{review_key}.resolved',
                target_area=target_area,
                target_surfaces=[
                    'ollmo_g/decision_contracts.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'GHOST.md',
                ],
                suggested_action=f'Keep as a positive eval trace for {summary_name.lower()} behavior.',
                metadata={
                    'graph_review_status': graph_status,
                    review_key: dict(review),
                },
            )
        ]
    return []


def _extract_aspiration_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _extract_orientation_review_cases(
        frame,
        graph_review,
        review_key='aspiration_review',
        frame_count_key='aspiration_frame_count',
        layer='aspiration',
        unresolved_kind='aspiration_review_unresolved',
        resolved_kind='aspiration_review_resolved',
        target_area='aspiration_policy',
        summary_name='Aspiration review',
    )


def _extract_commitment_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _extract_orientation_review_cases(
        frame,
        graph_review,
        review_key='commitment_review',
        frame_count_key='commitment_frame_count',
        layer='commitment',
        unresolved_kind='commitment_review_unresolved',
        resolved_kind='commitment_review_resolved',
        target_area='commitment_policy',
        summary_name='Commitment review',
    )


def _extract_global_semantic_closure_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    review = (
        graph_review.get('global_semantic_closure_review')
        if isinstance(graph_review.get('global_semantic_closure_review'), Mapping)
        else {}
    )
    if not review:
        return []
    status = _clean_text(review.get('status')).lower()
    verdict = (
        review.get('semantic_review_verdict')
        if isinstance(review.get('semantic_review_verdict'), Mapping)
        else {}
    )
    cases: list[dict[str, Any]] = []
    verdict_token = _clean_text(verdict.get('verdict')).lower() if verdict else ''
    if verdict and verdict_token in {'failed', 'uncertain'}:
        cases.append(
            _case(
                frame=frame,
                layer='semantic_verdict',
                kind=f'semantic_review_verdict_{verdict_token}',
                severity='high' if verdict_token == 'failed' else 'medium',
                summary='A completed semantic review verdict did not allow truthful freeze.',
                evidence='graph_closure_review.global_semantic_closure_review.semantic_review_verdict.unresolved',
                target_area='semantic_verdict_policy',
                target_surfaces=[
                    'ollmo_services/semantic_review_verdict.py',
                    'ollmo_server/response_semantics_runtime.py',
                    'GHOST.md',
                ],
                suggested_action='Review semantic verdict prompting, evidence fit, and the recommended transition chosen by the reviewer.',
                metadata={
                    'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                    'global_semantic_closure_status': status,
                    'semantic_review_verdict': dict(verdict),
                },
            )
        )
    elif verdict and verdict_token == 'passed' and status == 'fulfilled':
        cases.append(
            _case(
                frame=frame,
                layer='semantic_verdict',
                kind='semantic_review_verdict_passed',
                severity='positive',
                summary='A completed semantic review verdict passed before truthful freeze.',
                evidence='graph_closure_review.global_semantic_closure_review.semantic_review_verdict.passed',
                target_area='semantic_verdict_policy',
                target_surfaces=[
                    'ollmo_services/semantic_review_verdict.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Keep as a positive eval trace for structured semantic verdict behavior.',
                metadata={
                    'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                    'semantic_review_verdict': dict(verdict),
                },
            )
        )
    if status in {'pending', 'blocked', 'waiting_on_local_closure'}:
        cases.append(
            _case(
                frame=frame,
                layer='semantic_closure',
                kind='global_semantic_closure_unresolved',
                severity='high' if status == 'blocked' else 'medium',
                summary='Whole-turn semantic closure could not yet verify that local branch truth fits the current intent.',
                evidence='graph_closure_review.global_semantic_closure_review.unresolved',
                target_area='semantic_review_policy',
                target_surfaces=[
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_g/decision_contracts.py',
                    'GHOST.md',
                ],
                suggested_action='Review global semantic closure prompting, evidence handoff, and transition proposal quality.',
                metadata={
                    'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                    'global_semantic_closure_review': dict(review),
                },
            )
        )
        return cases
    if status == 'fulfilled':
        cases.append(
            _case(
                frame=frame,
                layer='semantic_closure',
                kind='global_semantic_closure_resolved',
                severity='positive',
                summary='Whole-turn semantic closure review completed before truthful freeze.',
                evidence='graph_closure_review.global_semantic_closure_review.fulfilled',
                target_area='semantic_review_policy',
                target_surfaces=[
                    'ollmo_server/response_semantics_runtime.py',
                    'ollmo_g/decision_contracts.py',
                ],
                suggested_action='Keep as a positive eval trace for global semantic closure behavior.',
                metadata={
                    'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                    'global_semantic_closure_review': dict(review),
                },
            )
        )
        return cases
    return cases


def _extract_branch_semantic_verdict_cases(frame: Mapping[str, Any], graph_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = graph_review.get('checks') if isinstance(graph_review.get('checks'), list) else []
    cases: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        verdict = check.get('semantic_review_verdict') if isinstance(check.get('semantic_review_verdict'), Mapping) else {}
        branch_review = check.get('branch_semantic_review') if isinstance(check.get('branch_semantic_review'), Mapping) else {}
        if not verdict and not branch_review:
            continue
        verdict_token = _clean_text(verdict.get('verdict')).lower() if verdict else ''
        check_kind = _clean_text(check.get('check_kind'))
        if check_kind != 'branch_semantic_review' and not branch_review:
            continue
        if verdict_token == 'passed' and _clean_text(check.get('status')).lower() == 'fulfilled':
            cases.append(
                _case(
                    frame=frame,
                    layer='semantic_verdict',
                    kind='branch_semantic_review_verdict_passed',
                    severity='positive',
                    summary='A branch-level semantic review verdict passed before branch freeze.',
                    evidence='graph_closure_review.checks.branch_semantic_review.semantic_review_verdict.passed',
                    target_area='semantic_verdict_policy',
                    target_surfaces=[
                        'ollmo_services/semantic_review_verdict.py',
                        'ollmo_server/response_semantics_runtime.py',
                    ],
                    suggested_action='Keep as a positive eval trace for branch semantic review behavior.',
                    metadata={
                        'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                        'check': dict(check),
                        'semantic_review_verdict': dict(verdict),
                    },
                )
            )
        elif verdict_token in {'failed', 'uncertain'}:
            cases.append(
                _case(
                    frame=frame,
                    layer='semantic_verdict',
                    kind=f'branch_semantic_review_verdict_{verdict_token}',
                    severity='high' if verdict_token == 'failed' else 'medium',
                    summary='A branch-level semantic review verdict did not allow branch freeze.',
                    evidence='graph_closure_review.checks.branch_semantic_review.semantic_review_verdict.unresolved',
                    target_area='semantic_verdict_policy',
                    target_surfaces=[
                        'ollmo_services/semantic_review_verdict.py',
                        'ollmo_server/response_semantics_runtime.py',
                        'GHOST.md',
                    ],
                    suggested_action='Review whether the branch semantic verifier received the right branch evidence and chose the right transition.',
                    metadata={
                        'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                        'check': dict(check),
                        'semantic_review_verdict': dict(verdict),
                    },
                )
            )
        elif check_kind == 'branch_semantic_review' and _clean_text(check.get('status')).lower() in {'pending', 'blocked'}:
            cases.append(
                _case(
                    frame=frame,
                    layer='semantic_verdict',
                    kind='branch_semantic_review_pending',
                    severity='medium',
                    summary='A branch-level semantic review check remained open at closure review.',
                    evidence='graph_closure_review.checks.branch_semantic_review.pending',
                    target_area='semantic_verdict_policy',
                    target_surfaces=[
                        'ollmo_server/response_semantics_runtime.py',
                        'ollmo_server/late_fill_runtime.py',
                    ],
                    suggested_action='Review branch semantic-review scheduling and evidence handoff.',
                    metadata={
                        'graph_review_status': _clean_text(graph_review.get('status')).lower(),
                        'check': dict(check),
                    },
                )
            )
    return cases


def _graph_repair_runtime_graphs(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    for value in (
        runtime.get('request_phase_graph'),
        runtime.get('graph'),
        _nested_mapping(frame, 'planning', 'request_phase_graph'),
        _nested_mapping(frame, 'planning', 'artifact_flow', 'request_phase_graph'),
        _nested_mapping(frame, 'planning', 'artifact_flow', 'graph'),
    ):
        if isinstance(value, Mapping) and value:
            graph = dict(value)
            identity = _digest(graph)
            if not any(_digest(existing) == identity for existing in graphs):
                graphs.append(graph)
    return graphs


def _collect_graph_repair_records(frame: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for item in graph.get('graph_repair_proposals') or []:
            if isinstance(item, Mapping):
                proposals.append(dict(item))
        for item in graph.get('graph_repair_reviews') or []:
            if isinstance(item, Mapping):
                reviews.append(dict(item))
        decision_contract = graph.get('decision_contract') if isinstance(graph.get('decision_contract'), Mapping) else {}
        for item in decision_contract.get('graph_repair_proposals') or []:
            if isinstance(item, Mapping):
                proposals.append(dict(item))
    if isinstance(diagnostics, Mapping):
        proposal = diagnostics.get('graph_repair_proposal')
        review = diagnostics.get('graph_repair_proposal_review')
        if isinstance(proposal, Mapping):
            proposals.append(dict(proposal))
        if isinstance(review, Mapping):
            reviews.append(dict(review))

    def dedupe(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            identity = ''
            for key in keys:
                identity = _clean_text(item.get(key))
                if identity:
                    break
            if not identity:
                identity = _digest(item)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        return unique

    return dedupe(proposals, 'proposal_id', 'id'), dedupe(reviews, 'review_id', 'id')


def _collect_redraw_scope_ladder_reviews(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        review = graph.get('redraw_scope_ladder_review')
        if isinstance(review, Mapping):
            reviews.append(dict(review))
        for item in graph.get('redraw_scope_ladder_reviews') or []:
            if isinstance(item, Mapping):
                reviews.append(dict(item))
    if isinstance(diagnostics, Mapping):
        review = diagnostics.get('redraw_scope_ladder_review')
        if isinstance(review, Mapping):
            reviews.append(dict(review))
        for item in diagnostics.get('redraw_scope_ladder_reviews') or []:
            if isinstance(item, Mapping):
                reviews.append(dict(item))

    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for review in reviews:
        identity = _clean_text(review.get('review_id')) or _digest(review)
        if identity not in by_identity:
            by_identity[identity] = review
            order.append(identity)
            continue
        existing = by_identity[identity]
        if len(json.dumps(_json_safe(review), sort_keys=True)) >= len(json.dumps(_json_safe(existing), sort_keys=True)):
            by_identity[identity] = review
    return [by_identity[identity] for identity in order]


def _extract_redraw_scope_ladder_cases(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviews = _collect_redraw_scope_ladder_reviews(frame)
    if not reviews:
        return []
    cases: list[dict[str, Any]] = []
    closure_status = _clean_text(graph_review.get('status') or graph_review.get('state')).lower().replace('-', '_')
    target_surfaces = [
        'ollmo_services/redraw_scope.py',
        'ollmo_server/responses_request_runtime.py',
        'ollmo_services/graph_repair.py',
        'ollmo_services/graph_rebase.py',
        'ollmo_services/self_learning.py',
    ]
    for review in reviews:
        review_id = _clean_text(review.get('review_id')) or _digest(review)
        selected_scope = _clean_text(review.get('selected_scope')).lower().replace('-', '_') or 'observe'
        blocked_reasons = _graph_patch_blocked_reasons(review)
        learning_orientation = (
            dict(review.get('learning_orientation'))
            if isinstance(review.get('learning_orientation'), Mapping)
            else {}
        )
        artifact_identity = (
            dict(review.get('artifact_identity'))
            if isinstance(review.get('artifact_identity'), Mapping)
            else {}
        )
        metadata = {
            'review_id': review_id,
            'review_status': review.get('status'),
            'selected_scope': selected_scope,
            'selected_candidate': dict(review.get('selected_candidate') or {})
            if isinstance(review.get('selected_candidate'), Mapping)
            else {},
            'closure_status': closure_status,
            'blocked_reasons': blocked_reasons,
            'learning_orientation': learning_orientation,
            'artifact_identity': artifact_identity,
            'authority': 'runtime_scope_review_orientation_only',
            'runtime_effect': 'none',
            'review': dict(review),
        }
        if selected_scope != 'observe':
            cases.append(
                _case(
                    frame=frame,
                    layer='redraw_scope',
                    kind='redraw_scope_selected',
                    severity='medium',
                    summary='Runtime selected the smallest intent-aligned repair or redraw scope from the scope ladder.',
                    evidence=f'redraw_scope.selected:{selected_scope}:{review_id}',
                    target_area='redraw_scope_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Use as soft orientation for future scope selection; current Runtime/Closure evidence must still validate every repair or rebase proposal.',
                    metadata=metadata,
                )
            )
        if selected_scope in {'partial_subtree_rebase', 'full_successor_rebase'}:
            cases.append(
                _case(
                    frame=frame,
                    layer='redraw_scope',
                    kind=f'redraw_scope_{selected_scope}_selected',
                    severity='medium',
                    summary='Runtime escalated beyond additive repair to a reviewed rebase scope.',
                    evidence=f'redraw_scope.rebase_selected:{selected_scope}:{review_id}',
                    target_area='redraw_scope_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep rebase scope successor-only, preservation-backed, and explicitly authorized before application.',
                    metadata=metadata,
                )
            )
        if learning_orientation.get('used'):
            cases.append(
                _case(
                    frame=frame,
                    layer='redraw_scope',
                    kind='redraw_scope_learning_orientation_soft_hint',
                    severity='positive',
                    summary='Accepted learning was visible to redraw scope selection only as soft orientation.',
                    evidence=f'redraw_scope.learning_soft_hint:{review_id}',
                    target_area='redraw_scope_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep accepted learning useful for attention and ordering, but never sufficient to validate graph repair or rebase.',
                    metadata=metadata,
                )
            )
        if any(
            reason in blocked_reasons
            for reason in (
                'current_runtime_evidence_required',
                'degraded_or_provider_signal_not_scope_authority',
            )
        ):
            cases.append(
                _case(
                    frame=frame,
                    layer='redraw_scope',
                    kind='redraw_scope_non_authoritative_evidence_ignored',
                    severity='positive',
                    summary='Learning-only, degraded, provider, cache, liveness, monitor, frontend, or UI-label evidence did not select an executable redraw scope.',
                    evidence=f'redraw_scope.non_authoritative_ignored:{review_id}',
                    target_area='redraw_scope_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep non-authoritative evidence diagnostic-only; require current Runtime/Closure truth for every executable graph transition.',
                    metadata=metadata,
                )
            )
        if artifact_identity.get('final_projection_blocked') or 'conflicting_duplicate_artifact_ref' in blocked_reasons:
            cases.append(
                _case(
                    frame=frame,
                    layer='redraw_scope',
                    kind='redraw_scope_duplicate_artifact_ref_conflict',
                    severity='high',
                    summary='Duplicate artifact refs could not be safely canonicalized and remained repair-needed.',
                    evidence=f'redraw_scope.duplicate_artifact_ref_conflict:{review_id}',
                    target_area='redraw_scope_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Preserve conflicting artifact identity evidence so final projection does not hide ambiguous artifact bindings.',
                    metadata=metadata,
                )
            )
    return cases


def _collect_graph_patch_lifecycle_records(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for key in ('graph_patch_lifecycle', 'applied_graph_patches', 'staged_graph_patches'):
            for item in graph.get(key) or []:
                if isinstance(item, Mapping):
                    records.append(dict(item))
    if isinstance(diagnostics, Mapping):
        for key in (
            'graph_patch_lifecycle',
            'graph_patch_lifecycle_results',
            'applied_graph_patches',
            'staged_graph_patches',
        ):
            for item in diagnostics.get(key) or []:
                if isinstance(item, Mapping):
                    records.append(dict(item))

    def identity_for(item: Mapping[str, Any]) -> str:
        return (
            _clean_text(item.get('patch_id'))
            or _clean_text(item.get('idempotency_key'))
            or _clean_text(item.get('proposal_id'))
            or _digest(item)
        )

    def score(item: Mapping[str, Any]) -> int:
        status = _clean_text(item.get('status')).lower().replace('-', '_')
        status_score = {
            'applied': 70,
            'already_applied': 68,
            'blocked': 62,
            'rejected': 62,
            'failed': 58,
            'invalid': 58,
            'staged': 30,
            'validated': 25,
            'proposed': 10,
        }.get(status, 0)
        value = status_score
        for key, points in (
            ('outcome', 10),
            ('risk_level', 6),
            ('repair_class', 6),
            ('blocked_reasons', 6),
            ('source_evidence_refs', 6),
            ('after_graph_digest', 4),
            ('before_graph_digest', 3),
            ('validation_review', 3),
            ('graph_patch_authorization', 3),
        ):
            if item.get(key) not in (None, '', [], {}):
                value += points
        return value

    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        identity = identity_for(item)
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        if score(item) >= score(by_identity[identity]):
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _collect_terminal_successor_reopen_requests(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for item in graph.get('successor_reopen_requests') or []:
            if isinstance(item, Mapping):
                records.append(dict(item))
    if isinstance(diagnostics, Mapping):
        for item in diagnostics.get('graph_patch_successor_reopen_requests') or []:
            if isinstance(item, Mapping):
                records.append(dict(item))

    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        identity = (
            _clean_text(item.get('patch_id'))
            or _clean_text(item.get('proposal_id'))
            or _clean_text(item.get('idempotency_key'))
            or _digest(item)
        )
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        existing = by_identity[identity]
        if len(json.dumps(_json_safe(item), sort_keys=True)) >= len(json.dumps(_json_safe(existing), sort_keys=True)):
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _extract_terminal_successor_reopen_cases(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    requests = _collect_terminal_successor_reopen_requests(frame)
    if not requests:
        return []
    closure_status = _clean_text(graph_review.get('status') or graph_review.get('state')).lower().replace('-', '_')
    closure_closed = closure_status in {'fulfilled', 'closed', 'completed'}
    cases: list[dict[str, Any]] = []
    target_surfaces = [
        'ollmo_server/responses_request_runtime.py',
        'ollmo_services/response_frames.py',
        'ollmo_services/self_learning.py',
    ]
    for request in requests:
        status = _clean_text(request.get('status')).lower().replace('-', '_')
        patch_id = _clean_text(request.get('patch_id') or request.get('proposal_id')) or _digest(request)
        blocked_reasons = _graph_patch_blocked_reasons(request)
        metadata = {
            'patch_id': patch_id,
            'proposal_id': request.get('proposal_id'),
            'successor_status': status,
            'closure_status': closure_status,
            'parent_response_id': request.get('parent_response_id'),
            'parent_frame_id': request.get('parent_frame_id'),
            'parent_frame_sequence': request.get('parent_frame_sequence'),
            'repair_class': request.get('repair_class'),
            'autonomy_level': request.get('autonomy_level'),
            'runtime_effect': request.get('runtime_effect'),
            'blocked_reasons': blocked_reasons,
            'successor_reopen_request': dict(request),
        }
        if status in {'candidate', 'staged', 'applied_to_successor'} and not closure_closed:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_terminal_successor_reopen_created',
                    severity='medium',
                    summary='A terminal graph patch produced explicit successor/reopen truth instead of mutating the frozen parent frame.',
                    evidence=f'graph_patch.terminal_successor_reopen.created:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep terminal repair movement append-only: parent frames stay frozen while successor owed work remains visible to Late Fill and Closure.',
                    metadata=metadata,
                )
            )
        if (status in {'solved', 'fulfilled', 'closed'} or status == 'applied_to_successor') and closure_closed:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_terminal_successor_reopen_solved',
                    severity='positive',
                    summary='A terminal successor/reopen graph patch later reached fulfilled Closure truth.',
                    evidence=f'graph_patch.terminal_successor_reopen.solved:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Use as soft orientation for future runtime-backed terminal safe-additive reopen proposals.',
                    metadata=metadata,
                )
            )
        if status in {'blocked', 'failed', 'repair_needed'} or blocked_reasons:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_terminal_successor_reopen_blocked',
                    severity='high',
                    summary='A terminal successor/reopen graph patch remained blocked or repair-needed after successor movement.',
                    evidence=f'graph_patch.terminal_successor_reopen.blocked:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Preserve blocked successor evidence so future repair stays bounded and does not collapse unresolved work to completion.',
                    metadata=metadata,
                )
            )
    return cases


def _collect_graph_rebase_records(frame: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for item in graph.get('graph_rebase_proposals') or []:
            if isinstance(item, Mapping):
                proposals.append(dict(item))
        for item in graph.get('graph_rebase_reviews') or []:
            if isinstance(item, Mapping):
                reviews.append(dict(item))
    if isinstance(diagnostics, Mapping):
        for item in diagnostics.get('runtime_graph_rebase_proposals') or []:
            if isinstance(item, Mapping):
                proposals.append(dict(item))
        for item in diagnostics.get('runtime_graph_rebase_reviews') or []:
            if isinstance(item, Mapping):
                reviews.append(dict(item))

    def dedupe(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            identity = ''
            for key in keys:
                identity = _clean_text(item.get(key))
                if identity:
                    break
            if not identity:
                identity = _digest(item)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        return unique

    return dedupe(proposals, 'proposal_id', 'id'), dedupe(reviews, 'review_id', 'proposal_id', 'id')


def _collect_graph_rebase_lifecycle_records(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for key in ('graph_rebase_lifecycle', 'applied_graph_rebases', 'staged_graph_rebases'):
            for item in graph.get(key) or []:
                if isinstance(item, Mapping):
                    records.append(dict(item))
    if isinstance(diagnostics, Mapping):
        for key in (
            'graph_rebase_lifecycle',
            'graph_rebase_lifecycle_results',
            'applied_graph_rebases',
            'staged_graph_rebases',
        ):
            for item in diagnostics.get(key) or []:
                if isinstance(item, Mapping):
                    records.append(dict(item))

    def identity_for(item: Mapping[str, Any]) -> str:
        return (
            _clean_text(item.get('rebase_id'))
            or _clean_text(item.get('idempotency_key'))
            or _clean_text(item.get('proposal_id'))
            or _digest(item)
        )

    def score(item: Mapping[str, Any]) -> int:
        status = _clean_text(item.get('status')).lower().replace('-', '_')
        status_score = {
            'applied': 72,
            'already_applied': 70,
            'blocked': 64,
            'rejected': 64,
            'failed': 58,
            'invalid': 58,
            'staged': 35,
            'validated': 28,
            'proposed': 10,
        }.get(status, 0)
        value = status_score
        for key, points in (
            ('outcome', 10),
            ('risk_level', 6),
            ('blocked_reasons', 6),
            ('source_evidence_refs', 6),
            ('after_graph_digest', 4),
            ('before_graph_digest', 3),
            ('validation_review', 3),
            ('graph_rebase_authorization', 3),
        ):
            if item.get(key) not in (None, '', [], {}):
                value += points
        return value

    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        identity = identity_for(item)
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        if score(item) >= score(by_identity[identity]):
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _collect_successor_rebase_requests(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for item in graph.get('successor_rebase_requests') or []:
            if isinstance(item, Mapping):
                records.append(dict(item))
    if isinstance(diagnostics, Mapping):
        for item in diagnostics.get('successor_rebase_requests') or []:
            if isinstance(item, Mapping):
                records.append(dict(item))

    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        identity = (
            _clean_text(item.get('rebase_id'))
            or _clean_text(item.get('proposal_id'))
            or _clean_text(item.get('idempotency_key'))
            or _digest(item)
        )
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        existing = by_identity[identity]
        if len(json.dumps(_json_safe(item), sort_keys=True)) >= len(json.dumps(_json_safe(existing), sort_keys=True)):
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _collect_successor_rebase_executions(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect concrete partial-successor executions as observer evidence only."""

    records: list[dict[str, Any]] = []
    runtime = _nested_mapping(frame, 'runtime')
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    for graph in _graph_repair_runtime_graphs(frame):
        for item in graph.get('successor_rebase_executions') or []:
            if isinstance(item, Mapping):
                records.append(dict(item))
    if isinstance(diagnostics, Mapping):
        execution = diagnostics.get('graph_rebase_partial_successor_execution')
        if isinstance(execution, Mapping):
            records.append(dict(execution))
        for item in diagnostics.get('successor_rebase_executions') or []:
            if isinstance(item, Mapping):
                records.append(dict(item))
    late_fill_execution = _late_fill_truth(frame).get('partial_rebase_execution')
    if isinstance(late_fill_execution, Mapping):
        records.append(dict(late_fill_execution))

    records = [
        item
        for item in records
        if _clean_text(item.get('kind'))
        == 'ollmo.graph_rebase_partial_successor_execution'
    ]

    def identity_for(item: Mapping[str, Any]) -> str:
        return (
            _clean_text(item.get('execution_key'))
            or _clean_text(item.get('successor_key'))
            or _clean_text(item.get('rebase_id'))
            or _clean_text(item.get('proposal_id'))
            or _digest(item)
        )

    def preference(item: Mapping[str, Any]) -> tuple[int, int]:
        status = _clean_text(item.get('status')).lower().replace('-', '_')
        status_rank = {
            'completed': 80,
            'succeeded': 80,
            'solved': 80,
            'fulfilled': 80,
            'failed': 90,
            'cancelled': 90,
            'blocked': 90,
            'repair_needed': 90,
            'running': 40,
            'queued': 30,
            'pending': 20,
        }.get(status, 0)
        return status_rank, len(json.dumps(_json_safe(item), sort_keys=True))

    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        identity = identity_for(item)
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        if preference(item) >= preference(by_identity[identity]):
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _extract_graph_rebase_outcome_cases(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    proposals, reviews = _collect_graph_rebase_records(frame)
    lifecycle_records = _collect_graph_rebase_lifecycle_records(frame)
    successor_requests = _collect_successor_rebase_requests(frame)
    successor_executions = _collect_successor_rebase_executions(frame)
    proposal_by_id = {
        _clean_text(proposal.get('proposal_id') or proposal.get('id')): proposal
        for proposal in proposals
        if _clean_text(proposal.get('proposal_id') or proposal.get('id'))
    }
    cases: list[dict[str, Any]] = []
    closure_status = _clean_text(graph_review.get('status') or graph_review.get('state')).lower().replace('-', '_')
    closure_closed = closure_status in {'fulfilled', 'closed', 'completed'}
    target_surfaces = [
        'ollmo_services/graph_rebase.py',
        'ollmo_server/responses_request_runtime.py',
        'ollmo_services/self_learning.py',
    ]

    for review in reviews:
        proposal_id = _clean_text(review.get('proposal_id'))
        proposal = proposal_by_id.get(proposal_id)
        status = _clean_text(review.get('status') or review.get('review_status')).lower().replace('-', '_')
        blocked_reasons = _graph_patch_blocked_reasons(review)
        metadata = {
            'proposal_id': proposal_id,
            'review_status': status,
            'blocked_reasons': blocked_reasons,
            'proposal': dict(proposal or {}),
            'review': dict(review),
        }
        if status == 'accepted':
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_proposal_accepted',
                    severity='positive',
                    summary='A reviewed graph-rebase proposal passed runtime validation and preservation proof.',
                    evidence=f'graph_rebase.review.accepted:{proposal_id}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep as soft evidence for when reviewed rebase candidates preserve current runtime truth.',
                    metadata=metadata,
                )
            )
        elif status in {'rejected', 'blocked', 'failed', 'invalid'}:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_proposal_rejected',
                    severity='high' if 'runtime_rebase_evidence_missing' in blocked_reasons else 'medium',
                    summary='A graph-rebase proposal was rejected or blocked by runtime validation.',
                    evidence=f'graph_rebase.review.rejected:{proposal_id or _digest(review)}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Preserve rejection reasons so future rebase candidates stay reviewed, successor-only, and preservation-backed.',
                    metadata=metadata,
                )
            )
            if any(reason.startswith('lost_') or 'preservation' in reason for reason in blocked_reasons):
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_rebase',
                        kind='graph_rebase_preservation_failed',
                        severity='high',
                        summary='A graph-rebase candidate failed preservation proof for existing runtime truth.',
                        evidence=f'graph_rebase.preservation_failed:{proposal_id or _digest(review)}',
                        target_area='graph_rebase_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Use as soft calibration for rejecting rebase candidates that lose obligations, artifacts, review duties, or lineage.',
                        metadata=metadata,
                    )
                )
            if 'accepted_learning_not_rebase_authority' in blocked_reasons:
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_rebase',
                        kind='graph_rebase_rejected_learning_only',
                        severity='positive',
                        summary='Accepted learning was kept as a soft hint and did not authorize graph rebase.',
                        evidence=f'graph_rebase.rejected_learning_only:{proposal_id or _digest(review)}',
                        target_area='graph_rebase_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Keep accepted learning orientation-only; runtime evidence and rebase authorization remain mandatory.',
                        metadata=metadata,
                    )
                )
            if any('route_health' in reason or 'provider' in reason or 'degraded' in reason for reason in blocked_reasons):
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_rebase',
                        kind='graph_rebase_rejected_degraded_signal',
                        severity='positive',
                        summary='Route-health, degraded, cache, or provider evidence stayed non-executable for graph rebase.',
                        evidence=f'graph_rebase.rejected_degraded_signal:{proposal_id or _digest(review)}',
                        target_area='graph_rebase_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Keep provider and liveness signals as route diagnostics, not graph rebase authority.',
                        metadata=metadata,
                    )
                )

    for record in lifecycle_records:
        status = _clean_text(record.get('status')).lower().replace('-', '_')
        outcome = record.get('outcome') if isinstance(record.get('outcome'), Mapping) else {}
        runtime_effect = _clean_text(outcome.get('runtime_effect') or record.get('runtime_effect'))
        blocked_reasons = _graph_patch_blocked_reasons(record)
        rebase_id = _clean_text(record.get('rebase_id') or record.get('idempotency_key') or record.get('proposal_id')) or _digest(record)
        enforced_policy_review = (
            record.get('enforced_policy_review')
            if isinstance(record.get('enforced_policy_review'), Mapping)
            else {}
        )
        enforced_class = _clean_text(record.get('enforced_class') or enforced_policy_review.get('enforced_class'))
        metadata = {
            'rebase_id': rebase_id,
            'proposal_id': record.get('proposal_id'),
            'rebase_status': status,
            'autonomy_level': record.get('autonomy_level'),
            'risk_level': record.get('risk_level'),
            'enforced_class': enforced_class,
            'authority': record.get('authority'),
            'policy_mode': record.get('policy_mode') or enforced_policy_review.get('policy_mode'),
            'enforced_policy_review': dict(enforced_policy_review),
            'runtime_effect': runtime_effect,
            'closure_status': closure_status,
            'blocked_reasons': blocked_reasons,
            'rebase_lifecycle': dict(record),
        }
        if status in {'staged', 'validated'} and runtime_effect in {'staged_no_executable_mutation', 'shadow_no_mutation'}:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_staged',
                    severity='medium',
                    summary='A graph-rebase candidate was staged or shadow-validated without executable mutation.',
                    evidence=f'graph_rebase.staged:{rebase_id}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Inspect staged rebase candidates before allowing reviewed successor creation.',
                    metadata=metadata,
                )
            )
        if (
            status in {'blocked', 'rejected', 'failed', 'invalid'}
            and (
                record.get('autonomy_level') == 'apply_enforced'
                or 'apply_enforced_reserved' in blocked_reasons
            )
        ):
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_apply_enforced_blocked',
                    severity='positive',
                    summary='Graph rebase apply_enforced was blocked by runtime enforced policy.',
                    evidence=f'graph_rebase.apply_enforced_blocked:{rebase_id}',
                    target_area='apply_enforced_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep graph rebase successor movement reviewed unless an explicit policy later proves a narrower class safe.',
                    metadata=metadata,
                )
            )
            blocked_text = ' '.join(blocked_reasons).lower()
            if enforced_class == 'full_successor_rebase' or 'full_successor_rebase_not_enforced_v1' in blocked_text:
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_rebase',
                        kind='apply_enforced_full_rebase_blocked',
                        severity='positive',
                        summary='Full successor rebase stayed blocked under enforced v1.',
                        evidence=f'apply_enforced.graph_rebase.full_blocked:{rebase_id}',
                        target_area='apply_enforced_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Use only as soft calibration that full successor rebase remains reviewed/authorized, not enforced.',
                        metadata=metadata,
                    )
                )
            if enforced_class == 'partial_subtree_rebase_strict' or 'partial_subtree_rebase_enforced_v1_audit_only' in blocked_text:
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_rebase',
                        kind='apply_enforced_partial_rebase_audit_only',
                        severity='positive',
                        summary='Partial subtree rebase remained audit-only under enforced v1.',
                        evidence=f'apply_enforced.graph_rebase.partial_audit:{rebase_id}',
                        target_area='apply_enforced_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Keep partial rebase reviewed until preservation and scope evidence justify a separate policy expansion.',
                        metadata=metadata,
                    )
                )

    for request in successor_requests:
        status = _clean_text(request.get('status')).lower().replace('-', '_')
        rebase_id = _clean_text(request.get('rebase_id') or request.get('proposal_id') or request.get('idempotency_key')) or _digest(request)
        blocked_reasons = _graph_patch_blocked_reasons(request)
        metadata = {
            'rebase_id': rebase_id,
            'proposal_id': request.get('proposal_id'),
            'successor_status': status,
            'closure_status': closure_status,
            'parent_response_id': request.get('parent_response_id'),
            'parent_frame_id': request.get('parent_frame_id'),
            'runtime_effect': request.get('runtime_effect'),
            'blocked_reasons': blocked_reasons,
            'successor_rebase_request': dict(request),
        }
        if status in {'candidate', 'pending', 'staged', 'applied_to_successor'} and not closure_closed:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_successor_created',
                    severity='medium',
                    summary='A reviewed graph rebase produced explicit successor truth instead of mutating the parent graph.',
                    evidence=f'graph_rebase.successor_created:{rebase_id}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep reviewed rebase movement successor-only so Closure and Late Fill inspect the new owed work explicitly.',
                    metadata=metadata,
                )
            )
        if status in {'solved', 'fulfilled', 'closed'} and closure_closed:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_successor_solved',
                    severity='positive',
                    summary='A graph-rebase successor later reached fulfilled Closure truth.',
                    evidence=f'graph_rebase.successor_solved:{rebase_id}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Use as soft orientation for future reviewed rebase proposals that preserve obligations and artifacts.',
                    metadata=metadata,
                )
            )
        if status in {'blocked', 'failed', 'repair_needed'} or blocked_reasons:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_successor_blocked',
                    severity='high',
                    summary='A graph-rebase successor remained blocked or repair-needed.',
                    evidence=f'graph_rebase.successor_blocked:{rebase_id}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Preserve successor failure evidence so future rebase candidates do not hide unresolved graph work.',
                    metadata=metadata,
                )
            )

    late_fill = _late_fill_truth(frame)
    late_fill_status = _clean_text(late_fill.get('status')).lower().replace('-', '_')
    late_fill_succeeded = late_fill_status in {
        'completed',
        'fulfilled',
        'closed',
        'late_fill_completed',
    }
    late_fill_failed = late_fill_status in {
        'partial_failed',
        'failed',
        'cancelled',
        'blocked',
        'repair_needed',
        'late_fill_failed',
    }
    closure_blocked = closure_status in {
        'blocked',
        'failed',
        'cancelled',
        'repair_needed',
    }
    for execution in successor_executions:
        status = _clean_text(execution.get('status')).lower().replace('-', '_')
        execution_key = (
            _clean_text(execution.get('execution_key'))
            or _clean_text(execution.get('successor_key'))
            or _clean_text(execution.get('rebase_id'))
            or _digest(execution)
        )
        blocked_reasons = _graph_patch_blocked_reasons(execution)
        execution_succeeded = status in {
            'completed',
            'succeeded',
            'solved',
            'fulfilled',
            'closed',
        }
        execution_failed = status in {
            'failed',
            'cancelled',
            'blocked',
            'repair_needed',
        }
        root_prompt_replay = execution.get('root_prompt_replay') is True
        observed_solved = closure_closed and (execution_succeeded or late_fill_succeeded)
        observed_blocked = bool(
            execution_failed
            or late_fill_failed
            or blocked_reasons
            or root_prompt_replay
            or (closure_blocked and (execution_succeeded or late_fill_succeeded))
        )
        raw_scheduled_branch_ids = (
            execution.get('scheduled_branch_ids')
            if isinstance(execution.get('scheduled_branch_ids'), list)
            else []
        )
        scheduled_branch_ids = [
            _clean_text(item)
            for item in raw_scheduled_branch_ids
            if _clean_text(item)
        ]
        metadata = {
            'execution_key': execution_key,
            'successor_key': execution.get('successor_key'),
            'proposal_id': execution.get('proposal_id'),
            'review_id': execution.get('review_id'),
            'rebase_id': execution.get('rebase_id'),
            'authorization_record_id': execution.get('authorization_record_id'),
            'execution_status': status,
            'closure_status': closure_status,
            'late_fill_status': late_fill_status,
            'scheduled_branch_ids': scheduled_branch_ids,
            'scheduled_branch_count': len(scheduled_branch_ids),
            'partial_rebase_depth': execution.get('partial_rebase_depth'),
            'root_prompt_replay': execution.get('root_prompt_replay'),
            'blocked_reasons': blocked_reasons,
            'authority': 'non_authoritative_observer',
            'observer_runtime_effect': 'none',
            'successor_rebase_execution': dict(execution),
        }
        if observed_blocked:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_partial_successor_execution_blocked',
                    severity='high',
                    summary='A reviewed partial-rebase successor execution remained blocked, failed, or failed to close.',
                    evidence=f'graph_rebase.partial_successor_execution.blocked:{execution_key}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep this as offline outcome evidence only; inspect the exact branch-local execution and Closure failure without changing gates or authorization.',
                    metadata=metadata,
                )
            )
        elif observed_solved:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_partial_successor_execution_solved',
                    severity='positive',
                    summary='A reviewed partial-rebase successor execution reached fulfilled Closure truth.',
                    evidence=f'graph_rebase.partial_successor_execution.solved:{execution_key}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Use only as offline soft evidence for future review; the execution observation cannot promote a gate or authorize another rebase.',
                    metadata=metadata,
                )
            )
        else:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_rebase',
                    kind='graph_rebase_partial_successor_execution_created',
                    severity='medium',
                    summary='A concrete reviewed partial-rebase successor execution became visible to the observer.',
                    evidence=f'graph_rebase.partial_successor_execution.created:{execution_key}',
                    target_area='graph_rebase_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Observe the same successor through Late Fill and Closure; this record is evidence, not promotion or authorization authority.',
                    metadata=metadata,
                )
            )

    return cases


def _graph_repair_review_reasons(review: Mapping[str, Any]) -> list[str]:
    reasons = review.get('reasons') or review.get('reject_reasons') or review.get('issues')
    if isinstance(reasons, str):
        return [reasons] if reasons else []
    if isinstance(reasons, (list, tuple)):
        return [_clean_text(item) for item in reasons if _clean_text(item)]
    return []


def _graph_patch_blocked_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons = record.get('blocked_reasons') or record.get('reasons') or record.get('reject_reasons')
    if isinstance(reasons, str):
        return [reasons] if reasons else []
    if isinstance(reasons, (list, tuple)):
        return [_clean_text(item) for item in reasons if _clean_text(item)]
    return []


def _extract_graph_patch_lifecycle_cases(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = _collect_graph_patch_lifecycle_records(frame)
    cases: list[dict[str, Any]] = []
    closure_status = _clean_text(graph_review.get('status') or graph_review.get('state')).lower().replace('-', '_')
    closure_closed = closure_status in {'fulfilled', 'closed', 'completed'}
    closure_blocked = closure_status in {'blocked', 'pending', 'repair_needed', 'repair_required', 'review_pending'}
    target_surfaces = [
        'ollmo_services/graph_repair.py',
        'ollmo_server/responses_request_runtime.py',
        'ollmo_services/self_learning.py',
    ]

    for record in records:
        status = _clean_text(record.get('status')).lower().replace('-', '_')
        outcome = record.get('outcome') if isinstance(record.get('outcome'), Mapping) else {}
        outcome_status = _clean_text(outcome.get('status')).lower().replace('-', '_')
        repair_class = _clean_text(record.get('repair_class'))
        patch_id = _clean_text(record.get('patch_id') or record.get('idempotency_key') or record.get('proposal_id')) or _digest(record)
        blocked_reasons = _graph_patch_blocked_reasons(record)
        source_refs = [
            _clean_text(item)
            for item in (record.get('source_evidence_refs') or [])
            if _clean_text(item)
        ]
        enforced_policy_review = (
            record.get('enforced_policy_review')
            if isinstance(record.get('enforced_policy_review'), Mapping)
            else {}
        )
        enforced_class = _clean_text(record.get('enforced_class') or enforced_policy_review.get('enforced_class'))
        metadata = {
            'patch_id': patch_id,
            'proposal_id': record.get('proposal_id'),
            'repair_class': repair_class,
            'enforced_class': enforced_class,
            'patch_status': status,
            'outcome_status': outcome_status,
            'autonomy_level': record.get('autonomy_level'),
            'risk_level': record.get('risk_level'),
            'authority': record.get('authority'),
            'policy_mode': record.get('policy_mode') or enforced_policy_review.get('policy_mode'),
            'enforced_policy_review': dict(enforced_policy_review),
            'closure_status': closure_status,
            'blocked_reasons': blocked_reasons,
            'source_evidence_refs': source_refs,
            'patch_lifecycle': dict(record),
        }

        if status in {'applied', 'already_applied'} and closure_closed:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_applied_and_closed',
                    severity='positive',
                    summary='A validated graph patch was applied and Closure later reported the graph contract closed.',
                    evidence=f'graph_patch.applied_and_closed:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep as positive evidence for bounded additive patch application and closure feedback.',
                    metadata=metadata,
                )
            )
            if repair_class in {'missing_materialization_branch', 'missing_dependency_edge', 'artifact_binding_repair_branch'}:
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_repair',
                        kind='graph_patch_solved_missing_obligation',
                        severity='positive',
                        summary='A graph patch repaired a missing obligation or dependency that Closure later accepted.',
                        evidence=f'graph_patch.solved_missing_obligation:{patch_id}',
                        target_area='graph_repair_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Use as soft orientation for future runtime-backed missing-obligation repair proposals.',
                        metadata=metadata,
                    )
                )
        elif status in {'applied', 'already_applied'} and (closure_blocked or outcome_status in {'blocked', 'repair_needed'}):
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_applied_but_blocked',
                    severity='high',
                    summary='A validated graph patch was applied but Closure still reported blocked or repair-needed work.',
                    evidence=f'graph_patch.applied_but_blocked:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Preserve dependency, scheduling, and closure evidence so future patches remain bounded and do not hide unresolved work.',
                    metadata=metadata,
                )
            )

        false_work = outcome.get('false_work') is True or outcome_status in {'created_false_work', 'false_work', 'unnecessary'}
        if false_work:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_created_false_work',
                    severity='high',
                    summary='A graph patch created work that later evidence judged unnecessary.',
                    evidence=f'graph_patch.created_false_work:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Tighten validation so future patches require stronger current runtime evidence and do not manufacture obligations.',
                    metadata=metadata,
                )
            )

        conflict_text = ' '.join(blocked_reasons).lower()
        if status in {'blocked', 'rejected', 'failed', 'invalid'} and any(
            token in conflict_text
            for token in ('conflict', 'deferred', 'reserved', 'explicit_user_intent', 'replace_user_intent')
        ) or (
            status in {'blocked', 'rejected', 'failed', 'invalid'}
            and 'requires_explicit_review_authorization' in conflict_text
        ):
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_rejected_due_to_conflict',
                    severity='medium',
                    summary='A graph patch was rejected or blocked because it conflicted with preserved intent or validation boundaries.',
                    evidence=f'graph_patch.rejected_due_to_conflict:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Keep conflict reasons as calibration evidence while preserving the rule that learning cannot validate patches.',
                    metadata=metadata,
                )
            )

        degraded_text = ' '.join([repair_class, conflict_text, ' '.join(source_refs)]).lower()
        if status in {'blocked', 'rejected', 'failed', 'invalid'} and (
            repair_class == 'degraded_liveness_only'
            or 'degraded' in degraded_text
            or 'liveness' in degraded_text
        ):
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_patch_degraded_signal_ignored',
                    severity='positive',
                    summary='A degraded/cache/liveness signal stayed advisory and did not mutate the graph.',
                    evidence=f'graph_patch.degraded_signal_ignored:{patch_id}',
                    target_area='graph_repair_policy',
                    target_surfaces=target_surfaces,
                    suggested_action='Preserve degraded and liveness evidence as route-health diagnostics, not graph repair proof.',
                    metadata=metadata,
                )
            )

        if record.get('autonomy_level') == 'apply_enforced':
            if status in {'applied', 'already_applied'} and closure_closed:
                solved_kind = (
                    'apply_enforced_artifact_identity_solved'
                    if enforced_class == 'duplicate_artifact_alias_canonicalization'
                    else 'apply_enforced_safe_additive_solved'
                )
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_repair',
                        kind=solved_kind,
                        severity='positive',
                        summary='An enforced-policy allowlisted graph patch solved under later Closure truth.',
                        evidence=f'apply_enforced.graph_patch.solved:{patch_id}',
                        target_area='apply_enforced_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Keep this as soft outcome calibration only; accepted learning cannot satisfy enforced gates.',
                        metadata=metadata,
                    )
                )
            elif status in {'blocked', 'rejected', 'failed', 'invalid'}:
                blocked_text = ' '.join(blocked_reasons).lower()
                if 'accepted_learning' in blocked_text or 'learning_only' in blocked_text:
                    blocked_kind = 'apply_enforced_learning_only_rejected'
                elif 'degraded' in blocked_text or 'provider' in blocked_text or 'route_health' in blocked_text:
                    blocked_kind = 'apply_enforced_degraded_signal_rejected'
                elif 'conflicting_duplicate_artifact_ref' in blocked_text:
                    blocked_kind = 'apply_enforced_conflicting_artifact_ref_blocked'
                elif 'enforced_policy_audit_only' in blocked_text:
                    blocked_kind = 'apply_enforced_policy_audit_only'
                else:
                    blocked_kind = 'apply_enforced_policy_blocked'
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_repair',
                        kind=blocked_kind,
                        severity='positive',
                        summary='An apply_enforced graph patch stayed blocked by the runtime policy gate.',
                        evidence=f'apply_enforced.graph_patch.blocked:{patch_id}',
                        target_area='apply_enforced_policy',
                        target_surfaces=target_surfaces,
                        suggested_action='Preserve the blocked policy review as calibration; learning may orient attention but cannot authorize enforcement.',
                        metadata=metadata,
                    )
                )

    return cases


def _extract_graph_repair_outcome_cases(
    frame: Mapping[str, Any],
    graph_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    proposals, reviews = _collect_graph_repair_records(frame)
    proposal_by_id = {
        _clean_text(proposal.get('proposal_id') or proposal.get('id')): proposal
        for proposal in proposals
        if _clean_text(proposal.get('proposal_id') or proposal.get('id'))
    }
    cases: list[dict[str, Any]] = []
    surface_state = (
        graph_review.get('surface_state')
        if isinstance(graph_review.get('surface_state'), Mapping)
        else {}
    )
    late_fill = frame.get('late_fill') if isinstance(frame.get('late_fill'), Mapping) else {}
    surface_actionability = classify_surface_repair_actionability(
        surface_state,
        closure_review=graph_review,
        late_fill=late_fill,
    ) if surface_state else {'status': 'neutral'}

    actionable_categories = {
        _clean_text(item).lower()
        for item in (surface_actionability.get('actionable_categories') or [])
        if _clean_text(item)
    }
    evidence_refs = {
        _clean_text(item)
        for item in (surface_actionability.get('evidence_refs') or [])
        if _clean_text(item)
    }
    late_fill_status = _clean_text(late_fill.get('status') or late_fill.get('state')).lower()
    active_surface_categories = {
        _clean_text(item).lower()
        for item in (surface_state.get('active_categories') or [])
        if _clean_text(item)
    }
    active_late_fill = (
        late_fill_status in {
            'pending',
            'queued',
            'running',
            'late_fill_pending',
            'late_fill_running',
        }
        or bool(active_surface_categories & {'late_fill_pending', 'late_fill_running'})
    )
    expected_active_continuation = (
        active_late_fill
        and bool(actionable_categories)
        and actionable_categories <= {'open', 'materialization_contract_unmet'}
    )
    surface_reason = _clean_text(surface_state.get('reason')).lower()
    missing_source_truth_guard = (
        bool(actionable_categories)
        and actionable_categories <= {'repair_needed', 'repair_pending'}
        and any(ref.startswith('closure_review:repair_action:truth_guard') for ref in evidence_refs)
        and (
            'without a current or selected source' in surface_reason
            or 'missing source' in surface_reason
            or 'select source' in surface_reason
        )
    )

    matched_proposal_ids: set[str] = set()
    for review in reviews:
        proposal_id = _clean_text(review.get('proposal_id'))
        proposal = proposal_by_id.get(proposal_id)
        status = _clean_text(review.get('status') or review.get('review_status')).lower()
        reasons = _graph_repair_review_reasons(review)
        if proposal:
            matched_proposal_ids.add(proposal_id)
            if status == 'accepted':
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_repair',
                        kind='graph_repair_proposal_accepted',
                        severity='positive',
                        summary='A graph-repair proposal passed runtime validation.',
                        evidence=f'graph_repair.review.accepted:{proposal_id}',
                        target_area='graph_repair_policy',
                        target_surfaces=[
                            'ollmo_services/graph_repair.py',
                            'scripts/ollmo_run_monitor.py',
                            'ollmo_server/response_semantics_runtime.py',
                        ],
                        suggested_action='Keep as a positive trace for proposal validation and idempotent additive patch policy.',
                        metadata={
                            'proposal_id': proposal_id,
                            'repair_type': proposal.get('repair_type'),
                            'review_status': status,
                            'proposal': dict(proposal),
                            'review': dict(review),
                        },
                    )
                )
            elif status in {'rejected', 'blocked', 'failed', 'invalid'}:
                cases.append(
                    _case(
                        frame=frame,
                        layer='graph_repair',
                        kind='graph_repair_proposal_rejected',
                        severity='high' if 'runtime_evidence_missing' in reasons else 'medium',
                        summary='A graph-repair proposal was rejected by runtime validation.',
                        evidence=f'graph_repair.review.rejected:{proposal_id}',
                        target_area='graph_repair_policy',
                        target_surfaces=[
                            'ollmo_services/graph_repair.py',
                            'scripts/ollmo_run_monitor.py',
                            'ollmo_services/self_learning.py',
                        ],
                        suggested_action='Preserve rejection reasons so future proposals become more precise without granting learning patch authority.',
                        metadata={
                            'proposal_id': proposal_id,
                            'repair_type': proposal.get('repair_type'),
                            'review_status': status,
                            'reasons': reasons,
                            'proposal': dict(proposal),
                            'review': dict(review),
                        },
                    )
                )
                if (
                    _clean_text(proposal.get('repair_type')) == 'reconcile_surface_state_or_reopen_contract'
                    and surface_actionability.get('status') == 'advisory'
                ):
                    cases.append(
                        _case(
                            frame=frame,
                            layer='graph_repair',
                            kind='graph_repair_false_positive_advisory_surface',
                            severity='high',
                            summary='A fulfilled run produced a surface-reconcile repair proposal from advisory-only movement state.',
                            evidence=f'graph_repair.false_positive_advisory_surface:{proposal_id}',
                            target_area='graph_repair_policy',
                            target_surfaces=[
                                'ollmo_services/graph_repair.py',
                                'scripts/ollmo_run_monitor.py',
                                'ollmo_server/response_semantics_runtime.py',
                            ],
                            suggested_action='Keep advisory movement visible but suppress repair proposals unless runtime-backed actionability is present.',
                            metadata={
                                'proposal_id': proposal_id,
                                'reasons': reasons,
                                'surface_actionability': surface_actionability.get('status'),
                                'surface_state': dict(surface_state),
                            },
                        )
                    )
        else:
            cases.append(
                _case(
                    frame=frame,
                    layer='graph_repair',
                    kind='graph_repair_review_unmatched',
                    severity='medium',
                    summary='A graph-repair review referenced a proposal id that was not visible in the same frame graph evidence.',
                    evidence=f'graph_repair.review.unmatched:{proposal_id or _digest(review)}',
                    target_area='graph_repair_policy',
                    target_surfaces=[
                        'scripts/ollmo_run_monitor.py',
                        'ollmo_services/graph_repair.py',
                    ],
                    suggested_action='Pair monitor and learning summaries by proposal_id so review evidence is not attributed to the wrong repair.',
                    metadata={
                        'proposal_id': proposal_id,
                        'review': dict(review),
                        'reasons': reasons,
                    },
                )
            )

    if (
        surface_actionability.get('status') == 'actionable'
        and not proposals
        and not expected_active_continuation
        and not missing_source_truth_guard
    ):
        cases.append(
            _case(
                frame=frame,
                layer='graph_repair',
                kind='graph_repair_missing_despite_evidence',
                severity='high',
                summary='Runtime truth showed actionable repair surface evidence but no graph-repair proposal was visible.',
                evidence='graph_repair.missing_despite_actionable_surface_evidence',
                target_area='graph_repair_policy',
                target_surfaces=[
                    'ollmo_services/graph_repair.py',
                    'scripts/ollmo_run_monitor.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Bridge current runtime/Closure/monitor evidence into proposal-only graph repair without bypassing validation.',
                metadata={
                    'surface_actionability': surface_actionability.get('status'),
                    'actionable_categories': surface_actionability.get('actionable_categories'),
                    'evidence_refs': surface_actionability.get('evidence_refs'),
                    'surface_state': dict(surface_state),
                },
            )
        )

    return cases


def build_eval_cases_from_response_frame(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract proposal-only self-learning eval cases from one frozen frame."""

    if not isinstance(frame, Mapping):
        return []
    graph_review = _nested_mapping(frame, 'runtime', 'graph_closure_review')
    context_contract = _nested_mapping(frame, 'planning', 'context_contract')
    cases: list[dict[str, Any]] = []
    if graph_review:
        cases.extend(_extract_intake_cases(frame, graph_review))
        cases.extend(_extract_closure_cases(frame, graph_review))
        cases.extend(_extract_positive_fulfillment_case(frame, graph_review))
        cases.extend(_extract_semantic_decision_cases(frame, graph_review))
        cases.extend(_extract_controlled_attention_cases(frame, graph_review))
        cases.extend(_extract_aspiration_cases(frame, graph_review))
        cases.extend(_extract_commitment_cases(frame, graph_review))
        cases.extend(_extract_global_semantic_closure_cases(frame, graph_review))
        cases.extend(_extract_branch_semantic_verdict_cases(frame, graph_review))
        cases.extend(_extract_redraw_scope_ladder_cases(frame, graph_review))
        cases.extend(_extract_graph_repair_outcome_cases(frame, graph_review))
        cases.extend(_extract_graph_patch_lifecycle_cases(frame, graph_review))
        cases.extend(_extract_terminal_successor_reopen_cases(frame, graph_review))
        cases.extend(_extract_graph_rebase_outcome_cases(frame, graph_review))
    if context_contract:
        cases.extend(_extract_context_cases(frame, context_contract))
    cases.extend(_extract_artifact_request_cases(frame))
    cases.extend(_extract_late_fill_contract_cases(frame))
    cases.extend(_extract_artifact_cases(frame))
    return cases


def _dedupe_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        key = (
            _clean_text(case.get('response_id')),
            _clean_text(case.get('case_kind')),
            _clean_text(case.get('evidence')),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(case))
    return deduped


def _monitor_case_frame(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'response_id': _clean_text(report.get('response_id')) or f'monitor-{_digest(report)}',
        'status': _clean_text(report.get('status')),
        'current_state': {
            'lifecycle_state': _clean_text(report.get('lifecycle_state')),
        },
        'late_fill': {
            'status': _clean_text(report.get('late_fill_status')),
            'failed_branch_count': report.get('failed_branch_count'),
            'final_materialization_contract_status': report.get('final_materialization_contract_status'),
            'materialization_contract_unmet': report.get('materialization_contract_unmet'),
        },
    }


def build_eval_cases_from_monitor_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract supporting eval cases from a monitor report.

    Monitor reports are diagnostic evidence. Frozen frames remain canonical runtime truth.
    """

    if not isinstance(report, Mapping):
        return []
    frame = _monitor_case_frame(report)
    cases: list[dict[str, Any]] = []
    artifacts = report.get('artifacts') if isinstance(report.get('artifacts'), Mapping) else {}
    notes = report.get('notes') if isinstance(report.get('notes'), list) else []
    branch_reports = _nested_list(report, 'timing_diagnostics', 'branches')
    note_text = '\n'.join(_iter_text_fragments(notes))
    branch_text = _text_from_payload(branch_reports[:8])
    monitor_text = '\n'.join([note_text, branch_text, _text_from_payload(artifacts)])

    if report.get('materialization_contract_unmet') is True or _clean_text(report.get('final_materialization_contract_status')).lower() == 'unmet':
        cases.append(
            _case(
                frame=frame,
                layer='closure',
                kind='materialization_contract_unmet',
                severity='high',
                summary='Monitor evidence reported an unmet final materialization contract.',
                evidence='monitor.materialization_contract_unmet',
                target_area='closure_review_policy',
                target_surfaces=[
                    'scripts/ollmo_run_monitor.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Use as supporting evidence for closure/materialization contract eval cases; confirm with response-frame truth.',
                metadata={
                    'truth_source': 'monitor_supporting_evidence',
                    'verdict': report.get('verdict'),
                    'late_fill_status': report.get('late_fill_status'),
                    'final_materialization_contract_status': report.get('final_materialization_contract_status'),
                    'failed_branch_count': report.get('failed_branch_count'),
                },
            )
        )

    html_issues = artifacts.get('html_issues') if isinstance(artifacts.get('html_issues'), list) else []
    for issue in html_issues:
        issue_text = _clean_text(issue)
        if '<can' not in issue_text and 'unsupported href element' not in issue_text.lower():
            continue
        cases.append(
            _case(
                frame=frame,
                layer='artifact',
                kind='html_navigation_tag_typo',
                severity='high',
                summary='Monitor syntax check found an unsupported navigation element such as <can>.',
                evidence='monitor.html_issues.unsupported_href_element',
                target_area='artifact_fulfillment_policy',
                target_surfaces=[
                    'scripts/ollmo_run_monitor.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Use as supporting syntax-repair evidence; response-frame artifact truth remains canonical.',
                metadata={
                    'truth_source': 'monitor_supporting_evidence',
                    'html_issue': issue_text,
                },
            )
        )
        break

    html_image_links = artifacts.get('html_image_links') if isinstance(artifacts.get('html_image_links'), list) else []
    missing_links = [
        link for link in html_image_links
        if isinstance(link, Mapping) and link.get('exists') is False
    ]
    if missing_links:
        cases.append(
            _case(
                frame=frame,
                layer='artifact',
                kind='broken_artifact_dependency_link',
                severity='high',
                summary='Monitor evidence found HTML image links that did not resolve to saved artifacts.',
                evidence='monitor.html_image_links.missing',
                target_area='artifact_fulfillment_policy',
                target_surfaces=[
                    'scripts/ollmo_run_monitor.py',
                    'ollmo_services/artifact_registry.py',
                    'ollmo_server/response_semantics_runtime.py',
                ],
                suggested_action='Use as supporting link-rebind evidence; confirm saved artifact paths through frame/registry truth.',
                metadata={
                    'truth_source': 'monitor_supporting_evidence',
                    'missing_links': missing_links[:5],
                },
            )
        )

    if '500' in monitor_text and '/v1/chat/completions' in monitor_text:
        cases.append(
            _case(
                frame=frame,
                layer='workload',
                kind=CHAT_ROUTE_HEALTH_CASE_KIND,
                severity='medium',
                summary='Monitor evidence reported a backend 500 on a chat-completions text branch; this is supporting route-health preference evidence only.',
                evidence='monitor.chat_completions_500',
                target_area='workload_decision_policy',
                target_surfaces=[
                    'scripts/ollmo_run_monitor.py',
                    'ollmo_server/late_fill_runtime.py',
                    'ollmo_server/backend_transport_runtime.py',
                ],
                suggested_action='Use as supporting non-authoritative route-health preference evidence for chat/coalesced text repair; never as provider-ban, offline, hard degraded, graph-repair, or route-mutation truth.',
                metadata={
                    'truth_source': 'monitor_supporting_evidence',
                    'backend_family_hint': _extract_backend_family_hint(monitor_text, frame, frame.get('late_fill') or {}),
                    'verdict': report.get('verdict'),
                    'evidence_preview': monitor_text[:800],
                },
            )
        )

    branch_error_seen = any(
        isinstance(branch, Mapping) and _clean_text(branch.get('status')).lower() in {'error', 'failed'}
        for branch in branch_reports
    )
    if _clean_text(report.get('verdict')).lower() == 'clean' and (branch_error_seen or 'failed' in note_text.lower()):
        cases.append(
            _case(
                frame=frame,
                layer='closure',
                kind='nonterminal_failed_branch_with_fulfilled_contract',
                severity='low',
                summary='Monitor evidence showed a clean final outcome with non-terminal branch failure evidence.',
                evidence='monitor.clean_with_nonterminal_failure_evidence',
                target_area='closure_review_policy',
                target_surfaces=[
                    'scripts/ollmo_run_monitor.py',
                    'docs/RESPONSES_CONTRACT.md',
                ],
                suggested_action='Preserve as status-reconciliation learning, not as a hard final failure.',
                metadata={
                    'truth_source': 'monitor_supporting_evidence',
                    'branch_error_seen': branch_error_seen,
                    'notes': notes[:5],
                },
            )
        )
    return cases


def enrich_eval_cases_from_monitor_reports(
    cases: list[dict[str, Any]],
    monitor_reports: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    enriched = list(cases)
    for report in monitor_reports:
        if isinstance(report, Mapping):
            enriched.extend(build_eval_cases_from_monitor_report(report))
    return _dedupe_cases(enriched)


def _count_by(cases: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(
        _canonical_case_kind(item.get(key)) or 'unknown'
        if key == 'case_kind'
        else _clean_text(item.get(key)) or 'unknown'
        for item in cases
        if isinstance(item, Mapping)
    )
    return dict(sorted(counts.items()))


def build_policy_improvement_candidates(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster eval cases into reviewed-patch improvement candidates."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        if _clean_text(case.get('severity')) == 'positive':
            continue
        target_area = _clean_text(case.get('target_area')) or 'general_runtime_policy'
        grouped.setdefault(target_area, []).append(case)

    candidates: list[dict[str, Any]] = []
    for target_area, items in sorted(grouped.items()):
        surfaces: list[str] = []
        evidence_ids: list[str] = []
        for item in items:
            evidence_id = _clean_text(item.get('case_id'))
            if evidence_id:
                evidence_ids.append(evidence_id)
            for surface in item.get('target_surfaces') or []:
                token = _clean_text(surface)
                if token and token not in surfaces:
                    surfaces.append(token)
        candidate_id = f'policy-improvement-{target_area}'
        candidates.append(
            {
                'kind': 'ollmo.policy_improvement_candidate',
                'candidate_id': candidate_id,
                'status': 'proposed',
                'target_area': target_area,
                'case_count': len(items),
                'case_kinds': _count_by(items, 'case_kind'),
                'severity_counts': _count_by(items, 'severity'),
                'target_surfaces': surfaces[:10],
                'evidence_case_ids': evidence_ids[:12],
                'optimization_policy': 'proposal_only_reviewed_patch_required',
                'summary': f'{len(items)} eval case(s) suggest reviewing {target_area}.',
            }
        )
    return candidates


def _shadow_suggestion_for_candidate(candidate: Mapping[str, Any]) -> str:
    target_area = _clean_text(candidate.get('target_area')) or 'runtime_policy'
    case_kinds = _canonical_case_kinds(candidate.get('case_kinds'))
    if case_kinds.get('intent_graph_inadequacy'):
        return 'When Closure shows the basic current-turn intent was not represented in the graph, propose a bounded graph repair from runtime evidence instead of freezing or learning it as executable truth.'
    if case_kinds.get('open_graph_obligation'):
        return 'When promoted graph obligations remain open at Closure, keep the basic intent visible and route repair through validated graph patches, late fill, waiver, or supersession rather than prose completion.'
    if case_kinds.get('artifact_request_collapsed_to_plain_chat'):
        return 'When a current turn explicitly requests saved files or generated media artifacts, do not close as single-phase chat unless promotion review records a waiver.'
    if case_kinds.get('artifact_control_json_leaked_to_user'):
        return 'Keep request IR, output obligations, route JSON, and candidate graph data out of visible user artifact/content output.'
    if case_kinds.get('materialization_contract_unmet'):
        return 'Before terminal freeze, verify final materialization contract truth and keep repair-needed state visible when artifacts or links remain unmet.'
    if case_kinds.get('html_navigation_tag_typo'):
        return 'Treat unsupported navigation elements such as <can href=...> as deterministic HTML repair defects; replace with valid <a> anchors.'
    if case_kinds.get(CHAT_ROUTE_HEALTH_CASE_KIND):
        return 'Treat repeated chat-completion failures as route-health preference evidence for this workload; prefer robust compatible chat/text-repair routes only when current runtime truth permits it, and do not infer provider bans, offline state, hard degraded truth, or graph repair proof from this hint.'
    if case_kinds.get('nonterminal_failed_branch_with_fulfilled_contract'):
        return 'Reconcile superseded or canonical-evidence-satisfied branch failures separately from final contract fulfillment.'
    if case_kinds.get('broken_artifact_dependency_link'):
        return 'Run deterministic linked-artifact rebind before closure when saved HTML/CSS/media references point at missing or placeholder dependencies.'
    if case_kinds.get('graph_repair_false_positive_advisory_surface'):
        return 'Do not turn advisory-only pending surface state into graph repair; require runtime-backed blocked, repair, semantic-review, dependency, artifact, or obligation evidence.'
    if case_kinds.get('graph_repair_missing_despite_evidence'):
        return 'When runtime truth shows actionable blocked, repair, semantic-review, dependency, artifact, or obligation evidence, produce a proposal-only graph repair that still requires validation.'
    if case_kinds.get('graph_repair_proposal_rejected') or case_kinds.get('graph_repair_review_unmatched'):
        return 'Preserve graph-repair proposal ids, review ids, and rejection reasons so future repair proposals can be calibrated without making learning executable authority.'
    if any(
        case_kinds.get(kind)
        for kind in (
            'redraw_scope_selected',
            'redraw_scope_partial_subtree_rebase_selected',
            'redraw_scope_full_successor_rebase_selected',
            'redraw_scope_learning_orientation_soft_hint',
            'redraw_scope_non_authoritative_evidence_ignored',
            'redraw_scope_duplicate_artifact_ref_conflict',
        )
    ):
        return 'Use redraw-scope learning only to orient the next scope review; the current Intent Contract, Runtime/Closure evidence, validators, and explicit rebase authorization still decide executable graph movement.'
    if any(
        case_kinds.get(kind)
        for kind in (
            'graph_rebase_partial_successor_execution_created',
            'graph_rebase_partial_successor_execution_solved',
            'graph_rebase_partial_successor_execution_blocked',
        )
    ):
        return 'Use partial-rebase successor execution outcomes only as offline soft evidence; they cannot promote a rollout gate, authorize another proposal, or replace current Runtime/Closure proof.'
    return f'Review {target_area} using the extracted eval-case evidence before accepting any policy change.'


def build_shadow_learning_hints(
    report: Mapping[str, Any],
    snapshot: Optional[Mapping[str, Any]] = None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Build diagnostic-only learning hints for shadow/eval mode."""

    candidates = report.get('improvement_candidates') if isinstance(report.get('improvement_candidates'), list) else []
    snapshot_enabled = isinstance(snapshot, Mapping) and snapshot.get('enabled') is True
    hints: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        target_area = _clean_text(candidate.get('target_area'))
        evidence_case_ids = candidate.get('evidence_case_ids') if isinstance(candidate.get('evidence_case_ids'), list) else []
        hint = {
            'kind': 'ollmo.self_learning_shadow_hint',
            'hint_id': f'shadow-{target_area or "runtime_policy"}-{_digest(candidate)}',
            'target_area': target_area or 'runtime_policy',
            'authority': 'shadow',
            'runtime_effect': 'none',
            'suggestion': _shadow_suggestion_for_candidate(candidate),
            'evidence_case_ids': evidence_case_ids[:8],
            'case_kinds': _canonical_case_kinds(candidate.get('case_kinds')),
            'conflict_boundary': 'Ignored whenever current user intent, live capability evidence, Graph, IR, Closure Review, output obligations, artifact evidence, or runtime truth conflict.',
            'accepted_snapshot_enabled': snapshot_enabled,
        }
        hints.append(_json_safe(hint))
        if len(hints) >= max(1, int(limit)):
            break
    return hints


def _find_policy_improvement_candidate(
    report: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    target_id = _clean_text(candidate_id)
    candidates = report.get('improvement_candidates') if isinstance(report.get('improvement_candidates'), list) else []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if _clean_text(candidate.get('candidate_id')) == target_id:
            return dict(candidate)
    raise ValueError(f'policy improvement candidate not found: {target_id}')


def _learning_id_for_candidate(candidate: Mapping[str, Any]) -> str:
    candidate_id = _clean_text(candidate.get('candidate_id')) or f'candidate-{_digest(candidate)}'
    return f'accepted-{candidate_id}'


def _bounded_hint_from_candidate(candidate: Mapping[str, Any]) -> str:
    target_area = _clean_text(candidate.get('target_area')) or 'runtime_policy'
    suggestion = _shadow_suggestion_for_candidate(candidate)
    generic_suffix = 'using the extracted eval-case evidence before accepting any policy change.'
    if suggestion and generic_suffix not in suggestion:
        return suggestion
    summary = _clean_text(candidate.get('summary'))
    if summary:
        return f'Review {target_area}: {summary}'
    return f'Review {target_area} using the accepted eval-case evidence.'


def _runtime_hint_from_accepted_learning(item: Mapping[str, Any]) -> str:
    case_kinds = _canonical_case_kinds(item.get('case_kinds'))
    if case_kinds.get(CHAT_ROUTE_HEALTH_CASE_KIND):
        return _shadow_suggestion_for_candidate({'case_kinds': case_kinds})
    return _clean_text(item.get('bounded_hint')) or _clean_text(item.get('summary'))


def accepted_learning_from_policy_candidate(
    candidate: Mapping[str, Any],
    *,
    reviewer: str = 'operator',
    review_note: str = '',
    accepted_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError('candidate must be a JSON object')
    target_area = _clean_text(candidate.get('target_area'))
    if not target_area:
        raise ValueError('candidate is missing target_area')
    candidate_id = _clean_text(candidate.get('candidate_id')) or f'policy-improvement-{target_area}'
    return _json_safe(
        {
            'kind': 'ollmo.accepted_learning',
            'learning_id': _learning_id_for_candidate(candidate),
            'status': 'accepted',
            'candidate_id': candidate_id,
            'target_area': target_area,
            'summary': _clean_text(candidate.get('summary')) or f'Accepted learning for {target_area}.',
            'bounded_hint': _bounded_hint_from_candidate(candidate),
            'allowed_use': 'soft_hint_only',
            'forbidden_use': 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth',
            'target_surfaces': candidate.get('target_surfaces') if isinstance(candidate.get('target_surfaces'), list) else [],
            'evidence_case_ids': candidate.get('evidence_case_ids') if isinstance(candidate.get('evidence_case_ids'), list) else [],
            'case_kinds': _canonical_case_kinds(candidate.get('case_kinds')),
            'severity_counts': candidate.get('severity_counts') if isinstance(candidate.get('severity_counts'), Mapping) else {},
            'review': {
                'reviewer': _clean_text(reviewer) or 'operator',
                'review_note': _clean_text(review_note),
                'accepted_at': _clean_text(accepted_at) or _now_iso_utc(),
            },
        }
    )


def promote_policy_improvement_candidate(
    snapshot: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    *,
    reviewer: str = 'operator',
    review_note: str = '',
    accepted_at: str | None = None,
) -> dict[str, Any]:
    base = dict(snapshot) if isinstance(snapshot, Mapping) else build_default_accepted_learning_policy_snapshot()
    learning = accepted_learning_from_policy_candidate(
        candidate,
        reviewer=reviewer,
        review_note=review_note,
        accepted_at=accepted_at,
    )
    learning_id = _clean_text(learning.get('learning_id'))
    existing = base.get('accepted_learnings') if isinstance(base.get('accepted_learnings'), list) else []
    updated_learnings: list[dict[str, Any]] = []
    replaced = False
    for item in existing:
        if not isinstance(item, Mapping):
            continue
        if _clean_text(item.get('learning_id')) == learning_id:
            updated_learnings.append(learning)
            replaced = True
        else:
            updated_learnings.append(_canonical_accepted_learning_record(item))
    if not replaced:
        updated_learnings.append(learning)

    base.update(
        {
            'kind': 'ollmo.accepted_learning_policy_snapshot',
            'snapshot_version': SELF_LEARNING_VERSION,
            'accepted_learnings': updated_learnings,
            'accepted_learning_count': len(updated_learnings),
            'authority': _normalize_accepted_learning_authority(base.get('authority')),
            'status': _clean_text(base.get('status')) or 'disabled',
            'enabled': base.get('enabled') is True,
            'runtime_effect': 'readable_policy_input' if base.get('enabled') is True else 'none',
            'activation_policy': (
                _clean_text(base.get('activation_policy'))
                or 'disabled_until_explicit_review'
            ),
        }
    )
    if base.get('enabled') is not True:
        base['enabled'] = False
        base['runtime_effect'] = 'none'
        base['activation_policy'] = 'disabled_until_explicit_review'
    return _json_safe(base)


def promote_policy_improvement_candidate_from_report(
    report: Mapping[str, Any],
    *,
    candidate_id: str,
    snapshot: Mapping[str, Any] | None = None,
    reviewer: str = 'operator',
    review_note: str = '',
    accepted_at: str | None = None,
) -> dict[str, Any]:
    candidate = _find_policy_improvement_candidate(report, candidate_id)
    return promote_policy_improvement_candidate(
        snapshot,
        candidate,
        reviewer=reviewer,
        review_note=review_note,
        accepted_at=accepted_at,
    )


def set_accepted_learning_policy_enabled(
    snapshot: Mapping[str, Any] | None,
    *,
    enabled: bool,
    reviewer: str = 'operator',
    reason: str = '',
    changed_at: str | None = None,
) -> dict[str, Any]:
    base = dict(snapshot) if isinstance(snapshot, Mapping) else build_default_accepted_learning_policy_snapshot()
    accepted_learnings = base.get('accepted_learnings') if isinstance(base.get('accepted_learnings'), list) else []
    next_enabled = bool(enabled)
    authority = _normalize_accepted_learning_authority(base.get('authority'))
    base.update(
        {
            'kind': 'ollmo.accepted_learning_policy_snapshot',
            'snapshot_version': SELF_LEARNING_VERSION,
            'authority': authority,
            'enabled': next_enabled,
            'status': 'enabled' if next_enabled else 'disabled',
            'activation_policy': 'explicitly_enabled_by_review' if next_enabled else 'disabled_until_explicit_review',
            'runtime_effect': 'readable_policy_input' if next_enabled else 'none',
            'accepted_learning_count': len([item for item in accepted_learnings if isinstance(item, Mapping)]),
            'accepted_learnings': [
                _canonical_accepted_learning_record(item)
                for item in accepted_learnings
                if isinstance(item, Mapping)
            ],
            'last_activation_change': {
                'reviewer': _clean_text(reviewer) or 'operator',
                'reason': _clean_text(reason),
                'changed_at': _clean_text(changed_at) or _now_iso_utc(),
                'enabled': next_enabled,
            },
        }
    )
    return _json_safe(base)


def build_accepted_learning_runtime_hints(
    snapshot: Mapping[str, Any] | None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        snapshot = build_default_accepted_learning_policy_snapshot()
    enabled = snapshot.get('enabled') is True
    authority = _normalize_accepted_learning_authority(snapshot.get('authority'))
    raw_learnings = snapshot.get('accepted_learnings') if isinstance(snapshot.get('accepted_learnings'), list) else []
    hints: list[dict[str, Any]] = []
    if enabled:
        explicit_limit = max(1, int(limit)) if limit is not None else None
        for item in raw_learnings:
            if not isinstance(item, Mapping):
                continue
            target_area = _clean_text(item.get('target_area'))
            if target_area not in ALLOWED_ACCEPTED_LEARNING_TARGET_AREAS:
                continue
            hint = _runtime_hint_from_accepted_learning(item)
            if not hint:
                continue
            hints.append(
                {
                    'kind': 'ollmo.accepted_learning_runtime_hint',
                    'learning_id': _clean_text(item.get('learning_id')),
                    'candidate_id': _clean_text(item.get('candidate_id')),
                    'target_area': target_area,
                    'hint': hint,
                    'case_kinds': _canonical_case_kinds(item.get('case_kinds')),
                    'severity_counts': item.get('severity_counts') if isinstance(item.get('severity_counts'), Mapping) else {},
                    'authority': authority,
                    'allowed_use': (
                        'soft_hint_only'
                        if authority == 'soft_hint'
                        else f'{authority}_accepted_learning_hint'
                    ),
                    'forbidden_use': 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth',
                    'conflict_boundary': 'Ignored whenever current user intent, live capability evidence, Graph, IR, Closure Review, output obligations, artifact evidence, or runtime truth conflict.',
                    'evidence_case_ids': item.get('evidence_case_ids') if isinstance(item.get('evidence_case_ids'), list) else [],
                }
            )
            if explicit_limit is not None and len(hints) >= explicit_limit:
                break
    return _json_safe(
        {
            'kind': 'ollmo.accepted_learning_runtime_hints',
            'enabled': enabled,
            'authority': authority,
            'status': 'active' if enabled and hints else 'disabled' if not enabled else 'empty',
            'runtime_effect': (
                'soft_hints_available'
                if enabled and hints and authority == 'soft_hint'
                else f'{authority}_hints_available'
                    if enabled and hints
                    else 'none'
            ),
            'hint_count': len(hints),
            'hints': hints,
            'source_snapshot_status': _clean_text(snapshot.get('status')) or None,
        }
    )


def build_default_accepted_learning_policy_snapshot(
    *,
    snapshot_path: Path | str | None = None,
) -> dict[str, Any]:
    target = (
        Path(snapshot_path)
        if snapshot_path
        else DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT
    )
    return {
        'kind': 'ollmo.accepted_learning_policy_snapshot',
        'snapshot_version': SELF_LEARNING_VERSION,
        'status': 'not_configured',
        'enabled': False,
        'authority': DEFAULT_ACCEPTED_LEARNING_AUTHORITY,
        'activation_policy': 'disabled_until_explicit_review',
        'optimization_policy': 'accepted_learnings_require_reviewed_patch_or_operator_enable',
        'path': str(target),
        'accepted_learning_count': 0,
        'accepted_learnings': [],
        'runtime_effect': 'none',
        'notes': [
            'This snapshot is a future bridge from reviewed eval cases into runtime policy.',
            'It is read-only and non-operative while enabled is false.',
        ],
    }


def load_accepted_learning_policy_snapshot(
    *,
    snapshot_path: Path | str | None = None,
) -> dict[str, Any]:
    target = (
        Path(snapshot_path)
        if snapshot_path
        else DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT
    )
    if not target.exists():
        return build_default_accepted_learning_policy_snapshot(snapshot_path=target)
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        snapshot = build_default_accepted_learning_policy_snapshot(snapshot_path=target)
        snapshot['status'] = 'invalid'
        snapshot['error'] = str(exc)
        return snapshot
    if not isinstance(payload, Mapping):
        snapshot = build_default_accepted_learning_policy_snapshot(snapshot_path=target)
        snapshot['status'] = 'invalid'
        snapshot['error'] = 'snapshot root must be a JSON object'
        return snapshot

    accepted_learnings = payload.get('accepted_learnings') if isinstance(payload.get('accepted_learnings'), list) else []
    enabled = payload.get('enabled') is True
    snapshot = {
        **build_default_accepted_learning_policy_snapshot(snapshot_path=target),
        **dict(payload),
        'kind': 'ollmo.accepted_learning_policy_snapshot',
        'snapshot_version': int(payload.get('snapshot_version') or SELF_LEARNING_VERSION),
        'enabled': enabled,
        'authority': _normalize_accepted_learning_authority(payload.get('authority')),
        'status': _clean_text(payload.get('status')) or ('enabled' if enabled else 'disabled'),
        'path': str(target),
        'accepted_learning_count': len([item for item in accepted_learnings if isinstance(item, Mapping)]),
        'accepted_learnings': [
            _canonical_accepted_learning_record(item)
            for item in accepted_learnings
            if isinstance(item, Mapping)
        ],
        'runtime_effect': 'readable_policy_input' if enabled else 'none',
    }
    if not enabled:
        snapshot['activation_policy'] = 'disabled_until_explicit_review'
        snapshot['runtime_effect'] = 'none'
    return _json_safe(snapshot)


def persist_accepted_learning_policy_snapshot(
    snapshot: Mapping[str, Any] | None = None,
    *,
    output_path: Path | str | None = None,
) -> Path:
    target = Path(output_path) if output_path else DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT
    payload = (
        dict(snapshot)
        if isinstance(snapshot, Mapping)
        else build_default_accepted_learning_policy_snapshot(snapshot_path=target)
    )
    payload['path'] = str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return target


def build_self_learning_report(
    *,
    response_frame_ledger_path: Path | str | None = None,
    additional_response_frame_ledger_paths: Iterable[Path | str] | None = None,
    monitor_report_path: Path | str | None = None,
    graph_rebase_corpus_dir: Path | str | None = None,
    self_learning_dir: Path | str | None = None,
    existing_eval_case_ledger_path: Path | str | None = None,
    merge_existing: bool = False,
    retention_manifest_path: Path | str | None = None,
    frame_limit: int = 200,
    max_cases: int = 80,
    include_cases: bool = True,
    include_shadow_hints: bool = True,
) -> dict[str, Any]:
    """Build a compact GEPA-like offline optimization report from frozen frames."""

    primary_ledger = (
        Path(response_frame_ledger_path)
        if response_frame_ledger_path
        else DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER
    )
    ledgers: list[Path] = []
    seen_ledger_paths: set[str] = set()
    for candidate in [primary_ledger, *(additional_response_frame_ledger_paths or [])]:
        ledger_path = Path(candidate)
        ledger_identity = str(ledger_path.resolve(strict=False))
        if ledger_identity in seen_ledger_paths:
            continue
        seen_ledger_paths.add(ledger_identity)
        ledgers.append(ledger_path)
    monitor_path = Path(monitor_report_path) if monitor_report_path else None
    corpus_dir = Path(graph_rebase_corpus_dir) if graph_rebase_corpus_dir else None
    learning_dir = Path(self_learning_dir) if self_learning_dir else DEFAULT_SELF_LEARNING_DIR
    existing_eval_path = (
        Path(existing_eval_case_ledger_path)
        if existing_eval_case_ledger_path is not None
        else learning_dir / DEFAULT_EVAL_CASE_LEDGER
    )
    retention_path = Path(retention_manifest_path) if retention_manifest_path else learning_dir / DEFAULT_RETENTION_MANIFEST.name
    retention_manifest = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=primary_ledger.parent,
        retained_sidecars_dir=learning_dir / 'retained_sidecars',
    )
    corpus_coverage, corpus_bindings = _load_graph_rebase_corpus(corpus_dir)
    case_limit = max(1, int(max_cases))
    frame_count = 0
    superseded_frame_count = 0
    recent_frames: dict[str, tuple[dict[str, Any], int, int, Path]] = {}
    for ledger_rank, ledger in enumerate(ledgers):
        for reverse_rank, frame in enumerate(_iter_jsonl(ledger, limit=frame_limit)):
            frame_count += 1
            response_id = _frame_response_id(frame)
            if response_id in recent_frames:
                superseded_frame_count += 1
                continue
            recent_frames[response_id] = (frame, ledger_rank, reverse_rank, ledger.parent)

    target_response_ids = {
        _clean_text(binding.get('response_id'))
        for binding in corpus_bindings
        if binding.get('binding_status') == 'pending_ledger_binding'
        and _clean_text(binding.get('response_id'))
    }
    target_frames: dict[str, tuple[dict[str, Any], int, int, Path]] = {}
    if target_response_ids:
        for ledger_rank, ledger in enumerate(ledgers):
            for reverse_rank, frame in enumerate(_iter_jsonl(ledger, limit=2**63 - 1)):
                response_id = _frame_response_id(frame)
                if response_id in target_response_ids and response_id not in target_frames:
                    target_frames[response_id] = (frame, ledger_rank, reverse_rank, ledger.parent)
                    if len(target_frames) >= len(target_response_ids):
                        break
            if len(target_frames) >= len(target_response_ids):
                break

    exact_bindings_by_response: dict[str, list[dict[str, Any]]] = {}
    binding_status_counts: Counter[str] = Counter()
    binding_records: list[dict[str, Any]] = []
    declared_last_frame_mismatch_count = 0
    declared_response_id_mismatch_count = 0
    for binding in corpus_bindings:
        binding_status = _clean_text(binding.get('binding_status'))
        if binding_status == 'pending_ledger_binding':
            response_id = _clean_text(binding.get('response_id'))
            target = target_frames.get(response_id)
            if target is None:
                binding_status = 'missing_response'
            else:
                frame, _ledger_rank, _reverse_rank, _snapshot_root = target
                actual_frame_id = _clean_text(frame.get('frame_id'))
                actual_frame_sequence = _integer_or_none(frame.get('frame_sequence'))
                if (
                    actual_frame_id == _clean_text(binding.get('frame_id'))
                    and actual_frame_sequence == _integer_or_none(binding.get('frame_sequence'))
                ):
                    binding_status = 'exact_bound'
                    exact_bindings_by_response.setdefault(response_id, []).append(binding)
                else:
                    binding_status = 'stale_frame'
                    binding['actual_latest_frame_id'] = actual_frame_id
                    binding['actual_latest_frame_sequence'] = actual_frame_sequence
        binding['binding_status'] = binding_status
        binding_status_counts[binding_status or 'malformed_case'] += 1
        declared_last_frame_id = _clean_text(binding.get('declared_last_frame_id'))
        declared_last_frame_sequence = _integer_or_none(binding.get('declared_last_frame_sequence'))
        if declared_last_frame_id and (
            declared_last_frame_id != _clean_text(binding.get('frame_id'))
            or declared_last_frame_sequence != _integer_or_none(binding.get('frame_sequence'))
        ):
            declared_last_frame_mismatch_count += 1
        declared_response_id = _clean_text(binding.get('declared_response_id'))
        bound_response_id = _clean_text(binding.get('response_id'))
        if (
            declared_response_id
            and bound_response_id
            and declared_response_id != bound_response_id
        ):
            declared_response_id_mismatch_count += 1
        record = _corpus_case_provenance(binding)
        record['binding_status'] = binding_status or 'malformed_case'
        for key in ('actual_latest_frame_id', 'actual_latest_frame_sequence', 'binding_error'):
            if binding.get(key) not in (None, ''):
                record[key] = binding.get(key)
        binding_records.append(record)

    corpus_coverage['binding_status_counts'] = dict(sorted(binding_status_counts.items()))
    corpus_coverage['exact_bound_case_count'] = binding_status_counts.get('exact_bound', 0)
    corpus_coverage['missing_response_case_count'] = binding_status_counts.get('missing_response', 0)
    corpus_coverage['stale_frame_case_count'] = binding_status_counts.get('stale_frame', 0)
    corpus_coverage['planned_case_count'] = binding_status_counts.get('planned', 0)
    corpus_coverage['dependency_blocked_case_count'] = binding_status_counts.get('dependency_blocked', 0)
    corpus_coverage['malformed_binding_case_count'] = binding_status_counts.get('malformed_binding', 0)
    corpus_coverage['declared_last_frame_mismatch_count'] = declared_last_frame_mismatch_count
    corpus_coverage['declared_response_id_mismatch_count'] = declared_response_id_mismatch_count
    corpus_coverage['bindings'] = binding_records
    if corpus_coverage.get('status') == 'completed' and (
        binding_status_counts.get('missing_response', 0)
        or binding_status_counts.get('stale_frame', 0)
        or binding_status_counts.get('malformed_binding', 0)
        or binding_status_counts.get('malformed_case', 0)
    ):
        corpus_coverage['status'] = 'partial'

    selected_frames = dict(recent_frames)
    for response_id in exact_bindings_by_response:
        target = target_frames.get(response_id)
        if target is not None:
            selected_frames[response_id] = target

    raw_cases: list[dict[str, Any]] = []
    corpus_raw_case_count = 0
    evaluated_response_ids: list[str] = []
    for response_id, (frame, ledger_rank, reverse_rank, snapshot_root) in sorted(
        selected_frames.items(),
        key=lambda item: (item[1][1], item[1][2], item[0]),
    ):
        evaluated_response_ids.append(response_id)
        hydrated_frame = _hydrate_frame_for_learning(frame, snapshot_root=snapshot_root)
        corpus_links = exact_bindings_by_response.get(response_id, [])
        if corpus_links:
            hydrated_frame = _overlay_corpus_eval_evidence(hydrated_frame, corpus_links)
        frame_cases = build_eval_cases_from_response_frame(hydrated_frame)
        if corpus_links:
            frame_cases = _attach_corpus_provenance(frame_cases, corpus_links)
            corpus_raw_case_count += len(frame_cases)
        raw_cases.extend(frame_cases)

    cases = _dedupe_cases(raw_cases)
    corpus_unique_case_count = len(
        [
            case
            for case in cases
            if isinstance(case.get('metadata'), Mapping)
            and isinstance(case['metadata'].get('graph_rebase_corpus'), Mapping)
        ]
    )
    if monitor_path is not None:
        monitor_reports = _read_jsonl(monitor_path, limit=frame_limit)
        cases = enrich_eval_cases_from_monitor_reports(cases, monitor_reports)
    unique_case_count_before_cap = len(cases)
    new_cases = cases[:case_limit]
    corpus_selected_case_count = len(
        [
            case
            for case in new_cases
            if isinstance(case.get('metadata'), Mapping)
            and isinstance(case['metadata'].get('graph_rebase_corpus'), Mapping)
        ]
    )
    corpus_coverage['corpus_linked_response_count'] = len(exact_bindings_by_response)
    corpus_coverage['raw_eval_case_count'] = corpus_raw_case_count
    corpus_coverage['unique_eval_case_count_before_cap'] = corpus_unique_case_count
    corpus_coverage['selected_eval_case_count'] = corpus_selected_case_count
    corpus_coverage['eval_case_truncated_count'] = max(
        0,
        corpus_unique_case_count - corpus_selected_case_count,
    )
    merge_counts = {
        'previous_case_count': 0,
        'new_case_count': len(new_cases),
        'preserved_case_count': 0,
        'replaced_case_count': 0,
        'removed_case_count': 0,
    }
    if merge_existing:
        previous_cases = load_eval_cases(existing_eval_path)
        cases, merge_counts = merge_eval_cases_by_case_id(previous_cases, new_cases)
    else:
        cases = new_cases
    merge_policy = _eval_case_merge_policy(merge_existing=merge_existing)
    improvement_candidates = build_policy_improvement_candidates(cases)
    report: dict[str, Any] = {
        'kind': 'ollmo.self_learning_report',
        'self_learning_version': SELF_LEARNING_VERSION,
        'status': 'completed' if frame_count else 'no_frames',
        'mode': 'offline_eval_cases',
        'optimization_policy': 'proposal_only_reviewed_patch_required',
        'source': {
            'response_frame_ledger': str(primary_ledger),
            'response_frame_ledgers': [str(ledger) for ledger in ledgers],
            'monitor_report_path': str(monitor_path) if monitor_path is not None else None,
            'graph_rebase_corpus_dir': str(corpus_dir) if corpus_dir is not None else None,
            'self_learning_dir': str(learning_dir),
            'existing_eval_case_ledger_path': str(existing_eval_path) if merge_existing else None,
            'frame_limit': frame_limit,
            'frame_limit_scope': 'per_response_frame_ledger',
            'max_cases': max_cases,
            'frame_selection': (
                'first_ledger_wins_latest_frame_per_response_recent_windows_union_exact_graph_rebase_corpus'
                if corpus_dir is not None
                else 'first_ledger_wins_latest_frame_per_response_within_recent_windows'
            ),
            'monitor_evidence_policy': 'supporting_only_response_frames_remain_canonical' if monitor_path is not None else None,
            'existing_eval_case_policy': (
                'historical_eval_evidence_only_not_current_runtime_truth'
                if merge_existing
                else None
            ),
        },
        'retention': retention_summary(retention_manifest, manifest_path=retention_path),
        'frame_count': frame_count,
        'recent_evaluated_response_count': len(recent_frames),
        'evaluated_response_count': len(evaluated_response_ids),
        'superseded_frame_count': superseded_frame_count,
        'case_count': len(cases),
        'unique_case_count_before_cap': unique_case_count_before_cap,
        'case_truncated_count': max(0, unique_case_count_before_cap - len(new_cases)),
        **merge_counts,
        'merge_policy': merge_policy,
        'graph_rebase_corpus': corpus_coverage,
        'counts_by_layer': _count_by(cases, 'layer'),
        'counts_by_kind': _count_by(cases, 'case_kind'),
        'counts_by_severity': _count_by(cases, 'severity'),
        'improvement_candidate_count': len(improvement_candidates),
        'improvement_candidates': improvement_candidates,
    }
    if include_shadow_hints:
        shadow_hints = build_shadow_learning_hints(report)
        report['shadow_hints'] = shadow_hints
        report['shadow_hint_count'] = len(shadow_hints)
        report['shadow_hint_policy'] = {
            'authority': 'shadow',
            'runtime_effect': 'none',
            'optimization_policy': 'proposal_only_reviewed_patch_required',
        }
    if include_cases:
        report['eval_cases'] = cases
    if not include_cases:
        return _json_safe(report)
    report_without_cases = dict(report)
    report_without_cases.pop('eval_cases', None)
    sanitized_report = _json_safe(report_without_cases)
    sanitized_report['eval_cases'] = [
        _json_safe_preserving_empty(case)
        for case in cases
    ]
    return sanitized_report


def _eval_cases_jsonl_bytes(cases: Iterable[Mapping[str, Any]]) -> bytes:
    lines: list[str] = []
    for ordinal, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f'eval case {ordinal} must be a mapping')
        lines.append(
            json.dumps(
                _json_safe_preserving_empty(dict(case)),
                ensure_ascii=False,
                sort_keys=True,
            )
            + '\n'
        )
    return ''.join(lines).encode('utf-8')


def _self_learning_report_json_bytes(report: Mapping[str, Any]) -> bytes:
    if not isinstance(report, Mapping):
        raise ValueError('self-learning report must be a mapping')
    return (
        json.dumps(
            _json_safe_preserving_empty(dict(report)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + '\n'
    ).encode('utf-8')


def _stage_atomic_file_bytes(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor: int | None = None
    temp_path: Path | None = None
    for _attempt in range(128):
        candidate = target.parent / f'.{target.name}.{secrets.token_hex(8)}.tmp'
        try:
            file_descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            continue
        temp_path = candidate
        break
    if file_descriptor is None or temp_path is None:
        raise FileExistsError(f'could not allocate atomic staging file for {target}')
    try:
        try:
            existing_mode = int(target.stat().st_mode) & 0o777
        except OSError:
            existing_mode = None
        if existing_mode is not None:
            os.fchmod(file_descriptor, existing_mode)
        with os.fdopen(file_descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _stage_atomic_rollback_copy(target: Path) -> Path | None:
    if not target.exists():
        return None
    file_descriptor, backup_name = tempfile.mkstemp(
        prefix=f'.{target.name}.',
        suffix='.rollback',
        dir=str(target.parent),
    )
    os.close(file_descriptor)
    backup_path = Path(backup_name)
    try:
        backup_path.unlink()
        try:
            os.link(target, backup_path)
        except OSError:
            shutil.copy2(target, backup_path)
            with backup_path.open('rb') as handle:
                os.fsync(handle.fileno())
        return backup_path
    except BaseException:
        try:
            backup_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        unsupported_errors = {
            errno.EINVAL,
            getattr(errno, 'ENOTSUP', errno.EINVAL),
            getattr(errno, 'EOPNOTSUPP', errno.EINVAL),
        }
        if exc.errno not in unsupported_errors:
            raise
        # Some filesystems do not support directory fsync. Same-directory
        # replacement and file fsync still prevent a partial visible body.


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_default_self_learning_output_targets(targets: Iterable[Path]) -> None:
    """Reject the repository's immutable runtime/learning input locations."""

    protected_files = {
        (DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER).resolve(strict=False),
        (DEFAULT_RESPONSE_FRAMES_DIR / 'current_index.json').resolve(strict=False),
        (DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT).resolve(strict=False),
        DEFAULT_RETENTION_MANIFEST.resolve(strict=False),
        Path('state/ollmo_run_monitor/reports.jsonl').resolve(strict=False),
        (
            _REPOSITORY_ROOT / DEFAULT_RESPONSE_FRAMES_DIR / DEFAULT_RESPONSE_FRAME_LEDGER
        ).resolve(strict=False),
        (_REPOSITORY_ROOT / DEFAULT_RESPONSE_FRAMES_DIR / 'current_index.json').resolve(
            strict=False
        ),
        (
            _REPOSITORY_ROOT / DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT
        ).resolve(strict=False),
        (_REPOSITORY_ROOT / DEFAULT_RETENTION_MANIFEST).resolve(strict=False),
        (_REPOSITORY_ROOT / 'state/ollmo_run_monitor/reports.jsonl').resolve(strict=False),
    }
    protected_roots = {
        (DEFAULT_RESPONSE_FRAMES_DIR / 'snapshots').resolve(strict=False),
        (DEFAULT_SELF_LEARNING_DIR / 'retained_sidecars').resolve(strict=False),
        Path('state/graph_rebase_shadow_corpus').resolve(strict=False),
        (_REPOSITORY_ROOT / DEFAULT_RESPONSE_FRAMES_DIR / 'snapshots').resolve(strict=False),
        (_REPOSITORY_ROOT / DEFAULT_SELF_LEARNING_DIR / 'retained_sidecars').resolve(
            strict=False
        ),
        (_REPOSITORY_ROOT / 'state/graph_rebase_shadow_corpus').resolve(strict=False),
    }
    for target in targets:
        resolved_target = Path(target).resolve(strict=False)
        if resolved_target in protected_files:
            raise ValueError(
                f'self-learning output target resolves to protected state: {resolved_target}'
            )
        for protected_root in protected_roots:
            if _path_is_within(resolved_target, protected_root):
                raise ValueError(
                    'self-learning output target resolves inside protected state: '
                    f'{resolved_target}'
                )


def _validate_output_path_has_no_symlinks(target: Path) -> None:
    """Refuse link-mediated output paths that cannot be rollback-safe by pathname."""

    absolute_target = Path(os.path.abspath(os.fspath(target)))
    current = Path(absolute_target.anchor)
    for component in absolute_target.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f'self-learning output paths cannot contain symlinks: {current}'
            )


@contextmanager
def self_learning_output_update_lock(
    output_paths: Iterable[Path | str],
) -> Iterator[None]:
    """Serialize cooperative builders that may read, merge, and replace outputs."""

    absolute_targets = sorted(
        {
            Path(os.path.abspath(os.fspath(Path(path))))
            for path in output_paths
        },
        key=str,
    )
    lock_descriptors: list[int] = []
    try:
        for target in absolute_targets:
            _validate_output_path_has_no_symlinks(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            lock_path = target.parent / f'.{target.name}.self_learning_update.lock'
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except BaseException:
                os.close(descriptor)
                raise
            lock_descriptors.append(descriptor)
        yield
    finally:
        for descriptor in reversed(lock_descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _atomic_replace_file_set(payloads: Iterable[tuple[Path, bytes]]) -> None:
    """Stage every payload, then install complete files with rollback on errors.

    Each target is a whole-file atomic replacement. The targets do not form a
    portable cross-file snapshot transaction, so callers must serialize writers;
    a hard stop between replacements can leave complete files from adjacent runs.
    """

    entries = [(Path(target), payload) for target, payload in payloads]
    identities = [str(target.resolve(strict=False)) for target, _payload in entries]
    if len(identities) != len(set(identities)):
        raise ValueError('atomic self-learning outputs must use distinct target paths')
    for target, _payload in entries:
        _validate_output_path_has_no_symlinks(target)

    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    installed: list[Path] = []
    preserve_recovery_files = False
    try:
        for target, payload in entries:
            staged.append((target, _stage_atomic_file_bytes(target, payload)))
        for target, _temp_path in staged:
            backups[target] = _stage_atomic_rollback_copy(target)

        try:
            for target, temp_path in staged:
                _validate_output_path_has_no_symlinks(target)
                os.replace(temp_path, target)
                installed.append(target)
                _fsync_directory(target.parent)
        except BaseException as install_error:
            rollback_errors: list[str] = []
            for target in reversed(installed):
                backup_path = backups.get(target)
                try:
                    if backup_path is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup_path, target)
                    _fsync_directory(target.parent)
                except BaseException as rollback_error:
                    rollback_errors.append(f'{target}: {rollback_error}')
            if rollback_errors:
                preserve_recovery_files = True
                recovery_paths = [
                    str(path)
                    for path in [
                        *(temp_path for _target, temp_path in staged),
                        *(backup for backup in backups.values() if backup is not None),
                    ]
                    if path.exists()
                ]
                raise RuntimeError(
                    'self-learning output installation failed and rollback was incomplete: '
                    + '; '.join(rollback_errors)
                    + (
                        '; recovery files retained at: ' + ', '.join(recovery_paths)
                        if recovery_paths
                        else ''
                    )
                ) from install_error
            raise
    finally:
        if not preserve_recovery_files:
            for _target, temp_path in staged:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
            for backup_path in backups.values():
                if backup_path is None:
                    continue
                try:
                    backup_path.unlink()
                except FileNotFoundError:
                    pass


def persist_eval_cases(
    cases: Iterable[Mapping[str, Any]],
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Atomically persist eval cases as JSONL and return the written path."""

    target = Path(output_path) if output_path else DEFAULT_SELF_LEARNING_DIR / DEFAULT_EVAL_CASE_LEDGER
    _validate_default_self_learning_output_targets([target])
    _atomic_replace_file_set([(target, _eval_cases_jsonl_bytes(cases))])
    return target


def persist_self_learning_report(
    report: Mapping[str, Any],
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Atomically persist a self-learning report and return the written path."""

    target = Path(output_path) if output_path else DEFAULT_SELF_LEARNING_DIR / DEFAULT_SELF_LEARNING_REPORT
    _validate_default_self_learning_output_targets([target])
    _atomic_replace_file_set([(target, _self_learning_report_json_bytes(report))])
    return target


def persist_self_learning_outputs(
    cases: Iterable[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    eval_case_output_path: Path | str | None = None,
    report_output_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Stage and atomically install the eval ledger and its derived report."""

    eval_target = (
        Path(eval_case_output_path)
        if eval_case_output_path
        else DEFAULT_SELF_LEARNING_DIR / DEFAULT_EVAL_CASE_LEDGER
    )
    report_target = (
        Path(report_output_path)
        if report_output_path
        else DEFAULT_SELF_LEARNING_DIR / DEFAULT_SELF_LEARNING_REPORT
    )
    _validate_default_self_learning_output_targets([eval_target, report_target])
    case_payload = _eval_cases_jsonl_bytes(cases)
    report_payload = _self_learning_report_json_bytes(report)
    _atomic_replace_file_set(
        [
            (eval_target, case_payload),
            (report_target, report_payload),
        ]
    )
    return eval_target, report_target
