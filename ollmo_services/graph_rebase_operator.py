"""Trusted, append-only operator review records for graph rebase proposals.

This module deliberately owns no web route and performs no graph mutation.  Its
input is a canonical response payload that a control-plane owner has already
hydrated from a frozen response frame.  Proposal, graph, frame, and digest
bindings are derived from that payload; caller-supplied identities are only
compare-and-swap expectations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Optional

from ollmo_services.graph_rebase import (
    GRAPH_REBASE_LIFECYCLE_KIND,
    GRAPH_REBASE_PROPOSAL_KIND,
    GRAPH_REBASE_REVIEW_KIND,
    parse_graph_rebase_frame_sequence,
    stable_graph_digest,
    stable_graph_rebase_prompt_digest,
    validate_graph_rebase_proposal,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Ollmo's production host is POSIX.
    fcntl = None


GRAPH_REBASE_OPERATOR_RECORD_KIND = 'ollmo.graph_rebase_operator_record'
GRAPH_REBASE_AUTHORIZATION_KIND = 'ollmo.graph_rebase_authorization'
GRAPH_REBASE_PROMOTION_GATE_KIND = 'ollmo.graph_rebase_promotion_gate'
GRAPH_REBASE_OPERATOR_REGISTRY_VERSION = 1
DEFAULT_GRAPH_REBASE_OPERATOR_REGISTRY_PATH = Path(
    'state/graph_rebase/operator_reviews.jsonl'
)

_ACTIONS = {
    'adjudicate',
    'stage',
    'authorize_partial',
}
_ADJUDICATIONS = {
    'accepted',
    'false_negative',
    'false_positive',
    'needs_investigation',
    'rejected_authorization',
    'useful_proposal',
}
_ACCEPTED_ADJUDICATIONS = {
    'accepted',
    'useful_proposal',
}
_REBASE_CLASSES = {
    'partial_subtree_rebase',
    'full_successor_rebase',
}
_NO_FORMAL_PROPOSAL_SENTINEL = 'no_formal_proposal'
_TERMINAL_LIFECYCLE_STATES = {
    'blocked',
    'cancelled',
    'completed',
    'failed',
    'late_fill_completed',
    'repair_needed',
}
_ACTIVE_LATE_FILL_STATES = {
    'active',
    'late_fill_pending',
    'late_fill_running',
    'pending',
    'queued',
    'running',
    'scheduled',
}
_WILDCARD_IDENTITY_TOKENS = {
    '*',
    'all',
    'any',
    'current',
    'latest',
}

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class GraphRebaseOperatorRegistryError(ValueError):
    """Fail-closed registry error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        *,
        message: str = '',
        status_code: int = 400,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.code = str(code or 'graph_rebase_operator_registry_error').strip()
        self.status_code = int(status_code)
        self.details = _json_safe(details or {})
        super().__init__(message or self.code)


def _fail(
    code: str,
    *,
    status_code: int = 400,
    message: str = '',
    **details: Any,
) -> None:
    raise GraphRebaseOperatorRegistryError(
        code,
        message=message,
        status_code=status_code,
        details=details,
    )


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _token(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_').replace(' ', '_')


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _clean_text(key): _json_safe(raw_value)
            for key, raw_value in value.items()
            if _clean_text(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )


def _sha256(value: Any, *, prefix: str = '') -> str:
    digest = hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()
    return f'{prefix}{digest}' if prefix else digest


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        return []
    result: list[str] = []
    for item in raw_items:
        text = _clean_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _contains_wildcard(value: Any) -> bool:
    text = _clean_text(value)
    return (
        not text
        or text.lower() in _WILDCARD_IDENTITY_TOKENS
        or '*' in text
    )


def _require_exact_identity(name: str, value: Any) -> str:
    text = _clean_text(value)
    if not text:
        _fail(f'{name}_required')
    if _contains_wildcard(text):
        _fail(f'{name}_wildcard_forbidden')
    return text


def _qualified_evidence_refs(value: Any, *, field: str = 'evidence_refs') -> list[str]:
    refs = _clean_string_list(value)
    if not refs:
        _fail(f'{field}_required')
    for ref in refs:
        if _contains_wildcard(ref):
            _fail(f'{field}_wildcard_forbidden', evidence_ref=ref)
        if ':' not in ref:
            _fail(f'{field}_must_be_qualified', evidence_ref=ref)
    return refs


def _thread_lock_for(path: Path) -> threading.RLock:
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path.absolute())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


@contextmanager
def _registry_lock(path: Path, *, create: bool = True) -> Iterator[None]:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _thread_lock_for(path)
    lock_path = path.with_name(f'{path.name}.lock')
    with thread_lock:
        if not create and not lock_path.exists():
            yield
            return
        mode = 'a+' if create else 'r+'
        with lock_path.open(mode, encoding='utf-8') as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _semantic_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(_json_safe(record))
    for key in (
        'authorization',
        'record_digest',
        'record_id',
        'recorded_at',
    ):
        payload.pop(key, None)
    return payload


def stable_graph_rebase_operator_record_id(record: Mapping[str, Any]) -> str:
    """Return the independently reproducible content identity for a record."""

    if not isinstance(record, Mapping):
        return ''
    return _sha256(
        _semantic_record_payload(record),
        prefix='graph-rebase-operator-',
    )


def _record_digest_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(_json_safe(record))
    payload.pop('record_digest', None)
    return payload


def _stable_record_digest(record: Mapping[str, Any]) -> str:
    return _sha256(
        _record_digest_payload(record),
        prefix='graph-rebase-operator-record-sha256:',
    )


def _authorization_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    gate = record.get('promotion_gate') if isinstance(record.get('promotion_gate'), Mapping) else {}
    evidence_refs = _clean_string_list(record.get('evidence_refs'))
    for ref in _clean_string_list(gate.get('evidence_refs')):
        if ref not in evidence_refs:
            evidence_refs.append(ref)
    return {
        'kind': GRAPH_REBASE_AUTHORIZATION_KIND,
        'status': 'accepted',
        'authority': 'operator_review',
        'source': 'runtime_operator_registry',
        'provenance': 'runtime_operator_registry',
        'registry_record_id': _clean_text(record.get('record_id')),
        'review_record_id': _clean_text(record.get('review_record_id')),
        'stage_record_id': _clean_text(record.get('stage_record_id')),
        'allowed_autonomy': ['apply_reviewed'],
        'response_id': _clean_text(record.get('response_id')),
        'frame_id': _clean_text(record.get('frame_id')),
        'proposal_id': _clean_text(record.get('proposal_id')),
        'base_graph_digest': _clean_text(record.get('base_graph_digest')),
        'candidate_graph_digest': _clean_text(record.get('candidate_graph_digest')),
        'requested_rebase_class': 'partial_subtree_rebase',
        'evidence_refs': evidence_refs,
        'reason': _clean_text(record.get('reason')),
    }


def _record_evidence_is_qualified(value: Any) -> bool:
    refs = _clean_string_list(value)
    return bool(refs) and all(
        not _contains_wildcard(ref) and ':' in ref
        for ref in refs
    )


def verify_graph_rebase_operator_record(record: Mapping[str, Any]) -> bool:
    """Verify semantic identity, full-line digest, and authorization projection."""

    if not isinstance(record, Mapping):
        return False
    if record.get('kind') != GRAPH_REBASE_OPERATOR_RECORD_KIND:
        return False
    if record.get('registry_version') != GRAPH_REBASE_OPERATOR_REGISTRY_VERSION:
        return False
    action = _token(record.get('action'))
    adjudication = _token(record.get('adjudication'))
    if action not in _ACTIONS or adjudication not in _ADJUDICATIONS:
        return False
    if action in {'stage', 'authorize_partial'} and adjudication not in _ACCEPTED_ADJUDICATIONS:
        return False
    if _clean_text(record.get('authority')) != 'runtime_operator_registry':
        return False
    if _clean_text(record.get('source')) != 'runtime_operator_registry':
        return False
    if _clean_text(record.get('provenance')) != 'runtime_operator_registry':
        return False
    if _contains_wildcard(record.get('operator_identity')):
        return False
    if _clean_text(record.get('operator_authentication')) != (
        'explicit_control_plane_credential'
    ):
        return False
    expected_status = {
        'adjudicate': 'recorded',
        'stage': 'staged',
        'authorize_partial': 'accepted',
    }[action]
    expected_effect = {
        'adjudicate': 'none',
        'stage': 'staged_no_executable_mutation',
        'authorize_partial': 'authorization_only_no_execution',
    }[action]
    if _token(record.get('status')) != expected_status:
        return False
    if _token(record.get('runtime_effect')) != expected_effect:
        return False
    if not _clean_text(record.get('reason')) or _contains_wildcard(record.get('reason')):
        return False
    if not _record_evidence_is_qualified(record.get('evidence_refs')):
        return False
    for key in (
        'response_id',
        'frame_id',
        'target_frame_id',
        'base_graph_digest',
        'candidate_graph_digest',
        'requested_rebase_class',
    ):
        if _contains_wildcard(record.get(key)):
            return False
    if _token(record.get('requested_rebase_class')) not in _REBASE_CLASSES:
        return False
    if action == 'adjudicate' and adjudication == 'false_negative':
        if _contains_wildcard(record.get('candidate_observation_id')):
            return False
        if any(
            record.get(key) not in (None, '', [], {})
            for key in ('proposal_id', 'proposal_digest', 'runtime_review_id')
        ):
            return False
        if record.get('replay_verification') not in (None, '', [], {}):
            return False
    else:
        for key in ('proposal_id', 'proposal_digest', 'runtime_review_id'):
            if _contains_wildcard(record.get(key)):
                return False
        if action == 'adjudicate':
            replay = (
                record.get('replay_verification')
                if isinstance(record.get('replay_verification'), Mapping)
                else {}
            )
            replay_status = _token(record.get('replay_status'))
            replay_verified = record.get('replay_verified') is True
            if (
                replay.get('kind') != 'ollmo.graph_rebase_review_replay_verification'
                or _token(replay.get('status')) != replay_status
                or (replay.get('replay_verified') is True) != replay_verified
            ):
                return False
            if adjudication in _ACCEPTED_ADJUDICATIONS and not replay_verified:
                return False
    if action in {'stage', 'authorize_partial'} and _contains_wildcard(
        record.get('review_record_id')
    ):
        return False
    if action == 'authorize_partial' and _contains_wildcard(record.get('stage_record_id')):
        return False
    resolves_record_id = _clean_text(record.get('resolves_record_id'))
    resolution_fields = (
        'resolved_candidate_observation_id',
        'resolved_response_id',
    )
    if resolves_record_id:
        if _contains_wildcard(resolves_record_id):
            return False
        if action != 'adjudicate' or adjudication != 'useful_proposal':
            return False
        if record.get('replay_verified') is not True:
            return False
        if any(_contains_wildcard(record.get(key)) for key in resolution_fields):
            return False
    elif any(record.get(key) not in (None, '', [], {}) for key in resolution_fields):
        return False
    if _clean_text(record.get('record_id')) != stable_graph_rebase_operator_record_id(record):
        return False
    if _clean_text(record.get('record_digest')) != _stable_record_digest(record):
        return False
    if action == 'authorize_partial':
        if _token(record.get('requested_rebase_class')) != 'partial_subtree_rebase':
            return False
        try:
            gate = _normalize_promotion_gate(record.get('promotion_gate'))
        except GraphRebaseOperatorRegistryError:
            return False
        if _json_safe(gate) != _json_safe(record.get('promotion_gate')):
            return False
        authorization = (
            record.get('authorization')
            if isinstance(record.get('authorization'), Mapping)
            else {}
        )
        if _json_safe(authorization) != _json_safe(_authorization_from_record(record)):
            return False
    elif record.get('authorization') not in (None, '', [], {}):
        return False
    return True


def _validate_resolution_chain_record(
    record: Mapping[str, Any],
    *,
    prior_records_by_id: Mapping[str, Mapping[str, Any]],
    resolution_record_ids_by_false_negative: Mapping[str, str],
) -> None:
    """Validate one append-ordered false-negative resolution link."""

    false_negative_id = _clean_text(record.get('resolves_record_id'))
    if not false_negative_id:
        return
    false_negative = prior_records_by_id.get(false_negative_id)
    if not isinstance(false_negative, Mapping):
        _fail(
            'operator_registry_resolution_target_missing_or_not_prior',
            status_code=500,
            resolves_record_id=false_negative_id,
        )
    if (
        _token(false_negative.get('action')) != 'adjudicate'
        or _token(false_negative.get('adjudication')) != 'false_negative'
    ):
        _fail(
            'operator_registry_resolution_target_not_false_negative',
            status_code=500,
            resolves_record_id=false_negative_id,
        )
    if _clean_text(record.get('resolved_candidate_observation_id')) != _clean_text(
        false_negative.get('candidate_observation_id')
    ):
        _fail(
            'operator_registry_resolution_candidate_binding_mismatch',
            status_code=500,
            resolves_record_id=false_negative_id,
        )
    if _clean_text(record.get('resolved_response_id')) != _clean_text(
        false_negative.get('response_id')
    ):
        _fail(
            'operator_registry_resolution_response_binding_mismatch',
            status_code=500,
            resolves_record_id=false_negative_id,
        )
    if _token(record.get('requested_rebase_class')) != _token(
        false_negative.get('requested_rebase_class')
    ):
        _fail(
            'operator_registry_resolution_class_mismatch',
            status_code=500,
            resolves_record_id=false_negative_id,
        )
    prior_resolution_id = _clean_text(
        resolution_record_ids_by_false_negative.get(false_negative_id)
    )
    current_resolution_id = _clean_text(record.get('record_id'))
    if prior_resolution_id and prior_resolution_id != current_resolution_id:
        _fail(
            'operator_registry_false_negative_multiply_resolved',
            status_code=500,
            resolves_record_id=false_negative_id,
            prior_resolution_record_id=prior_resolution_id,
            current_resolution_record_id=current_resolution_id,
        )


def _read_registry_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw_lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        _fail(
            'operator_registry_unreadable',
            status_code=500,
            error=str(exc),
            registry_path=str(path),
        )
    records: list[dict[str, Any]] = []
    seen_ids: dict[str, dict[str, Any]] = {}
    resolution_record_ids_by_false_negative: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            parsed = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _fail(
                'operator_registry_corrupt_json',
                status_code=500,
                line_number=line_number,
                error=str(exc),
                registry_path=str(path),
            )
        if not isinstance(parsed, dict) or not verify_graph_rebase_operator_record(parsed):
            _fail(
                'operator_registry_record_verification_failed',
                status_code=500,
                line_number=line_number,
                registry_path=str(path),
            )
        record = _json_safe(parsed)
        record_id = _clean_text(record.get('record_id'))
        existing = seen_ids.get(record_id)
        if existing is not None and existing != record:
            _fail(
                'operator_registry_record_id_collision',
                status_code=500,
                record_id=record_id,
                registry_path=str(path),
            )
        if existing is None:
            _validate_resolution_chain_record(
                record,
                prior_records_by_id=seen_ids,
                resolution_record_ids_by_false_negative=(
                    resolution_record_ids_by_false_negative
                ),
            )
            seen_ids[record_id] = record
            records.append(record)
            false_negative_id = _clean_text(record.get('resolves_record_id'))
            if false_negative_id:
                resolution_record_ids_by_false_negative[false_negative_id] = record_id
    return records


def load_graph_rebase_operator_records(
    *,
    registry_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load only verified trusted registry records, failing closed on corruption."""

    path = (
        Path(registry_path)
        if registry_path is not None
        else DEFAULT_GRAPH_REBASE_OPERATOR_REGISTRY_PATH
    )
    if not path.exists():
        return []
    with _registry_lock(path, create=False):
        return copy.deepcopy(_read_registry_unlocked(path))


def _append_record_unlocked(path: Path, record: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True).encode('utf-8')
        + b'\n'
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('append returned no progress')
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _nonempty_inline_authorization(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _token(key) == 'graph_rebase_authorization' and child not in (None, '', [], {}):
                return True
            if _nonempty_inline_authorization(child):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_nonempty_inline_authorization(item) for item in value)
    return False


def _canonical_response_target(
    response_payload: Mapping[str, Any],
    *,
    expected_response_id: Any,
    expected_frame_id: Any,
    expected_frame_sequence: Any,
    expected_proposal_id: Any,
    expected_base_graph_digest: Any,
    expected_candidate_graph_digest: Any,
    expected_requested_rebase_class: Any,
) -> dict[str, Any]:
    if not isinstance(response_payload, Mapping) or not response_payload:
        _fail('canonical_response_payload_required')
    if _nonempty_inline_authorization(response_payload):
        _fail(
            'inline_graph_rebase_authorization_forbidden',
            status_code=403,
        )

    expected = {
        'response_id': _require_exact_identity('expected_response_id', expected_response_id),
        'frame_id': _require_exact_identity('expected_frame_id', expected_frame_id),
        'proposal_id': _require_exact_identity('expected_proposal_id', expected_proposal_id),
        'base_graph_digest': _require_exact_identity(
            'expected_base_graph_digest', expected_base_graph_digest
        ),
        'candidate_graph_digest': _require_exact_identity(
            'expected_candidate_graph_digest', expected_candidate_graph_digest
        ),
        'requested_rebase_class': _require_exact_identity(
            'expected_requested_rebase_class', expected_requested_rebase_class
        ),
    }

    frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), Mapping)
        else {}
    )
    if frame.get('kind') != 'ollmo.response_frame':
        _fail('canonical_frozen_response_frame_required')
    frame_id = _require_exact_identity('canonical_frame_id', frame.get('frame_id'))
    try:
        expected_sequence = parse_graph_rebase_frame_sequence(expected_frame_sequence)
    except ValueError:
        _fail('expected_frame_sequence_invalid')
    try:
        current_sequence = parse_graph_rebase_frame_sequence(frame.get('frame_sequence'))
    except ValueError:
        _fail('canonical_frame_sequence_invalid')
    if expected_sequence != current_sequence:
        _fail(
            'stale_frame_sequence',
            status_code=409,
            expected=expected_sequence,
            actual=current_sequence,
        )

    runtime = (
        response_payload.get('runtime')
        if isinstance(response_payload.get('runtime'), Mapping)
        else {}
    )
    graph = (
        runtime.get('request_phase_graph')
        if isinstance(runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    if not graph:
        _fail('canonical_request_phase_graph_required')

    response_candidates = _clean_string_list(
        [
            response_payload.get('response_id'),
            response_payload.get('id'),
            frame.get('response_id'),
            frame.get('id'),
            graph.get('response_id'),
        ]
    )
    if not response_candidates:
        _fail('canonical_response_id_required')
    if len(set(response_candidates)) != 1:
        _fail(
            'canonical_response_id_conflict',
            status_code=409,
            response_ids=response_candidates,
        )
    response_id = response_candidates[0]

    current_lifecycle = _token(
        response_payload.get('lifecycle_state')
        or frame.get('lifecycle_state')
        or response_payload.get('status')
    )
    late_fill = (
        response_payload.get('late_fill')
        if isinstance(response_payload.get('late_fill'), Mapping)
        else {}
    )
    late_fill_state = _token(late_fill.get('status') or late_fill.get('lifecycle_state'))
    try:
        active_count = int(late_fill.get('active_count') or 0)
        pending_count = int(late_fill.get('pending_count') or 0)
    except (TypeError, ValueError, OverflowError):
        active_count = 1
        pending_count = 1
    if (
        current_lifecycle in _ACTIVE_LATE_FILL_STATES
        or late_fill_state in _ACTIVE_LATE_FILL_STATES
        or active_count > 0
        or pending_count > 0
    ):
        _fail('active_late_fill_must_settle', status_code=409)
    if current_lifecycle not in _TERMINAL_LIFECYCLE_STATES:
        _fail(
            'nonterminal_response_frame_forbidden',
            status_code=409,
            lifecycle_state=current_lifecycle,
        )

    proposals = [
        item
        for item in (graph.get('graph_rebase_proposals') or [])
        if isinstance(item, Mapping)
        and _clean_text(item.get('proposal_id')) == expected['proposal_id']
    ]
    if not proposals:
        _fail('graph_rebase_proposal_not_found', status_code=404)
    if len(proposals) != 1:
        _fail('graph_rebase_proposal_binding_ambiguous', status_code=409)
    proposal = copy.deepcopy(dict(proposals[0]))
    if proposal.get('kind') != GRAPH_REBASE_PROPOSAL_KIND:
        _fail('graph_rebase_proposal_kind_mismatch')

    candidate_graph = (
        proposal.get('candidate_graph')
        if isinstance(proposal.get('candidate_graph'), Mapping)
        else {}
    )
    if not candidate_graph:
        _fail('graph_rebase_candidate_graph_required')
    derived_base_digest = stable_graph_digest(graph)
    derived_candidate_digest = stable_graph_digest(candidate_graph)
    derived_class = _token(proposal.get('requested_rebase_class'))
    if derived_class not in _REBASE_CLASSES:
        _fail(
            'graph_rebase_requested_class_invalid',
            requested_rebase_class=derived_class,
        )

    request_payload = (
        response_payload.get('request')
        if isinstance(response_payload.get('request'), Mapping)
        else {}
    )
    root_prompt = _clean_text(request_payload.get('prompt'))
    if derived_class == 'partial_subtree_rebase':
        if not root_prompt:
            _fail(
                'partial_rebase_current_root_prompt_truth_unavailable',
                status_code=409,
            )
        root_prompt_guard = (
            proposal.get('root_prompt_guard')
            if isinstance(proposal.get('root_prompt_guard'), Mapping)
            else {}
        )
        guarded_digest = _clean_text(root_prompt_guard.get('digest'))
        if not guarded_digest:
            _fail('partial_rebase_root_prompt_guard_missing', status_code=409)
        current_root_digest = stable_graph_rebase_prompt_digest(root_prompt)
        if guarded_digest != current_root_digest:
            _fail(
                'partial_rebase_root_prompt_guard_mismatch',
                status_code=409,
                guarded_digest=guarded_digest,
                current_root_digest=current_root_digest,
            )

    derived = {
        'response_id': response_id,
        'frame_id': frame_id,
        'proposal_id': _clean_text(proposal.get('proposal_id')),
        'base_graph_digest': derived_base_digest,
        'candidate_graph_digest': derived_candidate_digest,
        'requested_rebase_class': derived_class,
    }
    for key, expected_value in expected.items():
        actual_value = derived.get(key)
        if expected_value != actual_value:
            _fail(
                f'stale_{key}',
                status_code=409,
                expected=expected_value,
                actual=actual_value,
            )

    for key, derived_value in (
        ('base_graph_digest', derived_base_digest),
        ('candidate_graph_digest', derived_candidate_digest),
    ):
        if _clean_text(proposal.get(key)) != derived_value:
            _fail(
                f'proposal_{key}_mismatch',
                status_code=409,
                proposal_value=proposal.get(key),
                derived_value=derived_value,
            )
    proposal_response_id = _clean_text(proposal.get('target_response_id'))
    if proposal_response_id and proposal_response_id != response_id:
        _fail('proposal_target_response_id_mismatch', status_code=409)
    graph_frame_id = _clean_text(graph.get('frame_id'))
    proposal_target_frame_id = _clean_text(proposal.get('target_frame_id'))
    target_frame_id = _require_exact_identity(
        'proposal_target_frame_id',
        proposal_target_frame_id or graph_frame_id,
    )
    if graph_frame_id and proposal_target_frame_id and graph_frame_id != proposal_target_frame_id:
        _fail('proposal_target_frame_id_mismatch', status_code=409)

    reviews = [
        item
        for item in (graph.get('graph_rebase_reviews') or [])
        if isinstance(item, Mapping)
        and _clean_text(item.get('proposal_id')) == derived['proposal_id']
    ]
    if not reviews:
        _fail('graph_rebase_runtime_review_not_found', status_code=409)
    if len(reviews) != 1:
        _fail('graph_rebase_runtime_review_binding_ambiguous', status_code=409)
    review = copy.deepcopy(dict(reviews[0]))
    if review.get('kind') != GRAPH_REBASE_REVIEW_KIND:
        _fail('graph_rebase_runtime_review_kind_mismatch')
    if _token(review.get('authority')) != 'runtime_rebase_validation':
        _fail('graph_rebase_runtime_review_authority_mismatch', status_code=409)
    if _token(review.get('runtime_effect')) != 'none':
        _fail('graph_rebase_runtime_review_effect_mismatch', status_code=409)
    for key, derived_value in (
        ('proposal_id', derived['proposal_id']),
        ('base_graph_digest', derived_base_digest),
        ('candidate_graph_digest', derived_candidate_digest),
    ):
        if _clean_text(review.get(key)) != derived_value:
            _fail(
                f'runtime_review_{key}_mismatch',
                status_code=409,
                review_value=review.get(key),
                derived_value=derived_value,
            )
    runtime_review_id = _require_exact_identity(
        'runtime_review_id', review.get('review_id')
    )
    if _clean_text(review.get('target_response_id')) != response_id:
        _fail('runtime_review_target_response_id_mismatch', status_code=409)
    if _clean_text(review.get('target_frame_id')) != target_frame_id:
        _fail('runtime_review_target_frame_id_mismatch', status_code=409)
    if _clean_text(review.get('target_graph_id')) != derived_base_digest:
        _fail('runtime_review_target_graph_id_mismatch', status_code=409)

    return {
        **derived,
        'frame_sequence': current_sequence,
        'target_frame_id': target_frame_id,
        'proposal': proposal,
        'proposal_digest': _clean_text(review.get('proposal_digest')),
        'runtime_review': review,
        'runtime_review_id': runtime_review_id,
        'runtime_review_status': _token(review.get('status')),
        'runtime': runtime,
        'graph': graph,
        'root_prompt': root_prompt,
    }


def _canonical_false_negative_target(
    response_payload: Mapping[str, Any],
    *,
    expected_response_id: Any,
    expected_frame_id: Any,
    expected_frame_sequence: Any,
    expected_proposal_id: Any,
    expected_base_graph_digest: Any,
    expected_candidate_graph_digest: Any,
    expected_requested_rebase_class: Any,
) -> dict[str, Any]:
    """Bind a missed-proposal adjudication to exact settled candidate truth."""

    if not isinstance(response_payload, Mapping) or not response_payload:
        _fail('canonical_response_payload_required')
    if _nonempty_inline_authorization(response_payload):
        _fail('inline_graph_rebase_authorization_forbidden', status_code=403)
    expected_response = _require_exact_identity(
        'expected_response_id', expected_response_id
    )
    expected_frame = _require_exact_identity('expected_frame_id', expected_frame_id)
    expected_proposal = _require_exact_identity(
        'expected_proposal_id', expected_proposal_id
    )
    if _token(expected_proposal) != _NO_FORMAL_PROPOSAL_SENTINEL:
        _fail(
            'false_negative_expected_proposal_sentinel_required',
            expected=_NO_FORMAL_PROPOSAL_SENTINEL,
        )
    expected_base = _require_exact_identity(
        'expected_base_graph_digest', expected_base_graph_digest
    )
    expected_candidate = _require_exact_identity(
        'expected_candidate_graph_digest', expected_candidate_graph_digest
    )
    adjudicated_class = _require_exact_identity(
        'expected_requested_rebase_class', expected_requested_rebase_class
    )
    if _token(adjudicated_class) not in _REBASE_CLASSES:
        _fail('graph_rebase_requested_class_invalid')

    frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), Mapping)
        else {}
    )
    if frame.get('kind') != 'ollmo.response_frame':
        _fail('canonical_frozen_response_frame_required')
    frame_id = _require_exact_identity('canonical_frame_id', frame.get('frame_id'))
    try:
        expected_sequence = parse_graph_rebase_frame_sequence(expected_frame_sequence)
    except ValueError:
        _fail('expected_frame_sequence_invalid')
    try:
        frame_sequence = parse_graph_rebase_frame_sequence(frame.get('frame_sequence'))
    except ValueError:
        _fail('canonical_frame_sequence_invalid')
    if frame_id != expected_frame or frame_sequence != expected_sequence:
        _fail(
            'stale_false_negative_frame_binding',
            status_code=409,
            expected_frame_id=expected_frame,
            current_frame_id=frame_id,
            expected_frame_sequence=expected_sequence,
            current_frame_sequence=frame_sequence,
        )

    runtime = (
        response_payload.get('runtime')
        if isinstance(response_payload.get('runtime'), Mapping)
        else {}
    )
    graph = (
        runtime.get('request_phase_graph')
        if isinstance(runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    response_id = _clean_text(
        response_payload.get('response_id')
        or response_payload.get('id')
        or frame.get('response_id')
    )
    if response_id != expected_response:
        _fail('stale_response_id', status_code=409)
    lifecycle_state = _token(
        response_payload.get('lifecycle_state')
        or frame.get('lifecycle_state')
        or response_payload.get('status')
    )
    late_fill = (
        response_payload.get('late_fill')
        if isinstance(response_payload.get('late_fill'), Mapping)
        else {}
    )
    try:
        active_count = int(late_fill.get('active_count') or 0)
        pending_count = int(late_fill.get('pending_count') or 0)
    except (TypeError, ValueError, OverflowError):
        active_count = 1
        pending_count = 1
    if (
        lifecycle_state in _ACTIVE_LATE_FILL_STATES
        or _token(late_fill.get('status')) in _ACTIVE_LATE_FILL_STATES
        or active_count > 0
        or pending_count > 0
    ):
        _fail('active_late_fill_must_settle', status_code=409)
    if lifecycle_state not in _TERMINAL_LIFECYCLE_STATES:
        _fail('nonterminal_response_frame_forbidden', status_code=409)
    formal_graph_keys = (
        'graph_rebase_proposals',
        'graph_rebase_reviews',
        'graph_rebase_lifecycle',
        'staged_graph_rebases',
        'applied_graph_rebases',
        'successor_rebase_requests',
        'successor_rebase_executions',
        'graph_rebase_outcomes',
        'partial_rebase_outcomes',
    )
    formal_diagnostic_keys = (
        'runtime_graph_rebase_proposals',
        'runtime_graph_rebase_reviews',
        'graph_rebase_lifecycle',
        'staged_graph_rebases',
        'applied_graph_rebases',
        'successor_rebase_requests',
        'graph_rebase_outcomes',
        'partial_rebase_outcomes',
    )
    if any(
        any(isinstance(item, Mapping) for item in source.get(key) or [])
        for source, keys in (
            (graph, formal_graph_keys),
            (diagnostics, formal_diagnostic_keys),
        )
        for key in keys
    ):
        _fail('false_negative_requires_no_formal_rebase_truth', status_code=409)

    candidate_review = (
        diagnostics.get('runtime_graph_rebase_candidate_review')
        if isinstance(diagnostics.get('runtime_graph_rebase_candidate_review'), Mapping)
        else {}
    )
    candidate_context = (
        diagnostics.get('response_time_graph_rebase_candidate')
        if isinstance(diagnostics.get('response_time_graph_rebase_candidate'), Mapping)
        else {}
    )
    if not candidate_review:
        _fail('false_negative_candidate_observation_required', status_code=409)
    if (
        candidate_review.get('kind')
        != 'ollmo.runtime_graph_rebase_candidate_review'
        or _token(candidate_review.get('status')) != 'not_proposed'
        or _token(candidate_review.get('runtime_effect')) != 'none'
        or not _clean_text(candidate_review.get('reason'))
    ):
        _fail('false_negative_candidate_observation_invalid', status_code=409)
    if _clean_text(candidate_review.get('proposal_id')):
        _fail('false_negative_candidate_already_has_proposal', status_code=409)
    observed_base = _clean_text(
        candidate_review.get('base_graph_digest')
        or candidate_context.get('base_graph_digest')
    )
    observed_candidate = _clean_text(
        candidate_review.get('candidate_graph_digest')
        or candidate_context.get('candidate_graph_digest')
    )
    candidate_graph = (
        candidate_context.get('candidate_graph')
        if isinstance(candidate_context.get('candidate_graph'), Mapping)
        else {}
    )
    if candidate_graph:
        derived_candidate = stable_graph_digest(candidate_graph)
        if observed_candidate != derived_candidate:
            _fail('false_negative_candidate_graph_digest_mismatch', status_code=409)
    derived_base = stable_graph_digest(graph)
    if observed_base and observed_base != derived_base:
        _fail('false_negative_observed_base_digest_stale', status_code=409)
    if expected_base != derived_base or expected_candidate != observed_candidate:
        _fail('stale_false_negative_candidate_binding', status_code=409)
    candidate_observation_id = _sha256(
        {
            'response_id': response_id,
            'frame_id': frame_id,
            'base_graph_digest': derived_base,
            'candidate_graph_digest': observed_candidate,
            'candidate_review': candidate_review,
        },
        prefix='graph-rebase-candidate-observation-',
    )
    return {
        'response_id': response_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'target_frame_id': frame_id,
        'proposal_id': '',
        'candidate_observation_id': candidate_observation_id,
        'base_graph_digest': derived_base,
        'candidate_graph_digest': observed_candidate,
        'requested_rebase_class': _token(adjudicated_class),
        'runtime_review_status': 'not_proposed',
        'runtime': runtime,
        'graph': graph,
    }


def _same_proposal_binding(record: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    return all(
        _clean_text(record.get(key)) == _clean_text(target.get(key))
        for key in (
            'response_id',
            'proposal_id',
            'base_graph_digest',
            'candidate_graph_digest',
            'requested_rebase_class',
        )
    )


def _prior_records_for_target(
    records: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in records
        if isinstance(record, Mapping) and _same_proposal_binding(record, target)
    ]


def _accepted_review_record(
    records: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        record
        for record in _prior_records_for_target(records, target)
        if _token(record.get('action')) == 'adjudicate'
        and _token(record.get('adjudication')) == 'useful_proposal'
    ]
    return copy.deepcopy(matches[-1]) if matches else {}


def _stage_record(
    records: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        record
        for record in _prior_records_for_target(records, target)
        if _token(record.get('action')) == 'stage'
        and _token(record.get('status')) == 'staged'
    ]
    return copy.deepcopy(matches[-1]) if matches else {}


def _runtime_review_allows_stage(target: Mapping[str, Any]) -> None:
    review = (
        target.get('runtime_review')
        if isinstance(target.get('runtime_review'), Mapping)
        else {}
    )
    if _token(review.get('status')) != 'accepted':
        _fail(
            'accepted_runtime_rebase_review_required',
            status_code=409,
            runtime_review_status=review.get('status'),
        )
    proof = (
        review.get('preservation_proof')
        if isinstance(review.get('preservation_proof'), Mapping)
        else {}
    )
    if _token(proof.get('status')) != 'passed':
        _fail(
            'passed_graph_rebase_preservation_proof_required',
            status_code=409,
            preservation_proof_status=proof.get('status'),
        )
    redraw_scope = (
        target.get('graph', {}).get('redraw_scope_ladder_review')
        if isinstance(target.get('graph'), Mapping)
        and isinstance(target.get('graph', {}).get('redraw_scope_ladder_review'), Mapping)
        else {}
    )
    selected_scope = _token(redraw_scope.get('selected_scope'))
    if selected_scope and selected_scope != _token(target.get('requested_rebase_class')):
        _fail(
            'current_redraw_scope_no_longer_selects_proposal_class',
            status_code=409,
            selected_scope=selected_scope,
            requested_rebase_class=target.get('requested_rebase_class'),
        )


def _matching_runtime_stage(target: Mapping[str, Any]) -> dict[str, Any]:
    graph = target.get('graph') if isinstance(target.get('graph'), Mapping) else {}
    matches: list[dict[str, Any]] = []
    for item in graph.get('staged_graph_rebases') or []:
        if not isinstance(item, Mapping):
            continue
        if item.get('kind') != GRAPH_REBASE_LIFECYCLE_KIND:
            continue
        if _clean_text(item.get('proposal_id')) != _clean_text(target.get('proposal_id')):
            continue
        if _clean_text(item.get('review_id')) != _clean_text(target.get('runtime_review_id')):
            continue
        if _clean_text(item.get('base_graph_digest')) != _clean_text(
            target.get('base_graph_digest')
        ):
            continue
        if _clean_text(item.get('before_graph_digest')) != _clean_text(
            target.get('base_graph_digest')
        ):
            continue
        if _clean_text(item.get('candidate_graph_digest')) != _clean_text(
            target.get('candidate_graph_digest')
        ):
            continue
        if _token(item.get('status')) != 'staged' or _token(item.get('autonomy_level')) != 'stage':
            continue
        outcome = item.get('outcome') if isinstance(item.get('outcome'), Mapping) else {}
        if _token(outcome.get('runtime_effect')) != 'staged_no_executable_mutation':
            continue
        if _contains_wildcard(item.get('rebase_id')) or _contains_wildcard(
            item.get('idempotency_key')
        ):
            continue
        matches.append(dict(item))
    return copy.deepcopy(matches[-1]) if matches else {}


def _review_replay_projection(review: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(_json_safe(review))
    # Trusted authorization is joined after review and is intentionally not a
    # product of deterministic proposal validation. Every other review field
    # is semantic replay truth and must match exactly.
    projected.pop('graph_rebase_authorization', None)
    return projected


def _build_runtime_review_replay(target: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate frozen proposal truth and compare deterministic review output."""

    runtime = target.get('runtime') if isinstance(target.get('runtime'), Mapping) else {}
    prior_review = (
        target.get('runtime_review')
        if isinstance(target.get('runtime_review'), Mapping)
        else {}
    )
    prior_proof = (
        prior_review.get('preservation_proof')
        if isinstance(prior_review.get('preservation_proof'), Mapping)
        else {}
    )
    replayed = validate_graph_rebase_proposal(
        target.get('proposal') if isinstance(target.get('proposal'), Mapping) else {},
        request_phase_graph=(
            target.get('graph') if isinstance(target.get('graph'), Mapping) else {}
        ),
        closure_review=(
            runtime.get('graph_closure_review')
            if isinstance(runtime.get('graph_closure_review'), Mapping)
            else prior_proof.get('closure_review_summary')
            if isinstance(prior_proof.get('closure_review_summary'), Mapping)
            else None
        ),
        artifact_payload=(
            prior_proof.get('artifact_registry_summary')
            if isinstance(prior_proof.get('artifact_registry_summary'), Mapping)
            else None
        ),
        accepted_learning_hints=(
            runtime.get('accepted_learning_hints')
            if isinstance(runtime.get('accepted_learning_hints'), Mapping)
            else None
        ),
        intent_lens_review=(
            prior_proof.get('intent_lens_review_summary')
            if isinstance(prior_proof.get('intent_lens_review_summary'), Mapping)
            else None
        ),
        root_prompt=_clean_text(target.get('root_prompt')),
    )
    prior_projection = _review_replay_projection(prior_review)
    replay_projection = _review_replay_projection(replayed)
    prior_digest = _sha256(prior_projection, prefix='graph-rebase-review-replay-')
    replay_digest = _sha256(replay_projection, prefix='graph-rebase-review-replay-')
    matched = bool(prior_projection and prior_digest == replay_digest)
    return {
        'kind': 'ollmo.graph_rebase_review_replay_verification',
        'status': 'matched' if matched else 'mismatch',
        'replay_verified': matched,
        'prior_review_digest': prior_digest,
        'replayed_review_digest': replay_digest,
        'proposal_id': target.get('proposal_id'),
        'candidate_observation_id': target.get('candidate_observation_id'),
        'base_graph_digest': target.get('base_graph_digest'),
        'candidate_graph_digest': target.get('candidate_graph_digest'),
        'runtime_effect': 'none',
    }


def _normalize_promotion_gate(value: Any) -> dict[str, Any]:
    gate = dict(value) if isinstance(value, Mapping) else {}
    if gate.get('kind') != GRAPH_REBASE_PROMOTION_GATE_KIND:
        _fail('trusted_partial_promotion_gate_kind_mismatch', status_code=403)
    gate_id = _require_exact_identity('promotion_gate_id', gate.get('gate_id'))
    if _token(gate.get('gate')) != 'partial_stage_to_apply_reviewed':
        _fail('trusted_partial_promotion_gate_scope_mismatch', status_code=403)
    if _token(gate.get('status')) != 'ready':
        _fail('trusted_partial_promotion_gate_not_ready', status_code=403)
    if _token(gate.get('decision')) != 'promote':
        _fail('trusted_partial_promotion_gate_does_not_promote', status_code=403)
    policy_digest = _require_exact_identity(
        'promotion_gate_policy_digest', gate.get('policy_digest')
    )
    evidence_refs = _qualified_evidence_refs(
        gate.get('evidence_refs'),
        field='promotion_gate_evidence_refs',
    )
    return {
        'kind': GRAPH_REBASE_PROMOTION_GATE_KIND,
        'gate_id': gate_id,
        'gate': 'partial_stage_to_apply_reviewed',
        'status': 'ready',
        'decision': 'promote',
        'evidence_refs': evidence_refs,
        'policy_digest': policy_digest,
    }


def _build_record(
    *,
    target: Mapping[str, Any],
    action: str,
    adjudication: str,
    reason: str,
    evidence_refs: Sequence[str],
    review_record: Optional[Mapping[str, Any]] = None,
    stage_record: Optional[Mapping[str, Any]] = None,
    runtime_stage: Optional[Mapping[str, Any]] = None,
    promotion_gate: Optional[Mapping[str, Any]] = None,
    operator_identity: str,
    replay_verification: Optional[Mapping[str, Any]] = None,
    resolved_false_negative: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    statuses = {
        'adjudicate': 'recorded',
        'stage': 'staged',
        'authorize_partial': 'accepted',
    }
    effects = {
        'adjudicate': 'none',
        'stage': 'staged_no_executable_mutation',
        'authorize_partial': 'authorization_only_no_execution',
    }
    record: dict[str, Any] = {
        'kind': GRAPH_REBASE_OPERATOR_RECORD_KIND,
        'registry_version': GRAPH_REBASE_OPERATOR_REGISTRY_VERSION,
        'action': action,
        'status': statuses[action],
        'adjudication': adjudication,
        'authority': 'runtime_operator_registry',
        'source': 'runtime_operator_registry',
        'provenance': 'runtime_operator_registry',
        'runtime_effect': effects[action],
        'operator_identity': operator_identity,
        'operator_authentication': 'explicit_control_plane_credential',
        'response_id': target.get('response_id'),
        'frame_id': target.get('frame_id'),
        'frame_sequence': target.get('frame_sequence'),
        'target_frame_id': target.get('target_frame_id'),
        'proposal_id': target.get('proposal_id'),
        'candidate_observation_id': target.get('candidate_observation_id'),
        'proposal_digest': target.get('proposal_digest'),
        'runtime_review_id': target.get('runtime_review_id'),
        'runtime_review_status': target.get('runtime_review_status'),
        'base_graph_digest': target.get('base_graph_digest'),
        'candidate_graph_digest': target.get('candidate_graph_digest'),
        'requested_rebase_class': target.get('requested_rebase_class'),
        'reason': reason,
        'evidence_refs': list(evidence_refs),
    }
    if review_record:
        record['review_record_id'] = review_record.get('record_id')
    if stage_record:
        record['stage_record_id'] = stage_record.get('record_id')
    if runtime_stage:
        record['runtime_stage_rebase_id'] = runtime_stage.get('rebase_id')
        record['runtime_stage_idempotency_key'] = runtime_stage.get('idempotency_key')
    if promotion_gate:
        record['promotion_gate'] = _json_safe(promotion_gate)
    if replay_verification:
        record['replay_verification'] = _json_safe(replay_verification)
        record['replay_verified'] = bool(
            replay_verification.get('replay_verified') is True
        )
        record['replay_status'] = replay_verification.get('status')
    if resolved_false_negative:
        record['resolves_record_id'] = resolved_false_negative.get('record_id')
        record['resolved_candidate_observation_id'] = resolved_false_negative.get(
            'candidate_observation_id'
        )
        record['resolved_response_id'] = resolved_false_negative.get('response_id')
    record = {
        key: _json_safe(value)
        for key, value in record.items()
        if value not in (None, '', [], {})
    }
    record['record_id'] = stable_graph_rebase_operator_record_id(record)
    record['recorded_at'] = _now_iso()
    if action == 'authorize_partial':
        record['authorization'] = _authorization_from_record(record)
    record['record_digest'] = _stable_record_digest(record)
    return _json_safe(record)


def record_graph_rebase_operator_action(
    response_payload: Mapping[str, Any],
    *,
    action: str,
    adjudication: str,
    reason: str,
    evidence_refs: Sequence[str] | str,
    expected_response_id: str,
    expected_frame_id: str,
    expected_proposal_id: str,
    expected_base_graph_digest: str,
    expected_candidate_graph_digest: str,
    expected_requested_rebase_class: str,
    expected_frame_sequence: int | None = None,
    trusted_partial_promotion_gate: Optional[Mapping[str, Any]] = None,
    resolves_record_id: str = '',
    operator_identity: str,
    registry_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate and append one exact trusted operator action.

    Ordinary request/model/candidate authorization dictionaries are never an
    input to this API.  Every expected identity is CAS-only and is compared
    with bindings derived from the frozen runtime payload.
    """

    action_token = _token(action)
    if action_token not in _ACTIONS:
        _fail('graph_rebase_operator_action_invalid', action=action)
    adjudication_token = _token(adjudication)
    if adjudication_token not in _ADJUDICATIONS:
        _fail('graph_rebase_operator_adjudication_invalid', adjudication=adjudication)
    if (
        action_token in {'stage', 'authorize_partial'}
        and adjudication_token not in _ACCEPTED_ADJUDICATIONS
    ):
        _fail(
            'graph_rebase_operator_action_requires_accepted_adjudication',
            action=action_token,
            adjudication=adjudication_token,
        )
    reason_text = _clean_text(reason)
    if not reason_text:
        _fail('graph_rebase_operator_reason_required')
    if _contains_wildcard(reason_text):
        _fail('graph_rebase_operator_reason_wildcard_forbidden')
    refs = _qualified_evidence_refs(evidence_refs)
    operator_identity_value = _require_exact_identity(
        'operator_identity',
        operator_identity,
    )
    resolution_target_id = _clean_text(resolves_record_id)
    if resolution_target_id:
        _require_exact_identity('resolves_record_id', resolution_target_id)
        if action_token != 'adjudicate' or adjudication_token != 'useful_proposal':
            _fail(
                'false_negative_resolution_requires_useful_proposal_adjudication',
                status_code=409,
            )

    if action_token == 'adjudicate' and adjudication_token == 'false_negative':
        target = _canonical_false_negative_target(
            response_payload,
            expected_response_id=expected_response_id,
            expected_frame_id=expected_frame_id,
            expected_frame_sequence=expected_frame_sequence,
            expected_proposal_id=expected_proposal_id,
            expected_base_graph_digest=expected_base_graph_digest,
            expected_candidate_graph_digest=expected_candidate_graph_digest,
            expected_requested_rebase_class=expected_requested_rebase_class,
        )
    else:
        target = _canonical_response_target(
            response_payload,
            expected_response_id=expected_response_id,
            expected_frame_id=expected_frame_id,
            expected_frame_sequence=expected_frame_sequence,
            expected_proposal_id=expected_proposal_id,
            expected_base_graph_digest=expected_base_graph_digest,
            expected_candidate_graph_digest=expected_candidate_graph_digest,
            expected_requested_rebase_class=expected_requested_rebase_class,
        )
    if action_token == 'authorize_partial':
        if target.get('requested_rebase_class') == 'full_successor_rebase':
            _fail(
                'full_successor_rebase_authorization_forbidden',
                status_code=403,
            )
        if target.get('requested_rebase_class') != 'partial_subtree_rebase':
            _fail(
                'partial_subtree_rebase_authorization_required',
                status_code=403,
            )
    path = (
        Path(registry_path)
        if registry_path is not None
        else DEFAULT_GRAPH_REBASE_OPERATOR_REGISTRY_PATH
    )

    with _registry_lock(path):
        existing_records = _read_registry_unlocked(path)
        accepted_review: dict[str, Any] = {}
        stage_record: dict[str, Any] = {}
        runtime_stage: dict[str, Any] = {}
        promotion_gate: dict[str, Any] = {}
        replay_verification: dict[str, Any] = {}
        resolved_false_negative: dict[str, Any] = {}

        if action_token == 'adjudicate' and adjudication_token != 'false_negative':
            replay_verification = _build_runtime_review_replay(target)
            if (
                adjudication_token in _ACCEPTED_ADJUDICATIONS
                and replay_verification.get('replay_verified') is not True
            ):
                _fail(
                    'graph_rebase_review_replay_mismatch',
                    status_code=409,
                    replay_verification=replay_verification,
                )

        if resolution_target_id:
            resolved_false_negative = next(
                (
                    dict(record)
                    for record in existing_records
                    if _clean_text(record.get('record_id')) == resolution_target_id
                ),
                {},
            )
            if not resolved_false_negative:
                _fail(
                    'false_negative_resolution_target_not_found',
                    status_code=404,
                    resolves_record_id=resolution_target_id,
                )
            if (
                _token(resolved_false_negative.get('action')) != 'adjudicate'
                or _token(resolved_false_negative.get('adjudication'))
                != 'false_negative'
            ):
                _fail(
                    'false_negative_resolution_target_invalid',
                    status_code=409,
                    resolves_record_id=resolution_target_id,
                )
            if _token(resolved_false_negative.get('requested_rebase_class')) != _token(
                target.get('requested_rebase_class')
            ):
                _fail(
                    'false_negative_resolution_class_mismatch',
                    status_code=409,
                    resolves_record_id=resolution_target_id,
                )

        if action_token in {'stage', 'authorize_partial'}:
            _runtime_review_allows_stage(target)
            accepted_review = _accepted_review_record(existing_records, target)
            if not accepted_review:
                _fail(
                    'trusted_useful_proposal_adjudication_required',
                    status_code=409,
                )

        if action_token == 'authorize_partial':
            stage_record = _stage_record(existing_records, target)
            if not stage_record:
                _fail('trusted_graph_rebase_stage_required', status_code=409)
            runtime_stage = _matching_runtime_stage(target)
            if not runtime_stage:
                _fail('exact_runtime_graph_rebase_stage_required', status_code=409)
            promotion_gate = _normalize_promotion_gate(trusted_partial_promotion_gate)

        record = _build_record(
            target=target,
            action=action_token,
            adjudication=adjudication_token,
            reason=reason_text,
            evidence_refs=refs,
            review_record=accepted_review,
            stage_record=stage_record,
            runtime_stage=runtime_stage,
            promotion_gate=promotion_gate,
            operator_identity=operator_identity_value,
            replay_verification=replay_verification,
            resolved_false_negative=resolved_false_negative,
        )
        record_id = _clean_text(record.get('record_id'))
        prior_resolution = next(
            (
                existing
                for existing in existing_records
                if _clean_text(existing.get('resolves_record_id'))
                == resolution_target_id
            ),
            None,
        ) if resolution_target_id else None
        if isinstance(prior_resolution, Mapping):
            if _clean_text(prior_resolution.get('record_id')) == record_id:
                return copy.deepcopy(dict(prior_resolution))
            _fail(
                'false_negative_already_resolved',
                status_code=409,
                resolves_record_id=resolution_target_id,
                resolution_record_id=prior_resolution.get('record_id'),
            )
        for existing in existing_records:
            if _clean_text(existing.get('record_id')) == record_id:
                return copy.deepcopy(existing)
        _append_record_unlocked(path, record)
        return copy.deepcopy(record)


def _authorization_chain_is_trusted(
    record: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> bool:
    by_id = {
        _clean_text(item.get('record_id')): item
        for item in records
        if isinstance(item, Mapping) and _clean_text(item.get('record_id'))
    }
    review_record = by_id.get(_clean_text(record.get('review_record_id')))
    stage_record = by_id.get(_clean_text(record.get('stage_record_id')))
    if not isinstance(review_record, Mapping) or not isinstance(stage_record, Mapping):
        return False
    if not _same_proposal_binding(review_record, record):
        return False
    if not _same_proposal_binding(stage_record, record):
        return False
    if _token(review_record.get('action')) != 'adjudicate':
        return False
    if _token(review_record.get('adjudication')) != 'useful_proposal':
        return False
    if _token(stage_record.get('action')) != 'stage':
        return False
    if _token(stage_record.get('status')) != 'staged':
        return False
    if _clean_text(stage_record.get('review_record_id')) != _clean_text(
        review_record.get('record_id')
    ):
        return False
    return True


def find_trusted_graph_rebase_authorization(
    *,
    response_id: str,
    proposal_id: str,
    base_graph_digest: str,
    candidate_graph_digest: str,
    requested_rebase_class: str = 'partial_subtree_rebase',
    frame_id: str = '',
    registry_path: Path | str | None = None,
) -> dict[str, Any]:
    """Join an exact authorization from verified registry truth only."""

    expected = {
        'response_id': _require_exact_identity('response_id', response_id),
        'proposal_id': _require_exact_identity('proposal_id', proposal_id),
        'base_graph_digest': _require_exact_identity('base_graph_digest', base_graph_digest),
        'candidate_graph_digest': _require_exact_identity(
            'candidate_graph_digest', candidate_graph_digest
        ),
        'requested_rebase_class': _require_exact_identity(
            'requested_rebase_class', requested_rebase_class
        ),
    }
    if expected['requested_rebase_class'] != 'partial_subtree_rebase':
        return {}
    expected_frame_id = _clean_text(frame_id)
    if expected_frame_id and _contains_wildcard(expected_frame_id):
        _fail('frame_id_wildcard_forbidden')
    records = load_graph_rebase_operator_records(registry_path=registry_path)
    matches = [
        record
        for record in records
        if _token(record.get('action')) == 'authorize_partial'
        and _token(record.get('status')) == 'accepted'
        and all(_clean_text(record.get(key)) == value for key, value in expected.items())
        and (
            not expected_frame_id
            or _clean_text(record.get('frame_id')) == expected_frame_id
        )
        and _authorization_chain_is_trusted(record, records)
    ]
    if not matches:
        return {}
    authorization = matches[-1].get('authorization')
    return copy.deepcopy(dict(authorization)) if isinstance(authorization, Mapping) else {}


__all__ = [
    'DEFAULT_GRAPH_REBASE_OPERATOR_REGISTRY_PATH',
    'GRAPH_REBASE_AUTHORIZATION_KIND',
    'GRAPH_REBASE_OPERATOR_RECORD_KIND',
    'GRAPH_REBASE_OPERATOR_REGISTRY_VERSION',
    'GRAPH_REBASE_PROMOTION_GATE_KIND',
    'GraphRebaseOperatorRegistryError',
    'find_trusted_graph_rebase_authorization',
    'load_graph_rebase_operator_records',
    'record_graph_rebase_operator_action',
    'stable_graph_rebase_operator_record_id',
    'verify_graph_rebase_operator_record',
]
