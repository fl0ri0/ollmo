"""Mutable Responses runtime state owners for Ollmo."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


_ACTIVE_LATE_FILL_STATUSES = {'pending', 'queued', 'scheduled', 'accepted', 'running', 'in_progress'}
_TERMINAL_LATE_FILL_STATUSES = {
    'completed',
    'skipped',
    'failed',
    'cancelled',
    'partial_failed',
    'partial_cancelled',
    'blocked',
    'waived',
    'superseded',
}
_OPEN_RESPONSE_LIFECYCLE_STATES = frozenset(
    {
        'accepted',
        'active',
        'in_progress',
        'late_fill_pending',
        'late_fill_running',
        'pending',
        'queued',
        'running',
        'streaming',
    }
)
_ACTIONABLE_RESPONSE_LIFECYCLE_STATES = frozenset(
    {
        'blocked',
        'late_fill_repair_needed',
        'rebuild_from_promoted_obligations',
        'repair_branch_contract',
        'repair_dependency_chain',
        'repair_needed',
    }
)
_TERMINAL_RESPONSE_LIFECYCLE_STATES = frozenset(
    {
        'canceled',
        'cancelled',
        'completed',
        'failed',
        'frozen',
        'late_fill_completed',
        'late_fill_failed',
        'partial_cancelled',
        'skipped',
        'superseded',
        'waived',
    }
)
_CANONICAL_RESPONSE_LIFECYCLE_STATES = frozenset(
    _OPEN_RESPONSE_LIFECYCLE_STATES
    | _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
    | _TERMINAL_RESPONSE_LIFECYCLE_STATES
)


def _late_fill_branches_need_repair(branches: list[Any]) -> bool:
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        status = str(branch.get('status') or '').strip().lower()
        if status in {
            'blocked',
            'failed',
            'partial_failed',
            'repair_branch_contract',
            'repair_dependency_chain',
            'repair_needed',
        }:
            return True
        recovery_context = branch.get('recovery_context') if isinstance(branch.get('recovery_context'), Mapping) else {}
        if recovery_context.get('suggested_action') not in (None, '', [], {}):
            return True
    return False


def _late_fill_has_repair_signal(late_fill: Mapping[str, Any], *branch_groups: list[Any]) -> bool:
    if isinstance(late_fill.get('recovery_candidates'), list) and late_fill.get('recovery_candidates'):
        return True
    if isinstance(late_fill.get('repair_actions'), list) and late_fill.get('repair_actions'):
        return True
    if late_fill.get('repair_action') not in (None, '', [], {}):
        return True
    recovery_state = late_fill.get('recovery_state') if isinstance(late_fill.get('recovery_state'), Mapping) else {}
    if recovery_state and str(recovery_state.get('status') or '').strip().lower() in {'candidate', 'ready', 'blocked'}:
        return True
    return any(_late_fill_branches_need_repair(group) for group in branch_groups)


def late_fill_has_actionable_repair_work(late_fill: Optional[Mapping[str, Any]]) -> bool:
    """Return current nested repair-loop work, excluding stale terminal labels."""

    payload = late_fill if isinstance(late_fill, Mapping) else {}
    repair_loop = payload.get('repair_loop') if isinstance(payload.get('repair_loop'), Mapping) else {}
    loop_status = str(repair_loop.get('status') or '').strip().lower()
    if loop_status not in {
        'active',
        'pending',
        'promoted',
        'queued',
        'repair_needed',
        'running',
        'scheduled',
    }:
        return False
    if repair_loop.get('repair_work_available') is True:
        return True
    for key in ('repair_work_available_count', 'executable_contract_count'):
        try:
            if int(repair_loop.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError, OverflowError):
            continue
    return False


def _canonical_response_lifecycle_token(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    normalized = value.strip().lower()
    return normalized if normalized in _CANONICAL_RESPONSE_LIFECYCLE_STATES else ''


def _response_frame_lifecycle_state(payload: Mapping[str, Any]) -> str:
    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), Mapping)
        else {}
    )
    current_state = (
        response_frame.get('current_state')
        if isinstance(response_frame.get('current_state'), Mapping)
        else {}
    )
    status_semantics = (
        current_state.get('status_semantics')
        if isinstance(current_state.get('status_semantics'), Mapping)
        else {}
    )
    return _canonical_response_lifecycle_token(
        current_state.get('lifecycle_state')
        or status_semantics.get('canonical_lifecycle_state')
    )


def _runtime_graph_closure_reviews(runtime: Any) -> list[Mapping[str, Any]]:
    if not isinstance(runtime, Mapping):
        return []
    review = runtime.get('graph_closure_review')
    if isinstance(review, Mapping) and review:
        return [review]
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    diagnostic_review = diagnostics.get('graph_closure_review')
    if isinstance(diagnostic_review, Mapping) and diagnostic_review:
        return [diagnostic_review]
    return []


def _response_graph_closure_reviews(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # A terminal successor is assembled while its frozen parent frame is still
    # attached for lineage.  Once Runtime has refreshed Closure on the current
    # payload, that review is the current-state authority; the nested parent
    # review is historical fallback, not a concurrent veto.
    current_reviews = _runtime_graph_closure_reviews(payload.get('runtime'))
    if current_reviews:
        return current_reviews
    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), Mapping)
        else {}
    )
    current_state = (
        response_frame.get('current_state')
        if isinstance(response_frame.get('current_state'), Mapping)
        else {}
    )
    runtime_candidates = [current_state.get('runtime'), response_frame.get('runtime')]
    reviews: list[Mapping[str, Any]] = []
    for runtime in runtime_candidates:
        reviews.extend(_runtime_graph_closure_reviews(runtime))
    return reviews


def _graph_closure_requires_repair(payload: Mapping[str, Any]) -> bool:
    """Return unresolved Closure truth without promoting advisory-only state."""

    for review in _response_graph_closure_reviews(payload):
        status = str(review.get('status') or '').strip().lower()
        if status in {'blocked', 'pending', 'repair_needed', 'repair_pending'}:
            return True
        if review.get('continuation_required') is True:
            return True
        counts = review.get('counts') if isinstance(review.get('counts'), Mapping) else {}
        for key in ('blocked', 'open', 'pending', 'repair_pending', 'semantic_review_pending'):
            try:
                if int(counts.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                continue
        surface_state = (
            review.get('surface_state')
            if isinstance(review.get('surface_state'), Mapping)
            else {}
        )
        surface_status = str(surface_state.get('status') or '').strip().lower()
        if surface_status in {'blocked', 'pending', 'repair_needed', 'repair_pending'}:
            return True
        category_counts = (
            surface_state.get('category_counts')
            if isinstance(surface_state.get('category_counts'), Mapping)
            else {}
        )
        for key in ('blocked', 'open', 'repair_pending', 'semantic_review_pending'):
            try:
                if int(category_counts.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                continue
    return False


def derive_response_lifecycle_state(
    response_payload: Optional[Mapping[str, Any]],
    *,
    requested_status: Optional[str] = None,
) -> str:
    """Return the canonical lifecycle state for response lookup/current-state views."""

    payload = response_payload if isinstance(response_payload, Mapping) else {}
    explicit_lifecycle = _canonical_response_lifecycle_token(payload.get('lifecycle_state'))
    late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
    late_fill_status = str(late_fill.get('status') or '').strip().lower()
    pending_branches = late_fill.get('pending_branches') if isinstance(late_fill.get('pending_branches'), list) else []
    active_branches = late_fill.get('active_branches') if isinstance(late_fill.get('active_branches'), list) else []
    failed_branches = late_fill.get('failed_branches') if isinstance(late_fill.get('failed_branches'), list) else []
    recovery_candidates = (
        late_fill.get('recovery_candidates')
        if isinstance(late_fill.get('recovery_candidates'), list)
        else []
    )
    derived_lifecycle = ''
    if recovery_candidates:
        derived_lifecycle = 'repair_needed'
    elif late_fill_status in {'blocked'}:
        derived_lifecycle = 'blocked'
    elif late_fill_status in {'failed', 'partial_failed'} or failed_branches:
        if _late_fill_has_repair_signal(late_fill, pending_branches, active_branches, failed_branches):
            derived_lifecycle = 'repair_needed'
        else:
            derived_lifecycle = 'late_fill_failed'
    elif late_fill_status in _ACTIVE_LATE_FILL_STATUSES or active_branches or pending_branches:
        if late_fill_status == 'running' or active_branches:
            derived_lifecycle = 'late_fill_running'
        else:
            derived_lifecycle = 'late_fill_pending'
    elif late_fill_status in {'cancelled', 'partial_cancelled', 'waived', 'superseded'}:
        derived_lifecycle = late_fill_status
    elif late_fill_status in _TERMINAL_LATE_FILL_STATUSES:
        if late_fill_has_actionable_repair_work(late_fill):
            derived_lifecycle = 'repair_needed'
        else:
            derived_lifecycle = 'completed'
    if not derived_lifecycle:
        requested = _canonical_response_lifecycle_token(requested_status)
        if requested_status is None or (
            isinstance(requested_status, str) and not requested_status.strip()
        ):
            requested = _canonical_response_lifecycle_token(payload.get('status'))
        derived_lifecycle = requested or 'in_progress'

    live_late_fill_is_active = bool(
        late_fill_status in _ACTIVE_LATE_FILL_STATUSES
        or active_branches
        or pending_branches
    )
    if live_late_fill_is_active:
        if explicit_lifecycle in {
            'canceled',
            'cancelled',
            'failed',
            'late_fill_failed',
            'partial_cancelled',
            'skipped',
            'superseded',
            'waived',
        }:
            return explicit_lifecycle
        # Current Late Fill branch truth outranks any stale outer lifecycle
        # projection, including repair_needed from the terminal state that an
        # explicit retry is reopening.  The derived state still preserves
        # blocked/failed repair truth when that is what the Late Fill payload
        # itself reports.
        return derived_lifecycle
    frame_lifecycle = _response_frame_lifecycle_state(payload)
    has_current_closure_review = bool(
        _runtime_graph_closure_reviews(payload.get('runtime'))
    )
    unresolved_closure = _graph_closure_requires_repair(payload)
    if not live_late_fill_is_active and (
        (
            not has_current_closure_review
            and frame_lifecycle in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
        )
        or unresolved_closure
    ):
        # Frozen frame/Closure truth is stronger than a stale successful
        # lookup projection. Non-success terminal states remain terminal.
        if explicit_lifecycle in {
            'canceled',
            'cancelled',
            'failed',
            'late_fill_failed',
            'partial_cancelled',
            'skipped',
            'superseded',
            'waived',
        }:
            return explicit_lifecycle
        if (
            not has_current_closure_review
            and frame_lifecycle in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
        ):
            return frame_lifecycle
        return 'repair_needed'

    if not explicit_lifecycle:
        return derived_lifecycle
    has_late_fill_truth = bool(late_fill)
    actionable_repair_loop = late_fill_has_actionable_repair_work(late_fill)
    terminal_without_open_repair = bool(
        late_fill_status in {'completed', 'skipped', 'cancelled', 'waived', 'superseded'}
        and not pending_branches
        and not active_branches
        and not failed_branches
        and not recovery_candidates
        and not actionable_repair_loop
    )
    actionable_repair = bool(
        not terminal_without_open_repair
        and (
            explicit_lifecycle in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
            or actionable_repair_loop
            or _late_fill_has_repair_signal(
                late_fill,
                pending_branches,
                active_branches,
                failed_branches,
            )
        )
    )
    if explicit_lifecycle in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES:
        if (
            derived_lifecycle in _TERMINAL_RESPONSE_LIFECYCLE_STATES
            and not actionable_repair
        ):
            return derived_lifecycle
        return explicit_lifecycle
    if explicit_lifecycle in _TERMINAL_RESPONSE_LIFECYCLE_STATES:
        if (
            derived_lifecycle in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
            and actionable_repair
        ):
            return derived_lifecycle
        return explicit_lifecycle
    if explicit_lifecycle in _OPEN_RESPONSE_LIFECYCLE_STATES:
        if derived_lifecycle in _OPEN_RESPONSE_LIFECYCLE_STATES or has_late_fill_truth:
            return derived_lifecycle or explicit_lifecycle
        return explicit_lifecycle
    return explicit_lifecycle


@dataclass
class ResponsesRuntimeOwner:
    response_lookup: dict[str, dict[str, Any]]
    response_lookup_lock: threading.Lock
    response_streams: dict[str, dict[str, Any]]
    response_streams_lock: threading.Lock
    response_late_fill_in_flight: set[str]
    response_late_fill_lock: threading.Lock
    response_lookup_ttl_sec: int
    normalize_response_lookup_id: Callable[[Any], str]
    response_registry_now_iso: Callable[[], str]

    def _prune_response_lookup_registry(self, now_ts: Optional[float] = None) -> None:
        cutoff = float(now_ts or time.time())
        expired_ids = [
            response_id
            for response_id, record in self.response_lookup.items()
            if float(record.get('expires_at_ts') or 0) < cutoff
        ]
        for response_id in expired_ids:
            self.response_lookup.pop(response_id, None)

    def register_response_lookup(
        self,
        *,
        response_id: str,
        message_id: str,
        instance_id: str,
        model_name: str,
        backend: str,
        capability: str,
        mode: str,
        route_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        now_ts = time.time()
        record = {
            'id': self.normalize_response_lookup_id(response_id),
            'message_id': str(message_id or '').strip() or f'msg_{uuid.uuid4().hex}',
            'instance_id': str(instance_id or '').strip(),
            'model_name': str(model_name or '').strip(),
            'backend': str(backend or '').strip(),
            'capability': str(capability or '').strip(),
            'mode': str(mode or 'chat').strip() or 'chat',
            'status': 'in_progress',
            'lifecycle_state': 'in_progress',
            'output_text': '',
            'error_message': None,
            'response_payload': None,
            'route_payload': dict(route_payload or {}),
            'created_at': self.response_registry_now_iso(),
            'updated_at': self.response_registry_now_iso(),
            'expires_at_ts': now_ts + self.response_lookup_ttl_sec,
        }
        with self.response_lookup_lock:
            self._prune_response_lookup_registry(now_ts)
            self.response_lookup[record['id']] = record
            return dict(record)

    def touch_response_lookup(
        self,
        response_id: str,
        *,
        status: Optional[str] = None,
        output_text: Optional[str] = None,
        error_message: Optional[str] = None,
        response_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        normalized_id = self.normalize_response_lookup_id(response_id)
        now_ts = time.time()
        with self.response_lookup_lock:
            self._prune_response_lookup_registry(now_ts)
            record = self.response_lookup.get(normalized_id)
            if not record:
                return None
            if status:
                record['status'] = str(status).strip() or record.get('status') or 'in_progress'
            if output_text is not None:
                record['output_text'] = str(output_text)
            if error_message is not None:
                record['error_message'] = str(error_message or '').strip() or None
            if response_payload is not None:
                record['response_payload'] = dict(response_payload)
            record['lifecycle_state'] = derive_response_lifecycle_state(
                record.get('response_payload') if isinstance(record.get('response_payload'), Mapping) else None,
                requested_status=str(record.get('status') or status or '').strip() or None,
            )
            record['updated_at'] = self.response_registry_now_iso()
            record['expires_at_ts'] = now_ts + self.response_lookup_ttl_sec
            return dict(record)

    def get_response_lookup_record(self, response_id: str) -> Optional[dict[str, Any]]:
        normalized_id = self.normalize_response_lookup_id(response_id)
        now_ts = time.time()
        with self.response_lookup_lock:
            self._prune_response_lookup_registry(now_ts)
            record = self.response_lookup.get(normalized_id)
            return dict(record) if record else None

    def advance_response_recovery_checkpoint(
        self,
        response_id: str,
        ledger_stat: Mapping[str, Any],
    ) -> bool:
        """Atomically advance only a recovered record's ledger checkpoint."""

        normalized_id = self.normalize_response_lookup_id(response_id)
        try:
            checkpoint = {
                key: int(ledger_stat[key])
                for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
            }
        except (KeyError, TypeError, ValueError):
            return False
        with self.response_lookup_lock:
            record = self.response_lookup.get(normalized_id)
            if not isinstance(record, dict):
                return False
            record['response_frame_ledger_stat'] = checkpoint
        return True

    def register_response_stream(self, response_id: str) -> dict[str, Any]:
        normalized_id = self.normalize_response_lookup_id(response_id)
        stream_state = {
            'events': [],
            'done': False,
            'condition': threading.Condition(),
        }
        with self.response_streams_lock:
            self.response_streams[normalized_id] = stream_state
        return stream_state

    def append_response_stream_events(self, response_id: str, events: list[str], *, done: bool = False) -> None:
        normalized_id = self.normalize_response_lookup_id(response_id)
        with self.response_streams_lock:
            stream_state = self.response_streams.get(normalized_id)
        if not stream_state:
            return
        condition = stream_state['condition']
        with condition:
            if events:
                stream_state['events'].extend(events)
            if done:
                stream_state['done'] = True
            condition.notify_all()

    def wait_for_response_stream_events(
        self,
        response_id: str,
        cursor: int,
        timeout_sec: float = 0.5,
    ) -> tuple[list[str], bool]:
        normalized_id = self.normalize_response_lookup_id(response_id)
        with self.response_streams_lock:
            stream_state = self.response_streams.get(normalized_id)
        if not stream_state:
            return [], True
        condition = stream_state['condition']
        with condition:
            if len(stream_state['events']) <= cursor and not stream_state['done']:
                condition.wait(timeout=timeout_sec)
            events = list(stream_state['events'][cursor:])
            done = bool(stream_state['done'])
        return events, done

    def close_response_stream(self, response_id: str) -> None:
        normalized_id = self.normalize_response_lookup_id(response_id)
        with self.response_streams_lock:
            self.response_streams.pop(normalized_id, None)

    def claim_response_late_fill(self, response_id: str) -> bool:
        normalized_id = self.normalize_response_lookup_id(response_id)
        if not normalized_id:
            return False
        with self.response_late_fill_lock:
            if normalized_id in self.response_late_fill_in_flight:
                return False
            self.response_late_fill_in_flight.add(normalized_id)
        return True

    def release_response_late_fill(self, response_id: str) -> None:
        normalized_id = self.normalize_response_lookup_id(response_id)
        if not normalized_id:
            return
        with self.response_late_fill_lock:
            self.response_late_fill_in_flight.discard(normalized_id)
