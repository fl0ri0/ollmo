"""Unified durable event/history layer for canonical Ollmo flows."""

from __future__ import annotations

from ollmo_services.state_flow import state_flow_scope, observed_call as state_flow_observed_call, operation as state_flow_operation, transition as state_flow_transition, _summary as state_flow_summary, _SCOPE as _STATE_FLOW_SCOPE

import datetime as dt
import copy
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_EVENT_LOG_PATH = Path('state/events.jsonl')
_EVENT_LOG_LOCK = threading.Lock()

# Observation budgets, not execution limits. Overflow is explicit and invalidates
# complete-interval claims. The event writer's 50-item bound also applies here.
CAUSAL_RECORD_LIMIT = 1024
CAUSAL_CAPTURE_TAIL = 24
CAUSAL_RECORD_BYTES = 32768
CAUSAL_FINALIZER_RESERVE = 32
CAUSAL_TRANSITION_RESERVE = 256
CAUSAL_SCHEMA = 'ollmo.causal_event.v1'
_PROCESS_BOOT = uuid.uuid4().hex
_CAUSAL_SCOPE = ContextVar('ollmo_causal_scope', default=None)
_CAUSAL_CALL = ContextVar('ollmo_causal_call', default=None)
_OPERATION_ROLE = ContextVar('ollmo_operation_role', default=None)
_TRANSITION_CONTEXT = ContextVar('ollmo_transition_context', default=None)
_TRANSITION_SPAN = ContextVar('ollmo_transition_span', default=None)
_TRANSITION_ATTEMPT_FIELDS = ('attempt_id', 'attempt_count', 'attempt_number',
                              'retry_count', 'auto_executable_repair_retry_count')


def relevant_digest(value: Any) -> str:
    """Hash an owner's selected inputs, never a response or output snapshot."""
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False,
                                  separators=(',', ':'), allow_nan=False).iterencode(value):
        digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


def select_fields(record: Any, keys) -> dict:
    return {key: record[key] for key in keys if key in record} if isinstance(record, Mapping) else {}


def exact_target(record: Any, *keys: str) -> dict:
    """No phase/branch fallback: absent identity stays absent."""
    if not isinstance(record, Mapping):
        return {}
    return {key: record[key] for key in (keys or (
        'response_id', 'frame_id', 'graph_id', 'branch_id', 'phase_id',
        'candidate_id', 'obligation_id', 'task_id', 'proposal_id', 'review_id',
    )) if record.get(key) not in (None, '')}


def judgment_summary(value):
    """Bounded result references, never a replacement for canonical evidence."""
    if not isinstance(value, Mapping):
        return None
    summary = select_fields(value, (
        'kind', 'status', 'action', 'accepted', 'authority', 'runtime_effect',
        'reason', 'lens', 'semantic_review_lens', 'scope', 'source_kind',
        'semantic_depth', 'structural_granularity', 'recommended_transition',
        'commitment_action', 'aspiration_action', 'decision',
        'base_graph_digest', 'candidate_graph_digest', 'applied',
    ))
    summary.update(exact_target(value))
    for key in ('evidence_refs', 'allowed_transitions', 'promotions', 'decisions',
                'checks', 'frames', 'lenses', 'rejection_reasons', 'owed_branch_ids'):
        items = value.get(key)
        if isinstance(items, (list, tuple)):
            summary[key + '_count'] = len(items)
            summary[key + '_hash'] = relevant_digest(items)
            summary[key] = [exact_target(item) if isinstance(item, Mapping) else item
                            for item in items[:12]]
            if len(items) > 12:
                summary[key + '_omitted'] = len(items) - 12
    return summary


class _CausalScope:
    def __init__(self, sink, response_id=None):
        self.sink = sink
        self.response_id = response_id
        self.scope_id = 'scope-' + uuid.uuid4().hex
        self.tail = deque(maxlen=CAUSAL_CAPTURE_TAIL)
        self.unbound = []
        self.count = 0
        self.dropped = 0
        self.delivery_failures = 0
        self.reserved_count = 0
        self.transition_reserved_count = 0
        self.transition_recorded_count = 0
        self.transition_dropped_count = 0
        self.transition_checks = {}
        self.previous = {}
        self.lock = threading.RLock()

    def _deliver(self, record):
        try:
            self.sink(category='responses', action='causal_observation',
                      status=record.get('status', 'observed'),
                      response_id=self.response_id, causal_event=record)
        except Exception:  # Telemetry is never an execution prerequisite.
            self.delivery_failures += 1

    def reserve_transition(self, count):
        with self.lock:
            if self.transition_reserved_count + count > CAUSAL_TRANSITION_RESERVE:
                self.dropped += count
                self.transition_dropped_count += count
                return False
            self.transition_reserved_count += count
            return True

    def normal_budget_exhausted(self):
        return self.count - self.transition_recorded_count >= CAUSAL_RECORD_LIMIT

    def retain(self, record, *, transition_reserved=False):
        with self.lock:
            reserved = (record.get('owner') == 'response_frame.finalize'
                        and self.reserved_count < CAUSAL_FINALIZER_RESERVE)
            if self.normal_budget_exhausted() and not reserved and not transition_reserved:
                self.dropped += 1
                return
            if not transition_reserved and self.normal_budget_exhausted():
                self.reserved_count += 1
            # Freeze the start witness before the enclosing span accumulates
            # timers; copied captures must never disagree on an event identity.
            record = copy.deepcopy(record)
            bounded = _truncate(record, max_chars=1024)
            if bounded != record:
                bounded['coverage_incomplete'] = True
                bounded['input_coverage_complete'] = False
            if len(json.dumps(bounded, ensure_ascii=False).encode('utf-8')) > CAUSAL_RECORD_BYTES:
                bounded = {key: value for key, value in bounded.items() if key not in (
                    'judgment', 'input_field_hashes', 'operations')}
                bounded['coverage_incomplete'] = True
                bounded['input_coverage_complete'] = False
            if len(json.dumps(bounded, ensure_ascii=False).encode('utf-8')) > CAUSAL_RECORD_BYTES:
                bounded = {key: bounded[key] for key in (
                    'schema', 'event_id', 'scope_id', 'owner', 'record_kind',
                    'invocation_id', 'start_event_id', 'parent_invocation_id',
                    'trigger_event_id', 'status', 'process_id', 'process_boot_id',
                    'start_monotonic_ns', 'end_monotonic_ns', 'monotonic_ns',
                ) if key in bounded}
                bounded.update(coverage_incomplete=True, input_coverage_complete=False,
                               target={}, omission_reason='causal_record_byte_limit')
            record = bounded
            if transition_reserved:
                self.transition_recorded_count += 1
            self.count += 1
            self.tail.append(record)
            if self.response_id:
                self._deliver(record)
            else:
                self.unbound.append(record)

    def bind(self, response_id):
        with self.lock:
            if not response_id or (self.response_id and self.response_id != response_id):
                return
            self.response_id = response_id
            for record in self.unbound:
                self._deliver(record)
            self.unbound.clear()

    def snapshot(self):
        with self.lock:
            return dict(kind='ollmo.causal_telemetry', version=1,
                        authority='observer_only', scope_id=self.scope_id,
                        record_limit=CAUSAL_RECORD_LIMIT, capture_tail_limit=CAUSAL_CAPTURE_TAIL,
                        finalizer_reserve=CAUSAL_FINALIZER_RESERVE,
                        transition_reserve=CAUSAL_TRANSITION_RESERVE,
                        transition_reserved_count=self.transition_reserved_count,
                        transition_recorded_count=self.transition_recorded_count,
                        transition_dropped_count=self.transition_dropped_count,
                        reserved_count=self.reserved_count,
                        recorded_count=self.count, dropped_count=self.dropped,
                        delivery_failure_count=self.delivery_failures,
                        tail_omitted_count=max(0, self.count - len(self.tail)),
                        interval_complete=False, events=list(self.tail))


@contextmanager
def causal_scope(sink, *, response_id=None):
    """Use the existing unified-event sink; no new storage or execution surface."""
    scope = _CausalScope(sink, response_id)
    token = _CAUSAL_SCOPE.set(scope)
    try:
        yield scope
    finally:
        # Unbound failures remain unbound. Never guess a response by timestamps.
        with scope.lock:
            for record in scope.unbound:
                scope._deliver(record)
            if scope.dropped or scope.delivery_failures:
                scope._deliver(dict(schema=CAUSAL_SCHEMA, record_kind='coverage_gap',
                                    event_id='event-' + uuid.uuid4().hex,
                                    scope_id=scope.scope_id, dropped_count=scope.dropped,
                                    transition_dropped_count=scope.transition_dropped_count,
                                    delivery_failure_count=scope.delivery_failures,
                                    status='incomplete', authority='observer_only'))
        _CAUSAL_SCOPE.reset(token)


def causal_snapshot(response_id=None):
    scope = _CAUSAL_SCOPE.get()
    if scope is None:
        return None
    scope.bind(response_id)
    return scope.snapshot()


def causal_lineage():
    call = _CAUSAL_CALL.get()
    return {'parent_invocation_id': call['invocation_id'],
            'trigger_event_id': call['event_id']} if call else {}


def causal_timing():
    call = _CAUSAL_CALL.get()
    if call is None:
        return {}
    return dict(invocation_id=call['invocation_id'], start_event_id=call['event_id'],
                start_monotonic_ns=call['start_monotonic_ns'],
                process_id=call['process_id'], process_boot_id=call['process_boot_id'],
                operations={k: dict(v) for k, v in call.get('operations', {}).items()})


def causal_event(owner, record_kind, *, target=None, **fields):
    scope = _CAUSAL_SCOPE.get()
    if scope is None:
        return None
    try:
        record = dict(schema=CAUSAL_SCHEMA, event_id='event-' + uuid.uuid4().hex,
                      scope_id=scope.scope_id, owner=owner, record_kind=record_kind,
                      authority='observer_only', target=target or {},
                      process_id=os.getpid(), process_boot_id=f'{os.getpid()}:{_PROCESS_BOOT}',
                      monotonic_ns=time.monotonic_ns(), **causal_lineage())
        record.update(fields)
        scope.retain(record)
        return record
    except Exception:
        scope.delivery_failures += 1
        return None


def traced_thread_target(target):
    """Carry observation context through an existing worker, without scheduling it."""
    context = copy_context()
    return lambda *args, **kwargs: context.run(target, *args, **kwargs)


def transition_attempt(record):
    """Observer identity for one actual handoff, never a runtime retry counter."""
    scope = _CAUSAL_SCOPE.get()
    if scope is None or scope.transition_reserved_count >= CAUSAL_TRANSITION_RESERVE:
        return None
    try:
        target = exact_target(record)
        branch = record
        for _ in range(2):
            nested = branch.get('branch')
            if not isinstance(nested, Mapping):
                break
            branch = nested
        target.update(exact_target(branch))
        if scope.response_id:
            target['response_id'] = scope.response_id
        attempt = select_fields(branch, _TRANSITION_ATTEMPT_FIELDS)
        key = (target.get('response_id'), target.get('branch_id'), target.get('phase_id'))
        with scope.lock:
            check = dict(scope.transition_checks.get(key) or {})
        return dict(target=target, handoff_attempt_id='handoff-' + uuid.uuid4().hex,
                    canonical_attempt=attempt, last_target_start_check=check,
                    start_check_attempt_binding='unknown' if not attempt or check.get('canonical_attempt') != attempt else 'matching_recorded_attempt',
                    predecessor_result_id=None)
    except Exception:
        scope.delivery_failures += 1
        return None


@contextmanager
def transition_binding(binding):
    token = _TRANSITION_CONTEXT.set(binding)
    try:
        yield
    finally:
        _TRANSITION_CONTEXT.reset(token)


def _transition_record(scope, owner, boundary, *, target=None, **fields):
    """Called only after reserving a bounded slot (or both ends of a span)."""
    try:
        binding = _TRANSITION_CONTEXT.get() or {}
        identity = dict(binding.get('target') or {})
        identity.update(target or {})
        if scope.response_id:
            identity.setdefault('response_id', scope.response_id)
        record = dict(schema=CAUSAL_SCHEMA, event_id='event-' + uuid.uuid4().hex,
                      scope_id=scope.scope_id, owner=owner,
                      record_kind='transition_boundary', boundary=boundary,
                      authority='observer_only', status='observed', target=identity,
                      process_id=os.getpid(), process_boot_id=f'{os.getpid()}:{_PROCESS_BOOT}',
                      monotonic_ns=time.monotonic_ns(),
                      handoff_attempt_id=binding.get('handoff_attempt_id'),
                      canonical_attempt=binding.get('canonical_attempt') or select_fields(
                          identity, _TRANSITION_ATTEMPT_FIELDS),
                      wave_id=binding.get('wave_id'),
                      last_target_start_check=binding.get('last_target_start_check') or None,
                      start_check_attempt_binding=binding.get('start_check_attempt_binding', 'unknown'),
                      parent_transition_event_id=(_TRANSITION_SPAN.get() or {}).get('event_id'),
                      predecessor_result_id=binding.get('predecessor_result_id'),
                      complete_start_authority=None, **causal_lineage())
        record.update(fields)
        # Empty/null projection fields may be removed by canonical snapshot
        # normalization. Keep absence explicit in durable captures as well.
        record['start_authority_status'] = 'unknown'
        record['canonical_attempt_status'] = 'recorded' if record.get('canonical_attempt') else 'unknown'
        record['handoff_attempt_status'] = 'recorded' if record.get('handoff_attempt_id') else 'unknown'
        record['predecessor_result_status'] = 'recorded' if record.get('predecessor_result_id') else 'unknown'
        if boundary == 'enter':
            record['requires_exit'] = True
        elif boundary == 'exit':
            record['span_start_status'] = 'recorded' if record.get('start_event_id') else 'unavailable'
        if boundary == 'available':
            record['result_id'] = record['event_id'] + ':result'
        scope.retain(record, transition_reserved=True)
        if owner == 'late_fill.start_check' and boundary == 'exit':
            key = (identity.get('response_id'), identity.get('branch_id'), identity.get('phase_id'))
            with scope.lock:
                scope.transition_checks[key] = select_fields(record, (
                    'event_id', 'monotonic_ns', 'canonical_attempt', 'gate_status',
                    'gate_action', 'gate_check_complete', 'outcome',
                ))
        return record
    except Exception:
        scope.delivery_failures += 1
        return None


def transition_marker(owner, boundary, *, target=None, **fields):
    scope = _CAUSAL_SCOPE.get()
    if scope is None or not scope.reserve_transition(1):
        return None
    return _transition_record(scope, owner, boundary, target=target, **fields)


@contextmanager
def _causal_transition_span(owner, *, target=None, **fields):
    """Reserve both boundaries; observe the original call once, including aborts.

    Yielded metadata belongs only to telemetry, never to a runtime payload.
    An absent end after a process exit remains an incomplete interval.
    """
    scope = _CAUSAL_SCOPE.get()
    retained = scope is not None and scope.reserve_transition(2)
    start = (_transition_record(scope, owner, 'enter', target=target, **fields)
             if retained else None)
    token = _TRANSITION_SPAN.set(start)
    outcome = 'returned'
    metadata = {}
    try:
        yield metadata
    except BaseException as exc:
        outcome = 'raised' if isinstance(exc, Exception) else 'aborted'
        metadata['exception_type'] = type(exc).__name__
        raise
    finally:
        _TRANSITION_SPAN.reset(token)
        if retained:
            _transition_record(scope, owner, 'exit', target=target,
                               start_event_id=(start or {}).get('event_id'),
                               start_monotonic_ns=(start or {}).get('monotonic_ns'),
                               outcome=outcome, **{**fields, **metadata,
                                   'predecessor_result_id': (start or {}).get('predecessor_result_id')})



@contextmanager
def transition_span(owner, *, target=None, **fields):
    # Separate bounded observer survives both existing causal reservations.
    with state_flow_transition(owner, value=target):
        with _causal_transition_span(owner, target=target, **fields) as observation:
            yield observation


def observe_transition(owner, *, target, result=None, only_within=()):
    """Small owner-bound adapter; selectors must return scalar identity/status."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            scope = _CAUSAL_SCOPE.get()
            if scope is None or (only_within and (_TRANSITION_SPAN.get() or {}).get('owner') not in only_within):
                return function(*args, **kwargs)
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                identity = target(bound.arguments)
            except Exception:
                scope.delivery_failures += 1
                return function(*args, **kwargs)
            with transition_span(owner, target=identity) as observation:
                value = function(*args, **kwargs)
                if result:
                    try:
                        observation.update(result(value))
                    except Exception:
                        scope.delivery_failures += 1
                return value
        return wrapped
    return decorate


def observe_request(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if _CAUSAL_SCOPE.get() is not None:
            return function(self, *args, **kwargs)
        sink = getattr(self, 'log_unified_event', None)
        if sink is None:
            sink = self._hook('log_unified_event')
        response = kwargs.get('response_payload') or {}
        with state_flow_scope(), causal_scope(sink, response_id=response.get('id')):
            return function(self, *args, **kwargs)
    return wrapped


def observe_call(owner, *, inputs=None, target=None, result=None,
                 record_kind='runtime_invocation', complete=False,
                 authority=None, evidence=None, required_rule=None, context=None):
    """Observe a real owner call. Selectors are deliberately supplied by owners.

    Read-model builders must name their kind; their results are not model calls,
    input identity is not output identity, and a digest alone proves no redundancy.
    """
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            scope = _CAUSAL_SCOPE.get()
            reserved = (scope is not None and owner == 'response_frame.finalize'
                        and scope.reserved_count < CAUSAL_FINALIZER_RESERVE)
            if scope is None or (scope.normal_budget_exhausted() and not reserved):
                if scope is not None:
                    scope.dropped += 1
                return function(*args, **kwargs)
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                selected = inputs(values) if inputs else None
                identities = {k: relevant_digest(v) for k, v in selected.items()} if selected is not None else {}
                local_target = target(values) if target else dict((_CAUSAL_CALL.get() or {}).get('target') or {})
                scope.bind(local_target.get('response_id'))
                key = (owner, relevant_digest(local_target))
                with scope.lock:
                    previous = scope.previous.get(key)
                changed = [k for k in sorted(set(identities) | set((previous or {}).get('inputs', {})))
                           if identities.get(k) != (previous or {}).get('inputs', {}).get(k)] if previous else None
                call = causal_event(owner, record_kind, target=local_target, status='started',
                    invocation_id='invocation-' + uuid.uuid4().hex,
                    effective_input_hash=relevant_digest(identities) if selected is not None else None,
                    input_field_hashes=identities, input_coverage_complete=complete,
                    changed_relevant_inputs=changed,
                    previous_invocation_id=(previous or {}).get('invocation_id'),
                    trigger_reason='owner_call_entered', relevance_cause='unknown',
                    relevant_change_proven=bool(complete and previous and changed),
                    authority_hash=relevant_digest(authority(values)) if authority else None,
                    evidence_hash=relevant_digest(evidence(values)) if evidence else None,
                    required_defensive_contract=required_rule,
                    semantic_context=(context(values) if context else
                                      (_CAUSAL_CALL.get() or {}).get('semantic_context')),
                    start_monotonic_ns=time.monotonic_ns())
            except Exception:
                scope.delivery_failures += 1
                return function(*args, **kwargs)
            if call is None:
                return function(*args, **kwargs)
            token = _CAUSAL_CALL.set(call)
            status = 'returned'
            judgment = None
            try:
                value = function(*args, **kwargs)
                if result:
                    try:
                        judgment = result(value)
                    except Exception:
                        scope.delivery_failures += 1
                return value
            except BaseException:
                status = 'raised'
                raise
            finally:
                _CAUSAL_CALL.reset(token)
                try:
                    ended = dict(call, event_id='event-' + uuid.uuid4().hex,
                                 start_event_id=call['event_id'], status=status,
                                 end_monotonic_ns=time.monotonic_ns(), judgment=judgment,
                                 result_id=call['invocation_id'] + ':result',
                                 judgment_hash=relevant_digest(judgment) if judgment is not None else None,
                                 runtime_state_delta=None, downstream_effect=None,
                                 stale_result_disposition='unknown')
                    scope.retain(ended)
                    with scope.lock:
                        scope.previous[key] = dict(inputs=identities, invocation_id=call['invocation_id'])
                except Exception:
                    scope.delivery_failures += 1
        return state_flow_observed_call(owner, wrapped)
    return decorate


@contextmanager
def measure_operation(name, *, role):
    """Aggregate existing synchronous operations on the enclosing invocation.

    Timers are inclusive and may nest. No additional flush, write or work is done.
    """
    call = _CAUSAL_CALL.get()
    if call is None and _STATE_FLOW_SCOPE.get() is None:
        yield
        return
    started = time.monotonic_ns()
    effective_role = _OPERATION_ROLE.get() if role == 'caller_durability' else role
    effective_role = effective_role or role
    operation_key = f'{name}@{effective_role}' if role == 'caller_durability' else name
    role_token = _OPERATION_ROLE.set(effective_role)
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            state_flow_operation(operation_key, time.monotonic_ns() - started, failed)
            stages = call.setdefault('operations', {}) if call is not None else {}
            stage = stages.setdefault(operation_key, dict(role=effective_role, calls=0, elapsed_ns=0,
                                                failures=0, inclusive=True))
            stage['calls'] += 1
            stage['elapsed_ns'] += time.monotonic_ns() - started
            stage['failures'] += int(failed)
        except Exception:
            pass
        finally:
            _OPERATION_ROLE.reset(role_token)


def timed_operation(name, *, role):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with measure_operation(name, role=role):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _truncate(value: Any, max_chars: int = 4000) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + '...[truncated]'
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_truncate(item, max_chars=max_chars) for item in value[:50]]
    if isinstance(value, dict):
        return {str(k): _truncate(v, max_chars=max_chars) for k, v in list(value.items())[:50]}
    return _truncate(str(value), max_chars=max_chars)


def make_event(
    *,
    category: str,
    action: str,
    status: str,
    message: Optional[str] = None,
    **fields: Any,
) -> dict:
    payload = {
        'id': f"event-{uuid.uuid4().hex}",
        'timestamp': _now_iso(),
        'category': str(category).strip(),
        'action': str(action).strip(),
        'status': str(status).strip(),
    }
    if message:
        payload['message'] = _truncate(message, max_chars=1200)
    for key, value in fields.items():
        if value is None or value == '':
            continue
        payload[str(key)] = _truncate(value)
    return payload


def append_event(entry: dict, *, path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_EVENT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    with _EVENT_LOG_LOCK:
        with target.open('a', encoding='utf-8') as handle:
            handle.write(line)


def log_event(
    *,
    category: str,
    action: str,
    status: str,
    path: Path | str | None = None,
    message: Optional[str] = None,
    **fields: Any,
) -> dict:
    entry = make_event(
        category=category,
        action=action,
        status=status,
        message=message,
        **fields,
    )
    append_event(entry, path=path)
    return entry


def read_events(
    *,
    path: Path | str | None = None,
    limit: int = 200,
    category: Optional[str] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict]:
    target = Path(path) if path else DEFAULT_EVENT_LOG_PATH
    if limit <= 0 or not target.exists():
        return []
    lines = target.read_text(encoding='utf-8').splitlines()
    entries: list[dict] = []
    for raw_line in reversed(lines):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if category and str(item.get('category') or '') != category:
            continue
        if action and str(item.get('action') or '') != action:
            continue
        if status and str(item.get('status') or '') != status:
            continue
        entries.append(item)
        if len(entries) >= limit:
            break
    return entries
