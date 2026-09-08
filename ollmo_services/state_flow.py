"""Opt-in, bounded owner-transition diagnostics. Never runtime authority.

No payload serialization, content walks, extra hydration or execution decisions.
Byte counters are fed only by existing operations. Missing measurements are None.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

STATE_FLOW_RECORD_LIMIT = 2048
STATE_FLOW_BYTE_LIMIT = 8 * 1024 * 1024
STATE_FLOW_IDENTITY_LIMIT = 2048
STATE_FLOW_OPERATION_LIMIT = 128
_SCOPE = ContextVar('ollmo_state_flow_scope', default=None)
_SPAN = ContextVar('ollmo_state_flow_span', default=None)
_BOOT = f'{os.getpid()}:{uuid.uuid4().hex}'
_TOKEN = re.compile(r'[A-Za-z0-9_.:@/+=-]{1,192}\Z')
_DIGEST = re.compile(r'[0-9a-f]{16,64}\Z')


def _token(value):
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def _summary(value):
    """Shallow, allowlisted metadata only; does not walk content/recursive children."""
    result = {'top_level_items': len(value) if isinstance(value, (dict, list, tuple)) else None,
              'serialized_bytes': len(value) if isinstance(value, bytes) else None}
    if not isinstance(value, Mapping):
        return result
    for key in ('response_id', 'frame_id', 'branch_id', 'phase_id', 'obligation_id',
                'task_id', 'lifecycle_state', 'status', 'kind'):
        result[key] = _token(value.get(key))
    for key in ('frame_sequence', 'sequence', 'size_bytes'):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            result[key] = item
    for key in ('source_digest', 'response_map_digest', 'sha256', 'index_sha256', 'ledger_sha256'):
        item = value.get(key)
        if isinstance(item, str) and _DIGEST.fullmatch(item):
            result[key] = item
    for key in ('outputs', 'artifacts', 'output_obligations', 'intent_obligations',
                'candidates', 'decisions', 'branches', 'phases', 'items', 'responses'):
        item = value.get(key)
        if isinstance(item, (dict, list, tuple)):
            result[key + '_count'] = len(item)
    for key in ('ok', 'index_used', 'index_stale', 'ledger_fallback_used'):
        if isinstance(value.get(key), bool):
            result[key] = value[key]
    epoch = value.get('source_epoch')
    if isinstance(epoch, Mapping):
        result['epoch_id'] = _token(epoch.get('epoch_id'))
    frame = value.get('response_frame')
    if isinstance(frame, Mapping):
        result['frame_id'] = _token(frame.get('frame_id'))
        result['frame_sequence'] = frame.get('frame_sequence') if isinstance(frame.get('frame_sequence'), int) else None
        result['response_id'] = _token(frame.get('response_id')) or result.get('response_id')
    if not result.get('response_id'):
        ident = value.get('id')
        if isinstance(ident, str) and ('lifecycle_state' in value or 'response_frame' in value or ident.startswith('resp_')):
            result['response_id'] = _token(ident)
    runtime = value.get('runtime')
    closure = runtime.get('graph_closure_review') if isinstance(runtime, Mapping) else None
    if isinstance(closure, Mapping):
        result['closure_status'] = _token(closure.get('status'))
    late = value.get('late_fill')
    if isinstance(late, Mapping):
        result['late_fill_status'] = _token(late.get('status'))
        for key in ('pending_branches', 'active_branches', 'completed_branches', 'failed_branches'):
            items = late.get(key)
            result[key + '_count'] = len(items) if isinstance(items, list) else None
    return result


class _Scope:
    def __init__(self, directory):
        self.scope_id = 'state-flow-' + uuid.uuid4().hex
        self.path = Path(directory) / (self.scope_id + '.jsonl')
        self.response_id = None
        self.lock = threading.RLock()
        self.records = self.bytes = self.dropped = self.failures = 0
        self.counters = {}
        self.identities = {}
        self.identity_drops = 0
        self.started = time.monotonic_ns()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record, *, summary=False):
        try:
            with self.lock:
                record = dict(record, schema='ollmo.state_flow.v1', authority='observer_only',
                              scope_id=self.scope_id, response_id=record.get('response_id') or self.response_id,
                              process_boot_id=_BOOT, process_id=os.getpid())
                raw = (json.dumps(record, separators=(',', ':'), ensure_ascii=True) + '\n').encode()
                # Leave a named reserve for coverage summaries after exhaustion.
                limit = STATE_FLOW_BYTE_LIMIT if summary else STATE_FLOW_BYTE_LIMIT - 256 * 1024
                if (not summary and self.records >= STATE_FLOW_RECORD_LIMIT) or self.bytes + len(raw) > limit:
                    self.dropped += 1
                    return
                with self.path.open('ab') as handle:
                    handle.write(raw)
                self.records += int(not summary)
                self.bytes += len(raw)
        except Exception:
            self.failures += 1

    def checkpoint(self, boundary):
        with self.lock:
            self.emit(dict(record_kind='scope_summary', boundary=boundary,
                           monotonic_ns=time.monotonic_ns(), recorded_count=self.records,
                           dropped_count=self.dropped, delivery_failures=self.failures,
                           identity_drops=self.identity_drops, counters=dict(self.counters),
                           identities={k: sorted(v) for k, v in self.identities.items()},
                           record_limit=STATE_FLOW_RECORD_LIMIT, byte_limit=STATE_FLOW_BYTE_LIMIT,
                           identity_limit=STATE_FLOW_IDENTITY_LIMIT,
                           byte_coverage='selected_existing_operations_only',
                           interval_complete=False), summary=True)


@contextmanager
def state_flow_scope():
    if _SCOPE.get() is not None:
        yield
        return
    directory = os.environ.get('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR')
    if not directory:
        yield
        return
    try:
        scope = _Scope(directory)
    except Exception:
        yield
        return
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        scope.checkpoint('request_scope_returned_workers_may_continue')
        _SCOPE.reset(token)


def note(*, identity_kind=None, identity=None, **counts):
    """Count actual existing work, once, on its nearest transition and scope."""
    scope, span = _SCOPE.get(), _SPAN.get()
    if scope is None or span is None:
        return
    try:
        with scope.lock:
            for key, value in counts.items():
                if isinstance(value, int) and value >= 0:
                    span['work'][key] = span['work'].get(key, 0) + value
                    scope.counters[key] = scope.counters.get(key, 0) + value
            if identity_kind and isinstance(identity, str) and _DIGEST.fullmatch(identity):
                ids = scope.identities.setdefault(identity_kind, set())
                if identity not in ids and len(ids) >= STATE_FLOW_IDENTITY_LIMIT:
                    scope.identity_drops += 1
                else:
                    ids.add(identity)
                local = span['_identities'].setdefault(identity_kind, set())
                if len(local) < STATE_FLOW_IDENTITY_LIMIT:
                    local.add(identity)
                else:
                    span['identity_coverage_incomplete'] = True
    except Exception:
        scope.failures += 1


def operation(name, elapsed_ns, failed=False):
    scope, span = _SCOPE.get(), _SPAN.get()
    if scope is None or span is None:
        return
    try:
        with scope.lock:
            ops = span['operations']
            if name not in ops and len(ops) >= STATE_FLOW_OPERATION_LIMIT:
                span['operation_drops'] += 1
                return
            item = ops.setdefault(name, dict(calls=0, inclusive_ns=0, failures=0))
            item['calls'] += 1
            item['inclusive_ns'] += elapsed_ns
            item['failures'] += int(failed)
    except Exception:
        scope.failures += 1


def _primary(values):
    for key in ('response_payload', 'framed_payload', 'response_frame', 'frame',
                'enriched_frame', 'current_payload', 'payload', 'request_payload',
                'data', 'plan', 'observation', 'projection', 'prior_state', 'artifact_gap', 'value'):
        if isinstance(values.get(key), (Mapping, list, tuple)):
            return values[key]
    return None


@contextmanager
def transition(owner, source='UNKNOWN', target='UNKNOWN', *, value=None, bindings=None,
               labels=(), **flags):
    scope = _SCOPE.get()
    if scope is None:
        yield None
        return
    parent = _SPAN.get()
    try:
        source_summary = _summary(value)
        if bindings:
            source_summary.update({k: v for k, v in bindings.items() if v is not None})
        if source_summary.get('response_id') and scope.response_id is None:
            scope.response_id = source_summary['response_id']
        span = dict(record_kind='transition', transition_id='transition-' + uuid.uuid4().hex,
                    parent_transition_id=parent['transition_id'] if parent else None,
                    thread_id=threading.get_ident(), owner=owner, source_representation=source,
                    target_representation=target, source=source_summary, target=None,
                    causal_labels=list(labels), new_evidence=None, new_authority_boundary=None,
                    full_reconstruction=None, full_hydration=None, full_normalization=None,
                    full_compaction=None, disk_write=None, cache_reuse=None,
                    logical_payload_bytes=None, bytes_read=None, bytes_serialized=None,
                    bytes_hashed=None, canonical_source_identity=None,
                    work={}, operations={}, operation_drops=0, _identities={})
        span.update(flags)
        span['start_monotonic_ns'] = time.monotonic_ns()
        token = _SPAN.set(span)
    except Exception:
        scope.failures += 1
        yield None
        return
    outcome = 'returned'
    try:
        yield span
    except BaseException:
        outcome = 'raised'
        raise
    finally:
        ended = time.monotonic_ns()
        _SPAN.reset(token)
        try:
            record = {k: v for k, v in span.items() if not k.startswith('_')}
            for name, suffix in (('bytes_read', '_read_bytes'), ('bytes_serialized', '_serialized_bytes'), ('bytes_hashed', '_hashed_bytes')):
                measured = [v for k, v in span['work'].items() if k.endswith(suffix)]
                if measured:
                    record[name] = sum(measured)
            record['byte_coverage'] = 'selected_existing_operations_only'
            mappings = span['_identities'].get('mapping_digests', set())
            record['mapping_digest'] = next(iter(mappings)) if len(mappings) == 1 else None
            record['disk_write'] = True if any(v for k, v in span['work'].items() if k.endswith('_writes') or k == 'ledger_appends') else record['disk_write']
            record['cache_reuse'] = True if span['work'].get('private_observation_reuses') else record['cache_reuse']
            record.update(end_monotonic_ns=ended, inclusive_ns=ended-span['start_monotonic_ns'],
                          outcome=outcome, response_id=scope.response_id,
                          unique_identity_counts={k: len(v) for k, v in span['_identities'].items()})
            scope.emit(record)
            if owner in ('late_fill.complete_response_late_fill', 'responses_request.handle_responses_request'):
                scope.checkpoint(owner + '.returned')
        except Exception:
            scope.failures += 1


def observe_state(owner, source='UNKNOWN', target='UNKNOWN', *, labels=(), **flags):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _SCOPE.get() is None:
                return function(*args, **kwargs)
            try:
                values = signature.bind(*args, **kwargs).arguments
                value = _primary(values)
                bindings = {'response_id': _token(values.get('response_id'))}
                caller = inspect.currentframe().f_back
                caller_name = caller.f_code.co_name
                caller_line = caller.f_lineno
                del caller
            except Exception:
                _SCOPE.get().failures += 1
                return function(*args, **kwargs)
            with transition(owner, source, target, value=value, bindings=bindings, labels=labels, **flags) as span:
                if span is not None:
                    span['caller'] = caller_name
                    span['caller_line'] = caller_line
                result = function(*args, **kwargs)
                if span is not None:
                    try:
                        effective = result
                        if isinstance(result, tuple):
                            effective = next((v for v in result if isinstance(v, Mapping)), None)
                        span['target'] = _summary(effective)
                        span['source_unchanged_identity_proves_check_unnecessary'] = False
                        if span['target'].get('response_id') and _SCOPE.get().response_id is None:
                            _SCOPE.get().response_id = span['target']['response_id']
                        if owner == 'response_frame.finalize':
                            before = span['source']
                            late = before.get('late_fill_status')
                            span['lifecycle_role'] = ('pre-Late-Fill' if late in ('pending', 'running')
                                else 'terminal' if late else 'successor' if before.get('frame_id') else 'initial')
                            span['persist_requested'] = values.get('persist', True)
                            span['frame_created'] = bool(span['target'].get('frame_id'))
                    except Exception:
                        _SCOPE.get().failures += 1
                return result
        return wrapped
    return decorate


# Existing observed calls: high-level transitions only; frequent advisory read
# models remain aggregated operations and do not flood the reserved carrier.
_CALLS = {
    'responses_request.handle_responses_request': ('parsed_request', 'public_response_projection'),
    'ghost_route.resolve_ghost_auto_route': ('parsed_request', 'ghost_intent_decision_phase_graph'),
    'candidate_contracts.review_candidate_promotions': ('candidate_state', 'promoted_obligations'),
    'response_semantics.build_global_semantic_closure_review': ('graph_artifact_branch_state', 'semantic_closure_review'),
    'response_semantics.build_graph_closure_review': ('graph_artifact_branch_state', 'graph_closure_review'),
    'response_frame.finalize': ('live_response_record', 'finalized_response_frame'),
    'late_fill.complete_response_late_fill': ('live_response_record', 'terminal_response_record'),
    'late_fill.execute_prepared_late_fill_branch': ('accepted_branch_plan', 'branch_result_state'),
    'multi_materialization.execute_materialization_branches': ('branch_plans', 'branch_results'),
    'backend_transport.execute_chat_backend_request': ('accepted_provider_input', 'provider_result'),
}


def observed_call(owner, function):
    if owner in _CALLS:
        labels = ('NEW_EVIDENCE',) if owner == 'late_fill.execute_prepared_late_fill_branch' else ('NEW_AUTHORITY_BOUNDARY',) if 'closure_review' in owner else ('NEW_REPRESENTATION',)
        return observe_state(owner, *_CALLS[owner], labels=labels)(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _SCOPE.get() is None:
            return function(*args, **kwargs)
        started = time.monotonic_ns()
        failed = False
        try:
            return function(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            operation(owner, time.monotonic_ns() - started, failed)
    return wrapped


def observe_mapping_digest(function):
    """Record the digest already produced by a map owner; never hash for telemetry."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        note(identity_kind='mapping_digests', identity=result)
        return result
    return wrapped
