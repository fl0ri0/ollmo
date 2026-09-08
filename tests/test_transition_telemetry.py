"""Bounded handoff witnesses; fake execution and temporary persistence only."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

import pytest

from ollmo_services import events, response_frames
from ollmo_server.multi_materialization_runtime import MultiMaterializationRuntimeOwner
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner


def collector():
    records = []
    return records, lambda **entry: records.append(deepcopy(entry['causal_event']))


def boundaries(records, owner=None):
    return [r for r in records if r.get('record_kind') == 'transition_boundary'
            and (owner is None or r['owner'] == owner)]


def assert_pairs(records):
    starts = {r['event_id']: r for r in boundaries(records) if r['boundary'] == 'enter'}
    ends = [r for r in boundaries(records) if r['boundary'] == 'exit']
    assert len(starts) == len(ends)
    for end in ends:
        start = starts[end['start_event_id']]
        assert end['start_monotonic_ns'] == start['monotonic_ns'] <= end['monotonic_ns']
        for field in ('scope_id', 'process_id', 'process_boot_id', 'target', 'handoff_attempt_id'):
            assert end.get(field) == start.get(field)


@pytest.mark.parametrize('failure', [None, ValueError, KeyboardInterrupt])
def test_pair_reservation_survives_normal_exhaustion_and_marks_abort(failure):
    records, sink = collector()
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), patch.object(events, 'CAUSAL_TRANSITION_RESERVE', 2):
        with events.causal_scope(sink, response_id='r') as scope:
            def work():
                with events.transition_span('test.owner'):
                    events.transition_marker('test.nested', 'omitted')
                    if failure:
                        raise failure('original')
                    return 7
            if failure:
                with pytest.raises(failure, match='original'):
                    work()
            else:
                assert work() == 7
            assert scope.snapshot()['transition_reserved_count'] == 2
            assert scope.snapshot()['transition_dropped_count'] == 1
    assert_pairs(records)
    end = boundaries(records)[-1]
    assert end['outcome'] == ('returned' if failure is None else 'raised' if failure is ValueError else 'aborted')
    assert end['complete_start_authority'] is None
    assert records[-1]['record_kind'] == 'coverage_gap'


def test_concurrent_reservation_is_bounded_and_preserves_finalizer_reserve():
    records, sink = collector()
    @events.observe_call('response_frame.finalize')
    def finalize():
        return 'unchanged'
    def work():
        with events.transition_span('parallel'):
            return 1
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), patch.object(events, 'CAUSAL_TRANSITION_RESERVE', 8):
        with events.causal_scope(sink, response_id='r') as scope:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(events.traced_thread_target(work)) for _ in range(20)]
                assert [f.result() for f in futures] == [1] * 20
            assert finalize() == 'unchanged'
            assert scope.snapshot()['transition_recorded_count'] == 8
            assert scope.snapshot()['transition_dropped_count'] == 32
    assert_pairs(records)
    assert len([r for r in records if r.get('owner') == 'response_frame.finalize']) == 2


class Gate:
    branch_id = staticmethod(lambda b: b.get('branch_id', ''))
    branch_capability = staticmethod(lambda b: b.get('capability', ''))
    parse_bool = staticmethod(lambda v, default=False: bool(v) if v is not None else default)
    _causal_execution_gate_inputs = LateFillRuntimeOwner._causal_execution_gate_inputs
    decide = LateFillRuntimeOwner.semantic_execution_gate_decision

    def __init__(self):
        self.reads = 0

    def late_fill_branch_control_records(self, payload):
        self.reads += 1
        return payload['controls']


def test_real_gate_runs_once_and_never_claims_aggregate_start_authority():
    gate = Gate()
    branch = {'branch_id': 'b', 'phase_id': 'p', 'attempt_count': 2}
    records, sink = collector()
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
        assert gate.decide(branch, {'controls': {}})['action'] == 'execute'
        binding = events.transition_attempt(branch)
        assert binding['start_check_attempt_binding'] == 'matching_recorded_attempt'
        assert binding['last_target_start_check']['gate_action'] == 'execute'
        assert events.transition_attempt(dict(branch, attempt_count=3))['start_check_attempt_binding'] == 'unknown'
        assert gate.decide(branch, {'controls': {'b': {'status': 'cancelled'}}})['action'] == 'skip'
    assert gate.reads == 2  # No instrumenter re-evaluates the check or its input selector.
    ends = [r for r in boundaries(records) if r['boundary'] == 'exit']
    assert [r['gate_action'] for r in ends] == ['execute', 'skip']
    assert all(r['gate_check_complete'] and r['complete_start_authority'] is None for r in ends)


def test_multiple_branch_attempts_results_errors_and_callbacks_do_not_change():
    def run(observed):
        records, sink = collector()
        calls = []
        def prepare(**kwargs):
            calls.append('prepare')
            return {'route_info': {'instance_id': 'fake'}}
        def execute(plan):
            calls.append(('execute', plan['branch_id']))
            if plan['branch_id'] == 'b2':
                raise ValueError('fake failure')
            return {'saved': 'unchanged'}
        def callback(event):
            calls.append(('callback', event['branch_id']))
            if event['branch_id'] == 'b2':
                raise RuntimeError('callback failure remains swallowed')
        results = []
        def work():
            for attempt in (1, 2):
                branches = [dict(branch_id='b'+str(i), phase_id='p'+str(i),
                                 capability='chat', attempt_count=attempt) for i in (1, 2)]
                result = MultiMaterializationRuntimeOwner().execute_materialization_branches(
                    branches, prepare_branch_plan=prepare, execute_prepared_branch=execute,
                    on_branch_progress=callback, max_workers=1)
                results.append((result['branch_results'], result['branch_errors'], result['prepared_branch_plans']))
        if observed:
            with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
                work()
        else:
            work()
        return results, calls, records
    plain, calls, _ = run(False)
    observed, observed_calls, records = run(True)
    assert observed == plain and observed_calls == calls
    assert_pairs(records)
    enqueued = boundaries(records, 'multi_materialization.queue')
    assert len({r['handoff_attempt_id'] for r in enqueued}) == 4
    assert {(r['target']['branch_id'], r['canonical_attempt']['attempt_count']) for r in enqueued} == {
        ('b1', 1), ('b1', 2), ('b2', 1), ('b2', 2)}
    available = boundaries(records, 'multi_materialization.result')
    assert len(available) == 2
    for item in available:
        callback = next(r for r in boundaries(records, 'multi_materialization.callback')
                        if r['boundary'] == 'enter' and r['handoff_attempt_id'] == item['handoff_attempt_id'])
        assert callback['predecessor_result_id'] == item['result_id']
        assert callback['monotonic_ns'] >= item['monotonic_ns']
    assert len([r for r in boundaries(records, 'multi_materialization.execution') if r.get('outcome') == 'raised']) == 2
    assert len([r for r in boundaries(records, 'multi_materialization.callback') if r.get('outcome') == 'raised']) == 2


def test_async_callback_order_and_drain_wait_are_preserved():
    callback_entered, release, second_executed, returned = [threading.Event() for _ in range(4)]
    records, sink = collector()
    order = []
    errors = []
    def execute(plan):
        if plan['branch_id'] == 'b2':
            second_executed.set()
        return {'ok': True}
    def callback(event):
        order.append(event['branch_id'])
        if event['branch_id'] == 'b1':
            callback_entered.set()
            assert release.wait(3)
    def run():
        try:
            with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
                MultiMaterializationRuntimeOwner().execute_materialization_branches(
                    [dict(branch_id=b, phase_id=b+'p', capability='chat') for b in ('b1', 'b2')],
                    prepare_branch_plan=lambda **kw: {}, execute_prepared_branch=execute,
                    on_branch_progress=callback, async_branch_progress=True, max_workers=1)
            returned.set()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert callback_entered.wait(3) and second_executed.wait(3)
        assert not returned.is_set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors and returned.is_set()
    assert order == ['b1', 'b2']
    assert_pairs(records)
    drain = boundaries(records, 'multi_materialization.callback_drain')
    last_callback = max(r['monotonic_ns'] for r in boundaries(records, 'multi_materialization.callback') if r['boundary'] == 'exit')
    assert drain[-1]['monotonic_ns'] >= last_callback
    assert all(r['target']['response_id'] == 'r' for r in boundaries(records))


def test_lookup_hydration_is_observed_only_when_it_really_runs(tmp_path):
    frame = response_frames.build_response_frame({'id': 'r', 'status': 'completed', 'output_text': 'saved'})
    response_frames.persist_response_frame(frame, frames_dir=tmp_path)
    baseline = response_frames.load_latest_response_state('r', frames_dir=tmp_path)
    records, sink = collector()
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
        with events.transition_binding(events.transition_attempt({'branch_id': 'b', 'phase_id': 'p'})):
            with events.transition_span('late_fill.callback.lookup'):
                observed = response_frames.load_latest_response_state('r', frames_dir=tmp_path)
    assert observed == baseline
    assert_pairs(records)
    assert len(boundaries(records, 'response_frame.callback_payload_hydration')) == 2
    assert len(boundaries(records, 'response_frame.callback_manifest_expansion')) == 2
    records.clear()
    with events.causal_scope(sink, response_id='r'):
        response_frames.load_latest_response_state('r', frames_dir=tmp_path)
    assert not boundaries(records)  # No new observation traffic outside the selected handoff.


def test_sink_or_selector_failure_never_repeats_work():
    calls = []
    @events.observe_transition('fake', target=lambda args: {}, result=lambda value: 1 / 0)
    def work():
        calls.append('once')
        return object_result
    object_result = object()
    with events.causal_scope(lambda **entry: (_ for _ in ()).throw(OSError('sink failed')), response_id='r') as scope:
        assert work() is object_result
        assert scope.delivery_failures >= 2
    assert calls == ['once']


def test_transition_reserve_does_not_consume_normal_budget_and_uses_existing_persistence(tmp_path):
    records, sink = collector()
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 2), events.causal_scope(sink, response_id='r') as scope:
        with events.transition_span('test'):
            pass
        events.causal_event('normal', 'observation', target={'response_id': 'r'})
        events.causal_event('normal', 'observation', target={'response_id': 'r'})
        snapshot = scope.snapshot()
        assert snapshot['recorded_count'] == 4 and snapshot['dropped_count'] == 0
    frame = response_frames.build_response_frame({
        'id': 'r', 'status': 'completed', 'output_text': 'saved',
        'runtime': {'developer_diagnostics': {'causal_telemetry': snapshot}}})
    response_frames.persist_response_frame(frame, frames_dir=tmp_path)
    loaded = response_frames.load_latest_response_state('r', frames_dir=tmp_path)
    saved = loaded['response_payload']['runtime']['developer_diagnostics']['causal_telemetry']
    assert saved['transition_recorded_count'] == snapshot['transition_recorded_count']
    for before, after in zip(snapshot['events'], saved['events'], strict=True):
        for key in ('event_id', 'scope_id', 'owner', 'boundary', 'monotonic_ns',
                    'start_event_id', 'start_monotonic_ns', 'process_boot_id', 'target',
                    'outcome', 'canonical_attempt_status', 'predecessor_result_status',
                    'start_authority_status', 'requires_exit'):
            assert before.get(key) == after.get(key)
    assert boundaries(saved['events'])[0]['start_authority_status'] == 'unknown'
    assert_pairs(saved['events'])


def test_worker_submission_context_reaches_existing_target_without_extra_start():
    from ollmo_server import late_fill_runtime
    records, sink = collector()
    calls = []
    @events.observe_transition('late_fill.worker', target=lambda a: {'response_id': a['response_payload']['id']})
    def worker(response_payload, **kwargs):
        calls.append(response_payload)
    class Scheduler:
        claim_response_late_fill = staticmethod(lambda rid: True)
        schedule = LateFillRuntimeOwner.schedule_response_late_fill
    class FakeThread:
        def __init__(self, *, target, kwargs, daemon):
            self.target, self.kwargs = target, kwargs
        def start(self):
            self.target(**self.kwargs)
    with patch.object(late_fill_runtime.threading, 'Thread', FakeThread), \
            patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
        assert Scheduler().schedule(response_payload={'id': 'r'}, request_payload={},
                                    assistant_message='', artifact_gap={}, source_route_payload=None,
                                    complete_response_late_fill=worker)
    assert calls == [{'id': 'r'}]
    assert_pairs(records)
    assert len({r['handoff_attempt_id'] for r in boundaries(records)
                if r['owner'] in {'late_fill.worker_submission', 'late_fill.worker'}}) == 1
    submission = boundaries(records, 'late_fill.worker_submission')[0]
    entry = boundaries(records, 'late_fill.worker')[0]
    assert submission['monotonic_ns'] <= entry['monotonic_ns']


def test_preparation_error_and_nested_attempt_identity_are_observation_only():
    records, sink = collector()
    calls = []
    def fail(**kwargs):
        calls.append('prepare')
        raise ValueError('preparation rejected')
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
        binding = events.transition_attempt({'branch_id': 'b', 'phase_id': 'p', 'branch': {
            'branch': {'branch_id': 'b', 'phase_id': 'p', 'auto_executable_repair_retry_count': 2}}})
        assert binding['canonical_attempt'] == {'auto_executable_repair_retry_count': 2}
        result = MultiMaterializationRuntimeOwner().execute_materialization_branches(
            [{'branch_id': 'b', 'phase_id': 'p', 'capability': 'chat'}],
            prepare_branch_plan=fail,
            execute_prepared_branch=lambda plan: pytest.fail('must remain unexecuted'))
    assert calls == ['prepare'] and result['branch_errors']['b']['stage'] == 'prepare_branch_plan'
    assert_pairs(records)
    assert boundaries(records, 'multi_materialization.prepare')[-1]['outcome'] == 'raised'
    assert not boundaries(records, 'multi_materialization.queue')


def test_failed_record_preparation_cannot_expand_normal_budget():
    records, sink = collector()
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 2), events.causal_scope(sink, response_id='r') as scope:
        with patch.object(events, '_truncate', side_effect=ValueError('observer serialization failed')):
            with events.transition_span('failed_observation'):
                pass
        for _ in range(5):
            events.causal_event('normal', 'observation')
        snapshot = scope.snapshot()
        assert snapshot['transition_reserved_count'] == 2
        assert snapshot['transition_recorded_count'] == 0
        assert snapshot['recorded_count'] == 2
        assert snapshot['delivery_failure_count'] == 2
        assert snapshot['dropped_count'] == 3


def test_existing_preparation_retry_gets_distinct_attempt_witnesses_only():
    records, sink = collector()
    calls, executions = [], []
    def prepare(branch_name, excluded_instance_ids):
        calls.append((branch_name, list(excluded_instance_ids)))
        if excluded_instance_ids:
            raise RuntimeError('no non-excluded instance')
        return {'route_info': {'instance_id': 'fake'}}
    def execute(plan):
        executions.append(plan['branch_id'])
        return {'saved': plan['branch_id']}
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 0), events.causal_scope(sink, response_id='r'):
        result = MultiMaterializationRuntimeOwner().execute_materialization_branches(
            [dict(branch_id=b, phase_id=b+'p', capability='chat', prepare_args={'branch_name': b})
             for b in ('b1', 'b2')], prepare_branch_plan=prepare, execute_prepared_branch=execute,
            max_workers=1)
    assert calls == [('b1', []), ('b2', ['fake']), ('b2', [])]
    assert executions == ['b1', 'b2'] and not result['branch_errors']
    ends = [r for r in boundaries(records, 'multi_materialization.prepare')
            if r['boundary'] == 'exit' and r['target']['branch_id'] == 'b2']
    assert [(r['prepare_attempt'], r['outcome']) for r in ends] == [(1, 'raised'), (2, 'returned')]
    assert_pairs(records)
