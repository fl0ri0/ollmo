"""Deterministic causal observations; no production roots or live backends."""
from copy import deepcopy
import hashlib
import json
import threading
from unittest.mock import patch

import pytest

from ollmo_services import events
from ollmo_g.decision_contracts import _semantic_review_lens_payload
from ollmo_services.semantic_review_verdict import semantic_review_verdict_freeze_acceptance
from ollmo_services import response_frames as frames
from ollmo_server.multi_materialization_runtime import MultiMaterializationRuntimeOwner
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_server.backend_transport_runtime import BackendTransportRuntimeOwner
from scripts.self_attack_convergence import analyze_causal_records, causal_records, extract_capture


def collector():
    records = []
    return records, lambda **entry: records.append(deepcopy(entry['causal_event']))


def completed(records, owner=None):
    return [r for r in records if r.get('status') == 'returned'
            and (owner is None or r['owner'] == owner)]


def test_lens_projection_is_not_an_invocation_and_real_rebuild_is_not_a_model():
    record = {'branch_id': 'b', 'phase_id': 'p', 'semantic_review_lens': 'doubt_challenger'}
    projected = _semantic_review_lens_payload(record)
    observed = extract_capture({'runtime': {'semantic_review_lens_review': projected}}, 'test', None)
    assert observed['invocations'] == []
    records, sink = collector()
    with events.causal_scope(sink, response_id='r'):
        assert _semantic_review_lens_payload(record) == projected
    analyzed = analyze_causal_records(records)
    assert len(analyzed['invocations']) == 1
    call = analyzed['invocations'][0]
    assert call['record_kind'] == 'read_model_invocation'
    assert call['model_execution'] is False
    assert call['target'] == {'branch_id': 'b', 'phase_id': 'p'}
    assert call['judgment']['lens'] == 'doubt_challenger'
    assert call['authority'] == 'observer_only'


def test_lens_effective_input_excludes_unrelated_projection_and_output_hash():
    records, sink = collector()
    source = {'branch_id': 'b', 'phase_id': 'p', 'semantic_review_lens': 'doubt_challenger'}
    with events.causal_scope(sink, response_id='r'):
        _semantic_review_lens_payload(source)
        _semantic_review_lens_payload(dict(source, ui_spinner=True, output_hash='changed', timestamp='later'))
        _semantic_review_lens_payload(dict(source, semantic_review_lens='quality_reviewer'))
    calls = completed(records)
    assert calls[0]['effective_input_hash'] == calls[1]['effective_input_hash']
    assert calls[1]['changed_relevant_inputs'] == []
    assert calls[2]['effective_input_hash'] != calls[1]['effective_input_hash']
    assert calls[2]['changed_relevant_inputs'] == ['source']


def test_defensive_rule_and_changed_inputs_are_proven_only_in_exact_scope():
    records, sink = collector()

    @events.observe_call('test.parent', target=lambda a: {'branch_id': a['branch_id'], 'phase_id': 'p'})
    def parent(branch_id):
        semantic_review_verdict_freeze_acceptance({}, required_criteria=['exact'])
        semantic_review_verdict_freeze_acceptance({}, required_criteria=['exact'])
        semantic_review_verdict_freeze_acceptance({}, required_criteria=['changed'])

    with events.causal_scope(sink, response_id='r'):
        parent('b')
    calls = analyze_causal_records(records)['invocations']
    checks = [c for c in calls if c['owner'] == 'semantic_review_verdict.freeze_acceptance']
    assert [c['classification'] for c in checks] == ['unknown', 'defensive_repeat', 'necessary_repeat']
    parent_call = next(c for c in calls if c['owner'] == 'test.parent')
    assert all(c['parent_invocation_id'] == parent_call['invocation_id'] for c in checks)
    assert all(c['trigger_event_id'] == parent_call['start_event_id'] for c in checks)
    assert checks[1]['target'] == {'branch_id': 'b', 'phase_id': 'p'}
    damaged = deepcopy(records)
    next(c for c in damaged if c.get('event_id') == checks[1]['event_id'])['target']['branch_id'] = 'sibling'
    assert not any(c['classification'] == 'defensive_repeat' for c in analyze_causal_records(damaged)['invocations'])
    assert not any(c['classification'] == 'redundant_repeat' for c in calls)


def test_lineage_and_monotonic_identity_survive_sha_compaction_and_successor(tmp_path):
    records, sink = collector()

    @events.observe_call('parent', target=lambda a: {'response_id': 'r', 'branch_id': 'b'})
    def parent():
        _semantic_review_lens_payload({'branch_id': 'b', 'semantic_review_lens': 'quality_reviewer'})

    with events.causal_scope(sink, response_id='r'):
        parent()
        telemetry = events.causal_snapshot('r')
    payload = {'id': 'r', 'status': 'completed', 'output_text': 'ok',
               'runtime': {'developer_diagnostics': {'causal_telemetry': telemetry},
                           'request_phase_graph': {'phase_graph_id': 'g', 'large_contract': 'x' * 100000}}}
    frame = frames.build_response_frame(payload, request_payload={'prompt': 'test'})
    frames.persist_response_frame(frame, frames_dir=tmp_path)
    first_ledger = (tmp_path / 'responses.jsonl').read_bytes()
    loaded = frames.load_latest_response_state('r', frames_dir=tmp_path)
    retained = causal_records(loaded['response_payload'])
    assert {r['event_id'] for r in retained} == {r['event_id'] for r in telemetry['events']}
    assert completed(retained)[0]['process_boot_id'] == completed(records)[0]['process_boot_id']
    assert completed(retained)[0]['start_monotonic_ns'] == completed(records)[0]['start_monotonic_ns']
    frames.persist_response_frame(frame, frames_dir=tmp_path)
    assert (tmp_path / 'responses.jsonl').read_bytes().startswith(first_ledger)
    loaded_again = frames.load_latest_response_state('r', frames_dir=tmp_path)
    assert len(analyze_causal_records(retained + causal_records(loaded_again['response_payload']))['invocations']) == 2
    snapshots = list((tmp_path / frames.DEFAULT_RESPONSE_FRAME_SNAPSHOT_DIR /
                      frames.DEFAULT_RESPONSE_FRAME_SNAPSHOT_CONTENT_DIR).rglob('*.json'))
    assert snapshots
    for path in snapshots:
        assert hashlib.sha256(path.read_bytes().rstrip(b'\n')).hexdigest() == path.stem


def test_same_instance_mutex_retains_actual_wait_and_handoff_without_changing_work():
    records, sink = collector()
    both_waiting = threading.Event()
    waiting_ids = set()
    execution_order = []

    def recording_sink(**entry):
        sink(**entry)
        event = entry['causal_event']
        if event.get('record_kind') == 'wait_started':
            waiting_ids.add(event['target'].get('branch_id'))
            if len(waiting_ids) == 2:
                both_waiting.set()

    def prepare(**kwargs):
        return {'route_info': {'instance_id': 'one'}}

    def execute(plan):
        execution_order.append(plan['branch_id'])
        if len(execution_order) == 1:
            assert both_waiting.wait(3)
        return {'branch_id': plan['branch_id'], 'value': 'saved'}

    branches = [{'branch_id': 'b1', 'phase_id': 'p1', 'capability': 'chat'},
                {'branch_id': 'b2', 'phase_id': 'p2', 'capability': 'chat'}]
    with events.causal_scope(recording_sink, response_id='r'):
        result = MultiMaterializationRuntimeOwner().execute_materialization_branches(
            branches, prepare_branch_plan=prepare, execute_prepared_branch=execute, max_workers=2)
    assert len(execution_order) == 2
    assert len(result['branch_results']) == 2
    waits = analyze_causal_records(records)['waits']
    second = next(w for w in waits if w['target'].get('branch_id') == execution_order[1])
    assert second['classification'] == 'necessary_wait'
    assert second['producer_target'] == {'branch_id': execution_order[0], 'phase_id': 'p' + execution_order[0][-1]}
    assert second['wake_event_id'] and second['wait_id']
    missing_producer = [r for r in records if r['event_id'] != second['wake_event_id']]
    assert all(w['classification'] == 'unknown' for w in analyze_causal_records(missing_producer)['waits'])
    branch_calls = completed(records, 'multi_materialization.execute_plan')
    assert len(branch_calls) == 2
    assert branch_calls[0]['parent_invocation_id'] == branch_calls[1]['parent_invocation_id']


def test_telemetry_failure_and_overflow_never_change_returns_or_exceptions():
    @events.observe_call('test', inputs=lambda a: {'value': a['value']})
    def identity(value):
        return value

    with events.causal_scope(lambda **entry: (_ for _ in ()).throw(OSError('disk full')), response_id='r') as scope:
        assert identity(3) == 3
        with patch.object(scope, 'retain', side_effect=TypeError('invalid observer metadata')):
            assert events.causal_event('test', 'bad_metadata') is None
        assert scope.delivery_failures > 0
    records, sink = collector()
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 3), events.causal_scope(sink, response_id='r') as scope:
        for index in range(8):
            assert identity(index) == index
        assert len(scope.snapshot()['events']) <= 3
        assert scope.snapshot()['dropped_count'] > 0
    assert records[-1]['record_kind'] == 'coverage_gap'

    @events.observe_call('raises')
    def fail():
        raise ValueError('original')

    with events.causal_scope(sink, response_id='r'), pytest.raises(ValueError, match='original'):
        fail()


def test_persistence_operations_preserve_bytes_and_original_failure(tmp_path):
    frame = frames.build_response_frame({'id': 'r', 'status': 'completed', 'output_text': 'ok'})
    records, sink = collector()

    @events.observe_call('persistence', target=lambda a: {'response_id': 'r'})
    def persist(root):
        return frames.persist_response_frame(frame, frames_dir=root)

    baseline, instrumented = tmp_path / 'baseline', tmp_path / 'instrumented'
    frames.persist_response_frame(frame, frames_dir=baseline)
    with events.causal_scope(sink, response_id='r'):
        persist(instrumented)
    assert (baseline / 'responses.jsonl').read_bytes() == (instrumented / 'responses.jsonl').read_bytes()
    call = completed(records, 'persistence')[0]
    assert {'ledger_serialization', 'ledger_append_flush_fsync', 'write_response_frame_index'} <= call['operations'].keys()
    assert call['operations']['write_response_frame_index']['role'] == 'derived_recovery_index'
    assert all(v['elapsed_ns'] >= 0 and v['inclusive'] for v in call['operations'].values())
    starts = [r for r in records if r.get('status') == 'started']
    assert all('operations' not in r for r in starts)


def test_historical_and_incomplete_records_never_invent_causality():
    assert causal_records({'runtime': {}}) == []
    assert analyze_causal_records([])['invocations'] == []
    assert analyze_causal_records([{'lens': 'quality', 'status': 'returned'}])['invocations'] == []
    records, sink = collector()
    with events.causal_scope(sink, response_id='r'):
        _semantic_review_lens_payload({'branch_id': 'b'})
        _semantic_review_lens_payload({'branch_id': 'b'})
    analysis = analyze_causal_records(records)
    assert all(c['classification'] == 'unknown' for c in analysis['invocations'])
    duplicate = analyze_causal_records(records + records)
    assert len(duplicate['invocations']) == 2


def test_transport_bound_does_not_silently_claim_complete_input_coverage():
    records, sink = collector()
    with events.causal_scope(sink, response_id='r'):
        events.causal_event('test', 'wait_started', target={'branch_id': 'b'},
                            input_coverage_complete=True, references=list(range(100)))
    assert records[0]['coverage_incomplete'] is True
    assert records[0]['input_coverage_complete'] is False
    assert len(json.dumps(records[0]).encode()) <= events.CAUSAL_RECORD_BYTES


def test_exact_gate_inputs_ignore_sibling_and_shadowed_control_changes():
    class Gate:
        branch_id = staticmethod(lambda b: b.get('branch_id', ''))
        branch_capability = staticmethod(lambda b: b.get('capability', ''))
        parse_bool = staticmethod(lambda v, default=False: bool(v) if v is not None else default)
        late_fill_branch_control_records = staticmethod(lambda p: p['controls'])
        _causal_execution_gate_inputs = LateFillRuntimeOwner._causal_execution_gate_inputs
        decide = LateFillRuntimeOwner.semantic_execution_gate_decision

    gate = Gate()
    records, sink = collector()
    branch = {'branch_id': 'b', 'phase_id': 'p', 'capability': 'chat'}
    with events.causal_scope(sink, response_id='r'):
        assert gate.decide(branch, {'controls': {}})['action'] == 'execute'
        assert gate.decide(branch, {'controls': {'sibling': {'status': 'cancelled'}}})['action'] == 'execute'
        assert gate.decide(branch, {'controls': {'b': {'status': 'cancelled', 'authority': 'runtime_contract'}}})['action'] == 'skip'
        cancelled = dict(branch, status='cancelled')
        gate.decide(cancelled, {'controls': {}})
        gate.decide(dict(cancelled, cancel_reason='ignored'), {'controls': {'b': {'status': 'waived'}}})
    calls = completed(records)
    assert calls[0]['effective_input_hash'] == calls[1]['effective_input_hash']
    assert calls[1]['effective_input_hash'] != calls[2]['effective_input_hash']
    assert calls[3]['effective_input_hash'] == calls[4]['effective_input_hash']
    assert all(c['target'] == {'branch_id': 'b', 'phase_id': 'p'} for c in calls)


def test_availability_wait_keeps_identity_and_does_not_consume_attempts():
    records, sink = collector()
    branch = {'branch_id': 'b', 'phase_id': 'p', 'attempt_count': 4}
    error = {'code': 'INSTANCE_UNAVAILABLE', 'stage': 'prepare_branch_plan',
             'route_diagnostics': {'availability_wait': {
                 'reason': 'live_candidates_in_cooldown', 'candidate_instance_ids': ['one']}}}
    with patch('ollmo_server.late_fill_runtime.time.time', return_value=100), events.causal_scope(sink, response_id='r'):
        waiting = LateFillRuntimeOwner.build_availability_wait_branch(branch, error=error)
        repeated = LateFillRuntimeOwner.build_availability_wait_branch(waiting, error=error)
    first, second = [r for r in records if r['record_kind'] == 'wait_started']
    assert waiting['attempt_count'] == repeated['attempt_count'] == 4
    assert waiting['availability_wait']['causal']['wait_id'] == first['event_id']
    assert second['previous_wait_id'] == first['event_id']
    assert first['target'] == {'branch_id': 'b', 'phase_id': 'p'}
    assert first['gates_complete'] is False


def test_actual_model_call_and_inherited_semantic_context_without_network():
    from unittest.mock import Mock
    response = Mock()
    response.json.return_value = {'choices': [{'message': {'content': 'answer'}}]}
    post = Mock(return_value=response)
    owner = BackendTransportRuntimeOwner(
        hooks={'chat_timeout_seconds': lambda *a: 30,
               'normalize_chat_messages_for_backend': lambda messages, **kw: messages,
               'requests_post': post}, capability_chat='chat',
        request_timeout_error=TimeoutError, request_connection_error=ConnectionError,
        request_exception_error=Exception)
    records, sink = collector()

    @events.observe_call('test.branch', target=lambda a: {'branch_id': 'b', 'phase_id': 'p'},
                         context=lambda a: {'selected_lens': 'evidence_verifier'})
    def execute():
        return owner.execute_chat_backend_request(target_port=1, model_name='fake',
            backend='llama_cpp', capability='chat', messages=[{'role': 'user', 'content': 'test'}])

    with events.causal_scope(sink, response_id='r'):
        assert execute() == 'answer'
    post.assert_called_once()
    model = next(c for c in analyze_causal_records(records)['invocations'] if c['model_execution'])
    assert model['semantic_context'] == {'selected_lens': 'evidence_verifier'}
    assert model['target'] == {'branch_id': 'b', 'phase_id': 'p'}


def test_reserve_keeps_finalization_timing_when_semantic_event_budget_is_exhausted():
    records, sink = collector()

    @events.observe_call('response_frame.finalize')
    def finalize():
        with events.measure_operation('durable_append', role='canonical_durability'):
            return 'unchanged'

    with patch.object(events, 'CAUSAL_RECORD_LIMIT', 1), events.causal_scope(sink, response_id='r'):
        events.causal_event('test', 'observation')
        assert finalize() == 'unchanged'
    assert completed(records)[0]['operations']['durable_append']['calls'] == 1


def test_analyzer_does_not_promote_malformed_or_truncated_identity():
    records, sink = collector()
    with events.causal_scope(sink, response_id='r'):
        events.causal_event('test', 'observation', target={'candidate_ids': ['x' * 2000] * 60})
    assert len(json.dumps(records[0]).encode()) <= events.CAUSAL_RECORD_BYTES
    assert records[0]['coverage_incomplete']
    assert analyze_causal_records([None, {}, records[0]])['invocations'] == []


def test_failed_observation_never_reexecutes_a_failing_owner():
    calls = []

    @events.observe_call('failing_owner')
    def execute():
        calls.append('executed')
        raise ValueError('original owner failure')

    with events.causal_scope(lambda **entry: None, response_id='r') as scope:
        with patch.object(scope, 'retain', side_effect=OSError('observation unavailable')):
            with pytest.raises(ValueError, match='original owner failure'):
                execute()
    assert calls == ['executed']
