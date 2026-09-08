"""Bounded diagnostics preserve owner results, bytes, exceptions and authority."""
import copy
import json
import threading
from unittest.mock import patch

import pytest

from ollmo_services import events, response_frames as rf, state_flow as sf


def records(tmp_path):
    return [json.loads(line) for p in tmp_path.glob('*.jsonl') for line in p.read_text().splitlines()]


def test_disabled_does_not_inspect_or_write(monkeypatch):
    monkeypatch.delenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', raising=False)
    sentinel = object()
    @sf.observe_state('test', 'a', 'b')
    def call(value): return value
    with patch.object(sf, '_summary', side_effect=AssertionError('must not inspect')):
        with sf.state_flow_scope():
            assert call(sentinel) is sentinel


def test_no_payloads_and_exact_result_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    value = {'id': 'resp_test', 'prompt': 'secret prompt', 'artifacts': [{'base64': 'secret bytes'}],
             'output_text': 'secret output', 'runtime': {'anything': ['private']}}
    original = copy.deepcopy(value)
    @sf.observe_state('test', 'request', 'response')
    def call(response_payload): return response_payload
    with sf.state_flow_scope(): assert call(value) is value
    assert value == original
    data = records(tmp_path)
    text = json.dumps(data)
    assert all(secret not in text for secret in ('secret prompt', 'secret bytes', 'secret output', 'private'))
    transition = next(r for r in data if r['record_kind'] == 'transition')
    assert transition['response_id'] == 'resp_test'
    assert transition['source']['artifacts_count'] == 1
    assert transition['logical_payload_bytes'] is None
    assert transition['inclusive_ns'] >= 0


def test_original_exception_survives_sink_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    error = ValueError('owner failure')
    @sf.observe_state('test')
    def call(): raise error
    with sf.state_flow_scope():
        with patch('pathlib.Path.open', side_effect=OSError('sink denied')):
            with pytest.raises(ValueError) as raised: call()
    assert raised.value is error
    assert records(tmp_path)[-1]['delivery_failures'] == 1


def test_initialization_failure_is_observation_only(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    with patch.object(sf, '_Scope', side_effect=OSError):
        with sf.state_flow_scope():
            assert sf._SCOPE.get() is None


def test_independent_of_causal_and_transition_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    @events.observe_call('response_frame.finalize')
    def call(response_payload):
        with events.measure_operation('work', role='canonical_truth'):
            sf.note(bytes_processed=42)
        return response_payload
    with sf.state_flow_scope(), events.causal_scope(lambda **k: None) as causal:
        causal.count = events.CAUSAL_RECORD_LIMIT + events.CAUSAL_FINALIZER_RESERVE
        causal.reserved_count = events.CAUSAL_FINALIZER_RESERVE
        causal.transition_reserved_count = events.CAUSAL_TRANSITION_RESERVE
        with events.transition_span('late_fill.callback.lookup'):
            call({'id': 'resp_test'})
    transitions = [r for r in records(tmp_path) if r['record_kind'] == 'transition']
    assert len(transitions) == 2
    finalizer = next(r for r in transitions if r['owner'] == 'response_frame.finalize')
    assert finalizer['operations']['work']['calls'] == 1
    assert finalizer['work']['bytes_processed'] == 42
    assert finalizer['lifecycle_role'] == 'initial'


def test_overflow_explicit_and_never_limits_execution(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    monkeypatch.setattr(sf, 'STATE_FLOW_RECORD_LIMIT', 3)
    calls = []
    @sf.observe_state('test')
    def call(): calls.append(1)
    with sf.state_flow_scope():
        for _ in range(10): call()
    assert len(calls) == 10
    data = records(tmp_path)
    assert len([r for r in data if r['record_kind'] == 'transition']) == 3
    assert data[-1]['dropped_count'] == 7
    assert data[-1]['interval_complete'] is False


def test_context_follows_existing_thread_handoff(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    @sf.observe_state('child')
    def child(): sf.note(child_work=1)
    with sf.state_flow_scope():
        with sf.transition('parent'):
            worker = threading.Thread(target=events.traced_thread_target(child))
            worker.start(); worker.join()
    ts = [r for r in records(tmp_path) if r['record_kind'] == 'transition']
    parent = next(r for r in ts if r['owner'] == 'parent')
    child = next(r for r in ts if r['owner'] == 'child')
    assert child['parent_transition_id'] == parent['transition_id']
    assert child['thread_id'] != parent['thread_id']
    assert child['scope_id'] == parent['scope_id']


def test_snapshot_observation_preserves_bytes_and_counts_cas_not_calls(tmp_path, monkeypatch):
    frame = {'kind': 'ollmo.response_frame', 'response_id': 'resp_test',
             'frame_id': 'resp_test:frame-1', 'frame_sequence': 1,
             'runtime': {'request_phase_graph': {'nodes': [{'id': 'p', 'body': 'X' * 80000}]}}}
    original = copy.deepcopy(frame)
    plain = rf.compact_response_frame_for_ledger(frame, frames_dir=tmp_path / 'plain')
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path / 'trace'))
    with sf.state_flow_scope():
        traced = rf.compact_response_frame_for_ledger(frame, frames_dir=tmp_path / 'traced')
        repeated = rf.compact_response_frame_for_ledger(frame, frames_dir=tmp_path / 'traced')
    assert frame == original
    assert plain == traced == repeated
    def bodies(root): return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.json')}
    assert bodies(tmp_path / 'plain') == bodies(tmp_path / 'traced')
    ts = [r for r in records(tmp_path / 'trace') if r['record_kind'] == 'transition']
    assert len(ts) == 2  # No recursive-child event flood.
    assert ts[0]['work']['cas_actual_writes'] > 0
    assert ts[1]['work'].get('cas_actual_writes', 0) == 0
    assert ts[1]['work']['cas_existing_matches'] > 0
    assert ts[0]['unique_identity_counts']['produced_cas'] == len(bodies(tmp_path / 'traced'))
    assert ts[0]['work']['snapshot_serializations'] >= ts[0]['unique_identity_counts']['produced_cas']


def test_identity_bound_and_failure_fields(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path))
    monkeypatch.setattr(sf, 'STATE_FLOW_IDENTITY_LIMIT', 2)
    with sf.state_flow_scope():
        with sf.transition('test'):
            for i in range(5): sf.note(identity_kind='produced_cas', identity=f'{i:064x}')
    summary = records(tmp_path)[-1]
    assert len(summary['identities']['produced_cas']) == 2
    assert summary['identity_drops'] == 3


def test_two_real_canonical_loads_keep_identity_checks_and_measure_existing_reads(tmp_path, monkeypatch):
    root = tmp_path / 'frames'
    frame = {'kind': 'ollmo.response_frame', 'frame_version': 9,
             'response_id': 'resp_test', 'status': 'completed',
             'current_state': {'id': 'resp_test', 'status': 'completed', 'lifecycle_state': 'completed'},
             'runtime': {'request_phase_graph': {'body': 'evidence ' * 10000}}}
    rf.persist_response_frame(frame, frames_dir=root)
    plain = rf.load_latest_response_state('resp_test', frames_dir=root)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path / 'trace'))
    with sf.state_flow_scope():
        for _ in range(2):
            assert rf.load_latest_response_state('resp_test', frames_dir=root) == plain
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    ts = [r for r in records(tmp_path / 'trace') if r['record_kind'] == 'transition']
    loads = [r for r in ts if r['owner'] == 'response_frame.canonical_load']
    reconstructions = [r for r in ts if r['owner'] == 'response_frame.canonical_reconstruction']
    assert len(loads) == len(reconstructions) == 2
    assert loads[0]['target']['frame_id'] == loads[1]['target']['frame_id']
    assert loads[0]['mapping_digest'] == loads[1]['mapping_digest']
    assert loads[0]['mapping_digest'] is not None
    assert all(r['work']['sidecar_payload_reads'] > 0 for r in reconstructions)
    assert all(r['bytes_read'] > 0 for r in reconstructions)
    assert all(r['new_authority_boundary'] is True for r in loads)
    assert all(r['source_unchanged_identity_proves_check_unnecessary'] is False for r in loads)


def test_corrupt_sidecar_failure_unchanged_with_observation(tmp_path, monkeypatch):
    frame = {'kind':'ollmo.response_frame', 'response_id':'resp_test',
             'current_state': {'id':'resp_test','lifecycle_state':'completed'},
             'runtime': {'large': 'evidence ' * 10000}}
    root=tmp_path/'frames'
    rf.persist_response_frame(frame, frames_dir=root)
    for p in (root/'snapshots').rglob('*.json'):
        p.write_bytes(b'{broken bytes')
    plain=rf.load_latest_response_state('resp_test', frames_dir=root)
    monkeypatch.setenv('OLLMO_STATE_FLOW_DIAGNOSTICS_DIR', str(tmp_path/'trace'))
    with sf.state_flow_scope():
        assert rf.load_latest_response_state('resp_test', frames_dir=root)==plain
    assert not plain['ok']
