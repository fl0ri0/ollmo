"""Structural-extraction regressions using injected fake work and local state only."""
from copy import deepcopy
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from unittest.mock import patch

import pytest

from ollmo_services import events
from ollmo_webserver import _LATE_FILL_RUNTIME
from ollmo_server.multi_materialization_runtime import MultiMaterializationRuntimeOwner


def run_worker(*, response_id='response-test', branches=2, sink_failure=False,
               log_failure=False, execution_failure=False, budget=1024, availability_wait=False, artifact_target=None):
    """Execute the real continuation loop with its existing dependency injections."""
    store, publications, logs, calls, causal = {}, [], [], [], []
    branch_records = [dict(branch_id=f'b{i}', phase_id=f'p{i}', capability='chat',
                           output_type='text', content_payload=f'branch {i}',
                           stage_direction='follow_up_text') for i in range(branches)]
    if artifact_target is not None:
        assert branches == 1
        branch_records[0].update(
            requires_artifact=True, text_artifact_extension='md',
            stage_direction='materialize_requested_text_artifact',
            text_artifact_target_path=str(artifact_target),
            text_artifact_source_name='repair',
            artifact_request={'extension': 'md', 'source_name': 'repair',
                              'target_path': str(artifact_target)},
        )
    # Separate waves exercise the lightweight and terminal checkpoint paths.
    for index, branch in enumerate(branch_records):
        branch['depends_on'] = [f'p{index - 1}'] if index else []
    payload = dict(id=response_id, mode='chat', status='completed', output_text='seed',
                   runtime={'request_phase_graph': {
                       'phases': deepcopy(branch_records),
                       'downstream_branches': deepcopy(branch_records),
                   }},
                   late_fill={'status': 'pending', 'pending_branches': deepcopy(branch_records),
                              'expected_capability': 'chat'})

    def touch(rid, **kwargs):
        calls.append(('publication', rid))
        publications.append(deepcopy(kwargs))
        if 'response_payload' in kwargs:
            store[rid] = {'response_payload': deepcopy(kwargs['response_payload'])}
        return store.get(rid)

    def finalize(value, **kwargs):
        calls.append(('finalize', kwargs.get('persist')))
        value = deepcopy(value)
        value.setdefault('runtime', {}).setdefault('developer_diagnostics', {})[
            'response_frame_finalize_timing'] = {'kind': 'fixture-finalizer', 'total_elapsed_ms': 7}
        return value

    def log(**entry):
        logs.append(deepcopy(entry))
        if log_failure and entry.get('action') == 'late_fill_post_wave_backend_timing':
            raise OSError('legacy timing sink failed')

    def sink(**entry):
        if sink_failure:
            raise OSError('causal sink failed')
        causal.append(deepcopy(entry['causal_event']))

    def prepare(**kwargs):
        calls.append(('prepare', kwargs['artifact_gap']['branch_id']))
        if availability_wait and sum(k == 'prepare' for k, _ in calls) == 1:
            error = RuntimeError("No ready late-fill instance for capability 'chat'.")
            error.route_diagnostics = {'availability_wait': {
                'reason': 'live_candidates_in_cooldown', 'candidate_instance_ids': ['fixture']}}
            raise error
        return {'route_info': {'instance_id': 'fixture'}, 'instance': {}, 'effective_data': {}}

    def execute(plan):
        calls.append(('execute', plan['branch_id']))
        if execution_failure:
            raise ValueError('fixture execution failed')
        infer_result = {'content': 'answer ' + plan['branch_id']}
        if artifact_target is not None:
            if sum(k == 'execute' for k, _ in calls) == 1:
                raise TimeoutError('fixture first attempt timed out')
            artifact_target.write_text('Saved repair result.\n')
            infer_result.update(saved_text_path=str(artifact_target), content='Saved repair result.')
        return {'route_info': plan['route_info'], 'instance': {},
                'effective_data': {}, 'infer_result': infer_result}

    owner = replace(
        _LATE_FILL_RUNTIME,
        get_response_lookup_record=lambda rid: deepcopy(store.get(rid)),
        touch_response_lookup=touch,
        ensure_response_lookup_for_payload=lambda *a, **k: None,
        finalize_response_frame_payload=finalize,
        response_registry_now_iso=lambda: '2026-09-09T12:00:00Z',
        log_unified_event=log,
        release_response_late_fill=lambda rid: calls.append(('release', rid)),
        schedule_post_response_substrate_hygiene=None,
        prepare_terminal_graph_patch_successor=None,
        project_terminal_closure_repair=None,
        review_terminal_graph_rebase=None,
        attach_runtime_graph_repair_evidence=None,
        load_latest_response_state=None,
        load_latest_response_observation_state=None,
        execute_materialization_branches=MultiMaterializationRuntimeOwner(
            max_parallel_workers=1).execute_materialization_branches,
    )
    with patch.object(events, 'CAUSAL_RECORD_LIMIT', budget), \
            patch.dict('os.environ', {'OLLMO_LATE_FILL_AVAILABILITY_POLL_SEC': '1'}), \
            events.causal_scope(sink, response_id=response_id) as scope:
        owner.complete_response_late_fill(
            response_payload=payload, request_payload={'prompt': 'Run the explicit branches.'},
            assistant_message='seed', artifact_gap={'expected_capability': 'chat',
                'pending_branches': deepcopy(branch_records), 'trigger': 'fixture'},
            source_route_payload=None, prepare_late_fill_branch_plan=prepare,
            execute_prepared_late_fill_branch=execute,
        )
        snapshot = scope.snapshot()
    return dict(payload=store[response_id]['response_payload'], publications=publications,
                logs=logs, calls=calls, causal=causal, snapshot=snapshot)


def test_real_worker_keeps_waves_callbacks_and_checkpoint_publication():
    run = run_worker()
    assert run['payload']['late_fill']['status'] == 'completed'
    assert [v for k, v in run['calls'] if k == 'execute'] == ['b0', 'b1']
    assert [k for k, v in run['calls']].count('finalize') == 1
    timing = [e for e in run['logs'] if e.get('action') == 'late_fill_post_wave_backend_timing']
    assert [e['phase'] for e in timing] == ['nonterminal', 'terminal']
    assert timing[0]['response_frame_finalize_timing']['skipped'] is True
    assert timing[1]['response_frame_finalize_timing']['kind'] == 'fixture-finalizer'
    completed = [p['response_payload']['late_fill'].get('branch_progress', []) for p in run['publications']]
    assert [b['branch_id'] for progress in completed for b in progress] == ['b0', 'b1']
    owners = Counter(e.get('owner') for e in run['causal'] if e.get('boundary') == 'exit')
    assert owners['late_fill.callback.lookup'] == owners['late_fill.callback.publication'] == 2
    assert run['calls'][-1] == ('release', 'response-test')


@pytest.mark.parametrize('sink_failure,budget', [(False, 0), (True, 1024), (True, 0)])
def test_real_worker_causal_failure_and_budget_do_not_change_runtime(sink_failure, budget):
    run = run_worker(sink_failure=sink_failure, budget=budget)
    assert run['payload']['late_fill']['status'] == 'completed'
    assert [v for k, v in run['calls'] if k == 'execute'] == ['b0', 'b1']
    assert run['snapshot']['delivery_failure_count'] > 0 if sink_failure else run['snapshot']['dropped_count'] > 0
    if budget == 0 and not sink_failure:
        assert any(e.get('owner') == 'late_fill.callback.publication' for e in run['causal'])
        assert run['snapshot']['transition_recorded_count'] > 0


def test_legacy_timing_log_exception_keeps_existing_worker_failure_path():
    run = run_worker(log_failure=True)
    assert run['payload']['late_fill']['status'] == 'failed'
    assert run['payload']['late_fill']['error'] == 'legacy timing sink failed'
    assert [v for k, v in run['calls'] if k == 'execute'] == ['b0']
    assert run['calls'][-1] == ('release', 'response-test')


def test_execution_exception_still_fails_branch_and_releases_worker():
    run = run_worker(branches=1, execution_failure=True)
    assert run['payload']['late_fill']['status'] == 'failed'
    failed = run['payload']['late_fill']['failed_branches']
    assert failed[0]['error']['message'] == 'fixture execution failed'
    assert run['calls'][-1] == ('release', 'response-test')


def test_concurrent_workers_keep_response_and_boot_bindings():
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(lambda rid: run_worker(response_id=rid, branches=1), ['left', 'right']))
    for rid, run in zip(['left', 'right'], runs):
        assert run['payload']['id'] == rid
        bound = [r for r in run['causal'] if r.get('target', {}).get('response_id')]
        assert bound and {r['target']['response_id'] for r in bound} == {rid}
        assert len({r['process_boot_id'] for r in run['causal'] if 'process_boot_id' in r}) == 1
    assert runs[0]['snapshot']['scope_id'] != runs[1]['snapshot']['scope_id']


def test_availability_wait_wake_and_settlement_keep_the_exact_wait_identity():
    run = run_worker(branches=1, availability_wait=True)
    assert run['payload']['late_fill']['status'] == 'completed'
    records = [r for r in run['causal'] if r.get('owner') == 'late_fill.availability']
    assert [r['record_kind'] for r in records] == ['wait_started', 'wait_wake', 'wait_settled']
    assert records[1]['wait_id'] == records[2]['wait_id'] == records[0]['event_id']
    assert records[0]['monotonic_ns'] <= records[1]['monotonic_ns'] <= records[2]['monotonic_ns']
    completed = run['payload']['late_fill']['completed_branches'][0]
    assert 'availability_wait' not in completed
    assert not completed.get('auto_executable_repair_retry_count')
    assert [v for k, v in run['calls'] if k == 'execute'] == ['b0']


def test_worker_preserves_pairing_and_result_callback_publication_drain_order():
    from tests.test_transition_telemetry import assert_pairs
    run = run_worker()
    records = run['causal']
    assert_pairs(records)
    available = [r for r in records if r.get('owner') == 'multi_materialization.result']
    for result in available:
        callback = [r for r in records if r.get('owner') == 'multi_materialization.callback'
                    and r.get('handoff_attempt_id') == result['handoff_attempt_id']]
        assert len(callback) == 2
        enter = next(r for r in callback if r['boundary'] == 'enter')
        end = next(r for r in callback if r['boundary'] == 'exit')
        assert enter['predecessor_result_id'] == result['result_id']
        assert result['monotonic_ns'] <= enter['monotonic_ns'] <= end['monotonic_ns']
        publication = next(r for r in records if r.get('owner') == 'late_fill.callback.publication'
                           and r.get('boundary') == 'exit'
                           and r.get('handoff_attempt_id') == result['handoff_attempt_id'])
        assert enter['monotonic_ns'] <= publication['monotonic_ns'] <= end['monotonic_ns']
        drain = next(r for r in records if r.get('owner') == 'multi_materialization.callback_drain'
                     and r.get('boundary') == 'exit' and r.get('handoff_attempt_id') == result['wave_id'])
        assert end['monotonic_ns'] <= drain['monotonic_ns']


def test_real_worker_retry_retains_attempt_binding_and_saved_file(tmp_path):
    target = tmp_path / 'repair.md'
    run = run_worker(branches=1, artifact_target=target)
    assert run['payload']['late_fill']['status'] == 'completed'
    assert [v for k, v in run['calls'] if k == 'execute'] == ['b0', 'b0']
    retry = [e for e in run['logs'] if e.get('action') == 'late_fill' and e.get('status') == 'queued']
    assert len(retry) == 1
    assert retry[0]['branch_id'] == 'b0' and retry[0]['phase_id'] == 'p0'
    assert retry[0]['attempt']['instance_id'] == 'fixture'
    completed = run['payload']['late_fill']['completed_branches'][0]
    assert completed['auto_executable_repair_retry_count'] == 1
    assert target.read_text() == 'Saved repair result.\n'
