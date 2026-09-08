from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.self_attack_production import (
    event_analysis, expand_exact_frames, latency, lock_waits, request_start,
    run_production_audit, terminal_event, instrumentation_summary, instrumented_cohort,
)
from scripts.self_attack_convergence import analyze_causal_records
from scripts.ollmo_self_attack import main


def event(identity='one',count=1,*,skipped=False,terminal=True):
    return {'record':{'id':identity,'response_id':'resp_test','timestamp':'2026-09-06T10:10:00Z',
        'action':'late_fill_post_wave_backend_timing','phase':'terminal' if terminal else 'nonterminal',
        'late_fill_status':'completed' if terminal else 'running','finalize_skipped':skipped,
        'finalize_elapsed_ms':1000,'response_frame_finalize_timing':{
            'phase':'terminal_late_fill','persist_requested':True,'persist_effective':True,
            'completed_branch_count':count,'steps':[{'name':'persist_response_frame','elapsed_ms':700}]}},
            'evidence':{'path':'events.jsonl','line':1}}


def history(at='2026-09-06T10:00:00Z',role='user'):
    return {'role':role,'request_created_at':at,'evidence':{'path':'history.json','pointer':'/messages/0'}}


def test_event_id_dedup_and_changed_input_prove_necessary_repeat():
    first=event();second=event('two',2)
    r=event_analysis([first,first,second])
    assert len(r['finalizer_invocations'])==2
    assert r['finalizer_invocations'][0]['classification']=='unknown'
    assert r['finalizer_invocations'][1]['classification']=='necessary_repeat'
    assert r['finalizer_invocations'][1]['triggering_state_delta']=={'completed_branch_count':{'previous':1,'current':2}}
    assert r['finalizer_seconds']==2
    assert r['step_seconds']['persist_response_frame']==1.4


def test_equal_counts_do_not_prove_redundancy_or_defensive_contract():
    r=event_analysis([event(),event('two')])
    assert all(x['classification']=='unknown' for x in r['finalizer_invocations'])


def test_skipped_finalizer_is_not_invocation():
    assert not event_analysis([event(skipped=True)])['finalizer_invocations']


def test_conflicting_event_ids_are_explicit():
    assert event_analysis([event(),event(count=2)])['event_id_conflicts']==['one']


def test_assistant_timestamp_and_response_id_are_not_start_witnesses():
    assert request_start([history(role='assistant')])['at'] is None
    assert request_start([history(),history('2026-09-06T10:00:01Z')])['at'] is None


def test_nonterminal_checkpoint_cannot_close_latency():
    assert not terminal_event(event(terminal=False)['record'])
    assert latency([history()],[event(terminal=False)],[])['seconds'] is None
    assert latency([history()],[event()],[])['seconds']==600


def test_late_fill_completion_boundary_labeled_without_claiming_browser_latency():
    r=latency([history()],[],[{'frame':{'late_fill':{'status':'completed','completed_at':'2026-09-06T10:02:00Z'}},'evidence':{'line':1}}])
    assert r['seconds']==120
    assert r['endpoint_kind']=='late_fill_completion_only'
    assert r['browser_visible_latency_seconds'] is None


def test_lock_wait_requires_observed_holder_and_owner_policy():
    batch={'same_instance_lock_groups':{'model':['a','b']},'evidence':['x'], 'branch_timings':[
        {'branch_id':'a','instance_id':'model','queued_elapsed_ms':0,'lock_wait_ms':0,'elapsed_ms':1000},
        {'branch_id':'b','instance_id':'model','queued_elapsed_ms':0,'lock_wait_ms':1000,'elapsed_ms':1600}]}
    r=lock_waits([batch]);assert len(r)==1
    assert r[0]['classification']=='necessary_wait' and r[0]['duration_seconds']==1
    batch['same_instance_lock_groups']={}
    assert lock_waits([batch])==[]


def test_missing_parent_never_borrows_previous_manifest():
    item={'frame':{'response_id':'r','frame_id':'f2','frame_relation':{'parent_frame_id':'missing'}},'evidence':{'line':1}}
    expanded,errors=expand_exact_frames([item])
    assert expanded[0][1] is None
    assert errors[0]['error']=='missing_exact_parent_manifest'


def test_parent_manifest_inherits_exact_refs():
    ref={'path':'snapshots/one.json','sha256':'a'}
    first={'frame':{'frame_id':'f1','external_snapshots':{'items':{'runtime':ref}}},'evidence':{'line':1}}
    second={'frame':{'frame_id':'f2','frame_relation':{'parent_frame_id':'f1'},'external_snapshots':{'items':{}}},'evidence':{'line':2}}
    values,errors=expand_exact_frames([first,second])
    assert not errors
    assert values[1][1]['external_snapshots']['items']['runtime']==ref


def test_production_cli_cannot_dispatch(tmp_path):
    with patch('scripts.self_attack_production.run_production_audit',return_value=0) as run, \
         patch('scripts.ollmo_self_attack.execute_sweep',side_effect=AssertionError('dispatch')):
        assert main(['--audit-production',str(tmp_path/'frames'),'--output',str(tmp_path/'audit')])==0
        run.assert_called_once()
    with pytest.raises(SystemExit):main(['--audit-production',str(tmp_path),'--live-after-fake'])


def test_small_production_audit_read_only_and_corpus_separation(tmp_path):
    state=tmp_path/'state';frames=state/'response_frames';frames.mkdir(parents=True)
    history_dir=state/'chat_history';history_dir.mkdir()
    rid='resp_123_abcd'
    frame={'kind':'ollmo.response_frame','response_id':rid,'frame_id':rid+':frame-1','frame_sequence':1,
           'request':{'prompt':'Write a poem'},'current_state':{'id':rid,'lifecycle_state':'completed','status':'completed','runtime':{}},
           'late_fill':{'status':'completed','started_at':'2026-09-06T10:00:30Z','completed_at':'2026-09-06T10:02:00Z'}}
    ledger=frames/'responses.jsonl';ledger.write_text(json.dumps(frame)+'\n'+json.dumps(dict(frame,response_id='resp_corpus_test'))+'\n')
    (history_dir/'chat.json').write_text(json.dumps({'messages':[{'role':'user','request_snapshot':{'response_id':rid,'created_at':'2026-09-06T10:00:00Z'}}]}))
    original=ledger.read_bytes()
    out=tmp_path/'audit'
    assert run_production_audit(frames,out,repo=tmp_path)==0
    r=json.loads((out/'results.json').read_text())
    assert r['aggregate']['ordinary_responses']==1
    assert r['aggregate']['self_attack_responses']==1
    assert r['ranked_responses'][0]['latency']['seconds']==120
    assert ledger.read_bytes()==original
    assert r['new_model_work'] is False
    assert not r['aggregate']['hydration_errors']
    assert r['instrumented_cohort']['response_count'] == 0
    assert r['instrumented_cohort']['historical_or_uninstrumented_count'] == 1


def test_instrumented_cohort_keeps_history_unknown_and_deduplicates_completion():
    call = dict(schema='ollmo.causal_event.v1', event_id='e', invocation_id='i',
        record_kind='runtime_invocation', owner='response_frame.finalize', status='returned',
        process_boot_id='boot', scope_id='scope', target={'response_id': 'r'},
        start_monotonic_ns=1, end_monotonic_ns=2000000001,
        operations={'nested': {'role': 'canonical_truth', 'calls': 3, 'elapsed_ns': 4000000000}})
    gap = dict(schema='ollmo.causal_event.v1', event_id='gap', record_kind='coverage_gap',
               scope_id='scope', dropped_count=9)
    events = [call, call, gap, gap]
    analysis = analyze_causal_records(events)
    coverage = instrumentation_summary(events, analysis)
    assert coverage['finalizer_seconds'] == 2
    assert coverage['reported_dropped_count_lower_bound'] == 9
    instrumented = dict(response_id='r', details_path='details', causal_coverage=coverage,
        instrumented_classification_counts=analysis['classification_counts'], instrumented_invocation_count=1)
    historical = dict(response_id='old', causal_coverage=instrumentation_summary([], analyze_causal_records([])))
    result = instrumented_cohort([historical, instrumented])
    assert result['response_count'] == result['historical_or_uninstrumented_count'] == 1
    assert result['classification_counts']['unknown'] == 1
    assert result['classification_counts']['redundant_repeat'] == 0
    assert result['finalizer_seconds'] == 2
    assert result['finalizer_operations'][0]['elapsed_seconds'] == 4  # nested, not exclusive


def test_conflicting_finalizer_identity_is_not_ranked_as_measured_time():
    call = dict(schema='ollmo.causal_event.v1', event_id='e', invocation_id='i',
        owner='response_frame.finalize', status='returned', process_boot_id='boot',
        start_monotonic_ns=1, end_monotonic_ns=2)
    events = [call, dict(call, end_monotonic_ns=3)]
    coverage = instrumentation_summary(events, analyze_causal_records(events))
    assert coverage['cohort'] == 'instrumented'
    assert coverage['finalizer_seconds'] is None
    assert coverage['event_id_conflicts'] == ['e']
