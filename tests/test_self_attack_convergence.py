from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import self_attack_convergence as audit
from scripts.ollmo_self_attack import main
from scripts.run_graph_rebase_shadow_corpus import CorpusError


def payload():
    return {'id': 'resp_test', 'runtime': {'request_phase_graph': {
        'phases': [{'phase_id': 'p1', 'status': 'fulfilled'}, {'phase_id': 'p2', 'depends_on': ['p1']}],
        'decision_contract': {'aspiration_review': {'status': 'pending', 'authority': 'advisory'}}}},
        'late_fill': {'trigger': 'deferred_follow_up', 'materialization_concurrency_history': [{
            'elapsed_ms': 1200, 'planning_elapsed_ms': 100, 'worker_count': 2,
            'branch_timings': [
                {'branch_id': 'b1', 'queued_elapsed_ms': 0, 'elapsed_ms': 1000, 'execution_ms': 1000},
                {'branch_id': 'b2', 'queued_elapsed_ms': 100, 'elapsed_ms': 500, 'execution_ms': 500}],
            'execution_submission_order': ['b1', 'b2']}]},
        'response_frame': {'frame_id': 'f1', 'late_fill': {
            'started_at': '2026-09-06T10:00:00Z', 'completed_at': '2026-09-06T10:00:02Z'}}}


def save_capture(path, value=None, *, at=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload=value or payload(), source='settled',
                                   observed_ns=at, captured_at=f'2026-09-06T10:00:0{at}Z')))
    return path


def test_unchanged_outputs_never_prove_redundancy():
    assert audit.classify_repeat({'output_hash': 'same'}, {'output_hash': 'same'}) == 'unknown'
    proof = dict(effective_input_hash='a', authority_hash='b', evidence_hash='c', input_coverage_complete=True)
    assert audit.classify_repeat(proof, proof) == 'unknown'
    assert audit.classify_repeat(proof, dict(proof, required_defensive_contract='pre_dispatch_gate')) == 'defensive_repeat'
    assert audit.classify_repeat(proof, dict(proof, evidence_hash='d')) == 'unknown'
    assert audit.classify_repeat(proof, dict(proof, evidence_hash='d', relevant_change_proven=True)) == 'necessary_repeat'
    complete = dict(proof, evidence_interval_complete=True, no_required_downstream_role_proven=True,
                    authority_contract_checked=True)
    assert audit.classify_repeat(proof, complete) == 'redundant_repeat'
    for key in ('evidence_interval_complete', 'no_required_downstream_role_proven', 'authority_contract_checked'):
        assert audit.classify_repeat(proof, dict(complete, **{key:False})) == 'unknown'


def test_overlapping_branch_time_not_summed_as_wall_time():
    extracted=audit.extract_capture(payload(), 'settled', '2026-09-06T10:00:03Z')
    batch=extracted['batches'][0]
    assert batch['branch_interval_union_ms']==1000
    assert batch['execution_effort_ms']==1500
    assert batch['after_last_branch_ms']==200
    assert batch['planning_elapsed_ms']==100
    assert extracted['late_fill_interval']['seconds']==2
    assert all(x['classification']=='unknown' for x in extracted['invocations'])


def test_copied_captures_and_repeated_projections_are_not_invocations(tmp_path):
    a=save_capture(tmp_path/'a/resp_test/one.json')
    b=save_capture(tmp_path/'b/resp_test/one.json')
    c=save_capture(tmp_path/'a/resp_test/two.json',at=2)
    out=tmp_path/'details.json'
    summary=audit.analyze_response('resp_test',[a,b,c],[],[],out)
    result=json.loads(out.read_text())
    assert summary['duplicate_capture_copies']==1
    assert summary['invocation_count']==2
    assert summary['unchanged_projection_observations']>0
    assert result['timing']['batch_seconds']['execution_effort']==1.5
    assert result['classification_counts']=={'unknown':2}
    assert a.read_bytes()==b.read_bytes()


def test_history_ordinal_conflicts_excluded_from_aggregate(tmp_path):
    a=save_capture(tmp_path/'resp_test/a.json')
    value=payload();value['late_fill']['materialization_concurrency_history'][0]['elapsed_ms']=1400
    b=save_capture(tmp_path/'resp_test/b.json',value,at=2)
    out=tmp_path/'out.json'
    summary=audit.analyze_response('resp_test',[a,b],[],[],out)
    assert summary['timing']['conflicting_history_ordinals']==[0]
    assert summary['timing']['batch_seconds']['wave_elapsed']==0


def test_pending_ready_dependency_does_not_prove_false_wait_or_missed_wakeup():
    value=payload();value['late_fill']['pending_branches']=[{
        'branch_id':'b2','depends_on':['p1'],'status':'waiting','wait_reason':'dependency'}]
    extracted=audit.extract_capture(value,'observation','2026-09-06T10:00:03Z')
    assert extracted['waits'][0]['classification']=='unknown'
    assert extracted['waits'][0]['wake_condition']['predicate'] is None
    assert extracted['waits'][0]['dependency_states']=={'p1':'fulfilled'}


def test_timestamps_missing_reversed_or_naive_remain_unknown():
    assert audit.interval('2026-09-06T00:00:00Z','2026-09-05T00:00:00Z') is None
    assert audit.interval(None,'2026-09-05T00:00:00Z') is None
    assert audit.interval('2026-09-06T00:00:00','2026-09-06T00:00:01') is None
    assert audit.number(float('nan')) is None


def test_malformed_and_identity_mismatch_capture_are_visible(tmp_path):
    a=tmp_path/'a.json';a.write_text('{')
    b=save_capture(tmp_path/'b.json')
    out=tmp_path/'out.json'
    result=audit.analyze_response('different',[a,b],[],[],out)
    assert len(result['errors'])==2
    assert result['invocation_count']==0


def test_complete_offline_run_and_source_preservation(tmp_path):
    source=tmp_path/'evidence';seq=source/'baseline/sequence-1';seq.mkdir(parents=True)
    capture=save_capture(seq/'captures/resp_test/one.json')
    (seq/'worker.json').write_text(json.dumps({'mode':'live','profile':{'id':'baseline'}}))
    (seq/'manifest.json').write_text(json.dumps({'cases':[{'case_id':'case','response_id':'resp_test',
        'submitting_at':'2026-09-06T10:00:00Z','settled_at':'2026-09-06T10:00:03Z',
        'final_debug':{'attempted_at':'2026-09-06T10:00:02Z','finished_at':'2026-09-06T10:00:03Z'}}]}))
    original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    output=tmp_path/'out'
    assert audit.run_audit(source,output,repo=tmp_path)==0
    result=json.loads((output/'results.json').read_text())
    assert result['aggregate']['by_mode']['live']['responses']==1
    assert result['new_live_work'] is False
    assert result['opportunities'][0]['measured_exposure_seconds']==1.5
    assert all(o['code'] != 'unattributed_end_to_end' for o in result['opportunities'])
    assert result['opportunities'][0]['estimated_recoverable_wall_seconds'] is None
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in original.items())
    assert audit.run_audit(source,output,repo=tmp_path)==0
    save_capture(capture,at=2)
    with pytest.raises(CorpusError,match='changed'):
        audit.run_audit(source,output,repo=tmp_path)


def test_cli_offline_branch_never_dispatches(tmp_path):
    with patch('scripts.self_attack_convergence.run_audit',return_value=0) as run, \
         patch('scripts.ollmo_self_attack.execute_sweep',side_effect=AssertionError('dispatch')), \
         patch('scripts.ollmo_self_attack.discover_knobs',side_effect=AssertionError('inventory runtime')):
        assert main(['--audit-convergence',str(tmp_path),'--output',str(tmp_path/'out')])==0
        run.assert_called_once()
    with pytest.raises(SystemExit):
        main(['--audit-convergence',str(tmp_path),'--live-after-fake'])


def test_ledger_custom_response_ids_and_no_hydration(tmp_path):
    path=tmp_path/'responses.jsonl'
    path.write_text(json.dumps({'response_id':'custom-id','frame_id':'one','frame_sequence':1})+'\n')
    result=audit.ledger_index([path],{'custom-id'},tmp_path/'index.json')
    assert result['frame_rows']==1
    assert result['files'][0]['stable_during_read']


def test_empty_field_compaction_does_not_duplicate_timing_witness(tmp_path):
    value=payload()
    batch=value['late_fill']['materialization_concurrency_history'][0]
    batch['same_instance_lock_groups']={}
    a=save_capture(tmp_path/'resp_test/a.json',value)
    del batch['same_instance_lock_groups']
    b=save_capture(tmp_path/'resp_test/b.json',value,at=2)
    result=audit.analyze_response('resp_test',[a,b],[],[],tmp_path/'out.json')
    assert result['timing']['conflicting_history_ordinals']==[]
    assert result['timing']['unique_batch_witnesses']==1
    assert result['timing']['batch_seconds']['wave_elapsed']==1.2


def test_bad_runtime_shape_reports_error_instead_of_crashing(tmp_path):
    value=payload();value['runtime']=['invalid']
    capture=save_capture(tmp_path/'resp_test/a.json',value)
    result=audit.analyze_response('resp_test',[capture],[],[],tmp_path/'out.json')
    assert len(result['errors'])==1


def test_observation_delay_does_not_rank_as_runtime_cost():
    summary={'response_id':'r','contexts':[{'mode':'live','profile':{'id':'baseline'}}],
             'details_path':'details.json', 'timing_evidence':{'batch':[],'observer':[]},
             'timing':{'observed_dispatch_to_settled_seconds':99999,'debug_capture_seconds':0,
                       'batch_seconds':{'after_last_branch':0,'execution_effort':0,'planning_elapsed':0}}}
    assert audit.rank_opportunities([summary])==[]
