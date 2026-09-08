from scripts.analyze_state_flow import analyze, duration


def event(ident, owner, start, end, *, parent=None, thread=1, boot='boot'):
    return dict(record_kind='transition', scope_id='scope', response_id='r',
                process_boot_id=boot, transition_id=ident, owner=owner,
                start_monotonic_ns=start,end_monotonic_ns=end,inclusive_ns=end-start,
                parent_transition_id=parent,thread_id=thread,source={},target={},work={})


def test_interval_union_and_unknown_rest_do_not_double_count_nested_or_parallel_work():
    records=[event('r','responses_request.handle_responses_request',0,100),
             event('f','response_frame.finalize',10,80,parent='r'),
             event('c','response_frame.compact',20,50,parent='f'),
             event('p','other_thread',40,90,parent='r',thread=2)]
    data=analyze(records,'r');timing=data['timing_by_process_boot']['boot']
    assert timing['measured_owner_union_ns']==80
    assert timing['unknown_rest_ns']==20
    assert timing['finalizer_inclusive_sum_ns']==70
    assert next(r for r in data['transitions'] if r['transition_id']=='f')['exclusive_or_unattributed_rest_ns']==40
    assert sum(b['duration_ns'] for b in data['top_10_nonoverlapping_outer_blocks'])==100
    assert timing['critical_path_coverage'] is None
    assert data['amplification']['total_processed_over_final_canonical_bytes'] is None


def test_duplicate_records_and_distinct_boot_clocks():
    one=event('f','response_frame.finalize',10,20)
    data=analyze([one,one,event('x','response_frame.finalize',0,10,boot='different')],'r')
    assert data['turn_summary']['full_finalizers_completed']==2
    assert len(data['timing_by_process_boot'])==2
    assert data['turn_summary']['snapshot_roots'] is None
    assert duration([(3,8),(2,4),(9,10)])==7
