"""Production evidence adapter for the existing self-attack convergence auditor.

No model/client/runtime execution. Hydration uses the existing authorized frame
reader; copied captures and reports are audit artifacts, never historical truth.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

from scripts.self_attack_convergence import (
    analyze_response, auxiliary_evidence, interval, number, read_json, records,
    short_state, stable_digest, timestamp, timing_identity, union_ms, INSTRUMENTATION, CLASSES,
)
from scripts.run_graph_rebase_shadow_corpus import atomic_write_json, utc_now, CorpusError

KIND = 'ollmo.production_ledger.convergence.v1'
TERMINAL = {'completed', 'failed', 'partial_failed', 'cancelled', 'repair_needed'}
INPUT_COUNTS = ('pending_branch_count', 'active_branch_count', 'completed_branch_count', 'failed_branch_count', 'late_fill_status')


def evidence_ref(path, **fields):
    return dict(path=str(path), **fields)


def read_jsonl(path):
    if not path.exists():
        return [], {'path':str(path), 'missing':True}, []
    before=path.stat();digest=hashlib.sha256();items=[];errors=[];offset=0
    with path.open('rb') as stream:
        # Freeze the read boundary; a concurrently appended record is a later epoch.
        remaining=before.st_size
        for line_no,raw in enumerate(iter(stream.readline,b''),1):
            if remaining <= 0: break
            if len(raw)>remaining:
                errors.append(dict(line=line_no,error='record crosses frozen byte boundary'));break
            remaining-=len(raw);digest.update(raw)
            try:
                value=json.loads(raw)
                if not isinstance(value,dict):raise ValueError('record is not an object')
                items.append((value,evidence_ref(path,line=line_no,byte_offset=offset,
                             row_sha256=hashlib.sha256(raw).hexdigest()),raw))
            except ValueError as exc:errors.append(dict(line=line_no,error=str(exc)))
            offset+=len(raw)
    after=path.stat()
    return items,dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest(),mtime_ns=before.st_mtime_ns,
                      stable_during_read=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)),errors


def index_history(state_dir):
    indexed=defaultdict(list);files=[];errors=[]
    for path in sorted((state_dir/'chat_history').glob('*.json')):
        try:
            data,digest=read_json(path)
            files.append(dict(path=str(path),sha256=digest))
            if not isinstance(data,dict):continue
            for i,message in enumerate(records(data.get('messages'))):
                snapshot=message.get('request_snapshot') or {}
                if not isinstance(snapshot,dict):continue
                rid=message.get('response_id') or snapshot.get('response_id')
                if not rid:continue
                indexed[rid].append(dict(role=message.get('role'),message_timestamp=message.get('timestamp'),
                    request_created_at=snapshot.get('created_at'),settings=snapshot.get('settings'),
                    retained_ghost_preview=snapshot.get('ghost_preview'),
                    request_id=snapshot.get('request_id'),conversation_id=snapshot.get('conversation_id'),
                    evidence=evidence_ref(path,pointer=f'/messages/{i}',sha256=digest)))
        except (ValueError,OSError) as exc:errors.append(dict(path=str(path),error=str(exc)))
    return indexed,files,errors


def request_start(history):
    candidates=[h for h in history if h.get('role')=='user' and timestamp(h.get('request_created_at')) is not None]
    identities={h['request_created_at'] for h in candidates}
    if len(identities)!=1:
        return dict(at=None,confidence='unknown',reason='Missing or conflicting response-bound user request snapshots',evidence=[h['evidence'] for h in candidates])
    return dict(at=candidates[0]['request_created_at'],confidence='high',
                reason='Saved user request snapshot bound to this response id; not decoded from an id or assistant message time.',
                evidence=[h['evidence'] for h in candidates])


def terminal_event(event):
    timing=event.get('response_frame_finalize_timing') or {}
    return (event.get('action')=='late_fill_post_wave_backend_timing'
            and event.get('phase')=='terminal' and event.get('finalize_skipped') is False
            and event.get('late_fill_status') in TERMINAL
            and timing.get('phase')=='terminal_late_fill')


def latency(history, events, frames):
    start=request_start(history)
    endpoints=[(e['record']['timestamp'],'terminal_checkpoint',e['evidence']) for e in events
               if terminal_event(e['record']) and timestamp(e['record'].get('timestamp')) is not None]
    if not endpoints:
        for item in frames:
            lf=item['frame'].get('late_fill') or {}
            if lf.get('status') in TERMINAL:
                for key in ('completed_at','failed_at'):
                    if timestamp(lf.get(key)) is not None:
                        endpoints.append((lf[key],'late_fill_completion_only',dict(item['evidence'],pointer='/late_fill/'+key)))
    endpoint=max(endpoints,key=lambda x:timestamp(x[0])) if endpoints else None
    seconds=interval(start['at'],endpoint[0]) if endpoint else None
    return dict(start=start,end_at=endpoint[0] if endpoint else None,
        endpoint_kind=endpoint[1] if endpoint else 'unknown',seconds=seconds,
        end_evidence=endpoint[2] if endpoint else None,
        confidence='medium' if seconds is not None else 'unknown',browser_visible_latency_seconds=None,
        limitation='Request-to-last retained terminal-owner checkpoint, or labeled Late Fill boundary. Browser rendering and external/reopen pauses are not measured. Not a continuously busy-work duration.')


def event_analysis(events):
    invocations=[];requeues=[];previous=None;step_totals=Counter();durations=[];seen={};conflicts=[]
    for item in events:
        event=item['record'];key=event.get('id')
        if key:
            digest=stable_digest(event)
            if key in seen:
                if seen[key]!=digest:conflicts.append(key)
                continue
            seen[key]=digest
        if event.get('action')=='late_fill' and ('requeued' in str(event.get('message','')).lower() or event.get('status')=='queued'):
            requeues.append(dict(event_id=key,at=event.get('timestamp'),reason=event.get('message'),
                branch_id=event.get('branch_id'),phase_id=event.get('phase_id'),classification='unknown',
                limit='The owner recorded a retry trigger. Without branch/attempt input identities this does not prove redundant or necessary repeated execution.',evidence=item['evidence']))
        if event.get('action')!='late_fill_post_wave_backend_timing':continue
        timing=event.get('response_frame_finalize_timing') or {}
        if event.get('finalize_skipped') is not False or timing.get('skipped') is True:continue
        duration=number(event.get('finalize_elapsed_ms'))
        if duration is None:continue
        inputs={k:timing[k] for k in INPUT_COUNTS if k in timing}
        state_delta={k:dict(previous=previous['input_summary'][k],current=v) for k,v in inputs.items()
                     if previous and k in previous['input_summary'] and previous['input_summary'][k]!=v}
        inv=dict(component='response_frame_finalization',owner='LateFillRuntime post-wave finalizer',
            invocation_id=key,previous_invocation_id=previous['invocation_id'] if previous else None,
            classification='necessary_repeat' if previous and state_delta else 'unknown',
            classification_basis=('A distinct non-skipped owner call received changed relevant branch/lifecycle state; persisted truth must reflect that change. This proves justification for reconsideration, not necessity of every elapsed millisecond.' if state_delta else
                                  'No complete effective-input/authority/evidence/defensive-contract witness for this repeat (or first observed invocation).'),
            trigger_reason=event.get('checkpoint_mode'),triggering_state_delta=state_delta or None,
            attention_target=event.get('response_id'),branch_id=event.get('branch_id'),frame_id=event.get('frame_id'),
            input_summary=inputs,relevant_input_projection_hash=stable_digest(inputs),
            effective_input_hash=None,authority_context={k:timing.get(k) for k in ('persist_requested','persist_effective','phase')},
            authority_context_complete=False,evidence_set_changed=None,contract_changed=None,
            start_at=None,end_at=event.get('timestamp'),duration_seconds=duration/1000,
            output_changed=None,runtime_state_changed=None,downstream_created_unblocked_consumed=None,superseded_by=None,
            steps=timing.get('steps',[]),evidence=item['evidence'])
        invocations.append(inv);previous=inv
        durations.append(duration/1000)
        for step in records(timing.get('steps')):
            value=number(step.get('elapsed_ms'))
            if value is not None:step_totals[step.get('name','unknown')]+=value/1000
    return dict(finalizer_invocations=invocations,requeues=requeues,event_id_conflicts=conflicts,
        finalizer_seconds=sum(durations),step_seconds=dict(step_totals),
        timing_limit='Step totals are nested within finalizer duration. Event timestamps follow the timed call and lookup; they are not exact call-end timestamps. Do not add frame-projected copies of these timings.')


def lock_waits(batches):
    waits=[]
    for batch in batches:
        timings=batch.get('branch_timings') or []
        groups=batch.get('same_instance_lock_groups') or {}
        for b in timings:
            q,w=number(b.get('queued_elapsed_ms')),number(b.get('lock_wait_ms'))
            if q is None or w is None or w<=0:continue
            owners=[]
            for other in timings:
                if other.get('branch_id')==b.get('branch_id') or other.get('instance_id')!=b.get('instance_id'):continue
                oq,ow,oe=(number(other.get(k)) for k in ('queued_elapsed_ms','lock_wait_ms','elapsed_ms'))
                if None not in (oq,ow,oe) and min(q+w,oq+oe)>max(q,oq+ow)+.01:
                    owners.append(other.get('branch_id'))
            if owners and b.get('instance_id') in groups:
                waits.append(dict(classification='necessary_wait',component='multi_materialization_runtime instance lock',
                    branch_id=b.get('branch_id'),instance_id=b.get('instance_id'),duration_seconds=w/1000,
                    wake_condition='The existing same-instance mutex is released by its current branch owner.',
                    blocking_branches=owners,confidence='high',evidence=batch['evidence'],
                    basis='Same-instance lock policy plus overlapping holder execution and recorded acquisition duration; 0.01ms margin covers millisecond rounding. Independence across model calls is not assumed.'))
    return waits


def expand_exact_frames(items):
    from ollmo_services.response_frames import _effective_snapshot_manifest, _expand_frame_snapshot_manifest
    known={};expanded=[];errors=[]
    for item in items:
        frame=item['frame'];relation=frame.get('frame_relation') or {}
        inheritance=(frame.get('external_snapshots') or {}).get('inheritance') or {}
        parent=relation.get('parent_frame_id') or inheritance.get('parent_frame_id')
        if parent and parent not in known:
            errors.append(dict(frame_id=frame.get('frame_id'),error='missing_exact_parent_manifest',parent_frame_id=parent,evidence=item['evidence']))
            expanded.append((item,None));continue
        manifest=_effective_snapshot_manifest(frame,parent_manifest=known.get(parent))
        known[frame.get('frame_id')]=manifest
        expanded.append((item,_expand_frame_snapshot_manifest(frame,effective_manifest=manifest)))
    return expanded,errors


def hydrate_response(rid, items, frames_dir, output):
    from ollmo_services.response_frames import _canonical_response_payload_from_frame
    expanded,errors=expand_exact_frames(items);paths=[];frame_index=[]
    folder=output/'captures'/rid;folder.mkdir(parents=True,exist_ok=True)
    for item,frame in expanded:
        info=dict(frame_id=item['frame'].get('frame_id'),frame_sequence=item['frame'].get('frame_sequence'),
                  evidence=item['evidence'],raw_copy=item['raw_copy'],status='not_hydrated')
        frame_index.append(info)
        if frame is None:continue
        start=time.monotonic()
        payload,error=_canonical_response_payload_from_frame(frame,frames_dir=frames_dir)
        info['audit_hydration_seconds']=round(time.monotonic()-start,6)
        if error:
            info['status']='integrity_error';info['error']=error
            errors.append(dict(frame_id=frame.get('frame_id'),error=error,evidence=item['evidence']));continue
        if payload.get('id')!=rid:
            raise CorpusError('Authorized frame reader returned a different response identity')
        path=folder/(str(item['evidence']['line']).zfill(6)+'.json')
        atomic_write_json(path,dict(payload=payload,source='production_ledger_authorized_hydration',
            captured_at=None,observed_ns=None,analysis_sequence_order=item['evidence']['line'],
            audit_created_at=utc_now(),ledger_evidence=item['evidence'],
            snapshot_manifest=frame.get('external_snapshots'),historical_observation_timestamp_unknown=True))
        info.update(status='hydrated',capture_path=str(path));paths.append(path)
    return paths,frame_index,errors


def configuration(frame, history):
    request=frame.get('request') or {}
    controls=frame.get('controls') or {}
    fields=dict(frame_version=frame.get('frame_version'),recorded_controls=controls,
                request_controls={k:request[k] for k in ('ghost_mode','developer_flags','ghost_preferences') if k in request})
    return dict(fingerprint=stable_digest(fields),**fields,
                request_day=(request_start(history).get('at') or 'unknown')[:10],
                runtime_build_identity=None,
                limitation='Recorded configuration and date stratum, not proof of identical runtime build or a controlled performance comparison.')


def frame_status(frame):
    state=frame.get('current_state') or {}
    return state.get('lifecycle_state') or ((state.get('status_semantics') or {}).get('canonical_lifecycle_state')) or 'unknown'


def instrumentation_summary(items, analysis):
    """Describe observed coverage, never infer an instrumentation epoch by date."""
    unique = {r['event_id']: r for r in items if isinstance(r, dict)
              and r.get('schema') == 'ollmo.causal_event.v1' and isinstance(r.get('event_id'), str)}
    boots = sorted({r['process_boot_id'] for r in unique.values() if r.get('process_boot_id')})
    gaps = [r for r in unique.values() if r.get('record_kind') == 'coverage_gap']
    dropped = {}
    for gap in gaps:
        scope = gap.get('scope_id', gap['event_id'])
        dropped[scope] = max(dropped.get(scope, 0), gap.get('dropped_count', 0))
    conflicts = set(analysis['event_id_conflicts'])
    finalizers = [r for r in analysis['invocations'] if r.get('owner') == 'response_frame.finalize'
                  and r['event_id'] not in conflicts and r.get('duration_seconds') is not None]
    return dict(cohort='instrumented' if boots else 'historical_or_uninstrumented',
        process_boot_ids=boots, retained_event_count=len(unique), coverage_gap_events=gaps,
        reported_dropped_count_lower_bound=sum(dropped.values()), event_id_conflicts=sorted(conflicts),
        incomplete_records=sum(bool(r.get('coverage_incomplete')) for r in unique.values()),
        complete_input_invocations=sum(r.get('input_coverage_complete') is True for r in analysis['invocations']),
        finalizer_invocations=finalizers,
        finalizer_seconds=sum(r['duration_seconds'] for r in finalizers) if finalizers else None,
        interval_coverage='incomplete' if gaps else 'not_established',
        limitation='Instrumentation presence proves recorded process identity, not a complete request trace or a controlled epoch comparison. Missing fields remain unknown.')


def instrumented_cohort(summaries):
    selected = [s for s in summaries if s.get('causal_coverage', {}).get('cohort') == 'instrumented']
    selected.sort(key=lambda s: (s['causal_coverage']['finalizer_seconds'] is None,
                                -(s['causal_coverage']['finalizer_seconds'] or 0), s['response_id']))
    counts = Counter(); stages = {}; ranked = []
    for rank, summary in enumerate(selected, 1):
        coverage = summary['causal_coverage']
        counts.update(summary['instrumented_classification_counts'])
        ranked.append(dict(rank=rank, response_id=summary['response_id'], details_path=summary['details_path'],
            finalizer_seconds=coverage['finalizer_seconds'],
            finalizer_invocation_count=len(coverage['finalizer_invocations']),
            observed_invocation_count=summary['instrumented_invocation_count'],
            process_boot_ids=coverage['process_boot_ids'],
            reported_dropped_count_lower_bound=coverage['reported_dropped_count_lower_bound'],
            instrumented_classification_counts=summary['instrumented_classification_counts']))
        # Use each deduplicated completion once. Recursive/nested stage timers overlap.
        for call in coverage['finalizer_invocations']:
            for name, value in (call.get('operations') or {}).items():
                if not isinstance(value, dict) or number(value.get('elapsed_ns')) is None:
                    continue
                key = (name, value.get('role', 'unknown'))
                stage = stages.setdefault(key, dict(name=name, role=key[1], elapsed_seconds=0,
                    calls=0, failures=0, invocation_ids=[], inclusive=True))
                stage['elapsed_seconds'] += value['elapsed_ns'] / 1e9
                stage['calls'] += value.get('calls', 0)
                stage['failures'] += value.get('failures', 0)
                stage['invocation_ids'].append(call['invocation_id'])
    return dict(response_count=len(selected), historical_or_uninstrumented_count=len(summaries)-len(selected),
        ranked_responses=ranked, classification_counts={name: counts[name] for name in CLASSES},
        finalizer_invocation_count=sum(r['finalizer_invocation_count'] for r in ranked),
        finalizer_seconds=sum(r['finalizer_seconds'] or 0 for r in ranked),
        finalizer_operations=sorted(stages.values(), key=lambda s: -s['elapsed_seconds']),
        reported_dropped_count_lower_bound=sum(r['reported_dropped_count_lower_bound'] for r in ranked),
        process_boot_ids=sorted({boot for r in ranked for boot in r['process_boot_ids']}),
        comparison='Descriptive cohort only. Higher confidence in exact invocation/timing attribution, not complete causality. Pre/post epochs are not a controlled performance comparison.')


def render_instrumented_cohort(cohort):
    lines = ['', '## Post-instrumentation production findings', '',
        f"{cohort['response_count']} instrumented ordinary responses; {cohort['historical_or_uninstrumented_count']} historical or uninstrumented ordinary responses. Membership requires retained causal schema and process boot identity; request date is not used as a proxy.", '',
        cohort['comparison'], '',
        'Ranked by exact completed finalizer inclusive effort. Missing finalizer timing stays unknown. The full-ledger ranking above retains all ordinary responses.', '',
        '| Rank | Response | Finalizer seconds | Finalizers | Observed calls | Reported dropped observations |',
        '|---:|---|---:|---:|---:|---:|']
    for r in cohort['ranked_responses']:
        seconds = 'unknown' if r['finalizer_seconds'] is None else f"{r['finalizer_seconds']:.3f}"
        lines.append(f"| {r['rank']} | [`{r['response_id']}`]({r['details_path']}) | {seconds} | {r['finalizer_invocation_count']} | {r['observed_invocation_count']} | {r['reported_dropped_count_lower_bound']} |")
    lines += ['', '### Proven instrumented classifications', '',
        'Counts below use the unchanged convergence proof gates. They are separate from legacy timing witnesses, which may describe the same work. Unknown includes first invocations and incomplete repeat/wait evidence; it is not a waste count.', '',
        '| Classification | Causal witnesses |', '|---|---:|']
    lines.extend(f'| {name} | {count} |' for name, count in cohort['classification_counts'].items())
    lines += ['', '### Newly instrumented persistence and finalization', '',
        f"{cohort['finalizer_invocation_count']} exact finalizer completions retain {cohort['finalizer_seconds']:.3f} seconds of inclusive effort. Stage rows preserve actual owner roles and counts. Canonical truth/durability and derived/readiness/index work remain distinguished by those roles.", '',
        'All operation timers are inclusive. Recursive snapshot splitting and nested helpers can exceed the enclosing elapsed duration when summed; do not add rows or subtract them to invent exclusive time or recoverable latency. Derived work occurring here is not thereby proven removable. Per-response JSON retains every invocation, process, target, operation and causal event identity.', '',
        '| Operation | Recorded role | Inclusive seconds | Calls | Failures |', '|---|---|---:|---:|---:|']
    for stage in cohort['finalizer_operations']:
        lines.append(f"| {stage['name']} | {stage['role']} | {stage['elapsed_seconds']:.6f} | {stage['calls']} | {stage['failures']} |")
    lines += ['', '## Remaining unknowns', '',
        f"The instrumented cohort explicitly reports at least {cohort['reported_dropped_count_lower_bound']} dropped observations. Absent historical telemetry remains unknown. No full trace coverage is inferred when a gap marker is absent; the persisted diagnostic tail is bounded.", '',
        'Most owner input selections remain partial; call-stack parentage alone does not explain semantic relevance. Unbound events are not assigned by timestamp. Missing complete eligibility predicates, producer/wake lineage, downstream consumption and negative interval evidence prevent assertions of redundant repeats, false waits, missed wake-ups, over-triggers or avoidable serialization. Zero proven witnesses does not establish absence.', '',
        'No new model work was launched. No historical ledger records or runtime behavior were changed.', '']
    return '\n'.join(lines)


def summarize_response(rid, items, history, events, monitor, output, frames_dir):
    paths,frames,errors=hydrate_response(rid,items,frames_dir,output)
    configuration_record=configuration(items[-1]['frame'],history)
    context=dict(case_id='production',response_id=rid,profile={'id':configuration_record['fingerprint'][:16]},
                 mode='production',manifest=str(output/'inventory.json'),category='ordinary_production',
                 state=frame_status(items[-1]['frame']))
    path=output/'responses'/(stable_digest(rid)[:24]+'.json');path.parent.mkdir(exist_ok=True)
    analyze_response(rid,paths,[context],[],path)
    report=json.loads(path.read_text())
    event_result=event_analysis(events)
    from scripts.self_attack_convergence import analyze_causal_records
    captured_events = report.pop('retained_causal_events', [])
    retained_events = captured_events + [e['record']['causal_event'] for e in events if isinstance(e['record'].get('causal_event'), dict)]
    report['causal_analysis'] = analyze_causal_records(retained_events)
    report['causal_coverage'] = instrumentation_summary(retained_events, report['causal_analysis'])
    report.update(kind=KIND,ledger_frames=frames,hydration_errors=errors,configuration=configuration_record,
        history=history,events=events,monitor=monitor,
        latency=latency(history,events,items),event_analysis=event_result,
        final_lifecycle_state=frame_status(items[-1]['frame']),
        source_population='history_bound_user_turn' if any(h.get('role')=='user' for h in history) else 'ordinary_response_id_unconfirmed_user_origin')
    lock_records=lock_waits(report['batches'])
    unknown_waits=[dict(w,classification='unknown_wait') for w in report['waits']]
    report['wait_analysis']=dict(necessary_waits=lock_records,unknown_waits=unknown_waits,false_waits=[],
        coverage='These are retained lock/blocked witnesses, not a census of scheduler waits. Pending state without a wait predicate is unknown, never an inferred false wait.')
    batch_seconds=report['timing']['batch_seconds']
    lf_spans=[(timestamp(v['start_at']),timestamp(v['end_at'])) for v in report['timing']['late_fill_intervals']]
    report['latency_decomposition'] = dict(
        preparation_seconds=batch_seconds['planning_elapsed'],
        semantic_planning_review_seconds=None,
        branch_execution_effort_seconds=batch_seconds['execution_effort'],
        dependency_waiting_seconds=None,
        necessary_instance_lock_wait_effort_seconds=sum(w['duration_seconds'] for w in lock_records),
        repair_rebase_execution_seconds=None,
        late_fill_interval_union_seconds=union_ms(lf_spans) if lf_spans else None,
        post_branch_batch_tail_seconds=batch_seconds['after_last_branch'],
        closure_finalization_event_effort_seconds=event_result['finalizer_seconds'],
        persistence_and_readiness_nested_step_seconds=event_result['step_seconds'],
        historical_hydration_observer_seconds=None,unattributed_seconds=None,
        basis='These are separate/inclusive evidence scopes, not additive slices of one wall-clock budget. Missing global spans prevent a trustworthy remainder subtraction.')
    counts=Counter(i['classification'] for i in event_result['finalizer_invocations'])
    counts['necessary_wait']+=len(lock_records);counts['unknown_wait']+=len(unknown_waits)
    report['causal_classification_counts']={key:counts[key] for key in (*CLASSES,'unknown_wait')}
    report['unknowns']={key:'unknown: no complete invocation/trigger/input/authority/evidence/consumption witness retained'
        for key in ('Ghost_internal_order','actual_lens_passes','attention_trigger_relevance','aspiration_doubt_commitment_triggers',
                    'promotion_trigger_completeness','repair_rebase_trigger_completeness','missed_wakeup','over_trigger','avoidable_serialization','redundant_repeat','defensive_repeat','superseded_execution_cost')}
    atomic_write_json(path,report)
    return dict(response_id=rid,details_path=str(path),source_population=report['source_population'],
        title=str((items[0]['frame'].get('request') or {}).get('prompt','')).split('\n')[0][:140],
        final_lifecycle_state=report['final_lifecycle_state'],frame_count=len(items),hydrated_frames=len(paths),hydration_errors=errors,
        latency=report['latency'],configuration=configuration_record,timing=report['timing'],
        latency_decomposition=report['latency_decomposition'],
        finalizer_seconds=event_result['finalizer_seconds'],finalizer_step_seconds=event_result['step_seconds'],
        finalizer_invocation_count=len(event_result['finalizer_invocations']),requeues=len(event_result['requeues']),
        causal_classification_counts=report['causal_classification_counts'],
        instrumented_classification_counts=report['causal_analysis']['classification_counts'],
        instrumented_invocation_count=len(report['causal_analysis']['invocations']),
        causal_coverage=report['causal_coverage'],
        causal_model_invocation_count=report['causal_analysis']['model_invocation_count'],
        causal_read_model_invocation_count=report['causal_analysis']['read_model_invocation_count'],
        persistence_operations=report['causal_analysis']['operations'],
        projection_count=len(report['component_observations']),
        unchanged_projection_count=sum(c['projection_changed'] is False for c in report['component_observations']),
        evidence=[x['evidence'] for x in items])


def render_production(result):
    a=result['aggregate'];lines=['# Production-ledger convergence audit','',
        '**Knobs may change strategy, never truth.** Every semantic movement should be explainable by the change that made it newly relevant.','',
        f"Audited {a['ordinary_responses']} ordinary production response identities ({a['ordinary_frames']} ledger frames). {a['self_attack_responses']} explicit corpus responses are indexed separately, excluded from production timing totals. {a['history_bound_user_turns']} ordinary responses have a saved user request witness.",'',
        'Existing production state provides stronger evidence than capture-only analysis: response-bound request snapshots, typed retry events, and event-identified finalization step timers. It still does not generally identify individual Ghost/lens calls or their complete causal inputs. That is a limit on the audit, not a defect in Ollmo’s truth model.','',
        '## Slowest retained ordinary turns','',
        'Sorted by saved request snapshot to the last recorded terminal-owner checkpoint (or a labeled Late Fill completion boundary). This approximates a user’s elapsed turn span, not exact browser-visible latency or continuously busy runtime. Reopen/external pauses may be included. Missing endpoints remain unknown; assistant message timestamps and uncorrelated logs are not substitutes.','',
        '| Response | Request day / config | Elapsed span (min) | Boundary | Final ledger lifecycle | Details |',
        '|---|---|---:|---|---|---|']
    for r in result['ranked_responses'][:20]:
        seconds=r['latency']['seconds'];duration=f'{seconds/60:.2f}' if seconds is not None else 'unknown'
        lines.append(f"| `{r['response_id']}` | {r['configuration']['request_day']} / `{r['configuration']['fingerprint'][:12]}` | {duration} | {r['latency']['endpoint_kind']} | {r['final_lifecycle_state']} | [evidence]({r['details_path']}) |")
    lines += ['','## Measured latency investigations','',
        'Ranked by measured wall-time exposure and confidence, not event count. Recoverable time remains unknown. These entries do not recommend caching, parallelism, shorter timeouts or weaker truth gates.','',
        '| Rank | Component | Response | Seconds | Confidence | Evidence |','|---:|---|---|---:|---|---|']
    for f in result['findings'][:25]:
        lines.append(f"| {f['rank']} | {f['component']} | `{f['response_id']}` | {f['measured_seconds']:.3f} | {f['confidence']} | [details]({f['details_path']}) |")
    lines += ['','## Latency decomposition and overlap','',
        'Each response JSON separates batch preparation, branch execution effort, same-instance lock time, post-branch batch tail, Late Fill start/end intervals, event finalization and nested persistence/readiness steps. Semantic planning/review, dependency waiting, repair/rebase execution, browser rendering and unattributed wall time remain null where there is no exclusive timer. Hydration performed by this audit is separately labeled and never counted as historical runtime overhead.','',
        'Batch execution effort can overlap across branches; batch wall includes callback drain; finalization steps are inside finalizer totals. Historical batches may be inherited or reset across successor frames: conflicting ordinals remain excluded from shared aggregate sums. No totals combine these overlapping populations. Response-bound finalizer event IDs deduplicate events; frame copies of the same timing are not added again.','',
        f"Recorded production finalizer calls: {a['finalizer_invocations']}; inclusive finalizer effort: {a['finalizer_seconds']:.3f} seconds across separate responses/events. No claim that this sum is elapsed wall time for the corpus.",'',
        '| Nested finalization step | Recorded seconds |','|---|---:|']
    for k,v in sorted(a['finalizer_step_seconds'].items(),key=lambda kv:-kv[1]):lines.append(f'| {k} | {v:.3f} |')
    lines += ['','## Causal classifications','',
        '| Classification | Proven witness count |','|---|---:|']
    for k,v in a['causal_classification_counts'].items():lines.append(f'| {k} | {v} |')
    lines += ['','Necessary finalizer repeats require a distinct non-skipped invocation event and changed relevant branch/lifecycle input recorded by that owner. This justifies reconsideration, not every internal step or its duration. Necessary lock waits require an explicit same-instance policy and a recorded overlapping holder. Unknown counts cover timed finalizers/blocked records; projected semantic state is a separate population. A zero count is not proof that the behavior never occurred.','',
        f"{a['requeues']} retry/requeue events retain concrete owner reasons; many omit exact branch and attempt identity. They are not declared redundant merely because repair recurs. {a['unchanged_projection_count']} unchanged projected component observations are not treated as actual lens/model invocations.",'',
        '## Semantic movement and wait boundaries','',
        'The per-response component timeline indexes lenses, attention, aspiration, doubt/quality, commitment, possibility/promotion, obligations, phases, branches, repair/rebase, Late Fill, evidence and closure as retained state. It preserves graph/frame order and source pointers, while invocation fields remain null when unsupported. Promotion cannot be inferred from aspiration; lens state cannot grant execution authority; rebase previews are not applications.','',
        'A blocked/pending record alone does not prove a scheduler wait. Each available wait record retains its reason, target and declared dependency state; an absent complete wake predicate is `unknown_wait`. Producer availability alone cannot prove that every consumer contract/authority gate is satisfied. No false waits, missed wake-ups, over-triggers or avoidable serialization are asserted without that evidence.','',
        'Events without exact response binding are counted as uncorrelated and never assigned by timestamp proximity. Monitor records are attached only by response/frame identity and remain observations. HTTP status/debug/truth reads are auxiliary, not semantic-execution counts.','',
        '## Comparison with the self-attack audit','',result['comparison']['description'],'',
        'Production and self-attack costs remain separate; configurations/date strata are not a controlled comparison and immutable build epochs are usually absent.','',
        '## Smallest optional instrumentation','']
    for rec in INSTRUMENTATION:lines.append(f"- **{rec['owner']}**: {rec['add']}")
    lines += ['','Prioritize adding branch/attempt/frame and triggering event IDs to existing response-bound retry/finalization events, then exact wait predicate and wake transition. Keep existing finalization step timers; extend them only to remaining expensive gaps. No separate scheduler, authority or live validation was introduced.','',
        '## Evidence integrity and scope','',
        f"Hydration/integrity errors: {a['hydration_errors']}; malformed input records: {len(result['input_errors'])}. Missing/corrupt sidecars or parent manifests remain explicit unknowns. Audit-owned reconstructed captures preserve lineage to copied ledger rows and authorized manifests. The production ledger/index/artifacts were not written.",'',
        'See `inventory.json`, `state-evidence.json`, `auxiliary-evidence.json`, `raw-frames/`, `captures/`, and `responses/` for hashes, byte offsets, JSON pointers, exact IDs and source provenance. `process.json`, `progress.json`, `completion.json`, and `stdout-stderr.log` preserve detached execution status.','']
    from scripts.self_attack_convergence import render_causal_summary
    return ('\n'.join(lines) + render_causal_summary(result.get('ranked_responses', []))
            + render_instrumented_cohort(result['instrumented_cohort']))


def run_production_audit(frames_dir, output, *, repo=None):
    frames_dir=Path(frames_dir).resolve();output=Path(output).resolve()
    repo=Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    state_dir=frames_dir.parent
    if output==frames_dir or output.is_relative_to(frames_dir):raise CorpusError('Audit output must not be inside production frames')
    output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(output/'progress.json',dict(stage='inventory',pid=os.getpid(),started_at=utc_now(),
        results_path=str(output/'results.json'),report_path=str(output/'report.md')))
    raw_rows,ledger_meta,input_errors=read_jsonl(frames_dir/'responses.jsonl')
    if ledger_meta.get('missing'):raise CorpusError('Production response ledger not found')
    history,history_files,history_errors=index_history(state_dir);input_errors+=history_errors
    event_rows,event_meta,errors=read_jsonl(state_dir/'events.jsonl');input_errors+=errors
    monitor_rows,monitor_meta,errors=read_jsonl(state_dir/'ollmo_run_monitor/reports.jsonl');input_errors+=errors
    events=defaultdict(list);monitors=defaultdict(list);unbound=Counter()
    for event,evidence,_ in event_rows:
        if event.get('response_id'):events[event['response_id']].append(dict(record=event,evidence=evidence))
        else:unbound[f"{event.get('category')}/{event.get('action')}"]+=1
    for event,evidence,_ in monitor_rows:
        if event.get('response_id'):monitors[event['response_id']].append(dict(record=event,evidence=evidence))
    grouped=defaultdict(list);raw_dir=output/'raw-frames';raw_dir.mkdir(exist_ok=True)
    for frame,evidence,raw in raw_rows:
        rid=frame.get('response_id')
        if not rid:input_errors.append(dict(evidence=evidence,error='missing response id'));continue
        path=raw_dir/(str(evidence['line']).zfill(6)+'.json');path.write_bytes(raw)
        grouped[rid].append(dict(frame=frame,evidence=evidence,raw_copy=str(path)))
    del raw_rows
    source_files=[Path(__file__),Path(__file__).with_name('self_attack_convergence.py'),Path(__file__).with_name('ollmo_self_attack.py'),
                  repo/'ollmo_services/response_frames.py']
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.is_file()}
    source_id=stable_digest(source_hashes)
    input_id=stable_digest([ledger_meta,event_meta,monitor_meta,history_files])
    manifest_path=output/'run.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old.get('source_id')!=source_id or old.get('input_id')!=input_id:raise CorpusError('Audit source/input changed; use a new output directory')
    atomic_write_json(manifest_path,dict(kind=KIND,source_id=source_id,input_id=input_id,started_at=utc_now(),authority='offline_diagnostic_only'))
    atomic_write_json(output/'inventory.json',dict(ledger=ledger_meta,events=event_meta,monitor=monitor_meta,history_files=history_files,
                     source_hashes=source_hashes,input_errors=input_errors))
    atomic_write_json(output/'state-evidence.json',dict(events={k:v for k,v in events.items() if k in grouped},
        uncorrelated_event_counts=dict(unbound),history={k:v for k,v in history.items() if k in grouped}))
    ordinary={rid:items for rid,items in grouped.items() if not rid.startswith('resp_corpus_') and not any(
        ((i['frame'].get('request') or {}).get('request_meta') or {}).get('source')=='graph_rebase_shadow_corpus_runner' for i in items)}
    excluded={rid:[i['evidence'] for i in items] for rid,items in grouped.items() if rid not in ordinary}
    ordered=sorted(ordinary,key=lambda rid:(not any(h.get('role')=='user' for h in history[rid]),
                   -(latency(history[rid],events[rid],ordinary[rid])['seconds'] or 0),rid))
    summaries=[];started=time.monotonic()
    for i,rid in enumerate(ordered):
        if (output/'STOP').exists():
            atomic_write_json(output/'progress.json',dict(stage='stopped',completed=i,total=len(ordered),pid=os.getpid(),finished_at=utc_now()))
            return 130
        atomic_write_json(output/'progress.json',dict(stage='production_responses',completed=i,total=len(ordered),response_id=rid,pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started,results_path=str(output/'results.json'),report_path=str(output/'report.md')))
        summaries.append(summarize_response(rid,ordinary[rid],history[rid],events[rid],monitors[rid],output,frames_dir))
        atomic_write_json(output/'checkpoint.json',dict(responses=summaries,completed=i+1,total=len(ordered)))
        print(f'{utc_now()} production {i+1}/{len(ordered)} {rid}',flush=True)
    counts=Counter();steps=Counter()
    findings=[]
    for s in summaries:
        counts.update(s['causal_classification_counts']);steps.update(s['finalizer_step_seconds'])
        for component,seconds in [('finalization_inclusive',s['finalizer_seconds']),
                                  ('post_branch_batch_tail',s['timing']['batch_seconds']['after_last_branch'])]:
            if seconds>0:
                findings.append(dict(response_id=s['response_id'],component=component,classification='unknown',
                    measured_seconds=seconds,estimated_removable_seconds=None,confidence='high',
                    finding_type='measured_cost_investigation',details_path=s['details_path'],
                    configuration=s['configuration'],evidence=s['evidence'],
                    interpretation='Measured owner cost; redundancy or avoidability is not established. Nested or overlapping with other listed durations.'))
    findings.sort(key=lambda x:-x['measured_seconds'])
    for i,f in enumerate(findings,1):f['rank']=i
    prior=repo/'state/self_attack/convergence-audit-20260906-verified/results.json'
    comparison=dict(source_path=str(prior),description='Prior self-attack analysis is unavailable in this checkout.')
    if prior.is_file():
        p,digest=read_json(prior)
        comparison.update(sha256=digest,prior_aggregate=p.get('aggregate'),description=
            'The earlier self-attack audit measured batch tails and recorded unchanged projections without proving semantic redundancy. Production state adds response-bound request times, explicit retry reasons and event-identified finalization steps, so some of those costs can now be attributed to persistence/readiness owners and some repeats to changed runtime state. Individual lens/Ghost input and trigger provenance remains largely unknown.')
    auxiliary=auxiliary_evidence(repo,[],set(ordinary),output/'auxiliary-evidence.json')
    latest=ledger_meta.copy();now=(frames_dir/'responses.jsonl').stat()
    latest['unchanged_through_audit']=(now.st_size,now.st_mtime_ns)==(ledger_meta['bytes'],ledger_meta['mtime_ns'])
    aggregate=dict(ordinary_responses=len(summaries),ordinary_frames=sum(s['frame_count'] for s in summaries),
        history_bound_user_turns=sum(s['source_population']=='history_bound_user_turn' for s in summaries),self_attack_responses=len(excluded),
        hydration_errors=sum(len(s['hydration_errors']) for s in summaries),finalizer_invocations=sum(s['finalizer_invocation_count'] for s in summaries),
        finalizer_seconds=sum(s['finalizer_seconds'] for s in summaries),finalizer_step_seconds=dict(steps),
        causal_classification_counts={k:counts[k] for k in (*CLASSES,'unknown_wait')},requeues=sum(s['requeues'] for s in summaries),
        unchanged_projection_count=sum(s['unchanged_projection_count'] for s in summaries))
    result=dict(kind=KIND,status='completed_with_observability_limits',finished_at=utc_now(),aggregate=aggregate,
        ranked_responses=sorted(summaries,key=lambda s:-(s['latency']['seconds'] or 0)),findings=findings,
        excluded_self_attack=excluded,input_errors=input_errors,ledger_integrity=latest,auxiliary=auxiliary,comparison=comparison,
        instrumentation=INSTRUMENTATION,new_model_work=False,runtime_behavior_changed=False)
    result['instrumented_cohort'] = instrumented_cohort(summaries)
    atomic_write_json(output/'results.json',result);(output/'report.md').write_text(render_production(result))
    atomic_write_json(output/'progress.json',dict(stage='complete',completed=len(ordered),total=len(ordered),pid=os.getpid(),
        elapsed_seconds=time.monotonic()-started,results_path=str(output/'results.json'),report_path=str(output/'report.md')))
    print(f'Production audit complete: {output / "report.md"}',flush=True)
    return 0
