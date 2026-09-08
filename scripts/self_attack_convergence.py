"""Offline convergence evidence audit; no runtime execution or mutation authority."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from scripts.run_graph_rebase_shadow_corpus import atomic_write_json, stable_digest, utc_now, CorpusError
from scripts.self_attack_checks import records

SCHEMA = 'ollmo.self_attack.convergence.v1'
CLASSES = ('necessary_repeat', 'defensive_repeat', 'redundant_repeat', 'false_wait',
           'necessary_wait', 'avoidable_serialization', 'missed_wakeup', 'over_trigger', 'unknown')
# These are projection owners, never presumed model invocations.
SURFACES = {
    'ghost_resolution': 'Ghost', 'semantic_role_profile': 'lens_selection',
    'execution_planner': 'resolver_planning',
    'candidate_graph': 'possibility', 'promotion_review': 'promotion',
    'controlled_attention_review': 'attention', 'semantic_review_lens_review': 'lenses',
    'semantic_role_orientation_review': 'lenses', 'aspiration_review': 'aspiration',
    'commitment_review': 'commitment', 'semantic_quality_review': 'doubt_quality',
    'semantic_decision_review': 'semantic_decision', 'active_reconsideration_review': 'reconsideration',
    'recursive_cycle_review': 'branch_cycles', 'graph_closure_review': 'closure',
    'global_semantic_closure_review': 'semantic_closure', 'intent_lens_review': 'intent_lenses',
    'redraw_scope_ladder_review': 'structural_zoom', 'graph_patch_lifecycle': 'graph_repair',
    'graph_patch_lifecycle_results': 'graph_repair', 'runtime_graph_rebase_candidate_review': 'graph_rebase',
    'graph_rebase_reviews': 'graph_rebase', 'graph_rebase_lifecycle': 'graph_rebase',
    'graph_repair_proposals': 'graph_repair', 'graph_repair_reviews': 'graph_repair',
    'applied_graph_patches': 'graph_repair_application', 'graph_rebase_proposals': 'graph_rebase_preview',
    'applied_graph_rebases': 'graph_rebase_application', 'successor_reopen_requests': 'successor_reopen',
    'successor_rebase_requests': 'successor_rebase',
    'request_phase_graph_refinements': 'graph_refinement', 'embedding_audit': 'evidence',
}
INSTRUMENTATION = [
    {'owner': 'Ghost / semantic review / decision-contract builders',
     'add': 'One invocation id and parent/trigger event id per actual call, owner+target+frame/graph version, trigger reason and changed relevant keys, owner-defined effective-input digest, evidence-set digest and authority/contract digest. Do not use output or whole-snapshot hashes as input hashes.'},
    {'owner': 'Lens / attention / aspiration / doubt / commitment / promotion',
     'add': 'Attach selected lens, attention target, semantic-depth and structural-scope identities to that same invocation; distinguish projection rebuild from model execution. Record required defensive-contract id when revalidation is compulsory.'},
    {'owner': 'Late Fill / dependency scheduler / repair / rebase',
     'add': 'On actual wait entry/exit record wait id, exact unresolved predicate and producer identity+version, eligibility/authority gates, wake event id, ready/enqueued/dequeued times, queue and instance-lock reason, cancellation/supersession identity. Use monotonic timestamps plus process/boot identity.'},
    {'owner': 'Closure / evidence / response-frame finalization / observer',
     'add': 'Carry the same invocation id through start/end, output/judgment digest, state delta, created/unblocked/consumed downstream ids and stale-result disposition. Time Closure, graph rebuild, callback persistence, hydration, serialization, client transfer and capture separately; retain volatile timestamps outside normalized sidecars.'},
]


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.timestamp() if parsed.tzinfo else None
    except ValueError:
        return None


def interval(start, end):
    a, b = timestamp(start), timestamp(end)
    return round(b - a, 6) if a is not None and b is not None and b >= a else None


def union_ms(spans):
    total, end = 0.0, None
    for a, b in sorted(spans):
        total += max(0, b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total


def timing_identity(value):
    """Empty-field omission is normal projection compaction, not another timer.

    This identity only deduplicates retained timing witnesses. It must never be
    used as an effective-input, evidence-set or authority hash.
    """
    if isinstance(value, dict):
        return {k: timing_identity(v) for k, v in value.items() if v not in (None, {}, [], '')}
    if isinstance(value, list):
        return [timing_identity(v) for v in value]
    return value


def classify_repeat(previous, current):
    """Proof gate, including negative evidence. Unknown is the default, not a bug."""
    required = ('effective_input_hash', 'authority_hash', 'evidence_hash')
    if not previous or not all(previous.get(k) and current.get(k) for k in required):
        return 'unknown'
    if not current.get('input_coverage_complete') or not previous.get('input_coverage_complete'):
        return 'unknown'
    if any(previous[k] != current[k] for k in required):
        return 'necessary_repeat' if current.get('relevant_change_proven') is True else 'unknown'
    if current.get('required_defensive_contract'):
        return 'defensive_repeat'
    if (current.get('evidence_interval_complete') is True
            and current.get('no_required_downstream_role_proven') is True
            and current.get('authority_contract_checked') is True):
        return 'redundant_repeat'
    return 'unknown'


def causal_records(payload):
    """Read only named retained observer surfaces; never infer calls from lenses."""
    frame = payload.get('response_frame') or {}
    runtimes = [payload.get('runtime'), frame.get('runtime'),
                (frame.get('current_state') or {}).get('runtime')]
    result = []
    for runtime in runtimes:
        if not isinstance(runtime, dict):
            continue
        telemetry = (runtime.get('developer_diagnostics') or {}).get('causal_telemetry') or {}
        if not isinstance(telemetry, dict):
            continue
        result.extend(records(telemetry.get('events')))
    return result


def analyze_causal_records(items):
    """Exact-ID joins only. Negative claims need complete owner/interval proof."""
    unique, conflicts = {}, set()
    for item in items:
        if not isinstance(item, dict) or item.get('schema') != 'ollmo.causal_event.v1' or not isinstance(item.get('event_id'), str):
            continue
        event_id = item['event_id']
        if event_id in unique and stable_digest(unique[event_id]) != stable_digest(item):
            conflicts.add(event_id)
        unique[event_id] = item
    completed = {}
    for item in unique.values():
        if (item.get('status') in ('returned', 'raised') and isinstance(item.get('invocation_id'), str)
                and item['event_id'] not in conflicts):
            key = item['invocation_id']
            if key in completed and completed[key] != item:
                conflicts.update((item['event_id'], completed[key]['event_id']))
            completed[key] = item
    calls = []
    for item in completed.values():
        previous = completed.get(item.get('previous_invocation_id'))
        exact_scope = bool(previous and item.get('target') and item.get('owner')
            and item.get('scope_id') and item.get('process_boot_id')
            and not previous.get('coverage_incomplete') and not item.get('coverage_incomplete')
            and all(previous.get(key) == item.get(key) for key in
                    ('owner', 'target', 'scope_id', 'process_boot_id'))
            and isinstance(previous.get('end_monotonic_ns'), int)
            and isinstance(item.get('start_monotonic_ns'), int)
            and previous['end_monotonic_ns'] <= item['start_monotonic_ns']
            and previous['event_id'] not in conflicts and item['event_id'] not in conflicts)
        classification = classify_repeat(previous, item) if exact_scope else 'unknown'
        calls.append(dict(item, component=item.get('owner', 'unknown'), classification=classification,
            model_execution=item.get('record_kind') == 'model_invocation',
            witness_key=item['invocation_id'], evidence=['/payload/runtime/developer_diagnostics/causal_telemetry'],
            duration_seconds=((item['end_monotonic_ns'] - item['start_monotonic_ns']) / 1e9
                if isinstance(item.get('end_monotonic_ns'), int)
                and isinstance(item.get('start_monotonic_ns'), int)
                and item['end_monotonic_ns'] >= item['start_monotonic_ns'] else None)))
    waits = []
    for item in unique.values():
        if item.get('record_kind') != 'wait_wake':
            continue
        start = unique.get(item.get('wait_id'))
        producer = unique.get(item.get('wake_event_id'))
        # Only the mutex predicate is closed here; no broader eligibility claim.
        necessary = bool(start and producer and item.get('predicate') == 'acquire_same_instance_mutex'
            and start.get('record_kind') == 'wait_started'
            and producer.get('record_kind') == 'release_boundary'
            and item.get('predicate_satisfied') is True
            and item.get('required_rule') == 'same_instance_mutex'
            and start.get('target') == item.get('target')
            and producer.get('target') != item.get('target')
            and all(start.get(k) == item.get(k) == producer.get(k) for k in
                    ('scope_id', 'process_boot_id', 'producer_instance_id'))
            and all(isinstance(r.get('monotonic_ns'), int) for r in (start, producer, item))
            and start.get('monotonic_ns', 0) < producer.get('monotonic_ns', 0) <= item.get('monotonic_ns', 0)
            and not any(r.get('coverage_incomplete') for r in (start, producer, item))
            and not {start['event_id'], producer['event_id'], item['event_id']} & conflicts)
        waits.append(dict(item, classification='necessary_wait' if necessary else 'unknown',
                          classification_scope='named_mutex_predicate_only' if necessary else 'unknown'))
    operations = []
    for call in calls:
        stages = call.get('operations')
        for name, stage in (stages.items() if isinstance(stages, dict) else []):
            if not isinstance(stage, dict):
                continue
            operations.append(dict(stage, name=name, invocation_id=call['invocation_id'],
                                   owner=call.get('owner', 'unknown'), inclusive=True))
    return dict(invocations=calls, waits=waits, operations=operations,
                model_invocation_count=sum(c['model_execution'] for c in calls),
                read_model_invocation_count=sum(c.get('record_kind') == 'read_model_invocation' for c in calls),
                event_id_conflicts=sorted(conflicts),
                classification_counts=dict(Counter(c['classification'] for c in calls + waits)),
                limitation='Missing, truncated, conflicting or unjoined provenance is unknown. Inclusive timers must not be added across nesting.')


def invocation(component, evidence, **extra):
    return dict(component=component, owner=component, record_kind='invocation_witness',
                classification='unknown', trigger_reason=None, triggering_state_delta=None,
                effective_input_hash=None, previous_effective_input_hash=None, authority_context=None,
                start_at=None, end_at=None, wait_reason=None, wake_condition=None,
                output_changed=None, runtime_state_changed=None, downstream_created_or_unblocked=None,
                superseded_by=None, evidence=[evidence], **extra)


def read_json(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('evidence changed during read')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def short_state(value):
    """Small evidence index; full semantics remain at the cited JSON pointer."""
    if isinstance(value, list):
        return {'record_count': len(value), 'records': [short_state(v) for v in value]}
    if not isinstance(value, dict):
        return value
    keep = ('kind', 'status', 'authority', 'runtime_effect', 'policy', 'reason', 'trigger',
            'branch_id', 'phase_id', 'task_id', 'candidate_id', 'obligation_id', 'proposal_id',
            'frame_id', 'frame_sequence', 'contract_state', 'required', 'depends_on', 'capability',
            'semantic_review_lens', 'lens', 'lens_id', 'selected_lens', 'target', 'target_id',
            'semantic_depth', 'structural_granularity', 'scale_movement', 'recommended_action',
            'recommended_transition', 'evidence_refs', 'allowed_transitions', 'repair_action',
            'superseded_by', 'replacement_branch_id', 'output_type', 'source', 'verdict')
    result = {k: value[k] for k in keep if k in value}
    for key, item in value.items():
        if key.endswith('_count') or key.endswith('_counts'):
            result[key] = item
        if key in ('frames', 'decisions', 'proposals', 'candidates', 'checks', 'items', 'tasks',
                   'attention_review', 'aspiration_review', 'commitment_review', 'semantic_review_verdict',
                   'orientation', 'loop', 'attention_scope'):
            result[key] = short_state(item)
    return result


def projection_components(payload):
    runtime = payload.get('runtime') or {}
    frame = payload.get('response_frame') or {}
    if not isinstance(runtime, dict) or not isinstance(frame, dict):
        raise ValueError('Malformed runtime/frame mapping')
    if not runtime:
        runtime = (frame.get('current_state') or {}).get('runtime') or frame.get('runtime') or {}
    graph = runtime.get('request_phase_graph') or {}
    if not isinstance(graph, dict):
        raise ValueError('Malformed request graph mapping')
    containers = [('/runtime', runtime), ('/runtime/request_phase_graph', graph),
                  ('/runtime/request_phase_graph/decision_contract', graph.get('decision_contract') or {}),
                  ('/runtime/developer_diagnostics', runtime.get('developer_diagnostics') or {}),
                  ('/runtime/graph_closure_review', runtime.get('graph_closure_review') or {}),
                  ('/runtime/graph_closure_review/intent_graph_adequacy',
                   (runtime.get('graph_closure_review') or {}).get('intent_graph_adequacy') or {})]
    lf = payload.get('late_fill') or runtime.get('late_fill') or (frame.get('current_state') or {}).get('late_fill') or frame.get('late_fill') or {}
    if not isinstance(lf, dict) or any(not isinstance(obj, dict) for _, obj in containers):
        raise ValueError('Malformed semantic/Late Fill mapping')
    containers.append(('/late_fill', lf))
    result = {}
    for base, obj in containers:
        for key, component in SURFACES.items():
            if key in obj and obj[key] not in (None, {}, []):
                result[base + '/' + key] = (component, obj[key])
    for key in ('phases', 'downstream_branches', 'output_obligations', 'intent_obligations'):
        if key in graph:
            result['/runtime/request_phase_graph/' + key] = ('graph_' + key, graph[key])
    for key in ('pending_branches', 'active_branches', 'completed_branches', 'failed_branches',
                'cancelled_branches', 'branch_controls', 'branch_progress'):
        if lf.get(key):
            result['/late_fill/' + key] = ('late_fill_' + key, lf[key])
    return result, runtime, graph, lf


def batch_analysis(batch, evidence, ordinal):
    branches = records(batch.get('branch_timings'))
    spans = []
    for branch in branches:
        start, elapsed = number(branch.get('queued_elapsed_ms')), number(branch.get('elapsed_ms'))
        if start is not None and elapsed is not None:
            spans.append((start, start + elapsed))
    elapsed = number(batch.get('elapsed_ms'))
    measured_union = union_ms(spans) if len(spans) == len(branches) and branches else None
    tail = None
    if elapsed is not None and measured_union is not None and max(b for _, b in spans) <= elapsed + 1:
        tail = max(0, elapsed - max(b for _, b in spans))
    witness_fields = ('branch_timings', 'prepare_timings', 'elapsed_ms', 'planning_elapsed_ms',
                      'execution_submission_order', 'worker_count', 'worker_count_source',
                      'instance_branch_groups', 'same_instance_lock_groups', 'branch_progress_dispatch')
    witness = {k: batch[k] for k in witness_fields if k in batch}
    # Candidate-diagnostic snapshots can evolve while the same timing stays retained.
    witness['prepare_timings'] = [{k: v for k, v in p.items() if k not in (
        'candidate_diagnostics', 'route_diagnostics', 'initial_route_diagnostics')}
        for p in records(batch.get('prepare_timings'))]
    return dict(ordinal=ordinal, evidence=evidence, batch_hash=stable_digest(timing_identity(witness)),
                retained_batch_hash=stable_digest(batch),
                execution_submission_order=batch.get('execution_submission_order'),
                worker_count=batch.get('worker_count'), worker_count_source=batch.get('worker_count_source'),
                instance_branch_groups=batch.get('instance_branch_groups'),
                same_instance_lock_groups=batch.get('same_instance_lock_groups'),
                branch_progress_dispatch=batch.get('branch_progress_dispatch'),
                wave_elapsed_ms=elapsed, planning_elapsed_ms=number(batch.get('planning_elapsed_ms')),
                branch_interval_union_ms=measured_union, after_last_branch_ms=tail,
                execution_effort_ms=sum(number(b.get('execution_ms')) or 0 for b in branches),
                lock_wait_effort_ms=sum(number(b.get('lock_wait_ms')) or 0 for b in branches),
                branch_timings=branches,
                prepare_timings=[{k: v for k, v in p.items() if k not in ('candidate_diagnostics', 'route_diagnostics', 'initial_route_diagnostics')}
                                 for p in records(batch.get('prepare_timings'))],
                attribution='Wave timer excludes preparation; includes result callbacks/drain. Tail is measured exposure, not removable latency.')


def extract_capture(payload, source, observed_at):
    components, runtime, graph, lf = projection_components(payload)
    frame = payload.get('response_frame') or {}
    result = dict(observed_at=observed_at, source=source, frame_id=frame.get('frame_id'),
                  frame_sequence=frame.get('frame_sequence'), lifecycle_state=payload.get('lifecycle_state'),
                  components=[], invocations=[], batches=[], waits=[], late_fill_interval=None,
                  causal_events=causal_records(payload))
    for pointer, (component, value) in components.items():
        result['components'].append(dict(component=component, pointer='/payload' + pointer,
            observed_state_hash=stable_digest(value), state=short_state(value), classification='unknown',
            record_kind='state_projection', effective_input_hash=None,
            reason='A retained projection does not establish invocation, triggering input or downstream consumption.'))
    for i, batch in enumerate(records(lf.get('materialization_concurrency_history'))):
        pointer = '/payload/late_fill/materialization_concurrency_history/' + str(i)
        result['batches'].append(batch_analysis(batch, pointer, i))
        for j, b in enumerate(records(batch.get('branch_timings'))):
            inv = invocation('multi_materialization_runtime.execute_plan', pointer + '/branch_timings/' + str(j),
                             branch_id=b.get('branch_id'), instance_id=b.get('instance_id'),
                             timing=b, ordinal=i)
            inv['trigger_reason'] = lf.get('trigger')
            inv['trigger_scope'] = 'Late Fill envelope; not a per-invocation trigger'
            inv['authority_context'] = {'recorded_batch_policy': {k: batch.get(k) for k in ('worker_count', 'worker_count_source', 'same_instance_lock_groups')}}
            inv['witness_key'] = stable_digest([i, b])
            graph_branch = next((v for v in records(graph.get('downstream_branches'))
                                 if v.get('branch_id') == b.get('branch_id')), {})
            matching_results = [v for v in records(lf.get('fill_results')) if v.get('branch_id') == b.get('branch_id')]
            inv['retained_branch_context'] = short_state(graph_branch)
            inv['branch_contract_projection_hash'] = stable_digest(graph_branch) if graph_branch else None
            inv['result_candidates'] = [dict(pointer='/payload/late_fill/fill_results/' + str(n),
                result_projection_hash=stable_digest(v), state=short_state(v))
                for n, v in enumerate(records(lf.get('fill_results'))) if v in matching_results]
            inv['result_binding_limit'] = 'Branch identity is retained; without attempt ids, a final fill result cannot be assigned to an exact invocation.'
            result['invocations'].append(inv)
        for j, preparation in enumerate(records(batch.get('prepare_timings'))):
            timing = {k: v for k, v in preparation.items() if k not in ('candidate_diagnostics', 'route_diagnostics', 'initial_route_diagnostics')}
            inv = invocation('multi_materialization_runtime.prepare_branch_plan',
                             pointer + '/prepare_timings/' + str(j), timing=timing,
                             branch_id=preparation.get('branch_id'), ordinal=i)
            inv['trigger_reason'] = preparation.get('initial_error_code')
            inv['trigger_scope'] = 'Initial preparation error when recorded; attempt_count does not establish individual attempt timings.'
            inv['witness_key'] = stable_digest([i, timing_identity(timing)])
            result['invocations'].append(inv)
    # Finalizer timing is an inclusive span, steps must not be added to its total.
    finalizer = (runtime.get('developer_diagnostics') or {}).get('response_frame_finalize_timing')
    if isinstance(finalizer, dict):
        inv = invocation('response_frame_finalize', '/payload/runtime/developer_diagnostics/response_frame_finalize_timing', timing=finalizer)
        inv['witness_key'] = stable_digest([frame.get('frame_id'), finalizer])
        inv['identity_limit'] = 'No invocation id: identical timing projections collapse; distinct timing values may be separate finalizations.'
        result['invocations'].append(inv)
    # Original frame metadata may retain timestamps stripped from normalized payloads.
    for location, item in [('/payload/response_frame/late_fill', frame.get('late_fill') or {}),
                           ('/payload/late_fill', lf)]:
        seconds = interval(item.get('started_at'), item.get('completed_at'))
        if seconds is not None:
            result['late_fill_interval'] = dict(start_at=item['started_at'], end_at=item['completed_at'], seconds=seconds, evidence=location)
            break
    phases = {p.get('phase_id') or p.get('id'): p for p in records(graph.get('phases'))}
    for i, branch in enumerate(records(lf.get('pending_branches')) + records(lf.get('failed_branches'))):
        reason = branch.get('wait_reason') or branch.get('blocked_reason') or branch.get('reason')
        wait = branch.get('availability_wait') or branch.get('dependency_wait')
        if not (reason or wait or branch.get('status') in ('waiting', 'availability_wait', 'blocked')):
            continue
        dependencies = branch.get('depends_on') or []
        result['waits'].append(dict(component='Late Fill', branch_id=branch.get('branch_id'),
            classification='unknown', wait_reason=reason, retained_wait=wait,
            wake_condition={'declared_dependencies': dependencies, 'predicate': None},
            dependency_states={dep: phases.get(dep, {}).get('status') for dep in dependencies},
            evidence='/payload/late_fill', reason='A blocked/pending record alone does not prove a scheduler wait or its complete wake predicate.'))
    return result


def discover(root, output):
    manifests, captures, supplemental, omitted = [], [], [], Counter()
    for parent, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('runtime', 'artifacts', 'snapshots', 'rechecks', 'reporting-backups')
                         and (Path(parent) / d).resolve() != output and not d.startswith('convergence-audit'))
        for name in sorted(files):
            path = Path(parent) / name
            if name == 'manifest.json' and (path.parent / 'captures').is_dir():
                manifests.append(path)
            elif name.endswith('.json') and 'captures' in path.parts and path.parent.name.startswith('resp'):
                captures.append(path)
            elif name.endswith('.json') and 'truth-fetch-failures' in path.parts:
                supplemental.append(path)
            else:
                omitted[path.suffix or 'no_extension'] += 1
    return manifests, captures, supplemental, dict(omitted)


def manifest_context(path):
    manifest, digest = read_json(path)
    worker_path = path.parent / 'worker.json'
    worker = read_json(worker_path)[0] if worker_path.exists() else {}
    epoch = {}
    # Only ascend within the established self-attack evidence hierarchy.
    for parent in path.parents:
        run = parent / 'run.json'
        if run.is_file():
            data, run_hash = read_json(run)
            epoch = dict(run_path=str(run), run_sha256=run_hash, source_digest=data.get('source_digest'))
            break
        if parent.name == 'self_attack':
            break
    return manifest, dict(path=str(path), sha256=digest, worker_path=str(worker_path) if worker else None,
                         mode=worker.get('mode', 'unknown'), profile=worker.get('profile', {}), epoch=epoch)


def context_for_case(case, origin):
    return dict(case_id=case.get('case_id'), response_id=case.get('response_id'), profile=origin['profile'],
                mode=origin['mode'], manifest=origin['path'], manifest_sha256=origin['sha256'],
                category=case.get('category'), state=case.get('state'),
                capture_source_epoch=origin.get('epoch'),
                dispatch_request_digest=case.get('dispatch_request_digest'),
                depends_on=case.get('depends_on'), source=case.get('provenance'))


def observation_record(case, origin):
    keys = ('submitting_at', 'submitted_at', 'post_finished_at', 'settled_at', 'last_observed_at',
            'status_observation_count', 'status_observations', 'final_debug', 'dispatch_unknown_at',
            'dispatch_unknown_reason', 'dependency_satisfied', 'stop_signals')
    record = {k: case[k] for k in keys if k in case}
    debug = record.get('final_debug') or {}
    record['final_debug'] = {k: v for k, v in debug.items() if k != 'summary'}
    record['final_debug']['summary_hash'] = stable_digest(debug.get('summary')) if debug.get('summary') else None
    record['manifest'] = origin['path']
    record['dispatch_to_settled_observation_seconds'] = interval(case.get('submitting_at'), case.get('settled_at'))
    record['debug_capture_wall_seconds'] = interval(debug.get('attempted_at'), debug.get('finished_at'))
    record['timing_limit'] = 'Observer interval includes client reads/capture/polling; not exact runtime completion or server-only latency.'
    return record


def analyze_response(response_id, paths, contexts, observations, out):
    unique, files, errors = {}, [], []
    for path in sorted(paths):
        try:
            before = path.stat()
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('moving capture')
            files.append(dict(path=str(path), sha256=digest, bytes=len(raw), mtime_ns=before.st_mtime_ns))
            if digest in unique:
                unique[digest]['aliases'].append(str(path))
                continue
            data = json.loads(raw)
            payload = data.get('payload')
            if not isinstance(payload, dict) or payload.get('id') != response_id:
                raise ValueError('capture payload/response identity mismatch')
            extracted = extract_capture(payload, data.get('source'), data.get('captured_at'))
            extracted.update(path=str(path), aliases=[], sha256=digest, observed_ns=data.get('observed_ns'),
                             analysis_sequence_order=data.get('analysis_sequence_order'))
            unique[digest] = extracted
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(dict(path=str(path), error=f'{type(exc).__name__}: {exc}'))
    snapshots = sorted(unique.values(), key=lambda x: (
        x.get('analysis_sequence_order') if x.get('analysis_sequence_order') is not None else x.get('observed_ns') or 0,
        x['path']))
    previous, invocations, batches, waits, transitions = {}, {}, {}, {}, []
    lf_intervals = {}
    for snap in snapshots:
        for c in snap['components']:
            key = c['pointer']
            old = previous.get(key)
            c['previous_observed_state_hash'] = old['hash'] if old else None
            c['projection_changed'] = old['hash'] != c['observed_state_hash'] if old else None
            if old:
                c['previous_evidence'] = old['evidence']
            previous[key] = dict(hash=c['observed_state_hash'], evidence=snap['path'] + '#' + key)
            transitions.append(dict(c, observed_at=snap['observed_at'], evidence=snap['path'] + '#' + key))
        for inv in snap['invocations']:
            key = (inv['component'], inv['witness_key'])
            refs = [snap['path'] + '#' + e for e in inv['evidence']]
            if key in invocations:
                invocations[key]['evidence'].extend(refs)
            else:
                invocations[key] = dict(inv, evidence=refs)
        for b in snap['batches']:
            key = (b['ordinal'], b['batch_hash'])
            evidence = snap['path'] + '#' + b['evidence']
            if key in batches:
                batches[key]['evidence'].append(evidence)
            else:
                batches[key] = dict(b, evidence=[evidence])
        for w in snap['waits']:
            evidence = snap['path'] + '#' + w['evidence']
            key = stable_digest(w)
            if key in waits:
                waits[key]['evidence'].append(evidence)
            else:
                waits[key] = dict(w, evidence=[evidence])
        lf = snap.get('late_fill_interval')
        if lf:
            key = (lf['start_at'], lf['end_at'])
            lf_intervals[key] = dict(lf, evidence=snap['path'] + '#' + lf['evidence'])
    # Ordinal collisions mean histories were revised/reset; expose alternatives, do not sum them.
    ordinal_counts = Counter(b['ordinal'] for b in batches.values())
    unambiguous_batches = [b for b in batches.values() if ordinal_counts[b['ordinal']] == 1]
    totals = {k.removesuffix('_ms'): round(sum(b.get(k) or 0 for b in unambiguous_batches) / 1000, 6)
              for k in ('wave_elapsed_ms', 'planning_elapsed_ms', 'execution_effort_ms',
                        'lock_wait_effort_ms', 'after_last_branch_ms')}
    timings = dict(batch_seconds=totals, unique_batch_witnesses=len(batches),
                   conflicting_history_ordinals=[i for i, n in ordinal_counts.items() if n > 1],
                   late_fill_intervals=list(lf_intervals.values()),
                   observed_dispatch_to_settled_seconds=max((o['dispatch_to_settled_observation_seconds'] or 0 for o in observations), default=0),
                   debug_capture_seconds=max((o['debug_capture_wall_seconds'] or 0 for o in observations), default=0),
                   removable_latency_seconds=None)
    causal = analyze_causal_records([event for snap in snapshots for event in snap.get('causal_events', [])])
    report = dict(kind=SCHEMA, causal_analysis=causal,
                  retained_causal_events=list({e['event_id']: e for snap in snapshots for e in snap.get('causal_events', []) if e.get('event_id')}.values()),
                  response_id=response_id, contexts=contexts, capture_files=files,
                  distinct_captures=len(unique), duplicate_capture_copies=len(files)-len(unique), errors=errors,
                  observations=observations, component_observations=transitions,
                  invocation_witnesses=list(invocations.values()), batches=list(batches.values()),
                  waits=list(waits.values()), timing=timings,
                  repeat_classification_limit='No invocation-specific complete input/evidence/authority and downstream provenance; repeated projections are not invocations.',
                  classification_counts=dict(Counter(i['classification'] for i in invocations.values())))
    atomic_write_json(out, report)
    return response_summary(report, out)


def response_summary(report, path):
    timing_evidence = {
        'batch': sorted({e for b in report['batches'] for e in b['evidence']}),
        'observer': sorted({o['manifest'] + '#/cases' for o in report['observations']})}
    return dict(response_id=report['response_id'], contexts=report['contexts'], details_path=str(path),
                causal_classification_counts=report.get('causal_analysis', {}).get('classification_counts', {}),
                causal_invocation_count=len(report.get('causal_analysis', {}).get('invocations', [])),
                causal_model_invocation_count=report.get('causal_analysis', {}).get('model_invocation_count', 0),
                causal_read_model_invocation_count=report.get('causal_analysis', {}).get('read_model_invocation_count', 0),
                timing_evidence=timing_evidence,
                capture_count=len(report['capture_files']), distinct_captures=report['distinct_captures'],
                duplicate_capture_copies=report['duplicate_capture_copies'], errors=report['errors'],
                timing=report['timing'], invocation_count=len(report['invocation_witnesses']),
                state_projection_count=len(report['component_observations']),
                unchanged_projection_observations=sum(c['projection_changed'] is False for c in report['component_observations']),
                observed_components=sorted({c['component'] for c in report['component_observations']}),
                classification_counts=report['classification_counts'], wait_count=len(report['waits']))


def ledger_index(paths, response_ids, output):
    """Stream raw frames as witnesses; never invoke runtime lookup or repair an index."""
    evidence, errors, files = defaultdict(list), [], []
    for path in paths:
        if not path.is_file():
            continue
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            offset = 0
            for lineno, raw in enumerate(stream, 1):
                digest.update(raw)
                # Only parse lines that can contain selected identities.
                # response_id starts early in some but not all ledger variants.
                if raw.strip():
                    try:
                        frame = json.loads(raw)
                        rid = frame.get('response_id') or (frame.get('current_state') or {}).get('id')
                        if rid in response_ids:
                            evidence[rid].append(dict(path=str(path), line=lineno, byte_offset=offset,
                                row_sha256=hashlib.sha256(raw).hexdigest(), frame_id=frame.get('frame_id'),
                                frame_sequence=frame.get('frame_sequence'), frame_relation=frame.get('frame_relation'),
                                status=frame.get('status'), late_fill=short_state(frame.get('late_fill') or {}),
                                snapshot_reference_count=len((frame.get('external_snapshots') or {}).get('items') or {})))
                    except (ValueError, TypeError) as exc:
                        errors.append(dict(path=str(path), line=lineno, error=str(exc)))
                offset += len(raw)
        after = path.stat()
        files.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=before.st_size,
                          stable_during_read=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)))
    value = dict(kind=SCHEMA, files=files, responses=dict(evidence), errors=errors,
                 limitation='Raw ledger metadata cross-check only. Hydrated captures supply graph state; no recursive production hydration or index attestation.')
    atomic_write_json(output, value)
    return dict(path=str(output), matched_response_count=len(evidence), frame_rows=sum(map(len, evidence.values())), files=files, errors=errors)


def rank_opportunities(summaries):
    opportunities = []
    for s in summaries:
        if not any(c['mode'] == 'live' for c in s['contexts']):
            continue
        t = s['timing']
        candidates = [
            ('post_execution_batch_tail', t['batch_seconds']['after_last_branch'],
             'Measured wave time after the last branch returned; callback drain/result publication is inside this owner timer. Closure/persistence breakdown is missing.', 'high'),
            ('observer_debug_capture', t['debug_capture_seconds'],
             'Timed debug/canonical companion capture path; may include full truth hydration, transfer and retained-artifact copying. This is audit observer latency, not demonstrated runtime blocking.', 'high'),
            ('branch_execution', t['batch_seconds']['execution_effort'],
             'Branch execution effort includes provider and branch-local acceptance work. Required output/evidence work cannot be labeled unnecessary from duration.', 'high'),
            ('serial_preparation', t['batch_seconds']['planning_elapsed'],
             'Measured batch preparation; source maintains per-group reservations/exclusions. No proof those decisions are independent.', 'medium'),
        ]
        for code, seconds, detail, confidence in candidates:
            if seconds <= 0:
                continue
            opportunities.append(dict(code=code, response_id=s['response_id'], contexts=s['contexts'],
                component={'post_execution_batch_tail':'multi_materialization_runtime callback drain',
                           'observer_debug_capture':'self-attack CaptureClient / shadow runner',
                           'branch_execution':'multi_materialization_runtime branch executor',
                           'serial_preparation':'multi_materialization_runtime preparation'}.get(code, 'Ghost → Late Fill → closure → observer'),
                classification='unknown', finding_type='instrumentation_opportunity',
                measured_exposure_seconds=seconds, estimated_recoverable_wall_seconds=None,
                timing_confidence=confidence, optimization_confidence='unknown', detail=detail,
                evidence=[s['details_path'] + '#/timing'],
                source_evidence=s['timing_evidence']['observer' if code == 'observer_debug_capture' else 'batch']))
    opportunities.sort(key=lambda x: (-x['measured_exposure_seconds'], {'high':0,'medium':1,'low':2}[x['timing_confidence']], x['response_id']))
    for i, item in enumerate(opportunities, 1):
        item['rank'] = i
    return opportunities


def render_causal_summary(summaries):
    counts = Counter()
    for summary in summaries:
        counts.update(summary.get('instrumented_classification_counts', summary.get('causal_classification_counts', {})))
    calls = sum(s.get('instrumented_invocation_count', s.get('causal_invocation_count', 0)) for s in summaries)
    models = sum(s.get('causal_model_invocation_count', 0) for s in summaries)
    rebuilds = sum(s.get('causal_read_model_invocation_count', 0) for s in summaries)
    lines = ['', '## Optional causal provenance', '',
             f'{calls} exact invocation identities, including {models} observed model calls and {rebuilds} deterministic read-model rebuilds.', '',
             'This population is separate from legacy timing witnesses to prevent double counting. Missing provenance is unknown. Per-response `causal_analysis` carries exact lineage, evidence limits, waits and inclusive persistence operation totals.', '',
             '| Classification | Causal witnesses |', '|---|---:|']
    lines.extend(f'| {name} | {counts[name]} |' for name in CLASSES)
    return '\n'.join(lines) + '\n'


def render_report(result):
    counts = result['aggregate']
    lines = ['# Retained convergence / trigger audit', '', '**Knobs may change strategy, never truth.**', '',
             'The retained evidence supports timing and state reconstruction, but does not prove redundant semantic work, false waits, missed wake-ups, over-triggers or avoidable serialization. No runtime optimization is justified by this audit alone.', '',
             f"Analyzed {counts['responses']} response identities, {counts['captures']} capture files ({counts['distinct_captures']} distinct byte identities; {counts['duplicate_capture_copies']} copied observations). Live, fake and unknown modes are separated below. No new model or API work ran.", '',
             '## Ranked latency investigations', '',
             'Ranking uses measured latency exposure, with timing confidence as a tie-breaker. Recoverable wall time is unknown for every entry. Execution effort and nested batch spans overlap and must not be added. Dispatch-to-observed-settlement envelopes remain in JSON but are excluded from ranking: delayed observation is not demonstrated runtime latency. These are investigation priorities, not optimization recommendations.', '',
             '| Rank | Exposure (s) | Timing confidence | Component / question | Response / profile | Evidence |',
             '|---:|---:|---|---|---|---|']
    for op in result['opportunities'][:30]:
        profiles = ', '.join(sorted({c['profile'].get('id','unknown') for c in op['contexts']}))
        lines.append(f"| {op['rank']} | {op['measured_exposure_seconds']:.3f} | {op['timing_confidence']} | {op['code']} | `{op['response_id']}` / {profiles} | [details]({op['evidence'][0].split('#')[0]}) |")
    lines += ['', 'Each details file carries concrete source capture paths, JSON pointers, input-file hashes, manifest/profile identities, component states, invocation witnesses and duration fields. Each machine finding also lists its original source evidence. The machine report includes every ranked entry, not only the first 30.', '',
              '## Timing by evidence mode', '', '| Mode | Responses | Branch execution effort (s) | Batch wave sum (s) | Preparation (s) | Post-branch tail (s) |', '|---|---:|---:|---:|---:|---:|']
    for mode, group in counts['by_mode'].items():
        b=group['batch_seconds']
        lines.append(f"| {mode} | {group['responses']} | {b['execution_effort']:.3f} | {b['wave_elapsed']:.3f} | {b['planning_elapsed']:.3f} | {b['after_last_branch']:.3f} |")
    lines += ['', 'Batch wave timers start after preparation and end after progress callbacks drain. Execution effort can overlap across branches. Post-branch tail is the interval after the last timed branch returns, not all non-provider work. Empty-field compaction does not create another timing witness; conflicting history ordinals are excluded from sums. Finalization timings are inclusive and remain separate. Late Fill runtime intervals and dispatch-to-settled observation envelopes are in each response report.', '',
              '## Necessary, defensive and redundant work', '',
              f"{counts['invocation_witnesses']} distinct invocation timing witnesses; all currently lack sufficient complete effective-input / authority / new-evidence / downstream-role provenance for repeat classification. {counts['unchanged_projection_observations']} unchanged component observations are explicitly **not** counted as redundant invocations.", '',
              '| Classification | Proven cycle count |', '|---|---:|']
    for name in CLASSES:
        lines.append(f"| {name} | {counts['classification_counts'].get(name, 0)} |")
    lines += ['', 'Unknown here counts invocation witnesses; unknown state projections and wait records are separate populations, not additional execution. A first required materialization has no repeat classification. Zero proven findings is not evidence of zero waste.', '',
              '## Causal boundaries and interaction coverage', '',
              '- Ghost order: request/case dependency order and retained route result are visible. Internal Ghost invocation order and per-call trigger deltas are not generally retained.',
              '- Lenses ↔ attention ↔ aspiration/doubt/commitment: selected/read-model state and changes are indexed; a frame or policy entry is not proof of a model pass. No pass count is inferred from projection multiplicity.',
              '- Possibility ↔ promotion ↔ graph/branches/phases: capture state and declared dependencies are preserved with pointers and hashes. Those are projected state identities, not effective invocation input identities.',
              '- Repair/rebase ↔ Late Fill ↔ evidence ↔ closure: lifecycle, refinement and closure surfaces are indexed where present. Branch histories retain timing/order and graph captures retain dependency edges. Rebase shadow evidence does not imply executable rebase.',
              '- Waits: pending/blocked records retain reason and dependencies where present. Without a complete wait predicate and waiter lifecycle, neither false_wait nor missed_wakeup is proven. Producer readiness alone does not prove all consumer gates satisfied.',
              '- Retries/requeues/supersession: branch timing history, prepare attempt counts, failures, cancellation and controls are retained. Historical graph/projection changes do not establish that a particular in-flight invocation was wasted or had no downstream effect.',
              '- Serialization: same-instance locks and preparation reservation policy can require ordering. Missing dependency edges or chronological order alone do not prove independence.',
              '- Evidence/truth/debug reads: manifest intervals and companion failures provide observer timing, not backend CPU or semantic invocation counts. Logs are auxiliary uncorrelated evidence unless an exact response id is present.', '',
              '## Smallest additional instrumentation', '']
    for item in INSTRUMENTATION:
        lines.append(f"- **{item['owner']}**: {item['add']}")
    lines += ['', 'Add this provenance to existing debug/frame/conformance capture records. Do not create another scheduler or authority. A future small representative rerun should select the slowest retained live dependency, repair and closure cases after instrumentation is approved; it was not launched.', '',
              '## Scope and omissions', '',
              f"Input/extraction errors: {counts['errors']}. Responses with revised/conflicting history ordinals: {counts['history_conflict_responses']}.",
              'Read every discovered versioned capture and sequence manifest. Large results/progress reports, retained artifact bytes, fake runtime sidecar trees and reporting backups are derived/duplicate evidence and are not recursively rehydrated. Raw production/fake ledger metadata is indexed separately where selected. Unmanifested ad-hoc investigation files are inventoried as a coverage limitation. No arbitrary prose mining supplies causal authority.',
              'Capture hashing verifies stable bytes during reads; this is an offline observation of an existing corpus, not a globally atomic snapshot. Ledger/log scans record stability. Historical source/profile provenance remains attached; evidence from different runtime epochs is not a controlled performance comparison.', '',
              'Machine outputs: `results.json`, `inventory.json`, `ledger-evidence.json`, `auxiliary-evidence.json`, `responses/*.json`; process/progress/completion and stdout/stderr paths are durable beside this report.', '']
    return '\n'.join(lines) + render_causal_summary(result.get('responses', []))


def auxiliary_evidence(repo, failures, response_ids, output):
    reads, logs = [], []
    for path in failures:
        try:
            data, digest = read_json(path)
            reads.append(dict(path=str(path), sha256=digest, record=data))
        except (OSError, ValueError) as exc:
            reads.append(dict(path=str(path), error=str(exc)))
    # Do not assign uncorrelated provider logs to a response by timestamp proximity.
    log_paths = set((repo / 'logs').glob('*.log'))
    # Archived logs are selected by retained archive manifests, never by backend ports.
    for manifest in (repo / 'logs/archive').glob('*/manifest.jsonl'):
        for line in manifest.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            raw_path = entry.get('archived_path', '')
            candidate = (repo / raw_path).resolve()
            if candidate.is_relative_to(repo / 'logs') and candidate.is_file():
                log_paths.add(candidate)
    for path in sorted(log_paths):
        before = path.stat()
        matched = []
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for i, raw in enumerate(stream, 1):
                digest.update(raw)
                if b'resp_' not in raw:
                    continue
                text = raw.decode('utf-8', errors='replace')
                ids = [rid for rid in re.findall(r'resp_[A-Za-z0-9_-]+', text) if rid in response_ids]
                if ids:
                    # Pointer/hash only: avoid copying unrelated request content or tokens.
                    matched.append(dict(line=i, response_ids=ids, line_sha256=hashlib.sha256(raw).hexdigest()))
        after = path.stat()
        logs.append(dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest(), matches=matched,
                         stable_during_read=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)))
    value = dict(truth_fetch_failures=reads, logs=logs, limitation='Uncorrelated log entries are not attributed from timestamps; archive paths come from existing retention manifests.')
    atomic_write_json(output, value)
    return dict(path=str(output), truth_fetch_failure_count=len(reads), log_count=len(logs), correlated_log_lines=sum(len(l['matches']) for l in logs))


def run_audit(root, output, *, repo=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or root == output:
        raise CorpusError('Audit needs an existing evidence root and a separate output directory.')
    output.mkdir(parents=True, exist_ok=True)
    manifests, captures, failures, omitted = discover(root, output)
    if not manifests and not captures:
        raise CorpusError('No self-attack sequence manifests or versioned captures found.')
    inventory = dict(root=str(root), manifests=[str(p) for p in manifests], captures=[str(p) for p in captures],
                     files=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in manifests+captures+failures],
                     omitted_by_extension=omitted)
    identity = stable_digest(inventory)
    source_files = [Path(__file__).with_name(name) for name in (
        'self_attack_convergence.py', 'ollmo_self_attack.py', 'self_attack_checks.py',
        'run_graph_rebase_shadow_corpus.py')]
    source_hash = stable_digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    run_path = output / 'run.json'
    if run_path.exists():
        old = json.loads(run_path.read_text())
        if old.get('input_identity') != identity or old.get('source_hash') != source_hash:
            raise CorpusError('Audit input or analysis source changed; use a new output directory.')
    atomic_write_json(run_path, dict(kind=SCHEMA, started_at=utc_now(), source_hash=source_hash,
                                    input_identity=identity, evidence_root=str(root), authority='offline_diagnostic_only'))
    atomic_write_json(output / 'inventory.json', inventory)
    source_copy = output / 'analysis-source'
    source_copy.mkdir(exist_ok=True)
    for path in source_files:
        (source_copy / path.name).write_bytes(path.read_bytes())
    by_response, contexts, observations = defaultdict(list), defaultdict(list), defaultdict(list)
    errors=[]
    for path in manifests:
        try:
            manifest, origin = manifest_context(path)
            for case in records(manifest.get('cases')):
                rid=case.get('response_id')
                if rid:
                    contexts[rid].append(context_for_case(case, origin))
                    observations[rid].append(observation_record(case, origin))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    for path in captures:
        by_response[path.parent.name].append(path)
    response_ids=set(by_response)|set(contexts)
    ordered=sorted(response_ids, key=lambda rid:(not any(c['mode']=='live' for c in contexts[rid]),
                    -max((o['dispatch_to_settled_observation_seconds'] or 0 for o in observations[rid]),default=0),rid))
    details=output/'responses';details.mkdir(exist_ok=True)
    summaries=[]
    started=time.monotonic()
    for i,rid in enumerate(ordered):
        out=details/(stable_digest(rid)[:24]+'.json')
        atomic_write_json(output/'progress.json',dict(stage='responses',pid=os.getpid(), completed=i,total=len(ordered),
            response_id=rid,elapsed_seconds=round(time.monotonic()-started,3),results_path=str(output/'results.json'),report_path=str(output/'report.md')))
        if out.exists():
            summaries.append(response_summary(json.loads(out.read_text()),out))
        else:
            summaries.append(analyze_response(rid,by_response[rid],contexts[rid],observations[rid],out))
        if i%25==0 or any(c['mode']=='live' for c in contexts[rid]):
            print(f'{utc_now()} analyzed {i+1}/{len(ordered)} {rid}',flush=True)
    atomic_write_json(output/'progress.json',dict(stage='ledger_and_logs',pid=os.getpid(),completed=len(ordered),total=len(ordered)))
    repo=Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    ledger_paths=[repo/'state/response_frames/responses.jsonl']
    ledger_paths += sorted({p.parent/'runtime/state/response_frames/responses.jsonl' for p in manifests})
    ledger=ledger_index(ledger_paths,response_ids,output/'ledger-evidence.json')
    auxiliary=auxiliary_evidence(repo,failures,response_ids,output/'auxiliary-evidence.json')
    totals=Counter();by_mode={}
    for s in summaries:
        modes={c['mode'] for c in s['contexts']}
        mode=next(iter(modes)) if len(modes)==1 else 'unknown_or_mixed'
        group=by_mode.setdefault(mode,dict(responses=0,batch_seconds=Counter()))
        group['responses']+=1;group['batch_seconds'].update(s['timing']['batch_seconds'])
        totals.update(s['classification_counts'])
    aggregate=dict(causal_invocations=sum(s.get('causal_invocation_count', 0) for s in summaries),
        responses=len(summaries), captures=sum(s['capture_count'] for s in summaries),
        distinct_captures=sum(s['distinct_captures'] for s in summaries), duplicate_capture_copies=sum(s['duplicate_capture_copies'] for s in summaries),
        invocation_witnesses=sum(s['invocation_count'] for s in summaries),
        unchanged_projection_observations=sum(s['unchanged_projection_observations'] for s in summaries),
        state_projection_observations=sum(s['state_projection_count'] for s in summaries),
        wait_records=sum(s['wait_count'] for s in summaries),classification_counts={c:totals[c] for c in CLASSES},
        history_conflict_responses=sum(bool(s['timing']['conflicting_history_ordinals']) for s in summaries),
        errors=len(errors)+sum(len(s['errors']) for s in summaries)+len(ledger['errors']),by_mode=by_mode)
    result=dict(kind=SCHEMA,status='completed_with_observability_limits',finished_at=utc_now(),
        source_hash=source_hash,input_identity=identity,aggregate=aggregate,responses=summaries,
        opportunities=rank_opportunities(summaries),instrumentation=INSTRUMENTATION,
        manifest_errors=errors,ledger_evidence=ledger,auxiliary_evidence=auxiliary,
        proven_optimization_findings=[],new_live_work=False,runtime_behavior_changed=False)
    atomic_write_json(output/'results.json',result)
    (output/'report.md').write_text(render_report(result),encoding='utf-8')
    atomic_write_json(output/'progress.json',dict(stage='complete',pid=os.getpid(),completed=len(ordered),total=len(ordered),
        elapsed_seconds=round(time.monotonic()-started,3),results_path=str(output/'results.json'),report_path=str(output/'report.md')))
    print(f'Audit complete: {output / "report.md"}',flush=True)
    return 0
