"""Deterministic negative-evidence oracle over canonical Ollmo response truth.

This does not decide runtime fulfillment. It detects contradictory runtime
records and reports absent evidence, rather than accepting assistant claims.
"""
from __future__ import annotations

import hashlib

from scripts.run_graph_rebase_shadow_corpus import stable_digest
from ollmo_services.graph_rebase import _semantic_record_payload

OPEN = {'pending', 'blocked', 'failed', 'repair_needed', 'repair_required', 'unmet',
        'semantic_review_pending', 'running', 'queued', 'in_progress'}
SUCCESS = {'completed', 'fulfilled', 'passed', 'frozen', 'late_fill_completed'}
STOPPED = {'cancelled', 'canceled', 'waived', 'superseded'}


def records(value):
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def walk(value, path=''):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from walk(item, f'{path}/{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk(item, f'{path}/{i}')


def finding(code, path, detail):
    return dict(code=code, path=path, detail=detail,
                signature=stable_digest({'code': code, 'path': path})[:20])


def audit_truth(payload: dict, *, artifact_evidence=None) -> dict:
    failures, missing, exercised = [], [], []
    runtime = payload.get('runtime') or {}
    graph = runtime.get('request_phase_graph') or {}
    closure = runtime.get('graph_closure_review') or {}
    frame = payload.get('response_frame') or {}
    for key, value in [('runtime.request_phase_graph', graph),
                       ('runtime.graph_closure_review', closure), ('response_frame', frame)]:
        if not value:
            missing.append(key)
    if 'outputs' not in payload:
        missing.append('outputs')
    for key in ('intent_obligations', 'output_obligations', 'phases'):
        if key not in graph:
            if key == 'intent_obligations' and (graph.get('prompt_intent') or {}).get('intent_obligation_count') == 0:
                continue
            missing.append(f'runtime.request_phase_graph.{key}')
    def fail(code, path, detail):
        failures.append(finding(code, path, detail))
    branches = records(graph.get('downstream_branches'))
    phases = records(graph.get('phases'))
    obligations = records(graph.get('output_obligations'))
    ids = [p.get('phase_id') for p in phases if p.get('phase_id')]
    if len(ids) != len(set(ids)):
        fail('duplicate_phase_identity', '/runtime/request_phase_graph/phases', 'Phase identities are not unique.')
    by_id = {p.get('phase_id'): p for p in phases}
    edges = {p.get('phase_id'): list(p.get('depends_on') or []) for p in phases + branches}
    for target, dependencies in edges.items():
        if any(d not in by_id for d in dependencies):
            fail('dangling_dependency', '/runtime/request_phase_graph/phases', f'{target} has an absent producer.')
    def cyclic(node, active, done):
        if node in active:
            return True
        if node in done:
            return False
        active.add(node)
        if any(cyclic(d, active, done) for d in edges.get(node, []) if d in edges):
            return True
        active.remove(node)
        done.add(node)
        return False
    if any(cyclic(node, set(), set()) for node in edges):
        fail('dependency_cycle', '/runtime/request_phase_graph', 'Executable dependencies contain a cycle.')
    exercised.append('graph_identity_and_dependencies')
    promoted = {o.get('phase_id') for o in obligations
                if o.get('status') not in {'reserved', 'candidate', 'omitted', 'rejected'}
                and o.get('contract_state') not in {'reserved', 'candidate'}}
    for branch in branches:
        state = branch.get('contract_state') or branch.get('status')
        if state in {'reserved', 'candidate', 'omitted', 'rejected'}:
            if branch.get('status') in {'running', 'fulfilled', 'completed'}:
                fail('unpromoted_execution', '/runtime/request_phase_graph/downstream_branches', 'Reserved work executed.')
        elif branch.get('phase_id') not in promoted and obligations:
            fail('branch_without_obligation', '/runtime/request_phase_graph/downstream_branches',
                 f"Branch {branch.get('phase_id')} has no output obligation.")
    exercised.append('promotion_boundary')
    late = payload.get('late_fill') or {}
    stopped = {b.get('branch_id') or b.get('phase_id')
               for b in records(late.get('cancelled_branches'))}
    controls = late.get('branch_controls') or {}
    if isinstance(controls, dict):
        stopped |= {key for key, value in controls.items() if isinstance(value, dict)
                    and (value.get('status') or value.get('action')) in STOPPED}
    for result in records(late.get('fill_results')):
        identity = result.get('branch_id') or result.get('phase_id')
        branch = next((b for b in branches + phases if identity in {b.get('branch_id'), b.get('phase_id')}
                       or result.get('phase_id') == b.get('phase_id')), {})
        identities = {identity, result.get('phase_id')} - {None, ''}
        # Graph phases can remain planned in a frozen response. Canonical
        # outputs and closure checks own fulfillment; fill results need not
        # repeat a status field at all.
        accepted = (result.get('status') in SUCCESS or any(
            item.get('status') in SUCCESS
            and bool(identities & {item.get('branch_id'), item.get('phase_id')})
            for item in phases + branches + records(payload.get('outputs')) + records(closure.get('checks'))))
        if identity in stopped and accepted and not result.get('stale'):
            fail('stale_result_accepted', '/late_fill/fill_results', 'Stopped branch result was accepted as successful.')
        evidence = result.get('tts_stt_semantic_evidence') or {}
        if evidence:
            exercised.append('exact_source_binding')
            if accepted and evidence.get('status') in {'mismatched', 'unavailable', 'failed'}:
                fail('invalid_source_evidence_accepted', '/late_fill/fill_results', 'Unbound or mismatched audio evidence was accepted.')
            if evidence.get('status') == 'matched':
                deps = branch.get('depends_on') or (branch.get('execution_contract') or {}).get('dependencies') or []
                if evidence.get('producer_phase_id') not in deps:
                    fail('sibling_evidence_substituted', '/late_fill/fill_results', 'Evidence was bound to an undeclared producer.')
                producer = next((p for p in records(late.get('fill_results'))
                                 if p.get('phase_id') == evidence.get('producer_phase_id')
                                 and p.get('capability') == 'text_to_speech'), {})
                source = producer.get('tts_semantic_source') or {}
                text = source.get('tts_source_text')
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text, str) else None
                if not digest or digest != source.get('tts_source_text_sha256') or digest != evidence.get('source_sha256'):
                    fail('source_digest_mismatch', '/late_fill/fill_results', 'Accepted semantic evidence does not bind the exact saved producer source.')
        elif result.get('capability') == 'speech_to_text' and any(
                p.get('capability') == 'text_to_speech' and p.get('phase_id') in branch.get('depends_on', [])
                for p in phases):
            missing.append(f'late_fill.fill_results.{identity}.tts_stt_semantic_evidence')
    if stopped:
        exercised.append('stale_result_gate')
    completed = payload.get('lifecycle_state') in SUCCESS
    open_checks = [check for check in records(closure.get('checks'))
                   if check.get('status') in OPEN and check.get('required', True)]
    if completed and (closure.get('status') in OPEN or open_checks):
        fail('false_closure', '/runtime/graph_closure_review', 'Successful lifecycle retains required open closure checks.')
    if completed and (late.get('status') in {'pending', 'running', 'queued'}
                      or records(late.get('active_branches'))):
        fail('active_work_frozen_successfully', '/late_fill', 'Successful closure still has active Late Fill work.')
    if closure:
        exercised.append('closure_before_success')
    for path, record in walk(closure):
        if record.get('check_kind') in {'branch_semantic_review', 'global_semantic_closure'}:
            exercised.append('semantic_review_gate')
            verdict = record.get('semantic_review_verdict') or {}
            if record.get('status') in SUCCESS and isinstance(verdict, dict) and verdict.get('status') in {'failed', 'uncertain', 'unparseable'}:
                fail('failed_review_claimed_passed', path, 'Failed semantic verdict was projected as fulfilled.')
    for path, record in walk(runtime):
        if record.get('kind') in {'ollmo.commitment_review', 'ollmo.aspiration_review', 'ollmo.controlled_attention_review'}:
            exercised.append('advisory_authority')
            if record.get('authority') and record['authority'] != 'advisory_read_model_only':
                fail('advisory_runtime_authority', path, 'Advisory movement claims promotion/closure authority.')
        if record.get('kind') == 'ollmo.semantic_role_profile':
            exercised.append('advisory_authority')
            effect = (record.get('runtime_orientation') or {}).get('runtime_effect') or (record.get('authority_boundary') or {}).get('runtime_effect')
            if effect and effect != 'none':
                fail('advisory_runtime_authority', path, 'Semantic role claims runtime effect.')
        if ('autonomy_level' in record and isinstance(record.get('outcome'), str)
                and record['outcome'] in {'applied', 'applied_safe'}):
            exercised.append('repair_authority')
            if record['autonomy_level'] in {'off', 'shadow', 'stage'}:
                fail('nonexecuting_profile_mutated_graph', path, 'Off/shadow/stage cannot apply a graph mutation.')
            if not record.get('evidence_refs'):
                fail('mutation_without_evidence', path, 'Applied graph mutation has no evidence refs.')
    artifacts = {a.get('artifact_ref'): a for a in records(payload.get('artifacts')) if a.get('artifact_ref')}
    for index, output in enumerate(records(payload.get('outputs'))):
        ref = output.get('artifact_ref')
        if output.get('status') != 'fulfilled' or not ref:
            continue
        exercised.append('artifact_fulfillment')
        artifact = artifacts.get(ref)
        if not artifact:
            fail('unbacked_artifact_output', f'/outputs/{index}', 'Fulfilled output has no canonical artifact record.')
        elif artifact_evidence is not None:
            evidence = artifact_evidence.get(ref)
            if not evidence or not evidence.get('exists'):
                fail('missing_saved_artifact', f'/outputs/{index}', 'Fulfilled artifact is absent on disk.')
            elif evidence.get('stable_during_capture') is False:
                missing.append(f'outputs.{index}.stable_artifact_snapshot')
    return dict(findings=failures, missing=missing, exercised=sorted(set(exercised)))


def semantic_contract(payload: dict) -> list:
    """An identity-independent multiset of anchored intent, not execution topology.

    Only runtime's intent ledger is compared: repair strategies can legitimately
    add phases and review branches. Cardinality and semantic content survive.
    """
    graph = (payload.get('runtime') or {}).get('request_phase_graph') or {}
    obligations = records(graph.get('intent_obligations'))
    identity_keys = {'obligation_id', 'phase_id', 'branch_id', 'task_id', 'queue_index', 'evidence'}
    dependency_keys = {'depends_on_obligation_ids', 'dependency_obligation_ids'}
    def meaning(item):
        # Use rebase's existing definition of stable record meaning. Retain new
        # semantic fields automatically, including cardinality and constraints.
        return {key: value for key, value in _semantic_record_payload(item, exclude_dependencies=True).items()
                if key not in identity_keys | dependency_keys}
    names = {item.get('obligation_id'): stable_digest(meaning(item)) for item in obligations}
    contracts = []
    for item in obligations:
        contract = meaning(item)
        for key in dependency_keys:
            if key in item:
                contract[key] = sorted(names.get(ref, ref) for ref in item[key])
        contracts.append(stable_digest(contract))
    return sorted(contracts)


def compare_profiles(baseline: dict, candidate: dict) -> list:
    if semantic_contract(baseline) != semantic_contract(candidate):
        return [finding('forbidden_semantic_divergence', '/runtime/request_phase_graph/intent_obligations',
                        'The same user turn produced a different anchored intent contract across controls.')]
    return []


def audit_history(snapshots: list[dict]) -> list:
    """Same frozen frame identity may not change its canonical frozen payload."""
    seen, findings = {}, []
    for snapshot in snapshots:
        frame = snapshot.get('response_frame') or {}
        frame_id = frame.get('frame_id')
        if not frame_id:
            continue
        # Frozen frame is the persisted unit; changing top-level projections is allowed.
        digest = stable_digest(frame)
        if frame_id in seen and seen[frame_id] != digest:
            findings.append(finding('frozen_frame_mutated', '/response_frame',
                                    'The same frozen frame identity changed during observation.'))
        # Working projections can legitimately update before the first freeze.
        # Once frozen, even a later projection back to working state is a fault.
        if frame.get('status') in SUCCESS | {'failed', 'incomplete', 'repair_needed', 'blocked', 'cancelled'}:
            seen[frame_id] = digest
    return findings


def audit_provider_bindings(call_records):
    """Check the actual fake-provider handoff against declared graph producers."""
    producers = {}
    results = []
    for call in call_records:
        request, response = call.get('payload') or {}, call.get('result') or {}
        identity = request.get('response_id')
        phase = request.get('phase_id')
        if call.get('capability') == 'text_to_speech':
            producers[(identity, phase)] = call
        if call.get('capability') != 'speech_to_text':
            continue
        dependencies = (request.get('execution_contract') or {}).get('depends_on') or []
        sources = [producers[(identity, p)] for p in dependencies if (identity, p) in producers]
        if len(sources) != 1:
            results.append(dict(response_id=identity, missing='exact_provider_source_handoff', findings=[]))
            continue
        source = sources[0]
        bound = (request.get('file_path') == (source.get('result') or {}).get('saved_audio_path')
                 and bool(call.get('input_sha256')) and call['input_sha256'] == source.get('output_sha256'))
        results.append(dict(response_id=identity, missing=None, findings=[] if bound else [finding(
            'consumer_artifact_binding_mismatch', '/provider_handoff/file_path',
            'Consumer input path/bytes differ from its exact declared producer artifact.')]))
    return results
