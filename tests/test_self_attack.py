from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.ollmo_self_attack import (
    CaptureClient, PRIORITIES, check_expectations, evaluate_capture, isolated_profile,
    materialize_corpus, minimize_failure, run_profile, signatures,
)
from scripts.self_attack_checks import audit_history, audit_truth, compare_profiles
from scripts.self_attack_knobs import build_profiles, discover_knobs
from scripts.run_graph_rebase_shadow_corpus import HttpResult

ROOT = Path(__file__).resolve().parents[1]


def truth():
    return {'id': 'response', 'lifecycle_state': 'completed', 'outputs': [], 'artifacts': [],
            'response_frame': {'frame_id': 'frame-1', 'frame_sequence': 1, 'status': 'frozen'},
            'runtime': {'request_phase_graph': {
                'phases': [{'phase_id': 'one', 'capability': 'chat'}], 'downstream_branches': [],
                'intent_obligations': [{'obligation_id': 'intent-1', 'kind': 'text', 'required': True}],
                'output_obligations': [{'phase_id': 'one', 'required': True, 'capability': 'chat'}]},
                'graph_closure_review': {'status': 'fulfilled', 'checks': []}}}


def codes(payload):
    return {f['code'] for f in audit_truth(payload)['findings']}


def test_oracle_never_uses_prose_and_missing_graph_cannot_pass():
    value = truth()
    value['output_text'] = 'FAIL! All checks failed!'
    assert not audit_truth(value)['findings']
    value['runtime']['graph_closure_review']['status'] = 'blocked'
    value['output_text'] = 'PASS! Every requirement fulfilled!'
    assert 'false_closure' in codes(value)
    value['runtime']['request_phase_graph'] = {'kind': 'ollmo.request_phase_graph'}
    assert 'runtime.request_phase_graph.intent_obligations' in audit_truth(value)['missing']


@pytest.mark.parametrize('mutation, expected', [
    (lambda p: p['runtime']['request_phase_graph']['phases'].append({'phase_id': 'one'}), 'duplicate_phase_identity'),
    (lambda p: p['runtime']['request_phase_graph']['phases'][0].update(depends_on=['absent']), 'dangling_dependency'),
    (lambda p: p['runtime']['request_phase_graph']['phases'][0].update(depends_on=['one']), 'dependency_cycle'),
    (lambda p: p.update(late_fill={'status': 'running'}), 'active_work_frozen_successfully'),
    (lambda p: p['outputs'].append({'status': 'fulfilled', 'artifact_ref': 'not-real'}), 'unbacked_artifact_output'),
])
def test_structural_corruption_is_rejected(mutation, expected):
    value = truth()
    mutation(value)
    assert expected in codes(value)


def test_stopped_branch_and_sibling_evidence_cannot_be_accepted():
    value = truth()
    value['late_fill'] = {'cancelled_branches': [{'branch_id': 'stt'}], 'fill_results': [
        {'branch_id': 'stt', 'status': 'fulfilled', 'tts_stt_semantic_evidence': {'status': 'matched', 'producer_phase_id': 'sibling'}}]}
    assert {'stale_result_accepted', 'sibling_evidence_substituted'} <= codes(value)


@pytest.mark.parametrize('kind', ['commitment', 'aspiration', 'controlled_attention'])
def test_advisory_surfaces_cannot_claim_authority(kind):
    value = truth()
    value['runtime']['request_phase_graph']['decision_contract'] = {
        f'{kind}_review': {'kind': f'ollmo.{kind}_review', 'authority': 'runtime_authority'}}
    assert 'advisory_runtime_authority' in codes(value)


def test_shadow_apply_is_rejected():
    value = truth()
    value['runtime']['request_phase_graph']['graph_patch_lifecycle'] = [
        {'outcome': 'applied', 'autonomy_level': 'shadow', 'evidence_refs': ['runtime:one']}]
    assert 'nonexecuting_profile_mutated_graph' in codes(value)


def test_comparison_permits_strategy_and_identity_changes_but_not_lost_intent():
    base, changed = truth(), truth()
    changed['runtime']['request_phase_graph']['phases'].append({'phase_id': 'review', 'capability': 'chat'})
    changed['runtime']['request_phase_graph']['intent_obligations'][0]['obligation_id'] = 'renamed'
    changed['output_text'] = 'A different valid narrative.'
    assert compare_profiles(base, changed) == []
    changed['runtime']['request_phase_graph']['intent_obligations'][0]['required'] = False
    assert compare_profiles(base, changed)[0]['code'] == 'forbidden_semantic_divergence'
    changed['runtime']['request_phase_graph']['intent_obligations'] = []
    assert compare_profiles(base, changed)


def test_comparison_preserves_multiplicity():
    base, changed = truth(), truth()
    changed['runtime']['request_phase_graph']['intent_obligations'] *= 2
    assert compare_profiles(base, changed)


def test_comparison_preserves_cardinality_and_renames_dependency_references():
    base = truth()
    graph = base['runtime']['request_phase_graph']
    graph['intent_obligations'][0]['count'] = 2
    graph['intent_obligations'].append({'obligation_id': 'consumer', 'kind': 'evidence',
                                      'depends_on_obligation_ids': ['intent-1']})
    changed = deepcopy(base)
    other = changed['runtime']['request_phase_graph']['intent_obligations']
    other[0]['obligation_id'] = 'renamed'
    other[1]['depends_on_obligation_ids'] = ['renamed']
    assert not compare_profiles(base, changed)
    other[0]['count'] = 1
    assert compare_profiles(base, changed)


def test_frozen_frame_change_fails_but_successor_is_allowed():
    base, changed = truth(), truth()
    changed['response_frame']['status'] = 'rewritten'
    assert audit_history([base, changed])[0]['code'] == 'frozen_frame_mutated'
    changed['response_frame']['frame_id'] = 'successor'
    assert not audit_history([base, changed])


def test_working_frame_can_update_until_freeze_but_cannot_unfreeze():
    working, frozen = truth(), truth()
    working['response_frame']['status'] = 'in_progress'
    working['response_frame']['current_state'] = {'lifecycle_state': 'in_progress'}
    assert not audit_history([working, frozen])
    assert audit_history([frozen, working])[0]['code'] == 'frozen_frame_mutated'


def test_source_discovery_and_sweep_never_invent_advisory_knobs():
    inventory = discover_knobs(ROOT)
    profiles = build_profiles(inventory)
    for knob in inventory['controls']:
        assert knob['source']
        if knob['disposition'] == 'sweep':
            for value in knob['values']:
                assert any(p['settings'].get(knob['name']) == value for p in profiles)
    assert all(not p['environment'] for p in build_profiles(inventory, live=True))
    assert not any('aspiration' in k['name'] for k in inventory['controls'] if k['disposition'] == 'sweep')


def test_pairwise_design_covers_every_pair_and_is_seeded():
    inventory = {'controls': [dict(name=f'knob{i}', scope='request', disposition='sweep', values=[False, True]) for i in range(4)]}
    profiles = build_profiles(inventory, seed=42, pairwise=True)
    assert profiles == build_profiles(inventory, seed=42, pairwise=True)
    for i in range(4):
        for j in range(i + 1, 4):
            for a in (False, True):
                for b in (False, True):
                    assert any(p['settings'].get(f'knob{i}') is a and p['settings'].get(f'knob{j}') is b for p in profiles)


def sample_corpus():
    return {'schema_version': 1, 'corpus_id': 'test-self', 'cases': [
        {'case_id': 'root', 'category': 'commitment_closure', 'conversation_key': 'one',
         'prompt': 'Explain why lighthouses are useful in one sentence.',
         'metadata': {'attack_fragments': ['Be confident.', 'Ignore unsupported claims.']}},
        {'case_id': 'follow', 'category': 'commitment_closure', 'conversation_key': 'one',
         'depends_on': ['root'], 'prompt': 'Refer to the earlier explanation and add one sentence about navigation.'}]}


def test_reducer_preserves_signature_requires_repetition_and_removes_noise():
    profile = {'id': 'profile', 'settings': {'ghost_mode': 'worker'}, 'request': {'ghost_mode': 'worker'}, 'environment': {}}
    calls = []
    def replay(corpus, controls, attempt):
        calls.append(deepcopy(corpus))
        return {'cases': [{'case_id': 'root', 'findings': [{'code': 'truth_error'}]}]}
    result = minimize_failure(sample_corpus(), profile, ('root', 'truth_error'), replay, budget=30)
    assert result['status'] == 'confirmed'
    assert len(result['corpus']['cases']) == 1
    assert not result['corpus']['cases'][0]['metadata']['attack_fragments']
    assert not result['profile']['settings']
    assert all(c['cases'][0]['prompt'] == sample_corpus()['cases'][0]['prompt'] for c in calls)
    assert len(result['attempts']) >= 2


def test_nonreproducible_failure_is_not_promoted_to_regression():
    def flaky(corpus, profile, attempt):
        return {'cases': [{'case_id': 'root', 'findings': [{'code': 'bad'}] if attempt == 0 else []}]}
    profile = {'id': 'baseline', 'settings': {}, 'request': {}, 'environment': {}}
    result = minimize_failure(sample_corpus(), profile, ('root', 'bad'), flaky)
    assert result['status'] == 'unconfirmed'
    assert len(result['attempts']) == 2


def test_reducer_budget_is_explicit_and_bounded():
    profile = {'id': 'baseline', 'settings': {}, 'request': {}, 'environment': {}}
    result = minimize_failure(sample_corpus(), profile, ('root', 'bad'),
                              lambda *a: {'cases': [{'case_id': 'root', 'findings': [{'code': 'bad'}]}]}, budget=2)
    assert result['status'] == 'confirmed'
    assert result['budget_exhausted'] and len(result['attempts']) == 2


def test_capture_gets_full_truth_instead_of_debug_projection(tmp_path):
    calls = []
    class Client:
        def get(self, path, *, timeout):
            calls.append(path)
            return HttpResult(200, truth() if path.endswith('view=truth') else {'id': 'response'})
    client = CaptureClient(Client(), tmp_path)
    client.get('/api/responses/response?view=debug', timeout=1)
    captures = list((tmp_path / 'response').glob('*.json'))
    assert calls[-1].endswith('view=truth')
    assert len(captures) == 1
    assert not audit_truth(json.loads(captures[0].read_text())['payload'])['missing']


def test_missing_capture_does_not_pass(tmp_path):
    manifest = {'cases': [{'case_id': 'x', 'category': 'commitment_closure', 'state': 'settled_terminal', 'response_id': 'r'}]}
    assert evaluate_capture(manifest, tmp_path)[0]['missing'] == ['settled_full_truth']


def test_capture_retains_truth_read_without_spending_budget_on_duplicate_get(tmp_path):
    calls = []
    class Client:
        def get(self, path, *, timeout):
            calls.append(path)
            assert len(calls) == 1, 'Canonical truth must not be fetched twice'
            return HttpResult(200, truth())
    result = CaptureClient(Client(), tmp_path).get('/api/responses/response?view=truth', timeout=1)
    captures = list((tmp_path / 'response').glob('*.json'))
    assert result.ok and len(captures) == 1
    assert not audit_truth(json.loads(captures[0].read_text())['payload'])['missing']


def test_settled_capture_survives_debug_companion_timeout_with_fresh_truth(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    now = [0.0]
    monkeypatch.setattr(command.time, 'monotonic', lambda: now[0])
    payload = truth()
    calls = []
    class Client:
        def get(self, path, *, timeout):
            calls.append(path)
            if path.endswith('view=debug'):
                return HttpResult(200, {'id': 'response'})
            if len(calls) in {2, 3}:
                now[0] += timeout
                return HttpResult(0, {}, error='timed out')
            now[0] += 19
            return HttpResult(200, payload)
    client = CaptureClient(Client(), tmp_path, deadline=1200)
    # This historical observation must remain an observation, even though its
    # payload was completed. Only the new successful GET may close the gap.
    client.capture(payload, 'observation')
    historical = next((tmp_path / 'response').glob('*.json'))
    original_bytes = historical.read_bytes()
    manifest = {'cases': [dict(case_id='root', category='repair_rebase_intent',
                              state='settled_terminal', response_id='response')]}
    client.get('/api/responses/response?view=debug', timeout=30)
    assert evaluate_capture(manifest, tmp_path)[0]['missing'] == ['settled_full_truth']
    client.get('/api/responses/response?view=truth', timeout=30)
    assert now[0] == 79 and len(calls) == 4
    assert historical.read_bytes() == original_bytes
    snapshots = [json.loads(p.read_text()) for p in (tmp_path / 'response').glob('*.json')]
    assert sorted(s['source'] for s in snapshots) == ['observation', 'settled']
    case = evaluate_capture(manifest, tmp_path)[0]
    assert not case['missing'] and not case['findings']
    failures = [json.loads(p.read_text()) for p in (tmp_path / 'truth-fetch-failures').glob('*.json')]
    assert len(failures) == 2
    assert {f['attempt'] for f in failures} == {1, 2}
    assert all(f['error'] == 'timed out' and f['timeout_seconds'] == 30 and f['elapsed_seconds'] == 30
               and f['trigger'].endswith('view=debug') for f in failures)


def test_capture_retries_unchanged_status_after_companion_timeout(tmp_path):
    attempts = []
    class Client:
        def get(self, path, *, timeout):
            if path.endswith('view=status'):
                return HttpResult(200, {'id': 'response', 'lifecycle_state': 'completed'})
            attempts.append(path)
            return HttpResult(0, {}, error='timed out') if len(attempts) <= 2 else HttpResult(200, truth())
    client = CaptureClient(Client(), tmp_path)
    client.get('/api/responses/response?view=status', timeout=30)
    assert not list((tmp_path / 'response').glob('*.json'))
    client.get('/api/responses/response?view=status', timeout=30)
    assert len(attempts) == 3
    snapshots = [json.loads(p.read_text()) for p in (tmp_path / 'response').glob('*.json')]
    assert len(snapshots) == 1 and snapshots[0]['source'] == 'settled'


def test_settled_companion_retry_recovers_without_repeating_debug(tmp_path):
    calls = []
    class Client:
        def get(self, path, *, timeout):
            calls.append(path)
            if path.endswith('view=debug'):
                return HttpResult(200, {'id': 'response'})
            return HttpResult(0, {}, error='timed out') if len(calls) == 2 else HttpResult(200, truth())
    CaptureClient(Client(), tmp_path).get('/api/responses/response?view=debug', timeout=30)
    assert sum(path.endswith('view=debug') for path in calls) == 1
    assert sum(path.endswith('view=truth') for path in calls) == 2
    snapshot = json.loads(next((tmp_path / 'response').glob('*.json')).read_text())
    assert snapshot['source'] == 'settled'
    assert len(list((tmp_path / 'truth-fetch-failures').glob('*.json'))) == 1


def test_companion_transport_exception_is_recorded_and_retried(tmp_path):
    calls = []
    class Client:
        def get(self, path, *, timeout):
            calls.append(path)
            if path.endswith('view=debug'):
                return HttpResult(200, {'id': 'response'})
            if len(calls) == 2:
                raise TimeoutError('transport read timed out')
            return HttpResult(200, truth())
    CaptureClient(Client(), tmp_path).get('/api/responses/response?view=debug', timeout=30)
    assert len(calls) == 3
    failure = json.loads(next((tmp_path / 'truth-fetch-failures').glob('*.json')).read_text())
    assert failure['error'] == 'TimeoutError: transport read timed out'
    assert json.loads(next((tmp_path / 'response').glob('*.json')).read_text())['source'] == 'settled'


def test_companion_retry_cannot_exceed_sequence_deadline(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    now, calls = [0.0], []
    monkeypatch.setattr(command.time, 'monotonic', lambda: now[0])
    class Client:
        def get(self, path, *, timeout):
            calls.append((path, timeout))
            if path.endswith('view=debug'):
                return HttpResult(200, {'id': 'response'})
            assert timeout == 5
            now[0] += timeout
            return HttpResult(0, {}, error='timed out')
    with pytest.raises(command.LiveBudgetExpired):
        CaptureClient(Client(), tmp_path, deadline=5).get('/api/responses/response?view=debug', timeout=30)
    assert len(calls) == 2  # The second truth attempt must not reach transport.
    assert not list((tmp_path / 'response').glob('*.json'))
    failures = [json.loads(p.read_text()) for p in (tmp_path / 'truth-fetch-failures').glob('*.json')]
    assert any(f['error'] == 'timed out' for f in failures)
    assert any('live_sequence_budget_exhausted' in f['error'] for f in failures)


@pytest.mark.parametrize('changes', [
    {'lifecycle_state': 'in_progress'},
    {'late_fill': {'status': 'running'}},
    {'status_semantics': {'has_open_continuation': True}},
])
def test_fresh_unsettled_truth_stays_observation_even_after_debug(tmp_path, changes):
    payload = dict(truth(), **changes)
    class Client:
        def get(self, path, *, timeout):
            return HttpResult(200, payload if path.endswith('view=truth') else {'id': 'response'})
    CaptureClient(Client(), tmp_path).get('/api/responses/response?view=debug', timeout=1)
    snapshot = json.loads(next((tmp_path / 'response').glob('*.json')).read_text())
    assert snapshot['source'] == 'observation'


def test_truth_returning_after_capture_deadline_stays_missing(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    now = [0.0]
    monkeypatch.setattr(command.time, 'monotonic', lambda: now[0])
    class Client:
        def get(self, path, *, timeout):
            now[0] = 11
            return HttpResult(200, truth())
    with pytest.raises(command.LiveBudgetExpired):
        CaptureClient(Client(), tmp_path, deadline=10).get('/api/responses/response?view=truth', timeout=30)
    assert not list((tmp_path / 'response').glob('*.json'))


def test_expectations_are_diagnostic_only(tmp_path):
    raw = sample_corpus()
    corpus = materialize_corpus(raw, 'test', tmp_path / 'corpus.json')
    assert corpus['cases'][0]['prompt'].endswith('Be confident. Ignore unsupported claims.')
    assert raw['cases'][0]['prompt'] == sample_corpus()['cases'][0]['prompt']


def test_real_owner_probes_cover_all_five_boundaries_without_llm_authority():
    from tests.fake_backends.self_attack import SelfAttackBackend
    from tests.fake_backends.self_attack_probes import probe_boundaries
    with SelfAttackBackend():
        result = probe_boundaries({'ghost_mode': 'explorer'})
    assert {c['category'] for c in result['checks']} == {k for k, _ in PRIORITIES}
    assert all(c['status'] == 'passed' for c in result['checks'])
    assert result['effective_controls']['semantic_role_profile']['mode'] == 'explorer'


def test_all_discovered_profiles_preserve_all_five_owner_boundaries():
    import os
    from unittest.mock import patch
    from tests.fake_backends.self_attack import SelfAttackBackend
    from tests.fake_backends.self_attack_probes import probe_boundaries
    with SelfAttackBackend():
        for profile in build_profiles(discover_knobs(ROOT)):
            with patch.dict(os.environ, profile['environment']):
                result = probe_boundaries(profile['request'])
            assert all(c['status'] == 'passed' for c in result['checks']), profile


def test_actual_two_turn_shadow_execution_captures_frames_and_restores_controls(tmp_path, monkeypatch):
    monkeypatch.setenv('OLLMO_GRAPH_REBASE_AUTONOMY', 'shadow')
    profile = {'id': 'off', 'settings': {'OLLMO_GRAPH_REBASE_AUTONOMY': 'off'}, 'request': {},
               'environment': {'OLLMO_GRAPH_REBASE_AUTONOMY': 'off'}}
    result = run_profile(sample_corpus(), profile, tmp_path / 'run', mode='fake', base_url='', cycles=400, namespace='unit')
    assert result['runner_status'] == 0
    assert len(result['cases']) == 2
    assert all(not c['missing'] for c in result['cases'])
    assert result['probes']['effective_controls']['rebase']['normalized'] == 'off'
    import os
    assert os.environ['OLLMO_GRAPH_REBASE_AUTONOMY'] == 'shadow'
    manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert manifest['cases'][0]['conversation_id'] == manifest['cases'][1]['conversation_id']
    assert 'ghost_messages' in manifest['cases'][1]['dispatch_request']
    assert 'metadata' not in manifest['cases'][1]['dispatch_request']
    # Rechecking captured evidence must never dispatch another backend request.
    import scripts.ollmo_self_attack as command
    prior = {'mode': 'fake', 'runs': [result], 'owner_tests': {k: {'status': 'passed'} for k, _ in PRIORITIES},
             'coverage': {'profiles_available': 1, 'profiles_selected': 1, 'design': 'test', 'inventory_only': 0}}
    (tmp_path / 'run/progress.json').write_text(json.dumps(prior))
    (tmp_path / 'run/run.json').write_text(json.dumps({'source_digest': 'captured-source'}))
    monkeypatch.setattr(command.subprocess, 'run', lambda *a, **kw: pytest.fail('Recheck dispatched work'))
    assert command.recheck_captures(tmp_path / 'run', tmp_path / 'recheck', {'source_digest': 'new-oracle'}) == 0
    rechecked = json.loads((tmp_path / 'recheck/results.json').read_text())
    assert rechecked['recheck']['execution_performed'] is False
    assert rechecked['recheck']['capture_source_digest'] == 'captured-source'


def test_command_persists_reproducible_failures_and_replays_resolved_cases(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    from scripts.self_attack_checks import finding
    inventory = {'source_digest': 'test', 'controls': []}
    baseline = {'id': 'baseline', 'settings': {}, 'request': {}, 'environment': {}}
    monkeypatch.setattr(command, 'discover_knobs', lambda root: inventory)
    monkeypatch.setattr(command, 'build_profiles', lambda *a, **kw: [baseline])
    monkeypatch.setattr(command, 'run_owner_tests', lambda *a, **kw: {k: {'status': 'passed', 'cases': []} for k, _ in PRIORITIES})
    broken = [True]
    def replay(raw, profile, output, **kwargs):
        return dict(profile=profile, runner_status=0, output=str(output), cases=[
            dict(case_id=c['case_id'], category=c['category'], state='settled_terminal', missing=[],
                 findings=[finding('test_corruption', '/runtime/test', 'Injected oracle test fault.')]
                 if broken[0] and c['case_id'] == 'root' else []) for c in raw['cases']])
    monkeypatch.setattr(command, 'isolated_profile', replay)
    corpus = tmp_path / 'corpus.json'
    corpus.write_text(json.dumps(sample_corpus()))
    regressions = tmp_path / 'regressions'
    args = ['--corpus', str(corpus), '--regressions', str(regressions)]
    assert command.main([*args, '--output', str(tmp_path / 'failing')]) == 1
    saved = list(regressions.glob('*.json'))
    assert len(saved) == 1
    envelope = json.loads(saved[0].read_text())
    assert len(envelope['target']) == 3
    assert envelope['status'] == 'confirmed'
    assert len(envelope['corpus']['cases']) == 1
    broken[0] = False
    assert command.main([*args, '--output', str(tmp_path / 'fixed')]) == 0
    report = json.loads((tmp_path / 'fixed/results.json').read_text())
    assert report['regression_replays'][0]['status'] == 'resolved'


def test_persisted_profile_cannot_inject_a_request_or_environment_control():
    from scripts.self_attack_knobs import validate_saved_profile
    from scripts.run_graph_rebase_shadow_corpus import CorpusError
    inventory = {'controls': [dict(name='ghost_mode', values=['worker'], scope='request', disposition='sweep')]}
    rebound = validate_saved_profile({'id': '../escape', 'settings': {'ghost_mode': 'worker'},
                                     'request': {'instance_id': 'untrusted'}, 'environment': {'PATH': '/invalid'}}, inventory)
    assert rebound['request'] == {'ghost_mode': 'worker'}
    assert rebound['environment'] == {} and '/' not in rebound['id']
    with pytest.raises(CorpusError):
        validate_saved_profile({'settings': {'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'untrusted'}}, inventory)


def test_isolated_worker_applies_startup_controls_before_runtime_import(tmp_path):
    from scripts.ollmo_self_attack import isolated_group
    profile = {'id': 'workers-16', 'settings': {'OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS': 16},
               'request': {}, 'environment': {'OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS': '16'}}
    corpus = sample_corpus()
    corpus['cases'] = corpus['cases'][:1]
    result = isolated_group(corpus, profile, tmp_path / 'startup', mode='fake', base_url='',
                            cycles=200, namespace='startup-test', timeout=30)
    assert result['runner_status'] == 0, result.get('error')
    assert result['probes']['effective_controls']['max_parallel_workers'] == 16


def test_runtime_outcome_envelopes_do_not_crash_the_oracle():
    value = truth()
    value['runtime']['review'] = {'autonomy_level': 'shadow', 'outcome': {'status': 'blocked'}}
    assert not audit_truth(value)['findings']


def test_invalid_evidence_cannot_hide_behind_absent_result_status():
    value = truth()
    value['runtime']['request_phase_graph']['phases'] = [
        {'phase_id': 'tts', 'capability': 'text_to_speech', 'status': 'completed'},
        {'phase_id': 'stt', 'capability': 'speech_to_text', 'status': 'completed', 'depends_on': ['tts']}]
    value['late_fill'] = {'fill_results': [dict(phase_id='stt', capability='speech_to_text',
        tts_stt_semantic_evidence={'status': 'mismatched'})]}
    assert 'invalid_source_evidence_accepted' in codes(value)
    value['late_fill']['fill_results'][0].pop('tts_stt_semantic_evidence')
    assert audit_truth(value)['missing']


def test_incomplete_classification_preserves_missing_evidence(tmp_path):
    from scripts.ollmo_self_attack import explain_incomplete
    cases = [dict(case_id='one', state='observing', missing=['settled_full_truth'], findings=[]),
             dict(case_id='two', state='settled_terminal', missing=['scenario_not_exercised:exact_source_binding'], findings=[])]
    explain_incomplete(cases, tmp_path, error='profile_budget_exhausted_120s')
    assert cases[0]['incomplete_observations'][0]['classification'] == 'missing_observability'
    assert 'profile_budget_exhausted' in cases[0]['incomplete_observations'][0]['detail']
    assert cases[1]['incomplete_observations'][0]['classification'] == 'missing_scenario_coverage'
    assert all(c['missing'] for c in cases)


def test_fake_stt_reads_the_exact_saved_audio_not_the_expected_source():
    from tests.fake_backends.self_attack import SelfAttackBackend
    with SelfAttackBackend() as backend:
        a, status = backend._invoke_internal_api_json_route(payload={'capability': 'text_to_speech', 'prompt': 'First source.'})
        b, _ = backend._invoke_internal_api_json_route(payload={'capability': 'text_to_speech', 'prompt': 'Sibling source.'})
        assert status == 200 and a['saved_audio_path'] != b['saved_audio_path']
        transcript, status = backend._invoke_internal_api_json_route(payload={
            'capability': 'speech_to_text', 'file_path': a['saved_audio_path'], 'prompt': 'Sibling source.'})
        assert status == 200 and transcript['result']['transcript'] == 'First source.'
        missing, status = backend._invoke_internal_api_json_route(payload={
            'capability': 'speech_to_text', 'file_path': str(backend.root / 'missing.wav'), 'prompt': 'First source.'})
        assert status == 422


def test_provider_binding_rejects_sibling_path_and_changed_bytes():
    from scripts.self_attack_checks import audit_provider_bindings
    calls = [dict(capability='text_to_speech', payload={'response_id': 'r', 'phase_id': 'a'},
                  result={'saved_audio_path': '/a.wav'}, output_sha256='bytes-a'),
             dict(capability='speech_to_text', payload={'response_id': 'r', 'phase_id': 'b',
                  'file_path': '/a.wav', 'execution_contract': {'depends_on': ['a']}}, input_sha256='bytes-a')]
    assert audit_provider_bindings(calls)[0]['findings'] == []
    calls[1]['payload']['file_path'] = '/sibling.wav'
    assert audit_provider_bindings(calls)[0]['findings'][0]['code'] == 'consumer_artifact_binding_mismatch'
    calls[1]['payload']['file_path'] = '/a.wav'
    calls[1]['input_sha256'] = 'changed-bytes'
    assert audit_provider_bindings(calls)[0]['findings']


def test_live_requires_complete_fake_evidence_before_any_execution(tmp_path):
    from scripts.ollmo_self_attack import main
    with pytest.raises(SystemExit):
        main(['--mode', 'live', '--output', str(tmp_path / 'live')])


def test_report_does_not_count_regression_replays_as_sweep_profiles():
    from scripts.ollmo_self_attack import render_report
    case = dict(case_id='one', category='aspiration_promotion', capture_path='/capture', missing=[], findings=[])
    run = dict(profile={'id': 'baseline'}, cases=[case], runner_status=0, probes={})
    report = render_report(dict(verdict='incomplete', mode='fake', runs=[run, run], owner_tests={}, regressions=[],
        coverage=dict(profiles_selected=1, profiles_available=38, design='finite', inventory_only=0)))
    assert 'profiles: 1' in report and 'complete captures: 1/38' in report
    assert '| 2 | aspiration ↔ promotion | 1/1 |' in report


def test_canonical_output_fulfillment_owns_the_late_fill_evidence_gate():
    value = truth()
    graph = value['runtime']['request_phase_graph']
    graph['phases'] = [dict(phase_id='producer', capability='text_to_speech', status='pending'),
                       dict(phase_id='consumer', capability='speech_to_text', status='pending', depends_on=['producer'])]
    graph['downstream_branches'] = [dict(branch_id='stt', phase_id='consumer', capability='speech_to_text', depends_on=['producer'])]
    value['outputs'] = [dict(branch_id='stt', phase_id='consumer', status='fulfilled', type='text', value='Wrong source')]
    value['late_fill'] = {'fill_results': [dict(branch_id='stt', phase_id='consumer', capability='speech_to_text',
        tts_stt_semantic_evidence={'status': 'mismatched'})]}
    assert 'invalid_source_evidence_accepted' in codes(value)
    value['outputs'] = []
    value['runtime']['graph_closure_review']['checks'] = [dict(phase_id='consumer', status='fulfilled')]
    assert 'invalid_source_evidence_accepted' in codes(value)


def live_budget_command(tmp_path, monkeypatch, *, corpus=None):
    """Synthetic transport for budget tests; never contacts a live model."""
    import scripts.ollmo_self_attack as command
    inventory = discover_knobs(ROOT)
    profiles = build_profiles(inventory, live=True, seed=0)[:2]
    monkeypatch.setattr(command, 'discover_knobs', lambda root: inventory)
    monkeypatch.setattr(command, 'build_profiles', lambda *a, **kw: profiles)
    raw = corpus or sample_corpus()
    corpus_path = tmp_path / 'corpus.json'
    corpus_path.write_text(json.dumps(raw))
    proof = tmp_path / 'proof'
    proof.mkdir()
    (proof / 'run.json').write_text(json.dumps(dict(source_digest=inventory['source_digest'], corpus=raw)))
    (proof / 'results.json').write_text(json.dumps(dict(mode='fake', verdict='passed',
        owner_tests={key: dict(status='passed') for key, _ in PRIORITIES})))
    class Client:
        def __init__(self, *a):
            pass
        def get(self, path, *, timeout):
            return HttpResult(200, {})
    monkeypatch.setattr(command, 'JsonHttpClient', Client)
    args = ['--mode', 'live', '--fake-evidence', str(proof / 'results.json'),
            '--corpus', str(corpus_path), '--output', str(tmp_path / 'live'),
            '--regressions', str(tmp_path / 'regressions')]
    return command, args, profiles


def observed_budget_cases(raw, profile, output, **kwargs):
    return dict(profile=profile, runner_status=0, output=str(output), probes={}, cases=[
        dict(case_id=c['case_id'], category=c['category'], state='settled_terminal',
             response_id=c['case_id'], capture_path=str(output / c['case_id']), missing=[], findings=[], payload=truth())
        for c in raw['cases']])


def test_high_risk_live_gate_is_diverse_dependency_closed_and_still_incomplete(tmp_path, monkeypatch):
    from scripts.self_attack_knobs import diverse_live_profiles
    raw = json.loads((ROOT / 'config/self_attack_corpus.json').read_text())
    command, args, _ = live_budget_command(tmp_path, monkeypatch, corpus=raw)
    profiles = build_profiles(discover_knobs(ROOT), live=True, seed=0)
    monkeypatch.setattr(command, 'build_profiles', lambda *a, **kw: profiles)
    chosen = diverse_live_profiles(profiles, 6)
    values = {(k, repr(v)) for p in chosen for k, v in p['settings'].items()}
    available = {(k, repr(v)) for p in profiles for k, v in p['settings'].items()}
    assert values == available
    assert chosen == diverse_live_profiles(profiles, 6)
    assert [p['id'] for p in chosen[:2]] == ['baseline', '836058e8fbf2']
    observed = {}
    def execute(corpus, profile, output, **kwargs):
        observed[profile['id']] = corpus['cases']
        return observed_budget_cases(corpus, profile, output)
    monkeypatch.setattr(command, 'isolated_profile', execute)
    assert command.main([*args, '--profile-limit', '6', '--live-high-risk']) == 2
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert sum(map(len, observed.values())) == 10
    assert {c['category'] for cases in observed.values() for c in cases} == {k for k, _ in PRIORITIES}
    for cases in observed.values():
        ids = {c['case_id'] for c in cases}
        assert all(set(c.get('depends_on', [])) <= ids for c in cases)
        assert all(c in raw['cases'] for c in cases)
    assert result['coverage']['profiles_selected'] == 6
    assert result['coverage']['profiles_available'] == 17
    assert result['verdict'] == 'incomplete'
    assert result['scoped_verdicts'] == dict(deterministic_fake='passed',
        representative_live_gate='passed', full_live_conformance='incomplete')
    report = (tmp_path / 'live/report.md').read_text()
    assert 'Representative live gate: **PASS**' in report
    assert 'Full live conformance: **INCOMPLETE**' in report
    assert sum(b['expected'] for b in result['coverage']['boundaries'].values()) == 10
    assert result['live_budget']['limits']['main_seconds'] == 3600
    assert result['live_budget']['attempts'] == []
    assert 'cannot establish full live conformance' in (tmp_path / 'live/report.md').read_text()


def representative_report_fixture():
    case = dict(case_id='one', category='aspiration_promotion', state='settled_repair_needed',
                response_id='existing-response', capture_path='/retained/capture', missing=[], findings=[])
    return dict(mode='live', verdict='incomplete', fake_conformance_verdict='passed',
        runs=[dict(profile={'id': 'baseline'}, runner_status=0, cases=[case], probes={})],
        owner_tests={key: dict(status='passed') for key, _ in PRIORITIES}, regressions=[],
        coverage=dict(profiles_selected=1, profiles_available=17, design='representative', inventory_only=0,
                      case_selection={'baseline': ['one']}))


@pytest.mark.parametrize('mutation, expected', [
    (lambda r: None, 'passed'),
    (lambda r: r['runs'][0]['cases'][0]['missing'].append('settled_full_truth'), 'incomplete'),
    (lambda r: r['runs'][0]['cases'].clear(), 'incomplete'),
    (lambda r: r['runs'][0]['cases'].append(deepcopy(r['runs'][0]['cases'][0])), 'incomplete'),
    (lambda r: r['runs'][0].update(runner_status=2), 'incomplete'),
    (lambda r: r['runs'][0]['cases'][0].update(state='observing'), 'incomplete'),
    (lambda r: r['runs'][0]['cases'][0].pop('capture_path'), 'incomplete'),
    (lambda r: r['runs'][0]['profile'].update(id='different'), 'incomplete'),
    (lambda r: r['owner_tests'].clear(), 'incomplete'),
    (lambda r: r.update(confirmation_stop_reason='live_confirmation_budget_exhausted'), 'incomplete'),
    (lambda r: r['runs'][0]['cases'][0]['findings'].append(
        dict(code='unrequested_promotion', signature='reserved', detail='Reserved output promoted.')), 'failed'),
    (lambda r: r['owner_tests']['aspiration_promotion'].update(status='failed'), 'failed'),
])
def test_representative_verdict_requires_exact_selected_evidence(mutation, expected):
    from scripts.ollmo_self_attack import verdict_scopes
    result = representative_report_fixture()
    mutation(result)
    before = deepcopy(result)
    scopes = verdict_scopes(result)
    assert scopes['representative_live_gate'] == expected
    assert scopes['full_live_conformance'] == 'incomplete'
    assert scopes['deterministic_fake'] == 'passed'
    assert result == before


def test_full_live_pass_and_failure_remain_independent_of_fake():
    from scripts.ollmo_self_attack import verdict_scopes
    result = representative_report_fixture()
    result['selected_live_cases'] = result['coverage'].pop('case_selection')
    result['coverage']['profiles_available'] = 1
    result['verdict'] = 'passed'
    assert set(verdict_scopes(result).values()) == {'passed'}
    result['verdict'] = 'failed'
    scopes = verdict_scopes(result)
    assert scopes == dict(deterministic_fake='passed', representative_live_gate='failed', full_live_conformance='failed')


def test_combined_report_keeps_full_verdict_when_representative_passes():
    from scripts.ollmo_self_attack import verdict_scopes, render_report
    result = representative_report_fixture()
    live_scopes = verdict_scopes(result)
    result.update(mode='fake', verdict='passed', overall_verdict='incomplete',
                  live=dict(status='incomplete', scoped_verdicts=live_scopes))
    assert verdict_scopes(result) == live_scopes
    report = render_report(result)
    assert 'Deterministic/fake conformance: **PASS**' in report
    assert 'Representative live gate: **PASS**' in report
    assert 'Full live conformance: **INCOMPLETE**' in report
    assert result['overall_verdict'] == 'incomplete'


def test_refresh_completed_report_preserves_evidence_without_execution(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    result = representative_report_fixture()
    result.pop('fake_conformance_verdict')
    proof = tmp_path / 'fake-results.json'
    proof.write_text(json.dumps(dict(mode='fake', verdict='passed')))
    result['fake_evidence'] = str(proof)
    (tmp_path / 'results.json').write_text(json.dumps(result))
    config = dict(source_digest='historical-source', case_selection={'baseline': ['one']})
    original_config = json.dumps(config)
    (tmp_path / 'run.json').write_text(original_config)
    original_report = 'Original report\n\n## Continuation accounting\nOriginal limits retained.\n'
    (tmp_path / 'report.md').write_text(original_report)
    def forbidden(*args, **kwargs):
        pytest.fail('Reporting must not execute, discover current controls or recheck captured truth.')
    for name in ('execute_sweep', 'run_owner_tests', 'run_profile', 'recheck_captures', 'discover_knobs', 'JsonHttpClient'):
        monkeypatch.setattr(command, name, forbidden)
    assert command.main(['--refresh-report', str(tmp_path)]) == 0
    after = json.loads((tmp_path / 'results.json').read_text())
    assert all(after[key] == value for key, value in result.items())
    assert (tmp_path / 'run.json').read_text() == original_config
    assert after['scoped_verdicts'] == dict(deterministic_fake='passed',
        representative_live_gate='passed', full_live_conformance='incomplete')
    assert after['reporting_refresh']['execution_performed'] is False
    assert after['reporting_refresh']['oracle_rechecked'] is False
    assert (Path(after['reporting_refresh']['previous_report']) / 'report.md').read_text() == original_report
    assert 'Original limits retained.' in (tmp_path / 'report.md').read_text()


def test_live_defaults_preserve_corpus_profiles_and_completed_resume_does_not_dispatch(tmp_path, monkeypatch):
    raw = json.loads((ROOT / 'config/self_attack_corpus.json').read_text())
    command, args, profiles = live_budget_command(tmp_path, monkeypatch, corpus=raw)
    monkeypatch.setattr(command, 'isolated_profile', observed_budget_cases)
    assert command.main(args) == 0
    config = json.loads((tmp_path / 'live/run.json').read_text())
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert config['corpus'] == raw
    assert config['profiles'] == profiles
    assert [p['id'] for p in profiles] == ['baseline', '836058e8fbf2']
    assert result['live_budget']['limits'] == dict(main_seconds=3600, confirmation_seconds=900,
                                                 sequence_seconds=1200, replay_seconds=300, attempt_cap=4)
    assert all(v['captured'] == 4 for v in result['coverage']['boundaries'].values())
    before = (tmp_path / 'live/live-budget.json').read_bytes()
    monkeypatch.setattr(command, 'isolated_profile', lambda *a, **kw: pytest.fail('Completed resume dispatched work'))
    monkeypatch.setattr(command, 'JsonHttpClient', lambda *a: pytest.fail('Completed resume contacted server'))
    assert command.main(args) == 0
    assert (tmp_path / 'live/live-budget.json').read_bytes() == before


def test_high_risk_global_timeout_retains_exact_selected_cases_without_retry(tmp_path, monkeypatch):
    import time
    raw = json.loads((ROOT / 'config/self_attack_corpus.json').read_text())
    command, args, _ = live_budget_command(tmp_path, monkeypatch, corpus=raw)
    profiles = build_profiles(discover_knobs(ROOT), live=True, seed=0)
    monkeypatch.setattr(command, 'build_profiles', lambda *a, **kw: profiles)
    class Blocking:
        def __init__(self, *a):
            pass
        def get(self, *a, **kw):
            time.sleep(3)
    monkeypatch.setattr(command, 'JsonHttpClient', Blocking)
    monkeypatch.setattr(command, 'isolated_profile', lambda *a, **kw: pytest.fail('Dispatch after deadline'))
    assert command.main([*args, '--profile-limit', '6', '--live-high-risk', '--live-main-budget', '.05']) == 2
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert sum(len(r['cases']) for r in result['runs']) == 10
    assert result['live_budget']['attempts'] == []
    assert len(result['manual_review']) == 10
    for run in result['runs']:
        assert {c['case_id'] for c in run['cases']} == {c['case_id'] for c in run['corpus']['cases']}
        assert all(c['missing'] and not c['findings'] for c in run['cases'])


def test_hard_main_deadline_interrupts_blocking_preflight_and_persists_incomplete(tmp_path, monkeypatch):
    import time
    command, args, _ = live_budget_command(tmp_path, monkeypatch)
    class Blocking:
        def __init__(self, *a):
            pass
        def get(self, *a, **kw):
            time.sleep(3)
            return HttpResult(200, {})
    monkeypatch.setattr(command, 'JsonHttpClient', Blocking)
    monkeypatch.setattr(command, 'isolated_profile', lambda *a, **kw: pytest.fail('Dispatch after incomplete preflight'))
    start = time.monotonic()
    assert command.main([*args, '--live-main-budget', '.08']) == 2
    assert time.monotonic() - start < 1.5
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert result['verdict'] == 'incomplete'
    assert not any(signatures(r) for r in result['runs'])
    assert result['coverage']['profiles_completed'] == 0
    assert result['live_budget']['attempts'] == []
    assert 'live_main_budget_exhausted' in result['live_budget']['exhausted']
    assert len(result['manual_review']) == 4
    for item in result['manual_review']:
        envelope = json.loads(Path(item['path']).read_text())
        assert envelope['status'] == 'incomplete' and not envelope['automatic_retry']
        assert envelope['corpus'] == sample_corpus()
    report = (tmp_path / 'live/report.md').read_text()
    assert 'Hard live limits' in report and 'Manual review required' in report


def test_hard_deadline_kills_diagnostic_subprocess_without_waiting_for_its_own_timeout(tmp_path, monkeypatch):
    import subprocess
    import time
    command, args, _ = live_budget_command(tmp_path, monkeypatch)
    real_run = subprocess.run
    marker = tmp_path / 'child-survived'
    calls = []
    def blocking_worker(cmd, **kwargs):
        calls.append(kwargs['timeout'])
        return real_run([command.sys.executable, '-c',
            'import pathlib,time; time.sleep(.6); pathlib.Path(' + repr(str(marker)) + ').touch()'], **kwargs)
    monkeypatch.setattr(command.subprocess, 'run', blocking_worker)
    assert command.main([*args, '--live-main-budget', '.15']) == 2
    time.sleep(.65)
    assert not marker.exists()
    assert len(calls) == 1 and calls[0] <= .15


def test_sequence_ceiling_and_expired_resume_never_spawn_another_worker(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    budget = command.LiveBudget(tmp_path, main_seconds=3600, confirmation_seconds=900, attempt_cap=4)
    profile = dict(id='baseline', settings={}, request={}, environment={})
    windows = []
    real_group = command.isolated_group
    def group(raw, profile, output, **kwargs):
        windows.append(kwargs['timeout'])
        return observed_budget_cases(raw, profile, output)
    monkeypatch.setattr(command, 'isolated_group', group)
    command.isolated_profile(sample_corpus(), profile, tmp_path / 'profile', mode='live', base_url='',
                             cycles=300, namespace='test', timeout=600, live_budget=budget)
    assert windows == [1200]
    output = tmp_path / 'expired'
    output.mkdir()
    (output / 'observation-budget.json').write_text(json.dumps(dict(deadline_at=0, seconds=300)))
    monkeypatch.setattr(command.subprocess, 'run', lambda *a, **kw: pytest.fail('Expired sequence spawned'))
    result = real_group(sample_corpus(), profile, output, mode='live', base_url='', cycles=300, namespace='test', timeout=300)
    assert result['runner_status'] == 2
    assert all(c['missing'] and not c['findings'] for c in result['cases'])


def test_live_attempt_counter_and_deadlines_survive_restart(tmp_path):
    import scripts.ollmo_self_attack as command
    budget = command.LiveBudget(tmp_path, main_seconds=3600, confirmation_seconds=900, attempt_cap=1)
    budget.begin_confirmation()
    assert budget.reserve_attempt('original') is None
    before = deepcopy(budget.state)
    resumed = command.LiveBudget(tmp_path, main_seconds=3600, confirmation_seconds=900, attempt_cap=1)
    resumed.begin_confirmation()
    assert resumed.state['confirmation_deadline'] == before['confirmation_deadline']
    assert resumed.state['whole_deadline'] == before['whole_deadline']
    assert resumed.reserve_attempt('original') == 'live_attempt_already_started'
    assert resumed.reserve_attempt('another') == 'live_attempt_cap_exhausted'
    assert len(resumed.state['attempts']) == 1


def test_paired_reproductions_share_four_attempt_cap_and_preserve_failure(tmp_path, monkeypatch):
    command, args, _ = live_budget_command(tmp_path, monkeypatch)
    calls = []
    def divergent(raw, profile, output, **kwargs):
        calls.append(str(output))
        run = observed_budget_cases(raw, profile, output)
        if profile['id'] != 'baseline':
            for case in run['cases']:
                case['payload']['runtime']['request_phase_graph']['intent_obligations'][0]['required'] = False
        return run
    monkeypatch.setattr(command, 'isolated_profile', divergent)
    assert command.main([*args, '--minimize-budget', '100']) == 1
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert len(calls) == 2 + 4
    assert len(result['live_budget']['attempts']) == 4
    assert 'live_attempt_cap_exhausted' in result['live_budget']['exhausted']
    confirmed = [r for r in result['regressions'] if r['status'] == 'confirmed']
    assert len(confirmed) == 1 and Path(confirmed[0]['path']).exists()
    assert confirmed[0]['budget_exhausted']
    assert result['manual_review']


@pytest.mark.parametrize('option', ['--live-confirmation-budget', '--live-replay-timeout'])
def test_confirmation_timeout_is_incomplete_and_saved_regressions_share_cap(tmp_path, monkeypatch, option):
    import time
    command, args, profiles = live_budget_command(tmp_path, monkeypatch)
    saved = tmp_path / 'regressions'
    saved.mkdir()
    for index in range(6):
        (saved / f'{index}.json').write_text(json.dumps(dict(mode='live', corpus=sample_corpus(),
            profile=profiles[0], target=['root', 'injected_failure', str(index)])))
    calls = []
    def slow_replay(raw, profile, output, **kwargs):
        calls.append(str(output))
        if 'regression-replays' in str(output):
            time.sleep(3)
        return observed_budget_cases(raw, profile, output)
    monkeypatch.setattr(command, 'isolated_profile', slow_replay)
    assert command.main([*args, option, '.08']) == 2
    result = json.loads((tmp_path / 'live/results.json').read_text())
    expected_replays = 1 if option == '--live-confirmation-budget' else 4
    assert len(calls) == 2 + expected_replays
    assert len(result['live_budget']['attempts']) == expected_replays
    assert all(r['status'] == 'incomplete' for r in result['regression_replays'])
    assert not any(signatures(r) for r in result['runs'])
    assert result['manual_review']


def test_incomplete_confirmation_does_not_confirm_even_with_matching_findings():
    calls = []
    def replay(*args):
        calls.append(args)
        return dict(runner_status=2, cases=[dict(case_id='root', missing=['settled_full_truth'], findings=[dict(code='bad')])])
    profile = dict(id='baseline', settings={}, request={}, environment={})
    result = minimize_failure(sample_corpus(), profile, ('root', 'bad'), replay)
    assert result['status'] == 'unconfirmed'
    assert result['stop_reason'] == 'confirmation_observation_incomplete' and len(calls) == 1


def test_capture_client_refuses_post_after_observation_deadline(tmp_path):
    class Transport:
        def post(self, *a, **kw):
            pytest.fail('Expired observer reached the transport')
    client = CaptureClient(Transport(), tmp_path, deadline=0)
    from scripts.ollmo_self_attack import LiveBudgetExpired
    with pytest.raises(LiveBudgetExpired):
        client.post('/api/responses', {}, timeout=300)


@pytest.mark.parametrize('option,value', [('--live-main-budget', 'nan'), ('--live-confirmation-budget', 'inf'),
                                         ('--live-sequence-timeout', '0'), ('--live-attempt-cap', '-1')])
def test_invalid_live_limits_are_rejected(option, value):
    from scripts.ollmo_self_attack import main
    with pytest.raises(SystemExit) as exc:
        main([option, value])
    assert exc.value.code == 2


def test_pending_live_cases_are_not_replayed_even_with_unused_attempt_budget(tmp_path, monkeypatch):
    command, args, _ = live_budget_command(tmp_path, monkeypatch)
    calls = []
    def pending(raw, profile, output, **kwargs):
        calls.append(str(output))
        run = observed_budget_cases(raw, profile, output)
        run['cases'][0].update(state='observing', missing=['settled_full_truth'])
        run['cases'][1]['findings'] = [dict(code='a_real_finding', signature='one', detail='Known invariant violation.')]
        run['runner_status'] = 2
        return run
    monkeypatch.setattr(command, 'isolated_profile', pending)
    assert command.main(args) == 1  # A separate observed violation is still a failure.
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert len(calls) == 2
    assert result['live_budget']['attempts'] == []
    assert result['regressions'][0]['stop_reason'] == 'live_pending_response_not_replayed'
    assert any('live_pending_response_not_replayed' in r['reason'] for r in result['manual_review'])


def test_zero_live_attempt_cap_never_replays_saved_regression(tmp_path, monkeypatch):
    command, args, profiles = live_budget_command(tmp_path, monkeypatch)
    saved = tmp_path / 'regressions'
    saved.mkdir()
    (saved / 'case.json').write_text(json.dumps(dict(mode='live', corpus=sample_corpus(),
        profile=profiles[0], target=['root', 'known_failure', 'one'])))
    calls = []
    def observe(*a, **kw):
        calls.append(a[2])
        return observed_budget_cases(*a, **kw)
    monkeypatch.setattr(command, 'isolated_profile', observe)
    assert command.main([*args, '--live-attempt-cap', '0']) == 2
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert len(calls) == 2 and not result['live_budget']['attempts']
    assert result['regression_replays'][0]['status'] == 'incomplete'
    assert result['manual_review']


def test_live_handoff_preserves_ambiguous_response_ids_and_is_get_only(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    from scripts.run_graph_rebase_shadow_corpus import build_manifest, atomic_write_json, load_manifest
    source, dest = tmp_path / 'source', tmp_path / 'dest'
    source.mkdir()
    dest.mkdir()
    raw = sample_corpus()
    profile = dict(id='baseline', settings={}, request={}, environment={})
    config = dict(mode='live', corpus=raw, profiles=[profile], seed=0, run_id='kept', source_digest='old')
    atomic_write_json(source / 'run.json', config)
    folder = source / 'baseline/sequence-1'
    folder.mkdir(parents=True)
    corpus = materialize_corpus(raw, 'kept-baseline-1', folder / 'corpus.json')
    manifest = build_manifest(corpus, folder / 'manifest.json')
    manifest['cases'][0]['state'] = 'submitting'
    original_id = manifest['cases'][0]['response_id']
    atomic_write_json(folder / 'manifest.json', manifest)
    budget = command.LiveBudget(source, main_seconds=3600, confirmation_seconds=900, attempt_cap=4)
    assert budget.reserve_attempt('reductions/reserved/0') is None
    assert command.import_live_evidence(source, dest, raw, [profile], 0) == 'kept'
    assert (dest / 'live-budget.json').read_bytes() == (source / 'live-budget.json').read_bytes()
    copied = load_manifest(dest / 'baseline/sequence-1/manifest.json')
    assert copied['cases'][0]['response_id'] == original_id and copied['cases'][0]['state'] == 'submitting'
    gets = []
    class Observer:
        def __init__(self, *a):
            pass
        def get(self, path, *, timeout):
            gets.append(path)
            return HttpResult(200, dict(id=original_id, lifecycle_state='in_progress'))
        def post(self, *a, **kw):
            pytest.fail('Handoff duplicated an ambiguous response')
    monkeypatch.setattr(command, 'JsonHttpClient', Observer)
    run = command.run_profile(raw, profile, dest / 'baseline/sequence-1', mode='live',
                              base_url='', cycles=1, namespace='kept-baseline-1')
    assert any(original_id in path for path in gets)
    assert run['runner_status'] == 3  # Existing shadow runner's cycle-limit result.
    assert run['cases'][0]['response_id'] == original_id


def test_detached_launch_disconnects_stdio_records_handles_and_deduplicates(tmp_path, monkeypatch):
    import scripts.ollmo_self_attack as command
    launches = []
    class Process:
        pid = 7654321
    def launch(argv, **kwargs):
        launches.append((argv, kwargs))
        return Process()
    monkeypatch.setattr(command.subprocess, 'Popen', launch)
    assert command.launch_detached(['--detach', '--live-after-fake'], tmp_path) == 0
    record = json.loads((tmp_path / 'process.json').read_text())
    assert record['pid'] == record['session_id'] == Process.pid
    assert record['detached']
    assert record['stdout_path'] == record['stderr_path'] == str(tmp_path / 'stdout-stderr.log')
    assert record['live_manifest_path'] == str(tmp_path / 'live/run.json')
    assert record['final_report_path'] == str(tmp_path / 'report.md')
    assert launches[0][1]['start_new_session'] is True
    assert launches[0][1]['stdin'] == command.subprocess.DEVNULL
    assert launches[0][1]['stderr'] == command.subprocess.STDOUT
    monkeypatch.setattr(command.os, 'kill', lambda *a: None)
    assert command.launch_detached(['--detach', '--live-after-fake'], tmp_path) == 0
    assert len(launches) == 1


def test_hard_deadline_is_not_swallowed_by_http_transport_error_handling(tmp_path):
    import time
    import scripts.ollmo_self_attack as command
    from scripts.run_graph_rebase_shadow_corpus import JsonHttpClient
    client = JsonHttpClient('http://127.0.0.1:5001')
    class Opener:
        def open(self, *a, **kw):
            time.sleep(3)
            pytest.fail('Hard HTTP budget was not enforced')
    client._opener = Opener()
    budget = command.LiveBudget(tmp_path, main_seconds=.03, confirmation_seconds=1, attempt_cap=4)
    with pytest.raises(command.LiveBudgetExpired), budget.deadline():
        client.get('/api/runtime_manifest', timeout=5)
    assert budget.state['exhausted'] == ['live_main_budget_exhausted']


def test_live_failure_is_preserved_in_combined_verdict(tmp_path, monkeypatch):
    import subprocess
    command, _, profiles = live_budget_command(tmp_path, monkeypatch)
    monkeypatch.setattr(command, 'isolated_profile', observed_budget_cases)
    monkeypatch.setattr(command, 'run_owner_tests', lambda *a, **kw: {k: dict(status='passed') for k, _ in PRIORITIES})
    output = tmp_path / 'combined'
    def live_child(argv, **kwargs):
        live = output / 'live'
        live.mkdir()
        (live / 'results.json').write_text(json.dumps(dict(verdict='failed')))
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(command.subprocess, 'run', live_child)
    assert command.main(['--corpus', str(tmp_path / 'corpus.json'), '--output', str(output),
                         '--regressions', str(tmp_path / 'regressions'), '--live-after-fake']) == 1
    result = json.loads((output / 'results.json').read_text())
    assert result['verdict'] == 'passed' and result['overall_verdict'] == 'failed'


def test_expired_main_resume_preserves_already_recorded_invariant_findings(tmp_path, monkeypatch):
    command, args, profiles = live_budget_command(tmp_path, monkeypatch)
    args += ['--live-attempt-cap', '0']
    def interrupted(raw, profile, output, **kwargs):
        if profile['id'] != 'baseline':
            raise KeyboardInterrupt()
        run = observed_budget_cases(raw, profile, output)
        run['cases'][0]['findings'] = [dict(code='known_violation', signature='kept', detail='Recorded before interruption.')]
        return run
    monkeypatch.setattr(command, 'isolated_profile', interrupted)
    with pytest.raises(KeyboardInterrupt):
        command.main(args)
    path = tmp_path / 'live/live-budget.json'
    budget = json.loads(path.read_text())
    budget['main_deadline'] = 0
    path.write_text(json.dumps(budget))
    monkeypatch.setattr(command, 'isolated_profile', lambda *a, **kw: pytest.fail('Expired resume dispatched work'))
    assert command.main(args) == 1
    result = json.loads((tmp_path / 'live/results.json').read_text())
    assert ('root', 'known_violation', 'kept') in signatures(result['runs'][0])
    assert result['manual_review']
