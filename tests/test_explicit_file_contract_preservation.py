"""Explicit file contract survives incomplete derived planning; no live models."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from ollmo_core.inference import detect_text_artifact_requests
from ollmo_g.request_phase_graph import build_request_phase_graph

PROMPT = (
    'Create exactly two small local files. note.txt must contain exactly "Ready." '
    'followed by a newline. status.json must be valid JSON with exactly '
    '{"status":"ok"}. Return links to both saved files.'
)
EXPLICIT_FILES = [
    {'extension': 'txt', 'source': 'explicit_extension', 'source_name': 'note'},
    {'extension': 'json', 'source': 'explicit_extension', 'source_name': 'status'},
]
PROMISE = 'I will create note.txt with the requested content and provide its link.'


def file_identities(requests):
    return {(item['source_name'], item['extension']) for item in requests}


def obligation_identities(graph):
    return {(item['target_name'], item['target_extension'])
            for item in graph['intent_obligations'] if item.get('kind') == 'text_artifact'}


def replay_file_contract(root, *, files=('note.txt',), reduced_detector=True):
    """Real graph, writer, Closure, terminal and frame owners in temporary roots."""
    import ollmo_webserver as web
    from tests.fake_backends import FakeBackendHarness

    request = {'prompt': PROMPT, 'ghost_route': True, 'response_id': 'explicit-files-fixture'}
    original = build_request_phase_graph(PROMPT, request_payload=request)
    original_copy = copy.deepcopy(original)
    route = {'capability': 'chat', 'route_runtime': {'request_phase_graph': original}}
    derived = EXPLICIT_FILES[:1] if reduced_detector else EXPLICIT_FILES
    with patch('ollmo_g.request_phase_graph.detect_text_artifact_requests', return_value=copy.deepcopy(derived)):
        graph = build_request_phase_graph(PROMPT, request_payload=request, route_payload=route,
                                         response_payload={'output_text': PROMISE})
    assert original == original_copy
    with FakeBackendHarness(root=root) as harness, \
         patch.object(web, 'ARTIFACT_OUTPUTS_DOCUMENTS_DIR', harness.documents_dir):
        artifacts = []
        saved = []
        for filename in files:
            path = harness.documents_dir / filename
            path.write_text('{"status":"ok"}\n' if filename.endswith('.json') else 'Ready.\n')
            artifacts.append({'type':'text', 'path':str(path), 'extension':path.suffix.lstrip('.'),
                              'artifact_ref':'artifact:fixture-' + filename,
                              'text_artifact_request': {'extension':path.suffix.lstrip('.'), 'source_name':path.stem, 'source':'explicit_extension'}})
            saved.append({'path':str(path), 'text_artifact_request':artifacts[-1]['text_artifact_request']})
        payload = harness.build_response_payload(response_id=request['response_id'], capability='chat',
            mode='responses_chat', output_text=PROMISE,
            source_payload={'content':PROMISE, 'artifacts':artifacts, 'saved_text_artifacts':saved})
        payload['runtime'] = {'request_phase_graph':graph}
        payload['late_fill'] = {'status':'pending', 'pending_branches':copy.deepcopy(graph['downstream_branches']),
                               'failed_branches':[], 'completed_branches':[]}
        closure_before = web._build_graph_closure_review(PROMISE, request_payload=request, artifact_payload=payload)
        terminal, terminal_status = web._LATE_FILL_RUNTIME.finalize_terminal_materialization_contract(
            payload, request_payload=request, route_payload=None, artifact_gap=None, terminal_status='completed')
        finalized = web._finalize_response_frame_payload(terminal, request_payload=request, persist=False)
        return {
            'prompt':PROMPT, 'model_output':PROMISE,
            'original_explicit_files':EXPLICIT_FILES,
            'initial_detected_requests':detect_text_artifact_requests(PROMPT),
            'initial_graph':original, 'derived_requests':derived, 'graph':graph,
            'saved_files':list(files), 'closure_before_terminal':closure_before,
            'terminal_materialization':terminal.get('late_fill'),
            'closure':terminal.get('runtime',{}).get('graph_closure_review'),
            'terminal_status':terminal_status, 'lifecycle_state':finalized.get('lifecycle_state'),
        }


def test_explicit_two_files_survive_incomplete_detector_and_block_closure(tmp_path):
    result = replay_file_contract(tmp_path)
    assert obligation_identities(result['initial_graph']) == {('note','txt'),('status','json')}
    assert len(result['derived_requests']) == 1
    assert obligation_identities(result['graph']) == {('note','txt'),('status','json')}
    assert result['closure']['status'] != 'fulfilled'
    assert result['terminal_materialization']['final_materialization_contract_status'] != 'fulfilled'
    assert result['lifecycle_state'] != 'completed'


def test_complete_original_files_can_still_close(tmp_path):
    result = replay_file_contract(tmp_path, files=('note.txt','status.json'))
    assert obligation_identities(result['graph']) == {('note','txt'),('status','json')}
    assert result['closure']['status'] == 'fulfilled'
    assert result['lifecycle_state'] == 'completed'


@pytest.mark.parametrize('wrong_file', ['status.txt', 'other.json'])
def test_same_count_wrong_file_identity_does_not_close(tmp_path, wrong_file):
    result = replay_file_contract(tmp_path, files=('note.txt', wrong_file))
    assert obligation_identities(result['graph']) == {('note','txt'),('status','json')}
    assert result['closure']['status'] != 'fulfilled'
    assert result['lifecycle_state'] != 'completed'


@pytest.mark.parametrize('prompt, expected', [
    (PROMPT, {('note','txt'), ('status','json')}),
    ('Create exactly two files:\nnote.txt\nstatus.json', {('note','txt'), ('status','json')}),
    ('Create exactly three local files. alpha.md must contain notes. '
     'beta.json must contain data. gamma.csv must contain rows.',
     {('alpha','md'), ('beta','json'), ('gamma','csv')}),
    ('Create exactly two files: overview.html and prices.json. Return a JSON object with links.',
     {('overview','html'), ('prices','json')}),
])
def test_explicit_named_counts_and_types_enter_premodel_contract(prompt, expected):
    assert file_identities(detect_text_artifact_requests(prompt)) == expected
    graph = build_request_phase_graph(prompt)
    assert obligation_identities(graph) == expected
    assert graph['prompt_intent']['text_artifact_output_count'] == len(expected)
    assert len(graph['downstream_branches']) == len(expected)


@pytest.mark.parametrize('prompt', [
    'Return a JSON object with the answer.',
    'Create a summary of these files. status.json must contain valid data.',
    'Explain how to create two files. status.json must contain valid data.',
    'Do not create files. status.json must contain valid data.',
    'Create two files. Read the existing input files. status.json must contain valid data.',
    'Create two files. "status.json must contain valid data."',
])
def test_file_declaration_authority_does_not_promote_response_or_reference_json(prompt):
    assert not any(item['extension'] == 'json' for item in detect_text_artifact_requests(prompt))


def test_model_prose_and_reduced_planner_cannot_replace_accepted_identity():
    original = build_request_phase_graph(PROMPT)
    branches = copy.deepcopy(original['downstream_branches'])
    snapshot = copy.deepcopy(original)
    for carrier in ('route', 'response'):
        kwargs = {'route_payload':{'route_runtime':{'request_phase_graph':original}}} if carrier == 'route' else {
            'response_payload':{'runtime':{'request_phase_graph':original}}}
        response = kwargs.setdefault('response_payload', {})
        response['output_text'] = PROMISE
        response.setdefault('runtime', {})['execution_planner'] = {'deferred_branches':branches[:1]}
        with patch('ollmo_g.request_phase_graph.detect_text_artifact_requests', return_value=EXPLICIT_FILES[:1]):
            rebuilt = build_request_phase_graph(PROMPT, **kwargs)
        assert obligation_identities(rebuilt) == {('note','txt'),('status','json')}
        actual = {b['branch_id']:b for b in rebuilt['downstream_branches']}
        for branch in branches:
            for key in ('branch_id','phase_id','depends_on','artifact_request'):
                assert actual[branch['branch_id']][key] == branch[key]
    assert original == snapshot


def test_unrelated_turn_does_not_inherit_file_obligations():
    original = build_request_phase_graph(PROMPT)
    graph = build_request_phase_graph('Tell me a short joke.',
        route_payload={'route_runtime':{'request_phase_graph':original}})
    assert not obligation_identities(graph)


def test_smaller_response_graph_does_not_release_route_contract():
    original = build_request_phase_graph(PROMPT)
    with patch('ollmo_g.request_phase_graph.detect_text_artifact_requests', return_value=EXPLICIT_FILES[:1]):
        smaller = build_request_phase_graph(PROMPT)
        rebuilt = build_request_phase_graph(PROMPT,
            route_payload={'route_runtime':{'request_phase_graph':original}},
            response_payload={'runtime':{'request_phase_graph':smaller}})
    assert obligation_identities(smaller) == {('note','txt')}
    assert obligation_identities(rebuilt) == {('note','txt'),('status','json')}
    assert len(rebuilt['downstream_branches']) == 2


def test_derived_plan_cannot_repurpose_accepted_branch_identity():
    original = build_request_phase_graph(PROMPT)
    plan = copy.deepcopy(original['downstream_branches'])
    plan[1]['artifact_request']['extension'] = 'md'
    plan[1]['text_artifact_extension'] = 'md'
    plan[1]['depends_on'] = []
    rebuilt = build_request_phase_graph(PROMPT,
        request_payload={'downstream_branches':plan},
        route_payload={'route_runtime':{'request_phase_graph':original}})
    actual = {branch['branch_id']:branch for branch in rebuilt['downstream_branches']}
    for branch in original['downstream_branches']:
        for key in ('phase_id','artifact_request','depends_on'):
            assert actual[branch['branch_id']][key] == branch[key]


def test_extra_derived_request_keeps_original_owners_and_existing_extra_behavior():
    original = build_request_phase_graph(PROMPT)
    extra = {'source_name':'appendix', 'extension':'md', 'source':'explicit_extension'}
    with patch('ollmo_g.request_phase_graph.detect_text_artifact_requests', return_value=[*EXPLICIT_FILES,extra]):
        graph = build_request_phase_graph(PROMPT, route_payload={'route_runtime':{'request_phase_graph':original}})
    assert obligation_identities(graph) == {('note','txt'),('status','json'),('appendix','md')}
    branches = graph['downstream_branches']
    assert len(branches) == len({b['branch_id'] for b in branches}) == 3
    assert len({b['phase_id'] for b in branches}) == 3
    for branch in original['downstream_branches']:
        actual = next(b for b in branches if b['branch_id'] == branch['branch_id'])
        assert actual['artifact_request'] == branch['artifact_request']


@pytest.mark.parametrize('status', ['waived', 'superseded'])
def test_newer_authoritative_branch_release_remains_visible(status):
    original = build_request_phase_graph(PROMPT)
    newer = copy.deepcopy(original)
    newer['downstream_branches'][1]['status'] = status
    newer['intent_obligations'][1]['status'] = status
    with patch('ollmo_g.request_phase_graph.detect_text_artifact_requests', return_value=EXPLICIT_FILES[:1]):
        graph = build_request_phase_graph(PROMPT,
            route_payload={'route_runtime':{'request_phase_graph':original}},
            response_payload={'runtime':{'request_phase_graph':newer}})
    assert obligation_identities(graph) == {('note','txt'),('status','json')}
    assert next(o for o in graph['intent_obligations'] if o.get('target_name')=='status')['status'] == status
    assert next(b for b in graph['downstream_branches'] if b.get('text_artifact_source_name')=='status')['status'] == status
