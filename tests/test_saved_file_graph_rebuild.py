"""Saved-read grammar and planner handoff regressions, no runtime or models."""
import copy
import pytest

from ollmo_g.request_phase_graph import build_request_phase_graph, _guard_unbound_saved_file_consumers
from ollmo_g.execution_planner import _apply_phase_graph_follow_up_result

TAIL = ('Read the actually saved sample.json again. Create report.html from the read data. '
        'Use self-contained HTML with embedded CSS. Return exactly these two files.')


@pytest.mark.parametrize('producer', [
    'Create sample.json containing {"values":[7,11]}. ',
    'Create a new file named sample.json containing {"values":[7,11]} and save it. ',
])
def test_create_named_file_is_a_saved_read_producer(producer):
    graph = build_request_phase_graph(producer + TAIL)
    branches = {b['text_artifact_extension']:b for b in graph['downstream_branches']}
    consumer = branches['html']
    assert consumer['artifact_request'].get('saved_file_dependency')
    assert consumer['depends_on'] == [branches['json']['phase_id']]


def test_planner_roundtrip_retains_exact_file_contracts_and_branch_ids():
    prompt = 'Save sample.json with values [7,11]. ' + TAIL
    original = build_request_phase_graph(prompt)
    before = copy.deepcopy(original)
    _, planner = _apply_phase_graph_follow_up_result({},
        follow_up_branches=original['downstream_branches'], trigger='test', prompt=prompt)
    rebuilt = build_request_phase_graph(prompt,
        response_payload={'output_text':'Preparation complete.', 'runtime':{'execution_planner':planner}})
    assert original == before
    actual = rebuilt['downstream_branches']
    expected = original['downstream_branches']
    assert len(actual) == len(expected) == 2
    for left, right in zip(actual, expected):
        for key in ('branch_id', 'phase_id', 'depends_on', 'artifact_request'):
            assert left[key] == right[key], (key, left, right)
    assert not any(b.get('branch_contract_error') for b in actual)


def test_incomplete_materializer_is_not_a_filename_dot():
    branch = {'branch_id':'incomplete', 'phase_id':'phase-2', 'capability':'chat',
              'stage_direction':'materialize_requested_text_artifact', 'depends_on':['phase-1']}
    _guard_unbound_saved_file_consumers([branch], 'Save sample.json with values [7,11]. ' + TAIL)
    assert 'artifact_request' not in branch


@pytest.mark.parametrize('producer', ['Do not create sample.json. ', 'Maybe create sample.json later. '])
def test_create_does_not_override_negation_or_reservation(producer):
    graph = build_request_phase_graph(producer + TAIL)
    assert not any((b.get('artifact_request') or {}).get('saved_file_dependency')
                   for b in graph['downstream_branches'])


@pytest.mark.parametrize('missing', ['producer', 'consumer'])
def test_missing_artifact_request_never_creates_a_dependency_or_crashes(missing):
    prompt = 'Save sample.json with values [7,11]. ' + TAIL
    branches = build_request_phase_graph(prompt)['downstream_branches']
    producer = next(b for b in branches if b['text_artifact_extension'] == 'json')
    consumer = next(b for b in branches if b['text_artifact_extension'] == 'html')
    consumer['depends_on'] = ['phase-1']
    consumer['artifact_request'].pop('saved_file_dependency')
    (producer if missing == 'producer' else consumer).pop('artifact_request')
    _guard_unbound_saved_file_consumers(branches, prompt)
    assert consumer['branch_contract_error'] == 'saved_file_dependency_unbound'
