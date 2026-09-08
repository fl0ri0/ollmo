"""Small owner regressions; no server, ledger, or model execution."""
import copy
from pathlib import Path

import pytest

from ollmo_core.inference import detect_text_artifact_requests
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_server.responses_runtime import late_fill_has_actionable_repair_work
from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner
from ollmo_services.responses import build_canonical_response_artifacts
from ollmo_g.request_meta import extract_request_meta


@pytest.mark.parametrize('prompt', [
    'Erstelle genau zwei Dateien: daten.json und bericht.html. Einfaches eingebettetes CSS genügt.',
    'Create exactly two files: data.json and report.html. Use inline CSS.',
    'Save report.html with embedded CSS and inline JavaScript.',
])
def test_embedded_formats_are_not_additional_files(prompt):
    requests = detect_text_artifact_requests(prompt)
    assert {r['extension'] for r in requests} <= {'html', 'json'}


def projection_owner():
    owner = object.__new__(ResponsesRequestRuntimeOwner)
    owner.hooks = {'normalize_capability': lambda value: value}
    owner.capability_chat = 'chat'
    return owner


def test_projected_repair_retains_exact_executable_binding(tmp_path):
    request = {'extension': 'html', 'source_name': 'report',
               'source': 'closure_link_rebind', 'target_path': str(tmp_path / 'report.html')}
    contract = {'kind': 'ollmo.repair_rebuild_contract', 'contract_id': 'repair-contract-cohort',
                'branch_id': 'cohort', 'phase_id': 'cohort', 'capability': 'chat',
                'output_type': 'text', 'status': 'promoted', 'auto_execute': True,
                'authority': 'closure_review_runtime_truth', 'promotion_source': 'graph_closure_review',
                'execution_policy': 'schedule_late_fill_branch', 'repair_work_available': True,
                'repair_action': 'rebind_dependency_evidence', 'artifact_request': request,
                'content_payload': 'Repair the exact saved target link.',
                'content_payload_source': 'closure_linked_artifact_binding_review'}
    feedback = {'status': 'repair_required', 'items': [dict(contract)],
                'repair_rebuild_contracts': [contract],
                'repair_loop': {'status': 'promoted', 'auto_execute': True,
                                'repair_work_available': True, 'promoted_contracts': [contract]}}
    original = copy.deepcopy(feedback)
    gap = projection_owner()._ghost_repair_feedback_gap({'ghost_repair_feedback': feedback})
    assert feedback == original
    branch = gap['pending_branches'][0]
    assert branch['branch_id'] == 'repair-cohort'
    terminal = {**gap, 'status': 'completed', 'pending_branches': [],
                'final_materialization_contract_status': 'fulfilled',
                'completed_branches': [{**branch, 'status': 'fulfilled'}]}
    assert not late_fill_has_actionable_repair_work(terminal)
    reconciled = LateFillRuntimeOwner._reconcile_terminal_satisfied_repair_loop(terminal)
    assert reconciled['repair_loop']['resolved_contract_count'] == 1
    assert reconciled['repair_loop']['promoted_contracts'] == []
    assert reconciled['repair_loop']['resolved_contracts'][0]['execution_binding'] == {
        'contract_id': contract['contract_id'], 'branch_id': 'repair-cohort', 'phase_id': 'repair-cohort'}
    # The observation/completion owner must not guess bindings for old records.
    legacy = copy.deepcopy(terminal)
    legacy['repair_loop']['promoted_contracts'][0].pop('execution_binding')
    assert late_fill_has_actionable_repair_work(legacy)
    for field in ('branch_id', 'phase_id', 'repair_contract_id'):
        wrong = copy.deepcopy(terminal)
        wrong['completed_branches'][0][field] = 'unrelated'
        assert late_fill_has_actionable_repair_work(wrong)
    wrong = copy.deepcopy(terminal)
    wrong['completed_branches'][0]['artifact_request'] = {
        **wrong['completed_branches'][0]['artifact_request'],
        'target_path': str(tmp_path / 'other.html'),
    }
    assert late_fill_has_actionable_repair_work(wrong)


def test_saved_file_read_cannot_become_preparation_text_dependency():
    graph = build_request_phase_graph(
        'Speichere daten.json mit den Werten [7,11]. Lies anschließend die tatsächlich '
        'gespeicherte JSON-Datei wieder ein. Erstelle daraus bericht.html mit der Summe.'
    )
    branches = {b.get('text_artifact_extension'): b for b in graph['downstream_branches']}
    consumer = branches['html']
    assert consumer['depends_on'] == [branches['json']['phase_id']]
    assert consumer['artifact_request']['saved_file_dependency']['producer_branch_id'] == branches['json']['branch_id']
    assert consumer['content_payload_source'] == 'saved_file_read_required'


@pytest.mark.parametrize('prompt', [
    'Save data.json and create report.html independently.',
    'Read the saved notes. Create poster.html about mountains.',
    'Do not read the saved file. Create report.html from data in this request.',
    'Speichere daten.json. Lies die gespeicherte Datei nicht wieder ein. Erstelle daraus bericht.html.',
])
def test_independent_or_negated_reads_do_not_create_dependency_blocks(prompt):
    graph = build_request_phase_graph(prompt)
    assert not any(b.get('branch_contract_error') == 'saved_file_dependency_unbound'
                   for b in graph['downstream_branches'])


def test_explicit_stylesheet_sibling_remains_requested():
    requests = detect_text_artifact_requests(
        'Create report.html with inline CSS. Also save a separate theme.css file.'
    )
    assert {(r['source_name'], r['extension']) for r in requests} == {
        ('report', 'html'), ('theme', 'css')}


def test_coalescing_cannot_erase_declared_file_dependency():
    owner = object.__new__(LateFillRuntimeOwner)
    owner.normalize_capability = lambda value: value
    owner.branch_id = lambda branch: branch['branch_id']
    owner.branch_capability = lambda branch: branch['capability']
    owner.artifact_type_for_capability = lambda capability: 'text'
    specs = []
    for name, extension, dependencies in [('data', 'json', ['phase-1']), ('report', 'html', ['data'])]:
        branch = {'branch_id': name, 'phase_id': name, 'capability': 'chat',
                  'requires_artifact': True, 'output_type': 'text',
                  'stage_direction': 'materialize_requested_text_artifact',
                  'text_artifact_extension': extension, 'text_artifact_source_name': name,
                  'depends_on': dependencies}
        specs.append({'branch': branch, 'capability': 'chat'})
    assert len(owner._coalesce_required_text_artifact_branch_specs(specs)) == 2


def test_saved_read_gate_rejects_preparation_even_with_a_file_path(tmp_path):
    owner = object.__new__(LateFillRuntimeOwner)
    owner.branch_capability = lambda branch: branch.get('capability')
    owner.failed_dependency_ids_for_branch = lambda *a, **kw: []
    owner.branch_dependency_payload = lambda *a, **kw: {'content_payload': 'same values',
                                                       'content_payload_source': 'current_phase_output'}
    path = tmp_path / 'data.json'
    path.write_text('{"values":[13,17]}')
    branch = {'branch_id': 'consumer', 'capability': 'chat', 'depends_on': ['phase-1'],
              'branch_contract_error': 'saved_file_dependency_unbound',
              'materialization_blocked': True, 'repair_action': 'repair_dependency_chain',
              'file_path': str(path)}
    error = owner.repair_branch_execution_error(branch, current_payload={'output_text': 'same values'})
    assert error and error['code'] == 'DEPENDENCY_CHAIN_REPAIR_REQUIRED'


def test_closure_cannot_discharge_unbound_read_by_matching_files(tmp_path):
    prompt = ('Save data.json and other.json. Read the actually saved file again. '
              'Create report.html from the read data.')
    graph = build_request_phase_graph(prompt)
    branches = graph['downstream_branches']
    artifacts = []
    for branch in branches:
        extension = branch.get('text_artifact_extension')
        if not extension:
            continue
        path = tmp_path / (branch['text_artifact_source_name'] + '.' + extension)
        path.write_text('{"values":[13,17]}' if extension == 'json' else '<html>13 17 30</html>')
        artifacts.append({'type': 'text', 'kind': 'text', 'extension': extension, 'path': str(path)})
    owner = ResponseSemanticsRuntimeOwner(hooks={
        'build_canonical_response_artifacts': build_canonical_response_artifacts,
        'normalize_capability_list': lambda values: values if isinstance(values, list) else [],
        'extract_request_meta': extract_request_meta,
        'extract_responses_prompt': lambda payload: payload.get('prompt', ''),
        'resolve_semantic_review_artifact_path': lambda path: Path(path),
        'load_running_instances': lambda: [],
        'merge_instances_with_runtime_status': lambda instances, **kw: instances,
    })
    payload = {'output_text': 'All values match.', 'artifacts': artifacts,
               'runtime': {'request_phase_graph': graph},
               'late_fill': {'status': 'completed', 'final_materialization_contract_status': 'fulfilled',
                             'completed_branches': [{**b, 'status': 'fulfilled'} for b in branches]}}
    review = owner.build_graph_closure_review(payload['output_text'],
        request_payload={'prompt': prompt, 'ghost_route': True}, artifact_payload=payload)
    check = next(c for c in review['checks'] if c.get('text_artifact_extension') == 'html')
    assert check['status'] == 'blocked'
    assert check['evidence'] == 'saved_file_dependency_unbound'
