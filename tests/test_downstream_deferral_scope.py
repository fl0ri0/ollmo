"""Category-scoped downstream constraints; deterministic isolated runtime truth."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from ollmo_core.inference import detect_text_artifact_requests, extract_text_artifact_payloads
from ollmo_g.intent import analyze_prompt_intent, materialization_is_deferred
from ollmo_g.request_phase_graph import build_request_phase_graph, _prompt_reserves_materialization_capability
from tests.test_generated_image_artifact_routing import semantics
from tests import test_explicit_file_contract_preservation as files

EXCLUSIONS = 'Do not create images, SVG, JavaScript, audio, bundles, web requests, external resources or other deliverables.'


@pytest.mark.parametrize('prompt,reserved', [
    ('Plan an image, keep it as an option.', True),
    ('Wenn ein Bild sinnvoll wäre, halte es nur als reservierte Option fest.', True),
    ('Plan three image ideas, generate only the second. Keep the first and third as options.', True),
    ('Generate one PNG image, keep the SVG as a reserved option.', False),
    ('Plan an image and an audio clip, keep it as an option.', False),
    ('Plan image ideas. Plan audio candidates. Keep the first and third as options.', False),
])
def test_image_reservation_pronouns_keep_their_nearest_artifact_scope(prompt, reserved):
    assert _prompt_reserves_materialization_capability(
        analyze_prompt_intent(prompt), 'image_generation',
    ) is reserved


def pending(owner, prompt):
    graph = build_request_phase_graph(prompt, request_payload={'prompt': prompt, 'ghost_route': True})
    branches = owner.extract_pending_deferred_branches(
        route_payload={'route_runtime': {'request_phase_graph': graph}})
    return graph, branches


@pytest.mark.parametrize('prompt,extensions,image_count', [
    (files.PROMPT + ' ' + EXCLUSIONS, {'txt', 'json'}, 0),
    ('Create index.html and a separate styles.css. No image.', {'html', 'css'}, 0),
    ('Generate one PNG image. Create index.html. Do not create SVG.', {'html'}, 1),
    (files.PROMPT + ' Do not create confusion.', {'txt', 'json'}, 0),
    (files.PROMPT + ' Only the current phase is the visible result. ' + EXCLUSIONS, {'txt', 'json'}, 0),
    (files.PROMPT + ' Do not create images, SVG,\nJavaScript, audio or external resources.', {'txt', 'json'}, 0),
])
def test_unrelated_constraints_preserve_exact_executable_contract(semantics, prompt, extensions, image_count):
    graph, branches = pending(semantics, prompt)
    assert not semantics._request_graph_defers_downstream_execution(graph)
    assert {b['text_artifact_extension'] for b in branches if b.get('text_artifact_extension')} == extensions
    assert sum(b['capability'] == 'image_generation' for b in branches) == image_count
    if extensions == {'txt', 'json'}:
        assert len(branches) == 2
        assert files.obligation_identities(graph) == {('note', 'txt'), ('status', 'json')}


@pytest.mark.parametrize('constraint,expected', [
    ('Do not create the JSON file.', {'txt'}),
    ('Do not create status.json.', {'txt'}),
    ('Do not create files.', set()),
    ('Do not create any files.', set()),
    ('Plan but do not materialize.', set()),
    ('Do not materialize downstream artifacts.', set()),
    ('Do not continue downstream.', set()),
    ('Do not generate the image.', {'txt', 'json'}),
])
def test_defer_preserves_obligations_and_only_filters_matching_branches(semantics, constraint, expected):
    graph, branches = pending(semantics, files.PROMPT + ' ' + constraint)
    assert files.obligation_identities(graph) == {('note', 'txt'), ('status', 'json')}
    assert len([b for b in graph['downstream_branches'] if b.get('text_artifact_extension')]) == 2
    assert {b['text_artifact_extension'] for b in branches} == expected
    # A legacy planner's duplicate view cannot sneak the deferred file back in.
    selected = semantics.extract_pending_deferred_branches(route_payload={'route_runtime': {
        'request_phase_graph': graph,
        'execution_planner': {'deferred_branches': copy.deepcopy(graph['downstream_branches']),
                              'deferred_capabilities': ['chat']}}})
    assert {b['text_artifact_extension'] for b in selected} == expected


@pytest.mark.parametrize('prompt,extension', [
    ('Plan the HTML but do not save it.', 'html'),
    ('Describe the JSON only; do not create it.', 'json'),
])
def test_deferred_pronoun_uses_local_category(prompt, extension):
    analysis = analyze_prompt_intent(prompt)
    assert materialization_is_deferred(analysis, {'text_artifact_extension': extension})
    assert not materialization_is_deferred(analysis, {'text_artifact_extension': 'css'})


def test_image_deferral_does_not_stop_text_sibling(semantics):
    graph, branches = pending(semantics,
        'Generate one image. Create index.html. Do not generate the image yet.')
    assert graph['prompt_intent']['explicit_visual_defer_materialization']
    assert not [b for b in branches if b['capability'] == 'image_generation']
    assert {b.get('text_artifact_extension') for b in branches} == {'html'}


@pytest.mark.parametrize('constraint,saved,closed', [
    (EXCLUSIONS, (), False),
    (EXCLUSIONS, ('note.txt',), False),
    (EXCLUSIONS, ('note.txt', 'status.json'), True),
    ('Do not create the JSON file.', ('note.txt',), False),
    ('Do not create files.', (), False),
])
def test_terminal_closure_judges_preserved_actual_files(tmp_path, monkeypatch, constraint, saved, closed):
    monkeypatch.setattr(files, 'PROMPT', files.PROMPT + ' ' + constraint)
    result = files.replay_file_contract(tmp_path, files=saved, reduced_detector=False)
    assert files.obligation_identities(result['graph']) == {('note', 'txt'), ('status', 'json')}
    assert (result['closure']['status'] == 'fulfilled') is closed
    assert (result['lifecycle_state'] == 'completed') is closed
    if not closed:
        assert result['lifecycle_state'] in {'repair_needed', 'late_fill_failed'}


def test_saved_file_producer_consumer_executes_with_unrelated_exclusions(tmp_path, monkeypatch):
    from tests import test_saved_file_consumer_path as saved
    monkeypatch.setattr(saved, 'PROMPT', saved.PROMPT + ' ' + EXCLUSIONS)
    saved.test_complete_late_fill_executes_save_then_read_consumer(tmp_path, monkeypatch, 'Save')


@pytest.mark.parametrize('constraint,expected_names,closed', [
    (EXCLUSIONS, {'note.txt', 'status.json'}, True),
    ('Do not create the JSON file.', {'note.txt'}, False),
    ('Do not create files.', set(), False),
])
def test_request_schedules_only_executable_files(tmp_path, constraint, expected_names, closed):
    import ollmo_webserver as web
    from tests.test_generated_image_artifact_routing import _GeneratedWebHarness

    prompt = files.PROMPT + ' ' + constraint
    prose = 'note.txt\nReady.\nstatus.json\n{"status":"ok"}'
    assert extract_text_artifact_payloads(prose, detect_text_artifact_requests(prompt)) == []

    class Harness(_GeneratedWebHarness):
        def _execute_chat_backend_request(self, **kwargs):
            return prose

        def _invoke_internal_api_json_route(self, path=None, *, payload=None, upload=None):
            data = payload or {}
            requests = data.get('text_artifact_requests') or [data.get('artifact_request') or {}]
            saved = []
            for request in requests:
                extension = request['extension']
                assert extension in {'txt', 'json'}
                path = self.documents_dir / (request['source_name'] + '.' + extension)
                path.write_text('Ready.\n' if extension == 'txt' else '{"status":"ok"}\n')
                saved.append({'path': str(path), 'text_artifact_request': request})
            return {'content': 'Saved.', 'saved_text_path': saved[0]['path'], 'saved_text_artifacts': saved}, 200

    scheduled = []
    with Harness(root=tmp_path) as harness:
        def complete_inline(**kwargs):
            scheduled.append(copy.deepcopy(kwargs))
            web._complete_response_late_fill(**kwargs)
            return True
        with patch.object(web, '_schedule_response_late_fill', side_effect=complete_inline), \
             patch.object(web, 'ARTIFACT_OUTPUTS_DOCUMENTS_DIR', harness.documents_dir):
            initial, status = harness.post_response({'response_id': 'scope-files', 'prompt': prompt, 'ghost_route': True})
        assert status == 200
        assert len(scheduled) == bool(expected_names)
        if scheduled:
            assert len(scheduled[0]['artifact_gap']['pending_branches']) == len(expected_names)
        final = harness.response_state('scope-files')['response_payload']
        assert (final['lifecycle_state'] == 'completed') is closed, final.get('runtime', {}).get('graph_closure_review')
        if closed:
            assert final['late_fill']['status'] == 'completed', final.get('late_fill')
        assert files.obligation_identities(final['runtime']['request_phase_graph']) == {('note', 'txt'), ('status', 'json')}
        assert {p.name for p in harness.documents_dir.iterdir()} == expected_names
