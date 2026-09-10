"""Registry consumes accepted frame identities, not stale saved-path shortcuts."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from ollmo_services import artifact_registry as registry
from ollmo_services.response_frames import attach_response_frame
from ollmo_services.responses import hoist_response_output_surfaces


def registry_handoff_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = [('note.txt', 'Ready.\n'), ('status.json', '{"status":"ok"}\n')]
    stale_ref = 'artifact:coalesced-provider-output'
    artifacts, slots, results = [], [], []
    for i, (name, content) in enumerate(files, 1):
        path = root / name
        path.write_text(content)
        binding = {'branch_id': f'branch-text_artifact-{i}', 'phase_id': f'phase-{i + 1}'}
        artifacts.append({'type': 'text', 'path': str(path), 'artifact_ref': stale_ref,
                          'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        slots.append({'slot_id': f'output-{i}', **binding, 'type': 'text',
                      'status': 'fulfilled', 'artifact_ref': stale_ref})
        results.append({**binding, 'capability': 'chat', 'output_type': 'text',
                        'status': 'fulfilled', 'saved_text_path': str(path),
                        'text_artifact_source_name': path.stem,
                        'fill_model': 'test-producer', 'fill_instance_id': 'test-producer-1'})
    payload = {'id': 'registry-successor', 'status': 'completed', 'lifecycle_state': 'completed',
               'artifacts': artifacts, 'output_slots': slots,
               'saved_text_path': str(root / 'status.json'),
               'saved_text_artifacts': [{'path': str(root / 'status.json'),
                                        'artifact_request': {'source_name': 'status', 'extension': 'json'}}],
               'late_fill': {'status': 'completed', 'fill_results': results},
               'frame_relation': {'kind': 'late_fill_successor',
                   'parent_response_id': 'registry-successor',
                   'parent_frame_id': 'registry-successor:frame-1', 'parent_frame_sequence': 1}}
    attached = attach_response_frame(payload, request_payload={'prompt': 'Create note.txt and status.json.'})
    return payload, attached, hoist_response_output_surfaces(attached)


def registry_table(payload, ledger_path):
    canonical = payload['response_frame']['artifacts']['output']
    records = registry.persist_output_artifact_registry_records(payload, ledger_path=ledger_path)
    table = []
    for artifact in canonical:
        record = registry.find_artifact_registry_record(artifact['path'], ledger_path=ledger_path)
        table.append({'name': Path(artifact['path']).name,
                      'canonical_id': artifact['artifact_id'], 'canonical_ref': artifact['artifact_ref'],
                      'registry_id': (record or {}).get('artifact_id'),
                      'registry_ref': (record or {}).get('artifact_ref'),
                      'path': artifact['path'],
                      'sha256': hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest(),
                      'present': record is not None})
    return {'table': table, 'records': records,
            'registry_record_count': len(Path(ledger_path).read_text().splitlines()),
            'frame_id': payload['response_frame'].get('frame_id'),
            'frame_relation': payload['response_frame'].get('frame_relation')}


def test_registry_keeps_both_accepted_files_after_frame_reconciliation(tmp_path):
    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    result = registry_table(payload, tmp_path / 'registry.jsonl')
    assert result['registry_record_count'] == 2
    assert all(row['present'] and row['canonical_ref'] == row['registry_ref']
               and row['canonical_id'] == row['registry_id'] for row in result['table'])


def test_final_frame_wins_over_stale_projection_and_refresh_is_idempotent(tmp_path):
    _, attached, payload = registry_handoff_fixture(tmp_path / 'files')
    ledger = tmp_path / 'registry.jsonl'
    canonical = payload['response_frame']['artifacts']['output']
    for artifact in canonical:
        predecessor = {**artifact, 'artifact_id': 'old-' + artifact['artifact_id'],
                       'artifact_ref': 'artifact:old-' + artifact['artifact_id'],
                       'ref': 'artifact:old-' + artifact['artifact_id']}
        registry.persist_artifact_registry_record(
            registry.build_artifact_registry_record(artifact=predecessor, roles=['output']),
            ledger_path=ledger)
    # Even a populated, stale top-level projection cannot supersede the frame.
    payload['artifacts'] = attached['artifacts']
    original = copy.deepcopy(payload)
    first = registry.persist_output_artifact_registry_records(payload, ledger_path=ledger)
    saved = ledger.read_bytes()
    second = registry.persist_output_artifact_registry_records(payload, ledger_path=ledger)
    assert payload == original
    assert ledger.read_bytes() == saved and second == first
    assert len(saved.splitlines()) == 2
    for artifact in canonical:
        old_ref = 'artifact:old-' + artifact['artifact_id']
        record = registry.find_artifact_registry_record_by_artifact_ref(old_ref, ledger_path=ledger)
        assert record['artifact_ref'] == artifact['artifact_ref']
        assert record['artifact']['artifact_ref'] == artifact['artifact_ref']
        assert record['artifact_id'] == artifact['artifact_id']
        assert artifact['artifact_ref'] in record['artifact_alias_refs']
        assert record['linked_response_ids'] == [payload['id']]


def test_unframed_shortcut_does_not_replace_named_canonical_identity(tmp_path):
    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    payload['artifacts'] = payload.pop('response_frame')['artifacts']['output']
    canonical = {a['artifact_ref'] for a in payload['artifacts']}
    records = registry.persist_output_artifact_registry_records(payload, ledger_path=tmp_path / 'registry.jsonl')
    assert {r['artifact_ref'] for r in records} == canonical
    assert len(records) == 2


def test_equal_bytes_and_same_basename_do_not_collapse_distinct_artifacts(tmp_path):
    artifacts = []
    for folder in ['left', 'right']:
        path = tmp_path / folder / 'note.txt'
        path.parent.mkdir()
        path.write_text('identical bytes\n')
        artifacts.append({'type': 'text', 'path': str(path), 'artifact_id': folder,
                          'artifact_ref': 'artifact:' + folder, 'name': 'note'})
    payload = {'id': 'two-identities', 'artifacts': artifacts,
               'saved_text_path': artifacts[-1]['path']}
    ledger = tmp_path / 'registry.jsonl'
    records = registry.persist_output_artifact_registry_records(payload, ledger_path=ledger)
    assert len(records) == len(ledger.read_text().splitlines()) == 2
    assert len({r['artifact']['file_sha256'] for r in records}) == 1
    assert {r['artifact_ref'] for r in records} == {'artifact:left', 'artifact:right'}
    for artifact in artifacts:
        record = registry.find_artifact_registry_record_by_artifact_ref(artifact['artifact_ref'], ledger_path=ledger)
        assert record['artifact']['path'] == artifact['path']
        assert record['artifact_alias_refs'] == [artifact['artifact_ref']]


def test_empty_frame_output_does_not_resurrect_saved_shortcut(tmp_path):
    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    payload['response_frame']['artifacts']['output'] = []
    assert registry.build_output_artifact_registry_records(payload) == []


def test_other_response_frame_cannot_supply_registry_identity(tmp_path):
    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    payload['response_frame']['response_id'] = 'another-response'
    with pytest.raises(ValueError, match='current response frame'):
        registry.persist_output_artifact_registry_records(payload, ledger_path=tmp_path / 'registry.jsonl')
    assert not (tmp_path / 'registry.jsonl').exists()


@pytest.mark.parametrize('state,expected', [('missing', 'skipped_missing_file'), ('invalid_json', 'refreshed')])
def test_missing_or_corrupt_saved_file_keeps_existing_refresh_semantics(tmp_path, state, expected):
    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    artifact = next(a for a in payload['response_frame']['artifacts']['output'] if a['path'].endswith('.json'))
    if state == 'missing':
        Path(artifact['path']).unlink()
    else:
        Path(artifact['path']).write_text('not JSON')
    record = next(r for r in registry.build_output_artifact_registry_records(payload)
                  if r['artifact_ref'] == artifact['artifact_ref'])
    assert record['metadata']['final_text_artifact_refresh_status'] == expected
    if state == 'invalid_json':
        assert record['metadata']['syntax_sanity_status'] != 'ok'
    assert record['artifact_ref'] == artifact['artifact_ref']


def test_registry_preserves_artifact_and_late_fill_producer_bindings(tmp_path):
    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    canonical = {a['artifact_ref']: a for a in payload['response_frame']['artifacts']['output']}
    records = registry.build_output_artifact_registry_records(payload)
    for record in records:
        artifact = canonical[record['artifact_ref']]
        for key in ['branch_id', 'phase_id', 'source_response_id']:
            if key in artifact:
                assert record['artifact'][key] == artifact[key]
        source = record['provenance']['source']
        result = next(r for r in payload['late_fill']['fill_results'] if r['saved_text_path'] == artifact['path'])
        assert source['response_id'] == payload['id']
        assert source['branch_id'] == result['branch_id']
        assert source['phase_id'] == result['phase_id']
        assert source['model'] == result['fill_model']
        assert source['instance_id'] == result['fill_instance_id']
        assert record['artifact']['file_sha256'] == hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest()


def test_successor_ledger_recovery_and_registry_agree(tmp_path):
    from ollmo_services.response_frames import persist_response_frame, load_latest_response_state

    _, _, payload = registry_handoff_fixture(tmp_path / 'files')
    frame_root, ledger = tmp_path / 'frames', tmp_path / 'registry.jsonl'
    parent = attach_response_frame({'id': payload['id'], 'status': 'in_progress'})
    assert registry.persist_output_artifact_registry_records(parent, ledger_path=ledger) == []
    persist_response_frame(parent['response_frame'], frames_dir=frame_root)
    registry.persist_output_artifact_registry_records(payload, ledger_path=ledger)
    persist_response_frame(payload['response_frame'], frames_dir=frame_root)
    recovered = load_latest_response_state(payload['id'], frames_dir=frame_root)
    assert recovered['ok'] and recovered['frame_count'] == 2
    final = recovered['response_payload']
    assert final['response_frame']['frame_relation']['parent_frame_id'] == payload['id'] + ':frame-1'
    saved = ledger.read_bytes()
    registry.persist_output_artifact_registry_records(final, ledger_path=ledger)
    assert ledger.read_bytes() == saved
    assert len(final['artifacts']) == len(saved.splitlines()) == 2
    for artifact in final['artifacts']:
        record = registry.find_artifact_registry_record_by_artifact_ref(artifact['artifact_ref'], ledger_path=ledger)
        assert record['artifact_id'] == artifact['artifact_id']
        assert record['artifact']['path'] == artifact['path']
        assert record['artifact']['file_sha256'] == artifact['file_sha256']
