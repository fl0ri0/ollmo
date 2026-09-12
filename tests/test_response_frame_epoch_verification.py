import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ollmo_services import response_frames as frames
from ollmo_services import graph_rebase_readiness_registry as registry
from ollmo_services import events
from ollmo_services.graph_rebase_readiness_registry import (
    GraphRebaseReadinessRegistryError,
    append_graph_rebase_readiness_observation,
)


def _frame(response_id='epoch-response'):
    return {
        'kind': 'ollmo.response_frame',
        'frame_version': 9,
        'response_id': response_id,
        'status': 'completed',
        'current_state': {
            'id': response_id,
            'status': 'completed',
            'lifecycle_state': 'completed',
        },
        'runtime': {
            'developer_diagnostics': {
                'response_time_graph_rebase_candidate': {'candidate': True},
            },
        },
    }


@pytest.mark.parametrize('binding', ['exact', 'entry_alias', 'missing_name', 'relocated'])
def test_epoch_digest_reuse_requires_every_entry_binding_to_be_unchanged(tmp_path, binding):
    root = tmp_path / 'frames'
    frames.persist_response_frame(_frame(), frames_dir=root)
    frames.persist_response_frame(_frame('second-response'), frames_dir=root)
    index_path = root / 'current_index.json'
    index = json.loads(index_path.read_bytes())
    # Unusual nested JSON remains subject to the existing canonicalizer, not
    # an assumption that JSON-decoded input is already canonical/idempotent.
    index['responses']['epoch-response']['extra'] = {'nested': {'empty': {}}}
    if binding == 'entry_alias':
        index['responses']['second-response']['ledger_path'] = str(root / '..' / 'frames' / 'responses.jsonl')
    elif binding == 'missing_name':
        index['responses']['second-response'].pop('ledger_name')
    index['response_map_digest'] = frames._response_map_digest(index['responses'])
    index_path.write_text(json.dumps(index))
    if binding == 'relocated':
        relocated = tmp_path / 'archive'
        root.rename(relocated)
        root = relocated
        index_path = root / 'current_index.json'
    before = index_path.read_bytes()
    with patch.object(frames, '_response_map_digest', wraps=frames._response_map_digest) as digest:
        verified = frames.verify_response_frame_epoch(frames_dir=root, allow_relocated=True)
    assert verified['ok'], verified
    assert digest.call_count == (1 if binding == 'exact' else 2)
    assert verified['response_map_digest'] == index['response_map_digest']
    rebound = verified['index_state']
    assert rebound['response_map_digest'] == frames._response_map_digest(rebound['responses'])
    assert index_path.read_bytes() == before
    assert all(entry['ledger_path'] == str(root / 'responses.jsonl') for entry in rebound['responses'].values())


@pytest.mark.parametrize('movement', ['ledger_append', 'ledger_replace', 'index_replace', 'index_corrupt'])
def test_epoch_digest_reuse_keeps_final_physical_evidence_checks(tmp_path, movement):
    frames.persist_response_frame(_frame(), frames_dir=tmp_path)
    original = frames._stable_response_frame_index_snapshot
    reads = 0

    def move_before_final_read(path):
        nonlocal reads
        reads += 1
        if reads == 2:
            target = tmp_path / ('responses.jsonl' if movement.startswith('ledger') else 'current_index.json')
            raw = target.read_bytes()
            if movement == 'ledger_append':
                with target.open('ab') as handle:
                    handle.write(raw)
            else:
                replacement = target.with_suffix('.replacement')
                replacement.write_bytes(b'{' if movement == 'index_corrupt' else raw)
                replacement.replace(target)
        return original(path)

    with patch.object(frames, '_stable_response_frame_index_snapshot', side_effect=move_before_final_read):
        verified = frames.verify_response_frame_epoch(frames_dir=tmp_path)
    assert reads == 2
    assert not verified['ok']
    assert verified['error']['code'] == (
        'response_frame_ledger_moved' if movement.startswith('ledger') else 'response_frame_index_moved'
    )


def test_returned_map_is_reverified_at_selection_and_hydration_boundaries(tmp_path):
    frames.persist_response_frame(_frame(), frames_dir=tmp_path)
    verified = frames.verify_response_frame_epoch(frames_dir=tmp_path)
    assert verified['ok']
    index = verified['index_state']
    index['responses']['epoch-response']['latest_frame_sequence'] += 1
    selection = frames.select_graph_rebase_observation_response_ids(frames_dir=tmp_path, index_state=index)
    observed = frames.load_latest_response_observation_state('epoch-response', frames_dir=tmp_path, index_state=index)
    assert selection['scan_error_count'] == 1
    assert selection['selected_response_ids'] == []
    assert not observed['ok']
    assert observed['error']['code'] == 'response_frame_index_unverified'


@pytest.mark.parametrize('movement', ['successor', 'ledger_replace', 'index_replace'])
def test_old_verified_epoch_cannot_publish_readiness_after_external_change(tmp_path, movement):
    root = tmp_path / 'frames'
    frames.persist_response_frame(_frame(), frames_dir=root)
    verified = frames.verify_response_frame_epoch(frames_dir=root)
    observed = frames.load_latest_response_observation_state('epoch-response', frames_dir=root, index_state=verified['index_state'])
    assert verified['ok'] and observed['ok']
    if movement == 'successor':
        frames.persist_response_frame(_frame(), frames_dir=root)
    else:
        target = root / ('responses.jsonl' if movement == 'ledger_replace' else 'current_index.json')
        replacement = target.with_suffix('.replacement')
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
    registry = tmp_path / 'readiness.jsonl'
    with pytest.raises(GraphRebaseReadinessRegistryError) as raised:
        append_graph_rebase_readiness_observation(
            observed['response_payload'],
            source_frame=verified['source_frame_sha256_by_response']['epoch-response'],
            verified_epoch=verified,
            frames_dir=root,
            registry_path=registry,
        )
    assert raised.value.code == 'readiness_epoch_moved'
    assert not registry.exists()


@pytest.fixture
def retention(tmp_path):
    root = tmp_path / 'frames'
    frames.persist_response_frame(_frame(), frames_dir=root)
    epoch = frames.verify_response_frame_epoch(frames_dir=root)
    observed = frames.load_latest_response_observation_state(
        'epoch-response', frames_dir=root, index_state=epoch['index_state'],
    )
    assert epoch['ok'] and observed['ok']
    return {
        'payload_or_projection': observed['response_payload'],
        'source_frame': epoch['source_frame_sha256_by_response']['epoch-response'],
        'verified_epoch': epoch, 'frames_dir': root,
        'registry_path': tmp_path / 'readiness.jsonl',
    }


def rejection(retention):
    with pytest.raises(GraphRebaseReadinessRegistryError) as caught:
        append_graph_rebase_readiness_observation(**retention)
    error = caught.value
    assert error.code == 'readiness_epoch_moved'
    assert error.status_code == 409
    assert str(error) == 'Preverified response-frame epoch is no longer current.'
    assert not retention['registry_path'].exists()
    return error


@pytest.mark.parametrize('file', ['ledger', 'index'])
@pytest.mark.parametrize('field,reason', [
    ('device', 'device'), ('inode', 'inode'), ('size_bytes', 'size'),
    ('mtime_ns', 'mtime'), ('ctime_ns', 'ctime'),
])
def test_retention_reports_each_exact_stat_binding(retention, file, field, reason):
    epoch = retention['verified_epoch']
    captured = {name: dict(epoch[f'{name}_file_state']) for name in ('ledger', 'index')}
    captured[file][field] += 1
    calls = []

    def stat(path):
        name = 'ledger' if path.name == 'responses.jsonl' else 'index'
        calls.append(name)
        return captured[name]

    # Rejection diagnostics consume stat captures, never contents or new hashes.
    with patch.object(registry, '_file_state', side_effect=stat), \
         patch.object(Path, 'open', side_effect=AssertionError('unexpected read')), \
         patch.object(registry, '_sha256', side_effect=AssertionError('unexpected hash')):
        error = rejection(retention)
    detail = error.as_dict()['epoch_retention']
    assert calls == ['ledger', 'index']
    assert detail['reasons'] == [f'{file}_{reason}_changed']
    for name in ('ledger', 'index'):
        assert detail['files'][name]['expected'] == epoch[f'{name}_file_state']
        assert detail['files'][name]['observed'] == captured[name]
        assert detail['files'][name]['path'] == epoch[f'{name}_path']
        assert detail['files'][name]['bytes_unchanged'] == 'unknown'
    assert detail['verified_index_entry']['latest_frame_id'] == 'epoch-response:frame-1'
    assert detail['frame_id'] == 'epoch-response:frame-1'
    assert detail['frame_sequence'] == 1
    assert detail['monotonic_ns'] > 0
    assert detail['verified_epoch']['index_sha256'] == epoch['index_sha256']
    assert detail['mutation_actor'] == detail['current_index_entry_identity'] == 'unknown'
    assert len(json.dumps(detail)) < 6000


@pytest.mark.parametrize('file', ['ledger', 'index'])
@pytest.mark.parametrize('movement,reason', [
    ('append', 'size'), ('timestamps', 'mtime'), ('replace', 'inode'),
])
def test_retention_diagnostic_real_file_movements(retention, file, movement, reason):
    path = Path(retention['verified_epoch'][f'{file}_path'])
    before = path.stat()
    if movement == 'append':
        with path.open('ab') as handle:
            handle.write(b'\n')
    elif movement == 'timestamps':
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    else:
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    detail = rejection(retention).details['epoch_retention']
    assert f'{file}_{reason}_changed' in detail['reasons']
    assert detail['files'][file]['observed'] == registry._file_state(path)
    if movement == 'timestamps':
        assert detail['files'][file]['expected']['inode'] == detail['files'][file]['observed']['inode']
        assert detail['files'][file]['expected']['size_bytes'] == detail['files'][file]['observed']['size_bytes']


def test_retention_reports_multiple_changes_and_does_not_infer_writer(retention):
    captured = {}
    for name in ('ledger', 'index'):
        captured[name] = {k: v + 1 for k, v in retention['verified_epoch'][f'{name}_file_state'].items()}
    with patch.object(registry, '_file_state', side_effect=lambda p: captured['ledger' if p.name == 'responses.jsonl' else 'index']), \
         patch.object(registry, 'causal_event', return_value={'process_id': 42, 'process_boot_id': '42:observer'}) as sink:
        detail = rejection(retention).details['epoch_retention']
    assert len(detail['reasons']) == 10
    assert detail['mutation_actor'] == 'unknown'
    assert detail['observer_process_boot_id'] == '42:observer'
    assert sink.call_count == 1


def test_unchanged_retention_has_no_rejection_diagnostic(retention):
    with patch.object(registry, '_readiness_epoch_mismatch_details') as build, \
         patch.object(registry, 'causal_event') as sink:
        result = append_graph_rebase_readiness_observation(**retention)
    assert result['ok']
    build.assert_not_called()
    sink.assert_not_called()
    assert 'epoch_retention' not in retention['registry_path'].read_text()


@pytest.mark.parametrize('failure', ['causal_event', '_emit_readiness_epoch_mismatch', '_readiness_epoch_mismatch_details'])
def test_retention_diagnostic_failure_preserves_original_rejection(retention, failure):
    with (retention['frames_dir'] / 'responses.jsonl').open('ab') as handle:
        handle.write(b'\n')
    with patch.object(registry, failure, side_effect=RuntimeError('broken diagnostic sink')):
        error = rejection(retention)
    if failure != '_readiness_epoch_mismatch_details':
        assert 'ledger_size_changed' in error.details['epoch_retention']['reasons']


def test_retention_diagnostic_omits_payloads_and_oversized_metadata(retention):
    retention['payload_or_projection']['request'] = {'prompt': 'PRIVATE_PROMPT'}
    epoch = retention['verified_epoch']
    epoch['epoch_anchor']['response_id'] = 'PRIVATE_OVERSIZED_' + 'x' * 10000
    epoch['index_state']['responses']['epoch-response']['payload'] = 'PRIVATE_INDEX_BODY'
    with (retention['frames_dir'] / 'responses.jsonl').open('ab') as handle:
        handle.write(b'\n')
    detail = rejection(retention).details['epoch_retention']
    encoded = json.dumps(detail)
    assert 'PRIVATE_' not in encoded
    assert 'verified_epoch_anchor.response_id' in detail['omitted_fields']
    assert len(encoded) < 6000


@pytest.mark.parametrize('sink_fails', [False, True])
def test_retention_uses_existing_causal_sink_without_changing_rejection(retention, sink_fails):
    with (retention['frames_dir'] / 'responses.jsonl').open('ab') as handle:
        handle.write(b'\n')
    records = []

    def sink(**entry):
        if sink_fails:
            raise OSError('diagnostic destination unavailable')
        records.append(entry['causal_event'])

    with events.causal_scope(sink, response_id='epoch-response'):
        detail = rejection(retention).details['epoch_retention']
    assert detail['mutation_actor'] == 'unknown'
    if not sink_fails:
        record = next(r for r in records if r['owner'] == 'readiness.epoch_retention')
        assert record['epoch_retention']['files'] == detail['files']
        assert record['epoch_retention']['reasons'] == detail['reasons']
        assert record['process_boot_id'] == detail['observer_process_boot_id']
