import json
from unittest.mock import patch

import pytest

from ollmo_services import response_frames as frames
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
