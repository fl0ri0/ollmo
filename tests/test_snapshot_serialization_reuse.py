"""Exact private preparation reuse must never reuse verification authority."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from ollmo_services import response_frames as frames


FRAME = {'kind': 'ollmo.response_frame', 'response_id': 'resp_serialization',
         'frame_id': 'resp_serialization:frame-1', 'frame_sequence': 1}


def payload():
    return {'nodes': [{'id': str(i), 'decision_contract': {
        'evidence': ['exact branch evidence ' * 160] * 12,
        'created_at': '2026-09-09T12:00:00Z',
    }} for i in range(12)]}


def writer(root, frame=None):
    frame = copy.deepcopy(FRAME) if frame is None else frame
    reuse = frames._SnapshotSerializationReuse(frame, root)

    def write(value, path='runtime.request_phase_graph'):
        return frames._write_snapshot_ref(value, frame=frame, frames_dir=root,
                                         json_path=path, _sidecar_child_ref=True, _serialization_reuse=reuse)
    return write, reuse


def test_stable_repeats_every_split_and_integrity_check(tmp_path):
    value = payload()
    write, reuse = writer(tmp_path)
    counts = {}

    def note(**values):
        for k, v in values.items():
            if isinstance(v, int):
                counts[k] = counts.get(k, 0) + v

    with patch.object(frames, 'state_flow_note', note), \
         patch.object(frames, '_normalize_snapshot_content', wraps=frames._normalize_snapshot_content) as normalize, \
         patch.object(frames, '_ensure_content_addressed_snapshot', wraps=frames._ensure_content_addressed_snapshot) as ensure:
        first = write(value)
        normalize_count = normalize.call_count
        ensure_count = ensure.call_count
        second = write(value)
        assert normalize.call_count == normalize_count
        assert ensure.call_count == ensure_count * 2
    assert first == second
    assert counts['snapshot_reuse_accepted'] > 12
    assert counts['snapshot_reuse_binding_checks'] == counts['snapshot_root_calls'] + counts['snapshot_child_calls']
    assert counts['sidecar_verification_reads'] >= counts['snapshot_reuse_binding_checks']
    assert frames._read_snapshot_ref_payload(second, frames_dir=tmp_path)['nodes'][0]['id'] == '0'
    assert reuse._retained_bytes <= frames._SNAPSHOT_SERIALIZATION_REUSE_MAX_BYTES


@pytest.mark.parametrize('change', ['input', 'timestamp', 'policy', 'frame_id', 'frame_sequence',
                                  'response_id', 'frame_relation', 'ledger', 'index', 'same_byte_epoch'])
def test_changed_input_or_authority_uses_full_path(tmp_path, change):
    frame = copy.deepcopy(FRAME)
    (tmp_path / 'responses.jsonl').write_bytes(b'ledger epoch\n')
    (tmp_path / 'current_index.json').write_bytes(b'index mapping\n')
    write, _reuse = writer(tmp_path, frame)
    value = {'evidence': 'bound truth', 'created_at': 'first'}
    first = write(value)
    path = 'runtime.request_phase_graph'
    if change == 'input':
        value['evidence'] = 'new truth'
    elif change == 'timestamp':
        value['created_at'] = 'second'  # same CAS bytes, different prepared state
    elif change == 'policy':
        path = 'request.input'
    elif change in ('frame_id', 'frame_sequence', 'response_id', 'frame_relation'):
        frame[change] = 2 if change == 'frame_sequence' else 'different'
    elif change in ('ledger', 'index'):
        (tmp_path / ('responses.jsonl' if change == 'ledger' else 'current_index.json')).write_bytes(b'changed')
    else:
        p = tmp_path / 'responses.jsonl'
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(p.read_bytes())
        replacement.replace(p)
    with patch.object(frames, '_normalize_snapshot_content', wraps=frames._normalize_snapshot_content) as full:
        second = write(value, path)
        assert full.call_count == 1
    if change == 'input':
        assert first['sha256'] != second['sha256']
        assert frames._read_snapshot_ref_payload(second, frames_dir=tmp_path)['evidence'] == 'new truth'
    if change == 'timestamp':
        assert first['sha256'] == second['sha256']


@pytest.mark.parametrize('change', ['missing', 'corrupt'])
def test_reused_preparation_repairs_fresh_sidecar_failure(tmp_path, change):
    write, _ = writer(tmp_path)
    value = {'evidence': 'bound truth'}
    ref = write(value)
    target = tmp_path / ref['path']
    if change == 'missing':
        target.unlink()
    else:
        target.write_bytes(b'corrupt')
    with patch.object(frames, '_normalize_snapshot_content', side_effect=AssertionError('must reuse preparation')), \
         patch.object(frames, '_snapshot_file_matches', wraps=frames._snapshot_file_matches) as checks:
        assert write(value) == ref
        assert checks.call_count == 2
    assert frames._read_snapshot_ref_payload(ref, frames_dir=tmp_path) == value


def test_reused_preparation_preserves_postwrite_failure(tmp_path):
    write, _ = writer(tmp_path)
    value = {'evidence': 'bound truth'}
    ref = write(value)
    (tmp_path / ref['path']).write_bytes(b'corrupt')
    with patch.object(frames, '_atomic_replace_file_bytes', return_value=None):
        with pytest.raises(OSError, match='post-write verification'):
            write(value)


def test_no_mutable_aliases_or_cross_owner_reuse(tmp_path):
    frame = copy.deepcopy(FRAME)
    write, reuse = writer(tmp_path, frame)
    value = {'evidence': ['original'], 'created_at': 'timestamp'}
    ref = write(value)
    original = copy.deepcopy(ref)
    ref['content_normalization']['volatile_timestamp_keys'].append('forged')
    value['evidence'].append('changed')
    assert write({'evidence': ['original'], 'created_at': 'timestamp'}) == original
    with patch.object(frames, '_normalize_snapshot_content', wraps=frames._normalize_snapshot_content) as full:
        frames._write_snapshot_ref({'evidence': ['original'], 'created_at': 'timestamp'},
            frame=copy.deepcopy(frame), frames_dir=tmp_path,
            json_path='runtime.request_phase_graph', _sidecar_child_ref=True, _serialization_reuse=reuse)
        assert full.call_count == 1


def test_unknown_binding_and_budget_use_original_path(tmp_path):
    value = {'evidence': 'truth'}
    write, reuse = writer(tmp_path)
    with patch.object(frames, '_SNAPSHOT_SERIALIZATION_REUSE_MAX_BYTES', 0), \
         patch.object(frames, '_normalize_snapshot_content', wraps=frames._normalize_snapshot_content) as full:
        first = write(value)
        assert write(value) == first
        assert full.call_count == 2
        assert not reuse._entries
    with patch.object(reuse, '_current_binding', return_value=None), \
         patch.object(frames, '_normalize_snapshot_content', wraps=frames._normalize_snapshot_content) as full:
        assert write(value) == first
        assert full.call_count == 1


def test_concurrent_compactions_preserve_exact_baseline_bytes(tmp_path):
    frame = dict(FRAME, runtime={'request_phase_graph': payload()},
                 current_state={'runtime': {'request_phase_graph': payload()}})
    original = copy.deepcopy(frame)
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    # Disable only the new preparation, leaving the exact former path active.
    with patch.object(frames._SnapshotSerializationReuse, 'serialize',
                      lambda self, value, *, json_path, **kw: frames._serialize_snapshot_content(value, json_path=json_path)):
        expected = frames.compact_response_frame_for_ledger(frame, frames_dir=baseline)
    files = {str(p.relative_to(baseline)): p.read_bytes() for p in baseline.rglob('*.json')}

    def run(i):
        root = tmp_path / str(i)
        root.mkdir()
        result = frames.compact_response_frame_for_ledger(frame, frames_dir=root)
        assert result == expected
        assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*.json')} == files
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(list(pool.map(run, range(8)))) == 8
    assert frame == original


def test_media_file_is_rechecked_before_preparation_match(tmp_path):
    import base64
    media = tmp_path / 'image.png'
    raw = b'bound saved image' * 200
    media.write_bytes(raw)
    value = {'saved_image_path': str(media), 'artifact_ref': 'artifact:one',
             'image': base64.b64encode(raw).decode()}
    write, _ = writer(tmp_path)
    first = write(value)
    media.write_bytes(b'changed saved bytes')
    with patch.object(frames, '_normalize_snapshot_content', wraps=frames._normalize_snapshot_content) as full:
        second = write(value)
        assert full.call_count == 1
    assert first['sha256'] != second['sha256']
    assert frames._read_snapshot_ref_payload(second, frames_dir=tmp_path)['image']['kind'] == 'ollmo.snapshot_stripped_raw_media_payload'


def test_concurrent_distinct_responses_and_frames_do_not_share_preparation(tmp_path):
    def run(i):
        root = tmp_path / str(i)
        root.mkdir()
        frame = dict(FRAME, response_id=f'resp_{i}', frame_id=f'resp_{i}:frame-{i + 1}', frame_sequence=i + 1)
        value = {'evidence': [f'private {i}'], 'created_at': str(i)}
        write, reuse = writer(root, frame)
        first = write(value)
        assert write(value) == first
        assert first['source_response_id'] == frame['response_id']
        assert first['source_frame_id'] == f'resp_{i}_frame-{i + 1}'
        assert frames._read_snapshot_ref_payload(first, frames_dir=root)['evidence'] == [f'private {i}']
        assert len(reuse._entries) == 1
        return first['sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(set(pool.map(run, range(8)))) == 8
