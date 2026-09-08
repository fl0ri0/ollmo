"""Preparation reuse must preserve snapshot bytes and fresh file checks."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ollmo_services import response_frames as frames


FRAME = {
    'kind': 'ollmo.response_frame',
    'response_id': 'resp_prepared_snapshot',
    'frame_id': 'resp_prepared_snapshot:frame-1',
    'frame_sequence': 1,
}


def write(value, root):
    return frames._write_snapshot_ref(
        value, frame=FRAME, frames_dir=root, json_path='runtime',
    )


def body(root, ref):
    return frames._read_snapshot_ref_payload(ref, frames_dir=root)


def nested_payload():
    return {
        'created_at': '2026-09-06T00:00:00Z',
        'request_phase_graph': {
            'nodes': [
                {
                    'id': f'phase-{index}',
                    'summary': 'retained branch-local evidence ' * 200,
                    'decision_contract': {
                        'evidence': 'exact dependency identity ' * 2000,
                        'created_at': '2026-09-06T00:00:01Z',
                    },
                }
                for index in range(8)
            ],
        },
    }


def test_recursive_children_reuse_one_media_preparation(tmp_path):
    value = nested_payload()
    original = copy.deepcopy(value)
    normalize = frames._normalize_snapshot_media_payloads
    depth = 0
    passes = []

    def counted(value, **kwargs):
        nonlocal depth
        if depth == 0:
            passes.append(value)
        depth += 1
        try:
            return normalize(value, **kwargs)
        finally:
            depth -= 1

    with patch.object(frames, '_normalize_snapshot_media_payloads', counted):
        ref = write(value, tmp_path)
    assert len(list(tmp_path.rglob('*.json'))) >= 4
    assert len(passes) == 1
    assert value == original
    expanded = body(tmp_path, ref)
    assert expanded['request_phase_graph']['nodes'][6]['id'] == 'phase-6'
    assert expanded['request_phase_graph']['nodes'][6]['decision_contract']['evidence'] == (
        value['request_phase_graph']['nodes'][6]['decision_contract']['evidence']
    )
    assert expanded['created_at'] == value['created_at']
    assert 'created_at' not in expanded['request_phase_graph']['nodes'][6]['decision_contract']


def test_preparation_matches_legacy_bytes_with_media_and_distinct_files(tmp_path):
    artifact_a = tmp_path / 'a.png'
    artifact_b = tmp_path / 'b.png'
    raw = b'opaque image bytes' * 200
    artifact_a.write_bytes(raw)
    artifact_b.write_bytes(raw)
    encoded = base64.b64encode(raw).decode()
    value = nested_payload()
    value['saved_image_path'] = str(artifact_a)
    value['artifact_ref'] = 'artifact:a'
    nodes = value['request_phase_graph']['nodes']
    nodes[0]['image'] = encoded
    nodes[1].update(saved_image_path=str(artifact_b), artifact_ref='artifact:b', image=encoded)
    value[' odd key '] = {'empty': {'empty': ''}, 'path': Path('kept'), 'sequence': (0, False, 'value')}
    value['image_data_url'] = 'excluded compatibility body'
    original = copy.deepcopy(value)
    current = tmp_path / 'current'
    legacy = tmp_path / 'legacy'
    current_ref = write(value, current)
    writer = frames._write_snapshot_ref

    # Reproduce the former child preparation at its exact call boundary.
    def legacy_writer(value, **kwargs):
        kwargs.pop('_media_normalized', None)
        return writer(value, **kwargs)

    with patch.object(frames, '_write_snapshot_ref', legacy_writer):
        legacy_ref = write(value, legacy)
    assert current_ref == legacy_ref
    assert {p.relative_to(current): p.read_bytes() for p in current.rglob('*.json')} == {
        p.relative_to(legacy): p.read_bytes() for p in legacy.rglob('*.json')
    }
    assert value == original
    restored = body(current, current_ref)
    a, b = restored['request_phase_graph']['nodes'][:2]
    assert a['image']['sha256'] == b['image']['sha256'] == hashlib.sha256(raw).hexdigest()
    assert a['image']['artifact_ref'] == 'artifact:a'
    assert b['image']['artifact_ref'] == 'artifact:b'
    assert a['image']['source_path'] != b['image']['source_path']
    assert restored['odd key'] == {'path': 'kept', 'sequence': [0, False, 'value']}
    assert 'image_data_url' not in restored


@pytest.mark.parametrize('mutation', ['missing', 'corrupt'])
def test_duplicate_child_still_verifies_changed_sidecar(tmp_path, mutation):
    contract = {'evidence': 'same retained content ' * 2000}
    value = {'nodes': [{'id': 'a', 'decision_contract': contract}, {'id': 'b', 'decision_contract': contract}]}
    ensure = frames._ensure_content_addressed_snapshot
    first_target = None
    duplicate_checks = 0

    def mutate_after_first(target, **kwargs):
        nonlocal first_target, duplicate_checks
        duplicate = target == first_target
        if duplicate:
            duplicate_checks += 1
        result = ensure(target, **kwargs)
        if first_target is None:
            first_target = target
            if mutation == 'missing':
                target.unlink()
            else:
                target.write_bytes(b'corrupt')
        return result

    with patch.object(frames, '_ensure_content_addressed_snapshot', mutate_after_first):
        ref = write(value, tmp_path)
    assert duplicate_checks == 1
    assert body(tmp_path, ref)['nodes'][1]['decision_contract'] == contract
    assert first_target is not None
    assert json.loads(first_target.read_bytes()) == contract


def test_separate_snapshot_calls_observe_mutated_input_and_media_file(tmp_path):
    media = tmp_path / 'image.png'
    raw = b'first image bytes' * 200
    media.write_bytes(raw)
    value = nested_payload()
    value.update(saved_image_path=str(media), artifact_ref='artifact:a')
    value['request_phase_graph']['nodes'][0]['image'] = base64.b64encode(raw).decode()
    first = write(value, tmp_path / 'frames')
    media.write_bytes(b'changed image bytes')
    value['request_phase_graph']['nodes'][1]['id'] = 'changed-phase'
    second = write(value, tmp_path / 'frames')
    assert first['sha256'] != second['sha256']
    first_node = body(tmp_path / 'frames', first)['request_phase_graph']['nodes'][0]
    second_nodes = body(tmp_path / 'frames', second)['request_phase_graph']['nodes']
    assert first_node['image']['kind'] == 'ollmo.snapshot_externalized_media_payload'
    assert second_nodes[0]['image']['kind'] == 'ollmo.snapshot_stripped_raw_media_payload'
    assert second_nodes[1]['id'] == 'changed-phase'


def test_child_verification_failure_prevents_parent_publication(tmp_path):
    def broken_write(target, payload):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'corrupt')

    with patch.object(frames, '_atomic_replace_file_bytes', broken_write):
        with pytest.raises(OSError, match='post-write verification'):
            write(nested_payload(), tmp_path)
    assert len(list(tmp_path.rglob('*.json'))) == 1
