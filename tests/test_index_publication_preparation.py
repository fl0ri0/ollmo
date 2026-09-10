"""Publication-local preparation must preserve bytes and repeat authority checks."""
import json
from unittest.mock import patch

import pytest

from ollmo_services import response_frames as frames


def _frame(response_id):
    return {
        'kind': 'ollmo.response_frame',
        'frame_version': 9,
        'response_id': response_id,
        'status': 'completed',
        'current_state': {'id': response_id, 'lifecycle_state': 'completed'},
    }


@pytest.mark.parametrize('edge', [
    {'nested': {'empty': {}}},
    {'items': [{}, {'empty': {}}, [], None, '', 0, False]},
    {'unicode': 'ä 雪 🙂', 'escaped': '\\"\n\t', 'number': -0.0},
    {' a ': 1, 'a': 2, 'image_data_url': 'omitted'},
])
def test_prepared_publication_matches_full_transformation_bytes(tmp_path, edge):
    frames.persist_response_frame(_frame('a'), frames_dir=tmp_path)
    frames.persist_response_frame(_frame('b'), frames_dir=tmp_path)
    target = tmp_path / 'current_index.json'
    ledger = tmp_path / 'responses.jsonl'
    seed = json.loads(target.read_bytes())
    seed['responses']['b']['extra'] = edge
    seed['response_map_digest'] = frames._response_map_digest(seed['responses'])
    row = dict(_frame('a'), frame_id='a:frame-2', frame_sequence=2)
    kwargs = dict(
        frames_dir=tmp_path, ledger_path=ledger, line_offset=2,
        byte_offset=ledger.stat().st_size, line_length=10,
        ledger_size_bytes=ledger.stat().st_size + 10,
    )
    results = []
    for fallback in (False, True):
        value = dict(seed)
        if fallback:
            # This key is discarded by the established canonicalizer, and
            # prevents reuse without changing the canonical publication.
            value['response_frame'] = {'ignored': True}
        target.write_text(json.dumps(value))
        with patch.object(frames, '_prepared_response_map_digest', wraps=frames._prepared_response_map_digest) as digest:
            frames._write_response_frame_index(row, **kwargs)
        assert digest.call_count == 2  # old coverage and new publication
        results.append(target.read_bytes())
    assert results[0] == results[1]


def test_next_publication_rechecks_fresh_index_bytes(tmp_path):
    frames.persist_response_frame(_frame('a'), frames_dir=tmp_path)
    frames.persist_response_frame(_frame('b'), frames_dir=tmp_path)
    target = tmp_path / 'current_index.json'
    raw = json.loads(target.read_bytes())
    raw['responses']['b']['latest_frame_id'] = 'tampered'
    replacement = tmp_path / 'replacement.json'
    replacement.write_text(json.dumps(raw))
    replacement.replace(target)
    frames.persist_response_frame(_frame('a'), frames_dir=tmp_path)
    published = json.loads(target.read_bytes())
    assert 'response_map_digest' not in published
    assert 'response_map_verified_size_bytes' not in published
    assert frames.load_latest_response_state('b', frames_dir=tmp_path)['response_frame']['frame_id'] == 'b:frame-1'
