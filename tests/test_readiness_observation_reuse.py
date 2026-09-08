"""Only private observation reuse changes; all existing authority gates still run."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from ollmo_services import response_frames as rf
from ollmo_services import graph_rebase_readiness_registry as registry
from ollmo_services.graph_rebase_rollout import project_graph_rebase_readiness_observation as project


def frame(response_id='reuse-response', reason='additive_repair_insufficient'):
    return {
        'kind': 'ollmo.response_frame', 'frame_version': 9,
        'response_id': response_id, 'status': 'completed',
        'current_state': {'id': response_id, 'status': 'completed', 'lifecycle_state': 'completed'},
        'request': {'prompt': 'One bounded readiness test', 'workload_family': 'reuse'},
        'runtime': {'developer_diagnostics': {
            'response_time_graph_rebase_candidate': {'candidate': True, 'reason': reason},
        }},
    }


@pytest.fixture
def stable(tmp_path):
    frames = tmp_path / 'frames'
    rf.persist_response_frame(frame(), frames_dir=frames)
    epoch = rf.verify_response_frame_epoch(frames_dir=frames)
    assert epoch['ok']
    candidates = []
    observed = rf.load_latest_response_observation_state(
        'reuse-response', frames_dir=frames, index_state=epoch['index_state'],
        _verified_epoch=epoch, _reuse_candidates=candidates,
    )
    assert observed['ok'] and len(candidates) == 1
    return frames, epoch, observed, candidates[0]


def append(stable, path, *, candidate=True):
    frames, epoch, observed, receipt = stable
    return registry.append_graph_rebase_readiness_observation(
        project(observed['response_payload']),
        source_frame=epoch['source_frame_sha256_by_response']['reuse-response'],
        verified_epoch=epoch, frames_dir=frames, registry_path=path,
        _observation_candidate=receipt if candidate else None,
    )


def source_bytes(frames):
    return {str(p.relative_to(frames)): p.read_bytes() for p in frames.rglob('*') if p.is_file()}


def used_sidecar(stable):
    frames, _, _, candidate = stable
    # Test-only inspection of private receipt: pick an actually consumed source.
    ref_bytes, _, _ = candidate._receipt[3][0]
    return frames / json.loads(ref_bytes)['path']


def test_stable_reuses_with_identical_record_and_all_read_checks(stable, tmp_path):
    before = source_bytes(stable[0])
    append(stable, tmp_path/'old.jsonl', candidate=False)
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load, \
         patch.object(rf, '_response_frame_index_has_verified_response_map', wraps=rf._response_frame_index_has_verified_response_map) as mapping, \
         patch.object(rf, '_read_indexed_response_frame', wraps=rf._read_indexed_response_frame) as indexed, \
         patch.object(rf, '_read_observation_snapshot_bytes', wraps=rf._read_observation_snapshot_bytes) as sidecars, \
         patch.object(rf, '_read_observation_snapshot_payload', wraps=rf._read_observation_snapshot_payload) as reconstruct, \
         patch.object(registry, '_file_state', wraps=registry._file_state) as freshness:
        result = append(stable, tmp_path/'new.jsonl')
    assert result['status'] == 'appended'
    assert load.call_count == reconstruct.call_count == 0
    assert mapping.call_count == indexed.call_count == 1
    assert sidecars.call_count > 0 and freshness.call_count == 2
    assert (tmp_path/'old.jsonl').read_bytes() == (tmp_path/'new.jsonl').read_bytes()
    assert before == source_bytes(stable[0])


@pytest.mark.parametrize('mutation', ['touch', 'rewrite_same', 'changed', 'missing', 'corrupt'])
def test_sidecar_mutation_from_other_thread_forces_normal_loader(stable, tmp_path, mutation):
    p = used_sidecar(stable)
    def mutate():
        if mutation == 'touch': p.touch()
        elif mutation == 'rewrite_same': p.write_bytes(p.read_bytes())
        elif mutation == 'changed': p.write_bytes(p.read_bytes().replace(b'true', b'null', 1))
        elif mutation == 'missing': p.unlink()
        else: p.write_bytes(b'not JSON')
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(mutate).result()
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        if mutation in ['touch', 'rewrite_same']:
            assert append(stable, tmp_path/'new.jsonl')['ok']
        else:
            with pytest.raises(registry.GraphRebaseReadinessRegistryError):
                append(stable, tmp_path/'new.jsonl')
        assert load.call_count == 1


@pytest.mark.parametrize('mutation', ['index_file', 'ledger_file', 'successor', 'epoch_digest'])
def test_moved_epoch_keeps_existing_fail_closed_authority(stable, tmp_path, mutation):
    frames, epoch, _, _ = stable
    if mutation == 'index_file': (frames/'current_index.json').touch()
    elif mutation == 'ledger_file': (frames/'responses.jsonl').touch()
    elif mutation == 'successor': rf.persist_response_frame(frame(reason='new evidence'), frames_dir=frames)
    else: epoch['ok'] = False
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        with pytest.raises(registry.GraphRebaseReadinessRegistryError, match='Preverified response-frame epoch is no longer current'):
            append(stable, tmp_path/'new.jsonl')
        # Existing registry rejects before hydration; no new authority/recovery is invented.
        assert load.call_count == 0


def test_new_verified_epoch_and_successor_fall_back_to_current_frame(stable, tmp_path):
    frames, _, _, candidate = stable
    rf.persist_response_frame(frame(reason='new evidence'), frames_dir=frames)
    epoch = rf.verify_response_frame_epoch(frames_dir=frames)
    observed = rf.load_latest_response_observation_state('reuse-response', frames_dir=frames, index_state=epoch['index_state'])
    assert observed['response_frame']['frame_sequence'] == 2
    updated = frames, epoch, observed, candidate
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        assert append(updated, tmp_path/'new.jsonl')['ok']
        assert load.call_count == 1
    append(updated, tmp_path/'old.jsonl', candidate=False)
    assert (tmp_path/'new.jsonl').read_bytes() == (tmp_path/'old.jsonl').read_bytes()


@pytest.mark.parametrize('mutation', ['map_digest', 'map_entry', 'sequence', 'frame_id', 'source_digest'])
def test_bad_mapping_identity_never_reuses(stable, tmp_path, mutation):
    frames, epoch, observed, candidate = stable
    altered = copy.deepcopy(epoch)
    entry = altered['index_state']['responses']['reuse-response']
    if mutation == 'map_digest': altered['index_state']['response_map_digest'] = '0'*64
    elif mutation == 'map_entry': entry['untrusted_extra'] = True
    elif mutation == 'sequence': entry['latest_frame_sequence'] += 1
    elif mutation == 'frame_id': entry['latest_frame_id'] = 'wrong-frame'
    else: altered['source_frame_sha256_by_response']['reuse-response'] = '0'*64
    # Direct reuse boundary always misses; registry still owns rejection ordering.
    assert rf._reuse_response_observation_if_current(candidate, 'reuse-response', frames_dir=frames, index_state=altered['index_state'], verified_epoch=altered) is None
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        if mutation in ['map_digest', 'map_entry']:
            with pytest.raises(registry.GraphRebaseReadinessRegistryError): append((frames, altered, observed, candidate), tmp_path/'bad.jsonl')
            assert load.call_count == 1


def test_absent_untrusted_and_consumed_receipts_use_existing_loader(stable, tmp_path):
    frames, epoch, observed, candidate = stable
    for i, supplied in enumerate([None, {'response_id': 'reuse-response'}, candidate, candidate]):
        with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
            assert append((frames, epoch, observed, supplied), tmp_path/f'{i}.jsonl')['ok']
            assert load.call_count == (0 if i == 2 else 1)


def test_private_copy_survives_caller_and_projection_mutation(stable, tmp_path):
    frames, epoch, observed, candidate = stable
    append(stable, tmp_path/'old.jsonl', candidate=False)
    observed['response_payload']['request']['workload_family'] = 'caller-mutated'
    observed['response_payload']['runtime']['developer_diagnostics'].clear()
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        assert append(stable, tmp_path/'new.jsonl')['ok']
        assert load.call_count == 0
    assert (tmp_path/'old.jsonl').read_bytes() == (tmp_path/'new.jsonl').read_bytes()
    current = rf.load_latest_response_observation_state('reuse-response', frames_dir=frames, index_state=epoch['index_state'])
    assert current['response_payload']['request']['workload_family'] == 'reuse'


def test_registry_concurrent_append_keeps_dedup_validation(stable, tmp_path):
    # Registry contents are not inputs to observation hydration. A concurrent
    # registry append cannot invalidate source observation, but record checks run.
    target = tmp_path/'registry.jsonl'
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(append, stable, target, candidate=False).result()
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        result = append(stable, target)
        assert result['already_present_count'] == 1 and load.call_count == 0
    assert len(target.read_text().splitlines()) == 1


def test_registry_corruption_is_not_hidden_by_reuse(stable, tmp_path):
    target = tmp_path/'registry.jsonl';target.write_text('corrupt\n')
    result = append(stable, target)
    assert result['ok'] is False
    assert target.read_text() == 'corrupt\n'


def test_read_mutation_prevents_candidate_issuance(stable):
    frames, epoch, _, _ = stable
    original = rf._read_observation_snapshot_bytes
    def mutate_after_read(ref, **kw):
        result = original(ref, **kw)
        if isinstance(ref, dict): (frames/ref['path']).touch()
        return result
    candidates = []
    with patch.object(rf, '_read_observation_snapshot_bytes', side_effect=mutate_after_read):
        state = rf.load_latest_response_observation_state('reuse-response', frames_dir=frames, index_state=epoch['index_state'], _verified_epoch=epoch, _reuse_candidates=candidates)
    # The currentness check at the end of the read window refuses this receipt.
    assert candidates == []
    assert state['ok']


def test_same_frame_identity_with_changed_ledger_bytes_requires_new_load(stable, tmp_path):
    frames, _, _, candidate = stable
    ledger = frames/'responses.jsonl'
    before = ledger.read_bytes()
    after = before.replace(b'One bounded readiness test', b'One changed readiness test')
    assert before != after and len(before) == len(after)
    ledger.write_bytes(after)
    epoch = rf.verify_response_frame_epoch(frames_dir=frames)
    assert epoch['ok']
    observed = rf.load_latest_response_observation_state('reuse-response', frames_dir=frames, index_state=epoch['index_state'])
    assert observed['response_frame']['frame_sequence'] == 1
    with patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        assert append((frames, epoch, observed, candidate), tmp_path/'new.jsonl')['ok']
        assert load.call_count == 1


def test_concurrent_consumers_can_reuse_receipt_only_once(stable, tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool, \
         patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        jobs = [pool.submit(append, stable, tmp_path/f'{i}.jsonl') for i in range(2)]
        assert all(job.result()['ok'] for job in jobs)
        assert load.call_count == 1
    assert (tmp_path/'0.jsonl').read_bytes() == (tmp_path/'1.jsonl').read_bytes()


def test_reuse_read_window_mutation_falls_back(stable, tmp_path):
    p = used_sidecar(stable)
    original = rf._read_observation_snapshot_bytes
    def move_after_validated_read(ref, **kw):
        result = original(ref, **kw)
        p.touch()
        return result
    with patch.object(rf, '_read_observation_snapshot_bytes', side_effect=move_after_validated_read), \
         patch.object(registry, 'load_latest_response_observation_state', wraps=rf.load_latest_response_observation_state) as load:
        assert append(stable, tmp_path/'new.jsonl')['ok']
        assert load.call_count == 1
