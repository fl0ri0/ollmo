from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import export_reference_run as exporter


RESPONSE_ID = 'resp_reference_fixture'
FRAME_ID = f'{RESPONSE_ID}:frame-2'
PROMPT = 'Create one small local image and describe it.'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, payload: dict) -> bytes:
    encoded = (json.dumps(payload, sort_keys=True) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def _fixture_truth(tmp_path: Path) -> dict:
    source_root = tmp_path / 'source'
    artifact_root = source_root / 'artifacts'
    artifact = artifact_root / 'images' / 'fixture.png'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\x0dIHDR'
        b'\x00\x00\x00\x20\x00\x00\x00\x10'
        b'fixture-image-bytes'
    )
    readme = source_root / 'examples' / 'reference-runs' / 'fixture.md'
    readme.parent.mkdir(parents=True)
    readme.write_text(
        '# Fixture\n\n'
        '## Request\n\n'
        f'```text\n{PROMPT}\n```\n\n'
        '## Result\n\n'
        '[Open the image](../../artifacts/images/fixture.png)\n\n'
        '[Structured monitor reports](../../state/reports.jsonl)\n',
        encoding='utf-8',
    )
    ledger_payload = {
        'response_id': RESPONSE_ID,
        'frame_id': FRAME_ID,
        'frame_sequence': 2,
        'status': 'completed',
    }
    ledger = tmp_path / 'frames' / 'responses.jsonl'
    row = _write_jsonl(ledger, ledger_payload)
    monitor_payload = {
        'response_id': RESPONSE_ID,
        'frame_id': FRAME_ID,
        'frame_sequence': 2,
        'reported_at': '2026-09-02T12:00:00Z',
        'verdict': 'clean',
        'lifecycle_state': 'completed',
        'late_fill_status': 'completed',
        'final_materialization_contract_status': 'fulfilled',
        'materialization_contract_unmet': False,
        'failed_branch_count': 0,
        'branch_counts': {'completed': 1, 'failed': 0, 'pending': 0},
        'branch_capability_counts': {'image_generation': 1},
        'branch_role_counts': {'image': 1},
        'output_obligation_counts': {'image': 1, 'text': 1},
        'image_branch_count': 1,
        'text_artifact_branch_count': 0,
        'artifacts': {
            'artifact_count': 1,
            'artifact_bytes': artifact.stat().st_size,
            'artifact_kind_counts': {'image': 1},
            'artifact_file_count_by_suffix': {'png': 1},
            'missing_files': [],
            'sha_mismatches': [],
            'html_issues': [],
            'css_issues': [],
        },
        'timing': {'terminal_seconds': 1.5},
    }
    monitor = tmp_path / 'monitor' / 'reports.jsonl'
    _write_jsonl(monitor, monitor_payload)
    index = {
        'ok': True,
        'ledger_path': str(ledger),
        'responses': {
            RESPONSE_ID: {
                'ledger_path': str(ledger),
                'byte_offset': 0,
                'line_length': len(row),
                'latest_frame_id': FRAME_ID,
                'latest_frame_sequence': 2,
            }
        },
    }
    wire = {
        'ok': True,
        'ledger_fallback_used': False,
        'response_payload': {
            'response_id': RESPONSE_ID,
            'status': 'completed',
            'lifecycle_state': 'completed',
            'response_frame': {
                'frame_id': FRAME_ID,
                'frame_sequence': 2,
                'frame_version': 9,
                'current_state': {
                    'status_semantics': {
                        'is_terminal': True,
                        'terminal': True,
                        'has_open_continuation': False,
                        'has_actionable_repair': False,
                    }
                },
            },
            'outputs': [
                {
                    'slot_id': 'output-phase-1',
                    'branch_id': 'phase-1',
                    'phase_id': 'phase-1',
                    'type': 'text',
                    'status': 'fulfilled',
                    'lifecycle': 'materialized_output',
                    'value': 'A small blue square.',
                },
                {
                    'slot_id': 'output-phase-2',
                    'branch_id': 'branch-image_generation-1',
                    'phase_id': 'phase-2',
                    'type': 'image',
                    'status': 'fulfilled',
                    'lifecycle': 'materialized_output',
                    'artifact_ref': 'artifact:image_fixture',
                },
            ],
            'artifacts': [
                {
                    'artifact_ref': 'artifact:image_fixture',
                    'source_response_id': RESPONSE_ID,
                    'type': 'image',
                    'path': str(artifact),
                    'phase_id': 'phase-2',
                    'branch_id': 'branch-image_generation-1',
                }
            ],
            'late_fill': {
                'status': 'completed',
                'final_materialization_contract_status': 'fulfilled',
                'completed_branch_count': 1,
                'failed_branch_count': 0,
                'pending_branch_count': 0,
                'cancelled_branch_count': 0,
            },
        },
    }
    observation = {
        'ok': True,
        'response_payload': {
            'response_id': RESPONSE_ID,
            'response_frame': {
                'frame_id': FRAME_ID,
                'frame_sequence': 2,
            },
            'request': {'prompt': exporter._single_user_message_repr(PROMPT)},
            'runtime': {
                'graph_closure_review': {
                    'status': 'fulfilled',
                    'continuation_required': False,
                    'obligation_count': 2,
                    'pending_branch_count': 0,
                    'counts': {
                        'fulfilled': 2,
                        'pending': 0,
                        'blocked': 0,
                    },
                    'surface_state': {
                        'status': 'fulfilled',
                        'active_categories': ['completed'],
                        'category_counts': {
                            'completed': 2,
                            'open': 0,
                            'blocked': 0,
                            'repair_pending': 0,
                        },
                    },
                }
            },
        },
    }
    return {
        'source_root': source_root,
        'artifact_root': artifact_root,
        'artifact': artifact,
        'readme': readme,
        'frames_dir': ledger.parent,
        'monitor': monitor,
        'index': index,
        'wire': wire,
        'observation': observation,
    }


def _patch_truth(monkeypatch: pytest.MonkeyPatch, fixture: dict) -> None:
    monkeypatch.setattr(
        exporter,
        'load_response_frame_index',
        lambda **_kwargs: fixture['index'],
    )
    monkeypatch.setattr(
        exporter,
        'load_latest_response_wire_state',
        lambda *_args, **_kwargs: fixture['wire'],
    )
    monkeypatch.setattr(
        exporter,
        'load_latest_response_observation_state',
        lambda *_args, **_kwargs: fixture['observation'],
    )


def test_export_reference_run_is_self_contained_and_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_truth(tmp_path)
    _patch_truth(monkeypatch, fixture)
    destination = tmp_path / 'published' / 'fixture'

    exporter.export_reference_run(
        response_id=RESPONSE_ID,
        source_readme=fixture['readme'],
        output_dir=destination,
        frames_dir=fixture['frames_dir'],
        monitor_reports=fixture['monitor'],
        artifact_root=fixture['artifact_root'],
    )

    manifest = exporter.verify_reference_package(destination)
    assert manifest['validation']['terminal_runtime_truth_verified'] is True
    assert (destination / 'artifacts/images/image-01.png').read_bytes() == (
        fixture['artifact'].read_bytes()
    )
    assert '(artifacts/images/image-01.png)' in (
        destination / 'README.md'
    ).read_text(encoding='utf-8')
    response = json.loads((destination / 'response.json').read_text(encoding='utf-8'))
    assert response['outputs'][0]['text'] == 'A small blue square.'
    assert response['outputs'][1]['published_path'] == 'artifacts/images/image-01.png'
    assert response['integrity']['ledger_fallback_used'] is False
    for path in destination.rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.md', '.txt'}:
            assert '/Users/' not in path.read_text(encoding='utf-8')


def test_verify_detects_changed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_truth(tmp_path)
    _patch_truth(monkeypatch, fixture)
    destination = tmp_path / 'published' / 'fixture'
    exporter.export_reference_run(
        response_id=RESPONSE_ID,
        source_readme=fixture['readme'],
        output_dir=destination,
        frames_dir=fixture['frames_dir'],
        monitor_reports=fixture['monitor'],
        artifact_root=fixture['artifact_root'],
    )
    (destination / 'artifacts/images/image-01.png').write_bytes(b'changed')

    with pytest.raises(exporter.ReferenceExportError, match='checksum mismatch'):
        exporter.verify_reference_package(destination)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda fixture: fixture['wire']['response_payload'].update(
                {'lifecycle_state': 'repair_needed'}
            ),
            'truthfully complete',
        ),
        (
            lambda fixture: fixture['observation']['response_payload']['request'].update(
                {'prompt': exporter._single_user_message_repr('Different prompt.')}
            ),
            'reviewed prompt',
        ),
    ],
)
def test_export_fails_closed_on_truth_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    fixture = _fixture_truth(tmp_path)
    mutation(fixture)
    _patch_truth(monkeypatch, fixture)
    destination = tmp_path / 'published' / 'fixture'

    with pytest.raises(exporter.ReferenceExportError, match=message):
        exporter.export_reference_run(
            response_id=RESPONSE_ID,
            source_readme=fixture['readme'],
            output_dir=destination,
            frames_dir=fixture['frames_dir'],
            monitor_reports=fixture['monitor'],
            artifact_root=fixture['artifact_root'],
        )
    assert not destination.exists()


def test_export_rejects_artifact_outside_approved_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture_truth(tmp_path)
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    fixture['wire']['response_payload']['artifacts'][0]['path'] = str(outside)
    _patch_truth(monkeypatch, fixture)

    with pytest.raises(exporter.ReferenceExportError, match='outside the approved root'):
        exporter.export_reference_run(
            response_id=RESPONSE_ID,
            source_readme=fixture['readme'],
            output_dir=tmp_path / 'published' / 'fixture',
            frames_dir=fixture['frames_dir'],
            monitor_reports=fixture['monitor'],
            artifact_root=fixture['artifact_root'],
        )


def test_bundle_copy_renames_assets_and_removes_local_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / 'artifacts'
    canonical = artifact_root / 'images' / 'model-output.png'
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b'image')
    bundle = artifact_root / 'bundles' / 'bundle'
    image = bundle / 'assets' / 'images' / 'model-output.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(canonical.read_bytes())
    index = bundle / 'index.html'
    index.write_text(
        '<!doctype html><img src="assets/images/model-output.png">\n',
        encoding='utf-8',
    )
    manifest = {
        'source_response_id': RESPONSE_ID,
        'bundle_id': 'bundle:fixture',
        'status': 'bundled',
        'link_check': {'status': 'passed'},
        'entrypoint_relative_path': 'index.html',
        'copied_artifacts': [
            {
                'artifact_ref': 'artifact:text_index',
                'type': 'text',
                'relative_path': 'index.html',
                'source_path': str(artifact_root / 'documents' / 'index.html'),
                'size_bytes': index.stat().st_size,
                'sha256': _sha256(index),
            },
            {
                'artifact_ref': 'artifact:image_fixture',
                'type': 'image',
                'relative_path': 'assets/images/model-output.png',
                'source_path': str(canonical),
                'size_bytes': image.stat().st_size,
                'sha256': _sha256(image),
            },
        ],
        'rewritten_links': [
            {
                'kind': 'html_src',
                'original': '../images/model-output.png',
                'rewritten': 'assets/images/model-output.png',
            }
        ],
    }
    (bundle / 'manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8'
    )
    staging = tmp_path / 'staging'
    staging.mkdir()

    public_manifest, ref_paths, _source_paths = exporter._copy_bundle(
        bundle,
        response_id=RESPONSE_ID,
        artifact_root=artifact_root,
        staging_root=staging,
    )

    assert ref_paths['artifact:image_fixture'] == Path(
        'bundle/assets/images/image-01.png'
    )
    assert 'assets/images/image-01.png' in (
        staging / 'bundle/index.html'
    ).read_text(encoding='utf-8')
    encoded = json.dumps(public_manifest)
    assert '/Users/' not in encoded
    assert 'model-output.png' not in encoded
