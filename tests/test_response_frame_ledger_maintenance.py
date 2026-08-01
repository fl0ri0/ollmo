import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from ollmo_services.response_frame_ledger_maintenance import (
    audit_response_frame_ledger,
    compact_response_frame_ledger,
)
from ollmo_services.response_frames import (
    _compact_request_ghost_preview,
    _read_snapshot_ref_payload,
    _write_snapshot_ref,
)


def _large_ghost_preview() -> dict[str, Any]:
    return {
        'instance': {
            'instance_id': 'ghost-preview-cas-test',
            'model': 'test-model',
            'backend': 'fake',
            'capability': 'chat',
        },
        'route': {
            'source': 'test',
            'reason': 'exercise historical ledger compaction',
            'confidence': 0.91,
        },
        'request_meta': {
            'ghost_mode': 'assistant',
            'capability_hint': 'chat',
        },
        # Keep every child below the recursive-sidecar split threshold while
        # making the complete preview large enough for Ghost-preview CAS.
        'legacy_routing_evidence': {
            f'evidence_{index:02d}': f'{index:02d}-' + ('evidence ' * 120)
            for index in range(20)
        },
    }


def _frame(
    response_id: str,
    *,
    frame_sequence: int = 1,
    ghost_preview: dict[str, Any] | None = None,
    parent_frame_id: str | None = None,
) -> dict[str, Any]:
    frame_id = f'frame-{response_id}-{frame_sequence}'
    relation = {
        'kind': 'initial' if parent_frame_id is None else 'late_fill_successor',
        'response_id': response_id,
        'parent_response_id': response_id if parent_frame_id else None,
        'parent_frame_id': parent_frame_id,
        'parent_frame_sequence': frame_sequence - 1 if parent_frame_id else None,
    }
    frame: dict[str, Any] = {
        'frame_version': 9,
        'kind': 'ollmo.response_frame',
        'response_id': response_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'frame_relation': relation,
        'object': 'response',
        'status': 'completed',
        'request': {'prompt': f'test prompt for {response_id}'},
        'current_state': {
            'id': response_id,
            'object': 'response',
            'status': 'completed',
            'lifecycle_state': 'completed',
        },
    }
    if ghost_preview is not None:
        frame['request']['ghost_preview'] = ghost_preview
    return frame


def _write_ledger(frames_dir: Path, frames: list[dict[str, Any]]) -> Path:
    frames_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = frames_dir / 'responses.jsonl'
    ledger_path.write_bytes(
        b''.join(
            json.dumps(frame, ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n'
            for frame in frames
        )
    )
    return ledger_path


def _legacy_recursive_snapshot_frame(
    frames_dir: Path,
    *,
    response_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frame = _frame(response_id)
    root_ref = _write_snapshot_ref(
        {
            'candidate_graph': {
                'kind': 'ollmo.test_candidate_graph',
                'payload': 'manifest-bound-child-' * 2_500,
            }
        },
        frame=frame,
        frames_dir=frames_dir,
        json_path='runtime',
    )
    root_payload = json.loads(
        (frames_dir / root_ref['path']).read_text(encoding='utf-8')
    )
    child_ref = root_payload['candidate_graph_snapshot_ref']
    child_authority = root_ref['sidecar_manifest']['child_refs'][0]
    root_ref['sidecar_manifest']['child_refs'] = [
        {
            key: child_authority[key]
            for key in ('json_path', 'key', 'sha256', 'size_bytes')
        }
    ]
    frame['external_snapshots'] = {
        'kind': 'ollmo.response_frame_external_snapshots',
        'items': {'runtime': root_ref},
        'storage': 'sidecar_json',
        'version': 1,
    }
    return frame, root_ref, child_ref


def _read_ledger_lines(ledger_path: Path) -> tuple[list[bytes], list[dict[str, Any]]]:
    raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
    return raw_lines, [json.loads(raw_line) for raw_line in raw_lines]


def _write_exact_source_index(frames_dir: Path) -> Path:
    ledger_path = frames_dir / 'responses.jsonl'
    raw_lines, frames = _read_ledger_lines(ledger_path)
    responses: dict[str, dict[str, Any]] = {}
    byte_offset = 0
    for line_offset, (raw_line, frame) in enumerate(zip(raw_lines, frames)):
        response_id = frame['response_id']
        relation = {
            key: value
            for key, value in (frame.get('frame_relation') or {}).items()
            if value is not None
        }
        current_state = frame.get('current_state') or {}
        entry: dict[str, Any] = {
            'response_id': response_id,
            'latest_frame_id': frame['frame_id'],
            'latest_frame_sequence': frame['frame_sequence'],
            'frame_relation': relation,
            'ledger_path': str(ledger_path),
            'ledger_name': ledger_path.name,
            'line_offset': line_offset,
            'byte_offset': byte_offset,
            'line_length': len(raw_line),
            'ledger_size_bytes': byte_offset + len(raw_line),
            'current_lifecycle_state': current_state.get('lifecycle_state'),
        }
        responses[response_id] = {
            key: value for key, value in entry.items() if value is not None
        }
        byte_offset += len(raw_line)
    payload = {
        'kind': 'ollmo.response_frame_current_index',
        'version': 2,
        'ledger_path': str(ledger_path),
        'ledger_name': ledger_path.name,
        'ledger_line_count': len(raw_lines),
        'ledger_size_bytes': byte_offset,
        'ledger_line_count_verified_size_bytes': byte_offset,
        'response_map_verified_size_bytes': byte_offset,
        'response_map_entry_count': len(responses),
        'response_map_digest': _response_map_digest(responses),
        'responses': responses,
    }
    index_path = frames_dir / 'current_index.json'
    index_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
        encoding='utf-8',
    )
    return index_path


def _tree_fingerprint(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


def _response_map_digest(responses: dict[str, Any]) -> str:
    canonical = json.dumps(
        responses,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


class ResponseFrameLedgerMaintenanceTests(unittest.TestCase):
    def test_dry_run_and_audit_are_strictly_read_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            _write_ledger(
                frames_dir,
                [_frame('resp-dry-run', ghost_preview=_large_ghost_preview())],
            )
            before = _tree_fingerprint(frames_dir)

            audit = audit_response_frame_ledger(frames_dir=frames_dir)
            report = compact_response_frame_ledger(frames_dir=frames_dir, execute=False)

            self.assertEqual(_tree_fingerprint(frames_dir), before)
            self.assertFalse((frames_dir / 'current_index.json').exists())
            self.assertFalse((frames_dir / 'snapshots').exists())

        self.assertEqual(audit['mode'], 'audit')
        self.assertFalse(audit['changed'])
        self.assertTrue(audit['ok'])
        self.assertEqual(audit['eligible_ghost_preview_frame_count'], 1)
        self.assertEqual(audit['authoritative_missing_sidecar_count'], 0)

        self.assertEqual(report['mode'], 'audit')
        self.assertFalse(report['changed'])
        self.assertTrue(report['ok'])
        self.assertEqual(report['ledger_line_count'], 1)
        self.assertEqual(report['response_count'], 1)
        self.assertEqual(report['eligible_ghost_preview_frame_count'], 1)
        self.assertGreater(report['inline_ghost_preview_bytes'], 8_192)
        self.assertGreater(report['estimated_reclaimable_inline_bytes'], 0)

    def test_execute_uses_existing_content_sha256_cas_and_dedupes_identical_previews(self):
        preview = _large_ghost_preview()
        first_parent_id = 'frame-resp-a-1'
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            ledger_path = _write_ledger(
                frames_dir,
                [
                    _frame('resp-a', ghost_preview=preview),
                    _frame('resp-b', ghost_preview=preview),
                    _frame(
                        'resp-a',
                        frame_sequence=2,
                        ghost_preview=preview,
                        parent_frame_id=first_parent_id,
                    ),
                ],
            )
            _write_exact_source_index(frames_dir)

            report = compact_response_frame_ledger(
                frames_dir=frames_dir,
                execute=True,
                writers_stopped=True,
            )
            raw_lines, migrated_frames = _read_ledger_lines(ledger_path)
            index = json.loads((frames_dir / 'current_index.json').read_bytes())
            cas_files = sorted(
                (frames_dir / 'snapshots' / 'content_sha256').rglob('*.json')
            )
            cas_payload = cas_files[0].read_bytes()

            refs = [
                frame['request']['ghost_preview_snapshot_ref']
                for frame in migrated_frames
            ]
            hydrated_previews = [
                _read_snapshot_ref_payload(ref, frames_dir=frames_dir)
                for ref in refs
            ]
            after_first_execute = _tree_fingerprint(frames_dir)
            second_report = compact_response_frame_ledger(
                frames_dir=frames_dir,
                execute=True,
                writers_stopped=True,
            )
            after_second_execute = _tree_fingerprint(frames_dir)

        self.assertTrue(report['ok'])
        self.assertEqual(report['status'], 'compacted')
        self.assertEqual(report['mode'], 'execute')
        self.assertTrue(report['changed'])
        self.assertEqual(report['rewrite']['changed_frame_count'], 3)
        self.assertEqual(report['preflight']['eligible_ghost_preview_frame_count'], 3)
        self.assertEqual(report['postflight']['eligible_ghost_preview_frame_count'], 0)

        self.assertEqual(len(cas_files), 1)
        self.assertEqual(len({ref['sha256'] for ref in refs}), 1)
        self.assertEqual(len({ref['path'] for ref in refs}), 1)
        effective_manifest_by_frame_id: dict[str, dict[str, Any]] = {}
        for frame, ref, hydrated in zip(migrated_frames, refs, hydrated_previews):
            self.assertEqual(ref['kind'], 'ollmo.response_frame_snapshot_ref')
            self.assertEqual(ref['json_path'], 'request.ghost_preview')
            self.assertTrue(ref['content_addressed'])
            self.assertEqual(ref['dedupe_scope'], 'response_frame_snapshot_store')
            relation = frame.get('frame_relation') or {}
            parent_manifest = effective_manifest_by_frame_id.get(
                relation.get('parent_frame_id'),
                {},
            )
            effective_manifest = {
                **parent_manifest,
                **(frame['external_snapshots'].get('items') or {}),
            }
            effective_manifest_by_frame_id[frame['frame_id']] = effective_manifest
            authorized_ref = effective_manifest['request.ghost_preview']
            self.assertEqual(authorized_ref['sha256'], ref['sha256'])
            self.assertEqual(authorized_ref['path'], ref['path'])
            self.assertEqual(frame['snapshot_policy']['dedupe_strategy'], 'content_sha256')
            self.assertEqual(frame['request']['ghost_preview'], _compact_request_ghost_preview(preview))
            self.assertLess(
                len(json.dumps(frame['request']['ghost_preview']).encode('utf-8')),
                8_192,
            )
            self.assertEqual(hydrated, preview)

        if cas_payload.endswith(b'\n'):
            cas_payload = cas_payload[:-1]
        self.assertEqual(hashlib.sha256(cas_payload).hexdigest(), refs[0]['sha256'])
        self.assertEqual(len(cas_payload), refs[0]['size_bytes'])

        self.assertEqual(index['kind'], 'ollmo.response_frame_current_index')
        self.assertEqual(index['version'], 2)
        self.assertEqual(index['ledger_line_count'], len(raw_lines))
        self.assertEqual(index['ledger_size_bytes'], sum(map(len, raw_lines)))
        self.assertEqual(
            index['ledger_line_count_verified_size_bytes'],
            index['ledger_size_bytes'],
        )
        self.assertEqual(
            index['response_map_verified_size_bytes'],
            index['ledger_size_bytes'],
        )
        self.assertEqual(index['response_map_entry_count'], 2)
        self.assertEqual(index['response_map_digest'], _response_map_digest(index['responses']))

        expected_latest_lines = {'resp-a': 2, 'resp-b': 1}
        byte_offsets: list[int] = []
        running_offset = 0
        for raw_line in raw_lines:
            byte_offsets.append(running_offset)
            running_offset += len(raw_line)
        for response_id, line_offset in expected_latest_lines.items():
            entry = index['responses'][response_id]
            frame = migrated_frames[line_offset]
            self.assertEqual(entry['line_offset'], line_offset)
            self.assertEqual(entry['byte_offset'], byte_offsets[line_offset])
            self.assertEqual(entry['line_length'], len(raw_lines[line_offset]))
            self.assertEqual(entry['ledger_size_bytes'], index['ledger_size_bytes'])
            self.assertEqual(entry['latest_frame_id'], frame['frame_id'])
            self.assertEqual(entry['latest_frame_sequence'], frame['frame_sequence'])
            self.assertEqual(
                entry['effective_snapshot_manifest']['request.ghost_preview']['sha256'],
                frame['request']['ghost_preview_snapshot_ref']['sha256'],
            )

        self.assertTrue(second_report['ok'])
        self.assertEqual(second_report['status'], 'already_compact')
        self.assertEqual(second_report['mode'], 'execute')
        self.assertFalse(second_report['changed'])
        self.assertEqual(
            second_report['preflight']['eligible_ghost_preview_frame_count'],
            0,
        )
        self.assertEqual(after_second_execute, after_first_execute)

    def test_manifest_authorized_missing_snapshot_blocks_execute_without_mutation(self):
        response_id = 'resp-missing-authoritative-sidecar'
        frame = _frame(response_id, ghost_preview=_large_ghost_preview())
        missing_digest = 'f' * 64
        missing_ref = {
            'kind': 'ollmo.response_frame_snapshot_ref',
            'json_path': 'runtime',
            'path': f'snapshots/content_sha256/ff/{missing_digest}.json',
            'sha256': missing_digest,
            'size_bytes': 1234,
            'content_addressed': True,
            'dedupe_scope': 'response_frame_snapshot_store',
            'source_response_id': response_id,
            'source_frame_id': frame['frame_id'],
        }
        frame['external_snapshots'] = {
            'kind': 'ollmo.response_frame_external_snapshots',
            'items': {'runtime': missing_ref},
            'storage': 'sidecar_json',
            'version': 1,
        }
        frame['snapshot_policy'] = {
            'kind': 'ollmo.response_frame_snapshot_policy',
            'dedupe_strategy': 'content_sha256',
            'snapshot_ref_count': 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            _write_ledger(frames_dir, [frame])
            before = _tree_fingerprint(frames_dir)

            report = compact_response_frame_ledger(
                frames_dir=frames_dir,
                execute=True,
                writers_stopped=True,
            )

            self.assertEqual(_tree_fingerprint(frames_dir), before)
            self.assertFalse((frames_dir / 'current_index.json').exists())
            self.assertFalse((frames_dir / 'snapshots').exists())

        self.assertFalse(report['ok'])
        self.assertFalse(report['changed'])
        self.assertEqual(report['mode'], 'execute')
        self.assertEqual(
            report['preflight']['authoritative_snapshot_unique_count'],
            1,
        )
        self.assertEqual(report['preflight']['authoritative_missing_sidecar_count'], 1)
        self.assertEqual(
            report['preflight']['digest_only_audit_identity_occurrence_count'],
            0,
        )
        self.assertEqual(
            report['preflight']['authoritative_missing_sidecars'][0]['reason'],
            'missing_file',
        )
        self.assertEqual(
            report['error']['code'],
            'response_frame_ledger_preflight_failed',
        )

    def test_stale_source_index_blocks_execute_before_cas_or_ledger_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            _write_ledger(
                frames_dir,
                [_frame('resp-stale-index', ghost_preview=_large_ghost_preview())],
            )
            index_path = _write_exact_source_index(frames_dir)
            index = json.loads(index_path.read_text(encoding='utf-8'))
            index['ledger_size_bytes'] += 1
            index_path.write_text(
                json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
                encoding='utf-8',
            )
            before = _tree_fingerprint(frames_dir)

            report = compact_response_frame_ledger(
                frames_dir=frames_dir,
                execute=True,
                writers_stopped=True,
            )

            self.assertEqual(_tree_fingerprint(frames_dir), before)
            self.assertFalse((frames_dir / 'snapshots').exists())

        self.assertFalse(report['ok'])
        self.assertFalse(report['changed'])
        self.assertEqual(
            report['error']['code'],
            'response_frame_index_preflight_failed',
        )
        self.assertFalse(report['index_preflight']['ok'])

    def test_legacy_child_manifest_without_path_resolves_ref_from_parent_cas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            frame, root_ref, child_ref = _legacy_recursive_snapshot_frame(
                frames_dir,
                response_id='resp-legacy-child-manifest',
            )
            _write_ledger(frames_dir, [frame])
            before = _tree_fingerprint(frames_dir)

            report = audit_response_frame_ledger(frames_dir=frames_dir)

            self.assertEqual(_tree_fingerprint(frames_dir), before)

        legacy_child_authority = root_ref['sidecar_manifest']['child_refs'][0]
        self.assertNotIn('path', legacy_child_authority)
        self.assertEqual(legacy_child_authority['sha256'], child_ref['sha256'])
        self.assertTrue(report['ok'])
        self.assertEqual(report['authoritative_snapshot_ref_occurrence_count'], 2)
        self.assertEqual(report['authoritative_snapshot_unique_count'], 2)
        self.assertEqual(report['authoritative_snapshot_verified_unique_count'], 2)
        self.assertEqual(report['authoritative_missing_sidecar_count'], 0)
        self.assertTrue(report['gc_safe_from_authoritative_integrity'])

    def test_legacy_child_manifest_reports_actual_missing_child_cas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            frame, _root_ref, child_ref = _legacy_recursive_snapshot_frame(
                frames_dir,
                response_id='resp-missing-legacy-child',
            )
            _write_ledger(frames_dir, [frame])
            (frames_dir / child_ref['path']).unlink()
            before = _tree_fingerprint(frames_dir)

            report = audit_response_frame_ledger(frames_dir=frames_dir)

            self.assertEqual(_tree_fingerprint(frames_dir), before)

        self.assertFalse(report['ok'])
        self.assertEqual(report['authoritative_snapshot_ref_occurrence_count'], 2)
        self.assertEqual(report['authoritative_snapshot_unique_count'], 2)
        self.assertEqual(report['authoritative_snapshot_verified_unique_count'], 1)
        self.assertEqual(report['authoritative_missing_sidecar_count'], 1)
        missing = report['authoritative_missing_sidecars'][0]
        self.assertEqual(missing['json_path'], 'runtime.candidate_graph')
        self.assertEqual(missing['path'], child_ref['path'])
        self.assertEqual(missing['sha256'], child_ref['sha256'])
        self.assertEqual(missing['reason'], 'missing_file')
        self.assertFalse(report['gc_safe_from_authoritative_integrity'])

    def test_digest_only_audit_identity_without_path_is_not_a_missing_sidecar(self):
        digest_only_ref = {
            'kind': 'ollmo.ghost_preview_content_digest_ref',
            'json_path': 'ghost_preview.working_frame',
            'sha256': hashlib.sha256(b'identity-only').hexdigest(),
            'size_bytes': len(b'identity-only'),
            'content_addressed': True,
            'storage': 'digest_only',
            'authority': 'audit_identity_only',
        }
        preview = _large_ghost_preview()
        preview['compaction'] = {
            'kind': 'ollmo.ghost_preview_response_frame_projection',
            'omitted_content_refs': [digest_only_ref],
        }
        preview['copied_model_data'] = {
            'snapshot_ref': {
                'kind': 'ollmo.response_frame_snapshot_ref',
                'json_path': 'model.copied_snapshot',
                'path': 'snapshots/content_sha256/ee/' + ('e' * 64) + '.json',
                'sha256': 'e' * 64,
                'size_bytes': 999,
                'content_addressed': True,
                'dedupe_scope': 'response_frame_snapshot_store',
                'source_response_id': 'untrusted-copied-response',
                'source_frame_id': 'untrusted-copied-frame',
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            _write_ledger(
                frames_dir,
                [_frame('resp-digest-only-audit-ref', ghost_preview=preview)],
            )
            before = _tree_fingerprint(frames_dir)

            report = audit_response_frame_ledger(frames_dir=frames_dir)

            self.assertEqual(_tree_fingerprint(frames_dir), before)

        self.assertTrue(report['ok'])
        self.assertEqual(report['mode'], 'audit')
        self.assertFalse(report['changed'])
        self.assertEqual(report['authoritative_snapshot_unique_count'], 0)
        self.assertEqual(report['authoritative_snapshot_verified_unique_count'], 0)
        self.assertEqual(report['authoritative_missing_sidecar_count'], 0)
        self.assertEqual(report['digest_only_audit_identity_occurrence_count'], 1)
        self.assertEqual(report['digest_only_audit_identity_unique_count'], 1)
        self.assertFalse(report['digest_only_audit_identities_are_sidecars'])
        self.assertEqual(report['opaque_preview_snapshot_ref_occurrence_count'], 1)
        self.assertTrue(report['gc_safe_from_authoritative_integrity'])

    def test_malformed_digest_only_lookalike_fails_closed(self):
        digest_only_ref = {
            'kind': 'ollmo.ghost_preview_content_digest_ref',
            'json_path': 'ghost_preview.working_frame',
            'path': 'snapshots/content_sha256/aa/lookalike.json',
            'sha256': hashlib.sha256(b'lookalike').hexdigest(),
            'size_bytes': len(b'lookalike'),
            'content_addressed': True,
            'storage': 'sidecar_json',
            'authority': 'audit_identity_only',
        }
        preview = _large_ghost_preview()
        preview['compaction'] = {
            'kind': 'ollmo.ghost_preview_response_frame_projection',
            'omitted_content_refs': [digest_only_ref],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            _write_ledger(
                frames_dir,
                [_frame('resp-malformed-digest-ref', ghost_preview=preview)],
            )
            before = _tree_fingerprint(frames_dir)

            report = audit_response_frame_ledger(frames_dir=frames_dir)

            self.assertEqual(_tree_fingerprint(frames_dir), before)

        self.assertFalse(report['ok'])
        self.assertEqual(report['authoritative_missing_sidecar_count'], 0)
        self.assertEqual(report['digest_only_audit_identity_occurrence_count'], 0)
        self.assertEqual(report['malformed_digest_only_audit_identity_count'], 1)
        self.assertEqual(
            report['malformed_digest_only_audit_identities'][0]['reason'],
            'invalid_digest_ref_storage',
        )
        self.assertFalse(report['authoritative_integrity_ok'])

    def test_manifest_ref_with_noncanonical_cas_path_fails_closed(self):
        response_id = 'resp-noncanonical-cas-path'
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / 'response_frames'
            frame = _frame(response_id)
            ref = _write_snapshot_ref(
                {'runtime_truth': 'bound-but-wrong-location'},
                frame=frame,
                frames_dir=frames_dir,
                json_path='runtime',
            )
            source_path = frames_dir / ref['path']
            noncanonical_path = Path('snapshots') / 'copied' / source_path.name
            (frames_dir / noncanonical_path).parent.mkdir(parents=True, exist_ok=True)
            (frames_dir / noncanonical_path).write_bytes(source_path.read_bytes())
            ref['path'] = noncanonical_path.as_posix()
            frame['external_snapshots'] = {
                'kind': 'ollmo.response_frame_external_snapshots',
                'items': {'runtime': ref},
                'storage': 'sidecar_json',
                'version': 1,
            }
            _write_ledger(frames_dir, [frame])
            before = _tree_fingerprint(frames_dir)

            report = audit_response_frame_ledger(frames_dir=frames_dir)

            self.assertEqual(_tree_fingerprint(frames_dir), before)

        self.assertFalse(report['ok'])
        self.assertEqual(report['authoritative_snapshot_unique_count'], 1)
        self.assertEqual(report['authoritative_missing_sidecar_count'], 1)
        self.assertEqual(
            report['authoritative_missing_sidecars'][0]['reason'],
            'noncanonical_content_sha256_path',
        )

    def test_existing_backup_directory_blocks_before_any_cas_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'response_frames'
            _write_ledger(
                frames_dir,
                [_frame('resp-existing-backup', ghost_preview=_large_ghost_preview())],
            )
            _write_exact_source_index(frames_dir)
            backup_dir = root / 'already-reserved-backup'
            backup_dir.mkdir()
            before = _tree_fingerprint(frames_dir)

            report = compact_response_frame_ledger(
                frames_dir=frames_dir,
                execute=True,
                writers_stopped=True,
                backup_dir=backup_dir,
            )

            self.assertEqual(_tree_fingerprint(frames_dir), before)
            self.assertFalse((frames_dir / 'snapshots').exists())
            self.assertEqual(list(backup_dir.iterdir()), [])

        self.assertFalse(report['ok'])
        self.assertFalse(report['changed'])
        self.assertEqual(
            report['error']['code'],
            'response_frame_backup_directory_exists',
        )

    def test_failure_after_cas_write_reports_and_preserves_new_cas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'response_frames'
            ledger_path = _write_ledger(
                frames_dir,
                [_frame('resp-cas-failure', ghost_preview=_large_ghost_preview())],
            )
            index_path = _write_exact_source_index(frames_dir)
            backup_dir = root / 'verified-backup'
            source_ledger = ledger_path.read_bytes()
            source_index = index_path.read_bytes()

            with mock.patch(
                'ollmo_services.response_frame_ledger_maintenance._atomic_install',
                side_effect=RuntimeError('injected_atomic_install_failure'),
            ):
                report = compact_response_frame_ledger(
                    frames_dir=frames_dir,
                    execute=True,
                    writers_stopped=True,
                    backup_dir=backup_dir,
                )

            cas_files = sorted(
                (frames_dir / 'snapshots' / 'content_sha256').rglob('*.json')
            )

            self.assertEqual(ledger_path.read_bytes(), source_ledger)
            self.assertEqual(index_path.read_bytes(), source_index)
            self.assertEqual((backup_dir / ledger_path.name).read_bytes(), source_ledger)
            self.assertEqual((backup_dir / index_path.name).read_bytes(), source_index)

        self.assertFalse(report['ok'])
        self.assertTrue(report['changed'])
        self.assertTrue(report['backup_created'])
        self.assertFalse(report['ledger_replaced'])
        self.assertFalse(report['index_replaced'])
        self.assertEqual(report['new_cas_file_count'], 1)
        self.assertEqual(len(report['new_cas_files']), 1)
        self.assertTrue(report['new_cas_files_preserved'])
        self.assertEqual(len(cas_files), 1)
        self.assertEqual(report['error']['code'], 'injected_atomic_install_failure')


if __name__ == '__main__':
    unittest.main()
