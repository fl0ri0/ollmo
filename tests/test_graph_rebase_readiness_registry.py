import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from ollmo_services.graph_rebase_readiness_registry import (
    GraphRebaseReadinessRegistryError,
    append_graph_rebase_readiness_observation,
    append_graph_rebase_readiness_registry_records,
    load_graph_rebase_readiness_observations,
    load_graph_rebase_readiness_records,
    load_graph_rebase_readiness_registry,
    sync_graph_rebase_readiness_epoch,
)
from ollmo_services.response_frames import (
    load_latest_response_observation_state,
    load_latest_response_state,
    load_response_frame_index,
    persist_response_frame,
    verify_response_frame_epoch,
)
from ollmo_services.graph_rebase_rollout import (
    project_graph_rebase_readiness_observation,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / 'scripts' / 'sync_graph_rebase_readiness_registry.py'


class GraphRebaseReadinessRegistryTests(unittest.TestCase):
    @staticmethod
    def _frame(
        response_id: str,
        *,
        relevant: bool = True,
        lifecycle_state: str = 'completed',
    ) -> dict:
        frame = {
            'kind': 'ollmo.response_frame',
            'frame_version': 9,
            'response_id': response_id,
            'status': 'completed',
            'current_state': {
                'id': response_id,
                'status': 'completed',
                'lifecycle_state': lifecycle_state,
            },
            'request': {
                'prompt': f'bounded graph-rebase workload {response_id}',
                'workload_family': f'family-{response_id}',
            },
        }
        if relevant:
            frame['runtime'] = {
                'developer_diagnostics': {
                    'response_time_graph_rebase_candidate': {
                        'candidate': True,
                        'reason': 'additive_repair_insufficient',
                    },
                },
            }
        if lifecycle_state == 'late_fill_running':
            frame['late_fill'] = {
                'status': 'running',
                'active_count': 1,
                'pending_count': 0,
            }
        return frame

    def test_relocated_epoch_verifies_and_rebinds_only_in_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = root / 'active' / 'state' / 'response_frames'
            archived = root / 'archive' / 'state' / 'response_frames'
            persist_response_frame(self._frame('resp_relocated'), frames_dir=original)
            index_before = (original / 'current_index.json').read_bytes()
            ledger_before = (original / 'responses.jsonl').read_bytes()
            archived.parent.mkdir(parents=True)
            shutil.move(str(original), str(archived))

            strict = verify_response_frame_epoch(
                frames_dir=archived,
                allow_relocated=False,
            )
            verified = verify_response_frame_epoch(
                frames_dir=archived,
                allow_relocated=True,
            )

            self.assertFalse(strict['ok'])
            self.assertEqual(strict['error']['code'], 'response_frame_index_ledger_mismatch')
            self.assertTrue(verified['ok'])
            self.assertTrue(verified['relocated'])
            self.assertEqual(verified['ledger_line_count'], 1)
            self.assertEqual(verified['response_map_entry_count'], 1)
            self.assertEqual(
                verified['index_state']['responses']['resp_relocated']['ledger_path'],
                str(archived / 'responses.jsonl'),
            )
            self.assertEqual((archived / 'current_index.json').read_bytes(), index_before)
            self.assertEqual((archived / 'responses.jsonl').read_bytes(), ledger_before)

    def test_sync_default_is_read_only_and_write_is_idempotent_after_growth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(self._frame('resp_candidate'), frames_dir=frames_dir)

            checked = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
            )
            self.assertTrue(checked['ok'])
            self.assertEqual(checked['status'], 'verified')
            self.assertEqual(checked['settled_observation_count'], 1)
            self.assertEqual(checked['missing_settled_observation_count'], 1)
            self.assertFalse(registry_path.exists())

            written = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
            )
            registry_before = registry_path.read_bytes()
            self.assertEqual(written['appended_record_count'], 1)
            self.assertEqual(written['registered_observation_count'], 1)
            self.assertEqual(written['missing_settled_observation_count'], 0)

            persist_response_frame(
                self._frame('resp_unrelated', relevant=False),
                frames_dir=frames_dir,
            )
            repeated = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
            )
            self.assertTrue(repeated['ok'])
            self.assertEqual(repeated['status'], 'unchanged')
            self.assertEqual(repeated['appended_record_count'], 0)
            self.assertEqual(repeated['already_present_count'], 1)
            self.assertEqual(registry_path.read_bytes(), registry_before)

            records = load_graph_rebase_readiness_records(registry_path)
            observations = load_graph_rebase_readiness_observations(registry_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0]['response_id'], 'resp_candidate')

    def test_registry_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(self._frame('resp_tampered'), frames_dir=frames_dir)
            result = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
            )
            self.assertTrue(result['ok'])
            record = json.loads(registry_path.read_text(encoding='utf-8'))
            record['observation']['workload_family'] = 'tampered-family'
            registry_path.write_text(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                + '\n',
                encoding='utf-8',
            )

            loaded = load_graph_rebase_readiness_registry(registry_path)
            self.assertFalse(loaded['ok'])
            self.assertEqual(
                loaded['error']['code'],
                'readiness_registry_observation_digest_mismatch',
            )
            with self.assertRaises(GraphRebaseReadinessRegistryError) as raised:
                load_graph_rebase_readiness_observations(registry_path)
            self.assertEqual(raised.exception.status_code, 409)

    def test_single_append_rejects_well_formed_but_wrong_source_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(
                self._frame('resp-wrong-source-digest'),
                frames_dir=frames_dir,
            )
            checked = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
            )
            self.assertTrue(checked['ok'])
            observed = load_latest_response_observation_state(
                'resp-wrong-source-digest',
                frames_dir=frames_dir,
            )
            self.assertTrue(observed['ok'])
            projection = project_graph_rebase_readiness_observation(
                observed['response_payload']
            )

            with self.assertRaises(GraphRebaseReadinessRegistryError) as raised:
                append_graph_rebase_readiness_observation(
                    projection,
                    source_frame='0' * 64,
                    source_epoch=checked['source_epoch'],
                    frames_dir=frames_dir,
                    registry_path=registry_path,
                )

            self.assertEqual(
                raised.exception.code,
                'readiness_registry_source_frame_digest_mismatch',
            )
            self.assertFalse(registry_path.exists())

    def test_single_append_canonicalizes_raw_payload_like_later_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            response_id = 'resp-raw-canonicalized'
            oversized = 'raw-proposal-payload-' * 1500
            frame = self._frame(response_id)
            frame['runtime']['request_phase_graph'] = {
                'kind': 'ollmo.request_phase_graph',
                'graph_rebase_proposals': [
                    {
                        'kind': 'ollmo.graph_rebase_proposal',
                        'proposal_id': 'proposal-raw-canonicalized',
                        'payload': oversized,
                    }
                ],
            }
            persist_response_frame(frame, frames_dir=frames_dir)
            full_state = load_latest_response_state(
                response_id,
                frames_dir=frames_dir,
            )
            verified = verify_response_frame_epoch(frames_dir=frames_dir)
            self.assertTrue(full_state['ok'])
            self.assertTrue(verified['ok'])

            appended = append_graph_rebase_readiness_observation(
                full_state['response_payload'],
                source_frame=verified['source_frame_sha256_by_response'][response_id],
                source_epoch=None,
                frames_dir=frames_dir,
                registry_path=registry_path,
            )
            repeated = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
            )

            self.assertTrue(appended['ok'])
            self.assertEqual(appended['appended_record_count'], 1)
            self.assertTrue(repeated['ok'])
            self.assertEqual(repeated['status'], 'unchanged')
            self.assertEqual(repeated['appended_record_count'], 0)
            loaded = load_graph_rebase_readiness_registry(registry_path)
            self.assertEqual(loaded['record_count'], 1)
            serialized = json.dumps(loaded['observations'][0], sort_keys=True)
            self.assertLess(len(serialized), 5_000)
            self.assertNotIn(oversized, serialized)

    def test_concurrent_exact_replay_appends_one_physical_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            seed_registry = root / 'seed.jsonl'
            registry_path = root / 'concurrent.jsonl'
            persist_response_frame(self._frame('resp_concurrent'), frames_dir=frames_dir)
            seeded = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=seed_registry,
                write=True,
            )
            self.assertTrue(seeded['ok'])
            record = load_graph_rebase_readiness_records(seed_registry)[0]

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(
                    lambda _index: append_graph_rebase_readiness_registry_records(
                        [record],
                        registry_path=registry_path,
                    ),
                    range(16),
                ))

            self.assertTrue(all(result['ok'] for result in results))
            self.assertEqual(
                sum(result['appended_record_count'] for result in results),
                1,
            )
            loaded = load_graph_rebase_readiness_registry(registry_path)
            self.assertTrue(loaded['ok'])
            self.assertEqual(loaded['record_count'], 1)
            self.assertEqual(loaded['physical_record_count'], 1)

    def test_require_all_settled_rejects_active_observation_without_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(
                self._frame(
                    'resp_active',
                    lifecycle_state='late_fill_running',
                ),
                frames_dir=frames_dir,
            )

            result = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
                require_all_settled=True,
            )

            self.assertNotIn('ok', result)
            self.assertEqual(result['active_observation_count'], 1)
            self.assertEqual(result['settled_observation_count'], 0)
            self.assertIn(
                'readiness_epoch_not_fully_settled',
                {error['code'] for error in result['errors']},
            )
            self.assertFalse(registry_path.exists())

    def test_missing_snapshot_is_reported_as_scan_error_and_never_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(self._frame('resp_missing_snapshot'), frames_dir=frames_dir)
            index = load_response_frame_index(frames_dir=frames_dir)
            runtime_ref = index['responses']['resp_missing_snapshot'][
                'effective_snapshot_manifest'
            ]['runtime']
            (frames_dir / runtime_ref['path']).unlink()

            result = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
            )

            self.assertNotIn('ok', result)
            self.assertGreaterEqual(result['scan_error_count'], 1)
            self.assertGreaterEqual(result['error_count'], 1)
            self.assertFalse(registry_path.exists())

    def test_expected_epoch_bindings_are_checked_before_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(self._frame('resp_expected'), frames_dir=frames_dir)
            index_sha256 = hashlib.sha256(
                (frames_dir / 'current_index.json').read_bytes()
            ).hexdigest()
            ledger_sha256 = hashlib.sha256(
                (frames_dir / 'responses.jsonl').read_bytes()
            ).hexdigest()

            accepted = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                expected_index_sha256=index_sha256,
                expected_ledger_sha256=ledger_sha256,
                expected_ledger_line_count=1,
                expected_response_count=1,
                expected_selected_count=1,
                expected_settled_count=1,
                expected_active_count=0,
            )
            rejected = sync_graph_rebase_readiness_epoch(
                frames_dir=frames_dir,
                registry_path=registry_path,
                write=True,
                expected_ledger_sha256='0' * 64,
            )

            self.assertTrue(accepted['ok'])
            self.assertNotIn('ok', rejected)
            self.assertIn(
                'readiness_epoch_expectation_mismatch',
                {error['code'] for error in rejected['errors']},
            )
            self.assertFalse(registry_path.exists())

    def test_cli_shell_summary_has_cleanup_preflight_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames_dir = root / 'frames'
            registry_path = root / 'readiness.jsonl'
            persist_response_frame(self._frame('resp_cli'), frames_dir=frames_dir)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SYNC_SCRIPT),
                    '--response-frames-dir',
                    str(frames_dir),
                    '--registry',
                    str(registry_path),
                    '--check-only',
                    '--require-all-settled',
                    '--shell-summary',
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = dict(
                line.split('=', 1)
                for line in completed.stdout.splitlines()
                if '=' in line
            )
            for key in (
                'status',
                'settled_observation_count',
                'registered_observation_count',
                'missing_settled_observation_count',
                'active_observation_count',
                'scan_error_count',
                'hydration_error_count',
                'registry_error_count',
            ):
                self.assertIn(key, summary)
            self.assertEqual(summary['status'], 'verified')
            self.assertEqual(summary['settled_observation_count'], '1')
            self.assertEqual(summary['registered_observation_count'], '0')
            self.assertEqual(summary['missing_settled_observation_count'], '1')
            self.assertFalse(registry_path.exists())


if __name__ == '__main__':
    unittest.main()
