import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path

from ollmo_server.response_lookup_runtime import ResponseLookupRuntimeOwner
from ollmo_server.responses_runtime import ResponsesRuntimeOwner
from ollmo_services.response_frames import inspect_response_frame_recovery_cache


class ResponseLookupRuntimeOwnerTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.frames_dir = Path(self._tmpdir.name) / 'response_frames'
        self.frames_dir.mkdir(parents=True)
        self.live_records = {}
        self.index_result = (
            None,
            {
                'ok': False,
                'status_code': 404,
                'error': {'code': 'response_frame_not_found'},
            },
        )
        self.recovery_result = (None, None, 404)
        self.index_calls = []
        self.recovery_calls = []
        self.fallback_calls = []
        self.checkpoints = []

        def get_live_record(response_id):
            record = self.live_records.get(response_id)
            return copy.deepcopy(record) if record else None

        def load_wire_payload(response_id):
            self.index_calls.append(response_id)
            return copy.deepcopy(self.index_result)

        def project_fallback(payload):
            self.fallback_calls.append(copy.deepcopy(dict(payload)))
            projected = copy.deepcopy(dict(payload))
            projection = dict(projected.get('wire_projection') or {})
            projection['source'] = 'test_in_memory_fallback'
            projected['wire_projection'] = projection
            return projected

        def recover(response_id):
            self.recovery_calls.append(response_id)
            return copy.deepcopy(self.recovery_result)

        def derive_lifecycle(payload, *, requested_status=None):
            return str(
                payload.get('lifecycle_state')
                or requested_status
                or payload.get('status')
                or 'in_progress'
            )

        def advance_checkpoint(response_id, ledger_stat):
            checkpoint = dict(ledger_stat)
            self.checkpoints.append((response_id, checkpoint))
            if response_id in self.live_records:
                self.live_records[response_id][
                    'response_frame_ledger_stat'
                ] = checkpoint

        self.owner = ResponseLookupRuntimeOwner(
            normalize_response_lookup_id=lambda value: str(value or '').strip(),
            get_live_response_lookup_record=get_live_record,
            load_wire_payload_from_index=load_wire_payload,
            project_fallback_payload=project_fallback,
            recover_response_lookup_record=recover,
            project_late_fill=lambda payload: copy.deepcopy(
                dict(payload.get('late_fill') or {})
            ),
            project_surface=lambda value: copy.deepcopy(dict(value or {})),
            derive_lifecycle_state=derive_lifecycle,
            response_payload_message_id=lambda payload: None,
            response_registry_now_iso=lambda: '2026-07-22T18:00:00Z',
            response_frames_dir_getter=lambda: self.frames_dir,
            inspect_recovered_cache=(
                lambda response_id, ledger_path, expected_state: (
                    inspect_response_frame_recovery_cache(
                        response_id,
                        frames_dir=self.owner.response_frames_dir_getter(),
                        expected_ledger_path=ledger_path,
                        expected_ledger_state=expected_state,
                    )
                )
            ),
            advance_recovered_cache_checkpoint=advance_checkpoint,
            response_lookup_ttl_sec=1800,
            now_ts=lambda: 1000.0,
            new_message_id=lambda: 'msg_test',
        )

    @staticmethod
    def _payload(
        response_id,
        *,
        frame_id='frame-1',
        frame_sequence=1,
        output_text='durable output',
        lifecycle_state='completed',
        **extra,
    ):
        return {
            'id': response_id,
            'status': 'completed',
            'lifecycle_state': lifecycle_state,
            'output_text': output_text,
            'response_frame': {
                'response_id': response_id,
                'frame_id': frame_id,
                'frame_sequence': frame_sequence,
            },
            **extra,
        }

    @staticmethod
    def _record(response_id, payload, **extra):
        return {
            'id': response_id,
            'message_id': 'msg_live',
            'status': payload.get('status', 'completed'),
            'lifecycle_state': payload.get('lifecycle_state', 'completed'),
            'response_payload': copy.deepcopy(payload),
            'expires_at_ts': 2000.0,
            **extra,
        }

    @staticmethod
    def _ledger_stat(path):
        stat = path.stat()
        return {
            'size_bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'device': stat.st_dev,
            'inode': stat.st_ino,
        }

    def test_verified_index_projection_builds_durable_wire_record(self):
        response_id = 'resp_index'
        durable = self._payload(response_id)
        self.index_result = (durable, {'ok': True})

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(error)
        self.assertEqual(status_code, 200)
        self.assertEqual(record['lookup_source'], 'response_frame_wire_projection')
        self.assertTrue(record['recovered_from_response_frame'])
        self.assertEqual(record['response_payload'], durable)
        self.assertEqual(record['message_id'], 'msg_test')
        self.assertEqual(record['expires_at_ts'], 2800.0)
        self.assertEqual(self.recovery_calls, [])

    def test_newer_live_frame_wins_without_copying_unbounded_record_body(self):
        response_id = 'resp_live_newer'
        durable = self._payload(response_id, frame_id='frame-1', frame_sequence=1)
        live = self._payload(
            response_id,
            frame_id='frame-2',
            frame_sequence=2,
            output_text='new live output',
        )
        self.index_result = (durable, {'ok': True})
        self.live_records[response_id] = self._record(
            response_id,
            live,
            bounded_response_payload={'must_not': 'replace projected body'},
        )

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(error)
        self.assertEqual(status_code, 200)
        self.assertEqual(record['lookup_source'], 'response_wire_live_newer')
        self.assertEqual(record['response_payload']['output_text'], 'new live output')
        self.assertNotIn('bounded_response_payload', record)
        self.assertEqual(len(self.fallback_calls), 1)

    def test_same_frame_overlay_changes_only_bounded_volatile_truth(self):
        response_id = 'resp_same_frame'
        durable = self._payload(
            response_id,
            outputs=[{'value': 'durable'}],
            artifacts=[{'artifact_ref': 'artifact:durable'}],
            runtime={'durable': True},
            late_fill={'status': 'pending'},
            surface_state={'status': 'pending'},
        )
        live = self._payload(
            response_id,
            lifecycle_state='late_fill_running',
            outputs=[{'value': 'must not overlay'}],
            artifacts=[{'artifact_ref': 'artifact:live'}],
            runtime={'live': True},
            late_fill={'status': 'running'},
            surface_state={'status': 'active'},
        )
        self.index_result = (durable, {'ok': True})
        self.live_records[response_id] = self._record(response_id, live)

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )
        payload = record['response_payload']

        self.assertIsNone(error)
        self.assertEqual(status_code, 200)
        self.assertEqual(
            record['lookup_source'],
            'response_wire_same_frame_live_overlay',
        )
        self.assertEqual(payload['outputs'], durable['outputs'])
        self.assertEqual(payload['artifacts'], durable['artifacts'])
        self.assertEqual(payload['runtime'], durable['runtime'])
        self.assertEqual(payload['late_fill']['status'], 'running')
        self.assertEqual(payload['surface_state']['status'], 'active')
        self.assertEqual(payload['lifecycle_state'], 'late_fill_running')
        self.assertEqual(
            payload['wire_projection']['live_state_overlay'],
            'same_frozen_frame_bounded_status_late_fill_surface_only',
        )

    def test_equal_sequence_with_different_frame_id_does_not_overlay(self):
        response_id = 'resp_frame_mismatch'
        durable = self._payload(
            response_id,
            frame_id='frame-durable',
            late_fill={'status': 'pending'},
        )
        live = self._payload(
            response_id,
            frame_id='frame-live',
            lifecycle_state='late_fill_running',
            late_fill={'status': 'running'},
        )
        self.index_result = (durable, {'ok': True})
        self.live_records[response_id] = self._record(response_id, live)

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(error)
        self.assertEqual(status_code, 200)
        self.assertEqual(record['lookup_source'], 'response_frame_wire_projection')
        self.assertEqual(
            record['response_payload']['late_fill']['status'],
            'pending',
        )
        self.assertNotIn('wire_projection', record['response_payload'])

    def test_live_only_fallback_uses_call_time_frames_directory(self):
        response_id = 'resp_live_only'
        live = self._payload(response_id)
        self.live_records[response_id] = self._record(response_id, live)
        alternate_root = Path(self._tmpdir.name) / 'alternate_frames'
        alternate_root.mkdir()
        selected_root = [self.frames_dir]
        self.owner.response_frames_dir_getter = lambda: selected_root[0]
        (self.frames_dir / 'responses.jsonl').write_text(
            json.dumps({'response_id': 'other'}) + '\n',
            encoding='utf-8',
        )
        self.index_result = (
            None,
            {
                'ok': False,
                'status_code': 409,
                'error': {'code': 'response_frame_index_unverified'},
            },
        )
        selected_root[0] = alternate_root

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(error)
        self.assertEqual(status_code, 200)
        self.assertEqual(record['lookup_source'], 'response_wire_live_only')
        self.assertEqual(self.recovery_calls, [])

    def test_exact_not_found_without_live_record_short_circuits_recovery(self):
        response_id = 'resp_missing'

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(record)
        self.assertEqual(error['code'], 'response_frame_not_found')
        self.assertEqual(status_code, 404)
        self.assertEqual(self.recovery_calls, [])

    def test_recovered_cache_reuses_unrelated_tail_and_recovers_target_tail(self):
        response_id = 'resp_cached'
        ledger_path = self.frames_dir / 'responses.jsonl'
        ledger_path.write_text(
            json.dumps({'response_id': 'other-initial'}) + '\n',
            encoding='utf-8',
        )
        cached = self._payload(response_id, frame_id='frame-cache', frame_sequence=2)
        self.live_records[response_id] = self._record(
            response_id,
            cached,
            recovered_from_response_frame=True,
            bounded_response_payload=copy.deepcopy(cached),
            response_frame_ledger_path=str(ledger_path),
            response_frame_ledger_stat=self._ledger_stat(ledger_path),
        )
        self.index_result = (
            None,
            {
                'ok': False,
                'status_code': 409,
                'error': {'code': 'response_frame_index_unverified'},
            },
        )

        first, first_error, first_status = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )
        with ledger_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'response_id': 'unrelated'}) + '\n')
        second, second_error, second_status = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(first_error)
        self.assertEqual(first_status, 200)
        self.assertEqual(
            first['lookup_source'],
            'response_wire_cached_ledger_recovery',
        )
        self.assertIsNone(second_error)
        self.assertEqual(second_status, 200)
        self.assertEqual(
            second['lookup_source'],
            'response_wire_cached_ledger_recovery',
        )
        self.assertEqual(len(self.checkpoints), 1)
        self.assertEqual(self.recovery_calls, [])

        with ledger_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'response_id': response_id}) + '\n')
        recovered = self._payload(
            response_id,
            frame_id='frame-recovered',
            frame_sequence=3,
            output_text='recovered target tail',
        )
        self.recovery_result = (
            {'id': response_id, 'response_payload': recovered},
            None,
            200,
        )

        third, third_error, third_status = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(third_error)
        self.assertEqual(third_status, 200)
        self.assertEqual(
            third['lookup_source'],
            'response_wire_exceptional_ledger_recovery',
        )
        self.assertEqual(
            third['response_payload']['output_text'],
            'recovered target tail',
        )
        self.assertEqual(self.recovery_calls, [response_id])

    def test_recovered_cache_rejects_malformed_appended_line(self):
        response_id = 'resp_malformed_tail'
        ledger_path = self.frames_dir / 'responses.jsonl'
        ledger_path.write_text(
            json.dumps({'response_id': 'initial'}) + '\n',
            encoding='utf-8',
        )
        record = {
            'response_frame_ledger_path': str(ledger_path),
            'response_frame_ledger_stat': self._ledger_stat(ledger_path),
        }
        with ledger_path.open('a', encoding='utf-8') as handle:
            handle.write('{not-json}\n')

        self.assertFalse(
            self.owner.response_frame_recovered_cache_valid(
                record,
                response_id=response_id,
            )
        )
        self.assertEqual(self.checkpoints, [])

    def test_recovered_cache_consumes_inspection_and_advances_checkpoint(self):
        response_id = 'resp_injected_cache_inspection'
        record = {
            'response_frame_ledger_path': '/not/read/by/lookup/owner.jsonl',
            'response_frame_ledger_stat': {
                'size_bytes': 10,
                'mtime_ns': 20,
                'device': 30,
                'inode': 40,
            },
        }
        inspections = []

        def inspect_cache(target_id, ledger_path, expected_state):
            inspections.append((target_id, ledger_path, dict(expected_state)))
            return {
                'cache_reusable': True,
                'checkpoint_ledger_state': {
                    'size_bytes': 50,
                    'mtime_ns': 60,
                    'device': 70,
                    'inode': 80,
                },
            }

        self.owner.inspect_recovered_cache = inspect_cache

        valid = self.owner.response_frame_recovered_cache_valid(
            record,
            response_id=response_id,
        )

        self.assertTrue(valid)
        self.assertEqual(
            inspections,
            [
                (
                    response_id,
                    '/not/read/by/lookup/owner.jsonl',
                    {
                        'size_bytes': 10,
                        'mtime_ns': 20,
                        'device': 30,
                        'inode': 40,
                    },
                )
            ],
        )
        checkpoint = {
            'size_bytes': 50,
            'mtime_ns': 60,
            'device': 70,
            'inode': 80,
        }
        self.assertEqual(record['response_frame_ledger_stat'], checkpoint)
        self.assertEqual(self.checkpoints, [(response_id, checkpoint)])

    def test_recovery_error_and_status_pass_through(self):
        response_id = 'resp_recovery_error'
        (self.frames_dir / 'responses.jsonl').write_text(
            '{not-json}\n',
            encoding='utf-8',
        )
        self.index_result = (
            None,
            {
                'ok': False,
                'status_code': 409,
                'error': {'code': 'response_frame_index_unverified'},
            },
        )
        self.recovery_result = (
            None,
            {'code': 'response_frame_ledger_corrupt', 'message': 'corrupt'},
            409,
        )

        record, error, status_code = (
            self.owner.get_bounded_response_lookup_record(response_id)
        )

        self.assertIsNone(record)
        self.assertEqual(error['code'], 'response_frame_ledger_corrupt')
        self.assertEqual(status_code, 409)
        self.assertEqual(self.recovery_calls, [response_id])


class ResponsesRuntimeRecoveryCheckpointTests(unittest.TestCase):
    def test_advance_response_recovery_checkpoint_updates_only_ledger_stat(self):
        response_id = 'resp_checkpoint'
        lookup = {
            response_id: {
                'id': response_id,
                'status': 'completed',
                'expires_at_ts': 1234.0,
                'response_payload': {'id': response_id},
            }
        }
        owner = ResponsesRuntimeOwner(
            response_lookup=lookup,
            response_lookup_lock=threading.Lock(),
            response_streams={},
            response_streams_lock=threading.Lock(),
            response_late_fill_in_flight=set(),
            response_late_fill_lock=threading.Lock(),
            response_lookup_ttl_sec=1800,
            normalize_response_lookup_id=lambda value: str(value or '').strip(),
            response_registry_now_iso=lambda: '2026-07-22T18:00:00Z',
        )
        checkpoint = {
            'size_bytes': 200,
            'mtime_ns': 300,
            'device': 400,
            'inode': 500,
        }

        advanced = owner.advance_response_recovery_checkpoint(
            response_id,
            checkpoint,
        )

        self.assertTrue(advanced)
        self.assertEqual(
            lookup[response_id]['response_frame_ledger_stat'],
            checkpoint,
        )
        self.assertEqual(lookup[response_id]['expires_at_ts'], 1234.0)
        self.assertEqual(
            lookup[response_id]['response_payload'],
            {'id': response_id},
        )


if __name__ == '__main__':
    unittest.main()
