import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ollmo_services.response_frames import (
    RESPONSE_FRAME_STALE_PARENT_REASON,
    ResponseFrameParentCASMismatch,
    append_response_frame_with_parent_cas,
    load_response_frame_index,
    persist_response_frame,
)


class ResponseFrameParentCASTests(unittest.TestCase):
    @staticmethod
    def _frame(response_id, writer):
        return {
            'kind': 'ollmo.response_frame',
            'frame_version': 9,
            'response_id': response_id,
            'current_state': {'writer': writer},
        }

    def test_expected_parent_sequence_mismatch_fails_closed_without_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = 'resp-parent-cas-sequence'
            ledger_path = persist_response_frame(
                self._frame(response_id, 'initial'),
                frames_dir=frames_dir,
            )
            parent = load_response_frame_index(frames_dir=frames_dir)['responses'][
                response_id
            ]

            with self.assertRaises(ResponseFrameParentCASMismatch) as raised:
                append_response_frame_with_parent_cas(
                    self._frame(response_id, 'stale-successor'),
                    expected_parent_frame_id=parent['latest_frame_id'],
                    expected_parent_frame_sequence=99,
                    frames_dir=frames_dir,
                )

            self.assertEqual(raised.exception.code, RESPONSE_FRAME_STALE_PARENT_REASON)
            self.assertEqual(
                raised.exception.current_parent_frame_id,
                parent['latest_frame_id'],
            )
            self.assertEqual(raised.exception.current_parent_frame_sequence, 1)
            self.assertEqual(len(ledger_path.read_text(encoding='utf-8').splitlines()), 1)
            current = load_response_frame_index(frames_dir=frames_dir)['responses'][
                response_id
            ]
            self.assertEqual(current['latest_frame_id'], parent['latest_frame_id'])
            self.assertEqual(current['latest_frame_sequence'], 1)

    def test_concurrent_successors_compare_and_append_under_one_service_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = 'resp-parent-cas-race'
            ledger_path = persist_response_frame(
                self._frame(response_id, 'initial'),
                frames_dir=frames_dir,
            )
            parent = load_response_frame_index(frames_dir=frames_dir)['responses'][
                response_id
            ]
            start = threading.Barrier(3)

            def append_contender(writer):
                start.wait()
                try:
                    appended = append_response_frame_with_parent_cas(
                        self._frame(response_id, writer),
                        expected_parent_frame_id=parent['latest_frame_id'],
                        expected_parent_frame_sequence=parent[
                            'latest_frame_sequence'
                        ],
                        frames_dir=frames_dir,
                    )
                    return ('appended', appended)
                except ResponseFrameParentCASMismatch as exc:
                    return ('stale', exc.as_dict())

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(append_contender, 'successor-a'),
                    executor.submit(append_contender, 'successor-b'),
                ]
                start.wait()
                results = [future.result(timeout=5) for future in futures]

            appended = [value for status, value in results if status == 'appended']
            stale = [value for status, value in results if status == 'stale']
            self.assertEqual(len(appended), 1)
            self.assertEqual(len(stale), 1)
            self.assertEqual(stale[0]['code'], RESPONSE_FRAME_STALE_PARENT_REASON)
            self.assertEqual(stale[0]['expected_parent_frame_id'], parent['latest_frame_id'])
            self.assertEqual(
                stale[0]['current_parent_frame_id'],
                appended[0]['response_frame']['frame_id'],
            )

            lines = [
                json.loads(line)
                for line in ledger_path.read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual(len(lines), 2)
            self.assertEqual([line['frame_sequence'] for line in lines], [1, 2])
            self.assertEqual(len({line['frame_id'] for line in lines}), 2)
            current = load_response_frame_index(frames_dir=frames_dir)['responses'][
                response_id
            ]
            self.assertEqual(current['latest_frame_sequence'], 2)
            self.assertEqual(
                current['latest_frame_id'],
                appended[0]['response_frame']['frame_id'],
            )

    def test_all_regular_appends_share_the_same_sequence_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            response_id = 'resp-shared-append-lock'
            ledger_path = persist_response_frame(
                self._frame(response_id, 'initial'),
                frames_dir=frames_dir,
            )
            worker_count = 8
            start = threading.Barrier(worker_count + 1)

            def append_regular(index):
                start.wait()
                persist_response_frame(
                    self._frame(response_id, f'regular-{index}'),
                    frames_dir=frames_dir,
                )

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(append_regular, index)
                    for index in range(worker_count)
                ]
                start.wait()
                for future in futures:
                    future.result(timeout=5)

            lines = [
                json.loads(line)
                for line in ledger_path.read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual(len(lines), worker_count + 1)
            self.assertEqual(
                [line['frame_sequence'] for line in lines],
                list(range(1, worker_count + 2)),
            )
            self.assertEqual(
                len({line['frame_id'] for line in lines}),
                worker_count + 1,
            )
            current = load_response_frame_index(frames_dir=frames_dir)['responses'][
                response_id
            ]
            self.assertEqual(current['latest_frame_sequence'], worker_count + 1)


if __name__ == '__main__':
    unittest.main()
