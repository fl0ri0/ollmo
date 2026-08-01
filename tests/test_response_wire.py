import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import ollmo_webserver as webserver
import ollmo_services.response_frames as response_frames
from ollmo_services import response_wire


class ResponseWireTests(unittest.TestCase):
    @staticmethod
    def _cyclic_in_memory_payload():
        runtime = {}
        runtime['self'] = runtime
        return {
            'id': 'resp_wire_composite_parity',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'output_text': 'Bounded response output.',
            'artifacts': [{'artifact_ref': 'artifact:parity', 'type': 'text'}],
            'outputs': [
                {
                    'slot_id': 'slot-parity',
                    'type': 'text',
                    'status': 'fulfilled',
                    'value': 'Bounded response output.',
                }
            ],
            'output': [
                {
                    'id': 'message-parity',
                    'type': 'message',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [
                        {'type': 'output_text', 'text': 'Bounded response output.'}
                    ],
                }
            ],
            'late_fill': {
                'status': 'completed',
                'pending_branches': [],
                'active_branches': [],
                'completed_branches': [],
                'failed_branches': [],
                'cancelled_branches': [],
            },
            'response_frame': {
                'frame_id': 'frame-wire-composite-parity',
                'frame_sequence': 1,
                'response_id': 'resp_wire_composite_parity',
            },
            'runtime': runtime,
        }

    def test_profiles_preserve_distinct_indexed_and_in_memory_limits(self):
        self.assertEqual(response_wire.RESPONSE_WIRE_INLINE_LIMIT_BYTES, 8 * 1024 * 1024)
        self.assertEqual(
            response_wire.RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
            (8 * 1024 * 1024) - 1024,
        )
        self.assertEqual(
            response_wire.IN_MEMORY_RESPONSE_WIRE_PROFILE.text_preview_chars,
            32 * 1024,
        )
        self.assertEqual(
            response_wire.INDEXED_RESPONSE_WIRE_PROFILE.text_preview_chars,
            4096,
        )
        self.assertEqual(
            response_wire.INDEXED_RESPONSE_WIRE_PROFILE.media_preview_chars,
            256,
        )
        self.assertIsNone(
            response_wire.INDEXED_RESPONSE_WIRE_PROFILE.collection_limit
        )

    def test_in_memory_digest_ref_preserves_current_compact_json_identity(self):
        value = {'unicode': 'Grüezi', 'empty': '', 'items': [1, 2]}
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        ).encode('utf-8')

        ref = response_wire.digest_ref(value, json_path='response_payload')

        self.assertEqual(ref['sha256'], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(ref['size_bytes'], len(encoded))
        self.assertEqual(ref['storage'], 'digest_only')
        self.assertEqual(ref['authority'], 'audit_identity_only')

    def test_in_memory_text_preview_uses_profile_without_changing_ref_shape(self):
        profile = replace(
            response_wire.IN_MEMORY_RESPONSE_WIRE_PROFILE,
            text_preview_chars=8,
        )

        preview, ref = response_wire.text_preview(
            'abcdefghijk',
            json_path='output_text',
            profile=profile,
        )

        self.assertEqual(preview, 'abcdefgh')
        self.assertEqual(ref['length_chars'], 11)
        self.assertEqual(ref['preview_chars'], 8)
        self.assertTrue(ref['preview_truncated'])

    def test_indexed_identity_keeps_legacy_json_safe_normalization(self):
        value = {
            'kept': 'value',
            'empty': '',
            'response_frame': {'frame_id': 'excluded'},
            'image_data_url': 'excluded',
            'path': Path('/tmp/example'),
        }

        encoded, sha256 = response_wire.indexed_json_identity(value)

        self.assertEqual(encoded, b'{"kept":"value","path":"/tmp/example"}')
        self.assertEqual(sha256, hashlib.sha256(encoded).hexdigest())

    def test_indexed_identity_unicode_sorting_preserves_legacy_size(self):
        value = {
            'z-last': 'Grüezi',
            'a-first': '東京',
            'empty': '',
        }
        expected = json.dumps(
            {'a-first': '東京', 'z-last': 'Grüezi'},
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        legacy_insertion_order = json.dumps(
            {'z-last': 'Grüezi', 'a-first': '東京'},
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8')

        encoded, sha256 = response_wire.indexed_json_identity(value)

        self.assertEqual(encoded, expected)
        self.assertEqual(sha256, hashlib.sha256(expected).hexdigest())
        self.assertEqual(len(encoded), len(legacy_insertion_order))

    def test_indexed_batch_record_handle_preserves_golden_contract(self):
        content = 'Grüezi 東京'
        result_text = 'résultat final'
        artifact_ref = 'artifact:' + ('x' * 520)
        item = {
            'content': content,
            'prompt': 'must remain omitted',
            'artifact_ref': artifact_ref,
            'nested_ref': {
                'path': Path('/tmp/Grüezi.txt'),
                'status': 'ready',
            },
            'status': 'completed',
            'result_text': result_text,
        }
        wrapper_stats = {}
        service_stats = {}

        wrapper = response_frames._response_wire_batch_item_handle(
            item,
            stats=wrapper_stats,
        )
        service = response_wire.indexed_record_handle(
            item,
            stats=service_stats,
            batch_item=True,
        )

        expected = {
            'artifact_ref': artifact_ref[:512],
            'artifact_ref_length_chars': len(artifact_ref),
            'artifact_ref_size_bytes': len(artifact_ref.encode('utf-8')),
            'artifact_ref_sha256': hashlib.sha256(
                artifact_ref.encode('utf-8')
            ).hexdigest(),
            'artifact_ref_preview_truncated': True,
            'nested_ref': {
                'path': '/tmp/Grüezi.txt',
                'status': 'ready',
            },
            'status': 'completed',
            'content_length_chars': len(content),
            'content_sha256': hashlib.sha256(
                content.encode('utf-8')
            ).hexdigest(),
            'result_text_length_chars': len(result_text),
            'result_text_sha256': hashlib.sha256(
                result_text.encode('utf-8')
            ).hexdigest(),
        }
        self.assertEqual(wrapper, expected)
        self.assertEqual(service, expected)
        self.assertEqual(list(wrapper), list(expected))
        self.assertEqual(wrapper_stats, {'truncated_text_value_count': 1})
        self.assertEqual(service_stats, wrapper_stats)
        self.assertNotIn('prompt', wrapper)
        self.assertNotIn('wire_body_sha256', wrapper)

        record_stats = {}
        record = response_wire.indexed_record_handle(item, stats=record_stats)
        encoded, sha256 = response_wire.indexed_json_identity(item)
        self.assertEqual(record['wire_body_size_bytes'], len(encoded))
        self.assertEqual(record['wire_body_sha256'], sha256)
        self.assertEqual(
            record_stats,
            {
                'truncated_text_value_count': 1,
                'truncated_record_count': 1,
            },
        )

    def test_response_frame_private_wrappers_match_pure_indexed_projection(self):
        long_prefix = 'shared-prefix-' + ('x' * 512)
        source = {
            f'{long_prefix}-alpha': {'status': 'completed', 'value': 'first'},
            f'{long_prefix}-bravo': {'status': 'failed', 'value': 'second'},
        }
        wrapper_stats = {}
        service_stats = {}

        wrapper_projection = response_frames._response_wire_bounded_value(
            source,
            stats=wrapper_stats,
        )
        service_projection = response_wire.indexed_bounded_value(
            source,
            stats=service_stats,
        )

        self.assertEqual(wrapper_projection, service_projection)
        self.assertEqual(wrapper_stats, service_stats)
        self.assertEqual(wrapper_stats['bounded_mapping_key_count'], 2)

    def test_indexed_late_fill_preserves_explicit_settled_branch_arrays(self):
        projected = response_wire.indexed_late_fill_projection(
            {
                'status': 'completed',
                'pending_branches': [],
                'active_branches': [],
                'completed_branches': [],
                'failed_branches': [],
                'cancelled_branches': [],
            }
        )

        for key in (
            'pending_branches',
            'active_branches',
            'completed_branches',
            'failed_branches',
            'cancelled_branches',
        ):
            self.assertEqual(projected[key], [])
        self.assertEqual(projected['pending_count'], 0)
        self.assertEqual(projected['active_count'], 0)
        self.assertEqual(projected['completed_count'], 0)
        self.assertEqual(projected['failed_count'], 0)
        self.assertEqual(projected['cancelled_count'], 0)

    def test_indexed_snapshot_manifest_keeps_every_sorted_reference(self):
        source = {
            'z.path': {'sha256': 'z' * 64, 'path': 'z.json'},
            'a.path': {'sha256': 'a' * 64, 'path': 'a.json'},
        }

        projected, metadata = response_wire.indexed_snapshot_manifest_projection(
            source
        )

        self.assertEqual(list(projected), ['a.path', 'z.path'])
        self.assertEqual(metadata['effective_snapshot_count'], 2)
        self.assertEqual(metadata['projected_snapshot_ref_count'], 2)
        self.assertTrue(metadata['manifest_projection_complete'])

    def test_web_emergency_projection_shim_matches_pure_service(self):
        payload = self._cyclic_in_memory_payload()

        shim = webserver._response_wire_emergency_projection(
            payload,
            source='test_composite_parity',
            limit_bytes=4096,
        )
        service = response_wire.emergency_projection(
            payload,
            source='test_composite_parity',
            limit_bytes=4096,
            output_message_projector=(
                lambda value: webserver._response_lookup_output_message_for_ui(value)
            ),
        )

        self.assertEqual(shim, service)

    def test_web_fallback_projection_shim_matches_pure_service(self):
        payload = self._cyclic_in_memory_payload()

        shim = webserver._response_wire_fallback_payload(payload)
        service = response_wire.fallback_payload(
            payload,
            artifact_projector=(
                lambda value: webserver._response_lookup_artifact_for_ui(value)
            ),
            output_projector=(
                lambda value: webserver._response_lookup_output_for_ui(value)
            ),
            output_message_projector=(
                lambda value: webserver._response_lookup_output_message_for_ui(value)
            ),
            late_fill_projector=(
                lambda value: webserver._response_lookup_late_fill_for_ui(value)
            ),
            emergency_projector=(
                lambda value, **kwargs: webserver._response_wire_emergency_projection(
                    value,
                    **kwargs,
                )
            ),
        )

        self.assertEqual(shim, service)

    def test_web_byte_ceiling_shim_matches_pure_service(self):
        payload = self._cyclic_in_memory_payload()

        shim = webserver._response_wire_enforce_byte_ceiling(
            payload,
            source_payload=payload,
            source='test_composite_byte_ceiling_parity',
        )
        service = response_wire.enforce_byte_ceiling(
            payload,
            source_payload=payload,
            source='test_composite_byte_ceiling_parity',
            emergency_projector=(
                lambda value, **kwargs: webserver._response_wire_emergency_projection(
                    value,
                    **kwargs,
                )
            ),
        )

        self.assertEqual(shim, service)

    def test_web_fallback_ui_adapters_are_resolved_at_call_time(self):
        payload = self._cyclic_in_memory_payload()

        with (
            patch.object(
                webserver,
                '_response_lookup_artifact_for_ui',
                return_value={'adapter': 'artifact'},
            ) as artifact_projector,
            patch.object(
                webserver,
                '_response_lookup_output_for_ui',
                return_value={'adapter': 'output'},
            ) as output_projector,
            patch.object(
                webserver,
                '_response_lookup_output_message_for_ui',
                return_value={'adapter': 'message'},
            ) as message_projector,
            patch.object(
                webserver,
                '_response_lookup_late_fill_for_ui',
                return_value={'adapter': 'late_fill'},
            ) as late_fill_projector,
        ):
            projected = webserver._response_wire_fallback_payload(payload)

        self.assertEqual(projected['artifacts'], [{'adapter': 'artifact'}])
        self.assertEqual(projected['outputs'], [{'adapter': 'output'}])
        self.assertEqual(projected['output'], [{'adapter': 'message'}])
        self.assertEqual(projected['late_fill'], {'adapter': 'late_fill'})
        artifact_projector.assert_called_once()
        output_projector.assert_called_once()
        message_projector.assert_called_once()
        late_fill_projector.assert_called_once()

    def test_web_byte_ceiling_resolves_emergency_shim_at_call_time(self):
        payload = self._cyclic_in_memory_payload()
        expected = {
            'id': payload['id'],
            'wire_projection': {'source': 'patched-emergency-projector'},
        }

        with patch.object(
            webserver,
            '_response_wire_emergency_projection',
            return_value=expected,
        ) as emergency_projector:
            projected = webserver._response_wire_enforce_byte_ceiling(
                payload,
                source_payload=payload,
                source='test_late_bound_emergency_projector',
            )

        self.assertEqual(projected, expected)
        emergency_projector.assert_called_once_with(
            payload,
            source='test_late_bound_emergency_projector',
        )


if __name__ == '__main__':
    unittest.main()
