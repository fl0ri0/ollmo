import hashlib
import tempfile
import unittest
from pathlib import Path

from ollmo_services.artifact_registry import (
    build_artifact_registry_record,
    build_generated_image_artifact_registry_record,
    build_generated_image_provenance,
    build_input_artifact_registry_records,
    build_output_artifact_registry_records,
    find_artifact_registry_record,
    find_artifact_registry_record_by_artifact_ref,
    find_generated_image_provenance,
    find_generated_image_provenance_by_artifact_ref,
    merge_artifact_registry_records,
    persist_artifact_registry_enrichment,
    persist_artifact_registry_record,
    persist_generated_image_provenance,
    persist_input_artifact_registry_records,
    persist_output_artifact_registry_records,
)


class ArtifactRegistryTests(unittest.TestCase):
    def test_generated_image_provenance_round_trips_through_artifact_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'dream.png'
            image_path.write_bytes(b'png')
            ledger_path = Path(tmpdir) / 'artifact_registry.jsonl'

            provenance = build_generated_image_provenance(
                image_path=str(image_path),
                prompt_text='dream city at dawn',
                prompt_preview='dream city at dawn',
                instance_id='image-1',
                model='flux',
                backend='mlx',
                capability='image_generation',
                mode='image_generation',
                response_id='resp_1',
                width=1280,
                height=720,
                seed=42,
            )
            persist_generated_image_provenance(
                provenance,
                ledger_path=ledger_path,
            )

            registry_record = find_artifact_registry_record(
                str(image_path),
                ledger_path=ledger_path,
            )
            self.assertIsNotNone(registry_record)
            self.assertEqual(registry_record['artifact_ref'], provenance['output']['artifact_ref'])
            self.assertEqual(registry_record['metadata']['width'], 1280)
            self.assertEqual(registry_record['metadata']['response_model'], 'flux')

            by_path = find_generated_image_provenance(
                str(image_path),
                ledger_path=ledger_path,
            )
            by_ref = find_generated_image_provenance_by_artifact_ref(
                provenance['output']['artifact_ref'],
                ledger_path=ledger_path,
            )
            self.assertEqual(by_path['provenance_id'], provenance['provenance_id'])
            self.assertEqual(by_ref['request']['prompt_text'], 'dream city at dawn')

    def test_build_generated_image_artifact_registry_record_preserves_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provenance = {
                'kind': 'ollmo.generated_image_provenance',
                'provenance_id': 'generated_image_deadbeef',
                'created_at': '2026-04-12T00:00:00Z',
                'image_path': 'artifacts/images/out.png',
                'source': {
                    'response_id': 'resp_2',
                    'model': 'z-image',
                    'instance_id': 'image-2',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'mode': 'image_generation',
                },
                'request': {
                    'prompt_text': 'misty temple',
                    'prompt_preview': 'misty temple',
                    'width': 1024,
                    'height': 1024,
                    'seed': 7,
                },
                'output': {
                    'saved_image_path': 'artifacts/images/out.png',
                    'artifact_id': 'image_deadbeef',
                    'artifact_ref': 'artifact:image_deadbeef',
                    'seed': 7,
                },
                'derived_from': ['artifact:text_story'],
            }

            record = build_generated_image_artifact_registry_record(provenance)
            ledger_path = Path(tmpdir) / 'artifact_registry.jsonl'
            persist_artifact_registry_record(record, ledger_path=ledger_path)
            found = find_artifact_registry_record_by_artifact_ref(
                'artifact:image_deadbeef',
                ledger_path=ledger_path,
            )

            self.assertEqual(found['artifact_ref'], 'artifact:image_deadbeef')
            self.assertEqual(found['artifact']['derived_from'], ['artifact:text_story'])
            self.assertEqual(found['provenance']['provenance_id'], 'generated_image_deadbeef')
            self.assertEqual(found['metadata']['backend'], 'ollama')
            self.assertEqual(found['linked_response_ids'], ['resp_2'])

    def test_persist_artifact_registry_record_upserts_same_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / 'audio' / 'speech.wav'
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b'wav')
            ledger_path = root / 'artifact_registry.jsonl'

            first = build_artifact_registry_record(
                artifact={
                    'type': 'audio',
                    'path': str(artifact_path),
                    'artifact_ref': 'artifact:old-audio-ref',
                },
                roles=['output'],
                metadata={'speaker': 'alpha'},
                linked_response_ids=['resp_1'],
            )
            second = build_artifact_registry_record(
                artifact={
                    'type': 'audio',
                    'path': str(artifact_path),
                    'artifact_ref': 'artifact:new-audio-ref',
                },
                roles=['reference'],
                metadata={'duration_seconds': 4},
                linked_response_ids=['resp_2'],
            )
            persist_artifact_registry_record(first, ledger_path=ledger_path)
            persist_artifact_registry_record(second, ledger_path=ledger_path)

            lines = ledger_path.read_text(encoding='utf-8').splitlines()
            found = find_artifact_registry_record(str(artifact_path), ledger_path=ledger_path)
            found_by_old_ref = find_artifact_registry_record_by_artifact_ref(
                'artifact:old-audio-ref',
                ledger_path=ledger_path,
            )

        self.assertEqual(len(lines), 1)
        self.assertEqual(found['artifact_ref'], 'artifact:new-audio-ref')
        self.assertEqual(found['artifact']['path'], str(artifact_path))
        self.assertIn('artifact:old-audio-ref', found['artifact_alias_refs'])
        self.assertIn('artifact:new-audio-ref', found['artifact_alias_refs'])
        self.assertEqual(found_by_old_ref['artifact']['path'], str(artifact_path))
        self.assertEqual(found['roles'], ['output', 'reference'])
        self.assertEqual(found['linked_response_ids'], ['resp_1', 'resp_2'])
        self.assertEqual(found['metadata']['speaker'], 'alpha')
        self.assertEqual(found['metadata']['duration_seconds'], 4)

    def test_persist_artifact_registry_enrichment_survives_later_provenance_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'dream.png'
            image_path.write_bytes(b'png')
            ledger_path = Path(tmpdir) / 'artifact_registry.jsonl'

            enrichment_record = persist_artifact_registry_enrichment(
                artifact_path=str(image_path),
                artifact_type='image',
                enrichments={
                    'image_state': {
                        'summary': 'dream temple floating above the sea',
                        'subject': 'temple',
                    },
                    'image_state_enrichment': {
                        'status': 'completed',
                        'mode': 'background_analysis',
                    },
                },
                ledger_path=ledger_path,
            )

            self.assertIsNotNone(enrichment_record)
            self.assertEqual(enrichment_record['artifact']['path'], str(image_path))
            self.assertEqual(
                enrichment_record['enrichments']['image_state']['subject'],
                'temple',
            )

            provenance = build_generated_image_provenance(
                image_path=str(image_path),
                prompt_text='dream temple at sea',
                prompt_preview='dream temple at sea',
                instance_id='image-1',
                model='flux',
                backend='ollama',
                capability='image_generation',
                mode='image_generation',
                response_id='resp_1',
                width=1024,
                height=768,
                seed=42,
            )
            persist_generated_image_provenance(
                provenance,
                ledger_path=ledger_path,
            )

            found = find_artifact_registry_record(
                str(image_path),
                ledger_path=ledger_path,
            )
            self.assertIsNotNone(found)
            self.assertEqual(found['artifact_ref'], provenance['output']['artifact_ref'])
            self.assertEqual(found['provenance']['provenance_id'], provenance['provenance_id'])
            self.assertEqual(found['metadata']['width'], 1024)
            self.assertEqual(found['enrichments']['image_state']['subject'], 'temple')
            self.assertEqual(
                found['enrichments']['image_state_enrichment']['mode'],
                'background_analysis',
            )

    def test_output_artifact_registry_records_cover_text_audio_image_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            text_path = root / 'documents' / 'brief.md'
            audio_path = root / 'audio' / 'brief.wav'
            image_path = root / 'images' / 'brief.png'
            other_path = root / 'ocr' / 'scan.md'
            for path in (text_path, audio_path, image_path, other_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('artifact', encoding='utf-8')

            records = build_output_artifact_registry_records(
                {
                    'id': 'resp_all_outputs',
                    'conversation_id': 'conv-1',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'mode': 'chat',
                    'saved_text_path': str(text_path),
                    'saved_audio_path': str(audio_path),
                    'audio_mimetype': 'audio/wav',
                    'saved_image_path': str(image_path),
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(other_path),
                            'mime_type': 'text/markdown',
                        }
                    ],
                }
            )

        by_type = {record['type']: record for record in records}
        self.assertIn('text', by_type)
        self.assertIn('audio', by_type)
        self.assertIn('image', by_type)
        self.assertGreaterEqual(len([record for record in records if record['type'] == 'text']), 2)
        for record in records:
            self.assertEqual(record['provenance']['kind'], 'ollmo.output_artifact_provenance')
            self.assertEqual(record['linked_response_ids'], ['resp_all_outputs'])
            self.assertIn('artifact_ref', record)

    def test_saved_transcript_registry_identity_matches_canonical_response_identity(self):
        from ollmo_services.responses import build_canonical_response_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / 'transcripts' / 'speech.md'
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text('Ein Leuchtturm im Sturm.\n', encoding='utf-8')
            ledger_path = Path(tmpdir) / 'artifact_registry.jsonl'
            response_id = 'resp_transcript_identity'
            canonical_artifact = build_canonical_response_artifacts(
                {
                    'id': response_id,
                    'saved_text_path': str(transcript_path),
                }
            )[0]
            payload = {
                'id': response_id,
                'saved_text_path': str(transcript_path),
                'artifacts': [canonical_artifact],
                'late_fill': {
                    'fill_results': [
                        {
                            'branch_id': 'branch-speech_to_text-1',
                            'phase_id': 'phase-3',
                            'capability': 'speech_to_text',
                            'saved_text_path': str(transcript_path),
                            'artifact_id': canonical_artifact['artifact_id'],
                            'artifact_ref': canonical_artifact['artifact_ref'],
                            'ref': canonical_artifact['ref'],
                            'artifacts': [canonical_artifact],
                        }
                    ]
                },
            }

            records = build_output_artifact_registry_records(payload)
            persist_output_artifact_registry_records(payload, ledger_path=ledger_path)
            found = find_artifact_registry_record(
                str(transcript_path),
                ledger_path=ledger_path,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['artifact_ref'], canonical_artifact['artifact_ref'])
        self.assertEqual(found['artifact_ref'], canonical_artifact['artifact_ref'])
        self.assertEqual(found['artifact']['artifact_ref'], canonical_artifact['artifact_ref'])
        self.assertEqual(found['metadata']['branch_id'], 'branch-speech_to_text-1')

    def test_input_artifact_registry_records_cover_true_external_inputs(self):
        records = build_input_artifact_registry_records(
            [
                {
                    'type': 'image',
                    'kind': 'image',
                    'path': '/tmp/artifacts/inputs/image/request.png',
                    'source_path': '/Users/example/request.png',
                    'origin': 'upload',
                    'mime_type': 'image/png',
                }
            ],
            request_payload={
                'response_id': 'resp_input',
                'conversation_id': 'conv-input',
                'request_id': 'req-input',
                'capability': 'vision_analysis',
            },
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['roles'], ['input'])
        self.assertEqual(record['provenance']['kind'], 'ollmo.input_artifact_provenance')
        self.assertEqual(record['provenance']['input']['origin'], 'upload')
        self.assertEqual(record['linked_response_ids'], ['resp_input'])

    def test_persist_input_artifacts_makes_external_inputs_findable_by_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / 'inputs' / 'photo.png'
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(b'png')
            ledger_path = root / 'artifact_registry.jsonl'

            persisted = persist_input_artifact_registry_records(
                [
                    {
                        'type': 'image',
                        'path': str(input_path),
                        'origin': 'upload',
                        'source_path': '/Users/example/photo.png',
                    }
                ],
                request_payload={'response_id': 'resp_upload'},
                ledger_path=ledger_path,
            )
            found = find_artifact_registry_record(str(input_path), ledger_path=ledger_path)

        self.assertEqual(len(persisted), 1)
        self.assertEqual(found['roles'], ['input'])
        self.assertEqual(found['metadata']['origin'], 'upload')

    def test_persist_output_artifacts_makes_non_image_outputs_findable_by_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            text_path = root / 'documents' / 'report.html'
            audio_path = root / 'audio' / 'report.wav'
            text_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text('<h1>Report</h1>', encoding='utf-8')
            audio_path.write_bytes(b'wav')
            ledger_path = root / 'artifact_registry.jsonl'

            persisted = persist_output_artifact_registry_records(
                {
                    'id': 'resp_text_audio',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'saved_text_path': str(text_path),
                    'saved_audio_path': str(audio_path),
                    'audio_mimetype': 'audio/wav',
                },
                ledger_path=ledger_path,
            )
            persisted_again = persist_output_artifact_registry_records(
                {
                    'id': 'resp_text_audio',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'saved_text_path': str(text_path),
                    'saved_audio_path': str(audio_path),
                    'audio_mimetype': 'audio/wav',
                },
                ledger_path=ledger_path,
            )

            found_text = find_artifact_registry_record(str(text_path), ledger_path=ledger_path)
            found_audio = find_artifact_registry_record(str(audio_path), ledger_path=ledger_path)
            line_count = len(ledger_path.read_text(encoding='utf-8').splitlines())

        self.assertEqual(len(persisted), 2)
        self.assertEqual(len(persisted_again), 2)
        self.assertEqual(line_count, 2)
        self.assertEqual(found_text['type'], 'text')
        self.assertEqual(found_audio['type'], 'audio')
        self.assertEqual(found_audio['metadata']['mime_type'], 'audio/wav')

    def test_output_artifact_registry_refreshes_text_content_from_final_saved_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            text_path = root / 'documents' / 'index.html'
            text_path.parent.mkdir(parents=True, exist_ok=True)
            final_content = '<!doctype html><html><body><section>Done</section></body></html>\n'
            text_path.write_text(final_content, encoding='utf-8')
            ledger_path = root / 'artifact_registry.jsonl'

            persisted = persist_output_artifact_registry_records(
                {
                    'id': 'resp_final_text',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'artifacts': [
                        {
                            'type': 'text',
                            'path': str(text_path),
                            'mime_type': 'text/html',
                            'artifact_ref': 'artifact:index',
                            'content': '<!doctype html><html><body><section>Done</section></body></htm',
                        },
                    ],
                    'late_fill': {
                        'fill_results': [
                            {
                                'branch_id': 'branch-text_artifact-1',
                                'phase_id': 'phase-2',
                                'capability': 'chat',
                                'saved_text_path': str(text_path),
                                'text_artifact_extension': 'html',
                            },
                        ],
                    },
                },
                ledger_path=ledger_path,
            )
            found = find_artifact_registry_record(str(text_path), ledger_path=ledger_path)

        self.assertEqual(len(persisted), 1)
        self.assertEqual(found['artifact']['content'], final_content)
        self.assertEqual(
            found['artifact']['content_sha256'],
            hashlib.sha256(final_content.encode('utf-8')).hexdigest(),
        )
        self.assertEqual(
            found['artifact']['file_sha256'],
            hashlib.sha256(final_content.encode('utf-8')).hexdigest(),
        )
        self.assertEqual(found['artifact']['file_size_bytes'], len(final_content.encode('utf-8')))
        self.assertEqual(found['artifact']['content_source'], 'final_saved_text_artifact')
        self.assertEqual(found['metadata']['final_text_artifact_refresh_status'], 'refreshed')
        self.assertEqual(found['metadata']['syntax_sanity_status'], 'ok')
        self.assertEqual(found['metadata']['syntax_sanity_issue_count'], 0)
        self.assertEqual(found['metadata']['branch_id'], 'branch-text_artifact-1')

    def test_registry_merge_clears_stale_syntax_issues_when_refresh_is_clean(self):
        merged = merge_artifact_registry_records(
            {
                'artifact_ref': 'artifact:index',
                'artifact_id': 'artifact:index',
                'type': 'text',
                'artifact': {'path': '/tmp/index.html'},
                'metadata': {
                    'syntax_sanity_status': 'issues',
                    'syntax_sanity_issue_count': 2,
                    'syntax_sanity_issues': ['HTML has stray closing tag </a> at line 16'],
                },
            },
            {
                'artifact_ref': 'artifact:index',
                'artifact_id': 'artifact:index',
                'type': 'text',
                'artifact': {'path': '/tmp/index.html'},
                'metadata': {
                    'syntax_sanity_status': 'ok',
                    'syntax_sanity_issue_count': 0,
                },
            },
        )

        self.assertEqual(merged['metadata']['syntax_sanity_status'], 'ok')
        self.assertEqual(merged['metadata']['syntax_sanity_issue_count'], 0)
        self.assertNotIn('syntax_sanity_issues', merged['metadata'])

    def test_output_artifact_registry_prefers_late_fill_branch_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / 'audio' / 'branch.wav'
            transcript_path = root / 'transcripts' / 'branch.md'
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b'wav')
            transcript_path.write_text('transcript', encoding='utf-8')

            records = build_output_artifact_registry_records(
                {
                    'id': 'resp_late_fill_outputs',
                    'model': 'gemma4:26b',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'mode': 'chat',
                    'saved_audio_path': str(audio_path),
                    'saved_text_path': str(transcript_path),
                    'artifacts': [
                        {'type': 'audio', 'path': str(audio_path), 'mime_type': 'audio/wav'},
                        {'type': 'text', 'path': str(transcript_path), 'mime_type': 'text/plain'},
                    ],
                    'late_fill': {
                        'fill_results': [
                            {
                                'branch_id': 'branch-text_to_speech-1',
                                'phase_id': 'phase-2',
                                'capability': 'text_to_speech',
                                'fill_instance_id': 'tts-1',
                                'fill_model': 'qwen3-tts',
                                'fill_backend': 'mlx',
                                'fill_mode': 'text_to_speech',
                                'saved_audio_path': str(audio_path),
                            },
                            {
                                'branch_id': 'branch-speech_to_text-1',
                                'phase_id': 'phase-3',
                                'capability': 'speech_to_text',
                                'fill_instance_id': 'whisper-1',
                                'fill_model': 'whisper-large',
                                'fill_backend': 'mlx',
                                'fill_mode': 'speech_to_text',
                                'saved_text_path': str(transcript_path),
                            },
                        ],
                    },
                }
            )

        by_type = {record['type']: record for record in records}
        self.assertEqual(by_type['audio']['provenance']['source']['capability'], 'text_to_speech')
        self.assertEqual(by_type['audio']['provenance']['source']['instance_id'], 'tts-1')
        self.assertEqual(by_type['audio']['metadata']['branch_id'], 'branch-text_to_speech-1')
        self.assertEqual(by_type['text']['provenance']['source']['capability'], 'speech_to_text')
        self.assertEqual(by_type['text']['provenance']['source']['instance_id'], 'whisper-1')
        self.assertEqual(by_type['text']['metadata']['branch_id'], 'branch-speech_to_text-1')

    def test_generic_output_registry_does_not_overwrite_generated_image_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / 'images' / 'dream.png'
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b'png')
            ledger_path = root / 'artifact_registry.jsonl'
            provenance = build_generated_image_provenance(
                image_path=str(image_path),
                prompt_text='dream city',
                instance_id='image-1',
                model='flux',
                backend='ollama',
                capability='image_generation',
                mode='image_generation',
                response_id='resp_image',
            )
            persist_generated_image_provenance(provenance, ledger_path=ledger_path)

            persist_output_artifact_registry_records(
                {
                    'id': 'resp_image',
                    'model': 'flux',
                    'backend': 'ollama',
                    'capability': 'image_generation',
                    'mode': 'image_generation',
                    'saved_image_path': str(image_path),
                },
                ledger_path=ledger_path,
            )
            found = find_artifact_registry_record(str(image_path), ledger_path=ledger_path)

        self.assertEqual(found['provenance']['kind'], 'ollmo.generated_image_provenance')
        self.assertEqual(found['provenance']['provenance_id'], provenance['provenance_id'])
        self.assertEqual(found['artifact']['origin'], 'generated_output')
        self.assertEqual(found['linked_response_ids'], ['resp_image'])


if __name__ == '__main__':
    unittest.main()
