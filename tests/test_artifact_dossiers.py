import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ollmo_services.artifact_dossiers import build_artifact_dossier_index


class ArtifactDossierTests(unittest.TestCase):
    @patch('ollmo_services.artifact_dossiers.find_artifact_registry_record_by_artifact_ref')
    def test_build_artifact_dossier_index_uses_actual_provenance_metadata_when_available(self, mock_find_registry):
        mock_find_registry.return_value = {
            'artifact_ref': 'artifact:image_deadbeef',
            'artifact_id': 'image_deadbeef',
            'type': 'image',
            'artifact': {
                'type': 'image',
                'path': 'artifacts/images/out.png',
                'artifact_id': 'image_deadbeef',
                'artifact_ref': 'artifact:image_deadbeef',
            },
            'metadata': {
                'width': 1280,
                'height': 720,
                'seed': 42,
                'response_model': 'flux',
                'response_instance_id': 'image-1',
            },
            'linked_response_ids': ['resp_1'],
            'provenance': {
                'provenance_id': 'generated_image_deadbeef',
                'source': {
                    'instance_id': 'image-1',
                    'model': 'flux',
                    'backend': 'mlx',
                    'response_id': 'resp_1',
                    'request_origin': 'responses',
                },
                'request': {
                    'width': 1280,
                    'height': 720,
                    'seed': 42,
                },
                'output': {
                    'artifact_id': 'image_deadbeef',
                    'artifact_ref': 'artifact:image_deadbeef',
                },
                'derived_from': ['artifact:text_story'],
            },
        }

        dossiers = build_artifact_dossier_index(
            output_artifacts=[
                {
                    'type': 'image',
                    'path': 'artifacts/images/out.png',
                    'artifact_id': 'image_deadbeef',
                    'artifact_ref': 'artifact:image_deadbeef',
                    'seed': 7,
                }
            ],
            response_payload={
                'saved_image_path': 'artifacts/images/out.png',
                'image_state': {'subject': 'dream temple'},
                'image_state_enrichment': {'status': 'completed', 'mode': 'background_analysis'},
            },
        )

        dossier = dossiers['artifact:image_deadbeef']
        self.assertEqual(dossier['provenance']['provenance_id'], 'generated_image_deadbeef')
        self.assertEqual(dossier['provenance']['derived_from'], ['artifact:text_story'])
        self.assertEqual(dossier['metadata']['width'], 1280)
        self.assertEqual(dossier['metadata']['height'], 720)
        self.assertEqual(dossier['metadata']['seed'], 42)
        self.assertEqual(dossier['metadata']['response_model'], 'flux')
        self.assertEqual(dossier['metadata']['response_instance_id'], 'image-1')
        self.assertEqual(dossier['linked_response_ids'], ['resp_1'])
        self.assertEqual(dossier['enrichments']['image_state']['subject'], 'dream temple')
        self.assertEqual(dossier['enrichments']['image_state_enrichment']['mode'], 'background_analysis')

    def test_build_artifact_dossier_index_exposes_runtime_metadata_and_selected_controls_for_primary_output(self):
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'story.wav'
            output_path.write_bytes(b'RIFF')

            dossiers = build_artifact_dossier_index(
                output_artifacts=[
                    {
                        'type': 'audio',
                        'path': str(output_path),
                        'artifact_id': 'audio_story',
                        'artifact_ref': 'artifact:audio_story',
                    }
                ],
                response_payload={
                    'saved_audio_path': str(output_path),
                    'model': 'qwen-tts',
                    'instance_id': 'tts-1',
                    'backend': 'mlx',
                    'capability': 'text_to_speech',
                    'mode': 'text_to_speech',
                    'runtime': {
                        'control_hints': {
                            'lang_code': 'de',
                            'voice': 'Bella',
                            'response_format': 'mp3',
                        }
                    },
                },
            )

            dossier = dossiers['artifact:audio_story']
            self.assertEqual(dossier['metadata']['response_model'], 'qwen-tts')
            self.assertEqual(dossier['metadata']['response_instance_id'], 'tts-1')
            self.assertEqual(dossier['metadata']['backend'], 'mlx')
            self.assertEqual(dossier['metadata']['capability'], 'text_to_speech')
            self.assertEqual(dossier['metadata']['selected_controls']['lang_code'], 'de')
            self.assertEqual(dossier['metadata']['selected_controls']['voice'], 'Bella')
            self.assertEqual(dossier['metadata']['selected_controls']['response_format'], 'mp3')
            self.assertEqual(dossier['metadata']['availability'], 'available')
            self.assertEqual(dossier['metadata']['availability_source'], 'filesystem')
            self.assertGreater(dossier['metadata']['size_bytes'], 0)

    @patch('ollmo_services.artifact_dossiers.find_artifact_registry_record')
    @patch('ollmo_services.artifact_dossiers.find_artifact_registry_record_by_artifact_ref')
    def test_late_fill_producer_metadata_precedes_root_fallback_before_registry_persistence(
        self,
        mock_find_registry_by_ref,
        mock_find_registry_by_path,
    ):
        mock_find_registry_by_ref.return_value = None
        mock_find_registry_by_path.return_value = None
        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'narration.wav'
            audio_path.write_bytes(b'RIFF')

            dossiers = build_artifact_dossier_index(
                output_artifacts=[
                    {
                        'type': 'audio',
                        'path': str(audio_path),
                        'artifact_id': 'audio_narration',
                        'artifact_ref': 'artifact:audio_narration',
                    }
                ],
                response_payload={
                    'id': 'resp_mixed_media',
                    'saved_audio_path': str(audio_path),
                    'model': 'gemma-root',
                    'instance_id': 'chat-root',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'mode': 'chat',
                    'late_fill': {
                        'fill_results': [
                            {
                                'branch_id': 'branch-text_to_speech-1',
                                'phase_id': 'phase-3',
                                'capability': 'text_to_speech',
                                'fill_instance_id': 'tts-1',
                                'fill_model': 'qwen3-tts',
                                'fill_backend': 'mlx',
                                'fill_mode': 'text_to_speech',
                                'saved_audio_path': str(audio_path),
                            }
                        ]
                    },
                },
            )

        dossier = dossiers['artifact:audio_narration']
        self.assertEqual(dossier['metadata']['response_model'], 'qwen3-tts')
        self.assertEqual(dossier['metadata']['response_instance_id'], 'tts-1')
        self.assertEqual(dossier['metadata']['backend'], 'mlx')
        self.assertEqual(dossier['metadata']['capability'], 'text_to_speech')
        self.assertEqual(dossier['metadata']['branch_id'], 'branch-text_to_speech-1')
        self.assertEqual(
            dossier['provenance']['record']['source']['capability'],
            'text_to_speech',
        )

    @patch('ollmo_services.artifact_dossiers.find_artifact_registry_record_by_artifact_ref')
    def test_build_artifact_dossier_index_reads_durable_registry_enrichments_without_response_overlay(self, mock_find_registry):
        mock_find_registry.return_value = {
            'artifact_ref': 'artifact:image_live',
            'artifact_id': 'image_live',
            'type': 'image',
            'artifact': {
                'type': 'image',
                'path': '/tmp/live.png',
                'artifact_id': 'image_live',
                'artifact_ref': 'artifact:image_live',
            },
            'metadata': {
                'response_model': 'flux',
            },
            'enrichments': {
                'image_state': {
                    'summary': 'floating temple over silver water',
                    'subject': 'temple',
                },
                'image_state_enrichment': {
                    'status': 'completed',
                    'mode': 'background_analysis',
                },
            },
        }

        dossiers = build_artifact_dossier_index(
            output_artifacts=[
                {
                    'type': 'image',
                    'path': '/tmp/live.png',
                    'artifact_ref': 'artifact:image_live',
                    'artifact_id': 'image_live',
                }
            ],
            response_payload={},
        )

        dossier = dossiers['artifact:image_live']
        self.assertEqual(dossier['enrichments']['image_state']['subject'], 'temple')
        self.assertEqual(
            dossier['enrichments']['image_state_enrichment']['status'],
            'completed',
        )


if __name__ == '__main__':
    unittest.main()
