import json
import tempfile
import unittest
from pathlib import Path

from ollmo_services.chat_history import write_chat_history
from ollmo_services.context_scan import build_history_scan_context_candidates


class ContextScanTests(unittest.TestCase):
    def test_promoted_history_scan_links_existing_ledgers_as_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'chat_history'
            artifact_ledger = root / 'artifact_registry.jsonl'
            frame_ledger = root / 'response_frames' / 'responses.jsonl'
            frame_ledger.parent.mkdir(parents=True, exist_ok=True)

            write_chat_history(
                'architecture-session',
                [
                    {'role': 'user', 'content': 'We discussed the moonlit castle motif.'},
                    {'role': 'assistant', 'content': 'The moonlit castle motif should stay fluid until promoted.'},
                ],
                history_dir=history_dir,
                model='gemma',
                backend='ollama',
                capability='chat',
            )
            frame_ledger.write_text(
                json.dumps(
                    {
                        'response_id': 'resp_castle',
                        'request': {'prompt': 'moonlit castle still image'},
                        'output_text': 'A frozen still image for the moonlit castle.',
                        'artifacts': {'output': [{'artifact_ref': 'artifact:image_castle'}]},
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )
            artifact_ledger.write_text(
                json.dumps(
                    {
                        'kind': 'ollmo.artifact_registry_record',
                        'artifact_ref': 'artifact:image_castle',
                        'artifact': {
                            'type': 'image',
                            'path': '/tmp/moonlit-castle.png',
                            'artifact_ref': 'artifact:image_castle',
                        },
                        'metadata': {'prompt': 'moonlit castle by the lake'},
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )

            result = build_history_scan_context_candidates(
                prompt='Search the entire conversation history for the moonlit castle.',
                history_dir=history_dir,
                artifact_registry_ledger=artifact_ledger,
            )

        self.assertEqual(result['status'], 'completed')
        surfaces = {
            item['source_surface']
            for item in result['context_candidates']
        }
        self.assertIn('chat_history', surfaces)
        self.assertIn('response_frame_ledger', surfaces)
        self.assertIn('artifact_registry', surfaces)
        self.assertEqual(result['matched']['chat_history'], 2)
        self.assertEqual(result['matched_candidate_count'], 4)
        self.assertEqual(result['promoted_candidate_count'], 4)
        self.assertEqual(result['omitted_candidate_count'], 0)
        self.assertEqual(result['ranking_policy'], 'lexical_term_overlap_then_source_order')
        artifact_candidate = next(
            item for item in result['context_candidates']
            if item['source_surface'] == 'artifact_registry'
        )
        self.assertEqual(artifact_candidate['artifact_ref'], 'artifact:image_castle')
        self.assertEqual(artifact_candidate['promotion_target'], 'active_reference')
        self.assertIn('rank', artifact_candidate)
        self.assertIn('relevance_score', artifact_candidate)

    def test_promoted_history_scan_audits_omitted_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'chat_history'
            for index in range(3):
                write_chat_history(
                    f'architecture-session-{index}',
                    [
                        {'role': 'user', 'content': f'The prism bridge motif appears in note {index}.'},
                    ],
                    history_dir=history_dir,
                    model='gemma',
                    backend='ollama',
                    capability='chat',
                )

            result = build_history_scan_context_candidates(
                prompt='Search the entire conversation history for the prism bridge motif.',
                history_dir=history_dir,
                max_candidates=2,
            )

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['candidate_count'], 2)
        self.assertGreaterEqual(result['matched_candidate_count'], 3)
        self.assertEqual(result['promoted_candidate_count'], 2)
        self.assertEqual(
            result['omitted_candidate_count'],
            result['matched_candidate_count'] - result['promoted_candidate_count'],
        )
        self.assertEqual(
            [item['rank'] for item in result['context_candidates']],
            [1, 2],
        )


if __name__ == '__main__':
    unittest.main()
