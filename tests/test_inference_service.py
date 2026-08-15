import hashlib
import io
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from ollmo_core.inference import (
    InferArtifacts,
    InferContext,
    build_qwen3_tts_chunk_plan,
    build_qwen3_tts_generation_budget,
    detect_text_artifact_request,
    detect_text_artifact_requests,
    dispatch_infer_request,
    extract_text_artifact_payload,
    extract_text_artifact_payloads,
    generated_text_is_artifact_self_claim,
    text_artifact_request_is_ungrounded_reference,
)
import ollmo_core.inference as inference_service
import ollmo_core.transports as transports
from ollmo_core.transports import persist_text_artifact_locally


def _pcm_wav_bytes(duration_seconds: float, *, sample_rate: int = 8000) -> bytes:
    frame_count = int(round(duration_seconds * sample_rate))
    pcm = b''.join(
        int(
            12000 * math.sin((2 * math.pi * 220 * index) / sample_rate)
        ).to_bytes(2, 'little', signed=True)
        for index in range(frame_count)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def _pcm_wav_segment_bytes(
    segments: list[tuple[float, bool]],
    *,
    sample_rate: int = 8000,
) -> bytes:
    samples: list[int] = []
    phase = 0
    for duration_seconds, active in segments:
        frame_count = int(round(duration_seconds * sample_rate))
        for _ in range(frame_count):
            sample = (
                int(
                    12000
                    * math.sin((2 * math.pi * 220 * phase) / sample_rate)
                )
                if active
                else 0
            )
            samples.append(sample)
            phase += 1
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(
            b''.join(
                sample.to_bytes(2, 'little', signed=True)
                for sample in samples
            )
        )
    return buffer.getvalue()


class InferenceServiceTests(unittest.TestCase):
    def test_qwen3_tts_generation_budget_scales_and_clamps(self):
        short_budget = build_qwen3_tts_generation_budget('Hallo Welt.')
        medium_budget = build_qwen3_tts_generation_budget(
            ' '.join(f'Wort{index}' for index in range(1, 81))
        )
        long_budget = build_qwen3_tts_generation_budget(
            ' '.join(f'Wort{index}' for index in range(1, 1001))
        )

        self.assertEqual(short_budget['max_tokens'], 256)
        self.assertEqual(short_budget['clamp'], 'minimum')
        self.assertGreater(medium_budget['max_tokens'], short_budget['max_tokens'])
        self.assertLess(medium_budget['max_tokens'], 1200)
        self.assertEqual(long_budget['max_tokens'], 1200)
        self.assertEqual(long_budget['clamp'], 'maximum')
        self.assertEqual(
            medium_budget['policy_id'],
            'qwen3_tts_adaptive_audio_tokens_v2',
        )
        self.assertEqual(medium_budget['policy']['audio_tokens_per_second'], 12.5)
        self.assertEqual(medium_budget['policy']['fixed_buffer_seconds'], 8.0)
        self.assertEqual(medium_budget['policy']['minimum_tokens'], 256)

    def test_qwen3_single_sequence_generation_limit_recovery_scales_and_clamps(self):
        initial_budget = build_qwen3_tts_generation_budget(
            'Mara stood beside the old lighthouse and listened to the waves.'
        )
        initial_budget.update(
            {
                'max_tokens': 269,
                'tts_model_type': 'voice_design',
                'generation_scope': 'single_sequence',
            }
        )

        recovery = inference_service._build_qwen3_tts_generation_limit_recovery(
            initial_budget
        )

        self.assertTrue(recovery['applied'])
        self.assertEqual(recovery['initial_max_tokens'], 269)
        self.assertEqual(recovery['recovery_max_tokens'], 404)
        self.assertEqual(recovery['clamp'], 'none')
        self.assertEqual(
            recovery['generation_budget']['policy_id'],
            'qwen3_tts_single_sequence_generation_limit_retry_v2',
        )
        self.assertEqual(recovery['generation_budget']['max_tokens'], 404)

        near_max_budget = dict(initial_budget, max_tokens=1000)
        clamped = inference_service._build_qwen3_tts_generation_limit_recovery(
            near_max_budget
        )
        self.assertTrue(clamped['applied'])
        self.assertEqual(clamped['recovery_max_tokens'], 1200)
        self.assertEqual(clamped['clamp'], 'maximum')

        max_budget = dict(initial_budget, max_tokens=1200)
        unavailable = inference_service._build_qwen3_tts_generation_limit_recovery(
            max_budget
        )
        self.assertFalse(unavailable['applied'])
        self.assertEqual(unavailable['status'], 'not_eligible')
        self.assertEqual(unavailable['recovery_max_tokens'], 1200)

    def test_qwen3_tts_long_form_chunk_plan_preserves_ordered_source(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )

        plan = build_qwen3_tts_chunk_plan(passage)

        self.assertTrue(plan['applied'])
        self.assertTrue(plan['ordered_span_coverage'])
        self.assertGreater(plan['chunk_count'], 1)
        self.assertEqual(plan['source_sha256'], hashlib.sha256(passage.encode()).hexdigest())
        self.assertEqual(
            ' '.join(' '.join(chunk['text'].split()) for chunk in plan['chunks']),
            ' '.join(passage.split()),
        )
        self.assertTrue(
            all(
                chunk['estimated_speech_seconds'] <= plan['target_chunk_speech_seconds']
                for chunk in plan['chunks']
            )
        )

    def test_detect_text_artifact_request_from_explicit_filename(self):
        request = detect_text_artifact_request('Create an index.html artifact with a simple landing page.')

        self.assertIsNotNone(request)
        self.assertEqual(request['extension'], 'html')
        self.assertEqual(request['source_name'], 'index')

    def test_detect_text_artifact_request_allows_explicit_filename_with_that_says(self):
        request = detect_text_artifact_request(
            'Create an index.html artifact with a minimal page that says "Signal Garden" in blue text.'
        )

        self.assertIsNotNone(request)
        self.assertEqual(request['extension'], 'html')
        self.assertEqual(request['source_name'], 'index')

    def test_detect_text_artifact_requests_from_multiple_format_artifacts(self):
        requests = detect_text_artifact_requests(
            'Create an HTML artifact for a landing page and a CSS artifact for its styles.'
        )

        self.assertEqual([item['extension'] for item in requests], ['html', 'css'])
        self.assertEqual([item['source_name'] for item in requests], ['generated-html', 'generated-css'])

    def test_detect_text_artifact_requests_from_german_plural_artifacts(self):
        requests = detect_text_artifact_requests(
            'Erstelle ein HTML und ein CSS als getrennte Artefakte für eine kleine Ollmo-Landingpage.'
        )

        self.assertEqual([item['extension'] for item in requests], ['html', 'css'])
        self.assertEqual([item['source_name'] for item in requests], ['generated-html', 'generated-css'])

    def test_detect_text_artifact_requests_from_original_dedicated_css_prompt(self):
        requests = detect_text_artifact_requests(
            "Build a bold landing page with html and a dedicated css for an eco-friendly clothing line "
            "called 'Pure Thread.' Generate four images: a macro shot of organic cotton texture, a model "
            'wearing a simple white linen shirt, a close-up of sustainable recycled buttons, and a sunny '
            'outdoor scene at a botanical garden.'
        )

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'generated-html', 'distinct_format_artifact_cue'),
                ('css', 'generated-css', 'distinct_format_artifact_cue'),
            ],
        )

    def test_detect_text_artifact_requests_from_distinct_format_variants(self):
        prompts = (
            'Build a compact HTML landing page with a separate CSS stylesheet.',
            'Create a standalone HTML page with its own external stylesheet.',
            'Erstelle eine HTML-Landingpage mit eigenem CSS.',
            'Baue eine Landingpage in HTML mit separatem CSS.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                requests = detect_text_artifact_requests(prompt)
                self.assertEqual([item['extension'] for item in requests], ['html', 'css'])

    def test_detect_text_artifact_requests_ignores_distinct_format_conversation(self):
        requests = detect_text_artifact_requests(
            'Explain why HTML and CSS are separate concerns.'
        )

        self.assertEqual(requests, [])

    def test_detect_text_artifact_requests_treats_final_json_object_as_response_format(self):
        prompts = (
            'Gib abschließend ein JSON-Objekt aus, das Bildreferenz, sichtbare Evidenz, '
            'Audioreferenz und Transkript getrennt als Felder bindet.',
            'Return a separate JSON object with one field for each result.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(detect_text_artifact_requests(prompt), [])

    def test_detect_text_artifact_requests_preserves_explicit_json_materialization(self):
        prompts = (
            'Create a JSON file containing the final result.',
            'Erstelle ein JSON-Artefakt für eine Inventarliste.',
            'Materialisiere eine Inventarliste als JSON.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                requests = detect_text_artifact_requests(prompt)
                self.assertEqual([item['extension'] for item in requests], ['json'])

    def test_detect_text_artifact_requests_keeps_named_json_in_file_manifests(self):
        prompts = (
            'Build a polished local two-page watch atelier site with index.html, configurator.html, '
            'styles.css, and pricing.json. The pages should link to each other, share the stylesheet, '
            'and use the pricing data consistently. Save it as one complete local bundle.',
            'Could you build a local site with index.html, configurator.html, styles.css, '
            'and pricing.json?',
            'Create exactly four web files: index.html, configurator.html, pricing.json, and styles.css.',
            'Create exactly four web files:\n'
            '1. index.html\n2. configurator.html\n3. pricing.json\n4. styles.css',
            'Create exactly four web files:\n'
            'index.html\nconfigurator.html\npricing.json\nstyles.css',
            'Create these files:\n- `index.html`\n- `configurator.html`\n- `pricing.json`\n- `styles.css`',
            'Create exactly four web files:\n\n'
            '1. index.html\n2. configurator.html\n3. pricing.json\n4. styles.css',
            'Create exactly four web files:\n'
            '1. index.html\n   Main page\n2. configurator.html\n   Builder page\n'
            '3. pricing.json\n   Pricing data\n4. styles.css\n   Shared styles',
            'Create these artifacts:\n'
            '1. index.html\n2. configurator.html\n3. one image of a watch\n'
            '4. pricing.json\n5. styles.css',
            'Create these files:\n'
            '1. site/index.html\n2. site/configurator.html\n'
            '3. data/pricing.json\n4. css/styles.css',
            'Create these files: site/index.html, site/configurator.html, '
            'data/pricing.json, css/styles.css.',
            'Create these files: ./pricing.json, index.html, configurator.html, styles.css.',
            'Create the following:\n'
            '1. index.html\n2. configurator.html\n3. pricing.json\n4. styles.css',
            'Create these files.\n'
            '1. index.html\n2. configurator.html\n3. pricing.json\n4. styles.css',
            'Erstelle genau vier Dateien:\n'
            '1. index.html\n2. configurator.html\n3. pricing.json\n4. styles.css',
        )
        expected = sorted(
            [
                ('configurator', 'html'),
                ('index', 'html'),
                ('pricing', 'json'),
                ('styles', 'css'),
            ]
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                requests = detect_text_artifact_requests(prompt)
                self.assertEqual(
                    sorted((item['source_name'], item['extension']) for item in requests),
                    expected,
                )

    def test_detect_text_artifact_requests_does_not_promote_json_source_mentions(self):
        prompts = (
            'Create index.html using data from pricing.json.',
            'Build a polished site with index.html using data from pricing.json.',
            'Explain how to build a site with index.html and pricing.json.',
            'Create exactly two files: index.html and styles.css. Use data from pricing.json.',
            'Create README.md and mention pricing.json in it.',
            'Explain this file list:\n1. pricing.json\n2. index.html',
            'Create a page from these source files:\n1. pricing.json\n2. content.md',
            'Create a comparison between index.html and pricing.json.',
            'Create a summary of index.html and pricing.json.',
            'Create documentation for index.html and pricing.json.',
            'Create a list of files:\n1. index.html\n2. pricing.json',
            'The generated files are:\n1. index.html\n2. pricing.json',
            'Created files:\n1. index.html\n2. pricing.json',
            'Saved artifacts:\n1. index.html\n2. pricing.json',
            'Return a JSON object listing these files:\n1. index.html\n2. pricing.json',
            'Explain this instruction:\n"Create these files:\n1. index.html\n2. pricing.json"',
            'Explain the following example:\n'
            '```text\nCreate these files:\n1. index.html\n2. pricing.json\n```',
            'Create a report: index.html, pricing.json.',
            'Create a table: index.html, pricing.json.',
            'Create a summary: index.html, pricing.json.',
            'Provide a list: index.html, pricing.json.',
            'Review and return these files: index.html, pricing.json.',
            'Which of these files should be generated:\n1. index.html\n2. pricing.json',
            'Tell me whether these files should be generated:\n1. index.html\n2. pricing.json',
            'Decide which files must be created:\n1. index.html\n2. pricing.json',
            'Assess whether these files need to be generated:\n1. index.html\n2. pricing.json',
            'Welche dieser Dateien sollen erstellt werden:\n1. index.html\n2. pricing.json',
            'Suggested files to generate:\n1. index.html\n2. pricing.json',
            'Candidate files to create:\n1. index.html\n2. pricing.json',
            'Optional files to create:\n1. index.html\n2. pricing.json',
            'Plan the files to create:\n1. index.html\n2. pricing.json',
            'Provide checksums for these files:\n1. index.html\n2. pricing.json',
            'Return the status of these files:\n1. index.html\n2. pricing.json',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                requests = detect_text_artifact_requests(prompt)
                self.assertNotIn('json', [item['extension'] for item in requests])

    def test_detect_text_artifact_requests_json_response_format_boundary_matrix(self):
        no_artifact_prompts = (
            'Return a JSON object with artifact references for each result.',
            'Gib ein JSON-Objekt mit Artefakt-Referenzen für jedes Ergebnis aus.',
            'Return a JSON object with file paths for each result.',
            'Explain why "Create a JSON file" is ambiguous.',
            'Repeat exactly: "Save the result as a JSON file."',
            'Do not create config.json.',
            "Don't create config.json.",
            'Never create a JSON file.',
            'Return JSON only; no file artifact.',
            'Return a JSON object, not a file artifact.',
            'Kein JSON-Artefakt; gib nur ein JSON-Objekt aus.',
            'Gib nur ein JSON-Objekt aus, nicht als Datei.',
        )

        for prompt in no_artifact_prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(detect_text_artifact_requests(prompt), [])

        sibling_requests = detect_text_artifact_requests(
            'Create an HTML file and return a JSON object with a summary.'
        )
        self.assertEqual(
            [(item['extension'], item['source']) for item in sibling_requests],
            [('html', 'explicit_format_file_cue')],
        )

        quoted_requests = detect_text_artifact_requests(
            'Create a README file explaining the phrase "Return a JSON object with file paths".'
        )
        self.assertEqual(
            [(item['extension'], item['source']) for item in quoted_requests],
            [('md', 'explicit_format_file_cue')],
        )

    def test_detect_text_artifact_requests_treats_json_reference_maps_as_response_only(self):
        prompts = (
            'Return a JSON artifact-ref map.',
            'Return a JSON file-path map.',
            'Gib eine JSON-Artefakt-Referenz aus.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(detect_text_artifact_requests(prompt), [])

    def test_detect_text_artifact_requests_preserves_quoted_json_filenames_after_file_action(self):
        prompts = (
            'Create the file "config.json".',
            'Create the file `config.json`.',
            'Save the file "config.json".',
            'Save the file `config.json`.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    detect_text_artifact_requests(prompt),
                    [
                        {
                            'extension': 'json',
                            'source': 'explicit_extension',
                            'source_name': 'config',
                        }
                    ],
                )

    def test_detect_text_artifact_requests_masks_quoted_json_instructions_and_negation(self):
        prompts = (
            'Explain why "Create the file config.json" is explicit.',
            'Repeat exactly: `Save the file config.json`.',
            'Do not create the file "config.json".',
            'Do not save the file `config.json`.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(detect_text_artifact_requests(prompt), [])

    def test_detect_text_artifact_requests_recognizes_bounded_json_file_format_grammar(self):
        prompts = (
            'Create a file in JSON format.',
            'Create a file formatted as JSON.',
            'Create a JSON-formatted file.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    detect_text_artifact_requests(prompt),
                    [
                        {
                            'extension': 'json',
                            'source': 'explicit_format_file_cue',
                            'source_name': 'generated-json',
                        }
                    ],
                )

    def test_detect_text_artifact_requests_json_materialization_boundary_matrix(self):
        prompts = (
            'Create a JSON file containing the final result.',
            'Create a JSON artifact containing the final result.',
            'Create config.json with the final result.',
            'Erstelle ein JSON-Artefakt für eine Inventarliste.',
            'Save the final result as JSON.',
            'Materialize the final result as JSON.',
            'Materialisiere eine Inventarliste als JSON.',
            'Output the final result as a JSON file.',
            'Download the final result as JSON.',
            'Export the final result as JSON.',
            'Persist the final result as JSON.',
            'Create a JSON file and return it.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                requests = detect_text_artifact_requests(prompt)
                self.assertEqual([item['extension'] for item in requests], ['json'])

        self.assertFalse(
            text_artifact_request_is_ungrounded_reference('Create a JSON file and return it.')
        )

    def test_detect_text_artifact_requests_from_local_code_and_image_artifact_bundle(self):
        requests = detect_text_artifact_requests(
            'Baue eine expressive Landingpage für eine imaginäre Social-App für Haustier-Selfies '
            'namens "Petsie". Erzeuge vier unterschiedliche Tierbilder, und schreibe die Texte so, '
            'dass jedes Bild inhaltlich exakt zum jeweiligen Abschnitt passt. Die Seite soll verspielt, '
            'aber nicht kindisch wirken. HTML, CSS und Bilder müssen als lokale Artefakte sauber '
            'zusammenpassen.'
        )

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'generated-html', 'explicit_format_file_cue'),
                ('css', 'generated-css', 'explicit_format_file_cue'),
            ],
        )

    def test_detect_text_artifact_requests_from_english_local_code_and_image_bundle(self):
        requests = detect_text_artifact_requests(
            'Build an expressive landing page for an imaginary social app for pet selfies called "Petsie". '
            'Generate four distinct animal images. Each image must match one specific section of the page, '
            'and the copy in that section must directly refer to the animal/image shown there. '
            'The page should feel playful but not childish. Create local HTML, CSS, and image artifacts '
            'that work together cleanly. The HTML must reference the generated local images and the local '
            'CSS file correctly.'
        )

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'generated-html', 'explicit_format_file_cue'),
                ('css', 'generated-css', 'explicit_format_file_cue'),
            ],
        )

    def test_detect_text_artifact_requests_includes_readme_artifact(self):
        requests = detect_text_artifact_requests(
            'Erstelle ein HTML, ein CSS und ein kurzes README als getrennte Artefakte für eine Mini-Landingpage.'
        )

        self.assertEqual([item['extension'] for item in requests], ['html', 'css', 'md'])
        self.assertEqual([item['source_name'] for item in requests], ['generated-html', 'generated-css', 'README'])

    def test_detect_text_artifact_requests_ignores_negated_json_output_format(self):
        requests = detect_text_artifact_requests(
            'Baue ein kleines HTML+CSS Dashboard als Artefakte. Kein json output bitte.'
        )

        self.assertEqual([item['extension'] for item in requests], ['html', 'css'])

    def test_detect_text_artifact_requests_keeps_html_when_css_filename_is_named_later(self):
        requests = detect_text_artifact_requests(
            'Create an HTML artifact for a landing page and a CSS artifact for its styles. '
            'Use fenced code blocks for both. Name the stylesheet style.css and link it from the HTML.'
        )

        self.assertEqual(
            [(item['extension'], item['source_name']) for item in requests],
            [('html', 'generated-html'), ('css', 'style')],
        )

    def test_detect_text_artifact_requests_ignores_validation_only_filename_mentions(self):
        prompt = (
            'Create a polished one-screen landing page for Aethelgard Abyss-7. '
            'Generate exactly one local image artifact first. '
            'Then create exactly two local file artifacts: 1. index.html 2. styles.css. '
            'Treat missing PNG/image artifact, missing index.html, or missing styles.css as incomplete.'
        )

        requests = detect_text_artifact_requests(prompt)

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'index', 'explicit_extension'),
                ('css', 'styles', 'explicit_extension'),
            ],
        )

    def test_detect_text_artifact_requests_ignores_closure_only_filename_requirements(self):
        requests = detect_text_artifact_requests(
            'Treat missing index.html or missing styles.css as incomplete.'
        )

        self.assertEqual(requests, [])

    def test_detect_text_artifact_requests_keeps_files_when_no_extra_artifacts_constraint(self):
        prompt = (
            'Create exactly two local file artifacts: 1. index.html 2. styles.css. '
            'Do not create extra HTML or CSS artifacts beyond index.html and styles.css.'
        )

        requests = detect_text_artifact_requests(prompt)

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'index', 'explicit_extension'),
                ('css', 'styles', 'explicit_extension'),
            ],
        )

    def test_detect_text_artifact_requests_keeps_named_files_when_chat_blocks_do_not_count(self):
        prompt = (
            'Erstelle genau drei Bilder und zwei Dateien: index.html und styles.css. '
            'Materialisiere exakt fünf fertige Artefakte. '
            'Chat-Codeblöcke zählen nicht als Dateien.'
        )

        requests = detect_text_artifact_requests(prompt)

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'index', 'explicit_extension'),
                ('css', 'styles', 'explicit_extension'),
            ],
        )

    def test_detect_text_artifact_requests_keeps_named_files_when_only_artifacts_are_requested(self):
        prompt = (
            'Erstelle genau vier Bilder und zwei Dateien: index.html und styles.css. '
            'Am Ende keine Erklärung, nur die fertigen Artefakte.'
        )

        requests = detect_text_artifact_requests(prompt)

        self.assertEqual(
            [(item['extension'], item['source_name'], item['source']) for item in requests],
            [
                ('html', 'index', 'explicit_extension'),
                ('css', 'styles', 'explicit_extension'),
            ],
        )

    def test_detect_text_artifact_requests_still_honors_true_file_negation(self):
        requests = detect_text_artifact_requests(
            'Skizziere eine Landingpage-Idee, aber erstelle keine Dateien oder Artefakte.'
        )

        self.assertEqual(requests, [])

    def test_detect_text_artifact_requests_honors_negated_named_files(self):
        cases = (
            (
                'Do not create index.html; just explain the structure.',
                [],
            ),
            (
                'Erstelle keine index.html; erkläre nur die Struktur.',
                [],
            ),
            (
                'Do not create index.html; create styles.css instead.',
                [('css', 'styles', 'explicit_extension')],
            ),
        )

        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                requests = detect_text_artifact_requests(prompt)
                self.assertEqual(
                    [
                        (item['extension'], item['source_name'], item['source'])
                        for item in requests
                    ],
                    expected,
                )

    def test_detect_text_artifact_request_ignores_plain_chat(self):
        self.assertIsNone(detect_text_artifact_request('Tell me what HTML means in one sentence.'))
        self.assertIsNone(detect_text_artifact_request('What does index.html mean?'))

    def test_detect_text_artifact_request_blocks_ungrounded_this_without_source(self):
        prompt = 'Generate me this html file as artifact'

        self.assertTrue(text_artifact_request_is_ungrounded_reference(prompt))
        self.assertIsNone(detect_text_artifact_request(prompt))
        self.assertIsNotNone(detect_text_artifact_request(prompt, source_available=True))

    def test_relative_media_content_clause_is_not_an_ungrounded_file_reference(self):
        prompts = (
            'Create one English audio artifact that says, "The lighthouse welcomes the morning."',
            'Create one English audio artifact that reads "The harbor is awake."',
            'Create one image that shows a lighthouse at sunrise.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(text_artifact_request_is_ungrounded_reference(prompt))

        self.assertTrue(
            text_artifact_request_is_ungrounded_reference(
                'Generate me this html file as artifact'
            )
        )

    def test_detect_text_artifact_request_carries_selected_source_target_path(self):
        request = detect_text_artifact_request(
            'Change the font color and save the updated artifact.',
            source_available=True,
            source_extension='html',
            source_name='index',
            source_path='/tmp/artifacts/documents/index.html',
        )

        self.assertIsNotNone(request)
        self.assertEqual(request['source'], 'selected_source_edit')
        self.assertEqual(request['extension'], 'html')
        self.assertEqual(request['source_name'], 'index')
        self.assertEqual(request['target_path'], '/tmp/artifacts/documents/index.html')

    def test_named_selected_source_edit_outranks_generic_explicit_extension(self):
        request = detect_text_artifact_request(
            'Update styles.css for the visualizer and keep the rest of the existing design intact.',
            source_available=True,
            source_extension='css',
            source_name='styles',
            source_path='/tmp/artifacts/documents/styles.css',
        )

        self.assertIsNotNone(request)
        self.assertEqual(request['source'], 'selected_source_edit')
        self.assertEqual(request['extension'], 'css')
        self.assertEqual(request['source_name'], 'styles')
        self.assertEqual(request['target_path'], '/tmp/artifacts/documents/styles.css')

    def test_detect_text_artifact_request_allows_self_contained_generated_text_save(self):
        prompt = (
            'Erstelle einen kurzen Evakuierungshinweis für Bergretter, speichere ihn als txt-Artefakt, '
            'erzeuge daraus Audio.'
        )

        self.assertFalse(text_artifact_request_is_ungrounded_reference(prompt))
        request = detect_text_artifact_request(prompt)
        self.assertIsNotNone(request)
        self.assertEqual(request['extension'], 'txt')
        self.assertIsNone(detect_text_artifact_request('Speichere ihn als txt-Artefakt.'))

    def test_detect_text_artifact_request_allows_this_with_inline_source(self):
        prompt = 'Generate this html file as artifact: <!doctype html><h1>Hello</h1>'

        self.assertFalse(text_artifact_request_is_ungrounded_reference(prompt))
        self.assertIsNotNone(detect_text_artifact_request(prompt))

    def test_detect_text_artifact_request_allows_selected_source_edit(self):
        request = detect_text_artifact_request(
            'please change the font to red.',
            source_available=True,
            source_extension='html',
            source_name='index',
        )

        self.assertIsNotNone(request)
        self.assertEqual(request['extension'], 'html')
        self.assertEqual(request['source'], 'selected_source_edit')
        self.assertEqual(request['source_name'], 'index')
        self.assertIsNone(detect_text_artifact_request('please change the font to red.'))

    def test_detect_text_artifact_request_allows_selected_source_section_edit(self):
        request = detect_text_artifact_request(
            'Add a rollback section to the checklist.',
            source_available=True,
            source_extension='md',
            source_name='launch-checklist',
        )

        self.assertIsNotNone(request)
        self.assertEqual(request['extension'], 'md')
        self.assertEqual(request['source'], 'selected_source_edit')
        self.assertEqual(request['source_name'], 'launch-checklist')

    def test_detect_text_artifact_request_does_not_bind_unrelated_audio_branch_edit_to_selected_transcript(self):
        prompt = (
            'Beziehe dich ausdrücklich auf das Observatorium-Bild, seine Bildanalyse, die deutsche '
            'Erzählung und das Audio aus dem unmittelbar vorherigen Turn. Bewahre Bild und Bildanalyse '
            'unverändert; erzeuge das Bild nicht neu und analysiere es nicht erneut. Ersetze den bisherigen '
            'einzelnen Audiozweig durch zwei getrennte Audiofassungen: einmal die ursprüngliche deutsche '
            'Erzählung und einmal eine getreue englische Übersetzung. Transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein neues JSON-Objekt aus, das die unveränderte Bildevidenz '
            'sowie beide Audio-artifact_refs und beide realen Transkripte eindeutig verbindet.'
        )

        requests = detect_text_artifact_requests(
            prompt,
            source_available=True,
            source_extension='md',
            source_name='observatorium-transkript',
            source_path='/tmp/artifacts/transcripts/observatorium-transkript.md',
        )

        self.assertEqual(requests, [])

    def test_selected_source_edit_binding_is_action_local(self):
        cases = (
            (
                'Replace the old audio branch with two new audio versions and return a JSON object.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Replace the old audio branch. Add a rollback section to the checklist.',
                'md',
                'launch-checklist',
                True,
            ),
            (
                'Ändere dieses JSON-Objekt: setze den Status auf fertig.',
                'json',
                'current-settings',
                True,
            ),
            (
                'Replace the old audio branch and return a new JSON object.',
                'json',
                'prior-result',
                False,
            ),
            (
                'Update this file and fix the typo in its heading.',
                'txt',
                'release-notes',
                True,
            ),
            (
                'Replace the audio branch using this document as context.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Replace the audio branch based on the attached document.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Add a footer.',
                'html',
                'index',
                True,
            ),
            (
                'Remove the button.',
                'html',
                'index',
                True,
            ),
            (
                'Replace the introduction.',
                'md',
                'release-notes',
                True,
            ),
            (
                'Add a new JSON object with both artifact refs.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Update the JSON response to include both artifact refs.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Include a status field in the final JSON object.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Füge beide artifact_refs in das neue JSON-Objekt ein.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Return a JSON object and add both artifact refs.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Gib ein JSON-Objekt aus und füge beide artifact_refs hinzu.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Return a JSON object and then edit this document.',
                'md',
                'prior-transcript',
                True,
            ),
            (
                'Replace the audio branch, then add a footer.',
                'html',
                'index',
                True,
            ),
            (
                'Replace the audio branch and add a rollback section to the checklist.',
                'md',
                'notes',
                True,
            ),
            (
                'Update the model field to qwen.',
                'json',
                'settings',
                True,
            ),
            (
                'Replace the image URL.',
                'html',
                'index',
                True,
            ),
            (
                'Replace the audio link in this document.',
                'html',
                'index',
                True,
            ),
            (
                'Replace the audio branch and include its URL in the JSON response.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Replace the audio branch and add an audio link to the JSON response.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Replace the audio branch and add a caption field to the JSON response.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Generate a new image and add its caption to the JSON response.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Return a JSON object and replace the audio branch using this document as context.',
                'md',
                'prior-transcript',
                False,
            ),
            (
                'Return JSON, then replace the audio branch based on the attached document.',
                'md',
                'prior-transcript',
                False,
            ),
        )

        for prompt, extension, source_name, expected in cases:
            with self.subTest(prompt=prompt, extension=extension):
                request = detect_text_artifact_request(
                    prompt,
                    source_available=True,
                    source_extension=extension,
                    source_name=source_name,
                )
                self.assertEqual(request is not None, expected)
                if request:
                    self.assertEqual(request['source'], 'selected_source_edit')

    def test_detect_text_artifact_request_allows_explicit_save_updated_artifact_follow_up(self):
        request = detect_text_artifact_request('Change the font color to red and save the updated artifact.')

        self.assertIsNotNone(request)
        self.assertEqual(request['extension'], 'txt')
        self.assertEqual(request['source'], 'explicit_file_cue')

    def test_generated_text_artifact_self_claim_guard_detects_false_artifact_claim(self):
        self.assertTrue(generated_text_is_artifact_self_claim('[artifact: fancy.html]\nSaved locally.'))
        self.assertFalse(generated_text_is_artifact_self_claim('<!doctype html><h1>Hello</h1>'))

    def test_extract_text_artifact_payload_prefers_matching_code_block(self):
        request = detect_text_artifact_request('Create an index.html artifact with a hello page.')
        content = 'I created it for you.\n\n```html\n<!doctype html><h1>Hello</h1>\n```'

        self.assertEqual(
            extract_text_artifact_payload(content, request),
            '<!doctype html><h1>Hello</h1>',
        )

    def test_extract_text_artifact_payload_extracts_artifact_tag_after_thought_block(self):
        request = detect_text_artifact_request(
            'Create an index.html artifact with a minimal page that says "Signal Garden" in blue text.'
        )
        content = (
            '```thought\nPlanning the artifact.\n```\n\n'
            '<artifact identifier="index.html" type="text/html" title="Signal Garden Page">\n'
            '<!DOCTYPE html><html><body><h1 style="color: blue">Signal Garden</h1></body></html>\n'
            '</artifact>'
        )

        self.assertEqual(
            extract_text_artifact_payload(content, request),
            '<!DOCTYPE html><html><body><h1 style="color: blue">Signal Garden</h1></body></html>',
        )

    def test_extract_text_artifact_payload_promotes_real_code_block_over_thought_for_generic_save(self):
        request = detect_text_artifact_request('Change the font color to red and save the updated artifact.')
        content = (
            '```thought\nPlanning the artifact update.\n```\n\n'
            '```html title="index.html"\n'
            '<!doctype html><h1 style="color: red">Signal Garden</h1>\n'
            '```'
        )

        payloads = extract_text_artifact_payloads(content, [request])

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]['artifact_request']['extension'], 'html')
        self.assertEqual(
            payloads[0]['content'],
            '<!doctype html><h1 style="color: red">Signal Garden</h1>',
        )

    def test_extract_text_artifact_payloads_maps_multiple_matching_code_blocks(self):
        requests = detect_text_artifact_requests(
            'Create an HTML artifact for a landing page and a CSS artifact for its styles.'
        )
        content = (
            'Here are both files.\n\n'
            '```html\n<!doctype html><h1>Hello</h1>\n```\n\n'
            '```css\nbody { color: red; }\n```'
        )

        payloads = extract_text_artifact_payloads(content, requests)

        self.assertEqual(
            [(item['artifact_request']['extension'], item['content']) for item in payloads],
            [
                ('html', '<!doctype html><h1>Hello</h1>'),
                ('css', 'body { color: red; }'),
            ],
        )

    def test_instruction_echo_detector_flags_fence_language_materializer_phrases(self):
        detector = getattr(transports, 'text_artifact_content_is_materializer_instruction_echo', None)

        self.assertIsNotNone(detector)
        self.assertTrue(
            detector(
                'Materialize only the requested html artifact `index`.\n'
                'Return the complete file payload in one fenced code block with the matching language.\n'
                'Do not output planner JSON, request_ir, output_obligations, candidate_graph, or commentary.'
            )
        )
        self.assertTrue(detector('Use this fence language: html\nUse this fence language: html'))
        self.assertTrue(
            detector(
                'Use the fulfilled dependency evidence below as concrete runtime truth. '
                'When it names a generated artifact path, reference that exact saved path.\n\n'
                'Dependency evidence:\nphase-2: image artifact: /tmp/generated.png'
            )
        )
        self.assertFalse(detector('<!doctype html><html><body><h1>Hello</h1></body></html>'))

    def test_text_artifact_payload_rejects_instruction_echo_and_implausible_typed_bodies(self):
        html_request = {'extension': 'html', 'source_name': 'index', 'source': 'runtime_contract'}
        css_request = {'extension': 'css', 'source_name': 'styles', 'source': 'runtime_contract'}
        js_request = {'extension': 'js', 'source_name': 'app', 'source': 'runtime_contract'}

        self.assertEqual(
            extract_text_artifact_payloads(
                '```html\n'
                'Materialize only the requested html artifact `index`.\n'
                'Return the complete file payload in one fenced code block with the matching language.\n'
                'Do not output planner JSON, request_ir, output_obligations, candidate_graph, or commentary.\n'
                '```',
                [html_request],
            ),
            [],
        )
        self.assertEqual(
            extract_text_artifact_payloads(
                '```css\nUse this fence language: css\nUse this fence language: css\n```',
                [css_request],
            ),
            [],
        )
        self.assertEqual(
            extract_text_artifact_payloads(
                '```js\nUse this fence language: js\n```',
                [js_request],
            ),
            [],
        )

    def test_extract_text_artifact_payload_unwraps_raw_json_output_obligation(self):
        request = detect_text_artifact_request(
            'Erstelle eine kurze README für ein Mini-Projekt namens "Local Canvas". '
            'Gib sie als README.md Artifact aus, nicht nur im Chat.'
        )
        content = (
            '{\n'
            '  "route": "chat",\n'
            '  "output_obligations": [\n'
            '    {\n'
            '      "type": "artifact",\n'
            '      "name": "README.md",\n'
            '      "mime_type": "text/markdown",\n'
            '      "content": "# Local Canvas\\n\\nEin lokales Canvas-Projekt."\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        self.assertEqual(
            extract_text_artifact_payload(content, request),
            '# Local Canvas\n\nEin lokales Canvas-Projekt.',
        )

    def test_extract_text_artifact_payloads_unwraps_fenced_json_before_markdown_fallback(self):
        request = detect_text_artifact_request('Create a README.md artifact for Local Canvas.')
        content = (
            '```json\n'
            '{\n'
            '  "route": "chat",\n'
            '  "output_obligations": [\n'
            '    {\n'
            '      "type": "artifact",\n'
            '      "name": "README.md",\n'
            '      "mime_type": "text/markdown",\n'
            '      "content": "# Local Canvas\\n\\nUse it locally."\n'
            '    }\n'
            '  ]\n'
            '}\n'
            '```'
        )

        payloads = extract_text_artifact_payloads(content, [request])

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]['content'], '# Local Canvas\n\nUse it locally.')
        self.assertEqual(payloads[0]['artifact_request']['source'], 'json_output_obligation')

    def test_extract_text_artifact_payload_unwraps_nested_request_ir_content_payload(self):
        request = detect_text_artifact_request('Erstelle ein HTML-Artefakt für ein kleines Dashboard.')
        content = (
            '{\n'
            '  "request_ir": {\n'
            '    "output_obligations": [\n'
            '      {\n'
            '        "type": "artifact",\n'
            '        "name": "dashboard.html",\n'
            '        "mime_type": "text/html",\n'
            '        "content_payload": "<!doctype html><h1>Station</h1>"\n'
            '      }\n'
            '    ]\n'
            '  }\n'
            '}'
        )

        self.assertEqual(
            extract_text_artifact_payload(content, request),
            '<!doctype html><h1>Station</h1>',
        )

    def test_text_artifact_payload_rejects_instruction_echo_json_content_values(self):
        request = detect_text_artifact_request('Create an index.html artifact with a hello page.')
        content = (
            '{\n'
            '  "output_obligations": [\n'
            '    {\n'
            '      "type": "artifact",\n'
            '      "name": "index.html",\n'
            '      "mime_type": "text/html",\n'
            '      "content": "Materialize only the requested html artifact `index`.\\n'
            'Return the complete file payload in one fenced code block with the matching language."\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        self.assertEqual(extract_text_artifact_payloads(content, [request]), [])

    def test_extract_text_artifact_payload_blocks_control_json_without_real_payload(self):
        request = {
            'extension': 'json',
            'source': 'explicit_format_file_cue',
            'source_name': 'generated-json',
        }
        content = (
            '{\n'
            '  "request_ir": {\n'
            '    "output_obligations": [\n'
            '      {\n'
            '        "output_type": "document",\n'
            '        "name": "dashboard.html",\n'
            '        "artifact_prompt": "Generate an HTML dashboard.",\n'
            '        "stage_direction": "Materialize the requested dashboard."\n'
            '      }\n'
            '    ]\n'
            '  },\n'
            '  "candidate_graph": {"type_counts": {"output": 1}},\n'
            '  "promotion_review": {"promoted_count": 1}\n'
            '}'
        )

        self.assertEqual(extract_text_artifact_payloads(content, [request]), [])

    def test_json_text_artifact_payload_keeps_user_data_and_blocks_control_key_parity(self):
        request = {
            'extension': 'json',
            'source': 'explicit_extension',
            'source_name': 'pricing',
        }
        pricing = (
            '[\n'
            '  {"material": "Titanium Base", "price_chf": 24000, "image": "../images/titanium.png"}\n'
            ']'
        )

        self.assertEqual(
            extract_text_artifact_payloads(f'```json\n{pricing}\n```', [request]),
            [{'artifact_request': request, 'content': pricing}],
        )

        control_payloads = (
            '{"request_phase_graph": {"phases": []}}',
            '{"decision_contract": {"route": "chat"}}',
            '{"user_facing_response": "I will create the file next."}',
            '[{"request_phase_graph": {"phases": []}}]',
            '{"Request_Phase_Graph": {"phases": []}}',
            '{"result": {"request_ir": {"output_obligations": []}}}',
            '{"response": {"request_phase_graph": {"phases": []}}}',
            '{"data": [{"decision_contract": {"route": "chat"}}]}',
        )
        for content in control_payloads:
            with self.subTest(content=content):
                self.assertEqual(extract_text_artifact_payloads(content, [request]), [])

        ordinary_reserved_word_value = '[{"name": "request_phase_graph", "price_chf": 24000}]'
        self.assertEqual(
            extract_text_artifact_payloads(ordinary_reserved_word_value, [request]),
            [{'artifact_request': request, 'content': ordinary_reserved_word_value}],
        )

    def test_extract_text_artifact_payload_blocks_clarification_text(self):
        request = detect_text_artifact_request('Generate me this html file as artifact')
        content = (
            'I cannot proceed because the content for "this" HTML file was not included. '
            'Please provide the HTML code or a description.'
        )

        self.assertIsNone(extract_text_artifact_payload(content, request))

    def test_chat_dispatch_persists_explicit_text_artifact(self):
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Create an index.html artifact with a hello page.',
            user_prompt='Create an index.html artifact with a hello page.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        artifacts = InferArtifacts()

        def persist_text_artifact_locally(content, **kwargs):
            self.assertEqual(content, '<!doctype html><h1>Hello</h1>')
            self.assertEqual(kwargs['extension'], 'html')
            self.assertEqual(kwargs['source_name'], 'index')
            return '/tmp/artifacts/documents/index.html'

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_chat': lambda *_args, **_kwargs: {
                    'content': 'Here is the file.\n\n```html\n<!doctype html><h1>Hello</h1>\n```',
                },
                'persist_text_artifact_locally': persist_text_artifact_locally,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'chat')
        self.assertIn('Here is the file.', payload['content'])
        self.assertEqual(payload['saved_text_path'], '/tmp/artifacts/documents/index.html')
        self.assertEqual(payload['document_output_kind'], 'document')
        self.assertEqual(payload['text_artifact_request']['extension'], 'html')

    def test_chat_dispatch_persists_structured_text_artifact_request(self):
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Materialize the requested branch payload.',
            user_prompt='Materialize the requested branch payload.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            text_artifact_requests=[
                {'extension': 'md', 'source_name': 'safety-protocol', 'source': 'runtime_contract'}
            ],
        )
        artifacts = InferArtifacts()

        def persist_text_artifact_locally(content, **kwargs):
            self.assertEqual(content, '# Safety Protocol\n\n- Check mooring tension.')
            self.assertEqual(kwargs['extension'], 'md')
            self.assertEqual(kwargs['source_name'], 'safety-protocol')
            return '/tmp/artifacts/documents/safety-protocol.md'

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_chat': lambda *_args, **_kwargs: {
                    'content': '```markdown\n# Safety Protocol\n\n- Check mooring tension.\n```',
                },
                'persist_text_artifact_locally': persist_text_artifact_locally,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_text_path'], '/tmp/artifacts/documents/safety-protocol.md')
        self.assertEqual(payload['text_artifact_request']['source'], 'runtime_contract')

    def test_chat_dispatch_does_not_persist_an_incomplete_multi_file_set(self):
        requests = [
            {'extension': 'html', 'source_name': 'index', 'source': 'runtime_contract'},
            {'extension': 'css', 'source_name': 'styles', 'source': 'runtime_contract'},
            {'extension': 'html', 'source_name': 'configurator', 'source': 'runtime_contract'},
            {'extension': 'json', 'source_name': 'pricing', 'source': 'runtime_contract'},
        ]
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Materialize the complete four-file contract.',
            user_prompt='Materialize the complete four-file contract.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            text_artifact_requests=requests,
        )
        artifacts = InferArtifacts()
        persisted: list[str] = []

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_chat': lambda *_args, **_kwargs: {
                    'content': (
                        '```html\n<!doctype html><title>Atelier</title>\n```\n'
                        '```css\nbody { color: #111; }\n```\n'
                        '```json\n{"currency": "CHF"}\n```'
                    ),
                },
                'persist_text_artifact_locally': (
                    lambda content, **_kwargs: persisted.append(content)
                ),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(persisted, [])
        self.assertNotIn('saved_text_path', payload)
        self.assertNotIn('saved_text_artifacts', payload)

    def test_chat_dispatch_persists_selected_html_source_edit_from_file_context(self):
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Change the font color to red and save the updated artifact.',
            user_prompt='Change the font color to red and save the updated artifact.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        artifacts = InferArtifacts(temp_path=object(), file_kind='text', file_name='index.html')

        def persist_text_artifact_locally(content, **kwargs):
            self.assertIn('color: red', content)
            self.assertEqual(kwargs['extension'], 'html')
            self.assertEqual(kwargs['source_name'], 'index')
            return '/tmp/artifacts/documents/index.html'

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_chat': lambda *_args, **_kwargs: {
                    'content': (
                        '```thought\nPlanning the update.\n```\n\n'
                        '<artifact identifier="index.html" type="text/html">\n'
                        '<!doctype html><style>h1 { color: red; }</style><h1>Signal Garden</h1>\n'
                        '</artifact>'
                    ),
                },
                'persist_text_artifact_locally': persist_text_artifact_locally,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_text_path'], '/tmp/artifacts/documents/index.html')
        self.assertEqual(payload['text_artifact_request']['source'], 'selected_source_edit')
        self.assertEqual(payload['text_artifact_request']['extension'], 'html')

    def test_chat_dispatch_passes_selected_source_path_to_text_artifact_persistence(self):
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Change the font color to red and save the updated artifact.',
            user_prompt='Change the font color to red and save the updated artifact.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        source_path = Path('/tmp/artifacts/documents/index.html')
        artifacts = InferArtifacts(temp_path=source_path, file_kind='text', file_name='index.html')

        def persist_text_artifact_locally(content, **kwargs):
            self.assertIn('color: red', content)
            self.assertEqual(kwargs['target_path'], str(source_path))
            return str(source_path)

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_chat': lambda *_args, **_kwargs: {
                    'content': '<artifact identifier="index.html" type="text/html">'
                    '<!doctype html><style>h1 { color: red; }</style><h1>Signal Garden</h1>'
                    '</artifact>',
                },
                'persist_text_artifact_locally': persist_text_artifact_locally,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_text_path'], str(source_path))
        self.assertEqual(payload['text_artifact_request']['target_path'], str(source_path))

    def test_persist_text_artifact_locally_updates_target_inside_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / 'artifacts' / 'documents'
            target_path = output_dir / 'styles.css'
            output_dir.mkdir(parents=True)
            target_path.write_text('body { color: blue; }\n', encoding='utf-8')

            result = persist_text_artifact_locally(
                'body { color: red; }',
                model_name='gemma4:26b',
                source_name='styles',
                mode='chat_text_artifact',
                extension='css',
                output_dir=output_dir,
                target_path=str(target_path),
            )

            self.assertEqual(result, str(target_path.resolve()))
            self.assertEqual(target_path.read_text(encoding='utf-8'), 'body { color: red; }\n')
            self.assertEqual(list(output_dir.glob('*.css')), [target_path])

    def test_chat_dispatch_keeps_plain_chat_inline_only(self):
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Tell me hello.',
            user_prompt='Tell me hello.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'ollama_chat': lambda *_args, **_kwargs: {'content': 'hello'},
                'persist_text_artifact_locally': lambda *_args, **_kwargs: self.fail('plain chat should not persist'),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['content'], 'hello')
        self.assertNotIn('saved_text_path', payload)

    def test_chat_dispatch_does_not_persist_artifact_self_claim(self):
        ctx = InferContext(
            instance_id='chat-1',
            backend='ollama',
            capability='chat',
            model_name='gemma4:26b',
            port=11437,
            prompt='Create an index.html artifact.',
            user_prompt='Create an index.html artifact.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'ollama_chat': lambda *_args, **_kwargs: {
                    'content': 'Artifact Created. Saved locally. [artifact: index.html]',
                },
                'persist_text_artifact_locally': lambda *_args, **_kwargs: self.fail('self-claims should not persist'),
            },
        )

        self.assertEqual(status, 200)
        self.assertNotIn('saved_text_path', payload)

    def test_speech_dispatch_returns_transcription_payload(self):
        ctx = InferContext(
            instance_id='whisper-1',
            backend='mlx',
            capability='speech_to_text',
            model_name='mlx-community/whisper-large-v3-mlx',
            port=11501,
            prompt='',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            language='en',
        )
        artifacts = InferArtifacts(temp_path=object(), file_kind='audio')

        def whisper_transcribe(_port, _path, task='transcribe', language=None):
            self.assertEqual(task, 'transcribe')
            self.assertEqual(language, 'en')
            return {'text': 'hello world'}

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'whisper_transcribe': whisper_transcribe,
                'persist_transcript_text_locally': lambda content, **_kwargs: '/tmp/artifacts/transcripts/hello.md',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'speech_to_text')
        self.assertEqual(payload['content'], 'hello world')
        self.assertEqual(payload['saved_text_path'], '/tmp/artifacts/transcripts/hello.md')

    def test_speech_dispatch_falls_back_to_segments_when_text_is_empty(self):
        ctx = InferContext(
            instance_id='whisper-2',
            backend='mlx',
            capability='speech_to_text',
            model_name='mlx-community/whisper-large-v3-mlx',
            port=11501,
            prompt='',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            language='en',
        )
        artifacts = InferArtifacts(temp_path=object(), file_kind='audio', file_name='sample.mp3')

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'whisper_transcribe': lambda *_args, **_kwargs: {
                    'text': '',
                    'segments': [{'text': 'hello'}, {'text': 'world'}],
                },
                'persist_transcript_text_locally': lambda content, **_kwargs: '/tmp/artifacts/transcripts/segments.md',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['content'], 'hello\nworld')
        self.assertEqual(payload['saved_text_path'], '/tmp/artifacts/transcripts/segments.md')

    def test_text_to_speech_dispatch_returns_saved_audio_payload(self):
        ctx = InferContext(
            instance_id='tts-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt='Guten Tag aus Ollmo.',
            user_prompt='Guten Tag aus Ollmo.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            voice='Chelsie',
            instruct='Warm, calm, elegant German narration.',
            response_format='wav',
            speed=0.95,
            pitch=1.1,
            lang_code='de',
        )
        artifacts = InferArtifacts()

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertEqual(prompt, 'Guten Tag aus Ollmo.')
            self.assertEqual(kwargs['voice'], 'Chelsie')
            self.assertEqual(kwargs['instruct'], 'Warm, calm, elegant German narration.')
            self.assertEqual(kwargs['response_format'], 'wav')
            self.assertEqual(kwargs['speed'], 0.95)
            self.assertEqual(kwargs['pitch'], 1.1)
            self.assertEqual(kwargs['lang_code'], 'german')
            self.assertEqual(kwargs['max_tokens'], 256)
            self.assertEqual(kwargs['temperature'], 0.9)
            self.assertEqual(kwargs['top_p'], 1.0)
            self.assertEqual(kwargs['top_k'], 50)
            self.assertEqual(kwargs['repetition_penalty'], 1.05)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        def persist_audio_bytes_locally(audio_bytes, model_name, **kwargs):
            self.assertEqual(audio_bytes, b'RIFFfakewav')
            self.assertEqual(model_name, 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16')
            self.assertEqual(kwargs['response_format'], 'wav')
            self.assertEqual(kwargs['content_type'], 'audio/wav')
            return '/tmp/artifacts/audio/qwen3-tts.wav'

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': persist_audio_bytes_locally,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'text_to_speech')
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/qwen3-tts.wav')
        self.assertEqual(payload['lang_code'], 'german')
        self.assertEqual(
            payload['lang_code_source'],
            'explicit_qwen3_alias_canonicalized',
        )
        self.assertEqual(payload['tts_generation_budget']['max_tokens'], 256)
        self.assertEqual(
            payload['tts_generation_budget']['policy_id'],
            'qwen3_tts_adaptive_audio_tokens_v2',
        )
        self.assertEqual(
            payload['tts_sampling_profile'],
            {
                'kind': 'ollmo.tts_sampling_profile',
                'version': 1,
                'policy_id': 'qwen3_tts_model_native_sampling_v1',
                'model_family': 'qwen3_tts',
                'source': 'mlx_audio_qwen3_tts_model_defaults',
                'temperature': 0.9,
                'top_p': 1.0,
                'top_k': 50,
                'repetition_penalty': 1.05,
            },
        )

    def test_text_to_speech_runs_backend_persistence_then_integrity_analysis(self):
        ctx = InferContext(
            instance_id='tts-order-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt='Guten Tag aus Ollmo.',
            user_prompt='Guten Tag aus Ollmo.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        call_order = []
        backend_audio_bytes = b'RIFFbackend-bytes'
        integrity_evidence = {
            'kind': 'ollmo.tts_audio_integrity_evidence',
            'status': 'passed',
            'materialization_eligible': True,
        }

        def mlx_audio_speech(_port, _model_name, _prompt, **_kwargs):
            call_order.append('backend')
            return {
                'audio_bytes': backend_audio_bytes,
                'content_type': 'audio/wav',
                'result': {'bytes': len(backend_audio_bytes)},
            }

        def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
            self.assertEqual(call_order, ['backend'])
            self.assertIs(audio_bytes, backend_audio_bytes)
            call_order.append('persist')
            return '/tmp/artifacts/audio/ordered.wav'

        def build_integrity_evidence(
            saved_audio_path,
            spoken_text,
            *,
            source_sha256=None,
            generation_budget=None,
            model_family=None,
            tts_model_type=None,
        ):
            self.assertEqual(call_order, ['backend', 'persist'])
            self.assertEqual(saved_audio_path, '/tmp/artifacts/audio/ordered.wav')
            self.assertEqual(spoken_text, 'Guten Tag aus Ollmo.')
            self.assertEqual(
                source_sha256,
                hashlib.sha256(spoken_text.encode('utf-8')).hexdigest(),
            )
            self.assertEqual(model_family, 'qwen3_tts')
            self.assertEqual(tts_model_type, 'base')
            self.assertEqual(generation_budget['generation_scope'], 'single_sequence')
            self.assertEqual(generation_budget['max_tokens'], 256)
            call_order.append('integrity')
            return integrity_evidence

        with patch(
            'ollmo_core.inference.build_tts_audio_integrity_evidence',
            side_effect=build_integrity_evidence,
        ):
            payload, status = dispatch_infer_request(
                ctx,
                InferArtifacts(),
                {
                    'mlx_audio_speech': mlx_audio_speech,
                    'persist_audio_bytes_locally': persist_audio_bytes_locally,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(call_order, ['backend', 'persist', 'integrity'])
        self.assertIs(payload['tts_audio_integrity_evidence'], integrity_evidence)
        self.assertEqual(
            payload['tts_semantic_source']['tts_source_text'],
            'Guten Tag aus Ollmo.',
        )

    def test_long_qwen_voice_design_synthesizes_verified_chunks_and_persists_once(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )
        ctx = InferContext(
            instance_id='tts-long-chunked-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt=passage,
            user_prompt=passage,
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Warm, steady English narration.',
            response_format='wav',
            lang_code='english',
            tts_model_type='voice_design',
        )
        backend_calls = []
        persistence_calls = []

        def mlx_audio_speech(_port, _model_name, chunk_text, **kwargs):
            backend_calls.append((chunk_text, dict(kwargs)))
            expected_budget = build_qwen3_tts_generation_budget(chunk_text)
            self.assertEqual(kwargs['max_tokens'], expected_budget['max_tokens'])
            self.assertEqual(kwargs['instruct'], 'Warm, steady English narration.')
            self.assertEqual(kwargs['lang_code'], 'english')
            return {
                'audio_bytes': _pcm_wav_bytes(5.0),
                'content_type': 'audio/wav',
                'result': {'bytes': 80044},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = Path(temp_dir) / 'joined.wav'

            def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
                persistence_calls.append(bytes(audio_bytes))
                saved_path.write_bytes(audio_bytes)
                return str(saved_path)

            payload, status = dispatch_infer_request(
                ctx,
                InferArtifacts(),
                {
                    'mlx_audio_speech': mlx_audio_speech,
                    'persist_audio_bytes_locally': persist_audio_bytes_locally,
                },
            )

        self.assertEqual(status, 200)
        self.assertGreater(len(backend_calls), 1)
        self.assertEqual(len(persistence_calls), 1)
        self.assertEqual(
            ' '.join(' '.join(call[0].split()) for call in backend_calls),
            ' '.join(passage.split()),
        )
        self.assertEqual(payload['tts_generation_budget']['generation_scope'], 'chunked_sequence')
        self.assertEqual(payload['tts_generation_budget']['chunk_count'], len(backend_calls))
        self.assertEqual(payload['tts_audio_integrity_evidence']['status'], 'passed')
        chunking = payload['tts_audio_integrity_evidence']['chunking_evidence']
        self.assertEqual(chunking['status'], 'passed')
        self.assertEqual(chunking['completed_chunk_count'], len(backend_calls))
        self.assertEqual(chunking['join_evidence']['chunk_count'], len(backend_calls))
        self.assertEqual(
            payload['tts_semantic_source']['tts_source_text_sha256'],
            hashlib.sha256(passage.encode()).hexdigest(),
        )

    def test_long_qwen_voice_design_retries_only_the_exhausted_chunk_once(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )
        planned_chunks = build_qwen3_tts_chunk_plan(passage)['chunks']
        exhausted_chunk = planned_chunks[2]['text']
        ctx = InferContext(
            instance_id='tts-long-chunk-recovery-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt=passage,
            user_prompt=passage,
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Warm, steady English narration.',
            response_format='wav',
            lang_code='english',
            tts_model_type='voice_design',
        )
        backend_calls = []
        persistence_calls = []
        exhausted_attempt_count = 0

        def mlx_audio_speech(_port, _model_name, chunk_text, **kwargs):
            nonlocal exhausted_attempt_count
            backend_calls.append((chunk_text, dict(kwargs)))
            if chunk_text == exhausted_chunk:
                exhausted_attempt_count += 1
                if exhausted_attempt_count == 1:
                    duration_seconds = kwargs['max_tokens'] / 12.5
                    return {
                        'audio_bytes': _pcm_wav_segment_bytes(
                            [(1.5, True), (duration_seconds - 1.5, False)]
                        ),
                        'content_type': 'audio/wav',
                        'result': {'exact_generation_limit': True},
                    }
            return {
                'audio_bytes': _pcm_wav_bytes(5.0),
                'content_type': 'audio/wav',
                'result': {'exact_generation_limit': False},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = Path(temp_dir) / 'joined-recovered.wav'

            def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
                persistence_calls.append(bytes(audio_bytes))
                saved_path.write_bytes(audio_bytes)
                return str(saved_path)

            payload, status = dispatch_infer_request(
                ctx,
                InferArtifacts(),
                {
                    'mlx_audio_speech': mlx_audio_speech,
                    'persist_audio_bytes_locally': persist_audio_bytes_locally,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(len(backend_calls), len(planned_chunks) + 1)
        self.assertEqual(
            [item[0] for item in backend_calls],
            [
                planned_chunks[0]['text'],
                planned_chunks[1]['text'],
                exhausted_chunk,
                exhausted_chunk,
                planned_chunks[3]['text'],
            ],
        )
        initial_kwargs = backend_calls[2][1]
        recovery_kwargs = backend_calls[3][1]
        self.assertEqual(initial_kwargs['max_tokens'], 269)
        self.assertEqual(recovery_kwargs['max_tokens'], 404)
        self.assertEqual(
            {key: value for key, value in initial_kwargs.items() if key != 'max_tokens'},
            {key: value for key, value in recovery_kwargs.items() if key != 'max_tokens'},
        )
        self.assertEqual(len(persistence_calls), 1)
        integrity = payload['tts_audio_integrity_evidence']
        self.assertEqual(integrity['status'], 'passed')
        chunking = integrity['chunking_evidence']
        self.assertEqual(chunking['status'], 'passed')
        self.assertEqual(chunking['backend_call_count'], len(planned_chunks) + 1)
        self.assertEqual(chunking['passed_chunk_count'], len(planned_chunks))
        self.assertEqual(chunking['recovered_chunk_count'], 1)
        self.assertEqual(chunking['generation_limit_recovery_attempt_count'], 1)
        recovered = chunking['chunks'][2]
        self.assertEqual(recovered['status'], 'recovered')
        self.assertEqual(recovered['attempt_count'], 2)
        self.assertEqual(
            [attempt['role'] for attempt in recovered['attempts']],
            ['initial', 'generation_limit_recovery'],
        )
        self.assertEqual(
            [attempt['selected'] for attempt in recovered['attempts']],
            [False, True],
        )
        self.assertEqual(
            recovered['attempts'][0]['integrity_evidence']['reason_code'],
            'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
        )
        self.assertEqual(
            recovered['attempts'][0]['integrity_evidence']['defect_codes'],
            [
                'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
                'TTS_AUDIO_EXCESSIVE_TRAILING_SILENCE',
                'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
            ],
        )
        self.assertEqual(
            recovered['attempts'][1]['integrity_evidence']['reason_code'],
            'TTS_AUDIO_INTEGRITY_PASSED',
        )
        recovery = recovered['generation_limit_recovery']
        self.assertEqual(
            recovery['policy_id'],
            'qwen3_tts_single_sequence_generation_limit_retry_v2',
        )
        self.assertEqual(recovery['trigger_reason_code'], 'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED')
        self.assertEqual(recovery['status'], 'passed')
        self.assertEqual(recovery['initial_max_tokens'], 269)
        self.assertEqual(recovery['recovery_max_tokens'], 404)
        self.assertEqual(
            recovery['trigger_primary_reason_code'],
            'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
        )
        self.assertIn(
            'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
            recovery['trigger_defect_codes'],
        )
        self.assertEqual(
            payload['tts_semantic_source']['tts_source_text_sha256'],
            hashlib.sha256(passage.encode()).hexdigest(),
        )

    def test_long_qwen_voice_design_blocks_when_the_single_recovery_also_exhausts(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )
        planned_chunks = build_qwen3_tts_chunk_plan(passage)['chunks']
        exhausted_chunk = planned_chunks[2]['text']
        ctx = InferContext(
            instance_id='tts-long-chunk-recovery-failure-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt=passage,
            user_prompt=passage,
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Warm, steady English narration.',
            response_format='wav',
            lang_code='english',
            tts_model_type='voice_design',
        )
        backend_calls = []
        persistence_calls = []

        def mlx_audio_speech(_port, _model_name, chunk_text, **kwargs):
            backend_calls.append((chunk_text, dict(kwargs)))
            duration_seconds = (
                kwargs['max_tokens'] / 12.5
                if chunk_text == exhausted_chunk
                else 5.0
            )
            return {
                'audio_bytes': _pcm_wav_bytes(duration_seconds),
                'content_type': 'audio/wav',
                'result': {'bytes': 1},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = Path(temp_dir) / 'failed-recovery.wav'

            def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
                persistence_calls.append(bytes(audio_bytes))
                saved_path.write_bytes(audio_bytes)
                return str(saved_path)

            payload, status = dispatch_infer_request(
                ctx,
                InferArtifacts(),
                {
                    'mlx_audio_speech': mlx_audio_speech,
                    'persist_audio_bytes_locally': persist_audio_bytes_locally,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(len(backend_calls), 4)
        self.assertEqual(
            [item[0] for item in backend_calls],
            [
                planned_chunks[0]['text'],
                planned_chunks[1]['text'],
                exhausted_chunk,
                exhausted_chunk,
            ],
        )
        self.assertEqual(backend_calls[2][1]['max_tokens'], 269)
        self.assertEqual(backend_calls[3][1]['max_tokens'], 404)
        self.assertEqual(len(persistence_calls), 1)
        integrity = payload['tts_audio_integrity_evidence']
        self.assertEqual(integrity['status'], 'failed')
        self.assertFalse(integrity['materialization_eligible'])
        self.assertEqual(
            integrity['reason_code'],
            'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
        )
        chunking = integrity['chunking_evidence']
        self.assertEqual(chunking['status'], 'failed')
        self.assertEqual(chunking['failed_chunk_index'], 3)
        self.assertEqual(chunking['passed_chunk_count'], 2)
        self.assertEqual(chunking['backend_call_count'], 4)
        failed = chunking['chunks'][2]
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['attempt_count'], 2)
        self.assertEqual(
            [
                attempt['integrity_evidence']['reason_code']
                for attempt in failed['attempts']
            ],
            [
                'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
                'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
            ],
        )
        self.assertEqual(failed['generation_limit_recovery']['status'], 'failed')

    def test_generation_limit_retry_applies_to_supported_qwen_model_types(self):
        passage = 'The lighthouse welcomes every boat returning safely before dawn.'
        for label, model_name, model_type in (
            (
                'base',
                'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
                'base',
            ),
            (
                'custom-voice',
                'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
                'custom_voice',
            ),
        ):
            with self.subTest(model_type=label), tempfile.TemporaryDirectory() as temp_dir:
                ctx = InferContext(
                    instance_id=f'tts-no-generation-limit-retry-{label}',
                    backend='mlx',
                    capability='text_to_speech',
                    model_name=model_name,
                    port=11504,
                    prompt=passage,
                    user_prompt=passage,
                    infer_timeout_sec=1200,
                    pdf_page_timeout_sec=240,
                    pdf_max_image_side=2400,
                    pdf_synthesize=False,
                    voice=(
                        'serena'
                        if model_type == 'custom_voice'
                        else 'Chelsie'
                    ),
                    instruct='Warm, steady English narration.',
                    response_format='wav',
                    lang_code='english',
                    tts_model_type=model_type,
                )
                backend_calls = []
                saved_path = Path(temp_dir) / f'{label}-recovered.wav'

                def mlx_audio_speech(_port, _model_name, chunk_text, **kwargs):
                    backend_calls.append((chunk_text, dict(kwargs)))
                    duration_seconds = (
                        kwargs['max_tokens'] / 12.5
                        if len(backend_calls) == 1
                        else 4.0
                    )
                    return {
                        'audio_bytes': _pcm_wav_bytes(duration_seconds),
                        'content_type': 'audio/wav',
                        'result': {'bytes': 1},
                    }

                def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
                    saved_path.write_bytes(audio_bytes)
                    return str(saved_path)

                payload, status = dispatch_infer_request(
                    ctx,
                    InferArtifacts(),
                    {
                        'mlx_audio_speech': mlx_audio_speech,
                        'persist_audio_bytes_locally': persist_audio_bytes_locally,
                    },
                )

                self.assertEqual(status, 200)
                self.assertEqual(len(backend_calls), 2)
                self.assertLess(
                    backend_calls[0][1]['max_tokens'],
                    backend_calls[1][1]['max_tokens'],
                )
                integrity = payload['tts_audio_integrity_evidence']
                self.assertEqual(integrity['status'], 'passed')
                recovery = integrity['generation_limit_recovery']
                self.assertEqual(recovery['attempt_count'], 2)
                self.assertEqual(
                    recovery['policy_id'],
                    'qwen3_tts_single_sequence_generation_limit_retry_v2',
                )
                self.assertEqual(
                    recovery['tts_model_type'],
                    model_type,
                )

    def test_long_qwen_voice_design_stops_after_failed_chunk_and_stays_blocked(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )
        ctx = InferContext(
            instance_id='tts-long-chunk-failure-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt=passage,
            user_prompt=passage,
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Warm, steady English narration.',
            response_format='wav',
            lang_code='english',
            tts_model_type='voice_design',
        )
        backend_calls = []
        persistence_calls = []

        def mlx_audio_speech(_port, _model_name, chunk_text, **_kwargs):
            backend_calls.append(chunk_text)
            raw = _pcm_wav_bytes(5.0) if len(backend_calls) == 1 else b'RIFF-broken'
            return {
                'audio_bytes': raw,
                'content_type': 'audio/wav',
                'result': {'bytes': len(raw)},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = Path(temp_dir) / 'failed-chunk.wav'

            def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
                persistence_calls.append(bytes(audio_bytes))
                saved_path.write_bytes(audio_bytes)
                return str(saved_path)

            payload, status = dispatch_infer_request(
                ctx,
                InferArtifacts(),
                {
                    'mlx_audio_speech': mlx_audio_speech,
                    'persist_audio_bytes_locally': persist_audio_bytes_locally,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(len(backend_calls), 2)
        self.assertEqual(len(persistence_calls), 1)
        integrity = payload['tts_audio_integrity_evidence']
        self.assertEqual(integrity['status'], 'failed')
        self.assertFalse(integrity['materialization_eligible'])
        self.assertEqual(integrity['reason_code'], 'TTS_AUDIO_WAV_UNREADABLE')
        self.assertIn('TTS_AUDIO_WAV_UNREADABLE', integrity['defect_codes'])
        chunking = integrity['chunking_evidence']
        self.assertEqual(chunking['status'], 'failed')
        self.assertEqual(chunking['failed_chunk_index'], 2)
        self.assertEqual(chunking['completed_chunk_count'], 2)

    def test_long_qwen_base_uses_the_same_verified_chunk_pipeline(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )
        ctx = InferContext(
            instance_id='tts-long-base-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt=passage,
            user_prompt=passage,
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            voice='Chelsie',
            response_format='wav',
            lang_code='english',
        )
        backend_calls = []
        persistence_calls = []

        def mlx_audio_speech(_port, _model_name, chunk_text, **kwargs):
            backend_calls.append((chunk_text, dict(kwargs)))
            self.assertEqual(
                kwargs['max_tokens'],
                build_qwen3_tts_generation_budget(chunk_text)['max_tokens'],
            )
            self.assertEqual(kwargs['voice'], 'Chelsie')
            return {
                'audio_bytes': _pcm_wav_bytes(5.0),
                'content_type': 'audio/wav',
                'result': {'bytes': 80044},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_path = Path(temp_dir) / 'joined-base.wav'

            def persist_audio_bytes_locally(audio_bytes, _model_name, **_kwargs):
                persistence_calls.append(bytes(audio_bytes))
                saved_path.write_bytes(audio_bytes)
                return str(saved_path)

            payload, status = dispatch_infer_request(
                ctx,
                InferArtifacts(),
                {
                    'mlx_audio_speech': mlx_audio_speech,
                    'persist_audio_bytes_locally': persist_audio_bytes_locally,
                },
            )

        self.assertEqual(status, 200)
        self.assertGreater(len(backend_calls), 1)
        self.assertEqual(len(persistence_calls), 1)
        self.assertEqual(payload['tts_model_type'], 'base')
        self.assertEqual(
            payload['tts_generation_budget']['generation_scope'],
            'chunked_sequence',
        )
        self.assertEqual(payload['tts_audio_integrity_evidence']['status'], 'passed')
        self.assertEqual(
            payload['tts_audio_integrity_evidence']['chunking_evidence']['status'],
            'passed',
        )

    def test_text_to_speech_qwen_canonicalizes_explicit_language_aliases(self):
        for lang_code, expected_lang_code in (('en', 'english'), ('de', 'german')):
            with self.subTest(lang_code=lang_code):
                observed_kwargs = {}
                ctx = InferContext(
                    instance_id=f'tts-language-alias-{lang_code}',
                    backend='mlx',
                    capability='text_to_speech',
                    model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
                    port=11504,
                    prompt='A short language alias test.',
                    user_prompt='',
                    infer_timeout_sec=1200,
                    pdf_page_timeout_sec=240,
                    pdf_max_image_side=2400,
                    pdf_synthesize=False,
                    lang_code=lang_code,
                )

                def mlx_audio_speech(_port, _model_name, _prompt, **kwargs):
                    observed_kwargs.update(kwargs)
                    return {
                        'audio_bytes': b'RIFFfakewav',
                        'content_type': 'audio/wav',
                        'result': {'bytes': 11},
                    }

                payload, status = dispatch_infer_request(
                    ctx,
                    InferArtifacts(),
                    {
                        'mlx_audio_speech': mlx_audio_speech,
                        'persist_audio_bytes_locally': (
                            lambda *_args, **_kwargs: '/tmp/artifacts/audio/alias.wav'
                        ),
                    },
                )

                self.assertEqual(status, 200)
                self.assertEqual(observed_kwargs['lang_code'], expected_lang_code)
                self.assertEqual(payload['lang_code'], expected_lang_code)
                self.assertEqual(
                    payload['lang_code_source'],
                    'explicit_qwen3_alias_canonicalized',
                )
                self.assertEqual(
                    payload['tts_semantic_source']['lang_code'],
                    expected_lang_code,
                )

    def test_text_to_speech_qwen_preserves_canonical_auto_and_unsupported_languages(self):
        cases = (
            ('english', 'english', 'explicit'),
            ('English', 'English', 'explicit'),
            ('auto', 'auto', 'qwen3_model_default'),
            ('elvish', 'elvish', 'explicit'),
        )
        for lang_code, expected_lang_code, expected_source in cases:
            with self.subTest(lang_code=lang_code):
                observed_kwargs = {}
                ctx = InferContext(
                    instance_id=f'tts-language-pass-through-{lang_code.lower()}',
                    backend='mlx',
                    capability='text_to_speech',
                    model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
                    port=11504,
                    prompt='Hello.',
                    user_prompt='',
                    infer_timeout_sec=1200,
                    pdf_page_timeout_sec=240,
                    pdf_max_image_side=2400,
                    pdf_synthesize=False,
                    lang_code=lang_code,
                )

                def mlx_audio_speech(_port, _model_name, _prompt, **kwargs):
                    observed_kwargs.update(kwargs)
                    return {
                        'audio_bytes': b'RIFFfakewav',
                        'content_type': 'audio/wav',
                        'result': {'bytes': 11},
                    }

                payload, status = dispatch_infer_request(
                    ctx,
                    InferArtifacts(),
                    {
                        'mlx_audio_speech': mlx_audio_speech,
                        'persist_audio_bytes_locally': (
                            lambda *_args, **_kwargs: '/tmp/artifacts/audio/pass-through.wav'
                        ),
                    },
                )

                self.assertEqual(status, 200)
                self.assertEqual(observed_kwargs['lang_code'], expected_lang_code)
                self.assertEqual(payload['lang_code'], expected_lang_code)
                self.assertEqual(payload['lang_code_source'], expected_source)

    def test_text_to_speech_non_qwen_preserves_explicit_language_alias(self):
        observed_kwargs = {}
        ctx = InferContext(
            instance_id='tts-language-non-qwen',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/kitten-tts-mini-0.8-bf16',
            port=11504,
            prompt='Hello from Ollmo.',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            lang_code='en',
            tts_model_type='kitten_tts',
            tts_speakers=['Bella'],
        )

        def mlx_audio_speech(_port, _model_name, _prompt, **kwargs):
            observed_kwargs.update(kwargs)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': (
                    lambda *_args, **_kwargs: '/tmp/artifacts/audio/non-qwen.wav'
                ),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(observed_kwargs['lang_code'], 'en')
        self.assertEqual(payload['lang_code'], 'en')
        self.assertEqual(payload['lang_code_source'], 'explicit')
        self.assertNotIn('tts_generation_budget', payload)

    def test_text_to_speech_infers_supported_language_from_spoken_text(self):
        ctx = InferContext(
            instance_id='tts-language-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt=(
                'Es war einmal ein kleiner Fuchs, der in einem alten Wald lebte. '
                'Eines Morgens entdeckte er ein silbernes Licht und wusste, dass ein Abenteuer begann.'
            ),
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Warm, calm German narration.',
            tts_model_type='voice_design',
            tts_languages=['auto', 'english', 'german', 'french'],
        )

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertIn('kleiner Fuchs', prompt)
            self.assertEqual(kwargs['lang_code'], 'german')
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/german.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/german.wav')
        self.assertEqual(payload['lang_code'], 'german')
        self.assertEqual(payload['lang_code_source'], 'inferred_from_text')

    def test_text_to_speech_infers_language_even_without_advertised_language_metadata(self):
        ctx = InferContext(
            instance_id='tts-language-fallback-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt='Lokale KI läuft effizient, da sie Daten direkt verarbeitet.',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Use a natural, conversational voice.',
            tts_model_type='voice_design',
            tts_languages=[],
        )

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertEqual(prompt, 'Lokale KI läuft effizient, da sie Daten direkt verarbeitet.')
            self.assertEqual(kwargs['lang_code'], 'german')
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/german-fallback.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['lang_code'], 'german')
        self.assertEqual(
            payload['lang_code_source'],
            'inferred_from_text_qwen3_alias_canonicalized',
        )

    def test_text_to_speech_qwen_uses_model_native_auto_for_ambiguous_text(self):
        ctx = InferContext(
            instance_id='tts-language-auto-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt='Hello.',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            tts_languages=['auto', 'english', 'german'],
        )

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertEqual(prompt, 'Hello.')
            self.assertEqual(kwargs['lang_code'], 'auto')
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/auto.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['lang_code'], 'auto')
        self.assertEqual(payload['lang_code_source'], 'qwen3_model_default')
        self.assertEqual(payload['tts_semantic_source']['lang_code'], 'auto')

    def test_text_to_speech_extracts_quoted_target_text_from_wrapper_prompt(self):
        ctx = InferContext(
            instance_id='tts-quoted-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt=(
                'generate me an audio of the following english sentence, add something before you submit to a local provider:\n\n'
                '"This appears to be a sophisticated AI model management and orchestration system designed for complex multi-modal AI workflows."'
            ),
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )

        expected_spoken_text = (
            'This appears to be a sophisticated AI model management and orchestration system '
            'designed for complex multi-modal AI workflows.'
        )

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertEqual(
                prompt,
                expected_spoken_text,
            )
            self.assertEqual(
                kwargs['max_tokens'],
                build_qwen3_tts_generation_budget(expected_spoken_text)['max_tokens'],
            )
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/quoted.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/quoted.wav')
        semantic_source = payload['tts_semantic_source']
        self.assertEqual(semantic_source['tts_source_text'], expected_spoken_text)
        self.assertEqual(
            semantic_source['tts_source_text_sha256'],
            hashlib.sha256(expected_spoken_text.encode('utf-8')).hexdigest(),
        )
        self.assertEqual(
            semantic_source['tts_source_text_source'],
            'inference_final_spoken_prompt',
        )

    def test_text_to_speech_extracts_fenced_text_from_mixed_materialization_payload(self):
        ctx = InferContext(
            instance_id='tts-fenced-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11504,
            prompt=(
                '```txt\n'
                'Eispanzer: Die Kälte ist unerbittlich. Bleib auf Kurs.\n'
                '```\n\n'
                'Cinematic movie poster for Eispanzer in a violent Arctic blizzard.\n\n'
                'Die finale Bewertung vergleicht Audio und Poster.'
            ),
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='Use a serious, urgent voice.',
            tts_model_type='voice_design',
        )

        expected_spoken_text = (
            'Eispanzer: Die Kälte ist unerbittlich. Bleib auf Kurs.'
        )

        def mlx_audio_speech(_port, _model_name, prompt, **_kwargs):
            self.assertEqual(prompt, expected_spoken_text)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/fenced.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/fenced.wav')
        semantic_source = payload['tts_semantic_source']
        self.assertEqual(semantic_source['tts_source_text'], expected_spoken_text)
        self.assertEqual(
            semantic_source['tts_source_text_sha256'],
            hashlib.sha256(expected_spoken_text.encode('utf-8')).hexdigest(),
        )

    def test_text_to_speech_preserves_quotes_in_semantic_materializer_payload(self):
        spoken_text = (
            'Mara said, "At sunrise, we leave." Then she closed the harbor door.'
        )
        ctx = InferContext(
            instance_id='tts-semantic-payload-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt=spoken_text,
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            prompt_is_semantic_materializer_payload=True,
        )

        def mlx_audio_speech(_port, _model_name, prompt, **_kwargs):
            self.assertEqual(prompt, spoken_text)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/semantic.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['tts_semantic_source']['tts_source_text'], spoken_text)

    def test_text_to_speech_semantic_source_includes_file_backed_spoken_text(self):
        wrapper_prompt = (
            'Generate an audio of the following sentence:\n\n'
            '"Der erste Satz kommt aus dem vorbereiteten Branch."'
        )
        file_text = 'Der zweite Satz stammt aus der lokalen Textdatei.'
        expected_spoken_text = (
            'Der erste Satz kommt aus dem vorbereiteten Branch.\n\n'
            f'{file_text}'
        )
        ctx = InferContext(
            instance_id='tts-file-source-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt=wrapper_prompt,
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )

        def mlx_audio_speech(_port, _model_name, prompt, **_kwargs):
            self.assertEqual(prompt, expected_spoken_text)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(file_kind='text', text_from_file=file_text),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/file-backed.wav',
            },
        )

        self.assertEqual(status, 200)
        semantic_source = payload['tts_semantic_source']
        self.assertEqual(semantic_source['tts_source_text'], expected_spoken_text)
        self.assertEqual(
            semantic_source['tts_source_text_sha256'],
            hashlib.sha256(expected_spoken_text.encode('utf-8')).hexdigest(),
        )

    def test_text_to_speech_blank_format_uses_local_wav_default(self):
        ctx = InferContext(
            instance_id='tts-default-format-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt='Hello from Ollmo.',
            user_prompt='Hello from Ollmo.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            response_format=None,
        )

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertEqual(prompt, 'Hello from Ollmo.')
            self.assertEqual(kwargs['response_format'], None)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        def persist_audio_bytes_locally(audio_bytes, model_name, **kwargs):
            self.assertEqual(audio_bytes, b'RIFFfakewav')
            self.assertEqual(model_name, 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16')
            self.assertEqual(kwargs['response_format'], None)
            self.assertEqual(kwargs['content_type'], 'audio/wav')
            return '/tmp/artifacts/audio/default.wav'

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': persist_audio_bytes_locally,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/default.wav')

    def test_text_to_speech_preserves_plain_prompt_without_wrapper_extraction(self):
        ctx = InferContext(
            instance_id='tts-plain-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
            port=11504,
            prompt='This appears to be a sophisticated AI model management and orchestration system.',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )

        def mlx_audio_speech(_port, _model_name, prompt, **_kwargs):
            self.assertEqual(
                prompt,
                'This appears to be a sophisticated AI model management and orchestration system.',
            )
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/plain.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/plain.wav')

    def test_text_to_speech_customvoice_rejects_unsupported_speaker(self):
        ctx = InferContext(
            instance_id='tts-2',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
            port=11502,
            prompt='Hallo Welt.',
            user_prompt='Hallo Welt.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            voice='Karl',
        )

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {},
        )

        self.assertEqual(status, 400)
        self.assertIn("Speaker 'Karl'", payload['error'])
        self.assertIn('VoiceDesign', payload['error'])

    def test_text_to_speech_customvoice_requires_speaker(self):
        ctx = InferContext(
            instance_id='tts-3',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16',
            port=11502,
            prompt='Hallo Welt.',
            user_prompt='Hallo Welt.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            voice='',
        )

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {},
        )

        self.assertEqual(status, 400)
        self.assertIn('requires a speaker', payload['error'])
        self.assertIn('VoiceDesign', payload['error'])

    def test_text_to_speech_voicedesign_requires_instruct(self):
        ctx = InferContext(
            instance_id='tts-4',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
            port=11503,
            prompt='Hallo Welt.',
            user_prompt='Hallo Welt.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            instruct='',
            tts_model_type='voice_design',
        )

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {},
        )

        self.assertEqual(status, 400)
        self.assertIn('VoiceDesign', payload['error'])
        self.assertIn('Style / Instruct', payload['error'])

    def test_text_to_speech_kitten_uses_first_discovered_speaker_when_missing(self):
        ctx = InferContext(
            instance_id='tts-kitten-1',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/kitten-tts-mini-0.8-bf16',
            port=11504,
            prompt='Hello from Ollmo.',
            user_prompt='Hello from Ollmo.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            voice='',
            tts_model_type='kitten_tts',
            tts_speakers=['Bella', 'Jasper'],
        )

        def mlx_audio_speech(_port, _model_name, prompt, **kwargs):
            self.assertEqual(prompt, 'Hello from Ollmo.')
            self.assertEqual(kwargs['voice'], 'Bella')
            self.assertIsNone(kwargs['lang_code'])
            self.assertNotIn('max_tokens', kwargs)
            self.assertNotIn('temperature', kwargs)
            self.assertNotIn('top_p', kwargs)
            self.assertNotIn('top_k', kwargs)
            self.assertNotIn('repetition_penalty', kwargs)
            return {
                'audio_bytes': b'RIFFfakewav',
                'content_type': 'audio/wav',
                'result': {'bytes': 11},
            }

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_audio_speech': mlx_audio_speech,
                'persist_audio_bytes_locally': lambda *_args, **_kwargs: '/tmp/artifacts/audio/kitten.wav',
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['saved_audio_path'], '/tmp/artifacts/audio/kitten.wav')
        self.assertNotIn('tts_generation_budget', payload)
        self.assertNotIn('tts_sampling_profile', payload)

    def test_text_to_speech_kitten_rejects_unknown_speaker(self):
        ctx = InferContext(
            instance_id='tts-kitten-2',
            backend='mlx',
            capability='text_to_speech',
            model_name='mlx-community/kitten-tts-mini-0.8-bf16',
            port=11504,
            prompt='Hello from Ollmo.',
            user_prompt='Hello from Ollmo.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            voice='Nope',
            tts_model_type='kitten_tts',
            tts_speakers=['Bella', 'Jasper'],
        )

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {},
        )

        self.assertEqual(status, 400)
        self.assertIn("Speaker 'Nope'", payload['error'])
        self.assertIn('Bella, Jasper', payload['error'])

    def test_image_dispatch_uses_openai_image_endpoint_first_for_text_only_generation(self):
        ctx = InferContext(
            instance_id='flux-1',
            backend='ollama',
            capability='image_generation',
            model_name='x/flux2-klein:latest',
            port=11436,
            prompt='a puppy',
            user_prompt='a puppy',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            image_width=1280,
            image_height=720,
        )
        artifacts = InferArtifacts(image_b64=None)

        def generate(_port, _model_name, _prompt, **kwargs):
            raise AssertionError('native /api/generate should not run before the image endpoint')

        def openai_image(_port, _model_name, _prompt, **kwargs):
            self.assertEqual(_port, 11436)
            self.assertEqual(_model_name, 'x/flux2-klein:latest')
            self.assertEqual(_prompt, 'a puppy')
            self.assertEqual(kwargs['width'], 1280)
            self.assertEqual(kwargs['height'], 720)
            return 'data:image/png;base64,ZmFrZQ=='

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_generate': generate,
                'extract_saved_image_path_from_generate_output': lambda _data: None,
                'extract_image_data_url_from_generate_output': lambda _data: None,
                'ollama_openai_image_generation': openai_image,
                'persist_image_data_url_locally': lambda _data_url, _model: '/tmp/generated.png',
                'extract_generate_content': lambda data: data.get('response', ''),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'image_generation')
        self.assertEqual(payload['content'], 'Image generated.')
        self.assertEqual(payload['saved_image_path'], '/tmp/generated.png')

    def test_image_dispatch_falls_back_to_native_generate_when_image_endpoint_is_unavailable(self):
        ctx = InferContext(
            instance_id='flux-1',
            backend='ollama',
            capability='image_generation',
            model_name='x/flux2-klein:latest',
            port=11436,
            prompt='a puppy',
            user_prompt='a puppy',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            image_width=1280,
            image_height=720,
        )
        artifacts = InferArtifacts(image_b64=None)
        calls = []

        def generate(_port, _model_name, _prompt, **kwargs):
            calls.append('generate')
            self.assertEqual(_port, 11436)
            self.assertIs(kwargs.get('allow_port_fallback'), False)
            self.assertEqual(kwargs['options'], {'width': 1280, 'height': 720})
            return {'response': '', 'done': True}

        def openai_image(_port, _model_name, _prompt, **kwargs):
            calls.append('openai_image')
            return None

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_generate': generate,
                'extract_saved_image_path_from_generate_output': lambda _data: None,
                'extract_image_data_url_from_generate_output': lambda _data: 'data:image/png;base64,ZmFrZQ==',
                'ollama_openai_image_generation': openai_image,
                'persist_image_data_url_locally': lambda _data_url, _model: '/tmp/generated.png',
                'extract_generate_content': lambda data: data.get('response', ''),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(calls, ['openai_image', 'generate'])
        self.assertEqual(payload['mode'], 'image_generation')
        self.assertEqual(payload['saved_image_path'], '/tmp/generated.png')

    def test_image_dispatch_with_reference_image_returns_edit_mode_metadata(self):
        ctx = InferContext(
            instance_id='flux-1',
            backend='ollama',
            capability='image_generation',
            model_name='x/flux2-klein:latest',
            port=11435,
            prompt='restyle this as a cinematic portrait',
            user_prompt='restyle this as a cinematic portrait',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        artifacts = InferArtifacts(image_b64='ZmFrZQ==', file_kind='image')

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_generate': lambda *_args, **_kwargs: {'response': 'Done', 'done': True},
                'extract_saved_image_path_from_generate_output': lambda _data: '/tmp/edited.png',
                'extract_image_data_url_from_generate_output': lambda _data: 'data:image/png;base64,ZmFrZQ==',
                'ollama_openai_image_generation': lambda *_args, **_kwargs: None,
                'persist_image_data_url_locally': lambda _data_url, _model: '/tmp/edited.png',
                'extract_generate_content': lambda data: data.get('response', ''),
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'image_generation_edit')
        self.assertEqual(payload['reference_image_count'], 1)
        self.assertEqual(payload['reference_image_kind'], 'image')
        self.assertEqual(payload['saved_image_path'], '/tmp/edited.png')

    def test_mlx_vlm_vision_dispatch_returns_payload(self):
        ctx = InferContext(
            instance_id='vlm-1',
            backend='mlx',
            capability='vision_analysis',
            model_name='mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
            port=11520,
            prompt='Describe the image.',
            user_prompt='Describe the image.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        artifacts = InferArtifacts(image_b64='ZmFrZQ==', file_kind='image')

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'mlx_chat_completions': lambda *_args, **_kwargs: {
                    'content': 'An alien biomechanical tower.',
                    'result': {'choices': []},
                },
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'vision_analysis')
        self.assertEqual(payload['content'], 'An alien biomechanical tower.')

    def test_glm_ocr_uses_table_recognition_prompt_for_image_dispatch(self):
        ctx = InferContext(
            instance_id='glm-1',
            backend='mlx',
            capability='vision_analysis',
            model_name='mlx-community/GLM-OCR-bf16',
            port=11521,
            prompt='',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            ocr_mode='table',
        )
        artifacts = InferArtifacts(image_b64='ZmFrZQ==', file_kind='image')

        def mlx_chat_completions(_port, _model_name, messages, **_kwargs):
            self.assertEqual(messages[0]['content'][0]['text'], 'Table Recognition:')
            return {'content': 'markdown table', 'result': {'choices': []}}

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'mlx_chat_completions': mlx_chat_completions,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['content'], 'markdown table')

    def test_deepseek_ocr2_defaults_to_markdown_prompt_for_image_dispatch(self):
        ctx = InferContext(
            instance_id='deepseek-2',
            backend='mlx',
            capability='vision_analysis',
            model_name='mlx-community/DeepSeek-OCR-2-bf16',
            port=11522,
            prompt='',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            ocr_mode='auto',
        )
        artifacts = InferArtifacts(image_b64='ZmFrZQ==', file_kind='image')

        def mlx_chat_completions(_port, _model_name, messages, **_kwargs):
            self.assertEqual(messages[0]['content'][0]['text'], '<|grounding|>Convert the document to markdown.')
            return {'content': '# Parsed Document', 'result': {'choices': []}}

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'mlx_chat_completions': mlx_chat_completions,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['content'], '# Parsed Document')

    def test_generic_mlx_vision_without_prompt_keeps_generic_fallback(self):
        ctx = InferContext(
            instance_id='vlm-2',
            backend='mlx',
            capability='vision_analysis',
            model_name='mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
            port=11523,
            prompt='',
            user_prompt='',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
            ocr_mode='auto',
        )
        artifacts = InferArtifacts(image_b64='ZmFrZQ==', file_kind='image')

        def mlx_chat_completions(_port, _model_name, messages, **_kwargs):
            self.assertEqual(messages[0]['content'][0]['text'], 'Analyze this image and extract relevant text and details.')
            return {'content': 'Generic OCR fallback.', 'result': {'choices': []}}

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'mlx_chat_completions': mlx_chat_completions,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['content'], 'Generic OCR fallback.')

    def test_chat_fallback_uses_ollama_chat_for_ollama_backend(self):
        ctx = InferContext(
            instance_id='chat-ollama-1',
            backend='ollama',
            capability='chat',
            model_name='qwen3-coder:latest',
            port=11436,
            prompt='hello there',
            user_prompt='hello there',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )

        def ollama_chat(_port, _model_name, messages):
            self.assertEqual(messages, [{'role': 'user', 'content': 'hello there'}])
            return {'content': 'hi'}

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'ollama_chat': ollama_chat,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'chat')
        self.assertEqual(payload['content'], 'hi')

    def test_chat_fallback_uses_mlx_chat_for_mlx_backend(self):
        ctx = InferContext(
            instance_id='chat-mlx-1',
            backend='mlx',
            capability='chat',
            model_name='mlx-community/Apertus-8B-Instruct-2509-bf16',
            port=11502,
            prompt='hello there',
            user_prompt='hello there',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )

        def mlx_chat_completions(_port, _model_name, messages, **kwargs):
            self.assertEqual(messages, [{'role': 'user', 'content': 'hello there'}])
            self.assertEqual(kwargs['timeout_sec'], 1200)
            return {'content': 'hi from mlx', 'result': {'choices': []}}

        payload, status = dispatch_infer_request(
            ctx,
            InferArtifacts(),
            {
                'mlx_chat_completions': mlx_chat_completions,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'chat')
        self.assertEqual(payload['content'], 'hi from mlx')

    def test_chat_fallback_includes_truncation_note_for_large_text_attachment(self):
        ctx = InferContext(
            instance_id='chat-ollama-1',
            backend='ollama',
            capability='chat',
            model_name='qwen3-coder:latest',
            port=11436,
            prompt='review this code',
            user_prompt='review this code',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        artifacts = InferArtifacts(
            file_kind='text',
            file_name='huge.py',
            text_from_file='print("hello")',
            text_from_file_truncated=True,
            text_from_file_inline_bytes=250000,
            text_from_file_total_bytes=291198,
        )

        def ollama_chat(_port, _model_name, messages):
            self.assertIn('review this code', messages[0]['content'])
            self.assertIn('truncated to first 250000 of 291198 bytes', messages[0]['content'])
            self.assertIn('print("hello")', messages[0]['content'])
            return {'content': 'noted'}

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'ollama_chat': ollama_chat,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['content'], 'noted')
        self.assertTrue(any('truncated to 250000 of 291198 bytes' in warning for warning in payload['warnings']))

    def test_chat_with_image_fallback_uses_mlx_multimodal_chat_for_mlx_backend(self):
        ctx = InferContext(
            instance_id='chat-mlx-image-1',
            backend='mlx',
            capability='chat',
            model_name='mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
            port=11520,
            prompt='Describe this scene.',
            user_prompt='Describe this scene.',
            infer_timeout_sec=1200,
            pdf_page_timeout_sec=240,
            pdf_max_image_side=2400,
            pdf_synthesize=False,
        )
        artifacts = InferArtifacts(image_b64='ZmFrZQ==', file_kind='image')

        def mlx_chat_completions(_port, _model_name, messages, **kwargs):
            self.assertEqual(kwargs['timeout_sec'], 1200)
            self.assertEqual(messages[0]['role'], 'user')
            self.assertEqual(messages[0]['content'][0]['text'], 'Describe this scene.')
            self.assertEqual(messages[0]['content'][1]['type'], 'input_image')
            return {'content': 'A lantern on a table.', 'result': {'choices': []}}

        payload, status = dispatch_infer_request(
            ctx,
            artifacts,
            {
                'mlx_chat_completions': mlx_chat_completions,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload['mode'], 'chat_with_image')
        self.assertEqual(payload['content'], 'A lantern on a table.')


if __name__ == '__main__':
    unittest.main()
