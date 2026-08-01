import tempfile
import unittest
from pathlib import Path

from ollmo_core.inference import (
    InferArtifacts,
    InferContext,
    detect_text_artifact_request,
    detect_text_artifact_requests,
    dispatch_infer_request,
    extract_text_artifact_payload,
    extract_text_artifact_payloads,
    generated_text_is_artifact_self_claim,
    text_artifact_request_is_ungrounded_reference,
)
import ollmo_core.transports as transports
from ollmo_core.transports import persist_text_artifact_locally


class InferenceServiceTests(unittest.TestCase):
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

    def test_detect_text_artifact_request_ignores_plain_chat(self):
        self.assertIsNone(detect_text_artifact_request('Tell me what HTML means in one sentence.'))
        self.assertIsNone(detect_text_artifact_request('What does index.html mean?'))

    def test_detect_text_artifact_request_blocks_ungrounded_this_without_source(self):
        prompt = 'Generate me this html file as artifact'

        self.assertTrue(text_artifact_request_is_ungrounded_reference(prompt))
        self.assertIsNone(detect_text_artifact_request(prompt))
        self.assertIsNotNone(detect_text_artifact_request(prompt, source_available=True))

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
            self.assertEqual(kwargs['lang_code'], 'de')
            self.assertNotIn('max_tokens', kwargs)
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
        self.assertEqual(payload['lang_code'], 'de')

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
            self.assertEqual(kwargs['lang_code'], 'de')
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
        self.assertEqual(payload['lang_code'], 'de')
        self.assertEqual(payload['lang_code_source'], 'inferred_from_text')

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

        def mlx_audio_speech(_port, _model_name, prompt, **_kwargs):
            self.assertEqual(
                prompt,
                'This appears to be a sophisticated AI model management and orchestration system designed for complex multi-modal AI workflows.',
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

        def mlx_audio_speech(_port, _model_name, prompt, **_kwargs):
            self.assertEqual(prompt, 'Eispanzer: Die Kälte ist unerbittlich. Bleib auf Kurs.')
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
