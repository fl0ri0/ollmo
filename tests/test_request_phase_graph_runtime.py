import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ollmo_g.request_phase_graph import (
    _structured_final_join_selected_phase_ids,
    build_request_phase_graph,
)
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner


class RequestPhaseGraphRuntimeTests(unittest.TestCase):
    def _executable_image_branches(self, graph):
        return [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('capability') == 'image_generation'
            and branch.get('contract_state') != 'reserved'
        ]

    def _text_artifact_branches(self, graph):
        return [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('role') == 'text_artifact_output'
        ]

    def _image_obligations(self, graph):
        return [
            obligation for obligation in (graph.get('output_obligations') or [])
            if obligation.get('capability') == 'image_generation'
            and obligation.get('output_type') == 'image'
        ]

    def _text_artifact_obligations(self, graph):
        return [
            obligation for obligation in (graph.get('output_obligations') or [])
            if obligation.get('role') == 'text_artifact_output'
        ]

    def _intent_obligations(self, graph, kind=None):
        obligations = [
            item for item in (graph.get('intent_obligations') or [])
            if isinstance(item, dict)
        ]
        if kind is None:
            return obligations
        return [item for item in obligations if item.get('kind') == kind]

    def test_discussion_and_negated_artifacts_do_not_promote_execution(self):
        discussion_prompts = (
            'Explain image generation and HTML artifacts.',
            'Explain how to create an image.',
            'Why would someone create an image?',
            'Do not create index.html; just explain the structure.',
        )

        for prompt in discussion_prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'prompt': prompt},
                    route_payload={'capability': 'chat'},
                )
                self.assertEqual(graph['mode'], 'single_phase')
                self.assertEqual(graph.get('downstream_branches'), [])
                self.assertFalse(graph['prompt_intent']['requests_visual_output'])
                self.assertFalse(
                    graph['prompt_intent']['requests_text_artifact_output']
                )

        executable_prompt = (
            'Explain image generation, then create one image of a lighthouse.'
        )
        executable_graph = build_request_phase_graph(
            executable_prompt,
            request_payload={'prompt': executable_prompt},
            route_payload={'capability': 'chat'},
        )
        self.assertEqual(
            [
                branch.get('capability')
                for branch in executable_graph.get('downstream_branches') or []
            ],
            ['image_generation'],
        )

    def test_generated_image_analysis_followup_builds_evidence_and_final_text(self):
        prompt = (
            'Erstelle ein realistisches Bild eines gelben Notizbuchs mit der Aufschrift "Plan A". '
            'Analysiere danach nur sichtbare Details des erzeugten Bildes auf Deutsch.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            ['image_generation', 'vision_analysis', 'chat'],
        )
        self.assertEqual(branches[1].get('depends_on'), [branches[0].get('phase_id')])
        self.assertEqual(branches[2].get('depends_on'), [branches[1].get('phase_id')])

    def test_legacy_zuerst_image_analysis_followup_still_builds_evidence_chain(self):
        prompt = (
            'Erstelle zuerst ein Bild von einem gelben Notizbuch mit der Aufschrift "Plan A". '
            'Analysiere danach das erzeugte Bild und nenne nur sichtbare Details.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            ['image_generation', 'vision_analysis', 'chat'],
        )

    def test_audio_and_image_materialization_are_sibling_branches_from_text(self):
        prompt = (
            'Write a short mystical story in 2 short paragraphs, then read it aloud '
            'and show it to me as an image.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            ['text_to_speech', 'image_generation'],
        )
        self.assertEqual([branch.get('depends_on') for branch in branches], [['phase-1'], ['phase-1']])

    def test_spoken_version_is_audio_and_does_not_create_image_work(self):
        prompt = (
            'Write a short original poem in English inspired by Ollmo – by open possibilities, '
            'intentions taking form, unfinished work remaining visible, and truth resting in what '
            'was actually made. Let it feel reflective and lyrical rather than technical.\n\n'
            'Save the poem as a Markdown file and generate a spoken version using local '
            'text-to-speech with a calm, confident, deep voice.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertEqual(self._executable_image_branches(graph), [])
        self.assertEqual(self._image_obligations(graph), [])
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech', 'chat'])
        self.assertEqual(len(self._text_artifact_branches(graph)), 1)
        self.assertEqual(
            len([
                branch for branch in (graph.get('downstream_branches') or [])
                if branch.get('capability') == 'text_to_speech'
            ]),
            1,
        )

    def test_explicit_no_image_keeps_poem_markdown_and_audio_without_image_promotion(self):
        prompt = (
            'Write a short original poem in English inspired by Ollmo – by open possibilities, '
            'intentions taking form, unfinished work remaining visible, and truth resting in what '
            'was actually made. Let it feel reflective and lyrical rather than technical.\n\n'
            'Save the poem as a Markdown file and generate a spoken version using local '
            'text-to-speech with a calm, confident, deep voice.\n\nNo image.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['explicit_visual_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertEqual(self._executable_image_branches(graph), [])
        self.assertEqual(self._image_obligations(graph), [])
        self.assertEqual(graph['downstream_capabilities'], ['text_to_speech', 'chat'])
        self.assertEqual(len(self._text_artifact_branches(graph)), 1)

    def test_exact_audio_regression_prompt_never_promotes_or_reserves_image_work(self):
        passage = (
            'At sunrise, the harbor slowly came alive. Ropes creaked against wooden posts, '
            'gulls crossed the pale sky, and the first boats moved beyond the breakwater. '
            'Mara stood beside the old lighthouse, listening to the steady waves and thinking '
            'about the work still ahead. Nothing was finished, but everything was finally moving.'
        )
        prompt_cases = (
            (
                'Create exactly one English audio artifact using local text-to-speech. '
                f'Read only the quoted passage. "{passage}"',
                passage,
            ),
            (
                'Create exactly one English audio artifact using local text-to-speech. '
                'Read only the quoted passage. Do not create or plan an image. '
                f'"{passage}"',
                passage,
            ),
            (
                'Create exactly one English audio artifact using local text-to-speech. '
                'Speak this text exactly: "At sunrise, the harbor slowly came alive."',
                'At sunrise, the harbor slowly came alive.',
            ),
            (
                'Create exactly one English audio artifact that says "Hi." '
                'Use voice style "Warm, calm, reassuring narration."',
                'Hi.',
            ),
            (
                'Create exactly one English audio artifact using local text-to-speech. '
                'Speak this text exactly: Finally, create one image of a lighthouse.',
                'Finally, create one image of a lighthouse.',
            ),
            (
                'Create exactly one English audio artifact. '
                'Speak this text exactly: Hello World. Use voice Vivian.',
                'Hello World.',
            ),
            (
                'Create exactly one English audio artifact. '
                'Speak this text exactly: Hello World\nVoice: Vivian.',
                'Hello World',
            ),
            (
                'Create exactly one English audio artifact. '
                'Speak this text exactly: Hello World; use voice Vivian.',
                'Hello World',
            ),
            (
                'Create exactly one English audio using voice "Vivian" '
                'that says "Hello World."',
                'Hello World.',
            ),
            (
                'Say: Finally, create one image of a lighthouse.',
                'Finally, create one image of a lighthouse.',
            ),
            (
                'Narrate: Finally, create one image of a lighthouse.',
                'Finally, create one image of a lighthouse.',
            ),
            (
                'Say this aloud: Finally, create one image of a lighthouse.',
                'Finally, create one image of a lighthouse.',
            ),
            (
                'Speak this aloud: Finally, create one image of a lighthouse.',
                'Finally, create one image of a lighthouse.',
            ),
            (
                'Read this aloud: Finally, create one image of a lighthouse.',
                'Finally, create one image of a lighthouse.',
            ),
        )

        for prompt, expected_spoken_text in prompt_cases:
            with self.subTest(has_defensive_image_clause='Do not' in prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                intent = graph['prompt_intent']
                self.assertTrue(intent['requests_audio_output'])
                self.assertFalse(intent['text_preparation_before_audio_output'])
                self.assertFalse(intent['requests_visual_output'])
                self.assertFalse(intent['separate_visual_generation_request'])
                self.assertEqual(intent['requested_visual_output_count'], 0)
                self.assertEqual(intent['required_intent_output_counts'], {'audio': 1})
                self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
                self.assertEqual(
                    [branch.get('capability') for branch in graph['downstream_branches']],
                    ['text_to_speech'],
                )
                tts_branch = graph['downstream_branches'][0]
                self.assertEqual(tts_branch.get('content_payload'), expected_spoken_text)
                self.assertEqual(
                    tts_branch.get('content_payload_source'),
                    'current_turn_direct_spoken_clause',
                )
                self.assertEqual(self._image_obligations(graph), [])
                self.assertFalse(
                    any(
                        item.get('capability') == 'image_generation'
                        for item in self._intent_obligations(graph)
                    )
                )
                self.assertFalse(
                    any(
                        candidate.get('capability') == 'image_generation'
                        for candidate in (graph.get('output_candidates') or [])
                        if isinstance(candidate, dict)
                    )
                )
                self.assertFalse(
                    any(
                        candidate.get('capability') == 'image_generation'
                        for candidate in (
                            (graph.get('candidate_graph') or {}).get('candidates') or []
                        )
                        if isinstance(candidate, dict)
                    )
                )

    def test_quoted_spoken_visual_language_is_literal_not_executable(self):
        for marker in ('Finally', 'Then', 'Afterwards', 'Lastly'):
            prompt = (
                'Create exactly one English audio artifact using local text-to-speech. '
                f'Read only the quoted sentence. "{marker}, create one image of a lighthouse."'
            )
            with self.subTest(marker=marker):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                self.assertFalse(graph['prompt_intent']['requests_visual_output'])
                self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
                self.assertEqual(graph['downstream_capabilities'], ['text_to_speech'])
                self.assertEqual(self._executable_image_branches(graph), [])
                self.assertEqual(self._image_obligations(graph), [])

    def test_outer_visual_command_remains_executable_with_quoted_tts_literal(self):
        prompt = (
            'Create one image of a lighthouse titled "Finally home", '
            'then read the title aloud.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertEqual(
            [branch.get('capability') for branch in graph['downstream_branches']],
            ['image_generation', 'text_to_speech'],
        )
        self.assertEqual(len(self._image_obligations(graph)), 1)

    def test_direct_mixed_media_clauses_bind_distinct_branch_local_payloads(self):
        prompts = (
            (
                'Create one image of a lighthouse at sunrise. Then create one English audio '
                'artifact that says, "The lighthouse welcomes the morning."'
            ),
            (
                'Create one image of a lighthouse at sunrise. Then create one English audio '
                'artifact that says "The lighthouse welcomes the morning."'
            ),
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                branches = graph.get('downstream_branches') or []
                self.assertEqual(
                    [branch.get('capability') for branch in branches],
                    ['image_generation', 'text_to_speech'],
                )
                image_branch, tts_branch = branches
                self.assertEqual(image_branch.get('depends_on'), ['phase-1'])
                self.assertEqual(tts_branch.get('depends_on'), ['phase-1'])
                self.assertEqual(
                    image_branch.get('artifact_prompt'),
                    'a lighthouse at sunrise',
                )
                self.assertEqual(
                    image_branch.get('artifact_prompt_source'),
                    'current_turn_direct_image_clause',
                )
                self.assertEqual(
                    tts_branch.get('content_payload'),
                    'The lighthouse welcomes the morning.',
                )
                self.assertEqual(
                    tts_branch.get('content_payload_source'),
                    'current_turn_direct_spoken_clause',
                )
                self.assertEqual(len(self._image_obligations(graph)), 1)
                self.assertEqual(self._text_artifact_obligations(graph), [])

    def test_shared_generated_media_flow_does_not_receive_direct_payload_binding(self):
        prompts = (
            'Write a short poem about the harbor, then read it aloud.',
            'Write a poem titled "At Sunrise", then speak it aloud.',
            'Write a poem, then speak it aloud. Style: concise',
            'Summarize the source, then read the summary aloud: '
            '```text\nSource material only.\n```',
            'Read this passage "The harbor was quiet.", summarize it in one sentence, '
            'then speak the summary aloud.',
            'Read this passage "The harbor was quiet.", condense it to one sentence, '
            'then speak the result aloud.',
            'Read this passage "The harbor was quiet.", create a one-sentence summary, '
            'then speak it aloud.',
            'Read this passage "The harbor was quiet.", shorten it, then speak it aloud.',
            'Read this passage "The harbor was quiet.", paraphrase it, then speak it aloud.',
            'Read this passage "The harbor was quiet.", convert it into a summary, '
            'then speak it aloud.',
            'Create exactly one audio that says "Hello" and then says "Goodbye".',
            'Write a short lighthouse story, then create one image of it and read it aloud.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                self.assertTrue(
                    graph['prompt_intent']['text_preparation_before_audio_output']
                )
                for branch in graph.get('downstream_branches') or []:
                    if branch.get('capability') == 'image_generation':
                        self.assertNotEqual(
                            branch.get('artifact_prompt_source'),
                            'current_turn_direct_image_clause',
                        )
                    if branch.get('capability') == 'text_to_speech':
                        self.assertNotEqual(
                            branch.get('content_payload_source'),
                            'current_turn_direct_spoken_clause',
                        )

    def test_negated_old_image_preserves_affirmative_new_image_and_tts(self):
        prompt = (
            'Do not create the old image again, but create one new image of a lighthouse, '
            'then create one English audio artifact that says "Welcome home."'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['separate_visual_generation_request'])
        self.assertTrue(intent['requests_visual_output'])
        self.assertTrue(intent['requests_audio_output'])
        self.assertFalse(intent['explicit_visual_defer_materialization'])
        self.assertFalse(intent['explicit_audio_defer_materialization'])
        self.assertEqual(intent['required_intent_output_counts'], {'image': 1, 'audio': 1})
        self.assertEqual(graph['downstream_capabilities'], ['image_generation', 'text_to_speech'])
        self.assertEqual(
            [branch.get('capability') for branch in graph['downstream_branches']],
            ['image_generation', 'text_to_speech'],
        )
        self.assertEqual(len(self._image_obligations(graph)), 1)

    def test_dont_just_image_with_audio_is_additive_not_deferred(self):
        prompt = (
            "Don't just create one image of a lighthouse; also create one English audio "
            'artifact that says "Welcome home."'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['requests_visual_output'])
        self.assertTrue(intent['requests_audio_output'])
        self.assertFalse(intent['explicit_visual_defer_materialization'])
        self.assertFalse(intent['explicit_audio_defer_materialization'])
        self.assertEqual(intent['required_intent_output_counts'], {'image': 1, 'audio': 1})
        branch_capabilities = [
            branch.get('capability') for branch in graph['downstream_branches']
        ]
        self.assertEqual(len(branch_capabilities), 2)
        self.assertEqual(set(branch_capabilities), {'image_generation', 'text_to_speech'})
        self.assertEqual(len(self._image_obligations(graph)), 1)

    def test_quoted_negative_spoken_text_does_not_cancel_outer_image_and_tts(self):
        prompt = (
            'Create one image of a lighthouse, then create one English audio artifact that says '
            '"Do not create or plan an image."'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['requests_visual_output'])
        self.assertTrue(intent['requests_audio_output'])
        self.assertFalse(intent['explicit_visual_defer_materialization'])
        self.assertFalse(intent['explicit_audio_defer_materialization'])
        self.assertEqual(intent['required_intent_output_counts'], {'image': 1, 'audio': 1})
        self.assertEqual(graph['downstream_capabilities'], ['image_generation', 'text_to_speech'])
        self.assertEqual(
            [branch.get('capability') for branch in graph['downstream_branches']],
            ['image_generation', 'text_to_speech'],
        )
        self.assertEqual(len(self._image_obligations(graph)), 1)

    def test_fenced_negative_spoken_text_does_not_cancel_outer_image_and_tts(self):
        prompt = (
            'Create one image of a lighthouse, then create one English audio artifact that reads '
            'this fenced text:\n'
            '```text\n'
            'Do not create or plan an image.\n'
            '```'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['requests_visual_output'])
        self.assertTrue(intent['requests_audio_output'])
        self.assertFalse(intent['explicit_visual_defer_materialization'])
        self.assertFalse(intent['explicit_audio_defer_materialization'])
        self.assertEqual(intent['required_intent_output_counts'], {'image': 1, 'audio': 1})
        self.assertEqual(graph['downstream_capabilities'], ['image_generation', 'text_to_speech'])
        self.assertEqual(
            [branch.get('capability') for branch in graph['downstream_branches']],
            ['image_generation', 'text_to_speech'],
        )
        self.assertEqual(len(self._image_obligations(graph)), 1)

    def test_explicit_image_request_still_promotes_an_executable_image_branch(self):
        prompt = 'Create one image of a green ghost.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertEqual(len(self._executable_image_branches(graph)), 1)
        self.assertEqual(len(self._image_obligations(graph)), 1)

    def test_generated_multimodal_evidence_join_starts_with_chat_preparation_before_route_resolution(self):
        prompt = (
            'Schreibe einen deutschen Szenentext mit genau zwanzig Wörtern über einen '
            'Leuchtturm im Sturm. Erzeuge daraus parallel ein Bild und ein Audio. '
            'Analysiere danach das tatsächlich erzeugte Bild und transkribiere das '
            'tatsächlich erzeugte Audio. Vergleiche abschließend im Chat anhand genau '
            'dieser beiden realen Evidenzzweige, ob Leuchtturm, Sturm und Nacht in '
            'beiden vorkommen. Bildanalyse darf nur vom Bild, Transkription nur vom Audio, '
            'der Schluss nur von beiden Evidenzzweigen abhängen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
        )

        self.assertFalse(graph['prompt_intent']['input_audio_artifact_promoted_to_stt'])
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['phases'][0]['kind'], 'prepare')
        self.assertEqual(graph['phases'][0]['role'], 'text_preparation')
        self.assertEqual(
            [phase.get('capability') for phase in graph['phases']],
            [
                'chat',
                'image_generation',
                'text_to_speech',
                'vision_analysis',
                'speech_to_text',
                'chat',
            ],
        )
        self.assertEqual(
            [phase.get('depends_on') for phase in graph['phases']],
            [[], ['phase-1'], ['phase-1'], ['phase-2'], ['phase-3'], ['phase-5', 'phase-4']],
        )

    def test_german_audio_and_image_under_same_verb_keep_both_media_branches(self):
        prompt = (
            'Schreibe einen kurzen deutschen Produkttext. Erzeuge daraus ein Audio und ein Bild. '
            'Danach nenne kurz, welche Artefakte erzeugt wurden.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            ['text_to_speech', 'image_generation', 'chat'],
        )
        audio_branch = branches[0]
        image_branch = branches[1]
        final_branch = branches[2]
        self.assertEqual(audio_branch.get('depends_on'), ['phase-1'])
        self.assertEqual(image_branch.get('depends_on'), ['phase-1'])
        self.assertEqual(final_branch.get('depends_on'), [audio_branch.get('phase_id'), image_branch.get('phase_id')])

    def test_german_image_and_audio_under_same_verb_keep_both_media_branches(self):
        prompt = (
            'Schreibe einen kurzen deutschen Produkttext. Erzeuge daraus ein Bild und ein Audio. '
            'Danach nenne kurz, welche Artefakte erzeugt wurden.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            ['image_generation', 'text_to_speech', 'chat'],
        )
        image_branch = branches[0]
        audio_branch = branches[1]
        final_branch = branches[2]
        self.assertEqual(image_branch.get('depends_on'), ['phase-1'])
        self.assertEqual(audio_branch.get('depends_on'), ['phase-1'])
        self.assertEqual(final_branch.get('depends_on'), [audio_branch.get('phase_id'), image_branch.get('phase_id')])

    def test_german_quality_negation_does_not_suppress_requested_landing_page_images(self):
        prompt = (
            'Erstelle eine starke, einseitige, premium Landing Page für:\n\n'
            '[Nocturne Sanctum – ein exklusives, hochmodernes Luxus-Refugium in einer abgelegenen Bergkette '
            'der Schweizer Alpen]\n\n'
            'Mach es cinematic, dunkel, edel und immersiv. Starke einheitliche Luxus-Welt mit konsistenter '
            'Farbgebung (tiefes Schwarz, warmer Goldton, kühles Anthrazit und dezentes Bernstein).\n\n'
            'Erstelle genau drei Bilder und zwei Dateien:\n'
            '- 3 Bilder: ein starkes Hero-Bild (wide cinematic exterior bei Dämmerung) + zwei passende '
            'Innenansichten, die zusammen eine kohärente Welt ergeben\n'
            '- index.html\n'
            '- styles.css\n\n'
            'Struktur:\n'
            '- Fixed Navigation oben\n'
            '- Hero Section (großes Bild + Overlay-Text + CTA)\n'
            '- Technical Capabilities (4 Feature Cards)\n'
            '- Zwei Content-Sections mit je einem Bild (abwechselnd links/rechts auf Desktop)\n'
            '- Starker finaler CTA\n'
            '- Footer\n\n'
            'Regeln:\n'
            '- Moderne, saubere CSS mit CSS-Variablen für Farben und Glows\n'
            '- Gute Typografie, viel Luft, subtile Hover-Effekte\n'
            '- Fixed Header + smooth scroll\n'
            '- Responsiv (gut auf Mobile)\n'
            '- Text selbstbewusst, atmosphärisch und immersiv – kein billiger Marketing-Sprech\n'
            '- Alle Bildpfade müssen am Ende korrekt auf die lokal gespeicherten Bilder verweisen\n'
            '- Keine unnötigen Effekte, nur das was wirklich zur Stimmung passt\n\n'
            'Arbeite präzise, still und professionell. Korrigiere Fehler direkt in den Dateien. '
            'Am Ende nur die fertigen Artefakte liefern, keine Erklärungen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        image_branches = [
            branch for branch in branches
            if branch.get('capability') == 'image_generation'
            and branch.get('contract_state') != 'reserved'
        ]
        text_artifact_branches = [
            branch for branch in branches
            if branch.get('role') == 'text_artifact_output'
        ]

        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(
            [branch.get('branch_id') for branch in image_branches],
            ['branch-image_generation-1', 'branch-image_generation-2', 'branch-image_generation-3'],
        )
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertFalse(
            any(
                branch.get('capability') == 'image_generation'
                and branch.get('contract_state') == 'reserved'
                for branch in branches
            )
        )
        self.assertEqual(
            [(branch.get('text_artifact_source_name'), branch.get('text_artifact_extension')) for branch in text_artifact_branches],
            [('index', 'html'), ('styles', 'css')],
        )

    def test_german_external_image_and_file_path_constraints_keep_exact_image_count(self):
        prompt = (
            'Erstelle genau drei Bilder und zwei Dateien: index.html und styles.css. '
            'Keine Platzhalter, erfundenen Dateipfade oder externen Bilder. '
            'Alle Bildpfade müssen am Ende korrekt auf die lokal gespeicherten Bilder verweisen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        reserved_image_branches = [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('capability') == 'image_generation'
            and branch.get('contract_state') == 'reserved'
        ]

        self.assertTrue(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(
            [branch.get('branch_id') for branch in image_branches],
            ['branch-image_generation-1', 'branch-image_generation-2', 'branch-image_generation-3'],
        )
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertEqual(reserved_image_branches, [])
        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        text_artifact_branches = [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('role') == 'text_artifact_output'
        ]
        self.assertEqual(len(text_artifact_branches), 2)
        self.assertTrue(
            all(branch.get('depends_on') == image_phase_ids for branch in text_artifact_branches)
        )
        self.assertTrue(
            all(
                branch.get('dependency_contract') == 'local_visual_asset_binding'
                for branch in text_artifact_branches
            )
        )

    def test_german_output_format_negations_keep_text_artifact_intent_obligations(self):
        prompts = (
            (
                'Erstelle genau drei Bilder und zwei Dateien: index.html und styles.css. '
                'Alle Bildpfade müssen auf tatsächlich gespeicherte lokale Bilder zeigen. '
                'Chat-Codeblöcke zählen nicht als Dateien.'
            ),
            (
                'Erstelle eine einseitige Landing Page mit genau vier Bildern und zwei Dateien: '
                'index.html und styles.css. '
                'Alle Medien- und Stylesheet-Links müssen korrekt relativ gebunden sein. '
                'Am Ende keine Erklärung, nur die fertigen Artefakte.'
            ),
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'input': [
                            {
                                'role': 'user',
                                'content': [{'type': 'input_text', 'text': prompt}],
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                text_obligations = self._intent_obligations(graph, 'text_artifact')
                image_obligations = self._intent_obligations(graph, 'media_artifact')
                dependency_obligations = self._intent_obligations(graph, 'dependency')

                self.assertEqual(
                    [
                        (item.get('target_name'), item.get('target_extension'))
                        for item in text_obligations
                    ],
                    [('index', 'html'), ('styles', 'css')],
                )
                if 'lokale Bilder' in prompt:
                    self.assertGreaterEqual(len(image_obligations), 3)
                    self.assertTrue(
                        any(
                            item.get('dependency_contract') == 'local_visual_asset_binding'
                            for item in dependency_obligations
                        )
                    )

    def test_wrapped_counted_visual_deliverables_create_requested_image_obligations(self):
        prompt = (
            'Here is a cleaner, more natural version of the prompt.\n\n'
            'Flexible prompt\n'
            'I need a strong, atmospheric one-page website for an exclusive handmade outdoor equipment label.\n\n'
            'At the end I need exactly three images and two code files:\n'
            '3 images: one large hero image with product in misty landscape, plus two detail shots with '
            'material texture and stitching in the same look and feel.\n'
            'index.html\n'
            'styles.css\n\n'
            'Link the images locally from the code and keep the design calm and responsive.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)

        self.assertTrue(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertEqual(len(self._image_obligations(graph)), 3)
        self.assertEqual(
            [(branch.get('text_artifact_source_name'), branch.get('text_artifact_extension')) for branch in text_artifact_branches],
            [('index', 'html'), ('styles', 'css')],
        )
        self.assertEqual(len(self._text_artifact_obligations(graph)), 2)

    def test_natural_site_component_list_promotes_all_named_files_without_images(self):
        prompt = (
            'Build a polished local two-page watch atelier site with index.html, configurator.html, '
            'styles.css, and pricing.json. The pages should link to each other, share the stylesheet, '
            'and use the pricing data consistently. Save it as one complete local bundle.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            sorted(
                (item.get('target_name'), item.get('target_extension'))
                for item in self._intent_obligations(graph, 'text_artifact')
            ),
            sorted(
                [
                    ('index', 'html'),
                    ('configurator', 'html'),
                    ('styles', 'css'),
                    ('pricing', 'json'),
                ]
            ),
        )
        self.assertEqual(self._executable_image_branches(graph), [])

    def test_named_json_file_manifest_builds_required_asset_bound_branch(self):
        prompt = (
            'Create a premium local app bundle. Generate exactly four web files and three local images:\n'
            '1. index.html\n'
            '2. configurator.html\n'
            '3. pricing.json\n'
            '4. styles.css\n'
            '5. 3 Images: a watch movement, an atelier exterior, and three material samples.\n\n'
            'The pricing.json file must contain a data array with the exact local image artifact paths. '
            'Both HTML files must link the generated images and shared styles.css locally.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_branches = [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('text_artifact_extension')
        ]
        text_obligations = self._intent_obligations(graph, 'text_artifact')
        json_branch = next(
            branch for branch in text_branches
            if branch.get('text_artifact_source_name') == 'pricing'
        )
        json_dependencies = [
            obligation for obligation in self._intent_obligations(graph, 'dependency')
            if obligation.get('target_name') == 'pricing'
            and obligation.get('target_extension') == 'json'
            and obligation.get('dependency_contract') == 'local_visual_asset_binding'
        ]
        image_phase_ids = [branch['phase_id'] for branch in image_branches]

        self.assertEqual(graph['prompt_intent']['text_artifact_output_count'], 4)
        self.assertEqual(
            sorted(
                (item.get('target_name'), item.get('target_extension'))
                for item in text_obligations
            ),
            sorted(
                [
                    ('index', 'html'),
                    ('configurator', 'html'),
                    ('pricing', 'json'),
                    ('styles', 'css'),
                ]
            ),
        )
        self.assertEqual(len(image_branches), 3)
        self.assertEqual(json_branch.get('depends_on'), image_phase_ids)
        self.assertEqual(json_branch.get('dependency_contract'), 'local_visual_asset_binding')
        self.assertEqual(len(json_dependencies), 1)
        self.assertEqual(json_dependencies[0].get('source_phase_ids'), image_phase_ids)

        json_only_prompt = 'Create exactly one local file:\n1. pricing.json'
        json_only_graph = build_request_phase_graph(
            json_only_prompt,
            request_payload={'ghost_route': True, 'prompt': json_only_prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        self.assertEqual(self._executable_image_branches(json_only_graph), [])

        data_bundle_prompt = (
            'Create exactly three local artifacts:\n'
            '1. styles.css\n'
            '2. pricing.json\n'
            '3. one local image of a watch.\n'
            'Generate the image locally. styles.css and pricing.json must contain '
            'the exact local image artifact path.'
        )
        data_bundle_graph = build_request_phase_graph(
            data_bundle_prompt,
            request_payload={'ghost_route': True, 'prompt': data_bundle_prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        data_bundle_dependencies = [
            obligation for obligation in self._intent_obligations(data_bundle_graph, 'dependency')
            if obligation.get('dependency_contract') == 'local_visual_asset_binding'
        ]
        self.assertEqual(
            {
                (obligation.get('target_name'), obligation.get('target_extension'))
                for obligation in data_bundle_dependencies
            },
            {('styles', 'css'), ('pricing', 'json')},
        )

    def test_exact_current_turn_image_manifests_bind_branch_local_prompts_in_order(self):
        cases = (
            (
                'numbered_image_rows',
                (
                    'Create exactly four web files and three local images:\n'
                    '1. index.html\n'
                    '2. configurator.html\n'
                    '3. pricing.json\n'
                    '4. styles.css\n'
                    '5. One image of a hyper-detailed tourbillon movement under watchmaker-bench lighting.\n'
                    '6. One image of a glass-and-stone atelier above misty Lake Neuchâtel.\n'
                    '7. One image of brushed titanium, forged carbon, and rose gold blocks in deep-focus macro.'
                ),
                [
                    'a hyper-detailed tourbillon movement under watchmaker-bench lighting.',
                    'a glass-and-stone atelier above misty Lake Neuchâtel.',
                    'brushed titanium, forged carbon, and rose gold blocks in deep-focus macro.',
                ],
                [5, 6, 7],
            ),
            (
                'grouped_exact_count_row',
                (
                    'Create exactly four web files and three local images:\n'
                    '1. index.html\n'
                    '2. configurator.html\n'
                    '3. pricing.json\n'
                    '4. styles.css\n'
                    '5. 3 Images: One hyper-detailed macro shot of a complex tourbillon watch movement '
                    'illuminated by subtle watchmaker-bench lighting (hero), one wide cinematic exterior '
                    'of the glass-and-stone atelier overlooking the mist of Lake Neuchâtel, and one deep '
                    'focus macro shot of raw materials (brushed grade 5 titanium, forged carbon, and rose '
                    'gold blocks side-by-side).'
                ),
                [
                    'hyper-detailed macro shot of a complex tourbillon watch movement illuminated by subtle '
                    'watchmaker-bench lighting (hero)',
                    'wide cinematic exterior of the glass-and-stone atelier overlooking the mist of Lake '
                    'Neuchâtel',
                    'deep focus macro shot of raw materials (brushed grade 5 titanium, forged carbon, and '
                    'rose gold blocks side-by-side).',
                ],
                [5],
            ),
            (
                'grouped_exact_count_row',
                (
                    'Create three images: one titanium watch in crisp side light, one forged-carbon watch '
                    'against dark slate, and one rose-gold watch with a warm macro highlight.'
                ),
                [
                    'titanium watch in crisp side light',
                    'forged-carbon watch against dark slate',
                    'rose-gold watch with a warm macro highlight.',
                ],
                [],
            ),
            (
                'grouped_exact_count_row',
                (
                    'Erstelle drei Bilder: ein Titanmodell in klarem Seitenlicht, ein Carbonmodell vor '
                    'dunklem Schiefer und ein Roségoldmodell mit warmem Makrolicht.'
                ),
                [
                    'Titanmodell in klarem Seitenlicht',
                    'Carbonmodell vor dunklem Schiefer',
                    'Roségoldmodell mit warmem Makrolicht.',
                ],
                [],
            ),
            (
                'numbered_exact_count_image_section',
                (
                    'Create exactly three square cinematic macro images of the same front-facing '
                    'luxury watch case:\n'
                    '1. Brushed titanium, cool diffuse bench light, a precise cyan seconds hand.\n'
                    '2. Forged carbon, directional light revealing the layered weave.\n'
                    '3. Polished rose gold, warm controlled highlights and restrained reflections.'
                ),
                [
                    'Brushed titanium, cool diffuse bench light, a precise cyan seconds hand.',
                    'Forged carbon, directional light revealing the layered weave.',
                    'Polished rose gold, warm controlled highlights and restrained reflections.',
                ],
                [1, 2, 3],
            ),
        )

        for source_kind, prompt, expected_prompts, item_numbers in cases:
            with self.subTest(source_kind=source_kind):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'current_predecessor_context': {
                            'status': 'authorized',
                            'authorization': 'canonical_same_conversation_predecessor',
                            'batch_prompts': [
                                'stale predecessor prompt one',
                                'stale predecessor prompt two',
                                'stale predecessor prompt three',
                                'stale predecessor prompt four',
                            ],
                        },
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                image_branches = self._executable_image_branches(graph)
                refinements = [
                    item for item in (graph.get('graph_refinements') or [])
                    if item.get('source') == 'current_turn_explicit_image_manifest'
                ]

                self.assertEqual(len(image_branches), 3)
                self.assertEqual(
                    [branch.get('artifact_prompt') for branch in image_branches],
                    expected_prompts,
                )
                self.assertEqual(len(set(expected_prompts)), 3)
                self.assertTrue(
                    all(
                        branch.get('artifact_prompt_source')
                        == 'current_turn_explicit_image_manifest'
                        for branch in image_branches
                    )
                )
                self.assertTrue(
                    all(branch.get('batch_prompts') == expected_prompts for branch in image_branches)
                )
                self.assertEqual(
                    refinements,
                    [
                        {
                            'source': 'current_turn_explicit_image_manifest',
                            'refinement': 'branch_local_image_prompt_binding',
                            'status': 'bound',
                            'manifest_source_kind': source_kind,
                            'item_numbers': item_numbers,
                            'bound_prompt_count': 3,
                            'bound_branch_ids': [
                                'branch-image_generation-1',
                                'branch-image_generation-2',
                                'branch-image_generation-3',
                            ],
                        }
                    ],
                )
                self.assertFalse(
                    any(
                        branch.get('capability') == 'vision_analysis'
                        for branch in (graph.get('downstream_branches') or [])
                    )
                )

    def test_exact_count_image_section_uses_numbered_current_turn_descriptions(self):
        prompt = (
            'Generate exactly 3 new cinematic images (square format):\n'
            '1. A front-facing macro shot of a titanium watch with a cyan seconds hand.\n'
            '2. The exact same titanium watch viewed from the crown side under cool bench light.\n'
            '3. A three-quarter rear view of the same watch revealing its exhibition caseback.'
        )
        stale_prompt = 'Create one image from the selected reference reply.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'reference_artifacts': [
                    {
                        'type': 'message',
                        'message_id': 'msg-stale-reference',
                        'content': stale_prompt,
                    }
                ],
                'current_predecessor_context': {
                    'status': 'authorized',
                    'authorization': 'canonical_same_conversation_predecessor',
                    'batch_prompts': [stale_prompt, stale_prompt, stale_prompt],
                },
            },
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )
        image_branches = self._executable_image_branches(graph)
        expected_prompts = [
            'A front-facing macro shot of a titanium watch with a cyan seconds hand.',
            'The exact same titanium watch viewed from the crown side under cool bench light.',
            'A three-quarter rear view of the same watch revealing its exhibition caseback.',
        ]

        self.assertEqual(len(image_branches), 3)
        self.assertEqual(
            [branch.get('artifact_prompt') for branch in image_branches],
            expected_prompts,
        )
        self.assertTrue(
            all(
                branch.get('artifact_prompt_source')
                == 'current_turn_explicit_image_manifest'
                and branch.get('batch_prompts') == expected_prompts
                for branch in image_branches
            )
        )
        self.assertNotIn(stale_prompt, expected_prompts)
        self.assertTrue(
            any(
                item.get('manifest_source_kind')
                == 'numbered_exact_count_image_section'
                and item.get('item_numbers') == [1, 2, 3]
                for item in (graph.get('graph_refinements') or [])
            )
        )
        self.assertFalse(
            any(
                branch.get('capability') == 'vision_analysis'
                for branch in (graph.get('downstream_branches') or [])
            )
        )

    def test_exact_count_image_section_with_noncontiguous_rows_blocks(self):
        prompt = (
            'Generate exactly 3 new cinematic images (square format):\n'
            '1. A front-facing macro shot of a titanium watch.\n'
            '3. A crown-side view of the titanium watch.\n'
            '4. A rear view of the titanium watch.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )
        image_branches = self._executable_image_branches(graph)

        self.assertEqual(len(image_branches), 3)
        self.assertTrue(all(not branch.get('artifact_prompt') for branch in image_branches))
        self.assertTrue(
            all(
                branch.get('branch_contract_error')
                == 'ambiguous_current_turn_image_manifest'
                and branch.get('blocked_by_branch_contract') is True
                for branch in image_branches
            )
        )

    def test_ambiguous_grouped_image_manifest_blocks_instead_of_repeating_prompts(self):
        prompt = (
            'Create exactly three local images and one index.html file:\n'
            '1. index.html\n'
            '2. 3 Images: one tourbillon macro and one atelier exterior.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        image_branches = self._executable_image_branches(graph)

        self.assertEqual(len(image_branches), 3)
        self.assertTrue(all(not branch.get('artifact_prompt') for branch in image_branches))
        self.assertTrue(
            all(
                branch.get('branch_contract_error')
                == 'ambiguous_current_turn_image_manifest'
                and branch.get('blocked_by_branch_contract') is True
                and branch.get('repair_action') == 'repair_branch_contract'
                for branch in image_branches
            )
        )
        self.assertFalse(
            any(
                branch.get('capability') == 'vision_analysis'
                for branch in (graph.get('downstream_branches') or [])
            )
        )

    def test_local_image_assets_for_multi_page_landing_page_create_image_obligations(self):
        prompt = (
            'Erstelle eine hochwertige Landingpage fuer ein fiktives Boutique-Hotel am See namens "Lumenhof". '
            'Baue eine moderne Startseite mit Hero, drei Vorteilsbereichen, zwei Bildbereichen, '
            'Testimonials und Kontakt-CTA. Erstelle zusaetzlich eine zweite Unterseite "Suiten" '
            'mit eigener HTML-Datei, gemeinsamem CSS, Navigation zwischen beiden Seiten und passenden '
            'Bild-Assets. Alle Bilder sollen lokal generiert/eingebunden werden. Am Schluss muessen '
            'beide Seiten im Browser direkt korrekt verlinkt sein.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)
        text_extensions = {
            branch.get('text_artifact_extension')
            for branch in text_artifact_branches
        }

        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual(
            graph['prompt_intent']['inferred_visual_output_count_source'],
            'structural_visual_sections_plus_subpage_assets',
        )
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertTrue(image_phase_ids)
        self.assertTrue(
            all(branch.get('depends_on') == image_phase_ids for branch in text_artifact_branches)
        )
        self.assertTrue(
            all(branch.get('image_asset_binding_required') is True for branch in text_artifact_branches)
        )
        self.assertEqual(len(self._image_obligations(graph)), 3)
        self.assertIn('html', text_extensions)
        self.assertIn('css', text_extensions)

    def test_natural_german_local_code_and_pet_image_bundle_creates_bound_obligations(self):
        prompt = (
            'Baue eine expressive Landingpage für eine imaginäre Social-App für Haustier-Selfies '
            'namens "Petsie". Erzeuge vier unterschiedliche Tierbilder, und schreibe die Texte so, '
            'dass jedes Bild inhaltlich exakt zum jeweiligen Abschnitt passt. Die Seite soll verspielt, '
            'aber nicht kindisch wirken. HTML, CSS und Bilder müssen als lokale Artefakte sauber '
            'zusammenpassen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)
        text_branch_pairs = [
            (branch.get('text_artifact_source_name'), branch.get('text_artifact_extension'))
            for branch in text_artifact_branches
        ]

        self.assertTrue(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3, 4])
        self.assertEqual(len(self._image_obligations(graph)), 4)
        self.assertEqual(
            text_branch_pairs,
            [('generated-html', 'html'), ('generated-css', 'css')],
        )
        self.assertEqual(len(self._text_artifact_obligations(graph)), 2)

        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertTrue(image_phase_ids)
        self.assertTrue(
            all(branch.get('depends_on') == image_phase_ids for branch in text_artifact_branches)
        )
        self.assertTrue(
            all(branch.get('image_asset_binding_required') is True for branch in text_artifact_branches)
        )

        self.assertEqual(len(self._intent_obligations(graph, 'media_artifact')), 4)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 2)
        self.assertTrue(
            any(
                item.get('dependency_contract') == 'local_visual_asset_binding'
                for item in self._intent_obligations(graph, 'dependency')
            )
        )

    def test_original_dedicated_css_prompt_creates_html_and_css_obligations(self):
        prompt = (
            "Build a bold landing page with html and a dedicated css for an eco-friendly clothing line "
            "called 'Pure Thread.' Generate four images: a macro shot of organic cotton texture, a model "
            'wearing a simple white linen shirt, a close-up of sustainable recycled buttons, and a sunny '
            'outdoor scene at a botanical garden.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(
            [
                (branch.get('text_artifact_source_name'), branch.get('text_artifact_extension'))
                for branch in self._text_artifact_branches(graph)
            ],
            [('generated-html', 'html'), ('generated-css', 'css')],
        )
        self.assertEqual(len(self._text_artifact_obligations(graph)), 2)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 2)
        self.assertEqual(len(self._image_obligations(graph)), 4)

    def test_natural_german_landingpage_with_images_promotes_bound_html_obligation(self):
        prompt = (
            'Ollmo, entwirf mir eine kleine Landingpage für ein Café namens "Morgenrot". '
            'Ich hätte gerne drei stimmungsvolle Bilder: eins von der Kaffeemaschine im Einsatz, '
            'eins von einem frischen Croissant und eins von der gemütlichen Fensterbank mit Blick nach draußen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)

        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertEqual(
            [(branch.get('text_artifact_source_name'), branch.get('text_artifact_extension')) for branch in text_artifact_branches],
            [('generated-html', 'html')],
        )

        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertEqual(text_artifact_branches[0].get('depends_on'), image_phase_ids)
        self.assertTrue(text_artifact_branches[0].get('image_asset_binding_required'))
        self.assertEqual(len(self._intent_obligations(graph, 'media_artifact')), 3)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 1)
        self.assertTrue(
            any(
                item.get('dependency_contract') == 'local_visual_asset_binding'
                for item in self._intent_obligations(graph, 'dependency')
            )
        )

    def test_all_inclusive_german_landingpage_lists_images_as_bound_assets(self):
        prompt = (
            'hey. ich brauche eine kleine landingpage für ein gemeinschaftsgarten. '
            'kannst du mir etwas kleines basteln, all inclusive, d.h. bilder, html, css?'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)

        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 1)
        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertEqual(len(image_branches), 1)
        self.assertEqual(len(text_artifact_branches), 1)
        self.assertEqual(
            text_artifact_branches[0].get('depends_on'),
            [image_branches[0].get('phase_id')],
        )
        self.assertTrue(
            text_artifact_branches[0].get('image_asset_binding_required')
        )

    def test_current_predecessor_prompts_expand_and_bind_site_repair_graph(self):
        prompt = (
            'can you please create the images ange link them to the site properly? '
            'thank you.'
        )
        image_prompts = [
            f'Concrete community garden image prompt {index}.'
            for index in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / 'community-garden.html'
            html_path.write_text(
                '<html><body><img src="hero-garden.jpg"></body></html>',
                encoding='utf-8',
            )
            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'reference_artifacts': [
                        {
                            'type': 'message',
                            'message_id': 'msg-site-root',
                            'source_response_id': 'resp-site-root',
                            'content': 'Artifact generated.',
                        },
                        {
                            'type': 'text',
                            'artifact_ref': 'artifact:site-root',
                            'path': str(html_path),
                            'source_message_id': 'msg-site-root',
                            'source_response_id': 'resp-site-root',
                        },
                    ],
                    'current_predecessor_context': {
                        'status': 'authorized',
                        'authorization': (
                            'canonical_same_conversation_predecessor'
                        ),
                        'source_response_id': 'resp-site-root',
                        'source_message_id': 'msg-site-root',
                        'batch_prompts': image_prompts,
                        'text_artifact_refs': ['artifact:site-root'],
                    },
                },
                route_payload={
                    'capability': 'image_generation',
                    'route_source': 'ghost_carried',
                },
            )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)
        self.assertEqual(len(image_branches), 5)
        self.assertEqual(
            [item.get('artifact_prompt') for item in image_branches],
            image_prompts,
        )
        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertEqual(len(text_artifact_branches), 1)
        self.assertEqual(
            text_artifact_branches[0].get('depends_on'),
            [item.get('phase_id') for item in image_branches],
        )
        self.assertTrue(
            text_artifact_branches[0].get('image_asset_binding_required')
        )

    def test_natural_german_page_with_images_promotes_bound_html_obligation(self):
        prompt = (
            'Erstelle eine edle Seite für ein exklusives Chalet in Zermatt. '
            'Generiere dafür genau vier Bilder – die Außenansicht im Schnee, '
            'das brennende Cheminée im Wohnzimmer, das moderne Badezimmer und '
            'den Blick vom Balkon auf das Matterhorn.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)

        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3, 4])
        self.assertEqual(
            [(branch.get('text_artifact_source_name'), branch.get('text_artifact_extension')) for branch in text_artifact_branches],
            [('generated-html', 'html')],
        )

        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertEqual(text_artifact_branches[0].get('depends_on'), image_phase_ids)
        self.assertTrue(text_artifact_branches[0].get('image_asset_binding_required'))
        self.assertEqual(len(self._intent_obligations(graph, 'media_artifact')), 4)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 1)
        self.assertTrue(
            any(
                item.get('dependency_contract') == 'local_visual_asset_binding'
                for item in self._intent_obligations(graph, 'dependency')
            )
        )

    def test_natural_german_product_page_compound_with_images_promotes_bound_html_obligation(self):
        prompt = (
            'Ich brauche eine schlichte Produktseite für eine neue, minimalistische Tastatur '
            'namens Luna Keys. Mach dazu drei cleane Bilder: die Tastatur auf einem '
            'aufgeräumten Schreibtisch, ein Makro-Foto der Tasten und ein Bild von oben '
            'bei Nacht mit sanfter Beleuchtung.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)

        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertEqual(
            [(branch.get('text_artifact_source_name'), branch.get('text_artifact_extension')) for branch in text_artifact_branches],
            [('generated-html', 'html')],
        )

        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertEqual(text_artifact_branches[0].get('depends_on'), image_phase_ids)
        self.assertTrue(text_artifact_branches[0].get('image_asset_binding_required'))
        self.assertEqual(len(self._intent_obligations(graph, 'media_artifact')), 3)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 1)
        self.assertTrue(
            any(
                item.get('dependency_contract') == 'local_visual_asset_binding'
                for item in self._intent_obligations(graph, 'dependency')
            )
        )

    def test_natural_german_travel_page_compound_with_images_promotes_bound_html_obligation(self):
        prompt = (
            'Bau eine immersive Reiseseite für geführte Touren durch Island. '
            'Ich hätte gerne vier Bilder: einen massiven Wasserfall, ein schwarzes Lavafeld, '
            'ein einsames Zelt unter Nordlichtern und einen Wanderer, der von einer Klippe '
            'auf den Ozean schaut.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)

        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3, 4])
        self.assertEqual(
            [(branch.get('text_artifact_source_name'), branch.get('text_artifact_extension')) for branch in text_artifact_branches],
            [('generated-html', 'html')],
        )

        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertEqual(text_artifact_branches[0].get('depends_on'), image_phase_ids)
        self.assertTrue(text_artifact_branches[0].get('image_asset_binding_required'))
        self.assertEqual(len(self._intent_obligations(graph, 'media_artifact')), 4)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 1)
        self.assertTrue(
            any(
                item.get('dependency_contract') == 'local_visual_asset_binding'
                for item in self._intent_obligations(graph, 'dependency')
            )
        )

    def test_english_local_code_and_pet_image_bundle_creates_bound_obligations(self):
        prompt = (
            'Build an expressive landing page for an imaginary social app for pet selfies called "Petsie". '
            'Generate four distinct animal images. Each image must match one specific section of the page, '
            'and the copy in that section must directly refer to the animal/image shown there. '
            'The page should feel playful but not childish. Create local HTML, CSS, and image artifacts '
            'that work together cleanly. The HTML must reference the generated local images and the local '
            'CSS file correctly.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        text_artifact_branches = self._text_artifact_branches(graph)
        text_branch_pairs = [
            (branch.get('text_artifact_source_name'), branch.get('text_artifact_extension'))
            for branch in text_artifact_branches
        ]

        self.assertTrue(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertTrue(graph['prompt_intent']['local_visual_asset_requirement'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3, 4])
        self.assertEqual(len(self._image_obligations(graph)), 4)
        self.assertEqual(
            text_branch_pairs,
            [('generated-html', 'html'), ('generated-css', 'css')],
        )
        self.assertEqual(len(self._text_artifact_obligations(graph)), 2)

        image_phase_ids = [branch.get('phase_id') for branch in image_branches]
        self.assertTrue(image_phase_ids)
        self.assertTrue(
            all(branch.get('depends_on') == image_phase_ids for branch in text_artifact_branches)
        )
        self.assertTrue(
            all(branch.get('image_asset_binding_required') is True for branch in text_artifact_branches)
        )

        self.assertEqual(len(self._intent_obligations(graph, 'media_artifact')), 4)
        self.assertEqual(len(self._intent_obligations(graph, 'text_artifact')), 2)
        self.assertTrue(
            any(
                item.get('dependency_contract') == 'local_visual_asset_binding'
                for item in self._intent_obligations(graph, 'dependency')
            )
        )

    def test_generic_intent_obligation_ledger_tracks_files_assets_navigation_and_bindings(self):
        prompt = (
            'Create a small two-page website with index.html, suiten.html, shared styles.css, '
            'navigation between both pages, and exactly two generated local images linked from the pages.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        text_obligations = self._intent_obligations(graph, 'text_artifact')
        media_obligations = self._intent_obligations(graph, 'media_artifact')
        dependency_obligations = self._intent_obligations(graph, 'dependency')
        navigation_obligations = self._intent_obligations(graph, 'navigation')

        self.assertEqual(
            [(item.get('target_name'), item.get('target_extension')) for item in text_obligations],
            [('index', 'html'), ('suiten', 'html'), ('styles', 'css')],
        )
        self.assertEqual(
            [(item.get('capability'), item.get('output_type'), item.get('queue_index')) for item in media_obligations],
            [('image_generation', 'image', 1), ('image_generation', 'image', 2)],
        )
        self.assertTrue(
            any(item.get('dependency_contract') == 'shared_css_binding' for item in dependency_obligations)
        )
        self.assertTrue(
            any(item.get('dependency_contract') == 'local_visual_asset_binding' for item in dependency_obligations)
        )
        self.assertEqual(
            [(item.get('from_target_name'), item.get('to_target_name')) for item in navigation_obligations],
            [('index', 'suiten'), ('suiten', 'index')],
        )

        image_phase_ids = [
            branch.get('phase_id') for branch in self._executable_image_branches(graph)
        ]
        html_branches = [
            branch for branch in self._text_artifact_branches(graph)
            if branch.get('text_artifact_extension') == 'html'
        ]
        self.assertTrue(image_phase_ids)
        self.assertTrue(all(branch.get('depends_on') == image_phase_ids for branch in html_branches))

    def test_media_evidence_obligation_ledger_tracks_generated_image_analysis_chain(self):
        prompt = (
            'Generate an image of a glass observatory in a forest, analyze the generated image, '
            'then summarize the visible details.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        evidence_obligations = self._intent_obligations(graph, 'evidence_branch')

        self.assertEqual(
            [(item.get('capability'), item.get('dependency_contract')) for item in evidence_obligations],
            [('vision_analysis', 'media_evidence_binding'), ('chat', 'media_evidence_binding')],
        )
        self.assertTrue(
            all(item.get('depends_on_obligation_ids') for item in evidence_obligations)
        )

    def test_counted_visual_deliverable_without_generation_verb_creates_image_branches(self):
        prompt = (
            'I need exactly three images for a landing page, plus index.html and styles.css. '
            'Use one hero image and two supporting detail shots.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)

        self.assertTrue(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertEqual(len(self._image_obligations(graph)), 3)

    def test_direct_animal_selfie_batch_creates_four_image_obligations(self):
        prompt = (
            'create four different animal selfies where they look curiously close into the lense. '
            'you choose the animals and situations. go.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'image_generation', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)

        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertTrue(graph['prompt_intent']['text_preparation_before_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3, 4])
        self.assertEqual(len(self._image_obligations(graph)), 4)

    def test_not_only_output_followup_correction_creates_four_image_obligations(self):
        prompt = (
            'please. do not only output prosa (or just one image). act. '
            'create all four images you described in your response.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'reference_artifacts': [
                    {
                        'type': 'message',
                        'message_role': 'assistant',
                        'content': (
                            'I described four close-up animal selfie concepts with distinct '
                            'animals, poses, and settings.'
                        ),
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)

        self.assertFalse(graph['prompt_intent']['explicit_defer_materialization'])
        self.assertFalse(graph['prompt_intent']['explicit_visual_defer_materialization'])
        self.assertTrue(graph['prompt_intent']['requests_visual_output'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 4)
        self.assertEqual(graph['downstream_capabilities'], ['image_generation'])
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3, 4])
        self.assertFalse(
            any(branch.get('contract_state') == 'reserved' for branch in graph.get('downstream_branches') or [])
        )

    def test_selected_html_fix_promotes_target_bound_html_and_linked_css(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / 'index.html'
            styles_path = root / 'styles.css'
            index_path.write_text(
                '<!doctype html><html><head><link rel="stylesheet" href="styles.css"></head>'
                '<body><main class="page">Broken</main></body></html>',
                encoding='utf-8',
            )
            styles_path.write_text('body { scroll-template: smooth; }', encoding='utf-8')
            prompt = (
                "please fix this html and the linked css. it's broken. "
                'use the images that are already in there.'
            )

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': [
                        {
                            'type': 'text',
                            'path': str(index_path),
                            'name': 'index',
                            'artifact_ref': 'artifact:index',
                        }
                    ],
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            text_artifact_branches = self._text_artifact_branches(graph)
            branch_targets = [
                (
                    branch.get('text_artifact_extension'),
                    branch.get('text_artifact_source_name'),
                    branch.get('text_artifact_target_path'),
                    (branch.get('artifact_request') or {}).get('target_path'),
                )
                for branch in text_artifact_branches
            ]

            self.assertTrue(graph['prompt_intent']['text_revision_turn'])
            self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
            self.assertEqual(
                branch_targets,
                [
                    ('html', 'index', str(index_path), str(index_path)),
                    ('css', 'styles', str(styles_path.resolve(strict=False)), str(styles_path.resolve(strict=False))),
                ],
            )
            self.assertEqual(len(self._text_artifact_obligations(graph)), 2)

    def test_named_predecessor_files_bind_as_revision_inputs_by_exact_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configurator_path = root / 'configurator.html'
            styles_path = root / 'styles.css'
            configurator_path.write_text('<main>Original configurator</main>', encoding='utf-8')
            styles_path.write_text('.watch { color: silver; }', encoding='utf-8')
            prompt = (
                'Reference the current configurator.html and styles.css. '
                'Update configurator.html and styles.css while keeping the rest intact.'
            )

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': [
                        {
                            'type': 'text',
                            'path': str(configurator_path),
                            'artifact_ref': 'artifact:configurator',
                        },
                        {
                            'type': 'text',
                            'path': str(styles_path),
                            'artifact_ref': 'artifact:styles',
                        },
                    ],
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            branches = {
                (
                    branch.get('text_artifact_extension'),
                    branch.get('text_artifact_source_name'),
                ): branch
                for branch in self._text_artifact_branches(graph)
            }
            self.assertEqual(
                set(branches),
                {('html', 'configurator'), ('css', 'styles')},
            )
            for identity, expected_path in {
                ('html', 'configurator'): str(configurator_path),
                ('css', 'styles'): str(styles_path),
            }.items():
                branch = branches[identity]
                artifact_request = branch.get('artifact_request') or {}
                self.assertEqual(branch.get('text_artifact_source'), 'selected_source_edit')
                self.assertEqual(branch.get('text_artifact_target_path'), expected_path)
                self.assertEqual(artifact_request.get('target_path'), expected_path)
                self.assertTrue(branch.get('text_artifact_revision_required'))
                self.assertEqual(
                    branch.get('text_artifact_revision_source'),
                    'canonical_predecessor_artifact',
                )
                self.assertEqual(branch.get('text_artifact_revision_binding_state'), 'bound')
                self.assertIs(branch.get('text_artifact_source_is_input'), True)
                self.assertEqual(branch.get('content_payload_source'), 'current_phase_output')
                self.assertTrue(artifact_request.get('text_artifact_revision_required'))

    def test_material_visualizer_follow_up_preserves_revision_sources_and_image_dependencies(self):
        prompt = (
            'Use the current configurator.html and styles.css. Upgrade only the material visualizer.\n\n'
            'Create exactly three square cinematic macro images of the same front-facing luxury watch case:\n'
            '1. Brushed titanium on a dark background.\n'
            '2. Dark, textured forged carbon.\n'
            '3. Glowing rose gold.\n\n'
            'Wire the material selector to those three images. Keep all other design, copy, navigation, '
            'and shared CSS intact.'
        )
        expected_image_prompts = [
            'Brushed titanium on a dark background.',
            'Dark, textured forged carbon.',
            'Glowing rose gold.',
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configurator_path = root / 'configurator.html'
            styles_path = root / 'styles.css'
            configurator_path.write_text(
                '<main><section id="material-visualizer">Original selector</section></main>',
                encoding='utf-8',
            )
            styles_path.write_text(
                '#material-visualizer { color: silver; }',
                encoding='utf-8',
            )
            selected_references = [
                {
                    'type': 'message',
                    'message_role': 'assistant',
                    'message_id': 'message-pricing-matrix',
                    'content': '{"Titanium Base": {"price_chf": 3850}}',
                },
                {
                    'type': 'text',
                    'path': str(configurator_path),
                    'artifact_ref': 'artifact:configurator',
                },
                {
                    'type': 'text',
                    'path': str(styles_path),
                    'artifact_ref': 'artifact:styles',
                },
            ]

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': selected_references,
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            image_branches = self._executable_image_branches(graph)
            text_branches = {
                (
                    branch.get('text_artifact_extension'),
                    branch.get('text_artifact_source_name'),
                ): branch
                for branch in self._text_artifact_branches(graph)
            }
            image_phase_ids = [branch['phase_id'] for branch in image_branches]

            self.assertEqual(len(image_branches), 3)
            self.assertEqual(
                [branch.get('artifact_prompt') for branch in image_branches],
                expected_image_prompts,
            )
            self.assertEqual(len(set(expected_image_prompts)), 3)
            self.assertTrue(all(
                branch.get('artifact_prompt_source') == 'current_turn_explicit_image_manifest'
                for branch in image_branches
            ))
            self.assertTrue(all(branch.get('depends_on') == ['phase-1'] for branch in image_branches))
            self.assertTrue(all(
                {'kind': 'phase_output', 'phase_id': 'phase-1', 'role': 'dependency'}
                in (branch.get('input_refs') or [])
                for branch in image_branches
            ))
            self.assertFalse(any(
                branch.get('capability') == 'vision_analysis'
                for branch in (graph.get('downstream_branches') or [])
            ))

            expected_text_targets = {
                ('html', 'configurator'): str(configurator_path),
                ('css', 'styles'): str(styles_path),
            }
            self.assertEqual(set(text_branches), set(expected_text_targets))
            for identity, expected_path in expected_text_targets.items():
                branch = text_branches[identity]
                artifact_request = branch.get('artifact_request') or {}
                self.assertEqual(branch.get('depends_on'), image_phase_ids)
                self.assertEqual(branch.get('required_image_phase_ids'), image_phase_ids)
                self.assertEqual(
                    [item.get('phase_id') for item in (branch.get('input_refs') or [])],
                    image_phase_ids,
                )
                self.assertEqual(branch.get('text_artifact_source'), 'selected_source_edit')
                self.assertEqual(branch.get('text_artifact_target_path'), expected_path)
                self.assertTrue(branch.get('text_artifact_revision_required'))
                self.assertEqual(
                    branch.get('text_artifact_revision_source'),
                    'canonical_predecessor_artifact',
                )
                self.assertEqual(branch.get('text_artifact_revision_binding_state'), 'bound')
                self.assertIs(branch.get('text_artifact_source_is_input'), True)
                self.assertEqual(artifact_request.get('target_path'), expected_path)
                self.assertTrue(artifact_request.get('text_artifact_revision_preservation_required'))
                self.assertEqual(
                    artifact_request.get('text_artifact_revision_preservation_policy'),
                    'structural_anchor_retention_v1',
                )

            text_obligations = {
                (
                    obligation.get('text_artifact_extension'),
                    obligation.get('text_artifact_source_name'),
                ): obligation
                for obligation in self._text_artifact_obligations(graph)
            }
            self.assertEqual(set(text_obligations), set(expected_text_targets))
            for identity, expected_path in expected_text_targets.items():
                obligation = text_obligations[identity]
                artifact_request = obligation.get('artifact_request') or {}
                self.assertEqual(obligation.get('depends_on'), image_phase_ids)
                self.assertEqual(obligation.get('text_artifact_target_path'), expected_path)
                self.assertTrue(obligation.get('text_artifact_revision_required'))
                self.assertEqual(
                    obligation.get('text_artifact_revision_source'),
                    'canonical_predecessor_artifact',
                )
                self.assertEqual(obligation.get('text_artifact_revision_binding_state'), 'bound')
                self.assertTrue(artifact_request.get('text_artifact_revision_preservation_required'))
                self.assertEqual(
                    artifact_request.get('text_artifact_revision_preservation_policy'),
                    'structural_anchor_retention_v1',
                )
                self.assertIn(
                    'runtime_text_artifact_revision_write_proven_when_fulfilled',
                    obligation.get('review_criteria') or [],
                )

    def test_named_predecessor_binding_preserves_multiple_same_extension_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / 'index.html'
            configurator_path = root / 'configurator.html'
            index_path.write_text('<main>Index</main>', encoding='utf-8')
            configurator_path.write_text('<main>Configurator</main>', encoding='utf-8')
            prompt = (
                'Reference index.html and configurator.html. '
                'Update index.html and configurator.html, preserving all unrelated sections.'
            )

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': [
                        {'type': 'text', 'path': str(index_path)},
                        {'type': 'text', 'path': str(configurator_path)},
                    ],
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            html_branches = {
                branch.get('text_artifact_source_name'): branch
                for branch in self._text_artifact_branches(graph)
                if branch.get('text_artifact_extension') == 'html'
            }
            self.assertEqual(set(html_branches), {'index', 'configurator'})
            self.assertEqual(
                html_branches['index'].get('text_artifact_target_path'),
                str(index_path),
            )
            self.assertEqual(
                html_branches['configurator'].get('text_artifact_target_path'),
                str(configurator_path),
            )
            self.assertTrue(all(
                branch.get('text_artifact_revision_binding_state') == 'bound'
                for branch in html_branches.values()
            ))

    def test_selected_source_edit_does_not_promote_anonymous_output_fence_as_second_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / 'index.html'
            index_path.write_text('<main>Original</main>', encoding='utf-8')
            prompt = 'Please change the font to red.'

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': [
                        {'type': 'text', 'path': str(index_path)},
                    ],
                },
                response_payload={
                    'output_text': (
                        '```html\n<!doctype html><style>body{color:red}</style>'
                        '<h1>Hello</h1>\n```'
                    ),
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            branches = self._text_artifact_branches(graph)
            self.assertEqual(len(branches), 1)
            self.assertEqual(branches[0].get('text_artifact_source_name'), 'index')
            self.assertEqual(
                branches[0].get('text_artifact_target_path'),
                str(index_path),
            )
            self.assertNotEqual(
                branches[0].get('text_artifact_source_name'),
                'updated-html',
            )

    def test_named_predecessor_binding_does_not_guess_from_same_extension_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / 'index.html'
            index_path.write_text('<main>Index</main>', encoding='utf-8')
            prompt = 'Reference the prior work and update configurator.html only.'

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': [
                        {'type': 'text', 'path': str(index_path)},
                    ],
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            branches = self._text_artifact_branches(graph)
            self.assertEqual(
                [
                    (
                        branch.get('text_artifact_extension'),
                        branch.get('text_artifact_source_name'),
                    )
                    for branch in branches
                ],
                [('html', 'configurator')],
            )
            self.assertIsNone(branches[0].get('text_artifact_target_path'))
            self.assertIsNone(branches[0].get('text_artifact_revision_required'))
            self.assertFalse(branches[0].get('blocked_by_branch_contract'))

    def test_named_predecessor_binding_blocks_ambiguous_same_name_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / 'first' / 'configurator.html'
            second_path = root / 'second' / 'configurator.html'
            first_path.parent.mkdir()
            second_path.parent.mkdir()
            first_path.write_text('<main>First</main>', encoding='utf-8')
            second_path.write_text('<main>Second</main>', encoding='utf-8')
            prompt = 'Reference the prior work and update configurator.html.'

            graph = build_request_phase_graph(
                prompt,
                request_payload={
                    'ghost_route': True,
                    'prompt': prompt,
                    'selected_reference_artifacts': [
                        {'type': 'text', 'path': str(first_path)},
                        {'type': 'text', 'path': str(second_path)},
                    ],
                },
                route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            )

            branches = self._text_artifact_branches(graph)
            self.assertEqual(len(branches), 1)
            branch = branches[0]
            self.assertEqual(branch.get('text_artifact_source_name'), 'configurator')
            self.assertIsNone(branch.get('text_artifact_target_path'))
            self.assertTrue(branch.get('text_artifact_revision_required'))
            self.assertEqual(
                branch.get('text_artifact_revision_source'),
                'canonical_predecessor_artifact',
            )
            self.assertEqual(branch.get('text_artifact_revision_binding_state'), 'ambiguous')
            self.assertIs(branch.get('text_artifact_source_is_input'), False)
            self.assertEqual(
                branch.get('branch_contract_error'),
                'ambiguous_text_artifact_revision_source',
            )
            self.assertTrue(branch.get('blocked_by_branch_contract'))

    def test_counted_visual_branch_guard_backfills_missing_explicit_branch_count(self):
        prompt = 'I need exactly three images for this landing page.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'downstream_branches': [
                    {
                        'branch_id': 'branch-image_generation-1',
                        'phase_id': 'phase-2',
                        'capability': 'image_generation',
                        'output_type': 'image',
                        'depends_on': ['phase-1'],
                        'queue_index': 1,
                        'source': 'explicit_test_branch',
                    },
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)
        guard_refinements = [
            item for item in (graph.get('graph_refinements') or [])
            if item.get('refinement') == 'explicit_visual_obligation_guard'
        ]

        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])
        self.assertEqual(len(self._image_obligations(graph)), 3)
        self.assertEqual(len(guard_refinements), 1)
        self.assertEqual(guard_refinements[0].get('requested_count'), 3)
        self.assertEqual(guard_refinements[0].get('existing_count'), 1)
        self.assertEqual(guard_refinements[0].get('added_count'), 2)

    def test_counted_input_image_review_does_not_create_generation_branches(self):
        prompt = 'Describe exactly three images in the attached gallery and compare their visible details.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
        self.assertEqual(self._executable_image_branches(graph), [])

    def test_counted_visual_deliverable_works_with_german_prompt_without_special_case(self):
        prompt = (
            'Am Ende brauche ich genau drei Bilder und zwei Code-Dateien: '
            '3 Bilder: Hero, Materialdetail und Verarbeitung. index.html styles.css.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = self._executable_image_branches(graph)

        self.assertTrue(graph['prompt_intent']['counted_visual_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 3)
        self.assertEqual([branch.get('queue_index') for branch in image_branches], [1, 2, 3])

    def test_true_german_image_deferral_still_reserves_without_executable_image_branch(self):
        prompt = 'Erstelle zuerst nur den Text für eine Landing Page. Generiere noch keine Bilder.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        image_branches = [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('capability') == 'image_generation'
        ]
        executable_image_branches = [
            branch for branch in image_branches
            if branch.get('contract_state') != 'reserved'
        ]

        self.assertTrue(graph['prompt_intent']['explicit_visual_defer_materialization'])
        self.assertEqual(executable_image_branches, [])
        self.assertEqual(
            [(branch.get('branch_id'), branch.get('contract_state')) for branch in image_branches],
            [('branch-image_generation-reserved-1', 'reserved')],
        )
        self.assertNotIn(
            'image_generation',
            graph['prompt_intent']['required_intent_capabilities'],
        )
        self.assertNotIn('image', graph['prompt_intent']['required_intent_output_counts'])

    def test_audio_transcript_compare_with_original_keeps_root_text_dependency(self):
        prompt = (
            'Schreibe einen deutschen Satz mit genau 11 Wörtern über lokale KI. '
            'Lies ihn als Audio vor. Transkribiere danach das Audio und vergleiche Transkript mit Original.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            ['text_to_speech', 'speech_to_text', 'chat'],
        )
        self.assertEqual(branches[2].get('depends_on'), ['phase-1', branches[1].get('phase_id')])

    def test_dependency_join_prompt_intent_projects_required_multimodal_obligations(self):
        prompt = (
            'Schreibe einen deutschen Szenentext mit genau zwanzig Wörtern über einen Leuchtturm im Sturm. '
            'Erzeuge daraus parallel ein Bild und ein Audio. Analysiere danach das tatsächlich erzeugte Bild '
            'und transkribiere das tatsächlich erzeugte Audio. Vergleiche abschließend im Chat anhand genau '
            'dieser beiden realen Evidenzzweigen, ob Leuchtturm, Sturm und Nacht in beiden vorkommen. '
            'Bildanalyse darf nur vom Bild, Transkription nur vom Audio, der Schluss nur von beiden '
            'Evidenzzweigen abhängen.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        prompt_intent = graph['prompt_intent']
        self.assertTrue(prompt_intent['requests_visual_output'])
        self.assertEqual(prompt_intent['requested_visual_output_count'], 1)
        self.assertTrue(prompt_intent['text_preparation_before_visual_output'])
        self.assertTrue(prompt_intent['text_preparation_before_audio_output'])
        self.assertEqual(
            prompt_intent['required_intent_output_counts'],
            {'image': 1, 'audio': 1},
        )
        self.assertEqual(
            prompt_intent['required_intent_capability_counts'],
            {
                'image_generation': 1,
                'text_to_speech': 1,
                'vision_analysis': 1,
                'speech_to_text': 1,
                'chat': 1,
            },
        )
        self.assertEqual(
            prompt_intent['required_intent_capabilities'],
            [
                'image_generation',
                'text_to_speech',
                'vision_analysis',
                'speech_to_text',
                'chat',
            ],
        )
        self.assertEqual(
            prompt_intent['downstream_follow_up_capabilities'],
            prompt_intent['required_intent_capabilities'],
        )

    def test_r4_root_final_json_joins_every_requested_media_and_evidence_producer(self):
        prompt = (
            'Erzeuge ein lokales Bild eines kleinen Observatoriums bei klarem Nachthimmel und analysiere '
            'danach nur sichtbare Details dieses Bildes. Schreibe außerdem eine deutsche Erzählung aus '
            'genau zwei kurzen Sätzen, erzeuge daraus ein Audio und transkribiere das tatsächlich erzeugte '
            'Audio. Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare '
            'Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph.get('downstream_branches') or []
        by_capability = {
            capability: [
                branch for branch in branches
                if branch.get('capability') == capability
            ]
            for capability in {
                'image_generation',
                'vision_analysis',
                'text_to_speech',
                'speech_to_text',
            }
        }
        final_branches = [
            branch for branch in branches
            if branch.get('capability') == 'chat'
            and branch.get('dependency_contract') == 'structured_multi_evidence_join'
        ]

        self.assertEqual(len(final_branches), 1)
        final_branch = final_branches[0]
        self.assertEqual(
            final_branch.get('depends_on'),
            [
                by_capability['image_generation'][0]['phase_id'],
                by_capability['vision_analysis'][0]['phase_id'],
                by_capability['text_to_speech'][0]['phase_id'],
                by_capability['speech_to_text'][0]['phase_id'],
            ],
        )
        self.assertEqual(
            [item.get('phase_id') for item in final_branch.get('input_refs') or []],
            final_branch.get('depends_on'),
        )
        self.assertFalse(any(
            branch.get('branch_id') == 'branch-text_artifact-1'
            for branch in branches
        ))
        self.assertEqual(graph['prompt_intent']['text_artifact_output_count'], 0)

        intermediate_visual_text = next(
            branch for branch in branches
            if branch.get('capability') == 'chat'
            and branch.get('dependency_contract') != 'structured_multi_evidence_join'
        )
        self.assertEqual(
            intermediate_visual_text.get('depends_on'),
            [by_capability['vision_analysis'][0]['phase_id']],
        )

    def test_structured_final_join_ignores_negated_and_quoted_contracts(self):
        root = (
            'Erzeuge ein lokales Bild eines kleinen Observatoriums bei klarem Nachthimmel und analysiere '
            'danach nur sichtbare Details dieses Bildes. Schreibe außerdem eine deutsche Erzählung aus '
            'genau zwei kurzen Sätzen, erzeuge daraus ein Audio und transkribiere das tatsächlich erzeugte '
            'Audio. '
        )
        endings = (
            'Gib abschließend kein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare Bildevidenz, '
            'den Audio-artifact_ref und das reale Transkript getrennt bindet; fasse die Ergebnisse '
            'stattdessen knapp zusammen.',
            'Die Formulierung "Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die '
            'sichtbare Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet" ist '
            'nur ein Beispiel; fasse die Ergebnisse knapp zusammen.',
            'Schreibe wörtlich: "Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die '
            'sichtbare Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet"; '
            'fasse danach die Ergebnisse knapp zusammen.',
            'Fasse abschließend die Ergebnisse knapp zusammen und schreibe wörtlich: '
            '"Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare '
            'Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet".',
            'Fasse abschließend die Ergebnisse knapp zusammen und schreibe wörtlich: '
            '„Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare '
            'Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet“.',
            'Fasse abschließend die Ergebnisse knapp zusammen und schreibe wörtlich: '
            '`Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare '
            'Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet`.',
            "Fasse abschließend die Ergebnisse knapp zusammen und schreibe wörtlich: "
            "'Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare "
            "Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet'.",
            'Gib abschließend ein JSON-Objekt aus, das den Bild-artifact_ref, die sichtbare Bildevidenz, '
            'den Audio-artifact_ref und das reale Transkript bindet ausdrücklich nicht; fasse die '
            'Ergebnisse stattdessen knapp zusammen.',
            'Gib abschließend ein JSON-Objekt aus, das nicht ausschließlich den ersten '
            'Bild-artifact_ref, dessen sichtbare Bildevidenz, sondern beide Bild-artifact_refs, den '
            'Audio-artifact_ref und das reale Transkript bindet.',
        )

        for ending in endings:
            with self.subTest(ending=ending):
                prompt = root + ending
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                branches = graph.get('downstream_branches') or []
                stt_branch = next(
                    branch for branch in branches
                    if branch.get('capability') == 'speech_to_text'
                )
                final_branch = branches[-1]

                self.assertEqual(final_branch.get('capability'), 'chat')
                self.assertNotEqual(
                    final_branch.get('dependency_contract'),
                    'structured_multi_evidence_join',
                )
                self.assertEqual(final_branch.get('depends_on'), [stt_branch['phase_id']])

    def test_structured_final_join_requires_active_executable_producer_and_evidence_contracts(self):
        branches = [
            {
                'branch_id': 'branch-image_generation-1',
                'phase_id': 'phase-2',
                'capability': 'image_generation',
                'depends_on': ['phase-1'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-vision_analysis-1',
                'phase_id': 'phase-3',
                'capability': 'vision_analysis',
                'depends_on': ['phase-2'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-text_to_speech-1',
                'phase_id': 'phase-4',
                'capability': 'text_to_speech',
                'depends_on': ['phase-1'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-speech_to_text-1',
                'phase_id': 'phase-5',
                'capability': 'speech_to_text',
                'depends_on': ['phase-4'],
                'resolution': 'pending_dependency',
            },
        ]
        final_contract = (
            'abschließend ein json-objekt, das den bild-artifact_ref, dessen sichtbare bildevidenz, '
            'den audio-artifact_ref und das reale transkript bindet'
        )

        self.assertEqual(
            _structured_final_join_selected_phase_ids(branches, final_contract),
            ['phase-2', 'phase-3', 'phase-4', 'phase-5'],
        )
        invalid_contracts = (
            (0, 'contract_state', 'candidate'),
            (2, 'contract_state', 'reserved'),
            (1, 'status', 'failed'),
            (3, 'status', 'superseded'),
            (0, 'contract_state', {'unexpected': 'mapping'}),
            (1, 'resolution', ['pending_dependency']),
        )
        for branch_index, field, value in invalid_contracts:
            with self.subTest(branch_index=branch_index, field=field, value=value):
                invalid_branches = deepcopy(branches)
                invalid_branches[branch_index][field] = value
                self.assertEqual(
                    _structured_final_join_selected_phase_ids(
                        invalid_branches,
                        final_contract,
                    ),
                    [],
                )

    def test_structured_final_join_ordinal_selection_uses_unique_explicit_indexes(self):
        final_contract = (
            'abschließend ein json-objekt, das ausschließlich den ersten bild-artifact_ref, dessen '
            'sichtbare bildevidenz, den audio-artifact_ref und das reale transkript bindet'
        )

        for index_field in ('queue_index', 'candidate_selection_index'):
            with self.subTest(index_field=index_field):
                branches = [
                    {
                        'branch_id': 'branch-image_generation-2',
                        'phase_id': 'phase-3',
                        'capability': 'image_generation',
                        'depends_on': ['phase-1'],
                        'resolution': 'pending_dependency',
                        index_field: 2,
                    },
                    {
                        'branch_id': 'branch-image_generation-1',
                        'phase_id': 'phase-2',
                        'capability': 'image_generation',
                        'depends_on': ['phase-1'],
                        'resolution': 'pending_dependency',
                        index_field: 1,
                    },
                    {
                        'branch_id': 'branch-vision_analysis-2',
                        'phase_id': 'phase-5',
                        'capability': 'vision_analysis',
                        'depends_on': ['phase-3'],
                        'resolution': 'pending_dependency',
                    },
                    {
                        'branch_id': 'branch-vision_analysis-1',
                        'phase_id': 'phase-4',
                        'capability': 'vision_analysis',
                        'depends_on': ['phase-2'],
                        'resolution': 'pending_dependency',
                    },
                    {
                        'branch_id': 'branch-text_to_speech-1',
                        'phase_id': 'phase-6',
                        'capability': 'text_to_speech',
                        'depends_on': ['phase-1'],
                        'resolution': 'pending_dependency',
                    },
                    {
                        'branch_id': 'branch-speech_to_text-1',
                        'phase_id': 'phase-7',
                        'capability': 'speech_to_text',
                        'depends_on': ['phase-6'],
                        'resolution': 'pending_dependency',
                    },
                ]

                self.assertEqual(
                    _structured_final_join_selected_phase_ids(branches, final_contract),
                    ['phase-2', 'phase-4', 'phase-6', 'phase-7'],
                )

                for invalid_indexes in ((None, 2), (1, 1), ('first', 2)):
                    with self.subTest(
                        index_field=index_field,
                        invalid_indexes=invalid_indexes,
                    ):
                        invalid_branches = deepcopy(branches)
                        for producer, value in zip(invalid_branches[:2], invalid_indexes):
                            if value is None:
                                producer.pop(index_field, None)
                            else:
                                producer[index_field] = value
                        self.assertEqual(
                            _structured_final_join_selected_phase_ids(
                                invalid_branches,
                                final_contract,
                            ),
                            [],
                        )

    def test_structured_final_join_rejects_evidence_bound_to_selected_and_sibling_producers(self):
        branches = [
            {
                'branch_id': 'branch-image_generation-1',
                'phase_id': 'phase-2',
                'capability': 'image_generation',
                'depends_on': ['phase-1'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-image_generation-2',
                'phase_id': 'phase-3',
                'capability': 'image_generation',
                'depends_on': ['phase-1'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-vision_analysis-1',
                'phase_id': 'phase-4',
                'capability': 'vision_analysis',
                'depends_on': ['phase-2', 'phase-3'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-vision_analysis-2',
                'phase_id': 'phase-5',
                'capability': 'vision_analysis',
                'depends_on': ['phase-3'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-text_to_speech-1',
                'phase_id': 'phase-6',
                'capability': 'text_to_speech',
                'depends_on': ['phase-1'],
                'resolution': 'pending_dependency',
            },
            {
                'branch_id': 'branch-speech_to_text-1',
                'phase_id': 'phase-7',
                'capability': 'speech_to_text',
                'depends_on': ['phase-6'],
                'resolution': 'pending_dependency',
            },
        ]
        final_contract = (
            'abschließend ein json-objekt, das ausschließlich den ersten bild-artifact_ref, dessen '
            'sichtbare bildevidenz, den audio-artifact_ref und das reale transkript bindet'
        )

        self.assertEqual(
            _structured_final_join_selected_phase_ids(branches, final_contract),
            [],
        )

    def test_structured_final_join_binds_only_selected_image_evidence_pair(self):
        for ordinal, selected_index in (('ersten', 0), ('zweiten', 1)):
            with self.subTest(ordinal=ordinal):
                prompt = (
                    'Erzeuge zwei lokale Bilder eines kleinen Observatoriums bei klarem Nachthimmel und '
                    'analysiere danach beide erzeugten Bilder nur auf sichtbare Details. Schreibe außerdem '
                    'eine deutsche Erzählung aus genau zwei kurzen Sätzen, erzeuge daraus ein Audio und '
                    'transkribiere das tatsächlich erzeugte Audio. Gib abschließend ein JSON-Objekt aus, '
                    f'das ausschließlich den {ordinal} Bild-artifact_ref, dessen sichtbare Bildevidenz, '
                    'den Audio-artifact_ref und das reale Transkript bindet.'
                )

                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                branches = graph.get('downstream_branches') or []
                images = [
                    branch for branch in branches
                    if branch.get('capability') == 'image_generation'
                ]
                vision = [
                    branch for branch in branches
                    if branch.get('capability') == 'vision_analysis'
                ]
                audio = next(
                    branch for branch in branches
                    if branch.get('capability') == 'text_to_speech'
                )
                transcript = next(
                    branch for branch in branches
                    if branch.get('capability') == 'speech_to_text'
                )
                final_branch = branches[-1]

                self.assertEqual(
                    final_branch.get('depends_on'),
                    [
                        images[selected_index]['phase_id'],
                        vision[selected_index]['phase_id'],
                        audio['phase_id'],
                        transcript['phase_id'],
                    ],
                )
                self.assertEqual(
                    final_branch.get('dependency_contract'),
                    'structured_multi_evidence_join',
                )

    def test_structured_final_join_binds_only_selected_audio_evidence_pair(self):
        prompt = (
            'Erzeuge ein lokales Bild eines Observatoriums und analysiere danach nur sichtbare Details '
            'dieses Bildes. Schreibe zwei kurze Erzählvarianten, erzeuge daraus zwei getrennte Audios und '
            'transkribiere beide tatsächlich erzeugten Audios separat. Gib abschließend ein JSON-Objekt '
            'aus, das ausschließlich den zweiten Audio-artifact_ref, dessen reales Transkript, den '
            'Bild-artifact_ref und die sichtbare Bildevidenz bindet.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        branches = graph.get('downstream_branches') or []
        image = next(
            branch for branch in branches
            if branch.get('capability') == 'image_generation'
        )
        vision = next(
            branch for branch in branches
            if branch.get('capability') == 'vision_analysis'
        )
        audio = [
            branch for branch in branches
            if branch.get('capability') == 'text_to_speech'
        ]
        transcripts = [
            branch for branch in branches
            if branch.get('capability') == 'speech_to_text'
        ]
        final_branch = branches[-1]

        self.assertEqual(
            final_branch.get('depends_on'),
            [image['phase_id'], vision['phase_id'], audio[1]['phase_id'], transcripts[1]['phase_id']],
        )
        self.assertEqual(
            final_branch.get('dependency_contract'),
            'structured_multi_evidence_join',
        )

    def test_structured_final_join_does_not_guess_ambiguous_media_selection(self):
        prompt = (
            'Erzeuge zwei lokale Bilder eines kleinen Observatoriums und analysiere danach beide erzeugten '
            'Bilder nur auf sichtbare Details. Schreibe eine kurze deutsche Erzählung, erzeuge daraus ein '
            'Audio und transkribiere das tatsächlich erzeugte Audio. Gib abschließend ein JSON-Objekt aus, '
            'das den Bild-artifact_ref, die sichtbare Bildevidenz, den Audio-artifact_ref und das reale '
            'Transkript bindet.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        branches = graph.get('downstream_branches') or []
        stt_branch = next(
            branch for branch in branches
            if branch.get('capability') == 'speech_to_text'
        )
        final_branch = branches[-1]

        self.assertNotEqual(
            final_branch.get('dependency_contract'),
            'structured_multi_evidence_join',
        )
        self.assertEqual(final_branch.get('depends_on'), [stt_branch['phase_id']])

    def test_explicit_json_artifact_structured_join_merges_into_final_evidence_branch(self):
        prompt = (
            'Erzeuge ein lokales Bild eines kleinen Observatoriums bei klarem Nachthimmel und analysiere '
            'danach nur sichtbare Details dieses Bildes. Schreibe außerdem eine deutsche Erzählung aus '
            'genau zwei kurzen Sätzen, erzeuge daraus ein Audio und transkribiere das tatsächlich erzeugte '
            'Audio. Gib abschließend ein JSON-Objekt als Datei-Artefakt aus, das den Bild-artifact_ref, die '
            'sichtbare Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        branches = graph.get('downstream_branches') or []
        final_branches = [
            branch for branch in branches
            if branch.get('dependency_contract') == 'structured_multi_evidence_join'
        ]

        self.assertEqual(len(final_branches), 1)
        final_branch = final_branches[0]
        expected_dependencies = [
            next(
                branch['phase_id'] for branch in branches
                if branch.get('capability') == capability
            )
            for capability in (
                'image_generation',
                'vision_analysis',
                'text_to_speech',
                'speech_to_text',
            )
        ]
        self.assertEqual(final_branch.get('depends_on'), expected_dependencies)
        self.assertTrue(final_branch.get('requires_artifact'))
        self.assertEqual(final_branch.get('text_artifact_extension'), 'json')
        self.assertEqual(
            final_branch.get('stage_direction'),
            'materialize_requested_json_after_artifact_evidence',
        )
        self.assertEqual(graph['prompt_intent']['text_artifact_output_count'], 1)
        self.assertFalse(any(
            branch.get('branch_id') == 'branch-text_artifact-1'
            for branch in branches
        ))

    def test_structured_json_artifact_coalesces_when_readme_sibling_is_requested(self):
        prompt = (
            'Erzeuge ein lokales Bild eines kleinen Observatoriums bei klarem Nachthimmel und analysiere '
            'danach nur sichtbare Details dieses Bildes. Schreibe außerdem eine deutsche Erzählung aus '
            'genau zwei kurzen Sätzen, erzeuge daraus ein Audio und transkribiere das tatsächlich erzeugte '
            'Audio. Gib abschließend ein JSON-Objekt als Datei-Artefakt aus, das den Bild-artifact_ref, die '
            'sichtbare Bildevidenz, den Audio-artifact_ref und das reale Transkript getrennt bindet. '
            'Erstelle außerdem README.md als separates Datei-Artefakt.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )
        branches = graph.get('downstream_branches') or []
        final_branch = next(
            branch for branch in branches
            if branch.get('dependency_contract') == 'structured_multi_evidence_join'
        )
        text_artifact_siblings = [
            branch for branch in branches
            if branch.get('role') == 'text_artifact_output'
        ]

        self.assertTrue(final_branch.get('requires_artifact'))
        self.assertEqual(final_branch.get('text_artifact_extension'), 'json')
        self.assertNotIn('phase-1', final_branch.get('depends_on') or [])
        self.assertEqual(len(text_artifact_siblings), 1)
        self.assertEqual(text_artifact_siblings[0].get('text_artifact_extension'), 'md')
        self.assertEqual(text_artifact_siblings[0].get('text_artifact_source_name'), 'README')
        self.assertFalse(any(
            branch is not final_branch
            and branch.get('text_artifact_extension') == 'json'
            for branch in branches
        ))
        self.assertEqual(graph['prompt_intent']['text_artifact_output_count'], 2)

    def test_german_preserved_visual_follow_up_builds_two_audio_transcript_pairs(self):
        prompt = (
            'Beziehe dich ausdrücklich auf das Observatorium-Bild, seine Bildanalyse, die deutsche '
            'Erzählung und das Audio aus dem unmittelbar vorherigen Turn. Bewahre Bild und Bildanalyse '
            'unverändert; erzeuge das Bild nicht neu und analysiere es nicht erneut. Ersetze den bisherigen '
            'einzelnen Audiozweig durch zwei getrennte Audiofassungen: einmal die ursprüngliche deutsche '
            'Erzählung und einmal eine getreue englische Übersetzung. Transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein neues JSON-Objekt aus, das die unveränderte Bildevidenz '
            'sowie beide Audio-artifact_refs und beide realen Transkripte eindeutig verbindet.'
        )

        reference_artifacts = [
            {
                'type': 'message',
                'message_id': 'msg-observatory-root',
                'source_response_id': 'resp-observatory-root',
                'content': (
                    '**Sichtbare Details:**\n'
                    'Die erhaltene Kuppel steht auf einem dunklen Hügel unter der Milchstraße.\n\n'
                    '**Deutsche Erzählung:**\nDie stille Kuppel blickte in die Nacht.'
                ),
            },
            {
                'type': 'image',
                'kind': 'image',
                'artifact_ref': 'artifact:image-observatory-root',
                'path': '/tmp/observatory-root.png',
                'source_message_id': 'msg-observatory-root',
                'source_response_id': 'resp-observatory-root',
            },
        ]
        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'reference_artifacts': reference_artifacts,
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        prompt_intent = graph['prompt_intent']
        self.assertTrue(prompt_intent['visual_artifact_preservation_without_regeneration'])
        self.assertTrue(prompt_intent['visual_analysis_preservation_without_reanalysis'])
        self.assertFalse(prompt_intent['requests_visual_output'])
        self.assertEqual(prompt_intent['requested_visual_output_count'], 0)
        self.assertTrue(prompt_intent['requests_audio_output'])
        self.assertEqual(prompt_intent['requested_audio_output_count'], 2)
        self.assertTrue(prompt_intent['counted_audio_output_obligation'])

        branches = graph.get('downstream_branches') or []
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            [
                'text_to_speech',
                'text_to_speech',
                'speech_to_text',
                'speech_to_text',
                'chat',
            ],
        )
        self.assertNotIn('image_generation', [branch.get('capability') for branch in branches])
        self.assertNotIn('vision_analysis', [branch.get('capability') for branch in branches])

        tts_branches = [branch for branch in branches if branch.get('capability') == 'text_to_speech']
        stt_branches = [branch for branch in branches if branch.get('capability') == 'speech_to_text']
        final_branch = branches[-1]
        self.assertEqual([branch.get('branch_id') for branch in tts_branches], [
            'branch-text_to_speech-1',
            'branch-text_to_speech-2',
        ])
        self.assertEqual([branch.get('queue_index') for branch in tts_branches], [1, 2])
        self.assertEqual([branch.get('candidate_selection_index') for branch in tts_branches], [1, 2])
        self.assertEqual([branch.get('candidate_selection_count') for branch in tts_branches], [2, 2])
        self.assertEqual(
            [branch.get('selection_policy') for branch in tts_branches],
            ['selected_candidate_only', 'selected_candidate_only'],
        )
        self.assertEqual([branch.get('lang_code') for branch in tts_branches], ['de', 'en'])
        self.assertEqual(
            [branch.get('audio_variant_role') for branch in tts_branches],
            ['original_narration', 'faithful_translation'],
        )
        self.assertEqual(
            [branch.get('content_payload_source') for branch in tts_branches],
            ['selected_candidate_from_phase_output', 'selected_candidate_from_phase_output'],
        )
        self.assertEqual(
            [branch.get('depends_on') for branch in stt_branches],
            [[tts_branches[0]['phase_id']], [tts_branches[1]['phase_id']]],
        )
        self.assertEqual(
            final_branch.get('depends_on'),
            [
                tts_branches[0]['phase_id'],
                stt_branches[0]['phase_id'],
                tts_branches[1]['phase_id'],
                stt_branches[1]['phase_id'],
            ],
        )
        self.assertEqual(
            final_branch.get('dependency_contract'),
            'structured_multi_evidence_join',
        )
        selected_refs = [
            item for item in (final_branch.get('input_refs') or [])
            if item.get('kind') == 'selected_reference'
        ]
        self.assertEqual(
            [item.get('role') for item in selected_refs],
            ['preserved_visual_artifact', 'preserved_visual_evidence'],
        )
        self.assertEqual(
            selected_refs[0].get('artifact_ref'),
            'artifact:image-observatory-root',
        )
        structured_contract = final_branch.get('structured_output_contract') or {}
        self.assertEqual(structured_contract.get('format'), 'json_object')
        self.assertEqual(structured_contract.get('cardinality'), 'exactly_one')
        self.assertEqual(
            [
                binding.get('field_name')
                for binding in structured_contract.get('required_bindings') or []
                if binding.get('field_name')
            ],
            [
                'preserved_visual_evidence',
                'audio_variant_1_artifact_ref',
                'audio_variant_1_transcript',
                'audio_variant_2_artifact_ref',
                'audio_variant_2_transcript',
            ],
        )
        self.assertEqual(
            prompt_intent['required_intent_output_counts'],
            {'audio': 2},
        )
        self.assertEqual(
            prompt_intent['required_intent_capability_counts'],
            {'text_to_speech': 2, 'speech_to_text': 2, 'chat': 1},
        )
        self.assertFalse(prompt_intent['requests_text_artifact_output'])
        self.assertNotIn('text_artifact_extension', final_branch)
        self.assertIsNot(final_branch.get('requires_artifact'), True)

        selected_reference_payload = (
            LateFillRuntimeOwner._selected_reference_dependency_payload(
                final_branch,
                current_payload={'reference_artifacts': reference_artifacts},
            )
        )
        selected_evidence = '\n'.join(
            selected_reference_payload.get('evidence_blocks') or []
        )
        self.assertIn('artifact:image-observatory-root', selected_evidence)
        self.assertIn('Die erhaltene Kuppel steht', selected_evidence)
        self.assertNotIn('Deutsche Erzählung', selected_evidence)
        self.assertTrue(
            selected_reference_payload.get('selected_reference_evidence_bound')
        )

        final_instruction = LateFillRuntimeOwner._post_artifact_follow_up_instruction(
            prompt,
            evidence=selected_evidence,
            structured_output_contract=structured_contract,
        )
        self.assertIn('Return exactly one JSON object and nothing else', final_instruction)
        self.assertIn('`audio_variant_1_artifact_ref`', final_instruction)
        self.assertNotIn('compare those strings directly', final_instruction)

        owner = object.__new__(LateFillRuntimeOwner)
        owner.normalize_capability = lambda value: str(value or '').strip() or None
        owner.capability_image_generation = 'image_generation'
        owner.capability_text_to_speech = 'text_to_speech'
        owner.build_canonical_response_artifacts = lambda payload: list(
            payload.get('artifacts') or []
        )
        fill_results = []
        current_artifacts = []
        for index, (tts_branch, stt_branch) in enumerate(
            zip(tts_branches, stt_branches),
            start=1,
        ):
            audio_artifact = {
                'type': 'audio',
                'kind': 'audio',
                'path': f'/tmp/audio-{index}.wav',
                'artifact_ref': f'artifact:audio-{index}',
                'phase_id': tts_branch['phase_id'],
                'branch_id': tts_branch['branch_id'],
            }
            current_artifacts.append(audio_artifact)
            fill_results.extend(
                [
                    {
                        'branch_id': tts_branch['branch_id'],
                        'phase_id': tts_branch['phase_id'],
                        'capability': 'text_to_speech',
                        'artifacts': [audio_artifact],
                    },
                    {
                        'branch_id': stt_branch['branch_id'],
                        'phase_id': stt_branch['phase_id'],
                        'capability': 'speech_to_text',
                        'result_text': f'Reales Transkript {index}.',
                        'execution_contract': {
                            'depends_on': [tts_branch['phase_id']],
                        },
                    },
                ]
            )
        dependency_payload = owner.branch_dependency_payload(
            final_branch,
            current_payload={
                'request': {
                    'prompt': prompt,
                    'reference_artifacts': reference_artifacts,
                },
                'runtime': {'request_phase_graph': graph},
                'artifacts': current_artifacts,
                'late_fill': {'fill_results': fill_results},
            },
        )
        dependency_evidence = dependency_payload.get('content_payload') or ''
        for expected_value in (
            'artifact:image-observatory-root',
            'Die erhaltene Kuppel steht',
            'artifact:audio-1',
            'Reales Transkript 1.',
            'artifact:audio-2',
            'Reales Transkript 2.',
        ):
            self.assertIn(expected_value, dependency_evidence)
        self.assertEqual(
            [item.get('artifact_ref') for item in dependency_payload['reference_artifacts']],
            ['artifact:image-observatory-root'],
        )
        self.assertTrue(dependency_payload['selected_reference_evidence_bound'])

    def test_preserved_visual_structured_join_fails_closed_without_exact_reference(self):
        prompt = (
            'Bewahre Bild und Bildanalyse unverändert; erzeuge das Bild nicht neu und analysiere es '
            'nicht erneut. Erzeuge zwei getrennte Audiofassungen, transkribiere beide tatsächlich '
            'erzeugten Audios und gib ein JSON-Objekt aus, das die unveränderte Bildevidenz sowie '
            'beide Audio-artifact_refs und beide Transkripte eindeutig verbindet.'
        )

        for reference_artifacts, expected_error in (
            ([], 'preserved_visual_reference_missing'),
            (
                [
                    {'type': 'image', 'artifact_ref': 'artifact:image-one', 'path': '/tmp/one.png'},
                    {'type': 'image', 'artifact_ref': 'artifact:image-two', 'path': '/tmp/two.png'},
                ],
                'preserved_visual_reference_ambiguous',
            ),
        ):
            with self.subTest(expected_error=expected_error):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'reference_artifacts': reference_artifacts,
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                branches = graph.get('downstream_branches') or []
                final_branch = branches[-1]
                tts = [item for item in branches if item.get('capability') == 'text_to_speech']
                stt = [item for item in branches if item.get('capability') == 'speech_to_text']
                self.assertEqual(
                    final_branch.get('depends_on'),
                    [
                        tts[0]['phase_id'], stt[0]['phase_id'],
                        tts[1]['phase_id'], stt[1]['phase_id'],
                    ],
                )
                self.assertEqual(final_branch.get('branch_contract_error'), expected_error)
                self.assertTrue(final_branch.get('blocked_by_branch_contract'))
                self.assertEqual(final_branch.get('repair_action'), 'repair_branch_contract')
                self.assertNotIn(
                    'image_generation',
                    [item.get('capability') for item in branches],
                )

    def test_preserved_visual_structured_join_rejects_shared_stt_consumer(self):
        prompt = (
            'Bewahre Bild und Bildanalyse unverändert; erzeuge das Bild nicht neu und analysiere es '
            'nicht erneut. Erzeuge zwei getrennte Audiofassungen, transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein JSON-Objekt aus, das die unveränderte Bildevidenz '
            'sowie beide Audio-artifact_refs und beide Transkripte eindeutig verbindet.'
        )
        references = [
            {
                'type': 'message',
                'message_id': 'msg-preserved',
                'source_response_id': 'resp-preserved',
                'content': '**Sichtbare Details:**\nEine Kuppel steht unter Sternen.',
            },
            {
                'type': 'image',
                'artifact_ref': 'artifact:image-preserved',
                'path': '/tmp/preserved.png',
                'source_message_id': 'msg-preserved',
                'source_response_id': 'resp-preserved',
            },
        ]
        explicit_branches = [
            {
                'branch_id': 'branch-text_to_speech-1',
                'phase_id': 'phase-2',
                'capability': 'text_to_speech',
                'queue_index': 1,
                'status': 'pending',
                'depends_on': ['phase-1'],
            },
            {
                'branch_id': 'branch-text_to_speech-2',
                'phase_id': 'phase-3',
                'capability': 'text_to_speech',
                'queue_index': 2,
                'status': 'pending',
                'depends_on': ['phase-1'],
            },
            {
                'branch_id': 'branch-speech_to_text-shared',
                'phase_id': 'phase-4',
                'capability': 'speech_to_text',
                'status': 'pending',
                'depends_on': ['phase-2', 'phase-3'],
            },
            {
                'branch_id': 'branch-chat-1',
                'phase_id': 'phase-5',
                'capability': 'chat',
                'status': 'pending',
                'role': 'post_artifact_text_follow_up',
                'stage_direction': 'write_text_after_artifact_generation',
                'depends_on': ['phase-4'],
            },
        ]

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'prompt': prompt,
                'reference_artifacts': references,
                'downstream_branches': explicit_branches,
            },
            route_payload={'capability': 'chat'},
        )
        final_branch = next(
            branch
            for branch in graph.get('downstream_branches') or []
            if branch.get('branch_id') == 'branch-chat-1'
        )

        self.assertEqual(
            final_branch.get('branch_contract_error'),
            'structured_audio_pair_lineage_ambiguous',
        )
        self.assertTrue(final_branch.get('blocked_by_branch_contract'))
        self.assertEqual(final_branch.get('repair_action'), 'repair_branch_contract')
        self.assertNotIn('structured_output_contract', final_branch)

    def test_selected_reference_dependency_payload_rejects_conflicting_duplicate_records(self):
        message_branch = {
            'input_refs': [
                {
                    'kind': 'selected_reference',
                    'role': 'preserved_visual_evidence',
                    'source_response_id': 'resp-conflict',
                }
            ]
        }
        message_payload = LateFillRuntimeOwner._selected_reference_dependency_payload(
            message_branch,
            current_payload={
                'reference_artifacts': [
                    {
                        'type': 'message',
                        'source_response_id': 'resp-conflict',
                        'content': '**Sichtbare Details:** erster Befund',
                    },
                    {
                        'type': 'message',
                        'source_response_id': 'resp-conflict',
                        'content': '**Sichtbare Details:** widersprechender Befund',
                    },
                ]
            },
        )
        self.assertEqual(
            message_payload.get('branch_contract_error'),
            'selected_reference_ambiguous',
        )
        self.assertTrue(message_payload.get('materialization_blocked'))

        artifact_branch = {
            'input_refs': [
                {
                    'kind': 'selected_reference',
                    'role': 'preserved_visual_artifact',
                    'artifact_ref': 'artifact:image-conflict',
                }
            ]
        }
        artifact_payload = LateFillRuntimeOwner._selected_reference_dependency_payload(
            artifact_branch,
            current_payload={
                'reference_artifacts': [
                    {
                        'type': 'image',
                        'artifact_ref': 'artifact:image-conflict',
                        'path': '/tmp/first.png',
                    },
                    {
                        'type': 'image',
                        'artifact_ref': 'artifact:image-conflict',
                        'path': '/tmp/second.png',
                    },
                ]
            },
        )
        self.assertEqual(
            artifact_payload.get('branch_contract_error'),
            'selected_reference_ambiguous',
        )
        self.assertTrue(artifact_payload.get('materialization_blocked'))

    def test_preserved_visual_structured_join_rejects_undeclared_extra_audio_pair(self):
        prompt = (
            'Bewahre Bild und Bildanalyse unverändert; erzeuge das Bild nicht neu und analysiere es '
            'nicht erneut. Erzeuge zwei getrennte Audiofassungen, transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein JSON-Objekt aus, das die unveränderte Bildevidenz '
            'sowie beide Audio-artifact_refs und beide Transkripte eindeutig verbindet.'
        )
        references = [
            {
                'type': 'message',
                'message_id': 'msg-preserved',
                'source_response_id': 'resp-preserved',
                'content': '**Sichtbare Details:**\nEine Kuppel steht unter Sternen.',
            },
            {
                'type': 'image',
                'artifact_ref': 'artifact:image-preserved',
                'path': '/tmp/preserved.png',
                'source_message_id': 'msg-preserved',
                'source_response_id': 'resp-preserved',
            },
        ]
        explicit_branches = []
        for index in range(1, 4):
            explicit_branches.extend(
                [
                    {
                        'branch_id': f'branch-text_to_speech-{index}',
                        'phase_id': f'phase-tts-{index}',
                        'capability': 'text_to_speech',
                        'queue_index': index,
                        'status': 'pending',
                        'depends_on': ['phase-1'],
                    },
                    {
                        'branch_id': f'branch-speech_to_text-{index}',
                        'phase_id': f'phase-stt-{index}',
                        'capability': 'speech_to_text',
                        'status': 'pending',
                        'depends_on': [f'phase-tts-{index}'],
                    },
                ]
            )
        explicit_branches.append(
            {
                'branch_id': 'branch-chat-1',
                'phase_id': 'phase-chat-final',
                'capability': 'chat',
                'status': 'pending',
                'role': 'post_artifact_text_follow_up',
                'stage_direction': 'write_text_after_artifact_generation',
                'depends_on': ['phase-stt-1', 'phase-stt-2', 'phase-stt-3'],
            }
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'prompt': prompt,
                'reference_artifacts': references,
                'downstream_branches': explicit_branches,
            },
            route_payload={'capability': 'chat'},
        )
        final_branch = next(
            branch
            for branch in graph.get('downstream_branches') or []
            if branch.get('branch_id') == 'branch-chat-1'
        )
        self.assertEqual(
            final_branch.get('branch_contract_error'),
            'structured_audio_pair_lineage_ambiguous',
        )
        self.assertTrue(final_branch.get('blocked_by_branch_contract'))

    def test_reduced_explicit_audio_branches_retain_counted_variant_contract(self):
        prompt = (
            'Bewahre das vorherige Bild unverändert. Erzeuge zwei getrennte Audiofassungen: einmal '
            'die ursprüngliche deutsche Erzählung und einmal eine getreue englische Übersetzung.'
        )
        explicit_branches = [
            {
                'branch_id': f'branch-text_to_speech-{index}',
                'phase_id': f'phase-{index + 1}',
                'capability': 'text_to_speech',
                'output_type': 'audio',
                'depends_on': ['phase-1'],
                'queue_index': index,
                'status': 'pending',
                'source': 'assistant_output_claim_refinement',
                'content_payload_source': 'selected_candidate_from_phase_output',
                'stage_direction': f'materialize_requested_audio_variant_{index}',
            }
            for index in (1, 2)
        ]

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'downstream_branches': explicit_branches,
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
            response_payload={
                'output_text': 'Audio wird in den nachgelagerten Zweigen erzeugt.',
                'status': 'completed',
            },
        )
        tts = graph.get('downstream_branches') or []
        self.assertEqual([item.get('candidate_selection_index') for item in tts], [1, 2])
        self.assertEqual([item.get('candidate_selection_count') for item in tts], [2, 2])
        self.assertEqual([item.get('lang_code') for item in tts], ['de', 'en'])
        self.assertEqual(
            [item.get('audio_variant_role') for item in tts],
            ['original_narration', 'faithful_translation'],
        )

    def test_counted_audio_variant_payload_fails_closed_when_prepare_output_is_ambiguous(self):
        owner = object.__new__(LateFillRuntimeOwner)
        owner.capability_text_to_speech = 'text_to_speech'
        owner.capability_image_generation = 'image_generation'
        branch = {
            'capability': 'text_to_speech',
            'selection_policy': 'selected_candidate_only',
            'candidate_selection_index': 1,
            'candidate_selection_count': 2,
            'audio_variant_index': 1,
            'stage_direction': 'materialize_requested_audio_variant_1',
        }

        valid = owner.focus_late_fill_branch_gap_payload(
            branch,
            {'content_payload': '1. Die stille Kuppel leuchtet.\n2. The silent dome glows.'},
            capability='text_to_speech',
        )
        self.assertEqual(valid.get('content_payload'), 'Die stille Kuppel leuchtet.')
        self.assertEqual(valid.get('selection_policy_applied'), 'selected_candidate_only')

        ambiguous = owner.focus_late_fill_branch_gap_payload(
            branch,
            {
                'content_payload': (
                    '1. Sichtbare Details.\n'
                    '2. Die stille Kuppel leuchtet.\n'
                    '3. The silent dome glows.'
                )
            },
            capability='text_to_speech',
        )
        self.assertEqual(ambiguous.get('branch_contract_error'), 'selected_candidate_unavailable')
        self.assertTrue(ambiguous.get('materialization_blocked'))

    def test_german_preserved_visual_follow_up_ignores_replaced_predecessor_audio_as_direct_input(self):
        prompt = (
            'Beziehe dich ausdrücklich auf das Observatorium-Bild, seine Bildanalyse, die deutsche '
            'Erzählung und das Audio aus dem unmittelbar vorherigen Turn. Bewahre Bild und Bildanalyse '
            'unverändert; erzeuge das Bild nicht neu und analysiere es nicht erneut. Ersetze den bisherigen '
            'einzelnen Audiozweig durch zwei getrennte Audiofassungen: einmal die ursprüngliche deutsche '
            'Erzählung und einmal eine getreue englische Übersetzung. Transkribiere beide tatsächlich '
            'erzeugten Audios separat und gib ein neues JSON-Objekt aus, das die unveränderte Bildevidenz '
            'sowie beide Audio-artifact_refs und beide realen Transkripte eindeutig verbindet.'
        )
        predecessor_response_id = 'resp_r5_root'
        predecessor_message_id = 'msg_r5_root'
        request_payload = {
            'ghost_route': True,
            'prompt': prompt,
            'ghost_messages': [
                {
                    'role': 'assistant',
                    'message_id': predecessor_message_id,
                    'response_id': predecessor_response_id,
                    'content': (
                        '```json\n'
                        '{"image_artifact_ref":"branch-image_generation-1",'
                        '"audio_artifact_ref":"branch-text_to_speech-1",'
                        '"real_transcript":"Alte deutsche Erzählung."}\n'
                        '```'
                    ),
                }
            ],
            'reference_artifacts': [
                {
                    'type': 'message',
                    'message_role': 'assistant',
                    'message_id': predecessor_message_id,
                    'source_response_id': predecessor_response_id,
                    'content': (
                        '```json\n'
                        '{"image_artifact_ref":"branch-image_generation-1",'
                        '"audio_artifact_ref":"branch-text_to_speech-1",'
                        '"real_transcript":"Alte deutsche Erzählung."}\n'
                        '```'
                    ),
                },
                {
                    'type': 'image',
                    'kind': 'image',
                    'artifact_ref': 'artifact:image_r5_root',
                    'path': '/artifacts/images/r5-root.png',
                    'source_response_id': predecessor_response_id,
                    'source_message_id': predecessor_message_id,
                },
                {
                    'type': 'audio',
                    'kind': 'audio',
                    'artifact_ref': 'artifact:audio_r5_root',
                    'path': '/artifacts/audio/r5-root.mp3',
                    'source_response_id': predecessor_response_id,
                    'source_message_id': predecessor_message_id,
                },
                {
                    'type': 'text',
                    'kind': 'text',
                    'extension': 'md',
                    'artifact_ref': 'artifact:text_r5_root',
                    'path': '/artifacts/transcripts/r5-root.md',
                    'source_response_id': predecessor_response_id,
                    'source_message_id': predecessor_message_id,
                },
            ],
        }

        graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
            route_payload={
                'capability': 'text_to_speech',
                'route_source': 'ghost_carried',
                'route_reason': 'text-to-speech cue',
                'route_reuse_last_artifact': True,
                'route_artifact_ref': 'artifact:text_r5_root',
                'route_artifact_path': '/artifacts/transcripts/r5-root.md',
            },
        )
        draft_graph = build_request_phase_graph(
            prompt,
            request_payload=request_payload,
        )

        branches = graph.get('downstream_branches') or []
        self.assertEqual(draft_graph['current_phase_capability'], 'chat')
        self.assertEqual(draft_graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(
            [branch.get('capability') for branch in draft_graph.get('downstream_branches') or []],
            [
                'text_to_speech',
                'text_to_speech',
                'speech_to_text',
                'speech_to_text',
                'chat',
            ],
        )
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertEqual(graph['current_phase_resolution'], 'graph_resolved')
        self.assertEqual(graph['phases'][0]['kind'], 'prepare')
        self.assertFalse(graph['prompt_intent']['input_audio_artifact_promoted_to_stt'])
        self.assertFalse(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(
            [branch.get('capability') for branch in branches],
            [
                'text_to_speech',
                'text_to_speech',
                'speech_to_text',
                'speech_to_text',
                'chat',
            ],
        )
        tts_branches = [branch for branch in branches if branch.get('capability') == 'text_to_speech']
        stt_branches = [branch for branch in branches if branch.get('capability') == 'speech_to_text']
        self.assertEqual(
            [branch.get('depends_on') for branch in stt_branches],
            [[tts_branches[0]['phase_id']], [tts_branches[1]['phase_id']]],
        )
        self.assertEqual(
            branches[-1].get('depends_on'),
            [
                tts_branches[0]['phase_id'],
                stt_branches[0]['phase_id'],
                tts_branches[1]['phase_id'],
                stt_branches[1]['phase_id'],
            ],
        )
        selected_refs = [
            item for item in (branches[-1].get('input_refs') or [])
            if item.get('kind') == 'selected_reference'
        ]
        self.assertEqual(
            [item.get('artifact_ref') for item in selected_refs if item.get('artifact_ref')],
            ['artifact:image_r5_root'],
        )
        self.assertNotIn('artifact:audio_r5_root', str(selected_refs))
        self.assertNotIn('text_artifact_extension', branches[-1])

    def test_current_turn_input_audio_remains_direct_stt_evidence(self):
        prompt = 'Transkribiere die hochgeladene Audiodatei und fasse das reale Transkript kurz zusammen.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'input_artifacts': [
                    {
                        'type': 'audio',
                        'kind': 'audio',
                        'artifact_ref': 'artifact:uploaded-audio',
                        'path': '/uploads/current-turn.wav',
                    }
                ],
            },
            route_payload={
                'capability': 'speech_to_text',
                'route_source': 'ghost_carried',
                'route_reuse_last_artifact': False,
            },
        )

        self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
        self.assertEqual(graph['current_phase_resolution'], 'router_required')
        self.assertTrue(graph['prompt_intent']['input_audio_artifact_promoted_to_stt'])

    def test_response_only_json_does_not_edit_unrelated_carried_json_source(self):
        prompts = (
            (
                'Ersetze den bisherigen einzelnen Audiozweig durch zwei neue Audiofassungen. '
                'Transkribiere beide tatsächlich erzeugten Audios separat und gib ein neues '
                'JSON-Objekt aus, das beide Audio-artifact_refs und Transkripte verbindet.'
            ),
            'Ersetze den bisherigen Audiozweig durch eine neue Audiofassung und gib ein neues JSON-Objekt aus.',
            'Replace the old audio branch with a new audio version and return a new JSON object.',
            'Gib ein neues JSON-Objekt aus und ersetze den alten Audiozweig durch eine neue Audiofassung.',
            'Return a new JSON object and replace the old audio branch with a new audio version.',
            'Return a new JSON object, replace the old audio branch with a new version.',
            'Return a new JSON object: replace the old audio branch.',
            'Gib ein neues JSON-Objekt aus: ersetze den alten Audiozweig.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'reference_artifacts': [
                            {
                                'type': 'text',
                                'kind': 'text',
                                'artifact_ref': 'artifact:prior-json',
                                'path': '/artifacts/transcripts/prior-result.json',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                self.assertFalse(graph['prompt_intent']['requests_text_artifact_output'])
                self.assertFalse(any(
                    branch.get('text_artifact_extension') == 'json'
                    for branch in graph.get('downstream_branches') or []
                ))

    def test_carried_audio_does_not_steal_pronoun_bound_generated_audio_transcription(self):
        cases = (
            (
                'Erzeuge aus diesem Text ein neues Audio und transkribiere es danach.',
                1,
            ),
            (
                'Generate two new audio versions of this sentence. Then transcribe them separately.',
                2,
            ),
            (
                'Erzeuge aus diesem Text ein neues Audio. Transkribiere danach dieses Audio.',
                1,
            ),
            (
                'Generate a new audio from this text. Then transcribe this audio.',
                1,
            ),
        )

        for prompt, expected_count in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'reference_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:prior-audio',
                                'path': '/artifacts/audio/prior-audio.mp3',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                branches = graph.get('downstream_branches') or []
                tts_branches = [
                    branch for branch in branches
                    if branch.get('capability') == 'text_to_speech'
                ]
                stt_branches = [
                    branch for branch in branches
                    if branch.get('capability') == 'speech_to_text'
                ]
                self.assertEqual(graph['current_phase_capability'], 'chat')
                self.assertEqual(len(tts_branches), expected_count)
                self.assertEqual(len(stt_branches), expected_count)
                self.assertEqual(
                    [branch.get('depends_on') for branch in stt_branches],
                    [[branch.get('phase_id')] for branch in tts_branches],
                )

    def test_explicit_uploaded_audio_target_can_remain_direct_before_new_tts(self):
        cases = (
            'Transcribe the uploaded audio, then generate one new audio from the summary.',
            'Transkribiere die hochgeladene Audiodatei und erzeuge danach ein neues Audio aus der Zusammenfassung.',
        )

        for prompt in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'input_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:uploaded-audio',
                                'path': '/uploads/current-turn.wav',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
                self.assertTrue(graph['prompt_intent']['input_audio_artifact_promoted_to_stt'])
                self.assertEqual(
                    [
                        branch.get('capability')
                        for branch in graph.get('downstream_branches') or []
                    ],
                    ['text_to_speech'],
                )

    def test_uploaded_audio_then_generated_audio_retains_generated_stt_suffix(self):
        cases = (
            'Transcribe the uploaded audio. Then generate a new audio from the summary and transcribe it.',
            'Transkribiere die hochgeladene Audiodatei. Erzeuge danach ein neues Audio aus der Zusammenfassung und transkribiere es.',
            'First transcribe the uploaded audio, then generate a new audio from the summary and transcribe the new audio too.',
        )

        for prompt in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'input_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:uploaded-audio',
                                'path': '/uploads/current-turn.wav',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                branches = graph.get('downstream_branches') or []
                self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
                self.assertTrue(graph['prompt_intent']['input_audio_artifact_promoted_to_stt'])
                self.assertEqual(
                    [branch.get('capability') for branch in branches],
                    ['text_to_speech', 'speech_to_text'],
                )
                self.assertEqual(branches[0].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[1].get('depends_on'),
                    [branches[0].get('phase_id')],
                )

    def test_post_tts_explicit_uploaded_audio_target_rebinds_to_request_input(self):
        cases = (
            (
                'Transcribe the uploaded audio. Then generate a new audio. '
                'Finally transcribe the uploaded audio again.'
            ),
            (
                'Transcribe the uploaded audio. Then generate a new audio. '
                'Finally transcribe only the uploaded audio again.'
            ),
            (
                'Transcribe the uploaded audio, generate a new audio, '
                'then transcribe only the uploaded audio again.'
            ),
            (
                'Transcribe the uploaded audio; generate a new audio; '
                'then transcribe the uploaded audio again.'
            ),
            (
                'Transkribiere die hochgeladene Audiodatei. Erzeuge danach ein neues Audio. '
                'Transkribiere abschließend die hochgeladene Audiodatei erneut.'
            ),
            (
                'Transkribiere die hochgeladene Audiodatei, erzeuge ein neues Audio, '
                'und transkribiere danach nur die hochgeladene Audiodatei erneut.'
            ),
            (
                'Transkribiere die hochgeladene Audiodatei; erzeuge ein neues Audio; '
                'transkribiere danach die hochgeladene Audiodatei erneut.'
            ),
        )

        for prompt in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'input_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:uploaded-audio',
                                'path': '/uploads/current-turn.wav',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                branches = graph.get('downstream_branches') or []
                self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
                self.assertEqual(
                    [branch.get('capability') for branch in branches],
                    ['text_to_speech', 'speech_to_text'],
                )
                self.assertEqual(branches[0].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[0].get('content_payload_source'),
                    'speech_to_text_branch_result',
                )
                self.assertEqual(branches[1].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[1].get('content_payload_source'),
                    'current_input_audio_artifact',
                )

    def test_generated_and_direct_input_stt_requests_preserve_both_source_classes(self):
        cases = (
            (
                'Generate a new audio and transcribe it. '
                'Then transcribe the uploaded audio.'
            ),
            (
                'Erzeuge ein neues Audio und transkribiere es. '
                'Transkribiere danach die hochgeladene Audiodatei.'
            ),
        )

        for prompt in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'input_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:uploaded-audio',
                                'path': '/uploads/current-turn.wav',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                branches = graph.get('downstream_branches') or []
                self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
                self.assertEqual(
                    [branch.get('capability') for branch in branches],
                    ['text_to_speech', 'speech_to_text'],
                )
                self.assertEqual(branches[0].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[0].get('content_payload_source'),
                    'speech_to_text_branch_result',
                )
                self.assertEqual(
                    branches[1].get('depends_on'),
                    [branches[0].get('phase_id')],
                )
                self.assertNotEqual(
                    branches[1].get('content_payload_source'),
                    'current_input_audio_artifact',
                )

    def test_pre_and_post_tts_stt_actions_keep_all_distinct_source_obligations(self):
        cases = (
            (
                'Transcribe the uploaded audio. Generate a new audio and transcribe it. '
                'Finally transcribe the uploaded audio again.'
            ),
            (
                'Transkribiere die hochgeladene Audiodatei. Erzeuge ein neues Audio und '
                'transkribiere es. Transkribiere abschließend die hochgeladene Audiodatei erneut.'
            ),
        )

        for prompt in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'input_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:uploaded-audio',
                                'path': '/uploads/current-turn.wav',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                branches = graph.get('downstream_branches') or []
                self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
                self.assertEqual(
                    [branch.get('capability') for branch in branches],
                    ['text_to_speech', 'speech_to_text', 'speech_to_text'],
                )
                self.assertEqual(branches[0].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[1].get('depends_on'),
                    [branches[0].get('phase_id')],
                )
                self.assertEqual(branches[2].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[2].get('content_payload_source'),
                    'current_input_audio_artifact',
                )

    def test_audio_target_qualifier_selects_source_when_input_and_reference_both_exist(self):
        cases = (
            (
                'Transcribe the selected audio. Then generate a new audio. '
                'Finally transcribe the selected audio again.',
                'selected_reference_audio_artifact',
            ),
            (
                'Transcribe the uploaded audio. Then generate a new audio. '
                'Finally transcribe the uploaded audio again.',
                'current_input_audio_artifact',
            ),
            (
                'Transkribiere die ausgewählte Audiodatei. Erzeuge danach ein neues Audio. '
                'Transkribiere abschließend die ausgewählte Audiodatei erneut.',
                'selected_reference_audio_artifact',
            ),
            (
                'Transkribiere die hochgeladene Audiodatei. Erzeuge danach ein neues Audio. '
                'Transkribiere abschließend die hochgeladene Audiodatei erneut.',
                'current_input_audio_artifact',
            ),
        )

        for prompt, expected_source in cases:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={
                        'ghost_route': True,
                        'prompt': prompt,
                        'input_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:uploaded-audio',
                                'path': '/uploads/current.wav',
                            }
                        ],
                        'selected_reference_artifacts': [
                            {
                                'type': 'audio',
                                'kind': 'audio',
                                'artifact_ref': 'artifact:selected-audio',
                                'path': '/artifacts/audio/selected.wav',
                            }
                        ],
                    },
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                branches = graph.get('downstream_branches') or []
                self.assertEqual(graph['current_phase_capability'], 'speech_to_text')
                self.assertEqual(
                    [branch.get('capability') for branch in branches],
                    ['text_to_speech', 'speech_to_text'],
                )
                self.assertEqual(branches[1].get('depends_on'), ['phase-1'])
                self.assertEqual(
                    branches[1].get('content_payload_source'),
                    expected_source,
                )

    def test_explicit_carried_json_source_edit_remains_materialized(self):
        source_path = '/artifacts/documents/current-settings.json'
        prompt = 'Ändere dieses JSON-Objekt: setze den Status auf fertig.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'selected_reference_artifacts': [
                    {
                        'type': 'text',
                        'kind': 'text',
                        'artifact_ref': 'artifact:current-settings',
                        'path': source_path,
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        json_branches = [
            branch
            for branch in graph.get('downstream_branches') or []
            if branch.get('text_artifact_extension') == 'json'
        ]
        self.assertTrue(graph['prompt_intent']['requests_text_artifact_output'])
        self.assertEqual(len(json_branches), 1)
        self.assertEqual(json_branches[0].get('text_artifact_source'), 'selected_source_edit')
        self.assertEqual(json_branches[0].get('text_artifact_target_path'), source_path)

    def test_visual_preservation_guard_does_not_suppress_requested_image_edit(self):
        prompt = (
            'Bewahre den bisherigen Seitenaufbau, aber ändere das Bild deutlich und erzeuge eine neue '
            'Bildversion. Analysiere danach das neu erzeugte Bild.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['visual_artifact_preservation_without_regeneration'])
        self.assertFalse(graph['prompt_intent']['visual_analysis_preservation_without_reanalysis'])
        self.assertIn('image_generation', graph['downstream_capabilities'])
        self.assertIn('vision_analysis', graph['downstream_capabilities'])

    def test_counted_audio_sources_do_not_create_tts_outputs(self):
        prompt = 'Transkribiere zwei vorhandene Audiodateien separat; erzeuge kein neues Audio.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertFalse(graph['prompt_intent']['requests_audio_output'])
        self.assertFalse(graph['prompt_intent']['counted_audio_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 0)
        self.assertNotIn(
            'text_to_speech',
            [branch.get('capability') for branch in (graph.get('downstream_branches') or [])],
        )

    def test_audio_output_count_fails_closed_above_bound(self):
        prompt = 'Erzeuge 99 getrennte Audiofassungen aus diesem Satz.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        tts_branches = [
            branch for branch in (graph.get('downstream_branches') or [])
            if branch.get('capability') == 'text_to_speech'
        ]
        executable_tts_branches = [
            branch for branch in tts_branches
            if branch.get('contract_state') != 'reserved'
        ]
        self.assertTrue(graph['prompt_intent']['requests_audio_output'])
        self.assertFalse(graph['prompt_intent']['counted_audio_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 0)
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count_raw'], 99)
        self.assertTrue(graph['prompt_intent']['audio_output_count_exceeds_bound'])
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count_max'], 6)
        self.assertEqual(executable_tts_branches, [])
        self.assertEqual(len(tts_branches), 1)
        self.assertEqual(tts_branches[0]['contract_state'], 'reserved')
        self.assertTrue(tts_branches[0]['blocked_by_branch_contract'])
        blocked_phase = next(
            phase for phase in graph['phases']
            if phase.get('branch_id') == tts_branches[0]['branch_id']
        )
        self.assertEqual(blocked_phase['resolution'], 'blocked_invalid_cardinality')
        self.assertEqual(graph['prompt_intent']['required_intent_output_counts'], {})
        self.assertTrue(graph['blocked_by_intent_contract'])
        self.assertTrue(graph['needs_clarification'])
        blocking_obligations = [
            obligation for obligation in graph['intent_obligations']
            if obligation.get('kind') == 'intent_cardinality_guard'
        ]
        self.assertEqual(len(blocking_obligations), 1)
        self.assertTrue(blocking_obligations[0]['required'])
        self.assertEqual(blocking_obligations[0]['status'], 'blocked')
        self.assertEqual(blocking_obligations[0]['resolution'], 'needs_clarification')
        self.assertEqual(blocking_obligations[0]['requested_count'], 99)
        self.assertEqual(
            graph['prompt_intent']['required_intent_capability_counts'],
            {},
        )

    def test_answer_as_audio_delivery_reuses_count_contract_with_existing_bound(self):
        prompt = 'Gib mir die Antwort als zwei generierte Audios.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        tts_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'text_to_speech'
            and branch.get('contract_state') != 'reserved'
        ]
        self.assertEqual(graph['current_phase_capability'], 'chat')
        self.assertTrue(graph['prompt_intent']['counted_audio_output_obligation'])
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 2)
        self.assertEqual(graph['prompt_intent']['required_intent_output_counts'], {'audio': 2})
        self.assertEqual(len(tts_branches), 2)
        self.assertEqual(
            [branch.get('source') for branch in tts_branches],
            ['request_phase_graph', 'request_phase_graph'],
        )

    def test_answer_as_audio_delivery_count_above_bound_stays_non_executable(self):
        prompt = 'Gib mir die Antwort als sieben generierte Audios.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 0)
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count_raw'], 7)
        self.assertTrue(graph['prompt_intent']['audio_output_count_exceeds_bound'])
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count_max'], 6)
        self.assertTrue(graph['blocked_by_intent_contract'])
        self.assertTrue(graph['needs_clarification'])
        self.assertFalse(any(
            branch.get('capability') == 'text_to_speech'
            and branch.get('contract_state') != 'reserved'
            for branch in graph['downstream_branches']
        ))

    def test_explicit_downstream_audio_contract_is_not_implicitly_expanded(self):
        prompt = 'Erzeuge zwei getrennte Audiofassungen und transkribiere beide separat.'
        explicit_branches = [
            {
                'branch_id': 'branch-explicit-audio',
                'phase_id': 'phase-explicit-audio',
                'capability': 'text_to_speech',
                'output_type': 'audio',
                'depends_on': ['phase-1'],
                'required': True,
            },
            {
                'branch_id': 'branch-explicit-transcript',
                'phase_id': 'phase-explicit-transcript',
                'capability': 'speech_to_text',
                'output_type': 'text',
                'depends_on': ['phase-explicit-audio'],
                'required': True,
            },
        ]

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'downstream_branches': explicit_branches,
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 2)
        self.assertEqual(
            [branch.get('branch_id') for branch in graph['downstream_branches']],
            ['branch-explicit-audio', 'branch-explicit-transcript'],
        )

    def test_counted_audio_variants_preserve_independent_image_batching(self):
        prompt = (
            'Erzeuge drei unterschiedliche Bilder und zwei getrennte Audiofassungen. Analysiere jedes '
            'erzeugte Bild, transkribiere beide tatsächlich erzeugten Audios separat und gib danach ein '
            'JSON-Objekt mit allen Artefakt-Referenzen und Befunden aus.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        branches = graph['downstream_branches']
        by_capability = {
            capability: [branch for branch in branches if branch.get('capability') == capability]
            for capability in {
                'image_generation',
                'text_to_speech',
                'vision_analysis',
                'speech_to_text',
                'chat',
            }
        }
        self.assertEqual(len(by_capability['image_generation']), 3)
        self.assertEqual(len(by_capability['vision_analysis']), 3)
        self.assertEqual(len(by_capability['text_to_speech']), 2)
        self.assertEqual(len(by_capability['speech_to_text']), 2)
        self.assertEqual(len(by_capability['chat']), 1)
        self.assertEqual(
            [branch['depends_on'] for branch in by_capability['speech_to_text']],
            [
                [by_capability['text_to_speech'][0]['phase_id']],
                [by_capability['text_to_speech'][1]['phase_id']],
            ],
        )
        self.assertEqual(
            graph['prompt_intent']['required_intent_output_counts'],
            {'image': 3, 'audio': 2},
        )

    def test_preserved_existing_image_allows_separate_new_visual_work(self):
        prompt = (
            'Bewahre das bisherige Bild unverändert; erzeuge es nicht neu. Erzeuge zusätzlich zwei neue '
            'Bilder eines Teleskops und analysiere beide.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['visual_artifact_preservation_without_regeneration'])
        self.assertTrue(intent['separate_visual_generation_request'])
        self.assertTrue(intent['separate_visual_analysis_request'])
        self.assertFalse(intent['visual_artifact_execution_suppressed_by_preservation'])
        self.assertFalse(intent['visual_analysis_execution_suppressed_by_preservation'])
        self.assertEqual(intent['requested_visual_output_count'], 2)
        images = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'image_generation'
        ]
        analyses = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'vision_analysis'
        ]
        self.assertEqual(len(images), 2)
        self.assertEqual(len(analyses), 2)
        self.assertEqual(
            [branch['depends_on'] for branch in analyses],
            [[images[0]['phase_id']], [images[1]['phase_id']]],
        )

    def test_english_preserved_image_allows_separate_new_illustration_work(self):
        prompts = (
            (
                'Keep the prior image unchanged and do not regenerate it. Create one new illustration '
                'of a telescope and analyze that new illustration.',
                1,
            ),
            (
                'Preserve the prior image unchanged; do not generate it again. Generate two additional '
                'new images of a telescope and analyze both.',
                2,
            ),
        )

        for prompt, expected_count in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                intent = graph['prompt_intent']
                self.assertTrue(intent['visual_artifact_preservation_without_regeneration'])
                self.assertTrue(intent['separate_visual_generation_request'])
                self.assertTrue(intent['separate_visual_analysis_request'])
                self.assertFalse(intent['visual_artifact_execution_suppressed_by_preservation'])
                self.assertFalse(intent['visual_analysis_execution_suppressed_by_preservation'])
                self.assertFalse(intent['explicit_defer_materialization'])
                self.assertFalse(intent['explicit_visual_defer_materialization'])
                self.assertFalse(intent['explicit_audio_defer_materialization'])
                self.assertTrue(intent['requests_visual_output'])
                self.assertEqual(intent['requested_visual_output_count'], expected_count)

                images = [
                    branch for branch in graph['downstream_branches']
                    if branch.get('capability') == 'image_generation'
                ]
                analyses = [
                    branch for branch in graph['downstream_branches']
                    if branch.get('capability') == 'vision_analysis'
                ]
                self.assertEqual(len(images), expected_count)
                self.assertEqual(len(analyses), expected_count)
                self.assertEqual(
                    [branch['depends_on'] for branch in analyses],
                    [[image['phase_id']] for image in images],
                )

    def test_english_plain_visual_preservation_remains_non_executable(self):
        prompt = (
            'Keep the prior image unchanged and do not regenerate it. '
            'Do not analyze it again.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['visual_artifact_preservation_without_regeneration'])
        self.assertTrue(intent['visual_artifact_execution_suppressed_by_preservation'])
        self.assertFalse(intent['separate_visual_generation_request'])
        self.assertFalse(intent['separate_visual_analysis_request'])
        self.assertNotIn(
            'image_generation',
            [branch.get('capability') for branch in graph['downstream_branches']],
        )
        self.assertNotIn(
            'vision_analysis',
            [branch.get('capability') for branch in graph['downstream_branches']],
        )

    def test_english_retained_picture_allows_separate_new_photo_work(self):
        prompt = (
            'Retain the old picture intact; never recreate it. '
            'Make another photo and inspect that new photo.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['visual_artifact_preservation_without_regeneration'])
        self.assertTrue(intent['separate_visual_generation_request'])
        self.assertTrue(intent['separate_visual_analysis_request'])
        self.assertFalse(intent['visual_artifact_execution_suppressed_by_preservation'])
        self.assertFalse(intent['visual_analysis_execution_suppressed_by_preservation'])
        images = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'image_generation'
        ]
        analyses = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'vision_analysis'
        ]
        self.assertEqual(len(images), 1)
        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0]['depends_on'], [images[0]['phase_id']])

    def test_english_recreate_preservation_without_new_work_remains_non_executable(self):
        prompt = 'Retain the old picture intact; never recreate it.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['visual_artifact_preservation_without_regeneration'])
        self.assertTrue(intent['visual_artifact_execution_suppressed_by_preservation'])
        self.assertFalse(intent['separate_visual_generation_request'])
        self.assertFalse(intent['separate_visual_analysis_request'])
        self.assertNotIn(
            'image_generation',
            [branch.get('capability') for branch in graph['downstream_branches']],
        )
        self.assertNotIn(
            'vision_analysis',
            [branch.get('capability') for branch in graph['downstream_branches']],
        )

    def test_photo_generation_and_analysis_form_one_to_one_phase_chain(self):
        prompts = (
            'Generate a photo and analyze that photo.',
            'Erzeuge ein Foto und analysiere das Foto.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )

                images = [
                    branch for branch in graph['downstream_branches']
                    if branch.get('capability') == 'image_generation'
                ]
                analyses = [
                    branch for branch in graph['downstream_branches']
                    if branch.get('capability') == 'vision_analysis'
                ]
                self.assertEqual(len(images), 1)
                self.assertEqual(len(analyses), 1)
                self.assertEqual(analyses[0]['depends_on'], [images[0]['phase_id']])

    def test_preserved_old_analysis_allows_new_attached_image_analysis(self):
        prompt = (
            'Bewahre die bisherige Bildanalyse unverändert und analysiere sie nicht erneut. '
            'Analysiere das neu angehängte Bild.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={
                'ghost_route': True,
                'prompt': prompt,
                'input_artifacts': [
                    {
                        'artifact_ref': 'artifact:new-attached-image',
                        'type': 'image',
                        'path': '/tmp/new-attached-image.png',
                    }
                ],
            },
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['visual_analysis_preservation_without_reanalysis'])
        self.assertTrue(intent['separate_visual_analysis_request'])
        self.assertFalse(intent['visual_analysis_execution_suppressed_by_preservation'])
        self.assertEqual(graph['current_phase_capability'], 'vision_analysis')
        self.assertNotIn(
            'image_generation',
            [branch.get('capability') for branch in graph['downstream_branches']],
        )

    def test_plain_visual_preservation_remains_non_executable(self):
        prompt = (
            'Bewahre Bild und Bildanalyse unverändert; erzeuge das Bild nicht neu und analysiere es '
            'nicht erneut.'
        )

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        intent = graph['prompt_intent']
        self.assertTrue(intent['visual_artifact_execution_suppressed_by_preservation'])
        self.assertTrue(intent['visual_analysis_execution_suppressed_by_preservation'])
        self.assertFalse(intent['separate_visual_generation_request'])
        self.assertFalse(intent['separate_visual_analysis_request'])
        self.assertEqual(graph['downstream_branches'], [])

    def test_non_audio_source_counts_do_not_expand_tts_cardinality(self):
        prompts = (
            'Erzeuge zwei Gründe, warum diese Audiofassung gut ist, und erzeuge danach ein Audio daraus.',
            'Schreibe zwei Absätze über Audiofassungen und erzeuge anschließend ein Audio.',
            'Erzeuge zwei Textvarianten über Audiofassungen; vertone nur die erste als genau ein Audio.',
            'Write two candidate texts about audio versions, then voice only the first as exactly one audio.',
            'Write two reasons why this audio version is good, then create one audio from the final text.',
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                tts_branches = [
                    branch for branch in graph['downstream_branches']
                    if branch.get('capability') == 'text_to_speech'
                    and branch.get('contract_state') != 'reserved'
                ]
                self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 1)
                self.assertEqual(graph['prompt_intent']['requested_visual_output_count'], 0)
                self.assertEqual(len(tts_branches), 1)
                self.assertNotIn(
                    'image_generation',
                    [branch.get('capability') for branch in graph['downstream_branches']],
                )

    def test_audio_count_words_above_bound_block_without_tts_execution(self):
        for prompt in (
            'Erzeuge sieben getrennte Audiofassungen.',
            'Generate seven separate audio versions.',
        ):
            with self.subTest(prompt=prompt):
                graph = build_request_phase_graph(
                    prompt,
                    request_payload={'ghost_route': True, 'prompt': prompt},
                    route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
                )
                self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 0)
                self.assertEqual(graph['prompt_intent']['requested_audio_output_count_raw'], 7)
                self.assertTrue(graph['prompt_intent']['audio_output_count_exceeds_bound'])
                self.assertTrue(graph['needs_clarification'])
                self.assertFalse(any(
                    branch.get('capability') == 'text_to_speech'
                    and branch.get('contract_state') != 'reserved'
                    for branch in graph['downstream_branches']
                ))

    def test_audio_variant_language_order_follows_prompt_position(self):
        prompt = 'Create one English and one German audio version.'

        graph = build_request_phase_graph(
            prompt,
            request_payload={'ghost_route': True, 'prompt': prompt},
            route_payload={'capability': 'chat', 'route_source': 'ghost_carried'},
        )

        tts_branches = [
            branch for branch in graph['downstream_branches']
            if branch.get('capability') == 'text_to_speech'
        ]
        self.assertEqual(graph['prompt_intent']['requested_audio_output_count'], 2)
        self.assertEqual([branch.get('lang_code') for branch in tts_branches], ['en', 'de'])
        self.assertEqual(
            [branch.get('candidate_selection_count') for branch in tts_branches],
            [2, 2],
        )


if __name__ == '__main__':
    unittest.main()
