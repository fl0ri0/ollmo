import unittest
from pathlib import Path

from ollmo_server.request_intake_runtime import RequestIntakeRuntimeOwner
from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner
from ollmo_services.artifact_contracts import sanitize_artifact_record


class RequestIntakePredecessorContextTests(unittest.TestCase):
    @staticmethod
    def _named_edit_owner(
        *,
        artifacts,
        resolve_saved_path=None,
        predecessor_conversation_id='conv-bundle-edit',
        predecessor_message_id='canonical-msg-bundle',
    ):
        predecessor_request = {}
        if predecessor_conversation_id is not None:
            predecessor_request['conversation_id'] = predecessor_conversation_id
        predecessor_payload = {
            'lifecycle_state': 'completed',
            'output_text': 'Canonical prior response with the existing bundle.',
            'artifacts': artifacts,
            'response_frame': {
                'request': predecessor_request,
            },
        }
        return RequestIntakeRuntimeOwner(
            hooks={
                'read_chat_history': lambda conversation_id, **kwargs: {
                    'messages': [
                        {
                            'role': 'assistant',
                            'message_id': 'ui-msg-bundle',
                            'response_id': 'resp-bundle',
                            'content': 'UI message copy.',
                        }
                    ]
                },
                'chat_history_dir_getter': lambda: '/unused',
                'get_response_lookup_record': lambda response_id: {
                    **{
                        'id': response_id,
                        'lifecycle_state': 'completed',
                        'response_payload': predecessor_payload,
                    },
                    **(
                        {'message_id': predecessor_message_id}
                        if predecessor_message_id is not None
                        else {}
                    ),
                },
                'resolve_saved_downloadable_artifact_path': (
                    resolve_saved_path
                    or (lambda raw_path: Path(str(raw_path)))
                ),
                'sanitize_artifact_record': sanitize_artifact_record,
                'get_cached_generated_image_state': lambda path: None,
                'sanitize_ghost_messages': lambda messages: messages,
            }
        )

    def test_selected_ui_message_promotes_canonical_repair_prompt_batch(self):
        prompts = [
            (
                f'A detailed community garden photograph number {index}, '
                'warm natural light, rich soil, healthy plants, realistic '
                'textures, and a welcoming neighborhood atmosphere.'
            )
            for index in range(1, 6)
        ]
        prepared_content = (
            '### Image Generation Prompts\n\n'
            + '\n\n'.join(
                f'**Prompt {index} (Site Image - `image-{index}.jpg`)**\n'
                f'> {prompt}'
                for index, prompt in enumerate(prompts, start=1)
            )
        )
        predecessor_payload = {
            'lifecycle_state': 'repair_needed',
            'output_text': 'Artifact generated.',
            'artifacts': [
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:site-root',
                    'path': '/artifacts/documents/community-garden.html',
                }
            ],
            'late_fill': {'content_payload': prepared_content},
            'response_frame': {
                'request': {'conversation_id': 'conv-site-repair'},
            },
        }
        semantic_owner = ResponseSemanticsRuntimeOwner(hooks={})
        owner = RequestIntakeRuntimeOwner(
            hooks={
                'read_chat_history': lambda conversation_id, **kwargs: {
                    'messages': [
                        {
                            'role': 'assistant',
                            'message_id': 'ui-msg-site-root',
                            'response_id': 'resp-site-root',
                            'content': 'Artifact generated.',
                        }
                    ]
                },
                'chat_history_dir_getter': lambda: '/unused',
                'get_response_lookup_record': lambda response_id: {
                    'id': response_id,
                    'message_id': 'canonical-msg-site-root',
                    'lifecycle_state': 'repair_needed',
                    'response_payload': predecessor_payload,
                },
                'extract_batch_image_prompts': (
                    lambda text, **kwargs: semantic_owner.extract_batch_image_prompts(
                        text,
                        **kwargs,
                    )
                ),
            }
        )

        promoted = owner.promote_current_predecessor_context(
            {
                'conversation_id': 'conv-site-repair',
                'prompt': (
                    'can you please create the images and link them to the site '
                    'properly? thank you.'
                ),
                'reference_artifacts': {
                    'type': 'message',
                    'message_id': 'ui-msg-site-root',
                    'content': 'Untrusted client copy is replaced by canonical truth.',
                },
            }
        )

        context = promoted['current_predecessor_context']
        self.assertEqual(context['batch_prompts'], prompts)
        self.assertEqual(
            context['source_message_id'],
            'canonical-msg-site-root',
        )
        self.assertEqual(
            context['selected_message_id'],
            'ui-msg-site-root',
        )
        self.assertEqual(len(promoted['reference_artifacts']), 2)
        self.assertEqual(
            promoted['reference_artifacts'][0]['content'],
            'Artifact generated.',
        )
        self.assertEqual(
            promoted['ghost_messages'][0]['message_id'],
            'canonical-msg-site-root',
        )

    def test_message_only_ui_reference_promotes_exact_named_text_edit_artifacts(self):
        owner = self._named_edit_owner(
            artifacts=[
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator',
                    'path': '/artifacts/documents/20260804_model_configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                },
                {
                    'type': 'document',
                    'artifact_ref': 'artifact:styles',
                    'path': '/artifacts/documents/20260804_model_styles.css',
                    'artifact_request': {'source_name': 'styles', 'extension': 'css'},
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:pricing',
                    'path': '/artifacts/documents/20260804_model_pricing.json',
                    'text_artifact_source_name': 'pricing',
                    'text_artifact_extension': 'json',
                },
                {
                    'type': 'image',
                    'artifact_ref': 'artifact:old-watch',
                    'path': '/artifacts/images/old-watch.png',
                },
            ]
        )
        prompt = (
            'Reference the current configurator.html and styles.css. '
            'We are upgrading the visualizer in the configurator. '
            'Generate exactly 3 new cinematic images. '
            'Update configurator.html and styles.css so the material selector '
            'switches between the new generated images. Keep the rest intact.'
        )

        promoted = owner.promote_current_predecessor_context(
            {
                'conversation_id': 'conv-bundle-edit',
                'prompt': prompt,
                'selected_reference_artifact': {
                    'type': 'message',
                    'message_id': 'ui-msg-bundle',
                },
            }
        )

        self.assertEqual(promoted['prompt'], prompt)
        context = promoted['current_predecessor_context']
        self.assertEqual(context['promotion_mode'], 'named_text_edit')
        self.assertEqual(context['current_prompt_authority'], 'payload.prompt')
        self.assertEqual(context['batch_prompts'], [])
        self.assertEqual(
            context['requested_text_artifact_names'],
            ['configurator.html', 'styles.css'],
        )
        self.assertEqual(
            [item['artifact_ref'] for item in context['matched_text_artifacts']],
            ['artifact:configurator', 'artifact:styles'],
        )
        self.assertEqual(len(promoted['reference_artifacts']), 3)
        self.assertEqual(
            [item.get('name') for item in promoted['reference_artifacts'][1:]],
            ['configurator.html', 'styles.css'],
        )
        self.assertNotIn(
            'artifact:pricing',
            [item.get('artifact_ref') for item in promoted['reference_artifacts']],
        )
        self.assertEqual(
            [
                item.get('artifact_ref')
                for item in context['carried_public_dependencies']
            ],
            ['artifact:pricing', 'artifact:old-watch'],
        )
        self.assertTrue(
            context['carried_public_dependencies'][0][
                'carried_public_dependency'
            ]
        )
        self.assertEqual(len(promoted['ghost_messages'][0]['artifacts']), 2)

        sanitized = owner._extract_selected_reference_artifacts(promoted)
        self.assertEqual(len(sanitized), 3)
        self.assertEqual(
            [item.get('name') for item in sanitized[1:]],
            ['configurator.html', 'styles.css'],
        )
        injected = owner._inject_selected_reference_message([], sanitized)
        self.assertEqual(len(injected), 3)
        self.assertEqual(
            [item['artifacts'][0]['name'] for item in injected[1:]],
            ['configurator.html', 'styles.css'],
        )

    def test_recovered_same_conversation_predecessor_uses_history_identity_fallback(self):
        owner = self._named_edit_owner(
            predecessor_conversation_id=None,
            predecessor_message_id=None,
            artifacts=[
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator',
                    'path': '/artifacts/documents/configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:styles',
                    'path': '/artifacts/documents/styles.css',
                    'text_artifact_source_name': 'styles',
                    'text_artifact_extension': 'css',
                },
            ],
        )

        promoted = owner.promote_current_predecessor_context(
            {
                'conversation_id': 'conv-bundle-edit',
                'prompt': (
                    'Use the current configurator.html and styles.css. '
                    'Update only the visualizer and keep everything else intact.'
                ),
                'selected_reference_artifact': {
                    'type': 'message',
                    'message_id': 'ui-msg-bundle',
                },
            }
        )

        context = promoted['current_predecessor_context']
        self.assertEqual(context['source_response_id'], 'resp-bundle')
        self.assertEqual(context['source_message_id'], 'ui-msg-bundle')
        self.assertEqual(
            context['predecessor_conversation_authority'],
            'current_conversation_history_mapping',
        )
        self.assertEqual(
            context['predecessor_message_authority'],
            'current_conversation_history_mapping',
        )
        self.assertEqual(
            [item['artifact_ref'] for item in context['matched_text_artifacts']],
            ['artifact:configurator', 'artifact:styles'],
        )

    def test_exact_natural_visualizer_follow_up_promotes_only_named_current_files(self):
        owner = self._named_edit_owner(
            artifacts=[
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator',
                    'path': '/artifacts/documents/configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:styles',
                    'path': '/artifacts/documents/styles.css',
                    'text_artifact_source_name': 'styles',
                    'text_artifact_extension': 'css',
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:pricing',
                    'path': '/artifacts/documents/pricing.json',
                    'text_artifact_source_name': 'pricing',
                    'text_artifact_extension': 'json',
                },
            ],
        )
        prompt = (
            'Use the current configurator.html and styles.css. '
            'Upgrade only the material visualizer.\n\n'
            'Create exactly three square cinematic macro images of the same '
            'front-facing luxury watch case:\n'
            '1. Brushed titanium, cool diffuse bench light.\n'
            '2. Forged carbon, directional light revealing the layered weave.\n'
            '3. Polished rose gold, warm controlled highlights.\n\n'
            'Wire the existing material selector to those three generated images. '
            'Keep all other design, copy, navigation, data files, and shared CSS intact.'
        )

        promoted = owner.promote_current_predecessor_context(
            {
                'conversation_id': 'conv-bundle-edit',
                'prompt': prompt,
                'selected_reference_artifact': {
                    'type': 'message',
                    'message_id': 'ui-msg-bundle',
                },
            }
        )

        context = promoted['current_predecessor_context']
        self.assertEqual(
            context['requested_text_artifact_names'],
            ['configurator.html', 'styles.css'],
        )
        self.assertEqual(
            [item['artifact_ref'] for item in context['matched_text_artifacts']],
            ['artifact:configurator', 'artifact:styles'],
        )
        self.assertNotIn(
            'artifact:pricing',
            [item.get('artifact_ref') for item in promoted['reference_artifacts']],
        )
        self.assertEqual(
            [
                item.get('artifact_ref')
                for item in context['carried_public_dependencies']
            ],
            ['artifact:pricing'],
        )

    def test_explicit_cross_conversation_predecessor_is_rejected_even_with_history_mapping(self):
        owner = self._named_edit_owner(
            predecessor_conversation_id='conv-someone-else',
            predecessor_message_id=None,
            artifacts=[
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator',
                    'path': '/artifacts/documents/configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                }
            ],
        )
        request_payload = {
            'conversation_id': 'conv-bundle-edit',
            'prompt': 'Update configurator.html and keep everything else intact.',
            'selected_reference_artifact': {
                'type': 'message',
                'message_id': 'ui-msg-bundle',
            },
        }

        promoted = owner.promote_current_predecessor_context(request_payload)

        self.assertEqual(promoted, request_payload)
        self.assertNotIn('current_predecessor_context', promoted)

    def test_named_text_edit_fails_closed_when_canonical_match_is_ambiguous(self):
        owner = self._named_edit_owner(
            artifacts=[
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator-old',
                    'path': '/artifacts/documents/old_configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator-new',
                    'path': '/artifacts/documents/new_configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:styles',
                    'path': '/artifacts/documents/styles.css',
                    'text_artifact_source_name': 'styles',
                    'text_artifact_extension': 'css',
                },
            ]
        )
        request_payload = {
            'conversation_id': 'conv-bundle-edit',
            'prompt': 'Update configurator.html and styles.css, preserving everything else.',
            'selected_reference_artifact': {
                'type': 'message',
                'message_id': 'ui-msg-bundle',
            },
        }

        promoted = owner.promote_current_predecessor_context(request_payload)

        self.assertEqual(promoted, request_payload)
        self.assertNotIn('current_predecessor_context', promoted)
        self.assertNotIn('ghost_messages', promoted)

    def test_selected_message_without_explicit_named_edit_does_not_promote_predecessor(self):
        request_payload = {
            'conversation_id': 'conv-bundle-edit',
            'prompt': 'Reference configurator.html and styles.css and describe their structure.',
            'selected_reference_artifact': {
                'type': 'message',
                'message_id': 'ui-msg-bundle',
            },
        }

        promoted = RequestIntakeRuntimeOwner(hooks={}).promote_current_predecessor_context(
            request_payload
        )

        self.assertEqual(promoted, request_payload)
        self.assertNotIn('current_predecessor_context', promoted)

    def test_named_text_edit_does_not_promote_partial_nonconcrete_source_set(self):
        owner = self._named_edit_owner(
            artifacts=[
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:configurator',
                    'path': '/artifacts/documents/configurator.html',
                    'text_artifact_source_name': 'configurator',
                    'text_artifact_extension': 'html',
                },
                {
                    'type': 'text',
                    'artifact_ref': 'artifact:styles',
                    'path': '/artifacts/documents/styles.css',
                    'text_artifact_source_name': 'styles',
                    'text_artifact_extension': 'css',
                },
            ],
            resolve_saved_path=lambda raw_path: (
                Path(str(raw_path)) if str(raw_path).endswith('configurator.html') else None
            ),
        )
        request_payload = {
            'conversation_id': 'conv-bundle-edit',
            'prompt': 'Update configurator.html and styles.css, preserving everything else.',
            'selected_reference_artifact': {
                'type': 'message',
                'message_id': 'ui-msg-bundle',
            },
        }

        promoted = owner.promote_current_predecessor_context(request_payload)

        self.assertEqual(promoted, request_payload)
        self.assertNotIn('current_predecessor_context', promoted)


if __name__ == '__main__':
    unittest.main()
