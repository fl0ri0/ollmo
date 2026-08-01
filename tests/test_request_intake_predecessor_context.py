import unittest

from ollmo_server.request_intake_runtime import RequestIntakeRuntimeOwner
from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner


class RequestIntakePredecessorContextTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
