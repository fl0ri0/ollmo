import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_core.registry import (
    filter_active_registry_entries,
    list_runtime_entries,
    read_registry_entries,
    sanitize_registry_entry_for_persistence,
    write_registry_entries,
)
from ollmo_core.lifecycle import list_running_instances


class RegistryCoreTests(unittest.TestCase):
    @patch('ollmo_core.registry._sync_downstream_integrations')
    def test_write_sync_external_routes_through_shared_downstream_helper(self, mock_sync):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'

            write_registry_entries(
                [
                    {
                        'instance_id': 'qwen3-coder:latest-1',
                        'model': 'qwen3-coder:latest',
                        'port': 11435,
                        'pid': 111,
                    }
                ],
                path=registry_path,
                preserve_agents=False,
                sync_external=True,
            )

            mock_sync.assert_called_once()
            synced_entries = mock_sync.call_args.args[0]
            self.assertEqual(len(synced_entries), 1)
            self.assertEqual(synced_entries[0]['instance_id'], 'qwen3-coder:latest-1')
            self.assertEqual(synced_entries[0]['backend'], 'ollama')

    def test_write_preserves_existing_agent_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'
            registry_path.write_text(
                json.dumps(
                    [
                        {
                            'instance_id': 'agent-orchestrator',
                            'model': 'agent:orchestrator',
                            'role': 'orchestrator',
                            'agent': True,
                            'port': 6200,
                        }
                    ]
                ),
                encoding='utf-8',
            )

            write_registry_entries(
                [
                    {
                        'instance_id': 'qwen3-coder:latest-1',
                        'model': 'qwen3-coder:latest',
                        'port': 11435,
                        'pid': 111,
                    }
                ],
                path=registry_path,
                preserve_agents=True,
                sync_external=False,
            )

            entries = read_registry_entries(registry_path)
            self.assertEqual({entry['instance_id'] for entry in entries}, {'agent-orchestrator', 'qwen3-coder:latest-1'})
            runtime_entry = next(entry for entry in entries if not entry.get('agent'))
            self.assertEqual(runtime_entry['backend'], 'ollama')
            self.assertEqual(runtime_entry['capability'], 'chat')

    @patch('ollmo_core.registry.pid_is_running', return_value=False)
    @patch('ollmo_core.registry.is_port_listening')
    def test_filter_prunes_inactive_runtime_entries_but_keeps_agents(self, mock_is_port_listening, _mock_pid):
        mock_is_port_listening.side_effect = lambda port, host='localhost': int(port) == 11435
        entries = [
            {
                'instance_id': 'qwen3-coder:latest-1',
                'model': 'qwen3-coder:latest',
                'port': 11435,
                'pid': 111,
            },
            {
                'instance_id': 'deepseek-r1:8b-1',
                'model': 'deepseek-r1:8b',
                'port': 11436,
                'pid': 222,
            },
            {
                'instance_id': 'agent-orchestrator',
                'model': 'agent:orchestrator',
                'role': 'orchestrator',
                'agent': True,
                'port': 6200,
            },
        ]

        filtered = filter_active_registry_entries(entries)

        self.assertEqual(
            {entry['instance_id'] for entry in filtered},
            {'qwen3-coder:latest-1', 'agent-orchestrator'},
        )
        self.assertTrue(any(entry.get('agent') for entry in filtered))

    @patch('ollmo_core.registry.pid_is_running', return_value=True)
    @patch('ollmo_core.registry.is_port_listening', return_value=False)
    def test_filter_prunes_port_backed_entry_when_port_is_dead_even_if_pid_still_exists(
        self,
        _mock_is_port_listening,
        _mock_pid_is_running,
    ):
        entries = [
            {
                'instance_id': 'ggml-org__gemma-4-E4B-it-GGUF-llama_cpp-11552',
                'model': 'ggml-org/gemma-4-E4B-it-GGUF',
                'backend': 'llama_cpp',
                'port': 11552,
                'pid': 99999,
            }
        ]

        filtered = filter_active_registry_entries(entries)

        self.assertEqual(filtered, [])

    @patch('ollmo_core.registry.pid_is_running', return_value=False)
    @patch('ollmo_core.registry.is_port_listening', return_value=False)
    def test_list_running_instances_preserves_registry_by_default(
        self,
        mock_is_port_listening,
        _mock_pid_is_running,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'
            registry_path.write_text(
                json.dumps(
                    [
                        {
                            'instance_id': 'tts-1',
                            'model': 'mlx-community/Qwen3-TTS',
                            'backend': 'mlx',
                            'capability': 'text_to_speech',
                            'port': 11503,
                            'pid': 99999,
                        }
                    ]
                ),
                encoding='utf-8',
            )

            instances = list_running_instances(config_path=str(registry_path))

            self.assertEqual([entry['instance_id'] for entry in instances], ['tts-1'])
            self.assertEqual(
                [entry['instance_id'] for entry in read_registry_entries(registry_path)],
                ['tts-1'],
            )
            mock_is_port_listening.assert_not_called()

            pruned = list_runtime_entries(
                prune=True,
                path=registry_path,
                sync_external=False,
            )
            self.assertEqual(pruned, [])
            self.assertEqual(read_registry_entries(registry_path), [])

    def test_sanitize_registry_entry_for_persistence_keeps_stable_truth_and_drops_runtime_noise(self):
        sanitized = sanitize_registry_entry_for_persistence(
            {
                'instance_id': 'mlx-community__DeepSeek-OCR-2-bf16-mlx-11503',
                'model': 'mlx-community/DeepSeek-OCR-2-bf16',
                'backend': 'mlx',
                'capability': 'vision_analysis',
                'backend_package': 'mlx_vlm',
                'backend_contract': 'mlx_vlm.server',
                'provider_capabilities': ['vision_analysis'],
                'backend_metadata': {'source': 'mlx_package_contract', 'package_label': 'mlx-vlm'},
                'backend_runtime': {'source': 'mlx_runtime_registry', 'single_loaded_model': True},
                'runtime_status': {'readiness': 'ready'},
                'session_controls': {'enabled': True},
                'readiness': 'ready',
                'activity': 'idle',
            }
        )

        self.assertEqual(sanitized['backend_package'], 'mlx_vlm')
        self.assertEqual(sanitized['backend_contract'], 'mlx_vlm.server')
        self.assertEqual(sanitized['provider_capabilities'], ['vision_analysis'])
        self.assertEqual(sanitized['backend_metadata']['package_label'], 'mlx-vlm')
        self.assertNotIn('backend_runtime', sanitized)
        self.assertNotIn('runtime_status', sanitized)
        self.assertNotIn('session_controls', sanitized)
        self.assertNotIn('readiness', sanitized)

    def test_write_registry_entries_derives_ollama_package_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'

            write_registry_entries(
                [
                    {
                        'instance_id': 'gpt-oss:20b-1',
                        'model': 'gpt-oss:20b',
                        'backend': 'ollama',
                        'port': 11437,
                    }
                ],
                path=registry_path,
                preserve_agents=False,
                sync_external=False,
            )

            entries = read_registry_entries(registry_path)
            self.assertEqual(entries[0]['backend_package'], 'ollama')
            self.assertEqual(entries[0]['backend_contract'], 'ollama.api')

    def test_sanitize_registry_entry_keeps_mlx_lm_chat_instance_from_advertising_vision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / 'qwen-vision-chat'
            model_dir.mkdir()
            (model_dir / 'config.json').write_text(
                json.dumps(
                    {
                        'model_type': 'qwen3_5',
                        'image_token_id': 248056,
                    }
                ),
                encoding='utf-8',
            )
            (model_dir / 'README.md').write_text(
                '---\n'
                'tags:\n'
                '  - mlx\n'
                '  - vision-language-model\n'
                '---\n',
                encoding='utf-8',
            )

            sanitized = sanitize_registry_entry_for_persistence(
                {
                    'instance_id': 'mlx-community__Qwen3.5-9B-MLX-4bit-mlx-11509',
                    'model': 'mlx-community/Qwen3.5-9B-MLX-4bit',
                    'backend': 'mlx',
                    'capability': 'chat',
                    'supported_capabilities': ['chat', 'vision_analysis'],
                    'inputs': ['text', 'image'],
                    'features': {
                        'vision_input': True,
                    },
                    'backend_package': 'mlx_lm',
                    'backend_contract': 'mlx_lm.server',
                    'provider_capabilities': ['chat'],
                    'backend_metadata': {
                        'source': 'mlx_package_contract',
                        'backend_package': 'mlx_lm',
                        'backend_contract': 'mlx_lm.server',
                        'capabilities': ['chat'],
                        'package_capabilities': ['chat'],
                    },
                    'model_path': str(model_dir),
                    'request_model': str(model_dir),
                    'mlx_server': 'mlx_lm',
                }
            )

        self.assertEqual(sanitized['capability'], 'chat')
        self.assertEqual(sanitized['provider_capabilities'], ['chat'])
        self.assertEqual(sanitized['supported_capabilities'], ['chat'])
        self.assertFalse(sanitized['features']['vision_input'])
        self.assertEqual(sanitized['inputs'], ['text'])


if __name__ == '__main__':
    unittest.main()
