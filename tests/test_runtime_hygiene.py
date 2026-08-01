import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_core.status import default_runtime_status, read_runtime_status
from ollmo_runtime.runtime_hygiene import cleanup_runtime_hygiene, finalize_runtime_shutdown
from ollmo_runtime.runtime_log_hygiene import sweep_stale_global_logs, sweep_stale_runtime_logs


class RuntimeHygieneTests(unittest.TestCase):
    def test_sweep_stale_global_logs_archives_stale_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            active_log = log_dir / 'flask_webserver.log'
            stale_global = log_dir / 'ollama_default_server_11434.log'
            legacy_log = log_dir / 'ollmo_webserver.log'
            unrelated_log = log_dir / 'notes.log'
            active_log.write_text('active web\n', encoding='utf-8')
            stale_global.write_text('stale default\n', encoding='utf-8')
            legacy_log.write_text('legacy\n', encoding='utf-8')
            unrelated_log.write_text('keep\n', encoding='utf-8')

            report = sweep_stale_global_logs([active_log], log_dir=log_dir)

            self.assertEqual(report['archived_count'], 2)
            self.assertTrue(active_log.exists())
            self.assertFalse(stale_global.exists())
            self.assertFalse(legacy_log.exists())
            self.assertTrue(unrelated_log.exists())
            archived_logs = list((log_dir / 'archive' / 'global').rglob('*.log'))
            self.assertEqual(len(archived_logs), 2)
            self.assertTrue((log_dir / 'archive' / 'global' / 'README.md').exists())
            self.assertTrue((log_dir / 'archive' / 'global' / 'manifest.jsonl').exists())

    def test_sweep_stale_runtime_logs_archives_only_runtime_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            active_log = log_dir / 'mlx_vlm_11503.log'
            stale_log = log_dir / 'mlx_vlm_11509.log'
            global_log = log_dir / 'flask_webserver.log'
            active_log.write_text('active\n', encoding='utf-8')
            stale_log.write_text('stale\n', encoding='utf-8')
            global_log.write_text('global\n', encoding='utf-8')

            report = sweep_stale_runtime_logs([active_log], log_dir=log_dir)

            self.assertEqual(report['archived_count'], 1)
            self.assertTrue(active_log.exists())
            self.assertTrue(global_log.exists())
            self.assertFalse(stale_log.exists())
            archived_logs = list((log_dir / 'archive' / 'runtime').rglob('*.log'))
            self.assertEqual(len(archived_logs), 1)
            self.assertIn('stale_runtime_log', archived_logs[0].name)
            self.assertIn('stale', archived_logs[0].read_text(encoding='utf-8'))
            self.assertTrue((log_dir / 'archive' / 'runtime' / 'README.md').exists())
            self.assertTrue((log_dir / 'archive' / 'runtime' / 'manifest.jsonl').exists())

    def test_sweep_stale_runtime_logs_keeps_inferred_active_ollama_instance_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            active_ollama_log = log_dir / 'ollama_server_qwen3-coder_latest-1_11435.log'
            stale_log = log_dir / 'mlx_vlm_11509.log'
            active_ollama_log.write_text('active ollama\n', encoding='utf-8')
            stale_log.write_text('stale\n', encoding='utf-8')

            summary = cleanup_runtime_hygiene(
                registry_path=Path(tmpdir) / 'model_ports.json',
                status_path=Path(tmpdir) / 'runtime_status.json',
                log_dir=log_dir,
                sync_external=False,
            )

            self.assertEqual(summary['live_instance_count'], 0)
            self.assertEqual(summary['archived_count'], 2)
            self.assertFalse(active_ollama_log.exists())
            self.assertFalse(stale_log.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            active_ollama_log = log_dir / 'ollama_server_qwen3-coder_latest-1_11435.log'
            stale_log = log_dir / 'mlx_vlm_11509.log'
            active_ollama_log.write_text('active ollama\n', encoding='utf-8')
            stale_log.write_text('stale\n', encoding='utf-8')

            live_entries = [
                {
                    'instance_id': 'qwen3-coder:latest-1',
                    'model': 'qwen3-coder:latest',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11435,
                    'pid': 12345,
                }
            ]

            with patch('ollmo_runtime.runtime_hygiene.list_runtime_entries', return_value=live_entries):
                summary = cleanup_runtime_hygiene(
                    registry_path=Path(tmpdir) / 'model_ports.json',
                    status_path=Path(tmpdir) / 'runtime_status.json',
                    log_dir=log_dir,
                    sync_external=False,
                )

            self.assertEqual(summary['live_instance_count'], 1)
            self.assertEqual(summary['archived_count'], 1)
            self.assertTrue(active_ollama_log.exists())
            self.assertFalse(stale_log.exists())

    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    @patch('ollmo_runtime.runtime_hygiene.list_runtime_entries')
    def test_cleanup_runtime_hygiene_refreshes_status_and_archives_stale_logs(
        self,
        mock_list_runtime_entries,
        _mock_process_alive,
        _mock_port_listening,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'
            registry_path.write_text('[]\n', encoding='utf-8')
            status_path = Path(tmpdir) / 'runtime_status.json'
            stale_payload = default_runtime_status()
            stale_payload['instances'] = {
                'stale-1': {
                    'instance_id': 'stale-1',
                    'readiness': 'ready',
                    'activity': 'idle',
                }
            }
            status_path.write_text(json.dumps(stale_payload), encoding='utf-8')

            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            active_log = log_dir / 'mlx_vlm_11503.log'
            stale_log = log_dir / 'mlx_vlm_11509.log'
            active_log.write_text('active\n', encoding='utf-8')
            stale_log.write_text('stale\n', encoding='utf-8')

            live_entries = [
                {
                    'instance_id': 'active-1',
                    'model': 'mlx-community/Qwen3.5-9B-MLX-4bit',
                    'backend': 'mlx',
                    'capability': 'vision_analysis',
                    'port': 11503,
                    'pid': 12345,
                    'log': str(active_log),
                }
            ]
            mock_list_runtime_entries.return_value = live_entries

            summary = cleanup_runtime_hygiene(
                registry_path=registry_path,
                status_path=status_path,
                log_dir=log_dir,
                sync_external=False,
                active_global_log_paths=[log_dir / 'flask_webserver.log'],
            )

            refreshed = read_runtime_status(status_path)
            self.assertEqual(list(refreshed['instances'].keys()), ['active-1'])
            self.assertEqual(summary['live_instance_count'], 1)
            self.assertEqual(summary['runtime_status_count'], 1)
            self.assertEqual(summary['archived_count'], 1)
            self.assertTrue(active_log.exists())
            self.assertFalse(stale_log.exists())
            archived_logs = list((log_dir / 'archive' / 'runtime').rglob('*.log'))
            self.assertEqual(len(archived_logs), 1)

    def test_cleanup_runtime_hygiene_preserves_active_global_logs_and_archives_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'
            registry_path.write_text('[]\n', encoding='utf-8')
            status_path = Path(tmpdir) / 'runtime_status.json'
            status_path.write_text(json.dumps(default_runtime_status()), encoding='utf-8')

            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            flask_log = log_dir / 'flask_webserver.log'
            default_log = log_dir / 'ollama_default_server_11434.log'
            legacy_log = log_dir / 'flask_webserver_auto.log'
            flask_log.write_text('active web\n', encoding='utf-8')
            default_log.write_text('active default\n', encoding='utf-8')
            legacy_log.write_text('legacy\n', encoding='utf-8')

            summary = cleanup_runtime_hygiene(
                registry_path=registry_path,
                status_path=status_path,
                log_dir=log_dir,
                sync_external=False,
                active_global_log_paths=[flask_log, default_log],
            )

            self.assertEqual(summary['global_archived_count'], 1)
            self.assertEqual(summary['archived_count'], 1)
            self.assertTrue(flask_log.exists())
            self.assertTrue(default_log.exists())
            self.assertFalse(legacy_log.exists())
            archived_logs = list((log_dir / 'archive' / 'global').rglob('*.log'))
            self.assertEqual(len(archived_logs), 1)

    @patch('ollmo_core.registry._sync_downstream_integrations')
    def test_finalize_runtime_shutdown_clears_runtime_state_and_archives_logs(self, mock_sync):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / 'model_ports.json'
            registry_path.write_text(
                json.dumps(
                    [
                        {
                            'instance_id': 'agent-orchestrator',
                            'model': 'agent:orchestrator',
                            'agent': True,
                            'port': 6200,
                        },
                        {
                            'instance_id': 'gemma4:e4b-1',
                            'model': 'gemma4:e4b',
                            'backend': 'ollama',
                            'port': 11435,
                            'pid': 111,
                            'log': 'logs/ollama_server_gemma4_e4b-1_11435.log',
                        },
                    ]
                ),
                encoding='utf-8',
            )
            status_path = Path(tmpdir) / 'runtime_status.json'
            stale_payload = default_runtime_status()
            stale_payload['instances'] = {
                'gemma4:e4b-1': {
                    'instance_id': 'gemma4:e4b-1',
                    'readiness': 'ready',
                }
            }
            status_path.write_text(json.dumps(stale_payload), encoding='utf-8')

            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            instance_log = log_dir / 'ollama_server_gemma4_e4b-1_11435.log'
            load_log = log_dir / 'gemma4_e4b_load_on_11435_gemma4_e4b-1.log'
            global_log = log_dir / 'flask_webserver.log'
            instance_log.write_text('instance\n', encoding='utf-8')
            load_log.write_text('load\n', encoding='utf-8')
            global_log.write_text('global\n', encoding='utf-8')

            summary = finalize_runtime_shutdown(
                registry_path=registry_path,
                status_path=status_path,
                log_dir=log_dir,
                sync_external=True,
                preserve_agents=True,
            )

            persisted_entries = json.loads(registry_path.read_text(encoding='utf-8'))
            self.assertEqual(len(persisted_entries), 1)
            self.assertTrue(persisted_entries[0]['agent'])
            refreshed = read_runtime_status(status_path)
            self.assertEqual(refreshed['instances'], {})
            self.assertEqual(summary['live_instance_count'], 0)
            self.assertEqual(summary['runtime_status_count'], 0)
            self.assertEqual(summary['runtime_archived_count'], 2)
            self.assertEqual(summary['global_archived_count'], 1)
            self.assertEqual(summary['archived_count'], 3)
            self.assertFalse(instance_log.exists())
            self.assertFalse(load_log.exists())
            self.assertFalse(global_log.exists())
            archived_logs = list((log_dir / 'archive' / 'runtime').rglob('*.log'))
            self.assertEqual(len(archived_logs), 2)
            archived_global_logs = list((log_dir / 'archive' / 'global').rglob('*.log'))
            self.assertEqual(len(archived_global_logs), 1)
            mock_sync.assert_called_once()


if __name__ == '__main__':
    unittest.main()
