import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ollmo_runtime import ollama_model_manager


class OllamaModelManagerTests(unittest.TestCase):
    @patch('ollmo_runtime.ollama_model_manager.time.sleep', return_value=None)
    @patch('ollmo_runtime.ollama_model_manager.subprocess.Popen')
    @patch('ollmo_runtime.ollama_model_manager.is_port_listening', side_effect=[False, True])
    def test_ensure_default_server_running_rotates_default_server_log(
        self,
        _mock_is_port_listening,
        mock_popen,
        _mock_sleep,
    ):
        mock_popen.return_value = SimpleNamespace(pid=33333)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            stale_log = log_dir / 'ollama_default_server_11434.log'
            stale_log.write_text('old default\n', encoding='utf-8')

            with patch.object(ollama_model_manager, 'LOG_DIR', log_dir):
                ok = ollama_model_manager.ensure_default_server_running()
                self.assertTrue(ok)
                self.assertTrue(stale_log.exists())
                self.assertEqual(stale_log.read_text(encoding='utf-8'), '')
                archived_logs = list((log_dir / 'archive' / 'global').rglob('*.log'))
                self.assertEqual(len(archived_logs), 1)

    @patch('ollmo_runtime.ollama_model_manager.wait_for_model_loaded', return_value=True)
    @patch('ollmo_runtime.ollama_model_manager._fetch_model_metadata', return_value={})
    @patch('ollmo_runtime.ollama_model_manager.is_port_listening', return_value=True)
    @patch('ollmo_runtime.ollama_model_manager.subprocess.Popen')
    @patch('ollmo_runtime.ollama_model_manager.time.sleep', return_value=None)
    @patch('ollmo_runtime.ollama_model_manager.find_free_port', return_value=11435)
    @patch('ollmo_runtime.ollama_model_manager.allocate_instance_id', return_value='qwen3-coder:latest-1')
    def test_start_model_records_active_ollama_instance_log_path(
        self,
        _mock_allocate_instance_id,
        _mock_find_free_port,
        _mock_sleep,
        mock_popen,
        _mock_is_port_listening,
        _mock_fetch_model_metadata,
        _mock_wait_for_model_loaded,
    ):
        mock_popen.side_effect = [
            SimpleNamespace(pid=11111),
            SimpleNamespace(pid=22222, terminate=lambda: None),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            with patch.object(ollama_model_manager, 'LOG_DIR', log_dir):
                result = ollama_model_manager.start_model(
                    'qwen3-coder:latest',
                    used_ports=set(),
                    existing_instances=[],
                    capability='chat',
                    start_source='api_start_model',
                )

        self.assertIsNotNone(result)
        self.assertEqual(result['port'], 11435)
        self.assertEqual(
            result['log'],
            str(log_dir / 'ollama_server_qwen3-coder_latest-1_11435.log'),
        )
        self.assertEqual(result['start_source'], 'api_start_model')


if __name__ == '__main__':
    unittest.main()
