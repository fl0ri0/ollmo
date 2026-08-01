import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_core.status import (
    default_runtime_status,
    merge_instances_with_runtime_status,
    read_runtime_status,
    record_instance_failure,
    record_instance_started,
    record_instance_success,
    refresh_runtime_status_entries,
    remove_instance_status,
    write_runtime_status,
)


class RuntimeStatusServiceTests(unittest.TestCase):
    def test_default_runtime_status_uses_utc_timestamp_shape(self):
        payload = default_runtime_status()

        self.assertEqual(payload['schema_version'], 1)
        self.assertTrue(payload['updated_at'].endswith('Z'))
        self.assertEqual(payload['instances'], {})

    @patch('ollmo_core.status._fetch_backend_runtime_metadata')
    @patch('ollmo_core.status._port_listening')
    @patch('ollmo_core.status._process_alive')
    def test_merge_instances_defaults_to_cached_observer_mode(
        self,
        mock_process_alive,
        mock_port_listening,
        mock_fetch_backend_runtime_metadata,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            instance = {
                'instance_id': 'gpt-oss:20b-1',
                'model': 'gpt-oss:20b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11435,
                'pid': 12345,
            }
            payload = default_runtime_status()
            payload['instances'][instance['instance_id']] = {
                'instance_id': instance['instance_id'],
                'readiness': 'ready',
                'activity': 'idle',
                'last_checked_at': '2026-05-25T00:00:00Z',
            }
            write_runtime_status(payload, path=status_path)
            before = status_path.read_text(encoding='utf-8')

            merged = merge_instances_with_runtime_status([instance], path=status_path)

            after = status_path.read_text(encoding='utf-8')
            self.assertEqual(after, before)
            self.assertEqual(merged[0]['runtime_status']['readiness'], 'ready')
            mock_process_alive.assert_not_called()
            mock_port_listening.assert_not_called()
            mock_fetch_backend_runtime_metadata.assert_not_called()

    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    def test_record_instance_lifecycle_and_prune_status(self, _mock_process_alive, _mock_port_listening):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            instance = {
                'instance_id': 'gpt-oss:20b-1',
                'model': 'gpt-oss:20b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11435,
                'pid': 12345,
            }

            _, started = record_instance_started(instance, path=status_path)
            self.assertEqual(started['readiness'], 'ready')

            _, ready = record_instance_success(
                instance['instance_id'],
                path=status_path,
                instance=instance,
                latency_sec=1.25,
            )
            self.assertEqual(ready['readiness'], 'ready')
            self.assertEqual(ready['last_latency_sec'], 1.25)
            self.assertNotIn('last_error', ready)

            _, failed = record_instance_failure(
                instance['instance_id'],
                path=status_path,
                instance=instance,
                message='Timeout Port 11435',
                latency_sec=5.0,
            )
            self.assertEqual(failed['readiness'], 'degraded')
            self.assertEqual(failed['last_error'], 'Timeout Port 11435')

            merged = merge_instances_with_runtime_status([instance], path=status_path, refresh=False)
            self.assertEqual(merged[0]['readiness'], 'degraded')
            self.assertEqual(merged[0]['runtime_status']['last_error'], 'Timeout Port 11435')

            removed = remove_instance_status(instance['instance_id'], path=status_path)
            self.assertEqual(removed['instance_id'], instance['instance_id'])
            payload = read_runtime_status(status_path)
            self.assertEqual(payload['instances'], {})

    @patch('ollmo_core.status._port_listening', return_value=False)
    @patch('ollmo_core.status._process_alive', return_value=False)
    def test_refresh_runtime_status_marks_unreachable_and_prunes_stale(self, _mock_process_alive, _mock_port_listening):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            record_instance_started(
                {
                    'instance_id': 'stale-1',
                    'model': 'old-model',
                    'backend': 'ollama',
                    'capability': 'chat',
                    'port': 11435,
                    'pid': 12345,
                },
                path=status_path,
            )

            refreshed = refresh_runtime_status_entries(
                [
                    {
                        'instance_id': 'active-1',
                        'model': 'qwen3.5:27b',
                        'backend': 'ollama',
                        'capability': 'chat',
                        'port': 11436,
                        'pid': 99999,
                    }
                ],
                path=status_path,
            )

            self.assertEqual(list(refreshed.keys()), ['active-1'])
            self.assertEqual(refreshed['active-1']['readiness'], 'unreachable')

    @patch('ollmo_core.status._fetch_backend_runtime_metadata', return_value={})
    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    def test_refresh_runtime_status_treats_restart_after_failure_as_recovery_evidence(
        self,
        _mock_process_alive,
        _mock_port_listening,
        _mock_fetch_backend_runtime_metadata,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            instance = {
                'instance_id': 'qwen3.6:latest-1',
                'model': 'qwen3.6:latest',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11438,
                'pid': 99999,
            }

            record_instance_started(instance, path=status_path)
            record_instance_failure(
                instance['instance_id'],
                path=status_path,
                instance=instance,
                message='llama runner terminated',
            )

            payload = read_runtime_status(status_path)
            entry = payload['instances'][instance['instance_id']]
            entry['last_error_at'] = '2026-05-16T08:00:00Z'
            entry['last_started_at'] = '2026-05-16T08:01:00Z'
            entry['last_error'] = 'llama runner terminated'
            entry['readiness'] = 'degraded'
            write_runtime_status(payload, path=status_path)

            refreshed = refresh_runtime_status_entries([instance], path=status_path)

            self.assertEqual(refreshed[instance['instance_id']]['readiness'], 'ready')
            self.assertEqual(
                refreshed[instance['instance_id']]['last_error_at'],
                '2026-05-16T08:00:00Z',
            )
            self.assertEqual(
                refreshed[instance['instance_id']]['last_error'],
                'llama runner terminated',
            )

            payload = read_runtime_status(status_path)
            entry = payload['instances'][instance['instance_id']]
            entry['last_error_at'] = '2026-05-16T08:02:00Z'
            entry['last_error'] = 'llama runner terminated again'
            write_runtime_status(payload, path=status_path)

            refreshed = refresh_runtime_status_entries([instance], path=status_path)

            self.assertEqual(refreshed[instance['instance_id']]['readiness'], 'degraded')

    @patch('ollmo_core.status._fetch_backend_runtime_metadata')
    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    def test_refresh_runtime_status_includes_ollama_ps_runtime_metadata(
        self,
        _mock_process_alive,
        _mock_port_listening,
        mock_fetch_backend_runtime_metadata,
    ):
        mock_fetch_backend_runtime_metadata.return_value = {
            'source': 'ollama_api_ps',
            'name': 'qwen3-coder:latest',
            'size_vram': 1234,
            'context_length': 32768,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            refreshed = refresh_runtime_status_entries(
                [
                    {
                        'instance_id': 'qwen3-coder:latest-1',
                        'model': 'qwen3-coder:latest',
                        'backend': 'ollama',
                        'capability': 'chat',
                        'port': 11438,
                        'pid': 99999,
                    }
                ],
                path=status_path,
            )

            entry = refreshed['qwen3-coder:latest-1']
            self.assertEqual(entry['backend_runtime']['source'], 'ollama_api_ps')
            self.assertEqual(entry['backend_runtime']['context_length'], 32768)

            merged = merge_instances_with_runtime_status(
                [
                    {
                        'instance_id': 'qwen3-coder:latest-1',
                        'model': 'qwen3-coder:latest',
                        'backend': 'ollama',
                        'capability': 'chat',
                        'port': 11438,
                        'pid': 99999,
                    }
                ],
                path=status_path,
                refresh=False,
            )
            self.assertEqual(merged[0]['backend_runtime']['size_vram'], 1234)

    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    def test_refresh_runtime_status_includes_mlx_runtime_metadata(self, _mock_process_alive, _mock_port_listening):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            refreshed = refresh_runtime_status_entries(
                [
                    {
                        'instance_id': 'mlx-community__DeepSeek-OCR-2-bf16-mlx-11502',
                        'model': 'mlx-community/DeepSeek-OCR-2-bf16',
                        'backend': 'mlx',
                        'backend_package': 'mlx_vlm',
                        'backend_contract': 'mlx_vlm.server',
                        'capability': 'vision_analysis',
                        'port': 11502,
                        'pid': 99999,
                        'mlx_server': 'mlx_vlm',
                        'model_path': '/tmp/deepseek-ocr',
                        'request_model': '/tmp/deepseek-ocr',
                        'log': '/tmp/mlx_vlm_11502.log',
                        'launch_defaults': {
                            'kv_bits': 3.0,
                            'kv_quant_scheme': 'turboquant',
                            'max_kv_size': 8192,
                        },
                    }
                ],
                path=status_path,
            )

            entry = refreshed['mlx-community__DeepSeek-OCR-2-bf16-mlx-11502']
            self.assertEqual(entry['backend_runtime']['source'], 'mlx_runtime_registry')
            self.assertEqual(entry['backend_runtime']['backend_package'], 'mlx_vlm')
            self.assertEqual(entry['backend_runtime']['health_url'], 'http://127.0.0.1:11502/health')
            self.assertTrue(entry['backend_runtime']['supports_unload'])
            self.assertIn('kv_bits', entry['backend_runtime']['runtime_knobs'])
            self.assertIn('token_queue_timeout', entry['backend_runtime']['runtime_knobs'])
            self.assertEqual(entry['backend_runtime']['launch_defaults']['kv_quant_scheme'], 'turboquant')

            merged = merge_instances_with_runtime_status(
                [
                    {
                        'instance_id': 'mlx-community__DeepSeek-OCR-2-bf16-mlx-11502',
                        'model': 'mlx-community/DeepSeek-OCR-2-bf16',
                        'backend': 'mlx',
                        'backend_package': 'mlx_vlm',
                        'backend_contract': 'mlx_vlm.server',
                        'capability': 'vision_analysis',
                        'port': 11502,
                        'pid': 99999,
                        'mlx_server': 'mlx_vlm',
                    }
                ],
                path=status_path,
                refresh=False,
            )
            self.assertEqual(merged[0]['backend_runtime']['models_url'], 'http://127.0.0.1:11502/v1/models')

    @patch(
        'ollmo_core.status._fetch_backend_runtime_metadata',
        return_value={'source': 'ollama_api_ps', 'model_active': False},
    )
    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    def test_refresh_runtime_status_clears_stale_timeout_degraded_and_busy_when_backend_is_idle(
        self,
        _mock_process_alive,
        _mock_port_listening,
        _mock_fetch_backend_runtime_metadata,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            instance = {
                'instance_id': 'qwen3-coder:latest-1',
                'model': 'qwen3-coder:latest',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11438,
                'pid': 99999,
            }
            record_instance_started(instance, path=status_path)
            record_instance_failure(
                instance['instance_id'],
                path=status_path,
                instance=instance,
                message='Timed out waiting for the next generated token',
            )

            payload = read_runtime_status(status_path)
            entry = payload['instances'][instance['instance_id']]
            entry['activity'] = 'busy'
            write_runtime_status(payload, path=status_path)

            refreshed = refresh_runtime_status_entries([instance], path=status_path)

            entry = refreshed[instance['instance_id']]
            self.assertEqual(entry['readiness'], 'ready')
            self.assertEqual(entry['activity'], 'idle')
            self.assertEqual(entry['busy_clear_reason'], 'backend_runtime_idle')
            self.assertEqual(entry['degraded_clear_reason'], 'backend_runtime_idle_after_timeout')
            self.assertNotIn('cooldown_until', entry)
            self.assertNotIn('failure_cooldown_until', entry)

    @patch(
        'ollmo_core.status._fetch_backend_runtime_metadata',
        return_value={'source': 'ollama_api_ps', 'model_active': True},
    )
    @patch('ollmo_core.status._port_listening', return_value=True)
    @patch('ollmo_core.status._process_alive', return_value=True)
    def test_refresh_runtime_status_clears_stale_timeout_degraded_when_ollama_model_is_active(
        self,
        _mock_process_alive,
        _mock_port_listening,
        _mock_fetch_backend_runtime_metadata,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / 'runtime_status.json'
            instance = {
                'instance_id': 'gemma4:26b-1',
                'model': 'gemma4:26b',
                'backend': 'ollama',
                'capability': 'chat',
                'port': 11437,
                'pid': 99999,
            }
            record_instance_started(instance, path=status_path)
            record_instance_failure(
                instance['instance_id'],
                path=status_path,
                instance=instance,
                message='Infer request timed out.',
            )

            refreshed = refresh_runtime_status_entries([instance], path=status_path)

            entry = refreshed[instance['instance_id']]
            self.assertEqual(entry['readiness'], 'ready')
            self.assertEqual(entry['activity'], 'idle')
            self.assertEqual(entry['degraded_clear_reason'], 'backend_runtime_idle_after_timeout')
            self.assertNotIn('cooldown_until', entry)
            self.assertNotIn('failure_cooldown_until', entry)


if __name__ == '__main__':
    unittest.main()
