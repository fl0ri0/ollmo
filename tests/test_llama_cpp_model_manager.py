import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ollmo_runtime import llama_cpp_model_manager


class LlamaCppModelManagerTests(unittest.TestCase):
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_llama_cli_bin')
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_llama_server_bin')
    def test_runtime_probe_reports_runnable_when_server_binary_exists(
        self,
        mock_server_bin,
        mock_cli_bin,
    ):
        mock_server_bin.return_value = '/opt/homebrew/bin/llama-server'
        mock_cli_bin.return_value = '/opt/homebrew/bin/llama-cli'

        payload = llama_cpp_model_manager.describe_llama_cpp_runtime_probe()

        self.assertEqual(payload['runtime_state'], 'runnable')
        self.assertTrue(payload['operations']['start_instance'])
        self.assertTrue(payload['detection']['server_detected'])
        self.assertTrue(payload['detection']['cli_detected'])

    @patch('ollmo_runtime.llama_cpp_model_manager.describe_llama_cpp_runtime_probe')
    @patch('ollmo_runtime.llama_cpp_model_manager._local_gguf_entries')
    def test_list_local_gguf_models_returns_catalog_entries(
        self,
        mock_local_entries,
        mock_probe,
    ):
        mock_probe.return_value = {'runtime_state': 'runnable', 'issues': []}
        mock_local_entries.return_value = [Path('/Users/example/Models/llama.cpp/gemma-3-1b-it-Q4_K_M.gguf')]

        items = llama_cpp_model_manager.list_local_gguf_models()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['backend'], 'llama_cpp')
        self.assertEqual(items[0]['model_source'], 'local_gguf')
        self.assertEqual(items[0]['model_path'], '/Users/example/Models/llama.cpp/gemma-3-1b-it-Q4_K_M.gguf')

    @patch('ollmo_runtime.llama_cpp_model_manager.describe_llama_cpp_runtime_probe')
    @patch('ollmo_runtime.llama_cpp_model_manager._read_catalog_entries', return_value=[])
    @patch('ollmo_runtime.llama_cpp_model_manager._local_gguf_entries', return_value=[])
    @patch('ollmo_runtime.llama_cpp_model_manager._huggingface_hub_cache_root')
    def test_list_available_llama_cpp_models_rediscovers_cached_hf_repo_without_catalog(
        self,
        mock_cache_root,
        _mock_local_entries,
        _mock_read_catalog,
        mock_probe,
    ):
        mock_probe.return_value = {'runtime_state': 'runnable', 'issues': []}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            repo_root = cache_root / 'models--ggml-org--gemma-4-E4B-it-GGUF'
            snapshot = repo_root / 'snapshots' / 'abc123'
            refs = repo_root / 'refs'
            snapshot.mkdir(parents=True)
            refs.mkdir(parents=True)
            (refs / 'main').write_text('abc123', encoding='utf-8')
            main_model = snapshot / 'gemma-4-e4b-it-Q4_K_M.gguf'
            alt_model = snapshot / 'gemma-4-e4b-it-Q8_0.gguf'
            mmproj = snapshot / 'mmproj-gemma-4-e4b-it-f16.gguf'
            main_model.write_bytes(b'q4')
            alt_model.write_bytes(b'q8')
            mmproj.write_bytes(b'mmproj')
            mock_cache_root.return_value = cache_root

            items = llama_cpp_model_manager.list_available_llama_cpp_models()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['model_source'], 'hf_repo')
        self.assertEqual(items[0]['hf_repo'], 'ggml-org/gemma-4-E4B-it-GGUF')
        self.assertTrue(items[0]['runnable'])
        self.assertEqual(items[0]['model_path'], str(main_model))
        self.assertEqual(items[0]['hf_file'], 'gemma-4-e4b-it-Q4_K_M.gguf')
        self.assertEqual(items[0]['mmproj_path'], str(mmproj))
        self.assertIn('chat', items[0]['supported_capabilities'])
        self.assertIn('vision_analysis', items[0]['supported_capabilities'])
        self.assertIn('image', items[0]['inputs'])
        self.assertEqual(items[0]['backend_metadata']['source'], 'llama_cpp_hf_cache_scan')

    @patch('ollmo_runtime.llama_cpp_model_manager.describe_llama_cpp_runtime_probe')
    @patch('ollmo_runtime.llama_cpp_model_manager._local_gguf_entries')
    def test_list_local_gguf_models_keeps_multimodal_family_text_only_without_mmproj(
        self,
        mock_local_entries,
        mock_probe,
    ):
        mock_probe.return_value = {'runtime_state': 'runnable', 'issues': []}
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'gemma-4-e4b-it-Q4_K_M.gguf'
            model_path.write_bytes(b'model')
            mock_local_entries.return_value = [model_path]

            items = llama_cpp_model_manager.list_local_gguf_models()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['provider_capabilities'], ['chat'])
        self.assertEqual(items[0]['supported_capabilities'], ['chat'])
        self.assertEqual(items[0]['inputs'], ['text'])
        self.assertFalse(items[0]['features']['vision_input'])

    def test_build_instance_record_adds_llama_cpp_contract_metadata(self):
        record = llama_cpp_model_manager.build_instance_record(
            'gemma-3-1b-it-Q4_K_M',
            source_kind='local_gguf',
            port=11551,
            log_path=Path('/tmp/llama_cpp_11551.log'),
            launch_defaults={'prompt_cache': True, 'kv_offload': True, 'flash_attention': 'auto'},
            capability='chat',
            pid=123,
            model_path='/Users/example/Models/llama.cpp/gemma-3-1b-it-Q4_K_M.gguf',
            request_model='gemma-3-1b-it-Q4_K_M',
        )

        self.assertEqual(record['backend_package'], 'llama_cpp')
        self.assertEqual(record['backend_contract'], 'llama.cpp.server')
        self.assertEqual(record['request_model'], 'gemma-3-1b-it-Q4_K_M')
        self.assertEqual(record['backend_metadata']['startup_source'], 'local_gguf')
        self.assertEqual(record['backend_metadata']['launch_defaults']['flash_attention'], 'auto')
        self.assertIn('/v1/chat/completions', record['backend_metadata']['native_endpoint_paths'])
        self.assertIsNone(record['mmproj_path'])

    def test_resolve_model_source_discovers_local_mmproj_for_multimodal_gguf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = Path(tmpdir)
            model_path = snapshot_dir / 'gemma-4-e4b-it-Q4_K_M.gguf'
            mmproj_path = snapshot_dir / 'mmproj-gemma-4-e4b-it-f16.gguf'
            model_path.write_bytes(b'model')
            mmproj_path.write_bytes(b'mmproj')

            payload = llama_cpp_model_manager._resolve_model_source(
                'ggml-org/gemma-4-E4B-it-GGUF',
                str(model_path),
            )

        self.assertEqual(payload['source_kind'], 'local_gguf')
        self.assertEqual(payload['model_path'], str(model_path))
        self.assertEqual(payload['mmproj_path'], str(mmproj_path))

    def test_build_instance_record_drops_vision_truth_without_local_mmproj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'gemma-4-e4b-it-Q4_K_M.gguf'
            model_path.write_bytes(b'model')

            record = llama_cpp_model_manager.build_instance_record(
                'ggml-org/gemma-4-E4B-it-GGUF',
                source_kind='local_gguf',
                port=11551,
                log_path=Path('/tmp/llama_cpp_11551.log'),
                launch_defaults={'prompt_cache': True},
                capability='chat',
                pid=123,
                model_path=str(model_path),
                request_model='ggml-org/gemma-4-E4B-it-GGUF',
            )

        self.assertEqual(record['provider_capabilities'], ['chat'])
        self.assertEqual(record['supported_capabilities'], ['chat'])
        self.assertEqual(record['inputs'], ['text'])
        self.assertFalse(record['features']['vision_input'])
        self.assertIsNone(record['mmproj_path'])

    def test_llama_cpp_launch_defaults_raise_ubatch_for_multimodal_family(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = llama_cpp_model_manager._llama_cpp_launch_defaults(
                None,
                model_name='ggml-org/gemma-4-E4B-it-GGUF',
            )

        self.assertEqual(defaults['batch_size'], 512)
        self.assertEqual(defaults['ubatch_size'], 512)

    def test_llama_cpp_launch_defaults_keep_text_only_ubatch_default(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = llama_cpp_model_manager._llama_cpp_launch_defaults(
                None,
                model_name='Qwen3-Coder-30B-A3B-Instruct',
            )

        self.assertEqual(defaults['batch_size'], 512)
        self.assertEqual(defaults['ubatch_size'], 128)

    def test_llama_cpp_launch_defaults_respect_explicit_ubatch_override(self):
        with patch.dict(os.environ, {'LLAMA_CPP_UBATCH_SIZE': '192'}, clear=True):
            defaults = llama_cpp_model_manager._llama_cpp_launch_defaults(
                None,
                model_name='ggml-org/gemma-4-E4B-it-GGUF',
            )

        self.assertEqual(defaults['batch_size'], 512)
        self.assertEqual(defaults['ubatch_size'], 192)

    @patch('ollmo_runtime.llama_cpp_model_manager.describe_llama_cpp_runtime_probe')
    @patch('ollmo_runtime.llama_cpp_model_manager._resolve_cached_hf_launch_artifacts')
    @patch('ollmo_runtime.llama_cpp_model_manager._read_catalog_entries')
    def test_list_llama_cpp_catalog_models_preserves_multimodal_truth_for_gemma4(
        self,
        mock_read_catalog,
        mock_cached_artifacts,
        mock_probe,
    ):
        mock_probe.return_value = {'runtime_state': 'runnable', 'issues': []}
        mock_cached_artifacts.return_value = {
            'model_path': '/Users/example/.cache/huggingface/hub/models--ggml-org--gemma-4-26B-A4B-it-GGUF/snapshots/abc123/gemma-4-q4.gguf',
            'hf_file': 'gemma-4-q4.gguf',
            'mmproj_path': '/Users/example/.cache/huggingface/hub/models--ggml-org--gemma-4-26B-A4B-it-GGUF/snapshots/abc123/mmproj.gguf',
        }
        mock_read_catalog.return_value = [
            {
                'source_key': 'hf::ggml-org/gemma-4-26B-A4B-it-GGUF::',
                'source_kind': 'hf_repo',
                'model_name': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'display_name': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'hf_repo': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'hf_file': None,
            }
        ]

        items = llama_cpp_model_manager.list_llama_cpp_catalog_models()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['model_source'], 'hf_repo')
        self.assertTrue(items[0]['runnable'])
        self.assertEqual(items[0]['model_path'], mock_cached_artifacts.return_value['model_path'])
        self.assertEqual(items[0]['backend_metadata']['startup_source'], 'local_gguf')
        self.assertIn('chat', items[0]['supported_capabilities'])
        self.assertIn('vision_analysis', items[0]['supported_capabilities'])
        self.assertIn('image', items[0]['inputs'])

    @patch('ollmo_runtime.llama_cpp_model_manager.describe_llama_cpp_runtime_probe')
    @patch('ollmo_runtime.llama_cpp_model_manager._resolve_cached_hf_launch_artifacts', return_value=None)
    @patch('ollmo_runtime.llama_cpp_model_manager._read_catalog_entries')
    def test_list_llama_cpp_catalog_models_marks_uncached_hf_repo_as_non_runnable(
        self,
        mock_read_catalog,
        _mock_cached_artifacts,
        mock_probe,
    ):
        mock_probe.return_value = {'runtime_state': 'runnable', 'issues': []}
        mock_read_catalog.return_value = [
            {
                'source_key': 'hf::nvidia/Gemma-4-31B-IT-NVFP4::',
                'source_kind': 'hf_repo',
                'model_name': 'nvidia/Gemma-4-31B-IT-NVFP4',
                'display_name': 'nvidia/Gemma-4-31B-IT-NVFP4',
                'hf_repo': 'nvidia/Gemma-4-31B-IT-NVFP4',
                'hf_file': None,
            }
        ]

        items = llama_cpp_model_manager.list_llama_cpp_catalog_models()

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0]['runnable'])
        self.assertIn('cannot be started locally', items[0]['disabled_reason'])
        self.assertIn('Pull the model first', items[0]['disabled_reason'])

    @patch('ollmo_runtime.llama_cpp_model_manager.describe_llama_cpp_runtime_probe')
    @patch('ollmo_runtime.llama_cpp_model_manager._huggingface_hub_cache_root')
    @patch('ollmo_runtime.llama_cpp_model_manager._read_catalog_entries')
    def test_list_llama_cpp_catalog_models_includes_size_from_hf_cache_snapshot(
        self,
        mock_read_catalog,
        mock_cache_root,
        mock_probe,
    ):
        mock_probe.return_value = {'runtime_state': 'runnable', 'issues': []}
        mock_read_catalog.return_value = [
            {
                'source_key': 'hf::ggml-org/gemma-4-26B-A4B-it-GGUF::',
                'source_kind': 'hf_repo',
                'model_name': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'display_name': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'hf_repo': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'hf_file': None,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            repo_root = cache_root / 'models--ggml-org--gemma-4-26B-A4B-it-GGUF'
            snapshot = repo_root / 'snapshots' / 'abc123'
            refs = repo_root / 'refs'
            snapshot.mkdir(parents=True)
            refs.mkdir(parents=True)
            (refs / 'main').write_text('abc123', encoding='utf-8')
            gguf_file = snapshot / 'gemma-4-q4.gguf'
            gguf_file.write_bytes(b'x' * (5 * 1024 * 1024))
            mock_cache_root.return_value = cache_root

            items = llama_cpp_model_manager.list_llama_cpp_catalog_models()

        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['size_gb'], round((5 * 1024 * 1024) / (1024 ** 3), 2))

    @patch('ollmo_runtime.llama_cpp_model_manager._upsert_catalog_entry')
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_hf_cli_bin')
    @patch('ollmo_runtime.llama_cpp_model_manager.subprocess.run')
    def test_pull_llama_cpp_model_uses_hf_cli_for_hf_repo_and_persists_catalog(
        self,
        mock_run,
        mock_hf_cli,
        mock_upsert,
    ):
        mock_hf_cli.return_value = '/opt/mlx/venv/bin/hf'
        mock_run.return_value = Mock(returncode=0, stdout='downloaded', stderr='')

        success, message = llama_cpp_model_manager.pull_llama_cpp_model('ggml-org/gemma-4-26B-A4B-it-GGUF')

        self.assertTrue(success)
        self.assertEqual(message, 'downloaded')
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[:2], ['/opt/mlx/venv/bin/hf', 'download'])
        self.assertIn('ggml-org/gemma-4-26B-A4B-it-GGUF', cmd)
        mock_upsert.assert_called_once()

    def test_estimate_llama_cpp_source_size_gb_for_local_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'robot.gguf'
            model_path.write_bytes(b'x' * (3 * 1024 * 1024))

            size_gb = llama_cpp_model_manager._estimate_llama_cpp_source_size_gb(model_path=str(model_path))

        self.assertAlmostEqual(size_gb, round((3 * 1024 * 1024) / (1024 ** 3), 2))

    def test_resolve_hf_cli_bin_finds_mlx_venv_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / 'bin'
            bin_dir.mkdir(parents=True)
            hf_path = bin_dir / 'hf'
            hf_path.write_text('#!/bin/sh\n', encoding='utf-8')
            os.chmod(hf_path, 0o755)

            with patch.object(llama_cpp_model_manager, 'ENV_HF_CLI', None):
                with patch.object(llama_cpp_model_manager, 'DEFAULT_HF_CLI', '/nonexistent/hf'):
                    with patch.object(llama_cpp_model_manager, 'DEFAULT_MLX_HF_CLI', str(hf_path)):
                        with patch('shutil.which', return_value=None):
                            resolved = llama_cpp_model_manager.resolve_hf_cli_bin()

        self.assertEqual(resolved, str(hf_path))

    @patch('ollmo_runtime.llama_cpp_model_manager.list_llama_cpp_instances', return_value=[])
    def test_remove_llama_cpp_model_removes_hf_cache_and_catalog_entry(
        self,
        _mock_list_instances,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / 'hf-cache'
            repo_dir = cache_root / 'models--ggml-org--gemma-4-E4B-it-GGUF'
            snapshot = repo_dir / 'snapshots' / 'abc123'
            snapshot.mkdir(parents=True)
            (snapshot / 'gemma-4-e4b-it-q4_k_m.gguf').write_bytes(b'gguf')
            catalog_path = Path(tmpdir) / 'llama_cpp_catalog.json'
            catalog_path.write_text(
                '[{"source_key":"hf::ggml-org/gemma-4-E4B-it-GGUF::","source_kind":"hf_repo","model_name":"ggml-org/gemma-4-E4B-it-GGUF","hf_repo":"ggml-org/gemma-4-E4B-it-GGUF"}]\n',
                encoding='utf-8',
            )

            with patch.object(llama_cpp_model_manager, 'CATALOG_PATH', catalog_path):
                with patch('ollmo_runtime.llama_cpp_model_manager._huggingface_hub_cache_root', return_value=cache_root):
                    success, message = llama_cpp_model_manager.remove_llama_cpp_model(
                        'ggml-org/gemma-4-E4B-it-GGUF',
                        model_source='hf_repo',
                        hf_repo='ggml-org/gemma-4-E4B-it-GGUF',
                    )
                catalog_contents = catalog_path.read_text(encoding='utf-8').strip()
                repo_exists = repo_dir.exists()

        self.assertTrue(success)
        self.assertIn('removed HF cache', message)
        self.assertFalse(repo_exists)
        self.assertEqual(catalog_contents, '[]')

    @patch('ollmo_runtime.llama_cpp_model_manager.list_llama_cpp_instances', return_value=[])
    def test_remove_llama_cpp_model_removes_local_gguf_and_mmproj(
        self,
        _mock_list_instances,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'gemma-4-e4b-it-Q4_K_M.gguf'
            mmproj_path = Path(tmpdir) / 'mmproj-gemma-4-e4b-it-f16.gguf'
            model_path.write_bytes(b'gguf')
            mmproj_path.write_bytes(b'mmproj')
            catalog_path = Path(tmpdir) / 'llama_cpp_catalog.json'
            catalog_path.write_text(
                f'[{{"source_key":"local::{model_path}","source_kind":"local_gguf","model_name":"ggml-org/gemma-4-E4B-it-GGUF","model_path":"{model_path}"}}]\n',
                encoding='utf-8',
            )

            with patch.object(llama_cpp_model_manager, 'CATALOG_PATH', catalog_path):
                success, message = llama_cpp_model_manager.remove_llama_cpp_model(
                    'ggml-org/gemma-4-E4B-it-GGUF',
                    model_source='local_gguf',
                    model_path=str(model_path),
                )
                catalog_contents = catalog_path.read_text(encoding='utf-8').strip()
                model_exists = model_path.exists()
                mmproj_exists = mmproj_path.exists()

        self.assertTrue(success)
        self.assertIn('local file', message)
        self.assertFalse(model_exists)
        self.assertFalse(mmproj_exists)
        self.assertEqual(catalog_contents, '[]')

    @patch('ollmo_runtime.llama_cpp_model_manager.list_llama_cpp_instances', return_value=[])
    def test_remove_llama_cpp_model_clears_stale_catalog_entry_when_cache_is_missing(
        self,
        _mock_list_instances,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / 'hf-cache'
            catalog_path = Path(tmpdir) / 'llama_cpp_catalog.json'
            catalog_path.write_text(
                '[{"source_key":"hf::ggml-org/gemma-4-26B-A4B-it-GGUF::","source_kind":"hf_repo","model_name":"ggml-org/gemma-4-26B-A4B-it-GGUF","hf_repo":"ggml-org/gemma-4-26B-A4B-it-GGUF"}]\n',
                encoding='utf-8',
            )

            with patch.object(llama_cpp_model_manager, 'CATALOG_PATH', catalog_path):
                with patch('ollmo_runtime.llama_cpp_model_manager._huggingface_hub_cache_root', return_value=cache_root):
                    success, message = llama_cpp_model_manager.remove_llama_cpp_model(
                        'ggml-org/gemma-4-26B-A4B-it-GGUF',
                        model_source='hf_repo',
                        hf_repo='ggml-org/gemma-4-26B-A4B-it-GGUF',
                    )
                catalog_contents = catalog_path.read_text(encoding='utf-8').strip()

        self.assertTrue(success)
        self.assertIn('removed catalog entry', message)
        self.assertEqual(catalog_contents, '[]')

    @patch('ollmo_runtime.llama_cpp_model_manager._register_instance')
    @patch('ollmo_runtime.llama_cpp_model_manager._wait_for_server_ready')
    @patch('ollmo_runtime.llama_cpp_model_manager._llama_cpp_launch_defaults')
    @patch('ollmo_runtime.llama_cpp_model_manager._next_free_port')
    @patch('ollmo_runtime.llama_cpp_model_manager._prune_stale_llama_cpp_entries')
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_llama_server_bin')
    @patch('ollmo_runtime.llama_cpp_model_manager.subprocess.Popen')
    def test_start_llama_cpp_instance_uses_local_gguf_model_path(
        self,
        mock_popen,
        mock_server_bin,
        _mock_prune,
        mock_next_port,
        mock_launch_defaults,
        mock_wait,
        mock_register,
    ):
        mock_server_bin.return_value = '/opt/homebrew/bin/llama-server'
        mock_next_port.return_value = 11551
        mock_wait.return_value = True
        mock_launch_defaults.return_value = {
            'ctx_size': 32768,
            'batch_size': 512,
            'ubatch_size': 128,
            'prompt_cache': True,
            'kv_offload': True,
            'flash_attention': 'auto',
            'supported_flags': {
                'ctx_size': True,
                'batch_size': True,
                'ubatch_size': True,
                'cache_prompt': True,
                'kv_offload': True,
                'flash_attention': True,
            },
        }
        process = Mock()
        process.pid = 4321
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'robot.gguf'
            model_path.write_bytes(b'gguf')
            with patch.object(llama_cpp_model_manager, 'LOG_DIR', Path(tmpdir)):
                instance = llama_cpp_model_manager.start_llama_cpp_instance(
                    'robot-q4',
                    model_path=str(model_path),
                    capability='chat',
                    start_source='api_start_model',
                )

        self.assertEqual(instance['backend'], 'llama_cpp')
        self.assertEqual(instance['port'], 11551)
        self.assertEqual(instance['model_path'], str(model_path))
        cmd = mock_popen.call_args.args[0]
        self.assertIn('/opt/homebrew/bin/llama-server', cmd)
        self.assertIn('-m', cmd)
        self.assertIn(str(model_path), cmd)
        self.assertIn('--alias', cmd)
        self.assertIn('--ctx-size', cmd)
        self.assertIn('32768', cmd)
        self.assertIn('--batch-size', cmd)
        self.assertIn('512', cmd)
        self.assertIn('--ubatch-size', cmd)
        self.assertIn('128', cmd)
        self.assertIn('--cache-prompt', cmd)
        self.assertIn('--kv-offload', cmd)
        self.assertIn('--flash-attn', cmd)
        self.assertIn('auto', cmd)
        mock_register.assert_called_once()

    @patch('ollmo_runtime.llama_cpp_model_manager._register_instance')
    @patch('ollmo_runtime.llama_cpp_model_manager._wait_for_server_ready')
    @patch('ollmo_runtime.llama_cpp_model_manager._llama_cpp_launch_defaults')
    @patch('ollmo_runtime.llama_cpp_model_manager._latest_hf_snapshot_dir')
    @patch('ollmo_runtime.llama_cpp_model_manager._next_free_port')
    @patch('ollmo_runtime.llama_cpp_model_manager._prune_stale_llama_cpp_entries')
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_llama_server_bin')
    @patch('ollmo_runtime.llama_cpp_model_manager.subprocess.Popen')
    def test_start_llama_cpp_instance_uses_cached_hf_repo_locally_when_snapshot_exists(
        self,
        mock_popen,
        mock_server_bin,
        _mock_prune,
        mock_next_port,
        mock_latest_snapshot,
        _mock_launch_defaults,
        mock_wait,
        mock_register,
    ):
        mock_server_bin.return_value = '/opt/homebrew/bin/llama-server'
        mock_next_port.return_value = 11552
        mock_wait.return_value = True
        _mock_launch_defaults.return_value = {
            'ctx_size': 32768,
            'batch_size': 512,
            'ubatch_size': 128,
            'prompt_cache': True,
            'kv_offload': True,
            'flash_attention': 'auto',
            'supported_flags': {
                'ctx_size': True,
                'batch_size': True,
                'ubatch_size': True,
                'cache_prompt': True,
                'kv_offload': True,
                'flash_attention': True,
                'mmproj': True,
            },
        }
        process = Mock()
        process.pid = 9876
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = Path(tmpdir) / 'snapshots' / 'abc123'
            snapshot_dir.mkdir(parents=True)
            main_model = snapshot_dir / 'gemma-4-e4b-it-Q4_K_M.gguf'
            alt_model = snapshot_dir / 'gemma-4-e4b-it-Q8_0.gguf'
            mmproj = snapshot_dir / 'mmproj-gemma-4-e4b-it-f16.gguf'
            main_model.write_bytes(b'q4')
            alt_model.write_bytes(b'q8')
            mmproj.write_bytes(b'mmproj')
            mock_latest_snapshot.return_value = snapshot_dir
            with patch.object(llama_cpp_model_manager, 'LOG_DIR', Path(tmpdir)):
                instance = llama_cpp_model_manager.start_llama_cpp_instance(
                    'ggml-org/gemma-4-E4B-it-GGUF',
                    capability='chat',
                    start_source='api_start_model',
                )

        self.assertEqual(instance['hf_repo'], 'ggml-org/gemma-4-E4B-it-GGUF')
        self.assertEqual(instance['model_path'], str(main_model))
        self.assertEqual(instance['mmproj_path'], str(mmproj))
        cmd = mock_popen.call_args.args[0]
        self.assertIn('-m', cmd)
        self.assertIn(str(main_model), cmd)
        self.assertIn('--mmproj', cmd)
        self.assertIn(str(mmproj), cmd)
        self.assertNotIn('-hf', cmd)
        self.assertIn('--ctx-size', cmd)
        self.assertIn('--batch-size', cmd)
        self.assertIn('--ubatch-size', cmd)
        mock_register.assert_called_once()

    @patch('ollmo_runtime.llama_cpp_model_manager._register_instance')
    @patch('ollmo_runtime.llama_cpp_model_manager._wait_for_server_ready')
    @patch('ollmo_runtime.llama_cpp_model_manager._llama_cpp_launch_defaults')
    @patch('ollmo_runtime.llama_cpp_model_manager._latest_hf_snapshot_dir', return_value=None)
    @patch('ollmo_runtime.llama_cpp_model_manager._next_free_port')
    @patch('ollmo_runtime.llama_cpp_model_manager._prune_stale_llama_cpp_entries')
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_llama_server_bin')
    @patch('ollmo_runtime.llama_cpp_model_manager.subprocess.Popen')
    def test_start_llama_cpp_instance_rejects_uncached_hf_repo(
        self,
        mock_popen,
        mock_server_bin,
        _mock_prune,
        mock_next_port,
        _mock_latest_snapshot,
        _mock_launch_defaults,
        mock_wait,
        _mock_register,
    ):
        mock_server_bin.return_value = '/opt/homebrew/bin/llama-server'
        mock_next_port.return_value = 11552
        mock_wait.return_value = True
        _mock_launch_defaults.return_value = {
            'ctx_size': 32768,
            'batch_size': 512,
            'ubatch_size': 128,
            'prompt_cache': True,
            'kv_offload': True,
            'flash_attention': 'auto',
            'supported_flags': {
                'ctx_size': True,
                'batch_size': True,
                'ubatch_size': True,
                'cache_prompt': True,
                'kv_offload': True,
                'flash_attention': True,
                'mmproj': True,
            },
        }
        process = Mock()
        process.pid = 1234
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(llama_cpp_model_manager, 'LOG_DIR', Path(tmpdir)):
                with self.assertRaisesRegex(ValueError, 'cannot be started locally'):
                    llama_cpp_model_manager.start_llama_cpp_instance(
                        'ggml-org/gemma-3-1b-it-GGUF',
                        capability='chat',
                        start_source='api_start_model',
                    )

        mock_popen.assert_not_called()

    @patch('ollmo_runtime.llama_cpp_model_manager._register_instance')
    @patch('ollmo_runtime.llama_cpp_model_manager._wait_for_server_ready')
    @patch('ollmo_runtime.llama_cpp_model_manager._llama_cpp_launch_defaults')
    @patch('ollmo_runtime.llama_cpp_model_manager._next_free_port')
    @patch('ollmo_runtime.llama_cpp_model_manager._prune_stale_llama_cpp_entries')
    @patch('ollmo_runtime.llama_cpp_model_manager._resolve_model_source')
    @patch('ollmo_runtime.llama_cpp_model_manager.resolve_llama_server_bin')
    @patch('ollmo_runtime.llama_cpp_model_manager.subprocess.Popen')
    def test_start_llama_cpp_instance_archives_existing_log_before_launch(
        self,
        mock_popen,
        mock_server_bin,
        mock_resolve_model_source,
        _mock_prune,
        mock_next_port,
        mock_launch_defaults,
        mock_wait,
        _mock_register,
    ):
        mock_server_bin.return_value = '/opt/homebrew/bin/llama-server'
        mock_next_port.return_value = 11552
        mock_wait.return_value = True
        mock_launch_defaults.return_value = {
            'ctx_size': 32768,
            'batch_size': 512,
            'ubatch_size': 512,
            'prompt_cache': True,
            'kv_offload': True,
            'flash_attention': 'auto',
            'supported_flags': {
                'ctx_size': True,
                'batch_size': True,
                'ubatch_size': True,
                'cache_prompt': True,
                'kv_offload': True,
                'flash_attention': True,
            },
        }
        process = Mock()
        process.pid = 2468
        mock_popen.return_value = process
        mock_resolve_model_source.return_value = {
            'source_kind': 'hf_repo',
            'display_name': 'ggml-org/gemma-4-E4B-it-GGUF',
            'model_path': None,
            'hf_repo': 'ggml-org/gemma-4-E4B-it-GGUF',
            'hf_file': None,
            'mmproj_path': None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            stale_log = log_dir / 'llama_cpp_server_ggml-org_gemma-4-E4B-it-GGUF_11552.log'
            stale_log.write_text('OLD CRASH\n', encoding='utf-8')

            with patch.object(llama_cpp_model_manager, 'LOG_DIR', log_dir):
                llama_cpp_model_manager.start_llama_cpp_instance(
                    'ggml-org/gemma-4-E4B-it-GGUF',
                    capability='chat',
                    start_source='api_start_model',
                )

            self.assertTrue(stale_log.exists())
            self.assertNotIn('OLD CRASH', stale_log.read_text(encoding='utf-8'))
            archived_logs = list((log_dir / 'archive' / 'runtime').rglob('*.log'))
            self.assertEqual(len(archived_logs), 1)
            self.assertIn('superseded_launch', archived_logs[0].name)
            self.assertIn('OLD CRASH', archived_logs[0].read_text(encoding='utf-8'))
            self.assertTrue((log_dir / 'archive' / 'runtime' / 'manifest.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
