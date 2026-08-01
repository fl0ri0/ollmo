from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import ollmoctl
from ollmo_runtime import (
    llama_cpp_model_manager,
    mlx_model_manager,
    ollama_model_manager,
)
from ollmo_runtime.child_process_env import (
    GRAPH_REBASE_OPERATOR_ENV_KEYS,
    sanitized_child_process_env,
)
from ollmo_runtime.ollama_model_manager import build_ollama_env


REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET_ENV = {
    key: f'dummy-secret-{index}'
    for index, key in enumerate(GRAPH_REBASE_OPERATOR_ENV_KEYS, start=1)
}


class ChildProcessEnvironmentTests(unittest.TestCase):
    def assert_operator_secrets_absent(self, env: dict[str, str]) -> None:
        for key in GRAPH_REBASE_OPERATOR_ENV_KEYS:
            self.assertNotIn(key, env)

    def test_shared_sanitizer_removes_all_operator_names_without_mutating_source(self):
        source = {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'}

        sanitized = sanitized_child_process_env(source)

        self.assert_operator_secrets_absent(sanitized)
        self.assertEqual(sanitized['SAFE_CHILD_VALUE'], 'preserved')
        self.assertEqual(source, {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'})

    @patch('ollmo_runtime.ollama_model_manager.OLLAMA_LIBRARY_DIR_CANDIDATES', [])
    def test_ollama_environment_removes_canonical_and_internal_operator_names(self):
        with patch.dict(
            os.environ,
            {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'},
            clear=True,
        ):
            env = build_ollama_env(port=11436)

        self.assert_operator_secrets_absent(env)
        self.assertEqual(env['SAFE_CHILD_VALUE'], 'preserved')
        self.assertEqual(env['OLLAMA_HOST'], '127.0.0.1:11436')

    def test_all_mlx_server_spawns_use_sanitized_environment(self):
        process = Mock(pid=1234)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / 'logs'
            speech_server = root / 'mlx_whisper_server.py'
            speech_server.write_text('raise SystemExit(0)\n', encoding='utf-8')
            with (
                patch.dict(
                    os.environ,
                    {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'},
                    clear=True,
                ),
                patch.object(mlx_model_manager, 'LOG_DIR', log_dir),
                patch.object(mlx_model_manager, 'SPEECH_SERVER_SCRIPT', speech_server),
                patch.object(
                    mlx_model_manager,
                    'resolve_mlx_python',
                    return_value='/opt/mlx/venv/bin/python',
                ),
                patch.object(
                    mlx_model_manager.subprocess,
                    'Popen',
                    return_value=process,
                ) as mock_popen,
            ):
                mlx_model_manager.launch('/tmp/model', 11501)
                mlx_model_manager.launch_vlm_server(11502)
                mlx_model_manager.launch_audio_server(11503)
                mlx_model_manager.launch_whisper_server('/tmp/whisper', 11504)

        self.assertEqual(mock_popen.call_count, 4)
        for call in mock_popen.call_args_list:
            env = call.kwargs.get('env')
            self.assertIsInstance(env, dict)
            self.assert_operator_secrets_absent(env)
            self.assertEqual(env['SAFE_CHILD_VALUE'], 'preserved')

    def test_llama_cpp_server_and_model_cli_use_sanitized_environment(self):
        process = Mock(pid=4321)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / 'model.gguf'
            model_path.write_bytes(b'gguf')
            with (
                patch.dict(
                    os.environ,
                    {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'},
                    clear=True,
                ),
                patch.object(llama_cpp_model_manager, 'LOG_DIR', root / 'logs'),
                patch.object(
                    llama_cpp_model_manager,
                    'resolve_llama_server_bin',
                    return_value='/opt/homebrew/bin/llama-server',
                ),
                patch.object(llama_cpp_model_manager, '_prune_stale_llama_cpp_entries'),
                patch.object(llama_cpp_model_manager, '_next_free_port', return_value=11551),
                patch.object(
                    llama_cpp_model_manager,
                    '_llama_cpp_launch_defaults',
                    return_value={'supported_flags': {}},
                ),
                patch.object(llama_cpp_model_manager, '_wait_for_server_ready', return_value=True),
                patch.object(llama_cpp_model_manager, '_register_instance'),
                patch.object(
                    llama_cpp_model_manager.subprocess,
                    'Popen',
                    return_value=process,
                ) as mock_popen,
            ):
                llama_cpp_model_manager.start_llama_cpp_instance(
                    'local-model',
                    model_path=str(model_path),
                    capability='chat',
                    start_source='api_start_model',
                )

            with (
                patch.dict(
                    os.environ,
                    {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'},
                    clear=True,
                ),
                patch.object(
                    llama_cpp_model_manager.subprocess,
                    'run',
                    return_value=Mock(returncode=0, stdout='prepared', stderr=''),
                ) as mock_run,
            ):
                success, _message = llama_cpp_model_manager._pull_hf_repo_with_llama_cli(
                    '/opt/homebrew/bin/llama-cli',
                    'example/model',
                )

        self.assertTrue(success)
        for call in (mock_popen.call_args, mock_run.call_args):
            env = call.kwargs.get('env')
            self.assertIsInstance(env, dict)
            self.assert_operator_secrets_absent(env)
            self.assertEqual(env['SAFE_CHILD_VALUE'], 'preserved')

    def test_ollmoctl_recovery_spawn_uses_sanitized_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            webserver_script = root / 'ollmo_webserver.py'
            webserver_script.write_text('raise SystemExit(0)\n', encoding='utf-8')
            webserver_log = root / 'logs' / 'ollmo_webserver.log'
            with (
                patch.dict(
                    os.environ,
                    {**SECRET_ENV, 'SAFE_CHILD_VALUE': 'preserved'},
                    clear=True,
                ),
                patch.object(ollmoctl, 'DEFAULT_LOCAL_WEBSERVER_SCRIPT', webserver_script),
                patch.object(ollmoctl, 'DEFAULT_LOCAL_WEBSERVER_LOG', webserver_log),
                patch.object(ollmoctl, '_is_default_local_control_plane', return_value=True),
                patch.object(
                    ollmoctl,
                    '_wait_for_local_control_plane',
                    side_effect=[False, True],
                ),
                patch.object(ollmoctl.subprocess, 'Popen', return_value=Mock()) as mock_popen,
            ):
                recovered = ollmoctl._attempt_local_control_plane_recovery(
                    'http://127.0.0.1:5001'
                )

        self.assertTrue(recovered)
        env = mock_popen.call_args.kwargs.get('env')
        self.assertIsInstance(env, dict)
        self.assert_operator_secrets_absent(env)
        self.assertEqual(env['SAFE_CHILD_VALUE'], 'preserved')
        self.assertEqual(env['PYTHONUNBUFFERED'], '1')

    def test_utility_child_processes_also_use_sanitized_environment(self):
        completed = Mock(returncode=1, stdout='', stderr='')
        with patch.dict(os.environ, SECRET_ENV, clear=False):
            with patch.object(
                ollama_model_manager.subprocess,
                'run',
                return_value=completed,
            ) as ollama_run:
                ollama_model_manager.list_listening_pids(11434)
            with patch.object(
                mlx_model_manager.subprocess,
                'run',
                return_value=completed,
            ) as mlx_run:
                mlx_model_manager.port_in_use(11500)
            with patch.object(
                ollmoctl.subprocess,
                'run',
                return_value=completed,
            ) as ctl_run:
                ollmoctl._run_local_command(['true'])

        for call in (
            ollama_run.call_args,
            mlx_run.call_args,
            ctl_run.call_args,
        ):
            env = call.kwargs.get('env')
            self.assertIsInstance(env, dict)
            self.assert_operator_secrets_absent(env)

    def test_startup_capture_retains_values_but_unexports_all_operator_names(self):
        script = (REPO_ROOT / 'start_multi_models.sh').read_text(encoding='utf-8')
        prefix, separator, _rest = script.partition(
            ': "${OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS:=4}"'
        )
        self.assertTrue(separator)
        probe = (
            prefix
            + "\nprintf 'CAPTURED:%s|%s\\n' "
            + '"$GRAPH_REBASE_OPERATOR_TOKEN" "$GRAPH_REBASE_OPERATOR_IDENTITY"\n'
            + 'env\n'
        )
        child_env = os.environ.copy()
        child_env.update(
            {
                'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN': 'canonical-token',
                'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY': 'canonical-identity',
                'GRAPH_REBASE_OPERATOR_TOKEN': 'preexported-token',
                'GRAPH_REBASE_OPERATOR_IDENTITY': 'preexported-identity',
            }
        )

        completed = subprocess.run(
            ['/bin/bash', '-c', probe, str(REPO_ROOT / 'start_multi_models.sh')],
            cwd=REPO_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn('CAPTURED:canonical-token|canonical-identity', completed.stdout)
        output_lines = completed.stdout.splitlines()
        for key in GRAPH_REBASE_OPERATOR_ENV_KEYS:
            self.assertFalse(
                any(line.startswith(f'{key}=') for line in output_lines),
                msg=f'{key} remained exported after startup capture',
            )


if __name__ == '__main__':
    unittest.main()
