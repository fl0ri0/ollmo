from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from ollmo_integrations.codex.execution import (
    CODEX_EXECUTABLE_ENV,
    CodexAccessState,
    CodexDiscovery,
    CodexDiscoverySource,
    CodexExecutionInput,
    CodexExecutionState,
    clear_codex_access_cache,
    discover_codex_executable,
    execute_codex_request,
    execute_codex_text,
    probe_codex_access,
)
from ollmo_runtime.child_process_env import GRAPH_REBASE_OPERATOR_ENV_KEYS


FAKE_CODEX_SOURCE = r'''
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def append_call(args):
    call_log = os.environ.get('FAKE_CODEX_CALL_LOG')
    if not call_log:
        return
    with Path(call_log).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(args) + '\n')


args = sys.argv[1:]
append_call(args)
mode = os.environ.get('FAKE_CODEX_MODE', 'success')

if args == ['--version']:
    print('codex-cli test-version')
    raise SystemExit(0)

if args == ['login', 'status']:
    counter_path = os.environ.get('FAKE_CODEX_STATUS_COUNTER')
    if counter_path:
        path = Path(counter_path)
        current = int(path.read_text(encoding='utf-8')) if path.exists() else 0
        path.write_text(str(current + 1), encoding='utf-8')
    if mode == 'auth_required':
        print('Not logged in. Run codex login.', file=sys.stderr)
        raise SystemExit(1)
    if mode == 'status_error':
        print('status backend unavailable', file=sys.stderr)
        raise SystemExit(2)
    print('Logged in using ChatGPT')
    raise SystemExit(0)

if not args or args[0] != 'exec':
    print('unexpected invocation', file=sys.stderr)
    raise SystemExit(64)

prompt = sys.stdin.read()
entries_before_output = sorted(os.listdir('.'))
output_index = args.index('--output-last-message') + 1
output_path = Path(args[output_index])
capture_path = os.environ.get('FAKE_CODEX_EXEC_CAPTURE')
if capture_path:
    staged_files = []
    for staged_path in sorted(Path('inputs').glob('*')):
        if not staged_path.is_file():
            continue
        staged_bytes = staged_path.read_bytes()
        staged_files.append(
            {
                'path': staged_path.as_posix(),
                'byte_size': len(staged_bytes),
                'sha256': hashlib.sha256(staged_bytes).hexdigest(),
            }
        )
    capture = {
        'args': args,
        'prompt': prompt,
        'cwd': os.getcwd(),
        'entries_before_output': entries_before_output,
        'output_parent': str(output_path.parent),
        'operator_env_present': [
            key
            for key in (
                'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN',
                'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY',
                'GRAPH_REBASE_OPERATOR_TOKEN',
                'GRAPH_REBASE_OPERATOR_IDENTITY',
            )
            if key in os.environ
        ],
        'codex_home': os.environ.get('CODEX_HOME'),
        'openai_api_key_present': 'OPENAI_API_KEY' in os.environ,
        'staged_files': staged_files,
    }
    Path(capture_path).write_text(json.dumps(capture), encoding='utf-8')

if mode == 'exec_timeout':
    time.sleep(1.0)
    raise SystemExit(0)
if mode == 'exec_fail':
    secret = os.environ.get('OPENAI_API_KEY', 'missing-secret')
    print(f'token={secret} direct={secret}', file=sys.stderr)
    raise SystemExit(7)
if mode == 'exec_loud_fail':
    print('E' * 10000, file=sys.stderr)
    raise SystemExit(8)
if mode == 'exec_empty':
    output_path.write_text('', encoding='utf-8')
    raise SystemExit(0)
if mode == 'exec_large':
    output_path.write_text('x' * 256, encoding='utf-8')
    raise SystemExit(0)
if mode == 'exec_warning':
    print('nonterminal warning', file=sys.stderr)

output_path.write_text(
    os.environ.get('FAKE_CODEX_OUTPUT', 'final answer'),
    encoding='utf-8',
)
raise SystemExit(0)
'''


class CodexExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_codex_access_cache()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        clear_codex_access_cache()
        self._tmpdir.cleanup()

    def _write_codex(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'#!{sys.executable}\n{FAKE_CODEX_SOURCE}',
            encoding='utf-8',
        )
        path.chmod(0o755)
        return path

    def _discovery(self, executable: Path) -> CodexDiscovery:
        return CodexDiscovery(
            available=True,
            source=CodexDiscoverySource.EXPLICIT,
            executable=executable.resolve(),
            version='codex-cli test-version',
        )

    def _base_env(self, **updates: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(updates)
        return env

    def test_discovery_prefers_explicit_override(self) -> None:
        explicit = self._write_codex(self.root / 'explicit-codex')
        system = self._write_codex(self.root / 'system-codex')
        path_dir = self.root / 'bin'
        self._write_codex(path_dir / 'codex')
        env = self._base_env(
            **{
                CODEX_EXECUTABLE_ENV: str(explicit),
                'PATH': str(path_dir),
            }
        )

        result = discover_codex_executable(
            env=env,
            home=self.root / 'home',
            system_app_executable=system,
        )

        self.assertTrue(result.available)
        self.assertEqual(result.source, CodexDiscoverySource.EXPLICIT)
        self.assertEqual(result.executable, explicit.resolve())
        self.assertEqual(result.version, 'codex-cli test-version')

    def test_invalid_explicit_override_falls_back_to_system_app(self) -> None:
        system = self._write_codex(self.root / 'system-codex')
        env = self._base_env(
            **{
                CODEX_EXECUTABLE_ENV: str(self.root / 'missing-codex'),
                'PATH': '',
            }
        )

        result = discover_codex_executable(
            env=env,
            home=self.root / 'home',
            system_app_executable=system,
            version_timeout_seconds=0,
        )

        self.assertEqual(result.source, CodexDiscoverySource.CHATGPT_APP_SYSTEM)
        self.assertIn('explicit_override_unusable', result.diagnostic or '')

    def test_discovery_prefers_system_app_over_user_app_and_path(self) -> None:
        system = self._write_codex(self.root / 'system-codex')
        home = self.root / 'home'
        self._write_codex(
            home
            / 'Applications'
            / 'ChatGPT.app'
            / 'Contents'
            / 'Resources'
            / 'codex'
        )
        path_dir = self.root / 'bin'
        self._write_codex(path_dir / 'codex')

        result = discover_codex_executable(
            env=self._base_env(PATH=str(path_dir)),
            home=home,
            system_app_executable=system,
            version_timeout_seconds=0,
        )

        self.assertEqual(result.source, CodexDiscoverySource.CHATGPT_APP_SYSTEM)

    def test_discovery_uses_user_app_before_path(self) -> None:
        home = self.root / 'home'
        user_app = self._write_codex(
            home
            / 'Applications'
            / 'ChatGPT.app'
            / 'Contents'
            / 'Resources'
            / 'codex'
        )
        path_dir = self.root / 'bin'
        self._write_codex(path_dir / 'codex')

        result = discover_codex_executable(
            env=self._base_env(PATH=str(path_dir)),
            home=home,
            system_app_executable=self.root / 'missing-system-codex',
            version_timeout_seconds=0,
        )

        self.assertEqual(result.source, CodexDiscoverySource.CHATGPT_APP_USER)
        self.assertEqual(result.executable, user_app.resolve())

    def test_discovery_uses_path_last_and_reports_not_found_truthfully(self) -> None:
        path_dir = self.root / 'bin'
        path_codex = self._write_codex(path_dir / 'codex')
        path_result = discover_codex_executable(
            env=self._base_env(PATH=str(path_dir)),
            home=self.root / 'home',
            system_app_executable=self.root / 'missing-system-codex',
            version_timeout_seconds=0,
        )
        missing_result = discover_codex_executable(
            env=self._base_env(PATH=''),
            home=self.root / 'other-home',
            system_app_executable=self.root / 'missing-system-codex',
            version_timeout_seconds=0,
        )

        self.assertEqual(path_result.source, CodexDiscoverySource.PATH)
        self.assertEqual(path_result.executable, path_codex.resolve())
        self.assertFalse(missing_result.available)
        self.assertIsNone(missing_result.executable)
        self.assertIn('codex_executable_not_found', missing_result.diagnostic or '')

    def test_login_status_probe_is_structured_and_cached(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        counter = self.root / 'status-counter'
        calls = self.root / 'calls.jsonl'
        env = self._base_env(
            FAKE_CODEX_STATUS_COUNTER=str(counter),
            FAKE_CODEX_CALL_LOG=str(calls),
        )

        first = probe_codex_access(
            self._discovery(executable),
            env=env,
        )
        second = probe_codex_access(
            self._discovery(executable),
            env=env,
        )

        self.assertEqual(first.status, CodexAccessState.AVAILABLE)
        self.assertEqual(first.auth_method, 'chatgpt')
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(counter.read_text(encoding='utf-8'), '1')
        call_rows = [
            json.loads(line)
            for line in calls.read_text(encoding='utf-8').splitlines()
        ]
        self.assertEqual(call_rows, [['login', 'status']])

    def test_login_status_distinguishes_auth_required_and_degraded(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        discovery = self._discovery(executable)

        auth_required = probe_codex_access(
            discovery,
            env=self._base_env(FAKE_CODEX_MODE='auth_required'),
            force_refresh=True,
        )
        degraded = probe_codex_access(
            discovery,
            env=self._base_env(FAKE_CODEX_MODE='status_error'),
            force_refresh=True,
        )

        self.assertEqual(auth_required.status, CodexAccessState.AUTH_REQUIRED)
        self.assertEqual(auth_required.exit_code, 1)
        self.assertEqual(degraded.status, CodexAccessState.DEGRADED)
        self.assertEqual(degraded.exit_code, 2)

    def test_execution_uses_safe_exact_contract_and_sanitized_environment(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        capture = self.root / 'execution.json'
        codex_home = self.root / 'codex-home'
        env = self._base_env(
            FAKE_CODEX_EXEC_CAPTURE=str(capture),
            FAKE_CODEX_OUTPUT='safe final text',
            CODEX_HOME=str(codex_home),
            OPENAI_API_KEY='preserved-auth-mechanism',
            **{
                key: f'must-not-reach-child-{index}'
                for index, key in enumerate(
                    GRAPH_REBASE_OPERATOR_ENV_KEYS,
                    start=1,
                )
            },
        )

        result = execute_codex_text(
            '  hello\r\nworld  ',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=env,
        )

        self.assertEqual(result.status, CodexExecutionState.COMPLETED)
        self.assertEqual(result.output_text, 'safe final text')
        payload = json.loads(capture.read_text(encoding='utf-8'))
        self.assertEqual(payload['prompt'], 'hello\nworld')
        self.assertEqual(payload['entries_before_output'], [])
        self.assertEqual(
            Path(payload['cwd']).resolve(),
            Path(payload['output_parent']).resolve(),
        )
        self.assertEqual(payload['operator_env_present'], [])
        self.assertEqual(payload['codex_home'], str(codex_home))
        self.assertTrue(payload['openai_api_key_present'])
        self.assertEqual(payload['args'][0], 'exec')
        self.assertEqual(payload['args'][-1], '-')
        self.assertIn('--ephemeral', payload['args'])
        self.assertIn('--ignore-user-config', payload['args'])
        self.assertIn('--ignore-rules', payload['args'])
        self.assertIn('--skip-git-repo-check', payload['args'])
        self.assertIn('--output-last-message', payload['args'])
        self.assertEqual(
            payload['args'][payload['args'].index('--sandbox') + 1],
            'read-only',
        )
        self.assertEqual(
            payload['args'][payload['args'].index('--color') + 1],
            'never',
        )
        self.assertNotIn('--model', payload['args'])
        self.assertNotIn('hello\nworld', payload['args'])
        self.assertFalse(Path(payload['cwd']).exists())

    def test_execution_stages_native_image_and_regular_file_with_handoff_truth(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        capture = self.root / 'execution-with-inputs.json'
        image_path = self.root / 'private-photo.png'
        image_bytes = b'\x89PNG\r\n\x1a\nprivate-image-bytes'
        image_path.write_bytes(image_bytes)
        document_path = self.root / 'AGENTS.md'
        document_bytes = b'user-provided reference text\n'
        document_path.write_bytes(document_bytes)

        result = execute_codex_request(
            'Inspect the selected files.',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=self._base_env(FAKE_CODEX_EXEC_CAPTURE=str(capture)),
            inputs=[
                CodexExecutionInput(
                    path=image_path,
                    display_name='holiday.png',
                    kind='image',
                    source='input_artifact',
                    artifact_ref='artifact:image:1',
                ),
                CodexExecutionInput(
                    path=document_path,
                    display_name='AGENTS.md',
                    kind='text',
                    source='selected_artifact',
                    artifact_ref='artifact:text:2',
                ),
            ],
        )

        self.assertEqual(result.status, CodexExecutionState.COMPLETED)
        self.assertEqual(len(result.input_handoff), 2)
        image_handoff, document_handoff = result.input_handoff
        self.assertEqual(image_handoff.name, 'holiday.png')
        self.assertEqual(image_handoff.kind, 'image')
        self.assertEqual(image_handoff.byte_size, len(image_bytes))
        self.assertEqual(image_handoff.sha256, hashlib.sha256(image_bytes).hexdigest())
        self.assertEqual(image_handoff.source, 'input_artifact')
        self.assertEqual(image_handoff.artifact_ref, 'artifact:image:1')
        self.assertTrue(image_handoff.native_image)
        self.assertEqual(document_handoff.name, 'AGENTS.md')
        self.assertEqual(document_handoff.kind, 'text')
        self.assertEqual(document_handoff.byte_size, len(document_bytes))
        self.assertEqual(
            document_handoff.sha256,
            hashlib.sha256(document_bytes).hexdigest(),
        )
        self.assertEqual(document_handoff.source, 'selected_artifact')
        self.assertEqual(document_handoff.artifact_ref, 'artifact:text:2')
        self.assertFalse(document_handoff.native_image)

        payload = json.loads(capture.read_text(encoding='utf-8'))
        self.assertEqual(payload['entries_before_output'], ['inputs'])
        self.assertEqual(
            payload['staged_files'],
            [
                {
                    'path': 'inputs/input-01.png',
                    'byte_size': len(image_bytes),
                    'sha256': hashlib.sha256(image_bytes).hexdigest(),
                },
                {
                    'path': 'inputs/input-02.md',
                    'byte_size': len(document_bytes),
                    'sha256': hashlib.sha256(document_bytes).hexdigest(),
                },
            ],
        )
        image_flag = payload['args'].index('--image')
        self.assertEqual(payload['args'][image_flag + 1], 'inputs/input-01.png')
        self.assertNotIn(str(image_path), payload['args'])
        self.assertNotIn(str(document_path), payload['args'])
        self.assertIn('inputs/input-01.png', payload['prompt'])
        self.assertIn('inputs/input-02.md', payload['prompt'])
        self.assertNotIn(str(image_path), payload['prompt'])
        self.assertNotIn(str(document_path), payload['prompt'])
        self.assertNotIn('inputs/AGENTS.md', payload['prompt'])
        self.assertFalse(Path(payload['cwd']).exists())

    def test_execution_rejects_symlink_and_file_limit_before_codex_exec(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        calls = self.root / 'calls.jsonl'
        target = self.root / 'target.txt'
        target.write_text('target', encoding='utf-8')
        symlink = self.root / 'selected-link.txt'
        symlink.symlink_to(target)
        oversized = self.root / 'oversized.txt'
        oversized.write_bytes(b'123')
        env = self._base_env(FAKE_CODEX_CALL_LOG=str(calls))

        symlink_result = execute_codex_request(
            'Inspect the selected file.',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=env,
            inputs=[CodexExecutionInput(path=symlink)],
        )
        oversized_result = execute_codex_request(
            'Inspect the selected file.',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=env,
            inputs=[CodexExecutionInput(path=oversized)],
            max_input_bytes=2,
        )

        self.assertEqual(symlink_result.status, CodexExecutionState.INVALID_REQUEST)
        self.assertEqual(symlink_result.diagnostic, 'codex_input_symlink_rejected')
        self.assertEqual(oversized_result.status, CodexExecutionState.INVALID_REQUEST)
        self.assertEqual(oversized_result.diagnostic, 'codex_input_file_too_large')
        call_rows = [
            json.loads(line)
            for line in calls.read_text(encoding='utf-8').splitlines()
        ]
        self.assertNotIn('exec', {row[0] for row in call_rows if row})

    def test_success_allows_bounded_stderr_warning(self) -> None:
        executable = self._write_codex(self.root / 'codex')

        result = execute_codex_text(
            'hello',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=self._base_env(FAKE_CODEX_MODE='exec_warning'),
        )

        self.assertEqual(result.status, CodexExecutionState.COMPLETED)
        self.assertEqual(result.output_text, 'final answer')
        self.assertIn('nonterminal warning', result.diagnostic or '')

    def test_execution_reports_empty_and_oversized_final_output(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        discovery = self._discovery(executable)

        empty = execute_codex_text(
            'hello',
            timeout_seconds=2,
            discovery=discovery,
            env=self._base_env(FAKE_CODEX_MODE='exec_empty'),
        )
        oversized = execute_codex_text(
            'hello',
            timeout_seconds=2,
            discovery=discovery,
            env=self._base_env(FAKE_CODEX_MODE='exec_large'),
            max_output_bytes=32,
        )

        self.assertEqual(empty.status, CodexExecutionState.EMPTY_OUTPUT)
        self.assertIsNone(empty.output_text)
        self.assertEqual(
            oversized.status,
            CodexExecutionState.OUTPUT_LIMIT_EXCEEDED,
        )
        self.assertTrue(oversized.output_truncated)
        self.assertIsNone(oversized.output_text)

    def test_execution_failure_redacts_credentials_and_bounds_diagnostics(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        secret = 'sk-super-secret-value'

        failed = execute_codex_text(
            'hello',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=self._base_env(
                FAKE_CODEX_MODE='exec_fail',
                OPENAI_API_KEY=secret,
            ),
        )
        loud = execute_codex_text(
            'hello',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=self._base_env(FAKE_CODEX_MODE='exec_loud_fail'),
            max_diagnostic_bytes=100,
        )

        self.assertEqual(failed.status, CodexExecutionState.FAILED)
        self.assertEqual(failed.exit_code, 7)
        self.assertNotIn(secret, failed.diagnostic or '')
        self.assertIn('[redacted]', failed.diagnostic or '')
        self.assertEqual(loud.status, CodexExecutionState.FAILED)
        self.assertTrue(loud.diagnostic_truncated)
        self.assertIn('[diagnostic truncated]', loud.diagnostic or '')

    def test_execution_timeout_is_terminal_and_has_no_output(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        discovery = self._discovery(executable)
        base_env = self._base_env()
        access = probe_codex_access(discovery, env=base_env)
        self.assertEqual(access.status, CodexAccessState.AVAILABLE)

        result = execute_codex_text(
            'hello',
            timeout_seconds=0.05,
            discovery=discovery,
            env={**base_env, 'FAKE_CODEX_MODE': 'exec_timeout'},
        )

        self.assertEqual(result.status, CodexExecutionState.TIMED_OUT)
        self.assertIsNone(result.output_text)
        self.assertLess(result.duration_seconds, 0.8)

    def test_execution_refuses_cloud_call_when_auth_is_missing(self) -> None:
        executable = self._write_codex(self.root / 'codex')
        calls = self.root / 'calls.jsonl'

        result = execute_codex_text(
            'must not execute',
            timeout_seconds=2,
            discovery=self._discovery(executable),
            env=self._base_env(
                FAKE_CODEX_MODE='auth_required',
                FAKE_CODEX_CALL_LOG=str(calls),
            ),
        )

        self.assertEqual(result.status, CodexExecutionState.AUTH_REQUIRED)
        call_rows = [
            json.loads(line)
            for line in calls.read_text(encoding='utf-8').splitlines()
        ]
        self.assertEqual(call_rows, [['login', 'status']])

    def test_execution_rejects_invalid_prompt_and_limits_before_spawn(self) -> None:
        missing = CodexDiscovery(available=False)

        empty = execute_codex_text(
            ' \r\n ',
            timeout_seconds=1,
            discovery=missing,
            env={},
        )
        invalid_timeout = execute_codex_text(
            'hello',
            timeout_seconds=0,
            discovery=missing,
            env={},
        )

        self.assertEqual(empty.status, CodexExecutionState.INVALID_REQUEST)
        self.assertEqual(
            invalid_timeout.status,
            CodexExecutionState.INVALID_REQUEST,
        )


if __name__ == '__main__':
    unittest.main()
