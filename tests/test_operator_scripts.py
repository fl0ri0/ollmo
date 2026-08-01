from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class OperatorScriptEntrypointTests(unittest.TestCase):
    def run_script(self, relative_path: str, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix='ollmo-script-home-') as home_dir:
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env.update(
                {
                    'HOME': home_dir,
                    'OLLMO_WEB_BASE': 'http://127.0.0.1:9',
                }
            )
            if extra_env:
                env.update(extra_env)
            return subprocess.run(
                [sys.executable, str(REPO_ROOT / relative_path)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def assert_no_import_failure(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        self.assertNotIn('ModuleNotFoundError', result.stderr)
        self.assertNotIn("No module named 'ollmo_integrations'", result.stderr)

    def test_sync_model_providers_script_runs_directly(self) -> None:
        result = self.run_script('scripts/sync_model_providers.py')

        self.assert_no_import_failure(result)
        self.assertIn('Codex config.toml', result.stdout)

    def test_cleanup_model_providers_script_runs_directly(self) -> None:
        result = self.run_script('scripts/cleanup_model_providers.py')

        self.assert_no_import_failure(result)
        self.assertIn('config.toml was not found', result.stdout)

    def test_unsync_model_providers_script_runs_directly(self) -> None:
        result = self.run_script('scripts/unsync_model_providers.py')

        self.assert_no_import_failure(result)
        self.assertIn('bereits unsynced', result.stdout)


if __name__ == '__main__':
    unittest.main()
