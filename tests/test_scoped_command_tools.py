import tempfile
import unittest
from pathlib import Path

from ollmo_services.scoped_command_tools import run_scoped_command


class ScopedCommandToolTests(unittest.TestCase):
    def test_runs_argv_command_inside_repo_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            result = run_scoped_command(
                ['python3', '-c', 'print("hello")'],
                cwd='.',
                repo_root=repo_root,
                allowed_cwd_roots=['.'],
                allowed_command_prefixes=[['python3', '-c']],
                timeout_sec=5,
            )

            self.assertEqual(result['returncode'], 0)
            self.assertFalse(result['timed_out'])
            self.assertEqual(result['stdout'].strip(), 'hello')
            self.assertEqual(Path(result['cwd']).resolve(), repo_root.resolve())

    def test_rejects_shell_string_and_disallowed_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            with self.assertRaises(TypeError):
                run_scoped_command('python3 -c print(1)', repo_root=repo_root)

            with self.assertRaises(ValueError):
                run_scoped_command(
                    ['python3', '-c', 'print(1)'],
                    repo_root=repo_root,
                    allowed_command_prefixes=[['rg']],
                )

    def test_rejects_cwd_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / 'repo'
            outside = Path(tmpdir) / 'outside'
            repo_root.mkdir()
            outside.mkdir()

            with self.assertRaises(ValueError):
                run_scoped_command(
                    ['python3', '-c', 'print(1)'],
                    cwd=outside,
                    repo_root=repo_root,
                    allowed_cwd_roots=['.'],
                    allowed_command_prefixes=[['python3', '-c']],
                )


if __name__ == '__main__':
    unittest.main()
