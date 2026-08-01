import tempfile
import unittest
from pathlib import Path

from ollmo_services.scoped_file_tools import (
    copy_scoped_file,
    read_scoped_text,
    replace_scoped_text,
    resolve_scoped_path,
    write_scoped_text,
)


class ScopedFileToolTests(unittest.TestCase):
    def test_read_write_replace_and_copy_inside_allowed_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / 'state').mkdir()

            write_result = write_scoped_text(
                'state/memory/policy.md',
                'old policy',
                repo_root=repo_root,
            )
            text, read_result = read_scoped_text('state/memory/policy.md', repo_root=repo_root)
            replace_result = replace_scoped_text(
                'state/memory/policy.md',
                'old',
                'new',
                repo_root=repo_root,
            )
            copy_result = copy_scoped_file(
                'state/memory/policy.md',
                'state/memory/policy-copy.md',
                repo_root=repo_root,
            )

            self.assertTrue(write_result['created'])
            self.assertEqual(text, 'old policy')
            self.assertFalse(read_result['truncated'])
            self.assertEqual(replace_result['replacements'], 1)
            self.assertEqual((repo_root / 'state/memory/policy.md').read_text(encoding='utf-8'), 'new policy')
            self.assertEqual(copy_result['operation'], 'copy')
            self.assertEqual((repo_root / 'state/memory/policy-copy.md').read_text(encoding='utf-8'), 'new policy')

    def test_rejects_outside_default_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / 'state').mkdir()

            with self.assertRaises(ValueError):
                resolve_scoped_path('ollmo_webserver.py', repo_root=repo_root)

            with self.assertRaises(ValueError):
                write_scoped_text('../outside.txt', 'nope', repo_root=repo_root)

    def test_rejects_overwrite_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            target = repo_root / 'state/policy.json'
            target.parent.mkdir(parents=True)
            target.write_text('{}', encoding='utf-8')

            with self.assertRaises(FileExistsError):
                write_scoped_text('state/policy.json', '{"x":1}', repo_root=repo_root, overwrite=False)

            with self.assertRaises(FileExistsError):
                copy_scoped_file('state/policy.json', 'state/policy.json', repo_root=repo_root)


if __name__ == '__main__':
    unittest.main()
