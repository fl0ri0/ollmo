import re
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import startup_model_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_OUTPUT_FILES = (
    REPO_ROOT / 'start_multi_models.sh',
    REPO_ROOT / 'stop_multi_models.sh',
    REPO_ROOT / 'restart.sh',
    REPO_ROOT / 'scripts' / 'startup_model_manager.py',
    REPO_ROOT / 'ollmo_runtime' / 'ollama_model_manager.py',
    REPO_ROOT / 'ollmo_runtime' / 'mlx_model_manager.py',
)
COMPOUND_STARTUP_ICONS = ('ℹ️', '⚠️', '▶️', '➡️')


class StartupModelManagerTests(unittest.TestCase):
    def test_compound_lifecycle_icons_have_terminal_safe_separator(self):
        single_space_prefix = re.compile(
            rf"(?:{'|'.join(re.escape(icon) for icon in COMPOUND_STARTUP_ICONS)}) (?! )"
        )
        offenders = []
        for path in LIFECYCLE_OUTPUT_FILES:
            for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if single_space_prefix.search(line):
                    offenders.append(f'{path.relative_to(REPO_ROOT)}:{line_number}')

        self.assertEqual(offenders, [], msg=f'single-space status prefixes: {offenders}')

    def test_image_category_badge_has_explicit_emoji_form_and_separator(self):
        self.assertEqual(
            startup_model_manager.format_capability_badge('image_generation'),
            '🖼️  Image',
        )

    @patch('builtins.input', side_effect=EOFError)
    def test_prompt_selection_treats_eof_as_no_selection(self, _mock_input):
        entries = [
            startup_model_manager.CatalogEntry(
                backend='mlx',
                model_name='model-1',
                display_label='Model 1',
                capability='chat',
                capability_badge='Chat',
                size='1 GB',
            )
        ]

        selected = startup_model_manager._prompt_selection(entries)

        self.assertEqual(selected, [])

    @patch('scripts.startup_model_manager._write_registry_once')
    @patch('scripts.startup_model_manager._discover_catalog', return_value=[])
    @patch(
        'scripts.startup_model_manager._read_active_runtime_entries',
        return_value=([], False),
    )
    def test_main_without_available_models_keeps_control_plane_startup_viable(
        self,
        _mock_read_active_runtime_entries,
        _mock_discover_catalog,
        mock_write_registry_once,
    ):
        result = startup_model_manager.main()

        self.assertEqual(result, 0)
        mock_write_registry_once.assert_not_called()

    @patch('scripts.startup_model_manager.describe_llama_cpp_runtime_probe')
    @patch('scripts.startup_model_manager.list_available_llama_cpp_models')
    def test_discover_llama_cpp_entries_skips_non_runnable_catalog_items(self, mock_list, mock_probe):
        mock_probe.return_value = {
            'runtime_state': 'runnable',
            'detection': {'server_bin': '/opt/homebrew/bin/llama-server'},
            'issues': [],
        }
        mock_list.return_value = [
            {
                'model': 'nvidia/Gemma-4-31B-IT-NVFP4',
                'backend': 'llama_cpp',
                'runnable': False,
                'disabled_reason': 'Hugging Face repo is cached, but no GGUF file or GGUF-backed repo contract is known yet for llama.cpp.',
            },
            {
                'model': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
                'backend': 'llama_cpp',
                'runnable': True,
                'capability': 'chat',
                'hf_repo': 'ggml-org/gemma-4-26B-A4B-it-GGUF',
            },
        ]

        entries = startup_model_manager._discover_llama_cpp_entries()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model_name, 'ggml-org/gemma-4-26B-A4B-it-GGUF')

    @patch('builtins.print')
    @patch('scripts.startup_model_manager.describe_llama_cpp_runtime_probe')
    def test_print_llama_cpp_runtime_hint_shows_server_binary(self, mock_probe, mock_print):
        mock_probe.return_value = {
            'runtime_state': 'runnable',
            'detection': {'server_bin': '/opt/homebrew/bin/llama-server'},
            'issues': [],
        }

        startup_model_manager._print_llama_cpp_runtime_hint()

        mock_print.assert_called_once_with('ℹ️  Using llama.cpp server: /opt/homebrew/bin/llama-server')

    @patch('scripts.startup_model_manager.cleanup_runtime_hygiene', return_value={})
    @patch('scripts.startup_model_manager._write_registry_once')
    @patch('scripts.startup_model_manager.start_llama_cpp_instance')
    @patch('scripts.startup_model_manager.start_model')
    @patch('scripts.startup_model_manager._prompt_selection')
    @patch('scripts.startup_model_manager._discover_catalog')
    @patch('scripts.startup_model_manager._read_active_runtime_entries')
    def test_main_does_not_fall_through_from_ollama_to_llama_cpp(
        self,
        mock_read_active_runtime_entries,
        mock_discover_catalog,
        mock_prompt_selection,
        mock_start_model,
        mock_start_llama_cpp_instance,
        mock_write_registry_once,
        _mock_cleanup_runtime_hygiene,
    ):
        ollama_entry = startup_model_manager.CatalogEntry(
            backend='ollama',
            model_name='gemma4:e4b',
            display_label='gemma4 e4b',
            capability='chat',
            capability_badge='💬 Chat',
            size='8 GB',
        )
        mock_read_active_runtime_entries.return_value = ([], False)
        mock_discover_catalog.return_value = [ollama_entry]
        mock_prompt_selection.return_value = [ollama_entry]
        mock_start_model.return_value = {
            'instance_id': 'gemma4:e4b-1',
            'model': 'gemma4:e4b',
            'backend': 'ollama',
            'capability': 'chat',
            'port': 11438,
        }

        result = startup_model_manager.main()

        self.assertEqual(result, 0)
        mock_start_model.assert_called_once()
        args, kwargs = mock_start_model.call_args
        self.assertEqual(args[0], 'gemma4:e4b')
        self.assertEqual(args[1], set())
        self.assertEqual(kwargs['capability'], 'chat')
        mock_start_llama_cpp_instance.assert_not_called()
        mock_write_registry_once.assert_called_once()


if __name__ == '__main__':
    unittest.main()
