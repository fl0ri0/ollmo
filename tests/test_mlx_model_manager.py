import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollmo_runtime import mlx_model_manager


class MlxModelManagerTests(unittest.TestCase):
    def test_build_instance_record_adds_mlx_lm_contract_metadata(self):
        record = mlx_model_manager.build_instance_record(
            "mlx-community/Apertus-8B-Instruct-2509-bf16",
            "/tmp/apertus",
            11501,
            Path("/tmp/mlx_11501.log"),
            capability="chat",
            pid=123,
        )

        self.assertEqual(record["backend_package"], "mlx_lm")
        self.assertEqual(record["backend_contract"], "mlx_lm.server")
        self.assertEqual(record["provider_capabilities"], ["chat"])
        self.assertEqual(record["backend_metadata"]["source"], "mlx_package_contract")
        self.assertEqual(record["backend_metadata"]["runtime_knobs"], ["prefill_step_size", "prompt_cache_size", "prompt_cache_bytes"])
        self.assertEqual(record["backend_metadata"]["launch_defaults"]["prefill_step_size"], 2048)
        self.assertIn("rotating_kv_cache", record["backend_metadata"]["cache_features"])

    def test_build_instance_record_adds_mlx_vlm_contract_metadata(self):
        record = mlx_model_manager.build_instance_record(
            "mlx-community/DeepSeek-OCR-2-bf16",
            "/tmp/deepseek-ocr",
            11502,
            Path("/tmp/mlx_vlm_11502.log"),
            capability="vision_analysis",
            pid=456,
        )

        self.assertEqual(record["backend_package"], "mlx_vlm")
        self.assertEqual(record["backend_contract"], "mlx_vlm.server")
        self.assertEqual(record["provider_capabilities"], ["vision_analysis"])
        self.assertTrue(record["backend_metadata"]["lazy_loads_model"])
        self.assertTrue(record["backend_metadata"]["supports_unload"])
        self.assertIn("/v1/chat/completions", record["backend_metadata"]["native_endpoint_paths"])
        self.assertIn("kv_bits", record["backend_metadata"]["runtime_knobs"])
        self.assertEqual(record["backend_metadata"]["launch_defaults"]["kv_quant_scheme"], "uniform")
        self.assertEqual(record["backend_metadata"]["instance_capabilities"], ["vision_analysis"])
        self.assertEqual(record["backend_metadata"]["package_capabilities"], ["vision_analysis"])

    def test_build_instance_record_uses_conservative_vlm_defaults_for_gemma4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "gemma4"}',
                encoding="utf-8",
            )

            record = mlx_model_manager.build_instance_record(
                "mlx-community/gemma-4-31b-it-bf16",
                str(model_dir),
                11512,
                Path("/tmp/mlx_vlm_11512.log"),
                capability="vision_analysis",
                pid=654,
            )

        self.assertEqual(record["backend_package"], "mlx_vlm")
        self.assertEqual(record["backend_contract"], "mlx_vlm.server")
        self.assertEqual(record["backend_metadata"]["launch_defaults"], {})
        self.assertIn("kv_quant_scheme", record["backend_metadata"]["runtime_knobs"])

    def test_build_instance_record_marks_legacy_gemma4_mlx_vlm_conversion_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "gemma4"}',
                encoding="utf-8",
            )
            (model_dir / "README.md").write_text(
                "# mlx-community/gemma-4-e4b-8bit\n\n"
                "This model was converted to MLX format from `google/gemma-4-e4b` "
                "using mlx-vlm version **0.4.3**.\n",
                encoding="utf-8",
            )

            with patch.object(
                mlx_model_manager,
                "_mlx_package_runtime_check",
                return_value={"package_version": "0.6.2"},
            ):
                record = mlx_model_manager.build_instance_record(
                    "mlx-community/gemma-4-e4b-8bit",
                    str(model_dir),
                    11512,
                    Path("/tmp/mlx_vlm_11512.log"),
                    capability="vision_analysis",
                    pid=654,
                )

        metadata = record["backend_metadata"]
        self.assertEqual(record["snapshot_mlx_vlm_conversion_version"], "0.4.3")
        self.assertEqual(metadata["package_version"], "0.6.2")
        self.assertEqual(metadata["snapshot_mlx_vlm_conversion_version"], "0.4.3")
        self.assertIn(
            "gemma4_snapshot_converted_with_pre_0_6_mlx_vlm_runtime_0_6_or_newer",
            metadata["compatibility_warnings"],
        )
        self.assertIn("snapshot_runtime_compatibility_advisory", metadata["runtime_constraints"])
        self.assertEqual(record["provider_capabilities"], ["chat", "vision_analysis"])

    def test_build_instance_record_adds_mlx_audio_contract_metadata(self):
        record = mlx_model_manager.build_instance_record(
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "/tmp/qwen3-tts",
            11503,
            Path("/tmp/mlx_audio_11503.log"),
            capability="text_to_speech",
            pid=789,
        )

        self.assertEqual(record["backend_package"], "mlx_audio")
        self.assertEqual(record["backend_contract"], "mlx_audio.server")
        self.assertEqual(record["provider_capabilities"], ["text_to_speech"])
        self.assertEqual(record["request_model"], "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        self.assertTrue(record["backend_metadata"]["lazy_loads_model"])
        self.assertIn("speech_to_speech", record["backend_metadata"]["package_capabilities"])
        self.assertIn("/v1/audio/speech", record["backend_metadata"]["native_endpoint_paths"])

    def test_build_instance_record_adds_whisper_shim_contract_metadata(self):
        record = mlx_model_manager.build_instance_record(
            "mlx-community/whisper-large-v3-mlx",
            "/tmp/whisper",
            11504,
            Path("/tmp/mlx_whisper_11504.log"),
            capability="speech_to_text",
            pid=999,
        )

        self.assertEqual(record["backend_package"], "mlx_whisper_shim")
        self.assertEqual(record["backend_contract"], "ollmo.scripts.mlx_whisper_server")
        self.assertEqual(record["provider_capabilities"], ["speech_to_text"])
        self.assertEqual(record["backend_metadata"]["shim_kind"], "local_http_compatibility_shim")
        self.assertIn("/v1/audio/transcriptions", record["backend_metadata"]["native_endpoint_paths"])

    def test_build_instance_record_uses_mlx_audio_contract_for_non_whisper_stt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "voxtral_realtime"}',
                encoding="utf-8",
            )
            (model_dir / "README.md").write_text(
                '---\n'
                'tags:\n'
                '  - automatic-speech-recognition\n'
                'pipeline_tag: automatic-speech-recognition\n'
                '---\n',
                encoding="utf-8",
            )
            record = mlx_model_manager.build_instance_record(
                "mlx-community/Voxtral-Realtime-bf16",
                str(model_dir),
                11505,
                Path("/tmp/mlx_audio_11505.log"),
                capability="speech_to_text",
                pid=1001,
            )

        self.assertEqual(record["backend_package"], "mlx_audio")
        self.assertEqual(record["backend_contract"], "mlx_audio.server")
        self.assertEqual(record["mlx_server"], "mlx_audio")
        self.assertEqual(record["provider_capabilities"], ["speech_to_text"])
        self.assertEqual(record["request_model"], "mlx-community/Voxtral-Realtime-bf16")

    def test_build_instance_record_corrects_stale_tts_capability_from_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "qwen3_tts"}',
                encoding="utf-8",
            )
            (model_dir / "README.md").write_text(
                '---\n'
                'tags:\n'
                '  - text-to-speech\n'
                'pipeline_tag: text-to-speech\n'
                '---\n',
                encoding="utf-8",
            )
            record = mlx_model_manager.build_instance_record(
                "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                str(model_dir),
                11506,
                Path("/tmp/mlx_audio_11506.log"),
                capability="speech_to_text",
                pid=1002,
            )

        self.assertEqual(record["capability"], "text_to_speech")
        self.assertEqual(record["backend_package"], "mlx_audio")
        self.assertEqual(record["backend_contract"], "mlx_audio.server")
        self.assertEqual(record["provider_capabilities"], ["text_to_speech"])

    def test_build_instance_record_uses_mlx_vlm_for_vision_snapshot_without_explicit_capability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "qwen3_5", "image_token_id": 248056}',
                encoding="utf-8",
            )
            (model_dir / "README.md").write_text(
                '---\n'
                'tags:\n'
                '  - mlx\n'
                '  - vision-language-model\n'
                '---\n',
                encoding="utf-8",
            )
            record = mlx_model_manager.build_instance_record(
                "mlx-community/Qwen3.5-9B-MLX-4bit",
                str(model_dir),
                11507,
                Path("/tmp/mlx_vlm_11507.log"),
                capability=None,
                pid=1003,
            )

        self.assertEqual(record["capability"], "vision_analysis")
        self.assertEqual(record["backend_package"], "mlx_vlm")
        self.assertEqual(record["backend_contract"], "mlx_vlm.server")
        self.assertEqual(record["mlx_server"], "mlx_vlm")
        self.assertEqual(record["provider_capabilities"], ["chat", "vision_analysis"])
        self.assertEqual(record["backend_metadata"]["instance_capabilities"], ["chat", "vision_analysis"])

    @patch("ollmo_runtime.mlx_model_manager._mlx_package_runtime_check")
    def test_mlx_discovery_details_marks_snapshot_cached_only_when_runtime_missing(self, mock_runtime_check):
        mock_runtime_check.return_value = {
            "required_python_module": "mlx_vlm",
            "required_runtime_module": "mlx_vlm",
            "python_path": None,
            "python_resolved": False,
            "runtime_module_available": False,
            "python_error": "missing mlx_vlm",
        }

        details = mlx_model_manager._mlx_discovery_details(
            "mlx-community/DeepSeek-OCR-2-bf16",
            "vision_analysis",
            config_present=True,
        )

        self.assertFalse(details["runnable"])
        self.assertEqual(details["discovery_state"], "cached_only")
        self.assertIn("mlx_vlm", details["disabled_reason"])
        self.assertFalse(details["runnable_checks"]["python_resolved"])

    @patch("ollmo_runtime.mlx_model_manager._mlx_package_runtime_check")
    def test_mlx_discovery_details_requires_whisper_shim_script(self, mock_runtime_check):
        mock_runtime_check.return_value = {
            "required_python_module": "mlx_lm",
            "required_runtime_module": "mlx_whisper",
            "python_path": "/opt/mlx/venv/bin/python",
            "python_resolved": True,
            "runtime_module_available": True,
            "server_script_path": "/missing/mlx_whisper_server.py",
            "server_script_present": False,
        }

        details = mlx_model_manager._mlx_discovery_details(
            "mlx-community/whisper-large-v3-mlx",
            "speech_to_text",
            config_present=True,
        )

        self.assertFalse(details["runnable"])
        self.assertEqual(details["discovery_state"], "cached_only")
        self.assertIn("Whisper shim script", details["disabled_reason"])
        self.assertFalse(details["runnable_checks"]["server_script_present"])

    @patch("ollmo_runtime.mlx_model_manager._mlx_package_runtime_check")
    def test_mlx_discovery_details_rejects_broken_whisper_runtime_dependency(self, mock_runtime_check):
        mock_runtime_check.return_value = {
            "required_python_module": "mlx_whisper",
            "required_runtime_module": "mlx_whisper",
            "python_path": "/opt/mlx/venv/bin/python",
            "python_resolved": True,
            "runtime_module_available": True,
            "runtime_dependencies_ready": False,
            "runtime_dependency_error": "ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.",
            "server_script_path": str(mlx_model_manager.SPEECH_SERVER_SCRIPT),
            "server_script_present": True,
        }

        details = mlx_model_manager._mlx_discovery_details(
            "mlx-community/whisper-large-v3-mlx",
            "speech_to_text",
            config_present=True,
        )

        self.assertFalse(details["runnable"])
        self.assertEqual(details["discovery_state"], "cached_only")
        self.assertIn("Numba needs NumPy", details["disabled_reason"])
        self.assertFalse(details["runnable_checks"]["runtime_dependencies_ready"])

    def test_whisper_runtime_uses_its_own_distribution_name(self):
        self.assertEqual(
            mlx_model_manager._mlx_distribution_name("mlx_whisper"),
            "mlx-whisper",
        )

    def test_whisper_runtime_check_preflights_numba_without_importing_mlx(self):
        mlx_model_manager._mlx_package_runtime_check.cache_clear()
        try:
            with (
                patch.object(
                    mlx_model_manager,
                    "resolve_mlx_python",
                    return_value="/opt/mlx/venv/bin/python",
                ) as mock_resolve_python,
                patch.object(
                    mlx_model_manager,
                    "_python_module_available",
                    return_value=True,
                ) as mock_module_available,
                patch.object(
                    mlx_model_manager,
                    "_python_package_version",
                    return_value="0.4.3",
                ) as mock_package_version,
                patch.object(
                    mlx_model_manager,
                    "_python_runtime_dependency_import_check",
                    return_value={
                        "module": "numba",
                        "ready": False,
                        "error": "ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.",
                    },
                ) as mock_dependency_check,
            ):
                checks = mlx_model_manager._mlx_package_runtime_check("mlx_whisper")
        finally:
            mlx_model_manager._mlx_package_runtime_check.cache_clear()

        mock_resolve_python.assert_called_once_with("mlx_whisper")
        mock_module_available.assert_called_once_with(
            "/opt/mlx/venv/bin/python",
            "mlx_whisper",
        )
        mock_package_version.assert_called_once_with(
            "/opt/mlx/venv/bin/python",
            "mlx-whisper",
        )
        mock_dependency_check.assert_called_once_with(
            "/opt/mlx/venv/bin/python",
            "numba",
        )
        self.assertFalse(checks["runtime_dependencies_ready"])
        self.assertIn("Numba needs NumPy", checks["runtime_dependency_error"])

    @patch("ollmo_runtime.mlx_model_manager._mlx_package_runtime_check")
    def test_runtime_variants_mark_broken_whisper_dependency_degraded(self, mock_runtime_check):
        def checks_for(server_kind):
            base = {
                "python_resolved": True,
                "runtime_module_available": True,
                "runtime_dependencies_ready": True,
            }
            if server_kind == "mlx_whisper":
                return {
                    **base,
                    "runtime_dependencies_ready": False,
                    "runtime_dependency_error": "ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.",
                    "server_script_present": True,
                }
            return base

        mock_runtime_check.side_effect = checks_for

        variants = mlx_model_manager.describe_mlx_runtime_variants()

        self.assertEqual(variants["mlx_whisper"]["runtime_state"], "degraded")
        self.assertFalse(variants["mlx_whisper"]["operations"]["start_instance"])
        self.assertTrue(
            any("Numba needs NumPy" in issue for issue in variants["mlx_whisper"]["issues"])
        )

    @patch("ollmo_runtime.mlx_model_manager._python_module_available")
    @patch("ollmo_runtime.mlx_model_manager._mlx_package_runtime_check")
    def test_mlx_discovery_details_marks_model_cached_only_when_server_kind_lacks_model_support(
        self,
        mock_runtime_check,
        mock_python_module_available,
    ):
        mock_runtime_check.return_value = {
            "required_python_module": "mlx_lm",
            "required_runtime_module": "mlx_lm",
            "python_path": "/opt/mlx/venv/bin/python",
            "python_resolved": True,
            "runtime_module_available": True,
        }
        mock_python_module_available.return_value = False

        details = mlx_model_manager._mlx_discovery_details(
            "nvidia/Gemma-4-31B-IT-NVFP4",
            "chat",
            config_present=True,
            metadata={"snapshot_model_type": "gemma4"},
        )

        self.assertFalse(details["runnable"])
        self.assertEqual(details["discovery_state"], "cached_only")
        self.assertIn("Model type 'gemma4' is not supported by mlx_lm", details["disabled_reason"])
        self.assertEqual(details["runnable_checks"]["model_runtime_module"], "mlx_lm.models.gemma4")
        self.assertFalse(details["runnable_checks"]["model_runtime_module_available"])

    def test_resolve_hf_cli_prefers_mlx_venv_sibling_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir(parents=True)
            python_path = bin_dir / "python"
            hf_path = bin_dir / "hf"
            python_path.write_text("", encoding="utf-8")
            hf_path.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(python_path, 0o755)
            os.chmod(hf_path, 0o755)

            with patch.object(mlx_model_manager, "ENV_MLX_PYTHON", str(python_path)):
                with patch.object(mlx_model_manager, "DEFAULT_MLX_PYTHON", Path("/nonexistent/python")):
                    with patch("shutil.which", return_value=None):
                        resolved = mlx_model_manager.resolve_hf_cli()

        self.assertEqual(Path(resolved), hf_path)

    @patch("ollmo_runtime.mlx_model_manager.infer_capability", return_value="chat")
    @patch("ollmo_runtime.mlx_model_manager.read_snapshot_model_metadata", return_value={})
    @patch("ollmo_runtime.mlx_model_manager._mlx_contract_details", return_value={})
    @patch(
        "ollmo_runtime.mlx_model_manager._mlx_discovery_details",
        return_value={"runnable": False, "disabled_reason": "cached", "discovery_state": "cached_only"},
    )
    def test_find_hf_cached_models_skips_llama_cpp_gguf_snapshots_but_keeps_other_hf_cache_entries(
        self,
        _mock_discovery_details,
        _mock_contract_details,
        _mock_snapshot_metadata,
        _mock_infer_capability,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            gguf_snapshot = cache_dir / "models--ggml-org--gemma-4-E4B-it-GGUF" / "snapshots" / "abc123"
            gguf_snapshot.mkdir(parents=True)
            (gguf_snapshot / "gemma-4-e4b-it-q4_k_m.gguf").write_bytes(b"gguf")

            starflow_snapshot = cache_dir / "models--apple--starflow" / "snapshots" / "def456"
            starflow_snapshot.mkdir(parents=True)
            (starflow_snapshot / "config.json").write_text("{}", encoding="utf-8")
            (starflow_snapshot / "weights.safetensors").write_bytes(b"mlx")

            with patch.object(mlx_model_manager, "CACHE_DIR", cache_dir):
                models = mlx_model_manager.find_hf_cached_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["repo"], "apple/starflow")

    def test_resolve_mlx_python_prefers_audio_specific_env_for_mlx_audio_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir(parents=True)
            audio_python = bin_dir / "python"
            audio_python.write_text("", encoding="utf-8")
            os.chmod(audio_python, 0o755)

            with patch.object(mlx_model_manager, "ENV_MLX_AUDIO_PYTHON", str(audio_python)):
                with patch.object(mlx_model_manager, "ENV_MLX_PYTHON", None):
                    with patch.object(mlx_model_manager, "DEFAULT_MLX_PYTHON", Path("/nonexistent/python")):
                        with patch.object(mlx_model_manager, "_python_module_available", return_value=True) as mock_probe:
                            resolved = mlx_model_manager.resolve_mlx_python("mlx_audio.server")

        self.assertEqual(Path(resolved), audio_python)
        mock_probe.assert_called_once_with(str(audio_python), "mlx_audio.server")

    def test_python_module_available_uses_non_import_probe(self):
        captured = {}

        def fake_run(cmd, **_kwargs):
            captured["cmd"] = cmd

            class Result:
                returncode = 0

            return Result()

        with patch("subprocess.run", side_effect=fake_run):
            mlx_model_manager._python_module_available.cache_clear()
            available = mlx_model_manager._python_module_available("/tmp/python", "mlx_lm")

        self.assertTrue(available)
        self.assertEqual(captured["cmd"][0], "/tmp/python")
        self.assertEqual(captured["cmd"][1], "-c")
        self.assertEqual(captured["cmd"][3], "mlx_lm")
        self.assertIn("module_exists", captured["cmd"][2])
        self.assertNotIn("import mlx_lm", captured["cmd"][2])

    @patch("ollmo_runtime.mlx_model_manager.schedule_recent_mlx_instance_reconciliation")
    @patch("ollmo_runtime.mlx_model_manager.register_mlx_instance")
    @patch("ollmo_runtime.mlx_model_manager.launch")
    @patch("ollmo_runtime.mlx_model_manager.wait_for_port")
    @patch("ollmo_runtime.mlx_model_manager.port_in_use", return_value=False)
    @patch("ollmo_runtime.mlx_model_manager.find_mlx_snapshots")
    @patch("ollmo_runtime.mlx_model_manager.prune_stale_mlx_entries")
    def test_start_mlx_model_registers_and_schedules_monitor(
        self,
        _mock_prune,
        mock_find_snapshots,
        _mock_port_in_use,
        mock_wait_for_port,
        mock_launch,
        mock_register,
        mock_schedule,
    ):
        mock_find_snapshots.return_value = [
            {
                "repo": "mlx-community/Qwen3.5-27B-4bit",
                "path": "/tmp/qwen35",
            }
        ]
        mock_wait_for_port.return_value = True
        mock_launch.return_value = (Path("/tmp/mlx_11506.log"), 43210)

        instance = mlx_model_manager.start_mlx_model(
            "mlx-community/Qwen3.5-27B-4bit",
            preferred_port=11506,
            capability="chat",
            start_source="api_start_model",
        )

        self.assertEqual(instance["instance_id"], "mlx-community__Qwen3.5-27B-4bit-mlx-11506")
        self.assertEqual(instance["pid"], 43210)
        mock_register.assert_called_once_with(instance)
        mock_schedule.assert_called_once_with(instance)
        mock_launch.assert_called_once_with("/tmp/qwen35", 11506, launch_defaults={
            "prefill_step_size": 2048,
            "prompt_cache_size": 10,
        })

    @patch("ollmo_runtime.mlx_model_manager.schedule_recent_mlx_instance_reconciliation")
    @patch("ollmo_runtime.mlx_model_manager.register_mlx_instance")
    @patch("ollmo_runtime.mlx_model_manager.launch_vlm_server")
    @patch("ollmo_runtime.mlx_model_manager.wait_for_port")
    @patch("ollmo_runtime.mlx_model_manager.port_in_use", return_value=False)
    @patch("ollmo_runtime.mlx_model_manager.find_mlx_snapshots")
    @patch("ollmo_runtime.mlx_model_manager.prune_stale_mlx_entries")
    def test_start_mlx_model_prefers_snapshot_vision_capability_over_chat_guess(
        self,
        _mock_prune,
        mock_find_snapshots,
        _mock_port_in_use,
        mock_wait_for_port,
        mock_launch_vlm_server,
        mock_register,
        mock_schedule,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "qwen3_5", "image_token_id": 248056}',
                encoding="utf-8",
            )
            (model_dir / "README.md").write_text(
                '---\n'
                'tags:\n'
                '  - mlx\n'
                '  - vision-language-model\n'
                '---\n',
                encoding="utf-8",
            )
            mock_find_snapshots.return_value = [
                {
                    "repo": "mlx-community/Qwen3.5-9B-MLX-4bit",
                    "path": str(model_dir),
                }
            ]
            mock_wait_for_port.return_value = True
            mock_launch_vlm_server.return_value = (Path("/tmp/mlx_vlm_11507.log"), 54321)

            instance = mlx_model_manager.start_mlx_model(
                "mlx-community/Qwen3.5-9B-MLX-4bit",
                preferred_port=11507,
                capability="chat",
                start_source="api_start_model",
            )

        self.assertEqual(instance["capability"], "vision_analysis")
        self.assertEqual(instance["backend_package"], "mlx_vlm")
        self.assertEqual(instance["backend_contract"], "mlx_vlm.server")
        self.assertEqual(instance["mlx_server"], "mlx_vlm")
        mock_register.assert_called_once_with(instance)
        mock_schedule.assert_called_once_with(instance)
        mock_launch_vlm_server.assert_called_once_with(11507, launch_defaults={
            "prefill_step_size": 2048,
            "kv_quant_scheme": "uniform",
            "kv_group_size": 64,
            "quantized_kv_start": 5000,
        })

    @patch("ollmo_runtime.mlx_model_manager.schedule_recent_mlx_instance_reconciliation")
    @patch("ollmo_runtime.mlx_model_manager.register_mlx_instance")
    @patch("ollmo_runtime.mlx_model_manager.launch_vlm_server")
    @patch("ollmo_runtime.mlx_model_manager.wait_for_port")
    @patch("ollmo_runtime.mlx_model_manager.port_in_use", return_value=False)
    @patch("ollmo_runtime.mlx_model_manager.find_mlx_snapshots")
    @patch("ollmo_runtime.mlx_model_manager.prune_stale_mlx_entries")
    def test_start_mlx_model_uses_conservative_vlm_defaults_for_gemma4(
        self,
        _mock_prune,
        mock_find_snapshots,
        _mock_port_in_use,
        mock_wait_for_port,
        mock_launch_vlm_server,
        mock_register,
        mock_schedule,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                '{"model_type": "gemma4"}',
                encoding="utf-8",
            )
            mock_find_snapshots.return_value = [
                {
                    "repo": "mlx-community/gemma-4-31b-it-bf16",
                    "path": str(model_dir),
                }
            ]
            mock_wait_for_port.return_value = True
            mock_launch_vlm_server.return_value = (Path("/tmp/mlx_vlm_11508.log"), 65432)

            instance = mlx_model_manager.start_mlx_model(
                "mlx-community/gemma-4-31b-it-bf16",
                preferred_port=11508,
                capability="vision_analysis",
                start_source="api_start_model",
            )

        self.assertEqual(instance["capability"], "vision_analysis")
        self.assertEqual(instance["backend_package"], "mlx_vlm")
        mock_register.assert_called_once_with(instance)
        mock_schedule.assert_called_once_with(instance)
        mock_launch_vlm_server.assert_called_once_with(11508, launch_defaults={})

    @patch("ollmo_runtime.mlx_model_manager.resolve_mlx_python", return_value="/opt/mlx/venv/bin/python")
    @patch("ollmo_runtime.mlx_model_manager.subprocess.Popen")
    def test_launch_vlm_server_applies_turboquant_launch_defaults(self, mock_popen, _mock_python):
        process = unittest.mock.Mock()
        process.pid = 1234
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mlx_model_manager, "LOG_DIR", Path(tmpdir)):
                mlx_model_manager.launch_vlm_server(
                    11522,
                    launch_defaults={
                        "prefill_step_size": 1024,
                        "kv_bits": 3.0,
                        "kv_quant_scheme": "turboquant",
                        "kv_group_size": 32,
                        "max_kv_size": 8192,
                        "quantized_kv_start": 2048,
                    },
                )

        cmd = mock_popen.call_args.args[0]
        self.assertIn("--kv-bits", cmd)
        self.assertIn("3.0", cmd)
        self.assertIn("--kv-quant-scheme", cmd)
        self.assertIn("turboquant", cmd)
        self.assertIn("--max-kv-size", cmd)
        self.assertIn("8192", cmd)

    @patch("ollmo_runtime.mlx_model_manager.resolve_mlx_python", return_value="/opt/mlx/venv/bin/python")
    @patch("ollmo_runtime.mlx_model_manager.subprocess.Popen")
    def test_launch_vlm_server_archives_existing_log_before_launch(self, mock_popen, _mock_python):
        process = unittest.mock.Mock()
        process.pid = 5678
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            stale_log = log_dir / "mlx_vlm_11522.log"
            stale_log.write_text("OLD VLM FAILURE\n", encoding="utf-8")

            with patch.object(mlx_model_manager, "LOG_DIR", log_dir):
                mlx_model_manager.launch_vlm_server(
                    11522,
                    launch_defaults={
                        "prefill_step_size": 1024,
                    },
                )

            self.assertTrue(stale_log.exists())
            self.assertNotIn("OLD VLM FAILURE", stale_log.read_text(encoding="utf-8"))
            archived_logs = list((log_dir / "archive" / "runtime").rglob("*.log"))
            self.assertEqual(len(archived_logs), 1)
            self.assertIn("superseded_launch", archived_logs[0].name)
            self.assertIn("OLD VLM FAILURE", archived_logs[0].read_text(encoding="utf-8"))
            self.assertTrue((log_dir / "archive" / "runtime" / "manifest.jsonl").exists())

    @patch("ollmo_runtime.mlx_model_manager.resolve_mlx_python", return_value="/opt/mlx-audio/venv/bin/python")
    @patch("ollmo_runtime.mlx_model_manager.subprocess.Popen")
    def test_launch_audio_server_omits_removed_workers_argument(self, mock_popen, _mock_python):
        process = unittest.mock.Mock()
        process.pid = 2468
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mlx_model_manager, "LOG_DIR", Path(tmpdir)):
                mlx_model_manager.launch_audio_server(11503)

        cmd = mock_popen.call_args.args[0]
        self.assertEqual(cmd[:3], ["/opt/mlx-audio/venv/bin/python", "-m", "mlx_audio.server"])
        self.assertIn("--host", cmd)
        self.assertIn("127.0.0.1", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("11503", cmd)
        self.assertNotIn("--workers", cmd)

    @patch("ollmo_runtime.mlx_model_manager.resolve_mlx_python", return_value="/opt/mlx/venv/bin/python")
    @patch("ollmo_runtime.mlx_model_manager.subprocess.Popen")
    def test_launch_mlx_lm_applies_prompt_cache_launch_defaults(self, mock_popen, _mock_python):
        process = unittest.mock.Mock()
        process.pid = 4321
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(mlx_model_manager, "LOG_DIR", Path(tmpdir)):
                mlx_model_manager.launch(
                    "/tmp/model",
                    11523,
                    launch_defaults={
                        "prefill_step_size": 1024,
                        "prompt_cache_size": 4,
                        "prompt_cache_bytes": "512mb",
                    },
                )

        cmd = mock_popen.call_args.args[0]
        self.assertIn("--prefill-step-size", cmd)
        self.assertIn("--prompt-cache-size", cmd)
        self.assertIn("--prompt-cache-bytes", cmd)

    @patch("ollmo_runtime.mlx_model_manager.remove_instance_status")
    @patch("ollmo_runtime.mlx_model_manager.remove_mlx_instance")
    @patch("ollmo_runtime.mlx_model_manager.port_in_use", return_value=False)
    def test_reconcile_recent_mlx_instance_removes_dead_entry(
        self,
        _mock_port_in_use,
        mock_remove_instance,
        mock_remove_status,
    ):
        result = mlx_model_manager.reconcile_recent_mlx_instance(
            {
                "instance_id": "mlx-community__Qwen3.5-27B-4bit-mlx-11506",
                "port": 11506,
            },
            monitor_sec=0.0,
            poll_sec=0.0,
        )

        self.assertFalse(result)
        mock_remove_instance.assert_called_once_with("mlx-community__Qwen3.5-27B-4bit-mlx-11506")
        mock_remove_status.assert_called_once_with("mlx-community__Qwen3.5-27B-4bit-mlx-11506")

    @patch("ollmo_runtime.mlx_model_manager.remove_instance_status")
    @patch("ollmo_runtime.mlx_model_manager.remove_mlx_instance")
    @patch("ollmo_runtime.mlx_model_manager.port_in_use", return_value=True)
    def test_reconcile_recent_mlx_instance_keeps_live_entry(
        self,
        _mock_port_in_use,
        mock_remove_instance,
        mock_remove_status,
    ):
        result = mlx_model_manager.reconcile_recent_mlx_instance(
            {
                "instance_id": "mlx-community__Qwen3.5-27B-4bit-mlx-11506",
                "port": 11506,
            },
            monitor_sec=0.0,
            poll_sec=0.0,
        )

        self.assertTrue(result)
        mock_remove_instance.assert_not_called()
        mock_remove_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
