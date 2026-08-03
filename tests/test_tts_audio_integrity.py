from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path

from ollmo_services.tts_audio_integrity import (
    TTS_AUDIO_INTEGRITY_POLICY_ID,
    TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE,
    build_tts_audio_integrity_evidence,
    tts_audio_has_qwen_generation_limit_exhaustion,
)


def _write_pcm_wav(
    path: Path,
    segments: list[tuple[float, bool]],
    *,
    sample_rate: int = 8000,
) -> Path:
    samples: list[int] = []
    phase = 0
    for duration, active in segments:
        frame_count = int(round(duration * sample_rate))
        for _ in range(frame_count):
            sample = (
                int(12000 * math.sin((2 * math.pi * 220 * phase) / sample_rate))
                if active
                else 0
            )
            samples.append(sample)
            phase += 1
    pcm = b''.join(sample.to_bytes(2, 'little', signed=True) for sample in samples)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return path


def _qwen_voice_design_budget(
    max_tokens: int,
    *,
    generation_scope: str = 'single_sequence',
    tts_model_type: str = 'voice_design',
) -> dict:
    return {
        'kind': 'ollmo.tts_generation_budget',
        'version': 1,
        'policy_id': 'qwen3_tts_adaptive_audio_tokens_v2',
        'model_family': 'qwen3_tts',
        'tts_model_type': tts_model_type,
        'generation_scope': generation_scope,
        'max_tokens': max_tokens,
        'policy': {'audio_tokens_per_second': 12.5},
    }


def test_healthy_speech_like_wav_passes_with_source_binding(tmp_path):
    source = 'Dies ist ein vollständiger deutscher Satz für den Audiotest.'
    source_sha256 = hashlib.sha256(source.encode('utf-8')).hexdigest()
    audio_path = _write_pcm_wav(
        tmp_path / 'healthy.wav',
        [(1.2, True), (0.25, False), (1.2, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        source,
        source_sha256=source_sha256,
    )

    assert evidence['kind'] == 'ollmo.tts_audio_integrity_evidence'
    assert evidence['policy_id'] == TTS_AUDIO_INTEGRITY_POLICY_ID
    assert evidence['status'] == 'passed'
    assert evidence['reason_code'] == 'TTS_AUDIO_INTEGRITY_PASSED'
    assert evidence['source_digest_match'] is True
    assert evidence['materialization_eligible'] is True
    assert evidence['effective_active_seconds'] >= 2.3
    assert evidence['silence_ratio'] < 0.2


def test_integrity_analysis_is_read_only_and_preserves_audio_bytes(tmp_path):
    source = 'The analyzer must inspect this audio without changing it.'
    audio_path = _write_pcm_wav(
        tmp_path / 'read-only.wav',
        [(1.4, True), (0.2, False), (1.4, True)],
    )
    original_bytes = audio_path.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    evidence = build_tts_audio_integrity_evidence(audio_path, source)

    analyzed_bytes = audio_path.read_bytes()
    assert analyzed_bytes == original_bytes
    assert hashlib.sha256(analyzed_bytes).hexdigest() == original_sha256
    assert evidence['artifact_sha256'] == original_sha256
    assert evidence['status'] == 'passed'


def test_long_source_with_short_signal_and_long_silence_fails(tmp_path):
    source = ' '.join(f'Wort{index}' for index in range(1, 81))
    audio_path = _write_pcm_wav(
        tmp_path / 'truncated.wav',
        [(6.0, True), (90.0, False)],
    )

    evidence = build_tts_audio_integrity_evidence(audio_path, source)

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT'
    assert evidence['nominal_duration_seconds'] == 96.0
    assert evidence['effective_active_seconds'] == 6.0
    assert evidence['trailing_silence_seconds'] == 90.0
    assert evidence['silence_ratio'] >= 0.93
    assert evidence['materialization_eligible'] is False


def test_regression_shape_with_96_second_padding_remains_blocked(tmp_path):
    source = ' '.join(f'Wort{index}' for index in range(1, 65))
    audio_path = _write_pcm_wav(
        tmp_path / 'regression-96s.wav',
        [(5.5, False), (2.5, True), (88.0, False)],
    )

    evidence = build_tts_audio_integrity_evidence(audio_path, source)

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT'
    assert evidence['source_word_count'] == 64
    assert evidence['minimum_expected_active_seconds'] == 12.8
    assert evidence['nominal_duration_seconds'] == 96.0
    assert evidence['effective_active_seconds'] == 2.5
    assert evidence['silence_seconds'] == 93.5
    assert evidence['trailing_silence_seconds'] == 88.0
    assert evidence['materialization_eligible'] is False


def test_short_source_with_extreme_trailing_silence_fails_padding_check(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'padded.wav',
        [(1.0, True), (10.0, False)],
    )

    evidence = build_tts_audio_integrity_evidence(audio_path, 'Hallo Welt.')

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_EXCESSIVE_TRAILING_SILENCE'
    assert evidence['effective_active_seconds'] == 1.0
    assert evidence['trailing_silence_seconds'] == 10.0


def test_all_silence_fails_even_when_wav_is_structurally_valid(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'silent.wav',
        [(2.0, False)],
    )

    evidence = build_tts_audio_integrity_evidence(audio_path, 'Sag etwas.')

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_NO_ACTIVE_SIGNAL'
    assert evidence['active_window_count'] == 0
    assert evidence['materialization_eligible'] is False


def test_natural_internal_pause_does_not_look_like_truncation(tmp_path):
    source = 'Eins zwei drei vier fünf sechs sieben acht.'
    audio_path = _write_pcm_wav(
        tmp_path / 'natural-pause.wav',
        [(0.9, True), (0.7, False), (0.9, True), (0.3, False)],
    )

    evidence = build_tts_audio_integrity_evidence(audio_path, source)

    assert evidence['status'] == 'passed'
    assert evidence['effective_active_seconds'] >= 1.7
    assert evidence['trailing_silence_seconds'] <= 0.4


def test_malformed_wav_fails_closed(tmp_path):
    audio_path = tmp_path / 'broken.wav'
    audio_path.write_bytes(b'RIFF-not-a-valid-wave')

    evidence = build_tts_audio_integrity_evidence(audio_path, 'Test.')

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_WAV_UNREADABLE'
    assert evidence['defect_codes'] == ['TTS_AUDIO_WAV_UNREADABLE']
    assert evidence['materialization_eligible'] is False


def test_missing_wav_is_unavailable_and_not_materialization_eligible(tmp_path):
    evidence = build_tts_audio_integrity_evidence(
        tmp_path / 'missing.wav',
        'Test.',
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['reason_code'] == 'TTS_AUDIO_FILE_MISSING'
    assert evidence['defect_codes'] == ['TTS_AUDIO_FILE_MISSING']
    assert evidence['materialization_eligible'] is False


def test_declared_source_digest_mismatch_fails_before_audio_acceptance(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'healthy.wav',
        [(1.0, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'Tatsächlicher Text.',
        source_sha256=hashlib.sha256(b'Anderer Text.').hexdigest(),
    )

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_SOURCE_DIGEST_MISMATCH'
    assert evidence['defect_codes'] == ['TTS_AUDIO_SOURCE_DIGEST_MISMATCH']
    assert evidence['source_digest_match'] is False


def test_qwen_exact_generation_ceiling_fails_even_with_enough_active_energy(tmp_path):
    source = 'The harbor wakes while the lighthouse watches every returning boat.'
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-at-cap.wav',
        [(0.8, True), (1.5, False), (1.7, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        source,
        generation_budget=_qwen_voice_design_budget(50),
        model_family='qwen3_tts',
        tts_model_type='voice_design',
    )

    assert evidence['effective_active_seconds'] == 2.5
    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED'
    assert evidence['materialization_eligible'] is False
    limit = evidence['generation_limit_evidence']
    assert limit['status'] == 'failed'
    assert limit['generation_limit_reached'] is True
    assert limit['expected_limit_frame_count'] == 32000
    assert limit['observed_frame_count'] == 32000
    assert limit['frame_delta_from_limit'] == 0


def test_qwen_exact_cap_is_independent_of_short_signal_and_trailing_silence(
    tmp_path,
):
    source = (
        'Mara stood beside the old lighthouse listening to the steady waves '
        'and thinking about the work still ahead.'
    )
    budget = _qwen_voice_design_budget(269)
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-short-padded-at-cap.wav',
        [(1.5, True), (20.02, False)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        source,
        generation_budget=budget,
        model_family='qwen3_tts',
        tts_model_type='voice_design',
        integrity_profile=TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE,
    )

    assert evidence['nominal_duration_seconds'] == 21.52
    assert evidence['effective_active_seconds'] == 1.5
    assert evidence['trailing_silence_seconds'] == 20.02
    assert evidence['reason_code'] == 'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT'
    assert evidence['defect_codes'] == [
        'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT',
        'TTS_AUDIO_EXCESSIVE_TRAILING_SILENCE',
        'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED',
    ]
    assert evidence['generation_limit_evidence']['generation_limit_reached'] is True
    assert tts_audio_has_qwen_generation_limit_exhaustion(
        evidence,
        generation_budget=budget,
    )


def test_qwen_audio_one_codec_token_below_ceiling_is_not_marked_exhausted(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-before-cap.wav',
        [(49 / 12.5, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'The lighthouse welcomes the morning.',
        generation_budget=_qwen_voice_design_budget(50),
        model_family='qwen3_tts',
        tts_model_type='voice_design',
    )

    assert evidence['status'] == 'passed'
    assert evidence['materialization_eligible'] is True
    limit = evidence['generation_limit_evidence']
    assert limit['status'] == 'passed'
    assert limit['generation_limit_reached'] is False
    assert limit['frame_delta_from_limit'] == -640


def test_qwen_base_exact_generation_ceiling_is_also_blocked(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-base-at-cap.wav',
        [(4.0, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'The Base model must stop with EOS before its declared ceiling.',
        generation_budget=_qwen_voice_design_budget(
            50,
            tts_model_type='base',
        ),
        model_family='qwen3_tts',
        tts_model_type='base',
    )

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED'
    assert evidence['generation_limit_evidence']['tts_model_type'] == 'base'
    assert evidence['generation_limit_evidence']['generation_limit_reached'] is True


def test_qwen_base_multiline_aggregate_does_not_claim_exact_single_sequence_proof(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-base-segmented.wav',
        [(4.0, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'First line.\nSecond line.',
        generation_budget=_qwen_voice_design_budget(
            50,
            generation_scope='segmented_sequence',
            tts_model_type='base',
        ),
        model_family='qwen3_tts',
        tts_model_type='base',
    )

    assert evidence['status'] == 'passed'
    assert evidence['generation_limit_evidence']['status'] == 'not_applicable'
    assert evidence['generation_limit_evidence']['generation_scope'] == 'segmented_sequence'


def test_non_qwen_audio_does_not_inherit_qwen_generation_ceiling_policy(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'other-model.wav',
        [(4.0, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'This ordinary audio remains valid for another TTS family.',
        model_family='kitten_tts',
        tts_model_type='kitten_tts',
    )

    assert evidence['status'] == 'passed'
    assert evidence['generation_limit_evidence']['status'] == 'not_applicable'


def test_qwen_single_sequence_without_declared_budget_fails_closed(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-missing-budget.wav',
        [(4.0, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'This voice design result has no declared generation budget.',
        model_family='qwen3_tts',
        tts_model_type='voice_design',
    )

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_GENERATION_LIMIT_EVIDENCE_UNAVAILABLE'
    assert evidence['materialization_eligible'] is False


def test_qwen_sentence_chunk_rejects_excessive_internal_silence(tmp_path):
    audio_path = _write_pcm_wav(
        tmp_path / 'qwen-internal-gap.wav',
        [(0.8, True), (3.4, False), (0.8, True)],
    )

    evidence = build_tts_audio_integrity_evidence(
        audio_path,
        'The harbor wakes and every boat begins moving.',
        integrity_profile=TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE,
    )

    assert evidence['status'] == 'failed'
    assert evidence['reason_code'] == 'TTS_AUDIO_EXCESSIVE_INTERNAL_SILENCE'
    assert evidence['longest_internal_silence_seconds'] >= 3.3
    assert evidence['materialization_eligible'] is False
