from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path

from ollmo_services.tts_audio_integrity import (
    TTS_AUDIO_INTEGRITY_POLICY_ID,
    build_tts_audio_integrity_evidence,
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
    assert evidence['materialization_eligible'] is False


def test_missing_wav_is_unavailable_and_not_materialization_eligible(tmp_path):
    evidence = build_tts_audio_integrity_evidence(
        tmp_path / 'missing.wav',
        'Test.',
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['reason_code'] == 'TTS_AUDIO_FILE_MISSING'
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
    assert evidence['source_digest_match'] is False
