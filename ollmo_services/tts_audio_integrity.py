"""Deterministic physical-integrity evidence for generated TTS audio."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import wave
from array import array
from pathlib import Path
from typing import Any, Iterable


TTS_AUDIO_INTEGRITY_POLICY_ID = 'tts_wav_signal_completeness_v1'
_WINDOW_SECONDS = 0.1
_ABSOLUTE_ACTIVE_RMS = 0.0015
_RELATIVE_ACTIVE_RMS = 0.05
_MAX_WORDS_PER_SECOND = 5.0
_MIN_EXPECTED_ACTIVE_SECONDS = 0.25
_MAX_TRAILING_SILENCE_SECONDS = 4.0
_MAX_TRAILING_TO_ACTIVE_RATIO = 2.5
_MAX_TRAILING_SILENCE_RATIO = 0.65
_SUPPORTED_SAMPLE_WIDTHS = {1, 2, 3, 4}


def _round_metric(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def _source_words(source_text: Any) -> list[str]:
    return re.findall(
        r"[^\W_]+(?:['\u2019][^\W_]+)?",
        str(source_text or ''),
        flags=re.UNICODE,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _pcm_samples(raw: bytes, sample_width: int) -> Iterable[int]:
    if sample_width == 1:
        return (sample - 128 for sample in raw)
    if sample_width == 2:
        samples = array('h')
        samples.frombytes(raw)
        if sys.byteorder != 'little':
            samples.byteswap()
        return samples
    if sample_width == 4:
        samples = array('i')
        samples.frombytes(raw)
        if sys.byteorder != 'little':
            samples.byteswap()
        return samples
    if sample_width == 3:
        def iter_24_bit_samples() -> Iterable[int]:
            for offset in range(0, len(raw) - 2, 3):
                value = int.from_bytes(
                    raw[offset:offset + 3],
                    byteorder='little',
                    signed=False,
                )
                if value & 0x800000:
                    value -= 0x1000000
                yield value

        return iter_24_bit_samples()
    return ()


def _normalized_rms(raw: bytes, sample_width: int) -> float:
    samples = _pcm_samples(raw, sample_width)
    square_sum = 0
    sample_count = 0
    for sample in samples:
        square_sum += sample * sample
        sample_count += 1
    if not sample_count:
        return 0.0
    full_scale = float(1 << (sample_width * 8 - 1))
    return math.sqrt(square_sum / sample_count) / full_scale


def _base_evidence(
    *,
    source_text: Any,
    source_sha256: str | None,
    path: Any,
) -> tuple[dict[str, Any], str]:
    source = str(source_text or '')
    computed_source_sha256 = hashlib.sha256(source.encode('utf-8')).hexdigest()
    words = _source_words(source)
    minimum_expected_active_seconds = max(
        _MIN_EXPECTED_ACTIVE_SECONDS,
        len(words) / _MAX_WORDS_PER_SECOND,
    )
    evidence = {
        'kind': 'ollmo.tts_audio_integrity_evidence',
        'version': 1,
        'policy_id': TTS_AUDIO_INTEGRITY_POLICY_ID,
        'authority': 'runtime_deterministic_audio_verification',
        'status': 'unavailable',
        'reason_code': 'TTS_AUDIO_INTEGRITY_UNAVAILABLE',
        'materialization_eligible': False,
        'artifact_path': str(path or '').strip() or None,
        'source_sha256': computed_source_sha256,
        'source_word_count': len(words),
        'source_character_count': len(source),
        'minimum_expected_active_seconds': _round_metric(
            minimum_expected_active_seconds
        ),
        'thresholds': {
            'window_seconds': _WINDOW_SECONDS,
            'absolute_active_rms': _ABSOLUTE_ACTIVE_RMS,
            'relative_active_rms': _RELATIVE_ACTIVE_RMS,
            'max_words_per_second': _MAX_WORDS_PER_SECOND,
            'min_expected_active_seconds': _MIN_EXPECTED_ACTIVE_SECONDS,
            'max_trailing_silence_seconds': _MAX_TRAILING_SILENCE_SECONDS,
            'max_trailing_to_active_ratio': _MAX_TRAILING_TO_ACTIVE_RATIO,
            'max_trailing_silence_ratio': _MAX_TRAILING_SILENCE_RATIO,
        },
    }
    declared_source_sha256 = str(source_sha256 or '').strip().lower()
    if declared_source_sha256:
        evidence['declared_source_sha256'] = declared_source_sha256
        evidence['source_digest_match'] = (
            declared_source_sha256 == computed_source_sha256
        )
    return evidence, computed_source_sha256


def build_tts_audio_integrity_evidence(
    path: str | Path,
    source_text: Any,
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Measure whether a persisted PCM WAV plausibly contains the full TTS source."""

    evidence, computed_source_sha256 = _base_evidence(
        source_text=source_text,
        source_sha256=source_sha256,
        path=path,
    )
    declared_source_sha256 = str(source_sha256 or '').strip().lower()
    if (
        declared_source_sha256
        and declared_source_sha256 != computed_source_sha256
    ):
        evidence.update(
            {
                'status': 'failed',
                'reason_code': 'TTS_AUDIO_SOURCE_DIGEST_MISMATCH',
            }
        )
        return {key: value for key, value in evidence.items() if value is not None}

    artifact_path = Path(str(path or '')).expanduser()
    if not str(path or '').strip() or not artifact_path.is_file():
        evidence['reason_code'] = 'TTS_AUDIO_FILE_MISSING'
        return {key: value for key, value in evidence.items() if value is not None}

    try:
        evidence['artifact_size_bytes'] = artifact_path.stat().st_size
        evidence['artifact_sha256'] = _file_sha256(artifact_path)
        with wave.open(str(artifact_path), 'rb') as audio:
            channel_count = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression_type = audio.getcomptype()
            if (
                channel_count <= 0
                or sample_rate <= 0
                or frame_count <= 0
                or sample_width not in _SUPPORTED_SAMPLE_WIDTHS
                or compression_type != 'NONE'
            ):
                evidence.update(
                    {
                        'status': 'failed',
                        'reason_code': 'TTS_AUDIO_WAV_FORMAT_UNSUPPORTED',
                        'channel_count': channel_count,
                        'sample_width_bytes': sample_width,
                        'sample_rate_hz': sample_rate,
                        'frame_count': frame_count,
                        'compression_type': compression_type,
                    }
                )
                return {
                    key: value
                    for key, value in evidence.items()
                    if value is not None
                }

            frames_per_window = max(1, int(round(sample_rate * _WINDOW_SECONDS)))
            window_rms: list[float] = []
            window_durations: list[float] = []
            while True:
                raw = audio.readframes(frames_per_window)
                if not raw:
                    break
                bytes_per_frame = channel_count * sample_width
                frames_read = len(raw) // bytes_per_frame
                if frames_read <= 0:
                    break
                window_rms.append(_normalized_rms(raw, sample_width))
                window_durations.append(frames_read / sample_rate)
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        evidence.update(
            {
                'status': 'failed',
                'reason_code': 'TTS_AUDIO_WAV_UNREADABLE',
                'error_type': type(exc).__name__,
            }
        )
        return {key: value for key, value in evidence.items() if value is not None}

    nominal_duration = frame_count / sample_rate
    peak_window_rms = max(window_rms, default=0.0)
    active_rms_threshold = max(
        _ABSOLUTE_ACTIVE_RMS,
        peak_window_rms * _RELATIVE_ACTIVE_RMS,
    )
    active_indexes = [
        index
        for index, rms in enumerate(window_rms)
        if rms >= active_rms_threshold
    ]
    active_duration = sum(
        window_durations[index]
        for index in active_indexes
    )
    leading_silence = (
        sum(window_durations[:active_indexes[0]])
        if active_indexes
        else nominal_duration
    )
    trailing_silence = (
        sum(window_durations[active_indexes[-1] + 1:])
        if active_indexes
        else nominal_duration
    )
    silence_duration = max(0.0, nominal_duration - active_duration)
    silence_ratio = (
        silence_duration / nominal_duration
        if nominal_duration > 0
        else 1.0
    )
    trailing_silence_ratio = (
        trailing_silence / nominal_duration
        if nominal_duration > 0
        else 1.0
    )
    minimum_expected_active_seconds = float(
        evidence['minimum_expected_active_seconds']
    )
    evidence.update(
        {
            'container': 'wav',
            'encoding': f'pcm_s{sample_width * 8}le',
            'channel_count': channel_count,
            'sample_width_bytes': sample_width,
            'sample_rate_hz': sample_rate,
            'frame_count': frame_count,
            'compression_type': compression_type,
            'nominal_duration_seconds': _round_metric(nominal_duration),
            'effective_active_seconds': _round_metric(active_duration),
            'silence_seconds': _round_metric(silence_duration),
            'silence_ratio': _round_metric(silence_ratio),
            'leading_silence_seconds': _round_metric(leading_silence),
            'trailing_silence_seconds': _round_metric(trailing_silence),
            'trailing_silence_ratio': _round_metric(trailing_silence_ratio),
            'peak_window_rms': _round_metric(peak_window_rms),
            'active_rms_threshold': _round_metric(active_rms_threshold),
            'window_count': len(window_rms),
            'active_window_count': len(active_indexes),
        }
    )

    if not active_indexes:
        reason_code = 'TTS_AUDIO_NO_ACTIVE_SIGNAL'
    elif active_duration + 1e-9 < minimum_expected_active_seconds:
        reason_code = 'TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT'
    elif (
        trailing_silence > _MAX_TRAILING_SILENCE_SECONDS
        and trailing_silence > active_duration * _MAX_TRAILING_TO_ACTIVE_RATIO
        and trailing_silence_ratio > _MAX_TRAILING_SILENCE_RATIO
    ):
        reason_code = 'TTS_AUDIO_EXCESSIVE_TRAILING_SILENCE'
    else:
        reason_code = 'TTS_AUDIO_INTEGRITY_PASSED'

    passed = reason_code == 'TTS_AUDIO_INTEGRITY_PASSED'
    evidence.update(
        {
            'status': 'passed' if passed else 'failed',
            'reason_code': reason_code,
            'materialization_eligible': passed,
        }
    )
    return {key: value for key, value in evidence.items() if value is not None}
