"""Deterministic physical-integrity evidence for generated TTS audio."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import wave
from array import array
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping


TTS_AUDIO_INTEGRITY_POLICY_ID = 'tts_wav_signal_completeness_v2'
TTS_SEMANTIC_SOURCE_POLICY_ID = 'tts_stt_lexical_fidelity_v1'
TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE = 'qwen3_tts_sentence_chunk_v1'
_QWEN_GENERATION_LIMIT_POLICY_ID = 'qwen3_tts_generation_limit_exhaustion_v1'
_WINDOW_SECONDS = 0.1
_ABSOLUTE_ACTIVE_RMS = 0.0015
_RELATIVE_ACTIVE_RMS = 0.05
_MAX_WORDS_PER_SECOND = 5.0
_MIN_EXPECTED_ACTIVE_SECONDS = 0.25
_MAX_TRAILING_SILENCE_SECONDS = 4.0
_MAX_TRAILING_TO_ACTIVE_RATIO = 2.5
_MAX_TRAILING_SILENCE_RATIO = 0.65
_MAX_SENTENCE_CHUNK_INTERNAL_SILENCE_SECONDS = 3.0
_SUPPORTED_SAMPLE_WIDTHS = {1, 2, 3, 4}
_QWEN_SINGLE_SEQUENCE_MODEL_TYPES = {
    'base',
    'voice_design',
    'custom_voice',
}


def build_tts_semantic_source(
    source_text: Any,
    *,
    source_authority: str = 'final_infer_prompt',
    source_text_source: Any = None,
    branch_id: Any = None,
    phase_id: Any = None,
    lang_code: Any = None,
) -> dict[str, Any]:
    """Bind the exact final TTS backend text to deterministic runtime evidence."""

    source = str(source_text) if source_text is not None else ''
    if not source.strip():
        return {}
    authority = str(source_authority or '').strip() or 'final_infer_prompt'
    source_bytes = source.encode('utf-8')
    return {
        key: value
        for key, value in {
            'kind': 'ollmo.tts_semantic_source',
            'version': 1,
            'policy_id': TTS_SEMANTIC_SOURCE_POLICY_ID,
            'authority': 'runtime_exact_backend_prompt',
            'source_authority': authority,
            'tts_source_text': source,
            'tts_source_text_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'tts_source_text_source': (
                str(source_text_source or '').strip() or authority
            ),
            'branch_id': str(branch_id or '').strip() or None,
            'phase_id': str(phase_id or '').strip() or None,
            'lang_code': str(lang_code or '').strip().lower() or None,
        }.items()
        if value not in (None, '', [], {})
    }


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
            'max_sentence_chunk_internal_silence_seconds': (
                _MAX_SENTENCE_CHUNK_INTERNAL_SILENCE_SECONDS
            ),
        },
    }
    declared_source_sha256 = str(source_sha256 or '').strip().lower()
    if declared_source_sha256:
        evidence['declared_source_sha256'] = declared_source_sha256
        evidence['source_digest_match'] = (
            declared_source_sha256 == computed_source_sha256
        )
    return evidence, computed_source_sha256


def _qwen_generation_limit_evidence(
    *,
    frame_count: int,
    sample_rate: int,
    generation_budget: Mapping[str, Any] | None,
    model_family: Any,
    tts_model_type: Any,
) -> dict[str, Any]:
    budget = generation_budget if isinstance(generation_budget, Mapping) else {}
    family = str(model_family or budget.get('model_family') or '').strip().lower()
    model_type = str(
        tts_model_type or budget.get('tts_model_type') or ''
    ).strip().lower()
    generation_scope = str(budget.get('generation_scope') or '').strip().lower()
    qwen_family = family == 'qwen3_tts'
    supported_single_sequence_model = (
        model_type in _QWEN_SINGLE_SEQUENCE_MODEL_TYPES
    )
    requires_single_sequence_evidence = bool(
        qwen_family
        and supported_single_sequence_model
        and generation_scope not in {'chunked_sequence', 'segmented_sequence'}
    )
    evidence: dict[str, Any] = {
        'kind': 'ollmo.tts_generation_limit_evidence',
        'version': 1,
        'policy_id': _QWEN_GENERATION_LIMIT_POLICY_ID,
        'model_family': family or None,
        'tts_model_type': model_type or None,
        'generation_scope': generation_scope or None,
        'completion_signal': 'not_exposed_by_backend',
        'status': 'not_applicable',
        'reason_code': 'TTS_AUDIO_GENERATION_LIMIT_NOT_APPLICABLE',
        'generation_limit_reached': False,
    }
    if not qwen_family or generation_scope in {
        'chunked_sequence',
        'segmented_sequence',
    }:
        return {key: value for key, value in evidence.items() if value is not None}
    if generation_scope != 'single_sequence':
        if requires_single_sequence_evidence:
            evidence.update(
                {
                    'status': 'unavailable',
                    'reason_code': 'TTS_AUDIO_GENERATION_LIMIT_EVIDENCE_UNAVAILABLE',
                }
            )
        return {key: value for key, value in evidence.items() if value is not None}

    raw_max_tokens = budget.get('max_tokens')
    policy = budget.get('policy') if isinstance(budget.get('policy'), Mapping) else {}
    raw_tokens_per_second = policy.get('audio_tokens_per_second')
    try:
        max_tokens = int(raw_max_tokens)
        tokens_per_second = Fraction(str(raw_tokens_per_second))
    except (TypeError, ValueError, ZeroDivisionError):
        max_tokens = 0
        tokens_per_second = Fraction(0, 1)
    if max_tokens <= 0 or tokens_per_second <= 0 or sample_rate <= 0:
        evidence.update(
            {
                'status': 'unavailable',
                'reason_code': 'TTS_AUDIO_GENERATION_LIMIT_EVIDENCE_UNAVAILABLE',
            }
        )
        return {key: value for key, value in evidence.items() if value is not None}

    samples_per_audio_token = Fraction(sample_rate, 1) / tokens_per_second
    expected_limit_frames = samples_per_audio_token * max_tokens
    expected_frames_integral = expected_limit_frames.denominator == 1
    expected_frame_count = (
        expected_limit_frames.numerator
        if expected_frames_integral
        else None
    )
    generation_limit_reached = bool(
        expected_frame_count is not None
        and frame_count == expected_frame_count
    )
    inferred_audio_tokens = Fraction(frame_count, 1) / samples_per_audio_token
    evidence.update(
        {
            'budget_policy_id': str(budget.get('policy_id') or '').strip() or None,
            'max_tokens': max_tokens,
            'audio_tokens_per_second': float(tokens_per_second),
            'samples_per_audio_token': (
                int(samples_per_audio_token)
                if samples_per_audio_token.denominator == 1
                else float(samples_per_audio_token)
            ),
            'expected_limit_frame_count': expected_frame_count,
            'expected_limit_duration_seconds': _round_metric(
                float(expected_limit_frames / sample_rate)
            ),
            'observed_frame_count': frame_count,
            'observed_duration_seconds': _round_metric(frame_count / sample_rate),
            'frame_delta_from_limit': (
                frame_count - expected_frame_count
                if expected_frame_count is not None
                else None
            ),
            'inferred_audio_tokens': (
                int(inferred_audio_tokens)
                if inferred_audio_tokens.denominator == 1
                else round(float(inferred_audio_tokens), 6)
            ),
            'generation_limit_reached': generation_limit_reached,
            'status': 'failed' if generation_limit_reached else 'passed',
            'reason_code': (
                'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED'
                if generation_limit_reached
                else 'TTS_AUDIO_GENERATION_LIMIT_NOT_REACHED'
            ),
        }
    )
    if not expected_frames_integral:
        evidence.update(
            {
                'status': 'unavailable',
                'reason_code': 'TTS_AUDIO_GENERATION_LIMIT_EVIDENCE_UNAVAILABLE',
                'generation_limit_reached': False,
            }
        )
    return {key: value for key, value in evidence.items() if value is not None}


def tts_audio_has_qwen_generation_limit_exhaustion(
    integrity_evidence: Mapping[str, Any] | None,
    *,
    generation_budget: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether one verified Qwen PCM-WAV sequence hit its exact cap."""

    evidence = (
        integrity_evidence
        if isinstance(integrity_evidence, Mapping)
        else {}
    )
    budget = generation_budget if isinstance(generation_budget, Mapping) else {}
    limit = (
        evidence.get('generation_limit_evidence')
        if isinstance(evidence.get('generation_limit_evidence'), Mapping)
        else {}
    )
    model_family = str(
        limit.get('model_family')
        or evidence.get('model_family')
        or budget.get('model_family')
        or ''
    ).strip().lower()
    model_type = str(
        limit.get('tts_model_type')
        or evidence.get('tts_model_type')
        or budget.get('tts_model_type')
        or ''
    ).strip().lower()
    evidence_scope = str(limit.get('generation_scope') or '').strip().lower()
    budget_scope = str(budget.get('generation_scope') or '').strip().lower()
    try:
        limit_max_tokens = int(limit.get('max_tokens') or 0)
        budget_max_tokens = int(budget.get('max_tokens') or 0)
    except (TypeError, ValueError):
        return False
    try:
        frame_delta_from_limit = int(limit.get('frame_delta_from_limit'))
    except (TypeError, ValueError):
        return False
    return bool(
        str(evidence.get('kind') or '').strip()
        == 'ollmo.tts_audio_integrity_evidence'
        and evidence.get('version') == 1
        and str(evidence.get('container') or '').strip().lower() == 'wav'
        and str(evidence.get('encoding') or '').strip().lower().startswith('pcm_')
        and str(evidence.get('status') or '').strip().lower() == 'failed'
        and evidence.get('materialization_eligible') is False
        and str(limit.get('kind') or '').strip()
        == 'ollmo.tts_generation_limit_evidence'
        and limit.get('version') == 1
        and str(limit.get('policy_id') or '').strip()
        == _QWEN_GENERATION_LIMIT_POLICY_ID
        and str(limit.get('status') or '').strip().lower() == 'failed'
        and str(limit.get('reason_code') or '').strip()
        == 'TTS_AUDIO_GENERATION_LIMIT_EXHAUSTED'
        and limit.get('generation_limit_reached') is True
        and frame_delta_from_limit == 0
        and model_family == 'qwen3_tts'
        and model_type in _QWEN_SINGLE_SEQUENCE_MODEL_TYPES
        and evidence_scope == 'single_sequence'
        and budget_scope == 'single_sequence'
        and limit_max_tokens > 0
        and limit_max_tokens == budget_max_tokens
    )


def _unusable_integrity_evidence(
    evidence: Mapping[str, Any],
    *,
    status: str,
    reason_code: str,
) -> dict[str, Any]:
    payload = dict(evidence)
    payload.update(
        {
            'status': status,
            'reason_code': reason_code,
            'materialization_eligible': False,
            'defect_codes': [reason_code],
        }
    )
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def build_tts_audio_integrity_evidence(
    path: str | Path,
    source_text: Any,
    *,
    source_sha256: str | None = None,
    generation_budget: Mapping[str, Any] | None = None,
    model_family: Any = None,
    tts_model_type: Any = None,
    integrity_profile: Any = None,
) -> dict[str, Any]:
    """Measure whether a persisted PCM WAV plausibly contains the full TTS source."""

    evidence, computed_source_sha256 = _base_evidence(
        source_text=source_text,
        source_sha256=source_sha256,
        path=path,
    )
    normalized_integrity_profile = str(integrity_profile or '').strip()
    if normalized_integrity_profile:
        evidence['integrity_profile'] = normalized_integrity_profile
    normalized_model_family = str(
        model_family
        or (
            generation_budget.get('model_family')
            if isinstance(generation_budget, Mapping)
            else ''
        )
        or ''
    ).strip().lower()
    normalized_tts_model_type = str(
        tts_model_type
        or (
            generation_budget.get('tts_model_type')
            if isinstance(generation_budget, Mapping)
            else ''
        )
        or ''
    ).strip().lower()
    if normalized_model_family:
        evidence['model_family'] = normalized_model_family
    if normalized_tts_model_type:
        evidence['tts_model_type'] = normalized_tts_model_type
    declared_source_sha256 = str(source_sha256 or '').strip().lower()
    if (
        declared_source_sha256
        and declared_source_sha256 != computed_source_sha256
    ):
        return _unusable_integrity_evidence(
            evidence,
            status='failed',
            reason_code='TTS_AUDIO_SOURCE_DIGEST_MISMATCH',
        )

    artifact_path = Path(str(path or '')).expanduser()
    if not str(path or '').strip() or not artifact_path.is_file():
        return _unusable_integrity_evidence(
            evidence,
            status='unavailable',
            reason_code='TTS_AUDIO_FILE_MISSING',
        )

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
                        'channel_count': channel_count,
                        'sample_width_bytes': sample_width,
                        'sample_rate_hz': sample_rate,
                        'frame_count': frame_count,
                        'compression_type': compression_type,
                    }
                )
                return _unusable_integrity_evidence(
                    evidence,
                    status='failed',
                    reason_code='TTS_AUDIO_WAV_FORMAT_UNSUPPORTED',
                )

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
        evidence['error_type'] = type(exc).__name__
        return _unusable_integrity_evidence(
            evidence,
            status='failed',
            reason_code='TTS_AUDIO_WAV_UNREADABLE',
        )

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
    longest_internal_silence = 0.0
    if len(active_indexes) > 1:
        previous_active_index = active_indexes[0]
        for active_index in active_indexes[1:]:
            if active_index > previous_active_index + 1:
                longest_internal_silence = max(
                    longest_internal_silence,
                    sum(
                        window_durations[
                            previous_active_index + 1:active_index
                        ]
                    ),
                )
            previous_active_index = active_index
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
            'longest_internal_silence_seconds': _round_metric(
                longest_internal_silence
            ),
            'peak_window_rms': _round_metric(peak_window_rms),
            'active_rms_threshold': _round_metric(active_rms_threshold),
            'window_count': len(window_rms),
            'active_window_count': len(active_indexes),
        }
    )

    generation_limit_evidence = _qwen_generation_limit_evidence(
        frame_count=frame_count,
        sample_rate=sample_rate,
        generation_budget=generation_budget,
        model_family=normalized_model_family,
        tts_model_type=normalized_tts_model_type,
    )
    evidence['generation_limit_evidence'] = generation_limit_evidence

    defect_codes: list[str] = []
    if not active_indexes:
        defect_codes.append('TTS_AUDIO_NO_ACTIVE_SIGNAL')
    if active_duration + 1e-9 < minimum_expected_active_seconds:
        defect_codes.append('TTS_AUDIO_EFFECTIVE_DURATION_TOO_SHORT')
    if (
        trailing_silence > _MAX_TRAILING_SILENCE_SECONDS
        and trailing_silence > active_duration * _MAX_TRAILING_TO_ACTIVE_RATIO
        and trailing_silence_ratio > _MAX_TRAILING_SILENCE_RATIO
    ):
        defect_codes.append('TTS_AUDIO_EXCESSIVE_TRAILING_SILENCE')
    if (
        normalized_integrity_profile
        == TTS_QWEN_SENTENCE_CHUNK_INTEGRITY_PROFILE
        and longest_internal_silence
        > _MAX_SENTENCE_CHUNK_INTERNAL_SILENCE_SECONDS
    ):
        defect_codes.append('TTS_AUDIO_EXCESSIVE_INTERNAL_SILENCE')
    if str(generation_limit_evidence.get('status') or '').strip().lower() in {
        'failed',
        'unavailable',
    }:
        generation_limit_reason = str(
            generation_limit_evidence.get('reason_code')
            or 'TTS_AUDIO_GENERATION_LIMIT_EVIDENCE_UNAVAILABLE'
        ).strip()
        if generation_limit_reason and generation_limit_reason not in defect_codes:
            defect_codes.append(generation_limit_reason)

    reason_code = (
        defect_codes[0]
        if defect_codes
        else 'TTS_AUDIO_INTEGRITY_PASSED'
    )
    passed = not defect_codes
    evidence.update(
        {
            'status': 'passed' if passed else 'failed',
            'reason_code': reason_code,
            'materialization_eligible': passed,
            'defect_codes': defect_codes,
        }
    )
    return {key: value for key, value in evidence.items() if value is not None}
