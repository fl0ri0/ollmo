"""Helpers for reading stable model metadata from local snapshots."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

from helpers.model_capabilities import (
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
)

_TTS_RESPONSE_FORMATS = ['mp3', 'wav', 'flac']
_SUPPORTED_PIPELINE_TAG_CAPABILITIES = {
    'text-to-speech': CAPABILITY_TEXT_TO_SPEECH,
    'automatic-speech-recognition': CAPABILITY_SPEECH_TO_TEXT,
    'image-to-text': CAPABILITY_VISION_ANALYSIS,
    'image-text-to-text': CAPABILITY_VISION_ANALYSIS,
}
_TTS_TAG_MARKERS = {
    'text-to-speech',
    'speech generation',
    'tts',
}
_STT_TAG_MARKERS = {
    'automatic-speech-recognition',
    'asr',
    'speech-to-text',
    'speech recognition',
    'transcription',
}
_VISION_TAG_MARKERS = {
    'vision-language-model',
    'vision_language_model',
    'image-to-text',
    'image_text_to_text',
    'image-text-to-text',
    'vlm',
    'multimodal',
}
_VOICE_SECTION_RE = re.compile(
    r'^\s{0,3}#{1,6}\s+available\s+voices\b(?P<body>.*?)(?=^\s{0,3}#{1,6}\s+|\Z)',
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_BACKTICK_TOKEN_RE = re.compile(r'`([^`]+)`')
_MLX_VLM_CONVERSION_RE = re.compile(
    r'converted\s+to\s+MLX\s+format.*?using\s+mlx-vlm\s+version\s+\*{0,2}([0-9][0-9A-Za-z_.-]*)\*{0,2}',
    re.IGNORECASE | re.DOTALL,
)
_REASONING_EFFORT_GUARD_RE = re.compile(
    r'(?:reasoning_effort|resolved_reasoning_effort)'
    r'.{0,2500}?\bnot\s+in\s*[\(\[](?P<body>[^\)\]]+)[\)\]]',
    re.IGNORECASE | re.DOTALL,
)
_REASONING_EFFORT_LITERAL_RE = re.compile(
    r"['\"](off|low|medium|xhigh)['\"]",
    re.IGNORECASE,
)
_REASONING_EFFORT_DEFAULT_RE = re.compile(
    r"(?:reasoning_effort|resolved_reasoning_effort)\s*\|\s*default\s*\(\s*['\"]"
    r"(?P<default>off|low|medium|xhigh)['\"]\s*\)",
    re.IGNORECASE,
)


def _zip_member_stems(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with zipfile.ZipFile(path) as archive:
            names = []
            for member in archive.namelist():
                stem = Path(str(member)).stem.strip()
                if stem and stem not in names:
                    names.append(stem)
            return names
    except Exception:
        return []


def _dedupe_text_list(values: list[Any]) -> list[str]:
    items: list[str] = []
    for raw in values:
        token = str(raw or '').strip()
        if token and token not in items:
            items.append(token)
    return items


def _parse_scalar(value: str) -> Any:
    token = str(value or '').strip()
    if not token:
        return ''
    if token[0:1] in {'"', "'"} and token[-1:] == token[0:1]:
        return token[1:-1]
    lowered = token.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    return token


def _parse_front_matter(text: str) -> dict[str, Any]:
    if not text.startswith('---\n'):
        return {}
    end_marker = '\n---\n'
    end_index = text.find(end_marker, 4)
    if end_index < 0:
        return {}
    body = text[4:end_index]
    parsed: dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in body.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith('- '):
            if not current_key:
                continue
            parsed.setdefault(current_key, [])
            if isinstance(parsed[current_key], list):
                parsed[current_key].append(_parse_scalar(raw_line.split('- ', 1)[1]))
            continue
        current_key = None
        if ':' not in raw_line:
            continue
        key, value = raw_line.split(':', 1)
        normalized_key = str(key or '').strip().lower().replace('-', '_')
        if not normalized_key:
            continue
        remainder = value.strip()
        if remainder:
            parsed[normalized_key] = _parse_scalar(remainder)
        else:
            parsed[normalized_key] = []
            current_key = normalized_key
    return parsed


def _extract_voice_names_from_model_card(text: str) -> list[str]:
    match = _VOICE_SECTION_RE.search(text or '')
    if not match:
        return []
    return _dedupe_text_list(_BACKTICK_TOKEN_RE.findall(match.group('body') or ''))


def _extract_mlx_vlm_conversion_version(text: str) -> str:
    match = _MLX_VLM_CONVERSION_RE.search(text or '')
    if not match:
        return ''
    return str(match.group(1) or '').strip()


def _read_chat_template_text(snapshot_path: Path) -> str:
    template_path = snapshot_path / 'chat_template.jinja'
    if not template_path.exists():
        return ''
    try:
        return template_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return template_path.read_text(errors='ignore')
        except Exception:
            return ''
    except Exception:
        return ''


def _extract_reasoning_efforts_from_chat_template(text: str) -> list[str]:
    """Return literal reasoning efforts declared by a chat template.

    This intentionally looks for a template guard such as
    ``reasoning_effort not in ('xhigh', 'medium', 'low')`` rather than
    inferring support from model names or prose in a model card.
    """

    efforts: list[str] = []
    for match in _REASONING_EFFORT_GUARD_RE.finditer(text or ''):
        body = match.group('body') or ''
        for raw in _REASONING_EFFORT_LITERAL_RE.findall(body):
            token = str(raw or '').strip().lower()
            if token and token not in efforts:
                efforts.append(token)
    return efforts


def _extract_reasoning_effort_default_from_chat_template(text: str) -> str | None:
    """Return the template's declared default, if it uses the supported contract."""
    match = _REASONING_EFFORT_DEFAULT_RE.search(text or '')
    if not match:
        return None
    return str(match.group('default') or '').strip().lower() or None


def _read_model_card_text(snapshot_path: Path) -> str:
    readme_path = snapshot_path / 'README.md'
    if not readme_path.exists():
        return ''
    try:
        return readme_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return readme_path.read_text(errors='ignore')
        except Exception:
            return ''
    except Exception:
        return ''


def _pipeline_capability_tag(front_matter: dict[str, Any], model_type: str, name: str) -> str:
    pipeline_tag = str(front_matter.get('pipeline_tag') or '').strip().lower()
    if pipeline_tag in _SUPPORTED_PIPELINE_TAG_CAPABILITIES:
        return _SUPPORTED_PIPELINE_TAG_CAPABILITIES[pipeline_tag]

    tags = [str(item or '').strip().lower() for item in front_matter.get('tags', []) if str(item or '').strip()]
    if any(tag in _TTS_TAG_MARKERS for tag in tags):
        return CAPABILITY_TEXT_TO_SPEECH
    if any(tag in _STT_TAG_MARKERS for tag in tags):
        return CAPABILITY_SPEECH_TO_TEXT
    if any(tag in _VISION_TAG_MARKERS for tag in tags):
        return CAPABILITY_VISION_ANALYSIS

    if 'tts' in model_type or 'tts' in name:
        return CAPABILITY_TEXT_TO_SPEECH
    if 'whisper' in model_type or 'whisper' in name:
        return CAPABILITY_SPEECH_TO_TEXT
    return ''


def _config_capability_tag(config_payload: dict[str, Any]) -> str:
    if not isinstance(config_payload, dict):
        return ''
    if config_payload.get('image_token_id') is not None:
        return CAPABILITY_VISION_ANALYSIS
    if isinstance(config_payload.get('vision_config'), dict):
        return CAPABILITY_VISION_ANALYSIS
    return ''


def read_snapshot_model_metadata(model_name: Optional[str], model_path: Optional[str]) -> dict[str, Any]:
    name = str(model_name or '').strip().lower()
    raw_path = str(model_path or '').strip()
    if not raw_path:
        return {}

    snapshot_path = Path(raw_path)
    config_path = snapshot_path / 'config.json'
    config_payload: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                config_payload = loaded
        except Exception:
            config_payload = {}

    model_card_text = _read_model_card_text(snapshot_path)
    chat_template_text = _read_chat_template_text(snapshot_path)
    front_matter = _parse_front_matter(model_card_text)
    model_type = str(config_payload.get('tts_model_type') or config_payload.get('model_type') or '').strip().lower()
    capability = _pipeline_capability_tag(front_matter, model_type, name)
    if not capability:
        capability = _config_capability_tag(config_payload)

    metadata: dict[str, Any] = {}
    if model_type:
        metadata['snapshot_model_type'] = model_type
        metadata['tts_model_type'] = model_type if capability == CAPABILITY_TEXT_TO_SPEECH else metadata.get('tts_model_type')
    if front_matter:
        pipeline_tag = str(front_matter.get('pipeline_tag') or '').strip().lower()
        if pipeline_tag:
            metadata['snapshot_pipeline_tag'] = pipeline_tag
        tags = _dedupe_text_list(front_matter.get('tags') if isinstance(front_matter.get('tags'), list) else [])
        if tags:
            metadata['snapshot_tags'] = tags
        languages = _dedupe_text_list(front_matter.get('language') if isinstance(front_matter.get('language'), list) else [])
        if languages:
            metadata['snapshot_languages'] = languages
    mlx_vlm_conversion_version = _extract_mlx_vlm_conversion_version(model_card_text)
    if mlx_vlm_conversion_version:
        metadata['snapshot_mlx_vlm_conversion_version'] = mlx_vlm_conversion_version
    reasoning_efforts = _extract_reasoning_efforts_from_chat_template(chat_template_text)
    if reasoning_efforts:
        metadata['reasoning_efforts'] = reasoning_efforts
        reasoning_effort_default = _extract_reasoning_effort_default_from_chat_template(
            chat_template_text
        )
        if reasoning_effort_default in reasoning_efforts:
            metadata['reasoning_effort_default'] = reasoning_effort_default

    if capability:
        metadata['provider_capabilities'] = [capability]
        if capability == CAPABILITY_TEXT_TO_SPEECH:
            metadata['inputs'] = ['text']
            metadata['outputs'] = ['audio']
            metadata.setdefault('tts_response_formats', list(_TTS_RESPONSE_FORMATS))
        elif capability == CAPABILITY_SPEECH_TO_TEXT:
            metadata['inputs'] = ['audio']
            metadata['outputs'] = ['text']
        elif capability == CAPABILITY_VISION_ANALYSIS:
            metadata['inputs'] = ['text', 'image']
            metadata['outputs'] = ['text']

    voice_aliases = config_payload.get('voice_aliases')
    if isinstance(voice_aliases, dict):
        speakers = _dedupe_text_list(list(voice_aliases.keys()))
        if speakers:
            metadata['tts_speakers'] = speakers

    voices_path = str(config_payload.get('voices_path') or '').strip()
    if voices_path and 'tts_speakers' not in metadata:
        speakers = _zip_member_stems(snapshot_path / voices_path)
        if speakers:
            metadata['tts_speakers'] = speakers

    talker_config = config_payload.get('talker_config')
    if isinstance(talker_config, dict):
        speakers = talker_config.get('spk_id')
        if isinstance(speakers, dict):
            metadata['tts_speakers'] = _dedupe_text_list(list(speakers.keys()))

        languages = talker_config.get('codec_language_id')
        if isinstance(languages, dict):
            supported_languages = ['auto']
            for key in languages.keys():
                token = str(key or '').strip()
                if not token or 'dialect' in token.lower():
                    continue
                if token not in supported_languages:
                    supported_languages.append(token)
            metadata['tts_languages'] = supported_languages

    if capability == CAPABILITY_TEXT_TO_SPEECH:
        if 'tts_languages' not in metadata:
            card_languages = _dedupe_text_list(front_matter.get('language') if isinstance(front_matter.get('language'), list) else [])
            if card_languages:
                metadata['tts_languages'] = ['auto', *card_languages]
        if 'tts_speakers' not in metadata and model_card_text:
            voice_names = _extract_voice_names_from_model_card(model_card_text)
            if voice_names:
                metadata['tts_speakers'] = voice_names

    if capability == CAPABILITY_SPEECH_TO_TEXT:
        card_languages = _dedupe_text_list(front_matter.get('language') if isinstance(front_matter.get('language'), list) else [])
        if card_languages:
            metadata['stt_languages'] = card_languages
        lowered_card = model_card_text.lower()
        if 'realtime' in lowered_card or 'real-time' in lowered_card or 'websocket' in lowered_card or 'stream' in lowered_card:
            metadata['stt_realtime'] = True

    return metadata


def read_tts_model_metadata(model_name: Optional[str], model_path: Optional[str]) -> dict:
    """Backward-compatible wrapper for callers/tests that still use the old name."""
    return read_snapshot_model_metadata(model_name, model_path)
