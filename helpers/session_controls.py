"""Derive truthful session-controls schema from instance capability and metadata."""

from __future__ import annotations

from typing import Any

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    normalize_capability,
)
from helpers.ocr_modes import get_ocr_mode_copy, get_ocr_mode_options

_DEFAULT_STT_LANGUAGES = ['de', 'en', 'fr', 'es', 'it', 'pt', 'ja', 'ko', 'ru', 'zh']
_DEFAULT_STT_TASKS = ['transcribe', 'translate']


def _field(
    kind: str,
    *,
    label: str | None = None,
    description: str | None = None,
    options: list[str] | None = None,
    default_value: Any | None = None,
    default_first_option: bool = False,
    required: bool = False,
    required_message: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {'visible': True, 'kind': kind}
    if label:
        payload['label'] = label
    if description:
        payload['description'] = description
    if options is not None:
        payload['options'] = list(options)
    if default_value is not None:
        payload['default_value'] = default_value
    if default_first_option:
        payload['default_first_option'] = True
    if required:
        payload['required'] = True
    if required_message:
        payload['required_message'] = required_message
    return payload


def _empty_schema() -> dict[str, Any]:
    return {
        'enabled': False,
        'hint': 'No model-specific session controls available for this model.',
        'fields': {},
    }


def _normalized_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        token = str(raw or '').strip()
        if token and token not in items:
            items.append(token)
    return items


def build_session_controls(instance: dict | None) -> dict[str, Any]:
    if not isinstance(instance, dict):
        return _empty_schema()

    capability = normalize_capability(instance.get('capability'))
    model_name = str(instance.get('model') or instance.get('modelName') or '').strip()
    model_name_lower = model_name.lower()

    if capability == CAPABILITY_CHAT:
        return {
            'enabled': True,
            'hint': 'Sampling controls for this chat model.',
            'fields': {
                'chat_meta': _field(
                    'meta',
                    label='Chat Sampling',
                    description='These controls are applied only to the active chat model.',
                ),
                'temperature': _field(
                    'number',
                    label='Temperature',
                    description='Optional. Leave blank for the model default. Lower values stay tighter to the prompt; higher values allow more variation.',
                    default_value=0.7,
                ),
                'top_p': _field(
                    'number',
                    label='Top P',
                    description='Optional. Leave blank for the model default. Limit generation to the most likely probability mass before sampling.',
                    default_value=0.9,
                ),
            },
        }

    if capability == CAPABILITY_SPEECH_TO_TEXT:
        stt_languages = _normalized_text_list(instance.get('stt_languages')) or _DEFAULT_STT_LANGUAGES
        stt_tasks = _normalized_text_list(instance.get('stt_tasks')) or _DEFAULT_STT_TASKS
        is_realtime = bool(instance.get('stt_realtime'))
        hint = 'Language and task controls for this speech-to-text model.'
        description = 'Language and task controls for the active speech-to-text model.'
        label = 'Whisper Input'
        if str(instance.get('backend_package') or '').strip() == 'mlx_whisper_shim':
            hint = 'Language and task controls for the local MLX Whisper shim.'
            description = 'Language and task controls for the active local Whisper compatibility shim.'
            label = 'Whisper Shim Input'
        elif is_realtime:
            hint = 'Realtime transcription controls for this speech-to-text model.'
            description = 'This model advertises realtime or streaming transcription semantics; Ollmo currently exposes the compatible request-time basics only.'
            label = 'Realtime STT Input'
        return {
            'enabled': True,
            'hint': hint,
            'fields': {
                'stt_meta': _field(
                    'meta',
                    label=label,
                    description=description,
                ),
                'stt_language': _field(
                    'select',
                    label='Language',
                    description='Leave empty for auto-detect, or pin the spoken language for more stable transcription.',
                    options=stt_languages,
                ),
                'stt_task': _field(
                    'select',
                    label='Task',
                    description='Choose whether Whisper should transcribe verbatim or translate speech into English.',
                    options=stt_tasks,
                ),
            },
        }

    if capability == CAPABILITY_IMAGE_GENERATION:
        return {
            'enabled': True,
            'hint': 'Image size and batch controls for this image-generation model.',
            'fields': {
                'image_meta': _field(
                    'meta',
                    label='Image Generation',
                    description='Set output size for this image model. Attach one image when you want edit/reference-style generation from a base image.',
                ),
                'image_aspect_ratio': _field(
                    'select',
                    label='Aspect Preset',
                    description='Optional helper preset. Choosing a preset fills Width and Height with a matching size; Custom leaves your manual dimensions untouched.',
                    options=['auto', '1:1', '4:3', '3:4', '3:2', '2:3', '16:9', '9:16', 'custom'],
                ),
                'image_width': _field(
                    'number',
                    label='Width',
                    description='Optional output width in pixels. Use together with Height. Must be a multiple of 16.',
                ),
                'image_height': _field(
                    'number',
                    label='Height',
                    description='Optional output height in pixels. Use together with Width. Must be a multiple of 16.',
                ),
                'image_count': _field(
                    'number',
                    label='Image Count',
                    description='Generate multiple variants from the same prompt in one request.',
                    default_value=1,
                ),
            },
        }

    if capability == CAPABILITY_VISION_ANALYSIS:
        ocr_copy = get_ocr_mode_copy(model_name)
        description = ocr_copy['description']
        hint = ocr_copy['hint']
        label = ocr_copy['label']
        if 'deepseek-ocr-2' in model_name_lower:
            label = 'DeepSeek OCR 2 Mode'
        elif 'deepseek-ocr' in model_name_lower:
            description = 'These controls are used for PDF OCR requests with DeepSeek-OCR.'
            hint = 'PDF OCR controls for this DeepSeek-OCR model.'
            label = 'DeepSeek OCR Mode'
        ocr_mode_options = get_ocr_mode_options(model_name)
        fields = {
            'ocr_meta': _field(
                'meta',
                label=label,
                description=description,
            ),
        }
        if ocr_mode_options:
            fields['ocr_mode'] = _field(
                'select',
                label='Document Mode',
                description=ocr_copy['mode_description'],
                options=ocr_mode_options,
            )
        fields.update(
            {
                'pdf_max_pages': _field('number'),
                'pdf_dpi': _field(
                    'number',
                    label='PDF DPI',
                    description='Render each PDF page at this resolution before OCR. Higher DPI may improve accuracy but costs time.',
                    default_value=300,
                ),
                'pdf_page_timeout_sec': _field(
                    'number',
                    label='Page Timeout (s)',
                    description='Maximum OCR time budget per rendered PDF page before Ollmo aborts that page.',
                    default_value=180,
                ),
                'pdf_synthesize': _field(
                    'boolean',
                    label='Synthesize',
                    description='Merge per-page OCR output into one final answer instead of returning page-by-page results.',
                ),
            }
        )
        fields['pdf_max_pages'] = _field(
            'number',
            label='Max Page Budget',
            description='Optional override. Leave blank to process all PDF pages. Enter a number only when you want to cap how many pages Ollmo renders and OCRs for this request.',
        )
        return {
            'enabled': True,
            'hint': hint,
            'fields': fields,
        }

    if capability == CAPABILITY_TEXT_TO_SPEECH:
        speakers = _normalized_text_list(instance.get('tts_speakers'))
        languages = _normalized_text_list(instance.get('tts_languages'))
        response_formats = _normalized_text_list(
            instance.get('tts_response_formats')
            or (instance.get('backend_metadata') or {}).get('tts_response_formats')
        )
        model_type = str(instance.get('tts_model_type') or '').strip().lower()
        supports_instruct = model_type in {'voice_design', 'custom_voice'}
        hint = 'Core synthesis controls for this TTS model.'
        description = 'Core synthesis controls for the active TTS model.'
        if model_type == 'custom_voice':
            hint = 'Speaker-based TTS controls for this CustomVoice model.'
            description = 'CustomVoice models are best for a stable named speaker.'
        elif model_type == 'kitten_tts':
            hint = 'Speaker-based TTS controls for this Kitten model.'
            description = 'Kitten TTS requires an explicit speaker selection.'
        elif model_type == 'voice_design':
            hint = 'Natural-language voice design controls for this VoiceDesign model.'
            description = 'VoiceDesign models are best for natural-language voice descriptions.'

        fields: dict[str, Any] = {
            'tts_meta': _field(
                'meta',
                label='TTS Mode',
                description=description,
            ),
            'tts_speed': _field(
                'number',
                label='Speed',
                description='Playback and speaking rate for the generated audio.',
                default_value=1.0,
            ),
            'tts_pitch': _field(
                'number',
                label='Pitch',
                description='Shift the generated voice lower or higher without changing the text.',
                default_value=1.0,
            ),
        }
        if speakers:
            fields['tts_voice'] = _field(
                'select',
                label='Voice / Speaker',
                description=(
                    'Pick a discovered speaker identity for this model.'
                    if model_type != 'kitten_tts'
                    else 'Pick one of the built-in Kitten speakers. A valid speaker is required.'
                ),
                options=speakers,
                default_first_option=(model_type in {'custom_voice', 'kitten_tts'}),
                required=(model_type == 'kitten_tts'),
                required_message=(
                    'Kitten TTS models require a valid speaker. Ollmo should auto-fill one from the discovered speaker list.'
                    if model_type == 'kitten_tts'
                    else None
                ),
            )
        if languages:
            fields['tts_language'] = _field(
                'select',
                label='Language',
                description='Leave empty for auto-detect, or pin the target speaking language explicitly.',
                options=languages,
            )
        if response_formats:
            fields['tts_response_format'] = _field(
                'select',
                label='Output Format',
                description='Optional. Leave blank for Ollmo’s local default WAV output, or choose an explicit audio container format for the generated file.',
                options=response_formats,
            )
        if supports_instruct:
            fields['tts_instruct'] = _field(
                'textarea',
                label='Style / Instruct',
                description=(
                    'Describe the target voice in natural language.'
                    if model_type == 'voice_design'
                    else 'Optional natural-language style guidance for this TTS model.'
                ),
                required=(model_type == 'voice_design'),
                required_message=(
                    'VoiceDesign models require a voice description in Style / Instruct.'
                    if model_type == 'voice_design'
                    else None
                ),
            )
        return {
            'enabled': True,
            'hint': hint,
            'fields': fields,
        }

    return _empty_schema()
