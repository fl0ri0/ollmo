"""Prompt-to-control hint extraction for simple routed requests."""

from __future__ import annotations

import re
from typing import Any, Optional

from ollmo_g.intent import (
    analyze_prompt_intent,
    infer_audio_response_format,
    infer_image_aspect_ratio,
    normalize_intent_text,
)
from ollmo_g.image_state import image_state_anchor
from ollmo_services.tts_source import extract_tts_source_text

IMAGE_ASPECT_PRESET_DIMENSIONS = {
    '1:1': {'width': 1024, 'height': 1024},
    '4:3': {'width': 1024, 'height': 768},
    '3:4': {'width': 768, 'height': 1024},
    '3:2': {'width': 1152, 'height': 768},
    '2:3': {'width': 768, 'height': 1152},
    '16:9': {'width': 1280, 'height': 720},
    '9:16': {'width': 720, 'height': 1280},
}

_EXPLICIT_ASPECT_RE = re.compile(r'\b(1:1|4:3|3:4|3:2|2:3|16:9|9:16)\b', re.IGNORECASE)
_SQUARE_RE = re.compile(r'\b(square|quadratisch)\b', re.IGNORECASE)
_LANDSCAPE_RE = re.compile(r'\b(landscape|wide|widescreen|cinematic|banner|horizontal)\b', re.IGNORECASE)
_PORTRAIT_RE = re.compile(r'\b(portrait|hochformat)\b', re.IGNORECASE)
_VERTICAL_RE = re.compile(r'\b(vertical|tall|story format|phone wallpaper)\b', re.IGNORECASE)
_DOUBLE_QUOTE_RE = re.compile(r'"([^"\n]+)"')
_SMART_QUOTE_RE = re.compile(r'[“„]([^”“\n]+)[”“]')
_DIRECT_GERMAN_IMAGE_PREFIX_RE = re.compile(
    r'^\s*(?:male|zeichne)\s+(?:(?:mir|bitte|doch|mal|schnell)\s+){0,4}',
    re.IGNORECASE,
)
_DIRECT_ENGLISH_IMAGE_PREFIX_RE = re.compile(
    r'^\s*(?:generate|create|make|produce|render|draw|illustrate|paint|sketch|visualize)\s+'
    r'(?:(?:me|us)\s+)?(?:(?:an?|the)\s+)?'
    r'(?:(?:image|img|photo|picture|portrait|scene|shot|illustration|artwork|drawing|painting)\s+)?'
    r'(?:of\s+)?',
    re.IGNORECASE,
)
_FOLLOW_UP_IMAGE_PREFIX_RE = re.compile(
    r'^\s*(?:mach(?:e)?\s+(?:daraus|draus)|wandle(?:\s+das)?\s+(?:in|zu)|transformiere(?:\s+das)?\s+(?:in|zu)|'
    r'make this into|turn this into|transform this into|convert this into|weiterverarbeitung(?:\s+(?:als|zu|zum|zur))?)\s+',
    re.IGNORECASE,
)
_TRIMMABLE_IMAGE_PREFIX_RE = re.compile(r'^[,;:\-.\s]+')
_IMAGE_REFERENCE_OF_RE = re.compile(r'\bof\s+(?:it|this|that|them)\b', re.IGNORECASE)
_IMAGE_REFERENCE_PLAIN_RE = re.compile(
    r'\b(?:same subject|same one|same animal|same character|same scene)\b',
    re.IGNORECASE,
)
_IMAGE_REFERENCE_OBJECT_RE = re.compile(r'\b(?:it|this|that|them|him|her)\b', re.IGNORECASE)
_IMAGE_CONTINUATION_PREFIX_RE = re.compile(
    r'^\s*(?:make|turn|render|draw|paint|push|keep|mache?|mach|halte|machs?)\b',
    re.IGNORECASE,
)
_IMAGE_EDIT_CONTINUATION_RE = re.compile(
    r'\b('
    r'add|remove|keep|change|adjust|move|reposition|refine|improve|stronger|more|less|'
    r'powerful|cinematic|dramatic|hero|superhero|weapon|weapons|background|foreground|'
    r'main character|subject|hold|holding|wield|pose|scene|picture|image|make sure|ensure|'
    r'specific prompt|before generating|instead|not just|wear|wearing|dress|outfit|clothes|'
    r'clothing|shirt|jacket|coat|cape|gown|armor|armour|hair|hairstyle|eyes?|face|skin|'
    r'color|colour|blue|red|green|black|white|gold|golden|silver'
    r')\b',
    re.IGNORECASE,
)
_STRUCTURED_IMAGE_EDIT_PREFIX_RE = re.compile(
    r'^\s*(?:make|turn|change|keep|convert|transform|adjust|give|set|mache?|mach|wandle|transformiere|gib)\b',
    re.IGNORECASE,
)
_STRUCTURED_IMAGE_EDIT_DETAIL_RE = re.compile(
    r'\b(robot|humanoid|character|subject|figure|alien|entity|creature|monster|human|person|man|woman|'
    r'girl|boy|he|she|him|her|blue|red|green|gold|golden|silver|black|white|armor|armour|eyes?|'
    r'face|body|scene|background|style|weapon|weapons|wear|wearing|dress|outfit|clothes|clothing|'
    r'shirt|jacket|coat|cape|gown|hair|hairstyle|skin|color|colour)\b',
    re.IGNORECASE,
)

_LANGUAGE_CONTENT_HINTS: dict[str, dict[str, Any]] = {
    'de': {
        'words': ('hallo', 'welt', 'guten', 'morgen', 'abend', 'zug', 'fahrt', 'sieben', 'bitte', 'der', 'die', 'das', 'und'),
        'chars': 'äöüß',
    },
    'en': {
        'words': ('hello', 'world', 'welcome', 'train', 'leaves', 'seven', 'please', 'the', 'and'),
        'chars': '',
    },
    'fr': {
        'words': ('bonjour', 'monde', 'voici', 'merci', 'train', 'sept', 'avec', 'une'),
        'chars': 'çœ',
    },
    'es': {
        'words': ('hola', 'mundo', 'buenos', 'dias', 'tarde', 'tren', 'siete', 'voz', 'una'),
        'chars': 'ñ¡¿',
    },
    'it': {
        'words': ('ciao', 'mondo', 'buongiorno', 'treno', 'sette', 'voce', 'una', 'con'),
        'chars': '',
    },
    'pt': {
        'words': ('ola', 'mundo', 'bom', 'dia', 'trem', 'sete', 'voz', 'uma'),
        'chars': 'ãõç',
    },
}

_LANGUAGE_INSTRUCT_SENTENCES = {
    'de': 'Speak in German with natural German pronunciation.',
    'en': 'Speak in English with natural English pronunciation.',
    'fr': 'Speak in French with natural French pronunciation.',
    'es': 'Speak in Spanish with natural Spanish pronunciation.',
    'it': 'Speak in Italian with natural Italian pronunciation.',
    'pt': 'Speak in Portuguese with natural Portuguese pronunciation.',
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


def _recent_user_prompt_candidates(
    prompt: str,
    context_messages: Optional[list[dict[str, Any]]],
    *,
    limit: int = 4,
) -> list[str]:
    current_prompt = str(prompt or '').strip()
    if not isinstance(context_messages, list):
        return []
    candidates: list[str] = []
    for raw in reversed(context_messages):
        if not isinstance(raw, dict):
            continue
        if str(raw.get('role') or '').strip().lower() != 'user':
            continue
        content = str(raw.get('content') or '').strip()
        if not content or content == current_prompt or content in candidates:
            continue
        candidates.append(content)
        if len(candidates) >= limit:
            break
    return candidates


def _detect_language_from_content(text: str) -> Optional[str]:
    raw = str(text or '').strip()
    if not raw:
        return None
    normalized = normalize_intent_text(raw)
    if not normalized:
        return None
    scores: list[tuple[int, str]] = []
    lowered_raw = raw.lower()
    for code, hints in _LANGUAGE_CONTENT_HINTS.items():
        score = 0
        special_chars = str(hints.get('chars') or '')
        if special_chars and any(char in lowered_raw for char in special_chars):
            score += 2
        for marker in hints.get('words') or ():
            if re.search(r'(?<!\w)' + re.escape(str(marker)) + r'(?!\w)', normalized):
                score += 1
        if score > 0:
            scores.append((score, code))
    if not scores:
        return None
    scores.sort(reverse=True)
    best_score, best_code = scores[0]
    if best_score < 2:
        return None
    if len(scores) > 1 and best_score == scores[1][0]:
        return None
    return best_code


def _build_language_instruct(lang_code: Optional[str]) -> Optional[str]:
    token = str(lang_code or '').strip().lower()
    if not token:
        return None
    return _LANGUAGE_INSTRUCT_SENTENCES.get(token, f'Speak in {token.upper()} with natural pronunciation.')


def _build_voice_instruct_clause(descriptors: list[str]) -> Optional[str]:
    if not descriptors:
        return None
    style_tokens = [token for token in descriptors if token not in {'female', 'male'}]
    gender_token = next((token for token in descriptors if token in {'female', 'male'}), None)
    parts = style_tokens[:]
    if 'urgent' in parts and 'serious' not in parts:
        parts.insert(0, 'serious')
    if gender_token == 'female':
        parts.append('clearly female')
    elif gender_token == 'male':
        parts.append('clearly male')
    if not parts:
        return None
    clause = f'Use a {", ".join(parts)} voice.'
    if any(token in descriptors for token in {'serious', 'urgent'}):
        clause = f'{clause} Avoid laughter, smiling, or playful delivery.'
    return clause


def _rewrite_image_prompt_locally(prompt: str) -> Optional[str]:
    raw = str(prompt or '').strip()
    if not raw:
        return None
    rewritten = _DIRECT_GERMAN_IMAGE_PREFIX_RE.sub('', raw)
    if rewritten == raw:
        rewritten = _FOLLOW_UP_IMAGE_PREFIX_RE.sub('', raw)
    rewritten = _TRIMMABLE_IMAGE_PREFIX_RE.sub('', rewritten).strip()
    if not rewritten or rewritten == raw:
        return None
    return rewritten


def _extract_image_subject_phrase(prompt: str) -> Optional[str]:
    raw = str(prompt or '').strip()
    if not raw:
        return None
    quoted: list[str] = []
    for pattern in (_DOUBLE_QUOTE_RE, _SMART_QUOTE_RE):
        for match in pattern.findall(raw):
            candidate = str(match or '').strip()
            if candidate:
                quoted.append(candidate)
    if quoted:
        return ' '.join(quoted).strip()
    rewritten = _DIRECT_GERMAN_IMAGE_PREFIX_RE.sub('', raw)
    if rewritten == raw:
        rewritten = _DIRECT_ENGLISH_IMAGE_PREFIX_RE.sub('', raw)
    rewritten = _TRIMMABLE_IMAGE_PREFIX_RE.sub('', rewritten).strip()
    if rewritten and rewritten != raw:
        return rewritten
    return raw


def _is_contextual_image_follow_up(prompt: str) -> bool:
    raw = str(prompt or '').strip()
    if not raw:
        return False
    if _FOLLOW_UP_IMAGE_PREFIX_RE.search(raw):
        return True
    if _IMAGE_REFERENCE_OF_RE.search(raw) or _IMAGE_REFERENCE_PLAIN_RE.search(raw):
        return True
    if _IMAGE_CONTINUATION_PREFIX_RE.search(raw) and _IMAGE_REFERENCE_OBJECT_RE.search(raw):
        return True
    return bool(_IMAGE_EDIT_CONTINUATION_RE.search(raw))


def _has_direct_image_command_prefix(prompt: str) -> bool:
    raw = str(prompt or '').strip()
    if not raw:
        return False
    return bool(_DIRECT_GERMAN_IMAGE_PREFIX_RE.search(raw) or _DIRECT_ENGLISH_IMAGE_PREFIX_RE.search(raw))


def _merge_subject_into_image_follow_up(prompt: str, subject: str) -> Optional[str]:
    raw = str(prompt or '').strip()
    cleaned_subject = str(subject or '').strip().rstrip(' ,.;:')
    if not raw or not cleaned_subject:
        return None
    replaced = _IMAGE_REFERENCE_OF_RE.sub(f'of {cleaned_subject}', raw, count=1)
    if replaced != raw:
        return replaced
    replaced = _IMAGE_REFERENCE_PLAIN_RE.sub(cleaned_subject, raw, count=1)
    if replaced != raw:
        return replaced
    if _IMAGE_CONTINUATION_PREFIX_RE.search(raw) and _IMAGE_REFERENCE_OBJECT_RE.search(raw):
        replaced = _IMAGE_REFERENCE_OBJECT_RE.sub(cleaned_subject, raw, count=1)
        if replaced != raw:
            return replaced
    return None


def _contextual_image_anchor_from_artifacts(context_messages: Optional[list[dict[str, Any]]]) -> Optional[str]:
    image_state = _contextual_image_state_from_artifacts(context_messages)
    if image_state:
        return image_state_anchor(image_state)
    return None


def _contextual_image_state_from_artifacts(context_messages: Optional[list[dict[str, Any]]]) -> Optional[dict[str, Any]]:
    if isinstance(context_messages, list):
        for raw in reversed(context_messages):
            if not isinstance(raw, dict):
                continue
            artifacts = raw.get('artifacts')
            if not isinstance(artifacts, list):
                continue
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                if str(artifact.get('type') or '').strip() != 'image':
                    continue
                image_state = artifact.get('image_state')
                if isinstance(image_state, dict) and image_state:
                    return image_state
    return None


def _build_structured_image_edit_prompt(prompt: str, image_state: dict[str, Any]) -> Optional[str]:
    raw = str(prompt or '').strip()
    if not raw or not isinstance(image_state, dict):
        return None
    if not _STRUCTURED_IMAGE_EDIT_PREFIX_RE.search(raw):
        return None
    if not _STRUCTURED_IMAGE_EDIT_DETAIL_RE.search(raw):
        return None
    subject = str(image_state.get('subject') or image_state.get('summary') or '').strip().rstrip(' ,.;:')
    scene = str(image_state.get('scene') or '').strip().rstrip(' ,.;:')
    style = str(image_state.get('style') or '').strip().rstrip(' ,.;:')
    if not subject:
        return None
    parts = [f'Base subject: {subject}.']
    keep_parts: list[str] = [subject]
    if scene:
        keep_parts.append(scene)
    if style:
        keep_parts.append(style)
    parts.append(f'Keep unchanged: {"; ".join(keep_parts)}.')
    parts.append(f'Requested change: {raw}.')
    parts.append('Preserve the same subject identity and scene unless the request explicitly changes them.')
    return ' '.join(parts)


def _contextual_image_subject_from_user_history(
    prompt: str,
    context_messages: Optional[list[dict[str, Any]]],
) -> Optional[str]:
    fallback_subject: Optional[str] = None
    for candidate in _recent_user_prompt_candidates(prompt, context_messages):
        analysis = analyze_prompt_intent(candidate)
        image_score = int((analysis.get('capability_scores') or {}).get('image_generation') or 0)
        if image_score < 4 and not _has_direct_image_command_prefix(candidate):
            continue
        subject = _extract_image_subject_phrase(candidate)
        if not subject:
            continue
        subject = _TRIMMABLE_IMAGE_PREFIX_RE.sub('', subject).strip()
        if not subject:
            continue
        if not _is_contextual_image_follow_up(candidate):
            return subject
        if not fallback_subject:
            fallback_subject = subject
    return fallback_subject


def infer_image_prompt_rewrite(
    prompt: str,
    *,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    raw = str(prompt or '').strip()
    if not raw:
        return None
    rewritten = _rewrite_image_prompt_locally(raw)
    image_state = _contextual_image_state_from_artifacts(context_messages)
    image_anchor = _contextual_image_anchor_from_artifacts(context_messages)
    subject = _contextual_image_subject_from_user_history(raw, context_messages)
    if not rewritten:
        structured_edit = _build_structured_image_edit_prompt(raw, image_state or {})
        if structured_edit and normalize_intent_text(structured_edit) != normalize_intent_text(raw):
            return structured_edit
        if not _is_contextual_image_follow_up(raw):
            return None
        merged = _merge_subject_into_image_follow_up(raw, subject or '')
        if merged:
            if normalize_intent_text(merged) == normalize_intent_text(raw):
                return None
            return merged
        descriptive_anchor = str(image_anchor or '').strip().rstrip(' ,.;:')
        if descriptive_anchor:
            normalized_raw = normalize_intent_text(raw)
            normalized_anchor = normalize_intent_text(descriptive_anchor)
            if normalized_anchor and normalized_anchor in normalized_raw:
                return None
            merged = f'{descriptive_anchor}, {raw}'
        else:
            if not subject:
                return None
            normalized_raw = normalize_intent_text(raw)
            normalized_subject = normalize_intent_text(subject)
            if normalized_subject and normalized_subject in normalized_raw:
                return None
            merged = f'{subject}, {raw}'
        if normalize_intent_text(merged) == normalize_intent_text(raw):
            return None
        return merged
    is_follow_up = bool(_FOLLOW_UP_IMAGE_PREFIX_RE.search(raw))
    if not is_follow_up:
        return rewritten
    subject = str(subject or image_anchor or '').strip().rstrip(' ,.;:')
    if not subject:
        return rewritten
    normalized_rewritten = normalize_intent_text(rewritten)
    normalized_subject = normalize_intent_text(subject)
    if normalized_rewritten and normalized_rewritten in normalized_subject:
        return subject
    return f'{subject}, {rewritten}'


def infer_image_dimensions_from_prompt(prompt: str) -> Optional[dict[str, Any]]:
    text = str(prompt or '').strip()
    if not text:
        return None
    ratio = infer_image_aspect_ratio(text)
    if ratio:
        dims = IMAGE_ASPECT_PRESET_DIMENSIONS.get(ratio)
        if dims:
            return {'aspect_ratio': ratio, **dims}
    explicit = _EXPLICIT_ASPECT_RE.search(text)
    if explicit:
        fallback_ratio = explicit.group(1)
        dims = IMAGE_ASPECT_PRESET_DIMENSIONS.get(fallback_ratio)
        if dims:
            return {'aspect_ratio': fallback_ratio, **dims}
    if _SQUARE_RE.search(text):
        return {'aspect_ratio': '1:1', **IMAGE_ASPECT_PRESET_DIMENSIONS['1:1']}
    if _VERTICAL_RE.search(text):
        return {'aspect_ratio': '9:16', **IMAGE_ASPECT_PRESET_DIMENSIONS['9:16']}
    if _PORTRAIT_RE.search(text):
        return {'aspect_ratio': '3:4', **IMAGE_ASPECT_PRESET_DIMENSIONS['3:4']}
    if _LANDSCAPE_RE.search(text):
        return {'aspect_ratio': '16:9', **IMAGE_ASPECT_PRESET_DIMENSIONS['16:9']}
    return None


def infer_tts_language_from_prompt(
    prompt: str,
    *,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    analysis = analyze_prompt_intent(prompt)
    languages = analysis.get('language_codes') if isinstance(analysis.get('language_codes'), list) else []
    if languages:
        return str(languages[0]).strip()
    detected = _detect_language_from_content(extract_tts_source_text(prompt))
    if detected:
        return detected
    for candidate in _recent_user_prompt_candidates(prompt, context_messages):
        candidate_analysis = analyze_prompt_intent(candidate)
        candidate_languages = candidate_analysis.get('language_codes') if isinstance(candidate_analysis.get('language_codes'), list) else []
        if candidate_languages:
            return str(candidate_languages[0]).strip()
        detected = _detect_language_from_content(extract_tts_source_text(candidate))
        if detected:
            return detected
    return None


def infer_tts_speaker_from_prompt(prompt: str, speakers: list[str]) -> Optional[str]:
    text = str(prompt or '').strip().lower()
    if not text:
        return None
    for speaker in speakers:
        token = str(speaker or '').strip()
        if not token:
            continue
        if re.search(r'(?<!\w)' + re.escape(token.lower()) + r'(?!\w)', text):
            return token
    return None


def infer_tts_instruct_from_prompt(
    prompt: str,
    *,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    analysis = analyze_prompt_intent(prompt)
    raw_descriptors = analysis.get('voice_descriptors') if isinstance(analysis.get('voice_descriptors'), list) else []
    descriptors = [str(item).strip() for item in raw_descriptors if str(item).strip()]
    if not descriptors:
        for candidate in _recent_user_prompt_candidates(prompt, context_messages):
            candidate_descriptors = analyze_prompt_intent(candidate).get('voice_descriptors')
            if isinstance(candidate_descriptors, list):
                descriptors = [str(item).strip() for item in candidate_descriptors if str(item).strip()]
            if descriptors:
                break
    lang_code = infer_tts_language_from_prompt(prompt, context_messages=context_messages)
    parts: list[str] = []
    language_clause = _build_language_instruct(lang_code)
    if language_clause:
        parts.append(language_clause)
    voice_clause = _build_voice_instruct_clause(descriptors)
    if voice_clause:
        parts.append(voice_clause)
    if not parts:
        return None
    return ' '.join(parts)


def apply_prompt_control_hints(
    payload: dict[str, Any],
    *,
    capability: str,
    instance: Optional[dict[str, Any]] = None,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(payload or {})
    prompt = str(updated.get('prompt') or updated.get('_prompt_hint') or '').strip()
    applied: dict[str, Any] = {}
    if not prompt:
        return updated, applied

    if capability == 'image_generation':
        rewritten_prompt = infer_image_prompt_rewrite(prompt, context_messages=context_messages)
        if rewritten_prompt:
            updated['prompt'] = rewritten_prompt
            applied['prompt_rewrite'] = rewritten_prompt
        has_width = updated.get('width') not in (None, '')
        has_height = updated.get('height') not in (None, '')
        if not has_width and not has_height:
            dims = infer_image_dimensions_from_prompt(prompt)
            if dims:
                updated['width'] = dims['width']
                updated['height'] = dims['height']
                applied['image_dimensions'] = dims

    if capability == 'text_to_speech':
        speakers = _normalized_text_list((instance or {}).get('tts_speakers'))
        session_controls = (instance or {}).get('session_controls') if isinstance((instance or {}).get('session_controls'), dict) else {}
        session_fields = session_controls.get('fields') if isinstance(session_controls.get('fields'), dict) else {}
        model_type = str((instance or {}).get('tts_model_type') or '').strip().lower()
        if updated.get('voice') in (None, ''):
            speaker = infer_tts_speaker_from_prompt(prompt, speakers)
            if speaker:
                updated['voice'] = speaker
                applied['voice'] = speaker
            elif model_type == 'kitten_tts' and speakers:
                updated['voice'] = speakers[0]
                applied['voice'] = speakers[0]
        if updated.get('lang_code') in (None, ''):
            lang_code = infer_tts_language_from_prompt(prompt, context_messages=context_messages)
            if lang_code:
                updated['lang_code'] = lang_code
                applied['lang_code'] = lang_code
        if updated.get('response_format') in (None, ''):
            response_format = infer_audio_response_format(prompt)
            if response_format:
                updated['response_format'] = response_format
                applied['response_format'] = response_format
        tts_instruct_field = (
            session_fields.get('tts_instruct')
            if isinstance(session_fields.get('tts_instruct'), dict)
            else {}
        )
        requires_tts_instruct = bool(tts_instruct_field.get('required'))
        can_infer_instruct = 'tts_instruct' in session_fields or str((instance or {}).get('tts_model_type') or '').strip().lower() in {'voice_design', 'custom_voice'}
        if can_infer_instruct and updated.get('instruct') in (None, ''):
            instruct = infer_tts_instruct_from_prompt(prompt, context_messages=context_messages)
            if instruct:
                updated['instruct'] = instruct
                applied['instruct'] = instruct
            elif model_type == 'voice_design' or requires_tts_instruct:
                updated['instruct'] = 'Use a natural, conversational voice.'
                applied['instruct'] = updated['instruct']

    return updated, applied
