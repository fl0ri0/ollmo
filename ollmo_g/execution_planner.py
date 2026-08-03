"""Compatibility-named execution planner that acts as Ollmo's local execution resolver for compound routed requests."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from helpers.model_capabilities import CAPABILITY_CHAT, normalize_backend, normalize_capability
from ollmo_g.control_hints import (
    infer_tts_instruct_from_prompt,
    infer_tts_language_from_prompt,
    infer_tts_speaker_from_prompt,
)
from ollmo_g.image_state import image_state_anchor
from ollmo_g.intent import analyze_prompt_intent, infer_audio_response_format
from ollmo_g.request_phase_graph import (
    build_request_phase_graph,
    downstream_phase_capabilities,
    downstream_phase_records,
)
from ollmo_g.request_meta import attach_request_meta, compact_request_meta, extract_request_meta
from ollmo_g.router import select_router_instance
from ollmo_services.tts_source import extract_explicit_tts_source_text

CAPABILITY_TEXT_TO_SPEECH = 'text_to_speech'
CAPABILITY_IMAGE_GENERATION = 'image_generation'

_FENCED_JSON_RE = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r'(\{[\s\S]*\})')
_TTS_COMPOUND_RE = re.compile(
    r'\b('
    r'imagine|invent|make up|come up with|draft|write|compose|caption|'
    r'something (?:he|she|they|it) would say|what would (?:he|she|they|it) say|'
    r'then generate (?:an )?audio|then make (?:an )?audio|then read it aloud|'
    r'and generate (?:an )?audio|audio clip of it|voice clip of it'
    r')\b',
    re.IGNORECASE,
)
_IMAGE_COMPOUND_RE = re.compile(
    r'\b('
    r'write (?:a |the )?(?:better |stronger |specific )?prompt|'
    r'specific prompt|before generating|before you generate|then generate|'
    r'then make (?:an |a )?(?:image|poster|banner|cover)|'
    r'turn this into|make this into|convert this into|poster of it|cover art of it'
    r')\b',
    re.IGNORECASE,
)
_NEW_IMAGE_RESET_RE = re.compile(
    r'\b('
    r'new image|another image|different image|from scratch|brand new|fresh scene|new scene|'
    r'nouvelle image|nueva imagen|neues bild|anderes bild'
    r')\b',
    re.IGNORECASE,
)
_TTS_SPEAKER_REQUEST_RE = re.compile(
    r'\b(custom voice|my voice|voice clone|clone voice|speaker|voice\s+[A-Z][a-z]+)\b',
    re.IGNORECASE,
)
_TTS_READ_REFERENCE_RE = re.compile(
    r'\b(read|speak|say|voice|audio|aloud)\b',
    re.IGNORECASE,
)
_TTS_REFERENCE_TARGET_RE = re.compile(
    r'\b('
    r'last reply|last response|last answer|previous reply|previous response|previous answer|'
    r'summary of the last reply|summary of the last response|summary of the previous reply|'
    r'pinned reply|pinned response|pinned reference|referenced reply|referenced response|'
    r'that reply|that response|that answer|that summary|the summary|'
    r'this|it|that story|that text|that message|that one'
    r')\b',
    re.IGNORECASE,
)
_TTS_PRIOR_ASSISTANT_REFERENCE_RE = re.compile(
    r'\b('
    r'last reply|last response|last answer|previous reply|previous response|previous answer|'
    r'pinned reply|pinned response|pinned reference|referenced reply|referenced response|'
    r'that(?:\s+exact)? reply|that(?:\s+exact)? response|that(?:\s+exact)? answer|'
    r'that message|that text'
    r')\b',
    re.IGNORECASE,
)
_PINNED_ASSISTANT_REFERENCE_PREFIX = 'Selected prior message reference for this conversation turn.'
_PINNED_ASSISTANT_REFERENCE_PREFIXES = (
    _PINNED_ASSISTANT_REFERENCE_PREFIX,
    'Selected prior assistant reply reference for this conversation turn.',
)
_DEFERRED_HANDOFF_ACK_RE = re.compile(
    r'(?is)^\s*(?:ok(?:ay)?[,.! ]+|sure[,.! ]+|certainly[,.! ]+|of course[,.! ]+|yes[,.! ]+)?'
    r'(?:i\s+(?:can|will|would|could|shall|am happy to|\'ll)\b|let me\b|happy to\b|glad to\b)'
    r'.{0,220}\b(?:prepare|read|hear|show|generate|do that|do this|help)\b.{0,220}$'
)
_TTS_STAGE_DIRECTION_RE = re.compile(
    r'(?im)^[ \t]*(?:[*_#>\-]+[ \t]*)*(?:\(|\[)?[ \t]*(?:i\s+will\s+now\s+)?'
    r'(?:read(?:ing)?|speak(?:ing)?|say(?:ing)?|voice(?:ing)?)\b'
    r'[^\n]{0,160}?(?:aloud|translation|version|story|reply)\b[^\n]{0,80}'
    r'[ \t]*(?:\)|\])?[ \t]*:?[ \t]*(?:[*_]+)?[ \t]*$'
)
_SEMANTIC_TTS_SECTION_TOKENS = (
    'tts-ready',
    'reading version',
    'spoken version',
    'narration',
    'english translation',
    'translation to english',
    'english version',
    'read aloud',
)
_TTS_META_NOTE_HINT_RE = re.compile(
    r'(?i)\b('
    r'self-correction|note|as an ai|as a text[- ]based ai|traditional sense|'
    r'i cannot\s+(?:read|speak)|i can\'t\s+(?:read|speak)|'
    r'i will present the text as if|fulfilling the instruction'
    r')\b'
)
_IMAGE_PROMPT_LINE_HINT_RE = re.compile(
    r'(?i)\b('
    r'here is your prompt|image generation prompt|image prompt|final image prompt|'
    r'prompt for the image|prompt for image generation|visual prompt|poster prompt|'
    r'bild\s*[- ]?\s*prompt|bildprompt|prompt\s+f(?:ü|ue)r\s+(?:das\s+)?bild'
    r')\b'
)
_BLOCKQUOTE_LINE_RE = re.compile(r'^\s*>\s?')
_IMAGE_META_NOTE_HINT_RE = re.compile(
    r'(?i)\b('
    r'as an ai(?: language model)?|as a text[- ]based ai|'
    r'i cannot physically generate (?:the )?image|i can\'t physically generate (?:the )?image|'
    r'i cannot generate (?:the )?image|i can\'t generate (?:the )?image|'
    r'ready-to-use prompt|copy and paste|bring this vision to life'
    r')\b'
)
_INLINE_IMAGE_PROMPT_CAPSULE_RE = re.compile(
    r'(?is)\[\s*(?:image generation prompt|prompt for image generation|image prompt|final image prompt|'
    r'visual prompt|poster prompt|bild\s*[- ]?\s*prompt|bildprompt|'
    r'prompt\s+f(?:ü|ue)r\s+(?:das\s+)?bild)\s*:\s*(.+?)\s*\]'
)
_QUOTED_IMAGE_PROMPT_SECTION_RE = re.compile(
    r'(?is)\b(?:image generation prompt|prompt for image generation|image prompt|final image prompt|'
    r'visual prompt|poster prompt|bild\s*[- ]?\s*prompt|bildprompt|'
    r'prompt\s+f(?:ü|ue)r\s+(?:das\s+)?bild|prompt)\b'
    r'\s*:\s*(?:[*_`]+\s*)*(?:\n\s*)*[\"“„«](.+?)[\"”»]'
)
_PLAIN_IMAGE_PROMPT_SECTION_RE = re.compile(
    r'(?is)\b(?:image generation prompt|prompt for image generation|image prompt|final image prompt|'
    r'visual prompt|poster prompt|bild\s*[- ]?\s*prompt|bildprompt|'
    r'prompt\s+f(?:ü|ue)r\s+(?:das\s+)?bild|prompt)\b'
    r'\s*:\s*(?:[*_`]+\s*)*(?:\n\s*)*(.+?)(?:\n\s*\n|$)'
)
_ACTION_PROMPT_KEYS = ('action_input', 'prompt', 'input', 'text', 'content')
_IMAGE_ACTION_ALIASES = {
    'image',
    'image_generation',
    'generate_image',
    'generate_images',
    'create_image',
    'create_images',
    'render_image',
    'render_images',
}


def _strip_markdown_wrappers(value: str) -> str:
    text = str(value or '').strip()
    while text:
        updated = text
        for prefix, suffix in (('**', '**'), ('__', '__'), ('*', '*'), ('_', '_'), ('`', '`')):
            if updated.startswith(prefix) and updated.endswith(suffix) and len(updated) > (len(prefix) + len(suffix)):
                updated = updated[len(prefix):-len(suffix)].strip()
        if updated == text:
            break
        text = updated
    return text


def _is_markdown_separator_line(value: str) -> bool:
    return bool(re.fullmatch(r'(?:\*{3,}|-{3,}|_{3,})', str(value or '').strip()))


def _heading_label(value: str) -> Optional[str]:
    text = _strip_markdown_wrappers(value)
    if not text.endswith(':'):
        return None
    label = text[:-1].strip()
    if not label or len(label) > 96:
        return None
    if any(token in label for token in '.!?'):
        return None
    return label


def _extract_markdown_heading_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_label: Optional[str] = None
    current_lines: list[str] = []
    for raw_line in str(text or '').splitlines():
        heading = _heading_label(raw_line)
        if heading:
            if current_label:
                body = '\n'.join(current_lines).strip()
                if body:
                    sections.append((current_label, body))
            current_label = heading
            current_lines = []
            continue
        if current_label is None:
            continue
        if _is_markdown_separator_line(raw_line) and not current_lines:
            continue
        current_lines.append(raw_line)
    if current_label:
        body = '\n'.join(current_lines).strip()
        if body:
            sections.append((current_label, body))
    return sections


def _normalize_action_name(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')


def _coerce_action_prompt(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in _ACTION_PROMPT_KEYS:
            prompt = _coerce_action_prompt(value.get(key))
            if prompt:
                return prompt
    return None


def _extract_visible_action_prompt(
    text: str,
    *,
    action_aliases: set[str],
) -> Optional[str]:
    candidates: list[str] = []
    raw_text = str(text or '').strip()
    if raw_text:
        candidates.append(raw_text)
    for pattern in (_FENCED_JSON_RE, _JSON_OBJECT_RE):
        match = pattern.search(raw_text)
        if match:
            candidates.append(str(match.group(1) or '').strip())
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        action = _normalize_action_name(
            payload.get('action')
            or payload.get('capability')
            or payload.get('tool')
            or payload.get('name')
        )
        if action not in action_aliases:
            continue
        for key in _ACTION_PROMPT_KEYS:
            prompt = _coerce_action_prompt(payload.get(key))
            if prompt:
                return prompt
    return None


def _extract_trailing_tts_meta_note(text: str) -> tuple[Optional[str], Optional[str]]:
    raw = str(text or '').strip()
    if not raw:
        return None, None
    separator_match = re.search(r'(?is)\n(?:\*{3,}|-{3,}|_{3,})\s*\n+(?P<note>.+)$', raw)
    if not separator_match:
        return None, None
    note = str(separator_match.group('note') or '').strip()
    normalized_note = _strip_markdown_wrappers(note).strip()
    normalized_note = normalized_note.strip('()[]')
    if not normalized_note or not _TTS_META_NOTE_HINT_RE.search(normalized_note):
        return None, None
    content_payload = raw[:separator_match.start()].rstrip()
    if not content_payload:
        return None, None
    return content_payload, note


def split_visible_tts_payload(display_text: str) -> dict[str, Optional[str]]:
    text = str(display_text or '').strip()
    payload: dict[str, Optional[str]] = {
        'display_text': text or None,
        'content_payload': None,
        'stage_direction': None,
        'phase_summary': None,
        'content_payload_source': None,
    }
    if not text:
        return payload

    stage_match = None
    for candidate in _TTS_STAGE_DIRECTION_RE.finditer(text):
        stage_match = candidate
    if stage_match:
        stage_direction = _strip_markdown_wrappers(stage_match.group(0))
        prefix = text[:stage_match.start()].rstrip()
        suffix = text[stage_match.end():].strip()
        content_payload = ''
        phase_summary = ''
        source = ''
        if suffix:
            content_payload = suffix
            phase_summary = prefix
            source = 'stage_direction_suffix'
        else:
            prefix_lines = prefix.splitlines()
            while prefix_lines and not prefix_lines[-1].strip():
                prefix_lines.pop()
            if prefix_lines and _is_markdown_separator_line(prefix_lines[-1]):
                prefix_lines.pop()
            content_payload = '\n'.join(prefix_lines).strip()
            source = 'stage_direction_prefix'
        if content_payload:
            if phase_summary:
                summary_lines = phase_summary.splitlines()
                while summary_lines and not summary_lines[-1].strip():
                    summary_lines.pop()
                if summary_lines and _is_markdown_separator_line(summary_lines[-1]):
                    summary_lines.pop()
                phase_summary = '\n'.join(summary_lines).strip()
            payload.update(
                {
                    'content_payload': content_payload,
                    'stage_direction': stage_direction or None,
                    'phase_summary': phase_summary or None,
                    'content_payload_source': source,
                }
            )
            return payload

    for label, body in reversed(_extract_markdown_heading_sections(text)):
        normalized_label = label.lower()
        if any(token in normalized_label for token in _SEMANTIC_TTS_SECTION_TOKENS):
            payload.update(
                {
                    'content_payload': body,
                    'content_payload_source': 'heading_section',
                }
            )
            return payload

    trailing_content_payload, trailing_note = _extract_trailing_tts_meta_note(text)
    if trailing_content_payload:
        payload.update(
            {
                'content_payload': trailing_content_payload,
                'stage_direction': trailing_note or None,
                'content_payload_source': 'trailing_meta_note',
            }
        )
        return payload

    payload.update(
        {
            'content_payload': text,
            'content_payload_source': 'full_display_text',
        }
    )
    return payload


def _join_compact_text_lines(lines: list[str]) -> str:
    compact: list[str] = []
    previous_blank = False
    for raw_line in lines:
        line = str(raw_line or '').rstrip()
        if not line.strip():
            if compact and not previous_blank:
                compact.append('')
            previous_blank = True
            continue
        compact.append(line.strip())
        previous_blank = False
    while compact and not compact[0].strip():
        compact.pop(0)
    while compact and not compact[-1].strip():
        compact.pop()
    return '\n'.join(compact).strip()


def _trim_image_phase_text(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    meta_match = _IMAGE_META_NOTE_HINT_RE.search(raw)
    if meta_match:
        if meta_match.start() > 0:
            raw = raw[:meta_match.start()].rstrip()
        else:
            blocks = re.split(r'\n\s*\n', raw)
            while blocks and (
                _IMAGE_META_NOTE_HINT_RE.search(str(blocks[0] or '').strip())
                or _is_markdown_separator_line(blocks[0])
            ):
                blocks.pop(0)
            raw = '\n\n'.join(str(block or '').strip() for block in blocks if str(block or '').strip()).strip()
    lines = raw.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _is_markdown_separator_line(lines[-1]):
        lines.pop()
    return '\n'.join(lines).strip()


def _extract_blockquote_prompt_after_line(lines: list[str], *, start_index: int) -> Optional[str]:
    if start_index < -1:
        start_index = -1
    index = start_index + 1
    while index < len(lines) and not str(lines[index] or '').strip():
        index += 1
    prompt_lines: list[str] = []
    while index < len(lines):
        raw_line = str(lines[index] or '')
        if _BLOCKQUOTE_LINE_RE.match(raw_line):
            prompt_lines.append(_BLOCKQUOTE_LINE_RE.sub('', raw_line, count=1).rstrip())
            index += 1
            continue
        if prompt_lines and not raw_line.strip():
            prompt_lines.append('')
            index += 1
            continue
        break
    prompt = _join_compact_text_lines(prompt_lines)
    return prompt or None


def _clean_extracted_image_prompt(value: str) -> Optional[str]:
    raw_lines = str(value or '').splitlines()
    if raw_lines and all(
        (not str(line or '').strip()) or _BLOCKQUOTE_LINE_RE.match(str(line or ''))
        for line in raw_lines
    ):
        raw_lines = [
            _BLOCKQUOTE_LINE_RE.sub('', str(line or ''), count=1).rstrip()
            if _BLOCKQUOTE_LINE_RE.match(str(line or ''))
            else str(line or '')
            for line in raw_lines
        ]
    text = _join_compact_text_lines(raw_lines)
    text = _strip_markdown_wrappers(text).strip()
    if (
        len(text) >= 2
        and ((text.startswith('"') and text.endswith('"')) or (text[0] in '“„«' and text[-1] in '”»'))
    ):
        text = text[1:-1].strip()
    return text or None


def _extract_inline_image_prompt_from_line(raw_line: str) -> Optional[str]:
    raw = str(raw_line or '').strip()
    if not raw:
        return None
    capsule_match = _INLINE_IMAGE_PROMPT_CAPSULE_RE.search(raw)
    if capsule_match:
        return _clean_extracted_image_prompt(str(capsule_match.group(1) or ''))
    normalized = _strip_markdown_wrappers(raw).strip()
    normalized = re.sub(r'^#{1,6}\s*', '', normalized).strip()
    normalized = normalized.strip('[]').strip()
    inline_match = re.match(
        r'(?is)^(?:image generation prompt|prompt for image generation|image prompt|final image prompt|'
        r'visual prompt|poster prompt|bild\s*[- ]?\s*prompt|bildprompt|'
        r'prompt\s+f(?:ü|ue)r\s+(?:das\s+)?bild|prompt)\s*:\s*(.+)$',
        normalized,
    )
    if inline_match:
        return _clean_extracted_image_prompt(str(inline_match.group(1) or ''))
    return None


def _extract_prompt_after_hint_line(lines: list[str], *, start_index: int) -> tuple[Optional[str], Optional[str]]:
    inline_prompt = None
    if 0 <= start_index < len(lines):
        inline_prompt = _extract_inline_image_prompt_from_line(str(lines[start_index] or ''))
    if inline_prompt:
        return inline_prompt, 'inline_prompt_capsule'
    blockquote_prompt = _extract_blockquote_prompt_after_line(lines, start_index=start_index)
    if blockquote_prompt:
        return blockquote_prompt, 'prompt_blockquote_section'
    remainder = '\n'.join(str(line or '') for line in lines[start_index + 1:])
    if not remainder.strip():
        return None, None
    quoted_match = _QUOTED_IMAGE_PROMPT_SECTION_RE.search(remainder)
    if quoted_match:
        prompt = _clean_extracted_image_prompt(str(quoted_match.group(1) or ''))
        if prompt:
            return prompt, 'quoted_prompt_section'
    plain_match = _PLAIN_IMAGE_PROMPT_SECTION_RE.search(remainder)
    if plain_match:
        raw_prompt = str(plain_match.group(1) or '')
        prompt = _clean_extracted_image_prompt(raw_prompt)
        if prompt:
            source = 'prompt_blockquote_section' if raw_prompt.lstrip().startswith('>') else 'plain_prompt_section'
            return prompt, source
    return None, None


def split_visible_image_payload(display_text: str) -> dict[str, Optional[str]]:
    text = str(display_text or '').strip()
    payload: dict[str, Optional[str]] = {
        'display_text': text or None,
        'content_payload': None,
        'phase_summary': None,
        'artifact_prompt': None,
        'artifact_prompt_source': None,
    }
    if not text:
        return payload

    action_prompt = _extract_visible_action_prompt(
        text,
        action_aliases=_IMAGE_ACTION_ALIASES,
    )
    if action_prompt:
        payload.update(
            {
                'content_payload': None,
                'phase_summary': action_prompt,
                'artifact_prompt': action_prompt,
                'artifact_prompt_source': 'action_input',
            }
        )
        return payload

    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        normalized_line = _strip_markdown_wrappers(raw_line).strip()
        if not normalized_line:
            continue
        if not _IMAGE_PROMPT_LINE_HINT_RE.search(normalized_line):
            continue
        artifact_prompt, artifact_prompt_source = _extract_prompt_after_hint_line(lines, start_index=index)
        if not artifact_prompt:
            continue
        content_payload = _trim_image_phase_text('\n'.join(lines[:index]))
        payload.update(
            {
                'content_payload': content_payload or None,
                'phase_summary': content_payload or None,
                'artifact_prompt': artifact_prompt,
                'artifact_prompt_source': artifact_prompt_source,
            }
        )
        return payload

    for label, body in reversed(_extract_markdown_heading_sections(text)):
        normalized_label = label.lower()
        if 'prompt' not in normalized_label or 'image' not in normalized_label:
            continue
        artifact_prompt, artifact_prompt_source = _extract_prompt_after_hint_line(body.splitlines(), start_index=-1)
        if not artifact_prompt:
            artifact_prompt = _trim_image_phase_text(body)
            artifact_prompt_source = 'heading_prompt_section' if artifact_prompt else None
        if artifact_prompt:
            payload.update(
                {
                    'artifact_prompt': artifact_prompt,
                    'artifact_prompt_source': artifact_prompt_source,
                }
            )
            return payload

    trimmed_prefix = _trim_image_phase_text(text)
    if trimmed_prefix and trimmed_prefix != text:
        payload.update(
            {
                'content_payload': trimmed_prefix,
                'phase_summary': trimmed_prefix,
                'artifact_prompt': trimmed_prefix,
                'artifact_prompt_source': 'trailing_meta_note',
            }
        )
        return payload

    payload.update(
        {
            'content_payload': text,
            'phase_summary': text,
            'artifact_prompt': text,
            'artifact_prompt_source': 'full_display_text',
        }
    )
    return payload


def _recent_context_messages(messages: Any, *, limit: int = 6) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    items: list[dict[str, str]] = []
    for raw in messages[-limit:]:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get('role') or '').strip().lower()
        content = str(raw.get('content') or '').strip()
        if role not in {'system', 'user', 'assistant'} or not content:
            continue
        items.append({'role': role, 'content': content})
    return items


def _extract_pinned_assistant_reference(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    marker = '[assistant]\n'
    for raw in reversed(messages):
        if not isinstance(raw, dict):
            continue
        if str(raw.get('role') or '').strip().lower() != 'system':
            continue
        content = str(raw.get('content') or '').strip()
        if not any(content.startswith(prefix) for prefix in _PINNED_ASSISTANT_REFERENCE_PREFIXES):
            continue
        if marker in content:
            candidate = content.split(marker, 1)[1].strip()
            if candidate:
                return candidate
    return None


def _extract_latest_assistant_content(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    deferred_handoff_fallback: Optional[str] = None
    for raw in reversed(messages):
        if not isinstance(raw, dict):
            continue
        if str(raw.get('role') or '').strip().lower() != 'assistant':
            continue
        content = str(raw.get('content') or '').strip()
        if not content:
            continue
        if content in {'Image generated.', 'Audio generated.'}:
            continue
        semantic_payload = split_visible_tts_payload(content)
        content_payload = str(semantic_payload.get('content_payload') or '').strip()
        candidate = content_payload or content
        if (
            deferred_handoff_fallback is None
            and len(candidate) <= 260
            and '\n\n' not in candidate
            and _DEFERRED_HANDOFF_ACK_RE.match(candidate)
        ):
            deferred_handoff_fallback = candidate
            continue
        return candidate
    return deferred_handoff_fallback


def _extract_payload_content_payload(payload: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ('content_payload', 'contentPayload', 'spoken_payload', 'spokenPayload'):
        candidate = str(payload.get(key) or '').strip()
        if candidate:
            return candidate
    return None


def _extract_tts_reference_prompt(
    prompt: str,
    *,
    context_messages: Optional[list[dict[str, Any]]] = None,
    require_prior_assistant_reference: bool = False,
) -> Optional[str]:
    raw = str(prompt or '').strip()
    if not raw or extract_explicit_tts_source_text(raw):
        return None
    if not _TTS_READ_REFERENCE_RE.search(raw):
        return None
    pinned_reference = _extract_pinned_assistant_reference(context_messages)
    if pinned_reference:
        return pinned_reference
    reference_pattern = (
        _TTS_PRIOR_ASSISTANT_REFERENCE_RE
        if require_prior_assistant_reference
        else _TTS_REFERENCE_TARGET_RE
    )
    if not reference_pattern.search(raw):
        return None
    return _extract_latest_assistant_content(context_messages)


def _recent_image_state(messages: Any) -> Optional[dict[str, Any]]:
    if not isinstance(messages, list):
        return None
    for raw in reversed(messages):
        if not isinstance(raw, dict):
            continue
        artifacts = raw.get('artifacts')
        if not isinstance(artifacts, list):
            continue
        for artifact in reversed(artifacts):
            if not isinstance(artifact, dict):
                continue
            if str(artifact.get('type') or '').strip() != 'image':
                continue
            image_state = artifact.get('image_state')
            if isinstance(image_state, dict) and image_state:
                return image_state
    return None


def _should_plan_contextual_image_edit(
    prompt: str,
    *,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> bool:
    raw = str(prompt or '').strip()
    if not raw:
        return False
    if _NEW_IMAGE_RESET_RE.search(raw):
        return False
    if not _recent_image_state(context_messages):
        return False
    analysis = analyze_prompt_intent(raw)
    scores = analysis.get('capability_scores') or {}
    if int(scores.get(CAPABILITY_TEXT_TO_SPEECH) or 0) >= 4:
        return False
    if len(raw.split()) > 36:
        return False
    return True


def _extract_json_payload(raw: str) -> Optional[dict[str, Any]]:
    text = str(raw or '').strip()
    if not text:
        return None
    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    generic = _JSON_OBJECT_RE.search(text)
    if generic:
        try:
            parsed = json.loads(generic.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return None


def _requires_voice_design_instruct(instance: dict[str, Any]) -> bool:
    model_type = str(instance.get('tts_model_type') or '').strip().lower()
    if model_type in {'voice_design', 'custom_voice'}:
        return True
    session_controls = instance.get('session_controls') if isinstance(instance.get('session_controls'), dict) else {}
    fields = session_controls.get('fields') if isinstance(session_controls.get('fields'), dict) else {}
    return 'tts_instruct' in fields


def _needs_tts_detail_refinement(prompt: str, payload: dict[str, Any], instance: dict[str, Any]) -> bool:
    raw = str(prompt or '').strip()
    if not raw or not extract_explicit_tts_source_text(raw):
        return False
    analysis = analyze_prompt_intent(raw)
    requested_languages = [
        str(item or '').strip()
        for item in (analysis.get('language_codes') or [])
        if str(item or '').strip()
    ]
    requested_voice_descriptors = [
        str(item or '').strip()
        for item in (analysis.get('voice_descriptors') or [])
        if str(item or '').strip()
    ]
    requested_format = str(analysis.get('audio_response_format') or '').strip()
    model_type = str(instance.get('tts_model_type') or '').strip().lower()
    if requested_languages and payload.get('lang_code') in (None, ''):
        return True
    if requested_format and payload.get('response_format') in (None, ''):
        return True
    if _TTS_SPEAKER_REQUEST_RE.search(raw) and payload.get('voice') in (None, ''):
        return True
    if requested_voice_descriptors and model_type == 'voice_design' and payload.get('instruct') in (None, ''):
        return True
    if requested_voice_descriptors and model_type not in {'voice_design'} and payload.get('voice') in (None, ''):
        return True
    return False


def _needs_compound_planning(
    prompt: str,
    *,
    capability: str,
    payload: Optional[dict[str, Any]] = None,
    instance: Optional[dict[str, Any]] = None,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    raw = str(prompt or '').strip()
    if not raw:
        return None
    capability = normalize_capability(capability)
    analysis = analyze_prompt_intent(raw)
    if capability in {'chat', 'vision_analysis'} and analysis.get('text_preparation_before_audio_output'):
        if normalize_capability(analysis.get('text_first_follow_up_capability')) == CAPABILITY_TEXT_TO_SPEECH:
            return 'text_first_tts_follow_up'
        return 'text_preparation_before_audio_output'
    if capability in {'chat', 'vision_analysis'} and analysis.get('text_preparation_before_visual_output'):
        if normalize_capability(analysis.get('text_first_follow_up_capability')) == CAPABILITY_IMAGE_GENERATION:
            return 'text_first_image_follow_up'
        return 'text_preparation_before_visual_output'
    if capability == CAPABILITY_TEXT_TO_SPEECH:
        if analysis.get('text_preparation_before_audio_output'):
            return 'compound_tts_prompt'
        if extract_explicit_tts_source_text(raw):
            if _needs_tts_detail_refinement(raw, payload or {}, instance or {}):
                return 'tts_detail_refinement'
            return None
        if _TTS_COMPOUND_RE.search(raw):
            return 'compound_tts_prompt'
        return None
    if capability == CAPABILITY_IMAGE_GENERATION:
        if _IMAGE_COMPOUND_RE.search(raw):
            return 'compound_image_prompt'
        if _should_plan_contextual_image_edit(raw, context_messages=context_messages):
            return 'contextual_image_edit'
        return None
    return None


def _apply_phase_graph_follow_up_result(
    payload: dict[str, Any],
    *,
    follow_up_branches: Optional[list[dict[str, Any]]] = None,
    follow_up_capabilities: Optional[list[str]] = None,
    follow_up_capability: Optional[str] = None,
    trigger: str,
    prompt: Optional[str] = None,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(payload or {})
    applied_fields: list[str] = []
    deferred_branches: list[dict[str, Any]] = []
    deferred_capabilities = []
    for raw_branch in follow_up_branches or []:
        if not isinstance(raw_branch, dict):
            continue
        normalized_capability = normalize_capability(raw_branch.get('capability'))
        branch_id = str(raw_branch.get('branch_id') or raw_branch.get('phase_id') or '').strip()
        phase_id = str(raw_branch.get('phase_id') or raw_branch.get('branch_id') or '').strip()
        if not normalized_capability or not (branch_id or phase_id):
            continue
        deferred_branches.append(
            {
                'branch_id': branch_id or phase_id,
                'phase_id': phase_id or branch_id,
                'capability': normalized_capability,
                'output_type': str(raw_branch.get('output_type') or '').strip().lower() or None,
                'depends_on': [
                    str(item).strip()
                    for item in (raw_branch.get('depends_on') or [])
                    if str(item).strip()
                ],
                **{
                    key: raw_branch.get(key)
                    for key in (
                        'queue_index',
                        'artifact_prompt',
                        'artifact_prompt_source',
                        'content_payload',
                        'content_payload_source',
                        'phase_summary',
                        'stage_direction',
                        'batch_prompts',
                    )
                    if raw_branch.get(key) not in (None, '', [], {})
                },
            }
        )
        if normalized_capability not in deferred_capabilities:
            deferred_capabilities.append(normalized_capability)
    if not deferred_branches:
        for candidate in list(follow_up_capabilities or []) + [follow_up_capability]:
            normalized_candidate = normalize_capability(candidate)
            if not normalized_candidate or normalized_candidate in deferred_capabilities:
                continue
            deferred_capabilities.append(normalized_candidate)
    primary_follow_up = deferred_capabilities[0] if deferred_capabilities else None
    if primary_follow_up:
        current_request_meta = extract_request_meta(updated)
        current_hint = normalize_capability((current_request_meta or {}).get('capability_hint'))
        if current_hint in {None, '', CAPABILITY_CHAT}:
            next_request_meta = dict(current_request_meta or {})
            next_request_meta['capability_hint'] = primary_follow_up
            updated = attach_request_meta(
                {
                    **updated,
                    'request_meta': compact_request_meta(next_request_meta),
                }
            )
            applied_fields.append('request_meta.capability_hint')
    planned_prompt = None
    if primary_follow_up == CAPABILITY_TEXT_TO_SPEECH:
        branch_payloads = list(
            dict.fromkeys(
                str(branch.get('content_payload') or '').strip()
                for branch in deferred_branches
                if normalize_capability(branch.get('capability'))
                == CAPABILITY_TEXT_TO_SPEECH
                and str(branch.get('content_payload') or '').strip()
            )
        )
        waits_for_phase_output = any(
            normalize_capability(branch.get('capability'))
            == CAPABILITY_TEXT_TO_SPEECH
            and not str(branch.get('content_payload') or '').strip()
            and bool(branch.get('depends_on'))
            for branch in deferred_branches
        )
        planned_prompt = (
            _extract_payload_content_payload(updated)
            or (branch_payloads[0] if len(branch_payloads) == 1 else None)
            or _extract_tts_reference_prompt(
                str(prompt or '').strip(),
                context_messages=context_messages,
                require_prior_assistant_reference=waits_for_phase_output,
            )
        )
        if not planned_prompt and not waits_for_phase_output:
            planned_prompt = (
                extract_explicit_tts_source_text(str(prompt or '').strip())
                or _extract_latest_assistant_content(context_messages)
            )
        if planned_prompt and str(updated.get('content_payload') or '').strip() != planned_prompt:
            updated['content_payload'] = planned_prompt
            applied_fields.append('content_payload')
    return updated, {
        'attempted': True,
        'applied': bool(applied_fields),
        'planned_prompt': planned_prompt,
        'applied_fields': applied_fields,
        'reason': 'request_phase_graph_follow_up',
        'trigger': trigger,
        'primary_follow_up_branch_id': (
            str((deferred_branches[0] or {}).get('branch_id') or '').strip()
            if deferred_branches
            else None
        ),
        'deferred_branches': deferred_branches,
        'deferred_capability': primary_follow_up,
        'deferred_capabilities': deferred_capabilities,
    }


def _apply_ghost_carried_deferred_result(
    payload: dict[str, Any],
    *,
    prompt: str,
    follow_up_capability: Optional[str],
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(payload or {})
    applied_fields: list[str] = []
    normalized_follow_up = normalize_capability(follow_up_capability)
    if normalized_follow_up:
        current_request_meta = extract_request_meta(updated)
        current_hint = normalize_capability((current_request_meta or {}).get('capability_hint'))
        if current_hint in {None, '', CAPABILITY_CHAT}:
            next_request_meta = dict(current_request_meta or {})
            next_request_meta['capability_hint'] = normalized_follow_up
            updated = attach_request_meta(
                {
                    **updated,
                    'request_meta': compact_request_meta(next_request_meta),
                }
            )
            applied_fields.append('request_meta.capability_hint')

    planned_prompt = None
    if normalized_follow_up == CAPABILITY_TEXT_TO_SPEECH:
        planned_prompt = (
            _extract_payload_content_payload(updated)
            or _extract_tts_reference_prompt(prompt, context_messages=context_messages)
            or extract_explicit_tts_source_text(prompt)
            or _extract_latest_assistant_content(context_messages)
        )
        if planned_prompt and str(updated.get('content_payload') or '').strip() != planned_prompt:
            updated['content_payload'] = planned_prompt
            applied_fields.append('content_payload')

    return updated, {
        'attempted': True,
        'applied': bool(applied_fields),
        'planned_prompt': planned_prompt,
        'applied_fields': applied_fields,
        'reason': 'ghost_carried_deferred_fulfillment',
        'trigger': 'ghost_carried_deferred_fulfillment',
        'deferred_capability': normalized_follow_up or None,
    }


def _planner_instructions(capability: str, instance: dict[str, Any]) -> str:
    return _planner_instructions_with_semantic_roles(capability, instance, semantic_role_profile=None)


def _semantic_role_guidance(semantic_role_profile: Optional[dict[str, Any]]) -> str:
    profile = semantic_role_profile if isinstance(semantic_role_profile, dict) else {}
    orientation = (
        profile.get('semantic_role_orientation')
        if isinstance(profile.get('semantic_role_orientation'), dict)
        else {}
    )
    mode = str(orientation.get('mode') or profile.get('mode') or '').strip()
    mode_source = str(orientation.get('mode_source') or profile.get('mode_source') or '').strip()
    recommended_action = str(
        orientation.get('recommended_action') or orientation.get('reason') or ''
    ).strip()
    boundary = str(
        orientation.get('non_authority_boundary')
        or profile.get('non_authority_boundary')
        or 'semantic_roles_are_advisory_contracts_runtime_and_closure_decide_truth'
    ).strip()
    suggested_lenses = [
        str(item or '').strip()
        for item in orientation.get('suggested_semantic_review_lenses', [])
        if str(item or '').strip()
    ] if isinstance(orientation.get('suggested_semantic_review_lenses'), list) else []
    role_ids = [
        str(item or '').strip()
        for item in profile.get('semantic_role_ids', [])
        if str(item or '').strip()
    ] if isinstance(profile.get('semantic_role_ids'), list) else []
    parts = []
    if mode:
        source = f' from {mode_source}' if mode_source else ''
        parts.append(
            f'Semantic role profile "{mode}"{source} is advisory orientation only, not planner authority.'
        )
    if recommended_action:
        parts.append(f'Orientation suggestion: {recommended_action}.')
    if role_ids:
        parts.append(f'Active semantic roles: {", ".join(role_ids)}.')
    if suggested_lenses:
        parts.append(f'Possible review lenses if no stronger branch-local lens exists: {", ".join(suggested_lenses)}.')
    parts.append(
        'Always follow branch-local semantic review lenses, execution contracts, runtime evidence, and closure truth first.'
    )
    parts.append(f'Non-authority boundary: {boundary}.')
    return ' '.join(parts).strip()


def _planner_instructions_with_semantic_roles(
    capability: str,
    instance: dict[str, Any],
    *,
    semantic_role_profile: Optional[dict[str, Any]],
) -> str:
    mode_guidance = _semantic_role_guidance(semantic_role_profile)
    if capability == CAPABILITY_TEXT_TO_SPEECH:
        instruct_rule = (
            'If the target TTS model requires a style/instruct field and the user did not specify one, '
            'set "instruct" to "Use a natural, conversational voice."'
            if _requires_voice_design_instruct(instance)
            else 'Only set "instruct" if the user clearly requested a specific speaking style.'
        )
        return (
            'You are Ollmo\'s local execution planner. Convert the user request into a single executable '
            'text-to-speech payload. Return JSON only with keys '
            '{"apply": boolean, "planned_prompt": string|null, "instruct": string|null, '
            '"lang_code": string|null, "voice": string|null, "response_format": string|null, "reason": string}. '
            'If planning is needed, "planned_prompt" must contain only the words to be spoken, not instructions '
            'to the synthesizer. Keep it short and natural unless the user explicitly asked for something long. '
            'If the spoken text is already explicit, keep "planned_prompt" null unless it still needs extraction or cleanup, '
            'but you may still set lang_code, voice, response_format, or instruct. '
            'Set "apply" to true whenever any field should be filled or refined. '
            f'{mode_guidance} {instruct_rule} Set "apply" to false only when the request is already fully executable as-is.'
        )
    if capability == CAPABILITY_IMAGE_GENERATION:
        return (
            'You are Ollmo\'s local execution planner. Convert the user request into a single executable '
            'image-generation prompt. Return JSON only with keys '
            '{"apply": boolean, "planned_prompt": string|null, "reason": string}. '
            'If planning is needed, "planned_prompt" must be a single direct image prompt. When recent_image_state '
            'exists and the user is editing or continuing the current image, preserve the current subject, scene, and '
            'style unless the user explicitly changes them. Convert short natural edit requests into a clearer prompt '
            'that distinguishes what to keep from what to change. Set "apply" to false only when the prompt is already '
            f'fully explicit or it clearly asks for a brand-new unrelated image. {mode_guidance}'
        )
    raise ValueError(f'Unsupported capability for planner: {capability}')


def _planner_user_message(
    prompt: str,
    *,
    capability: str,
    payload: dict[str, Any],
    route_info: dict[str, Any],
    context_messages: Optional[list[dict[str, Any]]] = None,
    request_meta: Optional[dict[str, Any]] = None,
    semantic_role_profile: Optional[dict[str, Any]] = None,
) -> str:
    instance = route_info.get('instance') if isinstance(route_info.get('instance'), dict) else {}
    current_fields = {
        key: payload.get(key)
        for key in ('prompt', 'instruct', 'voice', 'lang_code', 'response_format')
        if payload.get(key) not in (None, '')
    }
    recent_image_state = _recent_image_state(context_messages)
    summary = {
        'capability': capability,
        'prompt': prompt,
        'target_instance_id': str(instance.get('instance_id') or '').strip() or None,
        'target_model': str(instance.get('model') or '').strip() or None,
        'target_backend': normalize_backend(instance.get('backend')),
        'tts_model_type': str(instance.get('tts_model_type') or '').strip() or None,
        'required_session_fields': sorted(
            key
            for key, field in ((instance.get('session_controls') or {}).get('fields') or {}).items()
            if isinstance(field, dict) and field.get('required')
        ),
        'current_request_fields': current_fields,
        'request_meta': compact_request_meta(request_meta),
        'semantic_role_profile': {
            'mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
            'summary': str((semantic_role_profile or {}).get('summary') or '').strip() or None,
            'semantic_role_ids': (
                (semantic_role_profile or {}).get('semantic_role_ids')
                if isinstance((semantic_role_profile or {}).get('semantic_role_ids'), list)
                else []
            ),
            'semantic_role_orientation': (
                (semantic_role_profile or {}).get('semantic_role_orientation')
                if isinstance((semantic_role_profile or {}).get('semantic_role_orientation'), dict)
                else {}
            ),
        },
        'recent_context': _recent_context_messages(context_messages),
        'recent_image_state': recent_image_state,
        'recent_image_anchor': image_state_anchor(recent_image_state) if isinstance(recent_image_state, dict) else None,
    }
    return json.dumps(summary, ensure_ascii=True)


def _apply_planner_result(
    payload: dict[str, Any],
    *,
    capability: str,
    instance: dict[str, Any],
    planned: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(payload or {})
    applied_fields: list[str] = []
    planned_prompt = str(planned.get('planned_prompt') or '').strip()
    if planned_prompt:
        updated['prompt'] = planned_prompt
        updated['_prompt_hint'] = planned_prompt
        applied_fields.append('prompt')

    if capability == CAPABILITY_TEXT_TO_SPEECH:
        for field_name in ('instruct', 'voice', 'lang_code', 'response_format'):
            if updated.get(field_name) not in (None, ''):
                continue
            value = str(planned.get(field_name) or '').strip()
            if value:
                updated[field_name] = value
                applied_fields.append(field_name)
        if _requires_voice_design_instruct(instance) and updated.get('instruct') in (None, ''):
            updated['instruct'] = 'Use a natural, conversational voice.'
            applied_fields.append('instruct')

    return updated, {
        'applied': bool(applied_fields),
        'planned_prompt': planned_prompt or None,
        'applied_fields': applied_fields,
        'reason': str(planned.get('reason') or '').strip() or 'compound_request_planned',
    }


def _apply_local_tts_reference_result(
    payload: dict[str, Any],
    *,
    prompt: str,
    instance: dict[str, Any],
    referenced_content: str,
    context_messages: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(payload or {})
    updated['prompt'] = referenced_content
    updated['_prompt_hint'] = prompt
    applied_fields = ['prompt']

    session_controls = instance.get('session_controls') if isinstance(instance.get('session_controls'), dict) else {}
    session_fields = session_controls.get('fields') if isinstance(session_controls.get('fields'), dict) else {}
    speakers = [
        str(item).strip()
        for item in ((instance or {}).get('tts_speakers') or [])
        if str(item).strip()
    ]

    if updated.get('voice') in (None, '') and speakers:
        speaker = infer_tts_speaker_from_prompt(prompt, speakers)
        if speaker:
            updated['voice'] = speaker
            applied_fields.append('voice')
    if updated.get('lang_code') in (None, ''):
        lang_code = infer_tts_language_from_prompt(prompt, context_messages=context_messages)
        if lang_code:
            updated['lang_code'] = lang_code
            applied_fields.append('lang_code')
    if updated.get('response_format') in (None, ''):
        response_format = infer_audio_response_format(prompt)
        if response_format:
            updated['response_format'] = response_format
            applied_fields.append('response_format')
    can_infer_instruct = 'tts_instruct' in session_fields or _requires_voice_design_instruct(instance)
    if can_infer_instruct and updated.get('instruct') in (None, ''):
        instruct = infer_tts_instruct_from_prompt(prompt, context_messages=context_messages)
        if instruct:
            updated['instruct'] = instruct
            applied_fields.append('instruct')
        elif _requires_voice_design_instruct(instance):
            updated['instruct'] = 'Use a natural, conversational voice.'
            applied_fields.append('instruct')

    return updated, {
        'attempted': True,
        'applied': True,
        'planned_prompt': referenced_content,
        'applied_fields': applied_fields,
        'reason': 'referenced_reply_readaloud',
        'trigger': 'tts_reference_read',
    }


def plan_compound_execution(
    payload: dict[str, Any],
    *,
    route_info: dict[str, Any],
    instances: list[dict[str, Any]],
    context_messages: Optional[list[dict[str, Any]]] = None,
    execute_chat_request: Callable[..., str],
    planner_timeout_sec: Optional[int] = None,
    semantic_role_profile: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_payload = dict(payload or {})
    request_meta = extract_request_meta(base_payload)
    prompt = str(
        base_payload.get('_current_turn_prompt')
        or base_payload.get('prompt')
        or base_payload.get('_prompt_hint')
        or ''
    ).strip()
    instance = route_info.get('instance') if isinstance(route_info.get('instance'), dict) else {}
    capability = normalize_capability(
        route_info.get('capability') or instance.get('capability') or base_payload.get('capability')
    )
    route_source = str(route_info.get('route_source') or '').strip().lower()
    route_runtime = (
        route_info.get('route_runtime')
        if isinstance(route_info.get('route_runtime'), dict)
        else {}
    )
    request_phase_graph = (
        route_runtime.get('request_phase_graph')
        if isinstance(route_runtime.get('request_phase_graph'), dict)
        else None
    )
    if not isinstance(request_phase_graph, dict) or not request_phase_graph:
        request_phase_graph = build_request_phase_graph(
            prompt,
            intent_prompt=str(base_payload.get('_current_turn_prompt') or prompt).strip() or None,
            request_payload=base_payload,
            route_payload=route_info,
        )
    graph_deferred_branches = downstream_phase_records(request_phase_graph)
    graph_deferred_capabilities = downstream_phase_capabilities(request_phase_graph)
    if capability == CAPABILITY_TEXT_TO_SPEECH:
        referenced_tts_prompt = None
        if _TTS_READ_REFERENCE_RE.search(prompt):
            referenced_tts_prompt = _extract_payload_content_payload(base_payload)
        if not referenced_tts_prompt:
            referenced_tts_prompt = _extract_tts_reference_prompt(prompt, context_messages=context_messages)
        if referenced_tts_prompt:
            updated_payload, applied_meta = _apply_local_tts_reference_result(
                base_payload,
                prompt=prompt,
                instance=instance,
                referenced_content=referenced_tts_prompt,
                context_messages=context_messages,
            )
            applied_meta.update(
                {
                    'capability': capability or None,
                    'planner_instance_id': None,
                    'planner_model': None,
                    'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
                    'phase_graph': request_phase_graph,
                }
            )
            return updated_payload, applied_meta
    trigger = _needs_compound_planning(
        prompt,
        capability=capability,
        payload=base_payload,
        instance=instance,
        context_messages=context_messages,
    )
    prompt_analysis = analyze_prompt_intent(prompt)
    if (
        capability == CAPABILITY_CHAT
        and graph_deferred_capabilities
        and request_phase_graph.get('current_phase_resolution') == 'graph_resolved'
    ):
        updated_payload, applied_meta = _apply_phase_graph_follow_up_result(
            base_payload,
            follow_up_branches=graph_deferred_branches,
            follow_up_capabilities=graph_deferred_capabilities,
            trigger='request_phase_graph_follow_up',
            prompt=prompt,
            context_messages=context_messages,
        )
        applied_meta.update(
            {
                'capability': capability or None,
                'planner_instance_id': None,
                'planner_model': None,
                'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
                'phase_graph': request_phase_graph,
            }
        )
        return updated_payload, applied_meta
    if not trigger and capability == CAPABILITY_CHAT and route_source == 'ghost_carried':
        deferred_capability = normalize_capability(
            prompt_analysis.get('text_first_follow_up_capability')
            or prompt_analysis.get('primary_capability')
        )
        if deferred_capability in {CAPABILITY_TEXT_TO_SPEECH, CAPABILITY_IMAGE_GENERATION}:
            updated_payload, applied_meta = _apply_ghost_carried_deferred_result(
                base_payload,
                prompt=prompt,
                follow_up_capability=deferred_capability,
                context_messages=context_messages,
            )
            applied_meta.update(
                {
                    'capability': capability or None,
                    'planner_instance_id': None,
                    'planner_model': None,
                    'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
                    'phase_graph': request_phase_graph,
                }
            )
            return updated_payload, applied_meta
    if not trigger:
        return base_payload, {
            'attempted': False,
            'applied': False,
            'capability': capability or None,
            'reason': 'not_a_compound_request',
            'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
            'phase_graph': request_phase_graph,
        }
    if trigger in {
        'text_first_tts_follow_up',
        'text_preparation_before_audio_output',
        'text_first_image_follow_up',
        'text_preparation_before_visual_output',
    }:
        updated_payload, applied_meta = _apply_phase_graph_follow_up_result(
            base_payload,
            follow_up_branches=graph_deferred_branches,
            follow_up_capabilities=graph_deferred_capabilities,
            follow_up_capability=prompt_analysis.get('text_first_follow_up_capability'),
            trigger=trigger,
            prompt=prompt,
            context_messages=context_messages,
        )
        applied_meta.update(
            {
                'capability': capability or None,
                'planner_instance_id': None,
                'planner_model': None,
                    'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
                'phase_graph': request_phase_graph,
            }
        )
        return updated_payload, applied_meta

    planner_instance = select_router_instance(instances)
    if not planner_instance:
        return base_payload, {
            'attempted': True,
            'applied': False,
            'capability': capability or None,
            'reason': 'no_local_planner_instance',
            'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
            'phase_graph': request_phase_graph,
        }

    planner_messages = [
        {
            'role': 'system',
            'content': _planner_instructions_with_semantic_roles(
                capability,
                instance,
                semantic_role_profile=semantic_role_profile,
            ),
        },
        {
            'role': 'user',
            'content': _planner_user_message(
                prompt,
                capability=capability,
                payload=base_payload,
                route_info=route_info,
                context_messages=context_messages,
                request_meta=request_meta,
                semantic_role_profile=semantic_role_profile,
            ),
        },
    ]
    effective_timeout_sec = max(1, int(planner_timeout_sec or 90))

    planner_raw = execute_chat_request(
        target_port=int(planner_instance['port']),
        model_name=str(planner_instance.get('model') or ''),
        backend=normalize_backend(planner_instance.get('backend')),
        capability='chat',
        messages=planner_messages,
        request_model_override=str(planner_instance.get('request_model') or '').strip() or None,
        temperature=0.15,
        max_tokens=32768,
        timeout_override_sec=effective_timeout_sec,
    )
    parsed = _extract_json_payload(planner_raw)
    if not isinstance(parsed, dict):
        return base_payload, {
            'attempted': True,
            'applied': False,
            'capability': capability or None,
            'planner_instance_id': str(planner_instance.get('instance_id') or '').strip() or None,
            'planner_model': str(planner_instance.get('model') or '').strip() or None,
            'planner_timeout_ms': effective_timeout_sec * 1000,
            'reason': 'planner_output_not_json',
            'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
            'phase_graph': request_phase_graph,
        }
    if not parsed.get('apply'):
        return base_payload, {
            'attempted': True,
            'applied': False,
            'capability': capability or None,
            'planner_instance_id': str(planner_instance.get('instance_id') or '').strip() or None,
            'planner_model': str(planner_instance.get('model') or '').strip() or None,
            'planner_timeout_ms': effective_timeout_sec * 1000,
            'reason': str(parsed.get('reason') or '').strip() or 'planner_declined',
            'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
            'phase_graph': request_phase_graph,
        }

    updated_payload, applied_meta = _apply_planner_result(
        base_payload,
        capability=capability,
        instance=instance,
        planned=parsed,
    )
    applied_meta.update(
        {
            'attempted': True,
            'capability': capability or None,
            'trigger': trigger,
            'planner_instance_id': str(planner_instance.get('instance_id') or '').strip() or None,
            'planner_model': str(planner_instance.get('model') or '').strip() or None,
            'planner_timeout_ms': effective_timeout_sec * 1000,
            'semantic_role_mode': str((semantic_role_profile or {}).get('mode') or '').strip() or None,
            'phase_graph': request_phase_graph,
        }
    )
    return updated_payload, applied_meta
