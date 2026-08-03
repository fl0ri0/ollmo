"""Canonical extraction of spoken data from direct and legacy TTS requests."""

from __future__ import annotations

import re
from typing import Any


_FENCED_TEXT_RE = re.compile(
    r'```(?P<lang>[A-Za-z0-9_+.-]*)(?:[^\n`]*)?\n'
    r'(?P<body>.*?)(?:\n```|```)',
    re.DOTALL,
)
_TEXT_FENCE_LANGUAGES = {'text', 'txt', 'markdown', 'md'}
_QUOTED_TEXT_RE = re.compile(
    r'"(?P<double>.+?)"|'
    r'“(?P<curly>.+?)”|'
    r'„(?P<german>.+?)“|'
    r'«(?P<guillemet>.+?)»',
    re.DOTALL,
)
_COLON_SUFFIX_RE = re.compile(r':\s*(?P<body>[^:\n][\s\S]*?)\s*$')
_DIRECT_SOURCE_CUE_RE = re.compile(
    r'(?:'
    r'\b(?:speak|read)\s+(?:only\s+)?(?:this|these)\s+aloud'
    r'(?:\s+in\s+sound\s+format)?\s*[.,:;-]*\s*$|'
    r'\b(?:speak|read|narrate)\b[^"“„«\n]{0,180}'
    r'\b(?:text|sentence|passage|quote|words?|line|phrase|content)\b'
    r'(?:\s+(?:exactly|verbatim|only|aloud))?\s*[.,:;-]*'
    r'(?:\s+do\s+not\s+create\s+or\s+plan\s+an\s+image[.!]?)?\s*$|'
    r'\b(?:say|sag)\b\s*(?:this|that|it|dies|das|es)?\s*[.,:;-]*\s*$|'
    r'\bthat\s+says?\s*[.,:;-]*\s*$|'
    r'\b(?:and\s+)?then\s+says?\s*[.,:;-]*\s*$|'
    r'\b(?:audio|speech|tts|recording|clip|artifact)\b[^"“„«\n]{0,180}'
    r'\b(?:says?|following(?:\s+\w+){0,3}\s+sentence|'
    r'(?:with|of|from)\s+(?:the\s+)?(?:following\s+)?'
    r'(?:text|sentence|passage|words?|phrase|content))\b'
    r'(?:\s+(?:exactly|verbatim|only|aloud))?\s*[.,:;-]*\s*$|'
    r'\b(?:sprich|lies|lese|narr(?:iere|ieren)|vorlesen)\b[^"“„«\n]{0,180}'
    r'\b(?:text|satz|passage|zitat|w(?:o|ö)rter|zeile|inhalt)\b'
    r'(?:\s+(?:genau|wortgetreu|vor))?\s*[.,:;-]*\s*$|'
    r'\b(?:audio|aufnahme|sprachfassung)\b[^"“„«\n]{0,180}'
    r'\b(?:mit|von|aus)\s+(?:dem|diesem|den|der|dieser)?\s*'
    r'(?:folgenden?\s+)?(?:text|satz|passage|zitat|w(?:o|ö)rtern?|inhalt)\b'
    r'(?:\s+(?:genau|wortgetreu))?\s*[.,:;-]*\s*$'
    r')',
    re.IGNORECASE,
)
_DIRECT_COLON_SOURCE_PREFIX_RE = re.compile(
    r'(?:'
    r'\b(?:say|speak|read|narrate)\s+(?:this|that|it)\s+'
    r'(?:aloud|out\s+loud)\b|'
    r'\b(?:speak|read|narrate|voice|sprich|lies|lese|vorlesen)\b|'
    r'\b(?:speak|read|narrate)\b[^:\n]{0,160}'
    r'\b(?:text|sentence|passage|quote|words?|line|phrase|content)\b|'
    r'\b(?:say|sag)\b\s*(?:this|that|it|dies|das|es)?|'
    r'\b(?:audio|speech|tts|recording|clip|artifact)\b[^:\n]{0,160}'
    r'\b(?:says?|(?:with|of|from)\s+(?:the\s+)?(?:following\s+)?'
    r'(?:text|sentence|passage|words?|phrase|content))\b|'
    r'\b(?:sprich|lies|lese|narr(?:iere|ieren)|vorlesen)\b[^:\n]{0,160}'
    r'\b(?:text|satz|passage|zitat|w(?:o|ö)rter|zeile|inhalt)\b|'
    r'\b(?:audio|aufnahme|sprachfassung)\b[^:\n]{0,160}'
    r'\b(?:mit|von|aus)\s+(?:dem|diesem|den|der|dieser)?\s*'
    r'(?:folgenden?\s+)?(?:text|satz|passage|zitat|w(?:o|ö)rtern?|inhalt)\b'
    r')\s*(?:exactly|verbatim|only|genau|wortgetreu)?\s*$',
    re.IGNORECASE,
)
_TRAILING_TTS_CONTROL_RE = re.compile(
    r'(?:\n+|;\s+|(?<=[.!?])\s+)(?='
    r'(?:use|set|choose)\b[^.\n]{0,100}\b(?:voice|speaker|language|format|style)\b|'
    r'(?:voice|speaker|language|format|style)\s*:)',
    re.IGNORECASE,
)


def _match_body_span(match: re.Match[str]) -> tuple[int, int, str]:
    for group_name in ('double', 'curly', 'german', 'guillemet'):
        if match.groupdict().get(group_name) is not None:
            start, end = match.span(group_name)
            return start, end, str(match.group(group_name) or '').strip()
    return -1, -1, ''


def resolve_explicit_tts_source(value: Any) -> dict[str, Any]:
    """Resolve one direct TTS data span with dependency-free provenance."""

    original = str(value or '')
    leading_offset = len(original) - len(original.lstrip())
    raw = original.strip()
    if not raw:
        return {}

    def record(text: str, start: int, end: int, source_kind: str) -> dict[str, Any]:
        return {
            'text': text,
            'start': leading_offset + start,
            'end': leading_offset + end,
            'source_kind': source_kind,
        }

    fenced_matches = [
        match
        for match in _FENCED_TEXT_RE.finditer(raw)
        if str(match.group('lang') or '').strip().lower() in _TEXT_FENCE_LANGUAGES
        and str(match.group('body') or '').strip()
    ]
    if len(fenced_matches) == 1:
        match = fenced_matches[0]
        prefix = raw[:match.start()].strip()
        suffix = raw[match.end():].strip()
        if (not prefix and not suffix) or _DIRECT_SOURCE_CUE_RE.search(prefix):
            start, end = match.span('body')
            return record(str(match.group('body') or '').strip(), start, end, 'text_fence')

    qualified_quotes: list[tuple[int, int, str]] = []
    previous_end = 0
    for match in _QUOTED_TEXT_RE.finditer(raw):
        local_prefix = raw[previous_end:match.start()].strip()
        previous_end = match.end()
        start, end, body = _match_body_span(match)
        if body and _DIRECT_SOURCE_CUE_RE.search(local_prefix):
            qualified_quotes.append((start, end, body))
    if len(qualified_quotes) == 1:
        start, end, body = qualified_quotes[0]
        return record(body, start, end, 'quoted_literal')
    if not qualified_quotes:
        sole_quote = _QUOTED_TEXT_RE.fullmatch(raw)
        if sole_quote:
            start, end, body = _match_body_span(sole_quote)
            if body:
                return record(body, start, end, 'quoted_literal')

    colon_match = _COLON_SUFFIX_RE.search(raw)
    if colon_match and _DIRECT_COLON_SOURCE_PREFIX_RE.search(
        raw[:colon_match.start()].strip()
    ):
        start, end = colon_match.span('body')
        body = str(colon_match.group('body') or '')
        trailing_control = _TRAILING_TTS_CONTROL_RE.search(body)
        if trailing_control:
            body = body[:trailing_control.start()].rstrip()
            end = start + len(body)
        if body.strip():
            return record(body.strip(), start, end, 'colon_suffix')
    return {}


def extract_explicit_tts_source_text(value: Any) -> str:
    """Return an unambiguous direct-request spoken payload, or an empty string.

    A literal is data only when its local prefix identifies it as the speech
    source. Titles, voice names, style fields, and transformation inputs are
    therefore not promoted. Multiple qualifying literals remain ambiguous and
    fail closed instead of silently choosing one.
    """

    return str(resolve_explicit_tts_source(value).get('text') or '')


def extract_tts_source_text(value: Any) -> str:
    """Return a direct spoken payload when proven, otherwise the stripped input."""

    raw = str(value or '').strip()
    return extract_explicit_tts_source_text(raw) or raw


def extract_legacy_tts_wrapper_text(value: Any) -> str:
    """Unwrap a legacy raw ``/infer`` TTS envelope.

    Structured ``content_payload`` data must never pass through this fallback.
    It exists only for old callers that still submit wrapper prose as ``prompt``.
    """

    raw = str(value or '').strip()
    if not raw:
        return ''
    fenced = [
        str(match.group('body') or '').strip()
        for match in _FENCED_TEXT_RE.finditer(raw)
        if str(match.group('lang') or '').strip().lower() in _TEXT_FENCE_LANGUAGES
        and str(match.group('body') or '').strip()
    ]
    if fenced:
        return max(fenced, key=len)
    direct = extract_explicit_tts_source_text(raw)
    if direct:
        return direct
    quoted = [
        str(
            match.group('double')
            or match.group('curly')
            or match.group('german')
            or match.group('guillemet')
            or ''
        ).strip()
        for match in _QUOTED_TEXT_RE.finditer(raw)
    ]
    quoted = [item for item in quoted if item]
    return max(quoted, key=len) if quoted else raw
