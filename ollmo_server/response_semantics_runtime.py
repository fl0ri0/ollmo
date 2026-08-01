"""Response-semantics owners for Ollmo."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    normalize_capability,
)
from ollmo_core.inference import text_artifact_request_is_ungrounded_reference
from ollmo_g.execution_planner import (
    plan_compound_execution,
    split_visible_image_payload,
    split_visible_tts_payload,
)
from ollmo_g.intent import analyze_prompt_intent
from ollmo_g.intent_obligations import (
    required_intent_obligations,
    summarize_required_intent_obligations,
)
from ollmo_g.router import _load_runtime_ghost_policy
from ollmo_g.request_phase_graph import (
    build_request_phase_graph,
    current_phase_capability,
    current_phase_is_graph_resolved,
    current_phase_reason,
    downstream_phase_capabilities,
    downstream_phase_records,
)
from ollmo_g.request_ir import output_obligations_from_graph
from ollmo_server.recovery_contract import (
    RECOVERY_ACTION_MANUAL_REVIEW,
    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
    RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
    RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
    RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
    RECOVERY_ACTION_RETRY_SAME_BRANCH,
    RECOVERY_ACTION_SEMANTIC_REVIEW,
    RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
    RECOVERY_SUGGESTED_ACTIONS,
    normalize_recovery_suggested_action,
)
from ollmo_server.repair_gate_runtime import (
    classify_repair_execution_policy,
    repair_item_has_concrete_input_evidence,
)
from ollmo_services.responses import (
    extract_canonical_predecessor_image_prompts,
    extract_responses_current_turn_prompt,
)
from ollmo_services.semantic_review_verdict import semantic_review_verdict_from_text


_TEXT_ONLY_IMAGE_COMPLETION_RE = re.compile(
    r'(?im)(?:^|\n)\s*(?:\*\*[^*\n]{0,120}\*\*\s*)?image generated\.\s*(?:$|\n)'
)
_TEXT_ONLY_AUDIO_COMPLETION_RE = re.compile(
    r'(?im)(?:^|\n)\s*(?:\*\*[^*\n]{0,120}\*\*\s*)?audio generated\.\s*(?:$|\n)'
)
_INTENT_SEMANTIC_FIT_RE = re.compile(
    r'\b(?:'
    r'exactly\s+match(?:es|ed|ing)?|match(?:es|ed|ing)?\s+(?:its|their|the)\s+'
    r'(?:image|images|section|sections|artifact|artifacts)|'
    r'fit(?:s|ting)?\s+together|coherent\s+(?:local\s+)?artifact\s+set|'
    r'consistent\s+(?:artifact|visual|content|section)|'
    r'aligned\s+(?:text|image|visual|artifact|section)|'
    r'whole[-\s]?turn\s+fit|'
    r'inhaltlich\s+exakt|exakt\s+(?:zum|zur|zu|mit)|'
    r'passt\s+(?:exakt|genau|inhaltlich)|passen\s+(?:exakt|genau|inhaltlich)|'
    r'zusammenpassen|sauber\s+zusammenpassen|stimmig\s+zusammen|'
    r'texte?\s+.*(?:bild|bilder).*(?:passt|passen|exakt|stimmig)|'
    r'bilder?\s+.*(?:text|texte|abschnitt|abschnitte).*(?:passt|passen|exakt|stimmig)'
    r')\b',
    re.IGNORECASE,
)
_INTENT_SEMANTIC_QUALITY_RE = re.compile(
    r'\b(?:'
    r'(?:page|site|website|design|world|copy|text|image|images|section|sections)\s+'
    r'(?:must|should|needs?\s+to)\s+(?:feel|look|be|match|fit|refer|correspond)|'
    r'each\s+[^.;!?\n]{0,100}(?:image|section)\s+[^.;!?\n]{0,140}'
    r'(?:match|fit|refer|correspond|belong)|'
    r'(?:not|no)\s+[^.;!?\n]{0,60}'
    r'(?:dystopian|generic|luxury|wellness|saas|stock|childish|futuristic)|'
    r'(?:seite|gestaltung|welt|design|text|bilder?|tierbild|section|sections|abschnitt|abschnitte)\s+'
    r'(?:soll|sollen|muss|müssen|darf|dürfen)\s+[^.;!?\n]{0,180}'
    r'(?:wirken|sein|passen|passend|eingehen|verwendet|wiederverwendet|vertauscht)|'
    r'jed(?:e|es|er|en)\s+[^.;!?\n]{0,100}(?:bild|tierbild|section|abschnitt)\s+'
    r'[^.;!?\n]{0,160}(?:passen|passend|eingehen|verwendet|wiederverwendet|vertauscht)|'
    r'(?:nicht|kein|keine)\s+[^.;!?\n]{0,60}'
    r'(?:dystopisch|generisch|luxus|wellness|saas|stockfoto|kindisch|futuristisch)'
    r')\b',
    re.IGNORECASE,
)
_LINKED_IMAGE_INTENT_RE = re.compile(
    r'\b(?:use|insert|include|embed|link|reference|wire|place|background|hero|'
    r'nutze|verwende|fuege|füge|verlinke|referenziere|hintergrund)\b'
    r'[\s\S]{0,180}\b(?:generated|saved|local|actual|erzeugte(?:n|s|m|r)?|generierte(?:n|s|m|r)?|gespeicherte(?:n|s|m|r)?)?\s*'
    r'(?:image|images|picture|pictures|png|jpg|jpeg|webp|bild|bilder|bildartefakt)\b|'
    r'\b(?:generated|saved|local|actual|erzeugte(?:n|s|m|r)?|generierte(?:n|s|m|r)?|gespeicherte(?:n|s|m|r)?)?\s*'
    r'(?:image|images|picture|pictures|png|jpg|jpeg|webp|bild|bilder|bildartefakt)\b'
    r'[\s\S]{0,180}\b(?:hero|background|hintergrund|use|insert|include|embed|link|reference|place|nutze|verwende)\b',
    re.IGNORECASE,
)
_LINK_PLACEHOLDER_RE = re.compile(
    r'\b(?:placeholder|replace[_\-\s]?me|todo|tbd|dummy|'
    r'asset[_\-\s]?placeholder|image[_\-\s]?placeholder|generated[_\-\s]?image|'
    r'hero[_\-\s]?image|your[_\-\s]?(?:image|asset)|'
    r'platzhalter|zu[_\-\s]?ersetzen|ersetzen[_\-\s]?(?:bild|asset))\b',
    re.IGNORECASE,
)
_LINK_ATTRIBUTE_PLACEHOLDER_RE = re.compile(
    r'<(?P<tag_name>[a-z][a-z0-9:-]*)\b[^>]{0,800}?\b(?P<attr>src|href)\s*=\s*'
    r'(?P<quote>["\'])(?P<value>#|about:blank|placeholder:[^"\']*|)(?P=quote)[^>]*>',
    re.IGNORECASE,
)
_LINK_ATTRIBUTE_VALUE_RE = re.compile(
    r'<(?P<tag_name>[a-z][a-z0-9:-]*)\b[^>]{0,800}?\b(?P<attr>src|href)\s*=\s*'
    r'(?P<quote>["\'])(?P<value>[^"\']*)(?P=quote)[^>]*>',
    re.IGNORECASE,
)
_LINK_CSS_URL_VALUE_RE = re.compile(
    r'\burl\(\s*[\'"]?(?P<value>[^\'")]+)[\'"]?\s*\)',
    re.IGNORECASE,
)
_CSS_BACKGROUND_DECLARATION_RE = re.compile(
    r'(?P<property>\bbackground(?:-image)?\b)\s*:\s*(?P<value>[^;{}]+)',
    re.IGNORECASE,
)
_CSS_DECLARATION_VALUE_RE = re.compile(
    r'(?m)(?P<prefix>[{;]\s*)(?P<property>-{0,2}[a-z_][a-z0-9_-]*)\s*:\s*(?P<value>[^;{}]+)',
    re.IGNORECASE,
)
_CSS_HTML_ENTITY_FRAGMENT_IN_FUNCTION_RE = re.compile(
    r'\b(?:rgb|rgba|hsl|hsla|linear-gradient|radial-gradient|color-mix)\s*\([^;{}]*&#',
    re.IGNORECASE,
)
_CSS_INVALID_BACKGROUND_SHORTHAND_TOKEN_RE = re.compile(
    r'(?<![A-Za-z0-9_-])(?P<token>no-(?:scale|format|array))(?![A-Za-z0-9_-])',
    re.IGNORECASE,
)
_CSS_INVALID_BACKGROUND_SHORTHAND_TOKENS = {
    'no-scale': 'no-repeat',
    'no-format': 'no-repeat',
    'no-array': 'no-repeat',
}
_CSS_KNOWN_INVALID_PROPERTY_VALUES = {
    ('box-sizing', 'border-width'): 'border-box',
}
_HERO_LOCAL_IMAGE_SIGNAL_RE = re.compile(
    r'\b(?:hero|hero[-_\s]?image|background|hintergrund|titelbild|title\s+image)\b',
    re.IGNORECASE,
)
_HTML_HERO_BLOCK_RE = re.compile(
    r'<(?P<tag>header|section|main|div)\b(?P<attrs>[^>]*(?:\b(?:class|id)\s*=\s*["\'][^"\']*hero[^"\']*["\'][^>]*)?)>'
    r'(?P<body>[\s\S]{0,12000}?)</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_CSS_HERO_BLOCK_RE = re.compile(
    r'(?P<selectors>[^{}]*(?:[.#][A-Za-z0-9_-]*hero[A-Za-z0-9_-]*|\bheader\b)[^{}]*)\{(?P<body>[^{}]*)\}',
    re.IGNORECASE,
)
_HTML_CLASS_ATTR_RE = re.compile(
    r'\bclass\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
    re.IGNORECASE | re.DOTALL,
)
_HTML_ATTRIBUTE_VALUE_RE = re.compile(
    r'\b(?P<name>[a-z_:][a-z0-9_:.:-]*)\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
    re.IGNORECASE | re.DOTALL,
)
_HTML_BARE_DUPLICATE_ATTRIBUTE_RE = re.compile(
    r'(?<![A-Za-z0-9_:-])(?P<name>class|id|href|src|rel|alt|title|type|name|role|aria-[a-z0-9_-]+|data-[a-z0-9_-]+)\b'
    r'\s+(?P=name)\s*=',
    re.IGNORECASE,
)
_HTML_MALFORMED_STYLESHEET_LINK_ATTR_RE = re.compile(
    r'<link\b(?P<prefix>[^>]*?)\brel\s*=\s*(?P<quote>["\'])'
    r'stylesheet\s*=\s*(?P=quote)\s*(?P<href>href\s*=\s*(?P<hquote>["\'])(?P<value>[^"\']*)(?P=hquote))'
    r'(?P<suffix>[^>]*)>',
    re.IGNORECASE | re.DOTALL,
)
_CSS_BLOCK_SELECTOR_RE = re.compile(r'(?P<selectors>[^{}]+)\{', re.DOTALL)
_CSS_CLASS_SELECTOR_RE = re.compile(r'(?<![A-Za-z0-9_-])\.(?P<class>-?[_A-Za-z][A-Za-z0-9_-]*)')
_HTML_CSS_SELECTOR_BINDING_CONTENT_LIMIT = 60_000
_HTML_CSS_SELECTOR_BINDING_MIN_TOKENS = 6
_HTML_CSS_SELECTOR_BINDING_MIN_MISSING = 4
_HTML_CSS_SELECTOR_BINDING_MIN_RATIO = 0.35
_HTML_CSS_IGNORABLE_CLASS_TOKENS = {
    'active',
    'container',
    'hidden',
    'is-active',
    'is-hidden',
    'is-open',
    'open',
    'primary',
    'reverse',
    'secondary',
    'selected',
    'show',
    'visible',
}
_ASSET_LINK_PLACEHOLDER_TAGS = {
    'audio',
    'embed',
    'iframe',
    'image',
    'img',
    'link',
    'object',
    'script',
    'source',
    'track',
    'video',
}


def _link_placeholder_attribute_targets_asset(match: re.Match[str]) -> bool:
    tag_name = str(match.group('tag_name') or '').strip().lower()
    attr = str(match.group('attr') or '').strip().lower()
    if attr == 'src':
        return True
    return tag_name in _ASSET_LINK_PLACEHOLDER_TAGS


def _content_has_unresolved_link_placeholder(content: str) -> bool:
    if not content:
        return False
    text = str(content or '')
    for match in _LINK_ATTRIBUTE_PLACEHOLDER_RE.finditer(text):
        if _link_placeholder_attribute_targets_asset(match):
            return True
    for match in _LINK_ATTRIBUTE_VALUE_RE.finditer(text):
        if not _link_placeholder_attribute_targets_asset(match):
            continue
        value = str(match.group('value') or '').strip()
        if value and _LINK_PLACEHOLDER_RE.search(value):
            return True
    return any(
        _LINK_PLACEHOLDER_RE.search(str(match.group('value') or '').strip())
        for match in _LINK_CSS_URL_VALUE_RE.finditer(text)
    )
_TEXT_LINK_ARTIFACT_EXTENSIONS = {'html', 'htm', 'css', 'js', 'mjs', 'cjs'}
_HTML_LINK_TARGET_EXTENSIONS = {'css', 'js', 'mjs', 'cjs'}
_HTML_SYNTAX_VOID_TAGS = {
    'area',
    'base',
    'br',
    'col',
    'embed',
    'hr',
    'img',
    'input',
    'link',
    'meta',
    'param',
    'source',
    'track',
    'wbr',
}
_HTML_SYNTAX_REPAIR_CLOSING_TAGS = _HTML_SYNTAX_VOID_TAGS | {
    'a',
    'article',
    'aside',
    'body',
    'button',
    'canvas',
    'code',
    'dd',
    'details',
    'dialog',
    'div',
    'dl',
    'dt',
    'figcaption',
    'figure',
    'footer',
    'form',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'head',
    'header',
    'html',
    'label',
    'li',
    'main',
    'menu',
    'nav',
    'ol',
    'option',
    'p',
    'picture',
    'pre',
    'script',
    'section',
    'select',
    'small',
    'span',
    'strong',
    'style',
    'summary',
    'table',
    'tbody',
    'td',
    'template',
    'textarea',
    'tfoot',
    'th',
    'thead',
    'tr',
    'ul',
    'video',
}
_HTML_STRAY_KNOWN_CLOSING_TAG_REMOVABLE_TAGS = {
    'a',
    'article',
    'aside',
    'button',
    'code',
    'dd',
    'details',
    'dialog',
    'div',
    'dl',
    'dt',
    'figcaption',
    'figure',
    'footer',
    'form',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'header',
    'label',
    'li',
    'main',
    'menu',
    'nav',
    'ol',
    'p',
    'pre',
    'section',
    'small',
    'span',
    'strong',
    'summary',
    'ul',
}
_HTML_SYNTAX_REPAIR_TAG_ALT = '|'.join(
    re.escape(tag)
    for tag in sorted(_HTML_SYNTAX_REPAIR_CLOSING_TAGS, key=len, reverse=True)
)
_HTML_MALFORMED_CLOSE_EMBEDDED_TAG_RE = re.compile(
    rf'</<[^>\n]{{1,80}}>(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})\s*>',
    re.IGNORECASE,
)
_HTML_MALFORMED_CLOSE_PREFIX_RE = re.compile(
    rf'</(?P<prefix>[^a-z0-9\s<>/:-]+)(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})\s*>',
    re.IGNORECASE,
)
_HTML_MALFORMED_CLOSE_KNOWN_SUFFIX_RE = re.compile(
    rf'</\s*(?!(?:{_HTML_SYNTAX_REPAIR_TAG_ALT})\s*>)'
    rf'(?P<prefix>[a-z0-9]{{1,12}})(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})\s*>',
    re.IGNORECASE,
)
_HTML_STRAY_NESTED_CLOSE_RE = re.compile(
    rf'</\s*</\s*(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})\s*>',
    re.IGNORECASE,
)
_HTML_STRAY_FORMATTING_CLOSE_TAG_ALT = r'strong|em|b|i'
_HTML_STRAY_FORMATTING_CLOSE_IN_OPEN_TAG_RE = re.compile(
    rf'<\s*</\s*(?P<format_tag>{_HTML_STRAY_FORMATTING_CLOSE_TAG_ALT})\s*>\s*'
    rf'(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})(?P<boundary>[\s>/])',
    re.IGNORECASE,
)
_HTML_PARTIAL_OPEN_TAG_WITH_STRAY_FORMATTING_CLOSE_RE = re.compile(
    rf'<(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})(?P<attrs>[^<>\n]{{0,80}}?)'
    rf'</\s*(?P<format_tag>{_HTML_STRAY_FORMATTING_CLOSE_TAG_ALT})\s*>\s*'
    rf'(?=<(?P=tag)(?:[\s>/]))',
    re.IGNORECASE,
)
_HTML_MALFORMED_OPEN_PREFIX_KNOWN_TAG_RE = re.compile(
    rf'<(?P<prefix>[^a-z0-9\s<>/:-]{{1,12}})(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})(?P<boundary>[\s>/])',
    re.IGNORECASE,
)
_HTML_DUPLICATE_OPEN_ANGLE_KNOWN_TAG_RE = re.compile(
    rf'<<(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})(?P<boundary>[\s>/])',
    re.IGNORECASE,
)
_HTML_STRAY_PUNCTUATION_PSEUDO_TAG_BEFORE_KNOWN_TAG_RE = re.compile(
    rf'(?P<prefix>^|[\s>])(?P<fragment><[)\]}}]+)\s*'
    rf'(?=<\s*(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})(?P<boundary>[\s>/]))',
    re.IGNORECASE | re.MULTILINE,
)
_HTML_QUOTED_KNOWN_OPEN_TAG_RE = re.compile(
    rf'(?P<prefix>^|[\s>])(?P<quote>["\'])(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})(?P=quote)'
    r'(?P<attrs>\s+[^<>\n]*?)>',
    re.IGNORECASE | re.MULTILINE,
)
_HTML_REPEATED_SIBLING_CLASS_HINT_RE = re.compile(
    r'(?:^|[-_])(card|item|panel|tile|feature|story|gallery|slide|entry)(?:$|[-_])',
    re.IGNORECASE,
)
_HTML_SYNTAX_TAG_RE = re.compile(
    r'<!--[\s\S]*?-->|<![a-z][^>]*>|<(?P<close>/)?(?P<tag>[a-z][a-z0-9:-]*)(?P<attrs>[^<>]*?)(?P<self>/)?>',
    re.IGNORECASE,
)
_HTML_UNTERMINATED_KNOWN_OPEN_TAG_RE = re.compile(
    rf'<(?P<tag>{_HTML_SYNTAX_REPAIR_TAG_ALT})\b'
    r'(?P<attrs>[^<>\r\n]*[^\s<>\r\n][^<>\r\n]*)'
    r'(?=\r?\n[ \t]*<)',
    re.IGNORECASE,
)
_HTML_RAW_TEXT_BLOCK_RE = re.compile(
    r'<(?P<tag>script|style)\b[^>]*>[\s\S]*?</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_HTML_HERO_ATTR_VALUE_RE = re.compile(r'(^|[\s_-])hero($|[\s_-])', re.IGNORECASE)
_HTML_HERO_LANDMARK_TAGS = {'footer', 'main'}
_HTML_MARKUP_IN_ATTRIBUTE_RE = re.compile(
    r'=\s*"[^"]*</[a-z][^"]*"|=\s*\'[^\']*</[a-z][^\']*\'',
    re.IGNORECASE,
)
_HTML_MALFORMED_CLASS_ATTR_COLON_RE = re.compile(
    r'(?<!["\'=A-Za-z0-9_-])\bclass\s*:\s*(?P<value>[^\s"\'<>/][^"\'<>]*?)(?P<quote>["\'])',
    re.IGNORECASE,
)
_HTML_MALFORMED_ANCHOR_OPEN_TAG_RE = re.compile(
    rf'<(?P<tag>(?!(?:{_HTML_SYNTAX_REPAIR_TAG_ALT})\b)a[a-z0-9]{{1,15}})'
    r'(?P<attrs>[^<>\n]*\bhref\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s<>]+)[^<>\n]*)>'
    r'(?P<body>[\s\S]{0,2000}?)</a>'
    r'(?P<obsolete_close>\s*</\s*(?P=tag)\s*>)?',
    re.IGNORECASE,
)
_HTML_MALFORMED_FORMATTED_ANCHOR_FRAGMENT_RE = re.compile(
    r'<<(?P<wrapper>em|strong|b|i)>\s*a\s+'
    r'(?P<attrs>[^<>\n]{0,400}\bhref\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s<>]+)[^<>\n]*)>'
    r'(?P<body>[\s\S]{0,1200}?)</\s*(?P=wrapper)\s*>',
    re.IGNORECASE,
)
_HTML_HREF_ALLOWED_TAGS = {'a', 'area', 'base', 'link'}
_HTML_HREF_ALLOWED_TAG_ALT = '|'.join(
    re.escape(tag)
    for tag in sorted(_HTML_HREF_ALLOWED_TAGS, key=len, reverse=True)
)
_HTML_HREF_ATTR_RE = re.compile(r'\bhref\s*=', re.IGNORECASE)
_HTML_STYLESHEET_REL_TOKENS = {'stylesheet', 'preload'}
_HTML_UNSUPPORTED_HREF_ELEMENT_RE = re.compile(
    rf'<(?P<tag>(?!(?:{_HTML_HREF_ALLOWED_TAG_ALT})\b)[a-z][a-z0-9:-]{{0,30}})'
    r'(?P<attrs>[^<>\n]*\bhref\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s<>]+)[^<>\n]*)>'
    r'(?P<body>[\s\S]{0,2000}?)</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_HTML_UNSUPPORTED_NAV_ANCHOR_WRAPPER_RE = re.compile(
    rf'<(?P<tag>(?!(?:{_HTML_SYNTAX_REPAIR_TAG_ALT})\b)[a-z][a-z0-9:-]{{0,30}})'
    r'(?P<attrs>[^<>\n]*)>\s*'
    r'(?P<anchor><a\b[^>]*>[\s\S]{0,2000}?</a>)\s*'
    r'</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_HTML_LITERAL_UNSUPPORTED_TAG_RE = re.compile(
    r'<(?P<tag>unsupported)(?P<attrs>[^<>\n]*)>'
    r'(?P<body>[\s\S]{0,2000}?)</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_HTML_HREF_REPAIR_EXCLUDED_TAGS = _HTML_SYNTAX_REPAIR_CLOSING_TAGS | {
    'animate',
    'animatemotion',
    'animatetransform',
    'feimage',
    'image',
    'mpath',
    'textpath',
    'use',
}
_HTML_MALFORMED_CLASS_ONLY_OPEN_TAG_RE = re.compile(
    rf'<\s+(?P<class>[_A-Za-z][A-Za-z0-9_-]{{1,80}})(?P<quote>["\'])\s*>'
    rf'(?P<body>[\s\S]{{0,1200}}?)</\s*(?P<close>{_HTML_SYNTAX_REPAIR_TAG_ALT})\s*>',
    re.IGNORECASE,
)
_HTML_UNKNOWN_CLASS_WRAPPER_TAG_RE = re.compile(
    rf'<(?P<tag>(?!(?:{_HTML_SYNTAX_REPAIR_TAG_ALT})\b)[a-z][a-z0-9]{{1,30}})'
    r'(?P<attrs>[^<>\n]*\b(?:class|id)\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s<>]+)[^<>\n]*)>'
    r'(?P<body>[\s\S]{0,4000}?)</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_CSS_KNOWN_PROPERTY_TYPOS = {
    'align-s_items': 'align-items',
    'background-transparent': 'background',
    'font-mask': 'font-family',
    'font-scale': 'font-size',
    'font-width': 'font-weight',
    'letter-lag': 'letter-spacing',
    'letter-length': 'letter-spacing',
    'margin-blop': 'margin-bottom',
    'scroll-template': 'scroll-behavior',
    'text-allign': 'text-align',
}
_CSS_AMBIGUOUS_PROPERTY_TYPOS = {
    'margin-length': 'margin, margin-left, or another margin-side property',
}
_CSS_DECLARATION_NAME_RE = re.compile(
    r'(?m)(?P<prefix>[{;]\s*)(?P<property>[^{};:\n][^{};:]*?)\s*:',
)
_CSS_DECLARATION_NAME_FORMATTING_TAG_RE = re.compile(
    r'</?\s*(?:strong|b|em|i)\s*/?>',
    re.IGNORECASE,
)
_CSS_PROPERTY_IDENTIFIER_RE = re.compile(
    r'^(?:--[A-Za-z0-9_-]+|-?[A-Za-z_][A-Za-z0-9_-]*)$',
)
_CSS_CUSTOM_PROPERTY_DEFINITION_RE = re.compile(
    r'(?m)(?<![A-Za-z0-9_-])(?P<name>--[A-Za-z0-9_-]+)\s*:',
)
_CSS_VAR_REFERENCE_RE = re.compile(
    r'\bvar\(\s*(?P<name>---[A-Za-z0-9_-]+)(?P<suffix>\s*(?:,[^)]*)?\))',
    re.IGNORECASE,
)
_CSS_ANY_VAR_REFERENCE_RE = re.compile(
    r'\bvar\(\s*(?P<name>--[A-Za-z0-9_-]+)(?P<suffix>\s*(?:,[^)]*)?\))',
    re.IGNORECASE,
)
_CSS_INVALID_FUNCTION_TOKENS = (
    (
        re.compile(r'\bvar\s+var\s*\(', re.IGNORECASE),
        'var(',
        'css_invalid_duplicated_var_function_token',
    ),
    (
        re.compile(r'\bvarint\s*\(', re.IGNORECASE),
        'var(',
        'css_invalid_varint_function_token',
    ),
    (
        re.compile(r'\bvar\s*/\s*\(', re.IGNORECASE),
        'var(',
        'css_invalid_var_function_token',
    ),
)
_CSS_INVALID_VAR_SLASH_REFERENCE_RE = re.compile(
    r'\bvar\s*/\s*(?P<name>-{0,2}[A-Za-z0-9_-]+)\b',
    re.IGNORECASE,
)
_GHOST_RUNTIME_POLICY_SYSTEM_MARKER = 'Ollmo Ghost runtime policy: attached.'
_PREPARE_PHASE_SYSTEM_MARKER = 'Ollmo phase contract: prepare-only.'
_SEMANTIC_PHASE_PAYLOAD_KEYS = (
    'content_payload',
    'stage_direction',
    'phase_summary',
    'content_payload_source',
    'artifact_prompt',
    'artifact_prompt_source',
    'batch_prompts',
    'batch_prompts_source',
    'batch_prompt_source_phase_id',
    'batch_prompt_expected_count',
    'requires_artifact',
    'text_artifact_extension',
    'text_artifact_source_name',
    'text_artifact_source',
    'text_artifact_target_path',
    'artifact_request',
    'repair_scope',
    'resource_class',
    'dependency_policy',
    'runtime_scheduling_context',
)
_CLOSURE_REPAIR_PAYLOAD_KEYS = (
    'execution_contract',
    'workload_task_ref',
    'output_obligation_ref',
    'output_contract',
    'input_refs',
    'review_criteria',
    'artifact_prompt',
    'artifact_prompt_source',
    'batch_prompts',
    'batch_prompts_source',
    'batch_prompt_source_phase_id',
    'batch_prompt_expected_count',
    'artifact_request',
    'accepted_proposals',
    'requires_artifact',
    'text_artifact_extension',
    'text_artifact_source_name',
    'text_artifact_source',
    'text_artifact_target_path',
    'repair_scope',
    'resource_class',
    'dependency_policy',
    'runtime_scheduling_context',
    'allow_gpu_heavy_concurrency',
    'semantic_intent',
    'objective',
    'deliverable',
    'rationale',
    'advisory_role',
    'decision_notes',
    'evidence_requirements',
    'reconsideration_triggers',
    'semantic_review_criteria',
    'semantic_review_lens',
    'success_definition',
    'failure_modes',
    'semantic_lens_evidence_requirements',
    'semantic_review_lens_contract',
    'semantic_review_lens_review',
    'decision_contract_semantic_review_lenses',
    'promotion_suggestions',
    'waiver_candidates',
    'repair_candidates',
    'supersession_candidates',
    'learning_hint_refs',
    'decision_contract_repair_candidates',
    'decision_contract_semantic_review_candidates',
    'decision_contract_supersession_candidates',
    'decision_contract_block_resolution_signals',
    'decision_contract_active_reconsideration_decisions',
    'decision_contract_semantic_quality_contracts',
    'block_resolution_reflex',
    'block_resolution_signal',
    'block_resolution_action',
    'block_resolution_policy',
    'reconsideration_reflex',
    'active_reconsideration_review',
    'active_reconsideration_decision',
    'active_reconsideration_action',
    'active_reconsideration_review_type',
    'semantic_quality_review',
    'semantic_quality_contract',
    'semantic_quality_status',
    'semantic_quality_review_id',
    'recursive_cycle_review',
    'recursive_cycle_state',
    'aspiration_review',
    'aspiration_frame',
    'aspiration_frame_id',
    'aspiration_action',
    'aspiration_reason',
    'aspiration_allowed_actions',
    'decision_contract_aspiration_frames',
    'commitment_review',
    'commitment_frame',
    'commitment_frame_id',
    'commitment_action',
    'commitment_recommended_transition',
    'commitment_reason',
    'commitment_allowed_transitions',
    'decision_contract_commitment_frames',
    'semantic_decision_review',
    'semantic_decision_proposal',
    'semantic_decision_action',
    'semantic_decision_confidence',
    'semantic_decision_reason',
    'semantic_review_verdict',
    'semantic_review_verdict_status',
    'semantic_review_recommended_transition',
    'branch_semantic_review',
    'branch_semantic_review_branch_id',
    'branch_semantic_review_phase_id',
    'branch_semantic_review_status',
    'branch_semantic_review_reason',
    'branch_semantic_review_source_branch_id',
    'branch_semantic_review_source_phase_id',
    'decision_contract_semantic_decision_proposals',
    'controlled_attention_review',
    'controlled_attention_frame',
    'controlled_attention_frame_id',
    'controlled_attention_scope',
    'controlled_attention_priority',
    'controlled_attention_question',
    'controlled_attention_allowed_transitions',
    'decision_contract_controlled_attention_frames',
    'global_semantic_closure_review',
    'global_semantic_closure_proposal',
    'global_semantic_closure_status',
    'global_semantic_closure_reason',
    'global_semantic_closure_confidence',
    'content_payload',
    'content_payload_source',
    'stage_direction',
    'phase_summary',
    'surface_state',
    'repair_contract',
    'repair_contract_id',
    'repair_contract_status',
    'repair_execution_policy',
    'repair_promotion_source',
    'contract_state',
    'promotion_source',
    'supersession_review_required',
    'supersession_review_authority',
    'superseded_by',
    'superseded_by_candidate_id',
    'superseded_by_obligation_id',
    'supersession_reason',
)
_MULTI_IMAGE_PROMPT_SECTION_RE = re.compile(
    r'(?ims)^\s*(?:[-*#>]+\s*)?(?:[^\n:]{0,60}\b(?:image|bild|prompt|scene|variation|variante|version)\b[^\n:]{0,30})\s*:\s*(?P<body>.*?)(?=^\s*(?:[-*#>]+\s*)?(?:[^\n:]{0,60}\b(?:image|bild|prompt|scene|variation|variante|version)\b[^\n:]{0,30})\s*:|\Z)'
)
_PARAGRAPH_BREAK_RE = re.compile(r'(?:\r?\n){2,}')
_MULTI_IMAGE_HEADING_LINE_RE = re.compile(
    r'(?i)^(?:(?:image|bild|scene|variation|variante|variant|version|prompt)\s*(?:\d+|[ivx]+)?|(?:image|bild)\s+prompt\s*(?:\d+|[ivx]+)?)(?:\s*(?:[-–—:]|\().*)?$'
)
_INLINE_LABELED_IMAGE_PROMPT_LINE_RE = re.compile(
    r'(?i)^\s*(?:[-*#>\u2022]+\s*)?(?:\*\*+|__+|\*)?\s*'
    r'(?:'
    r'(?:image|bild|visual|scene)\s*(?:prompt\s*)?(?:\d+|[ivx]+)?(?:\s*\([^)]+\))?'
    r'|prompt\s*(?:\d+|[ivx]+)?(?:\s*\([^)]+\))?'
    r'|[a-z0-9][\w ._-]{0,80}\.(?:png|jpe?g|webp|gif|avif)'
    r')'
    r'\s*(?:\*\*+|__+|\*)?\s*:\s*(?P<body>.*)$'
)
_SEQUENTIAL_BOLD_ALPHA_IMAGE_PROMPT_LINE_RE = re.compile(
    r'^\s*(?:[-*#>\u2022]+\s+)?'
    r'(?P<wrapper>\*\*|__)'
    r'(?P<label>[A-Z])\s*:\s*'
    r'(?P=wrapper)\s*'
    r'(?P<body>.+?)\s*$'
)
_PLAIN_ALPHA_IMAGE_PROMPT_LINE_RE = re.compile(
    r'^\s*(?P<label>[A-Z])\s*:\s*(?P<body>.+?)\s*$'
)
_ARTIFACT_LABELED_IMAGE_PROMPT_HEADING_RE = re.compile(
    r'(?i)^\s*(?:[-*#>\u2022]+\s*)?(?:\*\*+|__+|\*)?\s*'
    r'(?:artifact|artefakt)\s*(?:\d+|[ivx]+)?\s*:\s*'
    r'(?:'
    r'(?:image|visual)\s*(?:generation\s*)?prompt'
    r'|bild(?:generierungs?|\s+generation)?[-\s]*prompt'
    r')'
    r'(?:\s*(?:\([^)]+\)|\[[^\]]+\]))?'
    r'\s*(?:\*\*+|__+|\*)?\s*:?\s*$'
)
_ARTIFACT_SECTION_HEADING_RE = re.compile(
    r'(?i)^\s*(?:[-*#>\u2022]+\s*)?(?:\*\*+|__+|\*)?\s*'
    r'(?:artifact|artefakt)\s*(?:\d+|[ivx]+)?\s*:\s*\S.*$'
)
_EMBEDDED_LABELED_IMAGE_PROMPT_RE = re.compile(
    r'(?ims)(?:^|[\r\n])\s*(?:[-*#>\u2022]+\s*)?(?:\*\*+|__+|\*)?\s*'
    r'(?:visual\s+prompt|image\s+prompt|bild[-\s]?prompt|prompt)'
    r'\s*(?:\*\*+|__+|\*)?\s*:\s*(?P<body>.*?)(?='
    r'(?:[\r\n]\s*(?:[-*#>\u2022]+\s*)?(?:\*\*+|__+|\*)?\s*'
    r'(?:asset\s+id|@?username|user\s*name|caption|influencer|'
    r'visual\s+prompt|image\s+prompt|bild[-\s]?prompt|prompt)\b'
    r'\s*(?:\*\*+|__+|\*)?\s*:)|\Z)'
)
_IMAGE_ASSET_LABEL_PREFIX_RE = re.compile(
    r'(?is)^(?:\*\*+|__+|`+|\*+)?\s*'
    r'(?:(?:<b>|<strong>)\s*)?'
    r'[a-z0-9][\w ._-]{0,100}\.(?:png|jpe?g|webp|gif|avif)'
    r'(?:\s*</(?:b|strong)>)?'
    r'\s*(?:\*\*+|__+|`+|\*+)?\s*:\s*'
)
_FILENAME_SOCIAL_ASSET_ROW_RE = re.compile(
    r'(?im)^\s*(?:[-*#>\u2022]+\s*)?(?:\d{1,3}|[ivx]+)[.)]\s*'
    r'(?:\*\*+|__+|`+|\*+)?\s*'
    r'(?:(?:<b>|<strong>)\s*)?'
    r'(?P<filename>[a-z0-9][\w ._-]{0,140}\.(?:png|jpe?g|webp|gif|avif))'
    r'(?:\s*</(?:b|strong)>)?'
    r'\s*(?:\*\*+|__+|`+|\*+)?\s*:\s*(?P<body>[^\n]+)$'
)
_STRUCTURAL_PAGE_IMAGE_ASSET_LABEL_RE = re.compile(
    r'(?i)\b(?:hero(?:[-_\s]?(?:bg|background))?|background|banner|cover|masthead)'
    r'[\w ._-]{0,40}\.(?:png|jpe?g|webp|gif|avif)\b'
)
_NUMBERED_PREPARED_IMAGE_PROMPT_RE = re.compile(
    r'(?ms)(?:^|\n)\s*(?:\d{1,2}|[ivx]+)[.)]\s+'
    r'(?P<body>.*?)(?=(?:\n\s*(?:\d{1,2}|[ivx]+)[.)]\s+)|\Z)'
)
_NUMBERED_IMAGE_PROMPT_SEGMENT_PATTERNS = (
    r'(?ms)(?:^|\n)\s*\d+[.)]\s*(?P<body>.*?)(?=(?:\n\s*\d+[.)]\s*)|\Z)',
    r'(?ms)(?:^|\s)\d+[.)]\s*(?P<body>.*?)(?=(?:\s+\d+[.)]\s+)|\Z)',
)
_NUMBERED_PREPARED_IMAGE_SIGNAL_RE = re.compile(
    r'\b(?:'
    r'image|bild|photo|photograph|photography|foto|selfie|portrait|macro|'
    r'cinematic|camera|lens|lighting|render|illustration|poster|scene|shot|'
    r'close[-\s]?up|wide[-\s]?angle|hyper[-\s]?realistic'
    r')\b',
    re.IGNORECASE,
)
_SOCIAL_MANIFEST_HANDLE_RE = re.compile(
    r'^(?:[-*#>\u2022]+\s*)?(?:\*\*+|__+|\*+|`+)?\s*@(?P<handle>[A-Za-z0-9][\w.-]{1,100})'
)
_HEX_BYTE_FRAGMENT_RE = re.compile(r'(?:<0x[0-9a-fA-F]{2}>)+')
_SOCIAL_MANIFEST_VISUAL_SIGNAL_RE = re.compile(
    r'\b(?:'
    r'image|photo|photograph|photography|selfie|portrait|macro|micro|cinematic|'
    r'camera|lens|lighting|glow|bokeh|underwater|aerial|wide[-\s]?angle|'
    r'close[-\s]?up|ultra[-\s]?wide|extreme|chiaroscuro|high[-\s]?contrast|'
    r'saturated|vibrant|shallow\s+depth|depth\s+of\s+field|motion\s+blur|'
    r'texture|textures|reflection|reflections|backlight|sunlight|forest|jungle|'
    r'desert|ocean|canopy|macro\s+lens'
    r')\b',
    re.IGNORECASE,
)
_SOCIAL_MANIFEST_CAPTION_COPY_RE = re.compile(
    r'\b(?:'
    r'my|me|mine|ich|mein|meine|mich|posted|post|snacks?|vibes?|lifestyle|'
    r'goals?|dreams?|mood|waiting|stay|just|found|living|levels?\s+are|'
    r'off\s+the\s+charts|isn[\'’]?t\s+just|hungry|nap\s+time'
    r')\b',
    re.IGNORECASE,
)
_SOCIAL_ASSET_FILENAME_STOPWORDS = {
    'asset',
    'assets',
    'background',
    'banner',
    'bg',
    'card',
    'cover',
    'file',
    'gallery',
    'global',
    'hero',
    'image',
    'img',
    'main',
    'photo',
    'pic',
    'picture',
    'post',
    'selfie',
    'snout',
    'thumb',
    'thumbnail',
    'visual',
}
_MARKDOWN_SEPARATOR_LINE_RE = re.compile(r'^\s*(?:\*{3,}|-{3,}|_{3,})\s*$')
_ACTIVE_LATE_FILL_STATUSES = {'pending', 'queued', 'running', 'scheduled', 'accepted'}
_WAIVED_OBLIGATION_STATUSES = {
    'waived',
    'not_needed',
    'not-needed',
    'not_needed_verified',
    'skipped_verified',
    'unnecessary_verified',
}
_SUPERSEDED_OBLIGATION_STATUSES = {
    'obsolete',
    'replaced',
    'superseded',
    'no-longer-relevant',
    'no_longer_relevant',
}
_RESERVED_CANDIDATE_STATUSES = {
    'draft',
    'optional',
    'possible',
    'reserved',
}
_NON_EXECUTABLE_BRANCH_STATUSES = {
    'candidate',
    'deferred_not_executable',
    'future',
    'future_candidate',
    'held',
    'non_executable',
    'not_executable',
    'optional',
    'possible',
    'reserved',
    'reserved_candidate',
    'skipped',
}
_TERMINAL_FULFILLED_BRANCH_STATUSES = {
    'canceled',
    'cancelled',
    'completed',
    'fulfilled',
    'skipped',
    'superseded',
    'waived',
}
_NON_EXECUTABLE_BRANCH_ROLES = {
    'candidate',
    'deferred_candidate',
    'future_candidate',
    'non_executable',
    'optional_output',
    'possible_future',
    'preparation_output',
    'reserved_candidate',
}
_NON_EXECUTABLE_POLICY_TOKENS = {
    'blocked_until_user_promotion',
    'candidate_only',
    'manual_promotion_required',
    'non_executable',
    'non_executable_until_promoted',
    'reserved_for_later',
}
_DEFER_DOWNSTREAM_PROMPT_RE = re.compile(
    r'\b('
    r'do\s+not\s+(?:execute|run|generate|create|make|materiali[sz]e)|'
    r'don[\'’]?t\s+(?:execute|run|generate|create|make|materiali[sz]e)|'
    r'not\s+(?:yet|now)|hold\s+(?:it|them|this|that|back|off)|keep\s+(?:it|them|this|that)?\s*(?:for\s+later|reserved)|'
    r'possible\s+later|later\s+(?:step|steps|phase|phases|turn)|'
    r'nicht\s+ausf(?:ue|ü|u)hren|nicht\s+(?:erzeugen|generieren|erstellen)|noch\s+nicht\s+ausgef(?:ue|ü|u)hrt|'
    r'nur\s+vorbereiten|vorbereitet,\s*aber\s+noch\s+nicht|'
    r'zur(?:ue|ü|u)ckhalten|halte\s+[^.;!?]{0,120}\s+zur(?:ue|ü|u)ck|'
    r'm(?:oe|ö|o)gliche\s+sp(?:ae|ä|a)tere\s+schritte|sp(?:ae|ä|a)tere\s+(?:schritte|phasen)'
    r')\b',
    re.IGNORECASE,
)
_CURRENT_ONLY_PROMPT_RE = re.compile(
    r'\b('
    r'only\s+(?:the\s+)?(?:first|current)\s+phase|execute\s+only\s+(?:the\s+)?(?:first|current)|'
    r'generate\s+only\s+(?:the\s+)?(?:first|current)|visible\s+result|'
    r'nur\s+(?:die\s+)?(?:erste|aktuelle)\s+phase|erzeuge\s+nur\s+(?:die\s+)?erste\s+phase|'
    r'nur\s+[^.;!?]{0,80}\s+als\s+sichtbares\s+ergebnis|sichtbares\s+ergebnis'
    r')\b',
    re.IGNORECASE,
)
_TEXT_ARTIFACT_EXTENSION_BY_MIME = {
    'application/javascript': 'js',
    'application/json': 'json',
    'text/css': 'css',
    'text/html': 'html',
    'text/javascript': 'js',
    'text/markdown': 'md',
    'text/plain': 'txt',
}
_TEXT_ARTIFACT_EXTENSION_ALIASES = {
    'htm': 'html',
    'markdown': 'md',
    'plain': 'txt',
    'text': 'txt',
    'xhtml': 'html',
}
_DETERMINISTIC_REVIEW_CRITERIA = {
    'consumes_declared_input_refs',
    'does_not_restart_root_request',
    'output_contract_matches_capability',
    'preparation_text_is_bounded_to_downstream_inputs',
    'runtime_artifact_exists_when_fulfilled',
    'runtime_evidence_text_exists_when_fulfilled',
    'runtime_status_reaches_fulfilled_blocked_failed_waived_or_pending',
    'runtime_status_reaches_fulfilled_blocked_failed_waived_superseded_or_pending',
    'runtime_text_artifact_exists_when_fulfilled',
    'runtime_text_exists_when_fulfilled',
    'uses_dependency_evidence',
}
_STRUCTURED_JSON_LIST_CONTRACT_RE = re.compile(
    r'\bjson\s*(?:[-_/]\s*|\s+)(?:list(?:e)?|array)\b',
    re.IGNORECASE,
)
_STRUCTURED_JOIN_ENUMERATED_LABEL_RE = re.compile(
    r'(?<![A-Za-z0-9_])(?P<label>[A-Za-z]|[0-9]+)\s*[-–—]\s*(?=\S)'
)
_STRUCTURED_JOIN_FOR_LABELS_RE = re.compile(
    r'\b(?:(?i:für|for))\s+'
    r'(?P<body>(?:[A-Za-z]|[0-9]+)'
    r'(?:\s*(?:,|/)\s*(?:[A-Za-z]|[0-9]+))*'
    r'(?:\s*,?\s*(?:(?i:und|and)|&)\s*(?:[A-Za-z]|[0-9]+))?)\b'
)
_STRUCTURED_JOIN_FIELDS_RE = re.compile(
    r'\b(?:fields?|felder(?:n)?)\b(?P<body>[^.;!\n]{0,180})',
    re.IGNORECASE,
)
_STRUCTURED_JOIN_FIELD_STOP_WORDS = {
    'and',
    'are',
    'den',
    'die',
    'mit',
    'oder',
    'or',
    'sind',
    'the',
    'und',
    'with',
}
_STRUCTURED_JOIN_ARTIFACT_REF_FIELDS = {
    'artifact_path',
    'artifact_ref',
    'path',
    'ref',
}
_STRUCTURED_JOIN_CONSUMER_PRODUCER_TYPES = {
    CAPABILITY_VISION_ANALYSIS: (CAPABILITY_IMAGE_GENERATION, 'image'),
    CAPABILITY_SPEECH_TO_TEXT: (CAPABILITY_TEXT_TO_SPEECH, 'audio'),
}
_STRUCTURED_JOIN_DIRECT_PRODUCER_TYPES = {
    CAPABILITY_IMAGE_GENERATION: 'image',
    CAPABILITY_TEXT_TO_SPEECH: 'audio',
}


def _bounded_visual_evidence_from_selected_message(content: Any) -> str:
    """Extract only the bounded visual-evidence section from a carried message."""

    text = str(content or '').strip()
    if not text:
        return ''

    evidence_values: list[str] = []

    def collect_json_evidence(value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = re.sub(
                    r'[^a-z0-9]+',
                    '_',
                    str(raw_key or '').casefold(),
                ).strip('_')
                is_visual = any(
                    token in key
                    for token in ('image', 'visual', 'bild', 'sichtbar')
                )
                is_evidence = any(
                    token in key
                    for token in ('evidence', 'analysis', 'analyse', 'detail')
                )
                if (
                    is_visual
                    and is_evidence
                    and isinstance(child, str)
                    and child.strip()
                ):
                    normalized = re.sub(r'\s+', ' ', child).strip()
                    if normalized not in evidence_values:
                        evidence_values.append(normalized)
                collect_json_evidence(child)
        elif isinstance(value, list):
            for child in value:
                collect_json_evidence(child)

    for match in re.finditer(r'(?is)```(?:json)?\s*(\{.*?\})\s*```', text):
        try:
            collect_json_evidence(json.loads(match.group(1)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if evidence_values:
        return '\n'.join(evidence_values)[:6000].rstrip()

    heading_match = re.search(
        r'(?ims)^\s*(?:#{1,6}\s*)?(?:\*\*)?'
        r'(?:visible\s+(?:image\s+)?(?:evidence|details?)|image\s+(?:evidence|analysis)|'
        r'sichtbar\w*\s+(?:bild)?(?:evidenz|details?)|bildevidenz|bildanalyse)'
        r'(?:\*\*)?\s*:\s*(?P<body>.*?)'
        r'(?=^\s*(?:#{1,6}\s+|\*\*)[^\n]{1,100}(?:\*\*)?\s*:?\s*$|\Z)',
        text,
    )
    if not heading_match:
        return ''
    body = re.sub(
        r'^\s*(?:\*\*|__)\s*',
        '',
        str(heading_match.group('body') or ''),
    )
    return re.sub(r'\s+', ' ', body).strip()[:6000].rstrip()


def _normalize_text_artifact_extension(value: Any) -> str:
    token = str(value or '').strip().lower().lstrip('.')
    return _TEXT_ARTIFACT_EXTENSION_ALIASES.get(token, token)


def _extension_from_path_like(value: Any) -> str:
    token = str(value or '').strip().split('?', 1)[0].split('#', 1)[0]
    if not token or '.' not in token.rsplit('/', 1)[-1]:
        return ''
    return _normalize_text_artifact_extension(token.rsplit('/', 1)[-1].rsplit('.', 1)[-1])


def _text_artifact_extension_from_record(record: Mapping[str, Any]) -> str:
    request_payload = record.get('text_artifact_request') if isinstance(record.get('text_artifact_request'), Mapping) else {}
    for value in (
        record.get('text_artifact_extension'),
        request_payload.get('extension'),
        _TEXT_ARTIFACT_EXTENSION_BY_MIME.get(str(record.get('mime_type') or '').strip().lower()),
        _extension_from_path_like(record.get('path') or record.get('source_path') or record.get('saved_text_path')),
        _extension_from_path_like(record.get('name')),
    ):
        normalized = _normalize_text_artifact_extension(value)
        if normalized:
            return normalized
    return ''


def _artifact_path(record: Mapping[str, Any]) -> str:
    return str(record.get('path') or record.get('source_path') or record.get('saved_text_path') or '').strip()


def _ollmo_relative_path(value: Any) -> str:
    raw = str(value or '').strip()
    if not raw:
        return ''
    try:
        target = Path(raw).expanduser()
        if not target.is_absolute():
            return raw
        repo_root = Path(__file__).resolve().parent.parent
        candidates: list[str] = []
        for root in (Path.cwd(), repo_root):
            try:
                relative = target.resolve().relative_to(root.resolve())
            except (OSError, RuntimeError, ValueError):
                continue
            normalized = relative.as_posix()
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        if candidates:
            return min(candidates, key=len)
    except (OSError, RuntimeError, ValueError):
        pass
    return raw


def _artifact_source_name(record: Mapping[str, Any]) -> str:
    request_payload = record.get('text_artifact_request') if isinstance(record.get('text_artifact_request'), Mapping) else {}
    path = _artifact_path(record)
    fallback_name = Path(path).stem if path else ''
    return str(
        record.get('name')
        or record.get('source_name')
        or request_payload.get('source_name')
        or fallback_name
        or ''
    ).strip()


def _artifact_link_tokens(record: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for value in (
        _artifact_path(record),
        str(record.get('source_path') or '').strip(),
        str(record.get('url') or '').strip(),
        str(record.get('name') or '').strip(),
    ):
        if value and value not in tokens:
            tokens.append(value)
        if value:
            basename = Path(value.split('?', 1)[0].split('#', 1)[0]).name
            if basename and basename not in tokens:
                tokens.append(basename)
    return [token for token in tokens if len(token) >= 3]


def _expected_text_artifact_extension(record: Mapping[str, Any]) -> str:
    request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
    for value in (
        record.get('text_artifact_extension'),
        request_payload.get('extension'),
    ):
        normalized = _normalize_text_artifact_extension(value)
        if normalized:
            return normalized
    return ''


def _expected_text_artifact_source_name(record: Mapping[str, Any]) -> str:
    request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
    return str(
        record.get('text_artifact_source_name')
        or request_payload.get('source_name')
        or ''
    ).strip().lower()
_FALSE_LOCAL_ARTIFACT_CLAIM_RE = re.compile(
    r'(?is)('
    r'\[artifact:\s*[^\]]+\]|'
    r'artifact created|'
    r'artifact generated|'
    r'saved locally|'
    r'saved as a local artifact|'
    r'saved into a final artifact container|'
    r'downloadable artifact|'
    r'download and run this file|'
    r'ready-to-run artifact|'
    r'ready to be downloaded'
    r')'
)
_TEXT_ONLY_MATERIALIZATION_CLAIM_RE = re.compile(
    r'(?is)('
    r'materiali[sz]ed artifacts?|'
    r'materialisierte artefakte|'
    r'materialisiertes artefakt|'
    r'artefakt\s*\d+[^.\n]{0,120}\b(?:fulfilled|erfüllt|erfuellt|materialisiert)\b|'
    r'artefakt\s*\d+[^.\n]{0,160}\b(?:nicht\s+materialisiert|not\s+materiali[sz]ed|'
    r'deferred|offen|open|failed|fehlgeschlagen)\b|'
    r'(?:bild|image)\s*\d+[^.\n]{0,80}\b(?:fulfilled|erfüllt|erfuellt|materialisiert)\b'
    r')'
)
_TEXT_ONLY_IMAGE_CAPABILITY_CLAIM_RE = re.compile(
    r'(?i)\b(?:image[_ -]?generation|image|images|bild|bilder|visual|visuell|grafik|illustration)\b'
)
_TEXT_ONLY_AUDIO_CAPABILITY_CLAIM_RE = re.compile(
    r'(?i)\b(?:text[_ -]?to[_ -]?speech|tts|audio|akustisch(?:e|er|es|en)?|'
    r'sprach(?:ausgabe|version)|speech|voice|vorlesen|hörfassung|hoerfassung)\b'
)
_VISIBLE_CONTROL_JSON_RE = re.compile(
    r'(?is)^\s*(?:```[a-z0-9_-]*\s*)?[\[{][\s\S]*\b(?:request_ir|request_phase_graph|decision_contract|'
    r'output_obligations|candidate_graph|promotion_review|workload_graph|user_facing_response)\b'
)
_INTERNAL_CONTROL_ENVELOPE_KEYS = frozenset(
    {
        'candidate_graph',
        'decision_contract',
        'output_obligations',
        'promotion_review',
        'request_ir',
        'request_phase_graph',
        'user_facing_response',
        'workload_graph',
    }
)
_EXPLICIT_CONTROL_DIAGNOSTIC_RE = re.compile(
    r'(?is)\b(?:show|display|print|return|inspect|debug|explain|dump|zeige|zeig|anzeigen|'
    r'ausgeben|prüf(?:e|en)?|pruef(?:e|en)?|debugg(?:e|en)?|erklär(?:e|en)?|erklaer(?:e|en)?)\b'
    r'[^.!?\n]{0,160}\b(?:request[_ -]?ir|request[_ -]?phase[_ -]?graph|decision[_ -]?contract|'
    r'candidate[_ -]?graph|promotion[_ -]?review|workload[_ -]?graph|control[_ -]?plane|planner[_ -]?json)\b'
)
_PHASE_OUTPUT_REPAIR_SYSTEM_MESSAGE = (
    'Ollmo phase-output repair: the previous attempt returned an internal planner/control envelope. '
    'Retry this same current phase once. Return only the substantive user-facing content required by '
    'the current phase. Do not return JSON, request IR, decision contracts, candidate graphs, route '
    'metadata, process narration, headings, or capability commentary.'
)
_PHASE_OUTPUT_REPAIR_NOTICE = (
    'Runtime truth: the current preparation phase did not produce accepted substantive user-facing '
    'content. Internal planner/control data, if any, was withheld. Downstream materialization is '
    'blocked until the branch contract is repaired.'
)
_MARKDOWN_CODE_BLOCK_RE = re.compile(r'```[^\n`]*\n.*?```', re.DOTALL)
_DEICTIC_SOURCE_TRANSFORM_RE = re.compile(
    r'(?is)\b(?:mach(?:e|en)?|make|create|generate|turn|convert|transform|show|display|'
    r'visuali[sz]e|render|use|using|zeig(?:e|en|st|t)?|darstell(?:e|en|st|t)?|'
    r'visualisier(?:e|en|st|t)?|'
    r'nutz(?:e|en|t)?|verwend(?:e|en|et)?|wandle|verwandle|erstelle|erzeuge)\b'
    r'(?:[^.!?\n]|\.(?=\S)){0,120}?\b(?:daraus|davon|hieraus|this|that|these|those|them|it|das|dies(?:e|er|es|en|em)?|sie|es)\b|'
    r'\b(?:daraus|davon|hieraus|(?:from|with|given|using)\s+(?:this|that|these|those)|'
    r'(?:mit|aus)\s+dies(?:e|er|es|en|em)?)\b'
    r'(?:[^.!?\n]|\.(?=\S)){0,120}?\b(?:mach(?:e|en)?|make|create|generate|turn|convert|transform|show|display|'
    r'visuali[sz]e|render|use|using|zeig(?:e|en|st|t)?|darstell(?:e|en|st|t)?|'
    r'visualisier(?:e|en|st|t)?|'
    r'nutz(?:e|en|t)?|verwend(?:e|en|et)?|wandle|verwandle|erstelle|erzeuge)\b'
)
_SOURCE_TRANSFORM_ARTIFACT_TARGET_RE = re.compile(
    r'(?i)\b(?:html|html[-_\s]?file|website|webseite|page|seite|datei|file|artifact|artefact|artefakt|'
    r'audio|speech|spoken|voice(?:\s+clip)?|sprachclip|recording|recordings|'
    r'(?:sprach)?aufnahme(?:n)?|mp3|wav|vorlesen|'
    r'image|images|picture|pictures|photo|photos|graphic|graphics|bild|bilder|foto|fotos|'
    r'grafik|grafiken|illustration|illustrations|illustrationen|poster)\b'
)
_DEICTIC_SOURCE_REFERENCE_RE = re.compile(
    r'(?i)\b(?:daraus|davon|hieraus|from\s+this|from\s+that|from\s+these|from\s+those|'
    r'this|that|these|those|them|it|das|dies(?:e|er|es|en|em)?|sie|es)\b'
)
_DEICTIC_SOURCE_TRANSFORM_ACTION_RE = re.compile(
    r'(?i)\b(?:mach(?:e|en)?|make|create|generate|turn|convert|transform|show|display|'
    r'visuali[sz]e|render|use|using|zeig(?:e|en|st|t)?|darstell(?:e|en|st|t)?|'
    r'visualisier(?:e|en|st|t)?|'
    r'nutz(?:e|en|t)?|verwend(?:e|en|et)?|wandle|verwandle|erstelle|erzeuge)\b'
)
_SAME_TURN_TEXT_SOURCE_ACTION_PATTERN = (
    r'write|draft|compose|invent|formulate|create|generate|produce|describe|tell|'
    r'schreib(?:e|en|st|t)?|verfass(?:e|en|est|t)?|'
    r'entwirf|entwerf(?:e|en|t)?|formulier(?:e|en|st|t)?|'
    r'erfind(?:e|en|est|et)?|dicht(?:e|en|est|et)?|'
    r'erstell(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|'
    r'generier(?:e|en|st|t)?|produzier(?:e|en|st|t)?|'
    r'beschreib(?:e|en|st|t)?|schilder(?:e|n|st|t)?|'
    r'erzähl(?:e|en|st|t)?|erzaehl(?:e|en|st|t)?'
)
_SAME_TURN_TEXT_SOURCE_NOUN_PATTERN = (
    r'scene(?:\s+text)?|place|location|setting|concept|idea|design|'
    r'source\s+text|starting\s+text|story|stories|sentence|description|'
    r'prompt|slogan|script|narration|poem|announcement|warning(?:\s+text)?|notice|summary|'
    r'report|article|e[-\s]?mail|message|copy|text|'
    r'(?:szenen|quell|ausgangs|produkt|werbe|warn)text(?:e|en|s)?|'
    r'geschichte(?:n)?|[\wäöüß-]{0,32}szene(?:n)?|ort|schauplatz|umgebung|konzept|idee|entwurf|'
    r'satz|sätze|saetze|beschreibung(?:en)?|'
    r'warnslogan|prompt|slogan|skript(?:e|en|s)?|erzählung(?:en)?|erzaehlung(?:en)?|'
    r'gedicht(?:e|en|s)?|ansage(?:n)?|warnung(?:en)?|hinweis(?:e|en)?|zusammenfassung(?:en)?|'
    r'bericht(?:e|en|s)?|artikel|e[-\s]?mail|nachricht(?:en)?'
)
_SAME_TURN_GENERATED_TEXT_SOURCE_RE = re.compile(
    rf'(?is)\b(?P<action>{_SAME_TURN_TEXT_SOURCE_ACTION_PATTERN})\b'
    rf'[^.!?\n]{{0,180}}?\b(?P<source>{_SAME_TURN_TEXT_SOURCE_NOUN_PATTERN})\b'
)
_SAME_TURN_ARTIFACT_SOURCE_ACTION_PATTERN = (
    r'create|generate|produce|draw|paint|illustrate|sketch|render|make|synthesize|'
    r'mach(?:e|en)?|erstell(?:e|en|st|t)?|erzeug(?:e|en|st|t)?|'
    r'generier(?:e|en|st|t)?|produzier(?:e|en|st|t)?|'
    r'zeichn(?:e|en|est|et)?|mal(?:e|en|st|t)?|'
    r'illustrier(?:e|en|st|t)?|skizzier(?:e|en|st|t)?|render(?:e|n|st|t)?'
)
_SAME_TURN_ARTIFACT_SOURCE_NOUN_PATTERN = (
    r'image(?:s)?|picture(?:s)?|illustration(?:s)?|graphic(?:s)?|photo(?:s)?|'
    r'audio(?:s)?|recording(?:s)?|speech|voice(?:\s+clip)?|sprachclip|'
    r'bild(?:e|er|ern|es)?|grafik(?:en)?|illustration(?:en)?|foto(?:s)?|'
    r'aufnahme(?:n)?|sprachaufnahme(?:n)?'
)
_SAME_TURN_GENERATED_ARTIFACT_SOURCE_RE = re.compile(
    rf'(?is)\b(?P<action>{_SAME_TURN_ARTIFACT_SOURCE_ACTION_PATTERN})\b'
    rf'[^.!?\n]{{0,180}}?\b(?P<source>{_SAME_TURN_ARTIFACT_SOURCE_NOUN_PATTERN})\b'
)
_SAME_TURN_IMPLICIT_IMAGE_SOURCE_RE = re.compile(
    r'(?is)\b(?P<action>paint|illustrate|sketch|'
    r'mal(?:e|en|st|t)?|illustrier(?:e|en|st|t)?|skizzier(?:e|en|st|t)?)\b'
    r'(?P<source>[^.!?\n]{1,180}?)'
    r'(?=(?:\s*[,;:-]?\s*(?:and|then|next|und|sowie|dann|danach)\s+|\s*[,;:-]\s*)'
    r'(?:analy[sz](?:e|ed|ing)?|inspect|describe|review|'
    r'analysier(?:e|en|st|t)?|untersuch(?:e|en|st|t)?|beschreib(?:e|en|st|t)?)\b)'
)


def _same_turn_generated_artifact_source_matches(
    prompt_text: str,
    start: int = 0,
    end: Optional[int] = None,
) -> list[re.Match[str]]:
    bounded_end = len(prompt_text) if end is None else max(start, min(len(prompt_text), end))
    matches_by_action_start: dict[int, re.Match[str]] = {}
    for match in _SAME_TURN_GENERATED_ARTIFACT_SOURCE_RE.finditer(
        prompt_text,
        start,
        bounded_end,
    ):
        matches_by_action_start[match.start('action')] = match
    for match in _SAME_TURN_IMPLICIT_IMAGE_SOURCE_RE.finditer(
        prompt_text,
        start,
    ):
        if match.end() > bounded_end:
            continue
        matches_by_action_start.setdefault(match.start('action'), match)
    return sorted(
        matches_by_action_start.values(),
        key=lambda item: (item.start(), item.end()),
    )
_SAME_TURN_IMAGE_CONSUMER_ACTION_RE = re.compile(
    r'(?is)\b(?:analy[sz](?:e|ed|ing)?|inspect|describe|review|'
    r'analysier(?:e|en|st|t)?|untersuch(?:e|en|st|t)?|beschreib(?:e|en|st|t)?)\b'
)
_SAME_TURN_AUDIO_CONSUMER_ACTION_RE = re.compile(
    r'(?is)\b(?:transcrib(?:e|es|ed|ing)?|listen|review|'
    r'transkribier(?:e|en|st|t)?|hör(?:e|en|st|t)?|hoer(?:e|en|st|t)?)\b'
)
_DEICTIC_ARTIFACT_CONSUMER_RE = re.compile(
    r'(?is)(?:\b(?:analy[sz](?:e|ed|ing)?|inspect|describe|review|'
    r'transcrib(?:e|es|ed|ing)?|listen|'
    r'analysier(?:e|en|st|t)?|untersuch(?:e|en|st|t)?|beschreib(?:e|en|st|t)?|'
    r'transkribier(?:e|en|st|t)?|hör(?:e|en|st|t)?|hoer(?:e|en|st|t)?)\b'
    r'(?:[^.!?\n]|\.(?=\S)){0,100}?'
    r'\b(?:daraus|davon|hieraus|this|that|these|those|them|it|das|dies(?:e|er|es|en|em)?|sie|es)\b|'
    r'\b(?:in|on|from|for|using|with|bei|aus|für|fuer|mit)\s+'
    r'(?:this|that|these|those|them|das|dies(?:e|er|es|en|em)?|sie|es)\b'
    r'[^.!?\n]{0,50}?\b(?:images?|pictures?|photos?|audios?|recordings?|'
    r'bild(?:e|er|ern|es)?|audio(?:s)?|aufnahme(?:n)?)\b'
    r'[^.!?\n]{0,60}?\b(?:analy[sz](?:e|ed|ing)?|inspect|describe|review|'
    r'transcrib(?:e|es|ed|ing)?|listen|analysier(?:e|en|st|t)?|'
    r'untersuch(?:e|en|st|t)?|beschreib(?:e|en|st|t)?|'
    r'transkribier(?:e|en|st|t)?|hör(?:e|en|st|t)?|hoer(?:e|en|st|t)?)\b)'
)
_INLINE_TEXTUAL_ARTIFACT_SOURCE_RE = re.compile(
    r'(?is)\b(?P<action>review|describe|inspect|analy[sz](?:e|ed|ing)?|'
    r'beschreib(?:e|en|st|t)?|analysier(?:e|en|st|t)?)\b'
    r'[^.!?\n]{0,40}?\b(?:this|that|these|those|das|dies(?:e|er|es|en|em)?)\b\s*'
    r'(?P<source>(?:image|audio)\s+(?:prompt|concept|idea|script|text)|'
    r'html|code|document|page)\s*:\s*(?P<content>[^.;!?\n]{1,400})'
)
_INLINE_TEXTUAL_ARTIFACT_LABEL_SOURCE_RE = re.compile(
    r'(?is)(?:^|[.!?;\n])\s*'
    r'(?P<label>(?:this\s+|dieser?\s+|dieses\s+)?'
    r'(?:image\s+(?:prompt|concept|idea)|audio\s+(?:script|text|prompt)|'
    r'bild(?:prompt|konzept|idee)|audio(?:skript|text|prompt)|'
    r'script|skript|html|code|document|dokument|page|seite))\s*:\s*'
    r'(?P<content>[^.;!?\n]{1,400})'
)
_ARTIFACT_SOURCE_NOUN_RE = re.compile(
    rf'(?is)\b(?:{_SAME_TURN_ARTIFACT_SOURCE_NOUN_PATTERN})\b'
)
_SAME_TURN_ARTIFACT_CONTEXT_PREFIX_RE = re.compile(
    r'(?is)^\s*(?:(?:'
    r'for\s+this\s+(?:test|request|task|experiment|exercise|demo)|'
    r'in\s+this\s+(?:test|request|task|experiment|exercise|demo)|'
    r'as\s+part\s+of\s+this\s+(?:test|request|task|experiment|exercise|demo)|'
    r'as\s+(?:a\s+)?(?:(?:quick|brief|small)\s+)?(?:test|experiment|exercise|demo)|'
    r'(?:in\s+)?step\s+1|to\s+begin|'
    r'für\s+diese(?:n|r)?\s+(?:test|anfrage|aufgabe|versuch|experiment|demo)|'
    r'in\s+dieser\s+(?:aufgabe|übung|uebung|demo)|'
    r'im\s+ersten\s+schritt|'
    r'als\s+(?:kurze(?:n|r|s)?\s+)?(?:test|versuch|experiment|demo)'
    r')\s*[,;:-]?\s*)$'
)
_EXTERNAL_ARTIFACT_SOURCE_LOCATOR_RE = re.compile(
    r'(?is)(?:\b[A-Za-z][A-Za-z0-9+.-]{1,31}://[^\s,;]+|'
    r'\b(?:data|blob|urn|magnet|ipfs):[^\s,;]+|'
    r'(?:^|\s)(?:\.\.?/|~/|/|\$[A-Za-z_][A-Za-z0-9_]*[\\/])(?:[^\s,;]+)|'
    r'(?:^|\s)\\\\[^\s\\]+\\[^\s,;]+|'
    r'\b(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+:[^\s,;]+|'
    r'\b[A-Za-z]:[\\/][^\s,;]+|\b[^\s,;]+\.(?:png|jpe?g|gif|webp|svg|wav|mp3|m4a|flac|ogg|html?|json|txt|md)\b|'
    r'\b(?:provided|supplied|uploaded|attached|selected|stored|hosted|fetched|downloaded)\s+'
    r'(?:by|in|at|on|from)\b|'
    r'\b(?:from|via|at|under|aus|von|über|ueber)\s+(?:an?\s+|the\s+|einer?\s+|dem\s+|der\s+)?'
    r'(?:remote|external|url|uri|path|web|internet|server|s3|bucket|storage|'
    r'extern(?:en|er|es|e)?|entfernt(?:en|er|es|e)?|pfad|netz|server)\b)'
)
_GENERATED_ARTIFACT_REFERENCE_RE = re.compile(
    r'(?i)\b(?:generated|created|rendered|produced|'
    r'erzeugt(?:e|en|er|es)?|erstellt(?:e|en|er|es)?|generiert(?:e|en|er|es)?)\b'
)
_REMOTE_ARTIFACT_SOURCE_QUALIFIER_RE = re.compile(
    r'(?i)\b(?:remote|external|url|uri|path|web|internet|server|'
    r'extern(?:e|en|er|es)?|entfernt(?:e|en|er|es)?|pfad)\b'
)
_SENTENCE_BOUNDARY_RE = re.compile(r'[.!?](?=\s|$)|\n')
_SAME_TURN_MODAL_TEXT_SOURCE_RE = re.compile(
    rf'(?is)\b(?P<directive>'
    rf'(?:kannst|könntest|koenntest|würdest|wuerdest)\s+du|'
    rf'du\s+(?:sollst|musst|kannst)|lass\s+uns|bitte'
    rf')\b[^.!?\n]{{0,120}}?\b(?P<source>{_SAME_TURN_TEXT_SOURCE_NOUN_PATTERN})\b'
    rf'[^.!?\n]{{0,80}}?\b(?P<action>{_SAME_TURN_TEXT_SOURCE_ACTION_PATTERN})\b'
)
_SAME_TURN_SOURCE_LEADING_NEGATION_RE = re.compile(
    r'(?is)\b(?:do\s+not|don[\'’]?t|never|without|no|not|'
    r'kein(?:e|en|er|es)?|nicht|ohne)\b'
    r'(?:\s+(?:ever|ever\s+again|please|bitte|under\s+any\s+circumstances))?\s*$'
)
_SAME_TURN_SOURCE_BETWEEN_NEGATION_RE = re.compile(
    r'(?is)\b(?:'
    r'no\s+(?!less\b)|'
    r'not\s+(?:a|an|any|the|this|that)\b|'
    r'without\s+(?:a|an|any|the|this|that)?\s*$|'
    r'kein(?:e|en|er|es)?\b|'
    r'nicht\s+(?:ein(?:e|en|em|er|es)?|den|die|das|diese(?:n|r|s|m)?)\b|'
    r'ohne\s+(?:ein(?:e|en|em|er|es)?)?\s*$'
    r')'
)
_SAME_TURN_SOURCE_DIRECT_PREFIX_RE = re.compile(
    r'(?is)^\s*(?:(?:please|kindly|first|then|next|now|and|'
    r'bitte|zuerst|dann|danach|nun|jetzt|und|anschließend|anschliessend)'
    r'[\s,:-]*)*$'
)
_SAME_TURN_SOURCE_YOU_DIRECTED_PREFIX_RE = re.compile(
    r'(?is)^\s*(?:please\s+|bitte\s+)?(?:'
    r'(?:can|could|would|will|should)\s+you\b|'
    r'you\s+(?:should|must|need\s+to|can|could|will|shall)\b|'
    r'i\s+(?:want|need|ask)\s+you\s+to\b|'
    r'(?:kannst|könntest|koenntest|würdest|wuerdest)\s+du\b|'
    r'du\s+(?:sollst|musst|kannst)\b|'
    r'ich\s+(?:möchte|moechte|will)\s*,?\s*dass\s+du\b'
    r')[^.!?\n]{0,100}$'
)
_SAME_TURN_SOURCE_AFFIRMATIVE_NEGATION_PREFIX_RE = re.compile(
    r'(?is)^\s*(?:do\s+not|don[\'’]?t|never)\s+(?:hesitate|forget|fail)\s+to\s*$'
)
_SAME_TURN_SOURCE_META_PREFIX_RE = re.compile(
    r'(?is)(?:'
    r'\b(?:how\s+to|whether\s+to|wie\s+man|ob\s+man)\b|'
    r'\b(?:example|for\s+example|hypothetical|quoted\s+instruction|'
    r'beispiel|zum\s+beispiel|hypothetisch|zitierte\s+anweisung)\b|'
    r'\bif\s+(?:a|the|someone)\s+user\s+(?:asks|says)\b|'
    r'\bwenn\s+(?:ein|der)\s+user\s+(?:fragt|sagt)\b'
    r')[^.!?\n]{0,120}$'
)
_SAME_TURN_SOURCE_COORDINATED_TARGET_RE = re.compile(
    r'(?is)^\s*,?\s*(?:and|und|sowie|plus)\s+'
    r'(?:(?:an?|the|ein(?:e|en|em|er|es)?)\s+)?'
    r'(?:html|website|webseite|page|seite|file|datei|artifact|artefact|artefakt|'
    r'image|picture|bild|audio|speech|mp3|wav)\b[^.!?\n]{0,80}$'
)
_EXPLICIT_DEICTIC_SOURCE_NOUN_RE = re.compile(
    rf'(?is)\b(?P<noun>{_SAME_TURN_TEXT_SOURCE_NOUN_PATTERN}|'
    r'files?|artifacts?|artefacts?|documents?|html|code|pages?|images?|pictures?|audios?|recordings?|'
    r'datei(?:en)?|artefakt(?:e|en)?|dokument(?:e|en)?|html|code|seite(?:n)?|'
    r'bild(?:e|er|ern|es)?|audio(?:s)?|aufnahme(?:n)?'
    r')\b'
)
_COMPETING_DEICTIC_SOURCE_QUALIFIER_RE = re.compile(
    r'(?i)\b(?:old|previous|existing|uploaded|attached|selected|different|another|other|'
    r'unrelated|named|specific|referenced|missing|local|alt|vorherig|bestehend|'
    r'hochgeladen|angehäng|angehaeng|referenziert|fehlend|lokal|'
    r'ausgewählt|ausgewaehlt|ander|weitere|fremd|unabhängig|unabhaengig|genannt|bestimmt)'
)
_EXTERNAL_DEICTIC_SOURCE_NOUN_RE = re.compile(
    r'(?i)\b(?:files?|artifacts?|artefacts?|documents?|html|code|pages?|images?|pictures?|'
    r'audios?|recordings?|datei(?:en)?|artefakt(?:e|en)?|dokument(?:e|en)?|'
    r'seite(?:n)?|bild(?:e|er|ern|es)?|aufnahme(?:n)?)\b'
)
_DEICTIC_SOURCE_CONNECTOR_RE = re.compile(r'(?i)\b(?:into|to|as|in|zu|als)\b')


def _strip_fenced_json_boundary(text: str) -> str:
    stripped = str(text or '').strip()
    match = re.fullmatch(r'```[a-z0-9_-]*\s*\n(?P<body>[\s\S]*?)\n?```', stripped, flags=re.IGNORECASE)
    if match:
        return match.group('body').strip()
    return stripped


def _parse_visible_json_boundary(text: str) -> Any:
    candidate = _strip_fenced_json_boundary(text)
    if not candidate or candidate[0:1] not in {'{', '['}:
        return None
    try:
        return json.loads(candidate)
    except (TypeError, ValueError):
        return None


def _control_envelope_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    normalized_keys = {
        str(key or '').strip().lower()
        for key in value.keys()
        if str(key or '').strip()
    }
    return bool(normalized_keys.intersection(_INTERNAL_CONTROL_ENVELOPE_KEYS))


def visible_control_json_envelope(text: str) -> bool:
    """Return whether *text* is one complete internal control JSON envelope.

    Parsing the outer JSON boundary avoids treating ordinary prose that merely
    mentions an internal field name as control data. Lists are accepted only
    when at least one direct member is itself a control envelope.
    """

    if not _VISIBLE_CONTROL_JSON_RE.search(str(text or '')):
        return False
    parsed = _parse_visible_json_boundary(text)
    if _control_envelope_mapping(parsed):
        return True
    return bool(
        isinstance(parsed, list)
        and any(_control_envelope_mapping(item) for item in parsed)
    )


def control_json_envelope_suspected(text: str) -> bool:
    """Catch complete and truncated JSON-like internal control boundaries."""

    return bool(_VISIBLE_CONTROL_JSON_RE.search(str(text or '')))


def _collect_control_json_content_payloads(value: Any) -> list[str]:
    payloads: list[str] = []
    if isinstance(value, Mapping):
        raw_payload = value.get('content_payload')
        if isinstance(raw_payload, str) and raw_payload.strip():
            payloads.append(raw_payload.strip())
        for child in value.values():
            payloads.extend(_collect_control_json_content_payloads(child))
    elif isinstance(value, list):
        for child in value:
            payloads.extend(_collect_control_json_content_payloads(child))
    return payloads


def _visible_control_json_substantive_payload(text: str) -> str:
    parsed = _parse_visible_json_boundary(text)
    if not visible_control_json_envelope(text):
        return ''
    payloads = [
        payload.strip()
        for payload in _collect_control_json_content_payloads(parsed)
        if payload.strip()
    ]
    if len(payloads) != 1:
        return ''
    substantive = payloads[0]
    if visible_control_json_envelope(substantive):
        return ''
    return substantive


def classify_phase_output_text(text: str) -> dict[str, Any]:
    """Classify one complete model phase result without retaining its bytes."""

    source_text = str(text or '').strip()
    source_bytes = source_text.encode('utf-8')
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if not source_text:
        return {
            'status': 'repair_required',
            'accepted_text': '',
            'source_sha256': source_sha256,
            'source_bytes': 0,
            'reason': 'current phase did not return substantive text',
        }
    if control_json_envelope_suspected(source_text) and not visible_control_json_envelope(source_text):
        return {
            'status': 'repair_required',
            'accepted_text': '',
            'source_sha256': source_sha256,
            'source_bytes': len(source_bytes),
            'reason': 'malformed or truncated internal control envelope is not substantive phase text',
        }
    if not visible_control_json_envelope(source_text):
        return {
            'status': 'accepted',
            'accepted_text': source_text,
            'source_sha256': source_sha256,
            'source_bytes': len(source_bytes),
            'reason': 'phase output is not an internal control envelope',
        }
    substantive_payload = _visible_control_json_substantive_payload(source_text)
    if substantive_payload:
        return {
            'status': 'unwrapped',
            'accepted_text': substantive_payload,
            'source_sha256': source_sha256,
            'source_bytes': len(source_bytes),
            'accepted_sha256': hashlib.sha256(substantive_payload.encode('utf-8')).hexdigest(),
            'accepted_bytes': len(substantive_payload.encode('utf-8')),
            'reason': 'one safe substantive content_payload was extracted from the control envelope',
        }
    return {
        'status': 'repair_required',
        'accepted_text': '',
        'source_sha256': source_sha256,
        'source_bytes': len(source_bytes),
        'reason': 'internal control envelope did not contain exactly one safe substantive content_payload',
    }


def phase_output_acceptance_metadata(
    attempts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build bounded response metadata from one or two classification attempts."""

    normalized_attempts: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            continue
        normalized_attempts.append(
            {
                key: value
                for key, value in {
                    'attempt': index,
                    'status': str(attempt.get('status') or '').strip() or 'repair_required',
                    'source_sha256': str(attempt.get('source_sha256') or '').strip() or None,
                    'source_bytes': int(attempt.get('source_bytes') or 0),
                    'accepted_sha256': str(attempt.get('accepted_sha256') or '').strip() or None,
                    'accepted_bytes': int(attempt.get('accepted_bytes') or 0),
                    'reason': str(attempt.get('reason') or '').strip() or None,
                }.items()
                if value not in (None, '', [], {})
            }
        )
    final_attempt = attempts[-1] if attempts else {}
    final_status = str(final_attempt.get('status') or '').strip() or 'repair_required'
    status = (
        'accepted_after_retry'
        if len(normalized_attempts) > 1 and final_status in {'accepted', 'unwrapped'}
        else final_status
    )
    accepted_text = str(final_attempt.get('accepted_text') or '')
    accepted_bytes = accepted_text.encode('utf-8')
    return {
        key: value
        for key, value in {
            'kind': 'ollmo.phase_output_acceptance',
            'version': 1,
            'status': status,
            'attempt_count': len(normalized_attempts),
            'attempts': normalized_attempts,
            'accepted_sha256': (
                hashlib.sha256(accepted_bytes).hexdigest()
                if accepted_text
                else None
            ),
            'accepted_bytes': len(accepted_bytes) if accepted_text else None,
            'runtime_effect': (
                'materialization_blocked'
                if final_status == 'repair_required'
                else 'canonical_phase_text'
            ),
        }.items()
        if value not in (None, '', [], {})
    }


def phase_output_repair_system_message() -> dict[str, str]:
    return {'role': 'system', 'content': _PHASE_OUTPUT_REPAIR_SYSTEM_MESSAGE}


def phase_output_repair_notice() -> str:
    return _PHASE_OUTPUT_REPAIR_NOTICE


def attach_phase_output_acceptance(
    payload: Optional[dict[str, Any]],
    attempts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach bounded acceptance truth and a fail-closed guard to a response."""

    updated = dict(payload or {})
    runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), Mapping) else {}
    metadata = phase_output_acceptance_metadata(attempts)
    runtime['phase_output_acceptance'] = metadata
    final_status = str((attempts[-1] if attempts else {}).get('status') or '').strip()
    if final_status == 'repair_required':
        guard_reason = str((attempts[-1] if attempts else {}).get('reason') or '').strip()
        runtime['truth_guard'] = {
            'kind': (
                'control_json_boundary'
                if 'control envelope' in guard_reason
                else 'phase_output_boundary'
            ),
            'status': 'repair_required',
            'reason': guard_reason
            or 'current phase did not return accepted substantive content',
            'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
            'runtime_effect': 'materialization_blocked',
        }
    updated['runtime'] = runtime
    return updated


def _phase_graph_from_runtime_payload(route_payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    route = route_payload if isinstance(route_payload, Mapping) else {}
    runtime = route.get('route_runtime') if isinstance(route.get('route_runtime'), Mapping) else {}
    graph = runtime.get('request_phase_graph') if isinstance(runtime.get('request_phase_graph'), Mapping) else {}
    return dict(graph)


def phase_output_is_graph_preparation(
    *,
    route_payload: Optional[Mapping[str, Any]],
    request_payload: Optional[Mapping[str, Any]],
    capability: Optional[str],
) -> bool:
    """Return whether the current chat phase directly prepares downstream work."""

    if normalize_capability(capability) != CAPABILITY_CHAT:
        return False
    request_info = request_payload if isinstance(request_payload, Mapping) else {}
    graph = _phase_graph_from_runtime_payload(route_payload)
    if not graph:
        prompt = str(
            extract_responses_current_turn_prompt(request_info)
            or request_info.get('prompt')
            or request_info.get('input')
            or ''
        ).strip()
        graph = build_request_phase_graph(
            prompt,
            intent_prompt=extract_responses_current_turn_prompt(request_info),
            request_payload=dict(request_info),
            route_payload=dict(route_payload or {}),
        )
    if not current_phase_is_graph_resolved(graph) or current_phase_capability(graph) != CAPABILITY_CHAT:
        return False
    current_phase_id = str(graph.get('current_phase_id') or '').strip()
    current_phase = next(
        (
            item
            for item in (graph.get('phases') or [])
            if isinstance(item, Mapping)
            and str(item.get('phase_id') or '').strip() == current_phase_id
        ),
        {},
    )
    phase_kind = str(current_phase.get('kind') or '').strip().lower()
    phase_role = str(current_phase.get('role') or '').strip().lower()
    return bool(
        downstream_phase_capabilities(graph)
        and (phase_kind == 'prepare' or phase_role == 'text_preparation')
    )


def request_explicitly_allows_control_diagnostics(
    *,
    request_payload: Optional[Mapping[str, Any]],
    route_payload: Optional[Mapping[str, Any]],
    capability: Optional[str],
) -> bool:
    """Allow explicitly requested diagnostics only when no materializer consumes them."""

    if phase_output_is_graph_preparation(
        route_payload=route_payload,
        request_payload=request_payload,
        capability=capability,
    ):
        return False
    request_info = request_payload if isinstance(request_payload, Mapping) else {}
    prompt = str(
        extract_responses_current_turn_prompt(request_info)
        or request_info.get('prompt')
        or request_info.get('input')
        or ''
    ).strip()
    return bool(prompt and _EXPLICIT_CONTROL_DIAGNOSTIC_RE.search(prompt))


def _payload_has_direct_artifact_source(*payloads: Optional[Mapping[str, Any]]) -> bool:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in (
            'input_artifacts',
            'selected_reference_artifacts',
            'selectedReferenceArtifacts',
        ):
            raw_items = payload.get(key)
            if not isinstance(raw_items, list):
                continue
            if any(
                isinstance(item, Mapping)
                and str(
                    item.get('path')
                    or item.get('source_path')
                    or item.get('url')
                    or item.get('content')
                    or ''
                ).strip()
                for item in raw_items
            ):
                return True
        raw_item = payload.get('selected_reference_artifact') or payload.get('selectedReferenceArtifact')
        if isinstance(raw_item, Mapping) and str(
            raw_item.get('path')
            or raw_item.get('source_path')
            or raw_item.get('url')
            or raw_item.get('content')
            or ''
        ).strip():
            return True
        for key in (
            'file_path',
            'route_artifact_path',
            'artifact_path',
        ):
            if str(payload.get(key) or '').strip():
                return True
    return False


def _direct_artifact_record_source_capabilities(record: Mapping[str, Any]) -> set[str]:
    if not isinstance(record, Mapping):
        return set()
    capabilities: set[str] = set()
    metadata_values = [
        str(record.get(key) or '').strip().lower()
        for key in (
            'type',
            'artifact_type',
            'output_type',
            'mime_type',
            'content_type',
        )
        if str(record.get(key) or '').strip()
    ]
    path_values = [
        str(record.get(key) or '').strip().lower()
        for key in ('path', 'source_path', 'url')
        if str(record.get(key) or '').strip()
    ]
    for source_probe in [*metadata_values, *path_values]:
        if re.search(
            r'(?:^|[\s/])image(?:/|\b)|'
            r'\.(?:png|jpe?g|gif|webp|svg|bmp|tiff?)(?:$|[?#])',
            source_probe,
            flags=re.IGNORECASE,
        ):
            capabilities.add(CAPABILITY_IMAGE_GENERATION)
        if re.search(
            r'(?:^|[\s/])audio(?:/|\b)|'
            r'\.(?:wav|mp3|m4a|flac|ogg|aac)(?:$|[?#])',
            source_probe,
            flags=re.IGNORECASE,
        ):
            capabilities.add(CAPABILITY_TEXT_TO_SPEECH)
        if re.search(
            r'(?:^|[\s/])(?:text|html|json|javascript)(?:/|\b)|'
            r'^(?:code|document|html|json|markdown|message|page|text)$|'
            r'\.(?:html?|json|txt|md|css|js|mjs|cjs|ts|py)(?:$|[?#])',
            source_probe,
            flags=re.IGNORECASE,
        ):
            capabilities.add(CAPABILITY_CHAT)
    if not capabilities and str(record.get('content') or '').strip():
        capabilities.add(CAPABILITY_CHAT)
    return capabilities


def _iter_direct_artifact_source_records(
    *payloads: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in (
            'input_artifacts',
            'selected_reference_artifacts',
            'selectedReferenceArtifacts',
        ):
            raw_items = payload.get(key)
            if isinstance(raw_items, list):
                records.extend(
                    dict(item)
                    for item in raw_items
                    if isinstance(item, Mapping)
                )
        raw_item = payload.get('selected_reference_artifact') or payload.get(
            'selectedReferenceArtifact'
        )
        if isinstance(raw_item, Mapping):
            records.append(dict(raw_item))
        for key in ('file_path', 'route_artifact_path', 'artifact_path'):
            source_path = str(payload.get(key) or '').strip()
            if source_path:
                records.append({'path': source_path})
    unique_records: list[dict[str, Any]] = []
    seen_identity_keys: set[tuple[str, str]] = set()
    for record in records:
        identity_keys: set[tuple[str, str]] = set()
        for key in ('path', 'source_path', 'url'):
            value = str(record.get(key) or '').strip()
            if value:
                identity_keys.add(('locator', value))
        for key in ('artifact_ref', 'artifact_id', 'ref', 'source_id', 'id'):
            value = str(record.get(key) or '').strip()
            if value:
                identity_keys.add(('artifact_ref', value))
        content = record.get('content')
        if isinstance(content, str) and content.strip():
            identity_keys.add(('content', content.strip()))
        if not identity_keys:
            identity_keys.add(
                (
                    'record',
                    json.dumps(record, sort_keys=True, ensure_ascii=False, default=str),
                )
            )
        if identity_keys.intersection(seen_identity_keys):
            continue
        unique_records.append(record)
        seen_identity_keys.update(identity_keys)
    return unique_records


def _transform_reference_source_capability_counts(
    prompt_text: str,
    transform_match: re.Match[str],
) -> dict[str, int]:
    reference_match, _, reference_end = _deictic_transform_reference(
        prompt_text,
        transform_match,
    )
    normalized_reference = re.sub(
        r'\s+',
        ' ',
        str(reference_match.group(0) if reference_match else '').strip().lower(),
    )
    if normalized_reference in {'daraus', 'davon', 'hieraus'}:
        return {}
    clause_end = _sentence_clause_end(prompt_text, reference_end)
    reference_tail = prompt_text[reference_end:min(clause_end, reference_end + 120)]
    boundary_matches = [
        match
        for pattern in (_DEICTIC_SOURCE_CONNECTOR_RE, _DEICTIC_SOURCE_TRANSFORM_ACTION_RE)
        for match in [pattern.search(reference_tail)]
        if match
    ]
    for consumer_re in (
        _SAME_TURN_IMAGE_CONSUMER_ACTION_RE,
        _SAME_TURN_AUDIO_CONSUMER_ACTION_RE,
    ):
        next_consumer = consumer_re.search(reference_tail)
        if next_consumer is not None:
            boundary_matches.append(next_consumer)
    if boundary_matches:
        reference_tail = reference_tail[:min(match.start() for match in boundary_matches)]
    capability_counts: dict[str, int] = {}
    number_words = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'ein': 1,
        'eine': 1,
        'zwei': 2,
        'drei': 3,
        'vier': 4,
        'fünf': 5,
        'fuenf': 5,
    }
    last_noun_end_by_capability: dict[str, int] = {}
    last_noun_count_by_capability: dict[str, int] = {}
    for noun_match in _EXPLICIT_DEICTIC_SOURCE_NOUN_RE.finditer(reference_tail):
        noun = str(noun_match.group('noun') or '').strip().lower()
        capability = _artifact_source_capability(noun)
        if capability:
            required_count = 1
            count_prefix = reference_tail[
                max(0, noun_match.start() - 24):noun_match.start()
            ]
            explicit_count = re.search(
                r'(?i)\b(?P<count>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|'
                r'ein|eine|zwei|drei|vier|fünf|fuenf)\s*$',
                count_prefix,
            )
            if explicit_count:
                raw_count = explicit_count.group('count').lower()
                required_count = (
                    int(raw_count)
                    if raw_count.isdigit()
                    else number_words.get(raw_count, 1)
                )
            elif normalized_reference in {'these', 'those', 'from these', 'from those'} or re.search(
                r'(?:images|pictures|audios|recordings|files|artifacts|artefacts|documents|pages|'
                r'dateien|artefakte|dokumente|seiten|bilder|aufnahmen)$',
                noun,
                flags=re.IGNORECASE,
            ):
                required_count = 2
            required_count = max(1, required_count)
            previous_end = last_noun_end_by_capability.get(capability)
            if previous_end is not None and not reference_tail[
                previous_end:noun_match.start()
            ].strip():
                previous_count = last_noun_count_by_capability.get(capability, 0)
                capability_counts[capability] = (
                    capability_counts.get(capability, 0)
                    - previous_count
                    + max(previous_count, required_count)
                )
                last_noun_count_by_capability[capability] = max(
                    previous_count,
                    required_count,
                )
            else:
                capability_counts[capability] = (
                    capability_counts.get(capability, 0) + required_count
                )
                last_noun_count_by_capability[capability] = required_count
            last_noun_end_by_capability[capability] = noun_match.end()
    reference_occurrence_count = (
        2
        if normalized_reference
        in {'these', 'those', 'them', 'from these', 'from those', 'sie'}
        else 1
    )
    reference_occurrence_count += len(
        list(_DEICTIC_SOURCE_REFERENCE_RE.finditer(reference_tail))
    )
    if len(capability_counts) == 1:
        capability = next(iter(capability_counts))
        capability_counts[capability] = max(
            capability_counts[capability],
            reference_occurrence_count,
        )
    elif not capability_counts and transform_match.re is _DEICTIC_ARTIFACT_CONSUMER_RE:
        consumer_text = transform_match.group(0)
        is_image_consumer = bool(
            _SAME_TURN_IMAGE_CONSUMER_ACTION_RE.search(consumer_text)
        )
        is_audio_consumer = bool(
            _SAME_TURN_AUDIO_CONSUMER_ACTION_RE.search(consumer_text)
        )
        if is_image_consumer != is_audio_consumer:
            capability_counts[
                CAPABILITY_IMAGE_GENERATION
                if is_image_consumer
                else CAPABILITY_TEXT_TO_SPEECH
            ] = reference_occurrence_count
    return capability_counts


def _transform_reference_source_capability(
    prompt_text: str,
    transform_match: re.Match[str],
) -> str:
    capability_counts = _transform_reference_source_capability_counts(
        prompt_text,
        transform_match,
    )
    return next(iter(capability_counts), '')


def _payload_has_compatible_direct_artifact_source(
    prompt_text: str,
    transform_match: re.Match[str],
    *payloads: Optional[Mapping[str, Any]],
) -> bool:
    records = _iter_direct_artifact_source_records(*payloads)
    source_capability_counts: dict[str, int] = {}
    has_direct_source = False
    direct_source_count = 0
    for record in records:
        has_source_value = bool(
            str(
                record.get('path')
                or record.get('source_path')
                or record.get('url')
                or record.get('content')
                or ''
            ).strip()
        )
        if not has_source_value:
            continue
        has_direct_source = True
        direct_source_count += 1
        record_capabilities = _direct_artifact_record_source_capabilities(record)
        if len(record_capabilities) > 1:
            return False
        for capability in record_capabilities:
            source_capability_counts[capability] = (
                source_capability_counts.get(capability, 0) + 1
            )
    expected_capability_counts = _transform_reference_source_capability_counts(
        prompt_text,
        transform_match,
    )
    if not expected_capability_counts:
        reference_match, _, _ = _deictic_transform_reference(
            prompt_text,
            transform_match,
        )
        normalized_reference = re.sub(
            r'\s+',
            ' ',
            str(reference_match.group(0) if reference_match else '').strip().lower(),
        )
        required_count = (
            2
            if normalized_reference
            in {'these', 'those', 'them', 'from these', 'from those', 'sie'}
            else 1
        )
        return has_direct_source and direct_source_count >= required_count
    return all(
        source_capability_counts.get(capability, 0) >= required_count
        for capability, required_count in expected_capability_counts.items()
    )


def _payload_has_grounded_predecessor_image_specification(
    prompt_text: str,
    transform_match: re.Match[str],
    *payloads: Optional[Mapping[str, Any]],
) -> bool:
    generation_request = bool(
        re.search(
            r'\b(?:create|generate|make|produce|render|'
            r'erstell(?:e|en|t)?|generier(?:e|en|t)?|erzeug(?:e|en|t)?|mach(?:e|en|t)?)\b'
            r'[^.;!?\n]{0,120}\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?)\b|'
            r'\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?)\b'
            r'[^.;!?\n]{0,120}\b(?:create|generate|make|produce|render|'
            r'erstell(?:e|en|t)?|generier(?:e|en|t)?|erzeug(?:e|en|t)?|mach(?:e|en|t)?)\b',
            prompt_text,
            flags=re.IGNORECASE,
        )
    )
    page_binding_request = bool(
        re.search(
            r'\b(?:link|bind|embed|insert|include|update|repair|fix|'
            r'verlink(?:e|en|t)?|verkn(?:u|ü)pf(?:e|en|t)?|einbind(?:e|en|et)?|'
            r'einf(?:u|ü)g(?:e|en|t)?|aktualisier(?:e|en|t)?|reparier(?:e|en|t)?)\b'
            r'[^.;!?\n]{0,160}\b(?:site|page|website|webseite|seite|landingpage|landing[\s-]?page|html)\b|'
            r'\b(?:site|page|website|webseite|seite|landingpage|landing[\s-]?page|html)\b'
            r'[^.;!?\n]{0,160}\b(?:link|bind|embed|insert|include|update|repair|fix|'
            r'verlink(?:e|en|t)?|verkn(?:u|ü)pf(?:e|en|t)?|einbind(?:e|en|et)?|'
            r'einf(?:u|ü)g(?:e|en|t)?|aktualisier(?:e|en|t)?|reparier(?:e|en|t)?)\b',
            prompt_text,
            flags=re.IGNORECASE,
        )
    )
    if not generation_request or not page_binding_request:
        return False

    context = next(
        (
            payload.get('current_predecessor_context')
            for payload in payloads
            if isinstance(payload, Mapping)
            and isinstance(payload.get('current_predecessor_context'), Mapping)
            and str(
                payload.get('current_predecessor_context', {}).get('status')
                or ''
            ).strip().lower()
            == 'authorized'
            and str(
                payload.get('current_predecessor_context', {}).get('authorization')
                or ''
            ).strip()
            == 'canonical_same_conversation_predecessor'
        ),
        None,
    )
    prompts = [
        str(item).strip()
        for item in ((context or {}).get('batch_prompts') or [])
        if str(item).strip()
    ]
    if not prompts:
        return False
    expected_counts = _transform_reference_source_capability_counts(
        prompt_text,
        transform_match,
    )
    if any(
        capability != CAPABILITY_IMAGE_GENERATION
        for capability in expected_counts
    ):
        return False
    if len(prompts) < max(expected_counts.values(), default=1):
        return False
    return any(
        CAPABILITY_CHAT in _direct_artifact_record_source_capabilities(record)
        and str(record.get('path') or record.get('source_path') or '').strip()
        for record in _iter_direct_artifact_source_records(*payloads)
    )


def _offset_is_inside_quote(text: str, offset: int) -> bool:
    prompt_text = str(text or '')
    bounded_offset = max(0, int(offset or 0))
    prefix = prompt_text[:bounded_offset]
    current_line = prefix[prefix.rfind('\n') + 1:]
    if re.match(r'^\s*>', current_line):
        return True
    quoted_span_patterns = (
        r'(?<!\\)"[^"\n]{1,1000}(?<!\\)"',
        r'“[^”\n]{1,1000}”',
        r'„[^“\n]{1,1000}“',
        r'«[^»\n]{1,1000}»',
        r'‹[^›\n]{1,1000}›',
        r'‘[^’\n]{1,1000}’',
        r'‚[^‘\n]{1,1000}‘',
        r"(?<![\w])'[^'\n]{1,1000}'(?![\w])",
        r'```[\s\S]{1,4000}?```',
        r'`[^`\n]{1,1000}`',
    )
    quoted_spans = [
        span_match
        for pattern in quoted_span_patterns
        for span_match in re.finditer(pattern, prompt_text)
    ]
    if any(
        span_match.start() < bounded_offset < span_match.end()
        for span_match in quoted_spans
    ):
        return True
    if len(re.findall(r'(?<!\\)"', prefix)) % 2:
        return True

    # German closing curly quotes are also English opening curly quotes.  A
    # completed German span must therefore consume its closing mark instead of
    # making all later text look quoted under the English convention (and the
    # same ambiguity exists for the single-curly pair).
    consumed_closing_offsets = {
        span_match.end() - 1
        for span_match in quoted_spans
    }
    for opening, closing in (
        ('“', '”'),
        ('„', '“'),
        ('«', '»'),
        ('‹', '›'),
        ('‘', '’'),
        ('‚', '‘'),
    ):
        opening_offsets = [
            index
            for index, char in enumerate(prefix)
            if char == opening and index not in consumed_closing_offsets
        ]
        if opening_offsets and opening_offsets[-1] > prefix.rfind(closing):
            return True
    return False


def _deictic_transform_reference(
    prompt_text: str,
    match: re.Match[str],
) -> tuple[Optional[re.Match[str]], int, int]:
    reference_match = _DEICTIC_SOURCE_REFERENCE_RE.search(match.group(0))
    if not reference_match:
        return None, match.start(), match.start()
    return (
        reference_match,
        match.start() + reference_match.start(),
        match.start() + reference_match.end(),
    )


def _action_offset_is_user_directive(prompt_text: str, action_start: int) -> bool:
    clause_start = max(
        prompt_text.rfind('.', 0, action_start),
        prompt_text.rfind('!', 0, action_start),
        prompt_text.rfind('?', 0, action_start),
        prompt_text.rfind('\n', 0, action_start),
        prompt_text.rfind(';', 0, action_start),
    ) + 1
    leading = prompt_text[clause_start:action_start]
    candidate_leadings = [leading]
    last_comma = leading.rfind(',')
    if last_comma >= 0:
        candidate_leadings.append(leading[last_comma + 1:])
    candidate_leadings.extend(
        re.sub(r'^\s*[\'"`„“”‚‘’«»‹›]+\s*', '', candidate)
        for candidate in list(candidate_leadings)
    )
    return any(
        _SAME_TURN_SOURCE_DIRECT_PREFIX_RE.fullmatch(candidate)
        or _SAME_TURN_SOURCE_YOU_DIRECTED_PREFIX_RE.fullmatch(candidate)
        or _SAME_TURN_SOURCE_AFFIRMATIVE_NEGATION_PREFIX_RE.fullmatch(candidate)
        or _SAME_TURN_ARTIFACT_CONTEXT_PREFIX_RE.fullmatch(candidate)
        for candidate in candidate_leadings
    )


def _deictic_consumer_is_inline_text_source(
    prompt_text: str,
    match: re.Match[str],
) -> bool:
    return any(
        inline_match.start() == match.start()
        and bool(str(inline_match.group('content') or '').strip())
        for inline_match in _INLINE_TEXTUAL_ARTIFACT_SOURCE_RE.finditer(
            prompt_text,
            match.start(),
            min(len(prompt_text), match.start() + 520),
        )
    )


def _deictic_consumer_reference_targets_artifact(
    prompt_text: str,
    match: re.Match[str],
) -> bool:
    reference_match, reference_start, reference_end = _deictic_transform_reference(
        prompt_text,
        match,
    )
    if _DEICTIC_SOURCE_TRANSFORM_ACTION_RE.search(
        prompt_text,
        min(match.end(), match.start() + 1),
        reference_start,
    ):
        return False
    reference = re.sub(
        r'\s+',
        ' ',
        str(reference_match.group(0) if reference_match else '').strip().lower(),
    )
    if reference in {'daraus', 'davon', 'hieraus', 'it', 'them', 'sie', 'es'}:
        return True
    boundary = _SENTENCE_BOUNDARY_RE.search(prompt_text, reference_end)
    reference_tail_end = min(
        boundary.start() if boundary else len(prompt_text),
        reference_end + 80,
    )
    reference_tail = prompt_text[reference_end:reference_tail_end]
    next_transform_action = _DEICTIC_SOURCE_TRANSFORM_ACTION_RE.search(
        reference_tail
    )
    if next_transform_action is not None:
        reference_tail = reference_tail[:next_transform_action.start()]
    noun_match = _EXPLICIT_DEICTIC_SOURCE_NOUN_RE.search(reference_tail)
    return bool(
        noun_match
        and _artifact_source_capability(noun_match.group('noun') or '')
    )


def _deictic_consumer_follows_generated_artifact_source(
    prompt_text: str,
    match: re.Match[str],
) -> bool:
    clause_start = max(
        prompt_text.rfind('.', 0, match.start()),
        prompt_text.rfind('!', 0, match.start()),
        prompt_text.rfind('?', 0, match.start()),
        prompt_text.rfind('\n', 0, match.start()),
        prompt_text.rfind(';', 0, match.start()),
    ) + 1
    candidate_sources = [
        source_match
        for source_match in _same_turn_generated_artifact_source_matches(
            prompt_text,
            clause_start,
            match.start(),
        )
        if source_match.end() <= match.start()
    ]
    for source_match in reversed(candidate_sources):
        if _offset_is_inside_quote(prompt_text, source_match.start()):
            continue
        if not _artifact_source_declaration_is_user_directive(
            prompt_text,
            source_match,
        ):
            continue
        if _source_declaration_is_negated(prompt_text, source_match):
            continue
        source_leading = prompt_text[
            max(clause_start, source_match.start() - 180):source_match.start()
        ]
        if _SAME_TURN_SOURCE_META_PREFIX_RE.search(source_leading):
            continue
        coordination = prompt_text[source_match.end():match.start()]
        if re.search(
            r'(?is)(?:\b(?:and|then|next|und|sowie|dann|danach|anschließend|anschliessend)\b|[,;])'
            r'[\s,:-]*$',
            coordination,
        ):
            return True
    return False


def _deictic_match_is_actionable(
    prompt_text: str,
    match: re.Match[str],
) -> bool:
    _, reference_start, _ = _deictic_transform_reference(prompt_text, match)
    if (
        _offset_is_inside_quote(prompt_text, match.start())
        or _offset_is_inside_quote(prompt_text, reference_start)
    ):
        return False
    clause_start = max(
        prompt_text.rfind('.', 0, match.start()),
        prompt_text.rfind('!', 0, match.start()),
        prompt_text.rfind('?', 0, match.start()),
        prompt_text.rfind('\n', 0, match.start()),
        prompt_text.rfind(';', 0, match.start()),
    ) + 1
    leading = prompt_text[clause_start:match.start()]
    negation_leading = re.sub(r'[\s,;:-]+$', '', leading)
    is_user_directive = _action_offset_is_user_directive(
        prompt_text,
        match.start(),
    ) or (
        match.re is _DEICTIC_ARTIFACT_CONSUMER_RE
        and _deictic_consumer_follows_generated_artifact_source(
            prompt_text,
            match,
        )
    )
    if (
        _SAME_TURN_SOURCE_LEADING_NEGATION_RE.search(negation_leading)
        or _SAME_TURN_SOURCE_META_PREFIX_RE.search(leading[-180:])
        or not is_user_directive
    ):
        return False
    if match.re is not _DEICTIC_ARTIFACT_CONSUMER_RE:
        return True
    if _deictic_consumer_is_inline_text_source(prompt_text, match):
        return False
    return _deictic_consumer_reference_targets_artifact(prompt_text, match)


def _deictic_source_transform_matches(prompt_text: str) -> list[re.Match[str]]:
    matches = [
        *_DEICTIC_SOURCE_TRANSFORM_RE.finditer(prompt_text),
        *_DEICTIC_ARTIFACT_CONSUMER_RE.finditer(prompt_text),
    ]
    matches_by_reference: dict[int, re.Match[str]] = {}
    for match in matches:
        if not _deictic_match_is_actionable(prompt_text, match):
            continue
        _, reference_start, _ = _deictic_transform_reference(prompt_text, match)
        previous = matches_by_reference.get(reference_start)
        if previous is None or match.start() < previous.start():
            matches_by_reference[reference_start] = match
    return sorted(
        matches_by_reference.values(),
        key=lambda item: (item.start(), item.end()),
    )


def _sentence_clause_start(prompt_text: str, offset: int) -> int:
    boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(prompt_text[:max(0, offset)]))
    return boundaries[-1].end() if boundaries else 0


def _sentence_clause_end(prompt_text: str, offset: int) -> int:
    boundary = _SENTENCE_BOUNDARY_RE.search(prompt_text, max(0, offset))
    return boundary.start() if boundary else len(prompt_text)


def _deictic_transform_order_anchor(prompt_text: str, match: re.Match[str]) -> int:
    reference_match, reference_start, _ = _deictic_transform_reference(prompt_text, match)
    if not reference_match:
        return match.start()
    preceding_actions = list(
        _DEICTIC_SOURCE_TRANSFORM_ACTION_RE.finditer(
            prompt_text[match.start():reference_start]
        )
    )
    if not preceding_actions:
        return reference_start
    return match.start() + preceding_actions[-1].start()


def _same_turn_generated_text_source_matches(prompt_text: str) -> list[re.Match[str]]:
    matches = [
        *_SAME_TURN_GENERATED_TEXT_SOURCE_RE.finditer(prompt_text),
        *_SAME_TURN_MODAL_TEXT_SOURCE_RE.finditer(prompt_text),
    ]
    return sorted(matches, key=lambda item: (item.start(), item.end()))


def _source_declaration_is_user_directive(
    prompt_text: str,
    source_match: re.Match[str],
) -> bool:
    action_start = source_match.start('action')
    clause_start = max(
        prompt_text.rfind('.', 0, action_start),
        prompt_text.rfind('!', 0, action_start),
        prompt_text.rfind('?', 0, action_start),
        prompt_text.rfind('\n', 0, action_start),
        prompt_text.rfind(';', 0, action_start),
    ) + 1
    action_leading = prompt_text[clause_start:action_start]
    return bool(
        _SAME_TURN_SOURCE_DIRECT_PREFIX_RE.fullmatch(action_leading)
        or _SAME_TURN_SOURCE_YOU_DIRECTED_PREFIX_RE.fullmatch(action_leading)
        or _SAME_TURN_SOURCE_AFFIRMATIVE_NEGATION_PREFIX_RE.fullmatch(action_leading)
    )


def _source_declaration_is_negated(
    prompt_text: str,
    source_match: re.Match[str],
) -> bool:
    action_start = source_match.start('action')
    clause_start = max(
        prompt_text.rfind('.', 0, action_start),
        prompt_text.rfind('!', 0, action_start),
        prompt_text.rfind('?', 0, action_start),
        prompt_text.rfind('\n', 0, action_start),
        prompt_text.rfind(';', 0, action_start),
    ) + 1
    leading = prompt_text[clause_start:action_start]
    if _SAME_TURN_SOURCE_LEADING_NEGATION_RE.search(leading):
        return True
    before_source = prompt_text[source_match.start():source_match.start('source')]
    return bool(_SAME_TURN_SOURCE_BETWEEN_NEGATION_RE.search(before_source))


def _text_source_kind(value: str) -> str:
    source = str(value or '').strip().lower()
    for kind, pattern in (
        (
            'scene',
            r'\b(?:scene|place|location|setting|concept|idea|design|'
            r'szenentext|[\wäöüß-]*szene|ort|schauplatz|umgebung|konzept|idee|entwurf)\b',
        ),
        ('story', r'\b(?:story|stories|geschichte|erzählung|erzaehlung)\b'),
        ('sentence', r'\b(?:sentence|satz|sätze|saetze)\b'),
        ('description', r'\b(?:description|beschreibung)\b'),
        ('prompt', r'\bprompt\b'),
        ('slogan', r'\b(?:slogan|warnslogan)\b'),
        ('script', r'\b(?:script|skript)\b'),
        ('narration', r'\b(?:narration|ansage)\b'),
        ('poem', r'\b(?:poem|gedicht)\b'),
        ('warning', r'\b(?:warning|notice|announcement|warntext|warnung|hinweis)\b'),
        ('summary', r'\b(?:summary|zusammenfassung)\b'),
        ('report', r'\b(?:report|bericht)\b'),
        ('article', r'\b(?:article|artikel)\b'),
        ('email', r'\b(?:e[-\s]?mail)\b'),
        ('message', r'\b(?:message|nachricht)\b'),
        ('text', r'\b(?:text|copy)\b'),
    ):
        if re.search(pattern, source, flags=re.IGNORECASE):
            return kind
    return ''


def _transform_has_competing_explicit_source(
    prompt_text: str,
    transform_match: re.Match[str],
    generated_source_match: re.Match[str],
) -> bool:
    reference_match, _, reference_end = _deictic_transform_reference(prompt_text, transform_match)
    if not reference_match:
        return False
    reference = re.sub(r'\s+', ' ', reference_match.group(0).strip().lower())
    if reference in {'daraus', 'davon', 'hieraus'}:
        return False
    clause_end_candidates = [
        index
        for index in (
            prompt_text.find('.', reference_end),
            prompt_text.find('!', reference_end),
            prompt_text.find('?', reference_end),
            prompt_text.find('\n', reference_end),
        )
        if index >= 0
    ]
    clause_end = min(clause_end_candidates) if clause_end_candidates else len(prompt_text)
    reference_tail = prompt_text[reference_end:min(clause_end, reference_end + 120)]
    connector_match = _DEICTIC_SOURCE_CONNECTOR_RE.search(reference_tail)
    explicit_source_span = (
        reference_tail[:connector_match.start()]
        if connector_match
        else reference_tail
    )
    explicit_match = _EXPLICIT_DEICTIC_SOURCE_NOUN_RE.search(explicit_source_span)
    if not explicit_match:
        return False
    if _COMPETING_DEICTIC_SOURCE_QUALIFIER_RE.search(explicit_source_span):
        return True
    explicit_noun = explicit_match.group('noun') or ''
    if _EXTERNAL_DEICTIC_SOURCE_NOUN_RE.search(explicit_noun):
        return True
    explicit_kind = _text_source_kind(explicit_noun)
    generated_kind = _text_source_kind(generated_source_match.group('source') or '')
    if explicit_kind == 'text':
        return False
    return bool(explicit_kind and generated_kind and explicit_kind != generated_kind)


def _source_declaration_is_ordered_or_coordinated(
    prompt_text: str,
    transform_match: re.Match[str],
    source_match: re.Match[str],
    *,
    transform_anchor: int,
    reference_start: int,
) -> bool:
    if source_match.end() <= transform_anchor:
        return True
    if source_match.start() != transform_match.start() or source_match.end() >= reference_start:
        return False
    coordinated_target = prompt_text[source_match.end():reference_start]
    return bool(_SAME_TURN_SOURCE_COORDINATED_TARGET_RE.search(coordinated_target))


def _prompt_declares_generated_text_source_before_transform(
    prompt_text: str,
    transform_match: re.Match[str],
) -> bool:
    expected_capabilities = set(
        _transform_reference_source_capability_counts(
            prompt_text,
            transform_match,
        )
    )
    if expected_capabilities and expected_capabilities != {CAPABILITY_CHAT}:
        return False
    transform_anchor = _deictic_transform_order_anchor(prompt_text, transform_match)
    _, reference_start, _ = _deictic_transform_reference(prompt_text, transform_match)
    candidate_prefix = prompt_text[:reference_start]
    for source_match in _same_turn_generated_text_source_matches(candidate_prefix):
        if not _source_declaration_is_ordered_or_coordinated(
            prompt_text,
            transform_match,
            source_match,
            transform_anchor=transform_anchor,
            reference_start=reference_start,
        ):
            continue
        if _offset_is_inside_quote(prompt_text, source_match.start()):
            continue
        if not _source_declaration_is_user_directive(prompt_text, source_match):
            continue
        source_scope = source_match.group(0)
        if _source_declaration_is_negated(prompt_text, source_match):
            continue
        leading = prompt_text[max(0, source_match.start() - 160):source_match.start()]
        if (
            _SAME_TURN_SOURCE_META_PREFIX_RE.search(leading)
            or _SAME_TURN_SOURCE_META_PREFIX_RE.search(source_scope)
        ):
            continue
        if _transform_has_competing_explicit_source(
            prompt_text,
            transform_match,
            source_match,
        ):
            continue
        return True
    return False


def _inline_text_source_records(
    prompt_text: str,
) -> list[tuple[int, int, int, str]]:
    records = [
        (
            source_match.start(),
            source_match.end(),
            source_match.start('action'),
            str(source_match.group('content') or '').strip(),
        )
        for source_match in _INLINE_TEXTUAL_ARTIFACT_SOURCE_RE.finditer(prompt_text)
    ]
    records.extend(
        (
            source_match.start('label'),
            source_match.end(),
            source_match.start('label'),
            str(source_match.group('content') or '').strip(),
        )
        for source_match in _INLINE_TEXTUAL_ARTIFACT_LABEL_SOURCE_RE.finditer(
            prompt_text
        )
    )
    return sorted(records, key=lambda item: (item[0], item[1]))


def _prompt_declares_inline_text_source_before_transform(
    prompt_text: str,
    transform_match: re.Match[str],
) -> bool:
    expected_capabilities = set(
        _transform_reference_source_capability_counts(
            prompt_text,
            transform_match,
        )
    )
    if expected_capabilities and expected_capabilities != {CAPABILITY_CHAT}:
        return False
    transform_anchor = _deictic_transform_order_anchor(prompt_text, transform_match)
    _, reference_start, _ = _deictic_transform_reference(prompt_text, transform_match)
    for source_start, source_end, action_start, content in _inline_text_source_records(
        prompt_text[:reference_start]
    ):
        if source_end > transform_anchor:
            continue
        if _offset_is_inside_quote(prompt_text, source_start):
            continue
        if not _action_offset_is_user_directive(prompt_text, action_start):
            continue
        leading = prompt_text[
            max(0, source_start - 180):source_start
        ]
        if _SAME_TURN_SOURCE_META_PREFIX_RE.search(leading):
            continue
        if not content:
            continue
        return True
    return False


def _prompt_has_actionable_inline_text_source(prompt_text: str) -> bool:
    for source_start, _, action_start, content in _inline_text_source_records(
        prompt_text
    ):
        if _offset_is_inside_quote(prompt_text, source_start):
            continue
        if not _action_offset_is_user_directive(prompt_text, action_start):
            continue
        leading = prompt_text[
            max(0, source_start - 180):source_start
        ]
        if _SAME_TURN_SOURCE_META_PREFIX_RE.search(leading):
            continue
        if content:
            return True
    return False


def _artifact_source_capability(value: Any) -> str:
    source = str(value or '').strip().lower()
    if re.search(
        r'\b(?:image(?:s)?|picture(?:s)?|illustration(?:s)?|graphic(?:s)?|photo(?:s)?|'
        r'poster(?:s)?|png|jpe?g|gif|webp|svg|'
        r'bild(?:e|er|ern|es)?|grafik(?:en)?|illustration(?:en)?|foto(?:s)?)\b',
        source,
        flags=re.IGNORECASE,
    ):
        return CAPABILITY_IMAGE_GENERATION
    if re.search(
        r'\b(?:audio(?:s)?|recording(?:s)?|speech|voice|mp3|wav|m4a|flac|ogg|'
        r'aufnahme(?:n)?|sprachaufnahme(?:n)?)\b',
        source,
        flags=re.IGNORECASE,
    ):
        return CAPABILITY_TEXT_TO_SPEECH
    if re.search(
        r'\b(?:html|code|document|page|text|message|json|markdown|css|javascript|'
        r'dokument(?:e|en|s)?|seite(?:n)?|text(?:e|en|s)?|nachricht(?:en)?|quellcode)\b',
        source,
        flags=re.IGNORECASE,
    ):
        return CAPABILITY_CHAT
    return ''


def _route_phase_graph_has_artifact_consumer_edge(
    route_payload: Optional[Mapping[str, Any]],
    producer_capability: str,
    *,
    minimum_edge_count: int = 1,
    minimum_producer_count: int = 1,
) -> bool:
    if not isinstance(route_payload, Mapping):
        return False
    if (
        type(minimum_edge_count) is not int
        or minimum_edge_count < 1
        or type(minimum_producer_count) is not int
        or minimum_producer_count < 1
    ):
        return False
    route_runtime = (
        route_payload.get('route_runtime')
        if isinstance(route_payload.get('route_runtime'), Mapping)
        else {}
    )
    phase_graph = (
        route_runtime.get('request_phase_graph')
        if isinstance(route_runtime.get('request_phase_graph'), Mapping)
        else route_payload.get('request_phase_graph')
        if isinstance(route_payload.get('request_phase_graph'), Mapping)
        else {}
    )
    if not current_phase_is_graph_resolved(dict(phase_graph)):
        return False
    raw_phases = phase_graph.get('phases')
    if not isinstance(raw_phases, list):
        return False
    phases = [
        dict(item)
        for item in raw_phases
        if isinstance(item, Mapping)
    ]
    if len(phases) != len(raw_phases):
        return False
    phase_ids = [str(item.get('phase_id') or '').strip() for item in phases]
    if not phases or any(not phase_id for phase_id in phase_ids):
        return False
    if len(phase_ids) != len(set(phase_ids)):
        return False
    phase_id_set = set(phase_ids)

    def _normalized_dependency_ids(
        value: Any,
        *,
        allow_missing: bool = False,
    ) -> Optional[list[str]]:
        if value is None and allow_missing:
            return []
        if not isinstance(value, list):
            return None
        dependency_ids = [str(item or '').strip() for item in value]
        if any(not dependency_id for dependency_id in dependency_ids):
            return None
        if len(dependency_ids) != len(set(dependency_ids)):
            return None
        return dependency_ids

    phase_dependencies: dict[str, set[str]] = {}
    phase_dependency_inputs: dict[str, set[str]] = {}
    for item in phases:
        phase_id = str(item.get('phase_id') or '').strip()
        dependency_ids = _normalized_dependency_ids(item.get('depends_on'))
        input_refs = item.get('input_refs')
        if dependency_ids is None or not isinstance(input_refs, list):
            return False
        if phase_id in dependency_ids or any(
            dependency_id not in phase_id_set for dependency_id in dependency_ids
        ):
            return False
        dependency_input_ids: list[str] = []
        for input_ref in input_refs:
            if not isinstance(input_ref, Mapping):
                return False
            if str(input_ref.get('kind') or '').strip().lower() != 'phase_output':
                continue
            input_phase_id = str(input_ref.get('phase_id') or '').strip()
            if not input_phase_id or input_phase_id not in phase_id_set:
                return False
            if str(input_ref.get('role') or '').strip().lower() == 'dependency':
                dependency_input_ids.append(input_phase_id)
        if len(dependency_input_ids) != len(set(dependency_input_ids)):
            return False
        if set(dependency_input_ids) != set(dependency_ids):
            return False
        phase_dependencies[phase_id] = set(dependency_ids)
        phase_dependency_inputs[phase_id] = set(dependency_input_ids)

    unresolved_dependency_count = {
        phase_id: len(dependencies)
        for phase_id, dependencies in phase_dependencies.items()
    }
    dependent_phase_ids: dict[str, set[str]] = {
        phase_id: set()
        for phase_id in phase_ids
    }
    for phase_id, dependencies in phase_dependencies.items():
        for dependency_id in dependencies:
            dependent_phase_ids[dependency_id].add(phase_id)
    ready_phase_ids = [
        phase_id
        for phase_id, dependency_count in unresolved_dependency_count.items()
        if dependency_count == 0
    ]
    visited_count = 0
    while ready_phase_ids:
        phase_id = ready_phase_ids.pop()
        visited_count += 1
        for dependent_phase_id in dependent_phase_ids.get(phase_id, set()):
            unresolved_dependency_count[dependent_phase_id] -= 1
            if unresolved_dependency_count[dependent_phase_id] == 0:
                ready_phase_ids.append(dependent_phase_id)
    if visited_count != len(phase_ids):
        return False
    candidate_graph = (
        phase_graph.get('candidate_graph')
        if isinstance(phase_graph.get('candidate_graph'), Mapping)
        else {}
    )
    raw_candidates = candidate_graph.get('candidates')
    if not isinstance(raw_candidates, list):
        return False
    if any(not isinstance(item, Mapping) for item in raw_candidates):
        return False
    output_candidates = [
        dict(item)
        for item in raw_candidates
        if isinstance(item, Mapping)
        and str(item.get('candidate_type') or '').strip().lower() == 'output'
    ]
    allowed_phase_statuses = {
        'completed',
        'fulfilled',
        'in_progress',
        'pending',
        'planned',
        'queued',
        'running',
    }
    allowed_phase_resolutions = {
        'completed',
        'executable',
        'fulfilled',
        'pending_dependency',
        'promoted',
        'ready',
        'resolved',
        'running',
    }
    allowed_contract_states = {
        'active',
        'executable',
        'promoted',
        'required',
    }

    def _expected_output_type(capability: Any) -> str:
        normalized = normalize_capability(capability)
        if normalized == CAPABILITY_IMAGE_GENERATION:
            return 'image'
        if normalized == CAPABILITY_TEXT_TO_SPEECH:
            return 'audio'
        if normalized in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_SPEECH_TO_TEXT}:
            return 'text'
        return ''

    def _promoted_output_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
        phase_id = str(item.get('phase_id') or '').strip()
        obligation_id = str(item.get('obligation_id') or '').strip()
        capability = normalize_capability(item.get('capability'))
        expected_output_type = _expected_output_type(capability)
        if not phase_id or not obligation_id or not capability or not expected_output_type:
            return {}
        matches = [
            candidate
            for candidate in output_candidates
            if str(candidate.get('phase_id') or '').strip() == phase_id
            and normalize_capability(candidate.get('capability')) == capability
            and str(candidate.get('kind') or '').strip().lower() == 'ollmo.candidate'
            and str(candidate.get('status') or '').strip().lower() == 'promoted'
            and str(candidate.get('execution_policy') or '').strip().lower()
            == 'executable_obligation'
            and str(candidate.get('promotion_policy') or '').strip().lower()
            == 'runtime_required'
            and str(candidate.get('output_type') or '').strip().lower()
            == expected_output_type
            and (
                not str(candidate.get('contract_state') or '').strip()
                or str(candidate.get('contract_state') or '').strip().lower()
                in allowed_contract_states
            )
            and str(candidate.get('contract_ref') or '').strip() == obligation_id
            and str(candidate.get('obligation_id') or '').strip() == obligation_id
            and set(
                _normalized_dependency_ids(
                    candidate.get('depends_on'),
                    allow_missing=True,
                )
                or []
            )
            == phase_dependencies.get(phase_id, set())
            and _normalized_dependency_ids(
                candidate.get('depends_on'),
                allow_missing=True,
            )
            is not None
        ]
        return matches[0] if len(matches) == 1 else {}

    def _phase_is_promoted_nonreserved(item: Mapping[str, Any]) -> bool:
        output_contract = (
            item.get('output_contract')
            if isinstance(item.get('output_contract'), Mapping)
            else {}
        )
        capability = normalize_capability(item.get('capability'))
        expected_output_type = _expected_output_type(capability)
        phase_status = str(item.get('status') or '').strip().lower()
        phase_resolution = str(item.get('resolution') or '').strip().lower()
        phase_contract_state = str(item.get('contract_state') or '').strip().lower()
        phase_execution_policy = str(item.get('execution_policy') or '').strip().lower()
        allowed_phase_kinds = (
            {'evidence', 'materialize'}
            if capability == CAPABILITY_VISION_ANALYSIS
            else {'materialize'}
        )
        if (
            str(item.get('kind') or '').strip().lower() not in allowed_phase_kinds
            or not expected_output_type
            or str(item.get('output_type') or '').strip().lower() != expected_output_type
            or phase_status not in allowed_phase_statuses
            or phase_resolution not in allowed_phase_resolutions
            or (phase_contract_state and phase_contract_state not in allowed_contract_states)
            or (
                phase_execution_policy
                and phase_execution_policy != 'executable_obligation'
            )
            or output_contract.get('required') is not True
            or str(output_contract.get('output_type') or '').strip().lower()
            != expected_output_type
            or str(output_contract.get('status') or '').strip().lower()
            not in allowed_phase_statuses
        ):
            return False
        return bool(_promoted_output_candidate(item))

    producer_ids = {
        str(item.get('phase_id') or '').strip()
        for item in phases
        if normalize_capability(item.get('capability')) == producer_capability
        and _phase_is_promoted_nonreserved(item)
    }
    if not producer_ids:
        return False
    consumer_capability = (
        CAPABILITY_VISION_ANALYSIS
        if producer_capability == CAPABILITY_IMAGE_GENERATION
        else CAPABILITY_SPEECH_TO_TEXT
        if producer_capability == CAPABILITY_TEXT_TO_SPEECH
        else ''
    )
    if not consumer_capability:
        return False
    valid_consumer_edges: set[tuple[str, str]] = set()
    for item in phases:
        if normalize_capability(item.get('capability')) != consumer_capability:
            continue
        if not _phase_is_promoted_nonreserved(item):
            continue
        consumer_phase_id = str(item.get('phase_id') or '').strip()
        dependencies = phase_dependencies.get(consumer_phase_id, set())
        direct_producers = producer_ids.intersection(dependencies)
        if len(direct_producers) != 1:
            continue
        direct_producer_id = next(iter(direct_producers))
        input_phase_ids = phase_dependency_inputs.get(consumer_phase_id, set())
        consumer_candidate = _promoted_output_candidate(item)
        candidate_dependencies = set(
            _normalized_dependency_ids(
                consumer_candidate.get('depends_on'),
                allow_missing=True,
            )
            or []
        )
        if (
            direct_producer_id in input_phase_ids
            and direct_producer_id in candidate_dependencies
        ):
            valid_consumer_edges.add((direct_producer_id, consumer_phase_id))
    valid_producer_ids = {
        producer_id
        for producer_id, _ in valid_consumer_edges
    }
    return bool(
        len(valid_consumer_edges) >= minimum_edge_count
        and len(valid_producer_ids)
        >= minimum_producer_count
    )


def _artifact_source_declaration_is_user_directive(
    prompt_text: str,
    source_match: re.Match[str],
) -> bool:
    if _action_offset_is_user_directive(
        prompt_text,
        source_match.start('action'),
    ):
        return True
    action_start = source_match.start('action')
    clause_start = max(
        _sentence_clause_start(prompt_text, action_start),
        prompt_text.rfind(';', 0, action_start) + 1,
    )
    return bool(
        _SAME_TURN_ARTIFACT_CONTEXT_PREFIX_RE.fullmatch(
            prompt_text[clause_start:action_start]
        )
    )


def _prompt_has_matching_artifact_consumer_cue(
    prompt_text: str,
    source_match: re.Match[str],
    transform_match: re.Match[str],
    producer_capability: str,
) -> bool:
    _, reference_start, reference_end = _deictic_transform_reference(
        prompt_text,
        transform_match,
    )
    if source_match.end() > reference_start:
        return False
    consumer_action_re = None
    if producer_capability == CAPABILITY_IMAGE_GENERATION:
        consumer_action_re = _SAME_TURN_IMAGE_CONSUMER_ACTION_RE
    elif producer_capability == CAPABILITY_TEXT_TO_SPEECH:
        consumer_action_re = _SAME_TURN_AUDIO_CONSUMER_ACTION_RE
    if consumer_action_re is None:
        return False
    consumer_segment_end = _sentence_clause_end(prompt_text, reference_end)
    consumer_scope = prompt_text[source_match.end():consumer_segment_end]
    consumer_actions = list(consumer_action_re.finditer(consumer_scope))
    if not consumer_actions:
        return False
    absolute_consumer_actions = [
        (
            source_match.end() + action_match.start(),
            source_match.end() + action_match.end(),
        )
        for action_match in consumer_actions
    ]
    preceding_actions = [
        action_span
        for action_span in absolute_consumer_actions
        if action_span[0] < reference_start
    ]
    following_actions = [
        action_span
        for action_span in absolute_consumer_actions
        if action_span[0] >= reference_end
    ]
    selected_action = (
        preceding_actions[-1]
        if preceding_actions
        else following_actions[0]
        if following_actions
        else None
    )
    if selected_action is None:
        return False
    consumer_action_start, consumer_action_end = selected_action
    if consumer_action_start < reference_start:
        later_transform_action = _DEICTIC_SOURCE_TRANSFORM_ACTION_RE.search(
            prompt_text,
            consumer_action_end,
            reference_start,
        )
        if later_transform_action is not None:
            return False
    opposite_consumer_action_re = (
        _SAME_TURN_AUDIO_CONSUMER_ACTION_RE
        if producer_capability == CAPABILITY_IMAGE_GENERATION
        else _SAME_TURN_IMAGE_CONSUMER_ACTION_RE
    )
    opposite_consumer_action = opposite_consumer_action_re.search(
        prompt_text,
        reference_end,
        consumer_segment_end,
    )
    if opposite_consumer_action is not None:
        consumer_segment_end = opposite_consumer_action.start()
    next_source = next(
        (
            match
            for match in _same_turn_generated_artifact_source_matches(
                prompt_text,
                reference_end,
                consumer_segment_end,
            )
            if match.start() > reference_start
            and not _offset_is_inside_quote(prompt_text, match.start())
        ),
        None,
    )
    if next_source is not None:
        consumer_segment_end = next_source.start()
    references = [
        match
        for match in _DEICTIC_SOURCE_REFERENCE_RE.finditer(
            prompt_text,
            min(reference_start, consumer_action_end),
            consumer_segment_end,
        )
        if not _offset_is_inside_quote(prompt_text, match.start())
    ]
    if not any(match.start() == reference_start for match in references):
        return False
    for index, reference_match in enumerate(references):
        next_reference_start = (
            references[index + 1].start()
            if index + 1 < len(references)
            else consumer_segment_end
        )
        reference_span = prompt_text[
            reference_match.end():min(
                next_reference_start,
                reference_match.end() + 120,
            )
        ]
        if _artifact_reference_is_competing_source(reference_span):
            return False
        explicit_capability = _artifact_source_capability(reference_span)
        if explicit_capability and explicit_capability != producer_capability:
            return False
    return True


def _artifact_reference_is_competing_source(
    explicit_reference_span: str,
) -> bool:
    reference_span = str(explicit_reference_span or '')
    if _EXTERNAL_ARTIFACT_SOURCE_LOCATOR_RE.search(reference_span):
        return True
    noun_match = _ARTIFACT_SOURCE_NOUN_RE.search(reference_span)
    qualifier_scope = (
        reference_span[:noun_match.end()]
        if noun_match
        else reference_span[:60]
    )
    if _REMOTE_ARTIFACT_SOURCE_QUALIFIER_RE.search(qualifier_scope):
        return True
    generated_reference = bool(_GENERATED_ARTIFACT_REFERENCE_RE.search(qualifier_scope))
    for qualifier_match in _COMPETING_DEICTIC_SOURCE_QUALIFIER_RE.finditer(qualifier_scope):
        qualifier = qualifier_match.group(0).strip().lower()
        if qualifier in {'local', 'lokal'} and generated_reference:
            continue
        return True
    return False


def _artifact_source_declaration_scope(
    prompt_text: str,
    source_match: re.Match[str],
    *,
    reference_start: int,
) -> str:
    declaration_end = min(
        _sentence_clause_end(prompt_text, source_match.end()),
        reference_start,
    )
    consumer_actions = [
        action_match
        for pattern in (
            _SAME_TURN_IMAGE_CONSUMER_ACTION_RE,
            _SAME_TURN_AUDIO_CONSUMER_ACTION_RE,
        )
        for action_match in [
            pattern.search(prompt_text, source_match.end(), declaration_end)
        ]
        if action_match is not None
    ]
    if consumer_actions:
        declaration_end = min(match.start() for match in consumer_actions)
    return prompt_text[source_match.start():declaration_end]


def _declared_artifact_source_count(
    prompt_text: str,
    source_match: re.Match[str],
    producer_capability: str,
    *,
    reference_start: int,
) -> int:
    declaration_scope = _artifact_source_declaration_scope(
        prompt_text,
        source_match,
        reference_start=reference_start,
    )
    count_words = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'ein': 1,
        'eine': 1,
        'zwei': 2,
        'drei': 3,
        'vier': 4,
        'fünf': 5,
        'fuenf': 5,
    }
    total = 0
    previous_noun_end: Optional[int] = None
    previous_count = 0
    for noun_match in _ARTIFACT_SOURCE_NOUN_RE.finditer(declaration_scope):
        noun = noun_match.group(0).strip().lower()
        if _artifact_source_capability(noun) != producer_capability:
            continue
        count_prefix = declaration_scope[
            max(0, noun_match.start() - 24):noun_match.start()
        ]
        explicit_count = re.search(
            r'(?i)\b(?P<count>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'ein|eine|zwei|drei|vier|fünf|fuenf)\s*$',
            count_prefix,
        )
        noun_count = 1
        if explicit_count:
            raw_count = explicit_count.group('count').lower()
            noun_count = (
                int(raw_count)
                if raw_count.isdigit()
                else count_words.get(raw_count, 1)
            )
        elif re.search(
            r'(?:images|pictures|audios|recordings|bilder|aufnahmen)$',
            noun,
            flags=re.IGNORECASE,
        ):
            noun_count = 2
        noun_count = max(1, noun_count)
        if previous_noun_end is not None and not declaration_scope[
            previous_noun_end:noun_match.start()
        ].strip():
            total = total - previous_count + max(previous_count, noun_count)
            previous_count = max(previous_count, noun_count)
        else:
            total += noun_count
            previous_count = noun_count
        previous_noun_end = noun_match.end()
    return max(1, total)


def _graph_grounded_artifact_sources_for_transform(
    prompt_text: str,
    transform_match: re.Match[str],
) -> list[tuple[re.Match[str], str]]:
    _, reference_start, reference_end = _deictic_transform_reference(
        prompt_text,
        transform_match,
    )
    clause_end = _sentence_clause_end(prompt_text, reference_end)
    explicit_reference_span = prompt_text[reference_end:min(clause_end, reference_end + 120)]
    eligible_sources: list[tuple[re.Match[str], str]] = []
    for source_match in _same_turn_generated_artifact_source_matches(prompt_text):
        if source_match.end() > reference_start:
            continue
        if _offset_is_inside_quote(prompt_text, source_match.start()):
            continue
        if not _artifact_source_declaration_is_user_directive(prompt_text, source_match):
            continue
        if _source_declaration_is_negated(prompt_text, source_match):
            continue
        leading = prompt_text[max(0, source_match.start() - 160):source_match.start()]
        if (
            _SAME_TURN_SOURCE_META_PREFIX_RE.search(leading)
            or _SAME_TURN_SOURCE_META_PREFIX_RE.search(source_match.group(0))
        ):
            continue
        source_declaration_scope = _artifact_source_declaration_scope(
            prompt_text,
            source_match,
            reference_start=reference_start,
        )
        producer_capabilities = {
            capability
            for noun_match in _ARTIFACT_SOURCE_NOUN_RE.finditer(
                source_declaration_scope
            )
            for capability in [_artifact_source_capability(noun_match.group(0))]
            if capability
        }
        if source_match.re is _SAME_TURN_IMPLICIT_IMAGE_SOURCE_RE:
            producer_capabilities.add(CAPABILITY_IMAGE_GENERATION)
        for producer_capability in producer_capabilities:
            if not _prompt_has_matching_artifact_consumer_cue(
                prompt_text,
                source_match,
                transform_match,
                producer_capability,
            ):
                continue
            if _artifact_reference_is_competing_source(explicit_reference_span):
                continue
            explicit_capability = _artifact_source_capability(explicit_reference_span)
            if explicit_capability and explicit_capability != producer_capability:
                continue
            eligible_sources.append((source_match, producer_capability))
    return eligible_sources


def _graph_grounded_artifact_pair_count(
    prompt_text: str,
    producer_capability: str,
) -> int:
    source_required_counts: dict[int, int] = {}
    for transform_match in _deictic_source_transform_matches(prompt_text):
        _, reference_start, _ = _deictic_transform_reference(
            prompt_text,
            transform_match,
        )
        sources = _graph_grounded_artifact_sources_for_transform(
            prompt_text,
            transform_match,
        )
        matching_sources = [
            source_match
            for source_match, capability in sources
            if capability == producer_capability
        ]
        if not matching_sources:
            continue
        nearest_source = max(matching_sources, key=lambda item: item.end())
        reference_counts = _transform_reference_source_capability_counts(
            prompt_text,
            transform_match,
        )
        source_key = nearest_source.start()
        source_required_counts[source_key] = max(
            source_required_counts.get(source_key, 0),
            reference_counts.get(producer_capability, 1),
            _declared_artifact_source_count(
                prompt_text,
                nearest_source,
                producer_capability,
                reference_start=reference_start,
            ),
        )
    return sum(source_required_counts.values())


def _graph_grounded_artifact_producer_count(
    prompt_text: str,
    producer_capability: str,
) -> int:
    source_required_counts: dict[int, int] = {}
    for transform_match in _deictic_source_transform_matches(prompt_text):
        _, reference_start, _ = _deictic_transform_reference(
            prompt_text,
            transform_match,
        )
        matching_sources = [
            source_match
            for source_match, capability in _graph_grounded_artifact_sources_for_transform(
                prompt_text,
                transform_match,
            )
            if capability == producer_capability
        ]
        if not matching_sources:
            continue
        nearest_source = max(matching_sources, key=lambda item: item.end())
        source_required_counts[nearest_source.start()] = max(
            source_required_counts.get(nearest_source.start(), 0),
            _declared_artifact_source_count(
                prompt_text,
                nearest_source,
                producer_capability,
                reference_start=reference_start,
            ),
        )
    return sum(source_required_counts.values())


def _prompt_declares_graph_grounded_artifact_source_before_transform(
    prompt_text: str,
    transform_match: re.Match[str],
    *,
    route_payload: Optional[Mapping[str, Any]],
) -> bool:
    eligible_sources = _graph_grounded_artifact_sources_for_transform(
        prompt_text,
        transform_match,
    )
    for _, producer_capability in eligible_sources:
        required_edge_count = _graph_grounded_artifact_pair_count(
            prompt_text,
            producer_capability,
        )
        required_producer_count = _graph_grounded_artifact_producer_count(
            prompt_text,
            producer_capability,
        )
        if _route_phase_graph_has_artifact_consumer_edge(
            route_payload,
            producer_capability,
            minimum_edge_count=required_edge_count,
            minimum_producer_count=required_producer_count,
        ):
            return True
    return False


def _request_requires_current_source_for_transform(
    prompt: Any,
    request_payload: Optional[Mapping[str, Any]] = None,
    route_payload: Optional[Mapping[str, Any]] = None,
    response_payload: Optional[Mapping[str, Any]] = None,
) -> bool:
    prompt_text = str(prompt or '').strip()
    if not prompt_text:
        return False
    transform_matches = _deictic_source_transform_matches(prompt_text)
    if not transform_matches:
        return False
    if not _SOURCE_TRANSFORM_ARTIFACT_TARGET_RE.search(prompt_text):
        return False
    return any(
        not _payload_has_compatible_direct_artifact_source(
            prompt_text,
            match,
            response_payload,
            request_payload,
            route_payload,
        )
        and not _payload_has_grounded_predecessor_image_specification(
            prompt_text,
            match,
            response_payload,
            request_payload,
            route_payload,
        )
        and not _prompt_declares_generated_text_source_before_transform(prompt_text, match)
        and not _prompt_declares_inline_text_source_before_transform(
            prompt_text,
            match,
        )
        and not _prompt_declares_graph_grounded_artifact_source_before_transform(
            prompt_text,
            match,
            route_payload=route_payload,
        )
        for match in transform_matches
    )


def _strip_handoff_markdown_wrappers(value: Any) -> str:
    text = str(value or '').strip()
    while text:
        updated = text
        for prefix, suffix in (('**', '**'), ('__', '__'), ('*', '*'), ('_', '_'), ('`', '`')):
            if updated.startswith(prefix) and updated.endswith(suffix) and len(updated) > len(prefix) + len(suffix):
                updated = updated[len(prefix):-len(suffix)].strip()
        updated = updated.strip('[]').strip()
        if updated == text:
            break
        text = updated
    return text


def _clean_social_manifest_piece(value: Any) -> str:
    text = _strip_handoff_markdown_wrappers(value).strip()
    text = _HEX_BYTE_FRAGMENT_RE.sub(' ', text)
    text = re.sub(r'^\s*(?:[-*#>\u2022]+\s*)+', '', text).strip()
    text = re.sub(
        r'^(?:id|asset\s*id|user|username|handle|influencer|caption|location|scenario|style|visual\s*style|visual\s+prompt|image\s+prompt|prompt)\s*:\s*',
        '',
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.split(
        r'\s+(?:\*\*+|__+|\*+)?(?:category|section|part|phase)\s*:',
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    text = re.split(r'\s+#{1,6}\s+', text, maxsplit=1)[0].strip()
    text = text.strip('`*_ ')
    text = text.strip().strip('"“”')
    text = re.sub(r'\s+', ' ', text).strip(' ,;')
    return text


def _split_social_manifest_labeled_piece(value: Any) -> tuple[str, str]:
    text = str(value or '').strip()
    if not text:
        return '', ''
    text = re.sub(r'^\s*(?:[-*#>\u2022]+\s*)+', '', text).strip()
    match = re.match(
        r'(?is)^(?:\*\*+|__+|`+|\*+)?\s*'
        r'(?P<label>[a-z][\w\s/-]{0,48}?)'
        r'\s*(?:\*\*+|__+|`+|\*+)?\s*:\s*'
        r'(?P<body>.*)$',
        text,
    )
    if not match:
        return '', _clean_social_manifest_piece(text)
    label = re.sub(r'\s+', ' ', str(match.group('label') or '')).strip().lower()
    body = _clean_social_manifest_piece(match.group('body'))
    return label, body


def _social_manifest_handle_to_subject(value: Any) -> str:
    text = str(value or '').strip().lstrip('@')
    text = re.sub(r'\.(?:png|jpe?g|webp|gif|avif)$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[_\-.]+', ' ', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(' ,;')


def _social_manifest_subject_to_prompt_piece(value: Any) -> str:
    subject = _social_manifest_handle_to_subject(value)
    if not subject:
        return ''
    if subject.lower() == subject:
        subject = subject.title()
    return subject


def _social_manifest_piece_looks_like_caption_copy(value: Any) -> bool:
    raw_text = str(value or '').strip()
    text = _clean_social_manifest_piece(raw_text)
    if not text:
        return False
    if _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(text):
        return False
    if _HEX_BYTE_FRAGMENT_RE.search(raw_text):
        return True
    if re.search(r'[!?]|\.{2,}', raw_text):
        return True
    return bool(_SOCIAL_MANIFEST_CAPTION_COPY_RE.search(text))


def _strip_unlabeled_social_manifest_image_prompt_metadata(raw_value: Any) -> str:
    text = str(raw_value or '').strip()
    if not text or ',' not in text or _image_prompt_candidate_is_code_or_css_polluted(text):
        return ''
    parts = [_clean_social_manifest_piece(part) for part in text.split(',')]
    parts = [part for part in parts if part]
    if len(parts) < 3:
        return ''
    # A grammatical image-prompt opener is prose, not an unlabeled social
    # handle.  Misclassifying it would normalize semantic punctuation in the
    # subject (for example, ``hyper-detailed`` became ``hyper detailed``).
    if re.match(r'(?i)^(?:a|an|the)\s+', parts[0]):
        return ''
    if _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(parts[0]):
        return ''
    visual_index = -1
    for index, part in enumerate(parts[1:], start=1):
        if _image_prompt_candidate_is_code_or_css_polluted(part):
            continue
        if _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(part):
            visual_index = index
            break
    if visual_index <= 0:
        return ''
    subject = _social_manifest_subject_to_prompt_piece(parts[0])
    context_parts = [
        part
        for part in parts[1:visual_index]
        if part
        and not _social_manifest_piece_looks_like_caption_copy(part)
        and not _image_prompt_candidate_is_code_or_css_polluted(part)
    ]
    visual_parts = [
        part
        for part in parts[visual_index:]
        if part and not _image_prompt_candidate_is_code_or_css_polluted(part)
    ]
    if not visual_parts:
        return ''
    cleaned = ', '.join(part for part in [subject, *context_parts, *visual_parts] if part)
    cleaned = re.sub(r'[`|@]+', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip(' ,;')


def _clean_social_manifest_prefix_prompt_body(raw_value: Any) -> str:
    body = _clean_social_manifest_piece(raw_value)
    body = re.sub(r'(?i)(?:\*\*+|__+|`+|\*+)?\s*style\s*:\s*', '', body)
    body = re.sub(r'(?:\*\*+|__+|`+)', '', body)
    return re.sub(r'\s+', ' ', body).strip(' ,;')


def _social_manifest_prefix_body_looks_like_image_prompt(raw_value: Any) -> bool:
    body = _clean_social_manifest_prefix_prompt_body(raw_value)
    if not body or _image_prompt_candidate_is_code_or_css_polluted(body):
        return False
    if (
        _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(body)
        or _NUMBERED_PREPARED_IMAGE_SIGNAL_RE.search(body)
    ):
        return True
    words = re.findall(r'\w+', body)
    if len(words) < 6:
        return False
    if re.search(r'(?i)(?:^|[;,.]\s*)(?:a|an|the)\s+[a-z][\w-]+', body):
        return True
    if re.search(r'(?i)\bstyle\s*:', str(raw_value or '')):
        return True
    return False


def _strip_social_manifest_prefix_label(raw_value: Any) -> str:
    text = str(raw_value or '').strip()
    if not text or ':' not in text or '|' in text or _image_prompt_candidate_is_code_or_css_polluted(text):
        return ''
    match = re.match(
        r'(?is)^\s*(?:[-*#>\u2022]+\s*)?'
        r'(?P<label>@?[A-Za-z0-9][\w ._-]{1,100}?)'
        r'(?P<wrapper>\s*(?:\*\*+|__+|\*+|`+)*)\s*:\s*'
        r'(?P<body>.+)$',
        text,
    )
    if not match:
        return ''
    label_raw = str(match.group('label') or '').strip()
    wrapper = str(match.group('wrapper') or '')
    body = _clean_social_manifest_prefix_prompt_body(match.group('body'))
    if not label_raw or not body or _image_prompt_candidate_is_code_or_css_polluted(body):
        return ''
    if not _social_manifest_prefix_body_looks_like_image_prompt(match.group('body')):
        return ''
    label = _clean_social_manifest_piece(label_raw)
    if not label:
        return ''
    if re.match(
        r'(?i)^(?:id|asset\s*id|prompt|image|bild|visual|scene|style|caption|user|username|handle)\b',
        label,
    ):
        return ''
    label_has_visual_signal = bool(
        _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(label)
        or _NUMBERED_PREPARED_IMAGE_SIGNAL_RE.search(label)
    )
    label_is_metadata = bool(
        label_raw.lstrip().startswith('@')
        or wrapper.strip()
        or re.search(r'[_@]', label_raw)
        or len(re.findall(r'\w+', label)) <= 4
    )
    if not label_is_metadata:
        return ''
    if label_has_visual_signal and not label_raw.lstrip().startswith('@') and not wrapper.strip():
        return ''
    return re.sub(r'\s+', ' ', body).strip(' ,;')


def _strip_social_manifest_image_prompt_metadata(raw_value: Any) -> str:
    text = str(raw_value or '').strip()
    if not text:
        return ''
    prefix_label_body = _strip_social_manifest_prefix_label(text)
    if prefix_label_body:
        return prefix_label_body
    if '|' not in text:
        return _strip_unlabeled_social_manifest_image_prompt_metadata(text)
    labeled_parts = [_split_social_manifest_labeled_piece(part) for part in text.split('|')]
    prompt_labels = {'prompt', 'visual prompt', 'image prompt', 'bild prompt', 'bild-prompt'}
    for label, body in labeled_parts:
        if label in prompt_labels and body and not _image_prompt_candidate_is_code_or_css_polluted(body):
            return re.sub(r'\s+', ' ', body).strip(' ,;')
    parts = [body for _label, body in labeled_parts]
    parts = [part for part in parts if part]
    if len(parts) < 3:
        return ''
    handle_part = ''
    for label, body in labeled_parts:
        if label in {'user', 'username', 'handle', 'influencer'} and body:
            handle_part = body
            break
    if not handle_part:
        handle_part = parts[0]
    handle_match = _SOCIAL_MANIFEST_HANDLE_RE.match(handle_part)
    if not handle_match:
        return ''
    visual_index = -1
    for index in range(len(parts) - 1, 0, -1):
        part = parts[index]
        if _image_prompt_candidate_is_code_or_css_polluted(part):
            continue
        if _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(part):
            visual_index = index
            break
    if visual_index <= 0:
        return ''
    subject = _social_manifest_subject_to_prompt_piece(handle_match.group('handle'))
    context_labels = {'location', 'scenario', 'setting', 'context', 'mood', 'style', 'visual style'}
    context_parts: list[str] = []
    for index, part in enumerate(parts[1:visual_index], start=1):
        if not part or _image_prompt_candidate_is_code_or_css_polluted(part):
            continue
        if _social_manifest_piece_looks_like_caption_copy(part):
            continue
        label = labeled_parts[index][0]
        if label in context_labels:
            context_parts.append(part)
    visual = parts[visual_index]
    cleaned = ', '.join(part for part in [subject, *context_parts, visual] if part)
    cleaned = re.sub(r'[`|@]+', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip(' ,;')


def _normalize_handoff_heading_line(raw_line: Any) -> str:
    text = _strip_handoff_markdown_wrappers(raw_line)
    text = re.sub(r'^\s*#{1,6}\s*', '', text).strip()
    text = re.sub(r'^\s*[-*>\u2022]+\s*', '', text).strip()
    return text.rstrip(':').strip()


def _is_image_prompt_section_heading(raw_line: Any) -> bool:
    heading = _normalize_handoff_heading_line(raw_line).lower()
    if not heading:
        return False
    if any(token in heading for token in ('index.html', 'styles.css', 'html', 'css')):
        return False
    match_heading = re.sub(r'\s*(?:\([^)]*\)|\[[^\]]*\])\s*$', '', heading).strip()
    if re.search(r'\b(?:image|bild|visual|asset|art)[\w\s/_-]{0,80}manifests?\b', match_heading):
        return True
    return bool(
        re.fullmatch(
            r'(?:image|bild|visual|asset|art)[\w\s/-]{0,80}(?:prompt|prompts|requirements|assets|briefs?|manifests?)',
            match_heading,
        )
        or re.fullmatch(
            r'(?:prompt|prompts|requirements|assets|briefs?|manifests?)[\w\s/-]{0,80}(?:image|bild|visual|art)',
            match_heading,
        )
    )


def _is_image_prompt_section_stop(raw_line: Any) -> bool:
    stripped = str(raw_line or '').strip()
    if not stripped:
        return False
    if _MARKDOWN_SEPARATOR_LINE_RE.fullmatch(stripped):
        return True
    lowered = _normalize_handoff_heading_line(stripped).lower()
    if stripped.startswith('```'):
        return True
    if any(token in lowered for token in ('index.html', 'styles.css')):
        return True
    if re.fullmatch(r'(?:html|css|javascript|js|index|styles?|page files?|text artifacts?)', lowered):
        return True
    if re.match(r'(?:phase|internal phase|closure|materialization|artifact|runtime)\s+contract\b', lowered):
        return True
    if lowered in {'summary', 'notes', 'implementation notes'} or lowered.startswith(
        ('summary:', 'notes:', 'implementation notes:')
    ):
        return True
    html_probe = re.sub(r'^\s*(?:\d{1,3}|[ivx]+)[.)]\s*', '', stripped, flags=re.IGNORECASE).strip()
    html_probe = _IMAGE_ASSET_LABEL_PREFIX_RE.sub('', html_probe).strip()
    if re.search(r'(?i)</?[a-z][^>]*>|(?:href|src)\s*=|url\(', html_probe):
        return True
    return False


def _image_prompt_candidate_is_code_or_css_polluted(raw_value: Any) -> bool:
    text = str(raw_value or '').strip()
    if not text:
        return False
    lowered = text.lower()
    if re.search(r'(?m)^\s*\|[^|\n]+\|', text):
        return True
    if any(token in lowered for token in ('index.html', 'styles.css', '```', '<html', '</', 'url(')):
        return True
    if re.search(r'(?is)</?[a-z][^>]*>|(?:href|src)\s*=', text):
        return True
    if re.match(r'(?is)^\s*@(?:media|keyframes|supports|container|font-face)\b', text):
        return True
    if re.search(
        r'(?is)\{[^}]*\b(?:margin|padding|box-sizing|font-family|'
        r'background(?:-color|-image)?|color|display|position|width|height|'
        r'line-height|overflow-x|transform|box-shadow|opacity|animation|'
        r'transition|font-size|border(?:-radius|-color|-width|-style)?|'
        r'object-fit|grid(?:-template(?:-columns|-rows)?|-area)?|'
        r'justify-content|align-items|gap|top|right|bottom|left|z-index|'
        r'--[\w-]+)\s*:',
        text,
    ):
        return True
    if re.match(
        r'(?is)^\s*(?:css\s+)?(?::root|body|html|[*#.]|'
        r'(?:[.#][\w-]+|[a-z][\w-]*)(?:[.#:\s>\[\]=\'"(),+-][^{;\n]*)?)\s*\{',
        text,
    ):
        return True
    if (
        re.match(r'(?is)^\s*(?:[.#][\w-]+|[a-z][\w-]*|[*])[^{}\n]{0,180}\{', text)
        and re.search(r'(?is)\{[^}]*\b[a-z-]+\s*:', text)
    ):
        return True
    if re.match(
        r'(?is)^\s*(?:css\s+)?(?:margin|padding|box-sizing|font-family|'
        r'background(?:-color|-image)?|color|display|position|width|height|'
        r'line-height|overflow-x|transform|box-shadow|opacity|animation|'
        r'transition|font-size|border(?:-radius|-color|-width|-style)?|'
        r'object-fit|grid(?:-template(?:-columns|-rows)?|-area)?|'
        r'justify-content|align-items|gap|top|right|bottom|left|z-index|'
        r'--[\w-]+)\s*:',
        text,
    ):
        return True
    return False


def _filename_social_asset_subject_piece(filename: Any) -> str:
    stem = Path(str(filename or '')).stem
    tokens = [
        token
        for token in re.split(r'[\s_.-]+', stem)
        if token
    ]
    filtered = [
        token
        for token in tokens
        if not re.fullmatch(r'\d+', token)
        and token.lower() not in _SOCIAL_ASSET_FILENAME_STOPWORDS
    ]
    if not filtered:
        filtered = [token for token in tokens if token and not re.fullmatch(r'\d+', token)]
    return _social_manifest_subject_to_prompt_piece(' '.join(filtered))


def _social_manifest_piece_is_quoted(raw_value: Any) -> bool:
    text = str(raw_value or '').strip()
    if not text:
        return False
    _label, body = _split_social_manifest_labeled_piece(text)
    if body and body != _clean_social_manifest_piece(text):
        text = body
    text = _strip_handoff_markdown_wrappers(text).strip()
    return (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith('“') and text.endswith('”'))
        or (text.startswith("'") and text.endswith("'"))
    )


def _filename_social_asset_prompt_from_row(filename: Any, body: Any) -> str:
    subject = _filename_social_asset_subject_piece(filename)
    if not subject:
        return ''
    raw_parts = [part for part in str(body or '').split('|') if str(part or '').strip()]
    if len(raw_parts) < 2:
        return ''
    labeled_parts = [_split_social_manifest_labeled_piece(part) for part in raw_parts]
    has_social_handle = any(
        (
            label in {'user', 'username', 'handle', 'influencer'}
            and value
        )
        or _SOCIAL_MANIFEST_HANDLE_RE.match(str(raw_parts[index] or '').strip())
        or str(raw_parts[index] or '').strip().lstrip('`*_ ').startswith('@')
        for index, (label, value) in enumerate(labeled_parts)
    )
    if not has_social_handle:
        return ''
    prompt_labels = {'prompt', 'visual prompt', 'image prompt', 'bild prompt', 'bild-prompt'}
    for label, value in labeled_parts:
        if label in prompt_labels and value and not _image_prompt_candidate_is_code_or_css_polluted(value):
            return re.sub(r'\s+', ' ', value).strip(' ,;')
    visual_parts: list[str] = []
    for index, raw_part in enumerate(raw_parts):
        label, value = labeled_parts[index]
        if not value or _image_prompt_candidate_is_code_or_css_polluted(value):
            continue
        if label in {'id', 'asset id', 'user', 'username', 'handle', 'influencer', 'caption'}:
            continue
        if _SOCIAL_MANIFEST_HANDLE_RE.match(value) or str(raw_part or '').strip().lstrip('`*_ ').startswith('@'):
            continue
        if _social_manifest_piece_is_quoted(raw_part):
            continue
        if _social_manifest_piece_looks_like_caption_copy(raw_part):
            continue
        is_last_piece = index == len(raw_parts) - 1
        if (
            label in {'location', 'scenario', 'setting', 'context', 'mood', 'style', 'visual style'}
            or is_last_piece
            or _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(value)
            or _NUMBERED_PREPARED_IMAGE_SIGNAL_RE.search(value)
        ):
            if value not in visual_parts:
                visual_parts.append(value)
    prompt = ', '.join(part for part in [subject, *visual_parts] if part)
    prompt = re.sub(r'[`|@]+', '', prompt)
    prompt = re.sub(r'\s+', ' ', prompt).strip(' ,;')
    if not prompt or _image_prompt_candidate_is_code_or_css_polluted(prompt):
        return ''
    if len(re.findall(r'\w+', prompt)) < 4:
        prompt = f'{prompt}, social media portrait'
    return re.sub(r'\s+', ' ', prompt).strip(' ,;')


def _extract_filename_social_asset_image_prompt_lines(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    prompts: list[str] = []
    for match in _FILENAME_SOCIAL_ASSET_ROW_RE.finditer(str(prepared_text or '')):
        prompt = _filename_social_asset_prompt_from_row(
            match.group('filename'),
            match.group('body'),
        )
        if prompt:
            prompts.append(prompt)
        if expected_count > 0 and len(prompts) >= expected_count:
            break
    if expected_count > 0:
        return prompts[:expected_count]
    return prompts


def _image_prompt_batch_looks_like_weak_social_copy(prompts: Any) -> bool:
    if not isinstance(prompts, list) or not prompts:
        return False
    weak_count = 0
    for raw_prompt in prompts:
        prompt = str(raw_prompt or '').strip()
        if not prompt or _image_prompt_candidate_is_code_or_css_polluted(prompt):
            weak_count += 1
            continue
        if (
            _SOCIAL_MANIFEST_VISUAL_SIGNAL_RE.search(prompt)
            or _NUMBERED_PREPARED_IMAGE_SIGNAL_RE.search(prompt)
        ):
            continue
        words = re.findall(r'\w+', prompt)
        if len(words) <= 8:
            weak_count += 1
            continue
        if _social_manifest_piece_looks_like_caption_copy(prompt):
            weak_count += 1
    return weak_count >= max(1, (len(prompts) + 1) // 2)


def _embedded_labeled_image_prompt_body(raw_value: Any) -> str:
    text = str(raw_value or '').strip()
    if not text:
        return ''
    matches = list(_EMBEDDED_LABELED_IMAGE_PROMPT_RE.finditer(text))
    if not matches:
        return ''
    return str(matches[-1].group('body') or '').strip()


def _clean_numbered_image_prompt_candidate(raw_value: Any) -> str:
    text = str(raw_value or '').strip()
    if not text:
        return ''
    embedded_body = _embedded_labeled_image_prompt_body(text)
    if embedded_body:
        text = embedded_body
    social_manifest_body = _strip_social_manifest_image_prompt_metadata(text)
    if social_manifest_body:
        text = social_manifest_body
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[:\-–—]\s*', '', text).strip()
    text = _strip_handoff_markdown_wrappers(text).strip()
    text = re.sub(r'^(?:>\s*)+', '', text).strip()
    text = _IMAGE_ASSET_LABEL_PREFIX_RE.sub('', text).strip()
    text = re.sub(r'^(?:\*\*+|__+|`+)\s*', '', text).strip()
    text = re.sub(r'\s*(?:\*\*+|__+|`+)$', '', text).strip()
    lowered = text.lower()
    if not text:
        return ''
    if _image_prompt_candidate_is_code_or_css_polluted(text):
        return ''
    if re.fullmatch(r'(?:image\s+generation\s+)?prompts?', lowered):
        return ''
    if len(re.findall(r'\w+', text)) < 3:
        return ''
    return text


def _clean_image_prompt_boundary_candidate(raw_value: Any) -> str:
    return _clean_numbered_image_prompt_candidate(raw_value)


def _is_non_image_markdown_section_heading(raw_line: Any) -> bool:
    line = str(raw_line or '').strip()
    if not line or _is_image_prompt_section_heading(line):
        return False
    if re.fullmatch(r'#{1,6}\s+\S.*', line):
        return True
    return bool(
        re.fullmatch(
            r'(?P<wrapper>\*\*|__)\s*\S(?:.*\S)?\s*(?P=wrapper)',
            line,
        )
    )


def _extract_sequential_bold_alpha_image_prompt_lines(
    prepared_text: str,
    *,
    expected_count: int = 0,
    require_image_heading: bool = False,
) -> list[str]:
    """Extract a strongly ordered Markdown A/B/C image-prompt sequence.

    Single-letter labels are otherwise too ambiguous to treat as image prompts.
    Requiring bold wrappers, an A start, at least two consecutive labels, and no
    gaps keeps this parser distinct from general prose and section headings.
    """
    prompts: list[str] = []
    next_label_code = ord('A')
    sequence_started = False
    image_section_seen = not require_image_heading
    for raw_line in str(prepared_text or '').splitlines():
        line = str(raw_line or '').strip()
        if not line:
            continue
        if require_image_heading and not image_section_seen:
            if _is_image_prompt_section_heading(raw_line):
                image_section_seen = True
            continue
        if (
            require_image_heading
            and not sequence_started
            and (
                _is_image_prompt_section_stop(raw_line)
                or _is_non_image_markdown_section_heading(raw_line)
            )
        ):
            break
        match = _SEQUENTIAL_BOLD_ALPHA_IMAGE_PROMPT_LINE_RE.fullmatch(raw_line)
        if not sequence_started:
            if not match or str(match.group('label') or '') != 'A':
                continue
            sequence_started = True
        elif not match:
            break
        label = str(match.group('label') or '')
        if label != chr(next_label_code):
            break
        prompt = _clean_image_prompt_boundary_candidate(match.group('body'))
        if not prompt:
            break
        prompts.append(prompt)
        next_label_code += 1
        if expected_count > 0 and len(prompts) >= expected_count:
            break
    if len(prompts) < 2:
        return []
    if expected_count > 0:
        return prompts[:expected_count]
    return prompts


def _extract_leading_plain_alpha_image_prompt_lines(
    prepared_text: str,
    *,
    expected_count: int,
) -> list[str]:
    """Extract an exact graph-counted leading A/B/C producer sequence.

    Plain single-letter labels are ambiguous without caller-owned preparation
    context. This helper therefore returns authority only for a complete leading
    sequence whose count is already proven by the image-generation graph.
    """
    normalized_count = int(expected_count or 0)
    if normalized_count < 2 or normalized_count > 26:
        return []
    lines = str(prepared_text or '').splitlines()
    first_index = next(
        (index for index, raw_line in enumerate(lines) if str(raw_line or '').strip()),
        None,
    )
    if first_index is None:
        return []
    prompts: list[str] = []
    for offset in range(normalized_count):
        line_index = first_index + offset
        if line_index >= len(lines) or not str(lines[line_index] or '').strip():
            return []
        match = _PLAIN_ALPHA_IMAGE_PROMPT_LINE_RE.fullmatch(lines[line_index])
        if not match or str(match.group('label') or '') != chr(ord('A') + offset):
            return []
        prompt = _clean_image_prompt_boundary_candidate(match.group('body'))
        if not prompt:
            return []
        prompts.append(prompt)
    next_index = first_index + normalized_count
    if next_index < len(lines):
        next_match = _PLAIN_ALPHA_IMAGE_PROMPT_LINE_RE.fullmatch(lines[next_index])
        if next_match:
            return []
    return prompts if len(prompts) == normalized_count else []


def _contains_plain_alpha_image_prompt_line(prepared_text: str) -> bool:
    return any(
        _PLAIN_ALPHA_IMAGE_PROMPT_LINE_RE.fullmatch(str(raw_line or ''))
        for raw_line in str(prepared_text or '').splitlines()
    )


def _inline_labeled_image_prompt_body(raw_line: Any) -> Optional[str]:
    match = _INLINE_LABELED_IMAGE_PROMPT_LINE_RE.match(str(raw_line or ''))
    if not match:
        if _ARTIFACT_LABELED_IMAGE_PROMPT_HEADING_RE.fullmatch(str(raw_line or '')):
            return ''
        return None
    body = str(match.group('body') or '').strip()
    body = re.sub(r'^(?:\*\*+|__+|`+)\s*', '', body).strip()
    return body


def _inline_labeled_image_prompt_body_is_title(raw_value: Any) -> bool:
    text = str(raw_value or '').strip()
    if not text or not re.search(r'(?:\*\*+|__+)\s*$', text):
        return False
    normalized = re.sub(r'(?:\*\*+|__+|`+)', '', text).strip()
    if not normalized or re.search(r'[.!?]\s*$', normalized):
        return False
    return len(re.findall(r'\w+', normalized)) <= 10


def _extract_inline_labeled_image_prompt_lines(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    lines = str(prepared_text or '').splitlines()
    prompts: list[str] = []
    index = 0
    while index < len(lines):
        raw_line = str(lines[index] or '')
        stripped = raw_line.strip()
        if prompts and (_MARKDOWN_SEPARATOR_LINE_RE.fullmatch(stripped) or _is_image_prompt_section_stop(stripped)):
            break
        body = _inline_labeled_image_prompt_body(raw_line)
        if body is None:
            index += 1
            continue
        collect_following_body = (
            not body
            or _inline_labeled_image_prompt_body_is_title(body)
        )
        if collect_following_body:
            collected: list[str] = []
            lookahead = index + 1
            while lookahead < len(lines):
                next_line = str(lines[lookahead] or '')
                next_stripped = next_line.strip()
                if (
                    _inline_labeled_image_prompt_body(next_line) is not None
                    or _ARTIFACT_SECTION_HEADING_RE.fullmatch(next_line)
                    or _MARKDOWN_SEPARATOR_LINE_RE.fullmatch(next_stripped)
                    or _is_image_prompt_section_stop(next_stripped)
                ):
                    break
                if next_stripped:
                    collected.append(next_line)
                lookahead += 1
            if collected or not body:
                body = '\n'.join(collected).strip()
            index = lookahead
        else:
            index += 1
        prompt = _clean_image_prompt_boundary_candidate(body)
        if prompt and prompt not in prompts:
            prompts.append(prompt)
        if expected_count > 0 and len(prompts) >= expected_count:
            break
    if expected_count > 0:
        return prompts[:expected_count]
    return prompts


def _split_numbered_image_prompt_segment(segment: str) -> list[str]:
    text = str(segment or '').strip()
    if not text:
        return []
    for pattern in _NUMBERED_IMAGE_PROMPT_SEGMENT_PATTERNS:
        prompts: list[str] = []
        for match in re.finditer(pattern, text):
            cleaned = _clean_numbered_image_prompt_candidate(match.group('body'))
            if cleaned and cleaned not in prompts:
                prompts.append(cleaned)
        if len(prompts) >= 2:
            return prompts
    return []


def _split_numbered_image_prompt_raw_items(segment: str) -> list[str]:
    text = str(segment or '').strip()
    if not text:
        return []
    for pattern in _NUMBERED_IMAGE_PROMPT_SEGMENT_PATTERNS:
        items = [
            str(match.group('body') or '').strip()
            for match in re.finditer(pattern, text)
            if str(match.group('body') or '').strip()
        ]
        if len(items) >= 2:
            return items
    return []


def _numbered_image_prompts_allow_structural_surplus(
    segment: str,
    prompts: list[str],
    *,
    expected_count: int = 0,
) -> bool:
    if expected_count <= 0 or len(prompts) != expected_count + 1:
        return False
    raw_items = _split_numbered_image_prompt_raw_items(segment)
    if len(raw_items) != len(prompts):
        return False
    structural_count = sum(
        1
        for item in raw_items
        if _STRUCTURAL_PAGE_IMAGE_ASSET_LABEL_RE.search(item)
    )
    return structural_count == 1 and len(raw_items) - structural_count >= expected_count


def _limit_numbered_image_prompts_for_expected_count(
    prompts: list[str],
    *,
    expected_count: int = 0,
    segment: str = '',
) -> list[str]:
    if expected_count <= 0 or len(prompts) <= expected_count:
        return prompts
    if _numbered_image_prompts_allow_structural_surplus(
        segment,
        prompts,
        expected_count=expected_count,
    ):
        return prompts
    return prompts[:expected_count]


def _markdown_table_cells(raw_line: Any) -> list[str]:
    line = str(raw_line or '').strip()
    if not line.startswith('|') or '|' not in line[1:]:
        return []
    line = line.strip('|')
    return [
        _strip_handoff_markdown_wrappers(cell).strip()
        for cell in line.split('|')
    ]


def _markdown_table_separator_cells(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r':?-{3,}:?', str(cell or '').strip())
        for cell in cells
    )


def _markdown_table_prompt_column_index(cells: list[str]) -> int:
    normalized = [
        re.sub(r'[^a-z0-9]+', ' ', str(cell or '').strip().lower()).strip()
        for cell in cells
    ]
    preferred_patterns = (
        r'\bvisual\b.*\bprompt\b',
        r'\bprompt\b.*\bspecification\b',
        r'\bvisual\b.*\bspecification\b',
        r'\bimage\b.*\bprompt\b',
        r'\bprompt\b',
    )
    for pattern in preferred_patterns:
        for index, cell in enumerate(normalized):
            if re.search(pattern, cell):
                return index
    return -1


def _markdown_table_header_token(raw_value: Any) -> str:
    return re.sub(r'[^a-z0-9@]+', ' ', str(raw_value or '').strip().lower()).strip()


def _markdown_table_row_context_role(header_value: Any) -> str:
    header = _markdown_table_header_token(header_value)
    if not header:
        return ''
    if re.search(r'(?:^|\s)(?:id|asset|file|path|filename|image|img|slot|order|number)(?:\s|$)', header):
        return ''
    if '@' in header or re.search(r'\b(?:user\s*name|username|handle|influencer|subject|species|animal|character|name)\b', header):
        return 'subject'
    if re.search(r'\bcaption\b', header):
        return ''
    if re.search(r'\b(?:location|scenario|setting|context|mood|description|place)\b', header):
        return 'context'
    return ''


def _markdown_table_row_context_piece_is_placeholder(value: str, *, role: str) -> bool:
    text = re.sub(r'\s+', ' ', str(value or '').strip().lower()).strip(' .,:;_-')
    if not text:
        return True
    compact = re.sub(r'[\s_-]+', '', text)
    if compact in {'na', 'none', 'null', 'tbd', 'todo', 'placeholder'}:
        return True
    if role == 'subject' and re.fullmatch(
        r'(?:animal|influencer|user|username|handle|subject|species|character|asset|image|img|snout|selfie)\d{0,4}',
        compact,
    ):
        return True
    if role == 'context' and re.fullmatch(
        r'(?:caption|location|scenario|setting|context|mood|description|place)\d{0,4}',
        compact,
    ):
        return True
    return False


def _markdown_table_row_context_piece(cell_value: Any, *, role: str) -> str:
    text = _clean_social_manifest_piece(cell_value)
    if not text:
        return ''
    handle_match = _SOCIAL_MANIFEST_HANDLE_RE.match(text)
    if role == 'subject' and handle_match:
        text = _social_manifest_handle_to_subject(handle_match.group('handle'))
    else:
        text = re.sub(r'^@', '', text).strip()
    text = re.sub(r'\.(?:png|jpe?g|webp|gif|avif)$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[`|]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip(' ,;.')
    if not text or _image_prompt_candidate_is_code_or_css_polluted(text):
        return ''
    if _markdown_table_row_context_piece_is_placeholder(text, role=role):
        return ''
    return text


def _image_prompt_candidate_has_subject_anchor(prompt: str) -> bool:
    text = str(prompt or '').strip()
    if not text:
        return False
    lowered = text.lower()
    visual_lead_words = {
        'aerial',
        'black',
        'bright',
        'cinematic',
        'cold',
        'crisp',
        'dappled',
        'dark',
        'deep',
        'dreamy',
        'extreme',
        'golden',
        'high',
        'low',
        'macro',
        'moody',
        'motion',
        'soft',
        'surface',
        'tropical',
        'ultra',
        'underwater',
        'vibrant',
        'warm',
        'wide',
    }
    if re.search(r'\b(?:of|with|showing|featuring|depicting|portrait of|photo of|shot of)\s+(?:a|an|the)?\s*[a-z0-9][\w-]+', lowered):
        return True
    article_match = re.search(r'\b(?:a|an|the)\s+([a-z][\w-]+)', lowered)
    if article_match and article_match.group(1) not in visual_lead_words:
        return True
    capitalized_entities = re.findall(r'(?<!^)\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+){0,3}\b', text)
    if capitalized_entities:
        return True
    return False


_HTML_IMG_TAG_RE = re.compile(r'(?is)<img\b(?P<attrs>[^>]*)>')
_HTML_ATTR_RE = re.compile(
    r'''(?is)\b(?P<name>[a-z_:][\w:.-]*)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)'''
)
_HTML_TEXT_TAG_RE = re.compile(
    r'(?is)<(?P<tag>p|span|h[1-6]|strong|em)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>'
)
_HTML_HEX_BYTE_FRAGMENT_RE = re.compile(r'(?:<0x[0-9a-fA-F]{2}>)+')
_HTML_CARD_CONTEXT_CLASS_RE = re.compile(
    r'(?i)\b(?:caption|style|tag|prompt|description|text|location|scenario|mood)\b'
)
_HTML_CARD_HANDLE_CLASS_RE = re.compile(r'(?i)\b(?:user|username|handle|influencer|author)\b')
_HTML_CARD_CONTEXT_STOP_RE = re.compile(r'(?is)<footer\b|</(?:section|main|article)\b')


def _html_attrs(raw_attrs: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _HTML_ATTR_RE.finditer(str(raw_attrs or '')):
        name = str(match.group('name') or '').strip().lower()
        value = html.unescape(str(match.group('value') or '').strip())
        if name:
            attrs[name] = value
    return attrs


def _clean_html_prompt_text(raw_value: Any) -> str:
    text = html.unescape(str(raw_value or ''))
    text = re.sub(r'(?is)<(?:script|style)\b.*?</(?:script|style)>', ' ', text)
    text = _HTML_HEX_BYTE_FRAGMENT_RE.sub(' ', text)
    text = re.sub(r'(?is)<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip().strip('"“”')
    return text.strip(' ,;')


def _html_card_context_piece(raw_value: Any, *, role: str = '') -> str:
    text = _clean_html_prompt_text(raw_value)
    if not text:
        return ''
    if role == 'handle':
        handle_match = _SOCIAL_MANIFEST_HANDLE_RE.match(text)
        if handle_match:
            text = _social_manifest_handle_to_subject(handle_match.group('handle'))
        else:
            text = text.lstrip('@')
    else:
        text = _clean_social_manifest_piece(text)
    text = re.sub(r'\s+', ' ', text).strip(' ,;.')
    if not text or _image_prompt_candidate_is_code_or_css_polluted(text):
        return ''
    return text


def _extract_html_image_card_prompt_units(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    text = str(prepared_text or '')
    if '<img' not in text.lower():
        return []
    image_matches = list(_HTML_IMG_TAG_RE.finditer(text))
    if len(image_matches) < 2:
        return []
    prompts: list[str] = []
    for index, match in enumerate(image_matches):
        attrs = _html_attrs(match.group('attrs'))
        alt = _html_card_context_piece(attrs.get('alt'))
        src = str(attrs.get('src') or '').strip()
        if not alt and not src:
            continue
        next_start = image_matches[index + 1].start() if index + 1 < len(image_matches) else len(text)
        raw_segment = text[match.end(): min(next_start, match.end() + 1800)]
        stop_match = _HTML_CARD_CONTEXT_STOP_RE.search(raw_segment)
        segment = raw_segment[:stop_match.start()] if stop_match else raw_segment
        context_parts: list[str] = []
        handle_parts: list[str] = []
        for text_match in _HTML_TEXT_TAG_RE.finditer(segment):
            tag_attrs = _html_attrs(text_match.group('attrs'))
            class_text = str(tag_attrs.get('class') or '').strip()
            body = str(text_match.group('body') or '')
            role = 'handle' if _HTML_CARD_HANDLE_CLASS_RE.search(class_text) else ''
            if role == 'handle' or _clean_html_prompt_text(body).lstrip().startswith('@'):
                piece = _html_card_context_piece(body, role='handle')
                if piece and piece not in handle_parts:
                    handle_parts.append(piece)
                continue
            if class_text and not _HTML_CARD_CONTEXT_CLASS_RE.search(class_text):
                continue
            piece = _html_card_context_piece(body)
            if piece and piece not in context_parts:
                context_parts.append(piece)
        seed_parts = [alt] if alt else handle_parts[:1]
        if not seed_parts:
            continue
        prompt = _clean_image_prompt_boundary_candidate(', '.join([*seed_parts, *context_parts]))
        if prompt and prompt not in prompts:
            prompts.append(prompt)
        if expected_count > 0 and len(prompts) >= expected_count:
            break
    if expected_count > 0:
        return prompts[:expected_count]
    return prompts


def _markdown_table_compose_row_context_prompt(
    header_cells: list[str],
    row_cells: list[str],
    prompt_column: int,
    prompt: str,
) -> str:
    cleaned_prompt = _clean_image_prompt_boundary_candidate(prompt)
    if not cleaned_prompt:
        return ''
    if _image_prompt_candidate_has_subject_anchor(cleaned_prompt):
        return cleaned_prompt
    context_parts: list[str] = []
    for index, header in enumerate(header_cells):
        if index == prompt_column or index >= len(row_cells):
            continue
        role = _markdown_table_row_context_role(header)
        if not role:
            continue
        piece = _markdown_table_row_context_piece(row_cells[index], role=role)
        if piece and piece not in context_parts:
            context_parts.append(piece)
    if not context_parts:
        return cleaned_prompt
    composed = ', '.join([*context_parts, cleaned_prompt])
    return _clean_image_prompt_boundary_candidate(composed)


def _extract_markdown_table_image_prompts_from_lines(
    raw_lines: list[str],
    *,
    expected_count: int = 0,
) -> list[str]:
    table_candidates: list[list[str]] = []
    prompt_column = -1
    header_cells: list[str] = []
    table_started = False
    current_prompts: list[str] = []

    def finish_current_table() -> None:
        nonlocal prompt_column, header_cells, table_started, current_prompts
        if len(current_prompts) >= 2:
            table_candidates.append(list(current_prompts))
        prompt_column = -1
        header_cells = []
        table_started = False
        current_prompts = []

    for raw_line in [*raw_lines, '']:
        cells = _markdown_table_cells(raw_line)
        if not cells:
            if table_started or prompt_column >= 0:
                finish_current_table()
            continue
        if prompt_column < 0:
            candidate_column = _markdown_table_prompt_column_index(cells)
            if candidate_column < 0:
                continue
            prompt_column = candidate_column
            header_cells = list(cells)
            table_started = True
            continue
        table_started = True
        if _markdown_table_separator_cells(cells):
            continue
        if prompt_column >= len(cells):
            continue
        prompt = _markdown_table_compose_row_context_prompt(
            header_cells,
            cells,
            prompt_column,
            cells[prompt_column],
        )
        if prompt and prompt not in current_prompts:
            current_prompts.append(prompt)
        if expected_count > 0 and len(current_prompts) >= expected_count:
            return current_prompts[:expected_count]

    if not table_candidates:
        return []
    if expected_count > 0:
        for prompts in table_candidates:
            if len(prompts) >= expected_count:
                return prompts[:expected_count]
    best_candidate = max(table_candidates, key=len)
    return best_candidate[:expected_count] if expected_count > 0 else best_candidate


def _extract_markdown_table_image_prompt_section(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    text = str(prepared_text or '').strip()
    if not text:
        return []
    lines = text.splitlines()
    start_index: Optional[int] = None
    for index, raw_line in enumerate(lines):
        if _is_image_prompt_section_heading(raw_line):
            start_index = index + 1
            break
    if start_index is None:
        return []
    section_lines: list[str] = []
    for raw_line in lines[start_index:]:
        if _is_image_prompt_section_stop(raw_line):
            break
        section_lines.append(str(raw_line or ''))
    return _extract_markdown_table_image_prompts_from_lines(
        section_lines,
        expected_count=expected_count,
    )


def _extract_markdown_table_image_prompt_anywhere(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    text = str(prepared_text or '').strip()
    if not text:
        return []
    return _extract_markdown_table_image_prompts_from_lines(
        text.splitlines(),
        expected_count=expected_count,
    )


def _extract_social_manifest_pipe_image_prompt_lines(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    prompts: list[str] = []
    for raw_line in str(prepared_text or '').splitlines():
        line = str(raw_line or '').strip()
        if not line:
            continue
        numbered_match = re.match(
            r'^\s*(?:[-*#>\u2022]+\s*)?(?:\d{1,3}|[ivx]+)[.)]\s+',
            line,
            flags=re.IGNORECASE,
        )
        prefix_prompt = _strip_social_manifest_prefix_label(line)
        if '|' not in line and numbered_match is None and not prefix_prompt:
            continue
        body = re.sub(
            r'^\s*(?:[-*#>\u2022]+\s*)?(?:\d{1,3}|[ivx]+)[.)]\s+',
            '',
            line,
            flags=re.IGNORECASE,
        ).strip()
        prompt = _strip_social_manifest_image_prompt_metadata(body)
        if prompt and prompt not in prompts:
            prompts.append(prompt)
        if expected_count > 0 and len(prompts) >= expected_count:
            break
    if expected_count > 0:
        return prompts[:expected_count]
    return prompts


def _extract_numbered_image_prompt_section(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    text = str(prepared_text or '').strip()
    if not text:
        return []
    lines = text.splitlines()
    start_index: Optional[int] = None
    for index, raw_line in enumerate(lines):
        if _is_image_prompt_section_heading(raw_line):
            start_index = index + 1
            break
    if start_index is None:
        return []
    section_lines: list[str] = []
    for raw_line in lines[start_index:]:
        if _is_image_prompt_section_stop(raw_line):
            break
        section_lines.append(str(raw_line or ''))
    section_text = '\n'.join(section_lines)
    prompts = _split_numbered_image_prompt_segment(section_text)
    return _limit_numbered_image_prompts_for_expected_count(
        prompts,
        expected_count=expected_count,
        segment=section_text,
    )


def _extract_numbered_prepared_image_prompt_units(
    prepared_text: str,
    *,
    expected_count: int = 0,
) -> list[str]:
    text = str(prepared_text or '').strip()
    if not text:
        return []
    prompts: list[str] = []
    for match in _NUMBERED_PREPARED_IMAGE_PROMPT_RE.finditer(text):
        body = str(match.group('body') or '').strip()
        if not body:
            continue
        if _is_image_prompt_section_stop(body):
            break
        cleaned = _clean_numbered_image_prompt_candidate(body)
        if not cleaned:
            continue
        if not _NUMBERED_PREPARED_IMAGE_SIGNAL_RE.search(cleaned):
            continue
        if cleaned not in prompts:
            prompts.append(cleaned)
        if expected_count > 1 and len(prompts) >= expected_count:
            break
    if len(prompts) < 2:
        return []
    return prompts[:expected_count] if expected_count > 0 else prompts


@dataclass
class ResponseSemanticsRuntimeOwner:
    hooks: dict[str, Any]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def artifact_type_for_capability(self, capability: Optional[str]) -> Optional[str]:
        normalized = normalize_capability(capability)
        if normalized == CAPABILITY_IMAGE_GENERATION:
            return 'image'
        if normalized == CAPABILITY_TEXT_TO_SPEECH:
            return 'audio'
        if normalized == CAPABILITY_SPEECH_TO_TEXT:
            return 'text'
        return None

    def response_payload_has_artifact_type(
        self,
        payload: Optional[dict[str, Any]],
        artifact_type: Optional[str],
    ) -> bool:
        build_canonical_response_artifacts = self._hook('build_canonical_response_artifacts')

        if not isinstance(payload, dict):
            return False
        normalized_type = str(artifact_type or '').strip().lower()
        if not normalized_type:
            return False
        artifacts = payload.get('artifacts')
        if not isinstance(artifacts, list):
            artifacts = build_canonical_response_artifacts(payload)
        artifact_types = {
            str(item.get('type') or '').strip().lower()
            for item in artifacts
            if isinstance(item, dict) and str(item.get('type') or '').strip()
        }
        if normalized_type == 'image':
            return bool(
                str(payload.get('saved_image_path') or payload.get('savedImagePath') or '').strip()
                or str(payload.get('image_data_url') or payload.get('imageDataUrl') or '').strip()
                or 'image' in artifact_types
            )
        if normalized_type == 'audio':
            return bool(
                str(payload.get('saved_audio_path') or payload.get('savedAudioPath') or '').strip()
                or 'audio' in artifact_types
            )
        if normalized_type == 'text':
            return bool(
                str(payload.get('saved_text_path') or payload.get('savedTextPath') or '').strip()
                or 'text' in artifact_types
            )
        return normalized_type in artifact_types

    def response_payload_artifact_type_count(
        self,
        payload: Optional[dict[str, Any]],
        artifact_type: Optional[str],
    ) -> int:
        build_canonical_response_artifacts = self._hook('build_canonical_response_artifacts')

        if not isinstance(payload, dict):
            return 0
        normalized_type = str(artifact_type or '').strip().lower()
        if not normalized_type:
            return 0
        artifacts = payload.get('artifacts')
        if not isinstance(artifacts, list):
            artifacts = build_canonical_response_artifacts(payload)
        aliases = {
            'image': {'image', 'png', 'jpg', 'jpeg', 'webp'},
            'audio': {'audio', 'wav', 'mp3', 'm4a', 'flac'},
            'text': {'text', 'markdown', 'md', 'json', 'csv', 'message'},
        }.get(normalized_type, {normalized_type})
        count = 0
        for item in artifacts or []:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get('type') or '').strip().lower()
            if item_type in aliases:
                count += 1
        if normalized_type == 'image' and count == 0 and (
            str(payload.get('saved_image_path') or payload.get('savedImagePath') or '').strip()
            or str(payload.get('image_data_url') or payload.get('imageDataUrl') or '').strip()
        ):
            return 1
        if normalized_type == 'audio' and count == 0 and str(
            payload.get('saved_audio_path') or payload.get('savedAudioPath') or ''
        ).strip():
            return 1
        if normalized_type == 'text' and count == 0 and str(
            payload.get('saved_text_path') or payload.get('savedTextPath') or ''
        ).strip():
            return 1
        return count

    def response_payload_real_text_artifact_count(
        self,
        payload: Optional[dict[str, Any]],
    ) -> int:
        build_canonical_response_artifacts = self._hook('build_canonical_response_artifacts')

        if not isinstance(payload, dict):
            return 0
        paths: set[str] = set()
        for key in ('saved_text_path', 'savedTextPath'):
            value = str(payload.get(key) or '').strip()
            if value:
                paths.add(value)
        for item in payload.get('saved_text_artifacts') or []:
            if not isinstance(item, Mapping):
                continue
            value = str(item.get('path') or item.get('source_path') or '').strip()
            if value:
                paths.add(value)
        artifacts = payload.get('artifacts')
        if not isinstance(artifacts, list):
            artifacts = build_canonical_response_artifacts(payload)
        for item in artifacts or []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get('type') or '').strip().lower() != 'text':
                continue
            value = str(item.get('path') or item.get('source_path') or '').strip()
            if value:
                paths.add(value)
        return len(paths)

    @staticmethod
    def source_requires_text_artifact(source: Mapping[str, Any], output_type: str) -> bool:
        if str(output_type or '').strip().lower() != 'text':
            return False
        value = source.get('requires_artifact')
        if isinstance(value, bool):
            return value
        token = str(value or '').strip().lower()
        if token in {'true', 'yes', '1', 'required'}:
            return True
        role = str(source.get('role') or '').strip().lower()
        policy = str(source.get('fulfillment_policy') or '').strip().lower()
        return role in {'text_artifact_output', 'document_output'} or policy == 'runtime_text_artifact'

    @staticmethod
    def _branch_record_is_non_executable_candidate(source: Mapping[str, Any]) -> bool:
        status = str(source.get('status') or source.get('contract_state') or '').strip().lower()
        role = str(source.get('role') or '').strip().lower()
        execution_policy = str(
            source.get('execution_policy')
            or source.get('promotion_policy')
            or source.get('repair_execution_policy')
            or ''
        ).strip().lower()
        output_contract = source.get('output_contract') if isinstance(source.get('output_contract'), Mapping) else {}
        output_contract_status = str(output_contract.get('status') or '').strip().lower()
        if status in _NON_EXECUTABLE_BRANCH_STATUSES:
            return True
        if output_contract_status in _NON_EXECUTABLE_BRANCH_STATUSES:
            return True
        if role in _NON_EXECUTABLE_BRANCH_ROLES:
            return True
        if execution_policy in _NON_EXECUTABLE_POLICY_TOKENS:
            return True
        if execution_policy.startswith('non_executable'):
            return True
        return bool(source.get('reserved_for_later') or source.get('deferred_not_executable'))

    @staticmethod
    def _branch_record_is_terminally_fulfilled(source: Mapping[str, Any]) -> bool:
        status = str(source.get('status') or source.get('contract_state') or '').strip().lower()
        output_contract = source.get('output_contract') if isinstance(source.get('output_contract'), Mapping) else {}
        output_contract_status = str(output_contract.get('status') or '').strip().lower()
        return (
            status in _TERMINAL_FULFILLED_BRANCH_STATUSES
            or output_contract_status in _TERMINAL_FULFILLED_BRANCH_STATUSES
        )

    @staticmethod
    def _request_graph_defers_downstream_execution(request_phase_graph: Optional[Mapping[str, Any]]) -> bool:
        if not isinstance(request_phase_graph, Mapping):
            return False
        prompt_intent = (
            request_phase_graph.get('prompt_intent')
            if isinstance(request_phase_graph.get('prompt_intent'), Mapping)
            else {}
        )
        if bool(prompt_intent.get('explicit_defer_materialization')) and not (
            bool(prompt_intent.get('requests_audio_output'))
            or bool(prompt_intent.get('requests_visual_output'))
            or bool(prompt_intent.get('requests_speech_to_text_output'))
            or bool(prompt_intent.get('has_audio_follow_up_request'))
            or bool(prompt_intent.get('has_visual_follow_up_request'))
            or bool(prompt_intent.get('text_preparation_before_audio_output'))
            or bool(prompt_intent.get('text_preparation_before_visual_output'))
        ):
            return True
        prompt = str(
            request_phase_graph.get('prompt')
            or prompt_intent.get('prompt')
            or prompt_intent.get('normalized_prompt')
            or ''
        ).strip()
        if not prompt:
            return False
        return bool(_CURRENT_ONLY_PROMPT_RE.search(prompt) and _DEFER_DOWNSTREAM_PROMPT_RE.search(prompt))

    def _coerce_positive_int(self, value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value if value > 0 else 0
        if isinstance(value, float):
            return int(value) if value > 0 else 0
        token = str(value or '').strip()
        if not token:
            return 0
        try:
            parsed = int(token)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    def _current_request_prompt_for_review(
        self,
        request_payload: Optional[dict[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> str:
        if isinstance(request_payload, dict):
            try:
                prompt = extract_responses_current_turn_prompt(request_payload)
            except Exception:
                logging.getLogger(__name__).debug(
                    'failed to extract current turn prompt for closure review',
                    exc_info=True,
                )
                prompt = ''
            if str(prompt or '').strip():
                return str(prompt or '').strip()
            for key in ('prompt', 'input', 'instructions', 'message', 'query'):
                value = request_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        graph_prompt = request_phase_graph.get('prompt') if isinstance(request_phase_graph, Mapping) else None
        return str(graph_prompt or '').strip()

    def _merged_prompt_intent_for_review(
        self,
        prompt: str,
        request_phase_graph: Mapping[str, Any],
    ) -> dict[str, Any]:
        graph_intent = (
            request_phase_graph.get('prompt_intent')
            if isinstance(request_phase_graph.get('prompt_intent'), Mapping)
            else {}
        )
        analyzed = analyze_prompt_intent(prompt) if str(prompt or '').strip() else {}
        current_turn_suppresses_visual_artifact_execution = bool(
            analyzed.get('visual_artifact_execution_suppressed_by_preservation')
        )
        current_turn_suppresses_visual_analysis_execution = bool(
            analyzed.get('visual_analysis_execution_suppressed_by_preservation')
        )
        merged: dict[str, Any] = {}
        for source in (analyzed, graph_intent):
            for key, value in source.items():
                if value not in (None, '', [], {}):
                    merged[key] = value
        for key in (
            'requests_visual_output',
            'has_visual_follow_up_request',
            'text_preparation_before_visual_output',
            'requests_audio_output',
            'requests_speech_to_text_output',
            'has_audio_follow_up_request',
            'text_preparation_before_audio_output',
            'direct_audio_materialization_request',
            'counted_audio_output_obligation',
            'audio_output_count_exceeds_bound',
            'explicit_defer_materialization',
            'explicit_visual_defer_materialization',
            'explicit_audio_defer_materialization',
        ):
            merged[key] = bool(analyzed.get(key) or graph_intent.get(key))
        intent_obligations = [
            item
            for item in (request_phase_graph.get('intent_obligations') or [])
            if isinstance(item, Mapping)
        ]
        required_obligations = required_intent_obligations(intent_obligations)
        required_summary = summarize_required_intent_obligations(required_obligations)
        required_capabilities = list(required_summary.get('capabilities') or [])
        required_capability_counts = dict(required_summary.get('capability_counts') or {})
        required_output_counts = dict(required_summary.get('material_output_counts') or {})

        analyzed_downstream = analyzed.get('downstream_follow_up_capabilities')
        graph_downstream = graph_intent.get('downstream_follow_up_capabilities')
        downstream: list[str] = []
        for raw_list in (analyzed_downstream, graph_downstream, required_capabilities):
            if not isinstance(raw_list, list):
                continue
            for item in raw_list:
                capability = normalize_capability(item)
                if capability and capability not in downstream:
                    downstream.append(capability)
        if downstream:
            merged['downstream_follow_up_capabilities'] = downstream
        merged['requested_visual_output_count'] = max(
            self._coerce_positive_int(analyzed.get('requested_visual_output_count')),
            self._coerce_positive_int(graph_intent.get('requested_visual_output_count')),
            self._coerce_positive_int(required_output_counts.get('image')),
        )
        if merged.get('audio_output_count_exceeds_bound'):
            merged['requested_audio_output_count'] = 0
        else:
            merged['requested_audio_output_count'] = max(
                self._coerce_positive_int(analyzed.get('requested_audio_output_count')),
                self._coerce_positive_int(graph_intent.get('requested_audio_output_count')),
                self._coerce_positive_int(required_output_counts.get('audio')),
            )
        if required_output_counts.get('image') and not merged.get('explicit_visual_defer_materialization'):
            merged['requests_visual_output'] = True
        if required_output_counts.get('audio') and not merged.get('explicit_audio_defer_materialization'):
            merged['requests_audio_output'] = True
        if required_capability_counts.get(CAPABILITY_SPEECH_TO_TEXT):
            merged['requests_speech_to_text_output'] = True

        required_phase_ids_by_capability: dict[str, set[str]] = {}
        for obligation in required_obligations:
            capability = normalize_capability(obligation.get('capability'))
            phase_id = str(obligation.get('phase_id') or obligation.get('branch_id') or '').strip()
            if capability and phase_id:
                required_phase_ids_by_capability.setdefault(capability, set()).add(phase_id)
        if normalize_capability(request_phase_graph.get('current_phase_capability')) == CAPABILITY_CHAT:
            current_phase_id = str(request_phase_graph.get('current_phase_id') or '').strip()
            for record in request_phase_graph.get('downstream_branches') or []:
                if not isinstance(record, Mapping):
                    continue
                capability = normalize_capability(record.get('capability'))
                phase_id = str(record.get('phase_id') or record.get('branch_id') or '').strip()
                dependencies = {
                    str(item or '').strip()
                    for item in (record.get('depends_on') or [])
                    if str(item or '').strip()
                }
                if (
                    not capability
                    or phase_id not in required_phase_ids_by_capability.get(capability, set())
                    or not current_phase_id
                    or current_phase_id not in dependencies
                ):
                    continue
                if capability == CAPABILITY_IMAGE_GENERATION:
                    merged['text_preparation_before_visual_output'] = True
                elif capability == CAPABILITY_TEXT_TO_SPEECH:
                    merged['text_preparation_before_audio_output'] = True

        # The fresh turn is the intent boundary.  A graph may still carry positive
        # visual cues or counts from the predecessor turn, but an explicit
        # preserve-without-regeneration instruction makes that image a reference,
        # not a newly owed output.  Only an independently detected request for a
        # new/separate visual artifact may keep image execution active.
        if current_turn_suppresses_visual_artifact_execution:
            merged['visual_artifact_preservation_without_regeneration'] = True
            merged['visual_artifact_execution_suppressed_by_preservation'] = True
            merged['requests_visual_output'] = False
            merged['has_visual_follow_up_request'] = False
            merged['text_preparation_before_visual_output'] = False
            merged['requested_visual_output_count'] = 0
            merged['counted_visual_output_obligation'] = False
            if normalize_capability(merged.get('primary_capability')) == CAPABILITY_IMAGE_GENERATION:
                merged.pop('primary_capability', None)
            downstream = [
                capability
                for capability in (merged.get('downstream_follow_up_capabilities') or [])
                if normalize_capability(capability) != CAPABILITY_IMAGE_GENERATION
            ]
            if downstream:
                merged['downstream_follow_up_capabilities'] = downstream
            else:
                merged.pop('downstream_follow_up_capabilities', None)
            required_output_counts.pop('image', None)
            required_capability_counts.pop(CAPABILITY_IMAGE_GENERATION, None)
            required_capabilities = [
                capability
                for capability in required_capabilities
                if normalize_capability(capability) != CAPABILITY_IMAGE_GENERATION
            ]
        if current_turn_suppresses_visual_analysis_execution:
            merged['visual_analysis_preservation_without_reanalysis'] = True
            merged['visual_analysis_execution_suppressed_by_preservation'] = True
            if normalize_capability(merged.get('primary_capability')) == CAPABILITY_VISION_ANALYSIS:
                merged.pop('primary_capability', None)
            downstream = [
                capability
                for capability in (merged.get('downstream_follow_up_capabilities') or [])
                if normalize_capability(capability) != CAPABILITY_VISION_ANALYSIS
            ]
            if downstream:
                merged['downstream_follow_up_capabilities'] = downstream
            else:
                merged.pop('downstream_follow_up_capabilities', None)
            required_capability_counts.pop(CAPABILITY_VISION_ANALYSIS, None)
            required_capabilities = [
                capability
                for capability in required_capabilities
                if normalize_capability(capability) != CAPABILITY_VISION_ANALYSIS
            ]
        merged['required_intent_obligation_count'] = int(required_summary.get('required_count') or 0)
        merged['required_intent_obligation_kinds'] = list(required_summary.get('kinds') or [])
        merged['required_intent_capabilities'] = required_capabilities
        merged['required_intent_capability_counts'] = required_capability_counts
        merged['required_intent_output_counts'] = required_output_counts
        return merged

    def _reserved_material_output_counts(
        self,
        request_phase_graph: Mapping[str, Any],
    ) -> dict[str, int]:
        if not isinstance(request_phase_graph, Mapping):
            return {}
        request_ir = (
            request_phase_graph.get('request_ir')
            if isinstance(request_phase_graph.get('request_ir'), Mapping)
            else {}
        )
        source_lists = (
            (request_phase_graph.get('candidate_graph') or {}).get('candidates')
            if isinstance(request_phase_graph.get('candidate_graph'), Mapping)
            else None,
            (request_ir.get('candidate_graph') or {}).get('candidates')
            if isinstance(request_ir.get('candidate_graph'), Mapping)
            else None,
            request_phase_graph.get('output_candidates'),
            request_ir.get('output_candidates'),
            request_phase_graph.get('downstream_branches'),
            request_phase_graph.get('phases'),
        )
        counts: dict[str, int] = {}
        seen: set[str] = set()
        for raw_list in source_lists:
            if not isinstance(raw_list, list):
                continue
            for raw_item in raw_list:
                if not isinstance(raw_item, Mapping):
                    continue
                status = ''
                for key in (
                    'status',
                    'candidate_status',
                    'contract_state',
                    'contract_status',
                    'obligation_state',
                    'intent_state',
                ):
                    status = str(raw_item.get(key) or '').strip().lower()
                    if status:
                        break
                if status not in _RESERVED_CANDIDATE_STATUSES:
                    continue
                capability = normalize_capability(raw_item.get('capability'))
                output_type = str(raw_item.get('output_type') or '').strip().lower()
                if not output_type:
                    output_type = str(self.artifact_type_for_capability(capability) or '').strip().lower()
                if output_type not in {'audio', 'image'}:
                    continue
                identity = str(
                    raw_item.get('branch_id')
                    or raw_item.get('phase_id')
                    or raw_item.get('obligation_id')
                    or raw_item.get('candidate_id')
                    or f'{output_type}:{len(seen) + 1}'
                ).strip()
                if identity in seen:
                    continue
                seen.add(identity)
                counts[output_type] = counts.get(output_type, 0) + 1
        return counts

    def _expected_material_output_counts(
        self,
        *,
        request_payload: Optional[dict[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> dict[str, int]:
        prompt = self._current_request_prompt_for_review(request_payload, request_phase_graph)
        prompt_intent = self._merged_prompt_intent_for_review(prompt, request_phase_graph)
        explicit_visual_defer = bool(prompt_intent.get('explicit_visual_defer_materialization'))
        explicit_audio_defer = bool(prompt_intent.get('explicit_audio_defer_materialization'))
        suppress_visual_artifact_execution = bool(
            prompt_intent.get('visual_artifact_execution_suppressed_by_preservation')
        )
        if (
            prompt_intent.get('explicit_defer_materialization')
            and not explicit_visual_defer
            and not explicit_audio_defer
        ):
            return {}

        downstream_capabilities = {
            normalize_capability(item)
            for item in (prompt_intent.get('downstream_follow_up_capabilities') or [])
            if normalize_capability(item)
        }
        current_capability = normalize_capability(request_phase_graph.get('current_phase_capability'))
        primary_capability = normalize_capability(prompt_intent.get('primary_capability'))
        expected: dict[str, int] = {}

        expects_image = bool(
            prompt_intent.get('requests_visual_output')
            or prompt_intent.get('has_visual_follow_up_request')
            or prompt_intent.get('text_preparation_before_visual_output')
            or self._coerce_positive_int(prompt_intent.get('requested_visual_output_count'))
            or CAPABILITY_IMAGE_GENERATION in downstream_capabilities
            or current_capability == CAPABILITY_IMAGE_GENERATION
            or primary_capability == CAPABILITY_IMAGE_GENERATION
        )
        if explicit_visual_defer or suppress_visual_artifact_execution:
            expects_image = False
        if expects_image:
            requested_count = self._coerce_positive_int(prompt_intent.get('requested_visual_output_count'))
            if isinstance(request_payload, Mapping):
                for key in ('batch_count', 'n', 'count'):
                    requested_count = max(requested_count, self._coerce_positive_int(request_payload.get(key)))
                batch_prompts = request_payload.get('batch_prompts')
                if isinstance(batch_prompts, list):
                    requested_count = max(requested_count, len([item for item in batch_prompts if item]))
            expected['image'] = max(1, requested_count)

        expects_audio = bool(
            not prompt_intent.get('audio_output_count_exceeds_bound')
            and (
                prompt_intent.get('requests_audio_output')
                or prompt_intent.get('has_audio_follow_up_request')
                or prompt_intent.get('text_preparation_before_audio_output')
                or prompt_intent.get('direct_audio_materialization_request')
                or CAPABILITY_TEXT_TO_SPEECH in downstream_capabilities
                or current_capability == CAPABILITY_TEXT_TO_SPEECH
                or primary_capability == CAPABILITY_TEXT_TO_SPEECH
            )
        )
        if explicit_audio_defer:
            expects_audio = False
        if expects_audio:
            expected['audio'] = max(
                1,
                self._coerce_positive_int(
                    prompt_intent.get('requested_audio_output_count')
                ),
            )

        required_summary = summarize_required_intent_obligations(
            request_phase_graph.get('intent_obligations')
            if isinstance(request_phase_graph.get('intent_obligations'), list)
            else []
        )
        required_output_counts = required_summary.get('material_output_counts')
        if not isinstance(required_output_counts, Mapping):
            required_output_counts = {}
        if not explicit_visual_defer and not suppress_visual_artifact_execution:
            image_count = self._coerce_positive_int(required_output_counts.get('image'))
            if image_count:
                expected['image'] = max(expected.get('image', 0), image_count)
        if not explicit_audio_defer:
            audio_count = self._coerce_positive_int(required_output_counts.get('audio'))
            if audio_count:
                expected['audio'] = max(expected.get('audio', 0), audio_count)

        return expected

    def _graph_material_output_counts(self, request_phase_graph: Mapping[str, Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        obligations = output_obligations_from_graph(request_phase_graph)
        sources = obligations if obligations else request_phase_graph.get('phases')
        if not isinstance(sources, list):
            return counts
        current_phase_id = str(request_phase_graph.get('current_phase_id') or '').strip()
        for raw_source in sources:
            if not isinstance(raw_source, Mapping):
                continue
            phase_id = str(raw_source.get('phase_id') or '').strip()
            if not obligations and current_phase_id and phase_id == current_phase_id:
                continue
            required = raw_source.get('required')
            if required is False:
                continue
            output_type = str(raw_source.get('output_type') or '').strip().lower()
            if not output_type:
                output_type = str(
                    self.artifact_type_for_capability(raw_source.get('capability')) or ''
                ).strip().lower()
            if output_type in {'image', 'audio'}:
                counts[output_type] = counts.get(output_type, 0) + 1
        return counts

    def _graph_required_capability_counts(self, request_phase_graph: Mapping[str, Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        obligations = output_obligations_from_graph(request_phase_graph)
        sources = obligations if obligations else request_phase_graph.get('phases')
        if not isinstance(sources, list):
            return counts
        current_phase_id = str(request_phase_graph.get('current_phase_id') or '').strip()
        for raw_source in sources:
            if not isinstance(raw_source, Mapping):
                continue
            if raw_source.get('required') is False:
                continue
            capability = normalize_capability(raw_source.get('capability'))
            if not capability:
                continue
            if capability == CAPABILITY_CHAT:
                phase_id = str(raw_source.get('phase_id') or raw_source.get('branch_id') or '').strip()
                if not phase_id or phase_id == current_phase_id:
                    continue
            counts[capability] = counts.get(capability, 0) + 1
        return counts

    @staticmethod
    def _intent_obligation_summary(intent_obligations: list[Mapping[str, Any]]) -> dict[str, Any]:
        kind_counts: dict[str, int] = {}
        dependency_contract_counts: dict[str, int] = {}
        required_count = 0
        execution_dependency_count = 0
        for obligation in required_intent_obligations(intent_obligations):
            required_count += 1
            kind = str(obligation.get('kind') or '').strip().lower()
            if kind:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            dependency_contract = str(obligation.get('dependency_contract') or '').strip().lower()
            if dependency_contract:
                dependency_contract_counts[dependency_contract] = (
                    dependency_contract_counts.get(dependency_contract, 0) + 1
                )
            if kind == 'dependency' and obligation.get('execution_dependency_required') is True:
                execution_dependency_count += 1
        return {
            'required_count': required_count,
            'kind_counts': kind_counts,
            'dependency_contract_counts': dependency_contract_counts,
            'execution_dependency_count': execution_dependency_count,
        }

    @staticmethod
    def _phase_lookup_for_intent_review(request_phase_graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        phase_lookup: dict[str, Mapping[str, Any]] = {}
        if not isinstance(request_phase_graph, Mapping):
            return phase_lookup
        for collection_key in ('phases', 'downstream_branches'):
            collection = request_phase_graph.get(collection_key)
            if not isinstance(collection, list):
                continue
            for record in collection:
                if not isinstance(record, Mapping):
                    continue
                for record_id in (
                    str(record.get('phase_id') or '').strip(),
                    str(record.get('branch_id') or '').strip(),
                    str(record.get('obligation_id') or '').strip(),
                ):
                    if record_id and record_id not in phase_lookup:
                        phase_lookup[record_id] = record
        return phase_lookup

    @staticmethod
    def _missing_intent_dependency_edges(
        intent_obligations: list[Mapping[str, Any]],
        phase_lookup: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        missing: list[dict[str, Any]] = []
        for obligation in required_intent_obligations(intent_obligations):
            if str(obligation.get('kind') or '').strip().lower() != 'dependency':
                continue
            if obligation.get('execution_dependency_required') is not True:
                continue
            target_phase_id = str(
                obligation.get('target_phase_id') or obligation.get('target_branch_id') or ''
            ).strip()
            source_phase_ids = [
                str(item or '').strip()
                for item in (obligation.get('source_phase_ids') or [])
                if str(item or '').strip()
            ]
            if not target_phase_id or not source_phase_ids:
                continue
            target_record = phase_lookup.get(target_phase_id)
            actual_dependencies = [
                str(item or '').strip()
                for item in ((target_record or {}).get('depends_on') or [])
                if str(item or '').strip()
            ]
            missing_source_phase_ids = [
                source_phase_id
                for source_phase_id in source_phase_ids
                if source_phase_id not in actual_dependencies
            ]
            if not target_record or missing_source_phase_ids:
                missing.append(
                    {
                        'obligation_id': str(obligation.get('obligation_id') or '').strip() or None,
                        'dependency_contract': str(obligation.get('dependency_contract') or '').strip() or None,
                        'target_phase_id': target_phase_id,
                        'target_branch_id': str(obligation.get('target_branch_id') or '').strip() or None,
                        'source_phase_ids': source_phase_ids,
                        'missing_source_phase_ids': missing_source_phase_ids or source_phase_ids,
                        'actual_depends_on': actual_dependencies,
                        'add_dependencies': [
                            {
                                'target_phase_id': target_phase_id,
                                'source_phase_id': source_phase_id,
                                'dependency_contract': obligation.get('dependency_contract'),
                            }
                            for source_phase_id in (missing_source_phase_ids or source_phase_ids)
                        ],
                    }
                )
        return missing

    @staticmethod
    def _intent_graph_has_explicit_promise_ledger(request_phase_graph: Mapping[str, Any]) -> bool:
        for key in ('output_obligations', 'intent_obligations'):
            records = request_phase_graph.get(key)
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                if record.get('required') is False:
                    continue
                capability = normalize_capability(record.get('capability'))
                output_type = str(record.get('output_type') or '').strip().lower()
                kind = str(record.get('kind') or '').strip().lower()
                if capability or output_type or kind:
                    return True
        return False

    @staticmethod
    def _intent_semantic_fit_requested(prompt: str, prompt_intent: Mapping[str, Any]) -> bool:
        prompt_text = str(prompt or '').strip()
        if prompt_text and (
            _INTENT_SEMANTIC_FIT_RE.search(prompt_text)
            or _INTENT_SEMANTIC_QUALITY_RE.search(prompt_text)
        ):
            return True
        cues = [
            str(item or '').strip().lower()
            for item in (
                prompt_intent.get('local_visual_asset_cues')
                if isinstance(prompt_intent.get('local_visual_asset_cues'), list)
                else []
            )
            if str(item or '').strip()
        ]
        return any(
            'fit' in cue
            or 'match' in cue
            or 'zusammen' in cue
            or 'stimmig' in cue
            or 'exakt' in cue
            for cue in cues
        )

    def _build_intent_lens_review(
        self,
        *,
        prompt: str,
        prompt_intent: Mapping[str, Any],
        request_phase_graph: Mapping[str, Any],
        intent_obligations: list[Mapping[str, Any]],
        expected_counts: Mapping[str, int],
        expected_capability_counts: Mapping[str, int],
        existing_structural_gap_count: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        summary = self._intent_obligation_summary(intent_obligations)
        kind_counts = summary.get('kind_counts') if isinstance(summary.get('kind_counts'), dict) else {}
        dependency_contract_counts = (
            summary.get('dependency_contract_counts')
            if isinstance(summary.get('dependency_contract_counts'), dict)
            else {}
        )
        required_count = int(summary.get('required_count') or 0)
        material_expected = bool(
            expected_counts
            or expected_capability_counts
            or prompt_intent.get('requests_text_artifact_output')
            or prompt_intent.get('requests_visual_output')
            or prompt_intent.get('requests_audio_output')
            or prompt_intent.get('requests_speech_to_text_output')
        )
        phase_lookup = self._phase_lookup_for_intent_review(request_phase_graph)
        missing_dependencies = self._missing_intent_dependency_edges(intent_obligations, phase_lookup)
        semantic_fit_requested = self._intent_semantic_fit_requested(prompt, prompt_intent)
        text_media_combo = bool(kind_counts.get('text_artifact') and kind_counts.get('media_artifact'))

        checks: list[dict[str, Any]] = []
        attention_status = 'not_applicable'
        if material_expected:
            explicit_promise_ledger = self._intent_graph_has_explicit_promise_ledger(
                request_phase_graph
            )
            attention_status = (
                'fulfilled'
                if required_count or explicit_promise_ledger or existing_structural_gap_count
                else 'pending'
            )
            if attention_status == 'pending':
                checks.append(
                    {
                        'check_kind': 'intent_attention_review',
                        'status': 'pending',
                        'evidence': 'intent_attention_missing_promise_ledger',
                        'reason': 'current material intent has no normalized intent obligations',
                        'repair_action': RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                        'recovery_action': RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                    }
                )

        commitment_applicable = bool(
            dependency_contract_counts
            or text_media_combo
            or prompt_intent.get('local_visual_asset_requirement')
        )
        commitment_status = 'not_applicable'
        if commitment_applicable:
            commitment_status = 'pending' if missing_dependencies else 'fulfilled'
        for item in missing_dependencies:
            checks.append(
                {
                    'check_kind': 'intent_commitment_review',
                    'obligation_id': item.get('obligation_id') or f"missing-dependency-{item.get('target_phase_id')}",
                    'status': 'pending',
                    'evidence': 'intent_commitment_missing_dependency_edge',
                    'reason': 'current intent requires producer/consumer binding before truthful materialization',
                    'dependency_contract': item.get('dependency_contract'),
                    'target_phase_id': item.get('target_phase_id'),
                    'target_branch_id': item.get('target_branch_id'),
                    'source_phase_ids': item.get('source_phase_ids'),
                    'missing_source_phase_ids': item.get('missing_source_phase_ids'),
                    'actual_depends_on': item.get('actual_depends_on'),
                    'repair_action': 'rebind_artifact_dependency',
                    'recovery_action': 'rebind_artifact_dependency',
                    'add_dependencies': item.get('add_dependencies'),
                }
            )

        aspiration_status = 'not_applicable'
        if semantic_fit_requested and text_media_combo:
            aspiration_status = 'fulfilled'
            checks.append(
                {
                    'check_kind': 'intent_aspiration_review',
                    'status': 'fulfilled',
                    'evidence': 'intent_aspiration_semantic_fit_requested',
                    'reason': 'current intent explicitly asks generated text/media/artifacts to fit together',
                    'semantic_review_required': True,
                    'semantic_review_action': RECOVERY_ACTION_SEMANTIC_REVIEW,
                    'semantic_review_authority': 'global_semantic_closure_review',
                    'semantic_review_criteria': [
                        'whole_current_intent_fit_between_text_media_and_artifacts',
                        'section_text_matches_its_declared_image_or_artifact_role',
                        'local_artifact_set_is_coherent_with_prompt_intent',
                        'explicit_visual_and_tone_constraints_match_prompt',
                    ],
                    'review_criteria': [
                        'runtime_artifacts_exist_before_semantic_fit_review',
                        'whole_current_intent_fit_between_text_media_and_artifacts',
                    ],
                }
            )

        status = 'not_applicable'
        if any(str(item.get('status') or '').strip().lower() in {'pending', 'blocked'} for item in checks):
            status = 'pending'
        elif attention_status != 'not_applicable' or commitment_status != 'not_applicable' or aspiration_status != 'not_applicable':
            status = 'fulfilled'

        review = {
            'kind': 'ollmo.intent_lens_review',
            'status': status,
            'authority': 'closure_review_runtime_projection',
            'policy': 'attention_commitment_aspiration_lenses_apply_to_current_intent_before_freeze',
            'required_intent_obligation_count': required_count,
            'intent_obligation_kinds': [
                key for key, value in sorted(kind_counts.items()) if int(value or 0) > 0
            ],
            'dependency_contract_counts': dependency_contract_counts,
            'semantic_fit_requested': bool(semantic_fit_requested and text_media_combo),
            'attention_review': {
                'kind': 'ollmo.intent_attention_review',
                'status': attention_status,
                'policy': 'current_material_intent_must_have_a_visible_promise_ledger',
            },
            'commitment_review': {
                'kind': 'ollmo.intent_commitment_review',
                'status': commitment_status,
                'policy': 'promised_dependency_and_binding_order_must_be_executable_or_repairable',
                'missing_dependency_count': len(missing_dependencies),
            },
            'aspiration_review': {
                'kind': 'ollmo.intent_aspiration_review',
                'status': aspiration_status,
                'policy': 'semantic_fit_promises_require_whole_turn_review_after_runtime_evidence',
            },
            'check_count': len(checks),
        }
        return self._compact_mapping(review), [
            self._compact_mapping(check) for check in checks
        ]

    def build_intent_graph_adequacy_review(
        self,
        *,
        request_payload: Optional[dict[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> dict[str, Any]:
        prompt = self._current_request_prompt_for_review(request_payload, request_phase_graph)
        prompt_intent = self._merged_prompt_intent_for_review(prompt, request_phase_graph)
        suppress_visual_artifact_execution = bool(
            prompt_intent.get('visual_artifact_execution_suppressed_by_preservation')
        )
        suppress_visual_analysis_execution = bool(
            prompt_intent.get('visual_analysis_execution_suppressed_by_preservation')
        )
        intent_obligations = [
            item for item in (request_phase_graph.get('intent_obligations') or [])
            if isinstance(item, Mapping)
        ]
        required_obligations = required_intent_obligations(intent_obligations)
        required_intent_summary = summarize_required_intent_obligations(required_obligations)
        required_output_counts = dict(
            required_intent_summary.get('material_output_counts') or {}
        )
        expected_counts = self._expected_material_output_counts(
            request_payload=request_payload,
            request_phase_graph=request_phase_graph,
        )
        graph_counts = self._graph_material_output_counts(request_phase_graph)
        reserved_counts = self._reserved_material_output_counts(request_phase_graph)
        for output_type, reserved_count in reserved_counts.items():
            if output_type not in expected_counts:
                continue
            if (
                output_type == 'image'
                and bool(prompt_intent.get('counted_visual_output_obligation'))
                and self._coerce_positive_int(prompt_intent.get('requested_visual_output_count')) > 0
                and not bool(prompt_intent.get('explicit_visual_defer_materialization'))
            ):
                continue
            expected_count = max(0, expected_counts.get(output_type, 0) - reserved_count)
            if expected_count <= graph_counts.get(output_type, 0):
                expected_counts.pop(output_type, None)
            else:
                expected_counts[output_type] = expected_count
        generic_materialization_defer = bool(
            prompt_intent.get('explicit_defer_materialization')
            and not prompt_intent.get('explicit_visual_defer_materialization')
            and not prompt_intent.get('explicit_audio_defer_materialization')
        )
        for output_type, required_count in required_output_counts.items():
            if generic_materialization_defer:
                continue
            if output_type == 'image' and (
                prompt_intent.get('explicit_visual_defer_materialization')
                or suppress_visual_artifact_execution
            ):
                continue
            if output_type == 'audio' and prompt_intent.get('explicit_audio_defer_materialization'):
                continue
            expected_counts[output_type] = max(
                expected_counts.get(output_type, 0),
                self._coerce_positive_int(required_count),
            )
        graph_capability_counts = self._graph_required_capability_counts(request_phase_graph)
        checks: list[dict[str, Any]] = []
        if prompt_intent.get('audio_output_count_exceeds_bound'):
            requested_audio_count = self._coerce_positive_int(
                prompt_intent.get('requested_audio_output_count_raw')
            )
            maximum_audio_count = self._coerce_positive_int(
                prompt_intent.get('requested_audio_output_count_max')
            ) or 6
            checks.append(
                {
                    'check_kind': 'intent_cardinality_guard',
                    'obligation_id': 'blocked-obligation-audio-cardinality',
                    'capability': CAPABILITY_TEXT_TO_SPEECH,
                    'output_type': 'audio',
                    'status': 'blocked',
                    'evidence': 'audio_output_count_exceeds_bound',
                    'reason': (
                        f'requested audio output count {requested_audio_count} exceeds '
                        f'the bounded maximum {maximum_audio_count}'
                    ),
                    'expected_count': requested_audio_count,
                    'actual_count': 0,
                    # This is one contract-clarification obligation, not N safe
                    # materialization attempts.
                    'missing_count': 1,
                    'requested_audio_output_count_raw': requested_audio_count,
                    'requested_audio_output_count_max': maximum_audio_count,
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    'recovery_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    'blocked_by_branch_contract': True,
                    'materialization_blocked': True,
                    'needs_external_input': True,
                }
            )
        for output_type, expected_count in sorted(expected_counts.items()):
            actual_count = graph_counts.get(output_type, 0)
            if actual_count >= expected_count:
                continue
            capability = (
                CAPABILITY_IMAGE_GENERATION
                if output_type == 'image'
                else CAPABILITY_TEXT_TO_SPEECH
                if output_type == 'audio'
                else None
            )
            checks.append(
                {
                    'check_kind': 'intent_graph_adequacy',
                    'obligation_id': f'missing-obligation-{output_type}',
                    'capability': capability,
                    'output_type': output_type,
                    'status': 'pending',
                    'evidence': 'intent_graph_adequacy_missing_output_obligation',
                    'expected_count': expected_count,
                    'actual_count': actual_count,
                    'missing_count': expected_count - actual_count,
                }
            )

        downstream_capabilities = {
            normalize_capability(item)
            for item in (prompt_intent.get('downstream_follow_up_capabilities') or [])
            if normalize_capability(item)
        }
        # Multiple intent-ledger records may describe different semantic roles
        # of the same executable branch (for example, one HTML branch can be
        # both the text artifact and the evidence join). Count execution
        # identities, not ledger rows, when comparing against the graph.
        expected_capability_counts: dict[str, int] = {}
        seen_capability_executions: set[tuple[str, str]] = set()
        for obligation in required_obligations:
            capability = normalize_capability(obligation.get('capability'))
            if not capability:
                continue
            execution_identity = str(
                obligation.get('branch_id')
                or obligation.get('phase_id')
                or ''
            ).strip()
            if execution_identity:
                identity_key = (capability, execution_identity)
                if identity_key in seen_capability_executions:
                    continue
                seen_capability_executions.add(identity_key)
            try:
                obligation_count = int(obligation.get('count') or 1)
            except (TypeError, ValueError):
                obligation_count = 1
            expected_capability_counts[capability] = (
                expected_capability_counts.get(capability, 0)
                + max(1, obligation_count)
            )
        if suppress_visual_artifact_execution:
            expected_capability_counts.pop(CAPABILITY_IMAGE_GENERATION, None)
        if suppress_visual_analysis_execution:
            expected_capability_counts.pop(CAPABILITY_VISION_ANALYSIS, None)
        if (
            prompt_intent.get('requests_speech_to_text_output')
            or normalize_capability(prompt_intent.get('primary_capability')) == CAPABILITY_SPEECH_TO_TEXT
            or CAPABILITY_SPEECH_TO_TEXT in downstream_capabilities
        ):
            expected_capability_counts[CAPABILITY_SPEECH_TO_TEXT] = max(
                expected_capability_counts.get(CAPABILITY_SPEECH_TO_TEXT, 0),
                1,
            )
        for capability, expected_count in sorted(expected_capability_counts.items()):
            actual_count = graph_capability_counts.get(capability, 0)
            if actual_count >= expected_count:
                continue
            checks.append(
                {
                    'check_kind': 'intent_graph_adequacy',
                    'obligation_id': f'missing-obligation-{capability}',
                    'capability': capability,
                    'output_type': self.artifact_type_for_capability(capability),
                    'status': 'pending',
                    'evidence': 'intent_graph_adequacy_missing_capability_obligation',
                    'expected_count': expected_count,
                    'actual_count': actual_count,
                    'missing_count': expected_count - actual_count,
                }
            )

        required_intent_obligation_count = int(
            required_intent_summary.get('required_count') or 0
        )
        phase_lookup: dict[str, Mapping[str, Any]] = {}
        for collection_key in ('phases', 'downstream_branches'):
            collection = request_phase_graph.get(collection_key)
            if not isinstance(collection, list):
                continue
            for record in collection:
                if not isinstance(record, Mapping):
                    continue
                for record_id in (
                    str(record.get('phase_id') or '').strip(),
                    str(record.get('branch_id') or '').strip(),
                    str(record.get('obligation_id') or '').strip(),
                ):
                    if record_id and record_id not in phase_lookup:
                        phase_lookup[record_id] = record

        for obligation in required_obligations:
            kind = str(obligation.get('kind') or '').strip().lower()
            obligation_id = str(obligation.get('obligation_id') or '').strip()
            if kind == 'text_artifact':
                target_name = str(obligation.get('target_name') or '').strip().lower()
                target_extension = str(obligation.get('target_extension') or '').strip().lower()
                if not target_name or not target_extension:
                    continue
                matched = False
                for record in phase_lookup.values():
                    if (
                        str(record.get('text_artifact_source_name') or '').strip().lower() == target_name
                        and str(record.get('text_artifact_extension') or '').strip().lower() == target_extension
                    ):
                        matched = True
                        break
                if not matched:
                    checks.append(
                        {
                            'check_kind': 'intent_graph_adequacy',
                            'obligation_id': obligation_id or f'missing-text-artifact-{target_name}-{target_extension}',
                            'intent_obligation_id': obligation_id,
                            'capability': CAPABILITY_CHAT,
                            'output_type': 'text',
                            'status': 'pending',
                            'evidence': 'intent_graph_adequacy_missing_text_artifact_obligation',
                            'target_name': target_name,
                            'target_extension': target_extension,
                            'repair_action': 'repair_missing_materialization_contract',
                        }
                    )
                continue
            if kind != 'dependency' or not obligation.get('execution_dependency_required'):
                continue
            target_phase_id = str(
                obligation.get('target_phase_id') or obligation.get('target_branch_id') or ''
            ).strip()
            source_phase_ids = [
                str(item or '').strip()
                for item in (obligation.get('source_phase_ids') or [])
                if str(item or '').strip()
            ]
            if not target_phase_id or not source_phase_ids:
                continue
            target_record = phase_lookup.get(target_phase_id)
            actual_dependencies = [
                str(item or '').strip()
                for item in ((target_record or {}).get('depends_on') or [])
                if str(item or '').strip()
            ]
            missing_dependencies = [
                source_phase_id
                for source_phase_id in source_phase_ids
                if source_phase_id not in actual_dependencies
            ]
            if not target_record or missing_dependencies:
                checks.append(
                    {
                        'check_kind': 'intent_graph_adequacy',
                        'obligation_id': obligation_id or f'missing-dependency-{target_phase_id}',
                        'intent_obligation_id': obligation_id,
                        'status': 'pending',
                        'evidence': 'intent_graph_adequacy_missing_dependency_edge',
                        'dependency_contract': obligation.get('dependency_contract'),
                        'target_phase_id': target_phase_id,
                        'source_phase_ids': source_phase_ids,
                        'missing_source_phase_ids': missing_dependencies or source_phase_ids,
                        'actual_depends_on': actual_dependencies,
                        'repair_action': 'rebind_artifact_dependency',
                        'add_dependencies': [
                            {
                                'target_phase_id': target_phase_id,
                                'source_phase_id': source_phase_id,
                                'dependency_contract': obligation.get('dependency_contract'),
                            }
                            for source_phase_id in (missing_dependencies or source_phase_ids)
                        ],
                    }
                )

        intent_lens_review, intent_lens_checks = self._build_intent_lens_review(
            prompt=prompt,
            prompt_intent=prompt_intent,
            request_phase_graph=request_phase_graph,
            intent_obligations=intent_obligations,
            expected_counts=expected_counts,
            expected_capability_counts=expected_capability_counts,
            existing_structural_gap_count=len(checks),
        )
        checks.extend(intent_lens_checks)

        open_checks = [
            item
            for item in checks
            if str(item.get('status') or '').strip().lower() in {
                'pending',
                'planned',
                'active',
                'deferred',
                'blocked',
            }
        ]
        if open_checks:
            status = 'pending'
            reason = 'graph/IR does not fully cover current user intent'
        elif (
            expected_counts
            or expected_capability_counts
            or required_intent_obligation_count
            or str(intent_lens_review.get('status') or '').strip().lower() != 'not_applicable'
        ):
            status = 'fulfilled'
            reason = 'graph/IR covers current materialization intent'
        else:
            status = 'not_applicable'
            reason = 'current intent has no material output requirement'
        payload = {
            'kind': 'ollmo.intent_graph_adequacy_review',
            'status': status,
            'reason': reason,
            'expected_output_counts': expected_counts,
            'graph_output_counts': graph_counts,
            'expected_capability_counts': expected_capability_counts,
            'graph_capability_counts': graph_capability_counts,
            'intent_obligation_count': len(intent_obligations),
            'required_intent_obligation_count': required_intent_obligation_count,
            'required_intent_capabilities': list(
                required_intent_summary.get('capabilities') or []
            ),
            'intent_obligation_kinds': list(
                dict.fromkeys(
                    str(item.get('kind') or '').strip()
                    for item in intent_obligations
                    if str(item.get('kind') or '').strip()
                )
            ),
            'checks': checks,
        }
        if intent_lens_review:
            payload['intent_lens_review'] = intent_lens_review
        return payload

    @staticmethod
    def _json_for_semantic_review_prompt(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _late_fill_branch_record(
        artifact_payload: Optional[dict[str, Any]],
        branch_id: str,
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(artifact_payload, Mapping) or not branch_id:
            return {}, ''
        late_fill = artifact_payload.get('late_fill') if isinstance(artifact_payload.get('late_fill'), Mapping) else {}
        for key, status in (
            ('completed_branches', 'fulfilled'),
            ('failed_branches', 'blocked'),
            ('pending_branches', 'pending'),
        ):
            for item in late_fill.get(key) or []:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get('branch_id') or item.get('phase_id') or '').strip() == branch_id:
                    return dict(item), status
        return {}, ''

    @classmethod
    def _late_fill_branch_status(cls, artifact_payload: Optional[dict[str, Any]], branch_id: str) -> str:
        _, status = cls._late_fill_branch_record(artifact_payload, branch_id)
        return status

    @classmethod
    def _semantic_review_verdict_from_late_fill(
        cls,
        artifact_payload: Optional[dict[str, Any]],
        branch_id: str,
        *,
        review_id: str = 'global-semantic-closure-review',
    ) -> dict[str, Any]:
        record, status = cls._late_fill_branch_record(artifact_payload, branch_id)
        if not record or status != 'fulfilled':
            return {}
        raw_verdict = record.get('semantic_review_verdict')
        if isinstance(raw_verdict, Mapping):
            verdict = semantic_review_verdict_from_text(
                raw_verdict,
                review_id=review_id,
                branch_id=branch_id,
                phase_id=str(record.get('phase_id') or '').strip(),
            )
        else:
            result_text = ''
            for key in ('result_text', 'output_text', 'text', 'transcript', 'content'):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    result_text = value
                    break
            verdict = semantic_review_verdict_from_text(
                result_text,
                review_id=review_id,
                branch_id=branch_id,
                phase_id=str(record.get('phase_id') or '').strip(),
            )
        verdict['completed_branch_status'] = status
        return cls._compact_mapping(verdict)

    def _global_semantic_evidence_payload(
        self,
        *,
        artifact_payload: Optional[dict[str, Any]],
        checks: Sequence[Mapping[str, Any]],
        intent_graph_adequacy: Mapping[str, Any],
        decision_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = artifact_payload if isinstance(artifact_payload, dict) else {}
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        artifacts = self._hook('build_canonical_response_artifacts')(payload) if payload else []
        artifact_refs: list[dict[str, Any]] = []
        for item in artifacts or []:
            if not isinstance(item, Mapping):
                continue
            artifact_refs.append(
                self._compact_mapping(
                    {
                        'type': item.get('type'),
                        'path': item.get('path') or item.get('source_path'),
                        'name': item.get('name'),
                        'ref': item.get('ref') or item.get('artifact_ref'),
                        'branch_id': item.get('branch_id'),
                        'phase_id': item.get('phase_id'),
                    }
                )
            )
        return self._compact_mapping(
            {
                'artifact_refs': artifact_refs,
                'late_fill': {
                    key: late_fill.get(key)
                    for key in (
                        'status',
                        'completed_capabilities',
                        'failed_capabilities',
                        'completed_branches',
                        'failed_branches',
                        'pending_branches',
                    )
                    if late_fill.get(key) not in (None, '', [], {})
                },
                'closure_checks': [
                    self._compact_mapping(
                        {
                            key: check.get(key)
                            for key in (
                                'check_kind',
                                'status',
                                'evidence',
                                'reason',
                                'obligation_id',
                                'phase_id',
                                'branch_id',
                                'capability',
                                'output_type',
                                'role',
                                'depends_on',
                                'review_criteria',
                                'review_criteria_status',
                                'semantic_review_required',
                                'semantic_review_criteria',
                                'repair_action',
                            )
                        }
                    )
                    for check in checks
                    if isinstance(check, Mapping)
                ],
                'intent_graph_adequacy': {
                    key: intent_graph_adequacy.get(key)
                    for key in ('status', 'reason', 'expected_output_counts', 'graph_output_counts', 'checks')
                    if isinstance(intent_graph_adequacy, Mapping) and intent_graph_adequacy.get(key) not in (None, '', [], {})
                },
                'semantic_decision_review': (
                    decision_contract.get('semantic_decision_review')
                    if isinstance(decision_contract, Mapping)
                    else None
                ),
                'controlled_attention_review': (
                    decision_contract.get('controlled_attention_review')
                    if isinstance(decision_contract, Mapping)
                    else None
                ),
                'aspiration_review': (
                    decision_contract.get('aspiration_review')
                    if isinstance(decision_contract, Mapping)
                    else None
                ),
                'commitment_review': (
                    decision_contract.get('commitment_review')
                    if isinstance(decision_contract, Mapping)
                    else None
                ),
                'semantic_quality_review': (
                    decision_contract.get('semantic_quality_review')
                    if isinstance(decision_contract, Mapping)
                    else None
                ),
            }
        )

    def _global_semantic_review_instruction(
        self,
        *,
        prompt: str,
        output_text: str,
        evidence_payload: Mapping[str, Any],
    ) -> str:
        return (
            'Run a whole-turn semantic closure review for the current Ollmo response.\n'
            '\n'
            'Authority boundary:\n'
            '- You are a semantic reviewer, not runtime truth.\n'
            '- Do not claim artifacts exist unless runtime evidence below proves them.\n'
            '- Decide whether local branch outputs fulfill their role in the whole user intent.\n'
            '- Preserve the current intent; do not invent a different request.\n'
            '- If a block exists, propose the right-sized verified transition that resolves the block without under-scoping the work.\n'
            '\n'
            'Return exactly one JSON object and no markdown, no prose outside JSON, no chain-of-thought.\n'
            'Required schema:\n'
            '{\n'
            '  "kind": "ollmo.semantic_review_verdict",\n'
            '  "verdict": "passed | failed | uncertain",\n'
            '  "overall_status": "fulfilled | blocked | pending",\n'
            '  "whole_intent_fit": "short evidence-grounded explanation",\n'
            '  "criterion_results": [\n'
            '    {"criterion": "criterion text", "status": "passed | failed | uncertain", '
            '"reason": "short reason", "evidence_refs": ["runtime evidence ref"]}\n'
            '  ],\n'
            '  "evidence_refs": ["runtime evidence ref"],\n'
            '  "defects": ["missing or wrong work, or empty list"],\n'
            '  "confidence": 0.0,\n'
            '  "recommended_transition": "truthful_freeze | semantic_review | repair_dependency_chain | '
            'repair_branch_contract | rebuild_from_promoted_obligations | waive_with_evidence | '
            'supersede_with_replacement_truth | clarify | manual_review",\n'
            '  "authority_boundary": "advisory_review_only_runtime_contracts_closure_decide_truth"\n'
            '}\n'
            'Use verdict "passed" only when runtime evidence proves all relevant semantic criteria and whole-intent fit.\n'
            'Use verdict "failed" when evidence proves missing or wrong work.\n'
            'Use verdict "uncertain" when evidence is insufficient or ambiguous.\n'
            '\n'
            f'Current user intent:\n{prompt}\n\n'
            f'Current response text:\n{output_text}\n\n'
            'Runtime evidence:\n'
            f'{self._json_for_semantic_review_prompt(evidence_payload)}'
        )

    @staticmethod
    def _semantic_review_branch_slug(value: Any) -> str:
        token = re.sub(r'[^a-z0-9_]+', '-', str(value or '').strip().lower())
        token = re.sub(r'-+', '-', token).strip('-')
        return token or 'unknown-branch'

    @classmethod
    def _branch_semantic_review_identity(cls, check: Mapping[str, Any]) -> tuple[str, str, str]:
        source_branch_id = str(check.get('branch_id') or check.get('phase_id') or check.get('obligation_id') or '').strip()
        source_phase_id = str(check.get('phase_id') or source_branch_id or '').strip()
        slug = cls._semantic_review_branch_slug(source_branch_id or source_phase_id or check.get('obligation_id'))
        return (
            f'branch-semantic-review-{slug}',
            f'phase-semantic-review-{slug}',
            f'obligation-semantic-review-{slug}',
        )

    @classmethod
    def _check_is_branch_semantic_review_target(cls, check: Mapping[str, Any]) -> bool:
        if not isinstance(check, Mapping):
            return False
        if check.get('semantic_review_required') is not True:
            return False
        if str(check.get('check_kind') or '').strip() in {
            'global_semantic_closure',
            'branch_semantic_review',
            'intent_graph_adequacy',
        }:
            return False
        if str(check.get('status') or '').strip().lower() != 'fulfilled':
            return False
        review_branch_id = str(check.get('branch_semantic_review_branch_id') or '').strip()
        source_branch_id = str(check.get('branch_id') or check.get('phase_id') or '').strip()
        criteria = cls._string_list(check.get('semantic_review_criteria')) or cls._string_list(check.get('review_criteria'))
        if not any(not cls._review_criterion_is_deterministic(item) for item in criteria):
            return False
        return bool(source_branch_id) and review_branch_id != source_branch_id

    def _branch_semantic_evidence_payload(
        self,
        *,
        artifact_payload: Optional[dict[str, Any]],
        check: Mapping[str, Any],
        output_text: str,
    ) -> dict[str, Any]:
        payload = artifact_payload if isinstance(artifact_payload, dict) else {}
        source_branch_id = str(check.get('branch_id') or check.get('phase_id') or '').strip()
        source_phase_id = str(check.get('phase_id') or source_branch_id or '').strip()
        branch_record, branch_status = self._late_fill_branch_record(payload, source_branch_id)
        artifacts = self._hook('build_canonical_response_artifacts')(payload) if payload else []
        artifact_refs: list[dict[str, Any]] = []
        for item in artifacts or []:
            if not isinstance(item, Mapping):
                continue
            item_branch = str(item.get('branch_id') or item.get('phase_id') or '').strip()
            if item_branch and item_branch not in {source_branch_id, source_phase_id}:
                continue
            artifact_refs.append(
                self._compact_mapping(
                    {
                        'type': item.get('type'),
                        'path': item.get('path') or item.get('source_path'),
                        'name': item.get('name'),
                        'ref': item.get('ref') or item.get('artifact_ref'),
                        'branch_id': item.get('branch_id'),
                        'phase_id': item.get('phase_id'),
                    }
                )
            )
        branch_output_text = ''
        for key in ('result_text', 'output_text', 'content_payload', 'text', 'transcript'):
            value = branch_record.get(key) if isinstance(branch_record, Mapping) else None
            if isinstance(value, str) and value.strip():
                branch_output_text = value.strip()
                break
        if not branch_output_text and str(check.get('evidence') or '').strip() == 'current_phase_output_text':
            branch_output_text = str(output_text or '').strip()
        return self._compact_mapping(
            {
                'review_scope': 'branch_only',
                'source_branch_id': source_branch_id,
                'source_phase_id': source_phase_id,
                'branch_late_fill_status': branch_status or None,
                'branch_record': {
                    key: branch_record.get(key)
                    for key in (
                        'branch_id',
                        'phase_id',
                        'capability',
                        'output_type',
                        'result_text',
                        'content_payload_source',
                        'saved_text_path',
                        'saved_audio_path',
                        'saved_image_path',
                    )
                    if isinstance(branch_record, Mapping) and branch_record.get(key) not in (None, '', [], {})
                },
                'branch_output_text': branch_output_text or None,
                'artifact_refs': artifact_refs,
                'branch_check': {
                    key: check.get(key)
                    for key in (
                        'check_kind',
                        'status',
                        'evidence',
                        'reason',
                        'obligation_id',
                        'phase_id',
                        'branch_id',
                        'capability',
                        'output_type',
                        'role',
                        'depends_on',
                        'review_criteria',
                        'review_criteria_status',
                        'semantic_review_required',
                        'semantic_review_criteria',
                        'semantic_review_lens',
                        'success_definition',
                        'failure_modes',
                        'semantic_lens_evidence_requirements',
                        'semantic_review_lens_contract',
                        'execution_contract',
                        'input_refs',
                        'content_payload_source',
                    )
                    if check.get(key) not in (None, '', [], {})
                },
            }
        )

    def _branch_semantic_review_instruction(
        self,
        *,
        prompt: str,
        check: Mapping[str, Any],
        evidence_payload: Mapping[str, Any],
    ) -> str:
        return (
            'Run a branch-local semantic review for the current Ollmo response graph.\n'
            '\n'
            'Authority boundary:\n'
            '- You are a semantic reviewer for one branch, not runtime truth.\n'
            '- Review only the branch named in the evidence payload.\n'
            '- Use the current user intent only as bounded context for that branch role.\n'
            '- Do not review the whole turn and do not invent different work.\n'
            '- Do not claim artifacts exist unless branch evidence proves them.\n'
            '- If the branch is blocked, propose the right-sized verified transition that resolves this branch without under-scoping the work.\n'
            '\n'
            'Return exactly one JSON object and no markdown, no prose outside JSON, no chain-of-thought.\n'
            'Required schema:\n'
            '{\n'
            '  "kind": "ollmo.semantic_review_verdict",\n'
            '  "verdict": "passed | failed | uncertain",\n'
            '  "overall_status": "fulfilled | blocked | pending",\n'
            '  "whole_intent_fit": "short explanation of whether this branch fulfills its declared role",\n'
            '  "criterion_results": [\n'
            '    {"criterion": "criterion text", "status": "passed | failed | uncertain", '
            '"reason": "short reason", "evidence_refs": ["branch evidence ref"]}\n'
            '  ],\n'
            '  "evidence_refs": ["branch evidence ref"],\n'
            '  "defects": ["missing or wrong branch work, or empty list"],\n'
            '  "confidence": 0.0,\n'
            '  "recommended_transition": "truthful_freeze | semantic_review | repair_dependency_chain | '
            'repair_branch_contract | rebuild_from_promoted_obligations | waive_with_evidence | '
            'supersede_with_replacement_truth | clarify | manual_review",\n'
            '  "authority_boundary": "advisory_branch_review_only_runtime_contracts_closure_decide_truth"\n'
            '}\n'
            'Use verdict "passed" only when branch evidence proves all branch-local criteria.\n'
            'Use verdict "failed" when evidence proves missing or wrong branch work.\n'
            'Use verdict "uncertain" when branch evidence is insufficient or ambiguous.\n'
            'Use the provided semantic_review_lens and success_definition as the review posture for this branch. '
            'They are advisory only, but they define what kind of success should be checked and which failure modes matter.\n'
            '\n'
            f'Current user intent for context only:\n{prompt}\n\n'
            'Branch under review:\n'
            f'{self._json_for_semantic_review_prompt(check)}\n\n'
            'Branch runtime evidence:\n'
            f'{self._json_for_semantic_review_prompt(evidence_payload)}'
        )

    def _apply_branch_semantic_verdict_to_check(
        self,
        check: Mapping[str, Any],
        *,
        artifact_payload: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        updated = dict(check)
        if not self._check_is_branch_semantic_review_target(updated):
            return updated
        review_branch_id, review_phase_id, _ = self._branch_semantic_review_identity(updated)
        verdict = self._semantic_review_verdict_from_late_fill(
            artifact_payload,
            review_branch_id,
            review_id=f'branch-semantic-review:{str(updated.get("branch_id") or updated.get("phase_id") or "").strip()}',
        )
        updated['branch_semantic_review_branch_id'] = review_branch_id
        updated['branch_semantic_review_phase_id'] = review_phase_id
        updated['branch_semantic_review_source_branch_id'] = str(updated.get('branch_id') or '').strip() or None
        updated['branch_semantic_review_source_phase_id'] = str(updated.get('phase_id') or '').strip() or None
        if not verdict:
            updated['branch_semantic_review_status'] = 'pending'
            return updated
        updated['semantic_review_verdict'] = verdict
        updated['semantic_review_verdict_status'] = verdict.get('status')
        updated['semantic_review_recommended_transition'] = verdict.get('recommended_transition')
        updated['branch_semantic_review'] = {
            'kind': 'ollmo.branch_semantic_review',
            'status': verdict.get('status'),
            'review_branch_id': review_branch_id,
            'review_phase_id': review_phase_id,
            'source_branch_id': updated.get('branch_id'),
            'source_phase_id': updated.get('phase_id'),
            'semantic_review_verdict': verdict,
            'authority': 'advisory_branch_review_runtime_closure_decides_truth',
        }
        updated['branch_semantic_review_status'] = verdict.get('status')
        transition_action = self._recovery_action_for_semantic_review_transition(verdict.get('recommended_transition'))
        if str(verdict.get('verdict') or '').strip() == 'passed':
            updated['semantic_review_required'] = False
            updated['review_criteria_status'] = 'passed_semantic_review'
            updated['semantic_review_action'] = None
            updated['repair_action'] = None
            updated['recovery_action'] = None
            updated['repair_action_reason'] = None
            return updated
        updated['status'] = 'blocked' if str(verdict.get('status') or '').strip() == 'blocked' else 'pending'
        updated['evidence'] = 'branch_semantic_review_verdict'
        updated['semantic_review_required'] = transition_action == RECOVERY_ACTION_SEMANTIC_REVIEW
        updated['semantic_review_action'] = transition_action or RECOVERY_ACTION_MANUAL_REVIEW
        updated['repair_required'] = True
        updated['repair_action'] = transition_action or RECOVERY_ACTION_MANUAL_REVIEW
        updated['recovery_action'] = transition_action or RECOVERY_ACTION_MANUAL_REVIEW
        updated['repair_action_reason'] = (
            str(verdict.get('whole_intent_fit') or verdict.get('reason') or '').strip()
            or 'branch semantic review did not pass'
        )
        updated['branch_semantic_review_reason'] = updated['repair_action_reason']
        if updated['repair_action'] == RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN:
            updated['blocked_by_dependency_input'] = True
        if updated['repair_action'] == RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT:
            updated['blocked_by_branch_contract'] = True
        return updated

    def _branch_semantic_review_checks(
        self,
        *,
        checks: Sequence[Mapping[str, Any]],
        request_payload: Optional[dict[str, Any]],
        request_phase_graph: Mapping[str, Any],
        artifact_payload: Optional[dict[str, Any]],
        output_text: str,
    ) -> list[dict[str, Any]]:
        prompt = self._current_request_prompt_for_review(request_payload, request_phase_graph)
        review_checks: list[dict[str, Any]] = []
        existing_review_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in checks
            if isinstance(item, Mapping)
            and str(item.get('check_kind') or '').strip() == 'branch_semantic_review'
            and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        for source_check in checks:
            if not isinstance(source_check, Mapping):
                continue
            if not self._check_is_branch_semantic_review_target(source_check):
                continue
            if isinstance(source_check.get('semantic_review_verdict'), Mapping):
                continue
            review_branch_id, review_phase_id, review_obligation_id = self._branch_semantic_review_identity(source_check)
            if review_branch_id in existing_review_branch_ids:
                continue
            evidence_payload = self._branch_semantic_evidence_payload(
                artifact_payload=artifact_payload,
                check=source_check,
                output_text=output_text,
            )
            content_payload = self._branch_semantic_review_instruction(
                prompt=prompt,
                check=source_check,
                evidence_payload=evidence_payload,
            )
            execution_contract = {
                'kind': 'ollmo.execution_contract',
                'branch_id': review_branch_id,
                'phase_id': review_phase_id,
                'capability': CAPABILITY_CHAT,
                'output_type': 'text',
                'workload_task_ref': {
                    'task_id': f'task-{review_phase_id}',
                    'phase_id': review_phase_id,
                    'branch_id': review_branch_id,
                },
                'output_obligation_ref': {
                    'obligation_id': review_obligation_id,
                    'phase_id': review_phase_id,
                    'branch_id': review_branch_id,
                    'output_type': 'text',
                },
                'output_contract': {
                    'output_type': 'text',
                    'required': True,
                    'fulfillment_policy': 'branch_semantic_review_verdict_required',
                    'semantic_review_lens': source_check.get('semantic_review_lens'),
                    'success_definition': source_check.get('success_definition'),
                },
                'input_refs': [
                    {'kind': 'branch_under_review', 'ref': str(source_check.get('branch_id') or source_check.get('phase_id') or '')},
                    {'kind': 'branch_review_criteria', 'ref': 'source_check.review_criteria'},
                    {'kind': 'runtime_evidence', 'ref': 'source_branch_evidence'},
                ],
            }
            review_checks.append(
                self._compact_mapping(
                    {
                        'check_kind': 'branch_semantic_review',
                        'status': 'pending',
                        'evidence': 'branch_semantic_review_required',
                        'reason': 'branch has qualitative review criteria that require branch-local semantic verdict before branch freeze',
                        'obligation_id': review_obligation_id,
                        'phase_id': review_phase_id,
                        'branch_id': review_branch_id,
                        'capability': CAPABILITY_CHAT,
                        'output_type': 'text',
                        'role': 'branch_semantic_review_output',
                        'depends_on': [str(source_check.get('branch_id') or source_check.get('phase_id') or '').strip()],
                        'repair_action': RECOVERY_ACTION_SEMANTIC_REVIEW,
                        'recovery_action': RECOVERY_ACTION_SEMANTIC_REVIEW,
                        'repair_action_reason': 'branch-local qualitative criteria need semantic review verdict',
                        'semantic_review_required': True,
                        'semantic_review_action': RECOVERY_ACTION_SEMANTIC_REVIEW,
                        'semantic_review_authority': 'closure_promoted_branch_semantic_review',
                        'semantic_review_criteria': source_check.get('semantic_review_criteria') or source_check.get('review_criteria'),
                        'semantic_review_lens': source_check.get('semantic_review_lens'),
                        'success_definition': source_check.get('success_definition'),
                        'failure_modes': source_check.get('failure_modes'),
                        'semantic_lens_evidence_requirements': source_check.get('semantic_lens_evidence_requirements') or source_check.get('evidence_requirements'),
                        'semantic_review_lens_contract': source_check.get('semantic_review_lens_contract'),
                        'review_criteria': [
                            'runtime_text_exists_when_fulfilled',
                            *(self._string_list(source_check.get('semantic_review_criteria')) or self._string_list(source_check.get('review_criteria'))),
                        ],
                        'branch_semantic_review': {
                            'kind': 'ollmo.branch_semantic_review',
                            'status': 'pending',
                            'review_branch_id': review_branch_id,
                            'review_phase_id': review_phase_id,
                            'source_branch_id': source_check.get('branch_id'),
                            'source_phase_id': source_check.get('phase_id'),
                            'authority': 'advisory_branch_review_runtime_closure_decides_truth',
                        },
                        'branch_semantic_review_branch_id': review_branch_id,
                        'branch_semantic_review_phase_id': review_phase_id,
                        'branch_semantic_review_status': 'pending',
                        'branch_semantic_review_source_branch_id': source_check.get('branch_id'),
                        'branch_semantic_review_source_phase_id': source_check.get('phase_id'),
                        'content_payload': content_payload,
                        'content_payload_source': f'branch_semantic_review:{str(source_check.get("branch_id") or source_check.get("phase_id") or "").strip()}',
                        'stage_direction': 'run_branch_semantic_review',
                        'execution_contract': execution_contract,
                        'input_refs': execution_contract['input_refs'],
                    }
                )
            )
        return review_checks

    @staticmethod
    def _recovery_action_for_semantic_review_transition(value: Any) -> str:
        transition = str(value or '').strip().lower()
        if transition == 'truthful_freeze':
            return ''
        if transition in {'clarify', 'waive_with_evidence', 'supersede_with_replacement_truth'}:
            return RECOVERY_ACTION_MANUAL_REVIEW
        return normalize_recovery_suggested_action(transition, default=RECOVERY_ACTION_SEMANTIC_REVIEW)

    @classmethod
    def _global_semantic_verdict_proposal(cls, verdict: Mapping[str, Any]) -> dict[str, Any]:
        transition = str(verdict.get('recommended_transition') or '').strip().lower()
        action = cls._recovery_action_for_semantic_review_transition(transition)
        defects = verdict.get('defects') if isinstance(verdict.get('defects'), list) else []
        reason = (
            str(verdict.get('whole_intent_fit') or '').strip()
            or str(verdict.get('reason') or '').strip()
            or 'semantic review verdict did not pass'
        )
        if defects:
            reason = f'{reason}; defects: {", ".join(str(item) for item in defects)}'
        return cls._compact_mapping(
            {
                'kind': 'ollmo.global_semantic_closure_proposal',
                'proposal_id': f"global-semantic-closure-verdict-{str(verdict.get('verdict') or 'uncertain').strip()}",
                'status': 'advisory',
                'authority': 'advisory_verdict_until_closure_confirms_transition',
                'decision_action': action or None,
                'recommended_transition': transition or None,
                'confidence': verdict.get('confidence'),
                'reason': reason,
                'allowed_transitions': [
                    RECOVERY_ACTION_SEMANTIC_REVIEW,
                    RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                    RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                    'clarify',
                    'waive_with_evidence',
                    'supersede_with_replacement_truth',
                    'manual_review',
                    'truthful_freeze_after_review',
                ],
                'evidence_refs': verdict.get('evidence_refs'),
                'semantic_review_verdict': verdict,
            }
        )

    def build_global_semantic_closure_review(
        self,
        *,
        output_text: str,
        request_payload: Optional[dict[str, Any]],
        request_phase_graph: Mapping[str, Any],
        artifact_payload: Optional[dict[str, Any]],
        checks: Sequence[Mapping[str, Any]],
        intent_graph_adequacy: Mapping[str, Any],
        decision_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        prompt = self._current_request_prompt_for_review(request_payload, request_phase_graph)
        def _check_needs_global_semantic_review(check: Mapping[str, Any]) -> bool:
            if check.get('semantic_review_required') is not True:
                return False
            if str(check.get('review_criteria_status') or '').strip().lower() == 'semantic_review_required':
                return True
            criteria = self._string_list(check.get('semantic_review_criteria')) or self._string_list(check.get('review_criteria'))
            return any(not self._review_criterion_is_deterministic(item) for item in criteria)

        semantic_checks = [
            check for check in checks
            if isinstance(check, Mapping) and _check_needs_global_semantic_review(check)
        ]
        semantic_decision_review = (
            decision_contract.get('semantic_decision_review')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('semantic_decision_review'), Mapping)
            else {}
        )
        semantic_decision_proposals = (
            semantic_decision_review.get('proposals')
            if isinstance(semantic_decision_review.get('proposals'), list)
            else []
        )
        semantic_quality_review = (
            decision_contract.get('semantic_quality_review')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('semantic_quality_review'), Mapping)
            else {}
        )
        semantic_quality_contracts = (
            semantic_quality_review.get('contracts')
            if isinstance(semantic_quality_review.get('contracts'), list)
            else []
        )
        controlled_attention_review = (
            decision_contract.get('controlled_attention_review')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('controlled_attention_review'), Mapping)
            else {}
        )
        aspiration_review = (
            decision_contract.get('aspiration_review')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('aspiration_review'), Mapping)
            else {}
        )
        commitment_review = (
            decision_contract.get('commitment_review')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('commitment_review'), Mapping)
            else {}
        )
        semantic_quality_required = any(
            isinstance(contract, Mapping)
            and any(
                not self._review_criterion_is_deterministic(item)
                for item in self._string_list(contract.get('review_criteria'))
            )
            for contract in semantic_quality_contracts
        )
        completed_status = self._late_fill_branch_status(artifact_payload, 'branch-global-semantic-closure-review')
        completed_semantic_review_verdict = self._semantic_review_verdict_from_late_fill(
            artifact_payload,
            'branch-global-semantic-closure-review',
            review_id='global-semantic-closure-review',
        ) if completed_status == 'fulfilled' else {}
        structural_pending = (
            isinstance(intent_graph_adequacy, Mapping)
            and str(intent_graph_adequacy.get('status') or '').strip().lower() == 'pending'
        )
        semantic_pending = bool(
            semantic_checks
            or semantic_quality_required
        )
        local_unresolved = any(
            isinstance(check, Mapping)
            and str(check.get('status') or '').strip().lower() in {'pending', 'planned', 'active', 'deferred', 'blocked'}
            and str(check.get('check_kind') or '').strip() not in {
                'intent_graph_adequacy',
                'global_semantic_closure',
            }
            for check in checks
        )
        if completed_status == 'fulfilled' and str(completed_semantic_review_verdict.get('verdict') or '').strip() == 'passed':
            return self._compact_mapping(
                {
                    'kind': 'ollmo.global_semantic_closure_review',
                    'status': 'fulfilled',
                    'authority': 'advisory_review_output_runtime_evidence',
                    'policy': 'whole_turn_semantic_fit_checked_after_runtime_materialization',
                    'reason': 'global semantic review branch returned a passed verdict',
                    'proposal_count': 0,
                    'completed_branch_id': 'branch-global-semantic-closure-review',
                    'semantic_review_verdict': completed_semantic_review_verdict,
                    'semantic_review_verdict_status': completed_semantic_review_verdict.get('status'),
                    'semantic_review_recommended_transition': completed_semantic_review_verdict.get('recommended_transition'),
                }
            )
        if completed_status == 'fulfilled':
            proposal = self._global_semantic_verdict_proposal(completed_semantic_review_verdict)
            verdict_status = str(completed_semantic_review_verdict.get('status') or '').strip().lower()
            status = 'blocked' if verdict_status == 'blocked' else 'pending'
            return self._compact_mapping(
                {
                    'kind': 'ollmo.global_semantic_closure_review',
                    'status': status,
                    'authority': 'advisory_until_closure_confirms_verdict_transition',
                    'policy': 'semantic_review_completion_requires_structured_pass_verdict_before_freeze',
                    'reason': (
                        'global semantic review branch completed but did not return a passed verdict'
                    ),
                    'intent_anchor': prompt or None,
                    'proposal_count': 1 if proposal else 0,
                    'semantic_check_count': len(semantic_checks),
                    'structural_adequacy_status': (
                        str(intent_graph_adequacy.get('status') or '').strip().lower()
                        if isinstance(intent_graph_adequacy, Mapping)
                        else None
                    ),
                    'semantic_decision_proposal_count': len(semantic_decision_proposals),
                    'controlled_attention_frame_count': int(controlled_attention_review.get('frame_count') or 0),
                    'aspiration_frame_count': int(aspiration_review.get('frame_count') or 0),
                    'commitment_frame_count': int(commitment_review.get('frame_count') or 0),
                    'semantic_quality_status': str(semantic_quality_review.get('status') or '').strip() or None,
                    'completed_branch_id': 'branch-global-semantic-closure-review',
                    'semantic_review_verdict': completed_semantic_review_verdict,
                    'semantic_review_verdict_status': completed_semantic_review_verdict.get('status'),
                    'semantic_review_recommended_transition': completed_semantic_review_verdict.get('recommended_transition'),
                    'proposals': [proposal] if proposal else [],
                    'review_branch_id': 'branch-global-semantic-closure-review',
                    'content_payload': self._global_semantic_review_instruction(
                        prompt=prompt,
                        output_text=output_text,
                        evidence_payload=self._global_semantic_evidence_payload(
                            artifact_payload=artifact_payload,
                            checks=checks,
                            intent_graph_adequacy=intent_graph_adequacy,
                            decision_contract=decision_contract,
                        ),
                    ),
                    'content_payload_source': 'global_semantic_closure_review',
                    'stage_direction': 'run_global_semantic_closure_review',
                }
            )
        if not structural_pending and not semantic_pending:
            return self._compact_mapping(
                {
                    'kind': 'ollmo.global_semantic_closure_review',
                    'status': 'not_applicable',
                    'authority': 'advisory_read_model_only',
                    'policy': 'whole_turn_semantic_fit_checked_after_runtime_materialization',
                    'reason': 'no structural or semantic whole-graph closure signal requires review',
                }
            )
        if semantic_pending and local_unresolved and not structural_pending:
            return self._compact_mapping(
                {
                    'kind': 'ollmo.global_semantic_closure_review',
                    'status': 'waiting_on_local_closure',
                    'authority': 'advisory_read_model_only',
                    'policy': 'local_branch_truth_must_fit_whole_current_intent_before_freeze',
                    'reason': 'whole-turn semantic review waits until local promoted obligations are fulfilled or repaired',
                    'intent_anchor': prompt or None,
                    'proposal_count': 0,
                    'semantic_check_count': len(semantic_checks),
                    'semantic_decision_proposal_count': len(semantic_decision_proposals),
                    'controlled_attention_frame_count': int(controlled_attention_review.get('frame_count') or 0),
                    'aspiration_frame_count': int(aspiration_review.get('frame_count') or 0),
                    'commitment_frame_count': int(commitment_review.get('frame_count') or 0),
                    'semantic_quality_status': str(semantic_quality_review.get('status') or '').strip() or None,
                }
            )

        evidence_payload = self._global_semantic_evidence_payload(
            artifact_payload=artifact_payload,
            checks=checks,
            intent_graph_adequacy=intent_graph_adequacy,
            decision_contract=decision_contract,
        )
        proposals: list[dict[str, Any]] = []
        if structural_pending:
            proposals.append(
                self._compact_mapping(
                    {
                        'kind': 'ollmo.global_semantic_closure_proposal',
                        'proposal_id': 'global-semantic-closure-structural-adequacy',
                        'status': 'advisory',
                        'authority': 'advisory_until_closure_confirms_promoted_obligation_gap',
                        'decision_action': RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                        'recommended_transition': RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                        'confidence': 0.82,
                        'reason': 'structural graph adequacy is pending before whole-turn semantic closure can freeze',
                        'allowed_transitions': [
                            RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                            RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                            'clarify',
                            'waive_with_evidence',
                            'truthful_freeze_after_repair',
                        ],
                        'evidence_refs': ['intent_graph_adequacy', 'graph_output_counts', 'expected_output_counts'],
                    }
                )
            )
        if semantic_pending:
            semantic_refs: list[str] = []
            for check in semantic_checks:
                for key in ('branch_id', 'phase_id', 'obligation_id', 'task_id'):
                    value = str(check.get(key) or '').strip()
                    if value and value not in semantic_refs:
                        semantic_refs.append(value)
            proposals.append(
                self._compact_mapping(
                    {
                        'kind': 'ollmo.global_semantic_closure_proposal',
                        'proposal_id': 'global-semantic-closure-whole-intent-fit',
                        'status': 'advisory',
                        'authority': 'advisory_until_promoted_semantic_review_branch_completes',
                        'decision_action': RECOVERY_ACTION_SEMANTIC_REVIEW,
                        'recommended_transition': RECOVERY_ACTION_SEMANTIC_REVIEW,
                        'confidence': 0.76,
                        'reason': 'local output existence does not prove the generated work fits the whole current intent',
                        'allowed_transitions': [
                            RECOVERY_ACTION_SEMANTIC_REVIEW,
                            RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                            RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                            RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                            'waive_with_evidence',
                            'supersede_with_replacement_truth',
                            'truthful_freeze_after_review',
                        ],
                        'evidence_refs': [
                            'semantic_quality_review',
                            'semantic_decision_review',
                            'controlled_attention_review',
                            'aspiration_review',
                            'commitment_review',
                            *semantic_refs,
                        ],
                    }
                )
            )
        status = 'blocked' if completed_status == 'blocked' else 'pending'
        reason = (
            'global semantic review branch failed'
            if completed_status == 'blocked'
            else 'whole-turn semantic closure requires review before truthful freeze'
        )
        return self._compact_mapping(
            {
                'kind': 'ollmo.global_semantic_closure_review',
                'status': status,
                'authority': 'advisory_until_closure_promotes_review_or_repair',
                'policy': 'local_branch_truth_must_fit_whole_current_intent_before_freeze',
                'reason': reason,
                'intent_anchor': prompt or None,
                'proposal_count': len(proposals),
                'semantic_check_count': len(semantic_checks),
                'structural_adequacy_status': (
                    str(intent_graph_adequacy.get('status') or '').strip().lower()
                    if isinstance(intent_graph_adequacy, Mapping)
                    else None
                ),
                'semantic_decision_proposal_count': len(semantic_decision_proposals),
                'controlled_attention_frame_count': int(controlled_attention_review.get('frame_count') or 0),
                'aspiration_frame_count': int(aspiration_review.get('frame_count') or 0),
                'commitment_frame_count': int(commitment_review.get('frame_count') or 0),
                'semantic_quality_status': str(semantic_quality_review.get('status') or '').strip() or None,
                'proposals': proposals,
                'review_branch_id': 'branch-global-semantic-closure-review',
                'content_payload': self._global_semantic_review_instruction(
                    prompt=prompt,
                    output_text=output_text,
                    evidence_payload=evidence_payload,
                ),
                'content_payload_source': 'global_semantic_closure_review',
                'stage_direction': 'run_global_semantic_closure_review',
            }
        )

    def _global_semantic_closure_checks(
        self,
        global_review: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(global_review, Mapping):
            return []
        if str(global_review.get('status') or '').strip().lower() not in {'pending', 'blocked'}:
            return []
        proposals = global_review.get('proposals') if isinstance(global_review.get('proposals'), list) else []
        checks: list[dict[str, Any]] = []
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                continue
            verdict_payload = (
                proposal.get('semantic_review_verdict')
                if isinstance(proposal.get('semantic_review_verdict'), Mapping)
                else {}
            )
            if verdict_payload:
                action = self._recovery_action_for_semantic_review_transition(
                    proposal.get('recommended_transition') or proposal.get('decision_action')
                )
                if not action:
                    continue
            else:
                action = normalize_recovery_suggested_action(proposal.get('decision_action'), default='')
                if action == RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS:
                    continue
                if action != RECOVERY_ACTION_SEMANTIC_REVIEW:
                    continue
            branch_id = str(global_review.get('review_branch_id') or 'branch-global-semantic-closure-review').strip()
            phase_id = 'phase-global-semantic-closure-review'
            obligation_id = 'obligation-global-semantic-closure-review'
            execution_contract = {
                'kind': 'ollmo.execution_contract',
                'branch_id': branch_id,
                'phase_id': phase_id,
                'capability': CAPABILITY_CHAT,
                'output_type': 'text',
                'workload_task_ref': {
                    'task_id': 'task-global-semantic-closure-review',
                    'phase_id': phase_id,
                    'branch_id': branch_id,
                },
                'output_obligation_ref': {
                    'obligation_id': obligation_id,
                    'phase_id': phase_id,
                    'branch_id': branch_id,
                    'output_type': 'text',
                },
                'output_contract': {
                    'output_type': 'text',
                    'required': True,
                    'fulfillment_policy': 'semantic_review_text_required',
                },
                'input_refs': [
                    {'kind': 'intent_anchor', 'ref': 'current_user_intent'},
                    {'kind': 'closure_checks', 'ref': 'runtime.graph_closure_review.checks'},
                    {'kind': 'runtime_evidence', 'ref': 'artifacts_and_late_fill'},
                ],
            }
            checks.append(
                self._compact_mapping(
                    {
                        'check_kind': 'global_semantic_closure',
                        'status': str(global_review.get('status') or '').strip().lower() or 'pending',
                        'evidence': 'global_semantic_closure_review',
                        'reason': str(proposal.get('reason') or global_review.get('reason') or '').strip(),
                        'obligation_id': obligation_id,
                        'phase_id': phase_id,
                        'branch_id': branch_id,
                        'capability': CAPABILITY_CHAT,
                        'output_type': 'text',
                        'role': 'semantic_review_transition',
                        'repair_action': action,
                        'recovery_action': action,
                        'repair_action_reason': str(proposal.get('reason') or '').strip(),
                        'semantic_review_required': action == RECOVERY_ACTION_SEMANTIC_REVIEW,
                        'semantic_review_action': action,
                        'semantic_review_authority': (
                            'closure_promoted_semantic_review_branch'
                            if action == RECOVERY_ACTION_SEMANTIC_REVIEW
                            else 'semantic_verdict_advisory_transition'
                        ),
                        'semantic_review_criteria': [
                            'whole_turn_output_fits_current_user_intent',
                            'local_branch_outputs_are_used_in_their_declared_roles',
                            'missing_or_wrong_work_is_reported_as_repair_waiver_or_supersession',
                        ],
                        'review_criteria': [
                            'runtime_text_exists_when_fulfilled',
                            'whole_turn_output_fits_current_user_intent',
                            'local_branch_outputs_are_used_in_their_declared_roles',
                        ],
                        'global_semantic_closure_review': {
                            key: global_review.get(key)
                            for key in (
                                'kind',
                                'status',
                                'authority',
                                'policy',
                                'reason',
                                'proposal_count',
                                'semantic_check_count',
                                'structural_adequacy_status',
                                'semantic_decision_proposal_count',
                                'semantic_quality_status',
                                'semantic_review_verdict_status',
                                'semantic_review_recommended_transition',
                            )
                            if global_review.get(key) not in (None, '', [], {})
                        },
                        'global_semantic_closure_proposal': proposal,
                        'global_semantic_closure_status': str(global_review.get('status') or '').strip().lower(),
                        'global_semantic_closure_reason': str(global_review.get('reason') or '').strip(),
                        'global_semantic_closure_confidence': proposal.get('confidence'),
                        'semantic_review_verdict': verdict_payload,
                        'semantic_review_verdict_status': (
                            str(verdict_payload.get('status') or '').strip().lower()
                            if verdict_payload
                            else None
                        ),
                        'semantic_review_recommended_transition': (
                            str(verdict_payload.get('recommended_transition') or '').strip()
                            if verdict_payload
                            else None
                        ),
                        'blocked_by_dependency_input': action in {
                            RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                            RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
                        },
                        'blocked_by_branch_contract': action == RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                        'content_payload': global_review.get('content_payload'),
                        'content_payload_source': global_review.get('content_payload_source'),
                        'stage_direction': global_review.get('stage_direction'),
                        'execution_contract': execution_contract,
                        'input_refs': execution_contract['input_refs'],
                    }
                )
            )
        return checks

    def artifact_gap_is_already_fulfilled(
        self,
        artifact_gap: Optional[dict[str, Any]],
        payload: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(artifact_gap, dict):
            return False
        missing_artifact_type = str(artifact_gap.get('missing_artifact_type') or '').strip().lower()
        if not missing_artifact_type:
            missing_artifact_type = str(
                self.artifact_type_for_capability(artifact_gap.get('expected_capability')) or ''
            ).strip().lower()
        if not missing_artifact_type:
            return False
        if missing_artifact_type == 'text' and self.source_requires_text_artifact(artifact_gap, 'text'):
            return self._text_artifact_gap_is_already_fulfilled(artifact_gap, payload)
        if normalize_capability(artifact_gap.get('expected_capability')) == CAPABILITY_SPEECH_TO_TEXT:
            return self._speech_to_text_gap_is_already_fulfilled(artifact_gap, payload)
        return self.response_payload_has_artifact_type(payload, missing_artifact_type)

    def _speech_to_text_gap_is_already_fulfilled(
        self,
        artifact_gap: Mapping[str, Any],
        payload: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(payload, Mapping):
            return False
        expected_branch_id = str(
            artifact_gap.get('expected_branch_id')
            or artifact_gap.get('branch_id')
            or ''
        ).strip()
        expected_phase_id = str(
            artifact_gap.get('expected_phase_id')
            or artifact_gap.get('phase_id')
            or ''
        ).strip()
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        for result in late_fill.get('fill_results') or []:
            if not isinstance(result, Mapping):
                continue
            if normalize_capability(result.get('capability')) != CAPABILITY_SPEECH_TO_TEXT:
                continue
            result_branch_id = str(result.get('branch_id') or '').strip()
            result_phase_id = str(result.get('phase_id') or '').strip()
            if expected_branch_id and result_branch_id and result_branch_id != expected_branch_id:
                continue
            if expected_phase_id and result_phase_id and result_phase_id != expected_phase_id:
                continue
            if (
                str(result.get('saved_text_path') or '').strip()
                or str(result.get('content_payload') or '').strip()
                or str(result.get('result_text') or '').strip()
            ):
                return True
        return normalize_capability(payload.get('response_capability') or payload.get('capability')) == CAPABILITY_SPEECH_TO_TEXT and bool(
            str(payload.get('saved_text_path') or payload.get('savedTextPath') or '').strip()
            or str(payload.get('output_text') or '').strip()
        )

    def _text_artifact_gap_is_already_fulfilled(
        self,
        artifact_gap: Mapping[str, Any],
        payload: Optional[dict[str, Any]],
    ) -> bool:
        build_canonical_response_artifacts = self._hook('build_canonical_response_artifacts')

        if not isinstance(payload, Mapping):
            return False
        expected_extension = _expected_text_artifact_extension(artifact_gap)
        expected_source_name = _expected_text_artifact_source_name(artifact_gap)
        artifacts = payload.get('artifacts')
        if not isinstance(artifacts, list):
            artifacts = build_canonical_response_artifacts(dict(payload))
        candidates: list[Mapping[str, Any]] = [
            item
            for item in artifacts or []
            if isinstance(item, Mapping)
            and str(item.get('type') or '').strip().lower() == 'text'
        ]
        if not candidates and str(payload.get('saved_text_path') or payload.get('savedTextPath') or '').strip():
            candidates = [
                {
                    'type': 'text',
                    'path': payload.get('saved_text_path') or payload.get('savedTextPath'),
                    'text_artifact_request': payload.get('text_artifact_request')
                    if isinstance(payload.get('text_artifact_request'), Mapping)
                    else {},
                }
            ]
        for artifact in candidates:
            artifact_extension = _text_artifact_extension_from_record(artifact)
            if expected_extension and artifact_extension != expected_extension:
                continue
            if expected_source_name and not expected_extension:
                artifact_source_name = str(
                    artifact.get('name')
                    or (
                        artifact.get('text_artifact_request', {}).get('source_name')
                        if isinstance(artifact.get('text_artifact_request'), Mapping)
                        else ''
                    )
                    or ''
                ).strip().lower()
                if artifact_source_name != expected_source_name:
                    continue
            if str(artifact.get('path') or artifact.get('source_path') or '').strip():
                return True
        return False

    def response_payload_has_real_saved_artifact(
        self,
        payload: Optional[dict[str, Any]],
    ) -> bool:
        build_canonical_response_artifacts = self._hook('build_canonical_response_artifacts')

        if not isinstance(payload, Mapping):
            return False
        artifacts = payload.get('artifacts')
        if not isinstance(artifacts, list):
            artifacts = build_canonical_response_artifacts(dict(payload))
        for artifact in artifacts or []:
            if not isinstance(artifact, Mapping):
                continue
            if str(artifact.get('path') or '').strip():
                return True
        return False

    def _claimed_materialization_capabilities_from_text(self, text: str) -> list[str]:
        normalized = str(text or '').strip()
        capabilities: list[str] = []
        for capability, pattern in (
            (CAPABILITY_IMAGE_GENERATION, _TEXT_ONLY_IMAGE_CAPABILITY_CLAIM_RE),
            (CAPABILITY_TEXT_TO_SPEECH, _TEXT_ONLY_AUDIO_CAPABILITY_CLAIM_RE),
        ):
            if pattern.search(normalized) and capability not in capabilities:
                capabilities.append(capability)
        return capabilities

    def _instance_capability_set(self, instance: Mapping[str, Any]) -> set[str]:
        capabilities: set[str] = set()

        def add(raw_value: Any) -> None:
            normalized = normalize_capability(raw_value)
            if normalized:
                capabilities.add(normalized)

        for key in ('capability', 'mode'):
            add(instance.get(key))
        for key in (
            'supported_capabilities',
            'provider_capabilities',
            'capabilities',
            'instance_capabilities',
            'package_capabilities',
        ):
            values = instance.get(key)
            if isinstance(values, list):
                for value in values:
                    add(value)
        backend_metadata = instance.get('backend_metadata')
        if isinstance(backend_metadata, Mapping):
            for key in ('capabilities', 'instance_capabilities', 'package_capabilities'):
                values = backend_metadata.get(key)
                if isinstance(values, list):
                    for value in values:
                        add(value)
        return capabilities

    def _instance_is_runtime_available(self, instance: Mapping[str, Any]) -> bool:
        runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), Mapping) else {}
        readiness = str(runtime_status.get('readiness') or instance.get('readiness') or '').strip().lower()
        if readiness and readiness not in {'ready', 'degraded'}:
            return False
        process_alive = runtime_status.get('process_alive', instance.get('process_alive'))
        if process_alive is False:
            return False
        port_listening = runtime_status.get('port_listening', instance.get('port_listening'))
        if port_listening is False:
            return False
        return True

    def runtime_capability_availability(self, capability: Optional[str]) -> dict[str, Any]:
        normalized_capability = normalize_capability(capability)
        payload: dict[str, Any] = {
            'capability': normalized_capability or None,
            'status': 'unknown',
            'available': False,
            'instances': [],
        }
        if not normalized_capability:
            return payload

        load_running_instances = self.hooks.get('load_running_instances')
        merge_instances_with_runtime_status = self.hooks.get('merge_instances_with_runtime_status')
        if not callable(load_running_instances):
            payload['reason'] = 'runtime_instance_loader_unavailable'
            return payload

        try:
            instances = load_running_instances()
            if callable(merge_instances_with_runtime_status):
                kwargs: dict[str, Any] = {'refresh': True}
                runtime_status_path_getter = self.hooks.get('runtime_status_path_getter')
                if callable(runtime_status_path_getter):
                    path = runtime_status_path_getter()
                    if path:
                        kwargs['path'] = path
                instances = merge_instances_with_runtime_status(instances, **kwargs)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                'failed to load runtime capability availability for %s',
                normalized_capability,
                exc_info=True,
            )
            payload['reason'] = 'runtime_instance_load_failed'
            payload['error'] = str(exc)
            return payload

        matching_instances: list[dict[str, Any]] = []
        for raw_instance in instances if isinstance(instances, list) else []:
            if not isinstance(raw_instance, Mapping):
                continue
            capabilities = self._instance_capability_set(raw_instance)
            if normalized_capability not in capabilities:
                continue
            instance_summary = {
                'instance_id': str(raw_instance.get('instance_id') or '').strip() or None,
                'model': str(raw_instance.get('model') or raw_instance.get('modelName') or '').strip() or None,
                'backend': str(raw_instance.get('backend') or '').strip() or None,
                'readiness': str(
                    (
                        raw_instance.get('runtime_status')
                        if isinstance(raw_instance.get('runtime_status'), Mapping)
                        else {}
                    ).get('readiness')
                    or raw_instance.get('readiness')
                    or ''
                ).strip() or None,
                'available': self._instance_is_runtime_available(raw_instance),
            }
            matching_instances.append(
                {
                    key: value
                    for key, value in instance_summary.items()
                    if value not in (None, '', [], {})
                }
            )
        available_instances = [item for item in matching_instances if item.get('available')]
        payload['instances'] = matching_instances
        if available_instances:
            payload['status'] = 'available'
            payload['available'] = True
            payload['available_instance_ids'] = [
                item.get('instance_id')
                for item in available_instances
                if item.get('instance_id')
            ]
            return payload
        if matching_instances:
            payload['status'] = 'unavailable'
            payload['reason'] = 'matching_instances_not_ready'
            return payload
        payload['status'] = 'unavailable'
        payload['reason'] = 'no_matching_running_instance'
        return payload

    def _truth_gate_text_only_artifact_claim(
        self,
        text: str,
    ) -> str:
        normalized = str(text or '').strip()
        if not normalized:
            return normalized
        match = _MARKDOWN_CODE_BLOCK_RE.search(normalized)
        if match:
            return (
                'No local file or downloadable artifact was created in this response. '
                'The content below exists only as response text.\n\n'
                f'{match.group(0).strip()}'
            ).strip()
        return (
            'No local file or downloadable artifact was created in this response. '
            'The content exists only as response text.'
        )

    def _truth_gate_annotated_text_only_artifact_claim(
        self,
        text: str,
    ) -> str:
        normalized = str(text or '').strip()
        prefix = (
            'Runtime truth: no separate local or downloadable artifacts were materialized in this response. '
            'Items labeled as artifacts, images, or fulfilled outputs below are response text unless an `outputs[]` '
            'entry or artifact record says otherwise.'
        )
        if normalized.startswith(prefix):
            return normalized
        if not normalized:
            return prefix
        return f'{prefix}\n\n{normalized}'

    def _truth_guard_predecessor_bundle_is_authorized(
        self,
        request_payload: Mapping[str, Any],
        raw_references: Any,
        sanitized_references: list[dict[str, Any]],
    ) -> bool:
        """Prove that a multi-reference expansion is one carried predecessor."""

        if not isinstance(raw_references, list):
            return False
        raw_items = [dict(item) for item in raw_references if isinstance(item, Mapping)]
        if len(raw_items) < 2 or len(raw_items) != len(sanitized_references):
            return False

        source_response_ids = {
            str(item.get('source_response_id') or '').strip()
            for item in raw_items
        }
        if len(source_response_ids) != 1 or not next(iter(source_response_ids), ''):
            return False
        source_response_id = next(iter(source_response_ids))
        current_conversation_id = str(
            request_payload.get('conversation_id')
            or request_payload.get('conversationId')
            or ''
        ).strip()
        if not current_conversation_id:
            return False

        carrier = next(
            (
                dict(item)
                for item in reversed(request_payload.get('ghost_messages') or [])
                if isinstance(item, Mapping)
                and str(item.get('role') or '').strip().lower() == 'assistant'
                and str(item.get('response_id') or '').strip() == source_response_id
            ),
            None,
        )
        if not carrier:
            return False
        carrier_message_id = str(
            carrier.get('message_id') or carrier.get('messageId') or ''
        ).strip()
        carrier_content = str(carrier.get('content') or '').strip()
        if not carrier_message_id or not carrier_content:
            return False

        get_response_lookup_record = self.hooks.get('get_response_lookup_record')
        if not callable(get_response_lookup_record):
            return False
        try:
            predecessor_record = get_response_lookup_record(source_response_id)
        except Exception:  # noqa: BLE001 - predecessor truth lookup must fail closed
            return False
        if not isinstance(predecessor_record, Mapping):
            return False
        if str(predecessor_record.get('id') or '').strip() != source_response_id:
            return False
        predecessor_payload = (
            predecessor_record.get('response_payload')
            if isinstance(predecessor_record.get('response_payload'), Mapping)
            else {}
        )
        predecessor_frame = (
            predecessor_payload.get('response_frame')
            if isinstance(predecessor_payload.get('response_frame'), Mapping)
            else {}
        )
        predecessor_request = (
            predecessor_frame.get('request')
            if isinstance(predecessor_frame.get('request'), Mapping)
            else {}
        )
        predecessor_conversation_id = str(
            predecessor_request.get('conversation_id')
            or predecessor_request.get('conversationId')
            or ''
        ).strip()
        predecessor_message_id = str(
            predecessor_record.get('message_id') or ''
        ).strip()
        predecessor_lifecycle = str(
            predecessor_record.get('lifecycle_state')
            or predecessor_payload.get('lifecycle_state')
            or predecessor_payload.get('status')
            or ''
        ).strip().lower()
        if (
            predecessor_conversation_id != current_conversation_id
            or predecessor_message_id != carrier_message_id
            or predecessor_lifecycle not in {'completed', 'repair_needed'}
        ):
            return False
        predecessor_text = str(predecessor_payload.get('output_text') or '').strip()
        if not predecessor_text or predecessor_text != carrier_content:
            return False

        def _artifacts_by_ref(raw_artifacts: Any) -> dict[str, dict[str, Any]]:
            artifacts: dict[str, dict[str, Any]] = {}
            for raw_artifact in raw_artifacts if isinstance(raw_artifacts, list) else []:
                if not isinstance(raw_artifact, Mapping):
                    continue
                artifact = dict(raw_artifact)
                artifact_ref = str(
                    artifact.get('artifact_ref') or artifact.get('ref') or ''
                ).strip()
                if artifact_ref:
                    artifacts[artifact_ref] = artifact
            return artifacts

        carrier_artifacts = _artifacts_by_ref(carrier.get('artifacts'))
        predecessor_artifacts = _artifacts_by_ref(predecessor_payload.get('artifacts'))
        sanitized_artifacts = _artifacts_by_ref(sanitized_references)
        seen_artifact_refs: set[str] = set()
        for raw_item in raw_items:
            artifact_type = str(raw_item.get('type') or '').strip().lower()
            raw_source_message_id = str(
                raw_item.get('source_message_id')
                or raw_item.get('message_id')
                or raw_item.get('messageId')
                or ''
            ).strip()
            if raw_source_message_id != carrier_message_id:
                return False
            if artifact_type == 'message':
                raw_content = str(
                    raw_item.get('content')
                    or raw_item.get('text')
                    or raw_item.get('prompt')
                    or ''
                ).strip()
                if raw_content != carrier_content:
                    return False
                continue

            artifact_ref = str(
                raw_item.get('artifact_ref') or raw_item.get('ref') or ''
            ).strip()
            if not artifact_ref or artifact_ref in seen_artifact_refs:
                return False
            seen_artifact_refs.add(artifact_ref)
            carrier_artifact = carrier_artifacts.get(artifact_ref)
            predecessor_artifact = predecessor_artifacts.get(artifact_ref)
            sanitized_artifact = sanitized_artifacts.get(artifact_ref)
            if not carrier_artifact or not predecessor_artifact or not sanitized_artifact:
                return False
            canonical_type = str(
                predecessor_artifact.get('type')
                or predecessor_artifact.get('kind')
                or ''
            ).strip().lower()
            if artifact_type and canonical_type and artifact_type != canonical_type:
                return False
            canonical_path = str(
                predecessor_artifact.get('path')
                or predecessor_artifact.get('source_path')
                or ''
            ).strip()
            carrier_path = str(
                carrier_artifact.get('path')
                or carrier_artifact.get('source_path')
                or ''
            ).strip()
            sanitized_path = str(
                sanitized_artifact.get('path')
                or sanitized_artifact.get('source_path')
                or ''
            ).strip()
            if not canonical_path or carrier_path != canonical_path or sanitized_path != canonical_path:
                return False
            canonical_artifact_id = str(predecessor_artifact.get('artifact_id') or '').strip()
            raw_artifact_id = str(raw_item.get('artifact_id') or '').strip()
            if raw_artifact_id and canonical_artifact_id != raw_artifact_id:
                return False
        return True

    def _truth_guard_request_payload(
        self,
        request_payload: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(request_payload, Mapping):
            return None
        sanitized_payload = dict(request_payload)
        sanitize_selected_references = self.hooks.get(
            'sanitize_selected_reference_artifacts'
        )
        if not callable(sanitize_selected_references):
            return sanitized_payload

        def _sanitize_records(raw_value: Any) -> list[dict[str, Any]]:
            try:
                sanitized_records = sanitize_selected_references(raw_value)
            except Exception:  # noqa: BLE001 - malformed client evidence must fail closed
                return []
            return [
                dict(item)
                for item in sanitized_records
                if isinstance(item, Mapping)
            ]

        def _sanitize_record_collection(raw_value: Any) -> list[dict[str, Any]]:
            if not isinstance(raw_value, list):
                return _sanitize_records(raw_value)
            sanitized_collection: list[dict[str, Any]] = []
            for raw_item in raw_value:
                sanitized_collection.extend(_sanitize_records(raw_item))
            return sanitized_collection

        raw_selected_references = (
            request_payload.get('reference_artifacts')
            if request_payload.get('reference_artifacts') is not None
            else request_payload.get('selected_reference_artifact')
            if request_payload.get('selected_reference_artifact') is not None
            else request_payload.get('selectedReferenceArtifact')
            if request_payload.get('selectedReferenceArtifact') is not None
            else request_payload.get('selected_reference_artifacts')
            if request_payload.get('selected_reference_artifacts') is not None
            else request_payload.get('selectedReferenceArtifacts')
        )
        # The intake sanitizer intentionally projects a list to one message plus
        # one compatibility artifact. Expanding beyond that legacy boundary is
        # allowed only for a canonical, same-conversation predecessor bundle.
        compatibility_selected_references = _sanitize_records(raw_selected_references)
        expanded_selected_references = _sanitize_record_collection(raw_selected_references)
        expands_compatibility_projection = (
            len(expanded_selected_references) > len(compatibility_selected_references)
        )
        predecessor_bundle_verified = (
            isinstance(raw_selected_references, list)
            and len(raw_selected_references) >= 2
            and self._truth_guard_predecessor_bundle_is_authorized(
                request_payload,
                raw_selected_references,
                expanded_selected_references,
            )
        )
        predecessor_bundle_authorized = bool(
            expands_compatibility_projection
            and predecessor_bundle_verified
        )
        lineage_bearing_bundle_claim = bool(
            isinstance(raw_selected_references, list)
            and len(raw_selected_references) >= 2
            and any(
                isinstance(item, Mapping)
                and str(item.get('source_response_id') or '').strip()
                for item in raw_selected_references
            )
        )
        if predecessor_bundle_authorized:
            selected_references = expanded_selected_references
        elif expands_compatibility_projection and lineage_bearing_bundle_claim:
            # A failed carried-bundle proof must not leave its unverified
            # message content behind as an independent text source.
            selected_references = [
                item
                for item in compatibility_selected_references
                if str(item.get('type') or '').strip().lower() != 'message'
            ]
        else:
            selected_references = compatibility_selected_references
        raw_predecessor_context = request_payload.get(
            'current_predecessor_context'
        )
        sanitized_payload.pop('current_predecessor_context', None)
        if predecessor_bundle_verified and isinstance(
            raw_predecessor_context,
            Mapping,
        ):
            context_source_response_id = str(
                raw_predecessor_context.get('source_response_id') or ''
            ).strip()
            reference_source_response_ids = {
                str(item.get('source_response_id') or '').strip()
                for item in (
                    raw_selected_references
                    if isinstance(raw_selected_references, list)
                    else []
                )
                if isinstance(item, Mapping)
                and str(item.get('source_response_id') or '').strip()
            }
            get_response_lookup_record = self.hooks.get(
                'get_response_lookup_record'
            )
            predecessor_record = None
            if (
                callable(get_response_lookup_record)
                and len(reference_source_response_ids) == 1
                and context_source_response_id in reference_source_response_ids
            ):
                try:
                    predecessor_record = get_response_lookup_record(
                        context_source_response_id
                    )
                except Exception:  # noqa: BLE001 - context proof must fail closed
                    predecessor_record = None
            predecessor_payload = (
                predecessor_record.get('response_payload')
                if isinstance(predecessor_record, Mapping)
                and isinstance(
                    predecessor_record.get('response_payload'),
                    Mapping,
                )
                else {}
            )
            canonical_prompts = extract_canonical_predecessor_image_prompts(
                predecessor_payload
            )
            if not canonical_prompts:
                predecessor_late_fill = (
                    predecessor_payload.get('late_fill')
                    if isinstance(predecessor_payload.get('late_fill'), Mapping)
                    else {}
                )
                prepared_content = str(
                    predecessor_late_fill.get('content_payload')
                    or predecessor_payload.get('content_payload')
                    or ''
                ).strip()
                if prepared_content:
                    canonical_prompts = self.extract_batch_image_prompts(
                        prepared_content,
                        expected_count=0,
                        allow_plain_alpha_sequence=False,
                    )
            claimed_prompts = [
                str(item).strip()
                for item in (
                    raw_predecessor_context.get('batch_prompts') or []
                )
                if str(item).strip()
            ]
            if canonical_prompts and claimed_prompts == canonical_prompts:
                sanitized_payload['current_predecessor_context'] = {
                    'kind': 'ollmo.current_predecessor_context',
                    'status': 'authorized',
                    'authorization': (
                        'canonical_same_conversation_predecessor'
                    ),
                    'source_response_id': context_source_response_id,
                    'source_message_id': str(
                        raw_predecessor_context.get('source_message_id') or ''
                    ).strip(),
                    'batch_prompts': canonical_prompts,
                    'text_artifact_refs': [
                        str(item).strip()
                        for item in (
                            raw_predecessor_context.get(
                                'text_artifact_refs'
                            )
                            or []
                        )
                        if str(item).strip()
                    ],
                }
        for key in (
            'reference_artifacts',
            'selected_reference_artifact',
            'selectedReferenceArtifact',
            'selected_reference_artifacts',
            'selectedReferenceArtifacts',
        ):
            sanitized_payload.pop(key, None)
        sanitized_payload['selected_reference_artifacts'] = selected_references

        sanitized_input_artifacts: list[dict[str, Any]] = []
        raw_input_artifacts = request_payload.get('input_artifacts')
        if isinstance(raw_input_artifacts, list):
            for raw_input_artifact in raw_input_artifacts:
                sanitized_input_artifacts.extend(
                    _sanitize_records(raw_input_artifact)
                )
        for key in ('file_path', 'route_artifact_path', 'artifact_path'):
            raw_path = str(request_payload.get(key) or '').strip()
            sanitized_payload.pop(key, None)
            if not raw_path:
                continue
            path_capabilities = _direct_artifact_record_source_capabilities(
                {'path': raw_path}
            )
            artifact_type = (
                'image'
                if path_capabilities == {CAPABILITY_IMAGE_GENERATION}
                else 'audio'
                if path_capabilities == {CAPABILITY_TEXT_TO_SPEECH}
                else 'text'
                if path_capabilities == {CAPABILITY_CHAT}
                else ''
            )
            if artifact_type:
                sanitized_input_artifacts.extend(
                    _sanitize_records({'type': artifact_type, 'path': raw_path})
                )
        sanitized_payload['input_artifacts'] = sanitized_input_artifacts
        return sanitized_payload

    def truth_gate_response_output_claims(
        self,
        payload: Optional[dict[str, Any]],
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        text = str(payload.get('output_text') or payload.get('content') or '').strip()
        truth_request_payload = self._truth_guard_request_payload(request_payload)
        request_prompt = ''
        if isinstance(truth_request_payload, Mapping):
            request_prompt = extract_responses_current_turn_prompt(truth_request_payload) or str(
                truth_request_payload.get('prompt') or truth_request_payload.get('input') or ''
            ).strip()
        if (
            text
            and request_prompt
            and (
                _request_requires_current_source_for_transform(
                    request_prompt,
                    request_payload=truth_request_payload,
                    route_payload=route_payload,
                    response_payload=payload,
                )
                or text_artifact_request_is_ungrounded_reference(
                    request_prompt,
                    source_available=_payload_has_direct_artifact_source(
                        truth_request_payload,
                        route_payload,
                        payload,
                    )
                    or _prompt_has_actionable_inline_text_source(request_prompt),
                )
            )
        ):
            rewritten_text = (
                'I need the source/content for the referenced file before I can create that artifact. '
                'Please provide or select the HTML/code/content to materialize.'
            )
            updated = dict(payload)
            updated['output_text'] = rewritten_text
            if 'content' in updated and isinstance(updated.get('content'), str):
                updated['content'] = rewritten_text
            runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), Mapping) else {}
            runtime['truth_guard'] = {
                'kind': 'ungrounded_text_artifact_reference',
                'status': 'clarification_required',
                'reason': 'artifact request used a demonstrative reference without a current or selected source',
            }
            updated['runtime'] = runtime
            return updated
        response_capability = str(
            payload.get('capability')
            or (route_payload or {}).get('capability')
            or (request_payload or {}).get('capability')
            or ''
        ).strip()
        control_json_leak = control_json_envelope_suspected(text) and not request_explicitly_allows_control_diagnostics(
            request_payload=truth_request_payload,
            route_payload=route_payload,
            capability=response_capability,
        )
        explicit_false_claim = bool(_FALSE_LOCAL_ARTIFACT_CLAIM_RE.search(text))
        soft_materialization_claim = bool(_TEXT_ONLY_MATERIALIZATION_CLAIM_RE.search(text))
        if not text or not (control_json_leak or explicit_false_claim or soft_materialization_claim):
            return dict(payload)
        if not control_json_leak and self.response_payload_has_real_saved_artifact(payload):
            return dict(payload)

        updated = dict(payload)
        if control_json_leak:
            acceptance = classify_phase_output_text(text)
            substantive_payload = str(acceptance.get('accepted_text') or '').strip()
            rewritten_text = substantive_payload or (
                'Runtime truth: internal planner/control JSON was withheld from the user-facing answer. '
                'The requested output remains open until a real response or artifact payload is materialized.'
            )
            guard_status = 'unwrapped' if substantive_payload else 'repair_required'
            guard_reason = (
                'internal control JSON was unwrapped to its substantive content_payload'
                if substantive_payload
                else 'internal control JSON leaked into visible output'
            )
            guard_kind = 'control_json_boundary'
        elif explicit_false_claim:
            rewritten_text = self._truth_gate_text_only_artifact_claim(text)
            guard_status = 'rewritten'
            guard_reason = 'response claimed local file or artifact creation without runtime artifact truth'
            guard_kind = 'local_artifact_claims'
        else:
            rewritten_text = self._truth_gate_annotated_text_only_artifact_claim(text)
            guard_status = 'annotated'
            guard_reason = 'response labeled text sections as materialized artifacts without runtime artifact truth'
            guard_kind = 'local_artifact_claims'
        claimed_capabilities = self._claimed_materialization_capabilities_from_text(text)
        updated['output_text'] = rewritten_text
        if 'content' in updated and isinstance(updated.get('content'), str):
            updated['content'] = rewritten_text
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), Mapping) else {}
        runtime['truth_guard'] = {
            'kind': guard_kind,
            'status': guard_status,
            'reason': guard_reason,
            **(
                {
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    'runtime_effect': 'materialization_blocked',
                }
                if guard_status == 'repair_required'
                else {}
            ),
            **({'claimed_capabilities': claimed_capabilities} if claimed_capabilities else {}),
        }
        if control_json_leak:
            runtime['phase_output_acceptance'] = phase_output_acceptance_metadata([acceptance])
        updated['runtime'] = runtime
        return updated

    def _current_phase_payload(self, phase_graph: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(phase_graph, dict):
            return {}
        current_phase_id = str(phase_graph.get('current_phase_id') or '').strip()
        phases = phase_graph.get('phases')
        if not current_phase_id or not isinstance(phases, list):
            return {}
        for raw_phase in phases:
            if not isinstance(raw_phase, dict):
                continue
            if str(raw_phase.get('phase_id') or '').strip() != current_phase_id:
                continue
            return dict(raw_phase)
        return {}

    def _request_is_ghost_owned(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> bool:
        if isinstance(request_payload, Mapping):
            raw_value = request_payload.get('ghost_route')
            if isinstance(raw_value, bool):
                return raw_value
            token = str(raw_value or '').strip().lower()
            if token in {'1', 'true', 'yes', 'on'}:
                return True
        if isinstance(route_payload, Mapping):
            token = str(route_payload.get('route_source') or '').strip().lower()
            if token in {'ghost_carried', 'self_heal'}:
                return True
        return False

    def build_ghost_runtime_policy_system_message(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, str]]:
        if not self._request_is_ghost_owned(
            route_payload=route_payload,
            request_payload=request_payload,
        ):
            return None
        runtime_policy = str(_load_runtime_ghost_policy() or '').strip()
        if not runtime_policy:
            return None
        return {
            'role': 'system',
            'content': '\n'.join(
                [
                    _GHOST_RUNTIME_POLICY_SYSTEM_MARKER,
                    'You are Ghost inside Ollmo for this turn.',
                    'The repo Ghost runtime policy is already attached below.',
                    'Do not claim that you cannot access, read, or know GHOST.md for this turn.',
                    runtime_policy,
                ]
            ),
        }

    def inject_ghost_runtime_policy_into_chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        injected = list(messages or [])
        if any(
            str(item.get('role') or '').strip().lower() == 'system'
            and _GHOST_RUNTIME_POLICY_SYSTEM_MARKER in str(item.get('content') or '')
            for item in injected
            if isinstance(item, dict)
        ):
            return injected
        system_message = self.build_ghost_runtime_policy_system_message(
            route_payload=route_payload,
            request_payload=request_payload,
        )
        if not system_message:
            return injected
        return [system_message] + injected

    def _extract_semantic_phase_payload_from_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        nested = payload.get('phase_payload') if isinstance(payload.get('phase_payload'), dict) else {}
        semantic_payload: dict[str, Any] = {}
        for key in _SEMANTIC_PHASE_PAYLOAD_KEYS:
            value = payload.get(key)
            if value in (None, '', [], {}):
                value = nested.get(key)
            if value not in (None, '', [], {}):
                semantic_payload[key] = value
        return semantic_payload

    def semantic_payload_for_capability(
        self,
        payload: Optional[dict[str, Any]],
        *,
        capability: Optional[str],
    ) -> dict[str, Any]:
        semantic_payload = self._extract_semantic_phase_payload_from_payload(payload)
        normalized_capability = normalize_capability(capability)
        if normalized_capability == CAPABILITY_IMAGE_GENERATION:
            artifact_prompt = str(semantic_payload.get('artifact_prompt') or '').strip()
            if not artifact_prompt:
                shared_body = str(semantic_payload.get('content_payload') or '').strip()
                if shared_body:
                    semantic_payload['artifact_prompt'] = shared_body
                    semantic_payload['artifact_prompt_source'] = (
                        str(semantic_payload.get('content_payload_source') or '').strip()
                        or 'shared_content_payload'
                    )
                    semantic_payload.setdefault('phase_summary', shared_body)
        if normalized_capability == CAPABILITY_TEXT_TO_SPEECH:
            content_payload = str(semantic_payload.get('content_payload') or '').strip()
            if not content_payload:
                shared_prompt = str(semantic_payload.get('artifact_prompt') or '').strip()
                if shared_prompt:
                    semantic_payload['content_payload'] = shared_prompt
                    semantic_payload['content_payload_source'] = (
                        str(semantic_payload.get('artifact_prompt_source') or '').strip()
                        or 'shared_artifact_prompt'
                    )
        return {
            key: value
            for key, value in semantic_payload.items()
            if value not in (None, '', [], {})
        }

    def extract_batch_image_prompts(
        self,
        prepared_text: str,
        *,
        expected_count: int = 0,
        allow_plain_alpha_sequence: bool = False,
    ) -> list[str]:
        normalized_text = str(prepared_text or '').strip()
        prompts: list[str] = []
        plain_alpha_candidate_present = bool(
            allow_plain_alpha_sequence
            and _contains_plain_alpha_image_prompt_line(normalized_text)
        )
        numbered_section_prompts = _extract_numbered_image_prompt_section(
            normalized_text,
            expected_count=expected_count,
        )
        filename_social_asset_prompts = _extract_filename_social_asset_image_prompt_lines(
            normalized_text,
            expected_count=expected_count,
        )
        if numbered_section_prompts and (
            expected_count <= 0 or len(numbered_section_prompts) >= expected_count
        ):
            if (
                filename_social_asset_prompts
                and (expected_count <= 0 or len(filename_social_asset_prompts) >= expected_count)
                and _image_prompt_batch_looks_like_weak_social_copy(numbered_section_prompts)
            ):
                return (
                    filename_social_asset_prompts[:expected_count]
                    if expected_count > 0
                    else filename_social_asset_prompts
                )
            return numbered_section_prompts
        table_section_prompts = _extract_markdown_table_image_prompt_section(
            normalized_text,
            expected_count=expected_count,
        )
        if table_section_prompts and (
            expected_count <= 0 or len(table_section_prompts) >= expected_count
        ):
            return table_section_prompts[:expected_count] if expected_count > 0 else table_section_prompts
        table_anywhere_prompts = _extract_markdown_table_image_prompt_anywhere(
            normalized_text,
            expected_count=expected_count,
        )
        if table_anywhere_prompts and (
            expected_count <= 0 or len(table_anywhere_prompts) >= expected_count
        ):
            return table_anywhere_prompts[:expected_count] if expected_count > 0 else table_anywhere_prompts
        if filename_social_asset_prompts and (
            expected_count <= 0 or len(filename_social_asset_prompts) >= expected_count
        ):
            return filename_social_asset_prompts[:expected_count] if expected_count > 0 else filename_social_asset_prompts
        inline_labeled_prompts = _extract_inline_labeled_image_prompt_lines(
            normalized_text,
            expected_count=expected_count,
        )
        if inline_labeled_prompts and (
            expected_count <= 0 or len(inline_labeled_prompts) >= expected_count
        ):
            return inline_labeled_prompts[:expected_count] if expected_count > 0 else inline_labeled_prompts
        sequential_alpha_prompts = _extract_sequential_bold_alpha_image_prompt_lines(
            normalized_text,
            expected_count=expected_count,
            require_image_heading=True,
        )
        if sequential_alpha_prompts and (
            expected_count <= 0 or len(sequential_alpha_prompts) >= expected_count
        ):
            return sequential_alpha_prompts[:expected_count] if expected_count > 0 else sequential_alpha_prompts
        if allow_plain_alpha_sequence:
            plain_alpha_prompts = _extract_leading_plain_alpha_image_prompt_lines(
                normalized_text,
                expected_count=expected_count,
            )
            if plain_alpha_prompts:
                return plain_alpha_prompts
        social_manifest_prompts = _extract_social_manifest_pipe_image_prompt_lines(
            normalized_text,
            expected_count=expected_count,
        )
        if social_manifest_prompts and (
            expected_count <= 0 or len(social_manifest_prompts) >= expected_count
        ):
            return social_manifest_prompts[:expected_count] if expected_count > 0 else social_manifest_prompts
        numbered_prepared_prompts = _extract_numbered_prepared_image_prompt_units(
            normalized_text,
            expected_count=expected_count,
        )
        if numbered_prepared_prompts and (
            expected_count <= 0 or len(numbered_prepared_prompts) >= expected_count
        ):
            return numbered_prepared_prompts[:expected_count] if expected_count > 0 else numbered_prepared_prompts
        for prompt in table_section_prompts:
            if prompt and prompt not in prompts:
                prompts.append(prompt)
            if expected_count > 0 and len(prompts) >= expected_count:
                break
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        for prompt in table_anywhere_prompts:
            if prompt and prompt not in prompts:
                prompts.append(prompt)
            if expected_count > 0 and len(prompts) >= expected_count:
                break
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        for prompt in filename_social_asset_prompts:
            if prompt:
                prompts.append(prompt)
            if expected_count > 0 and len(prompts) >= expected_count:
                break
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        if expected_count <= 1:
            for prompt in social_manifest_prompts:
                if prompt and prompt not in prompts:
                    prompts.append(prompt)
                if expected_count > 0 and len(prompts) >= expected_count:
                    break
            if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
                return prompts[:expected_count] if expected_count > 0 else prompts
        for prompt in inline_labeled_prompts:
            if prompt and prompt not in prompts:
                prompts.append(prompt)
            if expected_count > 0 and len(prompts) >= expected_count:
                break
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        for prompt in self._extract_labeled_multi_image_prompts(
            normalized_text,
            expected_count=expected_count,
        ):
            if prompt and prompt not in prompts:
                prompts.append(prompt)
            if expected_count > 0 and len(prompts) >= expected_count:
                break
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        for prompt in _extract_html_image_card_prompt_units(
            normalized_text,
            expected_count=expected_count,
        ):
            if prompt and prompt not in prompts:
                prompts.append(prompt)
            if expected_count > 0 and len(prompts) >= expected_count:
                break
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        for match in _MULTI_IMAGE_PROMPT_SECTION_RE.finditer(normalized_text):
            body = str(match.group('body') or '').strip()
            if not body:
                continue
            extracted = split_visible_image_payload(body)
            prompt = _clean_image_prompt_boundary_candidate(extracted.get('artifact_prompt') or body)
            if prompt and prompt not in prompts:
                prompts.append(prompt)
        if plain_alpha_candidate_present:
            return []
        if len(prompts) <= 1 and expected_count > 1:
            for paragraph in _PARAGRAPH_BREAK_RE.split(normalized_text):
                body = str(paragraph or '').strip()
                if not body:
                    continue
                extracted = split_visible_image_payload(body)
                prompt = _clean_image_prompt_boundary_candidate(extracted.get('artifact_prompt') or body)
                if prompt and prompt not in prompts:
                    prompts.append(prompt)
                if len(prompts) >= expected_count:
                    break
        if expected_count > 0 and len(prompts) > expected_count:
            return prompts[:expected_count]
        return prompts

    def _extract_labeled_multi_image_prompts(
        self,
        prepared_text: str,
        *,
        expected_count: int = 0,
    ) -> list[str]:
        lines = str(prepared_text or '').splitlines()
        prompts: list[str] = []
        current_lines: list[str] = []
        active_heading: Optional[str] = None

        def flush_current_section() -> None:
            body = '\n'.join(current_lines).strip()
            if not body:
                return
            extracted = split_visible_image_payload(body)
            prompt = _clean_image_prompt_boundary_candidate(extracted.get('artifact_prompt') or body)
            if prompt and prompt not in prompts:
                prompts.append(prompt)

        for raw_line in lines:
            inline_payload = split_visible_image_payload(str(raw_line or '').strip())
            inline_prompt = str(inline_payload.get('artifact_prompt') or '').strip()
            inline_source = str(inline_payload.get('artifact_prompt_source') or '').strip()
            if inline_prompt and inline_source and inline_source != 'full_display_text':
                if active_heading is not None:
                    flush_current_section()
                active_heading = None
                current_lines = []
                if inline_prompt not in prompts:
                    prompts.append(inline_prompt)
                if expected_count > 0 and len(prompts) >= expected_count:
                    break
                continue
            heading = _normalize_handoff_heading_line(raw_line)
            if heading and _MULTI_IMAGE_HEADING_LINE_RE.fullmatch(heading):
                if active_heading is not None:
                    flush_current_section()
                active_heading = heading
                current_lines = []
                continue
            if active_heading is None:
                continue
            if _is_image_prompt_section_stop(raw_line):
                flush_current_section()
                active_heading = None
                current_lines = []
                break
            current_lines.append(str(raw_line or ''))
        if active_heading is not None:
            flush_current_section()
        if prompts and (expected_count <= 0 or len(prompts) >= expected_count):
            return prompts[:expected_count] if expected_count > 0 else prompts
        if expected_count > 0 and len(prompts) > expected_count:
            return prompts[:expected_count]
        return prompts

    def _expand_image_pending_branches_from_batch_prompts(
        self,
        pending_branches: list[dict[str, Any]],
        batch_prompts: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompts = [
            str(item).strip()
            for item in (batch_prompts or [])
            if str(item).strip()
        ] if isinstance(batch_prompts, list) else []
        if len(prompts) <= 1:
            return pending_branches, {}
        expanded = [
            dict(item)
            for item in (pending_branches or [])
            if isinstance(item, Mapping)
        ]
        image_branches = [
            item
            for item in expanded
            if normalize_capability(item.get('capability')) == CAPABILITY_IMAGE_GENERATION
        ]
        existing_count = len(image_branches)
        if existing_count >= len(prompts) or existing_count <= 0:
            return expanded, {}
        seen_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in expanded
            if str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        template = dict(image_branches[-1])

        def sibling_identifier(seed: Any, index: int, fallback_prefix: str) -> str:
            token = str(seed or '').strip()
            match = re.match(r'^(?P<prefix>.*?)(?P<number>\d+)$', token)
            if match and match.group('prefix'):
                candidate = f"{match.group('prefix')}{index}"
            else:
                candidate = f'{fallback_prefix}{index}'
            base = candidate
            suffix = 2
            while candidate in seen_ids:
                candidate = f'{base}-{suffix}'
                suffix += 1
            seen_ids.add(candidate)
            return candidate

        for prompt_index in range(existing_count + 1, len(prompts) + 1):
            branch_id = sibling_identifier(
                template.get('branch_id') or 'branch-image_generation-1',
                prompt_index,
                'branch-image_generation-',
            )
            phase_id = sibling_identifier(
                template.get('phase_id') or 'phase-2',
                prompt_index + 1,
                'phase-',
            )
            obligation_id = ''
            raw_obligation_id = str(template.get('obligation_id') or '').strip()
            if raw_obligation_id:
                obligation_id = sibling_identifier(
                    raw_obligation_id,
                    prompt_index + 1,
                    'obligation-phase-',
                )
            branch = {
                'branch_id': branch_id,
                'phase_id': phase_id,
                'capability': CAPABILITY_IMAGE_GENERATION,
                'output_type': 'image',
                'depends_on': [
                    str(item).strip()
                    for item in (template.get('depends_on') or ['phase-1'])
                    if str(item).strip()
                ],
                'queue_index': prompt_index,
                'status': str(template.get('status') or 'pending').strip().lower() or 'pending',
                'branch_expansion_source': 'prepared_image_prompt_units',
                'branch_expansion_prompt_index': prompt_index,
                'branch_expansion_original_branch_count': existing_count,
            }
            if obligation_id:
                branch['obligation_id'] = obligation_id
            for key in (
                'stage_direction',
                'content_payload',
                'content_payload_source',
                'phase_summary',
                'requires_artifact',
                'runtime_scheduling_context',
                'allow_gpu_heavy_concurrency',
            ):
                value = template.get(key)
                if value not in (None, '', [], {}):
                    branch[key] = value
            expanded.append(branch)
        return expanded, {
            'status': 'applied',
            'source': 'prepared_image_prompt_units',
            'original_image_branch_count': existing_count,
            'prepared_prompt_count': len(prompts),
            'added_branch_count': len(prompts) - existing_count,
        }

    def _extract_pending_deferred_capabilities(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        pending_branches = self._extract_pending_deferred_branches(
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )
        return self._hook('normalize_capability_list')(
            [item.get('capability') for item in pending_branches if isinstance(item, Mapping)]
        )

    def _extract_pending_deferred_branches(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        normalize_capability_list = self._hook('normalize_capability_list')

        route_info = route_payload if isinstance(route_payload, dict) else {}
        artifact_info = artifact_payload if isinstance(artifact_payload, dict) else {}
        artifact_late_fill = (
            artifact_info.get('late_fill')
            if isinstance(artifact_info.get('late_fill'), dict)
            else {}
        )
        pending_from_state = artifact_late_fill.get('pending_branches')
        if isinstance(pending_from_state, list) and pending_from_state:
            return [dict(item) for item in pending_from_state if isinstance(item, Mapping)]
        completed_capabilities = set(
            normalize_capability_list(artifact_late_fill.get('completed_capabilities'))
        )
        completed_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in (artifact_late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        failed_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in (artifact_late_fill.get('failed_branches') or [])
            if isinstance(item, Mapping) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        candidates: list[dict[str, Any]] = []
        candidate_branch_ids: set[str] = set()
        route_runtime = (
            route_info.get('route_runtime')
            if isinstance(route_info.get('route_runtime'), dict)
            else {}
        )
        artifact_runtime = (
            artifact_info.get('runtime')
            if isinstance(artifact_info.get('runtime'), dict)
            else {}
        )
        artifact_request_phase_graph = (
            artifact_runtime.get('request_phase_graph')
            if isinstance(artifact_runtime.get('request_phase_graph'), dict)
            else None
        )
        route_request_phase_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else None
        )
        request_phase_graph = (
            artifact_request_phase_graph
            if artifact_request_phase_graph
            else route_request_phase_graph
        )
        defer_downstream_execution = self._request_graph_defers_downstream_execution(request_phase_graph)
        execution_planner = (
            route_runtime.get('execution_planner')
            if isinstance(route_runtime.get('execution_planner'), dict)
            else (
                ((artifact_info.get('runtime') or {}).get('execution_planner'))
                if isinstance(artifact_info.get('runtime'), dict)
                and isinstance((artifact_info.get('runtime') or {}).get('execution_planner'), dict)
                else {}
            )
        )
        for raw_branch in (
            execution_planner.get('deferred_branches')
            if isinstance(execution_planner.get('deferred_branches'), list)
            else []
        ):
            if not isinstance(raw_branch, Mapping):
                continue
            if (
                defer_downstream_execution
                or self._branch_record_is_non_executable_candidate(raw_branch)
                or self._branch_record_is_terminally_fulfilled(raw_branch)
            ):
                continue
            capability = normalize_capability(raw_branch.get('capability'))
            branch_id = str(raw_branch.get('branch_id') or raw_branch.get('phase_id') or '').strip()
            phase_id = str(raw_branch.get('phase_id') or raw_branch.get('branch_id') or '').strip()
            identity = branch_id or phase_id
            if not capability:
                continue
            if not identity:
                continue
            if identity in completed_branch_ids or identity in failed_branch_ids:
                continue
            if identity in candidate_branch_ids:
                continue
            candidate_branch_ids.add(identity)
            candidates.append(
                {
                    'branch_id': branch_id or phase_id,
                    'phase_id': phase_id or branch_id,
                    'capability': capability,
                    'output_type': str(raw_branch.get('output_type') or '').strip().lower() or self.artifact_type_for_capability(capability),
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
                            'requires_artifact',
                            'text_artifact_extension',
                            'text_artifact_source_name',
                            'text_artifact_source',
                            'text_artifact_target_path',
                            'artifact_request',
                        )
                        if raw_branch.get(key) not in (None, '', [], {})
                    },
                }
            )
        for candidate in downstream_phase_records(request_phase_graph or {}):
            if (
                defer_downstream_execution
                or self._branch_record_is_non_executable_candidate(candidate)
                or self._branch_record_is_terminally_fulfilled(candidate)
            ):
                continue
            capability = normalize_capability(candidate.get('capability'))
            branch_id = str(candidate.get('branch_id') or candidate.get('phase_id') or '').strip()
            phase_id = str(candidate.get('phase_id') or candidate.get('branch_id') or '').strip()
            identity = branch_id or phase_id
            if not capability:
                continue
            if not identity:
                continue
            if identity in completed_branch_ids or identity in failed_branch_ids:
                continue
            if identity in candidate_branch_ids:
                continue
            candidate_branch_ids.add(identity)
            candidates.append(
                {
                    'branch_id': branch_id or phase_id,
                    'phase_id': phase_id or branch_id,
                    'capability': capability,
                    'output_type': str(candidate.get('output_type') or '').strip().lower() or self.artifact_type_for_capability(capability),
                    'depends_on': [
                        str(item).strip()
                        for item in (candidate.get('depends_on') or [])
                        if str(item).strip()
                    ],
                    **{
                        key: candidate.get(key)
                        for key in (
                            'queue_index',
                            'artifact_prompt',
                            'artifact_prompt_source',
                            'content_payload',
                            'content_payload_source',
                            'phase_summary',
                            'stage_direction',
                            'batch_prompts',
                            'requires_artifact',
                            'text_artifact_extension',
                            'text_artifact_source_name',
                            'text_artifact_source',
                            'text_artifact_target_path',
                            'artifact_request',
                        )
                        if candidate.get(key) not in (None, '', [], {})
                    },
                }
            )
        for candidate in output_obligations_from_graph(request_phase_graph or {}):
            if str(candidate.get('role') or '').strip() == 'preparation_output':
                continue
            if (
                defer_downstream_execution
                or self._branch_record_is_non_executable_candidate(candidate)
                or self._branch_record_is_terminally_fulfilled(candidate)
            ):
                continue
            graph_current_phase_id = ''
            if isinstance(request_phase_graph, Mapping):
                graph_current_phase_id = str(request_phase_graph.get('current_phase_id') or '').strip()
            capability = normalize_capability(candidate.get('capability'))
            branch_id = str(candidate.get('branch_id') or candidate.get('phase_id') or '').strip()
            phase_id = str(candidate.get('phase_id') or candidate.get('branch_id') or '').strip()
            if graph_current_phase_id and phase_id == graph_current_phase_id:
                continue
            identity = branch_id or phase_id
            if not capability or not identity:
                continue
            if identity in completed_branch_ids or identity in failed_branch_ids:
                continue
            if identity in candidate_branch_ids:
                continue
            candidate_branch_ids.add(identity)
            candidates.append(
                {
                    'branch_id': branch_id or phase_id,
                    'phase_id': phase_id or branch_id,
                    'obligation_id': str(candidate.get('obligation_id') or '').strip() or None,
                    'capability': capability,
                    'output_type': str(candidate.get('output_type') or '').strip().lower() or self.artifact_type_for_capability(capability),
                    'depends_on': [
                        str(item).strip()
                        for item in (candidate.get('depends_on') or [])
                        if str(item).strip()
                    ],
                    **{
                        key: candidate.get(key)
                        for key in (
                            'queue_index',
                            'artifact_prompt',
                            'artifact_prompt_source',
                            'content_payload',
                            'content_payload_source',
                            'phase_summary',
                            'stage_direction',
                            'batch_prompts',
                            'requires_artifact',
                            'text_artifact_extension',
                            'text_artifact_source_name',
                            'text_artifact_source',
                            'text_artifact_target_path',
                            'artifact_request',
                        )
                        if candidate.get(key) not in (None, '', [], {})
                    },
                }
            )
        if not candidates:
            if not defer_downstream_execution:
                for candidate in normalize_capability_list(execution_planner.get('deferred_capabilities')):
                    if candidate in completed_capabilities:
                        continue
                    if candidate in {normalize_capability(item.get('capability')) for item in candidates}:
                        continue
                    candidates.append(
                        {
                            'branch_id': f'branch-{candidate}-{len(candidates) + 1}',
                            'phase_id': f'branch-{candidate}-{len(candidates) + 1}',
                            'capability': candidate,
                            'output_type': self.artifact_type_for_capability(candidate),
                            'depends_on': ['phase-1'],
                        }
                    )
                deferred_capability = normalize_capability(execution_planner.get('deferred_capability'))
                if deferred_capability and deferred_capability not in completed_capabilities and deferred_capability not in {
                    normalize_capability(item.get('capability')) for item in candidates
                }:
                    candidates.append(
                        {
                            'branch_id': f'branch-{deferred_capability}-{len(candidates) + 1}',
                            'phase_id': f'branch-{deferred_capability}-{len(candidates) + 1}',
                            'capability': deferred_capability,
                            'output_type': self.artifact_type_for_capability(deferred_capability),
                            'depends_on': ['phase-1'],
                        }
                    )
        return candidates

    def extract_pending_deferred_branches(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        return self._extract_pending_deferred_branches(
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )

    def extract_pending_deferred_capabilities(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        return self._extract_pending_deferred_capabilities(
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )

    def build_planner_deferred_follow_up_gap_spec(
        self,
        output_text: str,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        normalize_capability_list = self._hook('normalize_capability_list')

        route_info = route_payload if isinstance(route_payload, dict) else {}
        route_runtime = (
            route_info.get('route_runtime')
            if isinstance(route_info.get('route_runtime'), dict)
            else {}
        )
        pending_branches = self._extract_pending_deferred_branches(
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )
        if not pending_branches:
            return None
        pending_capabilities = normalize_capability_list(
            [item.get('capability') for item in pending_branches if isinstance(item, Mapping)]
        )
        for raw_expected_branch in pending_branches:
            if not isinstance(raw_expected_branch, Mapping):
                continue
            expected_branch = dict(raw_expected_branch)
            expected_capability = normalize_capability(expected_branch.get('capability'))
            missing_artifact_type = self.artifact_type_for_capability(expected_capability)
            if not missing_artifact_type and self.source_requires_text_artifact(
                expected_branch,
                str(expected_branch.get('output_type') or '').strip().lower(),
            ):
                missing_artifact_type = 'text'
            if not missing_artifact_type:
                continue
            semantic_payload = self.semantic_payload_for_capability(
                artifact_payload,
                capability=expected_capability,
            )
            payload = {
                'code': 'deferred_follow_up_fill',
                'trigger': 'execution_planner_deferred_follow_up',
                'expected_branch_id': str(expected_branch.get('branch_id') or expected_branch.get('phase_id') or '').strip() or None,
                'expected_phase_id': str(expected_branch.get('phase_id') or expected_branch.get('branch_id') or '').strip() or None,
                'expected_capability': expected_capability,
                'missing_artifact_type': missing_artifact_type,
                'pending_branches': pending_branches,
                'pending_capabilities': pending_capabilities,
                'completed_capabilities': normalize_capability_list(
                    (
                        (artifact_payload or {}).get('late_fill', {})
                        if isinstance((artifact_payload or {}).get('late_fill'), dict)
                        else {}
                    ).get('completed_capabilities')
                ),
                'planner_reason': str(
                    (
                        route_runtime.get('execution_planner')
                        if isinstance(route_runtime.get('execution_planner'), dict)
                        else {}
                    ).get('reason') or ''
                ).strip() or None,
            }
            for key in (
                'stage_direction',
                'phase_summary',
                'requires_artifact',
                'text_artifact_extension',
                'text_artifact_source_name',
                'text_artifact_source',
                'text_artifact_target_path',
                'artifact_request',
            ):
                value = expected_branch.get(key)
                if value not in (None, '', [], {}):
                    payload[key] = value
            if semantic_payload:
                for key in ('content_payload', 'stage_direction', 'phase_summary', 'content_payload_source'):
                    value = semantic_payload.get(key)
                    if value not in (None, '', [], {}):
                        payload[key] = value
                for key in (
                    'content_payload',
                    'phase_summary',
                    'artifact_prompt',
                    'artifact_prompt_source',
                    'batch_prompts',
                    'batch_prompts_source',
                    'batch_prompt_source_phase_id',
                    'batch_prompt_expected_count',
                ):
                    value = semantic_payload.get(key)
                    if value not in (None, '', [], {}):
                        payload[key] = value
                if self.artifact_gap_is_already_fulfilled(payload, artifact_payload):
                    continue
                expanded_branches, expansion = self._expand_image_pending_branches_from_batch_prompts(
                    pending_branches,
                    payload.get('batch_prompts'),
                )
                if expansion:
                    payload['pending_branches'] = expanded_branches
                    payload['pending_capabilities'] = normalize_capability_list(
                        [item.get('capability') for item in expanded_branches]
                    )
                    payload['prepared_image_prompt_branch_expansion'] = expansion
                return payload
            if str(output_text or '').strip():
                if expected_capability == CAPABILITY_TEXT_TO_SPEECH:
                    fallback_payload = split_visible_tts_payload(output_text)
                    for key in ('content_payload', 'stage_direction', 'phase_summary', 'content_payload_source'):
                        value = fallback_payload.get(key)
                        if value not in (None, '', [], {}):
                            payload[key] = value
                elif expected_capability == CAPABILITY_IMAGE_GENERATION:
                    fallback_payload = split_visible_image_payload(output_text)
                    for key in ('content_payload', 'phase_summary', 'artifact_prompt', 'artifact_prompt_source'):
                        value = fallback_payload.get(key)
                        if value not in (None, '', [], {}):
                            payload[key] = value
                    expected_count = 0
                    request_phase_graph = (
                        route_runtime.get('request_phase_graph')
                        if isinstance(route_runtime.get('request_phase_graph'), Mapping)
                        else {}
                    )
                    if isinstance(request_phase_graph.get('prompt_intent'), Mapping):
                        expected_count = int(
                            request_phase_graph.get('prompt_intent', {}).get('requested_visual_output_count') or 0
                        )
                    plain_alpha_authority = self.plain_alpha_image_prompt_prepare_authority(
                        request_phase_graph,
                        expected_count=expected_count,
                    )
                    batch_prompts = self.extract_batch_image_prompts(
                        output_text,
                        expected_count=expected_count,
                        allow_plain_alpha_sequence=bool(plain_alpha_authority),
                    )
                    if len(batch_prompts) > 1:
                        payload['batch_prompts'] = batch_prompts
                        payload['batch_prompt_expected_count'] = (
                            expected_count if expected_count >= 2 else len(batch_prompts)
                        )
                        if plain_alpha_authority:
                            payload['batch_prompts_source'] = 'semantic_prepare_phase_output'
                            payload['batch_prompt_source_phase_id'] = plain_alpha_authority[
                                'source_phase_id'
                            ]
                        artifact_prompt = str(payload.get('artifact_prompt') or '').strip()
                        artifact_prompt_clean = _clean_image_prompt_boundary_candidate(artifact_prompt)
                        if (
                            str(payload.get('artifact_prompt_source') or '').strip() == 'full_display_text'
                            or (artifact_prompt and not artifact_prompt_clean)
                        ):
                            payload['artifact_prompt'] = batch_prompts[0]
                            payload['artifact_prompt_source'] = 'semantic_batch_prompts'
                    expanded_branches, expansion = self._expand_image_pending_branches_from_batch_prompts(
                        pending_branches,
                        payload.get('batch_prompts'),
                    )
                    if expansion:
                        payload['pending_branches'] = expanded_branches
                        payload['pending_capabilities'] = normalize_capability_list(
                            [item.get('capability') for item in expanded_branches]
                        )
                        payload['prepared_image_prompt_branch_expansion'] = expansion
            if self.artifact_gap_is_already_fulfilled(payload, artifact_payload):
                continue
            return payload
        return None

    def _extract_expected_non_chat_capability_from_route(
        self,
        route_payload: Optional[dict[str, Any]],
    ) -> Optional[str]:
        normalize_capability_list = self._hook('normalize_capability_list')

        route_info = route_payload if isinstance(route_payload, dict) else {}
        route_runtime = (
            route_info.get('route_runtime')
            if isinstance(route_info.get('route_runtime'), dict)
            else {}
        )
        embedding_audit = (
            route_runtime.get('embedding_audit')
            if isinstance(route_runtime.get('embedding_audit'), dict)
            else {}
        )
        execution_planner = (
            route_runtime.get('execution_planner')
            if isinstance(route_runtime.get('execution_planner'), dict)
            else {}
        )
        request_phase_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else {}
        )
        if self._request_graph_defers_downstream_execution(request_phase_graph):
            return None
        request_meta = (
            route_runtime.get('request_meta')
            if isinstance(route_runtime.get('request_meta'), dict)
            else {}
        )
        for candidate in (
            execution_planner.get('deferred_capability'),
            *(normalize_capability_list(execution_planner.get('deferred_capabilities'))),
            *(normalize_capability_list(downstream_phase_capabilities(request_phase_graph))),
            request_meta.get('capability_hint'),
            embedding_audit.get('route_hint_capability') or embedding_audit.get('heuristic_capability'),
        ):
            normalized_candidate = normalize_capability(candidate)
            if normalized_candidate and normalized_candidate != CAPABILITY_CHAT:
                return normalized_candidate
        return None

    def _extract_expected_non_chat_capability(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        extract_request_meta = self._hook('extract_request_meta')

        route_capability = self._extract_expected_non_chat_capability_from_route(route_payload)
        if route_capability:
            return route_capability
        request_meta = extract_request_meta(request_payload or {})
        request_meta_hint = normalize_capability((request_meta or {}).get('capability_hint'))
        if request_meta_hint and request_meta_hint != CAPABILITY_CHAT:
            return request_meta_hint
        return None

    def build_artifact_completion_gap_spec(
        self,
        output_text: str,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        text = str(output_text or '').strip()
        if not text:
            return None
        expected_capability = self._extract_expected_non_chat_capability(
            route_payload=route_payload,
            request_payload=request_payload,
        )
        if expected_capability == CAPABILITY_IMAGE_GENERATION and _TEXT_ONLY_IMAGE_COMPLETION_RE.search(text):
            candidate_gap = {
                'code': 'late_artifact_fill',
                'expected_capability': CAPABILITY_IMAGE_GENERATION,
                'missing_artifact_type': 'image',
            }
            if not self.artifact_gap_is_already_fulfilled(candidate_gap, artifact_payload):
                return candidate_gap
        if expected_capability == CAPABILITY_TEXT_TO_SPEECH and _TEXT_ONLY_AUDIO_COMPLETION_RE.search(text):
            candidate_gap = {
                'code': 'late_artifact_fill',
                'expected_capability': CAPABILITY_TEXT_TO_SPEECH,
                'missing_artifact_type': 'audio',
            }
            if not self.artifact_gap_is_already_fulfilled(candidate_gap, artifact_payload):
                return candidate_gap
        planner_deferred_gap = self.build_planner_deferred_follow_up_gap_spec(
            text,
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )
        if planner_deferred_gap:
            if self.artifact_gap_is_already_fulfilled(planner_deferred_gap, artifact_payload):
                return None
            return planner_deferred_gap
        return None

    def _explicit_recovery_action_from_payload(self, payload: Mapping[str, Any]) -> Optional[str]:
        for source_key in (
            'repair_action',
            'recovery_action',
            'suggested_action',
            'suggestedAction',
        ):
            action = str(payload.get(source_key) or '').strip()
            if action:
                return normalize_recovery_suggested_action(action)
        for container_key in ('recovery_state', 'recovery_context'):
            container = payload.get(container_key)
            if isinstance(container, Mapping):
                action = self._explicit_recovery_action_from_payload(container)
                if action:
                    return action
        return None

    def _closure_check_recovery_action(self, check: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
        explicit_action = self._explicit_recovery_action_from_payload(check)
        if explicit_action:
            return explicit_action, 'explicit recovery action carried by runtime branch state'

        status = str(check.get('status') or '').strip().lower()
        check_kind = str(check.get('check_kind') or '').strip().lower()
        evidence = str(check.get('evidence') or '').strip().lower()
        capability = normalize_capability(check.get('capability'))
        depends_on = [
            str(item or '').strip()
            for item in (check.get('depends_on') or [])
            if str(item or '').strip()
        ] if isinstance(check.get('depends_on'), list) else []

        if check_kind == 'intent_graph_adequacy' or evidence.startswith('intent_graph_adequacy_'):
            return (
                RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
                'request graph does not contain enough promoted obligations for the current intent',
            )
        if evidence == 'runtime_capability_unavailable':
            return (
                RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
                'required capability is not currently available in runtime',
            )
        if (
            check.get('blocked_by_dependency_input') is True
            or 'dependency' in evidence
            or (
                status == 'blocked'
                and depends_on
                and evidence in {'late_fill_failed_branch', 'late_fill_failed_capability'}
            )
        ):
            if self._repair_item_has_concrete_input_evidence(check):
                return (
                    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
                    'branch has concrete dependency evidence that can be rebound to resolve the block',
                )
            return (
                RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                'branch is blocked by missing or failed dependency evidence',
            )
        if status == 'blocked' and evidence in {'late_fill_failed_branch', 'late_fill_failed_capability'}:
            if check.get('failed_instance_id') or check.get('exclude_instance_ids'):
                return (
                    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
                    'branch failed after selecting a concrete runtime instance',
                )
            return (
                RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'branch failed without dependency evidence; retry the same bounded branch',
            )
        if (
            check.get('blocked_by_branch_contract') is True
            or (
                status in {'pending', 'planned', 'active', 'deferred', 'blocked'}
                and capability
                and capability != CAPABILITY_CHAT
                and not isinstance(check.get('execution_contract'), Mapping)
                and not check.get('branch_id')
            )
        ):
            return (
                RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                'branch has material work but lacks a bounded executable branch contract',
            )
        if check_kind in {'truth_guard', 'truth_guard_capability'}:
            return (
                RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                'runtime truth detected an unmaterialized output claim that needs a promoted branch contract',
            )
        if (
            check.get('semantic_review_required') is True
            and status in {'pending', 'blocked'}
        ):
            return (
                RECOVERY_ACTION_SEMANTIC_REVIEW,
                'branch has qualitative review criteria that need semantic evaluation before freeze',
            )
        if status == 'blocked':
            return RECOVERY_ACTION_MANUAL_REVIEW, 'blocked branch has no safe automatic repair path'
        return None, None

    @staticmethod
    def _workload_task_indexes(request_phase_graph: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
        request_ir = (
            request_phase_graph.get('request_ir')
            if isinstance(request_phase_graph.get('request_ir'), Mapping)
            else {}
        )
        workload_graph = (
            request_phase_graph.get('workload_graph')
            if isinstance(request_phase_graph.get('workload_graph'), Mapping)
            else (
                request_ir.get('workload_graph')
                if isinstance(request_ir.get('workload_graph'), Mapping)
                else {}
            )
        )
        tasks = workload_graph.get('tasks') if isinstance(workload_graph.get('tasks'), list) else []
        by_phase: dict[str, Mapping[str, Any]] = {}
        by_branch: dict[str, Mapping[str, Any]] = {}
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            phase_id = str(task.get('phase_id') or '').strip()
            branch_id = str(task.get('branch_id') or phase_id or '').strip()
            if phase_id:
                by_phase[phase_id] = task
            if branch_id:
                by_branch[branch_id] = task
        return by_phase, by_branch

    @staticmethod
    def _check_has_dependency_runtime_evidence(check: Mapping[str, Any]) -> bool:
        source = str(check.get('content_payload_source') or '').strip()
        if (
            source.startswith('late_fill_result')
            or source.startswith('late_fill_results')
            or source in {'late_fill_dependency_artifacts', 'prior_artifact_result'}
        ):
            return True
        evidence = str(check.get('evidence') or '').strip()
        if evidence.startswith('late_fill_completed'):
            return True
        if str(check.get('file_path') or '').strip():
            return True
        for key in ('input_artifacts', 'reference_artifacts'):
            if isinstance(check.get(key), list) and check.get(key):
                return True
        return False

    @staticmethod
    def _sequential_structured_join_labels(values: list[str]) -> list[str]:
        labels: list[str] = []
        for value in values:
            label = str(value or '').strip()
            if label and label not in labels:
                labels.append(label)
        if len(labels) < 2:
            return []
        if all(len(label) == 1 and label.isalpha() for label in labels):
            if not (all(label.isupper() for label in labels) or all(label.islower() for label in labels)):
                return []
            first = ord(labels[0])
            return labels if all(
                ord(label) == first + index
                for index, label in enumerate(labels)
            ) else []
        if all(label.isdecimal() for label in labels):
            numeric_labels = [int(label) for label in labels]
            first = numeric_labels[0]
            return labels if all(
                label == first + index
                for index, label in enumerate(numeric_labels)
            ) else []
        return []

    @classmethod
    def _structured_dependency_join_prompt_contract(cls, prompt: str) -> dict[str, Any]:
        text = str(prompt or '').strip()
        if not text or not _STRUCTURED_JSON_LIST_CONTRACT_RE.search(text):
            return {}

        label_candidates: list[list[str]] = []
        enumerated = [
            str(match.group('label') or '').strip()
            for match in _STRUCTURED_JOIN_ENUMERATED_LABEL_RE.finditer(text)
        ]
        if enumerated:
            label_candidates.append(enumerated)
        for match in _STRUCTURED_JOIN_FOR_LABELS_RE.finditer(text):
            label_candidates.append(
                re.findall(
                    r'(?<![A-Za-z0-9_])([A-Za-z]|[0-9]+)(?![A-Za-z0-9_])',
                    str(match.group('body') or ''),
                )
            )
        sequential_candidates = [
            labels
            for labels in (
                cls._sequential_structured_join_labels(candidate)
                for candidate in label_candidates
            )
            if labels
        ]
        if not sequential_candidates:
            return {}
        labels = max(sequential_candidates, key=len)

        field_candidates: list[list[str]] = []
        for match in _STRUCTURED_JOIN_FIELDS_RE.finditer(text):
            body = re.split(
                r'\b(?:(?i:für|for))\b',
                str(match.group('body') or ''),
                maxsplit=1,
            )[0]
            body = re.sub(
                r'^\s*(?::|=)?\s*(?:(?i:are|sind)\s+)?',
                '',
                body,
            )
            body = re.sub(r'[`\'\"]', '', body)
            field_list_match = re.match(
                r'(?P<fields>[A-Za-z_][A-Za-z0-9_]*'
                r'(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*'
                r'(?:\s*,?\s*(?:(?i:und|and)|&)\s*[A-Za-z_][A-Za-z0-9_]*)?)',
                body,
            )
            field_text = (
                str(field_list_match.group('fields') or '')
                if field_list_match
                else ''
            )
            fields: list[str] = []
            for token in re.findall(
                r'(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])',
                field_text,
            ):
                if token.lower() in _STRUCTURED_JOIN_FIELD_STOP_WORDS:
                    continue
                if token not in fields:
                    fields.append(token)
            if fields:
                field_candidates.append(fields)
        required_fields = next(
            (
                fields
                for fields in field_candidates
                if any(field.lower() == 'label' for field in fields)
                and any(field.lower() in _STRUCTURED_JOIN_ARTIFACT_REF_FIELDS for field in fields)
            ),
            [],
        )
        if not required_fields:
            return {}
        label_field = next(field for field in required_fields if field.lower() == 'label')
        artifact_ref_field = next(
            field
            for field in required_fields
            if field.lower() in _STRUCTURED_JOIN_ARTIFACT_REF_FIELDS
        )
        return {
            'format': 'json_list',
            'labels': labels,
            'required_fields': required_fields,
            'label_field': label_field,
            'artifact_ref_field': artifact_ref_field,
        }

    @staticmethod
    def _structured_join_graph_records(
        request_phase_graph: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        ordered_records: list[dict[str, Any]] = []
        records_by_identity: dict[str, dict[str, Any]] = {}
        request_ir = (
            request_phase_graph.get('request_ir')
            if isinstance(request_phase_graph.get('request_ir'), Mapping)
            else {}
        )
        workload_graph = (
            request_phase_graph.get('workload_graph')
            if isinstance(request_phase_graph.get('workload_graph'), Mapping)
            else (
                request_ir.get('workload_graph')
                if isinstance(request_ir.get('workload_graph'), Mapping)
                else {}
            )
        )
        collections = (
            request_phase_graph.get('phases'),
            workload_graph.get('tasks'),
            request_phase_graph.get('downstream_branches'),
            output_obligations_from_graph(request_phase_graph),
        )
        for collection in collections:
            if not isinstance(collection, list):
                continue
            for raw_record in collection:
                if not isinstance(raw_record, Mapping):
                    continue
                phase_id = str(raw_record.get('phase_id') or '').strip()
                branch_id = str(raw_record.get('branch_id') or '').strip()
                identity = phase_id or branch_id
                if not identity:
                    continue
                record = records_by_identity.get(identity)
                if record is None:
                    record = dict(raw_record)
                    records_by_identity[identity] = record
                    ordered_records.append(record)
                else:
                    for key, value in raw_record.items():
                        if value not in (None, '', [], {}) and record.get(key) in (None, '', [], {}):
                            record[key] = value

        by_phase: dict[str, dict[str, Any]] = {}
        by_branch: dict[str, dict[str, Any]] = {}
        for record in ordered_records:
            phase_id = str(record.get('phase_id') or '').strip()
            branch_id = str(record.get('branch_id') or '').strip()
            if phase_id:
                by_phase[phase_id] = record
            if branch_id:
                by_branch[branch_id] = record
        return ordered_records, by_phase, by_branch

    @staticmethod
    def _structured_join_fill_result_for_record(
        fill_results: list[Any],
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        branch_id = str(record.get('branch_id') or '').strip()
        phase_id = str(record.get('phase_id') or '').strip()
        if branch_id:
            for result in reversed(fill_results):
                if (
                    isinstance(result, Mapping)
                    and str(result.get('branch_id') or '').strip() == branch_id
                ):
                    return dict(result)
        if phase_id:
            for result in reversed(fill_results):
                if (
                    isinstance(result, Mapping)
                    and str(result.get('phase_id') or '').strip() == phase_id
                ):
                    return dict(result)
        return {}

    @staticmethod
    def _structured_join_terminal_result_text(result: Mapping[str, Any]) -> str:
        for key in ('result_text', 'content', 'output_text', 'text'):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raw_result = result.get('result') if isinstance(result.get('result'), Mapping) else {}
        choices = raw_result.get('choices') if isinstance(raw_result.get('choices'), list) else []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get('message') if isinstance(choice.get('message'), Mapping) else {}
            content = message.get('content')
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ''

    @staticmethod
    def _structured_join_alias_tokens(value: Any) -> set[str]:
        raw = str(value or '').strip()
        if not raw:
            return set()
        tokens = {raw}
        path_value = raw[7:] if raw.startswith('file://') else raw
        if '/' in path_value or '\\' in path_value or path_value.startswith('.'):
            try:
                tokens.add(str(Path(path_value).expanduser().resolve(strict=False)))
            except (OSError, RuntimeError, ValueError):
                pass
        return tokens

    @classmethod
    def _structured_join_artifact_aliases(cls, record: Mapping[str, Any]) -> set[str]:
        aliases: set[str] = set()
        for key in (
            'artifact_ref',
            'ref',
            'path',
            'source_path',
            'saved_image_path',
            'saved_audio_path',
            'saved_text_path',
            'saved_video_path',
            'resolved_path',
            'url',
        ):
            aliases.update(cls._structured_join_alias_tokens(record.get(key)))
        for key in ('artifacts', 'input_artifacts', 'reference_artifacts'):
            values = record.get(key) if isinstance(record.get(key), list) else []
            for value in values:
                if isinstance(value, Mapping):
                    aliases.update(cls._structured_join_artifact_aliases(value))
        raw_result = record.get('result') if isinstance(record.get('result'), Mapping) else {}
        for artifact_key in ('image', 'audio'):
            raw_artifact = (
                raw_result.get(artifact_key)
                if isinstance(raw_result.get(artifact_key), Mapping)
                else {}
            )
            if raw_artifact:
                aliases.update(cls._structured_join_artifact_aliases(raw_artifact))
        return aliases

    @classmethod
    def _structured_join_artifact_path_aliases(cls, record: Mapping[str, Any]) -> set[str]:
        aliases: set[str] = set()
        for key in (
            'path',
            'source_path',
            'saved_image_path',
            'saved_audio_path',
            'saved_text_path',
            'saved_video_path',
            'resolved_path',
            'url',
        ):
            aliases.update(cls._structured_join_alias_tokens(record.get(key)))
        for key in ('artifacts', 'input_artifacts', 'reference_artifacts'):
            values = record.get(key) if isinstance(record.get(key), list) else []
            for value in values:
                if isinstance(value, Mapping):
                    aliases.update(cls._structured_join_artifact_path_aliases(value))
        raw_result = record.get('result') if isinstance(record.get('result'), Mapping) else {}
        for artifact_key in ('image', 'audio'):
            raw_artifact = (
                raw_result.get(artifact_key)
                if isinstance(raw_result.get(artifact_key), Mapping)
                else {}
            )
            if raw_artifact:
                aliases.update(cls._structured_join_artifact_path_aliases(raw_artifact))
        return aliases

    @staticmethod
    def _structured_join_artifact_matches_type(
        record: Mapping[str, Any],
        expected_type: str,
    ) -> bool:
        normalized_type = str(expected_type or '').strip().lower()
        artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
        extension = _extension_from_path_like(_artifact_path(record))
        aliases = {
            'image': {'image', 'png', 'jpg', 'jpeg', 'webp'},
            'audio': {'audio', 'wav', 'mp3', 'm4a', 'flac', 'ogg', 'opus'},
        }.get(normalized_type, {normalized_type})
        return artifact_type in aliases or extension in aliases

    @staticmethod
    def _structured_object_reference_records(
        *payloads: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            sources: list[Mapping[str, Any]] = [payload]
            request = payload.get('request')
            if isinstance(request, Mapping):
                sources.append(request)
            frame = payload.get('response_frame')
            frame_request = frame.get('request') if isinstance(frame, Mapping) else None
            if isinstance(frame_request, Mapping):
                sources.append(frame_request)
            for source in sources:
                for key in (
                    'selected_reference_artifacts',
                    'selectedReferenceArtifacts',
                    'reference_artifacts',
                    'input_artifacts',
                ):
                    values = source.get(key) if isinstance(source.get(key), list) else []
                    for value in values:
                        if not isinstance(value, Mapping):
                            continue
                        record = dict(value)
                        identity = (
                            str(record.get('type') or record.get('kind') or '').strip().lower(),
                            str(record.get('artifact_ref') or record.get('ref') or '').strip(),
                            str(record.get('path') or record.get('artifact_path') or '').strip(),
                            str(record.get('message_id') or '').strip(),
                            str(record.get('source_message_id') or '').strip(),
                            str(record.get('source_response_id') or '').strip(),
                            str(record.get('content') or record.get('text') or '').strip(),
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        records.append(record)
        return records

    @classmethod
    def _structured_object_source_artifact_refs(
        cls,
        source_result: Mapping[str, Any],
        source_record: Mapping[str, Any],
        artifact_payload: Mapping[str, Any],
    ) -> set[str]:
        source_phase_id = str(source_record.get('phase_id') or '').strip()
        source_branch_id = str(source_record.get('branch_id') or '').strip()
        source_path_aliases = cls._structured_join_artifact_path_aliases(source_result)

        def artifact_records(value: Any) -> list[Mapping[str, Any]]:
            records: list[Mapping[str, Any]] = []
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, Mapping))
            elif isinstance(value, Mapping):
                if value.get('artifact_ref') or value.get('ref') or _artifact_path(value):
                    records.append(value)
                for key in ('output', 'input', 'reference', 'artifacts'):
                    nested = value.get(key)
                    if isinstance(nested, list):
                        records.extend(item for item in nested if isinstance(item, Mapping))
            return records

        source_candidates = artifact_records(source_result.get('artifacts'))
        raw_result = source_result.get('result')
        if isinstance(raw_result, Mapping):
            for key in ('audio', 'image', 'artifact'):
                candidate = raw_result.get(key)
                if isinstance(candidate, Mapping):
                    source_candidates.append(candidate)
        direct_source_ref = str(
            source_result.get('artifact_ref') or source_result.get('ref') or ''
        ).strip()

        canonical_candidates = artifact_records(artifact_payload.get('artifacts'))
        canonical_refs: set[str] = set()
        for candidate in canonical_candidates:
            candidate_phase_id = str(candidate.get('phase_id') or '').strip()
            candidate_branch_id = str(candidate.get('branch_id') or '').strip()
            identity_match = bool(
                (source_phase_id and candidate_phase_id == source_phase_id)
                or (source_branch_id and candidate_branch_id == source_branch_id)
            )
            path_match = bool(
                source_path_aliases.intersection(
                    cls._structured_join_artifact_path_aliases(candidate)
                )
            )
            if not identity_match and not path_match:
                continue
            artifact_ref = str(
                candidate.get('artifact_ref') or candidate.get('ref') or ''
            ).strip()
            if artifact_ref:
                canonical_refs.add(artifact_ref)
        if canonical_refs:
            return canonical_refs

        source_refs = {
            str(candidate.get('artifact_ref') or candidate.get('ref') or '').strip()
            for candidate in source_candidates
            if str(candidate.get('artifact_ref') or candidate.get('ref') or '').strip()
        }
        if direct_source_ref:
            source_refs.add(direct_source_ref)
        return source_refs

    def _structured_output_object_contract_checks(
        self,
        *,
        request_payload: Optional[Mapping[str, Any]],
        artifact_payload: Optional[Mapping[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        ordered_records, records_by_phase, records_by_branch = (
            self._structured_join_graph_records(request_phase_graph)
        )
        terminal_contracts: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for record in ordered_records:
            contract = (
                record.get('structured_output_contract')
                if isinstance(record.get('structured_output_contract'), Mapping)
                else {}
            )
            if (
                normalize_capability(record.get('capability')) == CAPABILITY_CHAT
                and str(record.get('dependency_contract') or '').strip()
                == 'structured_multi_evidence_join'
                and str(contract.get('format') or '').strip().lower() == 'json_object'
                and str(contract.get('cardinality') or '').strip().lower() == 'exactly_one'
            ):
                terminal_contracts.append((record, dict(contract)))
        if not terminal_contracts:
            return []

        payload = artifact_payload if isinstance(artifact_payload, Mapping) else {}
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
        terminal, contract = terminal_contracts[0]
        terminal_result = self._structured_join_fill_result_for_record(fill_results, terminal)
        terminal_status = str(terminal.get('status') or '').strip().lower()
        if not terminal_result and terminal_status not in {'completed', 'fulfilled'}:
            return []

        terminal_branch_id = str(terminal.get('branch_id') or '').strip()
        terminal_phase_id = str(terminal.get('phase_id') or '').strip()
        bindings = [
            dict(item)
            for item in (contract.get('required_bindings') or [])
            if isinstance(item, Mapping)
        ]
        required_fields = [
            str(binding.get('field_name') or '').strip()
            for binding in bindings
            if str(binding.get('field_name') or '').strip()
        ]
        base_check: dict[str, Any] = {
            'check_kind': 'structured_dependency_join',
            'phase_id': terminal_phase_id or None,
            'branch_id': terminal_branch_id or None,
            'capability': CAPABILITY_CHAT,
            'output_type': 'text',
            'role': str(terminal.get('role') or '').strip() or None,
            'depends_on': list(terminal.get('depends_on') or []),
            'stage_direction': str(terminal.get('stage_direction') or '').strip() or None,
            'content_payload_source': str(
                terminal_result.get('content_payload_source')
                or terminal.get('content_payload_source')
                or f'late_fill_result:{terminal_branch_id or terminal_phase_id}'
            ).strip(),
            'terminal_result_source': 'late_fill.fill_results',
            'structured_output_contract': contract,
            'expected_count': 1,
            'required_fields': required_fields,
        }
        issues: list[dict[str, Any]] = []

        def add_issue(code: str, reason: str, **details: Any) -> None:
            issue = {'code': code, 'reason': reason}
            issue.update(
                {
                    key: value
                    for key, value in details.items()
                    if value not in (None, '', [], {})
                }
            )
            issues.append(issue)

        if len(terminal_contracts) != 1:
            add_issue(
                'ambiguous_structured_output_contract',
                'more than one terminal exactly-one JSON object contract is active',
                contract_count=len(terminal_contracts),
            )
        if len(required_fields) != len(set(required_fields)):
            add_issue(
                'duplicate_required_binding',
                'structured output contract repeats a required JSON field binding',
                fields=required_fields,
            )

        terminal_text = self._structured_join_terminal_result_text(terminal_result)
        parsed: Any = None
        duplicate_json_keys: list[str] = []

        def reject_nonstandard_json_constant(value: str) -> Any:
            raise ValueError(f'non-standard JSON constant: {value}')

        def capture_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            parsed_object: dict[str, Any] = {}
            for key, value in pairs:
                if key in parsed_object and key not in duplicate_json_keys:
                    duplicate_json_keys.append(key)
                parsed_object[key] = value
            return parsed_object

        if not terminal_text:
            add_issue(
                'terminal_result_missing',
                'terminal late-fill result does not contain structured result text',
            )
        else:
            try:
                parsed = json.loads(
                    terminal_text,
                    object_pairs_hook=capture_unique_json_object,
                    parse_constant=reject_nonstandard_json_constant,
                )
            except (TypeError, ValueError) as exc:
                add_issue(
                    'json_parse_error',
                    'terminal late-fill result is not exactly one standalone JSON value',
                    detail=str(exc),
                )
        if duplicate_json_keys:
            add_issue(
                'duplicate_json_key',
                'terminal JSON object contains duplicate keys',
                keys=duplicate_json_keys,
            )
        if terminal_text and not isinstance(parsed, Mapping):
            add_issue(
                'json_object_required',
                'structured output contract requires exactly one JSON object',
                actual_type=type(parsed).__name__,
            )

        parsed_object = dict(parsed) if isinstance(parsed, Mapping) else {}
        missing_fields = [field for field in required_fields if field not in parsed_object]
        invalid_fields = [
            field
            for field in required_fields
            if field in parsed_object
            and (
                not isinstance(parsed_object.get(field), str)
                or not str(parsed_object.get(field) or '').strip()
            )
        ]
        if missing_fields:
            add_issue(
                'missing_required_field',
                'terminal JSON object omits required runtime evidence bindings',
                fields=missing_fields,
            )
        if invalid_fields:
            add_issue(
                'invalid_required_field_value',
                'required runtime evidence fields must contain non-empty strings',
                fields=invalid_fields,
            )

        reference_records = self._structured_object_reference_records(
            request_payload,
            payload,
        )
        for binding in bindings:
            field_name = str(binding.get('field_name') or '').strip()
            field_role = str(binding.get('field_role') or '').strip()
            field_value = parsed_object.get(field_name) if field_name else None
            source_phase_id = str(binding.get('source_phase_id') or '').strip()
            source_branch_id = str(binding.get('source_branch_id') or '').strip()
            source_record = (
                records_by_phase.get(source_phase_id)
                or records_by_branch.get(source_branch_id)
                or {}
            )

            if field_role == 'audio_artifact_ref':
                source_result = self._structured_join_fill_result_for_record(
                    fill_results,
                    source_record,
                ) if source_record else {}
                artifact_refs = self._structured_object_source_artifact_refs(
                    source_result,
                    source_record,
                    payload,
                ) if source_result and source_record else set()
                if len(artifact_refs) != 1:
                    add_issue(
                        'artifact_ref_source_unresolved',
                        'audio artifact binding does not resolve to exactly one canonical artifact_ref',
                        field=field_name,
                        source_phase_id=source_phase_id,
                        artifact_refs=sorted(artifact_refs),
                    )
                elif isinstance(field_value, str) and field_value.strip() not in artifact_refs:
                    add_issue(
                        'artifact_ref_binding_mismatch',
                        'terminal JSON artifact_ref does not match its named producer',
                        field=field_name,
                        actual=field_value.strip(),
                        expected=next(iter(artifact_refs)),
                    )
                continue

            if field_role == 'real_transcript':
                source_result = self._structured_join_fill_result_for_record(
                    fill_results,
                    source_record,
                ) if source_record else {}
                transcript = self._structured_join_terminal_result_text(source_result)
                if not transcript:
                    add_issue(
                        'transcript_source_unresolved',
                        'transcript binding does not resolve to runtime STT evidence',
                        field=field_name,
                        source_phase_id=source_phase_id,
                    )
                elif isinstance(field_value, str) and re.sub(
                    r'\s+', ' ', field_value
                ).strip() != re.sub(r'\s+', ' ', transcript).strip():
                    add_issue(
                        'transcript_binding_mismatch',
                        'terminal JSON transcript does not match its named STT result',
                        field=field_name,
                        source_phase_id=source_phase_id,
                    )
                continue

            if field_role == 'preserved_visual_evidence':
                message_id = str(binding.get('message_id') or '').strip()
                source_response_id = str(binding.get('source_response_id') or '').strip()
                matched_messages: dict[tuple[str, str, str], Mapping[str, Any]] = {}
                for record in reference_records:
                    record_type = str(record.get('type') or record.get('kind') or '').strip().lower()
                    if record_type != 'message':
                        continue
                    if message_id and str(record.get('message_id') or '').strip() != message_id:
                        continue
                    if (
                        source_response_id
                        and str(record.get('source_response_id') or '').strip()
                        != source_response_id
                    ):
                        continue
                    content = str(record.get('content') or record.get('text') or '').strip()
                    identity = (
                        str(record.get('message_id') or '').strip(),
                        str(record.get('source_response_id') or '').strip(),
                        content,
                    )
                    matched_messages[identity] = record
                if len(matched_messages) != 1:
                    add_issue(
                        'preserved_visual_evidence_source_unresolved',
                        'preserved visual evidence does not resolve to exactly one carried message',
                        field=field_name,
                        match_count=len(matched_messages),
                    )
                else:
                    message = next(iter(matched_messages.values()))
                    bounded_evidence = _bounded_visual_evidence_from_selected_message(
                        message.get('content') or message.get('text')
                    )
                    if not bounded_evidence:
                        add_issue(
                            'preserved_visual_evidence_source_unresolved',
                            'carried message has no bounded visual-evidence section',
                            field=field_name,
                        )
                    elif isinstance(field_value, str) and re.sub(
                        r'\s+', ' ', field_value
                    ).strip() != re.sub(r'\s+', ' ', bounded_evidence).strip():
                        add_issue(
                            'preserved_visual_evidence_binding_mismatch',
                            'terminal JSON visual evidence does not equal the bounded preserved evidence',
                            field=field_name,
                        )
                continue

            if field_role == 'preserved_visual_artifact':
                artifact_ref = str(binding.get('artifact_ref') or '').strip()
                matching_artifacts = {
                    (
                        str(record.get('artifact_ref') or record.get('ref') or '').strip(),
                        str(record.get('path') or record.get('artifact_path') or '').strip(),
                        str(record.get('source_response_id') or '').strip(),
                    )
                    for record in reference_records
                    if artifact_ref
                    and str(record.get('artifact_ref') or record.get('ref') or '').strip()
                    == artifact_ref
                }
                if len(matching_artifacts) != 1:
                    add_issue(
                        'preserved_visual_artifact_source_unresolved',
                        'preserved visual artifact does not resolve to exactly one carried identity',
                        artifact_ref=artifact_ref,
                        match_count=len(matching_artifacts),
                    )

        if not issues:
            return [
                {
                    key: value
                    for key, value in {
                        **base_check,
                        'status': 'fulfilled',
                        'evidence': 'structured_object_contract_satisfied',
                        'reason': (
                            'terminal late-fill result satisfies the exactly-one JSON object '
                            'and named runtime evidence binding contract'
                        ),
                        'actual_count': 1,
                    }.items()
                    if value not in (None, '', [], {})
                }
            ]

        issue_codes = list(dict.fromkeys(str(issue.get('code') or '') for issue in issues))
        return [
            {
                key: value
                for key, value in {
                    **base_check,
                    'status': 'pending',
                    'evidence': 'structured_dependency_join_contract_unmet',
                    'reason': (
                        'terminal late-fill result violates its exactly-one JSON object '
                        f'contract: {", ".join(issue_codes)}'
                    ),
                    'actual_count': 1 if isinstance(parsed, Mapping) else 0,
                    'issue_codes': issue_codes,
                    'issues': issues,
                    'repair_required': True,
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    'recovery_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    'repair_action_reason': (
                        'terminal branch must rerun under its exact structured output and '
                        'runtime evidence binding contract'
                    ),
                    'blocked_by_branch_contract': True,
                }.items()
                if value not in (None, '', [], {})
            }
        ]

    def _structured_dependency_join_checks(
        self,
        *,
        request_payload: Optional[Mapping[str, Any]],
        artifact_payload: Optional[Mapping[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        object_contract_checks = self._structured_output_object_contract_checks(
            request_payload=request_payload,
            artifact_payload=artifact_payload,
            request_phase_graph=request_phase_graph,
        )
        if object_contract_checks:
            return object_contract_checks
        prompt = self._current_request_prompt_for_review(
            dict(request_payload) if isinstance(request_payload, Mapping) else None,
            request_phase_graph,
        )
        contract = self._structured_dependency_join_prompt_contract(prompt)
        if not contract:
            return []
        payload = artifact_payload if isinstance(artifact_payload, Mapping) else {}
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
        completed_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in (late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping)
            and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        completed_phase_ids = {
            str(item.get('phase_id') or '').strip()
            for item in (late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping) and str(item.get('phase_id') or '').strip()
        }

        ordered_records, records_by_phase, records_by_branch = self._structured_join_graph_records(
            request_phase_graph
        )
        terminal_candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for index, record in enumerate(ordered_records):
            fill_result = self._structured_join_fill_result_for_record(fill_results, record)
            execution_contract = (
                fill_result.get('execution_contract')
                if isinstance(fill_result.get('execution_contract'), Mapping)
                else {}
            )
            candidate = dict(record)
            for key, value in execution_contract.items():
                if value in (None, '', [], {}):
                    continue
                if key in {
                    'branch_id',
                    'capability',
                    'depends_on',
                    'phase_id',
                    'role',
                    'stage_direction',
                } or candidate.get(key) in (None, '', [], {}):
                    candidate[key] = value
            capability = normalize_capability(candidate.get('capability'))
            role = str(candidate.get('role') or '').strip().lower()
            stage_direction = str(candidate.get('stage_direction') or '').strip().lower()
            depends_on = [
                str(item or '').strip()
                for item in (candidate.get('depends_on') or [])
                if str(item or '').strip()
            ] if isinstance(candidate.get('depends_on'), list) else []
            candidate_branch_id = str(candidate.get('branch_id') or '').strip()
            candidate_phase_id = str(candidate.get('phase_id') or '').strip()
            candidate_completed = bool(
                fill_result
                or candidate_branch_id in completed_branch_ids
                or candidate_phase_id in completed_phase_ids
                or str(candidate.get('status') or '').strip().lower() in {'completed', 'fulfilled'}
            )
            if (
                capability == CAPABILITY_CHAT
                and candidate_completed
                and len(depends_on) >= 2
                and (
                    role == 'post_artifact_text_follow_up'
                    or stage_direction == 'write_text_after_artifact_generation'
                )
            ):
                score = (
                    (4 if role == 'post_artifact_text_follow_up' else 0)
                    + (2 if stage_direction == 'write_text_after_artifact_generation' else 0)
                    + len(depends_on)
                )
                terminal_candidates.append((score * 1000 + index, candidate, fill_result))
        if not terminal_candidates:
            return []
        _score, terminal, terminal_result = max(terminal_candidates, key=lambda item: item[0])
        terminal_depends_on = [
            str(item or '').strip()
            for item in (terminal.get('depends_on') or [])
            if str(item or '').strip()
        ]

        lineages: list[dict[str, Any]] = []
        lineage_validation_issues: list[dict[str, Any]] = []
        for dependency_id in terminal_depends_on:
            consumer_record = records_by_phase.get(dependency_id) or records_by_branch.get(dependency_id)
            consumer_capability = normalize_capability(
                consumer_record.get('capability')
                if isinstance(consumer_record, Mapping)
                else None
            )
            lineage_type = _STRUCTURED_JOIN_CONSUMER_PRODUCER_TYPES.get(consumer_capability)
            direct_producer_artifact_type = _STRUCTURED_JOIN_DIRECT_PRODUCER_TYPES.get(
                consumer_capability
            )
            if (
                not isinstance(consumer_record, Mapping)
                or (not lineage_type and not direct_producer_artifact_type)
            ):
                lineage_validation_issues.append(
                    {
                        'code': 'dependency_lineage_unresolved',
                        'reason': (
                            'terminal dependency does not resolve to a supported declared '
                            'artifact-evidence consumer phase'
                        ),
                        'terminal_dependency_id': dependency_id,
                        'consumer_capability': consumer_capability or None,
                    }
                )
                lineages.append(
                    {
                        'terminal_dependency_id': dependency_id,
                        'consumer_capability': consumer_capability or None,
                        'producer_identity': '',
                        'queue_index': None,
                        'aliases': set(),
                        'path_aliases': set(),
                    }
                )
                continue
            if lineage_type:
                expected_producer_capability, producer_artifact_type = lineage_type
                consumer_dependencies = [
                    str(item or '').strip()
                    for item in (consumer_record.get('depends_on') or [])
                    if str(item or '').strip()
                ] if isinstance(consumer_record.get('depends_on'), list) else []
                producer_records = [
                    records_by_phase.get(item) or records_by_branch.get(item)
                    for item in consumer_dependencies
                ]
                producer_records = [
                    record
                    for record in producer_records
                    if isinstance(record, Mapping)
                    and normalize_capability(record.get('capability'))
                    == expected_producer_capability
                ]
                lineage_relation = 'consumer_to_producer'
            else:
                expected_producer_capability = consumer_capability
                producer_artifact_type = str(direct_producer_artifact_type or '')
                producer_records = [consumer_record]
                lineage_relation = 'direct_producer'
            if len(producer_records) != 1:
                consumer_phase_id = str(consumer_record.get('phase_id') or '').strip()
                lineage_validation_issues.append(
                    {
                        'code': (
                            'dependency_lineage_ambiguous'
                            if len(producer_records) > 1
                            else 'dependency_lineage_unresolved'
                        ),
                        'reason': (
                            'artifact-evidence consumer dependency must resolve to exactly one '
                            f'{expected_producer_capability} producer phase'
                        ),
                        'terminal_dependency_id': dependency_id,
                        'consumer_phase_id': consumer_phase_id,
                        'consumer_capability': consumer_capability,
                        'expected_producer_capability': expected_producer_capability,
                        'vision_phase_id': (
                            consumer_phase_id
                            if consumer_capability == CAPABILITY_VISION_ANALYSIS
                            else None
                        ),
                        'producer_phase_ids': [
                            str(record.get('phase_id') or record.get('branch_id') or '').strip()
                            for record in producer_records
                        ],
                    }
                )
                lineages.append(
                    {
                        'terminal_dependency_id': dependency_id,
                        'consumer_phase_id': consumer_phase_id,
                        'consumer_branch_id': str(consumer_record.get('branch_id') or '').strip(),
                        'consumer_capability': consumer_capability,
                        'expected_producer_capability': expected_producer_capability,
                        'producer_artifact_type': producer_artifact_type,
                        'lineage_relation': lineage_relation,
                        'vision_phase_id': (
                            consumer_phase_id
                            if consumer_capability == CAPABILITY_VISION_ANALYSIS
                            else None
                        ),
                        'vision_branch_id': (
                            str(consumer_record.get('branch_id') or '').strip()
                            if consumer_capability == CAPABILITY_VISION_ANALYSIS
                            else None
                        ),
                        'producer_identity': '',
                        'queue_index': None,
                        'aliases': set(),
                        'path_aliases': set(),
                    }
                )
                continue
            producer_record = dict(producer_records[0])
            producer_result = self._structured_join_fill_result_for_record(
                fill_results,
                producer_record,
            )
            producer_identity = str(
                producer_record.get('phase_id')
                or producer_record.get('branch_id')
                or dependency_id
            ).strip()
            raw_queue_index = consumer_record.get('queue_index')
            if raw_queue_index in (None, ''):
                raw_queue_index = producer_record.get('queue_index')
            branch_ordinals = {
                int(match.group('ordinal'))
                for branch_id in (
                    str(consumer_record.get('branch_id') or '').strip(),
                    str(producer_record.get('branch_id') or '').strip(),
                )
                for match in [re.search(r'(?:-|_)(?P<ordinal>[1-9][0-9]*)$', branch_id)]
                if match
            }
            queue_index = 0
            queue_index_source = ''
            if raw_queue_index not in (None, ''):
                try:
                    queue_index = int(raw_queue_index)
                except (TypeError, ValueError):
                    queue_index = 0
                queue_index_source = 'queue_index'
            elif len(branch_ordinals) == 1:
                queue_index = next(iter(branch_ordinals))
                queue_index_source = 'branch_ordinal'
            elif len(branch_ordinals) > 1:
                lineage_validation_issues.append(
                    {
                        'code': 'dependency_label_identity_ambiguous',
                        'reason': (
                            'consumer and producer branch ordinals disagree on sequential label identity'
                        ),
                        'terminal_dependency_id': dependency_id,
                        'branch_ordinals': sorted(branch_ordinals),
                    }
                )
            if (
                queue_index > 0
                and branch_ordinals
                and any(ordinal != queue_index for ordinal in branch_ordinals)
            ):
                lineage_validation_issues.append(
                    {
                        'code': 'dependency_label_identity_ambiguous',
                        'reason': (
                            'queue index conflicts with a stable producer or consumer branch ordinal'
                        ),
                        'terminal_dependency_id': dependency_id,
                        'queue_index': queue_index,
                        'branch_ordinals': sorted(branch_ordinals),
                    }
                )
            if queue_index < 1 or queue_index > len(contract['labels']):
                lineage_validation_issues.append(
                    {
                        'code': 'dependency_label_identity_unresolved',
                        'reason': (
                            'consumer/producer lineage lacks a valid queue index for sequential label identity'
                        ),
                        'terminal_dependency_id': dependency_id,
                        'consumer_phase_id': str(consumer_record.get('phase_id') or '').strip(),
                        'consumer_capability': consumer_capability,
                        'expected_producer_capability': expected_producer_capability,
                        'vision_phase_id': (
                            str(consumer_record.get('phase_id') or '').strip()
                            if consumer_capability == CAPABILITY_VISION_ANALYSIS
                            else None
                        ),
                        'producer_phase_id': str(producer_record.get('phase_id') or '').strip(),
                        'queue_index': raw_queue_index,
                    }
                )
            lineages.append(
                {
                    'terminal_dependency_id': dependency_id,
                    'consumer_phase_id': str(consumer_record.get('phase_id') or '').strip(),
                    'consumer_branch_id': str(consumer_record.get('branch_id') or '').strip(),
                    'consumer_capability': consumer_capability,
                    'expected_producer_capability': expected_producer_capability,
                    'producer_artifact_type': producer_artifact_type,
                    'lineage_relation': lineage_relation,
                    'vision_phase_id': (
                        str(consumer_record.get('phase_id') or '').strip()
                        if consumer_capability == CAPABILITY_VISION_ANALYSIS
                        else None
                    ),
                    'vision_branch_id': (
                        str(consumer_record.get('branch_id') or '').strip()
                        if consumer_capability == CAPABILITY_VISION_ANALYSIS
                        else None
                    ),
                    'producer_phase_id': str(producer_record.get('phase_id') or '').strip(),
                    'producer_branch_id': str(producer_record.get('branch_id') or '').strip(),
                    'producer_identity': producer_identity,
                    'queue_index': queue_index if queue_index > 0 else None,
                    'queue_index_source': queue_index_source or None,
                    'aliases': self._structured_join_artifact_aliases(producer_result),
                    'path_aliases': self._structured_join_artifact_path_aliases(producer_result),
                }
            )

        # Queue indexes and branch ordinals are capability-local in the phase
        # graph.  A mixed image/audio join therefore legitimately has both an
        # image producer `-1` and an audio producer `-1`.  For mixed producer
        # capabilities only, derive the cross-capability label ordinal from the
        # stable declared producer order in graph truth.  This remains
        # independent of terminal `depends_on` order and does not relax missing
        # or duplicate identity for same-capability image joins.
        producer_capabilities = {
            str(lineage.get('expected_producer_capability') or '').strip()
            for lineage in lineages
            if str(lineage.get('expected_producer_capability') or '').strip()
        }
        producer_identities = [
            str(lineage.get('producer_identity') or '').strip()
            for lineage in lineages
        ]
        if (
            len(producer_capabilities) > 1
            and len(lineages) == len(contract['labels'])
            and all(producer_identities)
            and len(set(producer_identities)) == len(producer_identities)
        ):
            producer_graph_order: dict[str, int] = {}
            for record_index, record in enumerate(ordered_records):
                for identity in (
                    str(record.get('phase_id') or '').strip(),
                    str(record.get('branch_id') or '').strip(),
                ):
                    if identity:
                        producer_graph_order.setdefault(identity, record_index)
            ordered_lineages = sorted(
                lineages,
                key=lambda lineage: producer_graph_order.get(
                    str(lineage.get('producer_identity') or '').strip(),
                    len(ordered_records),
                ),
            )
            if all(
                str(lineage.get('producer_identity') or '').strip()
                in producer_graph_order
                for lineage in ordered_lineages
            ):
                for label_ordinal, lineage in enumerate(ordered_lineages, start=1):
                    lineage['capability_queue_index'] = lineage.get('queue_index')
                    lineage['queue_index'] = label_ordinal
                    lineage['queue_index_source'] = 'stable_mixed_producer_graph_order'

        queue_index_lineages: dict[int, list[dict[str, Any]]] = {}
        for lineage in lineages:
            queue_index = lineage.get('queue_index')
            if isinstance(queue_index, int) and queue_index > 0:
                queue_index_lineages.setdefault(queue_index, []).append(lineage)
        for queue_index, matches in queue_index_lineages.items():
            if len(matches) <= 1:
                continue
            lineage_validation_issues.append(
                {
                    'code': 'dependency_label_identity_ambiguous',
                    'reason': 'multiple terminal dependencies claim the same sequential label ordinal',
                    'queue_index': queue_index,
                    'terminal_dependency_ids': [
                        str(item.get('terminal_dependency_id') or '').strip()
                        for item in matches
                    ],
                }
            )

        canonical_artifacts: list[Mapping[str, Any]] = [
            item
            for item in (payload.get('artifacts') or [])
            if isinstance(item, Mapping)
        ]
        try:
            canonical_artifacts.extend(
                item
                for item in self._hook('build_canonical_response_artifacts')(dict(payload))
                if isinstance(item, Mapping)
            )
        except Exception:
            logging.getLogger(__name__).debug(
                'failed to build canonical artifacts for structured dependency join review',
                exc_info=True,
            )
        for lineage in lineages:
            seed_aliases = set(lineage['aliases'])
            seed_path_aliases = set(lineage.get('path_aliases') or set())
            producer_artifact_type = str(
                lineage.get('producer_artifact_type') or ''
            ).strip().lower()
            for artifact in canonical_artifacts:
                if not self._structured_join_artifact_matches_type(
                    artifact,
                    producer_artifact_type,
                ):
                    continue
                artifact_status = str(
                    artifact.get('artifact_status')
                    or artifact.get('lifecycle_state')
                    or artifact.get('status')
                    or ''
                ).strip().lower()
                if artifact_status and artifact_status not in {
                    'available',
                    'completed',
                    'fulfilled',
                    'materialized',
                    'ready',
                }:
                    continue
                artifact_aliases = self._structured_join_artifact_aliases(artifact)
                artifact_path_aliases = self._structured_join_artifact_path_aliases(artifact)
                aliases_match_current_fill = bool(
                    seed_path_aliases.intersection(artifact_path_aliases)
                    if seed_path_aliases
                    else seed_aliases.intersection(artifact_aliases)
                )
                if aliases_match_current_fill:
                    lineage['aliases'].update(artifact_aliases)
                    lineage['path_aliases'].update(artifact_path_aliases)

        alias_to_producers: dict[str, set[str]] = {}
        for lineage in lineages:
            producer_identity = str(lineage.get('producer_identity') or '').strip()
            for alias in lineage.get('aliases') or set():
                alias_to_producers.setdefault(alias, set()).add(producer_identity)

        expected_labels = list(contract['labels'])
        required_fields = list(contract['required_fields'])
        label_field = str(contract['label_field'])
        artifact_ref_field = str(contract['artifact_ref_field'])
        terminal_branch_id = str(terminal.get('branch_id') or terminal_result.get('branch_id') or '').strip()
        terminal_phase_id = str(terminal.get('phase_id') or terminal_result.get('phase_id') or '').strip()
        execution_contract = (
            terminal_result.get('execution_contract')
            if isinstance(terminal_result.get('execution_contract'), Mapping)
            else {}
        )
        base_check: dict[str, Any] = {
            'check_kind': 'structured_dependency_join',
            'phase_id': terminal_phase_id or None,
            'branch_id': terminal_branch_id or None,
            'capability': CAPABILITY_CHAT,
            'output_type': 'text',
            'role': str(terminal.get('role') or '').strip() or None,
            'depends_on': terminal_depends_on,
            'stage_direction': str(terminal.get('stage_direction') or '').strip() or None,
            'content_payload_source': str(
                execution_contract.get('content_payload_source')
                or terminal.get('content_payload_source')
                or f'late_fill_result:{terminal_branch_id or terminal_phase_id}'
            ).strip(),
            'terminal_result_source': 'late_fill.fill_results',
            'expected_count': len(expected_labels),
            'expected_labels': expected_labels,
            'required_fields': required_fields,
            'label_field': label_field,
            'artifact_ref_field': artifact_ref_field,
            'execution_contract': dict(execution_contract) if execution_contract else None,
        }
        issues: list[dict[str, Any]] = [
            dict(item) for item in lineage_validation_issues
        ]

        def add_issue(code: str, reason: str, **details: Any) -> None:
            issue = {'code': code, 'reason': reason}
            issue.update(
                {
                    key: value
                    for key, value in details.items()
                    if value not in (None, '', [], {})
                }
            )
            issues.append(issue)

        if len(expected_labels) != len(lineages):
            add_issue(
                'dependency_label_count_mismatch',
                'explicit labels do not map one-to-one to terminal dependencies',
                dependency_count=len(lineages),
                label_count=len(expected_labels),
            )

        terminal_text = self._structured_join_terminal_result_text(terminal_result)
        parsed: Any = None
        parsed_successfully = False
        duplicate_json_keys: list[str] = []

        def reject_nonstandard_json_constant(value: str) -> Any:
            raise ValueError(f'non-standard JSON constant: {value}')

        def capture_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            payload_object: dict[str, Any] = {}
            for key, value in pairs:
                if key in payload_object and key not in duplicate_json_keys:
                    duplicate_json_keys.append(key)
                payload_object[key] = value
            return payload_object

        if not terminal_text:
            add_issue(
                'terminal_result_missing',
                'terminal late-fill result does not contain structured result text',
            )
        else:
            try:
                parsed = json.loads(
                    _strip_fenced_json_boundary(terminal_text),
                    object_pairs_hook=capture_unique_json_object,
                    parse_constant=reject_nonstandard_json_constant,
                )
                parsed_successfully = True
            except (TypeError, ValueError) as exc:
                add_issue(
                    'json_parse_error',
                    'terminal late-fill result is not valid JSON or a fenced JSON payload',
                    detail=str(exc),
                )
        if duplicate_json_keys:
            add_issue(
                'duplicate_json_key',
                'terminal JSON contains duplicate object keys with ambiguous last-value semantics',
                keys=duplicate_json_keys,
            )
        if parsed_successfully and not isinstance(parsed, list):
            add_issue(
                'json_list_required',
                'explicit current-turn contract requires a JSON list',
                actual_type=type(parsed).__name__,
            )
            parsed = None

        rows = parsed if isinstance(parsed, list) else []
        if isinstance(parsed, list) and len(rows) != len(expected_labels):
            add_issue(
                'exact_count_mismatch',
                'terminal JSON list does not have exactly one row per requested label',
                actual_count=len(rows),
                expected_count=len(expected_labels),
            )
        row_objects: list[tuple[int, Mapping[str, Any]]] = []
        invalid_row_indexes: list[int] = []
        missing_fields: list[dict[str, Any]] = []
        invalid_fields: list[dict[str, Any]] = []
        unexpected_fields: list[dict[str, Any]] = []

        def required_field_value_is_valid(field: str, value: Any) -> bool:
            if value is None or isinstance(value, (Mapping, list, tuple, set)):
                return False
            normalized_field = str(field or '').strip().lower()
            requires_text_value = bool(
                normalized_field == 'label'
                or normalized_field in _STRUCTURED_JOIN_ARTIFACT_REF_FIELDS
                or normalized_field == 'evidence'
                or normalized_field.endswith('_evidence')
            )
            if requires_text_value and not isinstance(value, str):
                return False
            if isinstance(value, str):
                return bool(value.strip())
            return True

        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                invalid_row_indexes.append(index)
                continue
            row_objects.append((index, row))
            missing = [
                field
                for field in required_fields
                if field not in row
            ]
            if missing:
                missing_fields.append({'row_index': index, 'fields': missing})
            invalid = [
                field
                for field in required_fields
                if field in row and not required_field_value_is_valid(field, row.get(field))
            ]
            if invalid:
                invalid_fields.append({'row_index': index, 'fields': invalid})
            unexpected = [
                str(field)
                for field in row
                if str(field) not in required_fields
            ]
            if unexpected:
                unexpected_fields.append({'row_index': index, 'fields': unexpected})
        if invalid_row_indexes:
            add_issue(
                'item_not_object',
                'each structured terminal join item must be a JSON object',
                row_indexes=invalid_row_indexes,
            )
        if missing_fields:
            add_issue(
                'missing_required_field',
                'one or more terminal join rows omit requested fields',
                rows=missing_fields,
            )
        if invalid_fields:
            add_issue(
                'invalid_required_field_value',
                'one or more requested fields contain empty, non-scalar, or invalid values',
                rows=invalid_fields,
            )
        if unexpected_fields:
            add_issue(
                'unexpected_field',
                'terminal join rows contain fields outside the explicit current-turn contract',
                rows=unexpected_fields,
            )

        labels_by_value: dict[str, list[int]] = {}
        for row_index, row in row_objects:
            value = str(row.get(label_field) or '').strip()
            if value:
                labels_by_value.setdefault(value, []).append(row_index)
        duplicate_labels = {
            label: indexes
            for label, indexes in labels_by_value.items()
            if len(indexes) > 1
        }
        if duplicate_labels:
            add_issue(
                'duplicate_label',
                'requested labels must occur exactly once',
                labels=duplicate_labels,
            )
        missing_labels = [label for label in expected_labels if label not in labels_by_value]
        if missing_labels:
            add_issue(
                'missing_label',
                'one or more requested labels are absent from the terminal join',
                labels=missing_labels,
            )
        unexpected_labels = [
            label for label in labels_by_value
            if label not in expected_labels
        ]
        if unexpected_labels:
            add_issue(
                'unexpected_label',
                'terminal join contains labels outside the explicit current-turn contract',
                labels=unexpected_labels,
            )

        resolved_ref_rows: dict[tuple[str, str], list[int]] = {}
        row_ref_resolution: dict[int, set[str]] = {}
        for row_index, row in row_objects:
            raw_ref = str(row.get(artifact_ref_field) or '').strip()
            if not raw_ref:
                continue
            aliases = self._structured_join_alias_tokens(raw_ref)
            resolved = {
                producer
                for alias in aliases
                for producer in alias_to_producers.get(alias, set())
            }
            row_ref_resolution[row_index] = resolved
            if len(resolved) == 1:
                ref_key = ('producer', next(iter(resolved)))
            else:
                ref_key = ('artifact_ref', min(aliases) if aliases else raw_ref)
            resolved_ref_rows.setdefault(ref_key, []).append(row_index)
        duplicate_refs = [
            {
                'resolved_kind': ref_key[0],
                'resolved_value': ref_key[1],
                'row_indexes': indexes,
            }
            for ref_key, indexes in resolved_ref_rows.items()
            if len(indexes) > 1
        ]
        if duplicate_refs:
            add_issue(
                'duplicate_artifact_ref',
                'artifact refs or paths must resolve one-to-one across terminal join rows',
                duplicates=duplicate_refs,
            )

        producer_for_label: dict[str, str] = {}
        for queue_index, label in enumerate(expected_labels, start=1):
            matches = queue_index_lineages.get(queue_index) or []
            if len(matches) != 1:
                continue
            producer_identity = str(matches[0].get('producer_identity') or '').strip()
            if producer_identity:
                producer_for_label[label] = producer_identity
        label_ref_mismatches: list[dict[str, Any]] = []
        for row_index, row in row_objects:
            label = str(row.get(label_field) or '').strip()
            raw_ref = str(row.get(artifact_ref_field) or '').strip()
            expected_producer = producer_for_label.get(label)
            if not label or not raw_ref or not expected_producer:
                continue
            resolved = row_ref_resolution.get(row_index, set())
            if resolved != {expected_producer}:
                label_ref_mismatches.append(
                    {
                        'row_index': row_index,
                        'label': label,
                        'artifact_ref': raw_ref,
                        'expected_producer': expected_producer,
                        'resolved_producers': sorted(resolved),
                    }
                )
        if label_ref_mismatches:
            add_issue(
                'label_ref_dependency_mismatch',
                'label/artifact mapping does not match terminal dependency to vision to producer lineage',
                rows=label_ref_mismatches,
            )

        if not issues:
            return [
                {
                    key: value
                    for key, value in {
                        **base_check,
                        'status': 'fulfilled',
                        'evidence': 'structured_join_contract_satisfied',
                        'reason': (
                            'terminal late-fill result satisfies the explicit structured '
                            'dependency join contract'
                        ),
                        'actual_count': len(rows),
                    }.items()
                    if value not in (None, '', [], {})
                }
            ]

        issue_codes = list(dict.fromkeys(str(issue.get('code') or '') for issue in issues))
        identity_issue_codes = {
            'duplicate_artifact_ref',
            'label_ref_dependency_mismatch',
        }
        repair_action = (
            RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
            if issue_codes and set(issue_codes).issubset(identity_issue_codes)
            else RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
        )
        pending_check = {
            **base_check,
            'status': 'pending',
            'evidence': 'structured_dependency_join_contract_unmet',
            'reason': (
                'terminal late-fill result violates the explicit structured dependency '
                f'join contract: {", ".join(issue_codes)}'
            ),
            'actual_count': len(rows) if isinstance(parsed, list) else None,
            'issue_codes': issue_codes,
            'issues': issues,
            'repair_required': True,
            'repair_action': repair_action,
            'recovery_action': repair_action,
            'repair_action_reason': (
                'terminal artifact identities must be rebound to declared dependency lineage'
                if repair_action == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
                else 'terminal branch must rerun under its explicit structured output contract'
            ),
            'blocked_by_dependency_input': (
                repair_action == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
            ),
            'blocked_by_branch_contract': (
                repair_action == RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
            ),
        }
        return [
            {
                key: value
                for key, value in pending_check.items()
                if value not in (None, '', [], {})
            }
        ]

    def _artifact_records_for_link_binding(
        self,
        payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            return []
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def add_record(raw_record: Any, *, defaults: Optional[Mapping[str, Any]] = None) -> None:
            if not isinstance(raw_record, Mapping):
                return
            record = dict(defaults or {})
            record.update(dict(raw_record))
            artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
            path = _artifact_path(record)
            content = str(record.get('content') or record.get('text') or record.get('result_text') or '').strip()
            key = (
                artifact_type,
                path,
                str(record.get('branch_id') or '').strip(),
                str(record.get('phase_id') or '').strip(),
            )
            if not artifact_type and not path and not content:
                return
            if key in seen:
                return
            seen.add(key)
            records.append({key: value for key, value in record.items() if value not in (None, '', [], {})})

        for artifact in payload.get('artifacts') or []:
            add_record(artifact)
        for artifact in self._hook('build_canonical_response_artifacts')(dict(payload)):
            add_record(artifact)
        for item in payload.get('saved_text_artifacts') or []:
            source = item if isinstance(item, Mapping) else {'path': item}
            add_record({'type': 'text', **dict(source)})
        if str(payload.get('saved_text_path') or payload.get('savedTextPath') or '').strip():
            add_record({'type': 'text', 'path': payload.get('saved_text_path') or payload.get('savedTextPath')})
        if str(payload.get('saved_image_path') or payload.get('savedImagePath') or '').strip():
            add_record({'type': 'image', 'path': payload.get('saved_image_path') or payload.get('savedImagePath')})

        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        for result in late_fill.get('fill_results') or []:
            if not isinstance(result, Mapping):
                continue
            defaults = {
                'branch_id': result.get('branch_id'),
                'phase_id': result.get('phase_id'),
                'source': 'late_fill_result',
            }
            if str(result.get('saved_text_path') or '').strip():
                add_record(
                    {
                        'type': 'text',
                        'path': result.get('saved_text_path'),
                        'content': result.get('result_text') or result.get('content_payload'),
                        'text_artifact_extension': result.get('text_artifact_extension'),
                        'text_artifact_source_name': result.get('text_artifact_source_name'),
                        'text_artifact_target_path': result.get('text_artifact_target_path'),
                        'text_artifact_request': result.get('artifact_request'),
                    },
                    defaults=defaults,
                )
            if str(result.get('saved_image_path') or '').strip():
                add_record({'type': 'image', 'path': result.get('saved_image_path')}, defaults=defaults)
            for artifact in result.get('artifacts') or []:
                add_record(artifact, defaults=defaults)
        return records

    @staticmethod
    def _text_artifact_record_content(record: Mapping[str, Any]) -> str:
        for key in ('content', 'text', 'result_text', 'content_payload'):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        path = _artifact_path(record)
        if not path:
            return ''
        try:
            target = Path(path).expanduser()
            if not target.is_file() or target.stat().st_size > 512_000:
                return ''
            return target.read_text(encoding='utf-8', errors='replace').strip()
        except OSError:
            return ''

    @staticmethod
    def _bounded_repair_content_block(label: str, content: str, *, limit: int = _HTML_CSS_SELECTOR_BINDING_CONTENT_LIMIT) -> list[str]:
        text = str(content or '').strip()
        if not text:
            return []
        bounded_limit = max(1000, int(limit))
        truncated = len(text) > bounded_limit
        if truncated:
            text = text[:bounded_limit].rstrip()
        lines = [
            f'{label}:',
            '--- CURRENT SAVED FILE START ---',
            text,
            '--- CURRENT SAVED FILE END ---',
        ]
        if truncated:
            lines.append('Current saved file content was truncated for repair prompt size; preserve the unshown file structure unless directly required by the listed defect.')
        return lines

    @staticmethod
    def _class_token_is_meaningful(value: str) -> bool:
        token = str(value or '').strip().lower()
        if not token or token in _HTML_CSS_IGNORABLE_CLASS_TOKENS:
            return False
        if len(token) < 3 or token.isdigit():
            return False
        if token.startswith(('js-', 'is-', 'has-')):
            return False
        return True

    @classmethod
    def _html_class_tokens(cls, content: str) -> set[str]:
        tokens: set[str] = set()
        for match in _HTML_CLASS_ATTR_RE.finditer(str(content or '')):
            for token in re.split(r'\s+', str(match.group('value') or '').strip()):
                normalized = token.strip().lower()
                if cls._class_token_is_meaningful(normalized):
                    tokens.add(normalized)
        return tokens

    @classmethod
    def _css_class_selector_tokens(cls, content: str) -> set[str]:
        text = re.sub(r'/\*[\s\S]*?\*/', '', str(content or ''))
        tokens: set[str] = set()
        for match in _CSS_BLOCK_SELECTOR_RE.finditer(text):
            selectors = str(match.group('selectors') or '').strip()
            if not selectors or selectors.startswith('@'):
                continue
            selectors = re.sub(r'(?P<quote>["\']).*?(?P=quote)', '', selectors)
            for selector_match in _CSS_CLASS_SELECTOR_RE.finditer(selectors):
                normalized = str(selector_match.group('class') or '').strip().lower()
                if cls._class_token_is_meaningful(normalized):
                    tokens.add(normalized)
        return tokens

    @staticmethod
    def _html_links_css_artifact(html_content: str, css_record: Mapping[str, Any]) -> bool:
        content = str(html_content or '')
        if not content:
            return False
        return any(token and token in content for token in _artifact_link_tokens(css_record))

    @staticmethod
    def _line_number_for_offset(text: str, offset: int) -> int:
        try:
            bounded = max(0, min(int(offset), len(text)))
        except (TypeError, ValueError):
            bounded = 0
        return str(text or '').count('\n', 0, bounded) + 1

    @classmethod
    def _html_open_tag_stack_before_offset(cls, content: str, offset: int) -> list[str]:
        text = str(content or '')
        try:
            bounded = max(0, min(int(offset), len(text)))
        except (TypeError, ValueError):
            bounded = 0
        stack: list[str] = []
        for match in _HTML_SYNTAX_TAG_RE.finditer(text[:bounded]):
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!'):
                continue
            tag = str(match.group('tag') or '').strip().lower()
            if not tag:
                continue
            if match.group('close'):
                if tag in stack:
                    while stack:
                        current = stack.pop()
                        if current == tag:
                            break
                continue
            if tag not in _HTML_SYNTAX_VOID_TAGS and not match.group('self'):
                stack.append(tag)
        return stack

    @staticmethod
    def _html_attrs_have_hero_marker(attrs: str) -> bool:
        for match in _HTML_CLASS_ATTR_RE.finditer(str(attrs or '')):
            value = str(match.group('value') or '').strip()
            if value and _HTML_HERO_ATTR_VALUE_RE.search(value):
                return True
        return False

    @classmethod
    def _html_hero_header_landmark_containment_defects(
        cls,
        content: str,
    ) -> list[dict[str, Any]]:
        text = str(content or '')
        if not text.strip():
            return []
        stack: list[dict[str, Any]] = []
        defects: list[dict[str, Any]] = []
        for match in _HTML_SYNTAX_TAG_RE.finditer(text):
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!'):
                continue
            tag = str(match.group('tag') or '').strip().lower()
            if not tag:
                continue
            if match.group('close'):
                for index in range(len(stack) - 1, -1, -1):
                    if str(stack[index].get('tag') or '') == tag:
                        del stack[index:]
                        break
                continue

            if tag in _HTML_HERO_LANDMARK_TAGS:
                hero_header = next(
                    (
                        item
                        for item in reversed(stack)
                        if str(item.get('tag') or '') == 'header'
                        and item.get('hero_header') is True
                    ),
                    None,
                )
                if hero_header:
                    defects.append(
                        {
                            'landmark_tag': tag,
                            'landmark_line': cls._line_number_for_offset(text, match.start()),
                            'landmark_start': match.start(),
                            'header_line': hero_header.get('line'),
                            'header_start': hero_header.get('start'),
                        }
                    )

            if tag in _HTML_SYNTAX_VOID_TAGS or match.group('self'):
                continue
            attrs = str(match.group('attrs') or '')
            stack.append(
                {
                    'tag': tag,
                    'line': cls._line_number_for_offset(text, match.start()),
                    'start': match.start(),
                    'hero_header': tag == 'header' and cls._html_attrs_have_hero_marker(attrs),
                }
            )
        return defects

    @classmethod
    def _html_unsupported_nav_anchor_wrapper_issue(
        cls,
        content: str,
        match: re.Match[str],
    ) -> Optional[str]:
        tag = str(match.group('tag') or '').strip().lower()
        attrs = str(match.group('attrs') or '').strip()
        if (
            not tag
            or tag in _HTML_SYNTAX_REPAIR_CLOSING_TAGS
            or '-' in tag
            or ':' in tag
            or attrs
        ):
            return None
        stack = cls._html_open_tag_stack_before_offset(content, match.start())
        in_navigation_context = (
            'nav' in stack
            or (stack and stack[-1] == 'li' and any(parent in {'ul', 'ol', 'menu'} for parent in stack[:-1]))
        )
        if not in_navigation_context:
            return None
        return (
            f'HTML has unsupported navigation anchor wrapper <{tag}> at line '
            f'{cls._line_number_for_offset(content, match.start())}; unwrap it and keep the <a> element'
        )

    @staticmethod
    def _html_attribute_values(attrs: str) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}
        for match in _HTML_ATTRIBUTE_VALUE_RE.finditer(str(attrs or '')):
            name = str(match.group('name') or '').strip().lower()
            if name:
                values.setdefault(name, []).append(str(match.group('value') or ''))
        return values

    @staticmethod
    def _html_href_targets_stylesheet(value: str) -> bool:
        lowered = str(value or '').strip().lower()
        if not lowered:
            return False
        without_fragment = lowered.split('#', 1)[0]
        without_query = without_fragment.split('?', 1)[0]
        return (
            without_query.endswith('.css')
            or 'fonts.googleapis.com/css' in lowered
            or '/css2' in lowered
            or '/css?' in lowered
        )

    @classmethod
    def _html_link_stylesheet_rel_is_invalid(cls, attrs: str) -> bool:
        values = cls._html_attribute_values(attrs)
        href = next((item for item in values.get('href', []) if item.strip()), '')
        if not cls._html_href_targets_stylesheet(href):
            return False
        rel = next((item for item in values.get('rel', []) if item.strip()), '')
        if not rel:
            return True
        rel_tokens = {
            token.strip().lower()
            for token in re.split(r'\s+', rel)
            if token.strip()
        }
        return not bool(rel_tokens & _HTML_STYLESHEET_REL_TOKENS)

    @staticmethod
    def _html_attrs_have_unbalanced_quote(attrs: str) -> bool:
        quote: Optional[str] = None
        for char in str(attrs or ''):
            if quote:
                if char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
        return quote is not None

    @staticmethod
    def _offset_is_inside_spans(offset: int, spans: Sequence[tuple[int, int]]) -> bool:
        return any(start <= offset < end for start, end in spans)

    @classmethod
    def _html_stray_open_angle_offsets(cls, content: str) -> list[int]:
        text = str(content or '')
        valid_spans = [match.span() for match in _HTML_SYNTAX_TAG_RE.finditer(text)]
        raw_text_spans = [match.span() for match in _HTML_RAW_TEXT_BLOCK_RE.finditer(text)]
        offsets: list[int] = []
        for match in re.finditer(r'<', text):
            offset = match.start()
            if cls._offset_is_inside_spans(offset, valid_spans):
                continue
            if cls._offset_is_inside_spans(offset, raw_text_spans):
                continue
            offsets.append(offset)
            if len(offsets) >= 6:
                break
        return offsets

    @classmethod
    def _html_syntax_sanity_issues(cls, content: str) -> list[str]:
        text = str(content or '')
        if not text.strip():
            return []
        issues: list[str] = []
        for match in _HTML_MALFORMED_CLASS_ONLY_OPEN_TAG_RE.finditer(text):
            issues.append(
                f'HTML has malformed class-only opening tag at line {cls._line_number_for_offset(text, match.start())}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_MARKUP_IN_ATTRIBUTE_RE.finditer(text):
            issues.append(
                f'HTML attribute contains markup-like closing tag at line {cls._line_number_for_offset(text, match.start())}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_UNTERMINATED_KNOWN_OPEN_TAG_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            issues.append(
                f'HTML has unterminated opening tag <{tag}> at line {cls._line_number_for_offset(text, match.start())}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_DUPLICATE_OPEN_ANGLE_KNOWN_TAG_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            issues.append(
                f'HTML opening tag <{tag}> has duplicate opening angle bracket at line {cls._line_number_for_offset(text, match.start())}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_STRAY_PUNCTUATION_PSEUDO_TAG_BEFORE_KNOWN_TAG_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            fragment = str(match.group('fragment') or '').strip()
            issues.append(
                f'HTML has stray punctuation pseudo-tag `{fragment}` before <{tag}> at line {cls._line_number_for_offset(text, match.start("fragment"))}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_MALFORMED_OPEN_PREFIX_KNOWN_TAG_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            prefix = str(match.group('prefix') or '').strip()
            issues.append(
                f'HTML opening tag <{tag}> has stray prefix `{prefix}` at line {cls._line_number_for_offset(text, match.start())}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_QUOTED_KNOWN_OPEN_TAG_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            issues.append(
                f'HTML opening tag <{tag}> is quoted instead of angle-bracketed at line {cls._line_number_for_offset(text, match.start())}'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_MALFORMED_STYLESHEET_LINK_ATTR_RE.finditer(text):
            issues.append(
                'HTML stylesheet link tag has malformed rel/href attributes at line '
                f'{cls._line_number_for_offset(text, match.start())}; use rel="stylesheet" href="..."'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_MALFORMED_FORMATTED_ANCHOR_FRAGMENT_RE.finditer(text):
            issues.append(
                'HTML has malformed formatted anchor fragment at line '
                f'{cls._line_number_for_offset(text, match.start())}; use a normal <a href="..."> link'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_UNSUPPORTED_NAV_ANCHOR_WRAPPER_RE.finditer(text):
            issue = cls._html_unsupported_nav_anchor_wrapper_issue(text, match)
            if issue:
                issues.append(issue)
                if len(issues) >= 6:
                    return issues
        for match in _HTML_LITERAL_UNSUPPORTED_TAG_RE.finditer(text):
            issues.append(
                'HTML has literal unsupported placeholder tag <unsupported> at line '
                f'{cls._line_number_for_offset(text, match.start())}; unwrap it or replace it with a standard element'
            )
            if len(issues) >= 6:
                return issues
        for match in _HTML_UNKNOWN_CLASS_WRAPPER_TAG_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            if '-' in tag or ':' in tag:
                continue
            issues.append(
                f'HTML has unknown class/id wrapper tag <{tag}> at line '
                f'{cls._line_number_for_offset(text, match.start())}; use a standard wrapper such as <div>'
            )
            if len(issues) >= 6:
                return issues

        for defect in cls._html_hero_header_landmark_containment_defects(text):
            landmark = str(defect.get('landmark_tag') or '').strip().lower()
            landmark_line = defect.get('landmark_line')
            header_line = defect.get('header_line')
            issues.append(
                f'HTML <{landmark}> landmark is nested inside hero/header opened at line {header_line}; '
                f'close the header before line {landmark_line}'
            )
            if len(issues) >= 6:
                return issues

        href_context_stack: list[str] = []
        for match in _HTML_SYNTAX_TAG_RE.finditer(text):
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!'):
                continue
            tag = str(match.group('tag') or '').strip().lower()
            if not tag:
                continue
            if match.group('close'):
                if tag in href_context_stack:
                    while href_context_stack:
                        current = href_context_stack.pop()
                        if current == tag:
                            break
                continue
            attrs = str(match.group('attrs') or '')
            attrs_offset = match.start('attrs') if match.start('attrs') >= 0 else match.start()
            for attr_match in _HTML_MALFORMED_CLASS_ATTR_COLON_RE.finditer(attrs):
                issues.append(
                    f'HTML class attribute uses `:` instead of `=` at line {cls._line_number_for_offset(text, attrs_offset + attr_match.start())}'
                )
                if len(issues) >= 6:
                    return issues
            for attr_match in _HTML_BARE_DUPLICATE_ATTRIBUTE_RE.finditer(attrs):
                attr_name = str(attr_match.group('name') or '').strip().lower()
                issues.append(
                    f'HTML attribute `{attr_name}` has a bare duplicate token before its value at line {cls._line_number_for_offset(text, attrs_offset + attr_match.start())}'
                )
                if len(issues) >= 6:
                    return issues
            if cls._html_attrs_have_unbalanced_quote(attrs):
                issues.append(
                    f'HTML opening tag <{tag}> has an unbalanced quote in attributes at line {cls._line_number_for_offset(text, attrs_offset)}'
                )
                if len(issues) >= 6:
                    return issues
            if tag == 'link' and cls._html_link_stylesheet_rel_is_invalid(attrs):
                issues.append(
                    f'HTML stylesheet link tag has invalid or missing rel attribute at line {cls._line_number_for_offset(text, match.start())}; use rel="stylesheet" for CSS links'
                )
                if len(issues) >= 6:
                    return issues
            if (
                'svg' not in href_context_stack
                and tag not in _HTML_HREF_ALLOWED_TAGS
                and _HTML_HREF_ATTR_RE.search(attrs)
            ):
                issues.append(
                    f'HTML has unsupported href element <{tag}> at line {cls._line_number_for_offset(text, match.start())}; use <a> for navigation links'
                )
                if len(issues) >= 6:
                    return issues
            if tag not in _HTML_SYNTAX_VOID_TAGS and not match.group('self'):
                href_context_stack.append(tag)

        for offset in cls._html_stray_open_angle_offsets(text):
            issues.append(
                f'HTML has stray opening angle bracket at line {cls._line_number_for_offset(text, offset)}; escape text as `&lt;` or complete the intended tag'
            )
            if len(issues) >= 6:
                return issues

        stack: list[tuple[str, int]] = []
        for match in _HTML_SYNTAX_TAG_RE.finditer(text):
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!'):
                continue
            tag = str(match.group('tag') or '').strip().lower()
            if not tag:
                continue
            line = cls._line_number_for_offset(text, match.start())
            if match.group('close'):
                if not stack:
                    issues.append(f'HTML has stray closing tag </{tag}> at line {line}')
                elif stack[-1][0] == tag:
                    stack.pop()
                else:
                    open_tags = [item[0] for item in stack]
                    if tag in open_tags:
                        expected_tag, expected_line = stack[-1]
                        issues.append(
                            f'HTML closes </{tag}> at line {line} before closing <{expected_tag}> opened at line {expected_line}'
                        )
                        while stack and stack[-1][0] != tag:
                            stack.pop()
                        if stack and stack[-1][0] == tag:
                            stack.pop()
                    else:
                        issues.append(f'HTML has stray closing tag </{tag}> at line {line}')
            elif tag not in _HTML_SYNTAX_VOID_TAGS and not match.group('self'):
                stack.append((tag, line))
            if len(issues) >= 6:
                return issues
        for tag, line in reversed(stack[-6:]):
            issues.append(f'HTML tag <{tag}> opened at line {line} is not closed')
            if len(issues) >= 6:
                break
        return issues

    @classmethod
    def _css_custom_property_names(cls, content: str) -> set[str]:
        text = re.sub(r'/\*[\s\S]*?\*/', '', str(content or ''))
        return {
            str(match.group('name') or '').strip()
            for match in _CSS_CUSTOM_PROPERTY_DEFINITION_RE.finditer(text)
            if str(match.group('name') or '').strip()
        }

    @classmethod
    def _css_custom_property_var_reference_suggestion(
        cls,
        defined_properties: set[str],
        reference_name: str,
    ) -> Optional[str]:
        name = str(reference_name or '').strip()
        if not name.startswith('---'):
            return None
        suggestion = f'--{name[3:]}'
        if suggestion in defined_properties and name not in defined_properties:
            return suggestion
        return None

    @staticmethod
    def _bounded_edit_distance(left: str, right: str, max_distance: int) -> int:
        if abs(len(left) - len(right)) > max_distance:
            return max_distance + 1
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            row_min = current[0]
            for right_index, right_char in enumerate(right, start=1):
                cost = 0 if left_char == right_char else 1
                value = min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
                current.append(value)
                row_min = min(row_min, value)
            if row_min > max_distance:
                return max_distance + 1
            previous = current
        return previous[-1]

    @classmethod
    def _css_custom_property_var_reference_near_miss_suggestion(
        cls,
        defined_properties: set[str],
        reference_name: str,
    ) -> Optional[str]:
        name = str(reference_name or '').strip()
        if not name.startswith('--') or name.startswith('---') or name in defined_properties:
            return None
        stem = name[2:].lower()
        if len(stem) < 5:
            return None
        best: tuple[int, int, str] | None = None
        for candidate in defined_properties:
            candidate_name = str(candidate or '').strip()
            if not candidate_name.startswith('--') or candidate_name.startswith('---'):
                continue
            candidate_stem = candidate_name[2:].lower()
            if len(candidate_stem) < 5:
                continue
            shared_prefix = 0
            for left_char, right_char in zip(stem, candidate_stem):
                if left_char != right_char:
                    break
                shared_prefix += 1
            if shared_prefix < min(4, len(stem), len(candidate_stem)):
                continue
            max_distance = 3 if max(len(stem), len(candidate_stem)) >= 7 else 2
            distance = cls._bounded_edit_distance(stem, candidate_stem, max_distance)
            if distance > max_distance:
                continue
            rank = (distance, -shared_prefix, candidate_name)
            if best is None or rank < best:
                best = rank
        if best is None:
            return None
        return best[2]

    @classmethod
    def _css_custom_property_var_reference_typos(
        cls,
        content: str,
    ) -> list[tuple[re.Match[str], str, str]]:
        text = re.sub(r'/\*[\s\S]*?\*/', '', str(content or ''))
        defined_properties = cls._css_custom_property_names(text)
        typos: list[tuple[re.Match[str], str, str]] = []
        for match in _CSS_VAR_REFERENCE_RE.finditer(text):
            name = str(match.group('name') or '').strip()
            suggestion = cls._css_custom_property_var_reference_suggestion(
                defined_properties,
                name,
            )
            if suggestion:
                typos.append((match, name, suggestion))
        return typos

    @classmethod
    def _css_custom_property_var_reference_near_misses(
        cls,
        content: str,
    ) -> list[tuple[re.Match[str], str, str]]:
        text = re.sub(r'/\*[\s\S]*?\*/', '', str(content or ''))
        defined_properties = cls._css_custom_property_names(text)
        near_misses: list[tuple[re.Match[str], str, str]] = []
        for match in _CSS_ANY_VAR_REFERENCE_RE.finditer(text):
            name = str(match.group('name') or '').strip()
            suggestion = cls._css_custom_property_var_reference_near_miss_suggestion(
                defined_properties,
                name,
            )
            if suggestion:
                near_misses.append((match, name, suggestion))
        return near_misses

    @staticmethod
    def _css_property_identifier_is_valid(value: str) -> bool:
        return bool(_CSS_PROPERTY_IDENTIFIER_RE.fullmatch(str(value or '').strip()))

    @staticmethod
    def _css_invalid_property_value_suggestion(property_name: str, value: str) -> Optional[str]:
        property_key = str(property_name or '').strip().lower()
        raw_value = re.sub(r'\s+', ' ', str(value or '').strip())
        important = ''
        important_match = re.search(r'\s*!important\s*$', raw_value, flags=re.IGNORECASE)
        if important_match:
            important = raw_value[important_match.start():]
            raw_value = raw_value[:important_match.start()].strip()
        value_key = raw_value.lower()
        suggestion = _CSS_KNOWN_INVALID_PROPERTY_VALUES.get((property_key, value_key))
        if not suggestion:
            return None
        return f'{suggestion}{important}'

    @classmethod
    def _css_malformed_declaration_name_suggestion(cls, property_name: str) -> Optional[str]:
        raw = re.sub(r'\s+', ' ', str(property_name or '').strip())
        if not raw:
            return None
        if '/*' in raw or '*/' in raw:
            return None
        normalized = raw.lower()
        if cls._css_property_identifier_is_valid(normalized):
            return None
        tokens = [token.strip() for token in re.split(r'\s+', raw) if token.strip()]
        if len(tokens) < 2:
            return None
        candidate = tokens[-1].lower()
        prefix = ''.join(tokens[:-1])
        if not cls._css_property_identifier_is_valid(candidate):
            return None
        if not re.search(r'[^A-Za-z0-9_-]', prefix):
            return None
        return candidate

    @classmethod
    def _css_declaration_name_formatting_tag_suggestion(cls, property_name: str) -> Optional[str]:
        raw = str(property_name or '').strip()
        if not raw or not _CSS_DECLARATION_NAME_FORMATTING_TAG_RE.search(raw):
            return None
        candidate = _CSS_DECLARATION_NAME_FORMATTING_TAG_RE.sub('', raw).strip().lower()
        if candidate == raw.lower():
            return None
        if not cls._css_property_identifier_is_valid(candidate):
            return None
        return candidate

    @staticmethod
    def _css_offset_inside_string_or_parentheses(content: str, offset: int) -> bool:
        text = str(content or '')
        try:
            bounded = max(0, min(int(offset), len(text)))
        except (TypeError, ValueError):
            bounded = 0
        quote: Optional[str] = None
        escaped = False
        in_comment = False
        paren_depth = 0
        index = 0
        while index < bounded:
            char = text[index]
            next_char = text[index + 1] if index + 1 < bounded else ''
            if in_comment:
                if char == '*' and next_char == '/':
                    in_comment = False
                    index += 2
                    continue
                index += 1
                continue
            if quote:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue
            if char == '/' and next_char == '*':
                in_comment = True
                index += 2
                continue
            if char in {'"', "'"}:
                quote = char
                index += 1
                continue
            if char == '(':
                paren_depth += 1
            elif char == ')' and paren_depth > 0:
                paren_depth -= 1
            index += 1
        return bool(quote or in_comment or paren_depth > 0)

    @classmethod
    def _css_syntax_sanity_issues(cls, content: str) -> list[str]:
        text = re.sub(r'/\*[\s\S]*?\*/', '', str(content or ''))
        if not text.strip():
            return []
        issues: list[str] = []
        stack: list[int] = []
        for index, char in enumerate(text):
            if char == '{':
                stack.append(cls._line_number_for_offset(text, index))
            elif char == '}':
                if stack:
                    stack.pop()
                else:
                    issues.append(f'CSS has closing brace without matching opening brace at line {cls._line_number_for_offset(text, index)}')
            if len(issues) >= 6:
                return issues
        for line in reversed(stack[-6:]):
            issues.append(f'CSS block opened at line {line} is not closed')
            if len(issues) >= 6:
                return issues

        for match in _CSS_DECLARATION_NAME_RE.finditer(text):
            if cls._css_offset_inside_string_or_parentheses(text, match.start()):
                continue
            property_name = re.sub(r'\s+', ' ', str(match.group('property') or '').strip()).lower()
            if not property_name or cls._css_property_identifier_is_valid(property_name):
                continue
            suggestion = cls._css_malformed_declaration_name_suggestion(property_name)
            if suggestion:
                issues.append(
                    f'CSS declaration name `{property_name}` at line {cls._line_number_for_offset(text, match.start("property"))} is invalid; use `{suggestion}`'
                )
            else:
                issues.append(
                    f'CSS declaration name `{property_name}` at line {cls._line_number_for_offset(text, match.start("property"))} is invalid'
                )
            if len(issues) >= 6:
                return issues
        for match in re.finditer(r'(?m)(?P<property>-{0,2}[a-z_][a-z0-9_-]*)\s*:', text, flags=re.IGNORECASE):
            property_name = str(match.group('property') or '').strip().lower()
            suggestion = _CSS_KNOWN_PROPERTY_TYPOS.get(property_name)
            if suggestion:
                issues.append(
                    f'CSS property `{property_name}` at line {cls._line_number_for_offset(text, match.start())} is likely invalid; use `{suggestion}`'
                )
            ambiguous = _CSS_AMBIGUOUS_PROPERTY_TYPOS.get(property_name)
            if ambiguous:
                issues.append(
                    f'CSS property `{property_name}` at line {cls._line_number_for_offset(text, match.start())} is invalid and ambiguous; choose one of {ambiguous}'
                )
            if len(issues) >= 6:
                return issues
        for match in _CSS_DECLARATION_VALUE_RE.finditer(text):
            if cls._css_offset_inside_string_or_parentheses(text, match.start('property')):
                continue
            property_name = str(match.group('property') or '').strip().lower()
            value = str(match.group('value') or '').strip()
            suggestion = cls._css_invalid_property_value_suggestion(property_name, value)
            if suggestion:
                normalized_value = re.sub(r'\s+', ' ', value)
                issues.append(
                    f'CSS property `{property_name}` value `{normalized_value}` at line {cls._line_number_for_offset(text, match.start("value"))} is invalid; use `{suggestion}`'
                )
                if len(issues) >= 6:
                    return issues
            if _CSS_HTML_ENTITY_FRAGMENT_IN_FUNCTION_RE.search(value):
                issues.append(
                    f'CSS declaration value for `{property_name}` at line {cls._line_number_for_offset(text, match.start("value"))} contains an HTML entity fragment inside a CSS function'
                )
                if len(issues) >= 6:
                    return issues
        for pattern, replacement, _kind in _CSS_INVALID_FUNCTION_TOKENS:
            for match in pattern.finditer(text):
                issues.append(
                    f'CSS token `{match.group(0)}` at line {cls._line_number_for_offset(text, match.start())} is likely invalid; use `{replacement}`'
                )
                if len(issues) >= 6:
                    return issues
        for match in _CSS_INVALID_VAR_SLASH_REFERENCE_RE.finditer(text):
            name = str(match.group('name') or '').strip()
            suggestion_name = name if name.startswith('--') else f'--{name}'
            issues.append(
                f'CSS token `{match.group(0)}` at line {cls._line_number_for_offset(text, match.start())} is likely invalid; use `var({suggestion_name})`'
            )
            if len(issues) >= 6:
                return issues
        for declaration_match in _CSS_BACKGROUND_DECLARATION_RE.finditer(text):
            value = str(declaration_match.group('value') or '')
            value_offset = declaration_match.start('value')
            for token_match in _CSS_INVALID_BACKGROUND_SHORTHAND_TOKEN_RE.finditer(value):
                token = str(token_match.group('token') or '').strip().lower()
                suggestion = _CSS_INVALID_BACKGROUND_SHORTHAND_TOKENS.get(token)
                if not suggestion:
                    continue
                issues.append(
                    f'CSS background token `{token}` at line {cls._line_number_for_offset(text, value_offset + token_match.start())} is likely invalid; use `{suggestion}`'
                )
                if len(issues) >= 6:
                    return issues
        for match, name, suggestion in cls._css_custom_property_var_reference_typos(text):
            issues.append(
                f'CSS variable reference `{name}` at line {cls._line_number_for_offset(text, match.start())} is likely invalid; use `{suggestion}`'
            )
            if len(issues) >= 6:
                return issues
        for match, name, suggestion in cls._css_custom_property_var_reference_near_misses(text):
            issues.append(
                f'CSS variable reference `{name}` at line {cls._line_number_for_offset(text, match.start())} is likely undefined; use `{suggestion}`'
            )
            if len(issues) >= 6:
                return issues
        return issues

    @classmethod
    def _json_syntax_sanity_issues(cls, content: str) -> list[str]:
        text = str(content or '').strip()
        if not text:
            return []
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return [f'JSON syntax error at line {exc.lineno} column {exc.colno}: {exc.msg}']
        return []

    @classmethod
    def text_artifact_syntax_sanity_issues_for_extension(
        cls,
        extension: str,
        content: str,
    ) -> list[str]:
        normalized = _normalize_text_artifact_extension(extension)
        if normalized in {'html', 'htm'}:
            return cls._html_syntax_sanity_issues(content)
        if normalized == 'css':
            return cls._css_syntax_sanity_issues(content)
        if normalized == 'json':
            return cls._json_syntax_sanity_issues(content)
        return []

    @classmethod
    def _repair_html_syntax_sanity_content(cls, content: str) -> tuple[str, list[dict[str, Any]]]:
        original = str(content or '')
        before_issues = cls._html_syntax_sanity_issues(original)
        if not before_issues:
            return original, []

        repairs: list[dict[str, Any]] = []

        def normalize_embedded_close(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            replacement = f'</{tag}>'
            repairs.append(
                {
                    'kind': 'html_malformed_closing_tag',
                    'from': match.group(0),
                    'to': replacement,
                }
            )
            return replacement

        def normalize_prefixed_close(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            replacement = f'</{tag}>'
            repairs.append(
                {
                    'kind': 'html_malformed_closing_tag',
                    'from': match.group(0),
                    'to': replacement,
                }
            )
            return replacement

        def normalize_known_suffix_close(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            prefix = str(match.group('prefix') or '').strip().lower()
            replacement = f'</{tag}>'
            repairs.append(
                {
                    'kind': 'html_malformed_closing_tag_known_suffix',
                    'from': match.group(0),
                    'to': replacement,
                    'prefix': prefix,
                }
            )
            return replacement

        def remove_stray_nested_close(match: re.Match[str]) -> str:
            repairs.append(
                {
                    'kind': 'html_stray_nested_closing_tag_fragment',
                    'from': match.group(0),
                    'to': '',
                }
            )
            return ''

        def remove_stray_formatting_close_in_open_tag(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            boundary = str(match.group('boundary') or '')
            replacement = f'<{tag}{boundary}'
            repairs.append(
                {
                    'kind': 'html_stray_formatting_close_in_open_tag',
                    'from': match.group(0),
                    'to': replacement,
                }
            )
            return replacement

        def remove_partial_open_tag_with_stray_formatting_close(match: re.Match[str]) -> str:
            repairs.append(
                {
                    'kind': 'html_partial_open_tag_with_stray_formatting_close',
                    'from': match.group(0),
                    'to': '',
                }
            )
            return ''

        def normalize_malformed_open_prefix_known_tag(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            boundary = str(match.group('boundary') or '')
            replacement = f'<{tag}{boundary}'
            repairs.append(
                {
                    'kind': 'html_malformed_opening_tag_prefix',
                    'from': match.group(0),
                    'to': replacement,
                    'prefix': str(match.group('prefix') or ''),
                    'tag': tag,
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_duplicate_open_angle_known_tag(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            boundary = str(match.group('boundary') or '')
            replacement = f'<{tag}{boundary}'
            repairs.append(
                {
                    'kind': 'html_duplicate_opening_angle_known_tag',
                    'from': match.group(0),
                    'to': replacement,
                    'tag': tag,
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def remove_stray_punctuation_pseudo_tag_before_known_tag(match: re.Match[str]) -> str:
            fragment = str(match.group('fragment') or '')
            tag = str(match.group('tag') or '').strip().lower()
            prefix = str(match.group('prefix') or '')
            repairs.append(
                {
                    'kind': 'html_stray_punctuation_pseudotag_removed',
                    'from': fragment,
                    'to': '',
                    'before_tag': tag,
                    'line': cls._line_number_for_offset(original, match.start('fragment')),
                }
            )
            return prefix

        def normalize_quoted_known_open_tag(match: re.Match[str]) -> str:
            prefix = str(match.group('prefix') or '')
            tag = str(match.group('tag') or '').strip().lower()
            attrs = str(match.group('attrs') or '')
            replacement = f'{prefix}<{tag}{attrs}>'
            repairs.append(
                {
                    'kind': 'html_quoted_known_opening_tag',
                    'from': match.group(0),
                    'to': replacement,
                    'tag': tag,
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_malformed_stylesheet_link_attrs(match: re.Match[str]) -> str:
            prefix = str(match.group('prefix') or '')
            href = str(match.group('href') or '').strip()
            suffix = str(match.group('suffix') or '')
            replacement = f'<link{prefix}rel="stylesheet" {href}{suffix}>'
            repairs.append(
                {
                    'kind': 'html_malformed_stylesheet_link_attributes',
                    'from': match.group(0),
                    'to': replacement,
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_invalid_stylesheet_link_rel_in_tag(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!') or match.group('close'):
                return token
            tag = str(match.group('tag') or '').strip().lower()
            attrs = str(match.group('attrs') or '')
            if tag != 'link' or not cls._html_link_stylesheet_rel_is_invalid(attrs):
                return token
            values = cls._html_attribute_values(attrs)
            rel = next((item for item in values.get('rel', []) if item.strip()), '')
            line = cls._line_number_for_offset(original, min(match.start(), len(original)))

            def replace_rel_attr(rel_match: re.Match[str]) -> str:
                quote = str(rel_match.group('quote') or '"')
                repairs.append(
                    {
                        'kind': 'html_invalid_stylesheet_link_rel',
                        'from': rel,
                        'to': 'stylesheet',
                        'line': line,
                    }
                )
                return f'rel={quote}stylesheet{quote}'

            replaced = re.sub(
                r'\brel\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
                replace_rel_attr,
                token,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if replaced != token:
                return replaced
            repairs.append(
                {
                    'kind': 'html_missing_stylesheet_link_rel',
                    'from': '',
                    'to': 'stylesheet',
                    'line': line,
                }
            )
            if token.endswith('/>'):
                return f'{token[:-2].rstrip()} rel="stylesheet" />'
            return f'{token[:-1].rstrip()} rel="stylesheet">'

        def normalize_malformed_class_attr_in_tag(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!') or match.group('close'):
                return token

            def replace_class_attr(attr_match: re.Match[str]) -> str:
                value = re.sub(r'\s+', ' ', str(attr_match.group('value') or '')).strip()
                quote = str(attr_match.group('quote') or '"')
                if not value:
                    return attr_match.group(0)
                replacement = f'class={quote}{value}{quote}'
                repairs.append(
                    {
                        'kind': 'html_malformed_class_attribute_assignment',
                        'from': attr_match.group(0),
                        'to': replacement,
                    }
                )
                return replacement

            return _HTML_MALFORMED_CLASS_ATTR_COLON_RE.sub(replace_class_attr, token)

        def normalize_bare_duplicate_attr_in_tag(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.startswith('<!--') or token.startswith('<!') or match.group('close'):
                return token
            line = cls._line_number_for_offset(original, min(match.start(), len(original)))

            def replace_bare_duplicate_attr(attr_match: re.Match[str]) -> str:
                attr_name = str(attr_match.group('name') or '').strip()
                if not attr_name:
                    return attr_match.group(0)
                replacement = f'{attr_name}='
                repairs.append(
                    {
                        'kind': 'html_bare_duplicate_attribute_token',
                        'from': attr_match.group(0),
                        'to': replacement,
                        'attribute': attr_name.lower(),
                        'line': line,
                    }
                )
                return replacement

            return _HTML_BARE_DUPLICATE_ATTRIBUTE_RE.sub(
                replace_bare_duplicate_attr,
                token,
            )

        def normalize_malformed_anchor_opening_tag(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip()
            attrs = str(match.group('attrs') or '')
            body = str(match.group('body') or '')
            replacement = f'<a{attrs}>{body}</a>'
            repairs.append(
                {
                    'kind': 'html_malformed_anchor_opening_tag',
                    'from_tag': tag,
                    'to_tag': 'a',
                    'removed_obsolete_close': bool(match.group('obsolete_close')),
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_malformed_formatted_anchor_fragment(match: re.Match[str]) -> str:
            attrs = re.sub(r'\s+', ' ', str(match.group('attrs') or '').strip())
            body = str(match.group('body') or '').strip()
            replacement = f'<a {attrs}>{body}</a>'
            repairs.append(
                {
                    'kind': 'html_malformed_formatted_anchor_fragment',
                    'from': match.group(0),
                    'to': replacement,
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_unsupported_href_element(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip()
            normalized_tag = tag.lower()
            attrs = str(match.group('attrs') or '')
            if (
                not normalized_tag
                or normalized_tag in _HTML_HREF_ALLOWED_TAGS
                or normalized_tag in _HTML_HREF_REPAIR_EXCLUDED_TAGS
                or ':' in normalized_tag
                or not _HTML_HREF_ATTR_RE.search(attrs)
            ):
                return match.group(0)
            body = str(match.group('body') or '')
            replacement = f'<a{attrs}>{body}</a>'
            repairs.append(
                {
                    'kind': 'html_unsupported_href_element_rewritten_to_anchor',
                    'from_tag': tag,
                    'to_tag': 'a',
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_unsupported_navigation_anchor_wrapper(match: re.Match[str]) -> str:
            issue = cls._html_unsupported_nav_anchor_wrapper_issue(repaired, match)
            if not issue:
                return match.group(0)
            tag = str(match.group('tag') or '').strip()
            anchor = str(match.group('anchor') or '')
            if not anchor:
                return match.group(0)
            repairs.append(
                {
                    'kind': 'html_unsupported_navigation_anchor_wrapper_unwrapped',
                    'from_tag': tag,
                    'to': 'anchor_only',
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return anchor

        def normalize_literal_unsupported_tag(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip()
            body = str(match.group('body') or '')
            repairs.append(
                {
                    'kind': 'html_literal_unsupported_tag_unwrapped',
                    'from_tag': tag,
                    'to': 'inner_content',
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return body

        def normalize_malformed_class_only_opening_tag(match: re.Match[str]) -> str:
            class_token = str(match.group('class') or '').strip()
            close_tag = str(match.group('close') or '').strip().lower()
            if not class_token or close_tag in _HTML_SYNTAX_VOID_TAGS:
                return match.group(0)
            body = str(match.group('body') or '')
            replacement = f'<{close_tag} class="{class_token}">{body}</{close_tag}>'
            repairs.append(
                {
                    'kind': 'html_malformed_class_only_opening_tag',
                    'from': match.group(0),
                    'to': replacement,
                    'class': class_token,
                    'tag': close_tag,
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def normalize_unknown_class_wrapper_tag(match: re.Match[str]) -> str:
            tag = str(match.group('tag') or '').strip().lower()
            if not tag or '-' in tag or ':' in tag:
                return match.group(0)
            attrs = str(match.group('attrs') or '')
            body = str(match.group('body') or '')
            replacement = f'<div{attrs}>{body}</div>'
            repairs.append(
                {
                    'kind': 'html_unknown_class_wrapper_tag_rewritten_to_div',
                    'from_tag': tag,
                    'to_tag': 'div',
                    'line': cls._line_number_for_offset(original, match.start()),
                }
            )
            return replacement

        def class_tokens_for_attrs(attrs: str) -> set[str]:
            match = re.search(
                r'\bclass\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
                str(attrs or ''),
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                return set()
            return {
                token.strip().lower()
                for token in re.split(r'\s+', str(match.group('value') or ''))
                if token.strip()
            }

        def has_repeated_sibling_hint(tokens: set[str]) -> bool:
            return any(_HTML_REPEATED_SIBLING_CLASS_HINT_RE.search(token) for token in tokens)

        def close_repeated_same_class_div_siblings(value: str) -> tuple[str, list[dict[str, Any]]]:
            insertions: list[tuple[int, str]] = []
            sibling_repairs: list[dict[str, Any]] = []
            stack: list[dict[str, Any]] = []
            for match in _HTML_SYNTAX_TAG_RE.finditer(value):
                token = match.group(0)
                if token.startswith('<!--') or token.startswith('<!'):
                    continue
                tag = str(match.group('tag') or '').strip().lower()
                if not tag:
                    continue
                line = cls._line_number_for_offset(value, match.start())
                if match.group('close'):
                    if stack and str(stack[-1].get('tag') or '') == tag:
                        stack.pop()
                    else:
                        for index in range(len(stack) - 1, -1, -1):
                            if str(stack[index].get('tag') or '') == tag:
                                del stack[index:]
                                break
                    continue
                if tag in _HTML_SYNTAX_VOID_TAGS or match.group('self'):
                    continue
                tokens = class_tokens_for_attrs(match.group('attrs') or '')
                if tag == 'div' and stack and str(stack[-1].get('tag') or '') == 'div':
                    previous_tokens = set(stack[-1].get('class_tokens') or set())
                    shared_tokens = tokens & previous_tokens
                    if (
                        shared_tokens
                        and has_repeated_sibling_hint(shared_tokens | tokens | previous_tokens)
                    ):
                        insertions.append((match.start(), '</div>\n'))
                        sibling_repairs.append(
                            {
                                'kind': 'html_missing_repeated_sibling_div_close',
                                'inserted': '</div>',
                                'before_line': line,
                                'class_tokens': sorted(shared_tokens),
                            }
                        )
                        stack.pop()
                stack.append(
                    {
                        'tag': tag,
                        'line': line,
                        'class_tokens': tokens,
                    }
                )
            if not insertions:
                return value, []
            repaired_value = value
            for index, insertion in reversed(insertions):
                repaired_value = f'{repaired_value[:index]}{insertion}{repaired_value[index:]}'
            return repaired_value, sibling_repairs

        def close_hero_header_before_landmarks(value: str) -> tuple[str, list[dict[str, Any]]]:
            insertions: list[tuple[int, str]] = []
            containment_repairs: list[dict[str, Any]] = []
            seen_header_starts: set[int] = set()
            for defect in cls._html_hero_header_landmark_containment_defects(value):
                try:
                    header_start = int(defect.get('header_start'))
                    landmark_start = int(defect.get('landmark_start'))
                except (TypeError, ValueError):
                    continue
                if header_start in seen_header_starts:
                    continue
                seen_header_starts.add(header_start)
                landmark = str(defect.get('landmark_tag') or '').strip().lower()
                insertions.append((landmark_start, '</header>\n'))
                containment_repairs.append(
                    {
                        'kind': 'html_hero_header_landmark_containment_closed',
                        'inserted': '</header>',
                        'before_tag': f'<{landmark}>',
                        'header_line': defect.get('header_line'),
                        'before_line': defect.get('landmark_line'),
                    }
                )
            if not insertions:
                return value, []
            repaired_value = value
            for index, insertion in reversed(insertions):
                repaired_value = f'{repaired_value[:index]}{insertion}{repaired_value[index:]}'
            return repaired_value, containment_repairs

        def close_intervening_tags_before_parent_close(value: str) -> tuple[str, list[dict[str, Any]]]:
            insertions: list[tuple[int, str]] = []
            parent_repairs: list[dict[str, Any]] = []
            stack: list[dict[str, Any]] = []
            for match in _HTML_SYNTAX_TAG_RE.finditer(value):
                token = match.group(0)
                if token.startswith('<!--') or token.startswith('<!'):
                    continue
                tag = str(match.group('tag') or '').strip().lower()
                if not tag:
                    continue
                line = cls._line_number_for_offset(value, match.start())
                if match.group('close'):
                    if stack and str(stack[-1].get('tag') or '') == tag:
                        stack.pop()
                        continue
                    ancestor_index = -1
                    for index in range(len(stack) - 1, -1, -1):
                        if str(stack[index].get('tag') or '') == tag:
                            ancestor_index = index
                            break
                    if ancestor_index < 0:
                        continue
                    missing = stack[ancestor_index + 1:]
                    repairable_missing = [
                        item
                        for item in missing
                        if str(item.get('tag') or '') not in {'html', 'head', 'body'}
                    ]
                    if repairable_missing:
                        closing_tags = [
                            str(item.get('tag') or '').strip().lower()
                            for item in reversed(repairable_missing)
                            if str(item.get('tag') or '').strip()
                        ]
                        inserted = ''.join(f'</{missing_tag}>' for missing_tag in closing_tags)
                        if inserted:
                            insertion = f'{inserted}\n'
                            insertions.append((match.start(), insertion))
                            parent_repairs.append(
                                {
                                    'kind': 'html_missing_intervening_close_before_parent',
                                    'inserted': inserted,
                                    'closed_tags': closing_tags,
                                    'before_parent_close': f'</{tag}>',
                                    'before_line': line,
                                }
                            )
                    del stack[ancestor_index:]
                    continue
                if tag in _HTML_SYNTAX_VOID_TAGS or match.group('self'):
                    continue
                stack.append({'tag': tag, 'line': line})
            if not insertions:
                return value, []
            repaired_value = value
            for index, insertion in reversed(insertions):
                repaired_value = f'{repaired_value[:index]}{insertion}{repaired_value[index:]}'
            return repaired_value, parent_repairs

        def remove_stray_unknown_closing_tags(value: str) -> tuple[str, list[dict[str, Any]]]:
            removals: list[tuple[int, int]] = []
            unknown_close_repairs: list[dict[str, Any]] = []
            stack: list[dict[str, Any]] = []
            for match in _HTML_SYNTAX_TAG_RE.finditer(value):
                token = match.group(0)
                if token.startswith('<!--') or token.startswith('<!'):
                    continue
                tag = str(match.group('tag') or '').strip().lower()
                if not tag:
                    continue
                line = cls._line_number_for_offset(value, match.start())
                if match.group('close'):
                    if stack and str(stack[-1].get('tag') or '') == tag:
                        stack.pop()
                        continue
                    open_tags = [str(item.get('tag') or '') for item in stack]
                    if tag in open_tags:
                        while stack and str(stack[-1].get('tag') or '') != tag:
                            stack.pop()
                        if stack:
                            stack.pop()
                        continue
                    if tag not in _HTML_SYNTAX_REPAIR_CLOSING_TAGS:
                        removals.append((match.start(), match.end()))
                        unknown_close_repairs.append(
                            {
                                'kind': 'html_stray_unknown_closing_tag_removed',
                                'from': token,
                                'line': line,
                            }
                        )
                    continue
                if tag in _HTML_SYNTAX_VOID_TAGS or match.group('self'):
                    continue
                stack.append({'tag': tag, 'line': line})
            if not removals:
                return value, []
            repaired_value = value
            for start, end in reversed(removals):
                repaired_value = f'{repaired_value[:start]}{repaired_value[end:]}'
            return repaired_value, unknown_close_repairs

        def remove_stray_known_closing_tags(value: str) -> tuple[str, list[dict[str, Any]]]:
            removals: list[tuple[int, int]] = []
            known_close_repairs: list[dict[str, Any]] = []
            stack: list[dict[str, Any]] = []
            for match in _HTML_SYNTAX_TAG_RE.finditer(value):
                token = match.group(0)
                if token.startswith('<!--') or token.startswith('<!'):
                    continue
                tag = str(match.group('tag') or '').strip().lower()
                if not tag:
                    continue
                line = cls._line_number_for_offset(value, match.start())
                if match.group('close'):
                    if stack and str(stack[-1].get('tag') or '') == tag:
                        stack.pop()
                        continue
                    open_tags = [str(item.get('tag') or '') for item in stack]
                    if tag in open_tags:
                        while stack and str(stack[-1].get('tag') or '') != tag:
                            stack.pop()
                        if stack:
                            stack.pop()
                        continue
                    if tag in _HTML_STRAY_KNOWN_CLOSING_TAG_REMOVABLE_TAGS:
                        removals.append((match.start(), match.end()))
                        known_close_repairs.append(
                            {
                                'kind': 'html_stray_known_closing_tag_removed',
                                'from': token,
                                'line': line,
                            }
                        )
                    continue
                if tag in _HTML_SYNTAX_VOID_TAGS or match.group('self'):
                    continue
                stack.append({'tag': tag, 'line': line})
            if not removals:
                return value, []
            repaired_value = value
            for start, end in reversed(removals):
                repaired_value = f'{repaired_value[:start]}{repaired_value[end:]}'
            return repaired_value, known_close_repairs

        repaired = _HTML_QUOTED_KNOWN_OPEN_TAG_RE.sub(
            normalize_quoted_known_open_tag,
            original,
        )
        repaired = _HTML_DUPLICATE_OPEN_ANGLE_KNOWN_TAG_RE.sub(
            normalize_duplicate_open_angle_known_tag,
            repaired,
        )
        repaired = _HTML_STRAY_PUNCTUATION_PSEUDO_TAG_BEFORE_KNOWN_TAG_RE.sub(
            remove_stray_punctuation_pseudo_tag_before_known_tag,
            repaired,
        )
        repaired = _HTML_MALFORMED_STYLESHEET_LINK_ATTR_RE.sub(
            normalize_malformed_stylesheet_link_attrs,
            repaired,
        )
        repaired = _HTML_SYNTAX_TAG_RE.sub(normalize_invalid_stylesheet_link_rel_in_tag, repaired)
        repaired = _HTML_SYNTAX_TAG_RE.sub(normalize_malformed_class_attr_in_tag, repaired)
        repaired = _HTML_SYNTAX_TAG_RE.sub(normalize_bare_duplicate_attr_in_tag, repaired)
        repaired = _HTML_MALFORMED_ANCHOR_OPEN_TAG_RE.sub(
            normalize_malformed_anchor_opening_tag,
            repaired,
        )
        repaired = _HTML_MALFORMED_FORMATTED_ANCHOR_FRAGMENT_RE.sub(
            normalize_malformed_formatted_anchor_fragment,
            repaired,
        )
        repaired = _HTML_UNSUPPORTED_HREF_ELEMENT_RE.sub(
            normalize_unsupported_href_element,
            repaired,
        )
        repaired = _HTML_UNSUPPORTED_NAV_ANCHOR_WRAPPER_RE.sub(
            normalize_unsupported_navigation_anchor_wrapper,
            repaired,
        )
        repaired = _HTML_LITERAL_UNSUPPORTED_TAG_RE.sub(
            normalize_literal_unsupported_tag,
            repaired,
        )
        repaired = _HTML_MALFORMED_CLASS_ONLY_OPEN_TAG_RE.sub(
            normalize_malformed_class_only_opening_tag,
            repaired,
        )
        repaired = _HTML_UNKNOWN_CLASS_WRAPPER_TAG_RE.sub(
            normalize_unknown_class_wrapper_tag,
            repaired,
        )
        repaired = _HTML_MALFORMED_CLOSE_EMBEDDED_TAG_RE.sub(normalize_embedded_close, repaired)
        repaired = _HTML_MALFORMED_CLOSE_PREFIX_RE.sub(normalize_prefixed_close, repaired)
        repaired = _HTML_MALFORMED_CLOSE_KNOWN_SUFFIX_RE.sub(
            normalize_known_suffix_close,
            repaired,
        )
        repaired = _HTML_STRAY_NESTED_CLOSE_RE.sub(remove_stray_nested_close, repaired)
        repaired = _HTML_PARTIAL_OPEN_TAG_WITH_STRAY_FORMATTING_CLOSE_RE.sub(
            remove_partial_open_tag_with_stray_formatting_close,
            repaired,
        )
        repaired = _HTML_MALFORMED_OPEN_PREFIX_KNOWN_TAG_RE.sub(
            normalize_malformed_open_prefix_known_tag,
            repaired,
        )
        repaired = _HTML_STRAY_FORMATTING_CLOSE_IN_OPEN_TAG_RE.sub(
            remove_stray_formatting_close_in_open_tag,
            repaired,
        )
        sibling_repaired, sibling_repairs = close_repeated_same_class_div_siblings(repaired)
        if sibling_repairs:
            repaired = sibling_repaired
            repairs.extend(sibling_repairs)
        containment_repaired, containment_repairs = close_hero_header_before_landmarks(repaired)
        if containment_repairs:
            repaired = containment_repaired
            repairs.extend(containment_repairs)
        parent_repaired, parent_repairs = close_intervening_tags_before_parent_close(repaired)
        if parent_repairs:
            repaired = parent_repaired
            repairs.extend(parent_repairs)
        unknown_close_repaired, unknown_close_repairs = remove_stray_unknown_closing_tags(repaired)
        if unknown_close_repairs:
            repaired = unknown_close_repaired
            repairs.extend(unknown_close_repairs)
        known_close_repaired, known_close_repairs = remove_stray_known_closing_tags(repaired)
        if known_close_repairs:
            repaired = known_close_repaired
            repairs.extend(known_close_repairs)
        if repaired == original or not repairs:
            return original, []
        after_issues = cls._html_syntax_sanity_issues(repaired)
        if len(after_issues) < len(before_issues):
            return repaired, repairs
        return original, []

    @classmethod
    def _repair_css_syntax_sanity_content(cls, content: str) -> tuple[str, list[dict[str, Any]]]:
        original = str(content or '')
        before_issues = cls._css_syntax_sanity_issues(original)
        if not before_issues:
            return original, []

        repairs: list[dict[str, Any]] = []
        repaired = original
        def replace_malformed_declaration_name(match: re.Match[str]) -> str:
            if cls._css_offset_inside_string_or_parentheses(repaired, match.start()):
                return match.group(0)
            property_name = re.sub(r'\s+', ' ', str(match.group('property') or '').strip()).lower()
            formatting_tag_suggestion = cls._css_declaration_name_formatting_tag_suggestion(property_name)
            if formatting_tag_suggestion:
                repairs.append(
                    {
                        'kind': 'css_formatting_tag_in_declaration_name',
                        'from': property_name,
                        'to': formatting_tag_suggestion,
                        'line': cls._line_number_for_offset(original, match.start('property')),
                    }
                )
                return f"{match.group('prefix')}{formatting_tag_suggestion}:"
            suggestion = cls._css_malformed_declaration_name_suggestion(property_name)
            if not suggestion:
                return match.group(0)
            repairs.append(
                {
                    'kind': 'css_malformed_declaration_name',
                    'from': property_name,
                    'to': suggestion,
                    'line': cls._line_number_for_offset(original, match.start('property')),
                }
            )
            return f"{match.group('prefix')}{suggestion}:"

        repaired = _CSS_DECLARATION_NAME_RE.sub(replace_malformed_declaration_name, repaired)

        def replace_invalid_property_value(match: re.Match[str]) -> str:
            if cls._css_offset_inside_string_or_parentheses(repaired, match.start('property')):
                return match.group(0)
            property_name = str(match.group('property') or '').strip()
            value = str(match.group('value') or '').strip()
            suggestion = cls._css_invalid_property_value_suggestion(property_name, value)
            if not suggestion:
                return match.group(0)
            repairs.append(
                {
                    'kind': 'css_invalid_property_value',
                    'property': property_name.lower(),
                    'from': re.sub(r'\s+', ' ', value),
                    'to': suggestion,
                    'line': cls._line_number_for_offset(original, match.start('value')),
                }
            )
            return f"{match.group('prefix')}{property_name}: {suggestion}"

        repaired = _CSS_DECLARATION_VALUE_RE.sub(replace_invalid_property_value, repaired)
        for pattern, replacement, kind in _CSS_INVALID_FUNCTION_TOKENS:
            def replace_token(
                match: re.Match[str],
                *,
                replacement: str = replacement,
                kind: str = kind,
            ) -> str:
                repairs.append(
                    {
                        'kind': kind,
                        'from': match.group(0),
                        'to': replacement,
                    }
                )
                return replacement

            repaired = pattern.sub(replace_token, repaired)

        def replace_invalid_var_slash_reference(match: re.Match[str]) -> str:
            name = str(match.group('name') or '').strip()
            if not name:
                return match.group(0)
            suggestion_name = name if name.startswith('--') else f'--{name}'
            replacement = f'var({suggestion_name})'
            repairs.append(
                {
                    'kind': 'css_invalid_var_slash_reference_token',
                    'from': match.group(0),
                    'to': replacement,
                }
            )
            return replacement

        repaired = _CSS_INVALID_VAR_SLASH_REFERENCE_RE.sub(
            replace_invalid_var_slash_reference,
            repaired,
        )

        def replace_invalid_background_tokens(match: re.Match[str]) -> str:
            property_name = str(match.group('property') or '').strip()
            value = str(match.group('value') or '')
            changed = False

            def replace_token(token_match: re.Match[str]) -> str:
                nonlocal changed
                token = str(token_match.group('token') or '').strip().lower()
                suggestion = _CSS_INVALID_BACKGROUND_SHORTHAND_TOKENS.get(token)
                if not suggestion:
                    return token_match.group(0)
                changed = True
                repairs.append(
                    {
                        'kind': 'css_invalid_background_shorthand_token',
                        'from': token_match.group(0),
                        'to': suggestion,
                        'property': property_name.lower(),
                    }
                )
                if property_name.lower() == 'background-image':
                    return ''
                return suggestion

            replacement_value = _CSS_INVALID_BACKGROUND_SHORTHAND_TOKEN_RE.sub(replace_token, value)
            if not changed:
                return match.group(0)
            replacement_value = re.sub(r'\s{2,}', ' ', replacement_value).strip()
            return f'{property_name}: {replacement_value}'

        repaired = _CSS_BACKGROUND_DECLARATION_RE.sub(replace_invalid_background_tokens, repaired)
        for typo, replacement in _CSS_KNOWN_PROPERTY_TYPOS.items():
            pattern = re.compile(rf'(?m)(?P<prefix>\b){re.escape(typo)}(?P<suffix>\s*:)', re.IGNORECASE)

            def replace_property(match: re.Match[str], *, typo: str = typo, replacement: str = replacement) -> str:
                repairs.append(
                    {
                        'kind': 'css_known_property_typo',
                        'from': typo,
                        'to': replacement,
                    }
                )
                return f"{match.group('prefix')}{replacement}{match.group('suffix')}"

            repaired = pattern.sub(replace_property, repaired)
        defined_properties = cls._css_custom_property_names(repaired)

        def replace_var_reference(match: re.Match[str]) -> str:
            name = str(match.group('name') or '').strip()
            suggestion = cls._css_custom_property_var_reference_suggestion(
                defined_properties,
                name,
            )
            if not suggestion:
                return match.group(0)
            replacement = f"var({suggestion}{match.group('suffix') or ')'}"
            repairs.append(
                {
                    'kind': 'css_custom_property_var_reference_typo',
                    'from': name,
                    'to': suggestion,
                }
            )
            return replacement

        repaired = _CSS_VAR_REFERENCE_RE.sub(replace_var_reference, repaired)

        def replace_near_miss_var_reference(match: re.Match[str]) -> str:
            name = str(match.group('name') or '').strip()
            suggestion = cls._css_custom_property_var_reference_near_miss_suggestion(
                defined_properties,
                name,
            )
            if not suggestion:
                return match.group(0)
            replacement = f"var({suggestion}{match.group('suffix') or ')'}"
            repairs.append(
                {
                    'kind': 'css_custom_property_var_reference_near_miss',
                    'from': name,
                    'to': suggestion,
                }
            )
            return replacement

        repaired = _CSS_ANY_VAR_REFERENCE_RE.sub(replace_near_miss_var_reference, repaired)
        if repaired == original or not repairs:
            return original, []
        after_issues = cls._css_syntax_sanity_issues(repaired)
        if len(after_issues) < len(before_issues):
            return repaired, repairs
        return original, []

    @classmethod
    def _repair_json_syntax_sanity_content(cls, content: str) -> tuple[str, list[dict[str, Any]]]:
        original = str(content or '')
        before_issues = cls._json_syntax_sanity_issues(original)
        if not before_issues:
            return original, []

        candidate = _strip_fenced_json_boundary(original.strip())
        repairs: list[dict[str, Any]] = []
        if candidate != original.strip():
            repairs.append({'kind': 'json_strip_fence'})
        without_trailing_commas = re.sub(r',(\s*[}\]])', r'\1', candidate)
        if without_trailing_commas != candidate:
            repairs.append({'kind': 'json_remove_trailing_commas'})
            candidate = without_trailing_commas
        if not repairs:
            return original, []
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return original, []
        repaired = json.dumps(parsed, indent=2, ensure_ascii=False) + '\n'
        if not cls._json_syntax_sanity_issues(repaired):
            return repaired, repairs
        return original, []

    @classmethod
    def repair_text_artifact_syntax_content(
        cls,
        extension: str,
        content: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        normalized = _normalize_text_artifact_extension(extension)
        if normalized in {'html', 'htm'}:
            return cls._repair_html_syntax_sanity_content(content)
        if normalized == 'css':
            return cls._repair_css_syntax_sanity_content(content)
        if normalized == 'json':
            return cls._repair_json_syntax_sanity_content(content)
        return str(content or ''), []

    @classmethod
    def _text_artifact_syntax_sanity_issues(
        cls,
        record: Mapping[str, Any],
        content: str,
    ) -> list[str]:
        return cls.text_artifact_syntax_sanity_issues_for_extension(
            _text_artifact_extension_from_record(record),
            content,
        )

    @staticmethod
    def _artifact_record_is_image(record: Mapping[str, Any]) -> bool:
        artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
        extension = _extension_from_path_like(_artifact_path(record))
        return artifact_type in {'image', 'png', 'jpg', 'jpeg', 'webp'} or extension in {'png', 'jpg', 'jpeg', 'webp'}

    @staticmethod
    def _html_hero_snippets(content: str) -> list[str]:
        text = str(content or '')
        snippets: list[str] = []
        for match in _HTML_HERO_BLOCK_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            attrs = str(match.group('attrs') or '')
            if tag != 'header' and not _HERO_LOCAL_IMAGE_SIGNAL_RE.search(attrs):
                continue
            snippet = match.group(0)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        return snippets

    @staticmethod
    def _css_hero_snippets(content: str) -> list[str]:
        text = str(content or '')
        snippets: list[str] = []
        for match in _CSS_HERO_BLOCK_RE.finditer(text):
            snippet = match.group(0)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        return snippets

    @classmethod
    def _hero_local_image_required(
        cls,
        prompt_text: str,
        *,
        html_contents: list[str],
        css_contents: list[str],
    ) -> bool:
        if _HERO_LOCAL_IMAGE_SIGNAL_RE.search(str(prompt_text or '')):
            return True
        return any(cls._html_hero_snippets(content) for content in html_contents) or any(
            cls._css_hero_snippets(content) for content in css_contents
        )

    @classmethod
    def _hero_snippets_have_artifact_link(
        cls,
        *,
        html_contents: list[str],
        css_contents: list[str],
        image_records: list[Mapping[str, Any]],
    ) -> bool:
        snippets: list[str] = []
        for content in html_contents:
            snippets.extend(cls._html_hero_snippets(content))
        for content in css_contents:
            snippets.extend(cls._css_hero_snippets(content))
        if not snippets:
            return False
        hero_content = '\n'.join(snippets)
        return any(
            token and token in hero_content
            for image_record in image_records
            for token in _artifact_link_tokens(image_record)
        )

    @classmethod
    def _preferred_hero_image_rebind_target(
        cls,
        *,
        html_records: list[Mapping[str, Any]],
        css_records: list[Mapping[str, Any]],
        content_by_id: Mapping[int, str],
    ) -> Optional[Mapping[str, Any]]:
        for record in css_records:
            if cls._css_hero_snippets(content_by_id.get(id(record), '')):
                return record
        for record in html_records:
            if cls._html_hero_snippets(content_by_id.get(id(record), '')):
                return record
        return css_records[0] if css_records else (html_records[0] if html_records else None)

    def _linked_artifact_binding_checks(
        self,
        *,
        request_payload: Optional[Mapping[str, Any]],
        artifact_payload: Optional[Mapping[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        payload = artifact_payload if isinstance(artifact_payload, Mapping) else {}
        if not payload:
            return []
        prompt_text = str(
            (request_payload or {}).get('prompt')
            or (request_payload or {}).get('input')
            or ((request_phase_graph.get('prompt_intent') or {}).get('normalized_prompt') if isinstance(request_phase_graph.get('prompt_intent'), Mapping) else '')
            or ''
        ).strip()
        records = self._artifact_records_for_link_binding(payload)
        text_records = [
            record
            for record in records
            if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
            and _text_artifact_extension_from_record(record) in _TEXT_LINK_ARTIFACT_EXTENSIONS
        ]
        if not text_records:
            return []
        html_records = [
            record for record in text_records
            if _text_artifact_extension_from_record(record) in {'html', 'htm'}
        ]
        link_asset_records = [
            record for record in text_records
            if _text_artifact_extension_from_record(record) in _HTML_LINK_TARGET_EXTENSIONS
        ]
        if link_asset_records:
            best_by_path: dict[str, Mapping[str, Any]] = {}

            def asset_record_score(record: Mapping[str, Any]) -> int:
                score = 0
                if str(record.get('text_artifact_source_name') or '').strip():
                    score += 8
                if isinstance(record.get('artifact_request'), Mapping):
                    score += 4
                if str(record.get('source') or '').strip() == 'late_fill_result':
                    score += 3
                source_name = _artifact_source_name(record).lower()
                path_stem = Path(_artifact_path(record)).stem.lower()
                if source_name and source_name in path_stem:
                    score += 2
                return score

            for record in link_asset_records:
                path = _artifact_path(record)
                if not path:
                    continue
                existing = best_by_path.get(path)
                if existing is None or asset_record_score(record) > asset_record_score(existing):
                    best_by_path[path] = record
            link_asset_records = list(best_by_path.values()) or link_asset_records
        image_records = [record for record in records if self._artifact_record_is_image(record)]
        if not html_records and not link_asset_records:
            return []

        content_by_id: dict[int, str] = {
            id(record): self._text_artifact_record_content(record)
            for record in text_records
        }

        def has_any_token(content: str, artifact: Mapping[str, Any]) -> bool:
            return any(token and token in content for token in _artifact_link_tokens(artifact))

        issues_by_target: dict[int, dict[str, Any]] = {}

        def add_issue(target: Mapping[str, Any], issue: str) -> None:
            if not issue:
                return
            key = id(target)
            payload = issues_by_target.setdefault(key, {'target': target, 'issues': []})
            if issue not in payload['issues']:
                payload['issues'].append(issue)

        for record in text_records:
            content = content_by_id.get(id(record), '')
            if _content_has_unresolved_link_placeholder(content):
                add_issue(record, 'text artifact still contains placeholder or unresolved link tokens')

        for html_record in html_records:
            html_content = content_by_id.get(id(html_record), '')
            if not html_content:
                continue
            grouped_assets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
            for asset_record in link_asset_records:
                if asset_record is html_record:
                    continue
                extension = _text_artifact_extension_from_record(asset_record)
                source_name = _artifact_source_name(asset_record).lower()
                if not source_name:
                    source_name = Path(_artifact_path(asset_record)).stem.lower()
                grouped_assets.setdefault((extension, source_name or extension), []).append(asset_record)
            for asset_group in grouped_assets.values():
                if not asset_group:
                    continue
                if not any(has_any_token(html_content, asset_record) for asset_record in asset_group):
                    asset_record = asset_group[0]
                    add_issue(
                        html_record,
                        f'HTML does not link concrete artifact `{_ollmo_relative_path(_artifact_path(asset_record)) or _artifact_source_name(asset_record)}`',
                    )

        if image_records and _LINKED_IMAGE_INTENT_RE.search(prompt_text):
            linked_text_records = text_records or html_records
            all_link_content = '\n\n'.join(
                content_by_id.get(id(record), '')
                for record in linked_text_records
                if content_by_id.get(id(record), '')
            )
            for image_record in image_records:
                if not all_link_content or not has_any_token(all_link_content, image_record):
                    preferred_target = next(
                        (
                            record for record in link_asset_records
                            if _text_artifact_extension_from_record(record) == 'css'
                        ),
                        html_records[0] if html_records else linked_text_records[0],
                    )
                    add_issue(
                        preferred_target,
                        f'Generated image artifact is not linked by its real saved path `{_ollmo_relative_path(_artifact_path(image_record))}`',
                    )
            css_records = [
                record for record in text_records
                if _text_artifact_extension_from_record(record) == 'css'
            ]
            html_contents = [content_by_id.get(id(record), '') for record in html_records]
            css_contents = [content_by_id.get(id(record), '') for record in css_records]
            if (
                self._hero_local_image_required(
                    prompt_text,
                    html_contents=html_contents,
                    css_contents=css_contents,
                )
                and not self._hero_snippets_have_artifact_link(
                    html_contents=html_contents,
                    css_contents=css_contents,
                    image_records=image_records,
                )
            ):
                preferred_target = self._preferred_hero_image_rebind_target(
                    html_records=html_records,
                    css_records=css_records,
                    content_by_id=content_by_id,
                )
                if preferred_target is not None:
                    add_issue(
                        preferred_target,
                        'Hero section does not use a concrete generated image artifact path; bind one saved generated image into the hero/header image or background',
                    )

        if not issues_by_target:
            return []

        artifact_lines = [
            f"- {str(record.get('type') or record.get('kind') or 'artifact').strip() or 'artifact'}: {_ollmo_relative_path(_artifact_path(record))}"
            for record in records
            if _artifact_path(record)
        ]
        checks: list[dict[str, Any]] = []
        for payload_item in issues_by_target.values():
            target = payload_item['target']
            target_extension = _text_artifact_extension_from_record(target) or 'txt'
            target_name = _artifact_source_name(target) or f'generated-{target_extension}'
            target_path = _artifact_path(target)
            issues = payload_item['issues']
            target_display_path = _ollmo_relative_path(target_path) if target_path else ''
            target_content_lines = self._bounded_repair_content_block(
                'Current saved target file content',
                content_by_id.get(id(target), ''),
            )
            content_payload = '\n'.join(
                [
                    f'Target text artifact: {target_display_path or target_name}',
                    'Unresolved linked artifact binding:',
                    *[f'- {issue}' for issue in issues],
                    'Resolved runtime artifacts:',
                    *artifact_lines,
                    *target_content_lines,
                    'Update only the target text artifact. Replace placeholders or guessed asset references with the concrete saved artifact paths or basenames above.',
                ]
            ).strip()
            check = {
                'check_kind': 'linked_artifact_binding',
                'status': 'pending',
                'evidence': 'unresolved_linked_artifact_binding',
                'reason': 'linked artifacts must be rebound to concrete runtime artifact paths before closure',
                'capability': CAPABILITY_CHAT,
                'output_type': 'text',
                'role': 'linked_artifact_binding_review',
                'branch_id': str(target.get('branch_id') or '').strip() or None,
                'phase_id': str(target.get('phase_id') or target.get('branch_id') or '').strip() or None,
                'repair_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
                'recovery_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
                'repair_action_reason': 'runtime artifacts exist but linked text artifacts still contain placeholders or unbound asset references',
                'content_payload': content_payload,
                'content_payload_source': 'closure_linked_artifact_binding_review',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': target_extension,
                'text_artifact_source_name': target_name,
                'text_artifact_source': 'closure_link_rebind',
                'text_artifact_target_path': target_path,
                'artifact_request': {
                    'extension': target_extension,
                    'source_name': target_name,
                    'source': 'closure_link_rebind',
                    'target_path': target_path,
                },
                'review_criteria': [
                    'linked_artifact_paths_resolve_to_runtime_artifacts',
                    'no_unresolved_asset_placeholders',
                ],
            }
            checks.append({key: value for key, value in check.items() if value not in (None, '', [], {})})
        return checks

    def _html_css_selector_binding_checks(
        self,
        *,
        artifact_payload: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = artifact_payload if isinstance(artifact_payload, Mapping) else {}
        if not payload:
            return []
        records = self._artifact_records_for_link_binding(payload)
        text_records = [
            record
            for record in records
            if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
        ]
        html_records = [
            record
            for record in text_records
            if _text_artifact_extension_from_record(record) in {'html', 'htm'}
        ]
        css_records = [
            record
            for record in text_records
            if _text_artifact_extension_from_record(record) == 'css'
        ]
        if not html_records or not css_records:
            return []

        content_by_id: dict[int, str] = {
            id(record): self._text_artifact_record_content(record)
            for record in [*html_records, *css_records]
        }
        checks: list[dict[str, Any]] = []
        for html_record in html_records:
            html_content = content_by_id.get(id(html_record), '')
            html_tokens = self._html_class_tokens(html_content)
            if len(html_tokens) < _HTML_CSS_SELECTOR_BINDING_MIN_TOKENS:
                continue
            linked_css_records = [
                css_record
                for css_record in css_records
                if self._html_links_css_artifact(html_content, css_record)
            ]
            if not linked_css_records and len(css_records) == 1:
                linked_css_records = css_records
            for css_record in linked_css_records:
                css_content = content_by_id.get(id(css_record), '')
                css_tokens = self._css_class_selector_tokens(css_content)
                if len(css_tokens) < _HTML_CSS_SELECTOR_BINDING_MIN_TOKENS:
                    continue
                html_without_css = sorted(html_tokens - css_tokens)
                css_without_html = sorted(css_tokens - html_tokens)
                html_missing_ratio = len(html_without_css) / max(1, len(html_tokens))
                css_unused_ratio = len(css_without_html) / max(1, len(css_tokens))
                if (
                    len(html_without_css) < _HTML_CSS_SELECTOR_BINDING_MIN_MISSING
                    or len(css_without_html) < _HTML_CSS_SELECTOR_BINDING_MIN_MISSING
                    or html_missing_ratio < _HTML_CSS_SELECTOR_BINDING_MIN_RATIO
                    or css_unused_ratio < _HTML_CSS_SELECTOR_BINDING_MIN_RATIO
                ):
                    continue

                css_path = _artifact_path(css_record)
                css_name = _artifact_source_name(css_record) or 'styles'
                css_display_path = _ollmo_relative_path(css_path) if css_path else ''
                html_path = _artifact_path(html_record)
                html_display_path = _ollmo_relative_path(html_path) if html_path else _artifact_source_name(html_record)
                content_payload = '\n'.join(
                    [
                        f'Target text artifact: {css_display_path or css_name}',
                        f'Linked HTML artifact: {html_display_path}',
                        'HTML/CSS selector binding drift:',
                        f'- HTML classes not targeted by linked CSS: {", ".join(html_without_css[:16])}',
                        f'- CSS class selectors not present in linked HTML: {", ".join(css_without_html[:16])}',
                        *self._bounded_repair_content_block('Current saved HTML file content', html_content),
                        *self._bounded_repair_content_block('Current saved CSS target file content', css_content),
                        'Update only the target CSS artifact. Preserve declarations, copy, valid runtime artifact links, and visual intent. Rename or add selectors only as needed so the CSS targets the actual saved HTML classes.',
                    ]
                ).strip()
                check = {
                    'check_kind': 'html_css_selector_binding',
                    'status': 'pending',
                    'evidence': 'html_css_selector_drift',
                    'reason': 'saved HTML and linked CSS use divergent class vocabularies before closure',
                    'capability': CAPABILITY_CHAT,
                    'output_type': 'text',
                    'role': 'html_css_selector_binding_repair',
                    'branch_id': str(css_record.get('branch_id') or '').strip() or None,
                    'phase_id': str(css_record.get('phase_id') or css_record.get('branch_id') or '').strip() or None,
                    'repair_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                    'recovery_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                    'repair_action_reason': 'linked CSS must be patched to target the saved HTML class vocabulary',
                    'content_payload': content_payload,
                    'content_payload_source': 'closure_html_css_selector_binding_review',
                    'stage_direction': 'materialize_requested_text_artifact',
                    'requires_artifact': True,
                    'text_artifact_extension': 'css',
                    'text_artifact_source_name': css_name,
                    'text_artifact_source': 'closure_selector_binding_repair',
                    'text_artifact_target_path': css_path,
                    'artifact_request': {
                        'extension': 'css',
                        'source_name': css_name,
                        'source': 'closure_selector_binding_repair',
                        'target_path': css_path,
                    },
                    'review_criteria': [
                        'html_css_selector_binding',
                        'preserve_runtime_artifact_links',
                    ],
                    'html_class_tokens_missing_css_selectors': html_without_css[:24],
                    'css_class_selectors_missing_html_usage': css_without_html[:24],
                    'html_class_count': len(html_tokens),
                    'css_selector_class_count': len(css_tokens),
                }
                checks.append({key: value for key, value in check.items() if value not in (None, '', [], {})})
        return checks

    def _text_artifact_syntax_sanity_checks(
        self,
        *,
        artifact_payload: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = artifact_payload if isinstance(artifact_payload, Mapping) else {}
        if not payload:
            return []
        records = self._artifact_records_for_link_binding(payload)
        checks: list[dict[str, Any]] = []
        for record in records:
            if str(record.get('type') or record.get('kind') or '').strip().lower() != 'text':
                continue
            extension = _text_artifact_extension_from_record(record)
            if extension not in {'html', 'htm', 'css', 'json'}:
                continue
            content = self._text_artifact_record_content(record)
            issues = self._text_artifact_syntax_sanity_issues(record, content)
            if not issues:
                continue
            target_path = _artifact_path(record)
            target_display_path = _ollmo_relative_path(target_path) if target_path else ''
            target_name = _artifact_source_name(record) or f'generated-{extension}'
            target_content_lines = self._bounded_repair_content_block(
                'Current saved target file content',
                content,
            )
            content_payload = '\n'.join(
                [
                    f'Target text artifact: {target_display_path or target_name}',
                    'Deterministic syntax sanity issues:',
                    *[f'- {issue}' for issue in issues],
                    *target_content_lines,
                    'Update only the target text artifact. Preserve valid runtime artifact links, copy, and layout intent. Fix the syntax defects above before closure.',
                ]
            ).strip()
            check = {
                'check_kind': 'text_artifact_syntax_sanity',
                'status': 'pending',
                'evidence': 'text_artifact_syntax_issue',
                'reason': 'saved HTML/CSS artifact has deterministic syntax defects before closure',
                'capability': CAPABILITY_CHAT,
                'output_type': 'text',
                'role': 'text_artifact_syntax_repair',
                'repair_scope': 'syntax_only',
                'resource_class': 'text_io',
                'dependency_policy': 'target_artifact_snapshot_only',
                'runtime_scheduling_context': {
                    'repair_scope': 'syntax_only',
                    'resource_class': 'text_io',
                    'dependency_policy': 'target_artifact_snapshot_only',
                },
                'branch_id': str(record.get('branch_id') or '').strip() or None,
                'phase_id': str(record.get('phase_id') or record.get('branch_id') or '').strip() or None,
                'repair_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'recovery_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'repair_action_reason': 'text artifact has deterministic syntax defects; repair the same bounded artifact branch',
                'content_payload': content_payload,
                'content_payload_source': 'closure_text_artifact_syntax_sanity',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': extension,
                'text_artifact_source_name': target_name,
                'text_artifact_source': 'closure_syntax_repair',
                'text_artifact_target_path': target_path,
                'artifact_request': {
                    'extension': extension,
                    'source_name': target_name,
                    'source': 'closure_syntax_repair',
                    'target_path': target_path,
                },
                'review_criteria': [
                    'html_css_syntax_sanity',
                    'preserve_runtime_artifact_links',
                ],
            }
            checks.append({key: value for key, value in check.items() if value not in (None, '', [], {})})
        return checks

    @staticmethod
    def _normalized_review_criterion(value: Any) -> str:
        return re.sub(r'[^a-z0-9_]+', '_', str(value or '').lower()).strip('_')

    @classmethod
    def _review_criterion_is_deterministic(cls, value: Any) -> bool:
        return cls._normalized_review_criterion(value) in _DETERMINISTIC_REVIEW_CRITERIA

    @staticmethod
    def _decision_contract_from_graph(request_phase_graph: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request_phase_graph, Mapping):
            return {}
        direct = request_phase_graph.get('decision_contract')
        if isinstance(direct, Mapping) and direct:
            return dict(direct)
        request_ir = (
            request_phase_graph.get('request_ir')
            if isinstance(request_phase_graph.get('request_ir'), Mapping)
            else {}
        )
        nested = request_ir.get('decision_contract') if isinstance(request_ir, Mapping) else {}
        if isinstance(nested, Mapping) and nested:
            return dict(nested)
        return {}

    @staticmethod
    def _compact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in dict(value or {}).items()
            if item not in (None, '', [], {})
        }

    @staticmethod
    def _identity_tokens(value: Mapping[str, Any]) -> set[str]:
        if not isinstance(value, Mapping):
            return set()
        tokens: set[str] = set()
        for key in ('task_id', 'workload_task_id', 'phase_id', 'branch_id', 'obligation_id', 'candidate_id'):
            token = str(value.get(key) or '').strip()
            if token:
                tokens.add(token)
        for container_key in ('workload_task_ref', 'output_obligation_ref'):
            container = value.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for key in ('task_id', 'workload_task_id', 'phase_id', 'branch_id', 'obligation_id'):
                token = str(container.get(key) or '').strip()
                if token:
                    tokens.add(token)
        return tokens

    @classmethod
    def _decision_contract_item_matches_check(cls, item: Mapping[str, Any], check: Mapping[str, Any]) -> bool:
        return bool(cls._identity_tokens(item) & cls._identity_tokens(check))

    @classmethod
    def _decision_contract_matching_items(
        cls,
        decision_contract: Mapping[str, Any],
        check: Mapping[str, Any],
        key: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(decision_contract, Mapping):
            return []
        raw_items = decision_contract.get(key)
        if not isinstance(raw_items, list):
            return []
        matches: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            if cls._decision_contract_item_matches_check(raw_item, check):
                matches.append(cls._compact_mapping(raw_item))
        return matches

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            return []
        values: list[str] = []
        for raw_item in raw_items:
            text = str(raw_item or '').strip()
            if text and text not in values:
                values.append(text)
        return values

    @classmethod
    def _merge_string_list(cls, existing: Any, incoming: Any) -> list[str]:
        return cls._string_list([*cls._string_list(existing), *cls._string_list(incoming)])

    @classmethod
    def _apply_semantic_lens_metadata(cls, updated: dict[str, Any], source: Mapping[str, Any]) -> None:
        if not isinstance(source, Mapping):
            return
        for key in ('semantic_review_lens', 'success_definition', 'semantic_review_lens_contract'):
            if source.get(key) not in (None, '', [], {}) and updated.get(key) in (None, '', [], {}):
                updated[key] = source.get(key)
        failure_modes = cls._merge_string_list(updated.get('failure_modes'), source.get('failure_modes'))
        if failure_modes:
            updated['failure_modes'] = failure_modes
        lens_evidence = cls._merge_string_list(
            updated.get('semantic_lens_evidence_requirements'),
            source.get('semantic_lens_evidence_requirements') or source.get('evidence_requirements'),
        )
        if lens_evidence:
            updated['semantic_lens_evidence_requirements'] = lens_evidence

    @classmethod
    def _decision_contract_review_payload(cls, decision_contract: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(decision_contract, Mapping) or not decision_contract:
            return {}
        accepted_learning = (
            decision_contract.get('accepted_learning')
            if isinstance(decision_contract.get('accepted_learning'), Mapping)
            else {}
        )
        payload = {
            'kind': 'ollmo.decision_contract_closure_review',
            'status': 'available',
            'decision_contract_version': decision_contract.get('decision_contract_version'),
            'candidate_count': decision_contract.get('candidate_count'),
            'open_obligation_ids': decision_contract.get('open_obligation_ids'),
            'blocked_obligation_ids': decision_contract.get('blocked_obligation_ids'),
            'closed_obligation_ids': decision_contract.get('closed_obligation_ids'),
            'reconsideration_candidate_count': len(decision_contract.get('reconsideration_candidates') or []),
            'promotion_suggestion_count': len(decision_contract.get('promotion_suggestions') or []),
            'waiver_candidate_count': len(decision_contract.get('waiver_candidates') or []),
            'repair_candidate_count': len(decision_contract.get('repair_candidates') or []),
            'semantic_review_candidate_count': len(decision_contract.get('semantic_review_candidates') or []),
            'supersession_candidate_count': len(decision_contract.get('supersession_candidates') or []),
            'supersession_record_count': len(decision_contract.get('supersession_records') or []),
            'block_resolution_reflex': decision_contract.get('block_resolution_reflex'),
            'block_resolution_signal_count': (
                (decision_contract.get('block_resolution_reflex') or {}).get('signal_count')
                if isinstance(decision_contract.get('block_resolution_reflex'), Mapping)
                else None
            ),
            'active_reconsideration_review': decision_contract.get('active_reconsideration_review'),
            'active_reconsideration_decision_count': len(decision_contract.get('active_reconsideration_decisions') or []),
            'semantic_quality_review': decision_contract.get('semantic_quality_review'),
            'semantic_quality_contract_count': len(decision_contract.get('semantic_quality_contracts') or []),
            'recursive_cycle_review': decision_contract.get('recursive_cycle_review'),
            'recursive_cycle_task_count': len(decision_contract.get('recursive_cycle_tasks') or []),
            'aspiration_review': decision_contract.get('aspiration_review'),
            'aspiration_frame_count': len(decision_contract.get('aspiration_frames') or []),
            'commitment_review': decision_contract.get('commitment_review'),
            'commitment_frame_count': len(decision_contract.get('commitment_frames') or []),
            'semantic_decision_review': decision_contract.get('semantic_decision_review'),
            'semantic_decision_proposal_count': len(decision_contract.get('semantic_decision_proposals') or []),
            'controlled_attention_review': decision_contract.get('controlled_attention_review'),
            'controlled_attention_frame_count': len(decision_contract.get('controlled_attention_frames') or []),
            'semantic_review_lens_review': decision_contract.get('semantic_review_lens_review'),
            'semantic_review_lens_count': len(decision_contract.get('semantic_review_lenses') or []),
            'next_decision_priorities': decision_contract.get('next_decision_priorities'),
            'semantic_planning_contract': decision_contract.get('semantic_planning_contract'),
            'accepted_learning': {
                key: accepted_learning.get(key)
                for key in ('status', 'authority', 'runtime_effect', 'hint_count', 'allowed_use')
                if accepted_learning.get(key) not in (None, '', [], {})
            },
            'authority': 'read_model_only_not_runtime_truth',
        }
        return cls._compact_mapping(payload)

    @classmethod
    def _decision_contract_guidance_payload(cls, decision_contract: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(decision_contract, Mapping) or not decision_contract:
            return {}
        payload = {
            'kind': 'ollmo.decision_contract_repair_guidance',
            'authority': 'advisory_until_matched_to_open_closure_check',
            'next_decision_priorities': decision_contract.get('next_decision_priorities'),
            'reconsideration_candidates': decision_contract.get('reconsideration_candidates'),
            'promotion_suggestions': decision_contract.get('promotion_suggestions'),
            'waiver_candidates': decision_contract.get('waiver_candidates'),
            'supersession_records': decision_contract.get('supersession_records'),
            'block_resolution_reflex': decision_contract.get('block_resolution_reflex'),
            'reconsideration_reflex_signals': decision_contract.get('reconsideration_reflex_signals'),
            'active_reconsideration_review': decision_contract.get('active_reconsideration_review'),
            'active_reconsideration_decisions': decision_contract.get('active_reconsideration_decisions'),
            'semantic_quality_review': decision_contract.get('semantic_quality_review'),
            'semantic_quality_contracts': decision_contract.get('semantic_quality_contracts'),
            'recursive_cycle_review': decision_contract.get('recursive_cycle_review'),
            'recursive_cycle_tasks': decision_contract.get('recursive_cycle_tasks'),
            'aspiration_review': decision_contract.get('aspiration_review'),
            'aspiration_frames': decision_contract.get('aspiration_frames'),
            'commitment_review': decision_contract.get('commitment_review'),
            'commitment_frames': decision_contract.get('commitment_frames'),
            'semantic_decision_review': decision_contract.get('semantic_decision_review'),
            'semantic_decision_proposals': decision_contract.get('semantic_decision_proposals'),
            'controlled_attention_review': decision_contract.get('controlled_attention_review'),
            'controlled_attention_frames': decision_contract.get('controlled_attention_frames'),
            'semantic_review_lens_review': decision_contract.get('semantic_review_lens_review'),
            'semantic_review_lenses': decision_contract.get('semantic_review_lenses'),
            'semantic_planning_contract': decision_contract.get('semantic_planning_contract'),
            'workload_proposal_coverage': decision_contract.get('workload_proposal_coverage'),
        }
        accepted_learning = decision_contract.get('accepted_learning')
        if isinstance(accepted_learning, Mapping):
            payload['accepted_learning'] = {
                key: accepted_learning.get(key)
                for key in ('status', 'authority', 'runtime_effect', 'hint_count', 'allowed_use')
                if accepted_learning.get(key) not in (None, '', [], {})
            }
        return cls._compact_mapping(payload)

    @classmethod
    def _surface_state_item(cls, source: Mapping[str, Any], *, category: str) -> dict[str, Any]:
        return cls._compact_mapping(
            {
                'category': category,
                'status': str(source.get('status') or '').strip().lower() or None,
                'check_kind': str(source.get('check_kind') or '').strip() or None,
                'obligation_id': str(source.get('obligation_id') or '').strip() or None,
                'task_id': str(source.get('task_id') or source.get('workload_task_id') or '').strip() or None,
                'phase_id': str(source.get('phase_id') or '').strip() or None,
                'branch_id': str(source.get('branch_id') or '').strip() or None,
                'candidate_id': str(source.get('candidate_id') or '').strip() or None,
                'capability': normalize_capability(source.get('capability')) or None,
                'output_type': str(source.get('output_type') or '').strip() or None,
                'reason': str(source.get('reason') or source.get('repair_action_reason') or '').strip() or None,
                'action': str(
                    source.get('aspiration_action')
                    or source.get('commitment_recommended_transition')
                    or source.get('commitment_action')
                    or source.get('active_reconsideration_action')
                    or source.get('controlled_attention_question')
                    or source.get('attention_question')
                    or source.get('repair_action')
                    or source.get('recovery_action')
                    or source.get('recommended_action')
                    or ''
                ).strip() or None,
                'review_type': str(
                    source.get('aspiration_review_type')
                    or source.get('commitment_review_type')
                    or source.get('active_reconsideration_review_type')
                    or source.get('review_type')
                    or source.get('scope')
                    or ''
                ).strip() or None,
                'semantic_review_lens': str(source.get('semantic_review_lens') or '').strip() or None,
                'success_definition': str(source.get('success_definition') or '').strip() or None,
                'priority': str(source.get('controlled_attention_priority') or source.get('priority') or '').strip() or None,
            }
        )

    @classmethod
    def _closure_surface_state(
        cls,
        *,
        review_status: str,
        reason: str,
        counts: Mapping[str, Any],
        checks: Sequence[Mapping[str, Any]],
        decision_contract: Mapping[str, Any],
        late_fill_status: str,
        pending_branches: Sequence[Mapping[str, Any]],
        semantic_review_required_count: int,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        category_counts: dict[str, int] = {
            'open': int(counts.get('pending') or 0) + int(counts.get('deferred') or 0),
            'blocked': int(counts.get('blocked') or 0),
            'waived': int(counts.get('waived') or 0),
            'superseded': int(counts.get('superseded') or 0),
            'completed': int(counts.get('fulfilled') or 0),
            'semantic_review_pending': int(semantic_review_required_count or 0),
        }
        repair_pending_count = 0
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            status = str(check.get('status') or '').strip().lower()
            if status in {'pending', 'planned', 'active', 'deferred'}:
                items.append(cls._surface_state_item(check, category='open'))
            elif status == 'blocked':
                items.append(cls._surface_state_item(check, category='blocked'))
            elif status == 'waived':
                items.append(cls._surface_state_item(check, category='waived'))
            elif status == 'superseded':
                items.append(cls._surface_state_item(check, category='superseded'))
            elif status == 'fulfilled':
                items.append(cls._surface_state_item(check, category='completed'))
            unresolved_for_repair = status in {'pending', 'planned', 'active', 'deferred', 'blocked', ''}
            if (
                unresolved_for_repair
                and (
                    check.get('repair_action') not in (None, '', [], {})
                    or check.get('repair_required') is True
                )
            ):
                repair_pending_count += 1
                items.append(cls._surface_state_item(check, category='repair_pending'))
            if check.get('semantic_review_required') is True:
                items.append(cls._surface_state_item(check, category='semantic_review_pending'))
        category_counts['repair_pending'] = repair_pending_count

        reconsideration_decisions = (
            decision_contract.get('active_reconsideration_decisions')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('active_reconsideration_decisions'), list)
            else []
        )
        reconsiderable_count = 0
        waived_candidate_count = 0
        superseded_candidate_count = 0
        repair_advisory_count = 0
        semantic_review_advisory_count = 0
        for decision in reconsideration_decisions:
            if not isinstance(decision, Mapping):
                continue
            category = str(decision.get('source_category') or '').strip().lower()
            if category in {'reconsiderable_candidate', 'waiver_candidate', 'supersession_candidate'}:
                reconsiderable_count += 1
                items.append(cls._surface_state_item(decision, category='reconsiderable'))
            elif category in {'waived_candidate'}:
                waived_candidate_count += 1
                items.append(cls._surface_state_item(decision, category='waived'))
            elif category in {'superseded_candidate'}:
                superseded_candidate_count += 1
                items.append(cls._surface_state_item(decision, category='superseded'))
            elif category in {'repair_candidate'}:
                repair_advisory_count += 1
                items.append(cls._surface_state_item(decision, category='repair_advisory'))
            elif category in {'semantic_review_candidate'}:
                semantic_review_advisory_count += 1
                items.append(cls._surface_state_item(decision, category='semantic_review_advisory'))
        category_counts['reconsiderable'] = reconsiderable_count
        category_counts['waived'] = int(category_counts.get('waived') or 0) + waived_candidate_count
        category_counts['superseded'] = int(category_counts.get('superseded') or 0) + superseded_candidate_count
        if repair_advisory_count:
            category_counts['repair_advisory'] = repair_advisory_count
        if semantic_review_advisory_count:
            category_counts['semantic_review_advisory'] = semantic_review_advisory_count

        attention_frames = (
            decision_contract.get('controlled_attention_frames')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('controlled_attention_frames'), list)
            else []
        )
        controlled_attention_count = 0
        for frame in attention_frames:
            if not isinstance(frame, Mapping):
                continue
            controlled_attention_count += 1
            items.append(cls._surface_state_item(frame, category='controlled_attention_advisory'))
        if controlled_attention_count:
            category_counts['controlled_attention_advisory'] = controlled_attention_count

        aspiration_frames = (
            decision_contract.get('aspiration_frames')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('aspiration_frames'), list)
            else []
        )
        aspiration_count = 0
        for frame in aspiration_frames:
            if not isinstance(frame, Mapping):
                continue
            aspiration_count += 1
            items.append(cls._surface_state_item(frame, category='aspiration_advisory'))
        if aspiration_count:
            category_counts['aspiration_advisory'] = aspiration_count

        commitment_frames = (
            decision_contract.get('commitment_frames')
            if isinstance(decision_contract, Mapping) and isinstance(decision_contract.get('commitment_frames'), list)
            else []
        )
        commitment_count = 0
        for frame in commitment_frames:
            if not isinstance(frame, Mapping):
                continue
            commitment_count += 1
            items.append(cls._surface_state_item(frame, category='commitment_advisory'))
        if commitment_count:
            category_counts['commitment_advisory'] = commitment_count

        if pending_branches:
            category_counts['late_fill_pending'] = len([item for item in pending_branches if isinstance(item, Mapping)])
        active_categories = [
            category
            for category, count in category_counts.items()
            if int(count or 0) > 0
        ]
        return cls._compact_mapping(
            {
                'kind': 'ollmo.surface_state',
                'status': str(review_status or '').strip().lower() or None,
                'reason': str(reason or '').strip() or None,
                'authority': 'runtime_projection_not_ui_truth',
                'late_fill_status': str(late_fill_status or '').strip().lower() or None,
                'category_counts': category_counts,
                'active_categories': active_categories,
                'items': items,
            }
        )

    @classmethod
    def _apply_decision_contract_to_check(
        cls,
        check: Mapping[str, Any],
        decision_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        updated = dict(check)
        if not isinstance(decision_contract, Mapping) or not decision_contract:
            return updated

        matched_repairs = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'repair_candidates',
        )
        if matched_repairs:
            flattened_repair_candidates: list[dict[str, Any]] = []
            for item in matched_repairs:
                if item.get('repair_action') not in (None, '', [], {}) or item.get('recovery_action') not in (None, '', [], {}):
                    flattened_repair_candidates.append(cls._compact_mapping(item))
                for candidate in item.get('repair_candidates') or []:
                    if isinstance(candidate, Mapping):
                        flattened_repair_candidates.append(cls._compact_mapping(candidate))
                for key in ('advisory_role', 'evidence_requirements', 'reconsideration_triggers', 'learning_hint_refs'):
                    if item.get(key) not in (None, '', [], {}) and updated.get(key) in (None, '', [], {}):
                        updated[key] = item.get(key)
                cls._apply_semantic_lens_metadata(updated, item)
            if flattened_repair_candidates:
                updated['decision_contract_repair_candidates'] = flattened_repair_candidates
                if updated.get('repair_candidates') in (None, '', [], {}):
                    updated['repair_candidates'] = flattened_repair_candidates
                if updated.get('repair_action') in (None, '', [], {}):
                    for candidate in flattened_repair_candidates:
                        action = normalize_recovery_suggested_action(candidate.get('repair_action') or candidate.get('recovery_action'))
                        if action:
                            updated['repair_action'] = action
                            updated['recovery_action'] = action
                            updated['repair_action_reason'] = (
                                str(candidate.get('reason') or '').strip()
                                or 'decision contract repair candidate matched this open closure check'
                            )
                            break

        matched_semantic_reviews = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'semantic_review_candidates',
        )
        if matched_semantic_reviews:
            updated['decision_contract_semantic_review_candidates'] = matched_semantic_reviews
            for item in matched_semantic_reviews:
                criteria = item.get('review_criteria') or item.get('semantic_review_criteria')
                merged = cls._merge_string_list(updated.get('semantic_review_criteria'), criteria)
                if merged:
                    updated['semantic_review_criteria'] = merged
                    updated['review_criteria'] = cls._merge_string_list(updated.get('review_criteria'), merged)
                for key in ('advisory_role', 'evidence_requirements', 'reconsideration_triggers', 'learning_hint_refs'):
                    if item.get(key) not in (None, '', [], {}) and updated.get(key) in (None, '', [], {}):
                        updated[key] = item.get(key)
                cls._apply_semantic_lens_metadata(updated, item)

        matched_quality_contracts = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'semantic_quality_contracts',
        )
        if matched_quality_contracts:
            updated['decision_contract_semantic_quality_contracts'] = matched_quality_contracts
            first_contract = matched_quality_contracts[0]
            quality_criteria = cls._string_list(first_contract.get('semantic_review_criteria')) or cls._string_list(first_contract.get('review_criteria'))
            quality_requires_semantic_review = any(
                not cls._review_criterion_is_deterministic(item)
                for item in quality_criteria
            )
            updated['semantic_quality_contract'] = first_contract
            updated['semantic_quality_status'] = (
                str(first_contract.get('status') or '').strip()
                or 'pending_semantic_review'
            )
            updated['semantic_quality_review_id'] = str(first_contract.get('quality_review_id') or '').strip() or None
            cls._apply_semantic_lens_metadata(updated, first_contract)
            updated['semantic_review_authority'] = 'advisory_until_promoted_semantic_verifier'
            if quality_requires_semantic_review:
                updated['semantic_review_required'] = True
                updated['semantic_review_action'] = RECOVERY_ACTION_SEMANTIC_REVIEW
            merged = cls._merge_string_list(
                updated.get('semantic_review_criteria'),
                quality_criteria,
            )
            if merged:
                updated['semantic_review_criteria'] = merged
                updated['review_criteria'] = cls._merge_string_list(updated.get('review_criteria'), merged)

        matched_block_resolution = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'reconsideration_reflex_signals',
        )
        if matched_block_resolution:
            updated['decision_contract_block_resolution_signals'] = matched_block_resolution
            first_signal = matched_block_resolution[0]
            updated['block_resolution_signal'] = first_signal
            updated['block_resolution_action'] = str(first_signal.get('action') or '').strip() or None
            updated['block_resolution_policy'] = (
                str(first_signal.get('resolution_policy') or '').strip()
                or 'right_sized_verified_state_transition'
            )
            updated['reconsideration_reflex'] = {
                'kind': 'ollmo.check_reconsideration_reflex',
                'authority': 'advisory_read_model_only',
                'signal_count': len(matched_block_resolution),
                'principle': 'the_solution_to_a_block_is_the_blocks_own_resolution',
            }

        matched_active_reconsideration = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'active_reconsideration_decisions',
        )
        if matched_active_reconsideration:
            updated['decision_contract_active_reconsideration_decisions'] = matched_active_reconsideration
            first_decision = matched_active_reconsideration[0]
            updated['active_reconsideration_decision'] = first_decision
            updated['active_reconsideration_action'] = str(first_decision.get('recommended_action') or '').strip() or None
            updated['active_reconsideration_review_type'] = str(first_decision.get('review_type') or '').strip() or None
            updated['active_reconsideration_review'] = {
                'kind': 'ollmo.check_active_reconsideration_review',
                'authority': 'advisory_read_model_only',
                'decision_count': len(matched_active_reconsideration),
                'status': 'pending_review',
            }

        matched_recursive_cycle = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'recursive_cycle_tasks',
        )
        if matched_recursive_cycle:
            updated['recursive_cycle_state'] = matched_recursive_cycle[0]
            cls._apply_semantic_lens_metadata(updated, matched_recursive_cycle[0])

        matched_aspiration = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'aspiration_frames',
        )
        if matched_aspiration:
            updated['decision_contract_aspiration_frames'] = matched_aspiration
            first_frame = matched_aspiration[0]
            updated['aspiration_frame'] = first_frame
            updated['aspiration_frame_id'] = str(first_frame.get('frame_id') or '').strip() or None
            updated['aspiration_action'] = str(first_frame.get('aspiration_action') or first_frame.get('recommended_action') or '').strip() or None
            updated['aspiration_reason'] = str(first_frame.get('reason') or '').strip() or None
            updated['aspiration_allowed_actions'] = first_frame.get('allowed_actions')
            cls._apply_semantic_lens_metadata(updated, first_frame)
            updated['aspiration_review'] = {
                'kind': 'ollmo.check_aspiration_review',
                'authority': 'advisory_read_model_only',
                'frame_count': len(matched_aspiration),
                'status': 'attention_pending',
            }

        matched_commitment = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'commitment_frames',
        )
        if matched_commitment:
            updated['decision_contract_commitment_frames'] = matched_commitment
            first_frame = matched_commitment[0]
            updated['commitment_frame'] = first_frame
            updated['commitment_frame_id'] = str(first_frame.get('frame_id') or '').strip() or None
            updated['commitment_action'] = str(first_frame.get('commitment_action') or first_frame.get('recommended_action') or '').strip() or None
            updated['commitment_recommended_transition'] = str(first_frame.get('recommended_transition') or '').strip() or None
            updated['commitment_reason'] = str(first_frame.get('reason') or '').strip() or None
            updated['commitment_allowed_transitions'] = first_frame.get('allowed_transitions')
            cls._apply_semantic_lens_metadata(updated, first_frame)
            updated['commitment_review'] = {
                'kind': 'ollmo.check_commitment_review',
                'authority': 'advisory_read_model_only',
                'frame_count': len(matched_commitment),
                'status': 'attention_pending',
            }

        matched_semantic_decisions = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'semantic_decision_proposals',
        )
        if matched_semantic_decisions:
            updated['decision_contract_semantic_decision_proposals'] = matched_semantic_decisions
            first_proposal = matched_semantic_decisions[0]
            updated['semantic_decision_proposal'] = first_proposal
            updated['semantic_decision_action'] = str(first_proposal.get('decision_action') or '').strip() or None
            confidence = first_proposal.get('confidence')
            if isinstance(confidence, (int, float)):
                updated['semantic_decision_confidence'] = confidence
            updated['semantic_decision_reason'] = str(first_proposal.get('reason') or '').strip() or None
            cls._apply_semantic_lens_metadata(updated, first_proposal)
            action = str(first_proposal.get('decision_action') or '').strip()
            if action == RECOVERY_ACTION_SEMANTIC_REVIEW or action == 'semantic_review':
                updated['semantic_review_required'] = True
                updated['semantic_review_action'] = RECOVERY_ACTION_SEMANTIC_REVIEW
                if updated.get('semantic_review_authority') in (None, '', [], {}):
                    updated['semantic_review_authority'] = 'advisory_until_promoted_semantic_verifier'
            normalized_action = (
                normalize_recovery_suggested_action(action)
                if action.lower() in RECOVERY_SUGGESTED_ACTIONS
                else ''
            )
            if normalized_action and updated.get('repair_action') in (None, '', [], {}):
                updated['repair_action'] = normalized_action
                updated['recovery_action'] = normalized_action
                updated['repair_action_reason'] = (
                    str(first_proposal.get('reason') or '').strip()
                    or 'semantic decision proposal matched this open closure check'
                )

        matched_controlled_attention = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'controlled_attention_frames',
        )
        if matched_controlled_attention:
            updated['decision_contract_controlled_attention_frames'] = matched_controlled_attention
            first_frame = matched_controlled_attention[0]
            updated['controlled_attention_frame'] = first_frame
            updated['controlled_attention_frame_id'] = str(first_frame.get('frame_id') or '').strip() or None
            updated['controlled_attention_scope'] = str(first_frame.get('scope') or '').strip() or None
            updated['controlled_attention_priority'] = str(first_frame.get('priority') or '').strip() or None
            updated['controlled_attention_question'] = str(first_frame.get('attention_question') or '').strip() or None
            updated['controlled_attention_allowed_transitions'] = first_frame.get('allowed_transitions')
            cls._apply_semantic_lens_metadata(updated, first_frame)

        matched_semantic_lenses = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'semantic_review_lenses',
        )
        if matched_semantic_lenses:
            updated['decision_contract_semantic_review_lenses'] = matched_semantic_lenses
            cls._apply_semantic_lens_metadata(updated, matched_semantic_lenses[0])

        matched_promotions = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'promotion_suggestions',
        )
        if matched_promotions and updated.get('promotion_suggestions') in (None, '', [], {}):
            flattened_promotions: list[dict[str, Any]] = []
            for item in matched_promotions:
                for candidate in item.get('promotion_suggestions') or []:
                    if isinstance(candidate, Mapping):
                        flattened_promotions.append(cls._compact_mapping(candidate))
            if flattened_promotions:
                updated['promotion_suggestions'] = flattened_promotions

        matched_waivers = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'waiver_candidates',
        )
        if matched_waivers and updated.get('waiver_candidates') in (None, '', [], {}):
            flattened_waivers: list[dict[str, Any]] = []
            for item in matched_waivers:
                for candidate in item.get('waiver_candidates') or []:
                    if isinstance(candidate, Mapping):
                        flattened_waivers.append(cls._compact_mapping(candidate))
            if flattened_waivers:
                updated['waiver_candidates'] = flattened_waivers

        matched_supersessions = cls._decision_contract_matching_items(
            decision_contract,
            updated,
            'supersession_candidates',
        )
        if matched_supersessions:
            flattened_supersession_candidates: list[dict[str, Any]] = []
            for item in matched_supersessions:
                for candidate in item.get('supersession_candidates') or []:
                    if isinstance(candidate, Mapping):
                        flattened_supersession_candidates.append(cls._compact_mapping(candidate))
            if flattened_supersession_candidates:
                updated['decision_contract_supersession_candidates'] = flattened_supersession_candidates
                if updated.get('supersession_candidates') in (None, '', [], {}):
                    updated['supersession_candidates'] = flattened_supersession_candidates
                updated['supersession_review_required'] = True
                updated['supersession_review_authority'] = 'closure_review_required'

        return updated

    def _apply_review_criteria_to_check(self, check: Mapping[str, Any]) -> dict[str, Any]:
        updated = dict(check)
        criteria = [
            str(item or '').strip()
            for item in (updated.get('review_criteria') or [])
            if str(item or '').strip()
        ] if isinstance(updated.get('review_criteria'), list) else []
        if not criteria:
            return updated
        updated['review_criteria'] = criteria
        status = str(updated.get('status') or '').strip().lower()
        capability = normalize_capability(updated.get('capability'))
        depends_on = [
            str(item or '').strip()
            for item in (updated.get('depends_on') or [])
            if str(item or '').strip()
        ] if isinstance(updated.get('depends_on'), list) else []
        if status in {'superseded', 'waived'}:
            updated['review_criteria_status'] = 'not_required'
            return updated
        issues: list[dict[str, Any]] = []
        for criterion in criteria:
            normalized = self._normalized_review_criterion(criterion)
            if (
                status == 'fulfilled'
                and capability == CAPABILITY_CHAT
                and depends_on
                and normalized in {'uses_dependency_evidence', 'does_not_restart_root_request'}
                and not self._check_has_dependency_runtime_evidence(updated)
            ):
                issues.append(
                    {
                        'criterion': criterion,
                        'code': 'dependency_evidence_not_bound_to_branch',
                        'severity': 'blocking',
                        'suggested_action': RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                        'reason': 'fulfilled text branch lacks branch-local dependency evidence',
                    }
                )
        if not issues:
            semantic_criteria = [
                criterion
                for criterion in criteria
                if not self._review_criterion_is_deterministic(criterion)
            ]
            if semantic_criteria:
                updated['review_criteria_status'] = 'semantic_review_required'
                updated['semantic_review_required'] = True
                if updated.get('semantic_review_authority') in (None, '', [], {}):
                    updated['semantic_review_authority'] = 'advisory'
                if updated.get('semantic_review_action') in (None, '', [], {}):
                    updated['semantic_review_action'] = RECOVERY_ACTION_SEMANTIC_REVIEW
                updated['semantic_review_criteria'] = semantic_criteria
            else:
                updated['review_criteria_status'] = 'passed'
            return updated
        updated['review_criteria_status'] = 'repair_required'
        updated['review_criteria_issues'] = issues
        updated['status'] = 'pending'
        updated['evidence'] = 'review_criteria_unverified'
        updated['blocked_by_dependency_input'] = True
        updated['repair_required'] = True
        updated['repair_action'] = RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN
        updated['recovery_action'] = RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN
        updated['repair_action_reason'] = 'review criteria require dependency evidence before freeze'
        return updated

    @staticmethod
    def _repair_contract_id(item: Mapping[str, Any], *, index: int) -> str:
        raw_identity = (
            item.get('branch_id')
            or item.get('phase_id')
            or item.get('obligation_id')
            or item.get('task_id')
            or item.get('capability')
            or f'item-{index}'
        )
        token = re.sub(r'[^a-z0-9_]+', '-', str(raw_identity or '').strip().lower())
        token = re.sub(r'-+', '-', token).strip('-') or f'item-{index}'
        return f'repair-contract-{token}'

    @staticmethod
    def _repair_item_has_concrete_input_evidence(item: Mapping[str, Any]) -> bool:
        return repair_item_has_concrete_input_evidence(item)

    @classmethod
    def _repair_contract_execution_policy(cls, item: Mapping[str, Any]) -> tuple[str, bool]:
        policy = classify_repair_execution_policy(item)
        return (
            str(policy.get('execution_policy') or 'manual_review_required'),
            bool(policy.get('auto_execute')),
        )

    @classmethod
    def _repair_contract_resolution_state(
        cls,
        item: Mapping[str, Any],
        *,
        execution_policy: str,
        auto_execute: bool,
    ) -> dict[str, Any]:
        policy = classify_repair_execution_policy(item)
        return {
            key: value
            for key, value in policy.items()
            if key not in {'execution_policy', 'auto_execute'}
        }

    @staticmethod
    def _expand_counted_repair_items(repair_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for item in repair_items:
            if not isinstance(item, Mapping):
                continue
            try:
                missing_count = int(item.get('missing_count') or 1)
            except (TypeError, ValueError):
                missing_count = 1
            missing_count = max(1, missing_count)
            action = normalize_recovery_suggested_action(
                item.get('repair_action') or item.get('recovery_action')
            )
            should_expand = bool(
                missing_count > 1
                and str(item.get('check_kind') or '').strip() == 'intent_graph_adequacy'
                and action == RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS
                and normalize_capability(item.get('capability'))
                and str(item.get('output_type') or '').strip().lower() in {'audio', 'image'}
            )
            if not should_expand:
                expanded.append(dict(item))
                continue

            parent_obligation_id = str(item.get('obligation_id') or '').strip() or None
            raw_identity = str(
                item.get('branch_id')
                or item.get('phase_id')
                or parent_obligation_id
                or item.get('output_type')
                or 'material-output'
            ).strip()
            identity = re.sub(r'[^a-z0-9_]+', '-', raw_identity.lower())
            identity = re.sub(r'-+', '-', identity).strip('-') or 'material-output'
            if not identity.startswith('repair-'):
                identity = f'repair-{identity}'
            for occurrence in range(1, missing_count + 1):
                branch_id = f'{identity}-{occurrence}'
                occurrence_item = dict(item)
                occurrence_item.update(
                    {
                        'branch_id': branch_id,
                        'phase_id': branch_id,
                        'missing_count': 1,
                        'total_missing_count': missing_count,
                        'repair_occurrence_index': occurrence,
                        'repair_occurrence_count': missing_count,
                    }
                )
                if parent_obligation_id:
                    occurrence_item['parent_obligation_id'] = parent_obligation_id
                expanded.append(occurrence_item)
        return expanded

    @classmethod
    def _repair_rebuild_contracts(cls, repair_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contracts: list[dict[str, Any]] = []
        for index, item in enumerate(repair_items, start=1):
            if not isinstance(item, Mapping):
                continue
            action = normalize_recovery_suggested_action(item.get('repair_action') or item.get('recovery_action'))
            execution_policy, auto_execute = cls._repair_contract_execution_policy(item)
            resolution_state = cls._repair_contract_resolution_state(
                item,
                execution_policy=execution_policy,
                auto_execute=auto_execute,
            )
            contract = {
                'kind': 'ollmo.repair_rebuild_contract',
                'contract_id': cls._repair_contract_id(item, index=index),
                'status': 'promoted',
                'authority': 'closure_review_runtime_truth',
                'promotion_source': 'graph_closure_review',
                'promotion_reason': str(item.get('repair_action_reason') or item.get('reason') or '').strip()
                or 'Closure Review found open promoted work that needs repair or rebuild',
                'repair_action': action,
                'execution_policy': execution_policy,
                'auto_execute': auto_execute,
                **resolution_state,
                'check_kind': item.get('check_kind'),
                'task_id': item.get('task_id') or item.get('workload_task_id'),
                'phase_id': item.get('phase_id'),
                'branch_id': item.get('branch_id'),
                'obligation_id': item.get('obligation_id'),
                'parent_obligation_id': item.get('parent_obligation_id'),
                'capability': normalize_capability(item.get('capability')),
                'output_type': str(item.get('output_type') or '').strip() or None,
                'missing_count': item.get('missing_count'),
                'total_missing_count': item.get('total_missing_count'),
                'repair_occurrence_index': item.get('repair_occurrence_index'),
                'repair_occurrence_count': item.get('repair_occurrence_count'),
                'depends_on': item.get('depends_on'),
                'repair_scope': item.get('repair_scope'),
                'resource_class': item.get('resource_class'),
                'dependency_policy': item.get('dependency_policy'),
                'runtime_scheduling_context': item.get('runtime_scheduling_context'),
                'allow_gpu_heavy_concurrency': item.get('allow_gpu_heavy_concurrency'),
                'execution_contract': item.get('execution_contract'),
                'workload_task_ref': item.get('workload_task_ref'),
                'output_obligation_ref': item.get('output_obligation_ref'),
                'output_contract': item.get('output_contract'),
                'input_refs': item.get('input_refs'),
                'review_criteria': item.get('review_criteria'),
                'semantic_review_criteria': item.get('semantic_review_criteria'),
                'semantic_review_verdict': item.get('semantic_review_verdict'),
                'semantic_review_verdict_status': item.get('semantic_review_verdict_status'),
                'semantic_review_recommended_transition': item.get('semantic_review_recommended_transition'),
                'branch_semantic_review': item.get('branch_semantic_review'),
                'branch_semantic_review_branch_id': item.get('branch_semantic_review_branch_id'),
                'branch_semantic_review_phase_id': item.get('branch_semantic_review_phase_id'),
                'branch_semantic_review_status': item.get('branch_semantic_review_status'),
                'branch_semantic_review_reason': item.get('branch_semantic_review_reason'),
                'branch_semantic_review_source_branch_id': item.get('branch_semantic_review_source_branch_id'),
                'branch_semantic_review_source_phase_id': item.get('branch_semantic_review_source_phase_id'),
                'evidence_requirements': item.get('evidence_requirements'),
                'content_payload': item.get('content_payload'),
                'content_payload_source': item.get('content_payload_source'),
                'artifact_prompt': item.get('artifact_prompt'),
                'artifact_prompt_source': item.get('artifact_prompt_source'),
                'batch_prompts': item.get('batch_prompts'),
                'artifact_request': item.get('artifact_request'),
                'requires_artifact': item.get('requires_artifact'),
                'text_artifact_extension': item.get('text_artifact_extension'),
                'text_artifact_source_name': item.get('text_artifact_source_name'),
                'text_artifact_source': item.get('text_artifact_source'),
                'stage_direction': item.get('stage_direction'),
                'global_semantic_closure_review': item.get('global_semantic_closure_review'),
                'global_semantic_closure_proposal': item.get('global_semantic_closure_proposal'),
                'global_semantic_closure_status': item.get('global_semantic_closure_status'),
                'global_semantic_closure_reason': item.get('global_semantic_closure_reason'),
                'global_semantic_closure_confidence': item.get('global_semantic_closure_confidence'),
                'aspiration_review': item.get('aspiration_review'),
                'aspiration_frame': item.get('aspiration_frame'),
                'aspiration_frame_id': item.get('aspiration_frame_id'),
                'aspiration_action': item.get('aspiration_action'),
                'aspiration_reason': item.get('aspiration_reason'),
                'aspiration_allowed_actions': item.get('aspiration_allowed_actions'),
                'commitment_review': item.get('commitment_review'),
                'commitment_frame': item.get('commitment_frame'),
                'commitment_frame_id': item.get('commitment_frame_id'),
                'commitment_action': item.get('commitment_action'),
                'commitment_recommended_transition': item.get('commitment_recommended_transition'),
                'commitment_reason': item.get('commitment_reason'),
                'commitment_allowed_transitions': item.get('commitment_allowed_transitions'),
                'controlled_attention_review': item.get('controlled_attention_review'),
                'controlled_attention_frame': item.get('controlled_attention_frame'),
                'controlled_attention_frame_id': item.get('controlled_attention_frame_id'),
                'controlled_attention_scope': item.get('controlled_attention_scope'),
                'controlled_attention_priority': item.get('controlled_attention_priority'),
                'controlled_attention_question': item.get('controlled_attention_question'),
                'controlled_attention_allowed_transitions': item.get('controlled_attention_allowed_transitions'),
            }
            contracts.append(cls._compact_mapping(contract))
        return contracts

    def build_ghost_repair_feedback(
        self,
        *,
        review_status: str,
        reason: str,
        checks: list[dict[str, Any]],
        request_phase_graph: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        if review_status not in {'pending', 'blocked'}:
            return None
        current_phase_id = str(request_phase_graph.get('current_phase_id') or '').strip()
        current_phase_capability = normalize_capability(request_phase_graph.get('current_phase_capability'))
        decision_contract = self._decision_contract_from_graph(request_phase_graph)
        decision_contract_guidance = self._decision_contract_guidance_payload(decision_contract)
        graph_intent = (
            request_phase_graph.get('prompt_intent')
            if isinstance(request_phase_graph.get('prompt_intent'), Mapping)
            else {}
        )
        downstream_capabilities = {
            normalize_capability(item)
            for item in (graph_intent.get('downstream_follow_up_capabilities') or [])
            if normalize_capability(item)
        }
        graph_records: list[Mapping[str, Any]] = []
        for key in ('downstream_branches', 'phases', 'output_obligations'):
            records = request_phase_graph.get(key)
            if isinstance(records, list):
                graph_records.extend(item for item in records if isinstance(item, Mapping))

        def _normalized_depends_on(raw_value: Any) -> list[str]:
            if not isinstance(raw_value, list):
                return []
            return [
                str(item or '').strip()
                for item in raw_value
                if str(item or '').strip()
            ]

        def _exemplar_depends_on(capability: Optional[str]) -> list[str]:
            normalized_capability = normalize_capability(capability)
            if not normalized_capability:
                return []
            for record in graph_records:
                if normalize_capability(record.get('capability')) != normalized_capability:
                    continue
                depends_on = _normalized_depends_on(record.get('depends_on'))
                if depends_on:
                    return depends_on
            return []

        def _inferred_repair_depends_on(capability: Optional[str]) -> list[str]:
            normalized_capability = normalize_capability(capability)
            exemplar = _exemplar_depends_on(normalized_capability)
            if exemplar:
                return exemplar
            if not current_phase_id or current_phase_capability != CAPABILITY_CHAT:
                return []
            if normalized_capability == CAPABILITY_IMAGE_GENERATION and (
                graph_intent.get('text_preparation_before_visual_output')
                or CAPABILITY_IMAGE_GENERATION in downstream_capabilities
            ):
                return [current_phase_id]
            if normalized_capability == CAPABILITY_TEXT_TO_SPEECH and (
                graph_intent.get('text_preparation_before_audio_output')
                or CAPABILITY_TEXT_TO_SPEECH in downstream_capabilities
            ):
                return [current_phase_id]
            return []

        def _repair_check_uses_target_artifact_snapshot_only(raw_check: Mapping[str, Any]) -> bool:
            context = (
                raw_check.get('runtime_scheduling_context')
                if isinstance(raw_check.get('runtime_scheduling_context'), Mapping)
                else {}
            )
            dependency_policy = str(
                raw_check.get('dependency_policy')
                or context.get('dependency_policy')
                or ''
            ).strip().lower()
            if dependency_policy == 'target_artifact_snapshot_only':
                return True
            repair_scope = str(
                raw_check.get('repair_scope')
                or context.get('repair_scope')
                or ''
            ).strip().lower()
            resource_class = str(
                raw_check.get('resource_class')
                or context.get('resource_class')
                or ''
            ).strip().lower()
            if repair_scope in {'syntax_only', 'text_artifact_syntax'} and resource_class in {'text_io', 'local_text_io'}:
                return True
            check_kind = str(raw_check.get('check_kind') or '').strip()
            payload_source = str(raw_check.get('content_payload_source') or '').strip()
            artifact_request = (
                raw_check.get('artifact_request')
                if isinstance(raw_check.get('artifact_request'), Mapping)
                else {}
            )
            target_path = str(
                raw_check.get('text_artifact_target_path')
                or artifact_request.get('target_path')
                or ''
            ).strip()
            return bool(
                target_path
                and (
                    check_kind == 'text_artifact_syntax_sanity'
                    or payload_source == 'closure_text_artifact_syntax_sanity'
                )
            )

        repair_items: list[dict[str, Any]] = []
        for raw_check in checks:
            if not isinstance(raw_check, Mapping):
                continue
            status = str(raw_check.get('status') or '').strip().lower()
            if status not in {'pending', 'planned', 'active', 'deferred', 'blocked'}:
                continue
            capability = normalize_capability(raw_check.get('capability'))
            repair_action, repair_action_reason = self._closure_check_recovery_action(raw_check)
            if capability == CAPABILITY_CHAT and not repair_action:
                capability = None
            check_kind = str(raw_check.get('check_kind') or '').strip()
            if not capability and check_kind not in {'truth_guard', 'intent_graph_adequacy'} and not repair_action:
                continue
            depends_on = _normalized_depends_on(raw_check.get('depends_on'))
            if not depends_on and not _repair_check_uses_target_artifact_snapshot_only(raw_check):
                depends_on = _inferred_repair_depends_on(capability)
            repair_item = {
                'check_kind': check_kind or 'graph_obligation',
                'status': status,
                'capability': capability,
                'output_type': str(raw_check.get('output_type') or '').strip() or None,
                'role': str(raw_check.get('role') or '').strip() or None,
                'depends_on': depends_on or None,
                'missing_count': raw_check.get('missing_count'),
                'evidence': str(raw_check.get('evidence') or '').strip() or None,
                'reason': str(raw_check.get('reason') or '').strip() or None,
                'obligation_id': str(raw_check.get('obligation_id') or '').strip() or None,
                'phase_id': str(raw_check.get('phase_id') or '').strip() or None,
                'branch_id': str(raw_check.get('branch_id') or '').strip() or None,
                'repair_action': repair_action,
                'recovery_action': repair_action,
                'repair_action_reason': repair_action_reason,
            }
            for key in _CLOSURE_REPAIR_PAYLOAD_KEYS:
                value = raw_check.get(key)
                if value not in (None, '', [], {}):
                    repair_item[key] = value
            for key in (
                'failed_instance_id',
                'exclude_instance_ids',
                'blocked_by_dependency_input',
                'blocked_by_branch_contract',
            ):
                value = raw_check.get(key)
                if value not in (None, '', [], {}):
                    repair_item[key] = value
            repair_items.append(
                {
                    key: value
                    for key, value in repair_item.items()
                    if value not in (None, '', [], {})
                }
            )
        if not repair_items:
            return None
        repair_items = self._expand_counted_repair_items(repair_items)
        next_actions: list[str] = []
        for item in repair_items:
            raw_action = str(item.get('repair_action') or '').strip()
            if not raw_action:
                continue
            action = normalize_recovery_suggested_action(raw_action)
            if action not in next_actions:
                next_actions.append(action)
        if not next_actions:
            next_actions.append(RECOVERY_ACTION_MANUAL_REVIEW)
        repair_contracts = self._repair_rebuild_contracts(repair_items)
        contracts_by_identity = {
            str(
                contract.get('branch_id')
                or contract.get('phase_id')
                or contract.get('obligation_id')
                or contract.get('task_id')
                or contract.get('contract_id')
                or ''
            ).strip(): contract
            for contract in repair_contracts
            if str(
                contract.get('branch_id')
                or contract.get('phase_id')
                or contract.get('obligation_id')
                or contract.get('task_id')
                or contract.get('contract_id')
                or ''
            ).strip()
        }
        for item in repair_items:
            identity = str(
                item.get('branch_id')
                or item.get('phase_id')
                or item.get('obligation_id')
                or item.get('task_id')
                or ''
            ).strip()
            contract = contracts_by_identity.get(identity)
            if contract:
                item['repair_contract'] = contract
                item['repair_contract_id'] = contract.get('contract_id')
                item['repair_execution_policy'] = contract.get('execution_policy')
                item['repair_promotion_source'] = contract.get('promotion_source')
                item['contract_state'] = 'promoted'
                for key in (
                    'materialization_blocked',
                    'blocked_scope',
                    'blocked_prerequisite',
                    'repair_work_available',
                    'repair_work_policy',
                    'needs_external_input',
                ):
                    value = contract.get(key)
                    if value not in (None, '', [], {}):
                        item[key] = value
        executable_contract_count = sum(1 for item in repair_contracts if item.get('auto_execute') is True)
        blocked_contract_count = sum(1 for item in repair_contracts if item.get('auto_execute') is not True)
        repair_work_available_count = sum(1 for item in repair_contracts if item.get('repair_work_available') is True)
        needs_external_input_count = sum(1 for item in repair_contracts if item.get('needs_external_input') is True)
        repair_loop_status = 'promoted' if repair_contracts else 'candidate'
        payload = {
            'kind': 'ollmo.ghost_repair_feedback',
            'status': 'repair_required',
            'target': 'request_ir_patch',
            'patch_scope': 'current_working_frame_request_phase_graph',
            'preserve_request_id': True,
            'repair_mode': 'bounded_graph_patch',
            'intent_reanalysis_policy': 'only_if_repair_items_are_ambiguous',
            'reason': reason,
            'graph_mode': str(request_phase_graph.get('mode') or '').strip() or None,
            'current_phase_id': str(request_phase_graph.get('current_phase_id') or '').strip() or None,
            'instruction': (
                'Patch the Request IR / phase graph from runtime evidence; do not answer prose. '
                'Promote missing work to obligations, preserve fulfilled evidence, waive only with an explicit runtime reason, '
                'then rerun executable work before freeze.'
            ),
            'items': repair_items,
            'repair_rebuild_contracts': repair_contracts,
            'repair_loop': {
                'kind': 'ollmo.repair_loop',
                'status': repair_loop_status,
                'authority': 'runtime_review_promoted',
                'auto_execute': bool(executable_contract_count),
                'round': 1,
                'max_rounds_policy': 'bounded_runtime_repair_policy',
                'next_actions': next_actions,
                'requires_promotion': not bool(repair_contracts),
                'promoted_contract_count': len(repair_contracts),
                'executable_contract_count': executable_contract_count,
                'blocked_contract_count': blocked_contract_count,
                'materialization_blocked_contract_count': blocked_contract_count,
                'repair_work_available': bool(repair_work_available_count),
                'repair_work_available_count': repair_work_available_count,
                'needs_external_input_count': needs_external_input_count,
                'promotion_review': {
                    'status': 'promoted' if repair_contracts else 'candidate',
                    'authority': 'closure_review_runtime_truth',
                    'policy': 'promote_only_current_open_closure_checks',
                },
                'promoted_contracts': repair_contracts,
            },
        }
        if decision_contract_guidance:
            payload['decision_contract_guidance'] = decision_contract_guidance
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, '', [], {})
        }

    def build_graph_closure_review(
        self,
        output_text: str,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
        artifact_gap: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        artifact_info = artifact_payload if isinstance(artifact_payload, dict) else {}
        route_runtime = (
            route_payload.get('route_runtime')
            if isinstance(route_payload, Mapping) and isinstance(route_payload.get('route_runtime'), Mapping)
            else {}
        )
        artifact_runtime = (
            artifact_info.get('runtime')
            if isinstance(artifact_info.get('runtime'), Mapping)
            else {}
        )
        request_phase_graph = (
            artifact_runtime.get('request_phase_graph')
            if isinstance(artifact_runtime.get('request_phase_graph'), Mapping)
            else (
                route_runtime.get('request_phase_graph')
                if isinstance(route_runtime.get('request_phase_graph'), Mapping)
                else {}
            )
        )
        payload: dict[str, Any] = {
            'kind': 'ollmo.graph_closure_review',
            'intent_boundary': 'anchored_intent_fluid_state',
            'status': 'not_applicable',
            'reason': 'no_request_phase_graph',
            'checks': [],
        }
        if not isinstance(request_phase_graph, Mapping) or not request_phase_graph:
            return payload

        decision_contract = self._decision_contract_from_graph(request_phase_graph)
        decision_contract_review = self._decision_contract_review_payload(decision_contract)
        workload_tasks_by_phase, workload_tasks_by_branch = self._workload_task_indexes(request_phase_graph)
        late_fill = artifact_info.get('late_fill') if isinstance(artifact_info.get('late_fill'), Mapping) else {}
        late_fill_status = str(late_fill.get('status') or '').strip().lower()
        pending_branches = self._extract_pending_deferred_branches(
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )
        pending_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in pending_branches
            if isinstance(item, Mapping) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        completed_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in (late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        completed_branches_by_id = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip(): item
            for item in (late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        final_materialization_contract_fulfilled = bool(
            late_fill_status == 'completed'
            and str(late_fill.get('final_materialization_contract_status') or '').strip().lower()
            == 'fulfilled'
            and late_fill.get('materialization_contract_unmet') is not True
        )
        failed_branch_records = [
            item
            for item in (late_fill.get('failed_branches') or [])
            if isinstance(item, Mapping)
        ]
        cancelled_branch_records = [
            item
            for item in (late_fill.get('cancelled_branches') or [])
            if isinstance(item, Mapping)
        ]
        failed_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in failed_branch_records
            if str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        cancelled_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in cancelled_branch_records
            if str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        failed_branches_by_id = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip(): item
            for item in failed_branch_records
            if str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        completed_capabilities = set(self._hook('normalize_capability_list')(late_fill.get('completed_capabilities')))
        failed_capabilities = set(self._hook('normalize_capability_list')(late_fill.get('failed_capabilities')))
        cancelled_capabilities = set(self._hook('normalize_capability_list')(late_fill.get('cancelled_capabilities')))
        current_phase_id = str(request_phase_graph.get('current_phase_id') or '').strip()
        phases = request_phase_graph.get('phases') if isinstance(request_phase_graph.get('phases'), list) else []
        output_obligations = output_obligations_from_graph(request_phase_graph)
        check_sources: list[dict[str, Any]]
        if output_obligations:
            check_sources = output_obligations
        else:
            check_sources = [dict(item) for item in phases if isinstance(item, Mapping)]
        capability_source_counts: dict[str, int] = {}
        for raw_source in check_sources:
            if not isinstance(raw_source, Mapping):
                continue
            source_capability = normalize_capability(raw_source.get('capability'))
            if source_capability:
                capability_source_counts[source_capability] = (
                    capability_source_counts.get(source_capability, 0) + 1
                )
        checks: list[dict[str, Any]] = []
        artifact_type_counts: dict[str, int] = {}
        consumed_artifact_type_counts: dict[str, int] = {}

        def tts_integrity_evidence_for(
            branch_id: str,
            phase_id: str,
        ) -> dict[str, Any]:
            candidates: list[Mapping[str, Any]] = []
            for result in late_fill.get('fill_results') or []:
                if not isinstance(result, Mapping):
                    continue
                result_tokens = {
                    str(result.get('branch_id') or '').strip(),
                    str(result.get('phase_id') or '').strip(),
                }
                result_tokens.discard('')
                if (
                    (branch_id or phase_id)
                    and not result_tokens.intersection(
                        {token for token in (branch_id, phase_id) if token}
                    )
                ):
                    continue
                evidence = result.get('tts_audio_integrity_evidence')
                if isinstance(evidence, Mapping):
                    candidates.append(evidence)
            top_level = artifact_info.get('tts_audio_integrity_evidence')
            if isinstance(top_level, Mapping):
                candidates.append(top_level)
            return dict(candidates[0]) if candidates else {}

        for raw_source in check_sources:
            if not isinstance(raw_source, Mapping):
                continue
            obligation_id = str(raw_source.get('obligation_id') or '').strip()
            phase_id = str(raw_source.get('phase_id') or '').strip()
            branch_id = str(raw_source.get('branch_id') or phase_id or '').strip()
            workload_task_record = (
                workload_tasks_by_phase.get(phase_id)
                or workload_tasks_by_branch.get(branch_id)
                or {}
            )
            capability = normalize_capability(raw_source.get('capability'))
            is_unique_capability_source = (
                bool(capability)
                and capability_source_counts.get(capability, 0) <= 1
            )
            output_type = str(raw_source.get('output_type') or '').strip().lower()
            if not output_type:
                output_type = str(self.artifact_type_for_capability(capability) or '').strip().lower()
            source_status = str(raw_source.get('status') or '').strip().lower()
            evidence = 'output_obligation_status' if output_obligations else 'phase_status'
            source_role = str(raw_source.get('role') or '').strip()
            requires_text_artifact = self.source_requires_text_artifact(raw_source, output_type)
            source_depends_on = [
                str(item or '').strip()
                for item in (raw_source.get('depends_on') or [])
                if str(item or '').strip()
            ]
            failed_branch_record = failed_branches_by_id.get(branch_id) if branch_id else None
            completed_branch_record = completed_branches_by_id.get(branch_id) if branch_id else None
            completed_branch_status = str(
                completed_branch_record.get('status')
                if isinstance(completed_branch_record, Mapping)
                else ''
            ).strip().lower()
            completed_branch_evidence = str(
                completed_branch_record.get('evidence')
                if isinstance(completed_branch_record, Mapping)
                else ''
            ).strip()
            tts_audio_integrity_evidence = (
                tts_integrity_evidence_for(branch_id, phase_id)
                if capability == CAPABILITY_TEXT_TO_SPEECH
                and output_type == 'audio'
                else {}
            )
            tts_audio_artifact_present = bool(
                capability == CAPABILITY_TEXT_TO_SPEECH
                and output_type == 'audio'
                and (
                    str(artifact_info.get('saved_audio_path') or '').strip()
                    or self.response_payload_artifact_type_count(
                        artifact_info,
                        'audio',
                    )
                    or branch_id in completed_branch_ids
                    or source_status in {'completed', 'fulfilled'}
                )
            )
            tts_audio_integrity_passed = bool(
                str(
                    tts_audio_integrity_evidence.get('status') or ''
                ).strip().lower()
                == 'passed'
                and tts_audio_integrity_evidence.get(
                    'materialization_eligible'
                )
                is True
            )
            completed_in_place_materialization_repair = bool(
                isinstance(completed_branch_record, Mapping)
                and completed_branch_status in {'completed', 'fulfilled'}
                and (
                    completed_branch_record.get('non_blocking_after_final_contract_fulfilled') is True
                    or completed_branch_evidence
                    in {
                        'canonical_text_artifact_evidence',
                        'terminal_inline_text_artifact_materialized',
                        'terminal_linked_artifact_rebind_applied',
                    }
                    or (
                        final_materialization_contract_fulfilled
                        and source_role
                        in {
                            'linked_artifact_binding_review',
                            'text_artifact_syntax_repair',
                        }
                    )
                )
            )
            status = source_status or 'pending'

            if (
                tts_audio_artifact_present
                and not tts_audio_integrity_passed
                and source_status not in _SUPERSEDED_OBLIGATION_STATUSES
                and source_status not in _WAIVED_OBLIGATION_STATUSES
            ):
                status = 'blocked'
                evidence = (
                    'tts_audio_integrity_failed'
                    if tts_audio_integrity_evidence
                    else 'tts_audio_integrity_evidence_missing'
                )
            elif source_status in _SUPERSEDED_OBLIGATION_STATUSES:
                status = 'superseded'
                evidence = 'explicit_obligation_superseded'
            elif source_status in _WAIVED_OBLIGATION_STATUSES:
                status = 'waived'
                evidence = 'explicit_obligation_waiver'
            elif branch_id and branch_id in failed_branch_ids:
                status = 'blocked'
                evidence = 'late_fill_failed_branch'
            elif branch_id and branch_id in cancelled_branch_ids:
                status = 'waived'
                evidence = 'late_fill_cancelled_branch'
            elif completed_in_place_materialization_repair:
                # A verified in-place rebind/repair updates an existing saved
                # artifact; it must not consume a fictitious extra artifact.
                status = 'fulfilled'
                evidence = 'late_fill_completed_in_place_materialization_repair'
            elif branch_id and branch_id in completed_branch_ids and not requires_text_artifact:
                status = 'fulfilled'
                evidence = 'late_fill_completed_branch'
            elif is_unique_capability_source and capability in failed_capabilities:
                status = 'blocked'
                evidence = 'late_fill_failed_capability'
            elif is_unique_capability_source and capability in cancelled_capabilities:
                status = 'waived'
                evidence = 'late_fill_cancelled_capability'
            elif (
                is_unique_capability_source
                and capability in completed_capabilities
                and not requires_text_artifact
            ):
                status = 'fulfilled'
                evidence = 'late_fill_completed_capability'
            elif (
                phase_id == current_phase_id
                and capability == CAPABILITY_CHAT
                and str(output_text or '').strip()
                and not requires_text_artifact
            ):
                status = 'fulfilled'
                evidence = 'current_phase_output_text'
            elif output_type:
                if output_type not in artifact_type_counts:
                    artifact_type_counts[output_type] = (
                        self.response_payload_real_text_artifact_count(artifact_info)
                        if requires_text_artifact and output_type == 'text'
                        else self.response_payload_artifact_type_count(
                            artifact_info,
                            output_type,
                        )
                    )
                consumed_count = consumed_artifact_type_counts.get(output_type, 0)
                if artifact_type_counts.get(output_type, 0) > consumed_count:
                    consumed_artifact_type_counts[output_type] = consumed_count + 1
                    status = 'fulfilled'
                    evidence = f'output_artifact_type:{output_type}'
                elif branch_id and branch_id in pending_branch_ids:
                    status = 'pending'
                    evidence = 'pending_graph_branch'
                elif status == 'completed':
                    status = 'fulfilled'
                elif status in {'failed', 'error'}:
                    status = 'blocked'
                elif status in _SUPERSEDED_OBLIGATION_STATUSES:
                    status = 'superseded'
                    evidence = 'explicit_obligation_superseded'
                elif status in _WAIVED_OBLIGATION_STATUSES:
                    status = 'waived'
                    evidence = 'explicit_obligation_waiver'
                elif status not in {'fulfilled', 'blocked', 'pending', 'deferred', 'planned', 'active', 'waived', 'superseded'}:
                    status = 'pending'
            elif branch_id and branch_id in pending_branch_ids:
                status = 'pending'
                evidence = 'pending_graph_branch'
            elif status == 'completed':
                status = 'fulfilled'
            elif status in {'failed', 'error'}:
                status = 'blocked'
            elif status in _SUPERSEDED_OBLIGATION_STATUSES:
                status = 'superseded'
                evidence = 'explicit_obligation_superseded'
            elif status in _WAIVED_OBLIGATION_STATUSES:
                status = 'waived'
                evidence = 'explicit_obligation_waiver'
            elif status not in {'fulfilled', 'blocked', 'pending', 'deferred', 'planned', 'active', 'waived', 'superseded'}:
                status = 'pending'

            if (
                status == 'fulfilled'
                and evidence.startswith('late_fill_completed')
                and output_type
                and not (output_type == 'text' and not requires_text_artifact)
            ):
                if output_type not in artifact_type_counts:
                    artifact_type_counts[output_type] = self.response_payload_artifact_type_count(
                        artifact_info,
                        output_type,
                    )
                consumed_count = consumed_artifact_type_counts.get(output_type, 0)
                if artifact_type_counts.get(output_type, 0) > consumed_count:
                    consumed_artifact_type_counts[output_type] = consumed_count + 1

            check = {
                'obligation_id': obligation_id or None,
                'phase_id': phase_id or None,
                'branch_id': branch_id or None,
                'capability': capability or None,
                'output_type': output_type or None,
                'role': source_role or None,
                'depends_on': source_depends_on or None,
                'status': status,
                'evidence': evidence,
            }
            if (
                capability == CAPABILITY_TEXT_TO_SPEECH
                and output_type == 'audio'
                and tts_audio_artifact_present
            ):
                check['audio_integrity_status'] = (
                    str(tts_audio_integrity_evidence.get('status') or '').strip()
                    or 'unavailable'
                )
                check['audio_integrity_reason_code'] = (
                    str(
                        tts_audio_integrity_evidence.get('reason_code') or ''
                    ).strip()
                    or 'TTS_AUDIO_INTEGRITY_EVIDENCE_MISSING'
                )
                if tts_audio_integrity_evidence:
                    check['tts_audio_integrity_evidence'] = (
                        tts_audio_integrity_evidence
                    )
                if not tts_audio_integrity_passed:
                    check['reason'] = (
                        'generated audio exists but did not pass output-side '
                        'physical integrity verification'
                    )
                    check['repair_action'] = (
                        RECOVERY_ACTION_RETRY_SAME_BRANCH
                    )
                    check['recovery_action'] = (
                        RECOVERY_ACTION_RETRY_SAME_BRANCH
                    )
                    check['repair_action_reason'] = (
                        'retry the bounded TTS branch and require passed '
                        'audio-integrity evidence before fulfillment'
                    )
                    check['materialization_blocked'] = True
            for source in (
                raw_source,
                workload_task_record if isinstance(workload_task_record, Mapping) else {},
                failed_branch_record if isinstance(failed_branch_record, Mapping) else {},
            ):
                if not isinstance(source, Mapping):
                    continue
                for key in _CLOSURE_REPAIR_PAYLOAD_KEYS:
                    value = source.get(key)
                    if value not in (None, '', [], {}) and check.get(key) in (None, '', [], {}):
                        check[key] = value
                if check.get('depends_on') in (None, '', [], {}) and isinstance(source.get('depends_on'), list):
                    depends_on = [
                        str(item or '').strip()
                        for item in source.get('depends_on')
                        if str(item or '').strip()
                    ]
                    if depends_on:
                        check['depends_on'] = depends_on
                for key in ('recovery_context', 'recovery_state', 'attempt'):
                    value = source.get(key)
                    if value not in (None, '', [], {}) and check.get(key) in (None, '', [], {}):
                        check[key] = value
            check = self._apply_decision_contract_to_check(check, decision_contract)
            recovery_context = (
                check.get('recovery_context')
                if isinstance(check.get('recovery_context'), Mapping)
                else {}
            )
            recovery_state = (
                check.get('recovery_state')
                if isinstance(check.get('recovery_state'), Mapping)
                else {}
            )
            attempt_payload = check.get('attempt') if isinstance(check.get('attempt'), Mapping) else {}
            for key in ('blocked_by_dependency_input', 'blocked_by_branch_contract', 'repair_required'):
                value = recovery_context.get(key)
                if not isinstance(value, bool):
                    value = recovery_state.get(key)
                if isinstance(value, bool):
                    check[key] = value
            repair_action, repair_action_reason = self._closure_check_recovery_action(check)
            if repair_action:
                check['recovery_action'] = repair_action
                check['repair_action'] = repair_action
            if repair_action_reason:
                check['repair_action_reason'] = repair_action_reason
            failed_instance_id = str(
                attempt_payload.get('instance_id')
                or recovery_state.get('failed_instance_id')
                or ''
            ).strip()
            if failed_instance_id:
                check['failed_instance_id'] = failed_instance_id
            exclude_instance_ids = [
                str(item or '').strip()
                for item in (
                    recovery_context.get('exclude_instance_ids')
                    if isinstance(recovery_context.get('exclude_instance_ids'), list)
                    else recovery_state.get('exclude_instance_ids')
                    if isinstance(recovery_state.get('exclude_instance_ids'), list)
                    else []
                )
                if str(item or '').strip()
            ]
            if exclude_instance_ids:
                check['exclude_instance_ids'] = exclude_instance_ids
            check = self._apply_review_criteria_to_check(check)
            check = self._apply_branch_semantic_verdict_to_check(
                check,
                artifact_payload=artifact_payload,
            )
            repair_action, repair_action_reason = self._closure_check_recovery_action(check)
            if repair_action and check.get('repair_action') in (None, '', [], {}):
                check['recovery_action'] = repair_action
                check['repair_action'] = repair_action
            if repair_action_reason and check.get('repair_action_reason') in (None, '', [], {}):
                check['repair_action_reason'] = repair_action_reason
            checks.append({key: value for key, value in check.items() if value not in (None, '', [], {})})

        intent_graph_adequacy = self.build_intent_graph_adequacy_review(
            request_payload=request_payload,
            request_phase_graph=request_phase_graph,
        )
        adequacy_checks = (
            intent_graph_adequacy.get('checks')
            if isinstance(intent_graph_adequacy.get('checks'), list)
            else []
        )
        checks.extend(
            dict(item)
            for item in adequacy_checks
            if isinstance(item, Mapping)
        )
        truth_guard = artifact_runtime.get('truth_guard') if isinstance(artifact_runtime, Mapping) else {}
        if isinstance(truth_guard, Mapping):
            truth_guard_status = str(truth_guard.get('status') or '').strip().lower()
            if truth_guard_status in {'annotated', 'rewritten'}:
                checks.append(
                    {
                        'check_kind': 'truth_guard',
                        'status': 'pending',
                        'evidence': 'text_only_artifact_claim_guard',
                        'reason': str(truth_guard.get('reason') or '').strip()
                        or 'artifact/materialization claim was not backed by runtime truth',
                    }
                )
                claimed_capabilities = self._hook('normalize_capability_list')(
                    truth_guard.get('claimed_capabilities')
                )
                for claimed_capability in claimed_capabilities:
                    availability = self.runtime_capability_availability(claimed_capability)
                    availability_status = str(availability.get('status') or '').strip().lower()
                    if availability_status == 'available':
                        status = 'pending'
                        evidence = 'runtime_capability_available_but_unmaterialized'
                    elif availability_status == 'unavailable':
                        status = 'blocked'
                        evidence = 'runtime_capability_unavailable'
                    else:
                        status = 'pending'
                        evidence = 'runtime_capability_unverified'
                    capability_check = {
                        'check_kind': 'truth_guard_capability',
                        'capability': claimed_capability,
                        'output_type': self.artifact_type_for_capability(claimed_capability),
                        'status': status,
                        'evidence': evidence,
                        'runtime_availability_status': availability_status or None,
                        'available_instance_ids': availability.get('available_instance_ids'),
                        'reason': (
                            'claimed material output did not have matching runtime artifact evidence'
                            if status == 'pending'
                            else availability.get('reason')
                        ),
                    }
                    checks.append(
                        {
                            key: value
                            for key, value in capability_check.items()
                            if value not in (None, '', [], {})
                        }
                    )

        checks.extend(
            self._branch_semantic_review_checks(
                checks=checks,
                request_payload=request_payload,
                request_phase_graph=request_phase_graph,
                artifact_payload=artifact_payload,
                output_text=output_text,
            )
        )
        checks.extend(
            self._linked_artifact_binding_checks(
                request_payload=request_payload,
                artifact_payload=artifact_payload,
                request_phase_graph=request_phase_graph,
            )
        )
        checks.extend(
            self._html_css_selector_binding_checks(
                artifact_payload=artifact_payload,
            )
        )
        checks.extend(
            self._text_artifact_syntax_sanity_checks(
                artifact_payload=artifact_payload,
            )
        )
        checks.extend(
            self._structured_dependency_join_checks(
                request_payload=request_payload,
                artifact_payload=artifact_payload,
                request_phase_graph=request_phase_graph,
            )
        )

        global_semantic_closure_review = self.build_global_semantic_closure_review(
            output_text=output_text,
            request_payload=request_payload,
            request_phase_graph=request_phase_graph,
            artifact_payload=artifact_payload,
            checks=checks,
            intent_graph_adequacy=intent_graph_adequacy,
            decision_contract=decision_contract,
        )
        checks.extend(self._global_semantic_closure_checks(global_semantic_closure_review))

        normalized_checks: list[dict[str, Any]] = []
        for raw_check in checks:
            check = dict(raw_check)
            repair_action, repair_action_reason = self._closure_check_recovery_action(check)
            if repair_action and check.get('repair_action') in (None, '', [], {}):
                check['repair_action'] = repair_action
                check['recovery_action'] = repair_action
            if repair_action_reason and check.get('repair_action_reason') in (None, '', [], {}):
                check['repair_action_reason'] = repair_action_reason
            normalized_checks.append(
                {key: value for key, value in check.items() if value not in (None, '', [], {})}
            )
        checks = normalized_checks
        semantic_review_required_count = sum(
            1 for item in checks if item.get('semantic_review_required') is True
        )
        countable_checks = [
            item for item in checks
            if str(item.get('check_kind') or '').strip() not in {'intent_attention_review'}
        ]

        counts = {
            'fulfilled': sum(1 for item in countable_checks if item.get('status') == 'fulfilled'),
            'pending': sum(1 for item in countable_checks if item.get('status') in {'pending', 'planned', 'active'}),
            'deferred': sum(1 for item in countable_checks if item.get('status') == 'deferred'),
            'blocked': sum(1 for item in countable_checks if item.get('status') == 'blocked'),
            'superseded': sum(1 for item in countable_checks if item.get('status') == 'superseded'),
            'waived': sum(1 for item in countable_checks if item.get('status') == 'waived'),
        }
        open_check_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in checks
            if item.get('status') in {'pending', 'planned', 'active', 'deferred'}
            and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        if checks:
            effective_pending_branches = [
                item
                for item in pending_branches
                if not isinstance(item, Mapping)
                or str(item.get('branch_id') or item.get('phase_id') or '').strip() in open_check_branch_ids
            ]
        else:
            effective_pending_branches = pending_branches
        active_late_fill = late_fill_status in _ACTIVE_LATE_FILL_STATUSES and (
            not checks or counts['pending'] or counts['deferred'] or bool(effective_pending_branches)
        )

        if counts['blocked']:
            review_status = 'blocked'
            reason = 'one or more graph obligations are blocked by runtime truth'
        elif intent_graph_adequacy.get('status') == 'pending':
            review_status = 'pending'
            reason = 'graph/IR does not fully cover current user intent'
        elif counts['pending'] or counts['deferred'] or effective_pending_branches or active_late_fill:
            review_status = 'pending'
            reason = 'one or more graph obligations remain open inside the same intent'
        elif checks:
            review_status = 'fulfilled'
            reason = 'all checked graph obligations are fulfilled, waived, or superseded by runtime truth'
        else:
            review_status = 'not_applicable'
            reason = 'request phase graph has no output obligations or phase records'

        surface_state = self._closure_surface_state(
            review_status=review_status,
            reason=reason,
            counts=counts,
            checks=checks,
            decision_contract=decision_contract,
            late_fill_status=late_fill_status,
            pending_branches=effective_pending_branches,
            semantic_review_required_count=semantic_review_required_count,
        )
        continuation_required = review_status in {'pending', 'blocked'}
        payload.update(
            {
                'status': review_status,
                'reason': reason,
                'graph_mode': str(request_phase_graph.get('mode') or '').strip() or None,
                'current_phase_id': current_phase_id or None,
                'continuation_required': continuation_required,
                'late_fill_status': late_fill_status or None,
                'pending_branch_count': len(effective_pending_branches),
                'counts': counts,
                'checks': checks,
                'semantic_review_required_count': semantic_review_required_count,
                'semantic_review_status': 'required_advisory' if semantic_review_required_count else None,
                'surface_state': surface_state,
            }
        )
        if decision_contract_review:
            payload['decision_contract_review'] = decision_contract_review
        if global_semantic_closure_review:
            payload['global_semantic_closure_review'] = global_semantic_closure_review
        ghost_repair_feedback = self.build_ghost_repair_feedback(
            review_status=review_status,
            reason=reason,
            checks=checks,
            request_phase_graph=request_phase_graph,
        )
        if ghost_repair_feedback:
            payload['ghost_repair_feedback'] = ghost_repair_feedback
        if intent_graph_adequacy.get('status') != 'not_applicable':
            payload['intent_graph_adequacy'] = intent_graph_adequacy
        if output_obligations:
            payload['contract_source'] = 'request_ir.output_obligations'
            payload['obligation_count'] = len(output_obligations)
        if continuation_required and isinstance(artifact_gap, Mapping) and artifact_gap:
            payload['closure_gap_code'] = str(artifact_gap.get('code') or '').strip() or None
            payload['closure_gap_trigger'] = str(artifact_gap.get('trigger') or '').strip() or None
        if effective_pending_branches:
            payload['pending_branches'] = [
                {
                    key: item.get(key)
                    for key in ('branch_id', 'phase_id', 'capability', 'output_type')
                    if item.get(key) not in (None, '', [], {})
                }
                for item in effective_pending_branches
                if isinstance(item, Mapping)
            ]
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, '', [], {})
        }

    def build_pre_freeze_closure_review_gap(
        self,
        output_text: str,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
        artifact_gap: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        normalize_capability_list = self._hook('normalize_capability_list')

        artifact_info = artifact_payload if isinstance(artifact_payload, dict) else {}
        late_fill = artifact_info.get('late_fill') if isinstance(artifact_info.get('late_fill'), Mapping) else {}
        late_fill_status = str(late_fill.get('status') or '').strip().lower()
        if late_fill_status in _ACTIVE_LATE_FILL_STATUSES:
            return None
        route_runtime = (
            route_payload.get('route_runtime')
            if isinstance(route_payload, Mapping) and isinstance(route_payload.get('route_runtime'), Mapping)
            else {}
        )
        execution_planner = (
            route_runtime.get('execution_planner')
            if isinstance(route_runtime.get('execution_planner'), Mapping)
            else {}
        )
        planner_deferred_explicit = bool(
            execution_planner.get('deferred_capability')
            or execution_planner.get('deferred_capabilities')
            or execution_planner.get('deferred_branches')
        )

        pending_branches = self._extract_pending_deferred_branches(
            route_payload=route_payload,
            artifact_payload=artifact_payload,
        )
        pending_capabilities = normalize_capability_list(
            [item.get('capability') for item in pending_branches if isinstance(item, Mapping)]
        )
        completed_capabilities = normalize_capability_list(late_fill.get('completed_capabilities'))

        base_gap = dict(artifact_gap) if isinstance(artifact_gap, dict) else self.build_artifact_completion_gap_spec(
            output_text,
            route_payload=route_payload,
            request_payload=request_payload,
            artifact_payload=artifact_payload,
        )
        if isinstance(base_gap, dict):
            base_gap = dict(base_gap)

        if pending_branches:
            expected_branch = dict(pending_branches[0])
            expected_capability = normalize_capability(
                (base_gap or {}).get('expected_capability') or expected_branch.get('capability')
            )
            if not expected_capability:
                return None
            payload = dict(base_gap or {})
            preserve_base_gap = str(payload.get('code') or '').strip() == 'late_artifact_fill'
            planner_gap = self.build_planner_deferred_follow_up_gap_spec(
                output_text,
                route_payload=route_payload,
                artifact_payload=artifact_payload,
            )
            if isinstance(planner_gap, dict) and planner_gap:
                for key, value in planner_gap.items():
                    if preserve_base_gap and key in {'code', 'trigger'}:
                        continue
                    if preserve_base_gap and payload.get(key) not in (None, '', [], {}):
                        continue
                    payload[key] = value
            payload['expected_capability'] = expected_capability
            payload['active_capability'] = expected_capability
            payload['missing_artifact_type'] = (
                str(payload.get('missing_artifact_type') or '').strip()
                or str(self.artifact_type_for_capability(expected_capability) or '').strip()
                or None
            )
            if (
                len(pending_branches) == 1
                and payload['missing_artifact_type']
                and self.response_payload_has_artifact_type(
                    artifact_payload,
                    payload['missing_artifact_type'],
                )
            ):
                return None
            payload['expected_branch_id'] = (
                str(payload.get('expected_branch_id') or '').strip()
                or str(expected_branch.get('branch_id') or expected_branch.get('phase_id') or '').strip()
                or None
            )
            payload['expected_phase_id'] = (
                str(payload.get('expected_phase_id') or '').strip()
                or str(expected_branch.get('phase_id') or expected_branch.get('branch_id') or '').strip()
                or None
            )
            effective_pending_branches = (
                payload.get('pending_branches')
                if isinstance(payload.get('pending_branches'), list)
                else pending_branches
            )
            payload['pending_branches'] = effective_pending_branches
            payload['pending_capabilities'] = normalize_capability_list(
                [item.get('capability') for item in effective_pending_branches if isinstance(item, Mapping)]
            ) or pending_capabilities
            if completed_capabilities:
                payload['completed_capabilities'] = completed_capabilities
            if not planner_deferred_explicit:
                payload['code'] = 'closure_review_fill'
                payload['trigger'] = 'pre_freeze_closure_review'
            elif not str(payload.get('code') or '').strip():
                payload['code'] = 'closure_review_fill'
            if not str(payload.get('trigger') or '').strip():
                payload['trigger'] = 'pre_freeze_closure_review'
            semantic_payload = self.semantic_payload_for_capability(
                artifact_payload,
                capability=expected_capability,
            )
            for key, value in semantic_payload.items():
                if payload.get(key) in (None, '', [], {}):
                    payload[key] = value
            if (
                expected_capability == CAPABILITY_IMAGE_GENERATION
                and not isinstance(payload.get('batch_prompts'), list)
            ):
                route_runtime_for_count = (
                    ((route_payload or {}).get('route_runtime') or {})
                    if isinstance(((route_payload or {}).get('route_runtime') or {}), Mapping)
                    else {}
                )
                artifact_runtime_for_count = (
                    ((artifact_payload or {}).get('runtime') or {})
                    if isinstance(((artifact_payload or {}).get('runtime') or {}), Mapping)
                    else {}
                )
                request_phase_graph = (
                    artifact_runtime_for_count.get('request_phase_graph')
                    if isinstance(artifact_runtime_for_count.get('request_phase_graph'), Mapping)
                    else (
                        route_runtime_for_count.get('request_phase_graph')
                        if isinstance(route_runtime_for_count.get('request_phase_graph'), Mapping)
                        else {}
                    )
                )
                expected_count = 0
                if isinstance(request_phase_graph, Mapping) and isinstance(request_phase_graph.get('prompt_intent'), Mapping):
                    expected_count = int(request_phase_graph.get('prompt_intent', {}).get('requested_visual_output_count') or 0)
                if expected_count <= 0:
                    expected_count = len(pending_branches)
                batch_prompts = self.extract_batch_image_prompts(
                    output_text,
                    expected_count=expected_count,
                    allow_plain_alpha_sequence=False,
                )
                if len(batch_prompts) > 1:
                    payload['batch_prompts'] = batch_prompts
                    payload['batch_prompt_expected_count'] = (
                        expected_count if expected_count >= 2 else len(batch_prompts)
                    )
                    if str(payload.get('artifact_prompt') or '').strip() == str(batch_prompts[0]).strip():
                        payload['artifact_prompt_source'] = (
                            str(payload.get('artifact_prompt_source') or '').strip()
                            or 'semantic_batch_prompts'
                        )
            return payload

        if isinstance(base_gap, dict):
            if self.artifact_gap_is_already_fulfilled(base_gap, artifact_payload):
                return None
            payload = dict(base_gap)
            if not str(payload.get('trigger') or '').strip():
                payload['trigger'] = 'pre_freeze_closure_review'
            return payload
        return None

    def build_late_fill_state(
        self,
        artifact_gap: dict[str, Any],
        *,
        status: str,
        prior_state: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalize_capability_list = self._hook('normalize_capability_list')
        normalize_late_fill_branches = self._hook('normalize_late_fill_branches')
        response_registry_now_iso = self._hook('response_registry_now_iso')

        previous = prior_state if isinstance(prior_state, dict) else {}
        now_iso = response_registry_now_iso()
        status_token = str(status or '').strip().lower()
        terminal_without_pending_surface = status_token in {
            'completed',
            'skipped',
            'cancelled',
        }
        terminal_late_fill_status = status_token in {
            'completed',
            'skipped',
            'cancelled',
            'failed',
            'partial_failed',
            'partial_cancelled',
        }
        payload = {
            'status': str(status or '').strip() or 'pending',
            'code': str(artifact_gap.get('code') or 'late_artifact_fill'),
            'trigger': str(artifact_gap.get('trigger') or 'chat_text_only_completion'),
            'expected_capability': normalize_capability(artifact_gap.get('expected_capability')),
            'active_capability': normalize_capability(
                artifact_gap.get('active_capability') or artifact_gap.get('expected_capability')
            ),
            'missing_artifact_type': str(artifact_gap.get('missing_artifact_type') or '').strip() or None,
            'late_fill_supported': True,
            'created_at': str(previous.get('created_at') or now_iso),
            'updated_at': now_iso,
        }
        if payload['status'] == 'running':
            payload['started_at'] = str(previous.get('started_at') or now_iso)
        if payload['status'] in {'completed', 'partial_failed', 'partial_cancelled'}:
            payload['started_at'] = str(previous.get('started_at') or now_iso)
            payload['completed_at'] = now_iso
        if payload['status'] == 'cancelled':
            payload['started_at'] = str(previous.get('started_at') or now_iso)
            payload['cancelled_at'] = now_iso
        if payload['status'] in {'failed', 'partial_failed'}:
            payload['started_at'] = str(previous.get('started_at') or now_iso)
            payload['failed_at'] = now_iso
        for key in ('content_payload', 'stage_direction', 'phase_summary', 'content_payload_source'):
            value = artifact_gap.get(key)
            if value in (None, '', [], {}):
                value = previous.get(key)
            if value not in (None, '', [], {}):
                payload[key] = value
        for key in ('artifact_prompt', 'artifact_prompt_source'):
            value = artifact_gap.get(key)
            if value in (None, '', [], {}):
                value = previous.get(key)
            if value not in (None, '', [], {}):
                payload[key] = value
        for key in ('batch_prompts_source', 'batch_prompt_source_phase_id'):
            value = artifact_gap.get(key)
            if value in (None, '', [], {}):
                value = previous.get(key)
            if value not in (None, '', [], {}):
                payload[key] = str(value).strip()
        for key in ('expected_branch_id', 'expected_phase_id'):
            value = artifact_gap.get(key)
            if value in (None, '', [], {}):
                value = previous.get(key)
            if value not in (None, '', [], {}):
                payload[key] = str(value).strip() or None
        for key in (
            'ghost_repair_feedback',
            'repair_scope',
            'repair_mode',
            'repair_action',
            'repair_actions',
            'repair_action_reason',
            'repair_loop',
            'repair_rebuild_contracts',
            'reconsideration_rebuild',
            'block_resolution_reflex',
            'reconsideration_reflex',
            'active_reconsideration_review',
            'active_reconsideration_decisions',
            'semantic_quality_review',
            'semantic_quality_contracts',
            'aspiration_review',
            'aspiration_frames',
            'commitment_review',
            'commitment_frames',
            'semantic_decision_review',
            'semantic_decision_proposals',
            'semantic_review_verdict',
            'semantic_review_verdict_status',
            'semantic_review_recommended_transition',
            'branch_semantic_review',
            'branch_semantic_review_branch_id',
            'branch_semantic_review_phase_id',
            'branch_semantic_review_status',
            'branch_semantic_review_reason',
            'branch_semantic_review_source_branch_id',
            'branch_semantic_review_source_phase_id',
            'controlled_attention_review',
            'controlled_attention_frames',
            'recursive_cycle_review',
            'surface_state',
            'branch_controls',
            'execution_gate_status',
            'repair_code',
            'repair_trigger',
            'preserve_request_id',
            'pending_branch_scope',
            'authoritative_pending_branches',
            'successor_reopen_execution',
        ):
            value = artifact_gap.get(key)
            if value in (None, '', [], {}):
                value = previous.get(key)
            if key == 'surface_state' and terminal_without_pending_surface:
                value = None
            if value not in (None, '', [], {}):
                payload[key] = value
        batch_prompts = artifact_gap.get('batch_prompts')
        if not isinstance(batch_prompts, list):
            batch_prompts = previous.get('batch_prompts') if isinstance(previous.get('batch_prompts'), list) else None
        if isinstance(batch_prompts, list):
            normalized_batch_prompts = [
                str(item).strip()
                for item in batch_prompts
            ]
            if any(normalized_batch_prompts):
                payload['batch_prompts'] = normalized_batch_prompts
        raw_batch_prompt_expected_count = artifact_gap.get('batch_prompt_expected_count')
        if raw_batch_prompt_expected_count in (None, '', [], {}):
            raw_batch_prompt_expected_count = previous.get('batch_prompt_expected_count')
        try:
            batch_prompt_expected_count = int(raw_batch_prompt_expected_count or 0)
        except (TypeError, ValueError):
            batch_prompt_expected_count = 0
        if 2 <= batch_prompt_expected_count <= 26:
            payload['batch_prompt_expected_count'] = batch_prompt_expected_count
        for key in ('pending_branches', 'completed_branches', 'failed_branches', 'cancelled_branches', 'active_branches'):
            value = artifact_gap.get(key)
            if not isinstance(value, list):
                value = previous.get(key) if isinstance(previous.get(key), list) else None
            if isinstance(value, list):
                normalized_branches = normalize_late_fill_branches(value)
                if normalized_branches or key in previous or key in artifact_gap:
                    payload[key] = normalized_branches
        for key in ('pending_capabilities', 'completed_capabilities', 'failed_capabilities', 'cancelled_capabilities'):
            value = artifact_gap.get(key)
            if not isinstance(value, list):
                value = previous.get(key) if isinstance(previous.get(key), list) else None
            if isinstance(value, list):
                payload[key] = normalize_capability_list(value)
        if extra:
            payload.update(
                {
                    key: value
                    for key, value in extra.items()
                    if value not in (None, '', [], {})
                    and not (key == 'surface_state' and terminal_without_pending_surface)
                }
            )
            if terminal_without_pending_surface:
                payload.pop('surface_state', None)
            for key in ('pending_capabilities', 'completed_capabilities', 'failed_capabilities', 'cancelled_capabilities'):
                if key in extra and isinstance(extra.get(key), list):
                    payload[key] = normalize_capability_list(extra.get(key))
            for key in ('pending_branches', 'completed_branches', 'failed_branches', 'cancelled_branches', 'active_branches'):
                if key in extra and isinstance(extra.get(key), list):
                    payload[key] = normalize_late_fill_branches(extra.get(key))
            for key in ('expected_branch_id', 'expected_phase_id'):
                if key in extra and extra.get(key) not in (None, '', [], {}):
                    payload[key] = str(extra.get(key)).strip() or None
            if 'active_capability' in extra:
                payload['active_capability'] = normalize_capability(extra.get('active_capability'))
        for key, terminal_status in (
            ('completed_branches', 'fulfilled'),
            ('failed_branches', 'failed'),
            ('cancelled_branches', None),
        ):
            branches = payload.get(key)
            if not isinstance(branches, list):
                continue
            normalized: list[dict[str, Any]] = []
            for branch in branches:
                if not isinstance(branch, Mapping):
                    continue
                branch_payload = dict(branch)
                if terminal_status:
                    branch_payload['status'] = terminal_status
                elif not str(branch_payload.get('status') or '').strip():
                    branch_payload['status'] = 'cancelled'
                normalized.append(branch_payload)
            payload[key] = normalized
        if terminal_late_fill_status:
            payload['pending_branches'] = []
            payload['active_branches'] = []
            payload['pending_capabilities'] = []
            payload['active_capabilities'] = []
        successor_execution = (
            dict(payload.get('successor_reopen_execution'))
            if isinstance(payload.get('successor_reopen_execution'), Mapping)
            else {}
        )
        if successor_execution:
            execution_status = {
                'pending': 'queued',
                'queued': 'queued',
                'scheduled': 'queued',
                'running': 'running',
                'completed': 'completed',
                'skipped': 'completed',
                'partial_failed': 'failed',
                'failed': 'failed',
                'partial_cancelled': 'cancelled',
                'cancelled': 'cancelled',
            }.get(status_token, str(successor_execution.get('status') or 'queued').strip())
            successor_execution['status'] = execution_status
            successor_execution['late_fill_status'] = status_token or payload.get('status')
            successor_execution['updated_at'] = now_iso
            payload['successor_reopen_execution'] = successor_execution
        return payload

    def extract_semantic_materializer_prompt(
        self,
        payload: Any,
        *,
        capability: Optional[str],
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        normalized_capability = normalize_capability(capability)
        if normalized_capability == CAPABILITY_IMAGE_GENERATION:
            prompt = str(payload.get('artifact_prompt') or '').strip()
            return prompt or None
        if normalized_capability == CAPABILITY_TEXT_TO_SPEECH:
            prompt = str(payload.get('content_payload') or '').strip()
            return prompt or None
        return None

    def selected_reference_matches_capability(
        self,
        selected_reference_artifact: Optional[dict[str, Any]],
        capability: Optional[str],
        *,
        instance: Optional[dict[str, Any]] = None,
    ) -> bool:
        build_instance_trait_summary = self._hook('build_instance_trait_summary')

        if not isinstance(selected_reference_artifact, dict):
            return False
        artifact_type = str(selected_reference_artifact.get('type') or '').strip().lower()
        normalized_capability = normalize_capability(capability)
        if artifact_type == 'message':
            return normalized_capability in {
                CAPABILITY_CHAT,
                CAPABILITY_VISION_ANALYSIS,
                CAPABILITY_IMAGE_GENERATION,
            }
        if artifact_type == 'image':
            if normalized_capability in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_IMAGE_GENERATION}:
                return True
            if normalized_capability == CAPABILITY_CHAT and isinstance(instance, dict):
                traits = build_instance_trait_summary(instance)
                if traits.get('supports_vision'):
                    return True
                supported_capabilities = {
                    normalize_capability(item)
                    for item in (instance.get('supported_capabilities') or [])
                    if str(item or '').strip()
                }
                provider_capabilities = {
                    normalize_capability(item)
                    for item in (instance.get('provider_capabilities') or [])
                    if str(item or '').strip()
                }
                return (
                    CAPABILITY_VISION_ANALYSIS in supported_capabilities
                    or CAPABILITY_VISION_ANALYSIS in provider_capabilities
                )
            return False
        if artifact_type == 'audio':
            return normalized_capability == CAPABILITY_SPEECH_TO_TEXT
        artifact_kind = str(selected_reference_artifact.get('kind') or '').strip().lower()
        if artifact_type == 'document' or artifact_kind == 'pdf':
            return normalized_capability in {CAPABILITY_CHAT, CAPABILITY_VISION_ANALYSIS}
        if artifact_type == 'text':
            return normalized_capability in {CAPABILITY_TEXT_TO_SPEECH, CAPABILITY_CHAT, CAPABILITY_VISION_ANALYSIS}
        return False

    def select_matching_selected_reference_artifact(
        self,
        selected_reference_artifacts: Any,
        capability: Optional[str],
        *,
        instance: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        sanitize_selected_reference_artifacts = self._hook('sanitize_selected_reference_artifacts')

        normalized_capability = normalize_capability(capability)
        candidates = [
            candidate
            for candidate in sanitize_selected_reference_artifacts(selected_reference_artifacts)
            if str(candidate.get('type') or '').strip().lower() != 'message'
        ]
        if not candidates or not normalized_capability:
            return None
        for candidate in candidates:
            if self.selected_reference_matches_capability(
                candidate,
                normalized_capability,
                instance=instance,
            ):
                return candidate
        return None

    def should_attach_selected_reference_file_context(
        self,
        *,
        prompt: Any,
        capability: Optional[str],
        selected_reference_artifact: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(selected_reference_artifact, dict):
            return False
        if str(selected_reference_artifact.get('type') or '').strip().lower() == 'message':
            return False
        return True

    def build_selected_reference_prompt_prefix(
        self,
        selected_reference_artifacts: Any,
        capability: Optional[str],
    ) -> str:
        sanitize_selected_reference_artifacts = self._hook('sanitize_selected_reference_artifacts')

        normalized_capability = normalize_capability(capability)
        if normalized_capability not in {
            CAPABILITY_CHAT,
            CAPABILITY_VISION_ANALYSIS,
            CAPABILITY_IMAGE_GENERATION,
        }:
            return ''
        selected_references = sanitize_selected_reference_artifacts(selected_reference_artifacts)
        if not selected_references:
            return ''
        prompt_blocks: list[str] = []
        for message_reference in selected_references:
            if str(message_reference.get('type') or '').strip().lower() != 'message':
                continue
            content = str(message_reference.get('content') or '').strip()
            if not content:
                continue
            message_role = str(message_reference.get('message_role') or 'assistant').strip().lower() or 'assistant'
            if message_role not in {'user', 'assistant', 'system'}:
                message_role = 'assistant'
            role_label = (
                'assistant reply'
                if message_role == 'assistant'
                else 'user prompt'
                if message_role == 'user'
                else 'system message'
            )
            prompt_blocks.append(
                f'Selected prior {role_label} reference for this conversation turn. '
                'Treat it as bounded reference context only; the current user request remains the live instruction. '
                'Do not infer new tasks from this reference unless the current turn explicitly asks.\n\n'
                f'[{message_role}]\n{content}'
            )
        return '\n\n'.join(prompt_blocks)

    def apply_selected_reference_prompt_prefix(
        self,
        prompt: Any,
        selected_reference_artifacts: Any,
        capability: Optional[str],
    ) -> str:
        prompt_text = str(prompt or '').strip()
        prompt_prefix = self.build_selected_reference_prompt_prefix(
            selected_reference_artifacts,
            capability,
        )
        if not prompt_prefix:
            return prompt_text
        if not prompt_text:
            return prompt_prefix
        return f'{prompt_prefix}\n\nCurrent user request:\n{prompt_text}'

    def inject_selected_reference_into_chat_messages(
        self,
        messages: list[dict[str, Any]],
        selected_reference_artifact: Any,
    ) -> list[dict[str, Any]]:
        sanitize_selected_reference_artifacts = self._hook('sanitize_selected_reference_artifacts')

        injected = list(messages or [])
        message_references = [
            item for item in sanitize_selected_reference_artifacts(selected_reference_artifact)
            if str(item.get('type') or '').strip().lower() == 'message'
        ]
        for message_reference in reversed(message_references):
            content = str(message_reference.get('content') or '').strip()
            if not content:
                continue
            message_role = str(message_reference.get('message_role') or 'assistant').strip().lower() or 'assistant'
            if message_role not in {'user', 'assistant', 'system'}:
                message_role = 'assistant'
            role_label = (
                'assistant reply'
                if message_role == 'assistant'
                else 'user prompt'
                if message_role == 'user'
                else 'system message'
            )
            reference_note = {
                'role': 'system',
                'content': (
                    f'Selected prior {role_label} reference for this conversation turn. '
                    'Treat it as bounded reference context only; the current user request remains the live instruction. '
                    'Do not infer new tasks from this reference unless the current turn explicitly asks.\n\n'
                    f'[{message_role}]\n{content}'
                ),
            }
            insert_at = len(injected)
            if injected and str(injected[-1].get('role') or '').strip().lower() == 'user':
                insert_at = len(injected) - 1
            injected.insert(max(0, insert_at), reference_note)
        return injected

    def resolve_prepare_phase_contract(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        extract_responses_prompt = self._hook('extract_responses_prompt')

        route_info = route_payload if isinstance(route_payload, dict) else {}
        request_info = request_payload if isinstance(request_payload, dict) else {}
        route_runtime = (
            route_info.get('route_runtime')
            if isinstance(route_info.get('route_runtime'), dict)
            else {}
        )
        request_phase_graph = (
            dict(route_runtime.get('request_phase_graph'))
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else None
        )
        if not request_phase_graph:
            request_phase_graph = build_request_phase_graph(
                extract_responses_prompt(request_info),
                intent_prompt=extract_responses_current_turn_prompt(request_info),
                request_payload=request_info,
                route_payload=route_info,
            )
        if not current_phase_is_graph_resolved(request_phase_graph):
            return None
        if current_phase_capability(request_phase_graph) != CAPABILITY_CHAT:
            return None
        current_phase = self._current_phase_payload(request_phase_graph)
        current_phase_kind = str(current_phase.get('kind') or '').strip().lower()
        current_phase_role = str(current_phase.get('role') or '').strip().lower()
        downstream = downstream_phase_capabilities(request_phase_graph)
        if not downstream:
            return None
        if current_phase_kind != 'prepare' and current_phase_role != 'text_preparation':
            return None
        return {
            'phase_graph': request_phase_graph,
            'current_phase': current_phase,
            'downstream_capabilities': downstream,
            'reason': current_phase_reason(request_phase_graph),
        }

    def plain_alpha_image_prompt_prepare_authority(
        self,
        phase_graph: Any,
        *,
        expected_count: int,
    ) -> dict[str, Any]:
        """Prove that untyped A..N text is the direct producer for N image slots."""

        try:
            normalized_count = int(expected_count or 0)
        except (TypeError, ValueError):
            normalized_count = 0
        if not 2 <= normalized_count <= 26 or not isinstance(phase_graph, Mapping):
            return {}
        graph = dict(phase_graph)
        if not current_phase_is_graph_resolved(graph):
            return {}
        if current_phase_capability(graph) != CAPABILITY_CHAT:
            return {}
        current_phase = self._current_phase_payload(graph)
        current_phase_id = str(current_phase.get('phase_id') or '').strip()
        if not current_phase_id:
            return {}
        current_phase_kind = str(current_phase.get('kind') or '').strip().lower()
        current_phase_role = str(current_phase.get('role') or '').strip().lower()
        if current_phase_kind != 'prepare' and current_phase_role != 'text_preparation':
            return {}
        prompt_intent = (
            graph.get('prompt_intent')
            if isinstance(graph.get('prompt_intent'), Mapping)
            else {}
        )
        try:
            requested_count = int(prompt_intent.get('requested_visual_output_count') or 0)
        except (TypeError, ValueError):
            requested_count = 0
        if requested_count != normalized_count:
            return {}

        direct_children: list[dict[str, Any]] = []
        for raw_phase in graph.get('phases') or []:
            if not isinstance(raw_phase, Mapping):
                continue
            dependencies = [
                str(item or '').strip()
                for item in (raw_phase.get('depends_on') or [])
                if str(item or '').strip()
            ]
            if current_phase_id not in dependencies:
                continue
            direct_children.append(dict(raw_phase))
        if len(direct_children) != normalized_count:
            return {}
        if any(
            normalize_capability(item.get('capability')) != CAPABILITY_IMAGE_GENERATION
            or [
                str(value or '').strip()
                for value in (item.get('depends_on') or [])
                if str(value or '').strip()
            ] != [current_phase_id]
            for item in direct_children
        ):
            return {}
        queue_indexes: list[int] = []
        for child in direct_children:
            try:
                queue_indexes.append(int(child.get('queue_index') or 0))
            except (TypeError, ValueError):
                return {}
        if sorted(queue_indexes) != list(range(1, normalized_count + 1)):
            return {}
        return {
            'kind': 'graph_direct_image_producer_contract',
            'source_phase_id': current_phase_id,
            'expected_count': normalized_count,
        }

    def _format_prepare_phase_capability_label(self, capability: str) -> str:
        normalized = normalize_capability(capability)
        if normalized == CAPABILITY_IMAGE_GENERATION:
            return 'image generation'
        if normalized == CAPABILITY_TEXT_TO_SPEECH:
            return 'speech synthesis'
        if normalized == CAPABILITY_SPEECH_TO_TEXT:
            return 'speech transcription'
        if normalized == CAPABILITY_VISION_ANALYSIS:
            return 'vision analysis'
        return normalized.replace('_', ' ') if normalized else 'follow-up materialization'

    def numbered_audio_prepare_authority(
        self,
        phase_graph: Any,
    ) -> dict[str, Any]:
        """Prove that one prepare phase directly feeds N distinct TTS slots."""

        if not isinstance(phase_graph, Mapping):
            return {}
        graph = dict(phase_graph)
        if not current_phase_is_graph_resolved(graph):
            return {}
        if current_phase_capability(graph) != CAPABILITY_CHAT:
            return {}
        current_phase = self._current_phase_payload(graph)
        current_phase_id = str(current_phase.get('phase_id') or '').strip()
        current_phase_kind = str(current_phase.get('kind') or '').strip().lower()
        current_phase_role = str(current_phase.get('role') or '').strip().lower()
        if (
            not current_phase_id
            or (current_phase_kind != 'prepare' and current_phase_role != 'text_preparation')
        ):
            return {}
        prompt_intent = (
            graph.get('prompt_intent')
            if isinstance(graph.get('prompt_intent'), Mapping)
            else {}
        )
        if (
            not bool(prompt_intent.get('counted_audio_output_obligation'))
            or bool(prompt_intent.get('audio_output_count_exceeds_bound'))
        ):
            return {}
        requested_count = self._coerce_positive_int(
            prompt_intent.get('requested_audio_output_count')
        )
        if not 2 <= requested_count <= 6:
            return {}
        direct_tts_children: list[dict[str, Any]] = []
        for raw_branch in downstream_phase_records(graph):
            if not isinstance(raw_branch, Mapping):
                continue
            if normalize_capability(raw_branch.get('capability')) != CAPABILITY_TEXT_TO_SPEECH:
                continue
            dependencies = [
                str(item or '').strip()
                for item in (raw_branch.get('depends_on') or [])
                if str(item or '').strip()
            ]
            if dependencies != [current_phase_id]:
                continue
            direct_tts_children.append(dict(raw_branch))
        if len(direct_tts_children) != requested_count:
            return {}
        candidate_indexes: list[int] = []
        variants: list[dict[str, Any]] = []
        for branch in direct_tts_children:
            if str(branch.get('selection_policy') or '').strip().lower() != 'selected_candidate_only':
                return {}
            candidate_index = self._coerce_positive_int(
                branch.get('candidate_selection_index')
            )
            candidate_count = self._coerce_positive_int(
                branch.get('candidate_selection_count')
            )
            variant_index = self._coerce_positive_int(
                branch.get('audio_variant_index')
            )
            if (
                not candidate_index
                or candidate_count != requested_count
                or variant_index != candidate_index
            ):
                return {}
            candidate_indexes.append(candidate_index)
            variants.append(
                {
                    'index': candidate_index,
                    'lang_code': str(branch.get('lang_code') or '').strip().lower() or None,
                    'role': str(branch.get('audio_variant_role') or '').strip() or None,
                    'contract_source': str(
                        branch.get('audio_variant_contract_source') or ''
                    ).strip() or None,
                }
            )
        if sorted(candidate_indexes) != list(range(1, requested_count + 1)):
            return {}
        variants.sort(key=lambda item: int(item['index']))
        return {
            'kind': 'graph_direct_numbered_audio_producer_contract',
            'source_phase_id': current_phase_id,
            'expected_count': requested_count,
            'variants': variants,
        }

    def build_prepare_phase_system_message(
        self,
        prepare_contract: Optional[dict[str, Any]],
    ) -> Optional[dict[str, str]]:
        if not isinstance(prepare_contract, dict):
            return None
        downstream_capabilities = [
            normalize_capability(candidate)
            for candidate in (prepare_contract.get('downstream_capabilities') or [])
            if normalize_capability(candidate)
        ]
        if not downstream_capabilities:
            return None
        phase_graph = (
            prepare_contract.get('phase_graph')
            if isinstance(prepare_contract.get('phase_graph'), Mapping)
            else {}
        )
        numbered_audio_contract = self.numbered_audio_prepare_authority(phase_graph)
        downstream_labels = [
            self._format_prepare_phase_capability_label(candidate)
            for candidate in downstream_capabilities
        ]
        if len(downstream_labels) == 1:
            downstream_clause = downstream_labels[0]
        elif len(downstream_labels) == 2:
            downstream_clause = f'{downstream_labels[0]} and {downstream_labels[1]}'
        else:
            downstream_clause = ', '.join(downstream_labels[:-1]) + f', and {downstream_labels[-1]}'
        reason = str(prepare_contract.get('reason') or '').strip()
        lines = [
            _PREPARE_PHASE_SYSTEM_MARKER,
            'You are executing only the current text-preparation phase of a multi-phase Ollmo request.',
            f'Downstream phases will handle {downstream_clause} separately after this phase completes.',
            'Return only the prepared substantive text needed by those downstream phases.',
            'Do not pre-answer downstream artifact-dependent review, analysis, comparison, transcription, or final-summary tasks before their evidence branches have run.',
            'If the user says to review, analyze, compare, or summarize after an artifact is ready, leave that work to the downstream branch.',
            'Do not say that you cannot generate images, audio, or other media.',
            'Do not mention being text-only, incapable, unable to help, or that another model or route will finish the task.',
            'Preserve the user-requested tone, structure, wording constraints, and exactness requirements when they belong to the content itself.',
        ]
        if numbered_audio_contract:
            expected_count = int(numbered_audio_contract['expected_count'])
            lines.append(
                f'Return exactly {expected_count} numbered, directly speakable bodies using the labels '
                f'1. through {expected_count}. Each numbered body must be self-contained and distinct, '
                'and its text must contain only wording that should be spoken.'
            )
            lines.append(
                'Do not add any other heading, label, bullet, explanation, or meta commentary.'
            )
            language_labels = {'de': 'German', 'en': 'English'}
            for variant in numbered_audio_contract.get('variants') or []:
                lang_code = str(variant.get('lang_code') or '').strip().lower()
                role = str(variant.get('role') or '').strip()
                if not lang_code and not role:
                    continue
                language = language_labels.get(lang_code, lang_code or 'the requested language')
                role_clause = f' and fulfill the {role.replace("_", " ")} role' if role else ''
                lines.append(
                    f'Numbered body {int(variant["index"])} must be in {language}{role_clause}.'
                )
        else:
            lines.append(
                'Do not add headings, labels, prompt wrappers, bullet framing, or meta commentary unless the user explicitly requested them.'
            )
        if CAPABILITY_IMAGE_GENERATION in downstream_capabilities:
            lines.append(
                'If downstream image generation depends on this text, make the body prompt-ready and visually concrete.'
            )
        if CAPABILITY_TEXT_TO_SPEECH in downstream_capabilities:
            if numbered_audio_contract:
                lines.append(
                    'Each numbered body feeds exactly one downstream speech-synthesis branch.'
                )
            else:
                lines.append(
                    'If downstream speech depends on this text, make the body directly speakable without capability notes.'
                )
        if len(downstream_capabilities) > 1 and not numbered_audio_contract:
            lines.append(
                'When multiple downstream phases depend on the same text, produce one clean reusable body that all of them can consume.'
            )
        if reason:
            lines.append(f'Current phase reason: {reason}')
        return {
            'role': 'system',
            'content': '\n'.join(lines),
        }

    def inject_prepare_phase_contract_into_chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        prepare_contract = self.resolve_prepare_phase_contract(
            route_payload=route_payload,
            request_payload=request_payload,
        )
        if not prepare_contract:
            return list(messages or [])
        injected = list(messages or [])
        if any(
            str(item.get('role') or '').strip().lower() == 'system'
            and _PREPARE_PHASE_SYSTEM_MARKER in str(item.get('content') or '')
            for item in injected
            if isinstance(item, dict)
        ):
            return injected
        system_message = self.build_prepare_phase_system_message(prepare_contract)
        if not system_message:
            return injected
        return [system_message] + injected

    def build_response_semantic_phase_payload(
        self,
        *,
        output_text: str,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
        source_payload: Optional[dict[str, Any]] = None,
        capability: Optional[str] = None,
    ) -> dict[str, Any]:
        existing = self._extract_semantic_phase_payload_from_payload(source_payload)
        if existing:
            return existing
        if normalize_capability(capability) != CAPABILITY_CHAT:
            return {}
        prepare_contract = self.resolve_prepare_phase_contract(
            route_payload=route_payload,
            request_payload=request_payload,
        )
        prepared_text = str(output_text or '').strip()
        if not prepare_contract or not prepared_text:
            return {}
        downstream = [
            normalize_capability(candidate)
            for candidate in (prepare_contract.get('downstream_capabilities') or [])
            if normalize_capability(candidate)
        ]
        expected_visual_output_count = 0
        phase_graph = (
            prepare_contract.get('phase_graph')
            if isinstance(prepare_contract.get('phase_graph'), Mapping)
            else {}
        )
        if isinstance(phase_graph.get('prompt_intent'), Mapping):
            expected_visual_output_count = int(
                phase_graph.get('prompt_intent', {}).get('requested_visual_output_count') or 0
            )
        plain_alpha_authority = self.plain_alpha_image_prompt_prepare_authority(
            phase_graph,
            expected_count=expected_visual_output_count,
        )
        if not downstream:
            return {}
        semantic_payload: dict[str, Any] = {}
        if CAPABILITY_TEXT_TO_SPEECH in downstream:
            semantic_payload.update(
                {
                    key: value
                    for key, value in split_visible_tts_payload(prepared_text).items()
                    if value not in (None, '', [], {})
                }
            )
        if CAPABILITY_IMAGE_GENERATION in downstream:
            image_payload = {
                key: value
                for key, value in split_visible_image_payload(prepared_text).items()
                if value not in (None, '', [], {})
            }
            batch_prompts = self.extract_batch_image_prompts(
                prepared_text,
                expected_count=expected_visual_output_count,
                allow_plain_alpha_sequence=bool(plain_alpha_authority),
            )
            if len(batch_prompts) > 1:
                image_payload['batch_prompts'] = batch_prompts
                image_payload['batch_prompt_expected_count'] = (
                    expected_visual_output_count
                    if expected_visual_output_count >= 2
                    else len(batch_prompts)
                )
                if plain_alpha_authority:
                    image_payload['batch_prompts_source'] = 'semantic_prepare_phase_output'
                    image_payload['batch_prompt_source_phase_id'] = plain_alpha_authority[
                        'source_phase_id'
                    ]
                if str(image_payload.get('artifact_prompt_source') or '').strip() == 'full_display_text':
                    image_payload['artifact_prompt'] = batch_prompts[0]
                    image_payload['artifact_prompt_source'] = 'semantic_batch_prompts'
                    image_payload.setdefault('phase_summary', batch_prompts[0])
            if (
                semantic_payload.get('content_payload')
                and str(image_payload.get('artifact_prompt_source') or '').strip() == 'full_display_text'
            ):
                image_payload['artifact_prompt'] = semantic_payload.get('content_payload')
                image_payload['artifact_prompt_source'] = (
                    str(semantic_payload.get('content_payload_source') or '').strip()
                    or 'shared_content_payload'
                )
                image_payload.setdefault('phase_summary', semantic_payload.get('content_payload'))
            semantic_payload.update(image_payload)
        if not semantic_payload:
            semantic_payload = {
                'content_payload': prepared_text,
                'content_payload_source': 'current_phase_output',
            }
            if CAPABILITY_IMAGE_GENERATION in downstream:
                semantic_payload['artifact_prompt'] = prepared_text
                semantic_payload['artifact_prompt_source'] = 'current_phase_output'
                semantic_payload['phase_summary'] = prepared_text
            elif len(downstream) > 1:
                semantic_payload['phase_summary'] = prepared_text
        return semantic_payload

    def attach_response_semantic_phase_payload(
        self,
        payload: Optional[dict[str, Any]],
        *,
        output_text: str,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
        capability: Optional[str] = None,
    ) -> dict[str, Any]:
        updated = dict(payload or {})
        semantic_payload = self.build_response_semantic_phase_payload(
            output_text=output_text,
            route_payload=route_payload,
            request_payload=request_payload,
            source_payload=updated,
            capability=capability,
        )
        if not semantic_payload:
            return updated
        updated['phase_payload'] = {
            **(
                updated.get('phase_payload')
                if isinstance(updated.get('phase_payload'), dict)
                else {}
            ),
            **semantic_payload,
        }
        for key, value in semantic_payload.items():
            updated[key] = value
        return updated

    def plan_compound_execution_payload(
        self,
        payload: dict[str, Any],
        *,
        route_info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        plan_compound_execution_hook = self._hook('plan_compound_execution')
        normalize_request_payload = self._hook('normalize_request_payload')
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        load_running_instances = self._hook('load_running_instances')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        planner_timeout_seconds_for_payload = self._hook('planner_timeout_seconds_for_payload')
        extract_ghost_route_messages = self._hook('extract_ghost_route_messages')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')
        extract_responses_prompt = self._hook('extract_responses_prompt')
        execute_chat_backend_request = self._hook('execute_chat_backend_request')

        base_payload = normalize_request_payload(payload)
        current_turn_prompt = (
            extract_responses_current_turn_prompt(base_payload)
            or str(base_payload.get('_current_turn_prompt') or '').strip()
            or str(base_payload.get('_prompt_hint') or '').strip()
            or extract_responses_prompt(base_payload)
        )
        if current_turn_prompt:
            base_payload['_current_turn_prompt'] = current_turn_prompt
            base_payload['_prompt_hint'] = current_turn_prompt
        try:
            instances = merge_instances_with_runtime_status(
                load_running_instances(),
                path=runtime_status_path_getter(),
                refresh=True,
            )
            route_runtime = (
                route_info.get('route_runtime')
                if isinstance(route_info.get('route_runtime'), dict)
                else {}
            )
            semantic_role_profile = (
                route_runtime.get('semantic_role_profile')
                if isinstance(route_runtime.get('semantic_role_profile'), dict)
                else {}
            )
            planner_timeout_sec = planner_timeout_seconds_for_payload(
                base_payload,
                semantic_role_profile=semantic_role_profile,
            )
            planned_payload, planner_meta = plan_compound_execution_hook(
                base_payload,
                route_info=route_info,
                instances=instances,
                context_messages=extract_ghost_route_messages(base_payload),
                execute_chat_request=execute_chat_backend_request,
                planner_timeout_sec=planner_timeout_sec,
                semantic_role_profile=semantic_role_profile,
            )
            return planned_payload, planner_meta
        except Exception as exc:  # noqa: BLE001
            logging.warning('Ghost resolver fallback: %s', exc)
            return base_payload, {
                'attempted': True,
                'applied': False,
                'capability': normalize_capability(
                    (route_info or {}).get('capability')
                    or ((route_info or {}).get('instance') or {}).get('capability')
                    or base_payload.get('capability')
                ) or None,
                'reason': 'planner_error',
                'error': str(exc),
            }
