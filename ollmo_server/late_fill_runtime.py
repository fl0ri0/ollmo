"""Late-fill continuation resolver/executor owners for Ollmo."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from flask import current_app, has_app_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from helpers.session_controls import (
    normalize_reasoning_effort,
    resolve_reasoning_effort_for_instance,
)
from ollmo_g.control_hints import infer_tts_language_from_prompt
from ollmo_core.inference import TEXT_ARTIFACT_EXTENSIONS, extract_text_artifact_payloads
from ollmo_core.runtime_liveness import (
    runtime_instance_is_selectable,
    runtime_instance_score,
    runtime_liveness_summary,
)
from ollmo_core.transports import (
    ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
    persist_text_artifact_locally,
    strip_enclosing_text_artifact_fence,
    text_artifact_content_is_materializer_instruction_echo,
)
from ollmo_server.recovery_contract import (
    RECOVERY_ACTION_MANUAL_REVIEW,
    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
    RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
    RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
    RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
    RECOVERY_ACTION_SEMANTIC_REVIEW,
    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
    RECOVERY_ACTION_RETRY_SAME_BRANCH,
    RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
    normalize_recovery_suggested_action,
)
from ollmo_server.repair_gate_runtime import classify_repair_execution_policy
from ollmo_server.responses_runtime import terminal_repair_loop_is_fully_satisfied
from ollmo_server.response_semantics_runtime import (
    ResponseSemanticsRuntimeOwner,
    _bounded_visual_evidence_from_selected_message as _extract_bounded_visual_evidence_from_selected_message,
    _image_prompt_candidate_is_code_or_css_polluted,
    _image_prompt_batch_looks_like_weak_social_copy,
    _extract_filename_social_asset_image_prompt_lines,
    _extract_inline_labeled_image_prompt_lines,
    _extract_leading_plain_alpha_image_prompt_lines,
    _extract_sequential_bold_alpha_image_prompt_lines,
    _extract_html_image_card_prompt_units,
    _extract_numbered_image_prompt_section,
    _extract_social_manifest_pipe_image_prompt_lines,
    _strip_social_manifest_image_prompt_metadata,
    control_json_envelope_suspected,
)
from ollmo_services.artifact_contracts import sanitize_artifact_record
from ollmo_services.artifact_registry import refresh_text_artifact_record_from_saved_path
from ollmo_services.graph_rebase import (
    graph_rebase_prompt_contains_root,
    stable_graph_rebase_prompt_digest,
)
from ollmo_services.enforced_policy import (
    ENFORCED_POLICY_DEFAULT_ID,
    ENFORCED_POLICY_PRODUCT_DEFAULT_MODE,
    ENFORCED_POLICY_REVIEW_KIND,
)
from ollmo_services.response_frames import (
    RESPONSE_FRAME_STALE_PARENT_REASON,
    ResponseFrameParentCASMismatch,
)
from ollmo_services.tts_audio_integrity import (
    TTS_AUDIO_INTEGRITY_POLICY_ID,
    TTS_SEMANTIC_SOURCE_POLICY_ID,
    build_tts_audio_integrity_evidence,
    build_tts_semantic_source,
)


_DEPENDENCY_INPUT_MISSING_RE = re.compile(
    r'\b('
    r'(?:requires|expected|missing|needs?)\s+(?:an?\s+)?(?:audio|image|file)\s+file|'
    r'missing\s+dependency\s+(?:artifact|evidence|input)|'
    r'(?:audio|image)\s+file\s+(?:required|missing)|'
    r'no\s+(?:audio|image)\s+(?:file|artifact)'
    r')\b',
    re.IGNORECASE,
)
_PATH_ONLY_DEPENDENCY_EVIDENCE_RE = re.compile(
    r'^\s*(?:/[^ \t\r\n]+|[A-Za-z]:[\\/][^\r\n]+)\s*$'
)
_TTS_STT_SEMANTIC_POLICY_ID = TTS_SEMANTIC_SOURCE_POLICY_ID
_TTS_STT_MIN_SOURCE_TOKENS_FOR_FULL_POLICY = 4
_TTS_STT_MIN_TOKEN_RECALL = 0.85
_TTS_STT_MIN_TOKEN_PRECISION = 0.65
_TTS_STT_MIN_SEQUENCE_RATIO = 0.60
_TTS_STT_SHORT_MIN_TOKEN_F1 = 0.80
_TTS_STT_SHORT_MIN_SEQUENCE_RATIO = 0.86
_TTS_STT_NEGATION_TOKENS = {
    'kein',
    'keine',
    'keinem',
    'keinen',
    'keiner',
    'keines',
    'never',
    'nicht',
    'nie',
    'niemals',
    'no',
    'not',
}

_REASONING_EFFORT_KEYS = ('reasoning_effort', 'reasoningEffort')


def _retarget_model_scoped_reasoning_effort(
    payload: Mapping[str, Any],
    instance: Optional[dict[str, Any]],
    *,
    source_instance_id: str = '',
) -> dict[str, Any]:
    """Resolve inherited reasoning against the model that runs this branch."""

    updated = dict(payload or {})
    request_meta = (
        updated.get('request_meta')
        if isinstance(updated.get('request_meta'), Mapping)
        else {}
    )
    reasoning_provenance = (
        request_meta.get('reasoning_effort_control')
        if isinstance(request_meta.get('reasoning_effort_control'), Mapping)
        else {}
    )
    requested: Any = None
    for key in _REASONING_EFFORT_KEYS:
        if updated.get(key) not in (None, ''):
            requested = updated.get(key)
            break
    if requested in (None, '') and reasoning_provenance.get('value') not in (None, ''):
        requested = reasoning_provenance.get('value')
    for key in _REASONING_EFFORT_KEYS:
        updated.pop(key, None)

    target = instance if isinstance(instance, dict) else {}
    target_default = resolve_reasoning_effort_for_instance(None, target)
    if target_default is None:
        return updated

    try:
        normalized_requested = normalize_reasoning_effort(requested)
    except ValueError:
        normalized_requested = None
    target_instance_id = str(target.get('instance_id') or '').strip()
    if not source_instance_id:
        source_instance_id = str(
            reasoning_provenance.get('source_instance_id') or ''
        ).strip()
    same_instance = bool(
        source_instance_id
        and target_instance_id
        and source_instance_id == target_instance_id
    )

    if normalized_requested == 'off':
        resolved = 'off'
    elif same_instance and normalized_requested is not None:
        try:
            resolved = resolve_reasoning_effort_for_instance(
                normalized_requested,
                target,
            )
        except ValueError:
            resolved = target_default
    else:
        resolved = target_default
    if resolved is not None:
        updated['reasoning_effort'] = resolved
    return updated


_AUDIO_VARIANT_HEADER_RE = re.compile(
    r'^(?:audio\s*[- ]?\s*(?:version|variant|variante|fassung)|audiofassung|'
    r'spoken\s+(?:version|variant)|speech\s+(?:version|variant))\s*'
    r'(?P<index>\d{1,2})\b.*?:?\s*$',
    re.IGNORECASE,
)
_AUDIO_SPEAKABLE_FIELD_LABELS = {
    'audio text',
    'audio-text',
    'content',
    'erzaehlung',
    'erzählung',
    'inhalt',
    'narration',
    'script',
    'sprechtext',
    'spoken text',
    'tts text',
    'voiceover',
}
_AUDIO_SINGLE_SPEAKABLE_SECTION_TOKENS = (
    'audio text',
    'audio-text',
    'erzaehlung',
    'erzählung',
    'narration',
    'script',
    'sprechtext',
    'spoken text',
    'tts text',
    'voiceover',
)
_LATE_FILL_ALPHA_PROMPT_RESIDUE_RE = re.compile(
    r'^\s*(?:[-*#>\u2022]+\s*)?'
    r'(?:\*\*|__)?(?P<label>[A-Z])\s*'
    r'(?:(?:\*\*|__)\s*)?:\s*'
    r'(?:(?:\*\*|__)\s*)?.+$'
)
_LATE_FILL_COLLAPSED_ALPHA_LABEL_RE = re.compile(
    r'(?<![A-Za-z0-9_])(?P<label>[A-Z])\s*:\s+'
)
_BRANCH_LOCAL_ARTIFACT_ALPHA_LABEL_RE = re.compile(
    r'(?:^|[;:\n])\s*(?P<label>[A-Z])\s*(?:[-\u2013\u2014:])\s*',
    re.MULTILINE,
)
_BRANCH_LOCAL_VISION_DIRECTIVE_START_RE = re.compile(
    r'\b(?:'
    r'analy[sz](?:e|es|ed|ing)|inspect(?:s|ed|ing)?|examin(?:e|es|ed|ing)|'
    r'describ(?:e|es|ed|ing)|caption(?:s|ed|ing)?|summari[sz](?:e|es|ed|ing)|'
    r'analysier(?:e|en|t)?|untersuch(?:e|en|t)?|pr[uü]f(?:e|en|t)?|'
    r'beschreib(?:e|en|t)?|schilder(?:e|n|t)?|fass(?:e|en|t)?'
    r')\b',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_TARGET_RE = re.compile(
    r'\b(?:actual|attached|generated|visible|visual|image|picture|photo|artifact|'
    r'tats[aä]chlich|angeh[aä]ngt|erzeugt|sichtbar|bild|foto|artefakt)\w*\b|'
    r'\b(?:it|them)\b|\beach\s+one\b',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_DIRECTIVE_STOP_RE = re.compile(
    r'(?:,\s*)?\b(?:then|finally|afterwards|abschlie(?:ss|ß)end|anschlie(?:ss|ß)end)\s+'
    r'(?:compare|transcribe|generate|create|speak|read|vergleiche|transkribiere|erzeuge|erstelle|sprich|lies)\b|'
    r'\s+\b(?:and|und)\s+'
    r'(?:transcrib(?:e|es|ed|ing)|generate|create|compare|speak|read|'
    r'transkribier(?:e|en|t)?|erzeuge|erstelle|vergleiche|sprich|lies)\b',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_GLOBAL_JOIN_STOP_RE = re.compile(
    r'(?:'
    r'(?:,\s*)?\b(?:then|finally|afterwards|abschlie(?:ss|ß)end|anschlie(?:ss|ß)end)\s+|'
    r'\s+\b(?:and|und)\s+'
    r')'
    r'(?:return|give|provide|output|gib|liefere)\b'
    r'(?=[^.!?\n]{0,240}(?:'
    r'\bjson\s*(?:[-_/]\s*|\s+)(?:list(?:e)?|array)\b|'
    r'\btable\b|\b(?:in\s+the\s+chat|im\s+chat)\b|'
    r'\b(?:combined|global|final)\s+(?:answer|output|result|structure)\b'
    r'))',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_FOCUS_PATTERNS = (
    re.compile(
        r'\bob\s+(?P<focus>[^.!?\n]{1,180}?)\s+in\s+(?:beiden|allen)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:whether|if)\s+(?P<focus>[^.!?\n]{1,180}?)\s+'
        r'(?:appear|appears|occur|occurs|are\s+present|is\s+present)\s+in\s+both\b',
        re.IGNORECASE,
    ),
)
_BRANCH_LOCAL_STRUCTURED_JOIN_RE = re.compile(
    r'\bjson\s*(?:[-_/]\s*|\s+)(?:list(?:e)?|array)\b',
    re.IGNORECASE,
)
_BRANCH_LOCAL_ARTIFACT_IDENTITY_FIELD_RE = re.compile(
    r'\b(?:artifact_ref|artifact\s+(?:ref(?:erence)?|path)|'
    r'artefakt[_\s-]*(?:ref(?:erenz)?|pfad))\b',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_MULTI_ARTIFACT_RE = re.compile(
    r'\b(?:two|three|four|five|six|2|3|4|5|6|zwei|drei|vier|f[uü]nf|sechs|'
    r'multiple|several|mehrere)\b[^.!?\n]{0,56}\b'
    r'(?:images?|pictures?|artifacts?|bilder?|artefakte?)\b|'
    r'\b(?:each|every|all|jedes|jeden|alle)\b[^.!?\n]{0,32}\b'
    r'(?:images?|pictures?|artifacts?|bilder?|artefakte?)\b',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_FOR_MULTIPLE_LABELS_RE = re.compile(
    r'\b(?:for|f[uü]r)\s+(?:[A-Za-z]|[0-9]+)\s*(?:,|/)\s*'
    r'(?:[A-Za-z]|[0-9]+)',
    re.IGNORECASE,
)
_BRANCH_LOCAL_VISION_EXPLICIT_GLOBAL_JOIN_RE = re.compile(
    r'\b(?:in\s+the\s+chat|im\s+chat|multi[-\s]?artifact|'
    r'(?:combined|global|final)\s+(?:answer|output|result|structure|join))\b',
    re.IGNORECASE,
)
_DEPENDENCY_EVIDENCE_MISSING_CLAIM_RE = re.compile(
    r'\b('
    r'(?:i\s+)?(?:do\s+not|don[\'’]?t|cannot|can[\'’]?t)\s+(?:directly\s+)?(?:access|see|hear|inspect|analy[sz]e)|'
    r'no\s+(?:direct\s+)?access\s+to\s+(?:the\s+)?(?:generated\s+)?(?:audio|image|file|artifact)|'
    r'need\s+(?:a\s+)?(?:description|context|the\s+)?(?:generated\s+)?(?:audio|image|file|artifact)|'
    r'(?:please\s+)?(?:provide|upload|attach|send)\s+(?:an?\s+|the\s+)?(?:generated\s+)?(?:audio|image|file|artifact)[^.!?\n]{0,120}(?:transcribe|compare|analy[sz]e|inspect|process|use|review)?|'
    r'(?:audio|image|file|artifact)\s+(?:file\s+)?you\s+would\s+like\s+me\s+to\s+(?:transcribe|compare|analy[sz]e|inspect|process)|'
    r'keinen?\s+(?:direkten\s+)?zugriff|'
    r'kann\s+(?:ich\s+)?(?:das\s+)?(?:audio|bild|datei|artefakt|die\s+aufnahme|die\s+audiodatei|die\s+bilddatei)?[^.!?\n]{0,80}\b(?:nicht\s+)?(?:beurteilen|analysieren|sehen|hoeren|hören|verarbeiten)|'
    r'ben[oö]tige\s+(?:eine\s+)?(?:beschreibung|kontext|audiodatei|audioaufnahme|bildbeschreibung|bilddatei|datei|artefakt)|'
    r'kein(?:e[nsrm]?)?\s+(?:audio|audiodatei|audioaufnahme|bild|bilddatei|datei|artefakt|transkript)[^.!?\n]{0,120}(?:zur\s+verf[uü]gung\s+gestellt|bereitgestellt|vorhanden|verf[uü]gbar|hochgeladen|erhalten|erstellen)|'
    r'(?:stellen|lade|laden)\s+sie\s+mir\s+(?:das\s+|die\s+)?(?:audio|audiodatei|audioaufnahme|bild|bilddatei|datei|artefakt)[^.!?\n]{0,80}(?:zur\s+verf[uü]gung|hoch|bereit)|'
    r'kein(?:e[nsrm]?)?\s+(?:transkript|analyse|vergleich)[^.!?\n]{0,80}(?:erstellen|liefern|anfertigen|durchf[uü]hren)'
    r')\b',
    re.IGNORECASE,
)
_BRANCH_CONTRACT_REPAIR_RE = re.compile(
    r'\b('
    r'contract|execution\s+contract|branch\s+identity|obligation|'
    r'output\s+contract|validation|schema|control\s+validation'
    r')\b',
    re.IGNORECASE,
)
_SEMANTIC_EXECUTION_TERMINAL_STATUSES = {'cancelled', 'waived', 'superseded'}
_SEMANTIC_EXECUTION_CONTROL_ACTIONS = {
    'cancel': 'cancelled',
    'cancelled': 'cancelled',
    'stop': 'cancelled',
    'waive': 'waived',
    'waived': 'waived',
    'supersede': 'superseded',
    'superseded': 'superseded',
}
_LINK_REBIND_TEXT_EXTENSIONS = {'html', 'htm', 'css', 'js', 'mjs', 'cjs', 'json'}
_LINK_REBIND_HTML_TARGET_EXTENSIONS = {'css', 'js', 'mjs', 'cjs'}
_TERMINAL_MATERIALIZABLE_LOCAL_DEPENDENCY_EXTENSIONS = {'css', 'js', 'mjs', 'cjs', 'json'}
_LINK_REBIND_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
_LINK_REBIND_EXTENSION_FAMILIES = {
    'image': frozenset({'png', 'jpg', 'jpeg', 'webp', 'gif', 'avif', 'svg'}),
    'audio': frozenset({'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus', 'webm'}),
    'video': frozenset({'mp4', 'mov', 'm4v', 'webm', 'avi', 'mkv', 'ogv'}),
    'font': frozenset({'woff', 'woff2', 'ttf', 'otf', 'eot'}),
    'style': frozenset({'css', 'scss', 'sass', 'less'}),
    'script': frozenset({'js', 'mjs', 'cjs', 'ts'}),
}
_LINK_REBIND_URL_RE = re.compile(
    r'(?P<prefix>\burl\(\s*[\'"]?)(?P<url>[^\'")]+)(?P<suffix>[\'"]?\s*\))',
    re.IGNORECASE,
)
_LINK_REBIND_ATTR_RE = re.compile(
    r'(?P<prefix>\b(?:href|src)\s*=\s*)(?P<quote>[\'"])(?P<url>[^\'"]+)(?P=quote)',
    re.IGNORECASE,
)
_LINK_REBIND_FETCH_RE = re.compile(
    r'(?P<prefix>\bfetch\s*\(\s*)(?P<quote>[\'"])(?P<url>[^\'"]+)(?P=quote)(?=\s*[,\)])',
    re.IGNORECASE,
)
_INLINE_SCRIPT_RE = re.compile(
    r'<script\b[^>]*>(?P<body>[\s\S]*?)</script\s*>',
    re.IGNORECASE,
)
_JS_COLLECTION_ITERATOR_RE = re.compile(
    r'(?P<collection>(?:[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*)*[A-Za-z_$][A-Za-z0-9_$]*)'
    r'\s*\.\s*(?:forEach|map)\s*\(\s*(?:\(\s*)?'
    r'(?P<item>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\)\s*)?=>\s*\{',
    re.IGNORECASE,
)
_JS_FETCH_RESPONSE_ASSIGNMENT_RE = re.compile(
    r'\b(?:const|let|var)\s+(?P<response>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:await\s+)?$',
    re.IGNORECASE,
)
_JS_AWAITED_JSON_BINDING_RE = re.compile(
    r'\b(?:const|let|var)\s+(?P<binding>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*await\s+'
    r'(?P<response>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*json\s*\(\s*\)',
    re.IGNORECASE,
)
_JS_INVALID_IDENTIFIER_SEPARATOR_RE = re.compile(
    r'\b[A-Za-z_$][A-Za-z0-9_$]*(?P<separator>[^\x00-\x7f\s]+)'
    r'[A-Za-z_$][A-Za-z0-9_$]*\b'
)


def _static_fetch_url_is_local_file_dependency(value: str) -> bool:
    token = str(value or '').strip()
    if not token or _EXTERNAL_LINK_RE.match(token) or token.startswith(('/', '\\')):
        return False
    path_token = token.split('?', 1)[0].split('#', 1)[0].strip()
    extension = Path(path_token).suffix.lower().lstrip('.')
    return extension in _TERMINAL_MATERIALIZABLE_LOCAL_DEPENDENCY_EXTENSIONS
_LINK_REBIND_PLACEHOLDER_RE = re.compile(
    r'\b(?:(?:temp|temporary)[_\-\s]?(?:placeholder|image|asset|media)[_\-\s]?(?:src|source|url|path|ref|href)|'
    r'(?:placeholder|replace[_\-\s]?me|todo|tbd|dummy|sample|example)'
    r'(?:[_\-\s]?(?:src|source|url|path|ref|href))?|'
    r'asset[_\-\s]?placeholder|image[_\-\s]?placeholder|generated[_\-\s]?image|'
    r'(?:img|image|asset|media)[_\-\s]?(?:path|url|ref)(?:[_\-\s]?\d+)?|'
    r'hero[_\-\s]?image|image[_\-\s]?artifact|your[_\-\s]?(?:image|asset)|'
    r'platzhalter|ersetzen|beispiel)\b',
    re.IGNORECASE,
)
_LINK_REBIND_JSON_PATH_KEY_RE = re.compile(
    r'^(?:'
    r'(?:image|audio|video|media|asset|artifact|font|style|stylesheet|script)[_-](?:paths?|urls?|uris?|srcs?|hrefs?|files?)'
    r'|(?:paths?|urls?|uris?|srcs?|hrefs?|files?)'
    r'|.+[_-](?:paths?|urls?|uris?|srcs?|hrefs?|files?)'
    r')$',
    re.IGNORECASE,
)
_INLINE_TEXT_ARTIFACT_FENCE_RE = re.compile(
    r'```(?P<lang>[A-Za-z0-9_+.-]*)(?:[^\n`]*)?\n(?P<body>.*?)(?:\n```|```)',
    re.DOTALL,
)
_INLINE_COMPLETE_HTML_RE = re.compile(r'(?is)\b<!doctype\s+html\b|<html\b')
_INLINE_HTML_CSS_EXTENSIONS = {'html', 'htm', 'css'}
_TERMINAL_SYNTAX_REPAIR_EXTENSIONS = {'html', 'htm', 'css', 'json'}
_TEXT_ARTIFACT_REVISION_PRESERVATION_RE = re.compile(
    r'(?:'
    r'\b(?:keep|preserve|retain|leave)\b[\s\S]{0,140}'
    r'\b(?:rest|remainder|existing|unrelated|unchanged|intact|outside)\b|'
    r'\b(?:rest|remainder|existing|unrelated|unchanged)\b[\s\S]{0,100}'
    r'\b(?:intact|unchanged|preserved)\b|'
    r'\b(?:behalte|erhalte|bewahre|lass|halte)\b[\s\S]{0,140}'
    r'\b(?:rest|bestehende[nmrs]?|unver[aä]ndert|intakt|au(?:s|ß)erhalb)\b|'
    r'\b(?:rest|bestehende[nmrs]?)\b[\s\S]{0,100}'
    r'\b(?:unver[aä]ndert|intakt|erhalten)\b'
    r')',
    re.IGNORECASE,
)
_EXTERNAL_LINK_RE = re.compile(r'^(?:[a-z][a-z0-9+.-]*:|//|#)', re.IGNORECASE)
_TERMINAL_HERO_LOCAL_IMAGE_SIGNAL_RE = re.compile(
    r'\b(?:hero|hero[-_\s]?image|background|hintergrund|titelbild|title\s+image)\b',
    re.IGNORECASE,
)
_TERMINAL_HTML_HERO_BLOCK_RE = re.compile(
    r'<(?P<tag>header|section|main|div)\b(?P<attrs>[^>]*)>'
    r'(?P<body>[\s\S]{0,12000}?)</\s*(?P=tag)\s*>',
    re.IGNORECASE,
)
_TERMINAL_CSS_HERO_BLOCK_RE = re.compile(
    r'(?P<selectors>[^{}]*(?:[.#][A-Za-z0-9_-]*hero[A-Za-z0-9_-]*|\bheader\b)[^{}]*)\{(?P<body>[^{}]*)\}',
    re.IGNORECASE,
)
_RAW_IMAGE_RESULT_KEYS = {'image', 'image_base64', 'image_b64', 'image_data', 'png_base64', 'image_data_url'}
_RAW_IMAGE_LIST_KEYS = {'images'}
_AUTO_EXECUTABLE_REPAIR_RETRY_ACTIONS = {
    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
    RECOVERY_ACTION_RETRY_SAME_BRANCH,
}
_AUTO_EXECUTABLE_REPAIR_DEFAULT_MAX_ATTEMPTS = 6
_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS_LIMIT = 12
_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS_ENV = 'OLLMO_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS'
_COALESCED_TEXT_ARTIFACT_RECOVERY_TRIGGER = (
    'coalesced_text_artifact_atomic_retry'
)
_COALESCED_TEXT_ARTIFACT_RECOVERY_MAX_ATTEMPTS = 2
_TTS_AUTO_RECOVERY_POLICY_ID = 'tts_bounded_materialization_recovery_v1'
_TTS_AUTO_RECOVERY_TRIGGER = 'tts_auto_recovery'
_TTS_AUTO_RECOVERY_MAX_ATTEMPTS = 2
_PARTIAL_GRAPH_REBASE_HANDOFF_LOCK = threading.RLock()
_GRAPH_PATCH_TERMINAL_REVIEW_HANDOFF_LOCK = threading.RLock()
_GRAPH_PATCH_TERMINAL_REVIEW_RELATION = 'graph_patch_terminal_review'
_GRAPH_PATCH_TERMINAL_REVIEW_REASON = 'terminal_graph_patch_enforced_policy_denied'
_GRAPH_PATCH_TERMINAL_REVIEW_RUNTIME_EFFECT = 'audit_only_no_execution'
_GRAPH_PATCH_TERMINAL_SAFE_ENFORCED_CLASSES = frozenset({
    'duplicate_artifact_alias_canonicalization',
    'safe_additive_artifact_binding_repair',
    'safe_additive_dependency_repair',
    'safe_additive_missing_branch',
})
_GRAPH_PATCH_TERMINAL_REVIEW_RELATION_KEYS = frozenset({
    'audit_key',
    'audit_only',
    'executable',
    'kind',
    'owed_work',
    'parent_frame_id',
    'parent_frame_sequence',
    'parent_response_id',
    'patch_id',
    'policy_review_id',
    'proposal_id',
    'reason',
    'response_id',
    'review_id',
    'runtime_effect',
    'scheduled_branch_ids',
})


def _late_fill_instance_is_usable(instance: Mapping[str, Any], *, capability: Any = None) -> bool:
    return runtime_instance_is_selectable(instance, capability=capability)


def _late_fill_instance_is_mlx_vlm(instance: Mapping[str, Any]) -> bool:
    runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), Mapping) else {}
    backend = str(instance.get('backend') or runtime_status.get('backend') or '').strip().lower()
    backend_package = str(instance.get('backend_package') or runtime_status.get('backend_package') or '').strip().lower()
    backend_contract = str(instance.get('backend_contract') or runtime_status.get('backend_contract') or '').strip().lower()
    mlx_server = str(instance.get('mlx_server') or runtime_status.get('mlx_server') or '').strip().lower()
    server_kind = str(instance.get('server_kind') or runtime_status.get('server_kind') or '').strip().lower()
    return (
        backend_package == 'mlx_vlm'
        or backend_contract.startswith('mlx_vlm')
        or mlx_server == 'mlx_vlm'
        or server_kind == 'mlx_vlm'
        or (backend == 'mlx' and 'vlm' in backend_package)
    )


def _late_fill_instance_unusable_summary(instance: Mapping[str, Any], *, capability: Any = None) -> str:
    return runtime_liveness_summary(instance, capability=capability)


def _contract_text(value: Any) -> str:
    return str(value or '').strip()


def _first_contract_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, '', [], {}):
            return value
    return None


def _clean_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned = {
            str(key): _clean_contract_value(raw_value)
            for key, raw_value in value.items()
            if raw_value not in (None, '', [], {})
        }
        return {
            key: raw_value
            for key, raw_value in cleaned.items()
            if raw_value not in (None, '', [], {})
        }
    if isinstance(value, list):
        cleaned_items = [
            _clean_contract_value(item)
            for item in value
            if item not in (None, '', [], {})
        ]
        return [
            item
            for item in cleaned_items
            if item not in (None, '', [], {})
        ]
    if isinstance(value, tuple):
        return _clean_contract_value(list(value))
    if isinstance(value, str):
        return value.strip()
    return value


def _normalized_terminal_status(value: Any) -> str:
    token = str(value or '').strip().lower()
    return _SEMANTIC_EXECUTION_CONTROL_ACTIONS.get(token, token if token in _SEMANTIC_EXECUTION_TERMINAL_STATUSES else '')


def _payload_has_raw_image_result(value: Any) -> bool:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key or '').strip().lower()
            if key in _RAW_IMAGE_RESULT_KEYS and str(child or '').strip():
                return True
            if _payload_has_raw_image_result(child):
                return True
        return False
    if isinstance(value, list):
        return any(_payload_has_raw_image_result(item) for item in value)
    return False


def _raw_image_data_url_from_candidate(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    token = value.strip()
    if not token:
        return ''
    if token.lower().startswith('data:image/'):
        return token
    return f'data:image/png;base64,{token}'


def _extract_raw_image_data_url(value: Any) -> str:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key or '').strip().lower()
            if key in _RAW_IMAGE_RESULT_KEYS:
                data_url = _raw_image_data_url_from_candidate(child)
                if data_url:
                    return data_url
            if key in _RAW_IMAGE_LIST_KEYS and isinstance(child, list):
                for item in child:
                    data_url = _raw_image_data_url_from_candidate(item)
                    if data_url:
                        return data_url
            data_url = _extract_raw_image_data_url(child)
            if data_url:
                return data_url
        return ''
    if isinstance(value, list):
        for item in value:
            data_url = _extract_raw_image_data_url(item)
            if data_url:
                return data_url
    return ''


@dataclass
class LateFillRuntimeOwner:
    normalize_request_payload: Callable[[dict[str, Any]], dict[str, Any]]
    extract_request_meta: Callable[[dict[str, Any]], Any]
    attach_request_meta: Callable[[dict[str, Any]], dict[str, Any]]
    extract_responses_prompt: Callable[[Any], str]
    sanitize_selected_reference_artifacts: Callable[..., list[dict[str, Any]]]
    parse_bool: Callable[..., bool]
    extract_ghost_route_messages: Callable[[dict[str, Any]], list[dict[str, Any]]]
    response_registry_now_iso: Callable[[], str]
    max_recent_messages: int
    normalize_capability: Callable[[Any], Optional[str]]
    capability_text_to_speech: str
    capability_image_generation: str
    resolve_ghost_auto_route: Callable[..., tuple[Optional[dict[str, Any]], Optional[str]]]
    merge_instances_with_runtime_status: Callable[..., list[dict[str, Any]]]
    load_running_instances: Callable[[], list[dict[str, Any]]]
    runtime_status_path_getter: Callable[[], Any]
    instance_supports_capability: Callable[[dict[str, Any], str], bool]
    pick_prompt_preferred_instance: Callable[[list[dict[str, Any]], str], str]
    pick_default_capability_instance: Callable[[list[dict[str, Any]]], str]
    merge_request_meta_runtime_truth: Callable[..., dict[str, Any]]
    prepare_effective_request_data: Callable[..., tuple[dict[str, Any], dict[str, Any], Any, Any]]
    build_missing_required_session_controls: Callable[..., list[dict[str, Any]]]
    normalize_backend: Callable[[Any], Optional[str]]
    select_backend_request_model: Callable[..., str]
    build_responses_infer_execution_payload: Callable[..., tuple[dict[str, Any], dict[str, Any], bool, bool]]
    invoke_internal_api_json_route: Callable[..., tuple[dict[str, Any], int]]
    filter_responses_infer_result: Callable[..., dict[str, Any]]
    artifact_type_for_capability: Callable[[Any], Optional[str]]
    semantic_payload_for_capability: Callable[..., dict[str, Any]]
    get_response_lookup_record: Callable[[str], Optional[dict[str, Any]]]
    normalize_capability_list: Callable[[Any], list[str]]
    extract_pending_deferred_branches: Callable[..., list[dict[str, Any]]]
    extract_pending_deferred_capabilities: Callable[..., list[str]]
    build_pending_late_fill_branches: Callable[..., list[dict[str, Any]]]
    branch_id: Callable[[Mapping[str, Any]], str]
    branch_capability: Callable[[Mapping[str, Any]], Optional[str]]
    artifact_gap_is_already_fulfilled: Callable[..., bool]
    build_late_fill_state: Callable[..., dict[str, Any]]
    build_graph_closure_review: Callable[..., dict[str, Any]]
    attach_graph_closure_review_diagnostics: Callable[..., dict[str, Any]]
    finalize_response_frame_payload: Callable[..., dict[str, Any]]
    attach_late_fill_state: Callable[..., dict[str, Any]]
    touch_response_lookup: Callable[..., Optional[dict[str, Any]]]
    ensure_response_lookup_for_payload: Callable[..., Optional[dict[str, Any]]]
    build_canonical_response_artifacts: Callable[[dict[str, Any]], list[dict[str, Any]]]
    downstream_request_phase_batches: Callable[..., list[list[dict[str, Any]]]]
    late_fill_capability_counts: Callable[[list[dict[str, Any]]], dict[str, int]]
    normalize_late_fill_branches: Callable[[Any], list[dict[str, Any]]]
    log_unified_event: Callable[..., Any]
    execute_materialization_branches: Callable[..., dict[str, Any]]
    claim_response_late_fill: Callable[[str], bool]
    release_response_late_fill: Callable[[str], None]
    persist_image_data_url_locally: Callable[[Optional[str], str], Optional[str]]
    schedule_post_response_substrate_hygiene: Optional[Callable[..., Any]] = None
    attach_runtime_graph_repair_evidence: Optional[
        Callable[[dict[str, Any]], dict[str, Any]]
    ] = None
    review_terminal_graph_rebase: Optional[Callable[..., dict[str, Any]]] = None
    prepare_terminal_graph_patch_successor: Optional[Callable[..., dict[str, Any]]] = None
    load_latest_response_state: Optional[Callable[..., Mapping[str, Any]]] = None
    load_latest_response_observation_state: Optional[
        Callable[..., Mapping[str, Any]]
    ] = None
    load_external_targets: Optional[Callable[..., list[dict[str, Any]]]] = None
    execute_external_chat_phase: Optional[Callable[..., dict[str, Any]]] = None

    @staticmethod
    def _artifact_gap_is_required_text_materialization(artifact_gap: Optional[Mapping[str, Any]]) -> bool:
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        artifact_request = gap.get('artifact_request') if isinstance(gap.get('artifact_request'), Mapping) else {}
        text_artifact_requests = (
            gap.get('text_artifact_requests')
            if isinstance(gap.get('text_artifact_requests'), list)
            else []
        )
        extension = str(
            gap.get('text_artifact_extension')
            or artifact_request.get('extension')
            or ''
        ).strip().lower().lstrip('.')
        if not extension:
            for item in text_artifact_requests:
                if isinstance(item, Mapping):
                    extension = str(item.get('extension') or '').strip().lower().lstrip('.')
                    if extension:
                        break
        stage_direction = str(gap.get('stage_direction') or '').strip()
        return (
            stage_direction == 'materialize_requested_text_artifact'
            and (
                gap.get('requires_artifact') is True
                or bool(artifact_request)
                or extension in TEXT_ARTIFACT_EXTENSIONS
            )
            and extension in TEXT_ARTIFACT_EXTENSIONS
        )

    @staticmethod
    def _text_artifact_revision_required(*payloads: Mapping[str, Any]) -> bool:
        """Return whether existing text bytes are an edit input, not output proof."""

        revision_sources = {
            'canonical_predecessor_artifact',
            'history_source_edit',
            'selected_source_edit',
            'selected_text_source_edit',
        }
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            sources: list[Mapping[str, Any]] = [payload]
            for key in ('artifact_request', 'text_artifact_request', 'execution_contract'):
                nested = payload.get(key)
                if isinstance(nested, Mapping):
                    sources.append(nested)
            for source in sources:
                if (
                    source.get('text_artifact_revision_required') is True
                    or source.get('text_artifact_source_is_input') is True
                ):
                    return True
                provenance = str(
                    source.get('text_artifact_revision_source')
                    or source.get('text_artifact_source')
                    or source.get('source')
                    or ''
                ).strip().lower()
                if provenance in revision_sources:
                    return True
        return False

    @staticmethod
    def _text_artifact_revision_preservation_requested(prompt: Any) -> bool:
        """Recognize an explicit request to keep bytes outside the edit scope."""

        return bool(
            _TEXT_ARTIFACT_REVISION_PRESERVATION_RE.search(str(prompt or '').strip())
        )

    @staticmethod
    def _text_artifact_revision_anchor_tokens(
        content: Any,
        extension: str,
    ) -> set[str]:
        """Return stable structural anchors suitable for truncation detection."""

        text = str(content or '')
        normalized_extension = str(extension or '').strip().lower().lstrip('.')
        anchors: set[str] = set()
        if normalized_extension in {'html', 'htm'}:
            for match in re.finditer(
                r'\b(?P<kind>id|class)\s*=\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                kind = str(match.group('kind') or '').lower()
                for token in re.split(r'\s+', str(match.group('value') or '').strip()):
                    normalized = token.strip().lower()
                    if normalized:
                        anchors.add(f'{kind}:{normalized}')
            return anchors
        if normalized_extension == 'css':
            without_comments = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            for match in re.finditer(r'(?P<selectors>[^{}]+)\{', without_comments, flags=re.DOTALL):
                selectors = str(match.group('selectors') or '').strip()
                if not selectors:
                    continue
                for token_match in re.finditer(
                    r'(?<![\w-])(?P<token>[.#][A-Za-z_][A-Za-z0-9_-]*)',
                    selectors,
                ):
                    anchors.add(f'selector:{token_match.group("token").lower()}')
            return anchors
        if normalized_extension == 'json':
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                return anchors

            def visit(value: Any, path: tuple[str, ...]) -> None:
                if isinstance(value, Mapping):
                    for raw_key, child in value.items():
                        key = str(raw_key)
                        child_path = (*path, key)
                        anchors.add(f'json:{".".join(child_path)}')
                        visit(child, child_path)
                elif isinstance(value, list):
                    for child in value:
                        visit(child, (*path, '[]'))

            visit(decoded, ())
        return anchors

    @classmethod
    def _text_artifact_revision_preservation_review(
        cls,
        source_content: Any,
        candidate_content: Any,
        *,
        extension: str,
        target_path: str,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        """Fail closed when a preserve-scoped revision drops unrelated structure."""

        source = strip_enclosing_text_artifact_fence(
            str(source_content or ''),
            extension,
        ).strip()
        candidate = strip_enclosing_text_artifact_fence(
            str(candidate_content or ''),
            extension,
        ).strip()
        source_anchors = cls._text_artifact_revision_anchor_tokens(source, extension)
        candidate_anchors = cls._text_artifact_revision_anchor_tokens(candidate, extension)
        retained_anchors = source_anchors.intersection(candidate_anchors)
        missing_anchors = sorted(source_anchors.difference(candidate_anchors))
        anchor_retention = (
            len(retained_anchors) / len(source_anchors)
            if source_anchors
            else 1.0
        )
        size_retention = len(candidate) / len(source) if source else 1.0
        source_failure = not source
        anchor_failure = len(source_anchors) >= 5 and anchor_retention < 0.78
        size_failure = len(source) >= 800 and size_retention < 0.50
        evidence = {
            'kind': 'ollmo.text_artifact_revision_preservation_evidence',
            'version': 1,
            'policy': 'structural_anchor_retention_v1',
            'status': 'failed' if source_failure or anchor_failure or size_failure else 'passed',
            'target_path': target_path,
            'text_artifact_extension': str(extension or '').strip().lower().lstrip('.') or None,
            'source_size_chars': len(source),
            'candidate_size_chars': len(candidate),
            'size_retention_ratio': round(size_retention, 6),
            'source_anchor_count': len(source_anchors),
            'retained_anchor_count': len(retained_anchors),
            'anchor_retention_ratio': round(anchor_retention, 6),
            'missing_anchors': missing_anchors[:24],
            'failure_reason': (
                'source_snapshot_unavailable'
                if source_failure
                else 'structural_anchor_loss'
                if anchor_failure
                else 'severe_size_loss'
                if size_failure
                else None
            ),
        }
        if evidence['status'] == 'passed':
            return evidence, None
        return evidence, {
            'code': 'TEXT_ARTIFACT_REVISION_PRESERVATION_FAILED',
            'message': (
                'Required text artifact revision removed too much pre-existing structure despite an '
                'explicit preservation instruction. The source file remains authoritative until a '
                'complete branch-produced revision passes structural retention.'
            ),
            'target_path': target_path,
            'text_artifact_extension': evidence['text_artifact_extension'],
            'text_artifact_revision_preservation_evidence': evidence,
            'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
            'retryable': True,
        }

    @classmethod
    def _text_artifact_revision_write_proven(
        cls,
        branch: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        """Require branch-produced write evidence before a revision can fulfill."""

        if not cls._text_artifact_revision_required(branch):
            return False
        branch_id = str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
        target_path = cls._text_artifact_target_path_from_mapping(branch)
        artifact_request = (
            branch.get('artifact_request')
            if isinstance(branch.get('artifact_request'), Mapping)
            else {}
        )
        extension = str(
            branch.get('text_artifact_extension')
            or artifact_request.get('extension')
            or ''
        ).strip().lower().lstrip('.')

        # The synchronous chat path can persist a selected-source edit before
        # Late Fill exists. Its structured request plus saved artifact record
        # is current-response write evidence; a bare predecessor artifact is not.
        direct_records: list[Mapping[str, Any]] = []
        direct_request = (
            payload.get('text_artifact_request')
            if isinstance(payload, Mapping)
            and isinstance(payload.get('text_artifact_request'), Mapping)
            else {}
        )
        direct_saved_path = str(
            payload.get('saved_text_path') if isinstance(payload, Mapping) else ''
            or ''
        ).strip()
        if direct_request and direct_saved_path:
            direct_records.append(
                {
                    'type': 'text',
                    'path': direct_saved_path,
                    'text_artifact_request': direct_request,
                }
            )
        if isinstance(payload, Mapping):
            direct_records.extend(
                item
                for item in (payload.get('saved_text_artifacts') or [])
                if isinstance(item, Mapping)
            )
        for record in direct_records:
            request = (
                record.get('text_artifact_request')
                if isinstance(record.get('text_artifact_request'), Mapping)
                else record.get('artifact_request')
                if isinstance(record.get('artifact_request'), Mapping)
                else {}
            )
            source = str(request.get('source') or '').strip().lower()
            if source not in {'selected_source_edit', 'canonical_predecessor_artifact'}:
                continue
            if not cls._saved_text_artifact_matches_request(record, artifact_request):
                continue
            saved_path = str(record.get('path') or record.get('saved_text_path') or '').strip()
            if target_path and not cls._text_artifact_path_matches_target(saved_path, target_path):
                continue
            if saved_path and cls._text_artifact_saved_payload_error(
                saved_path,
                extension=extension,
            ) is None:
                return True

        late_fill = payload.get('late_fill') if isinstance(payload, Mapping) and isinstance(payload.get('late_fill'), Mapping) else {}
        for result in late_fill.get('fill_results') or []:
            if not isinstance(result, Mapping):
                continue
            result_branch_id = str(result.get('branch_id') or result.get('phase_id') or '').strip()
            if branch_id and result_branch_id != branch_id:
                continue
            proof = (
                result.get('text_artifact_revision_write_proof')
                if isinstance(result.get('text_artifact_revision_write_proof'), Mapping)
                else {}
            )
            if str(proof.get('status') or '').strip().lower() != 'applied':
                continue
            saved_path = str(result.get('saved_text_path') or proof.get('target_path') or '').strip()
            if target_path and not cls._text_artifact_path_matches_target(saved_path, target_path):
                continue
            if saved_path and cls._text_artifact_saved_payload_error(
                saved_path,
                extension=extension,
            ) is None:
                return True
        return False

    @classmethod
    def _artifact_gap_allows_excluded_candidate_reuse_for_text_repair(
        cls,
        artifact_gap: Optional[Mapping[str, Any]],
    ) -> bool:
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        if not cls._artifact_gap_is_required_text_materialization(gap):
            return False
        action = normalize_recovery_suggested_action(
            gap.get('repair_action') or gap.get('recovery_action') or gap.get('suggested_action'),
            default='',
        )
        if action not in {
            RECOVERY_ACTION_RETRY_SAME_BRANCH,
            RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
        }:
            return False
        execution_policy = str(
            gap.get('repair_execution_policy') or gap.get('execution_policy') or ''
        ).strip()
        if (
            not cls._runtime_scheduling_flag_enabled(gap.get('auto_execute'))
            and execution_policy != 'schedule_late_fill_branch'
        ):
            return False
        if gap.get('materialization_blocked') is True or gap.get('needs_external_input') is True:
            return False
        if gap.get('repair_work_available') is False:
            return False
        return True

    def _artifact_gap_allows_excluded_candidate_reuse_for_tts_recovery(
        self,
        artifact_gap: Optional[Mapping[str, Any]],
    ) -> bool:
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        recovery_attempt = (
            gap.get('recovery_attempt')
            if isinstance(gap.get('recovery_attempt'), Mapping)
            else {}
        )
        recovery_state = (
            gap.get('recovery_state')
            if isinstance(gap.get('recovery_state'), Mapping)
            else {}
        )
        execution_contract = (
            gap.get('execution_contract')
            if isinstance(gap.get('execution_contract'), Mapping)
            else {}
        )
        output_contract = (
            execution_contract.get('output_contract')
            if isinstance(execution_contract.get('output_contract'), Mapping)
            else {}
        )
        attempt_trigger = str(
            recovery_attempt.get('trigger') or ''
        ).strip().lower()
        state_trigger = str(
            recovery_state.get('trigger') or ''
        ).strip().lower()
        if attempt_trigger != state_trigger:
            return False
        if attempt_trigger not in {
            'explicit_retry_endpoint',
            _TTS_AUTO_RECOVERY_TRIGGER,
        }:
            return False
        automatic_tts_recovery = bool(
            attempt_trigger == _TTS_AUTO_RECOVERY_TRIGGER
        )
        capabilities = {
            self.normalize_capability(value)
            for value in (
                gap.get('expected_capability'),
                gap.get('active_capability'),
                execution_contract.get('capability'),
                recovery_attempt.get('capability'),
                recovery_state.get('capability'),
            )
            if str(value or '').strip()
        }
        if capabilities != {self.capability_text_to_speech}:
            return False
        output_type = str(
            gap.get('output_type')
            or execution_contract.get('output_type')
            or gap.get('missing_artifact_type')
            or ''
        ).strip().lower()
        if output_type != 'audio':
            return False
        if output_contract.get('required') is not True:
            return False
        action = normalize_recovery_suggested_action(
            recovery_state.get('suggested_action')
            or gap.get('repair_action')
            or gap.get('recovery_action')
            or gap.get('suggested_action'),
            default='',
        )
        if action not in {
            RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
            RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
        }:
            return False
        if automatic_tts_recovery and action not in {
            RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
            RECOVERY_ACTION_RETRY_SAME_BRANCH,
            RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
        }:
            return False
        if action == RECOVERY_ACTION_START_COMPATIBLE_INSTANCE:
            prior_error_code = str(
                recovery_attempt.get('prior_error_code')
                or recovery_state.get('prior_error_code')
                or gap.get('prior_error_code')
                or ''
            ).strip().upper()
            if prior_error_code != 'NO_COMPATIBLE_INSTANCE':
                return False
        if str(recovery_state.get('retry_scope') or '').strip() != 'same_branch':
            return False
        branch_id = str(
            gap.get('branch_id') or execution_contract.get('branch_id') or ''
        ).strip()
        attempt_branch_id = str(recovery_attempt.get('branch_id') or '').strip()
        state_branch_id = str(recovery_state.get('branch_id') or '').strip()
        if not branch_id or branch_id != attempt_branch_id or branch_id != state_branch_id:
            return False
        if recovery_attempt.get('preserve_intent') is not True:
            return False
        if recovery_state.get('preserve_intent') is not True:
            return False
        if (
            gap.get('needs_external_input') is True
            or recovery_state.get('needs_external_input') is True
        ):
            return False
        if automatic_tts_recovery:
            if (
                str(
                    recovery_attempt.get('recovery_policy_id') or ''
                ).strip()
                != _TTS_AUTO_RECOVERY_POLICY_ID
                or str(
                    recovery_state.get('recovery_policy_id') or ''
                ).strip()
                != _TTS_AUTO_RECOVERY_POLICY_ID
            ):
                return False
            if (
                recovery_attempt.get('auto_execute') is not True
                or recovery_state.get('auto_execute') is not True
            ):
                return False
            try:
                attempt_number = int(
                    recovery_attempt.get('attempt_number') or 0
                )
                maximum_attempts = int(
                    recovery_attempt.get('maximum_attempts') or 0
                )
                retry_count = int(
                    gap.get('auto_executable_repair_retry_count') or 0
                )
            except (TypeError, ValueError):
                return False
            if (
                attempt_number != 2
                or maximum_attempts
                != _TTS_AUTO_RECOVERY_MAX_ATTEMPTS
                or retry_count != 1
            ):
                return False
            prior_error_code = str(
                recovery_attempt.get('prior_error_code')
                or recovery_state.get('prior_error_code')
                or ''
            ).strip().upper()
            if not prior_error_code:
                return False
            if prior_error_code == 'TTS_AUDIO_INTEGRITY_REPAIR_REQUIRED':
                prior_integrity = (
                    recovery_attempt.get('prior_audio_integrity_evidence')
                    if isinstance(
                        recovery_attempt.get('prior_audio_integrity_evidence'),
                        Mapping,
                    )
                    else {}
                )
                if not (
                    str(prior_integrity.get('kind') or '').strip()
                    == 'ollmo.tts_audio_integrity_evidence'
                    and prior_integrity.get('version') == 1
                    and str(prior_integrity.get('policy_id') or '').strip()
                    == TTS_AUDIO_INTEGRITY_POLICY_ID
                    and prior_integrity.get('materialization_eligible') is False
                ):
                    return False
        return True

    @classmethod
    def _artifact_gap_is_authoritative_bounded_text_artifact_repair(
        cls,
        artifact_gap: Optional[Mapping[str, Any]],
    ) -> bool:
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        if not cls._artifact_gap_is_required_text_materialization(gap):
            return False
        artifact_request = (
            gap.get('artifact_request')
            if isinstance(gap.get('artifact_request'), Mapping)
            else {}
        )
        target_path = str(
            gap.get('text_artifact_target_path')
            or artifact_request.get('target_path')
            or ''
        ).strip()
        if not target_path:
            return False
        action = normalize_recovery_suggested_action(
            gap.get('repair_action') or gap.get('recovery_action') or gap.get('suggested_action'),
            default='',
        )
        if action not in {
            RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            RECOVERY_ACTION_RETRY_SAME_BRANCH,
        }:
            return False
        if gap.get('materialization_blocked') is True or gap.get('needs_external_input') is True:
            return False
        content_payload_source = str(gap.get('content_payload_source') or '').strip()
        text_artifact_source = str(
            gap.get('text_artifact_source')
            or artifact_request.get('source')
            or ''
        ).strip()
        return (
            content_payload_source in {
                'closure_html_css_selector_binding_review',
                'closure_linked_artifact_binding_review',
                'closure_local_dependency_link_review',
                'closure_composed_page_image_representation',
                'closure_hero_image_composition',
                'closure_text_artifact_syntax_sanity',
            }
            or text_artifact_source in {
                'closure_link_rebind',
                'closure_local_dependency_link',
                'closure_composed_page_image_representation',
                'closure_hero_image_composition',
                'closure_selector_binding_repair',
                'closure_syntax_repair',
            }
        )

    @staticmethod
    def _artifact_gap_is_required_image_materialization(artifact_gap: Optional[Mapping[str, Any]]) -> bool:
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        capability = str(
            gap.get('expected_capability')
            or gap.get('capability')
            or ''
        ).strip().lower()
        output_type = str(gap.get('output_type') or gap.get('artifact_type') or '').strip().lower()
        return (
            capability == 'image_generation'
            or output_type == 'image'
        ) and (
            gap.get('requires_artifact') is True
            or output_type == 'image'
            or bool(str(gap.get('artifact_prompt') or '').strip())
        )

    @staticmethod
    def _prefer_non_mlx_vlm_for_required_text_candidates(
        candidates: list[dict[str, Any]],
        artifact_gap: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not LateFillRuntimeOwner._artifact_gap_is_required_text_materialization(artifact_gap):
            return candidates
        non_mlx_candidates = [
            entry
            for entry in candidates
            if isinstance(entry, dict) and not _late_fill_instance_is_mlx_vlm(entry)
        ]
        return non_mlx_candidates or candidates

    @staticmethod
    def _runtime_scheduling_flag_enabled(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on', 'allow', 'allowed'}
        return False

    @staticmethod
    def _instance_identity_text(instance: Optional[Mapping[str, Any]]) -> str:
        payload = instance if isinstance(instance, Mapping) else {}
        runtime_status = payload.get('runtime_status') if isinstance(payload.get('runtime_status'), Mapping) else {}
        values: list[str] = []
        for source in (payload, runtime_status):
            for key in (
                'instance_id',
                'model',
                'model_name',
                'requested_model',
                'selected_model',
                'backend',
                'backend_package',
                'backend_contract',
                'server_kind',
                'capability',
            ):
                token = str(source.get(key) or '').strip()
                if token:
                    values.append(token)
        return ' '.join(values).lower()

    @classmethod
    def _late_fill_instance_is_large_chat_candidate(cls, instance: Optional[Mapping[str, Any]]) -> bool:
        payload = instance if isinstance(instance, Mapping) else {}
        capability = str(payload.get('capability') or '').strip().lower()
        text = cls._instance_identity_text(payload)
        return capability == 'chat' and ('gemma4:26b' in text or '26b' in text)

    @classmethod
    def _late_fill_instance_is_lightweight_chat_candidate(cls, instance: Optional[Mapping[str, Any]]) -> bool:
        payload = instance if isinstance(instance, Mapping) else {}
        capability = str(payload.get('capability') or '').strip().lower()
        if capability and capability != 'chat':
            return False
        if cls._late_fill_instance_is_large_chat_candidate(payload):
            return False
        text = cls._instance_identity_text(payload)
        return any(
            marker in text
            for marker in (
                'gemma4:e4b',
                'e4b',
                'granite',
                '8b',
                'lightweight',
                'small',
                'mini',
            )
        )

    @classmethod
    def _runtime_scheduling_context(
        cls,
        artifact_gap: Optional[Mapping[str, Any]],
        source_route_payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        gap_payload = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        route_payload = source_route_payload if isinstance(source_route_payload, Mapping) else {}
        for payload in (route_payload, gap_payload):
            nested_context = payload.get('runtime_scheduling_context') if isinstance(payload.get('runtime_scheduling_context'), Mapping) else {}
            context.update(dict(nested_context))
        for key in (
            'active_image_generation',
            'active_capabilities',
            'selected_chat_model_class',
            'selected_chat_model',
            'selected_chat_instance_id',
            'allow_gpu_heavy_concurrency',
            'repair_scope',
            'resource_class',
            'dependency_policy',
        ):
            for payload in (route_payload, gap_payload):
                value = payload.get(key)
                if value not in (None, '', [], {}):
                    context[key] = value
        route_instance = route_payload.get('instance') if isinstance(route_payload.get('instance'), Mapping) else {}
        if route_instance:
            context.setdefault('selected_chat_instance_id', route_instance.get('instance_id'))
            context.setdefault('selected_chat_model', route_instance.get('model') or route_instance.get('model_name'))
            if cls._late_fill_instance_is_lightweight_chat_candidate(route_instance):
                context.setdefault('selected_chat_model_class', 'lightweight')
            elif cls._late_fill_instance_is_large_chat_candidate(route_instance):
                context.setdefault('selected_chat_model_class', 'large')
        return {key: value for key, value in context.items() if value not in (None, '', [], {})}

    @classmethod
    def _runtime_context_has_active_image_generation(
        cls,
        artifact_gap: Optional[Mapping[str, Any]],
        source_route_payload: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        context = cls._runtime_scheduling_context(artifact_gap, source_route_payload)
        if cls._runtime_scheduling_flag_enabled(context.get('active_image_generation')):
            return True
        active_capabilities = context.get('active_capabilities')
        if isinstance(active_capabilities, list):
            return any(str(item or '').strip().lower() == 'image_generation' for item in active_capabilities)
        return str(active_capabilities or '').strip().lower() == 'image_generation'

    @classmethod
    def _runtime_context_prefers_lightweight_chat(
        cls,
        artifact_gap: Optional[Mapping[str, Any]],
        source_route_payload: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        context = cls._runtime_scheduling_context(artifact_gap, source_route_payload)
        model_class = str(context.get('selected_chat_model_class') or '').strip().lower()
        if model_class in {'lightweight', 'small', 'mini'}:
            return True
        selected_model = str(context.get('selected_chat_model') or '').strip().lower()
        if selected_model and any(marker in selected_model for marker in ('gemma4:e4b', 'e4b', 'granite', '8b')):
            return True
        return False

    @classmethod
    def _prefer_lightweight_chat_for_required_text_candidates(
        cls,
        candidates: list[dict[str, Any]],
        artifact_gap: Optional[Mapping[str, Any]],
        source_route_payload: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if not cls._artifact_gap_is_required_text_materialization(artifact_gap):
            return candidates
        active_image_generation = cls._runtime_context_has_active_image_generation(
            artifact_gap,
            source_route_payload,
        )
        lightweight_target = cls._runtime_context_prefers_lightweight_chat(
            artifact_gap,
            source_route_payload,
        )
        if not active_image_generation and not lightweight_target:
            return candidates
        lightweight_candidates = [
            entry
            for entry in candidates
            if isinstance(entry, dict) and cls._late_fill_instance_is_lightweight_chat_candidate(entry)
        ]
        if lightweight_candidates:
            return lightweight_candidates
        if active_image_generation:
            non_large_candidates = [
                entry
                for entry in candidates
                if isinstance(entry, dict) and not cls._late_fill_instance_is_large_chat_candidate(entry)
            ]
            if non_large_candidates:
                return non_large_candidates
        return candidates

    def _branch_is_image_generation(self, branch: Mapping[str, Any]) -> bool:
        return (
            self.branch_capability(branch) == self.capability_image_generation
            or str(branch.get('output_type') or branch.get('artifact_type') or '').strip().lower() == 'image'
        )

    def _branch_is_required_image_materialization(self, branch: Mapping[str, Any]) -> bool:
        if not isinstance(branch, Mapping) or not self._branch_is_image_generation(branch):
            return False
        output_contract = (
            branch.get('output_contract')
            if isinstance(branch.get('output_contract'), Mapping)
            else {}
        )
        if branch.get('required') is False or output_contract.get('required') is False:
            return False
        if branch.get('optional') is True or output_contract.get('optional') is True:
            return False
        return self._artifact_gap_is_required_image_materialization(branch) or self._branch_is_image_generation(branch)

    def _branch_is_required_tts_materialization(
        self,
        branch: Mapping[str, Any],
    ) -> bool:
        if (
            not isinstance(branch, Mapping)
            or self.branch_capability(branch) != self.capability_text_to_speech
            or not self.branch_id(branch)
        ):
            return False
        execution_contract = (
            branch.get('execution_contract')
            if isinstance(branch.get('execution_contract'), Mapping)
            else {}
        )
        output_contract = (
            branch.get('output_contract')
            if isinstance(branch.get('output_contract'), Mapping)
            else execution_contract.get('output_contract')
            if isinstance(execution_contract.get('output_contract'), Mapping)
            else {}
        )
        output_type = str(
            branch.get('output_type')
            or execution_contract.get('output_type')
            or output_contract.get('output_type')
            or ''
        ).strip().lower()
        if output_type and output_type != 'audio':
            return False
        if branch.get('required') is False or output_contract.get('required') is False:
            return False
        if branch.get('optional') is True or output_contract.get('optional') is True:
            return False
        return True

    def _required_tts_auto_recovery_allowed(
        self,
        branch: Mapping[str, Any],
        recovery_context: Mapping[str, Any],
    ) -> bool:
        if not self._branch_is_required_tts_materialization(branch):
            return False
        error_code = str(
            recovery_context.get('error_code') or ''
        ).strip().upper()
        if not error_code:
            return False
        if recovery_context.get('can_retry') is not True:
            return False
        if str(recovery_context.get('retry_scope') or '').strip() != 'same_branch':
            return False
        if recovery_context.get('preserve_intent') is not True:
            return False
        if recovery_context.get('needs_external_input') is True:
            return False
        if recovery_context.get('repair_work_available') is False:
            return False
        if any(
            recovery_context.get(key) is True
            for key in (
                'blocked_by_dependency_input',
                'blocked_by_branch_contract',
                'blocked_by_underplanned_promoted_obligations',
            )
        ):
            return False
        action = normalize_recovery_suggested_action(
            recovery_context.get('suggested_action'),
            default='',
        )
        if action not in {
            *_AUTO_EXECUTABLE_REPAIR_RETRY_ACTIONS,
            RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
        }:
            return False
        materialization_blocked = (
            recovery_context.get('materialization_blocked') is True
        )
        if error_code != 'TTS_AUDIO_INTEGRITY_REPAIR_REQUIRED':
            return not materialization_blocked
        if not materialization_blocked:
            return False
        if str(recovery_context.get('blocked_scope') or '').strip() != (
            'current_tts_branch'
        ):
            return False
        if recovery_context.get('repair_work_available') is not True:
            return False
        integrity_evidence = (
            recovery_context.get('audio_integrity_evidence')
            if isinstance(
                recovery_context.get('audio_integrity_evidence'),
                Mapping,
            )
            else {}
        )
        return bool(
            str(integrity_evidence.get('kind') or '').strip()
            == 'ollmo.tts_audio_integrity_evidence'
            and integrity_evidence.get('version') == 1
            and str(integrity_evidence.get('policy_id') or '').strip()
            == TTS_AUDIO_INTEGRITY_POLICY_ID
            and str(integrity_evidence.get('authority') or '').strip()
            == 'runtime_deterministic_audio_verification'
            and str(integrity_evidence.get('status') or '').strip().lower()
            in {'failed', 'unavailable'}
            and integrity_evidence.get('materialization_eligible') is False
        )

    def _branch_is_required_text_artifact(self, branch: Mapping[str, Any]) -> bool:
        return self._artifact_gap_is_required_text_materialization(branch)

    def _branch_is_required_artifact(self, branch: Mapping[str, Any]) -> bool:
        return (
            self._branch_is_image_generation(branch)
            or self._branch_is_required_text_artifact(branch)
            or branch.get('requires_artifact') is True
            or isinstance(branch.get('artifact_request'), Mapping)
        )

    def _branch_is_optional_advisory_helper(self, branch: Mapping[str, Any]) -> bool:
        if self._branch_is_required_artifact(branch):
            return False
        stage_direction = str(branch.get('stage_direction') or '').strip().lower()
        helper_text = ' '.join(
            str(branch.get(key) or '').strip().lower()
            for key in (
                'branch_id',
                'phase_id',
                'stage_direction',
                'role',
                'kind',
                'task_kind',
                'helper_kind',
                'description',
            )
            if str(branch.get(key) or '').strip()
        )
        return (
            stage_direction in {'run_global_semantic_closure_review', 'run_branch_semantic_review'}
            or any(
                marker in helper_text
                for marker in (
                    'semantic_review',
                    'semantic review',
                    'advisory',
                    'helper',
                    'image_state_enrichment',
                    'generated_image_state',
                )
            )
        )

    def _branch_scheduling_context(
        self,
        branch: Mapping[str, Any],
        artifact_gap: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        gap_payload = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        for payload in (gap_payload, branch):
            nested_context = payload.get('runtime_scheduling_context') if isinstance(payload.get('runtime_scheduling_context'), Mapping) else {}
            context.update(dict(nested_context))
            for key in ('repair_scope', 'resource_class', 'dependency_policy'):
                value = payload.get(key)
                if value not in (None, '', [], {}):
                    context[key] = value
        return {key: value for key, value in context.items() if value not in (None, '', [], {})}

    def _branch_is_target_artifact_snapshot_text_repair(
        self,
        branch: Mapping[str, Any],
        artifact_gap: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        capability = str(self.branch_capability(branch) or branch.get('capability') or '').strip().lower()
        if capability and capability != 'chat':
            return False
        output_type = str(branch.get('output_type') or branch.get('artifact_type') or '').strip().lower()
        if output_type and output_type not in {'text', 'code', 'document', 'html', 'css', 'json', 'markdown'}:
            return False
        context = self._branch_scheduling_context(branch, None)
        dependency_policy = str(context.get('dependency_policy') or '').strip().lower()
        if dependency_policy == 'target_artifact_snapshot_only':
            return True
        repair_scope = str(context.get('repair_scope') or '').strip().lower()
        resource_class = str(context.get('resource_class') or '').strip().lower()
        if repair_scope in {'syntax_only', 'text_artifact_syntax'} and resource_class in {'text_io', 'local_text_io'}:
            return True
        artifact_request = (
            branch.get('artifact_request')
            if isinstance(branch.get('artifact_request'), Mapping)
            else {}
        )
        target_path = str(
            branch.get('text_artifact_target_path')
            or artifact_request.get('target_path')
            or ''
        ).strip()
        role = str(branch.get('role') or '').strip()
        check_kind = str(branch.get('check_kind') or '').strip()
        content_payload_source = str(branch.get('content_payload_source') or '').strip()
        text_artifact_source = str(
            branch.get('text_artifact_source')
            or artifact_request.get('source')
            or ''
        ).strip()
        return bool(
            target_path
            and (
                check_kind == 'text_artifact_syntax_sanity'
                or role == 'text_artifact_syntax_repair'
                or content_payload_source == 'closure_text_artifact_syntax_sanity'
                or text_artifact_source == 'closure_syntax_repair'
            )
        )

    def _branch_conflicts_with_active_image_generation(
        self,
        branch: Mapping[str, Any],
        artifact_gap: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if self._branch_is_target_artifact_snapshot_text_repair(branch, artifact_gap):
            return False
        capability = str(self.branch_capability(branch) or '').strip().lower()
        if capability == 'chat':
            return True
        if capability in {'vision_analysis', 'image_analysis'}:
            return True
        return self._branch_is_optional_advisory_helper(branch)

    def _branch_allows_gpu_heavy_concurrency(
        self,
        branch: Mapping[str, Any],
        artifact_gap: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        gap_payload = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        context = self._branch_scheduling_context(branch, gap_payload)
        if self._branch_is_target_artifact_snapshot_text_repair(branch, gap_payload):
            return True
        for value in (
            branch.get('allow_gpu_heavy_concurrency'),
            gap_payload.get('allow_gpu_heavy_concurrency'),
            context.get('allow_gpu_heavy_concurrency'),
            context.get('explicit_allow_gpu_heavy_concurrency'),
        ):
            if self._runtime_scheduling_flag_enabled(value):
                return True
        return False

    def shape_active_late_fill_branches(
        self,
        active_branches: list[dict[str, Any]],
        *,
        artifact_gap: Optional[Mapping[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        branches = [dict(branch) for branch in active_branches if isinstance(branch, dict)]
        if len(branches) <= 1:
            return branches, {}
        if all(self._branch_allows_gpu_heavy_concurrency(branch, artifact_gap) for branch in branches):
            return branches, {}
        required_image_branches = [branch for branch in branches if self._branch_is_image_generation(branch)]
        if required_image_branches:
            non_conflicting = [
                branch
                for branch in branches
                if self._branch_is_image_generation(branch)
                or self._branch_allows_gpu_heavy_concurrency(branch, artifact_gap)
                or not self._branch_conflicts_with_active_image_generation(branch, artifact_gap)
            ]
            if len(non_conflicting) < len(branches):
                deferred = len(branches) - len(non_conflicting)
                return non_conflicting, {
                    'gpu_heavy_guard': 'deferred_non_image_branches',
                    'reason': 'required_image_generation_first',
                    'original_branch_count': len(branches),
                    'scheduled_branch_count': len(non_conflicting),
                    'deferred_branch_count': deferred,
                }
        required_artifact_branches = [
            branch
            for branch in branches
            if self._branch_is_required_artifact(branch)
        ]
        optional_helper_branches = [
            branch
            for branch in branches
            if self._branch_is_optional_advisory_helper(branch)
        ]
        if required_artifact_branches and optional_helper_branches and len(required_artifact_branches) < len(branches):
            deferred = len(branches) - len(required_artifact_branches)
            return required_artifact_branches, {
                'gpu_heavy_guard': 'deferred_optional_helpers',
                'reason': 'required_artifact_closure_first',
                'original_branch_count': len(branches),
                'scheduled_branch_count': len(required_artifact_branches),
                'deferred_branch_count': deferred,
                }
        return branches, {}

    @staticmethod
    def _text_artifact_request_from_branch_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
        branch = spec.get('branch') if isinstance(spec.get('branch'), Mapping) else {}
        prepare_args = spec.get('prepare_args') if isinstance(spec.get('prepare_args'), Mapping) else {}
        gap = prepare_args.get('artifact_gap') if isinstance(prepare_args.get('artifact_gap'), Mapping) else {}
        artifact_request = (
            branch.get('artifact_request')
            if isinstance(branch.get('artifact_request'), Mapping)
            else gap.get('artifact_request')
            if isinstance(gap.get('artifact_request'), Mapping)
            else {}
        )
        extension = str(
            branch.get('text_artifact_extension')
            or gap.get('text_artifact_extension')
            or artifact_request.get('extension')
            or ''
        ).strip().lower().lstrip('.')
        source_name = str(
            branch.get('text_artifact_source_name')
            or gap.get('text_artifact_source_name')
            or artifact_request.get('source_name')
            or f'generated-{extension or "txt"}'
        ).strip()
        source = str(
            branch.get('text_artifact_source')
            or gap.get('text_artifact_source')
            or artifact_request.get('source')
            or 'runtime_contract'
        ).strip()
        target_path = str(
            branch.get('text_artifact_target_path')
            or gap.get('text_artifact_target_path')
            or artifact_request.get('target_path')
            or ''
        ).strip()
        if not extension or not source_name:
            return {}
        request = {
            'extension': extension,
            'source_name': source_name,
            'source': source,
        }
        if target_path:
            request['target_path'] = target_path
        return request

    @staticmethod
    def _branch_spec_disables_required_text_artifact_coalescing(
        spec: Mapping[str, Any],
        branch: Mapping[str, Any],
    ) -> bool:
        prepare_args = spec.get('prepare_args') if isinstance(spec.get('prepare_args'), Mapping) else {}
        gap = prepare_args.get('artifact_gap') if isinstance(prepare_args.get('artifact_gap'), Mapping) else {}
        for source in (spec, branch, gap, prepare_args):
            if not isinstance(source, Mapping):
                continue
            if source.get('disable_coalesced_text_artifact_retry') is True:
                return True
            if source.get('coalesced_text_artifact_split_retry') is True:
                return True
            if str(source.get('failed_instance_id') or '').strip():
                return True
            for exclusion_key in (
                'excluded_instance_ids',
                'excludedInstanceIds',
                'exclude_instance_ids',
                'excludeInstanceIds',
            ):
                if isinstance(source.get(exclusion_key), list) and any(
                    str(item or '').strip()
                    for item in source.get(exclusion_key) or []
                ):
                    return True
            attempt = (
                source.get('attempt')
                if isinstance(source.get('attempt'), Mapping)
                else {}
            )
            if str(attempt.get('instance_id') or '').strip():
                return True
            for recovery_key in (
                'recovery_context',
                'recovery_state',
                'recovery_attempt',
            ):
                if isinstance(source.get(recovery_key), Mapping) and source.get(
                    recovery_key
                ):
                    return True
            if LateFillRuntimeOwner._text_artifact_revision_required(source):
                # Every revision owns a different complete source snapshot.
                # Combining those branches would give one model call only the
                # first file's edit base and would destroy branch-local truth.
                return True
            try:
                retry_count = int(source.get('auto_executable_repair_retry_count') or 0)
            except (TypeError, ValueError):
                retry_count = 0
            if retry_count > 0:
                return True
        return False

    def _coalesce_required_text_artifact_branch_specs(
        self,
        branch_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        specs = [dict(spec) for spec in branch_specs if isinstance(spec, dict)]
        if len(specs) <= 1:
            return specs
        branches = [
            spec.get('branch') if isinstance(spec.get('branch'), Mapping) else {}
            for spec in specs
        ]
        if not branches:
            return specs
        requests: list[dict[str, Any]] = []
        branch_refs: list[dict[str, Any]] = []
        seen_request_identities: set[tuple[str, ...]] = set()
        coalesced_indexes: set[int] = set()
        for index, (spec, branch) in enumerate(zip(specs, branches)):
            if self.normalize_capability(spec.get('capability')) != 'chat':
                continue
            if self._branch_spec_disables_required_text_artifact_coalescing(spec, branch):
                continue
            if not self._branch_is_required_text_artifact(branch):
                continue
            request = self._text_artifact_request_from_branch_spec(spec)
            extension = str(request.get('extension') or '').strip().lower()
            target_path = str(request.get('target_path') or '').strip()
            source_name = str(request.get('source_name') or '').strip().lower()
            request_identity = (
                ('target', target_path)
                if target_path
                else ('named', extension, source_name)
            )
            if (
                not request
                or not extension
                or request_identity in seen_request_identities
            ):
                continue
            seen_request_identities.add(request_identity)
            coalesced_indexes.add(index)
            requests.append(request)
            prepare_args = (
                spec.get('prepare_args')
                if isinstance(spec.get('prepare_args'), Mapping)
                else {}
            )
            branch_gap = (
                prepare_args.get('artifact_gap')
                if isinstance(prepare_args.get('artifact_gap'), Mapping)
                else {}
            )
            branch_refs.append(
                {
                    'branch_id': self.branch_id(branch),
                    'phase_id': str(branch.get('phase_id') or self.branch_id(branch)).strip() or self.branch_id(branch),
                    'capability': self.branch_capability(branch),
                    'output_type': branch.get('output_type'),
                    'text_artifact_extension': request.get('extension'),
                    'text_artifact_source_name': request.get('source_name'),
                    'text_artifact_source': request.get('source'),
                    'artifact_request': dict(request),
                    'execution_contract': self.build_execution_contract(
                        branch,
                        branch_gap,
                        capability='chat',
                    ),
                }
            )
        if len(requests) <= 1:
            return specs
        first_index = min(coalesced_indexes)
        base = dict(specs[first_index])
        base_prepare_args = (
            dict(base.get('prepare_args') or {})
            if isinstance(base.get('prepare_args'), Mapping)
            else {}
        )
        base_gap = (
            dict(base_prepare_args.get('artifact_gap') or {})
            if isinstance(base_prepare_args.get('artifact_gap'), Mapping)
            else {}
        )
        readable_cohort = '-'.join(
            str(ref.get('branch_id') or '').strip()
            for ref in branch_refs
            if str(ref.get('branch_id') or '').strip()
        )
        cohort_digest = hashlib.sha256(
            json.dumps(
                {'branches': branch_refs, 'requests': requests},
                sort_keys=True,
            ).encode('utf-8')
        ).hexdigest()[:12]
        coalesced_branch_id = (
            f'coalesced-text-artifacts-{readable_cohort[:64]}-{cohort_digest}'
        )
        coalesced_branch = {
            'branch_id': coalesced_branch_id,
            'phase_id': coalesced_branch_id,
            'capability': 'chat',
            'output_type': 'text',
            'stage_direction': 'materialize_requested_text_artifact',
            'requires_artifact': True,
            'text_artifact_requests': [dict(request) for request in requests],
            'coalesced_text_artifact_wave': True,
            'coalesced_text_artifact_branches': [dict(ref) for ref in branch_refs],
        }
        base_gap.update(
            {
                'branch_id': coalesced_branch_id,
                'phase_id': coalesced_branch_id,
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_requests': [dict(request) for request in requests],
                'artifact_request': dict(requests[0]),
                'text_artifact_extension': str(requests[0].get('extension') or ''),
                'text_artifact_source_name': str(requests[0].get('source_name') or ''),
                'text_artifact_source': str(requests[0].get('source') or ''),
                'coalesced_text_artifact_wave': True,
                'coalesced_text_artifact_branches': [dict(ref) for ref in branch_refs],
            }
        )
        base_prepare_args['artifact_gap'] = base_gap
        base['branch_id'] = coalesced_branch_id
        base['phase_id'] = coalesced_branch_id
        base['capability'] = 'chat'
        base['reservation_group'] = 'chat'
        base['branch'] = coalesced_branch
        base['text_artifact_requests'] = [dict(request) for request in requests]
        base['coalesced_text_artifact_wave'] = True
        base['coalesced_text_artifact_branches'] = [dict(ref) for ref in branch_refs]
        base['prepare_args'] = base_prepare_args
        coalesced_specs: list[dict[str, Any]] = []
        for index, spec in enumerate(specs):
            if index == first_index:
                coalesced_specs.append(base)
            if index in coalesced_indexes:
                continue
            coalesced_specs.append(spec)
        return coalesced_specs

    def _retry_failed_coalesced_text_artifact_materializations(
        self,
        branch_specs: list[dict[str, Any]],
        materialization_result: dict[str, Any],
        *,
        prepare_branch_plan: Callable[..., dict[str, Any]],
        execute_prepared_branch: Callable[[dict[str, Any]], dict[str, Any]],
        on_branch_progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Retry a failed fresh N-file branch once without splitting it."""

        def expand(raw: Mapping[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
            normalized = dict(raw or {})
            errors = dict(normalized.get('branch_errors') or {})
            plans = [
                dict(plan)
                for plan in (normalized.get('prepared_branch_plans') or [])
                if isinstance(plan, Mapping)
            ]
            planned = {str(plan.get('branch_id') or '').strip() for plan in plans}
            for spec in specs:
                cohort_id = str(spec.get('branch_id') or '').strip()
                if (
                    spec.get('coalesced_text_artifact_wave') is True
                    and cohort_id in errors
                    and cohort_id not in planned
                ):
                    plans.append(
                        {
                            'branch_id': cohort_id,
                            'phase_id': str(spec.get('phase_id') or cohort_id),
                            'capability': 'chat',
                            'branch': dict(spec),
                        }
                    )
            normalized['prepared_branch_plans'] = plans
            return self._expand_coalesced_text_artifact_materialization(normalized)

        initial = expand(materialization_result or {}, branch_specs)
        initial_results = dict(initial.get('branch_results') or {})
        initial_errors = dict(initial.get('branch_errors') or {})
        retry_specs: list[dict[str, Any]] = []
        markers: dict[str, dict[str, Any]] = {}
        for spec in branch_specs:
            requests = [
                dict(item)
                for item in (spec.get('text_artifact_requests') or [])
                if isinstance(item, Mapping)
            ]
            refs = [
                dict(item)
                for item in (spec.get('coalesced_text_artifact_branches') or [])
                if isinstance(item, Mapping)
            ]
            member_ids = [str(item.get('branch_id') or '').strip() for item in refs]
            member_errors = [initial_errors.get(member_id) for member_id in member_ids]
            if (
                spec.get('coalesced_text_artifact_wave') is not True
                or len(requests) < 2
                or len(requests) != len(member_ids)
                or any(not member_id for member_id in member_ids)
                or any(str(item.get('target_path') or '').strip() for item in requests)
                or all(
                    member_id in initial_results and member_id not in initial_errors
                    for member_id in member_ids
                )
                or not any(isinstance(error, Mapping) and error for error in member_errors)
                or any(
                    isinstance(error, Mapping) and error.get('retryable') is False
                    for error in member_errors
                )
            ):
                continue
            cohort_id = str(spec.get('branch_id') or '').strip()
            marker = {
                'kind': 'ollmo.coalesced_text_artifact_recovery',
                'trigger': _COALESCED_TEXT_ARTIFACT_RECOVERY_TRIGGER,
                'cohort_id': cohort_id,
                'member_branch_ids': member_ids,
                'artifact_manifest': requests,
                'attempt_number': 2,
                'maximum_attempts': _COALESCED_TEXT_ARTIFACT_RECOVERY_MAX_ATTEMPTS,
                'prior_error_code': str(
                    (member_errors[0] or {}).get('code') or 'BACKEND_ERROR'
                ).strip().upper(),
                'atomic_set_required': True,
                'materialization_authority': 'ollmo_runtime',
            }
            retry_spec = copy.deepcopy(spec)
            retry_spec['coalesced_text_artifact_recovery'] = dict(marker)
            prepare_args = dict(retry_spec.get('prepare_args') or {})
            gap = dict(prepare_args.get('artifact_gap') or {})
            gap['coalesced_text_artifact_recovery'] = dict(marker)
            prepare_args.update({'artifact_gap': gap, 'failed_instance_id': None})
            retry_spec['prepare_args'] = prepare_args
            retry_specs.append(retry_spec)
            markers[cohort_id] = marker

        if not retry_specs:
            return initial

        retry_raw = self.execute_materialization_branches(
            retry_specs,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
            on_branch_progress=on_branch_progress,
            async_branch_progress=True,
        )
        retry = expand(retry_raw, retry_specs)
        retry_results = dict(retry.get('branch_results') or {})
        retry_errors = dict(retry.get('branch_errors') or {})
        retried_members = {
            member_id
            for marker in markers.values()
            for member_id in marker['member_branch_ids']
        }
        combined_results = {
            key: value for key, value in initial_results.items()
            if key not in retried_members
        }
        combined_errors = {
            key: value for key, value in initial_errors.items()
            if key not in retried_members
        }
        history: list[dict[str, Any]] = []
        for marker in markers.values():
            member_ids = marker['member_branch_ids']
            saved_paths = [
                str(
                    (
                        (retry_results.get(member_id) or {}).get('infer_result')
                        or {}
                    ).get('saved_text_path')
                    or ''
                ).strip()
                for member_id in member_ids
            ]
            completed = (
                all(
                    member_id in retry_results and member_id not in retry_errors
                    for member_id in member_ids
                )
                and all(saved_paths)
                and len(set(saved_paths)) == len(member_ids)
                and len({str(Path(path).parent) for path in saved_paths}) == 1
            )
            if completed:
                combined_results.update(
                    {member_id: retry_results[member_id] for member_id in member_ids}
                )
            else:
                for member_id in member_ids:
                    combined_errors[member_id] = {
                        **dict(retry_errors.get(member_id) or {}),
                        'code': 'COALESCED_TEXT_ARTIFACT_COHORT_RETRY_EXHAUSTED',
                        'message': 'The bounded N-file retry did not return the complete set.',
                        'retryable': False,
                        'coalesced_text_artifact_recovery': dict(marker),
                    }
            history.append({**marker, 'status': 'completed' if completed else 'exhausted'})

        combined = dict(initial)
        combined['branch_results'] = combined_results
        combined['branch_errors'] = combined_errors
        combined['prepared_branch_plans'] = [
            *[
                dict(plan)
                for plan in (initial.get('prepared_branch_plans') or [])
                if isinstance(plan, Mapping)
                and str(plan.get('branch_id') or '').strip()
                not in retried_members
            ],
            *[
                dict(plan)
                for plan in (retry.get('prepared_branch_plans') or [])
                if isinstance(plan, Mapping)
            ],
        ]
        combined['coalesced_text_artifact_recovery_history'] = history
        combined['materialization_concurrency_policies'] = [
            dict(policy)
            for policy in (
                materialization_result.get('concurrency_policy'),
                retry_raw.get('concurrency_policy'),
            )
            if isinstance(policy, Mapping) and policy
        ]
        if isinstance(retry_raw.get('concurrency_policy'), Mapping):
            combined['concurrency_policy'] = dict(
                retry_raw.get('concurrency_policy') or {}
            )
        return combined

    @staticmethod
    def _normalized_text_artifact_path_for_match(path: Any) -> str:
        token = str(path or '').strip()
        if not token:
            return ''
        try:
            return str(Path(token).expanduser().resolve(strict=False))
        except OSError:
            return str(Path(token).expanduser())

    @classmethod
    def _text_artifact_path_matches_target(cls, path: Any, target_path: Any) -> bool:
        normalized_path = cls._normalized_text_artifact_path_for_match(path)
        normalized_target = cls._normalized_text_artifact_path_for_match(target_path)
        return bool(normalized_path and normalized_target and normalized_path == normalized_target)

    @classmethod
    def _text_artifact_target_path_from_mapping(cls, payload: Any) -> str:
        if not isinstance(payload, Mapping):
            return ''
        for key in ('target_path', 'text_artifact_target_path'):
            value = str(payload.get(key) or '').strip()
            if value:
                return value
        for key in ('text_artifact_request', 'artifact_request'):
            value = cls._text_artifact_target_path_from_mapping(payload.get(key))
            if value:
                return value
        execution_contract = payload.get('execution_contract')
        if isinstance(execution_contract, Mapping):
            value = cls._text_artifact_target_path_from_mapping(execution_contract)
            if value:
                return value
        return ''

    @classmethod
    def _saved_text_artifact_matches_request(cls, artifact: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
        request_target_path = cls._text_artifact_target_path_from_mapping(request)
        if request_target_path:
            artifact_path = str(artifact.get('path') or artifact.get('saved_text_path') or '').strip()
            if not cls._text_artifact_path_matches_target(artifact_path, request_target_path):
                return False
        artifact_request = (
            artifact.get('text_artifact_request')
            if isinstance(artifact.get('text_artifact_request'), Mapping)
            else artifact.get('artifact_request')
            if isinstance(artifact.get('artifact_request'), Mapping)
            else {}
        )
        artifact_extension = str(
            artifact.get('text_artifact_extension')
            or artifact_request.get('extension')
            or Path(str(artifact.get('path') or artifact.get('saved_text_path') or '')).suffix.lstrip('.')
            or ''
        ).strip().lower().lstrip('.')
        request_extension = str(request.get('extension') or '').strip().lower().lstrip('.')
        if request_extension and artifact_extension and artifact_extension != request_extension:
            return False
        artifact_source_name = str(
            artifact.get('text_artifact_source_name')
            or artifact_request.get('source_name')
            or Path(str(artifact.get('path') or artifact.get('saved_text_path') or '')).stem
            or ''
        ).strip().lower()
        request_source_name = str(request.get('source_name') or '').strip().lower()
        return not request_source_name or not artifact_source_name or artifact_source_name == request_source_name

    @staticmethod
    def _read_small_text_artifact(path: Any, *, max_bytes: int = 512_000) -> str:
        token = str(path or '').strip()
        if not token:
            return ''
        try:
            target = Path(token).expanduser()
            if not target.is_file() or target.stat().st_size > max_bytes:
                return ''
            return target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return ''

    @classmethod
    def _text_artifact_revision_source_payload(
        cls,
        branch: Mapping[str, Any],
        artifact_gap: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build one bounded, complete source snapshot for an explicit edit."""

        if not cls._text_artifact_revision_required(branch, artifact_gap):
            return {}
        target_path = (
            cls._text_artifact_target_path_from_mapping(branch)
            or cls._text_artifact_target_path_from_mapping(artifact_gap)
        )
        if not target_path:
            return {
                'branch_contract_error': 'text_revision_source_unavailable',
                'materialization_blocked': True,
                'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                'text_artifact_revision_binding_state': 'missing_target_path',
            }
        try:
            target = Path(target_path).expanduser()
            size_bytes = target.stat().st_size if target.is_file() else -1
            if size_bytes < 0:
                raise OSError('revision source is not a file')
            # A partial source packet cannot uphold "keep the rest intact".
            # Fail closed instead of truncating large files in the model prompt.
            if size_bytes > 90_000:
                return {
                    'branch_contract_error': 'text_revision_source_exceeds_prompt_bound',
                    'materialization_blocked': True,
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                    'text_artifact_revision_binding_state': 'source_too_large',
                    'text_artifact_revision_source_size_bytes': size_bytes,
                }
            source_content = target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return {
                'branch_contract_error': 'text_revision_source_unavailable',
                'materialization_blocked': True,
                'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                'text_artifact_revision_binding_state': 'source_unreadable',
            }
        if not source_content:
            return {
                'branch_contract_error': 'text_revision_source_unavailable',
                'materialization_blocked': True,
                'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                'text_artifact_revision_binding_state': 'source_empty',
            }
        return {
            'content_payload': source_content,
            'content_payload_source': 'canonical_predecessor_text_artifact_snapshot',
            'text_artifact_revision_required': True,
            'text_artifact_source_is_input': True,
            'text_artifact_revision_binding_state': 'bound',
            'text_artifact_revision_source_path': target_path,
            'text_artifact_revision_source_size_bytes': size_bytes,
            'text_artifact_revision_source_sha256': hashlib.sha256(
                source_content.encode('utf-8')
            ).hexdigest(),
            'dependency_payload_policy': 'preserve_text_artifact_revision_source',
            'suppress_reference_file_context': True,
        }

    def _expand_coalesced_text_artifact_materialization(
        self,
        materialization_result: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(materialization_result or {})
        prepared_plans = [
            dict(plan)
            for plan in (result.get('prepared_branch_plans') or [])
            if isinstance(plan, Mapping)
        ]
        branch_results = (
            dict(result.get('branch_results') or {})
            if isinstance(result.get('branch_results'), Mapping)
            else {}
        )
        branch_errors = (
            dict(result.get('branch_errors') or {})
            if isinstance(result.get('branch_errors'), Mapping)
            else {}
        )
        expanded_results: dict[str, dict[str, Any]] = {}
        expanded_errors: dict[str, dict[str, Any]] = {}
        expanded_plans: list[dict[str, Any]] = []
        expanded_coalesced_branch_ids: set[str] = set()

        def member_execution_contract(
            plan: Mapping[str, Any],
            ref: Mapping[str, Any],
        ) -> dict[str, Any]:
            branch_id = str(ref.get('branch_id') or '').strip()
            phase_id = str(ref.get('phase_id') or branch_id).strip() or branch_id
            contract = dict(
                ref.get('execution_contract')
                if isinstance(ref.get('execution_contract'), Mapping)
                else plan.get('execution_contract')
                if isinstance(plan.get('execution_contract'), Mapping)
                else {}
            )
            workload_ref = dict(contract.get('workload_task_ref') or {})
            obligation_ref = dict(contract.get('output_obligation_ref') or {})
            workload_ref.update({'branch_id': branch_id, 'phase_id': phase_id})
            obligation_ref.update(
                {
                    'branch_id': branch_id,
                    'phase_id': phase_id,
                    'output_type': 'text',
                }
            )
            contract.update(
                {
                    'branch_id': branch_id,
                    'phase_id': phase_id,
                    'capability': 'chat',
                    'output_type': 'text',
                    'workload_task_ref': workload_ref,
                    'output_obligation_ref': obligation_ref,
                }
            )
            return contract

        for plan in prepared_plans:
            branch = plan.get('branch') if isinstance(plan.get('branch'), Mapping) else {}
            nested_branch = branch.get('branch') if isinstance(branch.get('branch'), Mapping) else {}
            branch_refs = (
                branch.get('coalesced_text_artifact_branches')
                if isinstance(branch.get('coalesced_text_artifact_branches'), list)
                else nested_branch.get('coalesced_text_artifact_branches')
                if isinstance(nested_branch.get('coalesced_text_artifact_branches'), list)
                else []
            )
            coalesced_branch_id = str(
                plan.get('branch_id')
                or branch.get('branch_id')
                or nested_branch.get('branch_id')
                or ''
            ).strip()
            if not branch_refs or not coalesced_branch_id:
                expanded_plans.append(plan)
                continue
            expanded_coalesced_branch_ids.add(coalesced_branch_id)
            source_result = branch_results.get(coalesced_branch_id)
            source_error = branch_errors.get(coalesced_branch_id)
            if source_error:
                for ref in branch_refs:
                    branch_id = str((ref or {}).get('branch_id') or '').strip()
                    if branch_id:
                        expanded_errors[branch_id] = dict(source_error)
                continue
            if not isinstance(source_result, Mapping):
                continue
            infer_result = source_result.get('infer_result') if isinstance(source_result.get('infer_result'), Mapping) else {}
            external_provider_block = (
                infer_result.get('external_provider_block')
                if isinstance(
                    infer_result.get('external_provider_block'),
                    Mapping,
                )
                else {}
            )
            if external_provider_block:
                for ref in branch_refs:
                    if not isinstance(ref, Mapping):
                        continue
                    branch_id = str(ref.get('branch_id') or '').strip()
                    if not branch_id:
                        continue
                    phase_id = str(
                        ref.get('phase_id') or branch_id
                    ).strip() or branch_id
                    branch_plan = dict(plan)
                    branch_plan['branch_id'] = branch_id
                    branch_plan['phase_id'] = phase_id
                    branch_plan['execution_contract'] = member_execution_contract(
                        plan,
                        ref,
                    )
                    expanded_plans.append(branch_plan)
                    expanded_results[branch_id] = {
                        **dict(source_result),
                        'branch_id': branch_id,
                        'phase_id': phase_id,
                        'execution_contract': dict(
                            branch_plan['execution_contract']
                        ),
                        'infer_result': dict(infer_result),
                    }
                continue
            saved_text_artifacts: list[dict[str, Any]] = []
            seen_saved_text_paths: set[str] = set()

            def add_saved_text_artifact(raw_item: Any) -> None:
                if not isinstance(raw_item, Mapping):
                    return
                item = dict(raw_item)
                path = str(item.get('path') or item.get('saved_text_path') or '').strip()
                if path and path in seen_saved_text_paths:
                    return
                if path:
                    seen_saved_text_paths.add(path)
                saved_text_artifacts.append(item)

            def collect_saved_text_artifacts(source: Mapping[str, Any]) -> None:
                for key in ('saved_text_artifacts', 'text_artifacts', 'artifacts'):
                    values = source.get(key)
                    if not isinstance(values, list):
                        continue
                    for item in values:
                        add_saved_text_artifact(item)
                saved_path = str(source.get('saved_text_path') or '').strip()
                if saved_path:
                    add_saved_text_artifact(
                        {
                            'path': saved_path,
                            'saved_text_path': saved_path,
                            'text_artifact_request': (
                                dict(source.get('text_artifact_request') or {})
                                if isinstance(source.get('text_artifact_request'), Mapping)
                                else {}
                            ),
                            'text_artifact_extension': source.get('text_artifact_extension'),
                            'text_artifact_source_name': source.get('text_artifact_source_name'),
                            'text_artifact_source': source.get('text_artifact_source'),
                        }
                    )

            collect_saved_text_artifacts(infer_result)
            collect_saved_text_artifacts(source_result)
            for ref in branch_refs:
                if not isinstance(ref, Mapping):
                    continue
                branch_id = str(ref.get('branch_id') or '').strip()
                if not branch_id:
                    continue
                request = ref.get('artifact_request') if isinstance(ref.get('artifact_request'), Mapping) else {}
                matched_artifact = next(
                    (
                        artifact
                        for artifact in saved_text_artifacts
                        if self._saved_text_artifact_matches_request(artifact, request)
                    ),
                    {},
                )
                if matched_artifact:
                    request_extension = str(request.get('extension') or '').strip().lower().lstrip('.')
                    request_source_name = str(request.get('source_name') or '').strip()
                    request_target_path = self._text_artifact_target_path_from_mapping(request)
                    repaired_or_clean_artifact = self._canonical_text_artifact_saved_result(
                        {'artifacts': [dict(matched_artifact)]},
                        extension=request_extension,
                        source_name=request_source_name,
                        target_path=request_target_path,
                        allow_deterministic_repair=True,
                    )
                    if repaired_or_clean_artifact:
                        matched_artifact = repaired_or_clean_artifact
                if not matched_artifact:
                    request_target_path = self._text_artifact_target_path_from_mapping(request)
                    target_binding_error = self._text_artifact_target_binding_violation(
                        {'saved_text_artifacts': saved_text_artifacts},
                        target_path=request_target_path,
                        extension=str(request.get('extension') or '').strip().lower().lstrip('.'),
                        source_name=str(request.get('source_name') or '').strip(),
                    )
                    expanded_errors[branch_id] = {
                        **(
                            target_binding_error
                            or {
                                'code': 'COALESCED_TEXT_ARTIFACT_NOT_PERSISTED',
                                'message': (
                                    'Coalesced text artifact materialization returned without the expected saved file.'
                                ),
                            }
                        ),
                        'stage': 'execute_prepared_branch',
                        'retryable': True,
                    }
                    continue
                saved_path = str(matched_artifact.get('path') or matched_artifact.get('saved_text_path') or '').strip()
                artifact_content = self._read_small_text_artifact(saved_path)
                branch_infer_result = dict(infer_result)
                branch_infer_result['saved_text_path'] = saved_path
                branch_infer_result['saved_text_artifacts'] = [dict(matched_artifact)]
                branch_infer_result['text_artifact_request'] = dict(request)
                branch_infer_result['text_artifact_requests'] = [dict(request)]
                branch_infer_result['text_artifact_extension'] = request.get('extension')
                branch_infer_result['text_artifact_source_name'] = request.get('source_name')
                branch_infer_result['text_artifact_source'] = request.get('source')
                branch_infer_result['coalesced_text_artifact_wave'] = True
                branch_infer_result['coalesced_text_artifact_branch_id'] = coalesced_branch_id
                if artifact_content:
                    branch_infer_result['output_text'] = artifact_content
                    branch_infer_result['content'] = artifact_content
                    branch_infer_result['content_payload'] = artifact_content
                branch_plan = dict(plan)
                branch_plan['branch_id'] = branch_id
                branch_plan['phase_id'] = str(ref.get('phase_id') or branch_id).strip() or branch_id
                branch_plan['execution_contract'] = {
                    **member_execution_contract(plan, ref),
                    'artifact_request': dict(request),
                    'text_artifact_extension': request.get('extension'),
                    'text_artifact_source_name': request.get('source_name'),
                    'text_artifact_source': request.get('source'),
                }
                expanded_plans.append(branch_plan)
                expanded_results[branch_id] = {
                    **dict(source_result),
                    'branch_id': branch_id,
                    'phase_id': str(ref.get('phase_id') or branch_id).strip() or branch_id,
                    'execution_contract': dict(branch_plan['execution_contract']),
                    'infer_result': branch_infer_result,
                }
        for branch_id, value in branch_results.items():
            if branch_id not in expanded_results and not any(
                str((plan.get('branch') or {}).get('branch_id') or '').strip() == branch_id
                and isinstance((plan.get('branch') or {}).get('coalesced_text_artifact_branches'), list)
                for plan in prepared_plans
            ):
                expanded_results[branch_id] = value
        for branch_id, value in branch_errors.items():
            if (
                branch_id not in expanded_errors
                and branch_id not in expanded_coalesced_branch_ids
            ):
                expanded_errors[branch_id] = value
        if expanded_results or expanded_errors:
            result['branch_results'] = expanded_results
            result['branch_errors'] = expanded_errors
            result['prepared_branch_plans'] = expanded_plans or prepared_plans
        return result

    @staticmethod
    def materialization_concurrency_history(
        prior_state: Optional[Mapping[str, Any]],
        policy: Optional[Mapping[str, Any]],
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        policy_payload = dict(policy) if isinstance(policy, Mapping) else {}
        if not policy_payload:
            return []
        history = (
            prior_state.get('materialization_concurrency_history')
            if isinstance(prior_state, Mapping)
            and isinstance(prior_state.get('materialization_concurrency_history'), list)
            else []
        )
        entries = [
            dict(item)
            for item in history
            if isinstance(item, Mapping)
        ]
        entries.append(policy_payload)
        return entries[-max(1, int(limit)):]

    @staticmethod
    def _runtime_candidate_snapshot_meta(
        snapshot: list[dict[str, Any]],
        *,
        source: str,
    ) -> dict[str, Any]:
        instance_ids = [
            str(entry.get('instance_id') or '').strip()
            for entry in snapshot
            if isinstance(entry, Mapping) and str(entry.get('instance_id') or '').strip()
        ]
        return {
            'source': source,
            'candidate_count': len(snapshot),
            'candidate_instance_ids': instance_ids,
        }

    def runtime_candidate_snapshot_for_late_fill_wave(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            instances = self.merge_instances_with_runtime_status(
                self.load_running_instances(),
                path=self.runtime_status_path_getter(),
                refresh=True,
            )
        except Exception as exc:  # noqa: BLE001
            logging.info('Late-fill runtime candidate snapshot unavailable: %s', exc)
            return [], self._runtime_candidate_snapshot_meta([], source='late_fill_wave_error')
        snapshot = [
            dict(entry)
            for entry in (instances or [])
            if isinstance(entry, Mapping)
        ]
        return snapshot, self._runtime_candidate_snapshot_meta(snapshot, source='late_fill_wave')

    def runtime_scheduling_context_for_branch(
        self,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
        source_route_payload: Optional[Mapping[str, Any]],
        artifact_gap: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        late_fill_state = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), Mapping) else {}
        active_branches = late_fill_state.get('active_branches') if isinstance(late_fill_state.get('active_branches'), list) else []
        active_capabilities = self.normalize_capability_list(
            [self.branch_capability(item) for item in active_branches if isinstance(item, Mapping) and self.branch_capability(item)]
        )
        context = self._runtime_scheduling_context(artifact_gap, source_route_payload)
        branch_context = self._branch_scheduling_context(branch, artifact_gap)
        for key in ('repair_scope', 'resource_class', 'dependency_policy'):
            value = branch_context.get(key)
            if value not in (None, '', [], {}):
                context[key] = value
        if active_capabilities:
            context['active_capabilities'] = active_capabilities
        if any(capability == self.capability_image_generation for capability in active_capabilities):
            context['active_image_generation'] = True
        if source_route_payload and isinstance(source_route_payload, Mapping):
            route_instance = source_route_payload.get('instance') if isinstance(source_route_payload.get('instance'), Mapping) else {}
            if route_instance:
                if self._late_fill_instance_is_lightweight_chat_candidate(route_instance):
                    context.setdefault('selected_chat_model_class', 'lightweight')
                context.setdefault('selected_chat_model', route_instance.get('model') or route_instance.get('model_name'))
                context.setdefault('selected_chat_instance_id', route_instance.get('instance_id'))
        if self._branch_allows_gpu_heavy_concurrency(branch, artifact_gap):
            context['allow_gpu_heavy_concurrency'] = True
        return {key: value for key, value in context.items() if value not in (None, '', [], {})}

    def response_has_required_artifact_closure_work(self, current_payload: Mapping[str, Any]) -> bool:
        late_fill_state = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), Mapping) else {}
        for key in ('pending_branches', 'active_branches'):
            branches = late_fill_state.get(key) if isinstance(late_fill_state.get(key), list) else []
            for branch in branches:
                if isinstance(branch, Mapping) and self._branch_is_required_artifact(branch):
                    return True
        return False

    @staticmethod
    def _late_fill_payload_has_open_work(payload: Mapping[str, Any]) -> bool:
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        if not late_fill:
            return False
        for key in ('pending_branches', 'active_branches'):
            if isinstance(late_fill.get(key), list) and late_fill.get(key):
                return True
        if isinstance(late_fill.get('pending_capabilities'), list) and late_fill.get('pending_capabilities'):
            return True
        return str(late_fill.get('status') or '').strip().lower() in {'pending', 'running', 'active'}

    @classmethod
    def _prefer_explicit_late_fill_seed_payload(
        cls,
        explicit_payload: Mapping[str, Any],
        recovered_payload: Mapping[str, Any],
    ) -> bool:
        if not isinstance(explicit_payload, Mapping) or not explicit_payload:
            return False
        if not isinstance(recovered_payload, Mapping) or not recovered_payload:
            return True
        if not cls._late_fill_payload_has_open_work(explicit_payload):
            return False
        return not cls._late_fill_payload_has_open_work(recovered_payload)

    def build_execution_contract(
        self,
        branch: Mapping[str, Any],
        artifact_gap: Optional[Mapping[str, Any]] = None,
        *,
        capability: Optional[str] = None,
    ) -> dict[str, Any]:
        branch_payload = branch if isinstance(branch, Mapping) else {}
        gap_payload = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        normalized_capability = self.normalize_capability(
            capability
            or branch_payload.get('capability')
            or gap_payload.get('capability')
            or gap_payload.get('expected_capability')
        )
        branch_id = _contract_text(
            _first_contract_value(
                branch_payload.get('branch_id'),
                gap_payload.get('branch_id'),
                branch_payload.get('phase_id'),
                gap_payload.get('phase_id'),
            )
        )
        phase_id = _contract_text(
            _first_contract_value(
                branch_payload.get('phase_id'),
                gap_payload.get('phase_id'),
                branch_payload.get('branch_id'),
                gap_payload.get('branch_id'),
            )
        )
        task_id = _contract_text(
            _first_contract_value(
                branch_payload.get('task_id'),
                branch_payload.get('workload_task_id'),
                gap_payload.get('task_id'),
                gap_payload.get('workload_task_id'),
                f'task-{phase_id}' if phase_id else None,
            )
        )
        obligation_id = _contract_text(
            _first_contract_value(
                branch_payload.get('obligation_id'),
                gap_payload.get('obligation_id'),
                f'obligation-{phase_id}' if phase_id else None,
                f'obligation-{branch_id}' if branch_id else None,
            )
        )
        output_type = _contract_text(
            _first_contract_value(
                branch_payload.get('output_type'),
                gap_payload.get('output_type'),
                self.artifact_type_for_capability(normalized_capability),
            )
        ).lower()
        depends_on = [
            _contract_text(item)
            for item in _clean_contract_value(
                _first_contract_value(
                    branch_payload.get('depends_on'),
                    gap_payload.get('depends_on'),
                )
                or []
            )
            if _contract_text(item)
        ]
        input_refs = _clean_contract_value(
            _first_contract_value(
                branch_payload.get('input_refs'),
                gap_payload.get('input_refs'),
            )
            or []
        )
        if not input_refs and depends_on:
            input_refs = [
                {
                    'kind': 'phase_output',
                    'phase_id': dependency_id,
                    'role': 'dependency',
                }
                for dependency_id in depends_on
            ]

        raw_output_contract = _first_contract_value(
            branch_payload.get('output_contract'),
            gap_payload.get('output_contract'),
        )
        output_contract = (
            _clean_contract_value(raw_output_contract)
            if isinstance(raw_output_contract, Mapping)
            else {}
        )
        if output_type:
            output_contract.setdefault('output_type', output_type)
        required_value = _first_contract_value(branch_payload.get('required'), gap_payload.get('required'))
        if required_value not in (None, '', [], {}):
            output_contract.setdefault('required', required_value)
        elif output_type:
            output_contract.setdefault('required', True)
        requires_artifact = _first_contract_value(
            branch_payload.get('requires_artifact'),
            gap_payload.get('requires_artifact'),
        )
        if requires_artifact not in (None, '', [], {}):
            output_contract.setdefault('requires_artifact', requires_artifact)
        if output_contract and 'fulfillment_policy' not in output_contract:
            output_contract['fulfillment_policy'] = (
                'materialized_artifact_required'
                if bool(requires_artifact)
                else 'capability_output_required'
            )

        contract: dict[str, Any] = {
            'kind': 'ollmo.execution_contract',
            'branch_id': branch_id or None,
            'phase_id': phase_id or None,
            'capability': normalized_capability or None,
            'output_type': output_type or None,
            'workload_task_ref': {
                'task_id': task_id or None,
                'phase_id': phase_id or None,
                'branch_id': branch_id or None,
            },
            'output_obligation_ref': {
                'obligation_id': obligation_id or None,
                'phase_id': phase_id or None,
                'branch_id': branch_id or None,
                'output_type': output_type or None,
            },
            'output_contract': output_contract,
            'depends_on': depends_on,
            'input_refs': input_refs,
        }
        for key in (
            'queue_index',
            'source',
            'role',
            'stage_direction',
            'content_payload_source',
            'artifact_prompt_source',
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_policy',
            'selection_reason',
            'lang_code',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
            'structured_output_contract',
            'branch_contract_error',
            'audio_variant_contract_conflicting_fields',
            'candidate_id',
            'contract_state',
            'contract_status',
            'obligation_state',
            'promotion_policy',
            'promotion_reason',
            'promotion_source',
            'promoted_from_candidate_id',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'repair_contract',
            'repair_contract_id',
            'repair_contract_status',
            'repair_execution_policy',
            'repair_promotion_source',
            'contract_state',
            'promotion_source',
            'reconsideration_rebuild',
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
            'surface_state',
        ):
            value = _first_contract_value(branch_payload.get(key), gap_payload.get(key))
            if value not in (None, '', [], {}):
                contract[key] = _clean_contract_value(value)

        existing_contract = gap_payload.get('execution_contract')
        if isinstance(existing_contract, Mapping):
            for key, value in _clean_contract_value(existing_contract).items():
                if contract.get(key) in (None, '', [], {}):
                    contract[key] = value
        return _clean_contract_value(contract)

    def attach_execution_contract_to_gap(
        self,
        branch: Mapping[str, Any],
        artifact_gap: dict[str, Any],
        *,
        capability: Optional[str] = None,
    ) -> dict[str, Any]:
        updated = dict(artifact_gap or {})
        contract = self.build_execution_contract(branch, updated, capability=capability)
        if not contract:
            return updated
        updated['execution_contract'] = contract
        workload_task_ref = contract.get('workload_task_ref') if isinstance(contract.get('workload_task_ref'), Mapping) else {}
        output_obligation_ref = (
            contract.get('output_obligation_ref')
            if isinstance(contract.get('output_obligation_ref'), Mapping)
            else {}
        )
        for key in ('branch_id', 'phase_id', 'capability', 'output_type'):
            value = contract.get(key)
            if value not in (None, '', [], {}) and updated.get(key) in (None, '', [], {}):
                updated[key] = value
        task_id = workload_task_ref.get('task_id') if isinstance(workload_task_ref, Mapping) else None
        if task_id not in (None, '', [], {}) and updated.get('task_id') in (None, '', [], {}):
            updated['task_id'] = task_id
            updated['workload_task_id'] = task_id
        obligation_id = (
            output_obligation_ref.get('obligation_id')
            if isinstance(output_obligation_ref, Mapping)
            else None
        )
        if obligation_id not in (None, '', [], {}) and updated.get('obligation_id') in (None, '', [], {}):
            updated['obligation_id'] = obligation_id
        if workload_task_ref:
            updated.setdefault('workload_task_ref', dict(workload_task_ref))
        if output_obligation_ref:
            updated.setdefault('output_obligation_ref', dict(output_obligation_ref))
        return updated

    def normalize_late_fill_error_payload(
        self,
        error: Any,
        *,
        default_stage: str = 'execute_prepared_branch',
    ) -> dict[str, Any]:
        raw = error if isinstance(error, Mapping) else {}
        message = str(raw.get('message') if raw else error or '').strip()
        if not message:
            message = 'Late fill request failed.'
        stage = str(raw.get('stage') or default_stage or '').strip() or default_stage
        code = str(raw.get('code') or '').strip().upper()
        if not code:
            lowered = message.lower()
            if 'timeout' in lowered or 'timed out' in lowered:
                code = 'BACKEND_TIMEOUT'
            elif any(token in lowered for token in ('unavailable', 'connection refused', 'offline')):
                code = 'INSTANCE_UNAVAILABLE'
            else:
                code = 'BACKEND_ERROR' if stage == 'execute_prepared_branch' else 'PREPARE_FAILED'
        payload: dict[str, Any] = {
            'code': code,
            'message': message,
            'stage': stage,
        }
        retryable = raw.get('retryable') if raw else None
        if isinstance(retryable, bool):
            payload['retryable'] = retryable
        else:
            payload['retryable'] = code not in {'CAPABILITY_UNSUPPORTED', 'CONTROL_VALIDATION_FAILED', 'PREPARE_FAILED'}
        status_code = raw.get('status_code') if raw else None
        if status_code is None and raw:
            status_code = raw.get('statusCode')
        if status_code not in (None, ''):
            try:
                payload['status_code'] = int(status_code)
            except (TypeError, ValueError):
                pass
        exception_type = str(raw.get('exception_type') or raw.get('exceptionType') or '').strip() if raw else ''
        if exception_type:
            payload['exception_type'] = exception_type
        for key in (
            'reason_code',
            'defect_code',
            'defect_codes',
            'repair_action',
            'recovery_action',
            'suggested_action',
            'materialization_blocked',
            'blocked_scope',
            'blocked_prerequisite',
            'repair_work_available',
            'repair_work_policy',
            'needs_external_input',
            'failed_dependency_ids',
            'semantic_evidence',
            'audio_integrity_evidence',
            'tts_generation_budget',
            'tts_sampling_profile',
            'diagnostic_artifact',
            'external_execution',
            'coalesced_text_artifact_recovery',
        ):
            value = raw.get(key) if raw else None
            if value not in (None, '', [], {}):
                payload[key] = value
        return payload

    def late_fill_failure_attempt(
        self,
        branch: Mapping[str, Any],
        error: Mapping[str, Any],
        plan: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        plan_payload = plan if isinstance(plan, Mapping) else {}
        route_info = plan_payload.get('route_info') if isinstance(plan_payload.get('route_info'), Mapping) else {}
        instance = plan_payload.get('instance') if isinstance(plan_payload.get('instance'), Mapping) else {}
        capability = self.branch_capability(branch) or self.normalize_capability(plan_payload.get('capability'))
        payload: dict[str, Any] = {
            'stage': str(error.get('stage') or '').strip() or 'execute_prepared_branch',
            'capability': capability,
        }
        instance_id = str(
            route_info.get('instance_id')
            or instance.get('instance_id')
            or ''
        ).strip()
        if instance_id:
            payload['instance_id'] = instance_id
        backend = str(instance.get('backend') or route_info.get('backend') or '').strip()
        if backend:
            payload['backend'] = backend
        model = str(instance.get('model') or route_info.get('model') or '').strip()
        if model:
            payload['model'] = model
        for source_key, target_key in (
            ('route_source', 'route_source'),
            ('route_reason', 'route_reason'),
        ):
            value = str(route_info.get(source_key) or '').strip()
            if value:
                payload[target_key] = value
        route_runtime = (
            route_info.get('route_runtime')
            if isinstance(route_info.get('route_runtime'), Mapping)
            else {}
        )
        selection_policy = str(
            route_runtime.get('selection_policy') or ''
        ).strip()
        if selection_policy:
            payload['selection_policy'] = selection_policy
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, '', [], {})
        }

    def late_fill_recovery_context(
        self,
        *,
        error: Mapping[str, Any],
        attempt: Mapping[str, Any],
    ) -> dict[str, Any]:
        retryable = bool(error.get('retryable'))
        instance_id = str(attempt.get('instance_id') or '').strip()
        code = str(error.get('code') or '').strip().upper()
        message = str(error.get('message') or '').strip()
        explicit_action = normalize_recovery_suggested_action(
            error.get('repair_action') or error.get('recovery_action') or error.get('suggested_action'),
            default='',
        )
        dependency_input_missing = bool(
            code == 'DEPENDENCY_CHAIN_REPAIR_REQUIRED'
            or _DEPENDENCY_INPUT_MISSING_RE.search(message)
        )
        branch_contract_repair = bool(
            code in {'BRANCH_CONTRACT_REPAIR_REQUIRED', 'CONTROL_VALIDATION_FAILED', 'PREPARE_FAILED'}
            or _BRANCH_CONTRACT_REPAIR_RE.search(message)
        )
        rebuild_promoted_obligations = bool(
            code == 'REBUILD_FROM_PROMOTED_OBLIGATIONS_REQUIRED'
            or 'promoted obligation' in message.lower()
            or 'rebuild_from_promoted_obligations' in message.lower()
        )
        if rebuild_promoted_obligations:
            suggested_action = RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS
        elif branch_contract_repair:
            suggested_action = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
        elif dependency_input_missing and explicit_action == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE:
            suggested_action = RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
        elif dependency_input_missing:
            suggested_action = RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN
        elif not retryable:
            suggested_action = RECOVERY_ACTION_MANUAL_REVIEW
        elif code == 'NO_COMPATIBLE_INSTANCE':
            suggested_action = RECOVERY_ACTION_START_COMPATIBLE_INSTANCE
        elif instance_id:
            suggested_action = RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE
        else:
            suggested_action = RECOVERY_ACTION_RETRY_SAME_BRANCH
        payload: dict[str, Any] = {
            'can_retry': retryable
            and not dependency_input_missing
            and not branch_contract_repair
            and not rebuild_promoted_obligations,
            'retry_scope': (
                'promoted_obligations'
                if rebuild_promoted_obligations
                else
                'branch_contract'
                if branch_contract_repair
                else 'dependency_chain'
                if dependency_input_missing
                else 'same_branch'
            ),
            'suggested_action': suggested_action,
            'preserve_intent': True,
            'error_code': code,
        }
        reason_code = str(error.get('reason_code') or '').strip()
        if reason_code:
            payload['reason_code'] = reason_code
        defect_codes = [
            str(item).strip()
            for item in (error.get('defect_codes') or [])
            if str(item).strip()
        ] if isinstance(error.get('defect_codes'), list) else []
        if defect_codes:
            payload['defect_codes'] = defect_codes
        audio_integrity_evidence = (
            error.get('audio_integrity_evidence')
            if isinstance(error.get('audio_integrity_evidence'), Mapping)
            else {}
        )
        if audio_integrity_evidence:
            payload['audio_integrity_evidence'] = dict(
                audio_integrity_evidence
            )
        if branch_contract_repair:
            payload['repair_required'] = True
            payload['blocked_by_branch_contract'] = True
        if rebuild_promoted_obligations:
            payload['repair_required'] = True
            payload['blocked_by_underplanned_promoted_obligations'] = True
        if dependency_input_missing:
            payload['repair_required'] = True
            payload['blocked_by_dependency_input'] = True
        for key in (
            'materialization_blocked',
            'blocked_scope',
            'blocked_prerequisite',
            'repair_work_available',
            'repair_work_policy',
            'needs_external_input',
        ):
            value = error.get(key)
            if value not in (None, '', [], {}):
                payload[key] = value
        if instance_id and retryable:
            payload['exclude_instance_ids'] = [instance_id]
        return payload

    def late_fill_recovery_state(
        self,
        branch: Mapping[str, Any],
        *,
        recovery_context: Mapping[str, Any],
        attempt: Optional[Mapping[str, Any]] = None,
        status: str = 'candidate',
        trigger: str = 'late_fill_failure',
    ) -> dict[str, Any]:
        branch_id = self.branch_id(branch)
        capability = self.branch_capability(branch)
        attempt_payload = attempt if isinstance(attempt, Mapping) else {}
        state_status = str(status or '').strip().lower() or 'candidate'
        auto_execute_recovery = self.auto_executable_repair_recovery_allowed(
            branch,
            recovery_context=recovery_context,
        )
        payload: dict[str, Any] = {
            'kind': 'ollmo.late_fill_recovery_state',
            'status': state_status,
            'trigger': str(trigger or '').strip() or 'late_fill_failure',
            'branch_id': branch_id,
            'capability': capability,
            'promotion_required': state_status == 'candidate' and not auto_execute_recovery,
            'auto_execute': auto_execute_recovery,
            'preserve_intent': recovery_context.get('preserve_intent')
            if isinstance(recovery_context.get('preserve_intent'), bool)
            else True,
        }
        for key in ('retry_scope', 'suggested_action'):
            value = str(recovery_context.get(key) or '').strip()
            if value:
                payload[key] = value
        for key in (
            'repair_required',
            'blocked_by_dependency_input',
            'blocked_by_branch_contract',
            'blocked_by_underplanned_promoted_obligations',
            'materialization_blocked',
            'repair_work_available',
            'needs_external_input',
        ):
            value = recovery_context.get(key)
            if isinstance(value, bool):
                payload[key] = value
        for key in ('blocked_scope', 'blocked_prerequisite', 'repair_work_policy'):
            value = str(recovery_context.get(key) or '').strip()
            if value:
                payload[key] = value
        instance_id = str(attempt_payload.get('instance_id') or '').strip()
        if instance_id:
            payload['failed_instance_id'] = instance_id
        exclude_instance_ids = [
            str(item).strip()
            for item in (recovery_context.get('exclude_instance_ids') or [])
            if str(item).strip()
        ] if isinstance(recovery_context.get('exclude_instance_ids'), list) else []
        if exclude_instance_ids:
            payload['exclude_instance_ids'] = exclude_instance_ids
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, '', [], {})
        }

    @staticmethod
    def _repair_bool_from_branch_or_contract(
        branch: Mapping[str, Any],
        key: str,
    ) -> Optional[bool]:
        value = branch.get(key)
        if isinstance(value, bool):
            return value
        contract = branch.get('repair_contract') if isinstance(branch.get('repair_contract'), Mapping) else {}
        value = contract.get(key)
        if isinstance(value, bool):
            return value
        return None

    @staticmethod
    def _repair_text_from_branch_or_contract(branch: Mapping[str, Any], key: str) -> str:
        value = str(branch.get(key) or '').strip()
        if value:
            return value
        contract = branch.get('repair_contract') if isinstance(branch.get('repair_contract'), Mapping) else {}
        return str(contract.get(key) or '').strip()

    def auto_executable_repair_max_attempts(self, branch: Mapping[str, Any]) -> int:
        contract = branch.get('repair_contract') if isinstance(branch.get('repair_contract'), Mapping) else {}
        attempt_limit = (
            _TTS_AUTO_RECOVERY_MAX_ATTEMPTS
            if self._branch_is_required_tts_materialization(branch)
            else _AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS_LIMIT
        )
        for key in (
            'auto_executable_repair_max_attempts',
            'repair_auto_execute_max_attempts',
            'max_auto_execute_attempts',
        ):
            for source in (branch, contract):
                try:
                    value = int(source.get(key))
                except (AttributeError, TypeError, ValueError):
                    continue
                if value > 0:
                    return min(value, attempt_limit)
        try:
            env_value = int(os.environ.get(_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS_ENV, ''))
        except (TypeError, ValueError):
            env_value = 0
        if env_value > 0:
            return min(env_value, attempt_limit)
        return (
            _TTS_AUTO_RECOVERY_MAX_ATTEMPTS
            if self._branch_is_required_tts_materialization(branch)
            else _AUTO_EXECUTABLE_REPAIR_DEFAULT_MAX_ATTEMPTS
        )

    def auto_executable_repair_recovery_allowed(
        self,
        branch: Mapping[str, Any],
        *,
        recovery_context: Mapping[str, Any],
    ) -> bool:
        if not isinstance(branch, Mapping) or not isinstance(recovery_context, Mapping):
            return False
        required_text_artifact = self._branch_is_required_text_artifact(branch)
        required_image_materialization = self._branch_is_required_image_materialization(branch)
        required_tts_recovery = (
            self._required_tts_auto_recovery_allowed(
                branch,
                recovery_context,
            )
        )
        if (
            not required_text_artifact
            and not required_image_materialization
            and not required_tts_recovery
        ):
            return False
        try:
            retry_count = int(branch.get('auto_executable_repair_retry_count') or 0)
        except (TypeError, ValueError):
            retry_count = 0
        max_attempts = self.auto_executable_repair_max_attempts(branch)
        max_retries = max(0, max_attempts - 1)
        if retry_count >= max_retries:
            return False
        execution_policy = (
            self._repair_text_from_branch_or_contract(branch, 'repair_execution_policy')
            or self._repair_text_from_branch_or_contract(branch, 'execution_policy')
        )
        auto_requested = (
            self._repair_bool_from_branch_or_contract(branch, 'auto_execute') is True
            or execution_policy == 'schedule_late_fill_branch'
        )
        if required_text_artifact:
            auto_requested = True
        if required_image_materialization and not required_text_artifact:
            auto_requested = True
        if required_tts_recovery:
            auto_requested = True
        if not auto_requested:
            return False
        if self._repair_bool_from_branch_or_contract(branch, 'repair_work_available') is False:
            return False
        for source in (branch, recovery_context):
            if source.get('needs_external_input') is True:
                return False
            if (
                source.get('materialization_blocked') is True
                and not required_tts_recovery
            ):
                return False
        action = normalize_recovery_suggested_action(
            recovery_context.get('suggested_action') or branch.get('recovery_action') or branch.get('repair_action'),
            default='',
        )
        if action not in _AUTO_EXECUTABLE_REPAIR_RETRY_ACTIONS and not (
            required_tts_recovery
            and action == RECOVERY_ACTION_START_COMPATIBLE_INSTANCE
        ):
            return False
        retry_scope = str(recovery_context.get('retry_scope') or '').strip()
        return recovery_context.get('can_retry') is True and retry_scope in {'', 'same_branch'}

    def build_auto_executable_repair_retry_branch(
        self,
        branch: Mapping[str, Any],
        *,
        recovery_context: Mapping[str, Any],
        recovery_state: Mapping[str, Any],
        attempt: Mapping[str, Any],
        trigger: str,
    ) -> Optional[dict[str, Any]]:
        if not self.auto_executable_repair_recovery_allowed(
            branch,
            recovery_context=recovery_context,
        ):
            return None
        branch_id = self.branch_id(branch)
        capability = self.branch_capability(branch)
        if not branch_id or not capability:
            return None
        try:
            retry_count = int(branch.get('auto_executable_repair_retry_count') or 0)
        except (TypeError, ValueError):
            retry_count = 0
        tts_auto_recovery = (
            self._required_tts_auto_recovery_allowed(
                branch,
                recovery_context,
            )
        )
        resolved_trigger = (
            _TTS_AUTO_RECOVERY_TRIGGER
            if tts_auto_recovery
            else str(trigger or '').strip() or 'auto_executable_repair_retry'
        )
        max_attempts = self.auto_executable_repair_max_attempts(branch)
        retry_branch = {
            key: value
            for key, value in dict(branch).items()
            if key not in {'error', 'attempt', 'recovery_context', 'recovery_state', 'recovery_attempt'}
        }
        excluded_instance_ids: list[str] = []
        for values in (
            branch.get('excluded_instance_ids'),
            recovery_context.get('exclude_instance_ids'),
            recovery_state.get('exclude_instance_ids'),
        ):
            if not isinstance(values, list):
                continue
            for item in values:
                token = str(item or '').strip()
                if token and token not in excluded_instance_ids:
                    excluded_instance_ids.append(token)
        failed_instance_id = str(
            attempt.get('instance_id')
            or recovery_state.get('failed_instance_id')
            or branch.get('failed_instance_id')
            or ''
        ).strip()
        if failed_instance_id and failed_instance_id not in excluded_instance_ids:
            excluded_instance_ids.append(failed_instance_id)
        action = normalize_recovery_suggested_action(
            recovery_context.get('suggested_action') or recovery_state.get('suggested_action'),
            default=RECOVERY_ACTION_RETRY_SAME_BRANCH,
        )
        retry_recovery_state = dict(recovery_state)
        retry_recovery_state.update(
            {
                'kind': 'ollmo.late_fill_recovery_state',
                'status': 'attempting',
                'trigger': resolved_trigger,
                'branch_id': branch_id,
                'capability': capability,
                'promotion_required': False,
                'auto_execute': True,
                'preserve_intent': True,
                'retry_scope': 'same_branch',
                'suggested_action': action,
            }
        )
        if failed_instance_id:
            retry_recovery_state['failed_instance_id'] = failed_instance_id
        if excluded_instance_ids:
            retry_recovery_state['exclude_instance_ids'] = excluded_instance_ids
        recovery_attempt = {
            'kind': 'ollmo.late_fill_recovery_attempt',
            'trigger': resolved_trigger,
            'branch_id': branch_id,
            'capability': capability,
            'preserve_intent': True,
            'auto_execute': True,
            'failed_instance_id': failed_instance_id or None,
            'excluded_instance_ids': excluded_instance_ids,
        }
        if tts_auto_recovery:
            prior_error_code = str(
                recovery_context.get('error_code') or ''
            ).strip().upper()
            prior_reason_code = str(
                recovery_context.get('reason_code') or ''
            ).strip()
            prior_defect_codes = [
                str(item).strip()
                for item in (recovery_context.get('defect_codes') or [])
                if str(item).strip()
            ] if isinstance(recovery_context.get('defect_codes'), list) else []
            prior_integrity = (
                recovery_context.get('audio_integrity_evidence')
                if isinstance(
                    recovery_context.get('audio_integrity_evidence'),
                    Mapping,
                )
                else {}
            )
            retry_recovery_state.update(
                {
                    'recovery_policy_id': (
                        _TTS_AUTO_RECOVERY_POLICY_ID
                    ),
                    'attempt_number': retry_count + 2,
                    'maximum_attempts': max_attempts,
                    'prior_error_code': prior_error_code,
                    'prior_reason_code': prior_reason_code or None,
                    'prior_defect_codes': prior_defect_codes,
                }
            )
            recovery_attempt.update(
                {
                    'recovery_policy_id': (
                        _TTS_AUTO_RECOVERY_POLICY_ID
                    ),
                    'attempt_number': retry_count + 2,
                    'maximum_attempts': max_attempts,
                    'prior_error_code': prior_error_code,
                    'prior_reason_code': prior_reason_code or None,
                    'prior_defect_codes': prior_defect_codes,
                    'prior_audio_integrity_evidence': (
                        dict(prior_integrity) if prior_integrity else None
                    ),
                }
            )
        retry_branch.update(
            {
                'status': 'pending',
                'auto_execute': True,
                'repair_work_available': True,
                'needs_external_input': False,
                'repair_action': action,
                'recovery_action': action,
                'suggested_action': action,
                'auto_executable_repair_retry_count': retry_count + 1,
                'auto_executable_repair_max_attempts': max_attempts,
                'recovery_context': dict(recovery_context),
                'recovery_state': {
                    key: value
                    for key, value in retry_recovery_state.items()
                    if value not in (None, '', [], {})
                },
                'recovery_attempt': {
                    key: value
                    for key, value in recovery_attempt.items()
                    if value not in (None, '', [], {})
                },
            }
        )
        if failed_instance_id:
            retry_branch['failed_instance_id'] = failed_instance_id
        if excluded_instance_ids:
            retry_branch['excluded_instance_ids'] = excluded_instance_ids
        if tts_auto_recovery:
            retry_branch['recovery_policy_id'] = (
                _TTS_AUTO_RECOVERY_POLICY_ID
            )
        return retry_branch

    def late_fill_branch_control_records(self, current_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        payload = current_payload if isinstance(current_payload, Mapping) else {}
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        controls_by_branch_id: dict[str, dict[str, Any]] = {}

        def remember(raw_record: Any, *, source: str) -> None:
            if not isinstance(raw_record, Mapping):
                return
            branch_id = self.branch_id(raw_record)
            if not branch_id:
                return
            status = _normalized_terminal_status(
                raw_record.get('status')
                or raw_record.get('action')
                or raw_record.get('control_action')
                or raw_record.get('execution_gate_status')
            )
            if not status:
                return
            controls_by_branch_id[branch_id] = {
                'branch_id': branch_id,
                'status': status,
                'action': status,
                'source': source,
                'reason': str(
                    raw_record.get('reason')
                    or raw_record.get('cancel_reason')
                    or raw_record.get('waiver_reason')
                    or raw_record.get('supersession_reason')
                    or raw_record.get('execution_gate_reason')
                    or ''
                ).strip() or None,
                'authority': str(raw_record.get('authority') or raw_record.get('cancelled_by') or 'runtime_contract').strip(),
                'created_at': str(raw_record.get('created_at') or raw_record.get('cancelled_at') or '').strip() or None,
            }

        for key, status in (
            ('cancelled_branches', 'cancelled'),
            ('waived_branches', 'waived'),
            ('superseded_branches', 'superseded'),
        ):
            for raw_branch in late_fill.get(key) or []:
                if isinstance(raw_branch, Mapping):
                    record = dict(raw_branch)
                    record.setdefault('status', status)
                    remember(record, source=key)
        for raw_control in late_fill.get('branch_controls') or []:
            remember(raw_control, source='branch_controls')
        return controls_by_branch_id

    def semantic_execution_gate_decision(
        self,
        branch: Mapping[str, Any],
        current_payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        branch_payload = branch if isinstance(branch, Mapping) else {}
        branch_id = self.branch_id(branch_payload)
        capability = self.branch_capability(branch_payload)
        status = _normalized_terminal_status(branch_payload.get('status'))
        reason = str(branch_payload.get('execution_gate_reason') or '').strip()
        source = 'branch_status'
        control = None
        if not status and current_payload is not None:
            control = self.late_fill_branch_control_records(current_payload).get(branch_id)
            if control:
                status = _normalized_terminal_status(control.get('status') or control.get('action'))
                reason = str(control.get('reason') or '').strip()
                source = str(control.get('source') or 'branch_control').strip() or 'branch_control'
        if not status and self.parse_bool(branch_payload.get('cancel_requested'), default=False):
            status = 'cancelled'
            reason = str(branch_payload.get('cancel_reason') or '').strip()
            source = 'cancel_requested'
        if status in _SEMANTIC_EXECUTION_TERMINAL_STATUSES:
            return {
                'kind': 'ollmo.semantic_execution_gate',
                'scope': 'branch',
                'status': status,
                'action': 'skip',
                'branch_id': branch_id or None,
                'phase_id': str(branch_payload.get('phase_id') or branch_id or '').strip() or None,
                'capability': capability or None,
                'authority': str((control or {}).get('authority') or 'runtime_contract').strip() or 'runtime_contract',
                'reason': reason or f'branch marked {status} before execution',
                'source': source,
            }
        return {
            'kind': 'ollmo.semantic_execution_gate',
            'scope': 'branch',
            'status': 'allowed',
            'action': 'execute',
            'branch_id': branch_id or None,
            'phase_id': str(branch_payload.get('phase_id') or branch_id or '').strip() or None,
            'capability': capability or None,
            'authority': 'runtime_contract',
            'reason': 'branch remains relevant and executable',
            'source': 'runtime_contract',
        }

    def branch_record_with_execution_gate(
        self,
        branch: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        branch_payload = dict(branch or {})
        status = _normalized_terminal_status(decision.get('status')) or 'cancelled'
        branch_payload['status'] = status
        branch_payload['execution_gate'] = dict(decision)
        reason = str(decision.get('reason') or '').strip()
        now_iso = self.response_registry_now_iso()
        if status == 'cancelled':
            branch_payload['cancel_requested'] = True
            branch_payload['cancel_reason'] = reason or 'branch cancelled before completion'
            branch_payload['cancelled_by'] = str(decision.get('authority') or 'runtime_contract').strip() or 'runtime_contract'
            branch_payload['cancelled_at'] = now_iso
        elif status == 'waived':
            branch_payload['waiver_reason'] = reason or 'branch waived before execution'
        elif status == 'superseded':
            branch_payload['supersession_reason'] = reason or 'branch superseded before execution'
        return {
            key: value
            for key, value in branch_payload.items()
            if value not in (None, '', [], {})
        }

    def repair_action_for_branch(self, branch: Mapping[str, Any]) -> Optional[str]:
        for source in (
            branch,
            branch.get('recovery_state') if isinstance(branch.get('recovery_state'), Mapping) else {},
            branch.get('recovery_context') if isinstance(branch.get('recovery_context'), Mapping) else {},
        ):
            if not isinstance(source, Mapping):
                continue
            for key in ('repair_action', 'recovery_action', 'suggested_action', 'suggestedAction'):
                action = str(source.get(key) or '').strip()
                if action:
                    return normalize_recovery_suggested_action(action)
        return None

    @staticmethod
    def _branch_has_inline_dependency_evidence(branch: Mapping[str, Any]) -> bool:
        if str(branch.get('file_path') or '').strip():
            return True
        for key in ('input_artifacts', 'reference_artifacts'):
            value = branch.get(key)
            if isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
                return True
        source = str(branch.get('content_payload_source') or '').strip()
        if source in {'late_fill_dependency_artifacts', 'prior_artifact_result'}:
            return bool(str(branch.get('content_payload') or '').strip())
        if source.startswith('late_fill_result'):
            return bool(str(branch.get('content_payload') or '').strip())
        return False

    @staticmethod
    def _current_payload_requires_materialization_repair(current_payload: Mapping[str, Any]) -> bool:
        runtime = (
            current_payload.get('runtime')
            if isinstance(current_payload.get('runtime'), Mapping)
            else {}
        )
        truth_guard = (
            runtime.get('truth_guard')
            if isinstance(runtime.get('truth_guard'), Mapping)
            else {}
        )
        return str(truth_guard.get('status') or '').strip().lower() in {
            'clarification_required',
            'repair_required',
        }

    def _branch_has_current_turn_direct_media_payload(
        self,
        branch: Mapping[str, Any],
    ) -> bool:
        dependencies = {
            str(item or '').strip()
            for item in (branch.get('depends_on') or [])
            if str(item or '').strip()
        }
        if dependencies != {'phase-1'}:
            return False
        capability = self.branch_capability(branch)
        if capability == self.capability_text_to_speech:
            return bool(
                str(branch.get('content_payload') or '').strip()
                and str(branch.get('content_payload_source') or '').strip()
                == 'current_turn_direct_spoken_clause'
            )
        if capability == self.capability_image_generation:
            return bool(
                str(branch.get('artifact_prompt') or '').strip()
                and str(branch.get('artifact_prompt_source') or '').strip()
                == 'current_turn_direct_image_clause'
            )
        return False

    def _branch_depends_on_rejected_current_payload(
        self,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
    ) -> bool:
        capability = self.branch_capability(branch)
        if capability == 'chat':
            return False
        if self._branch_has_current_turn_direct_media_payload(branch):
            return False
        depends_on = {
            str(item or '').strip()
            for item in (branch.get('depends_on') or [])
            if str(item or '').strip()
        }
        source = str(branch.get('content_payload_source') or '').strip()
        consumes_current_phase = (
            'phase-1' in depends_on
            or source == 'current_phase_output'
            or not depends_on
        )
        if not consumes_current_phase:
            return False
        if self._current_payload_requires_materialization_repair(current_payload):
            return True
        return control_json_envelope_suspected(
            str(current_payload.get('output_text') or '').strip()
        )

    def repair_branch_execution_error(
        self,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
        artifact_gap: Optional[Mapping[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if self._branch_depends_on_rejected_current_payload(branch, current_payload=current_payload):
            runtime = (
                current_payload.get('runtime')
                if isinstance(current_payload.get('runtime'), Mapping)
                else {}
            )
            truth_guard = (
                runtime.get('truth_guard')
                if isinstance(runtime.get('truth_guard'), Mapping)
                else {}
            )
            truth_guard_status = str(truth_guard.get('status') or '').strip().lower()
            control_payload_rejected = (
                truth_guard_status == 'repair_required'
                or control_json_envelope_suspected(
                    str(current_payload.get('output_text') or '').strip()
                )
            )
            return {
                'code': (
                    'UPSTREAM_CONTROL_PAYLOAD_REJECTED'
                    if control_payload_rejected
                    else 'UPSTREAM_CLARIFICATION_REQUIRED'
                ),
                'message': (
                    'Branch contract repair required before this branch can execute: '
                    + (
                        'the upstream result is an internal control envelope, not speakable or materializable payload.'
                        if control_payload_rejected
                        else 'the upstream result is a clarification or missing-source contract error, not materializable payload.'
                    )
                ),
                'stage': 'truth_guard_gate',
                'retryable': False,
                'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                'reason': str(truth_guard.get('reason') or '').strip()
                or ('control_envelope_not_speakable' if control_payload_rejected else None),
            }
        failed_dependency_ids = self.failed_dependency_ids_for_branch(
            branch,
            current_payload=current_payload,
        )
        if failed_dependency_ids:
            return {
                'code': 'DEPENDENCY_CHAIN_REPAIR_REQUIRED',
                'message': (
                    'Dependency-chain repair required before this branch can execute: '
                    'a declared dependency failed or lacks usable evidence.'
                ),
                'stage': 'dependency_gate',
                'retryable': False,
                'repair_action': RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                'failed_dependency_ids': failed_dependency_ids,
            }
        gap_payload = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        dependency_payload = self.branch_dependency_payload(branch, current_payload=current_payload)
        repair_gate_item: dict[str, Any] = dict(gap_payload)
        repair_gate_item.update(dict(branch))
        if dependency_payload:
            for key, value in dependency_payload.items():
                repair_gate_item.setdefault(key, value)
        if gap_payload.get('depends_on') and not repair_gate_item.get('depends_on'):
            repair_gate_item['depends_on'] = gap_payload.get('depends_on')
        repair_action = self.repair_action_for_branch(branch)
        if repair_action in {
            RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
            RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
        }:
            repair_gate_item['repair_action'] = repair_action
            repair_policy = classify_repair_execution_policy(repair_gate_item)
            if repair_policy.get('execution_policy') == 'schedule_late_fill_branch':
                return None
            return {
                'code': 'DEPENDENCY_CHAIN_REPAIR_REQUIRED',
                'message': 'Dependency-chain repair required before this branch can execute: missing dependency artifact or evidence.',
                'stage': 'repair_gate',
                'retryable': False,
                'repair_action': repair_action,
                'blocked_prerequisite': repair_policy.get('blocked_prerequisite'),
                'materialization_blocked': repair_policy.get('materialization_blocked'),
                'repair_work_available': repair_policy.get('repair_work_available'),
                'repair_work_policy': repair_policy.get('repair_work_policy'),
                'needs_external_input': repair_policy.get('needs_external_input'),
            }
        if repair_action == RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS:
            repair_gate_item['repair_action'] = repair_action
            repair_policy = classify_repair_execution_policy(repair_gate_item)
            if repair_policy.get('execution_policy') == 'schedule_late_fill_branch':
                return None
            return {
                'code': 'REBUILD_FROM_PROMOTED_OBLIGATIONS_REQUIRED',
                'message': 'Rebuild from promoted obligations required before late fill can execute: graph planning missed promoted obligations.',
                'stage': 'repair_gate',
                'retryable': False,
                'repair_action': repair_action,
                'blocked_prerequisite': repair_policy.get('blocked_prerequisite') or 'promoted_obligation_branch',
                'materialization_blocked': True,
                'repair_work_available': repair_policy.get('repair_work_available'),
                'repair_work_policy': repair_policy.get('repair_work_policy'),
                'needs_external_input': repair_policy.get('needs_external_input'),
            }
        if repair_action == RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT:
            repair_gate_item['repair_action'] = repair_action
            repair_policy = classify_repair_execution_policy(repair_gate_item)
            if repair_policy.get('execution_policy') == 'schedule_late_fill_branch':
                return None
            return {
                'code': 'BRANCH_CONTRACT_REPAIR_REQUIRED',
                'message': 'Branch contract repair required before this branch can execute: missing bounded execution contract.',
                'stage': 'repair_gate',
                'retryable': False,
                'repair_action': repair_action,
                'blocked_prerequisite': repair_policy.get('blocked_prerequisite'),
                'materialization_blocked': repair_policy.get('materialization_blocked'),
                'repair_work_available': repair_policy.get('repair_work_available'),
                'repair_work_policy': repair_policy.get('repair_work_policy'),
                'needs_external_input': repair_policy.get('needs_external_input'),
            }
        has_inline_evidence = self._branch_has_inline_dependency_evidence(branch)
        has_dependency_payload = bool(dependency_payload)
        depends_on = [
            str(item or '').strip()
            for item in (branch.get('depends_on') or gap_payload.get('depends_on') or [])
            if str(item or '').strip()
        ]
        capability = self.branch_capability(branch) or self.normalize_capability(gap_payload.get('capability'))
        stage_direction = str(branch.get('stage_direction') or gap_payload.get('stage_direction') or '').strip()
        has_branch_payload = bool(
            str(branch.get('content_payload') or gap_payload.get('content_payload') or '').strip()
            or str(branch.get('artifact_prompt') or gap_payload.get('artifact_prompt') or '').strip()
            or str(branch.get('file_path') or gap_payload.get('file_path') or '').strip()
            or has_inline_evidence
            or has_dependency_payload
            or (
                isinstance(branch.get('batch_prompts') or gap_payload.get('batch_prompts'), list)
                and any(str(item or '').strip() for item in (branch.get('batch_prompts') or gap_payload.get('batch_prompts') or []))
            )
            or (
                capability == 'chat'
                and stage_direction == 'materialize_requested_text_artifact'
                and (
                    branch.get('artifact_request') not in (None, '', [], {})
                    or gap_payload.get('artifact_request') not in (None, '', [], {})
                    or str(branch.get('text_artifact_extension') or gap_payload.get('text_artifact_extension') or '').strip()
                )
            )
            or (
                isinstance(branch.get('execution_contract'), Mapping)
                and str(branch['execution_contract'].get('execution_scope') or '').strip() == 'root_scoped'
            )
        )
        if not has_branch_payload and depends_on:
            return {
                'code': 'DEPENDENCY_CHAIN_REPAIR_REQUIRED',
                'message': 'Dependency-chain repair required before this branch can execute: missing dependency artifact or evidence.',
                'stage': 'repair_gate',
                'retryable': False,
            }
        if not has_branch_payload and capability in {'chat', self.capability_image_generation, self.capability_text_to_speech, 'vision_analysis', 'speech_to_text'}:
            return {
                'code': 'BRANCH_CONTRACT_REPAIR_REQUIRED',
                'message': 'Branch contract repair required before this branch can execute: missing branch-local payload or bounded execution contract.',
                'stage': 'repair_gate',
                'retryable': False,
            }
        return None

    def _enforce_graph_patch_successor_branch_local_payload(
        self,
        late_fill_payload: dict[str, Any],
        *,
        original_prompt: str,
        artifact_gap: Mapping[str, Any],
        normalized_expected_capability: Optional[str],
        root_scoped_execution: bool,
    ) -> dict[str, Any]:
        """Keep every bounded successor execution local to its owed branch."""

        updated = dict(late_fill_payload or {})
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        trigger = str(gap.get('trigger') or '').strip().lower()
        error_prefix = (
            'graph_rebase_partial_successor'
            if trigger == 'graph_rebase_partial_successor'
            else 'graph_patch_successor'
        )
        effective_prompt = str(updated.get('prompt') or '').strip()
        root_prompt = str(original_prompt or '').strip()
        execution_contract = (
            gap.get('execution_contract')
            if isinstance(gap.get('execution_contract'), Mapping)
            else {}
        )
        forbidden_root_prompt_digest = str(
            execution_contract.get('forbidden_root_prompt_digest') or ''
        ).strip()
        artifact_prompt = str(gap.get('artifact_prompt') or '').strip()
        content_payload = str(gap.get('content_payload') or '').strip()
        phase_summary = str(gap.get('phase_summary') or '').strip()
        has_branch_artifact_input = bool(
            str(gap.get('file_path') or '').strip()
            or any(
                isinstance(item, Mapping)
                for key in ('input_artifacts', 'reference_artifacts')
                for item in (gap.get(key) if isinstance(gap.get(key), list) else [])
            )
        )

        # The successor request owns only the proven owed branches. Compatibility
        # inputs from the frozen parent must never survive as a second prompt path.
        for key in ('input', 'messages', 'ghost_messages', 'batch_prompts'):
            updated.pop(key, None)

        partial_root_guard_invalid = bool(
            trigger == 'graph_rebase_partial_successor'
            and (
                not root_prompt
                or not forbidden_root_prompt_digest
                or stable_graph_rebase_prompt_digest(root_prompt)
                != forbidden_root_prompt_digest
            )
        )
        prompt_replays_root = graph_rebase_prompt_contains_root(
            effective_prompt,
            root_prompt,
        )
        if not effective_prompt or prompt_replays_root:
            branch_prompt = next(
                (
                    candidate
                    for candidate in (artifact_prompt, content_payload, phase_summary)
                    if candidate
                    and not graph_rebase_prompt_contains_root(candidate, root_prompt)
                ),
                '',
            )
            if not branch_prompt and has_branch_artifact_input:
                if normalized_expected_capability == 'vision_analysis':
                    branch_prompt = self._vision_artifact_analysis_instruction('', evidence=content_payload)
                elif normalized_expected_capability == 'speech_to_text':
                    branch_prompt = (
                        'Transcribe only the attached branch-local audio artifact. '
                        'Return the transcript without restarting the original request.'
                    )
            effective_prompt = branch_prompt

        def prompt_carrier_strings(value: Any) -> list[str]:
            if isinstance(value, Mapping):
                values: list[str] = []
                for child in value.values():
                    values.extend(prompt_carrier_strings(child))
                return values
            if isinstance(value, (list, tuple, set)):
                values = []
                for child in value:
                    values.extend(prompt_carrier_strings(child))
                return values
            text = str(value or '').strip()
            return [text] if text else []

        prompt_carriers = [effective_prompt]
        for key in (
            'content_payload',
            'artifact_prompt',
            'batch_prompts',
            'phase_summary',
            'stage_direction',
            'instruct',
            'review_criteria',
            'semantic_review_criteria',
            'controlled_attention_question',
        ):
            prompt_carriers.extend(prompt_carrier_strings(updated.get(key)))
        carrier_replays_root = any(
            graph_rebase_prompt_contains_root(carrier, root_prompt)
            for carrier in prompt_carriers
        )
        carrier_matches_forbidden_digest = bool(
            forbidden_root_prompt_digest
            and any(
                stable_graph_rebase_prompt_digest(carrier)
                == forbidden_root_prompt_digest
                for carrier in prompt_carriers
                if carrier
            )
        )
        invalid_local_prompt = bool(
            partial_root_guard_invalid
            or not effective_prompt
            or carrier_replays_root
            or carrier_matches_forbidden_digest
        )
        if root_scoped_execution or invalid_local_prompt:
            updated.pop('prompt', None)
            updated.pop('_prompt_hint', None)
            updated.pop('instruct', None)
            updated['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
            updated['branch_contract_error'] = (
                f'{error_prefix}_root_prompt_guard_invalid'
                if partial_root_guard_invalid
                else f'{error_prefix}_root_scoped_contract_forbidden'
                if root_scoped_execution
                else f'{error_prefix}_missing_branch_local_payload'
            )
            updated['successor_root_prompt_replay_forbidden'] = True
            return updated

        updated['prompt'] = effective_prompt
        updated['_prompt_hint'] = effective_prompt
        updated['successor_branch_local_payload_enforced'] = True
        updated['successor_root_prompt_replay_forbidden'] = True
        return updated

    def prepare_late_fill_request_payload(
        self,
        request_payload: dict[str, Any],
        *,
        expected_capability: str,
        assistant_message: str,
        artifact_gap: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        late_fill_payload = self.normalize_request_payload(request_payload)
        request_meta = self.extract_request_meta(late_fill_payload)
        if expected_capability:
            request_meta = dict(request_meta or {})
            request_meta['capability_hint'] = expected_capability
            late_fill_payload['request_meta'] = request_meta
            late_fill_payload = self.attach_request_meta(late_fill_payload)
        artifact_gap_payload = artifact_gap if isinstance(artifact_gap, dict) else {}
        original_prompt = self.extract_responses_prompt(late_fill_payload)
        late_fill_trigger = str(artifact_gap_payload.get('trigger') or '').strip().lower()
        late_fill_code = str(artifact_gap_payload.get('code') or '').strip().lower()
        successor_branch_execution = late_fill_trigger in {
            'graph_patch_successor_reopen',
            'graph_rebase_partial_successor',
        }
        for key in (
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'branch_id',
            'phase_id',
            'obligation_id',
            'task_id',
            'workload_task_id',
            'output_type',
            'depends_on',
            'input_refs',
            'content_payload',
            'stage_direction',
            'phase_summary',
            'content_payload_source',
            'selection_policy',
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_reason',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
            'structured_output_contract',
            'artifact_prompt',
            'artifact_prompt_source',
            'review_criteria',
            'semantic_review_criteria',
            'semantic_review_required',
            'semantic_review_action',
            'semantic_review_authority',
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
            'aspiration_review',
            'aspiration_frame',
            'aspiration_frame_id',
            'aspiration_action',
            'aspiration_reason',
            'aspiration_allowed_actions',
            'commitment_review',
            'commitment_frame',
            'commitment_frame_id',
            'commitment_action',
            'commitment_recommended_transition',
            'commitment_reason',
            'commitment_allowed_transitions',
            'controlled_attention_review',
            'controlled_attention_frame',
            'controlled_attention_frame_id',
            'controlled_attention_scope',
            'controlled_attention_priority',
            'controlled_attention_question',
            'controlled_attention_allowed_transitions',
            'global_semantic_closure_review',
            'global_semantic_closure_proposal',
            'global_semantic_closure_status',
            'global_semantic_closure_reason',
            'global_semantic_closure_confidence',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'text_artifact_revision_required',
            'text_artifact_revision_source',
            'text_artifact_source_is_input',
            'text_artifact_revision_binding_state',
            'text_artifact_revision_source_path',
            'text_artifact_revision_source_size_bytes',
            'text_artifact_revision_source_sha256',
            'text_artifact_revision_preservation_required',
            'text_artifact_revision_preservation_policy',
            'text_artifact_requests',
            'artifact_request',
            'file_path',
            'input_artifacts',
            'reference_artifacts',
            'lang_code',
            'voice',
            'instruct',
            'response_format',
            'output_format',
            'speed',
            'pitch',
            'suppress_reference_file_context',
            'selected_reference_prompt_policy',
            'dependency_payload_policy',
            'suppress_image_state_enrichment',
            'suppress_generated_image_enrichment',
            'image_state_enrichment_suppression_reason',
            'branch_contract_error',
            'audio_variant_contract_conflicting_fields',
            'materialization_blocked',
        ):
            value = artifact_gap_payload.get(key)
            if value not in (None, '', [], {}):
                late_fill_payload[key] = value
        execution_contract = (
            late_fill_payload.get('execution_contract')
            if isinstance(late_fill_payload.get('execution_contract'), Mapping)
            else {}
        )
        if execution_contract:
            workload_task_ref = (
                execution_contract.get('workload_task_ref')
                if isinstance(execution_contract.get('workload_task_ref'), Mapping)
                else {}
            )
            output_obligation_ref = (
                execution_contract.get('output_obligation_ref')
                if isinstance(execution_contract.get('output_obligation_ref'), Mapping)
                else {}
            )
            for key in ('branch_id', 'phase_id', 'capability', 'output_type'):
                value = execution_contract.get(key)
                if value not in (None, '', [], {}) and late_fill_payload.get(key) in (None, '', [], {}):
                    late_fill_payload[key] = value
            task_id = workload_task_ref.get('task_id') if isinstance(workload_task_ref, Mapping) else None
            if task_id not in (None, '', [], {}) and late_fill_payload.get('task_id') in (None, '', [], {}):
                late_fill_payload['task_id'] = task_id
                late_fill_payload['workload_task_id'] = task_id
            obligation_id = (
                output_obligation_ref.get('obligation_id')
                if isinstance(output_obligation_ref, Mapping)
                else None
            )
            if obligation_id not in (None, '', [], {}) and late_fill_payload.get('obligation_id') in (None, '', [], {}):
                late_fill_payload['obligation_id'] = obligation_id
        root_scoped_execution = (
            str(execution_contract.get('execution_scope') or '').strip().lower()
            in {'root', 'root_scoped', 'whole_request', 'original_prompt'}
            or self.parse_bool(execution_contract.get('root_scoped'), default=False)
            or self.parse_bool(execution_contract.get('allow_root_prompt'), default=False)
        ) if execution_contract else False
        normalized_expected_capability = self.normalize_capability(expected_capability)
        artifact_prompt = str(artifact_gap_payload.get('artifact_prompt') or '').strip()
        content_payload = str(artifact_gap_payload.get('content_payload') or '').strip()
        content_payload_source = str(artifact_gap_payload.get('content_payload_source') or '').strip()
        selection_policy = str(artifact_gap_payload.get('selection_policy') or '').strip().lower()
        if (
            normalized_expected_capability == 'speech_to_text'
            and content_payload_source == 'selected_reference_audio_artifact'
        ):
            selected_audio_path = ''
            raw_selected_references: Any = None
            for key in (
                'selected_reference_artifacts',
                'selectedReferenceArtifacts',
                'selected_reference_artifact',
                'selectedReferenceArtifact',
            ):
                raw_value = late_fill_payload.get(key)
                if raw_value not in (None, '', [], {}):
                    raw_selected_references = raw_value
                    break
            if raw_selected_references in (None, '', [], {}):
                raw_selected_references = late_fill_payload.get('reference_artifacts')
            selected_reference_candidates = self.sanitize_selected_reference_artifacts(
                raw_selected_references,
                payload_source=late_fill_payload,
            )
            for artifact in selected_reference_candidates:
                artifact_type = str(
                    artifact.get('type') or artifact.get('kind') or ''
                ).strip().lower()
                if artifact_type not in {
                    'audio', 'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus'
                }:
                    continue
                selected_audio_path = self._artifact_record_path(artifact)
                if selected_audio_path:
                    break
            if selected_audio_path:
                # Explicit file authority outranks both current input fallback
                # and route reuse for this selected-reference branch.
                late_fill_payload['file_path'] = selected_audio_path
            else:
                late_fill_payload['branch_contract_error'] = (
                    'selected_reference_audio_unavailable'
                )
                late_fill_payload['materialization_blocked'] = True
        if (
            normalized_expected_capability == 'speech_to_text'
            and content_payload_source == 'current_input_audio_artifact'
        ):
            # The branch explicitly reselects the current request input. Keep
            # carried predecessor audio as reference truth, but prevent route
            # reuse from replacing the selected input at the infer boundary.
            late_fill_payload['suppress_reference_file_context'] = True
        successor_branch_prompt_context = str(
            artifact_gap_payload.get('phase_summary')
            or artifact_gap_payload.get('stage_direction')
            or ''
        ).strip().replace('_', ' ')
        bounded_prompt_context = (
            successor_branch_prompt_context
            if successor_branch_execution
            else original_prompt
        )
        if (
            normalized_expected_capability == self.capability_text_to_speech
            and late_fill_payload.get('response_format') in (None, '')
            and late_fill_payload.get('output_format') not in (None, '')
        ):
            late_fill_payload['response_format'] = str(late_fill_payload.get('output_format') or '').strip()
        if normalized_expected_capability == self.capability_text_to_speech and late_fill_payload.get('lang_code') in (None, ''):
            inferred_lang_code = (
                infer_tts_language_from_prompt(content_payload)
                or infer_tts_language_from_prompt(original_prompt)
            )
            if inferred_lang_code:
                late_fill_payload['lang_code'] = inferred_lang_code
        is_closure_or_repair = (
            late_fill_trigger in {
                'pre_freeze_closure_review',
                'ghost_repair_feedback',
                'phase_continuation',
                'runtime_applied_graph_patch',
                'graph_patch_successor_reopen',
                'graph_rebase_partial_successor',
            }
            or late_fill_code in {
                'closure_review_fill',
                'closure_review_repair',
                'late_artifact_fill',
                'graph_patch_late_fill',
                'graph_rebase_partial_late_fill',
            }
        )
        if (
            normalized_expected_capability == self.capability_image_generation
            and (artifact_prompt or (content_payload and is_closure_or_repair))
            and (is_closure_or_repair or late_fill_trigger == 'execution_planner_deferred_follow_up')
        ):
            focused_prompt = artifact_prompt or content_payload
            late_fill_payload['prompt'] = focused_prompt
            late_fill_payload['_prompt_hint'] = original_prompt or focused_prompt
        elif normalized_expected_capability == self.capability_text_to_speech and content_payload:
            late_fill_payload['prompt'] = content_payload
            late_fill_payload['_prompt_hint'] = original_prompt or content_payload
            # The branch-local payload is the exact text this TTS obligation
            # owns. Carried references remain request truth, but must not be
            # appended as a second, implicit speech source.
            late_fill_payload['suppress_reference_file_context'] = True
        elif (
            normalized_expected_capability == self.capability_text_to_speech
            and late_fill_trigger == 'execution_planner_deferred_follow_up'
        ):
            reference_prompt = 'Read that reply aloud.'
            late_fill_payload['prompt'] = reference_prompt
            late_fill_payload['_prompt_hint'] = original_prompt or reference_prompt
        elif (
            normalized_expected_capability == self.capability_image_generation
            and late_fill_trigger == 'execution_planner_deferred_follow_up'
        ):
            artifact_prompt = str(
                artifact_gap_payload.get('artifact_prompt')
                or (original_prompt if root_scoped_execution else '')
                or ''
            ).strip()
            if artifact_prompt:
                late_fill_payload['prompt'] = artifact_prompt
                late_fill_payload['_prompt_hint'] = original_prompt or artifact_prompt
            elif not root_scoped_execution:
                late_fill_payload.pop('prompt', None)
                late_fill_payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                late_fill_payload['branch_contract_error'] = 'missing_branch_local_payload'
        elif (
            normalized_expected_capability == 'vision_analysis'
            and (
                str(artifact_gap_payload.get('file_path') or '').strip()
                or artifact_gap_payload.get('input_artifacts')
                or content_payload_source == 'late_fill_dependency_artifacts'
            )
        ):
            vision_prompt = self._vision_artifact_analysis_instruction(
                bounded_prompt_context,
                evidence=content_payload,
                execution_contract=execution_contract,
            )
            if vision_prompt:
                late_fill_payload['prompt'] = vision_prompt
                late_fill_payload['_prompt_hint'] = original_prompt or vision_prompt
        elif (
            normalized_expected_capability == 'chat'
            and str(artifact_gap_payload.get('stage_direction') or '').strip() in {
                'run_global_semantic_closure_review',
                'run_branch_semantic_review',
            }
            and content_payload
        ):
            late_fill_payload['prompt'] = content_payload
            late_fill_payload['_prompt_hint'] = original_prompt or content_payload
            late_fill_payload['suppress_reference_file_context'] = True
        elif (
            normalized_expected_capability == 'chat'
            and str(artifact_gap_payload.get('stage_direction') or '').strip() == 'materialize_requested_text_artifact'
        ):
            text_artifact_prompt = self._text_artifact_materialization_instruction(
                bounded_prompt_context,
                artifact_gap_payload,
            )
            if text_artifact_prompt:
                late_fill_payload['prompt'] = text_artifact_prompt
                late_fill_payload['_prompt_hint'] = original_prompt or text_artifact_prompt
        elif (
            normalized_expected_capability == 'chat'
            and (
                str(artifact_gap_payload.get('stage_direction') or '').strip() == 'write_text_after_artifact_generation'
                or str(artifact_gap_payload.get('content_payload_source') or '').strip() == 'prior_artifact_result'
                or str(artifact_gap_payload.get('content_payload_source') or '').strip().startswith('late_fill_result')
                or (
                    is_closure_or_repair
                    and content_payload
                    and (
                        content_payload_source == 'current_phase_output'
                        or content_payload_source.startswith('late_fill_result')
                        or content_payload_source in {'prior_artifact_result', 'late_fill_dependency_artifacts'}
                    )
                )
            )
        ):
            follow_up_prompt = self._post_artifact_follow_up_instruction(
                bounded_prompt_context,
                evidence=str(artifact_gap_payload.get('content_payload') or '').strip(),
                structured_output_contract=(
                    artifact_gap_payload.get('structured_output_contract')
                    if isinstance(
                        artifact_gap_payload.get('structured_output_contract'),
                        Mapping,
                    )
                    else None
                ),
            )
            if follow_up_prompt:
                late_fill_payload['prompt'] = follow_up_prompt
                late_fill_payload['_prompt_hint'] = original_prompt or follow_up_prompt
                late_fill_payload['suppress_reference_file_context'] = True
        if successor_branch_execution:
            late_fill_payload = self._enforce_graph_patch_successor_branch_local_payload(
                late_fill_payload,
                original_prompt=original_prompt,
                artifact_gap=artifact_gap_payload,
                normalized_expected_capability=normalized_expected_capability,
                root_scoped_execution=root_scoped_execution,
            )
        late_fill_payload.pop('ghost_preview', None)
        should_attach_ghost_messages = (
            not successor_branch_execution
            and (
                self.parse_bool(late_fill_payload.get('ghost_route'), default=False)
                or late_fill_trigger == 'execution_planner_deferred_follow_up'
            )
        )
        if should_attach_ghost_messages:
            ghost_messages = self.extract_ghost_route_messages(late_fill_payload)
            assistant_entry = {
                'role': 'assistant',
                'content': str(assistant_message or '').strip(),
                'timestamp': self.response_registry_now_iso(),
            }
            if assistant_entry['content']:
                last_message = ghost_messages[-1] if ghost_messages else {}
                last_role = str(last_message.get('role') or '').strip().lower()
                last_content = str(last_message.get('content') or '').strip()
                if last_role != 'assistant' or last_content != assistant_entry['content']:
                    ghost_messages.append(assistant_entry)
            late_fill_payload['ghost_messages'] = ghost_messages[-self.max_recent_messages:]
        return late_fill_payload

    @staticmethod
    def _source_route_external_target_identity(
        source_route_payload: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Return a joined external target identity from live or persisted route truth."""

        source_route = (
            source_route_payload
            if isinstance(source_route_payload, Mapping)
            else {}
        )
        source_instance = (
            source_route.get('instance')
            if isinstance(source_route.get('instance'), Mapping)
            else {}
        )
        route_runtime = (
            source_route.get('route_runtime')
            if isinstance(source_route.get('route_runtime'), Mapping)
            else {}
        )
        runtime = (
            source_route.get('runtime')
            if isinstance(source_route.get('runtime'), Mapping)
            else {}
        )
        working_frame = (
            source_route.get('working_frame')
            if isinstance(source_route.get('working_frame'), Mapping)
            else {}
        )
        working_target = (
            working_frame.get('target')
            if isinstance(working_frame.get('target'), Mapping)
            else {}
        )
        live_external_descriptor = bool(
            source_instance
            and str(source_instance.get('target_kind') or '').strip().lower()
            == 'external'
        )
        external_descriptor: dict[str, Any] = (
            dict(source_instance) if live_external_descriptor else {}
        )
        if not external_descriptor:
            for candidate in (
                route_runtime.get('external_target'),
                runtime.get('external_target'),
                source_route.get('external_target'),
            ):
                if (
                    isinstance(candidate, Mapping)
                    and str(candidate.get('target_kind') or '')
                    .strip()
                    .lower()
                    == 'external'
                ):
                    external_descriptor = dict(candidate)
                    break
        if not external_descriptor:
            return '', {}

        descriptor_id = str(
            external_descriptor.get('instance_id')
            or external_descriptor.get('id')
            or ''
        ).strip()
        if not descriptor_id:
            return '', {}
        if not live_external_descriptor:
            external_execution = (
                route_runtime.get('external_execution')
                if isinstance(
                    route_runtime.get('external_execution'),
                    Mapping,
                )
                else runtime.get('external_execution')
                if isinstance(runtime.get('external_execution'), Mapping)
                else {}
            )
            external_phase_policy = (
                route_runtime.get('external_phase_execution_policy')
                if isinstance(
                    route_runtime.get('external_phase_execution_policy'),
                    Mapping,
                )
                else runtime.get('external_phase_execution_policy')
                if isinstance(
                    runtime.get('external_phase_execution_policy'),
                    Mapping,
                )
                else {}
            )
            execution_target_id = str(
                external_execution.get('target_id') or ''
            ).strip()
            descriptor_provider = str(
                external_descriptor.get('provider') or ''
            ).strip().lower()
            execution_provider = str(
                external_execution.get('provider') or ''
            ).strip().lower()
            if (
                str(external_execution.get('status') or '')
                .strip()
                .lower()
                != 'completed'
                or execution_target_id != descriptor_id
                or (
                    descriptor_provider
                    and execution_provider
                    and descriptor_provider != execution_provider
                )
                or str(external_phase_policy.get('status') or '')
                .strip()
                .lower()
                != 'bounded'
                or str(
                    external_phase_policy.get('materialization_authority')
                    or ''
                ).strip()
                != 'ollmo_runtime'
                or str(
                    external_phase_policy.get('root_request_authority')
                    or ''
                ).strip()
                != 'promoted_context_reference_only'
            ):
                return '', {}
        for candidate_id in (
            source_route.get('instance_id'),
            source_instance.get('instance_id'),
            source_instance.get('id'),
            working_target.get('instance_id'),
            working_target.get('id'),
        ):
            token = str(candidate_id or '').strip()
            if token and token != descriptor_id:
                return '', {}
        return descriptor_id, external_descriptor

    @staticmethod
    def _source_route_with_response_external_runtime(
        source_route_payload: Any,
        response_payload: Any,
    ) -> dict[str, Any]:
        """Carry only CAS-backed external authority into a recovered route view."""

        source_route = (
            dict(source_route_payload)
            if isinstance(source_route_payload, Mapping)
            else {}
        )
        response_runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload, Mapping)
            and isinstance(response_payload.get('runtime'), Mapping)
            else {}
        )
        external_target = (
            response_runtime.get('external_target')
            if isinstance(response_runtime.get('external_target'), Mapping)
            else {}
        )
        if (
            not external_target
            or str(external_target.get('target_kind') or '')
            .strip()
            .lower()
            != 'external'
        ):
            return source_route
        external_execution = (
            response_runtime.get('external_execution')
            if isinstance(response_runtime.get('external_execution'), Mapping)
            else {}
        )
        external_phase_policy = (
            response_runtime.get('external_phase_execution_policy')
            if isinstance(
                response_runtime.get('external_phase_execution_policy'),
                Mapping,
            )
            else {}
        )
        working_frame = (
            response_payload.get('working_frame')
            if isinstance(response_payload, Mapping)
            and isinstance(response_payload.get('working_frame'), Mapping)
            else {}
        )
        working_target = (
            working_frame.get('target')
            if isinstance(working_frame.get('target'), Mapping)
            else {}
        )
        response_frame = (
            response_payload.get('response_frame')
            if isinstance(response_payload, Mapping)
            and isinstance(response_payload.get('response_frame'), Mapping)
            else {}
        )
        if not working_target:
            frame_working_frame = (
                response_frame.get('working_frame')
                if isinstance(response_frame.get('working_frame'), Mapping)
                else {}
            )
            working_target = (
                frame_working_frame.get('target')
                if isinstance(frame_working_frame.get('target'), Mapping)
                else response_frame.get('target')
                if isinstance(response_frame.get('target'), Mapping)
                else {}
            )
        external_target_id = str(
            external_target.get('instance_id')
            or external_target.get('id')
            or ''
        ).strip()
        execution_target_id = str(
            external_execution.get('target_id') or ''
        ).strip()
        working_target_id = str(
            working_target.get('instance_id')
            or working_target.get('id')
            or ''
        ).strip()
        if (
            not external_target_id
            or execution_target_id != external_target_id
            or working_target_id != external_target_id
            or str(external_execution.get('status') or '').strip().lower()
            != 'completed'
            or str(external_phase_policy.get('status') or '')
            .strip()
            .lower()
            != 'bounded'
            or str(
                external_phase_policy.get('materialization_authority') or ''
            ).strip()
            != 'ollmo_runtime'
            or str(
                external_phase_policy.get('root_request_authority') or ''
            ).strip()
            != 'promoted_context_reference_only'
            or str(working_target.get('capability') or '').strip().lower()
            != 'chat'
            or str(working_target.get('mode') or '').strip().lower()
            != 'external_chat'
        ):
            return source_route
        route_runtime = (
            dict(source_route.get('route_runtime') or {})
            if isinstance(source_route.get('route_runtime'), Mapping)
            else {}
        )
        for key in (
            'external_target',
            'external_execution',
            'external_phase_execution_policy',
        ):
            if key in route_runtime:
                continue
            value = response_runtime.get(key)
            if isinstance(value, Mapping) and value:
                route_runtime[key] = dict(value)
        if route_runtime:
            source_route['route_runtime'] = route_runtime
            source_route['working_frame'] = {
                'target': dict(working_target),
            }
        return source_route

    def resolve_late_fill_route(
        self,
        request_payload: dict[str, Any],
        *,
        expected_capability: str,
        failed_instance_id: Optional[str],
        excluded_instance_ids: Optional[list[str]] = None,
        artifact_gap: Optional[dict[str, Any]] = None,
        source_route_payload: Optional[dict[str, Any]] = None,
        resolve_ghost_auto_route: Optional[Callable[..., tuple[Optional[dict[str, Any]], Optional[str]]]] = None,
        load_running_instances: Optional[Callable[[], list[dict[str, Any]]]] = None,
        merge_instances_with_runtime_status: Optional[Callable[..., list[dict[str, Any]]]] = None,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[str]]:
        resolve_auto_route = resolve_ghost_auto_route or self.resolve_ghost_auto_route
        load_instances = load_running_instances or self.load_running_instances
        merge_instances = merge_instances_with_runtime_status or self.merge_instances_with_runtime_status
        late_fill_payload = dict(request_payload or {})
        resolved_excluded_instance_ids = []
        for item in [
            *(excluded_instance_ids if isinstance(excluded_instance_ids, list) else []),
            failed_instance_id,
        ]:
            token = str(item or '').strip()
            if token and token not in resolved_excluded_instance_ids:
                resolved_excluded_instance_ids.append(token)
        artifact_gap_payload = artifact_gap if isinstance(artifact_gap, dict) else {}
        late_fill_trigger = str(artifact_gap_payload.get('trigger') or '').strip().lower()
        phase_continuation = (
            self.normalize_capability(expected_capability) is not None
            and late_fill_trigger == 'execution_planner_deferred_follow_up'
        )
        normalized_expected_capability = self.normalize_capability(
            expected_capability
        )
        source_route = (
            source_route_payload
            if isinstance(source_route_payload, dict)
            else {}
        )
        source_external_instance_id, source_external_descriptor = (
            self._source_route_external_target_identity(source_route)
        )
        execution_contract = (
            artifact_gap_payload.get('execution_contract')
            if isinstance(
                artifact_gap_payload.get('execution_contract'),
                Mapping,
            )
            else {}
        )
        graph_branch_id = str(
            execution_contract.get('branch_id')
            or artifact_gap_payload.get('branch_id')
            or ''
        ).strip()
        graph_phase_id = str(
            execution_contract.get('phase_id')
            or artifact_gap_payload.get('phase_id')
            or ''
        ).strip()
        selected_external_graph_chat = bool(
            normalized_expected_capability == 'chat'
            and execution_contract
            and (graph_branch_id or graph_phase_id)
            and source_external_instance_id
            and source_external_instance_id
            not in resolved_excluded_instance_ids
            and source_external_descriptor
            and callable(self.load_external_targets)
        )
        current_external_target: dict[str, Any] = {}
        if selected_external_graph_chat:
            try:
                current_external_target = next(
                    (
                        dict(item)
                        for item in (self.load_external_targets() or [])
                        if isinstance(item, Mapping)
                        and str(
                            item.get('instance_id') or item.get('id') or ''
                        ).strip()
                        == source_external_instance_id
                    ),
                    {},
                )
            except Exception as exc:  # noqa: BLE001
                logging.info(
                    'Could not refresh selected external chat target %s: %s',
                    source_external_instance_id,
                    exc,
                )
        if (
            current_external_target
            and current_external_target.get('available') is True
            and current_external_target.get('enabled') is True
            and current_external_target.get('selectable') is True
            and self.instance_supports_capability(
                current_external_target,
                expected_capability,
            )
        ):
            route_source = (
                'phase_continuation'
                if phase_continuation
                else 'late_fill'
            )
            route_reason = (
                'Selected external chat provider continued one graph-owned '
                'branch under Ollmo materialization authority.'
            )
            candidate_diagnostic = {
                'instance_id': source_external_instance_id,
                'capability': 'chat',
                'backend': str(
                    current_external_target.get('backend') or ''
                ).strip()
                or None,
                'model': str(
                    current_external_target.get('model') or ''
                ).strip()
                or None,
                'target_kind': 'external',
                'supports_expected_capability': True,
                'excluded': False,
                'usable': True,
                'selected': True,
                'readiness': str(
                    current_external_target.get('readiness') or ''
                ).strip()
                or None,
            }
            candidate_diagnostic = {
                key: value
                for key, value in candidate_diagnostic.items()
                if value not in (None, '', [], {})
            }
            route_runtime_seed = {
                **source_route,
                'route_source': route_source,
                'route_reason': route_reason,
                'capability': 'chat',
                'instance_id': source_external_instance_id,
                'instance': current_external_target,
                'selection_policy': (
                    'selected_external_provider_for_graph_chat_phase'
                ),
                'root_request_authority': (
                    'promoted_context_reference_only'
                ),
                'materialization_authority': 'ollmo_runtime',
                'excluded_instance_ids': list(
                    resolved_excluded_instance_ids
                ),
                'candidate_diagnostics': [candidate_diagnostic],
            }
            route_runtime = self.merge_request_meta_runtime_truth(
                {},
                late_fill_payload,
                route_payload=route_runtime_seed,
            )
            route_runtime.update(
                {
                    'selection_policy': (
                        'selected_external_provider_for_graph_chat_phase'
                    ),
                    'root_request_authority': (
                        'promoted_context_reference_only'
                    ),
                    'materialization_authority': 'ollmo_runtime',
                    'excluded_instance_ids': list(
                        resolved_excluded_instance_ids
                    ),
                    'candidate_diagnostics': [candidate_diagnostic],
                }
            )
            continuation_diagnostic = {
                'active': True,
                'trigger': late_fill_trigger,
                'expected_capability': 'chat',
                'branch_id': graph_branch_id or None,
                'phase_id': graph_phase_id or None,
            }
            route_runtime[
                'phase_continuation'
                if phase_continuation
                else 'graph_chat_continuation'
            ] = continuation_diagnostic
            route_info = {
                'instance_id': source_external_instance_id,
                'instance': current_external_target,
                'capability': 'chat',
                'route_source': route_source,
                'route_reason': route_reason,
                'route_confidence': 1.0,
                'route_reuse_last_artifact': False,
                'route_runtime': route_runtime,
            }
            request_meta = (
                route_runtime.get('request_meta')
                if isinstance(route_runtime.get('request_meta'), dict)
                else {}
            )
            if request_meta:
                route_info['request_meta'] = request_meta
            return late_fill_payload, route_info, None
        if self.parse_bool(late_fill_payload.get('ghost_route'), default=False) and not phase_continuation:
            route_info, resolution_error = resolve_auto_route(
                late_fill_payload,
                excluded_instance_ids=resolved_excluded_instance_ids,
            )
            if route_info and self.normalize_capability(route_info.get('capability')) == self.normalize_capability(expected_capability):
                route_instance = (
                    route_info.get('instance')
                    if isinstance(route_info.get('instance'), Mapping)
                    else {}
                )
                if not route_instance or _late_fill_instance_is_usable(
                    route_instance,
                    capability=expected_capability,
                ):
                    return late_fill_payload, route_info, None
                logging.info(
                    'Late fill auto-route fallback: selected instance %s is not usable.',
                    route_info.get('instance_id'),
                )
            if resolution_error:
                logging.info('Late fill auto-route fallback: %s', resolution_error)

        runtime_candidate_snapshot = (
            artifact_gap_payload.get('_runtime_candidate_snapshot')
            if isinstance(artifact_gap_payload.get('_runtime_candidate_snapshot'), list)
            else []
        )
        if runtime_candidate_snapshot:
            instances = [
                dict(entry)
                for entry in runtime_candidate_snapshot
                if isinstance(entry, Mapping)
            ]
            candidate_snapshot_meta = self._runtime_candidate_snapshot_meta(
                instances,
                source='late_fill_wave',
            )
        else:
            instances = merge_instances(
                load_instances(),
                path=self.runtime_status_path_getter(),
                refresh=True,
            )
            candidate_snapshot_meta = self._runtime_candidate_snapshot_meta(
                [
                    dict(entry)
                    for entry in (instances or [])
                    if isinstance(entry, Mapping)
                ],
                source='per_branch_refresh',
            )
        used_runtime_candidate_snapshot = bool(runtime_candidate_snapshot)

        def refresh_snapshot_candidates_from_live_truth(reason: str) -> bool:
            nonlocal instances, candidate_snapshot_meta
            if not used_runtime_candidate_snapshot:
                return False
            try:
                live_instances = merge_instances(
                    load_instances(),
                    path=self.runtime_status_path_getter(),
                    refresh=True,
                )
            except Exception as exc:  # noqa: BLE001
                logging.info('Late-fill live candidate refresh failed after %s: %s', reason, exc)
                late_fill_payload['_late_fill_route_candidate_refresh'] = {
                    'attempted': True,
                    'applied': False,
                    'reason': str(reason or '').strip(),
                    'error': str(exc),
                }
                return False

            if reason == 'snapshot_tts_recovery_requires_live_alternative_check':
                instances = [
                    dict(entry)
                    for entry in (live_instances or [])
                    if isinstance(entry, Mapping)
                ]
                candidate_snapshot_meta = self._runtime_candidate_snapshot_meta(
                    instances,
                    source='per_branch_refresh_for_tts_recovery',
                )
                candidate_snapshot_meta['refresh_reason'] = str(reason or '').strip()
                candidate_snapshot_meta['snapshot_candidate_count'] = len(runtime_candidate_snapshot)
                candidate_snapshot_meta['live_candidate_count'] = len(instances)
                late_fill_payload['_late_fill_route_candidate_refresh'] = {
                    'attempted': True,
                    'applied': True,
                    'reason': str(reason or '').strip(),
                    'snapshot_candidate_count': len(runtime_candidate_snapshot),
                    'live_candidate_count': len(instances),
                    'candidate_count': len(instances),
                }
                return True

            combined: list[dict[str, Any]] = []
            seen: set[str] = set()
            for entry in [*(instances or []), *(live_instances or [])]:
                if not isinstance(entry, Mapping):
                    continue
                item = dict(entry)
                instance_id = str(item.get('instance_id') or '').strip()
                dedupe_key = instance_id or f'candidate-{len(combined)}'
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                combined.append(item)
            if len(combined) <= len(instances or []):
                late_fill_payload['_late_fill_route_candidate_refresh'] = {
                    'attempted': True,
                    'applied': False,
                    'reason': str(reason or '').strip(),
                    'live_candidate_count': len(live_instances or []),
                    'candidate_count': len(instances or []),
                }
                return False

            instances = combined
            candidate_snapshot_meta = self._runtime_candidate_snapshot_meta(
                [
                    dict(entry)
                    for entry in combined
                    if isinstance(entry, Mapping)
                ],
                source='late_fill_wave_plus_per_branch_refresh',
            )
            candidate_snapshot_meta['refresh_reason'] = str(reason or '').strip()
            candidate_snapshot_meta['snapshot_candidate_count'] = len(runtime_candidate_snapshot)
            candidate_snapshot_meta['live_candidate_count'] = len(live_instances or [])
            late_fill_payload['_late_fill_route_candidate_refresh'] = {
                'attempted': True,
                'applied': True,
                'reason': str(reason or '').strip(),
                'snapshot_candidate_count': len(runtime_candidate_snapshot),
                'live_candidate_count': len(live_instances or []),
                'combined_candidate_count': len(combined),
            }
            return True

        def build_candidate_diagnostics(
            entries: list[dict[str, Any]],
            *,
            selected_instance_id: str = '',
            reused_excluded_instance_id: str = '',
        ) -> list[dict[str, Any]]:
            diagnostics: list[dict[str, Any]] = []
            for entry in entries[:12]:
                instance_id = str(entry.get('instance_id') or '').strip()
                if not instance_id:
                    continue
                supports = self.instance_supports_capability(entry, expected_capability)
                excluded = instance_id in resolved_excluded_instance_ids
                excluded_reuse_applied = bool(
                    excluded
                    and reused_excluded_instance_id
                    and instance_id == reused_excluded_instance_id
                )
                usable = bool(
                    supports
                    and (not excluded or excluded_reuse_applied)
                    and _late_fill_instance_is_usable(entry, capability=expected_capability)
                )
                rejection_reasons: list[str] = []
                if not supports:
                    rejection_reasons.append('capability_mismatch')
                if excluded and not excluded_reuse_applied:
                    rejection_reasons.append('excluded_instance')
                if supports and not _late_fill_instance_is_usable(entry, capability=expected_capability):
                    rejection_reasons.append('not_ready')
                item = {
                    'instance_id': instance_id,
                    'capability': self.normalize_capability(entry.get('capability')) or str(entry.get('capability') or '').strip() or None,
                    'backend': self.normalize_backend(entry.get('backend')) or str(entry.get('backend') or '').strip() or None,
                    'model': str(entry.get('model') or entry.get('modelName') or '').strip() or None,
                    'supports_expected_capability': supports,
                    'excluded': excluded,
                    'excluded_reuse_applied': excluded_reuse_applied,
                    'usable': usable,
                    'selected': bool(selected_instance_id and instance_id == selected_instance_id),
                    'readiness': str(
                        (entry.get('runtime_status') or {}).get('readiness')
                        if isinstance(entry.get('runtime_status'), Mapping)
                        else entry.get('readiness') or ''
                    ).strip() or None,
                    'activity': str(
                        (entry.get('runtime_status') or {}).get('activity')
                        if isinstance(entry.get('runtime_status'), Mapping)
                        else entry.get('activity') or ''
                    ).strip() or None,
                    'rejection_reasons': rejection_reasons,
                }
                diagnostics.append(
                    {
                        key: value
                        for key, value in item.items()
                        if value not in (None, '', [], {})
                    }
                )
            return diagnostics

        def attach_route_diagnostics(message: str) -> str:
            late_fill_payload['_late_fill_route_diagnostics'] = {
                'expected_capability': normalized_expected_capability or str(expected_capability or '').strip(),
                'excluded_instance_ids': list(resolved_excluded_instance_ids),
                'candidate_count': len(instances or []),
                'candidate_diagnostics': build_candidate_diagnostics(
                    [
                        dict(entry)
                        for entry in (instances or [])
                        if isinstance(entry, dict)
                    ]
                ),
                'message': str(message or '').strip(),
            }
            return message

        def compute_candidate_pools() -> tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            int,
            bool,
            bool,
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]:
            all_items = [
                entry
                for entry in instances
                if isinstance(entry, dict)
                and self.instance_supports_capability(entry, expected_capability)
            ]
            filtered_items = [
                entry
                for entry in all_items
                if str(entry.get('instance_id') or '').strip() not in resolved_excluded_instance_ids
            ]
            usable_compatible_items = [
                entry
                for entry in all_items
                if _late_fill_instance_is_usable(entry, capability=expected_capability)
            ]
            usable_nonexcluded_items = [
                entry
                for entry in usable_compatible_items
                if str(entry.get('instance_id') or '').strip() not in resolved_excluded_instance_ids
            ]
            excluded_count = sum(
                1
                for entry in all_items
                if str(entry.get('instance_id') or '').strip() in resolved_excluded_instance_ids
            )
            reuse_excluded_for_text_repair = bool(
                resolved_excluded_instance_ids
                and not filtered_items
                and self._artifact_gap_allows_excluded_candidate_reuse_for_text_repair(artifact_gap_payload)
            )
            sole_usable_candidate_id = (
                str(usable_compatible_items[0].get('instance_id') or '').strip()
                if len(usable_compatible_items) == 1
                else ''
            )
            recovery_attempt = (
                artifact_gap_payload.get('recovery_attempt')
                if isinstance(artifact_gap_payload.get('recovery_attempt'), Mapping)
                else {}
            )
            recovery_failed_instance_id = str(
                recovery_attempt.get('failed_instance_id') or ''
            ).strip()
            reuse_excluded_for_tts_recovery = bool(
                normalized_expected_capability == self.capability_text_to_speech
                and resolved_excluded_instance_ids
                and not usable_nonexcluded_items
                and len(usable_compatible_items) == 1
                and sole_usable_candidate_id
                and sole_usable_candidate_id == str(failed_instance_id or '').strip()
                and sole_usable_candidate_id == recovery_failed_instance_id
                and sole_usable_candidate_id in resolved_excluded_instance_ids
                and self._artifact_gap_allows_excluded_candidate_reuse_for_tts_recovery(
                    artifact_gap_payload
                )
            )
            base_items = (
                all_items
                if reuse_excluded_for_text_repair or reuse_excluded_for_tts_recovery
                else filtered_items
                if resolved_excluded_instance_ids
                else all_items
            )
            usable_items = [
                entry
                for entry in base_items
                if _late_fill_instance_is_usable(entry, capability=expected_capability)
            ]
            usable_items = sorted(
                usable_items,
                key=lambda entry: runtime_instance_score(entry, capability=expected_capability),
                reverse=True,
            )
            return (
                all_items,
                filtered_items,
                excluded_count,
                reuse_excluded_for_text_repair,
                reuse_excluded_for_tts_recovery,
                base_items,
                usable_items,
            )

        (
            all_candidates,
            filtered_candidates,
            excluded_candidate_count,
            reuse_excluded_candidates_for_text_repair,
            reuse_excluded_candidate_for_tts_recovery,
            base_candidates,
            usable_candidates,
        ) = compute_candidate_pools()
        refresh_reason = ''
        if reuse_excluded_candidate_for_tts_recovery and used_runtime_candidate_snapshot:
            refresh_reason = 'snapshot_tts_recovery_requires_live_alternative_check'
        elif not all_candidates:
            refresh_reason = 'snapshot_missing_expected_capability'
        elif resolved_excluded_instance_ids and not filtered_candidates and not reuse_excluded_candidates_for_text_repair:
            refresh_reason = 'snapshot_candidates_excluded'
        elif base_candidates and not usable_candidates:
            refresh_reason = 'snapshot_candidates_not_usable'
        if refresh_reason:
            refreshed_candidates = refresh_snapshot_candidates_from_live_truth(refresh_reason)
            if refreshed_candidates:
                (
                    all_candidates,
                    filtered_candidates,
                    excluded_candidate_count,
                    reuse_excluded_candidates_for_text_repair,
                    reuse_excluded_candidate_for_tts_recovery,
                    base_candidates,
                    usable_candidates,
                ) = compute_candidate_pools()
            elif refresh_reason == 'snapshot_tts_recovery_requires_live_alternative_check':
                reuse_excluded_candidate_for_tts_recovery = False
                base_candidates = list(filtered_candidates)
                usable_candidates = [
                    entry
                    for entry in base_candidates
                    if _late_fill_instance_is_usable(entry, capability=expected_capability)
                ]
        if not all_candidates:
            return late_fill_payload, None, attach_route_diagnostics(
                f"No running instance for late fill capability '{expected_capability}'."
            )
        if base_candidates and not usable_candidates:
            unusable_text = ', '.join(
                _late_fill_instance_unusable_summary(entry, capability=expected_capability)
                for entry in base_candidates
            )
            return (
                late_fill_payload,
                None,
                attach_route_diagnostics(
                    f"No ready late-fill instance for capability '{expected_capability}'. "
                    f"Unusable instance ids: {unusable_text}."
                ),
            )
        candidates = self._prefer_non_mlx_vlm_for_required_text_candidates(
            usable_candidates,
            artifact_gap_payload,
        )
        candidates = self._prefer_lightweight_chat_for_required_text_candidates(
            candidates,
            artifact_gap_payload,
            source_route_payload,
        )
        if not candidates:
            excluded_text = ', '.join(resolved_excluded_instance_ids)
            return (
                late_fill_payload,
                None,
                attach_route_diagnostics(
                    f"No non-excluded running instance for late fill capability '{expected_capability}'. "
                    f"Excluded instance ids: {excluded_text}."
                ),
            )
        prompt = self.extract_responses_prompt(late_fill_payload)
        spread_retry_preferred_ids = [
            str(item or '').strip()
            for item in artifact_gap_payload.get('_spread_retry_preferred_instance_ids') or []
            if str(item or '').strip()
        ]
        selected_via_spread_retry = ''
        if spread_retry_preferred_ids:
            candidate_ids = {
                str(entry.get('instance_id') or '').strip()
                for entry in candidates
                if str(entry.get('instance_id') or '').strip()
            }
            for instance_id in spread_retry_preferred_ids:
                if instance_id in candidate_ids:
                    selected_via_spread_retry = instance_id
                    break
        selected_instance_id = (
            selected_via_spread_retry
            or self.pick_prompt_preferred_instance(candidates, prompt)
            or self.pick_default_capability_instance(candidates)
            or ''
        )
        if not selected_instance_id:
            return late_fill_payload, None, attach_route_diagnostics(
                f"No selectable instance for late fill capability '{expected_capability}'."
            )
        selected_instance = next(
            (entry for entry in candidates if str(entry.get('instance_id') or '').strip() == selected_instance_id),
            None,
        )
        if not selected_instance:
            return late_fill_payload, None, attach_route_diagnostics(
                f"Late-fill instance '{selected_instance_id}' could not be resolved."
            )
        tts_recovery_attempt = (
            artifact_gap_payload.get('recovery_attempt')
            if isinstance(
                artifact_gap_payload.get('recovery_attempt'),
                Mapping,
            )
            else {}
        )
        tts_recovery_trigger = str(
            tts_recovery_attempt.get('trigger') or ''
        ).strip().lower()
        bounded_tts_recovery = (
            self._artifact_gap_allows_excluded_candidate_reuse_for_tts_recovery(
                artifact_gap_payload
            )
        )
        route_source = 'phase_continuation' if phase_continuation else 'late_fill'
        route_reason = (
            f'phase continuation materialized deferred {self.normalize_capability(expected_capability)} after current phase completion'
            if phase_continuation
            else f'late artifact fill continued closed chat completion as {self.normalize_capability(expected_capability)}'
        )
        route_runtime_seed = {
            **source_route,
            'route_source': route_source,
            'route_reason': route_reason,
            'capability': self.normalize_capability(expected_capability),
            'instance_id': selected_instance_id,
            'instance': selected_instance,
            'excluded_instance_ids': resolved_excluded_instance_ids,
            'excluded_candidate_count': excluded_candidate_count,
            'runtime_candidate_snapshot': candidate_snapshot_meta,
            'candidate_diagnostics': build_candidate_diagnostics(
                [
                    dict(entry)
                    for entry in (instances or [])
                    if isinstance(entry, dict)
                ],
                selected_instance_id=selected_instance_id,
                reused_excluded_instance_id=(
                    selected_instance_id
                    if (
                        reuse_excluded_candidates_for_text_repair
                        or reuse_excluded_candidate_for_tts_recovery
                    )
                    else ''
                ),
            ),
        }
        if reuse_excluded_candidates_for_text_repair:
            route_runtime_seed['selection_policy'] = 'excluded_reuse_for_authoritative_text_repair'
            route_runtime_seed['excluded_instance_reuse_reason'] = (
                'auto_executable_text_repair_candidate_pool_exhausted'
            )
        if selected_via_spread_retry:
            route_runtime_seed['selection_policy'] = 'spread_retry_preferred_instance'
            route_runtime_seed['spread_retry_reason'] = str(
                artifact_gap_payload.get('_spread_retry_reason') or 'internal_reservation_exhausted'
            ).strip()
            route_runtime_seed['spread_retry_preferred_instance_ids'] = list(spread_retry_preferred_ids)
        if (
            bounded_tts_recovery
            and not reuse_excluded_candidate_for_tts_recovery
            and selected_instance_id not in resolved_excluded_instance_ids
        ):
            route_runtime_seed['selection_policy'] = (
                'nonexcluded_alternative_for_tts_recovery'
            )
            route_runtime_seed['tts_recovery_trigger'] = (
                tts_recovery_trigger
            )
            route_runtime_seed['tts_recovery_policy_id'] = str(
                tts_recovery_attempt.get('recovery_policy_id') or ''
            ).strip() or None
        if reuse_excluded_candidate_for_tts_recovery:
            route_runtime_seed['selection_policy'] = 'excluded_reuse_for_single_tts_recovery'
            route_runtime_seed['excluded_instance_reuse_reason'] = (
                'bounded_auto_retry_single_compatible_tts_instance'
                if tts_recovery_trigger
                == _TTS_AUTO_RECOVERY_TRIGGER
                else 'explicit_retry_single_compatible_tts_instance'
            )
            route_runtime_seed['excluded_instance_reuse_instance_id'] = selected_instance_id
            route_runtime_seed['excluded_instance_reuse_recovery_trigger'] = (
                tts_recovery_trigger
            )
            route_runtime_seed['tts_recovery_policy_id'] = str(
                tts_recovery_attempt.get('recovery_policy_id') or ''
            ).strip() or None
            logging.info(
                'Late-fill TTS recovery is reusing sole compatible excluded instance %s '
                'for branch %s after live candidate refresh.',
                selected_instance_id,
                artifact_gap_payload.get('branch_id'),
            )
        route_runtime = self.merge_request_meta_runtime_truth({}, late_fill_payload, route_payload=route_runtime_seed)
        route_runtime['runtime_candidate_snapshot'] = dict(candidate_snapshot_meta)
        route_runtime['candidate_diagnostics'] = list(route_runtime_seed.get('candidate_diagnostics') or [])
        if reuse_excluded_candidate_for_tts_recovery:
            route_runtime['selection_policy'] = 'excluded_reuse_for_single_tts_recovery'
            route_runtime['excluded_instance_reuse_reason'] = str(
                route_runtime_seed.get('excluded_instance_reuse_reason') or ''
            ).strip()
            route_runtime['excluded_instance_reuse_instance_id'] = selected_instance_id
            route_runtime['excluded_instance_reuse_recovery_trigger'] = (
                tts_recovery_trigger
            )
            route_runtime['tts_recovery_policy_id'] = str(
                route_runtime_seed.get('tts_recovery_policy_id') or ''
            ).strip() or None
        elif selected_via_spread_retry:
            route_runtime['selection_policy'] = 'spread_retry_preferred_instance'
            route_runtime['spread_retry_reason'] = str(route_runtime_seed.get('spread_retry_reason') or '').strip()
            route_runtime['spread_retry_preferred_instance_ids'] = list(spread_retry_preferred_ids)
        elif (
            bounded_tts_recovery
            and selected_instance_id not in resolved_excluded_instance_ids
        ):
            route_runtime['selection_policy'] = (
                'nonexcluded_alternative_for_tts_recovery'
            )
            route_runtime['tts_recovery_trigger'] = tts_recovery_trigger
            route_runtime['tts_recovery_policy_id'] = str(
                route_runtime_seed.get('tts_recovery_policy_id') or ''
            ).strip() or None
        elif reuse_excluded_candidates_for_text_repair:
            route_runtime['selection_policy'] = 'excluded_reuse_for_authoritative_text_repair'
            route_runtime['excluded_instance_reuse_reason'] = str(
                route_runtime_seed.get('excluded_instance_reuse_reason') or ''
            ).strip()
        if resolved_excluded_instance_ids:
            route_runtime['excluded_instance_ids'] = list(resolved_excluded_instance_ids)
            route_runtime['excluded_candidate_count'] = excluded_candidate_count
        request_meta = route_runtime.get('request_meta') if isinstance(route_runtime.get('request_meta'), dict) else {}
        if phase_continuation:
            route_runtime['phase_continuation'] = {
                'active': True,
                'trigger': late_fill_trigger,
                'expected_capability': self.normalize_capability(expected_capability),
            }
        route_info = {
            'instance_id': selected_instance_id,
            'instance': selected_instance,
            'capability': self.normalize_capability(expected_capability),
            'route_source': route_source,
            'route_reason': route_reason,
            'route_confidence': 1.0,
            'route_reuse_last_artifact': bool(source_route.get('route_reuse_last_artifact')),
            'route_artifact_ref': str(source_route.get('route_artifact_ref') or '').strip() or None,
            'route_artifact_path': str(source_route.get('route_artifact_path') or '').strip() or None,
            'route_runtime': route_runtime,
        }
        if request_meta:
            route_info['request_meta'] = request_meta
        return late_fill_payload, route_info, None

    def request_phase_graph_for_late_fill(
        self,
        *,
        route_payload: Optional[dict[str, Any]] = None,
        artifact_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        def _overlay_runtime_phase_statuses(graph: dict[str, Any]) -> dict[str, Any]:
            phases = graph.get('phases')
            if not isinstance(phases, list):
                return graph
            payload = artifact_payload if isinstance(artifact_payload, dict) else {}
            late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), dict) else {}
            completed_phase_ids: set[str] = set()
            failed_phase_ids: set[str] = set()
            current_phase_id = str(graph.get('current_phase_id') or '').strip()
            payload_status = str(payload.get('status') or '').strip().lower()
            if current_phase_id and payload_status in {'completed', 'failed', 'incomplete'}:
                completed_phase_ids.add(current_phase_id)
            for branch in (late_fill.get('completed_branches') or []):
                if not isinstance(branch, Mapping):
                    continue
                phase_id = str(branch.get('phase_id') or branch.get('branch_id') or '').strip()
                if phase_id:
                    completed_phase_ids.add(phase_id)
            for branch in (late_fill.get('failed_branches') or []):
                if not isinstance(branch, Mapping):
                    continue
                phase_id = str(branch.get('phase_id') or branch.get('branch_id') or '').strip()
                if phase_id:
                    failed_phase_ids.add(phase_id)
            for result in (late_fill.get('fill_results') or []):
                if not isinstance(result, Mapping):
                    continue
                if not self.late_fill_result_has_missing_dependency_evidence(result):
                    continue
                phase_id = str(result.get('phase_id') or result.get('branch_id') or '').strip()
                if phase_id:
                    failed_phase_ids.add(phase_id)
                    completed_phase_ids.discard(phase_id)
            if not completed_phase_ids and not failed_phase_ids:
                return graph
            overlaid_phases: list[dict[str, Any]] = []
            for raw_phase in phases:
                if not isinstance(raw_phase, Mapping):
                    continue
                phase = dict(raw_phase)
                phase_id = str(phase.get('phase_id') or '').strip()
                if phase_id in failed_phase_ids:
                    phase['status'] = 'failed'
                elif phase_id in completed_phase_ids:
                    phase['status'] = 'completed'
                overlaid_phases.append(phase)
            return {
                **graph,
                'phases': overlaid_phases,
            }

        artifact_info = artifact_payload if isinstance(artifact_payload, dict) else {}
        artifact_runtime = artifact_info.get('runtime') if isinstance(artifact_info.get('runtime'), dict) else {}
        artifact_graph = (
            artifact_runtime.get('request_phase_graph')
            if isinstance(artifact_runtime.get('request_phase_graph'), dict)
            else {}
        )
        if artifact_graph:
            return _overlay_runtime_phase_statuses(dict(artifact_graph))
        route_info = route_payload if isinstance(route_payload, dict) else {}
        route_runtime = route_info.get('route_runtime') if isinstance(route_info.get('route_runtime'), dict) else {}
        route_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else {}
        )
        return _overlay_runtime_phase_statuses(dict(route_graph)) if route_graph else {}

    @staticmethod
    def _artifact_record_merge_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
        path = str(item.get('path') or '').strip()
        source_path = str(item.get('source_path') or '').strip()
        if path or source_path:
            return ('path', path, source_path)
        artifact_ref = str(item.get('artifact_ref') or item.get('ref') or '').strip()
        artifact_id = str(item.get('artifact_id') or '').strip()
        if artifact_ref or artifact_id:
            return ('ref', artifact_ref, artifact_id)
        return (
            'fallback',
            str(item.get('type') or item.get('kind') or '').strip().lower(),
            str(item.get('name') or item.get('source_name') or '').strip(),
        )

    @staticmethod
    def _merge_artifact_record_fields(
        existing: dict[str, Any],
        incoming: Mapping[str, Any],
    ) -> dict[str, Any]:
        for key, value in dict(incoming or {}).items():
            if value in (None, '', [], {}):
                continue
            if existing.get(key) in (None, '', [], {}):
                existing[key] = value
        artifact_ref = str(existing.get('artifact_ref') or '').strip()
        ref = str(existing.get('ref') or '').strip()
        if artifact_ref and not ref:
            existing['ref'] = artifact_ref
        elif ref and not artifact_ref:
            existing['artifact_ref'] = ref
        return existing

    @classmethod
    def _artifact_record_identity_signature(cls, item: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(item.get('type') or item.get('kind') or '').strip().lower(),
            cls._artifact_record_path(item),
            str(item.get('source_path') or '').strip(),
            cls._artifact_record_extension(item),
        )

    @staticmethod
    def _repair_artifact_record_identity(item: Mapping[str, Any]) -> dict[str, Any]:
        repaired = dict(item or {})
        identity_payload = dict(repaired)
        for key in ('artifact_id', 'artifactId', 'id', 'artifact_ref', 'artifactRef', 'ref'):
            identity_payload.pop(key, None)
        canonical = sanitize_artifact_record(
            identity_payload,
            default_kind=str(repaired.get('type') or repaired.get('kind') or '').strip().lower() or None,
            default_origin=str(repaired.get('origin') or '').strip().lower() or None,
        )
        if not canonical:
            return repaired
        for key in ('artifact_id', 'artifact_ref', 'ref'):
            if canonical.get(key):
                repaired[key] = canonical[key]
        return repaired

    @classmethod
    def _repair_colliding_artifact_record_identities(
        cls,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        token_to_indexes: dict[tuple[str, str], list[int]] = {}
        for index, item in enumerate(records):
            if not isinstance(item, Mapping):
                continue
            if not cls._artifact_record_path(item):
                continue
            for field in ('artifact_id', 'artifact_ref', 'ref'):
                token = str(item.get(field) or '').strip()
                if token:
                    token_to_indexes.setdefault((field, token), []).append(index)

        indexes_to_repair: set[int] = set()
        for indexes in token_to_indexes.values():
            if len(indexes) < 2:
                continue
            signatures = {
                cls._artifact_record_identity_signature(records[index])
                for index in indexes
            }
            if len(signatures) > 1:
                indexes_to_repair.update(indexes)
        if not indexes_to_repair:
            return records
        return [
            cls._repair_artifact_record_identity(item)
            if index in indexes_to_repair
            else item
            for index, item in enumerate(records)
        ]

    @staticmethod
    def merge_unique_artifact_records(
        existing_values: Any,
        incoming_values: Any,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        index_by_key: dict[tuple[str, str, str], int] = {}
        for raw_item in (existing_values or []):
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            key = LateFillRuntimeOwner._artifact_record_merge_key(item)
            if key in index_by_key:
                LateFillRuntimeOwner._merge_artifact_record_fields(merged[index_by_key[key]], item)
                continue
            index_by_key[key] = len(merged)
            merged.append(item)
        for raw_item in (incoming_values or []):
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            key = LateFillRuntimeOwner._artifact_record_merge_key(item)
            if key in index_by_key:
                LateFillRuntimeOwner._merge_artifact_record_fields(merged[index_by_key[key]], item)
                continue
            index_by_key[key] = len(merged)
            merged.append(item)
        return LateFillRuntimeOwner._repair_colliding_artifact_record_identities(merged)

    @staticmethod
    def artifact_records_from_payload(
        payload: Any,
        key: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            return []
        records: list[dict[str, Any]] = []

        def add_records(raw_value: Any) -> None:
            if not isinstance(raw_value, list):
                return
            for raw_item in raw_value:
                if isinstance(raw_item, dict):
                    records.append(dict(raw_item))

        add_records(payload.get(key))
        request_payload = payload.get('request') if isinstance(payload.get('request'), Mapping) else {}
        add_records(request_payload.get(key))
        frame_payload = payload.get('response_frame') if isinstance(payload.get('response_frame'), Mapping) else {}
        frame_request = frame_payload.get('request') if isinstance(frame_payload.get('request'), Mapping) else {}
        add_records(frame_request.get(key))
        artifacts_payload = payload.get('artifacts') if isinstance(payload.get('artifacts'), Mapping) else {}
        if key == 'input_artifacts':
            add_records(artifacts_payload.get('input'))
        elif key == 'reference_artifacts':
            add_records(artifacts_payload.get('reference'))
        return records

    def merge_late_fill_payload_artifacts(
        self,
        late_fill_payload: dict[str, Any],
        *sources: Any,
    ) -> dict[str, Any]:
        updated = dict(late_fill_payload or {})
        for key in ('input_artifacts', 'reference_artifacts'):
            merged_records = self.merge_unique_artifact_records(updated.get(key), [])
            for source in sources:
                merged_records = self.merge_unique_artifact_records(
                    merged_records,
                    self.artifact_records_from_payload(source, key),
                )
            if merged_records:
                updated[key] = merged_records
        return updated

    @staticmethod
    def late_fill_text_from_result_payload(payload: Any) -> str:
        if not isinstance(payload, Mapping):
            return ''
        for key in (
            'content_payload',
            'content',
            'output_text',
            'result_text',
            'result',
            'text',
            'transcript',
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        result = payload.get('result')
        if isinstance(result, Mapping):
            for key in ('content', 'output_text', 'text', 'transcript'):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ''

    @staticmethod
    def late_fill_result_has_missing_dependency_evidence(payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        capability = str(payload.get('capability') or payload.get('expected_capability') or '').strip().lower()
        text = LateFillRuntimeOwner.late_fill_text_from_result_payload(payload)
        if capability in {'vision_analysis', 'speech_to_text'} and _PATH_ONLY_DEPENDENCY_EVIDENCE_RE.match(text):
            return True
        if not text:
            return False
        if capability in {'vision_analysis', 'speech_to_text'}:
            return bool(_DEPENDENCY_EVIDENCE_MISSING_CLAIM_RE.search(text))
        if capability == 'chat':
            depends_on = [
                str(item or '').strip()
                for item in (payload.get('depends_on') or payload.get('dependsOn') or [])
                if str(item or '').strip()
            ]
            depends_on_non_root = any(item != 'phase-1' for item in depends_on)
            return depends_on_non_root and bool(_DEPENDENCY_EVIDENCE_MISSING_CLAIM_RE.search(text))
        return False

    @staticmethod
    def late_fill_branch_dependency_ids(
        branch: Mapping[str, Any],
        *,
        current_payload: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        """Collect one branch's dependency aliases from contract and graph truth."""

        dependency_ids: list[str] = []

        def _add(raw_values: Any) -> None:
            values = (
                raw_values
                if isinstance(raw_values, (list, tuple, set))
                else [raw_values]
            )
            for value in values:
                token = str(value or '').strip()
                if token and token not in dependency_ids:
                    dependency_ids.append(token)

        _add(branch.get('depends_on') or branch.get('dependsOn'))
        execution_contract = (
            branch.get('execution_contract')
            if isinstance(branch.get('execution_contract'), Mapping)
            else {}
        )
        _add(
            execution_contract.get('depends_on')
            or execution_contract.get('dependencies')
        )
        if not isinstance(current_payload, Mapping):
            return dependency_ids

        runtime = (
            current_payload.get('runtime')
            if isinstance(current_payload.get('runtime'), Mapping)
            else {}
        )
        phase_graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        branch_tokens = {
            str(branch.get('branch_id') or '').strip(),
            str(branch.get('phase_id') or '').strip(),
            str(execution_contract.get('branch_id') or '').strip(),
            str(execution_contract.get('phase_id') or '').strip(),
        }
        branch_tokens.discard('')
        if not branch_tokens:
            return dependency_ids
        for key in ('phases', 'downstream_branches'):
            records = phase_graph.get(key) if isinstance(phase_graph.get(key), list) else []
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                record_tokens = {
                    str(record.get('branch_id') or '').strip(),
                    str(record.get('phase_id') or '').strip(),
                }
                record_tokens.discard('')
                if branch_tokens and not branch_tokens.intersection(record_tokens):
                    continue
                _add(record.get('depends_on') or record.get('dependsOn'))
        return dependency_ids

    @staticmethod
    def tts_source_evidence_from_effective_data(
        effective_data: Mapping[str, Any],
        *,
        infer_payload: Optional[Mapping[str, Any]] = None,
        execution_contract: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Freeze the exact branch-local text handed to the TTS backend."""

        effective = effective_data if isinstance(effective_data, Mapping) else {}
        infer = infer_payload if isinstance(infer_payload, Mapping) else {}
        contract = execution_contract if isinstance(execution_contract, Mapping) else {}
        infer_prompt = infer.get('prompt')
        source_text = str(infer_prompt) if infer_prompt is not None else ''
        source_authority = 'final_infer_prompt' if source_text.strip() else ''
        if not source_text.strip():
            effective_content = effective.get('content_payload')
            source_text = (
                str(effective_content) if effective_content is not None else ''
            )
            if source_text.strip():
                source_authority = 'effective_branch_content_payload'
        if not source_text.strip():
            return {}
        content_payload_source = str(
            effective.get('content_payload_source')
            or contract.get('content_payload_source')
            or ''
        ).strip()
        return build_tts_semantic_source(
            source_text,
            source_authority=source_authority,
            source_text_source=content_payload_source or source_authority,
            branch_id=(
                contract.get('branch_id') or effective.get('branch_id')
            ),
            phase_id=contract.get('phase_id') or effective.get('phase_id'),
            lang_code=(
                infer.get('lang_code')
                or effective.get('lang_code')
                or contract.get('lang_code')
            ),
        )

    @classmethod
    def tts_source_evidence_from_prepared_plan(
        cls,
        plan: Mapping[str, Any],
        *,
        infer_result: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = plan if isinstance(plan, Mapping) else {}
        planned_source = cls.tts_source_evidence_from_effective_data(
            payload.get('effective_data')
            if isinstance(payload.get('effective_data'), Mapping)
            else {},
            infer_payload=(
                payload.get('infer_payload')
                if isinstance(payload.get('infer_payload'), Mapping)
                else {}
            ),
            execution_contract=(
                payload.get('execution_contract')
                if isinstance(payload.get('execution_contract'), Mapping)
                else {}
            ),
        )
        result_payload = infer_result if isinstance(infer_result, Mapping) else {}
        runtime_source = (
            result_payload.get('tts_semantic_source')
            if isinstance(result_payload.get('tts_semantic_source'), Mapping)
            else {}
        )
        if not runtime_source:
            return planned_source
        runtime_source_text = str(
            runtime_source.get('tts_source_text') or ''
        )
        runtime_source_sha256 = str(
            runtime_source.get('tts_source_text_sha256') or ''
        ).strip().lower()
        runtime_source_contract_valid = bool(
            str(runtime_source.get('kind') or '').strip()
            == 'ollmo.tts_semantic_source'
            and runtime_source.get('version') == 1
            and str(runtime_source.get('policy_id') or '').strip()
            == TTS_SEMANTIC_SOURCE_POLICY_ID
            and str(runtime_source.get('authority') or '').strip()
            == 'runtime_exact_backend_prompt'
            and runtime_source_text.strip()
            and runtime_source_sha256
            == hashlib.sha256(runtime_source_text.encode('utf-8')).hexdigest()
        )
        if not runtime_source_contract_valid:
            return planned_source

        # The inference owner has observed the post-extraction, post-file-merge
        # prompt actually handed to MLX Audio. Preserve that text/digest and add
        # only branch-local bindings that inference cannot know.
        enriched_source = dict(runtime_source)
        for key in ('branch_id', 'phase_id', 'lang_code'):
            value = planned_source.get(key)
            if (
                enriched_source.get(key) in (None, '', [], {})
                and value not in (None, '', [], {})
            ):
                enriched_source[key] = value
        return enriched_source

    def tts_audio_integrity_evidence_for_branch_result(
        self,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return runtime-owned physical TTS evidence bound to the exact source."""

        if self.branch_capability(branch) != self.capability_text_to_speech:
            return {}
        result_payload = infer_result if isinstance(infer_result, Mapping) else {}
        source = (
            result_payload.get('tts_semantic_source')
            if isinstance(result_payload.get('tts_semantic_source'), Mapping)
            else {}
        )
        source_text = str(source.get('tts_source_text') or '')
        source_sha256 = str(source.get('tts_source_text_sha256') or '').strip()
        path = str(result_payload.get('saved_audio_path') or '').strip()
        existing = (
            result_payload.get('tts_audio_integrity_evidence')
            if isinstance(
                result_payload.get('tts_audio_integrity_evidence'),
                Mapping,
            )
            else {}
        )
        generation_budget = (
            result_payload.get('tts_generation_budget')
            if isinstance(result_payload.get('tts_generation_budget'), Mapping)
            else {}
        )
        tts_model_type = str(
            result_payload.get('tts_model_type')
            or generation_budget.get('tts_model_type')
            or ''
        ).strip()
        existing_contract_valid = bool(
            str(existing.get('kind') or '').strip()
            == 'ollmo.tts_audio_integrity_evidence'
            and existing.get('version') == 1
            and str(existing.get('policy_id') or '').strip()
            == TTS_AUDIO_INTEGRITY_POLICY_ID
            and str(existing.get('authority') or '').strip()
            == 'runtime_deterministic_audio_verification'
        )
        if existing_contract_valid:
            evidence = dict(existing)
            evidence_source_sha256 = str(
                evidence.get('source_sha256') or ''
            ).strip()
            evidence_path = str(evidence.get('artifact_path') or '').strip()
            if source_sha256 and evidence_source_sha256 != source_sha256:
                defect_codes = [
                    str(item).strip()
                    for item in (evidence.get('defect_codes') or [])
                    if str(item).strip()
                ] if isinstance(evidence.get('defect_codes'), list) else []
                if 'TTS_AUDIO_SOURCE_BINDING_MISMATCH' not in defect_codes:
                    defect_codes.insert(0, 'TTS_AUDIO_SOURCE_BINDING_MISMATCH')
                evidence.update(
                    {
                        'status': 'failed',
                        'reason_code': 'TTS_AUDIO_SOURCE_BINDING_MISMATCH',
                        'materialization_eligible': False,
                        'expected_source_sha256': source_sha256,
                        'defect_codes': defect_codes,
                    }
                )
            elif path and evidence_path and Path(path).expanduser() != Path(
                evidence_path
            ).expanduser():
                defect_codes = [
                    str(item).strip()
                    for item in (evidence.get('defect_codes') or [])
                    if str(item).strip()
                ] if isinstance(evidence.get('defect_codes'), list) else []
                if 'TTS_AUDIO_ARTIFACT_BINDING_MISMATCH' not in defect_codes:
                    defect_codes.insert(0, 'TTS_AUDIO_ARTIFACT_BINDING_MISMATCH')
                evidence.update(
                    {
                        'status': 'failed',
                        'reason_code': 'TTS_AUDIO_ARTIFACT_BINDING_MISMATCH',
                        'materialization_eligible': False,
                        'expected_artifact_path': path,
                        'defect_codes': defect_codes,
                    }
                )
            return evidence
        if not source_text:
            return {
                'kind': 'ollmo.tts_audio_integrity_evidence',
                'version': 1,
                'policy_id': TTS_AUDIO_INTEGRITY_POLICY_ID,
                'authority': 'runtime_deterministic_audio_verification',
                'status': 'unavailable',
                'reason_code': 'TTS_AUDIO_SOURCE_EVIDENCE_UNAVAILABLE',
                'materialization_eligible': False,
                'artifact_path': path or None,
                'defect_codes': ['TTS_AUDIO_SOURCE_EVIDENCE_UNAVAILABLE'],
            }
        return build_tts_audio_integrity_evidence(
            path,
            source_text,
            source_sha256=source_sha256 or None,
            generation_budget=generation_budget or None,
            model_family=(generation_budget.get('model_family') or None),
            tts_model_type=tts_model_type or None,
        )

    def tts_audio_integrity_error_for_branch_result(
        self,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        if self.branch_capability(branch) != self.capability_text_to_speech:
            return None
        evidence = self.tts_audio_integrity_evidence_for_branch_result(
            branch,
            infer_result,
        )
        if (
            str(evidence.get('status') or '').strip().lower() == 'passed'
            and evidence.get('materialization_eligible') is True
        ):
            return None
        reason_code = str(
            evidence.get('reason_code')
            or 'TTS_AUDIO_INTEGRITY_UNAVAILABLE'
        ).strip()
        defect_codes = [
            str(item).strip()
            for item in (evidence.get('defect_codes') or [])
            if str(item).strip()
        ] if isinstance(evidence.get('defect_codes'), list) else []
        if reason_code and reason_code not in defect_codes:
            defect_codes.insert(0, reason_code)
        saved_audio_path = str(
            (infer_result or {}).get('saved_audio_path') or ''
        ).strip()
        generation_budget = (
            (infer_result or {}).get('tts_generation_budget')
            if isinstance((infer_result or {}).get('tts_generation_budget'), Mapping)
            else {}
        )
        sampling_profile = (
            (infer_result or {}).get('tts_sampling_profile')
            if isinstance((infer_result or {}).get('tts_sampling_profile'), Mapping)
            else {}
        )
        return {
            'code': 'TTS_AUDIO_INTEGRITY_REPAIR_REQUIRED',
            'reason_code': reason_code,
            'defect_code': reason_code,
            'defect_codes': defect_codes,
            'message': (
                'Generated audio failed output-side integrity verification; '
                'the preserved file is diagnostic evidence, not fulfilled audio.'
            ),
            'stage': 'tts_audio_integrity_gate',
            'retryable': True,
            'repair_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
            'materialization_blocked': True,
            'blocked_scope': 'current_tts_branch',
            'repair_work_available': True,
            'repair_work_policy': (
                'bounded_same_response_tts_integrity_retry'
            ),
            'needs_external_input': False,
            'audio_integrity_evidence': evidence,
            'tts_generation_budget': dict(generation_budget) if generation_budget else None,
            'tts_sampling_profile': dict(sampling_profile) if sampling_profile else None,
            'diagnostic_artifact': (
                {
                    'type': 'audio',
                    'path': saved_audio_path,
                    'status': 'failed_integrity',
                }
                if saved_audio_path
                else None
            ),
        }

    @staticmethod
    def _normalize_tts_stt_semantic_text(content: Any) -> tuple[str, list[str]]:
        text = unicodedata.normalize('NFKC', str(content or '')).casefold()
        normalized_chars = [
            char
            if char.isalnum() or char.isspace()
            else ' '
            for char in text
        ]
        normalized = re.sub(r'\s+', ' ', ''.join(normalized_chars)).strip()
        tokens = re.findall(r'[^\W_]+', normalized, flags=re.UNICODE)
        return ' '.join(tokens), tokens

    @classmethod
    def _tts_stt_similarity_metrics(
        cls,
        source_text: Any,
        transcript_text: Any,
    ) -> dict[str, Any]:
        normalized_source, source_tokens = cls._normalize_tts_stt_semantic_text(
            source_text
        )
        normalized_transcript, transcript_tokens = cls._normalize_tts_stt_semantic_text(
            transcript_text
        )
        source_counter = Counter(source_tokens)
        transcript_counter = Counter(transcript_tokens)
        overlap_count = sum((source_counter & transcript_counter).values())
        source_negation_count = sum(
            count
            for token, count in source_counter.items()
            if token in _TTS_STT_NEGATION_TOKENS
        )
        transcript_negation_count = sum(
            count
            for token, count in transcript_counter.items()
            if token in _TTS_STT_NEGATION_TOKENS
        )
        negation_consistent = source_negation_count == transcript_negation_count
        token_recall = overlap_count / len(source_tokens) if source_tokens else 0.0
        token_precision = (
            overlap_count / len(transcript_tokens) if transcript_tokens else 0.0
        )
        token_f1 = (
            (2 * token_precision * token_recall) / (token_precision + token_recall)
            if token_precision + token_recall
            else 0.0
        )
        sequence_ratio = (
            SequenceMatcher(
                None,
                normalized_source,
                normalized_transcript,
                autojunk=False,
            ).ratio()
            if normalized_source and normalized_transcript
            else 0.0
        )
        exact_match = bool(
            normalized_source
            and normalized_source == normalized_transcript
        )
        if exact_match:
            semantic_match = True
        elif len(source_tokens) < _TTS_STT_MIN_SOURCE_TOKENS_FOR_FULL_POLICY:
            semantic_match = bool(
                token_f1 >= _TTS_STT_SHORT_MIN_TOKEN_F1
                and sequence_ratio >= _TTS_STT_SHORT_MIN_SEQUENCE_RATIO
                and negation_consistent
            )
        else:
            semantic_match = bool(
                token_recall >= _TTS_STT_MIN_TOKEN_RECALL
                and token_precision >= _TTS_STT_MIN_TOKEN_PRECISION
                and sequence_ratio >= _TTS_STT_MIN_SEQUENCE_RATIO
                and negation_consistent
            )
        return {
            'normalized_source': normalized_source,
            'normalized_transcript': normalized_transcript,
            'source_token_count': len(source_tokens),
            'transcript_token_count': len(transcript_tokens),
            'overlap_token_count': overlap_count,
            'token_recall': round(token_recall, 6),
            'token_precision': round(token_precision, 6),
            'token_f1': round(token_f1, 6),
            'sequence_ratio': round(sequence_ratio, 6),
            'source_negation_count': source_negation_count,
            'transcript_negation_count': transcript_negation_count,
            'negation_consistent': negation_consistent,
            'exact_match': exact_match,
            'semantic_match': semantic_match,
        }

    def tts_stt_semantic_evidence_for_branch_result(
        self,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        *,
        current_payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.normalize_capability(branch.get('capability')) != 'speech_to_text':
            return {
                'kind': 'ollmo.tts_stt_semantic_evidence',
                'version': 1,
                'status': 'not_applicable',
                'policy_id': _TTS_STT_SEMANTIC_POLICY_ID,
                'reason_code': 'CONSUMER_IS_NOT_SPEECH_TO_TEXT',
            }
        dependency_ids = self.late_fill_branch_dependency_ids(
            branch,
            current_payload=current_payload,
        )
        if not dependency_ids or not isinstance(current_payload, Mapping):
            return {
                'kind': 'ollmo.tts_stt_semantic_evidence',
                'version': 1,
                'status': 'not_applicable',
                'policy_id': _TTS_STT_SEMANTIC_POLICY_ID,
                'reason_code': 'NO_RUNTIME_TTS_DEPENDENCY_CONTEXT',
            }
        late_fill = (
            current_payload.get('late_fill')
            if isinstance(current_payload.get('late_fill'), Mapping)
            else {}
        )
        runtime = (
            current_payload.get('runtime')
            if isinstance(current_payload.get('runtime'), Mapping)
            else {}
        )
        phase_graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        declared_records: list[Mapping[str, Any]] = []
        for key in ('phases', 'downstream_branches'):
            declared_records.extend(
                item
                for item in (
                    phase_graph.get(key)
                    if isinstance(phase_graph.get(key), list)
                    else []
                )
                if isinstance(item, Mapping)
            )
        for key in ('completed_branches', 'pending_branches', 'failed_branches'):
            declared_records.extend(
                item
                for item in (
                    late_fill.get(key) if isinstance(late_fill.get(key), list) else []
                )
                if isinstance(item, Mapping)
            )
        declared_tts_dependency_ids: list[str] = []
        for record in declared_records:
            if (
                self.normalize_capability(record.get('capability'))
                != self.capability_text_to_speech
            ):
                continue
            record_tokens = {
                str(record.get('branch_id') or '').strip(),
                str(record.get('phase_id') or '').strip(),
            }
            record_tokens.discard('')
            for dependency_id in dependency_ids:
                if dependency_id in record_tokens and dependency_id not in declared_tts_dependency_ids:
                    declared_tts_dependency_ids.append(dependency_id)
        producer_results: list[Mapping[str, Any]] = []
        for result in late_fill.get('fill_results') or []:
            if not isinstance(result, Mapping):
                continue
            if self.normalize_capability(result.get('capability')) != self.capability_text_to_speech:
                continue
            result_tokens = {
                str(result.get('branch_id') or '').strip(),
                str(result.get('phase_id') or '').strip(),
            }
            if not result_tokens.intersection(dependency_ids):
                continue
            if not any(dict(result) == dict(existing) for existing in producer_results):
                producer_results.append(result)
        consumer_branch_id = str(
            branch.get('branch_id') or branch.get('phase_id') or ''
        ).strip()
        consumer_phase_id = str(branch.get('phase_id') or '').strip()
        base_evidence = {
            'kind': 'ollmo.tts_stt_semantic_evidence',
            'version': 1,
            'policy_id': _TTS_STT_SEMANTIC_POLICY_ID,
            'authority': 'runtime_deterministic_verification',
            'consumer_branch_id': consumer_branch_id or None,
            'consumer_phase_id': consumer_phase_id or None,
        }
        if not producer_results:
            if declared_tts_dependency_ids:
                return {
                    **base_evidence,
                    'status': 'unavailable',
                    'reason_code': 'TTS_PRODUCER_RESULT_UNAVAILABLE',
                    'semantic_match': False,
                    'failed_dependency_ids': declared_tts_dependency_ids,
                }
            return {
                **base_evidence,
                'status': 'not_applicable',
                'reason_code': 'NO_DIRECT_TTS_PRODUCER_RESULT',
            }
        if len(producer_results) != 1:
            return {
                **base_evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_PRODUCER_EVIDENCE_AMBIGUOUS',
                'producer_result_count': len(producer_results),
                'semantic_match': False,
            }
        producer = producer_results[0]
        source = (
            producer.get('tts_semantic_source')
            if isinstance(producer.get('tts_semantic_source'), Mapping)
            else {}
        )
        producer_branch_id = str(
            producer.get('branch_id') or source.get('branch_id') or ''
        ).strip()
        producer_phase_id = str(
            producer.get('phase_id') or source.get('phase_id') or ''
        ).strip()
        evidence = {
            **base_evidence,
            'producer_branch_id': producer_branch_id or None,
            'producer_phase_id': producer_phase_id or None,
        }
        if not source:
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_SOURCE_EVIDENCE_UNAVAILABLE',
                'semantic_match': False,
            }
        if (
            str(source.get('kind') or '').strip() != 'ollmo.tts_semantic_source'
            or source.get('version') != 1
            or str(source.get('policy_id') or '').strip()
            != _TTS_STT_SEMANTIC_POLICY_ID
            or str(source.get('authority') or '').strip()
            != 'runtime_exact_backend_prompt'
            or str(source.get('source_authority') or '').strip()
            != 'final_infer_prompt'
        ):
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_SOURCE_EVIDENCE_CONTRACT_INVALID',
                'semantic_match': False,
            }
        source_branch_id = str(source.get('branch_id') or '').strip()
        source_phase_id = str(source.get('phase_id') or '').strip()
        if (
            (source_branch_id and producer_branch_id and source_branch_id != producer_branch_id)
            or (source_phase_id and producer_phase_id and source_phase_id != producer_phase_id)
        ):
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_SOURCE_EVIDENCE_BINDING_MISMATCH',
                'semantic_match': False,
            }
        raw_source_text = source.get('tts_source_text')
        source_text = str(raw_source_text) if raw_source_text is not None else ''
        if not source_text.strip():
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_SOURCE_EVIDENCE_UNAVAILABLE',
                'semantic_match': False,
            }
        source_sha256 = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
        declared_source_sha256 = str(
            source.get('tts_source_text_sha256') or ''
        ).strip().lower()
        if not re.fullmatch(r'[0-9a-f]{64}', declared_source_sha256):
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_SOURCE_EVIDENCE_DIGEST_UNAVAILABLE',
                'source_sha256': source_sha256,
                'semantic_match': False,
            }
        if declared_source_sha256 != source_sha256:
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_SOURCE_EVIDENCE_DIGEST_MISMATCH',
                'source_sha256': source_sha256,
                'declared_source_sha256': declared_source_sha256,
                'semantic_match': False,
            }
        transcript_text = self.late_fill_text_from_result_payload(infer_result)
        if not transcript_text:
            return {
                **evidence,
                'status': 'unavailable',
                'reason_code': 'TTS_TRANSCRIPT_EVIDENCE_UNAVAILABLE',
                'source_sha256': source_sha256,
                'semantic_match': False,
            }
        transcript_sha256 = hashlib.sha256(
            transcript_text.encode('utf-8')
        ).hexdigest()
        metrics = self._tts_stt_similarity_metrics(source_text, transcript_text)
        result_payload = (
            infer_result.get('result')
            if isinstance(infer_result.get('result'), Mapping)
            else {}
        )
        expected_lang_code = str(
            source.get('lang_code')
            or producer.get('lang_code')
            or ''
        ).strip().lower()
        detected_lang_code = str(
            infer_result.get('language')
            or result_payload.get('language')
            or ''
        ).strip().lower()
        return {
            key: value
            for key, value in {
                **evidence,
                'status': 'matched' if metrics['semantic_match'] else 'mismatched',
                'reason_code': (
                    'TTS_STT_SEMANTIC_MATCH'
                    if metrics['semantic_match']
                    else 'TTS_STT_SEMANTIC_MISMATCH'
                ),
                'semantic_match': bool(metrics['semantic_match']),
                'source_sha256': source_sha256,
                'transcript_sha256': transcript_sha256,
                'transcript_text': transcript_text,
                'expected_lang_code': expected_lang_code or None,
                'detected_lang_code': detected_lang_code or None,
                'metrics': metrics,
                'thresholds': {
                    'min_source_tokens_for_full_policy': (
                        _TTS_STT_MIN_SOURCE_TOKENS_FOR_FULL_POLICY
                    ),
                    'min_token_recall': _TTS_STT_MIN_TOKEN_RECALL,
                    'min_token_precision': _TTS_STT_MIN_TOKEN_PRECISION,
                    'min_sequence_ratio': _TTS_STT_MIN_SEQUENCE_RATIO,
                    'short_min_token_f1': _TTS_STT_SHORT_MIN_TOKEN_F1,
                    'short_min_sequence_ratio': _TTS_STT_SHORT_MIN_SEQUENCE_RATIO,
                },
            }.items()
            if value not in (None, '', [], {})
        }

    def dependency_evidence_error_for_branch_result(
        self,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        *,
        current_payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        capability = self.branch_capability(branch)
        result_payload = dict(infer_result or {})
        if capability and not result_payload.get('capability'):
            result_payload['capability'] = capability
        if not result_payload.get('depends_on') and branch.get('depends_on'):
            result_payload['depends_on'] = list(branch.get('depends_on') or [])
        if self.late_fill_result_has_missing_dependency_evidence(result_payload):
            return {
                'code': 'DEPENDENCY_CHAIN_REPAIR_REQUIRED',
                'message': 'Dependency evidence repair required: the branch did not receive or use the artifact evidence it depended on.',
                'stage': 'semantic_evidence_gate',
                'retryable': False,
            }
        audio_integrity_error = self.tts_audio_integrity_error_for_branch_result(
            branch,
            result_payload,
        )
        if audio_integrity_error:
            return audio_integrity_error
        semantic_evidence = (
            result_payload.get('tts_stt_semantic_evidence')
            if isinstance(result_payload.get('tts_stt_semantic_evidence'), Mapping)
            else self.tts_stt_semantic_evidence_for_branch_result(
                branch,
                result_payload,
                current_payload=current_payload,
            )
        )
        semantic_status = str(semantic_evidence.get('status') or '').strip().lower()
        if semantic_status not in {'mismatched', 'unavailable'}:
            return None
        reason_code = str(
            semantic_evidence.get('reason_code')
            or 'TTS_STT_SEMANTIC_EVIDENCE_UNAVAILABLE'
        ).strip()
        failed_dependency_ids = [
            str(item or '').strip()
            for item in (semantic_evidence.get('failed_dependency_ids') or [])
            if str(item or '').strip()
        ] if isinstance(semantic_evidence.get('failed_dependency_ids'), list) else []
        producer_phase_id = str(
            semantic_evidence.get('producer_phase_id') or ''
        ).strip()
        if producer_phase_id and producer_phase_id not in failed_dependency_ids:
            failed_dependency_ids.append(producer_phase_id)
        return {
            'code': 'DEPENDENCY_CHAIN_REPAIR_REQUIRED',
            'reason_code': reason_code,
            'defect_code': reason_code,
            'message': (
                'Audio transcript semantic evidence does not match the exact text '
                'handed to its TTS producer.'
                if semantic_status == 'mismatched'
                else 'Audio transcript semantic evidence cannot be verified from the exact TTS producer source.'
            ),
            'stage': 'semantic_evidence_gate',
            'retryable': False,
            'repair_action': RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
            'materialization_blocked': True,
            'failed_dependency_ids': [
                item for item in failed_dependency_ids if item
            ],
            'semantic_evidence': dict(semantic_evidence),
        }

    @staticmethod
    def _late_fill_plain_markdown_line(content: Any) -> str:
        text = str(content or '').strip()
        if not text:
            return ''
        text = re.sub(r'^#{1,6}\s*', '', text)
        text = re.sub(r'^[-+*]\s+', '', text)
        text = text.replace('**', '').replace('__', '').replace('`', '')
        return text.strip()

    @staticmethod
    def _late_fill_label_has_speakable_token(label: Any) -> bool:
        normalized = re.sub(r'\s+', ' ', str(label or '').casefold()).strip()
        if not normalized:
            return False
        return any(
            re.search(
                rf'(?<!\w){re.escape(token)}(?!\w)',
                normalized,
                flags=re.UNICODE,
            )
            for token in _AUDIO_SINGLE_SPEAKABLE_SECTION_TOKENS
        )

    @classmethod
    def _trim_late_fill_numbered_candidate_body(cls, content: Any) -> str:
        """Exclude non-authoritative transcript/JSON siblings after a candidate."""

        retained: list[str] = []
        for raw_line in str(content or '').splitlines():
            if retained and raw_line.strip().startswith('```'):
                break
            plain = cls._late_fill_plain_markdown_line(raw_line)
            field_match = re.match(r'^(?P<label>[^:\n]{1,120})\s*:', plain)
            if retained and field_match:
                label = field_match.group('label').casefold()
                if re.search(
                    r'(?<!\w)(?:json|transcript|transkript)(?!\w)',
                    label,
                    flags=re.UNICODE,
                ):
                    break
            retained.append(raw_line)
        return re.sub(r'\s+', ' ', '\n'.join(retained)).strip()

    @classmethod
    def _late_fill_labeled_audio_candidate_contract(cls, content: Any) -> dict[str, Any]:
        """Extract explicitly indexed audio bodies without trusting sibling prose.

        Once an audio-variant heading is present, the labelled slots are the only
        candidate authority. Missing, repeated, or ambiguous slots therefore stay
        fail-closed instead of falling back to paragraph order.
        """

        text = str(content or '').strip()
        if not text:
            return {
                'marker_detected': False,
                'source': 'empty',
                'units': [],
            }
        lines = text.splitlines()
        headers: list[tuple[int, int]] = []
        for line_index, line in enumerate(lines):
            plain = cls._late_fill_plain_markdown_line(line)
            match = _AUDIO_VARIANT_HEADER_RE.fullmatch(plain)
            if match:
                headers.append((line_index, int(match.group('index'))))
        if not headers:
            return {
                'marker_detected': False,
                'source': 'no_labeled_audio_variant_sections',
                'units': [],
            }

        indexes = [index for _line_index, index in headers]
        if len(set(indexes)) != len(indexes):
            return {
                'marker_detected': True,
                'source': 'labeled_audio_variant_sections',
                'units': [],
                'issue': 'duplicate_audio_variant_index',
                'indexes': indexes,
            }
        expected_indexes = list(range(1, len(indexes) + 1))
        if sorted(indexes) != expected_indexes:
            return {
                'marker_detected': True,
                'source': 'labeled_audio_variant_sections',
                'units': [],
                'issue': 'non_contiguous_candidate_indexes',
                'indexes': indexes,
            }

        bodies_by_index: dict[int, str] = {}
        for header_offset, (line_index, variant_index) in enumerate(headers):
            end_line = (
                headers[header_offset + 1][0]
                if header_offset + 1 < len(headers)
                else len(lines)
            )
            section_lines = lines[line_index + 1:end_line]
            speakable_bodies: list[str] = []
            for section_line_index, raw_line in enumerate(section_lines):
                plain = cls._late_fill_plain_markdown_line(raw_line)
                field_match = re.match(
                    r'^(?P<label>[^:\n]{1,80})\s*:\s*(?P<body>.*)$',
                    plain,
                )
                if not field_match:
                    continue
                label = re.sub(
                    r'\s+',
                    ' ',
                    field_match.group('label').strip().casefold(),
                )
                if label not in _AUDIO_SPEAKABLE_FIELD_LABELS:
                    continue
                body = field_match.group('body').strip()
                if not body:
                    continuation: list[str] = []
                    for continuation_line in section_lines[section_line_index + 1:]:
                        continuation_plain = cls._late_fill_plain_markdown_line(
                            continuation_line
                        )
                        if not continuation_plain:
                            if continuation:
                                break
                            continue
                        if re.match(r'^[^:\n]{1,80}\s*:', continuation_plain):
                            break
                        if continuation_plain.startswith(('```', '{', '[')):
                            break
                        continuation.append(continuation_plain)
                    body = ' '.join(continuation).strip()
                body = re.sub(r'\s+', ' ', body).strip()
                if body:
                    speakable_bodies.append(body)
            if len(speakable_bodies) != 1:
                return {
                    'marker_detected': True,
                    'source': 'labeled_audio_variant_sections',
                    'units': [],
                    'issue': (
                        'missing_audio_variant_body'
                        if not speakable_bodies
                        else 'ambiguous_audio_variant_body'
                    ),
                    'indexes': indexes,
                    'variant_index': variant_index,
                }
            bodies_by_index[variant_index] = speakable_bodies[0]

        return {
            'marker_detected': True,
            'source': 'labeled_audio_variant_sections',
            'units': [bodies_by_index[index] for index in expected_indexes],
            'indexes': expected_indexes,
        }

    @classmethod
    def _late_fill_candidate_selection_contract(cls, content: Any) -> dict[str, Any]:
        text = str(content or '').strip()
        if not text:
            return {'source': 'empty', 'units': []}
        labeled = cls._late_fill_labeled_audio_candidate_contract(text)
        if labeled.get('marker_detected'):
            return labeled

        numbered_matches = list(
            re.finditer(
                r'(?ms)^\s*(?P<index>\d+)[.)]\s*(?P<body>.*?)'
                r'(?=^\s*\d+[.)]\s+|\Z)',
                text,
            )
        )
        if numbered_matches:
            indexes = [int(match.group('index')) for match in numbered_matches]
            expected_indexes = list(range(1, len(indexes) + 1))
            if indexes != expected_indexes:
                return {
                    'source': 'numbered_candidate_bodies',
                    'units': [],
                    'indexes': indexes,
                    'issue': 'non_contiguous_candidate_indexes',
                }
            numbered: list[str] = []
            for match in numbered_matches:
                body = cls._trim_late_fill_numbered_candidate_body(
                    match.group('body')
                )
                if body:
                    numbered.append(body)
            if len(numbered) != len(numbered_matches):
                return {
                    'source': 'numbered_candidate_bodies',
                    'units': [],
                    'indexes': indexes,
                    'issue': 'missing_candidate_body',
                }
            return {
                'source': 'numbered_candidate_bodies',
                'units': numbered,
                'indexes': indexes,
            }
        return {
            'source': 'generic_content_units',
            'units': cls._split_late_fill_content_units(text),
        }

    @classmethod
    def _late_fill_candidate_units_for_selection(cls, content: Any) -> list[str]:
        contract = cls._late_fill_candidate_selection_contract(content)
        if contract.get('issue'):
            return []
        return [
            str(unit or '').strip()
            for unit in (contract.get('units') or [])
            if str(unit or '').strip()
        ]

    @classmethod
    def _late_fill_labeled_speakable_sections(cls, content: Any) -> list[str]:
        candidates: list[str] = []
        lines = str(content or '').splitlines()
        for line_index, raw_line in enumerate(lines):
            plain = cls._late_fill_plain_markdown_line(raw_line)
            match = re.match(
                r'^(?P<label>[^:\n]{1,120})\s*:\s*(?P<body>.*)$',
                plain,
            )
            if not match:
                continue
            label = re.sub(r'\s+', ' ', match.group('label').casefold()).strip()
            if any(token in label for token in ('transcript', 'transkript', 'json')):
                continue
            if not cls._late_fill_label_has_speakable_token(label):
                continue
            body_parts = [match.group('body').strip()] if match.group('body').strip() else []
            for continuation_line in lines[line_index + 1:]:
                if continuation_line.strip().startswith('```'):
                    break
                continuation_plain = cls._late_fill_plain_markdown_line(
                    continuation_line
                )
                if re.match(r'^[^:\n]{1,120}\s*:', continuation_plain):
                    break
                if continuation_plain:
                    body_parts.append(continuation_plain)
            body = re.sub(r'\s+', ' ', ' '.join(body_parts)).strip()
            if body:
                candidates.append(body)
        return candidates

    @staticmethod
    def _select_late_fill_candidate_unit(content: Any, index: int = 1) -> str:
        units = LateFillRuntimeOwner._late_fill_candidate_units_for_selection(content)
        if not units:
            return ''
        if isinstance(index, bool):
            return ''
        if isinstance(index, int):
            normalized_index = index
        elif (
            isinstance(index, str)
            and index.isascii()
            and index.isdigit()
            and 0 < len(index) <= 6
        ):
            normalized_index = int(index)
        else:
            return ''
        if normalized_index <= 0:
            return ''
        if normalized_index > len(units):
            return ''
        return units[normalized_index - 1].strip()

    @staticmethod
    def _branch_prompt_selection_index(branch: Mapping[str, Any], candidate_count: int) -> int:
        try:
            normalized_count = int(candidate_count)
        except (TypeError, ValueError):
            normalized_count = 0
        if normalized_count <= 0:
            return 0

        def _coerce_index(value: Any) -> int:
            try:
                index = int(value)
            except (TypeError, ValueError):
                return 0
            return index if 1 <= index <= normalized_count else 0

        for key in (
            'candidate_selection_index',
            'prompt_selection_index',
            'image_prompt_index',
            'artifact_prompt_index',
        ):
            raw_value = (branch or {}).get(key)
            try:
                explicit_index = int(raw_value)
            except (TypeError, ValueError):
                explicit_index = 0
            if explicit_index > normalized_count:
                return 0
            index = _coerce_index(raw_value)
            if index:
                return index

        for key in (
            'branch_id',
            'phase_id',
            'slot_id',
            'obligation_id',
            'task_id',
            'workload_task_id',
        ):
            token = str((branch or {}).get(key) or '').strip()
            if 'image' not in token.lower():
                continue
            match = re.search(r'(\d+)\s*$', token)
            if not match:
                continue
            if int(match.group(1)) > normalized_count:
                return 0
            index = _coerce_index(match.group(1))
            if index:
                return index

        raw_queue_index = (branch or {}).get('queue_index')
        try:
            explicit_queue_index = int(raw_queue_index)
        except (TypeError, ValueError):
            explicit_queue_index = 0
        if explicit_queue_index > normalized_count:
            return 0
        queue_index = _coerce_index(raw_queue_index)
        if queue_index:
            return queue_index
        return 1

    def _image_prompt_batch_expected_count(
        self,
        *,
        branch: Mapping[str, Any],
        branch_gap: Mapping[str, Any],
        current_payload: Mapping[str, Any],
    ) -> int:
        """Resolve the original image cohort size, not just the current retry wave."""

        def coerce_count(value: Any) -> int:
            try:
                count = int(value)
            except (TypeError, ValueError):
                return 0
            return count if 2 <= count <= 26 else 0

        current_late_fill = (
            current_payload.get('late_fill')
            if isinstance(current_payload.get('late_fill'), Mapping)
            else {}
        )
        runtime = (
            current_payload.get('runtime')
            if isinstance(current_payload.get('runtime'), Mapping)
            else {}
        )
        phase_graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        graph_phases = phase_graph.get('phases') if isinstance(phase_graph.get('phases'), list) else []
        target_dependencies = tuple(sorted({
            str(item or '').strip()
            for item in (branch.get('depends_on') or [])
            if str(item or '').strip()
        }))
        if not target_dependencies:
            target_tokens = {
                str(branch.get('branch_id') or '').strip(),
                str(branch.get('phase_id') or '').strip(),
            }
            target_tokens.discard('')
            for phase in graph_phases:
                if not isinstance(phase, Mapping):
                    continue
                phase_tokens = {
                    str(phase.get('branch_id') or '').strip(),
                    str(phase.get('phase_id') or '').strip(),
                }
                if not target_tokens.intersection(phase_tokens):
                    continue
                target_dependencies = tuple(sorted({
                    str(item or '').strip()
                    for item in (phase.get('depends_on') or [])
                    if str(item or '').strip()
                }))
                break

        lifecycle_records: list[Mapping[str, Any]] = []
        for source in (branch_gap, current_late_fill):
            for key in (
                'pending_branches',
                'active_branches',
                'completed_branches',
                'failed_branches',
                'cancelled_branches',
            ):
                lifecycle_records.extend(
                    item
                    for item in (source.get(key) or [])
                    if isinstance(item, Mapping)
                )
        lifecycle_records.extend(
            item for item in graph_phases if isinstance(item, Mapping)
        )
        cohort_indexes: dict[str, int] = {}
        if target_dependencies:
            for item in lifecycle_records:
                capability = self.normalize_capability(item.get('capability'))
                output_type = str(item.get('output_type') or '').strip().lower()
                if capability != self.capability_image_generation and output_type != 'image':
                    continue
                item_dependencies = tuple(sorted({
                    str(value or '').strip()
                    for value in (item.get('depends_on') or [])
                    if str(value or '').strip()
                }))
                if item_dependencies != target_dependencies:
                    continue
                try:
                    queue_index = int(item.get('queue_index') or 0)
                except (TypeError, ValueError):
                    continue
                if not 1 <= queue_index <= 26:
                    continue
                identity = str(
                    item.get('branch_id')
                    or item.get('phase_id')
                    or f'queue:{queue_index}'
                ).strip()
                if identity:
                    cohort_indexes[identity] = queue_index
        ordered_indexes = sorted(set(cohort_indexes.values()))
        if (
            len(cohort_indexes) >= 2
            and len(ordered_indexes) == len(cohort_indexes)
            and ordered_indexes == list(range(1, len(ordered_indexes) + 1))
        ):
            return len(ordered_indexes)

        for source in (branch_gap, current_late_fill):
            count = coerce_count(source.get('batch_prompt_expected_count'))
            source_phase_id = str(source.get('batch_prompt_source_phase_id') or '').strip()
            if (
                count
                and source_phase_id
                and target_dependencies == (source_phase_id,)
            ):
                return count

        batch_prompts = branch_gap.get('batch_prompts')
        raw_batch_count = len([
            item
            for item in (batch_prompts if isinstance(batch_prompts, list) else [])
            if str(item or '').strip()
        ])
        prompt_intent = (
            phase_graph.get('prompt_intent')
            if isinstance(phase_graph.get('prompt_intent'), Mapping)
            else {}
        )
        requested_count = coerce_count(prompt_intent.get('requested_visual_output_count'))
        if requested_count and requested_count == raw_batch_count:
            return requested_count

        pending_branches = branch_gap.get('pending_branches')
        pending_count = len([
            item
            for item in (pending_branches if isinstance(pending_branches, list) else [])
            if isinstance(item, Mapping)
            and (
                self.normalize_capability(item.get('capability')) == self.capability_image_generation
                or str(item.get('output_type') or '').strip().lower() == 'image'
            )
        ])
        return coerce_count(pending_count)

    @staticmethod
    def _image_branch_prompt_allows_batch_prompt_override(
        branch: Mapping[str, Any],
        branch_gap: Mapping[str, Any],
    ) -> bool:
        prompt = str((branch or {}).get('artifact_prompt') or '').strip()
        if not prompt:
            return True
        source = str(
            (branch or {}).get('artifact_prompt_source')
            or (branch_gap or {}).get('artifact_prompt_source')
            or ''
        ).strip()
        authoritative_sources = {
            'semantic_batch_prompt',
            'semantic_batch_prompts',
            'semantic_prepare_phase_output',
            'current_turn_direct_image_clause',
            'action_input',
            'prompt_blockquote_section',
            'quoted_prompt_section',
            'inline_prompt_capsule',
            'focused_image_prompt_slot',
            'focused_content_payload',
            'request_prompt_image_slots',
            'current_turn_explicit_image_manifest',
        }
        if source in authoritative_sources:
            return False
        normalized = re.sub(r'[`*]+', '', prompt).strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        if not normalized:
            return True
        mentions_image_generation = 'image_generation' in normalized or 'image generation' in normalized
        asks_for_assets = bool(re.search(r'\b(?:visual\s+)?assets?\b|\bimages?\b', normalized))
        generic_batch_shape = bool(
            re.search(r'\bgenerate\s+\d+\b', normalized)
            or 'distinct' in normalized
            or 'manifest' in normalized
        )
        manifest_based = 'manifest' in normalized and 'based on' in normalized
        return bool(
            mentions_image_generation
            and asks_for_assets
            and (generic_batch_shape or manifest_based)
        )

    @staticmethod
    def _split_late_fill_content_units(content: Any) -> list[str]:
        text = str(content or '').strip()
        if not text:
            return []
        units = [
            re.sub(r'\s+', ' ', item).strip()
            for item in re.split(r'\n\s*\n+', text)
            if re.sub(r'\s+', ' ', item).strip()
        ]
        if len(units) <= 1:
            units = [
                re.sub(r'\s+', ' ', item).strip()
                for item in re.split(r'(?m)^\s*(?:[-*]|\d+[.)])\s+', text)
                if re.sub(r'\s+', ' ', item).strip()
            ]
        if len(units) <= 1 and re.search(
            r'\b(?:image\s+prompt|visual prompt|poster|poster image|graphic design|generated artifacts?|final sentence)\b',
            text,
            flags=re.IGNORECASE,
        ):
            units = [
                re.sub(r'\s+', ' ', item).strip()
                for item in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
                if re.sub(r'\s+', ' ', item).strip()
            ]
        return units

    @staticmethod
    def _strip_late_fill_prompt_label(content: str) -> str:
        return re.sub(
            r'(?is)^\s*(?:[#>*\-\d.)\s]*)(?:(?:(?:image|visual|poster|bild)\s*[- ]?\s*)?prompt|bildprompt)'
            r'(?:\s*(?:for|idee|idea|bild|poster)?\s*\d*)?\s*[:\-]\s*',
            '',
            str(content or '').strip(),
        ).strip()

    @staticmethod
    def _late_fill_image_prompt_is_code_polluted(content: Any) -> bool:
        text = str(content or '').strip()
        if not text:
            return False
        return _image_prompt_candidate_is_code_or_css_polluted(text)

    @staticmethod
    def _late_fill_image_prompt_is_page_layout_instruction(content: Any) -> bool:
        text = str(content or '').strip()
        if not text:
            return False
        cleaned = re.sub(r'[`*#_]+', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        if not cleaned:
            return False
        layout_cues = {
            'background',
            'button',
            'cta',
            'headline',
            'hero section',
            'layout',
            'navigation',
            'section',
            'subheadline',
            'text',
            'text overlay',
            'visual',
        }
        cue_hits = sum(1 for cue in layout_cues if cue in cleaned)
        if cue_hits < 2:
            return False
        if re.search(r'\b(?:hero|about us|craftsmanship|portfolio|featured work|features?|gallery|contact|footer)\s+section\b', cleaned):
            return True
        if re.search(r'\b(?:layout|headline|subheadline|text overlay|text \(|visual \(|cta|navigation)\s*:', cleaned):
            return True
        if re.search(r'\b(?:using|implementation of|display of)\s+(?:the\s+)?[a-z0-9 \"\'-]*\bimage\b\s*\(?image\s*\d+\)?', cleaned):
            return True
        return False

    @staticmethod
    def _extract_late_fill_image_prompt_units(content: Any) -> list[str]:
        inline_prompts = _extract_inline_labeled_image_prompt_lines(str(content or ''))
        if inline_prompts:
            return inline_prompts
        units = LateFillRuntimeOwner._split_late_fill_content_units(content)
        prompts: list[str] = []
        for unit in units:
            raw_unit = str(unit or '').strip()
            if not raw_unit:
                continue
            if LateFillRuntimeOwner._late_fill_image_prompt_is_code_polluted(raw_unit):
                if prompts:
                    break
                continue
            lowered = raw_unit.lower()
            if re.search(
                r'\b('
                r'html\s+materialization|css\s+design|structure\s*&\s*elements|'
                r'document\s+metadata|asset\s+links|visual\s+theme|color\s+palette|'
                r'typography|layout\s*&\s*composition'
                r')\b',
                lowered,
            ):
                if prompts:
                    break
                continue
            prompt_like = (
                re.search(r'\b(?:image\s+generation\s+)?prompts?\s*\d*\s*[:\-]', lowered)
                or re.search(r'\b(?:image|visual|poster|bild)\s*[- ]?\s*prompt\s*\d*\s*[:\-]', lowered)
                or re.search(r'\bprompt\s*(?:\d+|[a-z])\s*[:\-]', lowered)
                or re.search(r'^\s*(?:\d+[.)]\s*)?[^:\n]{0,96}\bimage\b\s*[:\-]', raw_unit, re.IGNORECASE)
            )
            if not prompt_like:
                continue
            cleaned = LateFillRuntimeOwner._strip_late_fill_prompt_label(raw_unit)
            cleaned = re.sub(r'\*\*+', '', cleaned).strip()
            cleaned = re.sub(r'^\s*[:\-–—]\s*', '', cleaned).strip()
            if cleaned and LateFillRuntimeOwner._late_fill_image_prompt_score(raw_unit) > 0:
                prompts.append(cleaned)
        if prompts:
            return prompts
        root_asset_prompts = LateFillRuntimeOwner._extract_late_fill_numbered_image_asset_units(content)
        if root_asset_prompts:
            return root_asset_prompts
        return prompts

    @classmethod
    def _extract_late_fill_compact_numbered_image_prompt_units(cls, content: Any) -> list[str]:
        text = re.sub(r'\s+', ' ', str(content or '').strip())
        if not text:
            return []
        markers = list(re.finditer(r'(?<!\w)(?P<index>\d{1,2})[.)]\s+', text))
        if len(markers) < 2:
            return []
        prompts: list[str] = []
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            body = text[start:end].strip()
            body = re.sub(r'\(?\s*File\s*:\s*[^)]*\)?', '', body, flags=re.IGNORECASE).strip()
            body = re.sub(r'\$\s*\\text\s*\{[^}]+\}\s*', '', body).strip()
            body = cls._strip_late_fill_prompt_label(body)
            body = re.sub(r'\s+', ' ', body).strip(' ;,')
            if not cls._late_fill_image_batch_prompt_is_viable(body):
                continue
            prompts.append(body)
        return prompts if len(prompts) >= 2 else []

    @classmethod
    def _late_fill_image_batch_prompt_is_viable(cls, content: Any) -> bool:
        text = _strip_social_manifest_image_prompt_metadata(content)
        if not text:
            text = cls._strip_late_fill_prompt_label(str(content or '').strip())
        if not text:
            return False
        if cls._late_fill_image_prompt_is_code_polluted(text):
            return False
        if cls._late_fill_image_prompt_is_page_layout_instruction(text):
            return False
        lowered = text.lower()
        heading_only = re.sub(r'^[\s`#>*_-]+|[\s`#>*_-]+$', '', lowered).strip()
        heading_only = re.sub(
            r'^(?:text|plaintext|markdown|md)\s*(?://|::?|[-\u2013\u2014])\s*',
            '',
            heading_only,
        ).strip()
        if re.fullmatch(
            r'(?:image\s+generation\s+|visual\s+|bild(?:generierungs?|\s+generation)?[-\s]*)?'
            r'prompts?'
            r'(?:\s*(?:\([^)]+\)|\[[^\]]+\]))?'
            r'\s*:?',
            heading_only,
        ):
            return False
        if re.fullmatch(
            r'(?:(?:artifact|artefakt)\s*(?:\d+|[ivx]+)\s*:\s*)?'
            r'(?:'
            r'(?:image|visual|poster)\s*(?:generation\s*)?prompt'
            r'|bild(?:generierungs?|\s+generation)?[-\s]*prompt'
            r')'
            r'(?:\s*(?:\([^)]+\)|\[[^\]]+\]))?',
            lowered,
        ):
            return False
        if re.fullmatch(
            r'(?:image\s+)?(?:prompt|scene|section|hero|workflow|interior|exterior|lounge|detail)(?:\s+\d+)?',
            lowered,
        ):
            return False
        words = re.findall(r'\w+', text)
        if cls._late_fill_image_prompt_score(text) > 0:
            return True
        return len(words) >= 6

    @classmethod
    def _late_fill_structured_image_prompt_unit_is_viable(cls, content: Any) -> bool:
        text = cls._strip_late_fill_prompt_label(str(content or '').strip())
        if cls._late_fill_image_batch_prompt_is_viable(text):
            return True
        if not text or cls._late_fill_image_prompt_is_code_polluted(text):
            return False
        if cls._late_fill_image_prompt_is_page_layout_instruction(text):
            return False
        words = re.findall(r'\w+', text)
        return len(words) >= 4 and cls._late_fill_image_prompt_score(text) >= 0

    @staticmethod
    def _request_payload_prompt_text(payload: Any) -> str:
        if not isinstance(payload, Mapping):
            return ''
        for key in ('prompt', 'input_text'):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = payload.get('input')
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if not isinstance(item, Mapping):
                    if str(item or '').strip():
                        parts.append(str(item).strip())
                    continue
                content = item.get('content')
                if isinstance(content, str):
                    parts.append(content.strip())
                elif isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, Mapping):
                            text = str(content_item.get('text') or '').strip()
                            if text:
                                parts.append(text)
                        elif str(content_item or '').strip():
                            parts.append(str(content_item).strip())
            return '\n\n'.join(part for part in parts if part).strip()
        return ''

    @staticmethod
    def _late_fill_batch_has_nonsequential_alpha_prompt_residue(
        raw_items: list[str],
    ) -> bool:
        labels = [
            str(match.group('label') or '')
            for item in raw_items
            if (match := _LATE_FILL_ALPHA_PROMPT_RESIDUE_RE.fullmatch(str(item or '')))
        ]
        if len(labels) < 2:
            return False
        expected_labels = [chr(ord('A') + index) for index in range(len(labels))]
        return len(labels) != len(raw_items) or labels != expected_labels

    @staticmethod
    def _late_fill_batch_has_collapsed_sequential_alpha_prompt_residue(
        raw_items: list[str],
        *,
        expected_count: int,
    ) -> bool:
        return bool(
            LateFillRuntimeOwner._late_fill_collapsed_sequential_alpha_prompt_bodies(
                raw_items,
                expected_count=expected_count,
            )
        )

    @staticmethod
    def _late_fill_collapsed_sequential_alpha_prompt_bodies(
        raw_items: list[str],
        *,
        expected_count: int,
    ) -> list[str]:
        normalized_count = int(expected_count or 0)
        if normalized_count < 2 or not raw_items:
            return []
        first_item = str(raw_items[0] or '').strip()
        matches = list(_LATE_FILL_COLLAPSED_ALPHA_LABEL_RE.finditer(first_item))
        if len(matches) != normalized_count or not matches or matches[0].start() != 0:
            return []
        labels = [str(match.group('label') or '') for match in matches]
        if labels != [chr(ord('A') + index) for index in range(normalized_count)]:
            return []
        bodies: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(first_item)
            body = re.sub(r'\s+', ' ', first_item[match.end():end]).strip()
            if not body:
                return []
            bodies.append(body)
        return bodies

    @classmethod
    def _recover_plain_alpha_prompts_from_collapsed_batch_source(
        cls,
        raw_items: list[str],
        *,
        source: Any,
        expected_count: int,
    ) -> list[str]:
        """Recover only when one preserved source proves the corrupt batch lineage."""

        normalized_count = int(expected_count or 0)
        if normalized_count < 2 or len(raw_items) != normalized_count:
            return []
        collapsed_bodies = cls._late_fill_collapsed_sequential_alpha_prompt_bodies(
            raw_items,
            expected_count=normalized_count,
        )
        if not collapsed_bodies:
            return []
        source_text = str(source or '').strip()
        plain_alpha_prompts = _extract_leading_plain_alpha_image_prompt_lines(
            source_text,
            expected_count=normalized_count,
        )
        def normalize_whitespace(value: Any) -> str:
            return re.sub(r'\s+', ' ', str(value or '')).strip()

        if [normalize_whitespace(item) for item in plain_alpha_prompts] != [
            normalize_whitespace(item) for item in collapsed_bodies
        ]:
            return []

        # A full-size direct batch is only replaceable when the same source also
        # proves how the paragraph fallback produced every tail slot. This keeps
        # a legitimate composite first prompt plus healthy siblings authoritative.
        source_units = [
            normalize_whitespace(item)
            for item in re.split(r'\n\s*\n+', source_text)
            if normalize_whitespace(item)
        ]
        if len(source_units) < normalized_count:
            return []
        if source_units[:normalized_count] != [
            normalize_whitespace(item) for item in raw_items
        ]:
            return []
        return plain_alpha_prompts

    @classmethod
    def _normalize_late_fill_image_batch_prompts(
        cls,
        batch_prompts: Any,
        *,
        expected_count: int = 0,
        assistant_message: Any = '',
        content_payload: Any = '',
        request_payload: Any = None,
        artifact_prompt: Any = '',
    ) -> list[str]:
        raw_items = [
            str(item or '').strip()
            for item in (batch_prompts if isinstance(batch_prompts, list) else [])
        ]
        non_empty_items = [item for item in raw_items if item]
        normalized_count = int(expected_count or 0)
        if (
            normalized_count > 0
            and len(raw_items) == normalized_count
            and len(non_empty_items) != len(raw_items)
        ):
            return raw_items
        direct_prompts: list[str] = []
        for item in non_empty_items:
            prompt = _strip_social_manifest_image_prompt_metadata(item)
            if not prompt:
                prompt = cls._strip_late_fill_prompt_label(item).strip()
            if prompt and cls._late_fill_image_batch_prompt_is_viable(prompt):
                direct_prompts.append(prompt)
        compact_prompts: list[str] = []
        for item in non_empty_items:
            compact_prompts.extend(cls._extract_late_fill_compact_numbered_image_prompt_units(item))
        if (
            compact_prompts
            and len(compact_prompts) > len(direct_prompts)
            and (normalized_count <= 0 or len(compact_prompts) >= normalized_count)
        ):
            return compact_prompts

        batch_is_incomplete = bool(normalized_count and len(direct_prompts) < normalized_count)
        batch_has_collapsed_alpha_residue = (
            cls._late_fill_batch_has_collapsed_sequential_alpha_prompt_residue(
                non_empty_items,
                expected_count=normalized_count,
            )
        )
        batch_is_polluted = (
            len(direct_prompts) != len(non_empty_items)
            or len(non_empty_items) != len(raw_items)
            or cls._late_fill_batch_has_nonsequential_alpha_prompt_residue(non_empty_items)
            or batch_has_collapsed_alpha_residue
        )
        batch_looks_like_weak_social_copy = _image_prompt_batch_looks_like_weak_social_copy(
            direct_prompts
        )
        batch_needs_recovery = (
            batch_is_incomplete
            or batch_is_polluted
            or batch_looks_like_weak_social_copy
        )
        source_candidates = [
            '\n\n'.join(non_empty_items),
            artifact_prompt,
            assistant_message,
            content_payload,
            cls._request_payload_prompt_text(request_payload),
        ]
        sequential_recovery_sources = [
            (artifact_prompt, False),
            ('\n\n'.join(non_empty_items), True),
            (assistant_message, True),
            (content_payload, True),
            (cls._request_payload_prompt_text(request_payload), True),
        ]
        if batch_has_collapsed_alpha_residue:
            for source in (artifact_prompt, assistant_message, content_payload):
                plain_alpha_prompts = cls._recover_plain_alpha_prompts_from_collapsed_batch_source(
                    non_empty_items,
                    source=source,
                    expected_count=normalized_count,
                )
                if len(plain_alpha_prompts) == normalized_count:
                    return plain_alpha_prompts
            # Failed provenance must not compact or renumber producer slots.
            # Preserve the original batch shape so queue indexes cannot drift.
            return raw_items
        for source, require_image_heading in sequential_recovery_sources:
            if not batch_needs_recovery:
                break
            sequential_alpha_prompts = _extract_sequential_bold_alpha_image_prompt_lines(
                str(source or ''),
                expected_count=normalized_count,
                require_image_heading=require_image_heading,
            )
            if sequential_alpha_prompts and (
                not normalized_count or len(sequential_alpha_prompts) >= normalized_count
            ):
                return sequential_alpha_prompts
        for source in source_candidates:
            filename_prompt_units = _extract_filename_social_asset_image_prompt_lines(
                str(source or ''),
                expected_count=normalized_count,
            )
            if not filename_prompt_units:
                continue
            if normalized_count and len(filename_prompt_units) < normalized_count:
                continue
            if (
                batch_is_incomplete
                or batch_is_polluted
                or batch_looks_like_weak_social_copy
            ):
                return filename_prompt_units
        if direct_prompts and not batch_is_incomplete and not batch_is_polluted:
            return direct_prompts

        for source_index, source in enumerate(source_candidates):
            def complete_or_empty(items: list[str]) -> list[str]:
                if normalized_count and items and len(items) < normalized_count:
                    return []
                return items

            prompt_units = _extract_filename_social_asset_image_prompt_lines(
                str(source or ''),
                expected_count=normalized_count,
            )
            prompt_units = complete_or_empty(prompt_units)
            if not prompt_units:
                prompt_units = _extract_inline_labeled_image_prompt_lines(
                    str(source or ''),
                    expected_count=normalized_count,
                )
                prompt_units = complete_or_empty(prompt_units)
            if not prompt_units:
                prompt_units = _extract_sequential_bold_alpha_image_prompt_lines(
                    str(source or ''),
                    expected_count=normalized_count,
                    require_image_heading=source_index != 1,
                )
                prompt_units = complete_or_empty(prompt_units)
            if not prompt_units:
                prompt_units = _extract_social_manifest_pipe_image_prompt_lines(
                    str(source or ''),
                    expected_count=normalized_count,
                )
                prompt_units = complete_or_empty(prompt_units)
            if not prompt_units:
                prompt_units = _extract_numbered_image_prompt_section(
                    str(source or ''),
                    expected_count=normalized_count,
                )
                prompt_units = complete_or_empty(prompt_units)
            if (
                prompt_units
                and direct_prompts
                and _image_prompt_batch_looks_like_weak_social_copy(prompt_units)
            ):
                filename_prompt_units = _extract_filename_social_asset_image_prompt_lines(
                    str(source or ''),
                    expected_count=normalized_count,
                )
                if filename_prompt_units and (
                    not normalized_count or len(filename_prompt_units) >= normalized_count
                ):
                    prompt_units = filename_prompt_units
            if not prompt_units:
                prompt_units = _extract_html_image_card_prompt_units(
                    str(source or ''),
                    expected_count=normalized_count,
                )
                prompt_units = complete_or_empty(prompt_units)
            if not prompt_units:
                prompt_units = cls._extract_late_fill_image_prompt_units(source)
                prompt_units = complete_or_empty(prompt_units)
            if not prompt_units:
                prompt_units = cls._extract_late_fill_compact_numbered_image_prompt_units(source)
                prompt_units = complete_or_empty(prompt_units)
            prompt_units = [
                cls._strip_late_fill_prompt_label(item).strip()
                for item in prompt_units
                if cls._late_fill_structured_image_prompt_unit_is_viable(item)
            ]
            if prompt_units and (not normalized_count or len(prompt_units) >= normalized_count):
                return prompt_units
        if (
            normalized_count > 0
            and len(raw_items) == normalized_count
            and len(direct_prompts) != normalized_count
        ):
            return raw_items
        return direct_prompts

    @staticmethod
    def _extract_late_fill_numbered_image_asset_units(content: Any) -> list[str]:
        text = str(content or '').strip()
        if not text:
            return []
        heading_match = re.search(
            r'(?im)^\s*(?:[#>*-]+\s*)?(?:\*\*+|__+)?\s*'
            r'(?:(?:the\s+)?images?\s+should\s+be|image\s+generation\s+prompts?|visual\s+requirements|image\s+assets?)'
            r'\s*(?:\*\*+|__+)?\s*:?\s*$',
            text,
        )
        start_match = re.search(
            r'(?is)\b(?:first[,\s]+)?(?:identify\s+and\s+generate|create|generate)\s+'
            r'(?:exactly\s+)?\d+\s+(?:distinct\s+)?image\s+assets?\s*[:\-]',
            text,
        ) if not heading_match else None
        if not heading_match and not start_match:
            start_match = re.search(
                r'(?is)\b(?:(?:the\s+)?images?\s+should\s+be|visual\s+requirements|image\s+assets?|image\s+generation\s+prompts?)\s*[:\-]',
                text,
            )
        if not heading_match and not start_match:
            return []
        segment = text[(heading_match or start_match).end():]
        stop_match = re.search(
            r'(?ims)(?:^\s*(?:[#>*-]+\s*)?(?:\*\*+|__+)?\s*'
            r'(?:index\.html|styles\.css|html|css|javascript|js|page files?|text artifacts?|'
            r'(?:phase|internal phase|closure|materialization|artifact|runtime)\s+contract|'
            r'summary|notes|implementation notes)\b|'
            r'\b(?:then\s+create|then\s+write|the\s+html\s+must|html\s+must|'
            r'the\s+page\s+should|use\s+correct\s+relative\s+links|do\s+not\s+create\s+extra)\b)',
            segment,
        )
        if stop_match:
            segment = segment[:stop_match.start()]
        prompts = []
        for match in re.finditer(
            r'(?ms)(?:^|\s)(?:\d+)[.)]\s*(?P<body>.*?)(?=(?:\s+\d+[.)]\s+)|\Z)',
            segment,
        ):
            body = re.sub(r'\s+', ' ', match.group('body')).strip()
            body = re.sub(r'^[:\-–—]\s*', '', body).strip()
            if not body:
                continue
            lowered = body.lower()
            if any(token in lowered for token in ('index.html', 'styles.css', '<html', '</', 'url(')):
                continue
            if len(re.findall(r'\w+', body)) >= 3:
                prompts.append(body)
        return prompts if len(prompts) >= 2 else []

    @staticmethod
    def _late_fill_image_prompt_score(content: str) -> int:
        text = str(content or '').strip()
        if not text:
            return -100
        if LateFillRuntimeOwner._late_fill_image_prompt_is_code_polluted(text):
            return -100
        lowered = text.lower()
        words = re.findall(r'\w+', lowered)
        score = 0
        if re.search(r'\b(?:image\s*[- ]?\s*prompt|bild\s*[- ]?\s*prompt|bildprompt|prompt\s+\d*|visual prompt|poster\s*[- ]?\s*prompt|poster image)\b', lowered):
            score += 8
        if re.search(r'\b(?:poster|graphic design|cinematic|photo|photoreal|illustration|render|scene|lighting|composition|camera|style|fantasy|landscape|portrait|bild|szene)\b', lowered):
            score += 3
        if re.search(r'\b(?:compare|comparison|versus|caption|conclusion|after|nach der|slogan|ad copy|voiceover|audio|final sentence|references?\s+both|generated artifacts?)\b', lowered):
            score -= 5
        if len(words) < 6:
            score -= 3
        elif len(words) > 18:
            score += 1
        return score

    @staticmethod
    def _late_fill_speakable_text_score(content: str) -> int:
        text = str(content or '').strip()
        if not text:
            return -100
        lowered = text.lower()
        words = re.findall(r'\w+', lowered)
        score = 0
        if re.search(r'\b(?:ad copy|script|voiceover|narration|manuscript|spoken text|audio text|read aloud|tts)\b', lowered):
            score += 8
        if re.search(r'\b(?:image\s*[- ]?\s*prompt|bild\s*[- ]?\s*prompt|bildprompt|visual prompt|poster\s*[- ]?\s*prompt|poster image|graphic design|camera|photoreal|cinematic|render)\b', lowered):
            score -= 10
        if re.search(r'\b(?:slogan|headline|tagline)\b', lowered):
            score += 7
        if ':' in text and 3 <= len(words) <= 10:
            score += 3
        if re.search(r'\b(?:final sentence|references?\s+both|generated artifacts?|generated image|poster image)\b', lowered):
            score -= 6
        if (
            re.search(r'\b(?:audio|voice|spoken)\b', lowered)
            and re.search(r'\b(?:poster|image|visual|artifact)\b', lowered)
        ):
            score -= 10
        if len(words) < 4:
            score -= 4
        elif len(words) > 12:
            score += 2
        return score

    @staticmethod
    def _artifact_evidence_text(result: Mapping[str, Any]) -> str:
        artifacts = result.get('artifacts') if isinstance(result.get('artifacts'), list) else []

        def _matching_artifact(path: str) -> Mapping[str, Any]:
            exact = next(
                (
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, Mapping)
                    and str(artifact.get('path') or artifact.get('source_path') or '').strip()
                    == path
                ),
                None,
            )
            if isinstance(exact, Mapping):
                return exact
            return next(
                (artifact for artifact in artifacts if isinstance(artifact, Mapping)),
                {},
            )

        def _evidence(label: str, path: str, artifact: Mapping[str, Any]) -> str:
            artifact_ref = str(
                artifact.get('artifact_ref')
                or artifact.get('ref')
                or result.get('artifact_ref')
                or result.get('ref')
                or ''
            ).strip()
            lines = [f'{label}: {path}']
            if artifact_ref:
                lines.append(f'artifact_ref: {artifact_ref}')
            return '\n'.join(lines)

        for key, label in (
            ('saved_image_path', 'image artifact'),
            ('saved_audio_path', 'audio artifact'),
            ('saved_text_path', 'text artifact'),
        ):
            value = str(result.get(key) or '').strip()
            if value:
                return _evidence(label, value, _matching_artifact(value))
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            path = str(artifact.get('path') or artifact.get('source_path') or '').strip()
            artifact_type = str(artifact.get('type') or artifact.get('kind') or 'artifact').strip()
            if path:
                return _evidence(f'{artifact_type} artifact', path, artifact)
        return ''

    @staticmethod
    def _artifact_records_from_late_fill_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key, artifact_type in (
            ('saved_image_path', 'image'),
            ('saved_audio_path', 'audio'),
            ('saved_text_path', 'text'),
        ):
            path = str(result.get(key) or '').strip()
            if path:
                records.append({'type': artifact_type, 'kind': artifact_type, 'path': path})
        artifacts = result.get('artifacts') if isinstance(result.get('artifacts'), list) else []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            path = str(artifact.get('path') or artifact.get('source_path') or '').strip()
            if not path:
                continue
            artifact_type = str(artifact.get('type') or artifact.get('kind') or '').strip().lower() or 'artifact'
            record = dict(artifact)
            record['type'] = artifact_type
            record.setdefault('kind', artifact_type)
            record['path'] = path
            records.append(record)
        deduped: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for record in records:
            path = str(record.get('path') or '').strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            deduped.append(record)
        return deduped

    @staticmethod
    def _artifact_record_path(record: Mapping[str, Any]) -> str:
        return str(
            record.get('path')
            or record.get('source_path')
            or record.get('saved_text_path')
            or record.get('saved_image_path')
            or record.get('saved_audio_path')
            or ''
        ).strip()

    def _image_artifact_record_sort_key(self, record: Mapping[str, Any]) -> tuple[int, int, str]:
        branch_id = self.branch_id(record)
        phase_id = str(record.get('phase_id') or '').strip()
        for priority, token in enumerate((branch_id, phase_id)):
            match = re.search(r'-(\d+)$', token)
            if match:
                return priority, int(match.group(1)), token
        return 9, 999_999, self._artifact_record_path(record)

    @staticmethod
    def _artifact_record_extension(record: Mapping[str, Any]) -> str:
        path = LateFillRuntimeOwner._artifact_record_path(record)
        for value in (
            record.get('text_artifact_extension'),
            (record.get('artifact_request') or {}).get('extension')
            if isinstance(record.get('artifact_request'), Mapping)
            else '',
            record.get('extension'),
            Path(path).suffix.lstrip('.') if path else '',
            Path(str(record.get('name') or '')).suffix.lstrip('.'),
        ):
            extension = str(value or '').strip().lower().lstrip('.')
            if extension:
                return extension
        return ''

    @staticmethod
    def _artifact_record_source_name(record: Mapping[str, Any]) -> str:
        request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
        path = LateFillRuntimeOwner._artifact_record_path(record)
        return str(
            record.get('text_artifact_source_name')
            or record.get('source_name')
            or request_payload.get('source_name')
            or record.get('name')
            or (Path(path).stem if path else '')
            or ''
        ).strip()

    @staticmethod
    def _normalized_text_artifact_source_name_for_match(value: Any) -> str:
        token = str(value or '').strip()
        if not token:
            return ''
        name = Path(token).name
        if Path(name).suffix:
            name = Path(name).stem
        return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

    @classmethod
    def _artifact_record_matches_text_artifact_source_name(
        cls,
        record: Mapping[str, Any],
        expected_source_name: str,
    ) -> bool:
        expected = cls._normalized_text_artifact_source_name_for_match(expected_source_name)
        if not expected:
            return True
        artifact_request = (
            record.get('artifact_request')
            if isinstance(record.get('artifact_request'), Mapping)
            else record.get('text_artifact_request')
            if isinstance(record.get('text_artifact_request'), Mapping)
            else {}
        )
        explicit_names = [
            record.get('text_artifact_source_name'),
            record.get('source_name'),
            artifact_request.get('source_name'),
            record.get('name'),
        ]
        normalized_explicit_names = {
            cls._normalized_text_artifact_source_name_for_match(value)
            for value in explicit_names
            if str(value or '').strip()
        }
        if normalized_explicit_names:
            return expected in normalized_explicit_names

        path = cls._artifact_record_path(record)
        path_name = cls._normalized_text_artifact_source_name_for_match(path)
        return bool(
            path_name == expected
            or path_name.endswith(f'-{expected}')
        )

    @staticmethod
    def _text_artifact_candidate_content(record: Mapping[str, Any]) -> str:
        for key in ('result_text', 'content', 'content_payload', 'output_text'):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ''

    @staticmethod
    def _text_artifact_nested_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = source.get(key) if isinstance(source, Mapping) else {}
        return value if isinstance(value, Mapping) else {}

    @classmethod
    def _required_text_artifact_target_path(
        cls,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        effective_data: Mapping[str, Any],
    ) -> str:
        mappings: list[Mapping[str, Any]] = [
            source
            for source in (branch, effective_data, infer_result)
            if isinstance(source, Mapping)
        ]

        def target_from_mapping(source: Mapping[str, Any]) -> str:
            direct = str(source.get('text_artifact_target_path') or source.get('target_path') or '').strip()
            if direct:
                return direct
            for request_key in ('artifact_request', 'text_artifact_request'):
                request_payload = cls._text_artifact_nested_mapping(source, request_key)
                request_target = str(
                    request_payload.get('target_path')
                    or request_payload.get('text_artifact_target_path')
                    or ''
                ).strip()
                if request_target:
                    return request_target
            execution_contract = cls._text_artifact_nested_mapping(source, 'execution_contract')
            if execution_contract:
                nested_target = target_from_mapping(execution_contract)
                if nested_target:
                    return nested_target
            return ''

        for source in mappings:
            target_path = target_from_mapping(source)
            if target_path:
                return target_path
        return ''

    @classmethod
    def _required_text_artifact_request(
        cls,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        effective_data: Mapping[str, Any],
        *,
        extension: str,
        source_name: str,
        target_path: str,
    ) -> dict[str, Any]:
        for source in (branch, effective_data, infer_result):
            if not isinstance(source, Mapping):
                continue
            for key in ('artifact_request', 'text_artifact_request'):
                request_payload = source.get(key)
                if isinstance(request_payload, Mapping):
                    request = dict(request_payload)
                    if extension and not request.get('extension'):
                        request['extension'] = extension
                    if source_name and not request.get('source_name'):
                        request['source_name'] = source_name
                    if target_path and not request.get('target_path'):
                        request['target_path'] = target_path
                    return request
            execution_contract = source.get('execution_contract')
            if isinstance(execution_contract, Mapping):
                request = cls._required_text_artifact_request(
                    execution_contract,
                    {},
                    {},
                    extension=extension,
                    source_name=source_name,
                    target_path=target_path,
                )
                if request:
                    return request
        request: dict[str, Any] = {}
        if extension:
            request['extension'] = extension
        if source_name:
            request['source_name'] = source_name
        if target_path:
            request['target_path'] = target_path
        return request

    @classmethod
    def _text_artifact_content_payload_error(
        cls,
        content: Any,
        *,
        extension: str = '',
        saved_path: str = '',
    ) -> tuple[str, Optional[dict[str, Any]]]:
        raw_content = str(content or '').strip()
        normalized_extension = re.sub(
            r'[^a-zA-Z0-9]+',
            '',
            str(extension or Path(str(saved_path or '')).suffix).strip().lower().lstrip('.'),
        )
        unwrapped = strip_enclosing_text_artifact_fence(raw_content, normalized_extension).strip()
        if not unwrapped:
            return '', None
        if text_artifact_content_is_materializer_instruction_echo(unwrapped):
            return unwrapped, {
                'code': 'TEXT_ARTIFACT_INSTRUCTION_ECHO',
                'message': (
                    'Required text artifact branch returned internal materializer instructions instead of a file payload. '
                    'The branch remains repairable; instruction echoes do not fulfill local artifact obligations.'
                ),
                'saved_text_path': saved_path or None,
                'text_artifact_extension': normalized_extension or None,
                'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'retryable': True,
            }
        if not cls._text_artifact_candidate_looks_like_payload(unwrapped, normalized_extension):
            return '', None
        syntax_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            normalized_extension,
            unwrapped,
        )
        if syntax_issues:
            return unwrapped, {
                'code': 'TEXT_ARTIFACT_SYNTAX_SANITY_FAILED',
                'message': (
                    'Required text artifact repair content still fails syntax sanity. '
                    'The target file is not fulfilled until saved artifact bytes pass syntax checks.'
                ),
                'saved_text_path': saved_path or None,
                'text_artifact_extension': normalized_extension or None,
                'syntax_sanity_status': 'issues',
                'syntax_sanity_issue_count': len(syntax_issues),
                'syntax_sanity_issues': syntax_issues[:12],
                'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'retryable': True,
            }
        return unwrapped, None

    @staticmethod
    def _text_artifact_candidate_looks_like_payload(content: str, extension: str) -> bool:
        text = str(content or '').lstrip()
        if not text:
            return False
        ext = str(extension or '').strip().lower().lstrip('.')
        if ext in {'html', 'htm'}:
            return bool(re.match(r'(?is)^(?:<!doctype\s+html\b|<html\b|<[A-Za-z][\w:.-]*(?:\s|>|/))', text))
        if ext == 'css':
            return bool(re.search(r'[{};]', text)) and not re.match(
                r'(?is)^(?:here\s+is|this\s+is|the\s+(?:fixed|repaired)|i\s+(?:fixed|updated))\b',
                text,
            )
        if ext == 'json':
            return text.startswith(('{', '['))
        if ext in {'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx'}:
            return bool(re.match(r'(?m)^\s*(?:import|export|const|let|var|function|class|interface|type)\b', text))
        return True

    @classmethod
    def _required_text_artifact_candidate_payloads(
        cls,
        branch: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        effective_data: Mapping[str, Any],
        *,
        extension: str,
        source_name: str,
        target_path: str,
        revision_required: bool = False,
    ) -> list[str]:
        request = cls._required_text_artifact_request(
            branch,
            infer_result,
            effective_data,
            extension=extension,
            source_name=source_name,
            target_path=target_path,
        )
        candidates: list[str] = []
        seen: set[str] = set()

        def append_candidate(value: Any) -> None:
            if not isinstance(value, str) or not value.strip():
                return
            text = value.strip()
            if text in seen:
                return
            seen.add(text)
            candidates.append(text)

        def append_file_candidate(path_value: Any) -> None:
            path = str(path_value or '').strip()
            if not path:
                return
            if revision_required and cls._text_artifact_path_matches_target(path, target_path):
                # The target already existed before this branch. Its bytes are
                # source evidence, not proof that the backend produced output.
                return
            try:
                target = Path(path).expanduser()
                if not target.is_file() or target.stat().st_size > 512_000:
                    return
                append_candidate(target.read_text(encoding='utf-8', errors='replace'))
            except OSError:
                return

        def append_source_candidates(source: Mapping[str, Any]) -> None:
            append_file_candidate(source.get('saved_text_path'))
            for raw_record in source.get('saved_text_artifacts') or []:
                if not isinstance(raw_record, Mapping):
                    continue
                append_file_candidate(raw_record.get('path') or raw_record.get('saved_text_path'))
                append_candidate(cls._text_artifact_candidate_content(raw_record))
            for key in ('content', 'content_payload', 'result_text', 'output_text'):
                append_candidate(source.get(key))

        candidate_sources = (
            (infer_result,)
            if revision_required
            else (infer_result, effective_data, branch)
        )
        for source in candidate_sources:
            if isinstance(source, Mapping):
                append_source_candidates(source)

        payloads: list[str] = []
        for candidate in candidates:
            extracted_payloads = extract_text_artifact_payloads(candidate, [request] if request else [])
            extracted = [
                str(item.get('content') or '').strip()
                for item in extracted_payloads
                if isinstance(item, Mapping) and str(item.get('content') or '').strip()
            ]
            if extracted:
                payloads.extend(extracted)
                continue
            payloads.append(candidate)
        return payloads

    @classmethod
    def _with_required_text_artifact_saved_result(
        cls,
        infer_result: Mapping[str, Any],
        *,
        target_path: str,
        content: str,
        extension: str,
        source_name: str,
        artifact_request: Mapping[str, Any],
        evidence: str,
    ) -> dict[str, Any]:
        updated = dict(infer_result)
        record = {
            'type': 'text',
            'kind': 'text',
            'path': target_path,
            'saved_text_path': target_path,
            'content': content,
            'content_payload': content,
            'text_artifact_extension': extension or Path(target_path).suffix.lstrip('.'),
            'text_artifact_source_name': source_name or Path(target_path).stem,
            'text_artifact_source': evidence,
            'artifact_request': dict(artifact_request or {}),
            'evidence': evidence,
        }
        updated['saved_text_path'] = target_path
        updated['text_artifact_target_path'] = target_path
        updated['text_artifact_extension'] = extension or record['text_artifact_extension']
        updated['text_artifact_source_name'] = source_name or record['text_artifact_source_name']
        updated['text_artifact_source'] = evidence
        updated['content'] = content
        updated['content_payload'] = content
        updated['result_text'] = content
        existing_records = []
        for item in updated.get('saved_text_artifacts') or []:
            if not isinstance(item, dict):
                continue
            item_path = str(item.get('path') or item.get('saved_text_path') or '').strip()
            if item_path and cls._text_artifact_path_matches_target(item_path, target_path):
                existing_records.append(item)
        updated['saved_text_artifacts'] = cls.merge_unique_artifact_records(existing_records, [record])
        return updated

    def _materialize_required_text_artifact_target_path(
        self,
        branch: Mapping[str, Any],
        infer_result: dict[str, Any],
        effective_data: Mapping[str, Any],
        *,
        extension: str,
        source_name: str,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        target_path = self._required_text_artifact_target_path(branch, infer_result, effective_data)
        if not target_path:
            return infer_result, None
        normalized_extension = re.sub(
            r'[^a-zA-Z0-9]+',
            '',
            str(extension or Path(target_path).suffix).strip().lower().lstrip('.'),
        )
        artifact_request = self._required_text_artifact_request(
            branch,
            infer_result,
            effective_data,
            extension=normalized_extension,
            source_name=source_name,
            target_path=target_path,
        )
        target_binding_error = self._text_artifact_target_binding_violation(
            infer_result,
            target_path=target_path,
            extension=normalized_extension,
            source_name=source_name,
        )
        revision_required = self._text_artifact_revision_required(
            branch,
            effective_data,
            infer_result,
            artifact_request,
        )
        revision_preservation_required = bool(
            revision_required
            and (
                branch.get('text_artifact_revision_preservation_required') is True
                or effective_data.get('text_artifact_revision_preservation_required') is True
            )
        )

        target = Path(target_path).expanduser()
        revision_source_content = ''
        if revision_preservation_required:
            for source in (effective_data, branch):
                if not isinstance(source, Mapping):
                    continue
                if str(source.get('content_payload_source') or '').strip() != (
                    'canonical_predecessor_text_artifact_snapshot'
                ):
                    continue
                revision_source_content = str(source.get('content_payload') or '')
                if revision_source_content:
                    break
            if not revision_source_content and target.is_file():
                try:
                    revision_source_content = target.read_text(
                        encoding='utf-8',
                        errors='replace',
                    )
                except OSError:
                    revision_source_content = ''
        existing_payload_error = self._text_artifact_saved_payload_error(
            target_path,
            extension=normalized_extension,
        )
        if not revision_required and existing_payload_error is None and target.is_file():
            try:
                content = target.read_text(encoding='utf-8', errors='replace')
            except OSError:
                content = ''
            if content and not target_binding_error:
                return self._with_required_text_artifact_saved_result(
                    infer_result,
                    target_path=target_path,
                    content=strip_enclosing_text_artifact_fence(content, normalized_extension).strip(),
                    extension=normalized_extension,
                    source_name=source_name,
                    artifact_request=artifact_request,
                    evidence='target_path_saved_text_artifact_evidence',
                ), None

        if (
            not revision_required
            and
            existing_payload_error
            and existing_payload_error.get('code') == 'TEXT_ARTIFACT_SYNTAX_SANITY_FAILED'
            and target.is_file()
        ):
            try:
                original = target.read_text(encoding='utf-8', errors='replace')
            except OSError:
                original = ''
            if original:
                repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
                    normalized_extension,
                    original,
                )
                if repairs and repaired != original:
                    repaired_content, repaired_error = self._text_artifact_content_payload_error(
                        repaired,
                        extension=normalized_extension,
                        saved_path=target_path,
                    )
                    if not repaired_error and repaired_content:
                        try:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(repaired_content, encoding='utf-8')
                        except OSError as exc:
                            logging.warning(
                                'Could not write deterministic text artifact repair to %s: %s',
                                target_path,
                                exc,
                            )
                        else:
                            saved_error = self._text_artifact_saved_payload_error(
                                target_path,
                                extension=normalized_extension,
                            )
                            if not saved_error:
                                updated = self._with_required_text_artifact_saved_result(
                                    infer_result,
                                    target_path=target_path,
                                    content=repaired_content,
                                    extension=normalized_extension,
                                    source_name=source_name,
                                    artifact_request=artifact_request,
                                    evidence='target_path_deterministic_syntax_repair',
                                )
                                updated['target_path_authoritative_repair'] = {
                                    'status': 'applied',
                                    'target_path': target_path,
                                    'target_extension': normalized_extension or None,
                                    'repair_count': len(repairs),
                                    'repairs': repairs,
                                    'source': 'deterministic_syntax_repair',
                                }
                                return updated, None

        first_candidate_error: Optional[dict[str, Any]] = None
        for candidate in self._required_text_artifact_candidate_payloads(
            branch,
            infer_result,
            effective_data,
            extension=normalized_extension,
            source_name=source_name,
            target_path=target_path,
            revision_required=revision_required,
        ):
            candidate_content, candidate_error = self._text_artifact_content_payload_error(
                candidate,
                extension=normalized_extension,
                saved_path=target_path,
            )
            if candidate_error:
                first_candidate_error = first_candidate_error or candidate_error
                continue
            if not candidate_content:
                continue
            preservation_evidence: dict[str, Any] = {}
            if revision_preservation_required:
                preservation_evidence, preservation_error = (
                    self._text_artifact_revision_preservation_review(
                        revision_source_content,
                        candidate_content,
                        extension=normalized_extension,
                        target_path=target_path,
                    )
                )
                if preservation_error:
                    first_candidate_error = first_candidate_error or preservation_error
                    continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(candidate_content, encoding='utf-8')
            except OSError as exc:
                logging.warning('Could not write text artifact repair output to %s: %s', target_path, exc)
                continue
            saved_error = self._text_artifact_saved_payload_error(target_path, extension=normalized_extension)
            if saved_error:
                first_candidate_error = first_candidate_error or saved_error
                continue
            updated = self._with_required_text_artifact_saved_result(
                infer_result,
                target_path=target_path,
                content=candidate_content,
                extension=normalized_extension,
                source_name=source_name,
                artifact_request=artifact_request,
                evidence=(
                    'target_path_revision_output'
                    if revision_required
                    else 'target_path_authoritative_repair_output'
                ),
            )
            if revision_required:
                source_sha256 = str(
                    branch.get('text_artifact_revision_source_sha256')
                    or effective_data.get('text_artifact_revision_source_sha256')
                    or ''
                ).strip()
                write_proof = {
                    'kind': 'ollmo.text_artifact_revision_write_proof',
                    'version': 1,
                    'status': 'applied',
                    'target_path': target_path,
                    'source_sha256': source_sha256 or None,
                    'output_sha256': hashlib.sha256(
                        candidate_content.encode('utf-8')
                    ).hexdigest(),
                    'evidence': 'current_branch_output_written_to_target',
                }
                updated['text_artifact_revision_required'] = True
                updated['text_artifact_source_is_input'] = True
                updated['text_artifact_revision_write_proof'] = {
                    key: value
                    for key, value in write_proof.items()
                    if value not in (None, '')
                }
                if preservation_evidence:
                    updated['text_artifact_revision_preservation_evidence'] = (
                        preservation_evidence
                    )
            return updated, None

        if first_candidate_error:
            return infer_result, first_candidate_error
        if existing_payload_error:
            return infer_result, existing_payload_error
        if target_binding_error:
            return infer_result, target_binding_error
        if revision_required:
            return infer_result, {
                'code': 'TEXT_ARTIFACT_REVISION_OUTPUT_MISSING',
                'message': (
                    'Required text artifact revision returned without a valid branch-produced file body. '
                    'The existing source bytes are edit input only and cannot fulfill the revision.'
                ),
                'target_path': target_path,
                'text_artifact_extension': normalized_extension or None,
                'text_artifact_source_name': source_name or None,
                'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'retryable': True,
            }
        return infer_result, None

    @classmethod
    def _text_artifact_saved_payload_error(
        cls,
        saved_path: str,
        *,
        extension: str = '',
    ) -> Optional[dict[str, Any]]:
        path = str(saved_path or '').strip()
        if not path:
            return None
        try:
            target = Path(path).expanduser()
            if not target.is_file() or target.stat().st_size > 512_000:
                return None
            content = target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return None
        normalized_extension = re.sub(
            r'[^a-zA-Z0-9]+',
            '',
            str(extension or target.suffix).strip().lower().lstrip('.'),
        )
        unwrapped = strip_enclosing_text_artifact_fence(content, normalized_extension)
        if text_artifact_content_is_materializer_instruction_echo(unwrapped):
            return {
                'code': 'TEXT_ARTIFACT_INSTRUCTION_ECHO',
                'message': (
                    'Required text artifact branch saved internal materializer instructions instead of a file payload. '
                    'The branch remains repairable; instruction echoes do not fulfill local artifact obligations.'
                ),
                'saved_text_path': path,
                'text_artifact_extension': normalized_extension or None,
                'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'retryable': True,
            }
        syntax_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            normalized_extension,
            unwrapped,
        )
        if syntax_issues:
            return {
                'code': 'TEXT_ARTIFACT_SYNTAX_SANITY_FAILED',
                'message': (
                    'Required text artifact branch saved a local file that still fails syntax sanity. '
                    'Existing saved paths do not fulfill repair or materialization obligations until the '
                    'saved artifact bytes pass syntax checks.'
                ),
                'saved_text_path': path,
                'text_artifact_extension': normalized_extension or None,
                'syntax_sanity_status': 'issues',
                'syntax_sanity_issue_count': len(syntax_issues),
                'syntax_sanity_issues': syntax_issues[:12],
                'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                'retryable': True,
            }
        return None

    @staticmethod
    def _saved_text_paths_from_result(result: Mapping[str, Any]) -> list[str]:
        if not isinstance(result, Mapping):
            return []
        paths: list[str] = []

        def append_path(value: Any) -> None:
            token = str(value or '').strip()
            if token and token not in paths:
                paths.append(token)

        append_path(result.get('saved_text_path'))
        append_path(result.get('path') if str(result.get('type') or result.get('kind') or '').lower() == 'text' else '')
        for key in ('saved_text_artifacts', 'artifacts'):
            for artifact in result.get(key) or []:
                if not isinstance(artifact, Mapping):
                    continue
                if key == 'artifacts' and str(artifact.get('type') or artifact.get('kind') or 'text').lower() != 'text':
                    continue
                append_path(artifact.get('path') or artifact.get('saved_text_path'))
        late_fill = result.get('late_fill') if isinstance(result.get('late_fill'), Mapping) else {}
        for fill_result in late_fill.get('fill_results') or []:
            if not isinstance(fill_result, Mapping):
                continue
            append_path(fill_result.get('saved_text_path'))
        return paths

    @classmethod
    def _text_artifact_target_binding_violation(
        cls,
        result: Mapping[str, Any],
        *,
        target_path: str,
        extension: str,
        source_name: str,
    ) -> Optional[dict[str, Any]]:
        target = str(target_path or '').strip()
        if not target:
            return None
        saved_paths = cls._saved_text_paths_from_result(result)
        if not saved_paths:
            return None
        if any(cls._text_artifact_path_matches_target(path, target) for path in saved_paths):
            return None
        return {
            'code': 'TEXT_ARTIFACT_TARGET_BINDING_VIOLATION',
            'message': (
                'Required text artifact repair saved or returned a different text artifact path. '
                'Target-bound repairs must update the requested target file; a sibling artifact does not fulfill '
                'the repair or final materialization contract.'
            ),
            'target_path': target,
            'expected_target_path': target,
            'actual_saved_text_path': saved_paths[0],
            'actual_saved_text_paths': saved_paths,
            'text_artifact_extension': extension or None,
            'text_artifact_source_name': source_name or None,
            'evidence': 'text_artifact_target_path_mismatch',
            'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
            'retryable': True,
        }

    def _matching_canonical_text_artifact_path(
        self,
        payload: Mapping[str, Any],
        *,
        extension: str,
        source_name: str,
        target_path: str = '',
    ) -> str:
        result = self._canonical_text_artifact_saved_result(
            payload,
            extension=extension,
            source_name=source_name,
            target_path=target_path,
        )
        return str(result.get('saved_text_path') or result.get('path') or '').strip()

    def _canonical_text_artifact_saved_result(
        self,
        payload: Mapping[str, Any],
        *,
        extension: str,
        source_name: str,
        target_path: str = '',
        allow_deterministic_repair: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        records: list[Mapping[str, Any]] = []
        artifacts = payload.get('artifacts') if isinstance(payload.get('artifacts'), list) else []
        records.extend(item for item in artifacts if isinstance(item, Mapping))
        records.extend(
            item for item in self.build_canonical_response_artifacts(dict(payload))
            if isinstance(item, Mapping)
        )
        for artifact in payload.get('saved_text_artifacts') or []:
            if isinstance(artifact, Mapping):
                records.append({'type': 'text', **dict(artifact)})
        saved_text_path = str(payload.get('saved_text_path') or payload.get('savedTextPath') or '').strip()
        if saved_text_path:
            records.append(
                {
                    'type': 'text',
                    'path': saved_text_path,
                    'text_artifact_extension': payload.get('text_artifact_extension'),
                    'text_artifact_source_name': payload.get('text_artifact_source_name'),
                    'text_artifact_target_path': payload.get('text_artifact_target_path'),
                    'artifact_request': payload.get('text_artifact_request')
                    if isinstance(payload.get('text_artifact_request'), Mapping)
                    else {},
                }
            )
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        for result in late_fill.get('fill_results') or []:
            if not isinstance(result, Mapping):
                continue
            result_saved_text_path = str(result.get('saved_text_path') or '').strip()
            if not result_saved_text_path:
                continue
            execution_contract = (
                result.get('execution_contract')
                if isinstance(result.get('execution_contract'), Mapping)
                else {}
            )
            artifact_request = (
                result.get('artifact_request')
                if isinstance(result.get('artifact_request'), Mapping)
                else (
                    execution_contract.get('artifact_request')
                    if isinstance(execution_contract.get('artifact_request'), Mapping)
                    else {}
                )
            )
            records.append(
                {
                    'type': 'text',
                    'path': result_saved_text_path,
                    'text_artifact_extension': (
                        result.get('text_artifact_extension')
                        or execution_contract.get('text_artifact_extension')
                        or artifact_request.get('extension')
                    ),
                    'text_artifact_source_name': (
                        result.get('text_artifact_source_name')
                        or execution_contract.get('text_artifact_source_name')
                        or artifact_request.get('source_name')
                    ),
                    'text_artifact_target_path': (
                        result.get('text_artifact_target_path')
                        or execution_contract.get('text_artifact_target_path')
                        or artifact_request.get('target_path')
                    ),
                    'artifact_request': artifact_request,
                }
            )
        expected_extension = str(extension or '').strip().lower().lstrip('.')
        expected_source_name = str(source_name or '').strip().lower()
        expected_target_path = str(target_path or '').strip()
        matching_records_by_path: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if str(record.get('type') or record.get('kind') or 'text').strip().lower() != 'text':
                continue
            path = self._artifact_record_path(record)
            if not path:
                continue
            if expected_target_path and not self._text_artifact_path_matches_target(path, expected_target_path):
                continue
            artifact_extension = self._artifact_record_extension(record)
            if expected_extension and artifact_extension and artifact_extension != expected_extension:
                continue
            if expected_source_name and not self._artifact_record_matches_text_artifact_source_name(
                record,
                expected_source_name,
            ):
                continue
            matching_records_by_path.setdefault(path, record)

        # A name/extension pair is branch identity only when it resolves to one
        # concrete current-response file. Choosing the first of multiple saved
        # siblings would silently turn ambiguous evidence into fulfillment.
        if len(matching_records_by_path) != 1:
            return {}

        for path, record in matching_records_by_path.items():
            artifact_extension = self._artifact_record_extension(record)
            artifact_source_name = self._artifact_record_source_name(record).lower()
            payload_error = self._text_artifact_saved_payload_error(path, extension=expected_extension)
            source = 'canonical_text_artifact_evidence'
            repair_payload: dict[str, Any] = {}
            content = ''
            if payload_error:
                if (
                    not allow_deterministic_repair
                    or payload_error.get('code') != 'TEXT_ARTIFACT_SYNTAX_SANITY_FAILED'
                ):
                    continue
                try:
                    target = Path(path).expanduser()
                    original = target.read_text(encoding='utf-8', errors='replace') if target.is_file() else ''
                except OSError:
                    original = ''
                if not original:
                    continue
                repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
                    expected_extension,
                    original,
                )
                if not repairs or repaired == original:
                    continue
                repaired_content, repaired_error = self._text_artifact_content_payload_error(
                    repaired,
                    extension=expected_extension,
                    saved_path=path,
                )
                if repaired_error or not repaired_content:
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(repaired_content, encoding='utf-8')
                except OSError as exc:
                    logging.warning(
                        'Could not write deterministic canonical text artifact repair to %s: %s',
                        path,
                        exc,
                    )
                    continue
                if self._text_artifact_saved_payload_error(path, extension=expected_extension):
                    continue
                source = 'canonical_text_artifact_deterministic_syntax_repair'
                content = repaired_content
                repair_payload = {
                    'status': 'applied',
                    'target_path': path,
                    'target_extension': expected_extension or None,
                    'repair_count': len(repairs),
                    'repairs': repairs,
                    'source': 'deterministic_syntax_repair',
                }
            else:
                content = self._read_small_text_artifact(path)
            artifact_request = (
                record.get('artifact_request')
                if isinstance(record.get('artifact_request'), Mapping)
                else (
                    record.get('text_artifact_request')
                    if isinstance(record.get('text_artifact_request'), Mapping)
                    else {}
                )
            )
            result = {
                'type': 'text',
                'kind': 'text',
                'path': path,
                'saved_text_path': path,
                'text_artifact_target_path': expected_target_path or self._text_artifact_target_path_from_mapping(record),
                'text_artifact_extension': expected_extension or artifact_extension,
                'text_artifact_source_name': expected_source_name or artifact_source_name,
                'text_artifact_source': source,
                'artifact_request': dict(artifact_request or {}),
            }
            if content:
                result['content'] = content
                result['content_payload'] = content
                result['result_text'] = content
            if repair_payload:
                result['target_path_authoritative_repair'] = repair_payload
                result['canonical_text_artifact_repair'] = repair_payload
            return result
        return {}

    def _ensure_required_text_artifact_saved_truth(
        self,
        branch: Mapping[str, Any],
        infer_result: dict[str, Any],
        effective_data: Mapping[str, Any],
        current_payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        if not self._branch_is_required_text_artifact(branch):
            return infer_result, None
        artifact_request = branch.get('artifact_request') if isinstance(branch.get('artifact_request'), Mapping) else {}
        revision_required = self._text_artifact_revision_required(
            branch,
            effective_data,
            infer_result,
            artifact_request,
        )
        effective_request = (
            effective_data.get('artifact_request')
            if isinstance(effective_data.get('artifact_request'), Mapping)
            else {}
        )
        target_path = self._required_text_artifact_target_path(branch, infer_result, effective_data)
        if target_path:
            extension = str(
                branch.get('text_artifact_extension')
                or effective_request.get('extension')
                or artifact_request.get('extension')
                or effective_data.get('text_artifact_extension')
                or infer_result.get('text_artifact_extension')
                or Path(target_path).suffix
                or ''
            ).strip().lower().lstrip('.')
            source_name = str(
                branch.get('text_artifact_source_name')
                or effective_request.get('source_name')
                or artifact_request.get('source_name')
                or effective_data.get('text_artifact_source_name')
                or infer_result.get('text_artifact_source_name')
                or Path(target_path).stem
                or ''
            ).strip()
        else:
            extension = str(
                infer_result.get('text_artifact_extension')
                or effective_data.get('text_artifact_extension')
                or branch.get('text_artifact_extension')
                or effective_request.get('extension')
                or artifact_request.get('extension')
                or ''
            ).strip().lower().lstrip('.')
            source_name = str(
                infer_result.get('text_artifact_source_name')
                or effective_data.get('text_artifact_source_name')
                or branch.get('text_artifact_source_name')
                or effective_request.get('source_name')
                or artifact_request.get('source_name')
                or ''
            ).strip()
        infer_result, target_materialization_error = self._materialize_required_text_artifact_target_path(
            branch,
            infer_result,
            effective_data,
            extension=extension,
            source_name=source_name,
        )
        if target_materialization_error:
            return infer_result, target_materialization_error
        target_binding_error = self._text_artifact_target_binding_violation(
            infer_result,
            target_path=target_path,
            extension=extension,
            source_name=source_name,
        )
        if target_binding_error:
            return infer_result, target_binding_error
        saved_text_path = str(
            infer_result.get('saved_text_path')
            or effective_data.get('saved_text_path')
            or ''
        ).strip()
        saved_text_artifacts = [
            item
            for item in (
                infer_result.get('saved_text_artifacts')
                or effective_data.get('saved_text_artifacts')
                or []
            )
            if isinstance(item, Mapping)
        ]
        candidate_paths = [saved_text_path]
        candidate_paths.extend(str(item.get('path') or item.get('saved_text_path') or '').strip() for item in saved_text_artifacts)
        for candidate_path in candidate_paths:
            if not candidate_path:
                continue
            payload_error = self._text_artifact_saved_payload_error(candidate_path, extension=extension)
            if payload_error:
                return infer_result, payload_error
        if saved_text_path or any(str(item.get('path') or item.get('saved_text_path') or '').strip() for item in saved_text_artifacts):
            if revision_required and not isinstance(
                infer_result.get('text_artifact_revision_write_proof'),
                Mapping,
            ):
                return infer_result, {
                    'code': 'TEXT_ARTIFACT_REVISION_WRITE_PROOF_MISSING',
                    'message': (
                        'Required text artifact revision has a path but no current-branch write proof. '
                        'Existing source paths cannot fulfill an edit obligation.'
                    ),
                    'text_artifact_extension': extension or None,
                    'text_artifact_source_name': source_name or None,
                    'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                    'retryable': True,
                }
            return infer_result, None
        canonical_path = ''
        if not revision_required:
            canonical_path = self._matching_canonical_text_artifact_path(
                current_payload,
                extension=extension,
                source_name=source_name,
                target_path=target_path,
            )
        if canonical_path:
            updated = dict(infer_result)
            updated['saved_text_path'] = canonical_path
            updated['text_artifact_target_path'] = canonical_path
            updated.setdefault('text_artifact_extension', extension)
            updated.setdefault('text_artifact_source_name', source_name)
            updated.setdefault('text_artifact_source', 'canonical_text_artifact_evidence')
            return updated, None
        return infer_result, {
            'code': 'TEXT_ARTIFACT_NOT_PERSISTED',
            'message': (
                'Required text artifact branch returned without a saved text artifact path. '
                'Chat text alone does not fulfill local file obligations.'
            ),
            'text_artifact_extension': extension or None,
            'text_artifact_source_name': source_name or None,
            'suggested_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
            'retryable': True,
        }

    def _canonical_text_artifact_branch_fulfillment(
        self,
        branch: Mapping[str, Any],
        *payloads: Mapping[str, Any],
        allow_deterministic_repair: bool = True,
    ) -> dict[str, Any]:
        if not self._branch_is_required_text_artifact(branch):
            return {}
        if self._text_artifact_revision_required(branch) and not any(
            isinstance(payload, Mapping)
            and self._text_artifact_revision_write_proven(branch, payload)
            for payload in payloads
        ):
            return {}
        artifact_request = branch.get('artifact_request') if isinstance(branch.get('artifact_request'), Mapping) else {}
        extension = str(
            branch.get('text_artifact_extension')
            or artifact_request.get('extension')
            or ''
        ).strip().lower().lstrip('.')
        source_name = str(
            branch.get('text_artifact_source_name')
            or artifact_request.get('source_name')
            or ''
        ).strip()
        target_path = self._text_artifact_target_path_from_mapping(branch) or self._text_artifact_target_path_from_mapping(
            artifact_request
        )
        if not extension or not source_name:
            return {}
        canonical_result: dict[str, Any] = {}
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            canonical_result = self._canonical_text_artifact_saved_result(
                payload,
                extension=extension,
                source_name=source_name,
                target_path=target_path,
                allow_deterministic_repair=allow_deterministic_repair,
            )
            if canonical_result:
                break
        canonical_path = str(
            canonical_result.get('saved_text_path')
            or canonical_result.get('path')
            or ''
        ).strip()
        if not canonical_path:
            return {}
        text_source = str(
            canonical_result.get('text_artifact_source')
            or 'canonical_text_artifact_evidence'
        ).strip()
        if not text_source:
            text_source = 'canonical_text_artifact_evidence'
        event_message = (
            'Required text artifact satisfied by deterministic canonical saved file repair.'
            if text_source == 'canonical_text_artifact_deterministic_syntax_repair'
            else 'Required text artifact satisfied by canonical saved file evidence.'
        )
        branch_id = self.branch_id(branch)
        if not branch_id:
            return {}
        fill_record: dict[str, Any] = {
            'branch_id': branch_id,
            'phase_id': str(branch.get('phase_id') or branch_id).strip() or branch_id,
            'capability': self.branch_capability(branch) or self.capability_chat,
            'saved_text_path': canonical_path,
            'text_artifact_extension': extension,
            'text_artifact_source_name': source_name,
            'text_artifact_source': text_source,
            'result_text': event_message,
        }
        if target_path:
            fill_record['text_artifact_target_path'] = target_path
        if isinstance(artifact_request, Mapping) and artifact_request:
            fill_record['artifact_request'] = dict(artifact_request)
        target_repair = canonical_result.get('target_path_authoritative_repair')
        if isinstance(target_repair, Mapping):
            fill_record['target_path_authoritative_repair'] = dict(target_repair)
        return {
            key: value
            for key, value in fill_record.items()
            if value not in (None, '', [], {})
        }

    def _text_artifact_branch_has_canonical_evidence(
        self,
        branch: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        if self._text_artifact_revision_required(branch):
            return self._text_artifact_revision_write_proven(branch, payload)
        artifact_request = branch.get('artifact_request') if isinstance(branch.get('artifact_request'), Mapping) else {}
        extension = str(
            branch.get('text_artifact_extension')
            or artifact_request.get('extension')
            or ''
        ).strip().lower().lstrip('.')
        source_name = str(
            branch.get('text_artifact_source_name')
            or artifact_request.get('source_name')
            or ''
        ).strip()
        target_path = self._text_artifact_target_path_from_mapping(branch) or self._text_artifact_target_path_from_mapping(
            artifact_request
        )
        if not extension or not source_name:
            return False
        return bool(
            self._matching_canonical_text_artifact_path(
                payload,
                extension=extension,
                source_name=source_name,
                target_path=target_path,
            )
        )

    def _reconcile_satisfied_failed_text_artifact_branches(
        self,
        payload: Mapping[str, Any],
        late_fill: Mapping[str, Any],
    ) -> dict[str, Any]:
        updated = dict(late_fill or {})
        failed_records = [
            dict(item)
            for item in (updated.get('failed_branches') or [])
            if isinstance(item, Mapping)
        ]
        pending_records = [
            dict(item)
            for item in (updated.get('pending_branches') or [])
            if isinstance(item, Mapping)
        ]
        active_records = [
            dict(item)
            for item in (updated.get('active_branches') or [])
            if isinstance(item, Mapping)
        ]
        if not failed_records and not pending_records and not active_records:
            return updated
        completed_records = [
            dict(item)
            for item in (updated.get('completed_branches') or [])
            if isinstance(item, Mapping)
        ]
        completed_ids = {
            self.branch_id(item)
            for item in completed_records
            if self.branch_id(item)
        }
        reconciled_ids: set[str] = set()
        remaining_failed: list[dict[str, Any]] = []
        remaining_pending: list[dict[str, Any]] = []
        remaining_active: list[dict[str, Any]] = []
        reconciled: list[dict[str, Any]] = []

        def branch_has_satisfied_text_evidence(branch: Mapping[str, Any]) -> bool:
            role = str(branch.get('role') or '').strip()
            artifact_request = (
                branch.get('artifact_request')
                if isinstance(branch.get('artifact_request'), Mapping)
                else {}
            )
            looks_like_text_artifact = (
                str(branch.get('output_type') or '').strip().lower() == 'text'
                or role == 'text_artifact_output'
                or bool(str(branch.get('text_artifact_extension') or '').strip())
                or bool(str(artifact_request.get('extension') or '').strip())
            )
            return bool(
                looks_like_text_artifact
                and self._text_artifact_branch_has_canonical_evidence(branch, payload)
            )

        def remember_satisfied_branch(branch: Mapping[str, Any]) -> None:
            branch_id = self.branch_id(branch)
            fulfilled_branch = dict(branch)
            fulfilled_branch['status'] = 'fulfilled'
            fulfilled_branch['evidence'] = 'canonical_text_artifact_evidence'
            fulfilled_branch['non_blocking_after_final_contract_fulfilled'] = True
            for key in (
                'error',
                'recovery_context',
                'recovery_state',
                'attempt',
                'materialization_contract_open_checks',
                'materialization_contract_unmet',
            ):
                fulfilled_branch.pop(key, None)
            if branch_id:
                reconciled_ids.add(branch_id)
            if branch_id and branch_id not in completed_ids:
                completed_records.append(fulfilled_branch)
                completed_ids.add(branch_id)
            reconciled.append(
                {
                    key: value
                    for key, value in fulfilled_branch.items()
                    if key in {
                        'branch_id',
                        'phase_id',
                        'capability',
                        'output_type',
                        'artifact_request',
                        'status',
                        'evidence',
                        'non_blocking_after_final_contract_fulfilled',
                    }
                    and value not in (None, '', [], {})
                }
            )

        for branch in failed_records:
            if (
                branch_has_satisfied_text_evidence(branch)
            ):
                remember_satisfied_branch(branch)
                continue
            remaining_failed.append(branch)

        for branch in pending_records:
            if branch_has_satisfied_text_evidence(branch):
                remember_satisfied_branch(branch)
                continue
            remaining_pending.append(branch)

        for branch in active_records:
            branch_id = self.branch_id(branch)
            if branch_id and branch_id in reconciled_ids:
                continue
            if branch_has_satisfied_text_evidence(branch):
                remember_satisfied_branch(branch)
                continue
            remaining_active.append(branch)

        if not reconciled:
            return updated
        updated['completed_branches'] = completed_records
        updated['completed_branch_count'] = len(completed_records)
        updated['failed_branches'] = remaining_failed
        updated['failed_branch_count'] = len(remaining_failed)
        updated['pending_branches'] = remaining_pending
        updated['pending_branch_count'] = len(remaining_pending)
        updated['active_branches'] = remaining_active
        previous_reconciled = [
            dict(item)
            for item in (updated.get('satisfied_failed_branches') or [])
            if isinstance(item, Mapping)
        ]
        updated['satisfied_failed_branches'] = previous_reconciled + reconciled
        if reconciled_ids:
            updated['recovery_candidates'] = [
                dict(item)
                for item in (updated.get('recovery_candidates') or [])
                if isinstance(item, Mapping)
                and self.branch_id(item) not in reconciled_ids
            ]
            updated['materialization_contract_demoted_branches'] = [
                dict(item)
                for item in (updated.get('materialization_contract_demoted_branches') or [])
                if isinstance(item, Mapping)
                and self.branch_id(item) not in reconciled_ids
            ]
        if not remaining_failed and not remaining_pending and not remaining_active:
            updated['failed_capabilities'] = []
            updated['pending_capabilities'] = []
            updated['active_capabilities'] = []
            updated['recovery_candidates'] = []
            for key in ('error', 'error_summary', 'failed_at', 'partial_failure', 'auto_recovery_enabled'):
                updated.pop(key, None)
            updated['status'] = 'completed'
        return updated

    @staticmethod
    def _relative_artifact_link(*, from_path: str, to_path: str) -> str:
        if not from_path or not to_path:
            return to_path
        try:
            relative = os.path.relpath(to_path, start=str(Path(from_path).expanduser().parent))
        except (OSError, ValueError):
            return to_path
        return relative.replace(os.sep, '/')

    @staticmethod
    def _url_is_external_or_empty(value: str) -> bool:
        token = str(value or '').strip()
        return not token or bool(_EXTERNAL_LINK_RE.match(token))

    @staticmethod
    def _link_tokens_for_artifact(record: Mapping[str, Any]) -> set[str]:
        tokens: set[str] = set()
        path = LateFillRuntimeOwner._artifact_record_path(record)
        for value in (
            path,
            Path(path).name if path else '',
            str(record.get('url') or '').strip(),
            str(record.get('artifact_ref') or record.get('ref') or '').strip(),
            LateFillRuntimeOwner._artifact_record_source_name(record),
        ):
            token = str(value or '').strip()
            if token:
                tokens.add(token)
        return {token for token in tokens if len(token) >= 3}

    @staticmethod
    def _link_rebind_search_text(record: Mapping[str, Any]) -> str:
        values: list[str] = []
        request_payload = record.get('artifact_request') if isinstance(record.get('artifact_request'), Mapping) else {}
        metadata = record.get('metadata') if isinstance(record.get('metadata'), Mapping) else {}
        for value in (
            LateFillRuntimeOwner._artifact_record_path(record),
            LateFillRuntimeOwner._artifact_record_source_name(record),
            record.get('name'),
            record.get('prompt'),
            record.get('artifact_prompt'),
            record.get('prompt_preview'),
            record.get('content'),
            record.get('semantic_intent'),
            record.get('objective'),
            record.get('deliverable'),
            record.get('link_rebind_search_text'),
            request_payload.get('source_name'),
            request_payload.get('prompt'),
            metadata.get('prompt_preview'),
        ):
            text = str(value or '').strip()
            if text:
                values.append(text)
        return ' '.join(values)

    @staticmethod
    def _link_rebind_semantic_tokens(value: str) -> set[str]:
        raw_tokens = {
            token
            for token in re.split(r'[^a-z0-9]+', str(value or '').lower())
            if len(token) >= 3
        }
        expanded_raw_tokens = set(raw_tokens)
        for token in raw_tokens:
            match = re.fullmatch(r'([a-z]{3,})\d+', token)
            if match:
                expanded_raw_tokens.add(match.group(1))
        generic_tokens = {
            'artifact',
            'generated',
            'image',
            'images',
            'latest',
            'placeholder',
            'picture',
            'photo',
            'png',
            'jpg',
            'jpeg',
            'webp',
        }
        tokens = {token for token in expanded_raw_tokens if token not in generic_tokens}
        synonym_groups = (
            {'lab', 'laboratory', 'laboratories'},
            {'greenhouse', 'biome', 'botanical', 'verdant', 'aeroponic', 'aeroponics'},
            {'exterior', 'external', 'hero', 'outside'},
        )
        expanded = set(tokens)
        for group in synonym_groups:
            if tokens & group:
                expanded.update(group)
        return expanded

    @classmethod
    def _link_rebind_semantic_score(cls, url: str, record: Mapping[str, Any]) -> int:
        token = str(url or '').split('?', 1)[0].split('#', 1)[0]
        basename = Path(token).name
        stem = Path(basename).stem
        url_tokens = cls._link_rebind_semantic_tokens(f'{basename} {stem}')
        if not url_tokens:
            return 0
        record_tokens = cls._link_rebind_semantic_tokens(cls._link_rebind_search_text(record))
        if not record_tokens:
            return 0
        overlap = url_tokens & record_tokens
        if not overlap:
            return 0
        score = len(overlap)
        if stem.lower() in str(cls._link_rebind_search_text(record)).lower():
            score += 3
        return score

    def _collect_link_rebind_artifact_records(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            return []
        slot_by_ref: dict[str, Mapping[str, Any]] = {}
        for slot in payload.get('output_slots') or []:
            if not isinstance(slot, Mapping):
                continue
            artifact_ref = str(slot.get('artifact_ref') or slot.get('ref') or '').strip()
            if artifact_ref:
                slot_by_ref[artifact_ref] = slot

        records: list[dict[str, Any]] = []
        index_by_key: dict[tuple[str, str, str, str], int] = {}

        def merge_record_fields(existing: dict[str, Any], incoming: Mapping[str, Any]) -> None:
            for key, value in incoming.items():
                if value in (None, '', [], {}):
                    continue
                if existing.get(key) in (None, '', [], {}):
                    existing[key] = value
                    continue
                if key == 'link_rebind_search_text':
                    current = str(existing.get(key) or '').strip()
                    incoming_text = str(value or '').strip()
                    if incoming_text and incoming_text not in current:
                        existing[key] = f'{current} {incoming_text}'.strip()

        def add_record(raw_record: Any, *, defaults: Optional[Mapping[str, Any]] = None) -> None:
            if not isinstance(raw_record, Mapping):
                return
            record = dict(defaults or {})
            record.update(dict(raw_record))
            artifact_ref = str(record.get('artifact_ref') or record.get('ref') or '').strip()
            slot = slot_by_ref.get(artifact_ref) if artifact_ref else None
            if isinstance(slot, Mapping):
                for key in ('branch_id', 'phase_id', 'slot_id'):
                    if record.get(key) in (None, '', [], {}) and slot.get(key) not in (None, '', [], {}):
                        record[key] = slot.get(key)
            artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
            path = self._artifact_record_path(record)
            if not artifact_type and path:
                extension = Path(path).suffix.lower().lstrip('.')
                artifact_type = 'image' if extension in _LINK_REBIND_IMAGE_EXTENSIONS else 'text'
                record['type'] = artifact_type
                record.setdefault('kind', artifact_type)
            key = (
                artifact_type,
                path,
                str(record.get('branch_id') or '').strip(),
                str(record.get('phase_id') or '').strip(),
            )
            if not artifact_type and not path:
                return
            if key in index_by_key:
                merge_record_fields(records[index_by_key[key]], record)
                return
            index_by_key[key] = len(records)
            records.append({key: value for key, value in record.items() if value not in (None, '', [], {})})

        raw_artifacts = payload.get('artifacts')
        if isinstance(raw_artifacts, Mapping):
            for key in ('output', 'reference', 'input'):
                for artifact in raw_artifacts.get(key) or []:
                    add_record(artifact)
        else:
            for artifact in raw_artifacts or []:
                add_record(artifact)
        for key in ('input_artifacts', 'reference_artifacts', 'selected_reference_artifacts'):
            for artifact in payload.get(key) or []:
                add_record(artifact)
        for artifact in self.build_canonical_response_artifacts(dict(payload)):
            add_record(artifact)
        for item in payload.get('saved_text_artifacts') or []:
            source = item if isinstance(item, Mapping) else {'path': item}
            add_record({'type': 'text', **dict(source)})
        if str(payload.get('saved_text_path') or '').strip():
            add_record({'type': 'text', 'path': payload.get('saved_text_path')})
        if str(payload.get('saved_image_path') or '').strip():
            add_record({'type': 'image', 'path': payload.get('saved_image_path')})

        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        batch_prompts = [
            str(item or '').strip()
            for item in (late_fill.get('batch_prompts') or [])
            if str(item or '').strip()
        ]
        image_result_index = 0
        for result in late_fill.get('fill_results') or []:
            if not isinstance(result, Mapping):
                continue
            execution_contract = (
                result.get('execution_contract')
                if isinstance(result.get('execution_contract'), Mapping)
                else {}
            )
            artifact_request = (
                result.get('artifact_request')
                if isinstance(result.get('artifact_request'), Mapping)
                else (
                    execution_contract.get('artifact_request')
                    if isinstance(execution_contract.get('artifact_request'), Mapping)
                    else {}
                )
            )
            defaults = {
                'branch_id': result.get('branch_id'),
                'phase_id': result.get('phase_id'),
                'depends_on': (
                    result.get('depends_on')
                    or execution_contract.get('depends_on')
                    or execution_contract.get('dependencies')
                ),
                'dependency_contract': (
                    result.get('dependency_contract')
                    or execution_contract.get('dependency_contract')
                ),
                'source': 'late_fill_result',
            }
            if str(result.get('saved_text_path') or '').strip():
                add_record(
                    {
                        'type': 'text',
                        'path': result.get('saved_text_path'),
                        'content': result.get('result_text') or result.get('content_payload'),
                        'text_artifact_extension': (
                            result.get('text_artifact_extension')
                            or execution_contract.get('text_artifact_extension')
                            or artifact_request.get('extension')
                        ),
                        'text_artifact_source_name': (
                            result.get('text_artifact_source_name')
                            or execution_contract.get('text_artifact_source_name')
                            or artifact_request.get('source_name')
                        ),
                        'text_artifact_source': (
                            result.get('text_artifact_source')
                            or execution_contract.get('text_artifact_source')
                            or artifact_request.get('source')
                        ),
                        'text_artifact_target_path': (
                            result.get('text_artifact_target_path')
                            or execution_contract.get('text_artifact_target_path')
                            or artifact_request.get('target_path')
                        ),
                        'artifact_request': artifact_request or None,
                    },
                    defaults=defaults,
                )
            if str(result.get('saved_image_path') or '').strip():
                prompt_text = str(
                    result.get('artifact_prompt')
                    or result.get('prompt')
                    or result.get('content_payload')
                    or ''
                ).strip()
                if not prompt_text and image_result_index < len(batch_prompts):
                    prompt_text = batch_prompts[image_result_index]
                image_result_index += 1
                add_record(
                    {
                        'type': 'image',
                        'path': result.get('saved_image_path'),
                        'prompt': prompt_text or None,
                        'artifact_prompt': prompt_text or None,
                        'link_rebind_search_text': prompt_text or None,
                    },
                    defaults=defaults,
                )
            for artifact in result.get('artifacts') or []:
                add_record(artifact, defaults=defaults)
        return records

    @classmethod
    def _preferred_asset_for_url(
        cls,
        url: str,
        candidates: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not candidates:
            return None
        token = str(url or '').strip()
        basename = Path(token.split('?', 1)[0].split('#', 1)[0]).name.lower()
        stem = Path(basename).stem.lower()
        extension = Path(basename).suffix.lower().lstrip('.')
        matching_extension = [
            record
            for record in candidates
            if not extension or cls._artifact_record_extension(record) == extension
        ]
        pool = matching_extension or candidates
        for record in pool:
            path = cls._artifact_record_path(record)
            asset_basename = Path(path).name.lower() if path else ''
            asset_stem = Path(path).stem.lower() if path else ''
            source_name = cls._artifact_record_source_name(record).lower()
            if basename and basename == asset_basename:
                return record
            if stem and stem in {asset_stem, source_name}:
                return record
            if stem and source_name and (stem in source_name or source_name in stem):
                return record
        scored = [
            (cls._link_rebind_semantic_score(url, record), index, record)
            for index, record in enumerate(pool)
        ]
        scored = [item for item in scored if item[0] > 0]
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            return scored[0][2]
        if extension == 'css':
            for record in pool:
                source_name = cls._artifact_record_source_name(record).lower()
                if source_name in {'style', 'styles'} or source_name.endswith('_styles'):
                    return record
        return pool[0] if len(pool) == 1 else None

    @classmethod
    def _link_rebind_url_matches_record_identity(
        cls,
        url: str,
        record: Mapping[str, Any],
    ) -> bool:
        """Return whether a local link names the current text artifact itself."""
        token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
        if not token:
            return False
        requested_extension = Path(token).suffix.lower().lstrip('.')
        record_extension = cls._artifact_record_extension(record)
        if requested_extension and record_extension and requested_extension != record_extension:
            return False
        requested_name = Path(token).name
        record_path = cls._artifact_record_path(record)
        if requested_name and record_path and requested_name.lower() == Path(record_path).name.lower():
            return True
        return cls._artifact_record_matches_text_artifact_source_name(
            record,
            requested_name,
        )

    @classmethod
    def _exact_asset_record_for_url(
        cls,
        url: str,
        candidates: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Resolve a sibling text artifact only from exact path/name identity."""
        token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
        requested_name = Path(token).name
        if not requested_name:
            return None
        for record in candidates:
            path = cls._artifact_record_path(record)
            if path and Path(path).name.lower() == requested_name.lower():
                return record
            if cls._artifact_record_matches_text_artifact_source_name(record, requested_name):
                return record
        return None

    @classmethod
    def _unique_artifact_records_by_path(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        index_by_path: dict[str, int] = {}
        for record in records:
            path = cls._artifact_record_path(record)
            if not path:
                continue
            if path not in index_by_path:
                index_by_path[path] = len(unique)
                unique.append(dict(record))
                continue
            existing = unique[index_by_path[path]]
            for key, value in record.items():
                if value in (None, '', [], {}):
                    continue
                if existing.get(key) in (None, '', [], {}):
                    existing[key] = value
                    continue
                if key == 'link_rebind_search_text':
                    current = str(existing.get(key) or '').strip()
                    incoming = str(value or '').strip()
                    if incoming and incoming not in current:
                        existing[key] = f'{current} {incoming}'.strip()
        return unique

    @staticmethod
    def _link_rebind_extension_families(extension: str) -> set[str]:
        token = str(extension or '').strip().lower().lstrip('.')
        if not token:
            return set()
        return {
            family
            for family, extensions in _LINK_REBIND_EXTENSION_FAMILIES.items()
            if token in extensions
        }

    @classmethod
    def _link_rebind_record_declared_families(cls, record: Mapping[str, Any]) -> set[str]:
        families: set[str] = set()
        artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
        mime_type = str(record.get('mime_type') or record.get('content_type') or '').strip().lower()
        for family in ('image', 'audio', 'video', 'font'):
            if artifact_type == family or mime_type.startswith(f'{family}/'):
                families.add(family)
        if artifact_type in {'style', 'stylesheet'} or mime_type == 'text/css':
            families.add('style')
        if artifact_type in {'script', 'javascript'} or mime_type in {'text/javascript', 'application/javascript'}:
            families.add('script')
        return families

    @classmethod
    def _link_rebind_record_families(cls, record: Mapping[str, Any]) -> set[str]:
        return (
            cls._link_rebind_record_declared_families(record)
            or cls._link_rebind_extension_families(cls._artifact_record_extension(record))
        )

    @staticmethod
    def _link_rebind_record_identity_tokens(record: Mapping[str, Any]) -> set[str]:
        if not isinstance(record, Mapping):
            return set()
        return {
            token
            for key in (
                'branch_id',
                'phase_id',
                'task_id',
                'workload_task_id',
                'obligation_id',
                'output_obligation_ref',
            )
            if (token := str(record.get(key) or '').strip())
        }

    @staticmethod
    def _link_rebind_dependency_values(record: Mapping[str, Any]) -> list[str]:
        values: list[str] = []
        if not isinstance(record, Mapping):
            return values
        execution_contract = (
            record.get('execution_contract')
            if isinstance(record.get('execution_contract'), Mapping)
            else {}
        )
        for raw_values in (
            record.get('depends_on'),
            record.get('dependsOn'),
            execution_contract.get('depends_on'),
            execution_contract.get('dependencies'),
        ):
            candidates = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
            for value in candidates:
                token = str(value or '').strip()
                if token and token not in values:
                    values.append(token)
        return values

    @classmethod
    def _link_rebind_consumer_dependency_ids(
        cls,
        payload: Mapping[str, Any],
        consumer_record: Mapping[str, Any],
    ) -> list[str]:
        dependency_ids = cls._link_rebind_dependency_values(consumer_record)
        consumer_tokens = cls._link_rebind_record_identity_tokens(consumer_record)
        consumer_path = cls._artifact_record_path(consumer_record)

        runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        records: list[Mapping[str, Any]] = []
        for key in ('phases', 'downstream_branches'):
            records.extend(
                item
                for item in (graph.get(key) if isinstance(graph.get(key), list) else [])
                if isinstance(item, Mapping)
            )
        for key in (
            'pending_branches',
            'active_branches',
            'completed_branches',
            'failed_branches',
            'cancelled_branches',
            'fill_results',
        ):
            records.extend(
                item
                for item in (late_fill.get(key) if isinstance(late_fill.get(key), list) else [])
                if isinstance(item, Mapping)
            )

        def record_targets_consumer(record: Mapping[str, Any]) -> bool:
            if consumer_tokens & cls._link_rebind_record_identity_tokens(record):
                return True
            artifact_request = (
                record.get('artifact_request')
                if isinstance(record.get('artifact_request'), Mapping)
                else {}
            )
            target_path = str(
                record.get('text_artifact_target_path')
                or artifact_request.get('target_path')
                or ''
            ).strip()
            return bool(consumer_path and target_path and consumer_path == target_path)

        for record in records:
            if not record_targets_consumer(record):
                continue
            for dependency_id in cls._link_rebind_dependency_values(record):
                if dependency_id not in dependency_ids:
                    dependency_ids.append(dependency_id)

        expanded_ids = list(dependency_ids)
        for record in records:
            record_tokens = cls._link_rebind_record_identity_tokens(record)
            if not record_tokens.intersection(dependency_ids):
                continue
            for token in record_tokens:
                if token not in expanded_ids:
                    expanded_ids.append(token)
        return expanded_ids

    @classmethod
    def _link_rebind_asset_records_for_consumer(
        cls,
        payload: Mapping[str, Any],
        consumer_record: Mapping[str, Any],
        asset_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str], str]:
        dependency_ids = cls._link_rebind_consumer_dependency_ids(payload, consumer_record)
        if not dependency_ids:
            return list(asset_records), [], 'artifact_set_fallback_without_declared_dependencies'
        dependency_tokens = set(dependency_ids)
        runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        graph_records = [
            item
            for key in ('phases', 'downstream_branches')
            for item in (graph.get(key) if isinstance(graph.get(key), list) else [])
            if isinstance(item, Mapping)
        ]
        consumer_tokens = cls._link_rebind_record_identity_tokens(consumer_record)
        consumer_graph_records = [
            record
            for record in graph_records
            if cls._link_rebind_record_identity_tokens(record) & consumer_tokens
        ]

        def has_local_visual_binding(record: Mapping[str, Any]) -> bool:
            execution_contract = (
                record.get('execution_contract')
                if isinstance(record.get('execution_contract'), Mapping)
                else {}
            )
            values = (
                record.get('dependency_contract'),
                execution_contract.get('dependency_contract'),
            )
            return any(
                'local_visual_asset_binding' in str(value or '').strip().lower()
                for value in values
            )

        local_visual_binding = has_local_visual_binding(consumer_record) or any(
            has_local_visual_binding(record)
            for record in consumer_graph_records
        )
        dependency_graph_records = [
            record
            for record in graph_records
            if cls._link_rebind_record_identity_tokens(record) & dependency_tokens
        ]

        def graph_record_requires_artifact(record: Mapping[str, Any]) -> bool:
            output_contract = (
                record.get('output_contract')
                if isinstance(record.get('output_contract'), Mapping)
                else {}
            )
            output_type = str(record.get('output_type') or '').strip().lower()
            role = str(record.get('role') or '').strip().lower()
            fulfillment_policy = str(output_contract.get('fulfillment_policy') or '').strip().lower()
            return bool(
                record.get('requires_artifact') is True
                or output_type in {'image', 'audio', 'video', 'file', 'artifact'}
                or role == 'text_artifact_output'
                or 'artifact' in fulfillment_policy
            )

        artifact_dependency_records = [
            record
            for record in dependency_graph_records
            if graph_record_requires_artifact(record)
        ]
        if not local_visual_binding and not artifact_dependency_records:
            return list(asset_records), [], 'artifact_set_fallback_for_ordering_dependencies'

        matched = [
            record
            for record in asset_records
            if cls._link_rebind_record_identity_tokens(record) & dependency_tokens
        ]
        if not local_visual_binding:
            return matched, dependency_ids, 'consumer_declared_dependency_binding'

        shared_text_targets = [
            record
            for record in asset_records
            if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
            and cls._artifact_record_extension(record) in _LINK_REBIND_HTML_TARGET_EXTENSIONS
        ]
        selected = cls._unique_artifact_records_by_path([*matched, *shared_text_targets])
        media_dependency_records = [
            record
            for record in artifact_dependency_records
            if str(record.get('output_type') or '').strip().lower() in {'image', 'audio', 'video'}
            or str(record.get('capability') or '').strip().lower()
            in {'image_generation', 'text_to_speech', 'audio_generation', 'video_generation'}
        ]
        if not media_dependency_records and local_visual_binding:
            media_dependency_records = dependency_graph_records
        missing_media_dependency = any(
            not any(
                cls._link_rebind_record_identity_tokens(asset_record)
                & cls._link_rebind_record_identity_tokens(dependency_record)
                for asset_record in matched
            )
            for dependency_record in media_dependency_records
        )
        if missing_media_dependency:
            return selected, dependency_ids, 'consumer_declared_media_dependency_missing'
        return (
            selected,
            dependency_ids,
            'consumer_declared_dependency_binding_with_shared_text_targets',
        )

    @staticmethod
    def _link_rebind_normalize_json_key(key: str) -> str:
        value = re.sub(
            r'(?<=[a-z0-9])(?=[A-Z])',
            '_',
            str(key or '').strip(),
        )
        return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')

    @classmethod
    def _link_rebind_json_path_key_family(cls, key: str) -> str:
        normalized = cls._link_rebind_normalize_json_key(key)
        if not normalized or not _LINK_REBIND_JSON_PATH_KEY_RE.fullmatch(normalized):
            return ''
        for family, tokens in (
            ('image', ('image', 'picture', 'photo')),
            ('audio', ('audio', 'sound', 'voice')),
            ('video', ('video', 'movie')),
            ('font', ('font',)),
            ('style', ('style', 'stylesheet', 'css')),
            ('script', ('script', 'javascript', 'js')),
        ):
            if any(token in normalized.split('_') for token in tokens):
                return family
        return ''

    @classmethod
    def _link_rebind_infer_single_family(cls, records: list[dict[str, Any]]) -> str:
        families: set[str] = set()
        for record in records:
            families.update(cls._link_rebind_record_families(record))
        return next(iter(families)) if len(families) == 1 else ''

    @staticmethod
    def _link_rebind_attr_context_family(content: str, match_start: int) -> str:
        tag_start = str(content or '').rfind('<', 0, max(0, match_start))
        if tag_start < 0:
            return ''
        tag_prefix = str(content or '')[tag_start:match_start].lower()
        tag_match = re.match(r'<\s*([a-z0-9:-]+)', tag_prefix)
        tag_name = tag_match.group(1) if tag_match else ''
        if tag_name in {'img', 'picture'}:
            return 'image'
        if tag_name == 'audio':
            return 'audio'
        if tag_name == 'video':
            return 'video'
        if tag_name == 'script':
            return 'script'
        if tag_name == 'link':
            if re.search(r'\brel\s*=\s*[\'"]?stylesheet\b', tag_prefix):
                return 'style'
            if re.search(r'\bas\s*=\s*[\'"]?font\b', tag_prefix):
                return 'font'
        return ''

    @staticmethod
    def _link_rebind_css_context_family(content: str, match_start: int) -> str:
        prefix = str(content or '')[max(0, match_start - 240):match_start].lower()
        if '@font-face' in prefix and prefix.rfind('@font-face') > prefix.rfind('}'):
            return 'font'
        declaration_start = max(prefix.rfind('{'), prefix.rfind(';'))
        declaration_prefix = prefix[declaration_start + 1 :]
        if re.search(
            r'\b(?:background(?:-image)?|border-image(?:-source)?|list-style-image)\s*:[^{};]*$',
            declaration_prefix,
        ):
            return 'image'
        return ''

    @staticmethod
    def _link_rebind_family_matches(
        record: Mapping[str, Any],
        families: set[str],
    ) -> bool:
        if not families:
            return False
        return bool(LateFillRuntimeOwner._link_rebind_record_families(record) & families)

    @classmethod
    def _link_rebind_records_for_family(
        cls,
        family: str,
        records: list[dict[str, Any]],
        *,
        declared_only: bool = False,
    ) -> list[dict[str, Any]]:
        token = str(family or '').strip().lower()
        if not token:
            return []
        for family_name, extensions in _LINK_REBIND_EXTENSION_FAMILIES.items():
            if token in extensions:
                token = family_name
                break
        declared_matches = [
            record
            for record in records
            if token in cls._link_rebind_record_declared_families(record)
        ]
        if declared_matches or declared_only:
            return declared_matches
        return [
            record
            for record in records
            if token in cls._link_rebind_record_families(record)
        ]

    @classmethod
    def _link_rebind_records_for_extension(
        cls,
        extension: str,
        records: list[dict[str, Any]],
        *,
        preferred_family: str = '',
    ) -> list[dict[str, Any]]:
        token = str(extension or '').strip().lower().lstrip('.')
        if not token:
            family_records = cls._link_rebind_records_for_family(preferred_family, records)
            if family_records:
                return family_records
            return list(records)
        exact_matches = [
            record
            for record in records
            if cls._artifact_record_extension(record) == token
        ]
        if preferred_family and exact_matches:
            preferred_exact_matches = cls._link_rebind_records_for_family(
                preferred_family,
                exact_matches,
                declared_only=True,
            )
            if preferred_exact_matches:
                return preferred_exact_matches
        elif exact_matches:
            return exact_matches
        family_records = cls._link_rebind_records_for_family(preferred_family, records)
        if family_records:
            return family_records
        if exact_matches:
            return exact_matches
        families = cls._link_rebind_extension_families(token)
        if not families:
            return []
        return [
            record
            for record in records
            if cls._link_rebind_family_matches(record, families)
        ]

    @classmethod
    def _local_link_target_exists(cls, url: str, *, target_path: str) -> bool:
        if cls._url_is_external_or_empty(url):
            return True
        token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
        if not token:
            return True
        try:
            path = Path(token).expanduser()
            if not path.is_absolute():
                path = Path(target_path).expanduser().parent / token
            return path.exists()
        except (OSError, ValueError):
            return False

    @classmethod
    def _local_link_target_path(cls, url: str, *, target_path: str) -> str:
        if cls._url_is_external_or_empty(url):
            return ''
        token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
        if not token:
            return ''
        try:
            path = Path(token).expanduser()
            if not path.is_absolute():
                if not target_path:
                    return ''
                path = Path(target_path).expanduser().parent / token
            return str(path.resolve(strict=False))
        except (OSError, ValueError):
            return ''

    @classmethod
    def _link_url_needs_rebind(
        cls,
        url: str,
        asset_records: list[dict[str, Any]],
        *,
        target_path: str = '',
        preferred_family: str = '',
    ) -> bool:
        if cls._url_is_external_or_empty(url):
            token = str(url or '').strip()
            if not token or not _EXTERNAL_LINK_RE.match(token):
                return False
            if str(preferred_family or '').strip().lower() != 'image':
                return False
            return any(
                cls._artifact_record_path(record)
                and 'image' in cls._link_rebind_record_families(record)
                for record in asset_records
            )
        token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
        if not token:
            return False
        extension = Path(token).suffix.lower().lstrip('.')
        if not extension:
            matching_records = cls._link_rebind_records_for_extension(
                '',
                asset_records,
                preferred_family=preferred_family,
            )
            if not matching_records:
                return False
            if _LINK_REBIND_PLACEHOLDER_RE.search(url):
                return True
            if target_path and cls._local_link_target_exists(url, target_path=target_path):
                return False
            return bool(preferred_family)
        matching_records = cls._link_rebind_records_for_extension(
            extension,
            asset_records,
            preferred_family=preferred_family,
        )
        if not matching_records:
            return False
        if _LINK_REBIND_PLACEHOLDER_RE.search(url):
            return True
        if target_path and cls._local_link_target_exists(url, target_path=target_path):
            return False
        return True

    def _rebind_text_artifact_content(
        self,
        content: str,
        *,
        target_path: str,
        target_record: Mapping[str, Any],
        asset_records: list[dict[str, Any]],
        static_fetch_asset_records: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        changes: list[dict[str, Any]] = []
        used_asset_paths: set[str] = set()

        def is_html_sibling_link(url: str) -> bool:
            extension = Path(str(url or '').split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            return extension in {'html', 'htm'}

        def choose_asset_record(
            url: str,
            candidates: list[dict[str, Any]],
        ) -> Optional[dict[str, Any]]:
            viable_candidates = [
                record
                for record in candidates
                if self._artifact_record_path(record)
            ]
            if not viable_candidates:
                return None
            unused_candidates = [
                record
                for record in viable_candidates
                if self._artifact_record_path(record) not in used_asset_paths
            ]
            record = (
                self._exact_asset_record_for_url(url, unused_candidates)
                if is_html_sibling_link(url)
                else self._preferred_asset_for_url(url, unused_candidates)
            )
            if not record and unused_candidates and not is_html_sibling_link(url):
                record = unused_candidates[0]
            if not record:
                record = (
                    self._exact_asset_record_for_url(url, viable_candidates)
                    if is_html_sibling_link(url)
                    else self._preferred_asset_for_url(url, viable_candidates) or viable_candidates[0]
                )
            if not record:
                return None
            linked_path = self._artifact_record_path(record)
            if linked_path:
                used_asset_paths.add(linked_path)
            return record

        def replacement_for_asset(
            url: str,
            candidates: list[dict[str, Any]],
        ) -> tuple[str, Optional[dict[str, Any]]]:
            record = choose_asset_record(url, candidates)
            if not record:
                return url, None
            linked_path = self._artifact_record_path(record)
            if not linked_path:
                return url, None
            return self._relative_artifact_link(from_path=target_path, to_path=linked_path), record

        def replace_attr(match: re.Match[str]) -> str:
            url = match.group('url')
            if self._link_rebind_url_matches_record_identity(url, target_record):
                return match.group(0)
            extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            preferred_family = self._link_rebind_attr_context_family(content, match.start())
            candidates = self._link_rebind_records_for_extension(
                extension,
                asset_records,
                preferred_family=preferred_family,
            )
            if not self._link_url_needs_rebind(
                url,
                candidates,
                target_path=target_path,
                preferred_family=preferred_family,
            ):
                return match.group(0)
            replacement, record = replacement_for_asset(url, candidates)
            if not record or replacement == url:
                return match.group(0)
            changes.append(
                {
                    'from': url,
                    'to': replacement,
                    'linked_path': self._artifact_record_path(record),
                    'kind': (
                        'external_image_attribute_link'
                        if _EXTERNAL_LINK_RE.match(str(url or '').strip())
                        else 'attribute_link'
                    ),
                }
            )
            return f"{match.group('prefix')}{match.group('quote')}{replacement}{match.group('quote')}"

        def replace_url(match: re.Match[str]) -> str:
            url = match.group('url')
            if self._link_rebind_url_matches_record_identity(url, target_record):
                return match.group(0)
            extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            preferred_family = self._link_rebind_css_context_family(content, match.start())
            candidates = self._link_rebind_records_for_extension(
                extension,
                asset_records,
                preferred_family=preferred_family,
            )
            if not self._link_url_needs_rebind(
                url,
                candidates,
                target_path=target_path,
                preferred_family=preferred_family,
            ):
                return match.group(0)
            record = choose_asset_record(url, candidates)
            if not record:
                return match.group(0)
            linked_path = self._artifact_record_path(record)
            replacement = self._relative_artifact_link(from_path=target_path, to_path=linked_path)
            if not record or replacement == url:
                return match.group(0)
            changes.append(
                {
                    'from': url,
                    'to': replacement,
                    'linked_path': linked_path,
                    'kind': 'css_url',
                }
            )
            return f"{match.group('prefix')}{replacement}{match.group('suffix')}"

        def replace_fetch(match: re.Match[str]) -> str:
            url = match.group('url')
            if self._link_rebind_url_matches_record_identity(url, target_record):
                return match.group(0)
            if not _static_fetch_url_is_local_file_dependency(url):
                return match.group(0)
            extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            requested_token = url.split('?', 1)[0].split('#', 1)[0]
            requested_name = Path(requested_token).name.lower()
            fetch_records = (
                static_fetch_asset_records
                if static_fetch_asset_records is not None
                else asset_records
            )
            candidates = [
                record
                for record in self._link_rebind_records_for_extension(extension, fetch_records)
                if (
                    Path(self._artifact_record_path(record)).name.lower() == requested_name
                    or self._artifact_record_matches_text_artifact_source_name(
                        record,
                        requested_name,
                    )
                )
            ]
            if not self._link_url_needs_rebind(
                url,
                candidates,
                target_path=target_path,
            ):
                return match.group(0)
            replacement, record = replacement_for_asset(url, candidates)
            if not record or replacement == url:
                return match.group(0)
            changes.append(
                {
                    'from': url,
                    'to': replacement,
                    'linked_path': self._artifact_record_path(record),
                    'kind': 'static_fetch_link',
                    'selection_policy': 'exact_named_static_fetch_dependency',
                }
            )
            return f"{match.group('prefix')}{match.group('quote')}{replacement}{match.group('quote')}"

        rebound = _LINK_REBIND_ATTR_RE.sub(replace_attr, content)
        rebound = _LINK_REBIND_URL_RE.sub(replace_url, rebound)
        rebound = _LINK_REBIND_FETCH_RE.sub(replace_fetch, rebound)
        return rebound, changes

    def _rebind_json_artifact_content(
        self,
        content: str,
        *,
        target_path: str,
        asset_records: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            document = json.loads(str(content or ''))
        except (TypeError, ValueError, json.JSONDecodeError):
            return content, []

        changes: list[dict[str, Any]] = []
        used_asset_paths: set[str] = set()

        def choose_asset_record(
            url: str,
            candidates: list[dict[str, Any]],
        ) -> Optional[dict[str, Any]]:
            viable = [
                record
                for record in candidates
                if self._artifact_record_path(record)
            ]
            unused = [
                record
                for record in viable
                if self._artifact_record_path(record) not in used_asset_paths
            ]
            record = self._preferred_asset_for_url(url, unused)
            if not record and unused:
                record = unused[0]
            if not record:
                record = self._preferred_asset_for_url(url, viable)
                if not record and len(viable) == 1:
                    record = viable[0]
            linked_path = self._artifact_record_path(record or {})
            if linked_path:
                used_asset_paths.add(linked_path)
            return record

        def pointer_token(value: Any) -> str:
            return str(value).replace('~', '~0').replace('/', '~1')

        def visit(value: Any, *, field_key: str = '', pointer: str = '') -> Any:
            if isinstance(value, dict):
                return {
                    key: visit(
                        child,
                        field_key=str(key),
                        pointer=f'{pointer}/{pointer_token(key)}',
                    )
                    for key, child in value.items()
                }
            if isinstance(value, list):
                return [
                    visit(
                        child,
                        field_key=field_key,
                        pointer=f'{pointer}/{index}',
                    )
                    for index, child in enumerate(value)
                ]
            if not isinstance(value, str):
                return value
            preferred_family = self._link_rebind_json_path_key_family(field_key)
            normalized_key = self._link_rebind_normalize_json_key(field_key)
            if not normalized_key or not _LINK_REBIND_JSON_PATH_KEY_RE.fullmatch(normalized_key):
                return value
            token = value.split('?', 1)[0].split('#', 1)[0]
            extension = Path(token).suffix.lower().lstrip('.')
            candidates = self._link_rebind_records_for_extension(
                extension,
                asset_records,
                preferred_family=preferred_family,
            )
            if not preferred_family:
                preferred_family = self._link_rebind_infer_single_family(candidates)
                candidates = self._link_rebind_records_for_extension(
                    extension,
                    asset_records,
                    preferred_family=preferred_family,
                )
            if not self._link_url_needs_rebind(
                value,
                candidates,
                target_path=target_path,
                preferred_family=preferred_family,
            ):
                return value
            record = choose_asset_record(value, candidates)
            linked_path = self._artifact_record_path(record or {})
            if not linked_path:
                return value
            replacement = self._relative_artifact_link(
                from_path=target_path,
                to_path=linked_path,
            )
            if not replacement or replacement == value:
                return value
            changes.append(
                {
                    'from': value,
                    'to': replacement,
                    'linked_path': linked_path,
                    'kind': 'json_path_link',
                    'json_pointer': pointer or '/',
                    'json_field': field_key,
                }
            )
            return replacement

        rebound_document = visit(document)
        if not changes:
            return content, []
        rebound = json.dumps(rebound_document, ensure_ascii=False, indent=2)
        if str(content or '').endswith('\n'):
            rebound += '\n'
        return rebound, changes

    def rebind_terminal_linked_artifacts(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        records = self._collect_link_rebind_artifact_records(payload)
        if not records:
            return payload
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        asset_records = [
            record
            for record in records
            if self._artifact_record_path(record)
        ]
        asset_records = self._unique_artifact_records_by_path(asset_records)
        if not text_records or not asset_records:
            return payload

        updated_payload = dict(payload or {})
        rebinds: list[dict[str, Any]] = []
        updated_content_by_path: dict[str, str] = {}
        for record in text_records:
            target_path = self._artifact_record_path(record)
            available_asset_records = [
                item
                for item in asset_records
                if self._artifact_record_path(item) != target_path
            ]
            consumer_asset_records, dependency_ids, selection_policy = (
                self._link_rebind_asset_records_for_consumer(
                    payload,
                    record,
                    available_asset_records,
                )
            )
            try:
                target = Path(target_path).expanduser()
                if not target.is_file() or target.stat().st_size > 512_000:
                    continue
                original = target.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            if self._artifact_record_extension(record) == 'json':
                rebound, changes = self._rebind_json_artifact_content(
                    original,
                    target_path=target_path,
                    asset_records=consumer_asset_records,
                )
            else:
                rebound, changes = self._rebind_text_artifact_content(
                    original,
                    target_path=target_path,
                    target_record=record,
                    asset_records=consumer_asset_records,
                    static_fetch_asset_records=available_asset_records,
                )
            if not changes or rebound == original:
                continue
            for change in changes:
                change.setdefault('selection_policy', selection_policy)
                if (
                    dependency_ids
                    and change.get('selection_policy') != 'exact_named_static_fetch_dependency'
                ):
                    change['consumer_dependency_ids'] = list(dependency_ids)
            try:
                target.write_text(rebound, encoding='utf-8')
            except OSError as exc:
                logging.warning('Could not rebind linked artifact paths in %s: %s', target_path, exc)
                continue
            updated_content_by_path[target_path] = rebound
            rebinds.append(
                {
                    'kind': 'ollmo.linked_artifact_rebind',
                    'target_path': target_path,
                    'target_extension': self._artifact_record_extension(record),
                    'target_source_name': self._artifact_record_source_name(record),
                    'change_count': len(changes),
                    'changes': changes,
                    'status': 'applied',
                    'source': 'terminal_late_fill_link_rebind',
                }
            )

        if not rebinds:
            return updated_payload

        def update_text_record_content(raw_record: Any) -> Any:
            if not isinstance(raw_record, dict):
                return raw_record
            record = dict(raw_record)
            path = self._artifact_record_path(record)
            if path in updated_content_by_path:
                record['content'] = updated_content_by_path[path]
                if 'result_text' in record:
                    record['result_text'] = updated_content_by_path[path]
                if 'content_payload' in record:
                    record['content_payload'] = updated_content_by_path[path]
            return record

        if isinstance(updated_payload.get('artifacts'), list):
            updated_payload['artifacts'] = [
                update_text_record_content(item)
                for item in updated_payload.get('artifacts') or []
            ]
        if isinstance(updated_payload.get('saved_text_artifacts'), list):
            updated_payload['saved_text_artifacts'] = [
                update_text_record_content(item)
                for item in updated_payload.get('saved_text_artifacts') or []
            ]
        late_fill = dict(updated_payload.get('late_fill') or {}) if isinstance(updated_payload.get('late_fill'), Mapping) else {}
        if isinstance(late_fill.get('fill_results'), list):
            late_fill['fill_results'] = [
                update_text_record_content(item)
                for item in late_fill.get('fill_results') or []
            ]
        existing_rebinds = [
            dict(item)
            for item in (late_fill.get('linked_artifact_rebinds') or [])
            if isinstance(item, Mapping)
        ]
        late_fill['linked_artifact_rebinds'] = [*existing_rebinds, *rebinds]
        late_fill['linked_artifact_rebind_status'] = 'applied'
        updated_payload['late_fill'] = late_fill
        return updated_payload

    def refresh_terminal_graph_closure_review(
        self,
        payload: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        route_payload: Optional[dict[str, Any]],
        artifact_gap: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            review = self.build_graph_closure_review(
                str((payload or {}).get('output_text') or ''),
                route_payload=route_payload,
                request_payload=request_payload,
                artifact_payload=payload,
                artifact_gap=artifact_gap,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning('Could not refresh terminal graph closure review: %s', exc)
            return payload
        return self.attach_graph_closure_review_diagnostics(payload, review)

    def refresh_runtime_graph_repair_evidence(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Refresh current repair diagnostics while preserving graph audit records."""

        if not callable(self.attach_runtime_graph_repair_evidence):
            return payload
        try:
            refreshed = self.attach_runtime_graph_repair_evidence(payload)
        except Exception as exc:  # noqa: BLE001
            logging.warning('Could not refresh runtime graph repair evidence: %s', exc)
            return payload
        return refreshed if isinstance(refreshed, dict) else payload

    @staticmethod
    def _inline_text_artifact_block_languages(content: str) -> set[str]:
        languages: set[str] = set()
        for match in _INLINE_TEXT_ARTIFACT_FENCE_RE.finditer(str(content or '')):
            language = re.sub(
                r'[^a-z0-9]+',
                '',
                str(match.group('lang') or '').strip().lower(),
            )
            if language:
                languages.add(language)
        return languages

    @classmethod
    def _inline_text_artifact_requests_for_text(cls, content: str) -> list[dict[str, str]]:
        text = str(content or '').strip()
        if not text or text_artifact_content_is_materializer_instruction_echo(text):
            return []
        languages = cls._inline_text_artifact_block_languages(text)
        has_html_payload = bool(_INLINE_COMPLETE_HTML_RE.search(text)) or bool(languages & {'html', 'htm'})
        if not has_html_payload:
            return []
        requests = [{'extension': 'html', 'source_name': 'index', 'source': 'terminal_inline_output'}]
        if languages & {'css'}:
            requests.append({'extension': 'css', 'source_name': 'styles', 'source': 'terminal_inline_output'})
        return requests

    @staticmethod
    def _inline_text_source_items(payload: Mapping[str, Any]) -> list[dict[str, str]]:
        if not isinstance(payload, Mapping):
            return []
        source_items: list[dict[str, str]] = []
        seen_texts: set[str] = set()

        def add_source(raw_text: Any, source: str) -> None:
            if not isinstance(raw_text, str):
                return
            text = raw_text.strip()
            if not text or text in seen_texts:
                return
            seen_texts.add(text)
            source_items.append({'source': source, 'content': text})

        for output in payload.get('outputs') or []:
            if not isinstance(output, Mapping):
                continue
            output_type = str(output.get('type') or output.get('kind') or '').strip().lower()
            for key in ('value', 'text', 'content', 'content_payload', 'output_text'):
                raw_text = output.get(key)
                if not isinstance(raw_text, str) or not raw_text.strip():
                    continue
                if output_type and output_type not in {'text', 'document', 'html', 'css'}:
                    if not _INLINE_COMPLETE_HTML_RE.search(raw_text):
                        continue
                add_source(raw_text, f"outputs.{str(output.get('slot_id') or output.get('branch_id') or key).strip() or key}")
        add_source(payload.get('content_payload'), 'content_payload')
        add_source(payload.get('output_text'), 'output_text')
        return source_items

    @classmethod
    def find_inline_text_artifact_candidates(
        cls,
        payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for source_item in cls._inline_text_source_items(payload):
            content = source_item.get('content') or ''
            requests = cls._inline_text_artifact_requests_for_text(content)
            if not requests:
                continue
            payloads = extract_text_artifact_payloads(content, requests)
            source_candidates: list[dict[str, Any]] = []
            for payload_item in payloads:
                if not isinstance(payload_item, Mapping):
                    continue
                artifact_request = (
                    payload_item.get('artifact_request')
                    if isinstance(payload_item.get('artifact_request'), Mapping)
                    else {}
                )
                extension = re.sub(
                    r'[^a-z0-9]+',
                    '',
                    str(artifact_request.get('extension') or '').strip().lower().lstrip('.'),
                )
                if extension == 'htm':
                    extension = 'html'
                if extension not in _INLINE_HTML_CSS_EXTENSIONS:
                    continue
                candidate_content = str(payload_item.get('content') or '').strip()
                if not candidate_content or text_artifact_content_is_materializer_instruction_echo(candidate_content):
                    continue
                if extension == 'html' and not _INLINE_COMPLETE_HTML_RE.search(candidate_content):
                    continue
                if extension == 'css' and not re.search(r'(?s)[{};]', candidate_content):
                    continue
                source_name = str(
                    artifact_request.get('source_name')
                    or ('styles' if extension == 'css' else 'index')
                ).strip()
                key = (extension, source_name.lower(), candidate_content)
                if key in seen:
                    continue
                seen.add(key)
                source_candidates.append(
                    {
                        'extension': extension,
                        'source_name': source_name,
                        'content': candidate_content,
                        'source': source_item.get('source') or 'inline_output',
                        'artifact_request': {
                            'extension': extension,
                            'source_name': source_name,
                            'source': 'terminal_inline_output',
                        },
                    }
                )
            if any(candidate.get('extension') == 'html' for candidate in source_candidates):
                candidates.extend(source_candidates)
        return candidates

    @classmethod
    def _payload_has_inline_html_css_signal(cls, payload: Mapping[str, Any]) -> bool:
        for source_item in cls._inline_text_source_items(payload):
            content = source_item.get('content') or ''
            if cls._inline_text_artifact_requests_for_text(content):
                return True
        return False

    def _terminal_generated_image_records(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        records = self._collect_link_rebind_artifact_records(payload)
        return self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if (
                    str(record.get('type') or record.get('kind') or '').strip().lower() == 'image'
                    or self._artifact_record_extension(record) in _LINK_REBIND_IMAGE_EXTENSIONS
                )
                and self._artifact_record_path(record)
            ]
        )

    def _payload_has_durable_html_css_text_artifact(self, payload: Mapping[str, Any]) -> bool:
        records = self._collect_link_rebind_artifact_records(payload)
        for record in records:
            if str(record.get('type') or record.get('kind') or '').strip().lower() != 'text':
                continue
            if self._artifact_record_extension(record) not in _INLINE_HTML_CSS_EXTENSIONS:
                continue
            path = self._artifact_record_path(record)
            if not path:
                continue
            try:
                if Path(path).expanduser().is_file():
                    return True
            except OSError:
                continue
        return False

    @staticmethod
    def _inline_text_artifact_mime_type(extension: str) -> str:
        ext = str(extension or '').strip().lower().lstrip('.')
        if ext in {'html', 'htm'}:
            return 'text/html'
        if ext == 'css':
            return 'text/css'
        return 'text/plain'

    @staticmethod
    def _inline_text_artifact_open_check(
        candidate: Mapping[str, Any],
        *,
        reason: str,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        extension = str(candidate.get('extension') or 'html').strip().lower().lstrip('.') or 'html'
        source_name = str(candidate.get('source_name') or ('styles' if extension == 'css' else 'index')).strip()
        check = {
            'check_kind': 'inline_text_artifact_materialization',
            'status': 'pending',
            'evidence': 'inline_text_artifact_unmaterialized',
            'reason': reason,
            'capability': 'chat',
            'output_type': 'text',
            'role': 'text_artifact_output',
            'stage_direction': 'materialize_requested_text_artifact',
            'requires_artifact': True,
            'text_artifact_extension': extension,
            'text_artifact_source_name': source_name,
            'text_artifact_source': 'terminal_inline_output',
            'repair_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
            'recovery_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
            'repair_action_reason': 'inline HTML/CSS output must be saved as a durable text artifact before closure',
            'source': candidate.get('source') or 'inline_output',
        }
        if error:
            check['error'] = error
        return {key: value for key, value in check.items() if value not in (None, '', [], {})}

    @staticmethod
    def _canonical_terminal_inline_text_artifact_entry(
        entry: Mapping[str, Any],
        *,
        response_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw_entry = dict(entry or {})
        identity_record = {
            'type': 'text',
            'kind': 'text',
            'path': raw_entry.get('path'),
            'name': raw_entry.get('source_name') or raw_entry.get('name'),
            'mime_type': raw_entry.get('mime_type'),
            'origin': raw_entry.get('origin') or 'assistant_output',
            'source_response_id': response_payload.get('id'),
            'provenance_id': response_payload.get('provenance_id'),
            'derived_from': response_payload.get('derived_from'),
        }
        canonical = sanitize_artifact_record(
            identity_record,
            default_kind='text',
            default_origin='assistant_output',
        )
        if canonical:
            raw_entry.update(canonical)
        return {key: value for key, value in raw_entry.items() if value not in (None, '', [], {})}

    def _materialize_inline_text_artifacts_for_terminal_closure(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(payload, dict):
            return payload, []
        if not self._terminal_generated_image_records(payload):
            return payload, []
        if self._payload_has_durable_html_css_text_artifact(payload):
            return payload, []

        candidates = self.find_inline_text_artifact_candidates(payload)
        if not candidates:
            if self._payload_has_inline_html_css_signal(payload):
                return payload, [
                    self._inline_text_artifact_open_check(
                        {'extension': 'html', 'source_name': 'index', 'source': 'inline_output'},
                        reason='inline HTML/CSS-like output was present but no complete safe file payload could be extracted',
                    )
                ]
            return payload, []

        updated = dict(payload)
        model_name = str(
            ((updated.get('target') or {}).get('model') if isinstance(updated.get('target'), Mapping) else '')
            or updated.get('model')
            or updated.get('model_name')
            or 'terminal-inline-text-artifact'
        ).strip()
        saved_entries: list[dict[str, Any]] = []
        open_checks: list[dict[str, Any]] = []
        for candidate in candidates:
            extension = str(candidate.get('extension') or 'html').strip().lower().lstrip('.') or 'html'
            source_name = str(candidate.get('source_name') or ('styles' if extension == 'css' else 'index')).strip()
            content = str(candidate.get('content') or '').strip()
            try:
                saved_path = persist_text_artifact_locally(
                    content,
                    model_name=model_name,
                    source_name=source_name,
                    mode='terminal_inline_text_artifact',
                    extension=extension,
                    output_dir=ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
                    target_path=str(candidate.get('target_path') or '').strip() or None,
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning('Could not materialize inline %s artifact during terminal closure: %s', extension, exc)
                saved_path = None
                open_checks.append(
                    self._inline_text_artifact_open_check(
                        candidate,
                        reason='inline HTML/CSS output could not be saved as a durable text artifact',
                        error=str(exc),
                    )
                )
            if not saved_path:
                if not any(
                    item.get('text_artifact_extension') == extension
                    and item.get('text_artifact_source_name') == source_name
                    for item in open_checks
                ):
                    open_checks.append(
                        self._inline_text_artifact_open_check(
                            candidate,
                            reason='inline HTML/CSS output could not be saved as a durable text artifact',
                        )
                    )
                continue
            artifact_request = dict(candidate.get('artifact_request') or {})
            artifact_request.setdefault('extension', extension)
            artifact_request.setdefault('source_name', source_name)
            artifact_request.setdefault('source', 'terminal_inline_output')
            saved_entries.append(
                self._canonical_terminal_inline_text_artifact_entry(
                    {
                        'type': 'text',
                        'kind': 'text',
                        'path': saved_path,
                        'name': source_name,
                        'source_name': source_name,
                        'extension': extension,
                        'mime_type': self._inline_text_artifact_mime_type(extension),
                        'origin': 'assistant_output',
                        'document_output_kind': 'document',
                        'text_artifact_extension': extension,
                        'text_artifact_source_name': source_name,
                        'text_artifact_source': 'terminal_inline_output',
                        'text_artifact_request': artifact_request,
                        'artifact_request': artifact_request,
                        'content': content,
                        'result_text': content,
                        'source': candidate.get('source') or 'inline_output',
                    },
                    response_payload=updated,
                )
            )

        if not saved_entries:
            return updated, open_checks

        existing_saved_text_artifacts = [
            dict(item)
            for item in (updated.get('saved_text_artifacts') or [])
            if isinstance(item, Mapping)
        ]
        updated['saved_text_artifacts'] = self.merge_unique_artifact_records(
            existing_saved_text_artifacts,
            saved_entries,
        )
        first_entry = saved_entries[0]
        updated.setdefault('saved_text_path', first_entry.get('path'))
        updated.setdefault('document_output_kind', 'document')
        updated.setdefault('text_artifact_request', first_entry.get('artifact_request'))
        existing_requests = [
            dict(item)
            for item in (updated.get('text_artifact_requests') or [])
            if isinstance(item, Mapping)
        ]
        updated['text_artifact_requests'] = [
            *existing_requests,
            *[
                dict(entry.get('artifact_request') or {})
                for entry in saved_entries
                if isinstance(entry.get('artifact_request'), Mapping)
            ],
        ]
        updated['artifacts'] = self.merge_unique_artifact_records(
            updated.get('artifacts'),
            saved_entries,
        )
        late_fill = dict(updated.get('late_fill') or {}) if isinstance(updated.get('late_fill'), Mapping) else {}
        fill_results = [
            dict(item)
            for item in (late_fill.get('fill_results') or [])
            if isinstance(item, Mapping)
        ]
        completed_branches = [
            dict(item)
            for item in (late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping)
        ]
        existing_result_paths = {
            str(item.get('saved_text_path') or '').strip()
            for item in fill_results
            if str(item.get('saved_text_path') or '').strip()
        }
        for index, entry in enumerate(saved_entries, start=1):
            path = str(entry.get('path') or '').strip()
            branch_id = f'branch-terminal-inline_text_artifact-{index}'
            if path and path not in existing_result_paths:
                fill_results.append(
                    {
                        'branch_id': branch_id,
                        'phase_id': branch_id,
                        'capability': 'chat',
                        'output_type': 'text',
                        'status': 'fulfilled',
                        'saved_text_path': path,
                        'result_text': entry.get('result_text'),
                        'content_payload': entry.get('content'),
                        'text_artifact_extension': entry.get('text_artifact_extension'),
                        'text_artifact_source_name': entry.get('text_artifact_source_name'),
                        'text_artifact_source': 'terminal_inline_output',
                        'artifact_request': entry.get('artifact_request'),
                        'artifact_id': entry.get('artifact_id'),
                        'artifact_ref': entry.get('artifact_ref'),
                        'ref': entry.get('ref'),
                        'source_response_id': entry.get('source_response_id'),
                        'evidence': 'terminal_inline_text_artifact_materialized',
                    }
                )
                existing_result_paths.add(path)
            completed_branches.append(
                {
                    'branch_id': branch_id,
                    'phase_id': branch_id,
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'fulfilled',
                    'saved_text_path': path,
                    'text_artifact_extension': entry.get('text_artifact_extension'),
                    'text_artifact_source_name': entry.get('text_artifact_source_name'),
                    'text_artifact_source': 'terminal_inline_output',
                    'artifact_request': entry.get('artifact_request'),
                    'artifact_id': entry.get('artifact_id'),
                    'artifact_ref': entry.get('artifact_ref'),
                    'ref': entry.get('ref'),
                    'source_response_id': entry.get('source_response_id'),
                    'evidence': 'terminal_inline_text_artifact_materialized',
                }
            )
        late_fill['fill_results'] = fill_results
        late_fill['completed_branches'] = completed_branches
        late_fill['completed_branch_count'] = len(completed_branches)
        late_fill['inline_text_artifact_materialization_status'] = 'applied'
        late_fill['inline_text_artifact_materialized_count'] = len(saved_entries)
        late_fill['inline_text_artifact_materialized_paths'] = [
            entry.get('path') for entry in saved_entries if entry.get('path')
        ]
        updated = self.attach_late_fill_state(updated, late_fill)
        return updated, open_checks

    def _repair_terminal_text_artifact_syntax(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _TERMINAL_SYNTAX_REPAIR_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        if not text_records:
            return payload

        updated_payload = dict(payload or {})
        updated_content_by_path: dict[str, str] = {}
        repair_entries: list[dict[str, Any]] = []
        for record in text_records:
            target_path = self._artifact_record_path(record)
            extension = self._artifact_record_extension(record)
            try:
                target = Path(target_path).expanduser()
                if not target.is_file() or target.stat().st_size > 512_000:
                    continue
                original = target.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            repaired, repairs = ResponseSemanticsRuntimeOwner.repair_text_artifact_syntax_content(
                extension,
                original,
            )
            if not repairs or repaired == original:
                continue
            try:
                target.write_text(repaired, encoding='utf-8')
            except OSError as exc:
                logging.warning('Could not repair syntax in %s artifact %s: %s', extension, target_path, exc)
                continue
            updated_content_by_path[target_path] = repaired
            repair_entries.append(
                {
                    'kind': 'ollmo.text_artifact_syntax_repair',
                    'target_path': target_path,
                    'target_extension': extension,
                    'target_source_name': self._artifact_record_source_name(record),
                    'repair_count': len(repairs),
                    'repairs': repairs,
                    'status': 'applied',
                    'source': 'terminal_late_fill_syntax_repair',
                }
            )

        if not repair_entries:
            return updated_payload

        def update_text_record_content(raw_record: Any) -> Any:
            if not isinstance(raw_record, dict):
                return raw_record
            record = dict(raw_record)
            path = self._artifact_record_path(record)
            if path in updated_content_by_path:
                record['content'] = updated_content_by_path[path]
                if 'result_text' in record:
                    record['result_text'] = updated_content_by_path[path]
                if 'content_payload' in record:
                    record['content_payload'] = updated_content_by_path[path]
            return record

        if isinstance(updated_payload.get('artifacts'), list):
            updated_payload['artifacts'] = [
                update_text_record_content(item)
                for item in updated_payload.get('artifacts') or []
            ]
        if isinstance(updated_payload.get('saved_text_artifacts'), list):
            updated_payload['saved_text_artifacts'] = [
                update_text_record_content(item)
                for item in updated_payload.get('saved_text_artifacts') or []
            ]
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), Mapping)
            else {}
        )
        if isinstance(late_fill.get('fill_results'), list):
            late_fill['fill_results'] = [
                update_text_record_content(item)
                for item in late_fill.get('fill_results') or []
            ]
        existing_repairs = [
            dict(item)
            for item in (late_fill.get('syntax_sanity_repairs') or [])
            if isinstance(item, Mapping)
        ]
        late_fill['syntax_sanity_repairs'] = [*existing_repairs, *repair_entries]
        late_fill['syntax_sanity_repair_status'] = 'applied'
        late_fill['syntax_sanity_repaired_paths'] = [
            entry.get('target_path')
            for entry in repair_entries
            if entry.get('target_path')
        ]
        updated_payload['late_fill'] = late_fill
        return updated_payload

    def _refresh_terminal_text_artifacts_from_saved_files(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_path(record)
            ]
        )
        if not text_records:
            return payload

        def normalize_path(path: Any) -> str:
            token = str(path or '').strip()
            if not token:
                return ''
            return str(Path(token).expanduser().resolve(strict=False))

        updated_payload = dict(payload or {})
        refreshed_by_path: dict[str, dict[str, Any]] = {}
        refresh_entries: list[dict[str, Any]] = []
        for record in text_records:
            refreshed, metadata = refresh_text_artifact_record_from_saved_path(record)
            if metadata.get('final_text_artifact_refresh_status') != 'refreshed':
                continue
            path = normalize_path(self._artifact_record_path(refreshed))
            if not path:
                continue
            refreshed_by_path[path] = {
                key: refreshed.get(key)
                for key in (
                    'content',
                    'content_length_chars',
                    'content_preview_truncated',
                    'content_sha256',
                    'content_source',
                    'extension',
                    'file_sha256',
                    'file_size_bytes',
                    'path',
                )
                if refreshed.get(key) not in (None, '', [], {})
            }
            refresh_entries.append(
                {
                    'kind': 'ollmo.final_text_artifact_refresh',
                    'target_path': path,
                    'target_extension': metadata.get('text_artifact_extension')
                    or self._artifact_record_extension(refreshed),
                    'target_source_name': self._artifact_record_source_name(record),
                    'content_sha256': metadata.get('content_sha256'),
                    'file_sha256': metadata.get('file_sha256'),
                    'file_size_bytes': metadata.get('file_size_bytes'),
                    'syntax_sanity_status': metadata.get('syntax_sanity_status'),
                    'syntax_sanity_issue_count': metadata.get('syntax_sanity_issue_count'),
                    'status': 'applied',
                    'source': 'terminal_final_saved_text_artifact',
                }
            )
        if not refreshed_by_path:
            return updated_payload

        def refresh_text_record(raw_record: Any) -> Any:
            if not isinstance(raw_record, dict):
                return raw_record
            record = dict(raw_record)
            path = normalize_path(self._artifact_record_path(record))
            refreshed = refreshed_by_path.get(path)
            if not refreshed:
                return record
            record.update(refreshed)
            content = refreshed.get('content')
            if isinstance(content, str):
                if 'result_text' in record:
                    record['result_text'] = content
                if 'content_payload' in record:
                    record['content_payload'] = content
            return record

        if isinstance(updated_payload.get('artifacts'), list):
            updated_payload['artifacts'] = [
                refresh_text_record(item)
                for item in updated_payload.get('artifacts') or []
            ]
        if isinstance(updated_payload.get('saved_text_artifacts'), list):
            updated_payload['saved_text_artifacts'] = [
                refresh_text_record(item)
                for item in updated_payload.get('saved_text_artifacts') or []
            ]
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), Mapping)
            else {}
        )
        if isinstance(late_fill.get('fill_results'), list):
            late_fill['fill_results'] = [
                refresh_text_record(item)
                for item in late_fill.get('fill_results') or []
            ]
        existing_refreshes = [
            dict(item)
            for item in (late_fill.get('final_text_artifact_refreshes') or [])
            if isinstance(item, Mapping)
        ]
        late_fill['final_text_artifact_refreshes'] = [*existing_refreshes, *refresh_entries]
        late_fill['final_text_artifact_refresh_status'] = 'applied'
        late_fill['final_text_artifact_refreshed_paths'] = [
            entry.get('target_path')
            for entry in refresh_entries
            if entry.get('target_path')
        ]
        updated_payload['late_fill'] = late_fill
        return updated_payload

    @staticmethod
    def _terminal_materialization_contract_open_checks(
        payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
        review = runtime.get('graph_closure_review') if isinstance(runtime.get('graph_closure_review'), Mapping) else {}
        checks = review.get('checks') if isinstance(review.get('checks'), list) else []
        open_statuses = {'pending', 'planned', 'active', 'deferred', 'blocked'}
        materialization_roles = {
            'text_artifact_output',
            'linked_artifact_binding_review',
            'text_artifact_syntax_repair',
        }
        materialization_check_kinds = {
            'composed_page_image_representation',
            'hero_image_composition',
            'html_css_selector_binding',
            'linked_artifact_binding',
            'structured_dependency_join',
            'text_artifact_syntax_sanity',
            'truth_guard',
            'truth_guard_capability',
        }
        materialization_evidence = {
            'generated_image_not_represented_in_composed_page',
            'generated_image_not_represented_in_hero',
            'html_css_selector_drift',
            'unresolved_linked_artifact_binding',
            'unresolved_local_dependency_link',
            'text_artifact_syntax_issue',
            'text_only_artifact_claim_guard',
            'runtime_capability_available_but_unmaterialized',
            'runtime_capability_unverified',
            'structured_dependency_join_contract_unmet',
        }
        open_checks: list[dict[str, Any]] = []
        for raw_check in checks:
            if not isinstance(raw_check, Mapping):
                continue
            status = str(raw_check.get('status') or '').strip().lower()
            if status not in open_statuses:
                continue
            check_kind = str(raw_check.get('check_kind') or '').strip()
            evidence = str(raw_check.get('evidence') or '').strip()
            role = str(raw_check.get('role') or '').strip()
            stage_direction = str(raw_check.get('stage_direction') or '').strip()
            materialization_relevant = (
                role in materialization_roles
                or check_kind in materialization_check_kinds
                or evidence in materialization_evidence
                or stage_direction == 'materialize_requested_text_artifact'
                or raw_check.get('requires_artifact') is True
            )
            if not materialization_relevant:
                continue
            open_checks.append(dict(raw_check))
        return open_checks

    @staticmethod
    def _terminal_materialization_text_record_content(record: Mapping[str, Any]) -> str:
        path = LateFillRuntimeOwner._artifact_record_path(record)
        if not path:
            return ''
        try:
            target = Path(path).expanduser()
            if not target.is_file() or target.stat().st_size > 512_000:
                return ''
            return target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return ''

    @staticmethod
    def _terminal_materialization_record_has_link(content: str, record: Mapping[str, Any]) -> bool:
        return any(token and token in content for token in LateFillRuntimeOwner._link_tokens_for_artifact(record))

    @classmethod
    def _terminal_content_has_unresolved_link_placeholder(
        cls,
        content: str,
        dependency_records: list[dict[str, Any]],
        *,
        target_path: str = '',
    ) -> bool:
        if not content:
            return False

        for match in _LINK_REBIND_ATTR_RE.finditer(str(content or '')):
            url = str(match.group('url') or '').strip()
            if not url:
                continue
            extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            preferred_family = cls._link_rebind_attr_context_family(content, match.start())
            candidates = cls._link_rebind_records_for_extension(
                extension,
                dependency_records,
                preferred_family=preferred_family,
            )
            if cls._link_url_needs_rebind(
                url,
                candidates,
                target_path=target_path,
                preferred_family=preferred_family,
            ):
                return True

        for match in _LINK_REBIND_URL_RE.finditer(str(content or '')):
            url = str(match.group('url') or '').strip()
            if not url:
                continue
            extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            preferred_family = cls._link_rebind_css_context_family(content, match.start())
            candidates = cls._link_rebind_records_for_extension(
                extension,
                dependency_records,
                preferred_family=preferred_family,
            )
            if cls._link_url_needs_rebind(
                url,
                candidates,
                target_path=target_path,
                preferred_family=preferred_family,
            ):
                return True

        for match in _LINK_REBIND_FETCH_RE.finditer(str(content or '')):
            url = str(match.group('url') or '').strip()
            if not _static_fetch_url_is_local_file_dependency(url):
                continue
            extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
            candidates = cls._link_rebind_records_for_extension(extension, dependency_records)
            if cls._link_url_needs_rebind(
                url,
                candidates,
                target_path=target_path,
            ):
                return True
        return False

    @classmethod
    def _terminal_json_content_has_unresolved_dependency_link(
        cls,
        content: str,
        dependency_records: list[dict[str, Any]],
        *,
        target_path: str,
    ) -> bool:
        try:
            document = json.loads(str(content or ''))
        except (TypeError, ValueError, json.JSONDecodeError):
            return True

        def visit(value: Any, *, field_key: str = '') -> bool:
            if isinstance(value, Mapping):
                return any(
                    visit(child, field_key=str(key))
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return any(visit(child, field_key=field_key) for child in value)
            if not isinstance(value, str):
                return False
            normalized_key = cls._link_rebind_normalize_json_key(field_key)
            if not normalized_key or not _LINK_REBIND_JSON_PATH_KEY_RE.fullmatch(normalized_key):
                return False
            preferred_family = cls._link_rebind_json_path_key_family(field_key)
            token = value.split('?', 1)[0].split('#', 1)[0]
            extension = Path(token).suffix.lower().lstrip('.')
            candidates = cls._link_rebind_records_for_extension(
                extension,
                dependency_records,
                preferred_family=preferred_family,
            )
            if not preferred_family:
                preferred_family = cls._link_rebind_infer_single_family(candidates)
                candidates = cls._link_rebind_records_for_extension(
                    extension,
                    dependency_records,
                    preferred_family=preferred_family,
                )
            return cls._link_url_needs_rebind(
                value,
                candidates,
                target_path=target_path,
                preferred_family=preferred_family,
            )

        return visit(document)

    @staticmethod
    def _mask_inline_javascript_non_code(content: str) -> str:
        """Mask strings/comments before applying deliberately small JS checks."""
        text = str(content or '')
        token_re = re.compile(
            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`|'
            r'//[^\n]*|/\*[\s\S]*?\*/)'
        )

        def mask(match: re.Match[str]) -> str:
            return ''.join('\n' if char == '\n' else ' ' for char in match.group(0))

        return token_re.sub(mask, text)

    @classmethod
    def _inline_javascript_binding_issues(cls, content: str) -> list[str]:
        issues: list[str] = []
        for script_match in _INLINE_SCRIPT_RE.finditer(str(content or '')):
            script = cls._mask_inline_javascript_non_code(script_match.group('body') or '')
            invalid_identifier = next(
                (
                    match
                    for match in _JS_INVALID_IDENTIFIER_SEPARATOR_RE.finditer(script)
                    if len(str(match.group('separator') or '')) >= 2
                    and all(
                        unicodedata.category(char).startswith(('P', 'S'))
                        for char in str(match.group('separator') or '')
                    )
                ),
                None,
            )
            if invalid_identifier:
                issues.append(
                    'inline JavaScript contains a suspicious repeated identifier separator near '
                    f'`{invalid_identifier.group(0)}`'
                )
        return issues[:4]

    def _terminal_web_binding_contract_open_checks(
        self,
        payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Detect small, high-confidence HTML/inline-JS to local-JSON mismatches.

        This is intentionally not a JavaScript interpreter.  It only checks
        direct ``array.forEach(item => ...)``/``map`` consumers against the
        concrete JSON array they fetch, plus suspicious repeated punctuation
        inside an inline-JS identifier.  A directly consumed collection is
        required to exist and be an array; optional fields and arbitrary JS
        semantics stay outside this guard. Browser execution remains separate.
        """
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        if not text_records:
            return []
        html_paths = {
            self._artifact_record_path(record)
            for record in text_records
            if self._artifact_record_extension(record) in {'html', 'htm'}
        }
        dependency_records = [
            record
            for record in text_records
            if self._artifact_record_path(record) not in html_paths
            and self._artifact_record_extension(record)
            in _TERMINAL_MATERIALIZABLE_LOCAL_DEPENDENCY_EXTENSIONS
        ]
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        checks: list[dict[str, Any]] = []
        seen_checks: set[tuple[str, str, str]] = set()

        def dependency_for_url(url: str, source_path: str) -> Optional[dict[str, Any]]:
            token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
            requested_name = Path(token).name.lower()
            target_path = self._local_link_target_path(token, target_path=source_path)
            for record in dependency_records:
                record_path = self._artifact_record_path(record)
                if target_path and record_path and str(Path(record_path).resolve()) == target_path:
                    return record
                if record_path and Path(record_path).name.lower() == requested_name:
                    return record
                if self._artifact_record_source_name(record).lower() == Path(requested_name).stem.lower():
                    return record
            return None

        def append_check(
            target_record: Mapping[str, Any],
            reason: str,
            evidence: str,
            *,
            consumer_record: Optional[Mapping[str, Any]] = None,
        ) -> None:
            target_path = self._artifact_record_path(target_record)
            target_extension = self._artifact_record_extension(target_record)
            target_source_name = self._artifact_record_source_name(target_record)
            key = (target_path, evidence, reason)
            if not target_path or key in seen_checks:
                return
            seen_checks.add(key)
            branch_hint = self._terminal_text_branch_hint_for_dependency(
                late_fill=late_fill,
                extension=target_extension,
                source_name=target_source_name,
                target_path=target_path,
            )
            branch_id = str(
                branch_hint.get('branch_id')
                or target_record.get('branch_id')
                or ''
            ).strip()
            phase_id = str(
                branch_hint.get('phase_id')
                or target_record.get('phase_id')
                or branch_id
                or ''
            ).strip()
            target_content = self._terminal_materialization_text_record_content(target_record)
            if len(target_content) > 90_000:
                target_content = f'{target_content[:90_000].rstrip()}\n\n[content truncated for repair prompt size]'
            consumer_path = self._artifact_record_path(consumer_record or {})
            consumer_content = self._terminal_materialization_text_record_content(consumer_record or {})
            if len(consumer_content) > 45_000:
                consumer_content = f'{consumer_content[:45_000].rstrip()}\n\n[consumer content truncated for repair prompt size]'
            content_lines = [
                f'Target text artifact: {target_path}',
                'Deterministic web binding issue:',
                f'- {reason}',
            ]
            if consumer_path and consumer_path != target_path:
                content_lines.extend(
                    [
                        f'Bound local consumer: {consumer_path}',
                        '--- CURRENT CONSUMER START ---',
                        consumer_content,
                        '--- CURRENT CONSUMER END ---',
                    ]
                )
            content_lines.extend(
                [
                    '--- CURRENT TARGET START ---',
                    target_content,
                    '--- CURRENT TARGET END ---',
                    'Repair only the target artifact so the deterministic local web binding is coherent. '
                    'Preserve valid copy, layout, navigation, and artifact links. Output only the complete target file body.',
                ]
            )
            checks.append(
                {
                    'check_kind': 'web_runtime_binding',
                    'status': 'pending',
                    'evidence': evidence,
                    'role': 'linked_artifact_binding_review',
                    'requires_artifact': True,
                    'repair_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                    'recovery_action': RECOVERY_ACTION_RETRY_SAME_BRANCH,
                    'repair_action_reason': 'saved local web artifacts violate a deterministic binding contract',
                    'reason': reason,
                    'branch_id': branch_id or None,
                    'phase_id': phase_id or None,
                    'source_path': consumer_path or target_path,
                    'dependency_source_path': consumer_path or None,
                    'content_payload': '\n'.join(content_lines).strip(),
                    'stage_direction': 'materialize_requested_text_artifact',
                    'dependency_policy': 'target_artifact_snapshot_only',
                    'text_artifact_extension': target_extension,
                    'text_artifact_source_name': target_source_name,
                    'text_artifact_source': 'closure_web_binding_repair',
                    'text_artifact_target_path': target_path,
                    'artifact_request': {
                        'extension': target_extension,
                        'source_name': target_source_name,
                        'source': 'closure_web_binding_repair',
                        'target_path': target_path,
                    },
                    'content_payload_source': 'terminal_web_runtime_binding_review',
                    'review_criteria': ['static_local_web_binding'],
                }
            )

        for source_record in text_records:
            extension = self._artifact_record_extension(source_record)
            if extension not in {'html', 'htm', 'js', 'mjs', 'cjs'}:
                continue
            source_path = self._artifact_record_path(source_record)
            content = self._terminal_materialization_text_record_content(source_record)
            if not content:
                continue
            javascript_content = (
                content
                if extension in {'html', 'htm'}
                else f'<script>{content}</script>'
            )
            for issue in self._inline_javascript_binding_issues(javascript_content):
                append_check(
                    source_record,
                    issue,
                    'inline_javascript_binding_mismatch',
                    consumer_record=source_record,
                )

            for fetch_match in _LINK_REBIND_FETCH_RE.finditer(content):
                url = str(fetch_match.group('url') or '').strip()
                if (
                    not _static_fetch_url_is_local_file_dependency(url)
                    or not url.lower().split('?', 1)[0].split('#', 1)[0].endswith('.json')
                ):
                    continue
                dependency = dependency_for_url(url, source_path)
                if not dependency:
                    continue
                dependency_content = self._terminal_materialization_text_record_content(dependency)
                try:
                    document = json.loads(dependency_content)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                script_start = fetch_match.end()
                next_fetch = _LINK_REBIND_FETCH_RE.search(content, script_start)
                script_end = next_fetch.start() if next_fetch else len(content)
                script_content = content[script_start:script_end]
                fetch_prefix = content[max(0, fetch_match.start() - 240):fetch_match.start()]
                response_assignment = _JS_FETCH_RESPONSE_ASSIGNMENT_RE.search(fetch_prefix)
                response_name = str(
                    response_assignment.group('response')
                    if response_assignment
                    else ''
                ).strip()
                bound_json_roots = {
                    str(binding.group('binding') or '').strip()
                    for binding in _JS_AWAITED_JSON_BINDING_RE.finditer(script_content)
                    if response_name
                    and str(binding.group('response') or '').strip() == response_name
                    and str(binding.group('binding') or '').strip()
                }
                for iterator in _JS_COLLECTION_ITERATOR_RE.finditer(script_content):
                    collection_path = re.sub(r'\s+', '', iterator.group('collection') or '')
                    collection_key = collection_path.rsplit('.', 1)[-1]
                    body_end = script_content.find('});', iterator.end())
                    body = script_content[iterator.end(): body_end if body_end >= 0 else iterator.end() + 12000]
                    item_name = iterator.group('item')
                    properties = {
                        match.group(1)
                        for match in re.finditer(
                            rf'\b{re.escape(item_name)}\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)\b',
                            body,
                        )
                    }
                    path_parts = [part for part in collection_path.split('.') if part]
                    bound_collection = bool(
                        path_parts
                        and path_parts[0] in bound_json_roots
                    )
                    collection_found = True
                    values: Any
                    if bound_collection:
                        values = document
                        for path_part in path_parts[1:]:
                            if not isinstance(values, Mapping) or path_part not in values:
                                collection_found = False
                                values = None
                                break
                            values = values[path_part]
                    else:
                        values = document.get(collection_key) if isinstance(document, Mapping) else None
                    if bound_collection and not collection_found:
                        append_check(
                            dependency,
                            f'Fetched JSON path `{".".join(path_parts[1:]) or collection_path}` is consumed as `{collection_path}` but is missing',
                            'static_json_consumer_contract_mismatch',
                            consumer_record=source_record,
                        )
                        continue
                    if bound_collection and not isinstance(values, list):
                        append_check(
                            dependency,
                            f'Fetched JSON path `{".".join(path_parts[1:]) or collection_path}` is consumed as `{collection_path}` but is not an array',
                            'static_json_consumer_contract_mismatch',
                            consumer_record=source_record,
                        )
                        continue
                    if not isinstance(values, list) or not values:
                        continue
                    if not all(isinstance(item, Mapping) for item in values):
                        if bound_collection and properties:
                            append_check(
                                dependency,
                                f'JSON collection `{collection_path}` is consumed as object items but contains a non-object value',
                                'static_json_consumer_contract_mismatch',
                                consumer_record=source_record,
                            )
                        continue
                    for property_name in sorted(properties):
                        if any(property_name not in item for item in values):
                            append_check(
                                dependency,
                                f'JSON collection `{collection_key}` is consumed as `{item_name}.{property_name}` but at least one item lacks that field',
                                'static_json_consumer_contract_mismatch',
                                consumer_record=source_record,
                            )
        return checks

    @classmethod
    def _terminal_linked_artifact_dependency_records(
        cls,
        records: list[dict[str, Any]],
        text_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        html_paths = {
            cls._artifact_record_path(record)
            for record in text_records
            if cls._artifact_record_extension(record) in {'html', 'htm'}
        }
        dependency_records: list[dict[str, Any]] = []
        for record in records:
            path = cls._artifact_record_path(record)
            if not path or path in html_paths:
                continue
            artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
            extension = cls._artifact_record_extension(record)
            if artifact_type == 'text' and extension not in _LINK_REBIND_HTML_TARGET_EXTENSIONS:
                continue
            dependency_records.append(record)
        return cls._unique_artifact_records_by_path(dependency_records)

    def _terminal_text_branch_hint_for_dependency(
        self,
        *,
        late_fill: Mapping[str, Any],
        extension: str,
        source_name: str,
        target_path: str,
    ) -> dict[str, Any]:
        target_extension = str(extension or '').strip().lower().lstrip('.')
        target_source_name = str(source_name or '').strip().lower()
        normalized_target_path = str(target_path or '').strip()
        for branch_key in ('completed_branches', 'pending_branches', 'active_branches', 'failed_branches'):
            for branch in late_fill.get(branch_key) or []:
                if not isinstance(branch, Mapping):
                    continue
                artifact_request = (
                    branch.get('artifact_request')
                    if isinstance(branch.get('artifact_request'), Mapping)
                    else {}
                )
                branch_extension = str(
                    branch.get('text_artifact_extension')
                    or artifact_request.get('extension')
                    or ''
                ).strip().lower().lstrip('.')
                branch_source_name = str(
                    branch.get('text_artifact_source_name')
                    or artifact_request.get('source_name')
                    or ''
                ).strip().lower()
                branch_target_path = str(
                    branch.get('text_artifact_target_path')
                    or artifact_request.get('target_path')
                    or ''
                ).strip()
                if normalized_target_path and branch_target_path and normalized_target_path == branch_target_path:
                    return dict(branch)
                if target_extension and target_source_name and branch_extension == target_extension and branch_source_name == target_source_name:
                    return dict(branch)
        return {}

    def _terminal_unresolved_local_dependency_link_open_checks(
        self,
        payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        if not text_records:
            return []
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        checks: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def append_missing_link_check(source_record: Mapping[str, Any], url: str) -> None:
            source_path = self._artifact_record_path(source_record)
            token = str(url or '').split('?', 1)[0].split('#', 1)[0].strip()
            extension = Path(token).suffix.lower().lstrip('.')
            if extension not in _TERMINAL_MATERIALIZABLE_LOCAL_DEPENDENCY_EXTENSIONS:
                return
            target_path = self._local_link_target_path(token, target_path=source_path)
            if not target_path or self._local_link_target_exists(token, target_path=source_path):
                return
            key = (source_path, token, target_path)
            if key in seen:
                return
            seen.add(key)
            source_content = self._terminal_materialization_text_record_content(source_record)
            if len(source_content) > 90_000:
                source_content = f'{source_content[:90_000].rstrip()}\n\n[content truncated for repair prompt size]'
            source_name = Path(token).stem or ('styles' if extension == 'css' else f'generated-{extension}')
            branch_hint = self._terminal_text_branch_hint_for_dependency(
                late_fill=late_fill,
                extension=extension,
                source_name=source_name,
                target_path=target_path,
            )
            branch_id = str(
                branch_hint.get('branch_id')
                or source_record.get('branch_id')
                or ''
            ).strip()
            phase_id = str(
                branch_hint.get('phase_id')
                or source_record.get('phase_id')
                or branch_id
                or ''
            ).strip()
            if extension == 'json':
                repair_instruction = (
                    f'Create or repair only the missing `{Path(target_path).name}` JSON artifact. '
                    'Use the static fetch consumer and its expected data access as binding context. '
                    'Output only the complete, valid JSON document.'
                )
            else:
                repair_instruction = (
                    f'Create or repair only the missing `{Path(target_path).name}` {extension} artifact. '
                    'Use the current source file selectors, classes, ids, and script/style expectations as binding context. '
                    'Preserve the existing HTML structure, copy, image links, and layout intent. '
                    'Output only the complete target file body.'
                )
            content_payload = '\n'.join(
                [
                    f'Target text artifact: {target_path}',
                    f'Missing local dependency linked from: {source_path}',
                    f'Missing link token: {url}',
                    f'Dependency type: {extension}',
                    'Current saved source file content:',
                    '--- CURRENT SOURCE FILE START ---',
                    source_content,
                    '--- CURRENT SOURCE FILE END ---',
                    repair_instruction,
                ]
            ).strip()
            checks.append(
                {
                    'check_kind': 'linked_artifact_binding',
                    'status': 'pending',
                    'evidence': 'unresolved_local_dependency_link',
                    'reason': 'final web artifact links a local dependency file that does not exist or is not materialized',
                    'capability': 'chat',
                    'output_type': 'text',
                    'role': 'linked_artifact_binding_review',
                    'branch_id': branch_id or None,
                    'phase_id': phase_id or None,
                    'requires_artifact': True,
                    'repair_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
                    'recovery_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
                    'repair_action_reason': 'missing local web dependency must be materialized before terminal closure',
                    'content_payload': content_payload,
                    'content_payload_source': 'closure_local_dependency_link_review',
                    'stage_direction': 'materialize_requested_text_artifact',
                    'dependency_source_path': source_path,
                    'missing_dependency_url': str(url or '').strip(),
                    'missing_dependency_target_path': target_path,
                    'text_artifact_extension': extension,
                    'text_artifact_source_name': source_name,
                    'text_artifact_source': 'closure_local_dependency_link',
                    'text_artifact_target_path': target_path,
                    'artifact_request': {
                        'extension': extension,
                        'source_name': source_name,
                        'source': 'closure_local_dependency_link',
                        'target_path': target_path,
                    },
                }
            )

        for record in text_records:
            source_path = self._artifact_record_path(record)
            content = self._terminal_materialization_text_record_content(record)
            if not source_path or not content:
                continue
            for match in _LINK_REBIND_ATTR_RE.finditer(content):
                append_missing_link_check(record, str(match.group('url') or '').strip())
            for match in _LINK_REBIND_URL_RE.finditer(content):
                append_missing_link_check(record, str(match.group('url') or '').strip())
            for match in _LINK_REBIND_FETCH_RE.finditer(content):
                url = str(match.group('url') or '').strip()
                if _static_fetch_url_is_local_file_dependency(url):
                    append_missing_link_check(record, url)
        return checks

    def _terminal_linked_artifact_contract_required(self, payload: Mapping[str, Any]) -> bool:
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = [
            record
            for record in records
            if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
            and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
            and self._artifact_record_path(record)
        ]
        if not text_records:
            return False
        return bool(self._terminal_linked_artifact_dependency_records(records, text_records))

    def _terminal_linked_artifact_contract_is_fulfilled(self, payload: Mapping[str, Any]) -> bool:
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = [
            record
            for record in records
            if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
            and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
            and self._artifact_record_path(record)
        ]
        if not text_records:
            return False
        html_records = [
            record
            for record in text_records
            if self._artifact_record_extension(record) in {'html', 'htm'}
        ]
        if self._terminal_unresolved_local_dependency_link_open_checks(payload):
            return False
        if self._terminal_web_binding_contract_open_checks(payload):
            return False
        dependency_records = self._terminal_linked_artifact_dependency_records(records, text_records)
        if not html_records and not dependency_records:
            return False

        content_by_path: dict[str, str] = {}
        for record in text_records:
            path = self._artifact_record_path(record)
            content = self._terminal_materialization_text_record_content(record)
            if path and content:
                content_by_path[path] = content
        if not content_by_path:
            return False

        dependency_extensions = {
            self._artifact_record_extension(record)
            for record in dependency_records
            if self._artifact_record_extension(record)
        }
        text_record_by_path = {
            self._artifact_record_path(record): record
            for record in text_records
            if self._artifact_record_path(record)
        }
        for target_path, content in content_by_path.items():
            consumer_record = text_record_by_path.get(target_path, {})
            consumer_dependency_records, _dependency_ids, _selection_policy = (
                self._link_rebind_asset_records_for_consumer(
                    payload,
                    consumer_record,
                    dependency_records,
                )
            )
            if _selection_policy == 'consumer_declared_media_dependency_missing':
                return False
            if _dependency_ids and not consumer_dependency_records:
                return False
            if self._artifact_record_extension(consumer_record) == 'json':
                unresolved = self._terminal_json_content_has_unresolved_dependency_link(
                    content,
                    consumer_dependency_records,
                    target_path=target_path,
                )
            else:
                unresolved = self._terminal_content_has_unresolved_link_placeholder(
                    content,
                    consumer_dependency_records,
                    target_path=target_path,
                )
            if unresolved:
                return False
            for match in _LINK_REBIND_ATTR_RE.finditer(content):
                url = str(match.group('url') or '').strip()
                extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
                if (
                    extension
                    and extension in dependency_extensions
                    and not self._local_link_target_exists(url, target_path=target_path)
                ):
                    return False
            for match in _LINK_REBIND_URL_RE.finditer(content):
                url = str(match.group('url') or '').strip()
                extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
                if (
                    extension
                    and extension in dependency_extensions
                    and not self._local_link_target_exists(url, target_path=target_path)
                ):
                    return False
            for match in _LINK_REBIND_FETCH_RE.finditer(content):
                url = str(match.group('url') or '').strip()
                if not _static_fetch_url_is_local_file_dependency(url):
                    continue
                extension = Path(url.split('?', 1)[0].split('#', 1)[0]).suffix.lower().lstrip('.')
                if (
                    extension
                    and extension in _TERMINAL_MATERIALIZABLE_LOCAL_DEPENDENCY_EXTENSIONS
                    and not self._local_link_target_exists(url, target_path=target_path)
                ):
                    return False

        html_content = '\n\n'.join(
            content_by_path.get(self._artifact_record_path(record), '')
            for record in html_records
        )
        all_link_content = '\n\n'.join(content_by_path.values())
        for record in dependency_records:
            extension = self._artifact_record_extension(record)
            record_type = str(record.get('type') or record.get('kind') or '').strip().lower()
            required_content = (
                html_content
                if html_records and record_type == 'text' and extension in _LINK_REBIND_HTML_TARGET_EXTENSIONS
                else all_link_content
            )
            target_path = self._artifact_record_path(record)
            if target_path:
                exact_sources = [
                    (source_path, source_content)
                    for source_path, source_content in content_by_path.items()
                    if source_path and source_path != target_path and source_content
                ]
                if not any(
                    self._terminal_exact_content_has_artifact_link(
                        source_content,
                        source_path=source_path,
                        target_path=target_path,
                    )
                    for source_path, source_content in exact_sources
                ):
                    return False
                continue
            if not self._terminal_materialization_record_has_link(required_content, record):
                return False
        return True

    @staticmethod
    def _terminal_exact_link_tokens_for_artifact(
        *,
        source_path: str,
        target_path: str,
    ) -> set[str]:
        tokens: set[str] = set()
        target = str(target_path or '').strip()
        if not target:
            return tokens
        tokens.add(target)
        target_name = Path(target).name
        if target_name:
            tokens.add(target_name)
        if source_path:
            relative = LateFillRuntimeOwner._relative_artifact_link(
                from_path=source_path,
                to_path=target,
            )
            if relative:
                tokens.add(relative)
        return {token for token in tokens if token}

    @staticmethod
    def _terminal_exact_content_has_artifact_link(
        content: str,
        *,
        source_path: str,
        target_path: str,
    ) -> bool:
        text = str(content or '')
        if not text.strip():
            return False
        return any(
            token and token in text
            for token in LateFillRuntimeOwner._terminal_exact_link_tokens_for_artifact(
                source_path=source_path,
                target_path=target_path,
            )
        )

    @staticmethod
    def _terminal_html_hero_snippets(content: str) -> list[str]:
        text = str(content or '')
        snippets: list[str] = []
        for match in _TERMINAL_HTML_HERO_BLOCK_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            attrs = str(match.group('attrs') or '')
            if tag != 'header' and not _TERMINAL_HERO_LOCAL_IMAGE_SIGNAL_RE.search(attrs):
                continue
            snippet = match.group(0)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        return snippets

    @staticmethod
    def _terminal_css_hero_snippets(content: str) -> list[str]:
        text = str(content or '')
        snippets: list[str] = []
        for match in _TERMINAL_CSS_HERO_BLOCK_RE.finditer(text):
            snippet = match.group(0)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        return snippets

    @staticmethod
    def _terminal_html_inline_style_blocks(content: str) -> list[str]:
        text = str(content or '')
        return [
            str(match.group('body') or '')
            for match in re.finditer(
                r'(?is)<style\b[^>]*>(?P<body>.*?)</style\s*>',
                text,
            )
            if str(match.group('body') or '').strip()
        ]

    @classmethod
    def _terminal_hero_local_image_required(
        cls,
        prompt_text: str,
        *,
        html_contents: list[str],
        css_contents: list[str],
    ) -> bool:
        if _TERMINAL_HERO_LOCAL_IMAGE_SIGNAL_RE.search(str(prompt_text or '')):
            return True
        return any(cls._terminal_html_hero_snippets(content) for content in html_contents) or any(
            cls._terminal_css_hero_snippets(content) for content in css_contents
        )

    def _terminal_hero_image_composition_open_check(
        self,
        payload: Mapping[str, Any],
        *,
        request_payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        html_records = [
            record
            for record in text_records
            if self._artifact_record_extension(record) in {'html', 'htm'}
        ]
        if not html_records:
            return None
        css_records = [
            record
            for record in text_records
            if self._artifact_record_extension(record) == 'css'
        ]
        image_records = self._terminal_generated_image_records(payload)
        if not image_records:
            return None

        content_by_path: dict[str, str] = {}
        for record in text_records:
            source_path = self._artifact_record_path(record)
            content = self._terminal_materialization_text_record_content(record)
            if source_path and content:
                content_by_path[source_path] = content
        if not content_by_path:
            return None

        html_contents = [
            content_by_path.get(self._artifact_record_path(record), '')
            for record in html_records
        ]
        html_inline_css_contents = [
            style_block
            for content in html_contents
            for style_block in self._terminal_html_inline_style_blocks(content)
        ]
        css_contents = [
            content_by_path.get(self._artifact_record_path(record), '')
            for record in css_records
        ] + html_inline_css_contents
        prompt_text = str(
            (request_payload or {}).get('prompt')
            or (request_payload or {}).get('input')
            or ((payload.get('request') or {}).get('prompt') if isinstance(payload.get('request'), Mapping) else '')
            or ''
        ).strip()
        if not self._terminal_hero_local_image_required(
            prompt_text,
            html_contents=html_contents,
            css_contents=css_contents,
        ):
            return None

        snippets: list[tuple[str, str]] = []
        for record in html_records:
            source_path = self._artifact_record_path(record)
            for snippet in self._terminal_html_hero_snippets(content_by_path.get(source_path, '')):
                snippets.append((source_path, snippet))
        for record in css_records:
            source_path = self._artifact_record_path(record)
            for snippet in self._terminal_css_hero_snippets(content_by_path.get(source_path, '')):
                snippets.append((source_path, snippet))
        for record in html_records:
            source_path = self._artifact_record_path(record)
            html_content = content_by_path.get(source_path, '')
            for style_block in self._terminal_html_inline_style_blocks(html_content):
                for snippet in self._terminal_css_hero_snippets(style_block):
                    snippets.append((source_path, snippet))
        if not snippets:
            return None

        for source_path, snippet in snippets:
            for image_record in image_records:
                image_path = self._artifact_record_path(image_record)
                if (
                    image_path
                    and self._terminal_exact_content_has_artifact_link(
                        snippet,
                        source_path=source_path,
                        target_path=image_path,
                    )
                ):
                    return None

        target_record = next(
            (
                record for record in css_records
                if self._terminal_css_hero_snippets(content_by_path.get(self._artifact_record_path(record), ''))
            ),
            html_records[0],
        )
        target_extension = self._artifact_record_extension(target_record) or 'html'
        target_name = self._artifact_record_source_name(target_record) or ('styles' if target_extension == 'css' else 'index')
        target_path = self._artifact_record_path(target_record)
        target_content = content_by_path.get(target_path, '') if target_path else ''
        bounded_content = target_content
        if len(bounded_content) > 90_000:
            bounded_content = f'{bounded_content[:90_000].rstrip()}\n\n[content truncated for repair prompt size]'
        generated_paths = [
            self._artifact_record_path(record)
            for record in image_records
            if self._artifact_record_path(record)
        ]
        generated_lines = [
            f'- {path} (relative from target: {self._relative_artifact_link(from_path=target_path, to_path=path)})'
            for path in generated_paths
        ]
        content_payload = '\n'.join(
            [
                f'Target text artifact: {target_path or target_name}',
                'Hero image composition defect:',
                '- The composed page has a hero/header area but no concrete generated image artifact path is used in that hero area.',
                'Generated image artifact paths available for hero binding:',
                *generated_lines,
                'Current saved target file content:',
                '--- CURRENT SAVED FILE START ---',
                bounded_content,
                '--- CURRENT SAVED FILE END ---',
                'Update only the target text artifact. Bind one concrete generated image path above into the hero/header image or background. Preserve existing valid links, copy, layout intent, and unrelated structure.',
            ]
        ).strip()
        return {
            'check_kind': 'hero_image_composition',
            'status': 'pending',
            'evidence': 'generated_image_not_represented_in_hero',
            'reason': 'final composed page hero/header area does not use a generated local image artifact',
            'capability': 'chat',
            'output_type': 'text',
            'role': 'linked_artifact_binding_review',
            'branch_id': str(target_record.get('branch_id') or '').strip() or None,
            'phase_id': str(target_record.get('phase_id') or target_record.get('branch_id') or '').strip() or None,
            'requires_artifact': True,
            'repair_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            'recovery_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            'repair_action_reason': 'final composed page must bind a generated local image into the hero/header area',
            'content_payload': content_payload,
            'content_payload_source': 'closure_hero_image_composition',
            'stage_direction': 'materialize_requested_text_artifact',
            'generated_image_count': len(generated_paths),
            'represented_image_count': 0,
            'missing_image_count': 1,
            'missing_image_paths': generated_paths[:8],
            'text_artifact_extension': target_extension,
            'text_artifact_source_name': target_name,
            'text_artifact_source': 'closure_hero_image_composition',
            'text_artifact_target_path': target_path,
            'artifact_request': {
                'extension': target_extension,
                'source_name': target_name,
                'source': 'closure_hero_image_composition',
                'target_path': target_path,
            },
        }

    @staticmethod
    def _css_url_quote(value: str) -> str:
        return str(value or '').replace('\\', '/').replace("'", "\\'")

    @classmethod
    def _repair_terminal_hero_css_content(
        cls,
        content: str,
        *,
        image_link: str,
    ) -> tuple[str, str]:
        text = str(content or '')
        link = cls._css_url_quote(image_link)
        if not text.strip() or not link:
            return text, ''
        for match in _TERMINAL_CSS_HERO_BLOCK_RE.finditer(text):
            block = match.group(0)
            if image_link in block or Path(image_link).name in block:
                return text, ''
            body = str(match.group('body') or '')
            for declaration_match in re.finditer(
                r'(?P<prefix>\bbackground(?:-image)?\s*:\s*)'
                r'(?P<value>[^;{}]*)(?P<suffix>;|$)',
                body,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                url_match = _LINK_REBIND_URL_RE.search(str(declaration_match.group('value') or ''))
                if not url_match:
                    continue
                url = str(url_match.group('url') or '').strip()
                if cls._url_is_external_or_empty(url) or _LINK_REBIND_PLACEHOLDER_RE.search(url):
                    url_start = declaration_match.start('value') + url_match.start('url')
                    url_end = declaration_match.start('value') + url_match.end('url')
                    next_body = f'{body[:url_start]}{link}{body[url_end:]}'
                    return (
                        f'{text[:match.start("body")]}{next_body}{text[match.end("body"):]}',
                        'replace_hero_background_url',
                    )

            insertion = (
                "\n    background-image: url('"
                f"{link}"
                "');\n"
                '    background-size: cover;\n'
                '    background-position: center;\n'
                '    background-repeat: no-repeat;'
            )
            next_body = f'{body.rstrip()}{insertion}\n'
            return (
                f'{text[:match.start("body")]}{next_body}{text[match.end("body"):]}',
                'append_hero_background_image',
            )
        return text, ''

    @classmethod
    def _repair_terminal_hero_html_content(
        cls,
        content: str,
        *,
        image_link: str,
    ) -> tuple[str, str]:
        text = str(content or '')
        link = str(image_link or '').replace('\\', '/')
        if not text.strip() or not link:
            return text, ''
        css_declaration = (
            f"background-image: url('{cls._html_attribute_escape(link)}'); "
            'background-size: cover; background-position: center; background-repeat: no-repeat;'
        )
        for match in _TERMINAL_HTML_HERO_BLOCK_RE.finditer(text):
            tag = str(match.group('tag') or '').strip().lower()
            attrs = str(match.group('attrs') or '')
            if tag != 'header' and not _TERMINAL_HERO_LOCAL_IMAGE_SIGNAL_RE.search(attrs):
                continue
            block = match.group(0)
            if image_link in block or Path(image_link).name in block:
                return text, ''
            style_match = re.search(
                r'\bstyle\s*=\s*(?P<quote>[\'"])(?P<value>.*?)(?P=quote)',
                attrs,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if style_match:
                style_value = str(style_match.group('value') or '').strip()
                separator = '; ' if style_value and not style_value.endswith(';') else ' '
                next_style = f'{style_value}{separator}{css_declaration}'.strip()
                next_attrs = (
                    f'{attrs[:style_match.start("value")]}'
                    f'{next_style}'
                    f'{attrs[style_match.end("value"):]}'
                )
                operation = 'append_hero_inline_background_image'
            else:
                next_attrs = f'{attrs} style="{css_declaration}"'
                operation = 'add_hero_inline_background_image'
            return (
                f'{text[:match.start("attrs")]}{next_attrs}{text[match.end("attrs"):]}',
                operation,
            )
        return text, ''

    def _repair_terminal_hero_image_composition(
        self,
        payload: dict[str, Any],
        *,
        request_payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        check = self._terminal_hero_image_composition_open_check(
            payload,
            request_payload=request_payload,
        )
        if not check:
            return payload
        target_path = str(check.get('text_artifact_target_path') or '').strip()
        if not target_path:
            return payload
        image_paths = [
            str(path or '').strip()
            for path in (check.get('missing_image_paths') or [])
            if str(path or '').strip()
        ]
        image_path = next(
            (
                path for path in image_paths
                if Path(path).expanduser().is_file()
            ),
            '',
        )
        if not image_path:
            return payload
        try:
            target = Path(target_path).expanduser()
            if not target.is_file() or target.stat().st_size > 512_000:
                return payload
            original = target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return payload

        image_link = self._relative_artifact_link(from_path=target_path, to_path=image_path)
        extension = str(
            check.get('text_artifact_extension')
            or Path(target_path).suffix.lower().lstrip('.')
        ).strip().lower()
        if extension == 'css':
            repaired, operation = self._repair_terminal_hero_css_content(
                original,
                image_link=image_link,
            )
        elif extension in {'html', 'htm'}:
            repaired, operation = self._repair_terminal_hero_html_content(
                original,
                image_link=image_link,
            )
        else:
            return payload
        if not operation or repaired == original:
            return payload

        snippets = (
            self._terminal_css_hero_snippets(repaired)
            if extension == 'css'
            else self._terminal_html_hero_snippets(repaired)
        )
        if not any(
            self._terminal_exact_content_has_artifact_link(
                snippet,
                source_path=target_path,
                target_path=image_path,
            )
            for snippet in snippets
        ):
            return payload

        original_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            extension,
            original,
        )
        repaired_issues = ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            extension,
            repaired,
        )
        if len(repaired_issues) > len(original_issues):
            logging.warning(
                'Skipping hero image composition repair for %s because syntax sanity regressed from %s to %s issue(s)',
                target_path,
                len(original_issues),
                len(repaired_issues),
            )
            return payload

        try:
            target.write_text(repaired, encoding='utf-8')
        except OSError as exc:
            logging.warning('Could not repair hero image composition in %s: %s', target_path, exc)
            return payload

        updated_payload = dict(payload or {})
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), Mapping)
            else {}
        )
        repair_entry = {
            'kind': 'ollmo.hero_image_composition_repair',
            'operation': operation,
            'target_path': target_path,
            'target_extension': extension,
            'target_source_name': check.get('text_artifact_source_name'),
            'to_path': image_path,
            'to_token': image_link,
            'status': 'applied',
            'source': 'terminal_hero_image_composition_repair',
        }
        existing_repairs = [
            dict(item)
            for item in (late_fill.get('hero_image_composition_repairs') or [])
            if isinstance(item, Mapping)
        ]
        late_fill['hero_image_composition_repairs'] = [*existing_repairs, repair_entry]
        late_fill['hero_image_composition_repair_status'] = 'applied'
        late_fill['hero_image_composition_repaired_paths'] = [image_path]
        updated_payload['late_fill'] = late_fill
        return updated_payload

    def _terminal_composed_page_image_representation_open_check(
        self,
        payload: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        html_records = [
            record
            for record in text_records
            if self._artifact_record_extension(record) in {'html', 'htm'}
        ]
        if not html_records:
            return None

        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        image_branch_ids = {
            self.branch_id(item)
            for item in [
                *(late_fill.get('completed_branches') or []),
                *(late_fill.get('fill_results') or []),
            ]
            if isinstance(item, Mapping)
            and (
                self.normalize_capability(item.get('capability')) == self.capability_image_generation
                or str(item.get('output_type') or '').strip().lower() == 'image'
                or str(item.get('saved_image_path') or '').strip()
            )
            and self.branch_id(item)
        }
        if not image_branch_ids:
            return None

        image_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if (
                    str(record.get('type') or record.get('kind') or '').strip().lower() == 'image'
                    or self._artifact_record_extension(record) in _LINK_REBIND_IMAGE_EXTENSIONS
                )
                and self._artifact_record_path(record)
            ]
        )
        generated_image_records: list[dict[str, Any]] = []
        for record in image_records:
            record_branch_id = self.branch_id(record)
            if record_branch_id and record_branch_id in image_branch_ids:
                generated_image_records.append(record)
            elif not record_branch_id and len(image_records) == len(image_branch_ids):
                generated_image_records.append(record)
        if not generated_image_records:
            generated_image_records = image_records if image_records else []
        generated_image_records.sort(key=self._image_artifact_record_sort_key)
        if len(generated_image_records) <= 1:
            return None

        content_by_path: dict[str, str] = {}
        for record in text_records:
            source_path = self._artifact_record_path(record)
            content = self._terminal_materialization_text_record_content(record)
            if source_path and content:
                content_by_path[source_path] = content
        if not content_by_path:
            return None

        represented_paths: set[str] = set()
        for image_record in generated_image_records:
            image_path = self._artifact_record_path(image_record)
            if not image_path:
                continue
            for source_path, content in content_by_path.items():
                if self._terminal_exact_content_has_artifact_link(
                    content,
                    source_path=source_path,
                    target_path=image_path,
                ):
                    represented_paths.add(image_path)
                    break

        generated_paths = [
            self._artifact_record_path(record)
            for record in generated_image_records
            if self._artifact_record_path(record)
        ]
        missing_paths = [path for path in generated_paths if path not in represented_paths]
        if not missing_paths:
            return None

        target_record = html_records[0] if html_records else text_records[0]
        target_extension = self._artifact_record_extension(target_record) or 'html'
        target_name = self._artifact_record_source_name(target_record) or 'index'
        target_path = self._artifact_record_path(target_record)
        target_content = content_by_path.get(target_path, '') if target_path else ''
        bounded_content = target_content
        if len(bounded_content) > 90_000:
            bounded_content = f'{bounded_content[:90_000].rstrip()}\n\n[content truncated for repair prompt size]'
        generated_lines = [
            f'- {path} (relative from target: {self._relative_artifact_link(from_path=target_path, to_path=path)})'
            for path in generated_paths
        ]
        missing_lines = [
            f'- {path} (relative from target: {self._relative_artifact_link(from_path=target_path, to_path=path)})'
            for path in missing_paths
        ]
        represented_lines = [
            f'- {path}'
            for path in sorted(represented_paths)
        ]
        content_payload = '\n'.join(
            [
                f'Target text artifact: {target_path or target_name}',
                'Composed page image representation defect:',
                f'- Generated image artifacts: {len(generated_paths)}',
                f'- Represented image artifacts: {len(represented_paths)}',
                f'- Missing image artifacts: {len(missing_paths)}',
                'Missing generated image artifact paths to add:',
                *(missing_lines or ['- none']),
                'All generated image artifact paths:',
                *generated_lines,
                'Already represented generated image artifact paths:',
                *(represented_lines or ['- none']),
                'Current saved target file content:',
                '--- CURRENT SAVED FILE START ---',
                bounded_content,
                '--- CURRENT SAVED FILE END ---',
                'Update only the target HTML artifact. Add concrete references for the missing generated image artifact paths above, using relative paths from the target file when practical. Preserve existing valid links, copy, layout intent, and unrelated structure.',
            ]
        ).strip()
        return {
            'check_kind': 'composed_page_image_representation',
            'status': 'pending',
            'evidence': 'generated_image_not_represented_in_composed_page',
            'reason': 'final composed page does not represent every required generated image artifact',
            'capability': 'chat',
            'output_type': 'text',
            'role': 'linked_artifact_binding_review',
            'branch_id': str(target_record.get('branch_id') or '').strip() or None,
            'phase_id': str(target_record.get('phase_id') or target_record.get('branch_id') or '').strip() or None,
            'requires_artifact': True,
            'repair_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            'recovery_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            'repair_action_reason': 'final composed page must reference every required generated image artifact',
            'content_payload': content_payload,
            'content_payload_source': 'closure_composed_page_image_representation',
            'stage_direction': 'materialize_requested_text_artifact',
            'generated_image_count': len(generated_paths),
            'represented_image_count': len(represented_paths),
            'missing_image_count': len(missing_paths),
            'generated_image_paths': generated_paths,
            'missing_image_paths': missing_paths,
            'represented_image_paths': sorted(represented_paths),
            'text_artifact_extension': target_extension,
            'text_artifact_source_name': target_name,
            'text_artifact_source': 'closure_composed_page_image_representation',
            'text_artifact_target_path': target_path,
            'artifact_request': {
                'extension': target_extension,
                'source_name': target_name,
                'source': 'closure_composed_page_image_representation',
                'target_path': target_path,
            },
        }

    @staticmethod
    def _html_attribute_escape(value: str) -> str:
        return (
            str(value or '')
            .replace('&', '&amp;')
            .replace('"', '&quot;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )

    @staticmethod
    def _html_fragment_has_balanced_common_containers(fragment: str) -> bool:
        text = str(fragment or '')
        for tag_name in ('section', 'article', 'div', 'figure', 'ul', 'ol', 'li'):
            open_count = len(
                re.findall(
                    rf'(?is)<\s*{tag_name}\b(?![^>]*?/>)',
                    text,
                )
            )
            close_count = len(re.findall(rf'(?is)</\s*{tag_name}\s*>', text))
            if open_count != close_count:
                return False
        return True

    @staticmethod
    def _html_image_container_is_suspect_repair_target(
        *,
        attrs: str,
        body: str,
        image_count: int,
        missing_link_count: int,
    ) -> bool:
        haystack = f'{attrs}\n{body}'.lower()
        if not LateFillRuntimeOwner._html_fragment_has_balanced_common_containers(body):
            return True
        if (
            image_count <= 1
            and missing_link_count > 1
            and re.search(
                r'\b(?:template|repeatable|populate|populated|placeholder|sample|additional|remaining|manifest)\b',
                haystack,
            )
        ):
            return True
        if (
            image_count <= 1
            and missing_link_count > 1
            and re.search(r'\b(?:gallery-card|card|tile|post|item|image-wrapper)\b', haystack)
            and re.search(
                r'\b(?:username|user-tag|handle|caption|influencer|card-info|'
                r'card-overlay|post-meta|overlay|gallery-card)\b',
                haystack,
            )
        ):
            return True
        if (
            missing_link_count > max(1, image_count)
            and re.search(
                r'\b(?:for brevity|repeated for all|repeat(?:ed)? for all|following the same pattern|'
                r'template below|structural logic|will be injected|assets? will be injected|'
                r'additional|remaining|populate|up to \d+|all \d+)\b',
                haystack,
            )
            and re.search(
                r'\b(?:username|user-tag|handle|caption|influencer|gallery-card|card-info|'
                r'card-overlay|post-meta|overlay)\b',
                haystack,
            )
        ):
            return True
        return False

    @staticmethod
    def _html_has_unfinished_generated_image_template(
        content: str,
        *,
        missing_link_count: int,
    ) -> bool:
        text = str(content or '')
        if missing_link_count <= 1 or not text.strip():
            return False
        haystack = text.lower()
        if not re.search(
            r'\b(?:template|repeatable|repeat|populate|populated|placeholder|sample|additional|remaining|manifest|asset block)\b',
            haystack,
        ):
            return False
        image_count = len(re.findall(r'(?is)<\s*img\b', text))
        if image_count <= 1:
            return True
        if missing_link_count > image_count and re.search(
            r'\b(?:repeat|populate|sample|additional|remaining|up to|all \d+|assets? will be rendered)\b',
            haystack,
        ):
            return True
        return False

    @staticmethod
    def _extract_social_manifest_card_metadata(
        content: Any,
        *,
        expected_count: int = 0,
    ) -> list[dict[str, str]]:
        text = str(content or '')
        if not text.strip():
            return []
        records: list[dict[str, str]] = []
        for raw_line in text.splitlines():
            line = re.sub(
                r'^\s*(?:[-*#>\u2022]+\s*)?(?:\d{1,3}|[ivx]+)[.)]\s+',
                '',
                str(raw_line or '').strip(),
                flags=re.IGNORECASE,
            ).strip()
            if not line:
                continue
            match = re.match(
                r'(?is)^(?:\*\*+|__+|`+|\*+)?\s*'
                r'(?P<handle>@[A-Za-z0-9][\w.-]{1,100})'
                r'\s*(?:\*\*+|__+|`+|\*+)?\s*:\s*(?P<body>.+)$',
                line,
            )
            if not match:
                continue
            handle = str(match.group('handle') or '').strip()
            prompt = _strip_social_manifest_image_prompt_metadata(line)
            if not prompt:
                prompt = _strip_social_manifest_image_prompt_metadata(match.group('body'))
            prompt = re.sub(r'\s+', ' ', prompt).strip(' ,;')
            if not handle or not prompt:
                continue
            caption = re.split(r'[.;]', prompt, maxsplit=1)[0].strip(' ,;')
            if len(caption) > 120:
                caption = f'{caption[:117].rstrip()}...'
            records.append(
                {
                    'handle': handle,
                    'caption': caption or prompt,
                    'prompt': prompt,
                    'alt': f'{handle} - {prompt}',
                }
            )
            if expected_count > 0 and len(records) >= expected_count:
                break
        if expected_count > 0 and len(records) < expected_count:
            return []
        return records

    @staticmethod
    def _social_manifest_card_metadata_from_payload(
        payload: Mapping[str, Any],
        *,
        expected_count: int = 0,
    ) -> list[dict[str, str]]:
        late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
        for source in (
            late_fill.get('content_payload'),
            late_fill.get('phase_summary'),
            payload.get('content_payload'),
            payload.get('output_text'),
        ):
            records = LateFillRuntimeOwner._extract_social_manifest_card_metadata(
                source,
                expected_count=expected_count,
            )
            if records:
                return records
        return []

    @staticmethod
    def _matching_html_element_end(text: str, *, tag: str, open_end: int) -> tuple[int, int]:
        tag_name = re.escape(str(tag or '').strip().lower())
        if not tag_name:
            return -1, -1
        token_re = re.compile(rf'(?is)<\s*(?P<close>/)?\s*{tag_name}\b[^>]*>')
        depth = 1
        for match in token_re.finditer(text, open_end):
            token = match.group(0)
            if match.group('close'):
                depth -= 1
                if depth == 0:
                    return match.start(), match.end()
                continue
            if not re.search(r'/\s*>$', token):
                depth += 1
        return -1, -1

    @staticmethod
    def _manifest_gallery_card_html(*, link: str, metadata: Mapping[str, str]) -> str:
        escaped_link = LateFillRuntimeOwner._html_attribute_escape(link)
        handle = LateFillRuntimeOwner._html_attribute_escape(str(metadata.get('handle') or '@generated'))
        caption = LateFillRuntimeOwner._html_attribute_escape(
            str(metadata.get('caption') or metadata.get('prompt') or 'Generated image')
        )
        alt = LateFillRuntimeOwner._html_attribute_escape(str(metadata.get('alt') or handle))
        return (
            '                <div class="feed-card" data-ollmo-repair="manifest-backed-gallery-expansion">\n'
            '                    <div class="card-image-wrapper">\n'
            f'                        <img src="{escaped_link}" alt="{alt}" loading="lazy">\n'
            '                        <div class="card-overlay"></div>\n'
            '                    </div>\n'
            '                    <div class="card-content">\n'
            f'                        <span class="username">{handle}</span>\n'
            f'                        <p class="caption">{caption}</p>\n'
            '                    </div>\n'
            '                </div>'
        )

    @staticmethod
    def _expand_manifest_backed_unfinished_gallery(
        content: str,
        *,
        target_path: str,
        generated_paths: list[str],
        metadata_records: list[dict[str, str]],
    ) -> tuple[str, bool]:
        text = str(content or '')
        paths = [str(path or '').strip() for path in generated_paths if str(path or '').strip()]
        if not text or not paths or len(metadata_records) < len(paths):
            return text, False
        if not LateFillRuntimeOwner._html_has_unfinished_generated_image_template(
            text,
            missing_link_count=max(2, len(paths) - 1),
        ):
            return text, False
        best: Optional[tuple[int, int, int, int]] = None
        container_re = re.compile(r'(?is)<(?P<tag>div|section|ul|ol)\b(?P<attrs>[^>]*)>')
        for match in container_re.finditer(text):
            tag = str(match.group('tag') or '').lower()
            attrs = str(match.group('attrs') or '')
            haystack = attrs.lower()
            if not re.search(r'\b(?:image-grid|feed-grid|gallery|grid|posts?|trending)\b', haystack):
                continue
            close_start, _close_end = LateFillRuntimeOwner._matching_html_element_end(
                text,
                tag=tag,
                open_end=match.end(),
            )
            if close_start < 0:
                continue
            body = text[match.end():close_start]
            body_haystack = body.lower()
            if '<img' not in body_haystack:
                continue
            if not re.search(
                r'\b(?:repeat(?:ed)? for all|following the same pattern|for brevity|template|all \d+|assets? will be)',
                body_haystack,
            ):
                continue
            score = len(re.findall(r'(?is)<\s*img\b', body))
            if re.search(r'\b(?:image-grid|feed-grid)\b', haystack):
                score += 20
            if re.search(r'\b(?:gallery|trending|feed)\b', haystack):
                score += 10
            candidate = (score, match.start(), match.end(), close_start)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return text, False
        _score, _open_start, open_end, close_start = best
        cards = [
            LateFillRuntimeOwner._manifest_gallery_card_html(
                link=LateFillRuntimeOwner._relative_artifact_link(
                    from_path=target_path,
                    to_path=path,
                ),
                metadata=metadata_records[index],
            )
            for index, path in enumerate(paths)
        ]
        replacement_body = '\n'.join(cards)
        updated = f'{text[:open_end].rstrip()}\n{replacement_body}\n{text[close_start:]}'
        return updated, updated != text

    @staticmethod
    def _replace_first_link_token(content: str, tokens: list[str], replacement: str) -> tuple[str, str]:
        text = str(content or '')
        for token in sorted({str(item or '').strip() for item in tokens if str(item or '').strip()}, key=len, reverse=True):
            if token in text:
                return text.replace(token, replacement, 1), token
        return text, ''

    @staticmethod
    def _insert_html_before_terminal_anchor(content: str, insertion: str) -> str:
        text = str(content or '')
        if not insertion:
            return text
        for pattern in (r'</main\s*>', r'</body\s*>', r'</html\s*>'):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return f'{text[:match.start()]}{insertion}\n{text[match.start():]}'
        return f'{text.rstrip()}\n{insertion}\n'

    @staticmethod
    def _generated_image_repair_tags_for_container(links: list[str], *, container_body: str = '') -> list[str]:
        body = str(container_body or '')
        use_card_wrapper = bool(re.search(r'(?is)\bclass\s*=\s*[\'"][^\'"]*\b(?:card|tile|post|item)\b', body))
        use_list_item = bool(re.search(r'(?is)<\s*li\b', body))
        tags: list[str] = []
        for index, link in enumerate(links, start=1):
            escaped_link = LateFillRuntimeOwner._html_attribute_escape(link)
            image_tag = f'<img src="{escaped_link}" alt="Generated image {index}">'
            if use_list_item:
                tags.append(f'            <li data-ollmo-repair="composed-page-image-representation">{image_tag}</li>')
            elif use_card_wrapper:
                tags.append(
                    '            '
                    '<div class="card" data-ollmo-repair="composed-page-image-representation">'
                    f'{image_tag}</div>'
                )
            else:
                tags.append(f'            {image_tag}')
        return tags

    @staticmethod
    def _insert_html_into_existing_image_container(
        content: str,
        links: list[str],
    ) -> tuple[str, bool, str]:
        text = str(content or '')
        clean_links = [str(link or '').strip() for link in links if str(link or '').strip()]
        if not text or not clean_links:
            return text, False, ''
        best: Optional[tuple[int, int, int, int, str, str]] = None
        tag_patterns = ('section', 'ul', 'ol', 'div', 'main')
        for tag_name in tag_patterns:
            container_re = re.compile(
                rf'(?is)<(?P<tag>{tag_name})\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>'
            )
            for match in container_re.finditer(text):
                tag = str(match.group('tag') or '').lower()
                attrs = str(match.group('attrs') or '')
                body = str(match.group('body') or '')
                haystack = f'{attrs}\n{body}'.lower()
                image_count = len(re.findall(r'(?is)<\s*img\b', body))
                if image_count <= 0:
                    continue
                if LateFillRuntimeOwner._html_image_container_is_suspect_repair_target(
                    attrs=attrs,
                    body=body,
                    image_count=image_count,
                    missing_link_count=len(clean_links),
                ):
                    continue
                score = image_count
                if re.search(r'\b(?:gallery|feed|grid|cards?|posts?|trending|images?|media|portfolio)\b', haystack):
                    score += 10
                if re.search(r'\b(?:sample|placeholder|continue|continues|additional|remaining|populate|up to)\b', haystack):
                    score += 6
                if re.search(r'(?is)\bclass\s*=\s*[\'"][^\'"]*\b(?:card|tile|post|item)\b', body):
                    score += 3
                if tag in {'section', 'ul', 'ol'}:
                    score += 2
                if score < 10:
                    continue
                close_start = match.end() - len(f'</{tag}>')
                tag_priority = 0 if tag == 'main' else 1
                candidate = (tag_priority, score, image_count, close_start, tag, body)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
        if best is None:
            return text, False, ''
        _tag_priority, _score, _image_count, close_start, tag, body = best
        insertion = '\n'.join(LateFillRuntimeOwner._generated_image_repair_tags_for_container(clean_links, container_body=body))
        if not insertion:
            return text, False, ''
        updated = f'{text[:close_start].rstrip()}\n{insertion}\n{text[close_start:]}'
        return updated, True, tag

    def _repair_terminal_composed_page_image_representation(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        check = self._terminal_composed_page_image_representation_open_check(payload)
        if not check:
            return payload
        target_path = str(check.get('text_artifact_target_path') or '').strip()
        if not target_path:
            return payload
        missing_paths = [
            str(path or '').strip()
            for path in (check.get('missing_image_paths') or [])
            if str(path or '').strip()
        ]
        if not missing_paths:
            return payload
        generated_paths = [
            str(path or '').strip()
            for path in (check.get('generated_image_paths') or [])
            if str(path or '').strip()
        ]
        represented_paths = [
            str(path or '').strip()
            for path in (check.get('represented_image_paths') or [])
            if str(path or '').strip()
        ]
        try:
            target = Path(target_path).expanduser()
            if not target.is_file() or target.stat().st_size > 512_000:
                return payload
            original = target.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return payload

        records = self._collect_link_rebind_artifact_records(payload)
        text_records = self._unique_artifact_records_by_path(
            [
                record
                for record in records
                if str(record.get('type') or record.get('kind') or '').strip().lower() == 'text'
                and self._artifact_record_extension(record) in _LINK_REBIND_TEXT_EXTENSIONS
                and self._artifact_record_path(record)
            ]
        )
        content_by_path: dict[str, str] = {}
        for record in text_records:
            source_path = self._artifact_record_path(record)
            content = self._terminal_materialization_text_record_content(record)
            if source_path and content:
                content_by_path[source_path] = content

        target_record = next(
            (
                record
                for record in text_records
                if self._text_artifact_path_matches_target(
                    self._artifact_record_path(record),
                    target_path,
                )
            ),
            {},
        )
        dependency_records = self._terminal_linked_artifact_dependency_records(records, text_records)
        consumer_dependency_records, _dependency_ids, _selection_policy = (
            self._link_rebind_asset_records_for_consumer(
                payload,
                target_record,
                dependency_records,
            )
        )
        unresolved_detection_records = consumer_dependency_records
        if (
            _selection_policy == 'consumer_declared_media_dependency_missing'
            or (_dependency_ids and not unresolved_detection_records)
        ):
            unresolved_detection_records = dependency_records
        if self._terminal_content_has_unresolved_link_placeholder(
            original,
            unresolved_detection_records,
            target_path=target_path,
        ):
            return payload

        def represented_outside_target(image_path: str) -> bool:
            for source_path, content in content_by_path.items():
                if self._text_artifact_path_matches_target(source_path, target_path):
                    continue
                if self._terminal_exact_content_has_artifact_link(
                    content,
                    source_path=source_path,
                    target_path=image_path,
                ):
                    return True
            return False

        unsafe_duplicate_paths = [
            represented_path
            for represented_path in represented_paths
            if any(
                original.count(token) > 1
                for token in self._terminal_exact_link_tokens_for_artifact(
                    source_path=target_path,
                    target_path=represented_path,
                )
            )
            and not represented_outside_target(represented_path)
        ]
        if unsafe_duplicate_paths:
            return payload

        updated_content = original
        repair_entries: list[dict[str, Any]] = []
        missing_for_append: list[str] = []
        for missing_path in missing_paths:
            if self._terminal_exact_content_has_artifact_link(
                updated_content,
                source_path=target_path,
                target_path=missing_path,
            ):
                continue
            missing_link = self._relative_artifact_link(from_path=target_path, to_path=missing_path)
            replaced = False
            for represented_path in represented_paths:
                represented_link = self._relative_artifact_link(from_path=target_path, to_path=represented_path)
                target_tokens = [
                    token
                    for token in (represented_link, represented_path)
                    if token and token in updated_content
                ]
                if not target_tokens:
                    continue
                if not represented_outside_target(represented_path):
                    continue
                next_content, replaced_token = self._replace_first_link_token(
                    updated_content,
                    target_tokens,
                    missing_link,
                )
                if not replaced_token or next_content == updated_content:
                    continue
                updated_content = next_content
                repair_entries.append(
                    {
                        'kind': 'ollmo.composed_page_image_representation_repair',
                        'operation': 'replace_duplicate_image_link',
                        'target_path': target_path,
                        'from_path': represented_path,
                        'from_token': replaced_token,
                        'to_path': missing_path,
                        'to_token': missing_link,
                        'status': 'applied',
                        'source': 'terminal_composed_page_image_representation_repair',
                    }
                )
                replaced = True
                break
            if not replaced:
                missing_for_append.append(missing_path)

        if missing_for_append:
            manifest_expanded = False
            all_generated_paths = generated_paths or [*represented_paths, *missing_for_append]
            metadata_records = self._social_manifest_card_metadata_from_payload(
                payload,
                expected_count=len(all_generated_paths),
            )
            if metadata_records and self._html_has_unfinished_generated_image_template(
                updated_content,
                missing_link_count=len(missing_for_append),
            ):
                next_content, manifest_expanded = self._expand_manifest_backed_unfinished_gallery(
                    updated_content,
                    target_path=target_path,
                    generated_paths=all_generated_paths,
                    metadata_records=metadata_records,
                )
                if manifest_expanded and next_content != updated_content:
                    updated_content = next_content
            if manifest_expanded:
                for missing_path in missing_for_append:
                    repair_entries.append(
                        {
                            'kind': 'ollmo.composed_page_image_representation_repair',
                            'operation': 'expand_manifest_gallery_card',
                            'target_path': target_path,
                            'to_path': missing_path,
                            'to_token': self._relative_artifact_link(
                                from_path=target_path,
                                to_path=missing_path,
                            ),
                            'placement': 'manifest_backed_template_expansion',
                            'container_tag': 'manifest_gallery',
                            'status': 'applied',
                            'source': 'terminal_composed_page_image_representation_repair',
                        }
                    )
                missing_for_append = []

        if missing_for_append:
            image_links: list[str] = []
            for index, missing_path in enumerate(missing_for_append, start=1):
                link = self._relative_artifact_link(from_path=target_path, to_path=missing_path)
                image_links.append(link)
                repair_entries.append(
                    {
                        'kind': 'ollmo.composed_page_image_representation_repair',
                        'operation': 'append_missing_image_link',
                        'target_path': target_path,
                        'to_path': missing_path,
                        'to_token': link,
                        'placement': 'pending',
                        'status': 'applied',
                        'source': 'terminal_composed_page_image_representation_repair',
                    }
                )
            next_content, inserted_in_container, container_tag = self._insert_html_into_existing_image_container(
                updated_content,
                image_links,
            )
            if inserted_in_container and next_content != updated_content:
                updated_content = next_content
                for entry in repair_entries[-len(image_links):]:
                    entry['placement'] = 'existing_image_container'
                    entry['container_tag'] = container_tag
            else:
                if self._html_has_unfinished_generated_image_template(
                    updated_content,
                    missing_link_count=len(image_links),
                ):
                    return payload
                image_tags = self._generated_image_repair_tags_for_container(image_links)
                insertion = '\n'.join(
                    [
                        '',
                        '    <section class="ollmo-generated-media" data-ollmo-repair="composed-page-image-representation">',
                        '        <div class="container">',
                        *image_tags,
                        '        </div>',
                        '    </section>',
                    ]
                )
                updated_content = self._insert_html_before_terminal_anchor(updated_content, insertion)
                for entry in repair_entries[-len(image_links):]:
                    entry['placement'] = 'detached_repair_section'

        if not repair_entries or updated_content == original:
            return payload
        try:
            target.write_text(updated_content, encoding='utf-8')
        except OSError as exc:
            logging.warning('Could not repair composed-page image representation in %s: %s', target_path, exc)
            return payload

        updated_payload = dict(payload or {})
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), Mapping)
            else {}
        )
        existing_repairs = [
            dict(item)
            for item in (late_fill.get('composed_page_image_representation_repairs') or [])
            if isinstance(item, Mapping)
        ]
        late_fill['composed_page_image_representation_repairs'] = [*existing_repairs, *repair_entries]
        late_fill['composed_page_image_representation_repair_status'] = 'applied'
        late_fill['composed_page_image_representation_repaired_paths'] = [
            entry.get('to_path')
            for entry in repair_entries
            if entry.get('to_path')
        ]
        updated_payload['late_fill'] = late_fill
        return updated_payload

    def _terminal_materialization_branch_has_canonical_evidence(
        self,
        check: Mapping[str, Any],
        payload: Mapping[str, Any],
        late_fill: Mapping[str, Any],
    ) -> bool:
        branch_id = str(check.get('branch_id') or check.get('phase_id') or '').strip()
        completed_branch = {}
        for branch in late_fill.get('completed_branches') or []:
            if not isinstance(branch, Mapping):
                continue
            candidate_id = str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
            if branch_id and candidate_id == branch_id:
                completed_branch = branch
                break
        output_type = str(check.get('output_type') or completed_branch.get('output_type') or '').strip().lower()
        capability = str(check.get('capability') or completed_branch.get('capability') or '').strip().lower()
        role = str(check.get('role') or completed_branch.get('role') or '').strip()
        if output_type == 'text' or role == 'text_artifact_output':
            candidates = [check]
            if completed_branch:
                candidates.append(completed_branch)
            return any(
                self._text_artifact_branch_has_canonical_evidence(candidate, payload)
                for candidate in candidates
                if isinstance(candidate, Mapping)
            )
        if output_type == 'image' or capability == self.capability_image_generation:
            records = self._collect_link_rebind_artifact_records(payload)
            for record in records:
                record_type = str(record.get('type') or record.get('kind') or '').strip().lower()
                if record_type != 'image' and self._artifact_record_extension(record) not in _LINK_REBIND_IMAGE_EXTENSIONS:
                    continue
                record_branch_id = str(record.get('branch_id') or record.get('phase_id') or '').strip()
                if branch_id and record_branch_id and record_branch_id != branch_id:
                    continue
                path = self._artifact_record_path(record)
                if path and Path(path).expanduser().is_file():
                    return True
        return False

    def _terminal_materialization_check_still_open(
        self,
        check: Mapping[str, Any],
        payload: Mapping[str, Any],
        late_fill: Mapping[str, Any],
        *,
        linked_artifact_contract_fulfilled: bool,
    ) -> bool:
        check_kind = str(check.get('check_kind') or '').strip()
        evidence = str(check.get('evidence') or '').strip()
        role = str(check.get('role') or '').strip()
        repair_action = str(check.get('repair_action') or check.get('recovery_action') or '').strip()
        if (
            check_kind == 'composed_page_image_representation'
            or evidence == 'generated_image_not_represented_in_composed_page'
        ):
            return True
        if (
            check_kind == 'hero_image_composition'
            or evidence == 'generated_image_not_represented_in_hero'
        ):
            return self._terminal_hero_image_composition_open_check(payload) is not None
        if check_kind == 'text_artifact_syntax_sanity' or evidence == 'text_artifact_syntax_issue':
            return True
        if (
            check_kind == 'structured_dependency_join'
            or evidence == 'structured_dependency_join_contract_unmet'
        ):
            return True
        if check_kind == 'html_css_selector_binding' or evidence == 'html_css_selector_drift':
            # This check comes from the just-refreshed Closure review.  A CSS
            # file merely existing is not evidence that it targets the saved
            # HTML vocabulary.  When the final bytes agree, the fresh review
            # emits no selector check at all; otherwise it must remain open.
            return True
        if check_kind == 'web_runtime_binding' or evidence in {
            'inline_javascript_binding_mismatch',
            'static_json_consumer_contract_mismatch',
        }:
            # These checks are recomputed from the current saved web files.
            # Artifact existence cannot satisfy a still-current binding defect.
            return True
        if check_kind in {'truth_guard', 'truth_guard_capability'}:
            return True
        if evidence in {
            'text_only_artifact_claim_guard',
            'runtime_capability_available_but_unmaterialized',
            'runtime_capability_unverified',
        }:
            return True
        if (
            check_kind == 'linked_artifact_binding'
            or role == 'linked_artifact_binding_review'
            or evidence == 'unresolved_linked_artifact_binding'
            or evidence == 'unresolved_local_dependency_link'
            or repair_action == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
        ):
            return not linked_artifact_contract_fulfilled

        branch_id = str(check.get('branch_id') or check.get('phase_id') or '').strip()
        completed_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in (late_fill.get('completed_branches') or [])
            if isinstance(item, Mapping) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        if branch_id and branch_id in completed_branch_ids:
            if self._terminal_materialization_branch_has_canonical_evidence(check, payload, late_fill):
                return False
        return True

    def _filter_terminal_materialization_open_checks(
        self,
        checks: list[dict[str, Any]],
        payload: Mapping[str, Any],
        late_fill: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not checks:
            return []
        linked_artifact_contract_fulfilled = self._terminal_linked_artifact_contract_is_fulfilled(payload)
        return [
            dict(check)
            for check in checks
            if self._terminal_materialization_check_still_open(
                check,
                payload,
                late_fill,
                linked_artifact_contract_fulfilled=linked_artifact_contract_fulfilled,
            )
        ]

    @staticmethod
    def _is_linked_artifact_binding_check(check: Mapping[str, Any]) -> bool:
        if str(check.get('check_kind') or '').strip() == 'structured_dependency_join':
            return False
        return (
            str(check.get('check_kind') or '').strip() == 'linked_artifact_binding'
            or str(check.get('role') or '').strip() == 'linked_artifact_binding_review'
            or str(check.get('evidence') or '').strip() == 'unresolved_linked_artifact_binding'
            or str(check.get('evidence') or '').strip() == 'unresolved_local_dependency_link'
            or str(check.get('repair_action') or check.get('recovery_action') or '').strip()
            == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
        )

    @staticmethod
    def _terminal_linked_artifact_binding_open_check() -> dict[str, Any]:
        return {
            'check_kind': 'linked_artifact_binding',
            'status': 'pending',
            'evidence': 'unresolved_linked_artifact_binding',
            'role': 'linked_artifact_binding_review',
            'requires_artifact': True,
            'repair_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            'recovery_action': RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            'reason': 'final linked artifact output does not resolve every saved dependency artifact',
        }

    @staticmethod
    def _compact_terminal_materialization_checks(
        checks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for check in checks[:8]:
            compacted.append(
                {
                    key: value
                    for key, value in {
                        'check_kind': check.get('check_kind'),
                        'status': check.get('status'),
                        'evidence': check.get('evidence'),
                        'reason': check.get('reason'),
                        'role': check.get('role'),
                        'branch_id': check.get('branch_id'),
                        'phase_id': check.get('phase_id'),
                        'capability': check.get('capability'),
                        'output_type': check.get('output_type'),
                        'repair_action': check.get('repair_action') or check.get('recovery_action'),
                        'text_artifact_extension': check.get('text_artifact_extension'),
                        'text_artifact_source_name': check.get('text_artifact_source_name'),
                        'text_artifact_target_path': check.get('text_artifact_target_path'),
                        'content_payload_source': check.get('content_payload_source'),
                        'dependency_source_path': check.get('dependency_source_path'),
                        'missing_dependency_url': check.get('missing_dependency_url'),
                        'missing_dependency_target_path': check.get('missing_dependency_target_path'),
                        'generated_image_count': check.get('generated_image_count'),
                        'represented_image_count': check.get('represented_image_count'),
                        'missing_image_count': check.get('missing_image_count'),
                        'expected_count': check.get('expected_count'),
                        'actual_count': check.get('actual_count'),
                        'expected_labels': (
                            check.get('expected_labels')[:16]
                            if isinstance(check.get('expected_labels'), list)
                            else check.get('expected_labels')
                        ),
                        'required_fields': (
                            check.get('required_fields')[:16]
                            if isinstance(check.get('required_fields'), list)
                            else check.get('required_fields')
                        ),
                        'issue_codes': (
                            check.get('issue_codes')[:16]
                            if isinstance(check.get('issue_codes'), list)
                            else check.get('issue_codes')
                        ),
                        'missing_image_paths': (
                            check.get('missing_image_paths')[:8]
                            if isinstance(check.get('missing_image_paths'), list)
                            else check.get('missing_image_paths')
                        ),
                    }.items()
                    if value not in (None, '', [], {})
                }
            )
        return compacted

    def _demote_terminal_materialization_branches_with_open_checks(
        self,
        late_fill: Mapping[str, Any],
        open_checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not open_checks:
            return dict(late_fill or {})
        updated = dict(late_fill or {})
        blocking_branch_ids = {
            str(check.get('branch_id') or check.get('phase_id') or '').strip()
            for check in open_checks
            if str(check.get('branch_id') or check.get('phase_id') or '').strip()
        }
        repair_evidence_tokens = {
            'terminal_linked_artifact_rebind_applied',
            'terminal_inline_text_artifact_materialized',
            'canonical_text_artifact_evidence',
        }
        repair_roles = {
            'linked_artifact_binding_review',
            'text_artifact_syntax_repair',
            'text_artifact_output',
        }
        completed_records = [
            dict(item)
            for item in (updated.get('completed_branches') or [])
            if isinstance(item, Mapping)
        ]
        pending_records = [
            dict(item)
            for item in (updated.get('pending_branches') or [])
            if isinstance(item, Mapping)
        ]
        pending_ids = {
            self.branch_id(item)
            for item in pending_records
            if self.branch_id(item)
        }
        retained_completed: list[dict[str, Any]] = []
        demoted_records: list[dict[str, Any]] = []
        compact_checks = self._compact_terminal_materialization_checks(open_checks)

        def check_matches_branch(check: Mapping[str, Any], branch: Mapping[str, Any]) -> bool:
            branch_id = self.branch_id(branch)
            check_id = str(check.get('branch_id') or check.get('phase_id') or '').strip()
            if branch_id and check_id and branch_id == check_id:
                return True
            branch_request = branch.get('artifact_request') if isinstance(branch.get('artifact_request'), Mapping) else {}
            check_request = check.get('artifact_request') if isinstance(check.get('artifact_request'), Mapping) else {}
            branch_target = str(
                branch.get('text_artifact_target_path')
                or branch_request.get('target_path')
                or ''
            ).strip()
            check_target = str(
                check.get('text_artifact_target_path')
                or check_request.get('target_path')
                or ''
            ).strip()
            if branch_target and check_target and branch_target == check_target:
                return True
            branch_extension = str(
                branch.get('text_artifact_extension')
                or branch_request.get('extension')
                or ''
            ).strip().lower().lstrip('.')
            check_extension = str(
                check.get('text_artifact_extension')
                or check_request.get('extension')
                or ''
            ).strip().lower().lstrip('.')
            branch_source = str(
                branch.get('text_artifact_source_name')
                or branch_request.get('source_name')
                or ''
            ).strip().lower()
            check_source = str(
                check.get('text_artifact_source_name')
                or check_request.get('source_name')
                or ''
            ).strip().lower()
            return bool(branch_extension and branch_source and branch_extension == check_extension and branch_source == check_source)

        def repair_fields_for_branch(branch: Mapping[str, Any]) -> dict[str, Any]:
            matching_checks = [
                check
                for check in open_checks
                if isinstance(check, Mapping) and check_matches_branch(check, branch)
            ]
            if not matching_checks:
                return {}
            preferred = next(
                (
                    check for check in matching_checks
                    if str(check.get('content_payload') or '').strip()
                    and str(check.get('text_artifact_target_path') or '').strip()
                ),
                matching_checks[0],
            )
            fields: dict[str, Any] = {}
            for key in (
                'repair_action',
                'recovery_action',
                'suggested_action',
                'repair_action_reason',
                'content_payload',
                'content_payload_source',
                'stage_direction',
                'requires_artifact',
                'text_artifact_extension',
                'text_artifact_source_name',
                'text_artifact_source',
                'text_artifact_target_path',
                'artifact_request',
                'review_criteria',
                'generated_image_count',
                'represented_image_count',
                'missing_image_count',
                'missing_image_paths',
                'represented_image_paths',
            ):
                value = preferred.get(key)
                if value not in (None, '', [], {}):
                    fields[key] = value
            if fields.get('repair_action') == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE:
                fields.setdefault('repair_execution_policy', 'schedule_late_fill_branch')
                fields.setdefault('auto_execute', True)
                fields.setdefault('repair_work_available', True)
                fields.setdefault('needs_external_input', False)
                fields.setdefault('materialization_blocked', False)
            return fields

        for branch in completed_records:
            branch_id = self.branch_id(branch)
            evidence = str(branch.get('evidence') or '').strip()
            role = str(branch.get('role') or '').strip()
            should_demote = bool(
                branch_id
                and (
                    branch_id in blocking_branch_ids
                    or evidence in repair_evidence_tokens
                    or role in repair_roles
                )
                and str(branch.get('status') or '').strip().lower() == 'fulfilled'
            )
            if not should_demote:
                retained_completed.append(branch)
                continue
            demoted = dict(branch)
            demoted.update(repair_fields_for_branch(branch))
            demoted['status'] = 'repair_needed'
            demoted['evidence'] = 'terminal_materialization_contract_unmet'
            demoted['repair_action'] = (
                demoted.get('repair_action')
                or demoted.get('recovery_action')
                or RECOVERY_ACTION_RETRY_SAME_BRANCH
            )
            demoted['recovery_action'] = demoted.get('repair_action')
            demoted['materialization_contract_unmet'] = True
            demoted['materialization_contract_open_checks'] = compact_checks
            demoted_records.append(demoted)
            if branch_id and branch_id not in pending_ids:
                pending_records.append(demoted)
                pending_ids.add(branch_id)
        if not demoted_records:
            return updated
        updated['completed_branches'] = retained_completed
        updated['completed_branch_count'] = len(retained_completed)
        updated['pending_branches'] = pending_records
        updated['pending_branch_count'] = len(pending_records)
        previous_demoted = [
            dict(item)
            for item in (updated.get('materialization_contract_demoted_branches') or [])
            if isinstance(item, Mapping)
        ]
        updated['materialization_contract_demoted_branches'] = previous_demoted + demoted_records
        return updated

    def _review_terminal_graph_rebase_if_available(
        self,
        payload: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        route_payload: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not callable(self.review_terminal_graph_rebase):
            return payload
        try:
            reviewed = self.review_terminal_graph_rebase(
                payload,
                request_payload=request_payload,
                route_payload=route_payload,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning('Could not review terminal graph rebase shadow candidate: %s', exc)
            return payload
        return reviewed if isinstance(reviewed, dict) else payload

    @staticmethod
    def _reconcile_terminal_satisfied_repair_loop(
        late_fill: Mapping[str, Any],
    ) -> dict[str, Any]:
        updated = dict(late_fill or {})
        repair_loop = (
            dict(updated.get('repair_loop') or {})
            if isinstance(updated.get('repair_loop'), Mapping)
            else {}
        )
        if not repair_loop or not terminal_repair_loop_is_fully_satisfied(updated, repair_loop):
            return updated

        promoted_contracts = [
            dict(item)
            for item in (repair_loop.get('promoted_contracts') or [])
            if isinstance(item, Mapping)
        ]
        resolved_contracts = [
            {
                key: value
                for key, value in {
                    'contract_id': contract.get('contract_id'),
                    'branch_id': contract.get('branch_id'),
                    'phase_id': contract.get('phase_id'),
                    'obligation_id': contract.get('obligation_id'),
                    'task_id': contract.get('task_id'),
                    'artifact_request': contract.get('artifact_request'),
                }.items()
                if value not in (None, '', [], {})
            }
            for contract in promoted_contracts
        ]
        repair_loop.update(
            {
                'status': 'completed',
                'auto_execute': False,
                'repair_work_available': False,
                'repair_work_available_count': 0,
                'executable_contract_count': 0,
                'blocked_contract_count': 0,
                'materialization_blocked_contract_count': 0,
                'needs_external_input_count': 0,
                'next_actions': [],
                'requires_promotion': False,
                'resolved_contract_count': len(resolved_contracts),
                'resolved_contracts': resolved_contracts,
                'resolution': {
                    'status': 'completed',
                    'authority': 'terminal_exact_repair_contract_evidence',
                },
            }
        )
        updated['repair_loop'] = repair_loop
        action_values = [updated.get('repair_action')]
        if isinstance(updated.get('repair_actions'), list):
            action_values.extend(updated.get('repair_actions') or [])
        resolved_actions = [
            str(value or '').strip()
            for value in action_values
            if str(value or '').strip()
        ]
        if resolved_actions:
            updated['resolved_repair_actions'] = list(dict.fromkeys(resolved_actions))
        updated.pop('repair_action', None)
        updated.pop('repair_actions', None)

        feedback = (
            dict(updated.get('ghost_repair_feedback') or {})
            if isinstance(updated.get('ghost_repair_feedback'), Mapping)
            else {}
        )
        if feedback:
            feedback['status'] = 'resolved'
            feedback['repair_loop'] = dict(repair_loop)
            feedback['resolution_status'] = 'completed'
            updated['ghost_repair_feedback'] = feedback
        return updated

    def finalize_terminal_materialization_contract(
        self,
        payload: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        route_payload: Optional[dict[str, Any]],
        artifact_gap: Optional[dict[str, Any]],
        terminal_status: str,
    ) -> tuple[dict[str, Any], str]:
        effective_status = str(terminal_status or '').strip().lower() or 'completed'
        updated_payload, inline_materialization_open_checks = (
            self._materialize_inline_text_artifacts_for_terminal_closure(payload)
        )
        updated_payload = self.rebind_terminal_linked_artifacts(updated_payload)
        updated_payload = self._repair_terminal_text_artifact_syntax(updated_payload)
        updated_payload = self._repair_terminal_composed_page_image_representation(updated_payload)
        updated_payload = self._repair_terminal_hero_image_composition(
            updated_payload,
            request_payload=request_payload,
        )
        updated_payload = self._refresh_terminal_text_artifacts_from_saved_files(updated_payload)
        # A terminal repair may create or replace a named dependency after the
        # initial rebind. Re-run the idempotent resolver over final saved truth
        # before Closure evaluates local links and consumer contracts.
        updated_payload = self.rebind_terminal_linked_artifacts(updated_payload)
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), Mapping)
            else {}
        )
        precheck_late_fill = self._reconcile_satisfied_failed_text_artifact_branches(updated_payload, late_fill)
        if precheck_late_fill != late_fill:
            updated_payload = self.attach_late_fill_state(updated_payload, precheck_late_fill)
        updated_payload = self.refresh_terminal_graph_closure_review(
            updated_payload,
            request_payload=request_payload,
            route_payload=route_payload,
            artifact_gap=artifact_gap,
        )
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), Mapping)
            else {}
        )
        post_refresh_late_fill = self._reconcile_satisfied_failed_text_artifact_branches(
            updated_payload,
            late_fill,
        )
        if post_refresh_late_fill != late_fill:
            updated_payload = self.attach_late_fill_state(updated_payload, post_refresh_late_fill)
            late_fill = post_refresh_late_fill
        open_checks = self._filter_terminal_materialization_open_checks(
            [
                *self._terminal_materialization_contract_open_checks(updated_payload),
                *inline_materialization_open_checks,
                *self._terminal_unresolved_local_dependency_link_open_checks(updated_payload),
                *self._terminal_web_binding_contract_open_checks(updated_payload),
            ],
            updated_payload,
            late_fill,
        )
        composed_image_check = self._terminal_composed_page_image_representation_open_check(updated_payload)
        if (
            composed_image_check
            and not any(
                str(check.get('check_kind') or '').strip() == 'composed_page_image_representation'
                for check in open_checks
                if isinstance(check, Mapping)
            )
        ):
            open_checks = self._filter_terminal_materialization_open_checks(
                [*open_checks, composed_image_check],
                updated_payload,
                late_fill,
            )
        hero_image_check = self._terminal_hero_image_composition_open_check(
            updated_payload,
            request_payload=request_payload,
        )
        if (
            hero_image_check
            and not any(
                str(check.get('check_kind') or '').strip() == 'hero_image_composition'
                for check in open_checks
                if isinstance(check, Mapping)
            )
        ):
            open_checks = self._filter_terminal_materialization_open_checks(
                [*open_checks, hero_image_check],
                updated_payload,
                late_fill,
            )
        if (
            not open_checks
            and self._terminal_linked_artifact_contract_required(updated_payload)
            and not self._terminal_linked_artifact_contract_is_fulfilled(updated_payload)
        ):
            open_checks = self._filter_terminal_materialization_open_checks(
                [self._terminal_linked_artifact_binding_open_check()],
                updated_payload,
                late_fill,
            )
        elif (
            open_checks
            and not any(self._is_linked_artifact_binding_check(check) for check in open_checks)
            and self._terminal_linked_artifact_contract_required(updated_payload)
            and not self._terminal_linked_artifact_contract_is_fulfilled(updated_payload)
        ):
            open_checks = self._filter_terminal_materialization_open_checks(
                [*open_checks, self._terminal_linked_artifact_binding_open_check()],
                updated_payload,
                late_fill,
            )
        branch_state_open_checks: list[dict[str, Any]] = []
        for branch_key, branch_status, evidence in (
            ('failed_branches', 'failed', 'late_fill_failed_branch'),
            ('pending_branches', 'pending', 'late_fill_pending_branch'),
            ('active_branches', 'active', 'late_fill_active_branch'),
        ):
            for branch in self.normalize_late_fill_branches(late_fill.get(branch_key)):
                if branch.get('non_blocking_after_final_contract_fulfilled') is True:
                    continue
                branch_id = self.branch_id(branch)
                if str(branch_id or '').startswith('branch-repair-'):
                    continue
                branch_state_open_checks.append(
                    {
                        'check_kind': 'late_fill_branch_state',
                        'status': branch_status,
                        'evidence': evidence,
                        'branch_id': branch_id or None,
                        'phase_id': str(branch.get('phase_id') or branch_id or '').strip() or None,
                        'capability': self.branch_capability(branch),
                        'output_type': str(branch.get('output_type') or '').strip() or None,
                        'reason': 'late-fill branch has not reached fulfilled, waived, or superseded state',
                    }
                )
        if branch_state_open_checks:
            open_checks = self._filter_terminal_materialization_open_checks(
                [*open_checks, *branch_state_open_checks],
                updated_payload,
                late_fill,
            )
        if open_checks:
            late_fill = self._demote_terminal_materialization_branches_with_open_checks(
                late_fill,
                open_checks,
            )
            if effective_status in {'completed', 'skipped'}:
                effective_status = 'partial_failed'
            late_fill['status'] = effective_status
            late_fill['final_materialization_contract_status'] = 'unmet'
            late_fill['materialization_contract_unmet'] = True
            late_fill['final_materialization_contract_reason'] = (
                'terminal structured dependency join violates its explicit output contract'
                if any(
                    str(check.get('check_kind') or '').strip() == 'structured_dependency_join'
                    for check in open_checks
                    if isinstance(check, Mapping)
                )
                else 'artifact obligation exists but final materialization contract is unmet'
            )
            late_fill['materialization_contract_open_check_count'] = len(open_checks)
            late_fill['materialization_contract_open_checks'] = (
                self._compact_terminal_materialization_checks(open_checks)
            )
            if str(late_fill.get('skip_reason') or '').strip().lower() == 'already_fulfilled':
                late_fill['skip_reason'] = 'final_materialization_contract_unmet'
                late_fill['skip_kind'] = 'materialization_contract_unmet'
            updated_payload = self.attach_late_fill_state(updated_payload, late_fill)
            updated_payload = self.refresh_runtime_graph_repair_evidence(updated_payload)
            updated_payload = self._review_terminal_graph_rebase_if_available(
                updated_payload,
                request_payload=request_payload,
                route_payload=route_payload,
            )
            return updated_payload, effective_status

        late_fill['final_materialization_contract_status'] = 'fulfilled'
        late_fill['materialization_contract_unmet'] = False
        for key in (
            'materialization_contract_open_check_count',
            'materialization_contract_open_checks',
            'final_materialization_contract_reason',
        ):
            late_fill.pop(key, None)
        has_blocking_late_fill_state = any(
            isinstance(late_fill.get(key), list) and bool(late_fill.get(key))
            for key in ('failed_branches', 'pending_branches', 'active_branches', 'recovery_candidates')
        )
        if not has_blocking_late_fill_state and effective_status in {
            'failed',
            'partial_failed',
            'blocked',
            'repair_needed',
        }:
            effective_status = 'completed'
        if (
            str(late_fill.get('status') or '').strip().lower() == 'completed'
            and effective_status in {'failed', 'partial_failed', 'blocked', 'repair_needed'}
        ):
            effective_status = 'completed'
        if (
            not has_blocking_late_fill_state
            and str(late_fill.get('status') or '').strip().lower()
            in {'failed', 'partial_failed', 'blocked', 'repair_needed'}
        ):
            late_fill['status'] = 'completed'
        late_fill = self._reconcile_terminal_satisfied_repair_loop(late_fill)
        updated_payload = self.attach_late_fill_state(updated_payload, late_fill)
        # The first Closure pass intentionally runs before terminal contract
        # reconciliation so it can expose repairable checks. Once those checks
        # are proven closed and the terminal state is attached, refresh again;
        # otherwise a stale pre-repair review can keep canonical lifecycle open.
        updated_payload = self.refresh_terminal_graph_closure_review(
            updated_payload,
            request_payload=request_payload,
            route_payload=route_payload,
            artifact_gap=artifact_gap,
        )
        updated_payload = self.refresh_runtime_graph_repair_evidence(updated_payload)
        updated_payload = self._review_terminal_graph_rebase_if_available(
            updated_payload,
            request_payload=request_payload,
            route_payload=route_payload,
        )
        return updated_payload, effective_status

    @staticmethod
    def _post_artifact_follow_up_instruction(
        original_prompt: str,
        evidence: str = '',
        *,
        structured_output_contract: Optional[Mapping[str, Any]] = None,
    ) -> str:
        prompt_text = str(original_prompt or '').strip()
        tail = ''
        markers = list(re.finditer(
            r'\b(?:then|afterwards|danach|anschliessend|anschließend)\b',
            prompt_text,
            flags=re.IGNORECASE,
        ))
        if markers:
            tail = prompt_text[markers[-1].end():]
        if not tail:
            post_match = re.search(
                r'\b(?:compare|confirm|caption|describe|summari[sz]e|write|explain|'
                r'vergleiche|bestaetige|bestätige|beschreibe|schreibe|fasse)\b.+$',
                prompt_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if post_match:
                tail = post_match.group(0)
        tail = re.sub(r'\s+', ' ', tail).strip(' ,.;:-')
        if tail:
            task = tail[:1].upper() + tail[1:]
        else:
            task = 'Complete only the requested post-artifact text follow-up.'
        contract = (
            structured_output_contract
            if isinstance(structured_output_contract, Mapping)
            else {}
        )
        lines = [task]
        if contract:
            lines.extend(
                [
                    'Return exactly one JSON object and nothing else: no array, Markdown fence, commentary, or duplicate candidate rows.',
                    'Populate every required field from its named runtime evidence source. Copy canonical artifact_ref values verbatim and never substitute branch ids, paths, or sibling identities for them.',
                ]
            )
            for binding in contract.get('required_bindings') or []:
                if not isinstance(binding, Mapping):
                    continue
                field_name = str(binding.get('field_name') or '').strip()
                field_role = str(binding.get('field_role') or '').strip()
                source_phase_id = str(binding.get('source_phase_id') or '').strip()
                artifact_ref = str(binding.get('artifact_ref') or '').strip()
                message_id = str(binding.get('message_id') or '').strip()
                source = source_phase_id or artifact_ref or message_id or str(
                    binding.get('source_kind') or 'runtime evidence'
                ).strip()
                if field_name:
                    lines.append(
                        f'Required JSON field `{field_name}` ({field_role or "runtime binding"}) must use only source `{source}`.'
                    )
                elif field_role:
                    lines.append(
                        f'Runtime evidence role `{field_role}` is bound to source `{source}`; preserve it without regeneration.'
                    )
        lines.extend([
            'This is a post-artifact text phase. Use the fulfilled dependency artifacts/evidence; do not restart the original request, regenerate media, or say you cannot generate the artifacts.',
            'The dependency evidence is already fulfilled branch output for this phase; treat it as available input and do not ask the user to provide the artifact again.',
            'For a requested structured join, emit only the requested final structure. Treat each dependency block as evidence for its own branch; do not concatenate duplicate candidate rows or reuse one artifact reference for multiple branch identities.',
        ])
        if not contract:
            lines.append(
                'If dependency evidence includes both original text and transcript text, compare those strings directly and report whether they match.'
            )
        evidence_text = str(evidence or '').strip()
        if evidence_text:
            lines.append(f'Dependency evidence:\n{evidence_text}')
        return '\n\n'.join(lines).strip()

    @classmethod
    def _text_artifact_materialization_instruction(
        cls,
        original_prompt: str,
        artifact_gap: Mapping[str, Any],
    ) -> str:
        gap = artifact_gap if isinstance(artifact_gap, Mapping) else {}
        raw_requests = gap.get('text_artifact_requests') if isinstance(gap.get('text_artifact_requests'), list) else []
        text_artifact_requests = [
            {
                'extension': str(item.get('extension') or '').strip().lower().lstrip('.'),
                'source_name': str(item.get('source_name') or '').strip(),
            }
            for item in raw_requests
            if isinstance(item, Mapping)
            and str(item.get('extension') or '').strip()
            and str(item.get('source_name') or '').strip()
        ]
        artifact_request = gap.get('artifact_request') if isinstance(gap.get('artifact_request'), Mapping) else {}
        extension = str(
            gap.get('text_artifact_extension')
            or artifact_request.get('extension')
            or 'txt'
        ).strip().lower()
        source_name = str(
            gap.get('text_artifact_source_name')
            or artifact_request.get('source_name')
            or f'generated-{extension or "txt"}'
        ).strip()
        content_payload = str(gap.get('content_payload') or '').strip()
        target_path = str(
            gap.get('text_artifact_target_path')
            or artifact_request.get('target_path')
            or ''
        ).strip()
        if content_payload and cls._text_artifact_revision_required(gap):
            # Revision packets are constructed only from a complete, bounded
            # canonical source snapshot. The selected reply itself is not
            # repeated here and cannot become execution authority.
            prompt_text = str(original_prompt or '').strip()
            if len(prompt_text) > 12_000:
                prompt_text = f'{prompt_text[:12_000].rstrip()}\n[request context truncated at bounded revision limit]'
            lines = [
                f'Update the existing {extension or "text"} text artifact `{source_name}` only.',
                f'Target path: {target_path or source_name}',
                (
                    'The current user request below is the authoritative edit delta. '
                    'The source snapshot is input evidence only and does not fulfill this branch.'
                    if prompt_text
                    else 'The current user edit request in Ollmo promoted context is the authoritative edit delta. '
                    'The source snapshot is input evidence only and does not fulfill this branch.'
                ),
                'Output only the complete updated file body for that one target artifact.',
                'Preserve all existing copy, structure, selectors, behavior, links, and design outside the changes directly required by the current request.',
                'Do not redesign, summarize, omit unchanged sections, replace unrelated content, add commentary, or restart the earlier multi-file request.',
            ]
            if prompt_text:
                lines.extend(
                    [
                        'Current user edit request:',
                        prompt_text,
                    ]
                )
            lines.extend(
                [
                    'Complete canonical source snapshot:',
                    '--- SOURCE FILE START ---',
                    content_payload,
                    '--- SOURCE FILE END ---',
                ]
            )
            return '\n\n'.join(lines).strip()
        if content_payload and cls._artifact_gap_is_authoritative_bounded_text_artifact_repair(gap):
            if len(content_payload) > 90_000:
                content_payload = f'{content_payload[:90_000].rstrip()}\n\n[repair evidence truncated for prompt size]'
            lines = [
                f'Repair the existing {extension or "text"} text artifact `{source_name}` only.',
                f'Target path: {target_path or source_name}',
                'This is a bounded closure repair, not a regeneration task.',
                'Output only the complete corrected file body for that one target artifact.',
                'Preserve copy, layout intent, class names, and valid runtime artifact links unless the listed defect directly requires a narrow change.',
                'Do not redesign, translate, rename unrelated classes, replace unrelated content, add commentary, or restart the original request.',
                'Repair evidence and current saved file content:',
                content_payload,
            ]
            return '\n\n'.join(lines).strip()
        prompt_text = re.sub(r'\s+', ' ', str(original_prompt or '').strip())
        if len(prompt_text) > 1200:
            prompt_text = f'{prompt_text[:1200].rstrip()}...'
        if len(text_artifact_requests) > 1:
            request_lines = [
                f'- `{item["source_name"]}` as {item["extension"]}'
                for item in text_artifact_requests
            ]
            lines = [
                'Write the complete file payloads for these requested local text artifacts:',
                *request_lines,
                'Output exactly one fenced code block per file, in the same order, using each file extension as the fence language.',
                'Treat every listed file as one coherent local bundle: use the exact listed basenames for relative links, keep shared data and navigation consistent, and make HTML class names agree with the selectors in shared CSS.',
                'The set is atomic. Return every listed file again; do not return only a changed, missing, or previously failed member.',
                'Do not output planner JSON, request_ir, output_obligations, candidate_graph, or commentary outside those code blocks.',
            ]
            cohort_recovery = (
                gap.get('coalesced_text_artifact_recovery')
                if isinstance(
                    gap.get('coalesced_text_artifact_recovery'),
                    Mapping,
                )
                else {}
            )
            if cohort_recovery:
                lines.append(
                    'This is one bounded complete-set recovery attempt because '
                    'the prior response failed Ollmo atomic bundle validation. '
                    'Correct the complete ordered set in this response; do not '
                    'split the files into independent solutions.'
                )
            if prompt_text:
                lines.append(f'Original user request for bounded intent context: {prompt_text}')
            return '\n\n'.join(lines).strip()
        lines = [
            f'Write the complete {extension or "text"} file payload for `{source_name}`.',
            'Output only the file body for that one file.',
            'Do not output planner JSON, request_ir, output_obligations, candidate_graph, or commentary outside the artifact payload.',
        ]
        if prompt_text:
            lines.append(f'Original user request for bounded intent context: {prompt_text}')
        return '\n\n'.join(lines).strip()

    @staticmethod
    def _sequential_branch_local_artifact_labels(original_prompt: str) -> list[str]:
        labels = [
            str(match.group('label') or '').strip().upper()
            for match in _BRANCH_LOCAL_ARTIFACT_ALPHA_LABEL_RE.finditer(
                str(original_prompt or '')
            )
        ]
        if len(labels) < 2:
            return []
        expected = [chr(ord('A') + index) for index in range(len(labels))]
        return labels if labels == expected else []

    @classmethod
    def _branch_local_vision_analysis_directive(cls, original_prompt: str) -> str:
        """Extract the vision-owned instruction without replaying the root task."""

        prompt_text = re.sub(r'\s+', ' ', str(original_prompt or '')).strip()
        if not prompt_text:
            return ''
        candidates: list[tuple[int, int, int, str]] = []
        for segment_index, segment in enumerate(
            re.split(r'(?<=[.!?;])\s+', prompt_text)
        ):
            for start in _BRANCH_LOCAL_VISION_DIRECTIVE_START_RE.finditer(segment):
                directive = segment[start.start():].strip(' ,.;:-')
                stop_patterns = [_BRANCH_LOCAL_VISION_DIRECTIVE_STOP_RE]
                if cls._branch_local_vision_has_global_join(original_prompt):
                    stop_patterns.append(_BRANCH_LOCAL_VISION_GLOBAL_JOIN_STOP_RE)
                stop_matches = [
                    match
                    for pattern in stop_patterns
                    if (match := pattern.search(directive))
                ]
                if stop_matches:
                    stop = min(stop_matches, key=lambda match: match.start())
                    directive = directive[:stop.start()].strip(' ,.;:-')
                if not directive or not _BRANCH_LOCAL_VISION_TARGET_RE.search(directive):
                    continue
                action = str(start.group(0) or '').strip().lower()
                priority = 1
                if re.match(
                    r'(?:analy|inspect|examin|analysier|untersuch|pr[uü]f)',
                    action,
                    flags=re.IGNORECASE,
                ):
                    priority = 3
                elif re.match(
                    r'(?:describ|caption|beschreib|schilder)',
                    action,
                    flags=re.IGNORECASE,
                ):
                    priority = 2
                candidates.append(
                    (priority, segment_index, start.start(), directive[:600].rstrip())
                )
        if not candidates:
            return ''
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]

    @staticmethod
    def _branch_local_vision_evaluation_focus(original_prompt: str) -> str:
        """Extract only criteria that each evidence branch can judge independently."""

        prompt_text = re.sub(r'\s+', ' ', str(original_prompt or '')).strip()
        for pattern in _BRANCH_LOCAL_VISION_FOCUS_PATTERNS:
            match = pattern.search(prompt_text)
            if not match:
                continue
            focus = str(match.group('focus') or '').strip(' ,.;:-')
            if focus:
                return focus[:180].rstrip()
        return ''

    @staticmethod
    def _branch_local_vision_requires_identity_handoff(original_prompt: str) -> bool:
        prompt_text = str(original_prompt or '')
        return bool(
            _BRANCH_LOCAL_STRUCTURED_JOIN_RE.search(prompt_text)
            and _BRANCH_LOCAL_ARTIFACT_IDENTITY_FIELD_RE.search(prompt_text)
        )

    @classmethod
    def _branch_local_vision_has_global_join(cls, original_prompt: str) -> bool:
        prompt_text = str(original_prompt or '')
        if _BRANCH_LOCAL_VISION_EXPLICIT_GLOBAL_JOIN_RE.search(prompt_text):
            return True
        has_multiple_artifacts = bool(
            _BRANCH_LOCAL_VISION_MULTI_ARTIFACT_RE.search(prompt_text)
            or _BRANCH_LOCAL_VISION_FOR_MULTIPLE_LABELS_RE.search(prompt_text)
            or len(cls._sequential_branch_local_artifact_labels(prompt_text)) >= 2
        )
        return bool(
            has_multiple_artifacts
            and cls._branch_local_vision_requires_identity_handoff(prompt_text)
        )

    @classmethod
    def _vision_artifact_analysis_instruction(
        cls,
        original_prompt: str,
        evidence: str = '',
        *,
        execution_contract: Optional[Mapping[str, Any]] = None,
    ) -> str:
        contract = execution_contract if isinstance(execution_contract, Mapping) else {}
        try:
            branch_index = int(contract.get('queue_index') or 0)
        except (TypeError, ValueError):
            branch_index = 0
        labels = cls._sequential_branch_local_artifact_labels(original_prompt)
        branch_label = (
            labels[branch_index - 1]
            if branch_index > 0 and branch_index <= len(labels)
            else ''
        )
        local_analysis_directive = cls._branch_local_vision_analysis_directive(original_prompt)
        local_evaluation_focus = cls._branch_local_vision_evaluation_focus(original_prompt)
        requires_identity_handoff = cls._branch_local_vision_requires_identity_handoff(
            original_prompt
        )
        lines = [
            'Analyze exactly one actual attached generated image: the artifact supplied to this branch.',
            'The attachment and dependency evidence below are the complete execution authority for this branch.',
            'Do not generate a new image, do not describe hypothetical images, and do not restart the original multi-step request.',
            'Do not evaluate, compare, enumerate, label, or emit claims for sibling images or other artifacts that are not attached to this branch.',
            'Ignore root-level instructions to produce a multi-artifact table, list, or JSON join; a later text branch owns that global formatting task.',
            'Return one concise visual-evidence report for this attachment only so that a later text branch can safely use it.',
        ]
        if local_analysis_directive:
            lines.append(
                'Requested branch-local analysis directive (apply it only to this attachment; '
                f'plural wording means this single branch artifact): {local_analysis_directive}'
            )
        if local_evaluation_focus:
            lines.append(
                'Branch-local visual criteria inherited from the final evaluation: report visible '
                f'evidence for {local_evaluation_focus} in this attachment only; do not assess '
                'sibling modalities or perform the global comparison.'
            )
        if branch_index > 0:
            lines.append(f'Branch-local artifact ordinal: {branch_index}.')
        if branch_label:
            lines.append(
                f'Branch-local label: {branch_label}. Use only {branch_label} for this attachment.'
            )
        if requires_identity_handoff:
            if branch_label:
                lines.append(
                    f'Include label {branch_label} explicitly in this single-artifact evidence report.'
                )
            lines.append(
                'Include the exact attached artifact reference or path from Dependency evidence '
                'verbatim in this report. Do not invent, normalize, omit, or reuse a sibling identity.'
            )
        evidence_text = str(evidence or '').strip()
        if evidence_text:
            lines.append(f'Dependency evidence:\n{evidence_text}')
        return '\n\n'.join(lines).strip()

    def focus_late_fill_branch_gap_payload(
        self,
        branch: Mapping[str, Any],
        branch_gap: dict[str, Any],
        *,
        capability: Optional[str],
    ) -> dict[str, Any]:
        payload = dict(branch_gap or {})
        content_payload = str(payload.get('content_payload') or '').strip()
        if not content_payload:
            return payload

        if (
            capability == self.capability_text_to_speech
            and control_json_envelope_suspected(content_payload)
        ):
            payload.pop('prompt', None)
            payload['branch_contract_error'] = 'control_envelope_not_speakable'
            payload['materialization_blocked'] = True
            payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
            payload['content_payload_source'] = 'rejected_control_envelope'
            return payload

        artifact_prompt = str(payload.get('artifact_prompt') or '').strip()
        artifact_prompt_source = str(payload.get('artifact_prompt_source') or '').strip()
        authoritative_artifact_prompt_sources = {
            'semantic_batch_prompt',
            'semantic_batch_prompts',
            'semantic_prepare_phase_output',
            'current_turn_direct_image_clause',
            'action_input',
            'prompt_blockquote_section',
            'quoted_prompt_section',
            'inline_prompt_capsule',
            'current_turn_explicit_image_manifest',
        }
        should_focus_image_prompt = (
            not artifact_prompt
            or artifact_prompt == content_payload
            or (
                artifact_prompt_source not in authoritative_artifact_prompt_sources
                and len(artifact_prompt) > 240
                and artifact_prompt in content_payload
            )
        )
        if capability == self.capability_image_generation and should_focus_image_prompt:
            prompt_units = self._extract_late_fill_image_prompt_units(content_payload)
            if prompt_units:
                selected_index = self._branch_prompt_selection_index(branch, len(prompt_units))
                if selected_index <= 0:
                    payload.pop('artifact_prompt', None)
                    payload['branch_contract_error'] = 'incomplete_image_prompt_batch'
                    payload['candidate_extraction_issue'] = 'missing_branch_local_image_prompt_slot'
                    payload['materialization_blocked'] = True
                    payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                    return payload
                selected_prompt = prompt_units[selected_index - 1].strip()
                if selected_prompt:
                    payload['artifact_prompt'] = selected_prompt
                    payload['artifact_prompt_source'] = 'focused_image_prompt_slot'
                    payload['image_prompt_selection_index'] = selected_index
                    return payload
            candidates = [
                (index, self._strip_late_fill_prompt_label(unit), self._late_fill_image_prompt_score(unit))
                for index, unit in enumerate(self._split_late_fill_content_units(content_payload), start=1)
            ]
            viable = [(index, unit, score) for index, unit, score in candidates if unit and score > 0]
            if viable:
                selected_index = self._branch_prompt_selection_index(branch, len(viable))
                if selected_index <= 0:
                    payload.pop('artifact_prompt', None)
                    payload['branch_contract_error'] = 'incomplete_image_prompt_batch'
                    payload['candidate_extraction_issue'] = 'missing_branch_local_image_prompt_slot'
                    payload['materialization_blocked'] = True
                    payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                    return payload
                selected = viable[selected_index - 1]
                payload['artifact_prompt'] = selected[1]
                payload['artifact_prompt_source'] = 'focused_content_payload'
                payload['image_prompt_selection_index'] = selected_index
            return payload

        if capability == self.capability_text_to_speech:
            selection_policy = str(
                branch.get('selection_policy')
                or payload.get('selection_policy')
                or ''
            ).strip().lower()
            stage_direction = str(
                branch.get('stage_direction')
                or payload.get('stage_direction')
                or ''
            ).strip()
            counted_variant_match = re.fullmatch(
                r'materialize_requested_audio_variant_(\d+)',
                stage_direction,
            )
            if counted_variant_match and selection_policy != 'selected_candidate_only':
                payload['branch_contract_error'] = 'ambiguous_audio_variant_contract'
                payload['materialization_blocked'] = True
                payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                return payload
            if selection_policy in {'best_candidate_only', 'selected_candidate_only', 'selected_text_only'}:
                candidate_contract = self._late_fill_candidate_selection_contract(
                    content_payload
                )
                candidate_units = [
                    str(unit or '').strip()
                    for unit in (candidate_contract.get('units') or [])
                    if str(unit or '').strip()
                ]
                candidate_extraction_source = str(
                    candidate_contract.get('source') or ''
                ).strip()
                candidate_extraction_issue = str(
                    candidate_contract.get('issue') or ''
                ).strip()
                if candidate_extraction_issue:
                    payload['branch_contract_error'] = 'selected_candidate_unavailable'
                    payload['candidate_extraction_issue'] = candidate_extraction_issue
                    if candidate_extraction_source:
                        payload['candidate_extraction_source'] = candidate_extraction_source
                    payload['materialization_blocked'] = True
                    payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                    return payload
                raw_selection_count = branch.get('candidate_selection_count')
                if raw_selection_count in (None, ''):
                    raw_selection_count = payload.get('candidate_selection_count')
                if isinstance(raw_selection_count, bool):
                    normalized_selection_count = 0
                elif isinstance(raw_selection_count, int):
                    normalized_selection_count = raw_selection_count
                elif (
                    isinstance(raw_selection_count, str)
                    and raw_selection_count.isascii()
                    and raw_selection_count.isdigit()
                    and 0 < len(raw_selection_count) <= 6
                ):
                    normalized_selection_count = int(raw_selection_count)
                else:
                    normalized_selection_count = 0
                if counted_variant_match:
                    expected_index = int(counted_variant_match.group(1))
                    candidate_index = branch.get('candidate_selection_index')
                    if candidate_index in (None, ''):
                        candidate_index = payload.get('candidate_selection_index')
                    variant_index = branch.get('audio_variant_index')
                    if variant_index in (None, ''):
                        variant_index = payload.get('audio_variant_index')
                    try:
                        normalized_candidate_index = int(candidate_index)
                        normalized_variant_index = int(variant_index)
                    except (TypeError, ValueError):
                        normalized_candidate_index = 0
                        normalized_variant_index = 0
                    if (
                        normalized_selection_count <= 1
                        or normalized_candidate_index != expected_index
                        or normalized_variant_index != expected_index
                    ):
                        payload['branch_contract_error'] = 'ambiguous_audio_variant_contract'
                        payload['materialization_blocked'] = True
                        payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                        return payload
                if normalized_selection_count > 0:
                    normalized_units = [
                        re.sub(r'\s+', ' ', unit).strip().casefold()
                        for unit in candidate_units
                        if str(unit or '').strip()
                    ]
                    if len(normalized_units) != normalized_selection_count:
                        payload['branch_contract_error'] = 'selected_candidate_unavailable'
                        payload['materialization_blocked'] = True
                        payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                        return payload
                    if len(set(normalized_units)) != len(normalized_units):
                        payload['branch_contract_error'] = 'selected_candidate_not_distinct'
                        payload['materialization_blocked'] = True
                        payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                        return payload
                raw_selection_index = branch.get('candidate_selection_index')
                if raw_selection_index in (None, ''):
                    raw_selection_index = payload.get('candidate_selection_index')
                if selection_policy == 'best_candidate_only' and raw_selection_index in (None, ''):
                    raw_selection_index = 1
                if isinstance(raw_selection_index, bool):
                    normalized_selection_index = 0
                elif isinstance(raw_selection_index, int):
                    normalized_selection_index = raw_selection_index
                elif (
                    isinstance(raw_selection_index, str)
                    and raw_selection_index.isascii()
                    and raw_selection_index.isdigit()
                    and 0 < len(raw_selection_index) <= 6
                ):
                    normalized_selection_index = int(raw_selection_index)
                else:
                    normalized_selection_index = 0
                selected = (
                    candidate_units[normalized_selection_index - 1]
                    if 1 <= normalized_selection_index <= len(candidate_units)
                    else ''
                )
                if selected:
                    payload['content_payload'] = selected
                    payload['content_payload_source'] = 'selected_candidate_from_phase_output'
                    payload['selection_policy_applied'] = selection_policy
                    if candidate_extraction_source:
                        payload['candidate_extraction_source'] = candidate_extraction_source
                    return payload
                payload['branch_contract_error'] = 'selected_candidate_unavailable'
                if candidate_extraction_source:
                    payload['candidate_extraction_source'] = candidate_extraction_source
                payload['materialization_blocked'] = True
                payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                return payload
            labeled_speakable_sections = self._late_fill_labeled_speakable_sections(
                content_payload
            )
            labeled_audio_contract = self._late_fill_labeled_audio_candidate_contract(
                content_payload
            )
            if labeled_audio_contract.get('marker_detected'):
                labeled_audio_issue = str(
                    labeled_audio_contract.get('issue') or ''
                ).strip()
                labeled_audio_units = [
                    str(unit or '').strip()
                    for unit in (labeled_audio_contract.get('units') or [])
                    if str(unit or '').strip()
                ]
                if labeled_audio_issue or len(labeled_audio_units) != 1:
                    payload['branch_contract_error'] = 'ambiguous_audio_variant_contract'
                    payload['candidate_extraction_issue'] = (
                        labeled_audio_issue or 'multiple_audio_variant_bodies_without_selection'
                    )
                    payload['candidate_extraction_source'] = (
                        'labeled_audio_variant_sections'
                    )
                    payload['materialization_blocked'] = True
                    payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                    return payload
                payload['content_payload'] = labeled_audio_units[0]
                payload['content_payload_source'] = 'focused_labeled_speakable_text'
                payload['candidate_extraction_source'] = 'labeled_audio_variant_sections'
                return payload
            if len(labeled_speakable_sections) == 1:
                payload['content_payload'] = labeled_speakable_sections[0]
                payload['content_payload_source'] = 'focused_labeled_speakable_text'
                payload['candidate_extraction_source'] = 'labeled_speakable_section'
                return payload
            if len(labeled_speakable_sections) > 1:
                payload['branch_contract_error'] = 'ambiguous_audio_variant_contract'
                payload['candidate_extraction_issue'] = 'multiple_labeled_speakable_sections'
                payload['candidate_extraction_source'] = 'labeled_speakable_section'
                payload['materialization_blocked'] = True
                payload['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
                return payload
            units = self._split_late_fill_content_units(content_payload)
            scored = [
                (index, unit, self._late_fill_speakable_text_score(unit))
                for index, unit in enumerate(units, start=1)
                if unit
            ]
            has_non_speakable_media_prompt = any(score <= -5 for _index, _unit, score in scored)
            has_labeled_speakable_text = any(score >= 8 for _index, _unit, score in scored)
            viable = [(index, unit, score) for index, unit, score in scored if score > -5]
            if viable and (has_non_speakable_media_prompt or has_labeled_speakable_text):
                selected = max(viable, key=lambda item: item[2])
                payload['content_payload'] = selected[1]
                payload['content_payload_source'] = 'focused_content_payload'
            return payload

        return payload

    def failed_dependency_ids_for_branch(
        self,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
    ) -> list[str]:
        if not isinstance(branch, Mapping):
            return []
        dependency_ids: list[str] = []

        def _add_dependency_values(raw_value: Any) -> None:
            values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
            for value in values:
                token = str(value or '').strip()
                if token and token not in dependency_ids:
                    dependency_ids.append(token)

        _add_dependency_values(branch.get('depends_on') or branch.get('dependsOn'))
        execution_contract = (
            branch.get('execution_contract')
            if isinstance(branch.get('execution_contract'), Mapping)
            else {}
        )
        _add_dependency_values(execution_contract.get('depends_on') or execution_contract.get('dependencies'))
        if not dependency_ids:
            return []

        late_fill = (
            current_payload.get('late_fill')
            if isinstance((current_payload or {}).get('late_fill'), Mapping)
            else {}
        )
        failed_records = self.normalize_late_fill_branches(late_fill.get('failed_branches'))
        failed_tokens: set[str] = set()
        for record in failed_records:
            if not isinstance(record, Mapping):
                continue
            for key in ('branch_id', 'phase_id', 'task_id', 'workload_task_id', 'obligation_id'):
                token = str(record.get(key) or '').strip()
                if token:
                    failed_tokens.add(token)
            error = record.get('error') if isinstance(record.get('error'), Mapping) else {}
            failed_dependency_ids = error.get('failed_dependency_ids')
            if isinstance(failed_dependency_ids, (list, tuple, set)):
                for value in failed_dependency_ids:
                    token = str(value or '').strip()
                    if token:
                        failed_tokens.add(token)

        return [dependency_id for dependency_id in dependency_ids if dependency_id in failed_tokens]

    def _dependency_result_artifact_identity_evidence(
        self,
        result: Mapping[str, Any],
        *,
        fill_results: list[Any],
        current_payload: Mapping[str, Any],
    ) -> str:
        """Carry producer identity through an evidence consumer into a terminal join."""

        consumer_capability = self.normalize_capability(result.get('capability'))
        expected_producer_capability = {
            'vision_analysis': self.capability_image_generation,
            'speech_to_text': self.capability_text_to_speech,
        }.get(consumer_capability)
        if not expected_producer_capability:
            return ''

        runtime = (
            current_payload.get('runtime')
            if isinstance(current_payload.get('runtime'), Mapping)
            else {}
        )
        phase_graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        graph_records = [
            item
            for key in ('phases', 'downstream_branches')
            for item in (phase_graph.get(key) if isinstance(phase_graph.get(key), list) else [])
            if isinstance(item, Mapping)
        ]
        result_tokens = {
            str(result.get('phase_id') or '').strip(),
            str(result.get('branch_id') or '').strip(),
        }
        consumer_record = next(
            (
                item
                for item in graph_records
                if result_tokens.intersection(
                    {
                        str(item.get('phase_id') or '').strip(),
                        str(item.get('branch_id') or '').strip(),
                    }
                )
            ),
            {},
        )
        execution_contract = (
            result.get('execution_contract')
            if isinstance(result.get('execution_contract'), Mapping)
            else {}
        )
        dependency_ids: list[str] = []
        for raw_values in (
            result.get('depends_on'),
            execution_contract.get('depends_on'),
            consumer_record.get('depends_on') if isinstance(consumer_record, Mapping) else None,
        ):
            if not isinstance(raw_values, list):
                continue
            for value in raw_values:
                token = str(value or '').strip()
                if token and token not in dependency_ids:
                    dependency_ids.append(token)
        if not dependency_ids:
            return ''

        producer_results = [
            item
            for item in fill_results
            if isinstance(item, Mapping)
            and self.normalize_capability(item.get('capability'))
            == expected_producer_capability
            and {
                str(item.get('phase_id') or '').strip(),
                str(item.get('branch_id') or '').strip(),
            }.intersection(dependency_ids)
        ]
        if len(producer_results) != 1:
            return ''
        producer_result = producer_results[0]
        producer_artifacts = self._artifact_records_from_late_fill_result(producer_result)
        if not producer_artifacts:
            return ''

        canonical_artifacts: list[Mapping[str, Any]] = []
        try:
            canonical_artifacts = [
                item
                for item in self.build_canonical_response_artifacts(dict(current_payload))
                if isinstance(item, Mapping)
            ]
        except Exception:
            logging.getLogger(__name__).debug(
                'failed to build canonical artifacts for dependency identity evidence',
                exc_info=True,
            )
        evidence_blocks: list[str] = []
        producer_phase_id = str(producer_result.get('phase_id') or '').strip()
        producer_branch_id = str(producer_result.get('branch_id') or '').strip()
        for artifact in producer_artifacts:
            path = self._artifact_record_path(artifact)
            if not path:
                continue
            canonical = next(
                (
                    item
                    for item in canonical_artifacts
                    if self._artifact_record_path(item) == path
                ),
                {},
            )
            artifact_ref = str(
                canonical.get('artifact_ref')
                or canonical.get('ref')
                or artifact.get('artifact_ref')
                or artifact.get('ref')
                or ''
            ).strip()
            lines = [
                'Branch dependency artifact identity (runtime evidence):',
                f'- producer_phase_id: {producer_phase_id}' if producer_phase_id else '',
                f'- producer_branch_id: {producer_branch_id}' if producer_branch_id else '',
                f'- artifact_ref: {artifact_ref}' if artifact_ref else '',
                f'- artifact_path: {path}',
            ]
            block = '\n'.join(line for line in lines if line)
            if block and block not in evidence_blocks:
                evidence_blocks.append(block)
        return '\n\n'.join(evidence_blocks).strip()

    @staticmethod
    def _bounded_visual_evidence_from_selected_message(content: Any) -> str:
        """Extract only preserved visual evidence from a carried message."""
        return _extract_bounded_visual_evidence_from_selected_message(content)

    @staticmethod
    def _selected_reference_records_from_payload(
        current_payload: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        sources = [current_payload]
        request = current_payload.get('request')
        if isinstance(request, Mapping):
            sources.append(request)
        for source in sources:
            for key in (
                'selected_reference_artifacts',
                'selectedReferenceArtifacts',
                'reference_artifacts',
                'input_artifacts',
            ):
                for item in source.get(key) or []:
                    if isinstance(item, Mapping) and dict(item) not in records:
                        records.append(dict(item))
        return records

    @classmethod
    def _selected_reference_dependency_payload(
        cls,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected_refs = [
            dict(item)
            for item in (branch.get('input_refs') or [])
            if isinstance(item, Mapping)
            and str(item.get('kind') or '').strip()
            in {
                'selected_reference',
                'selected_reference_artifact',
                'selected_reference_evidence',
            }
        ]
        if not selected_refs:
            return {}
        available = cls._selected_reference_records_from_payload(current_payload)
        evidence_blocks: list[str] = []
        reference_artifacts: list[dict[str, Any]] = []
        for selected in selected_refs:
            selected_kind = str(selected.get('kind') or '').strip()
            selected_role = str(selected.get('role') or '').strip()
            is_artifact_reference = (
                selected_kind == 'selected_reference_artifact'
                or selected_role == 'preserved_visual_artifact'
            )
            match_keys = (
                ('artifact_ref', 'ref', 'path')
                if is_artifact_reference
                else ('message_id', 'source_response_id')
            )
            authoritative_values = {
                key: str(selected.get(key) or '').strip()
                for key in match_keys
                if str(selected.get(key) or '').strip()
            }
            if not authoritative_values:
                return {
                    'branch_contract_error': 'selected_reference_unavailable',
                    'materialization_blocked': True,
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                }
            matches: list[dict[str, Any]] = []
            matched_identities: set[tuple[str, ...]] = set()
            for record in available:
                record_ref = str(record.get('artifact_ref') or record.get('ref') or '').strip()
                record_values = {
                    'artifact_ref': record_ref,
                    'ref': record_ref,
                    'path': str(record.get('path') or record.get('artifact_path') or '').strip(),
                    'message_id': str(record.get('message_id') or '').strip(),
                    'source_response_id': str(record.get('source_response_id') or '').strip(),
                }
                if all(record_values.get(key) == value for key, value in authoritative_values.items()):
                    if is_artifact_reference:
                        identity = (
                            'artifact',
                            record_ref,
                            record_values['path'],
                            str(record.get('type') or record.get('kind') or '').strip().lower(),
                            str(record.get('source_response_id') or '').strip(),
                            str(
                                record.get('source_message_id')
                                or record.get('message_id')
                                or ''
                            ).strip(),
                        )
                    else:
                        identity = (
                            'message',
                            record_values['message_id'],
                            record_values['source_response_id'],
                            str(record.get('content') or record.get('text') or '').strip(),
                        )
                    if identity not in matched_identities:
                        matched_identities.add(identity)
                        matches.append(record)
            if len(matches) != 1:
                return {
                    'branch_contract_error': (
                        'selected_reference_unavailable'
                        if not matches
                        else 'selected_reference_ambiguous'
                    ),
                    'materialization_blocked': True,
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                }
            record = matches[0]
            if is_artifact_reference:
                artifact_ref = str(
                    record.get('artifact_ref') or record.get('ref') or ''
                ).strip()
                path = str(record.get('path') or record.get('artifact_path') or '').strip()
                artifact = {
                    key: value
                    for key, value in {
                        'type': record.get('type') or record.get('kind'),
                        'kind': record.get('kind') or record.get('type'),
                        'artifact_ref': artifact_ref or None,
                        'ref': artifact_ref or None,
                        'path': path or None,
                        'source_response_id': record.get('source_response_id'),
                        'source_message_id': record.get('source_message_id') or record.get('message_id'),
                    }.items()
                    if value not in (None, '')
                }
                reference_artifacts.append(artifact)
                evidence_blocks.append(
                    '\n'.join(
                        line
                        for line in (
                            'Selected preserved visual artifact (runtime-bound; do not regenerate):',
                            f'- artifact_ref: {artifact_ref}' if artifact_ref else '',
                            f'- artifact_path: {path}' if path else '',
                        )
                        if line
                    )
                )
                continue
            visual_evidence = cls._bounded_visual_evidence_from_selected_message(
                record.get('content') or record.get('text')
            )
            if not visual_evidence:
                return {
                    'branch_contract_error': 'selected_reference_evidence_unavailable',
                    'materialization_blocked': True,
                    'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                }
            evidence_blocks.append(
                'Selected preserved visual evidence (runtime-bound; do not reanalyze):\n'
                + visual_evidence
            )
        return {
            'evidence_blocks': evidence_blocks,
            'reference_artifacts': reference_artifacts,
            'selected_reference_evidence_bound': True,
        }

    def branch_dependency_payload(
        self,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        depends_on = [
            str(item or '').strip()
            for item in (branch.get('depends_on') or [])
            if str(item or '').strip()
        ]
        selected_reference_payload = self._selected_reference_dependency_payload(
            branch,
            current_payload=current_payload,
        )
        if selected_reference_payload.get('branch_contract_error'):
            return selected_reference_payload
        if self._branch_has_current_turn_direct_media_payload(branch):
            return {
                'dependency_payload_policy': (
                    'preserve_current_turn_direct_media_payload'
                )
            }
        if not depends_on and not selected_reference_payload:
            return {}
        target_capability = self.normalize_capability(branch.get('capability'))
        runtime_payload = (
            current_payload.get('runtime')
            if isinstance(current_payload.get('runtime'), Mapping)
            else {}
        )
        phase_graph_payload = (
            runtime_payload.get('request_phase_graph')
            if isinstance(runtime_payload.get('request_phase_graph'), Mapping)
            else {}
        )
        request_payload = (
            current_payload.get('request')
            if isinstance(current_payload.get('request'), Mapping)
            else {}
        )
        identity_handoff_prompt = str(
            phase_graph_payload.get('prompt')
            or self._request_payload_prompt_text(request_payload)
            or self._request_payload_prompt_text(current_payload)
            or ''
        ).strip()
        structured_identity_handoff = bool(
            target_capability == 'chat'
            and self._branch_local_vision_requires_identity_handoff(
                identity_handoff_prompt
            )
        )
        matched_result_payloads: list[tuple[str, str]] = [
            ('selected_reference', block)
            for block in (selected_reference_payload.get('evidence_blocks') or [])
            if str(block or '').strip()
        ]
        if 'phase-1' in depends_on:
            output_text = str((current_payload or {}).get('output_text') or '').strip()
            if output_text:
                if control_json_envelope_suspected(output_text):
                    return {
                        'branch_contract_error': 'control_envelope_not_speakable',
                        'materialization_blocked': True,
                        'repair_action': RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
                        'content_payload_source': 'rejected_current_phase_output',
                    }
                runtime = (
                    current_payload.get('runtime')
                    if isinstance(current_payload.get('runtime'), Mapping)
                    else {}
                )
                phase_graph = (
                    runtime.get('request_phase_graph')
                    if isinstance(runtime.get('request_phase_graph'), Mapping)
                    else {}
                )
                current_capability = self.normalize_capability(
                    phase_graph.get('current_phase_capability')
                )
                declared_source = str(branch.get('content_payload_source') or '').strip()
                if current_capability and current_capability != 'chat':
                    allowed_sources = {
                        'current_phase_output',
                        f'{current_capability}_branch_result',
                    }
                    if current_capability == 'speech_to_text':
                        allowed_sources.add('speech_to_text_branch_result')
                    if declared_source not in allowed_sources:
                        output_text = ''
                if output_text:
                    if len(depends_on) == 1:
                        return {
                            'content_payload': output_text,
                            'content_payload_source': 'current_phase_output',
                        }
                    matched_result_payloads.append(('phase-1', output_text))
        late_fill = (
            current_payload.get('late_fill')
            if isinstance(current_payload.get('late_fill'), Mapping)
            else {}
        )
        fill_results = (
            late_fill.get('fill_results')
            if isinstance(late_fill.get('fill_results'), list)
            else []
        )
        dependency_input_artifacts: list[dict[str, Any]] = []
        for result in fill_results:
            if not isinstance(result, Mapping):
                continue
            result_tokens = {
                str(result.get('branch_id') or '').strip(),
                str(result.get('phase_id') or '').strip(),
            }
            if not any(token and token in depends_on for token in result_tokens):
                continue
            if self.late_fill_result_has_missing_dependency_evidence(result):
                continue
            result_id = str(result.get('branch_id') or result.get('phase_id') or '').strip()
            if target_capability in {'vision_analysis', 'speech_to_text'}:
                for artifact in self._artifact_records_from_late_fill_result(result):
                    artifact_type = str(artifact.get('type') or artifact.get('kind') or '').strip().lower()
                    if (
                        target_capability == 'vision_analysis'
                        and artifact_type in {'image', 'png', 'jpg', 'jpeg', 'webp'}
                    ) or (
                        target_capability == 'speech_to_text'
                        and artifact_type in {'audio', 'wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg', 'opus'}
                    ):
                        dependency_input_artifacts.append(artifact)
            result_text = self.late_fill_text_from_result_payload(result)
            if result_text:
                if structured_identity_handoff:
                    identity_evidence = self._dependency_result_artifact_identity_evidence(
                        result,
                        fill_results=fill_results,
                        current_payload=current_payload,
                    )
                    if identity_evidence:
                        result_text = f'{result_text}\n\n{identity_evidence}'.strip()
                matched_result_payloads.append(
                    (
                        result_id,
                        result_text,
                    )
                )
                continue
            artifact_evidence = self._artifact_evidence_text(result)
            if artifact_evidence:
                matched_result_payloads.append(
                    (
                        result_id,
                        artifact_evidence,
                    )
                )
        if dependency_input_artifacts and target_capability in {'vision_analysis', 'speech_to_text'}:
            first_path = str(dependency_input_artifacts[0].get('path') or '').strip()
            evidence = '\n'.join(
                f"{str(item.get('type') or item.get('kind') or 'artifact').strip() or 'artifact'} artifact: {str(item.get('path') or '').strip()}"
                for item in dependency_input_artifacts
                if str(item.get('path') or '').strip()
            ).strip()
            payload: dict[str, Any] = {
                'input_artifacts': dependency_input_artifacts,
                'reference_artifacts': dependency_input_artifacts,
            }
            if first_path:
                payload['file_path'] = first_path
            if evidence:
                payload['content_payload'] = evidence
                payload['content_payload_source'] = 'late_fill_dependency_artifacts'
            return payload
        selected_reference_artifacts = [
            dict(item)
            for item in (selected_reference_payload.get('reference_artifacts') or [])
            if isinstance(item, Mapping)
        ]
        if len(matched_result_payloads) == 1:
            result_id, result_text = matched_result_payloads[0]
            if result_text:
                payload = {
                    'content_payload': result_text,
                    'content_payload_source': f'late_fill_result:{result_id}',
                }
                if selected_reference_artifacts:
                    payload['reference_artifacts'] = selected_reference_artifacts
                    payload['selected_reference_evidence_bound'] = True
                return payload
        if matched_result_payloads:
            combined = '\n\n'.join(
                f'{result_id}: {result_text}' if result_id else result_text
                for result_id, result_text in matched_result_payloads
                if result_text
            ).strip()
            if combined:
                payload = {
                    'content_payload': combined,
                    'content_payload_source': 'late_fill_results:' + ','.join(
                        result_id for result_id, _text in matched_result_payloads if result_id
                    ),
                }
                if selected_reference_artifacts:
                    payload['reference_artifacts'] = selected_reference_artifacts
                    payload['selected_reference_evidence_bound'] = True
                return payload
        return {}

    def merge_late_fill_result_fields(
        self,
        response_payload: dict[str, Any],
        infer_result: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(response_payload or {})
        existing_output_artifacts = self.merge_unique_artifact_records(
            updated.get('artifacts'),
            [],
        )
        identified_existing_paths = {
            self._artifact_record_path(item)
            for item in existing_output_artifacts
            if self._artifact_record_path(item)
            and str(
                item.get('artifact_id')
                or item.get('artifact_ref')
                or item.get('ref')
                or ''
            ).strip()
        }
        existing_refs = {
            str(item.get('artifact_ref') or item.get('ref') or '').strip()
            for item in existing_output_artifacts
            if str(item.get('artifact_ref') or item.get('ref') or '').strip()
        }
        missing_existing_artifacts = [
            item
            for item in self.build_canonical_response_artifacts(updated)
            if (
                (
                    not self._artifact_record_path(item)
                    or self._artifact_record_path(item) not in identified_existing_paths
                )
                and (
                    not str(item.get('artifact_ref') or item.get('ref') or '').strip()
                    or str(item.get('artifact_ref') or item.get('ref') or '').strip() not in existing_refs
                )
            )
        ]
        existing_output_artifacts = self.merge_unique_artifact_records(
            existing_output_artifacts,
            missing_existing_artifacts,
        )
        for key in (
            'saved_text_path',
            'saved_text_artifacts',
            'text_artifact_requests',
            'text_artifact_revision_required',
            'text_artifact_source_is_input',
            'text_artifact_revision_write_proof',
            'text_artifact_revision_preservation_evidence',
            'saved_audio_path',
            'saved_image_path',
            'image_data_url',
            'seed',
            'image_state',
            'audio_mimetype',
            'lang_code',
            'lang_code_source',
            'voice',
            'instruct',
            'response_format',
            'output_format',
            'speed',
            'pitch',
            'tts_semantic_source',
            'tts_generation_budget',
            'tts_sampling_profile',
            'tts_audio_integrity_evidence',
            'reference_image_count',
            'reference_image_kind',
            'cached',
            'cache_id',
            'result',
            'image_artifact_persisted_from_raw_late_fill',
            'provenance_id',
            'derived_from',
        ):
            if key in infer_result and infer_result.get(key) not in (None, ''):
                updated[key] = infer_result.get(key)
        for key in ('input_artifacts', 'reference_artifacts'):
            if key not in infer_result:
                continue
            merged_artifacts = self.merge_unique_artifact_records(
                updated.get(key),
                infer_result.get(key),
            )
            if merged_artifacts:
                updated[key] = merged_artifacts
        existing_warnings = [
            str(item).strip()
            for item in (updated.get('warnings') or [])
            if str(item).strip()
        ]
        new_warnings = [
            str(item).strip()
            for item in (infer_result.get('warnings') or [])
            if str(item).strip()
        ]
        if existing_warnings or new_warnings:
            updated['warnings'] = list(dict.fromkeys(existing_warnings + new_warnings))
        incoming_output_artifacts = self.build_branch_local_late_fill_artifacts(
            updated,
            infer_result,
        )
        merged_output_artifacts = self.merge_unique_artifact_records(
            existing_output_artifacts,
            incoming_output_artifacts,
        )
        updated['artifacts'] = merged_output_artifacts
        return updated

    def build_branch_local_late_fill_artifacts(
        self,
        response_payload: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        *,
        capability: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Canonicalize only one producer result without aggregate identity leakage."""
        branch_payload = dict(infer_result or {})
        response_id = str(
            response_payload.get('id')
            or response_payload.get('response_id')
            or ''
        ).strip()
        if response_id:
            branch_payload['id'] = response_id
            branch_payload['response_id'] = response_id
        artifacts = self.merge_unique_artifact_records(
            infer_result.get('artifacts'),
            self.build_canonical_response_artifacts(branch_payload),
        )

        normalized_capability = self.normalize_capability(
            capability
            or infer_result.get('capability')
            or infer_result.get('mode')
        )
        artifact_type = self.artifact_type_for_capability(normalized_capability)
        path_key_by_type = {
            'audio': 'saved_audio_path',
            'image': 'saved_image_path',
            'text': 'saved_text_path',
        }
        primary_path = str(
            infer_result.get(path_key_by_type.get(str(artifact_type or ''), ''))
            or ''
        ).strip()
        if not primary_path:
            for path_key in ('saved_image_path', 'saved_audio_path', 'saved_text_path'):
                candidate = str(infer_result.get(path_key) or '').strip()
                if candidate:
                    primary_path = candidate
                    break
        primary_artifact = next(
            (
                item
                for item in artifacts
                if (
                    (not primary_path or self._artifact_record_path(item) == primary_path)
                    and (
                        not artifact_type
                        or str(item.get('type') or item.get('kind') or '').strip().lower()
                        == artifact_type
                    )
                )
            ),
            None,
        )
        if primary_artifact is None and len(artifacts) == 1:
            primary_artifact = artifacts[0]

        if primary_artifact is not None:
            explicit_artifact_id = str(infer_result.get('artifact_id') or '').strip()
            explicit_artifact_ref = str(
                infer_result.get('artifact_ref')
                or infer_result.get('ref')
                or ''
            ).strip()
            if explicit_artifact_id:
                primary_artifact['artifact_id'] = explicit_artifact_id
            if explicit_artifact_ref:
                primary_artifact['artifact_ref'] = explicit_artifact_ref
                primary_artifact['ref'] = explicit_artifact_ref
            elif explicit_artifact_id:
                primary_artifact['artifact_ref'] = f'artifact:{explicit_artifact_id}'
                primary_artifact['ref'] = primary_artifact['artifact_ref']
        return artifacts

    def attach_late_fill_result_artifact_identity(
        self,
        fill_record: Mapping[str, Any],
        response_payload: Mapping[str, Any],
        infer_result: Mapping[str, Any],
        *,
        capability: Optional[str] = None,
    ) -> dict[str, Any]:
        """Persist a branch's own artifact identity on its durable fill result."""
        updated = dict(fill_record or {})
        artifacts = self.build_branch_local_late_fill_artifacts(
            response_payload,
            infer_result,
            capability=capability,
        )
        if not artifacts:
            return updated

        binding_keys = ('branch_id', 'phase_id', 'slot_id', 'obligation_id', 'output_type')
        bound_artifacts: list[dict[str, Any]] = []
        for raw_artifact in artifacts:
            artifact = dict(raw_artifact)
            for key in binding_keys:
                if updated.get(key) not in (None, '', [], {}):
                    artifact.setdefault(key, updated.get(key))
            bound_artifacts.append(artifact)
        updated['artifacts'] = self.merge_unique_artifact_records(
            updated.get('artifacts'),
            bound_artifacts,
        )

        artifact_type = self.artifact_type_for_capability(
            self.normalize_capability(
                capability
                or infer_result.get('capability')
                or infer_result.get('mode')
            )
        )
        path_key_by_type = {
            'audio': 'saved_audio_path',
            'image': 'saved_image_path',
            'text': 'saved_text_path',
        }
        primary_path = str(
            infer_result.get(path_key_by_type.get(str(artifact_type or ''), ''))
            or ''
        ).strip()
        primary_artifact = next(
            (
                item
                for item in updated['artifacts']
                if (
                    (not primary_path or self._artifact_record_path(item) == primary_path)
                    and (
                        not artifact_type
                        or str(item.get('type') or item.get('kind') or '').strip().lower()
                        == artifact_type
                    )
                )
            ),
            None,
        )
        if primary_artifact is None and len(updated['artifacts']) == 1:
            primary_artifact = updated['artifacts'][0]
        if primary_artifact is not None:
            for key in (
                'artifact_id',
                'artifact_ref',
                'ref',
                'provenance_id',
                'derived_from',
                'source_response_id',
            ):
                if updated.get(key) in (None, '', [], {}) and primary_artifact.get(key) not in (None, '', [], {}):
                    updated[key] = primary_artifact.get(key)
        return updated

    def persist_raw_image_result_if_possible(
        self,
        infer_result: dict[str, Any],
        *,
        effective_data: Optional[Mapping[str, Any]] = None,
        late_fill_instance: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        if not isinstance(infer_result, dict):
            return {}
        if str(infer_result.get('saved_image_path') or '').strip():
            return infer_result
        data_url = _extract_raw_image_data_url(infer_result)
        if not data_url:
            return infer_result
        effective_payload = effective_data if isinstance(effective_data, Mapping) else {}
        instance_payload = late_fill_instance if isinstance(late_fill_instance, Mapping) else {}
        model_name = str(
            infer_result.get('model')
            or effective_payload.get('model')
            or instance_payload.get('model')
            or 'image'
        ).strip() or 'image'
        try:
            saved_image_path = self.persist_image_data_url_locally(data_url, model_name)
        except Exception as exc:  # noqa: BLE001
            logging.info('Could not persist raw late-fill image result: %s', exc)
            return infer_result
        if not saved_image_path:
            return infer_result
        updated = dict(infer_result)
        updated['saved_image_path'] = saved_image_path
        updated.setdefault('image_data_url', data_url)
        updated['image_artifact_persisted_from_raw_late_fill'] = True
        return updated

    def branch_has_downstream_capability(
        self,
        branch: Mapping[str, Any],
        *,
        current_payload: Mapping[str, Any],
        downstream_capability: str,
    ) -> bool:
        runtime = current_payload.get('runtime') if isinstance(current_payload.get('runtime'), Mapping) else {}
        phase_graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        phases = phase_graph.get('phases') if isinstance(phase_graph.get('phases'), list) else []
        source_tokens = {
            str(branch.get('branch_id') or '').strip(),
            str(branch.get('phase_id') or '').strip(),
        }
        source_tokens.discard('')
        if not source_tokens:
            return False
        for phase in phases:
            if not isinstance(phase, Mapping):
                continue
            if self.normalize_capability(phase.get('capability')) != downstream_capability:
                continue
            depends_on = {
                str(item or '').strip()
                for item in (phase.get('depends_on') or [])
                if str(item or '').strip()
            }
            if depends_on.intersection(source_tokens):
                return True
        return False

    def merge_late_fill_result_into_response_payload(
        self,
        response_payload: dict[str, Any],
        infer_result: dict[str, Any],
        late_fill_state: dict[str, Any],
    ) -> dict[str, Any]:
        updated = self.merge_late_fill_result_fields(response_payload, infer_result)
        return self.attach_late_fill_state(updated, late_fill_state)

    def build_deferred_follow_up_gap_for_capability(
        self,
        artifact_gap: Optional[dict[str, Any]],
        *,
        capability: Optional[str],
        artifact_payload: Optional[dict[str, Any]] = None,
        pending_capabilities: Optional[list[str]] = None,
        completed_capabilities: Optional[list[str]] = None,
        failed_capabilities: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        normalized_capability = self.normalize_capability(capability)
        payload = dict(artifact_gap or {})
        payload['expected_capability'] = normalized_capability
        payload['active_capability'] = normalized_capability
        payload['missing_artifact_type'] = self.artifact_type_for_capability(normalized_capability)
        if pending_capabilities is not None:
            payload['pending_capabilities'] = self.normalize_capability_list(pending_capabilities)
        if completed_capabilities is not None:
            payload['completed_capabilities'] = self.normalize_capability_list(completed_capabilities)
        if failed_capabilities is not None:
            payload['failed_capabilities'] = self.normalize_capability_list(failed_capabilities)
        semantic_payload = self.semantic_payload_for_capability(
            artifact_payload,
            capability=normalized_capability,
        )
        for key, value in semantic_payload.items():
            if payload.get(key) in (None, '', [], {}):
                payload[key] = value
        return payload

    def prepare_late_fill_branch_plan(
        self,
        *,
        expected_capability: str,
        artifact_gap: dict[str, Any],
        current_payload: dict[str, Any],
        request_payload: dict[str, Any],
        assistant_message: str,
        source_route_payload: Optional[dict[str, Any]],
        failed_instance_id: Optional[str],
        excluded_instance_ids: Optional[list[str]] = None,
        build_deferred_follow_up_gap_for_capability: Optional[Callable[..., dict[str, Any]]] = None,
        prepare_late_fill_request_payload: Optional[Callable[..., dict[str, Any]]] = None,
        resolve_late_fill_route: Optional[Callable[..., tuple[dict[str, Any], Optional[dict[str, Any]], Optional[str]]]] = None,
    ) -> dict[str, Any]:
        build_gap = build_deferred_follow_up_gap_for_capability or self.build_deferred_follow_up_gap_for_capability
        prepare_payload = prepare_late_fill_request_payload or self.prepare_late_fill_request_payload
        resolve_route = resolve_late_fill_route or self.resolve_late_fill_route

        active_gap = build_gap(
            artifact_gap,
            capability=expected_capability,
            artifact_payload=current_payload,
        )
        if isinstance(active_gap.get('execution_contract'), Mapping) or active_gap.get('branch_id') or active_gap.get('phase_id'):
            active_gap = self.attach_execution_contract_to_gap(
                active_gap,
                active_gap,
                capability=expected_capability,
            )
        late_fill_request_payload = prepare_payload(
            request_payload,
            expected_capability=expected_capability or '',
            assistant_message=assistant_message,
            artifact_gap=active_gap,
        )
        branch_contract_error = str(
            late_fill_request_payload.get('branch_contract_error') or ''
        ).strip()
        if (
            str(active_gap.get('trigger') or '').strip().lower()
            in {'graph_patch_successor_reopen', 'graph_rebase_partial_successor'}
            and branch_contract_error
        ):
            raise RuntimeError(
                'Bounded successor execution requires a branch-local payload; '
                f'late-fill preparation failed closed ({branch_contract_error}).'
            )
        if branch_contract_error in {
            'selected_candidate_unavailable',
            'selected_candidate_not_distinct',
            'ambiguous_audio_variant_contract',
        }:
            raise RuntimeError(
                'Selected TTS candidate contract is invalid; '
                'late-fill preparation failed closed instead of reusing or duplicating another candidate.'
            )
        if branch_contract_error == 'selected_reference_audio_unavailable':
            raise RuntimeError(
                'Selected-reference STT contract is invalid; '
                'late-fill preparation failed closed instead of transcribing another audio source.'
            )
        if branch_contract_error in {
            'preserved_visual_reference_missing',
            'preserved_visual_reference_ambiguous',
            'preserved_visual_evidence_missing',
            'preserved_visual_evidence_ambiguous',
            'structured_audio_pair_lineage_ambiguous',
            'selected_reference_unavailable',
            'selected_reference_ambiguous',
            'selected_reference_evidence_unavailable',
            'text_revision_source_unavailable',
            'text_revision_source_exceeds_prompt_bound',
            'ambiguous_text_artifact_revision_source',
        }:
            raise RuntimeError(
                'Preserved selected-reference or text-revision contract is invalid; '
                'late-fill preparation failed closed instead of regenerating or guessing source evidence.'
            )
        late_fill_request_payload = self.merge_late_fill_payload_artifacts(
            late_fill_request_payload,
            current_payload,
            source_route_payload,
            request_payload,
        )
        late_fill_request_payload, late_fill_route_info, route_error = resolve_route(
            late_fill_request_payload,
            expected_capability=expected_capability or '',
            failed_instance_id=failed_instance_id,
            excluded_instance_ids=excluded_instance_ids,
            artifact_gap=active_gap,
            source_route_payload=source_route_payload,
        )
        if route_error or not late_fill_route_info:
            exc = RuntimeError(route_error or 'Late fill route could not be resolved.')
            route_diagnostics = (
                late_fill_request_payload.get('_late_fill_route_diagnostics')
                if isinstance(late_fill_request_payload.get('_late_fill_route_diagnostics'), Mapping)
                else {}
            )
            if route_diagnostics:
                setattr(exc, 'route_diagnostics', dict(route_diagnostics))
            raise exc

        late_fill_instance = late_fill_route_info.get('instance') if isinstance(late_fill_route_info.get('instance'), dict) else {}
        late_fill_request_payload = _retarget_model_scoped_reasoning_effort(
            late_fill_request_payload,
            late_fill_instance,
            source_instance_id=str(
                request_payload.get('instance_id')
                or request_payload.get('instanceId')
                or ''
            ).strip(),
        )
        if str(
            late_fill_instance.get('target_kind') or ''
        ).strip().lower() == 'external':
            if (
                self.normalize_capability(expected_capability) != 'chat'
                or not callable(self.execute_external_chat_phase)
            ):
                raise RuntimeError(
                    'External late-fill execution is supported only for '
                    'graph-owned chat phases.'
                )
            execution_contract = (
                late_fill_request_payload.get('execution_contract')
                if isinstance(
                    late_fill_request_payload.get('execution_contract'),
                    Mapping,
                )
                else active_gap.get('execution_contract')
                if isinstance(active_gap.get('execution_contract'), Mapping)
                else {}
            )
            if not execution_contract or not str(
                execution_contract.get('branch_id')
                or execution_contract.get('phase_id')
                or ''
            ).strip():
                raise RuntimeError(
                    'External chat continuation requires a graph-owned branch '
                    'execution contract.'
                )
            bounded_task_prompt = self.extract_responses_prompt(
                late_fill_request_payload
            )
            if (
                str(active_gap.get('stage_direction') or '').strip()
                == 'materialize_requested_text_artifact'
            ):
                bounded_task_prompt = (
                    self._text_artifact_materialization_instruction(
                        '',
                        active_gap,
                    )
                    or bounded_task_prompt
                )
            root_prompt = self.extract_responses_prompt(request_payload)
            if not root_prompt:
                root_prompt = str(
                    late_fill_request_payload.get('_prompt_hint') or ''
                ).strip()
            if not bounded_task_prompt or not root_prompt:
                raise RuntimeError(
                    'External graph-owned chat continuation requires both '
                    'root reference context and a branch-local bounded task.'
                )
            root_scoped = bool(
                str(
                    execution_contract.get('execution_scope') or ''
                ).strip().lower()
                in {'root', 'root_scoped', 'whole_request', 'original_prompt'}
                or self.parse_bool(
                    execution_contract.get('root_scoped'),
                    default=False,
                )
                or self.parse_bool(
                    execution_contract.get('allow_root_prompt'),
                    default=False,
                )
            )
            if bounded_task_prompt.strip() == root_prompt.strip() and not root_scoped:
                raise RuntimeError(
                    'External graph-owned chat continuation refused to replay '
                    'the root request as branch-local execution.'
                )
            context_messages = [
                dict(item)
                for item in (
                    late_fill_request_payload.get('ghost_messages') or []
                )
                if isinstance(item, Mapping)
            ]
            effective_data = dict(late_fill_request_payload)
            effective_data['execution_contract'] = dict(execution_contract)
            infer_payload = {
                'prompt': bounded_task_prompt,
                'execution_contract': dict(execution_contract),
            }
            return {
                'capability': 'chat',
                'branch_id': str(
                    execution_contract.get('branch_id')
                    or effective_data.get('branch_id')
                    or ''
                ).strip()
                or None,
                'phase_id': str(
                    execution_contract.get('phase_id')
                    or effective_data.get('phase_id')
                    or ''
                ).strip()
                or None,
                'execution_contract': dict(execution_contract),
                'route_info': late_fill_route_info,
                'instance': late_fill_instance,
                'effective_data': effective_data,
                'infer_payload': infer_payload,
                'external_chat_phase': {
                    'root_prompt': root_prompt,
                    'context_messages': context_messages,
                    'bounded_task_prompt': bounded_task_prompt,
                },
                'expose_input_artifacts': False,
            }
        effective_data, late_fill_route_info, _planner_meta, _control_hints = self.prepare_effective_request_data(
            late_fill_request_payload,
            route_info=late_fill_route_info,
            instance=late_fill_instance if isinstance(late_fill_instance, dict) else None,
        )
        missing_session_controls = self.build_missing_required_session_controls(
            late_fill_instance if isinstance(late_fill_instance, dict) else {},
            effective_data,
        )
        if missing_session_controls:
            raise RuntimeError(str(missing_session_controls[0].get('message') or 'Required session controls missing for late fill.'))

        late_fill_backend = self.normalize_backend(
            effective_data.get('backend')
            or (late_fill_instance or {}).get('backend')
        )
        late_fill_request_model = self.select_backend_request_model(
            late_fill_instance if isinstance(late_fill_instance, dict) else None,
            effective_data.get('request_model') or (late_fill_instance or {}).get('request_model'),
            effective_data.get('model') or (late_fill_instance or {}).get('model'),
        ) or str(effective_data.get('model') or (late_fill_instance or {}).get('model') or '').strip()
        infer_payload, late_fill_route_info, _has_file_context, late_fill_expose_input_artifacts = self.build_responses_infer_execution_payload(
            effective_data,
            route_info=late_fill_route_info,
            instance=late_fill_instance if isinstance(late_fill_instance, dict) else {},
            instance_id=str((late_fill_route_info or {}).get('instance_id') or '').strip(),
            backend=late_fill_backend,
            capability=str((late_fill_route_info or {}).get('capability') or '').strip(),
            request_model_override=late_fill_request_model,
            upload_present=False,
        )
        execution_contract = (
            infer_payload.get('execution_contract')
            if isinstance(infer_payload.get('execution_contract'), dict)
            else (
                effective_data.get('execution_contract')
                if isinstance(effective_data.get('execution_contract'), dict)
                else {}
            )
        )
        return {
            'capability': expected_capability,
            'branch_id': str((execution_contract or {}).get('branch_id') or effective_data.get('branch_id') or '').strip() or None,
            'phase_id': str((execution_contract or {}).get('phase_id') or effective_data.get('phase_id') or '').strip() or None,
            'execution_contract': dict(execution_contract) if execution_contract else {},
            'route_info': late_fill_route_info,
            'instance': late_fill_instance,
            'effective_data': effective_data,
            'infer_payload': infer_payload,
            'expose_input_artifacts': late_fill_expose_input_artifacts,
        }

    def _materialize_external_chat_text_artifact_outputs(
        self,
        plan: Mapping[str, Any],
        infer_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist external file bodies through Ollmo before coalesced expansion."""

        effective_data = (
            plan.get('effective_data')
            if isinstance(plan.get('effective_data'), Mapping)
            else {}
        )
        branch_wrapper = (
            plan.get('branch')
            if isinstance(plan.get('branch'), Mapping)
            else {}
        )
        branch = (
            branch_wrapper.get('branch')
            if isinstance(branch_wrapper.get('branch'), Mapping)
            else branch_wrapper
        )
        stage_direction = str(
            effective_data.get('stage_direction')
            or branch.get('stage_direction')
            or ''
        ).strip()
        if stage_direction != 'materialize_requested_text_artifact':
            return infer_result

        requests = [
            dict(item)
            for item in (effective_data.get('text_artifact_requests') or [])
            if isinstance(item, Mapping)
        ]
        if not requests:
            artifact_request = (
                effective_data.get('artifact_request')
                if isinstance(
                    effective_data.get('artifact_request'),
                    Mapping,
                )
                else branch.get('artifact_request')
                if isinstance(branch.get('artifact_request'), Mapping)
                else {}
            )
            if artifact_request:
                requests = [dict(artifact_request)]
        if not requests:
            return infer_result

        provider_result = dict(infer_result)
        normalized_requests: list[dict[str, Any]] = []
        for request in requests:
            normalized = dict(request)
            extension = str(
                normalized.get('extension')
                or effective_data.get('text_artifact_extension')
                or branch.get('text_artifact_extension')
                or ''
            ).strip().lower().lstrip('.')
            source_name = str(
                normalized.get('source_name')
                or effective_data.get('text_artifact_source_name')
                or branch.get('text_artifact_source_name')
                or ''
            ).strip()
            target_path = str(
                normalized.get('target_path')
                or effective_data.get('text_artifact_target_path')
                or branch.get('text_artifact_target_path')
                or ''
            ).strip()
            if extension:
                normalized['extension'] = extension
            if source_name:
                normalized['source_name'] = source_name
            if target_path:
                normalized['target_path'] = target_path
            normalized_requests.append(normalized)

        provider_text = self.late_fill_text_from_result_payload(provider_result)
        extracted_payloads = extract_text_artifact_payloads(
            provider_text,
            normalized_requests,
        )
        if len(extracted_payloads) != len(normalized_requests):
            raise RuntimeError(
                'TEXT_ARTIFACT_OUTPUT_SET_INCOMPLETE: External chat did not '
                'return one distinct valid file body for every coalesced text '
                'artifact request.'
            )

        def request_identity(value: Mapping[str, Any]) -> tuple[str, str, str]:
            return (
                str(value.get('target_path') or '').strip(),
                str(value.get('source_name') or '').strip().lower(),
                str(value.get('extension') or '').strip().lower().lstrip('.'),
            )

        unassigned_payload_indexes = set(range(len(extracted_payloads)))
        payload_by_request_index: dict[int, dict[str, Any]] = {}
        for request_index, request in enumerate(normalized_requests):
            identity = request_identity(request)
            candidates = [
                payload_index
                for payload_index in sorted(unassigned_payload_indexes)
                if isinstance(
                    extracted_payloads[payload_index].get('artifact_request'),
                    Mapping,
                )
                and request_identity(
                    extracted_payloads[payload_index]['artifact_request']
                )
                == identity
            ]
            if not candidates and request_index in unassigned_payload_indexes:
                # The extractor walks ordinary fenced output in request order.
                # This fallback is only needed for legacy requests without a
                # stable source name or target-path identity.
                candidates = [request_index]
            if len(candidates) != 1:
                raise RuntimeError(
                    'TEXT_ARTIFACT_OUTPUT_BINDING_AMBIGUOUS: External chat '
                    'file bodies could not be bound one-to-one to the '
                    'coalesced artifact manifest.'
                )
            payload_index = candidates[0]
            unassigned_payload_indexes.remove(payload_index)
            payload_by_request_index[request_index] = dict(
                extracted_payloads[payload_index]
            )

        prepared_outputs: list[tuple[dict[str, Any], str, str, str, str]] = []
        for request_index, request in enumerate(normalized_requests):
            extension = str(
                request.get('extension')
                or effective_data.get('text_artifact_extension')
                or branch.get('text_artifact_extension')
                or ''
            ).strip().lower().lstrip('.')
            source_name = str(
                request.get('source_name')
                or effective_data.get('text_artifact_source_name')
                or branch.get('text_artifact_source_name')
                or ''
            ).strip()
            target_path = str(
                request.get('target_path')
                or effective_data.get('text_artifact_target_path')
                or branch.get('text_artifact_target_path')
                or ''
            ).strip()
            extracted_content = str(
                payload_by_request_index[request_index].get('content') or ''
            ).strip()
            canonical_content, content_error = (
                self._text_artifact_content_payload_error(
                    extracted_content,
                    extension=extension,
                    saved_path=target_path,
                )
            )
            if content_error:
                code = str(
                    content_error.get('code')
                    or 'TEXT_ARTIFACT_OUTPUT_INVALID'
                ).strip()
                raise RuntimeError(
                    f"{code}: {content_error.get('message') or 'External chat returned an invalid file body.'}"
                )
            if not canonical_content:
                raise RuntimeError(
                    'TEXT_ARTIFACT_OUTPUT_INVALID: External chat did not '
                    'return a complete file body matching the requested '
                    f'{source_name or "text artifact"}.{extension or "txt"} identity.'
                )
            prepared_outputs.append(
                (
                    dict(request),
                    extension,
                    source_name,
                    target_path,
                    canonical_content,
                )
            )

        late_fill_instance = (
            plan.get('instance')
            if isinstance(plan.get('instance'), Mapping)
            else {}
        )
        persistence_model_name = str(
            late_fill_instance.get('model')
            or late_fill_instance.get('instance_id')
            or late_fill_instance.get('id')
            or 'external-chat'
        ).strip()
        runtime_bundle_dir: Optional[Path] = None
        if any(not target_path for _, _, _, target_path, _ in prepared_outputs):
            bundle_identity = '|'.join(
                [
                    str(plan.get('branch_id') or ''),
                    str(effective_data.get('response_id') or ''),
                    ','.join(
                        f'{source_name}.{extension}'
                        for _, extension, source_name, _, _ in prepared_outputs
                    ),
                    str(time.time_ns()),
                ]
            )
            bundle_suffix = hashlib.sha256(
                bundle_identity.encode('utf-8')
            ).hexdigest()[:12]
            bundle_timestamp = time.strftime(
                '%Y%m%dT%H%M%SZ',
                time.gmtime(),
            )
            runtime_bundle_dir = ARTIFACT_OUTPUTS_DOCUMENTS_DIR / (
                f'{bundle_timestamp}_external_chat_bundle_{bundle_suffix}'
            )
        runtime_allocated_target_paths: dict[int, Path] = {}
        allocated_filename_owners: dict[str, str] = {}
        if runtime_bundle_dir is not None:
            for output_index, (
                _request,
                extension,
                source_name,
                target_path,
                _content,
            ) in enumerate(prepared_outputs):
                if target_path:
                    continue
                safe_source_name = re.sub(
                    r'[^A-Za-z0-9._-]+',
                    '_',
                    Path(
                        source_name or f'generated-{extension or "txt"}'
                    ).stem,
                ).strip('._') or 'generated-text'
                allocated_filename = (
                    f'{safe_source_name}.{extension or "txt"}'
                )
                collision_key = unicodedata.normalize(
                    'NFC',
                    allocated_filename,
                ).casefold()
                request_identity = f'{source_name}.{extension}'
                prior_owner = allocated_filename_owners.get(collision_key)
                if prior_owner is not None:
                    raise RuntimeError(
                        'TEXT_ARTIFACT_PATH_COLLISION: Distinct requested '
                        f'artifacts `{prior_owner}` and `{request_identity}` '
                        f'would both map to `{allocated_filename}`. No file '
                        'was written.'
                    )
                allocated_filename_owners[collision_key] = request_identity
                runtime_allocated_target_paths[output_index] = (
                    runtime_bundle_dir / allocated_filename
                )
        materialized_results: list[dict[str, Any]] = []
        saved_records: list[dict[str, Any]] = []
        for output_index, (
            request,
            extension,
            source_name,
            target_path,
            extracted_content,
        ) in enumerate(prepared_outputs):
            request_payload = {
                **dict(effective_data),
                'artifact_request': dict(request),
                'text_artifact_request': dict(request),
                'text_artifact_extension': extension,
                'text_artifact_source_name': source_name,
                'text_artifact_target_path': target_path,
            }
            request_branch = {
                **dict(branch),
                'requires_artifact': True,
                'stage_direction': 'materialize_requested_text_artifact',
                'artifact_request': dict(request),
                'text_artifact_request': dict(request),
                'text_artifact_extension': extension,
                'text_artifact_source_name': source_name,
                'text_artifact_target_path': target_path,
            }
            materialization_result = {
                **dict(provider_result),
                'content': extracted_content,
                'content_payload': extracted_content,
                'result_text': extracted_content,
                'output_text': extracted_content,
            }
            if target_path:
                materialized, materialization_error = (
                    self._materialize_required_text_artifact_target_path(
                        request_branch,
                        materialization_result,
                        request_payload,
                        extension=extension,
                        source_name=source_name,
                    )
                )
                if materialization_error:
                    code = str(
                        materialization_error.get('code')
                        or 'TEXT_ARTIFACT_NOT_PERSISTED'
                    ).strip()
                    raise RuntimeError(
                        f"{code}: {materialization_error.get('message') or 'External chat file body did not pass Ollmo materialization checks.'}"
                    )
            else:
                if runtime_bundle_dir is None:
                    raise RuntimeError(
                        'TEXT_ARTIFACT_PATH_ALLOCATION_FAILED: Ollmo did not '
                        'allocate a bundle directory for a new text artifact.'
                    )
                allocated_target_path = runtime_allocated_target_paths.get(
                    output_index
                )
                if allocated_target_path is None:
                    raise RuntimeError(
                        'TEXT_ARTIFACT_PATH_ALLOCATION_FAILED: Ollmo did not '
                        'bind a unique target path to a new text artifact.'
                    )
                saved_path = persist_text_artifact_locally(
                    extracted_content,
                    model_name=persistence_model_name,
                    source_name=source_name or f'generated-{extension or "txt"}',
                    mode='external_chat_text_artifact',
                    extension=extension or 'txt',
                    output_dir=ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
                    target_path=str(allocated_target_path),
                )
                if not saved_path or not Path(saved_path).is_file():
                    raise RuntimeError(
                        'TEXT_ARTIFACT_NOT_PERSISTED: Ollmo could not allocate '
                        'and save the external chat file body.'
                    )
                saved_error = self._text_artifact_saved_payload_error(
                    saved_path,
                    extension=extension,
                )
                if saved_error:
                    code = str(
                        saved_error.get('code')
                        or 'TEXT_ARTIFACT_NOT_PERSISTED'
                    ).strip()
                    raise RuntimeError(
                        f"{code}: {saved_error.get('message') or 'The Ollmo-persisted file body did not pass saved-truth checks.'}"
                    )
                saved_artifact_request = self._required_text_artifact_request(
                    request_branch,
                    materialization_result,
                    request_payload,
                    extension=extension,
                    source_name=source_name,
                    target_path=saved_path,
                )
                materialized = self._with_required_text_artifact_saved_result(
                    materialization_result,
                    target_path=saved_path,
                    content=extracted_content,
                    extension=extension,
                    source_name=source_name,
                    artifact_request=saved_artifact_request,
                    evidence='external_chat_runtime_persisted_text_artifact',
                )
            materialized_results.append(materialized)
            for record in materialized.get('saved_text_artifacts') or []:
                if not isinstance(record, Mapping):
                    continue
                path = str(
                    record.get('path') or record.get('saved_text_path') or ''
                ).strip()
                if path and not any(
                    str(
                        item.get('path') or item.get('saved_text_path') or ''
                    ).strip()
                    == path
                    for item in saved_records
                ):
                    saved_records.append(dict(record))

        if len(materialized_results) == 1:
            updated = dict(materialized_results[0])
            if isinstance(provider_result.get('output_text'), str):
                updated['output_text'] = provider_result['output_text']
            return updated
        updated = dict(provider_result)
        updated['saved_text_artifacts'] = saved_records
        if saved_records:
            first = saved_records[0]
            updated['saved_text_path'] = str(
                first.get('path') or first.get('saved_text_path') or ''
            ).strip()
        return updated

    def execute_prepared_late_fill_branch(self, plan: dict[str, Any]) -> dict[str, Any]:
        infer_payload = plan.get('infer_payload') if isinstance(plan.get('infer_payload'), dict) else {}
        late_fill_instance = (
            plan.get('instance')
            if isinstance(plan.get('instance'), Mapping)
            else {}
        )
        external_chat_phase = (
            plan.get('external_chat_phase')
            if isinstance(plan.get('external_chat_phase'), Mapping)
            else {}
        )
        if self.normalize_capability(plan.get('capability')) == self.capability_text_to_speech:
            effective_data = (
                plan.get('effective_data')
                if isinstance(plan.get('effective_data'), Mapping)
                else {}
            )
            for candidate in (
                infer_payload.get('prompt'),
                effective_data.get('content_payload'),
            ):
                if control_json_envelope_suspected(str(candidate or '').strip()):
                    raise RuntimeError(
                        'control_envelope_not_speakable: repair_branch_contract is required '
                        'before text-to-speech execution'
                    )
        route_info = (
            dict(plan.get('route_info'))
            if isinstance(plan.get('route_info'), Mapping)
            else {}
        )
        if str(
            late_fill_instance.get('target_kind') or ''
        ).strip().lower() == 'external':
            if (
                self.normalize_capability(plan.get('capability')) != 'chat'
                or not external_chat_phase
                or not callable(self.execute_external_chat_phase)
            ):
                raise RuntimeError(
                    'Invalid external late-fill plan: only a bounded '
                    'graph-owned chat phase may use an external target.'
                )
            external_result = self.execute_external_chat_phase(
                request_payload=(
                    plan.get('effective_data')
                    if isinstance(plan.get('effective_data'), dict)
                    else {}
                ),
                target=dict(late_fill_instance),
                root_prompt=str(
                    external_chat_phase.get('root_prompt') or ''
                ).strip(),
                context_messages=[
                    dict(item)
                    for item in (
                        external_chat_phase.get('context_messages') or []
                    )
                    if isinstance(item, Mapping)
                ],
                bounded_task_prompt=str(
                    external_chat_phase.get('bounded_task_prompt') or ''
                ).strip(),
            )
            external_status = str(
                external_result.get('status') or ''
            ).strip().lower()
            external_execution = (
                external_result.get('external_execution')
                if isinstance(
                    external_result.get('external_execution'),
                    Mapping,
                )
                else {}
            )
            if external_execution:
                route_runtime = (
                    dict(route_info.get('route_runtime') or {})
                    if isinstance(route_info.get('route_runtime'), Mapping)
                    else {}
                )
                route_runtime['external_execution'] = dict(
                    external_execution
                )
                route_info['route_runtime'] = route_runtime
            if external_status == 'failed':
                error = (
                    external_result.get('error')
                    if isinstance(external_result.get('error'), Mapping)
                    else {}
                )
                exc = RuntimeError(
                    str(
                        error.get('message')
                        or 'External graph-owned chat phase failed.'
                    )
                )
                status_code = error.get('status_code')
                if status_code is not None:
                    setattr(exc, 'status_code', status_code)
                raise exc
            if external_status == 'blocked':
                infer_result = {
                    'external_provider_block': {
                        'kind': 'ollmo.external_provider_block',
                        'code': 'EXTERNAL_PROVIDER_BLOCKED',
                        'reason': str(
                            external_result.get('blocked_reason')
                            or 'The downstream provider blocked the bounded task.'
                        ).strip(),
                    },
                    'external_execution': dict(external_execution),
                }
            elif external_status == 'completed':
                output_text = str(
                    external_result.get('output_text') or ''
                )
                infer_result = {
                    'output_text': output_text,
                    'content': output_text,
                    'content_payload': output_text,
                    'result_text': output_text,
                    'mode': 'external_chat_phase',
                    'external_execution': dict(external_execution),
                }
                infer_result = (
                    self._materialize_external_chat_text_artifact_outputs(
                        plan,
                        infer_result,
                    )
                )
            else:
                raise RuntimeError(
                    'External graph-owned chat phase returned no terminal status.'
                )
        else:
            infer_result, status_code = self.invoke_internal_api_json_route(
                payload=infer_payload,
                upload=None,
            )
            if status_code >= 400:
                raise RuntimeError(str(infer_result.get('error') or 'Late fill request failed.'))
            infer_result = self.filter_responses_infer_result(
                infer_result,
                expose_input_artifacts=bool(plan.get('expose_input_artifacts')),
            )
        execution_contract = (
            plan.get('execution_contract')
            if isinstance(plan.get('execution_contract'), dict)
            else (
                infer_payload.get('execution_contract')
                if isinstance(infer_payload.get('execution_contract'), dict)
                else (
                    (plan.get('effective_data') or {}).get('execution_contract')
                    if isinstance(plan.get('effective_data'), dict)
                    and isinstance((plan.get('effective_data') or {}).get('execution_contract'), dict)
                    else {}
                )
            )
        )
        if isinstance(infer_result, dict) and execution_contract:
            infer_result = dict(infer_result)
            infer_result.setdefault('execution_contract', dict(execution_contract))
            workload_task_ref = (
                execution_contract.get('workload_task_ref')
                if isinstance(execution_contract.get('workload_task_ref'), Mapping)
                else {}
            )
            output_obligation_ref = (
                execution_contract.get('output_obligation_ref')
                if isinstance(execution_contract.get('output_obligation_ref'), Mapping)
                else {}
            )
            for key in ('branch_id', 'phase_id', 'capability', 'output_type'):
                value = execution_contract.get(key)
                if value not in (None, '', [], {}):
                    infer_result.setdefault(key, value)
            task_id = workload_task_ref.get('task_id') if isinstance(workload_task_ref, Mapping) else None
            if task_id not in (None, '', [], {}):
                infer_result.setdefault('task_id', task_id)
                infer_result.setdefault('workload_task_id', task_id)
                infer_result.setdefault('workload_task_ref', dict(workload_task_ref))
            obligation_id = (
                output_obligation_ref.get('obligation_id')
                if isinstance(output_obligation_ref, Mapping)
                else None
            )
            if obligation_id not in (None, '', [], {}):
                infer_result.setdefault('obligation_id', obligation_id)
                infer_result.setdefault('output_obligation_ref', dict(output_obligation_ref))
        if (
            isinstance(infer_result, dict)
            and self.normalize_capability(plan.get('capability'))
            == self.capability_text_to_speech
        ):
            semantic_source = self.tts_source_evidence_from_prepared_plan(
                plan,
                infer_result=infer_result,
            )
            if semantic_source:
                infer_result = dict(infer_result)
                infer_result['tts_semantic_source'] = semantic_source
            integrity_branch = {
                'capability': self.capability_text_to_speech,
                'branch_id': plan.get('branch_id'),
                'phase_id': plan.get('phase_id'),
            }
            if isinstance(execution_contract, Mapping):
                integrity_branch.update(dict(execution_contract))
            integrity_evidence = self.tts_audio_integrity_evidence_for_branch_result(
                integrity_branch,
                infer_result,
            )
            if integrity_evidence:
                infer_result = dict(infer_result)
                infer_result['tts_audio_integrity_evidence'] = (
                    integrity_evidence
                )
        return {
            'capability': self.normalize_capability(plan.get('capability')),
            'branch_id': str(plan.get('branch_id') or (execution_contract or {}).get('branch_id') or '').strip() or None,
            'phase_id': str(plan.get('phase_id') or (execution_contract or {}).get('phase_id') or '').strip() or None,
            'execution_contract': dict(execution_contract) if execution_contract else {},
            'route_info': route_info,
            'instance': plan.get('instance') if isinstance(plan.get('instance'), dict) else {},
            'effective_data': plan.get('effective_data') if isinstance(plan.get('effective_data'), dict) else {},
            'infer_result': infer_result if isinstance(infer_result, dict) else {},
        }

    def execute_late_fill_branch(
        self,
        *,
        expected_capability: str,
        artifact_gap: dict[str, Any],
        current_payload: dict[str, Any],
        request_payload: dict[str, Any],
        assistant_message: str,
        source_route_payload: Optional[dict[str, Any]],
        failed_instance_id: Optional[str],
        excluded_instance_ids: Optional[list[str]] = None,
        prepare_late_fill_branch_plan: Optional[Callable[..., dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        prepare_plan = prepare_late_fill_branch_plan or self.prepare_late_fill_branch_plan
        plan = prepare_plan(
            expected_capability=expected_capability,
            artifact_gap=artifact_gap,
            current_payload=current_payload,
            request_payload=request_payload,
            assistant_message=assistant_message,
            source_route_payload=source_route_payload,
            failed_instance_id=failed_instance_id,
            excluded_instance_ids=excluded_instance_ids,
        )
        return self.execute_prepared_late_fill_branch(plan)

    def build_late_fill_materialization_branch_spec(
        self,
        *,
        branch: dict[str, Any],
        artifact_gap: dict[str, Any],
        current_payload: dict[str, Any],
        request_payload: dict[str, Any],
        assistant_message: str,
        source_route_payload: Optional[dict[str, Any]],
        failed_instance_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        capability = self.branch_capability(branch)
        branch_id = self.branch_id(branch)
        if not capability or not branch_id:
            return None
        branch_gap = dict(artifact_gap or {})
        successor_branch_execution = (
            str(branch_gap.get('trigger') or branch.get('trigger') or '').strip().lower()
            in {'graph_patch_successor_reopen', 'graph_rebase_partial_successor'}
        )
        if successor_branch_execution:
            # A terminal repair successor may inherit the frozen parent's
            # Late Fill envelope, but none of its prompt carriers. Executable
            # prompt truth must be bound to the exact owed branch (or derived
            # from that branch's dependencies) rather than recovered from the
            # original request, assistant prose, or a prior global batch.
            for key in (
                'artifact_prompt',
                'artifact_prompt_source',
                'content_payload',
                'content_payload_source',
                'phase_summary',
                'stage_direction',
                'batch_prompts',
                'batch_prompts_source',
            ):
                branch_gap.pop(key, None)
        recovery_context = branch.get('recovery_context') if isinstance(branch.get('recovery_context'), Mapping) else {}
        attempt = branch.get('attempt') if isinstance(branch.get('attempt'), Mapping) else {}
        retry_excluded_instance_ids = [
            str(item).strip()
            for item in (
                branch.get('excluded_instance_ids')
                if isinstance(branch.get('excluded_instance_ids'), list)
                else recovery_context.get('exclude_instance_ids')
                if isinstance(recovery_context.get('exclude_instance_ids'), list)
                else []
            )
            if str(item).strip()
        ]
        branch_failed_instance_id = str(
            branch.get('failed_instance_id')
            or attempt.get('instance_id')
            or failed_instance_id
            or ''
        ).strip() or None
        if capability == self.capability_text_to_speech:
            # A retry-wave envelope carries the anchor's recovery markers.
            # TTS branch markers remain authoritative so sibling retries do
            # not inherit the anchor identity during the narrow reuse policy.
            for key in ('recovery_state', 'recovery_attempt'):
                value = branch.get(key)
                if isinstance(value, Mapping) and value:
                    branch_gap[key] = dict(value)
        for key in (
            'artifact_prompt',
            'artifact_prompt_source',
            'content_payload',
            'content_payload_source',
            'selection_policy',
            'candidate_selection_index',
            'candidate_selection_count',
            'selection_reason',
            'audio_variant_index',
            'audio_variant_role',
            'audio_variant_contract_source',
            'structured_output_contract',
            'branch_contract_error',
            'audio_variant_contract_conflicting_fields',
            'phase_summary',
            'stage_direction',
            'batch_prompts',
            'batch_prompts_source',
            'batch_prompt_source_phase_id',
            'batch_prompt_expected_count',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'text_artifact_revision_required',
            'text_artifact_revision_source',
            'text_artifact_source_is_input',
            'text_artifact_revision_binding_state',
            'text_artifact_revision_preservation_required',
            'text_artifact_revision_preservation_policy',
            'text_artifact_requests',
            'artifact_request',
            'lang_code',
            'voice',
            'instruct',
            'response_format',
            'output_format',
            'speed',
            'pitch',
            'repair_scope',
            'resource_class',
            'dependency_policy',
            'runtime_scheduling_context',
            'allow_gpu_heavy_concurrency',
            'repair_action',
            'recovery_action',
            'suggested_action',
            'repair_execution_policy',
            'execution_policy',
            'auto_execute',
            'repair_work_available',
            'needs_external_input',
            'materialization_blocked',
            'auto_executable_repair_retry_count',
            'auto_executable_repair_max_attempts',
            'repair_auto_execute_max_attempts',
            'max_auto_execute_attempts',
        ):
            value = branch.get(key)
            if value not in (None, '', [], {}):
                branch_gap[key] = value
        revision_source_payload = self._text_artifact_revision_source_payload(
            branch,
            branch_gap,
        )
        if revision_source_payload:
            branch_gap.update(revision_source_payload)
        if (
            self._text_artifact_revision_required(branch, branch_gap)
            and self._text_artifact_revision_preservation_requested(
                self._request_payload_prompt_text(request_payload)
            )
        ):
            branch_gap['text_artifact_revision_preservation_required'] = True
            branch_gap['text_artifact_revision_preservation_policy'] = (
                'structural_anchor_retention_v1'
            )
        preserve_authoritative_text_repair_payload = (
            self._artifact_gap_is_authoritative_bounded_text_artifact_repair(branch_gap)
            and str(branch_gap.get('content_payload') or '').strip()
        )
        preserve_text_artifact_revision_source = bool(
            self._text_artifact_revision_required(branch, branch_gap)
            and str(branch_gap.get('content_payload') or '').strip()
            and str(branch_gap.get('content_payload_source') or '').strip()
            == 'canonical_predecessor_text_artifact_snapshot'
        )
        preserve_direct_spoken_payload = bool(
            str(branch_gap.get('content_payload') or '').strip()
            and str(branch_gap.get('content_payload_source') or '').strip()
            == 'current_turn_direct_spoken_clause'
        )
        dependency_payload = self.branch_dependency_payload(
            branch,
            current_payload=current_payload,
        )
        for key, value in dependency_payload.items():
            if value not in (None, '', [], {}):
                if (
                    preserve_authoritative_text_repair_payload
                    and key in {'content_payload', 'content_payload_source'}
                ):
                    branch_gap['dependency_payload_policy'] = 'preserve_authoritative_text_repair_payload'
                    continue
                if (
                    preserve_text_artifact_revision_source
                    and key in {'content_payload', 'content_payload_source'}
                ):
                    branch_gap['dependency_payload_policy'] = (
                        'preserve_text_artifact_revision_source'
                    )
                    continue
                if (
                    preserve_direct_spoken_payload
                    and key in {'content_payload', 'content_payload_source'}
                ):
                    branch_gap['dependency_payload_policy'] = (
                        'preserve_current_turn_direct_media_payload'
                    )
                    continue
                branch_gap[key] = value
        branch_gap = self.focus_late_fill_branch_gap_payload(
            branch,
            branch_gap,
            capability=capability,
        )
        selected_reference_dependency_refs: list[Any] = []
        for source in (branch, branch_gap):
            selected_reference_dependency_refs.extend(source.get('input_refs') or [])
            execution_contract = (
                source.get('execution_contract')
                if isinstance(source.get('execution_contract'), Mapping)
                else {}
            )
            selected_reference_dependency_refs.extend(
                execution_contract.get('input_refs') or []
            )
        has_explicit_selected_reference_dependency = any(
            isinstance(item, Mapping)
            and str(item.get('kind') or '').strip().startswith('selected_reference')
            for item in selected_reference_dependency_refs
        )
        branch_local_image_prompt_source = str(
            branch_gap.get('artifact_prompt_source') or ''
        ).strip()
        branch_local_image_prompt_sources = {
            'semantic_batch_prompt',
            'semantic_batch_prompts',
            'semantic_prepare_phase_output',
            'current_turn_direct_image_clause',
            'action_input',
            'prompt_blockquote_section',
            'quoted_prompt_section',
            'inline_prompt_capsule',
            'focused_image_prompt_slot',
            'focused_content_payload',
            'request_prompt_image_slots',
            'current_turn_explicit_image_manifest',
        }
        if (
            capability == self.capability_image_generation
            and str(branch_gap.get('artifact_prompt') or '').strip()
            and branch_local_image_prompt_source in branch_local_image_prompt_sources
            and not has_explicit_selected_reference_dependency
        ):
            # Ambient selected-reply context is not execution authority for an
            # already focused branch-local prompt. Image edits that truly need
            # a selected reference opt in through an explicit dependency edge.
            branch_gap['suppress_reference_file_context'] = True
            branch_gap['selected_reference_prompt_policy'] = (
                'suppressed_for_current_turn_branch_prompt'
                if branch_local_image_prompt_source
                in {
                    'current_turn_direct_image_clause',
                    'current_turn_explicit_image_manifest',
                }
                else 'suppressed_for_branch_local_image_prompt'
            )
        image_prompt_allows_batch_override = (
            capability == self.capability_image_generation
            and self._image_branch_prompt_allows_batch_prompt_override(branch, branch_gap)
        )
        if (
            capability == self.capability_image_generation
            and image_prompt_allows_batch_override
            and not successor_branch_execution
            and not isinstance(branch_gap.get('batch_prompts'), list)
        ):
            request_prompt_units = self._extract_late_fill_image_prompt_units(
                self._request_payload_prompt_text(request_payload)
            )
            if len(request_prompt_units) >= 2:
                branch_gap['batch_prompts'] = request_prompt_units
                branch_gap['batch_prompts_source'] = 'request_prompt_image_slots'
        branch_gap = self.attach_execution_contract_to_gap(
            branch,
            branch_gap,
            capability=capability,
        )
        runtime_scheduling_context = self.runtime_scheduling_context_for_branch(
            branch,
            current_payload=current_payload,
            source_route_payload=source_route_payload,
            artifact_gap=branch_gap,
        )
        if runtime_scheduling_context:
            branch_gap['runtime_scheduling_context'] = runtime_scheduling_context
        if (
            capability == self.capability_image_generation
            and self.branch_has_downstream_capability(
                branch,
                current_payload=current_payload,
                downstream_capability='vision_analysis',
            )
        ):
            branch_gap['suppress_image_state_enrichment'] = True
            branch_gap.setdefault('image_state_enrichment_suppression_reason', 'downstream_vision_analysis')
        if (
            capability == self.capability_image_generation
            and not self._branch_allows_gpu_heavy_concurrency(branch, branch_gap)
            and self.response_has_required_artifact_closure_work(current_payload)
        ):
            branch_gap['suppress_image_state_enrichment'] = True
            branch_gap['suppress_generated_image_enrichment'] = True
            branch_gap.setdefault('image_state_enrichment_suppression_reason', 'required_artifact_closure_priority')
        if (
            capability == self.capability_image_generation
            and image_prompt_allows_batch_override
            and isinstance(branch_gap.get('batch_prompts'), list)
        ):
            expected_prompt_count = self._image_prompt_batch_expected_count(
                branch=branch,
                branch_gap=branch_gap,
                current_payload=current_payload,
            )
            batch_prompts = self._normalize_late_fill_image_batch_prompts(
                branch_gap.get('batch_prompts'),
                expected_count=expected_prompt_count,
                assistant_message='' if successor_branch_execution else assistant_message,
                content_payload=branch_gap.get('content_payload'),
                request_payload={} if successor_branch_execution else request_payload,
                artifact_prompt=branch_gap.get('artifact_prompt'),
            )
            if batch_prompts and batch_prompts != branch_gap.get('batch_prompts'):
                branch_gap['batch_prompts'] = batch_prompts
                branch_gap['batch_prompts_normalized'] = True
            selection_index = self._branch_prompt_selection_index(branch, len(batch_prompts))
            if 1 <= selection_index <= len(batch_prompts):
                branch_gap['artifact_prompt'] = batch_prompts[selection_index - 1]
                branch_gap['artifact_prompt_source'] = (
                    str(branch_gap.get('batch_prompts_source') or '').strip()
                    or 'semantic_batch_prompt'
                )
                branch_gap['image_prompt_selection_index'] = selection_index
            elif batch_prompts:
                branch_gap.pop('artifact_prompt', None)
                branch_gap['branch_contract_error'] = 'incomplete_image_prompt_batch'
                branch_gap['candidate_extraction_issue'] = 'missing_branch_local_image_prompt_slot'
                branch_gap['materialization_blocked'] = True
                branch_gap['repair_action'] = RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
        return {
            'branch_id': branch_id,
            'phase_id': str(branch.get('phase_id') or branch_id).strip() or branch_id,
            'capability': capability,
            'reservation_group': capability,
            'branch': dict(branch),
            'prepare_args': {
                'expected_capability': capability,
                'artifact_gap': branch_gap,
                'current_payload': dict(current_payload),
                'request_payload': dict(request_payload),
                'assistant_message': assistant_message,
                'source_route_payload': source_route_payload,
                'failed_instance_id': branch_failed_instance_id,
                'excluded_instance_ids': retry_excluded_instance_ids,
            },
        }

    def _prepare_terminal_graph_patch_successor_handoff(
        self,
        finalized_parent_payload: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        assistant_message: str,
        artifact_gap: dict[str, Any],
        source_route_payload: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        """Persist a validated pending successor before its Late Fill worker starts."""

        if not callable(self.prepare_terminal_graph_patch_successor):
            return finalized_parent_payload, None
        try:
            prepared = self.prepare_terminal_graph_patch_successor(finalized_parent_payload)
        except Exception as exc:  # noqa: BLE001
            logging.warning('Could not prepare terminal graph-patch successor: %s', exc)
            return finalized_parent_payload, None
        if not isinstance(prepared, dict):
            return finalized_parent_payload, None
        if prepared.get('status') != 'queued':
            return self._persist_terminal_graph_patch_review_audit(
                finalized_parent_payload,
                prepared=prepared,
                request_payload=request_payload,
            )
        successor_payload = (
            prepared.get('response_payload')
            if isinstance(prepared.get('response_payload'), dict)
            else {}
        )
        successor_gap = (
            prepared.get('artifact_gap')
            if isinstance(prepared.get('artifact_gap'), dict)
            else {}
        )
        if not successor_payload or not successor_gap:
            return finalized_parent_payload, None
        execution = prepared.get('execution') if isinstance(prepared.get('execution'), dict) else {}
        successor_execution_key = str(execution.get('successor_execution_key') or '').strip()
        response_id = str(
            successor_payload.get('id')
            or successor_payload.get('response_id')
            or finalized_parent_payload.get('id')
            or finalized_parent_payload.get('response_id')
            or ''
        ).strip()
        durable_record = self.get_response_lookup_record(response_id) if response_id else None
        durable_payload = (
            durable_record.get('response_payload')
            if isinstance(durable_record, dict)
            and isinstance(durable_record.get('response_payload'), dict)
            else {}
        )
        if successor_execution_key and self._payload_has_graph_patch_successor_execution(
            durable_payload,
            successor_execution_key,
        ):
            self.log_unified_event(
                category='responses',
                action='graph_patch_successor_reopen',
                status='already_recorded',
                response_id=response_id,
                successor_execution_key=successor_execution_key,
                message='Skipped replay of an already persisted graph-patch successor.',
            )
            return dict(durable_payload), {
                'status': 'already_recorded',
                'skip_schedule': True,
                'successor_execution_key': successor_execution_key,
            }
        finalized_successor = self.finalize_response_frame_payload(
            successor_payload,
            request_payload=request_payload,
            persist=True,
        )
        response_id = str(
            finalized_successor.get('id')
            or finalized_successor.get('response_id')
            or ''
        ).strip()
        if not response_id:
            return finalized_parent_payload, None
        self.touch_response_lookup(
            response_id,
            status='completed',
            output_text=str(finalized_successor.get('output_text') or ''),
            response_payload=finalized_successor,
        )
        self.log_unified_event(
            category='responses',
            action='graph_patch_successor_reopen',
            status='queued',
            response_id=response_id,
            patch_id=execution.get('patch_id'),
            successor_execution_key=execution.get('successor_execution_key'),
            successor_reopen_depth=execution.get('successor_reopen_depth'),
            scheduled_branch_ids=execution.get('scheduled_branch_ids'),
            message='Queued bounded graph-patch successor through existing Late Fill.',
        )
        handoff = {
            'response_payload': finalized_successor,
            'request_payload': dict(request_payload or {}),
            'assistant_message': str(assistant_message or '').strip(),
            'artifact_gap': dict(successor_gap),
            'source_route_payload': (
                dict(source_route_payload)
                if isinstance(source_route_payload, dict)
                else None
            ),
        }
        return finalized_successor, handoff

    @staticmethod
    def _terminal_graph_patch_request_phase_graph(
        response_payload: Any,
    ) -> Optional[dict[str, Any]]:
        """Return a structurally valid graph, or ``None`` without coercion."""

        if not isinstance(response_payload, Mapping):
            return None
        runtime = response_payload.get('runtime')
        if not isinstance(runtime, Mapping):
            return None
        graph = runtime.get('request_phase_graph')
        if not isinstance(graph, Mapping):
            return None
        if 'graph_patch_lifecycle' in graph:
            lifecycle = graph.get('graph_patch_lifecycle')
            if not isinstance(lifecycle, list) or any(
                not isinstance(item, Mapping) for item in lifecycle
            ):
                return None
        return dict(graph)

    @staticmethod
    def _terminal_graph_patch_stable_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True)

    @staticmethod
    def _terminal_graph_patch_identifier(value: Any) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            return ''
        return value

    @classmethod
    def _terminal_graph_patch_response_identity(cls, response_payload: Any) -> str:
        if not isinstance(response_payload, Mapping):
            return ''
        outer_ids: list[str] = []
        for key in ('id', 'response_id'):
            if key not in response_payload:
                continue
            value = cls._terminal_graph_patch_identifier(response_payload.get(key))
            if not value:
                return ''
            outer_ids.append(value)
        if not outer_ids or any(value != outer_ids[0] for value in outer_ids[1:]):
            return ''
        frame = cls._terminal_graph_patch_frame(response_payload)
        frame_response_id = cls._terminal_graph_patch_identifier(
            frame.get('response_id')
        )
        if frame_response_id != outer_ids[0]:
            return ''
        return outer_ids[0]

    @staticmethod
    def _terminal_graph_patch_frame_sequence(value: Any) -> Optional[int]:
        return value if type(value) is int and value > 0 else None

    @classmethod
    def _terminal_graph_patch_clean_string_list(
        cls,
        value: Any,
    ) -> Optional[list[str]]:
        if not isinstance(value, list) or not value:
            return None
        cleaned: list[str] = []
        for item in value:
            text = cls._terminal_graph_patch_identifier(item)
            if not text:
                return None
            if text not in cleaned:
                cleaned.append(text)
        return sorted(cleaned)

    @classmethod
    def _terminal_graph_patch_policy_denial_record(
        cls,
        lifecycle: Any,
    ) -> dict[str, Any]:
        """Validate and sanitize one explicit apply_enforced policy denial."""

        if not isinstance(lifecycle, Mapping):
            return {}
        outcome = lifecycle.get('outcome')
        policy_review = lifecycle.get('enforced_policy_review')
        if not isinstance(outcome, Mapping) or not isinstance(policy_review, Mapping):
            return {}
        nested_policy = policy_review.get('policy')
        if not isinstance(nested_policy, Mapping):
            return {}

        status = lifecycle.get('status')
        if (
            status not in {'blocked', 'rejected'}
            or outcome.get('status') != status
            or policy_review.get('status') != status
        ):
            return {}
        if (
            lifecycle.get('kind') != 'ollmo.graph_patch_lifecycle'
            or lifecycle.get('risk_level') != 'safe_additive'
            or lifecycle.get('autonomy_level') != 'apply_enforced'
            or lifecycle.get('authority') != 'runtime_enforced_policy_denied'
            or lifecycle.get('allowed_by_policy') is not False
            or lifecycle.get('enforced_policy_id') != ENFORCED_POLICY_DEFAULT_ID
            or lifecycle.get('policy_mode') != ENFORCED_POLICY_PRODUCT_DEFAULT_MODE
        ):
            return {}

        enforced_class = cls._terminal_graph_patch_identifier(
            lifecycle.get('enforced_class')
        )
        if enforced_class not in _GRAPH_PATCH_TERMINAL_SAFE_ENFORCED_CLASSES:
            return {}
        if (
            policy_review.get('kind') != ENFORCED_POLICY_REVIEW_KIND
            or policy_review.get('policy_id') != ENFORCED_POLICY_DEFAULT_ID
            or policy_review.get('policy_mode') != ENFORCED_POLICY_PRODUCT_DEFAULT_MODE
            or policy_review.get('mode') != ENFORCED_POLICY_PRODUCT_DEFAULT_MODE
            or policy_review.get('authority') != 'runtime_enforced_policy_denied'
            or policy_review.get('allowed') is not False
            or policy_review.get('enforced_class') != enforced_class
            or policy_review.get('selected_scope_allowed') is not False
        ):
            return {}
        if (
            nested_policy.get('policy_id') != ENFORCED_POLICY_DEFAULT_ID
            or nested_policy.get('mode') != ENFORCED_POLICY_PRODUCT_DEFAULT_MODE
            or nested_policy.get('normalized') != ENFORCED_POLICY_PRODUCT_DEFAULT_MODE
            or nested_policy.get('enabled') is not True
            or nested_policy.get('default_action') != 'deny'
        ):
            return {}

        identifiers: dict[str, str] = {}
        for key in (
            'patch_id',
            'proposal_id',
            'review_id',
            'idempotency_key',
            'repair_class',
        ):
            identifiers[key] = cls._terminal_graph_patch_identifier(
                lifecycle.get(key)
            )
            if not identifiers[key]:
                return {}
        policy_review_id = cls._terminal_graph_patch_identifier(
            policy_review.get('review_id')
        )
        selected_scope = cls._terminal_graph_patch_identifier(
            policy_review.get('selected_scope')
        )
        lifecycle_reasons = cls._terminal_graph_patch_clean_string_list(
            lifecycle.get('blocked_reasons')
        )
        policy_reasons = cls._terminal_graph_patch_clean_string_list(
            policy_review.get('blocked_reasons')
        )
        if (
            not policy_review_id
            or not selected_scope
            or not lifecycle_reasons
            or not policy_reasons
            or not set(policy_reasons).issubset(lifecycle_reasons)
        ):
            return {}

        return {
            'kind': 'ollmo.graph_patch_lifecycle',
            **identifiers,
            'risk_level': 'safe_additive',
            'autonomy_level': 'apply_enforced',
            'status': status,
            'blocked_reasons': lifecycle_reasons,
            'authority': 'runtime_enforced_policy_denied',
            'enforced_class': enforced_class,
            'enforced_policy_id': ENFORCED_POLICY_DEFAULT_ID,
            'policy_mode': ENFORCED_POLICY_PRODUCT_DEFAULT_MODE,
            'allowed_by_policy': False,
            'audit_only': True,
            'enforced_policy_review': {
                'kind': ENFORCED_POLICY_REVIEW_KIND,
                'review_id': policy_review_id,
                'policy_id': ENFORCED_POLICY_DEFAULT_ID,
                'status': status,
                'allowed': False,
                'authority': 'runtime_enforced_policy_denied',
                'policy_mode': ENFORCED_POLICY_PRODUCT_DEFAULT_MODE,
                'mode': ENFORCED_POLICY_PRODUCT_DEFAULT_MODE,
                'enforced_class': enforced_class,
                'blocked_reasons': policy_reasons,
                'selected_scope': selected_scope,
                'selected_scope_allowed': False,
            },
            'outcome': {
                'status': status,
                'runtime_effect': _GRAPH_PATCH_TERMINAL_REVIEW_RUNTIME_EFFECT,
            },
        }

    @staticmethod
    def _terminal_graph_patch_frame(response_payload: Any) -> dict[str, Any]:
        if not isinstance(response_payload, Mapping) or not isinstance(response_payload.get('response_frame'), Mapping):
            return {}
        return dict(response_payload['response_frame'])

    @staticmethod
    def _terminal_graph_patch_review_relation(response_payload: Any) -> dict[str, Any]:
        response_frame = LateFillRuntimeOwner._terminal_graph_patch_frame(response_payload)
        if not isinstance(response_frame.get('frame_relation'), Mapping):
            return {}
        relation = response_frame['frame_relation']
        if str(relation.get('kind') or '').strip() != _GRAPH_PATCH_TERMINAL_REVIEW_RELATION:
            return {}
        return dict(relation)

    @classmethod
    def _terminal_graph_patch_audit_is_valid(
        cls,
        response_payload: Any,
        *,
        response_id: str,
        parent_frame_id: str,
        parent_frame_sequence: int,
        audit_key: str,
        denial_record: Mapping[str, Any],
        expected_graph: Mapping[str, Any],
    ) -> bool:
        """Validate the complete durable audit and its current lineage."""

        try:
            if cls._terminal_graph_patch_response_identity(response_payload) != response_id:
                return False
            frame = cls._terminal_graph_patch_frame(response_payload)
            frame_sequence = cls._terminal_graph_patch_frame_sequence(
                frame.get('frame_sequence')
            )
            frame_id = cls._terminal_graph_patch_identifier(frame.get('frame_id'))
            if (
                not frame_id
                or frame_id == parent_frame_id
                or frame_sequence != parent_frame_sequence + 1
            ):
                return False
            relation = cls._terminal_graph_patch_review_relation(response_payload)
            if not relation or not set(relation).issubset(
                _GRAPH_PATCH_TERMINAL_REVIEW_RELATION_KEYS
            ):
                return False
            expected_relation = {
                'kind': _GRAPH_PATCH_TERMINAL_REVIEW_RELATION,
                'reason': _GRAPH_PATCH_TERMINAL_REVIEW_REASON,
                'response_id': response_id,
                'parent_response_id': response_id,
                'parent_frame_id': parent_frame_id,
                'parent_frame_sequence': parent_frame_sequence,
                'audit_key': audit_key,
                'audit_only': True,
                'executable': False,
                'runtime_effect': _GRAPH_PATCH_TERMINAL_REVIEW_RUNTIME_EFFECT,
                'owed_work': 'none',
                'patch_id': denial_record.get('patch_id'),
                'proposal_id': denial_record.get('proposal_id'),
                'review_id': denial_record.get('review_id'),
                'policy_review_id': (
                    denial_record.get('enforced_policy_review', {}).get('review_id')
                    if isinstance(denial_record.get('enforced_policy_review'), Mapping)
                    else None
                ),
            }
            if any(relation.get(key) != value for key, value in expected_relation.items()):
                return False
            if 'scheduled_branch_ids' in relation and relation.get('scheduled_branch_ids') != []:
                return False
            graph = cls._terminal_graph_patch_request_phase_graph(response_payload)
            if graph is None or dict(graph) != dict(expected_graph):
                return False
            lifecycle = graph.get('graph_patch_lifecycle') or []
            matches = [
                item
                for item in lifecycle
                if isinstance(item, Mapping)
                and item.get('patch_id') == denial_record.get('patch_id')
            ]
            return len(matches) == 1 and dict(matches[0]) == dict(denial_record)
        except Exception:  # noqa: BLE001 - malformed durable state fails closed
            return False

    def _load_durable_terminal_graph_patch_payload(self, response_id: str) -> dict[str, Any]:
        if not response_id or not callable(self.load_latest_response_state):
            return {}
        try:
            durable_state = self.load_latest_response_state(response_id)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                'Could not load durable terminal graph-patch audit parent: %s',
                exc,
            )
            return {}
        if not isinstance(durable_state, Mapping) or durable_state.get('ok') is not True or durable_state.get('errors'):
            return {}
        durable_payload = durable_state.get('response_payload')
        return dict(durable_payload) if isinstance(durable_payload, Mapping) else {}

    def _persist_terminal_graph_patch_review_audit(
        self,
        finalized_parent_payload: dict[str, Any],
        *,
        prepared: Mapping[str, Any],
        request_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], None]:
        """Append denied terminal patch policy truth without creating executable work."""

        if (
            prepared.get('status') != 'not_applicable'
            or prepared.get('reason') != 'no_safe_terminal_graph_patch_successor'
        ):
            return finalized_parent_payload, None
        reviewed_payload = (
            prepared.get('response_payload')
            if isinstance(prepared.get('response_payload'), Mapping)
            else None
        )
        if reviewed_payload is None:
            return finalized_parent_payload, None

        parent_graph = self._terminal_graph_patch_request_phase_graph(finalized_parent_payload)
        reviewed_graph = self._terminal_graph_patch_request_phase_graph(reviewed_payload)
        if parent_graph is None or reviewed_graph is None:
            return finalized_parent_payload, None

        parent_frame = self._terminal_graph_patch_frame(finalized_parent_payload)
        reviewed_frame = self._terminal_graph_patch_frame(reviewed_payload)
        response_id = self._terminal_graph_patch_response_identity(
            finalized_parent_payload
        )
        reviewed_response_id = self._terminal_graph_patch_response_identity(
            reviewed_payload
        )
        parent_frame_id = self._terminal_graph_patch_identifier(
            parent_frame.get('frame_id')
        )
        reviewed_frame_id = self._terminal_graph_patch_identifier(
            reviewed_frame.get('frame_id')
        )
        parent_frame_sequence = self._terminal_graph_patch_frame_sequence(
            parent_frame.get('frame_sequence')
        )
        reviewed_frame_sequence = self._terminal_graph_patch_frame_sequence(
            reviewed_frame.get('frame_sequence')
        )
        if (
            not response_id
            or reviewed_response_id != response_id
            or not parent_frame_id
            or parent_frame_sequence is None
            or reviewed_frame_id != parent_frame_id
            or reviewed_frame_sequence != parent_frame_sequence
        ):
            return finalized_parent_payload, None

        reviewed_denials = [
            denial
            for lifecycle in reviewed_graph.get('graph_patch_lifecycle') or []
            if (
                denial := self._terminal_graph_patch_policy_denial_record(
                    lifecycle
                )
            )
        ]
        if len(reviewed_denials) != 1:
            return finalized_parent_payload, None
        denial_record = reviewed_denials[0]
        parent_lifecycles = [
            copy.deepcopy(dict(item))
            for item in parent_graph.get('graph_patch_lifecycle') or []
        ]
        parent_patch_collisions = [
            item
            for item in parent_lifecycles
            if item.get('patch_id') == denial_record.get('patch_id')
        ]
        if parent_patch_collisions:
            return finalized_parent_payload, None

        audit_graph = copy.deepcopy(parent_graph)
        audit_graph['graph_patch_lifecycle'] = [
            *parent_lifecycles,
            copy.deepcopy(denial_record),
        ]
        audit_digest = hashlib.sha256(
            self._terminal_graph_patch_stable_json(
                {
                    'response_id': response_id,
                    'parent_frame_id': parent_frame_id,
                    'parent_frame_sequence': parent_frame_sequence,
                    'policy_denial': denial_record,
                }
            ).encode('utf-8')
        ).hexdigest()
        audit_key = f'graph-patch-terminal-review-{audit_digest[:24]}'
        audit_payload = copy.deepcopy(finalized_parent_payload)
        audit_runtime = copy.deepcopy(audit_payload.get('runtime'))
        if not isinstance(audit_runtime, dict):
            return finalized_parent_payload, None
        audit_runtime['request_phase_graph'] = audit_graph
        audit_payload['runtime'] = audit_runtime
        audit_payload['frame_relation'] = {
            'kind': _GRAPH_PATCH_TERMINAL_REVIEW_RELATION,
            'reason': _GRAPH_PATCH_TERMINAL_REVIEW_REASON,
            'response_id': response_id,
            'parent_response_id': response_id,
            'parent_frame_id': parent_frame_id,
            'parent_frame_sequence': parent_frame_sequence,
            'audit_key': audit_key,
            'audit_only': True,
            'executable': False,
            'runtime_effect': _GRAPH_PATCH_TERMINAL_REVIEW_RUNTIME_EFFECT,
            'owed_work': 'none',
            'scheduled_branch_ids': [],
            'patch_id': denial_record['patch_id'],
            'proposal_id': denial_record['proposal_id'],
            'review_id': denial_record['review_id'],
            'policy_review_id': denial_record['enforced_policy_review']['review_id'],
        }

        with _GRAPH_PATCH_TERMINAL_REVIEW_HANDOFF_LOCK:
            durable_payload = self._load_durable_terminal_graph_patch_payload(response_id)
            if not durable_payload:
                return finalized_parent_payload, None
            if self._terminal_graph_patch_audit_is_valid(
                durable_payload,
                response_id=response_id,
                parent_frame_id=parent_frame_id,
                parent_frame_sequence=parent_frame_sequence,
                audit_key=audit_key,
                denial_record=denial_record,
                expected_graph=audit_graph,
            ):
                return durable_payload, None
            durable_frame = self._terminal_graph_patch_frame(durable_payload)
            if (
                self._terminal_graph_patch_response_identity(durable_payload)
                != response_id
                or self._terminal_graph_patch_identifier(
                    durable_frame.get('frame_id')
                ) != parent_frame_id
                or self._terminal_graph_patch_frame_sequence(
                    durable_frame.get('frame_sequence')
                ) != parent_frame_sequence
                or self._terminal_graph_patch_request_phase_graph(durable_payload)
                != parent_graph
            ):
                return finalized_parent_payload, None
            try:
                finalized_audit = self.finalize_response_frame_payload(
                    audit_payload,
                    request_payload=request_payload,
                    persist=True,
                    expected_parent_frame_id=parent_frame_id,
                    expected_parent_frame_sequence=parent_frame_sequence,
                )
            except ResponseFrameParentCASMismatch:
                durable_payload = self._load_durable_terminal_graph_patch_payload(response_id)
                if self._terminal_graph_patch_audit_is_valid(
                    durable_payload,
                    response_id=response_id,
                    parent_frame_id=parent_frame_id,
                    parent_frame_sequence=parent_frame_sequence,
                    audit_key=audit_key,
                    denial_record=denial_record,
                    expected_graph=audit_graph,
                ):
                    return durable_payload, None
                return finalized_parent_payload, None
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    'Could not persist terminal graph-patch review audit: %s',
                    exc,
                )
                return finalized_parent_payload, None

            durable_audit = self._load_durable_terminal_graph_patch_payload(response_id)
            if not self._terminal_graph_patch_audit_is_valid(
                durable_audit,
                response_id=response_id,
                parent_frame_id=parent_frame_id,
                parent_frame_sequence=parent_frame_sequence,
                audit_key=audit_key,
                denial_record=denial_record,
                expected_graph=audit_graph,
            ):
                return finalized_parent_payload, None
            finalized_frame_id = self._terminal_graph_patch_identifier(
                self._terminal_graph_patch_frame(finalized_audit).get('frame_id')
            )
            durable_frame_id = self._terminal_graph_patch_identifier(
                self._terminal_graph_patch_frame(durable_audit).get('frame_id')
            )
            if not finalized_frame_id or durable_frame_id != finalized_frame_id:
                return finalized_parent_payload, None
            finalized_audit = durable_audit
            durable_relation = self._terminal_graph_patch_review_relation(
                finalized_audit
            )

            self.touch_response_lookup(
                response_id,
                status='completed',
                output_text=str(finalized_audit.get('output_text') or ''),
                response_payload=finalized_audit,
            )
            self.log_unified_event(
                category='responses',
                action='graph_patch_terminal_review',
                status='recorded',
                response_id=response_id,
                audit_key=audit_key,
                parent_frame_id=parent_frame_id,
                parent_frame_sequence=parent_frame_sequence,
                patch_id=durable_relation.get('patch_id'),
                proposal_id=durable_relation.get('proposal_id'),
                runtime_effect=_GRAPH_PATCH_TERMINAL_REVIEW_RUNTIME_EFFECT,
                message=(
                    'Recorded blocked terminal graph-patch policy truth without '
                    'creating executable work.'
                ),
            )
            return finalized_audit, None

    @staticmethod
    def _payload_has_graph_patch_successor_execution(
        response_payload: Any,
        successor_execution_key: str,
    ) -> bool:
        """Read durable response truth without treating an inert candidate as consumed."""

        if not isinstance(response_payload, Mapping) or not successor_execution_key:
            return False
        late_fill = (
            response_payload.get('late_fill')
            if isinstance(response_payload.get('late_fill'), Mapping)
            else {}
        )
        execution = (
            late_fill.get('successor_reopen_execution')
            if isinstance(late_fill.get('successor_reopen_execution'), Mapping)
            else {}
        )
        if str(execution.get('successor_execution_key') or '').strip() == successor_execution_key:
            return True
        runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), Mapping)
            else {}
        )
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        for record in graph.get('successor_reopen_executions') or []:
            if (
                isinstance(record, Mapping)
                and str(record.get('successor_execution_key') or '').strip()
                == successor_execution_key
            ):
                return True
        for record in graph.get('successor_reopen_requests') or []:
            if not isinstance(record, Mapping):
                continue
            record_execution = (
                record.get('execution')
                if isinstance(record.get('execution'), Mapping)
                else {}
            )
            if str(record_execution.get('successor_execution_key') or '').strip() == successor_execution_key:
                return True
            if (
                str(record.get('status') or '').strip().lower() == 'applied_to_successor'
                and str(record.get('successor_execution_key') or '').strip()
                == successor_execution_key
            ):
                return True
        return False

    @staticmethod
    def _payload_has_partial_graph_rebase_execution(
        response_payload: Any,
        execution_key: str,
    ) -> bool:
        if not isinstance(response_payload, Mapping) or not execution_key:
            return False
        late_fill = (
            response_payload.get('late_fill')
            if isinstance(response_payload.get('late_fill'), Mapping)
            else {}
        )
        execution = (
            late_fill.get('partial_rebase_execution')
            if isinstance(late_fill.get('partial_rebase_execution'), Mapping)
            else {}
        )
        if str(execution.get('execution_key') or '').strip() == execution_key:
            return True
        runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), Mapping)
            else {}
        )
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        return any(
            isinstance(item, Mapping)
            and str(item.get('execution_key') or '').strip() == execution_key
            for item in graph.get('successor_rebase_executions') or []
        )

    def persist_and_schedule_partial_graph_rebase_successor(
        self,
        prepared: Mapping[str, Any],
        *,
        source_route_payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Durably append one reviewed partial successor before scheduling it."""

        prepared_payload = dict(prepared) if isinstance(prepared, Mapping) else {}
        if str(prepared_payload.get('status') or '').strip().lower() != 'queued':
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_successor_not_prepared'],
            }
        successor_payload = (
            dict(prepared_payload.get('response_payload'))
            if isinstance(prepared_payload.get('response_payload'), Mapping)
            else {}
        )
        artifact_gap = (
            dict(prepared_payload.get('artifact_gap'))
            if isinstance(prepared_payload.get('artifact_gap'), Mapping)
            else {}
        )
        execution = (
            dict(prepared_payload.get('execution'))
            if isinstance(prepared_payload.get('execution'), Mapping)
            else {}
        )
        response_id = str(
            successor_payload.get('response_id')
            or successor_payload.get('id')
            or execution.get('response_id')
            or ''
        ).strip()
        execution_key = str(execution.get('execution_key') or '').strip()
        parent_frame_id = str(execution.get('parent_frame_id') or '').strip()
        parent_frame_sequence = execution.get('parent_frame_sequence')
        if (
            not response_id
            or not execution_key
            or not parent_frame_id
            or isinstance(parent_frame_sequence, bool)
            or parent_frame_sequence in (None, '')
        ):
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_successor_binding_incomplete'],
            }
        if not artifact_gap or str(artifact_gap.get('trigger') or '').strip() != (
            'graph_rebase_partial_successor'
        ):
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_successor_gap_invalid'],
            }
        if not callable(self.load_latest_response_state):
            return {
                'status': 'blocked',
                'blocked_reasons': ['durable_partial_rebase_verification_unavailable'],
            }

        with _PARTIAL_GRAPH_REBASE_HANDOFF_LOCK:
            durable_state = self.load_latest_response_state(response_id)
            durable_payload = (
                durable_state.get('response_payload')
                if isinstance(durable_state, Mapping)
                and isinstance(durable_state.get('response_payload'), Mapping)
                else {}
            )
            if (
                not isinstance(durable_state, Mapping)
                or durable_state.get('ok') is not True
                or not durable_payload
                or (
                    isinstance(durable_state.get('errors'), list)
                    and durable_state.get('errors')
                )
            ):
                return {
                    'status': 'blocked',
                    'blocked_reasons': [
                        'durable_partial_rebase_verification_unavailable'
                    ],
                }
            if self._payload_has_partial_graph_rebase_execution(
                durable_payload,
                execution_key,
            ):
                return {
                    'status': 'already_recorded',
                    'execution_key': execution_key,
                    'response_payload': dict(durable_payload),
                    'scheduled': False,
                }
            durable_parent_frame = (
                durable_state.get('response_frame')
                if isinstance(durable_state.get('response_frame'), Mapping)
                else durable_payload.get('response_frame')
                if isinstance(durable_payload.get('response_frame'), Mapping)
                else {}
            )
            durable_parent_frame_id = str(
                durable_parent_frame.get('frame_id') or ''
            ).strip()
            durable_parent_frame_sequence = durable_parent_frame.get('frame_sequence')
            if (
                durable_parent_frame_id != parent_frame_id
                or (
                    parent_frame_sequence is not None
                    and durable_parent_frame_sequence != parent_frame_sequence
                )
            ):
                return {
                    'status': 'blocked',
                    'blocked_reasons': [RESPONSE_FRAME_STALE_PARENT_REASON],
                    'expected_parent_frame_id': parent_frame_id,
                    'current_parent_frame_id': durable_parent_frame_id or None,
                    'expected_parent_frame_sequence': parent_frame_sequence,
                    'current_parent_frame_sequence': durable_parent_frame_sequence,
                }

            durable_request: dict[str, Any] = {}
            if callable(self.load_latest_response_observation_state):
                observation_state = self.load_latest_response_observation_state(
                    response_id
                )
                observation_frame = (
                    observation_state.get('response_frame')
                    if isinstance(observation_state, Mapping)
                    and isinstance(observation_state.get('response_frame'), Mapping)
                    else {}
                )
                observation_payload = (
                    observation_state.get('response_payload')
                    if isinstance(observation_state, Mapping)
                    and isinstance(observation_state.get('response_payload'), Mapping)
                    else {}
                )
                if (
                    not isinstance(observation_state, Mapping)
                    or observation_state.get('ok') is not True
                    or str(observation_frame.get('frame_id') or '').strip()
                    != durable_parent_frame_id
                    or observation_frame.get('frame_sequence')
                    != durable_parent_frame_sequence
                ):
                    return {
                        'status': 'blocked',
                        'blocked_reasons': [
                            'durable_partial_rebase_root_truth_unavailable'
                        ],
                    }
                if isinstance(observation_payload.get('request'), Mapping):
                    durable_request = dict(observation_payload.get('request') or {})
            if not durable_request and isinstance(
                durable_parent_frame.get('request'), Mapping
            ):
                durable_request = dict(durable_parent_frame.get('request') or {})
            if not durable_request and isinstance(durable_payload.get('request'), Mapping):
                durable_request = dict(durable_payload.get('request') or {})
            current_root_prompt = str(
                self.extract_responses_prompt(durable_request)
                or durable_request.get('prompt')
                or ''
            ).strip()
            execution_contract = (
                artifact_gap.get('execution_contract')
                if isinstance(artifact_gap.get('execution_contract'), Mapping)
                else {}
            )
            forbidden_root_prompt_digest = str(
                execution_contract.get('forbidden_root_prompt_digest') or ''
            ).strip()
            if not current_root_prompt:
                return {
                    'status': 'blocked',
                    'blocked_reasons': [
                        'partial_rebase_current_root_prompt_truth_unavailable'
                    ],
                }
            if not forbidden_root_prompt_digest:
                return {
                    'status': 'blocked',
                    'blocked_reasons': [
                        'partial_rebase_root_prompt_guard_missing'
                    ],
                }
            if (
                stable_graph_rebase_prompt_digest(current_root_prompt)
                != forbidden_root_prompt_digest
            ):
                return {
                    'status': 'blocked',
                    'blocked_reasons': [
                        'partial_rebase_root_prompt_guard_mismatch'
                    ],
                }
            successor_request_payload = {'prompt': current_root_prompt}
            durable_request_meta = (
                durable_request.get('request_meta')
                if isinstance(durable_request.get('request_meta'), Mapping)
                else {}
            )
            if durable_request_meta:
                successor_request_payload['request_meta'] = dict(
                    durable_request_meta
                )
                reasoning_provenance = (
                    durable_request_meta.get('reasoning_effort_control')
                    if isinstance(
                        durable_request_meta.get('reasoning_effort_control'),
                        Mapping,
                    )
                    else {}
                )
                if reasoning_provenance.get('value') not in (None, ''):
                    successor_request_payload['reasoning_effort'] = (
                        reasoning_provenance.get('value')
                    )

            try:
                finalized_successor = self.finalize_response_frame_payload(
                    successor_payload,
                    request_payload=successor_request_payload,
                    persist=True,
                    expected_parent_frame_id=parent_frame_id,
                    expected_parent_frame_sequence=parent_frame_sequence,
                )
            except ResponseFrameParentCASMismatch as exc:
                return {
                    'status': 'blocked',
                    'blocked_reasons': [exc.code],
                    **exc.as_dict(),
                }
            finalized_frame = (
                finalized_successor.get('response_frame')
                if isinstance(finalized_successor.get('response_frame'), Mapping)
                else {}
            )
            finalized_frame_id = str(finalized_frame.get('frame_id') or '').strip()
            relation = (
                finalized_frame.get('frame_relation')
                if isinstance(finalized_frame.get('frame_relation'), Mapping)
                else {}
            )
            if (
                not finalized_frame_id
                or finalized_frame_id == parent_frame_id
                or str(relation.get('kind') or '').strip()
                != 'graph_rebase_partial_successor'
                or str(relation.get('parent_frame_id') or '').strip()
                != parent_frame_id
                or str(relation.get('execution_key') or '').strip()
                != execution_key
            ):
                return {
                    'status': 'blocked',
                    'blocked_reasons': ['partial_rebase_successor_frame_relation_invalid'],
                }

            durable_state = self.load_latest_response_state(response_id)
            durable_successor = (
                durable_state.get('response_payload')
                if isinstance(durable_state, Mapping)
                and isinstance(durable_state.get('response_payload'), Mapping)
                else {}
            )
            durable_frame = (
                durable_state.get('response_frame')
                if isinstance(durable_state, Mapping)
                and isinstance(durable_state.get('response_frame'), Mapping)
                else {}
            )
            if (
                durable_state.get('ok') is not True
                or str(durable_frame.get('frame_id') or '').strip() != finalized_frame_id
                or not self._payload_has_partial_graph_rebase_execution(
                    durable_successor,
                    execution_key,
                )
            ):
                return {
                    'status': 'blocked',
                    'blocked_reasons': ['partial_rebase_successor_not_durable'],
                    'execution_key': execution_key,
                }

            self.touch_response_lookup(
                response_id,
                status='completed',
                output_text=str(finalized_successor.get('output_text') or ''),
                response_payload=finalized_successor,
            )
            scheduled = self.schedule_response_late_fill(
                response_payload=finalized_successor,
                request_payload=successor_request_payload,
                assistant_message='',
                artifact_gap=artifact_gap,
                source_route_payload=(
                    dict(source_route_payload)
                    if isinstance(source_route_payload, Mapping)
                    else dict(durable_payload.get('route_payload') or {})
                    if isinstance(durable_payload.get('route_payload'), Mapping)
                    else None
                ),
            )
            status = 'queued' if scheduled else 'durable_pending'
            self.log_unified_event(
                category='responses',
                action='graph_rebase_partial_successor',
                status=status,
                response_id=response_id,
                proposal_id=execution.get('proposal_id'),
                authorization_record_id=execution.get('authorization_record_id'),
                execution_key=execution_key,
                scheduled_branch_ids=execution.get('scheduled_branch_ids'),
                message=(
                    'Persisted exact partial graph-rebase successor before scheduling '
                    'its branch-local Late Fill wave.'
                ),
            )
            return {
                'status': status,
                'execution_key': execution_key,
                'execution': execution,
                'response_payload': finalized_successor,
                'scheduled': bool(scheduled),
            }

    @classmethod
    def _durable_graph_patch_successor_descends_from_parent(
        cls,
        durable_payload: Any,
        parent_payload: Any,
    ) -> bool:
        if not isinstance(durable_payload, Mapping) or not isinstance(parent_payload, Mapping):
            return False
        parent_frame = (
            parent_payload.get('response_frame')
            if isinstance(parent_payload.get('response_frame'), Mapping)
            else {}
        )
        parent_frame_id = str(parent_frame.get('frame_id') or '').strip()
        if not parent_frame_id:
            return False
        durable_frame = (
            durable_payload.get('response_frame')
            if isinstance(durable_payload.get('response_frame'), Mapping)
            else {}
        )
        durable_relation = (
            durable_frame.get('frame_relation')
            if isinstance(durable_frame.get('frame_relation'), Mapping)
            else {}
        )
        if (
            str(durable_relation.get('kind') or '').strip() == 'graph_patch_reopen_successor'
            and str(durable_relation.get('parent_frame_id') or '').strip() == parent_frame_id
        ):
            execution_key = str(durable_relation.get('successor_execution_key') or '').strip()
            if execution_key and cls._payload_has_graph_patch_successor_execution(
                durable_payload,
                execution_key,
            ):
                return True
        runtime = (
            durable_payload.get('runtime')
            if isinstance(durable_payload.get('runtime'), Mapping)
            else {}
        )
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), Mapping)
            else {}
        )
        for request in graph.get('successor_reopen_requests') or []:
            if not isinstance(request, Mapping):
                continue
            if str(request.get('parent_frame_id') or '').strip() != parent_frame_id:
                continue
            execution = (
                request.get('execution')
                if isinstance(request.get('execution'), Mapping)
                else {}
            )
            execution_key = str(
                request.get('successor_execution_key')
                or execution.get('successor_execution_key')
                or ''
            ).strip()
            if execution_key and cls._payload_has_graph_patch_successor_execution(
                durable_payload,
                execution_key,
            ):
                return True
        return False

    def complete_response_late_fill(
        self,
        *,
        response_payload: dict[str, Any],
        request_payload: dict[str, Any],
        assistant_message: str,
        artifact_gap: dict[str, Any],
        source_route_payload: Optional[dict[str, Any]],
        build_deferred_follow_up_gap_for_capability: Optional[Callable[..., dict[str, Any]]] = None,
        prepare_late_fill_branch_plan: Optional[Callable[..., dict[str, Any]]] = None,
        execute_prepared_late_fill_branch: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        build_gap = build_deferred_follow_up_gap_for_capability or self.build_deferred_follow_up_gap_for_capability
        using_default_prepare_plan = prepare_late_fill_branch_plan is None
        prepare_plan = prepare_late_fill_branch_plan or self.prepare_late_fill_branch_plan
        execute_plan = execute_prepared_late_fill_branch or self.execute_prepared_late_fill_branch

        response_id = str((response_payload or {}).get('id') or '').strip()
        if not response_id:
            return
        successor_handoff: Optional[dict[str, Any]] = None
        completed_capabilities: list[str] = []
        failed_capabilities: list[str] = []
        cancelled_capabilities: list[str] = []
        completed_branches: list[str] = []
        failed_branches: list[str] = []
        cancelled_branches: list[str] = []
        try:
            existing_record = self.get_response_lookup_record(response_id) or {}
            recovered_payload = (
                existing_record.get('response_payload')
                if isinstance(existing_record.get('response_payload'), Mapping)
                else {}
            )
            explicit_payload = response_payload if isinstance(response_payload, Mapping) else {}
            if self._durable_graph_patch_successor_descends_from_parent(
                recovered_payload,
                explicit_payload,
            ):
                self.log_unified_event(
                    category='responses',
                    action='graph_patch_successor_reopen',
                    status='stale_parent_ignored',
                    response_id=response_id,
                    message='Ignored stale terminal-parent replay after durable successor persistence.',
                )
                return
            current_payload = (
                dict(explicit_payload)
                if self._prefer_explicit_late_fill_seed_payload(explicit_payload, recovered_payload)
                else dict(recovered_payload or explicit_payload or {})
            )
            source_route_payload = (
                self._source_route_with_response_external_runtime(
                    source_route_payload,
                    current_payload,
                )
            )
            current_late_fill = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), dict) else {}
            legacy_completed_capabilities = self.normalize_capability_list(current_late_fill.get('completed_capabilities'))
            legacy_failed_capabilities = self.normalize_capability_list(current_late_fill.get('failed_capabilities'))
            legacy_cancelled_capabilities = self.normalize_capability_list(current_late_fill.get('cancelled_capabilities'))
            completed_branch_records = self.normalize_late_fill_branches(current_late_fill.get('completed_branches'))
            failed_branch_records = self.normalize_late_fill_branches(current_late_fill.get('failed_branches'))
            cancelled_branch_records = self.normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
            completed_branches = [self.branch_id(item) for item in completed_branch_records if self.branch_id(item)]
            failed_branches = [self.branch_id(item) for item in failed_branch_records if self.branch_id(item)]
            cancelled_branches = [self.branch_id(item) for item in cancelled_branch_records if self.branch_id(item)]
            extracted_pending_branches = self.extract_pending_deferred_branches(
                route_payload=source_route_payload,
                artifact_payload=current_payload,
            )
            gap_pending_branches = self.normalize_late_fill_branches(
                artifact_gap.get('pending_branches')
            )
            if gap_pending_branches and (
                artifact_gap.get('authoritative_pending_branches') is True
                or len(gap_pending_branches) > len(extracted_pending_branches)
                or artifact_gap.get('prepared_image_prompt_branch_expansion') not in (None, '', [], {})
            ):
                pending_branch_seed = gap_pending_branches
            else:
                pending_branch_seed = extracted_pending_branches
            pending_branches = self.build_pending_late_fill_branches(
                artifact_gap=artifact_gap,
                late_fill_state=current_late_fill,
                pending_branches=pending_branch_seed,
            )
            pending_capabilities = self.normalize_capability_list(artifact_gap.get('pending_capabilities'))
            if not pending_capabilities:
                pending_capabilities = self.normalize_capability_list(
                    [self.branch_capability(item) for item in pending_branches if self.branch_capability(item)]
                )
            if not pending_branches and not pending_capabilities:
                pending_capabilities = self.extract_pending_deferred_capabilities(
                    route_payload=source_route_payload,
                    artifact_payload=current_payload,
                )
            if not pending_branches and pending_capabilities:
                pending_branches = self.build_pending_late_fill_branches(
                    artifact_gap=artifact_gap,
                    late_fill_state=current_late_fill,
                    pending_capabilities=pending_capabilities,
                )
            if not pending_branches and not pending_capabilities:
                expected_capability = self.normalize_capability(artifact_gap.get('expected_capability'))
                if expected_capability:
                    pending_branches = self.build_pending_late_fill_branches(
                        artifact_gap=artifact_gap,
                        late_fill_state=current_late_fill,
                        pending_capabilities=[expected_capability],
                    )
            def _remember_branch(
                records: list[dict[str, Any]],
                record_ids: list[str],
                branch: Mapping[str, Any],
                *,
                status: Optional[str] = None,
                error: Optional[Mapping[str, Any]] = None,
                attempt: Optional[Mapping[str, Any]] = None,
                recovery_context: Optional[Mapping[str, Any]] = None,
                recovery_state: Optional[Mapping[str, Any]] = None,
            ) -> None:
                branch_id = self.branch_id(branch)
                capability = self.branch_capability(branch)
                if not branch_id or not capability:
                    return
                normalized_branch = dict(branch)
                normalized_branch['branch_id'] = branch_id
                normalized_branch['phase_id'] = str(branch.get('phase_id') or branch_id).strip() or branch_id
                normalized_branch['capability'] = capability
                if status:
                    normalized_branch['status'] = str(status).strip().lower()
                if error:
                    normalized_branch['error'] = dict(error)
                if attempt:
                    normalized_branch['attempt'] = dict(attempt)
                if recovery_context:
                    normalized_branch['recovery_context'] = dict(recovery_context)
                if recovery_state:
                    normalized_branch['recovery_state'] = dict(recovery_state)
                for index, existing in enumerate(records):
                    if self.branch_id(existing) != branch_id:
                        continue
                    records[index] = normalized_branch
                    break
                else:
                    records.append(normalized_branch)
                if branch_id not in record_ids:
                    record_ids.append(branch_id)

            def _final_lookup_status_for_late_fill(final_status: str) -> str:
                normalized = str(final_status or '').strip().lower()
                if normalized == 'partial_failed':
                    return 'incomplete'
                if normalized == 'failed':
                    return 'failed'
                return 'completed'

            def _status_for_terminal_state(
                active_pending_branches: list[dict[str, Any]],
            ) -> str:
                if active_pending_branches:
                    return 'running'
                if failed_branch_records and completed_branch_records:
                    return 'partial_failed'
                if failed_branch_records or failed_capabilities:
                    return 'failed'
                if cancelled_branch_records and completed_branch_records:
                    return 'partial_cancelled'
                if cancelled_branch_records or cancelled_capabilities:
                    return 'cancelled'
                return 'completed'

            completed_branch_records = [
                dict(item)
                for item in completed_branch_records
                if self.branch_id(item) not in failed_branches
                and self.branch_id(item) not in cancelled_branches
            ]

            def _capability_lists_for_branch_state(active_pending_branches: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
                pending_caps = self.normalize_capability_list(
                    [self.branch_capability(item) for item in active_pending_branches if self.branch_capability(item)]
                )
                pending_capability_set = set(pending_caps)
                completed_caps = [
                    capability
                    for capability in legacy_completed_capabilities
                    if capability not in pending_capability_set
                ]
                failed_caps = [
                    capability
                    for capability in legacy_failed_capabilities
                    if capability not in completed_caps and capability not in pending_capability_set
                ]
                for branch in completed_branch_records:
                    capability = self.branch_capability(branch)
                    if not capability or capability in pending_capability_set or capability in completed_caps:
                        continue
                    completed_caps.append(capability)
                for branch in failed_branch_records:
                    capability = self.branch_capability(branch)
                    if (
                        not capability
                        or capability in pending_capability_set
                        or capability in completed_caps
                        or capability in failed_caps
                    ):
                        continue
                    failed_caps.append(capability)
                return pending_caps, completed_caps, failed_caps

            def _cancelled_capability_list() -> list[str]:
                values = [
                    *legacy_cancelled_capabilities,
                    *[
                        self.branch_capability(branch)
                        for branch in cancelled_branch_records
                        if self.branch_capability(branch)
                    ],
                ]
                return self.normalize_capability_list(values)

            duplicate_capability_counts = self.late_fill_capability_counts(pending_branches)

            pruned_pending: list[dict[str, Any]] = []
            for branch in pending_branches:
                branch_id = self.branch_id(branch)
                candidate = self.branch_capability(branch)
                if not candidate:
                    continue
                if not branch_id and (
                    candidate in legacy_completed_capabilities
                    or candidate in legacy_failed_capabilities
                    or candidate in legacy_cancelled_capabilities
                ):
                    continue
                if branch_id and (branch_id in completed_branches or branch_id in failed_branches or branch_id in cancelled_branches):
                    continue
                candidate_gap = build_gap(
                    artifact_gap,
                    capability=candidate,
                    artifact_payload=current_payload,
                )
                branch_already_fulfilled = False
                if self._branch_is_required_text_artifact(branch):
                    branch_already_fulfilled = self._text_artifact_branch_has_canonical_evidence(
                        branch,
                        current_payload,
                    )
                else:
                    branch_already_fulfilled = self.artifact_gap_is_already_fulfilled(
                        candidate_gap,
                        current_payload,
                    )
                if (
                    duplicate_capability_counts.get(candidate, 0) <= 1
                    and branch_already_fulfilled
                ):
                    _remember_branch(completed_branch_records, completed_branches, branch, status='fulfilled')
                    continue
                pruned_pending.append(branch)
            pending_branches = pruned_pending
            pending_capabilities, completed_capabilities, failed_capabilities = _capability_lists_for_branch_state(
                pending_branches
            )
            cancelled_capabilities = _cancelled_capability_list()

            if not pending_branches:
                terminal_status = _status_for_terminal_state(pending_branches)
                skipped_late_fill = self.build_late_fill_state(
                    build_gap(
                        artifact_gap,
                        capability=self.normalize_capability(artifact_gap.get('expected_capability')),
                        artifact_payload=current_payload,
                        pending_capabilities=[],
                        completed_capabilities=completed_capabilities,
                        failed_capabilities=failed_capabilities,
                    ),
                    status=terminal_status if terminal_status != 'completed' else 'skipped',
                    prior_state=current_late_fill,
                    extra={
                        'route_source': 'already_fulfilled',
                        'route_reason': 'all requested downstream artifacts were already present before late fill started'
                        if not cancelled_branch_records
                        else 'all remaining downstream branches were cancelled, waived, or superseded before execution',
                        'skip_kind': 'no_work_needed' if not cancelled_branch_records else 'semantic_execution_gate',
                        'skip_reason': 'already_fulfilled' if not cancelled_branch_records else 'terminal_branch_state',
                        'skip_source': 'late_fill_prune',
                        'pending_capabilities': [],
                        'completed_capabilities': completed_capabilities,
                        'failed_capabilities': failed_capabilities,
                        'cancelled_capabilities': cancelled_capabilities,
                        'pending_branches': [],
                        'completed_branches': completed_branch_records,
                        'failed_branches': failed_branch_records,
                        'cancelled_branches': cancelled_branch_records,
                    },
                )
                payload_for_finalize, effective_terminal_status = self.finalize_terminal_materialization_contract(
                    self.attach_late_fill_state(current_payload, skipped_late_fill),
                    request_payload=request_payload,
                    route_payload=source_route_payload,
                    artifact_gap=artifact_gap,
                    terminal_status=str(skipped_late_fill.get('status') or terminal_status or 'skipped'),
                )
                finalized_payload = self.finalize_response_frame_payload(
                    payload_for_finalize,
                    request_payload=request_payload,
                    persist=True,
                )
                self.touch_response_lookup(
                    response_id,
                    status=_final_lookup_status_for_late_fill(effective_terminal_status),
                    output_text=str(finalized_payload.get('output_text') or ''),
                    response_payload=finalized_payload,
                )
                finalized_payload, successor_handoff = (
                    self._prepare_terminal_graph_patch_successor_handoff(
                        finalized_payload,
                        request_payload=request_payload,
                        assistant_message=assistant_message,
                        artifact_gap=artifact_gap,
                        source_route_payload=source_route_payload,
                    )
                )
                if successor_handoff is None:
                    self.schedule_terminal_substrate_hygiene(
                        finalized_payload,
                        route_payload=source_route_payload,
                        reason='late_fill_already_fulfilled',
                    )
                return

            self.ensure_response_lookup_for_payload(current_payload, mode_hint='chat', route_payload=source_route_payload)
            request_phase_graph = self.request_phase_graph_for_late_fill(
                route_payload=source_route_payload,
                artifact_payload=current_payload,
            )
            fill_results = [
                dict(item)
                for item in (current_late_fill.get('fill_results') or [])
                if isinstance(item, dict)
            ]
            last_effective_request_payload = request_payload

            while pending_branches:
                latest_record = self.get_response_lookup_record(response_id) or {}
                latest_payload = latest_record.get('response_payload') if isinstance(latest_record.get('response_payload'), dict) else None
                if latest_payload:
                    current_payload = dict(latest_payload)
                    current_late_fill = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), dict) else current_late_fill
                    cancelled_branch_records = self.normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
                    cancelled_branches = [self.branch_id(item) for item in cancelled_branch_records if self.branch_id(item)]
                    cancelled_capabilities = _cancelled_capability_list()
                gate_open_pending: list[dict[str, Any]] = []
                gate_terminalized = False
                for branch in pending_branches:
                    decision = self.semantic_execution_gate_decision(branch, current_payload)
                    if str(decision.get('action') or '').strip().lower() == 'skip':
                        gated_branch = self.branch_record_with_execution_gate(branch, decision)
                        _remember_branch(
                            cancelled_branch_records,
                            cancelled_branches,
                            gated_branch,
                            status=str(gated_branch.get('status') or decision.get('status') or 'cancelled'),
                        )
                        gate_terminalized = True
                        self.log_unified_event(
                            category='responses',
                            action='late_fill',
                            status=str(gated_branch.get('status') or 'cancelled'),
                            response_id=response_id,
                            capability=self.branch_capability(gated_branch),
                            message=str(decision.get('reason') or 'Late-fill branch skipped by semantic execution gate.'),
                        )
                    else:
                        gate_open_pending.append(branch)
                if gate_terminalized:
                    pending_branches = gate_open_pending
                    cancelled_capabilities = _cancelled_capability_list()
                    pending_capabilities, completed_capabilities, failed_capabilities = _capability_lists_for_branch_state(
                        pending_branches
                    )
                    gate_state = self.build_late_fill_state(
                        build_gap(
                            artifact_gap,
                            capability=self.normalize_capability(artifact_gap.get('expected_capability')),
                            artifact_payload=current_payload,
                            pending_capabilities=pending_capabilities,
                            completed_capabilities=completed_capabilities,
                            failed_capabilities=failed_capabilities,
                        ),
                        status=_status_for_terminal_state(pending_branches),
                        prior_state=current_late_fill,
                        extra={
                            'pending_capabilities': pending_capabilities,
                            'completed_capabilities': completed_capabilities,
                            'failed_capabilities': failed_capabilities,
                            'cancelled_capabilities': cancelled_capabilities,
                            'pending_branches': pending_branches,
                            'completed_branches': completed_branch_records,
                            'failed_branches': failed_branch_records,
                            'cancelled_branches': cancelled_branch_records,
                            'active_branches': [],
                            'fill_results': fill_results,
                            'execution_gate_status': 'applied',
                        },
                    )
                    payload_for_finalize, effective_gate_status = self.finalize_terminal_materialization_contract(
                        self.attach_late_fill_state(current_payload, gate_state),
                        request_payload=last_effective_request_payload,
                        route_payload=source_route_payload,
                        artifact_gap=artifact_gap,
                        terminal_status=str(gate_state.get('status') or 'completed'),
                    )
                    current_payload = self.finalize_response_frame_payload(
                        payload_for_finalize,
                        request_payload=last_effective_request_payload,
                        persist=True,
                    )
                    current_late_fill = gate_state
                    self.touch_response_lookup(
                        response_id,
                        status=_final_lookup_status_for_late_fill(effective_gate_status),
                        output_text=str(current_payload.get('output_text') or ''),
                        response_payload=current_payload,
                    )
                    if not pending_branches:
                        current_payload, successor_handoff = (
                            self._prepare_terminal_graph_patch_successor_handoff(
                                current_payload,
                                request_payload=last_effective_request_payload,
                                assistant_message=assistant_message,
                                artifact_gap=artifact_gap,
                                source_route_payload=source_route_payload,
                            )
                        )
                        if successor_handoff is None:
                            self.schedule_terminal_substrate_hygiene(
                                current_payload,
                                route_payload=source_route_payload,
                                reason='late_fill_execution_gate_terminal',
                            )
                        return

                pending_capabilities, completed_capabilities, failed_capabilities = _capability_lists_for_branch_state(
                    pending_branches
                )
                cancelled_capabilities = _cancelled_capability_list()
                executable_batches = self.downstream_request_phase_batches(
                    request_phase_graph,
                    pending_branches=pending_branches,
                    completed_branches=completed_branch_records,
                    failed_branches=failed_branch_records,
                    pending_capabilities=pending_capabilities,
                    completed_capabilities=completed_capabilities,
                    failed_capabilities=failed_capabilities,
                )
                active_branches = [dict(branch) for branch in (executable_batches[0] if executable_batches else [])]
                if not active_branches and pending_branches:
                    active_branches = [dict(pending_branches[0])]
                pending_branch_by_id = {
                    self.branch_id(branch): dict(branch)
                    for branch in pending_branches
                    if self.branch_id(branch)
                }
                active_branches = [
                    {
                        **dict(branch),
                        **pending_branch_by_id.get(self.branch_id(branch), {}),
                    }
                    for branch in active_branches
                ]
                active_branches, scheduling_policy = self.shape_active_late_fill_branches(
                    active_branches,
                    artifact_gap=artifact_gap,
                )
                active_capabilities = self.normalize_capability_list(
                    [self.branch_capability(branch) for branch in active_branches if self.branch_capability(branch)]
                )
                if not active_capabilities:
                    active_capabilities = pending_capabilities[:1]
                active_capability = active_capabilities[0] if active_capabilities else None
                active_gap = build_gap(
                    artifact_gap,
                    capability=active_capability,
                    artifact_payload=current_payload,
                    pending_capabilities=pending_capabilities,
                    completed_capabilities=completed_capabilities,
                    failed_capabilities=failed_capabilities,
                )
                running_state_extra = {
                    'pending_capabilities': pending_capabilities,
                    'completed_capabilities': completed_capabilities,
                    'failed_capabilities': failed_capabilities,
                    'cancelled_capabilities': cancelled_capabilities,
                    'active_capability': active_capability,
                    'active_capabilities': active_capabilities,
                    'scheduling_policy': scheduling_policy,
                    'pending_branches': pending_branches,
                    'completed_branches': completed_branch_records,
                    'failed_branches': failed_branch_records,
                    'cancelled_branches': cancelled_branch_records,
                    'active_branches': active_branches,
                    'fill_results': fill_results,
                }
                for key in ('materialization_concurrency_policy', 'materialization_concurrency_history'):
                    value = current_late_fill.get(key) if isinstance(current_late_fill, Mapping) else None
                    if value not in (None, '', [], {}):
                        running_state_extra[key] = value
                running_late_fill = self.build_late_fill_state(
                    active_gap,
                    status='running',
                    prior_state=current_late_fill,
                    extra=running_state_extra,
                )
                current_payload = self.attach_late_fill_state(current_payload, running_late_fill)
                self.touch_response_lookup(
                    response_id,
                    status='completed',
                    output_text=str(current_payload.get('output_text') or ''),
                    response_payload=current_payload,
                )

                def publish_live_branch_progress(event: dict[str, Any]) -> None:
                    if not isinstance(event, Mapping):
                        return
                    branch_id = self.branch_id(event)
                    if not branch_id:
                        return
                    status = str(event.get('status') or '').strip().lower()
                    progress_record: dict[str, Any] = {
                        'branch_id': branch_id,
                        'phase_id': str(event.get('phase_id') or branch_id).strip() or branch_id,
                        'capability': self.normalize_capability(event.get('capability')),
                        'status': status or 'running',
                        'progress_stage': str(event.get('progress_stage') or 'branch_execution').strip(),
                        'updated_at': self.response_registry_now_iso(),
                    }
                    instance_id = str(event.get('instance_id') or '').strip()
                    if instance_id:
                        progress_record['instance_id'] = instance_id
                    timing = event.get('timing') if isinstance(event.get('timing'), Mapping) else {}
                    if timing:
                        progress_record['timing'] = {
                            key: value
                            for key, value in dict(timing).items()
                            if value not in (None, '', [], {})
                        }
                    error = event.get('error') if isinstance(event.get('error'), Mapping) else {}
                    if error:
                        progress_record['error'] = {
                            key: value
                            for key, value in dict(error).items()
                            if value not in (None, '', [], {})
                        }
                    latest_record = self.get_response_lookup_record(response_id) or {}
                    latest_payload = (
                        latest_record.get('response_payload')
                        if isinstance(latest_record.get('response_payload'), dict)
                        else current_payload
                    )
                    latest_late_fill = (
                        dict(latest_payload.get('late_fill'))
                        if isinstance(latest_payload.get('late_fill'), dict)
                        else dict(running_late_fill)
                    )
                    branch_progress = [
                        dict(item)
                        for item in (latest_late_fill.get('branch_progress') or [])
                        if isinstance(item, Mapping) and self.branch_id(item) and self.branch_id(item) != branch_id
                    ]
                    branch_progress.append(
                        {
                            key: value
                            for key, value in progress_record.items()
                            if value not in (None, '', [], {})
                        }
                    )
                    terminal_progress_ids = {
                        self.branch_id(item)
                        for item in branch_progress
                        if str(item.get('status') or '').strip().lower()
                        in {'completed', 'fulfilled', 'failed', 'blocked', 'cancelled', 'waived', 'superseded'}
                        and self.branch_id(item)
                    }
                    if terminal_progress_ids:
                        for key in ('active_branches', 'pending_branches'):
                            latest_late_fill[key] = [
                                dict(branch)
                                for branch in self.normalize_late_fill_branches(latest_late_fill.get(key))
                                if self.branch_id(branch) not in terminal_progress_ids
                            ]
                    latest_late_fill['branch_progress'] = branch_progress
                    updated_payload = dict(latest_payload)
                    updated_payload = self.attach_late_fill_state(updated_payload, latest_late_fill)
                    self.touch_response_lookup(
                        response_id,
                        output_text=str(updated_payload.get('output_text') or ''),
                        response_payload=updated_payload,
                    )

                branch_results: dict[str, dict[str, Any]] = {}
                preblocked_branch_errors: dict[str, dict[str, Any]] = {}
                executable_active_branches: list[dict[str, Any]] = []
                for branch in active_branches:
                    branch_id = self.branch_id(branch)
                    if not branch_id:
                        continue
                    repair_error = self.repair_branch_execution_error(
                        branch,
                        current_payload=current_payload,
                        artifact_gap=artifact_gap,
                    )
                    if repair_error:
                        preblocked_branch_errors[branch_id] = self.normalize_late_fill_error_payload(repair_error)
                        continue
                    executable_active_branches.append(branch)
                branch_errors: dict[str, dict[str, Any]] = dict(preblocked_branch_errors)
                # The response target completed the source phase; that alone
                # is not failure evidence for any downstream branch. Actual
                # retry exclusions are carried branch-locally by
                # failed_instance_id, attempt.instance_id, and explicit
                # excluded_instance_ids.
                failed_instance_id = None
                runtime_candidate_snapshot: list[dict[str, Any]] = []
                runtime_candidate_snapshot_meta: dict[str, Any] = {}
                if using_default_prepare_plan and executable_active_branches:
                    runtime_candidate_snapshot, runtime_candidate_snapshot_meta = (
                        self.runtime_candidate_snapshot_for_late_fill_wave()
                    )
                branch_specs = [
                    branch_spec
                    for branch_spec in (
                        self.build_late_fill_materialization_branch_spec(
                            branch=dict(branch),
                            artifact_gap=artifact_gap,
                            current_payload=current_payload,
                            request_payload=request_payload,
                            assistant_message=assistant_message,
                            source_route_payload=source_route_payload,
                            failed_instance_id=failed_instance_id,
                        )
                        for branch in executable_active_branches
                    )
                    if isinstance(branch_spec, dict)
                ]
                if runtime_candidate_snapshot:
                    for branch_spec in branch_specs:
                        prepare_args = (
                            dict(branch_spec.get('prepare_args') or {})
                            if isinstance(branch_spec.get('prepare_args'), Mapping)
                            else {}
                        )
                        branch_gap = (
                            dict(prepare_args.get('artifact_gap') or {})
                            if isinstance(prepare_args.get('artifact_gap'), Mapping)
                            else {}
                        )
                        branch_gap['_runtime_candidate_snapshot'] = [
                            dict(entry)
                            for entry in runtime_candidate_snapshot
                            if isinstance(entry, Mapping)
                        ]
                        branch_gap['_runtime_candidate_snapshot_meta'] = dict(runtime_candidate_snapshot_meta)
                        prepare_args['artifact_gap'] = branch_gap
                        branch_spec['prepare_args'] = prepare_args
                prefulfilled_text_branch_ids: set[str] = set()
                executable_branch_specs_for_wave: list[dict[str, Any]] = []
                latest_record_for_preflight = self.get_response_lookup_record(response_id) or {}
                latest_payload_for_preflight = (
                    latest_record_for_preflight.get('response_payload')
                    if isinstance(latest_record_for_preflight.get('response_payload'), Mapping)
                    else {}
                )
                for branch_spec in branch_specs:
                    branch = (
                        dict(branch_spec.get('branch'))
                        if isinstance(branch_spec.get('branch'), Mapping)
                        else {}
                    )
                    prepare_args = (
                        dict(branch_spec.get('prepare_args') or {})
                        if isinstance(branch_spec.get('prepare_args'), Mapping)
                        else {}
                    )
                    branch_gap = (
                        dict(prepare_args.get('artifact_gap') or {})
                        if isinstance(prepare_args.get('artifact_gap'), Mapping)
                        else {}
                    )
                    evidence_branch = dict(branch)
                    for key in (
                        'branch_id',
                        'phase_id',
                        'capability',
                        'output_type',
                        'stage_direction',
                        'requires_artifact',
                        'text_artifact_extension',
                        'text_artifact_source_name',
                        'text_artifact_source',
                        'text_artifact_target_path',
                        'artifact_request',
                    ):
                        value = evidence_branch.get(key)
                        if value in (None, '', [], {}):
                            value = branch_gap.get(key)
                        if value in (None, '', [], {}):
                            value = branch_spec.get(key)
                        if value not in (None, '', [], {}):
                            evidence_branch[key] = value
                    canonical_fulfillment = self._canonical_text_artifact_branch_fulfillment(
                        evidence_branch,
                        current_payload,
                        latest_payload_for_preflight,
                        allow_deterministic_repair=True,
                    )
                    branch_id = str(canonical_fulfillment.get('branch_id') or '').strip()
                    if not canonical_fulfillment or not branch_id:
                        executable_branch_specs_for_wave.append(branch_spec)
                        continue
                    prefulfilled_text_branch_ids.add(branch_id)
                    text_source = str(
                        canonical_fulfillment.get('text_artifact_source')
                        or 'canonical_text_artifact_evidence'
                    ).strip() or 'canonical_text_artifact_evidence'
                    fulfilled_branch = dict(evidence_branch)
                    fulfilled_branch['status'] = 'fulfilled'
                    fulfilled_branch['evidence'] = text_source
                    fulfilled_branch['non_blocking_after_final_contract_fulfilled'] = True
                    _remember_branch(
                        completed_branch_records,
                        completed_branches,
                        fulfilled_branch,
                        status='fulfilled',
                    )
                    failed_branch_records = [
                        record
                        for record in failed_branch_records
                        if self.branch_id(record) != branch_id
                    ]
                    failed_branches = [
                        item
                        for item in failed_branches
                        if item != branch_id
                    ]
                    branch_errors.pop(branch_id, None)
                    if not any(
                        self.branch_id(item) == branch_id
                        and str(item.get('saved_text_path') or '').strip()
                        == str(canonical_fulfillment.get('saved_text_path') or '').strip()
                        for item in fill_results
                        if isinstance(item, Mapping)
                    ):
                        fill_results.append(dict(canonical_fulfillment))
                    self.log_unified_event(
                        category='responses',
                        action='late_fill',
                        status='ok',
                        response_id=response_id,
                        capability=str(canonical_fulfillment.get('capability') or 'chat'),
                        message=str(
                            canonical_fulfillment.get('result_text')
                            or 'Required text artifact satisfied by canonical saved file evidence.'
                        ),
                    )
                if prefulfilled_text_branch_ids:
                    active_branches = [
                        dict(branch)
                        for branch in active_branches
                        if self.branch_id(branch) not in prefulfilled_text_branch_ids
                    ]
                branch_specs = executable_branch_specs_for_wave
                branch_specs = self._coalesce_required_text_artifact_branch_specs(branch_specs)
                if branch_specs:
                    materialization_result = self.execute_materialization_branches(
                        branch_specs,
                        prepare_branch_plan=prepare_plan,
                        execute_prepared_branch=execute_plan,
                        on_branch_progress=publish_live_branch_progress,
                        async_branch_progress=True,
                    )
                    materialization_result = (
                        self._retry_failed_coalesced_text_artifact_materializations(
                            branch_specs,
                            materialization_result,
                            prepare_branch_plan=prepare_plan,
                            execute_prepared_branch=execute_plan,
                            on_branch_progress=publish_live_branch_progress,
                        )
                    )
                else:
                    materialization_result = {
                        'branch_results': {},
                        'branch_errors': {},
                        'prepared_branch_plans': [],
                        'concurrency_policy': {},
                    }
                materialization_concurrency_policy = (
                    dict(materialization_result.get('concurrency_policy'))
                    if isinstance(materialization_result.get('concurrency_policy'), Mapping)
                    else {}
                )
                materialization_concurrency_history = [
                    dict(item)
                    for item in (
                        (
                            (current_payload.get('late_fill') or {}).get(
                                'materialization_concurrency_history'
                            )
                            or []
                        )
                        if isinstance(current_payload.get('late_fill'), Mapping)
                        else []
                    )
                    if isinstance(item, Mapping)
                ]
                concurrency_policies = (
                    materialization_result.get(
                        'materialization_concurrency_policies'
                    )
                    if isinstance(
                        materialization_result.get(
                            'materialization_concurrency_policies'
                        ),
                        list,
                    )
                    else [materialization_concurrency_policy]
                )
                for concurrency_policy in concurrency_policies:
                    if not isinstance(concurrency_policy, Mapping) or not concurrency_policy:
                        continue
                    materialization_concurrency_history = (
                        self.materialization_concurrency_history(
                            {
                                'materialization_concurrency_history': (
                                    materialization_concurrency_history
                                )
                            },
                            concurrency_policy,
                        )
                    )
                branch_results = {
                    str(branch_id or '').strip(): result
                    for branch_id, result in (
                        (materialization_result.get('branch_results') or {}).items()
                        if isinstance(materialization_result.get('branch_results'), dict)
                        else []
                    )
                    if str(branch_id or '').strip() and isinstance(result, dict)
                }
                materialization_branch_errors = {
                    str(branch_id or '').strip(): self.normalize_late_fill_error_payload(error)
                    for branch_id, error in (
                        (materialization_result.get('branch_errors') or {}).items()
                        if isinstance(materialization_result.get('branch_errors'), dict)
                        else []
                    )
                    if str(branch_id or '').strip() and error
                }
                branch_errors.update(materialization_branch_errors)
                prepared_plans_by_branch_id = {
                    str((plan or {}).get('branch_id') or '').strip(): dict(plan)
                    for plan in (
                        materialization_result.get('prepared_branch_plans') or []
                        if isinstance(materialization_result.get('prepared_branch_plans'), list)
                        else []
                    )
                    if isinstance(plan, dict) and str((plan or {}).get('branch_id') or '').strip()
                }
                retry_branch_replacements: dict[str, dict[str, Any]] = {}

                for branch in active_branches:
                    branch_id = self.branch_id(branch)
                    capability = self.branch_capability(branch)
                    if not branch_id or not capability:
                        continue
                    latest_record = self.get_response_lookup_record(response_id) or {}
                    latest_payload = latest_record.get('response_payload') if isinstance(latest_record.get('response_payload'), dict) else None
                    gate_payload = current_payload
                    if latest_payload:
                        gate_payload = latest_payload
                        latest_late_fill = latest_payload.get('late_fill') if isinstance(latest_payload.get('late_fill'), dict) else {}
                        cancelled_branch_records = self.normalize_late_fill_branches(latest_late_fill.get('cancelled_branches'))
                        cancelled_branches = [self.branch_id(item) for item in cancelled_branch_records if self.branch_id(item)]
                    post_execution_decision = self.semantic_execution_gate_decision(branch, gate_payload)
                    if str(post_execution_decision.get('action') or '').strip().lower() == 'skip':
                        gated_branch = self.branch_record_with_execution_gate(branch, post_execution_decision)
                        _remember_branch(
                            cancelled_branch_records,
                            cancelled_branches,
                            gated_branch,
                            status=str(gated_branch.get('status') or post_execution_decision.get('status') or 'cancelled'),
                        )
                        self.log_unified_event(
                            category='responses',
                            action='late_fill',
                            status=str(gated_branch.get('status') or 'cancelled'),
                            response_id=response_id,
                            capability=capability,
                            message=str(post_execution_decision.get('reason') or 'Late-fill result ignored by semantic execution gate.'),
                        )
                        continue
                    branch_result = branch_results.get(branch_id)
                    if not branch_result:
                        canonical_text_result: dict[str, Any] = {}
                        if (
                            self._branch_is_required_text_artifact(branch)
                            and not self._text_artifact_revision_required(branch)
                        ):
                            artifact_request = (
                                branch.get('artifact_request')
                                if isinstance(branch.get('artifact_request'), Mapping)
                                else {}
                            )
                            text_extension = str(
                                branch.get('text_artifact_extension')
                                or artifact_request.get('extension')
                                or ''
                            ).strip().lower().lstrip('.')
                            text_source_name = str(
                                branch.get('text_artifact_source_name')
                                or artifact_request.get('source_name')
                                or ''
                            ).strip()
                            text_target_path = self._text_artifact_target_path_from_mapping(branch) or self._text_artifact_target_path_from_mapping(
                                artifact_request
                            )
                            for evidence_payload in (current_payload, latest_payload or {}):
                                if not isinstance(evidence_payload, Mapping):
                                    continue
                                canonical_text_result = self._canonical_text_artifact_saved_result(
                                    evidence_payload,
                                    extension=text_extension,
                                    source_name=text_source_name,
                                    target_path=text_target_path,
                                    allow_deterministic_repair=True,
                                )
                                if canonical_text_result:
                                    break
                        canonical_text_path = str(
                            canonical_text_result.get('saved_text_path')
                            or canonical_text_result.get('path')
                            or ''
                        ).strip()
                        if canonical_text_path:
                            text_source = str(
                                canonical_text_result.get('text_artifact_source')
                                or 'canonical_text_artifact_evidence'
                            ).strip()
                            if not text_source:
                                text_source = 'canonical_text_artifact_evidence'
                            event_message = (
                                'Required text artifact satisfied by deterministic canonical saved file repair.'
                                if text_source == 'canonical_text_artifact_deterministic_syntax_repair'
                                else 'Required text artifact satisfied by canonical saved file evidence.'
                            )
                            fulfilled_branch = dict(branch)
                            fulfilled_branch['status'] = 'fulfilled'
                            fulfilled_branch['evidence'] = text_source
                            fulfilled_branch['non_blocking_after_final_contract_fulfilled'] = True
                            _remember_branch(
                                completed_branch_records,
                                completed_branches,
                                fulfilled_branch,
                                status='fulfilled',
                            )
                            failed_branch_records = [
                                record
                                for record in failed_branch_records
                                if self.branch_id(record) != branch_id
                            ]
                            failed_branches = [
                                item
                                for item in failed_branches
                                if item != branch_id
                            ]
                            branch_errors.pop(branch_id, None)
                            fill_results.append(
                                {
                                    'branch_id': branch_id,
                                    'phase_id': str(branch.get('phase_id') or branch_id).strip() or branch_id,
                                    'capability': capability,
                                    'saved_text_path': canonical_text_path,
                                    'text_artifact_extension': text_extension,
                                    'text_artifact_source_name': text_source_name,
                                    'text_artifact_source': text_source,
                                    'result_text': event_message,
                                    **(
                                        {
                                            'target_path_authoritative_repair': dict(
                                                canonical_text_result.get('target_path_authoritative_repair') or {}
                                            )
                                        }
                                        if isinstance(
                                            canonical_text_result.get('target_path_authoritative_repair'),
                                            Mapping,
                                        )
                                        else {}
                                    ),
                                }
                            )
                            self.log_unified_event(
                                category='responses',
                                action='late_fill',
                                status='ok',
                                response_id=response_id,
                                capability=capability,
                                message=event_message,
                            )
                            continue
                        error_payload = branch_errors.get(branch_id) or self.normalize_late_fill_error_payload(None)
                        attempt_payload = self.late_fill_failure_attempt(
                            branch,
                            error_payload,
                            prepared_plans_by_branch_id.get(branch_id),
                        )
                        recovery_context = self.late_fill_recovery_context(
                            error=error_payload,
                            attempt=attempt_payload,
                        )
                        recovery_state = self.late_fill_recovery_state(
                            branch,
                            recovery_context=recovery_context,
                            attempt=attempt_payload,
                            status='candidate',
                        )
                        retry_branch = self.build_auto_executable_repair_retry_branch(
                            branch,
                            recovery_context=recovery_context,
                            recovery_state=recovery_state,
                            attempt=attempt_payload,
                            trigger='auto_executable_repair_retry',
                        )
                        if retry_branch:
                            retry_branch_replacements[branch_id] = retry_branch
                            branch_errors.pop(branch_id, None)
                            self.log_unified_event(
                                category='responses',
                                action='late_fill',
                                status='queued',
                                response_id=response_id,
                                capability=capability,
                                message='Retryable auto-executable repair branch requeued after failed materialization attempt.',
                            )
                            continue
                        _remember_branch(
                            failed_branch_records,
                            failed_branches,
                            branch,
                            status='failed',
                            error=error_payload,
                            attempt=attempt_payload,
                            recovery_context=recovery_context,
                            recovery_state=recovery_state,
                        )
                        self.log_unified_event(
                            category='responses',
                            action='late_fill',
                            status='failed',
                            response_id=response_id,
                            capability=capability,
                            message=str(error_payload.get('message') or 'Late fill request failed.'),
                        )
                        continue

                    late_fill_route_info = branch_result.get('route_info') if isinstance(branch_result.get('route_info'), dict) else {}
                    late_fill_instance = branch_result.get('instance') if isinstance(branch_result.get('instance'), dict) else {}
                    infer_result = branch_result.get('infer_result') if isinstance(branch_result.get('infer_result'), dict) else {}
                    effective_data = branch_result.get('effective_data') if isinstance(branch_result.get('effective_data'), dict) else {}
                    execution_contract = (
                        branch_result.get('execution_contract')
                        if isinstance(branch_result.get('execution_contract'), dict)
                        else (
                            infer_result.get('execution_contract')
                            if isinstance(infer_result.get('execution_contract'), dict)
                            else (
                                effective_data.get('execution_contract')
                                if isinstance(effective_data.get('execution_contract'), dict)
                                else {}
                            )
                        )
                    )
                    external_provider_block = (
                        infer_result.get('external_provider_block')
                        if isinstance(
                            infer_result.get('external_provider_block'),
                            Mapping,
                        )
                        else {}
                    )
                    if external_provider_block:
                        error_payload = self.normalize_late_fill_error_payload(
                            {
                                'code': 'EXTERNAL_PROVIDER_BLOCKED',
                                'message': str(
                                    external_provider_block.get('reason')
                                    or 'The downstream provider blocked the '
                                    'bounded graph-owned chat phase.'
                                ).strip(),
                                'stage': 'external_chat_phase',
                                'retryable': False,
                                'external_execution': dict(
                                    infer_result.get('external_execution') or {}
                                ),
                            }
                        )
                        attempt_payload = self.late_fill_failure_attempt(
                            branch,
                            error_payload,
                            prepared_plans_by_branch_id.get(branch_id),
                        )
                        recovery_context = self.late_fill_recovery_context(
                            error=error_payload,
                            attempt=attempt_payload,
                        )
                        recovery_state = self.late_fill_recovery_state(
                            branch,
                            recovery_context=recovery_context,
                            attempt=attempt_payload,
                            status='blocked',
                            trigger='external_provider_block',
                        )
                        _remember_branch(
                            failed_branch_records,
                            failed_branches,
                            branch,
                            status='blocked',
                            error=error_payload,
                            attempt=attempt_payload,
                            recovery_context=recovery_context,
                            recovery_state=recovery_state,
                        )
                        self.log_unified_event(
                            category='responses',
                            action='late_fill',
                            status='blocked',
                            response_id=response_id,
                            capability=capability,
                            message=str(error_payload.get('message') or ''),
                        )
                        continue
                    if (
                        capability == self.capability_image_generation
                        and not str(infer_result.get('saved_image_path') or effective_data.get('saved_image_path') or '').strip()
                        and _payload_has_raw_image_result(infer_result)
                    ):
                        infer_result = self.persist_raw_image_result_if_possible(
                            infer_result,
                            effective_data=effective_data,
                            late_fill_instance=late_fill_instance,
                        )
                    if capability == self.capability_text_to_speech:
                        integrity_evidence = (
                            self.tts_audio_integrity_evidence_for_branch_result(
                                branch,
                                infer_result,
                            )
                        )
                        if integrity_evidence:
                            infer_result = dict(infer_result)
                            infer_result['tts_audio_integrity_evidence'] = (
                                integrity_evidence
                            )
                    tts_stt_semantic_evidence = (
                        self.tts_stt_semantic_evidence_for_branch_result(
                            branch,
                            infer_result,
                            current_payload=current_payload,
                        )
                    )
                    if str(
                        tts_stt_semantic_evidence.get('status') or ''
                    ).strip().lower() != 'not_applicable':
                        infer_result = dict(infer_result)
                        infer_result['tts_stt_semantic_evidence'] = (
                            tts_stt_semantic_evidence
                        )
                    evidence_error = self.dependency_evidence_error_for_branch_result(
                        branch,
                        infer_result,
                        current_payload=current_payload,
                    )
                    if evidence_error:
                        error_payload = self.normalize_late_fill_error_payload(evidence_error)
                        attempt_payload = self.late_fill_failure_attempt(
                            branch,
                            error_payload,
                            prepared_plans_by_branch_id.get(branch_id),
                        )
                        recovery_context = self.late_fill_recovery_context(
                            error=error_payload,
                            attempt=attempt_payload,
                        )
                        recovery_state = self.late_fill_recovery_state(
                            branch,
                            recovery_context=recovery_context,
                            attempt=attempt_payload,
                            status='candidate',
                            trigger='semantic_evidence_gate',
                        )
                        retry_branch = (
                            self.build_auto_executable_repair_retry_branch(
                                branch,
                                recovery_context=recovery_context,
                                recovery_state=recovery_state,
                                attempt=attempt_payload,
                                trigger=(
                                    _TTS_AUTO_RECOVERY_TRIGGER
                                ),
                            )
                        )
                        if retry_branch:
                            retry_branch_replacements[branch_id] = retry_branch
                            branch_errors.pop(branch_id, None)
                            self.log_unified_event(
                                category='responses',
                                action='late_fill',
                                status='queued',
                                response_id=response_id,
                                capability=capability,
                                recovery_policy_id=(
                                    _TTS_AUTO_RECOVERY_POLICY_ID
                                ),
                                message=(
                                    'Required TTS branch requeued for one bounded '
                                    'integrity recovery attempt.'
                                ),
                            )
                            continue
                        _remember_branch(
                            failed_branch_records,
                            failed_branches,
                            branch,
                            status='blocked',
                            error=error_payload,
                            attempt=attempt_payload,
                            recovery_context=recovery_context,
                            recovery_state=recovery_state,
                        )
                        self.log_unified_event(
                            category='responses',
                            action='late_fill',
                            status='blocked',
                            response_id=response_id,
                            capability=capability,
                            message=str(error_payload.get('message') or 'Dependency evidence repair required.'),
                        )
                        continue
                    infer_result, text_artifact_error = self._ensure_required_text_artifact_saved_truth(
                        branch,
                        infer_result,
                        effective_data,
                        current_payload,
                    )
                    if text_artifact_error:
                        error_payload = self.normalize_late_fill_error_payload(text_artifact_error)
                        attempt_payload = self.late_fill_failure_attempt(
                            branch,
                            error_payload,
                            prepared_plans_by_branch_id.get(branch_id),
                        )
                        if not str(attempt_payload.get('instance_id') or '').strip():
                            attempt_payload = self.late_fill_failure_attempt(
                                branch,
                                error_payload,
                                {
                                    'route_info': late_fill_route_info,
                                    'instance': late_fill_instance,
                                },
                            )
                        recovery_context = self.late_fill_recovery_context(
                            error=error_payload,
                            attempt=attempt_payload,
                        )
                        recovery_state = self.late_fill_recovery_state(
                            branch,
                            recovery_context=recovery_context,
                            attempt=attempt_payload,
                            status='candidate',
                            trigger='text_artifact_saved_truth_gate',
                        )
                        retry_branch = self.build_auto_executable_repair_retry_branch(
                            branch,
                            recovery_context=recovery_context,
                            recovery_state=recovery_state,
                            attempt=attempt_payload,
                            trigger='auto_executable_repair_retry',
                        )
                        if retry_branch:
                            retry_branch_replacements[branch_id] = retry_branch
                            branch_errors.pop(branch_id, None)
                            self.log_unified_event(
                                category='responses',
                                action='late_fill',
                                status='queued',
                                response_id=response_id,
                                capability=capability,
                                message='Retryable auto-executable text artifact repair requeued after saved-truth failure.',
                            )
                            continue
                        _remember_branch(
                            failed_branch_records,
                            failed_branches,
                            branch,
                            status='failed',
                            error=error_payload,
                            attempt=attempt_payload,
                            recovery_context=recovery_context,
                            recovery_state=recovery_state,
                        )
                        self.log_unified_event(
                            category='responses',
                            action='late_fill',
                            status='failed',
                            response_id=response_id,
                            capability=capability,
                            message=str(error_payload.get('message') or 'Text artifact saved truth required.'),
                        )
                        continue
                    current_payload = self.merge_late_fill_result_fields(current_payload, infer_result)
                    _remember_branch(completed_branch_records, completed_branches, branch, status='fulfilled')
                    last_effective_request_payload = effective_data or last_effective_request_payload
                    fill_record = {
                        'branch_id': branch_id,
                        'phase_id': str(branch.get('phase_id') or branch_id).strip() or branch_id,
                        'capability': capability,
                        'fill_instance_id': str((late_fill_route_info or {}).get('instance_id') or '').strip() or None,
                        'fill_model': str((late_fill_instance or {}).get('model') or '').strip() or None,
                        'fill_backend': str((late_fill_instance or {}).get('backend') or '').strip() or None,
                        'fill_mode': str((infer_result or {}).get('mode') or '').strip() or None,
                        'route_source': str((late_fill_route_info or {}).get('route_source') or '').strip() or None,
                        'route_reason': str((late_fill_route_info or {}).get('route_reason') or '').strip() or None,
                    }
                    late_fill_route_runtime = (
                        late_fill_route_info.get('route_runtime')
                        if isinstance(
                            late_fill_route_info.get('route_runtime'),
                            Mapping,
                        )
                        else {}
                    )
                    for key in (
                        'selection_policy',
                        'excluded_instance_ids',
                        'excluded_candidate_count',
                        'candidate_diagnostics',
                        'tts_recovery_trigger',
                        'tts_recovery_policy_id',
                        'excluded_instance_reuse_reason',
                        'excluded_instance_reuse_instance_id',
                        'excluded_instance_reuse_recovery_trigger',
                        'external_execution',
                    ):
                        value = late_fill_route_runtime.get(key)
                        if value not in (None, '', [], {}):
                            fill_record[key] = value
                    if isinstance(branch.get('recovery_attempt'), Mapping):
                        fill_record['recovery_attempt'] = dict(
                            branch.get('recovery_attempt') or {}
                        )
                    if execution_contract:
                        fill_record['execution_contract'] = execution_contract
                        workload_task_ref = (
                            execution_contract.get('workload_task_ref')
                            if isinstance(execution_contract.get('workload_task_ref'), Mapping)
                            else {}
                        )
                        output_obligation_ref = (
                            execution_contract.get('output_obligation_ref')
                            if isinstance(execution_contract.get('output_obligation_ref'), Mapping)
                            else {}
                        )
                        if workload_task_ref:
                            fill_record['workload_task_ref'] = dict(workload_task_ref)
                            task_id = str(workload_task_ref.get('task_id') or '').strip()
                            if task_id:
                                fill_record['task_id'] = task_id
                                fill_record['workload_task_id'] = task_id
                        if output_obligation_ref:
                            fill_record['output_obligation_ref'] = dict(output_obligation_ref)
                            obligation_id = str(output_obligation_ref.get('obligation_id') or '').strip()
                            if obligation_id:
                                fill_record['obligation_id'] = obligation_id
                        execution_artifact_request = (
                            execution_contract.get('artifact_request')
                            if isinstance(execution_contract.get('artifact_request'), Mapping)
                            else {}
                        )
                        for key in (
                            'artifact_request',
                            'text_artifact_extension',
                            'text_artifact_source_name',
                            'text_artifact_source',
                            'text_artifact_target_path',
                        ):
                            value = branch.get(key)
                            if value in (None, '', [], {}):
                                value = execution_contract.get(key)
                            if (
                                value in (None, '', [], {})
                                and key == 'text_artifact_extension'
                                and execution_artifact_request
                            ):
                                value = execution_artifact_request.get('extension')
                            if (
                                value in (None, '', [], {})
                                and key == 'text_artifact_source_name'
                                and execution_artifact_request
                            ):
                                value = execution_artifact_request.get('source_name')
                            if (
                                value in (None, '', [], {})
                                and key == 'text_artifact_source'
                                and execution_artifact_request
                            ):
                                value = execution_artifact_request.get('source')
                            if (
                                value in (None, '', [], {})
                                and key == 'text_artifact_target_path'
                                and execution_artifact_request
                            ):
                                value = execution_artifact_request.get('target_path')
                            if value not in (None, '', [], {}):
                                fill_record[key] = value
                    result_text = self.late_fill_text_from_result_payload(infer_result)
                    result_artifact_type = self.artifact_type_for_capability(capability)
                    if result_text:
                        fill_record['result_text'] = result_text
                        if result_artifact_type == 'text':
                            fill_record['content_payload'] = result_text
                            fill_record['content_payload_source'] = 'late_fill_infer_result'
                    for key in (
                        'saved_text_path',
                        'saved_text_artifacts',
                        'text_artifact_requests',
                        'coalesced_text_artifact_recovery',
                        'text_artifact_revision_required',
                        'text_artifact_source_is_input',
                        'text_artifact_revision_write_proof',
                        'text_artifact_revision_preservation_evidence',
                        'saved_audio_path',
                        'saved_image_path',
                        'image_data_url',
                        'image_artifact_persisted_from_raw_late_fill',
                        'result',
                        'audio_mimetype',
                        'lang_code',
                        'lang_code_source',
                        'voice',
                        'instruct',
                        'response_format',
                        'output_format',
                        'speed',
                        'pitch',
                        'tts_semantic_source',
                        'tts_generation_budget',
                        'tts_sampling_profile',
                        'tts_audio_integrity_evidence',
                        'tts_stt_semantic_evidence',
                    ):
                        value = infer_result.get(key)
                        if value in (None, '', [], {}):
                            value = effective_data.get(key)
                        if value not in (None, '', [], {}):
                            fill_record[key] = value
                    for key in (
                        'artifact_id',
                        'artifact_ref',
                        'ref',
                        'provenance_id',
                        'derived_from',
                        'artifacts',
                    ):
                        value = infer_result.get(key)
                        if value not in (None, '', [], {}):
                            fill_record[key] = value
                    fill_record = self.attach_late_fill_result_artifact_identity(
                        fill_record,
                        current_payload,
                        infer_result,
                        capability=capability,
                    )
                    fill_results.append(
                        {
                            key: value
                            for key, value in fill_record.items()
                            if value not in (None, '', [], {})
                        }
                    )
                    self.log_unified_event(
                        category='responses',
                        action='late_fill',
                        status='ok',
                        instance_id=str((late_fill_route_info or {}).get('instance_id') or '').strip() or None,
                        model=str((late_fill_instance or {}).get('model') or '').strip() or None,
                        backend=str((late_fill_instance or {}).get('backend') or '').strip() or None,
                        capability=capability,
                        response_id=response_id,
                        message='Late artifact fill completed.',
                    )

                completed_branch_ids = set(completed_branches)
                failed_branch_ids = set(failed_branches)
                cancelled_branch_ids = set(cancelled_branches)
                next_pending_branches: list[dict[str, Any]] = []
                for branch in pending_branches:
                    pending_branch_id = self.branch_id(branch)
                    if (
                        pending_branch_id in completed_branch_ids
                        or pending_branch_id in failed_branch_ids
                        or pending_branch_id in cancelled_branch_ids
                    ):
                        continue
                    if pending_branch_id and pending_branch_id in retry_branch_replacements:
                        next_pending_branches.append(dict(retry_branch_replacements[pending_branch_id]))
                    else:
                        next_pending_branches.append(dict(branch))
                pending_branches = next_pending_branches
                pending_capabilities, completed_capabilities, failed_capabilities = _capability_lists_for_branch_state(
                    pending_branches
                )
                cancelled_capabilities = _cancelled_capability_list()
                next_batches = self.downstream_request_phase_batches(
                    request_phase_graph,
                    pending_branches=pending_branches,
                    completed_branches=completed_branch_records,
                    failed_branches=failed_branch_records,
                    pending_capabilities=pending_capabilities,
                    completed_capabilities=completed_capabilities,
                    failed_capabilities=failed_capabilities,
                )
                next_active_branches = [dict(branch) for branch in (next_batches[0] if next_batches else [])]
                if not next_active_branches and pending_branches:
                    next_active_branches = [dict(pending_branches[0])]
                next_pending_branch_by_id = {
                    self.branch_id(branch): dict(branch)
                    for branch in pending_branches
                    if self.branch_id(branch)
                }
                next_active_branches = [
                    {
                        **dict(branch),
                        **next_pending_branch_by_id.get(self.branch_id(branch), {}),
                    }
                    for branch in next_active_branches
                ]
                next_active_capabilities = self.normalize_capability_list(
                    [self.branch_capability(branch) for branch in next_active_branches if self.branch_capability(branch)]
                )
                next_active_capability = (
                    self.branch_capability(next_active_branches[0]) if next_active_branches else None
                ) or (next_active_capabilities[0] if next_active_capabilities else None)
                next_gap = build_gap(
                    artifact_gap,
                    capability=next_active_capability or active_capability,
                    artifact_payload=current_payload,
                    pending_capabilities=pending_capabilities,
                    completed_capabilities=completed_capabilities,
                    failed_capabilities=failed_capabilities,
                )
                next_status = _status_for_terminal_state(pending_branches)
                next_state_extra = {
                    'pending_capabilities': pending_capabilities,
                    'completed_capabilities': completed_capabilities,
                    'failed_capabilities': failed_capabilities,
                    'cancelled_capabilities': cancelled_capabilities,
                    'active_capability': next_active_capability or active_capability,
                    'active_capabilities': next_active_capabilities,
                    'pending_branches': pending_branches,
                    'completed_branches': completed_branch_records,
                    'failed_branches': failed_branch_records,
                    'cancelled_branches': cancelled_branch_records,
                    'active_branches': next_active_branches,
                    'fill_results': fill_results,
                    'completed_branch_count': len(completed_branch_records),
                    'failed_branch_count': len(failed_branch_records),
                    'cancelled_branch_count': len(cancelled_branch_records),
                }
                if materialization_concurrency_policy:
                    next_state_extra['materialization_concurrency_policy'] = materialization_concurrency_policy
                if materialization_concurrency_history:
                    next_state_extra['materialization_concurrency_history'] = materialization_concurrency_history
                cohort_recovery_history = [
                    dict(item)
                    for item in (
                        materialization_result.get(
                            'coalesced_text_artifact_recovery_history'
                        )
                        or []
                    )
                    if isinstance(item, Mapping)
                ]
                if cohort_recovery_history:
                    previous_cohort_recovery_history = [
                        dict(item)
                        for item in (
                            current_late_fill.get(
                                'coalesced_text_artifact_recovery_history'
                            )
                            or []
                        )
                        if isinstance(item, Mapping)
                    ]
                    next_state_extra[
                        'coalesced_text_artifact_recovery_history'
                    ] = [
                        *previous_cohort_recovery_history,
                        *cohort_recovery_history,
                    ][-8:]
                if next_status == 'partial_failed':
                    next_state_extra['partial_failure'] = True
                recovery_candidates = [
                    dict(branch['recovery_state'])
                    for branch in failed_branch_records
                    if isinstance(branch.get('recovery_state'), Mapping)
                ]
                if recovery_candidates:
                    next_state_extra['recovery_candidates'] = recovery_candidates
                    next_state_extra['auto_recovery_enabled'] = False
                if len(fill_results) == 1:
                    next_state_extra.update(fill_results[0])
                if branch_errors and not pending_branches:
                    next_state_extra['error'] = '; '.join(
                        f'{branch_id}: {message}'
                        for branch_id, error_payload in branch_errors.items()
                        for message in [str((error_payload or {}).get('message') or '').strip()]
                        if message
                    )
                next_state = self.build_late_fill_state(
                    next_gap,
                    status=next_status,
                    prior_state=running_late_fill,
                    extra=next_state_extra,
                )
                payload_for_finalize = self.attach_late_fill_state(current_payload, next_state)
                terminal_without_pending = not pending_branches
                effective_next_status = next_status
                if terminal_without_pending:
                    payload_for_finalize, effective_next_status = self.finalize_terminal_materialization_contract(
                        payload_for_finalize,
                        request_payload=request_payload,
                        route_payload=source_route_payload,
                        artifact_gap=next_gap,
                        terminal_status=next_status,
                    )
                checkpoint_mode = (
                    'terminal_full_frame'
                    if terminal_without_pending
                    else 'lightweight_nonterminal'
                )
                if terminal_without_pending:
                    finalize_started_at = time.perf_counter()
                    current_payload = self.finalize_response_frame_payload(
                        payload_for_finalize,
                        request_payload=request_payload,
                        persist=True,
                    )
                    finalize_elapsed_ms = round((time.perf_counter() - finalize_started_at) * 1000, 3)
                    runtime = current_payload.get('runtime') if isinstance(current_payload.get('runtime'), Mapping) else {}
                    diagnostics = (
                        runtime.get('developer_diagnostics')
                        if isinstance(runtime.get('developer_diagnostics'), Mapping)
                        else {}
                    )
                    response_frame_finalize_timing = (
                        diagnostics.get('response_frame_finalize_timing')
                        if isinstance(diagnostics.get('response_frame_finalize_timing'), Mapping)
                        else {}
                    )
                else:
                    finalize_started_at = time.perf_counter()
                    current_payload = dict(payload_for_finalize)
                    finalize_elapsed_ms = round((time.perf_counter() - finalize_started_at) * 1000, 3)
                    response_frame_finalize_timing = {
                        'kind': 'ollmo.response_frame_finalize_timing',
                        'phase': 'nonterminal_late_fill',
                        'skipped': True,
                        'reason': 'lightweight_nonterminal_checkpoint',
                        'checkpoint_mode': checkpoint_mode,
                        'persist_requested': False,
                        'total_elapsed_ms': finalize_elapsed_ms,
                        'pending_branch_count': len(pending_branches),
                        'active_branch_count': len(next_active_branches),
                        'completed_branch_count': len(completed_branch_records),
                        'failed_branch_count': len(failed_branch_records),
                    }
                current_late_fill = (
                    current_payload.get('late_fill')
                    if isinstance(current_payload.get('late_fill'), dict)
                    else next_state
                )
                post_wave_backend_timing = {
                    'kind': 'ollmo.late_fill_post_wave_backend_timing',
                    'phase': 'terminal' if terminal_without_pending else 'nonterminal',
                    'status': effective_next_status,
                    'checkpoint_mode': checkpoint_mode,
                    'finalize_skipped': not terminal_without_pending,
                    'finalize_elapsed_ms': finalize_elapsed_ms,
                    'pending_branch_count': len(pending_branches),
                    'active_branch_count': len(next_active_branches),
                    'completed_branch_count': len(completed_branch_records),
                    'failed_branch_count': len(failed_branch_records),
                }
                if response_frame_finalize_timing:
                    post_wave_backend_timing['response_frame_finalize_timing'] = dict(response_frame_finalize_timing)
                if isinstance(current_late_fill, dict):
                    current_late_fill = dict(current_late_fill)
                    current_late_fill['post_wave_backend_timing'] = dict(post_wave_backend_timing)
                    current_payload = self.attach_late_fill_state(current_payload, current_late_fill)
                lookup_touch_started_at = time.perf_counter()
                self.touch_response_lookup(
                    response_id,
                    status=_final_lookup_status_for_late_fill(effective_next_status),
                    output_text=str(current_payload.get('output_text') or ''),
                    response_payload=current_payload,
                    stream_view='ui' if terminal_without_pending else 'status',
                )
                lookup_touch_elapsed_ms = round((time.perf_counter() - lookup_touch_started_at) * 1000, 3)
                post_wave_backend_timing['touch_response_lookup_elapsed_ms'] = lookup_touch_elapsed_ms
                if isinstance(current_late_fill, dict):
                    current_late_fill = dict(current_late_fill)
                    current_late_fill['post_wave_backend_timing'] = dict(post_wave_backend_timing)
                    current_payload = self.attach_late_fill_state(current_payload, current_late_fill)
                self.log_unified_event(
                    category='responses',
                    action='late_fill_post_wave_backend_timing',
                    status='ok',
                    response_id=response_id,
                    phase=post_wave_backend_timing['phase'],
                    late_fill_status=effective_next_status,
                    checkpoint_mode=checkpoint_mode,
                    finalize_skipped=not terminal_without_pending,
                    finalize_elapsed_ms=finalize_elapsed_ms,
                    touch_response_lookup_elapsed_ms=lookup_touch_elapsed_ms,
                    response_frame_finalize_timing=(
                        dict(response_frame_finalize_timing)
                        if response_frame_finalize_timing
                        else None
                    ),
                    pending_branch_count=len(pending_branches),
                    active_branch_count=len(next_active_branches),
                    completed_branch_count=len(completed_branch_records),
                    failed_branch_count=len(failed_branch_records),
                    message=(
                        'Late-fill post-wave backend timing '
                        f"{post_wave_backend_timing['phase']} finalize="
                        f'{finalize_elapsed_ms}ms lookup={lookup_touch_elapsed_ms}ms'
                    ),
                )
                if terminal_without_pending:
                    current_payload, successor_handoff = (
                        self._prepare_terminal_graph_patch_successor_handoff(
                            current_payload,
                            request_payload=request_payload,
                            assistant_message=assistant_message,
                            artifact_gap=next_gap,
                            source_route_payload=source_route_payload,
                        )
                    )
                    if successor_handoff is None:
                        self.schedule_terminal_substrate_hygiene(
                            current_payload,
                            route_payload=source_route_payload,
                            reason='late_fill_terminal',
                        )
        except Exception as exc:  # noqa: BLE001
            existing_record = self.get_response_lookup_record(response_id) or {}
            current_payload = dict(existing_record.get('response_payload') or response_payload or {})
            current_late_fill = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), dict) else {}
            completed_branch_records = self.normalize_late_fill_branches(current_late_fill.get('completed_branches'))
            failed_branch_records = self.normalize_late_fill_branches(current_late_fill.get('failed_branches'))
            cancelled_branch_records = self.normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
            active_capability = self.normalize_capability(
                current_late_fill.get('active_capability') or artifact_gap.get('expected_capability')
            )
            active_branches = self.normalize_late_fill_branches(current_late_fill.get('active_branches'))
            if active_branches:
                for branch in active_branches:
                    branch_id = self.branch_id(branch)
                    if not branch_id:
                        continue
                    if branch_id not in failed_branches:
                        failed_branches.append(branch_id)
                    if branch_id not in {self.branch_id(item) for item in failed_branch_records}:
                        error_payload = self.normalize_late_fill_error_payload(exc)
                        attempt_payload = self.late_fill_failure_attempt(branch, error_payload)
                        failed_branch = dict(branch)
                        failed_branch['status'] = 'failed'
                        failed_branch['error'] = error_payload
                        failed_branch['attempt'] = attempt_payload
                        recovery_context = self.late_fill_recovery_context(
                            error=error_payload,
                            attempt=attempt_payload,
                        )
                        failed_branch['recovery_context'] = recovery_context
                        failed_branch['recovery_state'] = self.late_fill_recovery_state(
                            failed_branch,
                            recovery_context=recovery_context,
                            attempt=attempt_payload,
                            status='candidate',
                        )
                        failed_branch_records.append(failed_branch)
            elif active_capability and active_capability not in failed_capabilities:
                failed_capabilities.append(active_capability)
            failed_late_fill = self.build_late_fill_state(
                build_gap(
                    artifact_gap,
                    capability=active_capability,
                    artifact_payload=current_payload,
                    pending_capabilities=self.normalize_capability_list(current_late_fill.get('pending_capabilities')),
                    completed_capabilities=completed_capabilities,
                    failed_capabilities=failed_capabilities,
                ),
                status='failed',
                prior_state=current_late_fill,
                extra={
                    'error': str(exc),
                    'completed_capabilities': completed_capabilities,
                    'failed_capabilities': failed_capabilities,
                    'cancelled_capabilities': self.normalize_capability_list(current_late_fill.get('cancelled_capabilities')),
                    'pending_capabilities': self.normalize_capability_list(current_late_fill.get('pending_capabilities')),
                    'active_capability': active_capability,
                    'pending_branches': self.normalize_late_fill_branches(current_late_fill.get('pending_branches')),
                    'completed_branches': completed_branch_records,
                    'failed_branches': failed_branch_records,
                    'cancelled_branches': cancelled_branch_records,
                    'active_branches': active_branches,
                    'recovery_candidates': [
                        dict(branch['recovery_state'])
                        for branch in failed_branch_records
                        if isinstance(branch.get('recovery_state'), Mapping)
                    ],
                    'auto_recovery_enabled': False,
                },
            )
            updated_payload = self.attach_late_fill_state(current_payload, failed_late_fill)
            finalized_payload = self.finalize_response_frame_payload(
                updated_payload,
                request_payload=request_payload,
                persist=True,
            )
            self.touch_response_lookup(
                response_id,
                status='incomplete' if completed_branch_records else 'failed',
                output_text=str(finalized_payload.get('output_text') or ''),
                response_payload=finalized_payload,
            )
            self.log_unified_event(
                category='responses',
                action='late_fill',
                status='failed',
                response_id=response_id,
                capability=active_capability,
                message=str(exc),
            )
        finally:
            self.release_response_late_fill(response_id)
            if successor_handoff and successor_handoff.get('skip_schedule') is not True:
                try:
                    scheduled = self.schedule_response_late_fill(**successor_handoff)
                except Exception as exc:  # noqa: BLE001
                    scheduled = False
                    logging.warning('Could not schedule graph-patch successor Late Fill: %s', exc)
                if not scheduled:
                    execution = (
                        successor_handoff.get('artifact_gap', {}).get('successor_reopen_execution')
                        if isinstance(successor_handoff.get('artifact_gap'), dict)
                        else {}
                    )
                    self.log_unified_event(
                        category='responses',
                        action='graph_patch_successor_reopen',
                        status='pending',
                        response_id=response_id,
                        successor_execution_key=(execution or {}).get('successor_execution_key'),
                        message=(
                            'Successor frame is durable but its Late Fill worker was not newly claimed; '
                            'an existing worker or explicit recovery remains authoritative.'
                        ),
                    )

    def schedule_terminal_substrate_hygiene(
        self,
        response_payload: Mapping[str, Any],
        *,
        route_payload: Optional[Mapping[str, Any]] = None,
        reason: str = 'late_fill_terminal',
    ) -> None:
        if not callable(self.schedule_post_response_substrate_hygiene):
            return
        try:
            self.schedule_post_response_substrate_hygiene(
                response_payload,
                route_payload=route_payload,
                reason=reason,
            )
        except Exception:  # noqa: BLE001
            logging.exception('Could not schedule post-response substrate hygiene.')

    def schedule_response_late_fill(
        self,
        *,
        response_payload: dict[str, Any],
        request_payload: dict[str, Any],
        assistant_message: str,
        artifact_gap: dict[str, Any],
        source_route_payload: Optional[dict[str, Any]],
        complete_response_late_fill: Optional[Callable[..., None]] = None,
    ) -> bool:
        response_id = str((response_payload or {}).get('id') or '').strip()
        if not response_id or not self.claim_response_late_fill(response_id):
            return False
        if has_app_context() and bool(current_app.config.get('TESTING')):
            # Keep tests deterministic by avoiding background late-fill workers unless a test patches the scheduler explicitly.
            self.release_response_late_fill(response_id)
            return True
        target = complete_response_late_fill or self.complete_response_late_fill
        worker = threading.Thread(
            target=target,
            kwargs={
                'response_payload': dict(response_payload or {}),
                'request_payload': dict(request_payload or {}),
                'assistant_message': str(assistant_message or '').strip(),
                'artifact_gap': dict(artifact_gap or {}),
                'source_route_payload': dict(source_route_payload or {}) if isinstance(source_route_payload, dict) else None,
            },
            daemon=True,
        )
        worker.start()
        return True
