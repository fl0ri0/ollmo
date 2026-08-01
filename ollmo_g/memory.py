"""Deterministic Ghost memory-chain helpers."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Optional

from ollmo_g.control_hints import infer_tts_language_from_prompt
from ollmo_g.intent import analyze_prompt_intent

HOT_MEMORY_LIMIT = 8
HOT_ARTIFACT_LIMIT = 4
WARM_MEMORY_LINE_LIMIT = 8
DEEP_MEMORY_LIMIT = 8
DEEP_CONTINUITY_ARTIFACT_LIMIT = 3
RECENT_USER_CHAIN_LIMIT = 6
_ROUTER_TIMEOUT_LOG_RE = re.compile(r'Ghost router execution fallback:.*Read timed out', re.IGNORECASE)
_IMAGE_CONTEXT_CHAT_FALLBACK_RE = re.compile(r'default chat fallback.*supports image-aware chat', re.IGNORECASE)
_IMAGE_ROUTE_SUCCESS_RE = re.compile(
    r'(latest image artifact|modify.*previously generated image|sequence of image generation requests|image-generation cue)',
    re.IGNORECASE,
)
_EXPLICIT_STABLE_PREFERENCE_PATTERNS = (
    re.compile(r'\bprefer(?:ence|ences|red|s)?\b', re.IGNORECASE),
    re.compile(r'\bby default\b', re.IGNORECASE),
    re.compile(r'\bdefault to\b', re.IGNORECASE),
    re.compile(r'\bfrom now on\b', re.IGNORECASE),
    re.compile(r'\bplease always\b', re.IGNORECASE),
    re.compile(r'\balways (?:answer|reply|respond|speak|use|read|write)\b', re.IGNORECASE),
    re.compile(r'\bbevorzuge\b', re.IGNORECASE),
    re.compile(r'\bpräferiere\b', re.IGNORECASE),
    re.compile(r'\bpraeferiere\b', re.IGNORECASE),
    re.compile(r'\bstandardmäßig\b', re.IGNORECASE),
    re.compile(r'\bstandardmaessig\b', re.IGNORECASE),
    re.compile(r'\bab jetzt\b', re.IGNORECASE),
    re.compile(r'\bbitte immer\b', re.IGNORECASE),
    re.compile(r'\b(?:immer|stets) (?:auf|in|mit|als|antwort(?:e)?|nutze|verwende|lies|sprich)\b', re.IGNORECASE),
)



def _clip(value: Any, *, max_chars: int = 220) -> str:
    text = str(value or '').strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + '...[truncated]'


def _normalize_message(message: Any) -> Optional[dict[str, Any]]:
    if not isinstance(message, dict):
        return None
    role = str(message.get('role') or '').strip().lower() or 'user'
    if role not in {'user', 'assistant', 'system'}:
        role = 'user'
    content = str(message.get('content') or '').strip()
    payload = {
        'role': role,
        'content': content,
        'timestamp': str(message.get('timestamp') or message.get('ts') or '').strip() or None,
    }
    for source_key, target_key in (
        ('response_model', 'response_model'),
        ('response_instance_id', 'response_instance_id'),
        ('route_source', 'route_source'),
        ('route_reason', 'route_reason'),
        ('context_mode', 'context_mode'),
        ('saved_image_path', 'saved_image_path'),
        ('saved_text_path', 'saved_text_path'),
        ('saved_audio_path', 'saved_audio_path'),
    ):
        value = str(message.get(source_key) or '').strip()
        if value:
            payload[target_key] = value
    if not payload['content'] and not any(payload.get(key) for key in ('saved_image_path', 'saved_text_path', 'saved_audio_path')):
        return None
    return payload


def _normalize_artifact(artifact: Any) -> Optional[dict[str, Any]]:
    if not isinstance(artifact, dict):
        return None
    path = str(artifact.get('path') or '').strip()
    if not path:
        return None
    payload = {
        'type': str(artifact.get('type') or '').strip() or None,
        'path': path,
        'timestamp': str(artifact.get('timestamp') or '').strip() or None,
        'role': str(artifact.get('role') or '').strip() or None,
    }
    image_state = artifact.get('image_state')
    if isinstance(image_state, dict) and image_state:
        payload['image_state'] = image_state
    return payload


def _normalize_event(event: Any) -> Optional[dict[str, Any]]:
    if not isinstance(event, dict):
        return None
    action = str(event.get('action') or '').strip()
    if not action:
        return None
    payload = {
        'timestamp': str(event.get('timestamp') or '').strip() or None,
        'category': str(event.get('category') or '').strip() or None,
        'action': action,
        'status': str(event.get('status') or '').strip() or None,
        'instance_id': str(event.get('instance_id') or '').strip() or None,
        'model': str(event.get('model') or '').strip() or None,
        'backend': str(event.get('backend') or '').strip() or None,
        'capability': str(event.get('capability') or '').strip() or None,
        'route_source': str(event.get('route_source') or '').strip() or None,
        'message': str(event.get('message') or '').strip() or None,
        'previous_instance_id': str(event.get('previous_instance_id') or '').strip() or None,
        'previous_model': str(event.get('previous_model') or '').strip() or None,
        'new_instance_id': str(event.get('new_instance_id') or '').strip() or None,
        'new_model': str(event.get('new_model') or '').strip() or None,
        'conversation_id': str(event.get('conversation_id') or '').strip() or None,
        'response_id': str(event.get('response_id') or '').strip() or None,
        'observed_capability': str(event.get('observed_capability') or '').strip() or None,
        'response_instance_id': str(event.get('response_instance_id') or '').strip() or None,
        'response_model': str(event.get('response_model') or '').strip() or None,
        'route_reason': str(event.get('route_reason') or '').strip() or None,
        'suggested_capability': str(event.get('suggested_capability') or '').strip() or None,
        'suggested_instance_id': str(event.get('suggested_instance_id') or '').strip() or None,
        'route_hint_capability': (
            str(event.get('route_hint_capability') or event.get('heuristic_capability') or '').strip() or None
        ),
        'prompt_class': str(event.get('prompt_class') or '').strip() or None,
        'session_class': str(event.get('session_class') or '').strip() or None,
    }
    if 'route_hint_confidence' in event or 'heuristic_confidence' in event:
        payload['route_hint_confidence'] = float(
            event.get('route_hint_confidence')
            if event.get('route_hint_confidence') is not None
            else event.get('heuristic_confidence')
            or 0.0
        )
    if 'embedding_score' in event:
        payload['embedding_score'] = float(event.get('embedding_score') or 0.0)
    if 'embedding_score_gap' in event:
        payload['embedding_score_gap'] = float(event.get('embedding_score_gap') or 0.0)
    if 'bias_applied' in event:
        payload['bias_applied'] = bool(event.get('bias_applied'))
    expected_capabilities = event.get('expected_capabilities')
    if isinstance(expected_capabilities, list):
        payload['expected_capabilities'] = [
            str(item or '').strip()
            for item in expected_capabilities
            if str(item or '').strip()
        ][:8]
    payload['expected_none'] = bool(event.get('expected_none')) if 'expected_none' in event else None
    payload['assessment_scope'] = str(event.get('assessment_scope') or '').strip() or None
    payload['comment'] = str(event.get('comment') or '').strip() or None
    return payload


def _compact_memory_message(message: dict[str, Any], *, max_chars: int = 180) -> dict[str, Any]:
    payload = {
        'role': message.get('role'),
        'content': _clip(message.get('content') or '', max_chars=max_chars),
        'timestamp': message.get('timestamp'),
    }
    for key in ('response_model', 'response_instance_id', 'route_source', 'context_mode'):
        if message.get(key):
            payload[key] = message.get(key)
    return payload


def _build_hot_memory(
    messages: list[dict[str, Any]],
    recent_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    hot_messages = [_compact_memory_message(item) for item in messages[-HOT_MEMORY_LIMIT:]]
    artifact_hints = [artifact for artifact in recent_artifacts[:HOT_ARTIFACT_LIMIT] if isinstance(artifact, dict)]
    return {
        'turn_count': len(hot_messages),
        'messages': hot_messages,
        'artifact_hints': artifact_hints,
    }


def _build_warm_memory(messages: list[dict[str, Any]]) -> dict[str, Any]:
    older_messages = messages[:-HOT_MEMORY_LIMIT] if len(messages) > HOT_MEMORY_LIMIT else []
    summary_lines: list[str] = []
    for item in older_messages[-WARM_MEMORY_LINE_LIMIT:]:
        role = str(item.get('role') or 'user')
        content = _clip(item.get('content') or '', max_chars=120)
        descriptor = f'{role}: {content}' if content else role
        response_model = str(item.get('response_model') or '').strip()
        if response_model:
            descriptor += f' [model={response_model}]'
        if item.get('saved_text_path'):
            descriptor += f' [text={item.get("saved_text_path")}]'
        elif item.get('saved_image_path'):
            descriptor += f' [image={item.get("saved_image_path")}]'
        elif item.get('saved_audio_path'):
            descriptor += f' [audio={item.get("saved_audio_path")}]'
        summary_lines.append(descriptor)
    return {
        'older_turn_count': len(older_messages),
        'summary_lines': summary_lines,
    }


def _looks_like_explicit_stable_preference(content: str) -> bool:
    normalized = str(content or '').strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _EXPLICIT_STABLE_PREFERENCE_PATTERNS)


def _infer_explicit_preference_capability(content: str, analysis: dict[str, Any]) -> Optional[str]:
    normalized = str(content or '').strip().lower()
    if not normalized:
        return None
    if re.search(r'\b(audio|spoken|speech|voice|tts|read aloud)\b', normalized):
        return 'text_to_speech'
    if re.search(r'\b(image|images|picture|pictures|photo|photos|illustration|illustrations|visual|visuals)\b', normalized):
        return 'image_generation'
    if re.search(r'\b(chat|text[- ]only|plain text|text replies|text responses)\b', normalized):
        return 'chat'
    primary_capability = str(analysis.get('primary_capability') or '').strip()
    return primary_capability or None


def _build_stable_user_preferences(messages: list[dict[str, Any]]) -> dict[str, Any]:
    language_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()

    for item in messages:
        if str(item.get('role') or '').strip().lower() != 'user':
            continue
        content = str(item.get('content') or '').strip()
        if not content or not _looks_like_explicit_stable_preference(content):
            continue
        analysis = analyze_prompt_intent(content)
        explicit_languages = [str(code or '').strip() for code in (analysis.get('language_codes') or []) if str(code or '').strip()]
        detected_language = infer_tts_language_from_prompt(content)
        languages = explicit_languages[:]
        if detected_language and detected_language not in languages:
            languages.append(detected_language)
        for code in languages:
            token = str(code or '').strip()
            if token:
                language_counts[token] += 1
        for descriptor in analysis.get('voice_descriptors') or []:
            token = str(descriptor or '').strip()
            if token:
                voice_counts[token] += 1
        primary_capability = _infer_explicit_preference_capability(content, analysis)
        if primary_capability:
            modality_counts[primary_capability] += 1
        audio_format = str(analysis.get('audio_response_format') or '').strip()
        if audio_format:
            format_counts[audio_format] += 1

    return {
        'languages': [item for item, _count in language_counts.most_common(4)],
        'voice_descriptors': [item for item, _count in voice_counts.most_common(6)],
        'preferred_modalities': [item for item, _count in modality_counts.most_common(4)],
        'audio_formats': [item for item, _count in format_counts.most_common(3)],
    }


def _build_continuity_artifacts(recent_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for artifact in recent_artifacts[:DEEP_CONTINUITY_ARTIFACT_LIMIT]:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get('path') or '').strip()
        if not path:
            continue
        anchors.append(
            {
                'type': str(artifact.get('type') or '').strip() or None,
                'path': path,
                'timestamp': str(artifact.get('timestamp') or '').strip() or None,
            }
        )
    return anchors


def build_recent_learnings_from_events(
    events: Iterable[dict[str, Any]],
    *,
    limit: int = DEEP_MEMORY_LIMIT,
) -> list[dict[str, Any]]:
    embedding_audit_learnings: dict[tuple[str, str, str], dict[str, Any]] = {}
    success_learnings: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for raw in events or []:
        event = _normalize_event(raw)
        if not event:
            continue
        if event.get('action') == 'route_embedding_audit':
            prompt_class = str(event.get('prompt_class') or '').strip() or 'general_preference'
            suggested_capability = str(event.get('suggested_capability') or '').strip() or None
            observed_capability = str(event.get('capability') or '').strip() or None
            status = str(event.get('status') or '').strip() or 'observed'
            if not suggested_capability:
                continue
            key = (prompt_class, suggested_capability, status)
            entry = embedding_audit_learnings.get(key)
            if not entry:
                entry = {
                    'kind': 'embedding_route_audit',
                    'prompt_class': prompt_class,
                    'suggested_capability': suggested_capability,
                    'observed_capability': observed_capability,
                    'status': status,
                    'bias_applied': bool(event.get('bias_applied')),
                    'reason': _clip(event.get('message') or 'embedding route audit', max_chars=180),
                    'count': 0,
                    'last_seen': event.get('timestamp'),
                }
                embedding_audit_learnings[key] = entry
            entry['count'] = int(entry.get('count') or 0) + 1
            timestamp = str(event.get('timestamp') or '').strip()
            if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
                entry['last_seen'] = timestamp
            if event.get('bias_applied'):
                entry['bias_applied'] = True
            continue

        if event.get('action') != 'request' or event.get('status') != 'ok':
            continue
        if event.get('category') not in {'infer', 'chat'}:
            continue
        capability = str(event.get('capability') or '').strip() or None
        instance_id = str(event.get('instance_id') or '').strip() or None
        model = str(event.get('model') or '').strip() or None
        backend = str(event.get('backend') or '').strip() or None
        if not capability or not instance_id:
            continue
        key = (
            capability or '',
            instance_id or '',
            model or '',
            backend or '',
        )
        entry = success_learnings.get(key)
        if not entry:
            entry = {
                'kind': 'successful_execution',
                'capability': capability,
                'instance_id': instance_id,
                'model': model,
                'backend': backend,
                'status': event.get('status') or 'ok',
                'reason': _clip(event.get('message') or 'recent successful execution', max_chars=180),
                'count': 0,
                'last_seen': event.get('timestamp'),
            }
            success_learnings[key] = entry
        entry['count'] = int(entry.get('count') or 0) + 1
        timestamp = str(event.get('timestamp') or '').strip()
        if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
            entry['last_seen'] = timestamp

    ranked = sorted(
        [*embedding_audit_learnings.values(), *success_learnings.values()],
        key=lambda item: (
            2 if str(item.get('kind') or '') == 'embedding_route_audit' else 1,
            int(item.get('count') or 0),
            str(item.get('last_seen') or ''),
            str(item.get('suggested_capability') or item.get('model') or ''),
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_recent_self_observations(
    events: Iterable[dict[str, Any]],
    *,
    runtime_issues: Optional[Iterable[str]] = None,
    log_lines: Optional[Iterable[str]] = None,
    limit: int = DEEP_MEMORY_LIMIT,
) -> list[dict[str, Any]]:
    successful_patterns: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    route_success_patterns: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_patterns: dict[str, dict[str, Any]] = {}
    runtime_issue_patterns: dict[str, dict[str, Any]] = {}
    router_timeout_patterns: dict[tuple[str, str], dict[str, Any]] = {}

    for raw in events or []:
        event = _normalize_event(raw)
        if not event:
            continue
        timestamp = str(event.get('timestamp') or '').strip() or None
        action = str(event.get('action') or '').strip()
        status = str(event.get('status') or '').strip()
        category = str(event.get('category') or '').strip()
        message = str(event.get('message') or '').strip()
        capability = str(event.get('capability') or '').strip() or None
        instance_id = str(event.get('instance_id') or '').strip() or None
        model = str(event.get('model') or '').strip() or None
        backend = str(event.get('backend') or '').strip() or None

        if action == 'request' and status == 'ok' and category in {'infer', 'chat'} and capability and instance_id:
            key = (capability, instance_id, model or '', backend or '')
            entry = successful_patterns.get(key)
            if not entry:
                entry = {
                    'kind': 'successful_pattern',
                    'capability': capability,
                    'instance_id': instance_id,
                    'model': model,
                    'backend': backend,
                    'reason': _clip(message or 'recent successful execution pattern', max_chars=180),
                    'count': 0,
                    'last_seen': timestamp,
                }
                successful_patterns[key] = entry
            entry['count'] = int(entry.get('count') or 0) + 1
            if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
                entry['last_seen'] = timestamp
            continue

        if category == 'responses' and action == 'route' and status == 'ok':
            normalized_message = message.lower()
            if capability == 'image_generation' and _IMAGE_ROUTE_SUCCESS_RE.search(normalized_message):
                key = (capability or '', 'image_edit_or_follow_up_success')
                entry = route_success_patterns.get(key)
                if not entry:
                    entry = {
                        'kind': 'successful_route_pattern',
                        'capability': capability,
                        'pattern': 'image_edit_or_follow_up_success',
                        'reason': _clip(message or 'image follow-up route succeeded', max_chars=180),
                        'count': 0,
                        'last_seen': timestamp,
                    }
                    route_success_patterns[key] = entry
                entry['count'] = int(entry.get('count') or 0) + 1
                if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
                    entry['last_seen'] = timestamp
                continue
            if capability == 'chat' and _IMAGE_CONTEXT_CHAT_FALLBACK_RE.search(normalized_message):
                entry = fallback_patterns.get('image_context_chat_fallback')
                if not entry:
                    entry = {
                        'kind': 'fallback_pattern',
                        'pattern': 'image_context_chat_fallback',
                        'reason': _clip(message or 'image-context prompt fell back to chat', max_chars=180),
                        'count': 0,
                        'last_seen': timestamp,
                    }
                    fallback_patterns['image_context_chat_fallback'] = entry
                entry['count'] = int(entry.get('count') or 0) + 1
                if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
                    entry['last_seen'] = timestamp
                continue
            if capability == 'chat' and message.lower().startswith('default chat fallback'):
                entry = fallback_patterns.get('chat_fallback')
                if not entry:
                    entry = {
                        'kind': 'fallback_pattern',
                        'pattern': 'chat_fallback',
                        'reason': _clip(message or 'prompt fell back to chat', max_chars=180),
                        'count': 0,
                        'last_seen': timestamp,
                    }
                    fallback_patterns['chat_fallback'] = entry
                entry['count'] = int(entry.get('count') or 0) + 1
                if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
                    entry['last_seen'] = timestamp
                continue
        if category == 'responses' and action == 'router_runtime' and status == 'timeout':
            prompt_class = str(event.get('prompt_class') or '').strip() or None
            session_class = str(event.get('session_class') or '').strip() or None
            key = (prompt_class or '', session_class or '')
            entry = router_timeout_patterns.get(key)
            if not entry:
                entry = {
                    'kind': 'router_timeout',
                    'reason': _clip(message or 'Ghost router timed out.', max_chars=220),
                    'count': 0,
                    'last_seen': timestamp,
                    'prompt_class': prompt_class,
                    'session_class': session_class,
                }
                router_timeout_patterns[key] = entry
            entry['count'] = int(entry.get('count') or 0) + 1
            if timestamp and (not entry.get('last_seen') or timestamp > str(entry.get('last_seen') or '')):
                entry['last_seen'] = timestamp

    for raw_issue in runtime_issues or []:
        issue = str(raw_issue or '').strip()
        if not issue:
            continue
        entry = runtime_issue_patterns.get(issue)
        if not entry:
            entry = {
                'kind': 'runtime_issue',
                'reason': _clip(issue, max_chars=180),
                'count': 0,
                'last_seen': None,
            }
            runtime_issue_patterns[issue] = entry
        entry['count'] = int(entry.get('count') or 0) + 1

    for raw_line in log_lines or []:
        line = str(raw_line or '').strip()
        if not line:
            continue
        if _ROUTER_TIMEOUT_LOG_RE.search(line):
            key = ('', '')
            entry = router_timeout_patterns.get(key)
            if not entry:
                entry = {
                    'kind': 'router_timeout',
                    'reason': _clip(line, max_chars=220),
                    'count': 0,
                    'last_seen': None,
                    'prompt_class': None,
                    'session_class': None,
                }
                router_timeout_patterns[key] = entry
            entry['count'] = int(entry.get('count') or 0) + 1

    ranked = sorted(
        [
            *successful_patterns.values(),
            *route_success_patterns.values(),
            *fallback_patterns.values(),
            *runtime_issue_patterns.values(),
            *router_timeout_patterns.values(),
        ],
        key=lambda item: (
            3 if str(item.get('kind') or '') == 'successful_pattern' else
            2 if str(item.get('kind') or '') == 'successful_route_pattern' else
            1 if str(item.get('kind') or '') in {'fallback_pattern', 'router_timeout'} else 0,
            int(item.get('count') or 0),
            str(item.get('last_seen') or ''),
            str(item.get('reason') or ''),
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_self_healing_hints(
    self_observations: Iterable[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in self_observations or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get('kind') or '').strip()
        pattern = str(item.get('pattern') or '').strip()
        if kind == 'successful_pattern' and 'successful_pattern' not in seen:
            hints.append(
                {
                    'kind': 'preserve_successful_execution',
                    'reason': 'Prefer stable successful execution patterns before reacting to one-off failures.',
                }
            )
            seen.add('successful_pattern')
        elif kind == 'successful_route_pattern' and pattern == 'image_edit_or_follow_up_success' and pattern not in seen:
            hints.append(
                {
                    'kind': 'preserve_image_follow_up_success',
                    'reason': 'Prefer image_generation with latest-image reuse for edit/follow-up prompts when recent image chains have succeeded.',
                }
            )
            seen.add(pattern)
        elif kind == 'fallback_pattern' and pattern == 'image_context_chat_fallback' and pattern not in seen:
            hints.append(
                {
                    'kind': 'avoid_image_context_chat_fallback',
                    'reason': 'When the latest artifact is an image and the prompt sounds like an edit, avoid falling back to chat.',
                }
            )
            seen.add(pattern)
        elif kind == 'router_timeout' and 'router_timeout' not in seen:
            hints.append(
                {
                    'kind': 'degrade_gracefully_after_router_timeout',
                    'reason': 'If router timeouts recur, rely more on bounded local continuation heuristics instead of dropping to generic chat.',
                }
            )
            seen.add('router_timeout')
        elif kind == 'runtime_issue' and 'runtime_issue' not in seen:
            hints.append(
                {
                    'kind': 'avoid_degraded_runtime_paths',
                    'reason': 'Prefer ready instances and surface degraded runtime conditions as soft routing constraints.',
                }
            )
            seen.add('runtime_issue')
        if len(hints) >= limit:
            break
    return hints


def _build_deep_memory(
    messages: list[dict[str, Any]],
    recent_artifacts: list[dict[str, Any]],
    *,
    self_observations: Optional[list[dict[str, Any]]] = None,
    self_healing_hints: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    continuity_artifacts = _build_continuity_artifacts(recent_artifacts)
    return {
        'learning_count': 0,
        'learnings': [],
        'stable_user_preferences': _build_stable_user_preferences(messages),
        'continuity_artifacts': continuity_artifacts,
        'self_observation_count': 0,
        'self_observations': [],
        'self_healing_hints': [],
        'sources': (
            [
                'conversation_history',
                *(['recent_artifacts'] if continuity_artifacts else []),
            ]
            if (messages or continuity_artifacts)
            else []
        ),
    }


def build_ghost_memory(
    *,
    messages: Iterable[dict[str, Any]],
    recent_artifacts: Iterable[dict[str, Any]],
    recent_events: Iterable[dict[str, Any]],
    conversation_id: Optional[str] = None,
    self_observations: Optional[Iterable[dict[str, Any]]] = None,
    self_healing_hints: Optional[Iterable[dict[str, Any]]] = None,
) -> dict[str, Any]:
    normalized_messages = [item for item in (_normalize_message(raw) for raw in messages or []) if item]
    normalized_artifacts = [item for item in (_normalize_artifact(raw) for raw in recent_artifacts or []) if item]
    return {
        'strategy': 'hot_first',
        'expansion_order': ['hot_memory', 'warm_memory', 'deep_memory'],
        'conversation_id': str(conversation_id or '').strip() or None,
        'hot_memory': _build_hot_memory(normalized_messages, normalized_artifacts),
        'warm_memory': _build_warm_memory(normalized_messages),
        'deep_memory': _build_deep_memory(
            normalized_messages,
            normalized_artifacts,
            self_observations=[item for item in self_observations or [] if isinstance(item, dict)],
            self_healing_hints=[item for item in self_healing_hints or [] if isinstance(item, dict)],
        ),
    }
