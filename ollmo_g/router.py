"""Ghost route hints, embedding helpers, and instance-selection helpers for Ollmo."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    SUPPORTED_CAPABILITIES,
    normalize_backend,
    normalize_capability,
    supports_capability,
)
from ollmo_services.chat_history import read_chat_history
from ollmo_g.intent import (
    analyze_prompt_intent,
    normalize_intent_text,
    prompt_has_self_contained_direct_tts_source,
)
from ollmo_services.file_inputs import file_kind_from_name

MAX_REFERENTIAL_ROUTE_USER_TURNS = 1
MAX_RECENT_MESSAGES = 8
MAX_RECENT_MESSAGE_FALLBACK = 4
MAX_EMBEDDING_HINTS = 3
EMBEDDING_BIAS_MIN_SCORE = 0.84
EMBEDDING_BIAS_MIN_MARGIN = 0.05
GHOST_POLICY_PATH = Path(__file__).resolve().parent.parent / 'GHOST.md'
_DEFAULT_GHOST_RUNTIME_POLICY = (
    'Ghost is Ollmo runtime routing only. '
    'Trust the provided runtime manifest, attachments, recent artifacts, and session-control truth. '
    'Do not invent instances or files. Prefer capability-level routing unless a listed instance is clearly needed. '
    'If unsure, choose chat. Return exactly one JSON object.'
)
_GHOST_POLICY_CACHE: dict[str, Any] = {
    'mtime_ns': None,
    'text': None,
}

_IMAGE_GENERATION_INTENT_RE = re.compile(
    r"^(\/image|\/img|image:|draw:|illustrate:|render:)"
    r"|(^|\b)(generate|create|make|render|draw|illustrate|produce)\b[\s\S]{0,80}\b(image|photo|portrait|picture|scene|shot)\b"
    r"|\b(generate me an image|create an image|make an image|render an image)\b"
    r"|(^|\b)(generiere|erstelle|erzeuge|mache|male|zeichne|rendere)\b[\s\S]{0,80}\b(bild|foto|porträt|portrait|szene|aufnahme)\b",
    re.IGNORECASE,
)
_REFERENCE_IMAGE_GENERATION_RE = re.compile(
    r"\b(reference image|uploaded image|use the face|using the face|from the uploaded image|from uploaded image|"
    r"face from the uploaded image|keep the face|verwende das bild|nutze das bild|verwende dieses bild|"
    r"mit diesem bild|mit dem bild|mach daraus|mache daraus|weiterverarbeitung(?:\s+(?:als|zu|zum|zur))?|"
    r"make this into|turn this into|transform this into|convert this into)\b",
    re.IGNORECASE,
)
_TEXT_TO_SPEECH_INTENT_RE = re.compile(
    r"^(\/tts|\/speak|tts:|speak:|read-aloud:|read aloud:)"
    r"|(^|\b)(text[\s-]*to[\s-]*speech|synthesize speech|read aloud|generate (me )?(an )?audio|"
    r"generate audio|create audio|make audio|audio of this text|voiceover audio|audio voiceover)\b"
    r"|(^|\b)(vorlesen|lies(?:\s+\w+){0,2}\s+vor|sprich(?:\s+\w+){0,2}\s+vor|vertonen|generiere (mir )?(ein )?audio|"
    r"erzeuge (mir )?(ein )?audio)\b",
    re.IGNORECASE,
)
_EXISTING_IMAGE_REFERENCE_RE = re.compile(
    r"\b(this image|that image|the image|this picture|that picture|the picture|this photo|that photo|"
    r"the photo|screenshot|screen|full picture|dieses bild|das bild|dieses foto|dieses screenshot|"
    r"den screenshot|vollständige bild|ganze bild)\b"
    r"|\b(describe what you see|what do you see|what is in this image|what's in this image|read the text|"
    r"what does it say|all of it|describe context|was siehst du|was steht|lies den text|beschreibe)\b",
    re.IGNORECASE,
)
_TEXT_ARTIFACT_REFERENCE_RE = re.compile(
    r"\b(ocr|ocr result|that ocr|this ocr|transcript|transcription|that transcript|this transcript|"
    r"that text|this text|that markdown|this markdown|continue from|continue with|continue using|"
    r"work from|use that result|nutze das ergebnis|dieses ergebnis|jenes ergebnis|dieser text|"
    r"dieses transcript|dieses transkript|transkript|ocr-ergebnis)\b",
    re.IGNORECASE,
)
_AUDIO_ARTIFACT_REFERENCE_RE = re.compile(
    r"\b(this audio|that audio|the audio|that recording|this recording|"
    r"transcribe it|transcribe this|speech to text|stt|dieses audio|diese aufnahme|transkribiere)\b",
    re.IGNORECASE,
)
_DIRECT_VISUAL_CREATION_RE = re.compile(
    r"(^|\b)(draw|illustrate|paint|sketch|zeichne)\b"
    r"|(^|\b)male\b(?=\s+(?:mir|bitte|doch|mal|schnell|einen?|eine|den|die|das)\b)",
    re.IGNORECASE,
)
_VISUAL_OUTPUT_HINT_RE = re.compile(
    r"\b(image|img|photo|photograph|picture|portrait|scene|shot|poster|wallpaper|logo|icon|banner|cover|thumbnail|"
    r"sticker|meme|illustration|artwork|drawing|sketch|painting|avatar|headshot|selfie|comic|flyer|concept art)\b",
    re.IGNORECASE,
)
_TEXT_OR_AUDIO_OUTPUT_RE = re.compile(
    r"\b(poem|story|essay|article|email|letter|summary|outline|plan|code|json|yaml|xml|sql|regex|script|function|"
    r"markdown|table|list|audio|speech|voiceover|tts|transcribe|transcript|ocr|translation|translate)\b",
    re.IGNORECASE,
)
_DRAW_UP_RE = re.compile(r"\bdraw\s+up\b", re.IGNORECASE)
_FOLLOW_UP_CONTINUATION_RE = re.compile(
    r"\b(again|same (?:as|style|one|thing|prompt|scene|voice)|this time|make it|do it|turn it|"
    r"calmer|softer|warmer|male voice|female voice|auf deutsch|in german|german version|"
    r"deutsche version|männliche stimme|mannliche stimme|maennliche stimme|weibliche stimme|ruhiger|"
    r"nochmal|diesmal|mach daraus|"
    r"mach es|mache es|weiterverarbeitung)\b",
    re.IGNORECASE,
)
_FRESH_TASK_RE = re.compile(
    r"\b(new task|fresh task|start fresh|from scratch|ignore previous|ignore the previous|"
    r"unrelated(?:\s+(?:to|from))?\s+(?:the\s+)?previous|completely new task|completly new task)\b",
    re.IGNORECASE,
)
_IMAGE_EDIT_PREFIX_RE = re.compile(
    r"^\s*(?:make|turn|keep|change|transform|convert|render|give|set|mache?|mach|wandle|transformiere|gib)\b",
    re.IGNORECASE,
)
_IMAGE_EDIT_CONTINUATION_RE = re.compile(
    r"\b(add|remove|keep|change|adjust|move|reposition|refine|improve|stronger|more|less|"
    r"powerful|cinematic|dramatic|hero|superhero|weapon|weapons|background|foreground|"
    r"main character|subject|hold|holding|wield|pose|"
    r"specific prompt|before generating|instead|not just|wear|wearing|dress|outfit|clothes|"
    r"clothing|shirt|jacket|coat|cape|gown|armor|armour|hair|hairstyle|eyes?|face|skin|"
    r"color|colour|blue|red|green|black|white|gold|golden|silver)\b",
    re.IGNORECASE,
)
_IMAGE_EDIT_IDENTITY_RE = re.compile(
    r"\b(robot|humanoid|character|subject|figure|creature|monster|alien|entity|armor|armour|eyes?|"
    r"face|head|body|person|man|woman|girl|boy|her|him|she|he|dress|outfit|clothes|clothing|"
    r"shirt|jacket|coat|cape|gown|hair|hairstyle|skin)\b",
    re.IGNORECASE,
)
_IMAGE_EDIT_REFERENCE_RE = re.compile(
    r"\b(it|its|itself|them|their|theirs|they|him|his|himself|her|hers|herself|he|she|"
    r"main character|subject|figure|creature|monster|alien|entity|robot|humanoid|character|"
    r"person|man|woman|girl|boy|face|head|body|pose|scene|background|foreground|picture|"
    r"image|photo|shot|outfit|clothes|clothing|dress|shirt|jacket|coat|cape|gown|hair|"
    r"hairstyle|eyes?|skin|weapon|weapons)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_REFERENCE_RE = re.compile(
    r"\b(it|its|them|their|they|him|his|her|hers|he|she|daraus|darauf|davon|"
    r"transcript|transcription|ocr|result|output|image|picture|photo|screenshot|audio|recording|"
    r"text|markdown|reply|response|message)\b",
    re.IGNORECASE,
)
_PROMPT_CAPABILITY_THRESHOLD = 4
_PROMPT_CAPABILITY_TIEBREAK = {
    CAPABILITY_TEXT_TO_SPEECH: 4,
    CAPABILITY_IMAGE_GENERATION: 3,
    CAPABILITY_SPEECH_TO_TEXT: 2,
    CAPABILITY_VISION_ANALYSIS: 1,
}


def _clip(text: Any, *, max_chars: int = 700) -> str:
    value = str(text or '').strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + '...[truncated]'


def _load_runtime_ghost_policy() -> str:
    path = GHOST_POLICY_PATH
    try:
        stat_result = path.stat()
    except OSError:
        return _DEFAULT_GHOST_RUNTIME_POLICY

    cached_mtime_ns = _GHOST_POLICY_CACHE.get('mtime_ns')
    cached_text = _GHOST_POLICY_CACHE.get('text')
    if cached_text and cached_mtime_ns == stat_result.st_mtime_ns:
        return str(cached_text)

    try:
        raw_text = path.read_text(encoding='utf-8').strip()
    except OSError:
        return _DEFAULT_GHOST_RUNTIME_POLICY

    policy_text = raw_text or _DEFAULT_GHOST_RUNTIME_POLICY
    _GHOST_POLICY_CACHE['mtime_ns'] = stat_result.st_mtime_ns
    _GHOST_POLICY_CACHE['text'] = policy_text
    return policy_text


def _readiness_rank(value: object) -> int:
    token = str(value or '').strip().lower()
    if token == 'ready':
        return 3
    if token in {'started', 'idle'}:
        return 2
    if token == 'degraded':
        return 1
    if token in {'failed', 'unreachable', 'stopped'}:
        return 0
    return 1


def _activity_rank(value: object) -> int:
    token = str(value or '').strip().lower()
    if token in {'idle', 'ready'}:
        return 2
    if token in {'busy', 'working'}:
        return 1
    return 0


def _compact_string_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        token = str(raw or '').strip()
        if token and token not in items:
            items.append(token)
        if len(items) >= limit:
            break
    return items


_ROUTING_DYNAMIC_TRAIT_SKIP_KEYS = {
    'activity',
    'backend',
    'backend_contract',
    'backend_metadata',
    'backend_package',
    'backend_runtime',
    'capability',
    'canonical_responses',
    'direct_responses',
    'feature_sources',
    'features',
    'inputs',
    'instance_id',
    'last_error',
    'model',
    'modelname',
    'output_modality',
    'outputs',
    'path',
    'pid',
    'port',
    'provider_capabilities',
    'readiness',
    'request_model',
    'routing_summary',
    'runtime_status',
    'session_controls',
    'session_controls_summary',
    'supported_capabilities',
    'text_capable',
}


def _compact_dynamic_trait_value(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        return token[:80]
    if isinstance(value, list):
        items = _compact_string_list(value, limit=10)
        return items or None
    return None


def _summarize_dynamic_model_traits(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    traits: dict[str, Any] = {}
    for raw_key, raw_value in entry.items():
        key = str(raw_key or '').strip()
        if not key:
            continue
        lowered = key.lower()
        if lowered in _ROUTING_DYNAMIC_TRAIT_SKIP_KEYS:
            continue
        compact_value = _compact_dynamic_trait_value(raw_value)
        if compact_value is None:
            continue
        traits[key] = compact_value
        if len(traits) >= 14:
            break
    return traits


def _summarize_session_controls(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    fields = schema.get('fields') if isinstance(schema.get('fields'), dict) else {}
    visible_fields: list[str] = []
    required_fields: list[str] = []
    labels: list[str] = []
    field_types: dict[str, str] = {}
    field_options: dict[str, list[str]] = {}
    for field_key, field in fields.items():
        if not isinstance(field, dict) or field.get('visible') is False:
            continue
        key = str(field_key or '').strip()
        label = str(field.get('label') or key).strip()
        if key:
            visible_fields.append(key)
        if label and label not in labels:
            labels.append(label)
        if field.get('required') and key:
            required_fields.append(key)
        field_type = str(field.get('type') or field.get('kind') or '').strip()
        if key and field_type:
            field_types[key] = field_type
        options = [
            str(item or '').strip()
            for item in (field.get('options') or [])
            if str(item or '').strip()
        ]
        if key and options:
            field_options[key] = options[:12]
    return {
        'enabled': bool(schema.get('enabled')),
        'hint': str(schema.get('hint') or '').strip() or None,
        'visible_fields': visible_fields[:12],
        'required_fields': required_fields[:8],
        'labels': labels[:12],
        'field_types': field_types,
        'field_options': field_options,
    }


def _summarize_backend_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        'source': str(metadata.get('source') or '').strip() or None,
        'package_label': str(metadata.get('package_label') or '').strip() or None,
        'capabilities': _compact_string_list(metadata.get('capabilities')),
        'instance_capabilities': _compact_string_list(metadata.get('instance_capabilities')),
        'package_capabilities': _compact_string_list(metadata.get('package_capabilities')),
        'runtime_constraints': _compact_string_list(metadata.get('runtime_constraints')),
        'runtime_knobs': _compact_string_list(metadata.get('runtime_knobs')),
        'native_endpoint_paths': _compact_string_list(metadata.get('native_endpoint_paths')),
        'lazy_loads_model': bool(metadata.get('lazy_loads_model')) if 'lazy_loads_model' in metadata else None,
        'single_loaded_model': bool(metadata.get('single_loaded_model')) if 'single_loaded_model' in metadata else None,
        'supports_unload': bool(metadata.get('supports_unload')) if 'supports_unload' in metadata else None,
        'shim_kind': str(metadata.get('shim_kind') or '').strip() or None,
    }


def _summarize_backend_runtime(runtime: Any) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        return {}
    endpoint_urls = sorted(
        key
        for key, value in runtime.items()
        if key.endswith('_url') and value
    )
    return {
        'source': str(runtime.get('source') or '').strip() or None,
        'native_base_url': str(runtime.get('native_base_url') or '').strip() or None,
        'request_model_strategy': str(runtime.get('request_model_strategy') or '').strip() or None,
        'runtime_knobs': _compact_string_list(runtime.get('runtime_knobs')),
        'lazy_loads_model': bool(runtime.get('lazy_loads_model')) if 'lazy_loads_model' in runtime else None,
        'single_loaded_model': bool(runtime.get('single_loaded_model')) if 'single_loaded_model' in runtime else None,
        'supports_unload': bool(runtime.get('supports_unload')) if 'supports_unload' in runtime else None,
        'shim_kind': str(runtime.get('shim_kind') or '').strip() or None,
        'endpoint_urls': endpoint_urls[:10],
    }


def _instance_routing_summary(instance: dict[str, Any]) -> dict[str, Any]:
    existing = instance.get('routing_summary')
    if isinstance(existing, dict) and existing:
        return existing
    runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), dict) else {}
    backend_runtime = runtime_status.get('backend_runtime') if isinstance(runtime_status.get('backend_runtime'), dict) else {}
    features = instance.get('features') if isinstance(instance.get('features'), dict) else {}
    return {
        'backend_package': str(instance.get('backend_package') or '').strip() or None,
        'backend_contract': str(instance.get('backend_contract') or '').strip() or None,
        'provider_capabilities': _compact_string_list(instance.get('provider_capabilities')),
        'feature_flags': sorted(key for key, value in features.items() if value),
        'dynamic_model_traits': _summarize_dynamic_model_traits(instance),
        'session_controls': _summarize_session_controls(instance.get('session_controls')),
        'tts_model_type': str(instance.get('tts_model_type') or '').strip() or None,
        'tts_languages': _compact_string_list(instance.get('tts_languages')),
        'tts_speakers': _compact_string_list(instance.get('tts_speakers')),
        'backend_metadata': _summarize_backend_metadata(instance.get('backend_metadata')),
        'backend_runtime': _summarize_backend_runtime(backend_runtime),
    }


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), dict) else {}
    backend_metadata = instance.get('backend_metadata') if isinstance(instance.get('backend_metadata'), dict) else {}
    backend_details = backend_metadata.get('details') if isinstance(backend_metadata.get('details'), dict) else {}
    return {
        'instance_id': str(instance.get('instance_id') or '').strip(),
        'model': str(instance.get('model') or instance.get('modelName') or '').strip(),
        'backend': normalize_backend(instance.get('backend')),
        'capability': normalize_capability(instance.get('capability')),
        'target_kind': str(instance.get('target_kind') or 'local').strip().lower(),
        'lifecycle_managed': instance.get('lifecycle_managed')
        if isinstance(instance.get('lifecycle_managed'), bool)
        else True,
        'backend_package': str(instance.get('backend_package') or '').strip() or None,
        'backend_contract': str(instance.get('backend_contract') or '').strip() or None,
        'port': instance.get('port'),
        'request_model': str(instance.get('request_model') or '').strip() or None,
        'readiness': str(runtime_status.get('readiness') or instance.get('readiness') or '').strip() or None,
        'activity': str(runtime_status.get('activity') or instance.get('activity') or '').strip() or None,
        'last_error': str(runtime_status.get('last_error') or instance.get('last_error') or '').strip() or None,
        'features': instance.get('features') if isinstance(instance.get('features'), dict) else {},
        'provider_capabilities': _compact_string_list(instance.get('provider_capabilities')),
        'inputs': instance.get('inputs') if isinstance(instance.get('inputs'), list) else [],
        'outputs': instance.get('outputs') if isinstance(instance.get('outputs'), list) else [],
        'dynamic_model_traits': _summarize_dynamic_model_traits(instance),
        'context_length': (
            instance.get('context_length')
            or backend_metadata.get('context_length')
            or runtime_status.get('context_length')
            or None
        ),
        'model_family': (
            str(backend_details.get('family') or '').strip()
            or str(backend_metadata.get('architecture') or '').strip()
            or None
        ),
        'parameter_size': str(backend_details.get('parameter_size') or '').strip() or None,
        'quantization_level': str(backend_details.get('quantization_level') or '').strip() or None,
        'size_bytes': (
            runtime_status.get('size_vram')
            or runtime_status.get('size')
            or instance.get('size_vram')
            or instance.get('size')
            or None
        ),
        'tts_model_type': str(instance.get('tts_model_type') or '').strip() or None,
        'tts_languages': _compact_string_list(instance.get('tts_languages')),
        'tts_speakers': _compact_string_list(instance.get('tts_speakers')),
        'routing_summary': _instance_routing_summary(instance),
    }


def _instance_supports_embedding(instance: dict[str, Any]) -> bool:
    capability = normalize_capability(instance.get('capability'))
    if capability == CAPABILITY_EMBEDDING:
        return True
    provider_capabilities = {
        normalize_capability(item)
        for item in (instance.get('provider_capabilities') or [])
        if str(item or '').strip()
    }
    if CAPABILITY_EMBEDDING in provider_capabilities:
        return True
    outputs = {
        str(item or '').strip().lower()
        for item in (instance.get('outputs') or [])
        if str(item or '').strip()
    }
    return bool(outputs & {'embedding', 'embeddings', 'vector', 'vectors'})


def _instance_embedding_transport(instance: dict[str, Any]) -> Optional[str]:
    if not _instance_supports_embedding(instance):
        return None
    backend = normalize_backend(instance.get('backend'))
    if backend == 'ollama':
        return 'ollama_api_embed'
    routing_summary = instance.get('routing_summary') if isinstance(instance.get('routing_summary'), dict) else {}
    backend_metadata = routing_summary.get('backend_metadata') if isinstance(routing_summary.get('backend_metadata'), dict) else {}
    native_paths = {
        str(item or '').strip()
        for item in (backend_metadata.get('native_endpoint_paths') or [])
        if str(item or '').strip()
    }
    if native_paths & {'/v1/embeddings', '/embeddings', '/api/embeddings'}:
        return 'openai_embeddings'
    return None


def _sanitize_message(message: dict[str, Any]) -> Optional[dict[str, Any]]:
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
        ('saved_image_path', 'saved_image_path'),
        ('savedImagePath', 'saved_image_path'),
        ('saved_text_path', 'saved_text_path'),
        ('savedTextPath', 'saved_text_path'),
        ('saved_audio_path', 'saved_audio_path'),
        ('savedAudioPath', 'saved_audio_path'),
        ('response_model', 'response_model'),
        ('responseModel', 'response_model'),
        ('response_backend', 'response_backend'),
        ('responseBackend', 'response_backend'),
        ('response_instance_id', 'response_instance_id'),
        ('responseInstanceId', 'response_instance_id'),
        ('route_source', 'route_source'),
        ('routeSource', 'route_source'),
        ('route_reason', 'route_reason'),
        ('routeReason', 'route_reason'),
        ('context_mode', 'context_mode'),
        ('contextMode', 'context_mode'),
        ('context_reason', 'context_reason'),
        ('contextReason', 'context_reason'),
    ):
        value = str(message.get(source_key) or '').strip()
        if value:
            payload[target_key] = value
    artifacts = message.get('artifacts')
    if isinstance(artifacts, list):
        sanitized_artifacts: list[dict[str, Any]] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_type = str(artifact.get('type') or '').strip()
            path = str(artifact.get('path') or '').strip()
            if not artifact_type or not path:
                continue
            sanitized_artifact = {
                'type': artifact_type,
                'path': path,
            }
            seed = artifact.get('seed')
            if isinstance(seed, int) and seed >= 0:
                sanitized_artifact['seed'] = seed
            image_state = artifact.get('image_state')
            if isinstance(image_state, dict) and image_state:
                sanitized_artifact['image_state'] = image_state
            sanitized_artifacts.append(sanitized_artifact)
        if sanitized_artifacts:
            payload['artifacts'] = sanitized_artifacts
    if not payload['content'] and not any(payload.get(key) for key in ('saved_image_path', 'saved_text_path', 'saved_audio_path')):
        return None
    return payload


def sanitize_ghost_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in messages or []:
        cleaned = _sanitize_message(item)
        if cleaned:
            sanitized.append(cleaned)
    return sanitized


def _merge_ghost_messages(*message_sets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_index: dict[tuple[Any, ...], int] = {}
    for message_set in message_sets:
        for item in sanitize_ghost_messages(message_set):
            key = (
                item.get('role'),
                item.get('content'),
                item.get('timestamp'),
                item.get('saved_image_path'),
                item.get('saved_text_path'),
                item.get('saved_audio_path'),
                item.get('response_model'),
                item.get('response_instance_id'),
                item.get('route_source'),
                item.get('route_reason'),
                item.get('context_mode'),
            )
            if key in seen_index:
                existing = merged[seen_index[key]]
                if not existing.get('artifacts') and item.get('artifacts'):
                    existing['artifacts'] = item.get('artifacts')
                continue
            seen_index[key] = len(merged)
            merged.append(item)
    return merged


def _hydrate_conversation_messages(
    conversation_id: Optional[str],
    request_messages: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized_request_messages = sanitize_ghost_messages(request_messages)
    conversation_key = str(conversation_id or '').strip()
    if not conversation_key:
        return sanitized_request_messages
    stored_history = read_chat_history(conversation_key)
    stored_messages = stored_history.get('messages') if isinstance(stored_history.get('messages'), list) else []
    return _merge_ghost_messages(stored_messages, sanitized_request_messages)


def _compact_route_message(item: dict[str, Any]) -> dict[str, Any]:
    return {
        'role': item.get('role'),
        'content': _clip(item.get('content') or ''),
        'timestamp': item.get('timestamp'),
        'saved_image_path': item.get('saved_image_path'),
        'saved_text_path': item.get('saved_text_path'),
        'saved_audio_path': item.get('saved_audio_path'),
    }


def _message_artifacts(message: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact in message.get('artifacts') or []:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get('path') or '').strip()
        artifact_type = str(artifact.get('type') or '').strip()
        if not path or not artifact_type or path in seen_paths:
            continue
        artifact_payload = {
            'type': artifact_type,
            'path': path,
            'timestamp': message.get('timestamp'),
            'role': message.get('role'),
        }
        seed = artifact.get('seed')
        if isinstance(seed, int) and seed >= 0:
            artifact_payload['seed'] = seed
        if isinstance(artifact.get('image_state'), dict) and artifact.get('image_state'):
            artifact_payload['image_state'] = artifact.get('image_state')
        artifacts.append(artifact_payload)
        seen_paths.add(path)
    for artifact_type, key in (
        ('image', 'saved_image_path'),
        ('text', 'saved_text_path'),
        ('audio', 'saved_audio_path'),
    ):
        path = str(message.get(key) or '').strip()
        if not path or path in seen_paths:
            continue
        artifacts.append(
            {
                'type': artifact_type,
                'path': path,
                'timestamp': message.get('timestamp'),
                'role': message.get('role'),
            }
        )
        seen_paths.add(path)
    return artifacts


def select_recent_route_messages(
    messages: Iterable[dict[str, Any]],
    *,
    prompt: str,
    max_user_turns: int = MAX_REFERENTIAL_ROUTE_USER_TURNS,
) -> list[dict[str, Any]]:
    sanitized = sanitize_ghost_messages(messages)
    if not sanitized:
        return []

    raw_prompt = str(prompt or '').strip()
    if not raw_prompt or _signals_fresh_task(raw_prompt):
        return []

    normalized_prompt = normalize_intent_text(raw_prompt)
    if not (
        _has_follow_up_reference_anchor(raw_prompt, normalized_prompt)
        or _FOLLOW_UP_CONTINUATION_RE.search(raw_prompt)
    ):
        return []

    user_positions = [
        index
        for index, item in enumerate(sanitized)
        if str(item.get('role') or '').strip().lower() == 'user'
    ]
    if not user_positions:
        return [_compact_route_message(item) for item in sanitized[-MAX_RECENT_MESSAGE_FALLBACK:]]

    current_user_position: Optional[int] = None
    last_user_position = user_positions[-1]
    last_user_content = normalize_intent_text(str(sanitized[last_user_position].get('content') or '').strip())
    if last_user_content and last_user_content == normalized_prompt:
        current_user_position = last_user_position

    historical_user_positions = [
        position
        for position in user_positions
        if current_user_position is None or position < current_user_position
    ]
    if not historical_user_positions:
        return []

    answered_turns: list[tuple[int, list[int]]] = []
    for user_position in historical_user_positions:
        next_user_position = next(
            (position for position in user_positions if position > user_position),
            len(sanitized),
        )
        assistant_positions = [
            cursor
            for cursor in range(user_position + 1, next_user_position)
            if str(sanitized[cursor].get('role') or '').strip().lower() == 'assistant'
        ]
        if assistant_positions:
            answered_turns.append((user_position, assistant_positions))

    selected_positions: set[int] = set()
    if answered_turns:
        for user_position, assistant_positions in answered_turns[-max_user_turns:]:
            selected_positions.add(user_position)
            selected_positions.update(assistant_positions)
    else:
        selected_positions.add(historical_user_positions[-1])

    return [_compact_route_message(sanitized[index]) for index in sorted(selected_positions)]


def extract_recent_artifacts(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = sanitize_ghost_messages(messages)
    if not sanitized:
        return []

    artifacts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_types: set[str] = set()
    latest_artifact_message_index: Optional[int] = None

    for index in range(len(sanitized) - 1, -1, -1):
        message = sanitized[index]
        if str(message.get('role') or '').strip().lower() != 'assistant':
            continue
        message_artifacts = _message_artifacts(message)
        if not message_artifacts:
            continue
        latest_artifact_message_index = index
        for artifact in message_artifacts:
            path = str(artifact.get('path') or '').strip()
            artifact_type = str(artifact.get('type') or '').strip()
            if not path or not artifact_type or path in seen_paths:
                continue
            artifacts.append(artifact)
            seen_paths.add(path)
            seen_types.add(artifact_type)
        break

    for index in range(len(sanitized) - 1, -1, -1):
        if latest_artifact_message_index is not None and index == latest_artifact_message_index:
            continue
        for artifact in _message_artifacts(sanitized[index]):
            path = str(artifact.get('path') or '').strip()
            artifact_type = str(artifact.get('type') or '').strip()
            if not path or not artifact_type or path in seen_paths or artifact_type in seen_types:
                continue
            artifacts.append(artifact)
            seen_paths.add(path)
            seen_types.add(artifact_type)

    return artifacts


def _latest_artifact_by_type(recent_artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for artifact in recent_artifacts:
        artifact_type = str(artifact.get('type') or '').strip()
        if artifact_type and artifact_type not in latest:
            latest[artifact_type] = artifact
    return latest


def build_route_context(
    *,
    prompt: str,
    upload_filename: str,
    file_path: str,
    conversation_id: Optional[str],
    messages: Iterable[dict[str, Any]],
    runtime_manifest: dict[str, Any],
    ghost_payload: dict[str, Any],
    instances: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    stored_history = read_chat_history(str(conversation_id or '').strip()) if str(conversation_id or '').strip() else {}
    sanitized_messages = _hydrate_conversation_messages(conversation_id, messages)
    intent = analyze_prompt_intent(prompt)
    thread_context_requested = _prompt_requests_thread_context(prompt)
    recent_messages = (
        select_recent_route_messages(
            sanitized_messages,
            prompt=str(prompt or '').strip(),
        )
        if thread_context_requested
        else []
    )
    recent_artifacts = extract_recent_artifacts(sanitized_messages) if thread_context_requested else []
    latest_artifacts = _latest_artifact_by_type(recent_artifacts)
    explicit_file_name = str(upload_filename or '').strip() or str(file_path or '').strip()
    explicit_file_kind = file_kind_from_name(explicit_file_name) if explicit_file_name else ''

    manifest_capabilities = runtime_manifest.get('capabilities') if isinstance(runtime_manifest.get('capabilities'), dict) else {}
    manifest_instances = runtime_manifest.get('instances') if isinstance(runtime_manifest.get('instances'), list) else []
    manifest_external_targets = (
        runtime_manifest.get('external_targets')
        if isinstance(runtime_manifest.get('external_targets'), list)
        else []
    )
    selectable_external_targets = [
        item
        for item in manifest_external_targets
        if isinstance(item, dict) and item.get('selectable') is True
    ]

    available_capabilities = {
        str(capability): {
            'default_instance_id': entry.get('default_instance_id'),
            'count': entry.get('count'),
            'candidates': [
                {
                    'instance_id': candidate.get('instance_id'),
                    'model': candidate.get('model'),
                    'backend': candidate.get('backend'),
                    'backend_package': candidate.get('backend_package'),
                    'backend_contract': candidate.get('backend_contract'),
                    'provider_capabilities': candidate.get('provider_capabilities'),
                    'readiness': candidate.get('readiness'),
                    'activity': candidate.get('activity'),
                    'inputs': candidate.get('inputs') if isinstance(candidate.get('inputs'), list) else [],
                    'outputs': candidate.get('outputs') if isinstance(candidate.get('outputs'), list) else [],
                    'dynamic_model_traits': candidate.get('dynamic_model_traits') if isinstance(candidate.get('dynamic_model_traits'), dict) else {},
                    'tts_model_type': str(candidate.get('tts_model_type') or '').strip() or None,
                    'tts_languages': _compact_string_list(candidate.get('tts_languages')),
                    'tts_speakers': _compact_string_list(candidate.get('tts_speakers')),
                    'session_controls_summary': (
                        candidate.get('session_controls_summary')
                        if isinstance(candidate.get('session_controls_summary'), dict)
                        else {}
                    ),
                    'routing_summary': candidate.get('routing_summary') if isinstance(candidate.get('routing_summary'), dict) else {},
                }
                for candidate in (entry.get('candidates') or [])[:4]
                if isinstance(candidate, dict)
            ],
        }
        for capability, entry in manifest_capabilities.items()
        if isinstance(entry, dict)
        and normalize_capability(capability) != CAPABILITY_EMBEDDING
    }
    for target in selectable_external_targets:
        capability = normalize_capability(target.get('capability'))
        instance_id = str(target.get('instance_id') or target.get('id') or '').strip()
        if not capability or not instance_id:
            continue
        capability_entry = available_capabilities.setdefault(
            capability,
            {
                'default_instance_id': None,
                'count': 0,
                'candidates': [],
            },
        )
        external_candidate = {
            'instance_id': instance_id,
            'model': target.get('model'),
            'backend': target.get('backend'),
            'target_kind': 'external',
            'lifecycle_managed': False,
            'provider_capabilities': target.get('provider_capabilities') or [],
            'readiness': target.get('readiness'),
            'activity': target.get('activity'),
            'inputs': target.get('inputs') if isinstance(target.get('inputs'), list) else [],
            'outputs': target.get('outputs') if isinstance(target.get('outputs'), list) else [],
            'dynamic_model_traits': {},
            'session_controls_summary': {},
            'routing_summary': {
                'target_kind': 'external',
                'text_only': bool(target.get('text_only')),
            },
        }
        capability_entry['candidates'] = [
            *list(capability_entry.get('candidates') or []),
            external_candidate,
        ]
        capability_entry['count'] = int(capability_entry.get('count') or 0) + 1
        if not capability_entry.get('default_instance_id'):
            capability_entry['default_instance_id'] = instance_id

    available_instances = [
        {
            'instance_id': str(item.get('instance_id') or '').strip(),
            'model': str(item.get('model') or '').strip(),
            'backend': normalize_backend(item.get('backend')),
            'capability': normalize_capability(item.get('capability')),
            'target_kind': str(item.get('target_kind') or 'local').strip().lower(),
            'lifecycle_managed': item.get('lifecycle_managed')
            if isinstance(item.get('lifecycle_managed'), bool)
            else True,
            'backend_package': str(item.get('backend_package') or '').strip() or None,
            'backend_contract': str(item.get('backend_contract') or '').strip() or None,
            'provider_capabilities': _compact_string_list(item.get('provider_capabilities')),
            'readiness': item.get('readiness'),
            'activity': item.get('activity'),
            'inputs': item.get('inputs') if isinstance(item.get('inputs'), list) else [],
            'outputs': item.get('outputs') if isinstance(item.get('outputs'), list) else [],
            'dynamic_model_traits': item.get('dynamic_model_traits') if isinstance(item.get('dynamic_model_traits'), dict) else {},
            'tts_model_type': str(item.get('tts_model_type') or '').strip() or None,
            'tts_languages': _compact_string_list(item.get('tts_languages')),
            'tts_speakers': _compact_string_list(item.get('tts_speakers')),
            'session_controls_summary': item.get('session_controls_summary') if isinstance(item.get('session_controls_summary'), dict) else {},
            'routing_summary': item.get('routing_summary') if isinstance(item.get('routing_summary'), dict) else {},
        }
        for item in [*manifest_instances, *selectable_external_targets][:12]
        if isinstance(item, dict)
        and normalize_capability(item.get('capability')) != CAPABILITY_EMBEDDING
    ]

    return {
        'prompt': str(prompt or '').strip(),
        'conversation_id': str(conversation_id or '').strip() or None,
        'conversation_metadata': (
            stored_history.get('conversation_metadata')
            if isinstance(stored_history.get('conversation_metadata'), dict)
            else {}
        ),
        'request_attachment': {
            'upload_filename': str(upload_filename or '').strip() or None,
            'file_path': str(file_path or '').strip() or None,
            'file_kind': explicit_file_kind or None,
            'has_explicit_file': bool(explicit_file_name),
        },
        'recent_messages': recent_messages,
        'recent_artifacts': recent_artifacts,
        'latest_artifacts': latest_artifacts,
        'intent': {
            'primary_capability': intent.get('primary_capability'),
            'capability_scores': intent.get('capability_scores'),
            'capability_cues': intent.get('capability_cues'),
            'language_codes': intent.get('language_codes'),
            'voice_descriptors': intent.get('voice_descriptors'),
            'audio_response_format': intent.get('audio_response_format'),
            'image_aspect_ratio': intent.get('image_aspect_ratio'),
            'text_preparation_before_audio_output': bool(intent.get('text_preparation_before_audio_output')),
            'text_first_follow_up_capability': normalize_capability(intent.get('text_first_follow_up_capability')),
            'temperament_hint': str(intent.get('temperament_hint') or '').strip().lower() or None,
            'temperament_cues': intent.get('temperament_cues') if isinstance(intent.get('temperament_cues'), list) else [],
        },
        'runtime': {
            'available_capabilities': available_capabilities,
            'available_instances': available_instances,
            'ghost_recommendations': ghost_payload.get('recommendations') if isinstance(ghost_payload.get('recommendations'), list) else [],
            'ghost_issues': ghost_payload.get('issues') if isinstance(ghost_payload.get('issues'), list) else [],
            'accepted_learning_hints': (
                ghost_payload.get('accepted_learning_hints')
                if isinstance(ghost_payload.get('accepted_learning_hints'), dict)
                else {'enabled': False, 'hint_count': 0, 'hints': [], 'runtime_effect': 'none'}
            ),
            'intake_context': {
                'thread_context_requested': thread_context_requested,
                'history_binding': 'referential' if thread_context_requested else 'current_turn_only',
            },
        },
        'instances': [_normalize_instance(item) for item in instances if isinstance(item, dict)],
    }


def _parse_parameter_size_billions(value: Any) -> Optional[float]:
    token = str(value or '').strip()
    if not token:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*([bBmM])\b', token)
    if not match:
        return None
    magnitude = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == 'm':
        return round(magnitude / 1000.0, 6)
    return round(magnitude, 6)


def _infer_model_size_billions_from_name(model_name: str) -> Optional[float]:
    token = str(model_name or '').strip()
    if not token:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b', token)
    if match:
        return round(float(match.group(1)), 6)
    match = re.search(r'[-_:](\d+(?:\.\d+)?)b(?:\W|$)', token, re.IGNORECASE)
    if match:
        return round(float(match.group(1)), 6)
    return None


def _router_compactness_score(instance: dict[str, Any]) -> tuple[int, float, float, int]:
    parameter_size_b = (
        _parse_parameter_size_billions(instance.get('parameter_size'))
        or _infer_model_size_billions_from_name(str(instance.get('model') or ''))
    )
    if parameter_size_b is not None:
        parameter_rank = 2
        parameter_value = -float(parameter_size_b)
    else:
        parameter_rank = 0
        parameter_value = 0.0

    size_bytes = instance.get('size_bytes')
    size_rank = 1 if isinstance(size_bytes, (int, float)) and float(size_bytes) > 0 else 0
    size_value = -(float(size_bytes) / float(1024 ** 3)) if size_rank else 0.0

    family_bonus = 1 if str(instance.get('model_family') or '').strip() else 0
    return parameter_rank, parameter_value, size_value, family_bonus


def _router_context_budget_rank(instance: dict[str, Any]) -> int:
    raw_value = instance.get('context_length')
    try:
        context_length = int(raw_value or 0)
    except (TypeError, ValueError):
        context_length = 0
    if context_length >= 16384:
        return 1
    return 0


def _router_candidate_score(instance: dict[str, Any]) -> tuple[int, int, int, int, float, float, int, int, int, str]:
    features = instance.get('features') if isinstance(instance.get('features'), dict) else {}
    structured_outputs = 1 if features.get('structured_outputs') else 0
    function_calling = 1 if features.get('function_calling') else 0
    tool_calling = 1 if features.get('tool_calling') else 0
    backend_preference = 1 if normalize_backend(instance.get('backend')) == 'mlx' else 0
    compactness_rank, compactness_value, size_value, family_bonus = _router_compactness_score(instance)
    return (
        _readiness_rank(instance.get('readiness')),
        _activity_rank(instance.get('activity')),
        structured_outputs,
        _router_context_budget_rank(instance),
        compactness_value,
        size_value,
        family_bonus + compactness_rank,
        function_calling,
        tool_calling,
        backend_preference,
        str(instance.get('instance_id') or ''),
    )


def select_router_instance(instances: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw in instances:
        if not isinstance(raw, dict):
            continue
        instance = _normalize_instance(raw)
        if instance.get('capability') != CAPABILITY_CHAT:
            continue
        if not instance.get('instance_id') or not instance.get('port'):
            continue
        if _readiness_rank(instance.get('readiness')) <= 0:
            continue
        candidates.append(instance)
    if not candidates:
        return None
    return sorted(candidates, key=_router_candidate_score, reverse=True)[0]


def _normalize_stable_target_preference(
    value: Any,
    *,
    capability: Optional[str] = None,
    role: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    model = str(value.get('model') or '').strip()
    backend = normalize_backend(value.get('backend'))
    normalized_capability = normalize_capability(value.get('capability') or capability)
    normalized_role = str(value.get('role') or role or '').strip().lower() or None
    if not model and not backend:
        return None
    payload: dict[str, Any] = {}
    if model:
        payload['model'] = model
    if backend:
        payload['backend'] = backend
    if normalized_capability:
        payload['capability'] = normalized_capability
    if normalized_role:
        payload['role'] = normalized_role
    return payload or None


def _instance_matches_stable_target_preference(instance: dict[str, Any], preference: Optional[dict[str, Any]]) -> bool:
    normalized_preference = _normalize_stable_target_preference(preference)
    if not normalized_preference:
        return False
    preferred_model = str(normalized_preference.get('model') or '').strip()
    preferred_backend = normalize_backend(normalized_preference.get('backend'))
    if preferred_model and str(instance.get('model') or '').strip() != preferred_model:
        return False
    if preferred_backend and normalize_backend(instance.get('backend')) != preferred_backend:
        return False
    preferred_capability = normalize_capability(normalized_preference.get('capability'))
    if preferred_capability and not supports_capability(
        preferred_capability,
        model_name=instance.get('model'),
        backend=instance.get('backend'),
        capability=instance.get('capability'),
        metadata=instance,
    ):
        return False
    return True


def select_embedding_instance(
    instances: Iterable[dict[str, Any]],
    preferred_target: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw in instances:
        if not isinstance(raw, dict):
            continue
        instance = _normalize_instance(raw)
        if not instance.get('instance_id') or not instance.get('port'):
            continue
        if _readiness_rank(instance.get('readiness')) <= 0:
            continue
        transport = _instance_embedding_transport(instance)
        if not transport:
            continue
        instance['embedding_transport'] = transport
        candidates.append(instance)
    if not candidates:
        return None
    normalized_preference = _normalize_stable_target_preference(
        preferred_target,
        capability=CAPABILITY_EMBEDDING,
        role='embedding_helper',
    )
    if normalized_preference:
        preferred_candidates = [
            item for item in candidates
            if _instance_matches_stable_target_preference(item, normalized_preference)
        ]
        if preferred_candidates:
            candidates = preferred_candidates
    return sorted(
        candidates,
        key=lambda item: (
            _readiness_rank(item.get('readiness')),
            _activity_rank(item.get('activity')),
            1 if normalize_capability(item.get('capability')) == CAPABILITY_EMBEDDING else 0,
            1 if item.get('embedding_transport') == 'ollama_api_embed' else 0,
            str(item.get('instance_id') or ''),
        ),
        reverse=True,
    )[0]


def build_embedding_route_candidates(
    *,
    runtime_manifest: dict[str, Any],
    instances: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    manifest_capabilities = runtime_manifest.get('capabilities') if isinstance(runtime_manifest.get('capabilities'), dict) else {}

    for capability, entry in manifest_capabilities.items():
        normalized_capability = normalize_capability(capability)
        if normalized_capability not in SUPPORTED_CAPABILITIES or normalized_capability == CAPABILITY_EMBEDDING:
            continue
        if not isinstance(entry, dict):
            continue
        aliases = [str(item).strip() for item in (entry.get('aliases') or []) if str(item).strip()]
        default_instance_id = str(entry.get('default_instance_id') or '').strip() or None
        candidate_models = [
            str(item.get('model') or '').strip()
            for item in (entry.get('candidates') or [])
            if isinstance(item, dict) and str(item.get('model') or '').strip()
        ][:4]
        descriptor = ' '.join(
            part
            for part in (
                f'capability {normalized_capability}',
                f'aliases {" ".join(aliases)}' if aliases else '',
                f'default {default_instance_id}' if default_instance_id else '',
                f'models {" ".join(candidate_models)}' if candidate_models else '',
                f'packages {" ".join(_compact_string_list([candidate.get("backend_package") for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict)]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
                f'contracts {" ".join(_compact_string_list([candidate.get("backend_contract") for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict)]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
                f'required_controls {" ".join(_compact_string_list([required for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict) for required in ((candidate.get("session_controls_summary") or {}).get("required_fields") or [])]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
                f'visible_controls {" ".join(_compact_string_list([field for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict) for field in ((candidate.get("session_controls_summary") or {}).get("visible_fields") or [])]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
                f'dynamic_traits {" ".join(_compact_string_list([f"{key}:{value}" for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict) for key, value in (candidate.get("dynamic_model_traits") or {}).items()]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
                f'tts_models {" ".join(_compact_string_list([candidate.get("tts_model_type") for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict)]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
                f'tts_languages {" ".join(_compact_string_list([language for candidate in (entry.get("candidates") or []) if isinstance(candidate, dict) for language in (candidate.get("tts_languages") or [])]))}'
                if isinstance(entry.get('candidates'), list)
                else '',
            )
            if part
        )
        candidates.append(
            {
                'kind': 'capability',
                'key': normalized_capability,
                'capability': normalized_capability,
                'default_instance_id': default_instance_id,
                'text': descriptor,
            }
        )

    for raw in instances:
        if not isinstance(raw, dict):
            continue
        instance = _normalize_instance(raw)
        capability = normalize_capability(instance.get('capability'))
        if capability not in SUPPORTED_CAPABILITIES or capability == CAPABILITY_EMBEDDING:
            continue
        if _readiness_rank(instance.get('readiness')) <= 0:
            continue
        enabled_features = sorted(
            key
            for key, value in (instance.get('features') or {}).items()
            if value
        )
        inputs = [str(item).strip() for item in (instance.get('inputs') or []) if str(item).strip()]
        outputs = [str(item).strip() for item in (instance.get('outputs') or []) if str(item).strip()]
        routing_summary = instance.get('routing_summary') if isinstance(instance.get('routing_summary'), dict) else {}
        backend_metadata = routing_summary.get('backend_metadata') if isinstance(routing_summary.get('backend_metadata'), dict) else {}
        backend_runtime = routing_summary.get('backend_runtime') if isinstance(routing_summary.get('backend_runtime'), dict) else {}
        session_controls = routing_summary.get('session_controls') if isinstance(routing_summary.get('session_controls'), dict) else {}
        dynamic_model_traits = routing_summary.get('dynamic_model_traits') if isinstance(routing_summary.get('dynamic_model_traits'), dict) else {}
        field_types = session_controls.get('field_types') if isinstance(session_controls.get('field_types'), dict) else {}
        field_options = session_controls.get('field_options') if isinstance(session_controls.get('field_options'), dict) else {}
        dynamic_trait_tokens = [f'{key}:{value}' for key, value in dynamic_model_traits.items()]
        control_type_tokens = [f'{key}:{value}' for key, value in field_types.items()]
        control_option_tokens = [f'{key}:{"/".join(values)}' for key, values in field_options.items() if values]
        descriptor = ' '.join(
            part
            for part in (
                f'instance {instance.get("instance_id")}',
                f'model {instance.get("model")}',
                f'backend {instance.get("backend")}',
                f'package {instance.get("backend_package")}' if instance.get('backend_package') else '',
                f'contract {instance.get("backend_contract")}' if instance.get('backend_contract') else '',
                f'capability {capability}',
                f'provider_capabilities {" ".join(instance.get("provider_capabilities") or [])}' if instance.get('provider_capabilities') else '',
                f'inputs {" ".join(inputs)}' if inputs else '',
                f'outputs {" ".join(outputs)}' if outputs else '',
                f'features {" ".join(enabled_features)}' if enabled_features else '',
                f'package_label {backend_metadata.get("package_label")}' if backend_metadata.get('package_label') else '',
                f'runtime_constraints {" ".join(backend_metadata.get("runtime_constraints") or [])}' if backend_metadata.get('runtime_constraints') else '',
                f'runtime_knobs {" ".join(backend_metadata.get("runtime_knobs") or [])}' if backend_metadata.get('runtime_knobs') else '',
                f'native_endpoints {" ".join(backend_metadata.get("native_endpoint_paths") or [])}' if backend_metadata.get('native_endpoint_paths') else '',
                f'request_model_strategy {backend_runtime.get("request_model_strategy")}' if backend_runtime.get('request_model_strategy') else '',
                f'required_controls {" ".join(session_controls.get("required_fields") or [])}' if session_controls.get('required_fields') else '',
                f'visible_controls {" ".join(session_controls.get("visible_fields") or [])}' if session_controls.get('visible_fields') else '',
                f'control_labels {" ".join(session_controls.get("labels") or [])}' if session_controls.get('labels') else '',
                f'dynamic_traits {" ".join(dynamic_trait_tokens)}' if dynamic_trait_tokens else '',
                f'control_types {" ".join(control_type_tokens)}' if control_type_tokens else '',
                f'control_options {" ".join(control_option_tokens)}' if control_option_tokens else '',
                f'tts_model_type {instance.get("tts_model_type")}' if instance.get('tts_model_type') else '',
                f'tts_languages {" ".join(instance.get("tts_languages") or [])}' if instance.get('tts_languages') else '',
                f'tts_speakers {" ".join(instance.get("tts_speakers") or [])}' if instance.get('tts_speakers') else '',
            )
            if part
        )
        candidates.append(
            {
                'kind': 'instance',
                'key': str(instance.get('instance_id') or '').strip(),
                'instance_id': str(instance.get('instance_id') or '').strip(),
                'capability': capability,
                'model': str(instance.get('model') or '').strip(),
                'backend': normalize_backend(instance.get('backend')),
                'text': descriptor,
            }
        )
    return candidates


def _normalize_embedding_vector(value: Any) -> Optional[list[float]]:
    if not isinstance(value, list) or not value:
        return None
    normalized: list[float] = []
    for raw in value:
        if not isinstance(raw, (int, float)):
            return None
        normalized.append(float(raw))
    return normalized if normalized else None


def _cosine_similarity(left: list[float], right: list[float]) -> Optional[float]:
    if not left or not right or len(left) != len(right):
        return None
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


def build_embedding_hints_from_vectors(
    prompt_vector: list[float],
    candidates: list[dict[str, Any]],
    candidate_vectors: list[list[float]],
    *,
    limit: int = MAX_EMBEDDING_HINTS,
) -> Optional[dict[str, Any]]:
    normalized_prompt = _normalize_embedding_vector(prompt_vector)
    if not normalized_prompt or len(candidates) != len(candidate_vectors):
        return None

    capability_matches: list[dict[str, Any]] = []
    instance_matches: list[dict[str, Any]] = []
    for candidate, raw_vector in zip(candidates, candidate_vectors):
        normalized_vector = _normalize_embedding_vector(raw_vector)
        if not normalized_vector:
            continue
        score = _cosine_similarity(normalized_prompt, normalized_vector)
        if score is None:
            continue
        payload = {
            'score': round(float(score), 4),
            'capability': candidate.get('capability'),
        }
        if candidate.get('kind') == 'capability':
            payload['default_instance_id'] = candidate.get('default_instance_id')
            capability_matches.append(payload)
        elif candidate.get('kind') == 'instance':
            payload['instance_id'] = candidate.get('instance_id')
            payload['model'] = candidate.get('model')
            payload['backend'] = candidate.get('backend')
            instance_matches.append(payload)

    capability_matches.sort(key=lambda item: (float(item.get('score') or 0.0), str(item.get('capability') or '')), reverse=True)
    instance_matches.sort(key=lambda item: (float(item.get('score') or 0.0), str(item.get('instance_id') or '')), reverse=True)

    if not capability_matches and not instance_matches:
        return None

    return {
        'top_capabilities': capability_matches[:limit],
        'top_instances': instance_matches[:limit],
    }


def _top_embedding_capability_hint(context: dict[str, Any]) -> tuple[Optional[dict[str, Any]], float]:
    runtime = context.get('runtime') if isinstance(context.get('runtime'), dict) else {}
    embedding_hints = runtime.get('embedding_hints') if isinstance(runtime.get('embedding_hints'), dict) else {}
    top_capabilities = embedding_hints.get('top_capabilities') if isinstance(embedding_hints.get('top_capabilities'), list) else []
    runner_up_score = 0.0
    if len(top_capabilities) > 1 and isinstance(top_capabilities[1], dict):
        runner_up_score = float(top_capabilities[1].get('score') or 0.0)
    if not top_capabilities or not isinstance(top_capabilities[0], dict):
        return None, 0.0
    top_hint = dict(top_capabilities[0])
    top_hint['capability'] = normalize_capability(top_hint.get('capability'))
    return top_hint, max(0.0, float(top_hint.get('score') or 0.0) - runner_up_score)


def _top_embedding_instance_hint(context: dict[str, Any]) -> Optional[dict[str, Any]]:
    runtime = context.get('runtime') if isinstance(context.get('runtime'), dict) else {}
    embedding_hints = runtime.get('embedding_hints') if isinstance(runtime.get('embedding_hints'), dict) else {}
    top_instances = embedding_hints.get('top_instances') if isinstance(embedding_hints.get('top_instances'), list) else []
    if not top_instances or not isinstance(top_instances[0], dict):
        return None
    return dict(top_instances[0])


def _reference_artifacts(context: dict[str, Any]) -> list[dict[str, Any]]:
    payload = (
        context.get('reference_artifacts')
        if isinstance(context.get('reference_artifacts'), list)
        else (
            context.get('selected_reference_artifacts')
            if isinstance(context.get('selected_reference_artifacts'), list)
            else []
        )
    )
    return [item for item in payload if isinstance(item, dict)]


def _primary_reference_artifact(context: dict[str, Any]) -> Optional[dict[str, Any]]:
    explicit_reference = (
        context.get('selected_reference_artifact')
        if isinstance(context.get('selected_reference_artifact'), dict)
        else None
    )
    if explicit_reference:
        return explicit_reference
    references = _reference_artifacts(context)
    if not references:
        return None
    return next(
        (item for item in references if str(item.get('type') or '').strip().lower() != 'message'),
        references[0],
    )


def _embedding_prompt_class(
    context: dict[str, Any],
    *,
    route_hint: Optional[dict[str, Any]] = None,
    heuristic_route: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    if route_hint is None and isinstance(heuristic_route, dict):
        route_hint = heuristic_route
    prompt = str(context.get('prompt') or '').strip()
    if not prompt or _signals_fresh_task(prompt):
        return None
    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}
    latest_image = latest_artifacts.get('image') if isinstance(latest_artifacts.get('image'), dict) else None
    latest_text = latest_artifacts.get('text') if isinstance(latest_artifacts.get('text'), dict) else None
    latest_audio = latest_artifacts.get('audio') if isinstance(latest_artifacts.get('audio'), dict) else None
    selected_reference_artifact = _primary_reference_artifact(context)
    selected_reference_type = str((selected_reference_artifact or {}).get('type') or '').strip().lower()
    recent_capability = _recent_user_capability_context(context, current_prompt=prompt)
    lowered_prompt = prompt.lower()
    has_follow_up_cue = bool(_FOLLOW_UP_CONTINUATION_RE.search(prompt))

    image_anchor = selected_reference_type == 'image' or bool(latest_image and _references_existing_image(lowered_prompt))
    if image_anchor and (
        recent_capability == CAPABILITY_IMAGE_GENERATION
        or _is_image_edit_follow_up(prompt, context=context, latest_image=latest_image)
    ):
        return 'image_edit_follow_up'

    text_anchor = bool(latest_text and _references_recent_text(lowered_prompt))
    if text_anchor and (recent_capability == CAPABILITY_TEXT_TO_SPEECH or has_follow_up_cue):
        return 'text_to_speech_follow_up'

    audio_anchor = bool(latest_audio and _references_recent_audio(lowered_prompt))
    if audio_anchor and (recent_capability == CAPABILITY_SPEECH_TO_TEXT or has_follow_up_cue):
        return 'speech_to_text_follow_up'

    route_hint_capability = normalize_capability((route_hint or {}).get('capability'))
    if route_hint_capability == CAPABILITY_CHAT and image_anchor and has_follow_up_cue:
        return 'image_edit_follow_up'
    return None


def infer_route_prompt_class(
    context: dict[str, Any],
    *,
    route_hint: Optional[dict[str, Any]] = None,
    heuristic_route: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    if route_hint is None and isinstance(heuristic_route, dict):
        route_hint = heuristic_route
    prompt = str(context.get('prompt') or '').strip()
    if not prompt or _signals_fresh_task(prompt):
        return None
    if prompt_has_self_contained_direct_tts_source(prompt):
        return None
    specific_prompt_class = _embedding_prompt_class(context, route_hint=route_hint)
    if specific_prompt_class:
        return specific_prompt_class
    normalized_prompt = normalize_intent_text(prompt)
    if _has_follow_up_reference_anchor(prompt, normalized_prompt):
        return 'artifact_follow_up'
    recent_messages = context.get('recent_messages') if isinstance(context.get('recent_messages'), list) else []
    if recent_messages and _FOLLOW_UP_CONTINUATION_RE.search(prompt):
        return 'thread_follow_up'
    return None


def infer_route_session_class(context: dict[str, Any]) -> str:
    conversation_id = str(context.get('conversation_id') or '').strip().lower()
    conversation_metadata = (
        context.get('conversation_metadata')
        if isinstance(context.get('conversation_metadata'), dict)
        else {}
    )
    prompt = str(context.get('prompt') or '').strip()
    normalized_prompt = normalize_intent_text(prompt) if prompt else ''
    selected_reference_artifact = _primary_reference_artifact(context)
    selected_reference_artifacts = _reference_artifacts(context)
    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}
    explicit_selected_reference = bool(selected_reference_artifact or selected_reference_artifacts)
    if explicit_selected_reference:
        return 'artifact_chain'
    if (
        prompt
        and not _signals_fresh_task(prompt)
        and (
            _has_follow_up_reference_anchor(prompt, normalized_prompt)
            or _FOLLOW_UP_CONTINUATION_RE.search(prompt)
        )
        and (
            latest_artifacts.get('image')
            or latest_artifacts.get('text')
            or latest_artifacts.get('audio')
        )
    ):
        return 'artifact_chain'

    if bool(conversation_metadata.get('fresh_root')):
        if conversation_id or (context.get('recent_messages') if isinstance(context.get('recent_messages'), list) else []):
            return 'threaded_session'
        return 'ephemeral_session'

    if conversation_id and 'workbench' in conversation_id:
        return 'workbench_session'

    recent_messages = context.get('recent_messages') if isinstance(context.get('recent_messages'), list) else []
    if conversation_id or recent_messages:
        return 'threaded_session'
    return 'ephemeral_session'


def collect_routing_preferences(
    context: dict[str, Any],
    *,
    route_hint: Optional[dict[str, Any]] = None,
    heuristic_route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if route_hint is None and isinstance(heuristic_route, dict):
        route_hint = heuristic_route
    prompt_class = infer_route_prompt_class(context, route_hint=route_hint)
    session_class = infer_route_session_class(context)

    return {
        'prompt_class': prompt_class,
        'session_class': session_class,
        'matched_policy_ids': [],
    }


def build_route_memory_scope(
    context: dict[str, Any],
    *,
    route_hint: Optional[dict[str, Any]] = None,
    heuristic_route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if route_hint is None and isinstance(heuristic_route, dict):
        route_hint = heuristic_route
    routing_preferences = collect_routing_preferences(context, route_hint=route_hint)
    return {
        'prompt_class': routing_preferences.get('prompt_class'),
        'session_class': routing_preferences.get('session_class'),
        'routing_preferences': routing_preferences,
    }



def maybe_apply_embedding_route_bias(
    context: dict[str, Any],
    route_hint: Optional[dict[str, Any]] = None,
    heuristic_route: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if route_hint is None and isinstance(heuristic_route, dict):
        route_hint = heuristic_route
    if not isinstance(context, dict) or not isinstance(route_hint, dict):
        return None
    if normalize_capability(route_hint.get('capability')) != CAPABILITY_CHAT:
        return None

    prompt_class = _embedding_prompt_class(context, route_hint=route_hint)
    if not prompt_class:
        return None

    top_hint, score_gap = _top_embedding_capability_hint(context)
    if not top_hint:
        return None

    top_capability = normalize_capability(top_hint.get('capability'))
    top_score = float(top_hint.get('score') or 0.0)
    expected_capability = {
        'image_edit_follow_up': CAPABILITY_IMAGE_GENERATION,
        'text_to_speech_follow_up': CAPABILITY_TEXT_TO_SPEECH,
        'speech_to_text_follow_up': CAPABILITY_SPEECH_TO_TEXT,
    }.get(prompt_class)
    if top_capability != expected_capability:
        return None
    if top_score < EMBEDDING_BIAS_MIN_SCORE or score_gap < EMBEDDING_BIAS_MIN_MARGIN:
        return None

    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}
    request_attachment = context.get('request_attachment') if isinstance(context.get('request_attachment'), dict) else {}
    has_explicit_file = bool(request_attachment.get('has_explicit_file'))
    artifact = None
    if expected_capability == CAPABILITY_IMAGE_GENERATION:
        artifact = latest_artifacts.get('image') if isinstance(latest_artifacts.get('image'), dict) else None
    elif expected_capability == CAPABILITY_TEXT_TO_SPEECH:
        artifact = latest_artifacts.get('text') if isinstance(latest_artifacts.get('text'), dict) else None
    elif expected_capability == CAPABILITY_SPEECH_TO_TEXT:
        artifact = latest_artifacts.get('audio') if isinstance(latest_artifacts.get('audio'), dict) else None
    artifact_path = str((artifact or {}).get('path') or '').strip() or None

    confidence = min(0.94, max(0.86, top_score - 0.01 + score_gap / 2.0))
    return {
        'capability': expected_capability,
        'instance_id': None,
        'reuse_last_artifact': bool(artifact_path and not has_explicit_file),
        'artifact_path': artifact_path if artifact_path and not has_explicit_file else None,
        'confidence': round(confidence, 4),
        'reason': f'embedding tie-break for {prompt_class}',
    }


def build_embedding_route_audit(
    context: dict[str, Any],
    route_hint: Optional[dict[str, Any]] = None,
    final_route: Optional[dict[str, Any]] = None,
    *,
    bias_applied: bool = False,
    heuristic_route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if route_hint is None and isinstance(heuristic_route, dict):
        route_hint = heuristic_route
    runtime = context.get('runtime') if isinstance(context.get('runtime'), dict) else {}
    helper = runtime.get('embedding_helper') if isinstance(runtime.get('embedding_helper'), dict) else {}
    top_capability_hint, score_gap = _top_embedding_capability_hint(context)
    top_instance_hint = _top_embedding_instance_hint(context)
    route_hint_capability = normalize_capability((route_hint or {}).get('capability'))
    final_capability = normalize_capability((final_route or {}).get('capability'))
    prompt_class = _embedding_prompt_class(context, route_hint=route_hint)
    audit = {
        'available': bool(helper.get('available')),
        'attached': bool(helper.get('attached')),
        'status': 'unavailable',
        'prompt_class': prompt_class,
        'route_hint_capability': route_hint_capability or None,
        'route_hint_confidence': round(float((route_hint or {}).get('confidence') or 0.0), 4),
        'final_capability': final_capability or None,
        'bias_applied': bool(bias_applied),
        'embedding_helper_instance_id': str(helper.get('instance_id') or '').strip() or None,
        'embedding_model': str(helper.get('model') or '').strip() or None,
    }
    if not audit['available']:
        return audit
    if not top_capability_hint:
        audit['status'] = 'no_hints' if audit['attached'] else 'helper_available'
        return audit

    top_capability = normalize_capability(top_capability_hint.get('capability'))
    audit.update(
        {
            'embedding_capability': top_capability or None,
            'embedding_score': round(float(top_capability_hint.get('score') or 0.0), 4),
            'embedding_score_gap': round(float(score_gap or 0.0), 4),
            'embedding_default_instance_id': str(top_capability_hint.get('default_instance_id') or '').strip() or None,
            'embedding_instance_id': str((top_instance_hint or {}).get('instance_id') or '').strip() or None,
            'embedding_instance_score': round(float((top_instance_hint or {}).get('score') or 0.0), 4) if top_instance_hint else None,
        }
    )
    if top_capability == final_capability:
        audit['status'] = 'biased_alignment' if bias_applied else 'aligned'
    else:
        audit['status'] = 'diverged'
    return audit


def build_router_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
        'capability': 'chat|vision_analysis|image_generation|speech_to_text|text_to_speech',
        'instance_id': 'string|null',
        'reuse_last_artifact': 'boolean',
        'artifact_path': 'string|null',
        'confidence': 'number from 0.0 to 1.0',
        'reason': 'short string',
        'workload_task_proposals': (
            'advisory semantic annotations and bounded execution_contract candidates for existing '
            'runtime.request_phase_graph.workload_graph tasks; expected for multi-task or dependency work. '
            'May include advisory_role, evidence_requirements, reconsideration_triggers, '
            'semantic_review_criteria, promotion_suggestions, waiver_candidates, repair_candidates, '
            'supersession_candidates, semantic quality review hints, recursive cycle notes, semantic decision review inputs, '
            'controlled attention targets, and learning_hint_refs.'
        ),
    }
    runtime_policy = _load_runtime_ghost_policy()
    system_prompt = (
        'You are ollmo-ghost-router inside Ollmo. '
        'Your job is runtime routing only. '
        'Do not invent executable work, new phases, files, or instances; do not add commentary. '
        'Use runtime.request_phase_graph.decision_contract as the role/decision boundary when present: '
        'Ghost may propose candidates, task annotations, reconsideration, repair, semantic review, and supersession, '
        'but Runtime/Contracts decide promotion, execution, truth, waiver, supersession, and freeze. '
        'When decision_contract.semantic_planning_contract is present, use its planning cycle, proposal requirements, '
        'proposal obligations, and non-authority boundaries as the semantic rubric for this turn. '
        'When decision_contract.block_resolution_reflex is present, treat open, blocked, reserved, waived, '
        'superseded, repair, and review signals as current relevance inputs; propose the right-sized verified '
        'transition and do not force completion by rewriting intent. '
        'When decision_contract.active_reconsideration_review is present, use its decisions as the live '
        'relevance surface: decide what should be reviewed next, but do not promote, waive, supersede, or '
        'freeze anything yourself. '
        'When decision_contract.semantic_quality_review is required, express quality checks as review work '
        'bound to evidence and branch-local criteria; do not claim subjective quality is proven by output existence. '
        'Branch-level semantic review is demand-gated: use it when a specific branch should be reviewed, not for '
        'every branch, and return the same structured semantic_review_verdict contract when reviewing it. '
        'When decision_contract.recursive_cycle_review is present, apply the same prepare, gather evidence, '
        'execute, verify, repair-or-freeze thinking to each subtask, not only the root request. '
        'When decision_contract.semantic_decision_review is present, treat its proposals as advisory next-transition '
        'inputs with reason, confidence, evidence refs, and learning orientation; do not treat them as promotion, '
        'waiver, supersession, fulfillment, or freeze authority. '
        'When decision_contract.controlled_attention_review is present, use its frames as scoped attention targets '
        'between execution steps: answer the bounded attention question for that branch, task, candidate, or review '
        'surface only; do not replay the root prompt, do not start unpromoted work, and do not treat attention as '
        'execution permission. '
        'When runtime.graph_closure_review.global_semantic_closure_review is present, treat it as whole-turn '
        'semantic review evidence: local branch completion still has to fit the full current intent, but only '
        'Closure/Runtime may promote the semantic-review branch or freeze truth. If asked for semantic review, '
        'return a structured semantic_review_verdict with passed/failed/uncertain, criterion results, evidence '
        'refs, defects, confidence, and recommended transition; do not treat review completion itself as truth. '
        'When runtime.graph_closure_review.surface_state is present, treat it as UI-visible runtime projection '
        'for open, blocked, reconsiderable, waived, superseded, repair-pending, and semantic-review-pending state. '
        'When runtime.request_phase_graph.workload_graph contains multiple tasks or dependency edges, include '
        'workload_task_proposals for each existing executable task where semantic intent, input_refs, '
        'review_criteria, evidence requirements, promotion suggestions, waiver candidates, reconsideration triggers, '
        'repair candidates, supersession candidates, output_contract, or bounded execution_contract details clarify '
        'branch-local work. '
        'Use only existing task IDs or phase IDs already present in runtime.request_phase_graph.workload_graph. '
        'These proposals are advisory candidates and will be validated; never use them to change capability, '
        'dependencies, output type, visibility, required outputs, or executable topology. '
        'Return exactly one JSON object and nothing else. '
        'If you choose reuse_last_artifact=true, artifact_path must be one of the provided recent_artifacts paths. '
        'If an explicit uploaded file or file_path is present, reuse_last_artifact must be false. '
        'Prefer a capability-level choice unless a specific listed instance is clearly needed. '
        'When multiple live instances share one capability, prefer the one whose truthful visible controls, option lists, and model-specific metadata best fit the current request. '
        'If embedding_hints are present, treat them as soft routing signals rather than hard constraints. '
        'If accepted_learning_hints are present and enabled, treat them as reviewed soft hints only; they never override current runtime truth, explicit user intent, or available capability evidence. '
        'If request_meta contains ghost_mode, treat it as a compatibility alias for advisory semantic_role_profile wording only; capability_hint, language_hint, and developer_flags are explicit structured caller signals. '
        'Intent cues may include multilingual and indirect phrasing; treat them as soft evidence, not absolute rules. '
        'If intent.text_preparation_before_audio_output is true, the current turn still needs text work before any audio handoff; do not choose text_to_speech unless the spoken payload is already explicit. '
        'Use selected references, recent artifacts, and the latest assistant outputs as bounded context for current-turn references, not as hidden new intent. '
        'Only lean on the user\'s earlier turns when no stronger artifact or assistant-output anchor exists. '
        'Do not treat historical route corrections or prior capability wins as a deciding signal on their own. '
        'Always include capability, confidence, and reason. '
        'If unsure, choose capability="chat". '
        'Follow the repo Ghost runtime policy below.\n\n'
        f'{runtime_policy}'
    )
    user_prompt = json.dumps(
        {
            'schema': schema,
            'context': {
                'prompt': context.get('prompt'),
                'conversation_id': context.get('conversation_id'),
                'request_attachment': context.get('request_attachment'),
                'recent_messages': context.get('recent_messages'),
                'recent_artifacts': context.get('recent_artifacts'),
                'intent': context.get('intent'),
                'request_meta': context.get('request_meta'),
                'runtime': context.get('runtime'),
            },
        },
        ensure_ascii=False,
    )
    return [
        {'role': 'system', 'content': system_prompt},
        {
            'role': 'user',
            'content': (
                'Return JSON in exactly this shape:\n'
                '{"capability":"chat","instance_id":null,"reuse_last_artifact":false,'
                '"artifact_path":null,"confidence":0.72,"reason":"brief reason",'
                '"workload_task_proposals":[{"task_id":"existing-task-id","semantic_intent":"branch-local purpose",'
                '"evidence_requirements":[],"review_criteria":[],"reconsideration_triggers":[],'
                '"promotion_suggestions":[],"waiver_candidates":[],"repair_candidates":[],'
                '"supersession_candidates":[]}]}\n\n'
                f'{user_prompt}'
            ),
        },
    ]


def _normalize_router_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    for container_key in ('route', 'decision', 'result', 'selection'):
        nested = normalized.get(container_key)
        if isinstance(nested, dict):
            normalized = {**nested, **{k: v for k, v in normalized.items() if k not in {'route', 'decision', 'result', 'selection'}}}
            break

    artifact = normalized.get('artifact')
    if isinstance(artifact, dict):
        if 'artifact_path' not in normalized and artifact.get('path') is not None:
            normalized['artifact_path'] = artifact.get('path')
        if 'reuse_last_artifact' not in normalized and artifact.get('reuse') is not None:
            normalized['reuse_last_artifact'] = artifact.get('reuse')

    capability = (
        normalized.get('capability')
        or normalized.get('selected_capability')
        or normalized.get('target_capability')
        or normalized.get('mode')
        or normalized.get('intent')
        or normalized.get('task')
    )
    if capability is not None:
        normalized['capability'] = capability

    instance_id = (
        normalized.get('instance_id')
        or normalized.get('selected_instance_id')
        or normalized.get('target_instance_id')
        or normalized.get('instance')
    )
    if isinstance(instance_id, dict):
        instance_id = instance_id.get('instance_id') or instance_id.get('id')
    if instance_id is not None:
        normalized['instance_id'] = instance_id

    if 'reuse_last_artifact' not in normalized and normalized.get('artifact_path') not in (None, ''):
        normalized['reuse_last_artifact'] = True

    return normalized


def _extract_first_json_object(raw_text: str) -> Optional[str]:
    text = str(raw_text or '').strip()
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def parse_router_output(raw_text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    extracted = _extract_first_json_object(raw_text)
    if not extracted:
        return None, 'Router output did not contain a JSON object.'
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return None, f'Router output was not valid JSON: {exc}'
    if not isinstance(payload, dict):
        return None, 'Router output must be a JSON object.'
    return _normalize_router_payload(payload), None


def _available_instances_for_capability(instances: Iterable[dict[str, Any]], capability: str) -> list[dict[str, Any]]:
    normalized_capability = normalize_capability(capability)
    candidates: list[dict[str, Any]] = []
    for raw in instances:
        if not isinstance(raw, dict):
            continue
        instance = _normalize_instance(raw)
        if not supports_capability(
            normalized_capability,
            model_name=instance.get('model'),
            backend=instance.get('backend'),
            capability=instance.get('capability'),
            metadata=instance,
        ):
            continue
        if _readiness_rank(instance.get('readiness')) <= 0:
            continue
        candidates.append(instance)
    return candidates


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or '').strip().lower()
    return token in {'1', 'true', 'yes', 'on'}


def _coerce_confidence(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    token = str(value or '').strip()
    if not token:
        return None
    try:
        parsed = float(token)
    except ValueError:
        return None
    return max(0.0, min(1.0, parsed))


def _sanitize_route_string_list(value: Any, *, limit: int = 16, max_chars: int = 160) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, list):
        raw_items = list(value)
    else:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        text = _clip(raw_item, max_chars=max_chars)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _sanitize_route_mapping_list(value: Any, *, limit: int = 16) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, Any] = {}
        for key in ('kind', 'phase_id', 'task_id', 'ref', 'role', 'source'):
            text = _clip(raw_item.get(key), max_chars=120)
            if text:
                item[key] = text
        if item:
            cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def _sanitize_route_decision_mapping_list(value: Any, *, limit: int = 16) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    allowed_text_keys = (
        'candidate_id',
        'candidate_type',
        'task_id',
        'phase_id',
        'branch_id',
        'obligation_id',
        'action',
        'status',
        'reason',
        'evidence_ref',
        'target_ref',
        'promotion_target',
        'promotion_reason',
        'waiver_reason',
        'waiver_policy',
        'release_evidence',
        'repair_action',
        'recovery_action',
        'superseded_by',
        'superseded_by_candidate_id',
        'superseded_by_obligation_id',
        'supersession_reason',
    )
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, Any] = {}
        for key in allowed_text_keys:
            text = _clip(raw_item.get(key), max_chars=160)
            if text:
                item[key] = text
        evidence_refs = _sanitize_route_string_list(raw_item.get('evidence_refs'), limit=12, max_chars=160)
        if evidence_refs:
            item['evidence_refs'] = evidence_refs
        if item:
            cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def _sanitize_route_output_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    contract: dict[str, Any] = {}
    for key in ('output_type', 'fulfillment_policy'):
        text = _clip(value.get(key), max_chars=80)
        if text:
            contract[key] = text
    for key in ('required', 'requires_artifact'):
        if isinstance(value.get(key), bool):
            contract[key] = value.get(key)
    return contract


def _sanitize_route_ref_mapping(value: Any, *, keys: tuple[str, ...], max_chars: int = 120) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    item: dict[str, Any] = {}
    for key in keys:
        text = _clip(value.get(key), max_chars=max_chars)
        if text:
            item[key] = text
    return item


def _sanitize_route_artifact_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    request: dict[str, Any] = {}
    for key in ('extension', 'source_name', 'source'):
        text = _clip(value.get(key), max_chars=120)
        if text:
            request[key] = text
    return request


def _sanitize_route_execution_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    contract: dict[str, Any] = {}
    for key in (
        'kind',
        'branch_id',
        'phase_id',
        'task_id',
        'workload_task_id',
        'obligation_id',
        'capability',
        'output_type',
        'role',
        'stage_direction',
        'content_payload_source',
        'artifact_prompt_source',
        'text_artifact_extension',
        'text_artifact_source_name',
        'text_artifact_source',
    ):
        text = _clip(value.get(key), max_chars=160)
        if text:
            contract[key] = text
    for key in ('requires_artifact',):
        if isinstance(value.get(key), bool):
            contract[key] = value.get(key)
    depends_on = _sanitize_route_string_list(value.get('depends_on'), limit=24, max_chars=96)
    if depends_on:
        contract['depends_on'] = depends_on
    input_refs = _sanitize_route_mapping_list(value.get('input_refs'), limit=24)
    if input_refs:
        contract['input_refs'] = input_refs
    workload_task_ref = _sanitize_route_ref_mapping(
        value.get('workload_task_ref'),
        keys=('task_id', 'phase_id', 'branch_id'),
    )
    if workload_task_ref:
        contract['workload_task_ref'] = workload_task_ref
    output_obligation_ref = _sanitize_route_ref_mapping(
        value.get('output_obligation_ref'),
        keys=('obligation_id', 'phase_id', 'branch_id', 'output_type'),
    )
    if output_obligation_ref:
        contract['output_obligation_ref'] = output_obligation_ref
    output_contract = _sanitize_route_output_contract(value.get('output_contract'))
    if output_contract:
        contract['output_contract'] = output_contract
    artifact_request = _sanitize_route_artifact_request(value.get('artifact_request'))
    if artifact_request:
        contract['artifact_request'] = artifact_request
    return contract


def _sanitize_workload_task_proposals(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    proposals: list[dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        item: dict[str, Any] = {}
        for key in (
            'proposal_id',
            'id',
            'phase_id',
            'task_id',
            'workload_task_id',
            'branch_id',
            'capability',
            'output_type',
            'visibility',
            'semantic_intent',
            'intent',
            'task_intent',
            'objective',
            'deliverable',
            'advisory_role',
            'decision_notes',
            'promotion_policy',
            'reconsideration_policy',
            'rationale',
            'source',
            'stage_direction',
            'content_payload_source',
            'artifact_prompt_source',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
        ):
            text = _clip(raw_item.get(key), max_chars=320)
            if text:
                item[key] = text
        for key in (
            'depends_on',
            'parent_task_ids',
            'review_criteria',
            'acceptance_criteria',
            'evidence_requirements',
            'reconsideration_triggers',
            'semantic_review_criteria',
            'learning_hint_refs',
        ):
            values = _sanitize_route_string_list(raw_item.get(key), limit=24, max_chars=160)
            if values:
                item[key] = values
        repair_candidates = _sanitize_route_decision_mapping_list(raw_item.get('repair_candidates'), limit=16)
        if repair_candidates:
            item['repair_candidates'] = repair_candidates
        promotion_suggestions = _sanitize_route_decision_mapping_list(raw_item.get('promotion_suggestions'), limit=16)
        if promotion_suggestions:
            item['promotion_suggestions'] = promotion_suggestions
        waiver_candidates = _sanitize_route_decision_mapping_list(raw_item.get('waiver_candidates'), limit=16)
        if waiver_candidates:
            item['waiver_candidates'] = waiver_candidates
        supersession_candidates = _sanitize_route_decision_mapping_list(raw_item.get('supersession_candidates'), limit=16)
        if supersession_candidates:
            item['supersession_candidates'] = supersession_candidates
        input_refs = _sanitize_route_mapping_list(raw_item.get('input_refs'), limit=24)
        if input_refs:
            item['input_refs'] = input_refs
        output_contract = _sanitize_route_output_contract(raw_item.get('output_contract'))
        if output_contract:
            item['output_contract'] = output_contract
        execution_contract = _sanitize_route_execution_contract(raw_item.get('execution_contract'))
        if execution_contract:
            item['execution_contract'] = execution_contract
        artifact_request = _sanitize_route_artifact_request(raw_item.get('artifact_request'))
        if artifact_request:
            item['artifact_request'] = artifact_request
        if isinstance(raw_item.get('requires_artifact'), bool):
            item['requires_artifact'] = raw_item.get('requires_artifact')
        depth = raw_item.get('decomposition_level', raw_item.get('depth'))
        try:
            parsed_depth = int(depth)
        except (TypeError, ValueError):
            parsed_depth = None
        if parsed_depth is not None:
            item['decomposition_level'] = max(0, parsed_depth)
        if item:
            proposals.append(item)
        if len(proposals) >= 32:
            break
    return proposals


def _artifact_kind_from_path(path: str) -> str:
    token = str(path or '').strip()
    if not token:
        return ''
    return file_kind_from_name(Path(token).name)


def validate_route_decision(
    route: dict[str, Any],
    *,
    instances: Iterable[dict[str, Any]],
    recent_artifacts: Iterable[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    instance_id = str(route.get('instance_id') or '').strip() or None
    capability = normalize_capability(route.get('capability'))
    if not capability and instance_id:
        for raw in instances:
            if not isinstance(raw, dict):
                continue
            instance = _normalize_instance(raw)
            if instance.get('instance_id') == instance_id:
                capability = normalize_capability(instance.get('capability'))
                break
    if capability not in SUPPORTED_CAPABILITIES:
        return None, f"Unsupported capability '{route.get('capability')}'."

    reuse_last_artifact = _coerce_bool(route.get('reuse_last_artifact'))
    artifact_path = str(route.get('artifact_path') or '').strip() or None
    confidence = _coerce_confidence(route.get('confidence'))
    reason = _clip(route.get('reason') or '', max_chars=240)

    if confidence is None:
        return None, 'Router output confidence is missing or invalid.'
    if not reason:
        return None, 'Router output reason is missing.'

    recent_artifact_paths = {
        str(item.get('path') or '').strip()
        for item in recent_artifacts
        if isinstance(item, dict) and str(item.get('path') or '').strip()
    }

    if reuse_last_artifact:
        if not artifact_path:
            return None, 'Router requested artifact reuse without artifact_path.'
        if artifact_path not in recent_artifact_paths:
            return None, 'Router artifact_path is not one of the recent artifacts.'
        artifact_kind = _artifact_kind_from_path(artifact_path)
        if artifact_kind == 'image' and capability not in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_IMAGE_GENERATION}:
            return None, 'Image artifacts may only be reused for vision_analysis or image_generation.'
        if artifact_kind == 'audio' and capability != CAPABILITY_SPEECH_TO_TEXT:
            return None, 'Audio artifacts may only be reused for speech_to_text.'
        if artifact_kind in {'text', 'pdf'} and capability == CAPABILITY_IMAGE_GENERATION:
            return None, 'Text artifacts may not be reused for image_generation.'
    else:
        artifact_path = None

    candidates = _available_instances_for_capability(instances, capability)
    if not candidates:
        return None, f"No available instance for capability '{capability}'."

    if instance_id:
        chosen = next((item for item in candidates if item.get('instance_id') == instance_id), None)
        if not chosen:
            return None, f"Router selected unavailable instance '{instance_id}' for capability '{capability}'."

    validated = {
        'capability': capability,
        'instance_id': instance_id,
        'reuse_last_artifact': reuse_last_artifact,
        'artifact_path': artifact_path,
        'confidence': confidence,
        'reason': reason,
    }
    workload_task_proposals = _sanitize_workload_task_proposals(
        route.get('workload_task_proposals') or route.get('workload_tasks')
    )
    if workload_task_proposals:
        validated['workload_task_proposals'] = workload_task_proposals
    return (validated, None)


def _is_image_generation_intent(text: str) -> bool:
    normalized = normalize_intent_text(text)
    analysis = analyze_prompt_intent(text)
    if analysis.get('explicit_defer_materialization'):
        return False
    if _DRAW_UP_RE.search(normalized):
        return False
    if _TEXT_OR_AUDIO_OUTPUT_RE.search(normalized):
        return False
    selected_capability, _ = _select_prompt_capability(
        analysis,
        capabilities=(CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH),
    )
    if selected_capability == CAPABILITY_IMAGE_GENERATION:
        return True
    return bool(
        _IMAGE_GENERATION_INTENT_RE.search(normalized)
        or _DIRECT_VISUAL_CREATION_RE.search(normalized)
        or (
            _VISUAL_OUTPUT_HINT_RE.search(normalized)
            and re.search(
                r"(^|\b)(generate|create|make|produce|render|design|generiere|erstelle|erzeuge|mache|rendere)\b",
                normalized,
                re.IGNORECASE,
            )
        )
    )


def _is_reference_image_generation_intent(text: str) -> bool:
    return bool(_REFERENCE_IMAGE_GENERATION_RE.search(normalize_intent_text(text)))


def _is_text_to_speech_intent(text: str) -> bool:
    analysis = analyze_prompt_intent(text)
    if analysis.get('text_revision_turn') and not analysis.get('direct_audio_materialization_request'):
        return False
    if analysis.get('text_preparation_before_audio_output'):
        return False
    selected_capability, _ = _select_prompt_capability(
        analysis,
        capabilities=(CAPABILITY_TEXT_TO_SPEECH, CAPABILITY_IMAGE_GENERATION),
    )
    if selected_capability == CAPABILITY_TEXT_TO_SPEECH:
        return True
    return bool(_TEXT_TO_SPEECH_INTENT_RE.search(text))


def _references_existing_image(text: str) -> bool:
    return bool(_EXISTING_IMAGE_REFERENCE_RE.search(text))


def _references_recent_text(text: str) -> bool:
    return bool(_TEXT_ARTIFACT_REFERENCE_RE.search(text))


def _references_recent_audio(text: str) -> bool:
    return bool(_AUDIO_ARTIFACT_REFERENCE_RE.search(text))


def _signals_fresh_task(text: str) -> bool:
    return bool(_FRESH_TASK_RE.search(normalize_intent_text(text)))


def _has_image_edit_reference_anchor(raw_text: str, normalized_text: str) -> bool:
    if _references_existing_image(raw_text):
        return True
    return bool(_IMAGE_EDIT_REFERENCE_RE.search(normalized_text))


def _has_strong_image_edit_instruction(raw_text: str, normalized_text: str) -> bool:
    if not _IMAGE_EDIT_PREFIX_RE.search(raw_text):
        return False
    if _IMAGE_EDIT_CONTINUATION_RE.search(normalized_text):
        return True
    if _IMAGE_EDIT_IDENTITY_RE.search(normalized_text):
        return True
    if re.search(r'\b(unchanged|leave the rest|leave everything else|only change|just change)\b', normalized_text, re.IGNORECASE):
        return True
    return bool(re.search(r'\bchange\b.+\bto\b', normalized_text, re.IGNORECASE))


def _has_follow_up_reference_anchor(raw_text: str, normalized_text: str) -> bool:
    if _references_existing_image(raw_text):
        return True
    if _references_recent_text(raw_text):
        return True
    if _references_recent_audio(raw_text):
        return True
    if _IMAGE_EDIT_REFERENCE_RE.search(normalized_text):
        return True
    return bool(_FOLLOW_UP_REFERENCE_RE.search(normalized_text))


def _prompt_requests_thread_context(raw_text: str) -> bool:
    text = str(raw_text or '').strip()
    if not text or _signals_fresh_task(text):
        return False
    if prompt_has_self_contained_direct_tts_source(text):
        return False
    normalized_text = normalize_intent_text(text)
    if _references_existing_image(text) or _references_recent_text(text) or _references_recent_audio(text):
        return True
    if _has_strong_image_edit_instruction(text, normalized_text):
        return True
    if _FOLLOW_UP_CONTINUATION_RE.search(text):
        return True
    if re.search(
        r"\b(previous|prior|earlier|above|last|recent|before|as discussed|as mentioned|"
        r"vorher|vorhin|davor|oben|letzte|letzten|letzter|wie gesagt|wie besprochen)\b",
        normalized_text,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"\b(use|reuse|edit|change|revise|continue|redo|read|speak|show|turn|convert|"
            r"summarize|translate|compare|fix|make)\s+(?:it|that|this|them|those)\b",
            normalized_text,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(verwende|benutze|nutze|ändere|aendere|bearbeite|mach|mache|lies|zeige|"
            r"übersetze|uebersetze|vergleiche|korrigiere)\b.{0,80}\b"
            r"(?:das|dies|diese|diesen|dem|ihn|sie|es|vorherige|letzte)\b",
            normalized_text,
            re.IGNORECASE,
        )
    )


def _message_has_artifact_type(message: dict[str, Any], artifact_type: str) -> bool:
    normalized_type = str(artifact_type or '').strip().lower()
    if not normalized_type or not isinstance(message, dict):
        return False
    for artifact in (message.get('artifacts') or []):
        if not isinstance(artifact, dict):
            continue
        token = str(artifact.get('type') or '').strip().lower()
        if token == normalized_type:
            return True
    return False


def _follow_up_capability_for_artifact_type(
    artifact_type: Any,
    *,
    current_prompt: str,
    explicit_reference: bool = False,
) -> Optional[str]:
    normalized_type = str(artifact_type or '').strip().lower()
    raw_prompt = str(current_prompt or '').strip()
    lowered_prompt = raw_prompt.lower()
    normalized_prompt = normalize_intent_text(raw_prompt)
    primary_capability = normalize_capability(analyze_prompt_intent(raw_prompt).get('primary_capability'))
    if normalized_type == 'image':
        if _is_reference_image_generation_intent(raw_prompt):
            return CAPABILITY_IMAGE_GENERATION
        if _references_existing_image(raw_prompt):
            if (
                _has_strong_image_edit_instruction(raw_prompt, normalized_prompt)
                or (
                    _IMAGE_EDIT_REFERENCE_RE.search(normalized_prompt)
                    and (
                        _IMAGE_EDIT_CONTINUATION_RE.search(normalized_prompt)
                        or _IMAGE_EDIT_IDENTITY_RE.search(normalized_prompt)
                    )
                )
            ):
                if primary_capability == CAPABILITY_VISION_ANALYSIS and not explicit_reference:
                    return CAPABILITY_VISION_ANALYSIS
                return CAPABILITY_IMAGE_GENERATION
            if explicit_reference and _IMAGE_EDIT_PREFIX_RE.search(raw_prompt):
                return CAPABILITY_IMAGE_GENERATION
            return CAPABILITY_VISION_ANALYSIS
        if (
            _IMAGE_EDIT_REFERENCE_RE.search(normalized_prompt)
            and (
                _IMAGE_EDIT_CONTINUATION_RE.search(normalized_prompt)
                or _IMAGE_EDIT_IDENTITY_RE.search(normalized_prompt)
            )
        ):
            return CAPABILITY_IMAGE_GENERATION
        if explicit_reference and (
            not _references_existing_image(raw_prompt)
            or _FOLLOW_UP_CONTINUATION_RE.search(raw_prompt)
            or _has_strong_image_edit_instruction(raw_prompt, normalized_prompt)
        ):
            return CAPABILITY_IMAGE_GENERATION
        return None
    if normalized_type == 'audio' and _references_recent_audio(lowered_prompt):
        return CAPABILITY_SPEECH_TO_TEXT
    if normalized_type in {'text', 'markdown', 'md', 'json', 'csv', 'message'} and _references_recent_text(lowered_prompt):
        return CAPABILITY_TEXT_TO_SPEECH
    return None


def _artifact_follow_up_capability_context(
    context: dict[str, Any],
    *,
    current_prompt: str,
) -> Optional[str]:
    selected_reference_artifact = _primary_reference_artifact(context)
    reference_artifacts = _reference_artifacts(context)
    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}
    recent_artifacts = context.get('recent_artifacts') if isinstance(context.get('recent_artifacts'), list) else []

    for candidate, explicit_reference in (
        *((item, True) for item in [selected_reference_artifact, *reference_artifacts]),
        *((item, False) for item in [
            latest_artifacts.get('image'),
            latest_artifacts.get('text'),
            latest_artifacts.get('audio'),
            *recent_artifacts,
        ]),
    ):
        if not isinstance(candidate, dict):
            continue
        capability = _follow_up_capability_for_artifact_type(
            candidate.get('type'),
            current_prompt=current_prompt,
            explicit_reference=explicit_reference,
        )
        if capability:
            return capability
    return None


def _recent_assistant_follow_up_capability_context(
    context: dict[str, Any],
    *,
    current_prompt: str,
) -> Optional[str]:
    artifact_capability = _artifact_follow_up_capability_context(
        context,
        current_prompt=current_prompt,
    )
    if artifact_capability:
        return artifact_capability
    recent_messages = context.get('recent_messages') if isinstance(context.get('recent_messages'), list) else []
    if not recent_messages:
        return None
    lowered_prompt = str(current_prompt or '').strip().lower()
    wants_text_follow_up = _references_recent_text(lowered_prompt)
    for item in reversed(recent_messages):
        if not isinstance(item, dict):
            continue
        if str(item.get('role') or '').strip().lower() != 'assistant':
            continue
        if wants_text_follow_up and (
            str(item.get('saved_text_path') or '').strip()
            or _message_has_artifact_type(item, 'text')
            or str(item.get('content') or '').strip()
        ):
            return CAPABILITY_TEXT_TO_SPEECH
    return None


def _recent_user_capability_context(
    context: dict[str, Any],
    *,
    current_prompt: str,
) -> Optional[str]:
    if _signals_fresh_task(current_prompt):
        return None
    normalized_current = normalize_intent_text(current_prompt)
    selected_reference_artifact = _primary_reference_artifact(context)
    explicit_reference_anchor = str((selected_reference_artifact or {}).get('type') or '').strip().lower() in {
        'image',
        'text',
        'audio',
        'message',
    }
    if not explicit_reference_anchor and not _has_follow_up_reference_anchor(current_prompt, normalized_current):
        return None
    artifact_follow_up_capability = _recent_assistant_follow_up_capability_context(
        context,
        current_prompt=current_prompt,
    )
    if artifact_follow_up_capability:
        return artifact_follow_up_capability
    return None


def _is_image_edit_follow_up(
    prompt: str,
    *,
    context: dict[str, Any],
    latest_image: Optional[dict[str, Any]],
) -> bool:
    if not latest_image:
        return False
    raw = str(prompt or '').strip()
    if not raw:
        return False
    if _signals_fresh_task(raw):
        return False
    normalized = normalize_intent_text(raw)
    strong_instruction = _has_strong_image_edit_instruction(raw, normalized)
    recent_capability = _recent_user_capability_context(context, current_prompt=raw)
    if recent_capability != CAPABILITY_IMAGE_GENERATION and not strong_instruction:
        return False
    if not _has_image_edit_reference_anchor(raw, normalized) and not strong_instruction:
        return False
    if _IMAGE_EDIT_CONTINUATION_RE.search(normalized):
        return True
    if strong_instruction:
        return True
    return False


def _select_prompt_capability(
    prompt_intent: dict[str, Any],
    *,
    context: Optional[dict[str, Any]] = None,
    capabilities: Iterable[str] = (
        CAPABILITY_IMAGE_GENERATION,
        CAPABILITY_TEXT_TO_SPEECH,
        CAPABILITY_SPEECH_TO_TEXT,
        CAPABILITY_VISION_ANALYSIS,
    ),
) -> tuple[Optional[str], dict[str, int]]:
    capability_scores = prompt_intent.get('capability_scores') if isinstance(prompt_intent.get('capability_scores'), dict) else {}
    allowed = [normalize_capability(capability) for capability in capabilities if normalize_capability(capability)]
    scores = {
        capability: int(capability_scores.get(capability) or 0)
        for capability in allowed
    }
    candidates = [
        capability
        for capability in allowed
        if scores.get(capability, 0) >= _PROMPT_CAPABILITY_THRESHOLD
    ]
    if not candidates:
        return None, scores

    primary_capability = normalize_capability(prompt_intent.get('primary_capability'))
    if primary_capability in candidates:
        top_score = max(scores.get(capability, 0) for capability in candidates)
        if scores.get(primary_capability, 0) >= top_score:
            return primary_capability, scores

    selected = max(
        candidates,
        key=lambda capability: (
            scores.get(capability, 0),
            _PROMPT_CAPABILITY_TIEBREAK.get(capability, 0),
            capability,
        ),
    )
    return selected, scores


def is_obvious_route_hint_fast_path(context: dict[str, Any], route_hint: dict[str, Any]) -> bool:
    if not isinstance(context, dict) or not isinstance(route_hint, dict):
        return False
    capability = normalize_capability(route_hint.get('capability'))
    selected_reference_artifact = _primary_reference_artifact(context)
    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}
    latest_image = latest_artifacts.get('image') if isinstance(latest_artifacts.get('image'), dict) else None
    prompt = str(context.get('prompt') or '').strip()
    if selected_reference_artifact and capability in {
        CAPABILITY_IMAGE_GENERATION,
        CAPABILITY_TEXT_TO_SPEECH,
        CAPABILITY_SPEECH_TO_TEXT,
        CAPABILITY_VISION_ANALYSIS,
    }:
        return True
    if capability == CAPABILITY_IMAGE_GENERATION and _is_image_edit_follow_up(
        prompt,
        context=context,
        latest_image=latest_image,
    ):
        return True
    confidence = float(route_hint.get('confidence') or 0.0)
    if confidence < 0.9:
        return False

    request_attachment = context.get('request_attachment') if isinstance(context.get('request_attachment'), dict) else {}
    explicit_file_kind = str(request_attachment.get('file_kind') or '').strip().lower()
    has_explicit_file = bool(request_attachment.get('has_explicit_file'))
    if explicit_file_kind in {'audio', 'image', 'pdf', 'text'}:
        return True
    if has_explicit_file:
        return False

    lowered_prompt = prompt.lower()
    has_recent_artifact_reference = bool(
        (latest_artifacts.get('image') and _references_existing_image(lowered_prompt))
        or (latest_artifacts.get('audio') and _references_recent_audio(lowered_prompt))
        or (latest_artifacts.get('text') and _references_recent_text(lowered_prompt))
    )
    if has_recent_artifact_reference:
        return False

    if capability in {
        CAPABILITY_IMAGE_GENERATION,
        CAPABILITY_TEXT_TO_SPEECH,
        CAPABILITY_SPEECH_TO_TEXT,
        CAPABILITY_VISION_ANALYSIS,
    }:
        return True

    if capability != CAPABILITY_CHAT:
        return False

    prompt_intent = analyze_prompt_intent(lowered_prompt)
    selected_prompt_capability, _ = _select_prompt_capability(
        prompt_intent,
        context=context,
        capabilities=(CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH),
    )
    if _is_reference_image_generation_intent(lowered_prompt):
        return False
    if selected_prompt_capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}:
        return False
    return True


def build_route_hint(context: dict[str, Any]) -> dict[str, Any]:
    prompt = str(context.get('prompt') or '').strip()
    lowered_prompt = prompt.lower()
    fresh_task_requested = _signals_fresh_task(prompt)
    explicit_selected_reference = bool(
        _primary_reference_artifact(context)
        or _reference_artifacts(context)
    )
    allow_artifact_reuse = not fresh_task_requested or explicit_selected_reference
    prompt_intent = analyze_prompt_intent(prompt)
    explicit_defer_materialization = bool(prompt_intent.get('explicit_defer_materialization'))
    text_revision_turn = bool(prompt_intent.get('text_revision_turn'))
    direct_audio_materialization_request = bool(prompt_intent.get('direct_audio_materialization_request'))
    text_preparation_before_audio_output = bool(prompt_intent.get('text_preparation_before_audio_output'))
    text_preparation_before_visual_output = bool(prompt_intent.get('text_preparation_before_visual_output'))
    selected_output_capability, output_scores = _select_prompt_capability(
        prompt_intent,
        context=context,
        capabilities=(CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH),
    )
    materialization_intent_active = bool(
        selected_output_capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}
        or text_preparation_before_audio_output
        or text_preparation_before_visual_output
        or prompt_intent.get('requests_audio_output')
        or prompt_intent.get('requests_visual_output')
    )
    request_attachment = context.get('request_attachment') if isinstance(context.get('request_attachment'), dict) else {}
    explicit_file_kind = str(request_attachment.get('file_kind') or '').strip().lower()
    has_explicit_file = bool(request_attachment.get('has_explicit_file'))
    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}

    latest_image = latest_artifacts.get('image') if isinstance(latest_artifacts.get('image'), dict) else None
    latest_text = latest_artifacts.get('text') if isinstance(latest_artifacts.get('text'), dict) else None
    latest_audio = latest_artifacts.get('audio') if isinstance(latest_artifacts.get('audio'), dict) else None
    continuation_capability = (
        _recent_user_capability_context(context, current_prompt=prompt)
        if not fresh_task_requested and not selected_output_capability and _FOLLOW_UP_CONTINUATION_RE.search(prompt)
        else None
    )

    capability = CAPABILITY_CHAT
    reason = 'default chat fallback'
    reuse_last_artifact = False
    artifact_path = None
    confidence = 0.62

    if explicit_defer_materialization and (
        materialization_intent_active
        or explicit_file_kind in {'', 'text'}
    ):
        capability = CAPABILITY_CHAT
        reason = 'explicit deferal suppresses current-turn materialization'
        confidence = 0.96
    elif (
        text_revision_turn
        and not direct_audio_materialization_request
        and (
            selected_output_capability == CAPABILITY_TEXT_TO_SPEECH
            or text_preparation_before_audio_output
            or bool(prompt_intent.get('requests_audio_output'))
            or explicit_file_kind in {'', 'text'}
        )
    ):
        capability = CAPABILITY_CHAT
        reason = 'critique/edit/lock turn stays text-capable before audio handoff'
        confidence = 0.94
    elif explicit_file_kind == 'audio':
        capability = CAPABILITY_SPEECH_TO_TEXT
        reason = 'explicit audio input'
        confidence = 0.98
    elif explicit_file_kind == 'pdf':
        capability = CAPABILITY_VISION_ANALYSIS
        reason = 'explicit PDF input'
        confidence = 0.98
    elif explicit_file_kind == 'image':
        if selected_output_capability == CAPABILITY_IMAGE_GENERATION or _is_reference_image_generation_intent(lowered_prompt):
            capability = CAPABILITY_IMAGE_GENERATION
            reason = 'explicit image input plus image-generation cue'
        else:
            capability = CAPABILITY_VISION_ANALYSIS
            reason = 'explicit image input'
        confidence = 0.97
    elif explicit_file_kind == 'text':
        if selected_output_capability == CAPABILITY_TEXT_TO_SPEECH and not text_preparation_before_audio_output:
            capability = CAPABILITY_TEXT_TO_SPEECH
            reason = 'explicit text input plus text-to-speech cue'
        else:
            capability = CAPABILITY_CHAT
            reason = 'explicit text input plus text-preparation cue' if text_preparation_before_audio_output else 'explicit text input'
        confidence = 0.95
    elif text_preparation_before_audio_output:
        capability = CAPABILITY_CHAT
        reason = 'text preparation required before audio output'
        confidence = 0.86
    elif text_preparation_before_visual_output:
        capability = CAPABILITY_CHAT
        reason = 'text preparation required before image output'
        confidence = 0.86
    elif selected_output_capability == CAPABILITY_IMAGE_GENERATION:
        capability = CAPABILITY_IMAGE_GENERATION
        reason = 'image-generation cue'
        confidence = 0.9
    elif selected_output_capability == CAPABILITY_TEXT_TO_SPEECH:
        capability = CAPABILITY_TEXT_TO_SPEECH
        reason = 'text-to-speech cue'
        confidence = 0.9
    elif allow_artifact_reuse and _is_image_edit_follow_up(prompt, context=context, latest_image=latest_image):
        capability = CAPABILITY_IMAGE_GENERATION
        reuse_last_artifact = not has_explicit_file
        artifact_path = str(latest_image.get('path') or '').strip() or None
        reason = 'image-edit follow-up on latest image artifact'
        confidence = 0.88
    elif continuation_capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}:
        capability = continuation_capability
        reason = 'follow-up cue from recent artifact context'
        confidence = 0.78
    elif allow_artifact_reuse and latest_image and _references_existing_image(lowered_prompt):
        capability = CAPABILITY_VISION_ANALYSIS
        reuse_last_artifact = not has_explicit_file
        artifact_path = str(latest_image.get('path') or '').strip() or None
        reason = 'prompt refers to the latest image artifact'
        confidence = 0.84
    elif allow_artifact_reuse and latest_audio and _references_recent_audio(lowered_prompt):
        capability = CAPABILITY_SPEECH_TO_TEXT
        reuse_last_artifact = not has_explicit_file
        artifact_path = str(latest_audio.get('path') or '').strip() or None
        reason = 'prompt refers to the latest audio artifact'
        confidence = 0.82
    elif allow_artifact_reuse and latest_text and _references_recent_text(lowered_prompt):
        capability = (
            CAPABILITY_CHAT
            if text_preparation_before_audio_output
            else (CAPABILITY_TEXT_TO_SPEECH if selected_output_capability == CAPABILITY_TEXT_TO_SPEECH else CAPABILITY_CHAT)
        )
        reuse_last_artifact = not has_explicit_file
        artifact_path = str(latest_text.get('path') or '').strip() or None
        reason = (
            'prompt refers to the latest text artifact and still needs text preparation'
            if text_preparation_before_audio_output
            else 'prompt refers to the latest text artifact'
        )
        confidence = 0.8

    if (
        allow_artifact_reuse
        and capability == CAPABILITY_TEXT_TO_SPEECH
        and not text_preparation_before_audio_output
        and not has_explicit_file
        and latest_text
        and _references_recent_text(lowered_prompt)
    ):
        reuse_last_artifact = True
        artifact_path = str(latest_text.get('path') or '').strip() or None
        reason = 'text-to-speech prompt references the latest text artifact'
        confidence = max(confidence, 0.82)

    if allow_artifact_reuse and capability == CAPABILITY_IMAGE_GENERATION and not has_explicit_file and latest_image:
        if _is_reference_image_generation_intent(lowered_prompt):
            reuse_last_artifact = True
            artifact_path = str(latest_image.get('path') or '').strip() or None
            reason = 'image-generation prompt references the latest image artifact'
            confidence = max(confidence, 0.83)

    return {
        'capability': capability,
        'instance_id': None,
        'reuse_last_artifact': reuse_last_artifact and bool(artifact_path),
        'artifact_path': artifact_path if reuse_last_artifact else None,
        'confidence': confidence,
        'reason': reason,
    }


def build_failure_recovery_route(
    context: dict[str, Any],
    *,
    failed_capability: Optional[str],
    failed_error_message: str = '',
) -> Optional[dict[str, Any]]:
    normalized_failed_capability = normalize_capability(failed_capability)
    if normalized_failed_capability != CAPABILITY_CHAT:
        return None
    if not isinstance(context, dict):
        return None

    prompt = str(context.get('prompt') or '').strip()
    if not prompt or _signals_fresh_task(prompt):
        return None

    request_attachment = context.get('request_attachment') if isinstance(context.get('request_attachment'), dict) else {}
    if bool(request_attachment.get('has_explicit_file')):
        return None

    latest_artifacts = context.get('latest_artifacts') if isinstance(context.get('latest_artifacts'), dict) else {}
    latest_image = latest_artifacts.get('image') if isinstance(latest_artifacts.get('image'), dict) else None
    if not latest_image:
        return None

    selected_reference_artifact = _primary_reference_artifact(context)
    selected_reference_type = str((selected_reference_artifact or {}).get('type') or '').strip().lower()
    image_anchor = selected_reference_type == 'image' or bool(latest_image)
    if not image_anchor:
        return None

    normalized_prompt = normalize_intent_text(prompt)
    edit_cue = bool(
        _is_reference_image_generation_intent(prompt)
        or _is_image_edit_follow_up(prompt, context=context, latest_image=latest_image)
        or (
            _IMAGE_EDIT_PREFIX_RE.search(prompt)
            and (
                _IMAGE_EDIT_CONTINUATION_RE.search(normalized_prompt)
                or _IMAGE_EDIT_IDENTITY_RE.search(normalized_prompt)
                or re.search(r'\b(unchanged|leave the rest|leave everything else|only change|just change)\b', normalized_prompt, re.IGNORECASE)
                or re.search(r'\bchange\b.+\bto\b', normalized_prompt, re.IGNORECASE)
            )
        )
    )
    if not edit_cue:
        return None

    artifact_path = str((selected_reference_artifact or {}).get('path') or latest_image.get('path') or '').strip() or None
    if not artifact_path:
        return None

    failure_note = str(failed_error_message or '').strip().lower()
    confidence = 0.83
    if 'timeout' in failure_note or '500' in failure_note or 'internal server error' in failure_note:
        confidence = 0.87

    return {
        'capability': CAPABILITY_IMAGE_GENERATION,
        'instance_id': None,
        'reuse_last_artifact': True,
        'artifact_path': artifact_path,
        'confidence': confidence,
        'reason': 'self-heal retry: recover failed chat follow-up as image generation edit',
    }
