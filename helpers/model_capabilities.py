"""Capability, feature, and backend helpers for model registry and routing."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

CAPABILITY_CHAT = "chat"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_SPEECH_TO_TEXT = "speech_to_text"
CAPABILITY_TEXT_TO_SPEECH = "text_to_speech"
CAPABILITY_IMAGE_GENERATION = "image_generation"
CAPABILITY_VISION_ANALYSIS = "vision_analysis"

FEATURE_TOOL_CALLING = "tool_calling"
FEATURE_FUNCTION_CALLING = "function_calling"
FEATURE_COMPUTER_USE = "computer_use"
FEATURE_STRUCTURED_OUTPUTS = "structured_outputs"
FEATURE_VISION_INPUT = "vision_input"
FEATURE_AUDIO_INPUT = "audio_input"
FEATURE_IMAGE_OUTPUT = "image_output"
FEATURE_AUDIO_OUTPUT = "audio_output"

FEATURE_SOURCE_EXPLICIT = "explicit_metadata"
FEATURE_SOURCE_LOCAL_TEMPLATE = "local_template"
FEATURE_SOURCE_CURATED = "curated_override"
FEATURE_SOURCE_CAPABILITY = "capability_contract"
FEATURE_SOURCE_BACKEND_METADATA = "backend_metadata"
FEATURE_SOURCE_DEFAULT = "conservative_default"

SUPPORTED_FEATURE_FLAGS = {
    FEATURE_TOOL_CALLING,
    FEATURE_FUNCTION_CALLING,
    FEATURE_COMPUTER_USE,
    FEATURE_STRUCTURED_OUTPUTS,
    FEATURE_VISION_INPUT,
    FEATURE_AUDIO_INPUT,
    FEATURE_IMAGE_OUTPUT,
    FEATURE_AUDIO_OUTPUT,
}

_CURATED_FEATURE_RULES = (
    {
        "match": ("qwen3-coder",),
        "features": {
            FEATURE_TOOL_CALLING: True,
            FEATURE_FUNCTION_CALLING: True,
        },
    },
)

SUPPORTED_CAPABILITIES = {
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_VISION_ANALYSIS,
}


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            return int(token)
        except ValueError:
            return None
    return None


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        return value
    token = value.strip()
    if not token:
        return ""
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    numeric = _coerce_int(token)
    if numeric is not None:
        return numeric
    try:
        return float(token)
    except ValueError:
        return token


def _normalize_backend_capability_token(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        token = str(raw or "").strip()
        if not token:
            continue
        if token not in items:
            items.append(token)
    return items


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        compact[key] = value
    return compact


def parse_ollama_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    parsed: dict[str, Any] = {}
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, _, remainder = line.partition(" ")
        normalized_key = _normalize_backend_capability_token(key)
        if not normalized_key:
            continue
        candidate: Any = True if not remainder.strip() else _coerce_scalar(remainder.strip())
        existing = parsed.get(normalized_key)
        if existing is None:
            parsed[normalized_key] = candidate
            continue
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(candidate)
        parsed[normalized_key] = existing
    return parsed


def _ollama_model_info_context_length(model_info: Any) -> Optional[int]:
    if not isinstance(model_info, dict):
        return None
    direct = _coerce_int(model_info.get("context_length"))
    if direct is not None:
        return direct
    for key, value in model_info.items():
        if str(key).strip().lower().endswith(".context_length"):
            context_length = _coerce_int(value)
            if context_length is not None:
                return context_length
    return None


def build_ollama_show_summary(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    raw_capabilities = _normalize_string_list(payload.get("capabilities"))
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    model_info = payload.get("model_info") if isinstance(payload.get("model_info"), dict) else {}
    parameter_map = parse_ollama_parameters(payload.get("parameters"))
    context_length = (
        _ollama_model_info_context_length(model_info)
        or _coerce_int(parameter_map.get("num_ctx"))
        or _coerce_int(payload.get("context_length"))
    )

    summary = _compact_dict(
        {
            "source": "ollama_api_show",
            "capabilities": raw_capabilities,
            "modified_at": str(payload.get("modified_at") or "").strip() or None,
            "parameters": str(payload.get("parameters") or "").strip() or None,
            "parameter_map": parameter_map,
            "details": details,
            "context_length": context_length,
            "architecture": str(model_info.get("general.architecture") or "").strip() or None,
            "template_present": bool(str(payload.get("template") or "").strip()),
        }
    )
    return summary


def _ollama_model_variants(model_name: Optional[str]) -> set[str]:
    token = (model_name or "").strip().lower()
    if not token:
        return set()
    variants: set[str] = {token}
    base = token.split(":", 1)[0]
    variants.add(base)
    if "/" in base:
        variants.add(base.split("/", 1)[1])
    for item in list(variants):
        variants.add(f"{item}:latest")
    return variants


def _match_ollama_running_model(payload: Any, model_name: Optional[str]) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        candidates = [item for item in payload.get("models", []) if isinstance(item, dict)]
    elif isinstance(payload, dict):
        candidates = [payload]
    else:
        candidates = []

    target = str(model_name or "").strip()
    target_variants = _ollama_model_variants(target)

    for candidate in candidates:
        candidate_name = str(candidate.get("name") or candidate.get("model") or "").strip()
        if target and candidate_name == target:
            return candidate
    for candidate in candidates:
        candidate_name = str(candidate.get("name") or candidate.get("model") or "").strip()
        if target and candidate_name.lower().startswith(target.lower()):
            return candidate
    for candidate in candidates:
        candidate_name = str(candidate.get("name") or candidate.get("model") or "").strip()
        if _ollama_model_variants(candidate_name) & target_variants:
            return candidate
    return {}


def build_ollama_ps_summary(payload: Optional[dict[str, Any]], *, model_name: Optional[str] = None) -> dict[str, Any]:
    matched = _match_ollama_running_model(payload, model_name)
    if not matched:
        return {}

    details = matched.get("details") if isinstance(matched.get("details"), dict) else {}
    return _compact_dict(
        {
            "source": "ollama_api_ps",
            "name": str(matched.get("name") or "").strip() or None,
            "model": str(matched.get("model") or "").strip() or None,
            "size": _coerce_int(matched.get("size")),
            "digest": str(matched.get("digest") or "").strip() or None,
            "expires_at": str(matched.get("expires_at") or "").strip() or None,
            "size_vram": _coerce_int(matched.get("size_vram")),
            "context_length": _coerce_int(matched.get("context_length")),
            "details": details,
        }
    )


def _metadata_backend_capabilities(metadata: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(metadata, dict):
        return []

    raw_values: list[Any] = []
    existing_backend_metadata = metadata.get("backend_metadata")
    if isinstance(existing_backend_metadata, dict):
        raw_values.extend(existing_backend_metadata.get("capabilities") or [])

    for key in ("capabilities", "provider_capabilities"):
        value = metadata.get(key)
        if isinstance(value, list):
            raw_values.extend(value)

    normalized: list[str] = []
    for raw in raw_values:
        token = _normalize_backend_capability_token(raw)
        if token and token not in normalized:
            normalized.append(token)
    return normalized

_CAPABILITY_ALIASES = {
    "chat": CAPABILITY_CHAT,
    "text": CAPABILITY_CHAT,
    "text_chat": CAPABILITY_CHAT,
    "assistant": CAPABILITY_CHAT,
    "completion": CAPABILITY_CHAT,
    "completions": CAPABILITY_CHAT,
    "embedding": CAPABILITY_EMBEDDING,
    "embeddings": CAPABILITY_EMBEDDING,
    "embed": CAPABILITY_EMBEDDING,
    "speech_to_text": CAPABILITY_SPEECH_TO_TEXT,
    "speech2text": CAPABILITY_SPEECH_TO_TEXT,
    "stt": CAPABILITY_SPEECH_TO_TEXT,
    "transcription": CAPABILITY_SPEECH_TO_TEXT,
    "audio_transcription": CAPABILITY_SPEECH_TO_TEXT,
    "text_to_speech": CAPABILITY_TEXT_TO_SPEECH,
    "text2speech": CAPABILITY_TEXT_TO_SPEECH,
    "tts": CAPABILITY_TEXT_TO_SPEECH,
    "audio_speech": CAPABILITY_TEXT_TO_SPEECH,
    "speech_generation": CAPABILITY_TEXT_TO_SPEECH,
    "image_generation": CAPABILITY_IMAGE_GENERATION,
    "image_gen": CAPABILITY_IMAGE_GENERATION,
    "image": CAPABILITY_IMAGE_GENERATION,
    "vision_analysis": CAPABILITY_VISION_ANALYSIS,
    "vision": CAPABILITY_VISION_ANALYSIS,
    "ocr": CAPABILITY_VISION_ANALYSIS,
    "visual_qa": CAPABILITY_VISION_ANALYSIS,
}


def normalize_backend(value: Optional[str]) -> str:
    backend = (value or "ollama").strip().lower()
    if not backend:
        return "ollama"
    if backend in {"llama.cpp", "llama-cpp", "llama_cpp", "llamacpp"}:
        return "llama_cpp"
    if backend in {"mlx", "mlx-lm"}:
        return "mlx"
    if backend == "ollama":
        return "ollama"
    return backend


def normalize_capability(value: Optional[str]) -> str:
    if not value:
        return ""
    token = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    return _CAPABILITY_ALIASES.get(token, token)


def _metadata_capability_hint(metadata: Optional[dict[str, Any]]) -> str:
    if not isinstance(metadata, dict):
        return ""

    explicit_capability = normalize_capability(metadata.get("capability"))
    if explicit_capability in SUPPORTED_CAPABILITIES:
        return explicit_capability

    explicit_provider_capabilities = _normalize_string_list(metadata.get("provider_capabilities"))
    for raw in explicit_provider_capabilities:
        normalized = normalize_capability(raw)
        if normalized in SUPPORTED_CAPABILITIES:
            return normalized

    recognized_metadata_capabilities: set[str] = set()
    for raw in _metadata_backend_capabilities(metadata):
        normalized = normalize_capability(raw)
        if normalized in SUPPORTED_CAPABILITIES:
            recognized_metadata_capabilities.add(normalized)

    # Prefer text/chat as the primary capability when a provider explicitly
    # advertises completion plus other multimodal affordances like vision.
    for preferred in (
        CAPABILITY_EMBEDDING,
        CAPABILITY_SPEECH_TO_TEXT,
        CAPABILITY_TEXT_TO_SPEECH,
        CAPABILITY_IMAGE_GENERATION,
        CAPABILITY_CHAT,
        CAPABILITY_VISION_ANALYSIS,
    ):
        if preferred in recognized_metadata_capabilities:
            return preferred

    pipeline_tag = normalize_capability(metadata.get("snapshot_pipeline_tag"))
    if pipeline_tag in SUPPORTED_CAPABILITIES:
        return pipeline_tag

    for key in ("outputs", "output"):
        outputs = _normalize_modality_list(metadata.get(key))
        if any(token in {"embedding", "embeddings", "vector", "vectors"} for token in outputs):
            return CAPABILITY_EMBEDDING

    return ""


def infer_capability(
    model_name: Optional[str],
    backend: Optional[str] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    metadata_hint = _metadata_capability_hint(metadata)
    if metadata_hint:
        return metadata_hint

    name = (model_name or "").strip().lower()
    normalized_backend = normalize_backend(backend)

    if (
        "qwen3-tts" in name
        or "qwen3_tts" in name
        or "text-to-speech" in name
        or "text_to_speech" in name
        or ("tts" in name and "stt" not in name and "whisper" not in name)
    ):
        return CAPABILITY_TEXT_TO_SPEECH
    if "whisper" in name:
        return CAPABILITY_SPEECH_TO_TEXT
    if "flux" in name or "stable-diffusion" in name or "sdxl" in name:
        return CAPABILITY_IMAGE_GENERATION
    if (
        "ocr" in name
        or "llava" in name
        or "vision" in name
        or "vlm" in name
        or "vl" in name
        or "pixtral" in name
        or "idefics" in name
        or "paligemma" in name
        or "molmo" in name
        or "florence" in name
    ):
        return CAPABILITY_VISION_ANALYSIS
    if normalized_backend == "mlx":
        # MLX defaults to chat unless the model name or request metadata says
        # otherwise (Whisper, VLM/OCR, etc.).
        return CAPABILITY_CHAT
    if normalized_backend == "llama_cpp":
        return CAPABILITY_CHAT
    return CAPABILITY_CHAT


def _coerce_feature_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return None


def _normalize_modality_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token not in items:
            items.append(token)
    return items


def _resolve_local_model_path(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for key in ("model_path", "path", "request_model"):
        raw = str(metadata.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path.is_dir():
            return str(path)
    return None


@lru_cache(maxsize=128)
def _read_local_template_evidence(model_dir: str) -> dict[str, Any]:
    path = Path(model_dir)
    template_chunks: list[str] = []

    tokenizer_config = path / "tokenizer_config.json"
    if tokenizer_config.exists():
        try:
            payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
            chat_template = payload.get("chat_template")
            if isinstance(chat_template, str) and chat_template.strip():
                template_chunks.append(chat_template)
        except Exception:
            pass

    for candidate_name in ("chat_template.jinja", "chat_template.json", "chat_template.txt"):
        candidate = path / candidate_name
        if candidate.exists():
            try:
                template_chunks.append(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue

    combined = "\n".join(chunk for chunk in template_chunks if chunk).lower()
    has_tooling = any(
        marker in combined
        for marker in ("tool_call", "tool_calls", "<tools>", "function=", "functions", "[tool_calls]")
    )
    has_vision = any(
        marker in combined
        for marker in ("<|vision_start|>", "image_url", "input_image", "image_pad", "picture ")
    )
    has_structured = any(
        marker in combined
        for marker in ("json_schema", "response_format", "structured_output")
    )
    return {
        "has_tooling": has_tooling,
        "has_vision": has_vision,
        "has_structured": has_structured,
    }


def _curated_feature_overrides(model_name: Optional[str]) -> dict[str, bool]:
    lowered = str(model_name or "").strip().lower()
    if not lowered:
        return {}
    for rule in _CURATED_FEATURE_RULES:
        markers = rule.get("match") or ()
        if any(marker in lowered for marker in markers):
            features = rule.get("features")
            if isinstance(features, dict):
                return {key: bool(value) for key, value in features.items() if key in SUPPORTED_FEATURE_FLAGS}
    return {}


def default_inputs_for_capability(capability: Optional[str]) -> list[str]:
    resolved = normalize_capability(capability)
    if resolved == CAPABILITY_VISION_ANALYSIS:
        return ["text", "image"]
    if resolved == CAPABILITY_SPEECH_TO_TEXT:
        return ["audio"]
    return ["text"]


def default_outputs_for_capability(capability: Optional[str]) -> list[str]:
    resolved = normalize_capability(capability)
    if resolved == CAPABILITY_EMBEDDING:
        return ["embedding"]
    if resolved == CAPABILITY_IMAGE_GENERATION:
        return ["image"]
    if resolved == CAPABILITY_TEXT_TO_SPEECH:
        return ["audio"]
    return ["text"]


def infer_supported_capabilities(
    model_name: Optional[str],
    backend: Optional[str],
    capability: Optional[str] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> list[str]:
    source = metadata if isinstance(metadata, dict) else {}
    normalized_backend = normalize_backend(backend)
    resolved_capability = normalize_capability(capability)
    if not resolved_capability or resolved_capability not in SUPPORTED_CAPABILITIES:
        resolved_capability = infer_capability(model_name, normalized_backend, metadata=source)

    feature_contract = build_feature_contract(
        model_name,
        normalized_backend,
        resolved_capability,
        metadata=source,
    )
    inputs = _normalize_modality_list(feature_contract.get("inputs"))
    outputs = _normalize_modality_list(feature_contract.get("outputs"))

    supported: list[str] = []

    def add(token: Optional[str]) -> None:
        normalized = normalize_capability(token)
        if normalized in SUPPORTED_CAPABILITIES and normalized not in supported:
            supported.append(normalized)

    add(resolved_capability)

    provider_capabilities = _normalize_modality_list(source.get("provider_capabilities"))
    if not provider_capabilities:
        for raw in _metadata_backend_capabilities(source):
            normalized = normalize_capability(raw)
            if normalized in SUPPORTED_CAPABILITIES:
                provider_capabilities.append(normalized)
    for token in provider_capabilities:
        add(token)

    if "embedding" in outputs:
        add(CAPABILITY_EMBEDDING)
    if "text" in inputs and "audio" in outputs:
        add(CAPABILITY_TEXT_TO_SPEECH)
    if "audio" in inputs and "text" in outputs:
        add(CAPABILITY_SPEECH_TO_TEXT)
    if "text" in inputs and "image" in outputs:
        add(CAPABILITY_IMAGE_GENERATION)
    if "image" in inputs and "text" in outputs:
        add(CAPABILITY_VISION_ANALYSIS)
    if "text" in inputs and "text" in outputs:
        add(CAPABILITY_CHAT)

    return supported


def supports_capability(
    target_capability: Optional[str],
    *,
    model_name: Optional[str],
    backend: Optional[str],
    capability: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    normalized_target = normalize_capability(target_capability)
    if normalized_target not in SUPPORTED_CAPABILITIES:
        return False
    return normalized_target in infer_supported_capabilities(
        model_name,
        backend,
        capability,
        metadata=metadata,
    )


def is_text_capable(
    model_name: Optional[str],
    backend: Optional[str],
    capability: Optional[str] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    return supports_capability(
        CAPABILITY_CHAT,
        model_name=model_name,
        backend=backend,
        capability=capability,
        metadata=metadata,
    )


def build_feature_contract(
    model_name: Optional[str],
    backend: Optional[str],
    capability: Optional[str] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    resolved_capability = normalize_capability(capability)
    if not resolved_capability or resolved_capability not in SUPPORTED_CAPABILITIES:
        resolved_capability = infer_capability(model_name, backend, metadata=metadata)

    source = metadata if isinstance(metadata, dict) else {}
    inputs = _normalize_modality_list(source.get("inputs"))
    if not inputs:
        inputs = _normalize_modality_list(source.get("input"))
    if not inputs:
        inputs = default_inputs_for_capability(resolved_capability)

    outputs = _normalize_modality_list(source.get("outputs"))
    if not outputs:
        outputs = _normalize_modality_list(source.get("output"))
    if not outputs:
        outputs = default_outputs_for_capability(resolved_capability)

    backend_capabilities = _metadata_backend_capabilities(source)
    backend_declares_vision = CAPABILITY_VISION_ANALYSIS in {
        normalize_capability(token)
        for token in backend_capabilities
    }
    backend_contract_explicitly_nonvision = bool(backend_capabilities) and not backend_declares_vision
    if backend_contract_explicitly_nonvision:
        inputs = [item for item in inputs if item != "image"]

    explicit_features = source.get("features")
    if not isinstance(explicit_features, dict):
        explicit_features = {}

    feature_values: dict[str, bool] = {
        FEATURE_TOOL_CALLING: False,
        FEATURE_FUNCTION_CALLING: False,
        FEATURE_COMPUTER_USE: False,
        FEATURE_STRUCTURED_OUTPUTS: False,
        FEATURE_VISION_INPUT: "image" in inputs,
        FEATURE_AUDIO_INPUT: "audio" in inputs,
        FEATURE_IMAGE_OUTPUT: "image" in outputs,
        FEATURE_AUDIO_OUTPUT: "audio" in outputs,
    }
    feature_sources: dict[str, str] = {
        FEATURE_TOOL_CALLING: FEATURE_SOURCE_DEFAULT,
        FEATURE_FUNCTION_CALLING: FEATURE_SOURCE_DEFAULT,
        FEATURE_COMPUTER_USE: FEATURE_SOURCE_DEFAULT,
        FEATURE_STRUCTURED_OUTPUTS: FEATURE_SOURCE_DEFAULT,
        FEATURE_VISION_INPUT: FEATURE_SOURCE_CAPABILITY if "image" in inputs else FEATURE_SOURCE_DEFAULT,
        FEATURE_AUDIO_INPUT: FEATURE_SOURCE_CAPABILITY if "audio" in inputs else FEATURE_SOURCE_DEFAULT,
        FEATURE_IMAGE_OUTPUT: FEATURE_SOURCE_CAPABILITY if "image" in outputs else FEATURE_SOURCE_DEFAULT,
        FEATURE_AUDIO_OUTPUT: FEATURE_SOURCE_CAPABILITY if "audio" in outputs else FEATURE_SOURCE_DEFAULT,
    }
    if "vision" in backend_capabilities:
        if "image" not in inputs:
            inputs.append("image")
        feature_values[FEATURE_VISION_INPUT] = True
        feature_sources[FEATURE_VISION_INPUT] = FEATURE_SOURCE_BACKEND_METADATA
    if any(token in backend_capabilities for token in ("tools", "tool_calling", "function_calling")):
        feature_values[FEATURE_TOOL_CALLING] = True
        feature_values[FEATURE_FUNCTION_CALLING] = True
        feature_sources[FEATURE_TOOL_CALLING] = FEATURE_SOURCE_BACKEND_METADATA
        feature_sources[FEATURE_FUNCTION_CALLING] = FEATURE_SOURCE_BACKEND_METADATA

    explicit_key_map = {
        FEATURE_TOOL_CALLING: ("supports_tools", "supports_tool_calling", FEATURE_TOOL_CALLING),
        FEATURE_FUNCTION_CALLING: ("supports_function_calling", FEATURE_FUNCTION_CALLING),
        FEATURE_COMPUTER_USE: ("supports_computer_use", FEATURE_COMPUTER_USE),
        FEATURE_STRUCTURED_OUTPUTS: ("supports_structured_outputs", FEATURE_STRUCTURED_OUTPUTS),
        FEATURE_VISION_INPUT: ("supports_vision_input", FEATURE_VISION_INPUT),
        FEATURE_AUDIO_INPUT: ("supports_audio_input", FEATURE_AUDIO_INPUT),
        FEATURE_IMAGE_OUTPUT: ("supports_image_output", FEATURE_IMAGE_OUTPUT),
        FEATURE_AUDIO_OUTPUT: ("supports_audio_output", FEATURE_AUDIO_OUTPUT),
    }

    local_model_path = _resolve_local_model_path(source)
    if local_model_path:
        template_evidence = _read_local_template_evidence(local_model_path)
        if template_evidence.get("has_tooling"):
            feature_values[FEATURE_TOOL_CALLING] = True
            feature_values[FEATURE_FUNCTION_CALLING] = True
            feature_sources[FEATURE_TOOL_CALLING] = FEATURE_SOURCE_LOCAL_TEMPLATE
            feature_sources[FEATURE_FUNCTION_CALLING] = FEATURE_SOURCE_LOCAL_TEMPLATE
        if template_evidence.get("has_vision") and not backend_contract_explicitly_nonvision:
            if "image" not in inputs:
                inputs.append("image")
            feature_values[FEATURE_VISION_INPUT] = True
            feature_sources[FEATURE_VISION_INPUT] = FEATURE_SOURCE_LOCAL_TEMPLATE
        if template_evidence.get("has_structured"):
            feature_values[FEATURE_STRUCTURED_OUTPUTS] = True
            feature_sources[FEATURE_STRUCTURED_OUTPUTS] = FEATURE_SOURCE_LOCAL_TEMPLATE

    curated_overrides = _curated_feature_overrides(model_name)
    for feature_key, feature_value in curated_overrides.items():
        feature_values[feature_key] = feature_value
        feature_sources[feature_key] = FEATURE_SOURCE_CURATED

    for feature_key in SUPPORTED_FEATURE_FLAGS:
        explicit_value = _coerce_feature_bool(explicit_features.get(feature_key))
        if (
            feature_key == FEATURE_VISION_INPUT
            and backend_contract_explicitly_nonvision
            and explicit_value
        ):
            explicit_value = False
        if explicit_value is not None:
            feature_values[feature_key] = explicit_value
            feature_sources[feature_key] = FEATURE_SOURCE_EXPLICIT
            continue
        for source_key in explicit_key_map[feature_key]:
            explicit_value = _coerce_feature_bool(source.get(source_key))
            if (
                feature_key == FEATURE_VISION_INPUT
                and backend_contract_explicitly_nonvision
                and explicit_value
            ):
                explicit_value = False
            if explicit_value is not None:
                feature_values[feature_key] = explicit_value
                feature_sources[feature_key] = FEATURE_SOURCE_EXPLICIT
                break

    inputs = _normalize_modality_list(inputs)
    outputs = _normalize_modality_list(outputs)
    feature_values[FEATURE_VISION_INPUT] = "image" in inputs or feature_values[FEATURE_VISION_INPUT]
    if "image" in inputs and feature_sources[FEATURE_VISION_INPUT] == FEATURE_SOURCE_DEFAULT:
        feature_sources[FEATURE_VISION_INPUT] = FEATURE_SOURCE_CAPABILITY
    feature_values[FEATURE_AUDIO_INPUT] = "audio" in inputs or feature_values[FEATURE_AUDIO_INPUT]
    if "audio" in inputs and feature_sources[FEATURE_AUDIO_INPUT] == FEATURE_SOURCE_DEFAULT:
        feature_sources[FEATURE_AUDIO_INPUT] = FEATURE_SOURCE_CAPABILITY
    feature_values[FEATURE_IMAGE_OUTPUT] = "image" in outputs or feature_values[FEATURE_IMAGE_OUTPUT]
    if "image" in outputs and feature_sources[FEATURE_IMAGE_OUTPUT] == FEATURE_SOURCE_DEFAULT:
        feature_sources[FEATURE_IMAGE_OUTPUT] = FEATURE_SOURCE_CAPABILITY
    feature_values[FEATURE_AUDIO_OUTPUT] = "audio" in outputs or feature_values[FEATURE_AUDIO_OUTPUT]
    if "audio" in outputs and feature_sources[FEATURE_AUDIO_OUTPUT] == FEATURE_SOURCE_DEFAULT:
        feature_sources[FEATURE_AUDIO_OUTPUT] = FEATURE_SOURCE_CAPABILITY
    if backend_contract_explicitly_nonvision:
        feature_values[FEATURE_VISION_INPUT] = False
        feature_sources[FEATURE_VISION_INPUT] = FEATURE_SOURCE_BACKEND_METADATA

    return {
        "features": feature_values,
        "feature_sources": feature_sources,
        "inputs": inputs,
        "outputs": outputs,
    }


def build_registry_metadata(
    model_name: Optional[str],
    backend: Optional[str],
    capability: Optional[str] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> dict:
    normalized_backend = normalize_backend(backend)
    source = metadata if isinstance(metadata, dict) else {}
    resolved_capability = normalize_capability(capability)
    explicit_provider_capabilities = _normalize_modality_list(source.get("provider_capabilities"))
    if explicit_provider_capabilities:
        explicit_provider_capability = normalize_capability(explicit_provider_capabilities[0])
        if explicit_provider_capability in SUPPORTED_CAPABILITIES and (
            not resolved_capability or resolved_capability not in explicit_provider_capabilities
        ):
            resolved_capability = explicit_provider_capability
    if not resolved_capability or resolved_capability not in SUPPORTED_CAPABILITIES:
        resolved_capability = infer_capability(model_name, normalized_backend, metadata=metadata)

    feature_contract = build_feature_contract(
        model_name,
        normalized_backend,
        resolved_capability,
        metadata=metadata,
    )

    backend_metadata = source.get("backend_metadata") if isinstance(source.get("backend_metadata"), dict) else {}
    backend_package = (
        str(source.get("backend_package") or "").strip()
        or str(backend_metadata.get("backend_package") or "").strip()
        or None
    )
    backend_contract = (
        str(source.get("backend_contract") or "").strip()
        or str(backend_metadata.get("backend_contract") or "").strip()
        or None
    )
    if not backend_package and normalized_backend == "ollama":
        backend_package = "ollama"
    if not backend_contract and normalized_backend == "ollama":
        backend_contract = "ollama.api"
    if not backend_package and normalized_backend == "llama_cpp":
        backend_package = "llama_cpp"
    if not backend_contract and normalized_backend == "llama_cpp":
        backend_contract = "llama.cpp.server"
    supported_capabilities = infer_supported_capabilities(
        model_name,
        normalized_backend,
        resolved_capability,
        metadata=source,
    )
    provider_capabilities = list(explicit_provider_capabilities)
    if not provider_capabilities:
        provider_capabilities = list(supported_capabilities)

    payload = {
        "modelName": model_name or "",
        "backend": normalized_backend,
        "capability": resolved_capability,
        "supported_capabilities": supported_capabilities,
        "text_capable": CAPABILITY_CHAT in supported_capabilities,
        **feature_contract,
    }
    if backend_metadata:
        payload["backend_metadata"] = backend_metadata
    if backend_package:
        payload["backend_package"] = backend_package
    if backend_contract:
        payload["backend_contract"] = backend_contract
    if provider_capabilities:
        payload["provider_capabilities"] = provider_capabilities
    return payload
