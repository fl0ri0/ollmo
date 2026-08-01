"""Runtime projection for ChatGPT through Ollmo's internal Codex adapter."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Optional

from ollmo_integrations.codex.execution import (
    CodexAccessStatus,
    CodexExecutionInput,
    CodexExecutionResult,
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_INPUT_FILES,
    DEFAULT_MAX_TOTAL_INPUT_BYTES,
    probe_codex_access,
)


CODEX_TARGET_ID = 'external:codex'
CODEX_TARGET_LABEL = 'ChatGPT'
CODEX_TARGET_MODEL = 'codex:auto'
CODEX_TARGET_BACKEND = 'codex_cli'
CODEX_FILE_CONSENT_SCOPE = 'selected_files_v1'


def _preferences_payload(raw_value: Any) -> Mapping[str, Any]:
    source = raw_value if isinstance(raw_value, Mapping) else {}
    nested = source.get('preferences')
    return nested if isinstance(nested, Mapping) else source


def codex_consent_enabled(raw_preferences: Any) -> bool:
    preferences = _preferences_payload(raw_preferences)
    external_targets = (
        preferences.get('external_targets')
        if isinstance(preferences.get('external_targets'), Mapping)
        else preferences.get('externalTargets')
        if isinstance(preferences.get('externalTargets'), Mapping)
        else {}
    )
    codex_settings = (
        external_targets.get('codex')
        if isinstance(external_targets.get('codex'), Mapping)
        else {}
    )
    value = (
        codex_settings.get('enabled')
        if 'enabled' in codex_settings
        else preferences.get('codex_enabled')
        if 'codex_enabled' in preferences
        else preferences.get('codexEnabled')
    )
    if isinstance(value, bool):
        return value
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def codex_file_consent_enabled(raw_preferences: Any) -> bool:
    """Return whether the persisted consent explicitly includes selected files."""

    preferences = _preferences_payload(raw_preferences)
    external_targets = (
        preferences.get('external_targets')
        if isinstance(preferences.get('external_targets'), Mapping)
        else preferences.get('externalTargets')
        if isinstance(preferences.get('externalTargets'), Mapping)
        else {}
    )
    codex_settings = (
        external_targets.get('codex')
        if isinstance(external_targets.get('codex'), Mapping)
        else {}
    )
    scope = str(
        codex_settings.get('data_scope')
        or codex_settings.get('dataScope')
        or ''
    ).strip().lower()
    if scope == CODEX_FILE_CONSENT_SCOPE:
        return True
    explicit = (
        codex_settings.get('files_enabled')
        if 'files_enabled' in codex_settings
        else codex_settings.get('filesEnabled')
    )
    if isinstance(explicit, bool):
        return explicit
    return str(explicit or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def codex_recovery_hint(status: str, *, enabled: bool) -> Optional[str]:
    normalized = str(status or '').strip().lower()
    if normalized == 'available':
        if enabled:
            return None
        return 'Enable ChatGPT in the Ollmo dashboard before any text can be sent to OpenAI.'
    if normalized == 'auth_required':
        return 'Open ChatGPT or Codex, sign in, then refresh the ChatGPT status in Ollmo.'
    if normalized == 'unavailable':
        return 'Install or update ChatGPT for macOS, or install Codex CLI, then restart Ollmo.'
    return 'Update or restart ChatGPT/Codex, then refresh the ChatGPT integration status.'


def build_codex_external_target(
    raw_preferences: Any,
    *,
    access_status: Optional[CodexAccessStatus] = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    status = access_status or probe_codex_access(force_refresh=force_refresh)
    enabled = codex_consent_enabled(raw_preferences)
    files_enabled = codex_file_consent_enabled(raw_preferences)
    state = str(status.status.value)
    discovery = status.discovery
    source = discovery.source.value if discovery.source is not None else None
    available = bool(status.available)
    selectable = bool(available and enabled)
    return {
        'id': CODEX_TARGET_ID,
        'instance_id': CODEX_TARGET_ID,
        'label': CODEX_TARGET_LABEL,
        'target_kind': 'external',
        'lifecycle_managed': False,
        'model': CODEX_TARGET_MODEL,
        'backend': CODEX_TARGET_BACKEND,
        'model_selection': 'codex_default',
        'exact_model_exposed': False,
        'capability': 'chat',
        'supported_capabilities': ['chat'],
        'provider_capabilities': ['chat'],
        'text_capable': True,
        'inputs': ['text', 'image', 'file'],
        'outputs': ['text'],
        'text_only': False,
        'file_input': True,
        'native_image_input': True,
        'files_enabled': files_enabled,
        'file_consent_scope': CODEX_FILE_CONSENT_SCOPE,
        'input_policy': {
            'explicit_current_turn_only': True,
            'regular_files_only': True,
            'max_files': DEFAULT_MAX_INPUT_FILES,
            'max_file_bytes': DEFAULT_MAX_INPUT_BYTES,
            'max_total_bytes': DEFAULT_MAX_TOTAL_INPUT_BYTES,
            'native_image_input': True,
            'other_files_via_read_only_workspace': True,
        },
        'status': state,
        'readiness': 'ready' if selectable else state,
        'activity': 'idle' if selectable else 'disabled',
        'available': available,
        'enabled': enabled,
        'selectable': selectable,
        'source': source,
        'version': discovery.version,
        'auth_method': status.auth_method,
        'cached': bool(status.cached),
        'recovery_hint': codex_recovery_hint(state, enabled=enabled),
    }


def _has_value(value: Any) -> bool:
    return value not in (None, '', [], {})


def _contains_remote_or_inline_file(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key or '').strip().lower()
            if normalized_key in {'file_id', 'image_url', 'url'} and _has_value(nested):
                return True
            if (
                normalized_key
                in {
                    'artifact_path',
                    'file',
                    'file_path',
                    'image',
                    'image_url',
                    'path',
                    'url',
                }
                and isinstance(nested, str)
                and nested.strip().lower().startswith(('http://', 'https://', 'data:'))
            ):
                return True
            if _contains_remote_or_inline_file(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_remote_or_inline_file(item) for item in value)
    return False


def _content_block_is_supported(value: Any) -> bool:
    if isinstance(value, Mapping):
        content_type = str(value.get('type') or '').strip().lower()
        if content_type and content_type not in {
            'input_text',
            'output_text',
            'text',
            'message',
            'input_file',
            'input_image',
            'file',
            'image',
            'audio',
            'video',
        }:
            return False
        if content_type in {
            'input_file',
            'input_image',
            'file',
            'image',
            'audio',
            'video',
        }:
            return bool(
                str(
                    value.get('path')
                    or value.get('file_path')
                    or value.get('artifact_path')
                    or ''
                ).strip()
            )
        return all(_content_block_is_supported(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_content_block_is_supported(item) for item in value)
    return True


def _iter_file_records(value: Any):
    if isinstance(value, Mapping):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping):
                yield item
            elif isinstance(item, (str, Path)):
                yield {'path': str(item)}
    elif isinstance(value, (str, Path)):
        yield {'path': str(value)}


def _parse_file_paths_json(value: Any) -> list[str]:
    """Parse the bounded browser multipart field for explicit local paths."""

    if value in (None, '', [], ()):
        return []
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
    raw_items = decoded if isinstance(decoded, (list, tuple)) else [decoded]
    return [
        str(item).strip()
        for item in raw_items
        if isinstance(item, (str, Path)) and str(item).strip()
    ]


def codex_request_has_file_inputs(
    request_payload: Any,
    *,
    upload_present: bool = False,
) -> bool:
    if upload_present:
        return True
    payload = request_payload if isinstance(request_payload, Mapping) else {}
    if _parse_file_paths_json(payload.get('file_paths_json')):
        return True
    for key in (
        'artifact_path',
        'attachments',
        'audio',
        'file',
        'file_path',
        'files',
        'image',
        'images',
        'input_artifacts',
        'reference_artifacts',
        'selected_reference_artifact',
        'selected_reference_artifacts',
    ):
        value = payload.get(key)
        if key in {'reference_artifacts', 'selected_reference_artifact', 'selected_reference_artifacts'}:
            if any(
                str(
                    record.get('path')
                    or record.get('file_path')
                    or record.get('artifact_path')
                    or ''
                ).strip()
                for record in _iter_file_records(value)
            ):
                return True
            continue
        if _has_value(value):
            return True
    input_value = payload.get('input')
    if isinstance(input_value, (list, tuple, Mapping)):
        for record in _walk_file_content_records(input_value):
            if record:
                return True
    return False


def _walk_file_content_records(value: Any):
    if isinstance(value, Mapping):
        content_type = str(value.get('type') or '').strip().lower()
        path = str(
            value.get('path')
            or value.get('file_path')
            or value.get('artifact_path')
            or ''
        ).strip()
        if path and content_type in {
            'input_file',
            'input_image',
            'file',
            'image',
            'audio',
            'video',
        }:
            yield value
        for nested in value.values():
            yield from _walk_file_content_records(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_file_content_records(item)


def build_codex_execution_inputs(request_payload: Any) -> list[CodexExecutionInput]:
    """Extract only explicit path-bearing current-turn records."""

    payload = request_payload if isinstance(request_payload, Mapping) else {}
    inputs: list[CodexExecutionInput] = []

    def append_record(record: Mapping[str, Any], source: str) -> None:
        path = str(
            record.get('path')
            or record.get('file_path')
            or record.get('artifact_path')
            or ''
        ).strip()
        if not path:
            return
        inputs.append(
            CodexExecutionInput(
                path=path,
                display_name=str(
                    record.get('name')
                    or record.get('display_name')
                    or Path(path).name
                ).strip()
                or None,
                kind=str(record.get('kind') or record.get('type') or '').strip() or None,
                source=source,
                artifact_ref=str(record.get('artifact_ref') or record.get('ref') or '').strip() or None,
            )
        )

    direct_path = str(payload.get('file_path') or '').strip()
    if direct_path:
        append_record(
            {
                'path': direct_path,
                'name': payload.get('file_name') or Path(direct_path).name,
                'kind': payload.get('file_kind'),
            },
            'local_path',
        )
    artifact_path = str(payload.get('artifact_path') or '').strip()
    if artifact_path:
        append_record(
            {
                'path': artifact_path,
                'name': Path(artifact_path).name,
                'artifact_ref': payload.get('artifact_ref'),
            },
            'artifact_reference',
        )

    for path in _parse_file_paths_json(payload.get('file_paths_json')):
        append_record(
            {
                'path': path,
                'name': Path(path).name,
            },
            'local_path',
        )

    for key, source in (
        ('input_artifacts', 'input_artifact'),
        ('reference_artifacts', 'selected_artifact'),
        ('selected_reference_artifact', 'selected_artifact'),
        ('selected_reference_artifacts', 'selected_artifact'),
        ('attachments', 'attachment'),
        ('files', 'file'),
        ('images', 'image'),
        ('audio', 'audio'),
    ):
        for record in _iter_file_records(payload.get(key)):
            append_record(record, source)
    for record in _walk_file_content_records(payload.get('input')):
        append_record(record, 'responses_content')
    return inputs


def validate_codex_text_request(
    request_payload: Any,
    *,
    upload_present: bool = False,
    files_enabled: bool = False,
) -> tuple[bool, Optional[str], Optional[str]]:
    payload = request_payload if isinstance(request_payload, Mapping) else {}
    requested_capability = str(payload.get('capability') or '').strip().lower()
    if requested_capability and requested_capability != 'chat':
        return (
            False,
            'CODEX_CAPABILITY_UNSUPPORTED',
            "The external ChatGPT target supports only capability 'chat'.",
        )
    if _has_value(payload.get('batch_prompts')):
        return (
            False,
            'CODEX_CAPABILITY_UNSUPPORTED',
            'ChatGPT through Ollmo accepts one current-turn request at a time.',
        )
    if _contains_remote_or_inline_file(payload):
        return (
            False,
            'CODEX_FILE_SOURCE_UNSUPPORTED',
            'Attach a local file or select an Ollmo artifact; remote URLs and inline data are not accepted.',
        )
    multipart_paths = _parse_file_paths_json(payload.get('file_paths_json'))
    if any(path.lower().startswith(('http://', 'https://', 'data:')) for path in multipart_paths):
        return (
            False,
            'CODEX_FILE_SOURCE_UNSUPPORTED',
            'Attach a local file or select an Ollmo artifact; remote URLs and inline data are not accepted.',
        )
    if not _content_block_is_supported(payload.get('input')):
        return (
            False,
            'CODEX_INPUT_UNSUPPORTED',
            'This Responses input block cannot be handed to ChatGPT through Ollmo.',
        )
    if codex_request_has_file_inputs(payload, upload_present=upload_present) and not files_enabled:
        return (
            False,
            'CODEX_FILE_CONSENT_REQUIRED',
            'Enable sharing of explicitly selected files with ChatGPT before sending this request.',
        )
    return True, None, None


def codex_execution_failure(
    result: CodexExecutionResult,
) -> tuple[int, str, str]:
    status = str(result.status.value)
    if status == 'auth_required':
        return (
            401,
            'CODEX_AUTH_REQUIRED',
            'Open ChatGPT or Codex, sign in, then try the request again.',
        )
    if status == 'unavailable':
        return (
            503,
            'CODEX_UNAVAILABLE',
            'Install or update ChatGPT/Codex, then refresh the integration status.',
        )
    if status == 'timed_out':
        return (
            504,
            'CODEX_TIMEOUT',
            'Try the request again, or increase OLLMO_CODEX_TIMEOUT_SEC before starting Ollmo.',
        )
    if status == 'empty_output':
        return (
            502,
            'CODEX_EMPTY_OUTPUT',
            'Try the request again; the ChatGPT route via Codex ended without a final text response.',
        )
    if status == 'output_limit_exceeded':
        return (
            502,
            'CODEX_OUTPUT_LIMIT',
            'Ask for a shorter answer, then try the request again.',
        )
    if status == 'invalid_request':
        diagnostic = str(result.diagnostic or '').strip()
        if diagnostic.startswith('codex_input') or diagnostic.startswith('invalid_codex_input'):
            return (
                400,
                'CODEX_INPUT_INVALID',
                'Check that every selected item is a readable regular file within the reported file limits.',
            )
        return (
            400,
            'CODEX_INVALID_REQUEST',
            'Provide a non-empty request.',
        )
    return (
        502,
        'CODEX_EXECUTION_FAILED',
        'Refresh the ChatGPT status, then try the request again.',
    )
