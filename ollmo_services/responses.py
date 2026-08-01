"""Pure helpers for Ollmo's canonical Responses payload contract."""

from __future__ import annotations

import json
import mimetypes
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator, Optional

from ollmo_services.artifact_contracts import sanitize_artifact_record


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
)

_INTERNAL_TEXT_ARTIFACT_INSTRUCTION_MARKERS = (
    'Target text artifact:',
    'Linked HTML artifact:',
    'Current saved target file content:',
    'Current saved HTML file content:',
    'Current saved CSS file content:',
    'Deterministic syntax sanity issues:',
    'HTML/CSS selector binding drift:',
    'Materialize only the requested text artifact.',
    'Return only the complete file payload.',
    'Update only the target text artifact.',
    'Do not output planner JSON.',
)

_TERMINAL_LATE_FILL_STATUSES = {
    'completed',
    'skipped',
}

_INTERNAL_BRANCH_STATUS_LINE_RE = re.compile(
    r'^branch-[A-Za-z0-9_.-]+-\d+\s*:\s*'
    r'(?:Image|Audio|Video|Artifact|File|Text|Document)\s+'
    r'(?:generated|saved|created|ready|complete|completed)\.?\s*$',
    re.IGNORECASE,
)

_ARTIFACT_BUNDLE_STATUS_TEXT_RE = re.compile(
    r'^(?:'
    r'all requested\b.*\bartifacts?\b.*\b(?:generated|created|saved|complete|completed)\.?|'
    r'artifacts?\s+(?:generated|created|saved|ready|complete|completed)\.?|'
    r'created the requested files\.?|'
    r'generated the requested files\.?|'
    r'the requested files (?:are|were) (?:generated|created|saved|ready|complete|completed)\.?'
    r')$',
    re.IGNORECASE,
)

_ARTIFACT_HANDOFF_SECTION_RE = re.compile(
    r'\b(?:image[_\s-]*prompts?|web[_\s-]*content[_\s-]*specification|target\s+files?)\s*:',
    re.IGNORECASE,
)
_ARTIFACT_PLACEHOLDER_TEXT_RE = re.compile(
    r'\b(?:i\s+need|need|requires?)\s+(?:the\s+)?(?:source/content|source|content|file\s+content|referenced\s+file)\b'
    r'[\s\S]{0,180}\b(?:before|to)\s+(?:i\s+can\s+)?(?:create|generate|materialize|update|fix)\s+(?:that|the|this)?\s*(?:artifact|file)?\b|'
    r'\bplease\s+(?:provide|select|attach)\s+(?:the\s+)?(?:source|content|file|html|css|code)\b',
    re.IGNORECASE,
)


def _saved_text_mime_type(path: str) -> str:
    normalized_path = str(path or '').strip()
    if not normalized_path:
        return 'text/markdown'
    guessed_mime, _encoding = mimetypes.guess_type(normalized_path)
    return str(guessed_mime or 'text/plain').strip() or 'text/plain'


def _looks_like_internal_text_artifact_instruction(value: Any) -> bool:
    text = str(value or '').strip()
    if not text:
        return False
    marker_count = sum(1 for marker in _INTERNAL_TEXT_ARTIFACT_INSTRUCTION_MARKERS if marker in text)
    if marker_count >= 2 and 'Target text artifact:' in text:
        return True
    lowered = text.lower()
    return (
        text.startswith('Materialize only the requested text artifact.')
        or (
            'target text artifact:' in lowered
            and (
                'syntax sanity' in lowered
                or 'current saved target file content:' in lowered
                or 'complete file payload' in lowered
                or 'update only the target text artifact' in lowered
            )
        )
    )


def _looks_like_internal_branch_status_summary(value: Any) -> bool:
    text = str(value or '').strip()
    if not text or not text.startswith('branch-'):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    matching = [line for line in lines if _INTERNAL_BRANCH_STATUS_LINE_RE.match(line)]
    if len(lines) == 1:
        return bool(matching)
    return len(matching) >= 2 and len(matching) == len(lines)


def _looks_like_internal_public_text(value: Any) -> bool:
    return (
        _looks_like_internal_text_artifact_instruction(value)
        or _looks_like_internal_branch_status_summary(value)
    )


def _looks_like_artifact_handoff_text(value: Any) -> bool:
    text = str(value or '').strip()
    if not text:
        return False
    if _ARTIFACT_HANDOFF_SECTION_RE.search(text):
        return True
    lowered = text.lower()
    lowered_spaced = lowered.replace('_', ' ')
    if 'image generation prompts' in lowered:
        return True
    if 'image prompts:' in lowered_spaced and 'web content specification:' in lowered_spaced:
        return True
    if (
        'image generation artifacts are available' in lowered
        and 'branch-image_generation-' in lowered
        and (
            'successful image generation' in lowered
            or 'no inline image payload' in lowered
            or 'saved externally' in lowered
        )
    ):
        return True
    if 'file materialization content' in lowered:
        return True
    if (
        any(marker in lowered for marker in ('```html', '```css', '```javascript', '```js'))
        and any(marker in lowered for marker in ('index.html', 'styles.css'))
    ):
        return True
    return False


def _looks_like_artifact_placeholder_text(value: Any) -> bool:
    text = str(value or '').strip()
    return bool(text and _ARTIFACT_PLACEHOLDER_TEXT_RE.search(text))


def _looks_like_provisional_artifact_bundle_text(value: Any) -> bool:
    text = str(value or '').strip()
    if not text:
        return False
    if _looks_like_internal_public_text(text) or _looks_like_artifact_placeholder_text(text):
        return True
    collapsed = re.sub(r'\s+', ' ', text).strip()
    if _ARTIFACT_BUNDLE_STATUS_TEXT_RE.match(collapsed):
        return True
    return _looks_like_artifact_handoff_text(text)


def _artifact_backed_text_value_should_hydrate(value: Any) -> bool:
    text = str(value or '').strip()
    if not text:
        return False
    return (
        _looks_like_internal_public_text(text)
        or _looks_like_artifact_handoff_text(text)
        or _looks_like_artifact_placeholder_text(text)
    )


def _truth_guard_requires_clarification(payload: Mapping[str, Any]) -> bool:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    truth_guard = runtime.get('truth_guard') if isinstance(runtime.get('truth_guard'), Mapping) else {}
    return str(truth_guard.get('status') or '').strip().lower() in {
        'clarification_required',
        'repair_required',
    }


def _output_item_is_internal_materialization(item: Mapping[str, Any]) -> bool:
    identifiers = ' '.join(
        str(item.get(key) or '').strip().lower()
        for key in ('slot_id', 'branch_id', 'phase_id')
    )
    if any(token in identifiers for token in ('repair-', 'branch-repair', 'text_artifact')):
        return True
    return False


def _output_item_is_public_post_artifact_text_follow_up(
    payload: Mapping[str, Any],
    item: Mapping[str, Any],
) -> bool:
    item_capability = str(item.get('follow_up_capability') or '').strip().lower()
    if item_capability and item_capability != 'chat':
        return False
    item_tokens = {
        str(item.get(key) or '').strip()
        for key in ('slot_id', 'branch_id', 'phase_id')
        if str(item.get(key) or '').strip()
    }
    if not item_tokens:
        return False

    def record_matches(raw_record: Any) -> bool:
        if not isinstance(raw_record, Mapping):
            return False
        record_tokens = {
            str(raw_record.get(key) or '').strip()
            for key in ('slot_id', 'branch_id', 'phase_id')
            if str(raw_record.get(key) or '').strip()
        }
        if not item_tokens.intersection(record_tokens):
            return False
        execution_contract = (
            raw_record.get('execution_contract')
            if isinstance(raw_record.get('execution_contract'), Mapping)
            else {}
        )
        capability = str(
            raw_record.get('capability')
            or execution_contract.get('capability')
            or ''
        ).strip().lower()
        role = str(
            raw_record.get('role')
            or execution_contract.get('role')
            or ''
        ).strip().lower()
        return capability == 'chat' and role == 'post_artifact_text_follow_up'

    late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
    for raw_result in reversed(late_fill.get('fill_results') or []):
        if record_matches(raw_result):
            return True

    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    request_phase_graph = (
        runtime.get('request_phase_graph')
        if isinstance(runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    for collection_key in ('downstream_branches', 'phases'):
        for raw_branch in reversed(request_phase_graph.get(collection_key) or []):
            if record_matches(raw_branch):
                return True
    return False


def _output_item_is_fulfilled_repair_materialization(item: Mapping[str, Any]) -> bool:
    if not isinstance(item, Mapping):
        return False
    identifiers = ' '.join(
        str(item.get(key) or '').strip().lower()
        for key in ('slot_id', 'branch_id', 'phase_id')
    )
    if 'repair-' not in identifiers:
        return False
    output_type = str(item.get('type') or '').strip().lower()
    if output_type not in {'text', 'document'}:
        return False
    status = str(item.get('status') or '').strip().lower()
    return status in {'fulfilled', 'completed'}


def _output_item_is_provisional_artifact_bundle_text(item: Mapping[str, Any]) -> bool:
    if not isinstance(item, Mapping):
        return False
    output_type = str(item.get('type') or '').strip().lower()
    if output_type not in {'text', 'document'}:
        return False
    status = str(item.get('status') or '').strip().lower()
    if status not in {'fulfilled', 'completed'}:
        return False
    if _artifact_ref_token(item):
        return False
    return _looks_like_provisional_artifact_bundle_text(item.get('value') or item.get('text') or item.get('content'))


def _suppress_provisional_artifact_bundle_text_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not outputs:
        return outputs
    has_artifact_backed_output = any(
        isinstance(output, Mapping)
        and bool(_artifact_ref_token(output))
        for output in outputs
    )
    if not has_artifact_backed_output:
        return outputs
    filtered = [
        output
        for output in outputs
        if not _output_item_is_provisional_artifact_bundle_text(output)
    ]
    return filtered or outputs


def _artifact_text_content_value(artifact: Mapping[str, Any]) -> str:
    if not isinstance(artifact, Mapping):
        return ''
    for key in ('content', 'value', 'text', 'output_text', 'result_text'):
        value = artifact.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    path_value = str(
        artifact.get('path')
        or artifact.get('saved_text_path')
        or artifact.get('file_path')
        or ''
    ).strip()
    if not path_value:
        return ''
    suffix = Path(path_value).suffix.lower().lstrip('.')
    mime_type = str(artifact.get('mime_type') or mimetypes.guess_type(path_value)[0] or '').lower()
    if suffix not in {'html', 'htm', 'css', 'js', 'mjs', 'cjs', 'json', 'md', 'markdown', 'txt'} and not mime_type.startswith('text/'):
        return ''
    try:
        path = Path(path_value)
        if not path.is_file() or path.stat().st_size > 1_000_000:
            return ''
        return path.read_text(encoding='utf-8', errors='replace').strip()
    except (OSError, ValueError):
        return ''
    return ''


def _hydrate_artifact_backed_text_output_values(
    outputs: list[dict[str, Any]],
    artifacts: list[Any],
) -> list[dict[str, Any]]:
    if not outputs or not artifacts:
        return outputs
    artifacts_by_ref = {
        _artifact_ref_value(artifact): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and _artifact_ref_value(artifact)
    }
    if not artifacts_by_ref:
        return outputs
    hydrated: list[dict[str, Any]] = []
    for raw_output in outputs:
        if not isinstance(raw_output, dict):
            hydrated.append(raw_output)
            continue
        output = dict(raw_output)
        output_type = str(output.get('type') or '').strip().lower()
        artifact_ref = _artifact_ref_token(output)
        if output_type in {'text', 'document'} and artifact_ref:
            current_value = str(
                output.get('value')
                or output.get('content')
                or output.get('text')
                or ''
            ).strip()
            artifact_content = _artifact_text_content_value(artifacts_by_ref.get(artifact_ref) or {})
            should_hydrate = _artifact_backed_text_value_should_hydrate(current_value)
            if artifact_content and should_hydrate:
                output['value'] = artifact_content
            elif should_hydrate and current_value:
                for key in ('value', 'content', 'text', 'output_text', 'result_text'):
                    output.pop(key, None)
        hydrated.append(output)
    return hydrated


def _terminal_materialization_is_fulfilled(payload: Mapping[str, Any]) -> bool:
    late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
    if not late_fill:
        return False
    if late_fill.get('pending_branches') or late_fill.get('active_branches'):
        return False
    status = str(late_fill.get('status') or '').strip().lower()
    contract_status = str(late_fill.get('final_materialization_contract_status') or '').strip().lower()
    if contract_status == 'fulfilled':
        return True
    return status in _TERMINAL_LATE_FILL_STATUSES and bool(payload.get('artifacts'))


def _generic_terminal_public_output_text(payload: Mapping[str, Any]) -> str:
    artifacts = payload.get('artifacts') if isinstance(payload.get('artifacts'), list) else []
    if len(artifacts) == 1:
        return 'Artifact generated.'
    if artifacts:
        return 'Artifacts generated.'
    return 'Completed.'


def select_public_output_text(
    payload: Mapping[str, Any],
    outputs: Any,
    *,
    fallback_text: Optional[str] = None,
) -> str:
    response_payload = payload if isinstance(payload, Mapping) else {}
    fallback = str(
        response_payload.get('output_text') if fallback_text is None else fallback_text
    or '').strip()
    if _truth_guard_requires_clarification(response_payload):
        return fallback
    if isinstance(outputs, list):
        for item in reversed(outputs):
            if not isinstance(item, Mapping):
                continue
            output_type = str(item.get('type') or '').strip().lower()
            status = str(item.get('status') or '').strip().lower()
            value = str(item.get('value') or '').strip()
            if output_type not in {'text', 'document'} or status not in {'completed', 'fulfilled'} or not value:
                continue
            if (
                _artifact_ref_token(item)
                and not _output_item_is_public_post_artifact_text_follow_up(response_payload, item)
            ):
                continue
            if _output_item_is_internal_materialization(item):
                continue
            if _looks_like_internal_public_text(value):
                continue
            return value
    if (
        fallback
        and bool(response_payload.get('artifacts'))
        and _looks_like_provisional_artifact_bundle_text(fallback)
    ):
        return _generic_terminal_public_output_text(response_payload)
    if fallback and not _looks_like_internal_public_text(fallback):
        return fallback
    if fallback and _terminal_materialization_is_fulfilled(response_payload):
        return _generic_terminal_public_output_text(response_payload)
    return fallback


def _extract_semantic_phase_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    nested = source.get('phase_payload') if isinstance(source.get('phase_payload'), dict) else {}
    semantic_payload: dict[str, Any] = {}
    for key in _SEMANTIC_PHASE_PAYLOAD_KEYS:
        value = source.get(key)
        if value in (None, '', [], {}):
            value = nested.get(key)
        if value not in (None, '', [], {}):
            semantic_payload[key] = value
    return semantic_payload

def _parse_jsonish_field(raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, dict)):
        return raw_value
    text = str(raw_value or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_responses_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                item_type = str(item.get('type') or '').strip()
                if item_type in {'input_text', 'text', 'output_text'}:
                    text = str(item.get('text') or '').strip()
                    if text:
                        parts.append(text)
            else:
                text = str(item).strip()
                if text:
                    parts.append(text)
        return '\n'.join(parts).strip()
    if isinstance(value, dict):
        item_type = str(value.get('type') or '').strip()
        if item_type == 'message':
            return extract_responses_content_text(value.get('content'))
        for key in ('prompt', 'input', 'text', 'content'):
            if key in value:
                return extract_responses_content_text(value.get(key))
    return ''


def extract_responses_messages(payload: dict) -> list[dict]:
    messages: list[dict] = []
    instructions = str(payload.get('instructions') or '').strip()
    if instructions:
        messages.append({'role': 'system', 'content': instructions})

    prompt = str(payload.get('prompt') or '').strip()

    input_payload = payload.get('input')
    if isinstance(input_payload, str):
        text = input_payload.strip()
        if text:
            messages.append({'role': 'user', 'content': text})
        elif prompt:
            messages.append({'role': 'user', 'content': prompt})
        return messages

    if not isinstance(input_payload, list):
        if prompt:
            messages.append({'role': 'user', 'content': prompt})
        return messages

    for item in input_payload:
        if isinstance(item, dict) and str(item.get('type') or '').strip() == 'message':
            role = str(item.get('role') or 'user').strip() or 'user'
            content = extract_responses_content_text(item.get('content'))
            if content:
                messages.append({'role': role, 'content': content})
            continue
        content = extract_responses_content_text(item)
        if content:
            messages.append({'role': 'user', 'content': content})
    if not any(str(item.get('role') or '').strip().lower() == 'user' and str(item.get('content') or '').strip() for item in messages) and prompt:
        messages.append({'role': 'user', 'content': prompt})
    return messages


def extract_responses_current_turn_prompt(payload: dict) -> str:
    messages = extract_responses_messages(payload)
    for item in reversed(messages):
        role = str(item.get('role') or '').strip().lower()
        content = str(item.get('content') or '').strip()
        if role == 'user' and content:
            return content
    explicit_prompt = str(payload.get('prompt') or '').strip()
    if explicit_prompt:
        return explicit_prompt
    return str(payload.get('input') or payload.get('instructions') or '').strip()


def build_responses_output(
    text: str,
    model_name: str,
    *,
    response_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict:
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    message_id = message_id or f"msg_{uuid.uuid4().hex}"
    return {
        'id': response_id,
        'object': 'response',
        'status': 'completed',
        'model': model_name,
        'output': [
            {
                'id': message_id,
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [
                    {
                        'type': 'output_text',
                        'text': text,
                        'annotations': [],
                    }
                ],
            }
        ],
        'output_text': text,
        'parallel_tool_calls': False,
        'usage': {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
        },
    }


def _stream_in_progress_response_payload(payload: dict, response_id: str) -> dict:
    in_progress_payload: dict = {
        'id': response_id,
        'object': payload.get('object') or 'response',
        'status': 'in_progress',
        'output': [],
    }
    for key in (
        'model',
        'backend',
        'capability',
        'mode',
        'instance_id',
        'created_at',
        'response_id',
        'state_version',
    ):
        value = payload.get(key)
        if value not in (None, '', [], {}):
            in_progress_payload[key] = value
    return in_progress_payload


def build_canonical_response_stream_events(response_payload: dict) -> Iterator[str]:
    payload = dict(response_payload or {})
    response_id = str(payload.get('id') or f"resp_{uuid.uuid4().hex}")
    output_items = payload.get('output') if isinstance(payload.get('output'), list) else []
    first_item = output_items[0] if output_items and isinstance(output_items[0], dict) else None
    message_id = str((first_item or {}).get('id') or f"msg_{uuid.uuid4().hex}")
    output_text = str(payload.get('output_text') or '').strip()
    content_part = (
        ((first_item or {}).get('content') or [{}])[0]
        if isinstance((first_item or {}).get('content'), list) and (first_item or {}).get('content')
        else {'type': 'output_text', 'text': output_text, 'annotations': []}
    )
    in_progress_payload = _stream_in_progress_response_payload(payload, response_id)

    yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': in_progress_payload}, ensure_ascii=False)}\n\n"
    yield f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': in_progress_payload}, ensure_ascii=False)}\n\n"
    yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': {'id': message_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []}}, ensure_ascii=False)}\n\n"
    yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}}, ensure_ascii=False)}\n\n"
    if output_text:
        yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'delta': output_text}, ensure_ascii=False)}\n\n"
    yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': output_text}, ensure_ascii=False)}\n\n"
    yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': content_part}, ensure_ascii=False)}\n\n"
    completed_item = first_item or {
        'id': message_id,
        'type': 'message',
        'status': 'completed',
        'role': 'assistant',
        'content': [content_part],
    }
    yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': completed_item}, ensure_ascii=False)}\n\n"
    yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': {**payload, 'id': response_id}}, ensure_ascii=False)}\n\n"


def build_public_output_branches_from_slots(output_slots: Any) -> list[dict[str, Any]]:
    if not isinstance(output_slots, list):
        return []
    branches: list[dict[str, Any]] = []
    for raw_slot in output_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        payload = {
            'slot_id': str(raw_slot.get('slot_id') or '').strip() or None,
            'branch_id': str(raw_slot.get('branch_id') or raw_slot.get('phase_id') or '').strip() or None,
            'phase_id': str(raw_slot.get('phase_id') or raw_slot.get('branch_id') or '').strip() or None,
            'obligation_id': str(raw_slot.get('obligation_id') or '').strip() or None,
            'type': str(raw_slot.get('type') or '').strip().lower() or None,
            'status': str(raw_slot.get('status') or '').strip().lower() or None,
            'lifecycle': str(raw_slot.get('lifecycle') or '').strip().lower() or None,
            'follow_up_capability': str(raw_slot.get('follow_up_capability') or '').strip().lower() or None,
            'artifact_ref': str(raw_slot.get('artifact_ref') or '').strip() or None,
            'artifact_path': str(raw_slot.get('artifact_path') or '').strip() or None,
            'placeholder_ref': str(raw_slot.get('placeholder_ref') or '').strip() or None,
            'blocked_reason': str(raw_slot.get('blocked_reason') or '').strip() or None,
            'parent_slot_id': str(raw_slot.get('parent_slot_id') or '').strip() or None,
            'child_slot_ids': list(raw_slot.get('child_slot_ids') or []) if isinstance(raw_slot.get('child_slot_ids'), list) else [],
        }
        if isinstance(raw_slot.get('error_ref'), Mapping):
            error_ref = {
                'branch_id': str(raw_slot['error_ref'].get('branch_id') or '').strip() or None,
                'code': str(raw_slot['error_ref'].get('code') or '').strip() or None,
                'stage': str(raw_slot['error_ref'].get('stage') or '').strip() or None,
            }
            payload['error_ref'] = {
                key: value
                for key, value in error_ref.items()
                if value not in (None, '', [], {})
            }
        if isinstance(raw_slot.get('recovery_context'), Mapping):
            recovery_context = {
                'can_retry': raw_slot['recovery_context'].get('can_retry')
                if isinstance(raw_slot['recovery_context'].get('can_retry'), bool)
                else None,
                'retry_scope': str(raw_slot['recovery_context'].get('retry_scope') or '').strip() or None,
                'suggested_action': str(raw_slot['recovery_context'].get('suggested_action') or '').strip() or None,
                'preserve_intent': raw_slot['recovery_context'].get('preserve_intent')
                if isinstance(raw_slot['recovery_context'].get('preserve_intent'), bool)
                else None,
                'exclude_instance_ids': [
                    str(item).strip()
                    for item in (raw_slot['recovery_context'].get('exclude_instance_ids') or [])
                    if str(item).strip()
                ] if isinstance(raw_slot['recovery_context'].get('exclude_instance_ids'), list) else [],
            }
            payload['recovery_context'] = {
                key: value
                for key, value in recovery_context.items()
                if value not in (None, '', [], {})
            }
        if isinstance(raw_slot.get('recovery_state'), Mapping):
            recovery_state = {
                'kind': str(raw_slot['recovery_state'].get('kind') or '').strip() or None,
                'status': str(raw_slot['recovery_state'].get('status') or '').strip().lower() or None,
                'trigger': str(raw_slot['recovery_state'].get('trigger') or '').strip() or None,
                'branch_id': str(raw_slot['recovery_state'].get('branch_id') or '').strip() or None,
                'capability': str(raw_slot['recovery_state'].get('capability') or '').strip().lower() or None,
                'retry_scope': str(raw_slot['recovery_state'].get('retry_scope') or '').strip() or None,
                'suggested_action': str(raw_slot['recovery_state'].get('suggested_action') or '').strip() or None,
                'failed_instance_id': str(raw_slot['recovery_state'].get('failed_instance_id') or '').strip() or None,
                'promotion_required': raw_slot['recovery_state'].get('promotion_required')
                if isinstance(raw_slot['recovery_state'].get('promotion_required'), bool)
                else None,
                'auto_execute': raw_slot['recovery_state'].get('auto_execute')
                if isinstance(raw_slot['recovery_state'].get('auto_execute'), bool)
                else None,
                'preserve_intent': raw_slot['recovery_state'].get('preserve_intent')
                if isinstance(raw_slot['recovery_state'].get('preserve_intent'), bool)
                else None,
                'exclude_instance_ids': [
                    str(item).strip()
                    for item in (raw_slot['recovery_state'].get('exclude_instance_ids') or [])
                    if str(item).strip()
                ] if isinstance(raw_slot['recovery_state'].get('exclude_instance_ids'), list) else [],
            }
            payload['recovery_state'] = {
                key: value
                for key, value in recovery_state.items()
                if value not in (None, '', [], {})
            }
        if not any(payload.get(key) for key in ('slot_id', 'branch_id', 'phase_id', 'artifact_ref', 'placeholder_ref')):
            continue
        branches.append(
            {
                key: value
                for key, value in payload.items()
                if value not in (None, '', [], {})
            }
        )
    return branches


def _artifact_ref_value(artifact: Mapping[str, Any]) -> str:
    if not isinstance(artifact, Mapping):
        return ''
    return str(
        artifact.get('artifact_ref')
        or artifact.get('ref')
        or artifact.get('artifact_id')
        or ''
    ).strip()


_NON_PUBLIC_OUTPUT_STATUSES = {
    'blocked',
    'cancelled',
    'failed',
    'open',
    'partial_failed',
    'pending',
    'repair_needed',
    'rejected',
    'skipped',
    'superseded',
    'waived',
}

_IGNORED_LOCAL_REFERENCE_SCHEMES = (
    'http:',
    'https:',
    'data:',
    'blob:',
    'mailto:',
    'tel:',
    'javascript:',
)

_HTML_REFERENCE_RE = re.compile(
    r'\b(?:src|href|poster)\s*=\s*(?P<quote>["\'])(?P<value>[^"\']*)(?P=quote)',
    re.IGNORECASE,
)
_HTML_SRCSET_RE = re.compile(
    r'\bsrcset\s*=\s*(?P<quote>["\'])(?P<value>[^"\']*)(?P=quote)',
    re.IGNORECASE,
)
_CSS_URL_RE = re.compile(
    r'url\(\s*(?P<quote>["\']?)(?P<value>[^)"\']+)(?P=quote)\s*\)',
    re.IGNORECASE,
)
_CSS_IMPORT_RE = re.compile(
    r'@import\s+(?P<quote>["\'])(?P<value>[^"\']+)(?P=quote)',
    re.IGNORECASE,
)


def _status_token(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_').replace(' ', '_')


def _artifact_public_keys(record: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in ('artifact_ref', 'ref', 'artifact_id'):
        token = str(record.get(key) or '').strip()
        if token:
            keys.add(f'ref:{token}')
            if key == 'artifact_id':
                keys.add(f'ref:artifact:{token}')
    for key in ('path', 'source_path', 'saved_image_path', 'saved_audio_path', 'saved_text_path'):
        token = str(record.get(key) or '').strip()
        if token:
            expanded = Path(token).expanduser()
            keys.add(f'path:{expanded}')
            try:
                keys.add(f'path:{expanded.resolve(strict=False)}')
            except OSError:
                pass
    if not any(key.startswith('ref:') for key in keys):
        sanitized = sanitize_artifact_record(record)
        if sanitized and sanitized is not record:
            for key in ('artifact_ref', 'ref', 'artifact_id'):
                token = str(sanitized.get(key) or '').strip()
                if token:
                    keys.add(f'ref:{token}')
                    if key == 'artifact_id':
                        keys.add(f'ref:artifact:{token}')
    return keys


def _artifact_ref_token(record: Mapping[str, Any]) -> str:
    return str(
        record.get('artifact_ref')
        or record.get('ref')
        or (
            f'artifact:{record.get("artifact_id")}'
            if str(record.get('artifact_id') or '').strip()
            else ''
        )
        or ''
    ).strip()


def _is_generated_image_text_misbinding(record: Mapping[str, Any]) -> bool:
    artifact_type = _status_token(record.get('type') or record.get('kind'))
    if artifact_type not in {'text', 'document'}:
        return False
    artifact_ref = _artifact_ref_token(record)
    artifact_id = str(record.get('artifact_id') or '').strip()
    return (
        artifact_ref.startswith('artifact:text_generated_image_')
        or artifact_id.startswith('text_generated_image_')
    )


def _is_public_output_artifact_record(record: Mapping[str, Any]) -> bool:
    if not isinstance(record, Mapping):
        return False
    if _is_generated_image_text_misbinding(record):
        return False
    status = _status_token(record.get('status') or record.get('state'))
    if status in _NON_PUBLIC_OUTPUT_STATUSES:
        return False
    if bool(record.get('compatibility_derived')):
        return False
    source = _status_token(record.get('source'))
    if source in {'compatibility_derived', 'raw_saved_artifact_fallback'}:
        return False
    return bool(_artifact_public_keys(record))


def _iter_output_artifact_records(output: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if not isinstance(output, Mapping):
        return
    artifacts = output.get('artifacts')
    if isinstance(artifacts, Mapping):
        yield artifacts
    elif isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, Mapping):
                yield artifact


def _artifact_type_matches_output_type(artifact_type: Any, output_type: Any) -> bool:
    artifact_token = _status_token(artifact_type)
    output_token = _status_token(output_type)
    if not artifact_token or not output_token:
        return False
    if artifact_token == output_token:
        return True
    return {artifact_token, output_token} <= {'text', 'document'}


def _alias_artifact_to_output_ref(artifact: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(artifact)
    output_ref = _artifact_ref_token(output)
    if not output_ref:
        return payload
    existing_ref = _artifact_ref_token(payload)
    if existing_ref and existing_ref != output_ref:
        aliases = [
            str(item).strip()
            for item in (payload.get('alias_artifact_refs') or [])
            if str(item).strip()
        ] if isinstance(payload.get('alias_artifact_refs'), list) else []
        if existing_ref not in aliases:
            aliases.append(existing_ref)
        payload['alias_artifact_refs'] = aliases
    payload['artifact_ref'] = output_ref
    payload['ref'] = output_ref
    return payload


def _is_ignored_local_reference(value: Any) -> bool:
    token = str(value or '').strip()
    if not token or token.startswith('#'):
        return True
    lowered = token.lower()
    return lowered.startswith(_IGNORED_LOCAL_REFERENCE_SCHEMES) or lowered.startswith('//')


def _split_local_reference(value: Any) -> str:
    token = str(value or '').strip().replace('\\', '/')
    if not token:
        return ''
    return re.split(r'[?#]', token, maxsplit=1)[0].strip()


def _iter_text_file_local_references(path: Path) -> Iterator[str]:
    extension = path.suffix.lower().lstrip('.')
    if extension not in {'html', 'htm', 'css', 'js', 'mjs', 'cjs'}:
        return
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return
    if extension in {'html', 'htm'}:
        for match in _HTML_REFERENCE_RE.finditer(text):
            value = match.group('value')
            if not _is_ignored_local_reference(value):
                yield value
        for match in _HTML_SRCSET_RE.finditer(text):
            for raw_part in match.group('value').split(','):
                token = raw_part.strip().split()
                if token and not _is_ignored_local_reference(token[0]):
                    yield token[0]
        for match in _CSS_URL_RE.finditer(text):
            value = match.group('value')
            if not _is_ignored_local_reference(value):
                yield value
    if extension in {'css', 'html', 'htm'}:
        for match in _CSS_URL_RE.finditer(text):
            value = match.group('value')
            if not _is_ignored_local_reference(value):
                yield value
        for match in _CSS_IMPORT_RE.finditer(text):
            value = match.group('value')
            if not _is_ignored_local_reference(value):
                yield value
    if extension in {'js', 'mjs', 'cjs'}:
        for match in re.finditer(
            r'(?P<quote>["\'])(?P<value>[^"\']+\.(?:png|jpe?g|webp|gif|svg|css|js|mjs|wav|mp3|m4a|ogg|json))(?P=quote)',
            text,
            re.IGNORECASE,
        ):
            value = match.group('value')
            if not _is_ignored_local_reference(value):
                yield value


def _artifact_type_for_dependency(path: Path) -> str:
    extension = path.suffix.lower().lstrip('.')
    if extension in {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'svg'}:
        return 'image'
    if extension in {'wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'oga', 'opus'}:
        return 'audio'
    if extension in {'html', 'htm', 'css', 'js', 'mjs', 'cjs', 'json', 'md', 'txt', 'xml'}:
        return 'text'
    return 'file'


def _dependency_artifact_record(path: Path) -> Optional[dict[str, Any]]:
    artifact_type = _artifact_type_for_dependency(path)
    mime_type, _encoding = mimetypes.guess_type(str(path))
    record = sanitize_artifact_record(
        {
            'type': artifact_type,
            'kind': artifact_type,
            'path': str(path),
            'source_path': str(path),
            'name': path.stem,
            'mime_type': mime_type or None,
            'origin': 'linked_public_dependency',
            'source': 'linked_public_dependency',
        }
    )
    return record


def _resolved_artifact_path(record: Mapping[str, Any]) -> Optional[Path]:
    token = str(
        record.get('path')
        or record.get('source_path')
        or record.get('saved_image_path')
        or record.get('saved_audio_path')
        or record.get('saved_text_path')
        or ''
    ).strip()
    if not token:
        return None
    try:
        path = Path(token).expanduser().resolve(strict=False)
    except OSError:
        return None
    return path if path.exists() and path.is_file() else None


def _include_linked_public_dependencies(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not artifacts:
        return artifacts
    by_path: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    queue: list[Path] = []
    for artifact in artifacts:
        path = _resolved_artifact_path(artifact)
        path_key = str(path) if path else ''
        if path_key and path_key in by_path:
            continue
        if path_key:
            by_path[path_key] = artifact
            extension = path.suffix.lower().lstrip('.')
            if extension in {'html', 'htm', 'css', 'js', 'mjs', 'cjs'}:
                queue.append(path)
        ordered.append(artifact)

    scanned: set[str] = set()
    while queue:
        source = queue.pop(0)
        source_key = str(source)
        if source_key in scanned:
            continue
        scanned.add(source_key)
        for raw_ref in _iter_text_file_local_references(source):
            if _is_ignored_local_reference(raw_ref):
                continue
            ref_path = _split_local_reference(raw_ref)
            if not ref_path:
                continue
            try:
                dependency = (source.parent / ref_path).expanduser().resolve(strict=False)
            except OSError:
                continue
            if not dependency.exists() or not dependency.is_file():
                continue
            dependency_key = str(dependency)
            if dependency_key in by_path:
                continue
            record = _dependency_artifact_record(dependency)
            if not record:
                continue
            by_path[dependency_key] = record
            ordered.append(record)
            if dependency.suffix.lower().lstrip('.') in {'html', 'htm', 'css', 'js', 'mjs', 'cjs'}:
                queue.append(dependency)
    return ordered


def _has_authoritative_output_surface(payload: Mapping[str, Any], outputs: list[Mapping[str, Any]]) -> bool:
    if isinstance(payload.get('output_slots'), list) and payload.get('output_slots'):
        return True
    if isinstance(payload.get('output_branches'), list) and payload.get('output_branches'):
        return True
    if any(
        isinstance(output, Mapping)
        and not bool(output.get('compatibility_derived'))
        and _status_token(output.get('source')) != 'compatibility_derived'
        for output in outputs
    ):
        return True
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), Mapping) else {}
    frame_output = frame.get('output') if isinstance(frame.get('output'), Mapping) else {}
    frame_outputs = frame_output.get('outputs') if isinstance(frame_output.get('outputs'), list) else []
    if any(
        isinstance(output, Mapping)
        and not bool(output.get('compatibility_derived'))
        and _status_token(output.get('source')) != 'compatibility_derived'
        for output in frame_outputs
    ):
        return True
    planning = frame.get('planning') if isinstance(frame.get('planning'), Mapping) else {}
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), Mapping) else {}
    return bool(artifact_flow.get('output_slots') if isinstance(artifact_flow.get('output_slots'), list) else [])


def filter_public_response_artifacts(
    payload: Mapping[str, Any],
    artifacts: Any,
    *,
    outputs: Any = None,
) -> list[dict[str, Any]]:
    """Return artifacts that are bound to fulfilled public output truth.

    Raw saved artifacts can contain repair attempts, intermediate materializations, or
    misbound branch files. Those remain available through diagnostics and dossiers,
    but the public response/bundle surface should only expose artifacts whose ref or
    path is named by a usable canonical output.
    """

    response_payload = payload if isinstance(payload, Mapping) else {}
    normalized_artifacts = [dict(item) for item in artifacts if isinstance(item, Mapping)] if isinstance(artifacts, list) else []
    normalized_outputs = [item for item in outputs if isinstance(item, Mapping)] if isinstance(outputs, list) else []
    if not normalized_outputs and isinstance(response_payload.get('outputs'), list):
        normalized_outputs = [
            item for item in response_payload.get('outputs') or []
            if isinstance(item, Mapping)
        ]
    if not _has_authoritative_output_surface(response_payload, normalized_outputs):
        return normalized_artifacts
    public_keys: set[str] = set()
    for output in normalized_outputs:
        if not _is_public_output_artifact_record(output):
            continue
        public_keys.update(_artifact_public_keys(output))
        for artifact in _iter_output_artifact_records(output):
            if _is_public_output_artifact_record(artifact):
                public_keys.update(_artifact_public_keys(artifact))
    if not public_keys:
        return []
    public_artifacts = [
        artifact
        for artifact in normalized_artifacts
        if _artifact_public_keys(artifact) & public_keys
    ]
    if not public_artifacts and public_keys:
        matched_fallback_artifacts: list[dict[str, Any]] = []
        used_indexes: set[int] = set()
        for output in normalized_outputs:
            if not _is_public_output_artifact_record(output):
                continue
            if not _artifact_ref_token(output):
                continue
            output_type = output.get('type') or output.get('kind')
            candidates = [
                (index, artifact)
                for index, artifact in enumerate(normalized_artifacts)
                if index not in used_indexes
                and _is_public_output_artifact_record(artifact)
                and _artifact_type_matches_output_type(artifact.get('type') or artifact.get('kind'), output_type)
            ]
            if len(candidates) != 1:
                continue
            index, artifact = candidates[0]
            used_indexes.add(index)
            matched_fallback_artifacts.append(_alias_artifact_to_output_ref(artifact, output))
        if matched_fallback_artifacts:
            return _include_linked_public_dependencies(matched_fallback_artifacts)
    return _include_linked_public_dependencies(public_artifacts)


def _group_artifacts_for_outputs(artifacts: Any) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, Optional[int]], list[dict[str, Any]]]]:
    by_ref: dict[str, dict[str, Any]] = {}
    by_type_and_batch: dict[tuple[str, Optional[int]], list[dict[str, Any]]] = {}
    if not isinstance(artifacts, list):
        return by_ref, by_type_and_batch
    for raw_artifact in artifacts:
        artifact = _sanitize_artifact_record_for_outputs(raw_artifact)
        if not artifact:
            continue
        artifact_ref = _artifact_ref_value(artifact)
        if artifact_ref:
            by_ref[artifact_ref] = artifact
        artifact_type = str(artifact.get('type') or '').strip().lower()
        batch_index = artifact.get('batch_index')
        normalized_batch_index = None
        if batch_index not in (None, ''):
            try:
                normalized_batch_index = int(batch_index)
            except (TypeError, ValueError):
                normalized_batch_index = None
        if artifact_type:
            by_type_and_batch.setdefault((artifact_type, normalized_batch_index), []).append(artifact)
    return by_ref, by_type_and_batch


def _normalize_slot_batch_index(value: Any) -> Optional[int]:
    if value in (None, ''):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _slot_allows_implicit_artifact_assignment(slot: Mapping[str, Any]) -> bool:
    slot_type = str(slot.get('type') or '').strip().lower()
    if slot_type not in {'text', 'document'}:
        return True

    def clean(value: Any) -> str:
        return str(value or '').strip()

    slot_tokens = [
        clean(slot.get('branch_id')),
        clean(slot.get('phase_id')),
        clean(slot.get('slot_id')),
    ]
    for token in slot_tokens:
        normalized = token.lower()
        if normalized.startswith('branch-text_artifact-'):
            return True
        if normalized.startswith('branch-terminal-inline_text_artifact-'):
            return True

    role = clean(slot.get('role') or slot.get('output_role')).lower()
    if role == 'text_artifact_output':
        return True

    fulfillment_policy = clean(slot.get('fulfillment_policy')).lower()
    if fulfillment_policy in {'text_artifact', 'text_artifact_output'}:
        return True

    stage_direction = clean(slot.get('stage_direction')).lower()
    if stage_direction == 'materialize_requested_text_artifact':
        return True

    for key in (
        'text_artifact_extension',
        'text_artifact_source_name',
        'text_artifact_target_path',
        'text_artifact_source',
    ):
        if clean(slot.get(key)):
            return True

    artifact_request = slot.get('artifact_request')
    if isinstance(artifact_request, Mapping) and any(
        clean(artifact_request.get(key))
        for key in ('extension', 'source_name', 'target_path', 'path')
    ):
        return True

    return False


def _compact_output_slots_for_public_outputs(output_slots: Any) -> list[Mapping[str, Any]]:
    if not isinstance(output_slots, list):
        return []

    slot_order: list[str] = []
    latest_by_key: dict[str, Mapping[str, Any]] = {}
    anonymous_index = 0
    for raw_slot in output_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot_key = ''
        for key in ('slot_id', 'branch_id', 'phase_id'):
            token = str(raw_slot.get(key) or '').strip()
            if token:
                slot_key = f'{key}:{token}'
                break
        if not slot_key:
            slot_key = f'index:{anonymous_index}'
            anonymous_index += 1
        if slot_key not in latest_by_key:
            slot_order.append(slot_key)
        latest_by_key[slot_key] = raw_slot

    return [latest_by_key[key] for key in slot_order]


def _assign_output_artifacts_from_slots(
    output_slots: Any,
    artifacts: Any,
    *,
    output_branches: Any = None,
) -> list[dict[str, Any]]:
    normalized_slots = output_slots if isinstance(output_slots, list) else []
    normalized_output_branches = output_branches if isinstance(output_branches, list) else []
    artifacts_by_ref, artifacts_by_type_and_batch = _group_artifacts_for_outputs(artifacts)
    artifacts_by_branch_and_type: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if isinstance(artifacts, list):
        for raw_artifact in artifacts:
            artifact = _sanitize_artifact_record_for_outputs(raw_artifact)
            if not artifact:
                continue
            artifact_type = str(artifact.get('type') or '').strip().lower()
            if not artifact_type:
                continue
            for key in ('branch_id', 'phase_id', 'slot_id'):
                token = str(artifact.get(key) or '').strip()
                if token:
                    artifacts_by_branch_and_type.setdefault((token, artifact_type), []).append(artifact)
    assigned_refs: set[str] = set()
    outputs: list[dict[str, Any]] = []
    branch_lookup_by_slot: dict[str, Mapping[str, Any]] = {}
    for raw_branch in normalized_output_branches:
        if not isinstance(raw_branch, Mapping):
            continue
        slot_id = str(raw_branch.get('slot_id') or '').strip()
        if slot_id:
            branch_lookup_by_slot[slot_id] = raw_branch

    def collect_slot_artifacts(slot: Mapping[str, Any]) -> list[dict[str, Any]]:
        slot_artifacts: list[dict[str, Any]] = []
        explicit_ref = str(slot.get('artifact_ref') or '').strip()
        slot_type = str(slot.get('type') or '').strip().lower()
        missing_explicit_text_ref = False
        if explicit_ref:
            artifact = artifacts_by_ref.get(explicit_ref)
            if artifact and explicit_ref not in assigned_refs:
                assigned_refs.add(explicit_ref)
                slot_artifacts.append(artifact)
                return slot_artifacts
            if slot_type in {'text', 'document'}:
                missing_explicit_text_ref = True
        if not _slot_allows_implicit_artifact_assignment(slot):
            return slot_artifacts
        if slot_type:
            for key in ('branch_id', 'phase_id', 'slot_id'):
                token = str(slot.get(key) or '').strip()
                if not token:
                    continue
                candidates = artifacts_by_branch_and_type.get((token, slot_type)) or []
                while candidates:
                    artifact = candidates.pop(0)
                    artifact_ref = _artifact_ref_value(artifact)
                    if artifact_ref and artifact_ref in assigned_refs:
                        continue
                    if artifact_ref:
                        assigned_refs.add(artifact_ref)
                    slot_artifacts.append(artifact)
                    return slot_artifacts
        if missing_explicit_text_ref:
            return slot_artifacts
        slot_batch_index = _normalize_slot_batch_index(slot.get('batch_index'))
        candidate_groups = []
        if slot_type:
            candidate_groups.append((slot_type, slot_batch_index))
            if slot_batch_index is not None:
                candidate_groups.append((slot_type, None))
        for key in candidate_groups:
            candidates = artifacts_by_type_and_batch.get(key) or []
            while candidates:
                artifact = candidates.pop(0)
                artifact_ref = _artifact_ref_value(artifact)
                if artifact_ref and artifact_ref in assigned_refs:
                    continue
                if artifact_ref:
                    assigned_refs.add(artifact_ref)
                slot_artifacts.append(artifact)
                break
            if slot_artifacts:
                break
        return slot_artifacts

    for raw_slot in normalized_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot_id = str(raw_slot.get('slot_id') or '').strip()
        matching_branch = branch_lookup_by_slot.get(slot_id) if slot_id else None
        payload: dict[str, Any] = {
            'slot_id': slot_id or None,
            'branch_id': str(
                raw_slot.get('branch_id')
                or raw_slot.get('phase_id')
                or (matching_branch or {}).get('branch_id')
                or (matching_branch or {}).get('phase_id')
                or ''
            ).strip() or None,
            'phase_id': str(
                raw_slot.get('phase_id')
                or raw_slot.get('branch_id')
                or (matching_branch or {}).get('phase_id')
                or (matching_branch or {}).get('branch_id')
                or ''
            ).strip() or None,
            'type': str(raw_slot.get('type') or '').strip().lower() or None,
            'status': str(raw_slot.get('status') or '').strip().lower() or None,
            'lifecycle': str(raw_slot.get('lifecycle') or '').strip().lower() or None,
            'artifact_ref': str(raw_slot.get('artifact_ref') or (matching_branch or {}).get('artifact_ref') or '').strip() or None,
            'placeholder_ref': str(raw_slot.get('placeholder_ref') or (matching_branch or {}).get('placeholder_ref') or '').strip() or None,
            'blocked_reason': str(raw_slot.get('blocked_reason') or (matching_branch or {}).get('blocked_reason') or '').strip() or None,
            'parent_slot_id': str(raw_slot.get('parent_slot_id') or (matching_branch or {}).get('parent_slot_id') or '').strip() or None,
            'child_slot_ids': [
                str(item).strip()
                for item in (raw_slot.get('child_slot_ids') or [])
                if str(item).strip()
            ] if isinstance(raw_slot.get('child_slot_ids'), list) else [],
            'follow_up_capability': str(
                raw_slot.get('follow_up_capability')
                or (matching_branch or {}).get('follow_up_capability')
                or ''
            ).strip().lower() or None,
        }
        batch_index = _normalize_slot_batch_index(raw_slot.get('batch_index'))
        if batch_index is not None:
            payload['batch_index'] = batch_index
        if isinstance(raw_slot.get('error_ref'), Mapping):
            error_ref = {
                'branch_id': str(raw_slot['error_ref'].get('branch_id') or '').strip() or None,
                'code': str(raw_slot['error_ref'].get('code') or '').strip() or None,
                'stage': str(raw_slot['error_ref'].get('stage') or '').strip() or None,
            }
            normalized_error_ref = {
                key: value
                for key, value in error_ref.items()
                if value not in (None, '', [], {})
            }
            if normalized_error_ref:
                payload['error_ref'] = normalized_error_ref
        if isinstance(raw_slot.get('recovery_context'), Mapping):
            recovery_context = {
                'can_retry': raw_slot['recovery_context'].get('can_retry')
                if isinstance(raw_slot['recovery_context'].get('can_retry'), bool)
                else None,
                'retry_scope': str(raw_slot['recovery_context'].get('retry_scope') or '').strip() or None,
                'suggested_action': str(raw_slot['recovery_context'].get('suggested_action') or '').strip() or None,
                'preserve_intent': raw_slot['recovery_context'].get('preserve_intent')
                if isinstance(raw_slot['recovery_context'].get('preserve_intent'), bool)
                else None,
                'exclude_instance_ids': [
                    str(item).strip()
                    for item in (raw_slot['recovery_context'].get('exclude_instance_ids') or [])
                    if str(item).strip()
                ] if isinstance(raw_slot['recovery_context'].get('exclude_instance_ids'), list) else [],
            }
            normalized_recovery_context = {
                key: value
                for key, value in recovery_context.items()
                if value not in (None, '', [], {})
            }
            if normalized_recovery_context:
                payload['recovery_context'] = normalized_recovery_context
        if isinstance(raw_slot.get('recovery_state'), Mapping):
            recovery_state = {
                'kind': str(raw_slot['recovery_state'].get('kind') or '').strip() or None,
                'status': str(raw_slot['recovery_state'].get('status') or '').strip().lower() or None,
                'trigger': str(raw_slot['recovery_state'].get('trigger') or '').strip() or None,
                'branch_id': str(raw_slot['recovery_state'].get('branch_id') or '').strip() or None,
                'capability': str(raw_slot['recovery_state'].get('capability') or '').strip().lower() or None,
                'retry_scope': str(raw_slot['recovery_state'].get('retry_scope') or '').strip() or None,
                'suggested_action': str(raw_slot['recovery_state'].get('suggested_action') or '').strip() or None,
                'failed_instance_id': str(raw_slot['recovery_state'].get('failed_instance_id') or '').strip() or None,
                'promotion_required': raw_slot['recovery_state'].get('promotion_required')
                if isinstance(raw_slot['recovery_state'].get('promotion_required'), bool)
                else None,
                'auto_execute': raw_slot['recovery_state'].get('auto_execute')
                if isinstance(raw_slot['recovery_state'].get('auto_execute'), bool)
                else None,
                'preserve_intent': raw_slot['recovery_state'].get('preserve_intent')
                if isinstance(raw_slot['recovery_state'].get('preserve_intent'), bool)
                else None,
                'exclude_instance_ids': [
                    str(item).strip()
                    for item in (raw_slot['recovery_state'].get('exclude_instance_ids') or [])
                    if str(item).strip()
                ] if isinstance(raw_slot['recovery_state'].get('exclude_instance_ids'), list) else [],
            }
            normalized_recovery_state = {
                key: value
                for key, value in recovery_state.items()
                if value not in (None, '', [], {})
            }
            if normalized_recovery_state:
                payload['recovery_state'] = normalized_recovery_state
        assignment_slot = dict(matching_branch or {})
        assignment_slot.update(
            {
                key: value
                for key, value in raw_slot.items()
                if value not in (None, '', [], {})
            }
        )
        slot_artifacts = collect_slot_artifacts(assignment_slot)
        if slot_artifacts:
            canonical_artifact_ref = _artifact_ref_value(slot_artifacts[0])
            if canonical_artifact_ref:
                payload['artifact_ref'] = canonical_artifact_ref
        outputs.append(
            {
                key: value
                for key, value in payload.items()
                if value not in (None, '', [], {})
            }
        )
    return outputs


def _canonical_output_text_value(
    *,
    payload: Mapping[str, Any],
    slot: Mapping[str, Any],
    outputs: list[dict[str, Any]],
    prefer_existing_output_values: bool = False,
) -> Optional[str]:
    slot_type = str(slot.get('type') or '').strip().lower()
    if slot_type not in {'text', 'document'}:
        return None

    def clean_text(value: Any) -> str:
        return str(value or '').strip() if isinstance(value, str) or value is not None else ''

    def matching_existing_output_content() -> Optional[str]:
        existing_outputs = payload.get('outputs') if isinstance(payload.get('outputs'), list) else []
        for raw_output in reversed(existing_outputs):
            if not isinstance(raw_output, Mapping):
                continue
            output_type = clean_text(raw_output.get('type')).lower()
            if output_type and output_type not in {'text', 'document'}:
                continue
            output_status = clean_text(raw_output.get('status')).lower()
            if output_status and output_status not in {'completed', 'fulfilled'}:
                continue
            compared_identity = False
            identity_conflict = False
            for key in ('slot_id', 'phase_id', 'branch_id'):
                slot_token = clean_text(slot.get(key))
                output_token = clean_text(raw_output.get(key))
                if not slot_token or not output_token:
                    continue
                compared_identity = True
                if slot_token != output_token:
                    identity_conflict = True
                    break
            if not compared_identity or identity_conflict:
                continue
            for key in ('value', 'content_payload', 'result_text', 'output_text', 'text', 'transcript'):
                existing_content = clean_text(raw_output.get(key))
                if existing_content:
                    if _looks_like_internal_public_text(existing_content):
                        continue
                    return existing_content
        return None

    if prefer_existing_output_values:
        existing_content = matching_existing_output_content()
        if existing_content:
            return existing_content

    for key in ('value', 'content_payload', 'result_text', 'output_text', 'text', 'transcript'):
        explicit_slot_content = clean_text(slot.get(key))
        if explicit_slot_content:
            if _looks_like_internal_public_text(explicit_slot_content):
                continue
            return explicit_slot_content

    if slot_type == 'document' and clean_text(slot.get('artifact_ref')):
        return None

    slot_tokens = {
        clean_text(slot.get('branch_id')),
        clean_text(slot.get('phase_id')),
        clean_text(slot.get('slot_id')),
    }
    slot_tokens.discard('')
    late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
    fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
    for raw_result in reversed(fill_results):
        if not isinstance(raw_result, Mapping):
            continue
        result_tokens = {
            clean_text(raw_result.get('branch_id')),
            clean_text(raw_result.get('phase_id')),
            clean_text(raw_result.get('slot_id')),
        }
        result_tokens.discard('')
        if not slot_tokens.intersection(result_tokens):
            continue
        for key in ('content_payload', 'result_text', 'output_text', 'content', 'text', 'transcript'):
            branch_content = clean_text(raw_result.get(key))
            if branch_content:
                if _looks_like_internal_public_text(branch_content):
                    continue
                return branch_content

    existing_content = matching_existing_output_content()
    if existing_content:
        return existing_content

    parent_slot_id = clean_text(slot.get('parent_slot_id'))
    follow_up_capability = clean_text(slot.get('follow_up_capability'))
    branch_id = clean_text(slot.get('branch_id'))
    phase_id = clean_text(slot.get('phase_id'))
    slot_id = clean_text(slot.get('slot_id'))
    if parent_slot_id or follow_up_capability:
        return None
    if branch_id.startswith('branch-'):
        return None
    if phase_id and phase_id not in {'phase-1', 'current'} and not phase_id.endswith('-1'):
        return None
    if slot_id and slot_id not in {'output-1', 'output-phase-1'} and slot_id.startswith('output-phase-'):
        return None

    explicit_content = clean_text(payload.get('content_payload')) or clean_text(payload.get('output_text'))
    if _looks_like_internal_public_text(explicit_content):
        explicit_content = ''
    if explicit_content:
        return explicit_content
    return None


def build_canonical_outputs(
    payload: Mapping[str, Any],
    *,
    output_slots: Any = None,
    artifacts: Any = None,
    output_branches: Any = None,
    prefer_existing_output_values: bool = False,
) -> list[dict[str, Any]]:
    response_payload = payload if isinstance(payload, Mapping) else {}
    normalized_slots = _compact_output_slots_for_public_outputs(output_slots)
    public_slots = [
        slot
        for slot in normalized_slots
        if not _output_item_is_fulfilled_repair_materialization(slot)
    ]
    if public_slots:
        normalized_slots = public_slots
    normalized_artifacts = artifacts if isinstance(artifacts, list) else []
    outputs = _assign_output_artifacts_from_slots(
        normalized_slots,
        normalized_artifacts,
        output_branches=output_branches,
    )
    if outputs:
        for raw_slot, output in zip(normalized_slots, outputs):
            if not isinstance(raw_slot, Mapping):
                continue
            output.setdefault('source', 'promoted_output_slot')
            output.setdefault('compatibility_derived', False)
            value = _canonical_output_text_value(
                payload=response_payload,
                slot=raw_slot,
                outputs=outputs,
                prefer_existing_output_values=prefer_existing_output_values,
            )
            if not value:
                raw_slot_value = (
                    raw_slot.get('value')
                    or raw_slot.get('content')
                    or raw_slot.get('text')
                    or raw_slot.get('output_text')
                    or raw_slot.get('result_text')
                )
                if _artifact_backed_text_value_should_hydrate(raw_slot_value):
                    value = str(raw_slot_value or '').strip()
            if value:
                output['value'] = value
        filtered_outputs = [
            output
            for output in outputs
            if not _output_item_is_fulfilled_repair_materialization(output)
        ]
        if filtered_outputs:
            outputs = filtered_outputs
        outputs = _hydrate_artifact_backed_text_output_values(outputs, normalized_artifacts)
        outputs = _suppress_provisional_artifact_bundle_text_outputs(outputs)
        return outputs
    fallback_text = str(response_payload.get('content_payload') or response_payload.get('output_text') or '').strip()
    if _looks_like_internal_public_text(fallback_text):
        fallback_text = ''
    fallback_outputs: list[dict[str, Any]] = []
    if fallback_text:
        fallback_outputs.append(
            {
                'slot_id': 'output-1',
                'branch_id': 'phase-1',
                'phase_id': 'phase-1',
                'type': 'text',
                'status': 'fulfilled',
                'lifecycle': 'materialized_output',
                'source': 'compatibility_derived',
                'compatibility_derived': True,
                'value': fallback_text,
            }
        )
    if isinstance(normalized_artifacts, list):
        artifacts_by_ref, artifacts_by_type_and_batch = _group_artifacts_for_outputs(normalized_artifacts)
        consumed_refs = {
            str(output.get('artifact_ref') or '').strip()
            for output in fallback_outputs
            if str(output.get('artifact_ref') or '').strip()
        }
        index = len(fallback_outputs) + 1
        for _key, artifact_group in artifacts_by_type_and_batch.items():
            for artifact in artifact_group:
                artifact_ref = _artifact_ref_value(artifact)
                if artifact_ref and artifact_ref in consumed_refs:
                    continue
                if artifact_ref:
                    consumed_refs.add(artifact_ref)
                output_item = {
                    'slot_id': f'output-{index}',
                    'branch_id': f'branch-output-{index}',
                    'phase_id': f'branch-output-{index}',
                    'type': str(artifact.get('type') or '').strip().lower() or 'artifact',
                    'status': 'fulfilled',
                    'lifecycle': 'materialized_output',
                    'source': 'compatibility_derived',
                    'compatibility_derived': True,
                    'artifact_ref': artifact_ref or None,
                }
                batch_index = _normalize_slot_batch_index(artifact.get('batch_index'))
                if batch_index is not None:
                    output_item['batch_index'] = batch_index
                fallback_outputs.append(output_item)
                index += 1
        if not fallback_outputs and artifacts_by_ref:
            for artifact_ref, artifact in artifacts_by_ref.items():
                output_item = {
                    'slot_id': f'output-{index}',
                    'branch_id': f'branch-output-{index}',
                    'phase_id': f'branch-output-{index}',
                    'type': str(artifact.get('type') or '').strip().lower() or 'artifact',
                    'status': 'fulfilled',
                    'lifecycle': 'materialized_output',
                    'source': 'compatibility_derived',
                    'compatibility_derived': True,
                    'artifact_ref': artifact_ref or None,
                }
                batch_index = _normalize_slot_batch_index(artifact.get('batch_index'))
                if batch_index is not None:
                    output_item['batch_index'] = batch_index
                fallback_outputs.append(output_item)
                index += 1
    return fallback_outputs


def _compatibility_output_text_from_outputs(
    outputs: Any,
    fallback_text: str = '',
    *,
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    normalized_fallback = str(fallback_text or '').strip()
    if isinstance(payload, Mapping):
        return select_public_output_text(payload, outputs, fallback_text=normalized_fallback)
    if not isinstance(outputs, list):
        return normalized_fallback
    for item in reversed(outputs):
        if not isinstance(item, Mapping):
            continue
        output_type = str(item.get('type') or '').strip().lower()
        status = str(item.get('status') or '').strip().lower()
        value = str(item.get('value') or '').strip()
        if output_type in {'text', 'document'} and status in {'completed', 'fulfilled'} and value:
            if _artifact_ref_token(item):
                continue
            if _output_item_is_internal_materialization(item):
                continue
            if _looks_like_internal_public_text(value):
                continue
            return value
    if _looks_like_internal_public_text(normalized_fallback):
        return ''
    return normalized_fallback


def _project_compatibility_output(output_text: str, *, response_id: str, message_id: str) -> list[dict[str, Any]]:
    return [
        {
            'id': message_id,
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [
                {
                    'type': 'output_text',
                    'text': output_text,
                    'annotations': [],
                }
            ],
        }
    ]


def hoist_response_output_surfaces(payload: Mapping[str, Any]) -> dict[str, Any]:
    response_payload = dict(payload or {})
    response_frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), Mapping)
        else {}
    )
    planning = response_frame.get('planning') if isinstance(response_frame.get('planning'), Mapping) else {}
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), Mapping) else {}
    frame_output_slots = artifact_flow.get('output_slots') if isinstance(artifact_flow.get('output_slots'), list) else []
    payload_output_slots = response_payload.get('output_slots') if isinstance(response_payload.get('output_slots'), list) else []
    output_slots = payload_output_slots or frame_output_slots
    work_tree = planning.get('work_tree') if isinstance(planning.get('work_tree'), Mapping) else {}
    artifacts = response_payload.get('artifacts') if isinstance(response_payload.get('artifacts'), list) else []
    canonical_artifacts = build_canonical_response_artifacts(response_payload)
    if canonical_artifacts:
        artifacts = merge_canonical_response_artifacts(artifacts, canonical_artifacts)
        if artifacts != response_payload.get('artifacts'):
            response_payload['artifacts'] = artifacts
    if output_slots:
        response_payload['output_slots'] = output_slots
        response_payload['output_branches'] = build_public_output_branches_from_slots(output_slots)
    output_branches = response_payload.get('output_branches') if isinstance(response_payload.get('output_branches'), list) else []
    existing_outputs = response_payload.get('outputs') if isinstance(response_payload.get('outputs'), list) else []
    explicit_existing_outputs = (
        bool(existing_outputs)
        and not output_slots
        and any(
            isinstance(item, Mapping) and item.get('compatibility_derived') is False
            for item in existing_outputs
        )
    )
    outputs = (
        [dict(item) for item in existing_outputs if isinstance(item, Mapping)]
        if explicit_existing_outputs
        else build_canonical_outputs(
            response_payload,
            output_slots=output_slots,
            artifacts=artifacts,
            output_branches=output_branches,
        )
    )
    if outputs and existing_outputs:
        existing_by_token: dict[str, Mapping[str, Any]] = {}
        for item in existing_outputs:
            if not isinstance(item, Mapping):
                continue
            for key in ('slot_id', 'branch_id', 'phase_id'):
                token = str(item.get(key) or '').strip()
                if token:
                    existing_by_token[token] = item
        for output in outputs:
            if not isinstance(output, dict):
                continue
            existing = None
            for key in ('slot_id', 'branch_id', 'phase_id'):
                token = str(output.get(key) or '').strip()
                if token and token in existing_by_token:
                    existing = existing_by_token[token]
                    break
            if not existing:
                continue
            for key in ('value', 'artifact_ref', 'path', 'saved_image_path', 'saved_audio_path', 'saved_text_path'):
                if output.get(key) in (None, '', [], {}) and existing.get(key) not in (None, '', [], {}):
                    output[key] = existing.get(key)
    outputs = _hydrate_artifact_backed_text_output_values(outputs, artifacts)
    outputs = _suppress_provisional_artifact_bundle_text_outputs(outputs)
    if outputs:
        response_payload['outputs'] = outputs
    public_artifacts = filter_public_response_artifacts(
        response_payload,
        artifacts,
        outputs=outputs,
    )
    if public_artifacts != artifacts:
        response_payload['artifacts'] = public_artifacts
    if work_tree:
        response_payload['work_tree'] = work_tree
    public_output_text = select_public_output_text(
        response_payload,
        outputs,
        fallback_text=str(response_payload.get('output_text') or '').strip(),
    )
    if public_output_text:
        response_payload['output_text'] = public_output_text
        existing_output = response_payload.get('output')
        existing_item = (
            existing_output[0]
            if isinstance(existing_output, list) and existing_output and isinstance(existing_output[0], Mapping)
            else {}
        )
        response_payload['output'] = _project_compatibility_output(
            public_output_text,
            response_id=str(response_payload.get('id') or f'resp_{uuid.uuid4().hex}'),
            message_id=str(existing_item.get('id') or f'msg_{uuid.uuid4().hex}'),
        )
    return response_payload


def extract_responses_prompt(payload: dict) -> str:
    explicit_prompt = str(payload.get('prompt') or '').strip()
    if explicit_prompt:
        return explicit_prompt

    messages = extract_responses_messages(payload)
    if not messages:
        return ''
    if len(messages) == 1 and str(messages[0].get('role') or '').strip().lower() == 'user':
        return str(messages[0].get('content') or '').strip()

    parts: list[str] = []
    for item in messages:
        role = str(item.get('role') or '').strip().lower() or 'user'
        content = str(item.get('content') or '').strip()
        if not content:
            continue
        parts.append(f'[{role}]\n{content}')
    return '\n\n'.join(parts).strip()


def extract_responses_batch_prompts(payload: dict) -> list[str]:
    raw_batch_prompts = payload.get('batch_prompts')
    parsed_batch_prompts = _parse_jsonish_field(raw_batch_prompts)
    batch_prompts_payload = parsed_batch_prompts if parsed_batch_prompts is not None else raw_batch_prompts
    if not isinstance(batch_prompts_payload, list):
        return []

    prompts: list[str] = []
    for item in batch_prompts_payload:
        if isinstance(item, dict):
            prompt = extract_responses_content_text(item)
        else:
            prompt = str(item or '').strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def extract_canonical_predecessor_image_prompts(payload: Any) -> list[str]:
    """Return a prompt batch already accepted into canonical response truth."""

    if not isinstance(payload, Mapping):
        return []
    candidates: list[Any] = [payload.get('batch_prompts')]
    for key in ('phase_payload', 'late_fill'):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value.get('batch_prompts'))
    runtime = payload.get('runtime')
    if isinstance(runtime, Mapping):
        candidates.append(runtime.get('batch_prompts'))
        for key in ('phase_payload', 'semantic_phase_payload', 'late_fill'):
            value = runtime.get(key)
            if isinstance(value, Mapping):
                candidates.append(value.get('batch_prompts'))
    response_frame = payload.get('response_frame')
    if isinstance(response_frame, Mapping):
        current_state = response_frame.get('current_state')
        if isinstance(current_state, Mapping):
            late_fill = current_state.get('late_fill')
            if isinstance(late_fill, Mapping):
                candidates.append(late_fill.get('batch_prompts'))

    for candidate in candidates:
        prompts = [
            str(item).strip()
            for item in candidate
            if str(item).strip()
        ] if isinstance(candidate, list) else []
        if prompts:
            return list(dict.fromkeys(prompts))
    return []


def extract_responses_batch_items(payload: dict) -> list[dict]:
    raw_batch_prompts = payload.get('batch_prompts')
    parsed_batch_prompts = _parse_jsonish_field(raw_batch_prompts)
    batch_prompts_payload = parsed_batch_prompts if parsed_batch_prompts is not None else raw_batch_prompts
    if not isinstance(batch_prompts_payload, list):
        return []

    items: list[dict] = []
    for item in batch_prompts_payload:
        if isinstance(item, dict):
            prompt = extract_responses_content_text(item)
            width = item.get('width')
            height = item.get('height')
            aspect_ratio = str(item.get('aspect_ratio') or item.get('aspectRatio') or '').strip().lower() or None
        else:
            prompt = str(item or '').strip()
            width = None
            height = None
            aspect_ratio = None
        if not prompt:
            continue
        items.append(
            {
                'prompt': prompt,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
            }
        )
    return items


def _iter_saved_text_artifact_sources(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    response_payload = payload if isinstance(payload, Mapping) else {}

    def build_source(raw_path: Any, source: Any = None, *, response_level: bool = False) -> Optional[dict[str, Any]]:
        path = str(raw_path or '').strip()
        if not path:
            return None
        source_payload = source if isinstance(source, Mapping) else {}
        request_payload = {}
        for key in ('text_artifact_request', 'artifact_request'):
            value = source_payload.get(key)
            if isinstance(value, Mapping):
                request_payload = value
                break
        source_name = str(
            source_payload.get('source_name')
            or source_payload.get('text_artifact_source_name')
            or request_payload.get('source_name')
            or ''
        ).strip()
        return {
            'path': path,
            'source_name': source_name or None,
            'prompt': source_name or None,
            'source': source_payload,
            'response_level': response_level,
        }

    source_order: list[str] = []
    sources_by_path: dict[str, dict[str, Any]] = {}

    def has_runtime_binding(source: Mapping[str, Any]) -> bool:
        source_payload = source.get('source') if isinstance(source.get('source'), Mapping) else {}
        return any(str(source_payload.get(key) or '').strip() for key in ('branch_id', 'phase_id', 'slot_id'))

    def source_priority(source: Mapping[str, Any]) -> int:
        source_payload = source.get('source') if isinstance(source.get('source'), Mapping) else {}
        identifiers = ' '.join(
            str(source_payload.get(key) or '').strip().lower()
            for key in ('branch_id', 'phase_id', 'slot_id')
        )
        if 'branch-text_artifact-' in identifiers or 'branch-terminal-inline_text_artifact-' in identifiers:
            return 30
        if 'repair-' in identifiers:
            return 10
        if has_runtime_binding(source):
            return 20
        return 0

    def add_source(source: Optional[dict[str, Any]]) -> None:
        if not source:
            return
        path = str(source.get('path') or '').strip()
        if not path:
            return
        existing = sources_by_path.get(path)
        if existing is None:
            source_order.append(path)
            sources_by_path[path] = source
            return
        if source_priority(source) > source_priority(existing):
            upgraded = dict(existing)
            upgraded.update({key: value for key, value in source.items() if value not in (None, '', [], {})})
            sources_by_path[path] = upgraded
            return
        for key in ('source_name', 'prompt'):
            if existing.get(key) in (None, '', [], {}) and source.get(key) not in (None, '', [], {}):
                existing[key] = source.get(key)

    add_source(build_source(response_payload.get('saved_text_path'), response_payload, response_level=True))

    raw_artifacts = response_payload.get('saved_text_artifacts')
    if isinstance(raw_artifacts, list):
        for item in raw_artifacts:
            if isinstance(item, Mapping):
                source = build_source(item.get('path') or item.get('saved_text_path') or item.get('savedTextPath'), item)
            else:
                source = build_source(item)
            add_source(source)

    late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}
    fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
    for result in fill_results:
        if not isinstance(result, Mapping):
            continue
        source = build_source(
            result.get('saved_text_path') or result.get('path') or result.get('savedTextPath'),
            result,
        )
        add_source(source)
        nested_artifacts = result.get('saved_text_artifacts')
        if not isinstance(nested_artifacts, list):
            continue
        for item in nested_artifacts:
            if isinstance(item, Mapping):
                nested_source = build_source(
                    item.get('path') or item.get('saved_text_path') or item.get('savedTextPath'),
                    item,
                )
            else:
                nested_source = build_source(item)
            add_source(nested_source)

    for path in source_order:
        source = sources_by_path.get(path)
        if source:
            yield source


def _iter_late_fill_saved_artifact_sources(
    payload: Mapping[str, Any],
    *,
    path_key: str,
) -> Iterator[dict[str, Any]]:
    response_payload = payload if isinstance(payload, Mapping) else {}
    late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}
    seen_paths: set[str] = set()
    branch_lists = []
    for key in ('fill_results', 'completed_branches'):
        value = late_fill.get(key)
        if isinstance(value, list):
            branch_lists.append(value)

    for branch_list in branch_lists:
        for result in branch_list:
            if not isinstance(result, Mapping):
                continue
            path = str(result.get(path_key) or '').strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            yield {
                'path': path,
                'source': result,
            }


def _copy_artifact_runtime_binding(artifact: Optional[dict[str, Any]], source: Any) -> Optional[dict[str, Any]]:
    if not artifact or not isinstance(source, Mapping):
        return artifact
    for key in (
        'branch_id',
        'phase_id',
        'slot_id',
        'obligation_id',
        'output_type',
    ):
        value = source.get(key)
        if value not in (None, '', [], {}):
            artifact[key] = value
    return artifact


def _sanitize_artifact_record_for_outputs(raw_artifact: Any) -> Optional[dict[str, Any]]:
    artifact = sanitize_artifact_record(raw_artifact)
    if artifact:
        _copy_artifact_runtime_binding(artifact, raw_artifact)
    return artifact


def _artifact_record_merge_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    artifact_type = str(record.get('type') or record.get('kind') or '').strip().lower()
    path = str(record.get('path') or record.get('source_path') or '').strip()
    if path:
        return ('path', artifact_type, path)
    artifact_ref = str(record.get('artifact_ref') or record.get('ref') or '').strip()
    if artifact_ref:
        return ('ref', artifact_type, artifact_ref)
    artifact_id = str(record.get('artifact_id') or record.get('id') or '').strip()
    if artifact_id:
        return ('id', artifact_type, artifact_id)
    name = str(record.get('name') or record.get('source_name') or '').strip()
    return ('fallback', artifact_type, name)


def _merge_artifact_record_fields(existing: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    existing_misbound = _is_generated_image_text_misbinding(existing)
    incoming_misbound = _is_generated_image_text_misbinding(incoming)
    binding_keys = {'branch_id', 'phase_id', 'slot_id', 'obligation_id', 'task_id', 'workload_task_id'}

    def _binding_priority(record: Mapping[str, Any]) -> int:
        identifiers = ' '.join(
            str(record.get(key) or '').strip().lower()
            for key in ('branch_id', 'phase_id', 'slot_id')
        )
        if 'repair-' in identifiers:
            return 10
        if 'branch-' in identifiers:
            return 30
        return 20 if identifiers.strip() else 0

    should_upgrade_binding = _binding_priority(incoming) > _binding_priority(existing)
    for key, value in dict(incoming or {}).items():
        if value in (None, '', [], {}):
            continue
        if should_upgrade_binding and key in binding_keys:
            existing[key] = value
            continue
        if (
            existing_misbound
            and not incoming_misbound
            and key in {'artifact_id', 'artifact_ref', 'ref'}
        ):
            existing[key] = value
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


def merge_canonical_response_artifacts(existing_values: Any, canonical_values: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index_by_key: dict[tuple[str, str, str], int] = {}
    for raw_item in list(existing_values or []) + list(canonical_values or []):
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        key = _artifact_record_merge_key(item)
        if key in index_by_key:
            _merge_artifact_record_fields(merged[index_by_key[key]], item)
            continue
        index_by_key[key] = len(merged)
        merged.append(item)
    return merged


def build_canonical_response_artifacts(payload: dict) -> list[dict]:
    artifacts: list[dict] = []

    for saved_text_source in _iter_saved_text_artifact_sources(payload):
        saved_text_path = str(saved_text_source.get('path') or '').strip()
        source_payload = saved_text_source.get('source') if isinstance(saved_text_source.get('source'), Mapping) else {}
        response_level_source = bool(saved_text_source.get('response_level'))
        artifact = sanitize_artifact_record(
            {
                'type': 'text',
                'path': saved_text_path,
                'name': saved_text_source.get('source_name'),
                'prompt': saved_text_source.get('prompt'),
                'mime_type': _saved_text_mime_type(saved_text_path),
                'origin': 'assistant_output',
                'source_response_id': payload.get('id'),
                'provenance_id': None if response_level_source else source_payload.get('provenance_id'),
                'derived_from': None if response_level_source else source_payload.get('derived_from'),
            }
        )
        artifact = _copy_artifact_runtime_binding(artifact, source_payload)
        if artifact:
            artifacts.append(artifact)

    late_fill_audio_sources = list(
        _iter_late_fill_saved_artifact_sources(payload, path_key='saved_audio_path')
    )
    saved_audio_path = str(payload.get('saved_audio_path') or '').strip()
    saved_audio_has_branch_owner = bool(
        saved_audio_path
        and any(
            str(source.get('path') or '').strip() == saved_audio_path
            for source in late_fill_audio_sources
        )
    )
    saved_audio_has_ambiguous_root_provenance = bool(
        saved_audio_has_branch_owner
        or str(payload.get('saved_image_path') or '').strip()
    )
    if saved_audio_path:
        artifact = sanitize_artifact_record(
            {
                'type': 'audio',
                'path': saved_audio_path,
                'mime_type': str(payload.get('audio_mimetype') or 'audio/wav').strip() or 'audio/wav',
                'origin': 'assistant_output',
                'source_response_id': payload.get('id'),
                'provenance_id': None
                if saved_audio_has_ambiguous_root_provenance
                else payload.get('provenance_id'),
                'derived_from': None
                if saved_audio_has_ambiguous_root_provenance
                else payload.get('derived_from'),
            }
        )
        if artifact:
            artifacts.append(artifact)

    for saved_audio_source in late_fill_audio_sources:
        source_payload = saved_audio_source.get('source') if isinstance(saved_audio_source.get('source'), Mapping) else {}
        artifact = sanitize_artifact_record(
            {
                'type': 'audio',
                'path': saved_audio_source.get('path'),
                'mime_type': str(source_payload.get('audio_mimetype') or 'audio/wav').strip() or 'audio/wav',
                'origin': 'assistant_output',
                'artifact_id': source_payload.get('artifact_id'),
                'artifact_ref': source_payload.get('artifact_ref') or source_payload.get('ref'),
                'source_response_id': payload.get('id'),
                'provenance_id': source_payload.get('provenance_id'),
                'derived_from': source_payload.get('derived_from'),
            }
        )
        artifact = _copy_artifact_runtime_binding(artifact, source_payload)
        if artifact:
            artifacts.append(artifact)

    late_fill_image_sources = list(
        _iter_late_fill_saved_artifact_sources(payload, path_key='saved_image_path')
    )
    saved_image_path = str(payload.get('saved_image_path') or '').strip()
    saved_image_has_branch_owner = bool(
        saved_image_path
        and any(
            str(source.get('path') or '').strip() == saved_image_path
            for source in late_fill_image_sources
        )
    )
    saved_image_has_ambiguous_root_provenance = bool(
        saved_image_has_branch_owner
        or saved_audio_path
    )
    if saved_image_path:
        image_artifact = {
            'type': 'image',
            'path': saved_image_path,
            'origin': 'assistant_output',
            'artifact_id': payload.get('artifact_id'),
            'artifact_ref': payload.get('artifact_ref'),
            'source_response_id': payload.get('id'),
            'provenance_id': None
            if saved_image_has_ambiguous_root_provenance
            else payload.get('provenance_id'),
            'derived_from': None
            if saved_image_has_ambiguous_root_provenance
            else payload.get('derived_from'),
        }
        seed = payload.get('seed')
        if isinstance(seed, int) and seed >= 0:
            image_artifact['seed'] = seed
        if isinstance(payload.get('image_state'), dict) and payload.get('image_state'):
            image_artifact['image_state'] = payload.get('image_state')
        artifact = sanitize_artifact_record(image_artifact)
        if artifact:
            artifacts.append(artifact)

    for saved_image_source in late_fill_image_sources:
        source_payload = saved_image_source.get('source') if isinstance(saved_image_source.get('source'), Mapping) else {}
        image_artifact = {
            'type': 'image',
            'path': saved_image_source.get('path'),
            'origin': 'assistant_output',
            'artifact_id': source_payload.get('artifact_id'),
            'artifact_ref': source_payload.get('artifact_ref') or source_payload.get('ref'),
            'source_response_id': payload.get('id'),
            'provenance_id': source_payload.get('provenance_id'),
            'derived_from': source_payload.get('derived_from'),
        }
        seed = source_payload.get('seed')
        if isinstance(seed, int) and seed >= 0:
            image_artifact['seed'] = seed
        if isinstance(source_payload.get('image_state'), dict) and source_payload.get('image_state'):
            image_artifact['image_state'] = source_payload.get('image_state')
        artifact = sanitize_artifact_record(image_artifact)
        artifact = _copy_artifact_runtime_binding(artifact, source_payload)
        if artifact:
            artifacts.append(artifact)

    return artifacts


def build_canonical_batch_result(*, index: int, prompt: str, payload: dict) -> dict:
    result_payload = {
        'index': index,
        'prompt': prompt,
        'mode': str(payload.get('mode') or '').strip() or None,
        'output_text': str(payload.get('content') or payload.get('output_text') or '').strip(),
        'artifacts': build_canonical_response_artifacts(payload),
    }
    for key in (
        'saved_text_path',
        'saved_text_artifacts',
        'text_artifact_request',
        'text_artifact_requests',
        'saved_audio_path',
        'saved_image_path',
        'document_output_kind',
        'input_artifacts',
        'reference_artifacts',
        'provenance_id',
        'derived_from',
        'seed',
        'image_state',
        'reference_image_count',
        'reference_image_kind',
        'audio_mimetype',
        'tts_audio_integrity_evidence',
        'pdf_source',
        'pdf_total_pages',
        'pdf_processed_pages',
        'cached',
        'cache_id',
        'result',
    ):
        if key in payload and payload.get(key) not in (None, ''):
            result_payload[key] = payload.get(key)
    return result_payload


def flatten_canonical_batch_artifacts(results: list[dict]) -> list[dict]:
    artifacts: list[dict] = []
    for result in results:
        batch_index = int(result.get('index') or 0)
        prompt = str(result.get('prompt') or '').strip()
        for artifact in result.get('artifacts') or []:
            if not isinstance(artifact, dict):
                continue
            artifact_payload = dict(artifact)
            artifact_payload['batch_index'] = batch_index
            if prompt:
                artifact_payload['prompt'] = prompt
            artifacts.append(artifact_payload)
    return artifacts


def build_canonical_batch_outputs(artifacts: list[dict]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, Mapping):
            continue
        batch_index = _normalize_slot_batch_index(artifact.get('batch_index'))
        output_index = batch_index or index
        artifact_ref = _artifact_ref_value(artifact)
        output_item: dict[str, Any] = {
            'slot_id': f'output-batch-{output_index}',
            'branch_id': f'direct-batch-{output_index}',
            'phase_id': f'direct-batch-{output_index}',
            'type': str(artifact.get('type') or artifact.get('kind') or '').strip().lower() or 'artifact',
            'status': 'fulfilled',
            'lifecycle': 'materialized_output',
            'source': 'batch_result',
            'compatibility_derived': False,
            'artifact_ref': artifact_ref or None,
        }
        if batch_index is not None:
            output_item['batch_index'] = batch_index
        prompt = str(artifact.get('prompt') or '').strip()
        if prompt:
            output_item['prompt'] = prompt
        outputs.append(output_item)
    return outputs


def build_canonical_response_payload(
    *,
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    mode: str,
    output_text: str,
    source_payload: Optional[dict] = None,
    route_payload: Optional[dict] = None,
    response_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict:
    payload = dict(source_payload or {})
    response_payload = build_responses_output(
        output_text,
        model_name,
        response_id=response_id,
        message_id=message_id,
    )
    response_payload.update(
        {
            'instance_id': instance_id,
            'backend': backend,
            'capability': capability,
            'mode': mode,
            'warnings': payload.get('warnings') if isinstance(payload.get('warnings'), list) else [],
            'artifacts': build_canonical_response_artifacts(payload),
        }
    )

    passthrough_keys = (
        'saved_text_path',
        'saved_text_artifacts',
        'text_artifact_request',
        'text_artifact_requests',
        'saved_audio_path',
        'saved_image_path',
        'document_output_kind',
        'image_data_url',
        'input_artifacts',
        'reference_artifacts',
        'provenance_id',
        'derived_from',
        'seed',
        'image_state',
        'reference_image_count',
        'reference_image_kind',
        'audio_mimetype',
        'tts_audio_integrity_evidence',
        'pdf_source',
        'pdf_total_pages',
        'pdf_processed_pages',
        'cached',
        'cache_id',
        'result',
        'output_slots',
        'output_branches',
        'work_tree',
    )
    for key in passthrough_keys:
        if key in payload and payload.get(key) not in (None, ''):
            response_payload[key] = payload.get(key)
    semantic_phase_payload = _extract_semantic_phase_payload(payload)
    if semantic_phase_payload:
        response_payload['phase_payload'] = semantic_phase_payload
        for key, value in semantic_phase_payload.items():
            response_payload[key] = value
    if response_payload.get('saved_text_path') and 'document_output_kind' not in response_payload:
        response_payload['document_output_kind'] = 'document'
    route_info = dict(route_payload or {})
    for key in (
        'route_source',
        'route_reason',
        'route_confidence',
        'route_reuse_last_artifact',
        'route_artifact_ref',
        'route_artifact_path',
    ):
        if key in route_info and route_info.get(key) not in (None, ''):
            response_payload[key] = route_info.get(key)
    route_runtime = route_info.get('route_runtime') if isinstance(route_info.get('route_runtime'), dict) else {}
    if route_runtime:
        response_payload['runtime'] = route_runtime
        context_strategy = route_runtime.get('context_strategy') if isinstance(route_runtime.get('context_strategy'), dict) else {}
        if context_strategy.get('mode'):
            response_payload['context_mode'] = context_strategy.get('mode')
        if context_strategy.get('reason'):
            response_payload['context_reason'] = context_strategy.get('reason')
        if isinstance(route_runtime.get('route_traits'), dict) and route_runtime.get('route_traits'):
            response_payload['route_traits'] = route_runtime.get('route_traits')
    request_meta = route_info.get('request_meta') if isinstance(route_info.get('request_meta'), dict) else (
        route_runtime.get('request_meta') if isinstance(route_runtime.get('request_meta'), dict) else {}
    )
    if request_meta:
        response_payload['request_meta'] = request_meta
    response_payload = hoist_response_output_surfaces(response_payload)
    compatibility_output_text = _compatibility_output_text_from_outputs(
        response_payload.get('outputs'),
        str(response_payload.get('output_text') or output_text or '').strip(),
        payload=response_payload,
    )
    response_payload['output_text'] = compatibility_output_text
    response_payload['output'] = _project_compatibility_output(
        compatibility_output_text,
        response_id=str(response_payload.get('id') or response_id or f'resp_{uuid.uuid4().hex}'),
        message_id=str((response_payload.get('output') or [{}])[0].get('id') if isinstance(response_payload.get('output'), list) and response_payload.get('output') else message_id or f'msg_{uuid.uuid4().hex}'),
    )
    return response_payload


def build_canonical_error_response_payload(
    *,
    error_message: str,
    status_code: int,
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    mode: str,
    route_payload: Optional[dict[str, Any]] = None,
    response_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict[str, Any]:
    response_payload = build_canonical_response_payload(
        instance_id=instance_id,
        model_name=model_name,
        backend=backend,
        capability=capability,
        mode=mode,
        output_text='',
        source_payload={},
        route_payload=route_payload,
        response_id=response_id,
        message_id=message_id,
    )
    response_payload['status'] = 'failed'
    response_payload['http_status_code'] = int(status_code or 500)
    response_payload['error'] = str(error_message or 'Request failed.').strip() or 'Request failed.'
    response_payload['error_detail'] = {'message': response_payload['error']}
    return response_payload


def build_canonical_batch_response_payload(
    *,
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    batch_mode: str,
    batch_prompts: list[str],
    infer_results: list[dict],
    route_payload: Optional[dict] = None,
    response_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict:
    primary_result = infer_results[0] if infer_results else {}
    canonical_results = [
        build_canonical_batch_result(index=index, prompt=prompt, payload=result_payload)
        for index, (prompt, result_payload) in enumerate(zip(batch_prompts, infer_results), start=1)
    ]
    flattened_artifacts = flatten_canonical_batch_artifacts(canonical_results)
    total_images = len([artifact for artifact in flattened_artifacts if artifact.get('type') == 'image'])
    summary_text = (
        f'Generated {total_images} image{"s" if total_images != 1 else ""}.'
        if total_images
        else f'Processed {len(canonical_results)} batch request{"s" if len(canonical_results) != 1 else ""}.'
    )
    response_payload = build_canonical_response_payload(
        instance_id=instance_id,
        model_name=model_name,
        backend=backend,
        capability=capability,
        mode=batch_mode,
        output_text=summary_text,
        source_payload=primary_result,
        route_payload=route_payload,
        response_id=response_id,
        message_id=message_id,
    )
    response_payload['batch_prompts'] = batch_prompts
    response_payload['batch_count'] = len(canonical_results)
    response_payload['results'] = canonical_results
    response_payload['artifacts'] = flattened_artifacts
    response_payload['outputs'] = build_canonical_batch_outputs(flattened_artifacts)
    return response_payload


def translate_responses_payload_to_infer_payload(payload: dict) -> dict:
    translated: dict[str, Any] = {
        'instance_id': str(payload.get('instance_id') or '').strip(),
    }

    passthrough_keys = (
        'backend',
        'model',
        'capability',
        'request_model',
        'file_path',
        'task',
        'language',
        'voice',
        'instruct',
        'lang_code',
        'response_format',
        'speed',
        'pitch',
        'reuse_cached',
        'pdf_prefer_text',
        'infer_timeout_sec',
        'pdf_synthesize',
        'pdf_page_timeout_sec',
        'pdf_max_image_side',
        'pdf_page_retry_dpi',
        'pdf_max_pages',
        'pdf_dpi',
        'ocr_mode',
        'width',
        'height',
        'seed',
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
    )
    for key in passthrough_keys:
        value = payload.get(key)
        if value not in (None, ''):
            translated[key] = value

    prompt = extract_responses_prompt(payload)
    if prompt:
        translated['prompt'] = prompt
    return translated


def build_responses_stream_events(text: str, model_name: str) -> Iterator[str]:
    yield from build_canonical_response_stream_events(build_responses_output(text, model_name))
