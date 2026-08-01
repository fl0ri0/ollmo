"""Pure bounded response-wire projection helpers.

This module owns deterministic response-wire sizing, digest, preview, handle,
and indexed-frame projection policy.  It performs no filesystem, registry,
runtime, Flask, or model-lifecycle work.  Storage and live-source selection
remain the responsibility of their existing adapters.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


RESPONSE_WIRE_INLINE_LIMIT_BYTES = 8 * 1024 * 1024
RESPONSE_WIRE_SERIALIZATION_RESERVE_BYTES = 1024
RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES = (
    RESPONSE_WIRE_INLINE_LIMIT_BYTES - RESPONSE_WIRE_SERIALIZATION_RESERVE_BYTES
)
RESPONSE_WIRE_BATCH_PROJECTION_BUDGET_BYTES = 1024 * 1024
RESPONSE_WIRE_BATCH_DENSE_INDEX_LIMIT = 1024
RESPONSE_WIRE_MAPPING_KEY_LIMIT_BYTES = 256


@dataclass(frozen=True)
class ResponseWireProfile:
    """Named compatibility limits for one bounded projection source."""

    name: str
    inline_limit_bytes: int
    payload_limit_bytes: int
    text_preview_chars: int
    media_preview_chars: int
    collection_limit: Optional[int]
    depth_limit: int
    record_limit_bytes: int


IN_MEMORY_RESPONSE_WIRE_PROFILE = ResponseWireProfile(
    name='in_memory',
    inline_limit_bytes=RESPONSE_WIRE_INLINE_LIMIT_BYTES,
    payload_limit_bytes=RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
    text_preview_chars=32 * 1024,
    media_preview_chars=32 * 1024,
    collection_limit=64,
    depth_limit=32,
    record_limit_bytes=RESPONSE_WIRE_INLINE_LIMIT_BYTES,
)
INDEXED_RESPONSE_WIRE_PROFILE = ResponseWireProfile(
    name='indexed_response_frame',
    inline_limit_bytes=RESPONSE_WIRE_INLINE_LIMIT_BYTES,
    payload_limit_bytes=RESPONSE_WIRE_INLINE_LIMIT_BYTES,
    text_preview_chars=4096,
    media_preview_chars=256,
    collection_limit=None,
    depth_limit=32,
    record_limit_bytes=RESPONSE_WIRE_INLINE_LIMIT_BYTES,
)

IN_MEMORY_TEXT_PREVIEW_CHARS = IN_MEMORY_RESPONSE_WIRE_PROFILE.text_preview_chars
IN_MEMORY_COLLECTION_LIMIT = int(
    IN_MEMORY_RESPONSE_WIRE_PROFILE.collection_limit or 0
)
IN_MEMORY_PUBLIC_KEYS = (
    'id',
    'response_id',
    'object',
    'status',
    'lifecycle_state',
    'canonical_status_field',
    'status_compatibility',
    'status_semantics',
    'model',
    'backend',
    'capability',
    'mode',
    'instance_id',
    'route_source',
    'route_reason',
    'route_router_instance_id',
    'route_router_model',
    'route_artifact_ref',
    'route_artifact_path',
    'route_reuse_last_artifact',
    'reference_image_count',
    'reference_image_kind',
    'context_mode',
    'context_reason',
    'output_text',
    'saved_image_path',
    'saved_audio_path',
    'tts_audio_integrity_evidence',
    'saved_text_path',
    'saved_text_artifacts',
    'error',
    'error_detail',
    'error_ref',
    'recovery_hint',
    'batch_count',
    'batch_prompts',
    'results',
    'usage',
    'created_at',
    'updated_at',
    'lang_code',
    'lang_code_source',
    'response_format',
    'output_format',
    'input_artifacts',
    'reference_artifacts',
    'surface_state',
)
INDEXED_TEXT_PREVIEW_LIMIT = INDEXED_RESPONSE_WIRE_PROFILE.text_preview_chars
INDEXED_MEDIA_PREVIEW_LIMIT = INDEXED_RESPONSE_WIRE_PROFILE.media_preview_chars
INDEXED_DEPTH_LIMIT = INDEXED_RESPONSE_WIRE_PROFILE.depth_limit
INDEXED_RECORD_LIMIT_BYTES = INDEXED_RESPONSE_WIRE_PROFILE.record_limit_bytes

_RAW_MEDIA_PAYLOAD_KEYS = {
    'audio',
    'audio_base64',
    'audio_b64',
    'audio_data',
    'image',
    'image_base64',
    'image_b64',
    'image_data',
    'png_base64',
    'video',
    'video_base64',
    'video_b64',
    'video_data',
}


def json_size_up_to(value: Any, *, limit: int) -> int:
    """Estimate conservative Flask-compatible JSON size up to ``limit``."""

    total = 0
    try:
        encoder = json.JSONEncoder(
            ensure_ascii=True,
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        )
        for chunk in encoder.iterencode(value):
            total += len(chunk.encode('utf-8'))
            if total > limit:
                break
    except (TypeError, ValueError, RecursionError):
        return limit + 1
    return total


def digest_ref(value: Any, *, json_path: str) -> dict[str, Any]:
    """Bind an omitted in-memory body without claiming replay storage."""

    digest = hashlib.sha256()
    size_bytes = 0
    try:
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        )
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode('utf-8')
            digest.update(encoded)
            size_bytes += len(encoded)
    except (TypeError, ValueError, RecursionError):
        return {
            'kind': 'ollmo.response_wire_digest_ref',
            'json_path': json_path,
            'storage': 'unavailable',
            'authority': 'audit_identity_unavailable',
        }
    return {
        'kind': 'ollmo.response_wire_digest_ref',
        'json_path': json_path,
        'sha256': digest.hexdigest(),
        'size_bytes': size_bytes,
        'content_addressed': True,
        'storage': 'digest_only',
        'authority': 'audit_identity_only',
        'normalization': 'json_sort_keys_compact_utf8',
    }


def text_preview(
    value: Any,
    *,
    json_path: str,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> tuple[str, Optional[dict[str, Any]]]:
    text_value = str(value or '')
    if len(text_value) <= profile.text_preview_chars:
        return text_value, None
    return (
        text_value[:profile.text_preview_chars],
        {
            **digest_ref(text_value, json_path=json_path),
            'length_chars': len(text_value),
            'preview_chars': profile.text_preview_chars,
            'preview_truncated': True,
        },
    )


def error_handle(
    value: Any,
    *,
    json_path: str,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> tuple[Any, Optional[dict[str, Any]]]:
    """Keep actionable error identity while bounding exceptional bodies."""

    if isinstance(value, str):
        return text_preview(value, json_path=json_path, profile=profile)
    if not isinstance(value, Mapping):
        if value in (None, '', [], {}):
            return None, None
        return text_preview(str(value), json_path=json_path, profile=profile)
    if json_size_up_to(value, limit=64 * 1024) <= 64 * 1024:
        return copy.deepcopy(dict(value)), None
    handle: dict[str, Any] = {}
    for key in ('code', 'status', 'type', 'kind', 'name'):
        item = value.get(key)
        if isinstance(item, str) and item:
            handle[key] = item[:512]
        elif isinstance(item, (bool, int, float)):
            handle[key] = item
    for key in ('message', 'error', 'reason', 'detail'):
        item = value.get(key)
        if not isinstance(item, str) or not item:
            continue
        preview, preview_ref = text_preview(
            item,
            json_path=f'{json_path}.{key}',
            profile=profile,
        )
        handle[key] = preview
        if preview_ref:
            handle[f'{key}_ref'] = preview_ref
    if not handle:
        handle['message'] = 'Error details are available only in canonical response truth.'
    body_ref = digest_ref(value, json_path=json_path)
    body_ref['projection_truncated'] = True
    return handle, body_ref


def artifact_handle(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    handle: dict[str, Any] = {}
    for key in (
        'artifact_id',
        'artifact_ref',
        'batch_index',
        'ref',
        'type',
        'kind',
        'mime_type',
        'name',
        'path',
        'origin',
        'content_source',
        'content_length_chars',
        'content_sha256',
        'file_sha256',
        'file_size_bytes',
        'source_response_id',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'text_artifact_extension',
        'text_artifact_source_name',
        'syntax_sanity_status',
        'syntax_sanity_issue_count',
    ):
        item = value.get(key)
        if item in (None, '', [], {}):
            continue
        if isinstance(item, str):
            handle[key] = item[:4096]
        elif isinstance(item, (bool, int, float)):
            handle[key] = item
    return handle


def output_handle(
    value: Any,
    *,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    handle: dict[str, Any] = {}
    for key in (
        'id',
        'batch_index',
        'slot_id',
        'branch_id',
        'phase_id',
        'type',
        'status',
        'lifecycle',
        'artifact_ref',
        'path',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'placeholder_ref',
        'blocked_reason',
        'error_ref',
        'capability',
        'output_type',
    ):
        item = value.get(key)
        if item in (None, '', [], {}):
            continue
        if isinstance(item, str):
            handle[key] = item[:4096]
        elif isinstance(item, (bool, int, float)):
            handle[key] = item
    text_value = value.get('value')
    if text_value not in (None, ''):
        preview, preview_ref = text_preview(
            text_value,
            json_path=f"outputs.{str(value.get('slot_id') or value.get('id') or 'item')}.value",
            profile=profile,
        )
        handle['value'] = preview
        if preview_ref:
            handle['value_ref'] = preview_ref
    return handle


def compact_lookup_branch(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        'branch_id',
        'slot_id',
        'phase_id',
        'capability',
        'expected_capability',
        'output_type',
        'artifact_type',
        'status',
        'lifecycle',
        'repair_action',
        'recovery_action',
        'progress_stage',
        'instance_id',
    ):
        value = item.get(key)
        if value not in (None, '', [], {}):
            compact[key] = value
    depends_on = item.get('depends_on')
    if isinstance(depends_on, list) and depends_on:
        compact['depends_on_count'] = len(depends_on)
    return compact


def branch_handle(
    value: Any,
    *,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    compact = compact_lookup_branch(value)
    handle = {
        key: (item[:512] if isinstance(item, str) else item)
        for key, item in compact.items()
        if isinstance(item, (str, bool, int, float)) and item not in (None, '')
    }
    for key in (
        'blocked_reason',
        'cancel_reason',
        'waiver_reason',
        'supersession_reason',
        'error_ref',
        'artifact_ref',
        'path',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
    ):
        item = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(item, str) and item:
            handle[key] = item[:4096]
    for error_key in ('error', 'error_detail'):
        error_value = value.get(error_key) if isinstance(value, Mapping) else None
        if error_value in (None, '', [], {}):
            continue
        projected_error, error_ref = error_handle(
            error_value,
            json_path=f'late_fill.branch.{error_key}',
            profile=profile,
        )
        if projected_error not in (None, '', [], {}):
            handle[error_key] = projected_error
        if error_ref:
            handle[f'{error_key}_ref'] = error_ref
    for nested_key in ('attempt', 'recovery_context', 'recovery_state'):
        nested_value = value.get(nested_key) if isinstance(value, Mapping) else None
        if not isinstance(nested_value, Mapping) or not nested_value:
            continue
        if json_size_up_to(nested_value, limit=32 * 1024) <= 32 * 1024:
            handle[nested_key] = copy.deepcopy(dict(nested_value))
        else:
            handle[f'{nested_key}_ref'] = digest_ref(
                nested_value,
                json_path=f'late_fill.branch.{nested_key}',
            )
    execution_gate = value.get('execution_gate') if isinstance(value, Mapping) else None
    if isinstance(execution_gate, Mapping):
        gate_handle = {
            key: (item[:512] if isinstance(item, str) else item)
            for key in (
                'kind',
                'scope',
                'status',
                'action',
                'branch_id',
                'phase_id',
                'capability',
                'authority',
                'source',
                'reason',
            )
            if isinstance((item := execution_gate.get(key)), (str, bool, int, float))
            and item not in (None, '')
        }
        if gate_handle:
            handle['execution_gate'] = gate_handle
    return handle


def surface_handle(
    value: Any,
    *,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    collection_limit = int(profile.collection_limit or IN_MEMORY_COLLECTION_LIMIT)
    handle: dict[str, Any] = {}
    for key in ('status', 'summary', 'message'):
        item = value.get(key)
        if item not in (None, '', [], {}):
            handle[key] = item[:4096] if isinstance(item, str) else item
    category_counts = value.get('category_counts')
    if isinstance(category_counts, Mapping):
        handle['category_counts'] = {
            str(key)[:256]: (item[:512] if isinstance(item, str) else item)
            for key, item in list(category_counts.items())[:collection_limit]
            if isinstance(item, (str, bool, int, float))
        }
    active_categories = value.get('active_categories')
    if isinstance(active_categories, list):
        handle['active_categories'] = [
            str(item)[:256]
            for item in active_categories[:collection_limit]
        ]
    return {
        key: item
        for key, item in handle.items()
        if item not in (None, '', [], {})
    }


def frame_cas_handle(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    handle: dict[str, Any] = {}
    for key in ('frame_id', 'kind', 'response_id', 'status', 'created_at', 'updated_at'):
        item = value.get(key)
        if isinstance(item, str) and item:
            handle[key] = item[:512]
    for key in ('frame_sequence', 'frame_version'):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            handle[key] = item
    relation = value.get('frame_relation')
    if isinstance(relation, Mapping):
        handle['frame_relation'] = {
            key: item[:512]
            for key in ('kind', 'parent_frame_id', 'proposal_id', 'rebase_id')
            if isinstance((item := relation.get(key)), str) and item
        }
        parent_sequence = relation.get('parent_frame_sequence')
        if (
            isinstance(parent_sequence, int)
            and not isinstance(parent_sequence, bool)
            and parent_sequence >= 0
        ):
            handle['frame_relation']['parent_frame_sequence'] = parent_sequence
    external_snapshots = value.get('external_snapshots')
    if isinstance(external_snapshots, Mapping):
        items = external_snapshots.get('items')
        raw_count = external_snapshots.get('effective_snapshot_count')
        snapshot_count = (
            raw_count
            if isinstance(raw_count, int)
            and not isinstance(raw_count, bool)
            and raw_count >= 0
            else len(items)
            if isinstance(items, Mapping)
            else 0
        )
        handle['external_snapshot_manifest'] = {
            'effective_snapshot_count': snapshot_count,
            'manifest_ref': digest_ref(
                items if isinstance(items, Mapping) else external_snapshots,
                json_path='response_frame.external_snapshots.items',
            ),
        }
    return handle


def late_fill_handle(
    payload: Mapping[str, Any],
    *,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    late_fill = payload.get('late_fill') or payload.get('lateFill') or runtime.get('late_fill')
    if not isinstance(late_fill, Mapping):
        return {}
    collection_limit = int(profile.collection_limit or IN_MEMORY_COLLECTION_LIMIT)
    handle: dict[str, Any] = {}
    for key in (
        'status',
        'lifecycle_state',
        'code',
        'trigger',
        'expected_capability',
        'missing_artifact_type',
        'partial_failure',
        'pending_branch_count',
        'active_branch_count',
        'completed_branch_count',
        'failed_branch_count',
        'cancelled_branch_count',
        'final_materialization_contract_status',
        'materialization_contract_unmet',
    ):
        item = late_fill.get(key)
        if item in (None, '', [], {}):
            continue
        handle[key] = item[:512] if isinstance(item, str) else item
    for key in (
        'pending_capabilities',
        'active_capabilities',
        'completed_capabilities',
        'failed_capabilities',
        'cancelled_capabilities',
    ):
        values = late_fill.get(key)
        if isinstance(values, list):
            handle[key] = [str(item)[:256] for item in values[:collection_limit]]
    public_branch_keys = (
        'pending_branches',
        'active_branches',
        'completed_branches',
        'failed_branches',
        'cancelled_branches',
    )
    for key in public_branch_keys:
        values = late_fill.get(key)
        if isinstance(values, list):
            handle[key] = [
                branch_handle(item, profile=profile)
                for item in values[:collection_limit]
                if isinstance(item, Mapping)
            ]
    for key in public_branch_keys:
        handle.setdefault(key, [])
    return {
        key: value
        for key, value in handle.items()
        if value not in (None, '', [], {}) or key in public_branch_keys
    }


def snapshot_ref_handle(
    value: Any,
    *,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    collection_limit = int(profile.collection_limit or IN_MEMORY_COLLECTION_LIMIT)
    handle: dict[str, Any] = {}
    for key in (
        'kind',
        'json_path',
        'path',
        'sha256',
        'size_bytes',
        'content_addressed',
        'dedupe_scope',
        'source_frame_id',
        'source_response_id',
    ):
        item = value.get(key)
        if key == 'size_bytes':
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                handle[key] = item
        elif key == 'content_addressed':
            if isinstance(item, bool):
                handle[key] = item
        elif isinstance(item, str) and item:
            handle[key] = item[:512]
    normalization = value.get('content_normalization')
    if isinstance(normalization, Mapping):
        compact_normalization: dict[str, Any] = {}
        for key in ('kind', 'strategy'):
            item = normalization.get(key)
            if isinstance(item, str) and item:
                compact_normalization[key] = item[:512]
        count = normalization.get('volatile_timestamp_field_count')
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            compact_normalization['volatile_timestamp_field_count'] = count
        if compact_normalization:
            handle['content_normalization'] = compact_normalization
    sidecar_manifest = value.get('sidecar_manifest')
    if isinstance(sidecar_manifest, Mapping):
        compact_manifest: dict[str, Any] = {}
        for key in ('kind', 'strategy'):
            item = sidecar_manifest.get(key)
            if isinstance(item, str) and item:
                compact_manifest[key] = item[:512]
        for key in ('child_ref_count', 'max_depth', 'split_limit_bytes'):
            item = sidecar_manifest.get(key)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                compact_manifest[key] = item
        if isinstance(sidecar_manifest.get('child_refs_truncated'), bool):
            compact_manifest['child_refs_truncated'] = sidecar_manifest.get(
                'child_refs_truncated'
            )
        child_refs = sidecar_manifest.get('child_refs')
        if isinstance(child_refs, list):
            compact_manifest['child_refs'] = [
                {
                    key: (
                        child_value[:512]
                        if isinstance(child_value, str)
                        else child_value
                    )
                    for key in ('key', 'json_path', 'sha256', 'size_bytes')
                    if (
                        (child_value := item.get(key)) not in (None, '', [], {})
                        and (
                            isinstance(child_value, str)
                            or isinstance(child_value, int)
                            and not isinstance(child_value, bool)
                            and child_value >= 0
                        )
                    )
                }
                for item in child_refs[:collection_limit]
                if isinstance(item, Mapping)
            ]
            if len(child_refs) > collection_limit:
                compact_manifest['child_refs_truncated'] = True
        handle['sidecar_manifest'] = compact_manifest
    return handle


def frame_from_memory(
    value: Any,
    *,
    omitted: dict[str, Any],
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    frame = value
    projected: dict[str, Any] = {}
    for key in (
        'created_at',
        'frame_id',
        'frame_sequence',
        'frame_version',
        'kind',
        'response_id',
        'status',
        'updated_at',
    ):
        item = frame.get(key)
        if key in {'frame_sequence', 'frame_version'}:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                projected[key] = item
        elif isinstance(item, str) and item:
            projected[key] = item[:4096]
    frame_relation = frame.get('frame_relation')
    if isinstance(frame_relation, Mapping):
        projected['frame_relation'] = {
            key: item[:4096]
            for key in (
                'kind',
                'reason',
                'response_id',
                'parent_response_id',
                'parent_frame_id',
                'proposal_id',
                'rebase_id',
                'operator_record_id',
            )
            if isinstance((item := frame_relation.get(key)), str) and item
        }
        parent_sequence = frame_relation.get('parent_frame_sequence')
        if (
            isinstance(parent_sequence, int)
            and not isinstance(parent_sequence, bool)
            and parent_sequence >= 0
        ):
            projected['frame_relation']['parent_frame_sequence'] = parent_sequence
    current_state = (
        frame.get('current_state')
        if isinstance(frame.get('current_state'), Mapping)
        else {}
    )
    projected_current: dict[str, Any] = {}
    for key in (
        'canonical_status_field',
        'id',
        'lifecycle_state',
        'status',
        'status_compatibility',
        'status_semantics',
        'updated_at',
    ):
        item = current_state.get(key)
        if item in (None, '', [], {}):
            continue
        if isinstance(item, str):
            projected_current[key] = item[:4096]
        elif json_size_up_to(item, limit=64 * 1024) <= 64 * 1024:
            projected_current[key] = copy.deepcopy(item)
        else:
            omitted[f'response_frame.current_state.{key}'] = digest_ref(
                item,
                json_path=f'response_frame.current_state.{key}',
            )
    for key, item in current_state.items():
        if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
            projected_current[key] = snapshot_ref_handle(item, profile=profile)
    if projected_current:
        projected['current_state'] = projected_current
    for key, item in frame.items():
        if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
            projected[key] = snapshot_ref_handle(item, profile=profile)
    external_snapshots = frame.get('external_snapshots')
    if isinstance(external_snapshots, Mapping):
        projected_external: dict[str, Any] = {}
        for key in ('kind', 'storage'):
            item = external_snapshots.get(key)
            if isinstance(item, str) and item:
                projected_external[key] = item[:512]
        for key in ('version', 'effective_snapshot_count'):
            item = external_snapshots.get(key)
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                projected_external[key] = item
        if isinstance(external_snapshots.get('effective_manifest_expanded'), bool):
            projected_external['effective_manifest_expanded'] = external_snapshots.get(
                'effective_manifest_expanded'
            )
        for manifest_key in ('items', 'delta_items'):
            manifest = external_snapshots.get(manifest_key)
            if not isinstance(manifest, Mapping):
                continue
            projected_external[manifest_key] = {
                str(path)[:512]: snapshot_ref_handle(ref, profile=profile)
                for path, ref in list(manifest.items())[:256]
                if isinstance(ref, Mapping)
            }
            if len(manifest) > 256:
                projected_external[f'{manifest_key}_truncated'] = True
                projected_external[f'{manifest_key}_count'] = len(manifest)
                omitted[f'response_frame.external_snapshots.{manifest_key}'] = digest_ref(
                    manifest,
                    json_path=f'response_frame.external_snapshots.{manifest_key}',
                )
        projected['external_snapshots'] = projected_external
    snapshot_policy = frame.get('snapshot_policy')
    if isinstance(snapshot_policy, Mapping):
        if json_size_up_to(snapshot_policy, limit=64 * 1024) <= 64 * 1024:
            projected['snapshot_policy'] = copy.deepcopy(dict(snapshot_policy))
        else:
            omitted['response_frame.snapshot_policy'] = digest_ref(
                snapshot_policy,
                json_path='response_frame.snapshot_policy',
            )
    omitted['response_frame'] = digest_ref(frame, json_path='response_frame')
    return projected


def outer_value_handle(
    value: Any,
    *,
    json_path: str,
    profile: ResponseWireProfile = IN_MEMORY_RESPONSE_WIRE_PROFILE,
) -> Any:
    """Bound a non-response wrapper field while retaining typed status truth."""

    if isinstance(value, str):
        preview, preview_ref = text_preview(
            value,
            json_path=json_path,
            profile=profile,
        )
        if preview_ref:
            return {'preview': preview, 'value_ref': preview_ref}
        return preview
    if isinstance(value, list):
        return {
            'item_count': len(value),
            'items_ref': digest_ref(value, json_path=json_path),
            'projection_truncated': True,
        }
    if not isinstance(value, Mapping):
        return value if isinstance(value, (bool, int, float)) else str(value)[:512]

    handle: dict[str, Any] = {}
    for key in (
        'type',
        'object',
        'status',
        'lifecycle_state',
        'code',
        'action',
        'scope',
        'reason',
        'branch_id',
        'phase_id',
        'response_id',
        'late_fill_status',
        'pending_count',
        'active_count',
        'completed_count',
        'failed_count',
    ):
        item = value.get(key)
        if item in (None, '', [], {}):
            continue
        if isinstance(item, str):
            handle[key] = item[:512]
        elif isinstance(item, (bool, int, float)):
            handle[key] = item
    handle['payload_ref'] = digest_ref(value, json_path=json_path)
    handle['projection_truncated'] = True
    return handle


def emergency_projection(
    response_payload: Mapping[str, Any],
    *,
    source: str,
    output_message_projector: Callable[[Any], dict[str, Any]],
    limit_bytes: int = RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
) -> dict[str, Any]:
    """Return a hard-bounded public envelope for exceptional in-memory truth."""

    effective_limit = max(2048, min(int(limit_bytes), RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES))

    projected: dict[str, Any] = {}
    compact_scalar_keys = (
        'id',
        'response_id',
        'object',
        'status',
        'lifecycle_state',
        'canonical_status_field',
        'state_version',
        'model',
        'backend',
        'capability',
        'mode',
        'instance_id',
        'route_source',
        'route_artifact_ref',
        'route_artifact_path',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'created_at',
        'updated_at',
        'ui_compact',
    )
    for key in compact_scalar_keys:
        value = response_payload.get(key)
        if value in (None, '', [], {}):
            continue
        if isinstance(value, str):
            preview, _preview_ref = text_preview(value, json_path=key)
            projected[key] = preview
        elif isinstance(value, (bool, int, float)):
            projected[key] = value

    omitted: dict[str, Any] = {}
    output_text = response_payload.get('output_text')
    if output_text not in (None, ''):
        preview, preview_ref = text_preview(
            output_text,
            json_path='output_text',
        )
        projected['output_text'] = preview
        if preview_ref:
            projected['output_text_ref'] = preview_ref
    for error_key in ('error', 'error_detail'):
        error_value = response_payload.get(error_key)
        if error_value in (None, '', [], {}):
            continue
        projected_error, error_ref = error_handle(
            error_value,
            json_path=error_key,
        )
        if projected_error not in (None, '', [], {}):
            projected[error_key] = projected_error
        if error_ref:
            projected[f'{error_key}_ref'] = error_ref
    status_lookup = response_payload.get('status_lookup')
    if (
        isinstance(status_lookup, Mapping)
        and json_size_up_to(status_lookup, limit=128 * 1024) <= 128 * 1024
    ):
        projected['status_lookup'] = copy.deepcopy(dict(status_lookup))

    artifacts = response_payload.get('artifacts')
    if isinstance(artifacts, list):
        projected['artifacts'] = [
            artifact_handle(item)
            for item in artifacts[:IN_MEMORY_COLLECTION_LIMIT]
            if isinstance(item, Mapping)
        ]
        if len(artifacts) > IN_MEMORY_COLLECTION_LIMIT:
            omitted['artifacts_tail'] = {
                'item_count': len(artifacts),
                'inline_item_count': IN_MEMORY_COLLECTION_LIMIT,
            }
    for key in ('saved_text_artifacts', 'input_artifacts', 'reference_artifacts'):
        values = response_payload.get(key)
        if not isinstance(values, list):
            continue
        projected[key] = [
            artifact_handle(item)
            for item in values[:IN_MEMORY_COLLECTION_LIMIT]
            if isinstance(item, Mapping)
        ]
        if len(values) > IN_MEMORY_COLLECTION_LIMIT:
            omitted[f'{key}_tail'] = {
                'item_count': len(values),
                'inline_item_count': IN_MEMORY_COLLECTION_LIMIT,
            }
    for key in ('outputs', 'output_slots', 'output_branches'):
        values = response_payload.get(key)
        if not isinstance(values, list):
            continue
        projected[key] = [
            output_handle(item)
            for item in values[:IN_MEMORY_COLLECTION_LIMIT]
            if isinstance(item, Mapping)
        ]
        if len(values) > IN_MEMORY_COLLECTION_LIMIT:
            omitted[f'{key}_tail'] = {
                'item_count': len(values),
                'inline_item_count': IN_MEMORY_COLLECTION_LIMIT,
            }
    output_messages = response_payload.get('output')
    if isinstance(output_messages, list):
        projected['output'] = [
            output_message_projector(item)
            for item in output_messages[:IN_MEMORY_COLLECTION_LIMIT]
            if isinstance(item, Mapping)
        ]
    results = response_payload.get('results')
    if isinstance(results, list):
        projected['results'] = [
            {
                key: (
                    value[:IN_MEMORY_TEXT_PREVIEW_CHARS]
                    if isinstance(value, str)
                    else value
                )
                for key in (
                    'index',
                    'status',
                    'prompt',
                    'content',
                    'saved_image_path',
                    'saved_audio_path',
                    'saved_text_path',
                    'error',
                )
                if isinstance((value := item.get(key)), (str, bool, int, float))
                and value not in (None, '')
            }
            for item in results[:IN_MEMORY_COLLECTION_LIMIT]
            if isinstance(item, Mapping)
        ]
        if len(results) > IN_MEMORY_COLLECTION_LIMIT:
            omitted['results_tail'] = {
                'item_count': len(results),
                'inline_item_count': IN_MEMORY_COLLECTION_LIMIT,
            }
    late_fill = late_fill_handle(response_payload)
    if late_fill:
        projected['late_fill'] = late_fill
    frame = frame_from_memory(
        response_payload.get('response_frame'),
        omitted=omitted,
    )
    if frame:
        projected['response_frame'] = frame
    for key in ('runtime', 'working_frame', 'request'):
        value = response_payload.get(key)
        if value not in (None, '', [], {}):
            omitted[key] = digest_ref(value, json_path=key)
    omitted['response_payload'] = digest_ref(
        response_payload,
        json_path='response_payload',
    )
    projected['wire_projection'] = {
        'kind': 'ollmo.response_wire_projection',
        'version': 1,
        'runtime_effect': 'none',
        'source': source,
        'bounded': True,
        'inline_limit_bytes': RESPONSE_WIRE_INLINE_LIMIT_BYTES,
        'sidecar_hydration': 'none',
        'omitted_payloads': omitted,
        'truth_preservation': 'digest_identity_only_until_durable_frame_is_available',
    }
    projected = {key: value for key, value in projected.items() if value not in (None, '', [], {})}
    if json_size_up_to(
        projected,
        limit=effective_limit,
    ) <= effective_limit:
        return projected
    # Last-resort envelope: exact digest identity plus lifecycle and frame CAS.
    minimal = {
        key: value
        for key, value in projected.items()
        if key in {
            'id',
            'response_id',
            'object',
            'status',
            'lifecycle_state',
            'model',
            'backend',
            'capability',
            'mode',
            'instance_id',
            'state_version',
            'status_lookup',
            'ui_compact',
            'error',
            'error_ref',
            'error_detail',
            'error_detail_ref',
            'wire_projection',
        }
    }
    frame_cas = frame_cas_handle(response_payload.get('response_frame'))
    if frame_cas:
        minimal['response_frame'] = frame_cas
    minimal['wire_projection'] = dict(projected['wire_projection'])
    minimal['wire_projection']['emergency_minimal_envelope'] = True
    if json_size_up_to(
        minimal,
        limit=effective_limit,
    ) <= effective_limit:
        return minimal
    absolute_minimal = {
        key: (value[:512] if isinstance(value, str) else value)
        for key, value in response_payload.items()
        if key in {'id', 'response_id', 'object', 'status', 'lifecycle_state'}
        and isinstance(value, (str, bool, int, float))
    }
    frame_cas = frame_cas_handle(response_payload.get('response_frame'))
    if frame_cas:
        absolute_minimal['response_frame'] = frame_cas
    absolute_minimal['wire_projection'] = {
        'kind': 'ollmo.response_wire_projection',
        'version': 1,
        'runtime_effect': 'none',
        'source': source,
        'bounded': True,
        'inline_limit_bytes': RESPONSE_WIRE_INLINE_LIMIT_BYTES,
        'emergency_absolute_minimal_envelope': True,
        'response_payload_ref': digest_ref(
            response_payload,
            json_path='response_payload',
        ),
    }
    return absolute_minimal


def fallback_payload(
    response_payload: Mapping[str, Any],
    *,
    artifact_projector: Callable[[Any], dict[str, Any]],
    output_projector: Callable[[Any], dict[str, Any]],
    output_message_projector: Callable[[Any], dict[str, Any]],
    late_fill_projector: Callable[[Mapping[str, Any]], dict[str, Any]],
    emergency_projector: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Bound a response even when its durable compact row is unavailable."""

    if json_size_up_to(
        response_payload,
        limit=RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
    ) <= RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES:
        inline = copy.deepcopy(dict(response_payload))
        inline['wire_projection'] = {
            'kind': 'ollmo.response_wire_projection',
            'version': 1,
            'runtime_effect': 'none',
            'source': 'in_memory_inline_below_limit',
            'bounded': True,
            'inline_limit_bytes': RESPONSE_WIRE_INLINE_LIMIT_BYTES,
            'sidecar_hydration': 'none',
        }
        return inline

    projected: dict[str, Any] = {
        key: copy.deepcopy(response_payload.get(key))
        for key in IN_MEMORY_PUBLIC_KEYS
        if response_payload.get(key) not in (None, '', [], {})
    }
    artifacts = response_payload.get('artifacts')
    if isinstance(artifacts, list):
        projected['artifacts'] = [
            artifact_projector(item)
            for item in artifacts
            if isinstance(item, Mapping)
        ]
    for key in ('outputs', 'output_slots', 'output_branches'):
        values = response_payload.get(key)
        if isinstance(values, list):
            projected[key] = [
                output_projector(item)
                for item in values
                if isinstance(item, Mapping)
            ]
    output_messages = response_payload.get('output')
    if isinstance(output_messages, list):
        projected['output'] = [
            output_message_projector(item)
            for item in output_messages
            if isinstance(item, Mapping)
        ]
    late_fill = late_fill_projector(response_payload)
    if late_fill:
        projected['late_fill'] = late_fill

    omitted: dict[str, Any] = {}
    frame = frame_from_memory(
        response_payload.get('response_frame'),
        omitted=omitted,
    )
    if frame:
        projected['response_frame'] = frame
    for key in ('runtime', 'working_frame'):
        value = response_payload.get(key)
        if value not in (None, '', [], {}):
            omitted[key] = digest_ref(value, json_path=key)
    projected['wire_projection'] = {
        'kind': 'ollmo.response_wire_projection',
        'version': 1,
        'runtime_effect': 'none',
        'source': 'in_memory_digest_fallback',
        'bounded': True,
        'inline_limit_bytes': RESPONSE_WIRE_INLINE_LIMIT_BYTES,
        'sidecar_hydration': 'none',
        'omitted_payloads': omitted,
        'truth_preservation': 'digest_identity_only_until_durable_frame_is_available',
    }
    projected = {key: value for key, value in projected.items() if value not in (None, '', [], {})}
    if json_size_up_to(
        projected,
        limit=RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
    ) > RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES:
        return emergency_projector(
            response_payload,
            source='in_memory_emergency_bounded_fallback',
        )
    return projected


def enforce_byte_ceiling(
    projected: Mapping[str, Any],
    *,
    source_payload: Optional[Mapping[str, Any]] = None,
    source: str,
    emergency_projector: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if json_size_up_to(
        projected,
        limit=RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
    ) <= RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES:
        return copy.deepcopy(dict(projected))
    bounded = emergency_projector(
        source_payload if isinstance(source_payload, Mapping) else projected,
        source=source,
    )
    prior_projection = (
        projected.get('wire_projection')
        if isinstance(projected.get('wire_projection'), Mapping)
        else None
    )
    if prior_projection:
        projection = dict(bounded.get('wire_projection') or {})
        projection['pre_ceiling_projection_ref'] = digest_ref(
            prior_projection,
            json_path='wire_projection.pre_ceiling',
        )
        bounded['wire_projection'] = projection
        if json_size_up_to(
            bounded,
            limit=RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
        ) > RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES:
            projection.pop('pre_ceiling_projection_ref', None)
    return bounded


def _is_empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _indexed_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or '').strip()
            if not key or key in {'response_frame', 'image_data_url', 'imageDataUrl'}:
                continue
            if _is_empty(raw_value):
                continue
            payload[key] = _indexed_json_safe(raw_value)
        return payload
    if isinstance(value, (list, tuple)):
        return [_indexed_json_safe(item) for item in value if not _is_empty(item)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def bounded_projection_mapping_key(
    raw_key: Any,
    *,
    used_keys: set[str],
    limit_bytes: int = RESPONSE_WIRE_MAPPING_KEY_LIMIT_BYTES,
) -> Optional[str]:
    """Return a deterministic, collision-resistant key for bounded previews."""

    key = str(raw_key or '').strip()
    if not key:
        return None
    encoded = key.encode('utf-8')
    identity = (
        f'{type(raw_key).__module__}.{type(raw_key).__qualname__}:'.encode('utf-8')
        + encoded
    )
    digest = hashlib.sha256(identity).hexdigest()

    def key_with_suffix(suffix: str) -> str:
        suffix_bytes = suffix.encode('ascii')
        prefix_budget = max(0, limit_bytes - len(suffix_bytes))
        prefix = encoded[:prefix_budget].decode('utf-8', errors='ignore')
        return f'{prefix}{suffix}'

    candidate = key
    if len(encoded) > limit_bytes:
        candidate = key_with_suffix(f'#sha256:{digest}')
    if candidate in used_keys:
        candidate = key_with_suffix(f'#key-sha256:{digest}')
    if candidate in used_keys:
        collision = 1
        while True:
            suffix = f'#key-sha256:{digest}:{collision}'
            candidate = key_with_suffix(suffix)
            if candidate not in used_keys:
                break
            collision += 1
    used_keys.add(candidate)
    return candidate


def indexed_stat(
    stats: Optional[dict[str, int]],
    key: str,
    amount: int = 1,
) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key) or 0) + amount


def indexed_json_identity(value: Any) -> tuple[bytes, str]:
    encoded = json.dumps(
        _indexed_json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return encoded, hashlib.sha256(encoded).hexdigest()


def indexed_string_limit(
    key: str,
    *,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> int:
    normalized = str(key or '').strip().lower()
    if (
        normalized in _RAW_MEDIA_PAYLOAD_KEYS
        or normalized.endswith(('_base64', '_b64', '_data_url'))
    ):
        return profile.media_preview_chars
    return profile.text_preview_chars


def indexed_put_bounded_text(
    target: dict[str, Any],
    key: str,
    value: str,
    *,
    stats: Optional[dict[str, int]] = None,
    limit: Optional[int] = None,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> None:
    text_value = str(value)
    encoded = text_value.encode('utf-8')
    inline_limit = max(32, int(limit or profile.inline_limit_bytes))
    if len(encoded) <= inline_limit:
        target[key] = text_value
        return
    preview_limit = min(
        inline_limit,
        indexed_string_limit(key, profile=profile),
    )
    target[key] = text_value[:preview_limit]
    target.setdefault(f'{key}_length_chars', len(text_value))
    target.setdefault(f'{key}_size_bytes', len(encoded))
    target.setdefault(f'{key}_sha256', hashlib.sha256(encoded).hexdigest())
    target.setdefault(f'{key}_preview_truncated', True)
    indexed_stat(stats, 'truncated_text_value_count')


def indexed_bounded_value(
    value: Any,
    *,
    stats: Optional[dict[str, int]] = None,
    depth: int = 0,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        encoded = value.encode('utf-8')
        if len(encoded) <= profile.inline_limit_bytes:
            return value
        indexed_stat(stats, 'truncated_text_value_count')
        return value[:profile.text_preview_chars]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if depth >= profile.depth_limit:
        encoded, sha256 = indexed_json_identity(value)
        indexed_stat(stats, 'truncated_depth_value_count')
        return {
            'wire_projection_truncated': True,
            'source_sha256': sha256,
            'source_size_bytes': len(encoded),
        }
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        items = list(value.items())
        used_keys: set[str] = set()
        for raw_key, item in items:
            raw_key_text = str(raw_key or '').strip()
            key = bounded_projection_mapping_key(raw_key, used_keys=used_keys)
            if key is None:
                continue
            if key != raw_key_text:
                indexed_stat(stats, 'bounded_mapping_key_count')
            if isinstance(item, Path):
                item = str(item)
            if isinstance(item, str):
                indexed_put_bounded_text(
                    projected,
                    key,
                    item,
                    stats=stats,
                    profile=profile,
                )
                continue
            projected[key] = indexed_bounded_value(
                item,
                stats=stats,
                depth=depth + 1,
                profile=profile,
            )
        return projected
    if isinstance(value, list):
        return [
            indexed_bounded_value(
                item,
                stats=stats,
                depth=depth + 1,
                profile=profile,
            )
            for item in value
        ]
    return _indexed_json_safe(value)


INDEXED_RECORD_IDENTITY_KEYS = {
    'artifact_id',
    'artifact_ref',
    'batch_index',
    'branch_id',
    'capability',
    'code',
    'error_code',
    'fill_instance_id',
    'id',
    'kind',
    'lifecycle',
    'mime_type',
    'obligation_id',
    'output_type',
    'path',
    'phase_id',
    'provenance_id',
    'role',
    'saved_audio_path',
    'saved_image_path',
    'saved_text_path',
    'slot_id',
    'status',
    'type',
}


def indexed_record_handle(
    value: Mapping[str, Any],
    *,
    stats: Optional[dict[str, int]] = None,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
    batch_item: bool = False,
) -> dict[str, Any]:
    handle: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key or '').strip()
        if not key or (batch_item and key == 'prompt'):
            continue
        if not (
            key in INDEXED_RECORD_IDENTITY_KEYS
            or key.endswith(('_ref', '_path', '_sha256'))
        ):
            continue
        if isinstance(item, str):
            indexed_put_bounded_text(
                handle,
                key,
                item,
                stats=stats,
                limit=512,
                profile=profile,
            )
        elif isinstance(item, (bool, int, float)) or item is None:
            handle[key] = item
        elif isinstance(item, Mapping):
            handle[key] = indexed_bounded_value(
                item,
                stats=stats,
                profile=profile,
            )
    if batch_item:
        for body_key in (
            'content',
            'content_payload',
            'result',
            'result_text',
            'text',
            'value',
        ):
            body = value.get(body_key)
            if isinstance(body, str) and body:
                handle.setdefault(f'{body_key}_length_chars', len(body))
                handle.setdefault(
                    f'{body_key}_sha256',
                    hashlib.sha256(body.encode('utf-8')).hexdigest(),
                )
        return handle
    encoded, sha256 = indexed_json_identity(value)
    handle['wire_body_preview'] = encoded.decode('utf-8', errors='replace')[
        :profile.text_preview_chars
    ]
    handle['wire_body_size_bytes'] = len(encoded)
    handle['wire_body_sha256'] = sha256
    handle['wire_body_projection_truncated'] = True
    indexed_stat(stats, 'truncated_record_count')
    return handle


def indexed_project_record(
    value: Mapping[str, Any],
    *,
    stats: Optional[dict[str, int]] = None,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    projected = indexed_bounded_value(value, stats=stats, profile=profile)
    if not isinstance(projected, Mapping):
        return {}
    encoded, _sha256 = indexed_json_identity(projected)
    if len(encoded) <= profile.record_limit_bytes:
        return dict(projected)
    return indexed_record_handle(value, stats=stats, profile=profile)


def indexed_project_collection(
    value: Any,
    *,
    stats: Optional[dict[str, int]] = None,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        projected
        for item in value
        if isinstance(item, Mapping)
        for projected in [
            indexed_project_record(item, stats=stats, profile=profile)
        ]
        if projected
    ]


INDEXED_LATE_FILL_BRANCH_KEYS = (
    'active_branches',
    'blocked_branches',
    'cancelled_branches',
    'completed_branches',
    'failed_branches',
    'fill_results',
    'materialization_contract_open_checks',
    'open_branches',
    'pending_branches',
    'queued_branches',
    'running_branches',
)
INDEXED_LATE_FILL_CAPABILITY_KEYS = (
    'active_capabilities',
    'cancelled_capabilities',
    'completed_capabilities',
    'failed_capabilities',
    'pending_capabilities',
    'queued_capabilities',
    'running_capabilities',
    'skipped_capabilities',
)


def indexed_late_fill_projection(
    value: Any,
    *,
    stats: Optional[dict[str, int]] = None,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    scalar_keys = {
        'active_capability',
        'code',
        'error_code',
        'final_materialization_contract_reason',
        'final_materialization_contract_status',
        'kind',
        'late_fill_supported',
        'materialization_blocked',
        'needs_external_input',
        'repair_action',
        'repair_code',
        'repair_scope',
        'repair_trigger',
        'skip_kind',
        'skip_reason',
        'status',
        'trigger',
    }
    for raw_key, item in value.items():
        key = str(raw_key)
        if (
            key.endswith('_count')
            or key.endswith('_snapshot_ref')
            or key in scalar_keys
        ) and item not in (None, '', [], {}):
            if isinstance(item, str):
                indexed_put_bounded_text(
                    projected,
                    key,
                    item,
                    stats=stats,
                    limit=512,
                    profile=profile,
                )
            else:
                projected[key] = indexed_bounded_value(
                    item,
                    stats=stats,
                    profile=profile,
                )
    for key in INDEXED_LATE_FILL_CAPABILITY_KEYS:
        values = value.get(key)
        if not isinstance(values, list) or not values:
            continue
        projected[key] = [
            str(item)
            for item in values
            if str(item or '').strip()
        ]
    repair_actions = value.get('repair_actions')
    if isinstance(repair_actions, list) and repair_actions:
        projected['repair_actions'] = indexed_bounded_value(
            repair_actions,
            stats=stats,
            profile=profile,
        )
    for key in INDEXED_LATE_FILL_BRANCH_KEYS:
        values = value.get(key)
        if isinstance(values, list) and values:
            projected[key] = indexed_project_collection(
                values,
                stats=stats,
                profile=profile,
            )
        elif isinstance(value.get(f'{key}_snapshot_ref'), Mapping):
            projected[f'{key}_snapshot_ref'] = _indexed_json_safe(
                value.get(f'{key}_snapshot_ref')
            )

    for key in (
        'active_branches',
        'cancelled_branches',
        'completed_branches',
        'failed_branches',
        'pending_branches',
    ):
        projected.setdefault(key, [])

    for prefix in (
        'active',
        'blocked',
        'cancelled',
        'completed',
        'failed',
        'pending',
        'queued',
        'running',
    ):
        normalized_count_key = f'{prefix}_count'
        if projected.get(normalized_count_key) not in (None, ''):
            continue
        branch_count = value.get(f'{prefix}_branch_count')
        branch_values = value.get(f'{prefix}_branches')
        if branch_count not in (None, ''):
            projected[normalized_count_key] = branch_count
        elif isinstance(branch_values, list):
            projected[normalized_count_key] = len(branch_values)
        elif isinstance(projected.get(f'{prefix}_branches'), list):
            projected[normalized_count_key] = len(
                projected.get(f'{prefix}_branches') or []
            )
    return projected


def _coerce_frame_sequence(value: Any, fallback: int | None = None) -> int | None:
    if value in (None, ''):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def indexed_batch_projection(
    frame: Mapping[str, Any],
    *,
    outputs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    stats: Optional[dict[str, int]] = None,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    batch = frame.get('batch') if isinstance(frame.get('batch'), Mapping) else {}
    prompts = batch.get('prompts') if isinstance(batch.get('prompts'), list) else []
    count = _coerce_frame_sequence(batch.get('count'), 0) or 0
    outputs_by_index: dict[int, list[dict[str, Any]]] = {}
    artifacts_by_index: dict[int, list[dict[str, Any]]] = {}
    for item in outputs:
        index = _coerce_frame_sequence(item.get('batch_index')) or 0
        if index > 0:
            outputs_by_index.setdefault(index, []).append(item)
    for item in artifacts:
        index = _coerce_frame_sequence(item.get('batch_index')) or 0
        if index > 0:
            artifacts_by_index.setdefault(index, []).append(item)
    indexed_values = set(outputs_by_index) | set(artifacts_by_index)
    count = max([count, len(prompts), *indexed_values], default=0)
    if count <= 0:
        return {}

    if count <= RESPONSE_WIRE_BATCH_DENSE_INDEX_LIMIT:
        candidate_indices = list(range(1, count + 1))
    else:
        candidate_indices = sorted(
            indexed_values | set(range(1, len(prompts) + 1))
        )
    candidate_index_sha256 = indexed_json_identity(candidate_indices)[1]
    results: list[dict[str, Any]] = []
    projected_prompts: list[str] = []
    projected_indices: list[int] = []
    for index in candidate_indices:
        matching_outputs = outputs_by_index.get(index, [])
        matching_artifacts = artifacts_by_index.get(index, [])
        prompt = str(
            prompts[index - 1]
            if index <= len(prompts)
            else next(
                (
                    item.get('prompt')
                    for item in [*matching_outputs, *matching_artifacts]
                    if str(item.get('prompt') or '').strip()
                ),
                '',
            )
            or ''
        )
        result: dict[str, Any] = {
            'index': index,
            'result_handle': True,
            'outputs': [
                indexed_record_handle(
                    item,
                    stats=stats,
                    profile=profile,
                    batch_item=True,
                )
                for item in matching_outputs
            ],
            'artifacts': [
                indexed_record_handle(
                    item,
                    stats=stats,
                    profile=profile,
                    batch_item=True,
                )
                for item in matching_artifacts
            ],
        }
        if prompt:
            indexed_put_bounded_text(
                result,
                'prompt',
                prompt,
                stats=stats,
                limit=profile.text_preview_chars,
                profile=profile,
            )
            projected_prompt = str(result.get('prompt') or '')
        else:
            projected_prompt = ''
        candidate_results = [*results, result]
        candidate_prompts = [*projected_prompts, projected_prompt]
        if (
            results
            and len(
                indexed_json_identity(
                    {
                        'results': candidate_results,
                        'batch_prompts': candidate_prompts,
                    }
                )[0]
            )
            > RESPONSE_WIRE_BATCH_PROJECTION_BUDGET_BYTES
        ):
            break
        results.append(result)
        projected_prompts.append(projected_prompt)
        projected_indices.append(index)
    omitted_candidate_count = len(candidate_indices) - len(results)
    omitted_empty_count = max(0, count - len(candidate_indices))
    prompt_count = _coerce_frame_sequence(
        batch.get('prompts_count'),
        len(prompts),
    ) or 0
    prompt_sha256 = str(batch.get('prompts_sha256') or '').strip()
    if not prompt_sha256:
        prompt_sha256 = indexed_json_identity(prompts)[1]
    projected_prompt_count = sum(
        1 for prompt in projected_prompts if str(prompt or '').strip()
    )
    projection_complete = (
        omitted_candidate_count == 0
        and omitted_empty_count == 0
        and prompt_count <= projected_prompt_count
    )
    projection = {
        'batch_count': count,
        'batch_prompts': projected_prompts,
        'batch_prompts_count': prompt_count,
        'batch_prompts_projected_count': projected_prompt_count,
        'batch_prompts_sha256': prompt_sha256,
        'results': results,
        'batch_results_projected_count': len(results),
        'batch_results_candidate_count': len(candidate_indices),
        'batch_result_indices_sha256': candidate_index_sha256,
        'batch_projection_complete': projection_complete,
    }
    if projected_indices != list(range(1, len(projected_indices) + 1)):
        projection['batch_result_indices'] = projected_indices
    if omitted_candidate_count:
        projection['batch_results_projection_truncated'] = True
        projection['batch_results_omitted_candidate_count'] = omitted_candidate_count
        indexed_stat(
            stats,
            'truncated_batch_result_count',
            omitted_candidate_count,
        )
    if omitted_empty_count:
        projection['batch_empty_result_handles_omitted'] = omitted_empty_count
        projection['batch_results_projection_truncated'] = True
        indexed_stat(
            stats,
            'omitted_empty_batch_result_handle_count',
            omitted_empty_count,
        )
    if prompt_count > projected_prompt_count:
        projection['batch_prompts_projection_truncated'] = True
    prompts_ref = batch.get('prompts_snapshot_ref')
    if (
        isinstance(prompts_ref, Mapping)
        and str(prompts_ref.get('projection_role') or '').strip()
        == 'public_body_exact'
    ):
        projection['batch_prompts_snapshot_ref'] = _indexed_json_safe(prompts_ref)
    return projection


def indexed_snapshot_manifest_projection(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    items = sorted(
        (
            (str(path), _indexed_json_safe(ref))
            for path, ref in value.items()
            if isinstance(ref, Mapping)
        ),
        key=lambda item: item[0],
    )
    total_count = len(items)
    projected = dict(items)
    metadata: dict[str, Any] = {
        'effective_snapshot_count': total_count,
        'projected_snapshot_ref_count': len(projected),
        'manifest_projection_complete': total_count <= len(projected),
    }
    return projected, metadata


def indexed_frame_projection(
    frame: Mapping[str, Any],
    *,
    effective_snapshot_manifest: Mapping[str, Any],
    snapshot_manifest_projection: Optional[Mapping[str, Any]] = None,
    outputs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    lifecycle_state: str,
    profile: ResponseWireProfile = INDEXED_RESPONSE_WIRE_PROFILE,
) -> dict[str, Any]:
    def projected_total_count(*values: Any, fallback: int) -> int:
        total = max(0, fallback)
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                total = max(total, value)
        return total

    projected = {
        key: _indexed_json_safe(frame.get(key))
        for key in (
            'created_at',
            'frame_id',
            'frame_relation',
            'frame_sequence',
            'frame_version',
            'kind',
            'response_id',
            'status',
            'updated_at',
        )
        if frame.get(key) not in (None, '', [], {})
    }
    compact_current = (
        frame.get('current_state')
        if isinstance(frame.get('current_state'), Mapping)
        else {}
    )
    current_state = {
        key: _indexed_json_safe(compact_current.get(key))
        for key in (
            'canonical_status_field',
            'error_ref',
            'id',
            'lang_code',
            'lang_code_source',
            'lifecycle_state',
            'message_id',
            'output_format',
            'response_format',
            'recovery_hint',
            'route_artifact_path',
            'route_artifact_ref',
            'route_confidence',
            'route_reason',
            'route_reuse_last_artifact',
            'route_source',
            'status',
            'status_compatibility',
            'status_semantics',
            'updated_at',
            'usage',
        )
        if compact_current.get(key) not in (None, '', [], {})
    }
    current_state['lifecycle_state'] = lifecycle_state
    for key, item in compact_current.items():
        if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
            current_state[key] = _indexed_json_safe(item)
            if str(item.get('projection_role') or '').strip() == 'public_body_exact':
                for metadata_key in item.get('public_projection_metadata_keys') or []:
                    metadata_key = str(metadata_key or '').strip()
                    if (
                        metadata_key
                        and compact_current.get(metadata_key) not in (None, '', [], {})
                    ):
                        current_state[metadata_key] = _indexed_json_safe(
                            compact_current.get(metadata_key)
                        )
    projected['current_state'] = current_state
    if isinstance(frame.get('route'), Mapping):
        projected['route'] = indexed_bounded_value(
            frame.get('route'),
            profile=profile,
        )

    compact_output = frame.get('output') if isinstance(frame.get('output'), Mapping) else {}
    output_projection: dict[str, Any] = {
        'item_count': projected_total_count(
            compact_current.get('outputs_count'),
            compact_output.get('item_count'),
            compact_output.get('outputs_count'),
            fallback=len(outputs),
        ),
        'outputs': _indexed_json_safe(outputs),
    }
    if isinstance(compact_output.get('artifact_identity'), Mapping):
        output_projection['artifact_identity'] = _indexed_json_safe(
            compact_output.get('artifact_identity')
        )
    projected['output'] = output_projection

    compact_artifacts = (
        frame.get('artifacts') if isinstance(frame.get('artifacts'), Mapping) else {}
    )
    artifact_projection: dict[str, Any] = {
        'output': _indexed_json_safe(artifacts),
        'output_count': projected_total_count(
            compact_current.get('artifacts_count'),
            compact_artifacts.get('output_count'),
            fallback=len(artifacts),
        ),
    }
    if isinstance(compact_artifacts.get('identity'), Mapping):
        artifact_projection['identity'] = _indexed_json_safe(
            compact_artifacts.get('identity')
        )
    for key, item in compact_artifacts.items():
        if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
            artifact_projection[key] = _indexed_json_safe(item)
    projected['artifacts'] = artifact_projection

    late_fill = indexed_late_fill_projection(
        frame.get('late_fill'),
        profile=profile,
    )
    if late_fill:
        projected['late_fill'] = late_fill
    for key, item in frame.items():
        if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
            projected[key] = _indexed_json_safe(item)
    if isinstance(frame.get('snapshot_policy'), Mapping):
        projected['snapshot_policy'] = indexed_bounded_value(
            frame.get('snapshot_policy'),
            profile=profile,
        )
    if isinstance(frame.get('public_body_compaction'), Mapping):
        projected['public_body_compaction'] = indexed_bounded_value(
            frame.get('public_body_compaction'),
            profile=profile,
        )

    compact_external = (
        frame.get('external_snapshots')
        if isinstance(frame.get('external_snapshots'), Mapping)
        else {}
    )
    raw_delta_items = (
        compact_external.get('items')
        if isinstance(compact_external.get('items'), Mapping)
        else {}
    )
    delta_items, delta_projection = indexed_snapshot_manifest_projection(
        raw_delta_items
    )
    manifest_projection = dict(snapshot_manifest_projection or {})
    external_snapshots: dict[str, Any] = {
        'kind': str(
            compact_external.get('kind')
            or 'ollmo.response_frame_external_snapshots'
        ),
        'storage': str(compact_external.get('storage') or 'sidecar_json'),
        'version': compact_external.get('version') or 1,
        'items': _indexed_json_safe(effective_snapshot_manifest),
        'delta_items': _indexed_json_safe(delta_items),
        'effective_manifest_expanded': bool(
            manifest_projection.get('manifest_projection_complete', True)
        ),
        'effective_snapshot_count': manifest_projection.get(
            'effective_snapshot_count',
            len(effective_snapshot_manifest),
        ),
        'projected_snapshot_ref_count': len(effective_snapshot_manifest),
        'sidecar_hydration': 'none',
    }
    for key in (
        'manifest_omitted_ref_count',
        'manifest_projection_complete',
        'manifest_projection_truncated',
        'manifest_sha256',
        'manifest_size_bytes',
    ):
        if manifest_projection.get(key) not in (None, '', [], {}):
            external_snapshots[key] = _indexed_json_safe(
                manifest_projection.get(key)
            )
    if delta_projection.get('manifest_projection_truncated'):
        external_snapshots['delta_manifest_projection'] = _indexed_json_safe(
            delta_projection
        )
    if isinstance(compact_external.get('inheritance'), Mapping):
        external_snapshots['inheritance'] = _indexed_json_safe(
            compact_external.get('inheritance')
        )
    projected['external_snapshots'] = external_snapshots
    return _indexed_json_safe(projected)
