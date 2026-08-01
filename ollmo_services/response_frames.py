"""Response-frame helpers for canonical Ollmo Responses payloads."""

from __future__ import annotations

import base64
import binascii
import json
import hashlib
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_orchestration.working_frame import build_working_frame, compact_working_frame_for_serialization
from ollmo_services.artifact_dossiers import build_artifact_dossier_index
from ollmo_services.control_snapshots import build_control_snapshot
from ollmo_services.frame_planning import build_artifact_flow_plan
from ollmo_services.redraw_scope import canonicalize_duplicate_artifact_refs
from ollmo_services import response_wire as _response_wire_policy
from ollmo_services.responses import (
    build_canonical_response_artifacts,
    build_canonical_outputs,
    build_public_output_branches_from_slots,
    extract_responses_current_turn_prompt,
    filter_public_response_artifacts,
    merge_canonical_response_artifacts,
    select_public_output_text as select_canonical_public_output_text,
)

DEFAULT_RESPONSE_FRAMES_DIR = Path('state/response_frames')
DEFAULT_RESPONSE_FRAME_LEDGER = 'responses.jsonl'
DEFAULT_RESPONSE_FRAME_INDEX = 'current_index.json'
DEFAULT_RESPONSE_FRAME_SNAPSHOT_DIR = 'snapshots'
DEFAULT_RESPONSE_FRAME_SNAPSHOT_CONTENT_DIR = 'content_sha256'
RESPONSE_FRAME_VERSION = 9

RESPONSE_FRAME_STALE_PARENT_REASON = 'response_frame_parent_stale'
_RESPONSE_FRAME_APPEND_LOCK = threading.RLock()


class ResponseFrameParentCASMismatch(RuntimeError):
    """Raised when an atomic response-frame append sees a different parent."""

    code = RESPONSE_FRAME_STALE_PARENT_REASON

    def __init__(
        self,
        *,
        response_id: str,
        expected_parent_frame_id: str,
        current_parent_frame_id: str | None,
        expected_parent_frame_sequence: int | None = None,
        current_parent_frame_sequence: int | None = None,
    ) -> None:
        super().__init__(
            'Response-frame parent changed before the successor could be appended.'
        )
        self.response_id = response_id
        self.expected_parent_frame_id = expected_parent_frame_id
        self.current_parent_frame_id = current_parent_frame_id
        self.expected_parent_frame_sequence = expected_parent_frame_sequence
        self.current_parent_frame_sequence = current_parent_frame_sequence

    def as_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'response_id': self.response_id,
            'expected_parent_frame_id': self.expected_parent_frame_id,
            'current_parent_frame_id': self.current_parent_frame_id,
            'expected_parent_frame_sequence': self.expected_parent_frame_sequence,
            'current_parent_frame_sequence': self.current_parent_frame_sequence,
        }

_LEDGER_SNAPSHOT_REF_KIND = 'ollmo.response_frame_snapshot_ref'
_LEDGER_SNAPSHOT_POLICY_KIND = 'ollmo.response_frame_snapshot_policy'
_LEDGER_LARGE_CONTRACT_LIMIT_BYTES = 8192
_NESTED_SNAPSHOT_LIMIT_BYTES = 65536
_SIDECAR_SPLIT_LIMIT_BYTES = 32768
_SIDECAR_SPLIT_MAX_DEPTH = 8
_SIDECAR_SPLIT_MAX_REFS = 64
_ADVISORY_SIDECAR_SPLIT_LIMIT_BYTES = 512
_LEDGER_TEXT_SUMMARY_LIMIT = 1200
_MEDIA_PAYLOAD_MIN_CHARS = 1024
_PUBLIC_BODY_SNAPSHOT_MIN_BYTES = 256 * 1024
_PUBLIC_COLLECTION_SNAPSHOT_MIN_BYTES = 1024 * 1024
_PUBLIC_COLLECTION_PREVIEW_BUDGET_BYTES = 64 * 1024
_PUBLIC_BODY_PREVIEW_LIMIT_CHARS = 4096
_PUBLIC_PREVIEW_MAPPING_KEY_LIMIT_BYTES = 256
_RESPONSE_WIRE_INLINE_BUDGET_BYTES = (
    _response_wire_policy.RESPONSE_WIRE_INLINE_LIMIT_BYTES
)
_PUBLIC_AGGREGATE_LEDGER_TARGET_BYTES = 1024 * 1024
_PUBLIC_AGGREGATE_COLLECTION_MIN_BYTES = 32 * 1024
_RESPONSE_WIRE_BATCH_PROJECTION_BUDGET_BYTES = (
    _response_wire_policy.RESPONSE_WIRE_BATCH_PROJECTION_BUDGET_BYTES
)
_RESPONSE_WIRE_BATCH_DENSE_INDEX_LIMIT = (
    _response_wire_policy.RESPONSE_WIRE_BATCH_DENSE_INDEX_LIMIT
)
_COMPACT_RUNTIME_KEYS = (
    'routing_policy',
    'route_traits',
    'accepted_learning_hints',
)
_RAW_MEDIA_PAYLOAD_KEYS = {
    'audio': 'audio',
    'audio_base64': 'audio',
    'audio_b64': 'audio',
    'audio_data': 'audio',
    'image': 'image',
    'image_base64': 'image',
    'image_b64': 'image',
    'image_data': 'image',
    'png_base64': 'image',
    'video': 'video',
    'video_base64': 'video',
    'video_b64': 'video',
    'video_data': 'video',
}

_OUTPUT_SLOT_STATUS_SCORES = {
    'fulfilled': 12,
    'completed': 12,
    'blocked': 8,
    'failed': 8,
    'cancelled': 8,
    'waived': 8,
    'superseded': 8,
    'pending': 2,
    'queued': 2,
    'running': 2,
    'scheduled': 2,
}
_MEDIA_SAVED_PATH_KEYS = {
    'audio': ('saved_audio_path', 'audio_path', 'saved_audio_file'),
    'image': ('saved_image_path', 'image_path', 'saved_image_file'),
    'video': ('saved_video_path', 'video_path', 'saved_video_file'),
}
_GENERIC_ARTIFACT_PATH_KEYS = (
    'artifact_path',
    'path',
    'saved_path',
)
_SIDECAR_SPLIT_KEYS = {
    'active_reconsideration_review',
    'artifact_dossiers',
    'branches',
    'candidate_graph',
    'commitment_review',
    'context_candidates',
    'context_gate_review',
    'controlled_attention_review',
    'decision_contract',
    'checks',
    'dependencies',
    'dependency_chain',
    'edges',
    'evidence',
    'execution_contract',
    'execution_contracts',
    'execution_planner',
    'fill_results',
    'graph',
    'graph_closure_review',
    'history_candidates',
    'input_artifacts',
    'memory_candidates',
    'nodes',
    'obligations',
    'output_candidates',
    'output_obligations',
    'phases',
    'promotion_review',
    'recovery_context',
    'reference_artifacts',
    'reference_candidates',
    'request_phase_graph',
    'runtime_truth',
    'semantic_planning_contract',
    'semantic_quality_review',
    'semantic_review_lenses',
    'selected_references',
    'work_tree',
}
_ADVISORY_SIDECAR_SPLIT_KEYS = {
    'active_reconsideration_decision',
    'active_reconsideration_decisions',
    'active_reconsideration_review',
    'aspiration_frame',
    'aspiration_review',
    'block_resolution_reflex',
    'block_resolution_signal',
    'block_resolution_signals',
    'commitment_frame',
    'commitment_frames',
    'commitment_review',
    'controlled_attention_frame',
    'controlled_attention_frames',
    'controlled_attention_review',
    'decision_contract_active_reconsideration_decisions',
    'decision_contract_aspiration_frames',
    'decision_contract_commitment_frames',
    'decision_contract_controlled_attention_frames',
    'decision_contract_guidance',
    'decision_contract_semantic_decision_proposals',
    'decision_contract_semantic_review_lenses',
    'evidence_requirements',
    'failure_modes',
    'focus_questions',
    'ghost_repair_feedback',
    'accepted_learning',
    'accepted_learning_boundary',
    'accepted_learning_hints',
    'accepted_learning_policy',
    'graph_patch_successor_reopen_request',
    'graph_patch_successor_reopen_requests',
    'graph_rebase_autonomy',
    'graph_rebase_enforced_policy_review',
    'graph_rebase_enforced_policy_reviews',
    'graph_rebase_enforcement',
    'graph_rebase_lifecycle',
    'graph_rebase_lifecycles',
    'graph_repair',
    'graph_repair_proposal',
    'graph_repair_proposal_review',
    'graph_repair_proposal_reviews',
    'graph_repair_proposals',
    'graph_repair_review',
    'graph_repair_reviews',
    'intent_graph_adequacy',
    'intent_graph_adequacy_review',
    'redraw_scope',
    'redraw_scope_ladder_review',
    'redraw_scope_ladder_reviews',
    'redraw_scope_orientation',
    'reconsideration_reflex_signals',
    'recursive_cycle_review',
    'runtime_graph_repair_proposal_reviews',
    'runtime_graph_repair_proposals',
    'semantic_decision_proposal',
    'semantic_decision_proposals',
    'semantic_decision_review',
    'semantic_lens_evidence_requirements',
    'semantic_quality_contract',
    'semantic_quality_contracts',
    'semantic_quality_review',
    'semantic_review_lens_contract',
    'semantic_review_lenses',
    'successor_reopen_request',
    'successor_reopen_requests',
}
_ADVISORY_CONTEXT_SIDECAR_SPLIT_PATHS = {
    'accepted_learnings': (
        'accepted_learning_policy.accepted_learnings',
    ),
    'checks': (
        'intent_graph_adequacy.checks',
        'intent_graph_adequacy_review.checks',
    ),
    'frames': (
        'controlled_attention_review.frames',
    ),
    'hints': (
        'accepted_learning_hints.hints',
    ),
    'items': (
        'ghost_repair_feedback.items',
        'surface_state.items',
    ),
    'proposals': (
        'graph_repair.proposals',
        'semantic_decision_review.proposals',
    ),
    'reviews': (
        'graph_repair.reviews',
    ),
}
_SIDECAR_SPLIT_SUFFIXES = (
    '_candidates',
    '_contract',
    '_contracts',
    '_evidence',
    '_graph',
    '_graphs',
    '_review',
    '_reviews',
)
_SNAPSHOT_VOLATILE_TIMESTAMP_KEYS = {
    'created_at',
    'updated_at',
    'timestamp',
    'started_at',
    'completed_at',
    'finished_at',
    'last_seen_at',
    'last_updated_at',
    'expires_at',
}
_SNAPSHOT_REF_VOLATILE_KEYS = {
    'json_path',
    'sidecar_manifest',
    'sidecar_parent_json_path',
    'source_frame_id',
    'source_response_id',
}
_TIMESTAMP_NORMALIZED_SNAPSHOT_PATH_TOKENS = {
    'candidate_graph',
    'context_candidates',
    'context_contract',
    'context_strategy',
    'execution_planner',
    'graph_closure_review',
    'intent_contract',
    'late_fill',
    'promotion_review',
    'request_phase_graph',
    'semantic_role_profile',
    'work_tree',
    'working_frame',
}
_LATE_FILL_REVIEW_KEYS = (
    'active_reconsideration_review',
    'aspiration_review',
    'commitment_review',
    'controlled_attention_review',
    'recursive_cycle_review',
    'semantic_decision_review',
    'semantic_quality_review',
    'surface_state',
)
_LATE_FILL_SNAPSHOT_CHILD_KEYS = tuple(_LATE_FILL_REVIEW_KEYS) + (
    'fill_results',
    'recovery_context',
)
_CONTRACT_HEAVY_KEYS = {
    'active_reconsideration_review',
    'aspiration_review',
    'candidate_graph',
    'checks',
    'commitment_review',
    'context_candidates',
    'context_gate_review',
    'controlled_attention_review',
    'decision_contract_commitment_frames',
    'decision_contract_controlled_attention_frames',
    'decision_contract_semantic_decision_proposals',
    'decision_contract_semantic_review_lenses',
    'output_candidates',
    'promotion_review',
    'promotions',
    'recursive_cycle_review',
    'semantic_decision_review',
    'semantic_quality_review',
}
_ARTIFACT_PROJECTION_DETAIL_KEYS = {
    'artifact',
    'artifacts',
    'artifact_path',
    'path',
    'source_path',
    'saved_text_path',
    'saved_audio_path',
    'saved_image_path',
    'mime_type',
    'metadata',
    'provenance',
    'provenance_id',
    'derived_from',
    'image_state',
    'image_state_enrichment',
    'seed',
    'prompt',
    'source_response_id',
    'source_message_id',
    'url',
}
_TERMINAL_OUTPUT_STATUSES = {
    'blocked',
    'cancelled',
    'completed',
    'failed',
    'fulfilled',
    'skipped',
    'superseded',
    'waived',
}

_CURRENT_STATE_FRAME_KEYS = (
    'id',
    'object',
    'status',
    'lifecycle_state',
    'canonical_status_field',
    'status_compatibility',
    'status_semantics',
    'mode',
    'instance_id',
    'model',
    'request_model',
    'backend',
    'capability',
    'output_text',
    'output',
    'result',
    'outputs',
    'output_slots',
    'output_branches',
    'work_tree',
    'artifacts',
    'input_artifacts',
    'reference_artifacts',
    'late_fill',
    'runtime',
    'saved_text_path',
    'saved_text_artifacts',
    'text_artifact_requests',
    'saved_audio_path',
    'saved_image_path',
    'image_state',
    'audio_mimetype',
    'tts_audio_integrity_evidence',
    'seed',
    'provenance_id',
    'message_id',
    'route_source',
    'route_reason',
    'route_confidence',
    'route_reuse_last_artifact',
    'route_artifact_ref',
    'route_artifact_path',
    'usage',
    'lang_code',
    'lang_code_source',
    'response_format',
    'output_format',
    'error',
    'error_detail',
    'error_ref',
    'recovery_hint',
)

_REQUEST_FRAME_KEYS = (
    'request_id',
    'conversation_id',
    'instance_id',
    'model',
    'request_model',
    'backend',
    'capability',
    'mode',
    'prompt',
    'input',
    'instructions',
    'file_path',
    'batch_prompts',
    'ghost_route',
    'ghost_preview',
    'request_meta',
    'reference_artifacts',
    'selected_reference_artifacts',
    'context_candidates',
    'memory_candidates',
    'history_candidates',
    'reference_candidates',
    'input_artifacts',
    'stream',
)
_REQUEST_FRAME_DIRECT_KEYS = tuple(
    key for key in _REQUEST_FRAME_KEYS if key != 'ghost_preview'
)
_GHOST_PREVIEW_AUDIT_TEXT_LIMIT = 512
_GHOST_PREVIEW_AUDIT_ITEM_LIMIT = 16
_GHOST_PREVIEW_AUDIT_KEY_LIMIT = 128
_GHOST_PREVIEW_AUDIT_SCAN_LIMIT = 64
_GHOST_PREVIEW_INSTANCE_KEYS = (
    'instance_id',
    'model',
    'backend',
    'capability',
    'supported_capabilities',
    'text_capable',
    'backend_package',
    'backend_contract',
    'provider_capabilities',
    'tts_model_type',
    'tts_speakers',
    'tts_languages',
    'readiness',
    'activity',
)
_GHOST_PREVIEW_ROUTE_KEYS = (
    'source',
    'reason',
    'confidence',
    'reuse_last_artifact',
    'artifact_ref',
    'artifact_path',
    'context_mode',
    'context_reason',
)
_GHOST_PREVIEW_REQUEST_META_KEYS = (
    'ghost_mode',
    'ghost_mode_source',
    'capability_hint',
    'capability_hint_source',
    'language_hint',
    'language_hint_source',
    'source',
    'case_id',
    'corpus_id',
    'corpus_digest',
    'wave',
    'workload_family',
)
_GHOST_PREVIEW_DEVELOPER_DIAGNOSTIC_KEYS = (
    'routing_contract',
    'routing_policy',
    'heuristic_role',
    'embedding_signals_enabled',
    'accepted_learning_authority',
    'planner_timeout_ms',
    'planner_timeout_sec',
)
_GHOST_PREVIEW_LEGACY_KEYS = (
    'instance_id',
    'instanceId',
    'capability',
    'reuse_last_artifact',
    'artifact_ref',
    'artifactRef',
    'artifact_path',
    'confidence',
    'reason',
    'route_source',
    'routeSource',
)

_TARGET_FRAME_KEYS = (
    'instance_id',
    'model',
    'backend',
    'capability',
    'mode',
)

_ROUTE_FRAME_KEYS = (
    'route_source',
    'route_reason',
    'route_confidence',
    'route_reuse_last_artifact',
    'route_artifact_ref',
    'route_artifact_path',
    'route_traits',
)


def _is_empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or '').strip()
            if not key or key in {'response_frame', 'image_data_url', 'imageDataUrl'}:
                continue
            if _is_empty(raw_value):
                continue
            payload[key] = _json_safe(raw_value)
        return payload
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value if not _is_empty(item)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _select_frame_keys(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        if key not in source:
            continue
        value = source.get(key)
        if _is_empty(value):
            continue
        payload[key] = _json_safe(value)
    return payload


def _bounded_ghost_preview_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if len(value) <= _GHOST_PREVIEW_AUDIT_TEXT_LIMIT:
            return value
        return f'{value[:_GHOST_PREVIEW_AUDIT_TEXT_LIMIT - 3]}...'
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return None


def _bounded_ghost_preview_key(value: Any) -> Optional[str]:
    key = str(value or '').strip()
    if not key or len(key) > _GHOST_PREVIEW_AUDIT_KEY_LIMIT:
        return None
    return key


def _ghost_preview_content_digest_ref(value: Any, *, json_path: str) -> dict[str, Any]:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return {
        'kind': 'ollmo.ghost_preview_content_digest_ref',
        'json_path': json_path,
        'sha256': hashlib.sha256(encoded).hexdigest(),
        'size_bytes': len(encoded),
        'content_addressed': True,
        'storage': 'digest_only',
        'authority': 'audit_identity_only',
    }


def _compact_ghost_preview_scalar_mapping(
    value: Any,
    *,
    preferred_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    source = value
    compact: dict[str, Any] = {}
    seen_keys: set[str] = set()
    ordered_items: list[tuple[str, Any]] = []
    for preferred_key in preferred_keys:
        if preferred_key in source and preferred_key not in seen_keys:
            seen_keys.add(preferred_key)
            ordered_items.append((preferred_key, source.get(preferred_key)))
    scanned_unknown = 0
    for raw_key, raw in source.items():
        key = _bounded_ghost_preview_key(raw_key)
        if key is None or key in seen_keys:
            continue
        scanned_unknown += 1
        if scanned_unknown > _GHOST_PREVIEW_AUDIT_SCAN_LIMIT:
            break
        seen_keys.add(key)
        ordered_items.append((key, raw))
    for key, raw in ordered_items:
        if len(compact) >= _GHOST_PREVIEW_AUDIT_ITEM_LIMIT:
            break
        if raw in (None, '', [], {}):
            continue
        if isinstance(raw, (list, tuple)):
            items = [
                bounded
                for item in raw[:_GHOST_PREVIEW_AUDIT_ITEM_LIMIT]
                if (bounded := _bounded_ghost_preview_scalar(item)) not in (None, '')
            ]
            if items:
                compact[key] = items
            continue
        bounded = _bounded_ghost_preview_scalar(raw)
        if bounded not in (None, ''):
            compact[key] = bounded
    return compact


def _ghost_preview_omitted_key_summary(
    value: Any,
    *,
    retained_keys: set[str],
) -> tuple[int, list[str]]:
    if not isinstance(value, Mapping):
        return 0, []
    count = 0
    keys: list[str] = []
    for raw_key, item in value.items():
        key = _bounded_ghost_preview_key(raw_key)
        if key in retained_keys or item in (None, '', [], {}):
            continue
        count += 1
        if key is not None and len(keys) < _GHOST_PREVIEW_AUDIT_ITEM_LIMIT:
            keys.append(key)
    return count, sorted(keys)


def _compact_request_ghost_preview(value: Any) -> Optional[dict[str, Any]]:
    """Keep route identity/audit truth without duplicating orchestration bodies."""

    if not isinstance(value, Mapping):
        return None
    preview = value
    payload = _compact_ghost_preview_scalar_mapping(
        preview,
        preferred_keys=_GHOST_PREVIEW_LEGACY_KEYS,
    )

    instance_source = preview.get('instance') if isinstance(preview.get('instance'), Mapping) else {}
    instance = _compact_ghost_preview_scalar_mapping(
        instance_source,
        preferred_keys=_GHOST_PREVIEW_INSTANCE_KEYS,
    )
    if instance_source:
        for key in ('backend_metadata', 'backend_runtime', 'session_controls', 'session_controls_summary'):
            nested = _compact_ghost_preview_scalar_mapping(instance_source.get(key))
            if nested:
                instance[key] = nested
    if instance:
        payload['instance'] = instance

    route = _compact_ghost_preview_scalar_mapping(
        preview.get('route'),
        preferred_keys=_GHOST_PREVIEW_ROUTE_KEYS,
    )
    route_source = preview.get('route') if isinstance(preview.get('route'), Mapping) else {}
    route_traits = _compact_ghost_preview_scalar_mapping(route_source.get('traits'))
    if route_traits:
        route['traits'] = route_traits
    if route:
        payload['route'] = route

    request_meta = _compact_ghost_preview_scalar_mapping(
        preview.get('request_meta'),
        preferred_keys=_GHOST_PREVIEW_REQUEST_META_KEYS,
    )
    request_meta_source = (
        preview.get('request_meta')
        if isinstance(preview.get('request_meta'), Mapping)
        else {}
    )
    developer_flags = _compact_ghost_preview_scalar_mapping(
        request_meta_source.get('developer_flags')
    )
    if developer_flags:
        request_meta['developer_flags'] = developer_flags
    if request_meta:
        payload['request_meta'] = request_meta

    runtime = preview.get('runtime') if isinstance(preview.get('runtime'), Mapping) else {}
    diagnostics_source = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    diagnostics = _compact_ghost_preview_scalar_mapping(
        diagnostics_source,
        preferred_keys=_GHOST_PREVIEW_DEVELOPER_DIAGNOSTIC_KEYS,
    )
    route_graph_consistency = _compact_ghost_preview_scalar_mapping(
        diagnostics_source.get('route_graph_consistency')
    )
    if route_graph_consistency:
        diagnostics['route_graph_consistency'] = route_graph_consistency
    if diagnostics:
        payload['runtime'] = {'developer_diagnostics': diagnostics}

    retained_preview_keys = set(payload)
    retained_preview_keys.add('compaction')
    omitted_preview_key_count, omitted_preview_keys = _ghost_preview_omitted_key_summary(
        preview,
        retained_keys=retained_preview_keys,
    )
    omitted_runtime_key_count, omitted_runtime_keys = _ghost_preview_omitted_key_summary(
        runtime,
        retained_keys={'developer_diagnostics'},
    )
    omitted_diagnostic_key_count, omitted_diagnostic_keys = _ghost_preview_omitted_key_summary(
        diagnostics_source,
        retained_keys=set(diagnostics),
    )
    detail_omission_count = 0
    detail_omission_paths: list[str] = []
    detail_omission_counts_by_path: dict[str, int] = {}
    detail_sources: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = [
        ('ghost_preview.instance', instance_source, instance),
        ('ghost_preview.route', route_source, route),
        ('ghost_preview.request_meta', request_meta_source, request_meta),
    ]
    developer_flags_source = (
        request_meta_source.get('developer_flags')
        if isinstance(request_meta_source.get('developer_flags'), Mapping)
        else {}
    )
    if developer_flags:
        detail_sources.append(
            ('ghost_preview.request_meta.developer_flags', developer_flags_source, developer_flags)
        )
    route_graph_consistency_source = (
        diagnostics_source.get('route_graph_consistency')
        if isinstance(diagnostics_source.get('route_graph_consistency'), Mapping)
        else {}
    )
    if route_graph_consistency:
        detail_sources.append(
            (
                'ghost_preview.runtime.developer_diagnostics.route_graph_consistency',
                route_graph_consistency_source,
                route_graph_consistency,
            )
        )
    for key in ('backend_metadata', 'backend_runtime', 'session_controls', 'session_controls_summary'):
        nested_source = instance_source.get(key) if isinstance(instance_source.get(key), Mapping) else {}
        nested_compact = instance.get(key) if isinstance(instance.get(key), Mapping) else {}
        if nested_compact:
            detail_sources.append((f'ghost_preview.instance.{key}', nested_source, nested_compact))
    route_traits_source = route_source.get('traits') if isinstance(route_source.get('traits'), Mapping) else {}
    if route_traits:
        detail_sources.append(('ghost_preview.route.traits', route_traits_source, route_traits))
    for base_path, detail_source, detail_compact in detail_sources:
        omitted_count, omitted_keys = _ghost_preview_omitted_key_summary(
            detail_source,
            retained_keys=set(detail_compact),
        )
        detail_omission_counts_by_path[base_path] = omitted_count
        detail_omission_count += omitted_count
        for key in omitted_keys:
            if len(detail_omission_paths) >= _GHOST_PREVIEW_AUDIT_ITEM_LIMIT:
                break
            detail_omission_paths.append(f'{base_path}.{key}')

    content_digest_refs: list[dict[str, Any]] = []
    working_frame_source = (
        preview.get('working_frame')
        if isinstance(preview.get('working_frame'), Mapping)
        else {}
    )
    if working_frame_source:
        content_digest_refs.append(
            _ghost_preview_content_digest_ref(
                working_frame_source,
                json_path='ghost_preview.working_frame',
            )
        )
    runtime_detail_omitted = any(
        count > 0 and path.startswith('ghost_preview.runtime.')
        for path, count in detail_omission_counts_by_path.items()
    )
    if runtime and (
        omitted_runtime_key_count
        or omitted_diagnostic_key_count
        or runtime_detail_omitted
    ):
        content_digest_refs.append(
            _ghost_preview_content_digest_ref(runtime, json_path='ghost_preview.runtime')
        )
    for json_path, source_value in (
        ('ghost_preview.instance', instance_source),
        ('ghost_preview.route', route_source),
        ('ghost_preview.request_meta', request_meta_source),
    ):
        source_detail_omitted = any(
            count > 0 and (path == json_path or path.startswith(f'{json_path}.'))
            for path, count in detail_omission_counts_by_path.items()
        )
        if source_value and source_detail_omitted:
            content_digest_refs.append(
                _ghost_preview_content_digest_ref(source_value, json_path=json_path)
            )
    payload['compaction'] = {
        'kind': 'ollmo.ghost_preview_response_frame_projection',
        'policy': 'route_identity_and_bounded_routing_audit_only',
        'duplicate_orchestration_truth_omitted': bool(
            working_frame_source
            or omitted_runtime_key_count
            or omitted_diagnostic_key_count
        ),
        'preview_detail_omitted': bool(
            omitted_preview_key_count
            or omitted_runtime_key_count
            or omitted_diagnostic_key_count
            or detail_omission_count
        ),
        'canonical_response_truth_paths': [
            'runtime',
            'planning.request_phase_graph',
            'working_frame',
        ],
        'omitted_preview_key_count': omitted_preview_key_count,
        'omitted_preview_keys': omitted_preview_keys,
        'omitted_runtime_key_count': omitted_runtime_key_count,
        'omitted_runtime_keys': omitted_runtime_keys,
        'omitted_developer_diagnostic_key_count': omitted_diagnostic_key_count,
        'omitted_developer_diagnostic_keys': omitted_diagnostic_keys,
        'omitted_nested_detail_count': detail_omission_count,
        'omitted_nested_detail_paths': sorted(detail_omission_paths),
        'omitted_content_refs': content_digest_refs,
    }
    return _json_safe(payload)


def _build_request_frame(request_payload: Optional[Mapping[str, Any]], response_payload: Mapping[str, Any]) -> dict[str, Any]:
    source = request_payload if isinstance(request_payload, Mapping) else {}
    payload = _select_frame_keys(source, _REQUEST_FRAME_DIRECT_KEYS)
    ghost_preview = _compact_request_ghost_preview(source.get('ghost_preview'))
    if ghost_preview:
        payload['ghost_preview'] = ghost_preview
    if not payload and isinstance(response_payload.get('request_meta'), Mapping):
        payload['request_meta'] = _json_safe(response_payload.get('request_meta'))
    input_artifacts = response_payload.get('input_artifacts')
    if 'input_artifacts' not in payload and isinstance(input_artifacts, list) and input_artifacts:
        payload['input_artifacts'] = _json_safe(input_artifacts)
    reference_artifacts = response_payload.get('reference_artifacts')
    if 'reference_artifacts' not in payload:
        if isinstance(reference_artifacts, list) and reference_artifacts:
            payload['reference_artifacts'] = _json_safe(reference_artifacts)
        else:
            legacy_references = (
                source.get('reference_artifacts')
                if isinstance(source.get('reference_artifacts'), list)
                else (
                    source.get('selected_reference_artifacts')
                    if isinstance(source.get('selected_reference_artifacts'), list)
                    else None
                )
            )
            if isinstance(legacy_references, list) and legacy_references:
                payload['reference_artifacts'] = _json_safe(legacy_references)
    return payload


def _build_error_frame(response_payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    error_detail = response_payload.get('error_detail')
    if isinstance(error_detail, Mapping):
        return _json_safe(error_detail)
    error_value = response_payload.get('error')
    if isinstance(error_value, Mapping):
        return _json_safe(error_value)
    if not _is_empty(error_value):
        return {'message': str(error_value)}
    status = str(response_payload.get('status') or '').strip().lower()
    if status in {'failed', 'error', 'cancelled'}:
        return {'message': 'Request failed.'}
    return None


def _build_runtime_frame(response_payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(response_payload.get('runtime') or {}) if isinstance(response_payload.get('runtime'), Mapping) else {}
    runtime.pop('working_frame', None)
    return _json_safe(runtime)


def _select_public_output_text(
    response_payload: Mapping[str, Any],
    outputs: Any,
) -> str:
    fallback = str(response_payload.get('output_text') or '').strip()
    if _response_truth_guard_requires_clarification(response_payload):
        return fallback
    return select_canonical_public_output_text(response_payload, outputs, fallback_text=fallback)


def _response_truth_guard(response_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    truth_guard = runtime.get('truth_guard') if isinstance(runtime.get('truth_guard'), Mapping) else {}
    return truth_guard


def _response_truth_guard_requires_clarification(response_payload: Mapping[str, Any]) -> bool:
    truth_guard = _response_truth_guard(response_payload)
    return str(truth_guard.get('status') or '').strip().lower() in {
        'clarification_required',
        'repair_required',
    }


def _slot_is_root_text_output(slot: Mapping[str, Any]) -> bool:
    slot_type = str(slot.get('type') or '').strip().lower()
    if slot_type not in {'text', 'document'}:
        return False
    if str(slot.get('parent_slot_id') or '').strip():
        return False
    if str(slot.get('follow_up_capability') or '').strip():
        return False
    branch_id = str(slot.get('branch_id') or '').strip()
    phase_id = str(slot.get('phase_id') or '').strip()
    slot_id = str(slot.get('slot_id') or '').strip()
    if branch_id.startswith(('branch-', 'repair-')):
        return False
    if phase_id and phase_id not in {'phase-1', 'current'} and not phase_id.endswith('-1'):
        return False
    if slot_id and slot_id.startswith('output-phase-') and slot_id not in {'output-phase-1'}:
        return False
    return True


def _slot_is_deferred_materialization(slot: Mapping[str, Any]) -> bool:
    if str(slot.get('parent_slot_id') or '').strip():
        return True
    if str(slot.get('follow_up_capability') or '').strip():
        return True
    branch_id = str(slot.get('branch_id') or '').strip()
    phase_id = str(slot.get('phase_id') or '').strip()
    return branch_id.startswith(('branch-', 'repair-')) or phase_id.startswith(('branch-', 'repair-'))


def _apply_truth_guard_to_output_slots(
    response_payload: Mapping[str, Any],
    output_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not output_slots or not _response_truth_guard_requires_clarification(response_payload):
        return output_slots
    truth_guard = _response_truth_guard(response_payload)
    truth_guard_status = str(truth_guard.get('status') or '').strip().lower()
    reason = (
        str(truth_guard.get('reason') or '').strip()
        or 'current turn needs a concrete source before materialization'
    )
    clarification = str(response_payload.get('output_text') or '').strip()
    guarded_slots: list[dict[str, Any]] = []
    for raw_slot in output_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot = dict(raw_slot)
        if _slot_is_root_text_output(slot):
            if clarification:
                slot['value'] = clarification
                slot['content_payload'] = clarification
                slot['output_text'] = clarification
            slot['status'] = 'fulfilled'
            slot['lifecycle'] = 'materialized_output'
        elif _slot_is_deferred_materialization(slot) and not (
            str(slot.get('artifact_ref') or '').strip()
            or str(slot.get('artifact_path') or '').strip()
            or str(slot.get('saved_text_path') or '').strip()
            or str(slot.get('saved_audio_path') or '').strip()
            or str(slot.get('saved_image_path') or '').strip()
        ):
            slot['status'] = 'blocked'
            slot['lifecycle'] = 'blocked_output'
            slot['blocked_reason'] = reason
            slot['error_ref'] = {
                'branch_id': str(slot.get('branch_id') or slot.get('phase_id') or '').strip() or None,
                'code': (
                    'UPSTREAM_REPAIR_REQUIRED'
                    if truth_guard_status == 'repair_required'
                    else 'UPSTREAM_CLARIFICATION_REQUIRED'
                ),
                'stage': 'truth_guard',
            }
            slot['recovery_context'] = {
                'can_retry': False,
                'retry_scope': 'branch_contract',
                'suggested_action': 'repair_branch_contract',
                'preserve_intent': True,
            }
            slot.pop('placeholder_ref', None)
        guarded_slots.append(
            {
                key: value
                for key, value in slot.items()
                if value not in (None, '', [], {})
            }
        )
    return _json_safe(guarded_slots)


_ARTIFACT_BINDING_KEYS = (
    'branch_id',
    'phase_id',
    'slot_id',
    'obligation_id',
    'output_type',
)
_ARTIFACT_IDENTITY_KEYS = ('artifact_id', 'artifact_ref', 'ref')
_ARTIFACT_PATH_KEYS = (
    'path',
    'source_path',
    'artifact_path',
    'saved_text_path',
    'saved_audio_path',
    'saved_image_path',
    'saved_video_path',
)
_LATE_FILL_SAVED_PATH_KEYS = (
    ('saved_image_path', 'image'),
    ('saved_audio_path', 'audio'),
    ('saved_text_path', 'text'),
    ('saved_video_path', 'video'),
    ('path', ''),
    ('artifact_path', ''),
)
_NON_REBINDABLE_SLOT_STATUSES = {
    'blocked',
    'cancelled',
    'failed',
    'skipped',
    'superseded',
    'waived',
}


def _projection_token(value: Any) -> str:
    return str(value or '').strip()


def _artifact_projection_path(record: Mapping[str, Any]) -> str:
    for key in _ARTIFACT_PATH_KEYS:
        token = _projection_token(record.get(key))
        if token:
            return token
    return ''


def _artifact_projection_type(record: Mapping[str, Any]) -> str:
    token = _projection_token(
        record.get('type')
        or record.get('kind')
        or record.get('output_type')
    ).lower()
    if token == 'document':
        return 'text'
    return token


def _artifact_type_aliases(value: Any) -> set[str]:
    token = _projection_token(value).lower()
    if not token:
        return set()
    if token in {'text', 'document'}:
        return {'text', 'document'}
    if token in {'image', 'png', 'jpg', 'jpeg', 'webp'}:
        return {'image', 'png', 'jpg', 'jpeg', 'webp'}
    if token in {'audio', 'wav', 'mp3', 'm4a'}:
        return {'audio', 'wav', 'mp3', 'm4a'}
    return {token}


def _artifact_types_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return True
    return bool(_artifact_type_aliases(left).intersection(_artifact_type_aliases(right)))


def _artifact_binding_tokens(record: Mapping[str, Any]) -> set[str]:
    tokens = {
        _projection_token(record.get('branch_id')),
        _projection_token(record.get('phase_id')),
        _projection_token(record.get('slot_id')),
        _projection_token(record.get('obligation_id')),
    }
    tokens.discard('')
    return tokens


def _late_fill_result_type_and_path(result: Mapping[str, Any]) -> tuple[str, str]:
    declared_type = _artifact_projection_type(result)
    for key, fallback_type in _LATE_FILL_SAVED_PATH_KEYS:
        path = _projection_token(result.get(key))
        if path:
            return declared_type or fallback_type, path
    return declared_type, ''


def _iter_late_fill_result_records(response_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}
    records: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ('fill_results', 'completed_branches'):
        value = late_fill.get(key)
        if not isinstance(value, list):
            continue
        for raw_result in value:
            if not isinstance(raw_result, Mapping):
                continue
            _result_type, result_path = _late_fill_result_type_and_path(raw_result)
            signature = (
                _projection_token(raw_result.get('branch_id')),
                _projection_token(raw_result.get('phase_id')),
                result_path,
            )
            if signature in seen:
                continue
            seen.add(signature)
            records.append(raw_result)
    return records


def _conflicting_artifact_refs_by_path(artifacts: Any) -> set[str]:
    paths_by_ref: dict[str, set[str]] = {}
    if not isinstance(artifacts, list):
        return set()
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_ref = _artifact_ref_from_projection(raw_artifact)
        artifact_path = _artifact_projection_path(raw_artifact)
        if not artifact_ref or not artifact_path:
            continue
        paths_by_ref.setdefault(artifact_ref, set()).add(artifact_path)
    return {
        artifact_ref
        for artifact_ref, paths in paths_by_ref.items()
        if len(paths) > 1
    }


def _canonical_artifacts_by_late_fill_path(
    response_payload: Mapping[str, Any],
    canonical_artifacts: Any,
) -> dict[str, Mapping[str, Any]]:
    late_fill_paths = {
        path
        for _result_type, path in (
            _late_fill_result_type_and_path(result)
            for result in _iter_late_fill_result_records(response_payload)
        )
        if path
    }
    if not late_fill_paths or not isinstance(canonical_artifacts, list):
        return {}
    artifacts_by_path: dict[str, Mapping[str, Any]] = {}
    for raw_artifact in canonical_artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_path = _artifact_projection_path(raw_artifact)
        if artifact_path and artifact_path in late_fill_paths:
            artifacts_by_path.setdefault(artifact_path, raw_artifact)
    return artifacts_by_path


def _reconcile_output_artifacts_with_late_fill_branch_paths(
    response_payload: Mapping[str, Any],
    output_artifacts: Any,
    canonical_artifacts: Any,
) -> list[dict[str, Any]]:
    if not isinstance(output_artifacts, list) or not output_artifacts:
        return _json_safe(output_artifacts) if isinstance(output_artifacts, list) else []
    canonical_by_path = _canonical_artifacts_by_late_fill_path(response_payload, canonical_artifacts)
    if not canonical_by_path:
        return _json_safe(output_artifacts)
    conflicting_refs = _conflicting_artifact_refs_by_path(output_artifacts)
    reconciled: list[dict[str, Any]] = []
    for raw_artifact in output_artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact = dict(raw_artifact)
        artifact_path = _artifact_projection_path(artifact)
        canonical = canonical_by_path.get(artifact_path)
        if canonical and _artifact_types_compatible(
            _artifact_projection_type(artifact),
            _artifact_projection_type(canonical),
        ):
            current_ref = _artifact_ref_from_projection(artifact)
            canonical_ref = _artifact_ref_from_projection(canonical)
            for key in _ARTIFACT_BINDING_KEYS:
                value = canonical.get(key)
                if value not in (None, '', [], {}) and artifact.get(key) in (None, '', [], {}):
                    artifact[key] = value
            if current_ref in conflicting_refs and canonical_ref and canonical_ref != current_ref:
                for key in _ARTIFACT_IDENTITY_KEYS:
                    value = canonical.get(key)
                    if value not in (None, '', [], {}):
                        artifact[key] = value
        normalized = {
            key: value
            for key, value in artifact.items()
            if value not in (None, '', [], {})
        }
        if normalized:
            reconciled.append(normalized)
    return _json_safe(reconciled)


def _late_fill_artifacts_by_binding(
    response_payload: Mapping[str, Any],
    output_artifacts: Any,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not isinstance(output_artifacts, list):
        return {}
    artifacts_by_path: dict[str, Mapping[str, Any]] = {}
    artifacts_by_binding: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_artifact in output_artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_path = _artifact_projection_path(raw_artifact)
        if artifact_path:
            artifacts_by_path.setdefault(artifact_path, raw_artifact)
        artifact_type = _artifact_projection_type(raw_artifact)
        for token in _artifact_binding_tokens(raw_artifact):
            for alias in _artifact_type_aliases(artifact_type):
                artifacts_by_binding.setdefault((token, alias), raw_artifact)

    for result in _iter_late_fill_result_records(response_payload):
        result_type, result_path = _late_fill_result_type_and_path(result)
        if not result_path:
            continue
        artifact = artifacts_by_path.get(result_path)
        if not artifact:
            continue
        artifact_type = _artifact_projection_type(artifact)
        if not _artifact_types_compatible(result_type, artifact_type):
            continue
        aliases = _artifact_type_aliases(result_type or artifact_type) or {artifact_type}
        for token in _artifact_binding_tokens(result):
            for alias in aliases:
                artifacts_by_binding.setdefault((token, alias), artifact)
    return artifacts_by_binding


def _reconcile_output_slots_with_late_fill_artifacts(
    response_payload: Mapping[str, Any],
    output_slots: list[dict[str, Any]],
    output_artifacts: Any,
) -> list[dict[str, Any]]:
    if not output_slots:
        return output_slots
    artifacts_by_binding = _late_fill_artifacts_by_binding(response_payload, output_artifacts)
    if not artifacts_by_binding:
        return output_slots

    # An explicitly bound canonical/late-fill artifact has one owner (or a
    # bounded set of equivalent owner tokens). Persisted payload slots can
    # predate that binding and still carry the same ref/path on an earlier branch.
    # Aggregate the ownership evidence by artifact identity so reconciliation can
    # remove only those proven stale claims while leaving genuinely unbound
    # legacy refs untouched.
    ownership_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for (owner_token, artifact_type), artifact in artifacts_by_binding.items():
        artifact_ref = _artifact_ref_from_projection(artifact)
        artifact_path = _artifact_projection_path(artifact)
        if not artifact_ref and not artifact_path:
            continue
        identity = ('ref', artifact_ref) if artifact_ref else ('path', artifact_path)
        ownership = ownership_by_identity.setdefault(
            identity,
            {
                'artifact_ref': artifact_ref,
                'artifact_paths': set(),
                'artifact_types': set(),
                'owner_tokens': set(),
            },
        )
        if artifact_path:
            ownership['artifact_paths'].add(artifact_path)
        if artifact_type:
            ownership['artifact_types'].add(artifact_type)
        ownership['owner_tokens'].add(owner_token)

    reconciled: list[dict[str, Any]] = []
    for raw_slot in output_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot = dict(raw_slot)
        slot_type = _artifact_projection_type(slot)
        slot_types = _artifact_type_aliases(slot_type)
        slot_tokens = _artifact_binding_tokens(slot)

        for ownership in ownership_by_identity.values():
            owner_types = ownership['artifact_types']
            if slot_types and owner_types and not slot_types.intersection(owner_types):
                continue
            if slot_tokens.intersection(ownership['owner_tokens']):
                continue

            slot_ref = _artifact_ref_from_projection(slot)
            slot_path = _artifact_projection_path(slot)
            ref_matches = bool(ownership['artifact_ref'] and slot_ref == ownership['artifact_ref'])
            path_matches = bool(slot_path and slot_path in ownership['artifact_paths'])
            if not ref_matches and not path_matches:
                continue

            # A path match proves that any inline identity on the slot is the
            # same artifact, even when an older alias ref was persisted there.
            # A ref match likewise invalidates all direct identity aliases.
            for key in _ARTIFACT_IDENTITY_KEYS:
                slot.pop(key, None)
            for key in _ARTIFACT_PATH_KEYS:
                if _projection_token(slot.get(key)) in ownership['artifact_paths']:
                    slot.pop(key, None)

        matched_artifact = None
        for token in slot_tokens:
            for alias in slot_types:
                matched_artifact = artifacts_by_binding.get((token, alias))
                if matched_artifact:
                    break
            if matched_artifact:
                break
        if matched_artifact:
            artifact_ref = _artifact_ref_from_projection(matched_artifact)
            artifact_path = _artifact_projection_path(matched_artifact)
            if artifact_ref:
                slot['artifact_ref'] = artifact_ref
            if artifact_path:
                slot['artifact_path'] = artifact_path
            status = _projection_token(slot.get('status')).lower()
            if status not in _NON_REBINDABLE_SLOT_STATUSES:
                slot['status'] = 'fulfilled'
                slot['lifecycle'] = 'materialized_output'
                slot.pop('blocked_reason', None)
            if _projection_token(slot.get('status')).lower() in _TERMINAL_OUTPUT_STATUSES:
                slot.pop('placeholder_ref', None)
        normalized = {
            key: value
            for key, value in slot.items()
            if value not in (None, '', [], {})
        }
        if normalized:
            reconciled.append(normalized)
    return _json_safe(reconciled)


def _output_slot_truth_score(slots: list[dict[str, Any]]) -> int:
    score = 0
    for raw_slot in slots:
        if not isinstance(raw_slot, Mapping):
            continue
        status = str(raw_slot.get('status') or '').strip().lower()
        score += _OUTPUT_SLOT_STATUS_SCORES.get(status, 0)
        if str(raw_slot.get('artifact_ref') or '').strip():
            score += 4
        if any(
            str(raw_slot.get(key) or '').strip()
            for key in ('artifact_path', 'path', 'saved_image_path', 'saved_audio_path', 'saved_text_path')
        ):
            score += 4
        if str(raw_slot.get('placeholder_ref') or '').strip():
            score -= 1
    return score


def _output_slots_should_refresh_from_plan(
    payload_slots: list[dict[str, Any]],
    plan_slots: list[dict[str, Any]],
) -> bool:
    if not payload_slots or not plan_slots:
        return False
    plan_score = _output_slot_truth_score(plan_slots)
    payload_score = _output_slot_truth_score(payload_slots)
    if plan_score <= payload_score:
        return False
    payload_open = any(
        str(slot.get('status') or '').strip().lower() in {'pending', 'queued', 'running', 'scheduled'}
        for slot in payload_slots
        if isinstance(slot, Mapping)
    )
    plan_materialized = any(
        str(slot.get('status') or '').strip().lower() in {'fulfilled', 'completed'}
        and any(
            str(slot.get(key) or '').strip()
            for key in ('artifact_ref', 'artifact_path', 'path', 'saved_image_path', 'saved_audio_path', 'saved_text_path')
        )
        for slot in plan_slots
        if isinstance(slot, Mapping)
    )
    return payload_open and plan_materialized


def _project_output_items_text(output_items: Any, output_text: str) -> list[dict[str, Any]]:
    text = str(output_text or '').strip()
    if not text:
        return _json_safe(output_items) if isinstance(output_items, list) else []
    items = _json_safe(output_items) if isinstance(output_items, list) else []
    items = [dict(item) for item in items if isinstance(item, Mapping)]
    for item in items:
        content = item.get('content')
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if str(content_item.get('type') or '').strip() == 'output_text':
                content_item['text'] = text
                return _json_safe(items)
    if items:
        items[0]['content'] = [{'type': 'output_text', 'text': text, 'annotations': []}]
        return _json_safe(items)
    return [
        {
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
        }
    ]


def _reconcile_output_branches_with_slots(
    output_branches: Any,
    output_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not output_slots:
        return _json_safe(output_branches) if isinstance(output_branches, list) else []
    slot_lookup: dict[str, Mapping[str, Any]] = {}
    for slot in output_slots:
        if not isinstance(slot, Mapping):
            continue
        for key in ('slot_id', 'branch_id', 'phase_id'):
            token = str(slot.get(key) or '').strip()
            if token:
                slot_lookup[token] = slot

    raw_branches = (
        output_branches
        if isinstance(output_branches, list) and output_branches
        else build_public_output_branches_from_slots(output_slots)
    )
    reconciled: list[dict[str, Any]] = []
    for raw_branch in raw_branches:
        if not isinstance(raw_branch, Mapping):
            continue
        branch = dict(raw_branch)
        matching_slot = None
        for key in ('slot_id', 'branch_id', 'phase_id'):
            token = str(branch.get(key) or '').strip()
            if token and token in slot_lookup:
                matching_slot = slot_lookup[token]
                break
        if matching_slot is not None:
            for key in (
                'slot_id',
                'branch_id',
                'phase_id',
                'obligation_id',
                'type',
                'status',
                'lifecycle',
                'follow_up_capability',
                'artifact_ref',
                'artifact_path',
                'blocked_reason',
                'parent_slot_id',
            ):
                value = matching_slot.get(key)
                if value not in (None, '', [], {}):
                    branch[key] = value
                elif key in {'artifact_ref', 'blocked_reason'}:
                    branch.pop(key, None)
            if isinstance(matching_slot.get('child_slot_ids'), list):
                branch['child_slot_ids'] = list(matching_slot.get('child_slot_ids') or [])
            if str(matching_slot.get('status') or '').strip().lower() in _TERMINAL_OUTPUT_STATUSES:
                branch.pop('placeholder_ref', None)
            elif str(matching_slot.get('placeholder_ref') or '').strip():
                branch['placeholder_ref'] = str(matching_slot.get('placeholder_ref') or '').strip()
        normalized = {
            key: value
            for key, value in branch.items()
            if value not in (None, '', [], {})
        }
        if normalized:
            reconciled.append(normalized)
    return _json_safe(reconciled)


def _build_current_state_frame(
    response_payload: Mapping[str, Any],
    *,
    output_slots: list[dict[str, Any]],
    output_branches: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    work_tree: Mapping[str, Any],
    public_output_text: Optional[str] = None,
) -> dict[str, Any]:
    payload = _select_frame_keys(response_payload, _CURRENT_STATE_FRAME_KEYS)
    payload['id'] = str(response_payload.get('id') or response_payload.get('response_id') or '').strip() or None
    payload['status'] = str(response_payload.get('status') or '').strip() or None
    if public_output_text:
        payload['output_text'] = public_output_text
        payload['output'] = _project_output_items_text(response_payload.get('output'), public_output_text)
    if output_slots:
        payload['output_slots'] = _json_safe(output_slots)
    if output_branches:
        payload['output_branches'] = _json_safe(output_branches)
    if outputs:
        payload['outputs'] = _json_safe(outputs)
    if isinstance(response_payload.get('artifacts'), list):
        public_artifacts = filter_public_response_artifacts(
            response_payload,
            response_payload.get('artifacts'),
            outputs=outputs,
        )
        if public_artifacts:
            payload['artifacts'] = _json_safe(public_artifacts)
        else:
            payload.pop('artifacts', None)
    if work_tree:
        payload['work_tree'] = _json_safe(work_tree)
    if isinstance(response_payload.get('late_fill'), Mapping):
        payload['late_fill'] = _json_safe(response_payload.get('late_fill'))
    if isinstance(response_payload.get('runtime'), Mapping):
        payload['runtime'] = _build_runtime_frame(response_payload)
    return {
        key: value
        for key, value in _json_safe(payload).items()
        if value not in (None, '', [], {})
    }


def _frame_digest(response_frame: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(response_frame), ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:16]


def _ledger_path(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> Path:
    return Path(frames_dir) / (str(ledger_name or '').strip() or DEFAULT_RESPONSE_FRAME_LEDGER)


def _index_path(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
) -> Path:
    return Path(frames_dir) / (str(index_name or '').strip() or DEFAULT_RESPONSE_FRAME_INDEX)


def _frame_response_id(response_frame: Mapping[str, Any]) -> str:
    return str(response_frame.get('response_id') or response_frame.get('id') or '').strip()


def response_frame_ledger_record_response_id(frame: Mapping[str, Any]) -> str:
    """Return the response identifier carried by one ledger record."""

    current_state = (
        frame.get('current_state')
        if isinstance(frame.get('current_state'), Mapping)
        else {}
    )
    return str(
        frame.get('response_id')
        or current_state.get('response_id')
        or current_state.get('id')
        or frame.get('id')
        or ''
    ).strip()


def _safe_path_token(value: Any) -> str:
    token = ''.join(
        char if char.isalnum() or char in {'-', '_', '.'} else '_'
        for char in str(value or '').strip()
    ).strip('._')
    return token or 'snapshot'


def _json_size_bytes(value: Any) -> int:
    return len(json.dumps(_json_safe(value), ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def _snapshot_path_allows_timestamp_normalization(json_path: str) -> bool:
    normalized = str(json_path or '').strip().lower()
    if normalized.endswith('.__volatile_timestamps__'):
        return False
    path_tokens = {token for token in normalized.replace('[', '.').replace(']', '.').split('.') if token}
    return bool(path_tokens & _TIMESTAMP_NORMALIZED_SNAPSHOT_PATH_TOKENS)


def _media_kind_for_payload_key(key: str, value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) < _MEDIA_PAYLOAD_MIN_CHARS and not text.startswith('data:'):
        return None
    return _RAW_MEDIA_PAYLOAD_KEYS.get(str(key or '').strip().lower())


def _decode_snapshot_base64_payload(value: str) -> tuple[bytes | None, str | None]:
    text = str(value or '').strip()
    if not text:
        return None, None
    encoding = 'base64'
    if text.startswith('data:') and ',' in text:
        header, text = text.split(',', 1)
        encoding = header
    compact = ''.join(text.split())
    if not compact:
        return None, None
    padding = (-len(compact)) % 4
    if padding:
        compact += '=' * padding
    try:
        return base64.b64decode(compact.encode('ascii'), validate=True), encoding
    except (binascii.Error, UnicodeEncodeError, ValueError):
        return None, None


def _snapshot_candidate_roots(frames_dir: Path) -> list[Path]:
    roots = [Path.cwd()]
    try:
        resolved_frames_dir = frames_dir.resolve()
    except OSError:
        resolved_frames_dir = frames_dir
    if resolved_frames_dir.name == DEFAULT_RESPONSE_FRAME_SNAPSHOT_DIR:
        roots.append(resolved_frames_dir.parent.parent)
    if resolved_frames_dir.name == DEFAULT_RESPONSE_FRAMES_DIR.name:
        roots.append(resolved_frames_dir.parent.parent)
    if resolved_frames_dir.parent.name == 'state':
        roots.append(resolved_frames_dir.parent.parent)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        token = str(root)
        if token and token not in seen:
            unique.append(root)
            seen.add(token)
    return unique


def _resolve_snapshot_media_file_path(raw_path: Any, *, frames_dir: Path) -> Path | None:
    path_text = str(raw_path or '').strip()
    if not path_text:
        return None
    path = Path(path_text)
    candidates = [path] if path.is_absolute() else [root / path for root in _snapshot_candidate_roots(frames_dir)]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _file_sha256(path: Path) -> tuple[str | None, int | None]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open('rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError:
        return None, None
    return digest.hexdigest(), size


def _media_kind_hint_from_mapping(value: Mapping[str, Any]) -> str | None:
    for key in ('media_type', 'mime_type', 'output_type', 'type', 'capability'):
        text = str(value.get(key) or '').strip().lower()
        if not text:
            continue
        if 'image' in text:
            return 'image'
        if 'audio' in text or 'speech' in text:
            return 'audio'
        if 'video' in text:
            return 'video'
    return None


def _media_evidence_from_mapping(
    value: Mapping[str, Any],
    *,
    inherited: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    evidence = {
        str(kind): dict(payload)
        for kind, payload in inherited.items()
        if isinstance(payload, Mapping)
    }
    artifact_ref = str(
        value.get('artifact_ref')
        or value.get('ref')
        or value.get('artifact_id')
        or ''
    ).strip()
    for kind, keys in _MEDIA_SAVED_PATH_KEYS.items():
        for key in keys:
            path = str(value.get(key) or '').strip()
            if path:
                evidence[kind] = {
                    'artifact_ref': artifact_ref or evidence.get(kind, {}).get('artifact_ref'),
                    'path': path,
                    'path_key': key,
                }
                break
    kind_hint = _media_kind_hint_from_mapping(value)
    if kind_hint and kind_hint not in evidence:
        for key in _GENERIC_ARTIFACT_PATH_KEYS:
            path = str(value.get(key) or '').strip()
            if path:
                evidence[kind_hint] = {
                    'artifact_ref': artifact_ref or None,
                    'path': path,
                    'path_key': key,
                }
                break
    return evidence


def _externalized_media_payload_ref(
    value: str,
    *,
    key: str,
    media_kind: str,
    evidence: Mapping[str, Any] | None,
    frames_dir: Path,
) -> dict[str, Any] | None:
    if not isinstance(evidence, Mapping):
        return None
    raw_path = str(evidence.get('path') or '').strip()
    artifact_path = _resolve_snapshot_media_file_path(raw_path, frames_dir=frames_dir)
    if artifact_path is None:
        return None
    decoded, encoding = _decode_snapshot_base64_payload(value)
    if decoded is None:
        return None
    decoded_sha = hashlib.sha256(decoded).hexdigest()
    artifact_sha, artifact_size = _file_sha256(artifact_path)
    if not artifact_sha or decoded_sha != artifact_sha:
        return None
    ref: dict[str, Any] = {
        'kind': 'ollmo.snapshot_externalized_media_payload',
        'media_kind': media_kind,
        'media_key': str(key),
        'source_path': raw_path,
        'resolved_path': str(artifact_path),
        'path_key': evidence.get('path_key'),
        'sha256': artifact_sha,
        'size_bytes': artifact_size,
        'raw_payload_encoding': encoding,
        'raw_payload_character_count': len(str(value or '')),
        'raw_payload_externalized': True,
        'truth_preservation': 'artifact_file_sha256_matches_raw_payload',
    }
    artifact_ref = str(evidence.get('artifact_ref') or '').strip()
    if artifact_ref:
        ref['artifact_ref'] = artifact_ref
    return {
        key: _json_safe(payload)
        for key, payload in ref.items()
        if payload not in (None, '', [], {})
    }


def _stripped_raw_media_payload_ref(
    value: str,
    *,
    key: str,
    media_kind: str,
) -> dict[str, Any]:
    decoded, encoding = _decode_snapshot_base64_payload(value)
    if decoded is not None:
        digest = hashlib.sha256(decoded).hexdigest()
        decoded_size = len(decoded)
    else:
        raw_bytes = str(value or '').encode('utf-8', errors='replace')
        digest = hashlib.sha256(raw_bytes).hexdigest()
        decoded_size = None
        encoding = 'raw_text'
    ref: dict[str, Any] = {
        'kind': 'ollmo.snapshot_stripped_raw_media_payload',
        'media_kind': media_kind,
        'media_key': str(key),
        'sha256': digest,
        'decoded_size_bytes': decoded_size,
        'raw_payload_encoding': encoding,
        'raw_payload_character_count': len(str(value or '')),
        'raw_payload_externalized': True,
        'truth_preservation': 'raw_media_digest_without_saved_artifact_truth',
    }
    return {
        key: _json_safe(payload)
        for key, payload in ref.items()
        if payload not in (None, '', [], {})
    }


def _normalize_snapshot_media_payloads(
    value: Any,
    *,
    frames_dir: Path,
    inherited_media: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    inherited = inherited_media or {}
    if isinstance(value, Mapping):
        current_media = _media_evidence_from_mapping(value, inherited=inherited)
        payload: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            media_kind = _media_kind_for_payload_key(key, child)
            if media_kind:
                media_ref = _externalized_media_payload_ref(
                    child,
                    key=key,
                    media_kind=media_kind,
                    evidence=current_media.get(media_kind),
                    frames_dir=frames_dir,
                )
                if media_ref:
                    payload[key] = media_ref
                    continue
                payload[key] = _stripped_raw_media_payload_ref(
                    child,
                    key=key,
                    media_kind=media_kind,
                )
                continue
            normalized_child = _normalize_snapshot_media_payloads(
                child,
                frames_dir=frames_dir,
                inherited_media=current_media,
            )
            if not _is_empty(normalized_child):
                payload[key] = normalized_child
        return payload
    if isinstance(value, list):
        return [
            normalized_child
            for normalized_child in (
                _normalize_snapshot_media_payloads(
                    child,
                    frames_dir=frames_dir,
                    inherited_media=inherited,
                )
                for child in value
            )
            if not _is_empty(normalized_child)
        ]
    return _json_safe(value)


def _normalize_snapshot_content(value: Any, *, json_path: str) -> tuple[Any, dict[str, Any]]:
    safe_value = _json_safe(value)
    allow_timestamp_normalization = _snapshot_path_allows_timestamp_normalization(json_path)

    removed_keys: set[str] = set()
    removed_path_count = 0

    def normalize(item: Any) -> Any:
        nonlocal removed_path_count
        if isinstance(item, dict):
            payload: dict[str, Any] = {}
            is_snapshot_ref = str(item.get('kind') or '').strip() == _LEDGER_SNAPSHOT_REF_KIND
            for key, child in item.items():
                key_text = str(key)
                if (allow_timestamp_normalization and key_text in _SNAPSHOT_VOLATILE_TIMESTAMP_KEYS) or (
                    is_snapshot_ref and key_text in _SNAPSHOT_REF_VOLATILE_KEYS
                ):
                    removed_keys.add(key_text)
                    removed_path_count += 1
                    continue
                normalized_child = normalize(child)
                if not _is_empty(normalized_child):
                    payload[key_text] = normalized_child
            return payload
        if isinstance(item, list):
            return [
                normalized_child
                for normalized_child in (normalize(child) for child in item)
                if not _is_empty(normalized_child)
            ]
        return item

    normalized_value = normalize(safe_value)
    if not removed_keys:
        return normalized_value, {}
    return normalized_value, {
        'kind': 'ollmo.snapshot_content_normalization',
        'strategy': 'volatile_timestamp_keys_excluded_from_content_hash',
        'volatile_timestamp_keys': sorted(removed_keys),
        'volatile_timestamp_field_count': removed_path_count,
    }


def _json_child_path(base_json_path: str, child: str) -> str:
    base = str(base_json_path or '').strip()
    token = str(child or '').strip()
    if not base:
        return token
    if token.startswith('['):
        return f'{base}{token}'
    return f'{base}.{token}' if token else base


def _merge_snapshot_normalization(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left_payload = dict(left or {}) if isinstance(left, Mapping) else {}
    right_payload = dict(right or {}) if isinstance(right, Mapping) else {}
    if not left_payload:
        return _json_safe(right_payload)
    if not right_payload:
        return _json_safe(left_payload)
    keys = sorted(
        set(left_payload.get('volatile_timestamp_keys') or [])
        | set(right_payload.get('volatile_timestamp_keys') or [])
    )
    count = int(left_payload.get('volatile_timestamp_field_count') or 0) + int(
        right_payload.get('volatile_timestamp_field_count') or 0
    )
    return {
        'kind': 'ollmo.snapshot_content_normalization',
        'strategy': 'volatile_timestamp_keys_excluded_from_content_hash',
        'volatile_timestamp_keys': keys,
        'volatile_timestamp_field_count': count,
    }


def _sidecar_key_matches_advisory_context(key: str, json_path: str) -> bool:
    normalized = str(key or '').strip().lower()
    normalized_path = str(json_path or '').strip().lower()
    if not normalized or not normalized_path:
        return False
    return any(
        normalized_path.endswith(pattern)
        for pattern in _ADVISORY_CONTEXT_SIDECAR_SPLIT_PATHS.get(normalized, ())
    )


def _sidecar_split_key_is_worthwhile(key: str, *, json_path: str = '') -> bool:
    normalized = str(key or '').strip().lower()
    if not normalized or normalized.endswith('_snapshot_ref'):
        return False
    if normalized in {'external_snapshots', 'snapshot_policy'}:
        return False
    return (
        normalized in _SIDECAR_SPLIT_KEYS
        or normalized in _ADVISORY_SIDECAR_SPLIT_KEYS
        or _sidecar_key_matches_advisory_context(normalized, json_path)
        or any(
            normalized.endswith(suffix)
            for suffix in _SIDECAR_SPLIT_SUFFIXES
        )
    )


def _sidecar_split_limit_for_key(key: str, *, json_path: str = '') -> int:
    normalized = str(key or '').strip().lower()
    if normalized in _ADVISORY_SIDECAR_SPLIT_KEYS or _sidecar_key_matches_advisory_context(normalized, json_path):
        return _ADVISORY_SIDECAR_SPLIT_LIMIT_BYTES
    return _SIDECAR_SPLIT_LIMIT_BYTES


def _should_split_sidecar_child(key: str, value: Any, *, depth: int, json_path: str) -> bool:
    if depth >= _SIDECAR_SPLIT_MAX_DEPTH:
        return False
    if not isinstance(value, (dict, list)):
        return False
    if isinstance(value, Mapping) and str(value.get('kind') or '').strip() == _LEDGER_SNAPSHOT_REF_KIND:
        return False
    parent_json_path = str(json_path or '').rsplit('.', 1)[0]
    if (
        parent_json_path.endswith('semantic_review_lens_contract')
        and str(key or '').strip().lower()
        in {'evidence_requirements', 'failure_modes', 'focus_questions'}
    ):
        # The complete lens contract is already a highly reusable CAS unit.
        # Splitting its three small lists again only fragments authority and can
        # exhaust the bounded ref budget before peer advisory contracts split.
        return False
    if not _sidecar_split_key_is_worthwhile(key, json_path=json_path):
        return False
    return _json_size_bytes(value) >= _sidecar_split_limit_for_key(key, json_path=json_path)


def _split_sidecar_payload_for_cas(
    value: Any,
    *,
    frame: Mapping[str, Any],
    frames_dir: Path,
    json_path: str,
    depth: int,
    split_counter: list[int],
) -> tuple[Any, list[dict[str, Any]]]:
    if depth >= _SIDECAR_SPLIT_MAX_DEPTH:
        return _json_safe(value), []
    if isinstance(value, dict):
        payload: dict[str, Any] = {}
        child_refs: list[dict[str, Any]] = []
        entries = [(str(raw_key), child) for raw_key, child in value.items()]
        remaining_split_slots = max(
            0,
            _SIDECAR_SPLIT_MAX_REFS - split_counter[0],
        )
        reserved_split_keys: set[str] = set()
        for key, child in entries:
            if len(reserved_split_keys) >= remaining_split_slots:
                break
            child_json_path = _json_child_path(json_path, key)
            if _should_split_sidecar_child(
                key,
                child,
                depth=depth,
                json_path=child_json_path,
            ):
                reserved_split_keys.add(key)
        # Reserve direct siblings before descending into the first child's
        # subtree.  Otherwise one dense branch can consume the global CAS-ref
        # budget and starve later peer fields that should remain independently
        # addressable.
        split_counter[0] += len(reserved_split_keys)
        for key, child in entries:
            child_json_path = _json_child_path(json_path, key)
            if key in reserved_split_keys:
                ref = _write_snapshot_ref(
                    child,
                    frame=frame,
                    frames_dir=frames_dir,
                    json_path=child_json_path,
                    _sidecar_depth=depth + 1,
                    _sidecar_child_ref=True,
                    _sidecar_child_key=key,
                    _sidecar_parent_json_path=json_path,
                    _sidecar_split_counter=split_counter,
                )
                payload[f'{key}_snapshot_ref'] = _json_safe(ref)
                child_refs.append(
                    {
                        'json_path': child_json_path,
                        'key': key,
                        'path': ref.get('path'),
                        'sha256': ref.get('sha256'),
                        'size_bytes': ref.get('size_bytes'),
                        'content_addressed': ref.get('content_addressed'),
                        **(
                            {
                                'sidecar_manifest': _json_safe(
                                    ref.get('sidecar_manifest')
                                )
                            }
                            if isinstance(ref.get('sidecar_manifest'), Mapping)
                            else {}
                        ),
                    }
                )
                continue
            split_child, split_refs = _split_sidecar_payload_for_cas(
                child,
                frame=frame,
                frames_dir=frames_dir,
                json_path=child_json_path,
                depth=depth + 1,
                split_counter=split_counter,
            )
            if not _is_empty(split_child):
                payload[key] = split_child
            child_refs.extend(split_refs)
        return payload, child_refs
    if isinstance(value, list):
        items: list[Any] = []
        child_refs: list[dict[str, Any]] = []
        for index, child in enumerate(value):
            child_json_path = _json_child_path(json_path, f'[{index}]')
            split_child, split_refs = _split_sidecar_payload_for_cas(
                child,
                frame=frame,
                frames_dir=frames_dir,
                json_path=child_json_path,
                depth=depth + 1,
                split_counter=split_counter,
            )
            if not _is_empty(split_child):
                items.append(split_child)
            child_refs.extend(split_refs)
        return items, child_refs
    return _json_safe(value), []


def _coerce_frame_sequence(value: Any, fallback: int | None = None) -> int | None:
    if value in (None, ''):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _response_map_digest(responses: Mapping[str, Any]) -> str:
    """Return a stable digest binding the complete response-index mapping."""

    canonical = json.dumps(
        _json_safe(dict(responses)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _response_frame_index_targets_ledger(
    index_state: Mapping[str, Any],
    ledger_path: Path,
) -> bool:
    indexed_name = str(index_state.get('ledger_name') or '').strip()
    if indexed_name and indexed_name != ledger_path.name:
        return False
    indexed_path = str(index_state.get('ledger_path') or '').strip()
    if indexed_path:
        try:
            if Path(indexed_path).resolve() != ledger_path.resolve():
                return False
        except OSError:
            return False
    return True


def _response_frame_index_is_fresh(
    index_state: Mapping[str, Any],
    ledger_path: Path,
) -> bool:
    if index_state.get('ok') is not True:
        return False
    if not _response_frame_index_targets_ledger(index_state, ledger_path):
        return False
    indexed_size = _coerce_frame_sequence(index_state.get('ledger_size_bytes'))
    actual_size = ledger_path.stat().st_size if ledger_path.exists() else None
    return indexed_size is not None and actual_size is not None and indexed_size == actual_size


def _response_frame_index_has_verified_response_map(
    index_state: Mapping[str, Any],
    ledger_path: Path,
) -> bool:
    """Return whether index absence is authoritative for the current ledger size."""

    if not _response_frame_index_is_fresh(index_state, ledger_path):
        return False
    responses = index_state.get('responses')
    if not isinstance(responses, Mapping):
        return False
    indexed_size = _coerce_frame_sequence(index_state.get('ledger_size_bytes'))
    verified_size = _coerce_frame_sequence(
        index_state.get('response_map_verified_size_bytes')
    )
    verified_count = _coerce_frame_sequence(index_state.get('response_map_entry_count'))
    verified_digest = str(index_state.get('response_map_digest') or '').strip()
    if (
        verified_size is None
        or verified_size != indexed_size
        or verified_count != len(responses)
        or not verified_digest
    ):
        return False
    return verified_digest == _response_map_digest(responses)


def _index_parent_frame_stub(
    response_id: str,
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_state = load_response_frame_index(frames_dir=frames_dir)
    mutable_index_state = dict(index_state)
    responses = index_state.get('responses') if isinstance(index_state.get('responses'), Mapping) else {}
    entry = responses.get(response_id) if isinstance(responses, Mapping) else None
    if not index_state.get('ok') or not isinstance(entry, Mapping):
        mutable_index_state['_response_index_entry_missing'] = True
        return [], mutable_index_state
    ledger_path = Path(str(entry.get('ledger_path') or '').strip() or _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name))
    expected_path = _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)
    if ledger_path.name != expected_path.name:
        mutable_index_state['_response_index_entry_stale'] = True
        return [], mutable_index_state
    if not _response_frame_index_has_verified_response_map(index_state, expected_path):
        mutable_index_state['_response_index_entry_stale'] = True
        return [], mutable_index_state
    frame_id = str(entry.get('latest_frame_id') or '').strip()
    frame_sequence = entry.get('latest_frame_sequence')
    if not frame_id and frame_sequence in (None, ''):
        mutable_index_state['_response_index_entry_stale'] = True
        return [], mutable_index_state
    relation = entry.get('frame_relation') if isinstance(entry.get('frame_relation'), Mapping) else {}
    return [
        {
            'kind': 'ollmo.response_frame',
            'response_id': response_id,
            'frame_id': frame_id,
            'frame_sequence': frame_sequence,
            'frame_relation': dict(relation),
        }
    ], mutable_index_state


def _index_next_line_offset(index_state: Mapping[str, Any], target: Path) -> Optional[int]:
    actual_ledger_size = target.stat().st_size if target.exists() else 0
    indexed_ledger_size = _coerce_frame_sequence(index_state.get('ledger_size_bytes'))
    verified_line_count_size = _coerce_frame_sequence(
        index_state.get('ledger_line_count_verified_size_bytes')
    )
    for key in ('ledger_line_count', 'line_count'):
        value = _coerce_frame_sequence(index_state.get(key))
        if value is not None:
            if (
                indexed_ledger_size == actual_ledger_size
                and verified_line_count_size == actual_ledger_size
            ):
                return value
            if target.exists():
                try:
                    with target.open('rb') as handle:
                        return sum(1 for _line in handle)
                except OSError:
                    pass
            return value
    responses = index_state.get('responses') if isinstance(index_state.get('responses'), Mapping) else {}
    offsets: list[int] = []
    for entry in responses.values() if isinstance(responses, Mapping) else []:
        if not isinstance(entry, Mapping):
            continue
        ledger_path = str(entry.get('ledger_path') or '').strip()
        ledger_name = str(entry.get('ledger_name') or '').strip()
        if ledger_path and Path(ledger_path).name != target.name:
            continue
        if ledger_name and ledger_name != target.name:
            continue
        offset = _coerce_frame_sequence(entry.get('line_offset'))
        if offset is not None:
            offsets.append(offset)
    if offsets:
        return max(offsets) + 1
    return None


def _iter_ledger_frames(target: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not target.exists():
        return frames, errors
    with target.open('r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        'line': line_number,
                        'code': 'invalid_json',
                        'message': str(exc),
                    }
                )
                continue
            if not isinstance(payload, dict):
                errors.append(
                    {
                        'line': line_number,
                        'code': 'invalid_frame',
                        'message': 'Response frame ledger line is not a JSON object.',
                    }
                )
                continue
            frames.append(payload)
    return frames, errors


def enrich_response_frame_metadata(
    response_frame: Mapping[str, Any],
    *,
    previous_frames: Optional[list[dict[str, Any]]] = None,
    parent_frame: Optional[Mapping[str, Any]] = None,
    force_append_sequence: bool = False,
) -> dict[str, Any]:
    """Return a frame copy with stable identity and relation metadata."""

    if not isinstance(response_frame, Mapping) or not response_frame:
        raise ValueError('response_frame must be a non-empty mapping')
    enriched_frame = dict(response_frame)
    response_id = _frame_response_id(enriched_frame)
    if response_id:
        enriched_frame.setdefault('response_id', response_id)
    prior_frames = [
        frame
        for frame in (previous_frames or [])
        if isinstance(frame, Mapping) and _frame_response_id(frame) == response_id
    ] if response_id else []
    relation = (
        dict(enriched_frame.get('frame_relation'))
        if isinstance(enriched_frame.get('frame_relation'), Mapping)
        else {}
    )
    if force_append_sequence and response_id and prior_frames:
        parent = prior_frames[-1]
        parent_sequence = _coerce_frame_sequence(parent.get('frame_sequence'), len(prior_frames))
        frame_sequence = (parent_sequence or len(prior_frames)) + 1
        enriched_frame['frame_sequence'] = frame_sequence
        enriched_frame['frame_id'] = f'{response_id}:frame-{frame_sequence}'
        parent_frame_id = str(parent.get('frame_id') or '').strip()
        if not parent_frame_id and parent_sequence:
            parent_frame_id = f'{response_id}:frame-{parent_sequence}'
        relation_kind = str(relation.get('kind') or '').strip()
        if not relation_kind or relation_kind == 'initial':
            relation_kind = 'late_fill_successor' if isinstance(enriched_frame.get('late_fill'), Mapping) else 'successor'
        relation = {
            **relation,
            'kind': relation_kind,
            'response_id': response_id,
            'parent_response_id': response_id,
            'parent_frame_id': parent_frame_id or None,
            'parent_frame_sequence': parent_sequence,
        }
        enriched_frame['frame_relation'] = relation
        return _json_safe(enriched_frame)

    relation_parent_sequence = relation.get('parent_frame_sequence')
    parent_sequence = _coerce_frame_sequence(relation_parent_sequence)
    if parent_sequence is None and isinstance(parent_frame, Mapping):
        parent_sequence = _coerce_frame_sequence(parent_frame.get('frame_sequence'))
    if parent_sequence is None and prior_frames:
        parent_sequence = _coerce_frame_sequence(prior_frames[-1].get('frame_sequence'), len(prior_frames))
    frame_sequence = enriched_frame.get('frame_sequence')
    if frame_sequence in (None, ''):
        if response_id:
            if parent_sequence is not None:
                frame_sequence = parent_sequence + 1
            else:
                frame_sequence = len(prior_frames) + 1
            enriched_frame['frame_sequence'] = frame_sequence
    enriched_frame.setdefault(
        'frame_id',
        f'{response_id}:frame-{frame_sequence}' if response_id and frame_sequence else f'frame-{_frame_digest(enriched_frame)}',
    )
    if response_id:
        if not relation:
            if prior_frames:
                parent = prior_frames[-1]
                parent_frame_id = str(parent.get('frame_id') or '').strip()
                if not parent_frame_id:
                    parent_frame_id = f'{response_id}:frame-{len(prior_frames)}'
                relation_kind = 'late_fill_successor' if isinstance(enriched_frame.get('late_fill'), Mapping) else 'successor'
                relation = {
                    'kind': relation_kind,
                    'response_id': response_id,
                    'parent_response_id': response_id,
                    'parent_frame_id': parent_frame_id,
                    'parent_frame_sequence': parent.get('frame_sequence') or len(prior_frames),
                }
            else:
                relation = {
                    'kind': 'initial',
                    'response_id': response_id,
                }
        else:
            relation.setdefault('response_id', response_id)
            if relation.get('kind') != 'initial':
                relation.setdefault('parent_response_id', response_id)
                if isinstance(parent_frame, Mapping):
                    parent_frame_id = str(parent_frame.get('frame_id') or '').strip()
                    if parent_frame_id:
                        relation.setdefault('parent_frame_id', parent_frame_id)
                    parent_frame_sequence = parent_frame.get('frame_sequence')
                    if parent_frame_sequence not in (None, ''):
                        relation.setdefault('parent_frame_sequence', parent_frame_sequence)
        enriched_frame['frame_relation'] = relation
    return _json_safe(enriched_frame)


def enrich_response_frame_for_ledger_append(
    response_frame: Mapping[str, Any],
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> dict[str, Any]:
    """Return metadata for the next durable append to the response-frame ledger."""

    target = _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)
    response_id = _frame_response_id(response_frame)
    existing_frames, index_state = _index_parent_frame_stub(
        response_id,
        frames_dir=frames_dir,
        ledger_name=ledger_name,
    ) if response_id else ([], {})
    if (
        not existing_frames
        and target.exists()
        and (not index_state.get('ok') or index_state.get('_response_index_entry_stale'))
    ):
        existing_frames, _errors = _iter_ledger_frames(target)
    return enrich_response_frame_metadata(
        response_frame,
        previous_frames=existing_frames,
        force_append_sequence=True,
    )


def _write_snapshot_ref(
    value: Any,
    *,
    frame: Mapping[str, Any],
    frames_dir: Path,
    json_path: str,
    _sidecar_depth: int = 0,
    _sidecar_child_ref: bool = False,
    _sidecar_child_key: str | None = None,
    _sidecar_parent_json_path: str | None = None,
    _sidecar_split_counter: Optional[list[int]] = None,
) -> dict[str, Any]:
    response_id = _safe_path_token(_frame_response_id(frame) or 'unknown_response')
    frame_id = _safe_path_token(frame.get('frame_id') or f'frame-{_frame_digest(frame)}')
    snapshot_value = _normalize_snapshot_media_payloads(
        _json_safe(value),
        frames_dir=frames_dir,
    )
    split_counter = (
        _sidecar_split_counter
        if isinstance(_sidecar_split_counter, list)
        else [0]
    )
    safe_value, child_refs = _split_sidecar_payload_for_cas(
        snapshot_value,
        frame=frame,
        frames_dir=frames_dir,
        json_path=json_path,
        depth=_sidecar_depth,
        split_counter=split_counter,
    )
    safe_value, normalization = _normalize_snapshot_content(safe_value, json_path=json_path)
    encoded = json.dumps(safe_value, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    relative_path = (
        Path(DEFAULT_RESPONSE_FRAME_SNAPSHOT_DIR)
        / DEFAULT_RESPONSE_FRAME_SNAPSHOT_CONTENT_DIR
        / digest[:2]
        / f'{digest}.json'
    )
    target = frames_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_content_addressed_snapshot(
        target,
        encoded=encoded,
        expected_sha256=digest,
    )
    ref = {
        'kind': _LEDGER_SNAPSHOT_REF_KIND,
        'json_path': json_path,
        'path': str(relative_path),
        'sha256': digest,
        'size_bytes': len(encoded),
        'content_addressed': True,
        'dedupe_scope': 'response_frame_snapshot_store',
        'source_response_id': response_id,
        'source_frame_id': frame_id,
    }
    if _sidecar_child_ref:
        ref['sidecar_child_ref'] = True
        if _sidecar_child_key:
            ref['sidecar_child_key'] = str(_sidecar_child_key)
        if _sidecar_parent_json_path:
            ref['sidecar_parent_json_path'] = str(_sidecar_parent_json_path)
    if child_refs:
        ref['sidecar_manifest'] = {
            'kind': 'ollmo.response_frame_sidecar_manifest',
            'strategy': 'recursive_content_addressed_child_refs',
            'child_ref_count': len(child_refs),
            'child_refs': _json_safe(child_refs),
            'child_refs_truncated': False,
            'split_ref_limit': _SIDECAR_SPLIT_MAX_REFS,
            'split_limit_bytes': _SIDECAR_SPLIT_LIMIT_BYTES,
            'max_depth': _SIDECAR_SPLIT_MAX_DEPTH,
        }
    if normalization:
        ref['content_normalization'] = _json_safe(normalization)
    return ref


def _snapshot_file_matches(
    target: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> bool:
    try:
        raw = target.read_bytes()
    except OSError:
        return False
    payload = raw[:-1] if raw.endswith(b'\n') else raw
    return (
        len(payload) == expected_size_bytes
        and hashlib.sha256(payload).hexdigest() == expected_sha256
    )


def _atomic_replace_file_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f'.{target.name}.',
        suffix='.tmp',
        dir=str(target.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync. The same-dir
            # atomic replace and file fsync still prevent partial file bodies.
            pass
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _ensure_content_addressed_snapshot(
    target: Path,
    *,
    encoded: bytes,
    expected_sha256: str,
) -> None:
    """Atomically install or repair one content-addressed snapshot body."""

    if _snapshot_file_matches(
        target,
        expected_sha256=expected_sha256,
        expected_size_bytes=len(encoded),
    ):
        return
    _atomic_replace_file_bytes(target, encoded + b'\n')
    if not _snapshot_file_matches(
        target,
        expected_sha256=expected_sha256,
        expected_size_bytes=len(encoded),
    ):
        raise OSError(
            'Content-addressed response-frame snapshot failed post-write verification.'
        )


def _compact_context_strategy_for_ledger(
    context_strategy: Mapping[str, Any],
    *,
    snapshot,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        'kind',
        'mode',
        'reason',
        'context_scope',
        'selected_context_scope',
        'history_policy',
        'fresh_turn_only',
        'candidate_count',
        'promoted_candidate_count',
    ):
        value = context_strategy.get(key)
        if value not in (None, '', [], {}):
            compact[key] = _json_safe(value)
    candidates = context_strategy.get('context_candidates')
    if isinstance(candidates, list) and candidates:
        compact['context_candidate_count'] = len(candidates)
        compact['context_candidates_snapshot_ref'] = snapshot(
            candidates,
            'runtime.context_strategy.context_candidates',
        )
    gate_review = context_strategy.get('context_gate_review')
    if isinstance(gate_review, Mapping) and gate_review:
        compact['context_gate_review_snapshot_ref'] = snapshot(
            gate_review,
            'runtime.context_strategy.context_gate_review',
        )
    compact['full_snapshot_ref'] = snapshot(context_strategy, 'runtime.context_strategy')
    return _json_safe(compact)


def _compact_semantic_role_profile_for_ledger(
    semantic_role_profile: Mapping[str, Any],
    *,
    snapshot,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        'kind',
        'mode',
        'mode_source',
        'semantic_role',
        'semantic_role_id',
        'semantic_role_ids',
        'orientation_source',
        'authority_boundary',
    ):
        value = semantic_role_profile.get(key)
        if value not in (None, '', [], {}):
            compact[key] = _json_safe(value)
    orientation = semantic_role_profile.get('semantic_role_orientation')
    if isinstance(orientation, Mapping):
        review = orientation.get('review') or orientation.get('summary') or orientation.get('stance')
        if review not in (None, '', [], {}):
            compact['semantic_role_orientation_summary'] = _json_safe(review)
    compact['full_snapshot_ref'] = snapshot(semantic_role_profile, 'runtime.semantic_role_profile')
    return _json_safe(compact)


def _compact_runtime_for_ledger(runtime: Mapping[str, Any], *, snapshot) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _COMPACT_RUNTIME_KEYS:
        value = runtime.get(key)
        if value not in (None, '', [], {}):
            if (
                key in _ADVISORY_SIDECAR_SPLIT_KEYS
                and _json_size_bytes(value) >= _sidecar_split_limit_for_key(key, json_path=f'runtime.{key}')
            ):
                compact[f'{key}_snapshot_ref'] = _json_safe(snapshot(value, f'runtime.{key}'))
                if isinstance(value, Mapping):
                    summary = {
                        summary_key: value.get(summary_key)
                        for summary_key in (
                            'kind',
                            'status',
                            'enabled',
                            'authority',
                            'runtime_effect',
                            'hint_count',
                            'accepted_learning_count',
                        )
                        if value.get(summary_key) not in (None, '', [], {})
                    }
                    if summary:
                        compact[f'{key}_summary'] = _json_safe(summary)
                compact[f'{key}_in_snapshot'] = True
                continue
            compact[key] = _json_safe(value)
    context_strategy = runtime.get('context_strategy') if isinstance(runtime.get('context_strategy'), Mapping) else None
    if context_strategy:
        compact['context_strategy'] = _compact_context_strategy_for_ledger(
            context_strategy,
            snapshot=snapshot,
        )
    semantic_role_profile = (
        runtime.get('semantic_role_profile')
        if isinstance(runtime.get('semantic_role_profile'), Mapping)
        else None
    )
    if semantic_role_profile:
        compact['semantic_role_profile'] = _compact_semantic_role_profile_for_ledger(
            semantic_role_profile,
            snapshot=snapshot,
        )
    diagnostics = runtime.get('developer_diagnostics') if isinstance(runtime.get('developer_diagnostics'), Mapping) else {}
    diagnostic_summary = {
        key: diagnostics.get(key)
        for key in ('planner_timeout_ms', 'planner_timeout_sec', 'routing_contract', 'routing_policy')
        if diagnostics.get(key) not in (None, '', [], {})
    }
    if diagnostic_summary:
        compact['developer_diagnostics_summary'] = _json_safe(diagnostic_summary)
    return compact


def _ledger_text_summary(value: Any, *, limit: int = _LEDGER_TEXT_SUMMARY_LIMIT) -> dict[str, Any]:
    text = str(value or '')
    if not text:
        return {}
    preview = text[:limit]
    summary = {
        'length': len(text),
        'preview': preview,
    }
    if len(text) > limit:
        summary['truncated'] = True
        summary['omitted_character_count'] = len(text) - limit
    return summary


def _copy_compact_keys(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value not in (None, '', [], {}):
            payload[key] = _json_safe(value)
    return payload


def _compact_backend_result_for_ledger(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    compact = _copy_compact_keys(
        result,
        (
            'created_at',
            'done',
            'done_reason',
            'load_duration',
            'model',
            'prompt_eval_count',
            'prompt_eval_duration',
            'total_duration',
        ),
    )
    omitted_keys = [
        str(key)
        for key, value in result.items()
        if key not in compact and value not in (None, '', [], {})
    ]
    if omitted_keys:
        compact['omitted_result_keys'] = sorted(set(omitted_keys))
        compact['full_result_in_snapshot'] = True
    return compact


def _compact_saved_text_artifacts_for_ledger(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    artifacts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        compact = _copy_compact_keys(
            item,
            (
                'artifact_ref',
                'extension',
                'mime_type',
                'path',
                'source_name',
                'type',
            ),
        )
        request = item.get('text_artifact_request')
        if isinstance(request, Mapping):
            compact['text_artifact_request'] = _copy_compact_keys(
                request,
                ('extension', 'source', 'source_name', 'target_path'),
            )
        if compact:
            artifacts.append(compact)
    return artifacts


def _compact_execution_contract_for_ledger(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        return {}
    compact = _copy_compact_keys(
        contract,
        (
            'artifact_prompt_source',
            'branch_id',
            'capability',
            'content_payload_source',
            'kind',
            'output_type',
            'phase_id',
        ),
    )
    output_contract = contract.get('output_contract')
    if isinstance(output_contract, Mapping):
        compact['output_contract'] = _copy_compact_keys(
            output_contract,
            ('fulfillment_policy', 'output_type', 'required', 'status'),
        )
    artifact_request = contract.get('artifact_request')
    if isinstance(artifact_request, Mapping):
        compact['artifact_request'] = _copy_compact_keys(
            artifact_request,
            ('extension', 'source', 'source_name', 'target_path'),
        )
    return {
        key: value
        for key, value in _json_safe(compact).items()
        if value not in (None, '', [], {})
    }


def _compact_late_fill_result_for_ledger(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    compact = _copy_compact_keys(
        result,
        (
            'artifact_ref',
            'branch_id',
            'capability',
            'fill_instance_id',
            'fill_model',
            'obligation_id',
            'output_type',
            'phase_id',
            'provenance_id',
            'route_reason',
            'route_source',
            'saved_audio_path',
            'saved_image_path',
            'saved_text_path',
            'seed',
            'status',
            'text_artifact_source',
            'text_artifact_source_name',
            'text_artifact_target_path',
            'type',
        ),
    )
    for key in ('workload_task_ref', 'output_obligation_ref'):
        value = result.get(key)
        if isinstance(value, Mapping) and value:
            compact[key] = _json_safe(value)
    execution_contract = _compact_execution_contract_for_ledger(result.get('execution_contract'))
    if execution_contract:
        compact['execution_contract_summary'] = execution_contract
    saved_text_artifacts = _compact_saved_text_artifacts_for_ledger(result.get('saved_text_artifacts'))
    if saved_text_artifacts:
        compact['saved_text_artifacts'] = saved_text_artifacts
    text_artifact_requests = result.get('text_artifact_requests')
    if isinstance(text_artifact_requests, list) and text_artifact_requests:
        compact['text_artifact_requests'] = [
            _copy_compact_keys(item, ('extension', 'source', 'source_name', 'target_path'))
            for item in text_artifact_requests
            if isinstance(item, Mapping)
        ]
    backend_summary = _compact_backend_result_for_ledger(result.get('result'))
    if backend_summary:
        compact['backend_result_summary'] = backend_summary
    for key in ('content_payload', 'result_text'):
        summary = _ledger_text_summary(result.get(key))
        if summary:
            compact[f'{key}_summary'] = summary
    omitted_keys = [
        str(key)
        for key, value in result.items()
        if key not in compact
        and key not in {
            'content_payload',
            'execution_contract',
            'result',
            'result_text',
            'saved_text_artifacts',
            'text_artifact_requests',
        }
        and value not in (None, '', [], {})
    ]
    if omitted_keys:
        compact['externalized_result_keys'] = sorted(set(omitted_keys))
    compact['full_result_in_snapshot'] = True
    return {
        key: value
        for key, value in _json_safe(compact).items()
        if value not in (None, '', [], {})
    }


def _compact_late_fill_branch_for_ledger(branch: Any) -> dict[str, Any]:
    if not isinstance(branch, Mapping):
        return {}
    compact = _copy_compact_keys(
        branch,
        (
            'artifact_ref',
            'blocked_prerequisite',
            'blocked_reason',
            'blocked_scope',
            'branch_id',
            'capability',
            'code',
            'error_ref',
            'execution_gate_status',
            'fill_instance_id',
            'follow_up_capability',
            'lifecycle',
            'materialization_blocked',
            'needs_external_input',
            'obligation_id',
            'output_type',
            'phase_id',
            'queue_index',
            'repair_action',
            'repair_work_available',
            'repair_work_policy',
            'saved_audio_path',
            'saved_image_path',
            'saved_text_path',
            'status',
            'type',
        ),
    )
    for key in ('workload_task_ref', 'output_obligation_ref', 'recovery_context', 'recovery_state'):
        value = branch.get(key)
        if isinstance(value, Mapping) and value:
            compact[key] = _json_safe(value)
    if isinstance(branch.get('depends_on'), list):
        compact['depends_on'] = _json_safe(branch.get('depends_on'))
    if isinstance(branch.get('input_refs'), list):
        compact['input_refs'] = _json_safe(branch.get('input_refs'))
    output_contract = branch.get('output_contract')
    if isinstance(output_contract, Mapping):
        compact['output_contract'] = _copy_compact_keys(
            output_contract,
            ('fulfillment_policy', 'output_type', 'required', 'status'),
        )
    artifact_request = branch.get('artifact_request')
    if isinstance(artifact_request, Mapping):
        compact['artifact_request'] = _copy_compact_keys(
            artifact_request,
            ('extension', 'source', 'source_name', 'target_path'),
        )
    return {
        key: value
        for key, value in _json_safe(compact).items()
        if value not in (None, '', [], {})
    }


def _compact_late_fill_branch_list_for_ledger(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        compact
        for compact in (_compact_late_fill_branch_for_ledger(item) for item in value)
        if compact
    ]


def _compact_repair_loop_for_ledger(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return _copy_compact_keys(
        value,
        (
            'authority',
            'auto_execute',
            'blocked_contract_count',
            'executable_contract_count',
            'kind',
            'materialization_blocked_contract_count',
            'max_rounds_policy',
            'needs_external_input_count',
            'next_actions',
            'promoted_contract_count',
            'repair_work_available',
            'repair_work_available_count',
            'requires_promotion',
            'round',
            'status',
        ),
    )


def _compact_repair_feedback_for_ledger(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact = _copy_compact_keys(
        value,
        (
            'current_phase_id',
            'graph_mode',
            'intent_reanalysis_policy',
            'kind',
            'patch_scope',
            'preserve_request_id',
            'reason',
            'repair_mode',
            'status',
            'target',
        ),
    )
    repair_loop = _compact_repair_loop_for_ledger(value.get('repair_loop'))
    if repair_loop:
        compact['repair_loop'] = repair_loop
    items = value.get('items')
    if isinstance(items, list) and items:
        compact['items'] = [
            _copy_compact_keys(
                item,
                (
                    'action',
                    'auto_execute',
                    'blocked_prerequisite',
                    'branch_id',
                    'capability',
                    'check_id',
                    'code',
                    'materialization_blocked',
                    'needs_external_input',
                    'obligation_id',
                    'output_type',
                    'phase_id',
                    'reason',
                    'repair_action',
                    'repair_work_available',
                    'status',
                ),
            )
            for item in items
            if isinstance(item, Mapping)
        ]
    contracts = value.get('repair_rebuild_contracts')
    if isinstance(contracts, list) and contracts:
        compact['repair_rebuild_contract_count'] = len(contracts)
        compact['repair_rebuild_contracts'] = [
            _copy_compact_keys(
                contract,
                (
                    'branch_id',
                    'capability',
                    'kind',
                    'materialization_blocked',
                    'needs_external_input',
                    'obligation_id',
                    'output_type',
                    'phase_id',
                    'repair_action',
                    'repair_work_available',
                    'status',
                ),
            )
            for contract in contracts[:16]
            if isinstance(contract, Mapping)
        ]
        if len(contracts) > 16:
            compact['repair_rebuild_contracts_truncated'] = True
    if isinstance(value.get('decision_contract_guidance'), Mapping):
        compact['decision_contract_guidance_externalized'] = True
    compact['full_feedback_in_snapshot'] = True
    return {
        key: value
        for key, value in _json_safe(compact).items()
        if value not in (None, '', [], {})
    }


def _compact_late_fill_for_ledger(late_fill: Mapping[str, Any], *, snapshot_ref: Mapping[str, Any]) -> dict[str, Any]:
    compact = _copy_compact_keys(
        late_fill,
        (
            'active_branch_count',
            'active_capability',
            'active_capabilities',
            'active_count',
            'auto_recovery_enabled',
            'blocked_branch_count',
            'cancelled_branch_count',
            'cancelled_capabilities',
            'code',
            'completed_at',
            'completed_branch_count',
            'completed_capabilities',
            'created_at',
            'error_code',
            'expected_branch_id',
            'expected_capability',
            'expected_phase_id',
            'failed_at',
            'failed_branch_count',
            'failed_capabilities',
            'final_materialization_contract_reason',
            'final_materialization_contract_status',
            'kind',
            'late_fill_supported',
            'materialization_blocked',
            'needs_external_input',
            'pending_branch_count',
            'pending_capabilities',
            'pending_count',
            'queued_branch_count',
            'queued_capabilities',
            'queued_count',
            'repair_action',
            'repair_actions',
            'repair_code',
            'repair_scope',
            'repair_trigger',
            'skip_kind',
            'skip_reason',
            'skip_source',
            'skipped_capabilities',
            'started_at',
            'status',
            'trigger',
            'updated_at',
            'running_branch_count',
            'running_capabilities',
            'running_count',
        ),
    )
    error_summary = _ledger_text_summary(late_fill.get('error'))
    if error_summary:
        compact['error_summary'] = error_summary
    for key in (
        'active_branches',
        'cancelled_branches',
        'completed_branches',
        'failed_branches',
        'materialization_contract_open_checks',
        'open_branches',
        'pending_branches',
    ):
        branches = _compact_late_fill_branch_list_for_ledger(late_fill.get(key))
        if branches:
            compact[key] = branches
    fill_results = late_fill.get('fill_results')
    if isinstance(fill_results, list) and fill_results:
        compact['fill_result_count'] = len(fill_results)
        compact['fill_results'] = [
            result
            for result in (_compact_late_fill_result_for_ledger(item) for item in fill_results)
            if result
        ]
    recovery_candidates = late_fill.get('recovery_candidates')
    if isinstance(recovery_candidates, list) and recovery_candidates:
        compact['recovery_candidates'] = [
            _copy_compact_keys(
                candidate,
                (
                    'branch_id',
                    'capability',
                    'can_retry',
                    'code',
                    'exclude_instance_ids',
                    'failed_instance_id',
                    'preserve_intent',
                    'retry_scope',
                    'suggested_action',
                ),
            )
            for candidate in recovery_candidates
            if isinstance(candidate, Mapping)
        ]
    linked_rebinds = late_fill.get('linked_artifact_rebinds')
    if isinstance(linked_rebinds, list) and linked_rebinds:
        compact['linked_artifact_rebinds'] = _json_safe(linked_rebinds)
    repair_loop = _compact_repair_loop_for_ledger(late_fill.get('repair_loop'))
    if repair_loop:
        compact['repair_loop'] = repair_loop
    repair_feedback = _compact_repair_feedback_for_ledger(late_fill.get('ghost_repair_feedback'))
    if repair_feedback:
        compact['ghost_repair_feedback'] = repair_feedback
    rebuild_contracts = late_fill.get('repair_rebuild_contracts')
    if isinstance(rebuild_contracts, list) and rebuild_contracts:
        compact['repair_rebuild_contract_count'] = len(rebuild_contracts)
    reconsideration = late_fill.get('reconsideration_rebuild')
    if isinstance(reconsideration, Mapping):
        compact['reconsideration_rebuild'] = _copy_compact_keys(
            reconsideration,
            (
                'authority',
                'auto_execute',
                'kind',
                'needs_external_input_count',
                'pending_branch_count',
                'promoted_contract_count',
                'repair_work_available',
                'repair_work_available_count',
                'status',
            ),
        )
    externalized_skip_keys = set(_LATE_FILL_REVIEW_KEYS) | {
        'error',
        'fill_results',
        'ghost_repair_feedback',
        'linked_artifact_rebinds',
        'reconsideration_rebuild',
        'recovery_candidates',
        'repair_loop',
        'repair_rebuild_contracts',
    }
    externalized_keys = [
        str(key)
        for key, value in late_fill.items()
        if key not in compact
        and key not in externalized_skip_keys
        and value not in (None, '', [], {})
    ]
    compact['full_snapshot_ref'] = _json_safe(snapshot_ref)
    omitted_reviews = [key for key in _LATE_FILL_REVIEW_KEYS if late_fill.get(key) not in (None, '', [], {})]
    if omitted_reviews:
        compact['review_snapshot_ref'] = _json_safe(snapshot_ref)
        compact['externalized_review_keys'] = omitted_reviews
    if externalized_keys:
        compact['externalized_late_fill_keys'] = sorted(set(externalized_keys))
    return compact


def _compact_contract_for_ledger(contract: Mapping[str, Any], *, snapshot_ref: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    externalized_keys: list[str] = []
    for key, value in contract.items():
        if value in (None, '', [], {}):
            continue
        if key in _CONTRACT_HEAVY_KEYS or _json_size_bytes(value) > _LEDGER_LARGE_CONTRACT_LIMIT_BYTES:
            externalized_keys.append(str(key))
            continue
        compact[str(key)] = _json_safe(value)
    compact['full_snapshot_ref'] = _json_safe(snapshot_ref)
    if externalized_keys:
        compact['externalized_contract_keys'] = sorted(externalized_keys)
    return compact


def _externalize_child_ref(
    payload: dict[str, Any],
    key: str,
    *,
    json_path: str,
    snapshot,
    min_size_bytes: int = _NESTED_SNAPSHOT_LIMIT_BYTES,
) -> bool:
    value = payload.get(key)
    if not isinstance(value, (dict, list)):
        return False
    if _json_size_bytes(value) < min_size_bytes:
        return False
    payload[f'{key}_snapshot_ref'] = _json_safe(snapshot(value, json_path))
    payload.pop(key, None)
    return True


def _externalize_large_children(
    payload: dict[str, Any],
    *,
    base_json_path: str,
    snapshot,
    keys: tuple[str, ...] | set[str],
    min_size_bytes: int = _NESTED_SNAPSHOT_LIMIT_BYTES,
) -> None:
    for key in keys:
        _externalize_child_ref(
            payload,
            key,
            json_path=f'{base_json_path}.{key}',
            snapshot=snapshot,
            min_size_bytes=min_size_bytes,
        )


def _artifact_ref_from_projection(item: Mapping[str, Any]) -> str:
    direct = str(item.get('artifact_ref') or item.get('ref') or item.get('artifact_id') or '').strip()
    if direct:
        return direct
    artifacts = item.get('artifacts')
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            token = str(
                artifact.get('artifact_ref')
                or artifact.get('ref')
                or artifact.get('artifact_id')
                or ''
            ).strip()
            if token:
                return token
    artifact = item.get('artifact') if isinstance(item.get('artifact'), Mapping) else {}
    return str(
        artifact.get('artifact_ref')
        or artifact.get('ref')
        or artifact.get('artifact_id')
        or ''
    ).strip()


def _prune_artifact_projection_item(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    artifact_ref = _artifact_ref_from_projection(item)
    payload = dict(item)
    if artifact_ref:
        payload['artifact_ref'] = artifact_ref
        for key in _ARTIFACT_PROJECTION_DETAIL_KEYS:
            payload.pop(key, None)
    return {
        key: _json_safe(value)
        for key, value in payload.items()
        if value not in (None, '', [], {})
    }


def _prune_artifact_projection_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        pruned
        for pruned in (_prune_artifact_projection_item(item) for item in value)
        if isinstance(pruned, dict) and pruned
    ]


def _artifact_identity_summary(canonicalized: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        'canonicalization_required': canonicalized.get('canonicalization_required') or False,
        'duplicate_refs': canonicalized.get('duplicate_refs') or [],
        'final_projection_blocked': canonicalized.get('final_projection_blocked') or False,
        'conflicts': canonicalized.get('conflicts') or [],
    }
    return {
        key: _json_safe(value)
        for key, value in summary.items()
        if value not in (None, '', [], {})
    }


def _canonical_artifact_aliases(artifacts: Any) -> dict[str, dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    if not isinstance(artifacts, list):
        return aliases
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        artifact_ref = _artifact_ref_from_projection(artifact)
        if not artifact_ref:
            continue
        payload = {}
        if artifact.get('alias_artifact_refs') not in (None, '', [], {}):
            payload['alias_artifact_refs'] = artifact.get('alias_artifact_refs')
        if artifact.get('alias_metadata') not in (None, '', [], {}):
            payload['alias_metadata'] = artifact.get('alias_metadata')
        if payload:
            aliases[artifact_ref] = _json_safe(payload)
    return aliases


def _apply_artifact_identity_to_outputs(
    outputs: Any,
    *,
    artifact_identity: Mapping[str, Any],
    canonical_artifacts: Any,
) -> list[dict[str, Any]]:
    if not isinstance(outputs, list):
        return []
    identity = artifact_identity if isinstance(artifact_identity, Mapping) else {}
    if not identity.get('canonicalization_required'):
        return _json_safe(outputs)
    alias_by_ref = _canonical_artifact_aliases(canonical_artifacts)
    conflict_refs = {
        str(item.get('artifact_ref') or '').strip()
        for item in (identity.get('conflicts') or [])
        if isinstance(item, Mapping) and str(item.get('artifact_ref') or '').strip()
    }
    projected: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for item in outputs:
        if not isinstance(item, Mapping):
            continue
        payload = dict(item)
        artifact_ref = _artifact_ref_from_projection(payload)
        if artifact_ref in conflict_refs:
            if artifact_ref in seen_refs:
                continue
            seen_refs.add(artifact_ref)
            payload['artifact_ref'] = artifact_ref
            payload['status'] = 'repair_needed'
            payload['blocked_reason'] = 'conflicting_duplicate_artifact_ref'
            payload['final_projection_blocked'] = True
            projected.append(_json_safe(payload))
            continue
        if artifact_ref:
            if artifact_ref in seen_refs:
                continue
            seen_refs.add(artifact_ref)
            if artifact_ref in alias_by_ref:
                payload.update(alias_by_ref[artifact_ref])
        projected.append(_json_safe(payload))
    return projected


def _prune_artifact_projections(frame: dict[str, Any]) -> None:
    output = frame.get('output') if isinstance(frame.get('output'), dict) else {}
    if isinstance(output.get('outputs'), list):
        output['outputs'] = _prune_artifact_projection_list(output.get('outputs'))
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), dict) else {}
    if isinstance(current_state.get('outputs'), list):
        current_state['outputs'] = _prune_artifact_projection_list(current_state.get('outputs'))
    if isinstance(current_state.get('output_slots'), list):
        current_state['output_slots'] = _prune_artifact_projection_list(current_state.get('output_slots'))
    planning = frame.get('planning') if isinstance(frame.get('planning'), dict) else {}
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), dict) else {}
    if isinstance(artifact_flow.get('output_slots'), list):
        artifact_flow['output_slots'] = _prune_artifact_projection_list(artifact_flow.get('output_slots'))


def _snapshot_items_from_frame(frame: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    external = frame.get('external_snapshots') if isinstance(frame.get('external_snapshots'), Mapping) else {}
    items = external.get('items') if isinstance(external.get('items'), Mapping) else {}
    return {
        str(json_path): _json_safe(ref)
        for json_path, ref in items.items()
        if isinstance(ref, Mapping) and ref.get('sha256')
    }


def _snapshot_refs_same(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_sha = str(left.get('sha256') or '').strip()
    right_sha = str(right.get('sha256') or '').strip()
    return bool(left_sha and right_sha and left_sha == right_sha)


def _effective_snapshot_manifest(
    frame: Mapping[str, Any],
    *,
    parent_manifest: Optional[Mapping[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    effective: dict[str, dict[str, Any]] = {
        str(path): _json_safe(ref)
        for path, ref in (parent_manifest or {}).items()
        if isinstance(ref, Mapping) and ref.get('sha256')
    }
    effective.update(_snapshot_items_from_frame(frame))
    return effective


def _expand_frame_snapshot_manifest(
    frame: Mapping[str, Any],
    *,
    effective_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expanded = dict(frame)
    if not effective_manifest:
        return _json_safe(expanded)
    external = (
        dict(expanded.get('external_snapshots'))
        if isinstance(expanded.get('external_snapshots'), Mapping)
        else {
            'kind': 'ollmo.response_frame_external_snapshots',
            'storage': 'sidecar_json',
            'version': 1,
        }
    )
    delta_items = external.get('items') if isinstance(external.get('items'), Mapping) else {}
    if delta_items:
        external['delta_items'] = _json_safe(delta_items)
    external['items'] = _json_safe(effective_manifest)
    external['effective_manifest_expanded'] = True
    external['effective_snapshot_count'] = len(effective_manifest)
    expanded['external_snapshots'] = external
    return _json_safe(expanded)


def _expand_snapshot_manifests_for_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests_by_frame_id: dict[str, dict[str, dict[str, Any]]] = {}
    expanded_frames: list[dict[str, Any]] = []
    previous_manifest: dict[str, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        relation = frame.get('frame_relation') if isinstance(frame.get('frame_relation'), Mapping) else {}
        parent_frame_id = str(relation.get('parent_frame_id') or '').strip()
        parent_manifest = manifests_by_frame_id.get(parent_frame_id, previous_manifest if parent_frame_id else {})
        effective = _effective_snapshot_manifest(frame, parent_manifest=parent_manifest)
        expanded = _expand_frame_snapshot_manifest(frame, effective_manifest=effective)
        frame_id = str(expanded.get('frame_id') or '').strip()
        if frame_id:
            manifests_by_frame_id[frame_id] = effective
        previous_manifest = effective
        expanded_frames.append(expanded)
    return expanded_frames


def _apply_snapshot_manifest_to_frame(
    frame: dict[str, Any],
    *,
    snapshot_refs: Mapping[str, Mapping[str, Any]],
    parent_frame: Optional[Mapping[str, Any]] = None,
    parent_manifest: Optional[Mapping[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    current_manifest = {
        str(path): _json_safe(ref)
        for path, ref in snapshot_refs.items()
        if isinstance(ref, Mapping) and ref.get('sha256')
    }
    inherited_paths: list[str] = []
    delta_items: dict[str, dict[str, Any]] = {}
    normalized_parent_manifest = {
        str(path): _json_safe(ref)
        for path, ref in (parent_manifest or {}).items()
        if isinstance(ref, Mapping) and ref.get('sha256')
    }
    for json_path, ref in current_manifest.items():
        parent_ref = normalized_parent_manifest.get(json_path)
        if parent_ref and _snapshot_refs_same(ref, parent_ref):
            inherited_paths.append(json_path)
            continue
        delta_items[json_path] = ref
    effective_manifest = {**normalized_parent_manifest, **current_manifest}
    inherited_paths = sorted(set(inherited_paths) | (set(normalized_parent_manifest.keys()) - set(delta_items.keys())))
    if not delta_items and not inherited_paths and not effective_manifest:
        return {}

    external_snapshots = {
        'kind': 'ollmo.response_frame_external_snapshots',
        'items': _json_safe(delta_items),
        'storage': 'sidecar_json',
        'version': 1,
    }
    if parent_frame and (inherited_paths or normalized_parent_manifest):
        external_snapshots['inheritance'] = {
            'kind': 'ollmo.response_frame_snapshot_inheritance',
            'strategy': 'delta_manifest',
            'parent_frame_id': str(parent_frame.get('frame_id') or '').strip() or None,
            'parent_frame_sequence': parent_frame.get('frame_sequence'),
            'inherited_snapshot_count': len(inherited_paths),
            'inherited_json_paths': inherited_paths,
            'parent_snapshot_count': len(normalized_parent_manifest),
            'effective_snapshot_count': len(effective_manifest),
        }
    frame['external_snapshots'] = _json_safe(external_snapshots)
    unique_snapshot_paths = {
        str(ref.get('path') or '')
        for ref in effective_manifest.values()
        if isinstance(ref, Mapping) and ref.get('path') not in (None, '')
    }
    frame['snapshot_policy'] = {
        'kind': _LEDGER_SNAPSHOT_POLICY_KIND,
        'dedupe_strategy': 'content_sha256',
        'ledger_payload': 'compact',
        'snapshot_ref_count': len(delta_items),
        'effective_snapshot_ref_count': len(effective_manifest),
        'inherited_snapshot_ref_count': len(inherited_paths),
        'truth_preservation': 'large_runtime_and_public_bodies_externalized_by_ref',
        'snapshot_count': len(delta_items),
        'effective_snapshot_count': len(effective_manifest),
        'unique_snapshot_count': len(unique_snapshot_paths),
    }
    return effective_manifest


def _runtime_snapshot_payload(runtime: Mapping[str, Any], *, snapshot) -> dict[str, Any]:
    payload = _json_safe(runtime)
    if not isinstance(payload, dict):
        return {}

    developer_diagnostics = (
        payload.get('developer_diagnostics')
        if isinstance(payload.get('developer_diagnostics'), dict)
        else None
    )
    if developer_diagnostics:
        _externalize_large_children(
            developer_diagnostics,
            base_json_path='runtime.developer_diagnostics',
            snapshot=snapshot,
            keys={
                'closure_repair_graph_patch',
                'graph_closure_review',
                'request_phase_graph_refinements',
            },
        )

    execution_planner = (
        payload.get('execution_planner')
        if isinstance(payload.get('execution_planner'), dict)
        else None
    )
    if execution_planner:
        _externalize_large_children(
            execution_planner,
            base_json_path='runtime.execution_planner',
            snapshot=snapshot,
            keys={'deferred_branches', 'graph', 'phase_graph', 'request_phase_graph'},
        )

    context_strategy = (
        payload.get('context_strategy')
        if isinstance(payload.get('context_strategy'), dict)
        else None
    )
    if context_strategy:
        _externalize_large_children(
            context_strategy,
            base_json_path='runtime.context_strategy',
            snapshot=snapshot,
            keys={'context_candidates', 'context_gate_review'},
            min_size_bytes=1,
        )

    _externalize_large_children(
        payload,
        base_json_path='runtime',
        snapshot=snapshot,
        keys={
            'context_strategy',
            'developer_diagnostics',
            'execution_planner',
            'graph_closure_review',
            'late_fill',
            'request_phase_graph',
            'semantic_role_profile',
        },
        min_size_bytes=_LEDGER_LARGE_CONTRACT_LIMIT_BYTES,
    )
    return payload


def _working_frame_snapshot_payload(working_frame: Mapping[str, Any], *, snapshot) -> dict[str, Any]:
    payload = compact_working_frame_for_serialization(
        working_frame,
        snapshot=snapshot,
        base_json_path='working_frame',
    )
    if not isinstance(payload, dict):
        return {}
    return payload


def _late_fill_snapshot_payload(late_fill: Mapping[str, Any], *, snapshot, base_json_path: str) -> dict[str, Any]:
    payload = _json_safe(late_fill)
    if not isinstance(payload, dict):
        return {}
    _externalize_large_children(
        payload,
        base_json_path=base_json_path,
        snapshot=snapshot,
        keys=set(_LATE_FILL_REVIEW_KEYS)
        | {
            'fill_results',
            'recovery_context',
        },
    )
    return payload


def _expand_late_fill_snapshot_children(
    value: Mapping[str, Any],
    *,
    frames_dir: Path | str,
    trusted_manifest: Optional[Mapping[str, Any]] = None,
    response_id: str = '',
) -> dict[str, Any]:
    payload = dict(value)
    for key in _LATE_FILL_SNAPSHOT_CHILD_KEYS:
        if payload.get(key) not in (None, '', [], {}):
            continue
        ref = payload.get(f'{key}_snapshot_ref')
        expanded = (
            _read_manifest_authorized_snapshot_payload(
                ref,
                frames_dir=frames_dir,
                trusted_manifest=trusted_manifest,
                response_id=response_id,
                expected_json_path=f'late_fill.{key}',
            )
            if isinstance(trusted_manifest, Mapping)
            else None
        )
        if expanded not in (None, '', [], {}):
            payload[key] = expanded
    return _json_safe(payload)


def _public_body_snapshot_ref(
    value: Any,
    *,
    json_path: str,
    snapshot,
    metadata_keys: list[str],
) -> dict[str, Any]:
    ref = dict(snapshot(value, json_path))
    ref['projection_role'] = 'public_body_exact'
    ref['truth_preservation'] = 'exact_content_addressed_sidecar'
    if metadata_keys:
        ref['public_projection_metadata_keys'] = sorted(set(metadata_keys))
    return ref


def _bounded_projection_mapping_key(
    raw_key: Any,
    *,
    used_keys: set[str],
    limit_bytes: int = _PUBLIC_PREVIEW_MAPPING_KEY_LIMIT_BYTES,
) -> Optional[str]:
    """Return a deterministic, collision-resistant key for bounded previews."""

    return _response_wire_policy.bounded_projection_mapping_key(
        raw_key,
        used_keys=used_keys,
        limit_bytes=limit_bytes,
    )


def _public_collection_preview_handle(value: Any) -> dict[str, Any]:
    encoded, sha256 = _response_wire_json_identity(value)
    return {
        'wire_body_preview': encoded.decode(
            'utf-8', errors='replace'
        )[:_PUBLIC_BODY_PREVIEW_LIMIT_CHARS],
        'wire_body_size_bytes': len(encoded),
        'wire_body_sha256': sha256,
        'wire_body_projection_truncated': True,
    }


def _bounded_public_collection_preview(value: Any) -> Any:
    """Return a small typed preview; the caller retains an exact collection ref."""

    def preview_item(item: Any, *, depth: int = 0) -> Any:
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            if len(item) <= _PUBLIC_BODY_PREVIEW_LIMIT_CHARS:
                return item
            return item[:_PUBLIC_BODY_PREVIEW_LIMIT_CHARS]
        if isinstance(item, (bool, int, float)) or item is None:
            return item
        if depth >= 8:
            encoded, sha256 = _response_wire_json_identity(item)
            return {
                'preview_truncated': True,
                'sha256': sha256,
                'size_bytes': len(encoded),
            }
        if isinstance(item, Mapping):
            projected: dict[str, Any] = {}
            used_keys: set[str] = set()
            for raw_key, child in item.items():
                key = _bounded_projection_mapping_key(
                    raw_key,
                    used_keys=used_keys,
                )
                if key is None:
                    continue
                child_projection = preview_item(child, depth=depth + 1)
                candidate = {**projected, key: child_projection}
                if _json_size_bytes(candidate) <= _PUBLIC_COLLECTION_PREVIEW_BUDGET_BYTES:
                    projected[key] = child_projection
                    continue
                if projected:
                    break
                projected[key] = _public_collection_preview_handle(child)
            return projected
        if isinstance(item, list):
            projected_items: list[Any] = []
            for child in item:
                child_projection = preview_item(child, depth=depth + 1)
                candidate = [*projected_items, child_projection]
                if _json_size_bytes(candidate) <= _PUBLIC_COLLECTION_PREVIEW_BUDGET_BYTES:
                    projected_items.append(child_projection)
                    continue
                if projected_items:
                    break
                projected_items.append(_public_collection_preview_handle(child))
            return projected_items
        return _json_safe(item)

    if isinstance(value, list):
        preview: list[Any] = []
        for item in value:
            projected = preview_item(item)
            candidate = [*preview, projected]
            if (
                preview
                and _json_size_bytes(candidate)
                > _PUBLIC_COLLECTION_PREVIEW_BUDGET_BYTES
            ):
                break
            if _json_size_bytes(candidate) > _PUBLIC_COLLECTION_PREVIEW_BUDGET_BYTES:
                projected = _public_collection_preview_handle(item)
            preview.append(projected)
        return preview
    if isinstance(value, Mapping):
        preview_mapping: dict[str, Any] = {}
        used_keys: set[str] = set()
        for raw_key, item in value.items():
            key = _bounded_projection_mapping_key(
                raw_key,
                used_keys=used_keys,
            )
            if key is None:
                continue
            projected = preview_item(item)
            candidate = {**preview_mapping, key: projected}
            if _json_size_bytes(candidate) <= _PUBLIC_COLLECTION_PREVIEW_BUDGET_BYTES:
                preview_mapping[key] = projected
                continue
            if preview_mapping:
                break
            preview_mapping[key] = _public_collection_preview_handle(item)
        return preview_mapping
    return preview_item(value)


def _public_body_projection_metadata_keys(key: str) -> list[str]:
    return [
        f'{key}_count',
        f'{key}_length_chars',
        f'{key}_preview_truncated',
        f'{key}_projection_truncated',
        f'{key}_sha256',
        f'{key}_size_bytes',
    ]


def _ledger_frame_line_size_bytes(value: Any) -> int:
    return len(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
        ).encode('utf-8')
    ) + 1


def _public_aggregate_collection_candidates(
    frame: Mapping[str, Any],
) -> list[tuple[int, str, dict[str, Any], Any, Any, str]]:
    candidates: list[
        tuple[int, str, dict[str, Any], Any, Any, str]
    ] = []

    def visit_mapping(value: dict[str, Any], *, json_path: str) -> None:
        for raw_key in list(value):
            key = str(raw_key or '').strip()
            if not key or key.endswith('_snapshot_ref'):
                continue
            child = value.get(raw_key)
            child_path = '.'.join(
                token for token in (json_path, key) if token
            )
            if isinstance(child, str):
                size_bytes = len(child.encode('utf-8'))
                if size_bytes >= _PUBLIC_AGGREGATE_COLLECTION_MIN_BYTES:
                    candidates.append(
                        (
                            size_bytes,
                            child_path,
                            value,
                            raw_key,
                            child,
                            'string',
                        )
                    )
                continue
            if not isinstance(child, (Mapping, list)):
                continue
            if (
                isinstance(child, Mapping)
                and str(child.get('kind') or '').strip()
                == _LEDGER_SNAPSHOT_REF_KIND
            ):
                continue
            existing_ref = value.get(f'{key}_snapshot_ref')
            if (
                isinstance(existing_ref, Mapping)
                and str(existing_ref.get('projection_role') or '').strip()
                == 'public_body_exact'
            ):
                continue
            size_bytes = _json_size_bytes(child)
            if size_bytes >= _PUBLIC_AGGREGATE_COLLECTION_MIN_BYTES:
                candidates.append(
                    (
                        size_bytes,
                        child_path,
                        value,
                        raw_key,
                        child,
                        'collection',
                    )
                )
                continue
            if isinstance(child, dict):
                visit_mapping(child, json_path=child_path)
            elif isinstance(child, list):
                for index, item in enumerate(child):
                    if isinstance(item, dict):
                        visit_mapping(
                            item,
                            json_path=f'{child_path}.{index}',
                        )

    for public_key in (
        'artifacts',
        'batch',
        'current_state',
        'error',
        'output',
        'request',
    ):
        public_value = frame.get(public_key)
        if isinstance(public_value, dict):
            visit_mapping(public_value, json_path=public_key)
    return candidates


def _externalize_public_collection(
    parent: dict[str, Any],
    raw_key: Any,
    exact_item: Any,
    *,
    json_path: str,
    snapshot,
    preview_item: Any = None,
) -> None:
    key = str(raw_key or '').strip()
    if preview_item is None:
        preview_item = _compact_public_body_sidecars(
            exact_item,
            json_path=json_path,
            snapshot=snapshot,
        )
    metadata_keys = _public_body_projection_metadata_keys(key)
    for metadata_key in metadata_keys:
        parent.pop(metadata_key, None)
    encoded, sha256 = _response_wire_json_identity(exact_item)
    parent[f'{key}_count'] = len(exact_item)
    parent[f'{key}_size_bytes'] = len(encoded)
    parent[f'{key}_sha256'] = sha256
    parent[f'{key}_projection_truncated'] = True
    parent[f'{key}_snapshot_ref'] = _public_body_snapshot_ref(
        _json_safe(exact_item),
        json_path=json_path,
        snapshot=snapshot,
        metadata_keys=metadata_keys,
    )
    parent[raw_key] = _bounded_public_collection_preview(
        preview_item
    )


def _externalize_public_string(
    parent: dict[str, Any],
    raw_key: Any,
    exact_item: str,
    *,
    json_path: str,
    snapshot,
) -> None:
    key = str(raw_key or '').strip()
    metadata_keys = _public_body_projection_metadata_keys(key)
    for metadata_key in metadata_keys:
        parent.pop(metadata_key, None)
    encoded = exact_item.encode('utf-8')
    parent[f'{key}_length_chars'] = len(exact_item)
    parent[f'{key}_size_bytes'] = len(encoded)
    parent[f'{key}_sha256'] = hashlib.sha256(encoded).hexdigest()
    parent[f'{key}_preview_truncated'] = True
    parent[f'{key}_snapshot_ref'] = _public_body_snapshot_ref(
        exact_item,
        json_path=json_path,
        snapshot=snapshot,
        metadata_keys=metadata_keys,
    )
    parent[raw_key] = exact_item[:_PUBLIC_BODY_PREVIEW_LIMIT_CHARS]


def _compact_aggregate_public_collections(
    frame: dict[str, Any],
    *,
    snapshot,
) -> None:
    """Compact many medium public bodies when their aggregate exceeds budget."""

    source_size_bytes = _ledger_frame_line_size_bytes(frame)
    if source_size_bytes <= _RESPONSE_WIRE_INLINE_BUDGET_BYTES:
        return
    candidates = sorted(
        _public_aggregate_collection_candidates(frame),
        key=lambda item: (-item[0], item[1]),
    )
    externalized_paths: list[str] = []
    externalized_source_bytes = 0
    externalized_collection_count = 0
    externalized_string_count = 0
    for size_bytes, json_path, parent, raw_key, exact_item, body_kind in candidates:
        if body_kind == 'string':
            _externalize_public_string(
                parent,
                raw_key,
                exact_item,
                json_path=json_path,
                snapshot=snapshot,
            )
            externalized_string_count += 1
        else:
            _externalize_public_collection(
                parent,
                raw_key,
                exact_item,
                json_path=json_path,
                snapshot=snapshot,
            )
            externalized_collection_count += 1
        externalized_paths.append(json_path)
        externalized_source_bytes += size_bytes
        if (
            _ledger_frame_line_size_bytes(frame)
            <= _PUBLIC_AGGREGATE_LEDGER_TARGET_BYTES
        ):
            break
    frame['public_body_compaction'] = {
        'kind': 'ollmo.response_frame_public_body_compaction',
        'strategy': 'aggregate_content_sha256_sidecars',
        'source_size_bytes': source_size_bytes,
        'aggregate_wire_budget_bytes': _RESPONSE_WIRE_INLINE_BUDGET_BYTES,
        'ledger_target_bytes': _PUBLIC_AGGREGATE_LEDGER_TARGET_BYTES,
        'externalized_body_count': len(externalized_paths),
        'externalized_collection_count': externalized_collection_count,
        'externalized_string_count': externalized_string_count,
        'externalized_source_bytes': externalized_source_bytes,
        'externalized_json_paths': externalized_paths,
        'truth_preservation': 'exact_content_addressed_sidecar',
    }


def _compact_public_body_sidecars(
    value: Any,
    *,
    json_path: str,
    snapshot,
) -> Any:
    """Externalize only genuinely bulky public bodies with exact CAS refs."""

    if isinstance(value, list):
        return [
            _compact_public_body_sidecars(
                item,
                json_path=f'{json_path}.{index}',
                snapshot=snapshot,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        return value

    payload = dict(value)
    for raw_key in list(payload):
        key = str(raw_key or '').strip()
        if not key or key.endswith('_snapshot_ref'):
            continue
        item = payload.get(raw_key)
        child_path = '.'.join(token for token in (json_path, key) if token)
        if isinstance(item, str):
            encoded = item.encode('utf-8')
            if len(encoded) < _PUBLIC_BODY_SNAPSHOT_MIN_BYTES:
                continue
            metadata_keys = _public_body_projection_metadata_keys(key)
            for metadata_key in metadata_keys:
                payload.pop(metadata_key, None)
            for metadata_key, metadata_value in (
                (f'{key}_length_chars', len(item)),
                (f'{key}_size_bytes', len(encoded)),
                (f'{key}_sha256', hashlib.sha256(encoded).hexdigest()),
                (f'{key}_preview_truncated', True),
            ):
                payload[metadata_key] = metadata_value
            ref_key = f'{key}_snapshot_ref'
            payload[ref_key] = _public_body_snapshot_ref(
                item,
                json_path=child_path,
                snapshot=snapshot,
                metadata_keys=metadata_keys,
            )
            payload[key] = item[:_PUBLIC_BODY_PREVIEW_LIMIT_CHARS]
            continue
        if not isinstance(item, (Mapping, list)):
            continue
        exact_item = _json_safe(item)
        compact_item = _compact_public_body_sidecars(
            item,
            json_path=child_path,
            snapshot=snapshot,
        )
        if _json_size_bytes(compact_item) < _PUBLIC_COLLECTION_SNAPSHOT_MIN_BYTES:
            payload[raw_key] = compact_item
            continue
        _externalize_public_collection(
            payload,
            raw_key,
            exact_item,
            json_path=child_path,
            snapshot=snapshot,
            preview_item=compact_item,
        )
    return payload


def _trusted_snapshot_manifest(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    external = (
        value.get('external_snapshots')
        if isinstance(value.get('external_snapshots'), Mapping)
        else {}
    )
    items = (
        external.get('items')
        if isinstance(external.get('items'), Mapping)
        else {}
    )
    return {
        str(path): dict(ref)
        for path, ref in items.items()
        if isinstance(ref, Mapping)
    }


def _snapshot_ref_matches_authority(
    ref: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    response_id: str,
    expected_json_path: str,
    allow_normalized_volatile_fields: bool = False,
) -> bool:
    if str(ref.get('kind') or '').strip() != _LEDGER_SNAPSHOT_REF_KIND:
        return False
    ref_json_path = str(ref.get('json_path') or '').strip()
    if (
        ref_json_path != expected_json_path
        and not (allow_normalized_volatile_fields and not ref_json_path)
    ):
        return False
    ref_response_id = str(ref.get('source_response_id') or '').strip()
    authority_response_id = str(
        authority.get('source_response_id') or ''
    ).strip()
    if response_id and authority_response_id != response_id:
        return False
    if (
        response_id
        and ref_response_id != response_id
        and not (allow_normalized_volatile_fields and not ref_response_id)
    ):
        return False
    for key in ('path', 'sha256', 'size_bytes'):
        if ref.get(key) != authority.get(key):
            return False
    return (
        str(authority.get('json_path') or expected_json_path).strip()
        == expected_json_path
        and authority.get('content_addressed') is True
    )


def _authorized_sidecar_child_ref(
    ref: Any,
    authority: Any,
    *,
    expected_json_path: str,
    expected_child_key: str,
    source_response_id: str,
) -> Optional[dict[str, Any]]:
    """Restore normalized child-ref owner fields from authenticated authority."""

    if not isinstance(ref, Mapping) or not isinstance(authority, Mapping):
        return None
    if (
        str(ref.get('kind') or '').strip() != _LEDGER_SNAPSHOT_REF_KIND
        or ref.get('content_addressed') is not True
        or ref.get('sidecar_child_ref') is not True
        or str(ref.get('sidecar_child_key') or '').strip()
        != expected_child_key
        or str(authority.get('key') or '').strip() != expected_child_key
        or str(authority.get('json_path') or '').strip()
        != expected_json_path
    ):
        return None
    for identity_key in ('path', 'sha256', 'size_bytes'):
        authority_value = authority.get(identity_key)
        # Older generated manifests did not repeat the path.  The normalized
        # child ref is still authenticated by its parent's CAS bytes, so keep
        # that backward-compatible binding while requiring all present fields
        # to agree.
        if authority_value not in (None, '') and ref.get(identity_key) != authority_value:
            return None
    if authority.get('content_addressed') not in (None, True):
        return None

    authorized = dict(ref)
    authorized['json_path'] = expected_json_path
    if source_response_id:
        authorized['source_response_id'] = source_response_id
    nested_manifest = authority.get('sidecar_manifest')
    if isinstance(nested_manifest, Mapping):
        authorized['sidecar_manifest'] = _json_safe(nested_manifest)
    return _json_safe(authorized)


def _hydrate_manifest_authorized_snapshot_children(
    value: Any,
    *,
    frames_dir: Path | str,
    trusted_manifest: Mapping[str, Any],
    response_id: str,
    json_path: str,
    ref_depth: int = 0,
) -> Any:
    if ref_depth > 128:
        return _json_safe(value)
    if isinstance(value, list):
        return [
            _hydrate_manifest_authorized_snapshot_children(
                item,
                frames_dir=frames_dir,
                trusted_manifest=trusted_manifest,
                response_id=response_id,
                json_path=_json_child_path(json_path, f'[{index}]'),
                ref_depth=ref_depth,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    for raw_key, ref in list(payload.items()):
        key = str(raw_key or '').strip()
        if not key.endswith('_snapshot_ref') or not isinstance(ref, Mapping):
            continue
        body_key = key[: -len('_snapshot_ref')]
        body_path = _json_child_path(json_path, body_key)
        authority = trusted_manifest.get(body_path)
        if not isinstance(authority, Mapping) or not _snapshot_ref_matches_authority(
            ref,
            authority,
            response_id=response_id,
            expected_json_path=body_path,
            allow_normalized_volatile_fields=True,
        ):
            continue
        restored = _read_snapshot_ref_payload(
            authority,
            frames_dir=frames_dir,
            strict_child_refs=True,
            _max_expand_depth=128,
        )
        if restored is None:
            continue
        payload[body_key] = _hydrate_manifest_authorized_snapshot_children(
            restored,
            frames_dir=frames_dir,
            trusted_manifest=trusted_manifest,
            response_id=response_id,
            json_path=body_path,
            ref_depth=ref_depth + 1,
        )
        payload.pop(raw_key, None)
    for key, item in list(payload.items()):
        payload[key] = _hydrate_manifest_authorized_snapshot_children(
            item,
            frames_dir=frames_dir,
            trusted_manifest=trusted_manifest,
            response_id=response_id,
            json_path=_json_child_path(json_path, str(key)),
            ref_depth=ref_depth,
        )
    return _json_safe(payload)


def _read_manifest_authorized_snapshot_payload(
    ref: Any,
    *,
    frames_dir: Path | str,
    trusted_manifest: Mapping[str, Any],
    response_id: str,
    expected_json_path: str,
    content_json_path: Optional[str] = None,
) -> Any:
    if not isinstance(ref, Mapping):
        return None
    authority = trusted_manifest.get(expected_json_path)
    if not isinstance(authority, Mapping) or not _snapshot_ref_matches_authority(
        ref,
        authority,
        response_id=response_id,
        expected_json_path=expected_json_path,
    ):
        return None
    restored = _read_snapshot_ref_payload(
        authority,
        frames_dir=frames_dir,
        strict_child_refs=True,
        _max_expand_depth=128,
    )
    if restored is None:
        return None
    return _hydrate_manifest_authorized_snapshot_children(
        restored,
        frames_dir=frames_dir,
        trusted_manifest=trusted_manifest,
        response_id=response_id,
        json_path=str(content_json_path or expected_json_path),
        ref_depth=1,
    )


def _restore_public_body_sidecars(
    value: Any,
    *,
    frames_dir: Path | str,
    _trusted_manifest: Optional[Mapping[str, Any]] = None,
    _response_id: str = '',
    _json_path: str = '',
) -> Any:
    trusted_manifest = (
        dict(_trusted_manifest)
        if isinstance(_trusted_manifest, Mapping)
        else _trusted_snapshot_manifest(value)
    )
    response_id = _response_id or (
        _frame_response_id(value) if isinstance(value, Mapping) else ''
    )
    if isinstance(value, list):
        return [
            _restore_public_body_sidecars(
                item,
                frames_dir=frames_dir,
                _trusted_manifest=trusted_manifest,
                _response_id=response_id,
                _json_path=_json_child_path(_json_path, f'[{index}]'),
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    for raw_key, ref in list(payload.items()):
        key = str(raw_key or '').strip()
        if (
            not key.endswith('_snapshot_ref')
            or not isinstance(ref, Mapping)
            or str(ref.get('projection_role') or '').strip()
            != 'public_body_exact'
        ):
            continue
        body_key = key[: -len('_snapshot_ref')]
        body_path = _json_child_path(_json_path, body_key)
        authority = trusted_manifest.get(body_path)
        if not isinstance(authority, Mapping) or not _snapshot_ref_matches_authority(
            ref,
            authority,
            response_id=response_id,
            expected_json_path=body_path,
        ):
            continue
        restored = _read_snapshot_ref_payload(
            ref,
            frames_dir=frames_dir,
            strict_child_refs=True,
            _max_expand_depth=128,
        )
        if restored is None:
            continue
        payload[body_key] = restored
        payload.pop(raw_key, None)
        for metadata_key in ref.get('public_projection_metadata_keys') or []:
            payload.pop(str(metadata_key), None)
    for key, item in list(payload.items()):
        if key == 'external_snapshots':
            continue
        payload[key] = _restore_public_body_sidecars(
            item,
            frames_dir=frames_dir,
            _trusted_manifest=trusted_manifest,
            _response_id=response_id,
            _json_path=_json_child_path(_json_path, str(key)),
        )
    return payload


def _public_body_snapshot_integrity_error(
    value: Any,
    *,
    frames_dir: Path | str,
    json_path: str = '',
    _visited: Optional[set[tuple[str, str]]] = None,
    _depth: int = 0,
    _ref_counter: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """Validate only CAS refs authenticated by the generated manifest graph."""

    if not isinstance(value, Mapping):
        return None
    response_id = _frame_response_id(value)
    trusted_manifest = _trusted_snapshot_manifest(value)
    visited = _visited if _visited is not None else set()
    ref_counter = _ref_counter if _ref_counter is not None else [0]
    public_body_paths: set[str] = set()
    public_scan: list[tuple[Any, str]] = [(value, json_path)]
    while public_scan:
        public_item, public_parent_path = public_scan.pop()
        if isinstance(public_item, list):
            public_scan.extend(
                (
                    child,
                    _json_child_path(public_parent_path, f'[{index}]'),
                )
                for index, child in enumerate(public_item)
            )
            continue
        if not isinstance(public_item, Mapping):
            continue
        for raw_key, child in public_item.items():
            key = str(raw_key or '').strip()
            if key == 'external_snapshots':
                continue
            derived_key = (
                key[: -len('_snapshot_ref')]
                if key.endswith('_snapshot_ref')
                else ''
            )
            child_path = _json_child_path(public_parent_path, derived_key)
            authority = trusted_manifest.get(child_path)
            if (
                derived_key
                and isinstance(child, Mapping)
                and isinstance(authority, Mapping)
                and str(child.get('projection_role') or '').strip()
                == 'public_body_exact'
                and all(
                    child.get(identity_key) == authority.get(identity_key)
                    for identity_key in ('path', 'sha256', 'size_bytes')
                )
            ):
                public_body_paths.add(child_path)
                continue
            public_scan.append(
                (child, _json_child_path(public_parent_path, key))
            )

    def invalid_ref_error(ref: Mapping[str, Any], expected_path: str) -> dict[str, Any]:
        return {
            'code': 'response_frame_snapshot_ref_invalid',
            'message': 'An authoritative response-frame snapshot ref has an invalid CAS shape or owner binding.',
            'json_path': str(ref.get('json_path') or expected_path) or None,
            'snapshot_path': str(ref.get('path') or '').strip() or None,
            'expected_sha256': str(ref.get('sha256') or '').strip() or None,
        }

    def validate_ref(
        ref: Mapping[str, Any],
        *,
        expected_path: str,
        ref_depth: int,
        authority: Optional[Mapping[str, Any]] = None,
        normalized_child_ref: bool = False,
    ) -> Optional[dict[str, Any]]:
        raw_path = str(ref.get('path') or '').strip()
        expected_sha = str(ref.get('sha256') or '').strip().lower()
        size_bytes = ref.get('size_bytes')
        if (
            ref_depth > 128
            or ref_counter[0] >= 100_000
        ):
            return {
                'code': 'response_frame_snapshot_graph_limit_exceeded',
                'message': 'Response-frame snapshot references exceed the canonical ref-hop limit.',
                'json_path': expected_path or None,
            }
        if (
            not raw_path
            or str(ref.get('kind') or '').strip() != _LEDGER_SNAPSHOT_REF_KIND
            or (
                not normalized_child_ref
                and str(ref.get('json_path') or '').strip() != expected_path
            )
            or len(expected_sha) != 64
            or any(character not in '0123456789abcdef' for character in expected_sha)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or ref.get('content_addressed') is not True
            or (
                not normalized_child_ref
                and response_id
                and str(ref.get('source_response_id') or '').strip()
                != response_id
            )
            or (
                isinstance(authority, Mapping)
                and not _snapshot_ref_matches_authority(
                    ref,
                    authority,
                    response_id=response_id,
                    expected_json_path=expected_path,
                )
            )
        ):
            return invalid_ref_error(ref, expected_path)
        identity = (raw_path, expected_sha)
        if identity in visited:
            return None
        visited.add(identity)
        ref_counter[0] += 1
        restored = _read_snapshot_ref_payload(
            ref,
            frames_dir=frames_dir,
            expand_child_refs=False,
        )
        if restored is None:
            is_public_body = (
                str(ref.get('projection_role') or '').strip()
                == 'public_body_exact'
                or expected_path in public_body_paths
            )
            return {
                'code': (
                    'response_frame_public_body_snapshot_unavailable'
                    if is_public_body
                    else 'response_frame_snapshot_unavailable'
                ),
                'message': (
                    'An exact content-addressed public response body is missing, malformed, or fails its SHA-256 binding.'
                    if is_public_body
                    else 'An authoritative response-frame snapshot is missing, malformed, or fails its SHA-256 binding.'
                ),
                'json_path': expected_path or None,
                'snapshot_path': raw_path,
                'expected_sha256': expected_sha,
            }

        sidecar_manifest = (
            ref.get('sidecar_manifest')
            if isinstance(ref.get('sidecar_manifest'), Mapping)
            else {}
        )
        child_authorities = {
            str(item.get('json_path') or '').strip(): item
            for item in (sidecar_manifest.get('child_refs') or [])
            if isinstance(item, Mapping)
            and str(item.get('json_path') or '').strip()
        }
        stack: list[tuple[Any, str]] = [(restored, expected_path)]
        while stack:
            child_value, parent_path = stack.pop()
            if isinstance(child_value, list):
                stack.extend(
                    (
                        item,
                        _json_child_path(parent_path, f'[{index}]'),
                    )
                    for index, item in enumerate(child_value)
                )
                continue
            if not isinstance(child_value, Mapping):
                continue
            for raw_key, child in child_value.items():
                key = str(raw_key or '').strip()
                derived_key = (
                    key[: -len('_snapshot_ref')]
                    if key.endswith('_snapshot_ref')
                    else ''
                )
                expected_child_path = _json_child_path(parent_path, derived_key)
                child_authority = child_authorities.get(expected_child_path)
                authorized_child = _authorized_sidecar_child_ref(
                    child,
                    child_authority,
                    expected_json_path=expected_child_path,
                    expected_child_key=derived_key,
                    source_response_id=response_id,
                )
                if authorized_child is not None:
                    error = validate_ref(
                        authorized_child,
                        expected_path=expected_child_path,
                        ref_depth=ref_depth + 1,
                        authority=authorized_child,
                    )
                    if error:
                        return error
                    continue
                stack.append(
                    (child, _json_child_path(parent_path, key))
                )
        return None

    for manifest_path, authority in sorted(trusted_manifest.items()):
        error = validate_ref(
            authority,
            expected_path=manifest_path,
            ref_depth=1,
            authority=authority,
        )
        if error:
            return error

    # A generated inline ref at a manifested path must agree with the manifest.
    # Ref-shaped arbitrary user JSON at every other path remains ordinary data.
    stack: list[tuple[Any, str]] = [(value, json_path)]
    while stack:
        item, parent_path = stack.pop()
        if isinstance(item, list):
            stack.extend(
                (
                    child,
                    _json_child_path(parent_path, f'[{index}]'),
                )
                for index, child in enumerate(item)
            )
            continue
        if not isinstance(item, Mapping):
            continue
        for raw_key, child in item.items():
            key = str(raw_key or '').strip()
            if key == 'external_snapshots':
                continue
            derived_key = (
                key[: -len('_snapshot_ref')]
                if key.endswith('_snapshot_ref')
                else ''
            )
            expected_path = _json_child_path(parent_path, derived_key)
            authority = trusted_manifest.get(expected_path)
            if derived_key and isinstance(child, Mapping) and isinstance(authority, Mapping):
                if not _snapshot_ref_matches_authority(
                    child,
                    authority,
                    response_id=response_id,
                    expected_json_path=expected_path,
                ):
                    return invalid_ref_error(child, expected_path)
                continue
            stack.append((child, _json_child_path(parent_path, key)))
    return None


def compact_response_frame_for_ledger(
    response_frame: Mapping[str, Any],
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    parent_frame: Optional[Mapping[str, Any]] = None,
    parent_snapshot_manifest: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a compact ledger frame while preserving large truth as sidecar refs."""

    frame = _json_safe(response_frame)
    if not isinstance(frame, dict):
        raise ValueError('response_frame must serialize to a JSON object')
    # These namespaces are generated durability authority. Caller-supplied
    # values must never be able to self-authorize a forged snapshot ref.
    frame.pop('external_snapshots', None)
    frame.pop('snapshot_policy', None)
    frame.pop('public_body_compaction', None)
    target_dir = Path(frames_dir)
    snapshot_refs: dict[str, dict[str, Any]] = {}

    def snapshot(value: Any, json_path: str) -> dict[str, Any]:
        ref = _write_snapshot_ref(value, frame=frame, frames_dir=target_dir, json_path=json_path)
        snapshot_refs[json_path] = ref
        return ref

    _prune_artifact_projections(frame)
    artifacts_frame = frame.get('artifacts') if isinstance(frame.get('artifacts'), dict) else {}
    if artifacts_frame:
        dossiers = artifacts_frame.get('dossiers') if isinstance(artifacts_frame.get('dossiers'), Mapping) else None
        if dossiers:
            artifacts_frame['dossiers_snapshot_ref'] = snapshot(dossiers, 'artifacts.dossiers')
            artifacts_frame.pop('dossiers', None)

    request_frame = frame.get('request') if isinstance(frame.get('request'), dict) else {}
    if request_frame:
        _externalize_child_ref(
            request_frame,
            'input',
            json_path='request.input',
            snapshot=snapshot,
            min_size_bytes=_LEDGER_LARGE_CONTRACT_LIMIT_BYTES,
        )
        _externalize_child_ref(
            request_frame,
            'context_candidates',
            json_path='request.context_candidates',
            snapshot=snapshot,
            min_size_bytes=1,
        )
        ghost_preview = request_frame.get('ghost_preview')
        if (
            ghost_preview not in (None, '', [], {})
            and _json_size_bytes(ghost_preview) >= _LEDGER_LARGE_CONTRACT_LIMIT_BYTES
        ):
            request_frame['ghost_preview_snapshot_ref'] = snapshot(
                ghost_preview,
                'request.ghost_preview',
            )
            compact_ghost_preview = _compact_request_ghost_preview(ghost_preview)
            if compact_ghost_preview:
                request_frame['ghost_preview'] = compact_ghost_preview
            else:
                request_frame.pop('ghost_preview', None)

    runtime = frame.get('runtime') if isinstance(frame.get('runtime'), Mapping) else None
    if runtime:
        frame['runtime_snapshot_ref'] = snapshot(_runtime_snapshot_payload(runtime, snapshot=snapshot), 'runtime')
        compact_runtime = _compact_runtime_for_ledger(runtime, snapshot=snapshot)
        if compact_runtime:
            frame['runtime'] = compact_runtime
        else:
            frame.pop('runtime', None)

    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), dict) else {}
    if current_state:
        current_runtime = current_state.get('runtime') if isinstance(current_state.get('runtime'), Mapping) else None
        if current_runtime:
            current_state['runtime_snapshot_ref'] = snapshot(
                _runtime_snapshot_payload(current_runtime, snapshot=snapshot),
                'current_state.runtime',
            )
            current_state.pop('runtime', None)
        current_late_fill = current_state.get('late_fill') if isinstance(current_state.get('late_fill'), Mapping) else None
        if current_late_fill:
            current_state['late_fill_snapshot_ref'] = snapshot(
                _late_fill_snapshot_payload(
                    current_late_fill,
                    snapshot=snapshot,
                    base_json_path='late_fill',
                ),
                'current_state.late_fill',
            )
            current_state.pop('late_fill', None)
        current_work_tree = current_state.get('work_tree') if isinstance(current_state.get('work_tree'), Mapping) else None
        if current_work_tree:
            current_state['work_tree_snapshot_ref'] = snapshot(current_work_tree, 'current_state.work_tree')
            current_state.pop('work_tree', None)

    working_frame = frame.get('working_frame') if isinstance(frame.get('working_frame'), Mapping) else None
    if working_frame:
        frame['working_frame_snapshot_ref'] = snapshot(
            _working_frame_snapshot_payload(working_frame, snapshot=snapshot),
            'working_frame',
        )
        frame.pop('working_frame', None)

    planning = frame.get('planning') if isinstance(frame.get('planning'), dict) else {}
    if planning:
        intent_contract = (
            planning.get('intent_contract')
            if isinstance(planning.get('intent_contract'), Mapping)
            else None
        )
        if intent_contract and _json_size_bytes(intent_contract) > _LEDGER_LARGE_CONTRACT_LIMIT_BYTES:
            intent_contract_ref = snapshot(intent_contract, 'planning.intent_contract')
            planning['intent_contract_snapshot_ref'] = intent_contract_ref
            planning['intent_contract'] = _compact_contract_for_ledger(
                intent_contract,
                snapshot_ref=intent_contract_ref,
            )
        context_contract = (
            planning.get('context_contract')
            if isinstance(planning.get('context_contract'), Mapping)
            else None
        )
        if context_contract and _json_size_bytes(context_contract) > _LEDGER_LARGE_CONTRACT_LIMIT_BYTES:
            context_contract_ref = snapshot(context_contract, 'planning.context_contract')
            planning['context_contract_snapshot_ref'] = context_contract_ref
            planning['context_contract'] = _compact_contract_for_ledger(
                context_contract,
                snapshot_ref=context_contract_ref,
            )
        request_phase_graph = (
            planning.get('request_phase_graph')
            if isinstance(planning.get('request_phase_graph'), Mapping)
            else None
        )
        if request_phase_graph:
            planning['request_phase_graph_snapshot_ref'] = snapshot(
                request_phase_graph,
                'planning.request_phase_graph',
            )
            planning.pop('request_phase_graph', None)
        work_tree = (
            planning.get('work_tree')
            if isinstance(planning.get('work_tree'), Mapping)
            else None
        )
        if work_tree:
            planning['work_tree_snapshot_ref'] = snapshot(work_tree, 'planning.work_tree')
            planning.pop('work_tree', None)
        artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), dict) else {}
        artifact_flow_graph = (
            artifact_flow.get('request_phase_graph')
            if isinstance(artifact_flow.get('request_phase_graph'), Mapping)
            else None
        )
        if artifact_flow_graph:
            artifact_flow['request_phase_graph_snapshot_ref'] = snapshot(
                artifact_flow_graph,
                'planning.artifact_flow.request_phase_graph',
            )
            artifact_flow.pop('request_phase_graph', None)
        artifact_flow_work_tree = (
            artifact_flow.get('work_tree')
            if isinstance(artifact_flow.get('work_tree'), Mapping)
            else None
        )
        if artifact_flow_work_tree:
            artifact_flow['work_tree_snapshot_ref'] = snapshot(
                artifact_flow_work_tree,
                'planning.artifact_flow.work_tree',
            )
            artifact_flow.pop('work_tree', None)

    late_fill = frame.get('late_fill') if isinstance(frame.get('late_fill'), Mapping) else None
    if late_fill:
        late_fill_ref = snapshot(
            _late_fill_snapshot_payload(late_fill, snapshot=snapshot, base_json_path='late_fill'),
            'late_fill',
        )
        frame['late_fill_snapshot_ref'] = late_fill_ref
        frame['late_fill'] = _compact_late_fill_for_ledger(late_fill, snapshot_ref=late_fill_ref)

    _compact_aggregate_public_collections(
        frame,
        snapshot=snapshot,
    )
    for public_key in ('artifacts', 'batch', 'current_state', 'output', 'request'):
        public_value = frame.get(public_key)
        if isinstance(public_value, Mapping):
            frame[public_key] = _compact_public_body_sidecars(
                public_value,
                json_path=public_key,
                snapshot=snapshot,
            )
    if isinstance(frame.get('error'), Mapping):
        frame['error'] = _compact_public_body_sidecars(
            frame.get('error'),
            json_path='error',
            snapshot=snapshot,
        )

    parent_manifest = parent_snapshot_manifest
    if parent_manifest is None and isinstance(parent_frame, Mapping):
        parent_manifest = _snapshot_items_from_frame(parent_frame)
    _apply_snapshot_manifest_to_frame(
        frame,
        snapshot_refs=snapshot_refs,
        parent_frame=parent_frame,
        parent_manifest=parent_manifest,
    )
    final_line_size_bytes = _ledger_frame_line_size_bytes(frame)
    if final_line_size_bytes > _RESPONSE_WIRE_INLINE_BUDGET_BYTES:
        raise ValueError(
            'Compact response-frame ledger row exceeds the aggregate 8 MiB budget '
            f'({final_line_size_bytes} bytes).'
        )
    return _json_safe(frame)


def _write_response_frame_index(
    enriched_frame: Mapping[str, Any],
    *,
    ledger_path: Path,
    line_offset: int,
    byte_offset: int | None = None,
    line_length: int | None = None,
    ledger_size_bytes: int | None = None,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
    effective_snapshot_manifest: Optional[Mapping[str, Any]] = None,
) -> None:
    response_id = _frame_response_id(enriched_frame)
    if not response_id:
        return
    target = _index_path(frames_dir=frames_dir, index_name=index_name)
    index_payload: dict[str, Any] = {}
    prior_index_loaded = False
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding='utf-8'))
            if isinstance(raw, dict):
                index_payload = raw
                prior_index_loaded = True
        except (OSError, json.JSONDecodeError):
            index_payload = {}
    pre_append_size = _coerce_frame_sequence(byte_offset, 0) or 0
    if pre_append_size == 0:
        # A newly created or deliberately truncated ledger starts a new index
        # coverage epoch; entries from an older ledger must not survive it.
        index_payload = {}
        prior_index_loaded = False
    responses = index_payload.get('responses') if isinstance(index_payload.get('responses'), dict) else {}
    prior_ledger_size = _coerce_frame_sequence(index_payload.get('ledger_size_bytes'))
    prior_verified_size = _coerce_frame_sequence(
        index_payload.get('response_map_verified_size_bytes')
    )
    prior_verified_count = _coerce_frame_sequence(
        index_payload.get('response_map_entry_count')
    )
    prior_verified_digest = str(index_payload.get('response_map_digest') or '').strip()
    prior_response_map_verified = bool(
        prior_index_loaded
        and _response_frame_index_targets_ledger(index_payload, ledger_path)
        and prior_ledger_size == pre_append_size
        and prior_verified_size == pre_append_size
        and prior_verified_count == len(responses)
        and prior_verified_digest
        and prior_verified_digest == _response_map_digest(responses)
    )
    response_map_can_advance = pre_append_size == 0 or prior_response_map_verified
    responses[response_id] = {
        'response_id': response_id,
        'latest_frame_id': str(enriched_frame.get('frame_id') or '').strip(),
        'latest_frame_sequence': enriched_frame.get('frame_sequence'),
        'frame_relation': _json_safe(enriched_frame.get('frame_relation'))
        if isinstance(enriched_frame.get('frame_relation'), Mapping)
        else None,
        'ledger_path': str(ledger_path),
        'ledger_name': ledger_path.name,
        'line_offset': line_offset,
        'byte_offset': byte_offset,
        'line_length': line_length,
        'ledger_size_bytes': ledger_size_bytes,
        'current_lifecycle_state': (
            enriched_frame.get('current_state', {}).get('lifecycle_state')
            if isinstance(enriched_frame.get('current_state'), Mapping)
            else None
        ),
        'updated_at': _json_safe(enriched_frame.get('current_state', {})).get('updated_at')
        if isinstance(enriched_frame.get('current_state'), Mapping)
        else None,
    }
    if effective_snapshot_manifest:
        responses[response_id]['effective_snapshot_manifest'] = _json_safe(effective_snapshot_manifest)
    index_payload['kind'] = 'ollmo.response_frame_current_index'
    index_payload['version'] = 2
    index_payload['ledger_path'] = str(ledger_path)
    index_payload['ledger_name'] = ledger_path.name
    index_payload['ledger_line_count'] = max(
        _coerce_frame_sequence(index_payload.get('ledger_line_count'), 0) or 0,
        line_offset + 1,
    )
    if ledger_size_bytes is not None:
        index_payload['ledger_size_bytes'] = ledger_size_bytes
        index_payload['ledger_line_count_verified_size_bytes'] = ledger_size_bytes
    index_payload['responses'] = responses
    if response_map_can_advance and ledger_size_bytes is not None:
        index_payload['response_map_verified_size_bytes'] = ledger_size_bytes
        index_payload['response_map_entry_count'] = len(responses)
        index_payload['response_map_digest'] = _response_map_digest(responses)
    else:
        index_payload.pop('response_map_verified_size_bytes', None)
        index_payload.pop('response_map_entry_count', None)
        index_payload.pop('response_map_digest', None)
    encoded_index = (
        json.dumps(
            _json_safe(index_payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + '\n'
    ).encode('utf-8')
    _atomic_replace_file_bytes(target, encoded_index)


def load_response_frame_index(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
) -> dict[str, Any]:
    target = _index_path(frames_dir=frames_dir, index_name=index_name)
    if not target.exists():
        return {'ok': False, 'missing': True, 'index_path': str(target), 'responses': {}}
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            'ok': False,
            'corrupt': True,
            'index_path': str(target),
            'error': {'code': 'response_frame_index_corrupt', 'message': str(exc)},
            'responses': {},
        }
    responses = payload.get('responses') if isinstance(payload, Mapping) and isinstance(payload.get('responses'), dict) else {}
    result = {
        'ok': True,
        'index_path': str(target),
        'responses': responses,
        'ledger_path': payload.get('ledger_path') if isinstance(payload, Mapping) else None,
        'ledger_name': payload.get('ledger_name') if isinstance(payload, Mapping) else None,
        'ledger_line_count': payload.get('ledger_line_count') if isinstance(payload, Mapping) else None,
        'ledger_size_bytes': payload.get('ledger_size_bytes') if isinstance(payload, Mapping) else None,
        'ledger_line_count_verified_size_bytes': (
            payload.get('ledger_line_count_verified_size_bytes')
            if isinstance(payload, Mapping)
            else None
        ),
        'response_map_verified_size_bytes': (
            payload.get('response_map_verified_size_bytes')
            if isinstance(payload, Mapping)
            else None
        ),
        'response_map_entry_count': (
            payload.get('response_map_entry_count')
            if isinstance(payload, Mapping)
            else None
        ),
        'response_map_digest': (
            payload.get('response_map_digest')
            if isinstance(payload, Mapping)
            else None
        ),
    }
    if result['ledger_line_count'] in (None, ''):
        offsets = [
            _coerce_frame_sequence(entry.get('line_offset'))
            for entry in responses.values()
            if isinstance(entry, Mapping)
        ]
        offsets = [offset for offset in offsets if offset is not None]
        if offsets:
            result['ledger_line_count'] = max(offsets) + 1
    return result


def _response_frame_file_state(path: Path) -> Optional[dict[str, int]]:
    """Return the identity and mutation-sensitive state of one resolved file."""

    try:
        stat_result = path.stat()
    except OSError:
        return None
    return {
        'device': int(stat_result.st_dev),
        'inode': int(stat_result.st_ino),
        'size_bytes': int(stat_result.st_size),
        'mtime_ns': int(stat_result.st_mtime_ns),
        'ctime_ns': int(stat_result.st_ctime_ns),
    }


def inspect_response_frame_recovery_cache(
    response_id: str,
    *,
    frames_dir: Path | str,
    expected_ledger_path: Path | str,
    expected_ledger_state: Mapping[str, Any],
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> dict[str, Any]:
    """Inspect whether a cached recovery still represents current ledger truth.

    This helper is deliberately read-only. It reports a new checkpoint after
    an unrelated append-only tail, but the response registry remains the
    caller's responsibility.
    """

    normalized_id = str(response_id or '').strip()
    target = _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)

    def result(
        cache_reusable: bool,
        reason: str,
        *,
        ledger_state: Optional[Mapping[str, int]] = None,
        checkpoint_ledger_state: Optional[Mapping[str, int]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'kind': 'ollmo.response_frame_recovery_cache_inspection',
            'response_id': normalized_id,
            'cache_reusable': bool(cache_reusable),
            'reason': reason,
            'ledger_path': str(target),
        }
        if isinstance(ledger_state, Mapping):
            payload['ledger_state'] = {
                key: int(ledger_state[key])
                for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
                if key in ledger_state
            }
        if isinstance(checkpoint_ledger_state, Mapping):
            payload['checkpoint_ledger_state'] = {
                key: int(checkpoint_ledger_state[key])
                for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
                if key in checkpoint_ledger_state
            }
        return payload

    raw_expected_path = str(expected_ledger_path or '').strip()
    if not raw_expected_path or not isinstance(expected_ledger_state, Mapping):
        return result(False, 'missing_expected_ledger_state')
    try:
        if Path(raw_expected_path).resolve() != target.resolve():
            return result(False, 'ledger_path_mismatch')
    except (OSError, RuntimeError):
        return result(False, 'ledger_path_unresolvable')

    current = _response_frame_file_state(target)
    if not isinstance(current, Mapping):
        return result(False, 'ledger_unavailable')
    current_token = {
        key: int(current[key])
        for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
    }
    try:
        expected_size = int(expected_ledger_state.get('size_bytes'))
        expected_mtime = int(expected_ledger_state.get('mtime_ns'))
        expected_device = int(expected_ledger_state.get('device'))
        expected_inode = int(expected_ledger_state.get('inode'))
    except (TypeError, ValueError):
        return result(
            False,
            'invalid_expected_ledger_state',
            ledger_state=current_token,
        )
    if (
        current_token['device'] != expected_device
        or current_token['inode'] != expected_inode
    ):
        return result(
            False,
            'ledger_identity_changed',
            ledger_state=current_token,
        )
    if current_token['size_bytes'] < expected_size:
        return result(False, 'ledger_truncated', ledger_state=current_token)
    if current_token['size_bytes'] == expected_size:
        return result(
            current_token['mtime_ns'] == expected_mtime,
            'unchanged'
            if current_token['mtime_ns'] == expected_mtime
            else 'same_size_ledger_mutated',
            ledger_state=current_token,
        )

    try:
        with target.open('rb') as handle:
            opened = os.fstat(handle.fileno())
            opened_token = {
                'size_bytes': int(opened.st_size),
                'mtime_ns': int(opened.st_mtime_ns),
                'device': int(opened.st_dev),
                'inode': int(opened.st_ino),
            }
            if opened_token != current_token:
                return result(
                    False,
                    'ledger_changed_before_tail_scan',
                    ledger_state=current_token,
                )
            if expected_size > 0:
                handle.seek(expected_size - 1)
                if handle.read(1) != b'\n':
                    return result(
                        False,
                        'expected_tail_not_line_aligned',
                        ledger_state=current_token,
                    )
            handle.seek(expected_size)
            while handle.tell() < current_token['size_bytes']:
                raw_line = handle.readline()
                if not raw_line or not raw_line.endswith(b'\n') or raw_line.isspace():
                    return result(
                        False,
                        'invalid_appended_ledger_line',
                        ledger_state=current_token,
                    )
                try:
                    frame = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return result(
                        False,
                        'malformed_appended_ledger_line',
                        ledger_state=current_token,
                    )
                if not isinstance(frame, Mapping):
                    return result(
                        False,
                        'non_mapping_appended_ledger_line',
                        ledger_state=current_token,
                    )
                if response_frame_ledger_record_response_id(frame) == normalized_id:
                    return result(
                        False,
                        'target_response_appended',
                        ledger_state=current_token,
                    )
            closed = os.fstat(handle.fileno())
            closed_token = {
                'size_bytes': int(closed.st_size),
                'mtime_ns': int(closed.st_mtime_ns),
                'device': int(closed.st_dev),
                'inode': int(closed.st_ino),
            }
    except OSError:
        return result(False, 'ledger_tail_scan_failed', ledger_state=current_token)
    after = _response_frame_file_state(target)
    after_token = (
        {
            key: int(after[key])
            for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
        }
        if isinstance(after, Mapping)
        else None
    )
    if closed_token != current_token or after_token != current_token:
        return result(
            False,
            'ledger_changed_during_tail_scan',
            ledger_state=after_token,
        )
    return result(
        True,
        'unrelated_append_only_tail',
        ledger_state=current_token,
        checkpoint_ledger_state=current_token,
    )


def _stable_response_frame_index_snapshot(
    index_path: Path,
) -> tuple[Optional[bytes], Optional[dict[str, int]], Optional[dict[str, Any]]]:
    """Read the bounded index only when its start/end file state is identical."""

    before = _response_frame_file_state(index_path)
    if before is None:
        return None, None, {
            'code': 'response_frame_index_missing',
            'message': 'Response-frame index does not exist.',
        }
    try:
        with index_path.open('rb') as handle:
            raw = handle.read()
    except OSError as exc:
        return None, None, {
            'code': 'response_frame_index_read_failed',
            'message': str(exc),
        }
    after = _response_frame_file_state(index_path)
    if before != after:
        return None, None, {
            'code': 'response_frame_index_moved',
            'message': 'Response-frame index changed while it was being read.',
        }
    return raw, before, None


def _scan_response_frame_ledger_index_truth(
    ledger_path: Path,
) -> dict[str, Any]:
    """Stream the ledger once and retain only each response's latest coordinates."""

    path_before = _response_frame_file_state(ledger_path)
    if path_before is None:
        return {
            'ok': False,
            'error': {
                'code': 'response_frame_ledger_missing',
                'message': 'Response-frame ledger does not exist.',
            },
        }
    latest_by_response: dict[str, dict[str, Any]] = {}
    line_count = 0
    byte_offset = 0
    ledger_hasher = hashlib.sha256()
    epoch_anchor: dict[str, Any] = {}
    try:
        with ledger_path.open('rb') as handle:
            opened_stat = os.fstat(handle.fileno())
            opened_state = {
                'device': int(opened_stat.st_dev),
                'inode': int(opened_stat.st_ino),
                'size_bytes': int(opened_stat.st_size),
                'mtime_ns': int(opened_stat.st_mtime_ns),
                'ctime_ns': int(opened_stat.st_ctime_ns),
            }
            if opened_state != path_before:
                return {
                    'ok': False,
                    'error': {
                        'code': 'response_frame_ledger_moved',
                        'message': 'Response-frame ledger changed before its scan began.',
                    },
                }
            while True:
                raw_line = handle.readline()
                if not raw_line:
                    break
                ledger_hasher.update(raw_line)
                line_offset = line_count
                line_count += 1
                line_length = len(raw_line)
                if raw_line.isspace():
                    return {
                        'ok': False,
                        'error': {
                            'code': 'response_frame_ledger_malformed',
                            'message': 'Response-frame ledger contains a blank physical line.',
                            'line_offset': line_offset,
                            'byte_offset': byte_offset,
                        },
                    }
                try:
                    frame = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    return {
                        'ok': False,
                        'error': {
                            'code': 'response_frame_ledger_malformed',
                            'message': str(exc),
                            'line_offset': line_offset,
                            'byte_offset': byte_offset,
                        },
                    }
                if not isinstance(frame, dict):
                    return {
                        'ok': False,
                        'error': {
                            'code': 'response_frame_ledger_malformed',
                            'message': 'Response-frame ledger line is not a JSON object.',
                            'line_offset': line_offset,
                            'byte_offset': byte_offset,
                        },
                    }
                response_id = _frame_response_id(frame)
                frame_id = str(frame.get('frame_id') or '').strip()
                frame_sequence = frame.get('frame_sequence')
                if (
                    str(frame.get('kind') or '').strip() != 'ollmo.response_frame'
                    or not response_id
                    or not frame_id
                    or not isinstance(frame_sequence, int)
                    or isinstance(frame_sequence, bool)
                ):
                    return {
                        'ok': False,
                        'error': {
                            'code': 'response_frame_ledger_malformed',
                            'message': 'Ledger line lacks canonical response-frame identity.',
                            'line_offset': line_offset,
                            'byte_offset': byte_offset,
                        },
                    }
                if line_offset == 0:
                    epoch_anchor = {
                        'response_id': response_id,
                        'frame_id': frame_id,
                        'frame_sequence': frame_sequence,
                        'source_frame_sha256': hashlib.sha256(raw_line).hexdigest(),
                    }
                latest_by_response[response_id] = {
                    'latest_frame_id': frame_id,
                    'latest_frame_sequence': frame_sequence,
                    'line_offset': line_offset,
                    'byte_offset': byte_offset,
                    'line_length': line_length,
                    'source_frame_sha256': hashlib.sha256(raw_line).hexdigest(),
                }
                byte_offset += line_length
                del frame
            closed_stat = os.fstat(handle.fileno())
            closed_state = {
                'device': int(closed_stat.st_dev),
                'inode': int(closed_stat.st_ino),
                'size_bytes': int(closed_stat.st_size),
                'mtime_ns': int(closed_stat.st_mtime_ns),
                'ctime_ns': int(closed_stat.st_ctime_ns),
            }
    except OSError as exc:
        return {
            'ok': False,
            'error': {
                'code': 'response_frame_ledger_read_failed',
                'message': str(exc),
            },
        }
    path_after = _response_frame_file_state(ledger_path)
    if (
        opened_state != closed_state
        or opened_state != path_after
        or byte_offset != opened_state['size_bytes']
    ):
        return {
            'ok': False,
            'error': {
                'code': 'response_frame_ledger_moved',
                'message': 'Response-frame ledger changed during its scan.',
            },
        }
    return {
        'ok': True,
        'latest_by_response': latest_by_response,
        'ledger_state': opened_state,
        'ledger_line_count': line_count,
        'ledger_sha256': ledger_hasher.hexdigest(),
        'epoch_anchor': epoch_anchor,
    }


def _response_frame_attestation_failure(
    code: str,
    message: str,
    *,
    ledger_path: Path,
    index_path: Path,
    **details: Any,
) -> dict[str, Any]:
    error = {'code': code, 'message': message}
    error.update(_json_safe(details))
    return {
        'ok': False,
        'status': 'rejected',
        'changed': False,
        'ledger_path': str(ledger_path),
        'index_path': str(index_path),
        'error': error,
    }


def _write_attested_response_frame_index(
    index_path: Path,
    payload: Mapping[str, Any],
    *,
    expected_index_state: Mapping[str, int],
    expected_index_digest: str,
    ledger_path: Path,
    expected_ledger_state: Mapping[str, int],
) -> Optional[str]:
    """Atomically replace the index only while both evidence files stay fixed."""

    try:
        existing_mode = int(index_path.stat().st_mode) & 0o777
    except OSError:
        return 'index'
    fd, temp_name = tempfile.mkstemp(
        prefix=f'.{index_path.name}.',
        suffix='.tmp',
        dir=str(index_path.parent),
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, existing_mode)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if _response_frame_file_state(ledger_path) != dict(expected_ledger_state):
            return 'ledger'
        current_raw, current_index_state, current_error = _stable_response_frame_index_snapshot(index_path)
        if (
            current_error is not None
            or current_index_state != dict(expected_index_state)
            or current_raw is None
            or hashlib.sha256(current_raw).hexdigest() != expected_index_digest
        ):
            return 'index'
        os.replace(temp_path, index_path)
        directory_fd = os.open(index_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return None
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def attest_response_frame_index(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
    write: bool = True,
) -> dict[str, Any]:
    """Attest an exact legacy response map before granting v2 absence proof.

    The bounded index is loaded normally, while the potentially large ledger is
    scanned in binary mode one physical line at a time.  Existing response
    entries are never rebuilt or normalized: a complete, coordinate-exact match
    only adds the v2 coverage fields at the top level.
    """

    ledger_path = _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)
    index_path = _index_path(frames_dir=frames_dir, index_name=index_name)
    with _RESPONSE_FRAME_APPEND_LOCK:
        raw_index, index_start_state, index_read_error = _stable_response_frame_index_snapshot(
            index_path
        )
        if index_read_error is not None or raw_index is None or index_start_state is None:
            error = index_read_error or {
                'code': 'response_frame_index_read_failed',
                'message': 'Response-frame index could not be read.',
            }
            return _response_frame_attestation_failure(
                str(error.get('code') or 'response_frame_index_read_failed'),
                str(error.get('message') or 'Response-frame index could not be read.'),
                ledger_path=ledger_path,
                index_path=index_path,
            )
        index_start_digest = hashlib.sha256(raw_index).hexdigest()
        try:
            index_payload = json.loads(raw_index)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _response_frame_attestation_failure(
                'response_frame_index_corrupt',
                str(exc),
                ledger_path=ledger_path,
                index_path=index_path,
            )
        finally:
            del raw_index
        if not isinstance(index_payload, dict):
            return _response_frame_attestation_failure(
                'response_frame_index_corrupt',
                'Response-frame index is not a JSON object.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        if str(index_payload.get('kind') or '').strip() != 'ollmo.response_frame_current_index':
            return _response_frame_attestation_failure(
                'response_frame_index_schema_mismatch',
                'Response-frame index kind is not supported for attestation.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        version = index_payload.get('version')
        if not isinstance(version, int) or isinstance(version, bool) or version not in {1, 2}:
            return _response_frame_attestation_failure(
                'response_frame_index_schema_mismatch',
                'Only response-frame index versions 1 and 2 can be attested.',
                ledger_path=ledger_path,
                index_path=index_path,
                version=index_payload.get('version'),
            )
        responses = index_payload.get('responses')
        if not isinstance(responses, dict):
            return _response_frame_attestation_failure(
                'response_frame_index_incomplete',
                'Response-frame index has no response mapping.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        if (
            not str(index_payload.get('ledger_path') or '').strip()
            or not str(index_payload.get('ledger_name') or '').strip()
            or not _response_frame_index_targets_ledger(index_payload, ledger_path)
        ):
            return _response_frame_attestation_failure(
                'response_frame_index_ledger_mismatch',
                'Response-frame index targets a different ledger.',
                ledger_path=ledger_path,
                index_path=index_path,
            )

        scan = _scan_response_frame_ledger_index_truth(ledger_path)
        if scan.get('ok') is not True:
            scan_error = scan.get('error') if isinstance(scan.get('error'), Mapping) else {}
            return _response_frame_attestation_failure(
                str(scan_error.get('code') or 'response_frame_ledger_read_failed'),
                str(scan_error.get('message') or 'Response-frame ledger could not be scanned.'),
                ledger_path=ledger_path,
                index_path=index_path,
                **{
                    key: value
                    for key, value in scan_error.items()
                    if key not in {'code', 'message'}
                },
            )
        ledger_state = scan.get('ledger_state')
        scanned_responses = scan.get('latest_by_response')
        line_count = scan.get('ledger_line_count')
        if not isinstance(ledger_state, Mapping) or not isinstance(scanned_responses, Mapping):
            return _response_frame_attestation_failure(
                'response_frame_ledger_incomplete',
                'Ledger scan did not produce complete attestation evidence.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        indexed_ledger_size = index_payload.get('ledger_size_bytes')
        if (
            not isinstance(indexed_ledger_size, int)
            or isinstance(indexed_ledger_size, bool)
            or indexed_ledger_size != ledger_state.get('size_bytes')
        ):
            return _response_frame_attestation_failure(
                'response_frame_index_ledger_size_mismatch',
                'Index ledger size does not match the scanned ledger epoch.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        indexed_line_count = index_payload.get('ledger_line_count')
        if (
            not isinstance(indexed_line_count, int)
            or isinstance(indexed_line_count, bool)
            or indexed_line_count != line_count
        ):
            return _response_frame_attestation_failure(
                'response_frame_index_line_count_mismatch',
                'Index ledger line count does not match the complete scan.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        verified_line_count_size = index_payload.get(
            'ledger_line_count_verified_size_bytes'
        )
        if (
            verified_line_count_size not in (None, '')
            and (
                not isinstance(verified_line_count_size, int)
                or isinstance(verified_line_count_size, bool)
                or verified_line_count_size != ledger_state.get('size_bytes')
            )
        ):
            return _response_frame_attestation_failure(
                'response_frame_index_line_count_epoch_mismatch',
                'Index line-count evidence belongs to another ledger epoch.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        indexed_response_ids = set(responses)
        scanned_response_ids = set(scanned_responses)
        if indexed_response_ids != scanned_response_ids:
            return _response_frame_attestation_failure(
                'response_frame_index_response_set_mismatch',
                'Index response ids do not exactly match the complete ledger scan.',
                ledger_path=ledger_path,
                index_path=index_path,
                missing_from_index=sorted(scanned_response_ids - indexed_response_ids),
                absent_from_ledger=sorted(indexed_response_ids - scanned_response_ids),
            )
        coordinate_keys = (
            'latest_frame_id',
            'latest_frame_sequence',
            'byte_offset',
            'line_length',
        )
        for response_id in sorted(scanned_response_ids):
            entry = responses.get(response_id)
            scanned_entry = scanned_responses.get(response_id)
            if not isinstance(entry, Mapping) or not isinstance(scanned_entry, Mapping):
                return _response_frame_attestation_failure(
                    'response_frame_index_latest_entry_mismatch',
                    'Index response entry is missing or malformed.',
                    ledger_path=ledger_path,
                    index_path=index_path,
                    response_id=response_id,
                )
            mismatched_keys = []
            for key in coordinate_keys:
                indexed_value = entry.get(key)
                scanned_value = scanned_entry.get(key)
                if type(indexed_value) is not type(scanned_value) or indexed_value != scanned_value:
                    mismatched_keys.append(key)
            if entry.get('response_id') != response_id:
                mismatched_keys.append('response_id')
            if mismatched_keys:
                return _response_frame_attestation_failure(
                    'response_frame_index_latest_entry_mismatch',
                    'Index latest-frame coordinates do not match the ledger.',
                    ledger_path=ledger_path,
                    index_path=index_path,
                    response_id=response_id,
                    mismatched_keys=mismatched_keys,
                )

        index_end_raw, index_end_state, index_end_error = _stable_response_frame_index_snapshot(
            index_path
        )
        if (
            index_end_error is not None
            or index_end_raw is None
            or index_end_state != index_start_state
            or hashlib.sha256(index_end_raw).hexdigest() != index_start_digest
        ):
            return _response_frame_attestation_failure(
                'response_frame_index_moved',
                'Response-frame index changed during attestation.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        del index_end_raw
        if _response_frame_file_state(ledger_path) != dict(ledger_state):
            return _response_frame_attestation_failure(
                'response_frame_ledger_moved',
                'Response-frame ledger changed after its attestation scan.',
                ledger_path=ledger_path,
                index_path=index_path,
            )

        attested_payload = dict(index_payload)
        attested_payload['version'] = 2
        attested_payload['response_map_verified_size_bytes'] = ledger_state.get('size_bytes')
        attested_payload['response_map_entry_count'] = len(responses)
        attested_payload['response_map_digest'] = _response_map_digest(responses)
        changed = attested_payload != index_payload
        result = {
            'ok': True,
            'status': 'attested' if write and changed else 'verified',
            'changed': bool(write and changed),
            'would_change': changed,
            'ledger_path': str(ledger_path),
            'index_path': str(index_path),
            'ledger_size_bytes': ledger_state.get('size_bytes'),
            'ledger_line_count': line_count,
            'response_map_entry_count': len(responses),
            'response_map_digest': attested_payload['response_map_digest'],
            'index_start_sha256': index_start_digest,
        }
        if not write or not changed:
            return result
        moved = _write_attested_response_frame_index(
            index_path,
            attested_payload,
            expected_index_state=index_start_state,
            expected_index_digest=index_start_digest,
            ledger_path=ledger_path,
            expected_ledger_state=ledger_state,
        )
        if moved:
            return _response_frame_attestation_failure(
                f'response_frame_{moved}_moved',
                f'Response-frame {moved} changed before atomic index replacement.',
                ledger_path=ledger_path,
                index_path=index_path,
            )
        result['index_end_sha256'] = hashlib.sha256(index_path.read_bytes()).hexdigest()
        return result


def verify_response_frame_epoch(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
    allow_relocated: bool = False,
) -> dict[str, Any]:
    """Verify one complete response-frame epoch without mutating its index.

    Archived epochs commonly retain the relative ``ledger_path`` written when
    they were active.  When ``allow_relocated`` is true, this function accepts
    that stale parent path only after the complete index mapping, every latest
    frame coordinate, and the ledger bytes have been verified against the
    files in the explicitly supplied ``frames_dir``.  The returned index state
    is a rebound in-memory copy suitable for the bounded observation loaders;
    neither source file is rewritten.
    """

    resolved_frames_dir = Path(frames_dir)
    ledger_path = _ledger_path(
        frames_dir=resolved_frames_dir,
        ledger_name=ledger_name,
    )
    index_path = _index_path(
        frames_dir=resolved_frames_dir,
        index_name=index_name,
    )

    def rejected(code: str, message: str, **details: Any) -> dict[str, Any]:
        error = {'code': code, 'message': message}
        error.update(_json_safe(details))
        return {
            'ok': False,
            'status': 'rejected',
            'runtime_effect': 'none',
            'frames_dir': str(resolved_frames_dir),
            'ledger_path': str(ledger_path),
            'index_path': str(index_path),
            'error': error,
        }

    with _RESPONSE_FRAME_APPEND_LOCK:
        raw_index, index_start_state, index_error = _stable_response_frame_index_snapshot(
            index_path
        )
        if index_error is not None or raw_index is None or index_start_state is None:
            error = index_error or {
                'code': 'response_frame_index_read_failed',
                'message': 'Response-frame index could not be read.',
            }
            return rejected(
                str(error.get('code') or 'response_frame_index_read_failed'),
                str(error.get('message') or 'Response-frame index could not be read.'),
            )
        index_sha256 = hashlib.sha256(raw_index).hexdigest()
        try:
            index_payload = json.loads(raw_index)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return rejected('response_frame_index_corrupt', str(exc))
        finally:
            del raw_index
        if not isinstance(index_payload, dict):
            return rejected(
                'response_frame_index_corrupt',
                'Response-frame index is not a JSON object.',
            )
        if str(index_payload.get('kind') or '').strip() != 'ollmo.response_frame_current_index':
            return rejected(
                'response_frame_index_schema_mismatch',
                'Response-frame index kind is not supported.',
            )
        version = index_payload.get('version')
        if not isinstance(version, int) or isinstance(version, bool) or version != 2:
            return rejected(
                'response_frame_index_unverified',
                'A fully attested version 2 response-frame index is required.',
                version=version,
            )
        responses = index_payload.get('responses')
        if not isinstance(responses, dict):
            return rejected(
                'response_frame_index_incomplete',
                'Response-frame index has no response mapping.',
            )

        indexed_name = str(index_payload.get('ledger_name') or '').strip()
        indexed_path_text = str(index_payload.get('ledger_path') or '').strip()
        if indexed_name != ledger_path.name:
            return rejected(
                'response_frame_index_ledger_mismatch',
                'Response-frame index names a different ledger.',
                indexed_ledger_name=indexed_name,
            )
        if not indexed_path_text:
            return rejected(
                'response_frame_index_ledger_mismatch',
                'Response-frame index does not bind a ledger path.',
            )
        try:
            stored_path_matches = Path(indexed_path_text).resolve() == ledger_path.resolve()
        except OSError:
            stored_path_matches = False
        relocated = not stored_path_matches
        if relocated and (
            not allow_relocated or Path(indexed_path_text).name != ledger_path.name
        ):
            return rejected(
                'response_frame_index_ledger_mismatch',
                'Response-frame index targets a different ledger location.',
                indexed_ledger_path=indexed_path_text,
                relocated=relocated,
            )

        scan = _scan_response_frame_ledger_index_truth(ledger_path)
        if scan.get('ok') is not True:
            scan_error = scan.get('error') if isinstance(scan.get('error'), Mapping) else {}
            return rejected(
                str(scan_error.get('code') or 'response_frame_ledger_read_failed'),
                str(scan_error.get('message') or 'Response-frame ledger could not be scanned.'),
                **{
                    key: value
                    for key, value in scan_error.items()
                    if key not in {'code', 'message'}
                },
            )
        ledger_state = scan.get('ledger_state')
        scanned_responses = scan.get('latest_by_response')
        ledger_line_count = scan.get('ledger_line_count')
        ledger_sha256 = str(scan.get('ledger_sha256') or '').strip()
        epoch_anchor = (
            dict(scan.get('epoch_anchor'))
            if isinstance(scan.get('epoch_anchor'), Mapping)
            else {}
        )
        if not isinstance(ledger_state, Mapping) or not isinstance(scanned_responses, Mapping):
            return rejected(
                'response_frame_ledger_incomplete',
                'Ledger scan did not produce complete verification evidence.',
            )

        expected_size = ledger_state.get('size_bytes')
        integer_bindings = {
            'ledger_size_bytes': expected_size,
            'ledger_line_count': ledger_line_count,
            'ledger_line_count_verified_size_bytes': expected_size,
            'response_map_verified_size_bytes': expected_size,
            'response_map_entry_count': len(responses),
        }
        for key, expected in integer_bindings.items():
            actual = index_payload.get(key)
            if type(actual) is not type(expected) or actual != expected:
                return rejected(
                    'response_frame_index_epoch_mismatch',
                    f'Response-frame index field {key} does not match ledger truth.',
                    field=key,
                    indexed_value=actual,
                    verified_value=expected,
                )
        response_map_digest = str(index_payload.get('response_map_digest') or '').strip()
        verified_response_map_digest = _response_map_digest(responses)
        if not response_map_digest or response_map_digest != verified_response_map_digest:
            return rejected(
                'response_frame_index_response_map_digest_mismatch',
                'Response-frame index response-map digest is missing or invalid.',
                indexed_digest=response_map_digest,
                verified_digest=verified_response_map_digest,
            )

        indexed_response_ids = set(responses)
        scanned_response_ids = set(scanned_responses)
        if indexed_response_ids != scanned_response_ids:
            return rejected(
                'response_frame_index_response_set_mismatch',
                'Index response ids do not exactly match the complete ledger scan.',
                missing_from_index=sorted(scanned_response_ids - indexed_response_ids),
                absent_from_ledger=sorted(indexed_response_ids - scanned_response_ids),
            )

        coordinate_keys = (
            'latest_frame_id',
            'latest_frame_sequence',
            'line_offset',
            'byte_offset',
            'line_length',
        )
        rebound_responses: dict[str, Any] = {}
        source_frame_sha256_by_response: dict[str, str] = {}
        for response_id in sorted(scanned_response_ids):
            entry = responses.get(response_id)
            scanned_entry = scanned_responses.get(response_id)
            if not isinstance(entry, Mapping) or not isinstance(scanned_entry, Mapping):
                return rejected(
                    'response_frame_index_latest_entry_mismatch',
                    'Index response entry is missing or malformed.',
                    response_id=response_id,
                )
            mismatched_keys = [
                key
                for key in coordinate_keys
                if type(entry.get(key)) is not type(scanned_entry.get(key))
                or entry.get(key) != scanned_entry.get(key)
            ]
            if entry.get('response_id') != response_id:
                mismatched_keys.append('response_id')
            if mismatched_keys:
                return rejected(
                    'response_frame_index_latest_entry_mismatch',
                    'Index latest-frame coordinates do not match the ledger.',
                    response_id=response_id,
                    mismatched_keys=mismatched_keys,
                )
            rebound_entry = dict(entry)
            rebound_entry['ledger_path'] = str(ledger_path)
            rebound_entry['ledger_name'] = ledger_path.name
            rebound_responses[response_id] = rebound_entry
            source_frame_sha256_by_response[response_id] = str(
                scanned_entry.get('source_frame_sha256') or ''
            ).strip()

        end_raw, index_end_state, index_end_error = _stable_response_frame_index_snapshot(
            index_path
        )
        if (
            index_end_error is not None
            or end_raw is None
            or index_end_state != index_start_state
            or hashlib.sha256(end_raw).hexdigest() != index_sha256
        ):
            return rejected(
                'response_frame_index_moved',
                'Response-frame index changed during epoch verification.',
            )
        del end_raw
        if _response_frame_file_state(ledger_path) != dict(ledger_state):
            return rejected(
                'response_frame_ledger_moved',
                'Response-frame ledger changed after its verification scan.',
            )

        rebound_index = dict(index_payload)
        rebound_index.update({
            'ok': True,
            'index_path': str(index_path),
            'ledger_path': str(ledger_path),
            'ledger_name': ledger_path.name,
            'responses': rebound_responses,
            # The caller-facing copy intentionally has rebound ledger paths;
            # bind its derived acceleration digest to those in-memory bytes.
            # ``response_map_digest`` returned below remains the exact source
            # index digest and neither source file is changed.
            'response_map_digest': _response_map_digest(rebound_responses),
        })
        return {
            'ok': True,
            'status': 'verified',
            'runtime_effect': 'none',
            'frames_dir': str(resolved_frames_dir),
            'ledger_path': str(ledger_path),
            'index_path': str(index_path),
            'stored_ledger_path': indexed_path_text,
            'relocated': relocated,
            'index_sha256': index_sha256,
            'ledger_sha256': ledger_sha256,
            'ledger_size_bytes': expected_size,
            'ledger_line_count': ledger_line_count,
            'response_map_entry_count': len(responses),
            'response_map_digest': response_map_digest,
            'ledger_file_state': dict(ledger_state),
            'index_file_state': dict(index_start_state),
            'epoch_anchor': epoch_anchor,
            'source_frame_sha256_by_response': source_frame_sha256_by_response,
            'index_state': rebound_index,
        }


def build_response_frame(
    response_payload: Mapping[str, Any],
    *,
    request_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build an auditable frozen frame from a canonical Responses payload."""

    if not isinstance(response_payload, Mapping):
        raise TypeError('response_payload must be a mapping')

    output_items = response_payload.get('output') if isinstance(response_payload.get('output'), list) else []
    request_frame = _build_request_frame(request_payload, response_payload)
    output_artifacts = response_payload.get('artifacts') if isinstance(response_payload.get('artifacts'), list) else []
    canonical_artifacts = build_canonical_response_artifacts(dict(response_payload))
    if canonical_artifacts:
        output_artifacts = merge_canonical_response_artifacts(output_artifacts, canonical_artifacts)
    output_artifacts = _reconcile_output_artifacts_with_late_fill_branch_paths(
        response_payload,
        output_artifacts,
        canonical_artifacts,
    )
    if output_artifacts != response_payload.get('artifacts'):
        response_payload = {**dict(response_payload), 'artifacts': output_artifacts}
    input_artifacts = response_payload.get('input_artifacts') if isinstance(response_payload.get('input_artifacts'), list) else []
    reference_artifacts = response_payload.get('reference_artifacts') if isinstance(response_payload.get('reference_artifacts'), list) else []
    if not input_artifacts and isinstance(request_frame.get('input_artifacts'), list):
        input_artifacts = request_frame.get('input_artifacts') or []
    if not reference_artifacts and isinstance(request_frame.get('reference_artifacts'), list):
        reference_artifacts = request_frame.get('reference_artifacts') or []
    artifact_identity_canonicalization = canonicalize_duplicate_artifact_refs(output_artifacts)
    artifact_identity = _artifact_identity_summary(artifact_identity_canonicalization)
    if artifact_identity.get('canonicalization_required'):
        output_artifacts = artifact_identity_canonicalization.get('artifacts') or output_artifacts
    error_frame = _build_error_frame(response_payload)
    request_source = request_payload if isinstance(request_payload, Mapping) else request_frame
    artifact_flow_plan = build_artifact_flow_plan(
        input_artifacts,
        output_artifacts,
        reference_artifacts=reference_artifacts,
        request_payload=request_frame,
        response_payload=response_payload,
    )
    artifact_dossiers = build_artifact_dossier_index(
        input_artifacts=input_artifacts,
        reference_artifacts=reference_artifacts,
        output_artifacts=output_artifacts,
        response_payload=response_payload,
    )
    control_snapshot = build_control_snapshot(request_source, response_payload)
    runtime_payload = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    runtime_request_phase_graph = (
        runtime_payload.get('request_phase_graph')
        if isinstance(runtime_payload.get('request_phase_graph'), Mapping)
        else None
    )
    request_phase_graph = (
        runtime_request_phase_graph
        if runtime_request_phase_graph
        else build_request_phase_graph(
            str(response_payload.get('output_text') or request_frame.get('prompt') or request_frame.get('input') or ''),
            intent_prompt=extract_responses_current_turn_prompt(request_source),
            request_payload=request_source,
            response_payload=response_payload,
        )
    )
    plan_output_slots = artifact_flow_plan.get('output_slots') if isinstance(artifact_flow_plan.get('output_slots'), list) else []
    payload_output_slots = response_payload.get('output_slots') if isinstance(response_payload.get('output_slots'), list) else []
    artifact_flow_authoritative = bool(artifact_flow_plan.get('authoritative')) or (
        str(artifact_flow_plan.get('work_tree_source') or '').strip() == 'runtime_owned'
    )
    if artifact_flow_authoritative and plan_output_slots:
        output_slots = plan_output_slots
    elif _output_slots_should_refresh_from_plan(payload_output_slots, plan_output_slots):
        output_slots = plan_output_slots
    else:
        output_slots = payload_output_slots or plan_output_slots
    output_slots = _apply_truth_guard_to_output_slots(response_payload, output_slots)
    output_slots = _reconcile_output_slots_with_late_fill_artifacts(
        response_payload,
        output_slots,
        output_artifacts,
    )
    if output_slots:
        artifact_flow_plan = dict(artifact_flow_plan)
        artifact_flow_plan['output_slots'] = output_slots
        artifact_flow_plan['steps'] = [
            {
                **dict(step),
                'status': next(
                    (
                        slot.get('status')
                        for slot in output_slots
                        if isinstance(slot, Mapping)
                        and str(slot.get('slot_id') or '').strip()
                        == str(step.get('slot_id') or '').strip()
                    ),
                    step.get('status'),
                ),
                'lifecycle': next(
                    (
                        slot.get('lifecycle')
                        for slot in output_slots
                        if isinstance(slot, Mapping)
                        and str(slot.get('slot_id') or '').strip()
                        == str(step.get('slot_id') or '').strip()
                    ),
                    step.get('lifecycle'),
                ),
            }
            if isinstance(step, Mapping) and str(step.get('slot_id') or '').strip()
            else step
            for step in (artifact_flow_plan.get('steps') or [])
        ] if isinstance(artifact_flow_plan.get('steps'), list) else []
    output_branches = _reconcile_output_branches_with_slots(
        response_payload.get('output_branches'),
        output_slots,
    )
    outputs = build_canonical_outputs(
        response_payload,
        output_slots=output_slots,
        artifacts=output_artifacts,
        output_branches=output_branches,
    )
    if not outputs and isinstance(response_payload.get('outputs'), list):
        outputs = response_payload.get('outputs')
    outputs = _apply_artifact_identity_to_outputs(
        outputs,
        artifact_identity=artifact_identity,
        canonical_artifacts=output_artifacts,
    )
    public_output_artifacts = filter_public_response_artifacts(
        response_payload,
        output_artifacts,
        outputs=outputs,
    )
    public_output_text = _select_public_output_text(response_payload, outputs)
    output_items = _project_output_items_text(output_items, public_output_text)
    working_frame = (
        response_payload.get('working_frame')
        if isinstance(response_payload.get('working_frame'), Mapping)
        else build_working_frame(
            request_payload=request_payload,
            response_payload=response_payload,
            freeze=True,
        )
    )

    frame: dict[str, Any] = {
        'frame_version': RESPONSE_FRAME_VERSION,
        'kind': 'ollmo.response_frame',
        'response_id': str(response_payload.get('id') or '').strip() or None,
        'status': str(response_payload.get('status') or '').strip() or None,
        'object': str(response_payload.get('object') or '').strip() or None,
        'target': _select_frame_keys(response_payload, _TARGET_FRAME_KEYS),
        'route': _select_frame_keys(response_payload, _ROUTE_FRAME_KEYS),
        'runtime': _build_runtime_frame(response_payload),
        'request': request_frame,
        'artifacts': {
            'input': _json_safe(input_artifacts),
            'reference': _json_safe(reference_artifacts),
            'output': _json_safe(public_output_artifacts),
            'dossiers': _json_safe(artifact_dossiers),
            **({'identity': _json_safe(artifact_identity)} if artifact_identity else {}),
        },
        'memory_delta': _json_safe(response_payload.get('memory_delta')) if isinstance(response_payload.get('memory_delta'), Mapping) else {},
        'planning': {
            'intent_contract': _json_safe(
                working_frame.get('intent_contract')
                if isinstance(working_frame, Mapping) and isinstance(working_frame.get('intent_contract'), Mapping)
                else {}
            ),
            'context_contract': _json_safe(
                working_frame.get('context_contract')
                if isinstance(working_frame, Mapping) and isinstance(working_frame.get('context_contract'), Mapping)
                else {}
            ),
            'artifact_flow': _json_safe(artifact_flow_plan),
            'request_phase_graph': _json_safe(request_phase_graph),
            'work_tree': _json_safe(artifact_flow_plan.get('work_tree') or {}),
        },
        'output': {
            'text': public_output_text,
            'item_count': len(outputs) if outputs else len(output_items),
            'outputs': _json_safe(outputs),
            **({'artifact_identity': _json_safe(artifact_identity)} if artifact_identity else {}),
        },
        'current_state': _build_current_state_frame(
            response_payload,
            output_slots=output_slots,
            output_branches=output_branches,
            outputs=outputs,
            work_tree=artifact_flow_plan.get('work_tree') if isinstance(artifact_flow_plan.get('work_tree'), Mapping) else {},
            public_output_text=public_output_text,
        ),
        'working_frame': _json_safe(working_frame),
    }
    if isinstance(response_payload.get('late_fill'), Mapping):
        frame['late_fill'] = _json_safe(response_payload.get('late_fill'))
    if isinstance(response_payload.get('frame_relation'), Mapping):
        frame['frame_relation'] = _json_safe(response_payload.get('frame_relation'))
    if control_snapshot:
        frame['controls'] = _json_safe(control_snapshot)
    if error_frame:
        frame['error'] = error_frame
    if response_payload.get('batch_count') not in (None, ''):
        frame['batch'] = {
            'count': response_payload.get('batch_count'),
            'prompts': _json_safe(response_payload.get('batch_prompts') or []),
        }
    return _json_safe(frame)


def attach_response_frame(
    response_payload: Mapping[str, Any],
    *,
    request_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a payload copy with its current response-frame snapshot attached."""

    if not isinstance(response_payload, Mapping):
        raise TypeError('response_payload must be a mapping')
    payload = dict(response_payload)
    payload['response_frame'] = build_response_frame(payload, request_payload=request_payload)
    return payload


def _persist_response_frame_locked(
    response_frame: Mapping[str, Any],
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    expected_parent_frame_id: str | None = None,
    expected_parent_frame_sequence: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Append while the caller holds ``_RESPONSE_FRAME_APPEND_LOCK``."""

    if not isinstance(response_frame, Mapping) or not response_frame:
        raise ValueError('response_frame must be a non-empty mapping')
    target_dir = Path(frames_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (str(ledger_name or '').strip() or DEFAULT_RESPONSE_FRAME_LEDGER)
    response_id = _frame_response_id(response_frame)
    existing_frames, index_state = _index_parent_frame_stub(
        response_id,
        frames_dir=target_dir,
        ledger_name=ledger_name,
    ) if response_id else ([], {})
    if (
        not existing_frames
        and target.exists()
        and (not index_state.get('ok') or index_state.get('_response_index_entry_stale'))
    ):
        existing_frames, _errors = _iter_ledger_frames(target)
    index_responses = index_state.get('responses') if isinstance(index_state.get('responses'), Mapping) else {}
    index_entry = index_responses.get(response_id) if isinstance(index_responses, Mapping) else None
    parent_frame: Optional[dict[str, Any]] = None
    parent_snapshot_manifest: dict[str, dict[str, Any]] = {}
    if isinstance(index_entry, Mapping):
        indexed_manifest = index_entry.get('effective_snapshot_manifest')
        if isinstance(indexed_manifest, Mapping):
            parent_snapshot_manifest = {
                str(path): _json_safe(ref)
                for path, ref in indexed_manifest.items()
                if isinstance(ref, Mapping) and ref.get('sha256')
            }
        indexed_ledger_path = Path(
            str(index_entry.get('ledger_path') or '').strip()
            or _ledger_path(frames_dir=target_dir, ledger_name=ledger_name)
        )
        indexed_frame, _indexed_error = _read_indexed_response_frame(
            index_entry,
            ledger_path=indexed_ledger_path,
            response_id=response_id,
        )
        if indexed_frame:
            parent_frame = indexed_frame
    if parent_frame is None and existing_frames and not target.exists():
        candidate_parent = existing_frames[-1]
        if isinstance(candidate_parent, dict):
            parent_frame = candidate_parent
            parent_snapshot_manifest = _snapshot_items_from_frame(candidate_parent)
    if target.exists() and (parent_frame is None or not parent_snapshot_manifest):
        ledger_frames, _errors = _iter_ledger_frames(target)
        response_frames = [
            frame
            for frame in ledger_frames
            if _frame_response_id(frame) == response_id
            and str(frame.get('kind') or '').strip() == 'ollmo.response_frame'
        ]
        if response_frames:
            expanded_parents = _expand_snapshot_manifests_for_frames(response_frames)
            parent_frame = response_frames[-1]
            parent_snapshot_manifest = _snapshot_items_from_frame(expanded_parents[-1])
    if expected_parent_frame_id is not None:
        expected_parent_id = str(expected_parent_frame_id or '').strip()
        if not response_id:
            raise ValueError('response_frame must include response_id for parent CAS')
        if not expected_parent_id:
            raise ValueError('expected_parent_frame_id must be non-empty')
        current_parent_id = str(
            parent_frame.get('frame_id') if isinstance(parent_frame, Mapping) else ''
        ).strip()
        current_parent_sequence = _coerce_frame_sequence(
            parent_frame.get('frame_sequence') if isinstance(parent_frame, Mapping) else None
        )
        expected_parent_sequence = _coerce_frame_sequence(
            expected_parent_frame_sequence
        )
        if (
            current_parent_id != expected_parent_id
            or (
                expected_parent_sequence is not None
                and current_parent_sequence != expected_parent_sequence
            )
        ):
            raise ResponseFrameParentCASMismatch(
                response_id=response_id,
                expected_parent_frame_id=expected_parent_id,
                current_parent_frame_id=current_parent_id or None,
                expected_parent_frame_sequence=expected_parent_sequence,
                current_parent_frame_sequence=current_parent_sequence,
            )
    enriched_frame = enrich_response_frame_metadata(
        response_frame,
        previous_frames=existing_frames,
        force_append_sequence=True,
    )
    ledger_frame = compact_response_frame_for_ledger(
        enriched_frame,
        frames_dir=target_dir,
        parent_frame=parent_frame,
        parent_snapshot_manifest=parent_snapshot_manifest,
    )
    effective_snapshot_manifest = _effective_snapshot_manifest(
        ledger_frame,
        parent_manifest=parent_snapshot_manifest,
    )
    line_offset = _index_next_line_offset(index_state, target) if index_state else None
    if line_offset is None:
        if target.exists():
            try:
                line_offset = sum(1 for _line in target.open('r', encoding='utf-8'))
            except OSError:
                line_offset = len(existing_frames)
        else:
            line_offset = 0
    byte_offset = target.stat().st_size if target.exists() else 0
    encoded_line = json.dumps(_json_safe(ledger_frame), ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n'
    with target.open('ab') as handle:
        handle.write(encoded_line)
        handle.flush()
        os.fsync(handle.fileno())
    _write_response_frame_index(
        ledger_frame,
        ledger_path=target,
        line_offset=line_offset,
        byte_offset=byte_offset,
        line_length=len(encoded_line),
        ledger_size_bytes=byte_offset + len(encoded_line),
        frames_dir=target_dir,
        effective_snapshot_manifest=effective_snapshot_manifest,
    )
    return target, enriched_frame


def persist_response_frame(
    response_frame: Mapping[str, Any],
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> Path:
    """Append a response frame under the shared service lock and return its path."""

    with _RESPONSE_FRAME_APPEND_LOCK:
        target, _enriched_frame = _persist_response_frame_locked(
            response_frame,
            frames_dir=frames_dir,
            ledger_name=ledger_name,
        )
    return target


def append_response_frame_with_parent_cas(
    response_frame: Mapping[str, Any],
    *,
    expected_parent_frame_id: str,
    expected_parent_frame_sequence: int | None = None,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> dict[str, Any]:
    """Atomically append a successor only if its exact durable parent is current."""

    with _RESPONSE_FRAME_APPEND_LOCK:
        target, enriched_frame = _persist_response_frame_locked(
            response_frame,
            frames_dir=frames_dir,
            ledger_name=ledger_name,
            expected_parent_frame_id=expected_parent_frame_id,
            expected_parent_frame_sequence=expected_parent_frame_sequence,
        )
    return {
        'status': 'appended',
        'ledger_path': target,
        'response_frame': enriched_frame,
    }


def response_payload_from_frame(
    response_frame: Mapping[str, Any],
    *,
    frames_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Reconstruct a truthful current-state payload from one frozen frame."""

    if not isinstance(response_frame, Mapping) or not response_frame:
        return {}
    trusted_manifest = _trusted_snapshot_manifest(response_frame)
    response_id = _frame_response_id(response_frame)
    if frames_dir is not None:
        restored_frame = _restore_public_body_sidecars(
            response_frame,
            frames_dir=frames_dir,
            _trusted_manifest=trusted_manifest,
            _response_id=response_id,
        )
        if isinstance(restored_frame, Mapping):
            response_frame = restored_frame
    current_state = (
        response_frame.get('current_state')
        if isinstance(response_frame.get('current_state'), Mapping)
        else {}
    )
    payload = dict(current_state)
    if response_id:
        payload['id'] = response_id
    if not payload.get('object'):
        payload['object'] = str(response_frame.get('object') or 'response').strip() or 'response'
    if not payload.get('status'):
        payload['status'] = str(response_frame.get('status') or '').strip() or 'completed'
    if payload.get('error') in (None, '', [], {}) and isinstance(
        response_frame.get('error'), Mapping
    ):
        payload['error'] = _json_safe(response_frame.get('error'))
    target = response_frame.get('target') if isinstance(response_frame.get('target'), Mapping) else {}
    for source_key, target_key in (
        ('instance_id', 'instance_id'),
        ('model', 'model'),
        ('backend', 'backend'),
        ('capability', 'capability'),
        ('mode', 'mode'),
    ):
        if payload.get(target_key) in (None, '', [], {}) and target.get(source_key) not in (None, '', [], {}):
            payload[target_key] = target.get(source_key)
    route = (
        response_frame.get('route')
        if isinstance(response_frame.get('route'), Mapping)
        else {}
    )
    for key in _ROUTE_FRAME_KEYS:
        if (
            payload.get(key) in (None, '', [], {})
            and route.get(key) not in (None, '', [], {})
        ):
            payload[key] = _json_safe(route.get(key))
    output = response_frame.get('output') if isinstance(response_frame.get('output'), Mapping) else {}
    if payload.get('output_text') in (None, '', [], {}):
        payload['output_text'] = str(output.get('text') or '').strip()
    if payload.get('outputs') in (None, '', [], {}) and isinstance(output.get('outputs'), list):
        payload['outputs'] = _json_safe(output.get('outputs'))
    artifacts = response_frame.get('artifacts') if isinstance(response_frame.get('artifacts'), Mapping) else {}
    if payload.get('artifacts') in (None, '', [], {}) and isinstance(artifacts.get('output'), list):
        payload['artifacts'] = _json_safe(artifacts.get('output'))
    if payload.get('input_artifacts') in (None, '', [], {}) and isinstance(artifacts.get('input'), list):
        payload['input_artifacts'] = _json_safe(artifacts.get('input'))
    if payload.get('reference_artifacts') in (None, '', [], {}) and isinstance(artifacts.get('reference'), list):
        payload['reference_artifacts'] = _json_safe(artifacts.get('reference'))
    planning = response_frame.get('planning') if isinstance(response_frame.get('planning'), Mapping) else {}
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), Mapping) else {}
    if payload.get('output_slots') in (None, '', [], {}) and isinstance(artifact_flow.get('output_slots'), list):
        payload['output_slots'] = _json_safe(artifact_flow.get('output_slots'))
    if payload.get('output_branches') in (None, '', [], {}) and isinstance(artifact_flow.get('output_branches'), list):
        payload['output_branches'] = _json_safe(artifact_flow.get('output_branches'))
    if isinstance(payload.get('output_slots'), list) and payload.get('output_slots'):
        canonical_outputs = build_canonical_outputs(
            payload,
            output_slots=payload.get('output_slots'),
            artifacts=payload.get('artifacts'),
            output_branches=payload.get('output_branches'),
            prefer_existing_output_values=True,
        )
        if canonical_outputs:
            payload['outputs'] = _json_safe(canonical_outputs)
    if payload.get('work_tree') in (None, '', [], {}) and isinstance(planning.get('work_tree'), Mapping):
        payload['work_tree'] = _json_safe(planning.get('work_tree'))
    if payload.get('work_tree') in (None, '', [], {}) and frames_dir is not None:
        for ref, expected_path, content_path in (
            (
                current_state.get('work_tree_snapshot_ref'),
                'current_state.work_tree',
                'current_state.work_tree',
            ),
            (
                planning.get('work_tree_snapshot_ref'),
                'planning.work_tree',
                'planning.work_tree',
            ),
            (
                artifact_flow.get('work_tree_snapshot_ref'),
                'planning.artifact_flow.work_tree',
                'planning.artifact_flow.work_tree',
            ),
        ):
            restored_work_tree = _read_manifest_authorized_snapshot_payload(
                ref,
                frames_dir=frames_dir,
                trusted_manifest=trusted_manifest,
                response_id=response_id,
                expected_json_path=expected_path,
                content_json_path=content_path,
            )
            if isinstance(restored_work_tree, Mapping) and restored_work_tree:
                payload['work_tree'] = restored_work_tree
                break
    if payload.get('late_fill') in (None, '', [], {}) and frames_dir is not None:
        late_fill_frame = response_frame.get('late_fill') if isinstance(response_frame.get('late_fill'), Mapping) else {}
        for ref, expected_path, content_path in (
            (
                current_state.get('late_fill_snapshot_ref'),
                'current_state.late_fill',
                'late_fill',
            ),
            (response_frame.get('late_fill_snapshot_ref'), 'late_fill', 'late_fill'),
            (late_fill_frame.get('full_snapshot_ref'), 'late_fill', 'late_fill'),
            (late_fill_frame.get('review_snapshot_ref'), 'late_fill', 'late_fill'),
        ):
            restored_late_fill = _read_manifest_authorized_snapshot_payload(
                ref,
                frames_dir=frames_dir,
                trusted_manifest=trusted_manifest,
                response_id=response_id,
                expected_json_path=expected_path,
                content_json_path=content_path,
            )
            if isinstance(restored_late_fill, Mapping) and restored_late_fill:
                payload['late_fill'] = _expand_late_fill_snapshot_children(
                    restored_late_fill,
                    frames_dir=frames_dir,
                    trusted_manifest=trusted_manifest,
                    response_id=response_id,
                )
                break
    if payload.get('late_fill') in (None, '', [], {}) and isinstance(response_frame.get('late_fill'), Mapping):
        payload['late_fill'] = _json_safe(response_frame.get('late_fill'))
    if frames_dir is not None:
        for ref, expected_path, content_path in (
            (
                current_state.get('runtime_snapshot_ref'),
                'current_state.runtime',
                'runtime',
            ),
            (response_frame.get('runtime_snapshot_ref'), 'runtime', 'runtime'),
        ):
            restored_runtime = _read_manifest_authorized_snapshot_payload(
                ref,
                frames_dir=frames_dir,
                trusted_manifest=trusted_manifest,
                response_id=response_id,
                expected_json_path=expected_path,
                content_json_path=content_path,
            )
            if not isinstance(restored_runtime, Mapping) or not restored_runtime:
                continue
            compact_runtime = (
                payload.get('runtime')
                if isinstance(payload.get('runtime'), Mapping)
                else response_frame.get('runtime')
                if isinstance(response_frame.get('runtime'), Mapping)
                else {}
            )
            payload['runtime'] = _json_safe({**dict(restored_runtime), **dict(compact_runtime)})
            break
    if payload.get('runtime') in (None, '', [], {}) and isinstance(response_frame.get('runtime'), Mapping):
        payload['runtime'] = _json_safe(response_frame.get('runtime'))
    canonical_artifacts = build_canonical_response_artifacts(payload)
    if canonical_artifacts:
        existing_artifacts = payload.get('artifacts') if isinstance(payload.get('artifacts'), list) else []
        payload['artifacts'] = merge_canonical_response_artifacts(existing_artifacts, canonical_artifacts)
    if isinstance(payload.get('artifacts'), list):
        payload['artifacts'] = _reconcile_output_artifacts_with_late_fill_branch_paths(
            payload,
            payload.get('artifacts'),
            canonical_artifacts,
        )
    if isinstance(payload.get('output_slots'), list) and payload.get('output_slots'):
        payload['output_slots'] = _reconcile_output_slots_with_late_fill_artifacts(
            payload,
            payload.get('output_slots'),
            payload.get('artifacts'),
        )
        output_branches = payload.get('output_branches') if isinstance(payload.get('output_branches'), list) else []
        payload['output_branches'] = _reconcile_output_branches_with_slots(
            output_branches,
            payload.get('output_slots'),
        )
        canonical_outputs = build_canonical_outputs(
            payload,
            output_slots=payload.get('output_slots'),
            artifacts=payload.get('artifacts'),
            output_branches=payload.get('output_branches'),
            prefer_existing_output_values=True,
        )
        if canonical_outputs:
            payload['outputs'] = _json_safe(canonical_outputs)
            public_artifacts = filter_public_response_artifacts(
                payload,
                payload.get('artifacts'),
                outputs=canonical_outputs,
            )
            payload['artifacts'] = _json_safe(public_artifacts)
    payload['durability'] = {
        'source': 'response_frame_ledger',
        'recovered': True,
        'frame_id': str(response_frame.get('frame_id') or '').strip() or None,
        'frame_sequence': response_frame.get('frame_sequence'),
        'frame_relation': _json_safe(response_frame.get('frame_relation'))
        if isinstance(response_frame.get('frame_relation'), Mapping)
        else None,
    }
    safe_payload = {
        key: value
        for key, value in _json_safe(payload).items()
        if value not in (None, '', [], {})
    }
    safe_payload['response_frame'] = _json_safe(response_frame)
    return safe_payload


def _canonical_response_payload_from_frame(
    response_frame: Mapping[str, Any],
    *,
    frames_dir: Path | str,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    integrity_error = _public_body_snapshot_integrity_error(
        response_frame,
        frames_dir=frames_dir,
    )
    if integrity_error:
        return {}, integrity_error
    return response_payload_from_frame(
        response_frame,
        frames_dir=frames_dir,
    ), None


def load_response_frame_records(
    response_id: str,
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
) -> dict[str, Any]:
    """Load persisted frozen frames for one response id from the JSONL ledger."""

    normalized_id = str(response_id or '').strip()
    target = _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)
    frames, errors = _iter_ledger_frames(target)
    matched_frames = [
        frame
        for frame in frames
        if _frame_response_id(frame) == normalized_id
        and str(frame.get('kind') or '').strip() == 'ollmo.response_frame'
    ]
    return {
        'response_id': normalized_id,
        'frames': matched_frames,
        'errors': errors,
        'ledger_path': str(target),
        'ledger_exists': target.exists(),
    }


def _read_indexed_response_frame(
    index_entry: Mapping[str, Any],
    *,
    ledger_path: Path,
    response_id: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    byte_offset = _coerce_frame_sequence(index_entry.get('byte_offset'))
    if byte_offset is None or byte_offset < 0:
        return None, {'code': 'response_frame_index_missing_byte_offset', 'message': 'Index entry has no byte offset.'}
    if not ledger_path.exists():
        return None, {'code': 'response_frame_ledger_missing', 'message': 'Indexed ledger path does not exist.'}
    try:
        with ledger_path.open('rb') as handle:
            handle.seek(byte_offset)
            raw_line = handle.readline()
    except OSError as exc:
        return None, {'code': 'response_frame_index_read_failed', 'message': str(exc)}
    if not raw_line:
        return None, {'code': 'response_frame_index_empty_line', 'message': 'Index byte offset did not contain a ledger line.'}
    try:
        frame = json.loads(raw_line.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, {'code': 'response_frame_index_corrupt_line', 'message': str(exc)}
    if not isinstance(frame, dict):
        return None, {'code': 'response_frame_index_invalid_frame', 'message': 'Indexed ledger line is not a frame object.'}
    latest_frame_id = str(index_entry.get('latest_frame_id') or '').strip()
    latest_frame_sequence = index_entry.get('latest_frame_sequence')
    if _frame_response_id(frame) != response_id:
        return None, {'code': 'response_frame_index_response_mismatch', 'message': 'Indexed frame belongs to another response.'}
    if latest_frame_id and str(frame.get('frame_id') or '').strip() != latest_frame_id:
        return None, {'code': 'response_frame_index_frame_mismatch', 'message': 'Indexed frame id does not match the ledger line.'}
    if latest_frame_sequence not in (None, '') and frame.get('frame_sequence') != latest_frame_sequence:
        return None, {'code': 'response_frame_index_sequence_mismatch', 'message': 'Indexed frame sequence does not match the ledger line.'}
    return frame, None


def _expand_sidecar_child_refs(
    value: Any,
    *,
    frames_dir: Path | str,
    depth: int = 0,
    max_depth: int = 8,
    strict_child_refs: bool = False,
    current_json_path: str = '',
    source_response_id: str = '',
    trusted_child_refs: Optional[Mapping[str, Any]] = None,
) -> Any:
    if depth >= max_depth:
        return _json_safe(value)
    if isinstance(value, dict):
        payload: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            derived_child_key = (
                key[: -len('_snapshot_ref')]
                if key.endswith('_snapshot_ref')
                else ''
            )
            expected_child_path = _json_child_path(
                current_json_path,
                derived_child_key,
            )
            child_authority = (
                trusted_child_refs.get(expected_child_path)
                if isinstance(trusted_child_refs, Mapping)
                else None
            )
            authorized_child = _authorized_sidecar_child_ref(
                child,
                child_authority,
                expected_json_path=expected_child_path,
                expected_child_key=derived_child_key,
                source_response_id=source_response_id,
            )
            summarized_child_binding = authorized_child is not None
            generated_child_binding = bool(
                derived_child_key
                and isinstance(child, Mapping)
                and child.get('sidecar_child_ref') is True
                and str(child.get('sidecar_child_key') or '').strip()
                == derived_child_key
                and str(child.get('sidecar_parent_json_path') or '').strip()
                == current_json_path
                and str(child.get('json_path') or '').strip()
                == expected_child_path
                and (
                    not source_response_id
                    or str(child.get('source_response_id') or '').strip()
                    == source_response_id
                )
            ) or summarized_child_binding
            is_expandable_child_ref = bool(
                derived_child_key
                and isinstance(child, Mapping)
                and str(child.get('kind') or '').strip() == _LEDGER_SNAPSHOT_REF_KIND
                and (
                    generated_child_binding
                    # Runtime snapshot compaction also writes explicit child
                    # refs through ``snapshot(...)``.  Those refs predate the
                    # recursive sidecar marker but carry the same trusted CAS
                    # shape and must be expanded when the parent is restored.
                    or (
                        not strict_child_refs
                        and derived_child_key not in {'full', 'review'}
                    )
                )
            )
            if (
                is_expandable_child_ref
            ):
                expanded_child = _read_snapshot_ref_payload(
                    authorized_child if authorized_child is not None else child,
                    frames_dir=frames_dir,
                    expand_child_refs=True,
                    _expand_depth=depth + 1,
                    _max_expand_depth=max_depth,
                    strict_child_refs=strict_child_refs,
                )
                if expanded_child is not None:
                    child_key = str(
                        child.get('sidecar_child_key')
                        or derived_child_key
                        or ''
                    ).strip()
                    if child_key:
                        payload[child_key] = expanded_child
                        continue
            payload[key] = _expand_sidecar_child_refs(
                child,
                frames_dir=frames_dir,
                depth=depth,
                max_depth=max_depth,
                strict_child_refs=strict_child_refs,
                current_json_path=_json_child_path(current_json_path, key),
                source_response_id=source_response_id,
                trusted_child_refs=trusted_child_refs,
            )
        return _json_safe(payload)
    if isinstance(value, list):
        return [
            _expand_sidecar_child_refs(
                item,
                frames_dir=frames_dir,
                depth=depth,
                max_depth=max_depth,
                strict_child_refs=strict_child_refs,
                current_json_path=_json_child_path(
                    current_json_path,
                    f'[{index}]',
                ),
                source_response_id=source_response_id,
                trusted_child_refs=trusted_child_refs,
            )
            for index, item in enumerate(value)
        ]
    return _json_safe(value)


def _read_snapshot_ref_payload(
    ref: Any,
    *,
    frames_dir: Path | str,
    expand_child_refs: bool = True,
    _expand_depth: int = 0,
    _max_expand_depth: int = 8,
    strict_child_refs: bool = False,
) -> Any:
    if not isinstance(ref, Mapping):
        return None
    raw_path = str(ref.get('path') or '').strip()
    if not raw_path:
        return None
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        return None
    target = Path(frames_dir) / relative_path
    try:
        raw = target.read_bytes()
    except OSError:
        return None
    expected_sha = str(ref.get('sha256') or '').strip()
    if expected_sha:
        actual_sha = hashlib.sha256(raw.rstrip(b'\n')).hexdigest()
        if actual_sha != expected_sha:
            return None
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if expand_child_refs:
        sidecar_manifest = (
            ref.get('sidecar_manifest')
            if isinstance(ref.get('sidecar_manifest'), Mapping)
            else {}
        )
        trusted_child_refs = {
            str(item.get('json_path') or '').strip(): item
            for item in (sidecar_manifest.get('child_refs') or [])
            if isinstance(item, Mapping)
            and str(item.get('json_path') or '').strip()
        }
        payload = _expand_sidecar_child_refs(
            payload,
            frames_dir=frames_dir,
            depth=_expand_depth,
            max_depth=_max_expand_depth,
            strict_child_refs=strict_child_refs,
            current_json_path=str(ref.get('json_path') or '').strip(),
            source_response_id=str(
                ref.get('source_response_id') or ''
            ).strip(),
            trusted_child_refs=trusted_child_refs,
        )
    return _json_safe(payload)


_GRAPH_REBASE_OBSERVATION_GRAPH_KEYS = {
    'applied_graph_patches',
    'applied_graph_rebases',
    'frame_id',
    'graph_patch_lifecycle',
    'graph_patch_lifecycle_results',
    'graph_rebase_lifecycle',
    'graph_rebase_lifecycle_results',
    'graph_rebase_outcomes',
    'graph_rebase_proposals',
    'graph_rebase_reviews',
    'graph_repair_proposals',
    'graph_repair_reviews',
    'graph_version',
    'kind',
    'partial_rebase_outcomes',
    'redraw_scope_ladder_review',
    'response_id',
    'staged_graph_patches',
    'staged_graph_rebases',
    'successor_reopen_requests',
    'successor_rebase_executions',
    'successor_rebase_requests',
}
_GRAPH_REBASE_OBSERVATION_DIAGNOSTIC_KEYS = {
    'applied_graph_patches',
    'applied_graph_rebases',
    'graph_patch_autonomy',
    'graph_patch_enforced_policy_reviews',
    'graph_patch_lifecycle',
    'graph_patch_lifecycle_results',
    'graph_patch_successor_reopen_requests',
    'graph_rebase_autonomy',
    'graph_rebase_enforced_policy_reviews',
    'graph_rebase_lifecycle',
    'graph_rebase_lifecycle_results',
    'graph_rebase_outcomes',
    'partial_rebase_outcomes',
    'redraw_scope_ladder_review',
    'response_time_graph_rebase_candidate',
    'runtime_graph_rebase_candidate_review',
    'runtime_graph_rebase_proposals',
    'runtime_graph_rebase_reviews',
    'runtime_graph_repair_proposal_reviews',
    'runtime_graph_repair_proposals',
    'staged_graph_patches',
    'staged_graph_rebases',
    'successor_reopen_requests',
    'successor_rebase_requests',
    'surface_repair_actionability',
}
_GRAPH_CLOSURE_OBSERVATION_SCALAR_KEYS = (
    'action',
    'closure_status',
    'closure_gap_code',
    'closure_gap_trigger',
    'continuation_required',
    'contract_source',
    'current_phase_id',
    'decision',
    'decision_action',
    'graph_mode',
    'intent_boundary',
    'kind',
    'late_fill_status',
    'materialization_status',
    'needs_external_input',
    'obligation_count',
    'pending_branch_count',
    'reason',
    'recommended_transition',
    'recovery_action',
    'repair_action',
    'repair_needed',
    'repair_scope',
    'semantic_review_recommended_transition',
    'semantic_review_required_count',
    'status',
    'suggested_action',
    'transition',
)
_GRAPH_CLOSURE_OBSERVATION_CHECK_KEYS = (
    'action',
    'branch_id',
    'capability',
    'check_id',
    'check_kind',
    'code',
    'decision',
    'decision_action',
    'evidence',
    'needs_external_input',
    'obligation_id',
    'output_type',
    'phase_id',
    'reason',
    'recommended_transition',
    'recovery_action',
    'repair_action',
    'status',
    'suggested_action',
    'transition',
)
_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT = 32
_GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT = 512
_GRAPH_REBASE_OBSERVATION_MARKERS = (
    b'"applied_graph_rebases"',
    b'"graph_rebase_lifecycle"',
    b'"graph_rebase_outcomes"',
    b'"graph_rebase_partial_successor"',
    b'"graph_rebase_proposals"',
    b'"graph_rebase_reviews"',
    b'"graph_rebase_stage_successor"',
    b'"partial_rebase_outcomes"',
    b'"response_time_graph_rebase_candidate"',
    b'"runtime_graph_rebase_candidate_review"',
    b'"runtime_graph_rebase_proposals"',
    b'"runtime_graph_rebase_reviews"',
    b'"staged_graph_rebases"',
    b'"successor_rebase_executions"',
    b'"successor_rebase_requests"',
)
_GRAPH_REBASE_FRAME_RELATION_MARKERS = (
    b'"graph_rebase_partial_successor"',
    b'"graph_rebase_stage_successor"',
)
_GRAPH_REBASE_OBSERVATION_KEYS = {
    marker.decode('utf-8').strip('"')
    for marker in _GRAPH_REBASE_OBSERVATION_MARKERS
    if marker not in _GRAPH_REBASE_FRAME_RELATION_MARKERS
}
_GRAPH_REBASE_FRAME_RELATION_KINDS = {
    marker.decode('utf-8').strip('"')
    for marker in _GRAPH_REBASE_FRAME_RELATION_MARKERS
}


def _graph_rebase_marker_present(raw: bytes) -> bool:
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    stack = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key or '').strip()
                if key in _GRAPH_REBASE_OBSERVATION_KEYS:
                    return True
                if (
                    key in {'kind', 'relation_kind'}
                    and str(child or '').strip() in _GRAPH_REBASE_FRAME_RELATION_KINDS
                ):
                    return True
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return False


def _read_observation_snapshot_bytes(
    ref: Any,
    *,
    frames_dir: Path | str,
) -> tuple[bytes, Optional[dict[str, Any]]]:
    if not isinstance(ref, Mapping):
        return b'', None
    relative_path = Path(str(ref.get('path') or '').strip())
    if not str(relative_path) or relative_path.is_absolute() or '..' in relative_path.parts:
        return b'', {'code': 'response_frame_snapshot_ref_invalid'}
    target = Path(frames_dir) / relative_path
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return b'', {'code': 'response_frame_snapshot_read_failed', 'message': str(exc)}
    expected_sha = str(ref.get('sha256') or '').strip()
    if expected_sha and hashlib.sha256(raw.rstrip(b'\n')).hexdigest() != expected_sha:
        return b'', {'code': 'response_frame_snapshot_digest_mismatch'}
    return raw, None


def _read_observation_snapshot_payload(
    ref: Any,
    *,
    frames_dir: Path | str,
) -> tuple[Any, Optional[dict[str, Any]]]:
    if not isinstance(ref, Mapping) or not str(ref.get('path') or '').strip():
        return None, None
    raw, snapshot_error = _read_observation_snapshot_bytes(
        ref,
        frames_dir=frames_dir,
    )
    if snapshot_error:
        return None, snapshot_error
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, {
            'code': 'response_frame_snapshot_corrupt',
            'message': str(exc),
        }
    return _json_safe(payload), None


def _observation_snapshot_failure(
    response_id: str,
    snapshot_error: Mapping[str, Any],
    *,
    json_path: str,
) -> dict[str, Any]:
    error = dict(snapshot_error)
    error.setdefault('message', 'A referenced response-frame snapshot could not be verified.')
    error['response_id'] = response_id
    error['json_path'] = json_path
    return {
        'ok': False,
        'status_code': 409,
        'error': _json_safe(error),
    }


def select_graph_rebase_observation_response_ids(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    index_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Select only latest responses containing graph-rebase rollout evidence."""

    current_index = (
        dict(index_state)
        if isinstance(index_state, Mapping)
        else load_response_frame_index(frames_dir=frames_dir)
    )
    indexed_ledger_path = _ledger_path(frames_dir=frames_dir)
    if not _response_frame_index_has_verified_response_map(
        current_index,
        indexed_ledger_path,
    ):
        return {
            'kind': 'ollmo.graph_rebase_observation_selection',
            'runtime_effect': 'none',
            'indexed_response_count': 0,
            'selected_response_ids': [],
            'selected_response_count': 0,
            'scan_error_count': 1,
            'scan_errors': [{
                'code': 'response_frame_index_unverified',
                'message': 'Response-frame index is not completely verified for the current ledger.',
            }],
        }
    responses = (
        current_index.get('responses')
        if isinstance(current_index.get('responses'), Mapping)
        else {}
    )
    selected: list[str] = []
    errors: list[dict[str, Any]] = []
    scanned_snapshot_markers: dict[str, bool] = {}
    for raw_response_id, raw_entry in responses.items():
        response_id = str(raw_response_id or '').strip()
        if not response_id or not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        ledger_path = Path(
            str(entry.get('ledger_path') or '').strip()
            or str(current_index.get('ledger_path') or '').strip()
            or _ledger_path(frames_dir=frames_dir)
        )
        byte_offset = _coerce_frame_sequence(entry.get('byte_offset'))
        if byte_offset is None or byte_offset < 0:
            errors.append({'response_id': response_id, 'code': 'response_frame_index_missing_byte_offset'})
            continue
        try:
            with ledger_path.open('rb') as handle:
                handle.seek(byte_offset)
                raw_frame = handle.readline()
        except OSError as exc:
            errors.append({
                'response_id': response_id,
                'code': 'response_frame_index_read_failed',
                'message': str(exc),
            })
            continue
        try:
            compact_frame = json.loads(raw_frame.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append({
                'response_id': response_id,
                'code': 'response_frame_index_corrupt_line',
                'message': str(exc),
            })
            continue
        relation = (
            compact_frame.get('frame_relation')
            if isinstance(compact_frame, Mapping)
            and isinstance(compact_frame.get('frame_relation'), Mapping)
            else {}
        )
        if str(relation.get('kind') or '').strip() in _GRAPH_REBASE_FRAME_RELATION_KINDS:
            selected.append(response_id)
            continue
        manifest = (
            entry.get('effective_snapshot_manifest')
            if isinstance(entry.get('effective_snapshot_manifest'), Mapping)
            else {}
        )
        relevant_refs = [
            ref
            for path, ref in manifest.items()
            if isinstance(ref, Mapping)
            and (
                path in {
                    'runtime',
                    'current_state.runtime',
                    'planning.request_phase_graph',
                }
                or path.endswith('.request_phase_graph')
                or path.endswith('.developer_diagnostics')
                or 'graph_rebase' in path
            )
        ]
        matched = False
        for ref in relevant_refs:
            ref_identity = str(ref.get('sha256') or ref.get('path') or '').strip()
            if ref_identity and ref_identity in scanned_snapshot_markers:
                if scanned_snapshot_markers[ref_identity]:
                    matched = True
                    break
                continue
            raw_snapshot, snapshot_error = _read_observation_snapshot_bytes(
                ref,
                frames_dir=frames_dir,
            )
            if snapshot_error:
                errors.append({'response_id': response_id, **snapshot_error})
                continue
            marker_present = _graph_rebase_marker_present(raw_snapshot)
            if ref_identity:
                scanned_snapshot_markers[ref_identity] = marker_present
            if marker_present:
                matched = True
                break
        if matched:
            selected.append(response_id)
    return {
        'kind': 'ollmo.graph_rebase_observation_selection',
        'runtime_effect': 'none',
        'indexed_response_count': len(responses),
        'selected_response_ids': selected,
        'selected_response_count': len(selected),
        'scan_error_count': len(errors),
        'scan_errors': errors[:50],
    }


def _observation_snapshot_ref(
    frame: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *paths: str,
) -> dict[str, Any]:
    for path in paths:
        cursor: Any = frame
        for token in path.split('.'):
            cursor = cursor.get(token) if isinstance(cursor, Mapping) else None
        if isinstance(cursor, Mapping) and cursor.get('kind') == _LEDGER_SNAPSHOT_REF_KIND:
            return dict(cursor)
        manifest_ref = manifest.get(path) if isinstance(manifest, Mapping) else None
        if isinstance(manifest_ref, Mapping):
            return dict(manifest_ref)
    return {}


_GRAPH_OBSERVATION_MAPPING_LIMIT = 96
_GRAPH_OBSERVATION_DEPTH_LIMIT = 8
_GRAPH_OBSERVATION_BULK_KEYS = {
    'artifact_payload',
    'artifacts',
    'base_graph',
    'candidate_graph',
    'content_payload',
    'current_state',
    'input',
    'messages',
    'output',
    'outputs',
    'request_payload',
    'route_payload',
    'runtime',
    'source_route_payload',
}


def _bounded_observation_digest(value: Any) -> tuple[str, int]:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _project_bounded_observation_value(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """Keep rollout evidence exact where small and identity-bound where bulky."""

    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if len(value) <= _GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT:
            return value
        return f'{value[:_GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT - 3]}...'
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if depth >= _GRAPH_OBSERVATION_DEPTH_LIMIT:
        sha256, size_bytes = _bounded_observation_digest(value)
        return {
            'observation_projection_truncated': True,
            'sha256': sha256,
            'size_bytes': size_bytes,
        }
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        items = list(value.items())
        for raw_key, item in items[:_GRAPH_OBSERVATION_MAPPING_LIMIT]:
            key = str(raw_key or '').strip()[:256]
            if not key:
                continue
            if key in _GRAPH_OBSERVATION_BULK_KEYS and item not in (None, '', [], {}):
                sha256, size_bytes = _bounded_observation_digest(item)
                projected[f'{key}_observation_omitted'] = True
                projected.setdefault(f'{key}_sha256', sha256)
                projected.setdefault(f'{key}_size_bytes', size_bytes)
                continue
            child = _project_bounded_observation_value(item, depth=depth + 1)
            projected[key] = child
            if isinstance(item, str) and child != item:
                projected.setdefault(f'{key}_length_chars', len(item))
                projected.setdefault(
                    f'{key}_sha256',
                    hashlib.sha256(item.encode('utf-8')).hexdigest(),
                )
                projected.setdefault(f'{key}_truncated', True)
            if isinstance(item, list) and len(item) > _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT:
                projected.setdefault(f'{key}_count', len(item))
                projected.setdefault(f'{key}_truncated', True)
        if len(items) > _GRAPH_OBSERVATION_MAPPING_LIMIT:
            projected['observation_omitted_key_count'] = (
                len(items) - _GRAPH_OBSERVATION_MAPPING_LIMIT
            )
        return projected
    if isinstance(value, list):
        return [
            _project_bounded_observation_value(item, depth=depth + 1)
            for item in value[:_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT]
        ]
    return _json_safe(value)


def _hydrate_observation_named_snapshots(
    payload: Mapping[str, Any],
    *,
    keys: set[str],
    frames_dir: Path | str,
    base_json_path: str = '',
    errors: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key in keys:
        value = payload.get(key)
        if value not in (None, '', [], {}):
            projected[key] = _project_bounded_observation_value(value)
            continue
        ref = payload.get(f'{key}_snapshot_ref')
        expanded, snapshot_error = _read_observation_snapshot_payload(
            ref,
            frames_dir=frames_dir,
        )
        if snapshot_error:
            if errors is not None:
                errors.append({
                    **snapshot_error,
                    'json_path': '.'.join(
                        token
                        for token in (base_json_path, key)
                        if token
                    ),
                })
            continue
        if expanded not in (None, '', [], {}):
            projected[key] = _project_bounded_observation_value(expanded)
    return projected


def _project_graph_closure_observation(
    value: Mapping[str, Any],
    *,
    snapshot_ref: Optional[Mapping[str, Any]] = None,
    source_json_path: str,
) -> dict[str, Any]:
    """Project Closure truth without following any recursive child refs."""

    def bounded_scalar(item: Any) -> Any:
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            if len(item) <= _GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT:
                return item
            return f'{item[:_GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT - 3]}...'
        if isinstance(item, (bool, int, float)) or item is None:
            return item
        return None

    truncated_scalar_fields: list[str] = []
    projected: dict[str, Any] = {}
    for key in _GRAPH_CLOSURE_OBSERVATION_SCALAR_KEYS:
        raw_item = value.get(key)
        item = bounded_scalar(raw_item)
        if item in (None, ''):
            continue
        projected[key] = item
        if isinstance(raw_item, str) and item != raw_item:
            truncated_scalar_fields.append(key)
    counts = value.get('counts')
    if isinstance(counts, Mapping):
        projected['counts'] = {}
        for raw_key, raw_item in list(counts.items())[:_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT]:
            key = str(raw_key or '').strip()[:128]
            item = bounded_scalar(raw_item)
            if key and item not in (None, ''):
                projected['counts'][key] = item
        if len(counts) > _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT:
            projected['counts_truncated'] = True

    checks = value.get('checks')
    if isinstance(checks, list) and checks:
        projected['checks'] = [
            {
                key: bounded
                for key in _GRAPH_CLOSURE_OBSERVATION_CHECK_KEYS
                if isinstance(item, Mapping)
                and (bounded := bounded_scalar(item.get(key))) not in (None, '')
            }
            for item in checks[:_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT]
            if isinstance(item, Mapping)
        ]
        projected['check_count'] = len(checks)
        if len(checks) > _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT:
            projected['checks_truncated'] = True

    for adequacy_key in ('intent_graph_adequacy', 'intent_graph_adequacy_review'):
        adequacy = value.get(adequacy_key)
        if not isinstance(adequacy, Mapping) or not adequacy:
            continue
        compact_adequacy = _project_bounded_observation_value(adequacy)
        if isinstance(compact_adequacy, Mapping) and compact_adequacy:
            projected[adequacy_key] = _json_safe(compact_adequacy)

    repair_actions = value.get('repair_actions')
    if isinstance(repair_actions, list) and repair_actions:
        projected['repair_actions'] = [
            {
                key: bounded
                for key in _GRAPH_CLOSURE_OBSERVATION_CHECK_KEYS
                if (bounded := bounded_scalar(item.get(key))) not in (None, '')
            }
            if isinstance(item, Mapping)
            else bounded_scalar(item)
            for item in repair_actions[:_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT]
        ]
        projected['repair_action_count'] = len(repair_actions)
        if len(repair_actions) > _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT:
            projected['repair_actions_truncated'] = True

    surface_state = value.get('surface_state')
    if isinstance(surface_state, Mapping):
        compact_surface = {
            key: bounded
            for key in (
                'authority',
                'kind',
                'late_fill_status',
                'reason',
                'status',
            )
            if (bounded := bounded_scalar(surface_state.get(key))) not in (None, '')
        }
        active_categories = surface_state.get('active_categories')
        if isinstance(active_categories, list):
            compact_surface['active_categories'] = [
                bounded
                for item in active_categories[:_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT]
                if (bounded := bounded_scalar(item)) not in (None, '')
            ]
            if len(active_categories) > _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT:
                compact_surface['active_categories_truncated'] = True
        category_counts = surface_state.get('category_counts')
        if isinstance(category_counts, Mapping):
            compact_surface['category_counts'] = {}
            for raw_key, raw_item in list(category_counts.items())[
                :_GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT
            ]:
                key = str(raw_key or '').strip()[:128]
                item = bounded_scalar(raw_item)
                if key and item not in (None, ''):
                    compact_surface['category_counts'][key] = item
            if len(category_counts) > _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT:
                compact_surface['category_counts_truncated'] = True
        for key, item in surface_state.items():
            if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
                compact_surface[key] = _json_safe(item)
        if compact_surface:
            projected['surface_state'] = compact_surface

    for key, item in value.items():
        if key.endswith('_snapshot_ref') and isinstance(item, Mapping):
            projected[key] = _json_safe(item)
    if isinstance(snapshot_ref, Mapping) and snapshot_ref:
        projected['snapshot_ref'] = _json_safe(snapshot_ref)
    projected['observation_projection'] = {
        'kind': 'ollmo.graph_closure_observation_projection',
        'runtime_effect': 'none',
        'source_json_path': source_json_path,
        'outer_snapshot_payload_read': bool(snapshot_ref),
        'child_sidecar_hydration': 'none',
        'list_item_limit': _GRAPH_CLOSURE_OBSERVATION_LIST_LIMIT,
        'scalar_text_limit': _GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT,
    }
    if truncated_scalar_fields:
        projected['observation_projection']['truncated_scalar_fields'] = sorted(
            set(truncated_scalar_fields)
        )
    return _json_safe(projected)


def load_latest_response_observation_state(
    response_id: str,
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    index_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load bounded latest-frame truth for rollout observers without full hydration.

    This deliberately resolves only response identity, request metadata, Late
    Fill state, graph-rebase records, scope review, and rebase diagnostics.  It
    never expands artifact, output, work-tree, or unrelated recursive sidecars.
    """

    normalized_id = str(response_id or '').strip()
    current_index = (
        dict(index_state)
        if isinstance(index_state, Mapping)
        else load_response_frame_index(frames_dir=frames_dir)
    )
    responses = (
        current_index.get('responses')
        if isinstance(current_index.get('responses'), Mapping)
        else {}
    )
    entry = responses.get(normalized_id) if isinstance(responses, Mapping) else None
    if not isinstance(entry, Mapping):
        return {
            'ok': False,
            'status_code': 404,
            'error': {
                'code': 'response_frame_not_found',
                'message': 'Response frame is not indexed.',
            },
        }
    ledger_path = Path(
        str(entry.get('ledger_path') or '').strip()
        or _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)
    )
    # A size-aligned entry is not enough: a failed index write followed by an
    # unrelated append can make an older response coordinate look current.
    # Observation truth therefore requires the complete response-map digest.
    if not _response_frame_index_has_verified_response_map(current_index, ledger_path):
        stale_index = not _response_frame_index_is_fresh(
            current_index,
            ledger_path,
        )
        return {
            'ok': False,
            'status_code': 409,
            'error': {
                'code': (
                    'response_frame_index_stale'
                    if stale_index
                    else 'response_frame_index_unverified'
                ),
                'message': (
                    'Response-frame index does not match the current ledger.'
                    if stale_index
                    else 'Response-frame index is not completely verified for the current ledger.'
                ),
            },
        }
    frame, frame_error = _read_indexed_response_frame(
        entry,
        ledger_path=ledger_path,
        response_id=normalized_id,
    )
    if not frame:
        return {
            'ok': False,
            'status_code': 409,
            'error': frame_error or {
                'code': 'response_frame_index_read_failed',
                'message': 'Indexed response frame could not be read.',
            },
        }
    manifest = (
        entry.get('effective_snapshot_manifest')
        if isinstance(entry.get('effective_snapshot_manifest'), Mapping)
        else {}
    )
    runtime_ref = _observation_snapshot_ref(
        frame,
        manifest,
        'runtime',
        'current_state.runtime',
    )
    runtime_snapshot, runtime_snapshot_error = _read_observation_snapshot_payload(
        runtime_ref,
        frames_dir=frames_dir,
    )
    if runtime_snapshot_error:
        return _observation_snapshot_failure(
            normalized_id,
            runtime_snapshot_error,
            json_path='runtime',
        )
    runtime_snapshot = dict(runtime_snapshot) if isinstance(runtime_snapshot, Mapping) else {}

    graph_ref = (
        runtime_snapshot.get('request_phase_graph_snapshot_ref')
        if isinstance(runtime_snapshot.get('request_phase_graph_snapshot_ref'), Mapping)
        else _observation_snapshot_ref(
            frame,
            manifest,
            'runtime.request_phase_graph',
            'current_state.runtime.request_phase_graph',
            'planning.request_phase_graph',
        )
    )
    raw_graph, graph_snapshot_error = _read_observation_snapshot_payload(
        graph_ref,
        frames_dir=frames_dir,
    )
    if graph_snapshot_error:
        return _observation_snapshot_failure(
            normalized_id,
            graph_snapshot_error,
            json_path='runtime.request_phase_graph',
        )
    if not isinstance(raw_graph, Mapping):
        raw_graph = (
            runtime_snapshot.get('request_phase_graph')
            if isinstance(runtime_snapshot.get('request_phase_graph'), Mapping)
            else (frame.get('planning') or {}).get('request_phase_graph')
            if isinstance(frame.get('planning'), Mapping)
            and isinstance((frame.get('planning') or {}).get('request_phase_graph'), Mapping)
            else {}
        )
    graph_snapshot_errors: list[dict[str, Any]] = []
    graph = _hydrate_observation_named_snapshots(
        raw_graph,
        keys=_GRAPH_REBASE_OBSERVATION_GRAPH_KEYS,
        frames_dir=frames_dir,
        base_json_path='runtime.request_phase_graph',
        errors=graph_snapshot_errors,
    )
    if graph_snapshot_errors:
        graph_snapshot_error = graph_snapshot_errors[0]
        return _observation_snapshot_failure(
            normalized_id,
            graph_snapshot_error,
            json_path=str(
                graph_snapshot_error.get('json_path')
                or 'runtime.request_phase_graph'
            ),
        )

    diagnostics_ref = (
        runtime_snapshot.get('developer_diagnostics_snapshot_ref')
        if isinstance(runtime_snapshot.get('developer_diagnostics_snapshot_ref'), Mapping)
        else _observation_snapshot_ref(
            frame,
            manifest,
            'runtime.developer_diagnostics',
            'current_state.runtime.developer_diagnostics',
        )
    )
    raw_diagnostics, diagnostics_snapshot_error = _read_observation_snapshot_payload(
        diagnostics_ref,
        frames_dir=frames_dir,
    )
    if diagnostics_snapshot_error:
        return _observation_snapshot_failure(
            normalized_id,
            diagnostics_snapshot_error,
            json_path='runtime.developer_diagnostics',
        )
    if not isinstance(raw_diagnostics, Mapping):
        raw_diagnostics = (
            runtime_snapshot.get('developer_diagnostics')
            if isinstance(runtime_snapshot.get('developer_diagnostics'), Mapping)
            else {}
        )
    diagnostics_snapshot_errors: list[dict[str, Any]] = []
    diagnostics = _hydrate_observation_named_snapshots(
        raw_diagnostics,
        keys=_GRAPH_REBASE_OBSERVATION_DIAGNOSTIC_KEYS,
        frames_dir=frames_dir,
        base_json_path='runtime.developer_diagnostics',
        errors=diagnostics_snapshot_errors,
    )
    if diagnostics_snapshot_errors:
        diagnostics_snapshot_error = diagnostics_snapshot_errors[0]
        return _observation_snapshot_failure(
            normalized_id,
            diagnostics_snapshot_error,
            json_path=str(
                diagnostics_snapshot_error.get('json_path')
                or 'runtime.developer_diagnostics'
            ),
        )

    closure_source_json_path = ''
    closure_ref: dict[str, Any] = {}
    raw_closure: Any = None
    if isinstance(runtime_snapshot.get('graph_closure_review'), Mapping):
        raw_closure = runtime_snapshot.get('graph_closure_review')
        closure_source_json_path = 'runtime.graph_closure_review'
    else:
        closure_ref = _observation_snapshot_ref(
            frame,
            manifest,
            'runtime.graph_closure_review',
            'current_state.runtime.graph_closure_review',
        )
        if closure_ref:
            closure_source_json_path = str(
                closure_ref.get('json_path') or 'runtime.graph_closure_review'
            ).strip()
        elif isinstance(runtime_snapshot.get('graph_closure_review_snapshot_ref'), Mapping):
            closure_ref = dict(
                runtime_snapshot.get('graph_closure_review_snapshot_ref') or {}
            )
            closure_source_json_path = 'runtime.graph_closure_review'
    if raw_closure is None and not closure_ref:
        if isinstance(raw_diagnostics.get('graph_closure_review'), Mapping):
            raw_closure = raw_diagnostics.get('graph_closure_review')
            closure_source_json_path = 'runtime.developer_diagnostics.graph_closure_review'
        else:
            closure_ref = _observation_snapshot_ref(
                frame,
                manifest,
                'runtime.developer_diagnostics.graph_closure_review',
                'current_state.runtime.developer_diagnostics.graph_closure_review',
            )
            if closure_ref:
                closure_source_json_path = str(
                    closure_ref.get('json_path')
                    or 'runtime.developer_diagnostics.graph_closure_review'
                ).strip()
            elif isinstance(raw_diagnostics.get('graph_closure_review_snapshot_ref'), Mapping):
                closure_ref = dict(
                    raw_diagnostics.get('graph_closure_review_snapshot_ref') or {}
                )
                closure_source_json_path = (
                    'runtime.developer_diagnostics.graph_closure_review'
                )
    if raw_closure is None and closure_ref:
        raw_closure, closure_snapshot_error = _read_observation_snapshot_payload(
            closure_ref,
            frames_dir=frames_dir,
        )
        if closure_snapshot_error:
            return _observation_snapshot_failure(
                normalized_id,
                closure_snapshot_error,
                json_path=closure_source_json_path or 'runtime.graph_closure_review',
            )
    if isinstance(raw_closure, Mapping):
        raw_closure = dict(raw_closure)
        for adequacy_key in (
            'intent_graph_adequacy',
            'intent_graph_adequacy_review',
        ):
            if isinstance(raw_closure.get(adequacy_key), Mapping):
                continue
            adequacy_ref = raw_closure.get(f'{adequacy_key}_snapshot_ref')
            if not isinstance(adequacy_ref, Mapping):
                continue
            adequacy, adequacy_snapshot_error = _read_observation_snapshot_payload(
                adequacy_ref,
                frames_dir=frames_dir,
            )
            if adequacy_snapshot_error:
                return _observation_snapshot_failure(
                    normalized_id,
                    adequacy_snapshot_error,
                    json_path='.'.join(
                        token
                        for token in (
                            closure_source_json_path
                            or 'runtime.graph_closure_review',
                            adequacy_key,
                        )
                        if token
                    ),
                )
            if isinstance(adequacy, Mapping) and adequacy:
                raw_closure[adequacy_key] = dict(adequacy)
    graph_closure_review = (
        _project_graph_closure_observation(
            raw_closure,
            snapshot_ref=closure_ref,
            source_json_path=(
                closure_source_json_path or 'runtime.graph_closure_review'
            ),
        )
        if isinstance(raw_closure, Mapping)
        else {}
    )

    late_fill_ref = _observation_snapshot_ref(
        frame,
        manifest,
        'late_fill',
        'current_state.late_fill',
        'runtime.late_fill',
    )
    late_fill, late_fill_snapshot_error = _read_observation_snapshot_payload(
        late_fill_ref,
        frames_dir=frames_dir,
    )
    if late_fill_snapshot_error:
        return _observation_snapshot_failure(
            normalized_id,
            late_fill_snapshot_error,
            json_path='late_fill',
        )
    if not isinstance(late_fill, Mapping):
        late_fill = frame.get('late_fill') if isinstance(frame.get('late_fill'), Mapping) else {}
    raw_late_fill = dict(late_fill)
    late_fill = _response_wire_late_fill_projection(raw_late_fill)

    compact_current = frame.get('current_state') if isinstance(frame.get('current_state'), Mapping) else {}
    lifecycle_state = str(
        compact_current.get('lifecycle_state')
        or entry.get('current_lifecycle_state')
        or frame.get('status')
        or ''
    ).strip()
    runtime = {
        'request_phase_graph': graph,
        'developer_diagnostics': diagnostics,
    }
    if graph_closure_review:
        runtime['graph_closure_review'] = graph_closure_review
    raw_request = frame.get('request') if isinstance(frame.get('request'), Mapping) else {}
    request_payload: dict[str, Any] = {}
    for key in ('prompt', 'prompt_family', 'workload_family'):
        item = raw_request.get(key)
        if item in (None, '', [], {}):
            continue
        if isinstance(item, str):
            _response_wire_put_bounded_text(
                request_payload,
                key,
                item,
                limit=_GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT,
            )
        else:
            request_payload[key] = _project_bounded_observation_value(item)
    if isinstance(raw_request.get('request_meta'), Mapping):
        request_payload['request_meta'] = _project_bounded_observation_value(
            raw_request.get('request_meta')
        )
    if not str(request_payload.get('prompt') or '').strip():
        request_content_ref = _observation_snapshot_ref(
            frame,
            manifest,
            'working_frame.request.content',
        )
        request_content, request_content_snapshot_error = _read_observation_snapshot_payload(
            request_content_ref,
            frames_dir=frames_dir,
        )
        if request_content_snapshot_error:
            return _observation_snapshot_failure(
                normalized_id,
                request_content_snapshot_error,
                json_path='working_frame.request.content',
            )
        if isinstance(request_content, Mapping):
            prompt = str(request_content.get('prompt') or '').strip()
            if prompt:
                _response_wire_put_bounded_text(
                    request_payload,
                    'prompt',
                    prompt,
                    limit=_GRAPH_CLOSURE_OBSERVATION_TEXT_LIMIT,
                )
    projected_frame = {
        key: _json_safe(frame.get(key))
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
    projected_frame['current_state'] = {
        'lifecycle_state': lifecycle_state,
        'status': str(compact_current.get('status') or frame.get('status') or '').strip(),
    }
    response_payload = {
        'id': normalized_id,
        'response_id': normalized_id,
        'status': str(frame.get('status') or '').strip(),
        'lifecycle_state': lifecycle_state,
        'request': request_payload,
        'late_fill': late_fill,
        'runtime': runtime,
        'response_frame': projected_frame,
    }
    if isinstance(frame.get('frame_relation'), Mapping):
        response_payload['frame_relation'] = _json_safe(frame.get('frame_relation'))
    return {
        'ok': True,
        'response_payload': response_payload,
        'response_frame': projected_frame,
        'ledger_path': str(ledger_path),
        'index_used': True,
        'bounded_observation': True,
    }


_RESPONSE_WIRE_TEXT_PREVIEW_LIMIT = _response_wire_policy.INDEXED_TEXT_PREVIEW_LIMIT
_RESPONSE_WIRE_MEDIA_PREVIEW_LIMIT = _response_wire_policy.INDEXED_MEDIA_PREVIEW_LIMIT
_RESPONSE_WIRE_DEPTH_LIMIT = _response_wire_policy.INDEXED_DEPTH_LIMIT
_RESPONSE_WIRE_RECORD_LIMIT_BYTES = _response_wire_policy.INDEXED_RECORD_LIMIT_BYTES


def _response_wire_stat(
    stats: Optional[dict[str, int]],
    key: str,
    amount: int = 1,
) -> None:
    _response_wire_policy.indexed_stat(stats, key, amount)


def _response_wire_json_identity(value: Any) -> tuple[bytes, str]:
    return _response_wire_policy.indexed_json_identity(value)


def _response_wire_string_limit(key: str) -> int:
    return _response_wire_policy.indexed_string_limit(key)


def _response_wire_put_bounded_text(
    target: dict[str, Any],
    key: str,
    value: str,
    *,
    stats: Optional[dict[str, int]] = None,
    limit: Optional[int] = None,
) -> None:
    _response_wire_policy.indexed_put_bounded_text(
        target,
        key,
        value,
        stats=stats,
        limit=limit,
    )


def _response_wire_bounded_value(
    value: Any,
    *,
    stats: Optional[dict[str, int]] = None,
    depth: int = 0,
) -> Any:
    return _response_wire_policy.indexed_bounded_value(
        value,
        stats=stats,
        depth=depth,
    )


_RESPONSE_WIRE_RECORD_IDENTITY_KEYS = _response_wire_policy.INDEXED_RECORD_IDENTITY_KEYS


def _response_wire_record_handle(
    value: Mapping[str, Any],
    *,
    stats: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    return _response_wire_policy.indexed_record_handle(value, stats=stats)


def _response_wire_project_record(
    value: Mapping[str, Any],
    *,
    stats: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    return _response_wire_policy.indexed_project_record(value, stats=stats)


def _response_wire_project_collection(
    value: Any,
    *,
    stats: Optional[dict[str, int]] = None,
) -> list[dict[str, Any]]:
    return _response_wire_policy.indexed_project_collection(value, stats=stats)


_RESPONSE_WIRE_LATE_FILL_BRANCH_KEYS = (
    _response_wire_policy.INDEXED_LATE_FILL_BRANCH_KEYS
)
_RESPONSE_WIRE_LATE_FILL_CAPABILITY_KEYS = (
    _response_wire_policy.INDEXED_LATE_FILL_CAPABILITY_KEYS
)


def _response_wire_late_fill_projection(
    value: Any,
    *,
    stats: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    return _response_wire_policy.indexed_late_fill_projection(value, stats=stats)


def _response_wire_batch_item_handle(
    item: Mapping[str, Any],
    *,
    stats: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    return _response_wire_policy.indexed_record_handle(
        item,
        stats=stats,
        batch_item=True,
    )


def _response_wire_batch_projection(
    frame: Mapping[str, Any],
    *,
    outputs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    stats: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    return _response_wire_policy.indexed_batch_projection(
        frame,
        outputs=outputs,
        artifacts=artifacts,
        stats=stats,
    )


def _response_wire_snapshot_manifest_projection(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _response_wire_policy.indexed_snapshot_manifest_projection(value)


def _response_wire_frame_projection(
    frame: Mapping[str, Any],
    *,
    effective_snapshot_manifest: Mapping[str, Any],
    snapshot_manifest_projection: Optional[Mapping[str, Any]] = None,
    outputs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    lifecycle_state: str,
) -> dict[str, Any]:
    return _response_wire_policy.indexed_frame_projection(
        frame,
        effective_snapshot_manifest=effective_snapshot_manifest,
        snapshot_manifest_projection=snapshot_manifest_projection,
        outputs=outputs,
        artifacts=artifacts,
        lifecycle_state=lifecycle_state,
    )


def load_latest_response_wire_state(
    response_id: str,
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    index_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load an index-only response projection without reading any sidecar body.

    The wire path is deliberately fail-closed: it never scans the ledger and
    never falls back to the fully hydrated canonical loader.  Public output and
    artifact truth comes from the already compact indexed row; all effective
    current/inherited snapshot identities stay available as exact refs.
    """

    normalized_id = str(response_id or '').strip()
    current_index = (
        dict(index_state)
        if isinstance(index_state, Mapping)
        else load_response_frame_index(frames_dir=frames_dir)
    )
    ledger_path = _ledger_path(frames_dir=frames_dir, ledger_name=ledger_name)
    base_projection = {
        'kind': 'ollmo.response_wire_projection',
        'version': 1,
        'runtime_effect': 'none',
        'source': 'current_index_compact_ledger_frame',
        'read_only': True,
        'index_used': True,
        'ledger_fallback_used': False,
        'sidecar_reads': 0,
        'sidecar_hydration': 'none',
    }
    if not _response_frame_index_has_verified_response_map(current_index, ledger_path):
        return {
            'ok': False,
            'status_code': 409,
            'wire_projection': base_projection,
            'error': {
                'code': 'response_frame_index_unverified',
                'message': 'Response-frame index is not completely verified for the current ledger.',
                'response_id': normalized_id,
            },
        }
    responses = (
        current_index.get('responses')
        if isinstance(current_index.get('responses'), Mapping)
        else {}
    )
    entry = responses.get(normalized_id) if isinstance(responses, Mapping) else None
    if not isinstance(entry, Mapping):
        return {
            'ok': False,
            'status_code': 404,
            'wire_projection': base_projection,
            'error': {
                'code': 'response_frame_not_found',
                'message': 'Response frame is not indexed.',
                'response_id': normalized_id,
            },
        }
    frame, frame_error = _read_indexed_response_frame(
        entry,
        ledger_path=ledger_path,
        response_id=normalized_id,
    )
    if not frame:
        return {
            'ok': False,
            'status_code': 409,
            'wire_projection': base_projection,
            'error': frame_error or {
                'code': 'response_frame_index_read_failed',
                'message': 'Indexed response frame could not be read.',
            },
        }

    indexed_manifest = entry.get('effective_snapshot_manifest')
    manifest_complete = isinstance(indexed_manifest, Mapping)
    effective_manifest = (
        {
            str(path): _json_safe(ref)
            for path, ref in indexed_manifest.items()
            if isinstance(ref, Mapping) and ref.get('sha256')
        }
        if isinstance(indexed_manifest, Mapping)
        else _snapshot_items_from_frame(frame)
    )
    projected_manifest, manifest_projection = (
        _response_wire_snapshot_manifest_projection(effective_manifest)
    )
    projection_stats: dict[str, int] = {}
    aggregate_compaction = (
        frame.get('public_body_compaction')
        if isinstance(frame.get('public_body_compaction'), Mapping)
        else {}
    )
    aggregate_externalized_count = _coerce_frame_sequence(
        aggregate_compaction.get('externalized_body_count'),
        _coerce_frame_sequence(
            aggregate_compaction.get('externalized_collection_count'),
            0,
        ),
    ) or 0
    if aggregate_externalized_count > 0:
        _response_wire_stat(
            projection_stats,
            'aggregate_externalized_collection_count',
            aggregate_externalized_count,
        )
    compact_current = (
        frame.get('current_state')
        if isinstance(frame.get('current_state'), Mapping)
        else {}
    )
    compact_output = frame.get('output') if isinstance(frame.get('output'), Mapping) else {}
    outputs_source = (
        compact_current.get('outputs')
        if isinstance(compact_current.get('outputs'), list)
        else compact_output.get('outputs')
        if isinstance(compact_output.get('outputs'), list)
        else []
    )
    raw_outputs = [
        dict(item)
        for item in outputs_source
        if isinstance(item, Mapping)
    ]
    outputs = _response_wire_project_collection(
        raw_outputs,
        stats=projection_stats,
    )
    raw_output_slots = [
        dict(item)
        for item in (
            compact_current.get('output_slots')
            if isinstance(compact_current.get('output_slots'), list)
            else []
        )
        if isinstance(item, Mapping)
    ]
    output_slots = _response_wire_project_collection(
        raw_output_slots,
        stats=projection_stats,
    )
    raw_output_branches = [
        dict(item)
        for item in (
            compact_current.get('output_branches')
            if isinstance(compact_current.get('output_branches'), list)
            else []
        )
        if isinstance(item, Mapping)
    ]
    output_branches = _response_wire_project_collection(
        raw_output_branches,
        stats=projection_stats,
    )
    compact_artifacts = (
        frame.get('artifacts') if isinstance(frame.get('artifacts'), Mapping) else {}
    )
    artifacts_source = (
        compact_current.get('artifacts')
        if isinstance(compact_current.get('artifacts'), list)
        else compact_artifacts.get('output')
        if isinstance(compact_artifacts.get('output'), list)
        else []
    )
    raw_artifacts = [
        dict(item)
        for item in artifacts_source
        if isinstance(item, Mapping)
    ]
    artifacts = _response_wire_project_collection(
        raw_artifacts,
        stats=projection_stats,
    )
    lifecycle_state = str(
        compact_current.get('lifecycle_state')
        or entry.get('current_lifecycle_state')
        or frame.get('status')
        or ''
    ).strip()
    projected_frame = _response_wire_frame_projection(
        frame,
        effective_snapshot_manifest=projected_manifest,
        snapshot_manifest_projection=manifest_projection,
        outputs=outputs,
        artifacts=artifacts,
        lifecycle_state=lifecycle_state,
    )
    wire_projection = {
        **base_projection,
        'effective_snapshot_count': len(effective_manifest),
        'effective_snapshot_manifest_complete': manifest_complete,
        'effective_snapshot_manifest_projected_complete': bool(
            manifest_projection.get('manifest_projection_complete')
        ),
        'effective_snapshot_manifest_source': (
            'current_index'
            if manifest_complete
            else 'compact_frame_delta_fallback'
        ),
        'source_ledger_line_size_bytes': _coerce_frame_sequence(
            entry.get('line_length')
        ),
        'public_outputs_source': (
            'compact_current_state.outputs'
            if isinstance(compact_current.get('outputs'), list)
            else 'compact_output.outputs'
        ),
        'public_output_slots_source': 'compact_current_state.output_slots',
        'public_output_branches_source': 'compact_current_state.output_branches',
        'public_artifacts_source': (
            'compact_current_state.artifacts'
            if isinstance(compact_current.get('artifacts'), list)
            else 'compact_artifacts.output'
        ),
        'public_projection_limits': {
            'inline_byte_budget': _RESPONSE_WIRE_INLINE_BUDGET_BYTES,
            'public_body_snapshot_min_bytes': _PUBLIC_BODY_SNAPSHOT_MIN_BYTES,
            'public_collection_snapshot_min_bytes': (
                _PUBLIC_COLLECTION_SNAPSHOT_MIN_BYTES
            ),
            'record_limit_bytes': _RESPONSE_WIRE_RECORD_LIMIT_BYTES,
            'text_preview_limit_chars': _RESPONSE_WIRE_TEXT_PREVIEW_LIMIT,
        },
    }
    response_payload = {
        'id': normalized_id,
        'response_id': normalized_id,
        'status': str(frame.get('status') or compact_current.get('status') or '').strip(),
        'lifecycle_state': lifecycle_state,
        'outputs': outputs,
        'output_slots': output_slots,
        'output_branches': output_branches,
        'artifacts': artifacts,
        'response_frame': projected_frame,
        'wire_projection': wire_projection,
    }
    preserved_current_state_fields: list[str] = []
    for key in (
        'audio_mimetype',
        'canonical_status_field',
        'error',
        'error_detail',
        'error_ref',
        'image_state',
        'input_artifacts',
        'lang_code',
        'lang_code_source',
        'message_id',
        'output',
        'output_format',
        'output_text',
        'provenance_id',
        'reference_artifacts',
        'response_format',
        'recovery_hint',
        'route_artifact_path',
        'route_artifact_ref',
        'route_confidence',
        'route_reason',
        'route_reuse_last_artifact',
        'route_source',
        'saved_audio_path',
        'saved_image_path',
        'saved_text_artifacts',
        'saved_text_path',
        'seed',
        'status_compatibility',
        'status_semantics',
        'text_artifact_requests',
        'tts_audio_integrity_evidence',
        'usage',
    ):
        value = compact_current.get(key)
        if value not in (None, '', [], {}):
            if isinstance(value, str):
                _response_wire_put_bounded_text(
                    response_payload,
                    key,
                    value,
                    stats=projection_stats,
                )
            elif key in {
                'input_artifacts',
                'output',
                'reference_artifacts',
                'saved_text_artifacts',
                'text_artifact_requests',
            } and isinstance(value, list):
                response_payload[key] = _response_wire_project_collection(
                    value,
                    stats=projection_stats,
                )
            else:
                response_payload[key] = _response_wire_bounded_value(
                    value,
                    stats=projection_stats,
                )
            preserved_current_state_fields.append(key)
    for key in ('error', 'error_detail'):
        if response_payload.get(key) not in (None, '', [], {}):
            continue
        value = frame.get('error') if key == 'error' else None
        if value not in (None, '', [], {}):
            response_payload[key] = _response_wire_bounded_value(
                value,
                stats=projection_stats,
            )
            preserved_current_state_fields.append(key)
    for key in ('backend', 'capability', 'instance_id', 'mode', 'model', 'object'):
        value = compact_current.get(key)
        if value not in (None, '', [], {}):
            if isinstance(value, str):
                _response_wire_put_bounded_text(
                    response_payload,
                    key,
                    value,
                    stats=projection_stats,
                    limit=512,
                )
            else:
                response_payload[key] = _response_wire_bounded_value(
                    value,
                    stats=projection_stats,
                )
            preserved_current_state_fields.append(key)
    compact_route = (
        frame.get('route')
        if isinstance(frame.get('route'), Mapping)
        else {}
    )
    for key in _ROUTE_FRAME_KEYS:
        if response_payload.get(key) not in (None, '', [], {}):
            continue
        value = compact_route.get(key)
        if value in (None, '', [], {}):
            continue
        if isinstance(value, str):
            _response_wire_put_bounded_text(
                response_payload,
                key,
                value,
                stats=projection_stats,
                limit=_RESPONSE_WIRE_TEXT_PREVIEW_LIMIT,
            )
        else:
            response_payload[key] = _response_wire_bounded_value(
                value,
                stats=projection_stats,
            )
        preserved_current_state_fields.append(key)
    for ref_key, ref in compact_current.items():
        if (
            not str(ref_key).endswith('_snapshot_ref')
            or not isinstance(ref, Mapping)
            or str(ref.get('projection_role') or '').strip()
            != 'public_body_exact'
        ):
            continue
        response_payload[str(ref_key)] = _json_safe(ref)
        for metadata_key in ref.get('public_projection_metadata_keys') or []:
            metadata_key = str(metadata_key or '').strip()
            if (
                metadata_key
                and compact_current.get(metadata_key) not in (None, '', [], {})
            ):
                response_payload[metadata_key] = _json_safe(
                    compact_current.get(metadata_key)
                )
    wire_projection['preserved_public_current_state_fields'] = sorted(
        set(preserved_current_state_fields)
    )
    batch_projection = _response_wire_batch_projection(
        frame,
        outputs=raw_outputs,
        artifacts=raw_artifacts,
        stats=projection_stats,
    )
    if batch_projection:
        response_payload.update(batch_projection)
    late_fill = _response_wire_late_fill_projection(
        frame.get('late_fill'),
        stats=projection_stats,
    )
    if late_fill:
        response_payload['late_fill'] = late_fill
    wire_projection['public_projection_truncation'] = {
        key: value
        for key, value in sorted(projection_stats.items())
        if value
    }
    if manifest_projection.get('manifest_projection_truncated'):
        wire_projection['effective_snapshot_manifest_projection'] = _json_safe(
            manifest_projection
        )
    result = {
        'ok': True,
        'response_payload': response_payload,
        'response_frame': projected_frame,
        'wire_projection': wire_projection,
        'ledger_path': str(ledger_path),
        'index_path': current_index.get('index_path'),
        'index_used': True,
        'index_stale': False,
        'ledger_fallback_used': False,
        'bounded_wire_projection': True,
    }
    serialized_size_bytes = _json_size_bytes(result)
    if serialized_size_bytes > _RESPONSE_WIRE_INLINE_BUDGET_BYTES:
        return {
            'ok': False,
            'status_code': 409,
            'response_id': normalized_id,
            'wire_projection': {
                **base_projection,
                'source_ledger_line_size_bytes': _coerce_frame_sequence(
                    entry.get('line_length')
                ),
                'attempted_projection_size_bytes': serialized_size_bytes,
                'inline_byte_budget': _RESPONSE_WIRE_INLINE_BUDGET_BYTES,
            },
            'error': {
                'code': 'response_wire_projection_budget_exceeded',
                'message': (
                    'The bounded response-frame projection exceeded its '
                    'aggregate wire budget.'
                ),
                'response_id': normalized_id,
            },
            'bounded_wire_projection': False,
        }
    return result


def load_latest_response_state(
    response_id: str,
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    _stable_retry: bool = True,
) -> dict[str, Any]:
    """Load the latest recoverable current-state payload for a response id."""

    normalized_id = str(response_id or '').strip()
    expected_ledger_path = _ledger_path(
        frames_dir=frames_dir,
        ledger_name=ledger_name,
    )
    initial_ledger_state = _response_frame_file_state(expected_ledger_path)

    def stable_success(result: dict[str, Any]) -> dict[str, Any]:
        current_state = _response_frame_file_state(expected_ledger_path)
        if current_state != initial_ledger_state:
            if _stable_retry:
                return load_latest_response_state(
                    normalized_id,
                    frames_dir=frames_dir,
                    ledger_name=ledger_name,
                    _stable_retry=False,
                )
            return {
                'ok': False,
                'status_code': 409,
                'error': {
                    'code': 'response_frame_ledger_changed_during_recovery',
                    'message': (
                        'The response-frame ledger changed repeatedly while '
                        'canonical response truth was being recovered.'
                    ),
                    'response_id': normalized_id,
                    'ledger_path': str(expected_ledger_path),
                },
            }
        if isinstance(current_state, Mapping):
            result['ledger_state'] = {
                key: int(current_state[key])
                for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
                if key in current_state
            }
        return result

    def public_body_integrity_failure(
        error: Mapping[str, Any],
        *,
        ledger_path: Path | str,
        index_used: bool,
        index_stale: bool,
        ledger_fallback_used: bool,
    ) -> dict[str, Any]:
        return {
            'ok': False,
            'status_code': 409,
            'error': {
                **dict(error),
                'response_id': normalized_id,
                'ledger_path': str(ledger_path),
                'index_used': index_used,
                'index_stale': index_stale,
                'ledger_fallback_used': ledger_fallback_used,
            },
        }

    index_state = load_response_frame_index(frames_dir=frames_dir)
    index_entry = (
        index_state.get('responses', {}).get(normalized_id)
        if isinstance(index_state.get('responses'), Mapping)
        else None
    )
    index_error = index_state.get('error') if isinstance(index_state.get('error'), Mapping) else None
    if isinstance(index_entry, Mapping):
        indexed_ledger_path = _ledger_path(
            frames_dir=frames_dir,
            ledger_name=ledger_name,
        )
        index_size_fresh = _response_frame_index_has_verified_response_map(
            index_state,
            indexed_ledger_path,
        )
        if index_size_fresh:
            indexed_frame, indexed_read_error = _read_indexed_response_frame(
                index_entry,
                ledger_path=indexed_ledger_path,
                response_id=normalized_id,
            )
            if indexed_frame:
                indexed_manifest = (
                    index_entry.get('effective_snapshot_manifest')
                    if isinstance(index_entry.get('effective_snapshot_manifest'), Mapping)
                    else _snapshot_items_from_frame(indexed_frame)
                )
                indexed_frame = _expand_frame_snapshot_manifest(
                    indexed_frame,
                    effective_manifest=indexed_manifest,
                )
                response_payload, public_body_error = (
                    _canonical_response_payload_from_frame(
                        indexed_frame,
                        frames_dir=frames_dir,
                    )
                )
                if public_body_error:
                    return public_body_integrity_failure(
                        public_body_error,
                        ledger_path=indexed_ledger_path,
                        index_used=True,
                        index_stale=False,
                        ledger_fallback_used=False,
                    )
                if response_payload:
                    canonical_frame = (
                        response_payload.get('response_frame')
                        if isinstance(response_payload.get('response_frame'), Mapping)
                        else indexed_frame
                    )
                    return stable_success({
                        'ok': True,
                        'response_payload': response_payload,
                        'response_frame': canonical_frame,
                        'frame_count': _coerce_frame_sequence(index_entry.get('latest_frame_sequence')),
                        'ledger_path': str(indexed_ledger_path),
                        'index_path': index_state.get('index_path'),
                        'index_used': True,
                        'index_stale': False,
                        'ledger_fallback_used': False,
                        'index_error': index_error,
                        'errors': [],
                    })
            index_error = indexed_read_error or index_error
        frames, errors = _iter_ledger_frames(indexed_ledger_path)
        latest_frame_id = str(index_entry.get('latest_frame_id') or '').strip()
        latest_frame_sequence = index_entry.get('latest_frame_sequence')
        response_frames = [
            frame
            for frame in frames
            if _frame_response_id(frame) == normalized_id
            and str(frame.get('kind') or '').strip() == 'ollmo.response_frame'
        ]
        indexed_matches = [
            frame
            for frame in response_frames
            if (
                (latest_frame_id and str(frame.get('frame_id') or '').strip() == latest_frame_id)
                or (
                    latest_frame_sequence not in (None, '')
                    and frame.get('frame_sequence') == latest_frame_sequence
                )
            )
        ]
        if response_frames:
            indexed_frame = indexed_matches[-1] if indexed_matches else None
            expanded_response_frames = _expand_snapshot_manifests_for_frames(response_frames)
            latest_frame = expanded_response_frames[-1]
            index_stale = True
            if indexed_frame is not None:
                index_stale = not (
                    str(indexed_frame.get('frame_id') or '').strip() == str(latest_frame.get('frame_id') or '').strip()
                    and indexed_frame.get('frame_sequence') == latest_frame.get('frame_sequence')
                )
            response_payload, public_body_error = (
                _canonical_response_payload_from_frame(
                    latest_frame,
                    frames_dir=frames_dir,
                )
            )
            if public_body_error:
                return public_body_integrity_failure(
                    public_body_error,
                    ledger_path=indexed_ledger_path,
                    index_used=not index_stale,
                    index_stale=index_stale,
                    ledger_fallback_used=index_stale,
                )
            if response_payload:
                canonical_frame = (
                    response_payload.get('response_frame')
                    if isinstance(response_payload.get('response_frame'), Mapping)
                    else latest_frame
                )
                return stable_success({
                    'ok': True,
                    'response_payload': response_payload,
                    'response_frame': canonical_frame,
                    'frame_count': len(response_frames),
                    'ledger_path': str(indexed_ledger_path),
                    'index_path': index_state.get('index_path'),
                    'index_used': not index_stale,
                    'index_stale': index_stale,
                    'ledger_fallback_used': index_stale,
                    'index_error': index_error,
                    'errors': errors,
                })

    if (
        not isinstance(index_entry, Mapping)
        and _response_frame_index_has_verified_response_map(
            index_state,
            expected_ledger_path,
        )
    ):
        return {
            'ok': False,
            'status_code': 404,
            'error': {
                'code': 'response_frame_not_found',
                'message': 'Response not found in persisted response frames.',
                'response_id': normalized_id,
                'ledger_path': str(expected_ledger_path),
                'corrupt_entry_count': 0,
                'index_error': index_error,
                'index_stale': False,
                'index_used': True,
                'ledger_fallback_used': False,
            },
        }

    records = load_response_frame_records(
        normalized_id,
        frames_dir=frames_dir,
        ledger_name=ledger_name,
    )
    frames = records.get('frames') if isinstance(records.get('frames'), list) else []
    errors = records.get('errors') if isinstance(records.get('errors'), list) else []
    if not frames:
        error_code = 'response_frame_not_found'
        status_code = 404
        message = 'Response not found in persisted response frames.'
        if errors:
            error_code = 'response_frame_ledger_corrupt'
            status_code = 409
            message = 'Persisted response frame ledger contains corrupt entries and no valid frame for this response.'
        return {
            'ok': False,
            'status_code': status_code,
            'error': {
                'code': error_code,
                'message': message,
                'response_id': normalized_id,
                'ledger_path': records.get('ledger_path'),
                'corrupt_entry_count': len(errors),
                'index_error': index_error,
                'index_stale': False,
                'ledger_fallback_used': True,
            },
        }
    expanded_frames = _expand_snapshot_manifests_for_frames(frames)
    latest_frame = expanded_frames[-1]
    response_payload, public_body_error = _canonical_response_payload_from_frame(
        latest_frame,
        frames_dir=frames_dir,
    )
    if public_body_error:
        return public_body_integrity_failure(
            public_body_error,
            ledger_path=records.get('ledger_path') or expected_ledger_path,
            index_used=False,
            index_stale=False,
            ledger_fallback_used=True,
        )
    if not response_payload:
        return {
            'ok': False,
            'status_code': 409,
            'error': {
                'code': 'response_frame_unrecoverable',
                'message': 'Persisted response frame exists but does not contain recoverable response state.',
                'response_id': normalized_id,
                'ledger_path': records.get('ledger_path'),
                'index_error': index_error,
                'index_stale': False,
                'ledger_fallback_used': True,
            },
        }
    canonical_frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), Mapping)
        else latest_frame
    )
    return stable_success({
        'ok': True,
        'response_payload': response_payload,
        'response_frame': canonical_frame,
        'frame_count': len(frames),
        'ledger_path': records.get('ledger_path'),
        'index_path': index_state.get('index_path'),
        'index_used': False,
        'index_stale': False,
        'ledger_fallback_used': True,
        'index_error': index_error,
        'errors': errors,
    })
