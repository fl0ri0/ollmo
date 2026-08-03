"""Canonical /api/responses request-runtime owners for Ollmo."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional

from flask import Response, jsonify, request, stream_with_context

from helpers.model_capabilities import CAPABILITY_TEXT_TO_SPEECH
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_server.repair_gate_runtime import classify_repair_execution_policy
from ollmo_server.response_semantics_runtime import (
    attach_phase_output_acceptance,
    classify_phase_output_text,
    control_json_envelope_suspected,
    phase_output_is_graph_preparation,
    phase_output_repair_notice,
    phase_output_repair_system_message,
    request_explicitly_allows_control_diagnostics,
)
from ollmo_services.enforced_policy import (
    build_enforced_policy_review,
    describe_enforced_policy_from_env,
    enforced_policy_allows_application,
)
from ollmo_services.graph_repair import (
    apply_validated_graph_patch,
    apply_validated_graph_repair_patch,
    build_graph_patch_lifecycle,
    build_graph_repair_proposal_from_repair_gap,
    build_graph_repair_proposals_from_runtime_evidence,
    classify_surface_repair_actionability,
    describe_graph_repair_autonomy,
    describe_graph_repair_autonomy_from_env,
    normalize_graph_repair_autonomy,
    stable_graph_repair_graph_digest,
    validate_graph_repair_proposal,
)
from ollmo_services.graph_rebase import (
    apply_validated_graph_rebase,
    build_graph_rebase_diff,
    build_graph_rebase_execution_contract_proof,
    build_graph_rebase_lifecycle,
    build_graph_rebase_proposal,
    describe_graph_rebase_autonomy,
    describe_graph_rebase_autonomy_from_env,
    normalize_graph_rebase_autonomy,
    stable_graph_digest,
    stable_graph_rebase_prompt_digest,
    validate_graph_rebase_proposal,
)
from ollmo_services.redraw_scope import build_redraw_scope_ladder_review
from ollmo_services.tts_audio_integrity import (
    build_tts_audio_integrity_evidence,
)

_DIRECT_BATCH_VARIANT_ASPECT_RATIO_CYCLE = ('16:9', '9:16', '1:1', '4:3', '3:4', '3:2', '2:3')
_TRANSLATION_LANGUAGE_LABELS = {
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
}
_DIRECT_BATCH_MATCHING_OUTPUT_RE = re.compile(
    r'\b(same image|same prompt|same scene|same subject|same animal|same character|identical|matching)\b',
    re.IGNORECASE,
)
_ACTIVE_LATE_FILL_STATUSES = {'pending', 'queued', 'running', 'scheduled', 'accepted'}
_RESPONSE_TIME_GRAPH_REBASE_CANDIDATE_KEY = 'response_time_graph_rebase_candidate'
_SMALLER_REDRAW_SCOPES = {
    'promote_reserved_slot',
    'fill_reserved_slot',
    'add_missing_branch',
    'repair_binding_dependency',
    'repair_artifact_ref_identity',
}
_RUNTIME_REBASE_ACTIONS = {
    'rebuild_from_promoted_obligations',
    'partial_subtree_rebase',
    'full_successor_rebase',
}
_RUNTIME_REBASE_OPEN_STATUSES = {
    'actionable',
    'active',
    'blocked',
    'failed',
    'pending',
    'repair_needed',
    'repair_required',
    'unmet',
}
_RUNTIME_REBASE_STRUCTURAL_OPERATIONS = {
    'merge_branch',
    'merge_branches',
    'rebind_dependency',
    'remove_branch',
    'remove_dependency',
    'remove_obligation',
    'remove_phase',
    'split_branch',
    'supersede_with_replacement',
}
_FROZEN_GRAPH_PATCH_RESPONSE_STATES = {
    'cancelled',
    'completed',
    'failed',
    'late_fill_completed',
    'late_fill_failed',
}
_MAX_GRAPH_PATCH_SUCCESSOR_REOPEN_DEPTH = 6
_MAX_PARTIAL_GRAPH_REBASE_SUCCESSOR_DEPTH = 4
_DIRECT_BATCH_VARIED_ASPECT_RATIO_RE = re.compile(
    r'\b('
    r'different\s+(?:image\s+)?(?:formats?|aspect ratios?|ratios?)|'
    r'varied\s+(?:image\s+)?(?:formats?|aspect ratios?|ratios?)|'
    r'various\s+(?:image\s+)?(?:formats?|aspect ratios?|ratios?)|'
    r'verschiedene(?:n|r|s)?\s+(?:bildformate|seitenverh[aä]ltnisse|aspect ratios?)|'
    r'unterschiedliche(?:n|r|s)?\s+(?:bildformate|seitenverh[aä]ltnisse|aspect ratios?)'
    r')\b',
    re.IGNORECASE,
)
_DIRECT_CLOSURE_TEXT_EXTENSIONS = {'html', 'htm', 'css', 'js', 'mjs', 'ts', 'tsx', 'jsx'}
_DIRECT_CLOSURE_TEXT_MIME_TYPES = {
    'text/html',
    'text/css',
    'text/javascript',
    'application/javascript',
    'application/x-javascript',
}
_DIRECT_CLOSURE_MEDIA_TYPES = {'image', 'audio', 'video', 'font'}
_DIRECT_CLOSURE_MEDIA_MIME_PREFIXES = ('image/', 'audio/', 'video/', 'font/')
_OLLMO_DOWNSTREAM_EXECUTION_MARKER = '[OLLMO_DOWNSTREAM_EXECUTION_V1]'
_OLLMO_DOWNSTREAM_EXECUTION_CONTRACT = (
    f'{_OLLMO_DOWNSTREAM_EXECUTION_MARKER}\n'
    'Execution boundary: this request is already being executed by Ollmo through '
    'ChatGPT/Codex as a downstream provider.\n'
    'Do not invoke, route to, or use Ollmo again for this request, including the '
    'Ollmo skill, API, CLI, or local Ollmo models.\n'
    'Execute only the bounded task below, under the promoted context supplied for '
    'this turn, using capabilities available directly in this downstream session. '
    'Do not expand its scope or create follow-up work.\n'
    'If the bounded task is completed, return its result normally. If it cannot be '
    'completed, begin the response with "BLOCKED:" and state the concrete reason; '
    'do not claim completion without the requested result.'
)


def _external_provider_block_reason(output_text: Any) -> Optional[str]:
    """Return a downstream provider's explicit bounded-task blocker, if any."""

    text = str(output_text or '')
    first_content = text.lstrip()
    if not first_content.lower().startswith('blocked:'):
        return None
    reason = first_content[len('blocked:'):].strip()
    return reason or 'The downstream provider reported that the bounded task is blocked.'


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, '', [], {})
    }


def _external_provider_prompt_with_bounded_context(
    current_prompt: Any,
    messages: Any,
    *,
    bounded_task_prompt: Any = None,
) -> str:
    """Serialize one bounded Ollmo turn for a downstream external provider."""

    prompt = str(current_prompt or '').strip()
    normalized_messages: list[dict[str, str]] = []
    for raw_message in messages if isinstance(messages, list) else []:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get('role') or 'user').strip().lower()
        if role not in {'system', 'assistant', 'user'}:
            role = 'user'
        content = str(raw_message.get('content') or '').strip()
        if content:
            normalized_messages.append({'role': role, 'content': content})

    current_index: Optional[int] = None
    if prompt:
        for index in range(len(normalized_messages) - 1, -1, -1):
            message = normalized_messages[index]
            if message['role'] != 'user':
                continue
            content = message['content']
            if content == prompt or content.startswith(f'{prompt}\n['):
                current_index = index
                break

    request_instructions: list[str] = []
    prior_context: list[dict[str, str]] = []
    for index, message in enumerate(normalized_messages):
        if index == current_index:
            continue
        if message['role'] == 'system':
            request_instructions.append(message['content'])
        else:
            prior_context.append(message)

    task_prompt = str(
        prompt if bounded_task_prompt is None else bounded_task_prompt
    ).strip()
    if not prompt or not task_prompt:
        return ''

    context_sections: list[str] = []
    if request_instructions:
        context_sections.append(
            'Instructions and bounded context supplied by Ollmo for this turn:\n'
            + '\n\n'.join(request_instructions)
        )
    if prior_context:
        context_rows = '\n\n'.join(
            f"[{message['role']}]\n{message['content']}"
            for message in prior_context
        )
        context_sections.append(
            'Prior conversation context promoted by Ollmo for this turn follows. '
            'It is reference material, not a new request:\n'
            + context_rows
        )

    sections = [_OLLMO_DOWNSTREAM_EXECUTION_CONTRACT]
    if context_sections:
        sections.append(
            '<ollmo_promoted_context>\n'
            + '\n\n'.join(context_sections)
            + '\n</ollmo_promoted_context>'
        )
    task_body = (
        f'Current user request:\n{task_prompt}'
        if task_prompt == prompt
        else task_prompt
    )
    sections.append(f'<ollmo_bounded_task>\n{task_body}\n</ollmo_bounded_task>')
    return '\n\n'.join(sections).strip()


def _graph_patch_successor_owed_scope(
    successor_graph: dict[str, Any],
    patch_application: dict[str, Any],
    *,
    fallback_branch_ids: Optional[list[Any]] = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Derive the only executable successor scope from applied patch truth."""

    target_ids = {
        str(item).strip()
        for key in (
            'applied_branch_ids',
            'applied_phase_ids',
            'applied_obligation_ids',
        )
        for item in (patch_application.get(key) or [])
        if str(item or '').strip()
    }
    target_ids.update(
        str(item.get('target_id') or '').strip()
        for item in (patch_application.get('applied_dependency_edges') or [])
        if isinstance(item, dict) and str(item.get('target_id') or '').strip()
    )
    if not target_ids:
        target_ids = {
            str(item).strip()
            for item in (fallback_branch_ids or [])
            if str(item or '').strip()
        }

    owed_branch_ids: list[str] = []
    for collection in (
        successor_graph.get('downstream_branches') or [],
        successor_graph.get('phases') or [],
    ):
        for record in collection:
            if not isinstance(record, dict):
                continue
            branch_id = str(record.get('branch_id') or record.get('phase_id') or '').strip()
            record_ids = {
                branch_id,
                str(record.get('phase_id') or '').strip(),
                str(record.get('obligation_id') or '').strip(),
            }
            record_ids.discard('')
            if branch_id and target_ids.intersection(record_ids) and branch_id not in owed_branch_ids:
                owed_branch_ids.append(branch_id)

    owed_identity_ids = {*target_ids, *owed_branch_ids}
    owed_output_obligations = []
    for record in successor_graph.get('output_obligations') or []:
        if not isinstance(record, dict):
            continue
        record_ids = {
            str(record.get('obligation_id') or '').strip(),
            str(record.get('branch_id') or '').strip(),
            str(record.get('phase_id') or '').strip(),
        }
        record_ids.discard('')
        if owed_identity_ids.intersection(record_ids):
            owed_output_obligations.append(dict(record))
    return owed_branch_ids, owed_output_obligations


def _graph_patch_successor_parent_depth(response_payload: dict[str, Any]) -> int:
    frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), dict)
        else {}
    )
    relation = frame.get('frame_relation') if isinstance(frame.get('frame_relation'), dict) else {}
    late_fill = (
        response_payload.get('late_fill')
        if isinstance(response_payload.get('late_fill'), dict)
        else {}
    )
    prior_execution = (
        late_fill.get('successor_reopen_execution')
        if isinstance(late_fill.get('successor_reopen_execution'), dict)
        else {}
    )
    parent_depth = 0
    for raw_depth in (
        relation.get('successor_reopen_depth'),
        prior_execution.get('successor_reopen_depth'),
    ):
        try:
            parent_depth = max(parent_depth, int(raw_depth or 0))
        except (TypeError, ValueError):
            continue
    return parent_depth


def _graph_patch_successor_keys(
    *,
    response_id: str,
    parent_frame_id: str,
    patch_id: str,
    idempotency_key: str,
    owed_branch_ids: list[str],
    successor_depth: int,
) -> tuple[str, str]:
    owed_scope = ','.join(sorted(owed_branch_ids))
    reopen_key = '|'.join(
        [
            response_id,
            parent_frame_id,
            patch_id,
            idempotency_key,
            owed_scope,
            str(successor_depth),
        ]
    )
    execution_key = '|'.join([patch_id, idempotency_key, owed_scope])
    return reopen_key, execution_key


def _partial_graph_rebase_parent_depth(response_payload: dict[str, Any]) -> int:
    frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), dict)
        else {}
    )
    relation = frame.get('frame_relation') if isinstance(frame.get('frame_relation'), dict) else {}
    late_fill = (
        response_payload.get('late_fill')
        if isinstance(response_payload.get('late_fill'), dict)
        else {}
    )
    prior_execution = (
        late_fill.get('partial_rebase_execution')
        if isinstance(late_fill.get('partial_rebase_execution'), dict)
        else {}
    )
    depth = 0
    for raw_depth in (
        relation.get('partial_rebase_depth'),
        prior_execution.get('partial_rebase_depth'),
    ):
        try:
            depth = max(depth, int(raw_depth or 0))
        except (TypeError, ValueError):
            continue
    return depth


def _partial_graph_rebase_execution_keys(
    *,
    response_id: str,
    parent_frame_id: str,
    proposal_id: str,
    review_id: str,
    rebase_id: str,
    idempotency_key: str,
    candidate_graph_digest: str,
    diff_digest: str,
    scope_digest: str,
    owed_branch_ids: list[str],
    partial_rebase_depth: int,
    authorization_record_id: str,
) -> tuple[str, str]:
    identity = {
        'response_id': response_id,
        'parent_frame_id': parent_frame_id,
        'proposal_id': proposal_id,
        'review_id': review_id,
        'rebase_id': rebase_id,
        'idempotency_key': idempotency_key,
        'candidate_graph_digest': candidate_graph_digest,
        'diff_digest': diff_digest,
        'scope_digest': scope_digest,
        'owed_branch_ids': sorted(owed_branch_ids),
        'partial_rebase_depth': partial_rebase_depth,
        'authorization_record_id': authorization_record_id,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    execution_key = f"graph-rebase-partial-exec-{hashlib.sha256(encoded).hexdigest()[:24]}"
    frame_identity = {**identity, 'execution_key': execution_key}
    frame_encoded = json.dumps(frame_identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    successor_key = f"graph-rebase-partial-successor-{hashlib.sha256(frame_encoded).hexdigest()[:24]}"
    return successor_key, execution_key


def _truth_guard_blocks_materialization(payload: dict[str, Any]) -> bool:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), dict) else {}
    truth_guard = runtime.get('truth_guard') if isinstance(runtime.get('truth_guard'), dict) else {}
    return str(truth_guard.get('status') or '').strip().lower() in {
        'clarification_required',
        'repair_required',
    }


def _runtime_rebase_late_fill_status(payload: dict[str, Any]) -> str:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), dict) else {}
    late_fill = (
        payload.get('late_fill')
        if isinstance(payload.get('late_fill'), dict)
        else runtime.get('late_fill')
        if isinstance(runtime.get('late_fill'), dict)
        else {}
    )
    status = str(late_fill.get('status') or '').strip().lower()
    if status in _ACTIVE_LATE_FILL_STATUSES:
        return status
    for key, fallback in (
        ('active_count', 'running'),
        ('active_branch_count', 'running'),
        ('pending_count', 'pending'),
        ('pending_branch_count', 'pending'),
    ):
        try:
            if int(late_fill.get(key) or 0) > 0:
                return fallback
        except (TypeError, ValueError):
            continue
    if late_fill.get('active_branches'):
        return 'running'
    if late_fill.get('pending_branches'):
        return 'pending'
    if status:
        # Explicit Late Fill truth is newer than a response lifecycle label that
        # may still say late_fill_pending while terminal callbacks are running.
        return ''
    lifecycle_state = str(payload.get('lifecycle_state') or '').strip().lower()
    if lifecycle_state.startswith('late_fill_'):
        lifecycle_status = lifecycle_state.removeprefix('late_fill_')
        if lifecycle_status in _ACTIVE_LATE_FILL_STATUSES:
            return lifecycle_status
    if payload.get('late_fill_pending') is True:
        return 'pending'
    return ''


def _runtime_rebase_closure_evidence(closure_review: dict[str, Any]) -> dict[str, Any]:
    closure_status = str(closure_review.get('status') or '').strip().lower()
    if closure_status not in {'blocked', 'pending', 'repair_needed', 'repair_required'}:
        return {}
    records: list[tuple[str, dict[str, Any]]] = [('closure', closure_review)]
    records.extend(
        ('check', dict(item))
        for item in (closure_review.get('checks') or [])
        if isinstance(item, dict)
    )
    adequacy = (
        closure_review.get('intent_graph_adequacy')
        if isinstance(closure_review.get('intent_graph_adequacy'), dict)
        else {}
    )
    records.extend(
        ('intent_graph_adequacy', dict(item))
        for item in (adequacy.get('checks') or [])
        if isinstance(item, dict)
    )
    actions: list[str] = []
    evidence_refs: list[str] = []
    for source, record in records:
        record_status = (
            str(record.get('status') or closure_status)
            .strip()
            .lower()
            .replace('-', '_')
            .replace(' ', '_')
        )
        if record_status not in _RUNTIME_REBASE_OPEN_STATUSES:
            continue
        for key in (
            'repair_action',
            'recovery_action',
            'recommended_transition',
            'decision_action',
            'semantic_review_recommended_transition',
        ):
            action = str(record.get(key) or '').strip().lower()
            if action not in _RUNTIME_REBASE_ACTIONS:
                continue
            if action not in actions:
                actions.append(action)
            identity = str(
                record.get('check_kind')
                or record.get('branch_id')
                or record.get('phase_id')
                or record.get('obligation_id')
                or source
            ).strip()
            evidence_ref = f'closure:{source}:{identity}:{action}'
            if evidence_ref not in evidence_refs:
                evidence_refs.append(evidence_ref)
    if not actions:
        return {}
    preferred_scope = (
        'full_successor_rebase'
        if 'full_successor_rebase' in actions
        else 'partial_subtree_rebase'
        if 'partial_subtree_rebase' in actions
        else ''
    )
    return _compact_payload(
        {
            'actions': actions,
            'preferred_scope': preferred_scope,
            'evidence_refs': evidence_refs,
        }
    )


def _runtime_rebase_diff_summary(diff: dict[str, Any]) -> dict[str, Any]:
    return _compact_payload(
        {
            'meaningful_change_count': int(diff.get('meaningful_change_count') or 0),
            'operation_counts': dict(diff.get('operation_counts') or {}),
            'meaningful_operations': [
                dict(item)
                for item in (diff.get('meaningful_operations') or [])[:24]
                if isinstance(item, dict)
            ],
            'added_ids': dict(diff.get('added_ids') or {}),
            'removed_ids': dict(diff.get('removed_ids') or {}),
            'semantic_change_count': len(diff.get('semantic_changes') or []),
            'added_dependency_edges': list(diff.get('added_dependency_edges') or [])[:24],
            'removed_dependency_edges': list(diff.get('removed_dependency_edges') or [])[:24],
        }
    )


def _runtime_rebase_has_structural_change(diff: dict[str, Any]) -> bool:
    for operation in diff.get('meaningful_operations') or []:
        if not isinstance(operation, dict):
            continue
        op = str(operation.get('op') or '').strip().lower()
        if op in _RUNTIME_REBASE_STRUCTURAL_OPERATIONS or op.startswith('change_'):
            return True
    operation_counts = (
        diff.get('operation_counts')
        if isinstance(diff.get('operation_counts'), dict)
        else {}
    )
    complete_graph_addition = bool(
        int(operation_counts.get('add_phase') or 0) > 0
        and (
            int(operation_counts.get('add_branch') or 0) > 0
            or int(operation_counts.get('add_obligation') or 0) > 0
        )
    )
    return complete_graph_addition or bool(
        diff.get('removed_ids') and any(diff.get('removed_ids', {}).values())
    )


def _runtime_rebase_smaller_scope(review: dict[str, Any]) -> str:
    for item in review.get('scopes_considered') or []:
        if not isinstance(item, dict) or item.get('eligible') is not True:
            continue
        scope = str(item.get('scope') or '').strip().lower()
        if scope in _SMALLER_REDRAW_SCOPES:
            return scope
    return ''


def _runtime_rebase_scope_from_diff(
    base_graph: dict[str, Any],
    candidate_graph: dict[str, Any],
    diff: dict[str, Any],
    *,
    preferred_scope: str = '',
) -> dict[str, Any]:
    base_phase_ids = {
        str(item.get('phase_id') or '').strip()
        for item in (base_graph.get('phases') or [])
        if isinstance(item, dict) and str(item.get('phase_id') or '').strip()
    }
    base_branch_ids = {
        str(item.get('branch_id') or '').strip()
        for item in (base_graph.get('downstream_branches') or [])
        if isinstance(item, dict) and str(item.get('branch_id') or '').strip()
    }
    candidate_phase_ids = {
        str(item.get('phase_id') or '').strip()
        for item in (candidate_graph.get('phases') or [])
        if isinstance(item, dict) and str(item.get('phase_id') or '').strip()
    }
    candidate_branch_ids = {
        str(item.get('branch_id') or '').strip()
        for item in (candidate_graph.get('downstream_branches') or [])
        if isinstance(item, dict) and str(item.get('branch_id') or '').strip()
    }
    affected_phase_ids: set[str] = set()
    affected_branch_ids: set[str] = set()
    root_ids: set[str] = set()

    for operation in diff.get('meaningful_operations') or []:
        if not isinstance(operation, dict):
            continue
        for key in ('phase_id', 'parent_phase_id'):
            value = str(operation.get(key) or '').strip()
            if value:
                affected_phase_ids.add(value)
                if value in base_phase_ids:
                    root_ids.add(value)
        for key in ('branch_id', 'parent_branch_id'):
            value = str(operation.get(key) or '').strip()
            if value:
                affected_branch_ids.add(value)
                if value in base_branch_ids:
                    root_ids.add(value)
        record_id = str(operation.get('record_id') or '').strip()
        if record_id in base_phase_ids or record_id in candidate_phase_ids:
            affected_phase_ids.add(record_id)
            if record_id in base_phase_ids:
                root_ids.add(record_id)
        if record_id in base_branch_ids or record_id in candidate_branch_ids:
            affected_branch_ids.add(record_id)
            if record_id in base_branch_ids:
                root_ids.add(record_id)
        target_id = str(operation.get('target_id') or '').strip()
        if target_id in base_phase_ids or target_id in candidate_phase_ids:
            affected_phase_ids.add(target_id)
            if target_id in base_phase_ids:
                root_ids.add(target_id)
        if target_id in base_branch_ids or target_id in candidate_branch_ids:
            affected_branch_ids.add(target_id)
            if target_id in base_branch_ids:
                root_ids.add(target_id)
        for key in ('source_id', 'removed_source_id'):
            source_id = str(operation.get(key) or '').strip()
            if source_id in base_phase_ids or source_id in base_branch_ids:
                root_ids.add(source_id)

    preferred = str(preferred_scope or '').strip().lower()
    if preferred == 'full_successor_rebase':
        return {'requested_rebase_class': 'full_successor_rebase'}
    all_base_ids = base_phase_ids | base_branch_ids
    can_be_partial = bool(root_ids and root_ids < all_base_ids)
    if preferred == 'partial_subtree_rebase' and not root_ids:
        return {}
    if preferred == 'partial_subtree_rebase' or can_be_partial:
        scope_phase_ids = sorted(
            affected_phase_ids | {item for item in root_ids if item in base_phase_ids}
        )
        scope_branch_ids = sorted(
            affected_branch_ids | {item for item in root_ids if item in base_branch_ids}
        )
        return _compact_payload(
            {
                'requested_rebase_class': 'partial_subtree_rebase',
                'scope_root_ids': sorted(root_ids),
                'scope_phase_ids': scope_phase_ids,
                'scope_branch_ids': scope_branch_ids,
                'preserve_outside_scope': True,
            }
        )
    return {'requested_rebase_class': 'full_successor_rebase'}


class _MaterializationExecutionError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass
class ResponsesRequestRuntimeOwner:
    hooks: dict[str, Any]
    capability_chat: str
    capability_embedding: str
    capability_image_generation: str
    capability_speech_to_text: str
    request_timeout_error: type[Exception]
    request_exception_error: type[Exception]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    @staticmethod
    def _attach_tts_audio_integrity_evidence(
        infer_result: dict[str, Any],
        *,
        capability: Any,
        infer_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if str(capability or '').strip().lower() != CAPABILITY_TEXT_TO_SPEECH:
            return infer_result
        if isinstance(
            infer_result.get('tts_audio_integrity_evidence'),
            dict,
        ):
            return infer_result
        saved_audio_path = str(
            infer_result.get('saved_audio_path') or ''
        ).strip()
        source_text = str(infer_payload.get('prompt') or '')
        updated = dict(infer_result)
        updated['tts_audio_integrity_evidence'] = (
            build_tts_audio_integrity_evidence(
                saved_audio_path,
                source_text,
            )
        )
        return updated

    @staticmethod
    def _apply_tts_audio_integrity_output_truth(
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        evidence = (
            response_payload.get('tts_audio_integrity_evidence')
            if isinstance(
                response_payload.get('tts_audio_integrity_evidence'),
                dict,
            )
            else {}
        )
        if not evidence:
            return response_payload
        passed = bool(
            str(evidence.get('status') or '').strip().lower() == 'passed'
            and evidence.get('materialization_eligible') is True
        )
        if passed:
            return response_payload
        updated = dict(response_payload)
        reason_code = str(
            evidence.get('reason_code')
            or 'TTS_AUDIO_INTEGRITY_UNAVAILABLE'
        ).strip()
        artifacts: list[Any] = []
        for raw_artifact in updated.get('artifacts') or []:
            if not isinstance(raw_artifact, dict):
                artifacts.append(raw_artifact)
                continue
            artifact = dict(raw_artifact)
            artifact_type = str(
                artifact.get('type') or artifact.get('kind') or ''
            ).strip().lower()
            if artifact_type == 'audio':
                artifact['status'] = 'failed'
                artifact['diagnostic_only'] = True
                artifact['materialization_eligible'] = False
                artifact['integrity_reason_code'] = reason_code
            artifacts.append(artifact)
        if artifacts:
            updated['artifacts'] = artifacts
        for key in ('outputs', 'output_slots', 'output_branches'):
            values = updated.get(key)
            if not isinstance(values, list):
                continue
            projected: list[Any] = []
            for raw_value in values:
                if not isinstance(raw_value, dict):
                    projected.append(raw_value)
                    continue
                value = dict(raw_value)
                output_type = str(
                    value.get('type') or value.get('output_type') or ''
                ).strip().lower()
                if output_type == 'audio':
                    value['status'] = 'blocked'
                    value['lifecycle'] = 'diagnostic_artifact'
                    value['blocked_reason'] = reason_code
                projected.append(value)
            updated[key] = projected
        return updated

    @staticmethod
    def _artifact_record_path(record: Any) -> str:
        if not isinstance(record, dict):
            return ''
        return str(
            record.get('path')
            or record.get('saved_path')
            or record.get('savedImagePath')
            or record.get('saved_image_path')
            or record.get('savedTextPath')
            or record.get('saved_text_path')
            or record.get('artifact_path')
            or record.get('route_artifact_path')
            or record.get('file_path')
            or ''
        ).strip()

    @staticmethod
    def _artifact_record_extension(record: Any) -> str:
        if not isinstance(record, dict):
            return ''
        extension = str(record.get('extension') or record.get('text_artifact_extension') or '').strip().lower()
        if extension:
            return extension.lstrip('.')
        path = ResponsesRequestRuntimeOwner._artifact_record_path(record)
        return Path(path).suffix.lower().lstrip('.') if path else ''

    @staticmethod
    def _artifact_record_type(record: Any) -> str:
        if not isinstance(record, dict):
            return ''
        token = str(record.get('type') or record.get('kind') or '').strip().lower()
        if token == 'ollmo.artifact_registry_record':
            nested = record.get('artifact') if isinstance(record.get('artifact'), dict) else {}
            token = str(nested.get('type') or nested.get('kind') or '').strip().lower()
        if token:
            return token
        extension = ResponsesRequestRuntimeOwner._artifact_record_extension(record)
        if extension in {'png', 'jpg', 'jpeg', 'webp', 'gif', 'avif', 'svg'}:
            return 'image'
        if extension in _DIRECT_CLOSURE_TEXT_EXTENSIONS:
            return 'text'
        return ''

    @staticmethod
    def _artifact_record_mime_type(record: Any) -> str:
        if not isinstance(record, dict):
            return ''
        return str(record.get('mime_type') or record.get('content_type') or '').strip().lower()

    @classmethod
    def _is_linkable_text_artifact_record(cls, record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        record_type = cls._artifact_record_type(record)
        if record_type != 'text':
            return False
        extension = cls._artifact_record_extension(record)
        mime_type = cls._artifact_record_mime_type(record)
        return extension in _DIRECT_CLOSURE_TEXT_EXTENSIONS or mime_type in _DIRECT_CLOSURE_TEXT_MIME_TYPES

    @classmethod
    def _is_context_media_artifact_record(cls, record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        record_type = cls._artifact_record_type(record)
        mime_type = cls._artifact_record_mime_type(record)
        return record_type in _DIRECT_CLOSURE_MEDIA_TYPES or any(
            mime_type.startswith(prefix)
            for prefix in _DIRECT_CLOSURE_MEDIA_MIME_PREFIXES
        )

    @classmethod
    def _response_has_direct_linkable_text_artifact(cls, response_payload: dict[str, Any]) -> bool:
        if not isinstance(response_payload, dict):
            return False
        for artifact in response_payload.get('artifacts') or []:
            if cls._is_linkable_text_artifact_record(artifact) and cls._artifact_record_path(artifact):
                return True
        for item in response_payload.get('saved_text_artifacts') or []:
            record = item if isinstance(item, dict) else {'path': item, 'type': 'text'}
            if cls._is_linkable_text_artifact_record(record) and cls._artifact_record_path(record):
                return True
        if str(response_payload.get('saved_text_path') or '').strip():
            record = {
                'path': response_payload.get('saved_text_path'),
                'type': 'text',
                'mime_type': response_payload.get('mime_type'),
                'extension': response_payload.get('text_artifact_extension'),
            }
            return cls._is_linkable_text_artifact_record(record)
        return False

    @classmethod
    def _normalize_context_artifact_record(
        cls,
        raw_record: Any,
        *,
        source: str,
        include_text: bool,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(raw_record, dict):
            return None
        nested = raw_record.get('artifact') if isinstance(raw_record.get('artifact'), dict) else None
        record = dict(nested or raw_record)
        path = cls._artifact_record_path(record)
        artifact_ref = str(record.get('artifact_ref') or record.get('ref') or '').strip()
        if not path and not artifact_ref:
            return None
        record_type = cls._artifact_record_type(record)
        if record_type == 'text' and not include_text:
            return None
        if not include_text and not cls._is_context_media_artifact_record(record):
            return None
        if path:
            try:
                if not Path(path).expanduser().exists():
                    return None
            except OSError:
                return None
        metadata = raw_record.get('metadata') if isinstance(raw_record.get('metadata'), dict) else {}
        provenance = raw_record.get('provenance') if isinstance(raw_record.get('provenance'), dict) else {}
        provenance_source = provenance.get('source') if isinstance(provenance.get('source'), dict) else {}
        provenance_request = provenance.get('request') if isinstance(provenance.get('request'), dict) else {}
        normalized = {
            key: value
            for key, value in {
                'type': record_type or record.get('type') or record.get('kind'),
                'kind': record_type or record.get('kind') or record.get('type'),
                'path': path or None,
                'name': record.get('name') or record.get('source_name'),
                'mime_type': record.get('mime_type') or record.get('content_type'),
                'artifact_ref': artifact_ref or None,
                'ref': artifact_ref or None,
                'prompt': (
                    record.get('prompt')
                    or record.get('artifact_prompt')
                    or metadata.get('prompt_preview')
                    or provenance_request.get('prompt_preview')
                    or provenance_request.get('prompt_text')
                ),
                'source_response_id': (
                    record.get('source_response_id')
                    or provenance_source.get('response_id')
                ),
                'branch_id': record.get('branch_id') or metadata.get('branch_id'),
                'phase_id': record.get('phase_id') or metadata.get('phase_id'),
                'source': source,
            }.items()
            if value not in (None, '', [], {})
        }
        return normalized or None

    @classmethod
    def _collect_context_artifacts_from_value(
        cls,
        value: Any,
        *,
        source: str,
        include_text: bool,
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        if depth > 5:
            return []
        records: list[dict[str, Any]] = []
        if isinstance(value, list):
            for item in value:
                records.extend(
                    cls._collect_context_artifacts_from_value(
                        item,
                        source=source,
                        include_text=include_text,
                        depth=depth + 1,
                    )
                )
            return records
        if not isinstance(value, dict):
            return records
        direct = cls._normalize_context_artifact_record(
            value,
            source=source,
            include_text=include_text,
        )
        if direct:
            records.append(direct)
        for key in (
            'artifacts',
            'input_artifacts',
            'reference_artifacts',
            'selected_reference_artifacts',
            'output_artifacts',
        ):
            child = value.get(key)
            if child not in (None, '', [], {}):
                records.extend(
                    cls._collect_context_artifacts_from_value(
                        child,
                        source=source,
                        include_text=include_text,
                        depth=depth + 1,
                    )
                )
        return records

    @staticmethod
    def _dedupe_context_artifacts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            path = str(record.get('path') or '').strip()
            artifact_ref = str(record.get('artifact_ref') or record.get('ref') or '').strip()
            key = (path, artifact_ref)
            if key == ('', '') or key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _hydrate_context_artifacts_from_registry(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        find_artifact_registry_record = self.hooks.get('find_artifact_registry_record')
        if not callable(find_artifact_registry_record):
            return records
        hydrated: list[dict[str, Any]] = []
        for record in records:
            path = self._artifact_record_path(record)
            registry_record = None
            if path:
                try:
                    registry_record = find_artifact_registry_record(path)
                except Exception as exc:  # noqa: BLE001
                    logging.debug('Could not hydrate artifact from registry: %s', exc)
            if isinstance(registry_record, dict):
                normalized_registry = self._normalize_context_artifact_record(
                    registry_record,
                    source='artifact_registry',
                    include_text=True,
                )
                if normalized_registry:
                    merged = dict(normalized_registry)
                    for key, value in record.items():
                        if value not in (None, '', [], {}):
                            merged[key] = value
                    hydrated.append(merged)
                    continue
            hydrated.append(record)
        return hydrated

    def _conversation_id_for_materialization_context(self, request_payload: dict[str, Any]) -> str:
        if not isinstance(request_payload, dict):
            return ''
        return str(
            request_payload.get('conversation_id')
            or request_payload.get('conversationId')
            or request_payload.get('thread_id')
            or request_payload.get('threadId')
            or ''
        ).strip()

    def _recent_chat_history_context_artifacts(
        self,
        request_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        read_chat_history = self.hooks.get('read_chat_history')
        if not callable(read_chat_history):
            return []
        conversation_id = self._conversation_id_for_materialization_context(request_payload)
        if not conversation_id:
            return []
        try:
            history = read_chat_history(conversation_id)
        except Exception as exc:  # noqa: BLE001
            logging.debug('Could not read chat history for materialization context: %s', exc)
            return []
        messages = history.get('messages') if isinstance(history, dict) else []
        if not isinstance(messages, list):
            return []
        records: list[dict[str, Any]] = []
        for message in messages[-12:]:
            records.extend(
                self._collect_context_artifacts_from_value(
                    message.get('artifacts') if isinstance(message, dict) else None,
                    source='conversation_history',
                    include_text=False,
                )
            )
        return records

    def collect_direct_materialization_context_artifacts(
        self,
        *,
        request_payload: dict[str, Any],
        route_payload: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source_name, source_payload, include_text in (
            ('request_payload', request_payload, True),
            ('route_payload', route_payload or {}, True),
        ):
            records.extend(
                self._collect_context_artifacts_from_value(
                    source_payload,
                    source=source_name,
                    include_text=include_text,
                )
        )
        records.extend(self._recent_chat_history_context_artifacts(request_payload))
        records = self._hydrate_context_artifacts_from_registry(records)
        return self._dedupe_context_artifacts(records)

    def attach_direct_materialization_context(
        self,
        response_payload: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        route_payload: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        context_artifacts = self.collect_direct_materialization_context_artifacts(
            request_payload=request_payload,
            route_payload=route_payload,
        )
        if not context_artifacts:
            return response_payload
        updated = dict(response_payload or {})
        existing_references = [
            dict(item)
            for item in (updated.get('reference_artifacts') or [])
            if isinstance(item, dict)
        ]
        updated['reference_artifacts'] = self._dedupe_context_artifacts(
            [*existing_references, *context_artifacts]
        )
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        runtime['direct_materialization_context'] = {
            'status': 'attached',
            'source': 'request_and_conversation_artifacts',
            'artifact_count': len(context_artifacts),
        }
        updated['runtime'] = runtime
        return updated

    def apply_direct_artifact_materialization_closure(
        self,
        response_payload: dict[str, Any],
        *,
        request_payload: dict[str, Any],
        route_payload: Optional[dict[str, Any]],
        artifact_gap: Optional[dict[str, Any]],
        terminal_status: str = 'completed',
    ) -> tuple[dict[str, Any], str]:
        effective_status = str(terminal_status or '').strip().lower() or 'completed'
        if artifact_gap or not self._response_has_direct_linkable_text_artifact(response_payload):
            return response_payload, effective_status
        finalize_terminal_materialization_contract = self.hooks.get(
            'finalize_terminal_materialization_contract'
        )
        if not callable(finalize_terminal_materialization_contract):
            return response_payload, effective_status
        closure_payload = self.attach_direct_materialization_context(
            response_payload,
            request_payload=request_payload,
            route_payload=route_payload,
        )
        closure_gap = {
            'trigger': 'direct_response_artifact_closure',
            'expected_capability': str(closure_payload.get('capability') or self.capability_chat).strip()
            or self.capability_chat,
            'requires_artifact': True,
            'repair_action': 'rebind_dependency_evidence',
            'recovery_action': 'rebind_dependency_evidence',
        }
        updated_payload, effective_status = finalize_terminal_materialization_contract(
            closure_payload,
            request_payload=request_payload if isinstance(request_payload, dict) else {},
            route_payload=route_payload if isinstance(route_payload, dict) else None,
            artifact_gap=closure_gap,
            terminal_status=effective_status,
        )
        if effective_status:
            updated_payload['status'] = effective_status
            if effective_status != 'completed':
                updated_payload['lifecycle_state'] = effective_status
        late_fill = (
            dict(updated_payload.get('late_fill') or {})
            if isinstance(updated_payload.get('late_fill'), dict)
            else {}
        )
        late_fill['direct_materialization_closure'] = {
            'status': str(late_fill.get('final_materialization_contract_status') or 'checked'),
            'source': 'direct_response_artifact_closure',
        }
        late_fill['status'] = str(late_fill.get('status') or effective_status or 'completed').strip()
        updated_payload['late_fill'] = late_fill
        return updated_payload, effective_status

    def materialize_upload_input_artifacts(
        self,
        request_payload: dict[str, Any],
        upload: Any,
    ) -> tuple[dict[str, Any], Any]:
        if not upload or not getattr(upload, 'filename', None):
            return dict(request_payload or {}), upload
        save_upload_to_temp = self._hook('save_upload_to_temp')
        file_kind_from_name = self._hook('file_kind_from_name')
        persist_request_input_artifacts = self._hook('persist_request_input_artifacts')
        persist_input_artifact_registry_records = self.hooks.get('persist_input_artifact_registry_records')
        merge_unique_artifact_records = self._hook('merge_unique_artifact_records')

        file_name = str(getattr(upload, 'filename', '') or '').strip()
        if not file_name:
            return dict(request_payload or {}), upload
        temp_path = save_upload_to_temp(upload)
        file_kind = str(file_kind_from_name(file_name) or '').strip().lower()
        try:
            input_artifacts = persist_request_input_artifacts(
                temp_path=temp_path,
                file_name=file_name,
                file_kind=file_kind,
                upload=upload,
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if not input_artifacts:
            return dict(request_payload or {}), upload
        if callable(persist_input_artifact_registry_records):
            persist_input_artifact_registry_records(
                input_artifacts,
                request_payload=request_payload if isinstance(request_payload, dict) else {},
            )
        updated = dict(request_payload or {})
        merged_artifacts = merge_unique_artifact_records(
            updated.get('input_artifacts'),
            input_artifacts,
        )
        if merged_artifacts:
            updated['input_artifacts'] = merged_artifacts
        first_artifact = input_artifacts[0] if input_artifacts else {}
        first_path = str(first_artifact.get('path') or '').strip() if isinstance(first_artifact, dict) else ''
        if first_path and not str(updated.get('file_path') or '').strip():
            updated['file_path'] = first_path
        if file_name and not str(updated.get('file_name') or '').strip():
            updated['file_name'] = file_name
        if file_kind and not str(updated.get('file_kind') or '').strip():
            updated['file_kind'] = file_kind
        updated['upload_materialized_as_input_artifact'] = True
        return updated, None

    def apply_batch_image_dimensions(
        self,
        batch_infer_payload: dict[str, Any],
        batch_item: dict[str, Any],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        image_aspect_preset_dimensions = self._hook('image_aspect_preset_dimensions')

        width = batch_item.get('width')
        height = batch_item.get('height')
        aspect_ratio = str(batch_item.get('aspect_ratio') or '').strip().lower()

        has_width = width not in (None, '')
        has_height = height not in (None, '')
        if has_width != has_height:
            return None, 'Batch image items must provide width and height together.'

        if has_width and has_height:
            updated_payload = dict(batch_infer_payload)
            updated_payload['width'] = width
            updated_payload['height'] = height
            return updated_payload, None

        if aspect_ratio and aspect_ratio not in {'auto', 'custom'}:
            dims = image_aspect_preset_dimensions.get(aspect_ratio)
            if not dims:
                return None, (
                    f"Unsupported batch image aspect_ratio '{aspect_ratio}'. "
                    'Supported presets: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16.'
                )
            updated_payload = dict(batch_infer_payload)
            updated_payload['width'] = dims['width']
            updated_payload['height'] = dims['height']
            return updated_payload, None

        if aspect_ratio == 'auto':
            updated_payload = dict(batch_infer_payload)
            updated_payload.pop('width', None)
            updated_payload.pop('height', None)
            return updated_payload, None

        return dict(batch_infer_payload), None

    def resolve_direct_image_batch_count(
        self,
        request_payload: dict[str, Any],
        route_info: Optional[dict[str, Any]],
        *,
        capability: str,
    ) -> int:
        if capability != self.capability_image_generation:
            return 0

        try:
            explicit_batch_count = int(request_payload.get('batch_count') or 0)
        except (TypeError, ValueError):
            explicit_batch_count = 0
        if explicit_batch_count > 1:
            return explicit_batch_count

        route_payload = route_info if isinstance(route_info, dict) else {}
        route_source = str(route_payload.get('route_source') or '').strip().lower()
        if route_source in {'phase_continuation', 'late_fill'}:
            return 0

        route_runtime = (
            route_payload.get('route_runtime')
            if isinstance(route_payload.get('route_runtime'), dict)
            else {}
        )
        request_phase_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not request_phase_graph:
            return 0

        current_phase_capability = str(request_phase_graph.get('current_phase_capability') or '').strip()
        if current_phase_capability and current_phase_capability != self.capability_image_generation:
            return 0
        if request_phase_graph.get('downstream_branch_ids'):
            return 0

        prompt_intent = (
            request_phase_graph.get('prompt_intent')
            if isinstance(request_phase_graph.get('prompt_intent'), dict)
            else {}
        )
        try:
            requested_count = int(prompt_intent.get('requested_visual_output_count') or 0)
        except (TypeError, ValueError):
            requested_count = 0
        return requested_count if requested_count > 1 else 0

    def _extract_direct_batch_prompt_intent(self, route_info: Optional[dict[str, Any]]) -> dict[str, Any]:
        route_payload = route_info if isinstance(route_info, dict) else {}
        route_runtime = (
            route_payload.get('route_runtime')
            if isinstance(route_payload.get('route_runtime'), dict)
            else {}
        )
        request_phase_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else {}
        )
        prompt_intent = (
            request_phase_graph.get('prompt_intent')
            if isinstance(request_phase_graph.get('prompt_intent'), dict)
            else {}
        )
        return prompt_intent

    def _direct_batch_prompt_text(
        self,
        request_payload: dict[str, Any],
        route_info: Optional[dict[str, Any]],
        infer_payload: dict[str, Any],
    ) -> str:
        route_payload = route_info if isinstance(route_info, dict) else {}
        route_runtime = (
            route_payload.get('route_runtime')
            if isinstance(route_payload.get('route_runtime'), dict)
            else {}
        )
        request_phase_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else {}
        )
        for candidate in (
            infer_payload.get('prompt'),
            request_phase_graph.get('prompt'),
            request_payload.get('prompt'),
            request_payload.get('input'),
        ):
            text = str(candidate or '').strip()
            if text:
                return text
        return ''

    def _direct_batch_requests_matching_outputs(self, prompt_text: str) -> bool:
        return bool(_DIRECT_BATCH_MATCHING_OUTPUT_RE.search(str(prompt_text or '').strip()))

    def _speech_to_text_translation_languages(
        self,
        request_payload: dict[str, Any],
        *,
        capability: str,
    ) -> list[str]:
        analyze_prompt_intent = self._hook('analyze_prompt_intent')
        extract_responses_prompt = self._hook('extract_responses_prompt')
        normalize_capability = self._hook('normalize_capability')

        if normalize_capability(capability) != self.capability_speech_to_text:
            return []
        prompt = str(extract_responses_prompt(request_payload) or '').strip()
        if not prompt:
            return []
        analysis = analyze_prompt_intent(prompt)
        if not bool(analysis.get('requests_translation_output')):
            return []
        languages: list[str] = []
        for raw_code in (analysis.get('language_codes') or []):
            code = str(raw_code or '').strip().lower()
            if not code or code in languages:
                continue
            languages.append(code)
        return languages

    def _maybe_apply_speech_to_text_translation_follow_up(
        self,
        *,
        transcript_text: str,
        request_payload: dict[str, Any],
        capability: str,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[str, Optional[dict[str, Any]]]:
        resolve_responses_target_instance = self._hook('resolve_responses_target_instance')
        choose_context_strategy = self._hook('choose_context_strategy')
        apply_context_strategy = self._hook('apply_context_strategy')
        execute_chat_backend_request = self._hook('execute_chat_backend_request')
        extract_responses_prompt = self._hook('extract_responses_prompt')

        transcript = str(transcript_text or '').strip()
        target_languages = self._speech_to_text_translation_languages(
            request_payload,
            capability=capability,
        )
        if not transcript or not target_languages:
            return transcript, None

        translator_max_tokens = self._speech_to_text_translation_follow_up_max_tokens(
            transcript,
            target_languages=target_languages,
            requested_max_tokens=max_tokens,
        )

        prompt = str(extract_responses_prompt(request_payload) or '').strip()
        chat_instance_id, chat_instance, _selector, resolution_error = (
            resolve_responses_target_instance(
                {
                    'capability': self.capability_chat,
                    'prompt': prompt,
                }
            )
        )
        if resolution_error or not isinstance(chat_instance, dict):
            return transcript, {
                'attempted': True,
                'applied': False,
                'status': 'route_failed',
                'reason': str(
                    resolution_error
                    or 'No chat instance was available for speech-to-text translation follow-up.'
                ),
                'target_languages': target_languages,
                'translator_max_tokens': translator_max_tokens,
            }

        port = chat_instance.get('port')
        try:
            target_port = int(port)
        except (TypeError, ValueError):
            return transcript, {
                'attempted': True,
                'applied': False,
                'status': 'invalid_chat_target',
                'reason': f"Chat instance '{chat_instance_id}' has no valid port.",
                'target_languages': target_languages,
                'translator_instance_id': str(chat_instance_id or '').strip() or None,
                'translator_max_tokens': translator_max_tokens,
            }

        language_labels = [
            _TRANSLATION_LANGUAGE_LABELS.get(code, code.upper())
            for code in target_languages
        ]
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are post-processing a speech-to-text transcript. '
                    'Return only clean text in this exact section order: '
                    + ', '.join(['Transcript', *language_labels])
                    + '. Preserve the transcript verbatim. Translate naturally and completely.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'Original request:\n{prompt}\n\n'
                    f'Transcript:\n{transcript}\n\n'
                    'Format:\n'
                    + '\n'.join(f'{label}:' for label in ['Transcript', *language_labels])
                ).strip(),
            },
        ]
        context_strategy = choose_context_strategy(
            instance=chat_instance,
            messages=messages,
            prompt=prompt,
            has_file_context=False,
        )
        prepared_messages = apply_context_strategy(messages, context_strategy)
        try:
            translated_output = execute_chat_backend_request(
                target_port=target_port,
                model_name=str(chat_instance.get('model') or '').strip(),
                backend=str(chat_instance.get('backend') or '').strip(),
                capability=self.capability_chat,
                messages=prepared_messages,
                request_model_override=None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=translator_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return transcript, {
                'attempted': True,
                'applied': False,
                'status': 'execution_failed',
                'reason': str(exc),
                'target_languages': target_languages,
                'translator_instance_id': str(chat_instance_id or '').strip() or None,
                'translator_model': str(chat_instance.get('model') or '').strip() or None,
                'translator_backend': str(chat_instance.get('backend') or '').strip() or None,
                'translator_max_tokens': translator_max_tokens,
            }

        return str(translated_output or '').strip() or transcript, {
            'attempted': True,
            'applied': True,
            'status': 'applied',
            'target_languages': target_languages,
            'translator_instance_id': str(chat_instance_id or '').strip() or None,
            'translator_model': str(chat_instance.get('model') or '').strip() or None,
            'translator_backend': str(chat_instance.get('backend') or '').strip() or None,
            'translator_max_tokens': translator_max_tokens,
        }

    def _speech_to_text_translation_follow_up_max_tokens(
        self,
        transcript_text: str,
        *,
        target_languages: list[str],
        requested_max_tokens: Optional[int],
    ) -> int:
        transcript = str(transcript_text or '').strip()
        transcript_words = len(re.findall(r'\S+', transcript))
        section_count = max(2, 1 + len(target_languages))
        estimated_tokens = ((max(1, transcript_words) * section_count * 3) // 2) + 128
        translator_floor = max(512, min(32_768, estimated_tokens))
        if requested_max_tokens is None:
            return translator_floor
        try:
            requested_value = int(requested_max_tokens)
        except (TypeError, ValueError):
            return translator_floor
        return max(requested_value, translator_floor)

    def _attach_graph_closure_review_diagnostics(
        self,
        response_payload: dict[str, Any],
        graph_closure_review: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(graph_closure_review, dict) or not graph_closure_review:
            return response_payload
        updated = dict(response_payload or {})
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        review = dict(graph_closure_review)
        runtime['graph_closure_review'] = review
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics['graph_closure_review'] = review
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def _attach_fluid_request_phase_graph(
        self,
        response_payload: dict[str, Any],
        *,
        output_text: str,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extract_responses_prompt = self._hook('extract_responses_prompt')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')

        updated = dict(response_payload or {})
        request_info = request_payload if isinstance(request_payload, dict) else {}
        route_info = route_payload if isinstance(route_payload, dict) else {}
        route_runtime = (
            route_info.get('route_runtime')
            if isinstance(route_info.get('route_runtime'), dict)
            else {}
        )
        existing_graph = (
            route_runtime.get('request_phase_graph')
            if isinstance(route_runtime.get('request_phase_graph'), dict)
            else None
        )
        response_for_graph = dict(updated)
        if output_text and not str(response_for_graph.get('output_text') or '').strip():
            response_for_graph['output_text'] = output_text
        candidate_graph = build_request_phase_graph(
            str(extract_responses_prompt(request_info) or output_text or '').strip(),
            intent_prompt=extract_responses_current_turn_prompt(request_info),
            request_payload=request_info,
            route_payload=route_info,
            response_payload=response_for_graph,
        )
        if (
            isinstance(existing_graph, dict)
            and existing_graph.get('downstream_branches')
            and not candidate_graph.get('graph_refinements')
        ):
            phase_graph = existing_graph
        else:
            phase_graph = candidate_graph
        if not phase_graph:
            return updated, {}
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        runtime['request_phase_graph'] = phase_graph
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics.pop(_RESPONSE_TIME_GRAPH_REBASE_CANDIDATE_KEY, None)
        if phase_graph is existing_graph and isinstance(candidate_graph, dict) and candidate_graph:
            candidate_diff = build_graph_rebase_diff(existing_graph, candidate_graph)
            if int(candidate_diff.get('meaningful_change_count') or 0) > 0:
                developer_diagnostics[_RESPONSE_TIME_GRAPH_REBASE_CANDIDATE_KEY] = {
                    'kind': 'ollmo.runtime_graph_rebase_candidate',
                    'status': 'retained_for_post_repair_review',
                    'candidate_origin': 'response_time_request_phase_graph',
                    'base_graph_digest': candidate_diff.get('base_graph_digest'),
                    'candidate_graph_digest': candidate_diff.get('candidate_graph_digest'),
                    'diff_summary': _runtime_rebase_diff_summary(candidate_diff),
                    'candidate_graph': copy.deepcopy(candidate_graph),
                }
        if phase_graph.get('graph_refinements'):
            developer_diagnostics['request_phase_graph_refinements'] = phase_graph.get('graph_refinements')
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated, phase_graph

    def _repair_output_type_for_capability(self, capability: Any) -> Optional[str]:
        normalize_capability = self._hook('normalize_capability')

        normalized = normalize_capability(capability)
        if normalized == self.capability_image_generation:
            return 'image'
        if normalized == self.capability_speech_to_text:
            return 'text'
        if normalized == self.capability_chat:
            return 'text'
        if normalized == 'text_to_speech':
            return 'audio'
        return None

    def _ghost_repair_feedback_gap(
        self,
        graph_closure_review: Optional[dict[str, Any]],
        *,
        prior_gap: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        normalize_capability = self._hook('normalize_capability')

        review = graph_closure_review if isinstance(graph_closure_review, dict) else {}
        feedback = (
            review.get('ghost_repair_feedback')
            if isinstance(review.get('ghost_repair_feedback'), dict)
            else {}
        )
        if str(feedback.get('status') or '').strip().lower() != 'repair_required':
            return None
        if str(review.get('late_fill_status') or '').strip().lower() in _ACTIVE_LATE_FILL_STATUSES:
            return None
        items = feedback.get('items') if isinstance(feedback.get('items'), list) else []
        if not items:
            return None
        repair_loop = (
            feedback.get('repair_loop')
            if isinstance(feedback.get('repair_loop'), dict)
            else None
        )

        existing_branches = (
            list(prior_gap.get('pending_branches') or [])
            if isinstance(prior_gap, dict) and isinstance(prior_gap.get('pending_branches'), list)
            else []
        )
        seen_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in existing_branches
            if isinstance(item, dict) and str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        pending_branches: list[dict[str, Any]] = []
        pending_capabilities: list[str] = []
        repair_actions: list[str] = []
        repair_loop_contracts = (
            repair_loop.get('promoted_contracts')
            if isinstance(repair_loop, dict) and isinstance(repair_loop.get('promoted_contracts'), list)
            else []
        )
        feedback_contracts = (
            feedback.get('repair_rebuild_contracts')
            if isinstance(feedback.get('repair_rebuild_contracts'), list)
            else []
        )
        promoted_contracts = []
        seen_contract_ids: set[str] = set()

        for item in [*feedback_contracts, *repair_loop_contracts]:
            if not isinstance(item, dict):
                continue
            contract = dict(item)
            contract_id = str(contract.get('contract_id') or repr(sorted(contract.items()))).strip()
            if contract_id in seen_contract_ids:
                continue
            seen_contract_ids.add(contract_id)
            for key, value in classify_repair_execution_policy(contract).items():
                contract.setdefault(key, value)
            promoted_contracts.append(contract)
        repair_loop_payload = dict(repair_loop or {}) if isinstance(repair_loop, dict) else None
        if repair_loop_payload is not None:
            repair_work_available_count = sum(
                1 for item in promoted_contracts if item.get('repair_work_available') is True
            )
            needs_external_input_count = sum(
                1 for item in promoted_contracts if item.get('needs_external_input') is True
            )
            blocked_contract_count = sum(
                1 for item in promoted_contracts if item.get('auto_execute') is not True
            )
            repair_loop_payload.setdefault('repair_work_available', bool(repair_work_available_count))
            repair_loop_payload.setdefault('repair_work_available_count', repair_work_available_count)
            repair_loop_payload.setdefault('needs_external_input_count', needs_external_input_count)
            repair_loop_payload.setdefault('materialization_blocked_contract_count', blocked_contract_count)
            repair_loop_payload['promoted_contracts'] = promoted_contracts
        contract_by_identity: dict[str, dict[str, Any]] = {}
        for contract in promoted_contracts:
            for key in ('branch_id', 'phase_id', 'obligation_id', 'task_id', 'contract_id'):
                token = str(contract.get(key) or '').strip()
                if token:
                    contract_by_identity.setdefault(token, contract)

        def _slug(value: Any, fallback: str) -> str:
            token = str(value or '').strip().lower()
            token = re.sub(r'[^a-z0-9_]+', '-', token.replace(' ', '-'))
            token = re.sub(r'-+', '-', token).strip('-')
            return token or fallback

        def _repair_contract_for_item(raw_item: dict[str, Any]) -> dict[str, Any]:
            direct_contract = raw_item.get('repair_contract')
            if isinstance(direct_contract, dict):
                return dict(direct_contract)
            for key in ('branch_id', 'phase_id', 'obligation_id', 'task_id', 'repair_contract_id'):
                token = str(raw_item.get(key) or '').strip()
                if token and token in contract_by_identity:
                    return dict(contract_by_identity[token])
            return {}

        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            repair_contract = _repair_contract_for_item(raw_item)
            capability = normalize_capability(raw_item.get('capability'))
            if not capability:
                capability = normalize_capability(repair_contract.get('capability'))
            output_type = str(raw_item.get('output_type') or '').strip().lower()
            if not output_type:
                output_type = str(repair_contract.get('output_type') or '').strip().lower()
            if not output_type:
                output_type = self._repair_output_type_for_capability(capability) or ''
            repair_action = str(
                raw_item.get('repair_action')
                or raw_item.get('recovery_action')
                or repair_contract.get('repair_action')
                or ''
            ).strip()
            allow_chat_repair = (
                capability == self.capability_chat
                and output_type == 'text'
                and (
                    repair_action
                    or raw_item.get('depends_on')
                    or raw_item.get('execution_contract')
                    or raw_item.get('workload_task_ref')
                )
            )
            if not capability or (capability == self.capability_chat and not allow_chat_repair):
                continue
            if not output_type:
                continue
            if repair_action and repair_action not in repair_actions:
                repair_actions.append(repair_action)
            raw_branch_identity = str(raw_item.get('branch_id') or raw_item.get('phase_id') or '').strip()
            if raw_branch_identity and raw_branch_identity in seen_branch_ids:
                continue
            try:
                missing_count = int(raw_item.get('missing_count') or 1)
            except (TypeError, ValueError):
                missing_count = 1
            missing_count = max(1, missing_count)
            if capability not in pending_capabilities:
                pending_capabilities.append(capability)
            base_identity = (
                repair_contract.get('branch_id')
                or repair_contract.get('phase_id')
                or repair_contract.get('obligation_id')
                or raw_item.get('branch_id')
                or raw_item.get('phase_id')
                or raw_item.get('obligation_id')
                or f'repair-{capability}'
            )
            for occurrence in range(1, missing_count + 1):
                branch_seed = _slug(base_identity, f'repair-{capability}')
                branch_id = branch_seed if missing_count == 1 else f'{branch_seed}-{occurrence}'
                if not branch_id.startswith('repair-') and not branch_id.startswith('branch-'):
                    branch_id = f'repair-{branch_id}'
                while branch_id in seen_branch_ids:
                    branch_id = f'{branch_id}-{len(seen_branch_ids) + 1}'
                seen_branch_ids.add(branch_id)
                depends_on = [
                    str(item or '').strip()
                    for item in (raw_item.get('depends_on') or [])
                    if str(item or '').strip()
                ]
                branch_payload = {
                    'branch_id': branch_id,
                    'phase_id': branch_id,
                    'obligation_id': str(raw_item.get('obligation_id') or repair_contract.get('obligation_id') or '').strip() or None,
                    'capability': capability,
                    'output_type': output_type,
                    'role': str(raw_item.get('role') or '').strip() or None,
                    'depends_on': depends_on or None,
                    'repair_scope': str(raw_item.get('repair_scope') or repair_contract.get('repair_scope') or '').strip() or None,
                    'resource_class': str(raw_item.get('resource_class') or repair_contract.get('resource_class') or '').strip() or None,
                    'dependency_policy': str(raw_item.get('dependency_policy') or repair_contract.get('dependency_policy') or '').strip() or None,
                    'runtime_scheduling_context': raw_item.get('runtime_scheduling_context') or repair_contract.get('runtime_scheduling_context'),
                    'allow_gpu_heavy_concurrency': raw_item.get('allow_gpu_heavy_concurrency') or repair_contract.get('allow_gpu_heavy_concurrency'),
                    'status': 'pending',
                    'queue_index': len(existing_branches) + len(pending_branches) + 1,
                    'repair_source': 'ghost_repair_feedback',
                    'repair_evidence': str(raw_item.get('evidence') or '').strip() or None,
                    'repair_action': repair_action or None,
                    'recovery_action': repair_action or None,
                    'repair_action_reason': str(raw_item.get('repair_action_reason') or '').strip() or None,
                    'total_missing_count': (
                        raw_item.get('total_missing_count')
                        or repair_contract.get('total_missing_count')
                    ),
                    'repair_occurrence_index': (
                        raw_item.get('repair_occurrence_index')
                        or repair_contract.get('repair_occurrence_index')
                    ),
                    'repair_occurrence_count': (
                        raw_item.get('repair_occurrence_count')
                        or repair_contract.get('repair_occurrence_count')
                    ),
                    'repair_contract': repair_contract or None,
                    'repair_contract_id': str(repair_contract.get('contract_id') or '').strip() or None,
                    'repair_contract_status': str(repair_contract.get('status') or '').strip() or None,
                    'repair_execution_policy': str(repair_contract.get('execution_policy') or '').strip() or None,
                    'repair_promotion_source': str(repair_contract.get('promotion_source') or '').strip() or None,
                    'contract_state': str(repair_contract.get('status') or raw_item.get('contract_state') or '').strip() or None,
                    'promotion_source': str(repair_contract.get('promotion_source') or '').strip() or None,
                }
                execution_policy = str(repair_contract.get('execution_policy') or '').strip()
                if execution_policy == 'blocked_until_dependency_evidence':
                    branch_payload['blocked_by_dependency_input'] = True
                if execution_policy == 'blocked_until_branch_contract':
                    branch_payload['blocked_by_branch_contract'] = True
                if execution_policy == 'blocked_until_promoted_obligation_branch':
                    branch_payload['blocked_by_underplanned_promoted_obligations'] = True
                for key in (
                    'execution_contract',
                    'workload_task_ref',
                    'output_obligation_ref',
                    'output_contract',
                    'input_refs',
                    'review_criteria',
                    'artifact_prompt',
                    'artifact_prompt_source',
                    'batch_prompts',
                    'content_payload',
                    'content_payload_source',
                    'stage_direction',
                    'phase_summary',
                    'repair_scope',
                    'resource_class',
                    'dependency_policy',
                    'runtime_scheduling_context',
                    'allow_gpu_heavy_concurrency',
                    'artifact_request',
                    'requires_artifact',
                    'text_artifact_extension',
                    'text_artifact_source_name',
                    'text_artifact_source',
                    'text_artifact_target_path',
                    'semantic_intent',
                    'objective',
                    'deliverable',
                    'rationale',
                    'advisory_role',
                    'decision_notes',
                    'evidence_requirements',
                    'reconsideration_triggers',
                    'semantic_review_criteria',
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
                    'surface_state',
                    'supersession_review_required',
                    'supersession_review_authority',
                    'failed_instance_id',
                    'exclude_instance_ids',
                    'blocked_by_dependency_input',
                    'blocked_by_branch_contract',
                    'materialization_blocked',
                    'blocked_scope',
                    'blocked_prerequisite',
                    'repair_work_available',
                    'repair_work_policy',
                    'needs_external_input',
                    'contract_state',
                    'promotion_source',
                ):
                    value = raw_item.get(key)
                    if value not in (None, '', [], {}):
                        branch_payload[key] = value
                for key in (
                    'execution_contract',
                    'workload_task_ref',
                    'output_obligation_ref',
                    'output_contract',
                    'input_refs',
                    'review_criteria',
                    'semantic_review_criteria',
                    'evidence_requirements',
                    'artifact_prompt',
                    'artifact_prompt_source',
                    'batch_prompts',
                    'artifact_request',
                    'requires_artifact',
                    'text_artifact_extension',
                    'text_artifact_source_name',
                    'text_artifact_source',
                    'text_artifact_target_path',
                    'materialization_blocked',
                    'blocked_scope',
                    'blocked_prerequisite',
                    'repair_work_available',
                    'repair_work_policy',
                    'needs_external_input',
                ):
                    value = repair_contract.get(key)
                    if value not in (None, '', [], {}) and branch_payload.get(key) in (None, '', [], {}):
                        branch_payload[key] = value
                for key, value in classify_repair_execution_policy(branch_payload).items():
                    if value in (None, '', [], {}):
                        continue
                    target_key = 'repair_execution_policy' if key == 'execution_policy' else key
                    if branch_payload.get(target_key) in (None, '', [], {}):
                        branch_payload[target_key] = value
                pending_branches.append(_compact_payload(branch_payload))

        if not pending_branches:
            return None

        first_branch = pending_branches[0]
        merged_gap = dict(prior_gap or {})
        merged_pending_branches = [*existing_branches, *pending_branches]
        merged_gap.update(
            {
                'code': str(merged_gap.get('code') or '').strip() or 'closure_review_repair',
                'trigger': str(merged_gap.get('trigger') or '').strip() or 'ghost_repair_feedback',
                'expected_capability': (
                    normalize_capability(merged_gap.get('expected_capability'))
                    or first_branch.get('capability')
                ),
                'active_capability': (
                    normalize_capability(merged_gap.get('active_capability'))
                    or normalize_capability(merged_gap.get('expected_capability'))
                    or first_branch.get('capability')
                ),
                'missing_artifact_type': (
                    str(merged_gap.get('missing_artifact_type') or '').strip()
                    or str(first_branch.get('output_type') or '').strip()
                    or None
                ),
                'pending_branches': merged_pending_branches,
                'pending_capabilities': list(
                    dict.fromkeys(
                        [
                            *(
                                str(item).strip()
                                for item in (merged_gap.get('pending_capabilities') or [])
                                if str(item).strip()
                            ),
                            *pending_capabilities,
                        ]
                    )
                ),
                'ghost_repair_feedback': feedback,
                'repair_scope': str(feedback.get('patch_scope') or '').strip() or None,
                'repair_mode': str(feedback.get('repair_mode') or '').strip() or None,
                'repair_loop': repair_loop_payload,
                'repair_rebuild_contracts': promoted_contracts or None,
                'reconsideration_rebuild': {
                    'kind': 'ollmo.contract_driven_reconsideration_rebuild',
                    'status': 'promoted' if promoted_contracts else 'candidate',
                    'authority': 'closure_review_runtime_truth',
                    'promoted_contract_count': len(promoted_contracts),
                    'pending_branch_count': len(merged_pending_branches),
                    'auto_execute': bool((repair_loop_payload or {}).get('auto_execute')) if isinstance(repair_loop_payload, dict) else False,
                    'repair_work_available': bool((repair_loop_payload or {}).get('repair_work_available')) if isinstance(repair_loop_payload, dict) else False,
                    'repair_work_available_count': (
                        repair_loop_payload.get('repair_work_available_count')
                        if isinstance(repair_loop_payload, dict)
                        else None
                    ),
                    'needs_external_input_count': (
                        repair_loop_payload.get('needs_external_input_count')
                        if isinstance(repair_loop_payload, dict)
                        else None
                    ),
                },
                'repair_action': repair_actions[0] if repair_actions else None,
                'repair_actions': repair_actions or None,
                'preserve_request_id': bool(feedback.get('preserve_request_id')),
            }
        )
        if not str(merged_gap.get('code') or '').strip():
            merged_gap['code'] = 'closure_review_repair'
        if (
            isinstance(prior_gap, dict)
            and prior_gap
            and str(merged_gap.get('code') or '').strip() != 'closure_review_repair'
        ):
            merged_gap['repair_code'] = 'closure_review_repair'
            merged_gap['repair_trigger'] = 'ghost_repair_feedback'
        return _compact_payload(merged_gap)

    @staticmethod
    def _dedupe_graph_repair_records(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            identity = ''
            for key in keys:
                identity = str(item.get(key) or '').strip()
                if identity:
                    break
            if not identity:
                identity = repr(sorted(item.items()))
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        return unique

    @staticmethod
    def _terminal_graph_patch_parent_frame(response_payload: dict[str, Any]) -> dict[str, Any]:
        response_id = str(response_payload.get('response_id') or response_payload.get('id') or '').strip()
        frame = (
            dict(response_payload.get('response_frame'))
            if isinstance(response_payload.get('response_frame'), dict)
            else {}
        )
        relation = dict(frame.get('frame_relation')) if isinstance(frame.get('frame_relation'), dict) else {}
        return _compact_payload(
            {
                'parent_response_id': response_id or None,
                'parent_frame_id': str(frame.get('frame_id') or '').strip() or None,
                'parent_frame_sequence': frame.get('frame_sequence'),
                'parent_frame_relation': relation or None,
            }
        )

    def _build_graph_patch_successor_reopen_request(
        self,
        response_payload: dict[str, Any],
        lifecycle: dict[str, Any],
        successor_application: dict[str, Any],
        *,
        autonomy_level: str,
    ) -> dict[str, Any]:
        successor_graph = (
            dict(successor_application.get('graph') or {})
            if isinstance(successor_application.get('graph'), dict)
            else {}
        )
        if not successor_graph:
            return {}
        parent = self._terminal_graph_patch_parent_frame(response_payload)
        owed_branch_ids, output_obligations = _graph_patch_successor_owed_scope(
            successor_graph,
            successor_application,
            fallback_branch_ids=lifecycle.get('scheduled_branch_ids') or [],
        )
        if not owed_branch_ids:
            return {}
        successor_reopen_depth = _graph_patch_successor_parent_depth(response_payload) + 1
        successor_reopen_key, successor_execution_key = _graph_patch_successor_keys(
            response_id=str(parent.get('parent_response_id') or '').strip(),
            parent_frame_id=str(parent.get('parent_frame_id') or '').strip(),
            patch_id=str(lifecycle.get('patch_id') or '').strip(),
            idempotency_key=str(lifecycle.get('idempotency_key') or '').strip(),
            owed_branch_ids=owed_branch_ids,
            successor_depth=successor_reopen_depth,
        )
        request = {
            'kind': 'ollmo.graph_patch_successor_reopen_request',
            'status': 'candidate',
            **parent,
            'successor_reopen_key': successor_reopen_key,
            'successor_execution_key': successor_execution_key,
            'proposal_id': lifecycle.get('proposal_id'),
            'patch_id': lifecycle.get('patch_id'),
            'idempotency_key': lifecycle.get('idempotency_key'),
            'repair_class': lifecycle.get('repair_class'),
            'risk_level': lifecycle.get('risk_level'),
            'autonomy_level': autonomy_level,
            'runtime_effect': 'successor_reopen_required',
            'source_evidence_refs': lifecycle.get('source_evidence_refs') or [],
            'before_graph_digest': lifecycle.get('before_graph_digest'),
            'patch_digest': lifecycle.get('patch_digest'),
            'successor_graph_digest': (
                stable_graph_repair_graph_digest(successor_graph)
            ),
            'successor_reopen_depth': successor_reopen_depth,
            'enforced_policy_id': lifecycle.get('enforced_policy_id'),
            'enforced_class': lifecycle.get('enforced_class'),
            'policy_mode': lifecycle.get('policy_mode'),
            'allowed_by_policy': lifecycle.get('allowed_by_policy'),
            'current_evidence_refs': lifecycle.get('current_evidence_refs'),
            'forbidden_evidence_seen': lifecycle.get('forbidden_evidence_seen'),
            'blocked_reasons': [],
            'frame_relation': _compact_payload(
                {
                    'kind': 'graph_patch_reopen_successor',
                    'reason': 'graph_patch_reopen',
                    'parent_response_id': parent.get('parent_response_id'),
                    'parent_frame_id': parent.get('parent_frame_id'),
                    'parent_frame_sequence': parent.get('parent_frame_sequence'),
                    'successor_reopen_depth': successor_reopen_depth,
                }
            ),
            'owed_branch_ids': owed_branch_ids,
            'owed_output_obligations': output_obligations,
            'successor_request_phase_graph': successor_graph,
            'patch_application': _compact_payload(
                {
                    'kind': successor_application.get('kind'),
                    'status': successor_application.get('status'),
                    'proposal_id': successor_application.get('proposal_id'),
                    'patch_id': successor_application.get('patch_id'),
                    'idempotency_key': successor_application.get('idempotency_key'),
                    'original_graph_digest': successor_application.get('original_graph_digest'),
                    'patched_graph_digest': successor_application.get('patched_graph_digest'),
                    'applied_branch_ids': successor_application.get('applied_branch_ids'),
                    'applied_phase_ids': successor_application.get('applied_phase_ids'),
                    'applied_obligation_ids': successor_application.get('applied_obligation_ids'),
                    'applied_dependency_edges': successor_application.get('applied_dependency_edges'),
                }
            ),
            'execution_contract': {
                'execution_scope': 'successor_owed_branches_only',
                'root_scoped': False,
                'allow_root_prompt': False,
                'preserve_request_id': True,
            },
        }
        return _compact_payload(request)

    def _attach_runtime_graph_repair_evidence(
        self,
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach backend-owned runtime graph-repair proposal/review diagnostics."""

        if not isinstance(response_payload, dict):
            return response_payload
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            return response_payload
        closure_review = (
            dict(runtime.get('graph_closure_review') or {})
            if isinstance(runtime.get('graph_closure_review'), dict)
            else {}
        )
        late_fill = (
            dict(updated.get('late_fill') or {})
            if isinstance(updated.get('late_fill'), dict)
            else dict(runtime.get('late_fill') or {})
            if isinstance(runtime.get('late_fill'), dict)
            else {}
        )
        accepted_learning_hints = (
            dict(runtime.get('accepted_learning_hints') or {})
            if isinstance(runtime.get('accepted_learning_hints'), dict)
            else {}
        )
        surface_state = (
            closure_review.get('surface_state')
            if isinstance(closure_review.get('surface_state'), dict)
            else late_fill.get('surface_state')
            if isinstance(late_fill.get('surface_state'), dict)
            else {}
        )
        surface_actionability = (
            classify_surface_repair_actionability(
                surface_state,
                closure_review=closure_review,
                late_fill=late_fill,
                monitor_report={},
            )
            if isinstance(surface_state, dict) and surface_state
            else {}
        )
        proposals = build_graph_repair_proposals_from_runtime_evidence(
            response_frame=updated,
            request_phase_graph=graph,
            closure_review=closure_review,
            late_fill=late_fill,
            monitor_report={
                'response_id': updated.get('response_id') or updated.get('id'),
                'lifecycle_state': updated.get('lifecycle_state'),
                'status': updated.get('status'),
                'final_materialization_contract_status': late_fill.get('final_materialization_contract_status'),
                'materialization_contract_unmet': late_fill.get('materialization_contract_unmet'),
                'branch_counts': {
                    'pending': len(late_fill.get('pending_branches') or []),
                    'active': len(late_fill.get('active_branches') or []),
                    'failed': len(late_fill.get('failed_branches') or []),
                    'completed': len(late_fill.get('completed_branches') or []),
                },
            },
            accepted_learning_hints=accepted_learning_hints,
        )
        reviews = [
            validate_graph_repair_proposal(
                proposal,
                request_phase_graph=graph,
                closure_review=closure_review,
                promotion_review={},
                accepted_learning_hints=accepted_learning_hints,
            )
            for proposal in proposals
            if isinstance(proposal, dict)
        ]

        existing_proposals = [
            dict(item)
            for item in (graph.get('graph_repair_proposals') or [])
            if isinstance(item, dict)
        ]
        existing_reviews = [
            dict(item)
            for item in (graph.get('graph_repair_reviews') or [])
            if isinstance(item, dict)
        ]
        graph['graph_repair_proposals'] = self._dedupe_graph_repair_records(
            [*existing_proposals, *proposals],
            'proposal_id',
            'id',
        )
        graph['graph_repair_reviews'] = self._dedupe_graph_repair_records(
            [*existing_reviews, *reviews],
            'review_id',
            'proposal_id',
            'id',
        )
        runtime['request_phase_graph'] = graph

        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics['surface_repair_actionability'] = surface_actionability
        developer_diagnostics['runtime_graph_repair_proposals'] = proposals
        developer_diagnostics['runtime_graph_repair_proposal_reviews'] = reviews
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def _attach_redraw_scope_ladder_review(
        self,
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach backend-owned intent-aligned redraw scope review diagnostics."""

        if not isinstance(response_payload, dict):
            return response_payload
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            return response_payload
        closure_review = (
            dict(runtime.get('graph_closure_review') or {})
            if isinstance(runtime.get('graph_closure_review'), dict)
            else {}
        )
        late_fill = (
            dict(updated.get('late_fill') or {})
            if isinstance(updated.get('late_fill'), dict)
            else dict(runtime.get('late_fill') or {})
            if isinstance(runtime.get('late_fill'), dict)
            else {}
        )
        surface_state = (
            closure_review.get('surface_state')
            if isinstance(closure_review.get('surface_state'), dict)
            else late_fill.get('surface_state')
            if isinstance(late_fill.get('surface_state'), dict)
            else {}
        )
        accepted_learning_hints = (
            runtime.get('accepted_learning_hints')
            if isinstance(runtime.get('accepted_learning_hints'), list)
            else runtime.get('accepted_learning_hints', {}).get('hints')
            if isinstance(runtime.get('accepted_learning_hints'), dict)
            else []
        )
        review = build_redraw_scope_ladder_review(
            response_payload=updated,
            request_phase_graph=graph,
            closure_review=closure_review,
            surface_state=surface_state,
            accepted_learning_hints=accepted_learning_hints,
        )
        graph['redraw_scope_ladder_review'] = review
        runtime['request_phase_graph'] = graph

        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics['redraw_scope_ladder_review'] = review
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def _attach_graph_patch_lifecycle(
        self,
        response_payload: dict[str, Any],
        *,
        graph_repair_autonomy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Attach staged/applied graph patch lifecycle truth according to autonomy."""

        if not isinstance(response_payload, dict):
            return response_payload
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            return response_payload

        autonomy_description = (
            describe_graph_repair_autonomy(graph_repair_autonomy)
            if graph_repair_autonomy is not None
            else describe_graph_repair_autonomy_from_env()
        )
        autonomy_level = normalize_graph_repair_autonomy(autonomy_description.get('autonomy_level'))
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        autonomy_source = (
            'explicit_override'
            if graph_repair_autonomy is not None
            else str(autonomy_description.get('source') or 'environment').strip()
        )
        explicit_lifecycle_state = str(
            updated.get('lifecycle_state')
            or runtime.get('lifecycle_state')
            or ''
        ).strip().lower().replace('-', '_')
        response_frame = (
            updated.get('response_frame')
            if isinstance(updated.get('response_frame'), dict)
            else {}
        )
        identified_frame = bool(
            str(response_frame.get('frame_id') or '').strip()
            or response_frame.get('frame_sequence') not in (None, '')
        )
        compatibility_status = str(
            updated.get('status')
            or runtime.get('status')
            or ''
        ).strip().lower().replace('-', '_')
        response_state = (
            explicit_lifecycle_state
            or (compatibility_status if identified_frame else 'pre_freeze')
        )
        terminal_apply_blocked = (
            autonomy_level.startswith('apply_')
            and (
                identified_frame
                or response_state in _FROZEN_GRAPH_PATCH_RESPONSE_STATES
            )
        )
        developer_diagnostics['graph_patch_autonomy'] = {
            **autonomy_description,
            'autonomy_level': autonomy_level,
            'normalized': autonomy_level,
            'source': autonomy_source,
            'response_state': response_state,
            'response_state_source': (
                'lifecycle_state'
                if explicit_lifecycle_state
                else 'identified_response_frame'
                if identified_frame
                else 'pre_freeze_runtime'
            ),
            'terminal_apply_blocked': terminal_apply_blocked,
        }
        developer_diagnostics['enforced_policy'] = describe_enforced_policy_from_env()
        if autonomy_level == 'off':
            developer_diagnostics['graph_patch_lifecycle'] = graph.get('graph_patch_lifecycle') or []
            developer_diagnostics['graph_patch_enforced_policy_reviews'] = [
                item.get('enforced_policy_review')
                for item in graph.get('graph_patch_lifecycle') or []
                if isinstance(item, dict) and isinstance(item.get('enforced_policy_review'), dict)
            ]
            developer_diagnostics['staged_graph_patches'] = graph.get('staged_graph_patches') or []
            developer_diagnostics['applied_graph_patches'] = graph.get('applied_graph_patches') or []
            runtime['developer_diagnostics'] = developer_diagnostics
            updated['runtime'] = runtime
            return updated

        reviews = [
            dict(item)
            for item in (graph.get('graph_repair_reviews') or [])
            if isinstance(item, dict)
        ]
        working_graph = graph
        lifecycle_results: list[dict[str, Any]] = []
        successor_reopen_requests: list[dict[str, Any]] = []
        for review in reviews:
            lifecycle = build_graph_patch_lifecycle(
                request_phase_graph=working_graph,
                proposal_review=review,
                autonomy_level=autonomy_level,
            )
            successor_reopen_request = None
            if terminal_apply_blocked:
                if (
                    autonomy_level in {'apply_safe', 'apply_enforced'}
                    and lifecycle.get('risk_level') == 'safe_additive'
                    and lifecycle.get('status') not in {'blocked', 'rejected'}
                ):
                    successor_application = apply_validated_graph_patch(
                        working_graph,
                        lifecycle,
                        autonomy_level=autonomy_level,
                    )
                    if (
                        successor_application.get('status') == 'applied'
                        and isinstance(successor_application.get('graph'), dict)
                    ):
                        successor_reopen_request = self._build_graph_patch_successor_reopen_request(
                            updated,
                            lifecycle,
                            successor_application,
                            autonomy_level=autonomy_level,
                        )
                blocked_reasons = [
                    str(item)
                    for item in (lifecycle.get('blocked_reasons') or [])
                    if str(item or '').strip()
                ]
                if 'terminal_frame_requires_successor_reopen' not in blocked_reasons:
                    blocked_reasons.append('terminal_frame_requires_successor_reopen')
                lifecycle = {
                    **lifecycle,
                    'status': 'blocked',
                    'blocked_reasons': blocked_reasons,
                    'outcome': {
                        'status': 'blocked',
                        'runtime_effect': 'terminal_frame_not_mutated',
                    },
                }
            application = apply_validated_graph_patch(
                working_graph,
                lifecycle,
                autonomy_level=autonomy_level,
            )
            working_graph = (
                dict(application.get('graph') or {})
                if isinstance(application.get('graph'), dict)
                else working_graph
            )
            if successor_reopen_request:
                successor_reopen_requests.append(successor_reopen_request)
                working_graph['successor_reopen_requests'] = self._dedupe_graph_repair_records(
                    [
                        *[
                            dict(item)
                            for item in (working_graph.get('successor_reopen_requests') or [])
                            if isinstance(item, dict)
                        ],
                        successor_reopen_request,
                    ],
                    'successor_reopen_key',
                    'patch_id',
                    'proposal_id',
                    'idempotency_key',
                )
                application['graph'] = working_graph
            lifecycle_results.append(application)

        runtime['request_phase_graph'] = working_graph
        developer_diagnostics['graph_patch_lifecycle'] = working_graph.get('graph_patch_lifecycle') or []
        developer_diagnostics['graph_patch_lifecycle_results'] = lifecycle_results
        developer_diagnostics['graph_patch_enforced_policy_reviews'] = [
            item.get('enforced_policy_review')
            for item in working_graph.get('graph_patch_lifecycle') or []
            if isinstance(item, dict) and isinstance(item.get('enforced_policy_review'), dict)
        ]
        developer_diagnostics['graph_patch_successor_reopen_requests'] = (
            working_graph.get('successor_reopen_requests') or successor_reopen_requests
        )
        developer_diagnostics['staged_graph_patches'] = working_graph.get('staged_graph_patches') or []
        developer_diagnostics['applied_graph_patches'] = working_graph.get('applied_graph_patches') or []
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def _reconcile_applied_graph_patch_late_fill_gap(
        self,
        response_payload: dict[str, Any],
        closure_gap: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        """Expose branches added by this lifecycle pass as same-turn Late Fill work."""

        if not isinstance(response_payload, dict):
            return response_payload, closure_gap
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        lifecycle_results = [
            dict(item)
            for item in (developer_diagnostics.get('graph_patch_lifecycle_results') or [])
            if isinstance(item, dict)
            and str(item.get('status') or '').strip().lower() == 'applied'
        ]
        if not graph or not lifecycle_results:
            return updated, closure_gap

        applied_branch_ids: set[str] = set()
        applied_phase_ids: set[str] = set()
        applied_obligation_ids: set[str] = set()
        dependency_target_ids: set[str] = set()
        patch_ids: list[str] = []

        def _remember_tokens(target: set[str], values: Any) -> None:
            for value in values if isinstance(values, list) else []:
                token = str(value or '').strip()
                if token:
                    target.add(token)

        for result in lifecycle_results:
            patch_id = str(result.get('patch_id') or '').strip()
            if patch_id and patch_id not in patch_ids:
                patch_ids.append(patch_id)
            _remember_tokens(applied_branch_ids, result.get('applied_branch_ids'))
            _remember_tokens(applied_phase_ids, result.get('applied_phase_ids'))
            _remember_tokens(applied_obligation_ids, result.get('applied_obligation_ids'))
            for edge in result.get('applied_dependency_edges') or []:
                if not isinstance(edge, dict):
                    continue
                target_id = str(edge.get('target_id') or '').strip()
                if target_id:
                    dependency_target_ids.add(target_id)

        target_ids = {
            *applied_branch_ids,
            *applied_phase_ids,
            *dependency_target_ids,
        }
        obligations = [
            dict(item)
            for item in (graph.get('output_obligations') or [])
            if isinstance(item, dict)
        ]
        matched_obligations: list[dict[str, Any]] = []
        for obligation in obligations:
            obligation_id = str(obligation.get('obligation_id') or '').strip()
            record_ids = {
                obligation_id,
                str(obligation.get('branch_id') or '').strip(),
                str(obligation.get('phase_id') or '').strip(),
            }
            record_ids.discard('')
            if obligation_id in applied_obligation_ids or target_ids.intersection(record_ids):
                matched_obligations.append(obligation)
                target_ids.update(record_ids)

        phases = [
            dict(item)
            for item in (graph.get('phases') or [])
            if isinstance(item, dict)
        ]
        matched_phases: list[dict[str, Any]] = []
        for phase in phases:
            record_ids = {
                str(phase.get('branch_id') or '').strip(),
                str(phase.get('phase_id') or '').strip(),
                str(phase.get('obligation_id') or '').strip(),
            }
            record_ids.discard('')
            if target_ids.intersection(record_ids):
                matched_phases.append(phase)
                target_ids.update(record_ids)

        terminal_branch_states = {
            'cancelled',
            'candidate',
            'completed',
            'deferred_not_executable',
            'future',
            'future_candidate',
            'fulfilled',
            'held',
            'non_executable',
            'not_executable',
            'optional',
            'possible',
            'reserved',
            'reserved_candidate',
            'skipped',
            'superseded',
            'waived',
        }
        pending_branches: list[dict[str, Any]] = []
        unscheduled_branches: list[dict[str, Any]] = []
        seen_branch_ids: set[str] = set()
        seen_unscheduled_branch_ids: set[str] = set()

        def _record_unscheduled_branch(branch: dict[str, Any], reason: str) -> None:
            branch_id = str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
            if not branch_id or branch_id in seen_unscheduled_branch_ids:
                return
            seen_unscheduled_branch_ids.add(branch_id)
            unscheduled_branches.append(
                _compact_payload(
                    {
                        'branch_id': branch_id,
                        'phase_id': str(branch.get('phase_id') or branch_id).strip() or branch_id,
                        'obligation_id': str(branch.get('obligation_id') or '').strip() or None,
                        'capability': str(branch.get('capability') or '').strip().lower() or None,
                        'output_type': str(branch.get('output_type') or '').strip().lower() or None,
                        'status': str(branch.get('status') or '').strip().lower() or None,
                        'repair_action': str(branch.get('repair_action') or '').strip() or None,
                        'repair_execution_policy': str(
                            branch.get('repair_execution_policy')
                            or branch.get('execution_policy')
                            or ''
                        ).strip() or None,
                        'reason': reason,
                    }
                )
            )

        def _append_pending_branch(raw_record: dict[str, Any]) -> None:
            branch = dict(raw_record)
            branch_id = str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
            phase_id = str(branch.get('phase_id') or branch_id).strip() or branch_id
            capability = str(branch.get('capability') or '').strip().lower()
            if (
                not branch_id
                or not capability
                or branch_id in seen_branch_ids
                or branch_id in seen_unscheduled_branch_ids
            ):
                return
            if str(branch.get('status') or '').strip().lower() in terminal_branch_states:
                return
            if bool(branch.get('reserved_for_later') or branch.get('deferred_not_executable')):
                return
            if str(branch.get('role') or '').strip().lower() in {
                'candidate',
                'deferred_candidate',
                'future_candidate',
                'non_executable',
                'optional_output',
                'possible_future',
                'reserved_candidate',
            }:
                return
            matching_obligation = next(
                (
                    item
                    for item in matched_obligations
                    if branch_id
                    in {
                        str(item.get('branch_id') or '').strip(),
                        str(item.get('phase_id') or '').strip(),
                    }
                    or phase_id
                    in {
                        str(item.get('branch_id') or '').strip(),
                        str(item.get('phase_id') or '').strip(),
                    }
                ),
                None,
            )
            if matching_obligation:
                for key in (
                    'obligation_id',
                    'output_type',
                    'required',
                    'depends_on',
                    'input_refs',
                    'review_criteria',
                    'repair_action',
                    'repair_contract_id',
                    'repair_execution_policy',
                ):
                    if branch.get(key) in (None, '', [], {}) and matching_obligation.get(key) not in (None, '', [], {}):
                        branch[key] = matching_obligation.get(key)
            repair_policy_input = dict(branch)
            repair_execution_policy = str(
                repair_policy_input.get('repair_execution_policy')
                or repair_policy_input.get('execution_policy')
                or ''
            ).strip().lower()
            if repair_execution_policy and not repair_policy_input.get('execution_policy'):
                repair_policy_input['execution_policy'] = repair_execution_policy
            if repair_execution_policy in {
                'manual_review_required',
                'non_executable_until_promoted',
                'semantic_review_required',
            }:
                _record_unscheduled_branch(branch, 'repair_execution_policy_non_executable')
                return
            repair_policy = classify_repair_execution_policy(repair_policy_input)
            policy_schedules = (
                str(repair_policy.get('execution_policy') or '').strip().lower()
                == 'schedule_late_fill_branch'
                and repair_policy.get('auto_execute') is True
                and repair_policy.get('materialization_blocked') is not True
            )
            branch_status = str(branch.get('status') or '').strip().lower()
            resolved_blocked_policy = (
                repair_execution_policy.startswith('blocked_until_')
                and policy_schedules
            )
            if not policy_schedules:
                for key, value in repair_policy.items():
                    target_key = 'repair_execution_policy' if key == 'execution_policy' else key
                    if value not in (None, '', [], {}):
                        branch[target_key] = value
                _record_unscheduled_branch(branch, 'repair_execution_policy_blocked')
                return
            if branch_status in {'blocked', 'repair_needed', 'repair_pending'} and not resolved_blocked_policy:
                _record_unscheduled_branch(branch, 'branch_status_not_executable')
                return
            if (
                branch.get('auto_execute') is False
                or branch.get('materialization_blocked') is True
            ) and not resolved_blocked_policy:
                _record_unscheduled_branch(branch, 'branch_execution_contract_not_executable')
                return
            for key, value in repair_policy.items():
                target_key = 'repair_execution_policy' if key == 'execution_policy' else key
                if value not in (None, '', [], {}):
                    branch[target_key] = value
            branch['branch_id'] = branch_id
            branch['phase_id'] = phase_id
            branch['capability'] = capability
            branch['status'] = 'pending'
            branch.setdefault('source', 'runtime_applied_graph_patch')
            branch.setdefault('queue_index', len(pending_branches) + 1)
            seen_branch_ids.add(branch_id)
            pending_branches.append(_compact_payload(branch))

        for branch in graph.get('downstream_branches') or []:
            if not isinstance(branch, dict):
                continue
            record_ids = {
                str(branch.get('branch_id') or '').strip(),
                str(branch.get('phase_id') or '').strip(),
                str(branch.get('obligation_id') or '').strip(),
            }
            record_ids.discard('')
            if target_ids.intersection(record_ids):
                _append_pending_branch(branch)
        for phase in matched_phases:
            _append_pending_branch(phase)
        for obligation in matched_obligations:
            _append_pending_branch(obligation)

        reconciliation = {
            'kind': 'ollmo.graph_patch_late_fill_reconciliation',
            'status': 'applied' if pending_branches else 'no_executable_branches',
            'authority': 'runtime_applied_graph_patch',
            'patch_ids': patch_ids,
            'applied_branch_ids': sorted(applied_branch_ids),
            'applied_phase_ids': sorted(applied_phase_ids),
            'applied_obligation_ids': sorted(applied_obligation_ids),
            'dependency_target_ids': sorted(dependency_target_ids),
            'scheduled_branch_ids': [
                str(item.get('branch_id') or item.get('phase_id') or '').strip()
                for item in pending_branches
            ],
            'unscheduled_branch_ids': [
                str(item.get('branch_id') or item.get('phase_id') or '').strip()
                for item in unscheduled_branches
            ],
            'unscheduled_branches': unscheduled_branches,
        }
        developer_diagnostics['graph_patch_late_fill_reconciliation'] = reconciliation
        runtime['developer_diagnostics'] = developer_diagnostics

        if unscheduled_branches:
            unscheduled_ids = {
                str(item.get('branch_id') or item.get('phase_id') or '').strip()
                for item in unscheduled_branches
                if str(item.get('branch_id') or item.get('phase_id') or '').strip()
            }

            def _remove_unscheduled_late_fill_branches(raw_state: Any) -> dict[str, Any]:
                state = dict(raw_state) if isinstance(raw_state, dict) else {}
                raw_pending = state.get('pending_branches')
                if not isinstance(raw_pending, list):
                    return state
                remaining = [
                    dict(item)
                    for item in raw_pending
                    if isinstance(item, dict)
                    and str(item.get('branch_id') or item.get('phase_id') or '').strip()
                    not in unscheduled_ids
                ]
                if len(remaining) == len([item for item in raw_pending if isinstance(item, dict)]):
                    return state
                existing_blocked = [
                    dict(item)
                    for item in (state.get('blocked_branches') or [])
                    if isinstance(item, dict)
                ]
                blocked_by_id = {
                    str(item.get('branch_id') or item.get('phase_id') or '').strip(): index
                    for index, item in enumerate(existing_blocked)
                    if str(item.get('branch_id') or item.get('phase_id') or '').strip()
                }
                for item in unscheduled_branches:
                    branch_id = str(item.get('branch_id') or item.get('phase_id') or '').strip()
                    blocked_record = {**item, 'status': 'blocked'}
                    if branch_id in blocked_by_id:
                        existing_blocked[blocked_by_id[branch_id]] = blocked_record
                    else:
                        blocked_by_id[branch_id] = len(existing_blocked)
                        existing_blocked.append(blocked_record)
                remaining_capabilities = []
                for item in remaining:
                    capability = str(item.get('capability') or '').strip().lower()
                    if capability and capability not in remaining_capabilities:
                        remaining_capabilities.append(capability)
                state['pending_branches'] = remaining
                state['pending_capabilities'] = remaining_capabilities
                state['blocked_branches'] = existing_blocked
                state['graph_patch_reconciliation'] = reconciliation
                if remaining:
                    state['status'] = 'pending'
                    for key in ('expected_capability', 'active_capability'):
                        if str(state.get(key) or '').strip().lower() not in remaining_capabilities:
                            state[key] = remaining_capabilities[0] if remaining_capabilities else None
                else:
                    state['status'] = 'blocked'
                    state['expected_capability'] = None
                    state['active_capability'] = None
                return state

            top_late_fill = _remove_unscheduled_late_fill_branches(updated.get('late_fill'))
            if top_late_fill:
                updated['late_fill'] = top_late_fill
            runtime_late_fill = _remove_unscheduled_late_fill_branches(runtime.get('late_fill'))
            if runtime_late_fill:
                runtime['late_fill'] = runtime_late_fill
        updated['runtime'] = runtime

        reconciled_closure_gap = (
            dict(closure_gap)
            if isinstance(closure_gap, dict)
            else None
        )
        if reconciled_closure_gap is not None and unscheduled_branches:
            if isinstance(reconciled_closure_gap.get('pending_branches'), list):
                remaining_pending_branches = [
                    dict(item)
                    for item in reconciled_closure_gap.get('pending_branches') or []
                    if isinstance(item, dict)
                    and str(item.get('branch_id') or item.get('phase_id') or '').strip()
                    not in unscheduled_ids
                ]
                if remaining_pending_branches:
                    reconciled_closure_gap['pending_branches'] = remaining_pending_branches
                    remaining_capabilities = []
                    for item in remaining_pending_branches:
                        capability = str(item.get('capability') or '').strip().lower()
                        if capability and capability not in remaining_capabilities:
                            remaining_capabilities.append(capability)
                    reconciled_closure_gap['pending_capabilities'] = remaining_capabilities
                    if remaining_capabilities:
                        for key in ('expected_capability', 'active_capability'):
                            if str(reconciled_closure_gap.get(key) or '').strip().lower() not in remaining_capabilities:
                                reconciled_closure_gap[key] = remaining_capabilities[0]
                else:
                    reconciled_closure_gap = None
            else:
                gap_branch_id = str(
                    reconciled_closure_gap.get('branch_id')
                    or reconciled_closure_gap.get('phase_id')
                    or ''
                ).strip()
                if gap_branch_id and gap_branch_id in unscheduled_ids:
                    reconciled_closure_gap = None

        if not pending_branches:
            if reconciled_closure_gap is not None:
                reconciled_closure_gap['graph_patch_reconciliation'] = reconciliation
                return updated, _compact_payload(reconciled_closure_gap)
            return updated, None

        gap = dict(reconciled_closure_gap or {})
        existing_branches = [
            dict(item)
            for item in (gap.get('pending_branches') or [])
            if isinstance(item, dict)
        ]
        branch_positions = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip(): index
            for index, item in enumerate(existing_branches)
            if str(item.get('branch_id') or item.get('phase_id') or '').strip()
        }
        for branch in pending_branches:
            branch_id = str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
            if branch_id in branch_positions:
                index = branch_positions[branch_id]
                existing_branches[index] = {
                    **branch,
                    **existing_branches[index],
                    'depends_on': branch.get('depends_on') or existing_branches[index].get('depends_on'),
                }
                continue
            branch_positions[branch_id] = len(existing_branches)
            existing_branches.append(branch)

        pending_capabilities = [
            str(item or '').strip().lower()
            for item in (gap.get('pending_capabilities') or [])
            if str(item or '').strip()
        ]
        for branch in existing_branches:
            capability = str(branch.get('capability') or '').strip().lower()
            if capability and capability not in pending_capabilities:
                pending_capabilities.append(capability)

        first_branch = existing_branches[0]
        gap.update(
            {
                'code': str(gap.get('code') or 'graph_patch_late_fill').strip(),
                'trigger': str(gap.get('trigger') or 'runtime_applied_graph_patch').strip(),
                'expected_capability': str(
                    gap.get('expected_capability')
                    or first_branch.get('capability')
                    or ''
                ).strip(),
                'active_capability': str(
                    gap.get('active_capability')
                    or gap.get('expected_capability')
                    or first_branch.get('capability')
                    or ''
                ).strip(),
                'missing_artifact_type': str(
                    gap.get('missing_artifact_type')
                    or first_branch.get('output_type')
                    or ''
                ).strip(),
                'pending_branches': existing_branches,
                'pending_capabilities': pending_capabilities,
                'graph_patch_reconciliation': reconciliation,
            }
        )
        return updated, _compact_payload(gap)

    @staticmethod
    def _graph_patch_successor_lifecycle(
        graph: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        proposal_id = str(candidate.get('proposal_id') or '').strip()
        patch_id = str(candidate.get('patch_id') or '').strip()
        idempotency_key = str(candidate.get('idempotency_key') or '').strip()
        if not proposal_id or not patch_id or not idempotency_key:
            return {}
        for item in graph.get('graph_patch_lifecycle') or []:
            if not isinstance(item, dict):
                continue
            if str(item.get('proposal_id') or '').strip() != proposal_id:
                continue
            if str(item.get('patch_id') or '').strip() != patch_id:
                continue
            if str(item.get('idempotency_key') or '').strip() != idempotency_key:
                continue
            return dict(item)
        return {}

    @staticmethod
    def _rederive_graph_patch_successor_application(
        parent_graph: dict[str, Any],
        lifecycle: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Re-apply exact validated patch truth instead of trusting candidate scope."""

        base_graph = copy.deepcopy(parent_graph)
        # Terminal production may contain several proposal/lifecycle candidates.
        # Those records are audit ordering, not patch input. Re-derive the
        # candidate's semantic delta against the parent graph with only terminal
        # repair bookkeeping removed, then bind the executable scope to that
        # independently produced application result.
        for key in (
            'graph_patch_lifecycle',
            'staged_graph_patches',
            'applied_graph_patches',
            'successor_reopen_requests',
            'successor_reopen_executions',
        ):
            base_graph.pop(key, None)
        validation_review = (
            lifecycle.get('validation_review')
            if isinstance(lifecycle.get('validation_review'), dict)
            else {}
        )
        application = apply_validated_graph_repair_patch(
            base_graph,
            validation_review,
        )
        if str(application.get('status') or '').strip().lower() != 'applied':
            return {
                'status': 'blocked',
                'blocked_reasons': application.get('blocked_reasons') or [
                    'successor_patch_reapplication_not_applied'
                ],
                'application': application,
            }
        return {
            'status': 'applied',
            'base_graph': base_graph,
            'application': application,
        }

    def _prepare_graph_patch_successor_reopen(
        self,
        response_payload: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Project one validated terminal patch candidate into exact Late Fill work."""

        blocked_reasons: list[str] = []

        def block(reason: str) -> None:
            token = str(reason or '').strip()
            if token and token not in blocked_reasons:
                blocked_reasons.append(token)

        if not isinstance(response_payload, dict) or not isinstance(candidate, dict):
            return {'status': 'blocked', 'blocked_reasons': ['successor_payload_invalid']}
        if candidate.get('kind') != 'ollmo.graph_patch_successor_reopen_request':
            block('successor_request_kind_mismatch')
        if str(candidate.get('status') or '').strip().lower() != 'candidate':
            block('successor_request_not_candidate')
        if str(candidate.get('risk_level') or '').strip().lower() != 'safe_additive':
            block('successor_repair_not_safe_additive')
        if str(candidate.get('runtime_effect') or '').strip() != 'successor_reopen_required':
            block('successor_runtime_effect_mismatch')

        response_id = str(
            response_payload.get('response_id')
            or response_payload.get('id')
            or ''
        ).strip()
        parent_frame = (
            response_payload.get('response_frame')
            if isinstance(response_payload.get('response_frame'), dict)
            else {}
        )
        parent_frame_id = str(parent_frame.get('frame_id') or '').strip()
        parent_frame_sequence = parent_frame.get('frame_sequence')
        if not response_id or str(candidate.get('parent_response_id') or '').strip() != response_id:
            block('successor_parent_response_mismatch')
        if not parent_frame_id or str(candidate.get('parent_frame_id') or '').strip() != parent_frame_id:
            block('successor_parent_frame_mismatch')
        if candidate.get('parent_frame_sequence') != parent_frame_sequence:
            block('successor_parent_frame_sequence_mismatch')

        identity = {
            key: str(candidate.get(key) or '').strip()
            for key in ('proposal_id', 'patch_id', 'idempotency_key')
        }
        for key, value in identity.items():
            if not value:
                block(f'successor_{key}_missing')

        try:
            successor_depth = int(candidate.get('successor_reopen_depth') or 0)
        except (TypeError, ValueError):
            successor_depth = 0
        expected_successor_depth = _graph_patch_successor_parent_depth(response_payload) + 1
        if successor_depth < 1:
            block('successor_reopen_depth_invalid')
        if successor_depth != expected_successor_depth:
            block('successor_reopen_depth_parent_mismatch')
        if successor_depth > _MAX_GRAPH_PATCH_SUCCESSOR_REOPEN_DEPTH:
            block('successor_reopen_depth_exhausted')

        successor_graph = (
            copy.deepcopy(candidate.get('successor_request_phase_graph'))
            if isinstance(candidate.get('successor_request_phase_graph'), dict)
            else {}
        )
        if not successor_graph:
            block('successor_graph_missing')
        expected_successor_digest = str(candidate.get('successor_graph_digest') or '').strip()
        actual_successor_digest = stable_graph_repair_graph_digest(successor_graph)
        if not expected_successor_digest or expected_successor_digest != actual_successor_digest:
            block('successor_graph_digest_mismatch')

        parent_runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), dict)
            else {}
        )
        parent_graph = (
            parent_runtime.get('request_phase_graph')
            if isinstance(parent_runtime.get('request_phase_graph'), dict)
            else {}
        )
        lifecycle = self._graph_patch_successor_lifecycle(parent_graph, candidate)
        if not lifecycle:
            identity_lifecycle = next(
                (
                    dict(item)
                    for item in (parent_graph.get('graph_patch_lifecycle') or [])
                    if isinstance(item, dict)
                    and identity['patch_id']
                    and identity['idempotency_key']
                    and str(item.get('patch_id') or '').strip() == identity['patch_id']
                    and str(item.get('idempotency_key') or '').strip() == identity['idempotency_key']
                ),
                {},
            )
            if identity_lifecycle and str(identity_lifecycle.get('proposal_id') or '').strip() != identity['proposal_id']:
                block('successor_proposal_id_lifecycle_mismatch')
                lifecycle = identity_lifecycle
            else:
                block('successor_patch_lifecycle_missing')
        else:
            for key, value in identity.items():
                if str(lifecycle.get(key) or '').strip() != value:
                    block(f'successor_{key}_lifecycle_mismatch')
            if str(lifecycle.get('autonomy_level') or '').strip().lower() != str(
                candidate.get('autonomy_level') or ''
            ).strip().lower():
                block('successor_autonomy_lifecycle_mismatch')
            if str(lifecycle.get('risk_level') or '').strip().lower() != 'safe_additive':
                block('successor_patch_lifecycle_risk_mismatch')
            if str(lifecycle.get('before_graph_digest') or '').strip() != str(
                candidate.get('before_graph_digest') or ''
            ).strip():
                block('successor_before_graph_digest_mismatch')
            if str(lifecycle.get('patch_digest') or '').strip() != str(
                candidate.get('patch_digest') or ''
            ).strip():
                block('successor_patch_digest_mismatch')
            validation_review = (
                lifecycle.get('validation_review')
                if isinstance(lifecycle.get('validation_review'), dict)
                else {}
            )
            if (
                validation_review.get('kind') != 'ollmo.graph_repair_proposal_review'
                or str(validation_review.get('status') or '').strip().lower() != 'accepted'
            ):
                block('successor_runtime_validation_not_accepted')
            non_terminal_blockers = {
                str(item).strip()
                for item in (lifecycle.get('blocked_reasons') or [])
                if str(item or '').strip()
            } - {'terminal_frame_requires_successor_reopen'}
            if non_terminal_blockers:
                block('successor_patch_lifecycle_has_nonterminal_blocker')

        patch_application = (
            candidate.get('patch_application')
            if isinstance(candidate.get('patch_application'), dict)
            else {}
        )
        if str(patch_application.get('status') or '').strip().lower() != 'applied':
            block('successor_patch_application_not_applied')
        for key in ('proposal_id', 'patch_id', 'idempotency_key'):
            if str(patch_application.get(key) or '').strip() != str(candidate.get(key) or '').strip():
                block(f'successor_patch_application_{key}_mismatch')
        expected_execution_contract = {
            'execution_scope': 'successor_owed_branches_only',
            'root_scoped': False,
            'allow_root_prompt': False,
            'preserve_request_id': True,
        }
        execution_contract = (
            candidate.get('execution_contract')
            if isinstance(candidate.get('execution_contract'), dict)
            else {}
        )
        if execution_contract != expected_execution_contract:
            block('successor_execution_contract_mismatch')
        if blocked_reasons:
            return {'status': 'blocked', 'blocked_reasons': blocked_reasons}

        rederived = self._rederive_graph_patch_successor_application(
            parent_graph,
            lifecycle,
            candidate,
        )
        if rederived.get('status') != 'applied':
            return {
                'status': 'blocked',
                'blocked_reasons': rederived.get('blocked_reasons') or [
                    'successor_patch_reapplication_not_applied'
                ],
            }
        expected_application = (
            rederived.get('application')
            if isinstance(rederived.get('application'), dict)
            else {}
        )
        expected_graph = (
            expected_application.get('graph')
            if isinstance(expected_application.get('graph'), dict)
            else {}
        )
        expected_semantic_graph = copy.deepcopy(expected_graph)
        candidate_semantic_graph = copy.deepcopy(successor_graph)
        for graph_payload in (expected_semantic_graph, candidate_semantic_graph):
            for key in (
                'graph_patch_lifecycle',
                'staged_graph_patches',
                'applied_graph_patches',
                'successor_reopen_requests',
                'successor_reopen_executions',
            ):
                graph_payload.pop(key, None)
        if stable_graph_repair_graph_digest(expected_semantic_graph) != stable_graph_repair_graph_digest(
            candidate_semantic_graph
        ):
            block('successor_rederived_graph_digest_mismatch')

        if str(patch_application.get('kind') or '').strip() != 'ollmo.graph_patch_lifecycle':
            block('successor_patch_application_kind_rederived_mismatch')
        if str(patch_application.get('status') or '').strip().lower() != 'applied':
            block('successor_patch_application_status_rederived_mismatch')
        if str(patch_application.get('proposal_id') or '').strip() != identity['proposal_id']:
            block('successor_patch_application_proposal_id_rederived_mismatch')
        if str(patch_application.get('patch_id') or '').strip() != identity['patch_id']:
            block('successor_patch_application_patch_id_rederived_mismatch')
        if str(patch_application.get('idempotency_key') or '').strip() != identity['idempotency_key']:
            block('successor_patch_application_idempotency_key_rederived_mismatch')
        if str(patch_application.get('original_graph_digest') or '').strip() != str(
            lifecycle.get('before_graph_digest') or ''
        ).strip():
            block('successor_patch_application_original_graph_digest_rederived_mismatch')
        def is_current_patch_record(record: Any) -> bool:
            return bool(
                isinstance(record, dict)
                and all(
                    str(record.get(key) or '').strip() == value
                    for key, value in identity.items()
                )
            )

        patched_graph_before_lifecycle = copy.deepcopy(successor_graph)
        for key in ('graph_patch_lifecycle', 'applied_graph_patches'):
            remaining = [
                dict(item)
                for item in (patched_graph_before_lifecycle.get(key) or [])
                if isinstance(item, dict) and not is_current_patch_record(item)
            ]
            if remaining:
                patched_graph_before_lifecycle[key] = remaining
            else:
                patched_graph_before_lifecycle.pop(key, None)
        if str(patch_application.get('patched_graph_digest') or '').strip() != (
            stable_graph_repair_graph_digest(patched_graph_before_lifecycle)
        ):
            block('successor_patch_application_patched_graph_digest_rederived_mismatch')

        for key in ('applied_branch_ids', 'applied_phase_ids', 'applied_obligation_ids'):
            actual_ids = [
                str(item).strip()
                for item in (patch_application.get(key) or [])
                if str(item or '').strip()
            ]
            expected_ids = [
                str(item).strip()
                for item in (expected_application.get(key) or [])
                if str(item or '').strip()
            ]
            if actual_ids != expected_ids:
                block(f'successor_patch_application_{key}_rederived_mismatch')

        def normalized_edges(values: Any) -> list[tuple[str, str]]:
            return sorted(
                (
                    str(item.get('target_id') or '').strip(),
                    str(item.get('source_id') or '').strip(),
                )
                for item in (values or [])
                if isinstance(item, dict)
            )

        if normalized_edges(patch_application.get('applied_dependency_edges')) != normalized_edges(
            expected_application.get('applied_dependency_edges')
        ):
            block('successor_patch_application_applied_dependency_edges_rederived_mismatch')

        owed_branch_ids, owed_output_obligations = _graph_patch_successor_owed_scope(
            expected_graph,
            expected_application,
            fallback_branch_ids=lifecycle.get('scheduled_branch_ids') or [],
        )
        candidate_owed_branch_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in (candidate.get('owed_branch_ids') or [])
                if str(item or '').strip()
            )
        )
        if not owed_branch_ids:
            block('successor_owed_branch_ids_missing')
        if candidate_owed_branch_ids != owed_branch_ids:
            block('successor_owed_branch_scope_mismatch')
        if (candidate.get('owed_output_obligations') or []) != owed_output_obligations:
            block('successor_owed_obligation_scope_mismatch')

        expected_reopen_key, expected_execution_key = _graph_patch_successor_keys(
            response_id=response_id,
            parent_frame_id=parent_frame_id,
            patch_id=identity['patch_id'],
            idempotency_key=identity['idempotency_key'],
            owed_branch_ids=owed_branch_ids,
            successor_depth=expected_successor_depth,
        )
        if str(candidate.get('successor_reopen_key') or '').strip() != expected_reopen_key:
            block('successor_reopen_key_mismatch')
        if str(candidate.get('successor_execution_key') or '').strip() != expected_execution_key:
            block('successor_execution_key_mismatch')
        expected_relation = {
            'kind': 'graph_patch_reopen_successor',
            'reason': 'graph_patch_reopen',
            'parent_response_id': response_id,
            'parent_frame_id': parent_frame_id,
            'parent_frame_sequence': parent_frame_sequence,
            'successor_reopen_depth': expected_successor_depth,
        }
        if candidate.get('frame_relation') != expected_relation:
            block('successor_frame_relation_mismatch')
        if blocked_reasons:
            return {'status': 'blocked', 'blocked_reasons': blocked_reasons}

        synthetic_result = {
            'status': 'applied',
            'patch_id': candidate.get('patch_id'),
            'applied_branch_ids': owed_branch_ids,
            'applied_phase_ids': expected_application.get('applied_phase_ids') or owed_branch_ids,
            'applied_obligation_ids': expected_application.get('applied_obligation_ids') or [],
            'applied_dependency_edges': expected_application.get('applied_dependency_edges') or [],
        }
        successor_payload = copy.deepcopy(response_payload)
        successor_runtime = (
            dict(successor_payload.get('runtime') or {})
            if isinstance(successor_payload.get('runtime'), dict)
            else {}
        )
        successor_diagnostics = (
            dict(successor_runtime.get('developer_diagnostics') or {})
            if isinstance(successor_runtime.get('developer_diagnostics'), dict)
            else {}
        )
        successor_diagnostics['graph_patch_lifecycle_results'] = [synthetic_result]
        successor_runtime['request_phase_graph'] = successor_graph
        successor_runtime['developer_diagnostics'] = successor_diagnostics
        successor_payload['runtime'] = successor_runtime
        successor_payload, artifact_gap = self._reconcile_applied_graph_patch_late_fill_gap(
            successor_payload,
            None,
        )
        if not isinstance(artifact_gap, dict):
            return {
                'status': 'blocked',
                'blocked_reasons': ['successor_owed_branches_not_executable'],
            }
        pending_branches = [
            dict(item)
            for item in (artifact_gap.get('pending_branches') or [])
            if isinstance(item, dict)
        ]
        scheduled_branch_ids = [
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in pending_branches
            if str(item.get('branch_id') or item.get('phase_id') or '').strip()
        ]
        if set(scheduled_branch_ids) != set(owed_branch_ids):
            return {
                'status': 'blocked',
                'blocked_reasons': ['successor_owed_branches_not_exactly_executable'],
                'scheduled_branch_ids': scheduled_branch_ids,
                'owed_branch_ids': owed_branch_ids,
            }
        for branch in pending_branches:
            branch_contract = (
                branch.get('execution_contract')
                if isinstance(branch.get('execution_contract'), dict)
                else {}
            )
            if (
                str(branch_contract.get('execution_scope') or '').strip().lower()
                in {'root', 'root_scoped', 'whole_request', 'original_prompt'}
                or branch_contract.get('root_scoped') is True
                or branch_contract.get('allow_root_prompt') is True
            ):
                block('successor_branch_root_prompt_execution_forbidden')
            if branch.get('needs_external_input') is True:
                block('successor_branch_needs_external_input')
            if branch.get('materialization_blocked') is True:
                block('successor_branch_materialization_blocked')
        if blocked_reasons:
            return {'status': 'blocked', 'blocked_reasons': blocked_reasons}

        prior_late_fill = (
            response_payload.get('late_fill')
            if isinstance(response_payload.get('late_fill'), dict)
            else {}
        )
        owed_id_set = set(owed_branch_ids)

        def without_owed(values: Any) -> list[dict[str, Any]]:
            return [
                dict(item)
                for item in (values or [])
                if isinstance(item, dict)
                and str(item.get('branch_id') or item.get('phase_id') or '').strip()
                not in owed_id_set
            ]

        pending_capabilities = list(
            dict.fromkeys(
                str(item.get('capability') or '').strip().lower()
                for item in pending_branches
                if str(item.get('capability') or '').strip()
            )
        )
        execution_record = _compact_payload(
            {
                'kind': 'ollmo.graph_patch_successor_reopen_execution',
                'status': 'queued',
                'successor_reopen_key': candidate.get('successor_reopen_key'),
                'successor_execution_key': candidate.get('successor_execution_key'),
                'patch_id': candidate.get('patch_id'),
                'idempotency_key': candidate.get('idempotency_key'),
                'successor_graph_digest': expected_successor_digest,
                'successor_reopen_depth': successor_depth,
                'parent_response_id': response_id,
                'parent_frame_id': parent_frame_id,
                'parent_frame_sequence': parent_frame_sequence,
                'scheduled_branch_ids': scheduled_branch_ids,
                'root_prompt_replay': False,
            }
        )
        artifact_gap.update(
            {
                'code': 'graph_patch_late_fill',
                'trigger': 'graph_patch_successor_reopen',
                'pending_branch_scope': 'exact_successor_reopen',
                'authoritative_pending_branches': True,
                'preserve_request_id': True,
                'pending_branches': pending_branches,
                'pending_capabilities': pending_capabilities,
                'active_branches': [],
                'completed_branches': without_owed(prior_late_fill.get('completed_branches')),
                'failed_branches': without_owed(prior_late_fill.get('failed_branches')),
                'cancelled_branches': without_owed(prior_late_fill.get('cancelled_branches')),
                'successor_reopen_execution': execution_record,
                'execution_contract': {
                    'execution_scope': 'successor_owed_branches_only',
                    'root_scoped': False,
                    'allow_root_prompt': False,
                    'preserve_request_id': True,
                },
            }
        )
        build_late_fill_state = self._hook('build_late_fill_state')
        attach_late_fill_state = self._hook('attach_late_fill_state')
        pending_late_fill = build_late_fill_state(
            artifact_gap,
            status='pending',
            prior_state=prior_late_fill,
            extra={
                'pending_branch_scope': 'exact_successor_reopen',
                'authoritative_pending_branches': True,
                'successor_reopen_execution': execution_record,
                'fill_results': prior_late_fill.get('fill_results') or [],
            },
        )
        successor_payload = attach_late_fill_state(successor_payload, pending_late_fill)

        successor_runtime = (
            dict(successor_payload.get('runtime') or {})
            if isinstance(successor_payload.get('runtime'), dict)
            else {}
        )
        execution_graph = (
            copy.deepcopy(successor_runtime.get('request_phase_graph'))
            if isinstance(successor_runtime.get('request_phase_graph'), dict)
            else successor_graph
        )
        candidate_record = {
            key: copy.deepcopy(value)
            for key, value in candidate.items()
            if key != 'successor_request_phase_graph'
        }
        candidate_record.update(
            {
                'status': 'applied_to_successor',
                'runtime_effect': 'successor_late_fill_queued',
                'execution': execution_record,
            }
        )
        existing_requests = [
            dict(item)
            for item in (execution_graph.get('successor_reopen_requests') or [])
            if isinstance(item, dict)
            and str(item.get('successor_execution_key') or '').strip()
            != str(candidate.get('successor_execution_key') or '').strip()
        ]
        execution_graph['successor_reopen_requests'] = [*existing_requests, candidate_record]
        existing_executions = [
            dict(item)
            for item in (execution_graph.get('successor_reopen_executions') or [])
            if isinstance(item, dict)
            and str(item.get('successor_execution_key') or '').strip()
            != str(candidate.get('successor_execution_key') or '').strip()
        ]
        execution_graph['successor_reopen_executions'] = [*existing_executions, execution_record]
        successor_runtime['request_phase_graph'] = execution_graph
        successor_diagnostics = (
            dict(successor_runtime.get('developer_diagnostics') or {})
            if isinstance(successor_runtime.get('developer_diagnostics'), dict)
            else {}
        )
        successor_diagnostics['graph_patch_successor_reopen_execution'] = execution_record
        successor_diagnostics['graph_patch_successor_reopen_request'] = candidate_record
        successor_runtime['developer_diagnostics'] = successor_diagnostics
        successor_runtime['lifecycle_state'] = 'late_fill_pending'
        successor_payload['runtime'] = successor_runtime
        successor_payload['lifecycle_state'] = 'late_fill_pending'
        successor_payload['late_fill_pending'] = True
        successor_payload['frame_relation'] = {
            **dict(candidate.get('frame_relation') or {}),
            'kind': 'graph_patch_reopen_successor',
            'reason': 'graph_patch_reopen',
            'response_id': response_id,
            'parent_response_id': response_id,
            'parent_frame_id': parent_frame_id,
            'parent_frame_sequence': parent_frame_sequence,
            'successor_reopen_depth': successor_depth,
            'successor_execution_key': candidate.get('successor_execution_key'),
        }
        return {
            'status': 'queued',
            'response_payload': successor_payload,
            'artifact_gap': _compact_payload(artifact_gap),
            'successor_reopen_request': candidate_record,
            'execution': execution_record,
        }

    def prepare_terminal_graph_patch_successor(
        self,
        response_payload: dict[str, Any],
        *,
        graph_repair_autonomy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Produce and prepare one bounded terminal graph-patch successor wave."""

        if not isinstance(response_payload, dict):
            return {'status': 'not_applicable', 'reason': 'response_payload_invalid'}
        autonomy_description = (
            describe_graph_repair_autonomy(
                graph_repair_autonomy,
                source='explicit_override',
                configured=True,
            )
            if graph_repair_autonomy is not None
            else describe_graph_repair_autonomy_from_env()
        )
        autonomy_level = normalize_graph_repair_autonomy(
            autonomy_description.get('autonomy_level')
        )
        if autonomy_level == 'off':
            return {'status': 'not_applicable', 'reason': 'graph_repair_autonomy_off'}
        if autonomy_level not in {'apply_safe', 'apply_enforced'}:
            return {
                'status': 'not_applicable',
                'reason': 'graph_repair_autonomy_not_executable',
                'autonomy_level': autonomy_level,
            }
        parent_frame = (
            response_payload.get('response_frame')
            if isinstance(response_payload.get('response_frame'), dict)
            else {}
        )
        if not str(parent_frame.get('frame_id') or '').strip():
            return {'status': 'not_applicable', 'reason': 'terminal_parent_frame_missing'}

        reviewed = self._attach_redraw_scope_ladder_review(copy.deepcopy(response_payload))
        reviewed = self._attach_runtime_graph_repair_evidence(reviewed)
        reviewed = self._attach_graph_patch_lifecycle(
            reviewed,
            graph_repair_autonomy=autonomy_level,
        )
        runtime = reviewed.get('runtime') if isinstance(reviewed.get('runtime'), dict) else {}
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        candidates = [
            dict(item)
            for item in (graph.get('successor_reopen_requests') or [])
            if isinstance(item, dict)
            and str(item.get('status') or '').strip().lower() == 'candidate'
        ]
        if not candidates:
            return {
                'status': 'not_applicable',
                'reason': 'no_safe_terminal_graph_patch_successor',
                'response_payload': reviewed,
            }

        blocked_results: list[dict[str, Any]] = []
        for candidate in candidates:
            if str(candidate.get('autonomy_level') or '').strip().lower() != autonomy_level:
                blocked_results.append(
                    {
                        'status': 'blocked',
                        'blocked_reasons': ['successor_autonomy_binding_mismatch'],
                    }
                )
                continue
            lifecycle = self._graph_patch_successor_lifecycle(graph, candidate)
            if autonomy_level == 'apply_enforced':
                policy_review = build_enforced_policy_review(
                    autonomy_level=autonomy_level,
                    lifecycle=lifecycle,
                    request_phase_graph=graph,
                )
                if not enforced_policy_allows_application(policy_review):
                    blocked_results.append(
                        {
                            'status': 'blocked',
                            'blocked_reasons': policy_review.get('blocked_reasons') or [
                                'successor_enforced_policy_recheck_blocked'
                            ],
                            'enforced_policy_review': policy_review,
                        }
                    )
                    continue
                if (
                    candidate.get('allowed_by_policy') is not True
                    or str(candidate.get('policy_mode') or '').strip().lower() != 'safe_v1'
                ):
                    blocked_results.append(
                        {
                            'status': 'blocked',
                            'blocked_reasons': ['successor_enforced_policy_binding_mismatch'],
                        }
                    )
                    continue
            prepared = self._prepare_graph_patch_successor_reopen(reviewed, candidate)
            if prepared.get('status') == 'queued':
                prepared['autonomy'] = autonomy_description
                return prepared
            blocked_results.append(prepared)
        blocked_reasons = list(
            dict.fromkeys(
                str(reason).strip()
                for result in blocked_results
                for reason in (result.get('blocked_reasons') or [])
                if str(reason or '').strip()
            )
        )
        return {
            'status': 'blocked',
            'blocked_reasons': blocked_reasons or ['terminal_graph_patch_successor_blocked'],
            'response_payload': reviewed,
        }

    @staticmethod
    def _partial_graph_rebase_branch_records(
        graph: dict[str, Any],
        owed_branch_ids: list[str],
    ) -> list[dict[str, Any]]:
        owed = set(owed_branch_ids)
        records: dict[str, dict[str, Any]] = {}
        for collection in ('phases', 'downstream_branches'):
            for item in graph.get(collection) or []:
                if not isinstance(item, dict):
                    continue
                branch_id = str(item.get('branch_id') or item.get('phase_id') or '').strip()
                if branch_id not in owed:
                    continue
                records[branch_id] = {**records.get(branch_id, {}), **copy.deepcopy(item)}
        return [records[branch_id] for branch_id in owed_branch_ids if branch_id in records]

    @staticmethod
    def _payload_has_partial_graph_rebase_execution(
        response_payload: Any,
        execution_key: str,
    ) -> bool:
        if not isinstance(response_payload, dict) or not execution_key:
            return False
        late_fill = (
            response_payload.get('late_fill')
            if isinstance(response_payload.get('late_fill'), dict)
            else {}
        )
        execution = (
            late_fill.get('partial_rebase_execution')
            if isinstance(late_fill.get('partial_rebase_execution'), dict)
            else {}
        )
        if str(execution.get('execution_key') or '').strip() == execution_key:
            return True
        runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), dict)
            else {}
        )
        graph = (
            runtime.get('request_phase_graph')
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        return any(
            isinstance(item, dict)
            and str(item.get('execution_key') or '').strip() == execution_key
            for item in graph.get('successor_rebase_executions') or []
        )

    def prepare_terminal_partial_graph_rebase_successor(
        self,
        response_payload: dict[str, Any],
        *,
        proposal_id: str,
        trusted_authorization: Optional[dict[str, Any]],
        graph_rebase_autonomy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Prepare one exact operator-authorized partial-rebase successor wave."""

        if not isinstance(response_payload, dict):
            return {'status': 'not_applicable', 'reason': 'response_payload_invalid'}
        environment_autonomy = describe_graph_rebase_autonomy_from_env()
        if (
            environment_autonomy.get('configured') is True
            and normalize_graph_rebase_autonomy(environment_autonomy.get('autonomy_level')) == 'off'
        ):
            return {'status': 'not_applicable', 'reason': 'graph_rebase_autonomy_off'}
        requested_autonomy = normalize_graph_rebase_autonomy(
            graph_rebase_autonomy
            if graph_rebase_autonomy is not None
            else environment_autonomy.get('autonomy_level')
        )
        if requested_autonomy != 'apply_reviewed':
            return {
                'status': 'not_applicable',
                'reason': 'graph_rebase_autonomy_not_apply_reviewed',
                'autonomy_level': requested_autonomy,
            }

        # Recompute the ladder from the exact frozen parent at the execution
        # boundary.  A persisted review is evidence, not an evergreen grant;
        # smaller repair scopes may have become authoritative meanwhile.
        reviewed_parent = self._attach_redraw_scope_ladder_review(
            copy.deepcopy(response_payload)
        )
        if isinstance(reviewed_parent, dict):
            response_payload = reviewed_parent

        blocked_reasons: list[str] = []

        def block(reason: str) -> None:
            token = str(reason or '').strip()
            if token and token not in blocked_reasons:
                blocked_reasons.append(token)

        response_id = str(
            response_payload.get('response_id')
            or response_payload.get('id')
            or ''
        ).strip()
        parent_frame = (
            response_payload.get('response_frame')
            if isinstance(response_payload.get('response_frame'), dict)
            else {}
        )
        parent_frame_id = str(parent_frame.get('frame_id') or '').strip()
        parent_frame_sequence = parent_frame.get('frame_sequence')
        if not response_id:
            block('partial_rebase_parent_response_id_missing')
        if not parent_frame_id:
            block('partial_rebase_parent_frame_missing')
        lifecycle_state = str(
            response_payload.get('lifecycle_state')
            or response_payload.get('status')
            or ''
        ).strip().lower()
        if lifecycle_state not in _FROZEN_GRAPH_PATCH_RESPONSE_STATES:
            block('partial_rebase_parent_not_frozen')
        late_fill_status = _runtime_rebase_late_fill_status(response_payload)
        if late_fill_status in _ACTIVE_LATE_FILL_STATUSES:
            block('active_late_fill_must_settle')

        authorization = (
            copy.deepcopy(trusted_authorization)
            if isinstance(trusted_authorization, dict)
            else {}
        )
        if not authorization:
            block('trusted_graph_rebase_authorization_missing')
        if authorization.get('kind') != 'ollmo.graph_rebase_authorization':
            block('graph_rebase_authorization_kind_mismatch')
        if str(authorization.get('source') or '').strip() != 'runtime_operator_registry':
            block('graph_rebase_authorization_provenance_mismatch')
        authorization_record_id = str(
            authorization.get('registry_record_id')
            or authorization.get('record_id')
            or ''
        ).strip()
        if not authorization_record_id:
            block('graph_rebase_authorization_record_id_missing')
        if str(authorization.get('response_id') or '').strip() != response_id:
            block('graph_rebase_authorization_response_mismatch')
        if str(authorization.get('frame_id') or '').strip() != parent_frame_id:
            block('graph_rebase_authorization_frame_mismatch')

        runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), dict)
            else {}
        )
        graph = (
            copy.deepcopy(runtime.get('request_phase_graph'))
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            block('partial_rebase_parent_graph_missing')
        current_scope = (
            graph.get('redraw_scope_ladder_review')
            if isinstance(graph.get('redraw_scope_ladder_review'), dict)
            else {}
        )
        selected_scope = str(current_scope.get('selected_scope') or '').strip().lower()
        if selected_scope in _SMALLER_REDRAW_SCOPES:
            block('smaller_redraw_scope_precedes_rebase')
        elif selected_scope != 'partial_subtree_rebase':
            block('current_partial_rebase_scope_not_selected')

        requested_proposal_id = str(proposal_id or '').strip()
        proposal = next(
            (
                copy.deepcopy(item)
                for item in graph.get('graph_rebase_proposals') or []
                if isinstance(item, dict)
                and str(item.get('proposal_id') or '').strip() == requested_proposal_id
            ),
            {},
        )
        if not proposal:
            block('partial_rebase_source_proposal_missing')
        if str(authorization.get('proposal_id') or '').strip() != requested_proposal_id:
            block('graph_rebase_authorization_proposal_mismatch')
        requested_class = str(proposal.get('requested_rebase_class') or '').strip().lower()
        if requested_class != 'partial_subtree_rebase':
            block('apply_reviewed_partial_rebase_only')
        if str(authorization.get('requested_rebase_class') or '').strip().lower() != requested_class:
            block('graph_rebase_authorization_class_mismatch')
        if str(authorization.get('base_graph_digest') or '').strip() != str(
            proposal.get('base_graph_digest') or ''
        ).strip():
            block('graph_rebase_authorization_base_digest_mismatch')
        if str(authorization.get('candidate_graph_digest') or '').strip() != str(
            proposal.get('candidate_graph_digest') or ''
        ).strip():
            block('graph_rebase_authorization_candidate_digest_mismatch')
        if blocked_reasons:
            return {'status': 'blocked', 'blocked_reasons': blocked_reasons}

        closure_review = (
            runtime.get('graph_closure_review')
            if isinstance(runtime.get('graph_closure_review'), dict)
            else {}
        )
        guarded_root_prompt = self._graph_rebase_root_prompt(response_payload)
        if not guarded_root_prompt:
            return {
                'status': 'blocked',
                'blocked_reasons': [
                    'partial_rebase_current_root_prompt_truth_unavailable'
                ],
            }
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=closure_review,
            artifact_payload=(
                response_payload.get('artifacts')
                if isinstance(response_payload.get('artifacts'), dict)
                else None
            ),
            accepted_learning_hints=(
                runtime.get('accepted_learning_hints')
                if isinstance(runtime.get('accepted_learning_hints'), dict)
                else None
            ),
            trusted_authorization=authorization,
            root_prompt=guarded_root_prompt,
        )
        if str(review.get('status') or '').strip().lower() != 'accepted':
            return {
                'status': 'blocked',
                'blocked_reasons': review.get('blocked_reasons') or [
                    'partial_rebase_validation_not_accepted'
                ],
                'review': review,
            }
        execution_proof = build_graph_rebase_execution_contract_proof(
            graph,
            proposal,
            root_prompt=guarded_root_prompt,
        )
        if execution_proof.get('status') != 'passed':
            return {
                'status': 'blocked',
                'blocked_reasons': [
                    'apply_reviewed_execution_contract_proof_required',
                    *(execution_proof.get('blocked_reasons') or []),
                ],
                'review': review,
                'execution_contract_proof': execution_proof,
            }
        owed_branch_ids = [
            str(item).strip()
            for item in (execution_proof.get('owed_branch_ids') or [])
            if str(item or '').strip()
        ]
        current_branch_ids = {
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in graph.get('downstream_branches') or []
            if isinstance(item, dict)
        }
        reopened_existing = sorted(set(owed_branch_ids) & current_branch_ids)
        if reopened_existing:
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_existing_branch_reopen_not_supported'],
                'existing_branch_ids': reopened_existing,
            }

        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=graph,
            rebase_review=review,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
        )
        if lifecycle.get('status') != 'staged':
            return {
                'status': 'blocked',
                'blocked_reasons': lifecycle.get('blocked_reasons') or [
                    'partial_rebase_lifecycle_not_staged'
                ],
                'review': review,
                'lifecycle': lifecycle,
            }
        application = apply_validated_graph_rebase(
            graph,
            lifecycle,
            autonomy_level='apply_reviewed',
            trusted_authorization=authorization,
            root_prompt=guarded_root_prompt,
        )
        if application.get('status') != 'applied':
            return {
                'status': 'blocked',
                'blocked_reasons': application.get('blocked_reasons') or [
                    'partial_rebase_successor_request_not_created'
                ],
                'review': review,
                'lifecycle': lifecycle,
                'application': application,
            }
        successor_request = (
            application.get('successor_request')
            if isinstance(application.get('successor_request'), dict)
            else {}
        )
        successor_graph = (
            copy.deepcopy(successor_request.get('successor_graph'))
            if isinstance(successor_request.get('successor_graph'), dict)
            else {}
        )
        if stable_graph_digest(successor_graph) != str(
            proposal.get('candidate_graph_digest') or ''
        ).strip():
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_successor_graph_digest_mismatch'],
            }
        pending_branches = self._partial_graph_rebase_branch_records(
            successor_graph,
            owed_branch_ids,
        )
        if [
            str(item.get('branch_id') or item.get('phase_id') or '').strip()
            for item in pending_branches
        ] != owed_branch_ids:
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_owed_branch_records_missing'],
            }
        for branch in pending_branches:
            branch['execution_contract'] = {
                **dict(branch.get('execution_contract') or {}),
                'execution_scope': 'partial_rebase_owed_branches_only',
                'root_scoped': False,
                'allow_root_prompt': False,
                'preserve_request_id': True,
                'forbidden_root_prompt_digest': (
                    stable_graph_rebase_prompt_digest(guarded_root_prompt)
                    or str(execution_proof.get('root_prompt_guard_digest') or '').strip()
                ),
            }

        partial_rebase_depth = _partial_graph_rebase_parent_depth(response_payload) + 1
        if partial_rebase_depth > _MAX_PARTIAL_GRAPH_REBASE_SUCCESSOR_DEPTH:
            return {
                'status': 'blocked',
                'blocked_reasons': ['partial_rebase_successor_depth_exhausted'],
            }
        successor_key, execution_key = _partial_graph_rebase_execution_keys(
            response_id=response_id,
            parent_frame_id=parent_frame_id,
            proposal_id=requested_proposal_id,
            review_id=str(review.get('review_id') or '').strip(),
            rebase_id=str(lifecycle.get('rebase_id') or '').strip(),
            idempotency_key=str(lifecycle.get('idempotency_key') or '').strip(),
            candidate_graph_digest=str(proposal.get('candidate_graph_digest') or '').strip(),
            diff_digest=str(lifecycle.get('diff_digest') or '').strip(),
            scope_digest=str(execution_proof.get('scope_digest') or '').strip(),
            owed_branch_ids=owed_branch_ids,
            partial_rebase_depth=partial_rebase_depth,
            authorization_record_id=authorization_record_id,
        )
        if self._payload_has_partial_graph_rebase_execution(response_payload, execution_key):
            return {
                'status': 'already_recorded',
                'execution_key': execution_key,
            }

        execution_record = {
            'kind': 'ollmo.graph_rebase_partial_successor_execution',
            'status': 'queued',
            'successor_key': successor_key,
            'execution_key': execution_key,
            'response_id': response_id,
            'parent_frame_id': parent_frame_id,
            'parent_frame_sequence': parent_frame_sequence,
            'proposal_id': requested_proposal_id,
            'review_id': review.get('review_id'),
            'rebase_id': lifecycle.get('rebase_id'),
            'idempotency_key': lifecycle.get('idempotency_key'),
            'candidate_graph_digest': proposal.get('candidate_graph_digest'),
            'diff_digest': lifecycle.get('diff_digest'),
            'scope_digest': execution_proof.get('scope_digest'),
            'authorization_record_id': authorization_record_id,
            'partial_rebase_depth': partial_rebase_depth,
            'scheduled_branch_ids': owed_branch_ids,
            'root_prompt_replay': False,
        }
        execution_contract = {
            'execution_scope': 'partial_rebase_owed_branches_only',
            'root_scoped': False,
            'allow_root_prompt': False,
            'preserve_request_id': True,
            'forbidden_root_prompt_digest': (
                stable_graph_rebase_prompt_digest(guarded_root_prompt)
                or str(execution_proof.get('root_prompt_guard_digest') or '').strip()
            ),
        }
        prior_late_fill = (
            response_payload.get('late_fill')
            if isinstance(response_payload.get('late_fill'), dict)
            else {}
        )
        owed_set = set(owed_branch_ids)

        def without_owed(values: Any) -> list[dict[str, Any]]:
            return [
                copy.deepcopy(item)
                for item in (values or [])
                if isinstance(item, dict)
                and str(item.get('branch_id') or item.get('phase_id') or '').strip()
                not in owed_set
            ]

        pending_capabilities = list(
            dict.fromkeys(
                str(item.get('capability') or '').strip().lower()
                for item in pending_branches
                if str(item.get('capability') or '').strip()
            )
        )
        artifact_gap = {
            'code': 'graph_rebase_partial_late_fill',
            'trigger': 'graph_rebase_partial_successor',
            'pending_branch_scope': 'exact_partial_rebase_successor',
            'authoritative_pending_branches': True,
            'preserve_request_id': True,
            'pending_branches': pending_branches,
            'pending_capabilities': pending_capabilities,
            'active_branches': [],
            'completed_branches': without_owed(prior_late_fill.get('completed_branches')),
            'failed_branches': without_owed(prior_late_fill.get('failed_branches')),
            'cancelled_branches': without_owed(prior_late_fill.get('cancelled_branches')),
            'partial_rebase_execution': execution_record,
            'execution_contract': execution_contract,
        }
        successor_payload = copy.deepcopy(response_payload)
        pending_late_fill = self._hook('build_late_fill_state')(
            artifact_gap,
            status='pending',
            prior_state=prior_late_fill,
            extra={
                'pending_branch_scope': 'exact_partial_rebase_successor',
                'authoritative_pending_branches': True,
                'partial_rebase_execution': execution_record,
                'fill_results': prior_late_fill.get('fill_results') or [],
            },
        )
        successor_payload = self._hook('attach_late_fill_state')(
            successor_payload,
            pending_late_fill,
        )

        application_graph = (
            application.get('graph')
            if isinstance(application.get('graph'), dict)
            else {}
        )
        execution_graph = copy.deepcopy(successor_graph)
        for key in (
            'graph_rebase_proposals',
            'graph_rebase_reviews',
            'graph_rebase_lifecycle',
            'staged_graph_rebases',
            'applied_graph_rebases',
            'successor_rebase_requests',
        ):
            if application_graph.get(key) not in (None, '', [], {}):
                execution_graph[key] = copy.deepcopy(application_graph.get(key))
        execution_graph['successor_rebase_executions'] = [execution_record]
        successor_runtime = (
            copy.deepcopy(successor_payload.get('runtime'))
            if isinstance(successor_payload.get('runtime'), dict)
            else {}
        )
        successor_runtime['request_phase_graph'] = execution_graph
        diagnostics = (
            copy.deepcopy(successor_runtime.get('developer_diagnostics'))
            if isinstance(successor_runtime.get('developer_diagnostics'), dict)
            else {}
        )
        diagnostics['graph_rebase_partial_successor_execution'] = execution_record
        diagnostics['graph_rebase_partial_execution_contract_proof'] = execution_proof
        successor_runtime['developer_diagnostics'] = diagnostics
        successor_runtime['lifecycle_state'] = 'late_fill_pending'
        successor_payload['runtime'] = successor_runtime
        successor_payload['lifecycle_state'] = 'late_fill_pending'
        successor_payload['late_fill_pending'] = True
        successor_payload['frame_relation'] = {
            'kind': 'graph_rebase_partial_successor',
            'reason': 'operator_authorized_partial_graph_rebase',
            'response_id': response_id,
            'parent_response_id': response_id,
            'parent_frame_id': parent_frame_id,
            'parent_frame_sequence': parent_frame_sequence,
            'proposal_id': requested_proposal_id,
            'rebase_id': lifecycle.get('rebase_id'),
            'authorization_record_id': authorization_record_id,
            'successor_key': successor_key,
            'execution_key': execution_key,
            'scope_digest': execution_proof.get('scope_digest'),
            'partial_rebase_depth': partial_rebase_depth,
        }
        return {
            'status': 'queued',
            'response_payload': successor_payload,
            'artifact_gap': artifact_gap,
            'successor_rebase_request': successor_request,
            'execution': execution_record,
            'review': review,
            'lifecycle': lifecycle,
        }

    def _graph_rebase_root_prompt(self, payload: Any) -> str:
        """Resolve current-turn request truth only for root-replay rejection."""

        if not isinstance(payload, dict):
            return ''
        candidates = [
            payload.get('request') if isinstance(payload.get('request'), dict) else None,
            payload,
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ('prompt', 'prompt_text', 'current_turn_prompt'):
                direct_prompt = str(candidate.get(key) or '').strip()
                if direct_prompt:
                    return direct_prompt
            for hook_name in (
                'extract_responses_current_turn_prompt',
                'extract_responses_prompt',
            ):
                try:
                    prompt = str(self._hook(hook_name)(candidate) or '').strip()
                except (KeyError, TypeError, ValueError):
                    prompt = ''
                if prompt:
                    return prompt
        return ''

    @staticmethod
    def _collect_explicit_graph_rebase_proposals(
        graph: dict[str, Any],
        closure_review: dict[str, Any],
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        for item in graph.get('graph_rebase_proposals') or []:
            if isinstance(item, dict):
                proposals.append(dict(item))
        for source in (
            closure_review,
            closure_review.get('global_semantic_closure_review')
            if isinstance(closure_review.get('global_semantic_closure_review'), dict)
            else {},
            closure_review.get('semantic_review_verdict')
            if isinstance(closure_review.get('semantic_review_verdict'), dict)
            else {},
        ):
            if not isinstance(source, dict):
                continue
            for key in ('graph_rebase_proposals', 'graph_rebase_candidates', 'runtime_graph_rebase_proposals'):
                for item in source.get(key) or []:
                    if isinstance(item, dict):
                        proposals.append(dict(item))
        return ResponsesRequestRuntimeOwner._dedupe_graph_repair_records(
            proposals,
            'proposal_id',
            'id',
        )

    def _attach_runtime_graph_rebase_evidence(
        self,
        response_payload: dict[str, Any],
        *,
        root_prompt: str = '',
    ) -> dict[str, Any]:
        """Attach backend-owned runtime graph-rebase proposal/review diagnostics."""

        if not isinstance(response_payload, dict):
            return response_payload
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        candidate_context = developer_diagnostics.pop(
            _RESPONSE_TIME_GRAPH_REBASE_CANDIDATE_KEY,
            None,
        )
        developer_diagnostics.pop('runtime_graph_rebase_candidate_review', None)
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            runtime['developer_diagnostics'] = developer_diagnostics
            updated['runtime'] = runtime
            return updated
        guarded_root_prompt = str(
            root_prompt or self._graph_rebase_root_prompt(updated)
        ).strip()
        closure_review = (
            dict(runtime.get('graph_closure_review') or {})
            if isinstance(runtime.get('graph_closure_review'), dict)
            else {}
        )
        accepted_learning_hints = (
            dict(runtime.get('accepted_learning_hints') or {})
            if isinstance(runtime.get('accepted_learning_hints'), dict)
            else {}
        )
        proposals = self._collect_explicit_graph_rebase_proposals(graph, closure_review)
        candidate_review: dict[str, Any] = {}
        if isinstance(candidate_context, dict):
            candidate_graph = (
                copy.deepcopy(candidate_context.get('candidate_graph'))
                if isinstance(candidate_context.get('candidate_graph'), dict)
                else {}
            )
            diff = build_graph_rebase_diff(graph, candidate_graph)
            late_fill_status = _runtime_rebase_late_fill_status(updated)
            closure_evidence = _runtime_rebase_closure_evidence(closure_review)
            scope_review = (
                dict(graph.get('redraw_scope_ladder_review') or {})
                if isinstance(graph.get('redraw_scope_ladder_review'), dict)
                else {}
            )
            smaller_scope = _runtime_rebase_smaller_scope(scope_review)
            candidate_review = {
                'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                'candidate_origin': candidate_context.get('candidate_origin'),
                'base_graph_digest': diff.get('base_graph_digest'),
                'candidate_graph_digest': diff.get('candidate_graph_digest'),
                'diff_summary': _runtime_rebase_diff_summary(diff),
                'runtime_effect': 'none',
            }
            blocked_reason = ''
            if late_fill_status:
                blocked_reason = 'active_late_fill_must_settle'
                candidate_review['late_fill_status'] = late_fill_status
            elif not closure_evidence:
                blocked_reason = 'current_structural_closure_evidence_missing'
            elif int(diff.get('meaningful_change_count') or 0) < 1:
                blocked_reason = 'candidate_graph_has_no_meaningful_change'
            elif not _runtime_rebase_has_structural_change(diff):
                blocked_reason = 'candidate_change_is_additive_repair_only'
            elif smaller_scope:
                blocked_reason = 'smaller_redraw_scope_precedes_rebase'
                candidate_review['smaller_scope'] = smaller_scope

            scope_fields: dict[str, Any] = {}
            if not blocked_reason:
                scope_fields = _runtime_rebase_scope_from_diff(
                    graph,
                    candidate_graph,
                    diff,
                    preferred_scope=str(closure_evidence.get('preferred_scope') or ''),
                )
                if not scope_fields:
                    blocked_reason = 'partial_rebase_scope_could_not_be_proven'

            if blocked_reason:
                candidate_review['status'] = 'not_proposed'
                candidate_review['reason'] = blocked_reason
            else:
                requested_class = str(scope_fields.get('requested_rebase_class') or '').strip()
                evidence_refs = [
                    *[
                        str(item).strip()
                        for item in (closure_evidence.get('evidence_refs') or [])
                        if str(item).strip()
                    ],
                    f"runtime_candidate:{diff.get('candidate_graph_digest')}",
                ]
                graph['redraw_scope_evidence'] = _compact_payload(
                    {
                        'kind': 'ollmo.redraw_scope_evidence',
                        'status': 'repair_required',
                        'recommended_scope': requested_class,
                        'reason': 'current Closure requests structural rebuild and a concrete backend graph candidate exists',
                        'evidence_refs': evidence_refs,
                        'scope_root_ids': scope_fields.get('scope_root_ids'),
                    }
                )
                runtime['request_phase_graph'] = graph
                runtime['developer_diagnostics'] = developer_diagnostics
                updated['runtime'] = runtime
                updated = self._attach_redraw_scope_ladder_review(updated)
                runtime = (
                    dict(updated.get('runtime') or {})
                    if isinstance(updated.get('runtime'), dict)
                    else {}
                )
                graph = (
                    dict(runtime.get('request_phase_graph') or {})
                    if isinstance(runtime.get('request_phase_graph'), dict)
                    else graph
                )
                developer_diagnostics = (
                    dict(runtime.get('developer_diagnostics') or {})
                    if isinstance(runtime.get('developer_diagnostics'), dict)
                    else developer_diagnostics
                )
                selected_review = (
                    dict(graph.get('redraw_scope_ladder_review') or {})
                    if isinstance(graph.get('redraw_scope_ladder_review'), dict)
                    else {}
                )
                selected_scope = str(selected_review.get('selected_scope') or '').strip().lower()
                if selected_scope != requested_class:
                    candidate_review['status'] = 'not_proposed'
                    candidate_review['reason'] = 'runtime_scope_review_did_not_select_rebase'
                    candidate_review['selected_scope'] = selected_scope
                else:
                    prompt_intent = (
                        dict(graph.get('prompt_intent') or {})
                        if isinstance(graph.get('prompt_intent'), dict)
                        else {}
                    )
                    proposal = build_graph_rebase_proposal(
                        request_phase_graph=graph,
                        candidate_graph=candidate_graph,
                        target_response_id=str(
                            updated.get('response_id')
                            or updated.get('id')
                            or graph.get('response_id')
                            or ''
                        ).strip(),
                        target_frame_id=str(
                            graph.get('frame_id')
                            or (updated.get('response_frame') or {}).get('frame_id')
                            if isinstance(updated.get('response_frame'), dict)
                            else graph.get('frame_id')
                            or ''
                        ).strip(),
                        source='runtime_closure_review',
                        reason='Current Closure requests structural graph rebuild after smaller scopes were exhausted.',
                        intent_anchor=prompt_intent,
                        evidence_refs=evidence_refs,
                        candidate_origin=str(
                            candidate_context.get('candidate_origin')
                            or 'response_time_request_phase_graph'
                        ).strip(),
                        requested_rebase_class=requested_class,
                        scope_root_ids=scope_fields.get('scope_root_ids') or [],
                        scope_phase_ids=scope_fields.get('scope_phase_ids') or [],
                        scope_branch_ids=scope_fields.get('scope_branch_ids') or [],
                        preserve_outside_scope=scope_fields.get('preserve_outside_scope'),
                        redraw_scope_review_ref=str(selected_review.get('review_id') or '').strip(),
                        root_prompt=guarded_root_prompt,
                    )
                    proposals.append(proposal)
                    candidate_review['status'] = 'proposed_for_runtime_validation'
                    candidate_review['requested_rebase_class'] = requested_class
                    candidate_review['proposal_id'] = proposal.get('proposal_id')
                    candidate_review['redraw_scope_review_ref'] = selected_review.get('review_id')

        proposals = self._dedupe_graph_repair_records(
            proposals,
            'proposal_id',
            'id',
        )
        runtime_gate_reasons: list[str] = []
        current_late_fill_status = _runtime_rebase_late_fill_status(updated)
        if current_late_fill_status:
            runtime_gate_reasons.append('active_late_fill_must_settle')
        current_scope_review = (
            dict(graph.get('redraw_scope_ladder_review') or {})
            if isinstance(graph.get('redraw_scope_ladder_review'), dict)
            else {}
        )
        current_smaller_scope = _runtime_rebase_smaller_scope(current_scope_review)
        if current_smaller_scope:
            runtime_gate_reasons.append('smaller_redraw_scope_precedes_rebase')
        reviews = [
            validate_graph_rebase_proposal(
                proposal,
                request_phase_graph=graph,
                closure_review=closure_review,
                accepted_learning_hints=accepted_learning_hints,
                runtime_gate_reasons=runtime_gate_reasons,
                root_prompt=guarded_root_prompt,
            )
            for proposal in proposals
            if isinstance(proposal, dict)
        ]
        existing_reviews = [
            dict(item)
            for item in (graph.get('graph_rebase_reviews') or [])
            if isinstance(item, dict)
        ]
        current_review_proposal_ids = {
            str(item.get('proposal_id') or '').strip()
            for item in reviews
            if str(item.get('proposal_id') or '').strip()
        }
        retained_existing_reviews = [
            item
            for item in existing_reviews
            if str(item.get('proposal_id') or '').strip() not in current_review_proposal_ids
        ]
        graph['graph_rebase_proposals'] = self._dedupe_graph_repair_records(
            proposals,
            'proposal_id',
            'id',
        )
        graph['graph_rebase_reviews'] = self._dedupe_graph_repair_records(
            [*reviews, *retained_existing_reviews],
            'proposal_id',
            'review_id',
            'id',
        )
        runtime['request_phase_graph'] = graph

        developer_diagnostics['runtime_graph_rebase_proposals'] = proposals
        developer_diagnostics['runtime_graph_rebase_reviews'] = reviews
        if candidate_review:
            candidate_proposal_id = str(candidate_review.get('proposal_id') or '').strip()
            current_candidate_validation = next(
                (
                    item
                    for item in reviews
                    if candidate_proposal_id
                    and str(item.get('proposal_id') or '').strip() == candidate_proposal_id
                ),
                {},
            )
            validation_status = str(current_candidate_validation.get('status') or '').strip().lower()
            if validation_status == 'accepted':
                candidate_review['status'] = 'validated_by_runtime_review'
                candidate_review['validation_status'] = validation_status
            elif current_candidate_validation:
                blocked_reasons = [
                    str(item).strip()
                    for item in (current_candidate_validation.get('blocked_reasons') or [])
                    if str(item).strip()
                ]
                candidate_review['status'] = 'blocked_by_runtime_validation'
                candidate_review['validation_status'] = validation_status or 'blocked'
                candidate_review['reason'] = (
                    blocked_reasons[0]
                    if blocked_reasons
                    else 'runtime_graph_rebase_validation_did_not_accept_candidate'
                )
            developer_diagnostics['runtime_graph_rebase_candidate_review'] = candidate_review
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def review_terminal_graph_rebase_after_late_fill(
        self,
        response_payload: dict[str, Any],
        *,
        request_payload: Optional[dict[str, Any]] = None,
        route_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Re-derive a shadow candidate after Late Fill settles, without replacing the parent graph."""

        if not isinstance(response_payload, dict):
            return response_payload
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            return response_payload

        extract_responses_prompt = self._hook('extract_responses_prompt')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')
        request_info = request_payload if isinstance(request_payload, dict) else {}
        route_info = copy.deepcopy(route_payload) if isinstance(route_payload, dict) else {}
        route_runtime = (
            dict(route_info.get('route_runtime') or {})
            if isinstance(route_info.get('route_runtime'), dict)
            else {}
        )
        route_runtime['request_phase_graph'] = graph
        route_info['route_runtime'] = route_runtime
        response_for_graph = dict(updated)
        output_text = str(updated.get('output_text') or '').strip()
        if output_text:
            response_for_graph['output_text'] = output_text
        candidate_graph = build_request_phase_graph(
            str(extract_responses_prompt(request_info) or output_text or '').strip(),
            intent_prompt=extract_responses_current_turn_prompt(request_info),
            request_payload=request_info,
            route_payload=route_info,
            response_payload=response_for_graph,
        )

        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics.pop(_RESPONSE_TIME_GRAPH_REBASE_CANDIDATE_KEY, None)
        developer_diagnostics.pop('runtime_graph_rebase_candidate_review', None)
        candidate_diff = build_graph_rebase_diff(graph, candidate_graph)
        terminal_candidate_review: dict[str, Any] = {}
        if int(candidate_diff.get('meaningful_change_count') or 0) > 0:
            developer_diagnostics[_RESPONSE_TIME_GRAPH_REBASE_CANDIDATE_KEY] = {
                'kind': 'ollmo.runtime_graph_rebase_candidate',
                'status': 'rederived_from_terminal_materialization_truth',
                'candidate_origin': 'terminal_response_time_request_phase_graph',
                'base_graph_digest': candidate_diff.get('base_graph_digest'),
                'candidate_graph_digest': candidate_diff.get('candidate_graph_digest'),
                'diff_summary': _runtime_rebase_diff_summary(candidate_diff),
                'candidate_graph': copy.deepcopy(candidate_graph),
            }
        else:
            terminal_candidate_review = {
                'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                'status': 'not_proposed',
                'reason': 'terminal_candidate_has_no_meaningful_change',
                'base_graph_digest': candidate_diff.get('base_graph_digest'),
                'candidate_graph_digest': candidate_diff.get('candidate_graph_digest'),
                'runtime_effect': 'none',
            }
        runtime['request_phase_graph'] = graph
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        updated = self._attach_redraw_scope_ladder_review(updated)
        terminal_root_prompt = str(
            extract_responses_current_turn_prompt(request_info)
            or extract_responses_prompt(request_info)
            or ''
        ).strip()
        updated = self._attach_runtime_graph_rebase_evidence(
            updated,
            root_prompt=terminal_root_prompt,
        )
        updated = self._attach_graph_rebase_lifecycle(updated)
        if terminal_candidate_review:
            terminal_runtime = (
                dict(updated.get('runtime') or {})
                if isinstance(updated.get('runtime'), dict)
                else {}
            )
            terminal_diagnostics = (
                dict(terminal_runtime.get('developer_diagnostics') or {})
                if isinstance(terminal_runtime.get('developer_diagnostics'), dict)
                else {}
            )
            terminal_diagnostics['runtime_graph_rebase_candidate_review'] = (
                terminal_candidate_review
            )
            terminal_runtime['developer_diagnostics'] = terminal_diagnostics
            updated['runtime'] = terminal_runtime
        return updated

    def _attach_graph_rebase_lifecycle(
        self,
        response_payload: dict[str, Any],
        *,
        graph_rebase_autonomy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Attach staged/applied graph rebase lifecycle truth according to autonomy."""

        if not isinstance(response_payload, dict):
            return response_payload
        updated = dict(response_payload)
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            return response_payload

        autonomy_description = (
            describe_graph_rebase_autonomy(
                graph_rebase_autonomy,
                source='explicit_override',
                configured=True,
            )
            if graph_rebase_autonomy is not None
            else describe_graph_rebase_autonomy_from_env()
        )
        autonomy_level = normalize_graph_rebase_autonomy(autonomy_description.get('autonomy_level'))
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics['graph_rebase_autonomy'] = {
            **autonomy_description,
            'autonomy_level': autonomy_level,
            'normalized': autonomy_level,
        }
        developer_diagnostics['enforced_policy'] = describe_enforced_policy_from_env()
        if autonomy_level == 'off':
            developer_diagnostics['graph_rebase_lifecycle'] = graph.get('graph_rebase_lifecycle') or []
            developer_diagnostics['graph_rebase_enforced_policy_reviews'] = [
                item.get('enforced_policy_review')
                for item in graph.get('graph_rebase_lifecycle') or []
                if isinstance(item, dict) and isinstance(item.get('enforced_policy_review'), dict)
            ]
            developer_diagnostics['staged_graph_rebases'] = graph.get('staged_graph_rebases') or []
            developer_diagnostics['successor_rebase_requests'] = graph.get('successor_rebase_requests') or []
            runtime['developer_diagnostics'] = developer_diagnostics
            updated['runtime'] = runtime
            return updated

        raw_reviews = [
            dict(item)
            for item in (graph.get('graph_rebase_reviews') or [])
            if isinstance(item, dict)
        ]
        runtime_gate_reasons: list[str] = []
        current_late_fill_status = _runtime_rebase_late_fill_status(updated)
        if current_late_fill_status:
            runtime_gate_reasons.append('active_late_fill_must_settle')
        current_scope_review = (
            dict(graph.get('redraw_scope_ladder_review') or {})
            if isinstance(graph.get('redraw_scope_ladder_review'), dict)
            else {}
        )
        if _runtime_rebase_smaller_scope(current_scope_review):
            runtime_gate_reasons.append('smaller_redraw_scope_precedes_rebase')
        closure_review = (
            dict(runtime.get('graph_closure_review') or {})
            if isinstance(runtime.get('graph_closure_review'), dict)
            else {}
        )
        accepted_learning_hints = (
            dict(runtime.get('accepted_learning_hints') or {})
            if isinstance(runtime.get('accepted_learning_hints'), dict)
            else {}
        )
        reviews: list[dict[str, Any]] = []
        for raw_review in raw_reviews:
            source_proposal = (
                raw_review.get('source_proposal')
                if isinstance(raw_review.get('source_proposal'), dict)
                else {}
            )
            if source_proposal:
                proof = (
                    raw_review.get('preservation_proof')
                    if isinstance(raw_review.get('preservation_proof'), dict)
                    else {}
                )
                reviews.append(
                    validate_graph_rebase_proposal(
                        source_proposal,
                        request_phase_graph=graph,
                        closure_review=closure_review,
                        artifact_payload=(
                            proof.get('artifact_registry_summary')
                            if isinstance(proof.get('artifact_registry_summary'), dict)
                            else None
                        ),
                        accepted_learning_hints=accepted_learning_hints,
                        intent_lens_review=(
                            proof.get('intent_lens_review_summary')
                            if isinstance(proof.get('intent_lens_review_summary'), dict)
                            else None
                        ),
                        runtime_gate_reasons=runtime_gate_reasons,
                        root_prompt=self._graph_rebase_root_prompt(updated),
                    )
                )
                continue
            blocked_review = dict(raw_review)
            blocked_review['status'] = 'blocked'
            blocked_reasons = [
                str(item).strip()
                for item in (blocked_review.get('blocked_reasons') or [])
                if str(item).strip()
            ]
            for reason in [
                'rebase_review_source_proposal_missing',
                *runtime_gate_reasons,
            ]:
                if reason not in blocked_reasons:
                    blocked_reasons.append(reason)
            blocked_review['blocked_reasons'] = blocked_reasons
            reviews.append(blocked_review)
        reviews = self._dedupe_graph_repair_records(
            reviews,
            'proposal_id',
            'review_id',
            'id',
        )
        graph['graph_rebase_reviews'] = reviews
        working_graph = graph
        lifecycle_results: list[dict[str, Any]] = []
        for review in reviews:
            lifecycle = build_graph_rebase_lifecycle(
                request_phase_graph=working_graph,
                rebase_review=review,
                autonomy_level=autonomy_level,
            )
            application = apply_validated_graph_rebase(
                working_graph,
                lifecycle,
                autonomy_level=autonomy_level,
                runtime_gate_reasons=runtime_gate_reasons,
                root_prompt=self._graph_rebase_root_prompt(updated),
            )
            working_graph = (
                dict(application.get('graph') or {})
                if isinstance(application.get('graph'), dict)
                else working_graph
            )
            lifecycle_results.append(application)

        runtime['request_phase_graph'] = working_graph
        developer_diagnostics['graph_rebase_lifecycle'] = working_graph.get('graph_rebase_lifecycle') or []
        developer_diagnostics['graph_rebase_lifecycle_results'] = lifecycle_results
        developer_diagnostics['graph_rebase_enforced_policy_reviews'] = [
            item.get('enforced_policy_review')
            for item in working_graph.get('graph_rebase_lifecycle') or []
            if isinstance(item, dict) and isinstance(item.get('enforced_policy_review'), dict)
        ]
        developer_diagnostics['staged_graph_rebases'] = working_graph.get('staged_graph_rebases') or []
        developer_diagnostics['successor_rebase_requests'] = working_graph.get('successor_rebase_requests') or []
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def _attach_repair_gap_to_request_phase_graph(
        self,
        response_payload: dict[str, Any],
        repair_gap: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(repair_gap, dict) or not repair_gap:
            return response_payload
        pending_branches = [
            dict(item)
            for item in (repair_gap.get('pending_branches') or [])
            if isinstance(item, dict)
        ]
        if not pending_branches:
            return response_payload
        updated = dict(response_payload or {})
        runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
        graph = (
            dict(runtime.get('request_phase_graph') or {})
            if isinstance(runtime.get('request_phase_graph'), dict)
            else {}
        )
        if not graph:
            return response_payload
        repair_loop = (
            repair_gap.get('repair_loop')
            if isinstance(repair_gap.get('repair_loop'), dict)
            else {}
        )
        reconsideration_rebuild = (
            repair_gap.get('reconsideration_rebuild')
            if isinstance(repair_gap.get('reconsideration_rebuild'), dict)
            else {}
        )
        promotion_status = ''
        if str(repair_loop.get('status') or '').strip().lower() == 'promoted':
            promotion_status = 'promoted'
        elif str(reconsideration_rebuild.get('status') or '').strip().lower() == 'promoted':
            promotion_status = 'promoted'
        graph_repair_proposal = build_graph_repair_proposal_from_repair_gap(
            request_phase_graph=graph,
            repair_gap=repair_gap,
        )
        graph_repair_review = validate_graph_repair_proposal(
            graph_repair_proposal or {},
            request_phase_graph=graph,
            closure_review={
                'status': 'repair_required',
                'ghost_repair_feedback': repair_gap.get('ghost_repair_feedback'),
                'repair_loop': repair_loop,
                'pending_branches': pending_branches,
            },
            candidate_graph={},
            promotion_review={'status': promotion_status} if promotion_status else {},
        )
        if str(graph_repair_review.get('status') or '').strip().lower() != 'accepted':
            graph_refinements = [
                dict(item)
                for item in (graph.get('graph_refinements') or [])
                if isinstance(item, dict)
            ]
            graph_refinements.append(
                _compact_payload(
                    {
                        'kind': 'graph_repair_proposal_review',
                        'source': 'closure_repair_contract',
                        'status': graph_repair_review.get('status'),
                        'proposal_id': graph_repair_review.get('proposal_id'),
                        'review_id': graph_repair_review.get('review_id'),
                        'reasons': graph_repair_review.get('reasons'),
                    }
                )
            )
            graph['graph_refinements'] = graph_refinements
            graph['graph_repair_reviews'] = [
                *[
                    dict(item)
                    for item in (graph.get('graph_repair_reviews') or [])
                    if isinstance(item, dict)
                ],
                graph_repair_review,
            ]
            runtime['request_phase_graph'] = graph
            developer_diagnostics = (
                dict(runtime.get('developer_diagnostics') or {})
                if isinstance(runtime.get('developer_diagnostics'), dict)
                else {}
            )
            developer_diagnostics['graph_repair_proposal'] = graph_repair_proposal
            developer_diagnostics['graph_repair_proposal_review'] = graph_repair_review
            runtime['developer_diagnostics'] = developer_diagnostics
            updated['runtime'] = runtime
            return updated

        def _branch_id(value: dict[str, Any]) -> str:
            return str(value.get('branch_id') or value.get('phase_id') or '').strip()

        def _phase_id(value: dict[str, Any]) -> str:
            return str(value.get('phase_id') or value.get('branch_id') or '').strip()

        existing_branch_ids = {
            _branch_id(item)
            for item in (graph.get('downstream_branches') or [])
            if isinstance(item, dict) and _branch_id(item)
        }
        downstream_branches = [
            dict(item)
            for item in (graph.get('downstream_branches') or [])
            if isinstance(item, dict)
        ]
        phases = [
            dict(item)
            for item in (graph.get('phases') or [])
            if isinstance(item, dict)
        ]
        existing_phase_ids = {
            str(item.get('phase_id') or '').strip()
            for item in phases
            if str(item.get('phase_id') or '').strip()
        }
        output_obligations = [
            dict(item)
            for item in (graph.get('output_obligations') or [])
            if isinstance(item, dict)
        ]
        existing_obligation_ids = {
            str(item.get('obligation_id') or '').strip()
            for item in output_obligations
            if str(item.get('obligation_id') or '').strip()
        }

        patched_count = 0
        for branch in pending_branches:
            branch_id = _branch_id(branch)
            phase_id = _phase_id(branch) or branch_id
            capability = str(branch.get('capability') or '').strip()
            output_type = str(branch.get('output_type') or self._repair_output_type_for_capability(capability) or '').strip()
            if not branch_id or not capability:
                continue
            graph_branch = _compact_payload(
                {
                    **branch,
                    'branch_id': branch_id,
                    'phase_id': phase_id,
                    'capability': capability,
                    'output_type': output_type or None,
                    'status': str(branch.get('status') or '').strip() or 'pending',
                    'source': 'closure_repair_contract',
                }
            )
            if branch_id not in existing_branch_ids:
                downstream_branches.append(graph_branch)
                existing_branch_ids.add(branch_id)
                patched_count += 1
            if phase_id and phase_id not in existing_phase_ids:
                phases.append(
                    _compact_payload(
                        {
                            'phase_id': phase_id,
                            'branch_id': branch_id,
                            'capability': capability,
                            'output_type': output_type or None,
                            'status': 'pending',
                            'depends_on': branch.get('depends_on'),
                            'required': True,
                            'source': 'closure_repair_contract',
                            'repair_action': branch.get('repair_action'),
                            'repair_contract': branch.get('repair_contract'),
                            'repair_contract_id': branch.get('repair_contract_id'),
                            'repair_execution_policy': branch.get('repair_execution_policy'),
                            'contract_state': branch.get('contract_state') or 'promoted',
                            'promotion_source': branch.get('promotion_source') or 'graph_closure_review',
                            'semantic_intent': branch.get('semantic_intent'),
                            'objective': branch.get('objective'),
                            'deliverable': branch.get('deliverable'),
                            'content_payload': branch.get('content_payload'),
                            'content_payload_source': branch.get('content_payload_source'),
                            'stage_direction': branch.get('stage_direction'),
                            'phase_summary': branch.get('phase_summary'),
                            'input_refs': branch.get('input_refs'),
                            'review_criteria': branch.get('review_criteria'),
                            'semantic_review_criteria': branch.get('semantic_review_criteria'),
                            'semantic_review_verdict': branch.get('semantic_review_verdict'),
                            'semantic_review_verdict_status': branch.get('semantic_review_verdict_status'),
                            'semantic_review_recommended_transition': branch.get('semantic_review_recommended_transition'),
                            'branch_semantic_review': branch.get('branch_semantic_review'),
                            'branch_semantic_review_branch_id': branch.get('branch_semantic_review_branch_id'),
                            'branch_semantic_review_phase_id': branch.get('branch_semantic_review_phase_id'),
                            'branch_semantic_review_status': branch.get('branch_semantic_review_status'),
                            'branch_semantic_review_reason': branch.get('branch_semantic_review_reason'),
                            'branch_semantic_review_source_branch_id': branch.get('branch_semantic_review_source_branch_id'),
                            'branch_semantic_review_source_phase_id': branch.get('branch_semantic_review_source_phase_id'),
                            'global_semantic_closure_review': branch.get('global_semantic_closure_review'),
                            'global_semantic_closure_proposal': branch.get('global_semantic_closure_proposal'),
                        }
                    )
                )
                existing_phase_ids.add(phase_id)
            obligation_id = str(branch.get('obligation_id') or '').strip() or f'obligation-{phase_id}'
            if obligation_id and obligation_id not in existing_obligation_ids and output_type:
                output_obligations.append(
                    _compact_payload(
                        {
                            'obligation_id': obligation_id,
                            'phase_id': phase_id,
                            'branch_id': branch_id,
                            'capability': capability,
                            'output_type': output_type,
                            'status': 'pending',
                            'required': True,
                            'source': 'closure_repair_contract',
                            'repair_action': branch.get('repair_action'),
                            'repair_contract_id': branch.get('repair_contract_id'),
                            'repair_execution_policy': branch.get('repair_execution_policy'),
                            'depends_on': branch.get('depends_on'),
                            'input_refs': branch.get('input_refs'),
                            'review_criteria': branch.get('review_criteria'),
                            'semantic_review_criteria': branch.get('semantic_review_criteria'),
                            'semantic_review_verdict': branch.get('semantic_review_verdict'),
                            'semantic_review_verdict_status': branch.get('semantic_review_verdict_status'),
                            'semantic_review_recommended_transition': branch.get('semantic_review_recommended_transition'),
                            'branch_semantic_review': branch.get('branch_semantic_review'),
                            'branch_semantic_review_branch_id': branch.get('branch_semantic_review_branch_id'),
                            'branch_semantic_review_phase_id': branch.get('branch_semantic_review_phase_id'),
                            'branch_semantic_review_status': branch.get('branch_semantic_review_status'),
                            'branch_semantic_review_reason': branch.get('branch_semantic_review_reason'),
                            'branch_semantic_review_source_branch_id': branch.get('branch_semantic_review_source_branch_id'),
                            'branch_semantic_review_source_phase_id': branch.get('branch_semantic_review_source_phase_id'),
                            'content_payload_source': branch.get('content_payload_source'),
                            'stage_direction': branch.get('stage_direction'),
                            'global_semantic_closure_status': branch.get('global_semantic_closure_status'),
                        }
                    )
                )
                existing_obligation_ids.add(obligation_id)

        if not patched_count:
            return response_payload
        graph['downstream_branches'] = downstream_branches
        graph['downstream_branch_ids'] = [
            _branch_id(item)
            for item in downstream_branches
            if _branch_id(item)
        ]
        graph['downstream_capabilities'] = list(
            dict.fromkeys(
                str(item.get('capability') or '').strip()
                for item in downstream_branches
                if str(item.get('capability') or '').strip()
            )
        )
        graph['phases'] = phases
        graph['output_obligations'] = output_obligations
        graph['continuation_required'] = True
        graph_refinements = [
            dict(item)
            for item in (graph.get('graph_refinements') or [])
            if isinstance(item, dict)
        ]
        graph_refinements.append(
            {
                'kind': 'closure_repair_graph_patch',
                'source': 'closure_repair_contract',
                'status': 'promoted',
                'graph_repair_proposal_id': graph_repair_review.get('proposal_id'),
                'graph_repair_review_id': graph_repair_review.get('review_id'),
                'graph_repair_review_status': graph_repair_review.get('status'),
                'patched_branch_count': patched_count,
                'repair_action': repair_gap.get('repair_action'),
                'repair_actions': repair_gap.get('repair_actions'),
            }
        )
        graph['graph_refinements'] = graph_refinements
        graph['graph_repair_reviews'] = [
            *[
                dict(item)
                for item in (graph.get('graph_repair_reviews') or [])
                if isinstance(item, dict)
            ],
            graph_repair_review,
        ]
        runtime['request_phase_graph'] = graph
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics') or {})
            if isinstance(runtime.get('developer_diagnostics'), dict)
            else {}
        )
        developer_diagnostics['graph_repair_proposal'] = graph_repair_proposal
        developer_diagnostics['graph_repair_proposal_review'] = graph_repair_review
        developer_diagnostics['closure_repair_graph_patch'] = graph_refinements[-1]
        runtime['developer_diagnostics'] = developer_diagnostics
        updated['runtime'] = runtime
        return updated

    def attach_pre_freeze_closure_review(
        self,
        response_payload: dict[str, Any],
        *,
        output_text: str,
        route_payload: Optional[dict[str, Any]] = None,
        request_payload: Optional[dict[str, Any]] = None,
        artifact_gap: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        build_pre_freeze_closure_review_gap = self._hook('build_pre_freeze_closure_review_gap')
        build_late_fill_state = self._hook('build_late_fill_state')
        attach_late_fill_state = self._hook('attach_late_fill_state')

        def _clean_capability(value: Any) -> str:
            return re.sub(r'[^a-z0-9_]+', '_', str(value or '').strip().lower()).strip('_')

        def _has_direct_dependency_evidence() -> bool:
            for source in (response_payload, request_payload or {}, route_payload or {}):
                if not isinstance(source, dict):
                    continue
                for key in ('input_artifacts', 'reference_artifacts', 'selected_reference_artifacts'):
                    if isinstance(source.get(key), list) and source.get(key):
                        return True
                    if isinstance(source.get(key), dict) and source.get(key):
                        return True
                for key in (
                    'file_path',
                    'artifact_path',
                    'artifact_ref',
                    'route_artifact_path',
                    'route_artifact_ref',
                    'saved_image_path',
                    'saved_audio_path',
                    'saved_text_path',
                ):
                    if str(source.get(key) or '').strip():
                        return True
            return False

        def _gap_is_satisfied_by_direct_target(gap: Any) -> bool:
            if not isinstance(gap, dict) or not gap:
                return False
            target_capability = _clean_capability(
                response_payload.get('capability')
                or (route_payload or {}).get('capability')
                or (request_payload or {}).get('capability')
            )
            if not target_capability or target_capability == self.capability_chat:
                return False
            if not str(response_payload.get('output_text') or '').strip():
                return False
            if not _has_direct_dependency_evidence():
                return False
            pending_branches = [
                item for item in (gap.get('pending_branches') or []) if isinstance(item, dict)
            ]
            capability_tokens = {
                _clean_capability(gap.get('expected_capability')),
                _clean_capability(gap.get('active_capability')),
            }
            capability_tokens.update(
                _clean_capability(item)
                for item in (gap.get('pending_capabilities') or [])
                if str(item or '').strip()
            )
            capability_tokens.update(
                _clean_capability(branch.get('capability'))
                for branch in pending_branches
                if str(branch.get('capability') or '').strip()
            )
            capability_tokens.discard('')
            if not capability_tokens:
                return False
            return capability_tokens == {target_capability}

        def _late_fill_extra_from_closure_review(review: Any) -> dict[str, Any]:
            if not isinstance(review, dict):
                return {}
            extra: dict[str, Any] = {}
            if review.get('surface_state') not in (None, '', [], {}):
                extra['surface_state'] = review.get('surface_state')
            if review.get('global_semantic_closure_review') not in (None, '', [], {}):
                extra['global_semantic_closure_review'] = review.get('global_semantic_closure_review')
            decision_contract_review = review.get('decision_contract_review')
            if isinstance(decision_contract_review, dict):
                for key in (
                    'active_reconsideration_review',
                    'semantic_quality_review',
                    'recursive_cycle_review',
                    'aspiration_review',
                    'commitment_review',
                    'semantic_decision_review',
                    'controlled_attention_review',
                ):
                    if decision_contract_review.get(key) not in (None, '', [], {}):
                        extra[key] = decision_contract_review.get(key)
            return extra

        def _project_graph_patch_reconciliation_to_late_fill(
            state: Any,
            reconciliation: Any,
            *,
            extra: Optional[dict[str, Any]] = None,
        ) -> dict[str, Any]:
            projected = dict(state) if isinstance(state, dict) else {}
            if not isinstance(reconciliation, dict) or not reconciliation:
                if isinstance(extra, dict):
                    projected.update(extra)
                return projected
            projected['graph_patch_reconciliation'] = dict(reconciliation)
            existing_blocked = [
                dict(item)
                for item in (projected.get('blocked_branches') or [])
                if isinstance(item, dict)
            ]
            blocked_by_id = {
                str(item.get('branch_id') or item.get('phase_id') or '').strip(): index
                for index, item in enumerate(existing_blocked)
                if str(item.get('branch_id') or item.get('phase_id') or '').strip()
            }
            for item in reconciliation.get('unscheduled_branches') or []:
                if not isinstance(item, dict):
                    continue
                branch_id = str(item.get('branch_id') or item.get('phase_id') or '').strip()
                if not branch_id:
                    continue
                blocked_record = {**dict(item), 'status': 'blocked'}
                if branch_id in blocked_by_id:
                    existing_blocked[blocked_by_id[branch_id]] = blocked_record
                else:
                    blocked_by_id[branch_id] = len(existing_blocked)
                    existing_blocked.append(blocked_record)
            if existing_blocked:
                projected['blocked_branches'] = existing_blocked
            if isinstance(extra, dict):
                projected.update(extra)
            return projected

        response_payload, _phase_graph = self._attach_fluid_request_phase_graph(
            response_payload,
            output_text=output_text,
            route_payload=route_payload,
            request_payload=request_payload,
        )
        if _truth_guard_blocks_materialization(response_payload):
            runtime = (
                dict(response_payload.get('runtime') or {})
                if isinstance(response_payload.get('runtime'), dict)
                else {}
            )
            truth_guard = runtime.get('truth_guard') if isinstance(runtime.get('truth_guard'), dict) else {}
            truth_guard_status = str(truth_guard.get('status') or '').strip().lower()
            phase_acceptance = (
                runtime.get('phase_output_acceptance')
                if isinstance(runtime.get('phase_output_acceptance'), dict)
                else {}
            )
            control_payload_rejected = truth_guard_status == 'repair_required'
            check_kind = (
                'prepared_substantive_text_accepted'
                if control_payload_rejected
                else 'truth_guard'
            )
            evidence = (
                'control_envelope_not_speakable'
                if control_payload_rejected
                else 'clarification_required'
            )
            reason = str(truth_guard.get('reason') or '').strip() or (
                'current phase returned internal control data instead of substantive content'
                if control_payload_rejected
                else 'current turn needs a concrete source before materialization'
            )
            response_payload['lifecycle_state'] = 'repair_needed'
            response_payload = self._attach_graph_closure_review_diagnostics(
                response_payload,
                _compact_payload(
                    {
                        'kind': 'ollmo.graph_closure_review',
                        'status': 'blocked',
                        'surface_state': {
                            'state': 'repair_needed',
                            'reason': reason,
                        },
                        'checks': [
                            {
                                'check_kind': check_kind,
                                'status': 'blocked',
                                'evidence': evidence,
                                'reason': reason,
                                'repair_action': 'repair_branch_contract',
                                'recovery_action': 'repair_branch_contract',
                                **(
                                    {'acceptance_status': 'repair_required'}
                                    if control_payload_rejected
                                    else {}
                                ),
                                **(
                                    {'attempt_count': phase_acceptance.get('attempt_count')}
                                    if phase_acceptance.get('attempt_count') not in (None, '')
                                    else {}
                                ),
                            }
                        ],
                    }
                ),
            )
            response_payload = self._attach_redraw_scope_ladder_review(response_payload)
            response_payload = self._attach_runtime_graph_repair_evidence(response_payload)
            response_payload = self._attach_graph_patch_lifecycle(response_payload)
            response_payload = self._attach_runtime_graph_rebase_evidence(
                response_payload,
                root_prompt=self._graph_rebase_root_prompt(request_payload),
            )
            response_payload = self._attach_graph_rebase_lifecycle(response_payload)
            return response_payload, None
        closure_gap = build_pre_freeze_closure_review_gap(
            output_text,
            route_payload=route_payload,
            request_payload=request_payload,
            artifact_payload=response_payload,
            artifact_gap=artifact_gap,
        )
        build_graph_closure_review = self._hook('build_graph_closure_review')
        graph_closure_review = build_graph_closure_review(
            output_text,
            route_payload=route_payload,
            request_payload=request_payload,
            artifact_payload=response_payload,
            artifact_gap=closure_gap if isinstance(closure_gap, dict) else artifact_gap,
        )
        repair_gap = self._ghost_repair_feedback_gap(
            graph_closure_review,
            prior_gap=closure_gap if isinstance(closure_gap, dict) else None,
        )
        if _gap_is_satisfied_by_direct_target(repair_gap or closure_gap):
            graph_closure_review = dict(graph_closure_review)
            graph_closure_review['direct_target_fulfillment'] = {
                'status': 'satisfied',
                'capability': str(response_payload.get('capability') or '').strip(),
                'reason': 'primary response already fulfilled the same artifact-backed capability',
            }
            graph_closure_review.pop('ghost_repair_feedback', None)
            graph_closure_review.pop('repair_gap_code', None)
            graph_closure_review.pop('repair_gap_trigger', None)
            if str(graph_closure_review.get('status') or '').strip().lower() in {'blocked', 'pending'}:
                graph_closure_review['status'] = 'completed'
            surface_state = (
                dict(graph_closure_review.get('surface_state') or {})
                if isinstance(graph_closure_review.get('surface_state'), dict)
                else {}
            )
            if str(surface_state.get('state') or '').strip().lower() in {'blocked', 'repair_needed', 'pending'}:
                surface_state['state'] = 'completed'
                surface_state['reason'] = 'direct target response fulfilled the current artifact-backed request'
                graph_closure_review['surface_state'] = surface_state
            closure_gap = None
            repair_gap = None
        if repair_gap:
            response_payload = self._attach_repair_gap_to_request_phase_graph(response_payload, repair_gap)
            closure_gap = repair_gap
            graph_closure_review = dict(graph_closure_review)
            graph_closure_review['repair_gap_code'] = str(repair_gap.get('code') or '').strip() or None
            graph_closure_review['repair_gap_trigger'] = str(repair_gap.get('trigger') or '').strip() or None
            graph_closure_review['repair_pending_branch_count'] = len(repair_gap.get('pending_branches') or [])
        response_payload = self._attach_graph_closure_review_diagnostics(
            response_payload,
            _compact_payload(graph_closure_review),
        )
        prior_late_fill = (
            response_payload.get('late_fill')
            if isinstance(response_payload.get('late_fill'), dict)
            else None
        )
        late_fill_extra = _late_fill_extra_from_closure_review(graph_closure_review)
        if isinstance(closure_gap, dict) and closure_gap:
            pending_late_fill = build_late_fill_state(
                closure_gap,
                status='pending',
                prior_state=prior_late_fill,
                extra=late_fill_extra,
            )
            if isinstance(closure_gap.get('graph_patch_reconciliation'), dict):
                pending_late_fill['graph_patch_reconciliation'] = dict(
                    closure_gap['graph_patch_reconciliation']
                )
            response_payload = attach_late_fill_state(response_payload, pending_late_fill)
        response_payload = self._attach_redraw_scope_ladder_review(response_payload)
        response_payload = self._attach_runtime_graph_repair_evidence(response_payload)
        response_payload = self._attach_graph_patch_lifecycle(response_payload)
        response_payload, closure_gap = self._reconcile_applied_graph_patch_late_fill_gap(
            response_payload,
            closure_gap if isinstance(closure_gap, dict) else None,
        )
        response_runtime = (
            response_payload.get('runtime')
            if isinstance(response_payload.get('runtime'), dict)
            else {}
        )
        response_diagnostics = (
            response_runtime.get('developer_diagnostics')
            if isinstance(response_runtime.get('developer_diagnostics'), dict)
            else {}
        )
        graph_patch_reconciliation = (
            closure_gap.get('graph_patch_reconciliation')
            if isinstance(closure_gap, dict)
            and isinstance(closure_gap.get('graph_patch_reconciliation'), dict)
            else response_diagnostics.get('graph_patch_late_fill_reconciliation')
            if isinstance(response_diagnostics.get('graph_patch_late_fill_reconciliation'), dict)
            else {}
        )
        if (
            str(graph_patch_reconciliation.get('status') or '').strip().lower()
            in {'applied', 'no_executable_branches'}
            and graph_patch_reconciliation.get('patch_ids')
        ):
            graph_closure_review = build_graph_closure_review(
                output_text,
                route_payload=route_payload,
                request_payload=request_payload,
                artifact_payload=response_payload,
                artifact_gap=closure_gap,
            )
            response_payload = self._attach_graph_closure_review_diagnostics(
                response_payload,
                _compact_payload(graph_closure_review),
            )
            late_fill_extra = _late_fill_extra_from_closure_review(graph_closure_review)
        if isinstance(closure_gap, dict) and closure_gap:
            prior_late_fill = (
                response_payload.get('late_fill')
                if isinstance(response_payload.get('late_fill'), dict)
                else prior_late_fill
            )
            pending_late_fill = build_late_fill_state(
                closure_gap,
                status='pending',
                prior_state=prior_late_fill,
                extra=late_fill_extra,
            )
            if isinstance(closure_gap.get('graph_patch_reconciliation'), dict):
                pending_late_fill = _project_graph_patch_reconciliation_to_late_fill(
                    pending_late_fill,
                    closure_gap['graph_patch_reconciliation'],
                    extra=late_fill_extra,
                )
            response_payload = attach_late_fill_state(response_payload, pending_late_fill)
        elif (
            str(graph_patch_reconciliation.get('status') or '').strip().lower()
            == 'no_executable_branches'
        ):
            current_late_fill = (
                response_payload.get('late_fill')
                if isinstance(response_payload.get('late_fill'), dict)
                else {}
            )
            blocked_late_fill = _project_graph_patch_reconciliation_to_late_fill(
                current_late_fill,
                graph_patch_reconciliation,
                extra=late_fill_extra,
            )
            blocked_late_fill['status'] = 'blocked'
            blocked_late_fill['pending_branches'] = []
            blocked_late_fill['pending_capabilities'] = []
            blocked_late_fill['expected_capability'] = None
            blocked_late_fill['active_capability'] = None
            response_payload = attach_late_fill_state(response_payload, blocked_late_fill)
        response_payload = self._attach_redraw_scope_ladder_review(response_payload)
        response_payload = self._attach_runtime_graph_rebase_evidence(
            response_payload,
            root_prompt=self._graph_rebase_root_prompt(request_payload),
        )
        response_payload = self._attach_graph_rebase_lifecycle(response_payload)
        return response_payload, closure_gap if isinstance(closure_gap, dict) and closure_gap else None

    def _direct_batch_aspect_ratios(
        self,
        *,
        prompt_text: str,
        prompt_intent: dict[str, Any],
        infer_payload: dict[str, Any],
        direct_batch_count: int,
    ) -> list[Optional[str]]:
        if direct_batch_count <= 1:
            return []
        explicit_ratio = str(prompt_intent.get('image_aspect_ratio') or '').strip().lower()
        if explicit_ratio:
            return []
        if infer_payload.get('width') not in (None, '') or infer_payload.get('height') not in (None, ''):
            return []
        if not _DIRECT_BATCH_VARIED_ASPECT_RATIO_RE.search(str(prompt_text or '').strip()):
            return []
        ratios: list[Optional[str]] = []
        for index in range(direct_batch_count):
            ratios.append(_DIRECT_BATCH_VARIANT_ASPECT_RATIO_CYCLE[index % len(_DIRECT_BATCH_VARIANT_ASPECT_RATIO_CYCLE)])
        return ratios

    def _build_direct_batch_variant_prompt(
        self,
        *,
        base_prompt: str,
        index: int,
        total: int,
        matching_outputs: bool,
        aspect_ratio: Optional[str],
    ) -> str:
        prompt = str(base_prompt or '').strip()
        if not prompt or total <= 1:
            return prompt
        instructions = [f'Batch variant guidance: this is image {index} of {total}.']
        if matching_outputs:
            instructions.append(
                'Keep it aligned with the same core scene and subject as the other requested images.'
            )
        else:
            instructions.append(
                'Keep the user request intact, but make this image clearly distinct from the other requested images.'
            )
            instructions.append(
                'Vary composition, framing, perspective, lighting, and scene details, and choose a different subject whenever the request leaves that choice open.'
            )
        if aspect_ratio:
            instructions.append(f'Compose this variant for a {aspect_ratio} aspect ratio.')
        return f"{prompt}\n\n[{' '.join(instructions)}]"

    def build_direct_image_batch_items(
        self,
        request_payload: dict[str, Any],
        route_info: Optional[dict[str, Any]],
        infer_payload: dict[str, Any],
        *,
        direct_batch_count: int,
    ) -> list[dict[str, Any]]:
        prompt_text = self._direct_batch_prompt_text(request_payload, route_info, infer_payload)
        base_prompt = str(prompt_text or '').strip()
        if not base_prompt or direct_batch_count <= 1:
            return []
        prompt_intent = self._extract_direct_batch_prompt_intent(route_info)
        matching_outputs = self._direct_batch_requests_matching_outputs(prompt_text)
        aspect_ratios = self._direct_batch_aspect_ratios(
            prompt_text=prompt_text,
            prompt_intent=prompt_intent,
            infer_payload=infer_payload,
            direct_batch_count=direct_batch_count,
        )
        batch_items: list[dict[str, Any]] = []
        for index in range(direct_batch_count):
            aspect_ratio = aspect_ratios[index] if index < len(aspect_ratios) else None
            batch_items.append(
                {
                    'prompt': self._build_direct_batch_variant_prompt(
                        base_prompt=base_prompt,
                        index=index + 1,
                        total=direct_batch_count,
                        matching_outputs=matching_outputs,
                        aspect_ratio=aspect_ratio,
                    ),
                    'width': None,
                    'height': None,
                    'aspect_ratio': aspect_ratio,
                }
            )
        return batch_items

    def build_direct_batch_branch_specs(
        self,
        *,
        batch_items: list[dict[str, Any]],
        capability: str,
        base_request_payload: dict[str, Any],
        base_route_info: Optional[dict[str, Any]],
        base_instance_id: str,
        base_instance: Optional[dict[str, Any]],
        forced_instance_id: Optional[str],
        upload: Any,
    ) -> list[dict[str, Any]]:
        branch_specs: list[dict[str, Any]] = []
        resolved_request_forced_instance_id = str(forced_instance_id or '').strip()
        request_instance_id = (
            str((base_request_payload or {}).get('instance_id') or '').strip()
            if isinstance(base_request_payload, dict)
            else ''
        )
        effective_forced_instance_id = (
            resolved_request_forced_instance_id or request_instance_id
        )
        for index, batch_item in enumerate(batch_items, start=1):
            prompt = str(batch_item.get('prompt') or '').strip()
            if not prompt:
                continue
            branch_request_payload = dict(base_request_payload)
            branch_request_payload['capability'] = capability
            branch_request_payload['prompt'] = prompt
            branch_request_payload, batch_error = self.apply_batch_image_dimensions(
                branch_request_payload,
                batch_item,
            )
            if batch_error:
                raise ValueError(batch_error)
            branch_request_payload.pop('instance_id', None)
            branch_specs.append(
                {
                    'branch_id': f'direct-batch-{index}',
                    'phase_id': f'direct-batch-{index}',
                    'capability': capability,
                    'reservation_group': capability,
                    'batch_index': index,
                    'prompt': prompt,
                    'branch': {
                        'branch_id': f'direct-batch-{index}',
                        'phase_id': f'direct-batch-{index}',
                        'capability': capability,
                        'batch_index': index,
                        'prompt': prompt,
                    },
                    'prepare_args': {
                        'branch': {
                            'branch_id': f'direct-batch-{index}',
                            'phase_id': f'direct-batch-{index}',
                            'capability': capability,
                            'batch_index': index,
                            'prompt': prompt,
                        },
                        'branch_request_payload': branch_request_payload,
                        'base_route_info': dict(base_route_info or {}) if isinstance(base_route_info, dict) else None,
                        'base_instance_id': str(base_instance_id or '').strip(),
                        'base_instance': dict(base_instance or {}) if isinstance(base_instance, dict) else None,
                        'allow_base_instance_reuse': index == 1,
                        'forced_instance_id': effective_forced_instance_id or None,
                        'upload': upload,
                    },
                }
            )
        return branch_specs

    def prepare_direct_materialization_branch_plan(
        self,
        *,
        branch: dict[str, Any],
        branch_request_payload: dict[str, Any],
        base_route_info: Optional[dict[str, Any]],
        base_instance_id: str,
        base_instance: Optional[dict[str, Any]],
        allow_base_instance_reuse: bool = False,
        forced_instance_id: Optional[str],
        upload: Any,
        excluded_instance_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        parse_bool = self._hook('parse_bool')
        resolve_ghost_auto_route = self._hook('resolve_ghost_auto_route')
        resolve_responses_target_instance = self._hook('resolve_responses_target_instance')
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')
        select_backend_request_model = self._hook('select_backend_request_model')
        build_responses_infer_execution_payload = self._hook('build_responses_infer_execution_payload')

        branch_payload = dict(branch_request_payload or {})
        branch_id = str(branch.get('branch_id') or branch.get('phase_id') or '').strip()
        capability = normalize_capability(branch.get('capability') or branch_payload.get('capability')) or ''
        excluded = [
            str(item).strip()
            for item in (excluded_instance_ids or [])
            if str(item).strip()
        ]
        if capability and not str(branch_payload.get('capability') or '').strip():
            branch_payload['capability'] = capability
        route_info: Optional[dict[str, Any]] = None
        instance_id = ''
        instance: Optional[dict[str, Any]] = None
        resolved_capability = capability or None
        resolution_error: Optional[str] = None
        forced_pin = str(forced_instance_id or '').strip()
        branch_pin = str(branch_payload.get('instance_id') or '').strip()
        pinned_instance_id = forced_pin or branch_pin
        if pinned_instance_id and pinned_instance_id in excluded and not forced_pin:
            pinned_instance_id = ''

        if pinned_instance_id:
            instance_id, instance, resolved_capability, resolution_error = resolve_responses_target_instance(
                branch_payload,
                forced_instance_id=pinned_instance_id,
            )
        elif parse_bool(branch_payload.get('ghost_route'), default=False):
            if allow_base_instance_reuse and not excluded and isinstance(base_route_info, dict):
                route_info = dict(base_route_info)
                instance_id = str(route_info.get('instance_id') or base_instance_id or '').strip()
                route_instance = route_info.get('instance')
                if isinstance(route_instance, dict):
                    instance = dict(route_instance)
                elif isinstance(base_instance, dict):
                    instance = dict(base_instance)
            else:
                route_info, resolution_error = resolve_ghost_auto_route(
                    branch_payload,
                    upload=None,
                    excluded_instance_ids=excluded,
                )
                instance_id = str((route_info or {}).get('instance_id') or '').strip()
                route_instance = (route_info or {}).get('instance')
                if isinstance(route_instance, dict):
                    instance = dict(route_instance)
                resolved_capability = normalize_capability((route_info or {}).get('capability')) or resolved_capability
        else:
            if allow_base_instance_reuse and not excluded and str(base_instance_id or '').strip() and isinstance(base_instance, dict):
                instance_id = str(base_instance_id or '').strip()
                instance = dict(base_instance)
            else:
                instance_id, instance, resolved_capability, resolution_error = resolve_responses_target_instance(
                    branch_payload,
                    excluded_instance_ids=excluded,
                )

        if resolution_error:
            raise RuntimeError(resolution_error)
        if not instance_id or not isinstance(instance, dict):
            raise RuntimeError(f"Direct batch branch '{branch_id or capability or 'unknown'}' could not resolve an instance.")

        if not isinstance(route_info, dict):
            route_info = {}
            if isinstance(base_route_info, dict):
                for key in (
                    'route_source',
                    'route_reason',
                    'route_confidence',
                    'route_reuse_last_artifact',
                    'route_artifact_ref',
                    'route_artifact_path',
                    'route_runtime',
                    'request_meta',
                ):
                    value = base_route_info.get(key)
                    if value not in (None, '', [], {}):
                        route_info[key] = dict(value) if isinstance(value, dict) else value
            route_info['route_source'] = str(route_info.get('route_source') or 'responses_batch').strip() or 'responses_batch'
            route_info['route_reason'] = str(route_info.get('route_reason') or 'direct explicit batch item materialization').strip() or 'direct explicit batch item materialization'
        route_info['instance_id'] = instance_id
        route_info['instance'] = dict(instance)
        route_info['capability'] = normalize_capability(route_info.get('capability') or resolved_capability or capability) or capability

        backend = normalize_backend(branch_payload.get('backend') or instance.get('backend'))
        request_model_override = select_backend_request_model(
            instance,
            branch_payload.get('request_model') or instance.get('request_model'),
            branch_payload.get('model') or instance.get('model'),
        ) or str(branch_payload.get('model') or instance.get('model') or '').strip()
        infer_payload, route_info, _has_file_context, expose_input_artifacts = build_responses_infer_execution_payload(
            branch_payload,
            route_info=route_info,
            instance=instance,
            instance_id=instance_id,
            backend=backend,
            capability=route_info['capability'],
            request_model_override=request_model_override,
            upload_present=bool(upload and getattr(upload, 'filename', None)),
        )
        return {
            'capability': route_info['capability'],
            'route_info': route_info,
            'instance': instance,
            'effective_data': branch_payload,
            'infer_payload': infer_payload,
            'expose_input_artifacts': expose_input_artifacts,
            'upload': upload,
        }

    def execute_prepared_materialization_branch(self, plan: dict[str, Any]) -> dict[str, Any]:
        invoke_internal_api_json_route = self._hook('invoke_internal_api_json_route')
        filter_responses_infer_result = self._hook('filter_responses_infer_result')

        infer_payload = plan.get('infer_payload') if isinstance(plan.get('infer_payload'), dict) else {}
        infer_result, status_code = invoke_internal_api_json_route(
            payload=infer_payload,
            upload=plan.get('upload'),
        )
        if status_code >= 400:
            raise _MaterializationExecutionError(
                str(infer_result.get('error') or 'Request failed.'),
                status_code=status_code,
            )
        infer_result = filter_responses_infer_result(
            infer_result,
            expose_input_artifacts=bool(plan.get('expose_input_artifacts')),
        )
        infer_result = self._attach_tts_audio_integrity_evidence(
            infer_result,
            capability=plan.get('capability'),
            infer_payload=infer_payload,
        )
        return {
            'capability': str(plan.get('capability') or '').strip() or None,
            'route_info': plan.get('route_info') if isinstance(plan.get('route_info'), dict) else {},
            'instance': plan.get('instance') if isinstance(plan.get('instance'), dict) else {},
            'effective_data': plan.get('effective_data') if isinstance(plan.get('effective_data'), dict) else {},
            'infer_result': infer_result if isinstance(infer_result, dict) else {},
        }

    def build_preview_instance_payload(self, instance: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')
        infer_supported_capabilities = self._hook('infer_supported_capabilities')
        instance_effective_capability = self._hook('instance_effective_capability')
        compact_string_list = self._hook('compact_string_list')
        summarize_backend_metadata_for_routing = self._hook('summarize_backend_metadata_for_routing')
        summarize_backend_runtime_for_routing = self._hook('summarize_backend_runtime_for_routing')
        summarize_session_controls_for_routing = self._hook('summarize_session_controls_for_routing')

        if not isinstance(instance, dict):
            return None
        runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), dict) else {}
        supported_capabilities = [
            item for item in (
                instance.get('supported_capabilities')
                or infer_supported_capabilities(
                    str(instance.get('model') or instance.get('modelName') or '').strip(),
                    instance.get('backend'),
                    instance.get('capability'),
                    metadata=instance,
                )
            )
            if isinstance(item, str) and item
        ]
        return {
            'instance_id': str(instance.get('instance_id') or '').strip() or None,
            'model': str(instance.get('model') or instance.get('modelName') or '').strip() or None,
            'backend': normalize_backend(instance.get('backend')),
            'capability': normalize_capability(instance.get('capability')) or instance_effective_capability(instance),
            'supported_capabilities': supported_capabilities,
            'text_capable': bool(
                instance.get('text_capable')
                if isinstance(instance.get('text_capable'), bool)
                else self.capability_chat in supported_capabilities
            ),
            'backend_package': str(instance.get('backend_package') or '').strip() or None,
            'backend_contract': str(instance.get('backend_contract') or '').strip() or None,
            'provider_capabilities': compact_string_list(instance.get('provider_capabilities')),
            'backend_metadata': summarize_backend_metadata_for_routing(instance.get('backend_metadata')),
            'backend_runtime': summarize_backend_runtime_for_routing(runtime_status.get('backend_runtime')),
            'session_controls': instance.get('session_controls') if isinstance(instance.get('session_controls'), dict) else {},
            'session_controls_summary': summarize_session_controls_for_routing(instance.get('session_controls')),
            'tts_model_type': str(instance.get('tts_model_type') or '').strip() or None,
            'tts_speakers': instance.get('tts_speakers') if isinstance(instance.get('tts_speakers'), list) else [],
            'tts_languages': instance.get('tts_languages') if isinstance(instance.get('tts_languages'), list) else [],
            'readiness': str(runtime_status.get('readiness') or instance.get('readiness') or '').strip() or None,
            'activity': str(runtime_status.get('activity') or instance.get('activity') or '').strip() or None,
        }

    def build_ghost_route_preview_payload(
        self,
        route_info: dict[str, Any],
        request_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        build_working_frame = self._hook('build_working_frame')

        route_runtime = route_info.get('route_runtime') if isinstance(route_info.get('route_runtime'), dict) else {}
        context_strategy = route_runtime.get('context_strategy') if isinstance(route_runtime.get('context_strategy'), dict) else {}
        working_frame = route_runtime.get('working_frame') if isinstance(route_runtime.get('working_frame'), dict) else {}
        if not working_frame:
            working_frame = build_working_frame(
                request_payload=request_payload,
                route_payload=route_info,
                freeze=False,
            )
            if working_frame:
                route_runtime = dict(route_runtime)
                route_runtime['working_frame'] = working_frame
        payload = {
            'instance': self.build_preview_instance_payload(route_info.get('instance')),
            'route': {
                'source': route_info.get('route_source'),
                'reason': route_info.get('route_reason'),
                'confidence': route_info.get('route_confidence'),
                'reuse_last_artifact': route_info.get('route_reuse_last_artifact'),
                'artifact_ref': route_info.get('route_artifact_ref'),
                'artifact_path': route_info.get('route_artifact_path'),
                'context_mode': context_strategy.get('mode'),
                'context_reason': context_strategy.get('reason'),
                'traits': route_runtime.get('route_traits') if isinstance(route_runtime.get('route_traits'), dict) else None,
            },
        }
        request_meta = route_info.get('request_meta') if isinstance(route_info.get('request_meta'), dict) else (
            route_runtime.get('request_meta') if isinstance(route_runtime.get('request_meta'), dict) else {}
        )
        if request_meta:
            payload['request_meta'] = request_meta
        if route_runtime:
            payload['runtime'] = route_runtime
        if working_frame:
            payload['working_frame'] = working_frame
        return payload

    def attach_backend_handoff_truth_to_request_payload(
        self,
        data: Any,
        route_info: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = dict(data if isinstance(data, dict) else dict(data or {}))
        if not isinstance(route_info, dict) or not route_info:
            return payload

        route_runtime = route_info.get('route_runtime') if isinstance(route_info.get('route_runtime'), dict) else {}
        request_meta = route_info.get('request_meta') if isinstance(route_info.get('request_meta'), dict) else (
            route_runtime.get('request_meta') if isinstance(route_runtime.get('request_meta'), dict) else {}
        )
        if request_meta and payload.get('request_meta') in (None, '', [], {}):
            payload['request_meta'] = request_meta

        if payload.get('ghost_preview') in (None, '', [], {}):
            preview_payload = self.build_ghost_route_preview_payload(
                route_info,
                request_payload=payload,
            )
            if preview_payload:
                payload['ghost_preview'] = preview_payload
        return payload

    def refresh_live_route_target(
        self,
        *,
        request_payload: dict[str, Any],
        route_info: Optional[dict[str, Any]],
        instance_id: Optional[str],
        instance: Optional[dict[str, Any]],
        resolved_capability: Optional[str],
    ) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str], Optional[dict[str, Any]], Optional[str]]:
        resolve_responses_target_instance = self._hook('resolve_responses_target_instance')
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')

        route_payload = route_info if isinstance(route_info, dict) else {}
        if not route_payload or not str(instance_id or '').strip():
            return instance_id, instance, resolved_capability, route_info, None

        route_instance = route_payload.get('instance') if isinstance(route_payload.get('instance'), dict) else {}
        rebound_payload = dict(request_payload or {})
        requested_capability = normalize_capability(
            route_payload.get('capability')
            or resolved_capability
            or rebound_payload.get('capability')
        )
        if requested_capability:
            rebound_payload['capability'] = requested_capability
        backend_hint = normalize_backend(
            route_payload.get('backend')
            or route_instance.get('backend')
            or rebound_payload.get('backend')
        )
        if backend_hint:
            rebound_payload['backend'] = backend_hint
        model_hint = str(
            route_payload.get('model')
            or route_instance.get('model')
            or route_instance.get('modelName')
            or rebound_payload.get('model')
            or rebound_payload.get('modelName')
            or rebound_payload.get('request_model')
            or rebound_payload.get('requestModel')
            or ''
        ).strip()
        if model_hint:
            rebound_payload['model'] = model_hint

        rebound_instance_id, rebound_instance, rebound_capability, rebound_error = resolve_responses_target_instance(
            rebound_payload,
            forced_instance_id=str(instance_id or '').strip(),
        )
        if not rebound_instance_id or not rebound_instance:
            fallback_payload = dict(rebound_payload)
            fallback_payload.pop('instance_id', None)
            fallback_instance_id, fallback_instance, fallback_capability, fallback_error = resolve_responses_target_instance(
                fallback_payload,
            )
            if not fallback_instance_id or not fallback_instance:
                return None, None, fallback_capability or rebound_capability or requested_capability, route_info, (
                    fallback_error or rebound_error
                )
            rebound_instance_id = fallback_instance_id
            rebound_instance = fallback_instance
            rebound_capability = fallback_capability or rebound_capability

        updated_route_info = dict(route_payload)
        updated_route_info['instance_id'] = rebound_instance_id
        updated_route_info['instance'] = rebound_instance
        if model_hint:
            updated_route_info['model'] = model_hint
        if backend_hint:
            updated_route_info['backend'] = backend_hint
        if rebound_capability or requested_capability:
            updated_route_info['capability'] = rebound_capability or requested_capability
        if rebound_instance_id != str(instance_id or '').strip():
            route_runtime = (
                dict(updated_route_info.get('route_runtime') or {})
                if isinstance(updated_route_info.get('route_runtime'), dict)
                else {}
            )
            route_runtime['target_rebound'] = {
                'stale_instance_id': str(instance_id or '').strip(),
                'rebound_instance_id': rebound_instance_id,
                'capability': rebound_capability or requested_capability,
                'reason': 'stale_route_target_rebound',
            }
            updated_route_info['route_runtime'] = route_runtime
        return (
            rebound_instance_id,
            rebound_instance,
            rebound_capability or requested_capability or resolved_capability,
            updated_route_info,
            None,
        )

    def handle_responses_request(
        self,
        *,
        forced_instance_id: Optional[str] = None,
        data_override: Optional[Any] = None,
        upload_override: Any = None,
    ):
        normalize_request_payload = self._hook('normalize_request_payload')
        parse_bool = self._hook('parse_bool')
        resolve_ghost_auto_route = self._hook('resolve_ghost_auto_route')
        resolve_responses_target_instance = self._hook('resolve_responses_target_instance')
        log_unified_event = self._hook('log_unified_event')
        prepare_effective_request_data = self._hook('prepare_effective_request_data')
        build_missing_required_session_controls = self._hook('build_missing_required_session_controls')
        normalize_backend = self._hook('normalize_backend')
        select_backend_request_model = self._hook('select_backend_request_model')
        normalize_capability = self._hook('normalize_capability')
        instance_supports_capability = self._hook('instance_supports_capability')
        infer_capability = self._hook('infer_capability')
        normalize_response_lookup_id = self._hook('normalize_response_lookup_id')
        register_response_lookup = self._hook('register_response_lookup')
        touch_response_lookup = self._hook('touch_response_lookup')
        build_canonical_error_response_payload = self._hook('build_canonical_error_response_payload')
        finalize_response_frame_payload = self._hook('finalize_response_frame_payload')
        handle_responses_request = self._hook('handle_responses_request')
        extract_responses_batch_items = self._hook('extract_responses_batch_items')
        parse_float_with_bounds = self._hook('parse_float_with_bounds')
        parse_int_with_bounds = self._hook('parse_int_with_bounds')
        extract_selected_reference_artifacts = self._hook('extract_selected_reference_artifacts')
        select_matching_selected_reference_artifact = self._hook('select_matching_selected_reference_artifact')
        extract_responses_prompt = self._hook('extract_responses_prompt')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')
        should_attach_selected_reference_file_context = self._hook('should_attach_selected_reference_file_context')
        extract_responses_messages = self._hook('extract_responses_messages')
        extract_ghost_route_messages = self._hook('extract_ghost_route_messages')
        inject_selected_reference_into_chat_messages = self._hook('inject_selected_reference_into_chat_messages')
        inject_ghost_runtime_policy_into_chat_messages = self._hook('inject_ghost_runtime_policy_into_chat_messages')
        inject_prepare_phase_contract_into_chat_messages = self._hook('inject_prepare_phase_contract_into_chat_messages')
        choose_context_strategy = self._hook('choose_context_strategy')
        apply_context_strategy = self._hook('apply_context_strategy')
        stream_chat_backend_as_responses = self._hook('stream_chat_backend_as_responses')
        execute_chat_backend_request = self._hook('execute_chat_backend_request')
        execute_external_text_target = self._hook('execute_external_text_target')
        validate_external_text_request = self._hook('validate_external_text_request')
        build_external_target_inputs = self._hook('build_external_target_inputs')
        apply_selected_reference_prompt_prefix = self._hook('apply_selected_reference_prompt_prefix')
        external_execution_failure = self._hook('external_execution_failure')
        persist_generated_text_artifact_if_requested = self._hook('persist_generated_text_artifact_if_requested')
        request_exception_details = self._hook('request_exception_details')
        build_canonical_response_payload = self._hook('build_canonical_response_payload')
        attach_response_semantic_phase_payload = self._hook('attach_response_semantic_phase_payload')
        truth_gate_response_output_claims = self._hook('truth_gate_response_output_claims')
        build_late_fill_state = self._hook('build_late_fill_state')
        attach_late_fill_state = self._hook('attach_late_fill_state')
        ensure_response_lookup_for_payload = self._hook('ensure_response_lookup_for_payload')
        schedule_response_late_fill = self._hook('schedule_response_late_fill')
        schedule_post_response_substrate_hygiene = self.hooks.get('schedule_post_response_substrate_hygiene')
        build_responses_infer_execution_payload = self._hook('build_responses_infer_execution_payload')
        invoke_internal_api_json_route = self._hook('invoke_internal_api_json_route')
        filter_responses_infer_result = self._hook('filter_responses_infer_result')
        build_canonical_batch_response_payload = self._hook('build_canonical_batch_response_payload')
        build_canonical_response_stream_events = self._hook('build_canonical_response_stream_events')
        execute_materialization_branches = self._hook('execute_materialization_branches')
        project_response_payload_for_wire = self._hook('project_response_payload_for_wire')
        promote_current_predecessor_context = self.hooks.get(
            'promote_current_predecessor_context'
        )

        def accept_phase_output(
            raw_text: Any,
            *,
            effective_route_payload: Optional[dict[str, Any]],
            effective_request_payload: dict[str, Any],
            effective_capability: Optional[str],
            retry_same_phase: Optional[Any] = None,
        ) -> tuple[str, list[dict[str, Any]]]:
            source_text = str(raw_text or '').strip()
            is_preparation = phase_output_is_graph_preparation(
                route_payload=effective_route_payload,
                request_payload=effective_request_payload,
                capability=effective_capability,
            )
            explicit_diagnostics = request_explicitly_allows_control_diagnostics(
                request_payload=effective_request_payload,
                route_payload=effective_route_payload,
                capability=effective_capability,
            )
            if explicit_diagnostics or (
                not is_preparation and not control_json_envelope_suspected(source_text)
            ):
                return source_text, []
            attempts = [classify_phase_output_text(source_text)]
            if (
                attempts[-1].get('status') == 'repair_required'
                and is_preparation
                and callable(retry_same_phase)
            ):
                try:
                    retry_text = str(retry_same_phase() or '').strip()
                    attempts.append(classify_phase_output_text(retry_text))
                except Exception as exc:  # noqa: BLE001
                    attempts.append(
                        {
                            'status': 'repair_required',
                            'accepted_text': '',
                            'source_sha256': hashlib.sha256(b'').hexdigest(),
                            'source_bytes': 0,
                            'reason': (
                                'bounded same-phase repair call failed before producing '
                                f'content ({type(exc).__name__})'
                            ),
                        }
                    )
            accepted_text = str(attempts[-1].get('accepted_text') or '').strip()
            if attempts[-1].get('status') == 'repair_required':
                accepted_text = phase_output_repair_notice()
            return accepted_text, attempts

        def maybe_schedule_post_response_substrate_hygiene(
            response_payload: dict[str, Any],
            *,
            route_payload: Optional[dict[str, Any]],
            reason: str,
        ) -> None:
            if not callable(schedule_post_response_substrate_hygiene):
                return
            try:
                schedule_post_response_substrate_hygiene(
                    response_payload,
                    route_payload=route_payload,
                    reason=reason,
                )
            except Exception:  # noqa: BLE001
                logging.exception('Could not schedule post-response substrate hygiene.')

        if data_override is not None:
            data = data_override
            upload = upload_override
            uploads = (
                list(upload_override)
                if isinstance(upload_override, (list, tuple))
                else [upload_override]
                if upload_override is not None
                else []
            )
            is_multipart = False
        else:
            is_multipart = request.content_type and request.content_type.startswith('multipart/form-data')
            if is_multipart:
                data = request.form
                uploads = [
                    item
                    for item in [
                        *request.files.getlist('file'),
                        *request.files.getlist('files'),
                    ]
                    if item is not None and getattr(item, 'filename', None)
                ]
                upload = uploads[0] if uploads else None
            else:
                data = request.get_json(silent=True) or {}
                upload = None
                uploads = []
        data = normalize_request_payload(data)
        if callable(promote_current_predecessor_context):
            data = promote_current_predecessor_context(data)
        if uploads:
            for current_upload in uploads:
                data, _materialized_upload = self.materialize_upload_input_artifacts(
                    data,
                    current_upload,
                )
            upload = None

        route_info = None
        if not forced_instance_id and parse_bool(data.get('ghost_route'), default=False):
            route_info, resolution_error = resolve_ghost_auto_route(data, upload=upload)
            if resolution_error:
                status_code = 404 if 'was not found' in resolution_error or 'No running instance' in resolution_error else 400
                return jsonify({'error': resolution_error}), status_code
            instance_id = str((route_info or {}).get('instance_id') or '').strip()
            instance = (route_info or {}).get('instance')
            resolved_capability = str((route_info or {}).get('capability') or '').strip() or None
        else:
            instance_id, instance, resolved_capability, resolution_error = resolve_responses_target_instance(
                data,
                forced_instance_id=forced_instance_id,
            )
            if resolution_error:
                status_code = 404 if 'was not found' in resolution_error or 'No running instance' in resolution_error else 400
                return jsonify({'error': resolution_error}), status_code
        if (
            route_info
            and str((route_info or {}).get('route_source') or '').strip().lower() == 'ghost_carried'
            and bool((route_info or {}).get('route_reuse_last_artifact'))
            and any(
                str(value or '').strip()
                for value in (
                    (route_info or {}).get('route_artifact_ref'),
                    (route_info or {}).get('route_artifact_path'),
                    data.get('artifact_ref'),
                    data.get('artifact_path'),
                )
            )
        ):
            (
                instance_id,
                instance,
                resolved_capability,
                route_info,
                resolution_error,
            ) = self.refresh_live_route_target(
                request_payload=data,
                route_info=route_info,
                instance_id=instance_id,
                instance=instance if isinstance(instance, dict) else None,
                resolved_capability=resolved_capability,
            )
            if resolution_error:
                status_code = 404 if 'was not found' in resolution_error or 'No running instance' in resolution_error else 400
                return jsonify({'error': resolution_error}), status_code
        if route_info:
            route_runtime = route_info.get('route_runtime') if isinstance(route_info.get('route_runtime'), dict) else {}
            log_unified_event(
                category='responses',
                action='route',
                status='ok',
                instance_id=route_info.get('instance_id'),
                model=(instance or {}).get('model') if isinstance(instance, dict) else None,
                backend=(instance or {}).get('backend') if isinstance(instance, dict) else None,
                capability=route_info.get('capability'),
                route_source=route_info.get('route_source'),
                route_confidence=route_info.get('route_confidence'),
                route_artifact_path=route_info.get('route_artifact_path'),
                prompt_class=route_runtime.get('prompt_class'),
                session_class=route_runtime.get('session_class'),
                message=route_info.get('route_reason'),
            )
            embedding_audit = route_runtime.get('embedding_audit') if isinstance(route_runtime.get('embedding_audit'), dict) else {}
            if embedding_audit.get('available') or embedding_audit.get('attached'):
                log_unified_event(
                    category='responses',
                    action='route_embedding_audit',
                    status=str(embedding_audit.get('status') or 'observed'),
                    instance_id=route_info.get('instance_id'),
                    model=(instance or {}).get('model') if isinstance(instance, dict) else None,
                    backend=(instance or {}).get('backend') if isinstance(instance, dict) else None,
                    capability=route_info.get('capability'),
                    prompt_class=embedding_audit.get('prompt_class'),
                    session_class=route_runtime.get('session_class'),
                    route_hint_capability=(
                        embedding_audit.get('route_hint_capability')
                        or embedding_audit.get('heuristic_capability')
                    ),
                    route_hint_confidence=(
                        embedding_audit.get('route_hint_confidence')
                        if embedding_audit.get('route_hint_confidence') is not None
                        else embedding_audit.get('heuristic_confidence')
                    ),
                    suggested_capability=embedding_audit.get('embedding_capability'),
                    suggested_instance_id=embedding_audit.get('embedding_instance_id'),
                    embedding_score=embedding_audit.get('embedding_score'),
                    embedding_score_gap=embedding_audit.get('embedding_score_gap'),
                    bias_applied=embedding_audit.get('bias_applied'),
                    message=(
                        f"embedding audit: route_hint="
                        f"{embedding_audit.get('route_hint_capability') or embedding_audit.get('heuristic_capability') or 'none'} "
                        f"suggested={embedding_audit.get('embedding_capability') or 'none'} "
                        f"final={embedding_audit.get('final_capability') or 'none'}"
                    ),
                )
        if not instance_id or not instance:
            return jsonify({'error': "Parameter 'instance_id' is missing."}), 400
        is_external_target = (
            str(instance.get('target_kind') or '').strip().lower() == 'external'
        )
        effective_data, route_info, _planner_meta, _control_hints = prepare_effective_request_data(
            data,
            route_info=route_info,
            instance=instance if isinstance(instance, dict) else None,
        )
        data = effective_data
        if is_external_target:
            route_info = dict(route_info or {})
            route_info.update(
                {
                    'instance_id': instance_id,
                    'instance': instance,
                    'capability': self.capability_chat,
                    'route_source': str(
                        route_info.get('route_source')
                        or 'explicit_external_target'
                    ),
                    'route_reason': str(
                        route_info.get('route_reason')
                        or 'Explicitly selected external ChatGPT target via Codex.'
                    ),
                }
            )
            route_runtime = (
                dict(route_info.get('route_runtime') or {})
                if isinstance(route_info.get('route_runtime'), dict)
                else {}
            )
            route_runtime['external_target'] = {
                'id': instance_id,
                'provider': 'codex_cli',
                'target_kind': 'external',
                'lifecycle_managed': False,
                'text_only': False,
                'inputs': ['text', 'image', 'file'],
                'outputs': ['text'],
                'files_enabled': instance.get('files_enabled') is True,
            }
            route_info['route_runtime'] = route_runtime
        if route_info:
            data = self.attach_backend_handoff_truth_to_request_payload(data, route_info)
        if route_info:
            missing_session_controls = build_missing_required_session_controls(instance, data)
            if missing_session_controls:
                preview_payload = self.build_ghost_route_preview_payload(route_info)
                preview_payload['missing_fields'] = missing_session_controls
                return (
                    jsonify(
                        {
                            'error': missing_session_controls[0]['message'],
                            'missing_session_controls': preview_payload,
                        }
                    ),
                    400,
                )

        backend = normalize_backend(data.get('backend') or instance.get('backend'))
        model_name = str(data.get('model') or instance.get('model') or '').strip()
        request_model_override = select_backend_request_model(
            instance,
            data.get('request_model') or instance.get('request_model'),
            data.get('model') or model_name,
        ) or model_name
        requested_capability = str(route_info.get('capability') or '').strip() if route_info else normalize_capability(data.get('capability'))
        instance_capability = normalize_capability(instance.get('capability')) or resolved_capability
        if requested_capability and not instance_supports_capability(instance, requested_capability):
            return jsonify(
                {
                    'error': (
                        f"Capability '{requested_capability}' does not match the selected instance "
                        f"'{instance_id}' (capability '{instance_capability}')."
                    )
                }
            ), 400
        capability = requested_capability or instance_capability
        if not capability:
            capability = infer_capability(model_name, backend)
        if capability == self.capability_embedding:
            return jsonify(
                {
                    'error': (
                        'Embedding instances are internal Ghost helper models for pre-ranking '
                        'and are not executed through /api/responses.'
                    )
                }
            ), 400
        wants_stream = parse_bool(data.get('stream'), default=False)
        raw_requested_response_id = str(
            data.get('response_id') or data.get('responseId') or ''
        ).strip()
        requested_response_id = (
            normalize_response_lookup_id(raw_requested_response_id)
            if raw_requested_response_id
            else None
        )
        if requested_response_id and isinstance(data, dict):
            data = dict(data)
            data['response_id'] = requested_response_id
        response_lookup_record: Optional[dict[str, Any]] = None
        response_lookup_message_id: Optional[str] = None

        def ensure_response_lookup(mode_hint: str) -> Optional[dict[str, Any]]:
            nonlocal response_lookup_record, response_lookup_message_id
            if not requested_response_id:
                return None
            if response_lookup_record:
                return response_lookup_record
            response_lookup_record = register_response_lookup(
                response_id=requested_response_id,
                message_id='',
                instance_id=instance_id,
                model_name=model_name,
                backend=backend,
                capability=capability,
                mode=mode_hint,
                route_payload=route_info,
            )
            response_lookup_message_id = str(response_lookup_record.get('message_id') or '').strip() or None
            return response_lookup_record

        def mark_response_lookup_failed(
            error_message: str,
            mode_hint: str,
            *,
            response_payload: Optional[dict[str, Any]] = None,
        ) -> None:
            payload_response_id = str(
                (response_payload or {}).get('id')
                or (response_payload or {}).get('response_id')
                or ''
            ).strip()
            target_response_id = requested_response_id or payload_response_id
            if not target_response_id:
                return
            if requested_response_id:
                ensure_response_lookup(mode_hint)
            else:
                ensure_response_lookup_for_payload(
                    response_payload,
                    mode_hint=mode_hint,
                    route_payload=route_info,
                )
            touch_response_lookup(
                target_response_id,
                status='failed',
                error_message=error_message,
                response_payload=response_payload,
            )

        def mark_response_lookup_completed(response_payload: dict, mode_hint: str) -> None:
            payload_response_id = str(
                response_payload.get('id')
                or response_payload.get('response_id')
                or ''
            ).strip()
            target_response_id = requested_response_id or payload_response_id
            if not target_response_id:
                return
            if requested_response_id:
                ensure_response_lookup(mode_hint)
            else:
                ensure_response_lookup_for_payload(
                    response_payload,
                    mode_hint=mode_hint,
                    route_payload=route_info,
                )
            touch_response_lookup(
                target_response_id,
                status='completed',
                output_text=str(response_payload.get('output_text') or ''),
                response_payload=response_payload,
            )

        def build_runtime_error_payload(error_message: str, status_code: int, mode_hint: str) -> dict[str, Any]:
            error_payload = build_canonical_error_response_payload(
                error_message=error_message,
                status_code=status_code,
                instance_id=instance_id,
                model_name=model_name,
                backend=backend,
                capability=capability or mode_hint or 'response',
                mode=mode_hint or capability or 'response',
                route_payload=route_info,
                response_id=requested_response_id,
                message_id=response_lookup_message_id,
            )
            return finalize_response_frame_payload(error_payload, request_payload=data)

        def maybe_retry_self_healed_response(error_message: str, status_code: int) -> Optional[Response]:
            if forced_instance_id:
                return None
            if not route_info or not parse_bool(data.get('ghost_route'), default=False):
                return None
            if parse_bool(data.get('ghost_self_heal_attempted'), default=False):
                return None
            try:
                normalized_status_code = int(status_code or 0)
            except (TypeError, ValueError):
                normalized_status_code = 0
            if normalized_status_code < 500 or normalized_status_code >= 600:
                return None

            retry_failure = {
                'capability': capability,
                'failed_instance_id': instance_id,
                'status_code': normalized_status_code or 500,
                'error_message': str(error_message or '').strip() or 'Request failed.',
            }
            retry_payload = dict(data if isinstance(data, dict) else dict(data))
            retry_payload['ghost_self_heal_attempted'] = True
            retry_route_info, retry_error = resolve_ghost_auto_route(
                retry_payload,
                upload=upload,
                excluded_instance_ids=[instance_id],
                retry_failure=retry_failure,
            )
            if retry_error or not retry_route_info:
                return None

            retry_instance_id = str((retry_route_info or {}).get('instance_id') or '').strip()
            retry_capability = normalize_capability((retry_route_info or {}).get('capability'))
            if not retry_instance_id:
                return None
            if str((retry_route_info or {}).get('route_source') or '').strip() != 'self_heal':
                return None
            if retry_instance_id == instance_id and retry_capability == capability:
                return None

            retry_payload['ghost_preview'] = self.build_ghost_route_preview_payload(
                retry_route_info,
                request_payload=retry_payload,
            )
            log_unified_event(
                category='responses',
                action='self_heal_retry',
                status='ok',
                instance_id=retry_instance_id,
                model=((retry_route_info or {}).get('instance') or {}).get('model') if isinstance((retry_route_info or {}).get('instance'), dict) else None,
                backend=((retry_route_info or {}).get('instance') or {}).get('backend') if isinstance((retry_route_info or {}).get('instance'), dict) else None,
                capability=retry_capability,
                previous_instance_id=instance_id,
                previous_model=request_model_override or model_name,
                route_source=retry_route_info.get('route_source'),
                prompt_class=((retry_route_info.get('route_runtime') or {}).get('prompt_class') if isinstance(retry_route_info.get('route_runtime'), dict) else None),
                session_class=((retry_route_info.get('route_runtime') or {}).get('session_class') if isinstance(retry_route_info.get('route_runtime'), dict) else None),
                message=f'Self-heal retry after {status_code}: {error_message}',
            )
            return handle_responses_request(
                data_override=retry_payload,
                upload_override=upload,
            )

        def selected_instance_is_mlx_vlm(candidate: Any) -> bool:
            if not isinstance(candidate, dict):
                return False
            metadata = candidate.get('backend_metadata') if isinstance(candidate.get('backend_metadata'), dict) else {}
            backend_package = str(
                candidate.get('backend_package')
                or metadata.get('backend_package')
                or ''
            ).strip().lower()
            backend_contract = str(
                candidate.get('backend_contract')
                or metadata.get('backend_contract')
                or ''
            ).strip().lower()
            return (
                normalize_backend(candidate.get('backend')) == 'mlx'
                and (
                    backend_package == 'mlx_vlm'
                    or backend_contract.startswith('mlx_vlm.')
                )
            )

        def maybe_retry_single_chat_transport_fallback(
            error_message: str,
            status_code: int,
        ) -> Optional[Response]:
            if forced_instance_id:
                return None
            if parse_bool(data.get('single_chat_transport_fallback_attempted'), default=False):
                return None
            if normalize_capability(capability) != self.capability_chat:
                return None
            if not selected_instance_is_mlx_vlm(instance):
                return None
            try:
                normalized_status_code = int(status_code or 0)
            except (TypeError, ValueError):
                normalized_status_code = 0
            if normalized_status_code and normalized_status_code < 500:
                return None

            failed_instance_id = str(instance_id or '').strip()
            retry_payload = dict(data if isinstance(data, dict) else dict(data or {}))
            retry_payload['single_chat_transport_fallback_attempted'] = True
            retry_payload['capability'] = self.capability_chat
            for key in ('instance_id', 'model', 'modelName', 'request_model', 'requestModel', 'backend'):
                retry_payload.pop(key, None)
            fallback_instance_id, fallback_instance, fallback_capability, fallback_error = resolve_responses_target_instance(
                retry_payload,
                excluded_instance_ids=[failed_instance_id] if failed_instance_id else None,
            )
            if fallback_error or not fallback_instance_id or not isinstance(fallback_instance, dict):
                return None
            if failed_instance_id and fallback_instance_id == failed_instance_id:
                return None

            retry_payload['instance_id'] = fallback_instance_id
            retry_payload['model'] = str(fallback_instance.get('model') or fallback_instance.get('modelName') or '').strip()
            retry_payload['backend'] = normalize_backend(fallback_instance.get('backend'))
            request_model = str(fallback_instance.get('request_model') or fallback_instance.get('requestModel') or '').strip()
            if request_model:
                retry_payload['request_model'] = request_model
            retry_payload['runtime_transport_fallback'] = {
                'reason': 'selected_mlx_vlm_text_chat_transport_failed',
                'failed_instance_id': failed_instance_id or None,
                'fallback_instance_id': fallback_instance_id,
                'capability': fallback_capability or self.capability_chat,
                'status_code': normalized_status_code or status_code,
                'error_message': str(error_message or '').strip() or 'Request failed.',
            }
            log_unified_event(
                category='responses',
                action='single_chat_transport_fallback',
                status='ok',
                instance_id=fallback_instance_id,
                model=retry_payload.get('model'),
                backend=retry_payload.get('backend'),
                capability=fallback_capability or self.capability_chat,
                previous_instance_id=failed_instance_id,
                previous_model=request_model_override or model_name,
                message=f'Single chat fallback after selected MLX/VLM transport failure: {error_message}',
            )
            return handle_responses_request(
                data_override=retry_payload,
                upload_override=upload,
            )

        normalized_payload = data if isinstance(data, dict) else dict(data)
        batch_items = extract_responses_batch_items(normalized_payload)
        batch_prompts = [str(item.get('prompt') or '').strip() for item in batch_items if str(item.get('prompt') or '').strip()]
        if batch_prompts and capability != self.capability_image_generation:
            return jsonify(
                {
                    'error': 'batch_prompts is currently only supported for image_generation.'
                }
            ), 400
        temperature = None
        top_p = None
        max_tokens = None
        try:
            if data.get('temperature') not in (None, ''):
                temperature = parse_float_with_bounds(data.get('temperature'), default=0.7, minimum=0.0, maximum=2.0)
            if data.get('top_p') not in (None, '') or data.get('topP') not in (None, ''):
                top_p = parse_float_with_bounds(data.get('top_p') or data.get('topP'), default=0.9, minimum=0.0, maximum=1.0)
            if data.get('max_tokens') not in (None, '') or data.get('maxTokens') not in (None, ''):
                max_tokens = parse_int_with_bounds(
                    data.get('max_tokens') or data.get('maxTokens'),
                    default=1_000_000,
                    minimum=1,
                    maximum=1_000_000,
                )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        selected_reference_artifacts = extract_selected_reference_artifacts(data)
        matched_selected_reference = select_matching_selected_reference_artifact(
            selected_reference_artifacts,
            capability,
            instance=instance,
        )
        responses_prompt = extract_responses_prompt(normalized_payload)
        responses_current_turn_prompt = extract_responses_current_turn_prompt(normalized_payload)
        raw_file_path = str(data.get('file_path') or '').strip()
        if not raw_file_path and route_info and route_info.get('route_reuse_last_artifact'):
            raw_file_path = str(route_info.get('route_artifact_path') or '').strip()
        if (
            not raw_file_path
            and not upload
            and should_attach_selected_reference_file_context(
                prompt=responses_prompt,
                capability=capability,
                selected_reference_artifact=matched_selected_reference,
            )
        ):
            raw_file_path = str(matched_selected_reference.get('path') or '').strip()
        has_file_context = bool((upload and getattr(upload, 'filename', None)) or raw_file_path)

        if (
            normalize_capability(capability) != self.capability_chat
            and normalize_capability(capability) == 'vision_analysis'
            and not has_file_context
            and selected_instance_is_mlx_vlm(instance)
            and instance_supports_capability(instance, self.capability_chat)
        ):
            previous_capability = capability
            capability = self.capability_chat
            requested_capability = self.capability_chat
            data = dict(data if isinstance(data, dict) else dict(data or {}))
            data['capability'] = self.capability_chat
            normalized_payload = dict(normalized_payload)
            normalized_payload['capability'] = self.capability_chat
            if route_info:
                route_info = dict(route_info)
                route_info['capability'] = self.capability_chat
                route_runtime = (
                    dict(route_info.get('route_runtime') or {})
                    if isinstance(route_info.get('route_runtime'), dict)
                    else {}
                )
                route_runtime['capability_rebound'] = {
                    'from': previous_capability,
                    'to': self.capability_chat,
                    'reason': 'text_only_selected_mlx_vlm_turn',
                }
                route_info['route_runtime'] = route_runtime
            log_unified_event(
                category='responses',
                action='capability_rebound',
                status='ok',
                instance_id=instance_id,
                model=model_name,
                backend=backend,
                capability=self.capability_chat,
                previous_capability=previous_capability,
                message='Text-only selected MLX/VLM request rebounded to chat capability.',
            )

        if is_external_target:
            valid_external_request, validation_code, validation_error = (
                validate_external_text_request(
                    normalized_payload,
                    upload_present=bool(upload and getattr(upload, 'filename', None)),
                    files_enabled=instance.get('files_enabled') is True,
                )
            )
            external_prompt = str(
                responses_current_turn_prompt or responses_prompt or ''
            ).strip()
            external_context_messages = (
                extract_ghost_route_messages(
                    data,
                    include_selected_reference=False,
                )
                if parse_bool(data.get('ghost_route'), default=False)
                else extract_responses_messages(data)
            )
            if external_prompt and not any(
                str(item.get('role') or '').strip().lower() == 'user'
                and str(item.get('content') or '').strip() == external_prompt
                for item in external_context_messages
                if isinstance(item, dict)
            ):
                external_context_messages = [
                    *external_context_messages,
                    {'role': 'user', 'content': external_prompt},
                ]
            external_context_strategy = choose_context_strategy(
                instance=instance,
                messages=external_context_messages,
                prompt=external_prompt,
                has_file_context=False,
                conversation_id=(
                    normalized_payload.get('conversation_id')
                    or normalized_payload.get('conversationId')
                    or normalized_payload.get('thread_id')
                    or normalized_payload.get('threadId')
                ),
            )
            prepared_external_context = apply_context_strategy(
                external_context_messages,
                external_context_strategy,
            )
            bounded_external_prompt = apply_selected_reference_prompt_prefix(
                external_prompt,
                selected_reference_artifacts,
                self.capability_chat,
            )
            external_prompt = _external_provider_prompt_with_bounded_context(
                external_prompt,
                prepared_external_context,
                bounded_task_prompt=bounded_external_prompt,
            )
            external_inputs = build_external_target_inputs(normalized_payload)
            if valid_external_request and not external_prompt:
                valid_external_request = False
                validation_code = 'CODEX_INVALID_REQUEST'
                validation_error = 'No Responses text input was provided.'

            if not valid_external_request:
                error_message = str(
                    validation_error or 'The ChatGPT request contains unsupported input.'
                )
                error_payload = build_canonical_error_response_payload(
                    error_message=error_message,
                    status_code=400,
                    instance_id=instance_id,
                    model_name=model_name,
                    backend=backend,
                    capability=capability or self.capability_chat,
                    mode='external_chat',
                    route_payload=route_info,
                    response_id=requested_response_id,
                    message_id=response_lookup_message_id,
                )
                error_payload['error_ref'] = {
                    'code': str(validation_code or 'CODEX_INVALID_REQUEST'),
                    'stage': 'external_request_validation',
                }
                error_payload['recovery_hint'] = (
                    'Send a non-empty prompt with explicitly selected local files or Ollmo artifacts.'
                )
                error_payload = finalize_response_frame_payload(
                    error_payload,
                    request_payload=normalized_payload,
                )
                mark_response_lookup_failed(
                    error_message,
                    'external_chat',
                    response_payload=error_payload,
                )
                return jsonify(project_response_payload_for_wire(error_payload)), 400

            ensure_response_lookup('external_chat')
            execution_result = (
                execute_external_text_target(
                    external_prompt,
                    inputs=external_inputs,
                )
                if external_inputs
                else execute_external_text_target(external_prompt)
            )
            execution_status = str(execution_result.status.value)
            external_execution_evidence = {
                'kind': 'ollmo.external_provider_execution',
                'target_id': instance_id,
                'provider': 'codex_cli',
                'status': execution_status,
                'source': (
                    execution_result.discovery.source.value
                    if execution_result.discovery.source is not None
                    else None
                ),
                'version': execution_result.discovery.version,
                'duration_seconds': round(
                    float(execution_result.duration_seconds or 0.0),
                    3,
                ),
                'exit_code': execution_result.exit_code,
                'output_truncated': bool(execution_result.output_truncated),
                'diagnostic_truncated': bool(execution_result.diagnostic_truncated),
                'model_selection': 'codex_default',
                'exact_model_exposed': False,
                'input_handoff': [
                    item.as_dict()
                    for item in execution_result.input_handoff
                ],
                'input_count': len(execution_result.input_handoff),
            }
            route_info = dict(route_info or {})
            route_runtime = (
                dict(route_info.get('route_runtime') or {})
                if isinstance(route_info.get('route_runtime'), dict)
                else {}
            )
            route_runtime['context_strategy'] = external_context_strategy
            route_runtime['external_execution'] = {
                key: value
                for key, value in external_execution_evidence.items()
                if value is not None
            }
            route_info['route_runtime'] = route_runtime

            if not execution_result.succeeded:
                status_code, error_code, recovery_hint = external_execution_failure(
                    execution_result
                )
                error_message = (
                    f"ChatGPT request via Codex ended with status '{execution_status}'."
                )
                error_payload = build_canonical_error_response_payload(
                    error_message=error_message,
                    status_code=status_code,
                    instance_id=instance_id,
                    model_name=model_name,
                    backend=backend,
                    capability=capability,
                    mode='external_chat',
                    route_payload=route_info,
                    response_id=requested_response_id,
                    message_id=response_lookup_message_id,
                )
                error_payload['error_ref'] = {
                    'code': error_code,
                    'stage': 'external_execution',
                }
                error_payload['recovery_hint'] = recovery_hint
                if execution_result.diagnostic:
                    error_payload['error_detail'] = {
                        'message': error_message,
                        'diagnostic': str(execution_result.diagnostic),
                    }
                error_payload = finalize_response_frame_payload(
                    error_payload,
                    request_payload=normalized_payload,
                )
                mark_response_lookup_failed(
                    error_message,
                    'external_chat',
                    response_payload=error_payload,
                )
                return (
                    jsonify(project_response_payload_for_wire(error_payload)),
                    status_code,
                )

            provider_output_text = str(execution_result.output_text or '')
            provider_block_reason = _external_provider_block_reason(
                provider_output_text
            )
            if provider_block_reason is not None:
                blocked_surface_state = {
                    'state': 'blocked',
                    'status': 'blocked',
                    'summary': 'The downstream provider could not complete the bounded task.',
                    'message': provider_block_reason,
                    'reason': provider_block_reason,
                    'category_counts': {'blocked': 1},
                    'active_categories': ['blocked'],
                }
                external_execution_evidence.update(
                    {
                        'status': 'blocked',
                        'invocation_status': execution_status,
                        'blocked_reason': provider_block_reason,
                    }
                )
                route_runtime['external_execution'] = {
                    key: value
                    for key, value in external_execution_evidence.items()
                    if value is not None
                }
                route_runtime['surface_state'] = dict(blocked_surface_state)
                route_info['route_runtime'] = route_runtime

                response_payload = build_canonical_response_payload(
                    instance_id=instance_id,
                    model_name=model_name,
                    backend=backend,
                    capability=capability,
                    mode='external_chat',
                    output_text=provider_output_text,
                    source_payload={},
                    route_payload=route_info,
                    response_id=requested_response_id,
                    message_id=response_lookup_message_id,
                )
                blocked_output = {
                    'slot_id': 'output-1',
                    'branch_id': 'phase-1',
                    'phase_id': 'phase-1',
                    'type': 'text',
                    'status': 'blocked',
                    'lifecycle': 'blocked_output',
                    'source': 'external_provider_execution',
                    'compatibility_derived': False,
                    'value': provider_output_text,
                    'blocked_reason': provider_block_reason,
                }
                response_payload['output_slots'] = [dict(blocked_output)]
                response_payload['outputs'] = [dict(blocked_output)]
                response_payload['artifacts'] = []
                response_payload['surface_state'] = dict(blocked_surface_state)
                response_payload['lifecycle_state'] = 'blocked'
                response_payload = finalize_response_frame_payload(
                    response_payload,
                    request_payload=normalized_payload,
                )
                mark_response_lookup_completed(response_payload, 'external_chat')
                if wants_stream:
                    return Response(
                        stream_with_context(
                            build_canonical_response_stream_events(response_payload)
                        ),
                        mimetype='text/event-stream',
                    )
                return jsonify(project_response_payload_for_wire(response_payload))

            assistant_text, phase_acceptance_attempts = accept_phase_output(
                provider_output_text,
                effective_route_payload=route_info,
                effective_request_payload=normalized_payload,
                effective_capability=capability,
            )
            response_payload = build_canonical_response_payload(
                instance_id=instance_id,
                model_name=model_name,
                backend=backend,
                capability=capability,
                mode='external_chat',
                output_text=assistant_text,
                source_payload=attach_response_semantic_phase_payload(
                    {},
                    output_text=assistant_text,
                    route_payload=route_info,
                    request_payload=normalized_payload,
                    capability=capability,
                ),
                route_payload=route_info,
                response_id=requested_response_id,
                message_id=response_lookup_message_id,
            )
            if phase_acceptance_attempts:
                response_payload = attach_phase_output_acceptance(
                    response_payload,
                    phase_acceptance_attempts,
                )
            response_payload = truth_gate_response_output_claims(
                response_payload,
                route_payload=route_info,
                request_payload=normalized_payload,
            )
            artifact_completion_gap = None
            response_payload, artifact_completion_gap = self.attach_pre_freeze_closure_review(
                response_payload,
                output_text=assistant_text,
                route_payload=route_info,
                request_payload=normalized_payload,
                artifact_gap=artifact_completion_gap,
            )
            if not artifact_completion_gap:
                response_payload, _direct_closure_status = self.apply_direct_artifact_materialization_closure(
                    response_payload,
                    request_payload=normalized_payload,
                    route_payload=route_info,
                    artifact_gap=artifact_completion_gap,
                    terminal_status='completed',
                )
            response_payload = finalize_response_frame_payload(
                response_payload,
                request_payload=normalized_payload,
            )
            mark_response_lookup_completed(response_payload, 'external_chat')
            if artifact_completion_gap:
                schedule_response_late_fill(
                    response_payload=response_payload,
                    request_payload=normalized_payload,
                    assistant_message=assistant_text,
                    artifact_gap=artifact_completion_gap,
                    source_route_payload=route_info,
                )
            if wants_stream:
                return Response(
                    stream_with_context(
                        build_canonical_response_stream_events(response_payload)
                    ),
                    mimetype='text/event-stream',
                )
            return jsonify(project_response_payload_for_wire(response_payload))

        if capability == self.capability_chat and not has_file_context:
            messages = extract_responses_messages(data)
            if not messages:
                return jsonify({'error': 'No Responses input was provided.'}), 400
            messages = inject_selected_reference_into_chat_messages(messages, selected_reference_artifacts)
            messages = inject_ghost_runtime_policy_into_chat_messages(
                messages,
                route_payload=route_info,
                request_payload=normalized_payload,
            )
            messages = inject_prepare_phase_contract_into_chat_messages(
                messages,
                route_payload=route_info,
                request_payload=normalized_payload,
            )
            context_strategy = choose_context_strategy(
                instance=instance,
                messages=messages,
                prompt=extract_responses_prompt(normalized_payload),
                has_file_context=False,
                conversation_id=(
                    normalized_payload.get('conversation_id')
                    or normalized_payload.get('conversationId')
                    or normalized_payload.get('thread_id')
                    or normalized_payload.get('threadId')
                ),
            )
            prepared_messages = apply_context_strategy(messages, context_strategy)
            if route_info:
                route_runtime = dict(route_info.get('route_runtime') or {})
                route_runtime['context_strategy'] = context_strategy
                route_info['route_runtime'] = route_runtime
            port = instance.get('port')
            if not port:
                return jsonify({'error': 'Instance has no target port.'}), 400
            try:
                target_port = int(port)
            except (TypeError, ValueError):
                return jsonify({'error': f"Invalid target port '{port}'."}), 400
            if wants_stream:
                return stream_chat_backend_as_responses(
                    instance_id=instance_id,
                    target_port=target_port,
                    model_name=model_name,
                    backend=backend,
                    capability=capability,
                    messages=prepared_messages,
                    request_model_override=request_model_override,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    route_payload=route_info,
                    response_id=requested_response_id,
                    request_payload=normalized_payload,
                    artifact_prompt=responses_current_turn_prompt or responses_prompt,
                )
            ensure_response_lookup('chat')
            try:
                assistant_message = execute_chat_backend_request(
                    target_port=target_port,
                    model_name=model_name,
                    backend=backend,
                    capability=capability,
                    messages=prepared_messages,
                    request_model_override=request_model_override,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except self.request_timeout_error:
                error_message = f'Timeout Port {target_port}'
                retried_response = maybe_retry_single_chat_transport_fallback(error_message, 504)
                if retried_response is not None:
                    return retried_response
                retried_response = maybe_retry_self_healed_response(error_message, 504)
                if retried_response is not None:
                    return retried_response
                error_payload = build_runtime_error_payload(error_message, 504, 'chat')
                mark_response_lookup_failed(
                    error_message,
                    'chat',
                    response_payload=error_payload,
                )
                return jsonify(project_response_payload_for_wire(error_payload)), 504
            except self.request_exception_error as exc:
                details = request_exception_details(exc)
                error_message = f'Request to port {target_port} failed: {details}'
                retried_response = maybe_retry_single_chat_transport_fallback(error_message, 500)
                if retried_response is not None:
                    return retried_response
                retried_response = maybe_retry_self_healed_response(error_message, 500)
                if retried_response is not None:
                    return retried_response
                error_payload = build_runtime_error_payload(error_message, 500, 'chat')
                mark_response_lookup_failed(
                    error_message,
                    'chat',
                    response_payload=error_payload,
                )
                return jsonify(project_response_payload_for_wire(error_payload)), 500
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                retried_response = maybe_retry_single_chat_transport_fallback(error_message, 500)
                if retried_response is not None:
                    return retried_response
                retried_response = maybe_retry_self_healed_response(error_message, 500)
                if retried_response is not None:
                    return retried_response
                error_payload = build_runtime_error_payload(error_message, 500, 'chat')
                mark_response_lookup_failed(
                    error_message,
                    'chat',
                    response_payload=error_payload,
                )
                return jsonify(project_response_payload_for_wire(error_payload)), 500
            assistant_text, phase_acceptance_attempts = accept_phase_output(
                assistant_message,
                effective_route_payload=route_info,
                effective_request_payload=normalized_payload,
                effective_capability=capability,
                retry_same_phase=lambda: execute_chat_backend_request(
                    target_port=target_port,
                    model_name=model_name,
                    backend=backend,
                    capability=capability,
                    messages=[*prepared_messages, phase_output_repair_system_message()],
                    request_model_override=request_model_override,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                ),
            )
            phase_output_repair_required = bool(
                phase_acceptance_attempts
                and phase_acceptance_attempts[-1].get('status') == 'repair_required'
            )
            text_artifact_payload = (
                {}
                if phase_output_repair_required
                else persist_generated_text_artifact_if_requested(
                    assistant_text,
                    prompt=responses_current_turn_prompt or responses_prompt,
                    model_name=model_name,
                    mode='responses_chat_text_artifact',
                    request_payload=normalized_payload,
                )
            )
            response_payload = build_canonical_response_payload(
                instance_id=instance_id,
                model_name=model_name,
                backend=backend,
                capability=capability,
                mode='chat',
                output_text=assistant_text,
                source_payload=attach_response_semantic_phase_payload(
                    text_artifact_payload,
                    output_text=assistant_text,
                    route_payload=route_info,
                    request_payload=normalized_payload,
                    capability=capability,
                ),
                route_payload=route_info,
                response_id=requested_response_id,
                message_id=response_lookup_message_id,
            )
            if phase_acceptance_attempts:
                response_payload = attach_phase_output_acceptance(
                    response_payload,
                    phase_acceptance_attempts,
                )
            response_payload = truth_gate_response_output_claims(
                response_payload,
                route_payload=route_info,
                request_payload=normalized_payload,
            )
            artifact_completion_gap = None
            response_payload, artifact_completion_gap = self.attach_pre_freeze_closure_review(
                response_payload,
                output_text=assistant_text,
                route_payload=route_info,
                request_payload=normalized_payload,
                artifact_gap=artifact_completion_gap,
            )
            _direct_closure_status = 'completed'
            if not artifact_completion_gap:
                response_payload, _direct_closure_status = self.apply_direct_artifact_materialization_closure(
                    response_payload,
                    request_payload=normalized_payload,
                    route_payload=route_info,
                    artifact_gap=artifact_completion_gap,
                    terminal_status='completed',
                )
            response_payload = finalize_response_frame_payload(response_payload, request_payload=normalized_payload)
            ensure_response_lookup_for_payload(response_payload, mode_hint='chat', route_payload=route_info)
            touch_response_lookup(
                str(response_payload.get('id') or '').strip(),
                status='completed',
                output_text=str(response_payload.get('output_text') or ''),
                response_payload=response_payload,
            )
            if artifact_completion_gap:
                schedule_response_late_fill(
                    response_payload=response_payload,
                    request_payload=normalized_payload,
                    assistant_message=assistant_text,
                    artifact_gap=artifact_completion_gap,
                    source_route_payload=route_info,
                )
            else:
                if str(response_payload.get('lifecycle_state') or response_payload.get('status') or '').strip().lower() == 'completed':
                    maybe_schedule_post_response_substrate_hygiene(
                        response_payload,
                        route_payload=route_info,
                        reason='responses_chat_completed',
                    )
            return jsonify(project_response_payload_for_wire(response_payload))

        infer_payload, route_info, _has_file_context, expose_input_artifacts = build_responses_infer_execution_payload(
            data,
            route_info=route_info,
            instance=instance if isinstance(instance, dict) else {},
            instance_id=instance_id,
            backend=backend,
            capability=capability,
            request_model_override=request_model_override,
            upload_present=bool(upload and getattr(upload, 'filename', None)),
        )
        if not batch_prompts and capability == self.capability_image_generation:
            direct_batch_count = self.resolve_direct_image_batch_count(
                normalized_payload,
                route_info,
                capability=capability,
            )
            if direct_batch_count > 1:
                batch_items = self.build_direct_image_batch_items(
                    normalized_payload,
                    route_info,
                    infer_payload,
                    direct_batch_count=direct_batch_count,
                )
                batch_prompts = [
                    str(item.get('prompt') or '').strip()
                    for item in batch_items
                    if str(item.get('prompt') or '').strip()
                ]
        if batch_prompts:
            ensure_response_lookup(f'{capability}_batch')
            try:
                branch_specs = self.build_direct_batch_branch_specs(
                    batch_items=batch_items,
                    capability=capability,
                    base_request_payload=normalized_payload,
                    base_route_info=route_info,
                    base_instance_id=instance_id,
                    base_instance=instance if isinstance(instance, dict) else None,
                    forced_instance_id=forced_instance_id,
                    upload=upload,
                )
            except ValueError as exc:
                error_message = str(exc)
                error_payload = build_runtime_error_payload(error_message, 400, f'{capability}_batch')
                mark_response_lookup_failed(
                    error_message,
                    f'{capability}_batch',
                    response_payload=error_payload,
                )
                return jsonify(project_response_payload_for_wire(error_payload)), 400

            materialization_result = execute_materialization_branches(
                branch_specs,
                prepare_branch_plan=self.prepare_direct_materialization_branch_plan,
                execute_prepared_branch=self.execute_prepared_materialization_branch,
            )
            ordered_branch_results = (
                materialization_result.get('ordered_branch_results')
                if isinstance(materialization_result.get('ordered_branch_results'), list)
                else []
            )
            ordered_branch_errors = (
                materialization_result.get('ordered_branch_errors')
                if isinstance(materialization_result.get('ordered_branch_errors'), list)
                else []
            )
            first_branch_error = (
                ordered_branch_errors[0]
                if ordered_branch_errors and isinstance(ordered_branch_errors[0], dict)
                else None
            )
            if first_branch_error:
                error_message = str(
                    (
                        first_branch_error.get('error', {}) if isinstance(first_branch_error.get('error'), dict) else {}
                    ).get('message')
                    or 'Request failed.'
                ).strip() or 'Request failed.'
                try:
                    error_status_code = int(
                        (
                            first_branch_error.get('error', {}) if isinstance(first_branch_error.get('error'), dict) else {}
                        ).get('status_code')
                        or 500
                    )
                except (TypeError, ValueError):
                    error_status_code = 500
                retried_response = maybe_retry_self_healed_response(error_message, error_status_code)
                if retried_response is not None:
                    return retried_response
                error_payload = build_runtime_error_payload(error_message, error_status_code, f'{capability}_batch')
                mark_response_lookup_failed(
                    error_message,
                    f'{capability}_batch',
                    response_payload=error_payload,
                )
                return jsonify(project_response_payload_for_wire(error_payload)), error_status_code

            infer_results = [
                result_entry.get('result', {}).get('infer_result')
                for result_entry in ordered_branch_results
                if isinstance(result_entry, dict)
                and isinstance(result_entry.get('result'), dict)
                and isinstance(result_entry.get('result', {}).get('infer_result'), dict)
            ]

            response_payload = build_canonical_batch_response_payload(
                instance_id=instance_id,
                model_name=model_name,
                backend=backend,
                capability=capability,
                batch_mode=f'{capability}_batch',
                batch_prompts=batch_prompts,
                infer_results=infer_results,
                route_payload=route_info,
                response_id=requested_response_id,
                message_id=response_lookup_message_id,
            )
            artifact_completion_gap = None
            route_runtime_for_gateway = (
                route_info.get('route_runtime')
                if isinstance(route_info, dict) and isinstance(route_info.get('route_runtime'), dict)
                else {}
            )
            route_graph_for_gateway = (
                route_runtime_for_gateway.get('request_phase_graph')
                if isinstance(route_runtime_for_gateway.get('request_phase_graph'), dict)
                else {}
            )
            batch_declares_downstream_work = bool(
                route_graph_for_gateway
                and (
                    route_graph_for_gateway.get('continuation_required')
                    or route_graph_for_gateway.get('downstream_branches')
                    or route_graph_for_gateway.get('downstream_branch_ids')
                    or route_graph_for_gateway.get('downstream_phase_ids')
                )
            )
            if batch_declares_downstream_work:
                response_payload, artifact_completion_gap = self.attach_pre_freeze_closure_review(
                    response_payload,
                    output_text=str(response_payload.get('output_text') or ''),
                    route_payload=route_info,
                    request_payload=normalized_payload,
                    artifact_gap=artifact_completion_gap,
                )
            response_payload = finalize_response_frame_payload(response_payload, request_payload=normalized_payload)
            mark_response_lookup_completed(response_payload, f'{capability}_batch')
            if artifact_completion_gap:
                schedule_response_late_fill(
                    response_payload=response_payload,
                    request_payload=normalized_payload,
                    assistant_message=str(response_payload.get('output_text') or ''),
                    artifact_gap=artifact_completion_gap,
                    source_route_payload=route_info,
                )
            else:
                maybe_schedule_post_response_substrate_hygiene(
                    response_payload,
                    route_payload=route_info,
                    reason=f'{capability}_batch_completed',
                )
            if wants_stream:
                return Response(
                    stream_with_context(build_canonical_response_stream_events(response_payload)),
                    mimetype='text/event-stream',
                )
            return jsonify(project_response_payload_for_wire(response_payload))

        ensure_response_lookup(capability or 'response')
        infer_result, status_code = invoke_internal_api_json_route(
            payload=infer_payload,
            upload=upload,
        )
        if status_code >= 400:
            error_message = str(infer_result.get('error') or 'Request failed.')
            retried_response = maybe_retry_self_healed_response(error_message, status_code)
            if retried_response is not None:
                return retried_response
            error_payload = build_runtime_error_payload(error_message, status_code, capability or 'response')
            mark_response_lookup_failed(
                error_message,
                capability or 'response',
                response_payload=error_payload,
            )
            return jsonify(project_response_payload_for_wire(error_payload)), status_code
        infer_result = filter_responses_infer_result(
            infer_result,
            expose_input_artifacts=expose_input_artifacts,
        )
        infer_result = self._attach_tts_audio_integrity_evidence(
            infer_result,
            capability=capability,
            infer_payload=infer_payload,
        )

        final_output_text = str(infer_result.get('content') or '').strip()
        speech_to_text_translation_follow_up = None
        if capability == self.capability_speech_to_text:
            final_output_text, speech_to_text_translation_follow_up = (
                self._maybe_apply_speech_to_text_translation_follow_up(
                    transcript_text=final_output_text,
                    request_payload=normalized_payload,
                    capability=capability,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            )
        retry_infer_result: dict[str, Any] = {}

        def retry_infer_phase_output() -> str:
            retry_payload = copy.deepcopy(infer_payload)
            repair_instruction = phase_output_repair_system_message()['content']
            retry_messages = retry_payload.get('messages')
            if isinstance(retry_messages, list):
                retry_payload['messages'] = [
                    *[dict(item) for item in retry_messages if isinstance(item, dict)],
                    phase_output_repair_system_message(),
                ]
            else:
                retry_prompt = str(retry_payload.get('prompt') or '').strip()
                retry_payload['prompt'] = (
                    f'{retry_prompt}\n\n[{repair_instruction}]'
                    if retry_prompt
                    else repair_instruction
                )
            retried_result, retried_status = invoke_internal_api_json_route(
                payload=retry_payload,
                upload=upload,
            )
            if retried_status >= 400:
                raise RuntimeError(str(retried_result.get('error') or 'Phase-output repair failed.'))
            filtered_retry = filter_responses_infer_result(
                retried_result,
                expose_input_artifacts=expose_input_artifacts,
            )
            retry_infer_result.clear()
            retry_infer_result.update(filtered_retry)
            return str(filtered_retry.get('content') or '').strip()

        final_output_text, phase_acceptance_attempts = accept_phase_output(
            final_output_text,
            effective_route_payload=route_info,
            effective_request_payload=normalized_payload,
            effective_capability=capability,
            retry_same_phase=retry_infer_phase_output,
        )
        if (
            retry_infer_result
            and phase_acceptance_attempts
            and phase_acceptance_attempts[-1].get('status') in {'accepted', 'unwrapped'}
        ):
            infer_result = dict(retry_infer_result)
            infer_result['content'] = final_output_text
        phase_output_repair_required = bool(
            phase_acceptance_attempts
            and phase_acceptance_attempts[-1].get('status') == 'repair_required'
        )
        if (
            capability == self.capability_chat
            and final_output_text
            and not phase_output_repair_required
            and not str(infer_result.get('saved_text_path') or infer_result.get('savedTextPath') or '').strip()
            and not infer_result.get('saved_text_artifacts')
        ):
            text_artifact_payload = persist_generated_text_artifact_if_requested(
                final_output_text,
                prompt=responses_current_turn_prompt or responses_prompt,
                model_name=model_name,
                mode='responses_infer_text_artifact',
                request_payload=normalized_payload,
            )
            if text_artifact_payload:
                infer_result = {
                    **infer_result,
                    **text_artifact_payload,
                    'content': final_output_text,
                }
        semantic_source_payload = (
            self._apply_tts_audio_integrity_output_truth(dict(infer_result))
        )
        semantic_source_payload['content'] = final_output_text
        response_payload = build_canonical_response_payload(
            instance_id=instance_id,
            model_name=model_name,
            backend=backend,
            capability=capability,
            mode=str(infer_result.get('mode') or capability or 'response'),
            output_text=final_output_text,
            source_payload=attach_response_semantic_phase_payload(
                semantic_source_payload,
                output_text=final_output_text,
                route_payload=route_info,
                request_payload=normalized_payload,
                capability=capability,
            ),
            route_payload=route_info,
            response_id=requested_response_id,
            message_id=response_lookup_message_id,
        )
        response_payload = self._apply_tts_audio_integrity_output_truth(
            response_payload
        )
        if phase_acceptance_attempts:
            response_payload = attach_phase_output_acceptance(
                response_payload,
                phase_acceptance_attempts,
            )
        response_payload = truth_gate_response_output_claims(
            response_payload,
            route_payload=route_info,
            request_payload=normalized_payload,
        )
        artifact_completion_gap = None
        response_payload, artifact_completion_gap = self.attach_pre_freeze_closure_review(
            response_payload,
            output_text=final_output_text,
            route_payload=route_info,
            request_payload=normalized_payload,
            artifact_gap=artifact_completion_gap,
        )
        if not artifact_completion_gap:
            response_payload, _direct_closure_status = self.apply_direct_artifact_materialization_closure(
                response_payload,
                request_payload=normalized_payload,
                route_payload=route_info,
                artifact_gap=artifact_completion_gap,
                terminal_status='completed',
            )
        if isinstance(speech_to_text_translation_follow_up, dict):
            runtime = (
                dict(response_payload.get('runtime') or {})
                if isinstance(response_payload.get('runtime'), dict)
                else {}
            )
            runtime['speech_to_text_translation_follow_up'] = dict(
                speech_to_text_translation_follow_up
            )
            response_payload['runtime'] = runtime
        response_payload = finalize_response_frame_payload(response_payload, request_payload=normalized_payload)
        mark_response_lookup_completed(
            response_payload,
            str(infer_result.get('mode') or capability or 'response'),
        )
        if artifact_completion_gap:
            schedule_response_late_fill(
                response_payload=response_payload,
                request_payload=normalized_payload,
                assistant_message=final_output_text,
                artifact_gap=artifact_completion_gap,
                source_route_payload=route_info,
            )
        else:
            if str(response_payload.get('lifecycle_state') or response_payload.get('status') or '').strip().lower() == 'completed':
                maybe_schedule_post_response_substrate_hygiene(
                    response_payload,
                    route_payload=route_info,
                    reason=f'{capability}_completed',
                )
        if wants_stream:
            return Response(
                stream_with_context(build_canonical_response_stream_events(response_payload)),
                mimetype='text/event-stream',
            )
        return jsonify(project_response_payload_for_wire(response_payload))
