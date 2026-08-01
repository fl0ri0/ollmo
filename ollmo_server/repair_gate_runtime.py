"""Shared repair gate classification for runtime repair transitions."""

from __future__ import annotations

from typing import Any, Mapping

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_VISION_ANALYSIS,
    normalize_capability,
)
from ollmo_server.recovery_contract import (
    RECOVERY_ACTION_MANUAL_REVIEW,
    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
    RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
    RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
    RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
    RECOVERY_ACTION_SEMANTIC_REVIEW,
    RECOVERY_SUGGESTED_ACTIONS,
)


def _text(value: Any) -> str:
    return str(value or '').strip()


def _normalized_repair_action(value: Any) -> str:
    action = _text(value).lower()
    return action if action in RECOVERY_SUGGESTED_ACTIONS else ''


def _list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _contract_output_type(contract: Mapping[str, Any]) -> str:
    output_type = _text(contract.get('output_type')).lower()
    if output_type:
        return output_type
    output_contract = _mapping(contract.get('output_contract'))
    return _text(output_contract.get('output_type')).lower()


def _item_identity_available(item: Mapping[str, Any]) -> bool:
    contract = _mapping(item.get('execution_contract'))
    return any(
        _text(source.get(key))
        for source in (item, contract)
        for key in (
            'branch_id',
            'phase_id',
            'obligation_id',
            'task_id',
            'workload_task_id',
        )
    ) or bool(item.get('workload_task_ref') or item.get('output_obligation_ref'))


def _item_output_type(item: Mapping[str, Any]) -> str:
    output_type = _text(item.get('output_type')).lower()
    if output_type:
        return output_type
    output_contract = _mapping(item.get('output_contract'))
    output_type = _text(output_contract.get('output_type')).lower()
    if output_type:
        return output_type
    return _contract_output_type(_mapping(item.get('execution_contract')))


def repair_item_has_concrete_input_evidence(item: Mapping[str, Any]) -> bool:
    """Return true when a repair item already carries concrete input evidence."""

    if _text(item.get('file_path')):
        return True
    if _text(item.get('content_payload')):
        return True
    for key in ('input_artifacts', 'reference_artifacts'):
        value = item.get(key)
        if isinstance(value, list) and any(isinstance(entry, Mapping) for entry in value):
            return True
    for ref in _list_values(item.get('input_refs')):
        if not isinstance(ref, Mapping):
            continue
        if any(
            _text(ref.get(key))
            for key in (
                'artifact_ref',
                'ref',
                'path',
                'file_path',
                'saved_image_path',
                'saved_audio_path',
                'saved_text_path',
                'content_payload',
                'result_text',
            )
        ):
            return True
    return False


def repair_item_has_bounded_execution_contract(item: Mapping[str, Any]) -> bool:
    """Return true when a branch has enough execution contract to run safely."""

    contract = _mapping(item.get('execution_contract'))
    if not contract:
        return False
    capability = normalize_capability(item.get('capability')) or normalize_capability(contract.get('capability'))
    output_type = _item_output_type(item)
    has_identity = _item_identity_available(item) or any(
        _text(contract.get(key))
        for key in ('branch_id', 'phase_id', 'obligation_id', 'task_id', 'workload_task_id')
    )
    return bool(has_identity or capability or output_type or contract.get('output_contract'))


def repair_item_has_branch_local_payload(item: Mapping[str, Any]) -> bool:
    """Return true when a repair branch can execute without replaying root intent."""

    if _text(item.get('artifact_prompt')):
        return True
    if _text(item.get('content_payload')):
        return True
    if _text(item.get('file_path')):
        return True
    batch_prompts = _list_values(item.get('batch_prompts'))
    if any(_text(prompt) for prompt in batch_prompts):
        return True
    if item.get('artifact_request') not in (None, '', [], {}):
        return True
    if any(
        _text(item.get(key))
        for key in (
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
        )
    ):
        return True
    if _text(item.get('stage_direction')) == 'materialize_requested_text_artifact':
        return True
    contract = _mapping(item.get('execution_contract'))
    if _text(contract.get('execution_scope')) == 'root_scoped':
        return True
    return False


def _dependency_repair_source_available(item: Mapping[str, Any]) -> bool:
    return bool(
        item.get('input_refs')
        or item.get('depends_on')
        or item.get('workload_task_ref')
        or item.get('output_obligation_ref')
        or item.get('execution_contract')
        or item.get('output_contract')
        or _text(item.get('phase_id'))
        or _text(item.get('branch_id'))
    )


def _branch_contract_source_available(item: Mapping[str, Any]) -> bool:
    return bool(
        normalize_capability(item.get('capability'))
        and (
            _text(item.get('branch_id'))
            or _text(item.get('phase_id'))
            or _text(item.get('obligation_id'))
            or item.get('workload_task_ref')
            or item.get('output_obligation_ref')
            or item.get('output_contract')
            or _text(item.get('content_payload'))
            or _text(item.get('artifact_prompt'))
        )
    )


def _promoted_obligation_source_available(item: Mapping[str, Any]) -> bool:
    return bool(
        _item_identity_available(item)
        or item.get('workload_task_ref')
        or item.get('output_obligation_ref')
        or item.get('output_contract')
        or item.get('execution_contract')
        or _text(item.get('capability'))
        or _text(item.get('output_type'))
    )


def _blocked_state(
    *,
    execution_policy: str,
    blocked_prerequisite: str,
    repair_work_available: bool,
    repair_work_policy: str | None = None,
    blocked_scope: str = 'target_materialization',
    needs_external_input: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'execution_policy': execution_policy,
        'auto_execute': False,
        'materialization_blocked': True,
        'blocked_scope': blocked_scope,
        'blocked_prerequisite': blocked_prerequisite,
        'repair_work_available': bool(repair_work_available),
        'needs_external_input': (not repair_work_available)
        if needs_external_input is None
        else bool(needs_external_input),
    }
    if repair_work_policy:
        payload['repair_work_policy'] = repair_work_policy
    return payload


def _scheduled_state() -> dict[str, Any]:
    return {
        'execution_policy': 'schedule_late_fill_branch',
        'auto_execute': True,
        'materialization_blocked': False,
        'blocked_scope': None,
        'repair_work_available': True,
        'needs_external_input': False,
    }


def _manual_review_state() -> dict[str, Any]:
    return {
        'execution_policy': 'manual_review_required',
        'auto_execute': False,
        'materialization_blocked': True,
        'blocked_scope': 'review_gate',
        'repair_work_available': False,
        'needs_external_input': True,
    }


def _semantic_review_state() -> dict[str, Any]:
    return {
        'execution_policy': 'semantic_review_required',
        'auto_execute': False,
        'materialization_blocked': True,
        'blocked_scope': 'review_gate',
        'blocked_prerequisite': 'semantic_review_contract',
        'repair_work_available': False,
        'needs_external_input': False,
    }


def classify_repair_execution_policy(item: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a repair item as executable work or a still-blocked transition."""

    action = _normalized_repair_action(item.get('repair_action') or item.get('recovery_action'))
    existing_policy = _text(item.get('execution_policy'))
    capability = normalize_capability(item.get('capability'))
    depends_on = [_text(value) for value in _list_values(item.get('depends_on')) if _text(value)]
    input_refs = _list_values(item.get('input_refs'))
    has_concrete_input_evidence = repair_item_has_concrete_input_evidence(item)
    has_bounded_contract = repair_item_has_bounded_execution_contract(item)
    has_branch_payload = repair_item_has_branch_local_payload(item)

    if action == RECOVERY_ACTION_MANUAL_REVIEW:
        return _manual_review_state()

    if action == RECOVERY_ACTION_SEMANTIC_REVIEW:
        if (
            capability == CAPABILITY_CHAT
            and (_text(item.get('content_payload')) or has_bounded_contract or input_refs)
        ):
            return _scheduled_state()
        return _semantic_review_state()

    if (
        (
            action in {
                RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
                RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
            }
            or existing_policy == 'blocked_until_dependency_evidence'
            or item.get('blocked_by_dependency_input') is True
        )
        and not has_concrete_input_evidence
    ):
        repair_source_available = _dependency_repair_source_available(item)
        return _blocked_state(
            execution_policy='blocked_until_dependency_evidence',
            blocked_prerequisite='dependency_evidence',
            repair_work_available=repair_source_available,
            repair_work_policy=(
                'rebind_dependency_evidence_before_materialization'
                if action == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE
                else 'repair_dependency_chain_before_materialization'
            ),
        )

    if (
        (
            action == RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT
            or existing_policy == 'blocked_until_branch_contract'
            or item.get('blocked_by_branch_contract') is True
        )
        and not has_bounded_contract
    ):
        repair_source_available = _branch_contract_source_available(item)
        return _blocked_state(
            execution_policy='blocked_until_branch_contract',
            blocked_prerequisite='branch_execution_contract',
            repair_work_available=repair_source_available,
            repair_work_policy='build_branch_contract_before_materialization',
        )

    if action == RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS:
        if (
            capability in {CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_VISION_ANALYSIS}
            and not depends_on
            and not has_concrete_input_evidence
        ):
            repair_source_available = _dependency_repair_source_available(item)
            return _blocked_state(
                execution_policy='blocked_until_dependency_evidence',
                blocked_prerequisite='dependency_evidence',
                repair_work_available=repair_source_available,
                repair_work_policy='repair_dependency_chain_before_materialization',
            )
        output_type = _item_output_type(item)
        output_contract = _mapping(item.get('output_contract'))
        concrete_branch = bool(
            capability
            and output_type
            and _item_identity_available(item)
            and (
                has_branch_payload
                or has_bounded_contract
                or bool(output_contract)
                or bool(depends_on)
            )
        )
        if concrete_branch:
            if (
                capability in {CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_VISION_ANALYSIS}
                and not depends_on
                and not has_concrete_input_evidence
            ):
                repair_source_available = _dependency_repair_source_available(item)
                return _blocked_state(
                    execution_policy='blocked_until_dependency_evidence',
                    blocked_prerequisite='dependency_evidence',
                    repair_work_available=repair_source_available,
                    repair_work_policy='repair_dependency_chain_before_materialization',
                )
            return _scheduled_state()
        repair_source_available = _promoted_obligation_source_available(item)
        return _blocked_state(
            execution_policy='blocked_until_promoted_obligation_branch',
            blocked_prerequisite='promoted_obligation_branch',
            repair_work_available=repair_source_available,
            repair_work_policy='rebuild_promoted_obligation_branch_before_materialization',
        )

    if (
        capability in {CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_VISION_ANALYSIS}
        and not depends_on
        and not has_concrete_input_evidence
    ):
        repair_source_available = _dependency_repair_source_available(item)
        return _blocked_state(
            execution_policy='blocked_until_dependency_evidence',
            blocked_prerequisite='dependency_evidence',
            repair_work_available=repair_source_available,
            repair_work_policy='repair_dependency_chain_before_materialization',
        )

    return _scheduled_state()
