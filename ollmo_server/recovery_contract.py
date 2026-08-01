"""Shared recovery action contract for runtime-visible repair guidance."""

from __future__ import annotations

from typing import Any


RECOVERY_ACTION_MANUAL_REVIEW = 'manual_review'
RECOVERY_ACTION_START_COMPATIBLE_INSTANCE = 'start_compatible_instance'
RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE = 'retry_excluding_instance'
RECOVERY_ACTION_RETRY_SAME_BRANCH = 'retry_same_branch'
RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE = 'rebind_dependency_evidence'
RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN = 'repair_dependency_chain'
RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT = 'repair_branch_contract'
RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS = 'rebuild_from_promoted_obligations'
RECOVERY_ACTION_SEMANTIC_REVIEW = 'semantic_review'

RECOVERY_SUGGESTED_ACTIONS = frozenset(
    {
        RECOVERY_ACTION_MANUAL_REVIEW,
        RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
        RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
        RECOVERY_ACTION_RETRY_SAME_BRANCH,
        RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
        RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
        RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
        RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
        RECOVERY_ACTION_SEMANTIC_REVIEW,
    }
)


def normalize_recovery_suggested_action(
    value: Any,
    *,
    default: str = RECOVERY_ACTION_MANUAL_REVIEW,
) -> str:
    action = str(value or '').strip().lower()
    if action in RECOVERY_SUGGESTED_ACTIONS:
        return action
    return default if default in RECOVERY_SUGGESTED_ACTIONS else RECOVERY_ACTION_MANUAL_REVIEW
