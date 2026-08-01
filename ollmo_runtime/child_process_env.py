"""Shared environment boundary for Ollmo child processes."""

from __future__ import annotations

import os
from collections.abc import Mapping


GRAPH_REBASE_OPERATOR_ENV_KEYS = (
    'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN',
    'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY',
    'GRAPH_REBASE_OPERATOR_TOKEN',
    'GRAPH_REBASE_OPERATOR_IDENTITY',
)


def sanitized_child_process_env(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment without graph-rebase operator credentials."""

    env = dict(os.environ if source is None else source)
    for key in GRAPH_REBASE_OPERATOR_ENV_KEYS:
        env.pop(key, None)
    return env
