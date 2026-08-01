#!/usr/bin/env python3
"""Manual external-client provider unsync for Ollmo-managed client projections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_integrations.registry import integration_module, list_integration_adapters


def available_integration_ids() -> list[str]:
    return [adapter.integration_id for adapter in list_integration_adapters()]


def unsync_integrations(
    integration_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected_ids = [
        str(item or '').strip().lower()
        for item in (integration_ids if integration_ids is not None else available_integration_ids())
        if str(item or '').strip()
    ]
    summary: dict[str, Any] = {}
    for integration_id in selected_ids:
        module = integration_module(integration_id, 'unsync')
        if integration_id == 'codex':
            changed = bool(module.unsync_codex_config())
            summary[integration_id] = {'changed': changed}
            continue
        raise KeyError(f'Unsupported integration unsync handler: {integration_id}')
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Unsync Ollmo-managed external-client provider projections.')
    parser.add_argument(
        '--integration',
        dest='integration_ids',
        action='append',
        choices=available_integration_ids(),
        help='Limit unsync to one integration. Repeat to select more than one.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = unsync_integrations(args.integration_ids)
    if 'codex' in summary:
        if summary['codex'].get('changed'):
            print('✅ Codex provider projection unsynced.')
        else:
            print('ℹ️ Codex provider projection bereits unsynced.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
