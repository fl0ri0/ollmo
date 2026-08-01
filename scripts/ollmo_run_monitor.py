#!/usr/bin/env python3
"""Read-only Ollmo run monitor.

Observer only: reads local response truth, appends reports, and never starts work.
Generated reports remain local under ``state/ollmo_run_monitor``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
IDLE_AUTO_PAUSE_CHECKS = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value[:-1] + '+00:00' if value.endswith('Z') else value
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _response_start(response_id: str) -> datetime | None:
    parts = response_id.split('_')
    if len(parts) < 3:
        return None
    try:
        return datetime.fromtimestamp(int(parts[1]) / 1000.0, timezone.utc)
    except ValueError:
        return None


def _fmt_time(dt: datetime | None) -> str:
    if not dt:
        return 'unknown'
    return dt.strftime('%H:%M:%SZ')


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return 'unknown'
    if seconds < 90:
        return f'{seconds:.1f}s'
    minutes, sec = divmod(int(round(seconds)), 60)
    return f'{minutes}m{sec:02d}s'


def _seconds_from_ms(value: Any) -> float:
    try:
        return float(value or 0) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f'{singular}s')
    return f'{count} {word}'


def _count_values(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        token = str(value or '').strip()
        if not token:
            continue
        counts[token] = counts.get(token, 0) + 1
    return counts


def _format_counts(counts: dict[str, int] | None, *, limit: int = 8) -> str:
    if not counts:
        return 'none'
    parts = [
        f'{key}={counts[key]}'
        for key in sorted(counts.keys(), key=str)[:limit]
    ]
    if len(counts) > limit:
        parts.append(f'+{len(counts) - limit} more')
    return ', '.join(parts) if parts else 'none'


def _artifact_kind_token(value: Any) -> str:
    token = re.sub(r'[\s-]+', '_', str(value or '').strip().lower())
    return token or 'unknown'


def _artifact_suffix_kind(path: str | None) -> str | None:
    if not path:
        return None
    suffix = Path(path).suffix.lower().lstrip('.')
    return suffix or None


def _is_audio_kind(value: Any) -> bool:
    token = _artifact_kind_token(value)
    return token in {
        'audio',
        'wav',
        'mp3',
        'm4a',
        'aac',
        'ogg',
        'opus',
        'flac',
        'text_to_speech',
        'speech',
    }


def _prepared_branch_clause(count: int) -> str:
    if count == 1:
        return 'only 1 branch was prepared'
    return f'only {count} branches were prepared'


def _fmt_size(bytes_count: int | None) -> str:
    if bytes_count is None:
        return 'unknown'
    units = ['B', 'K', 'M', 'G']
    value = float(bytes_count)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == 'B':
                return f'{int(value):,}B'
            if value >= 100:
                return f'{value:.0f}{unit}'
            if value >= 10:
                return f'{value:.1f}{unit}'
            return f'{value:.2f}{unit}'
        value /= 1024.0
    return f'{bytes_count:,}B'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open('rb') as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                yield line_no, line, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, line, None


def _walk_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob('*'):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _load_response_helpers(root: Path):
    sys.path.insert(0, str(root))
    from ollmo_services.response_frames import load_latest_response_state

    try:
        from ollmo_services.response_frames import _read_snapshot_ref_payload
    except Exception:
        _read_snapshot_ref_payload = None

    try:
        from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner
    except Exception:
        ResponseSemanticsRuntimeOwner = None
    return load_latest_response_state, ResponseSemanticsRuntimeOwner, _read_snapshot_ref_payload


def _scan_responses(root: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    ledger = root / 'state/response_frames/responses.jsonl'
    order: list[str] = []
    latest: dict[str, dict[str, int]] = {}
    if not ledger.exists():
        return order, latest
    for line_no, line, record in _iter_jsonl(ledger):
        if not isinstance(record, dict):
            continue
        payload = record.get('response_payload') or record
        response_id = (
            record.get('response_id')
            or record.get('id')
            or payload.get('id')
        )
        if not response_id:
            continue
        if response_id not in latest:
            order.append(response_id)
        latest[response_id] = {'line': line_no, 'bytes': len(line)}
    return order, latest


def _events_for_window(root: Path, start: datetime, end: datetime | None) -> list[tuple[int, dict[str, Any], int]]:
    events: list[tuple[int, dict[str, Any], int]] = []
    for line_no, line, record in _iter_jsonl(root / 'state/events.jsonl'):
        if not isinstance(record, dict):
            continue
        dt = _parse_iso(record.get('timestamp'))
        if not dt or dt < start:
            continue
        if end and dt >= end:
            continue
        events.append((line_no, record, len(line)))
    return events


def _registry_records(root: Path, response_id: str) -> tuple[bool, list[tuple[int, dict[str, Any], int]]]:
    parse_clean = True
    records: list[tuple[int, dict[str, Any], int]] = []
    registry = root / 'state/artifact_registry.jsonl'
    if not registry.exists():
        return False, records
    needle = response_id.encode('utf-8')
    with registry.open('rb') as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_clean = False
                continue
            if needle in line:
                records.append((line_no, record, len(line)))
    return parse_clean, records


def _artifact_entry(
    *,
    source: str,
    artifact_ref: Any,
    path: Any = None,
    kind: Any = None,
    line: int | None = None,
) -> dict[str, Any] | None:
    ref = str(artifact_ref or '').strip()
    if not ref:
        return None
    raw_path = str(path or '').strip()
    raw_kind = str(kind or '').strip()
    if not raw_kind and raw_path:
        raw_kind = _artifact_suffix_kind(raw_path) or ''
    return {
        'source': source,
        'artifact_ref': ref,
        'path': raw_path or None,
        'kind': raw_kind or 'unknown',
        'line': line,
    }


def _collect_output_artifact_entries(value: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(item: Any, inherited_kind: Any = None, inherited_path: Any = None) -> None:
        if isinstance(item, dict):
            kind = (
                item.get('kind')
                or item.get('type')
                or item.get('output_type')
                or item.get('capability')
                or inherited_kind
            )
            path = (
                item.get('path')
                or item.get('file_path')
                or item.get('audio_path')
                or item.get('image_path')
                or inherited_path
            )
            for ref_key in ('artifact_ref', 'artifactRef', 'ref'):
                entry = _artifact_entry(
                    source='output',
                    artifact_ref=item.get(ref_key),
                    path=path,
                    kind=kind,
                )
                if entry:
                    entries.append(entry)
                    break
            for key, child in item.items():
                if key in {'artifact_ref', 'artifactRef', 'ref'}:
                    continue
                visit(child, kind, path)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, inherited_kind, inherited_path)
            return
        if isinstance(item, str) and item.startswith('artifact:'):
            entry = _artifact_entry(
                source='output',
                artifact_ref=item,
                path=inherited_path,
                kind=inherited_kind,
            )
            if entry:
                entries.append(entry)

    visit(value)
    return entries


def _classify_duplicate_ref_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        ref = str(entry.get('artifact_ref') or '').strip()
        if not ref:
            continue
        grouped.setdefault(ref, []).append(entry)

    duplicates: list[dict[str, Any]] = []
    for ref, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        signatures = {
            (
                str(item.get('path') or ''),
                _artifact_kind_token(item.get('kind')),
            )
            for item in items
        }
        classification = 'canonicalizable' if len(signatures) == 1 else 'conflict'
        reason = (
            'same ref, same path/type'
            if classification == 'canonicalizable'
            else 'same ref, different path/type'
        )
        duplicates.append(
            {
                'artifact_ref': ref,
                'count': len(items),
                'classification': classification,
                'reason': reason,
                'paths': sorted({str(item.get('path') or 'unknown') for item in items}),
                'kinds': sorted({_artifact_kind_token(item.get('kind')) for item in items}),
                'sources': sorted({str(item.get('source') or 'unknown') for item in items}),
            }
        )
    return duplicates


def _duplicate_ref_summary(
    final_entries: list[dict[str, Any]],
    registry_entries: list[dict[str, Any]],
    output_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    by_source = {
        'final_projection': _classify_duplicate_ref_entries(final_entries),
        'registry': _classify_duplicate_ref_entries(registry_entries),
        'output': _classify_duplicate_ref_entries(output_entries),
    }
    all_duplicates = [
        {**item, 'source': source}
        for source, items in by_source.items()
        for item in items
    ]
    return {
        **by_source,
        'canonicalizable': [
            item for item in all_duplicates
            if item.get('classification') == 'canonicalizable'
        ],
        'conflicts': [
            item for item in all_duplicates
            if item.get('classification') == 'conflict'
        ],
    }


def _format_duplicate_item(item: dict[str, Any]) -> str:
    ref = item.get('artifact_ref') or 'unknown'
    reason = item.get('reason') or 'unknown'
    paths = ', '.join(item.get('paths') or ['unknown'])
    kinds = ', '.join(item.get('kinds') or ['unknown'])
    return f'{ref} ({reason}; paths {paths}; types {kinds})'


def _duplicate_ref_lines(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return []
    if not (summary.get('final_projection') or summary.get('registry') or summary.get('output')):
        return []
    source_labels = {
        'final_projection': 'duplicate final projection',
        'registry': 'duplicate registry ref',
        'output': 'duplicate output ref',
    }
    source_bits = [
        f"{label} {'yes' if summary.get(key) else 'no'}"
        for key, label in source_labels.items()
    ]
    lines = ['- Artifact identity duplicates: ' + '; '.join(source_bits) + '.']
    source_items = [
        item
        for key in ('final_projection', 'registry', 'output')
        for item in (summary.get(key) or [])
    ]
    canonicalizable = summary.get('canonicalizable') or [
        item for item in source_items
        if item.get('classification') == 'canonicalizable'
    ]
    conflicts = summary.get('conflicts') or [
        item for item in source_items
        if item.get('classification') == 'conflict'
    ]
    if canonicalizable:
        lines.append(
            '- Artifact identity canonicalizable: '
            + '; '.join(_format_duplicate_item(item) for item in canonicalizable[:4])
            + (f'; +{len(canonicalizable) - 4} more' if len(canonicalizable) > 4 else '')
            + '.'
        )
    if conflicts:
        lines.append(
            '- Artifact identity conflicts: '
            + '; '.join(_format_duplicate_item(item) for item in conflicts[:4])
            + (f'; +{len(conflicts) - 4} more' if len(conflicts) > 4 else '')
            + '.'
        )
    return lines


def _artifact_checks(payload: dict[str, Any], registry_records: list[tuple[int, dict[str, Any], int]], syntax_owner: Any) -> dict[str, Any]:
    paths: list[Path] = []
    artifact_refs: list[str] = []
    final_artifact_entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact in payload.get('artifacts') or []:
        ref = artifact.get('artifact_ref')
        if ref:
            artifact_refs.append(ref)
        entry = _artifact_entry(
            source='final_projection',
            artifact_ref=ref,
            path=artifact.get('path'),
            kind=artifact.get('kind') or artifact.get('type') or artifact.get('output_type'),
        )
        if entry:
            final_artifact_entries.append(entry)
        raw_path = artifact.get('path')
        if raw_path and raw_path not in seen_paths:
            seen_paths.add(raw_path)
            paths.append(Path(raw_path))

    registry_by_path: dict[str, dict[str, Any]] = {}
    registry_parse_records = []
    registry_artifact_entries: list[dict[str, Any]] = []
    for line_no, record, _ in registry_records:
        artifact = record.get('artifact') or {}
        metadata = record.get('metadata') or artifact.get('metadata') or {}
        path = artifact.get('path')
        if path:
            registry_by_path[path] = metadata
        entry = _artifact_entry(
            source='registry',
            artifact_ref=artifact.get('artifact_ref'),
            path=path,
            kind=artifact.get('kind') or artifact.get('type'),
            line=line_no,
        )
        if entry:
            registry_artifact_entries.append(entry)
        registry_parse_records.append(
            {
                'line': line_no,
                'kind': artifact.get('kind') or artifact.get('type'),
                'path': path,
                'artifact_ref': artifact.get('artifact_ref'),
            }
        )

    output_artifact_entries = [
        *_collect_output_artifact_entries(payload.get('outputs')),
        *_collect_output_artifact_entries(payload.get('output')),
    ]
    duplicate_summary = _duplicate_ref_summary(
        final_artifact_entries,
        registry_artifact_entries,
        output_artifact_entries,
    )

    file_checks = []
    sha_mismatches = []
    missing_files = []
    html_issues: list[str] = []
    css_issues: list[str] = []
    html_image_links = []
    viewport_tags: list[str] = []

    for path in paths:
        exists = path.exists()
        size = path.stat().st_size if exists else None
        sha = _sha256(path) if exists else None
        metadata = registry_by_path.get(str(path)) or {}
        expected_sha = metadata.get('file_sha256') or metadata.get('content_sha256')
        if exists and expected_sha and sha != expected_sha:
            sha_mismatches.append(str(path))
        if not exists:
            missing_files.append(str(path))

        suffix = path.suffix.lower().lstrip('.')
        if exists and suffix in {'html', 'css'} and syntax_owner is not None:
            content = path.read_text(errors='replace')
            issues = syntax_owner.text_artifact_syntax_sanity_issues_for_extension(suffix, content)
            if suffix == 'html':
                html_issues.extend(issues)
            elif suffix == 'css':
                css_issues.extend(issues)
        if exists and suffix == 'html':
            content = path.read_text(errors='replace')
            image_sources = re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', content, flags=re.I)
            for src in image_sources:
                linked_path = Path(src) if src.startswith('/') else (path.parent / src).resolve()
                html_image_links.append({'src': src, 'exists': linked_path.exists()})
            viewport_tags.extend(re.findall(r'<meta\b[^>]*name=["\']viewport["\'][^>]*>', content, flags=re.I))

        file_checks.append(
            {
                'path': str(path),
                'exists': exists,
                'size_bytes': size,
                'sha256': sha,
                'registry_sha256': expected_sha,
                'suffix': suffix,
            }
        )

    weak_viewport = [
        tag for tag in viewport_tags
        if 'width=device-width' not in tag or 'initial-scale=1.0' not in tag
    ]
    duplicate_refs = sorted({ref for ref in artifact_refs if artifact_refs.count(ref) > 1})
    suffix_counts = _count_values([item.get('suffix') for item in file_checks])
    final_kinds = [
        entry.get('kind')
        for entry in final_artifact_entries
        if entry.get('kind') not in (None, '', 'unknown')
    ]
    artifact_kind_counts = _count_values(
        final_kinds
        or [
            item.get('suffix')
            for item in file_checks
            if item.get('suffix')
        ]
    )
    output_ref_counts = _count_values([entry.get('kind') for entry in output_artifact_entries])
    existing_paths = {
        str(item.get('path'))
        for item in file_checks
        if item.get('exists') and item.get('path')
    }
    audio_paths = {
        str(item.get('path'))
        for item in file_checks
        if item.get('exists')
        and (
            _is_audio_kind(item.get('suffix'))
            or _is_audio_kind(_artifact_suffix_kind(str(item.get('path') or '')))
        )
    }
    for entry in final_artifact_entries:
        path = str(entry.get('path') or '')
        if path in existing_paths and _is_audio_kind(entry.get('kind')):
            audio_paths.add(path)
    audio_artifact_count = len(audio_paths)
    return {
        'files': file_checks,
        'registry_records': registry_parse_records,
        'missing_files': missing_files,
        'sha_mismatches': sha_mismatches,
        'html_issues': html_issues,
        'css_issues': css_issues,
        'html_image_links': html_image_links,
        'viewport_tags': viewport_tags,
        'weak_viewport_tags': weak_viewport,
        'duplicate_artifact_refs': duplicate_refs,
        'duplicate_ref_summary': duplicate_summary,
        'artifact_kind_counts': artifact_kind_counts,
        'artifact_file_count_by_suffix': suffix_counts,
        'output_ref_counts': output_ref_counts,
        'audio_artifact_count': audio_artifact_count,
        'artifact_count': len(paths),
        'artifact_bytes': sum(item.get('size_bytes') or 0 for item in file_checks),
    }


def _duration(a: datetime | None, b: datetime | None) -> float | None:
    if not a or not b:
        return None
    return max(0.0, (b - a).total_seconds())


def _branch_wait_reason(timing: dict[str, Any]) -> str:
    queued = _seconds_from_ms(timing.get('queued_elapsed_ms'))
    lock = _seconds_from_ms(timing.get('lock_wait_ms'))
    execution = _seconds_from_ms(timing.get('execution_ms'))
    elapsed = _seconds_from_ms(timing.get('elapsed_ms'))
    if lock >= 1.0:
        return 'waited for the selected instance lock before backend execution'
    if queued >= 1.0:
        return 'queued behind earlier work on the selected instance or scheduler'
    if elapsed and execution / elapsed >= 0.8:
        return 'backend execution dominated'
    return 'no meaningful queue or lock wait'


def _branch_role(branch_id: str) -> str:
    if 'image_generation' in branch_id:
        return 'image'
    if 'text_to_speech' in branch_id:
        return 'text_to_speech'
    if 'speech_to_text' in branch_id:
        return 'speech_to_text'
    if branch_id.startswith('coalesced-text-artifacts'):
        return 'coalesced_text'
    if branch_id == 'branch-chat-1':
        return 'post_image_chat'
    if 'chat' in branch_id:
        return 'chat'
    if 'text_artifact' in branch_id:
        return 'text_artifact'
    return 'branch'


def _branch_record_role(branch: dict[str, Any]) -> str:
    return (
        str(
            branch.get('output_type')
            or branch.get('capability')
            or _branch_role(str(branch.get('branch_id') or branch.get('phase_id') or ''))
        )
        .strip()
        or 'branch'
    )


def _collect_output_obligations(payload: dict[str, Any], frame: dict[str, Any]) -> dict[str, int]:
    graphs: list[dict[str, Any]] = []

    def add_graph(value: Any) -> None:
        if isinstance(value, dict) and value:
            graphs.append(value)

    for source in (payload, frame):
        runtime = source.get('runtime') if isinstance(source.get('runtime'), dict) else {}
        add_graph(runtime.get('request_phase_graph'))
        add_graph(runtime.get('graph'))
        add_graph(source.get('request_phase_graph'))
        planning = source.get('planning') if isinstance(source.get('planning'), dict) else {}
        add_graph(planning.get('request_phase_graph'))
        add_graph(planning.get('graph'))
        artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), dict) else {}
        add_graph(artifact_flow.get('request_phase_graph'))
        add_graph(artifact_flow.get('graph'))

    kinds: list[str] = []
    for graph in graphs:
        obligations = graph.get('output_obligations') or []
        if not isinstance(obligations, list):
            continue
        for obligation in obligations:
            if not isinstance(obligation, dict):
                continue
            kinds.append(
                str(
                    obligation.get('output_type')
                    or obligation.get('kind')
                    or obligation.get('capability')
                    or 'unknown'
                )
            )
    return _count_values(kinds)


def _worker_limit_reasons(policy: dict[str, Any]) -> list[str]:
    worker = _as_int(policy.get('worker_count'))
    max_workers = _as_int(policy.get('max_parallel_workers'))
    default_workers = _as_int(policy.get('default_worker_count'))
    prepared = _as_int(policy.get('prepared_branch_count'))
    capacity = _as_int(policy.get('scheduling_capacity_units'))
    distinct_instances = _as_int(policy.get('distinct_instance_count'))
    source = str(policy.get('worker_count_source') or '').strip()
    guard = str(policy.get('gpu_heavy_guard') or '').strip()
    same_instance_groups = policy.get('same_instance_lock_groups')
    if not isinstance(same_instance_groups, dict):
        same_instance_groups = {}

    reasons: list[str] = []
    if source:
        reasons.append(f'worker count source {source}')
    if worker is not None and max_workers is not None:
        if worker < max_workers:
            if prepared is not None and prepared <= worker:
                reasons.append(
                    f'worker_count {worker} below max_parallel_workers {max_workers} because {_prepared_branch_clause(prepared)}'
                )
            elif capacity is not None and capacity < max_workers:
                reasons.append(f'worker_count {worker} below max_parallel_workers {max_workers} due to scheduling capacity {capacity}')
            elif distinct_instances is not None and distinct_instances < max_workers:
                reasons.append(f"worker_count {worker} below max_parallel_workers {max_workers} with {_count_phrase(distinct_instances, 'selected instance')}")
            else:
                reasons.append(f'worker_count {worker} below max_parallel_workers {max_workers}')
        elif worker == max_workers:
            reasons.append(f'worker_count reached max_parallel_workers {max_workers}')
    if (
        source != 'explicit_override'
        and capacity is not None
        and max_workers is not None
        and capacity < max_workers
    ):
        reasons.append(f'scheduling capacity {capacity} limited default workers below max')
    if (
        source != 'explicit_override'
        and default_workers is not None
        and prepared is not None
        and default_workers < prepared
    ):
        reasons.append(f'default worker count {default_workers} below {prepared} prepared branches')
    if prepared is not None and worker is not None and prepared > worker:
        reasons.append(f"{_count_phrase(prepared - worker, 'prepared branch', 'prepared branches')} waited for worker availability")
    if distinct_instances is not None and prepared is not None and distinct_instances < prepared:
        reasons.append(f"{_count_phrase(distinct_instances, 'selected instance')} for {prepared} prepared branches")
    if same_instance_groups:
        reasons.append(_count_phrase(len(same_instance_groups), 'same-instance lock group'))
    if guard:
        if guard == 'not_serialized':
            reasons.append('gpu-heavy guard did not serialize this wave')
        else:
            reasons.append(f'gpu-heavy guard {guard} shaped scheduling')
    if not reasons:
        reasons.append('no worker cap signal recorded')
    return reasons


def _wave_diagnostic_from_history(history: dict[str, Any]) -> dict[str, Any]:
    distinct_instance_ids = history.get('distinct_instance_ids') or []
    if not isinstance(distinct_instance_ids, list):
        distinct_instance_ids = []
    same_instance_groups = history.get('same_instance_lock_groups') or {}
    if not isinstance(same_instance_groups, dict):
        same_instance_groups = {}
    return {
        'scheduler': history.get('scheduler'),
        'elapsed_seconds': _seconds_from_ms(history.get('elapsed_ms')),
        'planning_seconds': _seconds_from_ms(history.get('planning_elapsed_ms')),
        'max_parallel_workers': history.get('max_parallel_workers'),
        'default_worker_count': history.get('default_worker_count'),
        'worker_count': history.get('worker_count'),
        'worker_count_source': history.get('worker_count_source'),
        'prepared_branch_count': history.get('prepared_branch_count'),
        'scheduling_capacity_units': history.get('scheduling_capacity_units'),
        'local_text_io_fast_lane_count': history.get('local_text_io_fast_lane_count'),
        'distinct_instance_count': history.get('distinct_instance_count') or len(distinct_instance_ids),
        'distinct_instance_ids': distinct_instance_ids,
        'same_instance_lock_groups': same_instance_groups,
        'instance_branch_groups': history.get('instance_branch_groups') or {},
        'gpu_heavy_guard': history.get('gpu_heavy_guard'),
        'branch_progress_dispatch': history.get('branch_progress_dispatch'),
        'branch_progress_callback_count': history.get('branch_progress_callback_count'),
        'worker_limit_reasons': _worker_limit_reasons(history),
    }


def _summarize_response_frame_finalize_timing(timing: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(timing, dict) or not timing:
        return {}
    steps = []
    for step in timing.get('steps') or []:
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                'name': step.get('name'),
                'elapsed_seconds': _seconds_from_ms(step.get('elapsed_ms')),
            }
        )
    return {
        'phase': timing.get('phase'),
        'persist_requested': timing.get('persist_requested'),
        'persist_effective': timing.get('persist_effective'),
        'late_fill_status': timing.get('late_fill_status'),
        'total_seconds': _seconds_from_ms(timing.get('total_elapsed_ms')),
        'pending_branch_count': timing.get('pending_branch_count'),
        'active_branch_count': timing.get('active_branch_count'),
        'completed_branch_count': timing.get('completed_branch_count'),
        'failed_branch_count': timing.get('failed_branch_count'),
        'steps': steps,
    }


def _summarize_post_wave_backend_timing(timing: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(timing, dict) or not timing:
        return {}
    response_frame_finalize = _summarize_response_frame_finalize_timing(
        timing.get('response_frame_finalize_timing')
        if isinstance(timing.get('response_frame_finalize_timing'), dict)
        else {}
    )
    summary = {
        'phase': timing.get('phase'),
        'status': timing.get('status') or timing.get('late_fill_status'),
        'finalize_seconds': _seconds_from_ms(timing.get('finalize_elapsed_ms')),
        'touch_response_lookup_seconds': _seconds_from_ms(timing.get('touch_response_lookup_elapsed_ms')),
        'pending_branch_count': timing.get('pending_branch_count'),
        'active_branch_count': timing.get('active_branch_count'),
        'completed_branch_count': timing.get('completed_branch_count'),
        'failed_branch_count': timing.get('failed_branch_count'),
    }
    if response_frame_finalize:
        summary['response_frame_finalize'] = response_frame_finalize
    return summary


def _format_scheduling_policy(policy: dict[str, Any]) -> str:
    guard = policy.get('gpu_heavy_guard') or 'none recorded'
    reason = policy.get('reason') or 'no reason recorded'
    original = policy.get('original_branch_count')
    scheduled = policy.get('scheduled_branch_count')
    deferred = policy.get('deferred_branch_count')
    parts = [f'gpu-heavy guard {guard}', f'reason {reason}']
    if original is not None and scheduled is not None:
        parts.append(f'scheduled {scheduled}/{original} branches')
    if deferred is not None:
        parts.append(f'deferred {deferred}')
    return '; '.join(parts)


def _format_instance_groups(groups: Any) -> str | None:
    if not isinstance(groups, dict) or not groups:
        return None
    formatted = []
    for instance_id, branch_ids in sorted(groups.items()):
        if not isinstance(branch_ids, list):
            continue
        clipped = [str(branch_id) for branch_id in branch_ids[:3]]
        suffix = '' if len(branch_ids) <= 3 else f', +{len(branch_ids) - 3} more'
        formatted.append(f"{instance_id}: {', '.join(clipped)}{suffix}")
    return '; '.join(formatted) if formatted else None


def _append_mapping(target: list[dict[str, Any]], value: Any) -> None:
    if isinstance(value, dict) and value:
        target.append(value)


def _append_list_items(target: list[dict[str, Any]], value: Any) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        _append_mapping(target, item)


def _unique_mappings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        try:
            key = json.dumps(item, sort_keys=True, default=str)
        except TypeError:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _clip_text(value: Any, limit: int = 180) -> str | None:
    if value in (None, '', [], {}):
        return None
    text = re.sub(r'\s+', ' ', str(value)).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + '...'


def _brief_value(value: Any, limit: int = 180) -> str | None:
    if value in (None, '', [], {}):
        return None
    if isinstance(value, (str, int, float, bool)):
        return _clip_text(value, limit)
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return _clip_text(text, limit)


def _brief_counts(value: Any, limit: int = 6) -> str | None:
    if isinstance(value, dict) and value:
        parts = [
            f'{key}={value[key]}'
            for key in sorted(value.keys(), key=str)[:limit]
            if value.get(key) not in (None, '', [], {})
        ]
        if len(value) > limit:
            parts.append(f'+{len(value) - limit} more')
        return ', '.join(parts) if parts else None
    return _brief_value(value)


def _status_token(value: Any) -> str:
    return re.sub(r'[\s-]+', '_', str(value or '').strip().lower())


def _bool_token(value: Any) -> str:
    return 'true' if bool(value) else 'false'


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, sort_keys=True, default=str))
    except TypeError:
        return len(str(value))


def _dedupe_records(items: list[dict[str, Any]], *identity_keys: str) -> list[dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in items:
        identity = ''
        for key in identity_keys:
            candidate = str(item.get(key) or '').strip()
            if candidate:
                identity = candidate
                break
        if not identity:
            identity = hashlib.sha256(
                json.dumps(item, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest()[:16]
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        if _json_size(item) >= _json_size(by_identity[identity]):
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _join_compact(values: list[str], *, limit: int = 6) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return 'none'
    unique: list[str] = []
    seen: set[str] = set()
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    clipped = unique[:limit]
    if len(unique) > limit:
        clipped.append(f'+{len(unique) - limit} more')
    return ', '.join(clipped)


def _collect_forbidden_evidence_labels(*sources: Any) -> list[str]:
    labels: list[str] = []

    def add_label(label: str) -> None:
        if label and label not in labels:
            labels.append(label)

    def classify(text: Any) -> None:
        token = _status_token(text)
        if not token:
            return
        if any(marker in token for marker in ('accepted_learning', 'learning_only', 'advisory')):
            add_label('learning_only')
        if any(
            marker in token
            for marker in (
                'degraded',
                'degraded_liveness_only',
                'cache_liveness',
                'liveness_only',
                'route_health',
                'provider_ban',
                'provider_family_ban',
            )
        ):
            add_label('degraded/provider')
        if any(marker in token for marker in ('frontend', 'ui_label')):
            add_label('frontend')
        if 'monitor_only' in token:
            add_label('monitor_only')

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                classify(key)
                visit(item)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        classify(value)

    for source in sources:
        visit(source)
    return labels


def _read_snapshot_payload(root: Path, snapshot_reader: Any, ref: Any) -> Any:
    if snapshot_reader is None:
        return None
    try:
        return snapshot_reader(ref, frames_dir=root / 'state/response_frames')
    except Exception:
        return None


def _mapping_value_or_snapshot(root: Path, snapshot_reader: Any, source: Any, key: str) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if isinstance(value, dict) and value:
        return value
    snapshot = _read_snapshot_payload(root, snapshot_reader, source.get(f'{key}_snapshot_ref'))
    if isinstance(snapshot, dict) and snapshot:
        return snapshot
    return None


def _first_mapping(values: list[Any]) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _summarize_learning_hint(hint: dict[str, Any]) -> dict[str, Any]:
    return {
        'target_area': hint.get('target_area') or hint.get('area') or hint.get('policy_area'),
        'case_kinds': _brief_counts(hint.get('case_kinds') or hint.get('case_kind_counts')),
        'severity_counts': _brief_counts(hint.get('severity_counts')),
        'learning_id': hint.get('learning_id'),
        'hint': _clip_text(hint.get('hint') or hint.get('summary') or hint.get('message'), 240),
    }


def _summarize_graph_repair_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    repair_gap = proposal.get('repair_gap') if isinstance(proposal.get('repair_gap'), dict) else {}
    patch = (
        proposal.get('graph_patch')
        if isinstance(proposal.get('graph_patch'), dict)
        else proposal.get('patch') if isinstance(proposal.get('patch'), dict) else {}
    )
    branch_additions = (
        patch.get('branches')
        or patch.get('add_branches')
        or patch.get('branch_additions')
        or patch.get('add_phases')
        or proposal.get('branches')
        or []
    )
    return {
        'proposal_id': proposal.get('proposal_id') or proposal.get('id'),
        'status': proposal.get('status'),
        'source': proposal.get('source') or repair_gap.get('source'),
        'repair_type': proposal.get('repair_type'),
        'repair_gap_code': proposal.get('repair_gap_code') or repair_gap.get('code'),
        'repair_actions': _brief_value(proposal.get('repair_actions') or repair_gap.get('repair_actions')),
        'branch_addition_count': len(branch_additions) if isinstance(branch_additions, list) else None,
    }


def _summarize_graph_repair_review(review: dict[str, Any]) -> dict[str, Any]:
    return {
        'review_id': review.get('review_id') or review.get('id'),
        'proposal_id': review.get('proposal_id'),
        'status': review.get('status') or review.get('review_status'),
        'source': review.get('source'),
        'reasons': _brief_value(review.get('reasons') or review.get('reject_reasons') or review.get('issues'), 260),
    }


def _unique_graph_repair_reviews(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in items:
        summary = _summarize_graph_repair_review(item)
        key = (
            str(summary.get('review_id') or ''),
            str(summary.get('proposal_id') or ''),
            str(summary.get('status') or ''),
            str(summary.get('reasons') or ''),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _pair_graph_repair_proposals_and_reviews(
    proposals: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_summaries = [_summarize_graph_repair_review(item) for item in reviews]
    reviews_by_proposal: dict[str, list[dict[str, Any]]] = {}
    for review in review_summaries:
        proposal_id = str(review.get('proposal_id') or '').strip()
        if not proposal_id:
            continue
        reviews_by_proposal.setdefault(proposal_id, []).append(review)

    pairs: list[dict[str, Any]] = []
    matched_review_ids: set[str] = set()
    for proposal in proposals:
        summary = _summarize_graph_repair_proposal(proposal)
        proposal_id = str(summary.get('proposal_id') or '').strip()
        matched_reviews = reviews_by_proposal.get(proposal_id, [])
        for review in matched_reviews:
            review_identity = str(review.get('review_id') or '') or json.dumps(review, sort_keys=True, default=str)
            matched_review_ids.add(review_identity)
        pairs.append({'proposal': summary, 'reviews': matched_reviews})

    unmatched: list[dict[str, Any]] = []
    for review in review_summaries:
        review_identity = str(review.get('review_id') or '') or json.dumps(review, sort_keys=True, default=str)
        if review_identity not in matched_review_ids:
            unmatched.append(review)
    return pairs, unmatched


def _collect_decision_contracts(graph_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for graph in graph_sources:
        _append_mapping(contracts, graph.get('decision_contract'))
        request_ir = graph.get('request_ir')
        if isinstance(request_ir, dict):
            _append_mapping(contracts, request_ir.get('decision_contract'))
    return _unique_mappings(contracts)


def _summarize_enforcement_group(
    reviews: list[dict[str, Any]],
    lifecycles: list[dict[str, Any]],
) -> dict[str, Any]:
    classes: list[str] = []
    blocked_reasons: list[str] = []
    authorities: list[str] = []
    allowed_count = 0
    blocked_count = 0

    for review in reviews:
        enforced_class = str(review.get('enforced_class') or '').strip()
        authority = str(review.get('authority') or '').strip()
        if enforced_class and enforced_class not in classes:
            classes.append(enforced_class)
        if authority and authority not in authorities:
            authorities.append(authority)
        for reason in review.get('blocked_reasons') or []:
            reason_text = str(reason or '').strip()
            if reason_text and reason_text not in blocked_reasons:
                blocked_reasons.append(reason_text)
        if bool(review.get('allowed')) and _status_token(review.get('status')) == 'allowed':
            allowed_count += 1
        else:
            blocked_count += 1

    for lifecycle in lifecycles:
        enforced_class = str(lifecycle.get('enforced_class') or '').strip()
        authority = str(lifecycle.get('authority') or '').strip()
        if enforced_class and enforced_class not in classes:
            classes.append(enforced_class)
        if authority and authority not in authorities:
            authorities.append(authority)
        for reason in lifecycle.get('blocked_reasons') or []:
            reason_text = str(reason or '').strip()
            if reason_text and reason_text not in blocked_reasons:
                blocked_reasons.append(reason_text)

    return {
        'allowed_count': allowed_count,
        'blocked_count': blocked_count,
        'classes': classes,
        'blocked_reasons': blocked_reasons,
        'authorities': authorities,
    }


def _summarize_partial_rebase_execution_observer(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project concrete partial-rebase execution truth without gaining authority."""

    if not records:
        return {}
    status_counts = _count_values(
        [_status_token(item.get('status')) or 'unknown' for item in records]
    )
    scheduled_branch_count = sum(
        len(item.get('scheduled_branch_ids') or [])
        for item in records
        if isinstance(item.get('scheduled_branch_ids'), list)
    )
    execution_keys = [
        str(item.get('execution_key') or '').strip()
        for item in records
        if str(item.get('execution_key') or '').strip()
    ]
    authorization_record_ids = {
        str(item.get('authorization_record_id') or '').strip()
        for item in records
        if str(item.get('authorization_record_id') or '').strip()
    }
    return {
        'kind': 'ollmo.run_monitor_partial_rebase_execution_observer',
        'authority': 'observer_only',
        'runtime_effect': 'none',
        'execution_count': len(records),
        'status_counts': status_counts,
        'scheduled_branch_count': scheduled_branch_count,
        'root_prompt_replay_count': sum(
            1 for item in records if item.get('root_prompt_replay') is True
        ),
        'authorization_record_count': len(authorization_record_ids),
        'execution_keys': execution_keys[:8],
    }


def _dedupe_partial_rebase_execution_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    status_rank = {
        'failed': 90,
        'cancelled': 90,
        'blocked': 90,
        'repair_needed': 90,
        'completed': 80,
        'succeeded': 80,
        'solved': 80,
        'fulfilled': 80,
        'running': 40,
        'queued': 30,
        'pending': 20,
    }
    by_identity: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in records:
        identity = next(
            (
                str(item.get(key) or '').strip()
                for key in (
                    'execution_key',
                    'successor_key',
                    'rebase_id',
                    'proposal_id',
                )
                if str(item.get(key) or '').strip()
            ),
            hashlib.sha256(
                json.dumps(item, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest()[:16],
        )
        if identity not in by_identity:
            by_identity[identity] = item
            order.append(identity)
            continue
        incoming_preference = (
            status_rank.get(_status_token(item.get('status')), 0),
            _json_size(item),
        )
        current = by_identity[identity]
        current_preference = (
            status_rank.get(_status_token(current.get('status')), 0),
            _json_size(current),
        )
        if incoming_preference >= current_preference:
            by_identity[identity] = item
    return [by_identity[identity] for identity in order]


def _collect_runtime_repair_authority(
    runtime_sources: list[dict[str, Any]],
    graph_sources: list[dict[str, Any]],
    developer_diagnostics: list[dict[str, Any]],
    accepted_learning: dict[str, Any],
) -> dict[str, Any]:
    enforced_policy = _first_mapping([item.get('enforced_policy') for item in developer_diagnostics])
    graph_patch_autonomy = _first_mapping([item.get('graph_patch_autonomy') for item in developer_diagnostics])
    graph_rebase_autonomy = _first_mapping([item.get('graph_rebase_autonomy') for item in developer_diagnostics])

    graph_patch_lifecycles: list[dict[str, Any]] = []
    graph_rebase_lifecycles: list[dict[str, Any]] = []
    redraw_scope_reviews: list[dict[str, Any]] = []
    successor_reopen_requests: list[dict[str, Any]] = []
    successor_rebase_requests: list[dict[str, Any]] = []
    successor_rebase_executions: list[dict[str, Any]] = []
    graph_patch_enforced_policy_reviews: list[dict[str, Any]] = []
    graph_rebase_enforced_policy_reviews: list[dict[str, Any]] = []

    for graph in graph_sources:
        _append_list_items(graph_patch_lifecycles, graph.get('graph_patch_lifecycle'))
        _append_list_items(graph_rebase_lifecycles, graph.get('graph_rebase_lifecycle'))
        _append_mapping(redraw_scope_reviews, graph.get('redraw_scope_ladder_review'))
        _append_list_items(redraw_scope_reviews, graph.get('redraw_scope_ladder_reviews'))
        _append_list_items(successor_reopen_requests, graph.get('successor_reopen_requests'))
        _append_list_items(successor_rebase_requests, graph.get('successor_rebase_requests'))
        _append_list_items(successor_rebase_executions, graph.get('successor_rebase_executions'))

    for diagnostics in developer_diagnostics:
        _append_list_items(graph_patch_lifecycles, diagnostics.get('graph_patch_lifecycle'))
        _append_list_items(graph_rebase_lifecycles, diagnostics.get('graph_rebase_lifecycle'))
        _append_mapping(redraw_scope_reviews, diagnostics.get('redraw_scope_ladder_review'))
        _append_list_items(redraw_scope_reviews, diagnostics.get('redraw_scope_ladder_reviews'))
        _append_list_items(successor_reopen_requests, diagnostics.get('graph_patch_successor_reopen_requests'))
        _append_list_items(successor_rebase_requests, diagnostics.get('successor_rebase_requests'))
        _append_mapping(
            successor_rebase_executions,
            diagnostics.get('graph_rebase_partial_successor_execution'),
        )
        _append_list_items(
            successor_rebase_executions,
            diagnostics.get('successor_rebase_executions'),
        )
        _append_list_items(graph_patch_enforced_policy_reviews, diagnostics.get('graph_patch_enforced_policy_reviews'))
        _append_list_items(graph_rebase_enforced_policy_reviews, diagnostics.get('graph_rebase_enforced_policy_reviews'))

    graph_patch_lifecycles = _dedupe_records(
        graph_patch_lifecycles,
        'patch_id',
        'idempotency_key',
        'proposal_id',
        'review_id',
    )
    graph_rebase_lifecycles = _dedupe_records(
        graph_rebase_lifecycles,
        'rebase_id',
        'idempotency_key',
        'proposal_id',
        'review_id',
    )
    redraw_scope_reviews = _dedupe_records(redraw_scope_reviews, 'review_id')
    successor_reopen_requests = _dedupe_records(
        successor_reopen_requests,
        'successor_reopen_key',
        'patch_id',
        'proposal_id',
        'idempotency_key',
    )
    successor_rebase_requests = _dedupe_records(
        successor_rebase_requests,
        'successor_rebase_key',
        'rebase_id',
        'proposal_id',
        'idempotency_key',
    )
    successor_rebase_executions = [
        item
        for item in _dedupe_partial_rebase_execution_records(
            successor_rebase_executions
        )
        if str(item.get('kind') or '').strip()
        == 'ollmo.graph_rebase_partial_successor_execution'
    ]
    partial_rebase_execution_observer = (
        _summarize_partial_rebase_execution_observer(
            successor_rebase_executions
        )
    )

    apply_enforced_patch_lifecycles = [
        item for item in graph_patch_lifecycles
        if _status_token(item.get('autonomy_level')) == 'apply_enforced'
    ]
    apply_enforced_rebase_lifecycles = [
        item for item in graph_rebase_lifecycles
        if _status_token(item.get('autonomy_level')) == 'apply_enforced'
    ]

    for lifecycle in apply_enforced_patch_lifecycles:
        _append_mapping(graph_patch_enforced_policy_reviews, lifecycle.get('enforced_policy_review'))
    for lifecycle in apply_enforced_rebase_lifecycles:
        _append_mapping(graph_rebase_enforced_policy_reviews, lifecycle.get('enforced_policy_review'))

    graph_patch_enforced_policy_reviews = _dedupe_records(
        graph_patch_enforced_policy_reviews,
        'review_id',
        'patch_id',
        'idempotency_key',
        'proposal_id',
    )
    graph_rebase_enforced_policy_reviews = _dedupe_records(
        graph_rebase_enforced_policy_reviews,
        'review_id',
        'rebase_id',
        'idempotency_key',
        'proposal_id',
    )

    repair_enforcement = _summarize_enforcement_group(
        graph_patch_enforced_policy_reviews,
        apply_enforced_patch_lifecycles,
    )
    rebase_enforcement = _summarize_enforcement_group(
        graph_rebase_enforced_policy_reviews,
        apply_enforced_rebase_lifecycles,
    )

    forbidden_evidence_labels = _collect_forbidden_evidence_labels(
        [item.get('forbidden_evidence_seen') for item in graph_patch_enforced_policy_reviews],
        [item.get('forbidden_evidence_seen') for item in graph_rebase_enforced_policy_reviews],
        [item.get('current_evidence_refs') for item in graph_patch_enforced_policy_reviews],
        [item.get('current_evidence_refs') for item in graph_rebase_enforced_policy_reviews],
        [item.get('source_evidence_refs') for item in apply_enforced_patch_lifecycles],
        [item.get('source_evidence_refs') for item in apply_enforced_rebase_lifecycles],
        redraw_scope_reviews,
    )

    selected_scopes: list[str] = []
    redraw_blocked_reasons: list[str] = []
    for review in redraw_scope_reviews:
        selected_scope = str(
            review.get('selected_scope')
            or (
                review.get('selected_candidate') or {}
            ).get('scope')
            or ''
        ).strip()
        if selected_scope and selected_scope not in selected_scopes:
            selected_scopes.append(selected_scope)
        for reason in review.get('blocked_reasons') or []:
            reason_text = str(reason or '').strip()
            if reason_text and reason_text not in redraw_blocked_reasons:
                redraw_blocked_reasons.append(reason_text)

    parent_frozen_unmutated = bool(
        successor_reopen_requests
        or successor_rebase_requests
        or successor_rebase_executions
    )
    for lifecycle in apply_enforced_patch_lifecycles:
        reasons = {str(item or '').strip() for item in (lifecycle.get('blocked_reasons') or [])}
        runtime_effect = _status_token((lifecycle.get('outcome') or {}).get('runtime_effect'))
        if 'terminal_frame_requires_successor_reopen' in reasons or runtime_effect == 'terminal_frame_not_mutated':
            parent_frozen_unmutated = True
    for request in successor_rebase_requests:
        if str(request.get('runtime_effect') or '').strip():
            parent_frozen_unmutated = True

    return {
        'enforced_policy': {
            'mode': enforced_policy.get('mode') or 'unknown',
            'enabled': bool(enforced_policy.get('enabled')),
            'default_action': enforced_policy.get('default_action') or 'unknown',
        } if enforced_policy else {},
        'graph_patch_autonomy': {
            'autonomy_level': graph_patch_autonomy.get('autonomy_level'),
            'source': graph_patch_autonomy.get('source'),
            'terminal_apply_blocked': bool(graph_patch_autonomy.get('terminal_apply_blocked')),
        } if graph_patch_autonomy else {},
        'graph_rebase_autonomy': {
            'autonomy_level': graph_rebase_autonomy.get('autonomy_level'),
            'source': graph_rebase_autonomy.get('source'),
        } if graph_rebase_autonomy else {},
        'graph_repair_enforcement': repair_enforcement,
        'graph_rebase_enforcement': {
            **rebase_enforcement,
            'full_successor_rebase_blocked': 'full_successor_rebase_not_enforced_v1' in rebase_enforcement.get('blocked_reasons', []),
            'partial_subtree_rebase_audit_only': 'partial_subtree_rebase_enforced_v1_audit_only' in rebase_enforcement.get('blocked_reasons', []),
        },
        'successor_runtime': {
            'reopen_count': len(successor_reopen_requests),
            'rebase_count': len(successor_rebase_requests),
            'rebase_execution_count': len(successor_rebase_executions),
            'parent_frozen_unmutated': parent_frozen_unmutated,
            'partial_rebase_execution_observer': (
                partial_rebase_execution_observer
            ),
        },
        'redraw_scope': {
            'selected_scopes': selected_scopes,
            'blocked_reasons': redraw_blocked_reasons,
        },
        'forbidden_evidence_labels': forbidden_evidence_labels,
        'accepted_learning_authority': accepted_learning.get('authority'),
        'accepted_learning_runtime_effect': accepted_learning.get('runtime_effect'),
    }


def _collect_learning_healing(
    root: Path,
    payload: dict[str, Any],
    frame: dict[str, Any],
    snapshot_reader: Any,
    monitor_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), dict) else {}

    runtime_sources: list[dict[str, Any]] = []
    for source in (payload, frame, current_state):
        if not isinstance(source, dict):
            continue
        _append_mapping(runtime_sources, source.get('runtime'))
        _append_mapping(runtime_sources, _read_snapshot_payload(root, snapshot_reader, source.get('runtime_snapshot_ref')))
    runtime_sources = _unique_mappings(runtime_sources)

    late_fill_sources: list[dict[str, Any]] = []
    for source in (payload, frame, current_state):
        if not isinstance(source, dict):
            continue
        _append_mapping(late_fill_sources, source.get('late_fill'))
        _append_mapping(late_fill_sources, _read_snapshot_payload(root, snapshot_reader, source.get('late_fill_snapshot_ref')))
    late_fill_sources = _unique_mappings(late_fill_sources)

    graph_sources: list[dict[str, Any]] = []
    planning = frame.get('planning') if isinstance(frame.get('planning'), dict) else {}
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), dict) else {}
    for source in (planning, artifact_flow):
        _append_mapping(graph_sources, _mapping_value_or_snapshot(root, snapshot_reader, source, 'request_phase_graph'))
        _append_mapping(graph_sources, _mapping_value_or_snapshot(root, snapshot_reader, source, 'graph'))
    for runtime in runtime_sources:
        _append_mapping(graph_sources, runtime.get('request_phase_graph'))
        _append_mapping(graph_sources, runtime.get('graph'))
    graph_sources = _unique_mappings(graph_sources)

    developer_diagnostics: list[dict[str, Any]] = []
    for runtime in runtime_sources:
        _append_mapping(developer_diagnostics, runtime.get('developer_diagnostics'))
    developer_diagnostics = _unique_mappings(developer_diagnostics)

    accepted = _first_mapping([source.get('accepted_learning_hints') for source in runtime_sources])
    accepted_hints = accepted.get('hints') if isinstance(accepted.get('hints'), list) else []
    accepted_learning = {}
    if accepted:
        accepted_learning = {
            'status': accepted.get('status'),
            'authority': accepted.get('authority'),
            'runtime_effect': accepted.get('runtime_effect'),
            'hint_count': accepted.get('hint_count') if accepted.get('hint_count') is not None else len(accepted_hints),
            'top_hints': [_summarize_learning_hint(item) for item in accepted_hints[:4] if isinstance(item, dict)],
        }

    closure_reviews: list[dict[str, Any]] = []
    for runtime in runtime_sources:
        _append_mapping(closure_reviews, _mapping_value_or_snapshot(root, snapshot_reader, runtime, 'graph_closure_review'))
    for diagnostics in developer_diagnostics:
        _append_mapping(closure_reviews, _mapping_value_or_snapshot(root, snapshot_reader, diagnostics, 'graph_closure_review'))
    closure_reviews = _unique_mappings(closure_reviews)
    closure_review = closure_reviews[0] if closure_reviews else {}

    decision_contracts = _collect_decision_contracts(graph_sources)
    graph_repair_proposals: list[dict[str, Any]] = []
    for contract in decision_contracts:
        _append_list_items(graph_repair_proposals, contract.get('graph_repair_proposals'))
    for diagnostics in developer_diagnostics:
        _append_mapping(graph_repair_proposals, diagnostics.get('graph_repair_proposal'))
    graph_repair_proposals = _unique_mappings(graph_repair_proposals)

    graph_repair_reviews: list[dict[str, Any]] = []
    for graph in graph_sources:
        _append_list_items(graph_repair_reviews, graph.get('graph_repair_reviews'))
        for refinement in graph.get('graph_refinements') or []:
            if isinstance(refinement, dict) and 'graph_repair_proposal_review' in str(refinement.get('kind') or ''):
                _append_mapping(graph_repair_reviews, refinement)
    for diagnostics in developer_diagnostics:
        _append_mapping(graph_repair_reviews, diagnostics.get('graph_repair_proposal_review'))
    graph_repair_reviews = _unique_graph_repair_reviews(graph_repair_reviews)

    surface_state = _first_mapping(
        [
            closure_review.get('surface_state') if isinstance(closure_review, dict) else None,
            *[
                _mapping_value_or_snapshot(root, snapshot_reader, late_fill, 'surface_state')
                for late_fill in late_fill_sources
            ],
        ]
    )
    global_semantic_closure = _first_mapping(
        [
            closure_review.get('global_semantic_closure_review') if isinstance(closure_review, dict) else None,
            *[
                _mapping_value_or_snapshot(root, snapshot_reader, late_fill, 'global_semantic_closure_review')
                for late_fill in late_fill_sources
            ],
        ]
    )
    active_reconsideration = _first_mapping(
        [
            *[
                _mapping_value_or_snapshot(root, snapshot_reader, late_fill, 'active_reconsideration_review')
                for late_fill in late_fill_sources
            ],
            *[contract.get('active_reconsideration_review') for contract in decision_contracts],
        ]
    )

    runtime_repair_authority = _collect_runtime_repair_authority(
        runtime_sources,
        graph_sources,
        developer_diagnostics,
        accepted_learning,
    )

    latest_late_fill = late_fill_sources[0] if late_fill_sources else {}
    recovery_candidates = latest_late_fill.get('recovery_candidates') if isinstance(latest_late_fill.get('recovery_candidates'), list) else []
    repair_actions = latest_late_fill.get('repair_actions') if isinstance(latest_late_fill.get('repair_actions'), list) else []
    if not repair_actions and latest_late_fill.get('repair_action'):
        repair_actions = [latest_late_fill.get('repair_action')]
    repair_loop = latest_late_fill.get('repair_loop') if isinstance(latest_late_fill.get('repair_loop'), dict) else {}
    reconsideration_rebuild = (
        latest_late_fill.get('reconsideration_rebuild')
        if isinstance(latest_late_fill.get('reconsideration_rebuild'), dict)
        else {}
    )

    decisions = active_reconsideration.get('decisions') or active_reconsideration.get('active_reconsideration_decisions') or []
    action_counts: dict[str, int] = {}
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            action = str(decision.get('recommended_action') or decision.get('action') or '').strip()
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1

    closure_summary = {}
    if closure_review or global_semantic_closure:
        closure_summary = {
            'status': closure_review.get('status'),
            'repair_gap_code': closure_review.get('repair_gap_code'),
            'repair_gap_trigger': closure_review.get('repair_gap_trigger'),
            'repair_pending_branch_count': closure_review.get('repair_pending_branch_count'),
            'semantic_review_status': global_semantic_closure.get('status') or global_semantic_closure.get('semantic_review_verdict'),
            'semantic_transition': global_semantic_closure.get('recommended_transition') or global_semantic_closure.get('transition'),
        }

    surface_summary = {}
    if surface_state:
        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from ollmo_services.graph_repair import classify_surface_repair_actionability

            surface_actionability = classify_surface_repair_actionability(
                surface_state,
                closure_review=closure_review,
                late_fill=latest_late_fill,
                monitor_report=monitor_report or {},
            )
        except Exception:
            surface_actionability = {}
        surface_summary = {
            'state': surface_state.get('state') or surface_state.get('status'),
            'reason': _clip_text(surface_state.get('reason') or surface_state.get('summary'), 220),
            'category_counts': _brief_counts(surface_state.get('category_counts')),
            'item_count': len(surface_state.get('items')) if isinstance(surface_state.get('items'), list) else surface_state.get('item_count'),
            'actionability': surface_actionability.get('status'),
            'actionable_categories': _brief_counts(surface_actionability.get('actionable_categories')),
            'advisory_categories': _brief_counts(surface_actionability.get('advisory_categories')),
        }

    reconsideration_summary = {}
    if active_reconsideration:
        reconsideration_summary = {
            'status': active_reconsideration.get('status'),
            'decision_count': len(decisions) if isinstance(decisions, list) else active_reconsideration.get('decision_count'),
            'category_counts': _brief_counts(active_reconsideration.get('category_counts')),
            'recommended_actions': _brief_counts(action_counts),
        }

    repair_runtime_summary = {}
    if recovery_candidates or repair_actions or repair_loop or reconsideration_rebuild:
        repair_runtime_summary = {
            'repair_loop_status': repair_loop.get('status') or repair_loop.get('state'),
            'repair_loop_attempts': repair_loop.get('attempt_count') or repair_loop.get('attempts'),
            'reconsideration_rebuild_status': reconsideration_rebuild.get('status'),
            'recovery_candidate_count': len(recovery_candidates),
            'repair_action_count': len(repair_actions),
            'repair_actions': _brief_value(repair_actions, 220),
        }

    latest_lifecycle = payload.get('lifecycle_state')
    latest_late_fill_status = latest_late_fill.get('status')
    final_contract = latest_late_fill.get('final_materialization_contract_status')
    synthetic_monitor_report = {
        **(monitor_report or {}),
        'response_id': payload.get('response_id') or payload.get('id') or frame.get('response_id') or frame.get('id'),
        'status': payload.get('status') or frame.get('status'),
        'lifecycle_state': latest_lifecycle,
        'late_fill_status': latest_late_fill_status,
        'final_materialization_contract_status': final_contract,
        'materialization_contract_unmet': latest_late_fill.get('materialization_contract_unmet'),
        'branch_counts': {
            'pending': len(latest_late_fill.get('pending_branches') or []),
            'active': len(latest_late_fill.get('active_branches') or []),
            'failed': len(latest_late_fill.get('failed_branches') or []),
            'completed': len(latest_late_fill.get('completed_branches') or []),
        },
        'surface_state': surface_summary,
    }
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ollmo_services.graph_repair import build_graph_repair_proposals_from_runtime_evidence

        synthesized_proposals = build_graph_repair_proposals_from_runtime_evidence(
            response_frame=frame,
            monitor_report=synthetic_monitor_report,
            request_phase_graph=graph_sources[0] if graph_sources else {},
            closure_review=closure_review,
            late_fill=latest_late_fill,
            accepted_learning_hints=accepted,
        )
    except Exception:
        synthesized_proposals = []
    _append_list_items(graph_repair_proposals, synthesized_proposals)
    graph_repair_proposals = _unique_mappings(graph_repair_proposals)
    graph_repair_pairs, unmatched_graph_repair_reviews = _pair_graph_repair_proposals_and_reviews(
        graph_repair_proposals,
        graph_repair_reviews,
    )

    surface_state_value = str(surface_summary.get('state') or '').lower()
    repair_needed_visible = (
        latest_lifecycle in {'repair_needed', 'failed', 'cancelled'}
        or latest_late_fill_status in {'partial_failed', 'failed', 'cancelled'}
        or final_contract == 'unmet'
        or bool(latest_late_fill.get('materialization_contract_unmet'))
        or surface_state_value in {'repair_needed', 'blocked', 'pending', 'review_pending', 'open'}
    )
    weak_freeze_watch = (
        latest_lifecycle == 'completed'
        and bool(recovery_candidates or repair_actions or graph_repair_reviews or graph_repair_proposals)
        and surface_state_value not in {'completed', 'fulfilled'}
        and final_contract != 'fulfilled'
    )
    freeze_evidence = []
    if latest_lifecycle:
        freeze_evidence.append(f'lifecycle={latest_lifecycle}')
    if latest_late_fill_status:
        freeze_evidence.append(f'late_fill={latest_late_fill_status}')
    if final_contract:
        freeze_evidence.append(f'contract={final_contract}')
    if surface_summary.get('state'):
        freeze_evidence.append(f"surface={surface_summary.get('state')}")

    return {
        'source_counts': {
            'runtime_sources': len(runtime_sources),
            'late_fill_sources': len(late_fill_sources),
            'graph_sources': len(graph_sources),
            'decision_contracts': len(decision_contracts),
        },
        'accepted_learning_hints': accepted_learning,
        'runtime_repair_authority': runtime_repair_authority,
        'graph_repair': {
            'proposal_count': len(graph_repair_proposals),
            'review_count': len(graph_repair_reviews),
            'proposals': [_summarize_graph_repair_proposal(item) for item in graph_repair_proposals[:4]],
            'reviews': [_summarize_graph_repair_review(item) for item in graph_repair_reviews[:4]],
            'proposal_review_pairs': graph_repair_pairs[:4],
            'unmatched_reviews': unmatched_graph_repair_reviews[:4],
        },
        'closure_review': closure_summary,
        'surface_state': surface_summary,
        'active_reconsideration': reconsideration_summary,
        'repair_runtime': repair_runtime_summary,
        'freeze_guard': {
            'repair_needed_visible': repair_needed_visible,
            'weak_freeze_watch': weak_freeze_watch,
            'evidence': '; '.join(freeze_evidence) if freeze_evidence else None,
        },
    }


def _render_learning_healing_lines(info: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    accepted = info.get('accepted_learning_hints') or {}
    if accepted:
        lines.append(
            '- Accepted learning hints: '
            f"{accepted.get('hint_count')} hints, status {accepted.get('status') or 'unknown'}, "
            f"runtime effect {accepted.get('runtime_effect') or 'unknown'}."
        )
        for hint in accepted.get('top_hints') or []:
            target = hint.get('target_area') or 'learning hint'
            details = []
            if hint.get('case_kinds'):
                details.append(f"cases {hint.get('case_kinds')}")
            if hint.get('severity_counts'):
                details.append(f"severity {hint.get('severity_counts')}")
            if hint.get('learning_id'):
                details.append(f"id {hint.get('learning_id')}")
            suffix = f" ({'; '.join(details)})" if details else ''
            hint_text = hint.get('hint') or 'no hint text recorded'
            lines.append(f"  {target}{suffix}: {hint_text}")

    runtime_authority = info.get('runtime_repair_authority') or {}
    if runtime_authority:
        lines.append('- Runtime repair/healing authority:')
        enforced_policy = runtime_authority.get('enforced_policy') or {}
        if enforced_policy:
            lines.append(
                '  Enforced policy: '
                f"mode {enforced_policy.get('mode') or 'unknown'}, "
                f"enabled {_bool_token(enforced_policy.get('enabled'))}, "
                f"default_action {enforced_policy.get('default_action') or 'unknown'}."
            )
        repair = runtime_authority.get('graph_repair_enforcement') or {}
        patch_autonomy = runtime_authority.get('graph_patch_autonomy') or {}
        if repair or patch_autonomy:
            lines.append(
                '  Graph repair enforcement: '
                f"autonomy {patch_autonomy.get('autonomy_level') or 'unknown'}, "
                f"allowed {repair.get('allowed_count', 0)}, "
                f"blocked {repair.get('blocked_count', 0)}, "
                f"classes {_join_compact(repair.get('classes') or [])}, "
                f"blocked reasons {_join_compact(repair.get('blocked_reasons') or [])}, "
                f"authority {_join_compact(repair.get('authorities') or [])}."
            )
        rebase = runtime_authority.get('graph_rebase_enforcement') or {}
        rebase_autonomy = runtime_authority.get('graph_rebase_autonomy') or {}
        if rebase or rebase_autonomy:
            full_blocked = 'yes' if rebase.get('full_successor_rebase_blocked') else 'no'
            partial_blocked = 'yes' if rebase.get('partial_subtree_rebase_audit_only') else 'no'
            lines.append(
                '  Graph rebase enforcement: '
                f"autonomy {rebase_autonomy.get('autonomy_level') or 'unknown'}, "
                f"allowed {rebase.get('allowed_count', 0)}, "
                f"blocked {rebase.get('blocked_count', 0)}, "
                f"full successor rebase blocked {full_blocked}, "
                f"partial subtree rebase audit-only/blocked {partial_blocked}, "
                f"classes {_join_compact(rebase.get('classes') or [])}, "
                f"blocked reasons {_join_compact(rebase.get('blocked_reasons') or [])}, "
                f"authority {_join_compact(rebase.get('authorities') or [])}."
            )
        successor_runtime = runtime_authority.get('successor_runtime') or {}
        if successor_runtime:
            lines.append(
                '  Successor/reopen: '
                f"{successor_runtime.get('reopen_count', 0)} reopen, "
                f"{successor_runtime.get('rebase_count', 0)} rebase, "
                f"parent frozen/unmutated {_bool_token(successor_runtime.get('parent_frozen_unmutated'))}."
            )
            execution_observer = (
                successor_runtime.get('partial_rebase_execution_observer')
                if isinstance(
                    successor_runtime.get('partial_rebase_execution_observer'),
                    dict,
                )
                else {}
            )
            if execution_observer:
                lines.append(
                    '  Partial rebase execution observer: '
                    f"{execution_observer.get('execution_count', 0)} executions, "
                    f"statuses {_brief_counts(execution_observer.get('status_counts')) or 'none'}, "
                    f"scheduled branches {execution_observer.get('scheduled_branch_count', 0)}, "
                    f"root prompt replays {execution_observer.get('root_prompt_replay_count', 0)}, "
                    f"authority {execution_observer.get('authority') or 'observer_only'}, "
                    f"runtime effect {execution_observer.get('runtime_effect') or 'none'}."
                )
        redraw_scope = runtime_authority.get('redraw_scope') or {}
        if redraw_scope.get('selected_scopes') or redraw_scope.get('blocked_reasons'):
            lines.append(
                '  Redraw scope ladder: '
                f"selected {_join_compact(redraw_scope.get('selected_scopes') or [])}, "
                f"blocked reasons {_join_compact(redraw_scope.get('blocked_reasons') or [])}."
            )
        forbidden_labels = runtime_authority.get('forbidden_evidence_labels') or []
        if forbidden_labels:
            lines.append(
                '  Forbidden evidence rejected: '
                f"{_join_compact(forbidden_labels)}."
            )
        if runtime_authority.get('accepted_learning_authority') == 'soft_hint':
            lines.append('  Accepted learning remains soft_hint_only.')

    closure = info.get('closure_review') or {}
    if closure:
        parts = [
            f"status {closure.get('status') or 'unknown'}",
            f"repair_gap_code {closure.get('repair_gap_code') or 'none'}",
            f"repair_gap_trigger {closure.get('repair_gap_trigger') or 'none'}",
        ]
        if closure.get('repair_pending_branch_count') is not None:
            parts.append(f"repair pending branches {closure.get('repair_pending_branch_count')}")
        if closure.get('semantic_review_status'):
            parts.append(f"semantic review {closure.get('semantic_review_status')}")
        if closure.get('semantic_transition'):
            parts.append(f"transition {closure.get('semantic_transition')}")
        lines.append('- Graph closure review: ' + '; '.join(parts) + '.')

    surface = info.get('surface_state') or {}
    if surface:
        parts = [f"state {surface.get('state') or 'unknown'}"]
        if surface.get('category_counts'):
            parts.append(f"categories {surface.get('category_counts')}")
        if surface.get('item_count') is not None:
            parts.append(f"items {surface.get('item_count')}")
        if surface.get('actionability'):
            parts.append(f"actionability {surface.get('actionability')}")
        if surface.get('actionable_categories'):
            parts.append(f"actionable {surface.get('actionable_categories')}")
        if surface.get('advisory_categories'):
            parts.append(f"advisory {surface.get('advisory_categories')}")
        if surface.get('reason'):
            parts.append(f"reason {surface.get('reason')}")
        lines.append('- Surface state: ' + '; '.join(parts) + '.')

    graph_repair = info.get('graph_repair') or {}
    if graph_repair.get('proposal_count') or graph_repair.get('review_count'):
        lines.append(
            '- Graph repair proposals/reviews: '
            f"{graph_repair.get('proposal_count', 0)} proposals, "
            f"{graph_repair.get('review_count', 0)} reviews."
        )
        proposal_review_pairs = graph_repair.get('proposal_review_pairs') or []
        if proposal_review_pairs:
            proposal_items = [
                pair.get('proposal') for pair in proposal_review_pairs
                if isinstance(pair, dict) and isinstance(pair.get('proposal'), dict)
            ]
        else:
            proposal_items = graph_repair.get('proposals') or []
        for proposal in proposal_items:
            parts = [
                f"id {proposal.get('proposal_id') or 'unknown'}",
                f"status {proposal.get('status') or 'unknown'}",
                f"type {proposal.get('repair_type') or 'unknown'}",
                f"gap {proposal.get('repair_gap_code') or 'unknown'}",
            ]
            if proposal.get('branch_addition_count') is not None:
                parts.append(f"branch additions {proposal.get('branch_addition_count')}")
            if proposal.get('repair_actions'):
                parts.append(f"actions {proposal.get('repair_actions')}")
            lines.append('  proposal: ' + '; '.join(parts) + '.')
            for pair in proposal_review_pairs:
                if not isinstance(pair, dict):
                    continue
                paired_proposal = pair.get('proposal') if isinstance(pair.get('proposal'), dict) else {}
                if paired_proposal.get('proposal_id') != proposal.get('proposal_id'):
                    continue
                for review in pair.get('reviews') or []:
                    parts = [
                        f"id {review.get('review_id') or 'unknown'}",
                        f"proposal {review.get('proposal_id') or 'unknown'}",
                        f"status {review.get('status') or 'unknown'}",
                    ]
                    if review.get('reasons'):
                        parts.append(f"reasons {review.get('reasons')}")
                    lines.append('  matched review: ' + '; '.join(parts) + '.')
        review_items = graph_repair.get('unmatched_reviews') if proposal_review_pairs else graph_repair.get('reviews')
        for review in review_items or []:
            parts = [
                f"id {review.get('review_id') or 'unknown'}",
                f"proposal {review.get('proposal_id') or 'unknown'}",
                f"status {review.get('status') or 'unknown'}",
            ]
            if review.get('reasons'):
                parts.append(f"reasons {review.get('reasons')}")
            prefix = '  unmatched review: ' if proposal_review_pairs else '  review: '
            lines.append(prefix + '; '.join(parts) + '.')

    reconsideration = info.get('active_reconsideration') or {}
    if reconsideration:
        parts = [f"status {reconsideration.get('status') or 'unknown'}"]
        if reconsideration.get('decision_count') is not None:
            parts.append(f"decisions {reconsideration.get('decision_count')}")
        if reconsideration.get('category_counts'):
            parts.append(f"categories {reconsideration.get('category_counts')}")
        if reconsideration.get('recommended_actions'):
            parts.append(f"recommended actions {reconsideration.get('recommended_actions')}")
        lines.append('- Active reconsideration: ' + '; '.join(parts) + '.')

    repair_runtime = info.get('repair_runtime') or {}
    if repair_runtime:
        parts = []
        if repair_runtime.get('repair_loop_status'):
            parts.append(f"repair loop {repair_runtime.get('repair_loop_status')}")
        if repair_runtime.get('repair_loop_attempts') is not None:
            parts.append(f"attempts {repair_runtime.get('repair_loop_attempts')}")
        if repair_runtime.get('reconsideration_rebuild_status'):
            parts.append(f"reconsideration rebuild {repair_runtime.get('reconsideration_rebuild_status')}")
        parts.append(f"recovery candidates {repair_runtime.get('recovery_candidate_count', 0)}")
        parts.append(f"repair actions {repair_runtime.get('repair_action_count', 0)}")
        if repair_runtime.get('repair_actions'):
            parts.append(f"actions {repair_runtime.get('repair_actions')}")
        lines.append('- Repair runtime: ' + '; '.join(parts) + '.')

    freeze_guard = info.get('freeze_guard') or {}
    if freeze_guard:
        visible = 'yes' if freeze_guard.get('repair_needed_visible') else 'no'
        weak = 'yes' if freeze_guard.get('weak_freeze_watch') else 'no'
        suffix = f" Evidence: {freeze_guard.get('evidence')}." if freeze_guard.get('evidence') else ''
        lines.append(f'- Repair-needed visibility: {visible}; weak-freeze watch: {weak}.{suffix}')

    if not lines:
        lines.append('- No self-learning/healing diagnostics found in expanded response-frame truth.')
    return lines


def _build_report(root: Path, response_id: str, next_response_id: str | None, latest_line: dict[str, int]) -> dict[str, Any]:
    load_latest_response_state, syntax_owner, snapshot_reader = _load_response_helpers(root)
    state = load_latest_response_state(response_id, frames_dir=root / 'state/response_frames')
    payload = state.get('response_payload') or {}
    frame = state.get('response_frame') or {}
    late_fill = payload.get('late_fill') or {}

    start = _response_start(response_id)
    next_start = _response_start(next_response_id) if next_response_id else None
    window_events = _events_for_window(root, start, next_start) if start else []
    response_events = [
        (line_no, event, size)
        for line_no, event, size in window_events
        if event.get('response_id') == response_id
    ]

    chat_ok = next(
        (
            _parse_iso(event.get('timestamp'))
            for _, event, _ in window_events
            if event.get('category') == 'chat'
            and event.get('action') == 'request'
            and event.get('status') == 'ok'
            and event.get('capability') == 'chat'
        ),
        None,
    )
    image_starts = [
        _parse_iso(event.get('timestamp'))
        for _, event, _ in window_events
        if event.get('category') == 'infer'
        and event.get('action') == 'request'
        and event.get('status') == 'started'
        and event.get('capability') == 'image_generation'
    ]
    image_oks = [
        _parse_iso(event.get('timestamp'))
        for _, event, _ in window_events
        if event.get('category') == 'infer'
        and event.get('action') == 'request'
        and event.get('status') == 'ok'
        and event.get('capability') == 'image_generation'
    ]
    image_start = min((dt for dt in image_starts if dt), default=None)
    image_end = max((dt for dt in image_oks if dt), default=None)

    chat_infer_events = [
        event for _, event, _ in window_events
        if event.get('category') == 'infer'
        and event.get('action') == 'request'
        and event.get('capability') == 'chat'
    ]
    chat_pairs = []
    pending_start = None
    for event in chat_infer_events:
        dt = _parse_iso(event.get('timestamp'))
        if event.get('status') == 'started':
            pending_start = dt
        elif event.get('status') == 'ok' and pending_start:
            chat_pairs.append((pending_start, dt))
            pending_start = None

    hygiene_times = [
        _parse_iso(event.get('timestamp'))
        for _, event, _ in response_events
        if event.get('action') == 'post_response_substrate_hygiene'
    ]
    hygiene_done = max((dt for dt in hygiene_times if dt), default=None)
    late_fill_times = [
        _parse_iso(event.get('timestamp'))
        for _, event, _ in response_events
        if event.get('action') == 'late_fill'
    ]
    terminal_output = max((dt for dt in late_fill_times if dt), default=None)

    histories = late_fill.get('materialization_concurrency_history') or []
    all_branch_timings = []
    wave_diagnostics = []
    image_branch_timings = []
    coalesced_timing = None
    post_image_chat_timing = None
    for history in histories:
        wave_diagnostics.append(_wave_diagnostic_from_history(history))
        for timing in history.get('branch_timings') or []:
            branch_id = timing.get('branch_id') or ''
            branch_summary = {
                'branch_id': branch_id,
                'role': _branch_role(branch_id),
                'instance_id': timing.get('instance_id'),
                'status': timing.get('status'),
                'elapsed_seconds': _seconds_from_ms(timing.get('elapsed_ms')),
                'queued_seconds': _seconds_from_ms(timing.get('queued_elapsed_ms')),
                'lock_wait_seconds': _seconds_from_ms(timing.get('lock_wait_ms')),
                'execution_seconds': _seconds_from_ms(timing.get('execution_ms')),
                'reason': _branch_wait_reason(timing),
            }
            all_branch_timings.append(branch_summary)
            if 'image_generation' in branch_id:
                image_branch_timings.append(timing)
            elif branch_id.startswith('coalesced-text-artifacts'):
                coalesced_timing = timing
            elif branch_id == 'branch-chat-1':
                post_image_chat_timing = timing

    payload_runtime = payload.get('runtime') if isinstance(payload.get('runtime'), dict) else {}
    frame_runtime = frame.get('runtime') if isinstance(frame.get('runtime'), dict) else {}
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), dict) else {}
    current_runtime = current_state.get('runtime') if isinstance(current_state.get('runtime'), dict) else {}
    diagnostics_sources = [
        source.get('developer_diagnostics')
        for source in (payload_runtime, frame_runtime, current_runtime)
        if isinstance(source.get('developer_diagnostics'), dict)
    ]
    response_frame_finalize_timing = _first_mapping(
        [
            diagnostics.get('response_frame_finalize_timing')
            for diagnostics in diagnostics_sources
            if isinstance(diagnostics, dict)
        ]
    )
    latest_post_wave_timing = (
        late_fill.get('post_wave_backend_timing')
        if isinstance(late_fill.get('post_wave_backend_timing'), dict)
        else {}
    )
    post_wave_events = [
        event
        for _, event, _ in response_events
        if event.get('action') == 'late_fill_post_wave_backend_timing'
    ]
    event_post_wave_timings = []
    for latest_post_wave_event in post_wave_events:
        event_timing = {
            'phase': latest_post_wave_event.get('phase'),
            'status': latest_post_wave_event.get('late_fill_status') or latest_post_wave_event.get('status'),
            'finalize_elapsed_ms': latest_post_wave_event.get('finalize_elapsed_ms'),
            'touch_response_lookup_elapsed_ms': latest_post_wave_event.get('touch_response_lookup_elapsed_ms'),
            'response_frame_finalize_timing': latest_post_wave_event.get('response_frame_finalize_timing'),
            'pending_branch_count': latest_post_wave_event.get('pending_branch_count'),
            'active_branch_count': latest_post_wave_event.get('active_branch_count'),
            'completed_branch_count': latest_post_wave_event.get('completed_branch_count'),
            'failed_branch_count': latest_post_wave_event.get('failed_branch_count'),
        }
        summarized_event = _summarize_post_wave_backend_timing(event_timing)
        if summarized_event:
            summarized_event['timestamp'] = latest_post_wave_event.get('timestamp')
            event_post_wave_timings.append(summarized_event)
    backend_finalize = {}
    summarized_frame_finalize = _summarize_response_frame_finalize_timing(response_frame_finalize_timing)
    if summarized_frame_finalize:
        backend_finalize['response_frame_finalize'] = summarized_frame_finalize
    summarized_post_wave = _summarize_post_wave_backend_timing(latest_post_wave_timing)
    if summarized_post_wave:
        backend_finalize['post_wave'] = summarized_post_wave
    if event_post_wave_timings:
        backend_finalize['post_wave_events'] = event_post_wave_timings
        backend_finalize['nonterminal_events'] = [
            item
            for item in event_post_wave_timings
            if str(item.get('phase') or '').strip().lower() == 'nonterminal'
        ]
        backend_finalize['latest_event'] = event_post_wave_timings[-1]
        backend_finalize['event_count'] = len(post_wave_events)

    registry_parse_clean, registry_records = _registry_records(root, response_id)
    artifacts = _artifact_checks(payload, registry_records, syntax_owner)

    response_line_bytes = 0
    response_lines = []
    needle = response_id.encode('utf-8')
    for line_no, line, _ in _iter_jsonl(root / 'state/response_frames/responses.jsonl'):
        if needle in line:
            response_line_bytes += len(line)
            response_lines.append({'line': line_no, 'bytes': len(line)})
    registry_bytes = sum(size for _, _, size in registry_records)
    response_event_bytes = sum(size for _, _, size in response_events)
    snapshot_files = []
    if start:
        for path in (root / 'state/response_frames/snapshots').rglob('*.json'):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime >= start.timestamp() and (not next_start or stat.st_mtime < next_start.timestamp()):
                snapshot_files.append({'path': str(path), 'bytes': stat.st_size})
    chat_history_bytes = 0
    if start:
        for path in (root / 'state/chat_history').glob('*'):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime >= start.timestamp() and (not next_start or stat.st_mtime < next_start.timestamp()):
                chat_history_bytes += stat.st_size

    branch_counts = {
        'pending': len(late_fill.get('pending_branches') or []),
        'active': len(late_fill.get('active_branches') or []),
        'failed': len(late_fill.get('failed_branches') or []),
        'completed': len(late_fill.get('completed_branches') or []),
    }
    clean = (
        payload.get('status') == 'completed'
        and payload.get('lifecycle_state') == 'completed'
        and late_fill.get('status') == 'completed'
        and late_fill.get('final_materialization_contract_status') == 'fulfilled'
        and branch_counts['pending'] == 0
        and branch_counts['active'] == 0
        and branch_counts['failed'] == 0
        and not late_fill.get('materialization_contract_unmet')
        and not artifacts['missing_files']
        and not artifacts['sha_mismatches']
        and registry_parse_clean
        and not artifacts['html_issues']
        and not artifacts['css_issues']
    )
    terminal_needs_attention = (
        payload.get('lifecycle_state') in {'repair_needed', 'failed', 'cancelled'}
        or late_fill.get('status') in {'partial_failed', 'failed', 'cancelled'}
        or late_fill.get('final_materialization_contract_status') == 'unmet'
        or bool(late_fill.get('materialization_contract_unmet'))
        or bool(late_fill.get('failed_branch_count'))
        or branch_counts['failed'] > 0
    )
    terminal = clean or terminal_needs_attention
    running = (
        not terminal
        and (
            branch_counts['pending'] > 0
            or branch_counts['active'] > 0
            or payload.get('lifecycle_state') in {'late_fill_running', 'running'}
            or late_fill.get('status') in {'running', 'pending'}
        )
    )
    notes = []
    for _, event, _ in response_events:
        message = event.get('message') or ''
        low = message.lower()
        if any(token in low for token in ['repair', 'requeue', 'saved_text_path', 'without', 'failure', 'failed']):
            notes.append(message)
    if artifacts['weak_viewport_tags']:
        notes.append('Weak viewport tag detected.')
    duplicate_identity_lines = _duplicate_ref_lines(artifacts.get('duplicate_ref_summary') or {})
    if duplicate_identity_lines:
        notes.extend(line[2:] if line.startswith('- ') else line for line in duplicate_identity_lines)
    if artifacts['missing_files']:
        notes.append('Missing artifact files: ' + ', '.join(artifacts['missing_files']))
    if artifacts['sha_mismatches']:
        notes.append('SHA mismatch for: ' + ', '.join(artifacts['sha_mismatches']))
    if not notes:
        notes.append('No repair/requeue/materialization failure events observed.')

    size_state_add = response_line_bytes + registry_bytes + response_event_bytes + sum(item['bytes'] for item in snapshot_files) + chat_history_bytes
    size_totals = {
        'state_total_bytes': _walk_size(root / 'state'),
        'snapshots_total_bytes': _walk_size(root / 'state/response_frames/snapshots'),
        'artifacts_total_bytes': _walk_size(root / 'artifacts'),
        'state_add_approx_bytes': size_state_add,
        'snapshots_add_bytes': sum(item['bytes'] for item in snapshot_files),
        'artifacts_add_bytes': artifacts['artifact_bytes'],
        'latest_response_line_bytes': latest_line.get('bytes'),
        'new_snapshot_file_count': len(snapshot_files),
        'new_artifact_file_count': artifacts['artifact_count'],
    }

    timing = {
        'start': start.isoformat().replace('+00:00', 'Z') if start else None,
        'initial_chat_finished': chat_ok.isoformat().replace('+00:00', 'Z') if chat_ok else None,
        'initial_chat_seconds': _duration(start, chat_ok),
        'image_wave_start': image_start.isoformat().replace('+00:00', 'Z') if image_start else None,
        'image_wave_end': image_end.isoformat().replace('+00:00', 'Z') if image_end else None,
        'image_wave_seconds': _duration(image_start, image_end),
        'terminal_output': terminal_output.isoformat().replace('+00:00', 'Z') if terminal_output else None,
        'terminal_seconds': _duration(start, terminal_output),
        'hygiene_finished': hygiene_done.isoformat().replace('+00:00', 'Z') if hygiene_done else None,
        'hygiene_seconds': _duration(start, hygiene_done),
    }
    if post_image_chat_timing:
        timing['post_image_chat_seconds'] = (post_image_chat_timing.get('elapsed_ms') or 0) / 1000.0
    elif chat_pairs:
        timing['post_image_chat_seconds'] = _duration(*chat_pairs[0])
    if coalesced_timing:
        timing['coalesced_text_seconds'] = (coalesced_timing.get('elapsed_ms') or 0) / 1000.0
    elif chat_pairs:
        timing['coalesced_text_seconds'] = _duration(*chat_pairs[-1])

    first_chat_pair_start = chat_pairs[0][0] if chat_pairs else None
    scheduling_policy = late_fill.get('scheduling_policy')
    if not isinstance(scheduling_policy, dict):
        scheduling_policy = {}
    timing_diagnostics = {
        'phase_gaps': {
            'initial_chat_to_image_start_seconds': _duration(chat_ok, image_start),
            'image_end_to_text_start_seconds': _duration(image_end, first_chat_pair_start),
            'terminal_to_hygiene_seconds': _duration(terminal_output, hygiene_done),
        },
        'scheduling_policy': scheduling_policy,
        'waves': wave_diagnostics,
        'branches': all_branch_timings,
        'backend_finalize': backend_finalize,
        'legend': {
            'planning': 'runtime branch preparation and route/instance selection',
            'queued': 'scheduler or selected-instance queue before work entered backend execution',
            'lock_wait': 'time waiting for the selected instance execution lock',
            'execution': 'backend model/tool execution time',
            'phase_gap': 'handoff time between completed phases, rebind/finalization, or hygiene scheduling',
        },
    }

    image_models = {}
    for timing_item in image_branch_timings:
        instance_id = timing_item.get('instance_id') or 'unknown'
        image_models[instance_id] = image_models.get(instance_id, 0) + 1

    all_late_fill_branches: list[dict[str, Any]] = []
    for key in ('pending_branches', 'active_branches', 'failed_branches', 'completed_branches'):
        for branch in late_fill.get(key) or []:
            if isinstance(branch, dict):
                all_late_fill_branches.append(branch)
    branch_role_counts = _count_values(
        [_branch_record_role(branch) for branch in all_late_fill_branches]
        or [branch.get('role') for branch in all_branch_timings]
    )
    branch_capability_counts = _count_values(
        [
            branch.get('capability')
            for branch in all_late_fill_branches
            if isinstance(branch, dict)
        ]
    )
    if not branch_capability_counts:
        branch_capability_counts = _count_values([branch.get('role') for branch in all_branch_timings])

    learning_healing = _collect_learning_healing(
        root,
        payload,
        frame,
        snapshot_reader,
        monitor_report={
            'response_id': response_id,
            'status': payload.get('status'),
            'lifecycle_state': payload.get('lifecycle_state'),
            'late_fill_status': late_fill.get('status'),
            'final_materialization_contract_status': late_fill.get('final_materialization_contract_status'),
            'materialization_contract_unmet': late_fill.get('materialization_contract_unmet'),
            'branch_counts': branch_counts,
            'artifacts': artifacts,
        },
    )

    report = {
        'response_id': response_id,
        'reported_at': _utc_now(),
        'frame_id': frame.get('frame_id'),
        'frame_sequence': frame.get('frame_sequence'),
        'status': payload.get('status'),
        'lifecycle_state': payload.get('lifecycle_state'),
        'late_fill_status': late_fill.get('status'),
        'final_materialization_contract_status': late_fill.get('final_materialization_contract_status'),
        'branch_counts': branch_counts,
        'failed_branch_count': late_fill.get('failed_branch_count'),
        'materialization_contract_unmet': late_fill.get('materialization_contract_unmet'),
        'verdict': 'running' if running else ('clean' if clean else 'needs_attention'),
        'timing': timing,
        'timing_diagnostics': timing_diagnostics,
        'sizes': size_totals,
        'artifacts': artifacts,
        'registry_parse_clean': registry_parse_clean,
        'registry_record_count': len(registry_records),
        'image_models': image_models,
        'image_branch_count': len(image_branch_timings),
        'branch_role_counts': branch_role_counts,
        'branch_capability_counts': branch_capability_counts,
        'output_obligation_counts': _collect_output_obligations(payload, frame),
        'text_artifact_branch_count': len([b for b in late_fill.get('completed_branches') or [] if b.get('output_type') == 'text']),
        'learning_healing': learning_healing,
        'notes': notes,
    }
    report['human_report'] = _render_human(report)
    return report


def _render_human(report: dict[str, Any]) -> str:
    verdict = report['verdict']
    if verdict == 'clean':
        first = 'Latest run is clean at the runtime truth level.'
    elif verdict == 'running':
        first = 'Latest run is still running at the runtime truth level.'
    else:
        first = 'Latest run needs attention at the runtime truth level.'

    timing = report['timing']
    sizes = report['sizes']
    state_total = sizes['state_total_bytes']
    state_add = sizes['state_add_approx_bytes']
    snap_total = sizes['snapshots_total_bytes']
    snap_add = sizes['snapshots_add_bytes']
    art_total = sizes['artifacts_total_bytes']
    art_add = sizes['artifacts_add_bytes']

    image_models = report.get('image_models') or {}
    model_bits = ', '.join(f'{name} handled {count}' for name, count in sorted(image_models.items())) or 'no image branches'
    notes = '\n'.join(report.get('notes') or ['No notes.'])
    artifact_info = report.get('artifacts') or {}
    html_image_links = artifact_info.get('html_image_links') or []
    all_links_ok = all(item['exists'] for item in html_image_links)
    branch_role_counts = report.get('branch_role_counts') or {}
    branch_capability_counts = report.get('branch_capability_counts') or {}
    output_obligation_counts = report.get('output_obligation_counts') or {}
    artifact_kind_counts = artifact_info.get('artifact_kind_counts') or {}
    artifact_suffix_counts = artifact_info.get('artifact_file_count_by_suffix') or {}
    output_ref_counts = artifact_info.get('output_ref_counts') or {}
    audio_required_count = sum(
        count
        for key, count in output_obligation_counts.items()
        if _is_audio_kind(key)
    )
    audio_artifact_count = artifact_info.get('audio_artifact_count') or 0
    branch_counts = report.get('branch_counts') or {}
    diagnostics = report.get('timing_diagnostics') or {}
    phase_gaps = diagnostics.get('phase_gaps') or {}
    scheduling_policy = diagnostics.get('scheduling_policy') or {}
    waves = diagnostics.get('waves') or []
    branches = diagnostics.get('branches') or []
    backend_finalize = diagnostics.get('backend_finalize') or {}
    learning_healing_lines = _render_learning_healing_lines(report.get('learning_healing') or {})

    diagnostic_lines = []
    if scheduling_policy:
        diagnostic_lines.append(
            '- Late-fill scheduling policy: '
            f"{_format_scheduling_policy(scheduling_policy)}."
        )
    chat_to_image = phase_gaps.get('initial_chat_to_image_start_seconds')
    image_to_text = phase_gaps.get('image_end_to_text_start_seconds')
    terminal_to_hygiene = phase_gaps.get('terminal_to_hygiene_seconds')
    if chat_to_image is not None:
        diagnostic_lines.append(
            f"- Chat finish -> image wave start: {_fmt_seconds(chat_to_image)}; phase handoff/materialization scheduling."
        )
    if image_to_text is not None:
        diagnostic_lines.append(
            f"- Image wave end -> text pass start: {_fmt_seconds(image_to_text)}; dependency evidence handoff into text materialization."
        )
    if terminal_to_hygiene is not None:
        diagnostic_lines.append(
            f"- Terminal output -> hygiene done: {_fmt_seconds(terminal_to_hygiene)}; post-response cleanup/substrate hygiene."
        )
    post_wave_events = backend_finalize.get('post_wave_events') if isinstance(backend_finalize.get('post_wave_events'), list) else []
    post_wave_backends = [
        item for item in post_wave_events if isinstance(item, dict)
    ] or [
        item
        for item in (
            backend_finalize.get('post_wave'),
            backend_finalize.get('latest_event'),
        )
        if isinstance(item, dict) and item
    ]
    response_finalize = backend_finalize.get('response_frame_finalize') or {}
    for index, post_wave_backend in enumerate(post_wave_backends, 1):
        event_prefix = (
            f'#{index} '
            if len(post_wave_backends) > 1
            else ''
        )
        diagnostic_lines.append(
            '- Backend post-wave finalize '
            f"{event_prefix}{post_wave_backend.get('phase') or 'unknown'} status "
            f"{post_wave_backend.get('status') or 'unknown'}, "
            f"finalize {_fmt_seconds(post_wave_backend.get('finalize_seconds'))}, "
            f"lookup touch {_fmt_seconds(post_wave_backend.get('touch_response_lookup_seconds'))}; "
            f"branches pending={post_wave_backend.get('pending_branch_count')}, "
            f"active={post_wave_backend.get('active_branch_count')}, "
            f"completed={post_wave_backend.get('completed_branch_count')}, "
            f"failed={post_wave_backend.get('failed_branch_count')}."
        )
        if isinstance(post_wave_backend.get('response_frame_finalize'), dict):
            response_finalize = post_wave_backend.get('response_frame_finalize')
    if response_finalize:
        step_parts = []
        for step in response_finalize.get('steps') or []:
            name = step.get('name')
            if not name:
                continue
            step_parts.append(f"{name} {_fmt_seconds(step.get('elapsed_seconds'))}")
            if len(step_parts) >= 6:
                break
        step_suffix = f"; steps {', '.join(step_parts)}" if step_parts else ''
        diagnostic_lines.append(
            '- Backend response-frame finalize timing: '
            f"{response_finalize.get('phase') or 'unknown'} total "
            f"{_fmt_seconds(response_finalize.get('total_seconds'))}, "
            f"persist_effective={response_finalize.get('persist_effective')}"
            f"{step_suffix}."
        )
    for index, wave in enumerate(waves, 1):
        worker = wave.get('worker_count')
        max_workers = wave.get('max_parallel_workers')
        worker_text = (
            f'{worker}/{max_workers} max'
            if max_workers is not None
            else str(worker if worker is not None else 'unknown')
        )
        default_workers = wave.get('default_worker_count')
        capacity = wave.get('scheduling_capacity_units')
        default_text = default_workers if default_workers is not None else 'unknown'
        capacity_text = capacity if capacity is not None else 'unknown'
        source = wave.get('worker_count_source') or 'unknown'
        prepared = wave.get('prepared_branch_count')
        instance_count = wave.get('distinct_instance_count')
        guard = wave.get('gpu_heavy_guard') or 'unknown'
        reasons = '; '.join(wave.get('worker_limit_reasons') or ['no worker cap signal recorded'])
        progress_bits = []
        if wave.get('branch_progress_dispatch') is not None:
            progress_bits.append(f"progress dispatch {wave.get('branch_progress_dispatch')}")
        if wave.get('branch_progress_callback_count') is not None:
            progress_bits.append(f"callbacks {wave.get('branch_progress_callback_count')}")
        progress_text = f"{', '.join(progress_bits)}; " if progress_bits else ''
        diagnostic_lines.append(
            '- Wave '
            f"{index}: planning {_fmt_seconds(wave.get('planning_seconds'))}, "
            f"elapsed {_fmt_seconds(wave.get('elapsed_seconds'))}, "
            f"workers {worker_text} "
            f"(default {default_text}, source {source}, capacity {capacity_text}), "
            f"prepared branches {prepared}, selected instances {instance_count}, "
            f"gpu-heavy guard {guard}; {progress_text}{reasons}."
        )
        lock_groups = _format_instance_groups(wave.get('same_instance_lock_groups'))
        if lock_groups:
            diagnostic_lines.append(f'- Wave {index} same-instance lock groups: {lock_groups}.')
    for branch in branches[:8]:
        diagnostic_lines.append(
            '- '
            f"{branch.get('role')} {branch.get('branch_id')} on {branch.get('instance_id')}: "
            f"total {_fmt_seconds(branch.get('elapsed_seconds'))}, "
            f"queue {_fmt_seconds(branch.get('queued_seconds'))}, "
            f"lock {_fmt_seconds(branch.get('lock_wait_seconds'))}, "
            f"execution {_fmt_seconds(branch.get('execution_seconds'))}; "
            f"{branch.get('reason')}."
        )
    if len(branches) > 8:
        diagnostic_lines.append(f"- {len(branches) - 8} additional branch timing rows omitted from human summary; full JSON has them.")

    bad_lines = []
    if verdict == 'running':
        bad_lines.append('- Run is still active; no terminal failure verdict yet.')
    elif verdict != 'clean':
        bad_lines.append(
            '- Terminal state is '
            f"{report.get('lifecycle_state')} with late_fill.{report.get('late_fill_status')} "
            f"and final materialization contract {report.get('final_materialization_contract_status')}."
        )
    if report.get('materialization_contract_unmet'):
        bad_lines.append('- Final materialization contract is unmet.')
    if report.get('failed_branch_count') or branch_counts.get('failed'):
        bad_lines.append(
            '- Failed branches: '
            f"{branch_counts.get('failed', 0)} in final frame; "
            f"failed_branch_count={report.get('failed_branch_count')}."
        )
    if branch_counts.get('pending') and verdict != 'running':
        bad_lines.append(
            f"- Terminal frame still carries {branch_counts.get('pending')} pending branch entries; treated as terminal failure evidence, not live progress."
        )
    if artifact_info.get('missing_files'):
        bad_lines.append('- Missing artifact files: ' + ', '.join(artifact_info.get('missing_files') or []))
    if artifact_info.get('sha_mismatches'):
        bad_lines.append('- SHA mismatches: ' + ', '.join(artifact_info.get('sha_mismatches') or []))
    if artifact_info.get('html_issues') or artifact_info.get('css_issues'):
        bad_lines.append(
            '- Syntax sanity issues: '
            f"{len(artifact_info.get('html_issues') or [])} HTML, "
            f"{len(artifact_info.get('css_issues') or [])} CSS."
        )
        for issue in (artifact_info.get('html_issues') or [])[:2]:
            bad_lines.append(f"  HTML: {issue}")
        for issue in (artifact_info.get('css_issues') or [])[:2]:
            bad_lines.append(f"  CSS: {issue}")
    broken_links = [item.get('src') for item in artifact_info.get('html_image_links') or [] if not item.get('exists')]
    if broken_links:
        bad_lines.append('- Broken HTML image links: ' + ', '.join(str(item) for item in broken_links))
    if artifact_info.get('weak_viewport_tags'):
        bad_lines.append('- Weak viewport tag detected.')
    duplicate_ref_summary = artifact_info.get('duplicate_ref_summary') or {}
    bad_lines.extend(_duplicate_ref_lines(duplicate_ref_summary))
    if audio_required_count and not audio_artifact_count:
        bad_lines.append(
            '- Audio output obligation is present, but no saved audio artifact was detected in final artifacts.'
        )
    if not report.get('registry_parse_clean'):
        bad_lines.append('- Artifact registry did not parse cleanly.')
    if not bad_lines:
        bad_lines.append('- No failed branches, unmet contracts, missing files, SHA mismatches, broken links, or syntax issues detected.')

    lines = [
        first,
        '',
        f"Response: {report['response_id']}",
        f"Final frame: frame-{report.get('frame_sequence')}",
        (
            'State: '
            f"{report.get('lifecycle_state')}, "
            f"late_fill.{report.get('late_fill_status')}, "
            f"final materialization contract {report.get('final_materialization_contract_status')}."
        ),
        '',
        'Timing shape:',
        '',
        f"Start/response id: {_fmt_time(_parse_iso(timing.get('start')))}",
        (
            'Initial Gemma chat/graph finished: '
            f"{_fmt_time(_parse_iso(timing.get('initial_chat_finished')))}, "
            f"about {_fmt_seconds(timing.get('initial_chat_seconds'))}"
        ),
        (
            'Image wave: '
            f"{_fmt_time(_parse_iso(timing.get('image_wave_start')))} to "
            f"{_fmt_time(_parse_iso(timing.get('image_wave_end')))}, "
            f"about {_fmt_seconds(timing.get('image_wave_seconds'))}"
        ),
    ]
    if 'post_image_chat_seconds' in timing:
        lines.append(f"Post-image chat pass: about {_fmt_seconds(timing.get('post_image_chat_seconds'))}")
    if 'coalesced_text_seconds' in timing:
        lines.append(f"Coalesced text pass: about {_fmt_seconds(timing.get('coalesced_text_seconds'))}")
    lines.extend(
        [
            f"Post-response hygiene finished: {_fmt_time(_parse_iso(timing.get('hygiene_finished')))}",
            (
                'So: about '
                f"{_fmt_seconds(timing.get('terminal_seconds'))} to terminal output, "
                f"{_fmt_seconds(timing.get('hygiene_seconds'))} including hygiene."
            ),
            '',
            'Size growth:',
            '',
            f"state: {_fmt_size(state_total)}, previously about {_fmt_size(state_total - state_add)}, about +{_fmt_size(state_add)}",
            f"snapshots: {_fmt_size(snap_total)}, previously about {_fmt_size(snap_total - snap_add)}, about +{_fmt_size(snap_add)}",
            f"artifacts: {_fmt_size(art_total)}, previously about {_fmt_size(art_total - art_add)}, about +{_fmt_size(art_add)}",
            f"latest response ledger line: {sizes.get('latest_response_line_bytes'):,} bytes",
            f"new snapshots for this run: {sizes.get('new_snapshot_file_count')} files, about {_fmt_size(snap_add)}",
            f"new artifacts: {sizes.get('new_artifact_file_count')} files, about {_fmt_size(art_add)}",
            '',
            'What improved / stayed good:',
            '',
            f"Branch roles: {_format_counts(branch_role_counts)}.",
            f"Capability routing/work: {_format_counts(branch_capability_counts)}.",
            f"Image routing: {model_bits}.",
            f"Output obligations: {_format_counts(output_obligation_counts)}.",
            f"Output artifact refs: {_format_counts(output_ref_counts)}.",
            f"Artifact kinds: {_format_counts(artifact_kind_counts)}.",
            f"Artifact file suffixes: {_format_counts(artifact_suffix_counts)}.",
            f"Audio artifacts: {audio_artifact_count} saved.",
            (
                f"HTML image links resolve: {'yes' if all_links_ok else 'no'}."
                if html_image_links
                else 'HTML image links: not applicable.'
            ),
            f"All artifact files exist: {'yes' if not report['artifacts'].get('missing_files') else 'no'}.",
            f"Registry parses cleanly: {'yes' if report.get('registry_parse_clean') else 'no'}.",
            (
                'Current syntax checker reports '
                f"{len(report['artifacts'].get('html_issues') or [])} HTML issues and "
                f"{len(report['artifacts'].get('css_issues') or [])} CSS issues."
            ),
            '',
            'What went bad / needs attention:',
            '',
            '\n'.join(bad_lines),
            '',
            'Learning / healing diagnostics:',
            '',
            '\n'.join(learning_healing_lines),
            '',
            'Notes:',
            '',
            notes,
            '',
            'Timing diagnostics / waits:',
            '',
            '\n'.join(diagnostic_lines) if diagnostic_lines else 'No detailed wait diagnostics available.',
        ]
    )
    return '\n'.join(lines)


def _load_last_seen(state_dir: Path) -> dict[str, Any]:
    path = state_dir / 'last_seen.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_last_seen(state_dir: Path, data: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / 'last_seen.json.tmp'
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(state_dir / 'last_seen.json')


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(text)


def run_once(root: Path, state_dir: Path, quiet: bool) -> int:
    order, latest = _scan_responses(root)
    if not order:
        if not quiet:
            print('No Ollmo responses found.')
        return 0

    last_seen = _load_last_seen(state_dir)
    last_terminal = last_seen.get('terminal_reported_response_id')
    if last_terminal in order:
        new_candidate_ids = order[order.index(last_terminal) + 1:]
    else:
        new_candidate_ids = [order[-1]]

    retry_running_ids = [
        response_id
        for response_id in last_seen.get('running_unreported_response_ids') or []
        if response_id in latest
    ]
    candidate_ids = []
    seen_candidate_ids = set()
    for response_id in [*retry_running_ids, *new_candidate_ids]:
        if response_id in seen_candidate_ids:
            continue
        candidate_ids.append(response_id)
        seen_candidate_ids.add(response_id)

    emitted = 0
    still_running_ids: list[str] = []
    for response_id in candidate_ids:
        next_index = order.index(response_id) + 1
        next_id = order[next_index] if next_index < len(order) else None
        report = _build_report(root, response_id, next_id, latest[response_id])

        if report['verdict'] == 'running':
            # Terminal-only monitor: do not emit or persist live/running snapshots.
            # The same response id will be reconsidered on a later heartbeat once it
            # reaches completed, repair_needed, failed, or another terminal state.
            still_running_ids.append(response_id)
            continue

        _append(state_dir / 'reports.jsonl', json.dumps(report, sort_keys=True) + '\n')
        _append(
            state_dir / 'reports.md',
            f"\n\n## {report['response_id']} ({report['reported_at']})\n\n{report['human_report']}\n",
        )
        current_terminal = last_seen.get('terminal_reported_response_id')
        should_advance_terminal = current_terminal not in order or order.index(response_id) >= order.index(current_terminal)
        if should_advance_terminal:
            last_seen.update(
                {
                    'terminal_reported_response_id': response_id,
                    'terminal_reported_frame_id': report.get('frame_id'),
                    'terminal_reported_at': report.get('reported_at'),
                }
            )
        else:
            last_seen.update(
                {
                    'backfill_terminal_reported_response_id': response_id,
                    'backfill_terminal_reported_frame_id': report.get('frame_id'),
                    'backfill_terminal_reported_at': report.get('reported_at'),
                }
            )
        last_seen.update(
            {
                'idle_no_running_check_count': 0,
                'idle_no_running_last_checked_at': None,
                'idle_no_running_latest_response_id': None,
                'running_snapshot_response_id': None,
                'idle_no_running_auto_pause_recommended': False,
            }
        )
        _save_last_seen(state_dir, last_seen)
        if not quiet:
            print(report['human_report'])
            print()
        emitted += 1

    if emitted:
        last_seen.update(
            {
                'idle_no_running_check_count': 0,
                'idle_no_running_last_checked_at': None,
                'idle_no_running_latest_response_id': None,
                'running_unreported_response_ids': still_running_ids,
                'idle_no_running_auto_pause_recommended': False,
            }
        )
        _save_last_seen(state_dir, last_seen)
        return 0

    if still_running_ids:
        last_seen.update(
            {
                'idle_no_running_check_count': 0,
                'idle_no_running_last_checked_at': None,
                'idle_no_running_latest_response_id': None,
                'running_unreported_response_ids': still_running_ids,
                'idle_no_running_auto_pause_recommended': False,
            }
        )
        _save_last_seen(state_dir, last_seen)
        if not quiet:
            print(
                json.dumps(
                    {
                        'latest_response_id': order[-1],
                        'running_response_ids': still_running_ids,
                        'status': 'no_new_terminal_reports',
                    },
                    sort_keys=True,
                )
            )
        return 0

    if last_seen.get('idle_no_running_auto_pause_recommended'):
        idle_count = 0
    else:
        idle_count = int(last_seen.get('idle_no_running_check_count') or 0) + 1
    now = _utc_now()
    last_seen.update(
        {
            'idle_no_running_check_count': idle_count,
            'idle_no_running_last_checked_at': now,
            'idle_no_running_latest_response_id': order[-1],
            'running_unreported_response_ids': [],
            'idle_no_running_auto_pause_recommended': False,
        }
    )
    _save_last_seen(state_dir, last_seen)

    if emitted == 0 and not quiet:
        status = (
            'auto_pause_recommended'
            if idle_count >= IDLE_AUTO_PAUSE_CHECKS
            else ('no_new_terminal_reports' if candidate_ids else 'no_change')
        )
        payload = {
            'auto_pause_threshold': IDLE_AUTO_PAUSE_CHECKS,
            'idle_no_running_check_count': idle_count,
            'latest_response_id': order[-1],
            'nothing_running': True,
            'status': status,
        }
        if status == 'auto_pause_recommended':
            payload['reason'] = 'no_running_responses_after_consecutive_checks'
            last_seen['idle_no_running_auto_pause_recommended'] = True
            _save_last_seen(state_dir, last_seen)
        print(json.dumps(payload, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Append Ollmo run monitor reports from local runtime truth.')
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--state-dir', default=None)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    state_dir = Path(args.state_dir).resolve() if args.state_dir else root / 'state/ollmo_run_monitor'
    return run_once(root, state_dir, args.quiet)


if __name__ == '__main__':
    raise SystemExit(main())
