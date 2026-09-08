#!/usr/bin/env python3
"""Export and verify self-contained, sanitized Ollmo reference runs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import mimetypes
import os
import re
import shutil
import struct
import sys
import tempfile
import wave
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ollmo_services.response_frames import (  # noqa: E402
    load_latest_response_observation_state,
    load_latest_response_wire_state,
    load_response_frame_index,
)


DEFAULT_FRAMES_DIR = REPO_ROOT / 'state' / 'response_frames'
DEFAULT_MONITOR_REPORTS = (
    REPO_ROOT / 'state' / 'ollmo_run_monitor' / 'reports.jsonl'
)
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / 'artifacts'

EXPORT_SCHEMA = 'ollmo.public_reference_run.v1'
MONITOR_SCHEMA = 'ollmo.public_monitor_snapshot.v1'
MANIFEST_SCHEMA = 'ollmo.reference_run_manifest.v1'
BUNDLE_SCHEMA = 'ollmo.public_bundle_manifest.v1'

LOCAL_PATH_PATTERNS = (
    re.compile(r'/Users/[^/\s]+/'),
    re.compile(r'file://', re.IGNORECASE),
    re.compile(r'[A-Za-z]:\\Users\\'),
)
MARKDOWN_LINK = re.compile(r'\[[^\]]*\]\((?P<target>[^)]+)\)')
REQUEST_BLOCK = re.compile(
    r'## Request\s*\n\s*```text\n(?P<prompt>.*?)\n```',
    re.DOTALL,
)


class ReferenceExportError(RuntimeError):
    """Raised when a reference run cannot be admitted or verified."""


def _required_frame_sequence(value: Any, *, label: str) -> int:
    if value in (None, '') or isinstance(value, bool):
        raise ReferenceExportError(f'{label} is missing or malformed.')
    try:
        sequence = int(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceExportError(f'{label} is missing or malformed.') from exc
    if sequence <= 0:
        raise ReferenceExportError(f'{label} is missing or malformed.')
    return sequence


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    ).encode('utf-8')


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    path.chmod(0o644)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str | Path) -> Path:
    text = str(value or '').replace('\\', '/').strip()
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or any(part in {'', '.', '..'} for part in pure.parts)
    ):
        raise ReferenceExportError(f'Unsafe package-relative path: {value!s}')
    return Path(*pure.parts)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_regular_source(path: Path, *, root: Path, label: str) -> Path:
    resolved_root = root.resolve()
    unresolved = path.absolute()
    resolved = path.resolve()
    if not _is_within(resolved, resolved_root):
        raise ReferenceExportError(f'{label} is outside the approved root.')
    if not resolved.is_file():
        raise ReferenceExportError(f'{label} is missing or not a regular file.')
    current = unresolved
    while _is_within(current, resolved_root):
        if current.is_symlink():
            raise ReferenceExportError(f'{label} traverses a symlink.')
        if current == resolved_root:
            break
        current = current.parent
    return resolved


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def _extract_prompt(readme_text: str) -> str:
    match = REQUEST_BLOCK.search(readme_text)
    if not match:
        raise ReferenceExportError('Source README has no fenced Request block.')
    prompt = match.group('prompt')
    if not prompt.strip():
        raise ReferenceExportError('Source README Request block is empty.')
    return prompt


def _single_user_message_repr(prompt: str) -> str:
    return repr(
        [
            {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': prompt}],
            }
        ]
    )


def _input_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, Mapping):
        if value.get('type') == 'input_text' and isinstance(value.get('text'), str):
            texts.append(str(value['text']))
        for child in value.values():
            texts.extend(_input_texts(child))
    elif isinstance(value, list):
        for child in value:
            texts.extend(_input_texts(child))
    return texts


def _verify_prompt(prompt: str, request: Mapping[str, Any]) -> dict[str, Any]:
    recorded = str(request.get('prompt') or '')
    truncated = bool(request.get('prompt_preview_truncated'))
    if recorded and not truncated:
        try:
            parsed = ast.literal_eval(recorded)
        except (SyntaxError, ValueError) as exc:
            raise ReferenceExportError(
                'The indexed request prompt is not safely parseable.'
            ) from exc
        if _input_texts(parsed) != [prompt]:
            raise ReferenceExportError(
                'The reviewed prompt does not match indexed request truth.'
            )
        serialized = recorded
    else:
        serialized = _single_user_message_repr(prompt)
        recorded_sha = str(request.get('prompt_sha256') or '').strip()
        recorded_length = int(request.get('prompt_length_chars') or 0)
        if (
            not recorded_sha
            or _sha256_bytes(serialized.encode('utf-8')) != recorded_sha
            or len(serialized) != recorded_length
        ):
            raise ReferenceExportError(
                'The reviewed prompt does not match the indexed prompt digest.'
            )
    return {
        'prompt_path': 'prompt.txt',
        'prompt_sha256': _sha256_bytes(prompt.encode('utf-8')),
        'runtime_request_sha256': _sha256_bytes(serialized.encode('utf-8')),
        'verified': True,
    }


def _read_exact_indexed_record(
    response_id: str,
    *,
    index_state: Mapping[str, Any],
) -> dict[str, Any]:
    responses = index_state.get('responses')
    entry = responses.get(response_id) if isinstance(responses, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ReferenceExportError('The response is not present in the frame index.')
    ledger_text = str(entry.get('ledger_path') or index_state.get('ledger_path') or '')
    ledger_path = Path(ledger_text)
    offset = int(entry.get('byte_offset') or 0)
    length = int(entry.get('line_length') or 0)
    if not ledger_text or offset < 0 or length <= 0:
        raise ReferenceExportError('The response index entry has no exact row coordinate.')
    before = ledger_path.stat()
    with ledger_path.open('rb') as handle:
        handle.seek(offset)
        row = handle.read(length)
    after = ledger_path.stat()
    if (
        len(row) != length
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ReferenceExportError('The response ledger changed during exact lookup.')
    try:
        payload = json.loads(row)
    except json.JSONDecodeError as exc:
        raise ReferenceExportError('The exact indexed response row is invalid JSON.') from exc
    expected_frame_id = str(entry.get('latest_frame_id') or '').strip()
    if not expected_frame_id:
        raise ReferenceExportError(
            'The indexed response frame has no immutable frame identity.'
        )
    expected_sequence = _required_frame_sequence(
        entry.get('latest_frame_sequence'),
        label='The indexed response frame sequence',
    )
    try:
        payload_sequence = _required_frame_sequence(
            payload.get('frame_sequence'),
            label='The exact indexed response frame sequence',
        )
    except ReferenceExportError:
        raise
    if (
        payload.get('response_id') != response_id
        or payload.get('frame_id') != expected_frame_id
        or payload_sequence != expected_sequence
    ):
        raise ReferenceExportError('The exact indexed row has a mismatched identity.')
    return {
        'frame_id': expected_frame_id,
        'frame_sequence': expected_sequence,
        'record_sha256': _sha256_bytes(row),
        'record_size_bytes': len(row),
    }


def _load_monitor_report(
    response_id: str,
    *,
    reports_path: Path,
    expected_frame_id: str | None = None,
    expected_frame_sequence: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], bytes]] = []
    with reports_path.open('rb') as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get('response_id') or '') == response_id:
                matches.append((payload, line))
    if expected_frame_id not in (None, ''):
        normalized_expected_frame_id = str(expected_frame_id)
        exact_matches: list[tuple[dict[str, Any], bytes]] = []
        for item in matches:
            payload = item[0]
            if str(payload.get('frame_id') or '') != normalized_expected_frame_id:
                continue
            candidate_sequence = payload.get('frame_sequence')
            if expected_frame_sequence not in (None, '') and candidate_sequence not in (
                None,
                '',
            ):
                parsed_sequence = _required_frame_sequence(
                    candidate_sequence,
                    label='The monitor frame sequence',
                )
                if parsed_sequence != int(expected_frame_sequence):
                    continue
            exact_matches.append(item)
        matches = exact_matches
        if not matches:
            raise ReferenceExportError(
                'No monitor record matches the indexed latest frame.'
            )
        # Duplicate snapshots for the same immutable frame are harmless; the
        # last append is the most recent observer record for that frame.
        payload, raw_line = matches[-1]
    elif len(matches) != 1:
        raise ReferenceExportError(
            f'Expected exactly one monitor record, found {len(matches)}.'
        )
    else:
        payload, raw_line = matches[0]
    return payload, {
        'record_sha256': _sha256_bytes(raw_line),
        'record_size_bytes': len(raw_line),
    }


def _fixed_scalars(source: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {
        key: source.get(key)
        for key in keys
        if source.get(key) not in (None, '', [], {})
    }


def _require_terminal_truth(
    response_id: str,
    *,
    wire: Mapping[str, Any],
    observation: Mapping[str, Any],
    monitor: Mapping[str, Any],
    indexed_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not wire.get('ok') or not observation.get('ok'):
        raise ReferenceExportError('Bounded response projections are unavailable.')
    if wire.get('ledger_fallback_used') is not False:
        raise ReferenceExportError('The wire lookup was not strictly index-only.')
    payload = wire.get('response_payload') or {}
    observed = observation.get('response_payload') or {}
    frame = payload.get('response_frame') or {}
    observed_frame = observed.get('response_frame') or {}
    frame_id = str(indexed_record.get('frame_id') or '')
    frame_sequence = _required_frame_sequence(
        indexed_record.get('frame_sequence'),
        label='The indexed response frame sequence',
    )
    if not frame_id:
        raise ReferenceExportError(
            'The indexed response frame has no immutable frame identity.'
        )
    for candidate in (frame, observed_frame):
        candidate_sequence = _required_frame_sequence(
            candidate.get('frame_sequence'),
            label='A response projection frame sequence',
        )
        if (
            candidate.get('frame_id') != frame_id
            or candidate_sequence != frame_sequence
        ):
            raise ReferenceExportError('Response projections disagree on frame identity.')
    semantics = (frame.get('current_state') or {}).get('status_semantics') or {}
    late_fill = payload.get('late_fill') or {}
    closure = ((observed.get('runtime') or {}).get('graph_closure_review') or {})
    surface = closure.get('surface_state') or {}
    invalid = (
        payload.get('response_id') != response_id
        or payload.get('lifecycle_state') != 'completed'
        or not semantics.get('is_terminal')
        or bool(semantics.get('has_open_continuation'))
        or bool(semantics.get('has_actionable_repair'))
        or late_fill.get('status') != 'completed'
        or late_fill.get('final_materialization_contract_status') != 'fulfilled'
        or int(late_fill.get('failed_branch_count') or 0) != 0
        or int(late_fill.get('pending_branch_count') or 0) != 0
        or closure.get('status') != 'fulfilled'
        or bool(closure.get('continuation_required'))
        or int(closure.get('pending_branch_count') or 0) != 0
        or surface.get('status') != 'fulfilled'
    )
    if invalid:
        raise ReferenceExportError('The response is not cleanly and truthfully complete.')
    artifacts = monitor.get('artifacts') or {}
    monitor_sequence = monitor.get('frame_sequence')
    monitor_sequence_matches = True
    if monitor_sequence not in (None, ''):
        monitor_sequence_matches = (
            _required_frame_sequence(
                monitor_sequence,
                label='The monitor frame sequence',
            )
            == frame_sequence
        )
    monitor_invalid = (
        monitor.get('response_id') != response_id
        or monitor.get('frame_id') != frame_id
        or not monitor_sequence_matches
        or monitor.get('verdict') != 'clean'
        or monitor.get('lifecycle_state') != 'completed'
        or monitor.get('late_fill_status') != 'completed'
        or monitor.get('final_materialization_contract_status') != 'fulfilled'
        or bool(monitor.get('materialization_contract_unmet'))
        or int(monitor.get('failed_branch_count') or 0) != 0
        or bool(artifacts.get('missing_files'))
        or bool(artifacts.get('sha_mismatches'))
        or bool(artifacts.get('html_issues'))
        or bool(artifacts.get('css_issues'))
    )
    if monitor_invalid:
        raise ReferenceExportError('The independent monitor did not report a clean run.')
    return dict(payload), dict(observed), dict(closure)


def _artifact_public_path(
    record: Mapping[str, Any],
    *,
    counters: dict[str, int],
) -> Path:
    kind = str(record.get('type') or record.get('kind') or 'file').lower()
    source = Path(str(record.get('path') or ''))
    suffix = source.suffix.lower() or mimetypes.guess_extension(
        str(record.get('mime_type') or '')
    ) or '.bin'
    counters[kind] = counters.get(kind, 0) + 1
    index = counters[kind]
    if kind == 'image':
        return Path('artifacts/images') / f'image-{index:02d}{suffix}'
    if kind == 'audio':
        name = 'narration' if index == 1 else f'audio-{index:02d}'
        return Path('artifacts/audio') / f'{name}{suffix}'
    if suffix == '.html':
        return Path('artifacts/documents/index.html')
    if suffix == '.css':
        return Path('artifacts/documents/styles.css')
    if 'transcripts' in source.parts:
        return Path('artifacts/transcripts') / f'transcript{suffix}'
    return Path('artifacts/documents') / f'document-{index:02d}{suffix}'


def _png_metadata(path: Path) -> dict[str, Any]:
    with path.open('rb') as handle:
        header = handle.read(24)
    if len(header) >= 24 and header[:8] == b'\x89PNG\r\n\x1a\n':
        width, height = struct.unpack('>II', header[16:24])
        return {'width': width, 'height': height}
    return {}


def _wav_metadata(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), 'rb') as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return {
                'duration_seconds': round(frames / rate, 6) if rate else None,
                'sample_rate_hz': rate,
                'channels': handle.getnchannels(),
                'sample_width_bits': handle.getsampwidth() * 8,
            }
    except (wave.Error, EOFError):
        return {}


def _artifact_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == '.png':
        return _png_metadata(path)
    if path.suffix.lower() == '.wav':
        return _wav_metadata(path)
    return {}


def _copy_artifacts(
    records: Sequence[Mapping[str, Any]],
    *,
    response_id: str,
    artifact_root: Path,
    staging_root: Path,
) -> tuple[list[dict[str, Any]], dict[Path, Path], dict[str, Path]]:
    counters: dict[str, int] = {}
    exported: list[dict[str, Any]] = []
    source_to_public: dict[Path, Path] = {}
    ref_to_public: dict[str, Path] = {}
    for record in records:
        if str(record.get('source_response_id') or '') != response_id:
            raise ReferenceExportError('Artifact source response identity mismatch.')
        artifact_ref = str(record.get('artifact_ref') or record.get('ref') or '')
        if not artifact_ref:
            raise ReferenceExportError('Artifact is missing its canonical reference.')
        source = _assert_regular_source(
            Path(str(record.get('path') or '')),
            root=artifact_root,
            label=f'Artifact {artifact_ref}',
        )
        relative = _artifact_public_path(record, counters=counters)
        if relative in source_to_public.values():
            raise ReferenceExportError('Two artifacts map to the same public path.')
        destination = staging_root / relative
        _copy_file(source, destination)
        mime_type = str(record.get('mime_type') or '').strip()
        if not mime_type:
            mime_type = mimetypes.guess_type(destination.name)[0] or 'application/octet-stream'
        item = {
            'artifact_ref': artifact_ref,
            'type': str(record.get('type') or record.get('kind') or 'file'),
            'mime_type': mime_type,
            'published_path': relative.as_posix(),
            'size_bytes': destination.stat().st_size,
            'sha256': _sha256_file(destination),
            'phase_id': str(record.get('phase_id') or ''),
            'branch_id': str(record.get('branch_id') or ''),
            **_artifact_metadata(destination),
        }
        exported.append({k: v for k, v in item.items() if v not in ('', None)})
        source_to_public[source] = relative
        ref_to_public[artifact_ref] = relative
    return exported, source_to_public, ref_to_public


def _copy_bundle(
    bundle_dir: Path,
    *,
    response_id: str,
    artifact_root: Path,
    staging_root: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[Path, Path]]:
    bundle_root = _assert_regular_source(
        bundle_dir / 'manifest.json',
        root=artifact_root / 'bundles',
        label='Bundle manifest',
    ).parent
    manifest = json.loads((bundle_root / 'manifest.json').read_text(encoding='utf-8'))
    if (
        manifest.get('source_response_id') != response_id
        or manifest.get('status') != 'bundled'
        or (manifest.get('link_check') or {}).get('status') != 'passed'
    ):
        raise ReferenceExportError('The source bundle is not a verified bundle for this response.')
    copied = manifest.get('copied_artifacts') or []
    if not copied:
        raise ReferenceExportError('The source bundle has no copied artifact records.')
    entrypoint = _safe_relative_path(
        str(manifest.get('entrypoint_relative_path') or '')
    )
    public_records: list[dict[str, Any]] = []
    ref_to_bundle: dict[str, Path] = {}
    source_to_bundle: dict[Path, Path] = {}
    expected_source_paths: set[Path] = {bundle_root / 'manifest.json'}
    source_entries: list[dict[str, Any]] = []
    old_to_new: dict[Path, Path] = {}
    kind_counts: dict[str, int] = {}
    for record in copied:
        relative = _safe_relative_path(str(record.get('relative_path') or ''))
        source = _assert_regular_source(
            bundle_root / relative,
            root=bundle_root,
            label=f'Bundle file {relative.as_posix()}',
        )
        expected_source_paths.add(source)
        if (
            source.stat().st_size != int(record.get('size_bytes') or -1)
            or _sha256_file(source) != str(record.get('sha256') or '')
        ):
            raise ReferenceExportError('A source bundle file fails its recorded checksum.')
        kind = str(record.get('type') or record.get('kind') or 'file').lower()
        suffix = source.suffix.lower()
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        index = kind_counts[kind]
        if relative == entrypoint:
            public_relative = Path('index.html')
        elif suffix == '.css':
            public_relative = Path('assets/css/styles.css')
        elif kind == 'image':
            public_relative = Path('assets/images') / f'image-{index:02d}{suffix}'
        elif kind == 'audio':
            name = 'narration' if index == 1 else f'audio-{index:02d}'
            public_relative = Path('assets/audio') / f'{name}{suffix}'
        else:
            public_relative = Path('assets/files') / f'file-{index:02d}{suffix}'
        if public_relative in old_to_new.values():
            raise ReferenceExportError('Two bundle files map to one public path.')
        old_to_new[relative] = public_relative
        published = Path('bundle') / public_relative
        artifact_ref = str(record.get('artifact_ref') or '')
        if artifact_ref:
            ref_to_bundle[artifact_ref] = published
        source_path_text = str(record.get('source_path') or '')
        if source_path_text:
            source_path = Path(source_path_text).resolve()
            if _is_within(source_path, artifact_root.resolve()):
                source_to_bundle[source_path] = published
        source_entries.append(
            {
                'source': source,
                'source_relative': relative,
                'public_relative': public_relative,
                'published': published,
                'artifact_ref': artifact_ref,
                'type': kind,
                'source_sha256': str(record.get('sha256') or ''),
            }
        )
    link_replacements: dict[str, str] = {}
    public_links: list[dict[str, Any]] = []
    for item in manifest.get('rewritten_links') or []:
        old_target = _safe_relative_path(str(item.get('rewritten') or ''))
        new_target = old_to_new.get(old_target)
        if new_target is None:
            raise ReferenceExportError('A verified bundle link has no public target.')
        link_replacements[old_target.as_posix()] = new_target.as_posix()
        public_links.append(
            {
                'kind': str(item.get('kind') or ''),
                'target': new_target.as_posix(),
            }
        )
    for entry in source_entries:
        source = entry['source']
        destination = staging_root / entry['published']
        if entry['source_relative'] == entrypoint:
            content = source.read_text(encoding='utf-8')
            for old_target, new_target in link_replacements.items():
                content = content.replace(old_target, new_target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding='utf-8')
            destination.chmod(0o644)
        else:
            _copy_file(source, destination)
        public_records.append(
            {
                'artifact_ref': entry['artifact_ref'],
                'type': entry['type'],
                'relative_path': entry['public_relative'].as_posix(),
                'size_bytes': destination.stat().st_size,
                'sha256': _sha256_file(destination),
                'source_sha256': entry['source_sha256'],
            }
        )
    actual_source_paths = {
        path.resolve()
        for path in bundle_root.rglob('*')
        if path.is_file()
    }
    if actual_source_paths != expected_source_paths:
        raise ReferenceExportError('The source bundle contains unmanifested files.')
    public_entrypoint = old_to_new.get(entrypoint)
    if public_entrypoint not in {
        _safe_relative_path(item['relative_path']) for item in public_records
    }:
        raise ReferenceExportError('The bundle entrypoint is not a copied artifact.')
    public_manifest = {
        'schema': BUNDLE_SCHEMA,
        'version': 1,
        'source_response_id': response_id,
        'source_bundle_id': str(manifest.get('bundle_id') or ''),
        'status': 'bundled',
        'entrypoint': public_entrypoint.as_posix(),
        'link_check': {'status': 'passed'},
        'files': sorted(public_records, key=lambda item: item['relative_path']),
        'resolved_links': public_links,
        'raw_runtime_manifest_published': False,
    }
    _write_json(staging_root / 'bundle' / 'manifest.json', public_manifest)
    return public_manifest, ref_to_bundle, source_to_bundle


def _public_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    ref_to_public: Mapping[str, Path],
    ref_to_bundle: Mapping[str, Path],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    allowed = (
        'slot_id',
        'branch_id',
        'phase_id',
        'batch_index',
        'type',
        'status',
        'lifecycle',
        'artifact_ref',
    )
    for output in outputs:
        item = _fixed_scalars(output, allowed)
        artifact_ref = str(output.get('artifact_ref') or '')
        if artifact_ref:
            if artifact_ref not in ref_to_public:
                raise ReferenceExportError('A public output points to an unpublished artifact.')
            item['published_path'] = ref_to_public[artifact_ref].as_posix()
            if artifact_ref in ref_to_bundle:
                item['bundle_path'] = ref_to_bundle[artifact_ref].as_posix()
        elif str(output.get('type') or '') == 'text':
            value = output.get('value')
            if not isinstance(value, str):
                raise ReferenceExportError('A public text output is not available exactly.')
            item['text'] = value
        projected.append(item)
    return projected


def _public_monitor_snapshot(
    monitor: Mapping[str, Any],
    *,
    raw_identity: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = monitor.get('artifacts') or {}
    timing = monitor.get('timing') or {}
    seconds = {
        key: value
        for key, value in timing.items()
        if key.endswith('_seconds') and isinstance(value, (int, float))
    }
    return {
        'schema': MONITOR_SCHEMA,
        'version': 1,
        'authority': 'independent_observer_evidence',
        'response_id': monitor.get('response_id'),
        'frame_id': monitor.get('frame_id'),
        'frame_sequence': monitor.get('frame_sequence'),
        'reported_at': monitor.get('reported_at'),
        'verdict': monitor.get('verdict'),
        'runtime_summary': _fixed_scalars(
            monitor,
            (
                'lifecycle_state',
                'late_fill_status',
                'final_materialization_contract_status',
                'materialization_contract_unmet',
                'failed_branch_count',
            ),
        ),
        'branch_summary': {
            'counts': monitor.get('branch_counts') or {},
            'capability_counts': monitor.get('branch_capability_counts') or {},
            'role_counts': monitor.get('branch_role_counts') or {},
            'output_obligation_counts': monitor.get('output_obligation_counts') or {},
            'image_branch_count': int(monitor.get('image_branch_count') or 0),
            'text_artifact_branch_count': int(
                monitor.get('text_artifact_branch_count') or 0
            ),
        },
        'artifact_summary': {
            'artifact_count': int(artifacts.get('artifact_count') or 0),
            'artifact_bytes': int(artifacts.get('artifact_bytes') or 0),
            'kind_counts': artifacts.get('artifact_kind_counts') or {},
            'file_count_by_suffix': artifacts.get('artifact_file_count_by_suffix') or {},
            'missing_file_count': len(artifacts.get('missing_files') or []),
            'sha_mismatch_count': len(artifacts.get('sha_mismatches') or []),
            'html_issue_count': len(artifacts.get('html_issues') or []),
            'css_issue_count': len(artifacts.get('css_issues') or []),
        },
        'timing_seconds': seconds,
        'provenance': {
            **raw_identity,
            'source': 'exact reports.jsonl record',
            'raw_record_published': False,
        },
    }


def _rewrite_readme(
    text: str,
    *,
    source_readme: Path,
    source_to_public: Mapping[Path, Path],
    bundle_dir: Path | None,
) -> str:
    for source, published in source_to_public.items():
        old = Path(os.path.relpath(source, source_readme.parent)).as_posix()
        text = text.replace(f']({old})', f']({published.as_posix()})')
    if bundle_dir is not None:
        for name in ('index.html', 'manifest.json'):
            source = (bundle_dir / name).resolve()
            old = Path(os.path.relpath(source, source_readme.parent)).as_posix()
            text = text.replace(f']({old})', f'](bundle/{name})')
    text = re.sub(
        r'\[Human-readable monitor reports\]\([^)]*\)',
        '[Published monitor snapshot](monitor-report.json)',
        text,
    )
    text = re.sub(
        r'\[Structured monitor reports\]\([^)]*\)',
        '[Published monitor snapshot](monitor-report.json)',
        text,
    )
    replacements = {
        'The generated media are not duplicated under `examples/`.': (
            'Verified publication copies are included in this package; the original '
            'runtime artifacts remain canonical.'
        ),
        'The generated media remains in the canonical artifact directory and is not\n'
        'duplicated under `examples/`.': (
            'A verified publication copy is included in this package; the original '
            'runtime artifact remains canonical.'
        ),
        'The generated media remains in the canonical artifact directories and is not\n'
        'duplicated under `examples/`.': (
            'Verified publication copies are included in this package; the original '
            'runtime artifacts remain canonical.'
        ),
        'The generated images remain in the canonical artifact directory and are not\n'
        'duplicated under `examples/`.': (
            'Verified publication copies are included in this package; the original '
            'runtime artifacts remain canonical.'
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    appendix = (
        '\n## Publication Package\n\n'
        'This directory is a sanitized, immutable publication copy of the reviewed run. '
        'It does not replace the original response frame or runtime artifacts.\n\n'
        '- [Exact reviewed prompt](prompt.txt)\n'
        '- [Sanitized final response truth](response.json)\n'
        '- [Sanitized independent monitor snapshot](monitor-report.json)\n'
        '- [Package checksums and provenance](manifest.json)\n'
    )
    return text.rstrip() + '\n' + appendix


def _manifest_role(relative: Path) -> str:
    if relative == Path('README.md'):
        return 'documentation'
    if relative == Path('prompt.txt'):
        return 'request'
    if relative == Path('response.json'):
        return 'response_truth'
    if relative == Path('monitor-report.json'):
        return 'observer_evidence'
    if relative.parts[0] == 'bundle':
        return 'openable_bundle'
    if relative.parts[0] == 'artifacts':
        return 'canonical_artifact_copy'
    return 'package_data'


def _assert_no_local_paths(data: bytes, *, label: str) -> None:
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            raise ReferenceExportError(f'Local path leaked into {label}.')


def _verify_markdown_links(package_dir: Path, readme_path: Path) -> None:
    text = readme_path.read_text(encoding='utf-8')
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group('target').strip().strip('<>')
        if (
            not target
            or target.startswith('#')
            or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target)
        ):
            continue
        target = target.split('#', 1)[0].split('?', 1)[0]
        if not target:
            continue
        relative = _safe_relative_path(target)
        resolved = (readme_path.parent / relative).resolve()
        if not _is_within(resolved, package_dir) or not resolved.is_file():
            raise ReferenceExportError(
                f'Broken package-local Markdown link: {match.group("target")}'
            )


def _verify_public_bundle(package_dir: Path) -> None:
    bundle_root = package_dir / 'bundle'
    manifest_path = bundle_root / 'manifest.json'
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ReferenceExportError('Public bundle manifest is invalid JSON.') from exc
    if manifest.get('schema') != BUNDLE_SCHEMA or manifest.get('version') != 1:
        raise ReferenceExportError('Unsupported public bundle manifest.')
    expected: set[Path] = set()
    for record in manifest.get('files') or []:
        relative = _safe_relative_path(str(record.get('relative_path') or ''))
        expected.add(relative)
        path = bundle_root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record.get('size_bytes') or -1)
            or _sha256_file(path) != str(record.get('sha256') or '')
        ):
            raise ReferenceExportError(
                f'Public bundle file verification failed: {relative.as_posix()}'
            )
    actual = {
        path.relative_to(bundle_root)
        for path in bundle_root.rglob('*')
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise ReferenceExportError('Public bundle file set differs from its manifest.')
    entrypoint = _safe_relative_path(str(manifest.get('entrypoint') or ''))
    if entrypoint not in expected:
        raise ReferenceExportError('Public bundle entrypoint is missing.')
    for link in manifest.get('resolved_links') or []:
        target = _safe_relative_path(str(link.get('target') or ''))
        if target not in expected:
            raise ReferenceExportError(
                f'Public bundle link target is missing: {target.as_posix()}'
            )


def _build_package_manifest(
    staging_root: Path,
    *,
    response_id: str,
    indexed_record: Mapping[str, Any],
    artifact_count: int,
    bundle_present: bool,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(staging_root.rglob('*'), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == 'manifest.json' and path.parent == staging_root:
            continue
        if path.is_symlink():
            raise ReferenceExportError('Publication packages cannot contain symlinks.')
        relative = path.relative_to(staging_root)
        _safe_relative_path(relative)
        data = path.read_bytes()
        _assert_no_local_paths(data, label=relative.as_posix())
        files.append(
            {
                'path': relative.as_posix(),
                'role': _manifest_role(relative),
                'mime_type': mimetypes.guess_type(path.name)[0]
                or 'application/octet-stream',
                'size_bytes': len(data),
                'sha256': _sha256_bytes(data),
            }
        )
    return {
        'schema': MANIFEST_SCHEMA,
        'version': 1,
        'response_id': response_id,
        'frame_id': indexed_record.get('frame_id'),
        'frame_sequence': indexed_record.get('frame_sequence'),
        'source_frame_record_sha256': indexed_record.get('record_sha256'),
        'source_frame_record_size_bytes': indexed_record.get('record_size_bytes'),
        'files': files,
        'summary': {
            'file_count': len(files),
            'total_bytes': sum(item['size_bytes'] for item in files),
            'artifact_count': artifact_count,
            'bundle_included': bundle_present,
        },
        'validation': {
            'indexed_frame_verified': True,
            'prompt_identity_verified': True,
            'terminal_runtime_truth_verified': True,
            'monitor_verdict': 'clean',
            'artifact_checksums_verified': True,
            'local_absolute_paths_absent': True,
        },
        'provenance': {
            'original_response_frame_authoritative': True,
            'original_runtime_artifacts_authoritative': True,
            'publication_copy_is_runtime_authority': False,
            'raw_frame_published': False,
            'raw_monitor_record_published': False,
        },
    }


def export_reference_run(
    *,
    response_id: str,
    source_readme: Path,
    output_dir: Path,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    monitor_reports: Path = DEFAULT_MONITOR_REPORTS,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    bundle_dir: Path | None = None,
) -> Path:
    """Export one reviewed response without mutating runtime truth."""

    if output_dir.exists():
        raise ReferenceExportError(f'Output directory already exists: {output_dir}')
    source_readme = source_readme.resolve()
    readme_text = source_readme.read_text(encoding='utf-8')
    prompt = _extract_prompt(readme_text)
    index_state = load_response_frame_index(frames_dir=frames_dir)
    if not index_state.get('ok'):
        raise ReferenceExportError('The response-frame index is unavailable.')
    indexed_record = _read_exact_indexed_record(response_id, index_state=index_state)
    wire = load_latest_response_wire_state(
        response_id,
        frames_dir=frames_dir,
        index_state=index_state,
    )
    observation = load_latest_response_observation_state(
        response_id,
        frames_dir=frames_dir,
        index_state=index_state,
    )
    monitor, monitor_identity = _load_monitor_report(
        response_id,
        reports_path=monitor_reports,
        expected_frame_id=str(indexed_record.get('frame_id') or ''),
        expected_frame_sequence=int(indexed_record.get('frame_sequence') or 0),
    )
    payload, observed, closure = _require_terminal_truth(
        response_id,
        wire=wire,
        observation=observation,
        monitor=monitor,
        indexed_record=indexed_record,
    )
    request_identity = _verify_prompt(prompt, observed.get('request') or {})

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f'.{output_dir.name}.', dir=output_dir.parent)
    )
    try:
        artifact_records = [
            item for item in payload.get('artifacts') or [] if isinstance(item, Mapping)
        ]
        artifacts, source_to_public, ref_to_public = _copy_artifacts(
            artifact_records,
            response_id=response_id,
            artifact_root=artifact_root,
            staging_root=staging,
        )
        bundle_manifest: dict[str, Any] | None = None
        ref_to_bundle: dict[str, Path] = {}
        source_to_bundle: dict[Path, Path] = {}
        if bundle_dir is not None:
            bundle_manifest, ref_to_bundle, source_to_bundle = _copy_bundle(
                bundle_dir,
                response_id=response_id,
                artifact_root=artifact_root,
                staging_root=staging,
            )
            for artifact in artifacts:
                bundle_path = ref_to_bundle.get(str(artifact.get('artifact_ref') or ''))
                if bundle_path is not None:
                    artifact['bundle_path'] = bundle_path.as_posix()
        (staging / 'prompt.txt').write_text(prompt + '\n', encoding='utf-8')
        (staging / 'prompt.txt').chmod(0o644)
        readme = _rewrite_readme(
            readme_text,
            source_readme=source_readme,
            source_to_public={**source_to_public, **source_to_bundle},
            bundle_dir=bundle_dir.resolve() if bundle_dir is not None else None,
        )
        (staging / 'README.md').write_text(readme, encoding='utf-8')
        (staging / 'README.md').chmod(0o644)

        frame = payload.get('response_frame') or {}
        semantics = (frame.get('current_state') or {}).get('status_semantics') or {}
        late_fill = payload.get('late_fill') or {}
        surface = closure.get('surface_state') or {}
        response_export = {
            'schema': EXPORT_SCHEMA,
            'version': 1,
            'response': {
                'id': response_id,
                'frame_id': indexed_record['frame_id'],
                'frame_sequence': indexed_record['frame_sequence'],
                'frame_version': frame.get('frame_version'),
                'status': payload.get('status'),
                'lifecycle_state': payload.get('lifecycle_state'),
                'status_semantics': _fixed_scalars(
                    semantics,
                    (
                        'is_terminal',
                        'terminal',
                        'has_open_continuation',
                        'has_actionable_repair',
                    ),
                ),
            },
            'request': request_identity,
            'outputs': _public_outputs(
                payload.get('outputs') or [],
                ref_to_public=ref_to_public,
                ref_to_bundle=ref_to_bundle,
            ),
            'artifacts': artifacts,
            'runtime_truth': {
                'late_fill': {
                    'status': late_fill.get('status'),
                    'final_materialization_contract_status': late_fill.get(
                        'final_materialization_contract_status'
                    ),
                    'counts': {
                        'completed': int(late_fill.get('completed_branch_count') or 0),
                        'failed': int(late_fill.get('failed_branch_count') or 0),
                        'pending': int(late_fill.get('pending_branch_count') or 0),
                        'cancelled': int(late_fill.get('cancelled_branch_count') or 0),
                    },
                },
                'closure': {
                    'status': closure.get('status'),
                    'continuation_required': bool(
                        closure.get('continuation_required')
                    ),
                    'obligation_count': int(closure.get('obligation_count') or 0),
                    'pending_branch_count': int(
                        closure.get('pending_branch_count') or 0
                    ),
                    'counts': closure.get('counts') or {},
                },
                'surface': {
                    'status': surface.get('status'),
                    'active_categories': surface.get('active_categories') or [],
                    'category_counts': surface.get('category_counts') or {},
                },
            },
            'integrity': {
                'verified_response_map': True,
                'index_only_wire_lookup': True,
                'ledger_fallback_used': False,
                'frame_identity_verified': True,
                'observation_snapshot_digests_verified': True,
                'source_frame_record_sha256': indexed_record['record_sha256'],
                'source_frame_record_size_bytes': indexed_record['record_size_bytes'],
            },
            'bundle': bundle_manifest,
            'provenance': {
                'source': 'indexed final response frame and canonical artifact bytes',
                'runtime_effect': 'none',
                'raw_frame_published': False,
                'original_runtime_truth_remains_authoritative': True,
            },
        }
        _write_json(staging / 'response.json', response_export)
        _write_json(
            staging / 'monitor-report.json',
            _public_monitor_snapshot(monitor, raw_identity=monitor_identity),
        )
        package_manifest = _build_package_manifest(
            staging,
            response_id=response_id,
            indexed_record=indexed_record,
            artifact_count=len(artifacts),
            bundle_present=bundle_manifest is not None,
        )
        _write_json(staging / 'manifest.json', package_manifest)
        verify_reference_package(staging)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def verify_reference_package(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / 'manifest.json'
    if not package_dir.is_dir() or package_dir.is_symlink() or not manifest_path.is_file():
        raise ReferenceExportError('Reference package or manifest is missing.')
    manifest_data = manifest_path.read_bytes()
    _assert_no_local_paths(manifest_data, label='manifest.json')
    try:
        manifest = json.loads(manifest_data)
    except json.JSONDecodeError as exc:
        raise ReferenceExportError('Reference package manifest is invalid JSON.') from exc
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('version') != 1:
        raise ReferenceExportError('Unsupported reference package manifest.')
    records = manifest.get('files') or []
    expected: set[Path] = set()
    for record in records:
        relative = _safe_relative_path(str(record.get('path') or ''))
        if relative in expected:
            raise ReferenceExportError('Duplicate path in package manifest.')
        expected.add(relative)
        path = package_dir / relative
        if path.is_symlink() or not path.is_file():
            raise ReferenceExportError(f'Missing package file: {relative.as_posix()}')
        data = path.read_bytes()
        if (
            len(data) != int(record.get('size_bytes') or -1)
            or _sha256_bytes(data) != str(record.get('sha256') or '')
        ):
            raise ReferenceExportError(
                f'Package checksum mismatch: {relative.as_posix()}'
            )
        _assert_no_local_paths(data, label=relative.as_posix())
    actual = {
        path.relative_to(package_dir)
        for path in package_dir.rglob('*')
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        extras = sorted((actual - expected), key=lambda item: item.as_posix())
        missing = sorted((expected - actual), key=lambda item: item.as_posix())
        raise ReferenceExportError(
            'Package file set differs from its manifest: '
            f'extras={[item.as_posix() for item in extras]}, '
            f'missing={[item.as_posix() for item in missing]}'
        )
    required = {
        Path('README.md'),
        Path('prompt.txt'),
        Path('response.json'),
        Path('monitor-report.json'),
    }
    if not required.issubset(expected):
        raise ReferenceExportError('Reference package lacks required public records.')
    _verify_markdown_links(package_dir, package_dir / 'README.md')
    _verify_public_bundle(package_dir)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, help='Verify an existing package.')
    parser.add_argument('--response-id')
    parser.add_argument('--source-readme', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bundle-dir', type=Path)
    parser.add_argument('--frames-dir', type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument(
        '--monitor-reports', type=Path, default=DEFAULT_MONITOR_REPORTS
    )
    parser.add_argument('--artifact-root', type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.verify is not None:
        manifest = verify_reference_package(args.verify)
        print(
            f"Verified {manifest['response_id']}: "
            f"{manifest['summary']['file_count']} files, "
            f"{manifest['summary']['total_bytes']} bytes"
        )
        return 0
    if not args.response_id or args.source_readme is None or args.output_dir is None:
        raise ReferenceExportError(
            '--response-id, --source-readme, and --output-dir are required.'
        )
    destination = export_reference_run(
        response_id=args.response_id,
        source_readme=args.source_readme,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        monitor_reports=args.monitor_reports,
        artifact_root=args.artifact_root,
        bundle_dir=args.bundle_dir,
    )
    print(f'Exported {args.response_id} to {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
