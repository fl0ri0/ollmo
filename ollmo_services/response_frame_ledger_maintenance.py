"""Explicit audit and historical Ghost-preview compaction for Response Frames.

This module is deliberately outside the hot append path.  It streams the
durable ledger, treats only generated manifest refs as sidecar authority, and
reuses :mod:`ollmo_services.response_frames` for every content-addressed write.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ollmo_services import response_frames


_SNAPSHOT_REF_KIND = 'ollmo.response_frame_snapshot_ref'
_GHOST_PREVIEW_DIGEST_REF_KIND = 'ollmo.ghost_preview_content_digest_ref'
_MAX_REPORTED_ITEMS = 200


def _json_size_bytes(value: Any) -> int:
    return len(
        json.dumps(
            response_frames._json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    )


def _stable_file_state(path: Path) -> Optional[dict[str, int]]:
    """Return identity fields unaffected by creating a hard-link backup."""

    try:
        stat_result = path.stat()
    except OSError:
        return None
    return {
        'device': int(stat_result.st_dev),
        'inode': int(stat_result.st_ino),
        'size_bytes': int(stat_result.st_size),
        'mtime_ns': int(stat_result.st_mtime_ns),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _frame_response_id(frame: Mapping[str, Any]) -> str:
    return response_frames._frame_response_id(frame)


def _frame_identity_record(frame: Mapping[str, Any], line_offset: int) -> dict[str, Any]:
    relation = (
        frame.get('frame_relation')
        if isinstance(frame.get('frame_relation'), Mapping)
        else {}
    )
    return {
        'line_offset': line_offset,
        'response_id': _frame_response_id(frame),
        'frame_id': str(frame.get('frame_id') or '').strip(),
        'frame_sequence': frame.get('frame_sequence'),
        'frame_relation': response_frames._json_safe(relation),
    }


def _update_digest_with_json(digest: Any, value: Any) -> None:
    digest.update(
        json.dumps(
            response_frames._json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    )
    digest.update(b'\n')


def _walk_json(value: Any, json_path: str = ''):
    if isinstance(value, Mapping):
        yield value, json_path
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = response_frames._json_child_path(json_path, key)
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = response_frames._json_child_path(json_path, f'[{index}]')
            yield from _walk_json(child, child_path)


def _ref_identity(ref: Mapping[str, Any]) -> tuple[str, str, int | None]:
    raw_size = ref.get('size_bytes')
    size = raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) else None
    return (
        str(ref.get('path') or '').strip(),
        str(ref.get('sha256') or '').strip().lower(),
        size,
    )


def _is_sha256(value: Any) -> bool:
    digest = str(value or '').strip().lower()
    return len(digest) == 64 and all(
        character in '0123456789abcdef' for character in digest
    )


def _snapshot_ref_contract_error(
    ref: Mapping[str, Any],
    *,
    expected_json_path: str,
    response_id: str,
) -> Optional[str]:
    """Return why a manifest-authorized ref is not a generated CAS ref."""

    if str(ref.get('kind') or '').strip() != _SNAPSHOT_REF_KIND:
        return 'invalid_snapshot_ref_kind'
    if str(ref.get('json_path') or '').strip() != expected_json_path:
        return 'snapshot_ref_json_path_mismatch'
    if ref.get('content_addressed') is not True:
        return 'snapshot_ref_not_content_addressed'
    if str(ref.get('dedupe_scope') or '').strip() != 'response_frame_snapshot_store':
        return 'snapshot_ref_dedupe_scope_mismatch'
    if str(ref.get('source_response_id') or '').strip() != response_id:
        return 'snapshot_ref_response_id_mismatch'

    raw_path, digest, size_bytes = _ref_identity(ref)
    if not _is_sha256(digest):
        return 'invalid_sha256'
    if size_bytes is None or size_bytes < 0:
        return 'invalid_size'
    expected_path = (
        Path(response_frames.DEFAULT_RESPONSE_FRAME_SNAPSHOT_DIR)
        / response_frames.DEFAULT_RESPONSE_FRAME_SNAPSHOT_CONTENT_DIR
        / digest[:2]
        / f'{digest}.json'
    ).as_posix()
    if raw_path != expected_path:
        return 'noncanonical_content_sha256_path'
    return None


def _ghost_preview_digest_ref_error(
    ref: Mapping[str, Any],
    *,
    container_json_path: str,
) -> Optional[str]:
    """Validate the non-file digest identities emitted by preview compaction."""

    prefix = 'request.ghost_preview.compaction.omitted_content_refs['
    if not container_json_path.startswith(prefix) or not container_json_path.endswith(']'):
        return 'unexpected_digest_ref_location'
    index_text = container_json_path[len(prefix) : -1]
    if not index_text.isdigit():
        return 'unexpected_digest_ref_location'
    if str(ref.get('kind') or '').strip() != _GHOST_PREVIEW_DIGEST_REF_KIND:
        return 'invalid_digest_ref_kind'
    if str(ref.get('storage') or '').strip() != 'digest_only':
        return 'invalid_digest_ref_storage'
    if str(ref.get('authority') or '').strip() != 'audit_identity_only':
        return 'invalid_digest_ref_authority'
    if ref.get('content_addressed') is not True:
        return 'invalid_digest_ref_content_addressed'
    if ref.get('path') not in (None, ''):
        return 'digest_ref_must_not_have_path'
    if not _is_sha256(ref.get('sha256')):
        return 'invalid_digest_ref_sha256'
    raw_size = ref.get('size_bytes')
    if (
        not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
        or raw_size < 0
    ):
        return 'invalid_digest_ref_size'
    identity_json_path = str(ref.get('json_path') or '').strip()
    if not identity_json_path.startswith('ghost_preview.'):
        return 'invalid_digest_ref_json_path'
    return None


def _read_verified_snapshot_payload(
    ref: Mapping[str, Any],
    *,
    frames_dir: Path,
) -> tuple[bool, str, Any]:
    raw_path, expected_sha256, expected_size = _ref_identity(ref)
    if not raw_path:
        return False, 'missing_path', None
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        return False, 'unsafe_path', None
    if not _is_sha256(expected_sha256):
        return False, 'invalid_sha256', None
    if expected_size is None or expected_size < 0:
        return False, 'invalid_size', None
    target = frames_dir / relative_path
    if not target.exists():
        return False, 'missing_file', None
    try:
        raw = target.read_bytes()
    except OSError:
        return False, 'unreadable_file', None
    encoded = raw[:-1] if raw.endswith(b'\n') else raw
    if (
        len(encoded) != expected_size
        or hashlib.sha256(encoded).hexdigest() != expected_sha256
    ):
        return False, 'size_or_sha256_mismatch', None
    try:
        payload = json.loads(encoded.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, 'invalid_json', None
    return True, 'verified', payload


def _snapshot_ref_candidates(payload: Any):
    """Return path-relative snapshot-ref values embedded in one CAS body."""

    stack: list[tuple[Any, str]] = [(payload, '')]
    while stack:
        value, current_json_path = stack.pop()
        if isinstance(value, list):
            stack.extend(
                (
                    child,
                    response_frames._json_child_path(
                        current_json_path,
                        f'[{index}]',
                    ),
                )
                for index, child in enumerate(value)
            )
            continue
        if not isinstance(value, Mapping):
            continue
        for raw_key, child in value.items():
            key = str(raw_key or '').strip()
            derived_key = (
                key[: -len('_snapshot_ref')]
                if key.endswith('_snapshot_ref')
                else ''
            )
            if (
                derived_key
                and isinstance(child, Mapping)
                and str(child.get('kind') or '').strip() == _SNAPSHOT_REF_KIND
            ):
                yield (
                    dict(child),
                    response_frames._json_child_path(
                        current_json_path,
                        derived_key,
                    ),
                    derived_key,
                )
                # An unbound ref is opaque payload.  Its metadata is not a
                # second authority source and must not be traversed as one.
                continue
            stack.append(
                (
                    child,
                    response_frames._json_child_path(current_json_path, key),
                )
            )


def _iter_manifest_authorized_children(
    snapshot_ref_candidates: Any,
    *,
    parent_ref: Mapping[str, Any],
    parent_json_path: str,
):
    """Yield actual child refs authenticated by one parent CAS manifest.

    Legacy manifests may omit ``path`` from their summarized child records.
    The complete generated ref, including its CAS path, lives inside the
    authenticated parent payload.  Resolve that ref through the same strict
    binding used by response replay instead of treating manifest summaries as
    standalone files.
    """

    sidecar_manifest = (
        parent_ref.get('sidecar_manifest')
        if isinstance(parent_ref.get('sidecar_manifest'), Mapping)
        else {}
    )
    child_authorities = {
        str(item.get('json_path') or '').strip(): item
        for item in (sidecar_manifest.get('child_refs') or [])
        if isinstance(item, Mapping)
        and str(item.get('json_path') or '').strip()
    }
    if not child_authorities:
        return

    source_response_id = str(parent_ref.get('source_response_id') or '').strip()
    for child, relative_child_path, derived_key in snapshot_ref_candidates:
        expected_child_path = response_frames._json_child_path(
            parent_json_path,
            relative_child_path,
        )
        child_authority = child_authorities.get(expected_child_path)
        authorized_child = response_frames._authorized_sidecar_child_ref(
            child,
            child_authority,
            expected_json_path=expected_child_path,
            expected_child_key=derived_key,
            source_response_id=source_response_id,
        )
        if authorized_child is not None:
            yield authorized_child, expected_child_path


def _iter_authoritative_snapshot_graph(
    root_ref: Mapping[str, Any],
    *,
    root_json_path: str,
    response_id: str,
    frames_dir: Path,
    verify_snapshot_hashes: bool,
    payload_cache: dict[
        tuple[str, str, int | None],
        tuple[bool, str, Any],
    ],
):
    """Walk a root ref and only the actual child refs bound by its CAS bytes."""

    stack: list[tuple[Mapping[str, Any], str, int]] = [
        (root_ref, root_json_path, 1)
    ]
    while stack:
        ref, json_path, depth = stack.pop()
        if depth > 128:
            yield ref, json_path, (False, 'snapshot_graph_limit_exceeded')
            continue
        contract_error = _snapshot_ref_contract_error(
            ref,
            expected_json_path=json_path,
            response_id=response_id,
        )
        if contract_error is not None:
            yield ref, json_path, (False, contract_error)
            continue
        if not verify_snapshot_hashes:
            yield ref, json_path, (None, 'not_verified')
            continue
        identity = _ref_identity(ref)
        cached = payload_cache.get(identity)
        if cached is None:
            ok, reason, payload = _read_verified_snapshot_payload(
                ref,
                frames_dir=frames_dir,
            )
            candidates = (
                tuple(_snapshot_ref_candidates(payload))
                if ok
                else ()
            )
            cached = (ok, reason, candidates)
            payload_cache[identity] = cached
        ok, reason, candidates = cached
        yield ref, json_path, (ok, reason)
        if not ok:
            continue
        children = list(
            _iter_manifest_authorized_children(
                candidates,
                parent_ref=ref,
                parent_json_path=json_path,
            )
        )
        stack.extend(
            (child_ref, child_json_path, depth + 1)
            for child_ref, child_json_path in reversed(children)
        )


def _parent_manifest_for_frame(
    frame: Mapping[str, Any],
    *,
    response_id: str,
    manifests_by_frame: Mapping[tuple[str, str], Mapping[str, Any]],
    latest_manifest_by_response: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relation = (
        frame.get('frame_relation')
        if isinstance(frame.get('frame_relation'), Mapping)
        else {}
    )
    parent_frame_id = str(relation.get('parent_frame_id') or '').strip()
    if parent_frame_id:
        parent = manifests_by_frame.get((response_id, parent_frame_id))
        return dict(parent) if isinstance(parent, Mapping) else {}
    if frame.get('frame_sequence') not in (None, 1):
        latest = latest_manifest_by_response.get(response_id)
        return dict(latest) if isinstance(latest, Mapping) else {}
    return {}


def audit_response_frame_ledger(
    *,
    frames_dir: Path | str = response_frames.DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = response_frames.DEFAULT_RESPONSE_FRAME_LEDGER,
    verify_snapshot_hashes: bool = True,
) -> dict[str, Any]:
    """Stream one ledger epoch and report preview/CAS integrity without writes."""

    frames_root = Path(frames_dir)
    ledger_path = frames_root / ledger_name
    source_state = _stable_file_state(ledger_path)
    if source_state is None:
        return {
            'ok': False,
            'changed': False,
            'error': {
                'code': 'response_frame_ledger_missing',
                'message': 'Response-frame ledger does not exist.',
            },
            'ledger_path': str(ledger_path),
        }

    line_count = 0
    response_ids: set[str] = set()
    identity_digest = hashlib.sha256()
    ledger_digest = hashlib.sha256()
    malformed: list[dict[str, Any]] = []
    eligible_frame_count = 0
    inline_preview_frame_count = 0
    inline_preview_bytes = 0
    estimated_reclaimable_inline_bytes = 0
    preview_over_10mb_count = 0
    max_inline_preview_bytes = 0
    digest_only_occurrence_count = 0
    digest_only_digests: set[str] = set()
    digest_only_by_json_path: dict[str, int] = {}
    malformed_digest_ref_occurrence_count = 0
    malformed_digest_refs: list[dict[str, Any]] = []
    opaque_snapshot_ref_occurrence_count = 0
    opaque_snapshot_refs: list[dict[str, Any]] = []
    authoritative_occurrence_count = 0
    authoritative_unique_identities: set[tuple[str, str, int | None]] = set()
    verified_unique_identities: set[tuple[str, str, int | None]] = set()
    missing_unique: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    snapshot_payload_cache: dict[
        tuple[str, str, int | None],
        tuple[bool, str, Any],
    ] = {}
    manifests_by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    latest_manifest_by_response: dict[str, dict[str, Any]] = {}

    try:
        with ledger_path.open('rb') as handle:
            for line_offset, raw_line in enumerate(handle):
                line_count += 1
                ledger_digest.update(raw_line)
                try:
                    frame = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    malformed.append(
                        {
                            'line_offset': line_offset,
                            'code': 'invalid_json',
                            'message': str(exc),
                        }
                    )
                    continue
                if not isinstance(frame, dict):
                    malformed.append(
                        {
                            'line_offset': line_offset,
                            'code': 'invalid_frame',
                            'message': 'Ledger line is not a JSON object.',
                        }
                    )
                    continue
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
                    malformed.append(
                        {
                            'line_offset': line_offset,
                            'code': 'invalid_frame_identity',
                            'message': 'Ledger line lacks canonical response-frame identity.',
                        }
                    )
                    continue
                response_ids.add(response_id)
                _update_digest_with_json(
                    identity_digest,
                    _frame_identity_record(frame, line_offset),
                )

                parent_manifest = _parent_manifest_for_frame(
                    frame,
                    response_id=response_id,
                    manifests_by_frame=manifests_by_frame,
                    latest_manifest_by_response=latest_manifest_by_response,
                )
                effective_manifest = response_frames._effective_snapshot_manifest(
                    frame,
                    parent_manifest=parent_manifest,
                )
                manifests_by_frame[(response_id, frame_id)] = effective_manifest
                latest_manifest_by_response[response_id] = effective_manifest

                authorized_identities: set[tuple[str, str, int | None]] = set()
                for json_path, root_ref in effective_manifest.items():
                    if not isinstance(root_ref, Mapping):
                        continue
                    for (
                        authoritative_ref,
                        authoritative_json_path,
                        integrity_status,
                    ) in _iter_authoritative_snapshot_graph(
                        root_ref,
                        root_json_path=str(json_path),
                        response_id=response_id,
                        frames_dir=frames_root,
                        verify_snapshot_hashes=verify_snapshot_hashes,
                        payload_cache=snapshot_payload_cache,
                    ):
                        authoritative_occurrence_count += 1
                        identity = _ref_identity(authoritative_ref)
                        authorized_identities.add(identity)
                        authoritative_unique_identities.add(identity)
                        ok, reason = integrity_status
                        if ok is None:
                            continue
                        if ok:
                            verified_unique_identities.add(identity)
                            continue
                        if identity not in missing_unique:
                            missing_unique[identity] = {
                                'response_id': response_id,
                                'frame_id': frame_id,
                                'frame_sequence': frame_sequence,
                                'line_offset': line_offset,
                                'json_path': authoritative_json_path,
                                'path': identity[0] or None,
                                'sha256': identity[1] or None,
                                'size_bytes': identity[2],
                                'reason': reason,
                            }

                request = frame.get('request') if isinstance(frame.get('request'), Mapping) else {}
                ghost_preview = request.get('ghost_preview')
                if ghost_preview not in (None, '', [], {}):
                    preview_bytes = _json_size_bytes(ghost_preview)
                    inline_preview_frame_count += 1
                    inline_preview_bytes += preview_bytes
                    max_inline_preview_bytes = max(max_inline_preview_bytes, preview_bytes)
                    if preview_bytes > 10 * 1024 * 1024:
                        preview_over_10mb_count += 1
                    existing_preview_ref = request.get('ghost_preview_snapshot_ref')
                    existing_preview_authority = effective_manifest.get(
                        'request.ghost_preview'
                    )
                    exact_preview_authorized = bool(
                        isinstance(existing_preview_ref, Mapping)
                        and isinstance(existing_preview_authority, Mapping)
                        and response_frames._snapshot_ref_matches_authority(
                            existing_preview_ref,
                            existing_preview_authority,
                            response_id=response_id,
                            expected_json_path='request.ghost_preview',
                        )
                    )
                    if (
                        preview_bytes
                        >= response_frames._LEDGER_LARGE_CONTRACT_LIMIT_BYTES
                        and not exact_preview_authorized
                    ):
                        eligible_frame_count += 1
                        compact_preview = response_frames._compact_request_ghost_preview(
                            ghost_preview
                        )
                        estimated_reclaimable_inline_bytes += max(
                            0,
                            preview_bytes - _json_size_bytes(compact_preview or {}),
                        )

                    for item, json_path in _walk_json(
                        ghost_preview,
                        'request.ghost_preview',
                    ):
                        kind = str(item.get('kind') or '').strip()
                        if kind == _GHOST_PREVIEW_DIGEST_REF_KIND:
                            digest_error = _ghost_preview_digest_ref_error(
                                item,
                                container_json_path=json_path,
                            )
                            if digest_error is not None:
                                malformed_digest_ref_occurrence_count += 1
                                if len(malformed_digest_refs) < _MAX_REPORTED_ITEMS:
                                    malformed_digest_refs.append(
                                        {
                                            'response_id': response_id,
                                            'frame_id': frame_id,
                                            'frame_sequence': frame_sequence,
                                            'line_offset': line_offset,
                                            'json_path': json_path,
                                            'reason': digest_error,
                                            'ref': response_frames._json_safe(item),
                                        }
                                    )
                                continue
                            digest_only_occurrence_count += 1
                            digest = str(item.get('sha256') or '').strip().lower()
                            if digest:
                                digest_only_digests.add(digest)
                            digest_only_by_json_path[json_path] = (
                                digest_only_by_json_path.get(json_path, 0) + 1
                            )
                        elif kind == _SNAPSHOT_REF_KIND:
                            identity = _ref_identity(item)
                            if identity not in authorized_identities:
                                opaque_snapshot_ref_occurrence_count += 1
                                if len(opaque_snapshot_refs) < _MAX_REPORTED_ITEMS:
                                    opaque_snapshot_refs.append(
                                        {
                                            'response_id': response_id,
                                            'frame_id': frame_id,
                                            'frame_sequence': frame_sequence,
                                            'line_offset': line_offset,
                                            'json_path': json_path,
                                            'path': identity[0] or None,
                                            'sha256': identity[1] or None,
                                            'size_bytes': identity[2],
                                            'authority': 'opaque_preview_data',
                                        }
                                    )
                del frame
    except OSError as exc:
        return {
            'ok': False,
            'changed': False,
            'ledger_path': str(ledger_path),
            'error': {
                'code': 'response_frame_ledger_read_failed',
                'message': str(exc),
            },
        }

    end_state = _stable_file_state(ledger_path)
    if end_state != source_state:
        return {
            'ok': False,
            'changed': False,
            'ledger_path': str(ledger_path),
            'error': {
                'code': 'response_frame_ledger_moved',
                'message': 'Response-frame ledger changed during audit.',
            },
            'source_state': source_state,
            'end_state': end_state,
        }

    missing_items = list(missing_unique.values())
    integrity_ok = (
        not malformed
        and not missing_items
        and not malformed_digest_ref_occurrence_count
    )
    return {
        'ok': integrity_ok,
        'changed': False,
        'mode': 'audit',
        'ledger_path': str(ledger_path),
        'ledger_size_bytes': source_state['size_bytes'],
        'ledger_sha256': ledger_digest.hexdigest(),
        'ledger_line_count': line_count,
        'response_count': len(response_ids),
        'frame_identity_digest': identity_digest.hexdigest(),
        'malformed_frame_count': len(malformed),
        'malformed_frames': malformed[:_MAX_REPORTED_ITEMS],
        'inline_ghost_preview_frame_count': inline_preview_frame_count,
        'eligible_ghost_preview_frame_count': eligible_frame_count,
        'inline_ghost_preview_bytes': inline_preview_bytes,
        'estimated_reclaimable_inline_bytes': estimated_reclaimable_inline_bytes,
        'ghost_preview_over_10mb_count': preview_over_10mb_count,
        'max_inline_ghost_preview_bytes': max_inline_preview_bytes,
        'authoritative_snapshot_ref_occurrence_count': authoritative_occurrence_count,
        'authoritative_snapshot_unique_count': len(authoritative_unique_identities),
        'authoritative_snapshot_verified_unique_count': len(
            verified_unique_identities
        ),
        'authoritative_missing_sidecar_count': len(missing_items),
        'authoritative_missing_sidecars': missing_items[:_MAX_REPORTED_ITEMS],
        'digest_only_audit_identity_occurrence_count': digest_only_occurrence_count,
        'digest_only_audit_identity_unique_count': len(digest_only_digests),
        'digest_only_audit_identity_paths': digest_only_by_json_path,
        'digest_only_audit_identities_are_sidecars': False,
        'malformed_digest_only_audit_identity_count': (
            malformed_digest_ref_occurrence_count
        ),
        'malformed_digest_only_audit_identities': malformed_digest_refs,
        'opaque_preview_snapshot_ref_occurrence_count': (
            opaque_snapshot_ref_occurrence_count
        ),
        'opaque_preview_snapshot_refs': opaque_snapshot_refs,
        'authoritative_integrity_ok': integrity_ok,
        # Backward-compatible report field only.  This maintenance command
        # never authorizes or performs garbage collection.
        'gc_safe_from_authoritative_integrity': integrity_ok,
        'source_state': source_state,
    }


def _merge_preview_snapshot_authority(
    frame: dict[str, Any],
    *,
    ref: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    external = (
        dict(frame.get('external_snapshots'))
        if isinstance(frame.get('external_snapshots'), Mapping)
        else {
            'kind': 'ollmo.response_frame_external_snapshots',
            'storage': 'sidecar_json',
            'version': 1,
        }
    )
    delta_items = (
        dict(external.get('items'))
        if isinstance(external.get('items'), Mapping)
        else {}
    )
    parent_ref = parent_manifest.get('request.ghost_preview')
    if not (
        isinstance(parent_ref, Mapping)
        and response_frames._snapshot_refs_same(ref, parent_ref)
    ):
        delta_items['request.ghost_preview'] = response_frames._json_safe(ref)
    else:
        delta_items.pop('request.ghost_preview', None)
    external['items'] = response_frames._json_safe(delta_items)

    effective_manifest = {
        **{
            str(path): response_frames._json_safe(value)
            for path, value in parent_manifest.items()
            if isinstance(value, Mapping) and value.get('sha256')
        },
        **{
            str(path): response_frames._json_safe(value)
            for path, value in delta_items.items()
            if isinstance(value, Mapping) and value.get('sha256')
        },
    }
    if isinstance(parent_ref, Mapping) and response_frames._snapshot_refs_same(
        ref,
        parent_ref,
    ):
        effective_manifest['request.ghost_preview'] = response_frames._json_safe(
            parent_ref
        )
    else:
        effective_manifest['request.ghost_preview'] = response_frames._json_safe(ref)

    inherited_paths = sorted(set(parent_manifest) - set(delta_items))
    relation = (
        frame.get('frame_relation')
        if isinstance(frame.get('frame_relation'), Mapping)
        else {}
    )
    parent_frame_id = str(relation.get('parent_frame_id') or '').strip()
    if parent_frame_id and parent_manifest:
        inheritance = (
            dict(external.get('inheritance'))
            if isinstance(external.get('inheritance'), Mapping)
            else {
                'kind': 'ollmo.response_frame_snapshot_inheritance',
                'strategy': 'delta_manifest',
            }
        )
        inheritance.update(
            {
                'parent_frame_id': parent_frame_id,
                'parent_frame_sequence': relation.get('parent_frame_sequence'),
                'inherited_snapshot_count': len(inherited_paths),
                'inherited_json_paths': inherited_paths,
                'parent_snapshot_count': len(parent_manifest),
                'effective_snapshot_count': len(effective_manifest),
            }
        )
        external['inheritance'] = response_frames._json_safe(inheritance)
    frame['external_snapshots'] = response_frames._json_safe(external)

    unique_snapshot_paths = {
        str(item.get('path') or '')
        for item in effective_manifest.values()
        if isinstance(item, Mapping) and item.get('path') not in (None, '')
    }
    policy = (
        dict(frame.get('snapshot_policy'))
        if isinstance(frame.get('snapshot_policy'), Mapping)
        else {}
    )
    policy.update(
        {
            'kind': response_frames._LEDGER_SNAPSHOT_POLICY_KIND,
            'dedupe_strategy': 'content_sha256',
            'ledger_payload': 'compact',
            'snapshot_ref_count': len(delta_items),
            'effective_snapshot_ref_count': len(effective_manifest),
            'inherited_snapshot_ref_count': len(inherited_paths),
            'truth_preservation': (
                'large_runtime_and_public_bodies_externalized_by_ref'
            ),
            'snapshot_count': len(delta_items),
            'effective_snapshot_count': len(effective_manifest),
            'unique_snapshot_count': len(unique_snapshot_paths),
        }
    )
    frame['snapshot_policy'] = response_frames._json_safe(policy)
    return effective_manifest


def _compact_one_frame(
    frame: dict[str, Any],
    *,
    frames_dir: Path,
    parent_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool, Optional[dict[str, Any]]]:
    request = frame.get('request') if isinstance(frame.get('request'), Mapping) else {}
    request = dict(request)
    ghost_preview = request.get('ghost_preview')
    changed = False
    written_ref: Optional[dict[str, Any]] = None
    if (
        ghost_preview not in (None, '', [], {})
        and _json_size_bytes(ghost_preview)
        >= response_frames._LEDGER_LARGE_CONTRACT_LIMIT_BYTES
    ):
        response_id = _frame_response_id(frame)
        existing_ref = request.get('ghost_preview_snapshot_ref')
        existing_authority = response_frames._effective_snapshot_manifest(
            frame,
            parent_manifest=parent_manifest,
        ).get('request.ghost_preview')
        already_authorized = bool(
            isinstance(existing_ref, Mapping)
            and isinstance(existing_authority, Mapping)
            and response_frames._snapshot_ref_matches_authority(
                existing_ref,
                existing_authority,
                response_id=response_id,
                expected_json_path='request.ghost_preview',
            )
        )
        if not already_authorized:
            written_ref = response_frames._write_snapshot_ref(
                ghost_preview,
                frame=frame,
                frames_dir=frames_dir,
                json_path='request.ghost_preview',
            )
            request['ghost_preview_snapshot_ref'] = response_frames._json_safe(
                written_ref
            )
            compact_preview = response_frames._compact_request_ghost_preview(
                ghost_preview
            )
            if compact_preview:
                request['ghost_preview'] = compact_preview
            else:
                request.pop('ghost_preview', None)
            frame['request'] = response_frames._json_safe(request)
            effective_manifest = _merge_preview_snapshot_authority(
                frame,
                ref=written_ref,
                parent_manifest=parent_manifest,
            )
            changed = True
            return frame, effective_manifest, changed, written_ref
    effective_manifest = response_frames._effective_snapshot_manifest(
        frame,
        parent_manifest=parent_manifest,
    )
    return frame, effective_manifest, changed, written_ref


def _write_compacted_ledger(
    *,
    frames_dir: Path,
    ledger_path: Path,
    temp_ledger_path: Path,
) -> dict[str, Any]:
    manifests_by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    latest_manifest_by_response: dict[str, dict[str, Any]] = {}
    latest_entries: dict[str, dict[str, Any]] = {}
    source_identity_digest = hashlib.sha256()
    target_identity_digest = hashlib.sha256()
    source_sha256 = hashlib.sha256()
    target_sha256 = hashlib.sha256()
    line_count = 0
    byte_offset = 0
    changed_frame_count = 0
    new_ref_occurrence_count = 0
    new_ref_unique_paths: set[str] = set()

    source_mode = int(ledger_path.stat().st_mode) & 0o777
    with ledger_path.open('rb') as source, temp_ledger_path.open('wb') as target:
        os.fchmod(target.fileno(), source_mode)
        for line_offset, raw_line in enumerate(source):
            line_count += 1
            source_sha256.update(raw_line)
            try:
                frame = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f'Ledger line {line_offset} is malformed: {exc}'
                ) from exc
            if not isinstance(frame, dict):
                raise ValueError(f'Ledger line {line_offset} is not a frame object.')
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
                raise ValueError(
                    f'Ledger line {line_offset} lacks canonical frame identity.'
                )
            source_identity = _frame_identity_record(frame, line_offset)
            _update_digest_with_json(source_identity_digest, source_identity)
            parent_manifest = _parent_manifest_for_frame(
                frame,
                response_id=response_id,
                manifests_by_frame=manifests_by_frame,
                latest_manifest_by_response=latest_manifest_by_response,
            )
            compacted, effective_manifest, changed, new_ref = _compact_one_frame(
                frame,
                frames_dir=frames_dir,
                parent_manifest=parent_manifest,
            )
            if changed:
                changed_frame_count += 1
            if isinstance(new_ref, Mapping):
                new_ref_occurrence_count += 1
                raw_path = str(new_ref.get('path') or '').strip()
                if raw_path:
                    new_ref_unique_paths.add(raw_path)
            manifests_by_frame[(response_id, frame_id)] = effective_manifest
            latest_manifest_by_response[response_id] = effective_manifest

            target_identity = _frame_identity_record(compacted, line_offset)
            _update_digest_with_json(target_identity_digest, target_identity)
            encoded_line = (
                json.dumps(
                    response_frames._json_safe(compacted),
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode('utf-8')
                + b'\n'
            )
            target.write(encoded_line)
            target_sha256.update(encoded_line)
            current_state = (
                compacted.get('current_state')
                if isinstance(compacted.get('current_state'), Mapping)
                else {}
            )
            latest_entries[response_id] = {
                'response_id': response_id,
                'latest_frame_id': frame_id,
                'latest_frame_sequence': frame_sequence,
                'frame_relation': response_frames._json_safe(
                    compacted.get('frame_relation')
                )
                if isinstance(compacted.get('frame_relation'), Mapping)
                else None,
                'ledger_path': str(ledger_path),
                'ledger_name': ledger_path.name,
                'line_offset': line_offset,
                'byte_offset': byte_offset,
                'line_length': len(encoded_line),
                'current_lifecycle_state': current_state.get('lifecycle_state'),
                'updated_at': current_state.get('updated_at'),
                'effective_snapshot_manifest': response_frames._json_safe(
                    effective_manifest
                ),
            }
            byte_offset += len(encoded_line)
            del frame
            del compacted
        target.flush()
        os.fsync(target.fileno())

    if source_identity_digest.hexdigest() != target_identity_digest.hexdigest():
        raise ValueError('Compacted ledger changed frame identity or order.')
    for entry in latest_entries.values():
        entry['ledger_size_bytes'] = byte_offset
    index_payload = {
        'kind': 'ollmo.response_frame_current_index',
        'version': 2,
        'ledger_path': str(ledger_path),
        'ledger_name': ledger_path.name,
        'ledger_line_count': line_count,
        'ledger_size_bytes': byte_offset,
        'ledger_line_count_verified_size_bytes': byte_offset,
        'responses': latest_entries,
    }
    index_payload['response_map_verified_size_bytes'] = byte_offset
    index_payload['response_map_entry_count'] = len(latest_entries)
    index_payload['response_map_digest'] = response_frames._response_map_digest(
        latest_entries
    )
    return {
        'line_count': line_count,
        'response_count': len(latest_entries),
        'source_size_bytes': ledger_path.stat().st_size,
        'target_size_bytes': byte_offset,
        'reclaimed_ledger_bytes': max(0, ledger_path.stat().st_size - byte_offset),
        'source_sha256': source_sha256.hexdigest(),
        'target_sha256': target_sha256.hexdigest(),
        'frame_identity_digest': source_identity_digest.hexdigest(),
        'changed_frame_count': changed_frame_count,
        'new_preview_ref_occurrence_count': new_ref_occurrence_count,
        'new_preview_ref_unique_path_count': len(new_ref_unique_paths),
        'new_preview_ref_paths': sorted(new_ref_unique_paths),
        'index_payload': index_payload,
    }


def _atomic_install(path: Path, temp_path: Path) -> None:
    os.replace(temp_path, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _backup_file(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return 'hard_link'
    except OSError:
        shutil.copy2(source, target)
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
        return 'byte_copy'


def _content_sha256_cas_files(frames_dir: Path) -> set[str]:
    cas_root = (
        frames_dir
        / response_frames.DEFAULT_RESPONSE_FRAME_SNAPSHOT_DIR
        / response_frames.DEFAULT_RESPONSE_FRAME_SNAPSHOT_CONTENT_DIR
    )
    if not cas_root.exists():
        return set()
    return {
        path.relative_to(frames_dir).as_posix()
        for path in cas_root.rglob('*.json')
        if path.is_file()
    }


def compact_response_frame_ledger(
    *,
    frames_dir: Path | str = response_frames.DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = response_frames.DEFAULT_RESPONSE_FRAME_LEDGER,
    index_name: str = response_frames.DEFAULT_RESPONSE_FRAME_INDEX,
    execute: bool = False,
    backup_dir: Path | str | None = None,
    writers_stopped: bool = False,
) -> dict[str, Any]:
    """Audit or explicitly rewrite historical oversized Ghost previews.

    Execute mode is intentionally offline-only.  ``writers_stopped=True`` is
    an operator assertion that the control plane and every external frame
    writer are quiescent.  Start/end file-state checks remain mandatory.
    """

    frames_root = Path(frames_dir)
    ledger_path = frames_root / ledger_name
    index_path = frames_root / index_name
    audit = audit_response_frame_ledger(
        frames_dir=frames_root,
        ledger_name=ledger_name,
    )
    if not execute:
        return audit
    if not writers_stopped:
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'error': {
                'code': 'response_frame_writers_not_confirmed_stopped',
                'message': (
                    'Execute mode requires explicit confirmation that the '
                    'control plane and all response-frame writers are stopped.'
                ),
            },
        }
    if audit.get('ok') is not True:
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'error': {
                'code': 'response_frame_ledger_preflight_failed',
                'message': (
                    'Ledger compaction is blocked by malformed frames or '
                    'authoritative sidecar integrity errors.'
                ),
            },
        }
    if int(audit.get('eligible_ghost_preview_frame_count') or 0) == 0:
        return {
            'ok': True,
            'changed': False,
            'mode': 'execute',
            'status': 'already_compact',
            'preflight': audit,
        }
    if not index_path.exists():
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'error': {
                'code': 'response_frame_index_missing',
                'message': 'Current response-frame index does not exist.',
            },
        }
    index_preflight = response_frames.attest_response_frame_index(
        frames_dir=frames_root,
        ledger_name=ledger_name,
        index_name=index_name,
        write=False,
    )
    if index_preflight.get('ok') is not True:
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'index_preflight': index_preflight,
            'error': {
                'code': 'response_frame_index_preflight_failed',
                'message': (
                    'Current response-frame index does not exactly attest '
                    'against the source ledger.'
                ),
            },
        }

    source_ledger_state = _stable_file_state(ledger_path)
    source_index_state = _stable_file_state(index_path)
    if source_ledger_state is None or source_index_state is None:
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'error': {
                'code': 'response_frame_source_state_unavailable',
                'message': 'Ledger or index state could not be captured.',
            },
        }
    source_index_sha256 = _file_sha256(index_path)
    if (
        source_ledger_state != audit.get('source_state')
        or _file_sha256(ledger_path) != audit.get('ledger_sha256')
    ):
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'error': {
                'code': 'response_frame_ledger_moved_after_preflight',
                'message': 'Response-frame ledger changed after its audit.',
            },
        }

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_root = (
        Path(backup_dir)
        if backup_dir is not None
        else frames_root.parent / 'response_frame_ledger_backups' / run_id
    )
    if backup_root.exists():
        return {
            'ok': False,
            'changed': False,
            'mode': 'execute',
            'preflight': audit,
            'backup_directory': str(backup_root),
            'error': {
                'code': 'response_frame_backup_directory_exists',
                'message': (
                    'Response-frame backup directory already exists: '
                    f'{backup_root}'
                ),
            },
        }

    cas_files_before = _content_sha256_cas_files(frames_root)
    temp_ledger_path: Optional[Path] = None
    temp_index_path: Optional[Path] = None
    backup_created = False
    ledger_backup: Optional[Path] = None
    index_backup: Optional[Path] = None
    ledger_backup_mode: Optional[str] = None
    index_backup_mode: Optional[str] = None
    ledger_replaced = False
    index_replaced = False
    try:
        # Reserve and verify the backup before the rewrite can create any CAS
        # object.  Failed compactions preserve both backups and any later CAS
        # additions for explicit operator inspection; they are never deleted.
        backup_root.mkdir(parents=True, exist_ok=False)
        backup_created = True
        ledger_backup = backup_root / ledger_path.name
        index_backup = backup_root / index_path.name
        ledger_backup_mode = _backup_file(ledger_path, ledger_backup)
        index_backup_mode = _backup_file(index_path, index_backup)
        if _file_sha256(ledger_backup) != audit.get('ledger_sha256'):
            raise RuntimeError('response_frame_ledger_backup_verification_failed')
        if _file_sha256(index_backup) != source_index_sha256:
            raise RuntimeError('response_frame_index_backup_verification_failed')

        if _stable_file_state(ledger_path) != source_ledger_state:
            raise RuntimeError('response_frame_ledger_moved_after_backup')
        if (
            _stable_file_state(index_path) != source_index_state
            or _file_sha256(index_path) != source_index_sha256
        ):
            raise RuntimeError('response_frame_index_moved_after_backup')

        temp_ledger_fd, temp_ledger_name = tempfile.mkstemp(
            prefix=f'.{ledger_path.name}.compaction.',
            suffix='.tmp',
            dir=str(frames_root),
        )
        os.close(temp_ledger_fd)
        temp_ledger_path = Path(temp_ledger_name)
        temp_index_fd, temp_index_name = tempfile.mkstemp(
            prefix=f'.{index_path.name}.compaction.',
            suffix='.tmp',
            dir=str(frames_root),
        )
        os.close(temp_index_fd)
        temp_index_path = Path(temp_index_name)

        rewrite = _write_compacted_ledger(
            frames_dir=frames_root,
            ledger_path=ledger_path,
            temp_ledger_path=temp_ledger_path,
        )
        index_payload = rewrite.pop('index_payload')
        source_index_mode = int(index_path.stat().st_mode) & 0o777
        with temp_index_path.open('w', encoding='utf-8') as handle:
            os.fchmod(handle.fileno(), source_index_mode)
            json.dump(
                response_frames._json_safe(index_payload),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())

        if _stable_file_state(ledger_path) != source_ledger_state:
            raise RuntimeError('response_frame_ledger_moved')
        if (
            _stable_file_state(index_path) != source_index_state
            or _file_sha256(index_path) != source_index_sha256
        ):
            raise RuntimeError('response_frame_index_moved')
        if rewrite['source_sha256'] != audit.get('ledger_sha256'):
            raise RuntimeError('response_frame_ledger_rewrite_source_mismatch')

        _atomic_install(ledger_path, temp_ledger_path)
        ledger_replaced = True
        _atomic_install(index_path, temp_index_path)
        index_replaced = True

        attestation = response_frames.attest_response_frame_index(
            frames_dir=frames_root,
            ledger_name=ledger_name,
            index_name=index_name,
            write=False,
        )
        if attestation.get('ok') is not True:
            raise RuntimeError('response_frame_index_postflight_failed')
        postflight = audit_response_frame_ledger(
            frames_dir=frames_root,
            ledger_name=ledger_name,
        )
        if postflight.get('ok') is not True:
            raise RuntimeError('response_frame_sidecar_postflight_failed')
        if postflight.get('frame_identity_digest') != rewrite['frame_identity_digest']:
            raise RuntimeError('response_frame_identity_postflight_mismatch')
        new_cas_files = sorted(
            _content_sha256_cas_files(frames_root) - cas_files_before
        )
        rewrite['new_cas_file_count'] = len(new_cas_files)
        rewrite['new_cas_files'] = new_cas_files
        rewrite['new_cas_files_preserved'] = True
        return {
            'ok': True,
            'changed': True,
            'mode': 'execute',
            'status': 'compacted',
            'preflight': audit,
            'index_preflight': index_preflight,
            'rewrite': rewrite,
            'backup': {
                'directory': str(backup_root),
                'ledger_path': str(ledger_backup),
                'index_path': str(index_backup),
                'ledger_mode': ledger_backup_mode,
                'index_mode': index_backup_mode,
            },
            'attestation': attestation,
            'postflight': postflight,
        }
    except Exception as exc:
        new_cas_files = sorted(
            _content_sha256_cas_files(frames_root) - cas_files_before
        )
        return {
            'ok': False,
            'changed': bool(
                backup_created
                or ledger_replaced
                or index_replaced
                or new_cas_files
            ),
            'mode': 'execute',
            'preflight': audit,
            'backup_directory': str(backup_root),
            'backup_created': backup_created,
            'ledger_replaced': ledger_replaced,
            'index_replaced': index_replaced,
            'new_cas_file_count': len(new_cas_files),
            'new_cas_files': new_cas_files,
            'new_cas_files_preserved': True,
            'error': {
                'code': (
                    str(exc)
                    if isinstance(exc, RuntimeError)
                    else 'response_frame_ledger_compaction_failed'
                ),
                'message': str(exc),
            },
        }
    finally:
        for temp_path in (temp_ledger_path, temp_index_path):
            if temp_path is None:
                continue
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    'audit_response_frame_ledger',
    'compact_response_frame_ledger',
]
