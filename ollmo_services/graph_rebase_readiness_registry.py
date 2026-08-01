"""Durable evidence-only graph-rebase readiness observations.

The response-frame ledger remains canonical runtime truth.  This append-only
registry retains only the existing bounded readiness projection so rollout
observation can span archived ledger epochs.  Registry records never grant
operator, staging, authorization, or execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Optional

from ollmo_services.graph_rebase_rollout import (
    GRAPH_REBASE_READINESS_OBSERVATION_KIND,
    build_graph_rebase_readiness_report,
    project_graph_rebase_readiness_observation,
)
from ollmo_services.response_frames import (
    DEFAULT_RESPONSE_FRAME_INDEX,
    DEFAULT_RESPONSE_FRAME_LEDGER,
    DEFAULT_RESPONSE_FRAMES_DIR,
    load_latest_response_observation_state,
    select_graph_rebase_observation_response_ids,
    verify_response_frame_epoch,
)


DEFAULT_GRAPH_REBASE_READINESS_REGISTRY = Path(
    'state/graph_rebase/readiness_observations.jsonl'
)
DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH = (
    DEFAULT_GRAPH_REBASE_READINESS_REGISTRY
)
GRAPH_REBASE_READINESS_REGISTRY_RECORD_KIND = (
    'ollmo.graph_rebase_readiness_registry_record'
)
GRAPH_REBASE_READINESS_SOURCE_EPOCH_KIND = (
    'ollmo.response_frame_epoch_identity'
)
GRAPH_REBASE_READINESS_REGISTRY_VERSION = 1

_REGISTRY_LOCK = threading.RLock()
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_MAX_OBSERVATION_BYTES = 2 * 1024 * 1024
_RECORD_KEYS = {
    'kind',
    'version',
    'runtime_effect',
    'record_id',
    'response_id',
    'frame_id',
    'frame_sequence',
    'source_frame_sha256',
    'observation_sha256',
    'source_epoch',
    'observation',
    'record_sha256',
}
_SOURCE_EPOCH_KEYS = {
    'kind',
    'version',
    'runtime_effect',
    'source_epoch_id',
    'ledger_name',
    'epoch_anchor_response_id',
    'epoch_anchor_frame_id',
    'epoch_anchor_frame_sequence',
    'epoch_anchor_frame_sha256',
}


class GraphRebaseReadinessRegistryError(RuntimeError):
    """Fail-closed registry error suitable for control-plane projection."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or 'readiness_registry_error')
        self.status_code = int(status_code)
        self.details = _json_safe(dict(details or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'message': str(self),
            **self.details,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or '').strip()))


def _file_state(path: Path) -> Optional[dict[str, int]]:
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


def _record_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    error = {'code': code, 'message': message}
    error.update(_json_safe(details))
    return error


def build_graph_rebase_source_epoch_identity(
    verified_epoch: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the path-independent identity bound into every registry record."""

    anchor = (
        verified_epoch.get('epoch_anchor')
        if isinstance(verified_epoch.get('epoch_anchor'), Mapping)
        else {}
    )
    payload = {
        'kind': GRAPH_REBASE_READINESS_SOURCE_EPOCH_KIND,
        'version': 1,
        'runtime_effect': 'none',
        'ledger_name': Path(str(verified_epoch.get('ledger_path') or '')).name,
        'epoch_anchor_response_id': str(anchor.get('response_id') or '').strip(),
        'epoch_anchor_frame_id': str(anchor.get('frame_id') or '').strip(),
        'epoch_anchor_frame_sequence': anchor.get('frame_sequence'),
        'epoch_anchor_frame_sha256': str(
            anchor.get('source_frame_sha256') or ''
        ).strip(),
    }
    identity = dict(payload)
    payload['source_epoch_id'] = f'response-frame-epoch-{_sha256(identity)}'
    return _json_safe(payload)


def _validate_source_epoch(source_epoch: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(source_epoch, Mapping):
        return [_record_error(
            'readiness_registry_source_epoch_invalid',
            'Registry record source_epoch is not an object.',
        )]
    if set(source_epoch) != _SOURCE_EPOCH_KEYS:
        errors.append(_record_error(
            'readiness_registry_source_epoch_schema_mismatch',
            'Registry record source_epoch fields do not match version 1.',
            missing_fields=sorted(_SOURCE_EPOCH_KEYS - set(source_epoch)),
            unexpected_fields=sorted(set(source_epoch) - _SOURCE_EPOCH_KEYS),
        ))
        return errors
    if source_epoch.get('kind') != GRAPH_REBASE_READINESS_SOURCE_EPOCH_KIND:
        errors.append(_record_error(
            'readiness_registry_source_epoch_schema_mismatch',
            'Registry record source_epoch kind is invalid.',
        ))
    if source_epoch.get('version') != 1 or source_epoch.get('runtime_effect') != 'none':
        errors.append(_record_error(
            'readiness_registry_source_epoch_schema_mismatch',
            'Registry record source_epoch version or runtime effect is invalid.',
        ))
    if str(source_epoch.get('ledger_name') or '').strip() != DEFAULT_RESPONSE_FRAME_LEDGER:
        errors.append(_record_error(
            'readiness_registry_source_epoch_ledger_invalid',
            'Registry source epoch ledger name is invalid.',
        ))
    for key in ('epoch_anchor_response_id', 'epoch_anchor_frame_id'):
        if not str(source_epoch.get(key) or '').strip():
            errors.append(_record_error(
                'readiness_registry_source_epoch_anchor_invalid',
                f'Registry source epoch {key} is empty.',
                field=key,
            ))
    anchor_sequence = source_epoch.get('epoch_anchor_frame_sequence')
    if (
        not isinstance(anchor_sequence, int)
        or isinstance(anchor_sequence, bool)
        or anchor_sequence < 0
    ):
        errors.append(_record_error(
            'readiness_registry_source_epoch_anchor_invalid',
            'Registry source epoch anchor sequence is invalid.',
        ))
    if not _is_sha256(source_epoch.get('epoch_anchor_frame_sha256')):
        errors.append(_record_error(
            'readiness_registry_source_epoch_digest_invalid',
            'Registry source epoch anchor digest is not a SHA-256 digest.',
        ))
    identity = {
        key: source_epoch.get(key)
        for key in _SOURCE_EPOCH_KEYS
        if key != 'source_epoch_id'
    }
    expected_id = f'response-frame-epoch-{_sha256(identity)}'
    if source_epoch.get('source_epoch_id') != expected_id:
        errors.append(_record_error(
            'readiness_registry_source_epoch_id_mismatch',
            'Registry source epoch id does not match its exact evidence.',
        ))
    return errors


def build_graph_rebase_readiness_registry_record(
    observation: Mapping[str, Any],
    *,
    source_epoch: Mapping[str, Any],
    source_frame_sha256: str,
) -> dict[str, Any]:
    """Build one deterministic registry record from a settled projection."""

    projected = _json_safe(dict(observation))
    response_id = str(projected.get('response_id') or '').strip()
    frame_id = str(projected.get('frame_id') or '').strip()
    frame_sequence = projected.get('ledger_sequence')
    observation_sha256 = _sha256(projected)
    identity = {
        'response_id': response_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'source_epoch_id': str(source_epoch.get('source_epoch_id') or '').strip(),
        'source_frame_sha256': str(source_frame_sha256 or '').strip(),
        'observation_sha256': observation_sha256,
    }
    record: dict[str, Any] = {
        'kind': GRAPH_REBASE_READINESS_REGISTRY_RECORD_KIND,
        'version': GRAPH_REBASE_READINESS_REGISTRY_VERSION,
        'runtime_effect': 'none',
        'record_id': f'graph-rebase-readiness-{_sha256(identity)}',
        'response_id': response_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'source_frame_sha256': str(source_frame_sha256 or '').strip(),
        'observation_sha256': observation_sha256,
        'source_epoch': _json_safe(dict(source_epoch)),
        'observation': projected,
    }
    record['record_sha256'] = _sha256(record)
    return record


def validate_graph_rebase_readiness_registry_record(
    record: Any,
) -> list[dict[str, Any]]:
    """Return fail-closed schema and digest errors for one registry record."""

    if not isinstance(record, Mapping):
        return [_record_error(
            'readiness_registry_record_invalid',
            'Registry line is not a JSON object.',
        )]
    errors: list[dict[str, Any]] = []
    if set(record) != _RECORD_KEYS:
        return [_record_error(
            'readiness_registry_record_schema_mismatch',
            'Registry record fields do not match version 1.',
            missing_fields=sorted(_RECORD_KEYS - set(record)),
            unexpected_fields=sorted(set(record) - _RECORD_KEYS),
        )]
    if (
        record.get('kind') != GRAPH_REBASE_READINESS_REGISTRY_RECORD_KIND
        or record.get('version') != GRAPH_REBASE_READINESS_REGISTRY_VERSION
        or record.get('runtime_effect') != 'none'
    ):
        errors.append(_record_error(
            'readiness_registry_record_schema_mismatch',
            'Registry record kind, version, or runtime effect is invalid.',
        ))
    errors.extend(_validate_source_epoch(record.get('source_epoch')))
    observation = record.get('observation')
    if not isinstance(observation, Mapping):
        errors.append(_record_error(
            'readiness_registry_observation_invalid',
            'Registry observation is not an object.',
        ))
        return errors
    encoded_observation = _canonical_bytes(observation)
    if len(encoded_observation) > _MAX_OBSERVATION_BYTES:
        errors.append(_record_error(
            'readiness_registry_observation_too_large',
            'Registry observation exceeds the bounded projection limit.',
            size_bytes=len(encoded_observation),
        ))
    if (
        observation.get('kind') != GRAPH_REBASE_READINESS_OBSERVATION_KIND
        or observation.get('projection_status') != 'projected'
        or observation.get('runtime_effect') != 'none'
    ):
        errors.append(_record_error(
            'readiness_registry_observation_schema_mismatch',
            'Registry observation is not a projected readiness observation.',
        ))
    response_id = str(record.get('response_id') or '').strip()
    frame_id = str(record.get('frame_id') or '').strip()
    frame_sequence = record.get('frame_sequence')
    if not response_id or observation.get('response_id') != response_id:
        errors.append(_record_error(
            'readiness_registry_response_id_mismatch',
            'Registry response id does not match the observation.',
        ))
    if not frame_id or observation.get('frame_id') != frame_id:
        errors.append(_record_error(
            'readiness_registry_frame_id_mismatch',
            'Registry frame id does not match the observation.',
        ))
    if (
        not isinstance(frame_sequence, int)
        or isinstance(frame_sequence, bool)
        or frame_sequence < 0
        or observation.get('ledger_sequence') != frame_sequence
    ):
        errors.append(_record_error(
            'readiness_registry_frame_sequence_mismatch',
            'Registry frame sequence does not match the observation.',
        ))
    readiness_state = observation.get('readiness_state')
    if (
        not isinstance(readiness_state, Mapping)
        or readiness_state.get('settled_final') is not True
        or readiness_state.get('active_late_fill') is True
    ):
        errors.append(_record_error(
            'readiness_registry_observation_not_settled',
            'Only settled, non-active readiness observations may be retained.',
        ))
    if not _is_sha256(record.get('source_frame_sha256')):
        errors.append(_record_error(
            'readiness_registry_source_frame_digest_invalid',
            'Registry source-frame digest is not a SHA-256 digest.',
        ))
    observation_sha256 = _sha256(observation)
    if record.get('observation_sha256') != observation_sha256:
        errors.append(_record_error(
            'readiness_registry_observation_digest_mismatch',
            'Registry observation digest does not match canonical bytes.',
        ))
    source_epoch = record.get('source_epoch')
    identity = {
        'response_id': response_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'source_epoch_id': (
            str(source_epoch.get('source_epoch_id') or '').strip()
            if isinstance(source_epoch, Mapping)
            else ''
        ),
        'source_frame_sha256': str(record.get('source_frame_sha256') or '').strip(),
        'observation_sha256': observation_sha256,
    }
    if record.get('record_id') != f'graph-rebase-readiness-{_sha256(identity)}':
        errors.append(_record_error(
            'readiness_registry_record_id_mismatch',
            'Registry record id does not match exact observation lineage.',
        ))
    record_body = {key: value for key, value in record.items() if key != 'record_sha256'}
    if record.get('record_sha256') != _sha256(record_body):
        errors.append(_record_error(
            'readiness_registry_record_digest_mismatch',
            'Registry record digest does not match canonical bytes.',
        ))
    return errors


def _parse_registry_bytes(
    raw: bytes,
    *,
    registry_path: Path,
) -> dict[str, Any]:
    if raw and not raw.endswith(b'\n'):
        return {
            'ok': False,
            'error': _record_error(
                'readiness_registry_partial_line',
                'Registry does not end at a complete JSONL record.',
            ),
        }
    records: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    record_bytes_by_id: dict[str, bytes] = {}
    duplicate_record_count = 0
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        if not raw_line.strip():
            return {
                'ok': False,
                'error': _record_error(
                    'readiness_registry_blank_line',
                    'Registry contains a blank physical line.',
                    line_number=line_number,
                ),
            }
        try:
            record = json.loads(raw_line.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                'ok': False,
                'error': _record_error(
                    'readiness_registry_corrupt_line',
                    str(exc),
                    line_number=line_number,
                ),
            }
        errors = validate_graph_rebase_readiness_registry_record(record)
        if errors:
            return {
                'ok': False,
                'error': {**errors[0], 'line_number': line_number},
            }
        canonical = _canonical_bytes(record)
        if canonical != raw_line:
            return {
                'ok': False,
                'error': _record_error(
                    'readiness_registry_noncanonical_line',
                    'Registry record bytes are not in canonical JSON form.',
                    line_number=line_number,
                ),
            }
        record_id = str(record.get('record_id') or '').strip()
        if record_id in records_by_id:
            if record_bytes_by_id[record_id] != raw_line:
                return {
                    'ok': False,
                    'error': _record_error(
                        'readiness_registry_record_id_collision',
                        'The same registry record id has different bytes.',
                        line_number=line_number,
                        record_id=record_id,
                    ),
                }
            duplicate_record_count += 1
            continue
        safe_record = _json_safe(record)
        records.append(safe_record)
        records_by_id[record_id] = safe_record
        record_bytes_by_id[record_id] = raw_line
    return {
        'ok': True,
        'registry_path': str(registry_path),
        'registry_sha256': hashlib.sha256(raw).hexdigest(),
        'record_count': len(records),
        'physical_record_count': len(raw.splitlines()),
        'duplicate_record_count': duplicate_record_count,
        'unique_response_count': len({record['response_id'] for record in records}),
        'records': records,
        'records_by_id': records_by_id,
        'record_bytes_by_id': record_bytes_by_id,
        'observations': [record['observation'] for record in records],
    }


def load_graph_rebase_readiness_registry(
    registry_path: Path | str = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY,
) -> dict[str, Any]:
    """Load and verify the complete registry without mutating it."""

    target = Path(registry_path)
    if not target.exists():
        return {
            'ok': True,
            'missing': True,
            'runtime_effect': 'none',
            'registry_path': str(target),
            'registry_sha256': hashlib.sha256(b'').hexdigest(),
            'record_count': 0,
            'physical_record_count': 0,
            'duplicate_record_count': 0,
            'unique_response_count': 0,
            'records': [],
            'records_by_id': {},
            'record_bytes_by_id': {},
            'observations': [],
        }
    with _REGISTRY_LOCK:
        try:
            with target.open('rb') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                raw = handle.read()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            return {
                'ok': False,
                'runtime_effect': 'none',
                'registry_path': str(target),
                'error': _record_error(
                    'readiness_registry_read_failed',
                    str(exc),
                ),
            }
    result = _parse_registry_bytes(raw, registry_path=target)
    result['runtime_effect'] = 'none'
    return result


def load_graph_rebase_readiness_records(
    registry_path: Path | str = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY,
) -> list[dict[str, Any]]:
    """Return verified records or raise a structured fail-closed error."""

    result = load_graph_rebase_readiness_registry(registry_path)
    if result.get('ok') is not True:
        error = result.get('error') if isinstance(result.get('error'), Mapping) else {}
        raise GraphRebaseReadinessRegistryError(
            str(error.get('code') or 'readiness_registry_read_failed'),
            str(error.get('message') or 'Readiness registry could not be verified.'),
            details={
                key: value
                for key, value in error.items()
                if key not in {'code', 'message'}
            },
        )
    return [dict(item) for item in result.get('records') or []]


def load_graph_rebase_readiness_observations(
    registry_path: Path | str = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY,
) -> list[dict[str, Any]]:
    """Return only bounded observations from the verified registry."""

    return [
        dict(record.get('observation') or {})
        for record in load_graph_rebase_readiness_records(registry_path)
    ]


def append_graph_rebase_readiness_registry_records(
    records: Sequence[Mapping[str, Any]],
    *,
    registry_path: Path | str = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY,
) -> dict[str, Any]:
    """Append validated records atomically under process and file locks."""

    target = Path(registry_path)
    proposed: list[dict[str, Any]] = []
    proposed_by_id: dict[str, bytes] = {}
    for index, raw_record in enumerate(records):
        record = _json_safe(dict(raw_record)) if isinstance(raw_record, Mapping) else raw_record
        errors = validate_graph_rebase_readiness_registry_record(record)
        if errors:
            return {
                'ok': False,
                'status': 'rejected',
                'runtime_effect': 'none',
                'registry_path': str(target),
                'error': {**errors[0], 'input_index': index},
                'appended_record_count': 0,
                'already_present_count': 0,
            }
        record_id = str(record.get('record_id') or '').strip()
        encoded = _canonical_bytes(record)
        if record_id in proposed_by_id and proposed_by_id[record_id] != encoded:
            return {
                'ok': False,
                'status': 'rejected',
                'runtime_effect': 'none',
                'registry_path': str(target),
                'error': _record_error(
                    'readiness_registry_record_id_collision',
                    'Input contains different bytes for one registry record id.',
                    record_id=record_id,
                ),
                'appended_record_count': 0,
                'already_present_count': 0,
            }
        if record_id not in proposed_by_id:
            proposed.append(record)
            proposed_by_id[record_id] = encoded

    target.parent.mkdir(parents=True, exist_ok=True)
    with _REGISTRY_LOCK:
        try:
            with target.open('a+b') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                raw = handle.read()
                current = _parse_registry_bytes(raw, registry_path=target)
                if current.get('ok') is not True:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return {
                        'ok': False,
                        'status': 'rejected',
                        'runtime_effect': 'none',
                        'registry_path': str(target),
                        'error': current.get('error'),
                        'appended_record_count': 0,
                        'already_present_count': 0,
                    }
                existing_bytes = current.get('record_bytes_by_id') or {}
                to_append: list[bytes] = []
                already_present = 0
                for record in proposed:
                    record_id = str(record.get('record_id') or '').strip()
                    encoded = proposed_by_id[record_id]
                    if record_id in existing_bytes:
                        if existing_bytes[record_id] != encoded:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                            return {
                                'ok': False,
                                'status': 'rejected',
                                'runtime_effect': 'none',
                                'registry_path': str(target),
                                'error': _record_error(
                                    'readiness_registry_record_id_collision',
                                    'Existing record id has different bytes.',
                                    record_id=record_id,
                                ),
                                'appended_record_count': 0,
                                'already_present_count': already_present,
                            }
                        already_present += 1
                        continue
                    to_append.append(encoded + b'\n')
                if to_append:
                    handle.seek(0, os.SEEK_END)
                    for line in to_append:
                        handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            return {
                'ok': False,
                'status': 'rejected',
                'runtime_effect': 'none',
                'registry_path': str(target),
                'error': _record_error('readiness_registry_write_failed', str(exc)),
                'appended_record_count': 0,
                'already_present_count': 0,
            }
        if to_append:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    verified = load_graph_rebase_readiness_registry(target)
    if verified.get('ok') is not True:
        return {
            'ok': False,
            'status': 'rejected',
            'runtime_effect': 'none',
            'registry_path': str(target),
            'error': verified.get('error'),
            'appended_record_count': len(to_append),
            'already_present_count': already_present,
        }
    return {
        'ok': True,
        'status': 'appended' if to_append else 'unchanged',
        'runtime_effect': 'none',
        'registry_path': str(target),
        'appended_record_count': len(to_append),
        'already_present_count': already_present,
        'record_count': verified.get('record_count'),
        'unique_response_count': verified.get('unique_response_count'),
        'registry_sha256': verified.get('registry_sha256'),
    }


def append_graph_rebase_readiness_observation(
    payload_or_projection: Mapping[str, Any],
    *,
    source_frame: Mapping[str, Any] | str,
    source_epoch: Optional[Mapping[str, Any]] = None,
    verified_epoch: Optional[Mapping[str, Any]] = None,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
    registry_path: Path | str = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY,
) -> dict[str, Any]:
    """Project and append one settled observation with exact source lineage."""

    if not isinstance(payload_or_projection, Mapping):
        raise TypeError('payload_or_projection must be a mapping')
    projection = (
        dict(payload_or_projection)
        if payload_or_projection.get('kind') == GRAPH_REBASE_READINESS_OBSERVATION_KIND
        else project_graph_rebase_readiness_observation(payload_or_projection)
    )
    if verified_epoch is None:
        verified = verify_response_frame_epoch(
            frames_dir=frames_dir,
            ledger_name=ledger_name,
            index_name=index_name,
            allow_relocated=False,
        )
    else:
        verified = dict(verified_epoch)
        expected_ledger_path = (
            Path(frames_dir)
            / (str(ledger_name or '').strip() or DEFAULT_RESPONSE_FRAME_LEDGER)
        )
        expected_index_path = (
            Path(frames_dir)
            / (str(index_name or '').strip() or DEFAULT_RESPONSE_FRAME_INDEX)
        )
        try:
            paths_match = (
                Path(str(verified.get('ledger_path') or '')).resolve()
                == expected_ledger_path.resolve()
                and Path(str(verified.get('index_path') or '')).resolve()
                == expected_index_path.resolve()
            )
        except OSError:
            paths_match = False
        ledger_file_state = (
            verified.get('ledger_file_state')
            if isinstance(verified.get('ledger_file_state'), Mapping)
            else None
        )
        index_file_state = (
            verified.get('index_file_state')
            if isinstance(verified.get('index_file_state'), Mapping)
            else None
        )
        if (
            verified.get('ok') is not True
            or verified.get('relocated') is True
            or not paths_match
            or ledger_file_state is None
            or index_file_state is None
            or _file_state(expected_ledger_path) != dict(ledger_file_state)
            or _file_state(expected_index_path) != dict(index_file_state)
        ):
            raise GraphRebaseReadinessRegistryError(
                'readiness_epoch_moved',
                'Preverified response-frame epoch is no longer current.',
            )
    if verified.get('ok') is not True:
        error = (
            verified.get('error')
            if isinstance(verified.get('error'), Mapping)
            else {}
        )
        raise GraphRebaseReadinessRegistryError(
            str(error.get('code') or 'readiness_epoch_verification_failed'),
            str(error.get('message') or 'Response-frame epoch could not be verified.'),
            details={
                key: value
                for key, value in error.items()
                if key not in {'code', 'message'}
            },
        )

    verified_source_epoch = build_graph_rebase_source_epoch_identity(
        verified
    )
    if source_epoch is not None and _canonical_bytes(
        source_epoch
    ) != _canonical_bytes(verified_source_epoch):
        raise GraphRebaseReadinessRegistryError(
            'readiness_registry_source_epoch_mismatch',
            'The supplied source epoch does not match current verified ledger truth.',
        )
    source_epoch = verified_source_epoch

    response_id = str(projection.get('response_id') or '').strip()
    frame_digests = (
        verified.get('source_frame_sha256_by_response')
        if isinstance(
            verified.get('source_frame_sha256_by_response'),
            Mapping,
        )
        else {}
    )
    expected_source_frame_sha256 = str(
        frame_digests.get(response_id) or ''
    ).strip()
    supplied_source_frame_sha256 = (
        str(source_frame or '').strip()
        if isinstance(source_frame, str)
        else ''
    )
    if (
        supplied_source_frame_sha256
        and supplied_source_frame_sha256 != expected_source_frame_sha256
    ):
        raise GraphRebaseReadinessRegistryError(
            'readiness_registry_source_frame_digest_mismatch',
            'The supplied source-frame digest does not match verified ledger truth.',
            details={'response_id': response_id},
        )
    source_frame_sha256 = expected_source_frame_sha256
    index_state = (
        verified.get('index_state')
        if isinstance(verified.get('index_state'), Mapping)
        else {}
    )
    entries = (
        index_state.get('responses')
        if isinstance(index_state.get('responses'), Mapping)
        else {}
    )
    entry = entries.get(response_id) if isinstance(entries, Mapping) else None
    if (
        not isinstance(entry, Mapping)
        or entry.get('latest_frame_id') != projection.get('frame_id')
        or entry.get('latest_frame_sequence') != projection.get('ledger_sequence')
        or not _is_sha256(source_frame_sha256)
    ):
        raise GraphRebaseReadinessRegistryError(
            'readiness_registry_source_frame_mismatch',
            'The supplied observation is not the verified latest source frame.',
            details={'response_id': response_id},
        )
    observed = load_latest_response_observation_state(
        response_id,
        frames_dir=frames_dir,
        ledger_name=ledger_name,
        index_state=index_state,
    )
    observed_payload = (
        observed.get('response_payload')
        if isinstance(observed.get('response_payload'), Mapping)
        else {}
    )
    if observed.get('ok') is not True or not observed_payload:
        error = (
            observed.get('error')
            if isinstance(observed.get('error'), Mapping)
            else {}
        )
        raise GraphRebaseReadinessRegistryError(
            str(error.get('code') or 'readiness_observation_hydration_failed'),
            str(
                error.get('message')
                or 'Verified readiness observation could not be hydrated.'
            ),
            details={'response_id': response_id},
        )
    canonical_projection = project_graph_rebase_readiness_observation(
        observed_payload
    )
    if (
        canonical_projection.get('response_id') != projection.get('response_id')
        or canonical_projection.get('frame_id') != projection.get('frame_id')
        or canonical_projection.get('ledger_sequence')
        != projection.get('ledger_sequence')
    ):
        raise GraphRebaseReadinessRegistryError(
            'readiness_registry_projection_binding_mismatch',
            'Caller projection does not bind the verified durable observation.',
            details={'response_id': response_id},
        )
    record = build_graph_rebase_readiness_registry_record(
        canonical_projection,
        source_epoch=source_epoch,
        source_frame_sha256=source_frame_sha256,
    )
    errors = validate_graph_rebase_readiness_registry_record(record)
    if errors:
        error = errors[0]
        raise GraphRebaseReadinessRegistryError(
            str(error.get('code') or 'readiness_registry_record_invalid'),
            str(error.get('message') or 'Readiness registry record is invalid.'),
            details={
                key: value
                for key, value in error.items()
                if key not in {'code', 'message'}
            },
        )
    return append_graph_rebase_readiness_registry_records(
        [record],
        registry_path=registry_path,
    )


def _expectation_error(
    actual: Any,
    expected: Any,
    *,
    field: str,
) -> Optional[dict[str, Any]]:
    if expected is None or actual == expected:
        return None
    return _record_error(
        'readiness_epoch_expectation_mismatch',
        f'Verified epoch field {field} does not match the required value.',
        field=field,
        actual=actual,
        expected=expected,
    )


def sync_graph_rebase_readiness_registry(
    *,
    frames_dir: Path | str = DEFAULT_RESPONSE_FRAMES_DIR,
    registry_path: Path | str = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY,
    ledger_name: str = DEFAULT_RESPONSE_FRAME_LEDGER,
    index_name: str = DEFAULT_RESPONSE_FRAME_INDEX,
    write: bool = False,
    require_all_settled: bool = False,
    allow_relocated: bool = True,
    expected_index_sha256: str | None = None,
    expected_ledger_sha256: str | None = None,
    expected_ledger_line_count: int | None = None,
    expected_response_count: int | None = None,
    expected_selected_count: int | None = None,
    expected_settled_count: int | None = None,
    expected_active_count: int | None = None,
) -> dict[str, Any]:
    """Verify one epoch and optionally retain all of its settled observations."""

    verified_epoch = verify_response_frame_epoch(
        frames_dir=frames_dir,
        ledger_name=ledger_name,
        index_name=index_name,
        allow_relocated=allow_relocated,
    )
    base_result: dict[str, Any] = {
        'kind': 'ollmo.graph_rebase_readiness_registry_sync',
        'runtime_effect': 'none',
        'mode': 'write' if write else 'check_only',
        'frames_dir': str(frames_dir),
        'registry_path': str(registry_path),
        'status': 'rejected',
        'selected_observation_count': 0,
        'settled_observation_count': 0,
        'active_observation_count': 0,
        'unsettled_observation_count': 0,
        'appended_record_count': 0,
        'already_present_count': 0,
        'would_append_record_count': 0,
        'registered_observation_count': 0,
        'missing_settled_observation_count': 0,
        'scan_error_count': 0,
        'hydration_error_count': 0,
        'registry_error_count': 0,
        'error_count': 0,
        'errors': [],
    }
    if verified_epoch.get('ok') is not True:
        base_result['errors'] = [verified_epoch.get('error') or _record_error(
            'readiness_epoch_verification_failed',
            'Response-frame epoch verification failed.',
        )]
        base_result['error_count'] = len(base_result['errors'])
        base_result['scan_error_count'] = 1
        return base_result

    base_result['source_epoch'] = build_graph_rebase_source_epoch_identity(
        verified_epoch
    )
    base_result['source'] = {
        key: verified_epoch.get(key)
        for key in (
            'index_path',
            'ledger_path',
            'stored_ledger_path',
            'relocated',
            'index_sha256',
            'ledger_sha256',
            'ledger_size_bytes',
            'ledger_line_count',
            'response_map_entry_count',
            'response_map_digest',
        )
    }
    expectations = (
        ('index_sha256', verified_epoch.get('index_sha256'), expected_index_sha256),
        ('ledger_sha256', verified_epoch.get('ledger_sha256'), expected_ledger_sha256),
        ('ledger_line_count', verified_epoch.get('ledger_line_count'), expected_ledger_line_count),
        ('response_map_entry_count', verified_epoch.get('response_map_entry_count'), expected_response_count),
    )
    for field, actual, expected in expectations:
        error = _expectation_error(actual, expected, field=field)
        if error:
            base_result['errors'].append(error)

    index_state = verified_epoch.get('index_state')
    selection = select_graph_rebase_observation_response_ids(
        frames_dir=frames_dir,
        index_state=index_state if isinstance(index_state, Mapping) else None,
    )
    base_result['selection'] = selection
    if int(selection.get('scan_error_count') or 0):
        base_result['scan_error_count'] = int(selection.get('scan_error_count') or 0)
        base_result['errors'].extend(
            selection.get('scan_errors')
            if isinstance(selection.get('scan_errors'), list)
            else []
        )
    selected_ids = (
        selection.get('selected_response_ids')
        if isinstance(selection.get('selected_response_ids'), list)
        else []
    )
    observations: list[dict[str, Any]] = []
    source_frame_digests = (
        verified_epoch.get('source_frame_sha256_by_response')
        if isinstance(verified_epoch.get('source_frame_sha256_by_response'), Mapping)
        else {}
    )
    for response_id in selected_ids:
        observed = load_latest_response_observation_state(
            str(response_id),
            frames_dir=frames_dir,
            ledger_name=ledger_name,
            index_state=index_state if isinstance(index_state, Mapping) else None,
        )
        if observed.get('ok') is not True:
            error = observed.get('error') if isinstance(observed.get('error'), Mapping) else {}
            base_result['errors'].append({
                'response_id': str(response_id),
                'code': str(error.get('code') or 'readiness_observation_hydration_failed'),
                'message': str(error.get('message') or 'Readiness observation could not be hydrated.'),
            })
            base_result['hydration_error_count'] += 1
            continue
        payload = observed.get('response_payload')
        projection = project_graph_rebase_readiness_observation(
            payload if isinstance(payload, Mapping) else {}
        )
        observations.append(projection)

    settled = [
        item
        for item in observations
        if isinstance(item.get('readiness_state'), Mapping)
        and item['readiness_state'].get('settled_final') is True
        and item['readiness_state'].get('active_late_fill') is not True
    ]
    active = [
        item
        for item in observations
        if isinstance(item.get('readiness_state'), Mapping)
        and item['readiness_state'].get('active_late_fill') is True
    ]
    unsettled = [item for item in observations if item not in settled]
    base_result['selected_observation_count'] = len(selected_ids)
    base_result['hydrated_observation_count'] = len(observations)
    base_result['settled_observation_count'] = len(settled)
    base_result['active_observation_count'] = len(active)
    base_result['unsettled_observation_count'] = len(unsettled)

    count_expectations = (
        ('selected_observation_count', len(selected_ids), expected_selected_count),
        ('settled_observation_count', len(settled), expected_settled_count),
        ('active_observation_count', len(active), expected_active_count),
    )
    for field, actual, expected in count_expectations:
        error = _expectation_error(actual, expected, field=field)
        if error:
            base_result['errors'].append(error)
    if require_all_settled and len(settled) != len(selected_ids):
        base_result['errors'].append(_record_error(
            'readiness_epoch_not_fully_settled',
            'All selected readiness observations must be settled before retention.',
            selected_observation_count=len(selected_ids),
            settled_observation_count=len(settled),
            active_observation_count=len(active),
        ))

    report = build_graph_rebase_readiness_report(
        observations,
        source_ledger_identity=base_result['source_epoch'],
    )
    corpus = report.get('corpus') if isinstance(report.get('corpus'), Mapping) else {}
    safety = report.get('safety') if isinstance(report.get('safety'), Mapping) else {}
    base_result['readiness_summary'] = {
        'settled_response_count': corpus.get('settled_final_response_count'),
        'active_response_count': corpus.get('nonterminal_active_late_fill_response_count'),
        'unique_workload_family_count': corpus.get('unique_workload_family_count'),
        'total_safety_finding_count': safety.get('total_finding_count'),
        'unresolved_critical_safety_finding_count': safety.get(
            'unresolved_critical_finding_count'
        ),
        'corpus_digest': corpus.get('corpus_digest'),
    }

    source_epoch = base_result['source_epoch']
    proposed_records = [
        build_graph_rebase_readiness_registry_record(
            observation,
            source_epoch=source_epoch,
            source_frame_sha256=str(
                source_frame_digests.get(observation.get('response_id')) or ''
            ),
        )
        for observation in settled
    ]
    for index, record in enumerate(proposed_records):
        errors = validate_graph_rebase_readiness_registry_record(record)
        if errors:
            base_result['errors'].append({**errors[0], 'input_index': index})

    current_registry = load_graph_rebase_readiness_registry(registry_path)
    if current_registry.get('ok') is not True:
        base_result['registry_error_count'] += 1
        base_result['errors'].append(
            current_registry.get('error') or _record_error(
                'readiness_registry_read_failed',
                'Readiness registry could not be verified.',
            )
        )
    existing_bytes = (
        current_registry.get('record_bytes_by_id')
        if isinstance(current_registry.get('record_bytes_by_id'), Mapping)
        else {}
    )
    already_present = 0
    would_append = 0
    for record in proposed_records:
        record_id = str(record.get('record_id') or '').strip()
        encoded = _canonical_bytes(record)
        if record_id in existing_bytes:
            if existing_bytes[record_id] != encoded:
                base_result['registry_error_count'] += 1
                base_result['errors'].append(_record_error(
                    'readiness_registry_record_id_collision',
                    'Existing record id has different bytes.',
                    record_id=record_id,
                ))
            else:
                already_present += 1
        else:
            would_append += 1
    base_result['already_present_count'] = already_present
    base_result['would_append_record_count'] = would_append
    base_result['registered_observation_count'] = already_present
    base_result['missing_settled_observation_count'] = would_append
    base_result['error_count'] = len(base_result['errors'])
    if base_result['errors']:
        return base_result

    if write:
        append_result = append_graph_rebase_readiness_registry_records(
            proposed_records,
            registry_path=registry_path,
        )
        base_result['append'] = append_result
        if append_result.get('ok') is not True:
            base_result['registry_error_count'] += 1
            base_result['errors'].append(
                append_result.get('error') or _record_error(
                    'readiness_registry_write_failed',
                    'Readiness registry append failed.',
                )
            )
            base_result['error_count'] = len(base_result['errors'])
            return base_result
        base_result['appended_record_count'] = int(
            append_result.get('appended_record_count') or 0
        )
        base_result['already_present_count'] = int(
            append_result.get('already_present_count') or 0
        )
        base_result['registered_observation_count'] = (
            base_result['already_present_count']
            + base_result['appended_record_count']
        )
        base_result['missing_settled_observation_count'] = max(
            0,
            len(settled) - base_result['registered_observation_count'],
        )
        base_result['status'] = (
            'written' if base_result['appended_record_count'] else 'unchanged'
        )
    else:
        base_result['status'] = 'verified'
    base_result['ok'] = True
    return base_result


def sync_graph_rebase_readiness_epoch(**kwargs: Any) -> dict[str, Any]:
    """Stable integration alias for syncing one verified ledger epoch."""

    return sync_graph_rebase_readiness_registry(**kwargs)


__all__ = [
    'DEFAULT_GRAPH_REBASE_READINESS_REGISTRY',
    'DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH',
    'GRAPH_REBASE_READINESS_REGISTRY_RECORD_KIND',
    'GRAPH_REBASE_READINESS_REGISTRY_VERSION',
    'GRAPH_REBASE_READINESS_SOURCE_EPOCH_KIND',
    'GraphRebaseReadinessRegistryError',
    'append_graph_rebase_readiness_observation',
    'append_graph_rebase_readiness_registry_records',
    'build_graph_rebase_readiness_registry_record',
    'build_graph_rebase_source_epoch_identity',
    'load_graph_rebase_readiness_observations',
    'load_graph_rebase_readiness_records',
    'load_graph_rebase_readiness_registry',
    'sync_graph_rebase_readiness_epoch',
    'sync_graph_rebase_readiness_registry',
    'validate_graph_rebase_readiness_registry_record',
]
