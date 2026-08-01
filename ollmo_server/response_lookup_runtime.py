"""Bounded durable/live response lookup arbitration for Ollmo."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


LookupRecord = dict[str, Any]
LookupResult = tuple[Optional[LookupRecord], Optional[dict[str, Any]], int]


def response_lookup_frame_sequence_value(payload: Mapping[str, Any]) -> Optional[int]:
    """Return the frozen frame sequence exposed by a response projection."""

    if not isinstance(payload, Mapping):
        return None
    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), Mapping)
        else {}
    )
    durability = (
        payload.get('durability')
        if isinstance(payload.get('durability'), Mapping)
        else {}
    )
    value = response_frame.get('frame_sequence')
    if value in (None, '', [], {}):
        value = durability.get('frame_sequence')
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def response_lookup_frame_id_value(payload: Mapping[str, Any]) -> str:
    """Return the frozen frame identifier exposed by a response projection."""

    if not isinstance(payload, Mapping):
        return ''
    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), Mapping)
        else {}
    )
    durability = (
        payload.get('durability')
        if isinstance(payload.get('durability'), Mapping)
        else {}
    )
    value = response_frame.get('frame_id')
    if value in (None, '', [], {}):
        value = durability.get('frame_id')
    return str(value or '').strip()


def response_wire_frame_identity(
    payload: Mapping[str, Any],
) -> Optional[tuple[str, int]]:
    """Return the exact frame CAS pair represented by a bounded payload."""

    frame_id = response_lookup_frame_id_value(payload)
    frame_sequence = response_lookup_frame_sequence_value(payload)
    if not frame_id or frame_sequence is None:
        return None
    return frame_id, frame_sequence


def _default_message_id() -> str:
    return f'msg_{uuid.uuid4().hex}'


@dataclass
class ResponseLookupRuntimeOwner:
    """Select bounded durable or live response truth without owning either."""

    normalize_response_lookup_id: Callable[[Any], str]
    get_live_response_lookup_record: Callable[[str], Optional[LookupRecord]]
    load_wire_payload_from_index: Callable[
        [str],
        tuple[Optional[dict[str, Any]], dict[str, Any]],
    ]
    project_fallback_payload: Callable[[Mapping[str, Any]], dict[str, Any]]
    recover_response_lookup_record: Callable[[str], LookupResult]
    project_late_fill: Callable[[Mapping[str, Any]], dict[str, Any]]
    project_surface: Callable[[Mapping[str, Any]], dict[str, Any]]
    derive_lifecycle_state: Callable[..., str]
    response_payload_message_id: Callable[[Optional[dict[str, Any]]], Optional[str]]
    response_registry_now_iso: Callable[[], str]
    response_frames_dir_getter: Callable[[], Path | str]
    inspect_recovered_cache: Callable[
        [str, Path | str, Mapping[str, Any]],
        Mapping[str, Any],
    ]
    advance_recovered_cache_checkpoint: Callable[[str, Mapping[str, int]], Any]
    response_lookup_ttl_sec: int
    now_ts: Callable[[], float] = time.time
    new_message_id: Callable[[], str] = _default_message_id

    def response_lookup_record_from_wire_payload(
        self,
        response_id: str,
        payload: Mapping[str, Any],
        *,
        lookup_source: str,
    ) -> LookupRecord:
        """Build the bounded registry envelope for one projected payload."""

        normalized_id = self.normalize_response_lookup_id(response_id)
        response_frame = (
            payload.get('response_frame')
            if isinstance(payload.get('response_frame'), Mapping)
            else {}
        )
        target = (
            response_frame.get('target')
            if isinstance(response_frame.get('target'), Mapping)
            else {}
        )
        lifecycle_state = self.derive_lifecycle_state(
            payload,
            requested_status=payload.get('status'),
        )
        payload_copy = copy.deepcopy(dict(payload))
        now_ts = self.now_ts()
        return {
            'id': normalized_id,
            'message_id': (
                self.response_payload_message_id(payload_copy)
                or self.new_message_id()
            ),
            'instance_id': str(
                payload.get('instance_id') or target.get('instance_id') or ''
            ).strip(),
            'model_name': str(
                payload.get('model') or target.get('model') or ''
            ).strip(),
            'backend': str(
                payload.get('backend') or target.get('backend') or ''
            ).strip(),
            'capability': str(
                payload.get('capability') or target.get('capability') or ''
            ).strip(),
            'mode': str(
                payload.get('mode') or target.get('mode') or 'chat'
            ).strip() or 'chat',
            'status': str(
                payload.get('status')
                or response_frame.get('status')
                or 'completed'
            ).strip() or 'completed',
            'lifecycle_state': lifecycle_state,
            'output_text': str(payload.get('output_text') or ''),
            'error_message': (
                str(payload.get('error', {}).get('message') or '').strip()
                if isinstance(payload.get('error'), Mapping)
                else None
            ),
            'response_payload': payload_copy,
            'route_payload': {},
            'created_at': self.response_registry_now_iso(),
            'updated_at': self.response_registry_now_iso(),
            'expires_at_ts': now_ts + self.response_lookup_ttl_sec,
            'lookup_source': lookup_source,
            'recovered_from_response_frame': (
                lookup_source == 'response_frame_wire_projection'
            ),
        }

    def response_wire_record_with_lookup_truth(
        self,
        projected_record: LookupRecord,
        source_record: Mapping[str, Any],
    ) -> LookupRecord:
        """Retain live registry metadata without replacing the bounded body."""

        updated = dict(projected_record)
        for key in (
            'status',
            'error_message',
            'created_at',
            'updated_at',
            'expires_at_ts',
            'message_id',
            'instance_id',
            'model_name',
            'backend',
            'capability',
            'mode',
            'route_payload',
        ):
            value = source_record.get(key)
            if value not in (None, '', [], {}):
                updated[key] = copy.deepcopy(value)
        response_payload = (
            updated.get('response_payload')
            if isinstance(updated.get('response_payload'), Mapping)
            else {}
        )
        if response_payload:
            updated['lifecycle_state'] = self.derive_lifecycle_state(
                response_payload,
                requested_status=updated.get('status'),
            )
        return updated

    def response_wire_overlay_live_state(
        self,
        projected: Mapping[str, Any],
        live_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Overlay only bounded volatile state for the same frozen frame."""

        updated = copy.deepcopy(dict(projected))
        live_payload = (
            live_record.get('response_payload')
            if isinstance(live_record.get('response_payload'), Mapping)
            else {}
        )
        live_late_fill = self.project_late_fill(live_payload)
        if live_late_fill:
            updated['late_fill'] = live_late_fill
        raw_live_surface = (
            live_payload.get('surface_state')
            if isinstance(live_payload.get('surface_state'), Mapping)
            else live_payload.get('runtime', {}).get('surface_state')
            if isinstance(live_payload.get('runtime'), Mapping)
            and isinstance(
                live_payload.get('runtime', {}).get('surface_state'),
                Mapping,
            )
            else {}
        )
        live_surface = self.project_surface(raw_live_surface)
        if live_surface:
            updated['surface_state'] = live_surface
        for key in ('status', 'lifecycle_state'):
            value = live_payload.get(key) or live_record.get(key)
            if value not in (None, '', [], {}):
                updated[key] = copy.deepcopy(value)
        updated['lifecycle_state'] = self.derive_lifecycle_state(
            updated,
            requested_status=updated.get('status'),
        )
        if live_record.get('error_message'):
            updated['error'] = {
                'message': str(live_record.get('error_message') or '').strip(),
            }
        projection = (
            dict(updated.get('wire_projection'))
            if isinstance(updated.get('wire_projection'), Mapping)
            else {}
        )
        projection['live_state_overlay'] = (
            'same_frozen_frame_bounded_status_late_fill_surface_only'
        )
        updated['wire_projection'] = projection
        return updated

    def response_frame_recovered_cache_valid(
        self,
        record: Mapping[str, Any],
        *,
        response_id: str,
    ) -> bool:
        """Decide cache reuse from read-only response-frame storage evidence."""

        expected = (
            record.get('response_frame_ledger_stat')
            if isinstance(record.get('response_frame_ledger_stat'), Mapping)
            else {}
        )
        ledger_path = str(record.get('response_frame_ledger_path') or '').strip()
        if not ledger_path or not expected:
            return False
        normalized_id = self.normalize_response_lookup_id(response_id)
        inspection = self.inspect_recovered_cache(
            normalized_id,
            ledger_path,
            expected,
        )
        if inspection.get('cache_reusable') is not True:
            return False
        checkpoint = (
            inspection.get('checkpoint_ledger_state')
            if isinstance(inspection.get('checkpoint_ledger_state'), Mapping)
            else None
        )
        if checkpoint is not None:
            checkpoint_token = {
                key: int(checkpoint[key])
                for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
                if key in checkpoint
            }
            if isinstance(record, dict):
                record['response_frame_ledger_stat'] = dict(checkpoint_token)
            self.advance_recovered_cache_checkpoint(
                normalized_id,
                checkpoint_token,
            )
        return True

    def get_bounded_response_lookup_record(
        self,
        response_id: str,
    ) -> LookupResult:
        """Resolve normal wire views without canonical hydration on success."""

        normalized_id = self.normalize_response_lookup_id(response_id)
        live_record = self.get_live_response_lookup_record(normalized_id)
        projected, wire_state = self.load_wire_payload_from_index(normalized_id)
        if projected is not None:
            wire_record = self.response_lookup_record_from_wire_payload(
                normalized_id,
                projected,
                lookup_source='response_frame_wire_projection',
            )
            if live_record:
                live_payload = (
                    live_record.get('response_payload')
                    if isinstance(live_record.get('response_payload'), Mapping)
                    else {}
                )
                live_sequence = response_lookup_frame_sequence_value(live_payload)
                wire_sequence = response_lookup_frame_sequence_value(projected)
                if (
                    live_sequence is not None
                    and wire_sequence is not None
                    and live_sequence > wire_sequence
                ):
                    live_projected = self.project_fallback_payload(live_payload)
                    return (
                        self.response_wire_record_with_lookup_truth(
                            self.response_lookup_record_from_wire_payload(
                                normalized_id,
                                live_projected,
                                lookup_source='response_wire_live_newer',
                            ),
                            live_record,
                        ),
                        None,
                        200,
                    )
                if (
                    live_sequence is not None
                    and wire_sequence is not None
                    and live_sequence == wire_sequence
                ):
                    live_frame_id = response_lookup_frame_id_value(live_payload)
                    wire_frame_id = response_lookup_frame_id_value(projected)
                    if (
                        live_frame_id
                        and wire_frame_id
                        and live_frame_id == wire_frame_id
                    ):
                        overlaid = self.response_wire_overlay_live_state(
                            projected,
                            live_record,
                        )
                        return (
                            self.response_wire_record_with_lookup_truth(
                                self.response_lookup_record_from_wire_payload(
                                    normalized_id,
                                    overlaid,
                                    lookup_source=(
                                        'response_wire_same_frame_live_overlay'
                                    ),
                                ),
                                live_record,
                            ),
                            None,
                            200,
                        )
            return wire_record, None, 200

        wire_error = (
            wire_state.get('error')
            if isinstance(wire_state.get('error'), Mapping)
            else {}
        )
        wire_error_code = str(wire_error.get('code') or '').strip()
        ledger_path = Path(self.response_frames_dir_getter()) / 'responses.jsonl'
        live_payload = (
            live_record.get('response_payload')
            if live_record
            and isinstance(live_record.get('response_payload'), Mapping)
            else {}
        )
        live_has_frozen_frame = bool(
            isinstance(live_payload.get('response_frame'), Mapping)
            and live_payload.get('response_frame')
        )
        if live_record and (
            wire_error_code == 'response_frame_not_found'
            or not ledger_path.exists()
            or not live_has_frozen_frame
        ):
            live_projected = self.project_fallback_payload(live_payload)
            return (
                self.response_wire_record_with_lookup_truth(
                    self.response_lookup_record_from_wire_payload(
                        normalized_id,
                        live_projected,
                        lookup_source='response_wire_live_only',
                    ),
                    live_record,
                ),
                None,
                200,
            )
        if not live_record and wire_error_code == 'response_frame_not_found':
            return (
                None,
                dict(wire_error),
                int(wire_state.get('status_code') or 404),
            )

        if (
            live_record
            and live_record.get('recovered_from_response_frame') is True
            and self.response_frame_recovered_cache_valid(
                live_record,
                response_id=normalized_id,
            )
        ):
            cached_payload = (
                live_record.get('bounded_response_payload')
                if isinstance(
                    live_record.get('bounded_response_payload'),
                    Mapping,
                )
                else self.project_fallback_payload(live_payload)
            )
            cached_payload = copy.deepcopy(dict(cached_payload))
            live_identity = response_wire_frame_identity(live_payload)
            cached_identity = response_wire_frame_identity(cached_payload)
            if live_identity and cached_identity:
                if live_identity[1] > cached_identity[1] or (
                    live_identity[1] == cached_identity[1]
                    and live_identity[0] != cached_identity[0]
                ):
                    live_projected = self.project_fallback_payload(live_payload)
                    return (
                        self.response_wire_record_with_lookup_truth(
                            self.response_lookup_record_from_wire_payload(
                                normalized_id,
                                live_projected,
                                lookup_source=(
                                    'response_wire_live_newer_than_recovery_cache'
                                ),
                            ),
                            live_record,
                        ),
                        None,
                        200,
                    )
                if live_identity == cached_identity:
                    cached_payload = self.response_wire_overlay_live_state(
                        cached_payload,
                        live_record,
                    )
            return (
                self.response_wire_record_with_lookup_truth(
                    self.response_lookup_record_from_wire_payload(
                        normalized_id,
                        cached_payload,
                        lookup_source='response_wire_cached_ledger_recovery',
                    ),
                    live_record,
                ),
                None,
                200,
            )

        recovered_record, recovery_error, recovery_status = (
            self.recover_response_lookup_record(normalized_id)
        )
        if recovered_record:
            recovered_payload = (
                recovered_record.get('response_payload')
                if isinstance(recovered_record.get('response_payload'), Mapping)
                else {}
            )
            recovered_projected = self.project_fallback_payload(
                recovered_payload,
            )
            return (
                self.response_lookup_record_from_wire_payload(
                    normalized_id,
                    recovered_projected,
                    lookup_source='response_wire_exceptional_ledger_recovery',
                ),
                None,
                200,
            )
        return None, recovery_error or dict(wire_error), recovery_status
