"""Intent-aligned redraw scope ladder review helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from typing import Any, Optional

REDRAW_SCOPE_REVIEW_KIND = 'ollmo.redraw_scope_ladder_review'

SCOPE_LADDER = [
    'observe',
    'promote_reserved_slot',
    'fill_reserved_slot',
    'add_missing_branch',
    'repair_binding_dependency',
    'repair_artifact_ref_identity',
    'partial_subtree_rebase',
    'full_successor_rebase',
    'apply_enforced_reserved',
]

_FORBIDDEN_AUTHORITY_TOKENS = {
    'cache_liveness',
    'degraded',
    'degraded_liveness',
    'frontend_label',
    'liveness_only',
    'monitor_only',
    'provider_degraded',
    'provider_family',
    'provider_warning',
    'route_health',
    'ui_label',
}
_REPAIR_STATUSES = {
    'actionable',
    'blocked',
    'failed',
    'pending',
    'repair_needed',
    'repair_pending',
    'repair_required',
    'review_pending',
}
_TEXT_ARTIFACT_IDENTITY_EXTENSIONS = {
    'cjs',
    'css',
    'csv',
    'htm',
    'html',
    'js',
    'json',
    'jsx',
    'markdown',
    'md',
    'mjs',
    'py',
    'sh',
    'sql',
    'svg',
    'ts',
    'tsx',
    'txt',
    'xml',
    'yaml',
    'yml',
}
_RESERVED_STATUSES = {
    'candidate',
    'candidate_only',
    'not_promoted',
    'possible',
    'reserved',
}
_DEFER_TOKENS = {
    'defer',
    'deferred',
    'explicit_defer',
    'for_later',
    'hold',
    'later',
    'reserved_by_user',
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _status(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_').replace(' ', '_')


def _nonnegative_integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text and text.isascii() and text.isdigit():
            try:
                return int(text)
            except (ValueError, OverflowError):
                return None
    return None


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        return []
    values: list[str] = []
    for item in raw_items:
        text = _clean_text(item)
        if text and text not in values:
            values.append(text)
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _clean_text(key): _json_safe(raw_value)
            for key, raw_value in value.items()
            if _clean_text(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        _clean_text(key): _json_safe(value)
        for key, value in payload.items()
        if _clean_text(key) and value not in (None, '', [], {})
    }


def _stable_digest(value: Any, *, prefix: str = '') -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
    return f'{prefix}{digest}' if prefix else digest


def _append_unique(values: list[str], value: str) -> None:
    text = _clean_text(value)
    if text and text not in values:
        values.append(text)


def _records(value: Any, key: str) -> list[dict[str, Any]]:
    source = value.get(key) if isinstance(value, Mapping) else []
    if not isinstance(source, list):
        return []
    return [dict(item) for item in source if isinstance(item, Mapping)]


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, Mapping) else {}


def _artifact_ref(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('artifact_ref') or record.get('ref') or record.get('artifact_id'))


def _artifact_path(record: Mapping[str, Any]) -> str:
    return _clean_text(
        record.get('path')
        or record.get('saved_path')
        or record.get('artifact_path')
        or record.get('file_path')
        or record.get('src')
    )


def _artifact_type(record: Mapping[str, Any]) -> str:
    return _status(record.get('type') or record.get('kind') or record.get('output_type'))


def _document_text_identity_alias_is_proven(
    records: Sequence[Mapping[str, Any]],
    *,
    paths: set[str],
    raw_types: set[str],
) -> bool:
    if raw_types != {'document', 'text'} or len(paths) != 1:
        return False
    if any(not _artifact_path(record) for record in records):
        return False
    if any(_artifact_path(record) != next(iter(paths)) for record in records):
        return False
    if any(_artifact_type(record) not in {'document', 'text'} for record in records):
        return False
    canonical_path = next(iter(paths))
    path_token = canonical_path.split('?', 1)[0].split('#', 1)[0].rstrip('/')
    extension = path_token.rsplit('.', 1)[-1].lower() if '.' in path_token else ''
    if extension in _TEXT_ARTIFACT_IDENTITY_EXTENSIONS:
        return True
    mime_types = {
        _clean_text(record.get('mime_type') or record.get('mime')).lower()
        for record in records
        if _clean_text(record.get('mime_type') or record.get('mime'))
    }
    return any(
        mime_type.startswith('text/')
        or mime_type in {
            'application/javascript',
            'application/json',
            'application/sql',
            'application/xml',
            'application/yaml',
        }
        for mime_type in mime_types
    )


def _collect_artifact_records(response_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ('artifacts', 'outputs', 'output'):
        for item in response_payload.get(key) or []:
            if isinstance(item, Mapping):
                records.append(dict(item))
    runtime = _nested_mapping(response_payload, 'runtime')
    for key in ('artifacts', 'outputs'):
        for item in runtime.get(key) or []:
            if isinstance(item, Mapping):
                records.append(dict(item))
    planning = _nested_mapping(response_payload, 'planning', 'artifact_flow')
    for item in planning.get('output_slots') or []:
        if isinstance(item, Mapping) and _artifact_ref(item):
            records.append(dict(item))
    return records


def _merge_alias_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch_prompts: list[str] = []
    branch_ids: list[str] = []
    phase_ids: list[str] = []
    output_slot_ids: list[str] = []
    for record in records:
        for key, values in (
            ('batch_prompt', batch_prompts),
            ('prompt', batch_prompts),
            ('branch_id', branch_ids),
            ('phase_id', phase_ids),
            ('output_slot_id', output_slot_ids),
            ('slot_id', output_slot_ids),
        ):
            _append_unique(values, _clean_text(record.get(key)))
    return _compact_payload(
        {
            'batch_prompts': batch_prompts,
            'branch_ids': branch_ids,
            'phase_ids': phase_ids,
            'output_slot_ids': output_slot_ids,
        }
    )


def canonicalize_duplicate_artifact_refs(artifacts: Sequence[Any]) -> dict[str, Any]:
    """Canonicalize proven duplicate artifact aliases without hiding conflicts."""

    groups: dict[str, list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for item in artifacts or []:
        if not isinstance(item, Mapping):
            continue
        record = dict(item)
        ref = _artifact_ref(record)
        if not ref:
            passthrough.append(record)
            continue
        groups.setdefault(ref, []).append(record)

    canonical_records: list[dict[str, Any]] = [*passthrough]
    duplicate_refs: list[str] = []
    conflicts: list[dict[str, Any]] = []
    canonicalization_required = False
    for ref, records in groups.items():
        if len(records) == 1:
            canonical_records.append(_json_safe(records[0]))
            continue
        duplicate_refs.append(ref)
        canonicalization_required = True
        paths = {_artifact_path(record) for record in records if _artifact_path(record)}
        raw_types = {_artifact_type(record) for record in records if _artifact_type(record)}
        identity_types = set(raw_types)
        if _document_text_identity_alias_is_proven(
            records,
            paths=paths,
            raw_types=raw_types,
        ):
            identity_types = {'text'}
        if len(paths) > 1 or len(identity_types) > 1:
            conflicts.append(
                {
                    'artifact_ref': ref,
                    'paths': sorted(paths),
                    'types': sorted(raw_types),
                    'reason': 'conflicting_duplicate_artifact_ref',
                }
            )
            canonical_records.extend(_json_safe(record) for record in records)
            continue
        canonical = copy.deepcopy(records[0])
        canonical['alias_artifact_refs'] = [ref]
        alias_metadata = _merge_alias_metadata(records)
        if alias_metadata:
            canonical['alias_metadata'] = alias_metadata
        canonical_records.append(_json_safe(canonical))

    return _json_safe(
        {
            'artifacts': canonical_records,
            'duplicate_refs': duplicate_refs,
            'canonicalization_required': canonicalization_required,
            'final_projection_blocked': bool(conflicts),
            'conflicts': conflicts,
        }
    )


def _intent_contract_digest(graph: Mapping[str, Any], response_payload: Mapping[str, Any]) -> str:
    runtime = _nested_mapping(response_payload, 'runtime')
    request_ir = (
        graph.get('request_ir')
        if isinstance(graph.get('request_ir'), Mapping)
        else runtime.get('request_ir')
        if isinstance(runtime.get('request_ir'), Mapping)
        else {}
    )
    payload = {
        'prompt_intent': graph.get('prompt_intent') or request_ir.get('prompt_intent'),
        'intent_obligations': graph.get('intent_obligations'),
        'output_obligations': graph.get('output_obligations'),
        'candidate_graph': graph.get('candidate_graph') or request_ir.get('candidate_graph'),
        'promotion_review': graph.get('promotion_review') or request_ir.get('promotion_review'),
    }
    return _stable_digest(payload, prefix='intent-contract-')


def summarize_intent_contract_for_scope(
    *,
    response_payload: Mapping[str, Any],
    request_phase_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize the current Intent Contract used by scope selection."""

    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    media_obligations = [
        item
        for item in _records(graph, 'intent_obligations')
        if _status(item.get('kind')) in {'media_artifact', 'image_artifact'}
        or _status(item.get('output_type')) == 'image'
        or _status(item.get('capability')) == 'image_generation'
    ]
    binding_obligations = [
        item
        for item in _records(graph, 'intent_obligations')
        if _status(item.get('dependency_contract')) or _status(item.get('kind')) == 'dependency'
    ]
    return _json_safe(
        {
            'intent_contract_digest': _intent_contract_digest(graph, response_payload),
            'required_intent_obligation_count': len(_records(graph, 'intent_obligations')),
            'required_output_obligation_count': len(_records(graph, 'output_obligations')),
            'media_obligation_count': len(media_obligations),
            'binding_obligation_count': len(binding_obligations),
            'has_local_visual_asset_requirement': bool(
                _nested_mapping(graph, 'prompt_intent').get('local_visual_asset_requirement')
                or any(_status(item.get('dependency_contract')) == 'local_visual_asset_binding' for item in binding_obligations)
            ),
        }
    )


def _closure_checks(closure_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for item in closure_review.get('checks') or []:
        if isinstance(item, Mapping):
            checks.append(dict(item))
    adequacy = (
        closure_review.get('intent_graph_adequacy')
        if isinstance(closure_review.get('intent_graph_adequacy'), Mapping)
        else {}
    )
    for item in adequacy.get('checks') or []:
        if isinstance(item, Mapping):
            checks.append(dict(item))
    return checks


def _current_runtime_evidence_present(
    closure_review: Mapping[str, Any],
    surface_state: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> bool:
    closure_status = _status(closure_review.get('status'))
    if closure_status in _REPAIR_STATUSES:
        return True
    if any(_status(item.get('status')) in _REPAIR_STATUSES for item in _closure_checks(closure_review)):
        return True
    if artifact_identity.get('duplicate_refs'):
        return True
    if _status(surface_state.get('state')) in {'blocked', 'repair_needed', 'repair_pending', 'review_pending'}:
        return True
    evidence = graph.get('redraw_scope_evidence')
    return isinstance(evidence, Mapping) and _status(evidence.get('status')) in _REPAIR_STATUSES


def _forbidden_evidence_seen(
    *,
    response_payload: Mapping[str, Any],
    closure_review: Mapping[str, Any],
    surface_state: Mapping[str, Any],
) -> list[str]:
    runtime = (
        dict(response_payload.get('runtime') or {})
        if isinstance(response_payload.get('runtime'), Mapping)
        else {}
    )
    runtime_health = {
        key: runtime.get(key)
        for key in (
            'backend_health',
            'health',
            'monitor',
            'provider_health',
            'route_health',
            'runtime_health',
            'warnings',
        )
        if runtime.get(key) not in (None, '', [], {})
    }
    closure_surface = {
        key: closure_review.get(key)
        for key in ('authority', 'evidence_refs', 'reason', 'source', 'status')
        if closure_review.get(key) not in (None, '', [], {})
    }
    closure_surface['checks'] = [
        {
            key: check.get(key)
            for key in (
                'authority',
                'check_kind',
                'evidence',
                'evidence_refs',
                'provider',
                'reason',
                'recovery_action',
                'repair_action',
                'route_health',
                'source',
                'status',
            )
            if check.get(key) not in (None, '', [], {})
        }
        for check in _closure_checks(closure_review)
    ]
    evidence_text = json.dumps(
        _json_safe(
            {
                'runtime_health': runtime_health,
                'closure_review': closure_surface,
                'surface_state': surface_state,
            }
        ),
        sort_keys=True,
    ).lower()
    return sorted(token for token in _FORBIDDEN_AUTHORITY_TOKENS if token in evidence_text)


def _learning_orientation(accepted_learning_hints: Any) -> dict[str, Any]:
    hints = [
        dict(item)
        for item in (accepted_learning_hints or [])
        if isinstance(item, Mapping)
    ]
    hint_refs: list[str] = []
    suggested_scope = ''
    for item in hints:
        _append_unique(hint_refs, _clean_text(item.get('hint_id') or item.get('id')))
        if not suggested_scope:
            suggested_scope = _status(item.get('suggested_scope') or item.get('scope'))
    return _json_safe(
        {
            'used': bool(hints),
            'hint_refs': hint_refs,
            'curation_role': 'existing_roles' if hints else 'none',
            'suggested_scope': suggested_scope,
            'authority': 'soft_hint_only',
            'forbidden_as_authority': True,
            'current_evidence_required': True,
            'used_as_authority': False,
        }
    )


def _active_graph_identities(graph: Mapping[str, Any]) -> dict[str, set[str]]:
    identities = {
        'phase_id': set(),
        'branch_id': set(),
        'obligation_id': set(),
    }
    for key in ('phases', 'downstream_branches', 'intent_obligations', 'output_obligations'):
        for record in _records(graph, key):
            for identity_key in identities:
                identity = _clean_text(record.get(identity_key))
                if identity:
                    identities[identity_key].add(identity)
    return identities


def _reserved_candidates(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_identities = _active_graph_identities(graph)
    candidates: list[dict[str, Any]] = []
    for container in (
        graph,
        graph.get('candidate_graph') if isinstance(graph.get('candidate_graph'), Mapping) else {},
        graph.get('request_ir') if isinstance(graph.get('request_ir'), Mapping) else {},
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ('candidates', 'output_candidates', 'candidate_outputs'):
            for item in container.get(key) or []:
                if not isinstance(item, Mapping):
                    continue
                status = _status(item.get('status') or item.get('contract_state') or item.get('promotion_status'))
                if status not in _RESERVED_STATUSES:
                    continue
                reason_text = ' '.join(
                    _status(item.get(key))
                    for key in ('reserved_reason', 'reason', 'defer_reason', 'reservation_reason')
                )
                if any(token in reason_text for token in _DEFER_TOKENS):
                    continue
                if any(
                    _clean_text(item.get(identity_key)) in active_identities[identity_key]
                    for identity_key in active_identities
                    if _clean_text(item.get(identity_key))
                ):
                    continue
                candidates.append(dict(item))
    return candidates


def _graph_phase_branch_ids(graph: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    phase_ids: set[str] = set()
    branch_ids: set[str] = set()
    for key in ('phases', 'downstream_branches'):
        for record in _records(graph, key):
            phase_id = _clean_text(record.get('phase_id'))
            if phase_id:
                phase_ids.add(phase_id)
            branch_id = _clean_text(record.get('branch_id'))
            if branch_id:
                branch_ids.add(branch_id)
    return phase_ids, branch_ids


def _has_missing_branch_evidence(closure_review: Mapping[str, Any], graph: Mapping[str, Any]) -> bool:
    existing_phase_ids, existing_branch_ids = _graph_phase_branch_ids(graph)
    for check in _closure_checks(closure_review):
        if _status(check.get('status')) not in _REPAIR_STATUSES:
            continue
        repair_action = _status(check.get('repair_action') or check.get('recovery_action'))
        evidence = _status(check.get('evidence'))
        reason = _status(check.get('reason'))
        check_kind = _status(check.get('check_kind'))
        if check_kind == 'intent_graph_adequacy' and evidence in {
            'intent_graph_adequacy_missing_output_obligation',
            'intent_graph_adequacy_missing_capability_obligation',
        }:
            missing_count = _nonnegative_integer(check.get('missing_count'))
            expected_count = _nonnegative_integer(check.get('expected_count'))
            actual_count = _nonnegative_integer(check.get('actual_count'))
            if (
                (missing_count is not None and missing_count > 0)
                or (
                    expected_count is not None
                    and actual_count is not None
                    and expected_count > actual_count
                )
            ):
                return True
        add_phases = [
            item for item in (check.get('add_phases') or []) if isinstance(item, Mapping)
        ]
        missing_additions = any(
            (
                bool(_clean_text(item.get('phase_id')))
                and _clean_text(item.get('phase_id')) not in existing_phase_ids
            )
            or (
                bool(_clean_text(item.get('branch_id')))
                and _clean_text(item.get('branch_id')) not in existing_branch_ids
            )
            for item in add_phases
        )
        if missing_additions:
            return True
        target_phase_id = _clean_text(check.get('phase_id'))
        target_branch_id = _clean_text(check.get('branch_id'))
        target_missing = bool(
            (target_phase_id and target_phase_id not in existing_phase_ids)
            or (target_branch_id and target_branch_id not in existing_branch_ids)
        )
        if (
            repair_action in {'add_missing_branch', 'repair_missing_materialization_contract'}
            and target_missing
        ):
            return True
        if 'missing' in (evidence or reason) and target_missing:
            return True
    required_output_phase_ids = {
        _clean_text(item.get('phase_id'))
        for item in _records(graph, 'output_obligations')
        if item.get('required') is not False and _clean_text(item.get('phase_id'))
    }
    return bool(required_output_phase_ids - existing_phase_ids)


def _has_binding_dependency_evidence(closure_review: Mapping[str, Any], graph: Mapping[str, Any]) -> bool:
    for check in _closure_checks(closure_review):
        if _status(check.get('status')) not in _REPAIR_STATUSES:
            continue
        repair_action = _status(check.get('repair_action') or check.get('recovery_action'))
        check_kind = _status(check.get('check_kind'))
        evidence = _status(check.get('evidence') or check.get('reason'))
        nested_recovery_actions = {
            _status(container.get(action_key))
            for container_key in ('recovery_context', 'recovery_state')
            for container in [check.get(container_key)]
            if isinstance(container, Mapping)
            for action_key in ('repair_action', 'recovery_action', 'suggested_action', 'retry_scope')
            if _status(container.get(action_key))
        }
        dependency_actions = {
            'dependency_chain',
            'rebind_artifact_dependency',
            'rebind_dependency_evidence',
            'repair_binding_dependency',
            'repair_dependency_chain',
        }
        if (
            repair_action in dependency_actions | {'repair_artifact_ref_identity'}
            or bool(nested_recovery_actions & dependency_actions)
            or check.get('blocked_by_dependency_input') is True
        ):
            return True
        signal_text = ' '.join([repair_action, check_kind, evidence, *sorted(nested_recovery_actions)])
        if (
            any(token in signal_text for token in ('dependency', 'binding', 'artifact_ref'))
            and any(
                token in signal_text
                for token in ('absent', 'broken', 'conflict', 'failed', 'invalid', 'missing', 'unmet', 'unresolved')
            )
        ):
            return True
    existing_phase_ids, existing_branch_ids = _graph_phase_branch_ids(graph)
    existing = existing_phase_ids | existing_branch_ids
    graph_records = [
        *_records(graph, 'phases'),
        *_records(graph, 'downstream_branches'),
    ]
    for obligation in _records(graph, 'intent_obligations'):
        if _status(obligation.get('dependency_contract')) != 'local_visual_asset_binding':
            continue
        target = _clean_text(obligation.get('target_phase_id'))
        sources = _clean_string_list(obligation.get('source_phase_ids'))
        existing_sources = [source for source in sources if source in existing]
        if target not in existing or not existing_sources:
            continue
        target_records = [
            record
            for record in graph_records
            if target in {
                _clean_text(record.get('phase_id')),
                _clean_text(record.get('branch_id')),
            }
        ]
        if not target_records:
            continue
        if any(
            any(
                source not in {
                    dependency
                    for key in ('depends_on', 'dependency_phase_ids', 'source_phase_ids', 'input_phase_ids')
                    for dependency in _clean_string_list(record.get(key))
                }
                for source in existing_sources
            )
            for record in target_records
        ):
            return True
    return False


def _explicit_scope_evidence(graph: Mapping[str, Any]) -> dict[str, Any]:
    evidence = graph.get('redraw_scope_evidence')
    return dict(evidence) if isinstance(evidence, Mapping) else {}


def _make_scope(
    scope: str,
    *,
    eligible: bool,
    reason: str,
    evidence_refs: Optional[Sequence[str]] = None,
    target_ids: Optional[Sequence[str]] = None,
    **extra: Any,
) -> dict[str, Any]:
    return _compact_payload(
        {
            'scope': scope,
            'eligible': eligible,
            'reason': reason,
            'evidence_refs': _clean_string_list(evidence_refs or []),
            'target_ids': _clean_string_list(target_ids or []),
            **extra,
        }
    )


def build_redraw_scope_ladder_review(
    *,
    response_payload: Mapping[str, Any],
    request_phase_graph: Mapping[str, Any],
    closure_review: Optional[Mapping[str, Any]] = None,
    surface_state: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Classify the smallest runtime-backed redraw/repair scope."""

    payload = response_payload if isinstance(response_payload, Mapping) else {}
    graph = request_phase_graph if isinstance(request_phase_graph, Mapping) else {}
    graph_digest_payload = copy.deepcopy(dict(graph))
    graph_digest_payload.pop('redraw_scope_ladder_review', None)
    closure = closure_review if isinstance(closure_review, Mapping) else {}
    surface = surface_state if isinstance(surface_state, Mapping) else {}
    if not surface and isinstance(closure.get('surface_state'), Mapping):
        surface = dict(closure.get('surface_state') or {})
    intent_summary = summarize_intent_contract_for_scope(
        response_payload=payload,
        request_phase_graph=graph,
    )
    canonical_artifacts = canonicalize_duplicate_artifact_refs(_collect_artifact_records(payload))
    artifact_identity = _compact_payload(
        {
            'duplicate_refs': canonical_artifacts.get('duplicate_refs') or [],
            'canonicalization_required': canonical_artifacts.get('canonicalization_required') or False,
            'final_projection_blocked': canonical_artifacts.get('final_projection_blocked') or False,
            'conflicts': canonical_artifacts.get('conflicts') or [],
        }
    )
    forbidden_seen = _forbidden_evidence_seen(
        response_payload=payload,
        closure_review=closure,
        surface_state=surface,
    )
    learning = _learning_orientation(accepted_learning_hints)
    current_evidence = _current_runtime_evidence_present(closure, surface, artifact_identity, graph)
    blocked_reasons: list[str] = []
    scopes: list[dict[str, Any]] = []

    explicit_scope = _explicit_scope_evidence(graph)
    explicit_recommended = _status(explicit_scope.get('recommended_scope'))
    explicit_refs = _clean_string_list(explicit_scope.get('evidence_refs'))
    reserved = _reserved_candidates(graph)
    reserved_ids = [
        _clean_text(item.get('candidate_id') or item.get('id') or item.get('phase_id') or item.get('branch_id'))
        for item in reserved
    ]
    reserved_ids = [item for item in reserved_ids if item]
    has_duplicate_refs = bool(artifact_identity.get('duplicate_refs'))
    has_missing_branch = _has_missing_branch_evidence(closure, graph)
    has_binding = _has_binding_dependency_evidence(closure, graph)

    scopes.append(
        _make_scope(
            'observe',
            eligible=not current_evidence,
            reason='wait_for_current_runtime_evidence' if not current_evidence else 'runtime_evidence_present',
        )
    )
    scopes.append(
        _make_scope(
            'promote_reserved_slot',
            eligible=bool(current_evidence and reserved_ids),
            reason='reserved_candidate_matches_current_intent_gap' if reserved_ids else 'no_reserved_candidate',
            evidence_refs=['candidate_graph:reserved', 'closure:intent_graph_adequacy'] if reserved_ids else [],
            target_ids=reserved_ids,
        )
    )
    scopes.append(
        _make_scope(
            'fill_reserved_slot',
            eligible=False,
            reason='reserved_slot_fill_not_needed_when_candidate_promotion_is_available'
            if reserved_ids
            else 'no_empty_reserved_slot_evidence',
        )
    )
    scopes.append(
        _make_scope(
            'add_missing_branch',
            eligible=bool(current_evidence and has_missing_branch and not reserved_ids),
            reason='missing_promoted_branch_or_obligation',
            evidence_refs=['intent_graph_adequacy'] if has_missing_branch else [],
            repair_type='repair_missing_materialization_contract',
        )
    )
    scopes.append(
        _make_scope(
            'repair_binding_dependency',
            eligible=bool(current_evidence and has_binding and not has_missing_branch and not reserved_ids),
            reason='existing_artifacts_or_branches_need_binding_dependency_repair',
            evidence_refs=['closure:binding_dependency'] if has_binding else [],
            repair_type='rebind_artifact_dependency',
        )
    )
    scopes.append(
        _make_scope(
            'repair_artifact_ref_identity',
            eligible=has_duplicate_refs,
            reason='duplicate_artifact_refs_require_identity_hygiene',
            evidence_refs=[f'artifact_identity:duplicate_ref:{item}' for item in artifact_identity.get('duplicate_refs') or []],
            repair_type='repair_artifact_ref_identity',
        )
    )
    scopes.append(
        _make_scope(
            'partial_subtree_rebase',
            eligible=bool(current_evidence and explicit_recommended == 'partial_subtree_rebase' and not forbidden_seen),
            reason=_clean_text(explicit_scope.get('reason')) or 'partial_subtree_rebase_requested_by_runtime_evidence',
            evidence_refs=explicit_refs,
            scope_root_ids=_clean_string_list(explicit_scope.get('scope_root_ids')),
            preserve_outside_scope=True,
            runtime_action='reviewed_graph_rebase_only',
        )
    )
    scopes.append(
        _make_scope(
            'full_successor_rebase',
            eligible=bool(current_evidence and explicit_recommended == 'full_successor_rebase' and not forbidden_seen),
            reason=_clean_text(explicit_scope.get('reason')) or 'full_successor_rebase_requested_by_runtime_evidence',
            evidence_refs=explicit_refs,
            runtime_action='reviewed_graph_rebase_only',
        )
    )
    scopes.append(
        _make_scope(
            'apply_enforced_reserved',
            eligible=False,
            reason='apply_enforced_reserved',
            evidence_refs=[],
        )
    )

    selected = next((item for item in scopes if item.get('scope') == 'observe'), scopes[0])
    for scope in SCOPE_LADDER[1:]:
        candidate = next((item for item in scopes if item.get('scope') == scope), {})
        if candidate.get('eligible'):
            selected = candidate
            break
    if has_duplicate_refs:
        selected = next(item for item in scopes if item.get('scope') == 'repair_artifact_ref_identity')
    if learning.get('used') and not current_evidence:
        _append_unique(blocked_reasons, 'current_runtime_evidence_required')
    if forbidden_seen and explicit_recommended in {'partial_subtree_rebase', 'full_successor_rebase'}:
        _append_unique(blocked_reasons, 'degraded_or_provider_signal_not_scope_authority')
        if selected.get('scope') in {'partial_subtree_rebase', 'full_successor_rebase'}:
            selected = next(item for item in scopes if item.get('scope') == 'observe')
    elif forbidden_seen and not current_evidence:
        _append_unique(blocked_reasons, 'degraded_or_provider_signal_not_scope_authority')
    elif forbidden_seen and selected.get('scope') in {'partial_subtree_rebase', 'full_successor_rebase'}:
        _append_unique(blocked_reasons, 'degraded_or_provider_signal_not_scope_authority')
        selected = next(item for item in scopes if item.get('scope') == 'observe')
    if artifact_identity.get('final_projection_blocked'):
        _append_unique(blocked_reasons, 'conflicting_duplicate_artifact_ref')

    selected_scope = _clean_text(selected.get('scope')) or 'observe'
    review_status = 'selected' if selected_scope != 'observe' and not (
        selected_scope in {'partial_subtree_rebase', 'full_successor_rebase'} and blocked_reasons
    ) else 'blocked' if blocked_reasons else 'none'
    review = {
        'kind': REDRAW_SCOPE_REVIEW_KIND,
        'review_id': _stable_digest(
            {
                'response_id': payload.get('response_id') or payload.get('id') or graph.get('response_id'),
                'intent_contract_digest': intent_summary.get('intent_contract_digest'),
                'selected_scope': selected_scope,
                'artifact_identity': artifact_identity,
            },
            prefix='redraw-scope-review-',
        ),
        'status': review_status,
        'intent_anchor': intent_summary,
        'intent_contract_digest': intent_summary.get('intent_contract_digest'),
        'base_graph_digest': _stable_digest(graph_digest_payload, prefix='graph-'),
        'scopes_considered': scopes,
        'selected_scope': selected_scope,
        'selected_scope_reason': selected.get('reason'),
        'selected_candidate': selected,
        'scope_floor': 'smallest_truthful_transition',
        'scope_ceiling': 'full_successor_rebase',
        'learning_orientation': learning,
        'artifact_identity': artifact_identity,
        'blocked_reasons': blocked_reasons,
        'forbidden_evidence_seen': forbidden_seen,
    }
    return _json_safe(review)


def select_redraw_scope_candidate(review: Mapping[str, Any]) -> dict[str, Any]:
    """Return the selected scope candidate from a ladder review."""

    if not isinstance(review, Mapping):
        return {}
    selected = review.get('selected_candidate')
    return dict(selected) if isinstance(selected, Mapping) else {}


def classify_scope_from_runtime_gap(
    *,
    response_payload: Mapping[str, Any],
    request_phase_graph: Mapping[str, Any],
    closure_review: Optional[Mapping[str, Any]] = None,
    surface_state: Optional[Mapping[str, Any]] = None,
    accepted_learning_hints: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that classify a runtime gap."""

    return build_redraw_scope_ladder_review(
        response_payload=response_payload,
        request_phase_graph=request_phase_graph,
        closure_review=closure_review,
        surface_state=surface_state,
        accepted_learning_hints=accepted_learning_hints,
    )


def collect_learning_orientation_for_scope(accepted_learning_hints: Any) -> dict[str, Any]:
    """Expose accepted-learning orientation in the scope-review shape."""

    return _learning_orientation(accepted_learning_hints)
