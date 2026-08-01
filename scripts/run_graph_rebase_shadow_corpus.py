#!/usr/bin/env python3
"""Plan, run, and inspect a resumable Ollmo graph-rebase shadow corpus.

The runner is deliberately conservative.  It talks only to Ollmo's passive
observer surfaces and canonical Responses endpoint, persists ``submitting``
before dispatch, and never repeats an ambiguous POST.  A resumed
``submitting`` or ``dispatch_unknown`` case is GET-only until Ollmo exposes the
client-chosen response id.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


DEFAULT_BASE_URL = (
    os.environ.get('OLLMO_WEB_BASE')
    or os.environ.get('OLLMO_BASE_URL')
    or 'http://127.0.0.1:5001'
)
DEFAULT_MANIFEST_ROOT = Path('state/graph_rebase_shadow_corpus')
MANIFEST_KIND = 'ollmo.graph_rebase_shadow_corpus_manifest'
MANIFEST_SCHEMA_VERSION = 1
CORPUS_SCHEMA_VERSION = 1
MAX_STATUS_OBSERVATIONS = 64
MAX_SUMMARY_RECORDS = 48
MAX_EVIDENCE_STRING_CHARS = 2_048
MAX_FINAL_TEXT_CHARS = 8_192
MAX_PREDECESSOR_ARTIFACTS = 16
MAX_PREDECESSOR_IDENTITY_CHARS = 512
MAX_PREDECESSOR_PATH_CHARS = 2_048
MAX_PREDECESSOR_CONTEXT_BYTES = 65_536

SMALLER_REBASE_PRECEDENCE_SCOPES = {
    'promote_reserved_slot',
    'fill_reserved_slot',
    'additive_repair',
    'add_missing_branch',
    'repair_binding_dependency',
    'repair_artifact_ref_identity',
}
STRUCTURAL_REBASE_ACTIONS = {
    'rebuild_from_promoted_obligations',
    'partial_subtree_rebase',
    'full_successor_rebase',
}
STRUCTURAL_REBASE_OPERATIONS = {
    'merge_branch',
    'merge_branches',
    'rebind_dependency',
    'remove_branch',
    'remove_dependency',
    'remove_obligation',
    'remove_phase',
    'split_branch',
    'supersede_with_replacement',
}
OPEN_REBASE_EVIDENCE_STATUSES = {
    'actionable',
    'active',
    'blocked',
    'failed',
    'pending',
    'repair_needed',
    'repair_required',
    'unmet',
}
ACTIVE_LATE_FILL_STATUSES = {
    'accepted',
    'active',
    'in_progress',
    'pending',
    'queued',
    'running',
    'scheduled',
    'started',
}
REPAIR_LATE_FILL_STATUSES = {
    'blocked',
    'failed',
    'late_fill_failed',
    'partial_failed',
    'repair_needed',
}
NON_SUCCESS_TERMINAL_LATE_FILL_STATUSES = {
    'canceled',
    'cancelled',
    'partial_cancelled',
    'superseded',
}
FORBIDDEN_OPPORTUNITY_CONTRACT_KEYS = {
    'authorization',
    'authority',
    'base_graph',
    'candidate_graph',
    'classification',
    'deterministic_fault',
    'expected_outcome',
    'expected_result',
    'expected_scope',
    'expected_status',
    'expected_classification',
    'expected_proposal',
    'fault',
    'fault_injection',
    'formal_rebase_proposal_expected',
    'ghost_messages',
    'graph_rebase_authorization',
    'graph_rebase_proposal',
    'operator_action',
    'outcome',
    'rebase_class',
    'redraw_scope_evidence',
    'repair_action',
    'requested_rebase_class',
    'runtime',
    'selected_scope',
}

ACTIVE_CASE_STATES = {
    'submitting',
    'submitted',
    'observing',
    'dispatch_unknown',
}
SETTLED_CASE_STATES = {
    'settled_terminal',
    'settled_repair_needed',
}
SUCCESSFUL_DEPENDENCY_LIFECYCLES = {
    'completed',
    'frozen',
    'late_fill_completed',
}
ALLOWED_DEPENDENCY_STATES = {
    'settled_terminal',
    'settled_repair_needed',
}
FINAL_CASE_STATES = {
    *SETTLED_CASE_STATES,
    'dependency_blocked',
}
GET_ONLY_CASE_STATES = {
    'submitting',
    'submitted',
    'observing',
    'dispatch_unknown',
}
PROTECTED_REQUEST_KEYS = {
    'backend',
    'conversationId',
    'conversation_id',
    'ghost_messages',
    'ghost_messages_json',
    'ghost_preferences',
    'ghost_route',
    'input',
    'input_artifacts',
    'instance_id',
    'model',
    'reference_artifacts',
    'request_meta',
    'responseId',
    'response_id',
    'selected_reference_artifact',
    'selected_reference_artifacts',
    'prompt',
    'workload_family',
}
EVIDENCE_BULK_KEYS = {
    'accepted_patch',
    'audio_data',
    'audio_data_url',
    'base_graph',
    'candidate_graph',
    'content',
    'content_payload',
    'image_data',
    'image_data_url',
    'messages',
    'output_text',
    'prompt',
    'request',
    'request_payload',
    'response_frame',
    'result_text',
    'source_proposal',
}
LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '::1'}
LOCAL_CONTROL_PLANE_PORT = 5001


class CorpusError(RuntimeError):
    """Raised when the corpus, manifest, or safe runner contract is invalid."""


class ManifestLockedError(CorpusError):
    """Raised when another process owns the run manifest."""


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Keep every corpus request on the validated local control plane."""

    def _reject(self, request, fp, code, message, headers):
        raise HTTPError(
            request.full_url,
            code,
            'Redirects are forbidden for shadow-corpus requests.',
            headers,
            fp,
        )

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    payload: Any
    byte_count: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return 200 <= int(self.status_code) < 300


class JsonHttpClient:
    """Small standard-library JSON client limited by runner-owned paths."""

    def __init__(self, base_url: str, *, default_timeout: float = 30.0):
        self.base_url = validate_base_url(base_url)
        self.default_timeout = max(0.1, float(default_timeout))
        self._opener = build_opener(ProxyHandler({}), _RejectRedirectHandler())

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResult:
        normalized_method = str(method or '').strip().upper()
        if normalized_method not in {'GET', 'POST'}:
            raise CorpusError(f'Unsupported HTTP method: {normalized_method or method!r}')
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        body = None
        headers = {'Accept': 'application/json'}
        if payload is not None:
            body = canonical_json_bytes(payload)
            headers['Content-Type'] = 'application/json'
        request = Request(url, data=body, headers=headers, method=normalized_method)
        try:
            with self._opener.open(
                request,
                timeout=timeout or self.default_timeout,
            ) as response:
                raw = response.read()
                return HttpResult(
                    status_code=int(response.status),
                    payload=decode_json_payload(raw),
                    byte_count=len(raw),
                )
        except HTTPError as exc:
            raw = exc.read()
            return HttpResult(
                status_code=int(exc.code),
                payload=decode_json_payload(raw),
                byte_count=len(raw),
                error=str(exc),
            )
        except (URLError, TimeoutError, OSError) as exc:
            return HttpResult(status_code=0, payload={}, error=str(exc))

    def get(self, path: str, *, timeout: Optional[float] = None) -> HttpResult:
        return self.request_json('GET', path, timeout=timeout)

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> HttpResult:
        return self.request_json('POST', path, payload=payload, timeout=timeout)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def decode_json_payload(raw: bytes) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {'error': 'non_json_response', 'body_bytes': len(raw)}


def validate_base_url(value: str) -> str:
    normalized = str(value or '').strip().rstrip('/')
    parsed = urlparse(normalized)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise CorpusError(f'Invalid Ollmo base URL: {value!r}')
    if str(parsed.hostname).lower() not in LOOPBACK_HOSTS:
        raise CorpusError('The shadow-corpus runner is restricted to the local Ollmo control plane.')
    try:
        port = parsed.port
    except ValueError as exc:
        raise CorpusError(f'Invalid Ollmo base URL port: {value!r}') from exc
    if (
        port != LOCAL_CONTROL_PLANE_PORT
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise CorpusError(
            'The shadow-corpus runner requires the exact loopback Ollmo control '
            f'plane on port {LOCAL_CONTROL_PLANE_PORT} without userinfo.'
        )
    if parsed.path not in {'', '/'} or parsed.params or parsed.query or parsed.fragment:
        raise CorpusError('The Ollmo base URL must not contain a path, query, or fragment.')
    return normalized


def clean_identifier(value: Any, *, field: str) -> str:
    token = str(value or '').strip()
    if not token:
        raise CorpusError(f"Corpus field '{field}' is required.")
    if not re.fullmatch(r'[A-Za-z0-9._-]+', token):
        raise CorpusError(
            f"Corpus field '{field}' may contain only letters, digits, '.', '_', and '-'."
        )
    return token


def slug(value: Any, *, maximum: int = 48) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value or '').strip()).strip('-._')
    token = token[:maximum].rstrip('-._')
    return token or 'case'


def normalize_string_list(value: Any, *, field: str) -> list[str]:
    if value in (None, ''):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise CorpusError(f"Corpus field '{field}' must be a string or list of strings.")
    result: list[str] = []
    for item in values:
        token = str(item or '').strip()
        if not token:
            continue
        if token not in result:
            result.append(token)
    return result


def _request_overrides(value: Any, *, field: str) -> dict[str, Any]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise CorpusError(f"Corpus field '{field}' must be an object.")
    result = dict(value)
    forbidden = sorted(PROTECTED_REQUEST_KEYS & set(result))
    if forbidden:
        raise CorpusError(
            f"Corpus field '{field}' cannot override runner-owned keys: {', '.join(forbidden)}."
        )
    return result


def _optional_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise CorpusError(f"Corpus field '{field}' must be an object.")
    return dict(value)


def _normalized_schema_key(value: Any) -> str:
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', str(value or '').strip())
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def _forbidden_nested_opportunity_keys(
    value: Any,
    *,
    path: str,
) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_schema_key(key)
            nested_path = f'{path}.{key}'
            if normalized in FORBIDDEN_OPPORTUNITY_CONTRACT_KEYS:
                findings.append(nested_path)
            findings.extend(
                _forbidden_nested_opportunity_keys(nested, path=nested_path)
            )
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            findings.extend(
                _forbidden_nested_opportunity_keys(
                    nested,
                    path=f'{path}[{index}]',
                )
            )
    return findings


def _opportunity_contract(value: Any, *, field: str) -> dict[str, Any]:
    contract = _optional_mapping(value, field=field)
    forbidden = sorted(
        set(_forbidden_nested_opportunity_keys(contract, path=field))
    )
    if forbidden:
        raise CorpusError(
            f"Corpus field '{field}' contains outcome labels or injected authority: "
            f"{', '.join(forbidden)}."
        )
    if not contract:
        return {}
    sequence_id = clean_identifier(
        contract.get('sequence_id'),
        field=f'{field}.sequence_id',
    )
    motif = clean_identifier(contract.get('motif'), field=f'{field}.motif')
    turn_role = str(contract.get('turn_role') or '').strip().lower()
    if turn_role not in {'root', 'follow_up'}:
        raise CorpusError(
            f"Corpus field '{field}.turn_role' must be 'root' or 'follow_up'."
        )
    try:
        turn_index = int(contract.get('turn_index'))
    except (TypeError, ValueError) as exc:
        raise CorpusError(
            f"Corpus field '{field}.turn_index' must be a positive integer."
        ) from exc
    if turn_index < 1:
        raise CorpusError(
            f"Corpus field '{field}.turn_index' must be a positive integer."
        )
    if (turn_role == 'root') != (turn_index == 1):
        raise CorpusError(
            f"Corpus field '{field}' must use root at turn 1 and follow_up afterward."
        )
    return {
        **contract,
        'sequence_id': sequence_id,
        'motif': motif,
        'turn_role': turn_role,
        'turn_index': turn_index,
    }


def load_corpus(path: Path | str) -> dict[str, Any]:
    corpus_path = Path(path)
    try:
        raw = json.loads(corpus_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise CorpusError(f'Corpus file not found: {corpus_path}') from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f'Corpus JSON is invalid: {exc}') from exc
    if not isinstance(raw, Mapping):
        raise CorpusError('Corpus root must be a JSON object.')
    schema_version = raw.get('schema_version', CORPUS_SCHEMA_VERSION)
    if schema_version != CORPUS_SCHEMA_VERSION:
        raise CorpusError(
            f'Unsupported corpus schema_version {schema_version!r}; expected {CORPUS_SCHEMA_VERSION}.'
        )
    corpus_id = clean_identifier(raw.get('corpus_id'), field='corpus_id')
    raw_cases = raw.get('cases')
    if not isinstance(raw_cases, list) or not raw_cases:
        raise CorpusError("Corpus field 'cases' must be a non-empty array.")
    request_defaults = _request_overrides(
        raw.get('request_defaults') or raw.get('default_request'),
        field='request_defaults',
    )
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise CorpusError(f'Corpus case at index {index} must be an object.')
        case_id = clean_identifier(raw_case.get('case_id') or raw_case.get('id'), field='case_id')
        if case_id in seen_ids:
            raise CorpusError(f'Duplicate corpus case_id: {case_id}')
        seen_ids.add(case_id)
        prompt = str(raw_case.get('prompt') or '').strip()
        if not prompt:
            raise CorpusError(f"Corpus case '{case_id}' has no prompt.")
        category = str(raw_case.get('category') or '').strip()
        workload_family = str(raw_case.get('workload_family') or category).strip()
        if not workload_family:
            raise CorpusError(
                f"Corpus case '{case_id}' needs category or workload_family for honest diversity accounting."
            )
        conversation_key = clean_identifier(
            raw_case.get('conversation_key') or case_id,
            field=f'{case_id}.conversation_key',
        )
        dependencies = normalize_string_list(
            raw_case.get('depends_on'),
            field=f'{case_id}.depends_on',
        )
        predecessor = str(raw_case.get('predecessor') or '').strip()
        if predecessor and predecessor not in dependencies:
            dependencies.append(predecessor)
        wave_raw = raw_case.get('wave', 1)
        wave = str(wave_raw).strip()
        if not wave:
            raise CorpusError(f"Corpus case '{case_id}' has an empty wave.")
        case_overrides = _request_overrides(
            raw_case.get('request_overrides') or raw_case.get('request'),
            field=f'{case_id}.request_overrides',
        )
        expected_evidence = _optional_mapping(
            raw_case.get('expected_evidence'),
            field=f'{case_id}.expected_evidence',
        )
        opportunity_contract = _opportunity_contract(
            raw_case.get('opportunity_contract'),
            field=f'{case_id}.opportunity_contract',
        )
        if opportunity_contract and (request_defaults or case_overrides):
            raise CorpusError(
                f"Opportunity case '{case_id}' cannot use request overrides; "
                'the unlabeled probe sends only runner-owned request fields.'
            )
        if opportunity_contract and expected_evidence:
            raise CorpusError(
                f"Opportunity case '{case_id}' cannot carry expected_evidence or outcome labels."
            )
        stop_signals = normalize_string_list(
            raw_case.get('stop_signals'),
            field=f'{case_id}.stop_signals',
        )
        allow_dependency_states = normalize_string_list(
            raw_case.get('allow_dependency_states'),
            field=f'{case_id}.allow_dependency_states',
        )
        invalid_dependency_states = sorted(
            set(allow_dependency_states) - ALLOWED_DEPENDENCY_STATES
        )
        if invalid_dependency_states:
            raise CorpusError(
                f"Corpus case '{case_id}' has unsupported allow_dependency_states: "
                f"{', '.join(invalid_dependency_states)}."
            )
        cases.append(
            {
                'case_id': case_id,
                'ordinal': index + 1,
                'wave': wave,
                'category': category or workload_family,
                'workload_family': workload_family,
                'conversation_key': conversation_key,
                'depends_on': dependencies,
                'expected_capability_families': normalize_string_list(
                    raw_case.get('expected_capability_families')
                    or raw_case.get('expected_capabilities'),
                    field=f'{case_id}.expected_capability_families',
                ),
                'prompt': prompt,
                'request_overrides': {**request_defaults, **case_overrides},
                'expected_evidence': expected_evidence,
                'opportunity_contract': opportunity_contract,
                'stop_signals': stop_signals,
                'allow_dependency_states': allow_dependency_states,
                'provenance': raw_case.get('provenance') or raw_case.get('source') or {},
                'metadata': dict(raw_case.get('metadata') or {})
                if isinstance(raw_case.get('metadata'), Mapping)
                else {},
            }
        )
    by_id = {case['case_id']: case for case in cases}
    for case in cases:
        for dependency in case['depends_on']:
            if dependency == case['case_id']:
                raise CorpusError(f"Corpus case '{case['case_id']}' depends on itself.")
            if dependency not in by_id:
                raise CorpusError(
                    f"Corpus case '{case['case_id']}' depends on unknown case '{dependency}'."
                )
    opportunity_sequences: dict[str, dict[int, dict[str, Any]]] = {}
    for case in cases:
        contract = case.get('opportunity_contract') or {}
        if not contract:
            continue
        sequence_id = str(contract.get('sequence_id'))
        turn_index = int(contract.get('turn_index') or 0)
        sequence = opportunity_sequences.setdefault(sequence_id, {})
        if turn_index in sequence:
            raise CorpusError(
                f"Opportunity sequence '{sequence_id}' repeats turn_index {turn_index}."
            )
        sequence[turn_index] = case
    for sequence_id, sequence in opportunity_sequences.items():
        if 1 not in sequence:
            raise CorpusError(f"Opportunity sequence '{sequence_id}' has no root turn.")
        for turn_index, case in sequence.items():
            if turn_index == 1:
                continue
            predecessor = sequence.get(turn_index - 1)
            if predecessor is None:
                raise CorpusError(
                    f"Opportunity sequence '{sequence_id}' skips turn {turn_index - 1}."
                )
            if predecessor['case_id'] not in case.get('depends_on', []):
                raise CorpusError(
                    f"Opportunity case '{case['case_id']}' must depend on immediate "
                    f"predecessor '{predecessor['case_id']}'."
                )
            if case.get('conversation_key') != predecessor.get('conversation_key'):
                raise CorpusError(
                    f"Opportunity sequence '{sequence_id}' must share one conversation_key."
                )
    _validate_acyclic_dependencies(cases)
    normalized = {
        'schema_version': CORPUS_SCHEMA_VERSION,
        'corpus_id': corpus_id,
        'title': str(raw.get('title') or corpus_id).strip(),
        'description': str(raw.get('description') or '').strip(),
        'provenance': raw.get('provenance') or {},
        'cases': cases,
    }
    normalized['corpus_digest'] = stable_digest(normalized)
    normalized['corpus_path'] = str(corpus_path.resolve())
    return normalized


def _validate_acyclic_dependencies(cases: Sequence[Mapping[str, Any]]) -> None:
    dependencies = {
        str(case['case_id']): [str(item) for item in case.get('depends_on') or []]
        for case in cases
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(case_id: str, chain: list[str]) -> None:
        if case_id in visited:
            return
        if case_id in visiting:
            cycle = ' -> '.join([*chain, case_id])
            raise CorpusError(f'Corpus dependency cycle: {cycle}')
        visiting.add(case_id)
        for dependency in dependencies.get(case_id, []):
            visit(dependency, [*chain, case_id])
        visiting.remove(case_id)
        visited.add(case_id)

    for case_id in dependencies:
        visit(case_id, [])


def deterministic_response_id(corpus: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    identity = {
        'corpus_digest': corpus.get('corpus_digest'),
        'case_id': case.get('case_id'),
        'prompt': case.get('prompt'),
        'request_overrides': case.get('request_overrides'),
    }
    suffix = stable_digest(identity)[:16]
    return (
        f"resp_corpus_{slug(corpus.get('corpus_id'), maximum=24)}_"
        f"{slug(case.get('case_id'), maximum=40)}_{suffix}"
    )


def deterministic_conversation_id(corpus: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    corpus_slug = slug(corpus.get('corpus_id'), maximum=32)
    conversation_slug = slug(case.get('conversation_key'), maximum=48)
    identity = stable_digest(
        {
            'corpus_digest': corpus.get('corpus_digest'),
            'conversation_key': case.get('conversation_key'),
        }
    )[:10]
    return f'ollmo-corpus-{corpus_slug}-{conversation_slug}-{identity}'


def build_manifest(corpus: Mapping[str, Any], manifest_path: Path | str) -> dict[str, Any]:
    now = utc_now()
    manifest_cases = []
    for case in corpus.get('cases') or []:
        manifest_cases.append(
            {
                **dict(case),
                'response_id': deterministic_response_id(corpus, case),
                'conversation_id': deterministic_conversation_id(corpus, case),
                'state': 'planned',
                'created_at': now,
                'updated_at': now,
                'status_observation_count': 0,
                'status_observations': [],
            }
        )
    return {
        'kind': MANIFEST_KIND,
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'manifest_path': str(Path(manifest_path).resolve()),
        'corpus_id': corpus.get('corpus_id'),
        'corpus_digest': corpus.get('corpus_digest'),
        'corpus_path': corpus.get('corpus_path'),
        'created_at': now,
        'updated_at': now,
        'cases': manifest_cases,
        'readiness_checkpoints': [],
        'run_history': [],
    }


def load_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise CorpusError(f'Manifest not found: {manifest_path}') from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f'Manifest JSON is invalid: {exc}') from exc
    if not isinstance(payload, dict) or payload.get('kind') != MANIFEST_KIND:
        raise CorpusError(f'Not an Ollmo shadow-corpus manifest: {manifest_path}')
    if payload.get('schema_version') != MANIFEST_SCHEMA_VERSION:
        raise CorpusError(
            f"Unsupported manifest schema_version {payload.get('schema_version')!r}."
        )
    return payload


def assert_manifest_matches_corpus(
    manifest: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> None:
    if manifest.get('corpus_id') != corpus.get('corpus_id'):
        raise CorpusError('Manifest corpus_id does not match the corpus file.')
    if manifest.get('corpus_digest') != corpus.get('corpus_digest'):
        raise CorpusError(
            'Corpus digest changed; refusing to resume response ids against modified prompts.'
        )
    expected = {
        str(case.get('case_id')): deterministic_response_id(corpus, case)
        for case in corpus.get('cases') or []
    }
    actual = {
        str(case.get('case_id')): str(case.get('response_id') or '')
        for case in manifest.get('cases') or []
        if isinstance(case, Mapping)
    }
    if expected != actual:
        raise CorpusError('Manifest case or deterministic response-id bindings differ from the corpus.')


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write JSON with file and parent-directory durability before returning."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    fd, temp_name = tempfile.mkstemp(
        prefix=f'.{target.name}.',
        suffix='.tmp',
        dir=str(target.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def manifest_lock(path: Path | str) -> Iterator[None]:
    lock_path = Path(f'{Path(path)}.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ManifestLockedError(f'Manifest is already owned by another runner: {path}') from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f'pid={os.getpid()} acquired_at={utc_now()}\n')
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def classify_compact_status(payload: Mapping[str, Any]) -> str:
    semantics = payload.get('status_semantics')
    semantics = semantics if isinstance(semantics, Mapping) else {}
    late_fill = payload.get('late_fill')
    late_fill = late_fill if isinstance(late_fill, Mapping) else {}
    open_count = _safe_count(late_fill.get('pending_count')) + _safe_count(
        late_fill.get('active_count')
    )
    failed_count = _safe_count(late_fill.get('failed_count'))
    late_fill_status = str(late_fill.get('status') or '').strip().lower()
    if late_fill_status in ACTIVE_LATE_FILL_STATUSES:
        return 'open'
    lifecycle = str(
        payload.get('lifecycle_state')
        or semantics.get('canonical_lifecycle_state')
        or ''
    ).strip().lower()
    if lifecycle in {
        'accepted',
        'active',
        'in_progress',
        'late_fill_pending',
        'late_fill_running',
        'pending',
        'queued',
        'running',
        'streaming',
    }:
        return 'open'
    if bool(semantics.get('has_open_continuation')) or open_count > 0:
        return 'open'
    if bool(semantics.get('has_actionable_repair')):
        return 'settled_repair_needed'
    if lifecycle in {
        'blocked',
        'late_fill_repair_needed',
        'rebuild_from_promoted_obligations',
        'repair_branch_contract',
        'repair_dependency_chain',
        'repair_needed',
    }:
        return 'settled_repair_needed'
    if late_fill_status in REPAIR_LATE_FILL_STATUSES or failed_count > 0:
        return 'settled_repair_needed'
    if late_fill_status in NON_SUCCESS_TERMINAL_LATE_FILL_STATUSES:
        return 'settled_terminal'
    if bool(semantics.get('is_terminal') or semantics.get('terminal')):
        return 'settled_terminal'
    if lifecycle in {
        'cancelled',
        'canceled',
        'completed',
        'failed',
        'frozen',
        'late_fill_completed',
        'late_fill_failed',
        'partial_cancelled',
        'skipped',
        'superseded',
        'waived',
    }:
        return 'settled_terminal'
    return 'unknown_nonterminal'


def compact_status_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    semantics = payload.get('status_semantics')
    late_fill = payload.get('late_fill')
    output_counts = payload.get('output_counts')
    surface_state = payload.get('surface_state')
    return {
        key: value
        for key, value in {
            'observed_at': utc_now(),
            'state_version': payload.get('state_version'),
            'status': payload.get('status'),
            'lifecycle_state': payload.get('lifecycle_state'),
            'status_semantics': dict(semantics) if isinstance(semantics, Mapping) else {},
            'frame_id': payload.get('frame_id'),
            'frame_sequence': payload.get('frame_sequence'),
            'late_fill': dict(late_fill) if isinstance(late_fill, Mapping) else {},
            'output_counts': dict(output_counts) if isinstance(output_counts, Mapping) else {},
            'surface_state': dict(surface_state) if isinstance(surface_state, Mapping) else {},
            'error': payload.get('error'),
        }.items()
        if value not in (None, '', [], {})
    }


def _bounded_evidence(value: Any, *, depth: int = 0) -> Any:
    """Bound nested diagnostic evidence while dropping prompt/media bodies."""

    if depth >= 6:
        return {'truncated': True, 'kind': type(value).__name__}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key in list(value)[:96]:
            key = str(raw_key)
            if key in EVIDENCE_BULK_KEYS or key.startswith('raw_'):
                continue
            item = value.get(raw_key)
            if item in (None, '', [], {}):
                continue
            result[key] = _bounded_evidence(item, depth=depth + 1)
        if len(value) > 96:
            result['_truncated_key_count'] = len(value) - 96
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = [_bounded_evidence(item, depth=depth + 1) for item in value[:MAX_SUMMARY_RECORDS]]
        if len(value) > MAX_SUMMARY_RECORDS:
            result.append({'_truncated_item_count': len(value) - MAX_SUMMARY_RECORDS})
        return result
    if isinstance(value, str) and len(value) > MAX_EVIDENCE_STRING_CHARS:
        return {
            'text': value[:MAX_EVIDENCE_STRING_CHARS],
            'length_chars': len(value),
            'sha256': hashlib.sha256(value.encode('utf-8')).hexdigest(),
            'truncated': True,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _bounded_final_text(payload: Mapping[str, Any]) -> dict[str, Any]:
    text = str(payload.get('output_text') or '')
    source = 'output_text'
    if not text:
        frame = payload.get('response_frame')
        frame = frame if isinstance(frame, Mapping) else {}
        output = frame.get('output') if isinstance(frame.get('output'), Mapping) else {}
        text = str(output.get('text') or '')
        source = 'response_frame.output.text'
    if not text:
        for item in payload.get('outputs') or []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get('type') or '').strip().lower() not in {'text', 'message', 'chat'}:
                continue
            candidate = str(item.get('text') or item.get('content') or '')
            if candidate:
                text = candidate
                source = 'outputs.text'
                break
    if not text:
        return {}
    return {
        'source': source,
        'text': text[:MAX_FINAL_TEXT_CHARS],
        'length_chars': len(text),
        'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'truncated': len(text) > MAX_FINAL_TEXT_CHARS,
    }


def _debug_message_identity(
    payload: Mapping[str, Any],
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract one exact assistant-message id without guessing among candidates."""

    current_state = (
        frame.get('current_state')
        if isinstance(frame.get('current_state'), Mapping)
        else {}
    )
    candidates: list[str] = []
    for raw_value in (payload.get('message_id'), current_state.get('message_id')):
        value = str(raw_value or '').strip()
        if value and value not in candidates:
            candidates.append(value)
    output = payload.get('output') if isinstance(payload.get('output'), list) else []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get('type') or '').strip().lower()
        if item_type not in {'assistant', 'assistant_message', 'message'}:
            continue
        value = str(item.get('id') or item.get('message_id') or '').strip()
        if value and value not in candidates:
            candidates.append(value)
    if len(candidates) == 1:
        return {'status': 'exact', 'message_id': candidates[0]}
    if len(candidates) > 1:
        return {'status': 'ambiguous', 'candidate_count': len(candidates)}
    return {'status': 'missing'}


def _bounded_records(
    values: Any,
    *,
    allowed_keys: Sequence[str],
    limit: int = MAX_SUMMARY_RECORDS,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    result: list[dict[str, Any]] = []
    for item in values[:limit]:
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                key: item.get(key)
                for key in allowed_keys
                if item.get(key) not in (None, '', [], {})
            }
        )
    return result


def _record_group_summary(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        return {'total': 0}
    statuses: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    ids: list[str] = []
    blocked_reasons: Counter[str] = Counter()
    for item in values:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get('status') or '').strip().lower()
        if status:
            statuses[status] += 1
        rebase_class = str(
            item.get('requested_rebase_class') or item.get('rebase_class') or ''
        ).strip().lower()
        if rebase_class:
            classes[rebase_class] += 1
        for key in ('proposal_id', 'review_id', 'rebase_id', 'idempotency_key', 'outcome_id'):
            identity = str(item.get(key) or '').strip()
            if identity:
                if identity not in ids and len(ids) < MAX_SUMMARY_RECORDS:
                    ids.append(identity)
                break
        for reason in item.get('blocked_reasons') or []:
            token = str(reason or '').strip()
            if token:
                blocked_reasons[token] += 1
    return {
        'total': len([item for item in values if isinstance(item, Mapping)]),
        'by_status': dict(sorted(statuses.items())),
        'by_class': dict(sorted(classes.items())),
        'blocked_reasons': dict(sorted(blocked_reasons.items())),
        'record_ids': ids,
    }


def _summarize_late_fill(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    branch_keys = (
        'branch_id',
        'slot_id',
        'phase_id',
        'capability',
        'expected_capability',
        'output_type',
        'artifact_type',
        'status',
        'lifecycle',
        'repair_action',
        'recovery_action',
        'progress_stage',
        'instance_id',
        'depends_on',
        'input_refs',
        'output_contract',
        'artifact_request',
        'execution_contract',
        'local_execution_contract',
        'requires_artifact',
        'repair_action_reason',
    )
    result: dict[str, Any] = {
        key: value.get(key)
        for key in (
            'status',
            'linked_artifact_rebind_status',
            'final_materialization_contract_status',
            'partial_failure',
            'stale_active_reconciled',
            'repair_action',
            'repair_actions',
            'recovery_state',
            'repair_loop',
            'open_checks',
        )
        if value.get(key) not in (None, '', [], {})
    }
    for key in ('repair_actions', 'recovery_state', 'repair_loop', 'open_checks'):
        if key in result:
            result[key] = _bounded_evidence(result[key])
    for key in (
        'pending_branches',
        'active_branches',
        'failed_branches',
        'completed_branches',
        'cancelled_branches',
        'branch_progress',
        'recovery_candidates',
    ):
        records = value.get(key) if isinstance(value.get(key), list) else []
        result[f'{key.removesuffix("_branches")}_count'] = len(records)
        if records:
            result[key] = [
                {
                    field: _bounded_evidence(item.get(field))
                    for field in branch_keys
                    if item.get(field) not in (None, '', [], {})
                }
                for item in records[:MAX_SUMMARY_RECORDS]
                if isinstance(item, Mapping)
            ]
    fill_results = value.get('fill_results') if isinstance(value.get('fill_results'), list) else []
    result['fill_result_count'] = len(fill_results)
    result['fill_results'] = [
        {
            key: _bounded_evidence(item.get(key))
            for key in (
                'branch_id',
                'phase_id',
                'capability',
                'status',
                'path',
                'saved_text_path',
                'saved_image_path',
                'saved_audio_path',
                'artifact_ref',
                'output_obligation_ref',
                'extension',
                'mime_type',
                'file_sha256',
                'file_size_bytes',
                'content_sha256',
                'content_length_chars',
                'content_source',
                'artifact_request',
                'execution_contract',
                'local_execution_contract',
                'repair_action',
                'recovery_action',
            )
            if item.get(key) not in (None, '', [], {})
        }
        for item in fill_results[:MAX_SUMMARY_RECORDS]
        if isinstance(item, Mapping)
    ]
    return result


def _summarize_intent_adequacy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = {
        key: value.get(key)
        for key in (
            'status',
            'decision',
            'reason',
            'adequate',
            'graph_adequate',
            'blocked_reasons',
            'missing_obligation_count',
            'unfulfilled_obligation_count',
            'missing_dependency_count',
            'orphan_count',
        )
        if value.get(key) not in (None, '', [], {})
    }
    for key in (
        'checks',
        'expected_output_counts',
        'graph_capability_counts',
        'graph_output_counts',
        'intent_lens_review',
        'intent_obligation_kinds',
    ):
        if value.get(key) not in (None, '', [], {}):
            result[key] = _bounded_evidence(value.get(key))
    return result


def _summarize_phase_records(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    allowed = (
        'kind',
        'phase_id',
        'branch_id',
        'task_id',
        'workload_task_id',
        'capability',
        'role',
        'status',
        'lifecycle',
        'resolution',
        'reason',
        'output_type',
        'obligation_id',
        'queue_index',
        'requires_artifact',
        'source',
        'input_refs',
        'depends_on',
        'downstream_phase_ids',
        'child_task_ids',
        'output_contract',
        'review_criteria',
        'execution_contract',
        'local_execution_contract',
        'artifact_request',
        'repair_action',
        'recovery_action',
    )
    return [
        {
            key: _bounded_evidence(item.get(key))
            for key in allowed
            if item.get(key) not in (None, '', [], {})
        }
        for item in values[:MAX_SUMMARY_RECORDS]
        if isinstance(item, Mapping)
    ]


def _summarize_obligations(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    allowed = (
        'kind',
        'obligation_id',
        'branch_id',
        'phase_id',
        'task_id',
        'capability',
        'role',
        'relationship',
        'required',
        'status',
        'count',
        'output_type',
        'target_name',
        'target_extension',
        'source',
        'input_refs',
        'depends_on',
        'depends_on_obligation_ids',
        'source_phase_ids',
        'target_phase_id',
        'execution_dependency_required',
        'dependency_contract',
        'evidence',
        'promotion_policy',
        'fulfillment_policy',
        'output_contract',
        'review_criteria',
    )
    return [
        {
            key: _bounded_evidence(item.get(key))
            for key in allowed
            if item.get(key) not in (None, '', [], {})
        }
        for item in values[:MAX_SUMMARY_RECORDS]
        if isinstance(item, Mapping)
    ]


def _summarize_workload_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = {
        key: _bounded_evidence(value.get(key))
        for key in (
            'kind',
            'graph_mode',
            'root_workload_id',
            'workload_graph_version',
            'intent_anchor',
            'task_ids',
            'leaf_task_ids',
            'visibility_summary',
            'proposal_review',
        )
        if value.get(key) not in (None, '', [], {})
    }
    tasks = value.get('tasks') if isinstance(value.get('tasks'), list) else []
    result['task_count'] = len(tasks)
    result['tasks'] = _summarize_phase_records(tasks)
    return result


def _summarize_request_phase_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    phases = graph.get('phases') if isinstance(graph.get('phases'), list) else []
    downstream = (
        graph.get('downstream_branches')
        if isinstance(graph.get('downstream_branches'), list)
        else []
    )
    intent_obligations = (
        graph.get('intent_obligations')
        if isinstance(graph.get('intent_obligations'), list)
        else []
    )
    output_obligations = (
        graph.get('output_obligations')
        if isinstance(graph.get('output_obligations'), list)
        else []
    )
    return {
        **{
            key: _bounded_evidence(graph.get(key))
            for key in (
                'kind',
                'mode',
                'graph_version',
                'is_multi_phase',
                'continuation_required',
                'current_phase_id',
                'current_phase_capability',
                'current_phase_resolution',
                'downstream_phase_ids',
                'downstream_branch_ids',
                'downstream_capabilities',
                'workload_task_ids',
                'prompt_intent',
                'workload_proposal_review',
                'promotion_review',
            )
            if graph.get(key) not in (None, '', [], {})
        },
        'phase_count': len(phases),
        'phases': _summarize_phase_records(phases),
        'downstream_branch_count': len(downstream),
        'downstream_branches': _summarize_phase_records(downstream),
        'intent_obligation_count': len(intent_obligations),
        'intent_obligations': _summarize_obligations(intent_obligations),
        'output_obligation_count': len(output_obligations),
        'output_obligations': _summarize_obligations(output_obligations),
        'workload_graph': _summarize_workload_graph(graph.get('workload_graph')),
    }


def summarize_debug_payload(payload: Mapping[str, Any], *, byte_count: int = 0) -> dict[str, Any]:
    """Return bounded runtime truth without copying model/artifact bodies."""

    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    graph = (
        runtime.get('request_phase_graph')
        if isinstance(runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    diagnostics = (
        runtime.get('developer_diagnostics')
        if isinstance(runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), Mapping) else {}
    closure = (
        runtime.get('graph_closure_review')
        if isinstance(runtime.get('graph_closure_review'), Mapping)
        else payload.get('graph_closure_review')
        if isinstance(payload.get('graph_closure_review'), Mapping)
        else {}
    )
    scope_review = (
        graph.get('redraw_scope_ladder_review')
        if isinstance(graph.get('redraw_scope_ladder_review'), Mapping)
        else {}
    )
    output_keys = (
        'slot_id',
        'output_id',
        'type',
        'kind',
        'status',
        'lifecycle',
        'artifact_ref',
        'path',
        'source',
        'capability',
        'branch_id',
        'phase_id',
        'parent_slot_id',
        'parent_branch_id',
        'parent_phase_id',
        'follow_up_capability',
        'follow_up_branch_id',
        'follow_up_phase_id',
        'obligation_id',
    )
    artifact_keys = (
        'artifact_id',
        'artifact_ref',
        'type',
        'kind',
        'status',
        'path',
        'source',
        'capability',
        'sha256',
        'file_sha256',
        'size_bytes',
        'file_size_bytes',
        'mime',
        'mime_type',
        'extension',
        'branch_id',
        'phase_id',
        'obligation_id',
        'origin',
        'ref',
        'name',
    )
    graph_groups = {
        key: _record_group_summary(graph.get(key))
        for key in (
            'graph_rebase_proposals',
            'graph_rebase_reviews',
            'graph_rebase_lifecycle',
            'staged_graph_rebases',
            'successor_rebase_requests',
            'applied_graph_rebases',
            'graph_rebase_outcomes',
            'partial_rebase_outcomes',
            'successor_rebase_executions',
            'graph_repair_proposals',
            'graph_repair_reviews',
            'graph_patch_lifecycle',
            'successor_reopen_requests',
        )
    }
    diagnostic_groups = {
        key: _record_group_summary(diagnostics.get(key))
        for key in (
            'runtime_graph_rebase_proposals',
            'runtime_graph_rebase_reviews',
            'graph_rebase_lifecycle',
            'staged_graph_rebases',
            'successor_rebase_requests',
            'applied_graph_rebases',
            'graph_rebase_outcomes',
            'partial_rebase_outcomes',
            'successor_rebase_executions',
            'runtime_graph_repair_proposals',
            'runtime_graph_repair_proposal_reviews',
            'graph_patch_lifecycle',
            'graph_patch_successor_reopen_requests',
        )
    }
    candidate_review = diagnostics.get('runtime_graph_rebase_candidate_review')
    candidate_context = diagnostics.get('response_time_graph_rebase_candidate')
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), Mapping) else {}
    message_identity = _debug_message_identity(payload, frame)
    global_semantic = (
        closure.get('global_semantic_closure_review')
        if isinstance(closure.get('global_semantic_closure_review'), Mapping)
        else {}
    )
    closure_surface = (
        closure.get('surface_state')
        if isinstance(closure.get('surface_state'), Mapping)
        else payload.get('surface_state')
        if isinstance(payload.get('surface_state'), Mapping)
        else {}
    )
    return {
        'response_bytes': int(byte_count or 0),
        'id': payload.get('id') or payload.get('response_id'),
        'status': payload.get('status'),
        'lifecycle_state': payload.get('lifecycle_state'),
        'status_semantics': dict(payload.get('status_semantics') or {})
        if isinstance(payload.get('status_semantics'), Mapping)
        else {},
        'target': {
            key: payload.get(key)
            for key in ('instance_id', 'model', 'backend', 'capability', 'mode')
            if payload.get(key) not in (None, '', [], {})
        },
        'message_identity': message_identity,
        'message_id': message_identity.get('message_id'),
        'response_frame': {
            key: frame.get(key)
            for key in ('frame_id', 'frame_sequence', 'status', 'lifecycle_state')
            if frame.get(key) not in (None, '', [], {})
        }
        | (
            {
                'current_state': {
                    key: current_state.get(key)
                    for key in ('status', 'lifecycle_state', 'canonical_status_field')
                    if current_state.get(key) not in (None, '', [], {})
                }
            }
            if current_state
            else {}
        ),
        'final_text': _bounded_final_text(payload),
        'output_count': len(payload.get('outputs') or [])
        if isinstance(payload.get('outputs'), list)
        else 0,
        'artifact_count': len(payload.get('artifacts') or [])
        if isinstance(payload.get('artifacts'), list)
        else 0,
        'outputs': _bounded_records(payload.get('outputs'), allowed_keys=output_keys),
        'output_branches': _bounded_records(
            payload.get('output_branches'),
            allowed_keys=(
                'branch_id',
                'phase_id',
                'slot_id',
                'parent_slot_id',
                'child_slot_ids',
                'obligation_id',
                'type',
                'status',
                'lifecycle',
            ),
        ),
        'artifacts': _bounded_records(payload.get('artifacts'), allowed_keys=artifact_keys),
        'late_fill': _summarize_late_fill(payload.get('late_fill')),
        'closure': {
            key: (
                _summarize_intent_adequacy(closure.get(key))
                if key == 'intent_graph_adequacy'
                else _bounded_evidence(closure.get(key))
            )
            for key in (
                'kind',
                'status',
                'decision',
                'reason',
                'continuation_required',
                'closure_status',
                'closure_gap_code',
                'closure_gap_trigger',
                'materialization_status',
                'blocked_reasons',
                'open_obligation_count',
                'fulfilled_obligation_count',
                'obligation_count',
                'pending_branch_count',
                'semantic_review_required_count',
                'semantic_review_status',
                'late_fill_status',
                'repair_action',
                'recovery_action',
                'recommended_transition',
                'decision_action',
                'semantic_review_recommended_transition',
                'counts',
                'checks',
                'repair_needed',
                'intent_graph_adequacy',
            )
            if closure.get(key) not in (None, '', [], {})
        }
        | {
            'global_semantic_closure_review': _bounded_evidence(global_semantic),
            'surface_state': _bounded_evidence(closure_surface),
        },
        'request_phase_graph': _summarize_request_phase_graph(graph),
        'redraw_scope': {
            key: _bounded_evidence(scope_review.get(key))
            for key in (
                'review_id',
                'status',
                'selected_scope',
                'selected_scope_reason',
                'smaller_scope',
                'selected_candidate',
                'scopes_considered',
                'artifact_identity',
                'scope_floor',
                'scope_ceiling',
                'base_graph_digest',
                'intent_contract_digest',
                'blocked_reasons',
            )
            if scope_review.get(key) not in (None, '', [], {})
        },
        'graph_evidence': graph_groups,
        'graph_records': {
            key: _bounded_evidence(graph.get(key))
            for key in (
                'graph_rebase_proposals',
                'graph_rebase_reviews',
                'graph_rebase_lifecycle',
                'staged_graph_rebases',
                'successor_rebase_requests',
                'graph_rebase_outcomes',
                'partial_rebase_outcomes',
                'successor_rebase_executions',
                'graph_repair_proposals',
                'graph_repair_reviews',
                'graph_patch_lifecycle',
                'successor_reopen_requests',
            )
            if graph.get(key) not in (None, '', [], {})
        },
        'diagnostic_evidence': diagnostic_groups,
        'diagnostic_records': {
            key: _bounded_evidence(diagnostics.get(key))
            for key in (
                'runtime_graph_rebase_proposals',
                'runtime_graph_rebase_reviews',
                'graph_rebase_lifecycle',
                'staged_graph_rebases',
                'successor_rebase_requests',
                'graph_rebase_outcomes',
                'partial_rebase_outcomes',
                'successor_rebase_executions',
                'runtime_graph_repair_proposals',
                'runtime_graph_repair_proposal_reviews',
                'graph_patch_lifecycle',
                'graph_patch_successor_reopen_requests',
            )
            if diagnostics.get(key) not in (None, '', [], {})
        },
        'rebase_candidate': {
            'review': {
                key: _bounded_evidence(candidate_review.get(key))
                for key in (
                    'kind',
                    'status',
                    'reason',
                    'proposal_id',
                    'requested_rebase_class',
                    'selected_scope',
                    'smaller_scope',
                    'base_graph_digest',
                    'candidate_graph_digest',
                    'candidate_origin',
                    'runtime_effect',
                    'diff_summary',
                    'blocked_reasons',
                )
                if isinstance(candidate_review, Mapping)
                and candidate_review.get(key) not in (None, '', [], {})
            },
            'context': {
                key: _bounded_evidence(candidate_context.get(key))
                for key in (
                    'kind',
                    'status',
                    'reason',
                    'requested_rebase_class',
                    'selected_scope',
                    'base_graph_digest',
                    'candidate_graph_digest',
                    'runtime_effect',
                    'diff_summary',
                )
                if isinstance(candidate_context, Mapping)
                and candidate_context.get(key) not in (None, '', [], {})
            },
        },
        'repair_actionability': {
            key: diagnostics.get('surface_repair_actionability', {}).get(key)
            for key in (
                'status',
                'actionable',
                'actionable_count',
                'advisory_count',
                'blocked_reasons',
                'actionable_classes',
                'advisory_classes',
            )
            if isinstance(diagnostics.get('surface_repair_actionability'), Mapping)
            and diagnostics.get('surface_repair_actionability', {}).get(key)
            not in (None, '', [], {})
        },
        'autonomy': {
            key: dict(diagnostics.get(key) or {})
            for key in ('graph_patch_autonomy', 'graph_rebase_autonomy')
            if isinstance(diagnostics.get(key), Mapping)
        },
    }


def derive_rebase_opportunity_summary(
    debug_summary: Mapping[str, Any],
    *,
    opportunity_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify inspection eligibility without creating rebase authority."""

    summary = debug_summary if isinstance(debug_summary, Mapping) else {}
    contract = (
        dict(opportunity_contract)
        if isinstance(opportunity_contract, Mapping)
        else {}
    )
    closure = summary.get('closure') if isinstance(summary.get('closure'), Mapping) else {}
    adequacy = (
        closure.get('intent_graph_adequacy')
        if isinstance(closure.get('intent_graph_adequacy'), Mapping)
        else {}
    )
    closure_status = str(closure.get('status') or '').strip().lower()
    closure_records: list[Mapping[str, Any]] = [closure]
    closure_records.extend(
        item for item in closure.get('checks') or [] if isinstance(item, Mapping)
    )
    closure_records.extend(
        item for item in adequacy.get('checks') or [] if isinstance(item, Mapping)
    )
    structural_actions: list[str] = []
    for record in closure_records:
        record_status = str(record.get('status') or closure_status).strip().lower()
        if record_status not in OPEN_REBASE_EVIDENCE_STATUSES:
            continue
        for key in (
            'repair_action',
            'recovery_action',
            'recommended_transition',
            'decision_action',
            'semantic_review_recommended_transition',
        ):
            action = str(record.get(key) or '').strip().lower()
            if action in STRUCTURAL_REBASE_ACTIONS and action not in structural_actions:
                structural_actions.append(action)

    candidate_root = (
        summary.get('rebase_candidate')
        if isinstance(summary.get('rebase_candidate'), Mapping)
        else {}
    )
    candidate = (
        candidate_root.get('review')
        if isinstance(candidate_root.get('review'), Mapping)
        else {}
    )
    diff = candidate.get('diff_summary') if isinstance(candidate.get('diff_summary'), Mapping) else {}
    meaningful_change_count = _safe_count(diff.get('meaningful_change_count'))
    operation_counts = (
        diff.get('operation_counts')
        if isinstance(diff.get('operation_counts'), Mapping)
        else {}
    )
    structural_change_count = sum(
        _safe_count(count)
        for operation, count in operation_counts.items()
        if str(operation).strip().lower() in STRUCTURAL_REBASE_OPERATIONS
        or str(operation).strip().lower().startswith('change_')
    )
    if (
        _safe_count(operation_counts.get('add_phase')) > 0
        and (
            _safe_count(operation_counts.get('add_branch')) > 0
            or _safe_count(operation_counts.get('add_obligation')) > 0
        )
    ):
        structural_change_count += _safe_count(operation_counts.get('add_phase'))
    removed_ids = diff.get('removed_ids') if isinstance(diff.get('removed_ids'), Mapping) else {}
    if any(value for value in removed_ids.values()):
        structural_change_count += 1

    redraw = summary.get('redraw_scope') if isinstance(summary.get('redraw_scope'), Mapping) else {}
    selected_scope = str(redraw.get('selected_scope') or '').strip().lower()
    smaller_scope = str(candidate.get('smaller_scope') or '').strip().lower()
    if not smaller_scope and selected_scope in SMALLER_REBASE_PRECEDENCE_SCOPES:
        smaller_scope = selected_scope

    graph_evidence = (
        summary.get('graph_evidence')
        if isinstance(summary.get('graph_evidence'), Mapping)
        else {}
    )
    diagnostic_evidence = (
        summary.get('diagnostic_evidence')
        if isinstance(summary.get('diagnostic_evidence'), Mapping)
        else {}
    )
    proposal_count = max(
        _safe_count((graph_evidence.get('graph_rebase_proposals') or {}).get('total'))
        if isinstance(graph_evidence.get('graph_rebase_proposals'), Mapping)
        else 0,
        _safe_count((diagnostic_evidence.get('runtime_graph_rebase_proposals') or {}).get('total'))
        if isinstance(diagnostic_evidence.get('runtime_graph_rebase_proposals'), Mapping)
        else 0,
    )
    late_fill = summary.get('late_fill') if isinstance(summary.get('late_fill'), Mapping) else {}
    late_fill_status = str(late_fill.get('status') or '').strip().lower()
    active_late_fill = bool(
        late_fill_status in ACTIVE_LATE_FILL_STATUSES
        or _safe_count(late_fill.get('active_count')) > 0
        or _safe_count(late_fill.get('pending_count')) > 0
    )

    blockers: list[str] = []
    disposition = 'insufficient_observation'
    false_negative_review = 'not_applicable'
    eligible_for_operator_inspection = False
    if proposal_count > 0:
        disposition = 'formal_proposal_present'
        eligible_for_operator_inspection = True
    elif not candidate:
        blockers.append('runtime_candidate_observation_missing')
    elif active_late_fill:
        disposition = 'active_late_fill'
        blockers.append('active_late_fill_must_settle')
    elif smaller_scope:
        disposition = 'smaller_scope_precedes_rebase'
        blockers.append(f'smaller_scope_selected:{smaller_scope}')
    elif not structural_actions:
        disposition = 'no_current_structural_closure_evidence'
        blockers.append('no_current_structural_closure_action')
    elif meaningful_change_count < 1:
        disposition = 'candidate_has_no_meaningful_change'
        blockers.append('candidate_graph_has_no_meaningful_change')
    elif structural_change_count < 1:
        disposition = 'candidate_change_is_not_structural'
        blockers.append('candidate_change_is_additive_or_non_structural')
    else:
        disposition = 'unproposed_structural_opportunity'
        false_negative_review = 'candidate_requires_operator_judgment'
        eligible_for_operator_inspection = True

    return {
        'kind': 'ollmo.graph_rebase_opportunity_summary',
        'disposition': disposition,
        'eligible_for_operator_inspection': eligible_for_operator_inspection,
        'false_negative_review': false_negative_review,
        'authority': 'diagnostic_only_operator_judgment_required',
        'runtime_effect': 'none',
        **{
            key: contract.get(key)
            for key in ('sequence_id', 'motif', 'turn_index', 'turn_role')
            if contract.get(key) not in (None, '')
        },
        'closure_status': closure_status or None,
        'structural_closure_actions': structural_actions,
        'selected_scope': selected_scope or None,
        'candidate_status': candidate.get('status'),
        'candidate_reason': candidate.get('reason'),
        'requested_rebase_class': candidate.get('requested_rebase_class'),
        'meaningful_change_count': meaningful_change_count,
        'structural_change_count': structural_change_count,
        'formal_proposal_count': proposal_count,
        'blockers': blockers,
    }


def summarize_readiness(payload: Mapping[str, Any], *, byte_count: int = 0) -> dict[str, Any]:
    corpus = payload.get('corpus') if isinstance(payload.get('corpus'), Mapping) else {}
    candidate_root = (
        payload.get('candidate_opportunities')
        if isinstance(payload.get('candidate_opportunities'), Mapping)
        else {}
    )
    settled_candidates = (
        candidate_root.get('settled_final')
        if isinstance(candidate_root.get('settled_final'), Mapping)
        else {}
    )
    formal = payload.get('formal_evidence') if isinstance(payload.get('formal_evidence'), Mapping) else {}
    qualifying = (
        payload.get('qualifying_evidence')
        if isinstance(payload.get('qualifying_evidence'), Mapping)
        else {}
    )
    safety = payload.get('safety') if isinstance(payload.get('safety'), Mapping) else {}
    observer = payload.get('observer') if isinstance(payload.get('observer'), Mapping) else {}
    gate_summary: dict[str, Any] = {}
    for name, gate in (payload.get('gates') or {}).items():
        if not isinstance(gate, Mapping):
            continue
        gate_summary[str(name)] = {
            'ready': gate.get('ready'),
            'decision': gate.get('decision'),
            'unmet_requirements': list(gate.get('unmet_requirements') or []),
        }
    return {
        'response_bytes': int(byte_count or 0),
        'report_digest': payload.get('report_digest'),
        'corpus_digest': corpus.get('corpus_digest'),
        'settled_final_response_count': corpus.get('settled_final_response_count'),
        'nonterminal_active_late_fill_response_count': corpus.get(
            'nonterminal_active_late_fill_response_count'
        ),
        'unique_workload_family_count': corpus.get('unique_workload_family_count'),
        'workload_family_counts': dict(corpus.get('workload_family_counts') or {}),
        'settled_candidates': {
            key: settled_candidates.get(key)
            for key in (
                'total',
                'not_proposed_count',
                'with_formal_proposal_count',
                'by_status',
                'by_reason',
                'by_smaller_scope',
                'by_selected_scope',
                'by_class',
            )
            if settled_candidates.get(key) not in (None, '', [], {})
        },
        'formal_proposals': dict(formal.get('proposals') or {})
        if isinstance(formal.get('proposals'), Mapping)
        else {},
        'qualifying_evidence': {
            key: qualifying.get(key)
            for key in (
                'qualifying_proposal_count',
                'partial_proposal_count',
                'full_proposal_count',
                'passed_preservation_proof_count',
                'partial_stage_count',
                'partial_useful_adjudication_count',
                'partial_replay_confirmation_count',
                'partial_local_execution_contract_proof_count',
            )
            if qualifying.get(key) not in (None, '', [], {})
        },
        'safety': {
            key: safety.get(key)
            for key in (
                'unresolved_critical_finding_count',
                'unresolved_partial_or_unknown_finding_count',
                'zero_tolerance_satisfied',
                'unresolved_by_category',
            )
            if safety.get(key) not in (None, '', [], {})
        },
        'gates': gate_summary,
        'observer': {
            key: observer.get(key)
            for key in (
                'hydrated_response_count',
                'selected_graph_rebase_observation_count',
                'trusted_operator_record_count',
                'load_error_count',
                'selection_scan_error_count',
            )
            if observer.get(key) not in (None, '', [], {})
        },
    }


def manifest_case_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get('case_id')): case
        for case in manifest.get('cases') or []
        if isinstance(case, dict) and str(case.get('case_id') or '').strip()
    }


def normalize_case_filter(values: Any) -> list[str]:
    if values in (None, '', []):
        return []
    raw_values = values if isinstance(values, list) else [values]
    result: list[str] = []
    for raw in raw_values:
        nested = raw if isinstance(raw, list) else [raw]
        for item in nested:
            for token in str(item or '').split(','):
                case_id = token.strip()
                if case_id and case_id not in result:
                    result.append(case_id)
    return result


def selected_manifest_cases(
    manifest: Mapping[str, Any],
    wave: Optional[str],
    case_ids: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    normalized_wave = str(wave).strip() if wave not in (None, '') else None
    selected_ids = set(normalize_case_filter(list(case_ids or [])))
    known_ids = set(manifest_case_map(manifest))
    unknown = sorted(selected_ids - known_ids)
    if unknown:
        raise CorpusError(f"Unknown --case-id value(s): {', '.join(unknown)}")
    return [
        case
        for case in manifest.get('cases') or []
        if isinstance(case, dict)
        and (normalized_wave is None or str(case.get('wave')) == normalized_wave)
        and (not selected_ids or str(case.get('case_id')) in selected_ids)
    ]


def dependency_state(
    case: Mapping[str, Any],
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str]]:
    waiting: list[str] = []
    blocked: list[str] = []
    for dependency_id in case.get('depends_on') or []:
        dependency = cases_by_id.get(str(dependency_id)) or {}
        state = str(dependency.get('state') or 'missing')
        allowed_dependency_states = set(case.get('allow_dependency_states') or [])
        late_fill_status = str(
            dependency.get('last_late_fill_status') or ''
        ).strip().lower()
        dependency_success = (
            late_fill_status
            not in REPAIR_LATE_FILL_STATUSES
            | NON_SUCCESS_TERMINAL_LATE_FILL_STATUSES
            and (
                dependency.get('dependency_satisfied') is True
                or (
                    dependency.get('dependency_satisfied') is None
                    and str(
                        dependency.get('last_lifecycle_state') or ''
                    ).strip().lower()
                    in SUCCESSFUL_DEPENDENCY_LIFECYCLES
                )
            )
        )
        if state == 'settled_terminal' and dependency_success:
            continue
        if state in allowed_dependency_states:
            continue
        if state in FINAL_CASE_STATES:
            blocked.append(str(dependency_id))
        else:
            waiting.append(str(dependency_id))
    if blocked:
        return 'blocked', blocked
    if waiting:
        return 'waiting', waiting
    return 'ready', []


def _required_bounded_string(
    value: Any,
    *,
    field: str,
    maximum: int = MAX_PREDECESSOR_IDENTITY_CHARS,
) -> str:
    if not isinstance(value, str):
        raise CorpusError(f"Predecessor context field '{field}' must be a string.")
    token = value.strip()
    if not token:
        raise CorpusError(f"Predecessor context field '{field}' is required.")
    if len(token) > maximum:
        raise CorpusError(
            f"Predecessor context field '{field}' exceeds its {maximum}-character bound."
        )
    return token


def _strict_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CorpusError(
            f"Predecessor context field '{field}' must be a positive integer."
        )
    return value


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusError(
            f"Predecessor context field '{field}' must be a nonnegative integer."
        )
    return value


def _exact_sha256(value: Any, *, field: str) -> str:
    token = _required_bounded_string(value, field=field, maximum=64).lower()
    if not re.fullmatch(r'[0-9a-f]{64}', token):
        raise CorpusError(
            f"Predecessor context field '{field}' must be an exact SHA-256 digest."
        )
    return token


def _immediate_opportunity_predecessor(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    contract = case.get('opportunity_contract')
    if not isinstance(contract, Mapping) or not contract:
        return None
    turn_index = _strict_positive_int(
        contract.get('turn_index'),
        field=f"{case.get('case_id')}.opportunity_contract.turn_index",
    )
    turn_role = str(contract.get('turn_role') or '').strip().lower()
    if turn_index == 1:
        if turn_role != 'root':
            raise CorpusError('Opportunity turn 1 must remain a root turn.')
        return None
    if turn_role != 'follow_up':
        raise CorpusError('Opportunity turns after turn 1 must remain follow_up turns.')
    sequence_id = _required_bounded_string(
        contract.get('sequence_id'),
        field=f"{case.get('case_id')}.opportunity_contract.sequence_id",
    )
    dependency_ids = [str(item) for item in case.get('depends_on') or []]
    candidates: list[Mapping[str, Any]] = []
    for candidate in manifest.get('cases') or []:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get('case_id') or '') not in dependency_ids:
            continue
        predecessor_contract = candidate.get('opportunity_contract')
        if not isinstance(predecessor_contract, Mapping):
            continue
        if str(predecessor_contract.get('sequence_id') or '').strip() != sequence_id:
            continue
        predecessor_index = predecessor_contract.get('turn_index')
        if (
            isinstance(predecessor_index, int)
            and not isinstance(predecessor_index, bool)
            and predecessor_index == turn_index - 1
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise CorpusError(
            f"Opportunity case '{case.get('case_id')}' has {len(candidates)} exact "
            'immediate settled predecessor candidates; expected one.'
        )
    return candidates[0]


def _predecessor_artifact_handles(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifact_count = _strict_nonnegative_int(
        summary.get('artifact_count'),
        field='final_debug.summary.artifact_count',
    )
    raw_artifacts = summary.get('artifacts')
    if not isinstance(raw_artifacts, list):
        raise CorpusError("Predecessor final_debug.summary.artifacts must be an array.")
    if artifact_count != len(raw_artifacts):
        raise CorpusError(
            'Predecessor artifact summary is incomplete or ambiguous; '
            'artifact_count does not match the retained handles.'
        )
    if artifact_count > MAX_PREDECESSOR_ARTIFACTS:
        raise CorpusError(
            f'Predecessor context has {artifact_count} artifacts; the bounded '
            f'limit is {MAX_PREDECESSOR_ARTIFACTS}.'
        )
    handles: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    seen_ids: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, Mapping):
            raise CorpusError(
                f'Predecessor artifact handle {index} is not an object.'
            )
        artifact_type = _required_bounded_string(
            raw_artifact.get('type') or raw_artifact.get('kind'),
            field=f'final_debug.summary.artifacts[{index}].type',
        ).lower()
        if artifact_type not in {'audio', 'document', 'image', 'text'}:
            raise CorpusError(
                f"Predecessor artifact handle {index} has unsupported type '{artifact_type}'."
            )
        artifact_ref = _required_bounded_string(
            raw_artifact.get('artifact_ref') or raw_artifact.get('ref'),
            field=f'final_debug.summary.artifacts[{index}].artifact_ref',
        )
        artifact_id = _required_bounded_string(
            raw_artifact.get('artifact_id'),
            field=f'final_debug.summary.artifacts[{index}].artifact_id',
        )
        path = _required_bounded_string(
            raw_artifact.get('path'),
            field=f'final_debug.summary.artifacts[{index}].path',
            maximum=MAX_PREDECESSOR_PATH_CHARS,
        )
        if artifact_ref in seen_refs or artifact_id in seen_ids:
            raise CorpusError(
                'Predecessor artifact handles contain duplicate identity and are ambiguous.'
            )
        seen_refs.add(artifact_ref)
        seen_ids.add(artifact_id)
        handle: dict[str, Any] = {
            'type': artifact_type,
            'path': path,
            'artifact_ref': artifact_ref,
            'artifact_id': artifact_id,
        }
        for key in ('kind', 'mime_type', 'extension', 'name'):
            raw_value = raw_artifact.get(key)
            if raw_value not in (None, ''):
                handle[key] = _required_bounded_string(
                    raw_value,
                    field=f'final_debug.summary.artifacts[{index}].{key}',
                )
        for key in ('sha256', 'file_sha256'):
            raw_value = raw_artifact.get(key)
            if raw_value not in (None, ''):
                handle[key] = _exact_sha256(
                    raw_value,
                    field=f'final_debug.summary.artifacts[{index}].{key}',
                )
        for key in ('size_bytes', 'file_size_bytes'):
            raw_value = raw_artifact.get(key)
            if raw_value not in (None, ''):
                handle[key] = _strict_nonnegative_int(
                    raw_value,
                    field=f'final_debug.summary.artifacts[{index}].{key}',
                )
        handles.append(handle)
    return handles


def _opportunity_predecessor_context(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    predecessor = _immediate_opportunity_predecessor(manifest, case)
    if predecessor is None:
        return None
    predecessor_id = _required_bounded_string(
        predecessor.get('response_id'),
        field='predecessor.response_id',
    )
    if str(predecessor.get('state') or '') not in SETTLED_CASE_STATES:
        raise CorpusError(
            f"Opportunity predecessor '{predecessor.get('case_id')}' is not settled."
        )
    if str(predecessor.get('conversation_id') or '') != str(case.get('conversation_id') or ''):
        raise CorpusError('Opportunity predecessor conversation identity is inconsistent.')
    final_debug = predecessor.get('final_debug')
    if (
        not isinstance(final_debug, Mapping)
        or final_debug.get('status') != 'captured'
        or not isinstance(final_debug.get('summary'), Mapping)
    ):
        raise CorpusError(
            f"Opportunity predecessor '{predecessor.get('case_id')}' has no immutable "
            'captured final_debug.summary context.'
        )
    summary = final_debug['summary']
    if str(summary.get('id') or '') != predecessor_id:
        raise CorpusError('Predecessor final_debug response identity does not match the manifest.')
    frame = summary.get('response_frame')
    if not isinstance(frame, Mapping):
        raise CorpusError('Predecessor final_debug frame identity is absent.')
    frame_id = _required_bounded_string(
        frame.get('frame_id'),
        field='final_debug.summary.response_frame.frame_id',
    )
    frame_sequence = _strict_positive_int(
        frame.get('frame_sequence'),
        field='final_debug.summary.response_frame.frame_sequence',
    )
    if (
        frame_id != str(predecessor.get('last_frame_id') or '')
        or frame_sequence != predecessor.get('last_frame_sequence')
    ):
        raise CorpusError(
            'Predecessor final_debug frame identity is stale or ambiguous against the '
            'manifest cursor.'
        )
    message_identity = summary.get('message_identity')
    if not isinstance(message_identity, Mapping) or message_identity.get('status') != 'exact':
        raise CorpusError('Predecessor assistant-message identity is absent or ambiguous.')
    message_id = _required_bounded_string(
        message_identity.get('message_id'),
        field='final_debug.summary.message_identity.message_id',
    )
    if summary.get('message_id') not in (None, message_id):
        raise CorpusError('Predecessor assistant-message identity is internally inconsistent.')
    final_text = summary.get('final_text')
    if not isinstance(final_text, Mapping):
        raise CorpusError('Predecessor final_text is absent.')
    text = final_text.get('text')
    if not isinstance(text, str) or not text.strip():
        raise CorpusError('Predecessor final_text.text is absent.')
    if len(text) > MAX_FINAL_TEXT_CHARS or final_text.get('truncated') is not False:
        raise CorpusError('Predecessor final_text is not available as bounded exact text.')
    length_chars = _strict_nonnegative_int(
        final_text.get('length_chars'),
        field='final_debug.summary.final_text.length_chars',
    )
    text_sha256 = _exact_sha256(
        final_text.get('sha256'),
        field='final_debug.summary.final_text.sha256',
    )
    if length_chars != len(text) or text_sha256 != hashlib.sha256(text.encode('utf-8')).hexdigest():
        raise CorpusError('Predecessor final_text identity does not match its exact text.')
    artifact_handles = _predecessor_artifact_handles(summary)
    context_artifact_handles = [
        {
            **dict(item),
            'source_response_id': predecessor_id,
            'message_id': message_id,
            'source_message_id': message_id,
        }
        for item in artifact_handles
    ]
    assistant_message = {
        'role': 'assistant',
        'content': text,
        'message_id': message_id,
        'response_id': predecessor_id,
        'artifacts': [dict(item) for item in context_artifact_handles],
    }
    message_reference = {
        'type': 'message',
        'message_role': 'assistant',
        'message_id': message_id,
        'source_response_id': predecessor_id,
        'content': text,
    }
    digest_source = {
        'predecessor_case_id': predecessor.get('case_id'),
        'response_id': predecessor_id,
        'frame_id': frame_id,
        'frame_sequence': frame_sequence,
        'message_id': message_id,
        'final_text_sha256': text_sha256,
        'artifact_handles': context_artifact_handles,
    }
    context_digest = stable_digest(digest_source)
    context = {
        'ghost_messages': [assistant_message],
        'reference_artifacts': [message_reference, *context_artifact_handles],
        'audit': {
            'kind': 'ollmo.graph_rebase_shadow_corpus_predecessor_context',
            'source': 'immutable_final_debug_summary',
            'predecessor_case_id': predecessor.get('case_id'),
            'response_id': predecessor_id,
            'frame_id': frame_id,
            'frame_sequence': frame_sequence,
            'message_id': message_id,
            'final_text_sha256': text_sha256,
            'artifact_count': len(context_artifact_handles),
            'artifact_refs': [item['artifact_ref'] for item in context_artifact_handles],
            'context_digest': context_digest,
        },
    }
    context_size = len(canonical_json_bytes(context))
    if context_size > MAX_PREDECESSOR_CONTEXT_BYTES:
        raise CorpusError(
            f'Predecessor context is {context_size} bytes; the bounded limit is '
            f'{MAX_PREDECESSOR_CONTEXT_BYTES} bytes.'
        )
    return context


def build_request_payload(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    ghost_preferences: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(case.get('request_overrides') or {})
    payload.update(
        {
            'response_id': case.get('response_id'),
            'conversation_id': case.get('conversation_id'),
            'ghost_route': True,
            'prompt': case.get('prompt'),
            'workload_family': case.get('workload_family'),
            'request_meta': {
                'source': 'graph_rebase_shadow_corpus_runner',
                'corpus_id': manifest.get('corpus_id'),
                'corpus_digest': manifest.get('corpus_digest'),
                'case_id': case.get('case_id'),
                'wave': case.get('wave'),
                'workload_family': case.get('workload_family'),
            },
        }
    )
    if ghost_preferences:
        payload['ghost_preferences'] = dict(ghost_preferences)
    predecessor_context = _opportunity_predecessor_context(manifest, case)
    if predecessor_context:
        payload['ghost_messages'] = predecessor_context['ghost_messages']
        payload['reference_artifacts'] = predecessor_context['reference_artifacts']
        payload['request_meta']['predecessor_context'] = predecessor_context['audit']
    return payload


class ShadowCorpusRunner:
    def __init__(
        self,
        *,
        corpus: Mapping[str, Any],
        manifest: dict[str, Any],
        manifest_path: Path | str,
        client: Any,
        wave: Optional[str] = None,
        case_ids: Optional[Sequence[str]] = None,
        max_in_flight: int = 1,
        poll_interval: float = 5.0,
        get_timeout: float = 30.0,
        post_timeout: float = 7200.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        emit: Callable[[str], None] = print,
    ):
        if max_in_flight < 1:
            raise CorpusError('--max-in-flight must be at least 1.')
        self.corpus = dict(corpus)
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.client = client
        self.wave = str(wave).strip() if wave not in (None, '') else None
        self.case_ids = normalize_case_filter(list(case_ids or []))
        self.max_in_flight = int(max_in_flight)
        self.poll_interval = max(0.0, float(poll_interval))
        self.get_timeout = max(0.1, float(get_timeout))
        self.post_timeout = max(0.1, float(post_timeout))
        self.sleep_fn = sleep_fn
        self.emit = emit
        self._dispatch_results: queue.Queue[tuple[str, HttpResult]] = queue.Queue()
        self._dispatch_threads: dict[str, threading.Thread] = {}
        self._ghost_preferences: dict[str, Any] = {}
        self._runtime_ready = False

    def persist(self) -> None:
        self.manifest['updated_at'] = utc_now()
        atomic_write_json(self.manifest_path, self.manifest)

    def _case(self, case_id: str) -> dict[str, Any]:
        case = manifest_case_map(self.manifest).get(case_id)
        if case is None:
            raise CorpusError(f'Manifest case disappeared: {case_id}')
        return case

    def _append_run_history(self, event: str, **fields: Any) -> None:
        history = self.manifest.setdefault('run_history', [])
        history.append(
            {
                'at': utc_now(),
                'event': event,
                **{key: value for key, value in fields.items() if value not in (None, '')},
            }
        )
        if len(history) > 256:
            del history[:-256]

    def _preflight(self) -> bool:
        running = self.client.get('/api/running_instances', timeout=self.get_timeout)
        preferences = self.client.get('/api/ghost_preferences', timeout=self.get_timeout)
        if not running.ok or not isinstance(running.payload, list):
            self._append_run_history(
                'runtime_preflight_failed',
                status_code=running.status_code,
                error=running.error or running.payload,
            )
            self.persist()
            return False
        ready_instances = [
            item
            for item in running.payload
            if isinstance(item, Mapping)
            and str(item.get('readiness') or '').strip().lower() == 'ready'
        ]
        if not ready_instances:
            self._append_run_history('runtime_preflight_failed', error='no_ready_instances')
            self.persist()
            return False
        if not preferences.ok or not isinstance(preferences.payload, Mapping):
            self._append_run_history(
                'ghost_preferences_unavailable',
                status_code=preferences.status_code,
                error=preferences.error or preferences.payload,
            )
            self.persist()
            return False
        preferences_value = preferences.payload.get('preferences')
        self._ghost_preferences = (
            dict(preferences_value) if isinstance(preferences_value, Mapping) else {}
        )
        ready_capabilities: set[str] = set()
        for item in ready_instances:
            for value in [
                item.get('capability'),
                *(item.get('provider_capabilities') or []),
            ]:
                token = str(value or '').strip()
                if token:
                    ready_capabilities.add(token)
        selected = selected_manifest_cases(self.manifest, self.wave, self.case_ids)
        missing = sorted(
            {
                expected
                for case in selected
                if case.get('state') in {'planned', 'waiting_runtime'}
                for expected in case.get('expected_capability_families') or []
                if expected not in ready_capabilities
            }
        )
        if missing:
            self._append_run_history(
                'runtime_preflight_failed',
                error='expected_capabilities_not_ready',
                missing_capabilities=missing,
                ready_capabilities=sorted(ready_capabilities),
            )
            self.persist()
            return False
        self.manifest['runtime_preflight'] = {
            'checked_at': utc_now(),
            'ready_instance_count': len(ready_instances),
            'ready_capabilities': sorted(ready_capabilities),
            'ghost_preferences_digest': stable_digest(self._ghost_preferences),
        }
        self._runtime_ready = True
        self.persist()
        return True

    def _checkpoint_readiness(self, phase: str, *, case_id: Optional[str] = None) -> None:
        checkpoints = self.manifest.setdefault('readiness_checkpoints', [])
        if any(
            item.get('phase') == phase and item.get('case_id') == case_id
            for item in checkpoints
            if isinstance(item, Mapping)
        ):
            return
        result = self.client.get('/api/graph_rebase/readiness', timeout=self.get_timeout)
        checkpoint: dict[str, Any] = {
            'captured_at': utc_now(),
            'phase': phase,
        }
        if case_id:
            checkpoint['case_id'] = case_id
        if result.ok and isinstance(result.payload, Mapping):
            checkpoint['status'] = 'captured'
            checkpoint['summary'] = summarize_readiness(
                result.payload,
                byte_count=result.byte_count,
            )
        else:
            checkpoint['status'] = 'failed'
            checkpoint['http_status'] = result.status_code
            checkpoint['error'] = result.error or result.payload
        checkpoints.append(checkpoint)
        self.persist()

    @staticmethod
    def _frame_sequence_number(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value in (None, ''):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _update_last_frame_observation(
        self,
        case: dict[str, Any],
        *,
        frame_id: Any = None,
        frame_sequence: Any = None,
    ) -> bool:
        observed_id = str(frame_id or '').strip() or None
        observed_sequence = self._frame_sequence_number(frame_sequence)
        current_id = str(case.get('last_frame_id') or '').strip() or None
        current_sequence = self._frame_sequence_number(case.get('last_frame_sequence'))

        if (
            observed_sequence is not None
            and current_sequence is not None
            and observed_sequence < current_sequence
        ):
            return False

        changed = False
        if observed_sequence is not None and (
            current_sequence is None or observed_sequence > current_sequence
        ):
            case['last_frame_sequence'] = observed_sequence
            changed = True
            if observed_id and observed_id != current_id:
                case['last_frame_id'] = observed_id
        elif observed_sequence == current_sequence and not current_id and observed_id:
            case['last_frame_id'] = observed_id
            changed = True
        elif observed_sequence is None and not current_id and observed_id:
            case['last_frame_id'] = observed_id
            changed = True

        return changed

    def _record_status(self, case: dict[str, Any], payload: Mapping[str, Any]) -> bool:
        observation = compact_status_observation(payload)
        version = str(observation.get('state_version') or '')
        observations = case.setdefault('status_observations', [])
        previous_version = str(case.get('last_state_version') or '')
        changed = not version or version != previous_version
        if changed:
            observations.append(observation)
            if len(observations) > MAX_STATUS_OBSERVATIONS:
                del observations[:-MAX_STATUS_OBSERVATIONS]
            case['status_observation_count'] = int(case.get('status_observation_count') or 0) + 1
            if version:
                case['last_state_version'] = version
        case['last_observed_at'] = utc_now()
        case['last_lifecycle_state'] = payload.get('lifecycle_state')
        late_fill = payload.get('late_fill')
        if isinstance(late_fill, Mapping):
            late_fill_status = str(late_fill.get('status') or '').strip().lower()
            if late_fill_status:
                case['last_late_fill_status'] = late_fill_status
        self._update_last_frame_observation(
            case,
            frame_id=payload.get('frame_id'),
            frame_sequence=payload.get('frame_sequence'),
        )
        case['updated_at'] = utc_now()
        return changed

    def _fetch_final_debug_once(self, case: dict[str, Any]) -> None:
        debug = case.get('final_debug')
        if isinstance(debug, Mapping) and debug.get('attempted_at'):
            if debug.get('status') == 'fetching':
                case['final_debug'] = {
                    **dict(debug),
                    'status': 'ambiguous_incomplete',
                    'finished_at': utc_now(),
                }
                self.persist()
            return
        case['final_debug'] = {'status': 'fetching', 'attempted_at': utc_now()}
        self.persist()
        response_id = quote(str(case.get('response_id') or ''), safe='')
        result = self.client.get(
            f'/api/responses/{response_id}?view=debug',
            timeout=self.get_timeout,
        )
        if result.ok and isinstance(result.payload, Mapping):
            debug_summary = summarize_debug_payload(
                result.payload,
                byte_count=result.byte_count,
            )
            debug_summary['rebase_opportunity'] = derive_rebase_opportunity_summary(
                debug_summary,
                opportunity_contract=case.get('opportunity_contract'),
            )
            case['final_debug'] = {
                'status': 'captured',
                'attempted_at': case['final_debug']['attempted_at'],
                'finished_at': utc_now(),
                'summary': debug_summary,
            }
            debug_frame = (
                debug_summary.get('response_frame')
                if isinstance(debug_summary.get('response_frame'), Mapping)
                else {}
            )
            self._update_last_frame_observation(
                case,
                frame_id=debug_frame.get('frame_id'),
                frame_sequence=debug_frame.get('frame_sequence'),
            )
        else:
            case['final_debug'] = {
                'status': 'failed',
                'attempted_at': case['final_debug']['attempted_at'],
                'finished_at': utc_now(),
                'http_status': result.status_code,
                'error': result.error or result.payload,
            }
        self.persist()

    def _settle_case(self, case: dict[str, Any], state: str) -> None:
        if case.get('state') in SETTLED_CASE_STATES:
            return
        case['state'] = state
        lifecycle = str(case.get('last_lifecycle_state') or '').strip().lower()
        late_fill_status = str(
            case.get('last_late_fill_status') or ''
        ).strip().lower()
        case['settled_outcome'] = (
            'success'
            if state == 'settled_terminal'
            and lifecycle in SUCCESSFUL_DEPENDENCY_LIFECYCLES
            and late_fill_status not in NON_SUCCESS_TERMINAL_LATE_FILL_STATUSES
            else 'repair_needed'
            if state == 'settled_repair_needed'
            else 'non_success_terminal'
        )
        case['dependency_satisfied'] = case['settled_outcome'] == 'success'
        case['settled_at'] = utc_now()
        case['updated_at'] = utc_now()
        self.persist()
        self._fetch_final_debug_once(case)
        self._checkpoint_readiness('after_case', case_id=str(case.get('case_id')))
        self.emit(f"{case.get('case_id')}: {state}")

    def _observe_case(self, case: dict[str, Any], *, resumed_submitting: bool = False) -> str:
        response_id = quote(str(case.get('response_id') or ''), safe='')
        result = self.client.get(
            f'/api/responses/{response_id}?view=status',
            timeout=self.get_timeout,
        )
        case['last_lookup_http_status'] = result.status_code
        if result.status_code == 404:
            case['last_observed_at'] = utc_now()
            case['lookup_not_found_count'] = int(case.get('lookup_not_found_count') or 0) + 1
            if resumed_submitting or case.get('state') == 'submitting':
                case['state'] = 'dispatch_unknown'
                case['dispatch_unknown_reason'] = 'submitting_resume_lookup_not_found'
                case['dispatch_unknown_at'] = utc_now()
            case['updated_at'] = utc_now()
            self.persist()
            return 'not_found'
        if not result.ok or not isinstance(result.payload, Mapping):
            case['last_observation_error'] = {
                'at': utc_now(),
                'http_status': result.status_code,
                'error': result.error or result.payload,
            }
            case['updated_at'] = utc_now()
            self.persist()
            return 'error'
        case.pop('last_observation_error', None)
        changed = self._record_status(case, result.payload)
        classification = classify_compact_status(result.payload)
        if classification == 'settled_repair_needed':
            self.persist()
            self._settle_case(case, 'settled_repair_needed')
            return classification
        if classification == 'settled_terminal':
            self.persist()
            self._settle_case(case, 'settled_terminal')
            return classification
        if case.get('state') != 'dispatch_unknown':
            case['state'] = 'observing'
        if classification == 'unknown_nonterminal':
            case['unknown_nonterminal_count'] = int(case.get('unknown_nonterminal_count') or 0) + 1
        case['updated_at'] = utc_now()
        if changed:
            self.emit(
                f"{case.get('case_id')}: {result.payload.get('lifecycle_state') or classification}"
            )
        self.persist()
        return classification

    def _dispatch_worker(self, case_id: str, payload: Mapping[str, Any]) -> None:
        try:
            result = self.client.post(
                '/api/responses',
                payload,
                timeout=self.post_timeout,
            )
        except Exception as exc:  # defensive for injected clients
            result = HttpResult(status_code=0, payload={}, error=str(exc))
        self._dispatch_results.put((case_id, result))

    def _start_dispatch(self, case: dict[str, Any]) -> None:
        payload = build_request_payload(self.manifest, case, self._ghost_preferences)
        case['state'] = 'submitting'
        case['submitting_at'] = utc_now()
        case['dispatch_request'] = payload
        case['dispatch_request_digest'] = stable_digest(payload)
        case['updated_at'] = utc_now()
        self.persist()
        thread = threading.Thread(
            target=self._dispatch_worker,
            args=(str(case.get('case_id')), payload),
            name=f"ollmo-shadow-{slug(case.get('case_id'), maximum=32)}",
            daemon=True,
        )
        self._dispatch_threads[str(case.get('case_id'))] = thread
        thread.start()
        self.emit(f"{case.get('case_id')}: submitting {case.get('response_id')}")

    def _drain_dispatch_results(self) -> bool:
        changed = False
        while True:
            try:
                case_id, result = self._dispatch_results.get_nowait()
            except queue.Empty:
                break
            changed = True
            self._dispatch_threads.pop(case_id, None)
            case = self._case(case_id)
            case['post_finished_at'] = utc_now()
            case['post_http_status'] = result.status_code
            case['post_response_bytes'] = result.byte_count
            if result.ok and isinstance(result.payload, Mapping):
                returned_id = str(
                    result.payload.get('id') or result.payload.get('response_id') or ''
                ).strip()
                if returned_id and returned_id != str(case.get('response_id')):
                    case['state'] = 'dispatch_unknown'
                    case['dispatch_unknown_reason'] = 'post_response_id_mismatch'
                    case['returned_response_id'] = returned_id
                elif case.get('state') not in SETTLED_CASE_STATES:
                    case['state'] = 'submitted'
                    case['submitted_at'] = utc_now()
            else:
                case['state'] = 'dispatch_unknown'
                case['dispatch_unknown_at'] = utc_now()
                case['dispatch_unknown_reason'] = 'post_failed_or_ambiguous'
                case['post_error'] = result.error or result.payload
            case['updated_at'] = utc_now()
            self.persist()
        return changed

    def _get_before_post(self, case: dict[str, Any]) -> str:
        response_id = quote(str(case.get('response_id') or ''), safe='')
        result = self.client.get(
            f'/api/responses/{response_id}?view=status',
            timeout=self.get_timeout,
        )
        case['pre_dispatch_lookup'] = {
            'at': utc_now(),
            'http_status': result.status_code,
        }
        if result.status_code == 404:
            self.persist()
            return 'absent'
        if result.ok and isinstance(result.payload, Mapping):
            case['state'] = 'observing'
            case['existing_response_found_at'] = utc_now()
            self._record_status(case, result.payload)
            self.persist()
            classification = classify_compact_status(result.payload)
            if classification == 'settled_repair_needed':
                self._settle_case(case, 'settled_repair_needed')
            elif classification == 'settled_terminal':
                self._settle_case(case, 'settled_terminal')
            return 'existing'
        case['state'] = 'waiting_runtime'
        case['pre_dispatch_lookup']['error'] = result.error or result.payload
        case['updated_at'] = utc_now()
        self.persist()
        return 'unavailable'

    def _mark_dependency_blocks(self) -> bool:
        changed = False
        cases_by_id = manifest_case_map(self.manifest)
        for case in selected_manifest_cases(self.manifest, self.wave, self.case_ids):
            if case.get('state') not in {'planned', 'waiting_runtime'}:
                continue
            state, dependency_ids = dependency_state(case, cases_by_id)
            if state == 'blocked':
                case['state'] = 'dependency_blocked'
                case['dependency_blocked_by'] = dependency_ids
                case['updated_at'] = utc_now()
                changed = True
        if changed:
            self.persist()
        return changed

    def _eligible_planned_cases(self) -> list[dict[str, Any]]:
        cases_by_id = manifest_case_map(self.manifest)
        eligible: list[dict[str, Any]] = []
        for case in selected_manifest_cases(self.manifest, self.wave, self.case_ids):
            if case.get('state') not in {'planned', 'waiting_runtime'}:
                continue
            state, _dependency_ids = dependency_state(case, cases_by_id)
            if state == 'ready':
                eligible.append(case)
        return sorted(eligible, key=lambda item: int(item.get('ordinal') or 0))

    def _active_count(self) -> int:
        return sum(
            1
            for case in selected_manifest_cases(self.manifest, self.wave, self.case_ids)
            if case.get('state') in ACTIVE_CASE_STATES
        )

    def run(self, *, max_cycles: Optional[int] = None) -> int:
        selected = selected_manifest_cases(self.manifest, self.wave, self.case_ids)
        if not selected:
            raise CorpusError(f'No corpus cases match wave {self.wave!r}.')
        self._append_run_history(
            'run_started',
            wave=self.wave,
            max_in_flight=self.max_in_flight,
        )
        self.persist()
        self._checkpoint_readiness('baseline')

        # Any state left by a prior process is observation-only.  Live dispatch
        # threads are known only to this process and are added below.
        for case in selected:
            if case.get('state') in SETTLED_CASE_STATES:
                self._fetch_final_debug_once(case)
                self._checkpoint_readiness('after_case', case_id=str(case.get('case_id')))
                continue
            if case.get('state') == 'submitting':
                self._observe_case(case, resumed_submitting=True)

        cycles = 0
        while True:
            cycles += 1
            if max_cycles is not None and cycles > max_cycles:
                return 3
            progress = self._drain_dispatch_results()
            self._mark_dependency_blocks()

            for case in selected_manifest_cases(self.manifest, self.wave, self.case_ids):
                case_id = str(case.get('case_id'))
                if case_id in self._dispatch_threads:
                    # Polling while the initial request is still in flight is
                    # safe, but a 404 cannot reclassify this live dispatch.
                    result = self._observe_case(case)
                    if result == 'not_found':
                        case['state'] = 'submitting'
                        self.persist()
                    continue
                if case.get('state') in GET_ONLY_CASE_STATES:
                    self._observe_case(case)

            # Observation can settle a predecessor during this cycle. Re-run
            # dependency classification before deciding that no progress is
            # possible so failed/repair-needed predecessors block dependents
            # immediately instead of requiring a second runner invocation.
            self._mark_dependency_blocks()

            selected = selected_manifest_cases(self.manifest, self.wave, self.case_ids)
            if all(case.get('state') in FINAL_CASE_STATES for case in selected):
                self._checkpoint_readiness('final')
                self._append_run_history('run_completed', wave=self.wave)
                self.persist()
                return 0

            ambiguous_missing = [
                case
                for case in selected
                if case.get('state') == 'dispatch_unknown'
                and int(case.get('last_lookup_http_status') or 0) == 404
            ]
            if ambiguous_missing and not self._dispatch_threads:
                self._append_run_history(
                    'reconciliation_required',
                    case_ids=[case.get('case_id') for case in ambiguous_missing],
                )
                self.persist()
                return 2

            slots = self.max_in_flight - self._active_count()
            if slots > 0:
                if not self._runtime_ready and not self._preflight():
                    return 2
                for case in self._eligible_planned_cases()[:slots]:
                    if case.get('state') == 'waiting_runtime':
                        case['state'] = 'planned'
                        self.persist()
                    preflight = self._get_before_post(case)
                    if preflight == 'absent':
                        self._start_dispatch(case)
                        progress = True

            if not self._dispatch_threads and self._active_count() == 0:
                waiting = [
                    case
                    for case in selected_manifest_cases(self.manifest, self.wave, self.case_ids)
                    if case.get('state') not in FINAL_CASE_STATES
                ]
                if waiting:
                    self._append_run_history(
                        'run_paused_no_progress',
                        case_ids=[case.get('case_id') for case in waiting],
                    )
                    self.persist()
                    return 2

            if not progress or self._dispatch_threads or self._active_count():
                self.sleep_fn(self.poll_interval)


def plan_payload(
    corpus: Mapping[str, Any],
    *,
    wave: Optional[str] = None,
    case_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    normalized_wave = str(wave).strip() if wave not in (None, '') else None
    selected_ids = set(normalize_case_filter(list(case_ids or [])))
    known_ids = {str(case.get('case_id')) for case in corpus.get('cases') or []}
    unknown = sorted(selected_ids - known_ids)
    if unknown:
        raise CorpusError(f"Unknown --case-id value(s): {', '.join(unknown)}")
    cases = [
        {
            'ordinal': case.get('ordinal'),
            'case_id': case.get('case_id'),
            'wave': case.get('wave'),
            'category': case.get('category'),
            'workload_family': case.get('workload_family'),
            'conversation_id': deterministic_conversation_id(corpus, case),
            'response_id': deterministic_response_id(corpus, case),
            'depends_on': list(case.get('depends_on') or []),
            'expected_capability_families': list(
                case.get('expected_capability_families') or []
            ),
            'opportunity_contract': dict(case.get('opportunity_contract') or {}),
        }
        for case in corpus.get('cases') or []
        if normalized_wave is None or str(case.get('wave')) == normalized_wave
        if not selected_ids or str(case.get('case_id')) in selected_ids
    ]
    return {
        'corpus_id': corpus.get('corpus_id'),
        'corpus_digest': corpus.get('corpus_digest'),
        'wave': normalized_wave,
        'case_count': len(cases),
        'cases': cases,
        'runtime_effect': 'none',
    }


def manifest_status_payload(
    manifest: Mapping[str, Any],
    *,
    wave: Optional[str] = None,
    case_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    cases = selected_manifest_cases(manifest, wave, case_ids)
    counts = Counter(str(case.get('state') or 'unknown') for case in cases)
    return {
        'corpus_id': manifest.get('corpus_id'),
        'corpus_digest': manifest.get('corpus_digest'),
        'manifest_path': manifest.get('manifest_path'),
        'wave': str(wave).strip() if wave not in (None, '') else None,
        'case_count': len(cases),
        'by_state': dict(sorted(counts.items())),
        'cases': [
            {
                key: case.get(key)
                for key in (
                    'case_id',
                    'wave',
                    'category',
                    'workload_family',
                    'state',
                    'response_id',
                    'conversation_id',
                    'last_lifecycle_state',
                    'last_late_fill_status',
                    'last_frame_id',
                    'last_frame_sequence',
                    'last_observed_at',
                    'settled_at',
                    'dispatch_unknown_reason',
                    'dependency_blocked_by',
                )
                if case.get(key) not in (None, '', [], {})
            }
            | (
                {
                    'rebase_opportunity': dict(
                        case.get('final_debug', {})
                        .get('summary', {})
                        .get('rebase_opportunity', {})
                    )
                }
                if isinstance(case.get('final_debug'), Mapping)
                and isinstance(case.get('final_debug', {}).get('summary'), Mapping)
                and isinstance(
                    case.get('final_debug', {})
                    .get('summary', {})
                    .get('rebase_opportunity'),
                    Mapping,
                )
                else {}
            )
            for case in cases
        ],
        'readiness_checkpoints': list(manifest.get('readiness_checkpoints') or []),
    }


def _print_plan(payload: Mapping[str, Any]) -> None:
    print(
        f"Corpus {payload.get('corpus_id')} digest={payload.get('corpus_digest')} "
        f"cases={payload.get('case_count')} wave={payload.get('wave') or 'all'}"
    )
    for case in payload.get('cases') or []:
        dependencies = ','.join(case.get('depends_on') or []) or '-'
        print(
            f"{int(case.get('ordinal') or 0):02d} {case.get('case_id')} "
            f"wave={case.get('wave')} family={case.get('workload_family')} "
            f"depends={dependencies} response_id={case.get('response_id')}"
        )


def _print_status(payload: Mapping[str, Any]) -> None:
    print(
        f"Corpus {payload.get('corpus_id')} cases={payload.get('case_count')} "
        f"wave={payload.get('wave') or 'all'} states={payload.get('by_state')}"
    )
    for case in payload.get('cases') or []:
        print(
            f"{case.get('case_id')}: {case.get('state')} "
            f"lifecycle={case.get('last_lifecycle_state') or '-'} "
            f"response_id={case.get('response_id')}"
        )


def default_manifest_path(corpus: Mapping[str, Any]) -> Path:
    return DEFAULT_MANIFEST_ROOT / f"{slug(corpus.get('corpus_id'), maximum=64)}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    plan = subparsers.add_parser('plan', help='Validate and display a corpus without HTTP or writes.')
    plan.add_argument('--corpus', type=Path, required=True)
    plan.add_argument('--wave')
    plan.add_argument(
        '--case-id',
        action='append',
        default=[],
        help='Limit to one case id; repeat the option or use comma-separated ids.',
    )
    plan.add_argument('--json', action='store_true')

    run = subparsers.add_parser('run', help='Start or safely resume corpus execution.')
    run.add_argument('--corpus', type=Path, required=True)
    run.add_argument('--manifest', type=Path)
    run.add_argument('--wave')
    run.add_argument(
        '--case-id',
        action='append',
        default=[],
        help='Run only the named case; repeat the option or use comma-separated ids.',
    )
    run.add_argument('--base-url', default=DEFAULT_BASE_URL)
    run.add_argument('--max-in-flight', type=int, default=1)
    run.add_argument('--poll-interval', type=float, default=5.0)
    run.add_argument('--get-timeout', type=float, default=30.0)
    run.add_argument('--post-timeout', type=float, default=7200.0)
    run.add_argument('--json', action='store_true')

    status = subparsers.add_parser('status', help='Read a persisted manifest without HTTP.')
    status.add_argument('--manifest', type=Path, required=True)
    status.add_argument('--corpus', type=Path)
    status.add_argument('--wave')
    status.add_argument(
        '--case-id',
        action='append',
        default=[],
        help='Show only the named case; repeat the option or use comma-separated ids.',
    )
    status.add_argument('--json', action='store_true')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'plan':
            corpus = load_corpus(args.corpus)
            payload = plan_payload(corpus, wave=args.wave, case_ids=args.case_id)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                _print_plan(payload)
            return 0

        if args.command == 'status':
            manifest = load_manifest(args.manifest)
            if args.corpus:
                assert_manifest_matches_corpus(manifest, load_corpus(args.corpus))
            payload = manifest_status_payload(
                manifest,
                wave=args.wave,
                case_ids=args.case_id,
            )
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                _print_status(payload)
            return 0

        corpus = load_corpus(args.corpus)
        manifest_path = args.manifest or default_manifest_path(corpus)
        with manifest_lock(manifest_path):
            if manifest_path.exists():
                manifest = load_manifest(manifest_path)
                assert_manifest_matches_corpus(manifest, corpus)
            else:
                manifest = build_manifest(corpus, manifest_path)
                atomic_write_json(manifest_path, manifest)
            client = JsonHttpClient(args.base_url, default_timeout=args.get_timeout)
            runner = ShadowCorpusRunner(
                corpus=corpus,
                manifest=manifest,
                manifest_path=manifest_path,
                client=client,
                wave=args.wave,
                case_ids=args.case_id,
                max_in_flight=args.max_in_flight,
                poll_interval=args.poll_interval,
                get_timeout=args.get_timeout,
                post_timeout=args.post_timeout,
            )
            exit_code = runner.run()
            payload = manifest_status_payload(
                runner.manifest,
                wave=args.wave,
                case_ids=args.case_id,
            )
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                _print_status(payload)
            return exit_code
    except (CorpusError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted; persisted cases will resume by response id.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
