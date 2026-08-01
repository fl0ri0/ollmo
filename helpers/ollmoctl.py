"""Thin API-first CLI layer over the Ollmo Flask service."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

from ollmo_g.reset_learning_state import reset_ghost_learning_state
from ollmo_runtime.child_process_env import sanitized_child_process_env
from ollmo_services.graph_rebase import (
    parse_graph_rebase_frame_sequence,
    stable_graph_digest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = (
    os.environ.get('OLLMOCTL_BASE')
    or os.environ.get('OLLMO_WEB_BASE')
    or 'http://127.0.0.1:5001'
)
DEFAULT_OLLAMA_CLI = '/opt/homebrew/bin/ollama'
DEFAULT_OLLAMA_OPT_CLI = '/opt/homebrew/opt/ollama/bin/ollama'
DEFAULT_MLX_PYTHON = os.environ.get('MLX_PYTHON') or '/opt/mlx/venv/bin/python'
DEFAULT_LOCAL_CONTROL_PLANE_HOSTS = {'127.0.0.1', 'localhost', '::1'}
DEFAULT_LOCAL_CONTROL_PLANE_PORT = 5001
DEFAULT_LOCAL_WEBSERVER_SCRIPT = REPO_ROOT / 'ollmo_webserver.py'
DEFAULT_LOCAL_WEBSERVER_LOG = REPO_ROOT / 'logs' / 'flask_webserver_auto.log'
DEFAULT_LOCAL_WEBSERVER_WAIT_SEC = 12.0
GRAPH_REBASE_OPERATOR_TOKEN_ENV = 'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN'
GRAPH_REBASE_OPERATOR_IDENTITY_ENV = 'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY'
GRAPH_REBASE_NO_FORMAL_PROPOSAL = 'no_formal_proposal'
GRAPH_REBASE_CLASSES = {
    'partial_subtree_rebase',
    'full_successor_rebase',
}
GRAPH_REBASE_ADJUDICATIONS = {
    'accepted',
    'false_negative',
    'false_positive',
    'needs_investigation',
    'rejected_authorization',
    'useful_proposal',
}
GRAPH_REBASE_FORMAL_GRAPH_KEYS = {
    'graph_rebase_proposals',
    'graph_rebase_reviews',
    'graph_rebase_lifecycle',
    'staged_graph_rebases',
    'applied_graph_rebases',
    'successor_rebase_requests',
    'successor_rebase_executions',
    'graph_rebase_outcomes',
    'partial_rebase_outcomes',
}
GRAPH_REBASE_FORMAL_DIAGNOSTIC_KEYS = {
    'runtime_graph_rebase_proposals',
    'runtime_graph_rebase_reviews',
    'graph_rebase_lifecycle',
    'staged_graph_rebases',
    'applied_graph_rebases',
    'successor_rebase_requests',
    'graph_rebase_outcomes',
    'partial_rebase_outcomes',
}
GRAPH_REBASE_SETTLED_LIFECYCLE_STATES = {
    'blocked',
    'cancelled',
    'completed',
    'failed',
    'late_fill_completed',
    'repair_needed',
}
OPEN_RESPONSE_LIFECYCLE_STATES = {
    'accepted',
    'active',
    'in_progress',
    'late_fill_pending',
    'late_fill_running',
    'pending',
    'queued',
    'running',
    'scheduled',
    'started',
    'streaming',
}
ACTIONABLE_RESPONSE_LIFECYCLE_STATES = {
    'blocked',
    'late_fill_blocked',
    'late_fill_repair_needed',
    'rebuild_from_promoted_obligations',
    'repair_branch_contract',
    'repair_dependency_chain',
    'repair_needed',
}
TERMINAL_RESPONSE_LIFECYCLE_STATES = {
    'cancelled',
    'canceled',
    'completed',
    'failed',
    'frozen',
    'late_fill_completed',
    'late_fill_failed',
    'partial_cancelled',
    'partial_failed',
    'skipped',
    'superseded',
    'waived',
}
LEGACY_ACTIVE_LATE_FILL_STATUSES = {
    'accepted',
    'pending',
    'queued',
    'running',
    'scheduled',
    'started',
    'active',
    'in_progress',
}
KNOWN_RESPONSE_LIFECYCLE_STATES = (
    OPEN_RESPONSE_LIFECYCLE_STATES
    | ACTIONABLE_RESPONSE_LIFECYCLE_STATES
    | TERMINAL_RESPONSE_LIFECYCLE_STATES
    | LEGACY_ACTIVE_LATE_FILL_STATUSES
    | {'partial_failed'}
)


class CliError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Fail closed instead of forwarding credentialed operator requests."""

    def _reject(self, request, fp, code, message, headers):
        raise HTTPError(
            request.full_url,
            code,
            'Redirects are forbidden for graph-rebase operator requests.',
            headers,
            fp,
        )

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


def _normalize_base_url(base_url: Optional[str]) -> str:
    return str(base_url or DEFAULT_BASE_URL).rstrip('/')


def _build_url(base_url: str, path: str, query: Optional[dict[str, Any]] = None) -> str:
    root = _normalize_base_url(base_url)
    suffix = path if path.startswith('/') else f'/{path}'
    url = f'{root}{suffix}'
    if query:
        clean = {key: value for key, value in query.items() if value not in (None, '')}
        if clean:
            url = f'{url}?{urlencode(clean, doseq=True)}'
    return url


def _extract_error_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        for key in ('error', 'message'):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return fallback


def _is_default_local_control_plane(base_url: str) -> bool:
    parsed = urlparse(_normalize_base_url(base_url))
    host = str(parsed.hostname or '').strip().lower()
    try:
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    except ValueError:
        return False
    return host in DEFAULT_LOCAL_CONTROL_PLANE_HOSTS and port == DEFAULT_LOCAL_CONTROL_PLANE_PORT


def _require_local_graph_rebase_operator_target(base_url: str) -> None:
    parsed = urlparse(_normalize_base_url(base_url))
    if (
        parsed.scheme not in {'http', 'https'}
        or not _is_default_local_control_plane(base_url)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {'', '/'}
    ):
        raise CliError(
            'Credentialed graph-rebase operator actions require the exact loopback '
            f'control plane on port {DEFAULT_LOCAL_CONTROL_PLANE_PORT}; refusing to send credentials.'
        )


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_for_local_control_plane(base_url: str, timeout_sec: float) -> bool:
    parsed = urlparse(_normalize_base_url(base_url))
    host = str(parsed.hostname or '').strip() or '127.0.0.1'
    port = parsed.port or DEFAULT_LOCAL_CONTROL_PLANE_PORT
    deadline = time.time() + max(1.0, timeout_sec)
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.25)
    return _is_port_open(host, port)


def _attempt_local_control_plane_recovery(base_url: str) -> bool:
    if not _is_default_local_control_plane(base_url):
        return False
    if _wait_for_local_control_plane(base_url, timeout_sec=0.5):
        return True
    if not DEFAULT_LOCAL_WEBSERVER_SCRIPT.exists():
        return False

    python_path = REPO_ROOT / '.venv' / 'bin' / 'python3'
    python_exec = str(python_path) if python_path.exists() else (sys.executable or 'python3')
    DEFAULT_LOCAL_WEBSERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = sanitized_child_process_env()
    env.setdefault('PYTHONUNBUFFERED', '1')

    try:
        with DEFAULT_LOCAL_WEBSERVER_LOG.open('ab') as log_handle:
            subprocess.Popen(  # noqa: S603
                [python_exec, str(DEFAULT_LOCAL_WEBSERVER_SCRIPT)],
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
    except OSError:
        return False

    return _wait_for_local_control_plane(base_url, timeout_sec=DEFAULT_LOCAL_WEBSERVER_WAIT_SEC)


def _request(
    method: str,
    base_url: str,
    path: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    query: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_sec: int = 30,
    allow_runtime_recovery: bool = False,
    allow_redirects: bool = True,
    _recovery_attempted: bool = False,
) -> tuple[bytes, dict[str, str]]:
    url = _build_url(base_url, path, query=query)
    request_headers = {'Accept': 'application/json'}
    for key, value in (headers or {}).items():
        clean_key = str(key or '').strip()
        clean_value = str(value or '').strip()
        if clean_key and clean_value:
            request_headers[clean_key] = clean_value
    data = None
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        request_headers['Content-Type'] = 'application/json'
    try:
        request = Request(url, data=data, headers=request_headers, method=method.upper())
        opener = (
            None
            if allow_redirects
            else build_opener(ProxyHandler({}), _RejectRedirectHandler())
        )
        response_context = (
            urlopen(request, timeout=timeout_sec)
            if opener is None
            else opener.open(request, timeout=timeout_sec)
        )
        with response_context as response:
            body = response.read()
            return body, dict(response.headers.items())
    except HTTPError as exc:
        body = exc.read()
        parsed = None
        try:
            parsed = json.loads(body.decode('utf-8'))
        except Exception:
            parsed = None
        message = _extract_error_message(parsed, f'HTTP {exc.code} for {url}')
        raise CliError(message, status_code=exc.code, payload=parsed if isinstance(parsed, dict) else None) from exc
    except URLError as exc:
        if allow_runtime_recovery and not _recovery_attempted and _attempt_local_control_plane_recovery(base_url):
            return _request(
                method,
                base_url,
                path,
                payload=payload,
                query=query,
                headers=headers,
                timeout_sec=timeout_sec,
                allow_runtime_recovery=False,
                allow_redirects=allow_redirects,
                _recovery_attempted=True,
            )
        recovery_note = ''
        if _recovery_attempted and _is_default_local_control_plane(base_url):
            recovery_note = (
                f" Local control-plane auto-start did not make {DEFAULT_BASE_URL} reachable; "
                "check logs/flask_webserver_auto.log or run `./ollmo start`."
            )
        reason = getattr(exc, 'reason', exc)
        raise CliError(f'Could not reach Ollmo at {url}: {reason}.{recovery_note}'.rstrip()) from exc
    except (ValueError, UnicodeError):
        raise CliError(
            'Ollmo request headers or URL could not be encoded safely.'
        ) from None
    except OSError as exc:
        if allow_runtime_recovery and not _recovery_attempted and _attempt_local_control_plane_recovery(base_url):
            return _request(
                method,
                base_url,
                path,
                payload=payload,
                query=query,
                headers=headers,
                timeout_sec=timeout_sec,
                allow_runtime_recovery=False,
                allow_redirects=allow_redirects,
                _recovery_attempted=True,
            )
        recovery_note = ''
        if _recovery_attempted and _is_default_local_control_plane(base_url):
            recovery_note = (
                f" Local control-plane auto-start did not make {DEFAULT_BASE_URL} reachable; "
                "check logs/flask_webserver_auto.log or run `./ollmo start`."
            )
        raise CliError(f'Could not reach Ollmo at {url}: {exc}.{recovery_note}'.rstrip()) from exc


def _request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    query: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_sec: int = 30,
    allow_runtime_recovery: bool = False,
    allow_redirects: bool = True,
) -> dict[str, Any] | list[Any]:
    body, _headers = _request(
        method,
        base_url,
        path,
        payload=payload,
        query=query,
        headers=headers,
        timeout_sec=timeout_sec,
        allow_runtime_recovery=allow_runtime_recovery,
        allow_redirects=allow_redirects,
    )
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode('utf-8'))
    except json.JSONDecodeError as exc:
        raise CliError(f'Ollmo returned invalid JSON for {path}: {exc}') from exc
    if isinstance(parsed, (dict, list)):
        return parsed
    raise CliError(f'Ollmo returned unsupported payload type for {path}: {type(parsed).__name__}')


def _request_bytes(
    method: str,
    base_url: str,
    path: str,
    *,
    query: Optional[dict[str, Any]] = None,
    timeout_sec: int = 60,
    allow_runtime_recovery: bool = False,
) -> tuple[bytes, dict[str, str]]:
    return _request(
        method,
        base_url,
        path,
        query=query,
        timeout_sec=timeout_sec,
        allow_runtime_recovery=allow_runtime_recovery,
    )


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _graph_rebase_token(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_').replace(' ', '_')


def _graph_rebase_contains_wildcard(value: Any) -> bool:
    text = str(value or '').strip()
    return (
        not text
        or '*' in text
        or text.lower() in {'all', 'any', 'current', 'latest'}
    )


def _graph_rebase_safe_header_value(value: str) -> bool:
    return bool(value) and value.isascii() and all(
        0x21 <= ord(character) <= 0x7E for character in value
    )


def _graph_rebase_operator_headers() -> dict[str, str]:
    token = str(os.environ.get(GRAPH_REBASE_OPERATOR_TOKEN_ENV) or '')
    if not token and sys.stdin.isatty():
        try:
            token = getpass.getpass('Graph-rebase operator token: ')
        except (EOFError, KeyboardInterrupt) as exc:
            raise CliError('Graph-rebase operator credential input was cancelled.') from exc
    if len(token) < 32 or not _graph_rebase_safe_header_value(token):
        raise CliError(
            f'{GRAPH_REBASE_OPERATOR_TOKEN_ENV} must provide at least 32 visible '
            'single-line ASCII characters or be entered through the hidden TTY prompt.'
        )

    identity = str(os.environ.get(GRAPH_REBASE_OPERATOR_IDENTITY_ENV) or '').strip()
    if (
        len(identity) > 128
        or _graph_rebase_contains_wildcard(identity)
        or not _graph_rebase_safe_header_value(identity)
    ):
        raise CliError(
            f'{GRAPH_REBASE_OPERATOR_IDENTITY_ENV} must contain one exact visible '
            'single-line ASCII operator identity.'
        )
    return {
        'Authorization': f'Bearer {token}',
        'X-Ollmo-Graph-Rebase-Operator': identity,
    }


def _qualified_graph_rebase_evidence_refs(value: Any) -> list[str]:
    refs = _split_csv_tokens(value if isinstance(value, list) else [value])
    if not refs:
        raise CliError('At least one --evidence-ref is required.')
    for ref in refs:
        if ':' not in ref or _graph_rebase_contains_wildcard(ref):
            raise CliError(
                f"Evidence ref '{ref}' must be exact, qualified as namespace:value, and contain no wildcard."
            )
    return refs


def _validate_graph_rebase_reason(value: Any) -> str:
    reason = str(value or '').strip()
    if _graph_rebase_contains_wildcard(reason):
        raise CliError('An exact non-wildcard --reason is required.')
    return reason


def _graph_rebase_response_truth(
    base_url: str,
    response_id: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    normalized_id = str(response_id or '').strip()
    if not normalized_id:
        raise CliError('Response id is required.')
    payload = _request_json(
        'GET',
        base_url,
        f"/api/responses/{quote(normalized_id, safe='')}",
        query={'view': 'truth'},
        timeout_sec=timeout_sec,
        allow_runtime_recovery=False,
    )
    if not isinstance(payload, dict):
        raise CliError('Unexpected graph-rebase response truth payload from Ollmo.')
    return payload


def _graph_rebase_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _graph_rebase_mapping_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _graph_rebase_formal_truth_present(
    graph: dict[str, Any],
    diagnostics: dict[str, Any],
) -> bool:
    return any(
        _graph_rebase_mapping_records(source.get(key))
        for source, keys in (
            (graph, GRAPH_REBASE_FORMAL_GRAPH_KEYS),
            (diagnostics, GRAPH_REBASE_FORMAL_DIAGNOSTIC_KEYS),
        )
        for key in keys
    )


def _graph_rebase_response_identity(
    payload: dict[str, Any],
    frame: dict[str, Any],
    graph: dict[str, Any],
) -> str:
    candidates: list[str] = []
    for value in (
        payload.get('response_id'),
        payload.get('id'),
        frame.get('response_id'),
        graph.get('response_id'),
    ):
        token = str(value or '').strip()
        if token and token not in candidates:
            candidates.append(token)
    if not candidates:
        raise CliError('Canonical response id is missing from graph-rebase truth.')
    if len(candidates) != 1:
        raise CliError(
            'Canonical response id conflicts across response, frame, and graph truth: '
            + ', '.join(candidates)
        )
    return candidates[0]


def _graph_rebase_frame_binding(frame: dict[str, Any]) -> tuple[str, int]:
    frame_id = str(frame.get('frame_id') or '').strip()
    if _graph_rebase_contains_wildcard(frame_id):
        raise CliError('Canonical latest graph-rebase frame id is missing or not exact.')
    sequence = frame.get('frame_sequence')
    if sequence in (None, ''):
        raise CliError('Canonical latest graph-rebase frame sequence is missing.')
    try:
        frame_sequence = parse_graph_rebase_frame_sequence(sequence)
    except ValueError as exc:
        raise CliError(
            'Canonical latest graph-rebase frame sequence must be a positive JSON integer.'
        ) from exc
    return frame_id, frame_sequence


def _graph_rebase_proposal_summary(
    proposal: dict[str, Any],
    *,
    graph: dict[str, Any],
    reviews: list[dict[str, Any]],
    response_id: str,
    frame_id: str,
    frame_sequence: int,
) -> dict[str, Any]:
    proposal_id = str(proposal.get('proposal_id') or '').strip()
    requested_class = _graph_rebase_token(proposal.get('requested_rebase_class'))
    candidate_graph = _graph_rebase_mapping(proposal.get('candidate_graph'))
    derived_base_digest = stable_graph_digest(graph)
    derived_candidate_digest = stable_graph_digest(candidate_graph) if candidate_graph else ''
    matching_reviews = [
        review
        for review in reviews
        if str(review.get('proposal_id') or '').strip() == proposal_id
    ]
    review = matching_reviews[0] if len(matching_reviews) == 1 else {}
    binding_errors: list[str] = []

    if proposal.get('kind') != 'ollmo.graph_rebase_proposal':
        binding_errors.append('proposal_kind_mismatch')
    if _graph_rebase_contains_wildcard(proposal_id):
        binding_errors.append('proposal_id_missing_or_not_exact')
    if requested_class not in GRAPH_REBASE_CLASSES:
        binding_errors.append('requested_rebase_class_invalid')
    if not candidate_graph:
        binding_errors.append('candidate_graph_missing')
    if str(proposal.get('base_graph_digest') or '').strip() != derived_base_digest:
        binding_errors.append('proposal_base_graph_digest_mismatch')
    if str(proposal.get('candidate_graph_digest') or '').strip() != derived_candidate_digest:
        binding_errors.append('proposal_candidate_graph_digest_mismatch')
    if len(matching_reviews) != 1:
        binding_errors.append(
            'runtime_review_missing'
            if not matching_reviews
            else 'runtime_review_binding_ambiguous'
        )
    elif (
        str(review.get('base_graph_digest') or '').strip() != derived_base_digest
        or str(review.get('candidate_graph_digest') or '').strip()
        != derived_candidate_digest
    ):
        binding_errors.append('runtime_review_graph_digest_mismatch')

    preservation_proof = _graph_rebase_mapping(review.get('preservation_proof'))
    execution_contract_proof = _graph_rebase_mapping(
        review.get('execution_contract_proof')
    )
    return {
        'proposal_id': proposal_id or None,
        'requested_rebase_class': requested_class or None,
        'binding_valid': not binding_errors,
        'binding_errors': binding_errors,
        'cas': {
            'expected_response_id': response_id,
            'expected_frame_id': frame_id,
            'expected_frame_sequence': frame_sequence,
            'expected_proposal_id': proposal_id,
            'expected_base_graph_digest': derived_base_digest,
            'expected_candidate_graph_digest': derived_candidate_digest,
            'expected_requested_rebase_class': requested_class,
        },
        'scope': {
            'scope_root_ids': proposal.get('scope_root_ids') or [],
            'scope_phase_ids': proposal.get('scope_phase_ids') or [],
            'scope_branch_ids': proposal.get('scope_branch_ids') or [],
            'scope_artifact_refs': proposal.get('scope_artifact_refs') or [],
            'preserve_outside_scope': proposal.get('preserve_outside_scope'),
        },
        'runtime_review': {
            'review_id': review.get('review_id'),
            'status': review.get('status'),
            'blocked_reasons': review.get('blocked_reasons') or [],
            'allowed_runtime_action': review.get('allowed_runtime_action'),
            'diff': review.get('diff') or {},
            'preservation_proof': preservation_proof,
            'execution_contract_proof': execution_contract_proof,
        },
        'eligible': {
            'adjudicate': not binding_errors,
            'stage': (
                not binding_errors
                and _graph_rebase_token(review.get('status')) == 'accepted'
                and _graph_rebase_token(preservation_proof.get('status')) == 'passed'
            ),
            'authorize_partial': (
                not binding_errors
                and requested_class == 'partial_subtree_rebase'
                and _graph_rebase_token(review.get('status')) == 'accepted'
                and _graph_rebase_token(preservation_proof.get('status')) == 'passed'
                and _graph_rebase_token(execution_contract_proof.get('status'))
                == 'passed'
            ),
        },
    }


def _build_graph_rebase_inspection(
    payload: dict[str, Any],
    *,
    proposal_id: str = '',
    expected_response_id: str = '',
) -> dict[str, Any]:
    frame = _graph_rebase_mapping(payload.get('response_frame'))
    if frame.get('kind') != 'ollmo.response_frame':
        raise CliError('Canonical frozen response frame is required for graph-rebase inspection.')
    frame_current_state = _graph_rebase_mapping(frame.get('current_state'))
    api_lifecycle_state = _graph_rebase_token(payload.get('lifecycle_state'))
    frozen_lifecycle_state = _graph_rebase_token(
        frame_current_state.get('lifecycle_state') or frame.get('lifecycle_state')
    )
    if (
        api_lifecycle_state
        and frozen_lifecycle_state
        and api_lifecycle_state != frozen_lifecycle_state
    ):
        raise CliError(
            'Canonical API lifecycle truth disagrees with the frozen response-frame '
            'current state; graph-rebase inspection is unsafe until Runtime truth converges.'
        )
    canonical_lifecycle_state = (
        api_lifecycle_state
        or frozen_lifecycle_state
        or _graph_rebase_token(payload.get('status'))
    )
    runtime = _graph_rebase_mapping(payload.get('runtime'))
    graph = _graph_rebase_mapping(runtime.get('request_phase_graph'))
    if not graph:
        raise CliError('Canonical request phase graph is required for graph-rebase inspection.')
    diagnostics = _graph_rebase_mapping(runtime.get('developer_diagnostics'))
    response_id = _graph_rebase_response_identity(payload, frame, graph)
    expected_id = str(expected_response_id or '').strip()
    if expected_id and response_id != expected_id:
        raise CliError(
            'Canonical graph-rebase response id does not match the exact requested response id.'
        )
    frame_id, frame_sequence = _graph_rebase_frame_binding(frame)
    proposals = _graph_rebase_mapping_records(graph.get('graph_rebase_proposals'))
    reviews = _graph_rebase_mapping_records(graph.get('graph_rebase_reviews'))
    summaries = [
        _graph_rebase_proposal_summary(
            proposal,
            graph=graph,
            reviews=reviews,
            response_id=response_id,
            frame_id=frame_id,
            frame_sequence=frame_sequence,
        )
        for proposal in proposals
    ]
    proposal_ids = [
        str(item.get('proposal_id') or '').strip()
        for item in summaries
        if str(item.get('proposal_id') or '').strip()
    ]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise CliError('Graph-rebase proposal ids are ambiguous in current runtime truth.')

    requested_proposal_id = str(proposal_id or '').strip()
    selected: dict[str, Any] = {}
    if requested_proposal_id:
        matches = [
            item
            for item in summaries
            if item.get('proposal_id') == requested_proposal_id
        ]
        if len(matches) != 1:
            raise CliError(
                f"Exactly one current graph-rebase proposal must match '{requested_proposal_id}'."
            )
        selected = matches[0]
    elif len(summaries) == 1:
        selected = summaries[0]
    elif len(summaries) > 1:
        raise CliError(
            'Multiple current graph-rebase proposals exist; select one with --proposal-id. '
            f"Candidates: {', '.join(proposal_ids)}"
        )

    candidate_review = _graph_rebase_mapping(
        diagnostics.get('runtime_graph_rebase_candidate_review')
    )
    candidate_base_digest = str(
        candidate_review.get('base_graph_digest') or ''
    ).strip()
    derived_base_digest = stable_graph_digest(graph)
    candidate_binding_errors: list[str] = []
    if candidate_base_digest and candidate_base_digest != derived_base_digest:
        candidate_binding_errors.append('candidate_review_base_graph_digest_mismatch')
    candidate_digest = str(
        candidate_review.get('candidate_graph_digest') or ''
    ).strip()
    formal_truth_present = _graph_rebase_formal_truth_present(graph, diagnostics)
    false_negative_eligible = bool(
        not formal_truth_present
        and candidate_review.get('kind')
        == 'ollmo.runtime_graph_rebase_candidate_review'
        and _graph_rebase_token(candidate_review.get('status')) == 'not_proposed'
        and _graph_rebase_token(candidate_review.get('runtime_effect')) == 'none'
        and str(candidate_review.get('reason') or '').strip()
        and candidate_digest
        and not candidate_binding_errors
        and canonical_lifecycle_state in GRAPH_REBASE_SETTLED_LIFECYCLE_STATES
    )
    relation = _graph_rebase_mapping(frame.get('frame_relation'))
    staged = _graph_rebase_mapping_records(graph.get('staged_graph_rebases'))
    return {
        'kind': 'ollmo.ollmoctl_graph_rebase_inspection',
        'runtime_effect': 'none',
        'response_id': response_id,
        'lifecycle_state': canonical_lifecycle_state,
        'frame': {
            'frame_id': frame_id,
            'frame_sequence': frame_sequence,
            'relation': relation,
        },
        'redraw_scope_ladder_review': graph.get('redraw_scope_ladder_review') or {},
        'proposal_count': len(summaries),
        'proposals': summaries,
        'selected_proposal': selected or None,
        'candidate_opportunity': {
            'review': candidate_review,
            'derived_base_graph_digest': derived_base_digest,
            'candidate_graph_digest': candidate_digest or None,
            'binding_errors': candidate_binding_errors,
            'formal_rebase_truth_present': formal_truth_present,
            'false_negative_eligible': false_negative_eligible,
        },
        'staged_graph_rebases': staged,
    }


def _selected_graph_rebase_cas(
    inspection: dict[str, Any],
    *,
    rebase_class: str = '',
    required_action: str = '',
) -> dict[str, Any]:
    selected = _graph_rebase_mapping(inspection.get('selected_proposal'))
    if not selected:
        raise CliError('A current formal graph-rebase proposal is required for this action.')
    if selected.get('binding_valid') is not True:
        reasons = ', '.join(selected.get('binding_errors') or []) or 'unknown binding error'
        raise CliError(f'Current graph-rebase proposal is not exactly bound: {reasons}.')
    action = _graph_rebase_token(required_action)
    if (
        action == 'authorize_partial'
        and _graph_rebase_token(
            _graph_rebase_mapping(selected.get('cas')).get(
                'expected_requested_rebase_class'
            )
        )
        == 'full_successor_rebase'
    ):
        raise CliError('authorize-partial cannot authorize a full successor rebase.')
    if action and _graph_rebase_mapping(selected.get('eligible')).get(action) is not True:
        raise CliError(
            f"Current runtime review/proofs do not permit graph-rebase action '{action}'."
        )
    cas = _graph_rebase_mapping(selected.get('cas'))
    requested_class = _graph_rebase_token(rebase_class)
    if requested_class and requested_class != cas.get('expected_requested_rebase_class'):
        raise CliError(
            '--rebase-class does not match the selected runtime proposal class.'
        )
    return cas


def _false_negative_graph_rebase_cas(
    inspection: dict[str, Any],
    *,
    rebase_class: str,
) -> dict[str, Any]:
    requested_class = _graph_rebase_token(rebase_class)
    if requested_class not in GRAPH_REBASE_CLASSES:
        raise CliError(
            'False-negative adjudication requires --rebase-class '
            'partial_subtree_rebase or full_successor_rebase.'
        )
    candidate = _graph_rebase_mapping(inspection.get('candidate_opportunity'))
    if candidate.get('false_negative_eligible') is not True:
        raise CliError(
            'Current truth is not an exact settled no-proposal candidate eligible for false-negative adjudication.'
        )
    frame = _graph_rebase_mapping(inspection.get('frame'))
    return {
        'expected_response_id': inspection.get('response_id'),
        'expected_frame_id': frame.get('frame_id'),
        'expected_frame_sequence': frame.get('frame_sequence'),
        'expected_proposal_id': GRAPH_REBASE_NO_FORMAL_PROPOSAL,
        'expected_base_graph_digest': candidate.get('derived_base_graph_digest'),
        'expected_candidate_graph_digest': candidate.get('candidate_graph_digest'),
        'expected_requested_rebase_class': requested_class,
    }


def _confirm_graph_rebase_action(
    *,
    action: str,
    cas: dict[str, Any],
    assume_yes: bool,
    scope: Optional[dict[str, Any]] = None,
    gate_name: str = '',
    gate: Optional[dict[str, Any]] = None,
    readiness_report_digest: str = '',
) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise CliError(f'Non-interactive graph-rebase {action} requires --yes.')
    effect = {
        'adjudicate': 'append trusted operator review truth; execute no response work',
        'stage': 'append audit-only stage truth; execute no response work',
        'authorize_partial': (
            'authorize and immediately queue the exact branch-local partial successor'
        ),
    }.get(action, action)
    bounded_scope = _graph_rebase_mapping(scope)
    gate_payload = _graph_rebase_mapping(gate)
    print(
        f"Confirm graph-rebase {action}: {effect}.\n"
        f"  response={cas.get('expected_response_id')}\n"
        f"  frame={cas.get('expected_frame_id')} seq={cas.get('expected_frame_sequence')}\n"
        f"  proposal={cas.get('expected_proposal_id')}\n"
        f"  class={cas.get('expected_requested_rebase_class')}\n"
        f"  scope_roots={','.join(bounded_scope.get('scope_root_ids') or []) or '-'}\n"
        f"  scope_phases={','.join(bounded_scope.get('scope_phase_ids') or []) or '-'}\n"
        f"  scope_branches={','.join(bounded_scope.get('scope_branch_ids') or []) or '-'}\n"
        f"  preserve_outside_scope={bounded_scope.get('preserve_outside_scope')}\n"
        f"  gate={gate_name or '-'} decision={gate_payload.get('decision') or '-'} "
        f"report={readiness_report_digest or '-'}\n"
        'Type yes to continue: ',
        file=sys.stderr,
        end='',
        flush=True,
    )
    if str(sys.stdin.readline() or '').strip().lower() != 'yes':
        raise CliError('Graph-rebase operator action cancelled.')


def _graph_rebase_readiness(
    base_url: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    payload = _request_json(
        'GET',
        base_url,
        '/api/graph_rebase/readiness',
        timeout_sec=timeout_sec,
        allow_runtime_recovery=False,
    )
    if not isinstance(payload, dict):
        raise CliError('Unexpected graph-rebase readiness payload from Ollmo.')
    return payload


def _require_graph_rebase_gate(
    readiness: dict[str, Any],
    gate_name: str,
) -> dict[str, Any]:
    observer = _graph_rebase_mapping(readiness.get('observer'))
    if int(observer.get('load_error_count') or 0) > 0:
        raise CliError(
            'Graph-rebase readiness corpus is not fully readable.',
            status_code=409,
            payload={'observer': observer},
        )
    gate = _graph_rebase_mapping(
        _graph_rebase_mapping(readiness.get('gates')).get(gate_name)
    )
    if gate.get('ready') is not True:
        raise CliError(
            f"Graph-rebase gate '{gate_name}' is not ready.",
            status_code=409,
            payload={
                'gate': gate,
                'readiness_report_digest': readiness.get('report_digest'),
            },
        )
    return gate


def _normalize_lifecycle_token(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    normalized = value.strip().lower()
    return normalized if normalized in KNOWN_RESPONSE_LIFECYCLE_STATES else ''


def _canonical_lifecycle_from_response(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return '', ''
    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), dict)
        else {}
    )
    current_state = (
        response_frame.get('current_state')
        if isinstance(response_frame.get('current_state'), dict)
        else {}
    )
    frame_lifecycle = _normalize_lifecycle_token(
        current_state.get('lifecycle_state')
        or current_state.get('lifecycleState')
    )
    if frame_lifecycle:
        return frame_lifecycle, 'response_frame.current_state.lifecycle_state'
    lifecycle_state = _normalize_lifecycle_token(payload.get('lifecycle_state') or payload.get('lifecycleState'))
    if lifecycle_state:
        return lifecycle_state, 'lifecycle_state'
    status_semantics = payload.get('status_semantics') or payload.get('statusSemantics') or {}
    if isinstance(status_semantics, dict):
        semantic_lifecycle = _normalize_lifecycle_token(
            status_semantics.get('canonical_lifecycle_state')
            or status_semantics.get('canonicalLifecycleState')
        )
        if semantic_lifecycle:
            return semantic_lifecycle, 'status_semantics.canonical_lifecycle_state'
    return '', ''


def _legacy_late_fill_status(payload: dict[str, Any]) -> str:
    late_fill = payload.get('late_fill') or payload.get('lateFill') or {}
    if not isinstance(late_fill, dict):
        return ''
    return _normalize_lifecycle_token(late_fill.get('status'))


def _bool_from_semantics(status_semantics: dict[str, Any], snake_key: str, camel_key: str) -> Optional[bool]:
    if snake_key in status_semantics:
        value = status_semantics.get(snake_key)
        return value if isinstance(value, bool) else None
    if camel_key in status_semantics:
        value = status_semantics.get(camel_key)
        return value if isinstance(value, bool) else None
    return None


def _response_has_open_continuation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    lifecycle_state, _source = _canonical_lifecycle_from_response(payload)
    if lifecycle_state:
        return lifecycle_state in OPEN_RESPONSE_LIFECYCLE_STATES

    status_semantics = payload.get('status_semantics') or payload.get('statusSemantics') or {}
    if isinstance(status_semantics, dict):
        semantic_open = _bool_from_semantics(status_semantics, 'has_open_continuation', 'hasOpenContinuation')
        if semantic_open is not None:
            return semantic_open
        semantic_terminal = _bool_from_semantics(status_semantics, 'is_terminal', 'isTerminal')
        if semantic_terminal:
            return False

    return _legacy_late_fill_status(payload) in LEGACY_ACTIVE_LATE_FILL_STATUSES


def _response_has_actionable_repair(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    lifecycle_state, _source = _canonical_lifecycle_from_response(payload)
    if lifecycle_state in ACTIONABLE_RESPONSE_LIFECYCLE_STATES:
        return True
    status_semantics = payload.get('status_semantics') or payload.get('statusSemantics') or {}
    if isinstance(status_semantics, dict):
        semantic_repair = _bool_from_semantics(status_semantics, 'has_actionable_repair', 'hasActionableRepair')
        if semantic_repair is not None:
            return semantic_repair
    late_fill = payload.get('late_fill') or payload.get('lateFill') or {}
    if not isinstance(late_fill, dict):
        return False
    if _legacy_late_fill_status(payload) in ACTIONABLE_RESPONSE_LIFECYCLE_STATES:
        return True
    for key in ('repair_action', 'repair_actions', 'recovery_candidates'):
        value = late_fill.get(key)
        if value not in (None, '', [], {}):
            return True
    return False


def _response_is_terminal(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    lifecycle_state, _source = _canonical_lifecycle_from_response(payload)
    if lifecycle_state:
        return lifecycle_state in TERMINAL_RESPONSE_LIFECYCLE_STATES
    status_semantics = payload.get('status_semantics') or payload.get('statusSemantics') or {}
    if isinstance(status_semantics, dict):
        semantic_terminal = _bool_from_semantics(status_semantics, 'is_terminal', 'isTerminal')
        if semantic_terminal is not None:
            return semantic_terminal
    return _normalize_lifecycle_token(payload.get('status')) in TERMINAL_RESPONSE_LIFECYCLE_STATES


def _response_frame_truth(payload: dict[str, Any]) -> dict[str, Any]:
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), dict) else {}
    relation = frame.get('frame_relation') if isinstance(frame.get('frame_relation'), dict) else {}
    if not relation and isinstance(payload.get('frame_relation'), dict):
        relation = payload.get('frame_relation') or {}
    return {
        'frame_id': frame.get('frame_id'),
        'frame_sequence': frame.get('frame_sequence'),
        'frame_relation': relation.get('kind') or frame.get('frame_relation_kind'),
        'parent_frame_id': relation.get('parent_frame_id') or frame.get('parent_frame_id'),
        'parent_frame_sequence': relation.get('parent_frame_sequence') or frame.get('parent_frame_sequence'),
    }


def _response_identity(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ''
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), dict) else {}
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), dict) else {}
    identities = {
        str(value).strip()
        for value in (
            payload.get('id'),
            payload.get('response_id'),
            frame.get('response_id'),
            current_state.get('id'),
        )
        if str(value or '').strip()
    }
    if len(identities) > 1:
        raise CliError(
            'Canonical response truth contains conflicting response identities.',
            status_code=409,
            payload={'response_ids': sorted(identities)},
        )
    return next(iter(identities), '')


def _complete_response_frame_identity(payload: Any) -> tuple[str, int] | None:
    if not isinstance(payload, dict):
        return None
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), dict) else {}
    frame_id = str(frame.get('frame_id') or '').strip()
    frame_sequence = frame.get('frame_sequence')
    if (
        not frame_id
        or not isinstance(frame_sequence, int)
        or isinstance(frame_sequence, bool)
        or frame_sequence <= 0
    ):
        return None
    return frame_id, frame_sequence


def _require_response_truth_binding(
    expected_response_id: str,
    truth_payload: Any,
    *,
    minimum_frame_payload: Any = None,
) -> None:
    actual_response_id = _response_identity(truth_payload)
    if actual_response_id != expected_response_id:
        raise CliError(
            'Canonical response truth does not match the requested response id.',
            status_code=409,
            payload={
                'expected_response_id': expected_response_id,
                'actual_response_id': actual_response_id or None,
            },
        )
    minimum_identity = _complete_response_frame_identity(minimum_frame_payload)
    if minimum_identity is None:
        return
    truth_identity = _complete_response_frame_identity(truth_payload)
    if truth_identity is None:
        raise CliError(
            'Canonical response truth is missing the frozen frame identity returned by POST.',
            status_code=409,
            payload={
                'expected_frame_id': minimum_identity[0],
                'expected_minimum_frame_sequence': minimum_identity[1],
            },
        )
    if (
        truth_identity[1] < minimum_identity[1]
        or (
            truth_identity[1] == minimum_identity[1]
            and truth_identity[0] != minimum_identity[0]
        )
    ):
        raise CliError(
            'Canonical response truth regressed behind the frame returned by POST.',
            status_code=409,
            payload={
                'post_frame_id': minimum_identity[0],
                'post_frame_sequence': minimum_identity[1],
                'truth_frame_id': truth_identity[0],
                'truth_frame_sequence': truth_identity[1],
            },
        )


def _response_outputs_truth(payload: dict[str, Any]) -> dict[str, Any]:
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), dict) else {}
    output_frame = frame.get('output') if isinstance(frame.get('output'), dict) else {}
    outputs = output_frame.get('outputs') if isinstance(output_frame.get('outputs'), list) else []
    if not outputs:
        outputs = payload.get('outputs') if isinstance(payload.get('outputs'), list) else []

    canonical_count = 0
    compatibility_count = 0
    unknown_count = 0
    sources: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        source = str(item.get('source') or '').strip() or 'unknown'
        if source not in sources:
            sources.append(source)
        if bool(item.get('compatibility_derived')) or source == 'compatibility_derived':
            compatibility_count += 1
        elif source == 'unknown':
            unknown_count += 1
        else:
            canonical_count += 1
    return {
        'count': len(outputs),
        'canonical_count': canonical_count,
        'compatibility_derived_count': compatibility_count,
        'unknown_provenance_count': unknown_count,
        'sources': sources,
    }


def _response_work_tree_truth(payload: dict[str, Any]) -> dict[str, Any]:
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), dict) else {}
    planning = frame.get('planning') if isinstance(frame.get('planning'), dict) else {}
    artifact_flow = planning.get('artifact_flow') if isinstance(planning.get('artifact_flow'), dict) else {}
    work_tree = artifact_flow.get('work_tree') if isinstance(artifact_flow.get('work_tree'), dict) else {}
    return {
        'work_tree_source': work_tree.get('work_tree_source') or artifact_flow.get('work_tree_source'),
        'authoritative': work_tree.get('authoritative'),
        'compatibility_derived': work_tree.get('compatibility_derived'),
    }


def _response_message_identity(payload: dict[str, Any]) -> str:
    direct = str(payload.get('message_id') or payload.get('messageId') or '').strip()
    if direct:
        return direct
    frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), dict) else {}
    current_state = frame.get('current_state') if isinstance(frame.get('current_state'), dict) else {}
    for candidate in (
        payload.get('output'),
        current_state.get('output'),
    ):
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            if not isinstance(item, dict):
                continue
            message_id = str(item.get('id') or '').strip()
            if message_id:
                return message_id
    return ''


def _build_response_truth_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CliError('Response truth summary requires a JSON object response payload.')
    status_semantics = payload.get('status_semantics') or payload.get('statusSemantics') or {}
    if not isinstance(status_semantics, dict):
        status_semantics = {}
    lifecycle_state, lifecycle_source = _canonical_lifecycle_from_response(payload)
    status = _normalize_lifecycle_token(payload.get('status'))
    return {
        'response_id': payload.get('id') or payload.get('response_id'),
        'message_id': _response_message_identity(payload) or None,
        'compatibility_status': status or None,
        'lifecycle_state': lifecycle_state or None,
        'canonical_lifecycle_source': lifecycle_source or 'legacy_fallback',
        'canonical_status_field': payload.get('canonical_status_field') or status_semantics.get('canonical_status_field'),
        'status_compatibility': bool(payload.get('status_compatibility') or status_semantics.get('status_compatibility')),
        'has_open_continuation': _response_has_open_continuation(payload),
        'has_actionable_repair': _response_has_actionable_repair(payload),
        'is_terminal': _response_is_terminal(payload),
        'late_fill_status': _legacy_late_fill_status(payload) or None,
        'response_frame': _response_frame_truth(payload),
        'outputs': _response_outputs_truth(payload),
        'work_tree': _response_work_tree_truth(payload),
        'durability': payload.get('durability') if isinstance(payload.get('durability'), dict) else {},
        'lookup_source': payload.get('lookup_source'),
    }


def _format_yes_no(value: Any) -> str:
    return 'yes' if bool(value) else 'no'


def _print_response_truth_summary(payload: dict[str, Any]) -> None:
    truth = _build_response_truth_summary(payload)
    print(f"response_id: {truth.get('response_id') or 'unknown'}")
    print(f"status: {truth.get('compatibility_status') or 'unknown'} (compatibility)")
    lifecycle_state = truth.get('lifecycle_state') or 'unknown'
    lifecycle_source = truth.get('canonical_lifecycle_source') or 'unknown'
    print(f"lifecycle_state: {lifecycle_state} (canonical via {lifecycle_source})")
    print(f"open_continuation: {_format_yes_no(truth.get('has_open_continuation'))}")
    print(f"actionable_repair: {_format_yes_no(truth.get('has_actionable_repair'))}")
    print(f"terminal: {_format_yes_no(truth.get('is_terminal'))}")
    frame = truth.get('response_frame') if isinstance(truth.get('response_frame'), dict) else {}
    print(
        'response_frame: '
        f"{frame.get('frame_id') or 'unknown'} "
        f"seq={frame.get('frame_sequence') if frame.get('frame_sequence') is not None else 'unknown'} "
        f"relation={frame.get('frame_relation') or 'unknown'} "
        f"parent={frame.get('parent_frame_id') or '-'}"
    )
    outputs = truth.get('outputs') if isinstance(truth.get('outputs'), dict) else {}
    sources = ','.join(outputs.get('sources') or []) or 'none'
    print(
        'outputs: '
        f"count={outputs.get('count', 0)} "
        f"canonical={outputs.get('canonical_count', 0)} "
        f"compatibility_derived={outputs.get('compatibility_derived_count', 0)} "
        f"sources={sources}"
    )
    work_tree = truth.get('work_tree') if isinstance(truth.get('work_tree'), dict) else {}
    if any(value not in (None, '', [], {}) for value in work_tree.values()):
        print(
            'work_tree: '
            f"source={work_tree.get('work_tree_source') or 'unknown'} "
            f"authoritative={_format_yes_no(work_tree.get('authoritative'))} "
            f"compatibility_derived={_format_yes_no(work_tree.get('compatibility_derived'))}"
        )
    durability = truth.get('durability') if isinstance(truth.get('durability'), dict) else {}
    if durability:
        source = durability.get('source') or 'unknown'
        recovered = _format_yes_no(durability.get('recovered'))
        print(f"durability: source={source} recovered={recovered}")


def _normalize_list_tokens(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        token = str(raw or '').strip()
        if token and token not in items:
            items.append(token)
    return items


def _summarize_feature_contract(item: dict[str, Any]) -> str:
    inputs = _normalize_list_tokens(item.get('inputs'))
    outputs = _normalize_list_tokens(item.get('outputs'))
    features = item.get('features') if isinstance(item.get('features'), dict) else {}
    enabled = [
        key for key, value in features.items()
        if bool(value) and key not in {'vision_input', 'audio_input', 'image_output', 'audio_output'}
    ]
    parts: list[str] = []
    if inputs:
        parts.append(f"in={','.join(inputs)}")
    if outputs:
        parts.append(f"out={','.join(outputs)}")
    if enabled:
        parts.append(f"feat={','.join(enabled)}")
    return ' '.join(parts)


def _path_snapshot(path_text: str) -> dict[str, Any]:
    raw = str(path_text or '').strip()
    path = Path(raw).expanduser()
    exists = path.exists()
    try:
        resolved = str(path.resolve(strict=False))
    except Exception:
        resolved = str(path)
    return {'path': str(path), 'exists': exists, 'resolved': resolved}


def _run_local_command(command: list[str], *, timeout_sec: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=sanitized_child_process_env(),
        )
    except FileNotFoundError:
        return {
            'ok': False,
            'returncode': None,
            'stdout': '',
            'stderr': f'command not found: {command[0]}',
        }
    except Exception as exc:  # noqa: BLE001
        return {
            'ok': False,
            'returncode': None,
            'stdout': '',
            'stderr': str(exc),
        }
    return {
        'ok': completed.returncode == 0,
        'returncode': completed.returncode,
        'stdout': completed.stdout,
        'stderr': completed.stderr,
    }


def _which_all(command_name: str) -> list[str]:
    result = _run_local_command(['which', '-a', command_name], timeout_sec=10)
    if not result['stdout']:
        return []
    return [line.strip() for line in str(result['stdout']).splitlines() if line.strip()]


def _brew_service_status(service_name: str) -> dict[str, Any]:
    result = _run_local_command(['brew', 'services', 'list'], timeout_sec=15)
    payload: dict[str, Any] = {
        'available': result['returncode'] is not None,
        'status': None,
        'raw_line': None,
        'error': str(result.get('stderr') or '').strip() or None,
    }
    if not result['stdout']:
        return payload
    for line in str(result['stdout']).splitlines():
        parts = line.split()
        if parts and parts[0] == service_name:
            payload['raw_line'] = line.strip()
            payload['status'] = parts[1] if len(parts) > 1 else None
            if len(parts) > 2:
                payload['user'] = parts[2]
            if len(parts) > 3:
                payload['plist'] = parts[3]
            break
    return payload


def _collect_mlx_runtime_versions(python_path: str) -> dict[str, Any]:
    snapshot = _path_snapshot(python_path)
    payload: dict[str, Any] = {
        'python': snapshot,
        'python_version': None,
        'packages': {
            'mlx': None,
            'mlx-lm': None,
            'mlx-vlm': None,
            'mlx-whisper': None,
            'mlx-audio': None,
        },
        'error': None,
    }
    if not snapshot['exists']:
        payload['error'] = 'MLX python not found.'
        return payload

    code = (
        'import importlib.metadata as md, json, sys\n'
        "packages = ['mlx', 'mlx-lm', 'mlx-vlm', 'mlx-whisper', 'mlx-audio']\n"
        "result = {'python_version': sys.version.split()[0], 'packages': {}}\n"
        'for name in packages:\n'
        '    try:\n'
        "        result['packages'][name] = md.version(name)\n"
        '    except md.PackageNotFoundError:\n'
        "        result['packages'][name] = None\n"
        'print(json.dumps(result))\n'
    )
    result = _run_local_command([snapshot['path'], '-c', code], timeout_sec=20)
    if not result['ok']:
        payload['error'] = str(result.get('stderr') or result.get('stdout') or 'Failed to inspect MLX packages.').strip()
        return payload
    try:
        parsed = json.loads(str(result['stdout']).strip() or '{}')
    except json.JSONDecodeError as exc:
        payload['error'] = f'Failed to parse MLX package output: {exc}'
        return payload
    if isinstance(parsed, dict):
        payload['python_version'] = parsed.get('python_version')
        packages = parsed.get('packages')
        if isinstance(packages, dict):
            for name in payload['packages']:
                payload['packages'][name] = packages.get(name)
    return payload


def _collect_running_instances_snapshot(base_url: str, timeout_sec: int) -> dict[str, Any]:
    try:
        payload = _request_json(
            'GET',
            base_url,
            '/api/running_instances',
            timeout_sec=timeout_sec,
            allow_runtime_recovery=False,
        )
    except CliError as exc:
        return {
            'reachable': False,
            'error': str(exc),
            'count': 0,
            'instances': [],
        }
    if not isinstance(payload, list):
        return {
            'reachable': False,
            'error': 'Unexpected running-instances payload from Ollmo.',
            'count': 0,
            'instances': [],
        }
    items: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        runtime_status = item.get('runtime_status') if isinstance(item.get('runtime_status'), dict) else {}
        items.append(
            {
                'instance_id': item.get('instance_id'),
                'model': item.get('model') or item.get('modelName'),
                'backend': item.get('backend'),
                'capability': item.get('capability'),
                'port': item.get('port'),
                'readiness': runtime_status.get('readiness') or item.get('readiness'),
                'activity': runtime_status.get('activity') or item.get('activity'),
                'last_error': runtime_status.get('last_error'),
            }
        )
    return {
        'reachable': True,
        'error': None,
        'count': len(items),
        'instances': items,
    }


def _collect_mlx_server_runtime_checks() -> dict[str, Any]:
    try:
        from ollmo_runtime.mlx_model_manager import _mlx_package_runtime_check
    except ImportError:
        return {}

    checks: dict[str, Any] = {}
    for server_kind in ('mlx_lm', 'mlx_vlm', 'mlx_audio', 'mlx_whisper'):
        try:
            checks[server_kind] = dict(_mlx_package_runtime_check(server_kind))
        except Exception as exc:  # noqa: BLE001
            checks[server_kind] = {
                'python_resolved': False,
                'runtime_module_available': False,
                'error': str(exc),
            }
    return checks


def _build_runtime_doctor_payload(base_url: str, timeout_sec: int) -> dict[str, Any]:
    configured = _path_snapshot(DEFAULT_OLLAMA_CLI)
    opt_path = _path_snapshot(DEFAULT_OLLAMA_OPT_CLI)
    which_paths = _which_all('ollama')
    brew_service = _brew_service_status('ollama')
    mlx = _collect_mlx_runtime_versions(DEFAULT_MLX_PYTHON)
    mlx_servers = _collect_mlx_server_runtime_checks()
    runtime = _collect_running_instances_snapshot(base_url, timeout_sec)

    issues: list[str] = []
    ownership_conflict = brew_service.get('status') == 'started'
    if ownership_conflict:
        issues.append("Homebrew service 'ollama' is started. Run `brew services stop ollama` before starting Ollmo.")
    if not configured.get('exists'):
        issues.append(f"Configured Ollama CLI not found: {configured.get('path')}")
    if not runtime.get('reachable'):
        issues.append(f"Ollmo runtime not reachable: {runtime.get('error')}")
    if not mlx['python'].get('exists'):
        issues.append(f"MLX python not found: {mlx['python'].get('path')}")
    for package_name in ('mlx', 'mlx-lm'):
        if not mlx['packages'].get(package_name):
            issues.append(f"Missing required MLX package in {mlx['python'].get('path')}: {package_name}")
    for server_kind, server_checks in mlx_servers.items():
        if not isinstance(server_checks, dict):
            continue
        if not server_checks.get('python_resolved'):
            message = (
                server_checks.get('python_error')
                or server_checks.get('error')
                or f"No Python interpreter resolved for {server_kind}."
            )
            issues.append(f"MLX runtime unavailable for {server_kind}: {message}")
            continue
        if not server_checks.get('runtime_module_available'):
            issues.append(
                f"MLX runtime module unavailable for {server_kind}: "
                f"{server_checks.get('required_runtime_module') or 'unknown module'} "
                f"via {server_checks.get('python_path') or 'unknown python'}"
            )
            continue
        if not server_checks.get('runtime_dependencies_ready', True):
            issues.append(
                f"MLX runtime dependency unavailable for {server_kind}: "
                f"{server_checks.get('runtime_dependency_error') or 'dependency import failed'}"
            )
            continue
        if server_kind == 'mlx_whisper' and not server_checks.get('server_script_present', True):
            issues.append(
                f"MLX runtime unavailable for {server_kind}: missing shim script "
                f"{server_checks.get('server_script_path') or ''}".strip()
            )
    for instance in runtime.get('instances', []):
        readiness = str(instance.get('readiness') or '').strip().lower()
        if readiness in {'failed', 'degraded', 'unreachable'}:
            issues.append(
                f"Instance {instance.get('instance_id')} is {readiness}"
                + (f": {instance.get('last_error')}" if instance.get('last_error') else '')
            )

    return {
        'ok': not issues,
        'issues': issues,
        'ollama': {
            'configured_path': configured,
            'opt_path': opt_path,
            'which': which_paths,
            'same_binary': bool(configured.get('resolved') and configured.get('resolved') == opt_path.get('resolved')),
            'brew_service': brew_service,
            'ownership_conflict': ownership_conflict,
        },
        'mlx': mlx,
        'mlx_servers': mlx_servers,
        'runtime': runtime,
    }


def _split_csv_tokens(values: Optional[list[str]]) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for token in str(value or '').split(','):
            item = token.strip()
            if item:
                out.append(item)
    return out


def _coerce_instance_lookup(
    base_url: str,
    instance_selector: str,
    *,
    allow_runtime_recovery: bool = True,
) -> dict[str, Any]:
    payload = _request_json(
        'GET',
        base_url,
        '/api/running_instances',
        timeout_sec=30,
        allow_runtime_recovery=allow_runtime_recovery,
    )
    if not isinstance(payload, list):
        raise CliError('Unexpected running-instances payload from Ollmo.')
    selector = str(instance_selector or '').strip()
    if not selector:
        raise CliError('Instance selector is required.')

    exact_matches: list[dict[str, Any]] = []
    model_matches: list[dict[str, Any]] = []
    prefix_matches: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(item, dict):
            continue
        instance_id = str(item.get('instance_id') or '').strip()
        model_name = str(item.get('model') or item.get('modelName') or '').strip()
        if selector == instance_id:
            exact_matches.append(item)
        elif selector == model_name:
            model_matches.append(item)
        elif instance_id.startswith(selector) or model_name.startswith(selector):
            prefix_matches.append(item)

    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(model_matches) == 1:
        return model_matches[0]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    if len(model_matches) > 1 or len(prefix_matches) > 1:
        candidates = model_matches or prefix_matches
        names = ', '.join(str(item.get('instance_id') or '') for item in candidates)
        raise CliError(f"Selector '{selector}' is ambiguous. Matching instances: {names}")

    raise CliError(
        f"Instance '{selector}' not found. "
        "Use `ollmoctl instances list --json` to inspect current running instance IDs."
    )


def _resolve_local_file(path_text: Optional[str]) -> str:
    raw = str(path_text or '').strip()
    if not raw:
        return ''
    resolved = Path(raw).expanduser().resolve()
    if not resolved.exists():
        raise CliError(f'Local file not found: {resolved}')
    if resolved.is_dir():
        raise CliError(f'Expected a file path, received a directory: {resolved}')
    return str(resolved)


def _fetch_event_items(
    base_url: str,
    *,
    limit: int,
    category: Optional[str] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
    timeout_sec: int = 30,
    allow_runtime_recovery: bool = True,
) -> list[dict[str, Any]]:
    payload = _request_json(
        'GET',
        base_url,
        '/api/event_history',
        query={
            'limit': limit,
            'category': category,
            'action': action,
            'status': status,
        },
        timeout_sec=timeout_sec,
        allow_runtime_recovery=allow_runtime_recovery,
    )
    if not isinstance(payload, dict):
        raise CliError('Unexpected event-history payload from Ollmo.')
    items = payload.get('items', [])
    if not isinstance(items, list):
        raise CliError('Unexpected event-history items payload from Ollmo.')
    return [item for item in items if isinstance(item, dict)]


def _filter_events_by_instance(items: list[dict[str, Any]], instance_id: str) -> list[dict[str, Any]]:
    token = str(instance_id or '').strip()
    if not token:
        return items
    return [item for item in items if str(item.get('instance_id') or '').strip() == token]


def _event_sort_key(item: dict[str, Any]) -> str:
    return str(item.get('timestamp') or item.get('ts') or '')


def _latest_terminal_for_wait(items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    terminal_statuses = {'ok', 'failed', 'cancelled', 'completed'}
    terminals = [item for item in items if str(item.get('status') or '').strip() in terminal_statuses]
    if not terminals:
        return None
    return max(terminals, key=_event_sort_key)


def _latest_started_for_wait(items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    started = [item for item in items if str(item.get('status') or '').strip() == 'started']
    if not started:
        return None
    return max(started, key=_event_sort_key)


def cmd_models_list(args: argparse.Namespace) -> int:
    payload = _request_json(
        'GET',
        args.base_url,
        '/api/available_models',
        timeout_sec=args.timeout,
        allow_runtime_recovery=bool(getattr(args, 'recover_control_plane', False)),
    )
    models = payload.get('models', []) if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        raise CliError('Unexpected models payload from Ollmo.')
    if args.backend:
        models = [item for item in models if str(item.get('backend') or '').strip().lower() == args.backend.lower()]
    if args.capability:
        models = [item for item in models if str(item.get('capability') or '').strip().lower() == args.capability.lower()]
    if args.runnable_only:
        models = [item for item in models if item.get('runnable', True)]

    if args.json:
        _emit_json({'models': models, 'count': len(models)})
        return 0

    for item in models:
        model = str(item.get('model') or item.get('name') or '').strip()
        backend = str(item.get('backend') or '').strip() or 'unknown'
        capability = str(item.get('capability') or '').strip() or 'unknown'
        runnable = item.get('runnable', True)
        status = 'runnable' if runnable else f"not runnable ({item.get('disabled_reason') or 'unknown'})"
        feature_summary = _summarize_feature_contract(item)
        suffix = f' {feature_summary}' if feature_summary else ''
        print(f'- {model} [{backend} | {capability}] {status}{suffix}')
    return 0


def cmd_models_pull(args: argparse.Namespace) -> int:
    payload = _request_json(
        'POST',
        args.base_url,
        '/api/pull_model',
        payload={'model': args.model, 'backend': args.backend},
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
    else:
        print(payload.get('message') or payload.get('status') or 'Pull completed.')
    return 0


def cmd_models_remove(args: argparse.Namespace) -> int:
    payload = _request_json(
        'POST',
        args.base_url,
        '/api/remove_model',
        payload={'model': args.model, 'backend': args.backend},
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
    else:
        print(payload.get('message') or payload.get('status') or 'Remove completed.')
    return 0


def cmd_instances_list(args: argparse.Namespace) -> int:
    payload = _request_json(
        'GET',
        args.base_url,
        '/api/running_instances',
        timeout_sec=args.timeout,
        allow_runtime_recovery=bool(getattr(args, 'recover_control_plane', False)),
    )
    if not isinstance(payload, list):
        raise CliError('Unexpected instances payload from Ollmo.')
    instances = payload
    if args.backend:
        instances = [item for item in instances if str(item.get('backend') or '').strip().lower() == args.backend.lower()]
    if args.capability:
        instances = [item for item in instances if str(item.get('capability') or '').strip().lower() == args.capability.lower()]
    if args.json:
        _emit_json({'instances': instances, 'count': len(instances)})
        return 0
    for item in instances:
        feature_summary = _summarize_feature_contract(item)
        suffix = f' {feature_summary}' if feature_summary else ''
        print(
            f"- {item.get('instance_id')} -> {item.get('model')} "
            f"[{item.get('backend')} | {item.get('capability')}] port={item.get('port')}{suffix}"
        )
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    payload = _request_json(
        'POST',
        args.base_url,
        '/api/start_model',
        payload={
            'model': args.model,
            'backend': args.backend,
            'capability': args.capability,
            'model_path': args.model_path,
            'preferred_port': args.preferred_port,
        },
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
    else:
        instance = payload.get('instance') if isinstance(payload, dict) else {}
        print(instance.get('instance_id') or payload.get('status') or 'started')
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    payload = _request_json(
        'POST',
        args.base_url,
        '/api/stop_model',
        payload={'instance_id': args.instance_id},
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
    else:
        print(payload.get('message') or payload.get('status') or 'stopped')
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    instance = _coerce_instance_lookup(args.base_url, args.instance_id, allow_runtime_recovery=True)
    resolved_instance_id = str(instance.get('instance_id') or args.instance_id).strip()
    file_path = _resolve_local_file(args.file)
    prompt = ' '.join(args.prompt).strip() if args.prompt else ''
    if getattr(args, 'prompt_file', None):
        prompt_file = Path(args.prompt_file)
        if not prompt_file.exists():
            raise CliError(f'Prompt file not found: {prompt_file}')
        prompt_text = prompt_file.read_text(encoding='utf-8')
        prompt = f'{prompt_text}\n{prompt}'.strip() if prompt else prompt_text.strip()
    payload: dict[str, Any] = {'instance_id': resolved_instance_id}
    if prompt:
        payload['input'] = prompt
    if args.instructions:
        payload['instructions'] = args.instructions
    if file_path:
        payload['file_path'] = file_path
    for field in ('language', 'voice', 'instruct', 'lang_code', 'response_format'):
        value = getattr(args, field, None)
        if value not in (None, ''):
            payload[field] = value
    for field in ('speed', 'pitch'):
        value = getattr(args, field, None)
        if value is not None:
            payload[field] = value
    payload['infer_timeout_sec'] = max(60, min(7200, int(args.timeout) - 15))
    response = _request_json(
        'POST',
        args.base_url,
        '/api/responses',
        payload=payload,
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )

    if getattr(args, 'truth_json', False):
        if not isinstance(response, dict):
            raise CliError('Unexpected response payload from Ollmo.')
        response_id = str(response.get('id') or response.get('response_id') or '').strip()
        if not response_id:
            raise CliError(
                'Ollmo response did not include a response id required for canonical truth lookup.'
            )
        truth_response = _request_json(
            'GET',
            args.base_url,
            f"/api/responses/{quote(response_id, safe='')}?view=truth",
            timeout_sec=args.timeout,
            allow_runtime_recovery=False,
        )
        if not isinstance(truth_response, dict):
            raise CliError('Unexpected response truth payload from Ollmo.')
        _require_response_truth_binding(
            response_id,
            truth_response,
            minimum_frame_payload=response,
        )
        _emit_json(_build_response_truth_summary(truth_response))
        return 0

    if args.json:
        _emit_json(response)
        return 0

    content = ''
    if isinstance(response, dict):
        content = str(response.get('output_text') or response.get('content') or '').strip()
    if content:
        print(content)
    for key in ('saved_text_path', 'saved_audio_path', 'saved_image_path'):
        if isinstance(response, dict) and response.get(key):
            print(f'{key}: {response.get(key)}')
    if not content and not any(isinstance(response, dict) and response.get(key) for key in ('saved_text_path', 'saved_audio_path', 'saved_image_path')):
        print('ok')
    return 0


def cmd_responses_get(args: argparse.Namespace) -> int:
    response_id = str(args.response_id or '').strip()
    if not response_id:
        raise CliError('Response id is required.')
    response = _request_json(
        'GET',
        args.base_url,
        f"/api/responses/{quote(response_id, safe='')}?view=truth",
        timeout_sec=args.timeout,
        allow_runtime_recovery=bool(getattr(args, 'recover_control_plane', False)),
    )
    if not isinstance(response, dict):
        raise CliError('Unexpected response lookup payload from Ollmo.')
    _require_response_truth_binding(response_id, response)
    if args.truth_json:
        _emit_json(_build_response_truth_summary(response))
        return 0
    if args.json:
        _emit_json(response)
        return 0
    _print_response_truth_summary(response)
    return 0


def _print_graph_rebase_readiness(payload: dict[str, Any]) -> None:
    corpus = _graph_rebase_mapping(payload.get('corpus'))
    candidates = _graph_rebase_mapping(
        _graph_rebase_mapping(payload.get('candidate_opportunities')).get(
            'settled_final'
        )
    )
    observer = _graph_rebase_mapping(payload.get('observer'))
    print(f"report_digest: {payload.get('report_digest') or 'unknown'}")
    print(f"runtime_effect: {payload.get('runtime_effect') or 'unknown'}")
    print(
        'corpus: '
        f"settled={corpus.get('settled_final_response_count', 0)} "
        f"candidate_opportunities={candidates.get('total', 0)} "
        f"not_proposed={candidates.get('not_proposed_count', 0)} "
        f"workload_families={corpus.get('unique_workload_family_count', 0)}"
    )
    print(
        'observer: '
        f"index_ok={_format_yes_no(observer.get('index_ok'))} "
        f"load_errors={observer.get('load_error_count', 0)}"
    )
    for gate_name, gate_value in _graph_rebase_mapping(payload.get('gates')).items():
        gate = _graph_rebase_mapping(gate_value)
        print(
            f"gate {gate_name}: ready={_format_yes_no(gate.get('ready'))} "
            f"decision={gate.get('decision') or 'unknown'}"
        )
        for requirement_value in gate.get('requirements') or []:
            requirement = _graph_rebase_mapping(requirement_value)
            if not requirement:
                continue
            print(
                f"  [{'x' if requirement.get('met') else ' '}] "
                f"{requirement.get('requirement') or 'unknown'} "
                f"actual={requirement.get('actual')} "
                f"threshold={requirement.get('threshold')}"
            )


def _print_graph_rebase_inspection(payload: dict[str, Any]) -> None:
    frame = _graph_rebase_mapping(payload.get('frame'))
    print(f"response_id: {payload.get('response_id') or 'unknown'}")
    print(f"lifecycle_state: {payload.get('lifecycle_state') or 'unknown'}")
    print(
        'frame: '
        f"{frame.get('frame_id') or 'unknown'} "
        f"seq={frame.get('frame_sequence') if frame.get('frame_sequence') is not None else 'unknown'} "
        f"relation={_graph_rebase_mapping(frame.get('relation')).get('kind') or 'unknown'}"
    )
    print(f"proposals: {payload.get('proposal_count', 0)}")
    for proposal_value in payload.get('proposals') or []:
        proposal = _graph_rebase_mapping(proposal_value)
        review = _graph_rebase_mapping(proposal.get('runtime_review'))
        preservation = _graph_rebase_mapping(review.get('preservation_proof'))
        execution = _graph_rebase_mapping(review.get('execution_contract_proof'))
        print(
            f"  - {proposal.get('proposal_id') or 'unknown'} "
            f"class={proposal.get('requested_rebase_class') or 'unknown'} "
            f"binding={'valid' if proposal.get('binding_valid') else 'invalid'} "
            f"review={review.get('status') or 'unknown'} "
            f"preservation={preservation.get('status') or 'unknown'} "
            f"execution_contract={execution.get('status') or 'unknown'}"
        )
        for error in proposal.get('binding_errors') or []:
            print(f'      binding_error: {error}')
    candidate = _graph_rebase_mapping(payload.get('candidate_opportunity'))
    candidate_review = _graph_rebase_mapping(candidate.get('review'))
    if candidate_review:
        print(
            'candidate_opportunity: '
            f"status={candidate_review.get('status') or 'unknown'} "
            f"reason={candidate_review.get('reason') or 'unknown'} "
            f"false_negative_eligible={_format_yes_no(candidate.get('false_negative_eligible'))}"
        )
    selected = _graph_rebase_mapping(payload.get('selected_proposal'))
    if selected:
        scope = _graph_rebase_mapping(selected.get('scope'))
        print(f"selected_proposal: {selected.get('proposal_id')}")
        print(
            'scope: '
            f"roots={','.join(scope.get('scope_root_ids') or []) or '-'} "
            f"phases={','.join(scope.get('scope_phase_ids') or []) or '-'} "
            f"branches={','.join(scope.get('scope_branch_ids') or []) or '-'} "
            f"preserve_outside={_format_yes_no(scope.get('preserve_outside_scope'))}"
        )


def _graph_rebase_action_result(
    *,
    action: str,
    cas: dict[str, Any],
    result: dict[str, Any],
    latest_inspection: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        'kind': 'ollmo.ollmoctl_graph_rebase_action_result',
        'action': action,
        'submitted_binding': cas,
        'result': result,
    }
    if latest_inspection is not None:
        payload['latest_inspection'] = latest_inspection
    return payload


def _post_graph_rebase_operator_action(
    args: argparse.Namespace,
    *,
    action: str,
    adjudication: str,
    cas: dict[str, Any],
    reason: str,
    evidence_refs: list[str],
    resolves_record_id: str = '',
) -> dict[str, Any]:
    _require_local_graph_rebase_operator_target(args.base_url)
    payload: dict[str, Any] = {
        'action': action,
        'adjudication': adjudication,
        'reason': reason,
        'evidence_refs': evidence_refs,
        **cas,
    }
    if resolves_record_id:
        payload['resolves_record_id'] = resolves_record_id
    response_id = str(cas.get('expected_response_id') or '').strip()
    result = _request_json(
        'POST',
        args.base_url,
        f"/api/responses/{quote(response_id, safe='')}/graph_rebase/operator",
        payload=payload,
        headers=_graph_rebase_operator_headers(),
        timeout_sec=args.timeout,
        allow_runtime_recovery=False,
        allow_redirects=False,
    )
    if not isinstance(result, dict):
        raise CliError('Unexpected graph-rebase operator result from Ollmo.')
    return result


def cmd_graph_rebase_readiness(args: argparse.Namespace) -> int:
    payload = _graph_rebase_readiness(args.base_url, timeout_sec=args.timeout)
    if args.json:
        _emit_json(payload)
    else:
        _print_graph_rebase_readiness(payload)
    return 0


def cmd_graph_rebase_inspect(args: argparse.Namespace) -> int:
    truth = _graph_rebase_response_truth(
        args.base_url,
        args.response_id,
        timeout_sec=args.timeout,
    )
    inspection = _build_graph_rebase_inspection(
        truth,
        proposal_id=args.proposal_id,
        expected_response_id=args.response_id,
    )
    if args.json:
        _emit_json(inspection)
    else:
        _print_graph_rebase_inspection(inspection)
    return 0


def cmd_graph_rebase_adjudicate(args: argparse.Namespace) -> int:
    _require_local_graph_rebase_operator_target(args.base_url)
    adjudication = _graph_rebase_token(args.adjudication)
    if adjudication not in GRAPH_REBASE_ADJUDICATIONS:
        raise CliError('Unsupported graph-rebase adjudication.')
    reason = _validate_graph_rebase_reason(args.reason)
    evidence_refs = _qualified_graph_rebase_evidence_refs(args.evidence_ref)
    resolves_record_id = str(args.resolves_record_id or '').strip()
    if resolves_record_id and (
        adjudication != 'useful_proposal'
        or _graph_rebase_contains_wildcard(resolves_record_id)
    ):
        raise CliError(
            '--resolves-record-id is exact and valid only for useful_proposal adjudication.'
        )

    truth = _graph_rebase_response_truth(
        args.base_url,
        args.response_id,
        timeout_sec=args.timeout,
    )
    inspection = _build_graph_rebase_inspection(
        truth,
        proposal_id=args.proposal_id,
        expected_response_id=args.response_id,
    )
    if adjudication == 'false_negative':
        cas = _false_negative_graph_rebase_cas(
            inspection,
            rebase_class=args.rebase_class,
        )
    else:
        cas = _selected_graph_rebase_cas(
            inspection,
            rebase_class=args.rebase_class,
            required_action='adjudicate',
        )
    selected = _graph_rebase_mapping(inspection.get('selected_proposal'))
    _confirm_graph_rebase_action(
        action='adjudicate',
        cas=cas,
        assume_yes=args.yes,
        scope=_graph_rebase_mapping(selected.get('scope')),
    )
    result = _post_graph_rebase_operator_action(
        args,
        action='adjudicate',
        adjudication=adjudication,
        cas=cas,
        reason=reason,
        evidence_refs=evidence_refs,
        resolves_record_id=resolves_record_id,
    )
    output = _graph_rebase_action_result(
        action='adjudicate',
        cas=cas,
        result=result,
    )
    if args.json:
        _emit_json(output)
    else:
        record = _graph_rebase_mapping(result.get('operator_record'))
        print(
            f"{result.get('status') or 'recorded'}: "
            f"{record.get('record_id') or 'operator record created'}"
        )
    return 0


def cmd_graph_rebase_stage(args: argparse.Namespace) -> int:
    _require_local_graph_rebase_operator_target(args.base_url)
    reason = _validate_graph_rebase_reason(args.reason)
    evidence_refs = _qualified_graph_rebase_evidence_refs(args.evidence_ref)
    truth = _graph_rebase_response_truth(
        args.base_url,
        args.response_id,
        timeout_sec=args.timeout,
    )
    inspection = _build_graph_rebase_inspection(
        truth,
        proposal_id=args.proposal_id,
        expected_response_id=args.response_id,
    )
    cas = _selected_graph_rebase_cas(
        inspection,
        required_action='stage',
    )
    readiness = _graph_rebase_readiness(args.base_url, timeout_sec=args.timeout)
    gate = _require_graph_rebase_gate(readiness, 'shadow_to_stage')
    selected = _graph_rebase_mapping(inspection.get('selected_proposal'))
    _confirm_graph_rebase_action(
        action='stage',
        cas=cas,
        assume_yes=args.yes,
        scope=_graph_rebase_mapping(selected.get('scope')),
        gate_name='shadow_to_stage',
        gate=gate,
        readiness_report_digest=str(readiness.get('report_digest') or ''),
    )
    result = _post_graph_rebase_operator_action(
        args,
        action='stage',
        adjudication='accepted',
        cas=cas,
        reason=reason,
        evidence_refs=evidence_refs,
    )
    latest_truth = _graph_rebase_response_truth(
        args.base_url,
        args.response_id,
        timeout_sec=args.timeout,
    )
    latest_inspection = _build_graph_rebase_inspection(
        latest_truth,
        proposal_id=str(cas.get('expected_proposal_id') or ''),
        expected_response_id=args.response_id,
    )
    output = _graph_rebase_action_result(
        action='stage',
        cas=cas,
        result=result,
        latest_inspection=latest_inspection,
    )
    if args.json:
        _emit_json(output)
    else:
        frame = _graph_rebase_mapping(latest_inspection.get('frame'))
        print(
            f"{result.get('status') or 'staged'}: staged_no_executable_mutation; "
            f"latest_frame={frame.get('frame_id')} seq={frame.get('frame_sequence')}"
        )
    return 0


def cmd_graph_rebase_authorize_partial(args: argparse.Namespace) -> int:
    if not args.execute:
        raise CliError(
            'authorize-partial immediately queues branch-local successor work; '
            'repeat with --execute to acknowledge that effect.'
        )
    _require_local_graph_rebase_operator_target(args.base_url)
    reason = _validate_graph_rebase_reason(args.reason)
    evidence_refs = _qualified_graph_rebase_evidence_refs(args.evidence_ref)
    truth = _graph_rebase_response_truth(
        args.base_url,
        args.response_id,
        timeout_sec=args.timeout,
    )
    inspection = _build_graph_rebase_inspection(
        truth,
        proposal_id=args.proposal_id,
        expected_response_id=args.response_id,
    )
    cas = _selected_graph_rebase_cas(
        inspection,
        required_action='authorize_partial',
    )
    if cas.get('expected_requested_rebase_class') != 'partial_subtree_rebase':
        raise CliError('authorize-partial cannot authorize a full successor rebase.')
    readiness = _graph_rebase_readiness(args.base_url, timeout_sec=args.timeout)
    gate = _require_graph_rebase_gate(readiness, 'partial_stage_to_apply_reviewed')
    selected = _graph_rebase_mapping(inspection.get('selected_proposal'))
    _confirm_graph_rebase_action(
        action='authorize_partial',
        cas=cas,
        assume_yes=args.yes,
        scope=_graph_rebase_mapping(selected.get('scope')),
        gate_name='partial_stage_to_apply_reviewed',
        gate=gate,
        readiness_report_digest=str(readiness.get('report_digest') or ''),
    )
    result = _post_graph_rebase_operator_action(
        args,
        action='authorize_partial',
        adjudication='accepted',
        cas=cas,
        reason=reason,
        evidence_refs=evidence_refs,
    )
    output = _graph_rebase_action_result(
        action='authorize_partial',
        cas=cas,
        result=result,
    )
    if args.json:
        _emit_json(output)
    else:
        frame = _graph_rebase_mapping(result.get('response_frame'))
        print(
            f"{result.get('status') or 'queued'}: "
            'branch_local_partial_successor_queued; '
            f"frame={frame.get('frame_id') or 'pending'}"
        )
    return 0


def cmd_history_chat(args: argparse.Namespace) -> int:
    payload = _request_json(
        'GET',
        args.base_url,
        '/api/chat_history',
        query={'instance_id': args.instance_id},
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
        return 0
    for item in payload.get('messages', []):
        print(f"[{item.get('timestamp')}] {item.get('role')}: {item.get('content')}")
    return 0


def cmd_history_infer(args: argparse.Namespace) -> int:
    payload = _request_json(
        'GET',
        args.base_url,
        '/api/infer_history',
        query={
            'limit': args.limit,
            'capability': args.capability,
            'mode': args.mode,
            'file_kind': args.file_kind,
        },
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
        return 0
    for item in payload.get('items', []):
        print(f"- {item.get('timestamp')} {item.get('mode')} {item.get('file_name') or ''}")
    return 0


def cmd_events_list(args: argparse.Namespace) -> int:
    items = _fetch_event_items(
        args.base_url,
        limit=args.limit,
        category=args.category,
        action=args.action,
        status=args.status,
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.instance_id:
        items = _filter_events_by_instance(items, args.instance_id)
    payload = {'items': items, 'count': len(items)}
    if args.json:
        _emit_json(payload)
        return 0
    for item in items:
        timestamp = item.get('timestamp') or item.get('ts') or ''
        category = item.get('category') or 'event'
        action = item.get('action') or 'action'
        status = item.get('status') or ''
        message = item.get('message') or ''
        print(f"- {timestamp} {category}/{action} {status} {message}".rstrip())
    return 0


def cmd_events_tail(args: argparse.Namespace) -> int:
    seen_ids: set[str] = set()
    emitted = 0
    iterations = 0
    while True:
        items = _fetch_event_items(
            args.base_url,
            limit=args.limit,
            category=args.category,
            action=args.action,
            status=args.status,
            timeout_sec=args.timeout,
            allow_runtime_recovery=True,
        )
        if args.instance_id:
            items = _filter_events_by_instance(items, args.instance_id)
        new_items = [item for item in reversed(items) if str(item.get('id') or '') not in seen_ids]
        for item in new_items:
            item_id = str(item.get('id') or '').strip()
            if item_id:
                seen_ids.add(item_id)
            if args.json:
                _emit_json(item)
            else:
                timestamp = item.get('timestamp') or item.get('ts') or ''
                category = item.get('category') or 'event'
                action = item.get('action') or 'action'
                status = item.get('status') or ''
                message = item.get('message') or ''
                print(f"- {timestamp} {category}/{action} {status} {message}".rstrip())
            emitted += 1
            if args.count and emitted >= args.count:
                return 0
        iterations += 1
        if args.iterations and iterations >= args.iterations:
            return 0
        time.sleep(args.interval)


def cmd_wait(args: argparse.Namespace) -> int:
    instance = _coerce_instance_lookup(args.base_url, args.instance_id, allow_runtime_recovery=True)
    resolved_instance_id = str(instance.get('instance_id') or args.instance_id).strip()
    started_at = time.time()
    while True:
        items = _fetch_event_items(
            args.base_url,
            limit=args.limit,
            category=args.category,
            action=args.action,
            timeout_sec=args.timeout,
            allow_runtime_recovery=True,
        )
        items = _filter_events_by_instance(items, resolved_instance_id)
        latest_terminal = _latest_terminal_for_wait(items)
        latest_started = _latest_started_for_wait(items)

        waiting_on_started = bool(
            latest_started and (
                latest_terminal is None or _event_sort_key(latest_started) > _event_sort_key(latest_terminal)
            )
        )
        if not waiting_on_started and latest_terminal is not None:
            payload = {'instance_id': resolved_instance_id, 'event': latest_terminal}
            if args.json:
                _emit_json(payload)
            else:
                print(f"{latest_terminal.get('status')}: {latest_terminal.get('message') or latest_terminal.get('action')}")
            return 0

        elapsed = time.time() - started_at
        if elapsed >= args.max_wait:
            raise CliError(
                f"Timed out waiting for {args.category}/{args.action} on {resolved_instance_id} after {int(args.max_wait)}s."
            )
        time.sleep(args.interval)


def cmd_artifact_open(args: argparse.Namespace) -> int:
    payload = _request_json(
        'POST',
        args.base_url,
        '/api/open_saved_artifact',
        payload={'path': args.path},
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    if args.json:
        _emit_json(payload)
    else:
        print(payload.get('status') or 'opened')
    return 0


def cmd_artifact_download(args: argparse.Namespace) -> int:
    body, headers = _request_bytes(
        'GET',
        args.base_url,
        '/api/download_saved_artifact',
        query={'path': args.path},
        timeout_sec=args.timeout,
        allow_runtime_recovery=True,
    )
    target = Path(args.output).expanduser().resolve() if args.output else Path.cwd() / Path(args.path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    if args.json:
        _emit_json({'saved_to': str(target), 'bytes': len(body), 'content_type': headers.get('Content-Type')})
    else:
        print(str(target))
    return 0


def cmd_doctor_runtime(args: argparse.Namespace) -> int:
    payload = _build_runtime_doctor_payload(args.base_url, args.timeout)
    if args.json:
        _emit_json(payload)
        return 0

    print('Ollama')
    print(f"  configured: {payload['ollama']['configured_path']['path']}")
    print(f"  resolved:   {payload['ollama']['configured_path']['resolved']}")
    print(f"  opt path:   {payload['ollama']['opt_path']['path']}")
    service_status = payload['ollama']['brew_service'].get('status') or 'unknown'
    print(f'  brew service: {service_status}')
    if payload['ollama']['which']:
        print('  which:')
        for item in payload['ollama']['which']:
            print(f'    - {item}')

    print('MLX')
    print(f"  python: {payload['mlx']['python']['path']}")
    if payload['mlx'].get('python_version'):
        print(f"  python version: {payload['mlx']['python_version']}")
    for name, version in payload['mlx']['packages'].items():
        print(f"  {name}: {version or 'not installed'}")

    print('Runtime')
    if payload['runtime']['reachable']:
        print(f"  reachable: yes ({payload['runtime']['count']} instance(s))")
        for item in payload['runtime']['instances']:
            summary = (
                f"  - {item.get('instance_id')} [{item.get('backend')} | {item.get('capability')}] "
                f"port={item.get('port')} readiness={item.get('readiness') or 'unknown'} "
                f"activity={item.get('activity') or 'unknown'}"
            )
            print(summary)
            if item.get('last_error'):
                print(f"    last_error: {item.get('last_error')}")
    else:
        print(f"  reachable: no ({payload['runtime']['error']})")

    print('Issues')
    if payload['issues']:
        for issue in payload['issues']:
            print(f'  - {issue}')
    else:
        print('  - none')
    return 0


def cmd_ghost(args: argparse.Namespace) -> int:
    if getattr(args, 'reset_learning_state', False):
        payload = reset_ghost_learning_state()
        if args.json:
            _emit_json(payload)
        else:
            print(f"Reset Ghost learning state at {payload.get('reset_at')}.")
            print(f"Archive: {payload.get('archive_dir')}")
            print(f"Preserved response frames: {((payload.get('preserved_paths') or {}).get('response_frame_ledger') or 'state/response_frames/responses.jsonl')}")
        return 0
    payload = _request_json(
        'GET',
        args.base_url,
        '/api/ghost',
        timeout_sec=args.timeout,
        allow_runtime_recovery=bool(getattr(args, 'recover_control_plane', False)),
    )
    if args.json:
        _emit_json(payload)
        return 0
    markdown = str(payload.get('markdown') or '').strip()
    if markdown:
        print(markdown)
    else:
        print(payload.get('summary') or 'No ghost summary available.')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Unified CLI adapter for the Ollmo Flask API.')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='Ollmo base URL')
    parser.add_argument('--timeout', type=int, default=None, help='Request timeout in seconds')
    parser.add_argument(
        '--recover-control-plane',
        action='store_true',
        help='Allow read commands to auto-start the local Ollmo control plane when it is unreachable.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    models = sub.add_parser('models', help='Manage available models')
    models_sub = models.add_subparsers(dest='models_command', required=True)
    models_list = models_sub.add_parser('list', help='List available models')
    models_list.add_argument('--backend')
    models_list.add_argument('--capability')
    models_list.add_argument('--runnable-only', action='store_true')
    models_list.add_argument('--json', action='store_true')
    models_list.set_defaults(func=cmd_models_list)

    models_pull = models_sub.add_parser('pull', help='Pull or cache a model')
    models_pull.add_argument('model')
    models_pull.add_argument('--backend', default='ollama')
    models_pull.add_argument('--json', action='store_true')
    models_pull.set_defaults(func=cmd_models_pull)

    models_remove = models_sub.add_parser('remove', help='Remove a model')
    models_remove.add_argument('model')
    models_remove.add_argument('--backend', default='ollama')
    models_remove.add_argument('--json', action='store_true')
    models_remove.set_defaults(func=cmd_models_remove)

    instances = sub.add_parser('instances', help='Inspect running instances')
    instances_sub = instances.add_subparsers(dest='instances_command', required=True)
    instances_list = instances_sub.add_parser('list', help='List running instances')
    instances_list.add_argument('--backend')
    instances_list.add_argument('--capability')
    instances_list.add_argument('--json', action='store_true')
    instances_list.set_defaults(func=cmd_instances_list)

    start = sub.add_parser('start', help='Start a model instance')
    start.add_argument('--model', required=True)
    start.add_argument('--backend', default='ollama')
    start.add_argument('--capability')
    start.add_argument('--model-path')
    start.add_argument('--preferred-port', type=int)
    start.add_argument('--json', action='store_true')
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser('stop', help='Stop a running instance')
    stop.add_argument('instance_id')
    stop.add_argument('--json', action='store_true')
    stop.set_defaults(func=cmd_stop)

    send = sub.add_parser(
        'send',
        help='Send a unified request to a running instance',
        description=(
            'Route a prompt or attached file through Ollmo. '
            'Requires a running instance_id; use `ollmoctl instances list --json` to inspect current IDs.'
        ),
        epilog=(
            'Examples:\n'
            '  python3 scripts/ollmoctl.py send x/flux2-klein:latest-1 "a giraffe and a T-rex taking a selfie" --json\n'
            '  python3 scripts/ollmoctl.py send mlx-community__Qwen3-TTS-12Hz-0.6B-Base-bf16-mlx-11504 "Hallo Dev" --json'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    send.add_argument('instance_id', help='Running instance id (required)')
    send.add_argument('prompt', nargs='*')
    send.add_argument('--prompt-file', help='Read prompt text from a local file')
    send.add_argument('--file', help='Local file path to attach via Ollmo file_path support')
    send.add_argument('--instructions', help='Optional system instruction for chat-capable instances')
    send.add_argument('--language')
    send.add_argument('--voice')
    send.add_argument('--instruct')
    send.add_argument('--lang-code')
    send.add_argument('--response-format')
    send.add_argument('--speed', type=float)
    send.add_argument('--pitch', type=float)
    send_output = send.add_mutually_exclusive_group()
    send_output.add_argument('--json', action='store_true')
    send_output.add_argument(
        '--truth-json',
        action='store_true',
        help=(
            'POST once, fetch the exact canonical truth view, and print its '
            'normalized lifecycle/frame summary'
        ),
    )
    send.set_defaults(func=cmd_send, timeout=3600)

    responses = sub.add_parser('responses', help='Inspect canonical response state')
    responses_sub = responses.add_subparsers(dest='responses_command', required=True)
    responses_get = responses_sub.add_parser(
        'get',
        help='Read /api/responses/<id> and show canonical lifecycle/frame truth',
    )
    responses_get.add_argument('response_id')
    responses_output = responses_get.add_mutually_exclusive_group()
    responses_output.add_argument(
        '--json',
        action='store_true',
        help='Print the raw canonical truth-view payload',
    )
    responses_output.add_argument(
        '--truth-json',
        action='store_true',
        help='Print a normalized canonical lifecycle/frame truth summary',
    )
    responses_get.add_argument(
        '--recover-control-plane',
        action='store_true',
        default=argparse.SUPPRESS,
        help='Allow this read command to auto-start the local Ollmo control plane when it is unreachable.',
    )
    responses_get.set_defaults(func=cmd_responses_get, timeout=30)

    graph_rebase = sub.add_parser(
        'graph-rebase',
        help='Inspect and operate the reviewed graph-rebase rollout',
        description=(
            'Read canonical graph-rebase evidence or perform one exact credentialed '
            'operator action. These commands never recover or start the control plane.'
        ),
    )
    graph_rebase_sub = graph_rebase.add_subparsers(
        dest='graph_rebase_command',
        required=True,
    )

    graph_rebase_readiness = graph_rebase_sub.add_parser(
        'readiness',
        help='Read the canonical passive rollout gate report',
    )
    graph_rebase_readiness.add_argument('--json', action='store_true')
    graph_rebase_readiness.set_defaults(
        func=cmd_graph_rebase_readiness,
        timeout=30,
    )

    graph_rebase_inspect = graph_rebase_sub.add_parser(
        'inspect',
        help='Inspect one response and derive exact current proposal bindings',
    )
    graph_rebase_inspect.add_argument('response_id')
    graph_rebase_inspect.add_argument('--proposal-id')
    graph_rebase_inspect.add_argument('--json', action='store_true')
    graph_rebase_inspect.set_defaults(func=cmd_graph_rebase_inspect, timeout=30)

    graph_rebase_adjudicate = graph_rebase_sub.add_parser(
        'adjudicate',
        help='Append one exact trusted operator adjudication',
    )
    graph_rebase_adjudicate.add_argument('response_id')
    graph_rebase_adjudicate.add_argument('--proposal-id')
    graph_rebase_adjudicate.add_argument(
        '--adjudication',
        required=True,
        choices=sorted(GRAPH_REBASE_ADJUDICATIONS),
    )
    graph_rebase_adjudicate.add_argument(
        '--rebase-class',
        default='',
        choices=sorted(GRAPH_REBASE_CLASSES),
        help='Required for false_negative; otherwise must match the selected proposal when supplied.',
    )
    graph_rebase_adjudicate.add_argument('--reason', required=True)
    graph_rebase_adjudicate.add_argument(
        '--evidence-ref',
        action='append',
        required=True,
        help='Qualified namespace:value evidence ref; repeat for multiple refs.',
    )
    graph_rebase_adjudicate.add_argument('--resolves-record-id', default='')
    graph_rebase_adjudicate.add_argument('--yes', action='store_true')
    graph_rebase_adjudicate.add_argument('--json', action='store_true')
    graph_rebase_adjudicate.set_defaults(
        func=cmd_graph_rebase_adjudicate,
        timeout=30,
    )

    graph_rebase_stage = graph_rebase_sub.add_parser(
        'stage',
        help='Append an exact durable non-executable stage',
    )
    graph_rebase_stage.add_argument('response_id')
    graph_rebase_stage.add_argument('--proposal-id')
    graph_rebase_stage.add_argument('--reason', required=True)
    graph_rebase_stage.add_argument(
        '--evidence-ref',
        action='append',
        required=True,
        help='Qualified namespace:value evidence ref; repeat for multiple refs.',
    )
    graph_rebase_stage.add_argument('--yes', action='store_true')
    graph_rebase_stage.add_argument('--json', action='store_true')
    graph_rebase_stage.set_defaults(func=cmd_graph_rebase_stage, timeout=30)

    graph_rebase_authorize = graph_rebase_sub.add_parser(
        'authorize-partial',
        help='Authorize and immediately queue one exact branch-local partial successor',
    )
    graph_rebase_authorize.add_argument('response_id')
    graph_rebase_authorize.add_argument('--proposal-id')
    graph_rebase_authorize.add_argument('--reason', required=True)
    graph_rebase_authorize.add_argument(
        '--evidence-ref',
        action='append',
        required=True,
        help='Qualified namespace:value evidence ref; repeat for multiple refs.',
    )
    graph_rebase_authorize.add_argument(
        '--execute',
        action='store_true',
        help='Acknowledge that authorization immediately queues branch-local successor work.',
    )
    graph_rebase_authorize.add_argument('--yes', action='store_true')
    graph_rebase_authorize.add_argument('--json', action='store_true')
    graph_rebase_authorize.set_defaults(
        func=cmd_graph_rebase_authorize_partial,
        timeout=30,
    )

    history = sub.add_parser('history', help='Inspect persisted history')
    history_sub = history.add_subparsers(dest='history_command', required=True)
    history_chat = history_sub.add_parser('chat', help='Read chat history for an instance')
    history_chat.add_argument('--instance-id', required=True)
    history_chat.add_argument('--json', action='store_true')
    history_chat.set_defaults(func=cmd_history_chat)

    history_infer = history_sub.add_parser('infer', help='Read infer history')
    history_infer.add_argument('--limit', type=int, default=50)
    history_infer.add_argument('--capability')
    history_infer.add_argument('--mode')
    history_infer.add_argument('--file-kind')
    history_infer.add_argument('--json', action='store_true')
    history_infer.set_defaults(func=cmd_history_infer)

    events = sub.add_parser('events', help='Inspect runtime/request events')
    events_sub = events.add_subparsers(dest='events_command', required=True)
    events_list = events_sub.add_parser('list', help='List event log entries')
    events_list.add_argument('--limit', type=int, default=50)
    events_list.add_argument('--category')
    events_list.add_argument('--action')
    events_list.add_argument('--status')
    events_list.add_argument('--instance-id')
    events_list.add_argument('--json', action='store_true')
    events_list.set_defaults(func=cmd_events_list)

    events_tail = events_sub.add_parser('tail', help='Poll and print new event log entries')
    events_tail.add_argument('--limit', type=int, default=50)
    events_tail.add_argument('--category')
    events_tail.add_argument('--action')
    events_tail.add_argument('--status')
    events_tail.add_argument('--instance-id')
    events_tail.add_argument('--interval', type=float, default=2.0)
    events_tail.add_argument('--count', type=int)
    events_tail.add_argument('--iterations', type=int)
    events_tail.add_argument('--json', action='store_true')
    events_tail.set_defaults(func=cmd_events_tail)

    wait = sub.add_parser('wait', help='Wait for a terminal event for an instance')
    wait.add_argument('instance_id')
    wait.add_argument('--category', default='infer')
    wait.add_argument('--action', default='request')
    wait.add_argument('--limit', type=int, default=200)
    wait.add_argument('--interval', type=float, default=3.0)
    wait.add_argument('--max-wait', type=float, default=7200.0)
    wait.add_argument('--json', action='store_true')
    wait.set_defaults(func=cmd_wait, timeout=30)

    artifact = sub.add_parser('artifact', help='Open or download saved artifacts')
    artifact_sub = artifact.add_subparsers(dest='artifact_command', required=True)
    artifact_open = artifact_sub.add_parser('open', help='Open a saved artifact in the local file manager')
    artifact_open.add_argument('path')
    artifact_open.add_argument('--json', action='store_true')
    artifact_open.set_defaults(func=cmd_artifact_open)

    artifact_download = artifact_sub.add_parser('download', help='Download a saved artifact to a local path')
    artifact_download.add_argument('path')
    artifact_download.add_argument('--output')
    artifact_download.add_argument('--json', action='store_true')
    artifact_download.set_defaults(func=cmd_artifact_download, timeout=120)

    doctor = sub.add_parser('doctor', help='Local runtime diagnostics')
    doctor_sub = doctor.add_subparsers(dest='doctor_command', required=True)
    doctor_runtime = doctor_sub.add_parser('runtime', help='Inspect Ollama ownership, MLX versions, and runtime state')
    doctor_runtime.add_argument('--json', action='store_true')
    doctor_runtime.set_defaults(func=cmd_doctor_runtime, timeout=20)

    ghost = sub.add_parser('ghost', help='Read Ollmo runtime-intelligence summary')
    ghost.add_argument('--json', action='store_true')
    ghost.add_argument(
        '--reset-learning-state',
        action='store_true',
        help='Archive Ghost event/policy/compiled-memory state and recreate a fresh baseline while preserving response frames',
    )
    ghost.set_defaults(func=cmd_ghost, timeout=20)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.base_url = _normalize_base_url(getattr(args, 'base_url', None))
    args.timeout = int(getattr(args, 'timeout', None) or 30)
    func = getattr(args, 'func', None)
    if not callable(func):
        parser.error('No command selected.')
    try:
        if (
            getattr(args, 'command', None) == 'graph-rebase'
            and bool(getattr(args, 'recover_control_plane', False))
        ):
            raise CliError(
                'Graph-rebase commands never recover or start the control plane; '
                'remove --recover-control-plane and use the already-running trusted process.'
            )
        return int(func(args) or 0)
    except CliError as exc:
        message = str(exc)
        if getattr(args, 'json', False) or getattr(args, 'truth_json', False):
            _emit_json({'error': message, 'status_code': exc.status_code, 'details': exc.payload})
        else:
            print(f'Error: {message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
