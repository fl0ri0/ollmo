# web_server.py
import copy
import io
import html as html_lib
import json
import logging
import socket
import datetime as dt
import hashlib
import hmac
import secrets
import re
from collections import Counter
from collections.abc import Mapping
import shutil
import base64
import os
import subprocess
import sys
import tempfile
import mimetypes
import time
import threading
from pathlib import Path
from typing import Any, Optional, List
from urllib.parse import quote, unquote, urlparse
import uuid

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - lightweight fallback for air-gapped envs
    import json as _json
    import socket as _socket
    from types import SimpleNamespace as _SimpleNamespace
    from urllib import error as _urlerror
    from urllib import request as _urlrequest

    class _RequestException(Exception):
        def __init__(self, message, response=None):
            super().__init__(message)
            self.response = response

    class _Timeout(_RequestException):
        """Raised when an HTTP request times out."""

    class _Response:
        def __init__(self, body: bytes, status_code: int, headers: Optional[dict] = None):
            self._body = body
            self.status_code = status_code
            self.headers = headers or {}

        @property
        def ok(self):
            return 200 <= self.status_code < 400

        def json(self):
            if not self._body:
                return None
            return _json.loads(self._body.decode('utf-8'))

        @property
        def text(self):
            return self._body.decode('utf-8', errors='replace')

        @property
        def content(self):
            return self._body

        def raise_for_status(self):
            if not self.ok:
                raise _RequestException(f"HTTP {self.status_code}", response=self)

    class _SimpleRequests:
        RequestException = _RequestException
        exceptions = _SimpleNamespace(RequestException=_RequestException, Timeout=_Timeout)

        @staticmethod
        def _request(method: str, url: str, *, json=None, timeout: Optional[float] = None):
            data = None
            headers = {}
            if json is not None:
                data = _json.dumps(json).encode('utf-8')
                headers['Content-Type'] = 'application/json'
            req = _urlrequest.Request(url, data=data, method=method.upper(), headers=headers)
            try:
                with _urlrequest.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    status = getattr(resp, 'status', resp.getcode())
                    headers_dict = dict(resp.headers.items())
                    response = _Response(body, status, headers_dict)
                    return response
            except _urlerror.HTTPError as exc:
                body = exc.read()
                response = _Response(body, exc.code, dict(exc.headers or {}))
                raise _RequestException(str(exc), response=response) from exc
            except _urlerror.URLError as exc:
                if isinstance(exc.reason, _socket.timeout):
                    raise _Timeout(f"Timeout contacting {url}") from exc
                raise _RequestException(str(exc)) from exc

        @classmethod
        def get(cls, url: str, **kwargs):
            return cls._request('GET', url, **kwargs)

        @classmethod
        def post(cls, url: str, **kwargs):
            return cls._request('POST', url, **kwargs)

    requests = _SimpleRequests()  # type: ignore[assignment]

REQUEST_TIMEOUT_ERROR = getattr(requests.exceptions, "Timeout", Exception)
REQUEST_CONNECTION_ERROR = getattr(requests.exceptions, "ConnectionError", REQUEST_TIMEOUT_ERROR)
REQUEST_EXCEPTION_ERROR = getattr(requests.exceptions, "RequestException", Exception)

from flask import (
    Flask,
    Response,
    has_app_context,
    has_request_context,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    stream_with_context,
)
from flask_cors import CORS
from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    SUPPORTED_CAPABILITIES,
    build_feature_contract,
    build_registry_metadata,
    infer_capability,
    infer_supported_capabilities,
    normalize_backend,
    normalize_capability,
    supports_capability,
)
from ollmo_integrations.codex.execution import execute_codex_request
from ollmo_integrations.codex.runtime_target import (
    CODEX_TARGET_BACKEND,
    CODEX_TARGET_ID,
    CODEX_TARGET_MODEL,
    build_codex_execution_inputs,
    build_codex_external_target,
    codex_execution_failure,
    validate_codex_text_request,
)
from ollmo_services.file_inputs import (
    expand_local_paths,
    file_kind_from_name,
    hash_file_sha256,
    normalize_local_path_input,
    parse_bool,
    parse_int_with_bounds,
    read_text_file,
    read_text_file_with_metadata,
    resolve_existing_local_path,
    save_local_path_to_temp,
    save_upload_to_temp,
    to_base64,
)
from ollmo_services.events import log_event, read_events
from ollmo_g import build_ghost_payload
from ollmo_g.control_hints import (
    IMAGE_ASPECT_PRESET_DIMENSIONS,
    apply_prompt_control_hints,
    infer_tts_speaker_from_prompt,
)
from ollmo_g.semantic_role_profile import build_semantic_role_profile
from ollmo_g.execution_planner import (
    plan_compound_execution,
    split_visible_image_payload,
    split_visible_tts_payload,
)
from ollmo_g.image_state import parse_image_state_response
from ollmo_g.intent import analyze_prompt_intent
from ollmo_g.request_phase_graph import (
    build_request_phase_graph,
    current_phase_capability as _current_request_phase_capability,
    current_phase_is_graph_resolved,
    current_phase_reason as _current_request_phase_reason,
    downstream_phase_branch_batches as _downstream_request_phase_branch_batches,
    downstream_phase_batches as _downstream_request_phase_batches,
    downstream_phase_capabilities as _downstream_request_phase_capabilities,
)
from ollmo_g.request_meta import (
    apply_request_meta_to_route_context,
    attach_request_meta,
    compact_request_meta,
    effective_developer_flags,
    extract_request_meta,
)
from ollmo_g.router import (
    MAX_RECENT_MESSAGES,
    build_failure_recovery_route,
    build_embedding_route_audit,
    build_embedding_hints_from_vectors,
    build_embedding_route_candidates,
    build_route_memory_scope,
    build_route_context,
    collect_routing_preferences,
    maybe_apply_embedding_route_bias,
    select_embedding_instance,
    sanitize_ghost_messages,
    validate_route_decision,
)
from ollmo_server.ghost_route_runtime import GhostRouteRuntimeOwner
from ollmo_server.backend_transport_runtime import BackendTransportRuntimeOwner
from ollmo_server.chat_runtime import ChatRuntimeOwner
from ollmo_server.infer_support_runtime import InferSupportRuntimeOwner
from ollmo_server.model_control_runtime import ModelControlRuntimeOwner
from ollmo_server.ocr_pdf_runtime import OcrPdfRuntimeOwner
from ollmo_server.request_intake_runtime import RequestIntakeRuntimeOwner
from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner
from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_server.infer_postprocess import GeneratedImagePostprocessOwner
from ollmo_server.infer_runtime import InferRuntimeOwner
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_server.multi_materialization_runtime import (
    MultiMaterializationRuntimeOwner,
    normalize_max_parallel_workers,
)
from ollmo_server.recovery_contract import (
    RECOVERY_ACTION_MANUAL_REVIEW,
    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
    RECOVERY_ACTION_RETRY_SAME_BRANCH,
    normalize_recovery_suggested_action,
)
from ollmo_server.responses_runtime import (
    ResponsesRuntimeOwner,
    derive_response_lifecycle_state,
    late_fill_has_actionable_repair_work,
)
from ollmo_server.response_lookup_runtime import (
    ResponseLookupRuntimeOwner,
    response_lookup_frame_id_value as _runtime_response_lookup_frame_id_value,
    response_lookup_frame_sequence_value as _runtime_response_lookup_frame_sequence_value,
    response_wire_frame_identity as _runtime_response_wire_frame_identity,
)
from ollmo_server.substrate_hygiene_runtime import (
    PostResponseSubstrateHygieneRuntimeOwner,
    normalize_post_response_substrate_unload_policy,
)
from ollmo_services.chat_history import (
    delete_chat_history,
    list_chat_history_index,
    read_chat_history,
    resolve_chat_history_slot,
    rotate_chat_history,
    write_chat_history,
)
from ollmo_services.history import (
    acquire_infer_slot,
    append_infer_history,
    build_infer_dedupe_key,
    read_infer_history,
    release_infer_slot,
    truncate_for_history,
)
from ollmo_services.inference import (
    InferArtifacts,
    InferContext,
    detect_text_artifact_requests,
    dispatch_infer_request,
    extract_text_artifact_payloads,
    normalize_text_artifact_extension,
)
from ollmo_services.ocr_pdf import (
    clean_ocr_output_text,
    collapse_repeated_ocr_lines,
    detect_low_quality_ocr_reason,
    extract_pdf_text_content,
    is_generic_ocr_instruction_prompt,
    line_has_ocr_garbage_pattern,
    looks_like_ocr_prompt_echo,
    normalize_ocr_line,
    ocr_image_with_deepseek,
    ocr_pdf_page_with_ollama,
    render_pdf_pages_to_base64,
    render_single_pdf_page_to_base64,
    sanitize_ocr_noise_lines,
    strip_ocr_structural_lines,
)
from ollmo_services.responses import (
    build_canonical_batch_response_payload as _build_canonical_batch_response_payload,
    build_canonical_error_response_payload as _build_canonical_error_response_payload,
    build_canonical_response_artifacts as _build_canonical_response_artifacts,
    build_canonical_response_payload as _build_canonical_response_payload,
    build_canonical_response_stream_events as _build_canonical_response_stream_events,
    build_public_output_branches_from_slots as _build_public_output_branches_from_slots,
    build_responses_output as _build_responses_output,
    build_responses_stream_events as _build_responses_stream_events,
    extract_responses_batch_items as _extract_responses_batch_items,
    extract_responses_batch_prompts as _extract_responses_batch_prompts,
    extract_responses_content_text as _extract_responses_content_text,
    extract_responses_current_turn_prompt as _extract_responses_current_turn_prompt,
    extract_responses_messages as _extract_responses_messages,
    extract_responses_prompt as _extract_responses_prompt,
    flatten_canonical_batch_artifacts as _flatten_canonical_batch_artifacts,
    hoist_response_output_surfaces as _hoist_response_output_surfaces,
    translate_responses_payload_to_infer_payload as _translate_responses_payload_to_infer_payload,
)
from ollmo_services import response_wire as _response_wire_policy
from ollmo_services.response_frames import (
    RESPONSE_FRAME_STALE_PARENT_REASON,
    ResponseFrameParentCASMismatch,
    append_response_frame_with_parent_cas as _append_response_frame_with_parent_cas,
    attach_response_frame as _attach_response_frame,
    enrich_response_frame_for_ledger_append as _enrich_response_frame_for_ledger_append,
    enrich_response_frame_metadata as _enrich_response_frame_metadata,
    inspect_response_frame_recovery_cache as _inspect_response_frame_recovery_cache,
    load_latest_response_observation_state as _load_latest_response_observation_state,
    load_latest_response_state as _load_latest_response_state,
    load_latest_response_wire_state as _load_latest_response_wire_state,
    load_response_frame_index as _load_response_frame_index,
    persist_response_frame as _persist_response_frame,
    response_frame_ledger_record_response_id as _response_frame_ledger_record_response_id,
    select_graph_rebase_observation_response_ids as _select_graph_rebase_observation_response_ids,
    verify_response_frame_epoch as _verify_response_frame_epoch,
)
from ollmo_services.graph_rebase import (
    apply_validated_graph_rebase as _apply_validated_graph_rebase,
    build_graph_rebase_lifecycle as _build_graph_rebase_lifecycle,
    describe_graph_rebase_autonomy_from_env as _describe_graph_rebase_autonomy_from_env,
    parse_graph_rebase_frame_sequence as _parse_graph_rebase_frame_sequence,
    stable_graph_digest as _stable_graph_digest,
    validate_graph_rebase_proposal as _validate_graph_rebase_proposal,
)
from ollmo_services.graph_rebase_operator import (
    DEFAULT_GRAPH_REBASE_OPERATOR_REGISTRY_PATH,
    GraphRebaseOperatorRegistryError,
    find_trusted_graph_rebase_authorization as _find_trusted_graph_rebase_authorization,
    load_graph_rebase_operator_records as _load_graph_rebase_operator_records,
    record_graph_rebase_operator_action as _record_graph_rebase_operator_action,
)
from ollmo_services.graph_rebase_readiness_registry import (
    DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH,
    GraphRebaseReadinessRegistryError,
    append_graph_rebase_readiness_observation as _append_graph_rebase_readiness_observation,
    build_graph_rebase_source_epoch_identity as _build_graph_rebase_source_epoch_identity,
    load_graph_rebase_readiness_registry as _load_graph_rebase_readiness_registry,
)
from ollmo_services.graph_rebase_rollout import (
    build_graph_rebase_readiness_report as _build_graph_rebase_readiness_report,
    build_partial_graph_rebase_promotion_gate as _build_partial_graph_rebase_promotion_gate,
    project_graph_rebase_readiness_observation as _project_graph_rebase_readiness_observation,
)
from ollmo_services.artifact_registry import (
    DEFAULT_ARTIFACT_REGISTRY_LEDGER,
    build_generated_image_provenance as _build_generated_image_provenance,
    find_artifact_registry_record as _find_artifact_registry_record,
    find_artifact_registry_record_by_artifact_ref as _find_artifact_registry_record_by_artifact_ref,
    persist_artifact_registry_enrichment as _persist_artifact_registry_enrichment,
    persist_artifact_registry_record as _persist_artifact_registry_record,
    persist_generated_image_provenance as _persist_generated_image_provenance,
    persist_input_artifact_registry_records as _persist_input_artifact_registry_records,
    persist_output_artifact_registry_records as _persist_output_artifact_registry_records,
)
from ollmo_services.artifact_contracts import (
    extract_artifact_ref as _extract_artifact_ref,
    sanitize_artifact_record as _sanitize_artifact_record,
    sanitize_artifact_records as _sanitize_artifact_records,
)
from ollmo_orchestration.working_frame import build_working_frame as _build_working_frame
from ollmo_services.settings_artifacts import (
    DEFAULT_SETTINGS_ARTIFACTS_DIR,
    list_settings_artifacts as _list_settings_artifacts,
    load_settings_artifact as _load_settings_artifact,
    persist_settings_artifact as _persist_settings_artifact,
)
from ollmo_services.response_artifact_bundles import (
    build_response_artifact_bundle_registry_record as _build_response_artifact_bundle_registry_record,
    bundle_response_artifacts as _bundle_response_artifacts,
)
from ollmo_services.transports import (
    ARTIFACT_INPUTS_ROOT,
    ARTIFACT_OUTPUTS_AUDIO_DIR,
    ARTIFACT_OUTPUTS_BUNDLES_DIR,
    ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
    ARTIFACT_OUTPUTS_IMAGES_DIR,
    ARTIFACT_OUTPUTS_OCR_DIR,
    ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
    expand_repo_relative_roots,
    is_path_within,
    open_path_in_file_manager,
    persist_input_file_locally,
    persist_text_artifact_locally,
    persist_text_markdown_locally,
    resolve_saved_artifact_path,
)
from ollmo_core.backend_fabric import build_backend_fabric_snapshot
from ollmo_core.runtime_liveness import (
    runtime_instance_is_selectable,
    runtime_instance_score,
)
from ollmo_runtime.lifecycle import (
    StartModelRequestError,
    StopResult,
    canonical_model_name,
    list_available_models,
    list_running_instances as load_running_instances,
    lookup_instance,
    pull_model,
    remove_model,
    start_instance,
    stop_instance,
)
from ollmo_runtime.runtime_hygiene import cleanup_runtime_hygiene
from ollmo_runtime.status import (
    DEFAULT_RUNTIME_STATUS_PATH,
    merge_instances_with_runtime_status,
    read_runtime_status,
    record_instance_activity,
    record_instance_failure,
    record_instance_started,
    record_instance_success,
    refresh_runtime_status_entries,
    remove_instance_status,
)

# --- Configuration ---
APP_PORT = 5001
DASHBOARD_HTML_FILE = "ollmo_webUI.html"
LANDING_SITE_DIRECTORY = Path(__file__).resolve().parent / "site"
LANDING_HTML_FILE = "index.html"
TEMPLATE_FOLDER = "."
CONFIG_FILE_NAME = "model_ports.json"
INFER_HISTORY_PATH = Path("state/infer_history.jsonl")
EVENT_LOG_PATH = Path("state/events.jsonl")
RESPONSE_FRAMES_DIR = Path("state/response_frames")
GRAPH_REBASE_OPERATOR_REGISTRY_PATH = DEFAULT_GRAPH_REBASE_OPERATOR_REGISTRY_PATH
GRAPH_REBASE_READINESS_REGISTRY_PATH = DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH
GRAPH_REBASE_OPERATOR_TOKEN_ENV = 'OLLMO_GRAPH_REBASE_OPERATOR_TOKEN'
GRAPH_REBASE_OPERATOR_IDENTITY_ENV = 'OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY'
_GRAPH_REBASE_OPERATOR_TOKEN = os.environ.pop(
    GRAPH_REBASE_OPERATOR_TOKEN_ENV,
    '',
)
_GRAPH_REBASE_OPERATOR_IDENTITY = os.environ.pop(
    GRAPH_REBASE_OPERATOR_IDENTITY_ENV,
    '',
)
# Internal startup aliases are never authentication inputs.  Remove any
# pre-exported copies at control-plane ingress as defense in depth.
os.environ.pop('GRAPH_REBASE_OPERATOR_TOKEN', None)
os.environ.pop('GRAPH_REBASE_OPERATOR_IDENTITY', None)
GHOST_PREFERENCES_PATH = Path("state/ghost_preferences.json")
CODEX_TIMEOUT_ENV = 'OLLMO_CODEX_TIMEOUT_SEC'
DEFAULT_CODEX_TIMEOUT_SEC = 600
FLASK_LOG_PATH = Path("logs/flask_webserver.log")
OLLAMA_DEFAULT_LOG_PATH = Path("logs/ollama_default_server_11434.log")
RUNTIME_STATUS_PATH = DEFAULT_RUNTIME_STATUS_PATH
CHAT_HISTORY_DIR = Path("state/chat_history")
GHOST_GUIDE_PATH = Path("GHOST.md")
OCR_EXPORT_DIR = ARTIFACT_OUTPUTS_OCR_DIR
TRANSCRIPT_EXPORT_DIR = ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR
GENERATED_AUDIO_DIR = ARTIFACT_OUTPUTS_AUDIO_DIR
GENERATED_IMAGES_DIR = ARTIFACT_OUTPUTS_IMAGES_DIR
SAVED_INPUTS_DIR = ARTIFACT_INPUTS_ROOT
SETTINGS_ARTIFACTS_DIR = DEFAULT_SETTINGS_ARTIFACTS_DIR
ARTIFACT_BUNDLES_DIR = ARTIFACT_OUTPUTS_BUNDLES_DIR
ARTIFACT_REGISTRY_LEDGER = DEFAULT_ARTIFACT_REGISTRY_LEDGER
INFER_SLOT_TTL_SEC = 1800
RESPONSE_LOOKUP_TTL_SEC = 1800
MAX_PDF_INLINE_RESPONSE_CHARS = 400_000


def _bounded_environment_integer(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


SAVED_HTML_PREVIEW_TTL_SECONDS = _bounded_environment_integer(
    'OLLMO_HTML_PREVIEW_TTL_SECONDS',
    default=1800,
    minimum=60,
    maximum=86400,
)
SAVED_HTML_PREVIEW_MAX_PACKAGES = _bounded_environment_integer(
    'OLLMO_HTML_PREVIEW_MAX_PACKAGES',
    default=24,
    minimum=1,
    maximum=128,
)
SAVED_HTML_PREVIEW_MAX_FILES = _bounded_environment_integer(
    'OLLMO_HTML_PREVIEW_MAX_FILES',
    default=64,
    minimum=1,
    maximum=512,
)
SAVED_HTML_PREVIEW_MAX_PACKAGE_BYTES = _bounded_environment_integer(
    'OLLMO_HTML_PREVIEW_MAX_PACKAGE_BYTES',
    default=256 * 1024 * 1024,
    minimum=1024 * 1024,
    maximum=2 * 1024 * 1024 * 1024,
)
SAVED_HTML_PREVIEW_MAX_TOTAL_BYTES = _bounded_environment_integer(
    'OLLMO_HTML_PREVIEW_MAX_TOTAL_BYTES',
    default=512 * 1024 * 1024,
    minimum=1024 * 1024,
    maximum=4 * 1024 * 1024 * 1024,
)
_SAVED_HTML_PREVIEW_TEMP_DIR = tempfile.TemporaryDirectory(prefix='ollmo-html-preview-')
_SAVED_HTML_PREVIEW_TEMP_ROOT = Path(_SAVED_HTML_PREVIEW_TEMP_DIR.name).resolve()
_SAVED_HTML_PREVIEW_PACKAGES: dict[str, dict[str, Any]] = {}
_SAVED_HTML_PREVIEW_PACKAGES_LOCK = threading.RLock()
_INFER_INFLIGHT: dict[str, float] = {}
_INFER_INFLIGHT_LOCK = threading.Lock()
_RESPONSE_LOOKUP: dict[str, dict[str, Any]] = {}
_RESPONSE_LOOKUP_LOCK = threading.Lock()
_RESPONSE_STREAMS: dict[str, dict[str, Any]] = {}
_RESPONSE_STREAMS_LOCK = threading.Lock()
_RESPONSE_LATE_FILL_IN_FLIGHT: set[str] = set()
_RESPONSE_LATE_FILL_LOCK = threading.Lock()
_GENERATED_IMAGE_STATE_CACHE: dict[str, dict[str, Any]] = {}
_GENERATED_IMAGE_STATE_CACHE_LOCK = threading.Lock()
_GENERATED_IMAGE_STATE_ENRICHMENT_IN_FLIGHT: set[str] = set()
_GENERATED_IMAGE_STATE_ENRICHMENT_LOCK = threading.Lock()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Flask App ---
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
app.config['GRAPH_REBASE_OPERATOR_TOKEN'] = _GRAPH_REBASE_OPERATOR_TOKEN
app.config['GRAPH_REBASE_OPERATOR_IDENTITY'] = _GRAPH_REBASE_OPERATOR_IDENTITY
# CORS spezifischer auf neuen Port
CORS(app, resources={r"/api/*": {"origins": f"http://127.0.0.1:{APP_PORT}"}})


def _effective_graph_rebase_readiness_registry_path() -> Path:
    """Keep test/alternate response-frame epochs from writing active state."""

    configured = Path(GRAPH_REBASE_READINESS_REGISTRY_PATH)
    default_frames = Path('state/response_frames')
    if (
        Path(RESPONSE_FRAMES_DIR) != default_frames
        and configured == Path(DEFAULT_GRAPH_REBASE_READINESS_REGISTRY_PATH)
    ):
        return (
            Path(RESPONSE_FRAMES_DIR).parent
            / 'graph_rebase'
            / 'readiness_observations.jsonl'
        )
    return configured


# --- Hilfsfunktionen ---
def is_port_listening(port, host="localhost"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _active_global_log_paths(*, include_webserver: bool) -> list[Path]:
    active_logs: list[Path] = []
    if include_webserver:
        active_logs.append(FLASK_LOG_PATH)
    if is_port_listening(11434):
        active_logs.append(OLLAMA_DEFAULT_LOG_PATH)
    return active_logs


def build_stop_payload(result: StopResult, instance: Optional[dict]):
    payload = {
        "status": result.state,
        "message": result.message,
        "details": result.details,
        "instance": instance,
    }
    if result.state == "stopped":
        status_code = 200
    elif result.state == "stopping":
        status_code = 202
    else:
        status_code = 500
        payload["error"] = result.message
    return payload, status_code


def _lookup_instance(instance_id: str) -> Optional[dict]:
    return lookup_instance(instance_id)


_IDENTIFIER_CONTROL_CHARS_RE = re.compile(r'[\x00-\x1f\x7f]')
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r'^[A-Za-z]:[\\/]')


def _normalize_external_identifier(
    raw_value: Any,
    *,
    field_name: str = 'instance_id',
    allow_slashes: bool = True,
) -> str:
    token = str(raw_value or '').strip()
    if not token:
        raise ValueError(f"Parameter '{field_name}' is required.")
    if len(token) > 255:
        raise ValueError(f"Parameter '{field_name}' is too long.")
    if _IDENTIFIER_CONTROL_CHARS_RE.search(token):
        raise ValueError(f"Parameter '{field_name}' contains invalid control characters.")
    if token.startswith(('~', '/', '\\')) or _WINDOWS_ABSOLUTE_PATH_RE.match(token):
        raise ValueError(f"Parameter '{field_name}' must not be a path.")
    normalized = token.replace('\\', '/')
    if not allow_slashes and '/' in normalized:
        raise ValueError(f"Parameter '{field_name}' contains invalid separators.")
    parts = normalized.split('/')
    if any(part in {'', '.', '..'} for part in parts):
        raise ValueError(f"Parameter '{field_name}' contains invalid path segments.")
    return token


def _log_unified_event(
    *,
    category: str,
    action: str,
    status: str,
    message: Optional[str] = None,
    **fields,
):
    if app.config.get("TESTING"):
        return None
    try:
        entry = log_event(
            category=category,
            action=action,
            status=status,
            path=EVENT_LOG_PATH,
            message=message,
            **fields,
        )
        return entry
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not append unified event log entry: %s", exc)


_OPEN_RESPONSE_LIFECYCLE_STATES = {
    'accepted',
    'active',
    'in_progress',
    'late_fill_pending',
    'late_fill_running',
    'pending',
    'queued',
    'running',
    'streaming',
}

_ACTIVE_LATE_FILL_BRANCH_STATUSES = {
    'accepted',
    'active',
    'attempting',
    'in_progress',
    'pending',
    'queued',
    'running',
    'scheduled',
}

_SAFE_LATE_FILL_RETRY_WAVE_ACTIONS = {
    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
    RECOVERY_ACTION_RETRY_SAME_BRANCH,
}

_ACTIONABLE_RESPONSE_LIFECYCLE_STATES = {
    'blocked',
    'late_fill_repair_needed',
    'rebuild_from_promoted_obligations',
    'repair_branch_contract',
    'repair_dependency_chain',
    'repair_needed',
}

_TERMINAL_RESPONSE_LIFECYCLE_STATES = {
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
}


def _response_lifecycle_has_open_continuation(lifecycle_state: Any) -> bool:
    normalized = str(lifecycle_state or '').strip().lower()
    return normalized in _OPEN_RESPONSE_LIFECYCLE_STATES


def _response_lifecycle_has_actionable_repair(
    lifecycle_state: Any,
    response_payload: Mapping[str, Any],
) -> bool:
    normalized = str(lifecycle_state or '').strip().lower()
    if normalized in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES:
        return True
    payload = response_payload if isinstance(response_payload, Mapping) else {}
    late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
    if late_fill:
        late_fill_status = str(late_fill.get('status') or '').strip().lower()
        pending_branches = late_fill.get('pending_branches') if isinstance(late_fill.get('pending_branches'), list) else []
        active_branches = late_fill.get('active_branches') if isinstance(late_fill.get('active_branches'), list) else []
        failed_branches = late_fill.get('failed_branches') if isinstance(late_fill.get('failed_branches'), list) else []
        recovery_candidates = (
            late_fill.get('recovery_candidates')
            if isinstance(late_fill.get('recovery_candidates'), list)
            else []
        )
        actionable_repair_loop = late_fill_has_actionable_repair_work(late_fill)
        if (
            late_fill_status in {'completed', 'skipped', 'cancelled', 'waived', 'superseded'}
            and not pending_branches
            and not active_branches
            and not failed_branches
            and not recovery_candidates
            and not actionable_repair_loop
        ):
            return False
    if not late_fill:
        return False
    if late_fill_has_actionable_repair_work(late_fill):
        return True
    if isinstance(late_fill.get('recovery_candidates'), list) and late_fill.get('recovery_candidates'):
        return True
    if isinstance(late_fill.get('repair_actions'), list) and late_fill.get('repair_actions'):
        return True
    if late_fill.get('repair_action') not in (None, '', [], {}):
        return True
    recovery_state = late_fill.get('recovery_state') if isinstance(late_fill.get('recovery_state'), Mapping) else {}
    if recovery_state and str(recovery_state.get('status') or '').strip().lower() in {'candidate', 'ready', 'blocked'}:
        return True
    failed_branches = late_fill.get('failed_branches') if isinstance(late_fill.get('failed_branches'), list) else []
    for branch in failed_branches:
        if not isinstance(branch, Mapping):
            continue
        recovery_context = (
            branch.get('recovery_context')
            if isinstance(branch.get('recovery_context'), Mapping)
            else {}
        )
        if recovery_context.get('suggested_action') not in (None, '', [], {}):
            return True
    for branch in list(pending_branches) + list(active_branches):
        if not isinstance(branch, Mapping):
            continue
        branch_status = str(branch.get('status') or '').strip().lower()
        if branch_status in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES or branch_status in {
            'blocked',
            'failed',
            'partial_failed',
            'repair_needed',
        }:
            return True
        recovery_context = (
            branch.get('recovery_context')
            if isinstance(branch.get('recovery_context'), Mapping)
            else {}
        )
        if recovery_context.get('suggested_action') not in (None, '', [], {}):
            return True
    return False


def _response_lifecycle_is_terminal(lifecycle_state: Any) -> bool:
    normalized = str(lifecycle_state or '').strip().lower()
    return normalized in _TERMINAL_RESPONSE_LIFECYCLE_STATES


def _select_canonical_response_lifecycle_state(
    response_payload: Mapping[str, Any],
    *,
    compatibility_status: Optional[str] = None,
) -> str:
    """Choose lifecycle truth without preserving stale active continuation state."""

    payload = response_payload if isinstance(response_payload, Mapping) else {}
    return str(
        derive_response_lifecycle_state(payload, requested_status=compatibility_status)
        or compatibility_status
        or ''
    ).strip() or 'completed'


def _attach_response_status_semantics(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Make lifecycle_state the machine-readable continuation authority."""

    payload = dict(response_payload)
    compatibility_status = str(payload.get('status') or '').strip() or None
    lifecycle_state = _select_canonical_response_lifecycle_state(
        payload,
        compatibility_status=compatibility_status,
    )
    payload['lifecycle_state'] = lifecycle_state
    has_open_continuation = _response_lifecycle_has_open_continuation(lifecycle_state)
    has_actionable_repair = _response_lifecycle_has_actionable_repair(lifecycle_state, payload)
    is_terminal = _response_lifecycle_is_terminal(lifecycle_state)
    status_compatibility = bool(
        compatibility_status
        and compatibility_status != lifecycle_state
        and (
            compatibility_status == 'completed'
            or has_open_continuation
            or has_actionable_repair
            or is_terminal
        )
    )
    payload['canonical_status_field'] = 'lifecycle_state'
    payload['status_compatibility'] = status_compatibility
    payload['status_semantics'] = {
        'compatibility_status': compatibility_status,
        'canonical_lifecycle_state': lifecycle_state,
        'canonical_status_field': 'lifecycle_state',
        'status_compatibility': status_compatibility,
        'has_open_continuation': has_open_continuation,
        'has_actionable_repair': has_actionable_repair,
        'is_terminal': is_terminal,
        'terminal': is_terminal,
    }
    return payload


def _preserve_frozen_response_frame_identity(
    projected_frame: Mapping[str, Any],
    frozen_frame: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep a read projection attached to the exact frozen frame it represents."""

    preserved = dict(projected_frame)
    for key in ('frame_id', 'frame_sequence', 'frame_relation'):
        if key in frozen_frame:
            preserved[key] = copy.deepcopy(frozen_frame.get(key))
        else:
            preserved.pop(key, None)
    return preserved


def _register_durable_graph_rebase_readiness_observation(
    framed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain one relevant settled frame without changing primary frame truth.

    The response-frame append has already committed when this helper runs.  A
    registry failure is therefore evidence about the secondary durability
    surface; it must never roll the primary append back.
    """

    projection = _project_graph_rebase_readiness_observation(framed_payload)
    readiness_state = (
        projection.get('readiness_state')
        if isinstance(projection.get('readiness_state'), Mapping)
        else {}
    )
    if (
        readiness_state.get('settled_final') is not True
        or readiness_state.get('active_late_fill') is True
    ):
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'not_settled',
            'runtime_effect': 'none',
        }

    observation_keys = {
        'applied_graph_rebases',
        'graph_rebase_lifecycle',
        'graph_rebase_outcomes',
        'graph_rebase_proposals',
        'graph_rebase_reviews',
        'partial_rebase_outcomes',
        'response_time_graph_rebase_candidate',
        'runtime_graph_rebase_candidate_review',
        'runtime_graph_rebase_proposals',
        'runtime_graph_rebase_reviews',
        'staged_graph_rebases',
        'successor_rebase_executions',
        'successor_rebase_requests',
    }
    relation_kinds = {
        'graph_rebase_partial_successor',
        'graph_rebase_stage_successor',
    }
    stack: list[Any] = [
        projection.get('runtime'),
        projection.get('frame_relation'),
    ]
    locally_relevant = False
    while stack and not locally_relevant:
        value = stack.pop()
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key or '').strip()
                if key in observation_keys or (
                    key in {'kind', 'relation_kind'}
                    and str(child or '').strip() in relation_kinds
                ):
                    locally_relevant = True
                    break
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    if not locally_relevant:
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'not_relevant',
            'runtime_effect': 'none',
            'response_id': str(projection.get('response_id') or '').strip()
            or None,
        }

    verified_epoch = _verify_response_frame_epoch(
        frames_dir=RESPONSE_FRAMES_DIR,
        allow_relocated=False,
    )
    if verified_epoch.get('ok') is not True:
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'verification_failed',
            'runtime_effect': 'none',
            'error': dict(verified_epoch.get('error') or {}),
        }
    index_state = (
        verified_epoch.get('index_state')
        if isinstance(verified_epoch.get('index_state'), Mapping)
        else {}
    )
    selection = _select_graph_rebase_observation_response_ids(
        frames_dir=RESPONSE_FRAMES_DIR,
        index_state=index_state,
    )
    if int(selection.get('scan_error_count') or 0) > 0:
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'selection_failed',
            'runtime_effect': 'none',
            'error_count': int(selection.get('scan_error_count') or 0),
            'errors': list(selection.get('scan_errors') or [])[:20],
        }

    response_id = str(projection.get('response_id') or '').strip()
    if response_id not in set(selection.get('selected_response_ids') or []):
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'not_relevant',
            'runtime_effect': 'none',
            'response_id': response_id or None,
        }

    response_index = (
        index_state.get('responses')
        if isinstance(index_state.get('responses'), Mapping)
        else {}
    )
    current_entry = (
        response_index.get(response_id)
        if isinstance(response_index.get(response_id), Mapping)
        else {}
    )
    projected_frame_id = str(projection.get('frame_id') or '').strip()
    projected_sequence = projection.get('ledger_sequence')
    if (
        str(current_entry.get('latest_frame_id') or '').strip() != projected_frame_id
        or current_entry.get('latest_frame_sequence') != projected_sequence
    ):
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'superseded_before_registration',
            'runtime_effect': 'none',
            'response_id': response_id,
            'frame_id': projected_frame_id or None,
        }


    observed_state = _load_latest_response_observation_state(
        response_id,
        frames_dir=RESPONSE_FRAMES_DIR,
        index_state=index_state,
    )
    observed_payload = (
        observed_state.get('response_payload')
        if isinstance(observed_state.get('response_payload'), Mapping)
        else {}
    )
    if observed_state.get('ok') is not True or not observed_payload:
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'hydration_failed',
            'runtime_effect': 'none',
            'response_id': response_id,
            'frame_id': projected_frame_id or None,
            'error': dict(observed_state.get('error') or {}),
        }
    durable_projection = _project_graph_rebase_readiness_observation(
        observed_payload
    )
    durable_readiness_state = (
        durable_projection.get('readiness_state')
        if isinstance(durable_projection.get('readiness_state'), Mapping)
        else {}
    )
    if (
        str(durable_projection.get('response_id') or '').strip() != response_id
        or str(durable_projection.get('frame_id') or '').strip()
        != projected_frame_id
        or durable_projection.get('ledger_sequence') != projected_sequence
        or durable_readiness_state.get('settled_final') is not True
        or durable_readiness_state.get('active_late_fill') is True
    ):
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'hydration_binding_mismatch',
            'runtime_effect': 'none',
            'response_id': response_id,
            'frame_id': projected_frame_id or None,
        }

    source_frame_digests = (
        verified_epoch.get('source_frame_sha256_by_response')
        if isinstance(
            verified_epoch.get('source_frame_sha256_by_response'),
            Mapping,
        )
        else {}
    )
    append_result = _append_graph_rebase_readiness_observation(
        durable_projection,
        source_frame=str(source_frame_digests.get(response_id) or '').strip(),
        source_epoch=_build_graph_rebase_source_epoch_identity(verified_epoch),
        verified_epoch=verified_epoch,
        frames_dir=RESPONSE_FRAMES_DIR,
        registry_path=_effective_graph_rebase_readiness_registry_path(),
    )
    if append_result.get('ok') is not True:
        return {
            'kind': 'ollmo.graph_rebase_readiness_registry_append',
            'status': 'append_failed',
            'runtime_effect': 'none',
            'response_id': response_id,
            'frame_id': projected_frame_id or None,
            'error': dict(append_result.get('error') or {}),
        }
    return {
        'kind': 'ollmo.graph_rebase_readiness_registry_append',
        'status': str(append_result.get('status') or 'appended'),
        'runtime_effect': 'none',
        'response_id': response_id,
        'frame_id': projected_frame_id or None,
        'appended_record_count': int(
            append_result.get('appended_record_count') or 0
        ),
        'already_present_count': int(
            append_result.get('already_present_count') or 0
        ),
        'registry_record_count': int(append_result.get('record_count') or 0),
        'registry_sha256': append_result.get('registry_sha256'),
    }


def _finalize_response_frame_payload(
    response_payload: dict[str, Any],
    *,
    request_payload: Optional[dict[str, Any]] = None,
    persist: bool = True,
    expected_parent_frame_id: Optional[str] = None,
    expected_parent_frame_sequence: Optional[int] = None,
) -> dict[str, Any]:
    total_started_at = time.perf_counter()
    raw_late_fill = response_payload.get('late_fill') if isinstance(response_payload.get('late_fill'), Mapping) else {}

    def _branch_count(key: str) -> int:
        values = raw_late_fill.get(key) if isinstance(raw_late_fill, Mapping) else None
        return len(values) if isinstance(values, list) else 0

    pending_count = _branch_count('pending_branches')
    active_count = _branch_count('active_branches')
    completed_count = _branch_count('completed_branches')
    failed_count = _branch_count('failed_branches')
    late_fill_status = str(raw_late_fill.get('status') or '').strip().lower() if raw_late_fill else ''
    if raw_late_fill and (pending_count or active_count or late_fill_status in {'pending', 'running'}):
        timing_phase = 'nonterminal_late_fill'
    elif raw_late_fill:
        timing_phase = 'terminal_late_fill'
    else:
        timing_phase = 'ordinary_response'
    finalize_timing: dict[str, Any] = {
        'kind': 'ollmo.response_frame_finalize_timing',
        'phase': timing_phase,
        'persist_requested': bool(persist),
        'persist_effective': bool(persist and not app.config.get("TESTING")),
        'late_fill_status': late_fill_status or None,
        'pending_branch_count': pending_count,
        'active_branch_count': active_count,
        'completed_branch_count': completed_count,
        'failed_branch_count': failed_count,
        'steps': [],
    }

    def _mark_step(name: str, started_at: float, **metadata: Any) -> None:
        entry: dict[str, Any] = {
            'name': name,
            'elapsed_ms': round((time.perf_counter() - started_at) * 1000, 3),
        }
        for key, value in metadata.items():
            if value not in (None, '', [], {}):
                entry[key] = value
        finalize_timing['steps'].append(entry)

    def _publish_finalize_timing(payload: dict[str, Any]) -> dict[str, Any]:
        finalize_timing['total_elapsed_ms'] = round((time.perf_counter() - total_started_at) * 1000, 3)
        runtime = dict(payload.get('runtime') or {}) if isinstance(payload.get('runtime'), Mapping) else {}
        developer_diagnostics = (
            dict(runtime.get('developer_diagnostics'))
            if isinstance(runtime.get('developer_diagnostics'), Mapping)
            else {}
        )
        developer_diagnostics['response_frame_finalize_timing'] = dict(finalize_timing)
        runtime['developer_diagnostics'] = developer_diagnostics
        payload['runtime'] = runtime
        frame = payload.get('response_frame') if isinstance(payload.get('response_frame'), Mapping) else None
        if isinstance(frame, dict):
            frame_runtime = dict(frame.get('runtime') or {}) if isinstance(frame.get('runtime'), Mapping) else {}
            frame_diagnostics = (
                dict(frame_runtime.get('developer_diagnostics'))
                if isinstance(frame_runtime.get('developer_diagnostics'), Mapping)
                else {}
            )
            frame_diagnostics['response_frame_finalize_timing'] = dict(finalize_timing)
            frame_runtime['developer_diagnostics'] = frame_diagnostics
            frame['runtime'] = frame_runtime
            current_state = frame.get('current_state') if isinstance(frame.get('current_state'), Mapping) else None
            if isinstance(current_state, dict):
                current_runtime = (
                    dict(current_state.get('runtime'))
                    if isinstance(current_state.get('runtime'), Mapping)
                    else {}
                )
                current_diagnostics = (
                    dict(current_runtime.get('developer_diagnostics'))
                    if isinstance(current_runtime.get('developer_diagnostics'), Mapping)
                    else {}
                )
                current_diagnostics['response_frame_finalize_timing'] = dict(finalize_timing)
                current_runtime['developer_diagnostics'] = current_diagnostics
                current_state['runtime'] = current_runtime
        return payload

    step_started_at = time.perf_counter()
    finalized_payload = _attach_response_status_semantics(dict(response_payload))
    _mark_step(
        'status_semantics',
        step_started_at,
        lifecycle_state=finalized_payload.get('lifecycle_state'),
        compatibility_status=finalized_payload.get('status'),
    )
    existing_frame = (
        copy.deepcopy(finalized_payload.get('response_frame'))
        if isinstance(finalized_payload.get('response_frame'), Mapping)
        else {}
    )
    if existing_frame:
        finalized_payload['response_frame'] = existing_frame
    step_started_at = time.perf_counter()
    frame_relation_added = False
    preserve_bounded_successor_relation = False
    if persist and existing_frame:
        response_id = str(finalized_payload.get('id') or existing_frame.get('response_id') or '').strip()
        parent_frame_id = str(existing_frame.get('frame_id') or '').strip()
        if response_id:
            explicit_relation = (
                dict(finalized_payload.get('frame_relation'))
                if isinstance(finalized_payload.get('frame_relation'), Mapping)
                else {}
            )
            explicit_relation_kind = str(explicit_relation.get('kind') or '').strip()
            if explicit_relation_kind == 'graph_patch_terminal_review':
                audit_identifiers = {
                    key: explicit_relation.get(key)
                    for key in (
                        'audit_key',
                        'patch_id',
                        'proposal_id',
                        'review_id',
                        'policy_review_id',
                    )
                }
                if all(
                    isinstance(value, str)
                    and bool(value)
                    and value == value.strip()
                    for value in audit_identifiers.values()
                ):
                    explicit_relation = {
                        'kind': 'graph_patch_terminal_review',
                        'reason': 'terminal_graph_patch_enforced_policy_denied',
                        **audit_identifiers,
                        'audit_only': True,
                        'executable': False,
                        'runtime_effect': 'audit_only_no_execution',
                        'owed_work': 'none',
                        'scheduled_branch_ids': [],
                    }
                else:
                    explicit_relation = {}
                    explicit_relation_kind = ''
            preserve_bounded_successor_relation = explicit_relation_kind in {
                'graph_patch_reopen_successor',
                'graph_patch_terminal_review',
                'graph_rebase_partial_successor',
                'graph_rebase_stage_successor',
            }
            finalized_payload['frame_relation'] = {
                **(explicit_relation if preserve_bounded_successor_relation else {}),
                'kind': (
                    explicit_relation_kind
                    if preserve_bounded_successor_relation
                    else 'late_fill_successor'
                    if isinstance(finalized_payload.get('late_fill'), Mapping)
                    else 'successor'
                ),
                'response_id': response_id,
                'parent_response_id': response_id,
                'parent_frame_id': parent_frame_id or None,
                'parent_frame_sequence': existing_frame.get('frame_sequence'),
            }
            frame_relation_added = True
    _mark_step(
        'frame_relation',
        step_started_at,
        had_existing_frame=bool(existing_frame),
        relation_added=frame_relation_added,
    )
    step_started_at = time.perf_counter()
    working_frame = _build_working_frame(
        request_payload=request_payload,
        response_payload=finalized_payload,
        freeze=True,
    )
    _mark_step('build_working_frame', step_started_at)
    finalized_payload['working_frame'] = working_frame
    runtime = dict(finalized_payload.get('runtime') or {}) if isinstance(finalized_payload.get('runtime'), dict) else {}
    runtime['working_frame'] = working_frame
    finalized_payload['runtime'] = runtime
    _publish_finalize_timing(finalized_payload)
    step_started_at = time.perf_counter()
    framed_payload = _hoist_response_output_surfaces(
        _attach_response_frame(finalized_payload, request_payload=request_payload)
    )
    if preserve_bounded_successor_relation:
        # The explicit relation applies to this append only. Later frames in the
        # successor wave derive their ordinary parent relation from response_frame.
        framed_payload.pop('frame_relation', None)
    _mark_step(
        'attach_response_frame_and_hoist_outputs',
        step_started_at,
        output_count=len(framed_payload.get('outputs') or []) if isinstance(framed_payload.get('outputs'), list) else None,
        artifact_count=len(framed_payload.get('artifacts') or []) if isinstance(framed_payload.get('artifacts'), list) else None,
    )
    _publish_finalize_timing(framed_payload)
    response_frame = framed_payload.get('response_frame')
    if isinstance(response_frame, Mapping):
        step_started_at = time.perf_counter()
        framed_payload['response_frame'] = _enrich_response_frame_metadata(
            response_frame,
            parent_frame=existing_frame if isinstance(existing_frame, Mapping) else None,
        )
        if not persist and existing_frame:
            framed_payload['response_frame'] = _preserve_frozen_response_frame_identity(
                framed_payload['response_frame'],
                existing_frame,
            )
        _mark_step('response_frame_metadata', step_started_at)
        _publish_finalize_timing(framed_payload)
    frame_persisted = False
    if persist and not app.config.get("TESTING"):
        step_started_at = time.perf_counter()
        try:
            _persist_output_artifact_registry_records(
                framed_payload,
                ledger_path=ARTIFACT_REGISTRY_LEDGER,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not append output artifact registry records: %s", exc)
        _mark_step('persist_output_artifact_registry_records', step_started_at)
        _publish_finalize_timing(framed_payload)
        step_started_at = time.perf_counter()
        try:
            if str(expected_parent_frame_id or '').strip():
                append_result = _append_response_frame_with_parent_cas(
                    framed_payload['response_frame'],
                    expected_parent_frame_id=str(expected_parent_frame_id).strip(),
                    expected_parent_frame_sequence=expected_parent_frame_sequence,
                    frames_dir=RESPONSE_FRAMES_DIR,
                )
                framed_payload['response_frame'] = append_result['response_frame']
            else:
                framed_payload['response_frame'] = _enrich_response_frame_for_ledger_append(
                    framed_payload['response_frame'],
                    frames_dir=RESPONSE_FRAMES_DIR,
                )
                _persist_response_frame(framed_payload['response_frame'], frames_dir=RESPONSE_FRAMES_DIR)
            frame_persisted = True
        except ResponseFrameParentCASMismatch:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not append response frame: %s", exc)
        _mark_step('persist_response_frame', step_started_at)
        if frame_persisted:
            step_started_at = time.perf_counter()
            try:
                registry_diagnostic = (
                    _register_durable_graph_rebase_readiness_observation(
                        framed_payload
                    )
                )
            except GraphRebaseReadinessRegistryError as exc:
                registry_diagnostic = {
                    'kind': 'ollmo.graph_rebase_readiness_registry_append',
                    'status': 'append_failed',
                    'runtime_effect': 'none',
                    'error': exc.as_dict(),
                }
            except Exception as exc:  # noqa: BLE001
                registry_diagnostic = {
                    'kind': 'ollmo.graph_rebase_readiness_registry_append',
                    'status': 'append_failed',
                    'runtime_effect': 'none',
                    'error': {
                        'code': 'readiness_registry_unexpected_failure',
                        'message': str(exc),
                    },
                }
            registry_status = str(registry_diagnostic.get('status') or '')
            if registry_status in {
                'append_failed',
                'hydration_binding_mismatch',
                'hydration_failed',
                'selection_failed',
                'verification_failed',
            }:
                logging.warning(
                    'Graph-rebase readiness observation was not retained: %s',
                    registry_diagnostic,
                )
            framed_runtime = (
                dict(framed_payload.get('runtime') or {})
                if isinstance(framed_payload.get('runtime'), Mapping)
                else {}
            )
            developer_diagnostics = (
                dict(framed_runtime.get('developer_diagnostics') or {})
                if isinstance(
                    framed_runtime.get('developer_diagnostics'),
                    Mapping,
                )
                else {}
            )
            developer_diagnostics['graph_rebase_readiness_registry'] = (
                registry_diagnostic
            )
            framed_runtime['developer_diagnostics'] = developer_diagnostics
            framed_payload['runtime'] = framed_runtime
            _mark_step(
                'persist_graph_rebase_readiness_observation',
                step_started_at,
                status=registry_status,
            )
    _publish_finalize_timing(framed_payload)
    return framed_payload


def _response_registry_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def _normalize_response_lookup_id(value: Any) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '', str(value or '').strip())
    return token or f'resp_{uuid.uuid4().hex}'


_RESPONSES_RUNTIME = ResponsesRuntimeOwner(
    response_lookup=_RESPONSE_LOOKUP,
    response_lookup_lock=_RESPONSE_LOOKUP_LOCK,
    response_streams=_RESPONSE_STREAMS,
    response_streams_lock=_RESPONSE_STREAMS_LOCK,
    response_late_fill_in_flight=_RESPONSE_LATE_FILL_IN_FLIGHT,
    response_late_fill_lock=_RESPONSE_LATE_FILL_LOCK,
    response_lookup_ttl_sec=RESPONSE_LOOKUP_TTL_SEC,
    normalize_response_lookup_id=lambda value: _normalize_response_lookup_id(value),
    response_registry_now_iso=lambda: _response_registry_now_iso(),
)

_RESPONSE_LOOKUP_RUNTIME = ResponseLookupRuntimeOwner(
    normalize_response_lookup_id=lambda value: _normalize_response_lookup_id(value),
    get_live_response_lookup_record=(
        lambda response_id: _RESPONSES_RUNTIME.get_response_lookup_record(response_id)
    ),
    load_wire_payload_from_index=(
        lambda response_id: _response_wire_payload_from_index(response_id)
    ),
    project_fallback_payload=(
        lambda payload: _response_wire_fallback_payload(payload)
    ),
    recover_response_lookup_record=(
        lambda response_id: _recover_response_lookup_record_from_frames(response_id)
    ),
    project_late_fill=lambda payload: _response_wire_late_fill_handle(payload),
    project_surface=lambda value: _response_wire_surface_handle(value),
    derive_lifecycle_state=(
        lambda payload, **kwargs: derive_response_lifecycle_state(payload, **kwargs)
    ),
    response_payload_message_id=lambda payload: _response_payload_message_id(payload),
    response_registry_now_iso=lambda: _response_registry_now_iso(),
    response_frames_dir_getter=lambda: RESPONSE_FRAMES_DIR,
    inspect_recovered_cache=(
        lambda response_id, ledger_path, expected_state: (
            _inspect_response_frame_recovery_cache(
                response_id,
                frames_dir=RESPONSE_FRAMES_DIR,
                expected_ledger_path=ledger_path,
                expected_ledger_state=expected_state,
            )
        )
    ),
    advance_recovered_cache_checkpoint=(
        lambda response_id, ledger_stat: (
            _RESPONSES_RUNTIME.advance_response_recovery_checkpoint(
                response_id,
                ledger_stat,
            )
        )
    ),
    response_lookup_ttl_sec=RESPONSE_LOOKUP_TTL_SEC,
    now_ts=lambda: time.time(),
    new_message_id=lambda: f'msg_{uuid.uuid4().hex}',
)


def _prune_response_lookup_registry(now_ts: Optional[float] = None) -> None:
    _RESPONSES_RUNTIME._prune_response_lookup_registry(now_ts)


def _register_response_lookup(
    *,
    response_id: str,
    message_id: str,
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    mode: str,
    route_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _RESPONSES_RUNTIME.register_response_lookup(
        response_id=response_id,
        message_id=message_id,
        instance_id=instance_id,
        model_name=model_name,
        backend=backend,
        capability=capability,
        mode=mode,
        route_payload=route_payload,
    )


def _format_response_sse_event(event_name: str, payload: Mapping[str, Any]) -> str:
    normalized_event_name = str(event_name or 'message').strip()[:256] or 'message'
    response_source = payload.get('response') if isinstance(payload.get('response'), Mapping) else None
    bounded_payload = _response_wire_enforce_outer_envelope_byte_ceiling(
        payload,
        source_payload=response_source,
        source=f'sse_{normalized_event_name}_outer_envelope_byte_ceiling',
    )
    return (
        f"event: {normalized_event_name}\n"
        f"data: {json.dumps(bounded_payload, ensure_ascii=False)}\n\n"
    )


_RESPONSE_STREAM_SAFE_PAYLOAD_KEYS = (
    'id',
    'response_id',
    'object',
    'status',
    'lifecycle_state',
    'canonical_status_field',
    'status_compatibility',
    'status_semantics',
    'state_version',
    'model',
    'backend',
    'capability',
    'mode',
    'instance_id',
    'route_source',
    'route_reason',
    'route_router_instance_id',
    'route_router_model',
    'route_artifact_ref',
    'route_artifact_path',
    'route_reuse_last_artifact',
    'reference_image_count',
    'reference_image_kind',
    'context_mode',
    'context_reason',
    'output_text',
    'output',
    'artifacts',
    'outputs',
    'output_slots',
    'output_branches',
    'artifact_bundles',
    'artifactBundles',
    'artifactBundle',
    'saved_image_path',
    'saved_audio_path',
    'saved_text_path',
    'input_artifacts',
    'surface_state',
    'status_lookup',
    'error',
    'usage',
    'created_at',
    'updated_at',
)


_RESPONSE_STREAM_HEAVY_PAYLOAD_KEYS = {
    'response_frame',
    'runtime',
    'working_frame',
    'work_tree',
    'phase_payload',
}


def _fallback_compact_response_payload_for_stream(payload: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _RESPONSE_STREAM_SAFE_PAYLOAD_KEYS:
        value = payload.get(key)
        if value not in (None, '', [], {}):
            compact[key] = value
    late_fill = payload.get('late_fill') if isinstance(payload.get('late_fill'), Mapping) else {}
    if late_fill:
        compact_late_fill = {
            key: late_fill.get(key)
            for key in (
                'status',
                'route_source',
                'surface_state',
                'pending_branch_count',
                'active_branch_count',
                'completed_branch_count',
                'failed_branch_count',
            )
            if late_fill.get(key) not in (None, '', [], {})
        }
        if compact_late_fill:
            compact['late_fill'] = compact_late_fill
    compact['ui_compact'] = True
    return compact


def _response_lookup_record_from_stream_payload(
    response_payload: Mapping[str, Any],
    response_id: str = '',
) -> dict[str, Any]:
    payload = response_payload if isinstance(response_payload, Mapping) else {}
    target = payload.get('target') if isinstance(payload.get('target'), Mapping) else {}
    normalized_response_id = _normalize_response_lookup_id(
        response_id
        or payload.get('id')
        or payload.get('response_id')
        or ''
    )
    lifecycle_state = str(
        payload.get('lifecycle_state')
        or derive_response_lifecycle_state(payload, requested_status=payload.get('status'))
        or payload.get('status')
        or ''
    ).strip()
    return {
        'id': normalized_response_id,
        'message_id': _response_payload_message_id(dict(payload)) or '',
        'instance_id': str(payload.get('instance_id') or target.get('instance_id') or '').strip(),
        'model_name': str(payload.get('model') or payload.get('model_name') or target.get('model') or '').strip(),
        'backend': str(payload.get('backend') or target.get('backend') or '').strip(),
        'capability': str(payload.get('capability') or target.get('capability') or '').strip(),
        'mode': str(payload.get('mode') or target.get('mode') or payload.get('capability') or 'chat').strip() or 'chat',
        'status': str(payload.get('status') or 'in_progress').strip() or 'in_progress',
        'lifecycle_state': lifecycle_state,
        'output_text': str(payload.get('output_text') or ''),
        'error_message': None,
        'response_payload': dict(payload),
        'route_payload': payload.get('route') if isinstance(payload.get('route'), dict) else {},
    }


def _compact_response_payload_for_stream(
    response_payload: Any,
    response_id: str = '',
) -> dict[str, Any]:
    if not isinstance(response_payload, Mapping):
        return {}
    if (
        response_payload.get('ui_compact') is True
        and not any(key in response_payload for key in _RESPONSE_STREAM_HEAVY_PAYLOAD_KEYS)
    ):
        return _response_wire_enforce_byte_ceiling(
            response_payload,
            source_payload=response_payload,
            source='stream_ui_projection_emergency_byte_ceiling',
        )
    if (
        response_payload.get('compact') is True
        and response_payload.get('object') == 'response.status'
        and not any(key in response_payload for key in _RESPONSE_STREAM_HEAVY_PAYLOAD_KEYS)
    ):
        compact_status = dict(response_payload)
        compact_status['ui_compact'] = True
        return _response_wire_enforce_byte_ceiling(
            compact_status,
            source_payload=compact_status,
            source='stream_status_projection_emergency_byte_ceiling',
        )
    try:
        record = _response_lookup_record_from_stream_payload(response_payload, response_id=response_id)
        ui_payload = _build_response_ui_lookup_payload(record)
        return _response_wire_enforce_byte_ceiling(
            ui_payload,
            source_payload=ui_payload,
            source='stream_projection_emergency_byte_ceiling',
        )
    except Exception:  # noqa: BLE001
        logging.exception('Could not compact response stream payload.')
        fallback = _fallback_compact_response_payload_for_stream(response_payload)
        return _response_wire_enforce_byte_ceiling(
            fallback,
            source_payload=fallback,
            source='stream_fallback_emergency_byte_ceiling',
        )


def _compact_response_stream_event(response_id: str, raw_event: str) -> str:
    text = str(raw_event or '')
    if 'response' not in text or 'data:' not in text:
        return text
    event_name = ''
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith('event:'):
            event_name = line.split(':', 1)[1].strip()
        elif line.startswith('data:'):
            data_lines.append(line.split(':', 1)[1].lstrip())
    if not data_lines:
        return text
    try:
        payload = json.loads('\n'.join(data_lines))
    except Exception:  # noqa: BLE001
        return text
    if not isinstance(payload, dict) or not isinstance(payload.get('response'), Mapping):
        return text
    compact_response = _compact_response_payload_for_stream(
        payload.get('response'),
        response_id=response_id,
    )
    if not compact_response:
        return text
    payload['response'] = compact_response
    if isinstance(payload.get('late_fill'), Mapping):
        payload['late_fill'] = compact_response.get('late_fill') or payload.get('late_fill')
    return _format_response_sse_event(
        event_name or str(payload.get('type') or 'message').strip() or 'message',
        payload,
    )


def _compact_response_stream_events(response_id: str, events: list[str]) -> list[str]:
    return [
        _compact_response_stream_event(response_id, event)
        for event in (events or [])
    ]


def _build_canonical_response_stream_events_for_ui(response_payload: dict):
    compact_payload = _compact_response_payload_for_stream(response_payload)
    yield from _build_canonical_response_stream_events(compact_payload)


def _response_stream_is_registered(response_id: str) -> bool:
    normalized_id = _normalize_response_lookup_id(response_id)
    with _RESPONSE_STREAMS_LOCK:
        return normalized_id in _RESPONSE_STREAMS


def _response_stream_payload_is_terminal(payload: Mapping[str, Any]) -> bool:
    status_semantics = (
        payload.get('status_semantics')
        if isinstance(payload.get('status_semantics'), Mapping)
        else {}
    )
    lifecycle_state = str(
        payload.get('lifecycle_state')
        or status_semantics.get('canonical_lifecycle_state')
        or ''
    ).strip().lower()
    has_open_continuation = bool(status_semantics.get('has_open_continuation'))
    is_terminal = bool(
        status_semantics.get('is_terminal')
        or status_semantics.get('terminal')
        or _response_lifecycle_is_terminal(lifecycle_state)
    )
    return is_terminal and not has_open_continuation


def _response_stream_payload_requires_attention(payload: Mapping[str, Any]) -> bool:
    status_semantics = (
        payload.get('status_semantics')
        if isinstance(payload.get('status_semantics'), Mapping)
        else {}
    )
    lifecycle_state = str(
        payload.get('lifecycle_state')
        or status_semantics.get('canonical_lifecycle_state')
        or ''
    ).strip().lower()
    has_open_continuation = bool(status_semantics.get('has_open_continuation'))
    has_actionable_repair = bool(status_semantics.get('has_actionable_repair'))
    return (
        not has_open_continuation
        and (
            has_actionable_repair
            or lifecycle_state in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
        )
    )


def _publish_response_lookup_stream_update(record: Mapping[str, Any]) -> None:
    response_id = str(record.get('id') or '').strip()
    if not response_id or not _response_stream_is_registered(response_id):
        return
    try:
        ui_payload = _build_response_ui_lookup_payload(dict(record))
    except Exception:  # noqa: BLE001
        logging.exception('Could not build response stream update payload for %s.', response_id)
        return
    ui_payload['ui_compact'] = True
    ui_payload = _response_wire_enforce_byte_ceiling(
        ui_payload,
        source_payload=ui_payload,
        source='stream_publish_emergency_byte_ceiling',
    )

    events = [
        _format_response_sse_event(
            'response.state.updated',
            {'type': 'response.state.updated', 'response': ui_payload},
        )
    ]
    if isinstance(ui_payload.get('late_fill'), Mapping):
        late_fill_payload = ui_payload.get('late_fill')
        events.append(
            _format_response_sse_event(
                'response.late_fill.updated',
                {
                    'type': 'response.late_fill.updated',
                    'response': ui_payload,
                    'late_fill': late_fill_payload,
                },
            )
        )
        for branch in late_fill_payload.get('branch_progress') or []:
            if not isinstance(branch, Mapping):
                continue
            branch_payload = {
                key: branch.get(key)
                for key in (
                    'branch_id',
                    'phase_id',
                    'capability',
                    'status',
                    'progress_stage',
                    'instance_id',
                    'timing',
                    'error',
                    'updated_at',
                )
                if branch.get(key) not in (None, '', [], {})
            }
            if not branch_payload.get('branch_id'):
                continue
            events.append(
                _format_response_sse_event(
                    'response.late_fill.branch.updated',
                    {
                        'type': 'response.late_fill.branch.updated',
                        'response_id': response_id,
                        'branch': branch_payload,
                        'late_fill_status': late_fill_payload.get('status'),
                        'pending_count': late_fill_payload.get('pending_count'),
                        'active_count': late_fill_payload.get('active_count'),
                        'completed_count': late_fill_payload.get('completed_count'),
                        'failed_count': late_fill_payload.get('failed_count'),
                    },
                )
            )

    terminal_stream = bool(
        isinstance(ui_payload.get('late_fill'), Mapping)
        and _response_stream_payload_is_terminal(ui_payload)
    )
    attention_stream = bool(
        isinstance(ui_payload.get('late_fill'), Mapping)
        and _response_stream_payload_requires_attention(ui_payload)
    )
    if terminal_stream:
        events.append(
            _format_response_sse_event(
                'response.completed',
                {'type': 'response.completed', 'response': ui_payload},
            )
        )
    elif attention_stream:
        events.append(
            _format_response_sse_event(
                'response.requires_action',
                {'type': 'response.requires_action', 'response': ui_payload},
            )
        )
    _append_response_stream_events(response_id, events, done=terminal_stream or attention_stream)


def _publish_response_lookup_stream_status_update(record: Mapping[str, Any]) -> None:
    response_id = str(record.get('id') or '').strip()
    if not response_id or not _response_stream_is_registered(response_id):
        return
    try:
        status_payload = _build_response_status_lookup_payload(dict(record))
    except Exception:  # noqa: BLE001
        logging.exception('Could not build compact response stream status payload for %s.', response_id)
        return

    status_payload = dict(status_payload)
    status_payload['ui_compact'] = True
    status_payload = _response_wire_enforce_byte_ceiling(
        status_payload,
        source_payload=status_payload,
        source='stream_status_publish_emergency_byte_ceiling',
    )
    status_payload['ui_compact'] = True
    events = [
        _format_response_sse_event(
            'response.state.updated',
            {'type': 'response.state.updated', 'response': status_payload},
        )
    ]
    late_fill_payload = (
        status_payload.get('late_fill')
        if isinstance(status_payload.get('late_fill'), Mapping)
        else {}
    )
    if late_fill_payload:
        events.append(
            _format_response_sse_event(
                'response.late_fill.updated',
                {
                    'type': 'response.late_fill.updated',
                    'response': status_payload,
                    'late_fill': late_fill_payload,
                },
            )
        )
        for branch in late_fill_payload.get('branch_progress') or []:
            if not isinstance(branch, Mapping):
                continue
            branch_payload = {
                key: branch.get(key)
                for key in (
                    'branch_id',
                    'phase_id',
                    'capability',
                    'status',
                    'progress_stage',
                    'instance_id',
                    'timing',
                    'error',
                    'updated_at',
                )
                if branch.get(key) not in (None, '', [], {})
            }
            if not branch_payload.get('branch_id'):
                continue
            events.append(
                _format_response_sse_event(
                    'response.late_fill.branch.updated',
                    {
                        'type': 'response.late_fill.branch.updated',
                        'response_id': response_id,
                        'branch': branch_payload,
                        'late_fill_status': late_fill_payload.get('status'),
                        'pending_count': late_fill_payload.get('pending_count'),
                        'active_count': late_fill_payload.get('active_count'),
                        'completed_count': late_fill_payload.get('completed_count'),
                        'failed_count': late_fill_payload.get('failed_count'),
                    },
                )
            )

    _append_response_stream_events(response_id, events, done=False)


def _touch_response_lookup(
    response_id: str,
    *,
    status: Optional[str] = None,
    output_text: Optional[str] = None,
    error_message: Optional[str] = None,
    response_payload: Optional[dict[str, Any]] = None,
    publish_stream: bool = True,
    stream_view: str = 'ui',
) -> Optional[dict[str, Any]]:
    record = _RESPONSES_RUNTIME.touch_response_lookup(
        response_id,
        status=status,
        output_text=output_text,
        error_message=error_message,
        response_payload=response_payload,
    )
    if record and response_payload is not None and publish_stream:
        if str(stream_view or '').strip().lower() == 'status':
            _publish_response_lookup_stream_status_update(record)
        else:
            _publish_response_lookup_stream_update(record)
    return record


def _response_lookup_frame_sequence_value(payload: Mapping[str, Any]) -> Optional[int]:
    return _runtime_response_lookup_frame_sequence_value(payload)


def _response_lookup_frame_id_value(payload: Mapping[str, Any]) -> str:
    return _runtime_response_lookup_frame_id_value(payload)


def _response_lookup_has_ledger_durability(payload: Mapping[str, Any]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    durability = payload.get('durability') if isinstance(payload.get('durability'), Mapping) else {}
    return str(durability.get('source') or '').strip() == 'response_frame_ledger'


def _response_lookup_terminal_projection_count(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    total = 0
    for item in values:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get('status') or item.get('lifecycle') or '').strip().lower()
        if status in {'completed', 'fulfilled', 'blocked', 'failed', 'cancelled', 'waived', 'superseded'}:
            total += 1
    return total


def _response_lookup_public_projection_score(payload: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    if not isinstance(payload, Mapping):
        return (0, 0, 0, 0, 0)
    artifacts = payload.get('artifacts') if isinstance(payload.get('artifacts'), list) else []
    outputs = payload.get('outputs') if isinstance(payload.get('outputs'), list) else []
    output_slots = payload.get('output_slots') if isinstance(payload.get('output_slots'), list) else []
    return (
        len(artifacts),
        _response_lookup_terminal_projection_count(outputs),
        _response_lookup_terminal_projection_count(output_slots),
        len(outputs),
        len(output_slots),
    )


def _response_lookup_lifecycle_value(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        return ''
    return str(
        derive_response_lifecycle_state(payload, requested_status=payload.get('status'))
        or payload.get('lifecycle_state')
        or payload.get('status')
        or ''
    ).strip().lower()


def _response_lookup_record_should_refresh_from_frame(
    record: Mapping[str, Any],
    latest_state: Mapping[str, Any],
) -> bool:
    if not isinstance(record, Mapping) or not isinstance(latest_state, Mapping):
        return False
    latest_payload = latest_state.get('response_payload') if isinstance(latest_state.get('response_payload'), Mapping) else {}
    if not latest_payload:
        return False
    current_payload = record.get('response_payload') if isinstance(record.get('response_payload'), Mapping) else {}
    latest_sequence = _response_lookup_frame_sequence_value(latest_payload)
    current_sequence = _response_lookup_frame_sequence_value(current_payload)
    if latest_sequence is not None and current_sequence is not None and latest_sequence > current_sequence:
        return True
    same_sequence = latest_sequence is not None and latest_sequence == current_sequence
    latest_frame_id = _response_lookup_frame_id_value(latest_payload)
    current_frame_id = _response_lookup_frame_id_value(current_payload)
    same_frame_id = bool(latest_frame_id and latest_frame_id == current_frame_id)
    if same_sequence or same_frame_id:
        if latest_frame_id and current_frame_id != latest_frame_id:
            return True
        if (
            _response_lookup_has_ledger_durability(latest_payload)
            and not _response_lookup_has_ledger_durability(current_payload)
        ):
            return True
    latest_score = _response_lookup_public_projection_score(latest_payload)
    current_score = _response_lookup_public_projection_score(current_payload)
    if any(latest > current for latest, current in zip(latest_score, current_score)):
        return True
    latest_lifecycle = _response_lookup_lifecycle_value(latest_payload)
    current_lifecycle = _response_lookup_lifecycle_value(current_payload)
    record_lifecycle = str(record.get('lifecycle_state') or '').strip().lower()
    if (
        (same_sequence or same_frame_id)
        and latest_lifecycle in _ACTIONABLE_RESPONSE_LIFECYCLE_STATES
        and record_lifecycle in {'completed', 'late_fill_completed', 'frozen'}
    ):
        return True
    if latest_lifecycle == 'completed' and current_lifecycle and current_lifecycle != 'completed':
        return True
    return False


def _get_response_lookup_record(
    response_id: str,
    *,
    recover_missing: bool = True,
) -> Optional[dict[str, Any]]:
    """Return a live record and optionally recover one missing from memory.

    Routes that need the detailed persisted-frame error perform recovery
    themselves exactly once.  Other compatibility callers retain the historic
    automatic recovery behavior.
    """

    record = _RESPONSES_RUNTIME.get_response_lookup_record(response_id)
    if record:
        normalized_id = _normalize_response_lookup_id(response_id)
        latest_state = _load_latest_response_state(normalized_id, frames_dir=RESPONSE_FRAMES_DIR)
        if latest_state.get('ok') and _response_lookup_record_should_refresh_from_frame(record, latest_state):
            recovered_record, _error, _status_code = _recover_response_lookup_record_from_frames(normalized_id)
            if recovered_record:
                return recovered_record
        return record
    if not recover_missing:
        return None
    recovered_record, _error, _status_code = _recover_response_lookup_record_from_frames(response_id)
    return recovered_record


def _recover_response_lookup_record_from_frames(
    response_id: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], int]:
    normalized_id = _normalize_response_lookup_id(response_id)
    state = _load_latest_response_state(normalized_id, frames_dir=RESPONSE_FRAMES_DIR)
    if not state.get('ok'):
        error = state.get('error') if isinstance(state.get('error'), dict) else {'message': 'Response not found.'}
        return None, error, int(state.get('status_code') or 404)
    payload = state.get('response_payload') if isinstance(state.get('response_payload'), dict) else {}
    if not payload:
        return None, {'code': 'response_frame_unrecoverable', 'message': 'Response frame is not recoverable.'}, 409
    response_frame = state.get('response_frame') if isinstance(state.get('response_frame'), dict) else {}
    target = response_frame.get('target') if isinstance(response_frame.get('target'), Mapping) else {}
    lifecycle_state = derive_response_lifecycle_state(payload, requested_status=payload.get('status'))
    now_ts = time.time()
    ledger_path = Path(str(state.get('ledger_path') or (Path(RESPONSE_FRAMES_DIR) / 'responses.jsonl')))
    loaded_ledger_state = (
        state.get('ledger_state')
        if isinstance(state.get('ledger_state'), Mapping)
        else None
    )
    ledger_stat_token: Optional[dict[str, int]] = (
        {
            key: int(loaded_ledger_state[key])
            for key in ('size_bytes', 'mtime_ns', 'device', 'inode')
            if key in loaded_ledger_state
        }
        if isinstance(loaded_ledger_state, Mapping)
        else None
    )
    record = {
        'id': normalized_id,
        'message_id': _response_payload_message_id(payload) or f'msg_{uuid.uuid4().hex}',
        'instance_id': str(payload.get('instance_id') or target.get('instance_id') or '').strip(),
        'model_name': str(payload.get('model') or target.get('model') or '').strip(),
        'backend': str(payload.get('backend') or target.get('backend') or '').strip(),
        'capability': str(payload.get('capability') or target.get('capability') or '').strip(),
        'mode': str(payload.get('mode') or target.get('mode') or 'chat').strip() or 'chat',
        'status': str(payload.get('status') or response_frame.get('status') or 'completed').strip() or 'completed',
        'lifecycle_state': lifecycle_state,
        'output_text': str(payload.get('output_text') or ''),
        'error_message': None,
        'response_payload': payload,
        'route_payload': response_frame.get('route') if isinstance(response_frame.get('route'), dict) else {},
        'created_at': _response_registry_now_iso(),
        'updated_at': _response_registry_now_iso(),
        'expires_at_ts': now_ts + RESPONSE_LOOKUP_TTL_SEC,
        'lookup_source': 'response_frame_ledger',
        'recovered_from_response_frame': True,
        'recovered_frame_count': int(state.get('frame_count') or 0),
        'response_frame_ledger_path': str(state.get('ledger_path') or ''),
        'response_frame_ledger_stat': ledger_stat_token,
        # Exceptional canonical recovery is allowed to hydrate once. Cache its
        # bounded counterpart with the ledger stat so a stale index cannot turn
        # every normal status poll into another full ledger scan.
        'bounded_response_payload': _response_wire_fallback_payload(payload),
    }
    with _RESPONSE_LOOKUP_LOCK:
        _prune_response_lookup_registry(now_ts)
        _RESPONSE_LOOKUP[normalized_id] = dict(record)
    return dict(record), None, 200


_RESPONSE_WIRE_INLINE_LIMIT_BYTES = (
    _response_wire_policy.RESPONSE_WIRE_INLINE_LIMIT_BYTES
)
# Leave room for Flask's trailing newline and for the small protocol wrapper
# around a projected response.  The public contract is an 8 MiB serialized
# envelope, not an 8 MiB nested object which may subsequently be wrapped.
_RESPONSE_WIRE_SERIALIZATION_RESERVE_BYTES = (
    _response_wire_policy.RESPONSE_WIRE_SERIALIZATION_RESERVE_BYTES
)
_RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES = (
    _response_wire_policy.RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES
)
_RESPONSE_WIRE_TEXT_PREVIEW_CHARS = (
    _response_wire_policy.IN_MEMORY_TEXT_PREVIEW_CHARS
)
_RESPONSE_WIRE_COLLECTION_LIMIT = _response_wire_policy.IN_MEMORY_COLLECTION_LIMIT
_RESPONSE_WIRE_PUBLIC_KEYS = _response_wire_policy.IN_MEMORY_PUBLIC_KEYS


def _response_wire_json_size_up_to(value: Any, *, limit: int) -> int:
    return _response_wire_policy.json_size_up_to(value, limit=limit)


def _response_wire_digest_ref(value: Any, *, json_path: str) -> dict[str, Any]:
    return _response_wire_policy.digest_ref(value, json_path=json_path)


def _response_wire_text_preview(
    value: Any,
    *,
    json_path: str,
) -> tuple[str, Optional[dict[str, Any]]]:
    return _response_wire_policy.text_preview(value, json_path=json_path)


def _response_wire_error_handle(
    value: Any,
    *,
    json_path: str,
) -> tuple[Any, Optional[dict[str, Any]]]:
    return _response_wire_policy.error_handle(value, json_path=json_path)


def _response_wire_artifact_handle(value: Any) -> dict[str, Any]:
    return _response_wire_policy.artifact_handle(value)


def _response_wire_output_handle(value: Any) -> dict[str, Any]:
    return _response_wire_policy.output_handle(value)


def _response_wire_branch_handle(value: Any) -> dict[str, Any]:
    return _response_wire_policy.branch_handle(value)


def _response_wire_surface_handle(value: Any) -> dict[str, Any]:
    return _response_wire_policy.surface_handle(value)


def _response_wire_frame_cas_handle(value: Any) -> dict[str, Any]:
    return _response_wire_policy.frame_cas_handle(value)


def _response_wire_late_fill_handle(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _response_wire_policy.late_fill_handle(payload)


def _response_wire_emergency_projection(
    response_payload: Mapping[str, Any],
    *,
    source: str,
    limit_bytes: int = _RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
) -> dict[str, Any]:
    return _response_wire_policy.emergency_projection(
        response_payload,
        source=source,
        limit_bytes=limit_bytes,
        output_message_projector=(
            lambda value: _response_lookup_output_message_for_ui(value)
        ),
    )


def _response_wire_snapshot_ref_handle(value: Any) -> dict[str, Any]:
    return _response_wire_policy.snapshot_ref_handle(value)


def _response_wire_frame_from_memory(
    value: Any,
    *,
    omitted: dict[str, Any],
) -> dict[str, Any]:
    return _response_wire_policy.frame_from_memory(value, omitted=omitted)


def _response_wire_fallback_payload(
    response_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _response_wire_policy.fallback_payload(
        response_payload,
        artifact_projector=lambda value: _response_lookup_artifact_for_ui(value),
        output_projector=lambda value: _response_lookup_output_for_ui(value),
        output_message_projector=(
            lambda value: _response_lookup_output_message_for_ui(value)
        ),
        late_fill_projector=(
            lambda payload: _response_lookup_late_fill_for_ui(payload)
        ),
        emergency_projector=(
            lambda payload, **kwargs: _response_wire_emergency_projection(
                payload,
                **kwargs,
            )
        ),
    )


def _response_wire_enforce_byte_ceiling(
    projected: Mapping[str, Any],
    *,
    source_payload: Optional[Mapping[str, Any]] = None,
    source: str,
) -> dict[str, Any]:
    return _response_wire_policy.enforce_byte_ceiling(
        projected,
        source_payload=source_payload,
        source=source,
        emergency_projector=(
            lambda payload, **kwargs: _response_wire_emergency_projection(
                payload,
                **kwargs,
            )
        ),
    )


def _response_wire_outer_value_handle(value: Any, *, json_path: str) -> Any:
    return _response_wire_policy.outer_value_handle(value, json_path=json_path)


def _response_wire_enforce_outer_envelope_byte_ceiling(
    envelope: Mapping[str, Any],
    *,
    source_payload: Optional[Mapping[str, Any]] = None,
    source: str,
    response_key: str = 'response',
    limit_bytes: int = _RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES,
) -> dict[str, Any]:
    """Enforce the byte contract on the serialized outer protocol object.

    Projecting only ``envelope['response']`` is insufficient: retry/control
    JSON and SSE add their own fields after that projection.  This helper
    budgets the response against those fields and has a final typed fallback
    for an independently oversized wrapper value.
    """

    effective_limit = max(4096, min(int(limit_bytes), _RESPONSE_WIRE_JSON_PAYLOAD_LIMIT_BYTES))
    if _response_wire_json_size_up_to(envelope, limit=effective_limit) <= effective_limit:
        return copy.deepcopy(dict(envelope))

    bounded = dict(envelope)
    nested_response = bounded.get(response_key)
    response_source = (
        source_payload
        if isinstance(source_payload, Mapping)
        else nested_response
        if isinstance(nested_response, Mapping)
        else {}
    )
    if isinstance(nested_response, Mapping):
        wrapper_without_response = dict(bounded)
        wrapper_without_response.pop(response_key, None)
        wrapper_size = _response_wire_json_size_up_to(
            wrapper_without_response,
            limit=effective_limit,
        )
        nested_budget = max(2048, effective_limit - min(wrapper_size, effective_limit) - 512)
        bounded[response_key] = _response_wire_emergency_projection(
            response_source,
            source=f'{source}_nested_response',
            limit_bytes=nested_budget,
        )
        if _response_wire_json_size_up_to(bounded, limit=effective_limit) <= effective_limit:
            return copy.deepcopy(bounded)

    # A duplicated late-fill object or another outer field can itself consume
    # the envelope.  Replace only those independently oversized wrapper bodies
    # with typed handles and digest identity.
    for key, value in list(bounded.items()):
        if key == response_key or value in (None, '', [], {}):
            continue
        if isinstance(value, str) and len(value) <= 4096:
            continue
        if isinstance(value, (bool, int, float)):
            continue
        if _response_wire_json_size_up_to(value, limit=256 * 1024) <= 256 * 1024:
            continue
        bounded[key] = _response_wire_outer_value_handle(
            value,
            json_path=f'envelope.{key}',
        )

    if isinstance(bounded.get(response_key), Mapping):
        wrapper_without_response = dict(bounded)
        wrapper_without_response.pop(response_key, None)
        wrapper_size = _response_wire_json_size_up_to(
            wrapper_without_response,
            limit=effective_limit,
        )
        nested_budget = max(2048, effective_limit - min(wrapper_size, effective_limit) - 512)
        bounded[response_key] = _response_wire_emergency_projection(
            response_source,
            source=f'{source}_rebalanced_response',
            limit_bytes=nested_budget,
        )
    if _response_wire_json_size_up_to(bounded, limit=effective_limit) <= effective_limit:
        return copy.deepcopy(bounded)

    absolute: dict[str, Any] = {}
    for key in ('type', 'object', 'status', 'action'):
        value = envelope.get(key)
        if isinstance(value, str) and value:
            absolute[key] = value[:512]
        elif isinstance(value, (bool, int, float)):
            absolute[key] = value
    if response_key in envelope and isinstance(response_source, Mapping):
        absolute[response_key] = _response_wire_emergency_projection(
            response_source,
            source=f'{source}_absolute_response',
            limit_bytes=64 * 1024,
        )
    control = envelope.get('control')
    if isinstance(control, Mapping):
        absolute['control'] = _response_wire_outer_value_handle(
            control,
            json_path='envelope.control',
        )
    absolute['wire_projection'] = {
        'kind': 'ollmo.response_wire_projection',
        'version': 1,
        'runtime_effect': 'none',
        'source': source,
        'bounded': True,
        'inline_limit_bytes': _RESPONSE_WIRE_INLINE_LIMIT_BYTES,
        'outer_envelope_emergency': True,
        'envelope_ref': _response_wire_digest_ref(envelope, json_path='envelope'),
    }
    if _response_wire_json_size_up_to(absolute, limit=effective_limit) <= effective_limit:
        return absolute

    # This can only be reached with pathological scalar wrappers.  Preserve a
    # small response/frame identity and a digest binding, never an oversized
    # best-effort object.
    minimal_response = (
        _response_wire_emergency_projection(
            response_source,
            source=f'{source}_absolute_minimal_response',
            limit_bytes=16 * 1024,
        )
        if isinstance(response_source, Mapping)
        else {}
    )
    return {
        **({'response': minimal_response} if minimal_response else {}),
        'wire_projection': {
            'kind': 'ollmo.response_wire_projection',
            'version': 1,
            'runtime_effect': 'none',
            'source': source,
            'bounded': True,
            'inline_limit_bytes': _RESPONSE_WIRE_INLINE_LIMIT_BYTES,
            'outer_envelope_absolute_minimal': True,
            'envelope_ref': _response_wire_digest_ref(envelope, json_path='envelope'),
        },
    }


def _response_wire_payload_from_index(
    response_id: str,
    *,
    include_observation: bool = False,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    normalized_id = _normalize_response_lookup_id(response_id)
    index_state = _load_response_frame_index(frames_dir=RESPONSE_FRAMES_DIR)
    state = _load_latest_response_wire_state(
        normalized_id,
        frames_dir=RESPONSE_FRAMES_DIR,
        index_state=index_state,
    )
    if not state.get('ok'):
        return None, state
    payload = copy.deepcopy(state.get('response_payload') or {})
    payload['durability'] = {
        'source': 'response_frame_ledger',
        'recovered': True,
        'index_used': True,
        'ledger_fallback_used': False,
        'sidecar_hydration': 'none',
        'frame_id': _response_lookup_frame_id_value(payload) or None,
        'frame_sequence': _response_lookup_frame_sequence_value(payload),
        'ledger_path': state.get('ledger_path'),
        'index_path': state.get('index_path'),
    }
    if include_observation:
        observation = _load_latest_response_observation_state(
            normalized_id,
            frames_dir=RESPONSE_FRAMES_DIR,
            index_state=index_state,
        )
        if observation.get('ok'):
            observed_payload = (
                observation.get('response_payload')
                if isinstance(observation.get('response_payload'), Mapping)
                else {}
            )
            observed_runtime = (
                observed_payload.get('runtime')
                if isinstance(observed_payload.get('runtime'), Mapping)
                else {}
            )
            if observed_runtime:
                payload['runtime'] = copy.deepcopy(observed_runtime)
            observed_request = (
                observed_payload.get('request')
                if isinstance(observed_payload.get('request'), Mapping)
                else {}
            )
            if observed_request:
                payload['request'] = copy.deepcopy(observed_request)
            projection = (
                dict(payload.get('wire_projection'))
                if isinstance(payload.get('wire_projection'), Mapping)
                else {}
            )
            projection['bounded_developer_observation'] = True
            projection['observation_sidecar_hydration'] = 'selected_graph_rebase_truth_only'
            payload['wire_projection'] = projection
        else:
            projection = (
                dict(payload.get('wire_projection'))
                if isinstance(payload.get('wire_projection'), Mapping)
                else {}
            )
            projection['bounded_developer_observation'] = False
            projection['observation_error'] = copy.deepcopy(observation.get('error') or {})
            payload['wire_projection'] = projection
    return payload, state


def _response_lookup_record_from_wire_payload(
    response_id: str,
    payload: Mapping[str, Any],
    *,
    lookup_source: str,
) -> dict[str, Any]:
    return _RESPONSE_LOOKUP_RUNTIME.response_lookup_record_from_wire_payload(
        response_id,
        payload,
        lookup_source=lookup_source,
    )


def _response_wire_record_with_lookup_truth(
    projected_record: dict[str, Any],
    source_record: Mapping[str, Any],
) -> dict[str, Any]:
    return _RESPONSE_LOOKUP_RUNTIME.response_wire_record_with_lookup_truth(
        projected_record,
        source_record,
    )


def _response_wire_frame_identity(payload: Mapping[str, Any]) -> Optional[tuple[str, int]]:
    return _runtime_response_wire_frame_identity(payload)


def _response_wire_overlay_live_state(
    projected: Mapping[str, Any],
    live_record: Mapping[str, Any],
) -> dict[str, Any]:
    return _RESPONSE_LOOKUP_RUNTIME.response_wire_overlay_live_state(
        projected,
        live_record,
    )


def _response_wire_ledger_frame_response_id(frame: Mapping[str, Any]) -> str:
    return _response_frame_ledger_record_response_id(frame)


def _response_frame_recovered_cache_valid(
    record: Mapping[str, Any],
    *,
    response_id: str,
) -> bool:
    return _RESPONSE_LOOKUP_RUNTIME.response_frame_recovered_cache_valid(
        record,
        response_id=response_id,
    )


def _project_response_payload_for_wire(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Serialize public response truth without rehydrating canonical sidecars."""

    response_id = str(response_payload.get('id') or response_payload.get('response_id') or '').strip()
    projected: Optional[dict[str, Any]] = None
    indexed_projection_rejected: Optional[dict[str, Any]] = None
    if response_id:
        projected, _state = _response_wire_payload_from_index(response_id)
        if projected is not None:
            source_identity = _response_wire_frame_identity(response_payload)
            indexed_identity = _response_wire_frame_identity(projected)
            if source_identity is None or indexed_identity != source_identity:
                indexed_projection_rejected = {
                    'reason': 'source_frame_identity_missing'
                    if source_identity is None
                    else 'source_frame_identity_mismatch',
                    'source_frame_id': source_identity[0] if source_identity else None,
                    'source_frame_sequence': source_identity[1] if source_identity else None,
                    'indexed_frame_id': indexed_identity[0] if indexed_identity else None,
                    'indexed_frame_sequence': indexed_identity[1] if indexed_identity else None,
                }
                projected = None
    if projected is None:
        projected = _response_wire_fallback_payload(response_payload)
        if indexed_projection_rejected:
            projection = (
                dict(projected.get('wire_projection'))
                if isinstance(projected.get('wire_projection'), Mapping)
                else {}
            )
            projection['indexed_projection_rejected'] = {
                key: value
                for key, value in indexed_projection_rejected.items()
                if value is not None
            }
            projection['durability_match'] = False
            projected['wire_projection'] = projection
    else:
        for key in _RESPONSE_WIRE_PUBLIC_KEYS:
            if key in projected:
                continue
            value = response_payload.get(key)
            if value not in (None, '', [], {}) and key not in {'surface_state'}:
                projected[key] = copy.deepcopy(value)
    projected = _attach_response_status_semantics(projected)
    record = _response_lookup_record_from_wire_payload(
        response_id or str(projected.get('id') or ''),
        projected,
        lookup_source='response_wire_post',
    )
    projected = _attach_response_lookup_state_version(projected, status_record=record)
    return _response_wire_enforce_byte_ceiling(
        projected,
        source_payload=response_payload,
        source='post_projection_emergency_byte_ceiling',
    )


def _build_bounded_response_debug_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    response_id = str(record.get('id') or '').strip()
    projected, _state = _response_wire_payload_from_index(
        response_id,
        include_observation=True,
    ) if response_id else (None, {})
    if projected is None:
        source_payload = record.get('response_payload') if isinstance(record.get('response_payload'), Mapping) else {}
        projected = _response_wire_fallback_payload(source_payload)
    projected = _attach_response_status_semantics(projected)
    projected = _attach_response_artifact_bundles_from_registry(projected)
    status_record = _response_lookup_record_from_wire_payload(
        response_id or str(projected.get('id') or ''),
        projected,
        lookup_source='response_wire_debug',
    )
    projected = _attach_response_lookup_state_version(projected, status_record=status_record)
    source_payload = (
        record.get('response_payload')
        if isinstance(record.get('response_payload'), Mapping)
        else projected
    )
    return _response_wire_enforce_byte_ceiling(
        projected,
        source_payload=source_payload,
        source='debug_projection_emergency_byte_ceiling',
    )


def _get_bounded_response_lookup_record(
    response_id: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], int]:
    return _RESPONSE_LOOKUP_RUNTIME.get_bounded_response_lookup_record(response_id)


def _register_response_stream(response_id: str) -> dict[str, Any]:
    return _RESPONSES_RUNTIME.register_response_stream(response_id)


def _append_response_stream_events(response_id: str, events: list[str], *, done: bool = False) -> None:
    _RESPONSES_RUNTIME.append_response_stream_events(
        response_id,
        _compact_response_stream_events(response_id, events),
        done=done,
    )


def _wait_for_response_stream_events(response_id: str, cursor: int, timeout_sec: float = 0.5) -> tuple[list[str], bool]:
    return _RESPONSES_RUNTIME.wait_for_response_stream_events(response_id, cursor, timeout_sec=timeout_sec)


def _close_response_stream(response_id: str) -> None:
    _RESPONSES_RUNTIME.close_response_stream(response_id)


def _response_payload_message_id(response_payload: Optional[dict[str, Any]]) -> Optional[str]:
    payload = response_payload if isinstance(response_payload, dict) else {}
    output = payload.get('output') if isinstance(payload.get('output'), list) else []
    first_item = output[0] if output and isinstance(output[0], dict) else {}
    message_id = str(first_item.get('id') or '').strip()
    return message_id or None


def _ensure_response_lookup_for_payload(
    response_payload: Optional[dict[str, Any]],
    *,
    mode_hint: str,
    route_payload: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    payload = response_payload if isinstance(response_payload, dict) else {}
    response_id = str(payload.get('id') or payload.get('response_id') or '').strip()
    if not response_id:
        return None
    # This helper runs on the live POST completion path.  A missing in-memory
    # record must be registered from the just-finalized payload, not recovered
    # by hydrating canonical sidecars that were written milliseconds earlier.
    existing = _RESPONSES_RUNTIME.get_response_lookup_record(response_id)
    if existing:
        return existing
    return _register_response_lookup(
        response_id=response_id,
        message_id=_response_payload_message_id(payload) or '',
        instance_id=str(payload.get('instance_id') or '').strip(),
        model_name=str(payload.get('model') or '').strip(),
        backend=str(payload.get('backend') or '').strip(),
        capability=str(payload.get('capability') or '').strip(),
        mode=str(mode_hint or payload.get('mode') or 'chat').strip() or 'chat',
        route_payload=route_payload if isinstance(route_payload, dict) else None,
    )


def _resolve_generated_image_state_cache_key(raw_path: Any) -> Optional[str]:
    token = str(raw_path or '').strip()
    if not token:
        return None
    try:
        resolved = _resolve_saved_downloadable_artifact_path(token)
    except Exception:  # noqa: BLE001
        resolved = None
    if resolved is not None:
        return str(resolved)
    try:
        candidate = Path(token).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return None
    return str(candidate) if candidate.exists() else None


def _get_cached_generated_image_state(raw_path: Any) -> Optional[dict[str, Any]]:
    cache_key = _resolve_generated_image_state_cache_key(raw_path)
    if not cache_key:
        return None
    with _GENERATED_IMAGE_STATE_CACHE_LOCK:
        payload = _GENERATED_IMAGE_STATE_CACHE.get(cache_key)
        return dict(payload) if isinstance(payload, dict) and payload else None


def _store_cached_generated_image_state(raw_path: Any, image_state: Any) -> Optional[dict[str, Any]]:
    cache_key = _resolve_generated_image_state_cache_key(raw_path)
    if not cache_key or not isinstance(image_state, dict) or not image_state:
        return None
    payload = dict(image_state)
    with _GENERATED_IMAGE_STATE_CACHE_LOCK:
        _GENERATED_IMAGE_STATE_CACHE[cache_key] = payload
    return payload


def _build_image_state_enrichment_state(
    *,
    status: str,
    mode: str = 'background_analysis',
    reason: str = '',
) -> dict[str, Any]:
    payload = {
        'status': str(status or '').strip().lower() or 'pending',
        'mode': str(mode or '').strip() or 'background_analysis',
        'updated_at': _response_registry_now_iso(),
    }
    reason_text = str(reason or '').strip()
    if reason_text:
        payload['reason'] = reason_text
    return payload


def _attach_cached_generated_image_state_to_response_lookups(raw_path: Any, image_state: Any) -> None:
    cache_key = _resolve_generated_image_state_cache_key(raw_path)
    if not cache_key or not isinstance(image_state, dict) or not image_state:
        return
    now_ts = time.time()
    with _RESPONSE_LOOKUP_LOCK:
        _prune_response_lookup_registry(now_ts)
        for record in _RESPONSE_LOOKUP.values():
            response_payload = record.get('response_payload')
            if not isinstance(response_payload, dict):
                continue
            payload_cache_key = _resolve_generated_image_state_cache_key(
                response_payload.get('saved_image_path')
            )
            if payload_cache_key != cache_key:
                continue
            existing_state = response_payload.get('image_state')
            if isinstance(existing_state, dict) and existing_state:
                continue
            updated_payload = dict(response_payload)
            updated_payload['image_state'] = dict(image_state)
            updated_payload['image_state_enrichment'] = _build_image_state_enrichment_state(status='completed')
            record['response_payload'] = updated_payload
            record['updated_at'] = _response_registry_now_iso()
            record['expires_at_ts'] = now_ts + RESPONSE_LOOKUP_TTL_SEC


def _claim_generated_image_state_enrichment(raw_path: Any) -> Optional[str]:
    cache_key = _resolve_generated_image_state_cache_key(raw_path)
    if not cache_key:
        return None
    with _GENERATED_IMAGE_STATE_ENRICHMENT_LOCK:
        if cache_key in _GENERATED_IMAGE_STATE_ENRICHMENT_IN_FLIGHT:
            return None
        _GENERATED_IMAGE_STATE_ENRICHMENT_IN_FLIGHT.add(cache_key)
    return cache_key


def _release_generated_image_state_enrichment(cache_key: Any) -> None:
    token = str(cache_key or '').strip()
    if not token:
        return
    with _GENERATED_IMAGE_STATE_ENRICHMENT_LOCK:
        _GENERATED_IMAGE_STATE_ENRICHMENT_IN_FLIGHT.discard(token)


def _build_runtime_status_stub(
    *,
    instance_id: Optional[str],
    model: Optional[str],
    backend: Optional[str],
    capability: Optional[str],
    port: Optional[int],
    pid: Optional[int] = None,
) -> Optional[dict]:
    token = str(instance_id or '').strip()
    if not token:
        return None
    payload = {
        'instance_id': token,
        'model': model,
        'backend': backend,
        'capability': capability,
        'port': port,
        'pid': pid,
    }
    return {key: value for key, value in payload.items() if value is not None and value != ''}


def _select_backend_request_model(
    instance: Optional[dict],
    requested_model: Any,
    fallback_model_name: Any,
) -> str:
    requested = str(requested_model or '').strip()
    fallback = str(fallback_model_name or '').strip()
    if not instance:
        return requested or fallback

    backend = normalize_backend(instance.get('backend'))
    backend_package = str(instance.get('backend_package') or '').strip().lower()
    backend_contract = str(instance.get('backend_contract') or '').strip().lower()
    capability = normalize_capability(instance.get('capability'))
    resolved_model_name = str(instance.get('model') or fallback or '').strip().lower()
    looks_like_mlx_audio_contract = (
        backend_package == 'mlx_audio'
        or backend_contract == 'mlx_audio.server'
        or capability == CAPABILITY_TEXT_TO_SPEECH
        or (capability == CAPABILITY_SPEECH_TO_TEXT and 'whisper' not in resolved_model_name)
    )
    if backend == 'mlx' and looks_like_mlx_audio_contract:
        if requested and not os.path.isabs(requested):
            return requested
        return fallback or requested
    return requested or fallback


def _log_runtime_status_transition(previous: Optional[dict], current: Optional[dict]) -> None:
    if app.config.get("TESTING"):
        return
    previous_readiness = str((previous or {}).get('readiness') or '').strip().lower()
    current_readiness = str((current or {}).get('readiness') or '').strip().lower()
    if not current_readiness or current_readiness == previous_readiness:
        return

    event_status = 'ok'
    event_severity = None
    runtime_truth_note = None
    if current_readiness == 'degraded':
        event_status = 'warning'
        event_severity = 'advisory'
        runtime_truth_note = 'degraded_readiness_is_advisory_until_live_truth_fails'
    elif current_readiness == 'unreachable':
        event_status = 'failed'
    elif current_readiness in {'stopped'}:
        event_status = 'completed'

    message = f"Runtime status changed: {previous_readiness or 'unknown'} -> {current_readiness}"
    _log_unified_event(
        category="runtime",
        action="status_transition",
        status=event_status,
        instance_id=(current or previous or {}).get('instance_id'),
        model=(current or previous or {}).get('model'),
        backend=(current or previous or {}).get('backend'),
        capability=(current or previous or {}).get('capability'),
        port=(current or previous or {}).get('port'),
        previous_readiness=previous_readiness or None,
        readiness=current_readiness,
        severity=event_severity,
        runtime_truth_note=runtime_truth_note,
        message=message,
    )


_WRAPPER_CAPABILITY_ALIASES: dict[str, list[str]] = {
    'chat': ['chat'],
    CAPABILITY_IMAGE_GENERATION: ['image', 'image_generation'],
    CAPABILITY_VISION_ANALYSIS: ['ocr', 'vision', 'vision_analysis'],
    CAPABILITY_SPEECH_TO_TEXT: ['stt', 'speech_to_text', 'transcription'],
    CAPABILITY_TEXT_TO_SPEECH: ['tts', 'text_to_speech'],
}

_SESSION_CONTROL_REQUEST_KEYS: dict[str, str] = {
    'temperature': 'temperature',
    'top_p': 'top_p',
    'stt_language': 'language',
    'stt_task': 'task',
    'image_count': 'image_count',
    'ocr_mode': 'ocr_mode',
    'pdf_max_pages': 'pdf_max_pages',
    'pdf_dpi': 'pdf_dpi',
    'pdf_page_timeout_sec': 'pdf_page_timeout_sec',
    'pdf_synthesize': 'pdf_synthesize',
    'tts_voice': 'voice',
    'tts_language': 'lang_code',
    'tts_instruct': 'instruct',
    'tts_speed': 'speed',
    'tts_pitch': 'pitch',
}


def _request_base_url() -> str:
    if has_request_context():
        return str(request.host_url or '').rstrip('/')
    return f'http://127.0.0.1:{APP_PORT}'


def _build_instance_responses_path(instance_id: str) -> str:
    return f"/api/local_provider/{quote(str(instance_id or ''), safe='')}/v1/responses"


def _pick_default_capability_instance(instances: list[dict]) -> Optional[str]:
    if not instances:
        return None
    selectable = [
        item
        for item in instances
        if isinstance(item, dict)
        and runtime_instance_is_selectable(
            item,
            capability=normalize_capability(item.get('capability')),
        )
    ]
    if not selectable:
        return None
    selected = sorted(
        selectable,
        key=lambda item: runtime_instance_score(
            item,
            capability=normalize_capability(item.get('capability')),
        ),
        reverse=True,
    )[0]
    return str(selected.get('instance_id') or '').strip() or None


def _model_preference_tokens(value: Any) -> list[str]:
    tokens = []
    for part in re.split(r'[^a-z0-9]+', str(value or '').lower()):
        token = part.strip()
        if not token or len(token) < 3:
            continue
        if token in {
            'mlx', 'community', 'latest', 'bf16', 'fp16', '4bit', '8bit', 'mini', 'base',
            'model', 'audio', 'speech', 'text', 'instruct',
        }:
            continue
        tokens.append(token)
        alpha_only = re.sub(r'[^a-z]+', '', token)
        if len(alpha_only) >= 3 and alpha_only != token and alpha_only not in tokens:
            tokens.append(alpha_only)
    return tokens


def _pick_prompt_preferred_instance(instances: list[dict], prompt: str) -> Optional[str]:
    lowered_prompt = str(prompt or '').lower()
    if not lowered_prompt.strip():
        return None

    scored: list[tuple[Any, ...]] = []
    for item in instances:
        if not isinstance(item, dict):
            continue
        if not runtime_instance_is_selectable(
            item,
            capability=normalize_capability(item.get('capability')),
        ):
            continue
        model_tokens = set(_model_preference_tokens(item.get('model')))
        instance_tokens = set(_model_preference_tokens(item.get('instance_id')))
        candidate_tokens = model_tokens | instance_tokens
        if not candidate_tokens:
            continue
        matched = {token for token in candidate_tokens if token in lowered_prompt}
        if not matched:
            continue
        liveness_score = runtime_instance_score(
            item,
            capability=normalize_capability(item.get('capability')),
        )
        session_controls = item.get('session_controls') if isinstance(item.get('session_controls'), dict) else {}
        fields = session_controls.get('fields') if isinstance(session_controls.get('fields'), dict) else {}
        required_count = sum(
            1 for field in fields.values()
            if isinstance(field, dict) and field.get('visible') is not False and field.get('required')
        )
        model_type = str(item.get('tts_model_type') or '').strip().lower()
        simplicity_rank = 2
        if model_type in {'custom_voice', 'kitten_tts'}:
            simplicity_rank = 1
        elif model_type == 'voice_design':
            simplicity_rank = 0
        scored.append((
            len(matched),
            -required_count,
            simplicity_rank,
            *liveness_score[:-1],
            str(item.get('instance_id') or ''),
        ))

    if not scored:
        return None
    best_instance_id = sorted(scored, reverse=True)[0][-1]
    return best_instance_id or None


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _build_instance_trait_summary(instance: dict[str, Any]) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.build_instance_trait_summary(instance)


def _estimate_route_context_tokens(*, prompt: str = '', messages: Optional[list[dict[str, Any]]] = None) -> int:
    total_chars = len(str(prompt or '').strip())
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        total_chars += len(str(message.get('content') or ''))
    return max(1, (total_chars + 3) // 4)


def _normalize_request_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        payload = dict(data)
    else:
        try:
            payload = dict(data or {})
        except (TypeError, ValueError):
            payload = {}
    return attach_request_meta(payload)


def _timeout_ms_to_seconds(timeout_ms: Any) -> Optional[int]:
    try:
        value = int(timeout_ms or 0)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return max(1, (value + 999) // 1000)


def _planner_timeout_seconds_for_payload(
    payload: Any,
    *,
    semantic_role_profile: Optional[dict[str, Any]] = None,
) -> Optional[int]:
    return _GHOST_ROUTE_RUNTIME._planner_timeout_seconds_for_payload(
        payload,
        semantic_role_profile=semantic_role_profile,
    )


def _effective_request_meta_payload(payload: Any) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME._effective_request_meta_payload(payload)


def _build_developer_diagnostics_payload(
    payload: Any,
    *,
    planner_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME._build_developer_diagnostics_payload(
        payload,
        planner_meta=planner_meta,
    )


def _merge_request_meta_runtime_truth(
    route_runtime: Optional[dict[str, Any]],
    payload: Any,
    *,
    planner_meta: Optional[dict[str, Any]] = None,
    route_payload: Optional[dict[str, Any]] = None,
    response_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.merge_request_meta_runtime_truth(
        route_runtime,
        payload,
        planner_meta=planner_meta,
        route_payload=route_payload,
        response_payload=response_payload,
    )

def _prompt_requests_structured_or_tooling(prompt: str) -> bool:
    return _GHOST_ROUTE_RUNTIME._prompt_requests_structured_or_tooling(prompt)


def _route_context_prefers_multimodal_chat(route_context: dict[str, Any]) -> bool:
    return _GHOST_ROUTE_RUNTIME._route_context_prefers_multimodal_chat(route_context)


def _route_context_prefers_long_context(route_context: dict[str, Any]) -> bool:
    return _GHOST_ROUTE_RUNTIME._route_context_prefers_long_context(route_context)


def _route_context_prefers_richer_tts(route_context: dict[str, Any]) -> bool:
    return _GHOST_ROUTE_RUNTIME._route_context_prefers_richer_tts(route_context)


def _tts_language_supported(instance: dict[str, Any], requested_codes: list[str]) -> bool:
    return _GHOST_ROUTE_RUNTIME._tts_language_supported(instance, requested_codes)


def _pick_tts_trait_aware_instance(instances: list[dict[str, Any]], route_context: dict[str, Any]) -> Optional[str]:
    return _GHOST_ROUTE_RUNTIME._pick_tts_trait_aware_instance(instances, route_context)


def _pick_trait_aware_instance(instances: list[dict[str, Any]], route_context: dict[str, Any]) -> Optional[str]:
    return _GHOST_ROUTE_RUNTIME.pick_trait_aware_instance(instances, route_context)


def _build_route_trait_reasons(instance: dict[str, Any], route_context: dict[str, Any]) -> list[str]:
    return _GHOST_ROUTE_RUNTIME._build_route_trait_reasons(instance, route_context)


def _augment_route_reason(base_reason: str, instance: dict[str, Any], route_context: dict[str, Any]) -> str:
    return _GHOST_ROUTE_RUNTIME.augment_route_reason(base_reason, instance, route_context)


def _build_compressed_history_message(messages: list[dict[str, Any]], *, keep_last: int = 4) -> Optional[dict[str, str]]:
    return _GHOST_ROUTE_RUNTIME._build_compressed_history_message(messages, keep_last=keep_last)


def _choose_context_strategy(
    *,
    instance: Optional[dict[str, Any]],
    messages: list[dict[str, Any]],
    prompt: str,
    has_file_context: bool,
    conversation_id: Any = None,
) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.choose_context_strategy(
        instance=instance,
        messages=messages,
        prompt=prompt,
        has_file_context=has_file_context,
        conversation_id=conversation_id,
        history_dir=CHAT_HISTORY_DIR,
        artifact_registry_ledger=ARTIFACT_REGISTRY_LEDGER,
    )


def _apply_context_strategy(messages: list[dict[str, Any]], strategy: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    return _GHOST_ROUTE_RUNTIME.apply_context_strategy(messages, strategy)


def _normalize_chat_content_for_backend(content: Any, backend: Optional[str]) -> Any:
    return _GHOST_ROUTE_RUNTIME._normalize_chat_content_for_backend(content, backend)


def _normalize_chat_messages_for_backend(
    messages: list[dict[str, Any]],
    *,
    backend: Optional[str] = None,
) -> list[dict[str, Any]]:
    return _GHOST_ROUTE_RUNTIME.normalize_chat_messages_for_backend(
        messages,
        backend=backend,
    )


def _compact_string_list(value: Any, *, limit: int = 8) -> list[str]:
    return _GHOST_ROUTE_RUNTIME._compact_string_list(value, limit=limit)


def _compact_dynamic_trait_value(value: Any) -> Optional[Any]:
    return _GHOST_ROUTE_RUNTIME._compact_dynamic_trait_value(value)


def _summarize_dynamic_model_traits_for_routing(entry: Any) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME._summarize_dynamic_model_traits_for_routing(entry)


def _summarize_session_controls_for_routing(schema: Any) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.summarize_session_controls_for_routing(schema)


def _summarize_backend_metadata_for_routing(metadata: Any) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.summarize_backend_metadata_for_routing(metadata)


def _summarize_backend_runtime_for_routing(runtime: Any) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.summarize_backend_runtime_for_routing(runtime)


def _build_instance_routing_summary(entry: dict[str, Any], runtime_status: dict[str, Any]) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.build_instance_routing_summary(entry, runtime_status)


def _build_routing_manifest_payload(
    instances: list[dict],
    *,
    backend_fabric: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.build_routing_manifest_payload(
        instances,
        backend_fabric=backend_fabric,
    )


def _resolve_wrapper_capability(selector: str) -> Optional[str]:
    return _REQUEST_INTAKE_RUNTIME._resolve_wrapper_capability(selector)


def _instance_effective_capability(instance: dict) -> str:
    model_name = str(instance.get('model') or instance.get('modelName') or '').strip()
    backend = normalize_backend(instance.get('backend'))
    return normalize_capability(instance.get('capability')) or infer_capability(model_name, backend)


def _instance_supports_capability(instance: dict, capability: str) -> bool:
    model_name = str(instance.get('model') or instance.get('modelName') or '').strip()
    backend = normalize_backend(instance.get('backend'))
    return supports_capability(
        capability,
        model_name=model_name,
        backend=backend,
        capability=instance.get('capability'),
        metadata=instance,
    )


def _resolve_responses_target_instance(
    data: Any,
    *,
    forced_instance_id: Optional[str] = None,
    excluded_instance_ids: Optional[list[str]] = None,
) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[str]]:
    explicit_instance_id = str(
        forced_instance_id
        or (data.get('instance_id') if hasattr(data, 'get') else '')
        or ''
    ).strip()
    if explicit_instance_id == CODEX_TARGET_ID:
        target = _codex_external_target_payload()
        if target.get('enabled') is not True:
            return (
                None,
                None,
                'chat',
                'ChatGPT is disabled. Enable it explicitly in Ollmo Preferences before sending text to OpenAI.',
            )
        if target.get('status') != 'available':
            return (
                None,
                None,
                'chat',
                str(target.get('recovery_hint') or 'ChatGPT is not available.'),
            )
        valid, _error_code, validation_error = validate_codex_text_request(
            data,
            files_enabled=target.get('files_enabled') is True,
        )
        if not valid:
            return None, None, 'chat', validation_error
        return CODEX_TARGET_ID, target, 'chat', None
    return _REQUEST_INTAKE_RUNTIME._resolve_responses_target_instance(
        data,
        forced_instance_id=forced_instance_id,
        excluded_instance_ids=excluded_instance_ids,
    )


def _recover_missing_explicit_target_instance(
    explicit_instance_id: str,
    data: Any,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    return _REQUEST_INTAKE_RUNTIME._recover_missing_explicit_target_instance(
        explicit_instance_id,
        data,
    )


def _parse_jsonish_field(raw_value: Any) -> Any:
    return _REQUEST_INTAKE_RUNTIME._parse_jsonish_field(raw_value)


def _build_preview_instance_payload(instance: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return _RESPONSES_REQUEST_RUNTIME.build_preview_instance_payload(instance)


def _build_ghost_route_preview_payload(
    route_info: dict[str, Any],
    request_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _RESPONSES_REQUEST_RUNTIME.build_ghost_route_preview_payload(
        route_info,
        request_payload=request_payload,
    )


def _observer_refresh_requested(payload: Optional[Mapping[str, Any]] = None) -> bool:
    if has_request_context() and _parse_bool(request.args.get('refresh'), default=False):
        return True
    if isinstance(payload, Mapping):
        for key in ('refresh', 'refresh_runtime_status', 'refreshRuntimeStatus'):
            if key in payload and _parse_bool(payload.get(key), default=False):
                return True
    return False


_GHOST_PREVIEW_COMPUTE_SEMANTICS_ENV = 'OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS'
_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE_ENV = (
    'OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE'
)
_GHOST_PREVIEW_COMPUTE_SEMANTICS_KEYS = (
    'compute_semantics',
    'computeSemantics',
    'semantic_compute',
    'semanticCompute',
)


def _normalize_ghost_preview_compute_semantics_policy(value: Any) -> str:
    token = str(value or '').strip().lower()
    if token in {'0', 'false', 'no', 'n', 'off'}:
        return 'off'
    if token == 'auto':
        return 'auto'
    if token in {'1', 'true', 'yes', 'y', 'on'}:
        return 'on'
    return 'off'


def _normalize_ghost_preview_compute_semantics_false_override(value: Any) -> str:
    token = str(value or '').strip().lower()
    if token in {'1', 'true', 'yes', 'y', 'on', 'allow', 'allowed'}:
        return 'allow'
    if token in {'0', 'false', 'no', 'n', 'off', 'deny', 'denied'}:
        return 'deny'
    return 'allow'


def _explicit_observer_semantic_compute(payload: Optional[Mapping[str, Any]] = None) -> Optional[bool]:
    if has_request_context():
        for key in _GHOST_PREVIEW_COMPUTE_SEMANTICS_KEYS:
            if key in request.args:
                return _parse_bool(request.args.get(key), default=False)
    if isinstance(payload, Mapping):
        for key in _GHOST_PREVIEW_COMPUTE_SEMANTICS_KEYS:
            if key in payload:
                return _parse_bool(payload.get(key), default=False)
    return None


def _observer_semantic_compute_policy(payload: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    policy = _normalize_ghost_preview_compute_semantics_policy(
        os.environ.get(_GHOST_PREVIEW_COMPUTE_SEMANTICS_ENV)
    )
    false_override = _normalize_ghost_preview_compute_semantics_false_override(
        os.environ.get(_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE_ENV)
    )
    policy_enabled = policy in {'on', 'auto'}
    explicit_value = _explicit_observer_semantic_compute(payload)
    if explicit_value is True:
        return {
            'enabled': True,
            'source': 'explicit_request',
            'policy': policy,
            'false_override': false_override,
        }
    if explicit_value is False:
        if policy_enabled and false_override != 'allow':
            return {
                'enabled': True,
                'source': 'policy_default',
                'policy': policy,
                'false_override': false_override,
            }
        return {
            'enabled': False,
            'source': 'explicit_request',
            'policy': policy,
            'false_override': false_override,
        }
    return {
        'enabled': policy_enabled,
        'source': 'policy_default',
        'policy': policy,
        'false_override': false_override,
    }


def _runtime_truth_metadata(
    *,
    refresh_requested: bool,
    semantic_compute_requested: Optional[bool] = None,
    semantic_compute_performed: Optional[bool] = None,
    compute_semantics_source: Optional[str] = None,
    compute_semantics_policy: Optional[str] = None,
    compute_semantics_false_override: Optional[str] = None,
) -> dict[str, Any]:
    computed = bool(semantic_compute_performed)
    metadata = {
        'truth_mode': 'refreshed' if refresh_requested else 'cached',
        'refresh_performed': bool(refresh_requested),
        'runtime_status_source': str(RUNTIME_STATUS_PATH),
    }
    if computed:
        metadata['truth_mode'] = 'computed'
    if semantic_compute_requested is not None:
        metadata['semantic_compute_requested'] = bool(semantic_compute_requested)
        metadata['semantic_compute_performed'] = computed
    if compute_semantics_source is not None:
        metadata['compute_semantics_source'] = str(compute_semantics_source or '')
    if compute_semantics_policy is not None:
        metadata['compute_semantics_policy'] = str(compute_semantics_policy or '')
    if compute_semantics_false_override is not None:
        metadata['compute_semantics_false_override'] = str(compute_semantics_false_override or '')
    return metadata


def _attach_runtime_truth_headers(response: Any, metadata: Mapping[str, Any]) -> Any:
    response.headers['X-Ollmo-Truth-Mode'] = str(metadata.get('truth_mode') or '')
    response.headers['X-Ollmo-Refresh-Performed'] = 'true' if metadata.get('refresh_performed') else 'false'
    response.headers['X-Ollmo-Runtime-Status-Source'] = str(metadata.get('runtime_status_source') or '')
    return response


_INFER_RUNTIME = InferRuntimeOwner(
    hooks={
        'rewind_upload_stream': lambda upload: _rewind_upload_stream(upload),
        'normalize_external_identifier': lambda value, **kwargs: _normalize_external_identifier(value, **kwargs),
        'lookup_instance': lambda instance_id: _lookup_instance(instance_id),
        'normalize_backend': lambda value: normalize_backend(value),
        'select_backend_request_model': lambda *args, **kwargs: _select_backend_request_model(*args, **kwargs),
        'normalize_capability': lambda value: normalize_capability(value),
        'infer_capability': lambda *args, **kwargs: infer_capability(*args, **kwargs),
        'normalize_request_payload': lambda payload: _normalize_request_payload(payload),
        'extract_responses_prompt': lambda payload: _extract_responses_prompt(payload),
        'analyze_prompt_intent': lambda prompt: analyze_prompt_intent(prompt),
        'extract_responses_current_turn_prompt': lambda payload: _extract_responses_current_turn_prompt(payload),
        'plan_compound_execution_payload': lambda *args, **kwargs: _plan_compound_execution_payload(*args, **kwargs),
        'apply_prompt_control_hints': lambda *args, **kwargs: apply_prompt_control_hints(*args, **kwargs),
        'extract_ghost_route_messages': lambda *args, **kwargs: _extract_ghost_route_messages(*args, **kwargs),
        'extract_semantic_materializer_prompt': lambda *args, **kwargs: _extract_semantic_materializer_prompt(*args, **kwargs),
        'merge_request_meta_runtime_truth': lambda *args, **kwargs: _merge_request_meta_runtime_truth(*args, **kwargs),
        'build_working_frame': lambda *args, **kwargs: _build_working_frame(*args, **kwargs),
        'session_control_request_keys': _SESSION_CONTROL_REQUEST_KEYS,
        'extract_selected_reference_artifacts': lambda payload: _extract_selected_reference_artifacts(payload),
        'select_matching_selected_reference_artifact': lambda *args, **kwargs: _select_matching_selected_reference_artifact(*args, **kwargs),
        'should_attach_selected_reference_file_context': lambda *args, **kwargs: _should_attach_selected_reference_file_context(*args, **kwargs),
        'apply_selected_reference_prompt_prefix': lambda *args, **kwargs: _apply_selected_reference_prompt_prefix(*args, **kwargs),
        'translate_responses_payload_to_infer_payload': lambda payload: _translate_responses_payload_to_infer_payload(payload),
        'coerce_seed': lambda value: _coerce_seed(value),
        'find_image_artifact_seed': lambda *args, **kwargs: _find_image_artifact_seed(*args, **kwargs),
        'choose_context_strategy': lambda *args, **kwargs: _choose_context_strategy(*args, **kwargs),
        'parse_float_with_bounds': lambda *args, **kwargs: _parse_float_with_bounds(*args, **kwargs),
        'parse_int_with_bounds': lambda *args, **kwargs: _parse_int_with_bounds(*args, **kwargs),
        'parse_bool': lambda value, **kwargs: _parse_bool(value, **kwargs),
        'build_runtime_status_stub': lambda *args, **kwargs: _build_runtime_status_stub(*args, **kwargs),
        'build_infer_dedupe_key': lambda *args, **kwargs: _build_infer_dedupe_key(*args, **kwargs),
        'acquire_infer_slot': lambda slot_key: _acquire_infer_slot(slot_key),
        'release_infer_slot': lambda slot_key: _release_infer_slot(slot_key),
        'log_unified_event': lambda **kwargs: _log_unified_event(**kwargs),
        'file_kind_from_name': lambda name: _file_kind_from_name(name),
        'save_upload_to_temp': lambda upload: _save_upload_to_temp(upload),
        'save_local_path_to_temp': lambda path: _save_local_path_to_temp(path),
        'persist_request_input_artifacts': lambda *args, **kwargs: _persist_request_input_artifacts(*args, **kwargs),
        'persist_input_artifact_registry_records': lambda *args, **kwargs: _persist_input_artifact_registry_records(
            *args,
            ledger_path=ARTIFACT_REGISTRY_LEDGER,
            **kwargs,
        ),
        'find_artifact_registry_record': lambda artifact_path: _find_artifact_registry_record(
            artifact_path,
            ledger_path=ARTIFACT_REGISTRY_LEDGER,
        ),
        'to_base64': lambda path: _to_base64(path),
        'read_text_file': lambda path: _read_text_file(path),
        'hash_file_sha256': lambda path: _hash_file_sha256(path),
        'find_cached_pdf_insight': lambda *args, **kwargs: _find_cached_pdf_insight(*args, **kwargs),
        'extract_pdf_text_content': lambda path: _extract_pdf_text_content(path),
        'render_pdf_pages_to_base64': lambda *args, **kwargs: _render_pdf_pages_to_base64(*args, **kwargs),
        'log_pdf_infer_event': lambda **kwargs: _log_pdf_infer_event(**kwargs),
        'record_instance_activity': lambda *args, **kwargs: record_instance_activity(*args, **kwargs),
        'record_instance_success': lambda *args, **kwargs: record_instance_success(*args, **kwargs),
        'record_instance_failure': lambda *args, **kwargs: record_instance_failure(*args, **kwargs),
        'log_runtime_status_transition': lambda *args, **kwargs: _log_runtime_status_transition(*args, **kwargs),
        'InferContext': InferContext,
        'InferArtifacts': InferArtifacts,
        'dispatch_infer_request': lambda *args, **kwargs: dispatch_infer_request(*args, **kwargs),
        'whisper_transcribe': lambda *args, **kwargs: _whisper_transcribe(*args, **kwargs),
        'mlx_audio_speech': lambda *args, **kwargs: _mlx_audio_speech(*args, **kwargs),
        'ollama_generate': lambda *args, **kwargs: _ollama_generate(*args, **kwargs),
        'extract_saved_image_path_from_generate_output': lambda *args, **kwargs: _extract_saved_image_path_from_generate_output(*args, **kwargs),
        'extract_image_data_url_from_generate_output': lambda *args, **kwargs: _extract_image_data_url_from_generate_output(*args, **kwargs),
        'extract_generate_seed': lambda *args, **kwargs: _extract_generate_seed(*args, **kwargs),
        'ollama_openai_image_generation': lambda *args, **kwargs: _ollama_openai_image_generation(*args, **kwargs),
        'persist_audio_bytes_locally': lambda *args, **kwargs: _persist_audio_bytes_locally(*args, **kwargs),
        'persist_image_data_url_locally': lambda *args, **kwargs: _persist_image_data_url_locally(*args, **kwargs),
        'extract_generate_content': lambda *args, **kwargs: _extract_generate_content(*args, **kwargs),
        'persist_text_artifact_locally': lambda *args, **kwargs: _persist_text_artifact_locally(*args, **kwargs),
        'persist_text_markdown_locally': lambda *args, **kwargs: _persist_text_markdown_locally(*args, **kwargs),
        'persist_transcript_text_locally': lambda *args, **kwargs: _persist_transcript_text_locally(*args, **kwargs),
        'ocr_pdf_page_with_ollama': lambda *args, **kwargs: _ocr_pdf_page_with_ollama(*args, **kwargs),
        'render_single_pdf_page_to_base64': lambda *args, **kwargs: _render_single_pdf_page_to_base64(*args, **kwargs),
        'ocr_image_with_deepseek': lambda *args, **kwargs: _ocr_image_with_deepseek(*args, **kwargs),
        'is_generic_ocr_instruction_prompt': lambda *args, **kwargs: _is_generic_ocr_instruction_prompt(*args, **kwargs),
        'clean_ocr_output_text': lambda *args, **kwargs: _clean_ocr_output_text(*args, **kwargs),
        'looks_like_ocr_prompt_echo': lambda *args, **kwargs: _looks_like_ocr_prompt_echo(*args, **kwargs),
        'ollama_chat': lambda *args, **kwargs: _ollama_chat(*args, **kwargs),
        'openai_chat_completions': lambda *args, **kwargs: _openai_chat_completions(*args, **kwargs),
        'mlx_chat_completions': lambda *args, **kwargs: _openai_chat_completions('mlx', *args, **kwargs),
        'enrich_generated_image_payload': lambda payload: _enrich_generated_image_payload(payload),
        'persist_generated_image_provenance_for_infer_result': lambda *args, **kwargs: _persist_generated_image_provenance_for_infer_result(*args, **kwargs),
        'sanitize_artifact_records': lambda value: _sanitize_artifact_records(value),
        'extract_artifact_ref': lambda value: _extract_artifact_ref(value),
        'merge_unique_artifact_records': lambda existing, incoming: _merge_unique_artifact_records(existing, incoming),
        'max_pdf_inline_response_chars': lambda: MAX_PDF_INLINE_RESPONSE_CHARS,
    },
    capability_embedding=CAPABILITY_EMBEDDING,
    capability_image_generation=CAPABILITY_IMAGE_GENERATION,
    capability_vision_analysis=CAPABILITY_VISION_ANALYSIS,
    runtime_status_path_getter=lambda: RUNTIME_STATUS_PATH,
    request_timeout_error=REQUEST_TIMEOUT_ERROR,
    request_connection_error=REQUEST_CONNECTION_ERROR,
    request_exception_error=REQUEST_EXCEPTION_ERROR,
)


def _prepare_effective_request_data(
    data: Any,
    *,
    route_info: Optional[dict[str, Any]] = None,
    instance: Optional[dict[str, Any]] = None,
    compute_semantics: bool = True,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return _INFER_RUNTIME.prepare_effective_request_data(
        data,
        route_info=route_info,
        instance=instance,
        compute_semantics=compute_semantics,
    )


def _apply_required_session_control_defaults(
    data: dict[str, Any],
    *,
    instance: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _INFER_RUNTIME.apply_required_session_control_defaults(
        data,
        instance=instance,
    )


def _build_missing_required_session_controls(instance: dict[str, Any], data: Any) -> list[dict[str, Any]]:
    return _INFER_RUNTIME.build_missing_required_session_controls(instance, data)


def _build_responses_infer_execution_payload(
    data: Any,
    *,
    route_info: Optional[dict[str, Any]],
    instance: dict[str, Any],
    instance_id: str,
    backend: str,
    capability: str,
    request_model_override: Optional[str],
    upload_present: bool = False,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], bool, bool]:
    return _INFER_RUNTIME.build_responses_infer_execution_payload(
        data,
        route_info=route_info,
        instance=instance,
        instance_id=instance_id,
        backend=backend,
        capability=capability,
        request_model_override=request_model_override,
        upload_present=upload_present,
    )


def _filter_responses_infer_result(
    payload: Any,
    *,
    expose_input_artifacts: bool,
) -> dict[str, Any]:
    return _INFER_RUNTIME.filter_responses_infer_result(
        payload,
        expose_input_artifacts=expose_input_artifacts,
    )


def _artifact_identity_tokens(value: Any) -> set[str]:
    return _INFER_RUNTIME.artifact_identity_tokens(value)


def _normalize_reference_mirror_input_artifacts(payload: Any) -> dict[str, Any]:
    return _INFER_RUNTIME.normalize_reference_mirror_input_artifacts(payload)


def _claim_response_late_fill(response_id: str) -> bool:
    return _RESPONSES_RUNTIME.claim_response_late_fill(response_id)


def _release_response_late_fill(response_id: str) -> None:
    _RESPONSES_RUNTIME.release_response_late_fill(response_id)


_GHOST_ROUTE_RUNTIME = GhostRouteRuntimeOwner(
    hooks={
        'request_base_url': lambda: _request_base_url(),
        'build_instance_routing_summary': lambda entry, runtime_status: _build_instance_routing_summary(entry, runtime_status),
        'build_instance_responses_path': lambda instance_id: _build_instance_responses_path(instance_id),
        'pick_default_capability_instance': lambda candidates: _pick_default_capability_instance(candidates),
        'external_targets': lambda: _external_targets_payload(),
        'validate_external_target_request': lambda payload, **kwargs: _validate_codex_external_request(payload, **kwargs),
        'normalize_request_payload': lambda payload: _normalize_request_payload(payload),
        'merge_instances_with_runtime_status': lambda instances, **kwargs: merge_instances_with_runtime_status(instances, **kwargs),
        'load_running_instances': lambda: load_running_instances(),
        'runtime_status_path_getter': lambda: RUNTIME_STATUS_PATH,
        'extract_responses_prompt': lambda payload: _extract_responses_prompt(payload),
        'extract_responses_current_turn_prompt': lambda payload: _extract_responses_current_turn_prompt(payload),
        'extract_selected_reference_artifacts': lambda payload: _extract_selected_reference_artifacts(payload),
        'extract_ghost_preferences': lambda payload: _extract_ghost_preferences(payload),
        'extract_ghost_route_messages': lambda payload, **kwargs: _extract_ghost_route_messages(payload, **kwargs),
        'apply_ghost_preferences_to_route_context': lambda route_context, ghost_preferences: _apply_ghost_preferences_to_route_context(route_context, ghost_preferences),
        'extract_artifact_ref': lambda payload: _extract_artifact_ref(payload),
        'sanitize_selected_reference_artifacts': lambda payload: _sanitize_selected_reference_artifacts(payload),
        'extract_ghost_preview_route': lambda payload: _extract_ghost_preview_route(payload),
        'ghost_guide_path_getter': lambda: GHOST_GUIDE_PATH,
        'flask_log_path_getter': lambda: FLASK_LOG_PATH,
        'event_log_path_getter': lambda: EVENT_LOG_PATH,
        'attach_embedding_hints_to_route_context': lambda route_context, **kwargs: _attach_embedding_hints_to_route_context(route_context, **kwargs),
        'timeout_ms_to_seconds': lambda timeout_ms: _timeout_ms_to_seconds(timeout_ms),
        'estimate_route_context_tokens': lambda **kwargs: _estimate_route_context_tokens(**kwargs),
        'execute_chat_backend_request': lambda **kwargs: _execute_chat_backend_request(**kwargs),
        'log_unified_event': lambda **kwargs: _log_unified_event(**kwargs),
        'pick_prompt_preferred_instance': lambda instances, prompt: _pick_prompt_preferred_instance(instances, prompt),
        'instance_supports_capability': lambda instance, capability: _instance_supports_capability(instance, capability),
        'build_working_frame': lambda *args, **kwargs: _build_working_frame(*args, **kwargs),
        'requests_post': lambda *args, **kwargs: requests.post(*args, **kwargs),
        'chat_timeout_seconds': lambda *args, **kwargs: _chat_timeout_seconds(*args, **kwargs),
        'execute_embedding_backend_request': lambda **kwargs: _execute_embedding_backend_request(**kwargs),
    },
    wrapper_capability_aliases=_WRAPPER_CAPABILITY_ALIASES,
    max_recent_messages=MAX_RECENT_MESSAGES,
)


_BACKEND_TRANSPORT_RUNTIME = BackendTransportRuntimeOwner(
    hooks={
        'chat_timeout_seconds': lambda *args, **kwargs: _chat_timeout_seconds(*args, **kwargs),
        'normalize_chat_messages_for_backend': lambda *args, **kwargs: _normalize_chat_messages_for_backend(*args, **kwargs),
        'requests_post': lambda *args, **kwargs: requests.post(*args, **kwargs),
        'requests_module': requests,
        'to_base64': lambda path: _to_base64(path),
        'generated_audio_dir': GENERATED_AUDIO_DIR,
    },
    capability_chat=CAPABILITY_CHAT,
    request_timeout_error=REQUEST_TIMEOUT_ERROR,
    request_connection_error=REQUEST_CONNECTION_ERROR,
    request_exception_error=REQUEST_EXCEPTION_ERROR,
)


_POST_RESPONSE_SUBSTRATE_HYGIENE_RUNTIME = PostResponseSubstrateHygieneRuntimeOwner(
    policy=normalize_post_response_substrate_unload_policy(
        os.environ.get('OLLMO_POST_RESPONSE_SUBSTRATE_UNLOAD')
    ),
    load_running_instances=lambda: load_running_instances(),
    merge_instances_with_runtime_status=lambda instances, **kwargs: merge_instances_with_runtime_status(instances, **kwargs),
    log_unified_event=lambda **kwargs: _log_unified_event(**kwargs),
    requests_post=lambda *args, **kwargs: requests.post(*args, **kwargs),
    runtime_status_path_getter=lambda: RUNTIME_STATUS_PATH,
)


def _schedule_post_response_substrate_hygiene(
    response_payload: dict[str, Any],
    *,
    route_payload: Optional[dict[str, Any]] = None,
    reason: str = 'response_terminal',
) -> dict[str, Any]:
    return _POST_RESPONSE_SUBSTRATE_HYGIENE_RUNTIME.schedule_post_response_substrate_hygiene(
        response_payload,
        route_payload=route_payload,
        reason=reason,
    )


_CHAT_RUNTIME = ChatRuntimeOwner(
    hooks={
        'chat_timeout_seconds': lambda *args, **kwargs: _chat_timeout_seconds(*args, **kwargs),
        'normalize_backend': lambda value: normalize_backend(value),
        'build_runtime_status_stub': lambda *args, **kwargs: _build_runtime_status_stub(*args, **kwargs),
        'build_canonical_response_payload': lambda *args, **kwargs: _build_canonical_response_payload(*args, **kwargs),
        'attach_response_semantic_phase_payload': lambda *args, **kwargs: _attach_response_semantic_phase_payload(*args, **kwargs),
        'normalize_response_lookup_id': lambda value: _normalize_response_lookup_id(value),
        'register_response_lookup': lambda **kwargs: _register_response_lookup(**kwargs),
        'touch_response_lookup': lambda response_id, **kwargs: _touch_response_lookup(response_id, **kwargs),
        'register_response_stream': lambda response_id: _register_response_stream(response_id),
        'append_response_stream_events': lambda response_id, events, **kwargs: _append_response_stream_events(response_id, events, **kwargs),
        'wait_for_response_stream_events': lambda response_id, cursor, **kwargs: _wait_for_response_stream_events(response_id, cursor, **kwargs),
        'close_response_stream': lambda response_id: _close_response_stream(response_id),
        'attach_pre_freeze_closure_review': lambda *args, **kwargs: _attach_pre_freeze_closure_review(*args, **kwargs),
        'finalize_response_frame_payload': lambda *args, **kwargs: _finalize_response_frame_payload(*args, **kwargs),
        'schedule_response_late_fill': lambda **kwargs: _schedule_response_late_fill(**kwargs),
        'late_fill_stream_waits_for_terminal': lambda: not bool(app.config.get('TESTING')),
        'schedule_post_response_substrate_hygiene': lambda *args, **kwargs: _schedule_post_response_substrate_hygiene(*args, **kwargs),
        'apply_direct_artifact_materialization_closure': lambda *args, **kwargs: _RESPONSES_REQUEST_RUNTIME.apply_direct_artifact_materialization_closure(*args, **kwargs),
        'log_unified_event': lambda **kwargs: _log_unified_event(**kwargs),
        'record_instance_activity': lambda *args, **kwargs: record_instance_activity(*args, **kwargs),
        'record_instance_success': lambda *args, **kwargs: record_instance_success(*args, **kwargs),
        'record_instance_failure': lambda *args, **kwargs: record_instance_failure(*args, **kwargs),
        'log_runtime_status_transition': lambda *args, **kwargs: _log_runtime_status_transition(*args, **kwargs),
        'runtime_status_path_getter': lambda: RUNTIME_STATUS_PATH,
        'open_openai_chat_stream': lambda **kwargs: _open_openai_chat_stream(**kwargs),
        'open_ollama_chat_stream': lambda **kwargs: _open_ollama_chat_stream(**kwargs),
        'iter_openai_stream_deltas': lambda response: _iter_openai_stream_deltas(response),
        'iter_ollama_stream_deltas': lambda response: _iter_ollama_stream_deltas(response),
        'execute_chat_backend_request': lambda **kwargs: _execute_chat_backend_request(**kwargs),
        'persist_generated_text_artifact_if_requested': lambda *args, **kwargs: _persist_generated_text_artifact_if_requested(*args, **kwargs),
        'extract_responses_current_turn_prompt': lambda payload: _extract_responses_current_turn_prompt(payload),
        'request_exception_details': lambda exc: _request_exception_details(exc),
        'normalize_capability': lambda value: normalize_capability(value),
        'parse_float_with_bounds': lambda *args, **kwargs: _parse_float_with_bounds(*args, **kwargs),
        'parse_int_with_bounds': lambda *args, **kwargs: _parse_int_with_bounds(*args, **kwargs),
        'normalize_external_identifier': lambda value, **kwargs: _normalize_external_identifier(value, **kwargs),
        'load_running_instances': lambda: load_running_instances(),
        'infer_capability': lambda *args, **kwargs: infer_capability(*args, **kwargs),
        'is_port_listening': lambda port: is_port_listening(port),
    },
    capability_embedding=CAPABILITY_EMBEDDING,
    capability_image_generation=CAPABILITY_IMAGE_GENERATION,
    capability_speech_to_text=CAPABILITY_SPEECH_TO_TEXT,
    capability_text_to_speech=CAPABILITY_TEXT_TO_SPEECH,
    request_timeout_error=REQUEST_TIMEOUT_ERROR,
    request_connection_error=REQUEST_CONNECTION_ERROR,
    request_exception_error=REQUEST_EXCEPTION_ERROR,
)


_REQUEST_INTAKE_RUNTIME = RequestIntakeRuntimeOwner(
    hooks={
        'normalize_backend': lambda value: normalize_backend(value),
        'normalize_capability': lambda value: normalize_capability(value),
        'normalize_external_identifier': lambda value, **kwargs: _normalize_external_identifier(value, **kwargs),
        'lookup_instance': lambda instance_id: _lookup_instance(instance_id),
        'merge_instances_with_runtime_status': lambda instances, **kwargs: merge_instances_with_runtime_status(instances, **kwargs),
        'load_running_instances': lambda: load_running_instances(),
        'runtime_status_path_getter': lambda: RUNTIME_STATUS_PATH,
        'instance_supports_capability': lambda instance, capability: _instance_supports_capability(instance, capability),
        'pick_default_capability_instance': lambda candidates: _pick_default_capability_instance(candidates),
        'wrapper_capability_aliases': _WRAPPER_CAPABILITY_ALIASES,
        'resolve_saved_downloadable_artifact_path': lambda path: _resolve_saved_downloadable_artifact_path(path),
        'sanitize_artifact_record': lambda *args, **kwargs: _sanitize_artifact_record(*args, **kwargs),
        'get_cached_generated_image_state': lambda path: _get_cached_generated_image_state(path),
        'read_chat_history': lambda conversation_id, **kwargs: read_chat_history(conversation_id, **kwargs),
        'chat_history_dir_getter': lambda: CHAT_HISTORY_DIR,
        'extract_responses_messages': lambda payload: _extract_responses_messages(payload),
        'extract_ghost_route_messages': lambda *args, **kwargs: _extract_ghost_route_messages(*args, **kwargs),
        'sanitize_ghost_messages': lambda messages: sanitize_ghost_messages(messages),
        'get_response_lookup_record': lambda response_id: _get_response_lookup_record(response_id),
        'extract_batch_image_prompts': (
            lambda text, **kwargs: _RESPONSE_SEMANTICS_RUNTIME.extract_batch_image_prompts(
                text,
                **kwargs,
            )
        ),
        'parse_bool': lambda value, **kwargs: _parse_bool(value, **kwargs),
    }
)

_RESPONSE_SEMANTICS_RUNTIME = ResponseSemanticsRuntimeOwner(
    hooks={
        'plan_compound_execution': lambda *args, **kwargs: plan_compound_execution(*args, **kwargs),
        'normalize_request_payload': lambda payload: _normalize_request_payload(payload),
        'merge_instances_with_runtime_status': lambda instances, **kwargs: merge_instances_with_runtime_status(instances, **kwargs),
        'load_running_instances': lambda: load_running_instances(),
        'runtime_status_path_getter': lambda: RUNTIME_STATUS_PATH,
        'planner_timeout_seconds_for_payload': lambda payload, **kwargs: _planner_timeout_seconds_for_payload(payload, **kwargs),
        'extract_ghost_route_messages': lambda payload: _extract_ghost_route_messages(payload),
        'execute_chat_backend_request': lambda **kwargs: _execute_chat_backend_request(**kwargs),
        'response_registry_now_iso': lambda: _response_registry_now_iso(),
        'extract_responses_prompt': lambda payload: _extract_responses_prompt(payload),
        'extract_responses_current_turn_prompt': lambda payload: _extract_responses_current_turn_prompt(payload),
        'build_instance_trait_summary': lambda instance: _build_instance_trait_summary(instance),
        'sanitize_selected_reference_artifacts': lambda payload: _sanitize_selected_reference_artifacts(payload),
        'get_response_lookup_record': lambda response_id: _get_response_lookup_record(response_id),
        'normalize_capability_list': lambda values: _normalize_capability_list(values),
        'normalize_late_fill_branches': lambda values: _normalize_late_fill_branches(values),
        'extract_request_meta': lambda payload: extract_request_meta(payload),
        'build_canonical_response_artifacts': lambda payload: _build_canonical_response_artifacts(payload),
    }
)

_MODEL_CONTROL_RUNTIME = ModelControlRuntimeOwner(
    hooks={
        'list_available_models': lambda **kwargs: list_available_models(**kwargs),
        'merge_instances_with_runtime_status': lambda instances, **kwargs: merge_instances_with_runtime_status(instances, **kwargs),
        'load_running_instances': lambda: load_running_instances(),
        'runtime_status_path_getter': lambda: RUNTIME_STATUS_PATH,
        'build_backend_fabric_snapshot': lambda **kwargs: build_backend_fabric_snapshot(**kwargs),
        'normalize_backend': lambda value: normalize_backend(value),
        'normalize_capability': lambda value: normalize_capability(value),
        'infer_capability': lambda *args, **kwargs: infer_capability(*args, **kwargs),
        'build_registry_metadata': lambda *args, **kwargs: build_registry_metadata(*args, **kwargs),
        'pull_model': lambda model_name, backend: pull_model(model_name, backend),
        'remove_model': lambda model_name, backend, **kwargs: remove_model(model_name, backend, **kwargs),
        'log_unified_event': lambda **kwargs: _log_unified_event(**kwargs),
        'canonical_model_name': lambda data: canonical_model_name(data),
        'start_model_request_error': lambda *args, **kwargs: StartModelRequestError(*args, **kwargs),
        'start_instance': lambda *args, **kwargs: start_instance(*args, **kwargs),
        'record_instance_started': lambda instance, **kwargs: record_instance_started(instance, **kwargs),
        'log_runtime_status_transition': lambda previous, current: _log_runtime_status_transition(previous, current),
        'instance_supports_capability': lambda instance, capability: _instance_supports_capability(instance, capability),
        'normalize_external_identifier': lambda value, **kwargs: _normalize_external_identifier(value, **kwargs),
        'stop_instance': lambda instance_id: stop_instance(instance_id),
        'remove_instance_status': lambda instance_id, **kwargs: remove_instance_status(instance_id, **kwargs),
        'cleanup_runtime_hygiene': lambda **kwargs: cleanup_runtime_hygiene(**kwargs),
        'config_file_name': lambda: Path(CONFIG_FILE_NAME),
        'flask_log_path': lambda: FLASK_LOG_PATH,
        'active_global_log_paths': lambda **kwargs: _active_global_log_paths(**kwargs),
        'build_stop_payload': lambda result, instance: build_stop_payload(result, instance),
    }
)

_INFER_SUPPORT_RUNTIME = InferSupportRuntimeOwner(
    hooks={
        'file_kind_from_name': lambda name: _file_kind_from_name(name),
        'persist_input_file_locally': lambda *args, **kwargs: _persist_input_file_locally(*args, **kwargs),
        'truncate_for_history': lambda text, **kwargs: truncate_for_history(text, **kwargs),
        'read_infer_history': lambda history_path, **kwargs: read_infer_history(history_path, **kwargs),
        'append_infer_history': lambda entry, **kwargs: append_infer_history(entry, **kwargs),
        'app_testing': lambda: app.config.get('TESTING'),
        'infer_history_path_getter': lambda: INFER_HISTORY_PATH,
        'infer_history_timestamp': lambda: dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
        'infer_history_entry_id': lambda: f"infer-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
    }
)

_OCR_PDF_RUNTIME = OcrPdfRuntimeOwner(
    hooks={
        'extract_pdf_text_content': lambda pdf_path, **kwargs: extract_pdf_text_content(pdf_path, **kwargs),
        'render_pdf_pages_to_base64': lambda pdf_path, **kwargs: render_pdf_pages_to_base64(pdf_path, **kwargs),
        'render_single_pdf_page_to_base64': lambda pdf_path, **kwargs: render_single_pdf_page_to_base64(pdf_path, **kwargs),
        'looks_like_ocr_prompt_echo': lambda content, **kwargs: looks_like_ocr_prompt_echo(content, **kwargs),
        'normalize_ocr_line': lambda line: normalize_ocr_line(line),
        'strip_ocr_structural_lines': lambda text: strip_ocr_structural_lines(text),
        'collapse_repeated_ocr_lines': lambda text, **kwargs: collapse_repeated_ocr_lines(text, **kwargs),
        'line_has_ocr_garbage_pattern': lambda line: line_has_ocr_garbage_pattern(line),
        'sanitize_ocr_noise_lines': lambda text: sanitize_ocr_noise_lines(text),
        'detect_low_quality_ocr_reason': lambda content: detect_low_quality_ocr_reason(content),
        'clean_ocr_output_text': lambda raw_text: clean_ocr_output_text(raw_text),
        'ocr_pdf_page_with_ollama': lambda **kwargs: ocr_pdf_page_with_ollama(**kwargs),
        'is_generic_ocr_instruction_prompt': lambda prompt: is_generic_ocr_instruction_prompt(prompt),
        'ocr_image_with_deepseek': lambda **kwargs: ocr_image_with_deepseek(**kwargs),
        'generate_func': lambda *args, **kwargs: _ollama_generate(*args, **kwargs),
        'extract_generate_content_func': lambda data: _extract_generate_content(data),
    },
    request_timeout_error=REQUEST_TIMEOUT_ERROR,
    request_connection_error=REQUEST_CONNECTION_ERROR,
    request_exception_error=REQUEST_EXCEPTION_ERROR,
)


_RESPONSES_REQUEST_RUNTIME = ResponsesRequestRuntimeOwner(
    hooks={
        'normalize_backend': lambda value: normalize_backend(value),
        'normalize_capability': lambda value: normalize_capability(value),
        'infer_supported_capabilities': lambda *args, **kwargs: infer_supported_capabilities(*args, **kwargs),
        'instance_effective_capability': lambda instance: _instance_effective_capability(instance),
        'compact_string_list': lambda values: _compact_string_list(values),
        'summarize_backend_metadata_for_routing': lambda value: _summarize_backend_metadata_for_routing(value),
        'summarize_backend_runtime_for_routing': lambda value: _summarize_backend_runtime_for_routing(value),
        'summarize_session_controls_for_routing': lambda value: _summarize_session_controls_for_routing(value),
        'build_working_frame': lambda *args, **kwargs: _build_working_frame(*args, **kwargs),
        'normalize_request_payload': lambda payload: _normalize_request_payload(payload),
        'promote_current_predecessor_context': (
            lambda payload: _REQUEST_INTAKE_RUNTIME.promote_current_predecessor_context(payload)
        ),
        'parse_bool': lambda value, **kwargs: _parse_bool(value, **kwargs),
        'save_upload_to_temp': lambda upload: _save_upload_to_temp(upload),
        'file_kind_from_name': lambda file_name: file_kind_from_name(file_name),
        'persist_request_input_artifacts': lambda *args, **kwargs: _persist_request_input_artifacts(*args, **kwargs),
        'persist_input_artifact_registry_records': lambda *args, **kwargs: _persist_input_artifact_registry_records(
            *args,
            ledger_path=ARTIFACT_REGISTRY_LEDGER,
            **kwargs,
        ),
        'merge_unique_artifact_records': lambda existing, incoming: _merge_unique_artifact_records(existing, incoming),
        'resolve_ghost_auto_route': lambda *args, **kwargs: _resolve_ghost_auto_route(*args, **kwargs),
        'resolve_responses_target_instance': lambda payload, **kwargs: _resolve_responses_target_instance(payload, **kwargs),
        'log_unified_event': lambda **kwargs: _log_unified_event(**kwargs),
        'prepare_effective_request_data': lambda *args, **kwargs: _prepare_effective_request_data(*args, **kwargs),
        'build_missing_required_session_controls': lambda *args, **kwargs: _build_missing_required_session_controls(*args, **kwargs),
        'select_backend_request_model': lambda *args, **kwargs: _select_backend_request_model(*args, **kwargs),
        'instance_supports_capability': lambda instance, capability: _instance_supports_capability(instance, capability),
        'infer_capability': lambda *args, **kwargs: infer_capability(*args, **kwargs),
        'normalize_response_lookup_id': lambda value: _normalize_response_lookup_id(value),
        'register_response_lookup': lambda **kwargs: _register_response_lookup(**kwargs),
        'touch_response_lookup': lambda response_id, **kwargs: _touch_response_lookup(response_id, **kwargs),
        'build_canonical_error_response_payload': lambda **kwargs: _build_canonical_error_response_payload(**kwargs),
        'finalize_response_frame_payload': lambda *args, **kwargs: _finalize_response_frame_payload(*args, **kwargs),
        'handle_responses_request': lambda **kwargs: _handle_responses_request(**kwargs),
        'extract_responses_batch_items': lambda payload: _extract_responses_batch_items(payload),
        'parse_float_with_bounds': lambda *args, **kwargs: _parse_float_with_bounds(*args, **kwargs),
        'parse_int_with_bounds': lambda *args, **kwargs: _parse_int_with_bounds(*args, **kwargs),
        'extract_selected_reference_artifacts': lambda payload: _extract_selected_reference_artifacts(payload),
        'select_matching_selected_reference_artifact': lambda *args, **kwargs: _select_matching_selected_reference_artifact(*args, **kwargs),
        'extract_responses_prompt': lambda payload: _extract_responses_prompt(payload),
        'extract_responses_current_turn_prompt': lambda payload: _extract_responses_current_turn_prompt(payload),
        'should_attach_selected_reference_file_context': lambda *args, **kwargs: _should_attach_selected_reference_file_context(*args, **kwargs),
        'extract_responses_messages': lambda payload: _extract_responses_messages(payload),
        'extract_ghost_route_messages': lambda *args, **kwargs: _extract_ghost_route_messages(*args, **kwargs),
        'inject_selected_reference_into_chat_messages': lambda messages, selected_reference_artifacts: _inject_selected_reference_into_chat_messages(messages, selected_reference_artifacts),
        'inject_ghost_runtime_policy_into_chat_messages': lambda messages, **kwargs: _inject_ghost_runtime_policy_into_chat_messages(messages, **kwargs),
        'inject_prepare_phase_contract_into_chat_messages': lambda messages, **kwargs: _inject_prepare_phase_contract_into_chat_messages(messages, **kwargs),
        'resolve_prepare_phase_contract': lambda **kwargs: _resolve_prepare_phase_contract(**kwargs),
        'build_external_prepare_phase_bounded_task': lambda **kwargs: _build_external_prepare_phase_bounded_task(**kwargs),
        'choose_context_strategy': lambda **kwargs: _choose_context_strategy(**kwargs),
        'apply_context_strategy': lambda messages, context_strategy: _apply_context_strategy(messages, context_strategy),
        'stream_chat_backend_as_responses': lambda **kwargs: _stream_chat_backend_as_responses(**kwargs),
        'execute_chat_backend_request': lambda **kwargs: _execute_chat_backend_request(**kwargs),
        'execute_external_text_target': lambda prompt, **kwargs: _execute_codex_external_text(prompt, **kwargs),
        'validate_external_text_request': lambda payload, **kwargs: _validate_codex_external_request(payload, **kwargs),
        'build_external_target_inputs': lambda payload: build_codex_execution_inputs(payload),
        'apply_selected_reference_prompt_prefix': lambda prompt, selected, capability: _apply_selected_reference_prompt_prefix(prompt, selected, capability),
        'external_execution_failure': lambda result: codex_execution_failure(result),
        'persist_generated_text_artifact_if_requested': lambda *args, **kwargs: _persist_generated_text_artifact_if_requested(*args, **kwargs),
        'request_exception_details': lambda exc: _request_exception_details(exc),
        'build_canonical_response_payload': lambda *args, **kwargs: _build_canonical_response_payload(*args, **kwargs),
        'attach_response_semantic_phase_payload': lambda *args, **kwargs: _attach_response_semantic_phase_payload(*args, **kwargs),
        'truth_gate_response_output_claims': lambda payload, **kwargs: _truth_gate_response_output_claims(payload, **kwargs),
        'build_artifact_completion_gap_spec': lambda *args, **kwargs: _build_artifact_completion_gap_spec(*args, **kwargs),
        'build_pre_freeze_closure_review_gap': lambda *args, **kwargs: _build_pre_freeze_closure_review_gap(*args, **kwargs),
        'build_graph_closure_review': lambda *args, **kwargs: _build_graph_closure_review(*args, **kwargs),
        'build_late_fill_state': lambda *args, **kwargs: _build_late_fill_state(*args, **kwargs),
        'attach_late_fill_state': lambda *args, **kwargs: _attach_late_fill_state(*args, **kwargs),
        'ensure_response_lookup_for_payload': lambda payload, **kwargs: _ensure_response_lookup_for_payload(payload, **kwargs),
        'schedule_response_late_fill': lambda **kwargs: _schedule_response_late_fill(**kwargs),
        'schedule_post_response_substrate_hygiene': lambda *args, **kwargs: _schedule_post_response_substrate_hygiene(*args, **kwargs),
        'read_chat_history': lambda conversation_id, **kwargs: read_chat_history(conversation_id, history_dir=CHAT_HISTORY_DIR, **kwargs),
        'find_artifact_registry_record': lambda artifact_path: _find_artifact_registry_record(
            artifact_path,
            ledger_path=ARTIFACT_REGISTRY_LEDGER,
        ),
        'finalize_terminal_materialization_contract': lambda *args, **kwargs: _LATE_FILL_RUNTIME.finalize_terminal_materialization_contract(*args, **kwargs),
        'build_responses_infer_execution_payload': lambda payload, **kwargs: _build_responses_infer_execution_payload(payload, **kwargs),
        'analyze_prompt_intent': lambda prompt: analyze_prompt_intent(prompt),
        'image_aspect_preset_dimensions': IMAGE_ASPECT_PRESET_DIMENSIONS,
        'invoke_internal_api_json_route': lambda **kwargs: _invoke_internal_api_json_route(**kwargs),
        'filter_responses_infer_result': lambda payload, **kwargs: _filter_responses_infer_result(payload, **kwargs),
        'build_canonical_batch_response_payload': lambda **kwargs: _build_canonical_batch_response_payload(**kwargs),
        'build_canonical_response_stream_events': lambda payload: _build_canonical_response_stream_events_for_ui(payload),
        'execute_materialization_branches': lambda branches, **kwargs: _MULTI_MATERIALIZATION_RUNTIME.execute_materialization_branches(branches, **kwargs),
        'project_response_payload_for_wire': lambda payload: _project_response_payload_for_wire(payload),
    },
    capability_chat=CAPABILITY_CHAT,
    capability_embedding=CAPABILITY_EMBEDDING,
    capability_image_generation=CAPABILITY_IMAGE_GENERATION,
    capability_speech_to_text=CAPABILITY_SPEECH_TO_TEXT,
    request_timeout_error=REQUEST_TIMEOUT_ERROR,
    request_exception_error=REQUEST_EXCEPTION_ERROR,
)


def _attach_pre_freeze_closure_review(
    response_payload: dict[str, Any],
    *,
    output_text: str,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
    artifact_gap: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    return _RESPONSES_REQUEST_RUNTIME.attach_pre_freeze_closure_review(
        response_payload,
        output_text=output_text,
        route_payload=route_payload,
        request_payload=request_payload,
        artifact_gap=artifact_gap,
    )


_MULTI_MATERIALIZATION_RUNTIME = MultiMaterializationRuntimeOwner(
    max_parallel_workers=normalize_max_parallel_workers(
        os.environ.get('OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS')
    )
)


_LATE_FILL_RUNTIME = LateFillRuntimeOwner(
    normalize_request_payload=lambda payload: _normalize_request_payload(payload),
    extract_request_meta=lambda payload: extract_request_meta(payload),
    attach_request_meta=lambda payload: attach_request_meta(payload),
    extract_responses_prompt=lambda payload: _extract_responses_prompt(payload),
    sanitize_selected_reference_artifacts=lambda *args, **kwargs: _sanitize_selected_reference_artifacts(
        *args,
        **kwargs,
    ),
    parse_bool=lambda value, **kwargs: _parse_bool(value, **kwargs),
    extract_ghost_route_messages=lambda payload: _extract_ghost_route_messages(payload),
    response_registry_now_iso=lambda: _response_registry_now_iso(),
    max_recent_messages=MAX_RECENT_MESSAGES,
    normalize_capability=lambda value: normalize_capability(value),
    capability_text_to_speech=CAPABILITY_TEXT_TO_SPEECH,
    capability_image_generation=CAPABILITY_IMAGE_GENERATION,
    resolve_ghost_auto_route=lambda payload, **kwargs: _resolve_ghost_auto_route(payload, **kwargs),
    merge_instances_with_runtime_status=merge_instances_with_runtime_status,
    load_running_instances=load_running_instances,
    runtime_status_path_getter=lambda: RUNTIME_STATUS_PATH,
    instance_supports_capability=lambda instance, capability: _instance_supports_capability(instance, capability),
    pick_prompt_preferred_instance=lambda candidates, prompt: _pick_prompt_preferred_instance(candidates, prompt),
    pick_default_capability_instance=lambda candidates: _pick_default_capability_instance(candidates),
    merge_request_meta_runtime_truth=lambda *args, **kwargs: _merge_request_meta_runtime_truth(*args, **kwargs),
    prepare_effective_request_data=lambda *args, **kwargs: _prepare_effective_request_data(*args, **kwargs),
    build_missing_required_session_controls=lambda *args, **kwargs: _build_missing_required_session_controls(*args, **kwargs),
    normalize_backend=lambda value: normalize_backend(value),
    select_backend_request_model=lambda *args, **kwargs: _select_backend_request_model(*args, **kwargs),
    build_responses_infer_execution_payload=lambda *args, **kwargs: _build_responses_infer_execution_payload(*args, **kwargs),
    invoke_internal_api_json_route=lambda *args, **kwargs: _invoke_internal_api_json_route(*args, **kwargs),
    filter_responses_infer_result=lambda *args, **kwargs: _filter_responses_infer_result(*args, **kwargs),
    artifact_type_for_capability=lambda capability: _artifact_type_for_capability(capability),
    semantic_payload_for_capability=lambda *args, **kwargs: _semantic_payload_for_capability(*args, **kwargs),
    get_response_lookup_record=lambda response_id: _get_response_lookup_record(response_id),
    normalize_capability_list=lambda values: _normalize_capability_list(values),
    extract_pending_deferred_branches=lambda *args, **kwargs: _extract_pending_deferred_branches(*args, **kwargs),
    extract_pending_deferred_capabilities=lambda *args, **kwargs: _extract_pending_deferred_capabilities(*args, **kwargs),
    build_pending_late_fill_branches=lambda *args, **kwargs: _build_pending_late_fill_branches(*args, **kwargs),
    branch_id=lambda branch: _branch_id(branch),
    branch_capability=lambda branch: _branch_capability(branch),
    artifact_gap_is_already_fulfilled=lambda *args, **kwargs: _artifact_gap_is_already_fulfilled(*args, **kwargs),
    build_late_fill_state=lambda *args, **kwargs: _build_late_fill_state(*args, **kwargs),
    build_graph_closure_review=lambda *args, **kwargs: _build_graph_closure_review(*args, **kwargs),
    attach_graph_closure_review_diagnostics=lambda *args, **kwargs: _RESPONSES_REQUEST_RUNTIME._attach_graph_closure_review_diagnostics(*args, **kwargs),
    finalize_response_frame_payload=lambda *args, **kwargs: _finalize_response_frame_payload(*args, **kwargs),
    attach_late_fill_state=lambda *args, **kwargs: _attach_late_fill_state(*args, **kwargs),
    touch_response_lookup=lambda *args, **kwargs: _touch_response_lookup(*args, **kwargs),
    ensure_response_lookup_for_payload=lambda *args, **kwargs: _ensure_response_lookup_for_payload(*args, **kwargs),
    build_canonical_response_artifacts=lambda payload: _build_canonical_response_artifacts(payload),
    downstream_request_phase_batches=lambda *args, **kwargs: _downstream_request_phase_branch_batches(*args, **kwargs),
    late_fill_capability_counts=lambda branches: _late_fill_capability_counts(branches),
    normalize_late_fill_branches=lambda value: _normalize_late_fill_branches(value),
    log_unified_event=lambda **kwargs: _log_unified_event(**kwargs),
    execute_materialization_branches=lambda branches, **kwargs: _MULTI_MATERIALIZATION_RUNTIME.execute_materialization_branches(branches, **kwargs),
    schedule_post_response_substrate_hygiene=lambda *args, **kwargs: _schedule_post_response_substrate_hygiene(*args, **kwargs),
    attach_runtime_graph_repair_evidence=lambda payload: _RESPONSES_REQUEST_RUNTIME._attach_runtime_graph_repair_evidence(payload),
    claim_response_late_fill=lambda response_id: _claim_response_late_fill(response_id),
    release_response_late_fill=lambda response_id: _release_response_late_fill(response_id),
    persist_image_data_url_locally=lambda image_data_url, model_name: _persist_image_data_url_locally(image_data_url, model_name),
    review_terminal_graph_rebase=lambda *args, **kwargs: _RESPONSES_REQUEST_RUNTIME.review_terminal_graph_rebase_after_late_fill(*args, **kwargs),
    prepare_terminal_graph_patch_successor=lambda payload: _RESPONSES_REQUEST_RUNTIME.prepare_terminal_graph_patch_successor(payload),
    load_latest_response_state=lambda response_id: _load_latest_response_state(
        response_id,
        frames_dir=RESPONSE_FRAMES_DIR,
    ),
    load_latest_response_observation_state=lambda response_id: (
        _load_latest_response_observation_state(
            response_id,
            frames_dir=RESPONSE_FRAMES_DIR,
        )
    ),
    load_external_targets=lambda: _external_targets_payload(),
    execute_external_chat_phase=lambda **kwargs: (
        _RESPONSES_REQUEST_RUNTIME.execute_bounded_external_chat_phase(
            **kwargs
        )
    ),
)


def _prepare_late_fill_request_payload(
    request_payload: dict[str, Any],
    *,
    expected_capability: str,
    assistant_message: str,
    artifact_gap: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.prepare_late_fill_request_payload(
        request_payload,
        expected_capability=expected_capability,
        assistant_message=assistant_message,
        artifact_gap=artifact_gap,
    )


def _resolve_late_fill_route(
    request_payload: dict[str, Any],
    *,
    expected_capability: str,
    failed_instance_id: Optional[str],
    excluded_instance_ids: Optional[list[str]] = None,
    artifact_gap: Optional[dict[str, Any]] = None,
    source_route_payload: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[str]]:
    return _LATE_FILL_RUNTIME.resolve_late_fill_route(
        request_payload,
        expected_capability=expected_capability,
        failed_instance_id=failed_instance_id,
        excluded_instance_ids=excluded_instance_ids,
        artifact_gap=artifact_gap,
        source_route_payload=source_route_payload,
        resolve_ghost_auto_route=_resolve_ghost_auto_route,
        load_running_instances=load_running_instances,
        merge_instances_with_runtime_status=merge_instances_with_runtime_status,
    )


def _merge_late_fill_result_into_response_payload(
    response_payload: dict[str, Any],
    infer_result: dict[str, Any],
    late_fill_state: dict[str, Any],
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.merge_late_fill_result_into_response_payload(
        response_payload,
        infer_result,
        late_fill_state,
    )


def _request_phase_graph_for_late_fill(
    *,
    route_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.request_phase_graph_for_late_fill(
        route_payload=route_payload,
        artifact_payload=artifact_payload,
    )


def _merge_unique_artifact_records(
    existing_values: Any,
    incoming_values: Any,
) -> list[dict[str, Any]]:
    return _LATE_FILL_RUNTIME.merge_unique_artifact_records(
        existing_values,
        incoming_values,
    )


def _merge_late_fill_result_fields(
    response_payload: dict[str, Any],
    infer_result: dict[str, Any],
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.merge_late_fill_result_fields(
        response_payload,
        infer_result,
    )


def _prepare_late_fill_branch_plan(
    *,
    expected_capability: str,
    artifact_gap: dict[str, Any],
    current_payload: dict[str, Any],
    request_payload: dict[str, Any],
    assistant_message: str,
    source_route_payload: Optional[dict[str, Any]],
    failed_instance_id: Optional[str],
    excluded_instance_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.prepare_late_fill_branch_plan(
        expected_capability=expected_capability,
        artifact_gap=artifact_gap,
        current_payload=current_payload,
        request_payload=request_payload,
        assistant_message=assistant_message,
        source_route_payload=source_route_payload,
        failed_instance_id=failed_instance_id,
        excluded_instance_ids=excluded_instance_ids,
        build_deferred_follow_up_gap_for_capability=_build_deferred_follow_up_gap_for_capability,
        prepare_late_fill_request_payload=_prepare_late_fill_request_payload,
        resolve_late_fill_route=_resolve_late_fill_route,
    )


def _execute_prepared_late_fill_branch(plan: dict[str, Any]) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.execute_prepared_late_fill_branch(plan)


def _execute_late_fill_branch(
    *,
    expected_capability: str,
    artifact_gap: dict[str, Any],
    current_payload: dict[str, Any],
    request_payload: dict[str, Any],
    assistant_message: str,
    source_route_payload: Optional[dict[str, Any]],
    failed_instance_id: Optional[str],
    excluded_instance_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.execute_late_fill_branch(
        expected_capability=expected_capability,
        artifact_gap=artifact_gap,
        current_payload=current_payload,
        request_payload=request_payload,
        assistant_message=assistant_message,
        source_route_payload=source_route_payload,
        failed_instance_id=failed_instance_id,
        excluded_instance_ids=excluded_instance_ids,
        prepare_late_fill_branch_plan=_prepare_late_fill_branch_plan,
    )


def _build_deferred_follow_up_gap_for_capability(
    artifact_gap: Optional[dict[str, Any]],
    *,
    capability: Optional[str],
    artifact_payload: Optional[dict[str, Any]] = None,
    pending_capabilities: Optional[list[str]] = None,
    completed_capabilities: Optional[list[str]] = None,
    failed_capabilities: Optional[list[str]] = None,
) -> dict[str, Any]:
    return _LATE_FILL_RUNTIME.build_deferred_follow_up_gap_for_capability(
        artifact_gap,
        capability=capability,
        artifact_payload=artifact_payload,
        pending_capabilities=pending_capabilities,
        completed_capabilities=completed_capabilities,
        failed_capabilities=failed_capabilities,
    )


def _complete_response_late_fill(
    *,
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
    assistant_message: str,
    artifact_gap: dict[str, Any],
    source_route_payload: Optional[dict[str, Any]],
) -> None:
    _LATE_FILL_RUNTIME.complete_response_late_fill(
        response_payload=response_payload,
        request_payload=request_payload,
        assistant_message=assistant_message,
        artifact_gap=artifact_gap,
        source_route_payload=source_route_payload,
        build_deferred_follow_up_gap_for_capability=_build_deferred_follow_up_gap_for_capability,
        prepare_late_fill_branch_plan=_prepare_late_fill_branch_plan,
        execute_prepared_late_fill_branch=_execute_prepared_late_fill_branch,
    )


def _schedule_response_late_fill(
    *,
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
    assistant_message: str,
    artifact_gap: dict[str, Any],
    source_route_payload: Optional[dict[str, Any]],
) -> bool:
    return _LATE_FILL_RUNTIME.schedule_response_late_fill(
        response_payload=response_payload,
        request_payload=request_payload,
        assistant_message=assistant_message,
        artifact_gap=artifact_gap,
        source_route_payload=source_route_payload,
        complete_response_late_fill=_complete_response_late_fill,
    )


def _request_payload_for_late_fill_retry(response_payload: dict[str, Any]) -> dict[str, Any]:
    frame = response_payload.get('response_frame') if isinstance(response_payload.get('response_frame'), Mapping) else {}
    frame_request = frame.get('request') if isinstance(frame.get('request'), Mapping) else {}
    request_payload = dict(frame_request)
    runtime_payload = response_payload.get('runtime') if isinstance(response_payload.get('runtime'), Mapping) else {}
    request_phase_graph = (
        runtime_payload.get('request_phase_graph')
        if isinstance(runtime_payload.get('request_phase_graph'), Mapping)
        else {}
    )
    if not str(request_payload.get('prompt') or request_payload.get('input') or '').strip():
        graph_prompt = str(request_phase_graph.get('prompt') or '').strip()
        if graph_prompt:
            request_payload['prompt'] = graph_prompt
    if response_payload.get('request_meta') and isinstance(response_payload.get('request_meta'), Mapping):
        request_payload.setdefault('request_meta', dict(response_payload.get('request_meta') or {}))
    return request_payload


def _response_late_fill_is_in_flight(response_id: str) -> bool:
    normalized_response_id = _normalize_response_lookup_id(response_id)
    if not normalized_response_id:
        return False
    with _RESPONSE_LATE_FILL_LOCK:
        return normalized_response_id in _RESPONSE_LATE_FILL_IN_FLIGHT


def _branch_retry_trigger(branch: Mapping[str, Any]) -> str:
    recovery_attempt = (
        branch.get('recovery_attempt')
        if isinstance(branch.get('recovery_attempt'), Mapping)
        else {}
    )
    recovery_state = (
        branch.get('recovery_state')
        if isinstance(branch.get('recovery_state'), Mapping)
        else {}
    )
    return str(
        recovery_attempt.get('trigger')
        or recovery_state.get('trigger')
        or ''
    ).strip().lower()


def _late_fill_branch_is_orphanable_retry_attempt(branch: Any) -> bool:
    if not isinstance(branch, Mapping):
        return False
    if _branch_retry_trigger(branch) != 'explicit_retry_endpoint':
        return False
    branch_status = str(branch.get('status') or '').strip().lower()
    recovery_state = (
        branch.get('recovery_state')
        if isinstance(branch.get('recovery_state'), Mapping)
        else {}
    )
    state_status = str(recovery_state.get('status') or '').strip().lower()
    return (
        branch_status in _ACTIVE_LATE_FILL_BRANCH_STATUSES
        or state_status in _ACTIVE_LATE_FILL_BRANCH_STATUSES
    )


def _list_text_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        token = str(item or '').strip()
        if token and token not in result:
            result.append(token)
    return result


def _first_text_value(*values: Any) -> str:
    for value in values:
        token = str(value or '').strip()
        if token:
            return token
    return ''


def _build_orphaned_retry_failed_branch(branch: dict[str, Any]) -> dict[str, Any]:
    recovery_context = (
        branch.get('recovery_context')
        if isinstance(branch.get('recovery_context'), Mapping)
        else {}
    )
    recovery_state = (
        branch.get('recovery_state')
        if isinstance(branch.get('recovery_state'), Mapping)
        else {}
    )
    recovery_attempt = (
        branch.get('recovery_attempt')
        if isinstance(branch.get('recovery_attempt'), Mapping)
        else {}
    )
    attempt = branch.get('attempt') if isinstance(branch.get('attempt'), Mapping) else {}
    branch_id = _branch_id(branch)
    capability = _branch_capability(branch)
    failed_instance_id = _first_text_value(
        branch.get('failed_instance_id'),
        recovery_state.get('failed_instance_id'),
        recovery_attempt.get('failed_instance_id'),
        attempt.get('instance_id'),
    )
    excluded_instance_ids: list[str] = []
    for value in (
        branch.get('excluded_instance_ids'),
        recovery_context.get('exclude_instance_ids'),
        recovery_state.get('exclude_instance_ids'),
        recovery_attempt.get('excluded_instance_ids'),
    ):
        for token in _list_text_values(value):
            if token not in excluded_instance_ids:
                excluded_instance_ids.append(token)
    if failed_instance_id and failed_instance_id not in excluded_instance_ids:
        excluded_instance_ids.append(failed_instance_id)
    retry_scope = _first_text_value(
        recovery_state.get('retry_scope'),
        recovery_context.get('retry_scope'),
        'same_branch',
    )
    suggested_action = normalize_recovery_suggested_action(
        recovery_state.get('suggested_action') or recovery_context.get('suggested_action'),
        default=RECOVERY_ACTION_RETRY_SAME_BRANCH,
    )
    failed_branch = {
        key: value
        for key, value in branch.items()
        if key not in {
            'error',
            'recovery_attempt',
            'recovery_context',
            'recovery_state',
            'status',
        }
    }
    failed_branch['status'] = 'failed'
    if failed_instance_id:
        failed_branch['failed_instance_id'] = failed_instance_id
    if excluded_instance_ids:
        failed_branch['excluded_instance_ids'] = excluded_instance_ids
    failed_branch['error'] = {
        'code': 'ORPHANED_RETRY_ATTEMPT',
        'message': 'Late-fill retry attempt was interrupted before completion; retry can be scheduled again.',
        'stage': 'explicit_retry_endpoint',
        'retryable': True,
    }
    failed_branch['recovery_context'] = {
        'can_retry': True,
        'retry_scope': retry_scope,
        'suggested_action': suggested_action,
        'preserve_intent': True,
    }
    if excluded_instance_ids:
        failed_branch['recovery_context']['exclude_instance_ids'] = excluded_instance_ids
    failed_branch['recovery_state'] = {
        'kind': 'ollmo.late_fill_recovery_state',
        'status': 'candidate',
        'trigger': 'orphaned_retry_attempt',
        'branch_id': branch_id,
        'capability': capability,
        'promotion_required': True,
        'auto_execute': False,
        'preserve_intent': True,
        'retry_scope': retry_scope,
        'suggested_action': suggested_action,
    }
    if failed_instance_id:
        failed_branch['recovery_state']['failed_instance_id'] = failed_instance_id
    if excluded_instance_ids:
        failed_branch['recovery_state']['exclude_instance_ids'] = excluded_instance_ids
    return {
        key: value
        for key, value in failed_branch.items()
        if value not in (None, '', [], {})
    }


_STALE_LATE_FILL_ACTIVE_RECONCILE_AFTER_SEC = 60
_ACTIVE_BACKEND_ACTIVITIES = {'busy', 'running', 'active', 'in_progress', 'starting', 'loading'}


def _parse_response_timestamp(value: Any) -> Optional[dt.datetime]:
    token = str(value or '').strip()
    if not token:
        return None
    try:
        parsed = dt.datetime.fromisoformat(token.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _late_fill_stale_active_updated_at(
    *,
    record: Optional[Mapping[str, Any]],
    response_payload: Mapping[str, Any],
    late_fill: Mapping[str, Any],
) -> Optional[dt.datetime]:
    for value in (
        late_fill.get('updated_at'),
        response_payload.get('updated_at'),
        (record or {}).get('updated_at') if isinstance(record, Mapping) else None,
    ):
        parsed = _parse_response_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _runtime_has_active_backend_work() -> bool:
    try:
        instances = merge_instances_with_runtime_status(
            load_running_instances(),
            path=RUNTIME_STATUS_PATH,
            refresh=True,
        )
    except Exception as exc:  # noqa: BLE001
        logging.info('Could not inspect runtime activity for stale late-fill reconciliation: %s', exc)
        return True
    for instance in instances or []:
        if not isinstance(instance, Mapping):
            continue
        runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), Mapping) else {}
        activity = str(instance.get('activity') or runtime_status.get('activity') or '').strip().lower()
        readiness = str(instance.get('readiness') or runtime_status.get('readiness') or '').strip().lower()
        if activity in _ACTIVE_BACKEND_ACTIVITIES or readiness in {'starting', 'loading'}:
            return True
    return False


def _build_stale_active_failed_branch(branch: dict[str, Any]) -> dict[str, Any]:
    branch_id = _branch_id(branch)
    capability = _branch_capability(branch)
    attempt = branch.get('attempt') if isinstance(branch.get('attempt'), Mapping) else {}
    failed_instance_id = _first_text_value(
        branch.get('failed_instance_id'),
        branch.get('fill_instance_id'),
        branch.get('instance_id'),
        attempt.get('instance_id'),
    )
    excluded_instance_ids = [
        token
        for token in _list_text_values(branch.get('excluded_instance_ids'))
        if token
    ]
    if failed_instance_id and failed_instance_id not in excluded_instance_ids:
        excluded_instance_ids.append(failed_instance_id)
    suggested_action = (
        RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE
        if failed_instance_id
        else RECOVERY_ACTION_RETRY_SAME_BRANCH
    )
    failed_branch = {
        key: value
        for key, value in branch.items()
        if key not in {'error', 'recovery_context', 'recovery_state', 'status'}
    }
    failed_branch['status'] = 'failed'
    failed_branch['error'] = {
        'code': 'STALE_LATE_FILL_ACTIVE_RECONCILED',
        'message': (
            'Late-fill branch was still marked active, but no backend work is in flight '
            'and runtime backends are idle.'
        ),
        'stage': 'late_fill_observer',
        'retryable': True,
    }
    failed_branch['recovery_context'] = {
        'can_retry': True,
        'retry_scope': 'same_branch',
        'suggested_action': suggested_action,
        'preserve_intent': True,
    }
    if failed_instance_id:
        failed_branch['failed_instance_id'] = failed_instance_id
    if excluded_instance_ids:
        failed_branch['excluded_instance_ids'] = excluded_instance_ids
        failed_branch['recovery_context']['exclude_instance_ids'] = excluded_instance_ids
    failed_branch['recovery_state'] = {
        'kind': 'ollmo.late_fill_recovery_state',
        'status': 'candidate',
        'trigger': 'stale_active_reconciliation',
        'branch_id': branch_id,
        'capability': capability,
        'promotion_required': True,
        'auto_execute': False,
        'preserve_intent': True,
        'retry_scope': 'same_branch',
        'suggested_action': suggested_action,
    }
    if failed_instance_id:
        failed_branch['recovery_state']['failed_instance_id'] = failed_instance_id
    if excluded_instance_ids:
        failed_branch['recovery_state']['exclude_instance_ids'] = excluded_instance_ids
    return {
        key: value
        for key, value in failed_branch.items()
        if value not in (None, '', [], {})
    }


def _project_stale_late_fill_active_branches(
    response_id: str,
    response_payload: dict[str, Any],
    *,
    record: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], bool]:
    current_late_fill = (
        response_payload.get('late_fill')
        if isinstance(response_payload.get('late_fill'), Mapping)
        else {}
    )
    if not current_late_fill:
        return response_payload, False
    if _response_late_fill_is_in_flight(response_id):
        return response_payload, False
    active_branches = _normalize_late_fill_branches(current_late_fill.get('active_branches'))
    if not active_branches:
        return response_payload, False
    updated_at = _late_fill_stale_active_updated_at(
        record=record,
        response_payload=response_payload,
        late_fill=current_late_fill,
    )
    if updated_at is None:
        return response_payload, False
    age_sec = (dt.datetime.now(dt.timezone.utc) - updated_at).total_seconds()
    if age_sec < _STALE_LATE_FILL_ACTIVE_RECONCILE_AFTER_SEC:
        return response_payload, False
    if _runtime_has_active_backend_work():
        return response_payload, False

    pending_branches = _normalize_late_fill_branches(current_late_fill.get('pending_branches'))
    completed_branches = _normalize_late_fill_branches(current_late_fill.get('completed_branches'))
    cancelled_branches = _normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
    stale_by_id: dict[str, dict[str, Any]] = {}
    for branch in [*pending_branches, *active_branches]:
        branch_id = _branch_id(branch)
        if branch_id and branch_id not in stale_by_id:
            stale_by_id[branch_id] = _build_stale_active_failed_branch(dict(branch))
    if not stale_by_id:
        return response_payload, False
    failed_branch_ids = set(stale_by_id)
    failed_branches = [
        dict(branch)
        for branch in _normalize_late_fill_branches(current_late_fill.get('failed_branches'))
        if _branch_id(branch) not in failed_branch_ids
    ]
    failed_branches.extend(stale_by_id.values())
    recovery_candidates = [
        dict(item)
        for item in current_late_fill.get('recovery_candidates') or []
        if isinstance(item, Mapping) and _branch_id(item) not in failed_branch_ids
    ]
    for branch in failed_branches:
        recovery_state = branch.get('recovery_state')
        if isinstance(recovery_state, Mapping):
            recovery_candidates.append(dict(recovery_state))
    artifact_records = _build_canonical_response_artifacts(response_payload)
    has_artifact_progress = bool(response_payload.get('artifacts') or artifact_records or completed_branches)
    next_status = 'partial_failed' if has_artifact_progress else 'failed'
    failed_capabilities = _normalize_capability_list([
        _branch_capability(branch)
        for branch in failed_branches
        if _branch_capability(branch)
    ])
    completed_capabilities = _normalize_capability_list([
        _branch_capability(branch)
        for branch in completed_branches
        if _branch_capability(branch)
    ])
    cancelled_capabilities = _normalize_capability_list([
        _branch_capability(branch)
        for branch in cancelled_branches
        if _branch_capability(branch)
    ])
    next_late_fill = _build_late_fill_state(
        dict(current_late_fill),
        status=next_status,
        prior_state=current_late_fill,
        extra={
            'pending_branches': [],
            'active_branches': [],
            'completed_branches': completed_branches,
            'failed_branches': failed_branches,
            'cancelled_branches': cancelled_branches,
            'pending_capabilities': [],
            'active_capabilities': [],
            'completed_capabilities': completed_capabilities,
            'failed_capabilities': failed_capabilities,
            'cancelled_capabilities': cancelled_capabilities,
            'recovery_candidates': recovery_candidates,
            'auto_recovery_enabled': False,
            'partial_failure': next_status == 'partial_failed',
            'stale_active_reconciled': True,
            'stale_active_reconciled_at': _response_registry_now_iso(),
            'stale_active_branch_count': len(stale_by_id),
        },
    )
    updated_payload = _attach_late_fill_state(response_payload, next_late_fill)
    updated_payload['lifecycle_state'] = derive_response_lifecycle_state(
        updated_payload,
        requested_status=updated_payload.get('status'),
    )
    return updated_payload, True


def _recovery_mapping(branch: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = branch.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _late_fill_branch_recovery_action(
    branch: Mapping[str, Any],
    *,
    default: str = RECOVERY_ACTION_MANUAL_REVIEW,
) -> str:
    for source in (
        _recovery_mapping(branch, 'recovery_context'),
        _recovery_mapping(branch, 'recovery_state'),
        _recovery_mapping(branch, 'error'),
        branch,
    ):
        for key in ('suggested_action', 'recovery_action', 'repair_action'):
            value = source.get(key)
            if value not in (None, '', [], {}):
                return normalize_recovery_suggested_action(value, default=default)
    return normalize_recovery_suggested_action(default, default=RECOVERY_ACTION_MANUAL_REVIEW)


def _late_fill_branch_recovery_scope(
    branch: Mapping[str, Any],
    *,
    action: str,
) -> str:
    for source in (
        _recovery_mapping(branch, 'recovery_context'),
        _recovery_mapping(branch, 'recovery_state'),
    ):
        scope = str(source.get('retry_scope') or '').strip()
        if scope:
            return scope
    if action == RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE:
        return 'dependency_chain'
    return 'same_branch'


def _late_fill_branch_needs_external_input(branch: Mapping[str, Any]) -> bool:
    for source in (
        _recovery_mapping(branch, 'recovery_context'),
        _recovery_mapping(branch, 'recovery_state'),
        _recovery_mapping(branch, 'error'),
        branch,
    ):
        if source.get('needs_external_input') is True:
            return True
    return False


def _late_fill_branch_has_action_hint(branch: Mapping[str, Any]) -> bool:
    for source in (
        _recovery_mapping(branch, 'recovery_context'),
        _recovery_mapping(branch, 'recovery_state'),
        _recovery_mapping(branch, 'error'),
        branch,
    ):
        for key in ('suggested_action', 'recovery_action', 'repair_action'):
            if source.get(key) not in (None, '', [], {}):
                return True
    return False


def _late_fill_branch_can_join_retry_wave(branch: Any) -> bool:
    if not isinstance(branch, Mapping):
        return False
    if not _branch_id(branch) or not _branch_capability(branch):
        return False
    status = str(branch.get('status') or '').strip().lower()
    if status in {'cancelled', 'canceled', 'completed', 'failed_manual_review', 'skipped', 'superseded', 'waived'}:
        return False
    if _late_fill_branch_needs_external_input(branch):
        return False
    recovery_context = _recovery_mapping(branch, 'recovery_context')
    error = _recovery_mapping(branch, 'error')
    default_action = (
        RECOVERY_ACTION_RETRY_SAME_BRANCH
        if recovery_context.get('can_retry') is True or error.get('retryable') is True
        else RECOVERY_ACTION_MANUAL_REVIEW
    )
    action = _late_fill_branch_recovery_action(branch, default=default_action)
    if action not in _SAFE_LATE_FILL_RETRY_WAVE_ACTIONS:
        return False
    return (
        recovery_context.get('can_retry') is True
        or error.get('retryable') is True
        or _late_fill_branch_has_action_hint(branch)
    )


def _late_fill_request_phase_graph(response_payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime_payload = (
        response_payload.get('runtime')
        if isinstance(response_payload.get('runtime'), Mapping)
        else {}
    )
    graph = runtime_payload.get('request_phase_graph') if isinstance(runtime_payload.get('request_phase_graph'), Mapping) else {}
    if graph:
        return dict(graph)
    frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), Mapping)
        else {}
    )
    frame_runtime = frame.get('runtime') if isinstance(frame.get('runtime'), Mapping) else {}
    graph = (
        frame_runtime.get('request_phase_graph')
        if isinstance(frame_runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    return dict(graph) if graph else {}


def _iter_late_fill_retry_wave_graph_branches(response_payload: Mapping[str, Any]):
    graph = _late_fill_request_phase_graph(response_payload)
    if not graph:
        return
    for key in ('downstream_branches', 'phases'):
        for branch in _normalize_late_fill_branches(graph.get(key)):
            yield branch


def _iter_late_fill_retry_wave_open_check_branches(late_fill_state: Mapping[str, Any]):
    for check in late_fill_state.get('materialization_contract_open_checks') or []:
        if not isinstance(check, Mapping):
            continue
        branch = {
            'branch_id': check.get('branch_id') or check.get('phase_id'),
            'phase_id': check.get('phase_id') or check.get('branch_id'),
            'capability': check.get('capability'),
            'output_type': check.get('output_type'),
            'status': check.get('status') or 'pending',
            'repair_action': check.get('repair_action') or check.get('suggested_action'),
            'recovery_action': check.get('recovery_action'),
        }
        if check.get('recovery_context') not in (None, '', [], {}):
            branch['recovery_context'] = check.get('recovery_context')
        if check.get('recovery_state') not in (None, '', [], {}):
            branch['recovery_state'] = check.get('recovery_state')
        for normalized in _normalize_late_fill_branches([branch]):
            yield normalized


def _build_late_fill_retry_wave_branch(
    target_branch: Mapping[str, Any],
    *,
    payload_body: Optional[dict[str, Any]] = None,
    anchor_branch_id: str = '',
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    payload_body = payload_body if isinstance(payload_body, dict) else {}
    recovery_context = _recovery_mapping(target_branch, 'recovery_context')
    recovery_state = _recovery_mapping(target_branch, 'recovery_state')
    attempt = _recovery_mapping(target_branch, 'attempt')
    error = _recovery_mapping(target_branch, 'error')
    branch_id = _branch_id(target_branch)
    capability = _branch_capability(target_branch) or ''
    action = _late_fill_branch_recovery_action(
        target_branch,
        default=RECOVERY_ACTION_RETRY_SAME_BRANCH,
    )
    body_excluded = (
        payload_body.get('exclude_instance_ids')
        if isinstance(payload_body.get('exclude_instance_ids'), list)
        else payload_body.get('excludeInstanceIds')
        if isinstance(payload_body.get('excludeInstanceIds'), list)
        else []
    )
    recovery_excluded = (
        recovery_context.get('exclude_instance_ids')
        if isinstance(recovery_context.get('exclude_instance_ids'), list)
        else []
    )
    branch_excluded = (
        target_branch.get('excluded_instance_ids')
        if isinstance(target_branch.get('excluded_instance_ids'), list)
        else []
    )
    failed_instance_id = str(
        attempt.get('instance_id')
        or target_branch.get('failed_instance_id')
        or recovery_state.get('failed_instance_id')
        or ''
    ).strip()
    prior_error_code = str(
        error.get('code')
        or recovery_state.get('prior_error_code')
        or ''
    ).strip().upper()
    excluded_instance_ids: list[str] = []
    for item in [*branch_excluded, *recovery_excluded, *body_excluded, failed_instance_id]:
        token = str(item or '').strip()
        if token and token not in excluded_instance_ids:
            excluded_instance_ids.append(token)
    recovery_attempt = {
        'kind': 'ollmo.late_fill_recovery_attempt',
        'trigger': 'explicit_retry_endpoint',
        'branch_id': branch_id,
        'capability': capability,
        'preserve_intent': True,
        'auto_execute': False,
        'failed_instance_id': failed_instance_id or None,
        'prior_error_code': prior_error_code or None,
        'excluded_instance_ids': excluded_instance_ids,
        'retry_wave_anchor_branch_id': anchor_branch_id if anchor_branch_id and anchor_branch_id != branch_id else None,
    }
    recovery_attempt = {
        key: value
        for key, value in recovery_attempt.items()
        if value not in (None, '', [], {})
    }
    retry_recovery_state = {
        'kind': 'ollmo.late_fill_recovery_state',
        'status': 'attempting',
        'trigger': 'explicit_retry_endpoint',
        'branch_id': branch_id,
        'capability': capability,
        'promotion_required': False,
        'auto_execute': False,
        'preserve_intent': True,
        'retry_scope': _late_fill_branch_recovery_scope(target_branch, action=action),
        'suggested_action': action,
        'failed_instance_id': failed_instance_id or None,
        'prior_error_code': prior_error_code or None,
        'exclude_instance_ids': excluded_instance_ids,
        'retry_wave_anchor_branch_id': anchor_branch_id if anchor_branch_id and anchor_branch_id != branch_id else None,
    }
    retry_recovery_state = {
        key: value
        for key, value in retry_recovery_state.items()
        if value not in (None, '', [], {})
    }
    retry_branch = {
        key: value
        for key, value in target_branch.items()
        if key not in {'error', 'attempt', 'recovery_context', 'recovery_state', 'recovery_attempt'}
    }
    retry_branch['status'] = 'pending'
    if failed_instance_id:
        retry_branch['failed_instance_id'] = failed_instance_id
    if excluded_instance_ids:
        retry_branch['excluded_instance_ids'] = excluded_instance_ids
    retry_branch['recovery_state'] = retry_recovery_state
    retry_branch['recovery_attempt'] = recovery_attempt
    return retry_branch, retry_recovery_state, recovery_attempt, excluded_instance_ids


def _collect_late_fill_retry_wave_candidates(
    *,
    response_payload: Mapping[str, Any],
    late_fill_state: Mapping[str, Any],
    failed_branches: list[dict[str, Any]],
    completed_branches: list[dict[str, Any]],
    cancelled_branches: list[dict[str, Any]],
    selected_branch_id: str,
) -> list[dict[str, Any]]:
    closed_branch_ids = {
        _branch_id(branch)
        for branch in [*completed_branches, *cancelled_branches]
        if _branch_id(branch)
    }
    selected_and_seen = {selected_branch_id, *closed_branch_ids}
    candidates: list[dict[str, Any]] = []

    def maybe_add(branch: Any) -> None:
        if not isinstance(branch, Mapping):
            return
        normalized_items = _normalize_late_fill_branches([branch])
        if not normalized_items:
            return
        normalized = normalized_items[0]
        branch_id = _branch_id(normalized)
        if not branch_id or branch_id in selected_and_seen:
            return
        if not _late_fill_branch_can_join_retry_wave(normalized):
            return
        selected_and_seen.add(branch_id)
        candidates.append(normalized)

    for branch in failed_branches:
        maybe_add(branch)
    for branch in _normalize_late_fill_branches(late_fill_state.get('pending_branches')):
        maybe_add(branch)
    for branch in _iter_late_fill_retry_wave_graph_branches(response_payload):
        maybe_add(branch)
    for branch in _iter_late_fill_retry_wave_open_check_branches(late_fill_state):
        maybe_add(branch)
    return candidates


def _project_orphaned_late_fill_retry_attempts(
    response_id: str,
    response_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    current_late_fill = (
        response_payload.get('late_fill')
        if isinstance(response_payload.get('late_fill'), Mapping)
        else {}
    )
    if not current_late_fill:
        return response_payload, False
    if _response_late_fill_is_in_flight(response_id):
        return response_payload, False
    pending_branches = _normalize_late_fill_branches(current_late_fill.get('pending_branches'))
    active_branches = _normalize_late_fill_branches(current_late_fill.get('active_branches'))
    orphaned_failed_branches: list[dict[str, Any]] = []
    orphaned_branch_ids: set[str] = set()
    for branch in [*pending_branches, *active_branches]:
        if not _late_fill_branch_is_orphanable_retry_attempt(branch):
            continue
        failed_branch = _build_orphaned_retry_failed_branch(dict(branch))
        branch_id = _branch_id(failed_branch)
        if branch_id and branch_id in orphaned_branch_ids:
            continue
        if branch_id:
            orphaned_branch_ids.add(branch_id)
        orphaned_failed_branches.append(failed_branch)
    if not orphaned_failed_branches:
        return response_payload, False

    def _without_orphaned(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not orphaned_branch_ids:
            return list(branches)
        return [
            dict(branch)
            for branch in branches
            if _branch_id(branch) not in orphaned_branch_ids
        ]

    remaining_pending_branches = _without_orphaned(pending_branches)
    remaining_active_branches = _without_orphaned(active_branches)
    completed_branches = _normalize_late_fill_branches(current_late_fill.get('completed_branches'))
    cancelled_branches = _normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
    existing_failed_branches = [
        dict(branch)
        for branch in _normalize_late_fill_branches(current_late_fill.get('failed_branches'))
        if _branch_id(branch) not in orphaned_branch_ids
    ]
    failed_branches = [*existing_failed_branches, *orphaned_failed_branches]
    pending_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in remaining_pending_branches if _branch_capability(branch)]
    )
    active_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in remaining_active_branches if _branch_capability(branch)]
    )
    completed_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in completed_branches if _branch_capability(branch)]
    )
    failed_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in failed_branches if _branch_capability(branch)]
    )
    cancelled_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in cancelled_branches if _branch_capability(branch)]
    )
    recovery_candidates = [
        dict(item)
        for item in current_late_fill.get('recovery_candidates') or []
        if isinstance(item, Mapping) and _branch_id(item) not in orphaned_branch_ids
    ]
    for branch in orphaned_failed_branches:
        recovery_state = branch.get('recovery_state')
        if isinstance(recovery_state, Mapping):
            recovery_candidates.append(dict(recovery_state))
    late_fill_status = _late_fill_control_response_status(
        pending_branches=remaining_pending_branches,
        active_branches=remaining_active_branches,
        completed_branches=completed_branches,
        failed_branches=failed_branches,
        cancelled_branches=cancelled_branches,
    )
    artifact_gap = dict(current_late_fill)
    artifact_gap.update(
        {
            'pending_branches': remaining_pending_branches,
            'active_branches': remaining_active_branches,
            'completed_branches': completed_branches,
            'failed_branches': failed_branches,
            'cancelled_branches': cancelled_branches,
            'pending_capabilities': pending_capabilities,
            'active_capabilities': active_capabilities,
            'completed_capabilities': completed_capabilities,
            'failed_capabilities': failed_capabilities,
            'cancelled_capabilities': cancelled_capabilities,
            'recovery_candidates': recovery_candidates,
            'auto_recovery_enabled': False,
        }
    )
    retry_late_fill = _build_late_fill_state(
        artifact_gap,
        status=late_fill_status,
        prior_state=current_late_fill,
        extra={
            'pending_branches': remaining_pending_branches,
            'active_branches': remaining_active_branches,
            'completed_branches': completed_branches,
            'failed_branches': failed_branches,
            'cancelled_branches': cancelled_branches,
            'pending_capabilities': pending_capabilities,
            'active_capabilities': active_capabilities,
            'completed_capabilities': completed_capabilities,
            'failed_capabilities': failed_capabilities,
            'cancelled_capabilities': cancelled_capabilities,
            'recovery_candidates': recovery_candidates,
            'auto_recovery_enabled': False,
            'partial_failure': bool(failed_branches and completed_branches),
            'pending_branch_count': len(remaining_pending_branches),
            'active_branch_count': len(remaining_active_branches),
            'completed_branch_count': len(completed_branches),
            'failed_branch_count': len(failed_branches),
        },
    )
    updated_payload = _attach_late_fill_state(response_payload, retry_late_fill)
    updated_payload['lifecycle_state'] = derive_response_lifecycle_state(
        updated_payload,
        requested_status=updated_payload.get('status'),
    )
    return updated_payload, True


def _retry_response_late_fill_branch(response_id: str, body: Optional[dict[str, Any]] = None):
    payload_body = body if isinstance(body, dict) else {}
    normalized_response_id = _normalize_response_lookup_id(response_id)
    if not normalized_response_id:
        return jsonify({'error': 'Response id is required.'}), 400
    record = _get_response_lookup_record(normalized_response_id)
    if not record:
        return jsonify({'error': 'Response not found.'}), 404
    current_payload = dict(record.get('response_payload') or {})
    if not current_payload:
        return jsonify({'error': 'Response has no retryable payload.'}), 409
    current_payload, _orphaned_retry_projected = _project_orphaned_late_fill_retry_attempts(
        normalized_response_id,
        current_payload,
    )
    current_late_fill = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), Mapping) else {}
    failed_branch_records = _normalize_late_fill_branches(current_late_fill.get('failed_branches'))
    if not failed_branch_records:
        return jsonify({'error': 'Response has no failed late-fill branches.'}), 409
    requested_branch_id = str(payload_body.get('branch_id') or payload_body.get('branchId') or '').strip()
    target_branch: Optional[dict[str, Any]] = None
    for branch in failed_branch_records:
        if requested_branch_id and _branch_id(branch) != requested_branch_id:
            continue
        recovery_context = branch.get('recovery_context') if isinstance(branch.get('recovery_context'), Mapping) else {}
        if requested_branch_id or recovery_context.get('can_retry') is True:
            target_branch = dict(branch)
            break
    if not target_branch:
        return jsonify({'error': 'No retryable failed branch matched the request.'}), 404 if requested_branch_id else 409
    recovery_context = target_branch.get('recovery_context') if isinstance(target_branch.get('recovery_context'), Mapping) else {}
    if recovery_context.get('can_retry') is not True:
        return jsonify({'error': 'Selected failed branch is not retryable.'}), 409
    branch_id = _branch_id(target_branch)
    capability = _branch_capability(target_branch)
    if not branch_id or not capability:
        return jsonify({'error': 'Selected failed branch is missing branch identity.'}), 409
    completed_branch_records = _normalize_late_fill_branches(current_late_fill.get('completed_branches'))
    cancelled_branch_records = _normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
    retry_branch, retry_recovery_state, recovery_attempt, excluded_instance_ids = _build_late_fill_retry_wave_branch(
        target_branch,
        payload_body=payload_body,
        anchor_branch_id=branch_id,
    )
    retry_wave_candidates = _collect_late_fill_retry_wave_candidates(
        response_payload=current_payload,
        late_fill_state=current_late_fill,
        failed_branches=failed_branch_records,
        completed_branches=completed_branch_records,
        cancelled_branches=cancelled_branch_records,
        selected_branch_id=branch_id,
    )
    retry_wave_branches: list[dict[str, Any]] = []
    for candidate in retry_wave_candidates:
        wave_branch, _wave_state, _wave_attempt, _wave_excluded = _build_late_fill_retry_wave_branch(
            candidate,
            anchor_branch_id=branch_id,
        )
        retry_wave_branches.append(wave_branch)
    requeued_branch_ids = {
        _branch_id(branch)
        for branch in [retry_branch, *retry_wave_branches]
        if _branch_id(branch)
    }
    existing_pending_branches = [
        dict(branch)
        for branch in _normalize_late_fill_branches(current_late_fill.get('pending_branches'))
        if _branch_id(branch) not in requeued_branch_ids
    ]
    remaining_failed_branches = [
        dict(branch)
        for branch in failed_branch_records
        if _branch_id(branch) not in requeued_branch_ids
    ]
    pending_branches = [retry_branch, *retry_wave_branches, *existing_pending_branches]
    pending_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in pending_branches if _branch_capability(branch)]
    )
    completed_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in completed_branch_records if _branch_capability(branch)]
    )
    failed_capabilities = _normalize_capability_list(
        [_branch_capability(branch) for branch in remaining_failed_branches if _branch_capability(branch)]
    )
    artifact_gap = dict(current_late_fill)
    retry_wave_context = {
        'anchor_branch_id': branch_id,
        'scheduled_branch_ids': [
            _branch_id(branch)
            for branch in [retry_branch, *retry_wave_branches]
            if _branch_id(branch)
        ],
        'auto_requeued_branch_ids': [
            _branch_id(branch)
            for branch in retry_wave_branches
            if _branch_id(branch)
        ],
        'safe_actions': sorted(_SAFE_LATE_FILL_RETRY_WAVE_ACTIONS),
    }
    retry_context = {
        'branch_id': branch_id,
        'excluded_instance_ids': excluded_instance_ids,
        'preserve_intent': True,
        'retry_wave': retry_wave_context,
    }
    artifact_gap.update(
        {
            'expected_capability': capability,
            'active_capability': capability,
            'pending_capabilities': pending_capabilities,
            'completed_capabilities': completed_capabilities,
            'failed_capabilities': failed_capabilities,
            'pending_branches': pending_branches,
            'completed_branches': completed_branch_records,
            'failed_branches': remaining_failed_branches,
            'active_branches': [retry_branch],
            'retry_context': retry_context,
            'recovery_state': retry_recovery_state,
            'recovery_attempt': recovery_attempt,
            'auto_recovery_enabled': False,
            'retry_wave_enabled': True,
        }
    )
    retry_late_fill = _build_late_fill_state(
        artifact_gap,
        status='pending',
        prior_state=current_late_fill,
        extra={
            'pending_capabilities': pending_capabilities,
            'completed_capabilities': completed_capabilities,
            'failed_capabilities': failed_capabilities,
            'active_capability': capability,
            'active_capabilities': [capability],
            'pending_branches': pending_branches,
            'completed_branches': completed_branch_records,
            'failed_branches': remaining_failed_branches,
            'active_branches': [retry_branch],
            'retry_context': artifact_gap['retry_context'],
            'recovery_state': retry_recovery_state,
            'recovery_attempt': recovery_attempt,
            'auto_recovery_enabled': False,
            'retry_wave_enabled': True,
            'partial_failure': bool(remaining_failed_branches and completed_branch_records),
            'pending_branch_count': len(pending_branches),
            'completed_branch_count': len(completed_branch_records),
            'failed_branch_count': len(remaining_failed_branches),
        },
    )
    if not _claim_response_late_fill(normalized_response_id):
        return jsonify({'error': 'Late-fill retry is already running for this response.'}), 409
    updated_payload = _attach_late_fill_state(current_payload, retry_late_fill)
    try:
        finalized_payload = _finalize_response_frame_payload(
            updated_payload,
            request_payload=_request_payload_for_late_fill_retry(current_payload),
            persist=True,
        )
        _touch_response_lookup(
            normalized_response_id,
            status='completed',
            output_text=str(finalized_payload.get('output_text') or ''),
            response_payload=finalized_payload,
        )
        request_payload = _request_payload_for_late_fill_retry(finalized_payload)
        source_route_payload = record.get('route_payload') if isinstance(record.get('route_payload'), dict) else None
        if app.config.get('TESTING'):
            _release_response_late_fill(normalized_response_id)
        else:
            worker = threading.Thread(
                target=_complete_response_late_fill,
                kwargs={
                    'response_payload': finalized_payload,
                    'request_payload': request_payload,
                    'assistant_message': str(finalized_payload.get('output_text') or ''),
                    'artifact_gap': artifact_gap,
                    'source_route_payload': source_route_payload,
                },
                daemon=True,
            )
            worker.start()
    except Exception:
        _release_response_late_fill(normalized_response_id)
        raise
    response_projection = _project_response_payload_for_wire(finalized_payload)
    retry_envelope = _response_wire_enforce_outer_envelope_byte_ceiling({
        'status': 'retry_scheduled',
        'response': response_projection,
    },
        source_payload=finalized_payload,
        source='late_fill_retry_outer_envelope_byte_ceiling',
    )
    return jsonify(retry_envelope), 202


def _normalize_late_fill_control_action(value: Any) -> str:
    token = str(value or '').strip().lower()
    aliases = {
        'cancel': 'cancelled',
        'cancelled': 'cancelled',
        'stop': 'cancelled',
        'waive': 'waived',
        'waived': 'waived',
        'supersede': 'superseded',
        'superseded': 'superseded',
    }
    return aliases.get(token, '')


def _late_fill_control_response_status(
    *,
    pending_branches: list[dict[str, Any]],
    active_branches: list[dict[str, Any]],
    completed_branches: list[dict[str, Any]],
    failed_branches: list[dict[str, Any]],
    cancelled_branches: list[dict[str, Any]],
) -> str:
    if pending_branches or active_branches:
        return 'running'
    if failed_branches and completed_branches:
        return 'partial_failed'
    if failed_branches:
        return 'failed'
    if cancelled_branches and completed_branches:
        return 'partial_cancelled'
    if cancelled_branches:
        return 'cancelled'
    return 'completed'


def _control_response_late_fill_branch(response_id: str, body: Optional[dict[str, Any]] = None):
    normalized_response_id = _normalize_response_lookup_id(response_id)
    if not normalized_response_id:
        return jsonify({'error': 'Response id is required.'}), 400
    record = _get_response_lookup_record(normalized_response_id)
    if not record:
        return jsonify({'error': 'Response not found.'}), 404
    current_payload = dict(record.get('response_payload') or {})
    if not current_payload:
        return jsonify({'error': 'Response payload is not available for late-fill control.'}), 409
    current_late_fill = current_payload.get('late_fill') if isinstance(current_payload.get('late_fill'), Mapping) else {}
    if not current_late_fill:
        return jsonify({'error': 'Response has no late-fill state to control.'}), 409

    request_body = body if isinstance(body, dict) else {}
    action_status = _normalize_late_fill_control_action(request_body.get('action') or request_body.get('status') or 'cancel')
    if not action_status:
        return jsonify({'error': 'Unsupported late-fill control action.'}), 400
    target_branch_id = str(request_body.get('branch_id') or request_body.get('branchId') or '').strip()
    scope = str(request_body.get('scope') or ('branch' if target_branch_id else 'all')).strip().lower()
    if scope not in {'branch', 'batch', 'all'}:
        return jsonify({'error': 'Unsupported late-fill control scope.'}), 400
    reason = str(request_body.get('reason') or '').strip() or f'user requested {action_status} late-fill work'
    now_iso = _response_registry_now_iso()

    pending_branches = _normalize_late_fill_branches(current_late_fill.get('pending_branches'))
    active_branches = _normalize_late_fill_branches(current_late_fill.get('active_branches'))
    completed_branches = _normalize_late_fill_branches(current_late_fill.get('completed_branches'))
    failed_branches = _normalize_late_fill_branches(current_late_fill.get('failed_branches'))
    cancelled_branches = _normalize_late_fill_branches(current_late_fill.get('cancelled_branches'))
    branch_controls = [
        dict(item)
        for item in (current_late_fill.get('branch_controls') or [])
        if isinstance(item, Mapping)
    ]

    available_by_id: dict[str, dict[str, Any]] = {}
    for branch in [*pending_branches, *active_branches]:
        branch_id = _branch_id(branch)
        if branch_id:
            available_by_id.setdefault(branch_id, branch)
    if scope == 'branch':
        if not target_branch_id:
            return jsonify({'error': 'branch_id is required for branch-scope late-fill control.'}), 400
        target_ids = {target_branch_id}
    elif scope == 'batch':
        batch_source = active_branches or pending_branches
        target_ids = {_branch_id(branch) for branch in batch_source if _branch_id(branch)}
    else:
        target_ids = set(available_by_id)
    if not target_ids:
        return jsonify({'error': 'No pending or active late-fill branches matched the control request.'}), 404

    existing_cancelled_ids = {_branch_id(branch) for branch in cancelled_branches if _branch_id(branch)}
    controlled_ids: set[str] = set()

    def terminalize(branch: dict[str, Any]) -> dict[str, Any]:
        branch_id = _branch_id(branch)
        updated = dict(branch)
        updated['status'] = action_status
        updated['execution_gate'] = {
            'kind': 'ollmo.semantic_execution_gate',
            'scope': 'branch',
            'status': action_status,
            'action': 'skip',
            'branch_id': branch_id or None,
            'phase_id': str(updated.get('phase_id') or branch_id or '').strip() or None,
            'capability': _branch_capability(updated),
            'authority': 'user_control',
            'reason': reason,
            'source': 'late_fill_control_endpoint',
        }
        if action_status == 'cancelled':
            updated['cancel_requested'] = True
            updated['cancel_reason'] = reason
            updated['cancelled_by'] = 'user_control'
            updated['cancelled_at'] = now_iso
        elif action_status == 'waived':
            updated['waiver_reason'] = reason
        elif action_status == 'superseded':
            updated['supersession_reason'] = reason
        return {key: value for key, value in updated.items() if value not in (None, '', [], {})}

    for branch_id in sorted(target_ids):
        branch = available_by_id.get(branch_id)
        if not branch:
            continue
        controlled_ids.add(branch_id)
        terminal_branch = terminalize(branch)
        if branch_id not in existing_cancelled_ids:
            cancelled_branches.append(terminal_branch)
            existing_cancelled_ids.add(branch_id)
        branch_controls.append(
            {
                'branch_id': branch_id,
                'phase_id': str(terminal_branch.get('phase_id') or branch_id).strip() or branch_id,
                'capability': _branch_capability(terminal_branch),
                'status': action_status,
                'action': action_status,
                'scope': scope,
                'reason': reason,
                'authority': 'user_control',
                'created_at': now_iso,
            }
        )

    if not controlled_ids:
        return jsonify({'error': 'No pending or active late-fill branches matched the control request.'}), 404

    pending_branches = [branch for branch in pending_branches if _branch_id(branch) not in controlled_ids]
    active_branches = [branch for branch in active_branches if _branch_id(branch) not in controlled_ids]
    cancelled_capabilities = _normalize_capability_list([
        _branch_capability(branch)
        for branch in cancelled_branches
        if _branch_capability(branch)
    ])
    pending_capabilities = _normalize_capability_list([
        _branch_capability(branch)
        for branch in pending_branches
        if _branch_capability(branch)
    ])
    active_capabilities = _normalize_capability_list([
        _branch_capability(branch)
        for branch in active_branches
        if _branch_capability(branch)
    ])
    completed_capabilities = _normalize_capability_list(current_late_fill.get('completed_capabilities'))
    failed_capabilities = _normalize_capability_list(current_late_fill.get('failed_capabilities'))
    next_status = _late_fill_control_response_status(
        pending_branches=pending_branches,
        active_branches=active_branches,
        completed_branches=completed_branches,
        failed_branches=failed_branches,
        cancelled_branches=cancelled_branches,
    )
    next_late_fill = _build_late_fill_state(
        dict(current_late_fill),
        status=next_status,
        prior_state=current_late_fill,
        extra={
            'pending_capabilities': pending_capabilities,
            'active_capabilities': active_capabilities,
            'completed_capabilities': completed_capabilities,
            'failed_capabilities': failed_capabilities,
            'cancelled_capabilities': cancelled_capabilities,
            'pending_branches': pending_branches,
            'active_branches': active_branches,
            'completed_branches': completed_branches,
            'failed_branches': failed_branches,
            'cancelled_branches': cancelled_branches,
            'branch_controls': branch_controls,
            'execution_gate_status': 'user_control_applied',
            'control_reason': reason,
        },
    )
    updated_payload = _finalize_response_frame_payload(
        _attach_late_fill_state(current_payload, next_late_fill),
        request_payload=current_payload.get('request') if isinstance(current_payload.get('request'), dict) else {},
        persist=True,
    )
    _touch_response_lookup(
        normalized_response_id,
        status='completed',
        output_text=str(updated_payload.get('output_text') or ''),
        response_payload=updated_payload,
    )
    reason_preview, reason_ref = _response_wire_text_preview(
        reason,
        json_path='control.reason',
    )
    sorted_controlled_ids = sorted(controlled_ids)
    control_projection: dict[str, Any] = {
        'status': action_status[:512],
        'scope': scope[:512],
        'branch_ids': [
            str(branch_id)[:512]
            for branch_id in sorted_controlled_ids[:_RESPONSE_WIRE_COLLECTION_LIMIT]
        ],
        'reason': reason_preview,
    }
    if reason_ref:
        control_projection['reason_ref'] = reason_ref
    if len(sorted_controlled_ids) > _RESPONSE_WIRE_COLLECTION_LIMIT:
        control_projection['branch_ids_count'] = len(sorted_controlled_ids)
        control_projection['branch_ids_projection_truncated'] = True
        control_projection['branch_ids_ref'] = _response_wire_digest_ref(
            sorted_controlled_ids,
            json_path='control.branch_ids',
        )
    control_envelope = _response_wire_enforce_outer_envelope_byte_ceiling({
        'response': _project_response_payload_for_wire(updated_payload),
        'control': control_projection,
    },
        source_payload=updated_payload,
        source='late_fill_control_outer_envelope_byte_ceiling',
    )
    return jsonify(control_envelope)


def _input_artifact_type_from_kind(file_kind: str) -> str:
    return _INFER_SUPPORT_RUNTIME.input_artifact_type_from_kind(file_kind)


def _build_input_artifact_payload(
    saved_path: str,
    *,
    file_name: str = '',
    file_kind: str = '',
    origin: str = '',
    source_path: str = '',
) -> Optional[dict[str, Any]]:
    return _INFER_SUPPORT_RUNTIME.build_input_artifact_payload(
        saved_path,
        file_name=file_name,
        file_kind=file_kind,
        origin=origin,
        source_path=source_path,
    )


def _persist_request_input_artifacts(
    *,
    temp_path: Optional[Path] = None,
    file_name: str = '',
    file_kind: str = '',
    upload=None,
    source_path: str = '',
) -> list[dict[str, Any]]:
    return _INFER_SUPPORT_RUNTIME.persist_request_input_artifacts(
        temp_path=temp_path,
        file_name=file_name,
        file_kind=file_kind,
        upload=upload,
        source_path=source_path,
    )

def _sanitize_selected_reference_artifact(
    raw_value: Any,
    *,
    payload_source: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._sanitize_selected_reference_artifact(
        raw_value,
        payload_source=payload_source,
    )


def _sanitize_selected_reference_artifacts(
    raw_value: Any,
    *,
    payload_source: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._sanitize_selected_reference_artifacts(
        raw_value,
        payload_source=payload_source,
    )


def _extract_selected_reference_artifacts(data: Any) -> list[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._extract_selected_reference_artifacts(data)


def _extract_selected_reference_artifact(data: Any) -> Optional[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._extract_selected_reference_artifact(data)


def _inject_selected_reference_message(
    messages: list[dict[str, Any]],
    selected_reference_artifact: Any,
) -> list[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._inject_selected_reference_message(
        messages,
        selected_reference_artifact,
    )


def _extract_ghost_route_messages(data: Any, *, include_selected_reference: bool = True) -> list[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._extract_ghost_route_messages(
        data,
        include_selected_reference=include_selected_reference,
    )


def _extract_ghost_preview_route(data: Any) -> Optional[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._extract_ghost_preview_route(data)


def _normalize_ghost_preference_target(
    raw_value: Any,
    *,
    capability: Optional[str] = None,
    role: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return _REQUEST_INTAKE_RUNTIME._normalize_ghost_preference_target(
        raw_value,
        capability=capability,
        role=role,
    )


def _coerce_ghost_preferences_payload(raw_value: Any) -> dict[str, Any]:
    return _REQUEST_INTAKE_RUNTIME._coerce_ghost_preferences_payload(raw_value)


def _extract_ghost_preferences(data: Any) -> dict[str, Any]:
    return _REQUEST_INTAKE_RUNTIME._extract_ghost_preferences(data)


def _default_persisted_ghost_preferences_state() -> dict[str, Any]:
    return {
        'version': 1,
        'updated_at': None,
        'preferences': {},
        'expanded': False,
    }


def _normalize_persisted_ghost_preferences_state(raw_value: Any) -> dict[str, Any]:
    source = raw_value if isinstance(raw_value, dict) else {}
    preferences_source = (
        source.get('preferences')
        if isinstance(source.get('preferences'), dict)
        else source.get('ghost_preferences')
    )
    normalized = _default_persisted_ghost_preferences_state()
    normalized['preferences'] = _coerce_ghost_preferences_payload(preferences_source)
    normalized['expanded'] = _parse_bool(source.get('expanded'), default=False)
    updated_at = str(source.get('updated_at') or source.get('updatedAt') or '').strip()
    normalized['updated_at'] = updated_at or None
    version = source.get('version')
    if isinstance(version, int) and version > 0:
        normalized['version'] = version
    return normalized


def load_persisted_ghost_preferences(path: Optional[Any] = None) -> dict[str, Any]:
    target = Path(path) if path else GHOST_PREFERENCES_PATH
    if not target.exists():
        return _default_persisted_ghost_preferences_state()
    try:
        raw = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return _default_persisted_ghost_preferences_state()
    return _normalize_persisted_ghost_preferences_state(raw)


def persist_ghost_preferences(raw_value: Any, path: Optional[Any] = None) -> dict[str, Any]:
    target = Path(path) if path else GHOST_PREFERENCES_PATH
    normalized = _normalize_persisted_ghost_preferences_state(raw_value)
    normalized['updated_at'] = _response_registry_now_iso()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return normalized


def _codex_timeout_seconds() -> int:
    raw_value = str(os.environ.get(CODEX_TIMEOUT_ENV) or '').strip()
    try:
        parsed = int(raw_value) if raw_value else DEFAULT_CODEX_TIMEOUT_SEC
    except ValueError:
        parsed = DEFAULT_CODEX_TIMEOUT_SEC
    return max(10, min(parsed, 3600))


def _codex_external_target_payload(*, force_refresh: bool = False) -> dict[str, Any]:
    return build_codex_external_target(
        load_persisted_ghost_preferences(),
        force_refresh=force_refresh,
    )


def _external_targets_payload(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    return [_codex_external_target_payload(force_refresh=force_refresh)]


def _validate_codex_external_request(
    payload: Any,
    *,
    files_enabled: Optional[bool] = None,
    **kwargs,
):
    effective_files_enabled = files_enabled
    if effective_files_enabled is None:
        effective_files_enabled = (
            _codex_external_target_payload().get('files_enabled') is True
        )
    return validate_codex_text_request(
        payload,
        files_enabled=bool(effective_files_enabled),
        **kwargs,
    )


def _execute_codex_external_text(prompt: str, inputs=None):
    return execute_codex_request(
        prompt,
        inputs=inputs or (),
        timeout_seconds=float(_codex_timeout_seconds()),
    )


def _apply_ghost_preferences_to_route_context(
    route_context: dict[str, Any],
    ghost_preferences: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(route_context, dict):
        return route_context
    if not isinstance(ghost_preferences, dict) or not ghost_preferences:
        return route_context
    updated = dict(route_context)
    runtime = dict(updated.get('runtime') or {})
    runtime['ghost_preferences'] = ghost_preferences
    updated['runtime'] = runtime
    return updated


def _candidate_matches_ghost_preference(candidate: dict[str, Any], preference: Optional[dict[str, Any]]) -> bool:
    return _GHOST_ROUTE_RUNTIME._candidate_matches_ghost_preference(candidate, preference)


def _ghost_execution_preference_applies_to_capability(
    target: Optional[dict[str, Any]],
    capability: Optional[str],
) -> bool:
    return _GHOST_ROUTE_RUNTIME._ghost_execution_preference_applies_to_capability(target, capability)


def _pick_ghost_preference_instance(
    candidates: list[dict[str, Any]],
    route_context: dict[str, Any],
    *,
    route_selected_instance_id: Optional[str] = None,
    requested_capability: Optional[str] = None,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    return _GHOST_ROUTE_RUNTIME.pick_ghost_preference_instance(
        candidates,
        route_context,
        route_selected_instance_id=route_selected_instance_id,
        requested_capability=requested_capability,
    )


def _should_ignore_preview_route_for_live_heuristic(
    preview_route: Optional[dict[str, Any]],
    heuristic_route: Optional[dict[str, Any]],
) -> bool:
    return _GHOST_ROUTE_RUNTIME.should_ignore_preview_route_for_live_heuristic(
        preview_route,
        heuristic_route,
    )


def _artifact_type_for_capability(capability: Optional[str]) -> Optional[str]:
    return _RESPONSE_SEMANTICS_RUNTIME.artifact_type_for_capability(capability)


def _response_payload_has_artifact_type(
    payload: Optional[dict[str, Any]],
    artifact_type: Optional[str],
) -> bool:
    return _RESPONSE_SEMANTICS_RUNTIME.response_payload_has_artifact_type(
        payload,
        artifact_type,
    )


def _artifact_gap_is_already_fulfilled(
    artifact_gap: Optional[dict[str, Any]],
    payload: Optional[dict[str, Any]],
) -> bool:
    return _RESPONSE_SEMANTICS_RUNTIME.artifact_gap_is_already_fulfilled(
        artifact_gap,
        payload,
    )


def _normalize_capability_list(values: Any) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        capability = normalize_capability(value)
        if capability and capability not in normalized:
            normalized.append(capability)
    return normalized


def _normalize_late_fill_branches(values: Any) -> list[dict[str, Any]]:
    def normalized_excluded_instance_ids(
        source: Mapping[str, Any],
        *keys: str,
    ) -> list[str]:
        normalized: list[str] = []
        for key in keys:
            raw_values = source.get(key)
            if not isinstance(raw_values, list):
                continue
            for item in raw_values:
                token = str(item or '').strip()
                if token and token not in normalized:
                    normalized.append(token)
        return normalized

    branches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_value in enumerate(values or [], start=1):
        raw_branch = raw_value if isinstance(raw_value, Mapping) else {'capability': raw_value}
        capability = normalize_capability(raw_branch.get('capability'))
        if not capability:
            continue
        branch_id = str(
            raw_branch.get('branch_id')
            or raw_branch.get('phase_id')
            or f'branch-{index}-{capability}'
        ).strip()
        if not branch_id or branch_id in seen_ids:
            continue
        seen_ids.add(branch_id)
        branch = {
            'branch_id': branch_id,
            'phase_id': str(raw_branch.get('phase_id') or branch_id).strip() or branch_id,
            'capability': capability,
            'output_type': str(
                raw_branch.get('output_type')
                or _artifact_type_for_capability(capability)
                or ''
            ).strip() or None,
            'queue_index': _coerce_positive_int(raw_branch.get('queue_index')) or index,
        }
        obligation_id = str(raw_branch.get('obligation_id') or '').strip()
        if obligation_id:
            branch['obligation_id'] = obligation_id
        depends_on = [
            str(item).strip()
            for item in (raw_branch.get('depends_on') or [])
            if str(item).strip()
        ]
        if depends_on:
            branch['depends_on'] = depends_on
        for key in (
            'artifact_prompt',
            'artifact_prompt_source',
            'batch_prompts_source',
            'batch_prompt_source_phase_id',
            'batch_prompt_expected_count',
            'content_payload',
            'content_payload_source',
            'phase_summary',
            'stage_direction',
            'semantic_intent',
            'objective',
            'deliverable',
            'rationale',
            'advisory_role',
            'decision_notes',
            'evidence_requirements',
            'reconsideration_triggers',
            'semantic_review_criteria',
            'promotion_suggestions',
            'waiver_candidates',
            'repair_candidates',
            'supersession_candidates',
            'learning_hint_refs',
            'decision_contract_repair_candidates',
            'decision_contract_semantic_review_candidates',
            'decision_contract_supersession_candidates',
            'decision_contract_block_resolution_signals',
            'decision_contract_active_reconsideration_decisions',
            'decision_contract_semantic_quality_contracts',
            'block_resolution_reflex',
            'block_resolution_signal',
            'block_resolution_action',
            'block_resolution_policy',
            'reconsideration_reflex',
            'active_reconsideration_review',
            'active_reconsideration_decision',
            'active_reconsideration_action',
            'active_reconsideration_review_type',
            'semantic_quality_review',
            'semantic_quality_contract',
            'semantic_quality_status',
            'semantic_quality_review_id',
            'recursive_cycle_review',
            'recursive_cycle_state',
            'semantic_decision_review',
            'semantic_decision_proposal',
            'semantic_decision_action',
            'semantic_decision_confidence',
            'semantic_decision_reason',
            'semantic_review_verdict',
            'semantic_review_verdict_status',
            'semantic_review_recommended_transition',
            'branch_semantic_review',
            'branch_semantic_review_branch_id',
            'branch_semantic_review_phase_id',
            'branch_semantic_review_status',
            'branch_semantic_review_reason',
            'branch_semantic_review_source_branch_id',
            'branch_semantic_review_source_phase_id',
            'decision_contract_semantic_decision_proposals',
            'controlled_attention_review',
            'controlled_attention_frame',
            'controlled_attention_frame_id',
            'controlled_attention_scope',
            'controlled_attention_priority',
            'controlled_attention_question',
            'controlled_attention_allowed_transitions',
            'decision_contract_controlled_attention_frames',
            'global_semantic_closure_review',
            'global_semantic_closure_proposal',
            'global_semantic_closure_status',
            'global_semantic_closure_reason',
            'global_semantic_closure_confidence',
            'surface_state',
            'supersession_review_required',
            'supersession_review_authority',
            'execution_contract',
            'workload_task_ref',
            'output_obligation_ref',
            'output_contract',
            'accepted_proposals',
            'input_refs',
            'review_criteria',
            'requires_artifact',
            'text_artifact_extension',
            'text_artifact_source_name',
            'text_artifact_source',
            'text_artifact_target_path',
            'artifact_request',
            'repair_action',
            'recovery_action',
            'repair_action_reason',
            'repair_contract',
            'repair_contract_id',
            'repair_contract_status',
            'repair_execution_policy',
            'repair_promotion_source',
            'contract_state',
            'promotion_source',
            'retry_wave_anchor_branch_id',
            'reconsideration_rebuild',
            'blocked_by_dependency_input',
            'blocked_by_branch_contract',
            'execution_gate',
            'execution_gate_status',
            'execution_gate_reason',
            'cancel_requested',
            'cancel_reason',
            'cancelled_by',
            'cancelled_at',
            'waiver_reason',
        ):
            value = raw_branch.get(key)
            if value not in (None, '', [], {}):
                branch[key] = value
        for source_key, target_key in (
            ('suggested_action', 'suggested_action'),
            ('suggestedAction', 'suggested_action'),
            ('repair_work_policy', 'repair_work_policy'),
            ('repairWorkPolicy', 'repair_work_policy'),
            ('repair_work_available', 'repair_work_available'),
            ('repairWorkAvailable', 'repair_work_available'),
            ('materialization_blocked', 'materialization_blocked'),
            ('materializationBlocked', 'materialization_blocked'),
            ('needs_external_input', 'needs_external_input'),
            ('needsExternalInput', 'needs_external_input'),
            ('blocked_scope', 'blocked_scope'),
            ('blockedScope', 'blocked_scope'),
            ('blocked_prerequisite', 'blocked_prerequisite'),
            ('blockedPrerequisite', 'blocked_prerequisite'),
            ('auto_execute', 'auto_execute'),
            ('autoExecute', 'auto_execute'),
            ('auto_executable_repair_retry_count', 'auto_executable_repair_retry_count'),
            ('autoExecutableRepairRetryCount', 'auto_executable_repair_retry_count'),
            ('auto_executable_repair_max_attempts', 'auto_executable_repair_max_attempts'),
            ('autoExecutableRepairMaxAttempts', 'auto_executable_repair_max_attempts'),
            ('repair_auto_execute_max_attempts', 'repair_auto_execute_max_attempts'),
            ('repairAutoExecuteMaxAttempts', 'repair_auto_execute_max_attempts'),
            ('max_auto_execute_attempts', 'max_auto_execute_attempts'),
            ('maxAutoExecuteAttempts', 'max_auto_execute_attempts'),
            ('execution_policy', 'execution_policy'),
            ('executionPolicy', 'execution_policy'),
            ('repair_execution_policy', 'repair_execution_policy'),
            ('repairExecutionPolicy', 'repair_execution_policy'),
        ):
            value = raw_branch.get(source_key)
            if value not in (None, '', [], {}):
                branch[target_key] = value
        if capability == 'text_to_speech':
            recovery_policy_id = str(
                raw_branch.get('recovery_policy_id')
                or raw_branch.get('recoveryPolicyId')
                or ''
            ).strip()
            if recovery_policy_id:
                branch['recovery_policy_id'] = recovery_policy_id
        batch_prompts = [
            str(item).strip()
            for item in (raw_branch.get('batch_prompts') or [])
        ] if isinstance(raw_branch.get('batch_prompts'), list) else []
        if any(batch_prompts):
            branch['batch_prompts'] = batch_prompts
        status = str(raw_branch.get('status') or '').strip().lower()
        if status:
            branch['status'] = status
        failed_instance_id = str(
            raw_branch.get('failed_instance_id')
            or raw_branch.get('failedInstanceId')
            or ''
        ).strip()
        if failed_instance_id:
            branch['failed_instance_id'] = failed_instance_id
        excluded_instance_ids = normalized_excluded_instance_ids(
            raw_branch,
            'excluded_instance_ids',
            'excludedInstanceIds',
            'exclude_instance_ids',
            'excludeInstanceIds',
        )
        if excluded_instance_ids:
            branch['excluded_instance_ids'] = excluded_instance_ids
        error = raw_branch.get('error') if isinstance(raw_branch.get('error'), Mapping) else None
        if error:
            branch_error: dict[str, Any] = {}
            for source_key, target_key in (
                ('code', 'code'),
                ('reason_code', 'reason_code'),
                ('defect_code', 'defect_code'),
                ('message', 'message'),
                ('stage', 'stage'),
                ('exception_type', 'exception_type'),
                ('exceptionType', 'exception_type'),
                ('repair_action', 'repair_action'),
                ('recovery_action', 'recovery_action'),
                ('suggested_action', 'suggested_action'),
                ('blocked_scope', 'blocked_scope'),
                ('blocked_prerequisite', 'blocked_prerequisite'),
                ('repair_work_policy', 'repair_work_policy'),
            ):
                value = error.get(source_key)
                text = str(value or '').strip()
                if text:
                    branch_error[target_key] = text
            retryable = error.get('retryable')
            if isinstance(retryable, bool):
                branch_error['retryable'] = retryable
            for source_key, target_key in (
                ('materialization_blocked', 'materialization_blocked'),
                ('repair_work_available', 'repair_work_available'),
                ('needs_external_input', 'needs_external_input'),
            ):
                value = error.get(source_key)
                if isinstance(value, bool):
                    branch_error[target_key] = value
            status_code = error.get('status_code') if error.get('status_code') is not None else error.get('statusCode')
            if status_code not in (None, ''):
                try:
                    branch_error['status_code'] = int(status_code)
                except (TypeError, ValueError):
                    pass
            failed_dependency_ids = [
                str(item or '').strip()
                for item in (error.get('failed_dependency_ids') or [])
                if str(item or '').strip()
            ] if isinstance(error.get('failed_dependency_ids'), list) else []
            if failed_dependency_ids:
                branch_error['failed_dependency_ids'] = list(
                    dict.fromkeys(failed_dependency_ids)
                )
            if capability == 'text_to_speech':
                defect_codes = [
                    str(item or '').strip()
                    for item in (error.get('defect_codes') or [])
                    if str(item or '').strip()
                ] if isinstance(error.get('defect_codes'), list) else []
                if defect_codes:
                    branch_error['defect_codes'] = list(
                        dict.fromkeys(defect_codes)
                    )
            semantic_evidence = error.get('semantic_evidence')
            if isinstance(semantic_evidence, Mapping) and semantic_evidence:
                branch_error['semantic_evidence'] = dict(semantic_evidence)
            audio_integrity_evidence = error.get('audio_integrity_evidence')
            if (
                isinstance(audio_integrity_evidence, Mapping)
                and audio_integrity_evidence
            ):
                branch_error['audio_integrity_evidence'] = dict(
                    audio_integrity_evidence
                )
            for key in ('tts_generation_budget', 'tts_sampling_profile'):
                value = error.get(key)
                if isinstance(value, Mapping) and value:
                    branch_error[key] = dict(value)
            diagnostic_artifact = error.get('diagnostic_artifact')
            if isinstance(diagnostic_artifact, Mapping) and diagnostic_artifact:
                branch_error['diagnostic_artifact'] = {
                    key: value
                    for key, value in diagnostic_artifact.items()
                    if key in {'type', 'path', 'artifact_ref', 'status'}
                    and value not in (None, '', [], {})
                }
            if branch_error:
                branch['error'] = branch_error
        attempt = raw_branch.get('attempt') if isinstance(raw_branch.get('attempt'), Mapping) else None
        if attempt:
            branch_attempt: dict[str, Any] = {}
            for source_key, target_key in (
                ('stage', 'stage'),
                ('capability', 'capability'),
                ('backend', 'backend'),
                ('instance_id', 'instance_id'),
                ('instanceId', 'instance_id'),
                ('model', 'model'),
                ('route_source', 'route_source'),
                ('routeSource', 'route_source'),
                ('route_reason', 'route_reason'),
                ('routeReason', 'route_reason'),
            ):
                value = attempt.get(source_key)
                text = str(value or '').strip()
                if text:
                    branch_attempt[target_key] = normalize_capability(text) if target_key == 'capability' else text
            if branch_attempt:
                branch['attempt'] = branch_attempt
        recovery_context = (
            raw_branch.get('recovery_context')
            if isinstance(raw_branch.get('recovery_context'), Mapping)
            else raw_branch.get('recoveryContext')
            if isinstance(raw_branch.get('recoveryContext'), Mapping)
            else None
        )
        if recovery_context:
            branch_recovery: dict[str, Any] = {}
            for source_key, target_key in (
                ('retry_scope', 'retry_scope'),
                ('retryScope', 'retry_scope'),
                ('suggested_action', 'suggested_action'),
                ('suggestedAction', 'suggested_action'),
                ('blocked_scope', 'blocked_scope'),
                ('blockedScope', 'blocked_scope'),
                ('blocked_prerequisite', 'blocked_prerequisite'),
                ('blockedPrerequisite', 'blocked_prerequisite'),
                ('repair_work_policy', 'repair_work_policy'),
                ('repairWorkPolicy', 'repair_work_policy'),
            ):
                value = recovery_context.get(source_key)
                text = str(value or '').strip()
                if text:
                    branch_recovery[target_key] = text
            if capability == 'text_to_speech':
                for source_key, target_key in (
                    ('error_code', 'error_code'),
                    ('errorCode', 'error_code'),
                    ('reason_code', 'reason_code'),
                    ('reasonCode', 'reason_code'),
                ):
                    value = recovery_context.get(source_key)
                    text = str(value or '').strip()
                    if text:
                        branch_recovery[target_key] = text
            for source_key, target_key in (
                ('can_retry', 'can_retry'),
                ('canRetry', 'can_retry'),
                ('preserve_intent', 'preserve_intent'),
                ('preserveIntent', 'preserve_intent'),
                ('repair_required', 'repair_required'),
                ('repairRequired', 'repair_required'),
                ('blocked_by_dependency_input', 'blocked_by_dependency_input'),
                ('blockedByDependencyInput', 'blocked_by_dependency_input'),
                ('blocked_by_branch_contract', 'blocked_by_branch_contract'),
                ('blockedByBranchContract', 'blocked_by_branch_contract'),
                ('blocked_by_underplanned_promoted_obligations', 'blocked_by_underplanned_promoted_obligations'),
                ('blockedByUnderplannedPromotedObligations', 'blocked_by_underplanned_promoted_obligations'),
                ('materialization_blocked', 'materialization_blocked'),
                ('materializationBlocked', 'materialization_blocked'),
                ('repair_work_available', 'repair_work_available'),
                ('repairWorkAvailable', 'repair_work_available'),
                ('needs_external_input', 'needs_external_input'),
                ('needsExternalInput', 'needs_external_input'),
            ):
                value = recovery_context.get(source_key)
                if isinstance(value, bool):
                    branch_recovery[target_key] = value
            exclude_instance_ids = normalized_excluded_instance_ids(
                recovery_context,
                'exclude_instance_ids',
                'excludeInstanceIds',
                'excluded_instance_ids',
                'excludedInstanceIds',
            )
            if exclude_instance_ids:
                branch_recovery['exclude_instance_ids'] = exclude_instance_ids
            if capability == 'text_to_speech':
                recovery_defect_codes = [
                    str(item or '').strip()
                    for item in (recovery_context.get('defect_codes') or [])
                    if str(item or '').strip()
                ] if isinstance(recovery_context.get('defect_codes'), list) else []
                if recovery_defect_codes:
                    branch_recovery['defect_codes'] = list(
                        dict.fromkeys(recovery_defect_codes)
                    )
                recovery_integrity = recovery_context.get(
                    'audio_integrity_evidence'
                )
                if (
                    isinstance(recovery_integrity, Mapping)
                    and recovery_integrity
                ):
                    branch_recovery['audio_integrity_evidence'] = dict(
                        recovery_integrity
                    )
            if branch_recovery:
                branch['recovery_context'] = branch_recovery
        recovery_state = (
            raw_branch.get('recovery_state')
            if isinstance(raw_branch.get('recovery_state'), Mapping)
            else raw_branch.get('recoveryState')
            if isinstance(raw_branch.get('recoveryState'), Mapping)
            else None
        )
        if recovery_state:
            branch_recovery_state: dict[str, Any] = {}
            for source_key, target_key in (
                ('kind', 'kind'),
                ('status', 'status'),
                ('trigger', 'trigger'),
                ('branch_id', 'branch_id'),
                ('branchId', 'branch_id'),
                ('capability', 'capability'),
                ('retry_scope', 'retry_scope'),
                ('retryScope', 'retry_scope'),
                ('suggested_action', 'suggested_action'),
                ('suggestedAction', 'suggested_action'),
                ('failed_instance_id', 'failed_instance_id'),
                ('failedInstanceId', 'failed_instance_id'),
                ('prior_error_code', 'prior_error_code'),
                ('priorErrorCode', 'prior_error_code'),
                ('retry_wave_anchor_branch_id', 'retry_wave_anchor_branch_id'),
                ('retryWaveAnchorBranchId', 'retry_wave_anchor_branch_id'),
                ('blocked_scope', 'blocked_scope'),
                ('blockedScope', 'blocked_scope'),
                ('blocked_prerequisite', 'blocked_prerequisite'),
                ('blockedPrerequisite', 'blocked_prerequisite'),
                ('repair_work_policy', 'repair_work_policy'),
                ('repairWorkPolicy', 'repair_work_policy'),
            ):
                value = recovery_state.get(source_key)
                text = str(value or '').strip()
                if text:
                    branch_recovery_state[target_key] = normalize_capability(text) if target_key == 'capability' else text
            if capability == 'text_to_speech':
                for source_key, target_key in (
                    ('recovery_policy_id', 'recovery_policy_id'),
                    ('recoveryPolicyId', 'recovery_policy_id'),
                    ('prior_reason_code', 'prior_reason_code'),
                    ('priorReasonCode', 'prior_reason_code'),
                ):
                    value = recovery_state.get(source_key)
                    text = str(value or '').strip()
                    if text:
                        branch_recovery_state[target_key] = text
            for source_key, target_key in (
                ('promotion_required', 'promotion_required'),
                ('promotionRequired', 'promotion_required'),
                ('auto_execute', 'auto_execute'),
                ('autoExecute', 'auto_execute'),
                ('preserve_intent', 'preserve_intent'),
                ('preserveIntent', 'preserve_intent'),
                ('repair_required', 'repair_required'),
                ('repairRequired', 'repair_required'),
                ('blocked_by_dependency_input', 'blocked_by_dependency_input'),
                ('blockedByDependencyInput', 'blocked_by_dependency_input'),
                ('blocked_by_branch_contract', 'blocked_by_branch_contract'),
                ('blockedByBranchContract', 'blocked_by_branch_contract'),
                ('blocked_by_underplanned_promoted_obligations', 'blocked_by_underplanned_promoted_obligations'),
                ('blockedByUnderplannedPromotedObligations', 'blocked_by_underplanned_promoted_obligations'),
                ('materialization_blocked', 'materialization_blocked'),
                ('materializationBlocked', 'materialization_blocked'),
                ('repair_work_available', 'repair_work_available'),
                ('repairWorkAvailable', 'repair_work_available'),
                ('needs_external_input', 'needs_external_input'),
                ('needsExternalInput', 'needs_external_input'),
            ):
                value = recovery_state.get(source_key)
                if isinstance(value, bool):
                    branch_recovery_state[target_key] = value
            recovery_exclude_instance_ids = normalized_excluded_instance_ids(
                recovery_state,
                'exclude_instance_ids',
                'excludeInstanceIds',
                'excluded_instance_ids',
                'excludedInstanceIds',
            )
            if recovery_exclude_instance_ids:
                branch_recovery_state['exclude_instance_ids'] = recovery_exclude_instance_ids
            if capability == 'text_to_speech':
                for source_key, target_key in (
                    ('attempt_number', 'attempt_number'),
                    ('attemptNumber', 'attempt_number'),
                    ('maximum_attempts', 'maximum_attempts'),
                    ('maximumAttempts', 'maximum_attempts'),
                ):
                    try:
                        value = int(recovery_state.get(source_key))
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        branch_recovery_state[target_key] = value
                prior_defect_codes = [
                    str(item or '').strip()
                    for item in (recovery_state.get('prior_defect_codes') or [])
                    if str(item or '').strip()
                ] if isinstance(recovery_state.get('prior_defect_codes'), list) else []
                if prior_defect_codes:
                    branch_recovery_state['prior_defect_codes'] = list(
                        dict.fromkeys(prior_defect_codes)
                    )
            if branch_recovery_state:
                branch['recovery_state'] = branch_recovery_state
        recovery_attempt = (
            raw_branch.get('recovery_attempt')
            if isinstance(raw_branch.get('recovery_attempt'), Mapping)
            else raw_branch.get('recoveryAttempt')
            if isinstance(raw_branch.get('recoveryAttempt'), Mapping)
            else None
        )
        if recovery_attempt:
            branch_recovery_attempt: dict[str, Any] = {}
            for source_key, target_key in (
                ('kind', 'kind'),
                ('trigger', 'trigger'),
                ('branch_id', 'branch_id'),
                ('branchId', 'branch_id'),
                ('capability', 'capability'),
                ('failed_instance_id', 'failed_instance_id'),
                ('failedInstanceId', 'failed_instance_id'),
                ('prior_error_code', 'prior_error_code'),
                ('priorErrorCode', 'prior_error_code'),
                ('retry_wave_anchor_branch_id', 'retry_wave_anchor_branch_id'),
                ('retryWaveAnchorBranchId', 'retry_wave_anchor_branch_id'),
            ):
                value = recovery_attempt.get(source_key)
                text = str(value or '').strip()
                if text:
                    branch_recovery_attempt[target_key] = normalize_capability(text) if target_key == 'capability' else text
            if capability == 'text_to_speech':
                for source_key, target_key in (
                    ('recovery_policy_id', 'recovery_policy_id'),
                    ('recoveryPolicyId', 'recovery_policy_id'),
                    ('prior_reason_code', 'prior_reason_code'),
                    ('priorReasonCode', 'prior_reason_code'),
                ):
                    value = recovery_attempt.get(source_key)
                    text = str(value or '').strip()
                    if text:
                        branch_recovery_attempt[target_key] = text
            for source_key, target_key in (
                ('auto_execute', 'auto_execute'),
                ('autoExecute', 'auto_execute'),
                ('preserve_intent', 'preserve_intent'),
                ('preserveIntent', 'preserve_intent'),
            ):
                value = recovery_attempt.get(source_key)
                if isinstance(value, bool):
                    branch_recovery_attempt[target_key] = value
            attempt_excluded_instance_ids = normalized_excluded_instance_ids(
                recovery_attempt,
                'excluded_instance_ids',
                'excludedInstanceIds',
                'exclude_instance_ids',
                'excludeInstanceIds',
            )
            if attempt_excluded_instance_ids:
                branch_recovery_attempt['excluded_instance_ids'] = attempt_excluded_instance_ids
            if capability == 'text_to_speech':
                for source_key, target_key in (
                    ('attempt_number', 'attempt_number'),
                    ('attemptNumber', 'attempt_number'),
                    ('maximum_attempts', 'maximum_attempts'),
                    ('maximumAttempts', 'maximum_attempts'),
                ):
                    try:
                        value = int(recovery_attempt.get(source_key))
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        branch_recovery_attempt[target_key] = value
                prior_attempt_defect_codes = [
                    str(item or '').strip()
                    for item in (recovery_attempt.get('prior_defect_codes') or [])
                    if str(item or '').strip()
                ] if isinstance(recovery_attempt.get('prior_defect_codes'), list) else []
                if prior_attempt_defect_codes:
                    branch_recovery_attempt['prior_defect_codes'] = list(
                        dict.fromkeys(prior_attempt_defect_codes)
                    )
                prior_audio_integrity = recovery_attempt.get(
                    'prior_audio_integrity_evidence'
                )
                if (
                    isinstance(prior_audio_integrity, Mapping)
                    and prior_audio_integrity
                ):
                    branch_recovery_attempt[
                        'prior_audio_integrity_evidence'
                    ] = dict(prior_audio_integrity)
            if branch_recovery_attempt:
                branch['recovery_attempt'] = branch_recovery_attempt
        canonical_excluded_instance_ids: list[str] = []
        for source, key in (
            (branch, 'excluded_instance_ids'),
            (branch.get('recovery_context'), 'exclude_instance_ids'),
            (branch.get('recovery_state'), 'exclude_instance_ids'),
            (branch.get('recovery_attempt'), 'excluded_instance_ids'),
        ):
            if not isinstance(source, Mapping):
                continue
            for item in source.get(key) or []:
                token = str(item or '').strip()
                if token and token not in canonical_excluded_instance_ids:
                    canonical_excluded_instance_ids.append(token)
        if canonical_excluded_instance_ids:
            branch['excluded_instance_ids'] = canonical_excluded_instance_ids
        branches.append(branch)
    return branches


def _branch_id(branch: Any) -> str:
    if not isinstance(branch, Mapping):
        return ''
    return str(branch.get('branch_id') or branch.get('phase_id') or '').strip()


def _branch_capability(branch: Any) -> Optional[str]:
    if not isinstance(branch, Mapping):
        return None
    return normalize_capability(branch.get('capability'))


def _late_fill_capability_counts(branches: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for branch in branches:
        capability = _branch_capability(branch)
        if not capability:
            continue
        counts[capability] = counts.get(capability, 0) + 1
    return counts


def _build_pending_late_fill_branches(
    *,
    artifact_gap: Optional[dict[str, Any]] = None,
    late_fill_state: Optional[dict[str, Any]] = None,
    pending_branches: Optional[list[dict[str, Any]]] = None,
    pending_capabilities: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    normalized_state = late_fill_state if isinstance(late_fill_state, dict) else {}
    normalized_gap = artifact_gap if isinstance(artifact_gap, dict) else {}
    explicit_branches = _normalize_late_fill_branches(
        pending_branches
        or normalized_state.get('pending_branches')
        or normalized_gap.get('pending_branches')
    )
    if explicit_branches:
        return explicit_branches
    branches: list[dict[str, Any]] = []
    capability_occurrences: dict[str, int] = {}
    for capability in pending_capabilities or []:
        normalized_capability = normalize_capability(capability)
        if not normalized_capability:
            continue
        capability_occurrences[normalized_capability] = capability_occurrences.get(normalized_capability, 0) + 1
        branch_suffix = capability_occurrences[normalized_capability]
        branches.append(
            {
                'branch_id': f'branch-{normalized_capability}-{branch_suffix}',
                'phase_id': f'branch-{normalized_capability}-{branch_suffix}',
                'capability': normalized_capability,
                'output_type': _artifact_type_for_capability(normalized_capability),
                'queue_index': len(branches) + 1,
            }
        )
    return branches


def _lookup_replay_pending_output_slots_from_late_fill(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized_payload = payload if isinstance(payload, Mapping) else {}
    if isinstance(normalized_payload.get('output_slots'), list) and normalized_payload.get('output_slots'):
        return []
    late_fill = (
        normalized_payload.get('late_fill')
        if isinstance(normalized_payload.get('late_fill'), Mapping)
        else normalized_payload.get('lateFill')
        if isinstance(normalized_payload.get('lateFill'), Mapping)
        else {}
    )
    if not late_fill:
        runtime = normalized_payload.get('runtime') if isinstance(normalized_payload.get('runtime'), Mapping) else {}
        late_fill = runtime.get('late_fill') if isinstance(runtime.get('late_fill'), Mapping) else {}
    if not late_fill:
        return []
    late_fill_status = str(late_fill.get('status') or '').strip().lower()
    if late_fill_status not in {'pending', 'queued', 'running', 'scheduled', 'repair_needed', 'partial_failed', 'failed'}:
        return []
    expected_capability = normalize_capability(late_fill.get('expected_capability'))
    missing_artifact_type = str(late_fill.get('missing_artifact_type') or '').strip().lower()
    output_type = missing_artifact_type or _artifact_type_for_capability(expected_capability)
    if not expected_capability and not output_type:
        return []
    if not expected_capability and output_type:
        expected_capability = 'artifact'
    output_type = output_type or 'artifact'
    blocked_late_fill = late_fill_status in {'failed', 'partial_failed', 'repair_needed'}
    pending_slot_status = 'blocked' if blocked_late_fill else 'pending'
    pending_slot_lifecycle = 'blocked_output' if blocked_late_fill else 'deferred_output'

    slots: list[dict[str, Any]] = []
    output_text = str(
        normalized_payload.get('content_payload')
        or normalized_payload.get('output_text')
        or ''
    ).strip()
    if output_text:
        slots.append(
            {
                'slot_id': 'output-phase-1',
                'branch_id': 'phase-1',
                'phase_id': 'phase-1',
                'type': 'text',
                'status': 'fulfilled',
                'lifecycle': 'materialized_output',
            }
        )
    slots.append(
        {
            'slot_id': f'output-phase-{len(slots) + 1}',
            'branch_id': f'branch-{expected_capability}-1',
            'phase_id': f'branch-{expected_capability}-1',
            'type': output_type,
            'status': pending_slot_status,
            'lifecycle': pending_slot_lifecycle,
            'follow_up_capability': expected_capability,
        }
    )
    if blocked_late_fill:
        reason = str(
            late_fill.get('error')
            or late_fill.get('error_message')
            or late_fill.get('reason')
            or 'late fill branch failed'
        ).strip()
        if reason:
            slots[-1]['blocked_reason'] = reason
    return slots


def _attach_lookup_replay_pending_output_slots(payload: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(payload or {})
    late_fill = (
        updated.get('late_fill')
        if isinstance(updated.get('late_fill'), Mapping)
        else updated.get('lateFill')
        if isinstance(updated.get('lateFill'), Mapping)
        else {}
    )
    if not late_fill:
        runtime = updated.get('runtime') if isinstance(updated.get('runtime'), Mapping) else {}
        late_fill = runtime.get('late_fill') if isinstance(runtime.get('late_fill'), Mapping) else {}
    late_fill_status = str((late_fill or {}).get('status') or '').strip().lower()
    existing_slots = updated.get('output_slots') if isinstance(updated.get('output_slots'), list) else []
    if existing_slots and late_fill_status in {'failed', 'partial_failed', 'repair_needed'}:
        blocked_reason = str(
            (late_fill or {}).get('error')
            or (late_fill or {}).get('error_message')
            or (late_fill or {}).get('reason')
            or 'late fill branch failed'
        ).strip()
        blocked_slots: list[dict[str, Any]] = []
        for raw_slot in existing_slots:
            if not isinstance(raw_slot, Mapping):
                continue
            slot = dict(raw_slot)
            slot_type = str(slot.get('type') or '').strip().lower()
            slot_status = str(slot.get('status') or '').strip().lower()
            if slot_type not in {'text', 'document'} and slot_status in {'pending', 'queued', 'running', 'scheduled'}:
                slot['status'] = 'blocked'
                slot['lifecycle'] = 'blocked_output'
                if blocked_reason:
                    slot['blocked_reason'] = blocked_reason
            blocked_slots.append(slot)
        if blocked_slots:
            updated['output_slots'] = blocked_slots
            updated['output_branches'] = _build_public_output_branches_from_slots(blocked_slots)
            updated['_lookup_replay_response_frame_required'] = True
        return updated
    slots = _lookup_replay_pending_output_slots_from_late_fill(updated)
    if not slots:
        return updated
    updated['output_slots'] = slots
    updated['output_branches'] = _build_public_output_branches_from_slots(slots)
    updated['_lookup_replay_response_frame_required'] = True
    return updated


def _lookup_slot_tokens(value: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    if not isinstance(value, Mapping):
        return tokens
    for key in ('slot_id', 'branch_id', 'phase_id'):
        token = str(value.get(key) or '').strip()
        if token:
            tokens.add(token)
    return tokens


def _lookup_slot_matches_record(slot: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    if not isinstance(slot, Mapping) or not isinstance(record, Mapping):
        return False
    if _lookup_slot_tokens(slot).intersection(_lookup_slot_tokens(record)):
        return True
    slot_type = str(slot.get('type') or '').strip().lower()
    record_type = str(record.get('output_type') or record.get('type') or '').strip().lower()
    slot_capability = normalize_capability(slot.get('follow_up_capability') or slot.get('capability'))
    record_capability = normalize_capability(record.get('capability') or record.get('expected_capability'))
    if slot_capability and record_capability and slot_capability == record_capability:
        return True
    if slot_type and record_type and slot_type == record_type:
        return True
    if slot_type == 'audio' and record_capability == 'text_to_speech':
        return True
    if slot_type == 'image' and record_capability == 'image_generation':
        return True
    return False


def _lookup_record_artifact_path(record: Mapping[str, Any]) -> str:
    if not isinstance(record, Mapping):
        return ''
    for key in (
        'path',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'target_path',
    ):
        value = str(record.get(key) or '').strip()
        if value:
            return value
    return ''


def _lookup_artifact_for_slot_record(
    *,
    slot: Mapping[str, Any],
    record: Mapping[str, Any],
    artifacts: list[Any],
    allow_type_fallback: bool = False,
) -> dict[str, Any]:
    record_path = _lookup_record_artifact_path(record)
    record_ref = str(record.get('artifact_ref') or record.get('ref') or '').strip()
    slot_type = str(slot.get('type') or '').strip().lower()
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_ref = str(raw_artifact.get('artifact_ref') or raw_artifact.get('ref') or '').strip()
        artifact_path = str(raw_artifact.get('path') or raw_artifact.get('target_path') or '').strip()
        artifact_type = str(raw_artifact.get('type') or raw_artifact.get('kind') or '').strip().lower()
        if record_ref and artifact_ref and record_ref == artifact_ref:
            return dict(raw_artifact)
        if record_path and artifact_path and record_path == artifact_path:
            return dict(raw_artifact)
        if (
            allow_type_fallback
            and not record_path
            and not record_ref
            and slot_type
            and artifact_type
            and slot_type == artifact_type
        ):
            return dict(raw_artifact)
    return {}


def _lookup_error_ref_from_failed_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    if isinstance(record.get('error_ref'), Mapping):
        source = record['error_ref']
    else:
        source = record.get('error') if isinstance(record.get('error'), Mapping) else {}
    branch_id = str(
        source.get('branch_id')
        or record.get('branch_id')
        or ''
    ).strip()
    payload = {
        'branch_id': branch_id or None,
        'code': str(source.get('code') or record.get('code') or '').strip() or None,
        'stage': str(source.get('stage') or record.get('stage') or '').strip() or None,
    }
    return {key: value for key, value in payload.items() if value not in (None, '', [], {})}


def _reconcile_lookup_output_slots_with_late_fill_truth(payload: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(payload or {})
    output_slots = updated.get('output_slots') if isinstance(updated.get('output_slots'), list) else []
    if not output_slots:
        return updated
    late_fill = (
        updated.get('late_fill')
        if isinstance(updated.get('late_fill'), Mapping)
        else updated.get('lateFill')
        if isinstance(updated.get('lateFill'), Mapping)
        else {}
    )
    if not late_fill:
        runtime = updated.get('runtime') if isinstance(updated.get('runtime'), Mapping) else {}
        late_fill = runtime.get('late_fill') if isinstance(runtime.get('late_fill'), Mapping) else {}
    if not late_fill:
        return updated
    artifacts = updated.get('artifacts') if isinstance(updated.get('artifacts'), list) else []
    outputs = updated.get('outputs') if isinstance(updated.get('outputs'), list) else []
    fill_results = late_fill.get('fill_results') if isinstance(late_fill.get('fill_results'), list) else []
    completed_branches = late_fill.get('completed_branches') if isinstance(late_fill.get('completed_branches'), list) else []
    failed_branches = late_fill.get('failed_branches') if isinstance(late_fill.get('failed_branches'), list) else []
    completed_records = [item for item in [*fill_results, *completed_branches] if isinstance(item, Mapping)]
    failed_records = [item for item in failed_branches if isinstance(item, Mapping)]
    if not completed_records and not failed_records:
        return updated

    non_text_slot_type_counts: dict[str, int] = {}
    for raw_slot in output_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot_type = str(raw_slot.get('type') or '').strip().lower()
        if slot_type and slot_type not in {'text', 'document'}:
            non_text_slot_type_counts[slot_type] = non_text_slot_type_counts.get(slot_type, 0) + 1
    artifact_type_counts: dict[str, int] = {}
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_type = str(
            raw_artifact.get('type') or raw_artifact.get('kind') or ''
        ).strip().lower()
        if artifact_type:
            artifact_type_counts[artifact_type] = artifact_type_counts.get(artifact_type, 0) + 1

    reconciled_slots: list[dict[str, Any]] = []
    changed = False
    for raw_slot in output_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot = dict(raw_slot)
        slot_type = str(slot.get('type') or '').strip().lower()
        slot_tokens = _lookup_slot_tokens(slot)
        completed_record = next(
            (
                record
                for record in completed_records
                if slot_tokens and slot_tokens.intersection(_lookup_slot_tokens(record))
            ),
            None,
        )
        if completed_record is None and non_text_slot_type_counts.get(slot_type, 0) <= 1:
            completed_record = next(
                (record for record in completed_records if _lookup_slot_matches_record(slot, record)),
                None,
            )
        if completed_record is not None and slot_type not in {'text', 'document'}:
            matching_output = next(
                (
                    output
                    for output in outputs
                    if isinstance(output, Mapping)
                    and slot_tokens
                    and slot_tokens.intersection(_lookup_slot_tokens(output))
                ),
                {},
            )
            artifact_identity_record = (
                completed_record
                if _lookup_record_artifact_path(completed_record)
                or str(
                    completed_record.get('artifact_ref')
                    or completed_record.get('ref')
                    or ''
                ).strip()
                else matching_output
                if isinstance(matching_output, Mapping) and matching_output
                else slot
            )
            artifact = _lookup_artifact_for_slot_record(
                slot=slot,
                record=artifact_identity_record,
                artifacts=artifacts,
                allow_type_fallback=(
                    non_text_slot_type_counts.get(slot_type, 0) <= 1
                    and artifact_type_counts.get(slot_type, 0) <= 1
                ),
            )
            path = (
                _lookup_record_artifact_path(completed_record)
                or _lookup_record_artifact_path(matching_output)
                or _lookup_record_artifact_path(slot)
                or _lookup_record_artifact_path(artifact)
            )
            slot['status'] = 'fulfilled'
            slot['lifecycle'] = 'materialized_output'
            if path:
                slot['path'] = path
                if slot_type == 'image':
                    slot['saved_image_path'] = path
                elif slot_type == 'audio':
                    slot['saved_audio_path'] = path
                elif slot_type in {'text', 'document'}:
                    slot['saved_text_path'] = path
            artifact_ref = str(
                completed_record.get('artifact_ref')
                or completed_record.get('ref')
                or matching_output.get('artifact_ref')
                or matching_output.get('ref')
                or slot.get('artifact_ref')
                or slot.get('ref')
                or artifact.get('artifact_ref')
                or artifact.get('ref')
                or ''
            ).strip()
            if artifact_ref:
                slot['artifact_ref'] = artifact_ref
            for key in ('lang_code', 'lang_code_source', 'response_format', 'output_format'):
                value = completed_record.get(key)
                if value not in (None, '', [], {}):
                    slot[key] = value
            changed = True
        failed_record = next(
            (
                record
                for record in failed_records
                if slot_tokens and slot_tokens.intersection(_lookup_slot_tokens(record))
            ),
            None,
        )
        if failed_record is None and non_text_slot_type_counts.get(slot_type, 0) <= 1:
            failed_record = next(
                (record for record in failed_records if _lookup_slot_matches_record(slot, record)),
                None,
            )
        if failed_record is not None:
            slot_status = str(slot.get('status') or '').strip().lower()
            if slot_status in {'pending', 'queued', 'running', 'scheduled'}:
                slot['status'] = 'blocked'
                slot['lifecycle'] = 'blocked_output'
            reason = str(
                failed_record.get('blocked_reason')
                or failed_record.get('error_message')
                or (
                    failed_record.get('error', {}).get('message')
                    if isinstance(failed_record.get('error'), Mapping)
                    else ''
                )
                or late_fill.get('error')
                or late_fill.get('error_message')
                or 'late fill branch failed'
            ).strip()
            if reason and slot.get('blocked_reason') in (None, '', [], {}):
                slot['blocked_reason'] = reason
            error_ref = _lookup_error_ref_from_failed_record(failed_record)
            if error_ref:
                slot['error_ref'] = error_ref
            for key in ('recovery_context', 'recovery_state'):
                value = failed_record.get(key)
                if isinstance(value, Mapping) and value:
                    slot[key] = dict(value)
            changed = True
        reconciled_slots.append(slot)
    if changed and reconciled_slots:
        updated['output_slots'] = reconciled_slots
        updated['output_branches'] = _build_public_output_branches_from_slots(reconciled_slots)
        updated['_lookup_replay_response_frame_required'] = True
    return updated


def _extract_pending_deferred_branches(
    *,
    route_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.extract_pending_deferred_branches(
        route_payload=route_payload,
        artifact_payload=artifact_payload,
    )


def _extract_pending_deferred_capabilities(
    *,
    route_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
) -> list[str]:
    return _RESPONSE_SEMANTICS_RUNTIME.extract_pending_deferred_capabilities(
        route_payload=route_payload,
        artifact_payload=artifact_payload,
    )


def _semantic_payload_for_capability(
    payload: Optional[dict[str, Any]],
    *,
    capability: Optional[str],
) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME.semantic_payload_for_capability(
        payload,
        capability=capability,
    )


def _build_planner_deferred_follow_up_gap_spec(
    output_text: str,
    *,
    route_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_planner_deferred_follow_up_gap_spec(
        output_text,
        route_payload=route_payload,
        artifact_payload=artifact_payload,
    )


def _extract_expected_non_chat_capability_from_route(route_payload: Optional[dict[str, Any]]) -> Optional[str]:
    return _RESPONSE_SEMANTICS_RUNTIME._extract_expected_non_chat_capability_from_route(route_payload)


def _extract_expected_non_chat_capability(
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    return _RESPONSE_SEMANTICS_RUNTIME._extract_expected_non_chat_capability(
        route_payload=route_payload,
        request_payload=request_payload,
    )


def _build_artifact_completion_gap_spec(
    output_text: str,
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_artifact_completion_gap_spec(
        output_text,
        route_payload=route_payload,
        request_payload=request_payload,
        artifact_payload=artifact_payload,
    )


def _build_pre_freeze_closure_review_gap(
    output_text: str,
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
    artifact_gap: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_pre_freeze_closure_review_gap(
        output_text,
        route_payload=route_payload,
        request_payload=request_payload,
        artifact_payload=artifact_payload,
        artifact_gap=artifact_gap,
    )


def _build_graph_closure_review(
    output_text: str,
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
    artifact_payload: Optional[dict[str, Any]] = None,
    artifact_gap: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_graph_closure_review(
        output_text,
        route_payload=route_payload,
        request_payload=request_payload,
        artifact_payload=artifact_payload,
        artifact_gap=artifact_gap,
    )


def _truth_gate_response_output_claims(
    payload: Optional[dict[str, Any]],
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME.truth_gate_response_output_claims(
        payload,
        route_payload=route_payload,
        request_payload=request_payload,
    )


def _build_late_fill_state(
    artifact_gap: dict[str, Any],
    *,
    status: str,
    prior_state: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_late_fill_state(
        artifact_gap,
        status=status,
        prior_state=prior_state,
        extra=extra,
    )


def _extract_semantic_materializer_prompt(
    payload: Any,
    *,
    capability: Optional[str],
) -> Optional[str]:
    return _RESPONSE_SEMANTICS_RUNTIME.extract_semantic_materializer_prompt(
        payload,
        capability=capability,
    )


_GRAPH_PATCH_SUCCESSOR_EXECUTION_TERMINAL_STATUSES = {
    'completed',
    'failed',
    'cancelled',
    'blocked',
}


def _graph_patch_successor_execution_status_rank(value: Any) -> int:
    status = str(value or '').strip().lower()
    if status in _GRAPH_PATCH_SUCCESSOR_EXECUTION_TERMINAL_STATUSES:
        return 3
    if status in {'running', 'in_progress', 'late_fill_running'}:
        return 2
    if status in {'queued', 'pending', 'scheduled', 'candidate', 'late_fill_pending'}:
        return 1
    return 0


def _merge_graph_patch_successor_execution_truth(
    existing: Any,
    incoming: Any,
) -> dict[str, Any]:
    """Merge one execution key without allowing lifecycle regression."""

    current = dict(existing) if isinstance(existing, Mapping) else {}
    proposed = dict(incoming) if isinstance(incoming, Mapping) else {}
    if not current:
        return proposed
    if not proposed:
        return current
    current_status = str(current.get('status') or '').strip().lower()
    proposed_status = str(proposed.get('status') or '').strip().lower()
    current_rank = _graph_patch_successor_execution_status_rank(current_status)
    proposed_rank = _graph_patch_successor_execution_status_rank(proposed_status)

    # A persisted terminal result is immutable for one stable execution key.
    # A conflicting terminal projection or an older active projection is stale.
    if current_status in _GRAPH_PATCH_SUCCESSOR_EXECUTION_TERMINAL_STATUSES:
        if proposed_status != current_status:
            return current
        merged = {**current, **proposed}
        merged['status'] = current_status
        return merged
    if proposed_status in _GRAPH_PATCH_SUCCESSOR_EXECUTION_TERMINAL_STATUSES:
        return {**current, **proposed}
    if proposed_rank < current_rank:
        return current
    return {**current, **proposed}


def _graph_patch_successor_execution_truth_for_key(
    runtime: Any,
    execution_key: str,
    *,
    response_late_fill: Any = None,
) -> dict[str, Any]:
    """Read the strongest existing projection for one execution key."""

    if not isinstance(runtime, Mapping) or not execution_key:
        return {}
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
    candidates: list[Any] = []
    candidates.extend(graph.get('successor_reopen_executions') or [])
    candidates.extend(
        record.get('execution')
        for record in (graph.get('successor_reopen_requests') or [])
        if isinstance(record, Mapping)
    )
    candidates.append(diagnostics.get('graph_patch_successor_reopen_execution'))
    diagnostic_request = diagnostics.get('graph_patch_successor_reopen_request')
    if isinstance(diagnostic_request, Mapping):
        candidates.append(diagnostic_request.get('execution'))
    candidates.extend(
        record.get('execution')
        for record in (diagnostics.get('graph_patch_successor_reopen_requests') or [])
        if isinstance(record, Mapping)
    )
    runtime_late_fill = runtime.get('late_fill')
    if isinstance(runtime_late_fill, Mapping):
        candidates.append(runtime_late_fill.get('successor_reopen_execution'))
    if isinstance(response_late_fill, Mapping):
        candidates.append(response_late_fill.get('successor_reopen_execution'))

    strongest: dict[str, Any] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get('successor_execution_key') or '').strip() != execution_key:
            continue
        strongest = _merge_graph_patch_successor_execution_truth(strongest, candidate)
    return strongest


def _merge_graph_patch_successor_late_fill_truth(
    existing: Any,
    incoming: Any,
    effective_execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the Late Fill envelope aligned with monotone successor execution."""

    current = dict(existing) if isinstance(existing, Mapping) else {}
    proposed = dict(incoming) if isinstance(incoming, Mapping) else {}
    execution_key = str(effective_execution.get('successor_execution_key') or '').strip()
    current_execution = (
        current.get('successor_reopen_execution')
        if isinstance(current.get('successor_reopen_execution'), Mapping)
        else {}
    )
    proposed_execution = (
        proposed.get('successor_reopen_execution')
        if isinstance(proposed.get('successor_reopen_execution'), Mapping)
        else {}
    )
    current_matches = bool(
        execution_key
        and str(current_execution.get('successor_execution_key') or '').strip()
        == execution_key
    )
    current_execution_status = str(current_execution.get('status') or '').strip().lower()
    proposed_execution_status = str(proposed_execution.get('status') or '').strip().lower()
    preserve_current_envelope = bool(
        current_matches
        and (
            (
                current_execution_status
                in _GRAPH_PATCH_SUCCESSOR_EXECUTION_TERMINAL_STATUSES
                and proposed_execution_status != current_execution_status
            )
            or (
                _graph_patch_successor_execution_status_rank(current_execution_status)
                > _graph_patch_successor_execution_status_rank(proposed_execution_status)
            )
        )
    )
    merged = current if preserve_current_envelope else proposed
    merged = dict(merged)
    merged['successor_reopen_execution'] = dict(effective_execution)

    effective_status = str(effective_execution.get('status') or '').strip().lower()
    if effective_status in _GRAPH_PATCH_SUCCESSOR_EXECUTION_TERMINAL_STATUSES:
        late_fill_status = str(
            effective_execution.get('late_fill_status') or effective_status
        ).strip().lower()
        if late_fill_status in {
            'pending',
            'queued',
            'scheduled',
            'running',
            'in_progress',
            'late_fill_pending',
            'late_fill_running',
        }:
            late_fill_status = effective_status
        merged['status'] = late_fill_status or effective_status
        merged['pending_branches'] = []
        merged['active_branches'] = []
        merged['pending_capabilities'] = []
        merged['active_capabilities'] = []
    return merged


def _attach_late_fill_state(
    response_payload: dict[str, Any],
    late_fill_state: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(response_payload or {})
    normalized_state = dict(late_fill_state or {})
    runtime = dict(updated.get('runtime') or {}) if isinstance(updated.get('runtime'), dict) else {}
    incoming_execution = (
        normalized_state.get('successor_reopen_execution')
        if isinstance(normalized_state.get('successor_reopen_execution'), Mapping)
        else {}
    )
    execution_key = str(incoming_execution.get('successor_execution_key') or '').strip()
    if execution_key:
        existing_execution = _graph_patch_successor_execution_truth_for_key(
            runtime,
            execution_key,
            response_late_fill=updated.get('late_fill'),
        )
        effective_execution = _merge_graph_patch_successor_execution_truth(
            existing_execution,
            incoming_execution,
        )
        existing_late_fill = (
            updated.get('late_fill')
            if isinstance(updated.get('late_fill'), Mapping)
            else runtime.get('late_fill')
            if isinstance(runtime.get('late_fill'), Mapping)
            else {}
        )
        normalized_state = _merge_graph_patch_successor_late_fill_truth(
            existing_late_fill,
            normalized_state,
            effective_execution,
        )
    updated['late_fill'] = normalized_state
    runtime['late_fill'] = normalized_state
    runtime = _sync_graph_patch_successor_execution_truth(runtime, normalized_state)
    updated['runtime'] = runtime
    return updated


def _sync_graph_patch_successor_execution_truth(
    runtime: dict[str, Any],
    late_fill_state: dict[str, Any],
) -> dict[str, Any]:
    """Keep every durable graph-patch successor execution projection monotonic."""

    execution = (
        dict(late_fill_state.get('successor_reopen_execution'))
        if isinstance(late_fill_state.get('successor_reopen_execution'), Mapping)
        else {}
    )
    execution_key = str(execution.get('successor_execution_key') or '').strip()
    if not execution_key:
        return runtime
    updated_runtime = copy.deepcopy(runtime)
    graph = (
        dict(updated_runtime.get('request_phase_graph') or {})
        if isinstance(updated_runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    existing_execution = _graph_patch_successor_execution_truth_for_key(
        updated_runtime,
        execution_key,
    )
    execution = _merge_graph_patch_successor_execution_truth(existing_execution, execution)

    executions: list[dict[str, Any]] = []
    execution_matched = False
    for record in graph.get('successor_reopen_executions') or []:
        if not isinstance(record, Mapping):
            continue
        record_payload = dict(record)
        if str(record_payload.get('successor_execution_key') or '').strip() == execution_key:
            record_payload = _merge_graph_patch_successor_execution_truth(
                record_payload,
                execution,
            )
            execution_matched = True
        executions.append(record_payload)
    if not execution_matched:
        executions.append(dict(execution))
    graph['successor_reopen_executions'] = executions

    requests: list[dict[str, Any]] = []
    for record in graph.get('successor_reopen_requests') or []:
        if not isinstance(record, Mapping):
            continue
        record_payload = dict(record)
        record_execution = (
            record_payload.get('execution')
            if isinstance(record_payload.get('execution'), Mapping)
            else {}
        )
        record_key = str(
            record_payload.get('successor_execution_key')
            or record_execution.get('successor_execution_key')
            or ''
        ).strip()
        if record_key == execution_key:
            merged_request_execution = _merge_graph_patch_successor_execution_truth(
                record_execution,
                execution,
            )
            record_payload['execution'] = merged_request_execution
            request_execution_status = str(
                merged_request_execution.get('status') or ''
            ).strip().lower()
            record_payload['execution_status'] = request_execution_status or None
            if request_execution_status in {'failed', 'cancelled'}:
                record_payload['status'] = request_execution_status
                record_payload['runtime_effect'] = f'successor_late_fill_{request_execution_status}'
                blocked_reasons = [
                    str(item).strip()
                    for item in (record_payload.get('blocked_reasons') or [])
                    if str(item or '').strip()
                ]
                reason = f'successor_execution_{request_execution_status}'
                if reason not in blocked_reasons:
                    blocked_reasons.append(reason)
                record_payload['blocked_reasons'] = blocked_reasons
        requests.append(record_payload)
    graph['successor_reopen_requests'] = requests
    updated_runtime['request_phase_graph'] = graph

    diagnostics = (
        dict(updated_runtime.get('developer_diagnostics') or {})
        if isinstance(updated_runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    diagnostic_execution = (
        diagnostics.get('graph_patch_successor_reopen_execution')
        if isinstance(diagnostics.get('graph_patch_successor_reopen_execution'), Mapping)
        else {}
    )
    if str(diagnostic_execution.get('successor_execution_key') or '').strip() in {
        '',
        execution_key,
    }:
        diagnostics['graph_patch_successor_reopen_execution'] = (
            _merge_graph_patch_successor_execution_truth(
                diagnostic_execution,
                execution,
            )
        )
    diagnostic_request = (
        diagnostics.get('graph_patch_successor_reopen_request')
        if isinstance(diagnostics.get('graph_patch_successor_reopen_request'), Mapping)
        else {}
    )
    diagnostic_request_execution = (
        diagnostic_request.get('execution')
        if isinstance(diagnostic_request.get('execution'), Mapping)
        else {}
    )
    diagnostic_request_key = str(
        diagnostic_request.get('successor_execution_key')
        or diagnostic_request_execution.get('successor_execution_key')
        or ''
    ).strip()
    if diagnostic_request and diagnostic_request_key == execution_key:
        matching_request = next(
            (
                dict(item)
                for item in requests
                if str(item.get('successor_execution_key') or '').strip() == execution_key
            ),
            {},
        )
        diagnostics['graph_patch_successor_reopen_request'] = (
            matching_request
            or {
                **dict(diagnostic_request),
                'execution': _merge_graph_patch_successor_execution_truth(
                    diagnostic_request_execution,
                    execution,
                ),
            }
        )
    diagnostic_requests = []
    for record in diagnostics.get('graph_patch_successor_reopen_requests') or []:
        if not isinstance(record, Mapping):
            continue
        record_payload = dict(record)
        record_execution = (
            record_payload.get('execution')
            if isinstance(record_payload.get('execution'), Mapping)
            else {}
        )
        record_key = str(
            record_payload.get('successor_execution_key')
            or record_execution.get('successor_execution_key')
            or ''
        ).strip()
        if record_key == execution_key:
            record_payload = next(
                (
                    dict(item)
                    for item in requests
                    if str(item.get('successor_execution_key') or '').strip() == execution_key
                ),
                record_payload,
            )
        diagnostic_requests.append(record_payload)
    if diagnostic_requests:
        diagnostics['graph_patch_successor_reopen_requests'] = diagnostic_requests
    updated_runtime['developer_diagnostics'] = diagnostics
    return updated_runtime


def _selected_reference_matches_capability(
    selected_reference_artifact: Optional[dict[str, Any]],
    capability: Optional[str],
    *,
    instance: Optional[dict[str, Any]] = None,
) -> bool:
    return _RESPONSE_SEMANTICS_RUNTIME.selected_reference_matches_capability(
        selected_reference_artifact,
        capability,
        instance=instance,
    )


def _select_matching_selected_reference_artifact(
    selected_reference_artifacts: Any,
    capability: Optional[str],
    *,
    instance: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.select_matching_selected_reference_artifact(
        selected_reference_artifacts,
        capability,
        instance=instance,
    )


def _should_attach_selected_reference_file_context(
    *,
    prompt: Any,
    capability: Optional[str],
    selected_reference_artifact: Optional[dict[str, Any]],
) -> bool:
    return _RESPONSE_SEMANTICS_RUNTIME.should_attach_selected_reference_file_context(
        prompt=prompt,
        capability=capability,
        selected_reference_artifact=selected_reference_artifact,
    )


def _build_selected_reference_prompt_prefix(
    selected_reference_artifacts: Any,
    capability: Optional[str],
) -> str:
    return _RESPONSE_SEMANTICS_RUNTIME.build_selected_reference_prompt_prefix(
        selected_reference_artifacts,
        capability,
    )


def _apply_selected_reference_prompt_prefix(
    prompt: Any,
    selected_reference_artifacts: Any,
    capability: Optional[str],
) -> str:
    return _RESPONSE_SEMANTICS_RUNTIME.apply_selected_reference_prompt_prefix(
        prompt,
        selected_reference_artifacts,
        capability,
    )


def _apply_selected_reference_artifact_to_route_context(
    route_context: dict[str, Any],
    selected_reference_artifact: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.apply_selected_reference_artifact_to_route_context(
        route_context,
        selected_reference_artifact,
    )


def _apply_selected_reference_artifacts_to_route_context(
    route_context: dict[str, Any],
    selected_reference_artifacts: Any,
) -> dict[str, Any]:
    return _GHOST_ROUTE_RUNTIME.apply_selected_reference_artifacts_to_route_context(
        route_context,
        selected_reference_artifacts,
    )


def _route_context_reference_artifacts(route_context: Any) -> list[dict[str, Any]]:
    return _GHOST_ROUTE_RUNTIME._route_context_reference_artifacts(route_context)


def _find_route_artifact_ref(recent_artifacts: Any, artifact_path: Any) -> Optional[str]:
    return _GHOST_ROUTE_RUNTIME._find_route_artifact_ref(recent_artifacts, artifact_path)


def _resolve_route_artifact_ref(
    route_context: Any,
    *,
    artifact_path: Any,
    preview_payload: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    return _GHOST_ROUTE_RUNTIME.resolve_route_artifact_ref(
        route_context,
        artifact_path=artifact_path,
        preview_payload=preview_payload,
    )


def _inject_selected_reference_into_chat_messages(
    messages: list[dict[str, Any]],
    selected_reference_artifact: Any,
) -> list[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.inject_selected_reference_into_chat_messages(
        messages,
        selected_reference_artifact,
    )


def _current_phase_payload(phase_graph: Optional[dict[str, Any]]) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME._current_phase_payload(phase_graph)


def _resolve_prepare_phase_contract(
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.resolve_prepare_phase_contract(
        route_payload=route_payload,
        request_payload=request_payload,
    )


def _format_prepare_phase_capability_label(capability: str) -> str:
    return _RESPONSE_SEMANTICS_RUNTIME._format_prepare_phase_capability_label(capability)


def _build_prepare_phase_system_message(prepare_contract: Optional[dict[str, Any]]) -> Optional[dict[str, str]]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_prepare_phase_system_message(prepare_contract)


def _build_external_prepare_phase_bounded_task(
    *,
    prepare_contract: Optional[dict[str, Any]] = None,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
) -> str:
    return _RESPONSE_SEMANTICS_RUNTIME.build_external_prepare_phase_bounded_task(
        prepare_contract=prepare_contract,
        route_payload=route_payload,
        request_payload=request_payload,
    )


def _inject_ghost_runtime_policy_into_chat_messages(
    messages: list[dict[str, Any]],
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.inject_ghost_runtime_policy_into_chat_messages(
        messages,
        route_payload=route_payload,
        request_payload=request_payload,
    )


def _inject_prepare_phase_contract_into_chat_messages(
    messages: list[dict[str, Any]],
    *,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.inject_prepare_phase_contract_into_chat_messages(
        messages,
        route_payload=route_payload,
        request_payload=request_payload,
    )


def _extract_semantic_phase_payload_from_payload(payload: Any) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME._extract_semantic_phase_payload_from_payload(payload)


def _build_response_semantic_phase_payload(
    *,
    output_text: str,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
    source_payload: Optional[dict[str, Any]] = None,
    capability: Optional[str] = None,
) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME.build_response_semantic_phase_payload(
        output_text=output_text,
        route_payload=route_payload,
        request_payload=request_payload,
        source_payload=source_payload,
        capability=capability,
    )


def _attach_response_semantic_phase_payload(
    payload: Optional[dict[str, Any]],
    *,
    output_text: str,
    route_payload: Optional[dict[str, Any]] = None,
    request_payload: Optional[dict[str, Any]] = None,
    capability: Optional[str] = None,
) -> dict[str, Any]:
    return _RESPONSE_SEMANTICS_RUNTIME.attach_response_semantic_phase_payload(
        payload,
        output_text=output_text,
        route_payload=route_payload,
        request_payload=request_payload,
        capability=capability,
    )


def _plan_compound_execution_payload(
    payload: dict[str, Any],
    *,
    route_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _RESPONSE_SEMANTICS_RUNTIME.plan_compound_execution_payload(
        payload,
        route_info=route_info,
    )


def _resolve_ghost_auto_route(
    data: Any,
    *,
    upload=None,
    excluded_instance_ids: Optional[list[str]] = None,
    retry_failure: Optional[dict[str, Any]] = None,
    preview_mode: bool = False,
    refresh_runtime_status: bool = False,
    compute_semantics: bool = False,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    return _GHOST_ROUTE_RUNTIME.resolve_ghost_auto_route(
        data,
        upload=upload,
        excluded_instance_ids=excluded_instance_ids,
        retry_failure=retry_failure,
        preview_mode=preview_mode,
        refresh_runtime_status=refresh_runtime_status,
        compute_semantics=compute_semantics,
    )


@app.route('/api/ghost_route_preview', methods=['POST'])
def api_ghost_route_preview():
    raw_payload = request.get_json(silent=True) or {}
    observer_payload = raw_payload if isinstance(raw_payload, Mapping) else None
    refresh_requested = _observer_refresh_requested(observer_payload)
    semantic_compute_policy = _observer_semantic_compute_policy(observer_payload)
    semantic_compute_requested = bool(semantic_compute_policy.get('enabled'))
    compute_semantics_source = str(semantic_compute_policy.get('source') or 'policy_default')
    compute_semantics_policy = str(semantic_compute_policy.get('policy') or 'off')
    compute_semantics_false_override = str(semantic_compute_policy.get('false_override') or 'allow')
    data = _normalize_request_payload(raw_payload)
    route_info, resolution_error = _resolve_ghost_auto_route(
        data,
        upload=None,
        preview_mode=True,
        refresh_runtime_status=refresh_requested,
        compute_semantics=semantic_compute_requested,
    )
    route_runtime = (route_info or {}).get('route_runtime') if isinstance(route_info, dict) else {}
    semantic_compute = (
        route_runtime.get('semantic_compute')
        if isinstance(route_runtime, Mapping) and isinstance(route_runtime.get('semantic_compute'), Mapping)
        else {}
    )
    semantic_compute_performed = bool(semantic_compute.get('performed'))
    runtime_truth = _runtime_truth_metadata(
        refresh_requested=refresh_requested,
        semantic_compute_requested=semantic_compute_requested,
        semantic_compute_performed=semantic_compute_performed,
        compute_semantics_source=compute_semantics_source,
        compute_semantics_policy=compute_semantics_policy,
        compute_semantics_false_override=compute_semantics_false_override,
    )
    if resolution_error:
        status_code = 404 if 'was not found' in resolution_error or 'No running instance' in resolution_error else 400
        return jsonify({'error': resolution_error, 'runtime_truth': runtime_truth}), status_code
    instance = (route_info or {}).get('instance') if isinstance(route_info, dict) else None
    _effective_data, enriched_route_info, _planner_meta, _control_hints = _prepare_effective_request_data(
        data,
        route_info=route_info,
        instance=instance if isinstance(instance, dict) else None,
        compute_semantics=semantic_compute_requested,
    )
    if semantic_compute_requested and isinstance(enriched_route_info or route_info, dict):
        preview_route_info = dict(enriched_route_info or route_info or {})
        route_runtime_for_compute = (
            dict(preview_route_info.get('route_runtime'))
            if isinstance(preview_route_info.get('route_runtime'), Mapping)
            else {}
        )
        semantic_compute_state = (
            dict(route_runtime_for_compute.get('semantic_compute'))
            if isinstance(route_runtime_for_compute.get('semantic_compute'), Mapping)
            else {}
        )
        semantic_compute_state.setdefault('requested', True)
        semantic_compute_state.setdefault('allowed', True)
        semantic_compute_state.setdefault('preview', True)
        semantic_compute_state['source'] = compute_semantics_source
        semantic_compute_state['policy'] = compute_semantics_policy
        semantic_compute_state['false_override'] = compute_semantics_false_override
        semantic_compute_state['learnable'] = False
        if isinstance(_planner_meta, Mapping) and _planner_meta.get('attempted'):
            semantic_compute_state['performed'] = True
            semantic_compute_state['evidence_role'] = 'preview_computed_non_learnable'
        else:
            semantic_compute_state['performed'] = bool(semantic_compute_state.get('performed'))
            semantic_compute_state.setdefault(
                'evidence_role',
                'preview_computed_non_learnable'
                if semantic_compute_state.get('performed')
                else 'preview_cached_non_learnable',
            )
        route_runtime_for_compute['semantic_compute'] = semantic_compute_state
        preview_route_info['route_runtime'] = route_runtime_for_compute
        enriched_route_info = preview_route_info
    route_runtime = (
        (enriched_route_info or route_info or {}).get('route_runtime')
        if isinstance(enriched_route_info or route_info, dict)
        else {}
    )
    semantic_compute = (
        route_runtime.get('semantic_compute')
        if isinstance(route_runtime, Mapping) and isinstance(route_runtime.get('semantic_compute'), Mapping)
        else {}
    )
    semantic_compute_performed = bool(semantic_compute.get('performed'))
    runtime_truth = _runtime_truth_metadata(
        refresh_requested=refresh_requested,
        semantic_compute_requested=semantic_compute_requested,
        semantic_compute_performed=semantic_compute_performed,
        compute_semantics_source=compute_semantics_source,
        compute_semantics_policy=compute_semantics_policy,
        compute_semantics_false_override=compute_semantics_false_override,
    )
    payload = _build_ghost_route_preview_payload(enriched_route_info or route_info or {}, request_payload=_effective_data)
    payload['runtime_truth'] = runtime_truth
    return jsonify(payload)


@app.route('/api/ghost_preferences', methods=['GET'])
def api_get_ghost_preferences():
    return jsonify(load_persisted_ghost_preferences())


@app.route('/api/integrations/codex/status', methods=['GET'])
def api_codex_integration_status():
    force_refresh = _parse_bool(request.args.get('refresh'), default=False)
    return jsonify(_codex_external_target_payload(force_refresh=force_refresh))


@app.route('/api/ghost_preferences', methods=['POST'])
def api_save_ghost_preferences():
    payload = request.get_json(silent=True) or {}
    return jsonify(persist_ghost_preferences(payload))


def _chat_timeout_seconds(model_name: str, backend: str, capability: str) -> int:
    normalized_backend = normalize_backend(backend)
    normalized_capability = normalize_capability(capability)
    model_lower = str(model_name or '').lower()

    if normalized_capability != 'chat':
        return 180

    size_hint_billion = _model_size_hint_billion(model_lower)
    heavy_markers = (
        'qwen3.5',
        ':27b',
        ':30b',
        'qwen3-coder',
        'gpt-oss',
        'deepseek-r1',
    )
    is_heavy = any(marker in model_lower for marker in heavy_markers) or (
        normalized_backend == 'ollama' and size_hint_billion is not None and size_hint_billion >= 20
    )

    if normalized_backend in {'mlx', 'llama_cpp'}:
        return 900 if is_heavy else 600

    if is_heavy:
        return 600
    return 180


def _model_size_hint_billion(model_name: str) -> Optional[float]:
    for match in re.finditer(r'(?:^|[:/_-])(\d+(?:\.\d+)?)b(?:$|[:/_-])', str(model_name or '').lower()):
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def _apply_batch_image_dimensions(batch_infer_payload: dict, batch_item: dict) -> tuple[Optional[dict], Optional[str]]:
    return _RESPONSES_REQUEST_RUNTIME.apply_batch_image_dimensions(
        batch_infer_payload,
        batch_item,
    )


def _response_lookup_version_for_status(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    except Exception:  # noqa: BLE001
        encoded = repr(value).encode('utf-8', errors='replace')
    return hashlib.sha256(encoded).hexdigest()


def _response_lookup_count_by_status(items: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not isinstance(items, list):
        return {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get('status') or item.get('lifecycle') or 'unknown').strip().lower() or 'unknown'
        counts[status] += 1
    return dict(sorted(counts.items()))


def _compact_response_lookup_branch(item: Any) -> dict[str, Any]:
    return _response_wire_policy.compact_lookup_branch(item)


def _response_lookup_late_fill_for_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    late_fill = payload.get('late_fill') or payload.get('lateFill') or runtime.get('late_fill') or runtime.get('lateFill')
    if not isinstance(late_fill, Mapping):
        return {}
    pending_branches = late_fill.get('pending_branches') if isinstance(late_fill.get('pending_branches'), list) else []
    active_branches = late_fill.get('active_branches') if isinstance(late_fill.get('active_branches'), list) else []
    failed_branches = late_fill.get('failed_branches') if isinstance(late_fill.get('failed_branches'), list) else []
    completed_branches = (
        late_fill.get('completed_branches')
        if isinstance(late_fill.get('completed_branches'), list)
        else []
    )
    branch_progress = (
        late_fill.get('branch_progress')
        if isinstance(late_fill.get('branch_progress'), list)
        else []
    )
    recovery_candidates = (
        late_fill.get('recovery_candidates')
        if isinstance(late_fill.get('recovery_candidates'), list)
        else []
    )
    compact: dict[str, Any] = {
        'status': str(late_fill.get('status') or '').strip() or None,
        'pending_count': len(pending_branches),
        'active_count': len(active_branches),
        'failed_count': len(failed_branches),
        'completed_count': len(completed_branches),
        'branch_progress_count': len(branch_progress),
        'recovery_candidate_count': len(recovery_candidates),
    }
    for key in ('linked_artifact_rebind_status', 'final_materialization_contract_status'):
        value = late_fill.get(key)
        if value not in (None, '', [], {}):
            compact[key] = value
    if pending_branches:
        compact['pending_branches'] = [_response_wire_branch_handle(item) for item in pending_branches[:_RESPONSE_WIRE_COLLECTION_LIMIT]]
        compact['pending_status_counts'] = _response_lookup_count_by_status(pending_branches)
    if active_branches:
        compact['active_branches'] = [_response_wire_branch_handle(item) for item in active_branches[:_RESPONSE_WIRE_COLLECTION_LIMIT]]
        compact['active_status_counts'] = _response_lookup_count_by_status(active_branches)
    if failed_branches:
        compact['failed_branches'] = [_response_wire_branch_handle(item) for item in failed_branches[:_RESPONSE_WIRE_COLLECTION_LIMIT]]
        compact['failed_status_counts'] = _response_lookup_count_by_status(failed_branches)
    if completed_branches:
        compact['completed_status_counts'] = _response_lookup_count_by_status(completed_branches)
    if branch_progress:
        compact['branch_progress'] = [_response_wire_branch_handle(item) for item in branch_progress[:_RESPONSE_WIRE_COLLECTION_LIMIT]]
        compact['branch_progress_status_counts'] = _response_lookup_count_by_status(branch_progress)
    return {key: value for key, value in compact.items() if value not in (None, '', [], {})}


def _response_lookup_output_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    output_slots = payload.get('output_slots') if isinstance(payload.get('output_slots'), list) else []
    outputs = payload.get('outputs') if isinstance(payload.get('outputs'), list) else []
    artifacts = payload.get('artifacts') if isinstance(payload.get('artifacts'), list) else []

    def total_count(metadata_key: str, preview_count: int, *fallbacks: Any) -> int:
        count = preview_count
        for value in (payload.get(metadata_key), *fallbacks):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                count = max(count, value)
        return count

    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), Mapping)
        else {}
    )
    frame_current = (
        response_frame.get('current_state')
        if isinstance(response_frame.get('current_state'), Mapping)
        else {}
    )
    frame_output = (
        response_frame.get('output')
        if isinstance(response_frame.get('output'), Mapping)
        else {}
    )
    frame_artifacts = (
        response_frame.get('artifacts')
        if isinstance(response_frame.get('artifacts'), Mapping)
        else {}
    )
    output_slot_count = total_count(
        'output_slots_count',
        len(output_slots),
        frame_current.get('output_slots_count'),
    )
    output_count = total_count(
        'outputs_count',
        len(outputs),
        frame_current.get('outputs_count'),
        frame_output.get('item_count'),
    )
    artifact_count = total_count(
        'artifacts_count',
        len(artifacts),
        frame_current.get('artifacts_count'),
        frame_artifacts.get('output_count'),
    )
    result = {
        'output_slot_count': output_slot_count,
        'output_count': output_count,
        'artifact_count': artifact_count,
        'output_slot_status_counts': _response_lookup_count_by_status(output_slots),
        'output_status_counts': _response_lookup_count_by_status(outputs),
    }
    if output_slot_count > len(output_slots):
        result['output_slot_status_counts_projection_complete'] = False
    if output_count > len(outputs):
        result['output_status_counts_projection_complete'] = False
    return result


def _response_lookup_surface_for_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    surface_state = (
        payload.get('surface_state')
        if isinstance(payload.get('surface_state'), Mapping)
        else runtime.get('surface_state')
        if isinstance(runtime.get('surface_state'), Mapping)
        else None
    )
    if not isinstance(surface_state, Mapping):
        return {}
    return _response_wire_surface_handle(surface_state)


def _build_response_status_lookup_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record.get('response_payload') or {})
    orphaned_retry_projected = False
    response_id = str(record.get('id') or payload.get('id') or '').strip()
    if payload and response_id:
        payload, orphaned_retry_projected = _project_orphaned_late_fill_retry_attempts(response_id, payload)
        payload, stale_active_projected = _project_stale_late_fill_active_branches(
            response_id,
            payload,
            record=record,
        )
        orphaned_retry_projected = orphaned_retry_projected or stale_active_projected
    if payload:
        if record.get('status'):
            payload['status'] = str(record.get('status') or payload.get('status') or 'completed')
        payload['lifecycle_state'] = str(
            (None if orphaned_retry_projected else record.get('lifecycle_state'))
            or derive_response_lifecycle_state(payload, requested_status=payload.get('status'))
            or payload.get('status')
            or 'completed'
        )
        if record.get('error_message'):
            payload['error'] = {'message': str(record.get('error_message') or '').strip()}
        semantic_payload = _attach_response_status_semantics(dict(payload))
    else:
        semantic_payload = _attach_response_status_semantics({
            'id': response_id,
            'status': str(record.get('status') or 'in_progress'),
            'lifecycle_state': str(record.get('lifecycle_state') or record.get('status') or 'in_progress'),
            **({'error': {'message': str(record.get('error_message') or '').strip()}} if record.get('error_message') else {}),
        })
    status_semantics = (
        semantic_payload.get('status_semantics')
        if isinstance(semantic_payload.get('status_semantics'), Mapping)
        else {}
    )
    status_payload: dict[str, Any] = {
        'id': response_id or str(semantic_payload.get('id') or '').strip(),
        'object': 'response.status',
        'status': str(semantic_payload.get('status') or record.get('status') or 'in_progress'),
        'lifecycle_state': str(
            semantic_payload.get('lifecycle_state')
            or status_semantics.get('canonical_lifecycle_state')
            or record.get('lifecycle_state')
            or 'in_progress'
        ),
        'canonical_status_field': 'lifecycle_state',
        'status_compatibility': semantic_payload.get('status_compatibility'),
        'status_semantics': dict(status_semantics),
        'late_fill': _response_lookup_late_fill_for_status(semantic_payload),
        'output_counts': _response_lookup_output_counts(semantic_payload),
        'surface_state': _response_lookup_surface_for_status(semantic_payload),
    }
    response_frame = (
        semantic_payload.get('response_frame')
        if isinstance(semantic_payload.get('response_frame'), Mapping)
        else {}
    )
    frame_sequence = response_frame.get('frame_sequence') if isinstance(response_frame, Mapping) else None
    if frame_sequence not in (None, '', [], {}):
        status_payload['frame_sequence'] = frame_sequence
    frame_id = str(response_frame.get('frame_id') or '').strip() if isinstance(response_frame, Mapping) else ''
    if frame_id:
        status_payload['frame_id'] = frame_id
    if semantic_payload.get('error'):
        status_payload['error'] = semantic_payload.get('error')
    status_payload = {key: value for key, value in status_payload.items() if value not in (None, '', [], {})}
    status_payload['state_version'] = _response_lookup_version_for_status(status_payload)
    status_payload['compact'] = True
    return status_payload


def _attach_response_lookup_state_version(
    payload: dict[str, Any],
    *,
    status_record: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    status_payload = _build_response_status_lookup_payload(
        status_record
        if isinstance(status_record, dict)
        else {
            'id': str(payload.get('id') or payload.get('response_id') or '').strip(),
            'status': payload.get('status'),
            'lifecycle_state': payload.get('lifecycle_state'),
            'error_message': payload.get('error', {}).get('message') if isinstance(payload.get('error'), Mapping) else None,
            'response_payload': payload,
        }
    )
    payload['state_version'] = status_payload.get('state_version')
    payload['status_lookup'] = status_payload
    return payload


def _compact_response_lookup_branch_for_ui(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    compact = _compact_response_lookup_branch(item)
    for key in (
        'artifact_ref',
        'ref',
        'path',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'blocked_reason',
        'cancel_reason',
        'waiver_reason',
        'supersession_reason',
        'error',
        'error_ref',
        'target_path',
        'text_artifact_extension',
        'text_artifact_source_name',
        'image_artifact_persisted_from_raw_late_fill',
        'image_artifact_persisted_path',
        'lang_code',
        'lang_code_source',
        'response_format',
        'output_format',
    ):
        value = item.get(key)
        if value not in (None, '', [], {}):
            compact[key] = value
    attempt = item.get('attempt')
    if isinstance(attempt, Mapping) and attempt:
        compact['attempt'] = {
            key: value
            for key, value in dict(attempt).items()
            if value not in (None, '', [], {})
        }
    artifacts = item.get('artifacts')
    if isinstance(artifacts, list) and artifacts:
        compact['artifacts'] = [_response_lookup_artifact_for_ui(artifact) for artifact in artifacts if isinstance(artifact, Mapping)]
    recovery_context = item.get('recovery_context')
    if isinstance(recovery_context, Mapping) and recovery_context:
        compact['recovery_context'] = {
            key: value
            for key, value in dict(recovery_context).items()
            if value not in (None, '', [], {})
        }
    recovery_state = item.get('recovery_state')
    if isinstance(recovery_state, Mapping) and recovery_state:
        compact['recovery_state'] = {
            key: value
            for key, value in dict(recovery_state).items()
            if value not in (None, '', [], {})
        }
    return {key: value for key, value in compact.items() if value not in (None, '', [], {})}


def _compact_late_fill_repair_state_for_ui(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        'status',
        'auto_execute',
        'repair_work_available',
        'materialization_blocked',
        'needs_external_input',
        'reason',
    ):
        item = value.get(key)
        if item not in (None, '', [], {}):
            compact[key] = item
    return compact


def _response_lookup_late_fill_for_ui(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
    late_fill = payload.get('late_fill') or payload.get('lateFill') or runtime.get('late_fill') or runtime.get('lateFill')
    if not isinstance(late_fill, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        'status',
        'lifecycle_state',
        'code',
        'trigger',
        'expected_capability',
        'missing_artifact_type',
        'fill_model',
        'fill_backend',
        'fill_instance_id',
        'route_source',
        'route_reason',
        'error',
        'content_payload',
        'stage_direction',
        'planned_prompt',
        'partial_failure',
        'pending_branch_count',
        'active_branch_count',
        'completed_branch_count',
        'failed_branch_count',
        'cancelled_branch_count',
        'pending_capabilities',
        'active_capabilities',
        'completed_capabilities',
        'failed_capabilities',
        'cancelled_capabilities',
        'auto_recovery_enabled',
        'auto_executable_repair_attempted',
        'final_materialization_contract_status',
        'final_materialization_contract_reason',
        'materialization_contract_unmet',
        'materialization_contract_open_check_count',
        'linked_artifact_rebind_status',
    ):
        value = late_fill.get(key)
        if value not in (None, '', [], {}):
            compact[key] = value
    for source_key, target_key in (
        ('pending_branches', 'pending_branches'),
        ('active_branches', 'active_branches'),
        ('completed_branches', 'completed_branches'),
        ('failed_branches', 'failed_branches'),
        ('cancelled_branches', 'cancelled_branches'),
        ('branch_progress', 'branch_progress'),
        ('fill_results', 'fill_results'),
        ('recovery_candidates', 'recovery_candidates'),
        ('repair_rebuild_contracts', 'repair_rebuild_contracts'),
    ):
        values = late_fill.get(source_key)
        if isinstance(values, list):
            compact[target_key] = [
                _compact_response_lookup_branch_for_ui(item)
                for item in values
                if isinstance(item, Mapping)
            ]
    open_checks = late_fill.get('materialization_contract_open_checks')
    if isinstance(open_checks, list) and open_checks:
        check_payloads: list[dict[str, Any]] = []
        for check in open_checks[:20]:
            if not isinstance(check, Mapping):
                continue
            check_payload: dict[str, Any] = {}
            for key in (
                'status',
                'check_kind',
                'evidence',
                'role',
                'reason',
                'branch_id',
                'phase_id',
                'text_artifact_extension',
                'text_artifact_source_name',
            ):
                value = check.get(key)
                if value not in (None, '', [], {}):
                    check_payload[key] = value
            if check_payload:
                check_payloads.append(check_payload)
        if check_payloads:
            compact['materialization_contract_open_checks'] = check_payloads
    linked_rebinds = late_fill.get('linked_artifact_rebinds')
    if isinstance(linked_rebinds, list) and linked_rebinds:
        compact_rebinds: list[dict[str, Any]] = []
        for rebind in linked_rebinds[:20]:
            if not isinstance(rebind, Mapping):
                continue
            compact_rebind: dict[str, Any] = {}
            for key in ('status', 'target_path', 'target_extension', 'change_count'):
                value = rebind.get(key)
                if value not in (None, '', [], {}):
                    compact_rebind[key] = value
            changes = rebind.get('changes')
            if isinstance(changes, list) and changes:
                compact_rebind['changes'] = [
                    {
                        child_key: child_value
                        for child_key, child_value in {
                            'kind': change.get('kind') if isinstance(change, Mapping) else None,
                            'from': change.get('from') if isinstance(change, Mapping) else None,
                            'to': change.get('to') if isinstance(change, Mapping) else None,
                            'linked_path': change.get('linked_path') if isinstance(change, Mapping) else None,
                        }.items()
                        if child_value not in (None, '', [], {})
                    }
                    for change in changes[:20]
                    if isinstance(change, Mapping)
                ]
                compact_rebind['changes'] = [item for item in compact_rebind['changes'] if item]
            if compact_rebind:
                compact_rebinds.append(compact_rebind)
        if compact_rebinds:
            compact['linked_artifact_rebinds'] = compact_rebinds
    for source_key in ('repair_loop', 'reconsideration_rebuild'):
        repair_state = _compact_late_fill_repair_state_for_ui(late_fill.get(source_key))
        if repair_state:
            compact[source_key] = repair_state
    surface_state = _response_lookup_surface_for_status({'surface_state': late_fill.get('surface_state')})
    if surface_state:
        compact['surface_state'] = surface_state
    filtered = {key: value for key, value in compact.items() if value not in (None, '', [], {})}
    for key in (
        'pending_branches',
        'active_branches',
        'completed_branches',
        'failed_branches',
        'cancelled_branches',
        'branch_progress',
        'fill_results',
        'recovery_candidates',
    ):
        if key in compact and isinstance(compact.get(key), list) and not compact.get(key):
            filtered[key] = []
    return filtered


def _response_lookup_artifact_for_ui(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        'artifact_id',
        'artifact_ref',
        'batch_index',
        'ref',
        'type',
        'kind',
        'mime_type',
        'name',
        'path',
        'origin',
        'prompt',
        'content',
        'content_source',
        'content_length_chars',
        'content_preview_truncated',
        'content_sha256',
        'file_sha256',
        'file_size_bytes',
        'source_response_id',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'text_artifact_extension',
        'text_artifact_source_name',
        'syntax_sanity_status',
        'syntax_sanity_issue_count',
    ):
        item = value.get(key)
        if item not in (None, '', [], {}):
            compact[key] = item
    if 'content' in compact and isinstance(compact['content'], str) and len(compact['content']) > 100_000:
        compact['content_preview'] = compact['content'][:100_000]
        compact['content_preview_truncated'] = True
        compact['content_length_chars'] = len(compact['content'])
        compact.pop('content', None)
    return compact


def _response_lookup_output_for_ui(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        'id',
        'batch_index',
        'slot_id',
        'branch_id',
        'phase_id',
        'type',
        'status',
        'lifecycle',
        'artifact_ref',
        'path',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'value',
        'placeholder_ref',
        'blocked_reason',
        'error_ref',
        'recovery_context',
        'recovery_state',
        'capability',
        'output_type',
    ):
        item = value.get(key)
        if item not in (None, '', [], {}):
            compact[key] = item
    if 'value' in compact and isinstance(compact['value'], str) and len(compact['value']) > 100_000:
        compact['value_preview'] = compact['value'][:100_000]
        compact['value_preview_truncated'] = True
        compact['value_length_chars'] = len(compact['value'])
        compact.pop('value', None)
    return compact


def _response_lookup_output_message_for_ui(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in ('id', 'type', 'role', 'status'):
        item = value.get(key)
        if item not in (None, '', [], {}):
            compact[key] = item
    content = value.get('content')
    if isinstance(content, list):
        compact_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            child: dict[str, Any] = {}
            for key in ('type', 'text'):
                child_value = item.get(key)
                if child_value not in (None, '', [], {}):
                    child[key] = child_value
            if child:
                compact_content.append(child)
        if compact_content:
            compact['content'] = compact_content
    return compact


def _build_response_lookup_ui_source_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record.get('response_payload') or {})
    if payload:
        response_id = str(record.get('id') or payload.get('id') or '').strip()
        if response_id:
            payload, orphaned_retry_projected = _project_orphaned_late_fill_retry_attempts(response_id, payload)
            payload, stale_active_projected = _project_stale_late_fill_active_branches(
                response_id,
                payload,
                record=record,
            )
            orphaned_retry_projected = orphaned_retry_projected or stale_active_projected
        else:
            orphaned_retry_projected = False
        if record.get('status'):
            payload['status'] = str(record.get('status') or payload.get('status') or 'completed')
        payload['lifecycle_state'] = str(
            (None if orphaned_retry_projected else record.get('lifecycle_state'))
            or derive_response_lifecycle_state(payload, requested_status=payload.get('status'))
            or payload.get('status')
            or 'completed'
        )
        if record.get('error_message'):
            payload['error'] = {'message': str(record.get('error_message') or '').strip()}
        payload = _attach_response_artifact_bundles_from_registry(payload)
        status_record = dict(record)
        status_record['response_payload'] = dict(payload)
        original_response_frame = (
            payload.get('response_frame')
            if isinstance(payload.get('response_frame'), Mapping)
            else {}
        )
        original_planning = (
            original_response_frame.get('planning')
            if isinstance(original_response_frame.get('planning'), Mapping)
            else {}
        )
        original_artifact_flow = (
            original_planning.get('artifact_flow')
            if isinstance(original_planning.get('artifact_flow'), Mapping)
            else {}
        )
        if isinstance(original_artifact_flow.get('output_slots'), list) and original_artifact_flow.get('output_slots'):
            payload['_lookup_original_response_frame_output_slots'] = True
        payload = _attach_response_status_semantics(payload)
        payload = _attach_lookup_replay_pending_output_slots(payload)
        payload = _reconcile_lookup_output_slots_with_late_fill_truth(payload)
        if not isinstance(payload.get('response_frame'), Mapping):
            payload = _attach_response_frame(payload, request_payload={})
        if (
            record.get('lookup_source') == 'response_frame_ledger'
            and isinstance(payload.get('response_frame'), Mapping)
            and not orphaned_retry_projected
        ):
            payload['response_frame'] = _enrich_response_frame_metadata(payload['response_frame'])
        payload = _hoist_response_output_surfaces(payload)
        if isinstance(payload.get('response_frame'), Mapping):
            frozen_response_frame = copy.deepcopy(payload['response_frame'])
            frame_payload = _attach_response_frame(payload, request_payload={})
            if isinstance(frame_payload.get('response_frame'), Mapping):
                payload['response_frame'] = _preserve_frozen_response_frame_identity(
                    frame_payload['response_frame'],
                    frozen_response_frame,
                )
        return _attach_response_lookup_state_version(
            payload,
            status_record=status_record,
        )

    response_payload = _build_canonical_response_payload(
        instance_id=str(record.get('instance_id') or '').strip(),
        model_name=str(record.get('model_name') or '').strip(),
        backend=str(record.get('backend') or '').strip(),
        capability=str(record.get('capability') or '').strip(),
        mode=str(record.get('mode') or 'chat').strip() or 'chat',
        output_text=str(record.get('output_text') or ''),
        source_payload={},
        route_payload=record.get('route_payload') if isinstance(record.get('route_payload'), dict) else None,
        response_id=str(record.get('id') or '').strip() or None,
        message_id=str(record.get('message_id') or '').strip() or None,
    )
    response_payload['status'] = str(record.get('status') or 'in_progress')
    response_payload['lifecycle_state'] = str(
        record.get('lifecycle_state')
        or derive_response_lifecycle_state(response_payload, requested_status=response_payload.get('status'))
    )
    if response_payload.get('output') and isinstance(response_payload['output'], list):
        first_item = response_payload['output'][0]
        if isinstance(first_item, dict):
            first_item['status'] = 'completed' if response_payload['status'] == 'completed' else 'in_progress'
    if record.get('error_message'):
        response_payload['error'] = {'message': str(record.get('error_message') or '').strip()}
    response_payload = _attach_response_artifact_bundles_from_registry(response_payload)
    status_record = dict(record)
    status_record['response_payload'] = dict(response_payload)
    return _attach_response_lookup_state_version(
        _hoist_response_output_surfaces(_attach_response_status_semantics(response_payload)),
        status_record=status_record,
    )


def _build_response_ui_lookup_payload(record: dict[str, Any]) -> dict[str, Any]:
    full_payload = _build_response_lookup_ui_source_payload(record)
    status_lookup = (
        full_payload.get('status_lookup')
        if isinstance(full_payload.get('status_lookup'), Mapping)
        else _build_response_status_lookup_payload(record)
    )
    ui_payload: dict[str, Any] = {}
    for key in (
        'id',
        'response_id',
        'object',
        'status',
        'lifecycle_state',
        'canonical_status_field',
        'status_compatibility',
        'status_semantics',
        'state_version',
        'model',
        'backend',
        'capability',
        'mode',
        'instance_id',
        'route_source',
        'route_reason',
        'route_router_instance_id',
        'route_router_model',
        'route_artifact_ref',
        'route_artifact_path',
        'route_reuse_last_artifact',
        'reference_image_count',
        'reference_image_kind',
        'context_mode',
        'context_reason',
        'output_text',
        'saved_image_path',
        'saved_audio_path',
        'saved_text_path',
        'error',
        'lang_code',
        'lang_code_source',
        'response_format',
        'output_format',
        'usage',
        'created_at',
        'updated_at',
        'input_artifacts',
    ):
        value = full_payload.get(key)
        if value not in (None, '', [], {}):
            ui_payload[key] = value
    if full_payload.get('image_data_url') and not full_payload.get('saved_image_path'):
        ui_payload['image_data_url'] = full_payload.get('image_data_url')
    if status_lookup.get('frame_sequence') not in (None, '', [], {}):
        ui_payload['frame_sequence'] = status_lookup.get('frame_sequence')
    if status_lookup.get('frame_id'):
        ui_payload['frame_id'] = status_lookup.get('frame_id')
    response_frame = full_payload.get('response_frame') if isinstance(full_payload.get('response_frame'), Mapping) else {}
    response_frame_planning = response_frame.get('planning') if isinstance(response_frame.get('planning'), Mapping) else {}
    response_frame_artifact_flow = (
        response_frame_planning.get('artifact_flow')
        if isinstance(response_frame_planning.get('artifact_flow'), Mapping)
        else {}
    )
    response_frame_has_output_slots = isinstance(response_frame_artifact_flow.get('output_slots'), list) and bool(
        response_frame_artifact_flow.get('output_slots')
    )
    if response_frame and (
        full_payload.get('_lookup_replay_response_frame_required') is True
        or full_payload.get('_lookup_original_response_frame_output_slots') is True
    ):
        frame_for_ui = dict(response_frame)
        outputs_for_ui = full_payload.get('outputs') if isinstance(full_payload.get('outputs'), list) else []
        if response_frame_has_output_slots and outputs_for_ui:
            planning_for_ui = (
                dict(frame_for_ui.get('planning'))
                if isinstance(frame_for_ui.get('planning'), Mapping)
                else {}
            )
            artifact_flow_for_ui = (
                dict(planning_for_ui.get('artifact_flow'))
                if isinstance(planning_for_ui.get('artifact_flow'), Mapping)
                else {}
            )
            slots_for_ui: list[dict[str, Any]] = []
            used_output_indices: set[int] = set()
            for raw_slot in response_frame_artifact_flow.get('output_slots') or []:
                if not isinstance(raw_slot, Mapping):
                    continue
                slot = dict(raw_slot)
                matching_output = None
                for index, raw_output in enumerate(outputs_for_ui):
                    if index in used_output_indices or not isinstance(raw_output, Mapping):
                        continue
                    if any(
                        str(slot.get(key) or '').strip()
                        and str(slot.get(key) or '').strip() == str(raw_output.get(key) or '').strip()
                        for key in ('slot_id', 'branch_id', 'phase_id')
                    ):
                        matching_output = raw_output
                        used_output_indices.add(index)
                        break
                if matching_output is None:
                    slot_type = str(slot.get('type') or '').strip().lower()
                    for index, raw_output in enumerate(outputs_for_ui):
                        if index in used_output_indices or not isinstance(raw_output, Mapping):
                            continue
                        output_type = str(raw_output.get('type') or '').strip().lower()
                        output_status = str(raw_output.get('status') or '').strip().lower()
                        if slot_type and slot_type == output_type and output_status in {'fulfilled', 'completed', 'blocked'}:
                            matching_output = raw_output
                            used_output_indices.add(index)
                            break
                if isinstance(matching_output, Mapping):
                    for key in (
                        'status',
                        'lifecycle',
                        'artifact_ref',
                        'path',
                        'saved_image_path',
                        'saved_audio_path',
                        'saved_text_path',
                        'blocked_reason',
                        'error_ref',
                        'recovery_context',
                        'recovery_state',
                    ):
                        value = matching_output.get(key)
                        if value not in (None, '', [], {}):
                            slot[key] = value
                slots_for_ui.append(slot)
            if slots_for_ui:
                artifact_flow_for_ui['output_slots'] = slots_for_ui
                planning_for_ui['artifact_flow'] = artifact_flow_for_ui
                frame_for_ui['planning'] = planning_for_ui
        ui_payload['response_frame'] = frame_for_ui
    for source_key, target_key in (
        ('frame_sequence', 'frame_sequence'),
        ('frame_id', 'frame_id'),
    ):
        value = response_frame.get(source_key) if isinstance(response_frame, Mapping) else None
        if target_key not in ui_payload and value not in (None, '', [], {}):
            ui_payload[target_key] = value
    for key in ('artifacts',):
        values = full_payload.get(key)
        if isinstance(values, list):
            ui_payload[key] = [_response_lookup_artifact_for_ui(item) for item in values if isinstance(item, Mapping)]
    for key in ('outputs', 'output_slots', 'output_branches'):
        values = full_payload.get(key)
        if isinstance(values, list):
            ui_payload[key] = [_response_lookup_output_for_ui(item) for item in values if isinstance(item, Mapping)]
    output_messages = full_payload.get('output')
    if isinstance(output_messages, list):
        ui_payload['output'] = [
            _response_lookup_output_message_for_ui(item)
            for item in output_messages
            if isinstance(item, Mapping)
        ]
    late_fill = _response_lookup_late_fill_for_ui(full_payload)
    if late_fill:
        ui_payload['late_fill'] = late_fill
    surface_state = _response_lookup_surface_for_status(full_payload)
    if surface_state:
        ui_payload['surface_state'] = surface_state
    for key in ('artifact_bundles', 'artifactBundles'):
        values = full_payload.get(key)
        if isinstance(values, list):
            ui_payload[key] = [dict(item) for item in values if isinstance(item, Mapping)]
    artifact_bundle = full_payload.get('artifactBundle')
    if isinstance(artifact_bundle, Mapping):
        ui_payload['artifactBundle'] = dict(artifact_bundle)
    ui_payload['status_lookup'] = dict(status_lookup)
    ui_payload['state_version'] = status_lookup.get('state_version') or ui_payload.get('state_version')
    ui_payload['ui_compact'] = True
    return {key: value for key, value in ui_payload.items() if value not in (None, '', [], {})}


def _response_artifact_bundle_source_response_id(record: Mapping[str, Any]) -> str:
    bundle = record.get('bundle') if isinstance(record.get('bundle'), Mapping) else {}
    artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
    source = record.get('source') if isinstance(record.get('source'), Mapping) else {}
    return str(
        bundle.get('source_response_id')
        or artifact.get('source_response_id')
        or source.get('response_id')
        or ''
    ).strip()


def _normalize_response_artifact_bundle_payload(record: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    bundle = record.get('bundle') if isinstance(record.get('bundle'), Mapping) else {}
    artifact = record.get('artifact') if isinstance(record.get('artifact'), Mapping) else {}
    source_kind = ''
    source = record.get('source') if isinstance(record.get('source'), Mapping) else {}
    if isinstance(source, Mapping):
        source_kind = str(source.get('kind') or '').strip()
    artifact_kind = str(artifact.get('kind') or artifact.get('type') or '').strip()
    if artifact_kind != 'response_artifact_bundle' and source_kind != 'response_artifact_bundle' and not bundle:
        return None
    payload = dict(bundle) if bundle else {
        'bundle_id': record.get('artifact_ref') or artifact.get('artifact_ref'),
        'bundle_path': artifact.get('path'),
        'entrypoint': artifact.get('entrypoint'),
        'manifest_path': artifact.get('manifest_path'),
        'source_response_id': artifact.get('source_response_id') or source.get('response_id'),
        'source_artifact_refs': artifact.get('source_artifact_refs'),
        'link_check': artifact.get('link_check'),
    }
    bundle_path = str(payload.get('bundle_path') or payload.get('bundlePath') or '').strip()
    entrypoint = str(payload.get('entrypoint') or '').strip()
    if not bundle_path and not entrypoint:
        return None
    if not str(payload.get('source_response_id') or '').strip():
        source_response_id = _response_artifact_bundle_source_response_id(record)
        if source_response_id:
            payload['source_response_id'] = source_response_id
    return {key: value for key, value in payload.items() if value not in (None, '', [], {})}


def _response_artifact_bundle_key(bundle: Mapping[str, Any]) -> str:
    return str(
        bundle.get('bundle_id')
        or bundle.get('bundleId')
        or bundle.get('bundle_path')
        or bundle.get('bundlePath')
        or bundle.get('entrypoint')
        or ''
    ).strip()


def _resolve_response_artifact_bundle_path(value: Any) -> Optional[Path]:
    token = str(value or '').strip()
    if not token:
        return None
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve(strict=False)
    except OSError:
        return candidate


def _openable_response_artifact_bundle_payload(bundle: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    bundle_path = str(bundle.get('bundle_path') or bundle.get('bundlePath') or '').strip()
    bundle_dir = _resolve_response_artifact_bundle_path(bundle_path)
    if bundle_dir is None or not bundle_dir.is_dir():
        return None
    payload = dict(bundle)
    entrypoint = str(payload.get('entrypoint') or '').strip()
    if entrypoint:
        entrypoint_path = _resolve_response_artifact_bundle_path(entrypoint)
        if entrypoint_path is None or not entrypoint_path.is_file():
            payload.pop('entrypoint', None)
            payload.pop('entrypoint_relative_path', None)
            payload.pop('entrypointRelativePath', None)
    return payload


def _response_artifact_bundles_for_response(response_id: Any) -> list[dict[str, Any]]:
    target_response_id = str(response_id or '').strip()
    if not target_response_id:
        return []
    ledger_path = Path(ARTIFACT_REGISTRY_LEDGER)
    try:
        lines = ledger_path.read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        logging.debug('Could not read response artifact bundle registry %s: %s', ledger_path, exc)
        return []

    bundles: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        if _response_artifact_bundle_source_response_id(record) != target_response_id:
            continue
        bundle = _normalize_response_artifact_bundle_payload(record)
        if not bundle:
            continue
        bundle = _openable_response_artifact_bundle_payload(bundle)
        if not bundle:
            continue
        key = _response_artifact_bundle_key(bundle)
        if not key:
            continue
        previous_index = index_by_key.get(key)
        if previous_index is not None:
            bundles.pop(previous_index)
            index_by_key = {
                _response_artifact_bundle_key(item): index
                for index, item in enumerate(bundles)
                if _response_artifact_bundle_key(item)
            }
        index_by_key[key] = len(bundles)
        bundles.append(bundle)
    return bundles


def _attach_response_artifact_bundles_from_registry(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    response_id = str(payload.get('id') or payload.get('response_id') or '').strip()
    if not response_id:
        return payload

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    existing_candidates: list[Any] = []
    for key in ('artifact_bundles', 'artifactBundles'):
        value = payload.get(key)
        if isinstance(value, list):
            existing_candidates.extend(value)
    artifact_bundle = payload.get('artifactBundle')
    if isinstance(artifact_bundle, Mapping):
        existing_candidates.append(artifact_bundle)

    for item in existing_candidates:
        if not isinstance(item, Mapping):
            continue
        bundle = _openable_response_artifact_bundle_payload(item)
        if not bundle:
            continue
        key = _response_artifact_bundle_key(bundle)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(bundle)

    for bundle in _response_artifact_bundles_for_response(response_id):
        key = _response_artifact_bundle_key(bundle)
        if not key:
            continue
        if key in seen:
            merged = [item for item in merged if _response_artifact_bundle_key(item) != key]
        seen.add(key)
        merged.append(bundle)

    if not merged:
        return payload
    merged = [merged[-1]]
    updated = dict(payload)
    updated['artifact_bundles'] = merged
    updated['artifactBundles'] = merged
    updated['artifactBundle'] = merged[-1]
    return updated


def _build_response_lookup_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record.get('response_payload') or {})
    if payload:
        response_id = str(record.get('id') or payload.get('id') or '').strip()
        if response_id:
            payload, orphaned_retry_projected = _project_orphaned_late_fill_retry_attempts(response_id, payload)
            payload, stale_active_projected = _project_stale_late_fill_active_branches(
                response_id,
                payload,
                record=record,
            )
            orphaned_retry_projected = orphaned_retry_projected or stale_active_projected
        else:
            orphaned_retry_projected = False
        if record.get('status'):
            payload['status'] = str(record.get('status') or payload.get('status') or 'completed')
        payload['lifecycle_state'] = str(
            (None if orphaned_retry_projected else record.get('lifecycle_state'))
            or derive_response_lifecycle_state(payload, requested_status=payload.get('status'))
            or payload.get('status')
            or 'completed'
        )
        if record.get('error_message'):
            payload['error'] = {'message': str(record.get('error_message') or '').strip()}
        payload = _attach_response_artifact_bundles_from_registry(payload)
        status_record = dict(record)
        status_record['response_payload'] = dict(payload)
        if isinstance(payload.get('response_frame'), Mapping):
            if orphaned_retry_projected:
                return _attach_response_lookup_state_version(
                    _finalize_response_frame_payload(payload, persist=False),
                    status_record=status_record,
                )
            # A response already bound to a frozen frame is canonical whether
            # it came from the durable ledger or the live lookup registry.
            # Re-freezing a live frame with no original request payload can
            # silently discard exact request/ghost-preview truth.
            payload = _attach_response_status_semantics(payload)
            payload['response_frame'] = _enrich_response_frame_metadata(payload['response_frame'])
            return _attach_response_lookup_state_version(
                _hoist_response_output_surfaces(payload),
                status_record=status_record,
            )
        return _attach_response_lookup_state_version(
            _finalize_response_frame_payload(payload, persist=False),
            status_record=status_record,
        )

    response_payload = _build_canonical_response_payload(
        instance_id=str(record.get('instance_id') or '').strip(),
        model_name=str(record.get('model_name') or '').strip(),
        backend=str(record.get('backend') or '').strip(),
        capability=str(record.get('capability') or '').strip(),
        mode=str(record.get('mode') or 'chat').strip() or 'chat',
        output_text=str(record.get('output_text') or ''),
        source_payload={},
        route_payload=record.get('route_payload') if isinstance(record.get('route_payload'), dict) else None,
        response_id=str(record.get('id') or '').strip() or None,
        message_id=str(record.get('message_id') or '').strip() or None,
    )
    response_payload['status'] = str(record.get('status') or 'in_progress')
    response_payload['lifecycle_state'] = str(
        record.get('lifecycle_state')
        or derive_response_lifecycle_state(response_payload, requested_status=response_payload.get('status'))
    )
    if response_payload.get('output') and isinstance(response_payload['output'], list):
        first_item = response_payload['output'][0]
        if isinstance(first_item, dict):
            first_item['status'] = 'completed' if response_payload['status'] == 'completed' else 'in_progress'
    if record.get('error_message'):
        response_payload['error'] = {'message': str(record.get('error_message') or '').strip()}
    response_payload = _attach_response_artifact_bundles_from_registry(response_payload)
    status_record = dict(record)
    status_record['response_payload'] = dict(response_payload)
    return _attach_response_lookup_state_version(
        _finalize_response_frame_payload(response_payload, persist=False),
        status_record=status_record,
    )


def _collect_artifact_refs_for_bundle_hydration(value: Any, refs: Optional[set[str]] = None, *, depth: int = 0) -> set[str]:
    collected = refs if refs is not None else set()
    if depth > 8:
        return collected
    if isinstance(value, list):
        for item in value:
            _collect_artifact_refs_for_bundle_hydration(item, collected, depth=depth + 1)
        return collected
    if not isinstance(value, Mapping):
        return collected
    for key in ('artifact_ref', 'artifactRef', 'ref'):
        token = str(value.get(key) or '').strip()
        if token:
            collected.add(token)
    for key in ('artifacts', 'outputs', 'output', 'response_frame', 'results'):
        child = value.get(key)
        if isinstance(child, (Mapping, list)):
            _collect_artifact_refs_for_bundle_hydration(child, collected, depth=depth + 1)
    return collected


def _hydrate_bundle_payload_artifacts_from_registry(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    artifacts = list(updated.get('artifacts') or []) if isinstance(updated.get('artifacts'), list) else []
    seen_paths = {
        str(item.get('path') or item.get('source_path') or '').strip()
        for item in artifacts
        if isinstance(item, Mapping)
    }
    refs = _collect_artifact_refs_for_bundle_hydration(updated)
    for artifact_ref in refs:
        try:
            registry_record = _find_artifact_registry_record_by_artifact_ref(
                artifact_ref,
                ledger_path=ARTIFACT_REGISTRY_LEDGER,
            )
        except Exception as exc:  # noqa: BLE001
            logging.debug('Could not resolve bundle artifact ref %s from registry: %s', artifact_ref, exc)
            continue
        registry_artifact = (
            registry_record.get('artifact')
            if isinstance(registry_record, Mapping) and isinstance(registry_record.get('artifact'), Mapping)
            else None
        )
        if not registry_artifact:
            continue
        path = str(registry_artifact.get('path') or registry_artifact.get('source_path') or '').strip()
        if not path or path in seen_paths:
            continue
        artifacts.append(dict(registry_artifact))
        seen_paths.add(path)
    if artifacts:
        updated['artifacts'] = artifacts
    return updated


def _response_artifact_bundle_requires_canonical_payload(payload: Mapping[str, Any]) -> bool:
    projection = (
        payload.get('wire_projection')
        if isinstance(payload.get('wire_projection'), Mapping)
        else {}
    )
    source = str(projection.get('source') or '').strip()
    if source and source not in {
        'current_index_compact_ledger_frame',
        'in_memory_inline_below_limit',
    }:
        return True
    truncation = (
        projection.get('public_projection_truncation')
        if isinstance(projection.get('public_projection_truncation'), Mapping)
        else {}
    )
    if any(
        value > 0
        for value in truncation.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        return True
    if payload.get('batch_results_truncated') is True:
        return True
    if any(
        payload.get(key) is True
        for key in (
            'artifacts_projection_truncated',
            'outputs_projection_truncated',
            'output_slots_projection_truncated',
            'output_branches_projection_truncated',
        )
    ):
        return True
    response_frame = (
        payload.get('response_frame')
        if isinstance(payload.get('response_frame'), Mapping)
        else {}
    )
    frame_current = (
        response_frame.get('current_state')
        if isinstance(response_frame.get('current_state'), Mapping)
        else {}
    )
    frame_artifacts = (
        response_frame.get('artifacts')
        if isinstance(response_frame.get('artifacts'), Mapping)
        else {}
    )
    relevant_refs = (
        payload.get('artifacts_snapshot_ref'),
        frame_current.get('artifacts_snapshot_ref'),
        frame_artifacts.get('output_snapshot_ref'),
    )
    if any(
        isinstance(ref, Mapping)
        and str(ref.get('projection_role') or '').strip() == 'public_body_exact'
        for ref in relevant_refs
    ):
        return True
    omitted = (
        projection.get('omitted_payloads')
        if isinstance(projection.get('omitted_payloads'), Mapping)
        else {}
    )
    return any(key in omitted for key in ('artifacts_tail', 'outputs_tail'))


def _load_response_artifact_bundle_source_payload(
    response_id: str,
    *,
    hydrate_registry: bool = True,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], int]:
    """Load exact response-owned artifact truth without creating bundle state."""
    record, error, status_code = _get_bounded_response_lookup_record(response_id)
    if error:
        public_error = 'Response not found.'
        if error.get('code') and error.get('code') != 'response_frame_not_found':
            public_error = error.get('message') or public_error
        return None, {'error': public_error, 'error_detail': error}, status_code
    if not record:
        return None, {'error': 'Response not found.'}, 404

    bounded_payload = (
        copy.deepcopy(record.get('response_payload'))
        if isinstance(record.get('response_payload'), Mapping)
        else {}
    )
    if not bounded_payload:
        return None, {'error': 'Response has no artifact payload to bundle.'}, 409
    if _response_artifact_bundle_requires_canonical_payload(bounded_payload):
        expected_frame_identity = _response_wire_frame_identity(bounded_payload)
        if expected_frame_identity is None:
            return None, {
                'error': 'Exact response artifact truth cannot be bound to a frozen frame.',
                'error_detail': {
                    'code': 'response_artifact_bundle_frame_binding_missing',
                },
            }, 409
        canonical_state = _load_latest_response_state(
            _normalize_response_lookup_id(response_id),
            frames_dir=RESPONSE_FRAMES_DIR,
        )
        canonical_payload = (
            canonical_state.get('response_payload')
            if isinstance(canonical_state.get('response_payload'), Mapping)
            else None
        )
        if canonical_state.get('ok') is not True or not canonical_payload:
            return None, {
                'error': 'Exact response artifact truth is unavailable for complete bundling.',
                'error_detail': canonical_state.get('error') or {
                    'code': 'response_artifact_bundle_exact_truth_unavailable',
                },
            }, 409
        canonical_frame_identity = _response_wire_frame_identity(canonical_payload)
        if canonical_frame_identity != expected_frame_identity:
            return None, {
                'error': 'Response artifact truth changed before complete bundling.',
                'error_detail': {
                    'code': 'response_artifact_bundle_frame_binding_mismatch',
                    'expected_frame_id': expected_frame_identity[0],
                    'expected_frame_sequence': expected_frame_identity[1],
                    'canonical_frame_id': canonical_frame_identity[0]
                    if canonical_frame_identity
                    else None,
                    'canonical_frame_sequence': canonical_frame_identity[1]
                    if canonical_frame_identity
                    else None,
                },
            }, 409
        bounded_payload = copy.deepcopy(dict(canonical_payload))
    response_payload = (
        _hydrate_bundle_payload_artifacts_from_registry(bounded_payload)
        if hydrate_registry
        else bounded_payload
    )
    return response_payload, None, 200


def _persist_response_artifact_bundle_record(bundle_payload: Mapping[str, Any]) -> None:
    record = _build_response_artifact_bundle_registry_record(bundle_payload)
    _persist_artifact_registry_record(record, ledger_path=ARTIFACT_REGISTRY_LEDGER)


_GENERATED_IMAGE_POSTPROCESS = GeneratedImagePostprocessOwner(
    runtime_status_path_getter=lambda: RUNTIME_STATUS_PATH,
    artifact_registry_ledger_getter=lambda: ARTIFACT_REGISTRY_LEDGER,
    load_running_instances=lambda: load_running_instances(),
    merge_instances_with_runtime_status=lambda instances, **kwargs: merge_instances_with_runtime_status(instances, **kwargs),
    build_instance_trait_summary=lambda instance: _build_instance_trait_summary(instance),
    normalize_capability=lambda value: normalize_capability(value),
    normalize_backend=lambda value: normalize_backend(value),
    capability_vision_analysis=CAPABILITY_VISION_ANALYSIS,
    capability_chat=CAPABILITY_CHAT,
    parse_image_state_response=lambda content, describer_instance_id=None, describer_model=None: parse_image_state_response(
        content,
        describer_instance_id=describer_instance_id,
        describer_model=describer_model,
    ),
    invoke_internal_api_json_route=lambda payload: _invoke_internal_api_json_route(payload=payload),
    get_cached_generated_image_state=lambda raw_path: _get_cached_generated_image_state(raw_path),
    store_cached_generated_image_state=lambda raw_path, image_state: _store_cached_generated_image_state(raw_path, image_state),
    build_image_state_enrichment_state=lambda **kwargs: _build_image_state_enrichment_state(**kwargs),
    attach_cached_generated_image_state_to_response_lookups=lambda raw_path, image_state: _attach_cached_generated_image_state_to_response_lookups(raw_path, image_state),
    claim_generated_image_state_enrichment=lambda raw_path: _claim_generated_image_state_enrichment(raw_path),
    release_generated_image_state_enrichment=lambda raw_path: _release_generated_image_state_enrichment(raw_path),
    extract_semantic_materializer_prompt=lambda payload, capability: _extract_semantic_materializer_prompt(payload, capability=capability),
    compact_request_meta=lambda payload: compact_request_meta(payload),
    extract_request_meta=lambda payload: extract_request_meta(payload),
    build_generated_image_provenance=lambda **kwargs: _build_generated_image_provenance(**kwargs),
    persist_generated_image_provenance=lambda record, ledger_path=None: _persist_generated_image_provenance(record, ledger_path=ledger_path),
    persist_artifact_registry_enrichment=lambda **kwargs: _persist_artifact_registry_enrichment(**kwargs),
    coerce_seed=lambda value: _coerce_seed(value),
    schedule_post_response_substrate_hygiene=lambda *args, **kwargs: _schedule_post_response_substrate_hygiene(*args, **kwargs),
)


def _pick_image_state_helper_instance(instances: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return _GENERATED_IMAGE_POSTPROCESS.pick_image_state_helper_instance(instances)


def _build_image_state_for_generated_image(image_path: str) -> Optional[dict[str, Any]]:
    return _GENERATED_IMAGE_POSTPROCESS.build_image_state_for_generated_image(image_path)


def _background_enrich_generated_image_payload(image_path: str) -> None:
    _GENERATED_IMAGE_POSTPROCESS._background_enrich_generated_image_payload(image_path)


def _schedule_generated_image_payload_enrichment(payload: dict[str, Any]) -> dict[str, Any]:
    return _GENERATED_IMAGE_POSTPROCESS.schedule_generated_image_payload_enrichment(payload)


def _enrich_generated_image_payload(payload: dict[str, Any], *, blocking: bool = False) -> dict[str, Any]:
    return _GENERATED_IMAGE_POSTPROCESS.enrich_generated_image_payload(payload, blocking=blocking)


def _coerce_seed(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    token = str(value or '').strip()
    if not token or not token.isdigit():
        return None
    parsed = int(token)
    return parsed if parsed >= 0 else None


def _find_image_artifact_seed(messages: list[dict[str, Any]], artifact_path: str) -> Optional[int]:
    target_path = str(artifact_path or '').strip()
    if not target_path:
        return None
    for message in reversed(sanitize_ghost_messages(messages)):
        for artifact in message.get('artifacts') or []:
            if not isinstance(artifact, dict):
                continue
            if str(artifact.get('type') or '').strip() != 'image':
                continue
            if str(artifact.get('path') or '').strip() != target_path:
                continue
            seed = _coerce_seed(artifact.get('seed'))
            if seed is not None:
                return seed
    return None


def _rewind_upload_stream(upload) -> None:
    if not upload or not getattr(upload, 'filename', None):
        return
    stream = getattr(upload, 'stream', None)
    if stream is None:
        return
    try:
        stream.seek(0)
    except Exception:  # noqa: BLE001
        pass


def _coerce_internal_json_result(result: Any) -> tuple[dict, int]:
    response = app.make_response(result)
    data = response.get_json(silent=True)
    if not isinstance(data, dict):
        data = {'error': response.get_data(as_text=True) or 'Unexpected response payload.'}
    return data, response.status_code


def _invoke_internal_api_json_route(
    path: Optional[str] = None,
    *,
    payload: Optional[dict] = None,
    upload=None,
) -> tuple[dict, int]:
    def _invoke() -> tuple[dict, int]:
        if path in (None, '', '/api/infer'):
            return _coerce_internal_json_result(
                _execute_infer_request(dict(payload or {}), upload=upload)
            )
        if path == '/api/chat':
            return _coerce_internal_json_result(
                _execute_chat_request(dict(payload or {}))
            )
        raise ValueError(f"Unsupported internal route '{path}'.")

    if has_app_context():
        return _invoke()
    with app.app_context():
        return _invoke()


_execute_chat_backend_request = _BACKEND_TRANSPORT_RUNTIME.execute_chat_backend_request


def _embedding_endpoint_url(instance: dict[str, Any], target_port: int, transport: str) -> str:
    return _GHOST_ROUTE_RUNTIME._embedding_endpoint_url(instance, target_port, transport)


def _parse_embedding_backend_response(payload: Any, transport: str) -> list[list[float]]:
    return _GHOST_ROUTE_RUNTIME._parse_embedding_backend_response(payload, transport)


def _execute_embedding_backend_request(
    *,
    target_port: int,
    model_name: str,
    backend: str,
    inputs: list[str],
    request_model_override: Optional[str] = None,
    embedding_transport: Optional[str] = None,
    instance: Optional[dict[str, Any]] = None,
) -> list[list[float]]:
    return _GHOST_ROUTE_RUNTIME._execute_embedding_backend_request(
        target_port=target_port,
        model_name=model_name,
        backend=backend,
        inputs=inputs,
        request_model_override=request_model_override,
        embedding_transport=embedding_transport,
        instance=instance,
    )


def _attach_embedding_hints_to_route_context(
    route_context: dict[str, Any],
    *,
    runtime_manifest: dict[str, Any],
    instances: list[dict[str, Any]],
) -> None:
    _GHOST_ROUTE_RUNTIME.attach_embedding_hints_to_route_context(
        route_context,
        runtime_manifest=runtime_manifest,
        instances=instances,
    )


_extract_stream_delta_payload = _BACKEND_TRANSPORT_RUNTIME.extract_stream_delta_payload
_normalize_stream_line = _BACKEND_TRANSPORT_RUNTIME.normalize_stream_line
_open_ollama_chat_stream = _BACKEND_TRANSPORT_RUNTIME.open_ollama_chat_stream
_iter_ollama_stream_deltas = _BACKEND_TRANSPORT_RUNTIME.iter_ollama_stream_deltas
_open_openai_chat_stream = _BACKEND_TRANSPORT_RUNTIME.open_openai_chat_stream
_request_exception_details = _BACKEND_TRANSPORT_RUNTIME.request_exception_details
_iter_openai_stream_deltas = _BACKEND_TRANSPORT_RUNTIME.iter_openai_stream_deltas
_iter_mlx_stream_deltas = _BACKEND_TRANSPORT_RUNTIME.iter_mlx_stream_deltas


def _stream_chat_backend_as_responses(
    *,
    instance_id: str,
    target_port: int,
    model_name: str,
    backend: str,
    capability: str,
    messages: list[dict],
    request_model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    route_payload: Optional[dict[str, Any]] = None,
    response_id: Optional[str] = None,
    request_payload: Optional[dict[str, Any]] = None,
    artifact_prompt: Optional[str] = None,
):
    return _CHAT_RUNTIME.stream_chat_backend_as_responses(
        instance_id=instance_id,
        target_port=target_port,
        model_name=model_name,
        backend=backend,
        capability=capability,
        messages=messages,
        request_model_override=request_model_override,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        route_payload=route_payload,
        response_id=response_id,
        request_payload=request_payload,
        artifact_prompt=artifact_prompt,
    )


def _file_kind_from_name(filename: str) -> str:
    return file_kind_from_name(filename)


def _save_upload_to_temp(upload) -> Path:
    return save_upload_to_temp(upload)


def _normalize_local_path_input(raw_path: str) -> str:
    return normalize_local_path_input(raw_path)


def _resolve_existing_local_path(raw_path: str) -> Path:
    return resolve_existing_local_path(raw_path)


def _save_local_path_to_temp(raw_path: str) -> tuple[Path, Path]:
    return save_local_path_to_temp(raw_path)


def _persist_input_file_locally(
    source_path: Path,
    *,
    source_name: str,
    file_kind: str,
    output_root: Path = SAVED_INPUTS_DIR,
) -> Optional[str]:
    return persist_input_file_locally(
        source_path,
        source_name=source_name,
        file_kind=file_kind,
        output_root=output_root,
    )


def _expand_local_paths(raw_paths: list[str], *, max_items: int = 1000) -> tuple[list[str], list[dict], bool]:
    return expand_local_paths(raw_paths, max_items=max_items)


def _read_text_file(path: Path, max_bytes: int = 250_000) -> tuple[str, bool, int, int]:
    return read_text_file_with_metadata(path, max_bytes=max_bytes)


def _to_base64(path: Path) -> str:
    return to_base64(path)


def _parse_bool(raw_value: Any, *, default: bool = False) -> bool:
    return parse_bool(raw_value, default=default)


def _parse_int_with_bounds(
    raw_value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return parse_int_with_bounds(
        raw_value,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _parse_float_with_bounds(
    raw_value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    token = str(raw_value or '').strip()
    if not token:
        return default
    try:
        parsed = float(token)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{raw_value}'.") from exc
    return max(minimum, min(maximum, parsed))


def _hash_file_sha256(path: Path) -> str:
    return hash_file_sha256(path)


def _truncate_for_history(text: str, max_chars: int = 120_000) -> str:
    return _INFER_SUPPORT_RUNTIME.truncate_for_history(text, max_chars=max_chars)


def _read_infer_history(limit: int = 200) -> list[dict]:
    return _INFER_SUPPORT_RUNTIME.read_infer_history(limit=limit)


def _append_infer_history(entry: dict) -> None:
    _INFER_SUPPORT_RUNTIME.append_infer_history(entry)


def _find_cached_pdf_insight(
    *,
    file_sha256: str,
    model_name: str,
    backend: str,
    capability: str,
    prompt: str,
) -> Optional[dict]:
    return _INFER_SUPPORT_RUNTIME.find_cached_pdf_insight(
        file_sha256=file_sha256,
        model_name=model_name,
        backend=backend,
        capability=capability,
        prompt=prompt,
        read_infer_history_fn=_read_infer_history,
        looks_like_ocr_prompt_echo_fn=_looks_like_ocr_prompt_echo,
    )


def _log_pdf_infer_event(
    *,
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    prompt: str,
    file_name: str,
    file_sha256: str,
    status: str,
    mode: Optional[str] = None,
    content: Optional[str] = None,
    error: Optional[str] = None,
    warnings: Optional[list[str]] = None,
    pdf_source: Optional[str] = None,
    pdf_total_pages: Optional[int] = None,
    pdf_processed_pages: Optional[int] = None,
    artifact_path: Optional[str] = None,
) -> None:
    _INFER_SUPPORT_RUNTIME.log_pdf_infer_event(
        instance_id=instance_id,
        model_name=model_name,
        backend=backend,
        capability=capability,
        prompt=prompt,
        file_name=file_name,
        file_sha256=file_sha256,
        status=status,
        mode=mode,
        content=content,
        error=error,
        warnings=warnings,
        pdf_source=pdf_source,
        pdf_total_pages=pdf_total_pages,
        pdf_processed_pages=pdf_processed_pages,
        artifact_path=artifact_path,
        truncate_for_history_fn=_truncate_for_history,
        append_infer_history_fn=_append_infer_history,
    )


def _persist_generated_image_provenance_for_infer_result(
    payload: Any,
    *,
    request_payload: Optional[dict[str, Any]],
    instance_id: str,
    model_name: str,
    backend: str,
    capability: str,
    user_prompt: str,
    prompt: str,
    raw_file_path: str,
    input_artifacts: list[dict[str, Any]],
    reference_artifacts: Any,
    image_width: Optional[int],
    image_height: Optional[int],
    image_seed: Optional[int],
) -> Optional[dict[str, Any]]:
    return _GENERATED_IMAGE_POSTPROCESS.persist_generated_image_provenance_for_infer_result(
        payload,
        request_payload=request_payload,
        instance_id=instance_id,
        model_name=model_name,
        backend=backend,
        capability=capability,
        user_prompt=user_prompt,
        prompt=prompt,
        raw_file_path=raw_file_path,
        input_artifacts=input_artifacts,
        reference_artifacts=reference_artifacts,
        image_width=image_width,
        image_height=image_height,
        image_seed=image_seed,
    )


def _build_infer_dedupe_key(
    *,
    instance_id: str,
    backend: str,
    capability: str,
    model_name: str,
    prompt: str,
    upload,
    local_file_path: str = "",
) -> str:
    return build_infer_dedupe_key(
        instance_id=instance_id,
        backend=backend,
        capability=capability,
        model_name=model_name,
        prompt=prompt,
        upload=upload,
        local_file_path=local_file_path,
    )


def _acquire_infer_slot(key: str, *, now: Optional[float] = None, ttl_sec: int = INFER_SLOT_TTL_SEC) -> bool:
    return acquire_infer_slot(
        key,
        slots=_INFER_INFLIGHT,
        lock=_INFER_INFLIGHT_LOCK,
        now=now,
        ttl_sec=ttl_sec,
    )


def _release_infer_slot(key: Optional[str]) -> None:
    release_infer_slot(key, slots=_INFER_INFLIGHT, lock=_INFER_INFLIGHT_LOCK)


_extract_pdf_text_content = _OCR_PDF_RUNTIME.extract_pdf_text_content
_render_pdf_pages_to_base64 = _OCR_PDF_RUNTIME.render_pdf_pages_to_base64
_render_single_pdf_page_to_base64 = _OCR_PDF_RUNTIME.render_single_pdf_page_to_base64
_looks_like_ocr_prompt_echo = _OCR_PDF_RUNTIME.looks_like_ocr_prompt_echo
_normalize_ocr_line = _OCR_PDF_RUNTIME.normalize_ocr_line
_strip_ocr_structural_lines = _OCR_PDF_RUNTIME.strip_ocr_structural_lines
_collapse_repeated_ocr_lines = _OCR_PDF_RUNTIME.collapse_repeated_ocr_lines
_line_has_ocr_garbage_pattern = _OCR_PDF_RUNTIME.line_has_ocr_garbage_pattern
_sanitize_ocr_noise_lines = _OCR_PDF_RUNTIME.sanitize_ocr_noise_lines
_detect_low_quality_ocr_reason = _OCR_PDF_RUNTIME.detect_low_quality_ocr_reason
_clean_ocr_output_text = _OCR_PDF_RUNTIME.clean_ocr_output_text
_ocr_pdf_page_with_ollama = _OCR_PDF_RUNTIME.ocr_pdf_page_with_ollama
_is_generic_ocr_instruction_prompt = _OCR_PDF_RUNTIME.is_generic_ocr_instruction_prompt
_ocr_image_with_deepseek = _OCR_PDF_RUNTIME.ocr_image_with_deepseek


_ollama_generate = _BACKEND_TRANSPORT_RUNTIME.ollama_generate
_extract_generate_content = _BACKEND_TRANSPORT_RUNTIME.extract_generate_content
_locate_saved_image_file_from_generate_output = _BACKEND_TRANSPORT_RUNTIME.locate_saved_image_file_from_generate_output
_extract_image_data_url_from_generate_output = _BACKEND_TRANSPORT_RUNTIME.extract_image_data_url_from_generate_output
_extract_saved_image_path_from_generate_output = _BACKEND_TRANSPORT_RUNTIME.extract_saved_image_path_from_generate_output
_extract_generate_seed = _BACKEND_TRANSPORT_RUNTIME.extract_generate_seed
_persist_image_data_url_locally = _BACKEND_TRANSPORT_RUNTIME.persist_image_data_url_locally


def _persist_text_markdown_locally(
    content: Optional[str],
    *,
    model_name: str,
    source_file_name: str,
    mode: str,
) -> Optional[str]:
    return persist_text_markdown_locally(
        content,
        model_name=model_name,
        source_file_name=source_file_name,
        mode=mode,
        output_dir=OCR_EXPORT_DIR,
    )


def _persist_transcript_text_locally(
    content: Optional[str],
    *,
    model_name: str,
    source_file_name: str,
    mode: str,
) -> Optional[str]:
    return persist_text_markdown_locally(
        content,
        model_name=model_name,
        source_file_name=source_file_name,
        mode=mode,
        output_dir=TRANSCRIPT_EXPORT_DIR,
    )


def _persist_text_artifact_locally(
    content: Optional[str],
    *,
    model_name: str,
    source_name: str,
    mode: str,
    extension: str,
    target_path: Optional[str] = None,
) -> Optional[str]:
    return persist_text_artifact_locally(
        content,
        model_name=model_name,
        source_name=source_name,
        mode=mode,
        extension=extension,
        output_dir=ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
        target_path=target_path,
    )


_TEXT_ARTIFACT_SOURCE_MIME_TYPES = {
    'application/ecmascript',
    'application/javascript',
    'application/json',
    'application/sql',
    'application/typescript',
    'application/x-javascript',
    'application/xhtml+xml',
    'application/xml',
    'application/yaml',
    'image/svg+xml',
}
_TEXT_ARTIFACT_SOURCE_TYPES = {
    'csv',
    'document',
    'html',
    'javascript',
    'json',
    'markdown',
    'md',
    'text',
    'txt',
    'xml',
    'yaml',
}


def _text_artifact_source_meta_from_value(value: Any) -> dict[str, str]:
    if value in (None, '', [], {}):
        return {}
    if isinstance(value, str):
        path_text = value.strip()
        extension = normalize_text_artifact_extension(Path(path_text).suffix)
        if not extension:
            return {}
        return {
            'path': path_text,
            'extension': extension,
            'source_name': Path(path_text).stem or f'source-{extension}',
        }
    if not isinstance(value, dict):
        return {}

    path_text = str(
        value.get('path')
        or value.get('source_path')
        or value.get('file_path')
        or value.get('saved_text_path')
        or value.get('url')
        or ''
    ).strip()
    mime_type = str(value.get('mime_type') or value.get('mimetype') or value.get('content_type') or '').strip().lower()
    artifact_type = str(value.get('type') or value.get('kind') or value.get('artifact_type') or '').strip().lower()
    raw_extension = (
        Path(path_text).suffix
        or str(value.get('extension') or value.get('ext') or '').strip()
        or (mimetypes.guess_extension(mime_type) or '')
    )
    extension = normalize_text_artifact_extension(raw_extension)
    source_name = str(
        value.get('source_name')
        or value.get('name')
        or value.get('file_name')
        or value.get('filename')
        or value.get('title')
        or ''
    ).strip()
    if not source_name and path_text:
        source_name = Path(urlparse(path_text).path or path_text).stem

    is_text_source = bool(
        extension
        or artifact_type in _TEXT_ARTIFACT_SOURCE_TYPES
        or mime_type.startswith('text/')
        or mime_type in _TEXT_ARTIFACT_SOURCE_MIME_TYPES
    )
    if not is_text_source:
        return {}
    if not extension and mime_type:
        extension = normalize_text_artifact_extension(mimetypes.guess_extension(mime_type) or '')
    if not extension:
        extension = 'txt'
    return {
        key: val
        for key, val in {
            'path': path_text,
            'extension': extension,
            'source_name': source_name or f'source-{extension}',
        }.items()
        if val
    }


def _request_payload_text_artifact_source(request_payload: Optional[dict[str, Any]]) -> dict[str, str]:
    payload = request_payload if isinstance(request_payload, dict) else {}
    for key in (
        'file_path',
        'input_artifact_path',
        'selected_reference_artifact',
        'selected_reference_artifacts',
        'input_artifacts',
        'reference_artifacts',
    ):
        value = payload.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            source_meta = _text_artifact_source_meta_from_value(item)
            if source_meta:
                return source_meta
    return {}


def _request_payload_has_text_artifact_source(request_payload: Optional[dict[str, Any]]) -> bool:
    return bool(_request_payload_text_artifact_source(request_payload))


def _persist_generated_text_artifact_if_requested(
    content: Optional[str],
    *,
    prompt: str,
    model_name: str,
    mode: str,
    request_payload: Optional[dict[str, Any]] = None,
    source_available: Optional[bool] = None,
) -> dict[str, Any]:
    source_meta = _request_payload_text_artifact_source(request_payload)
    has_source = (
        bool(source_available)
        if source_available is not None
        else bool(source_meta)
    )
    artifact_requests = detect_text_artifact_requests(
        prompt,
        source_available=has_source,
        source_extension=source_meta.get('extension'),
        source_name=source_meta.get('source_name'),
        source_path=source_meta.get('path'),
    )
    artifact_payloads = extract_text_artifact_payloads(str(content or ''), artifact_requests)
    if not artifact_payloads:
        return {}
    predecessor_edit_targets: dict[tuple[str, str], str] = {}
    predecessor_context = (
        request_payload.get('current_predecessor_context')
        if isinstance(request_payload, dict)
        and isinstance(request_payload.get('current_predecessor_context'), dict)
        else {}
    )
    if (
        predecessor_context.get('status') == 'authorized'
        and predecessor_context.get('authorization')
        == 'canonical_same_conversation_predecessor'
        and predecessor_context.get('promotion_mode') == 'named_text_edit'
    ):
        ambiguous_target_keys: set[tuple[str, str]] = set()
        for item in predecessor_context.get('matched_text_artifacts') or []:
            if not isinstance(item, dict):
                continue
            filename = Path(str(item.get('filename') or '')).name
            target_path = str(item.get('path') or '').strip()
            identity = (
                Path(filename).suffix.lower().lstrip('.'),
                Path(filename).stem.lower(),
            )
            if not all(identity) or not target_path:
                continue
            if identity in predecessor_edit_targets:
                ambiguous_target_keys.add(identity)
                predecessor_edit_targets.pop(identity, None)
                continue
            if identity not in ambiguous_target_keys:
                predecessor_edit_targets[identity] = target_path
    saved_text_artifacts: list[dict[str, Any]] = []
    preservation_rejections: list[dict[str, Any]] = []
    for artifact_payload in artifact_payloads:
        artifact_request = (
            artifact_payload.get('artifact_request')
            if isinstance(artifact_payload, dict)
            else {}
        )
        artifact_content = str((artifact_payload or {}).get('content') or '').strip()
        if not artifact_content or not isinstance(artifact_request, dict):
            continue
        request_identity = (
            str(artifact_request.get('extension') or '').strip().lower().lstrip('.'),
            str(artifact_request.get('source_name') or '').strip().lower(),
        )
        predecessor_target_path = predecessor_edit_targets.get(request_identity)
        if predecessor_target_path:
            artifact_request = {
                **artifact_request,
                'source': 'selected_source_edit',
                'target_path': predecessor_target_path,
            }
        revision_preservation_evidence: dict[str, Any] = {}
        revision_write_proof: dict[str, Any] = {}
        target_path = str(artifact_request.get('target_path') or '').strip()
        revision_required = bool(
            target_path
            and str(artifact_request.get('source') or '').strip().lower()
            in {'selected_source_edit', 'canonical_predecessor_artifact'}
        )
        resolved_source = (
            _resolve_saved_text_artifact_path(target_path)
            if revision_required
            else None
        )
        source_content = ''
        source_sha256 = ''
        if resolved_source:
            try:
                source_bytes = resolved_source.read_bytes()
                source_content = source_bytes.decode('utf-8', errors='replace')
                source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            except OSError:
                source_content = ''
                source_sha256 = ''
        if (
            revision_required
            and _LATE_FILL_RUNTIME._text_artifact_revision_preservation_requested(prompt)
        ):
            revision_preservation_evidence, preservation_error = (
                _LATE_FILL_RUNTIME._text_artifact_revision_preservation_review(
                    source_content,
                    artifact_content,
                    extension=str(artifact_request.get('extension') or ''),
                    target_path=target_path,
                )
            )
            if preservation_error:
                preservation_rejections.append(preservation_error)
                continue
            artifact_request = {
                **artifact_request,
                'text_artifact_revision_preservation_required': True,
                'text_artifact_revision_preservation_policy': (
                    'structural_anchor_retention_v1'
                ),
            }
        saved_text_path = _persist_text_artifact_locally(
            artifact_content,
            model_name=model_name,
            source_name=artifact_request.get('source_name') or 'generated-text',
            mode=mode,
            extension=artifact_request.get('extension') or 'txt',
            target_path=target_path or None,
        )
        if not saved_text_path:
            continue
        if revision_required:
            output_sha256 = ''
            try:
                output_sha256 = hashlib.sha256(
                    Path(saved_text_path).read_bytes()
                ).hexdigest()
            except OSError:
                output_sha256 = ''
            revision_write_proof = {
                key: value
                for key, value in {
                    'kind': 'ollmo.text_artifact_revision_write_proof',
                    'version': 1,
                    'status': 'applied',
                    'target_path': target_path,
                    'source_sha256': source_sha256 or None,
                    'output_sha256': output_sha256 or None,
                    'evidence': 'current_response_output_written_to_target',
                }.items()
                if value not in (None, '')
            }
        saved_text_artifacts.append(
            {
                'path': saved_text_path,
                'text_artifact_request': artifact_request,
                'document_output_kind': 'document',
                **(
                    {
                        'text_artifact_revision_required': True,
                        'text_artifact_revision_write_proof': revision_write_proof,
                    }
                    if revision_write_proof
                    else {}
                ),
                **(
                    {
                        'text_artifact_revision_preservation_evidence': (
                            revision_preservation_evidence
                        )
                    }
                    if revision_preservation_evidence
                    else {}
                ),
            }
        )
    if not saved_text_artifacts:
        return (
            {'text_artifact_revision_preservation_rejections': preservation_rejections}
            if preservation_rejections
            else {}
        )
    first_artifact = saved_text_artifacts[0]
    return {
        'saved_text_path': first_artifact['path'],
        'saved_text_artifacts': saved_text_artifacts,
        'document_output_kind': 'document',
        'text_artifact_request': first_artifact.get('text_artifact_request'),
        'text_artifact_requests': [
            item.get('text_artifact_request')
            for item in saved_text_artifacts
            if isinstance(item.get('text_artifact_request'), dict)
        ],
        **(
            {
                'text_artifact_revision_write_proof': first_artifact.get(
                    'text_artifact_revision_write_proof'
                )
            }
            if isinstance(
                first_artifact.get('text_artifact_revision_write_proof'),
                dict,
            )
            else {}
        ),
        **(
            {
                'text_artifact_revision_preservation_evidence': first_artifact.get(
                    'text_artifact_revision_preservation_evidence'
                )
            }
            if isinstance(
                first_artifact.get('text_artifact_revision_preservation_evidence'),
                dict,
            )
            else {}
        ),
        **(
            {'text_artifact_revision_preservation_rejections': preservation_rejections}
            if preservation_rejections
            else {}
        ),
    }


def _is_path_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_generated_image_path(raw_path: str) -> Optional[Path]:
    allowed_roots = expand_repo_relative_roots(
        ARTIFACT_OUTPUTS_IMAGES_DIR,
    )
    return _resolve_saved_artifact_path(raw_path, allowed_roots=allowed_roots)


def _resolve_saved_artifact_path(raw_path: str, *, allowed_roots: set[Path]) -> Optional[Path]:
    return resolve_saved_artifact_path(raw_path, allowed_roots=allowed_roots)


def _resolve_saved_text_artifact_path(raw_path: str) -> Optional[Path]:
    allowed_roots = expand_repo_relative_roots(
        ARTIFACT_OUTPUTS_OCR_DIR,
        ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
        ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
    )
    return _resolve_saved_artifact_path(raw_path, allowed_roots=allowed_roots)


def _resolve_saved_downloadable_artifact_path(raw_path: str) -> Optional[Path]:
    allowed_roots = expand_repo_relative_roots(
        ARTIFACT_OUTPUTS_AUDIO_DIR,
        ARTIFACT_OUTPUTS_IMAGES_DIR,
        ARTIFACT_OUTPUTS_OCR_DIR,
        ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
        ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
        SETTINGS_ARTIFACTS_DIR,
        ARTIFACT_INPUTS_ROOT,
    )
    return _resolve_saved_artifact_path(raw_path, allowed_roots=allowed_roots)


def _saved_viewable_artifact_roots() -> set[Path]:
    return expand_repo_relative_roots(
        ARTIFACT_OUTPUTS_AUDIO_DIR,
        ARTIFACT_OUTPUTS_IMAGES_DIR,
        ARTIFACT_OUTPUTS_OCR_DIR,
        ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
        ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
        SETTINGS_ARTIFACTS_DIR,
        ARTIFACT_INPUTS_ROOT,
        ARTIFACT_BUNDLES_DIR,
    )


def _resolve_existing_path_under_roots(raw_path: str, *, allowed_roots: set[Path], allow_directory: bool) -> Optional[Path]:
    raw_value = str(raw_path or '').strip()
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, Path(__file__).resolve().parent / candidate]
    for item in candidates:
        try:
            resolved = item.resolve(strict=False)
        except OSError:
            continue
        if not resolved.exists():
            continue
        if resolved.is_dir() and not allow_directory:
            continue
        if not resolved.is_dir() and not resolved.is_file():
            continue
        if any(is_path_within(resolved, root) for root in allowed_roots):
            return resolved
    return None


def _resolve_saved_openable_artifact_path(raw_path: str) -> Optional[Path]:
    allowed_roots = expand_repo_relative_roots(
        ARTIFACT_OUTPUTS_AUDIO_DIR,
        ARTIFACT_OUTPUTS_IMAGES_DIR,
        ARTIFACT_OUTPUTS_OCR_DIR,
        ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR,
        ARTIFACT_OUTPUTS_DOCUMENTS_DIR,
        SETTINGS_ARTIFACTS_DIR,
        ARTIFACT_INPUTS_ROOT,
        ARTIFACT_BUNDLES_DIR,
    )
    return _resolve_existing_path_under_roots(raw_path, allowed_roots=allowed_roots, allow_directory=True)


def _resolve_saved_viewable_artifact_path(raw_path: str) -> Optional[Path]:
    return _resolve_saved_artifact_path(raw_path, allowed_roots=_saved_viewable_artifact_roots())


_SAVED_IMAGE_PATH_ERROR = (
    "Invalid file path. Only existing files under artifacts/images/ are allowed."
)
_SAVED_DOWNLOADABLE_ARTIFACT_PATH_ERROR = (
    "Invalid file path. Only existing files under artifacts/audio/, "
    "artifacts/images/, artifacts/ocr/, artifacts/transcripts/, "
    "artifacts/documents/, artifacts/settings/, or artifacts/inputs/ are allowed."
)
_SAVED_OPENABLE_ARTIFACT_PATH_ERROR = (
    "Invalid file path. Only existing files or folders under artifacts/audio/, "
    "artifacts/images/, artifacts/ocr/, artifacts/transcripts/, artifacts/documents/, "
    "artifacts/settings/, artifacts/inputs/, or artifacts/bundles/ are allowed."
)
_SAVED_VIEWABLE_ARTIFACT_PATH_ERROR = (
    "Invalid file path. Only existing files under artifacts/audio/, artifacts/images/, "
    "artifacts/ocr/, artifacts/transcripts/, artifacts/documents/, artifacts/settings/, "
    "artifacts/inputs/, or artifacts/bundles/ are allowed."
)
_SAVED_INTERACTIVE_PREVIEW_PATH_ERROR = (
    "Invalid HTML preview path. Only existing HTML files under Ollmo artifact roots are allowed."
)
_SAVED_INTERACTIVE_PREVIEW_TOKEN_VERSION = b'v1'
_SAVED_INTERACTIVE_PREVIEW_SIGNING_KEY = secrets.token_bytes(32)


def _saved_html_preview_temp_subdir(name: str) -> Path:
    root = Path(_SAVED_HTML_PREVIEW_TEMP_ROOT).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = (root / name).resolve(strict=False)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    return target


def _delete_saved_html_preview_package_path(raw_path: Any) -> None:
    try:
        path = Path(str(raw_path or '')).expanduser().resolve(strict=False)
        packages_root = _saved_html_preview_temp_subdir('packages')
    except (OSError, ValueError):
        return
    if path == packages_root or not is_path_within(path, packages_root):
        return
    shutil.rmtree(path, ignore_errors=True)


def _retire_saved_html_preview_package(
    preview_id: str,
    expected_path: str,
    *,
    reason: str,
) -> None:
    delete_path = ''
    with _SAVED_HTML_PREVIEW_PACKAGES_LOCK:
        record = _SAVED_HTML_PREVIEW_PACKAGES.get(preview_id)
        if not isinstance(record, Mapping):
            return
        if str(record.get('bundle_path') or '') != str(expected_path or ''):
            return
        record['retire_reason'] = reason
        if int(record.get('lease_count') or 0) <= 0:
            _SAVED_HTML_PREVIEW_PACKAGES.pop(preview_id, None)
            timer = record.get('timer')
            if isinstance(timer, threading.Timer):
                timer.cancel()
            delete_path = str(record.get('bundle_path') or '')
    if delete_path:
        _delete_saved_html_preview_package_path(delete_path)


def _expire_saved_html_preview_package(preview_id: str, expected_path: str) -> None:
    _retire_saved_html_preview_package(
        preview_id,
        expected_path,
        reason='absolute_ttl_expired',
    )


def _saved_html_preview_package_for_root(base_root: Path) -> tuple[str, Optional[dict[str, Any]]]:
    resolved_root = base_root.resolve(strict=False)
    packages_root = _saved_html_preview_temp_subdir('packages')
    if not is_path_within(resolved_root, packages_root):
        return '', None
    now = time.monotonic()
    expired: tuple[str, str] | None = None
    with _SAVED_HTML_PREVIEW_PACKAGES_LOCK:
        for preview_id, record in _SAVED_HTML_PREVIEW_PACKAGES.items():
            if str(record.get('bundle_path') or '') != str(resolved_root):
                continue
            if record.get('retire_reason'):
                return preview_id, None
            if now >= float(record.get('expires_at_monotonic') or 0.0):
                expired = (preview_id, str(record.get('bundle_path') or ''))
                break
            return preview_id, record
    if expired:
        _expire_saved_html_preview_package(*expired)
        return expired[0], None
    return 'unregistered', None


def _acquire_saved_html_preview_package_lease(base_root: Path) -> tuple[bool, Optional[str]]:
    preview_id, record = _saved_html_preview_package_for_root(base_root)
    if not preview_id:
        return False, None
    if not isinstance(record, Mapping):
        return True, None
    expired: tuple[str, str] | None = None
    with _SAVED_HTML_PREVIEW_PACKAGES_LOCK:
        current = _SAVED_HTML_PREVIEW_PACKAGES.get(preview_id)
        if not isinstance(current, dict) or current.get('retire_reason'):
            return True, None
        if time.monotonic() >= float(current.get('expires_at_monotonic') or 0.0):
            expired = (preview_id, str(current.get('bundle_path') or ''))
        else:
            current['lease_count'] = int(current.get('lease_count') or 0) + 1
            current['last_access_monotonic'] = time.monotonic()
    if expired:
        _expire_saved_html_preview_package(*expired)
        return True, None
    return True, preview_id


def _release_saved_html_preview_package_lease(preview_id: str) -> None:
    delete_path = ''
    with _SAVED_HTML_PREVIEW_PACKAGES_LOCK:
        record = _SAVED_HTML_PREVIEW_PACKAGES.get(preview_id)
        if not isinstance(record, dict):
            return
        record['lease_count'] = max(0, int(record.get('lease_count') or 0) - 1)
        if record.get('retire_reason') and int(record.get('lease_count') or 0) == 0:
            _SAVED_HTML_PREVIEW_PACKAGES.pop(preview_id, None)
            timer = record.get('timer')
            if isinstance(timer, threading.Timer):
                timer.cancel()
            delete_path = str(record.get('bundle_path') or '')
    if delete_path:
        _delete_saved_html_preview_package_path(delete_path)


def _saved_html_preview_source_fingerprints(
    copied_artifacts: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for artifact in copied_artifacts:
        source_path = str(artifact.get('source_path') or '').strip()
        if not source_path:
            continue
        source = Path(source_path).expanduser().resolve(strict=False)
        fingerprints.append(
            {
                'path': str(source),
                'size_bytes': source.stat().st_size,
                'sha256': _hash_file_sha256(source),
            }
        )
    return sorted(fingerprints, key=lambda item: item['path'])


def _create_response_bound_saved_html_preview_package(
    response_id: str,
    source_path: Path,
) -> tuple[Optional[Path], Optional[dict[str, Any]], int]:
    response_payload, error_payload, status_code = _load_response_artifact_bundle_source_payload(
        response_id,
        hydrate_registry=False,
    )
    if error_payload or response_payload is None:
        return None, error_payload or {'error': 'Response artifact truth is unavailable.'}, status_code

    staging_root = _saved_html_preview_temp_subdir('staging')
    packages_root = _saved_html_preview_temp_subdir('packages')
    staging_dir = Path(tempfile.mkdtemp(prefix='preview-', dir=staging_root)).resolve()
    final_dir: Optional[Path] = None
    try:
        bundle_payload = _bundle_response_artifacts(
            response_payload,
            target_name=f'preview-{secrets.token_hex(6)}',
            bundle_root=staging_dir,
            required_public_source_path=source_path,
            require_public_output_surface=True,
            allow_unregistered_linked_dependencies=False,
            max_artifact_count=SAVED_HTML_PREVIEW_MAX_FILES,
            max_total_source_bytes=min(
                SAVED_HTML_PREVIEW_MAX_PACKAGE_BYTES,
                SAVED_HTML_PREVIEW_MAX_TOTAL_BYTES,
            ),
        )
        bundle_dir = Path(str(bundle_payload.get('bundle_path') or '')).resolve(strict=False)
        copied_artifacts = [
            item
            for item in (bundle_payload.get('copied_artifacts') or [])
            if isinstance(item, Mapping)
        ]
        requested_source = source_path.resolve(strict=False)
        selected_relative_path = ''
        for artifact in copied_artifacts:
            raw_source = str(artifact.get('source_path') or '').strip()
            raw_copied = str(artifact.get('path') or '').strip()
            if not raw_source or not raw_copied:
                continue
            if Path(raw_source).expanduser().resolve(strict=False) != requested_source:
                continue
            selected_relative_path = Path(raw_copied).resolve(strict=False).relative_to(
                bundle_dir
            ).as_posix()
            break
        if not selected_relative_path:
            raise ValueError('The requested HTML artifact was not copied into the preview package.')
        manifest_path = bundle_dir / 'manifest.json'
        if manifest_path.exists():
            manifest_path.unlink()
        source_fingerprints = _saved_html_preview_source_fingerprints(copied_artifacts)
        preview_id = secrets.token_urlsafe(18)
        final_dir = (packages_root / preview_id).resolve(strict=False)
        while final_dir.exists():
            preview_id = secrets.token_urlsafe(18)
            final_dir = (packages_root / preview_id).resolve(strict=False)
        os.replace(bundle_dir, final_dir)
        selected_path = (final_dir / selected_relative_path).resolve(strict=False)
        package_size = sum(
            path.stat().st_size
            for path in final_dir.rglob('*')
            if path.is_file()
        )
        if package_size > SAVED_HTML_PREVIEW_MAX_PACKAGE_BYTES:
            raise ValueError('The generated preview package exceeds the preview byte limit.')
        now = time.monotonic()
        frame_identity = _response_wire_frame_identity(response_payload)
        record: dict[str, Any] = {
            'preview_id': preview_id,
            'response_id': _normalize_response_lookup_id(response_id),
            'frame_id': frame_identity[0] if frame_identity else None,
            'frame_sequence': frame_identity[1] if frame_identity else None,
            'source_path': str(requested_source),
            'source_fingerprints': source_fingerprints,
            'bundle_path': str(final_dir),
            'selected_path': str(selected_path),
            'size_bytes': package_size,
            'created_at_monotonic': now,
            'last_access_monotonic': now,
            'expires_at_monotonic': now + SAVED_HTML_PREVIEW_TTL_SECONDS,
            'lease_count': 0,
            'retire_reason': None,
        }
        timer = threading.Timer(
            SAVED_HTML_PREVIEW_TTL_SECONDS,
            _expire_saved_html_preview_package,
            args=(preview_id, str(final_dir)),
        )
        timer.daemon = True
        record['timer'] = timer

        delete_paths: list[str] = []
        keep_new_package = True
        with _SAVED_HTML_PREVIEW_PACKAGES_LOCK:
            _SAVED_HTML_PREVIEW_PACKAGES[preview_id] = record
            while (
                len(_SAVED_HTML_PREVIEW_PACKAGES) > SAVED_HTML_PREVIEW_MAX_PACKAGES
                or sum(
                    int(item.get('size_bytes') or 0)
                    for item in _SAVED_HTML_PREVIEW_PACKAGES.values()
                ) > SAVED_HTML_PREVIEW_MAX_TOTAL_BYTES
            ):
                candidates = [
                    (key, item)
                    for key, item in _SAVED_HTML_PREVIEW_PACKAGES.items()
                    if key != preview_id and int(item.get('lease_count') or 0) == 0
                ]
                if not candidates:
                    keep_new_package = False
                    removed = _SAVED_HTML_PREVIEW_PACKAGES.pop(preview_id, None)
                    if isinstance(removed, Mapping):
                        delete_paths.append(str(removed.get('bundle_path') or ''))
                    break
                victim_id, victim = min(
                    candidates,
                    key=lambda pair: float(pair[1].get('last_access_monotonic') or 0.0),
                )
                _SAVED_HTML_PREVIEW_PACKAGES.pop(victim_id, None)
                victim_timer = victim.get('timer')
                if isinstance(victim_timer, threading.Timer):
                    victim_timer.cancel()
                delete_paths.append(str(victim.get('bundle_path') or ''))
        for delete_path in delete_paths:
            _delete_saved_html_preview_package_path(delete_path)
        if not keep_new_package:
            return None, {'error': 'HTML preview capacity is currently in use.'}, 503
        timer.start()
        return selected_path, None, 200
    except ValueError as exc:
        if final_dir is not None:
            _delete_saved_html_preview_package_path(final_dir)
        return None, {'error': str(exc)}, 409
    except OSError as exc:
        logging.exception('Could not create temporary HTML preview package: %s', exc)
        if final_dir is not None:
            _delete_saved_html_preview_package_path(final_dir)
        return None, {'error': 'Temporary HTML preview package could not be created.'}, 500
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _resolve_saved_artifact_request_path(raw_path: str, *, resolver, invalid_error: str):
    normalized_path = str(raw_path or "").strip()
    if not normalized_path:
        return None, (jsonify({"error": "Parameter 'path' is required."}), 400)
    resolved = resolver(normalized_path)
    if not resolved:
        return None, (jsonify({"error": invalid_error}), 400)
    return resolved, None


def _open_path_in_file_manager(path: Path) -> None:
    open_path_in_file_manager(path)


def _open_path_with_default_app(path: Path) -> None:
    target = str(path)
    if os.name == 'nt':
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if sys.platform == 'darwin':
        subprocess.Popen(['open', target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    opener = shutil.which('xdg-open')
    if opener:
        subprocess.Popen([opener, target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    raise RuntimeError('No supported default-app launcher was found on this system.')


def _open_saved_artifact_path_response(resolved: Path, *, log_message: str, open_default_app: bool = False):
    try:
        if open_default_app and resolved.is_file():
            _open_path_with_default_app(resolved)
        else:
            _open_path_in_file_manager(resolved)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # noqa: BLE001
        logging.exception("%s: %s", log_message, exc)
        return jsonify({"error": "The file could not be opened."}), 500
    return jsonify({"status": "opened", "path": str(resolved)})


def _encode_saved_artifact_preview_base_root(base_root: Path) -> str:
    raw = str(base_root.resolve()).encode('utf-8')
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _decode_saved_artifact_preview_base_root(token: str) -> Optional[Path]:
    cleaned = str(token or '').strip()
    if not cleaned:
        return None
    try:
        padded = cleaned + ('=' * (-len(cleaned) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8')
        return Path(decoded).expanduser().resolve(strict=False)
    except Exception:
        return None


def _saved_artifact_preview_base_root_for_path(resolved: Path) -> Path:
    roots = sorted(_saved_viewable_artifact_roots(), key=lambda item: len(str(item)), reverse=True)
    for root in roots:
        if is_path_within(resolved, root):
            return root.parent.resolve(strict=False)
    return resolved.parent.resolve(strict=False)


def _saved_artifact_preview_base_href(resolved: Path) -> str:
    base_root = _saved_artifact_preview_base_root_for_path(resolved)
    token = _encode_saved_artifact_preview_base_root(base_root)
    try:
        relative_dir = resolved.parent.resolve(strict=False).relative_to(base_root).as_posix().strip('/')
    except ValueError:
        relative_dir = ''
    prefix = f'/api/view_saved_artifact_assets/{quote(token, safe="")}/'
    if relative_dir:
        return f'{prefix}{quote(relative_dir, safe="/")}/'
    return prefix


def _saved_artifact_interactive_preview_base_root_for_path(resolved: Path) -> Path:
    """Scope a preview to one portable bundle or one source-artifact directory."""
    resolved_path = resolved.resolve(strict=False)
    temp_packages_root = _saved_html_preview_temp_subdir('packages')
    if is_path_within(resolved_path, temp_packages_root):
        try:
            relative = resolved_path.relative_to(temp_packages_root)
        except ValueError:
            relative = Path()
        if relative.parts:
            package_root = (temp_packages_root / relative.parts[0]).resolve(strict=False)
            _preview_id, record = _saved_html_preview_package_for_root(package_root)
            if package_root.is_dir() and isinstance(record, Mapping):
                return package_root
        raise ValueError('Temporary HTML preview package is unavailable or expired.')
    bundles_root = Path(ARTIFACT_BUNDLES_DIR).expanduser().resolve(strict=False)
    if is_path_within(resolved_path, bundles_root):
        try:
            relative = resolved_path.relative_to(bundles_root)
        except ValueError:
            relative = Path()
        if relative.parts:
            bundle_root = (bundles_root / relative.parts[0]).resolve(strict=False)
            if bundle_root.is_dir():
                return bundle_root
    for artifact_root in _saved_viewable_artifact_roots():
        if resolved_path.parent == artifact_root.resolve(strict=False):
            raise ValueError(
                'Flat artifact roots are not an interactive preview capability boundary.'
            )
    return resolved_path.parent


def _encode_saved_artifact_interactive_preview_base_root(base_root: Path) -> str:
    root_bytes = str(base_root.resolve(strict=False)).encode('utf-8')
    payload = _SAVED_INTERACTIVE_PREVIEW_TOKEN_VERSION + b'\0' + root_bytes
    signature = hmac.new(
        _SAVED_INTERACTIVE_PREVIEW_SIGNING_KEY,
        payload,
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(payload + b'\0' + signature).decode('ascii').rstrip('=')


def _decode_saved_artifact_interactive_preview_base_root(token: str) -> Optional[Path]:
    cleaned = str(token or '').strip()
    if not cleaned:
        return None
    try:
        padded = cleaned + ('=' * (-len(cleaned) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode('ascii'))
        if len(decoded) <= hashlib.sha256().digest_size:
            return None
        signature = decoded[-hashlib.sha256().digest_size:]
        signed_payload = decoded[:-hashlib.sha256().digest_size]
        if not signed_payload.endswith(b'\0'):
            return None
        payload = signed_payload[:-1]
        expected = hmac.new(
            _SAVED_INTERACTIVE_PREVIEW_SIGNING_KEY,
            payload,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        version, root_bytes = payload.split(b'\0', 1)
        if version != _SAVED_INTERACTIVE_PREVIEW_TOKEN_VERSION:
            return None
        base_root = Path(root_bytes.decode('utf-8')).expanduser().resolve(strict=False)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not base_root.exists() or not base_root.is_dir():
        return None
    return base_root


def _saved_artifact_interactive_preview_urls(
    resolved: Path,
    *,
    base_token: str | None = None,
) -> tuple[str, str]:
    base_root = (
        _decode_saved_artifact_interactive_preview_base_root(base_token)
        if base_token
        else _saved_artifact_interactive_preview_base_root_for_path(resolved)
    )
    if base_root is None or not is_path_within(resolved.resolve(strict=False), base_root):
        raise ValueError('HTML preview path is outside its signed artifact base.')
    token = base_token or _encode_saved_artifact_interactive_preview_base_root(base_root)
    route_prefix = f'/api/preview_saved_artifact_assets/{quote(token, safe="")}/'
    try:
        relative_file = resolved.resolve(strict=False).relative_to(base_root).as_posix().strip('/')
    except ValueError as exc:
        raise ValueError('HTML preview path is outside its signed artifact base.') from exc
    if not relative_file:
        raise ValueError('HTML preview path does not identify a file inside its signed artifact base.')
    base_href = f'{route_prefix}{quote(relative_file, safe="/")}'
    absolute_asset_prefix = f'{request.url_root.rstrip("/")}{route_prefix}'
    return base_href, absolute_asset_prefix


def _inject_saved_artifact_preview_base(html_text: str, *, base_href: str) -> str:
    base_tag = f'<base href="{html_lib.escape(base_href, quote=True)}">'
    head_match = re.search(r'<head\b[^>]*>', html_text, flags=re.IGNORECASE)
    if head_match:
        insert_at = head_match.end()
        return f'{html_text[:insert_at]}\n    {base_tag}{html_text[insert_at:]}'
    return f'{base_tag}\n{html_text}'


def _html_saved_artifact_preview_response(resolved: Path, *, mimetype: str):
    try:
        html_text = resolved.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        html_text = resolved.read_text(encoding='utf-8', errors='replace')
    body = _inject_saved_artifact_preview_base(
        html_text,
        base_href=_saved_artifact_preview_base_href(resolved),
    )
    response = Response(body, mimetype=mimetype or 'text/html')
    response.headers.set('Content-Disposition', 'inline', filename=resolved.name)
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    _apply_saved_artifact_html_security_headers(response)
    return response


def _apply_saved_artifact_interactive_preview_headers(response, *, asset_prefix: str):
    response.headers['Content-Security-Policy'] = (
        "default-src 'none'; "
        f"img-src {asset_prefix} data: blob:; "
        f"style-src {asset_prefix} 'unsafe-inline' data:; "
        f"script-src {asset_prefix} 'unsafe-inline'; "
        f"connect-src {asset_prefix}; "
        f"font-src {asset_prefix} data:; "
        f"media-src {asset_prefix} data: blob:; "
        "object-src 'none'; frame-src 'none'; worker-src 'none'; "
        f"base-uri {asset_prefix}; "
        "form-action 'none'; frame-ancestors 'self'; "
        "sandbox allow-scripts allow-top-navigation-to-custom-protocols"
    )
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), midi=()'
    )
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cache-Control'] = 'no-store'
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    return response


def _html_saved_artifact_interactive_preview_wrapper_response(resolved: Path):
    frame_src, asset_prefix = _saved_artifact_interactive_preview_urls(resolved)
    escaped_frame_src = html_lib.escape(frame_src, quote=True)
    escaped_title_text = html_lib.escape(resolved.name, quote=False)
    escaped_title_attribute = html_lib.escape(resolved.name, quote=True)
    body = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>Preview — {escaped_title_text}</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>html,body,iframe{width:100%;height:100%;margin:0;border:0}'
        'body{overflow:hidden;background:#111}iframe{display:block}</style>'
        '</head><body>'
        f'<iframe src="{escaped_frame_src}" '
        'sandbox="allow-scripts allow-top-navigation-to-custom-protocols" '
        f'title="Preview of {escaped_title_attribute}" referrerpolicy="no-referrer"></iframe>'
        '</body></html>'
    )
    response = Response(body, mimetype='text/html')
    response.headers.set('Content-Disposition', 'inline', filename=resolved.name)
    response.headers['Content-Security-Policy'] = (
        f"default-src 'none'; style-src 'unsafe-inline'; frame-src {asset_prefix}; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
    )
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), midi=()'
    )
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cache-Control'] = 'no-store'
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    return response


def _html_saved_artifact_interactive_preview_response(
    resolved: Path,
    *,
    mimetype: str,
    base_token: str | None = None,
):
    try:
        html_text = resolved.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        html_text = resolved.read_text(encoding='utf-8', errors='replace')
    base_href, asset_prefix = _saved_artifact_interactive_preview_urls(
        resolved,
        base_token=base_token,
    )
    body = _inject_saved_artifact_preview_base(html_text, base_href=base_href)
    response = Response(body, mimetype=mimetype or 'text/html')
    response.headers.set('Content-Disposition', 'inline', filename=resolved.name)
    return _apply_saved_artifact_interactive_preview_headers(
        response,
        asset_prefix=asset_prefix,
    )


def _resolve_saved_artifact_preview_asset_path(base_token: str, asset_path: str) -> Optional[Path]:
    base_root = _decode_saved_artifact_preview_base_root(base_token)
    if base_root is None:
        return None
    raw_asset_path = str(asset_path or '').strip()
    if not raw_asset_path:
        return None
    parsed = urlparse(raw_asset_path)
    if parsed.scheme or parsed.netloc:
        return None
    try:
        candidate = (base_root / unquote(parsed.path)).resolve(strict=False)
    except (OSError, ValueError):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    allowed_roots = _saved_viewable_artifact_roots()
    if not any(is_path_within(candidate, root) for root in allowed_roots):
        return None
    return candidate


def _resolve_saved_artifact_interactive_preview_asset_path(
    base_token: str,
    asset_path: str,
) -> Optional[Path]:
    base_root = _decode_saved_artifact_interactive_preview_base_root(base_token)
    if base_root is None:
        return None
    raw_asset_path = str(asset_path or '').strip()
    if not raw_asset_path:
        return None
    parsed = urlparse(raw_asset_path)
    if parsed.scheme or parsed.netloc:
        return None
    try:
        candidate = (base_root / unquote(parsed.path)).resolve(strict=False)
    except (OSError, ValueError):
        return None
    if not is_path_within(candidate, base_root):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    in_saved_root = any(
        is_path_within(candidate, root)
        for root in _saved_viewable_artifact_roots()
    )
    temp_preview_id, temp_record = _saved_html_preview_package_for_root(base_root)
    in_active_temp_package = bool(temp_preview_id and isinstance(temp_record, Mapping))
    if not in_saved_root and not in_active_temp_package:
        return None
    return candidate


def _send_saved_artifact_interactive_preview_asset_response(
    resolved: Path,
    *,
    base_token: str,
):
    mimetype, _ = mimetypes.guess_type(str(resolved))
    normalized_mimetype = str(mimetype or '').split(';', 1)[0].strip().lower()
    if (
        normalized_mimetype in {'text/html', 'application/xhtml+xml'}
        or resolved.suffix.lower() in {'.html', '.htm'}
    ):
        response = _html_saved_artifact_interactive_preview_response(
            resolved,
            mimetype=mimetype or 'text/html',
            base_token=base_token,
        )
    else:
        response = send_file(
            str(resolved),
            as_attachment=False,
            download_name=resolved.name,
            mimetype=mimetype or 'application/octet-stream',
        )
        response.headers['Content-Security-Policy'] = "sandbox; default-src 'none'"
        response.headers['Cache-Control'] = 'no-store'
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    return response


def _apply_saved_artifact_html_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self' data: blob:; "
        "img-src 'self' data: blob: file:; "
        "style-src 'self' 'unsafe-inline' data: blob:; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "font-src 'self' data: blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'self'"
    )
    return response


def _send_saved_artifact_file_response(resolved: Path, *, as_attachment: bool):
    mimetype, _ = mimetypes.guess_type(str(resolved))
    normalized_mimetype = str(mimetype or '').split(';', 1)[0].strip().lower()
    if (
        not as_attachment
        and (
            normalized_mimetype in {'text/html', 'application/xhtml+xml'}
            or resolved.suffix.lower() in {'.html', '.htm'}
        )
    ):
        return _html_saved_artifact_preview_response(resolved, mimetype=mimetype or 'text/html')
    response = send_file(
        str(resolved),
        as_attachment=as_attachment,
        download_name=resolved.name,
        mimetype=mimetype or "application/octet-stream",
    )
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    if not as_attachment:
        if normalized_mimetype in {'text/html', 'application/xhtml+xml'} or resolved.suffix.lower() in {'.html', '.htm'}:
            _apply_saved_artifact_html_security_headers(response)
    return response


_ollama_openai_image_generation = _BACKEND_TRANSPORT_RUNTIME.ollama_openai_image_generation
_persist_audio_bytes_locally = _BACKEND_TRANSPORT_RUNTIME.persist_audio_bytes_locally
_ollama_chat = _BACKEND_TRANSPORT_RUNTIME.ollama_chat
_extract_text_payload = _BACKEND_TRANSPORT_RUNTIME.extract_text_payload
_ollama_chat_with_options = _BACKEND_TRANSPORT_RUNTIME.ollama_chat_with_options
_whisper_transcribe = _BACKEND_TRANSPORT_RUNTIME.whisper_transcribe
_mlx_audio_speech = _BACKEND_TRANSPORT_RUNTIME.mlx_audio_speech
_mlx_chat_completions = _BACKEND_TRANSPORT_RUNTIME.mlx_chat_completions


def _openai_chat_completions(
    backend: str,
    port: int,
    model_name: str,
    messages: list[dict],
    timeout_sec: int = 600,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> dict:
    if backend == 'mlx':
        return _mlx_chat_completions(
            port,
            model_name,
            messages,
            timeout_sec=timeout_sec,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
        )
    return _BACKEND_TRANSPORT_RUNTIME.openai_chat_completions(
        backend,
        port,
        model_name,
        messages,
        timeout_sec=timeout_sec,
        temperature=temperature,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
    )

# --- Routen / Endpunkte ---

def _render_product_page(template_name: str):
    logging.info("Serving template: %s", template_name)
    try:
        return render_template(template_name)
    except Exception as e:
        logging.error("Error rendering %s: %s", template_name, e)
        return f"Error: HTML file '{template_name}' could not be loaded.", 500


@app.route('/')
def index():
    """Return Ollmo's local dashboard."""
    return _render_product_page(DASHBOARD_HTML_FILE)


@app.route("/dashboard")
def dashboard():
    """Return the dashboard through its backwards-compatible alias."""
    return _render_product_page(DASHBOARD_HTML_FILE)


@app.route('/site/')
def landing_site():
    """Return the standalone repository landing page for local preview."""
    return send_from_directory(LANDING_SITE_DIRECTORY, LANDING_HTML_FILE)


@app.route('/site/<path:filename>')
def landing_site_asset(filename: str):
    """Serve only the landing page's self-contained publication tree."""
    return send_from_directory(LANDING_SITE_DIRECTORY, filename)

@app.route('/api/running_instances', methods=['GET'])
def get_running_instances():
    logging.info("API call: /api/running_instances")
    refresh_requested = _observer_refresh_requested()
    runtime_truth = _runtime_truth_metadata(refresh_requested=refresh_requested)
    instances = merge_instances_with_runtime_status(
        load_running_instances(),
        path=RUNTIME_STATUS_PATH,
        refresh=refresh_requested,
    )
    logging.info(f"Found instances: {len(instances)}")
    return _attach_runtime_truth_headers(jsonify(instances), runtime_truth)


@app.route('/api/runtime_status', methods=['GET'])
def api_runtime_status():
    instance_id = str(request.args.get('instance_id') or '').strip()
    refresh_requested = _observer_refresh_requested()
    runtime_truth = _runtime_truth_metadata(refresh_requested=refresh_requested)
    if instance_id:
        try:
            instance_id = _normalize_external_identifier(instance_id, field_name='instance_id')
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    if refresh_requested:
        instances = load_running_instances()
        statuses = refresh_runtime_status_entries(instances, path=RUNTIME_STATUS_PATH)
    else:
        cached_payload = read_runtime_status(RUNTIME_STATUS_PATH)
        statuses = cached_payload.get('instances') if isinstance(cached_payload.get('instances'), dict) else {}
    if instance_id:
        entry = statuses.get(instance_id)
        if not entry:
            return _attach_runtime_truth_headers(jsonify({"error": "Instance was not found."}), runtime_truth), 404
        return _attach_runtime_truth_headers(jsonify(entry), runtime_truth)
    items = list(statuses.values())
    return _attach_runtime_truth_headers(jsonify({"items": items, "count": len(items)}), runtime_truth)


@app.route('/api/runtime_manifest', methods=['GET'])
@app.route('/api/routing_table', methods=['GET'])
def api_runtime_manifest():
    refresh_requested = _observer_refresh_requested()
    instances = merge_instances_with_runtime_status(
        load_running_instances(),
        path=RUNTIME_STATUS_PATH,
        refresh=refresh_requested,
    )
    backend_fabric = build_backend_fabric_snapshot(instances=instances)
    payload = _build_routing_manifest_payload(instances, backend_fabric=backend_fabric)
    payload['runtime_truth'] = _runtime_truth_metadata(refresh_requested=refresh_requested)
    return jsonify(payload)


@app.route('/api/backend_fabric', methods=['GET'])
def api_backend_fabric():
    include_catalog = str(request.args.get('with_catalog') or '').strip().lower() in {'1', 'true', 'yes'}
    refresh_requested = _observer_refresh_requested()
    payload, status_code = _MODEL_CONTROL_RUNTIME.build_backend_fabric_response(
        include_catalog=include_catalog,
        refresh_runtime_status=refresh_requested,
    )
    if isinstance(payload, dict):
        payload['runtime_truth'] = _runtime_truth_metadata(refresh_requested=refresh_requested)
    return jsonify(payload), status_code


@app.route('/api/ghost', methods=['GET'])
@app.route('/api/agent_contract', methods=['GET'])
def api_runtime_ghost():
    refresh_requested = _observer_refresh_requested()
    instances = merge_instances_with_runtime_status(
        load_running_instances(),
        path=RUNTIME_STATUS_PATH,
        refresh=refresh_requested,
    )
    recent_items = read_events(path=EVENT_LOG_PATH, limit=12)
    payload = build_ghost_payload(
        instances,
        recent_events=recent_items,
        base_url=str(request.host_url or '').rstrip('/'),
        contract_path=GHOST_GUIDE_PATH,
        runtime_log_path=FLASK_LOG_PATH,
    )
    payload['runtime_truth'] = _runtime_truth_metadata(refresh_requested=refresh_requested)
    output_format = str(request.args.get('format') or '').strip().lower()
    if output_format in {'md', 'markdown', 'text'}:
        return Response(payload.get('markdown') or '', mimetype='text/markdown')
    return jsonify(payload)


@app.route('/api/available_models', methods=['GET'])
def available_models():
    logging.info('API call: /api/available_models')
    include_limits = request.args.get('with_limits', 'false').lower() == 'true'
    payload, status_code = _MODEL_CONTROL_RUNTIME.build_available_models_response(
        include_limits=include_limits,
    )
    return jsonify(payload), status_code


@app.route('/api/pull_model', methods=['POST'])
def api_pull_model():
    payload, status_code = _MODEL_CONTROL_RUNTIME.pull_model_response(
        request.get_json(silent=True) or {},
    )
    return jsonify(payload), status_code


@app.route('/api/remove_model', methods=['POST'])
def api_remove_model():
    payload, status_code = _MODEL_CONTROL_RUNTIME.remove_model_response(
        request.get_json(silent=True) or {},
    )
    return jsonify(payload), status_code


@app.route('/api/start_model', methods=['POST'])
def api_start_model():
    payload, status_code = _MODEL_CONTROL_RUNTIME.start_model_response(
        request.get_json(silent=True) or {},
        default_start_source='api_start_model',
    )
    return jsonify(payload), status_code


@app.route('/api/stop_model', methods=['POST'])
def api_stop_model():
    payload, status_code = _MODEL_CONTROL_RUNTIME.stop_model_response(
        request.get_json(silent=True) or {},
    )
    return jsonify(payload), status_code


def _execute_infer_request(data: Any, *, upload=None):
    """Execute capability-aware inference for text, image, audio, and file prompts."""
    return _INFER_RUNTIME.execute_infer_request(data, upload=upload)


@app.route('/api/infer', methods=['POST'])
def api_infer():
    is_multipart = request.content_type and request.content_type.startswith("multipart/form-data")
    if is_multipart:
        data = request.form
        upload = request.files.get("file")
    else:
        data = request.get_json(silent=True) or {}
        upload = None
    return _execute_infer_request(data, upload=upload)


@app.route('/api/expand_local_paths', methods=['POST'])
def api_expand_local_paths():
    data = request.get_json(silent=True) or {}
    raw_paths = data.get("paths")
    if isinstance(raw_paths, str):
        paths = [line.strip() for line in raw_paths.splitlines() if line.strip()]
    elif isinstance(raw_paths, list):
        paths = [str(item).strip() for item in raw_paths if str(item).strip()]
    else:
        paths = []
    if not paths:
        return jsonify({"error": "Parameter 'paths' is required or empty."}), 400

    max_items = _parse_int_with_bounds(data.get("max_items"), default=1000, minimum=1, maximum=10_000)
    expanded, skipped, truncated = _expand_local_paths(paths, max_items=max_items)
    return jsonify(
        {
            "paths": expanded,
            "count": len(expanded),
            "skipped": skipped,
            "truncated": truncated,
            "max_items": max_items,
        }
    )


@app.route('/api/open_saved_image', methods=['POST'])
def api_open_saved_image():
    data = request.get_json(silent=True) or {}
    resolved, error_response = _resolve_saved_artifact_request_path(
        data.get("path"),
        resolver=_resolve_generated_image_path,
        invalid_error=_SAVED_IMAGE_PATH_ERROR,
    )
    if error_response:
        return error_response
    return _open_saved_artifact_path_response(
        resolved,
        log_message="Could not open image path in the file manager",
    )


@app.route('/api/open_saved_artifact', methods=['POST'])
def api_open_saved_artifact():
    data = request.get_json(silent=True) or {}
    resolved, error_response = _resolve_saved_artifact_request_path(
        data.get("path"),
        resolver=_resolve_saved_openable_artifact_path,
        invalid_error=_SAVED_OPENABLE_ARTIFACT_PATH_ERROR,
    )
    if error_response:
        return error_response
    return _open_saved_artifact_path_response(
        resolved,
        log_message="Could not open artifact path in the file manager",
        open_default_app=_parse_bool(data.get("open_file") or data.get("openFile"), default=False),
    )


@app.route('/api/download_saved_artifact', methods=['GET'])
def api_download_saved_artifact():
    resolved, error_response = _resolve_saved_artifact_request_path(
        request.args.get("path"),
        resolver=_resolve_saved_downloadable_artifact_path,
        invalid_error=_SAVED_DOWNLOADABLE_ARTIFACT_PATH_ERROR,
    )
    if error_response:
        return error_response
    return _send_saved_artifact_file_response(resolved, as_attachment=True)


@app.route('/api/view_saved_artifact', methods=['GET'])
def api_view_saved_artifact():
    resolved, error_response = _resolve_saved_artifact_request_path(
        request.args.get("path"),
        resolver=_resolve_saved_viewable_artifact_path,
        invalid_error=_SAVED_VIEWABLE_ARTIFACT_PATH_ERROR,
    )
    if error_response:
        return error_response
    return _send_saved_artifact_file_response(resolved, as_attachment=False)


@app.route('/api/preview_saved_artifact', methods=['GET'])
def api_preview_saved_artifact():
    resolved, error_response = _resolve_saved_artifact_request_path(
        request.args.get("path"),
        resolver=_resolve_saved_viewable_artifact_path,
        invalid_error=_SAVED_INTERACTIVE_PREVIEW_PATH_ERROR,
    )
    if error_response:
        return error_response
    mimetype, _ = mimetypes.guess_type(str(resolved))
    normalized_mimetype = str(mimetype or '').split(';', 1)[0].strip().lower()
    if (
        normalized_mimetype not in {'text/html', 'application/xhtml+xml'}
        and resolved.suffix.lower() not in {'.html', '.htm'}
    ):
        return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 400
    try:
        return _html_saved_artifact_interactive_preview_wrapper_response(resolved)
    except ValueError:
        raw_response_id = str(request.args.get('response_id') or '').strip()
        if not raw_response_id:
            return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 400
        response_id = _normalize_response_lookup_id(raw_response_id)
        preview_path, preview_error, status_code = (
            _create_response_bound_saved_html_preview_package(response_id, resolved)
        )
        if preview_error or preview_path is None:
            return jsonify(
                preview_error or {"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}
            ), status_code
        try:
            return _html_saved_artifact_interactive_preview_wrapper_response(preview_path)
        except ValueError:
            return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 400


@app.route('/api/view_saved_artifact_assets/<base_token>/', defaults={'asset_path': ''}, methods=['GET'])
@app.route('/api/view_saved_artifact_assets/<base_token>/<path:asset_path>', methods=['GET'])
def api_view_saved_artifact_asset(base_token: str, asset_path: str):
    resolved = _resolve_saved_artifact_preview_asset_path(base_token, asset_path)
    if not resolved:
        return jsonify({"error": _SAVED_VIEWABLE_ARTIFACT_PATH_ERROR}), 404
    return _send_saved_artifact_file_response(resolved, as_attachment=False)


@app.route('/api/preview_saved_artifact_assets/<base_token>/', defaults={'asset_path': ''}, methods=['GET'])
@app.route('/api/preview_saved_artifact_assets/<base_token>/<path:asset_path>', methods=['GET'])
def api_preview_saved_artifact_asset(base_token: str, asset_path: str):
    base_root = _decode_saved_artifact_interactive_preview_base_root(base_token)
    if base_root is None:
        return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 404
    is_temporary, lease_id = _acquire_saved_html_preview_package_lease(base_root)
    if is_temporary and not lease_id:
        return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 404
    resolved = _resolve_saved_artifact_interactive_preview_asset_path(base_token, asset_path)
    if not resolved:
        if lease_id:
            _release_saved_html_preview_package_lease(lease_id)
        return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 404
    try:
        response = _send_saved_artifact_interactive_preview_asset_response(
            resolved,
            base_token=base_token,
        )
    except ValueError:
        if lease_id:
            _release_saved_html_preview_package_lease(lease_id)
        return jsonify({"error": _SAVED_INTERACTIVE_PREVIEW_PATH_ERROR}), 404
    except Exception:
        if lease_id:
            _release_saved_html_preview_package_lease(lease_id)
        raise
    if lease_id:
        # Flask's send_file responses use direct passthrough, which bypasses
        # Response.close() and therefore its registered lease callback.
        response.direct_passthrough = False
        response.call_on_close(
            lambda: _release_saved_html_preview_package_lease(lease_id)
        )
    return response


@app.route('/api/delete_saved_artifact', methods=['POST'])
def api_delete_saved_artifact():
    data = request.get_json(silent=True) or {}
    resolved, error_response = _resolve_saved_artifact_request_path(
        data.get("path"),
        resolver=_resolve_saved_downloadable_artifact_path,
        invalid_error=_SAVED_DOWNLOADABLE_ARTIFACT_PATH_ERROR,
    )
    if error_response:
        return error_response

    try:
        resolved.unlink()
    except FileNotFoundError:
        return jsonify({"error": "Artifact was not found."}), 404
    except OSError as exc:
        logging.exception("Could not delete artifact: %s", exc)
        return jsonify({"error": "Artifact could not be deleted."}), 500

    _log_unified_event(
        category='artifact',
        action='delete',
        status='ok',
        path=str(resolved),
        message=f'Deleted saved artifact {resolved.name}',
    )
    return jsonify({"status": "deleted", "deleted": True, "path": str(resolved)})


@app.route('/api/settings_artifacts', methods=['POST'])
def api_create_settings_artifact():
    data = request.get_json(silent=True) or {}
    source = (
        data.get('response_frame')
        if isinstance(data.get('response_frame'), dict)
        else (
            data.get('controls')
            if isinstance(data.get('controls'), dict)
            else (
                data.get('settings_artifact')
                if isinstance(data.get('settings_artifact'), dict)
                else data
            )
        )
    )
    try:
        artifact = _persist_settings_artifact(
            source,
            label=str(data.get('label') or '').strip() or None,
            artifacts_dir=SETTINGS_ARTIFACTS_DIR,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except OSError as exc:
        logging.exception("Could not persist settings artifact: %s", exc)
        return jsonify({'error': 'Settings artifact could not be persisted.'}), 500

    _log_unified_event(
        category='artifact',
        action='settings_artifact_create',
        status='ok',
        path=artifact.get('path'),
        artifact_id=artifact.get('artifact_id'),
        message=f"Created settings artifact {artifact.get('artifact_id')}",
    )
    return jsonify({'ok': True, 'artifact': artifact})


@app.route('/api/settings_artifacts', methods=['GET'])
def api_list_settings_artifacts():
    return jsonify({'artifacts': _list_settings_artifacts(artifacts_dir=SETTINGS_ARTIFACTS_DIR)})


@app.route('/api/settings_artifacts/<path:artifact_id>', methods=['GET'])
def api_get_settings_artifact(artifact_id: str):
    try:
        artifact = _load_settings_artifact(artifact_id, artifacts_dir=SETTINGS_ARTIFACTS_DIR)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except FileNotFoundError:
        return jsonify({'error': 'Settings artifact not found.'}), 404
    return jsonify({'artifact': artifact})


@app.route('/api/infer_history', methods=['GET'])
def api_infer_history():
    limit = _parse_int_with_bounds(request.args.get("limit"), default=50, minimum=1, maximum=500)
    capability = str(request.args.get("capability") or "").strip()
    mode = str(request.args.get("mode") or "").strip()
    file_kind = str(request.args.get("file_kind") or "").strip()

    entries = _read_infer_history(limit=2000)
    if capability:
        entries = [item for item in entries if str(item.get("capability") or "") == capability]
    if mode:
        entries = [item for item in entries if str(item.get("mode") or "") == mode]
    if file_kind:
        entries = [item for item in entries if str(item.get("file_kind") or "") == file_kind]
    entries = entries[:limit]
    return jsonify({"items": entries, "count": len(entries)})


@app.route('/api/event_history', methods=['GET'])
def api_event_history():
    limit = _parse_int_with_bounds(request.args.get("limit"), default=100, minimum=1, maximum=1000)
    category = str(request.args.get("category") or "").strip() or None
    action = str(request.args.get("action") or "").strip() or None
    status = str(request.args.get("status") or "").strip() or None
    items = read_events(
        path=EVENT_LOG_PATH,
        limit=limit,
        category=category,
        action=action,
        status=status,
    )
    return jsonify({"items": items, "count": len(items)})


@app.route('/api/chat_history', methods=['GET'])
def api_chat_history_get():
    try:
        instance_id = _normalize_external_identifier(request.args.get('instance_id'), field_name='instance_id')
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_hydrate_chat_history_from_response_lookup(read_chat_history(instance_id, history_dir=CHAT_HISTORY_DIR)))


def _response_id_from_history_message(message: Mapping[str, Any]) -> str:
    response_id = str(message.get('response_id') or message.get('responseId') or '').strip()
    if response_id:
        return response_id
    request_snapshot = (
        message.get('request_snapshot')
        if isinstance(message.get('request_snapshot'), Mapping)
        else message.get('requestSnapshot')
        if isinstance(message.get('requestSnapshot'), Mapping)
        else {}
    )
    return str(request_snapshot.get('response_id') or request_snapshot.get('responseId') or '').strip()


def _response_lookup_payload_is_terminal_history_surface(response_payload: Mapping[str, Any]) -> bool:
    if not isinstance(response_payload, Mapping):
        return False
    status = str(response_payload.get('status') or '').strip().lower()
    lifecycle_state = str(response_payload.get('lifecycle_state') or '').strip().lower()
    status_semantics = (
        response_payload.get('status_semantics')
        if isinstance(response_payload.get('status_semantics'), Mapping)
        else {}
    )
    semantic_lifecycle = str(
        status_semantics.get('canonical_lifecycle_state')
        or status_semantics.get('canonicalLifecycleState')
        or ''
    ).strip().lower()
    if not {
        status,
        lifecycle_state,
        semantic_lifecycle,
    }.intersection({'completed', 'failed', 'cancelled', 'canceled'}):
        return False
    return any(
        response_payload.get(key) not in (None, '', [], {})
        for key in (
            'output_text',
            'artifacts',
            'outputs',
            'output_slots',
            'output_branches',
            'saved_image_path',
            'saved_audio_path',
            'saved_text_path',
            'error',
        )
    )


def _hydrate_chat_history_message_from_response_payload(
    message: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> dict[str, Any]:
    hydrated = dict(message)
    if isinstance(response_payload.get('artifacts'), list):
        hydrated['artifacts'] = response_payload['artifacts']
    if isinstance(response_payload.get('artifact_bundles'), list):
        hydrated['artifact_bundles'] = response_payload['artifact_bundles']
    for key in ('outputs', 'output_slots', 'output_branches'):
        value = response_payload.get(key)
        if isinstance(value, list):
            hydrated[key] = value
    late_fill = response_payload.get('late_fill')
    if isinstance(late_fill, Mapping):
        hydrated['late_fill'] = dict(late_fill)
    status_semantics = response_payload.get('status_semantics')
    if isinstance(status_semantics, Mapping):
        hydrated['status_semantics'] = dict(status_semantics)
    surface_state = response_payload.get('surface_state')
    if isinstance(surface_state, Mapping):
        hydrated['surface_state'] = dict(surface_state)
    for key in ('lifecycle_state', 'state_version', 'canonical_status_field', 'status_compatibility'):
        value = response_payload.get(key)
        if value not in (None, '', [], {}):
            hydrated[key] = value
    public_output_text = str(response_payload.get('output_text') or '').strip()
    if public_output_text and str(hydrated.get('role') or '').strip().lower() != 'user':
        hydrated['content'] = public_output_text
        hydrated['output_text'] = public_output_text
    response_frame = (
        response_payload.get('response_frame')
        if isinstance(response_payload.get('response_frame'), Mapping)
        else {}
    )
    frame_sequence = response_frame.get('frame_sequence') if isinstance(response_frame, Mapping) else None
    if frame_sequence not in (None, '', [], {}):
        hydrated['response_frame_sequence'] = frame_sequence
    frame_id = str(response_frame.get('frame_id') or '').strip() if isinstance(response_frame, Mapping) else ''
    if frame_id:
        hydrated['response_frame_id'] = frame_id
    for source_key, target_key in (
        ('id', 'response_id'),
        ('capability', 'response_capability'),
        ('model', 'response_model'),
        ('backend', 'response_backend'),
        ('instance_id', 'response_instance_id'),
        ('saved_image_path', 'saved_image_path'),
        ('saved_audio_path', 'saved_audio_path'),
        ('saved_text_path', 'saved_text_path'),
    ):
        value = response_payload.get(source_key)
        if value not in (None, '', [], {}):
            hydrated[target_key] = value
    return hydrated


def _build_missing_assistant_history_message_from_response_lookup(
    user_message: Mapping[str, Any],
    response_id: str,
) -> Optional[dict[str, Any]]:
    record = _get_response_lookup_record(response_id)
    if not record:
        return None
    response_payload = _build_response_lookup_payload(record)
    if not _response_lookup_payload_is_terminal_history_surface(response_payload):
        return None
    request_snapshot = (
        user_message.get('request_snapshot')
        if isinstance(user_message.get('request_snapshot'), Mapping)
        else user_message.get('requestSnapshot')
        if isinstance(user_message.get('requestSnapshot'), Mapping)
        else None
    )
    assistant_message: dict[str, Any] = {
        'role': 'assistant',
        'content': '',
        'response_id': response_id,
    }
    message_id = str(record.get('message_id') or '').strip()
    if message_id:
        assistant_message['message_id'] = message_id
    if isinstance(request_snapshot, Mapping):
        assistant_message['request_snapshot'] = dict(request_snapshot)
    return _hydrate_chat_history_message_from_response_payload(assistant_message, response_payload)


def _hydrate_chat_history_from_response_lookup(history: dict[str, Any]) -> dict[str, Any]:
    payload = dict(history or {})
    messages = payload.get('messages') if isinstance(payload.get('messages'), list) else []
    existing_assistant_response_ids = {
        _response_id_from_history_message(message)
        for message in messages
        if isinstance(message, Mapping)
        and str(message.get('role') or '').strip().lower() != 'user'
        and _response_id_from_history_message(message)
    }
    hydrated_messages: list[dict[str, Any]] = []
    synthesized_assistant_response_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        hydrated = dict(message)
        response_id = str(hydrated.get('response_id') or hydrated.get('responseId') or '').strip()
        if response_id:
            record = _get_response_lookup_record(response_id)
            if record:
                hydrated = _hydrate_chat_history_message_from_response_payload(
                    hydrated,
                    _build_response_lookup_payload(record),
                )
        hydrated_messages.append(hydrated)
        if str(hydrated.get('role') or '').strip().lower() != 'user':
            continue
        snapshot_response_id = _response_id_from_history_message(hydrated)
        if (
            not snapshot_response_id
            or snapshot_response_id in existing_assistant_response_ids
            or snapshot_response_id in synthesized_assistant_response_ids
        ):
            continue
        synthesized = _build_missing_assistant_history_message_from_response_lookup(
            hydrated,
            snapshot_response_id,
        )
        if synthesized:
            hydrated_messages.append(synthesized)
            synthesized_assistant_response_ids.add(snapshot_response_id)
    payload['messages'] = hydrated_messages
    return payload


@app.route('/api/chat_history/index', methods=['GET'])
def api_chat_history_index_get():
    return jsonify({
        "items": list_chat_history_index(history_dir=CHAT_HISTORY_DIR),
    })


@app.route('/api/chat_history/slot', methods=['GET'])
def api_chat_history_slot_get():
    try:
        workspace = _normalize_external_identifier(request.args.get('workspace'), field_name='workspace', allow_slashes=False)
        slot_id = _normalize_external_identifier(request.args.get('slot_id'), field_name='slot_id')
        raw_fallback_instance_id = request.args.get('fallback_instance_id')
        fallback_instance_id = (
            _normalize_external_identifier(raw_fallback_instance_id, field_name='fallback_instance_id')
            if str(raw_fallback_instance_id or '').strip()
            else None
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        resolve_chat_history_slot(
            workspace=workspace,
            slot_id=slot_id,
            history_dir=CHAT_HISTORY_DIR,
            fallback_instance_id=fallback_instance_id,
        )
    )


@app.route('/api/chat_history', methods=['POST'])
def api_chat_history_upsert():
    payload = request.get_json(silent=True) or {}
    try:
        instance_id = _normalize_external_identifier(payload.get('instance_id'), field_name='instance_id')
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return jsonify({"error": "Parameter 'messages' muss eine Liste sein."}), 400
    hydrated_payload = _hydrate_chat_history_from_response_lookup({'messages': messages})
    hydrated_messages = (
        hydrated_payload.get('messages')
        if isinstance(hydrated_payload.get('messages'), list)
        else messages
    )
    history = write_chat_history(
        instance_id,
        hydrated_messages,
        history_dir=CHAT_HISTORY_DIR,
        model=str(payload.get('model') or '').strip() or None,
        backend=str(payload.get('backend') or '').strip() or None,
        capability=str(payload.get('capability') or '').strip() or None,
        conversation_metadata=payload.get('conversation_metadata'),
    )
    return jsonify(history)


@app.route('/api/chat_history/rotate', methods=['POST'])
def api_chat_history_rotate():
    payload = request.get_json(silent=True) or {}
    try:
        current_instance_id = _normalize_external_identifier(payload.get('current_instance_id'), field_name='current_instance_id')
        source_instance_id = (
            _normalize_external_identifier(payload.get('source_instance_id'), field_name='source_instance_id')
            if str(payload.get('source_instance_id') or '').strip()
            else None
        )
        workspace = (
            _normalize_external_identifier(payload.get('workspace'), field_name='workspace', allow_slashes=False)
            if str(payload.get('workspace') or '').strip()
            else None
        )
        slot_id = (
            _normalize_external_identifier(payload.get('slot_id'), field_name='slot_id')
            if str(payload.get('slot_id') or '').strip()
            else None
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        history = rotate_chat_history(
            current_instance_id,
            history_dir=CHAT_HISTORY_DIR,
            workspace=workspace,
            slot_id=slot_id,
            source_instance_id=source_instance_id,
            label=str(payload.get('label') or '').strip() or None,
            model=str(payload.get('model') or '').strip() or None,
            backend=str(payload.get('backend') or '').strip() or None,
            capability=str(payload.get('capability') or '').strip() or None,
            fresh_root=_parse_bool(payload.get('fresh_root'), default=False),
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(history)


@app.route('/api/chat_history', methods=['DELETE'])
def api_chat_history_delete():
    try:
        instance_id = _normalize_external_identifier(request.args.get('instance_id'), field_name='instance_id')
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    deleted = delete_chat_history(instance_id, history_dir=CHAT_HISTORY_DIR)
    return jsonify({"deleted": deleted, "instance_id": instance_id})


def _graph_rebase_runtime_readiness() -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate verified registry history plus current durable frame truth."""

    registry_path = _effective_graph_rebase_readiness_registry_path()
    registry_state = _load_graph_rebase_readiness_registry(registry_path)
    if registry_state.get('ok') is not True:
        registry_error = (
            registry_state.get('error')
            if isinstance(registry_state.get('error'), Mapping)
            else {}
        )
        raise GraphRebaseReadinessRegistryError(
            str(
                registry_error.get('code')
                or 'readiness_registry_verification_failed'
            ),
            str(
                registry_error.get('message')
                or 'Graph-rebase readiness registry could not be verified.'
            ),
            details={
                key: value
                for key, value in registry_error.items()
                if key not in {'code', 'message'}
            },
        )
    registry_records = [
        dict(item)
        for item in (registry_state.get('records') or [])
        if isinstance(item, Mapping)
    ]

    current_ledger_path = Path(RESPONSE_FRAMES_DIR) / 'responses.jsonl'
    current_index_path = Path(RESPONSE_FRAMES_DIR) / 'current_index.json'
    empty_current_epoch = (
        not current_ledger_path.exists() and not current_index_path.exists()
    )
    index_state = (
        {
            'ok': True,
            'status': 'empty',
            'runtime_effect': 'none',
            'index_path': str(current_index_path),
            'ledger_path': str(current_ledger_path),
            'ledger_line_count': 0,
            'ledger_size_bytes': 0,
            'response_map_digest': hashlib.sha256(b'{}').hexdigest(),
            'responses': {},
        }
        if empty_current_epoch
        else _load_response_frame_index(frames_dir=RESPONSE_FRAMES_DIR)
    )
    response_index = (
        index_state.get('responses')
        if isinstance(index_state.get('responses'), Mapping)
        else {}
    )
    response_payloads: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    selection = (
        {
            'kind': 'ollmo.graph_rebase_observation_selection',
            'runtime_effect': 'none',
            'indexed_response_count': 0,
            'selected_response_ids': [],
            'selected_response_count': 0,
            'scan_error_count': 0,
            'scan_errors': [],
        }
        if empty_current_epoch
        else _select_graph_rebase_observation_response_ids(
            frames_dir=RESPONSE_FRAMES_DIR,
            index_state=index_state,
        )
    )
    observation_response_ids = [
        str(item).strip()
        for item in (selection.get('selected_response_ids') or [])
        if str(item or '').strip()
    ]
    for response_id in observation_response_ids:
        state = _load_latest_response_observation_state(
            response_id,
            frames_dir=RESPONSE_FRAMES_DIR,
            index_state=index_state,
        )
        payload = (
            state.get('response_payload')
            if isinstance(state.get('response_payload'), Mapping)
            else {}
        )
        if state.get('ok') is True and payload:
            response_payloads.append(
                _project_graph_rebase_readiness_observation(payload)
            )
            continue
        load_errors.append(
            {
                'response_id': response_id,
                'error': dict(state.get('error') or {})
                if isinstance(state.get('error'), Mapping)
                else {'code': 'response_frame_unrecoverable'},
            }
        )
    current_response_ids = {
        str(response_id).strip()
        for response_id in response_index
        if str(response_id or '').strip()
    }
    historical_observations = [
        dict(record.get('observation') or {})
        for record in registry_records
        if isinstance(record.get('observation'), Mapping)
        and str(record.get('response_id') or '').strip()
        not in current_response_ids
    ]
    overlaid_registry_records = len(registry_records) - len(
        historical_observations
    )
    combined_observations = [*historical_observations, *response_payloads]

    trusted_records = _load_graph_rebase_operator_records(
        registry_path=GRAPH_REBASE_OPERATOR_REGISTRY_PATH,
    )
    source_epoch_ids = sorted({
        str(record.get('source_epoch', {}).get('source_epoch_id') or '').strip()
        for record in registry_records
        if isinstance(record.get('source_epoch'), Mapping)
        and str(record.get('source_epoch', {}).get('source_epoch_id') or '').strip()
    })
    source_epoch_digest = hashlib.sha256(
        json.dumps(
            source_epoch_ids,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    source_identity = {
        'kind': 'ollmo.graph_rebase_readiness_multi_epoch_source',
        'runtime_effect': 'none',
        'registry': {
            'registry_sha256': registry_state.get('registry_sha256'),
            'record_count': len(registry_records),
            'unique_response_count': registry_state.get(
                'unique_response_count'
            ),
            'source_epoch_count': len(source_epoch_ids),
            'source_epoch_digest': source_epoch_digest,
        },
        'current_epoch': {
            'status': 'empty' if empty_current_epoch else 'present',
            'response_map_digest': index_state.get('response_map_digest'),
            'ledger_line_count': index_state.get('ledger_line_count'),
            'ledger_size_bytes': index_state.get('ledger_size_bytes'),
            'indexed_response_count': len(response_index),
        },
        'overlay': {
            'current_response_id_count': len(current_response_ids),
            'registry_record_count_excluded': overlaid_registry_records,
        },
    }
    report = _build_graph_rebase_readiness_report(
        combined_observations,
        trusted_review_records=trusted_records,
        corpus_window={
            'selection': (
                'verified_registry_history_overlaid_by_latest_current_epoch'
            ),
            'registry_observation_count': len(registry_records),
            'registry_observation_count_after_current_overlay': len(
                historical_observations
            ),
            'current_epoch_indexed_response_count': len(response_index),
            'current_epoch_observation_count': len(response_payloads),
            'combined_input_observation_count': len(combined_observations),
        },
        source_ledger_identity=source_identity,
    )
    observer = {
        'kind': 'ollmo.graph_rebase_readiness_observer',
        'runtime_effect': 'none',
        'hydrated_response_count': len(response_payloads),
        'selected_graph_rebase_observation_count': len(observation_response_ids),
        'current_epoch_indexed_response_count': len(response_index),
        'current_epoch_observation_count': len(response_payloads),
        'current_epoch_status': 'empty' if empty_current_epoch else 'present',
        'registry_status': (
            'missing_empty'
            if registry_state.get('missing') is True
            else 'verified'
        ),
        'registry_path': str(
            registry_state.get('registry_path')
            or registry_path
        ),
        'registry_sha256': registry_state.get('registry_sha256'),
        'registry_record_count': len(registry_records),
        'registry_observation_count': len(registry_records),
        'registry_unique_response_count': int(
            registry_state.get('unique_response_count') or 0
        ),
        'registry_source_epoch_count': len(source_epoch_ids),
        'registry_record_count_excluded_by_current_overlay': (
            overlaid_registry_records
        ),
        'historical_observation_count_after_current_overlay': len(
            historical_observations
        ),
        'combined_observation_count': len(combined_observations),
        'multi_epoch_source_identity': source_identity,
        'trusted_operator_record_count': len(trusted_records),
        'selection_scan_error_count': int(selection.get('scan_error_count') or 0),
        'registry_error_count': 0,
        'load_error_count': len(load_errors) + int(selection.get('scan_error_count') or 0),
        'load_errors': [
            *(selection.get('scan_errors') or []),
            *load_errors,
        ][:20],
        'index_ok': index_state.get('ok') is True,
    }
    return report, observer


def _graph_rebase_operator_expected_bindings(data: Mapping[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for key in (
        'expected_response_id',
        'expected_frame_id',
        'expected_proposal_id',
        'expected_base_graph_digest',
        'expected_candidate_graph_digest',
        'expected_requested_rebase_class',
    ):
        value = str(data.get(key) or '').strip()
        if not value:
            raise ValueError(f"Parameter '{key}' is required for exact graph-rebase CAS.")
        expected[key] = value
    frame_sequence = data.get('expected_frame_sequence')
    if frame_sequence in (None, ''):
        raise ValueError(
            "Parameter 'expected_frame_sequence' is required for exact graph-rebase CAS."
        )
    try:
        expected['expected_frame_sequence'] = _parse_graph_rebase_frame_sequence(
            frame_sequence
        )
    except ValueError as exc:
        raise ValueError(
            "Parameter 'expected_frame_sequence' must be a positive JSON integer."
        ) from exc
    return expected


def _authenticate_graph_rebase_operator() -> tuple[str, Optional[dict[str, Any]], int]:
    """Require an explicit startup credential before writing trusted truth."""

    configured_token = str(app.config.get('GRAPH_REBASE_OPERATOR_TOKEN') or '')
    configured_identity = str(
        app.config.get('GRAPH_REBASE_OPERATOR_IDENTITY') or ''
    ).strip()
    if len(configured_token) < 32 or not configured_identity:
        return '', {
            'code': 'graph_rebase_operator_credential_not_configured',
            'message': (
                f'{GRAPH_REBASE_OPERATOR_TOKEN_ENV} (at least 32 characters) and '
                f'{GRAPH_REBASE_OPERATOR_IDENTITY_ENV} must both be explicitly '
                'configured at startup before operator mutations are available.'
            ),
        }, 503
    authorization_header = str(request.headers.get('Authorization') or '').strip()
    bearer_token = (
        authorization_header[7:].strip()
        if authorization_header.lower().startswith('bearer ')
        else ''
    )
    provided_token = str(
        request.headers.get('X-Ollmo-Graph-Rebase-Operator-Token')
        or bearer_token
        or ''
    )
    if not provided_token or not secrets.compare_digest(
        provided_token,
        configured_token,
    ):
        return '', {
            'code': 'graph_rebase_operator_authentication_failed',
            'message': 'A valid graph-rebase operator credential is required.',
        }, 401
    operator_identity = str(
        request.headers.get('X-Ollmo-Graph-Rebase-Operator') or ''
    ).strip()
    if (
        not operator_identity
        or len(operator_identity) > 128
        or '*' in operator_identity
        or operator_identity.lower() in {'all', 'any', 'current', 'latest'}
    ):
        return '', {
            'code': 'graph_rebase_operator_identity_invalid',
            'message': 'An exact X-Ollmo-Graph-Rebase-Operator identity is required.',
        }, 400
    if not secrets.compare_digest(operator_identity, configured_identity):
        return '', {
            'code': 'graph_rebase_operator_identity_mismatch',
            'message': 'The authenticated credential is not bound to this operator identity.',
        }, 401
    return operator_identity, None, 200


def _graph_rebase_payload_for_operator(
    response_id: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], int]:
    normalized_id = _normalize_response_lookup_id(response_id)
    durable_state = _load_latest_response_state(
        normalized_id,
        frames_dir=RESPONSE_FRAMES_DIR,
    )
    observation_state = _load_latest_response_observation_state(
        normalized_id,
        frames_dir=RESPONSE_FRAMES_DIR,
    )
    for source, state in (
        ('full_state', durable_state),
        ('observation_state', observation_state),
    ):
        if not isinstance(state, Mapping) or state.get('ok') is not True:
            error = state.get('error') if isinstance(state, Mapping) else None
            return None, {
                'code': 'graph_rebase_operator_durable_truth_unavailable',
                'message': 'Verified durable response truth is required for graph-rebase review.',
                'source': source,
                'load_error': error,
            }, int(state.get('status_code') or 409) if isinstance(state, Mapping) else 409
    durable_errors = durable_state.get('errors')
    if isinstance(durable_errors, list) and durable_errors:
        return None, {
            'code': 'graph_rebase_operator_durable_truth_load_errors',
            'message': 'Durable response truth contains ledger load errors.',
            'load_errors': durable_errors,
        }, 409

    durable_frame = (
        durable_state.get('response_frame')
        if isinstance(durable_state.get('response_frame'), Mapping)
        else {}
    )
    observation_frame = (
        observation_state.get('response_frame')
        if isinstance(observation_state.get('response_frame'), Mapping)
        else {}
    )
    durable_frame_id = str(durable_frame.get('frame_id') or '').strip()
    observation_frame_id = str(observation_frame.get('frame_id') or '').strip()
    durable_sequence = durable_frame.get('frame_sequence')
    observation_sequence = observation_frame.get('frame_sequence')
    if (
        not durable_frame_id
        or durable_frame_id != observation_frame_id
        or durable_sequence != observation_sequence
    ):
        return None, {
            'code': 'graph_rebase_operator_durable_truth_binding_mismatch',
            'message': 'Full and bounded durable projections do not bind the same latest frame.',
            'full_frame_id': durable_frame_id or None,
            'observation_frame_id': observation_frame_id or None,
            'full_frame_sequence': durable_sequence,
            'observation_frame_sequence': observation_sequence,
        }, 409

    payload = (
        copy.deepcopy(durable_state.get('response_payload'))
        if isinstance(durable_state.get('response_payload'), Mapping)
        else {}
    )
    observation_payload = (
        observation_state.get('response_payload')
        if isinstance(observation_state.get('response_payload'), Mapping)
        else {}
    )
    durable_request = (
        observation_payload.get('request')
        if isinstance(observation_payload.get('request'), Mapping)
        else {}
    )
    payload['response_frame'] = copy.deepcopy(dict(durable_frame))
    payload['request'] = copy.deepcopy(dict(durable_request))
    if not payload or not isinstance(payload.get('response_frame'), Mapping):
        return None, {
            'code': 'canonical_response_frame_required',
            'message': 'A frozen response frame is required for graph-rebase review.',
        }, 409
    return payload, None, 200


def _graph_rebase_proposal_from_payload(
    response_payload: Mapping[str, Any],
    proposal_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = (
        response_payload.get('runtime')
        if isinstance(response_payload.get('runtime'), Mapping)
        else {}
    )
    graph = (
        runtime.get('request_phase_graph')
        if isinstance(runtime.get('request_phase_graph'), Mapping)
        else {}
    )
    matches = [
        dict(item)
        for item in graph.get('graph_rebase_proposals') or []
        if isinstance(item, Mapping)
        and str(item.get('proposal_id') or '').strip() == str(proposal_id or '').strip()
    ]
    if len(matches) != 1:
        raise ValueError('Exactly one current graph-rebase proposal must match proposal_id.')
    return dict(runtime), dict(graph), matches[0]


def _persist_graph_rebase_stage_successor(
    response_payload: Mapping[str, Any],
    *,
    proposal_id: str,
    operator_record: Mapping[str, Any],
) -> dict[str, Any]:
    runtime, graph, proposal = _graph_rebase_proposal_from_payload(
        response_payload,
        proposal_id,
    )
    for staged in graph.get('staged_graph_rebases') or []:
        if (
            isinstance(staged, Mapping)
            and str(staged.get('proposal_id') or '').strip() == proposal_id
            and str(staged.get('candidate_graph_digest') or '').strip()
            == str(proposal.get('candidate_graph_digest') or '').strip()
            and str(staged.get('status') or '').strip().lower() == 'staged'
        ):
            return {
                'status': 'already_staged',
                'response_payload': dict(response_payload),
                'lifecycle': dict(staged),
            }
    closure_review = (
        runtime.get('graph_closure_review')
        if isinstance(runtime.get('graph_closure_review'), Mapping)
        else {}
    )
    review = _validate_graph_rebase_proposal(
        proposal,
        request_phase_graph=graph,
        closure_review=closure_review,
        root_prompt=str(
            _extract_responses_current_turn_prompt(
                response_payload.get('request')
                if isinstance(response_payload.get('request'), Mapping)
                else response_payload
            )
            or _extract_responses_prompt(
                response_payload.get('request')
                if isinstance(response_payload.get('request'), Mapping)
                else response_payload
            )
            or ''
        ).strip(),
        accepted_learning_hints=(
            runtime.get('accepted_learning_hints')
            if isinstance(runtime.get('accepted_learning_hints'), Mapping)
            else None
        ),
    )
    if str(review.get('status') or '').strip().lower() != 'accepted':
        return {
            'status': 'blocked',
            'blocked_reasons': review.get('blocked_reasons') or [
                'graph_rebase_stage_review_not_accepted'
            ],
            'review': review,
        }
    lifecycle = _build_graph_rebase_lifecycle(
        request_phase_graph=graph,
        rebase_review=review,
        autonomy_level='stage',
    )
    application = _apply_validated_graph_rebase(
        graph,
        lifecycle,
        autonomy_level='stage',
    )
    if application.get('status') != 'staged':
        return {
            'status': 'blocked',
            'blocked_reasons': application.get('blocked_reasons') or [
                'graph_rebase_stage_application_failed'
            ],
            'review': review,
            'lifecycle': lifecycle,
        }
    staged_payload = copy.deepcopy(dict(response_payload))
    staged_runtime = (
        dict(staged_payload.get('runtime') or {})
        if isinstance(staged_payload.get('runtime'), Mapping)
        else {}
    )
    staged_runtime['request_phase_graph'] = application.get('graph') or graph
    diagnostics = (
        dict(staged_runtime.get('developer_diagnostics') or {})
        if isinstance(staged_runtime.get('developer_diagnostics'), Mapping)
        else {}
    )
    diagnostics['graph_rebase_operator_stage'] = {
        'kind': 'ollmo.graph_rebase_operator_stage_projection',
        'status': 'staged',
        'runtime_effect': 'staged_no_executable_mutation',
        'operator_record_id': operator_record.get('record_id'),
        'proposal_id': proposal_id,
        'rebase_id': lifecycle.get('rebase_id'),
    }
    staged_runtime['developer_diagnostics'] = diagnostics
    staged_payload['runtime'] = staged_runtime
    parent_frame = response_payload.get('response_frame')
    staged_payload['frame_relation'] = {
        'kind': 'graph_rebase_stage_successor',
        'reason': 'operator_reviewed_audit_only_stage',
        'response_id': response_payload.get('response_id') or response_payload.get('id'),
        'parent_response_id': response_payload.get('response_id') or response_payload.get('id'),
        'parent_frame_id': parent_frame.get('frame_id') if isinstance(parent_frame, Mapping) else None,
        'parent_frame_sequence': parent_frame.get('frame_sequence') if isinstance(parent_frame, Mapping) else None,
        'proposal_id': proposal_id,
        'rebase_id': lifecycle.get('rebase_id'),
        'operator_record_id': operator_record.get('record_id'),
    }
    parent_frame_id = str(
        parent_frame.get('frame_id') if isinstance(parent_frame, Mapping) else ''
    ).strip()
    parent_frame_sequence = (
        parent_frame.get('frame_sequence') if isinstance(parent_frame, Mapping) else None
    )
    response_id = str(
        response_payload.get('response_id') or response_payload.get('id') or ''
    ).strip()
    if (
        not response_id
        or not parent_frame_id
        or isinstance(parent_frame_sequence, bool)
        or parent_frame_sequence in (None, '')
    ):
        return {
            'status': 'blocked',
            'blocked_reasons': ['graph_rebase_stage_parent_binding_incomplete'],
        }
    current_state = _load_latest_response_state(response_id, frames_dir=RESPONSE_FRAMES_DIR)
    current_frame = (
        current_state.get('response_frame')
        if isinstance(current_state.get('response_frame'), Mapping)
        else {}
    )
    if (
        current_state.get('ok') is not True
        or str(current_frame.get('frame_id') or '').strip() != parent_frame_id
        or current_frame.get('frame_sequence') != parent_frame_sequence
    ):
        return {
            'status': 'blocked',
            'blocked_reasons': [RESPONSE_FRAME_STALE_PARENT_REASON],
            'expected_parent_frame_id': parent_frame_id or None,
            'current_parent_frame_id': str(current_frame.get('frame_id') or '').strip() or None,
            'expected_parent_frame_sequence': parent_frame_sequence,
            'current_parent_frame_sequence': current_frame.get('frame_sequence'),
        }
    current_frame_request = (
        current_frame.get('request')
        if isinstance(current_frame.get('request'), Mapping)
        else {}
    )
    stage_request = (
        copy.deepcopy(dict(current_frame_request))
        if str(current_frame_request.get('prompt') or '').strip()
        else copy.deepcopy(dict(response_payload.get('request')))
        if isinstance(response_payload.get('request'), Mapping)
        else {}
    )
    try:
        finalized = _finalize_response_frame_payload(
            staged_payload,
            request_payload=stage_request,
            persist=True,
            expected_parent_frame_id=parent_frame_id,
            expected_parent_frame_sequence=parent_frame_sequence,
        )
    except ResponseFrameParentCASMismatch as exc:
        return {
            'status': 'blocked',
            'blocked_reasons': [exc.code],
            **exc.as_dict(),
        }
    response_id = str(finalized.get('response_id') or finalized.get('id') or '').strip()
    finalized_frame = (
        finalized.get('response_frame')
        if isinstance(finalized.get('response_frame'), Mapping)
        else {}
    )
    durable_state = _load_latest_response_state(response_id, frames_dir=RESPONSE_FRAMES_DIR)
    durable_frame = (
        durable_state.get('response_frame')
        if isinstance(durable_state.get('response_frame'), Mapping)
        else {}
    )
    finalized_relation = (
        finalized_frame.get('frame_relation')
        if isinstance(finalized_frame.get('frame_relation'), Mapping)
        else {}
    )
    durable_relation = (
        durable_frame.get('frame_relation')
        if isinstance(durable_frame.get('frame_relation'), Mapping)
        else {}
    )
    if (
        durable_state.get('ok') is not True
        or str(durable_frame.get('frame_id') or '').strip()
        != str(finalized_frame.get('frame_id') or '').strip()
        or str(finalized_relation.get('kind') or '').strip()
        != 'graph_rebase_stage_successor'
        or str(finalized_relation.get('parent_frame_id') or '').strip()
        != parent_frame_id
        or str(durable_relation.get('kind') or '').strip()
        != 'graph_rebase_stage_successor'
        or str(durable_relation.get('parent_frame_id') or '').strip()
        != parent_frame_id
    ):
        return {
            'status': 'blocked',
            'blocked_reasons': ['graph_rebase_stage_successor_not_durable'],
            'lifecycle': lifecycle,
        }
    _touch_response_lookup(
        response_id,
        status=str(finalized.get('status') or 'completed'),
        output_text=str(finalized.get('output_text') or ''),
        response_payload=finalized,
    )
    return {
        'status': 'staged',
        'response_payload': finalized,
        'lifecycle': lifecycle,
        'operator_record': dict(operator_record),
    }


@app.route('/api/graph_rebase/readiness', methods=['GET'])
def api_graph_rebase_readiness_get():
    try:
        report, observer = _graph_rebase_runtime_readiness()
    except (
        GraphRebaseOperatorRegistryError,
        GraphRebaseReadinessRegistryError,
    ) as exc:
        return jsonify({
            'error': str(exc),
            'error_detail': {'code': exc.code, 'details': exc.details},
        }), exc.status_code
    return jsonify({**report, 'observer': observer})


@app.route('/api/responses/<path:response_id>/graph_rebase/operator', methods=['POST'])
def api_response_graph_rebase_operator(response_id: str):
    operator_identity, authentication_error, authentication_status = (
        _authenticate_graph_rebase_operator()
    )
    if authentication_error:
        response = jsonify({
            'error': authentication_error.get('message'),
            'error_detail': authentication_error,
        })
        if authentication_status == 401:
            response.headers['WWW-Authenticate'] = 'Bearer realm="ollmo-graph-rebase-operator"'
        return response, authentication_status
    data = request.get_json(silent=True) or {}
    action = str(data.get('action') or '').strip().lower()
    if action not in {'adjudicate', 'stage', 'authorize_partial'}:
        return jsonify({'error': "Parameter 'action' must be adjudicate, stage, or authorize_partial."}), 400
    if action in {'stage', 'authorize_partial'}:
        autonomy = _describe_graph_rebase_autonomy_from_env()
        if str(autonomy.get('autonomy_level') or '').strip().lower() == 'off':
            return jsonify({
                'error': 'Graph-rebase autonomy is off; rebase transitions are unavailable.',
                'error_detail': {
                    'code': 'graph_rebase_autonomy_off',
                    'message': (
                        'Explicit or fail-closed graph-rebase autonomy off blocks '
                        'the operator transition before trusted transition truth is recorded.'
                    ),
                },
                'autonomy': autonomy,
            }), 409
    try:
        expected = _graph_rebase_operator_expected_bindings(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    if _normalize_response_lookup_id(response_id) != _normalize_response_lookup_id(
        expected['expected_response_id']
    ):
        return jsonify({'error': 'Path response_id does not match expected_response_id.'}), 409
    response_payload, payload_error, payload_status = _graph_rebase_payload_for_operator(response_id)
    if response_payload is None:
        return jsonify({'error': 'Response not available for graph-rebase review.', 'error_detail': payload_error}), payload_status

    adjudication = str(data.get('adjudication') or '').strip().lower()
    if not adjudication:
        adjudication = 'accepted' if action != 'adjudicate' else ''
    reason = str(data.get('reason') or '').strip()
    evidence_refs = data.get('evidence_refs')
    try:
        readiness_report: Optional[dict[str, Any]] = None
        promotion_gate: Optional[dict[str, Any]] = None
        if action in {'stage', 'authorize_partial'}:
            readiness_report, _observer = _graph_rebase_runtime_readiness()
            if int(_observer.get('load_error_count') or 0) > 0:
                return jsonify({
                    'error': 'Graph-rebase readiness corpus is not fully readable.',
                    'observer': _observer,
                    'readiness_report_digest': readiness_report.get('report_digest'),
                }), 409
        if action == 'stage':
            shadow_gate = (
                readiness_report.get('gates', {}).get('shadow_to_stage', {})
                if isinstance(readiness_report, Mapping)
                else {}
            )
            if shadow_gate.get('ready') is not True:
                return jsonify({
                    'error': 'Graph-rebase shadow evidence is not ready for durable stage.',
                    'gate': shadow_gate,
                    'readiness_report_digest': readiness_report.get('report_digest')
                    if isinstance(readiness_report, Mapping)
                    else None,
                }), 409
        if action == 'authorize_partial':
            promotion_gate = _build_partial_graph_rebase_promotion_gate(
                readiness_report or {}
            )
            if promotion_gate.get('status') != 'ready':
                return jsonify({
                    'error': 'Partial graph-rebase evidence is not ready for apply_reviewed authorization.',
                    'gate': promotion_gate,
                    'readiness_report_digest': readiness_report.get('report_digest')
                    if isinstance(readiness_report, Mapping)
                    else None,
                }), 409

        # Read the exact parent again after the potentially expensive corpus
        # evaluation.  The registry's CAS must bind the newest frozen frame,
        # never the stale payload that started this HTTP request.
        current_payload, current_error, current_status = _graph_rebase_payload_for_operator(
            response_id
        )
        if current_payload is None:
            return jsonify({
                'error': 'Response changed before graph-rebase operator commit.',
                'error_detail': current_error,
            }), current_status
        response_payload = current_payload

        operator_record = _record_graph_rebase_operator_action(
            response_payload,
            action=action,
            adjudication=adjudication,
            reason=reason,
            evidence_refs=evidence_refs,
            resolves_record_id=str(data.get('resolves_record_id') or '').strip(),
            trusted_partial_promotion_gate=promotion_gate,
            operator_identity=operator_identity,
            registry_path=GRAPH_REBASE_OPERATOR_REGISTRY_PATH,
            **expected,
        )
    except (
        GraphRebaseOperatorRegistryError,
        GraphRebaseReadinessRegistryError,
    ) as exc:
        return jsonify({
            'error': str(exc),
            'error_detail': {'code': exc.code, 'details': exc.details},
        }), exc.status_code

    if action == 'adjudicate':
        return jsonify({
            'status': 'recorded',
            'runtime_effect': 'none',
            'operator_record': operator_record,
        }), 201

    proposal_id = expected['expected_proposal_id']
    if action == 'stage':
        try:
            staged = _persist_graph_rebase_stage_successor(
                response_payload,
                proposal_id=proposal_id,
                operator_record=operator_record,
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 409
        if staged.get('status') == 'blocked':
            return jsonify(staged), 409
        staged_frame = (
            staged.get('response_payload', {}).get('response_frame', {})
            if isinstance(staged.get('response_payload'), Mapping)
            else {}
        )
        return jsonify({
            'status': staged.get('status'),
            'runtime_effect': 'staged_no_executable_mutation',
            'operator_record': operator_record,
            'lifecycle': staged.get('lifecycle'),
            'response_frame': staged_frame,
        }), 201

    trusted_authorization = _find_trusted_graph_rebase_authorization(
        response_id=expected['expected_response_id'],
        frame_id=expected['expected_frame_id'],
        proposal_id=proposal_id,
        base_graph_digest=expected['expected_base_graph_digest'],
        candidate_graph_digest=expected['expected_candidate_graph_digest'],
        requested_rebase_class=expected['expected_requested_rebase_class'],
        registry_path=GRAPH_REBASE_OPERATOR_REGISTRY_PATH,
    )
    if not trusted_authorization:
        return jsonify({
            'error': 'Trusted partial graph-rebase authorization could not be rejoined.',
        }), 409
    prepared = _RESPONSES_REQUEST_RUNTIME.prepare_terminal_partial_graph_rebase_successor(
        response_payload,
        proposal_id=proposal_id,
        trusted_authorization=trusted_authorization,
        graph_rebase_autonomy='apply_reviewed',
    )
    if prepared.get('status') != 'queued':
        return jsonify({
            'status': prepared.get('status') or 'blocked',
            'blocked_reasons': prepared.get('blocked_reasons') or [
                prepared.get('reason') or 'partial_graph_rebase_successor_not_prepared'
            ],
            'operator_record': operator_record,
        }), 409
    current_record = _get_response_lookup_record(response_id) or {}
    handoff = _LATE_FILL_RUNTIME.persist_and_schedule_partial_graph_rebase_successor(
        prepared,
        source_route_payload=(
            current_record.get('route_payload')
            if isinstance(current_record.get('route_payload'), Mapping)
            else None
        ),
    )
    if handoff.get('status') == 'blocked':
        return jsonify(handoff), 409
    successor_frame = (
        handoff.get('response_payload', {}).get('response_frame', {})
        if isinstance(handoff.get('response_payload'), Mapping)
        else {}
    )
    return jsonify({
        'status': handoff.get('status'),
        'runtime_effect': 'branch_local_partial_successor_queued',
        'operator_record': operator_record,
        'execution': handoff.get('execution'),
        'response_frame': successor_frame,
        'scheduled': handoff.get('scheduled'),
    }), 202


@app.route('/api/responses/<path:response_id>', methods=['GET'])
@app.route('/v1/responses/<path:response_id>', methods=['GET'])
def api_responses_get(response_id: str):
    view = str(request.args.get('view') or request.args.get('mode') or '').strip().lower()
    compact = _parse_bool(request.args.get('compact'), default=False)
    if not compact and view in {'full', 'raw', 'truth'}:
        record = _get_response_lookup_record(response_id, recover_missing=False)
        if not record:
            record, error, status_code = _recover_response_lookup_record_from_frames(response_id)
            if error:
                public_error = 'Response not found.'
                if error.get('code') and error.get('code') != 'response_frame_not_found':
                    public_error = error.get('message') or public_error
                return jsonify({'error': public_error, 'error_detail': error}), status_code
            if not record:
                return jsonify({'error': 'Response not found.'}), 404
        return jsonify(_build_response_lookup_payload(record))

    record, error, status_code = _get_bounded_response_lookup_record(response_id)
    if error:
        public_error = 'Response not found.'
        if error.get('code') and error.get('code') != 'response_frame_not_found':
            public_error = error.get('message') or public_error
        return jsonify({'error': public_error, 'error_detail': error}), status_code
    if not record:
        return jsonify({'error': 'Response not found.'}), 404
    if compact or view in {'status', 'compact', 'observer'}:
        status_payload = _build_response_status_lookup_payload(record)
        status_payload['compact'] = True
        status_payload['object'] = 'response.status'
        status_payload = _response_wire_enforce_byte_ceiling(
            status_payload,
            source_payload=status_payload,
            source='status_projection_emergency_byte_ceiling',
        )
        return jsonify(status_payload)
    if view == 'debug':
        return jsonify(_build_bounded_response_debug_payload(record))
    ui_payload = _build_response_ui_lookup_payload(record)
    ui_payload['ui_compact'] = True
    ui_payload = _response_wire_enforce_byte_ceiling(
        ui_payload,
        source_payload=ui_payload,
        source='ui_projection_emergency_byte_ceiling',
    )
    return jsonify(ui_payload)


@app.route('/api/responses/<response_id>/bundle_artifacts', methods=['POST'])
def api_responses_bundle_artifacts(response_id: str):
    data = request.get_json(silent=True) or {}
    target_name = str(data.get('target_name') or '').strip() or None
    response_payload, error_payload, status_code = _load_response_artifact_bundle_source_payload(
        response_id
    )
    if error_payload:
        return jsonify(error_payload), status_code
    if response_payload is None:
        return jsonify({'error': 'Response has no artifact payload to bundle.'}), 409
    try:
        bundle_payload = _bundle_response_artifacts(
            response_payload,
            target_name=target_name,
            bundle_root=ARTIFACT_BUNDLES_DIR,
        )
        _persist_response_artifact_bundle_record(bundle_payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except OSError as exc:
        logging.exception('Could not bundle response artifacts: %s', exc)
        return jsonify({'error': 'Response artifacts could not be bundled.'}), 500
    except Exception as exc:  # noqa: BLE001
        logging.exception('Unexpected response artifact bundle failure: %s', exc)
        return jsonify({'error': 'Response artifacts could not be bundled.'}), 500

    _log_unified_event(
        category='artifact',
        action='response_artifact_bundle_create',
        status=str(bundle_payload.get('status') or 'bundled'),
        path=bundle_payload.get('bundle_path'),
        response_id=str(response_payload.get('id') or response_id),
        message=f"Created response artifact bundle for {response_id}",
    )
    return jsonify({'status': bundle_payload.get('status') or 'bundled', 'bundle': bundle_payload})


@app.route('/api/responses/<path:response_id>/late_fill/retry', methods=['POST'])
@app.route('/v1/responses/<path:response_id>/late_fill/retry', methods=['POST'])
def api_responses_late_fill_retry(response_id: str):
    return _retry_response_late_fill_branch(response_id, request.get_json(silent=True) or {})


@app.route('/api/responses/<path:response_id>/late_fill/control', methods=['POST'])
@app.route('/v1/responses/<path:response_id>/late_fill/control', methods=['POST'])
def api_responses_late_fill_control(response_id: str):
    return _control_response_late_fill_branch(response_id, request.get_json(silent=True) or {})


@app.route('/api/responses', methods=['POST'])
@app.route('/v1/responses', methods=['POST'])
def api_responses():
    return _handle_responses_request()


def _handle_responses_request(
    *,
    forced_instance_id: Optional[str] = None,
    data_override: Optional[Any] = None,
    upload_override: Any = None,
):
    return _RESPONSES_REQUEST_RUNTIME.handle_responses_request(
        forced_instance_id=forced_instance_id,
        data_override=data_override,
        upload_override=upload_override,
    )


@app.route('/api/local_provider/<path:instance_id>/v1/responses', methods=['POST'])
@app.route('/api/local_provider/<path:instance_id>/responses', methods=['POST'])
def api_local_provider_responses(instance_id: str):
    try:
        instance_id = _normalize_external_identifier(instance_id, field_name='instance_id')
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _handle_responses_request(forced_instance_id=instance_id)

def _execute_chat_request(data: Any):
    return _CHAT_RUNTIME.execute_chat_request(data)


@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.get_json(silent=True) or {}
    return _execute_chat_request(data)

# --- Server starten ---
if __name__ == '__main__':
    try:
        hygiene_summary = cleanup_runtime_hygiene(
            registry_path=Path(CONFIG_FILE_NAME),
            status_path=RUNTIME_STATUS_PATH,
            log_dir=FLASK_LOG_PATH.parent,
            sync_external=False,
            active_global_log_paths=_active_global_log_paths(include_webserver=True),
        )
        print(
            "🧹 Runtime-Hygiene: "
            f"{hygiene_summary.get('live_instance_count', 0)} live instances, "
            f"{hygiene_summary.get('archived_count', 0)} stale logs archived."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Runtime hygiene failed: {exc}")
    print(f"===================================================")
    print(f"🚀 Starting Flask proxy server for the Ollama orchestrator")
    print(f"   Dashboard: http://127.0.0.1:{APP_PORT}/")
    print(f"   Dashboard alias: http://127.0.0.1:{APP_PORT}/dashboard")
    print(f"   Project page: http://127.0.0.1:{APP_PORT}/site/")
    print(f"   Dashboard HTML: {DASHBOARD_HTML_FILE}")
    print(f"   Config file: {CONFIG_FILE_NAME}")
    print(f"===================================================")
    app.run(host='127.0.0.1', port=APP_PORT, debug=False)
