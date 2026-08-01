# Architecture Map

This note is the practical repo-navigation companion to `docs/CANONICAL_STACK.md`.

Use it when you need to answer:

- where the current ownership boundaries are
- how a canonical request flows through the system
- which paths are compatibility-only
- where to edit when a specific subsystem breaks

For the abstract layer model, read `docs/CANONICAL_STACK.md`. For backend-fabric detail, read `docs/BACKEND_FABRIC.md`. For Ghost routing detail, read `docs/GHOST_ROUTER.md`.

Current request/closure diagrams:

- [State/Substrate Architecture](diagrams/ollmo-state-substrate-architecture.html)

## Current Shape

Ollmo is currently organized around one main composition root plus a handful of strong subsystem directories:

- `ollmo_webserver.py`
  Main Flask API/UI composition root for the current stack. Owns the route surface, request normalization, large portions of orchestration glue, route-level response assembly, and the bridge into extracted service/runtime modules.
- `ollmo_server/`
  Internal server-owner package for request-shell-adjacent runtime owners that no longer belong inline in the Flask composition root. Extracted owners now include generated-image infer post-processing in `ollmo_server/infer_postprocess.py`, mutable Responses runtime state in `ollmo_server/responses_runtime.py`, canonical Responses request orchestration in `ollmo_server/responses_request_runtime.py`, request-intake normalization in `ollmo_server/request_intake_runtime.py`, response-semantics shaping in `ollmo_server/response_semantics_runtime.py`, runtime/model control route bodies in `ollmo_server/model_control_runtime.py`, infer-side input/PDF history support in `ollmo_server/infer_support_runtime.py`, late fill branch orchestration in `ollmo_server/late_fill_runtime.py`, infer request shaping/execution in `ollmo_server/infer_runtime.py`, Ghost route-runtime orchestration plus route-support shaping in `ollmo_server/ghost_route_runtime.py`, provider transport/media adapters in `ollmo_server/backend_transport_runtime.py`, and chat request/stream lifecycle orchestration in `ollmo_server/chat_runtime.py`.
- `static/ui/`
  First-party frontend modules. This is the current browser-side behavior layer for conversations, model controls, history, request transport, and voice input.
- `ollmo_services/`
  Product-service layer behind the routes: Responses payload helpers, response-frame snapshots, backend control snapshots, durable chat history, file intake, scoped internal file/command tools, inference helpers, OCR/PDF helpers, events, and transport helpers.
- `ollmo_server/infer_postprocess.py`
  Owner for generated-image infer post-processing: helper-model image-state enrichment, background enrichment scheduling, and artifact-registry provenance persistence for infer results.
- `ollmo_server/responses_runtime.py`
  Owner for mutable Responses runtime state: response lookup registry, response stream registry, and late fill in-flight coordination.
- `ollmo_server/responses_request_runtime.py`
  Owner for canonical `/api/responses` request orchestration: route-preview payloads, response lookup/error/self-heal bookkeeping, chat-vs-infer dispatch, batch handling, batch-image dimension shaping, graph closure review attachment, and final payload freeze.
- `ollmo_server/request_intake_runtime.py`
  Owner for request-intake normalization: wrapper-capability resolution, explicit-target recovery, selected-reference extraction/sanitization, selected-reference message injection, and Ghost preference coercion.
- `ollmo_server/response_semantics_runtime.py`
  Owner for response-semantics shaping: selected-reference follow-up matching, selected-reference prompt-prefix shaping, prepare-phase contract injection, semantic phase-payload truth, graph closure review construction, contract-driven Closure Repair action classification, resolver deferred-gap shaping under `execution_planner`, and late fill state construction.
- `ollmo_server/model_control_runtime.py`
  Owner for runtime/model control route bodies: backend-fabric payloads, available-model normalization, model pull/remove flows, and start/stop lifecycle route handling.
- `ollmo_server/infer_support_runtime.py`
  Owner for infer-support helpers: input-artifact typing/persistence, generic file/audio/PDF intake support, infer-history append/read support, cached PDF insight lookup, and PDF infer event logging.
- `ollmo_server/late_fill_runtime.py`
  Owner for late fill continuation orchestration: deferred payload shaping, bounded execution-contract projection, repair-action scheduling gates, continuation route resolution, branch plan preparation, branch execution/batching, failed-branch recovery context, and worker scheduling.
- `ollmo_server/infer_runtime.py`
  Owner for infer request shaping and infer execution: effective request shaping, session-control defaults, Responses infer payload construction including branch-local execution contracts, infer-result filtering, and the `/api/infer` execution shell.
- `ollmo_server/ghost_route_runtime.py`
  Owner for Ghost route-runtime orchestration and route-support shaping: route-manifest payloads, selected-reference route-context shaping, current-turn-only history hygiene, embedding-helper attachment, trait-aware route support, context-strategy selection, backend chat-message normalization, and current-phase auto-route resolution.
- `ollmo_server/backend_transport_runtime.py`
  Owner for provider request execution, provider stream open/iterate adapters, media transport wrappers, and transport-adjacent parsing helpers.
- `ollmo_server/chat_runtime.py`
  Owner for chat request lifecycle and chat streaming orchestration: Responses chat streaming, runtime status transitions, and the `/api/chat` execution shell.
- `ollmo_g/`
  Ollmo Ghost runtime-intelligence package: provider-neutral current-phase routing, request phase graph derivation, pure candidate/promotion contracts, file-backed semantic roles projected into advisory `semantic_role_profile`, edge-only compatibility `ghost_mode` hints, Ghost-owned intent/phase decisions, control hints, the resolver surface under `execution_planner`, archive/diagnostic Ghost memory surfaces, accepted-learning soft-hint orientation, request metadata, diagnostics, and bounded self-healing context. Ghost is the semantic/current-turn interpretation layer inside Ollmo, not the whole runtime/control-plane substrate. It anchors intent; runtime evidence refines graph state before freeze.
- `ollmo_integrations/`
  External-client integration boundary. It contains the implemented shared sync/unsync orchestration and the current Codex adapter.
- `ollmo_runtime/`
  Backend-specific runtime managers and lifecycle helpers for Ollama, MLX, and llama.cpp.
- `ollmo_core/`
  Shared substrate and compatibility backbone. Contains core registry/lifecycle/status logic plus older shared helpers that still sit underneath the service layer.
- `helpers/`
  CLI/support helpers such as `ollmoctl`, compatibility imports, session-control metadata, and backend-specific helper logic.
- `scripts/`
  Operator-facing scripts and utilities. This includes the Python `ollmoctl` entrypoint wrapper and compatibility/operator sync utilities.

## Top-Level Ownership

### API and UI Boundary

Primary files:

- `ollmo_webserver.py`
- `ollmo_webUI.html`
- `static/ui/request-lifecycle.js`
- `static/ui/request-transport.js`
- `static/ui/messages.js`
- `static/ui/conversations.js`
- `static/ui/settings-history.js`
- `static/ui/models.js`
- `static/ui/message-state.js`
- `static/ui/voice-input.js`

This layer owns:

- HTTP routes
- request/response envelope handling
- page shell and startup wiring
- first-party browser interactions
- the canonical frontend transport to `/api/responses`

Important current truth:

- first-party request execution goes through `/api/responses`
- fresh-turn Auto/Ghost execution uses `current_turn_only` context unless the current turn explicitly asks for prior thread/artifact context
- `ollmo_webserver.py` is still the main integration point even though behavior has been extracted into modules
- generated-image infer post-processing now delegates to `ollmo_server/infer_postprocess.py` instead of living inline in the webserver
- response lookup/stream/late fill in-flight state now delegates to `ollmo_server/responses_runtime.py` instead of living inline in the webserver
- canonical `/api/responses` orchestration now delegates to `ollmo_server/responses_request_runtime.py` instead of living inline in the webserver
- batch-image dimension shaping now delegates to `ollmo_server/responses_request_runtime.py` instead of living inline in the webserver
- request-intake normalization now delegates to `ollmo_server/request_intake_runtime.py` instead of living inline in the webserver
- response-semantics shaping now delegates to `ollmo_server/response_semantics_runtime.py` instead of living inline in the webserver
- pre-freeze graph closure review is built in `ollmo_server/response_semantics_runtime.py`, attached in `ollmo_server/responses_request_runtime.py`, and surfaced as `runtime.graph_closure_review`; open checks can carry repair actions such as backend retry, dependency repair, branch-contract repair, or graph rebuild from promoted obligations
- runtime/model control route bodies now delegate to `ollmo_server/model_control_runtime.py` instead of living inline in the webserver
- infer-side input/PDF history support now delegates to `ollmo_server/infer_support_runtime.py` instead of living inline in the webserver
- late fill branch resolver/executor now delegates to `ollmo_server/late_fill_runtime.py` instead of living inline in the webserver
- infer request shaping/execution now delegates to `ollmo_server/infer_runtime.py` instead of living inline in the webserver
- Ghost route manifest/context/auto-route orchestration and route-support shaping now delegate to `ollmo_server/ghost_route_runtime.py` instead of living inline in the webserver
- provider request/stream/media adapters now delegate to `ollmo_server/backend_transport_runtime.py` instead of living inline in the webserver or chat runtime
- chat request/stream lifecycle orchestration now delegates to `ollmo_server/chat_runtime.py` instead of living inline in the webserver
- `ollmo_webUI.html` is now mostly the page shell and remaining glue, not the whole frontend
- selected reference artifacts/messages are conversation-scoped next-turn anchors in the UI, not global workbench state
- `request_snapshot.input_artifacts[]` should track explicit user-side inputs only, while assistant outputs stay in `message.artifacts[]`

### Service Layer

Primary files:

- `ollmo_services/inference.py`
- `ollmo_services/responses.py`
- `ollmo_services/response_frames.py`
- `ollmo_services/frame_planning.py`
- `ollmo_services/control_snapshots.py`
- `ollmo_services/settings_artifacts.py`
- `ollmo_orchestration/working_frame.py`
- `ollmo_services/file_inputs.py`
- `ollmo_services/scoped_file_tools.py`
- `ollmo_services/scoped_command_tools.py`
- `ollmo_services/history.py`
- `ollmo_services/chat_history.py`
- `ollmo_services/transports.py`
- `ollmo_services/ocr_pdf.py`
- `ollmo_services/events.py`
- `ollmo_g/payload.py`
- `ollmo_g/semantic_role_profile.py`
- `ollmo_g/ghost_mode_compat.py`
- `ollmo_g/semantic_roles/registry.py`
- `ollmo_g/semantic_roles/*.md`
- `ollmo_g/router.py`
- `ollmo_g/control_hints.py`
- `ollmo_g/request_phase_graph.py`
- `ollmo_g/candidate_contracts.py`
- `ollmo_g/execution_planner.py`
- `ollmo_g/memory.py`
- `ollmo_g/request_meta.py`
- `ollmo_g/image_state.py`
- `ollmo_g/diagnostics_harness.py`

This layer owns:

- request-time capability handling
- Responses payload parsing, envelope shaping, artifact shaping, response-frame snapshot shaping, non-default control snapshot shaping, explicit settings-artifact promotion, and synthetic streaming event generation
- durable chat-history shaping
- artifact-aware request metadata
- scoped read/write/copy/replace helpers for internal policy, memory, and artifact files
- scoped argv-only command execution helpers for internal control-plane loops
- OCR/PDF execution helpers
- event logging helpers
- Ghost request-phase derivation, current-phase routing, detail-fill, diagnostics, archive/diagnostic memory surfaces, and bounded self-observation support
- pure candidate normalization and promotion review through `ollmo_g/candidate_contracts.py`; candidates describe possibility, while promotion review decides which candidates become executable contracts
- branch-local workload task continuation; downstream branches consume focused payloads, prompts, dependencies, artifact refs, and evidence rather than the whole root prompt
- artifact dossiers keyed by durable `artifact_ref`, with provenance, metadata, enrichments, linked response/message ids, and availability as reusable evidence
- file-backed semantic roles projected into advisory `semantic_role_profile`; compatibility `ghost_mode` hints are accepted only at the API edge
- current-turn-only route context hygiene for fresh turns, with referential history/artifact context only when the current request points back
- deterministic graph closure review before response freeze
- truth-gated visible file/artifact claims before final response freeze

### Runtime Layer

Primary files:

- `ollmo_runtime/ollama_model_manager.py`
- `ollmo_runtime/mlx_model_manager.py`
- `ollmo_runtime/llama_cpp_model_manager.py`
- `ollmo_runtime/lifecycle.py`
- `ollmo_runtime/registry.py`
- `ollmo_runtime/status.py`
- `ollmo_runtime/runtime_hygiene.py`
- `ollmo_runtime/runtime_log_hygiene.py`

This layer owns:

- backend-specific start/stop behavior
- per-backend runtime probing and validation
- instance metadata and runtime-state tracking
- runtime cleanup/hygiene helpers

### Core Backbone

Primary files:

- `ollmo_core/backend_fabric.py`
- `ollmo_core/registry.py`
- `ollmo_core/lifecycle.py`
- `ollmo_core/status.py`
- `ollmo_core/inference.py`
- `ollmo_core/file_inputs.py`
- `ollmo_core/history.py`
- `ollmo_core/transports.py`
- `ollmo_core/ocr_pdf.py`

This layer owns:

- shared substrate logic used across runtime/service layers
- backend-fabric normalization
- compatibility-era shared helpers that still underpin newer service modules

Treat `ollmo_core/` as important shared infrastructure, not as the preferred place for new product-level route behavior unless the change is clearly core/shared.

### CLI and Operator Surface

Primary files:

- `ollmo`
- `helpers/ollmoctl.py`
- `scripts/ollmoctl.py`
- `start_multi_models.sh`
- `stop_multi_models.sh`
- `restart.sh`
- `scripts/sync_model_providers.py`
- `ollmo_integrations/downstream_sync.py`
- `ollmo_integrations/registry.py`
- `ollmo_integrations/adapter_manifest.py`
- `ollmo_integrations/provider_sync.py`
- `ollmo_integrations/codex/config_sync.py`
- `ollmo_integrations/codex/provider_cleanup.py`

This layer owns:

- operator entrypoints
- stable CLI command surface for local control-plane access
- manual downstream provider/config sync for external tools

Important current truth:

- `./ollmo` is the operator wrapper
- `scripts/ollmoctl.py` is the thin Python entrypoint
- `helpers/ollmoctl.py` contains the real CLI behavior
- `ollmoctl send` targets `/api/responses`

### External Integration Tunnel

Primary files:

- `ollmo_integrations/downstream_sync.py`
- `ollmo_integrations/registry.py`
- `ollmo_integrations/adapter_manifest.py`
- `ollmo_integrations/provider_sync.py`
- `ollmo_integrations/provider_unsync.py`
- `ollmo_integrations/codex/config_sync.py`
- `ollmo_integrations/codex/provider_cleanup.py`
- `ollmo_integrations/codex/provider_unsync.py`
- `scripts/sync_model_providers.py`
- `scripts/unsync_model_providers.py`
- `scripts/cleanup_model_providers.py`

This layer owns:

- shared sync orchestration for external clients
- declarative adapter-manifest metadata for external-client docks
- client-specific config projection through dedicated integration subpackages
- manual sync, unsync, cleanup, and execution support for the Codex integration
- a connector boundary implemented for Codex

Important current truth:

- general startup should not absorb every external-client integration detail
- `ollmo_integrations/` is the implementation home for external-client sync, unsync, and cleanup behavior
- `ollmo_integrations/registry.py` is the adapter lookup layer for client subpackages
- `ollmo_integrations/adapter_manifest.py` is the declarative adapter-manifest layer for external-client docks
- `scripts/sync_model_providers.py`, `scripts/unsync_model_providers.py`, and `scripts/cleanup_model_providers.py` remain stable operator commands
- Ghost `primary_target` / `fallback_target` preferences are Ghost/chat-routing hints, not blanket downstream execution locks across every capability
- canonical `/api/responses` execution can freeze a truthful chat moment first, then continue late image/audio artifact fill under the same `response_id` when the real multimodal artifact lands after the initial close

## Canonical Request Flow

The current canonical path is:

1. caller builds a request
   - first-party UI via `static/ui/request-lifecycle.js` and `static/ui/request-transport.js`
   - CLI via `helpers/ollmoctl.py`
2. request hits `POST /api/responses` in `ollmo_webserver.py`
3. the route layer normalizes payload, derives or updates the request phase graph and candidate graph, resolves promotion review, resolves the current truthful phase route, and prepares effective request data
   - every Ghost-owned request freezes into a request phase graph
   - `candidate_graph` plus `promotion_review` is the general possibility-to-contract layer for outputs, workload tasks, context, references, evidence, repairs, continuations, and learning hints
   - promoted workload tasks carry branch-local inputs, dependencies, lifecycle, output contract, visibility, review criteria, and optionally validated Ghost-proposed execution-contract details
   - accepted workload task enrichments are projected into downstream branch records before late fill execution
   - plain chat may end at phase 1
   - Ghost-owned image/audio requests normally keep the current phase on `chat` and continue through downstream materialization branches
4. service/helpers handle Responses payload/envelope shaping, file intake, history metadata, transport details, OCR/PDF handling, Ghost route hints, and continuation metadata
5. runtime managers execute the selected backend/runtime for the current phase
   - capability-safe Ghost preference gating prevents chat locks from hijacking image/TTS execution
   - chat-mode execution can freeze a completed text moment while marking unresolved image/TTS output as normal `late_fill` continuation work
6. Ollmo persists durable outputs and metadata
   - `artifacts/` for user-visible files
   - response lookups for the current mutable view of an evolving `response_id`
   - `state/response_frames/` for response-frame JSONL snapshots
   - successor response frames when late fill or graph-patch reopen changes a previously frozen response
   - `state/chat_history/` for durable conversation state
   - `state/runtime_status.json` for live runtime state
   - `model_ports.json` for stable registry truth
7. the normalized response envelope returns to UI/CLI callers

Short version:

- `UI or ollmoctl`
- `-> /api/responses`
- `-> phase graph, candidate promotion, and current-phase resolution`
- `-> services`
- `-> runtime manager`
- `-> artifacts and history`
- `-> normalized response`

## Data and Runtime Directories

These are the important durable/runtime roots:

- `model_ports.json`
  Stable instance registry.
- `state/runtime_status.json`
  Live readiness, activity, and runtime-health layer.
- `state/chat_history/`
  Canonical durable chat/history store for the active UI and Responses workbench.
- `state/response_frames/`
  Local JSONL response-frame ledger for final canonical Responses payload snapshots.
- `state/self_learning/`
  Bounded self-learning reports, accepted-policy snapshots, retention manifests, and learning-owned retained sidecars.
- `state/infer_history.jsonl`
  Persisted infer-history cache for compatibility/specialized flows such as PDF inference history.
- `artifacts/`
  Canonical user-visible artifact root.
- `logs/`
  Operational diagnostics only. Safe to clean when you want a fresh local runtime state.

Canonical Responses payloads now include a mutable `working_frame` plus a frozen `response_frame`, and non-test runtime calls append final frames under `state/response_frames/`. The working frame is built by `ollmo_orchestration/working_frame.py` and carries the live goal stack, bounded loop metadata, artifact-flow plan, candidate graph, promotion review, review state, revision/self-heal journal, explicit `possibility_space`, and explicit `closure` state for the fluid middle. The frozen frame captures input, route decision, target endpoint or integration tunnel, runtime metadata, artifact states, memory deltas where available, errors, final normalized output, and the final `working_frame` snapshot. It also includes `planning.artifact_flow` from `ollmo_services/frame_planning.py`. When non-default effective controls matter, it includes `controls` from `ollmo_services/control_snapshots.py`; those snapshots are backend replay/diagnostic metadata and are not automatically promoted to user-visible settings artifacts. `ollmo_services/settings_artifacts.py` and `/api/settings_artifacts` provide the explicit promotion path for reusable JSON settings artifacts under `artifacts/settings/`. Response frames and event logs now carry the durable runtime truth that replaced the old compiled-memory-as-live-routing-authority model.

Request-shape reminder:

- every Ghost-owned request freezes into `request_phase_graph`
- `candidate_graph` and `promotion_review` keep possible, reserved, rejected, waived, and promoted work visible without turning every possibility into owed work
- text/file artifact wrappers such as `output_obligations[].content` are payload envelopes; persistence saves the declared content, not router/control JSON
- plain chat may end at phase 1
- Ghost-owned image/audio requests normally keep the current phase on `chat` and continue through downstream materialization branches
- visible file/artifact claims in assistant text are truth-gated by runtime outputs before freeze
- explicit direct contracts remain explicit exceptions
- accepted-learning soft hints can orient Ghost and decision contracts when the accepted snapshot is enabled, but they are not runtime truth
- backend runtime evidence can synthesize graph-repair proposals for known repair classes into response truth; validation still decides whether an additive graph patch is allowed, and monitor output remains observer-only

Inside `artifacts/`, the current main buckets are:

- `artifacts/audio/`
- `artifacts/audits/`
- `artifacts/images/`
- `artifacts/ocr/`
- `artifacts/transcripts/`
- `artifacts/documents/`
- `artifacts/bundles/`
- `artifacts/manifests/`
- `artifacts/benchmarks/`
- `artifacts/settings/`
- `artifacts/inputs/`

## Compatibility and Non-Canonical Edges

These still exist, but they are not the preferred first-party path:

- `/api/infer`
  Lower-level compatibility and specialized capability route backed by the same shared infer execution used by `/api/responses`. It may receive a bounded `execution_contract` from `/api/responses`, but it is executor substrate, not workload-planning authority.
- `/api/chat`
  Lower-level chat compatibility wrapper for specialized/simple direct callers.
- `ollmo_core/*`
  Still active and important, but often underneath newer `ollmo_services/*` surfaces.
- `ollmo_orchestration/*`
  Active again for mutable pre-freeze orchestration state such as `working_frame`, but still not the place for external-client adapter logic.

If you are adding or changing a first-party flow, start from `/api/responses` and only touch `/api/infer` or `/api/chat` when the change is intentionally compatibility-oriented.

## Where To Change What

### Request Routing or Target Selection

Start here:

- `ollmo_webserver.py`
- `ollmo_g/semantic_role_profile.py`
- `ollmo_g/ghost_mode_compat.py`
- `ollmo_g/semantic_roles/registry.py`
- `ollmo_g/router.py`
- `ollmo_g/control_hints.py`
- `ollmo_g/candidate_contracts.py`
- `ollmo_g/execution_planner.py`
- `ollmo_g/payload.py`

Tests to read first:

- `tests/test_responses_api.py`
- `tests/test_ghost_router.py`
- `tests/test_candidate_contracts.py`
- `tests/test_ghost_execution_planner.py`
- `tests/test_ghost_diagnostics_harness.py`

### Responses Payloads, Artifacts, or Streaming Envelopes

Start here:

- `ollmo_orchestration/working_frame.py`
- `ollmo_services/responses.py`
- `ollmo_services/response_frames.py`
- `ollmo_services/frame_planning.py`
- `ollmo_services/control_snapshots.py`
- `ollmo_services/settings_artifacts.py`
- `ollmo_services/scoped_file_tools.py`
- `ollmo_services/scoped_command_tools.py`
- `ollmo_webserver.py`

Tests to read first:

- `tests/test_responses_api.py`
- `tests/test_infer_api.py`
- `tests/test_chat_api.py`

### Durable Chat History or Artifact Metadata

Start here:

- `ollmo_services/chat_history.py`
- `ollmo_services/history.py`
- `ollmo_webserver.py`
- `static/ui/message-state.js`
- `static/ui/settings-history.js`
- `static/ui/conversations.js`

Tests to read first:

- `tests/test_chat_history_api.py`
- `tests/test_chat_history_service.py`

### Backend Startup, Runtime Health, or Model Registration

Start here:

- `ollmo_runtime/ollama_model_manager.py`
- `ollmo_runtime/mlx_model_manager.py`
- `ollmo_runtime/llama_cpp_model_manager.py`
- `ollmo_runtime/lifecycle.py`
- `ollmo_runtime/status.py`
- `ollmo_core/backend_fabric.py`
- `ollmo_core/registry.py`

Tests to read first:

- `tests/test_backend_fabric.py`
- `tests/test_runtime_core.py`
- `tests/test_runtime_status_api.py`
- `tests/test_ollama_model_manager.py`
- `tests/test_mlx_model_manager.py`
- `tests/test_llama_cpp_model_manager.py`

### CLI or Operator Commands

Start here:

- `ollmo`
- `helpers/ollmoctl.py`
- `scripts/ollmoctl.py`
- `start_multi_models.sh`
- `stop_multi_models.sh`

Tests to read first:

- `tests/test_ollmoctl.py`

### External Client Sync, Unsync, or Connectors

Start here:

- `ollmo_integrations/downstream_sync.py`
- `ollmo_integrations/registry.py`
- `ollmo_integrations/adapter_manifest.py`
- `ollmo_integrations/provider_sync.py`
- `ollmo_integrations/codex/config_sync.py`
- `ollmo_integrations/codex/provider_cleanup.py`

Tests to read first:

- `tests/test_sync_model_providers.py`
- `tests/test_codex_config_sync.py`
- `tests/test_provider_cleanup.py`

### Frontend Request Transport or UX Behavior

Start here:

- `ollmo_webUI.html`
- `static/ui/request-lifecycle.js`
- `static/ui/request-transport.js`
- `static/ui/messages.js`
- `static/ui/models.js`
- `static/ui/voice-input.js`

When debugging browser-visible behavior, check whether the issue lives in:

- request construction
- local UI state shaping
- backend response envelope
- durable history replay

## Recommended Reading Order

If you are new to the repo, this order gives the fastest orientation:

1. `README.md`
2. `docs/ARCHITECTURE_MAP.md`
3. `docs/CANONICAL_STACK.md`
4. `GHOST.md`
5. `OLLMO_FOR_AGENTS.md`
6. `docs/BACKEND_FABRIC.md`
7. `docs/GHOST_ROUTER.md`
8. `helpers/ollmoctl.py`
9. `ollmo_webserver.py`

## Current Practical Summary

If you want the shortest possible mental model, use this:

- `ollmo_webserver.py` is the current API/UI composition root
- `/api/responses` is the canonical first-party execution path
- `ollmo_services/` is the main product-logic layer behind that path
- `ollmo_g/` is the Ghost runtime-intelligence package, including file-backed semantic roles and edge-only `ghost_mode` compatibility projected into advisory `semantic_role_profile`
- `ollmo_integrations/` is the external-client integration tunnel
- `ollmo_integrations/adapter_manifest.py` is the adapter-manifest surface for client docks
- `ollmo_runtime/` owns backend-specific runtime truth and lifecycle
- `ollmo_core/` is the shared substrate and compatibility backbone
- `static/ui/` is the modular frontend
- `helpers/ollmoctl.py` is the real CLI implementation
- `artifacts/` and `state/chat_history/` are the main durable user-facing data roots
- `state/response_frames/` is the local response-frame ledger for final `/api/responses` snapshots
