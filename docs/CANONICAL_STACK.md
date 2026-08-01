# Canonical Stack

For a practical repo-navigation view, read [Architecture Map](ARCHITECTURE_MAP.md). This note focuses on the layered model and canonical boundaries.

The stack is the implementation body for Ollmo's deeper state model:

```text
Intent -> possibility space -> relevance -> promoted contracts -> runtime truth -> review -> freeze
```

Each layer exists to make that loop executable, inspectable, persistent, and available to external clients.

`ollmo` is now organized around five primary layers:

## 1. `runtime core`

Authoritative local runtime substrate.

- Model lifecycle and ports
- `model_ports.json` as source of truth
- `state/runtime_status.json` as the dynamic runtime-status registry beside the stable model registry
- `state/llama_cpp_catalog.json` as the durable source catalog for pulled/registered llama.cpp models
- `ollmo_core/backend_fabric.py` as the normalized backend discovery/lifecycle contract layer above backend-specific runtime managers
- `state/chat_history/` as the canonical durable history store for active UI conversations, including the Responses workbench, lineage rotation state, and the persisted message/request payloads that rebuild the frontend timeline
- `artifacts/` as the canonical user-visible artifact tree, with generated outputs stored directly under `artifacts/`, saved request inputs under `artifacts/inputs/`, and audit/report outputs under `artifacts/audits/`
- Ollama + MLX + llama.cpp runtime handling
- Capability-aware startup/stop
- Backend-native runtime defaults surfaced in registry/status metadata where supported
- manual external integration sync hooks for Codex

Primary package surface:

- `ollmo_runtime/ollama_model_manager.py`
- `ollmo_runtime/llama_cpp_model_manager.py`
- `ollmo_runtime/mlx_model_manager.py`
- `ollmo_runtime/registry.py`
- `ollmo_runtime/lifecycle.py`
- `ollmo_runtime/status.py`
- `ollmo_core/backend_fabric.py`

Compatibility backbone:

- `ollmo_core/registry.py`
- `ollmo_core/lifecycle.py`

## 2. `service layer`

Runtime-adjacent product services built on top of the substrate.

- public execution contract is canonical `/api/responses`; lower-level chat and infer routes remain below that layer only as backward-compatibility wrappers, not as the preferred internal execution path
- Responses payload parsing, envelope shaping, artifact shaping, response-frame snapshots, non-default control snapshots, explicit settings-artifact promotion, and synthetic streaming events
- chat and infer dispatch
- file intake
- scoped internal file tools for policy/memory/artifact files
- scoped internal command tools for control-plane loops
- infer history/cache
- transport and artifact helpers
- OCR/PDF helpers
- Ghost runtime intelligence:
  - request phase graph derivation plus Auto routing from merged per-instance runtime truth
  - pure candidate normalization and promotion review through `ollmo_g/candidate_contracts.py`
  - `candidate_graph` plus `promotion_review` as the general possibility-to-contract layer for outputs, workload tasks, context, references, evidence, repairs, continuations, and learning hints
  - deterministic `review_criteria` as runtime/closure checks, with `semantic_review_criteria` reserved for demand-gated semantic review
  - current-turn-only intake hygiene for fresh turns, with old history/artifacts admitted only as explicit reference or continuation context
  - capability-safe Ghost preference boundaries for Auto routing
  - file-backed semantic roles projected into advisory `semantic_role_profile`; compatibility `ghost_mode` handling is an API-edge alias only
  - post-route detail-fill / control-hint extraction
  - resolver/detail-fill and graph closure review aware of late fill with successor-frame updates
  - branch-local continuation: downstream branches consume their own payloads, prompts, dependency artifacts, and evidence instead of rerunning the full root prompt
  - artifact dossiers as the read-side evidence surface for durable artifact identity, provenance, enrichments, linked messages/responses, and availability
  - structured text/file artifact envelopes such as `output_obligations[].content` are unwrapped to the payload content before persistence
  - route preview/runtime metadata plus archive/diagnostic Ghost memory and bounded self-observation support
  - Ghost-owned image/audio requests now default to prepare-first on chat with downstream materialization branches; plain chat may still end at phase 1
  - local backend model calls execute selected phases or materialize branches; graph/runtime truth decides fulfillment
  - visible file/artifact claims are truth-gated against runtime outputs before freeze

Primary package surface:

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
- `ollmo_services/transports.py`
- `ollmo_services/ocr_pdf.py`
- `ollmo_services/chat_history.py`
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

Compatibility backbone:

- `ollmo_core/inference.py`
- `ollmo_core/file_inputs.py`
- `ollmo_core/history.py`
- `ollmo_core/transports.py`
- `ollmo_core/ocr_pdf.py`

## 3. `ollmo ui`

Operator control room.

- Model management
- Arena comparisons
- Speech/image/vision flows
- Artifact access and logs
- Responses workbench `Auto` routing, preview, and conversation rotation surfaces
- fresh-draft/current-thread-first main workspaces instead of one merged history stream
- durable `History` archive surfaces for reopening older chats independently of the currently running instance set
- durable conversation replay through modular frontend history/rendering helpers

Primary implementation:

- `ollmo_webserver.py`
- `ollmo_server/infer_runtime.py` for infer request-shaping/execution ownership
- `ollmo_server/infer_postprocess.py` for generated-image infer post-processing ownership
- `ollmo_server/responses_runtime.py` for mutable Responses lookup/stream/late fill in-flight ownership
- `ollmo_server/responses_request_runtime.py` for canonical `/api/responses` orchestration and batch-image dimension shaping ownership
- `ollmo_server/request_intake_runtime.py` for request-intake normalization, explicit-target recovery, selected-reference extraction, and Ghost preference coercion ownership
- `ollmo_server/response_semantics_runtime.py` for selected-reference semantics, prepare-phase contracts, semantic phase payloads, graph closure review construction, resolver deferred-gap shaping under `execution_planner`, and late fill state ownership
- `ollmo_server/model_control_runtime.py` for backend-fabric, available-models, and lifecycle route-body ownership
- `ollmo_server/infer_support_runtime.py` for input-artifact persistence, generic file/audio/PDF intake support, infer-history support, and cached PDF lookup/logging ownership
- `ollmo_server/late_fill_runtime.py` for late fill branch resolver/executor ownership
- `ollmo_server/ghost_route_runtime.py` for Ghost route-manifest/context/auto-route plus route-support ownership
- `ollmo_server/ghost_route_runtime.py` also owns current-turn-only route-context hygiene and backend chat-message normalization for fresh turns
- `ollmo_server/backend_transport_runtime.py` for provider request/stream/media adapter ownership
- `ollmo_server/chat_runtime.py` for chat lifecycle and chat-streaming orchestration ownership
- `ollmo_webUI.html` as the page shell, root state, startup wiring, and remaining inline conversation-timeline glue
- `static/ui/messages.js`
- `static/ui/conversations.js`
- `static/ui/settings-history.js`
- `static/ui/models.js`
- `static/ui/message-state.js`
- `static/ui/request-lifecycle.js`
- `static/ui/request-transport.js`
- `static/ui/voice-input.js`

Current frontend history contract:

- `message.artifacts[]` is the canonical reusable-file ledger for both user inputs and assistant outputs
- `message.outputs[]` is the canonical public output surface persisted in history; `output_slots` and `output_branches` remain richer substrate projections beside it
- `request_snapshot` stores durable request metadata and keeps `input_artifacts[]` only for explicit user-side inputs
- selected reference artifacts/messages are conversation-scoped next-turn anchors, not global workbench state
- conversation lineage is rebuilt from persisted conversation metadata plus slot history ids returned by the chat-history service
- active workspace panes render the selected/current conversation, while older durable chats reopen from archive/history surfaces instead of being merged into the active thread pane

## 4. Scripts

Executable command/operator surface.

- `scripts/ollmoctl.py`
- `scripts/startup_model_manager.py`
- `scripts/sync_model_providers.py`
- `scripts/cleanup_model_providers.py`
- `scripts/mlx_whisper_server.py`
- optional diagnostics/utilities such as `scripts/model_provider_overview.py` and `scripts/probe_provider_concurrency.py`
- `scripts/ollmoctl.py ghost` as the CLI surface for runtime-intelligence summaries

The provider sync/cleanup scripts are operator entrypoints. Their external-client implementation lives under `ollmo_integrations/`.

## 5. External Integrations

Consumers and client tunnels for the local runtime substrate.

- shared downstream sync orchestration
- Codex config sync
- current Codex client sync/unsync boundary
- declarative adapter-manifest metadata for external-client docks

Primary implementation:

- `ollmo_integrations/downstream_sync.py`
- `ollmo_integrations/registry.py`
- `ollmo_integrations/adapter_manifest.py`
- `ollmo_integrations/provider_sync.py`
- `ollmo_integrations/provider_unsync.py`
- `ollmo_integrations/codex/config_sync.py`
- `ollmo_integrations/codex/provider_cleanup.py`
- `ollmo_integrations/codex/provider_unsync.py`
- `GHOST.md` as the canonical Ghost runtime-policy source
- `OLLMO_FOR_AGENTS.md` as the human/operator and external-client guide

Operator entrypoints:

- `scripts/sync_model_providers.py`
- `scripts/unsync_model_providers.py`
- `scripts/cleanup_model_providers.py`

Adapter contract note:

- Codex sync remains OpenAI-compatible and targets Ollmo control-plane `/v1` provider URLs.
- shared external-client orchestration should live under `ollmo_integrations/` instead of being folded into general startup.

## Output Layout

User-visible artifacts now live under:

- `artifacts/audio/`
- `artifacts/images/`
- `artifacts/ocr/`
- `artifacts/transcripts/`
- `artifacts/documents/`
- `artifacts/manifests/`
- `artifacts/benchmarks/`
- `artifacts/settings/`
- `artifacts/inputs/`
- `state/response_frames/responses.jsonl`

Mutable/frozen request-state note:

- live request orchestration now uses `ollmo_orchestration/working_frame.py`
- the live working frame keeps explicit `possibility_space`, `candidate_graph`, `promotion_review`, and `closure` state while a request remains fluid
- final frozen snapshots still persist through `ollmo_services/response_frames.py`
- graph-patch reopen after a terminal frame is represented as successor/reopen truth with parent-frame lineage, not as mutation of the old frozen frame
- canonical Responses payloads may expose `runtime.graph_closure_review`, which records pre-freeze fulfillment/pending/blocked truth for the request phase graph
- Archive/diagnostic Ghost memory surfaces can still read recent frozen working-frame outcomes from `state/response_frames/responses.jsonl` together with event-derived learnings and self-observations, while active self-learning keeps retained sidecars under `state/self_learning/retained_sidecars/`; live routing no longer consumes a separate derived memory chain

Naming policy:

- timestamp-first UTC filenames for better chronological sorting in Finder/terminal
- collision-safe suffixing when multiple outputs would otherwise share the same timestamp/name stem

## Startup policy

Default startup should favor the canonical stack:

1. Start/manage models.
2. Start the current Flask UI/API.
3. Leave external client projections untouched unless the operator explicitly runs a sync command.

External integration sync belongs to explicit operator hooks such as `./ollmo sync`, not to start/stop/restart lifecycle points.

Startup selection should stay capability-aware and runnable-only:

- interactive startup surfaces should offer only sources Ollmo can actually launch now
- cached-only discovery entries remain visible through `/api/available_models` and backend-fabric views, but they are not start choices until the required backend contract is available

Canonical project environment:

- `.venv/`

The current HTTP control-plane implementation is Flask.

## Current Boundary

The canonical stack above is the implemented Ollmo 0.1 runtime boundary.

Ghost note:

- Ghost is part of the current canonical stack
- Ghost is not the whole of Ollmo; it is the semantic/current-turn interpretation layer inside the larger runtime/control-plane substrate
- Ghost owns intent anchoring and graph derivation; the resolver and late fill continue open graph obligations; runtime closure review decides what is real before freeze
- file-backed semantic roles now live in `ollmo_g/semantic_roles/*.md` and compile into advisory orientation frames via `semantic_role_profile`; compatibility `ghost_mode` hints remain API-edge aliases and must not shape resolver patience, branch topology, payloads, or runtime truth outside the decision contract
- its job is provider-neutral runtime intelligence inside the control plane:
  - route from live per-instance truth
  - freeze every Ghost-owned request into `request_phase_graph`
  - surface route/runtime metadata back to the UI and clients
  - stay short of full orchestration
