# OLLMO_FOR_AGENTS

`ollmo` is a continuable AI-work state model with a focused local runtime/control-plane body.

Its local Ghost intelligence should understand and explain the runtime, but it is not the orchestration layer. External clients can do the orchestration magic on top.

Ollmo may act agentically for its own narrow goal: local routing, runtime truth, artifact handling, context-candidate hygiene, bounded self-observation, recovery guidance, and external-client integration tunnels. It should not become the general everything-agent.

## Checkout And Saved Builds

Resolve the active checkout from the current working directory or an explicit
`OLLMO_HOME`. Treat saved or historical builds as read-only references unless
the operator explicitly promotes one to the active workspace. Historical
plans can describe superseded intermediate paths and are not edit targets.

For practical repo navigation, start with [Vision Alignment](docs/VISION_ALIGNMENT.md), [Core Contracts](docs/CORE_CONTRACTS.md), [Canonical Stack](docs/CANONICAL_STACK.md), [Principles](docs/PRINCIPLES.md), [Patterns](docs/PATTERNS.md), [Control Knobs](docs/CONTROL_KNOBS.md), [Ghost Runtime Policy](GHOST.md), and [Ghost Router](docs/GHOST_ROUTER.md). Use [Architecture Map](docs/ARCHITECTURE_MAP.md) for code ownership and repo navigation.

Runtime-policy note:

- `GHOST.md` is Ghost's canonical runtime policy source.
- `OLLMO_FOR_AGENTS.md` is the human/operator and external-client guide.

## Core Boundary

- `ollmo` owns runtime state, control-plane APIs, routing hints, artifact handling, and recovery guidance.
- external clients own planning, subagents, debate patterns, recipes, and higher orchestration.
- external clients dock through integration boundaries; Ollmo startup and Ghost routing remain client-neutral.

## First Commands

Inside this repo:

- `./ollmo ctl ghost`
- `./ollmo ctl instances list --json`
- `./ollmo ctl doctor runtime --json`
- `./ollmo ctl models list --json`
- `./ollmo ctl models list --json --runnable-only`

Direct Python entrypoint:

- `python3 scripts/ollmoctl.py ghost`
- `python3 scripts/ollmoctl.py instances list --json`

Agent safety:

- prefer read-only status, manifest, Ghost, and route-preview checks unless the user explicitly asks for lifecycle changes
- do not run clean, archive, reset, start, stop, or execution commands just to inspect docs or route policy
- do not use `/api/responses` for observation or route selection; use runtime files, manifest/running-instance status, Ghost status, or route preview
- read-like `ollmoctl` status helpers default to no control-plane recovery; use the top-level `--recover-control-plane` flag only when you explicitly want local control-plane recovery/start behavior. For strict no-mutation observation, prefer local runtime files first.

Repo-local green-field reset:

- `./ollmo clean`
- `./ollmo archiv`
- `./ollmo archive`
- `./ollmo archive --full`
- `./ollmo clean --dry-run`

This reset helper only touches files inside this repo. By default it clears runtime ballast such as generated contents under `artifacts/` buckets, `logs/`, volatile `state/` history/status/provenance/frame files, and Python/test caches while preserving the standard empty artifact bucket directories, including `artifacts/bundles/`. Missing standard artifact directories are created only as a fallback if they were manually deleted or do not exist yet. It preserves `model_ports.json`, Ghost preferences, Ghost compiled-memory files, bounded self-learning snapshots, and `state/llama_cpp_catalog.json` unless you opt into the deeper flags. Before response-frame cleanup, it collects sidecar refs reachable from active `state/self_learning/` JSON/JSONL, writes `state/self_learning/retention_manifest.json`, and copies retained evidence into `state/self_learning/retained_sidecars/`. Dry-run output shows retained and missing learning sidecar counts. The response-frame cleanup covers the compact ledger, `current_index.json`, and sidecar snapshots under `state/response_frames/`.

`./ollmo archiv` and `./ollmo archive` do the same live cleanup, but first copy the `artifacts/` tree and archive the other useful runtime/generated ballast into `.ollmo_archiv/<timestamp>/` as hidden repo-local data storage. Live artifact cleanup then removes generated contents while preserving the standard bucket directories, including `artifacts/bundles/`; missing standard directories are created only as a fallback.

If you want archive-first rotation with protected Ghost state preserved, use:

- `./ollmo archive --full`

`archive --full` snapshots protected Ghost state when present, but leaves active copies in the checkout. Protected Ghost state includes `state/ghost_preferences.json`, `state/ghost_compiled_memory.json`, `state/ghost_compiled_memory.md`, `state/self_learning/`, `state/self_learning/accepted_policy_snapshot.json`, `state/self_learning/retention_manifest.json`, and retained learning sidecars under `state/self_learning/retained_sidecars/`.

If you want a truly blank repo-local runtime state, use:

- `./ollmo clean --full`
- `./ollmo clean --forget-ghost --reset-registry --reset-llama-catalog`

For `clean`, `--full` is equivalent to `--forget-ghost --reset-registry --reset-llama-catalog`. For `archive`, `--full` resets registry/catalog state but does not imply `--forget-ghost`. Prefer the archive form when rotating real test data so old truth remains inspectable under `.ollmo_archiv/<timestamp>/`.

macOS/Chrome local preview caveat: `clean` and `archive` preserve the standard `artifacts/` bucket structure. Chrome may need Desktop Folder or Full Disk Access permission re-granted before `file://` bundle previews can load `assets/css` or `assets/images`. If Chrome shows `net::ERR_ACCESS_DENIED` for a local `artifacts/bundles/.../index.html`, quit Chrome and re-grant the permission in macOS Privacy & Security, or reset the Desktop Folder permission with `tccutil reset SystemPolicyDesktopFolder com.google.Chrome` and approve it again. Safari may keep working while Chrome is blocked. To make macOS ask again on the next Chrome file access, use the opt-in flag `./ollmo archive --full --reset-chrome-file-access-prompt`.

If you need to reset Ghost's local diagnostic/self-observation state without touching the durable frame ledger, use:

- `python3 scripts/ollmoctl.py ghost --reset-learning-state --json`

That archives the old Ghost event/diagnostic state plus `logs/flask_webserver.log` into `state/ghost_learning_archives/<timestamp>/`, recreates fresh Ghost baselines, removes retired learned-policy files if they are present, and preserves `state/response_frames/responses.jsonl`. If it removes `state/self_learning/`, it archives that directory first.

## Runtime Sources Of Truth

- `model_ports.json`
  Stable runtime registry for currently known local instances.
- `state/runtime_status.json`
  Live readiness/activity/error status beside the stable registry.
- `state/chat_history/`
  Canonical durable conversation history for the active UI, including Responses workbench slots, rotation lineage, persisted request snapshots, and reusable artifact metadata.
- merged per-instance backend/runtime metadata and session-control schemas
  Dynamic truth for what a running instance actually supports right now, including visible controls, required fields, option lists, generic dynamic traits, and backend-native runtime metadata.
- `/api/available_models`
  Catalog/discovery view that may include cached-only entries which are not startable yet; check `runnable` and `disabled_reason` before assuming a source can be launched.
- `artifacts/`
  Canonical user-visible artifact root for generated files, saved inputs, audits, benchmarks, and manifests.
- `logs/`
  Operational diagnostics.

## Canonical Control Plane

Prefer the control plane over raw ports when possible:

- `POST /api/responses`
- `GET /api/running_instances`
- `GET /api/runtime_manifest`
- `GET /api/ghost`

For external OpenAI-style clients, Ollmo also exposes:

- `POST /v1/responses`

Public contract note:

- `POST /api/responses` and `POST /v1/responses` are the official external execution contract.
- `/api/infer` and `/api/chat` remain compatibility-only for older or specialized callers; first-party flows and new clients should stay on `responses`.
- successful non-streaming POSTs and the default, `ui`, `status`, and `debug` reads are bounded public wire projections after canonical frame persistence. The complete serialized outer envelope is at most 8 MiB, including retry/control/status wrappers, and each serialized Responses-style SSE event obeys the same ceiling. These paths never hydrate sidecars.
- public compaction is byte-budgeted, not a fixed character or record-count rule. Ordinary 5,000-character text and 65 tiny records remain intact when they fit. Strings from 256 KiB and collections from 1 MiB may be represented by a bounded preview plus exact length/count, byte size, SHA-256, and an adjacent content-addressed `*_snapshot_ref`.
- lifecycle/frame identity, public outputs/slots/artifact handles, and effective SHA sidecar refs remain available on bounded views without copying hydrated Runtime or Working Frame bodies. `view=debug` adds bounded Closure/rebase diagnostics. `view=full`, `view=raw`, and `view=truth` recursively restore exact CAS-backed truth and may be larger; missing, malformed, or corrupt authoritative refs fail closed with HTTP 409. This is lossless truth by reference, not semantic summarization.

CLI wire/truth choices:

- `ollmo ctl send ... --json` performs one bounded POST and prints that public wire result.
- `ollmo ctl send ... --truth-json` performs the bounded POST, fetches the same response id through exact `view=truth`, and prints a normalized canonical summary.
- `ollmo ctl responses get <response_id> --json` prints the raw canonical truth-view payload; `--truth-json` prints its normalized summary.
- `--json` and `--truth-json` are mutually exclusive for each command.

## Routing Rules

- Use exact `instance_id` values from `ollmo ctl instances list --json` when you want one fixed running target.
- Use `capability` when choosing between chat, vision/OCR, image generation, speech-to-text, and text-to-speech without hard-wiring one specific instance.
- Prefer the control plane's routed/default paths when the caller wants Ollmo to choose among current running capability matches instead of pinning a single instance.
- For Auto/Ghost flows, assume provider choice is dynamic and provider-neutral:
  - Ghost should prefer the instance whose live controls, traits, runtime health, and helper context best match the request
  - do not assume a named provider or family should win unless the user explicitly asked for it and the runtime truth supports it
- Prefer `ollmo ctl ghost` or `GET /api/ghost` when you want the current recommended defaults and recovery hints.
- Prefer `POST /api/ghost_route_preview` when you need the resolved Auto target plus its truthful required controls before execution.
- Treat Ghost route preview as non-execution. Default runtime status truth is cached unless `refresh=true` is explicit, and preview must not start, stop, unload, materialize, write artifacts, freeze response frames, or call `/api/responses`.
- Semantic helper or embedding evidence in Ghost route preview is controlled by non-UI policy `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS` plus the explicit `compute_semantics` request flag. The compute policy default is `off` for omitted or invalid policy values; explicit `compute_semantics=true`, `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS=on`, or `auto` opts into computed preview. Explicit `compute_semantics=false` is passive by default because `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE` defaults to `allow`; operators can set that override to `deny` when policy should overrule explicit false. Automatic UI route preview uses explicit false. Preview responses disclose `truth_mode`, `semantic_compute_requested`, `semantic_compute_performed`, `compute_semantics_source`, `compute_semantics_policy`, and `compute_semantics_false_override`. Here `semantic_compute_requested` is the effective resolved permission after policy and explicit-request handling, not merely the caller's raw flag; `semantic_compute_performed` says whether helper work actually ran.
- Prefer `./ollmo ctl models list --json --runnable-only` when you need a startable source list rather than the broader discovery catalog.

Ghost boundary note:

- route selection is separate from resolver transformation, post-route detail fill, and late fill
- current-turn intent is the primary bracket for fresh requests; old chat/tool/artifact history is available only as explicit reference or continuation context, not as automatic new intent
- the request phase graph is the shared substrate for multi-phase requests such as `write -> speak`, `describe -> image`, or `write -> save`
- `candidate_graph` plus `promotion_review` is the general possibility-to-contract layer for outputs, workload tasks, context, references, evidence, repairs, continuations, and learning hints
- only promoted contracts are owed runtime work; reserved, waived, rejected, stale, or merely possible candidates remain visible state, not pending failures
- branch-local workload tasks carry their own payloads, prompts, dependencies, artifact refs, evidence, output contracts, visibility, review criteria, and freeze state
- deterministic `review_criteria` are runtime/closure checks; they do not create semantic review by themselves
- explicit `semantic_review_criteria` or non-deterministic qualitative criteria are the demand gate for branch semantic review
- the graph is frozen in intent and fluid in state: Ghost anchors what the user asked for, while runtime truth marks branches fulfilled, pending, blocked, failed, or clarified
- canonical `/api/responses` runs one deterministic pre-freeze graph closure review and exposes it as `runtime.graph_closure_review`
- local model calls can produce the current phase or downstream branch output, but the resolver/runtime decides whether graph obligations are fulfilled
- `request_phase_graph.intent_obligations` is the generic current-turn promise ledger. It may decompose a coarse ask into text artifacts, media artifacts, evidence branches, dependency bindings, and navigation promises, but it does not bypass promotion, Closure, or runtime validation.
- if Closure or `intent_graph_adequacy` shows that basic current-turn intent was not represented in the graph, treat it as graph repair work: proposals are advisory until runtime validation accepts a bounded additive patch
- if a producer/consumer binding was planned in parallel or backwards, safe additive dependency repair can add a missing edge before execution; after terminal/frozen state, use target-bound repair or successor/reopen truth rather than mutating the parent frame
- if the whole graph shape must be redrawn or rebased, treat that as reviewed rebase successor truth on the upper rungs of the same scope ladder, not additive graph repair or a new shadow layer: Ghost may propose, but Runtime must compute diff/preservation proof; only the trusted operator chain may advance an exact partial proposal from adjudication through durable stage to authorization
- a correct current-phase route does not mean fields like language, style, speaker, OCR mode, or response format were already filled
- visible assistant text is only one surface; typed semantic payloads and artifact references are more truthful when they are available
- answer/result/output-as-audio wording is a prepare-first contract: current chat creates the substantive answer and a dependent TTS branch speaks exactly that accepted phase payload
- preparation output crosses one canonical acceptance boundary before public completion, persistence, Closure, or downstream materialization; internal planner/control JSON may unwrap exactly one safe `content_payload`, gets at most one same-phase repair attempt, and otherwise stays `repair_needed` with TTS blocked
- later branches should consume branch-local payloads and artifact evidence rather than replaying the full root prompt
- missing upstream artifacts should trigger `repair_dependency_chain`; retrying the same branch without the dependency usually repeats the graph error
- for counted TTS work, explicitly labelled contiguous audio variants are candidate authority: each index supplies exactly one speakable field, while transcript claims, analysis, code, JSON, duplicates, gaps, and ambiguity must not become speech payloads
- a successful TTS fill retains the exact final backend prompt and SHA-256 as `tts_semantic_source`; a directly dependent STT fill compares only its actual transcript with that producer source and records `tts_stt_semantic_evidence`
- missing, drifted, ambiguous, or mismatching TTS/STT evidence blocks with `DEPENDENCY_CHAIN_REPAIR_REQUIRED` / `repair_dependency_chain`; a physically generated wrong WAV may remain immutable evidence, but neither it nor a non-empty transcript fulfills the semantic contract, and expected text must never be injected into STT
- backend-family route-health events should become preference, cooldown, or retry diagnostics by default, not broad provider disablement or graph patches, unless hard runtime evidence or explicit operator disablement proves unavailability
- optional Ghost plan-refinement or critique/reviewer experiments are not the canonical closure loop and should not be treated as required architecture
- legacy `ghost_mode` values (`repair`, `worker`, `explorer`, `improviser`) are API-edge compatibility aliases only
- those aliases translate into `semantic_role_profile` / `semantic_role_orientation_review` as advisory wording; they must not change planner timeout, branch topology, payload authority, promotion, waiver, supersession, execution, or freeze
- explicit frontend play/start is the only frontend path that should force a duplicate same-model start; it uses `start_source: "frontend_button"` and `force_start: true` when a running same-model instance already exists
- Ghost, route preview, late fill, and backend automatic paths must not use `force_start`

## Recovery Order

1. If commands hit `connection refused` on `127.0.0.1:5001`, restore the control plane first.
2. If `5001` is up but no instances are running, start the required model.
3. If an instance is failed, unreachable, or live process/port/backend truth proves it unusable, inspect the runtime state before stopping/restarting that instance.
4. If an instance is only marked `degraded` while process, port, or backend truth still proves it live, treat that as advisory cache/readiness evidence. Refresh or inspect it; do not turn it into a provider ban, offline state, hard recovery, or learning/self-healing failure signal by itself.
5. Treat Late Fill wave candidate snapshots as scheduling evidence, not the whole backend truth. If a snapshot is exhausted, excluded, or has no usable candidate, refresh live runtime truth before declaring that no route exists.
6. Treat MLX/VLM instances by advertised capability, not by package name alone. A VLM with `provider_capabilities` including both `chat` and `vision_analysis` may serve either task when it is the best compatible route; vision-only/OCR-style VLMs remain vision routes.
7. For Single Chat, a text-only selected MLX/VLM turn may rebound from the instance's startup `vision_analysis` label to effective `chat`. If that selected VLM then fails with a provider 5xx/load error, backend runtime may retry once through normal chat routing with the failed instance excluded. This is bounded transport fallback, not provider disablement or degraded truth.

## Artifact Rules

- Do not invent artifact directories.
- Generated files live under `artifacts/`.
- Saved request inputs live under `artifacts/inputs/`.
- Audit/report-style outputs live under `artifacts/audits/`.
- For linked artifact sets, every surviving generated output path must resolve to a concrete saved local artifact before final closure. Placeholder names, guessed paths, stale links, or duplicate reuse before all owed saved artifacts are bound leave the response repair-needed/non-complete unless Closure records a waiver or supersession.
- Reusable user inputs and assistant outputs are normalized into the same message-level artifact ledger in durable history: `message.artifacts[]`.
- `request_snapshot` is request-context metadata, not a separate output store; `request_snapshot.input_artifacts[]` should only reflect explicit user-side uploads/local-path inputs, not assistant outputs mirrored back into user turns.
- canonical durable request/reference state should use `input_artifacts[]` for explicit user inputs and `reference_artifacts[]` for pinned prior outputs/messages.
- Explicit selected reference artifacts/messages are one-turn conversation-scoped anchors in the UI, not a global workbench carry-over state.
- canonical read-side artifact continuity now also exposes `artifact_dossiers` keyed by `artifact_ref`, so one lookup can return artifact identity, provenance, metadata, enrichments, and linked response/message IDs together.
- response artifact bundle operations must not treat truncated, previewed, or emergency wire handles as the complete artifact set. They resolve exact CAS-backed response truth first and fail closed if recursive ref validation cannot produce the complete bundle inputs.
- structured text/file artifact envelopes such as `output_obligations[].content` are payload wrappers; persist the declared content, not the surrounding router/control JSON.
- `state/artifact_registry.jsonl` is the durable materialized artifact index for concrete artifacts. It may merge provenance and integrity metadata by artifact identity; final saved artifact files remain the canonical artifact bytes. Generated-image provenance now lives there as artifact-linked provenance instead of in a separate image-only ledger, and generated-image helper enrichments are appended back onto the original artifact record rather than being treated as fresh inputs.
- `ollmo_webserver.py` owns the Flask route shell and delegates request intake, response semantics, model control, backend transport, chat, infer, Ghost routing, in-flight Responses state, and Late Fill execution to focused modules under `ollmo_server/`. The current ownership map is documented in [Architecture Map](docs/ARCHITECTURE_MAP.md).
- Persistent runtime data lives under `state/`.
- Logs live under `logs/`.

## Response Frame Contract

Treat each Ollmo request as a response frame: an auditable frozen snapshot of input, route decision, target endpoint or integration tunnel, runtime metadata, artifact states, memory delta, errors, and final normalized output.

The fluid middle before freeze now has a first-class mutable owner: `working_frame`. Ghost can pan and edit that state while it is routing, revising artifact plans, applying control hints, or entering a bounded self-heal loop. The `working_frame` carries the goal stack, artifact flow, bounded loop state, revision/self-heal journal, explicit `possibility_space`, and explicit `closure` state for the current request pass.

Current exact canonical Responses truth includes both a `working_frame` object and a `response_frame` object, while bounded normal wire views expose their handles and CAS refs without hydrating those bodies. Non-test runtime calls append the frozen `response_frame` snapshots to `state/response_frames/responses.jsonl`. The frame includes `planning.artifact_flow` for multi-artifact input routing hints, output slots/placeholders, review state, and memory-delta state. `planning.artifact_flow.work_tree` is now the canonical internal tree, and `output_slots` are one projection of that tree rather than the deepest truth. The public API surface now also exposes top-level canonical `outputs`, which are projected from that finalized substrate truth; legacy `output` and `output_text` remain compatibility fields only. The frozen frame mirrors that same canonical output surface under `response_frame.output.outputs`, so replay and lookup paths can read substrate-shaped outputs directly. The frame also includes the final frozen `working_frame` snapshot plus `controls` when non-default effective settings such as voice, seed, image size, OCR limits, or sampling values matter for replay, diagnosis, or bounded self-observation. The mutable and frozen frame surfaces expose `candidate_graph`, `promotion_review`, branch-local workload state, and `artifact_dossiers` keyed by `artifact_ref`. Context and history enter live turns through candidate/promotion gates, not through a separate live memory authority. Self-learning reads frozen frames as audit evidence and may produce accepted learnings only through a reviewed policy snapshot that defaults disabled for new/reset state and can be explicitly enabled with runtime effect `soft_hint_only`.

`state/response_frames/current_index.json` is derived recovery acceleration, not substrate truth. Fresh v2 coverage can prove an unknown response id without scanning the ledger. To migrate an existing complete v1 map, run `.venv/bin/python scripts/attest_response_frame_index.py --check-only`, inspect the exact-match report, then rerun without `--check-only`. The command streams the ledger, preserves the complete entry map and snapshot manifests, and fails without writing if the ledger/index is incomplete, malformed, mismatched, or changes during inspection.

Control snapshots are backend metadata, not automatic user-visible artifacts. Reusable settings artifacts are created only when Ollmo or a client intentionally promotes a snapshot through `/api/settings_artifacts`; those JSON artifacts live under `artifacts/settings/` and expose replay `request_overrides`.

The internal work between frames can remain fluid: retries, narrow resolver rewrites, artifact placeholder filling, route-detail repair, goal-stack edits, and local self-healing are allowed when they serve Ollmo's control-plane job.

Hidden arbitrary hard caps are not architecture. If a bound is technically required, it should be named as an explicit budget or safety knob with a reason, not silently applied to canonical policy, graph, artifact, or contract sources.

Internal file manipulation should go through scoped service helpers. The current helpers are `ollmo_services/scoped_file_tools.py` and `ollmo_services/scoped_command_tools.py`; they default to repo-local roots and are not public filesystem or shell routes.

## Context And Learning Contract

The current live rule is candidate/promotion based:

- prior history, memory, and artifacts may become context candidates
- broader history scan is itself a candidate until promoted
- only promoted context becomes active reference/input for the current turn
- not-promoted context remains visible audit state, not Ghost prompt truth

Bounded self-learning surfaces now include:

- frozen response frames under `state/response_frames/`
- eval cases and reports under `state/self_learning/`
- retention integrity under `state/self_learning/retention_manifest.json`
- learning-owned retained response-frame sidecars under `state/self_learning/retained_sidecars/`
- `state/self_learning/accepted_policy_snapshot.json`
- `scripts/build_self_learning_eval_cases.py`
- `scripts/collect_self_learning_retention_roots.py`
- `scripts/manage_self_learning_policy.py`

`scripts/build_self_learning_eval_cases.py` reverse-streams bounded recent ledger windows instead of loading complete response-frame files. `--frames` may be repeated to combine a fresh active epoch with read-only archived epochs; input order is the same-response-id precedence, while every selected frame hydrates from the CAS root beside its own ledger. It evaluates only the newest durable frame for each response id inside each input window, so a superseded pending Late Fill frame cannot teach an unresolved defect beside its later fulfilled successor. The report keeps `frame_count` as the number of inspected physical rows and separately records `evaluated_response_count` and `superseded_frame_count`. The builder can optionally read `state/ollmo_run_monitor/reports.jsonl` as supporting evidence and emits `shadow_hints` plus retention diagnostics in `state/self_learning/report.json`. Shadow hints are diagnostic-only: they carry `authority: shadow` and `runtime_effect: none`, and they must not alter live routing, Graph, IR, Closure Review, context promotion, output obligations, provider selection, or artifact truth. Retention diagnostics say whether learning's referenced sidecars are present, missing, or copied; after an original epoch moves, only an existing SHA-valid learning-owned retained copy satisfies that root. Retention does not increase learning authority.

The default builder mode remains replacement: without an explicit merge flag, the fresh bounded extraction replaces the eval-case ledger. `--merge-existing` is the opt-in non-destructive update mode. It reads the existing ledger named by `--output`, unions existing and freshly extracted cases by `case_id`, preserves every old-only case, and uses the fresh case when both sets contain the same id. `--max-cases` limits only the fresh extraction, so it never evicts preserved historical cases and the merged ledger may exceed that limit. Preserved cases, including graph-rebase cases whose synthetic source frames are no longer present in the active Response Ledger, remain historical eval evidence only; they do not become current frame, corpus-binding, or runtime truth. Report case counts, improvement candidates, and shadow hints are rebuilt from the merged case set, while current-run frame and graph-rebase coverage fields continue to describe only current source truth. `--no-persist --merge-existing` previews `previous_case_count`, `new_case_count`, `preserved_case_count`, `replaced_case_count`, `removed_case_count`, and the merge policy without writing.

Persisted `eval_cases.jsonl` and `report.json` updates are both serialized and durably staged before either target changes, then each uses whole-file atomic replacement; caught installation failures restore the prior pair, and an interruption cannot expose a truncated JSON or JSONL body. This is not a portable cross-file snapshot transaction: a hard stop precisely between the two replacements can leave complete files from adjacent generations, and rerunning the deterministic command converges them. The CLI holds cooperative per-output locks across read, merge, and commit; direct service workflows that compose these operations must use `self_learning_output_update_lock()` around the same scope. Merge rejects output destinations that resolve onto Response Frame ledgers/indexes, CAS or retained-sidecar trees, graph-rebase corpus state, retention state, monitor input, or accepted-policy state. Merge does not delete or mutate Response Frames, their CAS sidecars, learning-owned retained sidecars, or `accepted_policy_snapshot.json`; accepted learning remains reviewed, proposal-only soft orientation. Policy initialization, promotion, enablement, and disablement remain separate explicit operations.

Use this command for the normal non-destructive update:

```bash
python3 scripts/build_self_learning_eval_cases.py \
  --monitor-reports state/ollmo_run_monitor/reports.jsonl \
  --frame-limit 200 \
  --max-cases 300 \
  --merge-existing
```

Accepted learnings are disabled by default for new or reset snapshots. Promotion records a reviewed candidate, but does not open the gate. In this checkout the accepted snapshot may be enabled as readable policy input when `state/self_learning/accepted_policy_snapshot.json` says `enabled=true`, but its runtime effect remains `soft_hint_only`. Explicit enable exposes bounded hints with concrete hint text, case kinds, severity counts, evidence ids, and a conflict boundary. It does not grant execution, graph-patch, rebase, waiver, or closure authority and cannot override current user intent, runtime truth, Graph, IR, Closure Review, context promotion, output obligations, or artifact evidence.

Accepted learnings may point Ghost at repeated basic-intent failures, such as `intent_graph_inadequacy`, and can orient graph repair proposals. They still cannot validate or apply `runtime.request_phase_graph` patches without current Closure/runtime evidence.

Current intent is checked through `runtime.graph_closure_review.intent_graph_adequacy.intent_lens_review`. Attention covers visible promise ledgers, commitment covers executable producer/consumer bindings and dependency order, and aspiration covers explicit whole-artifact fit promises by promoting semantic-review evidence. These checks live inside Closure truth; they do not let advisory movement, accepted learning, provider health, or frontend state rewrite the graph without runtime evidence.

Backend response-runtime evidence can produce proposal-only graph repairs through `ollmo_services.graph_repair.build_graph_repair_proposals_from_runtime_evidence(...)`, and current response frames should expose those diagnostics under `runtime.request_phase_graph` / `runtime.developer_diagnostics`. Known bridge classes include unmet materialization contracts, terminal pending branches, duplicate artifact refs, broken or fake artifact dependencies, and fulfilled-contract/surface-state mismatches when the surface classifier finds actionable blocked, repair, semantic-review, dependency, artifact, or promoted owed-work evidence. Advisory-only pending surfaces from `controlled_attention_review`, `aspiration_review`, `commitment_review`, or reconsideration remain visible but are not graph-repair evidence by themselves. The Codex-side run monitor may summarize or supplement old frames, but it is observer-only and not product truth. Proposals remain inert until `validate_graph_repair_proposal(...)` accepts them, and only the validated additive patch helper may mutate the graph.

Runtime now attaches `runtime.request_phase_graph.redraw_scope_ladder_review` before graph repair/rebase lifecycle decisions. The ladder is anchored to the current Intent Contract and orders movement as reserved slot/candidate fill, additive repair, binding/dependency repair, artifact-ref identity repair, partial subtree rebase, then full successor rebase. Graph repair proposals can carry `redraw_scope_orientation`, and graph rebase proposals can carry bounded scope fields, but the selected scope is not executable authority. Accepted learning, advisory roles, degraded/provider/cache/liveness evidence, frontend state, monitor-only summaries, and UI labels remain soft orientation or diagnostics only. Duplicate artifact refs are final-output hygiene too: proven aliases may collapse to one projection with preserved alias metadata, while conflicting refs stay `repair_needed` and block successful final projection.

Validated graph patches carry lifecycle truth under `runtime.request_phase_graph.graph_patch_lifecycle`, with staged/applied ledgers and developer diagnostics. If `OLLMO_GRAPH_REPAIR_AUTONOMY` is absent, the product default is `apply_enforced`; if `OLLMO_APPLY_ENFORCED_POLICY` is absent, the product default is `safe_v1`. This is a default-deny pairing: v1 applies only narrow runtime-policy classes for missing branches, dependency repair, artifact binding repair, and proven duplicate artifact alias canonicalization after validation, safe-additive risk classification for safe-additive classes, scope, current evidence, idempotency, and forbidden-evidence gates pass. Explicit autonomy `off` is the full diagnostics-only rollback, explicit enforced-policy `off` disables enforced application, and invalid values fail closed to `off`. `shadow` and `stage` remain non-executable; `apply_safe` remains allowlisted; `apply_reviewed` still needs per-review accepted runtime/operator `graph_patch_authorization` with evidence refs. After a pre-freeze application, Runtime recomputes Closure against the patched graph and schedules only branches whose current repair policy and execution contract classify as executable; a still-blocked branch must not be normalized to pending merely because its graph record was added.

When a safe patch is applied before freeze, Runtime reconciles its new branches into the same response turn's closure gap and Late Fill state. Late Fill resolves the graph from the current artifact payload's `runtime.request_phase_graph` before an older route payload graph, so applied graph truth is not lost to stale routing state. Terminal/frozen parents remain blocked and immutable; when a newly applied safe additive patch creates `runtime.request_phase_graph.successor_reopen_requests[]`, the production terminal owner revalidates the current autonomy/policy, exact parent lineage, patch and graph digests, exact owed branch set, and no-root-replay contract. It then appends a `graph_patch_reopen_successor` frame under the same response id and hands only those branches to the existing Late Fill executor. Both request preparation and materialization-spec construction refuse inherited root/assistant prompt recovery when an exact branch-local payload is absent. The queued/running/terminal execution remains monotone across the complete Late Fill envelope and response projections; stale callbacks cannot restore pending/active work, regress canonical lifecycle, replace the first terminal result, or create the same successor wave again. Explicit graph-repair autonomy `off` prevents every such continuation. Enforced-policy `off` prevents the product-default `apply_enforced` continuation, but not an explicitly selected `apply_safe`. Accepted learning remains `soft_hint_only`; current Closure/runtime evidence and validation are always required.

Reviewed graph rebase/redraw is the higher-risk upper end of the same redraw scope ladder, not a graph-patch shortcut or a parallel layer. Lower bounded additive redraw remains autonomously active through graph-repair `safe_v1`; rebase's own validator and rollout knob isolate the stronger authority required by the partial-subtree and full-successor rungs. Their product default is non-executable `OLLMO_GRAPH_REBASE_AUTONOMY=shadow`; `shadow` is a mode of those rungs, not an additional rung. When the knob is absent, normal startup does not synthesize/export it, preserving product-default provenance. Explicit or fail-closed `off` is the immediate rollback: it blocks `stage` and `authorize_partial` before trusted transition truth or a successor frame is written, while evidence-only adjudication remains possible. Runtime may synthesize a proposal only from a concrete backend-built candidate after post-repair Closure/scope recomputation, settled Late Fill, and proof that no smaller scope is eligible. If active Late Fill consumed an earlier bounded comparison, terminal materialization deterministically re-derives the candidate from final runtime truth. Runtime recomputes the digest and meaningful diff, proves dependency, graph-wide semantic, derived-topology, bookkeeping, failure-visibility, and scope preservation, rejects Ghost-feedback/learning-only/provider/degraded/advisory authority, and records `shadow_no_mutation`.

Read rollout evidence with `GET /api/graph_rebase/readiness`. This endpoint is canonical and read-only: it combines verified bounded observations from the active response-frame epoch with the evidence-only multi-epoch readiness registry, reports evidence and separate promotion gates, and cannot change the current mode. Registry durability is never operator authority. The product default remains `shadow` while evidence is insufficient. For one exact response, `POST /api/responses/<response_id>/graph_rebase/operator` accepts the ordered operator actions `adjudicate`, `stage`, and `authorize_partial`. Every action must carry the exact finalized frame/proposal/base/candidate/class compare-and-swap identities plus a reason and evidence refs. `stage` is durable `staged_no_executable_mutation` truth. `authorize_partial` requires a trusted useful-proposal adjudication, the matching trusted and runtime stage, and a green partial-promotion gate; inline request/model/candidate authorization is not trusted.

Operator mutation is additionally credential-gated: startup must explicitly configure `OLLMO_GRAPH_REBASE_OPERATOR_TOKEN` with at least 32 characters and an exact `OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY`; each request must provide the matching token plus that identity in `X-Ollmo-Graph-Rebase-Operator`. Do not log or persist either value, and do not pass them to model subprocesses. Useful adjudication records receive replay evidence only from Runtime's deterministic frozen-review revalidation. Readiness counts only exact trusted/runtime stage pairs. For a settled candidate that emitted no proposal, `false_negative` is recorded with exact frame/base/candidate bindings and `expected_proposal_id=no_formal_proposal`; this response-bound record is diagnostic only and can never satisfy proposal, stage, or authorization preconditions. It remains historical truth even when one later same-class, replay-verified `useful_proposal` adjudication appends an exact `resolves_record_id` link; only unresolved false negatives block promotion.

Prefer `./ollmo ctl graph-rebase readiness|inspect|adjudicate|stage|authorize-partial` over manual HTTP payloads. The CLI obtains the newest exact CAS identities from durable truth, requires explicit reasons/evidence, rejects ambiguous or locally ineligible proposals, and treats `authorize-partial --execute` as immediate execution authority. It never recovers or starts Ollmo. Credentialed calls are loopback-only on the canonical control-plane port, do not use environment proxies, do not follow redirects, and never accept the token as a visible argument. The requested response id must equal the canonical returned id. For `false_negative`, the option is `--adjudication false_negative` plus an explicit `--rebase-class`; this remains human diagnostic judgment, not an automatic corpus label.

The accepted partial path appends a `graph_rebase_partial_successor` frame under the same response id, preserves the frozen parent, and schedules only the exact branch-local owed work through normal Late Fill. Full-state and bounded-observation projections must identify the same durable parent, and current root truth is rechecked through replay and the final sink. It must not recover work from the root request, root/current phase, assistant output, phase summary, stage direction, instruction, or criteria when a branch-local payload or dependency binding is absent. `successor_rebase_requests[]` is therefore not a general execution queue: untrusted, staged-only, stale, widened, gate-blocked, or full records remain audit/lineage truth, and only the exact registry-trusted partial request may be consumed. Full successor rebase remains shadow/non-executable under safe partial v1.

Core runtime code, operator scripts, and external integration internals remain gated surfaces rather than autonomous rewrite targets. Broad planning and general agent work should stay with external clients.

## External Integrations

Shared external-client integration orchestration belongs under `ollmo_integrations/`.

Current boundary:

- `ollmo_integrations/downstream_sync.py`
- `ollmo_integrations/registry.py`
- `ollmo_integrations/adapter_manifest.py`
- `ollmo_integrations/provider_sync.py`
- `ollmo_integrations/provider_unsync.py`
- `ollmo_integrations/codex/config_sync.py`
- `ollmo_integrations/codex/provider_cleanup.py`
- `ollmo_integrations/codex/provider_unsync.py`

Script paths under `scripts/` are operator commands. External sync, cleanup, adapter-manifest metadata, and unsync remain separate from general startup.

## Current Boundary

In short:

`ollmo = continuable AI-work state model`

`current body = focused local runtime/control plane + Ghost + artifacts + adapters`

`external clients = orchestration and composition around it`
