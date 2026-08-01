# Ollmo Contract

Use this reference first for normal Codex work with Ollmo. Keep larger Ollmo docs out of context unless the task needs them.

## Ollmo Checkout Resolution

## Codex And Ollmo/Ghost Boundary

Codex is the user-facing supervisor, workspace editor, reviewer, cross-tool coordinator, and final narrator in the current thread. Ollmo is the local runtime truth substrate. Ghost may own local route planning, phase graph formation, contract shaping, closure checks, and no-prose-only completion checks for Ollmo work.

Ollmo is also an optional local execution surface inside broader Codex work. Before a suitable text, image, vision/OCR, speech, embedding, or multimodal tool call, consider whether an already-running Ollmo capability materially improves privacy, locality, context/token use, repeatability, or durable artifact evidence. Do not force Ollmo when another requested or deterministic tool is a better fit, do not start a model without explicit permission, and target confirmed local runtime truth when the work must stay local.

When handing a suitable bounded subtask to a subagent, write `Use $ollmo` and include the desired capability, inputs or artifact refs, expected output, and closure criteria. If current runtime truth already resolved one exact target, pass its `instance_id` and use a direct `/api/responses` request; use `capability` when target identity does not matter. Do not ask Ghost to repeat a resolved single-capability route. Use Ghost when route choice, phase formation, dependencies, multimodal graph shaping, or closure planning remains useful. Never bypass the control plane through a raw provider port. Codex remains responsible for the overall task, integration, validation, and final narration.

When using Ollmo, treat response-frame and artifact state as substrate truth. Do not reduce Ollmo results to model prose, and do not override runtime lifecycle, output, artifact, or graph-closure truth with Codex assumptions.

Accepted learning is reviewed soft orientation only. It may help Ghost and decision contracts notice repeated patterns, but it cannot promote, execute, waive, supersede, validate graph patches, satisfy review, freeze closure, or override current runtime truth.

Resolve the active checkout in order:

1. `$OLLMO_HOME`, if it points to an Ollmo checkout.
2. The current workspace root, if it contains `ollmo`, `ollmo_webserver.py`,
   and `ollmo_core/`.
3. Ask the user for the checkout path.

Do not search arbitrary parent directories for a checkout. Treat archived,
saved, or older checkouts as read-only references unless the user explicitly
selects one as active.

Do not assume Ollmo is running. Check local files or the control plane before sending execution requests.

## Read-Only Status Order

Use read-only checks first. Inspect local files:

- `<ollmo>/model_ports.json`
- `<ollmo>/state/runtime_status.json`

Then try direct control-plane HTTP if the sandbox permits it and the user has not forbidden endpoint probes:

- `GET http://127.0.0.1:5001/api/runtime_manifest`
- `GET http://127.0.0.1:5001/api/running_instances`
- `GET http://127.0.0.1:5001/api/backend_fabric`
- `GET http://127.0.0.1:5001/api/ghost`
- `GET http://127.0.0.1:5001/api/runtime_status`

A direct HTTP failure such as `EPERM`, `Operation not permitted`, `connection refused`, or `couldn't connect` is inconclusive until runtime files have also been checked. If files show running instances, answer from that truth and mention that direct HTTP was unavailable from the current Codex tool process.

Read-like `ollmoctl` commands default to no control-plane recovery in the current repo. Use explicit `--recover-control-plane` only when the user wants recovery/start behavior; `ollmoctl responses get` follows the same rule. For strict observation, prefer local files and already-running cached/passive HTTP surfaces.

Do not run lifecycle commands during read-only status, Ghost, routing, capability, or runtime-truth requests.

Read-only mode must not write `model_ports.json`, `state/runtime_status.json`, response frames, artifacts, logs, Codex config, or downstream provider projections.

The read/status surfaces are cached/passive observer surfaces by default: `runtime_manifest`, `running_instances`, `backend_fabric`, `ghost`, and `runtime_status` may read existing runtime files, frames, registries, history, logs, and artifacts, but they must not probe backend ports, fetch backend runtime metadata, recover the control plane, start/stop/unload models, execute response work, or write status files unless `refresh=true` is explicit. `refresh=true` may probe local process/port/backend facts and update `state/runtime_status.json`, but it is still not execution and must not materialize branches, write artifacts, or freeze response frames.

## Runtime Truth Sources

Prefer these sources over Codex config projections:

- `<ollmo>/model_ports.json`: stable registry of known local instances.
- `<ollmo>/state/runtime_status.json`: live readiness, process, port, activity, and error status.
- `GET http://127.0.0.1:5001/api/runtime_manifest`: merged live runtime manifest and capability/control truth.
- `GET http://127.0.0.1:5001/api/running_instances`: currently running instances.
- `GET http://127.0.0.1:5001/api/backend_fabric`: cached backend/capability fabric truth.
- `GET http://127.0.0.1:5001/api/ghost`: Ghost defaults and recovery guidance.
- `GET http://127.0.0.1:5001/api/runtime_status`: cached runtime status projection.
- `GET http://127.0.0.1:5001/api/ghost_preferences`: current preferred Ghost routing preferences.
- `GET http://127.0.0.1:5001/api/responses/<response_id>?view=status`: compact observer truth for existing response work.

Codex `model_providers` blocks may be downstream projections from Ollmo. They are not the upstream runtime registry.

Live process, listening port, backend runtime, and control-plane truth beat stale readiness projections. A live `degraded`, `busy`, timeout, cooldown, cache, or provider-family warning is advisory route-health evidence unless hard liveness proves the instance unusable. Do not turn advisory degraded state into `failed`, offline, a broad provider ban, or graph repair proof by itself.

## Control-Plane Endpoints

Read-only status and diagnostics:

- `GET http://127.0.0.1:5001/api/runtime_manifest`
- `GET http://127.0.0.1:5001/api/running_instances`
- `GET http://127.0.0.1:5001/api/backend_fabric`
- `GET http://127.0.0.1:5001/api/ghost`
- `GET http://127.0.0.1:5001/api/runtime_status`
- `GET http://127.0.0.1:5001/api/ghost_preferences`
- `GET http://127.0.0.1:5001/api/responses/<response_id>?view=status`

Pre-execution route preview:

- `POST http://127.0.0.1:5001/api/ghost_route_preview`

Execution only:

- `POST http://127.0.0.1:5001/api/responses`
- `POST http://127.0.0.1:5001/v1/responses`

Prefer these over raw provider or backend ports. Raw ports are implementation details unless the user explicitly asks for backend-level debugging.

## Network-Enabled Happy Path

If Codex can reach `http://127.0.0.1:5001`, use the control-plane endpoints directly:

1. Use `runtime_manifest` and `running_instances` for live capability, readiness, controls, and exact `instance_id` truth.
2. Use `ghost_route_preview` for read-only route/capability decisions.
3. Use `/api/responses` or `/v1/responses` only for execution.
4. If execution returns pending late-fill work, observe the existing `response_id`; do not start duplicate work.
5. Inspect runtime truth after execution before reporting success.

For long-running response work, poll compact status first:

    GET /api/responses/<response_id>?view=status

Fetch full `GET /api/responses/<response_id>` when the compact state changes or when artifact/detail content is needed. Compact status is only for observation; full response and artifact views remain the source for copyable artifact content.

Do not provide Terminal-safe `curl` commands as the primary path when direct HTTP works from Codex. Those commands are fallback guidance for sandbox denial or user preference.

If `CODEX_SANDBOX_NETWORK_DISABLED=1` or `Operation not permitted` appears, treat it as a Codex sandbox boundary and follow the failure path below.

## Read-Only Routing And Capability Decisions

Use this path for questions like "which capability should handle this", "ask Ghost where this would route", or "what currently running model should handle this." These are not execution requests.

Endpoint:

    POST /api/ghost_route_preview

Payload:

    {
      "prompt": "<user request>"
    }

Ghost route preview is non-execution. It must not start, stop, unload, materialize, write artifacts, freeze frames, or call `/api/responses`. It may perform computed semantic preview according to the non-UI `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS` policy: omitted or invalid defaults to `off`, `auto` currently behaves like `on`, explicit `compute_semantics=true` opts into compute, and explicit `compute_semantics=false` is the standard passive escape hatch unless `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE=deny`. Computed preview is computed preview truth, not pure observer truth, and preview evidence must not automatically feed self-learning.

If route preview is unavailable, answer from `runtime_manifest`, `running_instances`, `GET /api/ghost`, and local runtime files. Do not call `/api/responses` just to answer a route/capability question. Do not use `ollmoctl --recover-control-plane` as a fallback unless the user explicitly allows recovery behavior.

Ghost anchors current-turn intent and routing. Runtime graph truth, promotion review, artifacts, and closure decide what becomes executable work only when execution is requested.

## Ghost-First Execution Requests

Use Ghost-first execution only when the user asks Ollmo to actually run/generate the request.

Endpoint:

    POST /api/responses

Payload:

    {
      "ghost_route": true,
      "prompt": "<user request>",
      "conversation_id": "<stable conversation/session id>"
    }

This may create response frames, artifacts, late-fill state, logs, and other runtime state. Do not use it for read-only routing.

Use `prompt` for current `/api/responses` and route-preview requests because it matches the Ollmo UI transport and current control-plane behavior.

Before Ghost-first execution, read:

    GET /api/ghost_preferences

If it returns a non-empty `preferences` object, include that exact object as `ghost_preferences`. This mirrors the UI's preferred Ghost instance selection.

For multi-phase Ghost work, a successful first response may return `lifecycle_state: late_fill_pending` or `late_fill_running` with pending artifact branches. In that case, poll compact status first:

    GET /api/responses/<response_id>?view=status

Fetch full response truth when compact state changes or details are needed. Do not start a duplicate `/api/responses` request unless the user explicitly wants a new run.

## Direct Instance Requests

Use direct instance targeting when the user asks for a specific running local model or gives an exact local model target and wants execution.

Endpoint:

    POST /api/responses

Payload:

    {
      "instance_id": "<instance_id from runtime_manifest>",
      "prompt": "<user request>"
    }

Always resolve the exact `instance_id` from `runtime_manifest`, `running_instances`, or `model_ports.json` first. Do not guess IDs from display names.

## Capability Requests

Use capability routing when the user asks for a local capability and wants execution, but does not care which running model handles it.

Endpoint:

    POST /api/responses

Payload:

    {
      "capability": "chat",
      "prompt": "<user request>"
    }

Common capabilities:

- `chat`
- `image_generation`
- `vision_analysis`
- `speech_to_text`
- `text_to_speech`

Capability routing lets Ollmo choose a compatible live target from runtime truth.

## Runnable Catalog

Use running truth for active execution and the runnable catalog for startable options:

- active runtime: `GET /api/runtime_manifest`, `GET /api/running_instances`, `model_ports.json`, `state/runtime_status.json`
- startable catalog: `./ollmo ctl models list --json --runnable-only`

The runnable catalog may include models that are available but not running. Listing them is read-only. Starting, stopping, unloading, removing, or syncing models is a lifecycle action and requires explicit user wording.

## Ghost Prompt Shaping

For Ghost-first execution, make the `prompt` precise. Include:

- exact outcome owed
- expected capability, if known
- input paths or artifact refs
- output format
- controls and constraints such as language, voice/style, image dimensions, preservation rules, or no-inference rules
- closure criteria and blocked/failure behavior

Example:

    {
      "ghost_route": true,
      "prompt": "Use local speech_to_text on /path/audio.wav. Transcribe in German, preserve audible punctuation, do not translate. Return JSON with transcript, uncertain_spans, language_detected, and artifact refs. Treat missing or unreadable audio as blocked, not completed."
    }

For artifact creation, specify the exact count and completion condition. Example: `Imagine three visually distinct places to visit and generate exactly three image artifacts, one for each place. Return each place description, image artifact path, and artifact ref. Treat missing image artifacts as incomplete.`

For linked artifacts, specify materialization and binding closure. Example: `Create index.html and styles.css plus exactly three generated image artifacts. The first draft may use placeholders, but before final closure all image and CSS links must be rebound to concrete saved local artifacts using correct relative links from the HTML/CSS files. Chat-only code blocks or placeholder links do not count as final materialized files.`

## Artifact Fulfillment

Ollmo distinguishes evidence from fulfilled artifacts:

- Model prose, Markdown code blocks, prompt lists, or chat text are evidence, not fulfilled file artifacts.
- A requested file artifact is fulfilled only when runtime materializes it and records it in `outputs`, `artifacts`, response-frame truth, or artifact registry truth.
- A prompt for an image is not the image artifact. A script for audio is not the audio artifact. HTML/CSS prose is not a saved page unless materialized as files.
- `state/artifact_registry.jsonl` is a durable materialized artifact index for concrete artifact identity, provenance, metadata, enrichments, and lookup. It is not a strict append-only event log. Final saved artifact files are canonical bytes, while response frames and late-fill closure truth own owed/fulfilled/blocked/waived/superseded/repair-needed state.
- Public response projection must come from canonical runtime truth. Top-level `outputs` and `response_frame.output.outputs` are the public output contract; UI, chat-history, SSE, `view=ui`, and compact status payloads are projections that should converge from response-frame/artifact truth. Legacy `output`, `output_text`, `saved_image_path`, planner/status handoff prose, and raw history snapshots are compatibility or provisional surfaces, not proof of public artifacts when fulfilled artifact outputs exist.
- `late_fill.fill_results` is branch-local materialization truth. Final frame construction and hydration may recover missing saved image/audio/text artifacts from exact branch/path late-fill evidence; branch text-artifact sources outrank stale top-level saved paths and internal repair sources for the same saved file.
- Linked artifact sets close only when concrete saved files point to their concrete saved dependencies. Every surviving generated output path must resolve to a saved local artifact before final closure.
- Public web artifacts and bundles include the local dependency closure of the final public entrypoint, such as saved local images, CSS, scripts, audio, or other bundleable files referenced by final HTML/CSS/JS. Raw dossiers, repair-needed records, generated-image text misbindings, stale duplicate records, and internal repair artifacts stay diagnostic unless they are required dependencies of an included final entrypoint.
- Placeholder names, guessed paths, stale paths, wrong relative links, or duplicate path reuse before all owed saved artifacts are bound are not final closure.
- If the user requested an exact artifact count, treat a smaller count as incomplete unless runtime truth records an explicit waiver or supersession.

Required text-artifact branches may be fulfilled from existing canonical saved-file evidence before late-fill execution, preventing duplicate `index.html`/`styles.css` siblings when the owed file is already present and clean. If saved-truth validation fails, same-branch auto-retry is allowed only under bounded retry gates and blockers; dependency failures require dependency repair. Typed HTML/CSS syntax sanity may create target-bound repair evidence or deterministic repair only for mechanically safe defects.

## Multimodal Branch And Source Truth

- Follow the current request graph's producer-before-consumer dependencies and branch-local payloads. Same-turn source generation can ground a later dependent transform; a bare "make this/from that" request still requires selected, current, or explicit source evidence.
- Treat prior, selected, carried, and preserved artifacts as typed current-turn references with exact lineage/source qualification. They are not fresh uploads or newly generated outputs, and an unrelated media/output action must not inherit selected text-source write authority.
- Bind each vision/evidence consumer to its exact artifact. Do not let one branch inspect sibling or response-global media. Explicit structured joins must validate parseability, count, labels, required fields, distinct producer identity, and one-to-one refs; invalid model output remains inspectable while Closure stays pending with `repair_branch_contract`.
- For counted TTS work, explicitly labelled contiguous variants are candidate authority and each index supplies exactly one speakable field. Successful TTS records the exact final backend prompt and digest as `tts_semantic_source`.
- A directly dependent STT branch compares its actual transcript only with the bound producer source under named deterministic fidelity and records `tts_stt_semantic_evidence`. Missing, drifted, ambiguous, or mismatching evidence blocks with `DEPENDENCY_CHAIN_REPAIR_REQUIRED` / `repair_dependency_chain`; expected text is verification truth and must not be inserted into the STT request.
- Preserve/no-regenerate/no-reanalyze intent carries exact predecessor artifact/message evidence without creating replacement producer work. Missing or conflicting preserved evidence fails closed and must not leave stale generated-output or analysis projections.

## Local Workload Offload

Ollmo can offload bounded local/private work: image generation, OCR/vision, speech-to-text, translation/transcription, text-to-speech, embeddings, alternate local LLM passes, and artifact comparisons. Codex should shape the task and integrate the result; Ollmo/Ghost owns local runtime and artifact/output truth for Ollmo work.

For read-only capability selection, do not send this payload to `/api/responses`; inspect manifest/preview truth and report the selected capability.

## Response-Frame And Runtime Diagnostics

Treat response truth as substrate state, not model wording. Inspect:

- top-level `outputs`
- `response_frame`
- `lifecycle_state`
- `artifacts`
- `status_lookup` or `status_semantics` when present
- `runtime.graph_closure_review`
- `surface_state`
- `late_fill` state when present
- `runtime.request_phase_graph.graph_repair_proposals`
- `runtime.request_phase_graph.graph_repair_reviews`
- `runtime.developer_diagnostics.surface_repair_actionability`
- `runtime.developer_diagnostics.runtime_graph_repair_proposals`
- `runtime.developer_diagnostics.runtime_graph_repair_proposal_reviews`
- `graph_patch_lifecycle`, `staged_graph_patches`, or `applied_graph_patches` when present
- `accepted_learning` or accepted-learning runtime hints when present

`outputs` is the public contract. `response_frame.output.outputs` mirrors canonical public output truth for replay/recovery. `lifecycle_state` is the canonical continuation/terminal state when it differs from compatibility `status`. `runtime.graph_closure_review` explains whether graph obligations are fulfilled, pending, blocked, failed, waived, superseded, or repair-pending. `surface_state` is a UI/diagnostic projection of that runtime truth.

When public surfaces disagree, backend response-frame/output truth beats UI, chat-history, SSE, compact status, stale in-memory lookup state, and raw compatibility text fields. Compact `?view=status` is for observation counts and lifecycle state, not copyable artifact content.

Never claim a file, image, audio, or other artifact exists unless `outputs`, `artifacts`, response-frame truth, or artifact registry truth shows it.

Do not treat `status=completed` alone as proof of final success. Inspect `lifecycle_state`, `status_semantics`, `late_fill`, and `surface_state` for open continuation, repair, block, waiver, or supersession truth.

For terminal public text, prefer the promoted graph-terminal chat/join over earlier preparation or evidence text. Each terminal artifact has one exact branch/phase/path owner; generic same-type fallback is only for unbound legacy evidence. Active Late Fill outranks stale completed/actionable-repair projections, while durable hard-terminal truth outranks legacy active residue.

When artifacts exist, report concrete local paths and artifact refs. In Codex Desktop, image artifacts may be displayed with Markdown image syntax using an absolute filesystem path.

## Response Ledger Safety

`state/response_frames/responses.jsonl` is durable recovery truth. `state/response_frames/current_index.json` is derived acceleration: a globally fresh coverage-verified map may serve validated historical hits and prove an absent id, while legacy, stale, incomplete, malformed, or corrupt coverage must fall back to the ledger.

Responses API tests must set `ollmo_webserver.RESPONSE_FRAMES_DIR` to a per-test temporary root and must never scan or write the checkout's production ledger. Use `.venv/bin/python scripts/attest_response_frame_index.py --check-only` only for explicit operator preflight. Attestation must stream and exact-match the complete map before atomic replacement; never attest production state implicitly from tests or ordinary observation.

## Graph Repair And Self-Healing Diagnostics

Current response frames should carry graph-repair truth directly. The run monitor may summarize this state, but it is observer-only and is not the product authority when backend response truth is available.

Graph repair follows this authority chain:

1. Ghost, Closure, decision contracts, runtime evidence, or accepted learning may orient or propose.
2. Runtime validates a proposal with current evidence.
3. Runtime stages or applies only allowed additive patches.
4. Late Fill sees newly owed work only after runtime graph truth changes.
5. Closure judges the repaired graph from runtime truth.
6. Self-learning records outcomes without becoming proof by itself.

`OLLMO_GRAPH_REPAIR_AUTONOMY` controls rollout:

- `off`: diagnostics only.
- `shadow`: validated lifecycle records, no mutation.
- `stage`: staged patch records, no executable mutation.
- `apply_safe`: applies only validated safe additive patch classes.
- `apply_reviewed`: requires `graph_patch_authorization` on the concrete proposal review with accepted status, runtime/operator authority, `allowed_autonomy` containing `apply_reviewed`, and evidence refs.
- `apply_enforced`: default-deny and separately gated by `OLLMO_APPLY_ENFORCED_POLICY`; current safe-v1 policy only allows narrow validated runtime-policy classes such as safe additive missing branches, dependency repair, artifact binding repair, and proven duplicate artifact alias canonicalization.

Invalid autonomy values normalize to safe `off` and should appear in `runtime.developer_diagnostics.graph_patch_autonomy` with the raw value and invalid marker. `shadow` and `stage` are non-executable. Terminal/frozen response frames must not be silently rewritten. A newly applied safe additive terminal repair records exact `successor_reopen_requests[]` lineage on the blocked parent, then the production terminal owner revalidates autonomy, enforced policy for `apply_enforced`, parent/patch/graph bindings, exact owed scope, successor depth, and a no-root-replay contract. It appends one `graph_patch_reopen_successor` frame under the same response id and schedules only those branches through normal Late Fill. Request preparation and materialization-spec construction fail closed instead of recovering missing branch work from the frozen root request or assistant output. Replayed parent delivery reuses durable execution truth, while the complete Late Fill envelope and queued/running/terminal execution projections stay monotone: stale callbacks cannot restore active work, regress canonical lifecycle, or replace the first terminal result. Graph-repair autonomy `off` is the global stop; enforced-policy `off` stops `apply_enforced` but not explicit `apply_safe`. This graph-patch contract consumes only additive-rung `successor_reopen_requests[]`; the separate trusted partial-rebase path below is the only consumer of an exact upper-rung `successor_rebase_requests[]` record.

`OLLMO_GRAPH_REBASE_AUTONOMY` is the dedicated authority boundary for the successor-only partial-subtree and full-successor rungs at the upper end of the same redraw scope ladder. It is not a separate layer: lower bounded additive redraw remains autonomously active under graph-repair `safe_v1`, while the stronger validator and knob prevent that authority from silently climbing the ladder. Those upper rungs default to non-executable `shadow`, which is a mode rather than another rung; when the variable is absent, normal startup does not synthesize/export it, preserving product-default provenance. Explicit `off` blocks every rebase transition. Rebase/redraw can be proposed only from a current backend-built candidate plus actionable Closure/scope truth after active Late Fill settles and smaller scopes are exhausted. Terminal materialization re-derives a final candidate when needed. Ghost feedback items cannot supply rebase authority. Runtime requires a meaningful diff plus dependency, graph-wide semantic, derived-topology, hidden-failure, bookkeeping, and partial-scope preservation.

Read canonical rollout evidence from `GET /api/graph_rebase/readiness`; the observer never mutates or grants authority, and product default remains `shadow` while evidence gates are not green. Advance one exact response through `POST /api/responses/<response_id>/graph_rebase/operator` in order: `adjudicate`, durable audit-only `stage`, then gate-controlled `authorize_partial`. Actions require exact finalized response/frame/proposal/base/candidate/class CAS identities, reason, and evidence refs. Authorization is trusted only when joined from the operator registry after a useful-proposal adjudication and matching trusted/runtime stage; inline authorization is ignored.

The operator writer additionally requires an explicit startup `OLLMO_GRAPH_REBASE_OPERATOR_TOKEN` of at least 32 characters and an exact `OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY`, the same token in a Bearer or `X-Ollmo-Graph-Rebase-Operator-Token` header, and that configured identity in `X-Ollmo-Graph-Rebase-Operator`. These control-plane credentials are never durable truth and must not enter any child-process environment. Runtime automatically produces replay confirmation by comparing a deterministic revalidation with the frozen review; input replay claims are ignored. Mutating actions require full durable state and bounded observation truth to bind the same latest frame, with current root truth recovered from the bounded projection. Readiness counts only exact trusted/runtime stage pairs. `false_negative` can bind a settled no-proposal candidate through `expected_proposal_id=no_formal_proposal` and exact frame/base/candidate CAS, but remains evidence-only. A later same-class, replay-verified useful-proposal record may append one exact `resolves_record_id` link; the original record remains visible and only unresolved false negatives block promotion.

The accepted partial path appends one `graph_rebase_partial_successor` frame under the same response id, leaves the frozen parent unchanged, and schedules only exact branch-local owed work. Deterministic replay and the apply sink require the same current root digest; every executable prompt carrier on both phase and downstream records is normalized and checked. Missing or drifted current root truth, missing local payload, or missing dependency bindings block; root request, root/current-phase prompt, and assistant output are forbidden fallback. Untrusted, staged-only, stale, widened, gate-blocked, and full `successor_rebase_requests[]` remain audit/lineage truth. Full successor rebase remains shadow/non-executable under safe partial v1.

`surface_repair_actionability` distinguishes actionable repair evidence from advisory movement. Advisory-only `controlled_attention_review`, `aspiration_review`, `commitment_review`, reconsideration, reserved/deferred candidates, cached liveness, provider-family warnings, and live advisory degraded state should stay visible but must not create graph repair work by themselves.

## When To Read GHOST.md

`GHOST.md` is Ollmo-specific runtime policy. Read it only when:

- debugging Ghost behavior
- asking why Ghost routed a request a certain way
- modifying Ollmo Ghost policy
- inspecting Ghost's current-turn intent/routing boundaries
- comparing runtime behavior against Ghost policy

Do not copy `GHOST.md` wholesale into ordinary Codex context. For design orientation, use `docs/GHOST_ROUTER.md`. For operator/client usage, use `OLLMO_FOR_AGENTS.md`.

## Common Failure Handling

If direct HTTP to `127.0.0.1:5001` fails, do not immediately conclude that Ollmo is down. Check local runtime files first.

If a normal Terminal can reach Ollmo but Codex receives `EPERM`, `Operation not permitted`, socket denial, or cannot connect to `127.0.0.1`, treat the failure as a Codex sandbox/network boundary. Do not diagnose Ollmo as down, do not start/stop/clean/recover anything, and do not retry through raw backend ports.

For execution requests blocked by Codex localhost access, stop after the first clear sandbox denial and provide a Terminal/API command for the user to run outside Codex:

    curl -sS http://127.0.0.1:5001/api/responses \
      -H 'Content-Type: application/json' \
      -d '{"ghost_route":true,"prompt":"<user request>"}'

For read-only routing or capability questions blocked by Codex localhost access, answer from local runtime files or already-provided Terminal truth and state that Ghost/control-plane HTTP could not be safely queried from Codex.

If all read-only checks agree that the control plane is unavailable, report that state. You may tell the user the usual start command is:

    <ollmo>/ollmo start

Do not start it unless the user asks.

If the control plane is up but no relevant instance is running, inspect `runtime_manifest`, `running_instances`, and `state/runtime_status.json`; then report which capability or instance is missing.

If an instance is degraded or failed, inspect process, port, backend, and runtime status before guessing. A live degraded instance is advisory, not hard failed truth. Prefer stopping/restarting the affected instance only when the user asks for operational action.

If `lifecycle_state` or `late_fill` shows pending or blocked work, preserve that truth. Do not turn a blocked obligation into a successful answer.

If a branch failed because a dependency artifact is missing, look for `repair_dependency_chain`; retrying the same branch usually repeats the graph error.

## Lifecycle Safety

"Use Ollmo", "ask Ghost", "what is running", "which capability", and "runtime truth" are read-only. They do not authorize lifecycle actions.

Only run a lifecycle command when the user explicitly asks for that exact action:

- Start only on explicit start.
- Stop or shutdown only on explicit stop/shutdown.
- Restart only on explicit restart.
- Clean, archive, reset registry, unload, remove, or config-sync only when named explicitly.

Do not convert a stop/start request into a clean/archive/reset workflow. Do not clean `model_ports.json`, runtime state, logs, artifacts, or Codex config as an implied repair step.

Treat cleanup and archive commands as operationally meaningful, not observation. `./ollmo clean` is a repo-local dev reset/archive cleanup tool, not a harmless safe clean. `./ollmo archive --full` snapshots protected Ghost state when present and should preserve active protected Ghost files in the checkout; it is not a Ghost-forget or self-learning reset shortcut. Use `./ollmo clean --forget-ghost` only for explicit Ghost preference/compiled-memory forgetting, and `python3 scripts/ollmoctl.py ghost --reset-learning-state --json` only for explicit Ghost learning reset.

## Registry Pruning Warning

If a read-only status path causes `model_ports.json` to shrink or become empty, treat it as an Ollmo source bug, not a skill permission to repair by cleaning or restarting. The intended source behavior is:

- read endpoints may report readiness, reachability, degraded state, or blocks
- read endpoints must not delete stable registry entries
- pruning belongs to explicit lifecycle/hygiene commands only

The likely source fix is to make `ollmo_core.lifecycle.list_running_instances()` non-pruning by default and require explicit `prune=True` for hygiene paths.

## Ollmoctl Recovery Warning

Read-like `ollmoctl` commands should not recover or start the control plane by default where the current repo enforces that boundary. Use explicit `--recover-control-plane` only when the user wants local control-plane recovery; `ollmoctl responses get` follows the same explicit recovery rule. For strict read-only Codex use, prefer local files and already-running cached/passive HTTP endpoints.

## Anti-Patterns

- Do not use stale `~/.codex/config.toml` `model_providers` as runtime truth.
- Do not auto-sync or mutate Codex config unless explicitly requested.
- Do not start, stop, restart, clean, archive, unload, remove, reset, or sync anything for a read-only status/routing question.
- Do not call `/api/responses` or `/v1/responses` for read-only route/capability decisions.
- Do not use `ollmoctl --recover-control-plane` as a read-only fallback unless the user explicitly asks for control-plane recovery.
- Do not write `model_ports.json`, runtime status, response frames, artifacts, logs, or provider projections during read-only use.
- Do not treat cached observer surfaces as live probes, or use `refresh=true` without explicit intent to refresh runtime status.
- Do not treat Ghost route-preview semantic compute as pure observer truth or as automatic self-learning input.
- Do not treat one failed localhost HTTP probe as proof that no Ollmo instances are running.
- Do not treat Codex-localhost sandbox denial as an Ollmo failure when Terminal/API evidence shows Ollmo works.
- Do not retry raw backend ports, lifecycle commands, recovery, cleanup, or config edits after `EPERM` or `Operation not permitted`.
- Do not hard-code raw backend ports when Ollmo control-plane endpoints can be used.
- Do not treat Ghost as a general swarm, debate, or orchestration layer.
- Do not copy `GHOST.md` into ordinary Codex context unless Ghost policy/debugging is the task.
- Do not let old history or artifacts become fresh intent without current-turn reference.
- Do not infer successful artifact creation from assistant prose; check `outputs`, `artifacts`, response frames, artifact registry, and runtime truth.
- Do not treat UI, chat-history, SSE, compact status, stale lookup state, or raw `output_text` as stronger than canonical outputs and response-frame truth.
- Do not expose provisional planner/status prose as public output when fulfilled artifact outputs exist.
- Do not treat non-empty audio or transcript output as TTS/STT semantic fulfillment without bound producer-source evidence.
- Do not let a vision/evidence consumer inspect sibling media or rewrite invalid structured-join output into success.
- Do not ignore `late_fill.fill_results` branch/path truth when final projection or hydration misses saved artifacts.
- Do not let repair dossiers, generated-image text misbindings, duplicate internal text artifacts, or stale bundle records become normal public deliverables.
- Do not retry required text artifacts by replaying the root prompt or by bypassing same-branch bounded retry gates.
- Do not treat accepted learning as executable authority, graph patch validation, closure truth, or proof of fulfillment.
- Do not treat run-monitor summaries as the only authority when backend response-frame truth is available.
- Do not treat advisory degraded liveness, cached/busy/provider-family warnings, or monitor-only evidence as a hard failure, broad provider ban, or graph patch proof.
- Do not apply `apply_reviewed` graph patches without explicit `graph_patch_authorization` on the concrete proposal review.
- Do not treat `shadow` or `stage` graph patch lifecycle records as executable graph mutation.
