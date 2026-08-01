---
name: ollmo
description: Use when Codex needs to work with Ollmo, Ghost, local Ollmo models, runtime_manifest, response frames, local model routing, direct instance calls, capability routing, Ollmo-assisted decisions, Ollmo runtime truth, read-only Ollmo status checks, ollmoctl recovery safety, Codex localhost sandbox boundaries, or explicit Ollmo lifecycle operations. Also use when already-available local Ollmo capabilities could usefully support a broader Codex task with private/local text work, image generation, vision or OCR, speech-to-text, text-to-speech, embeddings, multimodal work, or durable artifacts. This skill helps Codex consider and use Ollmo as an optional local execution surface while Codex remains the supervisor using the user's normal Codex account/API key.
---

# Ollmo

## Project Provenance and License

Copyright 2026 fl0ri0.

The Ollmo Skill is part of Ollmo. The Ollmo Skill was conceived, created,
built, and developed by [@fl0ri0](https://github.com/fl0ri0). It is distributed
with Ollmo under the Apache License 2.0; see the repository's `LICENSE` and
`NOTICE` files. It is an Ollmo integration for Codex and ChatGPT. No OpenAI
authorship or endorsement is claimed.

## Core Boundary

Keep Codex and Ollmo roles separate:

- Codex remains the user-facing supervisor, coding agent, reviewer, workspace editor, cross-tool coordinator, and final narrator in the current thread.
- Ollmo owns local runtime truth, Ghost routing, response frames, artifacts, and recovery guidance.
- Ghost may also own local route planning, phase graph formation, contract shaping, closure checks, and no-prose-only completion checks for Ollmo work.
- Accepted learning may provide reviewed soft hints to Ghost, decision contracts, and bounded guardrails, but it is not executable proof and cannot satisfy or mutate current-turn obligations by itself.
- Runtime owns graph repair truth. Ghost or learning may orient proposals, but backend runtime validates, stages, applies, and records any graph patch.
- Do not turn Ollmo into Codex's default model provider.
- Do not mutate `~/.codex/config.toml` unless the user explicitly asks for Codex config work.
- Do not treat stale Codex `model_providers` blocks as runtime truth.

For normal Codex use, read `references/ollmo-contract.md` first. It is the concise operational contract for status checks, request shapes, response diagnostics, and failure handling.

## Read-Only First

Treat "use Ollmo", "ask Ghost", "which capability", "what is running", and "runtime truth" as read-only requests unless the user explicitly asks for a lifecycle change.

For read-only requests:

- Do not run `./ollmo start`, `./ollmo shutdown`, `./ollmo stop`, `./ollmo restart`, `./ollmo clean`, `./ollmo archive`, model unload/remove commands, or Codex config sync.
- Do not write `model_ports.json`, `state/runtime_status.json`, response frames, artifacts, logs, Codex config, or downstream provider projections.
- Do not call `/api/responses` or `/v1/responses` just to decide a route or capability; those are execution endpoints.
- Do not infer that Ollmo is down from one failed `curl` to `127.0.0.1:5001`.
- Read-like `ollmoctl` commands should not recover or start the control plane by default. Use the explicit `--recover-control-plane` flag only when the user wants local control-plane recovery; `ollmoctl responses get` follows the same explicit recovery rule.
- If direct HTTP fails with `EPERM`, `Operation not permitted`, `connection refused`, or `couldn't connect`, fall back to local runtime files before answering. Use `ollmoctl` only when it is a verified no-recovery read mode or when the user explicitly permits `--recover-control-plane`.
- If runtime files show ready/running instances but HTTP is blocked, answer from file/CLI truth and state that direct HTTP was unavailable from the current Codex tool process.

Lifecycle operations require explicit user wording for the exact action. If the user asks to start, start only. If the user asks to stop, stop only. Do not add clean/archive/reset/config-sync as an implied step.

## Codex Localhost Sandbox Boundary

If a normal Terminal can reach Ollmo but Codex gets `EPERM`, `Operation not permitted`, socket denial, or cannot connect to `127.0.0.1`, treat that as a Codex sandbox/network boundary.

For that case:

- Do not diagnose Ollmo as down or broken when external Terminal/API evidence shows it works.
- Do not retry through raw backend ports, `ollmoctl --recover-control-plane`, lifecycle commands, config sync, registry edits, or cleanup.
- For read-only questions, answer from local runtime files or already-collected terminal truth and say Ghost/control-plane HTTP was unavailable from Codex.
- For execution requests, stop after the first clear sandbox denial and provide a Terminal-safe Ollmo API command for the user to run outside Codex.
- If an Ollmo response frame exists, report its `lifecycle_state`, `outputs`, and `artifacts`; do not reinterpret a Codex socket denial as an Ollmo model failure.

## Resolve The Checkout

Before using Ollmo, resolve the active checkout in this order:

1. `$OLLMO_HOME`, if it points to an Ollmo checkout.
2. The current workspace root, if it contains `ollmo`, `ollmo_webserver.py`,
   and `ollmo_core/`.
3. Ask the user for the checkout path.

Do not search arbitrary parent directories for a checkout. Treat archived,
saved, or older checkouts as read-only references unless the user explicitly
selects one as active.

Never assume Ollmo is running or stopped. Check status before executing requests. Prefer cached control-plane observer surfaces and local runtime files over raw backend ports; opt into refresh truth only with explicit `refresh=true`.

Useful truth sources:

- `<ollmo>/model_ports.json`
- `<ollmo>/state/runtime_status.json`
- `GET http://127.0.0.1:5001/api/runtime_manifest`
- `GET http://127.0.0.1:5001/api/running_instances`
- `GET http://127.0.0.1:5001/api/backend_fabric`
- `GET http://127.0.0.1:5001/api/ghost`
- `GET http://127.0.0.1:5001/api/runtime_status`
- `GET http://127.0.0.1:5001/api/ghost_preferences`
- `POST http://127.0.0.1:5001/api/ghost_route_preview` for pre-execution route preview only, if available
- `GET http://127.0.0.1:5001/api/responses/<response_id>?view=status` for compact observation of existing response work

`/api/running_instances`, `/api/runtime_manifest`, `/api/backend_fabric`, `/api/ghost`, and `/api/runtime_status` are cached/passive observer surfaces by default. They may read `model_ports.json`, existing `state/runtime_status.json`, in-memory request state, frames, registries, history, logs, and artifacts, but they must not probe backend ports, fetch backend runtime metadata, recover the control plane, start/stop/unload models, execute response work, or write status files unless `refresh=true` is explicit. `refresh=true` may probe process/port/backend facts and update `state/runtime_status.json`, but it is still not execution and must not materialize branches, write artifacts, or freeze response frames.

Live process, listening port, backend runtime, and control-plane truth beat stale readiness projections. A live `degraded`, `busy`, timeout, cooldown, cache, or provider-family warning is advisory route-health evidence unless hard liveness proves the instance unusable. Do not turn advisory degraded state into `failed`, offline, a broad provider ban, or graph repair proof by itself.

## Operational Happy Path

When Codex can reach `http://127.0.0.1:5001`, use the Ollmo control plane directly:

1. Read `runtime_manifest`, `running_instances`, `backend_fabric`, `ghost`, and `runtime_status` as cached/passive observer surfaces for current capability, readiness, controls, and exact `instance_id` truth.
2. Use `refresh=true` only when explicitly opting into runtime probing/status refresh; refresh is still non-execution.
3. Use `ghost_route_preview` for pre-execution route/capability decisions, with the preview truth boundary described below.
4. Use `/api/responses` only for actual execution.
5. If execution returns `late_fill_pending` or `late_fill_running`, observe the same `response_id` instead of starting duplicate work.
6. After execution, inspect `outputs`, `artifacts`, `response_frame`, `lifecycle_state`, `runtime.graph_closure_review`, `surface_state`, and `late_fill` before reporting completion.

For long-running responses, poll compact status first:

- Use `GET /api/responses/<response_id>?view=status` to observe lifecycle and late-fill state.
- Fetch full `GET /api/responses/<response_id>` when the compact state changes or when artifact/detail content is needed.
- Do not repeatedly ask `/api/responses` with the full prompt/history just to check whether existing work is done.
- Compact status is only an observer surface; full response/artifact truth remains the source for copyable artifact content.

Do not fall back to Terminal-safe commands when direct control-plane HTTP works. Terminal commands are only for Codex localhost sandbox denial or user preference.

If all read-only checks agree that the control plane is unavailable, say that plainly. You may tell the user the usual start command is:

    <ollmo>/ollmo start

Do not start Ollmo unless the user asks. Do not stop, clean, archive, or reset anything as part of a read-only diagnosis.

## Choose The Request Path

Consider Ollmo whenever it is explicitly useful for the task:

- the user asks for Ollmo or Ghost
- the user asks about local Ollmo models, routing, runtime truth, response frames, or artifacts
- the user wants an Ollmo-assisted decision
- the task benefits from Ollmo's response-frame or local runtime truth
- a broader Codex task contains a bounded text, image, vision/OCR, speech, embedding, or multimodal subtask that an already-running local Ollmo capability can handle well
- local execution materially helps privacy, locality, context/token use, repeatability, or durable artifact evidence without compromising the requested result

Treat Ollmo as an available tool surface, not a mandatory detour. Do not use it merely because a local capability exists. Respect an explicitly requested provider or tool, prefer deterministic workspace tools for deterministic work, and keep Codex responsible for the overall plan, judgment, integration, validation, and final answer.

Passive capability inspection does not authorize execution or lifecycle changes. Prefer already-running compatible instances. If a useful capability is installed but not running, tell the user what is available; do not start it unless the user explicitly asks. For work that must remain local, target confirmed local runtime truth and do not route through an optional cloud provider.

First decide whether the user wants a read-only route/capability answer or actual execution.

Read-only route/capability question examples:

- "Which currently running capability should handle this?"
- "Ask Ghost where this would route."
- "What local model should I use?"
- "Show the current runtime truth."

For those, use cached status/manifest/Ghost read surfaces and, if available, `POST /api/ghost_route_preview`. Do not use `/api/responses`, because it may execute work and write runtime/response state.

If the HTTP read surfaces are unavailable from Codex and local files are the only safe truth, report that Ghost could not be safely queried and give the best file-based capability answer. Do not use `ollmoctl --recover-control-plane` as a workaround unless the user explicitly allows it.

Route preview request:

    {
      "prompt": "<user request>"
    }

Ghost route preview is not execution. It must not start, stop, unload, materialize, write artifacts, freeze response frames, or call `/api/responses`. It may perform computed semantic preview according to non-UI policy: `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS` accepts `off`, `on`, or `auto`; unset or invalid defaults to `off`, and `auto` currently behaves like `on`. Explicit `compute_semantics=true` opts into compute even when the default policy is `off`. Explicit `compute_semantics=false` is the standard pure cached/passive preview escape hatch because `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE` defaults to `allow`; operators can set that override to `deny` when an active `on` or `auto` policy must remain in force despite explicit false. Automatic UI route preview sends `compute_semantics=false` to stay passive. Computed semantic preview may use embedding or semantic helper backends for route-quality signals; treat it as computed preview truth, not pure observer truth. Preview evidence is preview-only and self-learning must not automatically learn from it.

Execution request examples:

- "Generate this with Ollmo."
- "Run this on instance X."
- "Use the local TTS model to create audio."
- "Call `/api/responses`."

Only for execution, use `POST http://127.0.0.1:5001/api/responses` as the canonical local execution endpoint. OpenAI-style clients may use `POST http://127.0.0.1:5001/v1/responses`.

If Codex is blocked from localhost execution but the user's normal Terminal can reach Ollmo, do not keep retrying. Give the user the same payload as a `curl` command or tell them to use Ollmo's UI/API outside Codex.

Ghost-first request:

    {
      "ghost_route": true,
      "prompt": "<user request>",
      "conversation_id": "<stable conversation/session id>"
    }

Use Ghost-first execution only when the user wants Ollmo to actually run the request through Ghost. For read-only Ghost routing or Ollmo-assisted decisions, use route preview/status surfaces instead.

Use `prompt` for `/api/responses` and route-preview payloads because it matches the Ollmo UI transport and current control-plane behavior.

Before Ghost-first execution, read `GET /api/ghost_preferences` when available and pass the returned `preferences` object as `ghost_preferences`. This mirrors the UI's preferred Ghost instance behavior and prevents avoidable differences between Codex-triggered and UI-triggered Ghost plans. If the preferences response is empty, omit `ghost_preferences`.

Direct instance request:

    {
      "instance_id": "<instance_id from runtime_manifest>",
      "prompt": "<user request>"
    }

Use direct targeting when the user names or asks for a specific running local model. Resolve the exact `instance_id` from runtime truth first.

Capability request:

    {
      "capability": "chat",
      "prompt": "<user request>"
    }

Use capability routing when the user asks for a local capability without caring which model handles it. Common capabilities: `chat`, `image_generation`, `vision_analysis`, `speech_to_text`, and `text_to_speech`.

## Runnable Catalog Awareness

Use the running runtime for current execution truth and the runnable catalog for startable-but-not-running options:

- Running truth: `GET /api/runtime_manifest`, `GET /api/running_instances`, `model_ports.json`, `state/runtime_status.json`.
- Runnable catalog: `./ollmo ctl models list --json --runnable-only`.

The runnable catalog can include local models that are available but not active, such as alternate chat/vision models, OCR specialists, TTS voices, or image generators. Listing these models is read-only. Starting, stopping, unloading, or removing any model remains a lifecycle operation and requires explicit user wording.

## Local Workload Offload

During task planning and before suitable tool calls, briefly consider whether an already-running Ollmo capability is the better execution surface. Use Ollmo to offload bounded local/private work when helpful:

- image generation
- vision analysis, OCR, and document/image inspection
- speech-to-text and translation/transcription workflows
- text-to-speech and voice generation
- embeddings
- alternate local LLM passes for drafting, checking, comparison, or local-only reasoning
- artifact comparisons or multimodal pipelines where response-frame truth matters

Codex remains responsible for shaping the request, integrating the result into the user-facing task, and validating workspace changes. Ollmo/Ghost owns the local runtime graph and artifact/output truth for Ollmo work.

If Codex delegates a suitable bounded subtask to a subagent, write `Use $ollmo` and name the intended capability, inputs or artifact refs, expected output, and closure criteria in the handoff. If the parent already inspected current runtime truth and resolved the route, pass the exact `instance_id` for a direct `/api/responses` request, or pass `capability` when model identity does not matter. Do not invoke Ghost merely to repeat a route Codex has already resolved. Use Ghost when route choice, phase formation, dependencies, multimodal graph shaping, or closure planning still benefits from Ollmo's runtime intelligence. Never hand off a stale provider command or raw backend port. Do not assume that an unnamed skill or the parent's already-loaded skill context will automatically guide every subagent.

## Ghost Prompt Shaping

Use Ghost when the user wants Ollmo to decide and execute a local route, form a phase graph, or run a multi-step/multimodal job through Ollmo's runtime truth layer. Ghost is not just a text model call; it may create a preparation phase, promote artifact-producing branches, and return `late_fill_pending` or `late_fill_running` while downstream branches continue.

Ghost execution happy path:

1. Check `runtime_manifest`/`running_instances` for ready capabilities.
2. Read `GET /api/ghost_preferences`; if it returns a non-empty `preferences` object, include it as `ghost_preferences`.
3. Send `POST /api/responses` with `ghost_route: true`, `prompt`, a stable `conversation_id`, and `ghost_preferences` when executing.
4. Include `ghost_messages`, `ghost_preview`, or `request_meta` only when mirroring UI state or preserving an active conversation contract; do not add old history just because it exists.
5. If the response returns `late_fill_pending` or `late_fill_running`, poll the existing response id instead of starting a duplicate request. Prefer compact observer polling with `GET /api/responses/<response_id>?view=status`, then fetch full response truth when state changes or details are needed.
6. Report completion only from `outputs`, `artifacts`, `response_frame`, `lifecycle_state`, `runtime.graph_closure_review`, `surface_state`, and `late_fill`.

Precise Ghost prompts produce better local plans and cleaner closure. Include:

- exact outcome owed
- expected capability when known
- input file paths or artifact refs
- output format
- constraints such as language, dimensions, voice/style, no-inference rules, or preservation requirements
- closure criteria and what should count as blocked or failed

Example: `Use local speech_to_text on /path/audio.wav. Transcribe in German, preserve audible punctuation, do not translate. Return JSON with transcript, uncertain_spans, language_detected, and artifact refs. Treat missing or unreadable audio as blocked, not completed.`

For artifact creation, make the artifact count and closure criteria explicit. Example: `Imagine three visually distinct places to visit and generate exactly three image artifacts, one for each place. Return each place description, image artifact path, and artifact ref. Treat missing image artifacts as incomplete.`

For linked artifacts, make closure concrete. Example: `Create index.html and styles.css plus exactly three generated image artifacts. The first draft may use placeholders, but before final closure all image and CSS links must be rebound to the concrete saved local artifacts using correct relative links from the HTML/CSS files. Chat-only code blocks or placeholder links do not count as final materialized files.`

For page/site prompts that ask for local generated images, embedded local images, local image assets, or non-external image paths, treat those image assets as graph adequacy obligations before HTML/CSS consumers run. Structural hints such as image sections set the minimum count. HTML/CSS artifact branches should depend on generated image phases and carry local visual asset binding instead of closing against guessed paths.

## Artifact Fulfillment

Ollmo distinguishes evidence from fulfilled artifacts:

- Model prose, Markdown code blocks, prompt lists, or chat text can be useful evidence, but they do not fulfill a requested file artifact unless runtime materializes the file and records it in `outputs`, `artifacts`, response-frame truth, or artifact registry truth.
- A prompt for an image is not the image artifact. A script for audio is not the audio artifact. HTML/CSS prose is not a saved page unless materialized as files.
- `state/artifact_registry.jsonl` is a durable materialized artifact index for concrete artifacts, identity, provenance, metadata, enrichments, and lookup. It is not a strict append-only event log, and index merge/rewrite is not artifact-byte mutation.
- Final saved artifact files are canonical bytes. Response frames and late-fill closure truth own what the response owed, fulfilled, blocked, waived, superseded, or left repair-needed.
- Public response projection must come from canonical runtime truth. Top-level `outputs` and `response_frame.output.outputs` are the public output contract; UI, chat-history, SSE, `view=ui`, and compact status payloads are projections that should converge from response-frame/artifact truth. Legacy `output`, `output_text`, `saved_image_path`, root planner prose, generated-artifact handoff text, and generic status summaries are compatibility or provisional surfaces, not proof of public artifacts when fulfilled artifact outputs exist. Explicit user-visible non-artifact chat/text remains public only when runtime truth preserves it as a distinct output.
- `late_fill.fill_results` is branch-local materialization truth for deferred work. Final frame construction and hydration may recover missing saved image/audio/text artifacts from exact branch/path late-fill evidence, while preserving branch/phase identity. Branch text-artifact sources outrank stale top-level saved paths and internal repair sources for the same saved file; do not use generic same-type text fallback to swap explicit text refs.
- Linked artifact sets close only when concrete saved files point to the concrete saved dependencies. Every surviving generated output path must resolve to a saved local artifact before final closure.
- Placeholder names, guessed paths, stale paths, wrong relative links, or duplicate path reuse before all owed saved artifacts are bound are not final closure.
- Missing or wrong links should lead to deterministic rebind or bounded repair against existing artifacts, not a false completed answer.
- If the user requested an exact artifact count, treat a smaller count as incomplete unless runtime truth records an explicit waiver or supersession.

Closure-promoted text artifact repairs are bounded evidence packets, not generic retries. A repair branch should receive the target artifact identity/path, the concrete defect class, relevant runtime artifact evidence, and the current saved bytes needed to patch or replace only that target. Syntax repair, link rebind, and HTML/CSS selector-binding repair should fix the evidenced local defect; they should not redesign, translate, regenerate unrelated structure, or replay the root prompt. If broad root-prompt or `current_phase_output` context conflicts with the bounded closure repair payload, the bounded repair payload wins.

Required text-artifact branches may be fulfilled from existing canonical saved-file evidence before late-fill execution, preventing duplicate `index.html`/`styles.css` siblings when the owed file is already present and clean. If saved-truth validation fails, same-branch auto-retry is allowed only under the existing bounded retry gates and blockers; dependency failures require dependency repair, not root-prompt replay. Typed HTML/CSS syntax sanity may create target-bound repair evidence or deterministic repair only for mechanically safe defects such as malformed stylesheet link attributes, safe CSS grammar aliases, or unsupported navigation-anchor wrappers.

## Multimodal Branch Truth

- Execute multi-phase media work from current graph dependencies and branch-local payloads. An explicitly generated same-turn source may feed its dependent transform; a bare deictic transform still needs selected, current, or explicit source evidence. Prior, selected, carried, or preserved artifacts are typed reference evidence, not newly generated outputs or fresh `input_artifacts`.
- Bind evidence to the exact producer and consumer. Vision analysis owns only its attached artifact, not sibling or response-global media. Explicit machine-checkable joins must preserve exact count, labels, required fields, distinct producer identity, and one-to-one artifact refs; keep an invalid join inspectable and leave Closure pending with `repair_branch_contract` instead of rewriting model bytes into success.
- Explicitly labelled contiguous audio variants are the authority for counted TTS work, with one speakable field per index. A successful TTS fill records the exact final backend prompt and SHA-256 as `tts_semantic_source`; a direct STT consumer compares its actual transcript only with that bound source under named deterministic fidelity and records `tts_stt_semantic_evidence`.
- Missing, drifted, ambiguous, or mismatching TTS/STT evidence blocks the STT branch with `DEPENDENCY_CHAIN_REPAIR_REQUIRED` / `repair_dependency_chain`; never inject the expected source text into the STT request. A non-empty audio file or transcript alone is not semantic fulfillment.
- Explicit current-turn preserve/no-regenerate/no-reanalyze intent carries exact predecessor artifact/message evidence without creating a replacement producer. Missing or conflicting preserved evidence fails closed and must not leave stale generated-output, capability, or analysis obligations in canonical truth.

## Read Response Truth

Do not infer success from assistant prose. Inspect Ollmo runtime truth:

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
- `runtime.request_phase_graph.intent_obligations`
- `runtime.graph_closure_review.intent_graph_adequacy`
- `runtime.request_phase_graph.redraw_scope_ladder_review`
- `graph_patch_lifecycle`, `staged_graph_patches`, or `applied_graph_patches` when present
- `successor_reopen_requests` or `successor_rebase_requests` when present
- graph repair/rebase autonomy diagnostics when present
- `accepted_learning` or accepted-learning runtime hints when present

Treat `outputs` as the public contract. `output` and `output_text` may be compatibility projections. Local model calls execute selected phases or branches; Ollmo runtime decides whether obligations are fulfilled.

Treat `lifecycle_state` as canonical when it differs from compatibility `status`. `status=completed` can still carry open late-fill, repair, or closure truth; inspect `status_semantics`, `late_fill`, and `surface_state`.

For terminal projection, prefer the promoted graph-terminal chat/join over earlier preparation or evidence text. Bind each terminal artifact to one exact branch/phase/path owner; generic same-type fallback is only for unbound legacy evidence. Active Late Fill outranks stale completed/actionable-repair projections, while durable hard-terminal truth outranks legacy active residue.

When artifacts exist, surface the concrete local path or artifact ref only after runtime truth confirms it. In Codex Desktop, image artifacts can be shown with Markdown image syntax using an absolute filesystem path; audio and other files should be reported with absolute paths and artifact refs.

When artifact sources disagree, use final saved files for bytes, response frames and late-fill closure truth for obligations/closure, and `state/artifact_registry.jsonl` for lookup/provenance/metadata. If a saved file exists but registry metadata is stale, refresh projections from the file. If an owed saved file is missing, keep closure repair-needed or blocked instead of inventing completion.

When public surfaces disagree, backend response-frame/output truth beats UI, chat-history, SSE, compact status, and stale in-memory lookup projections. Repair the projection or hydrate from the latest frame; do not change runtime truth to match a reduced UI surface. Compact `?view=status` is for observation counts and lifecycle state, not copyable artifact content.

## Graph Repair And Self-Healing Truth

Current response frames should carry graph-repair truth directly. The run monitor may summarize this state, but it is observer-only and must not be treated as the only product authority for current frames.

Inspect backend response truth first:

- `runtime.request_phase_graph.graph_repair_proposals`
- `runtime.request_phase_graph.graph_repair_reviews`
- `runtime.developer_diagnostics.surface_repair_actionability`
- `runtime.developer_diagnostics.runtime_graph_repair_proposals`
- `runtime.developer_diagnostics.runtime_graph_repair_proposal_reviews`
- `runtime.request_phase_graph.intent_obligations`
- `runtime.graph_closure_review.intent_graph_adequacy`
- `runtime.request_phase_graph.redraw_scope_ladder_review`
- `graph_patch_lifecycle`
- `staged_graph_patches`
- `applied_graph_patches`
- `successor_reopen_requests`
- `successor_rebase_requests`

Graph repair follows this authority chain:

1. Ghost, Closure, decision contracts, runtime evidence, or accepted learning may orient or propose.
2. Runtime validates a proposal with current evidence.
3. Runtime stages or applies only allowed additive patches.
4. Late Fill sees newly owed work only after runtime graph truth changes.
5. Closure judges the repaired graph from runtime truth.
6. Self-learning records outcomes without becoming proof by itself.

`OLLMO_GRAPH_REPAIR_AUTONOMY` is the rollout knob:

- `off`: diagnostics only; no graph mutation.
- `shadow`: record validated lifecycle truth with no mutation.
- `stage`: record staged patches with no executable graph mutation.
- `apply_safe`: apply only validated safe additive classes such as missing materialization branches, missing dependency edges, semantic-review branches, artifact-binding repairs, text I/O repairs, and repairable block reopens.
- `apply_reviewed`: may apply review-required classes only when the concrete proposal review includes `graph_patch_authorization` with accepted status, runtime/operator authority, `allowed_autonomy` containing `apply_reviewed`, and evidence refs.
- `apply_enforced`: default-deny and separately gated by `OLLMO_APPLY_ENFORCED_POLICY`. Current `safe_v1` policy can apply only narrow runtime-policy classes such as safe additive missing branches, dependency repair, artifact binding repair, and proven duplicate artifact alias canonicalization after validation, safe-additive risk classification, current evidence, redraw-scope alignment, idempotency, lineage/audit recording, and forbidden-evidence checks pass.

Invalid autonomy values normalize to safe `off` and should appear in `runtime.developer_diagnostics.graph_patch_autonomy` with the raw value and invalid marker. `shadow` and `stage` are non-executable. Terminal/frozen response frames must not be silently rewritten. If newly applied safe additive repair reaches a terminal/frozen parent frame, keep the parent blocked and unmutated, record `successor_reopen_requests[]` with exact parent/patch/graph lineage, then let the production terminal owner independently revalidate current autonomy and enforced policy when autonomy is `apply_enforced` before it appends one `graph_patch_reopen_successor` frame under the same response id and schedules only the exact owed branches through normal Late Fill. The continuation must be idempotent, branch-local, bounded in depth, and must never replay the root prompt; request preparation and materialization-spec construction both fail closed instead of recovering prompt work from the frozen root or assistant output. Its complete Late Fill envelope and same-key execution projections are monotone, so stale pending/running callbacks cannot restore active work, regress canonical lifecycle, or replace the first terminal result. Explicit graph-repair autonomy `off` prevents every continuation. Enforced-policy `off` prevents `apply_enforced`, including the product-default path, but does not override explicit `apply_safe`. This graph-patch continuation consumes only additive-rung `successor_reopen_requests[]`; the separately authorized partial-rebase owner described below is the only consumer of an exact upper-rung `successor_rebase_requests[]` record.

`OLLMO_GRAPH_REBASE_AUTONOMY` is the dedicated authority boundary for the partial-subtree and full-successor rungs at the upper end of the same redraw scope ladder. Rebase/redraw is not a parallel layer; lower bounded additive redraw remains autonomously active under graph-repair `safe_v1`, while the stronger rebase validator and knob prevent that permission from silently climbing the ladder. Its absent-environment product default is non-executable `shadow`, which is a mode of those upper rungs rather than another rung; when the variable is absent, normal startup does not synthesize/export it, preserving product-default provenance. `off` is the immediate diagnostics-only rollback, blocks the operator path, and invalid values normalize to safe `off`. Runtime may propose only a concrete backend-built candidate after post-repair Closure/scope recomputation, settled Late Fill, and smaller-scope precedence; terminal materialization re-derives a final candidate when active Late Fill consumed the earlier bounded comparison. Ghost feedback items remain advisory and cannot supply rebase authority.

Use `GET /api/graph_rebase/readiness` for canonical read-only rollout evidence. The product default stays `shadow` while its evidence gates are not green. Promotion is explicit per response through `POST /api/responses/<response_id>/graph_rebase/operator` and must proceed `adjudicate -> stage -> authorize_partial`. Stage is durable audit-only. Every action is bound to the exact finalized response/frame/proposal/base/candidate/class identity; `authorize_partial` additionally requires the trusted registry chain, matching runtime stage, green partial gate, and branch-local execution-contract proof. Inline request/model/candidate authorization is not authority.

The mutating operator route is disabled unless `OLLMO_GRAPH_REBASE_OPERATOR_TOKEN` is explicitly present at startup and at least 32 characters long and `OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY` names one exact operator. Authenticate with that value as a Bearer token or `X-Ollmo-Graph-Rebase-Operator-Token`, and repeat the configured identity in `X-Ollmo-Graph-Rebase-Operator`; never place either credential in request bodies, frames, registry evidence, logs, chat output, or any child-process environment. Runtime, not the caller, produces `replay_verified` by deterministic revalidation of the frozen review. Mutating actions require full durable state and bounded observation truth to identify the same latest frame; the bounded projection supplies successor-safe current root truth. Only exact trusted/runtime stage pairs count. A response-bound `false_negative` without a proposal uses `expected_proposal_id=no_formal_proposal` plus exact frame/base/candidate bindings and remains non-executable append-only evidence. One later same-class, replay-verified `useful_proposal` adjudication may reference it with `resolves_record_id`; history retains both records and gates count only unresolved false negatives.

Only the exact authorized partial request may append a `graph_rebase_partial_successor` frame under the same response id and schedule its exact owed branches through normal Late Fill. Proposal replay and the final apply sink require the same current root digest. Primary payloads, batch prompts, phase summaries, stage directions, instructions, and criteria on both phase and downstream records are normalized and checked for root replay. The parent remains immutable, execution is idempotent and bounded, and missing/drifted root truth or missing local payload/dependency bindings block rather than falling back to the root request, root/current-phase prompt, or assistant output. All other `successor_rebase_requests[]` records remain audit/lineage truth. Full successor rebase remains shadow/non-executable under safe partial v1.

`redraw_scope_ladder_review` is orientation, not authority. It chooses the smallest current-intent-aligned scope: reserved slot/candidate fill, additive repair, binding/dependency repair, duplicate artifact-ref identity repair, partial subtree rebase, then full successor rebase. Accepted learning, advisory roles, degraded/provider/cache/liveness evidence, frontend state, monitor-only summaries, and UI labels remain orientation or diagnostics only.

Duplicate artifact refs are final-output hygiene. Proven aliases may collapse to one projection while preserving alias metadata. Conflicting refs must remain `repair_needed` and block successful final projection instead of being silently projected as success.

Accepted learning is enabled in the current checkout as reviewed soft orientation. It can carry concrete hints into Ghost payloads, route context, request phase graph construction, decision contracts, and controlled-attention frames. It cannot promote obligations, execute work, waive, supersede, satisfy review, validate graph patches, apply graph patches, freeze closure, override explicit targets, or override current runtime truth. Provider-family trouble, advisory degraded liveness, monitor-only summaries, and advisory movement surfaces are route-health or attention evidence unless current runtime/Closure truth makes them actionable.

`surface_repair_actionability` separates actionable repair evidence from advisory movement. Advisory-only `controlled_attention_review`, `aspiration_review`, `commitment_review`, reconsideration, reserved/deferred candidates, and cached liveness should stay visible but must not create `reconcile_surface_state_or_reopen_contract` by themselves. Actionable blocked, repair-pending, semantic-review-pending, dependency, artifact, materialization, or promoted owed-work evidence remains repairable.

## Response Artifact Bundles

Response artifact bundles are post-response export/openable surfaces for users, not original model-generated outputs and not response closure authority. Creating a bundle must not run models, change Ghost routing, alter late-fill state, mutate original artifacts, or change the original response-frame closure truth.

Bundle folders live under `artifacts/bundles/` and are derived from existing saved response artifacts. The bundle manifest is the bundle's own truth for copied files, rewritten relative links, entrypoint, checksums, and link-check status. The original saved artifact files remain canonical bytes for the response itself.

Bundle roots and frontend artifact lists should prefer fulfilled canonical output truth over raw artifact/dossier harvesting. Public web artifacts include the local dependency closure of the final public entrypoint, such as saved local images, CSS, scripts, audio, or other bundleable files referenced by the final HTML/CSS/JS. Raw dossiers, repair-needed records, generated-image text misbindings, stale duplicate registry rows, and internal repair artifacts remain diagnostic truth unless they are required dependencies of an included final entrypoint.

`state/artifact_registry.jsonl` may carry historical `response_artifact_bundle` records, and response/chat-history payloads may expose `artifact_bundles`, `artifactBundles`, or `artifactBundle`. Treat those fields as current Open Bundle truth only when they point to the latest existing/openable bundle directory. Deleted bundle folders, stale entrypoints, or duplicate historical registry records must not block creating a new bundle or appear as current bundle truth.

## Cleanup And Archive Boundaries

Treat cleanup and archive commands as operationally meaningful, not observation:

- `./ollmo clean` is a repo-local dev reset/archive cleanup tool, not a harmless safe clean. Do not run it from status, routing, or diagnosis work unless the user explicitly asks for cleanup/reset.
- Clean/archive should preserve standard artifact bucket directories, including `artifacts/bundles/`, while removing generated contents only when explicitly requested.
- `./ollmo archive --full` snapshots protected Ghost state when present and should preserve active protected Ghost files in the checkout. It is not a Ghost-forget or self-learning reset shortcut.
- Protected Ghost state includes `state/ghost_preferences.json`, `state/ghost_compiled_memory.json`, `state/ghost_compiled_memory.md`, `state/self_learning/`, `state/self_learning/accepted_policy_snapshot.json`, `state/self_learning/retention_manifest.json`, and retained learning sidecars under `state/self_learning/retained_sidecars/` when present.
- Before response-frame cleanup, clean/archive should collect sidecar refs reachable from active `state/self_learning/` JSON/JSONL, write `state/self_learning/retention_manifest.json`, and copy retained evidence into `state/self_learning/retained_sidecars/`. Missing retained sidecars are diagnostics; do not silently hydrate empty learning evidence.
- Use `./ollmo clean --forget-ghost` only for explicit Ghost preference/compiled-memory forgetting.
- Use `python3 scripts/ollmoctl.py ghost --reset-learning-state --json` only for explicit Ghost learning reset. Do not infer Ghost state reset from archive rotation.

## Testing Protocol

For Ghost self-learning, response-frame, artifact, late-fill, observer/read-surface, or larger runtime refactors, run the fake-backend E2E truth harness:

    .venv/bin/python -m pytest tests/test_fake_backend_e2e.py -q

The harness uses deterministic test-only fake backends and temp `artifacts/`, `state/response_frames/`, `state/artifact_registry.jsonl`, `state/chat_history/`, and `logs/` roots. Assertions are based on runtime truth fields and saved files, not model prose. Do not substitute live API calls or lifecycle commands for this harness when validating those areas.

The harness also covers the current learning/healing truth boundary: fake `/api/responses` payloads must expose `runtime.request_phase_graph.intent_obligations`, local producer-before-consumer dependency edges, and structural `intent_graph_adequacy`; accepted-learning hints may surface as soft decision-contract orientation but must not create executable graph repair proposals, graph patch lifecycle truth, staged patches, or applied patches by themselves.

For audio branch semantics, include `tests/test_response_semantics_runtime.py -k "tts or speech_to_text or audio_variant or semantic_evidence"` and focused Responses API Late Fill coverage. Responses API tests must redirect `RESPONSE_FRAMES_DIR` to a per-test temporary root and must never scan, attest, or write the production response ledger.

Treat `state/response_frames/current_index.json` as recovery acceleration, not durable substrate truth. A globally fresh coverage-verified index may serve validated hits and prove misses; stale, incomplete, malformed, or corrupt coverage falls back safely to the ledger. Use `scripts/attest_response_frame_index.py --check-only` only as an explicit operator preflight, never implicitly from tests or ordinary observation.

For accepted-learning and graph-repair changes, run:

    .venv/bin/python -m pytest tests/test_graph_repair_self_healing.py tests/test_self_learning.py -q

For graph rebase changes, include:

    .venv/bin/python -m pytest tests/test_graph_rebase_review.py tests/test_runtime_graph_rebase_shadow_producer.py tests/test_graph_rebase_readiness.py tests/test_graph_rebase_operator.py tests/test_graph_rebase_partial_successor.py -q

For enforced graph policy changes, run:

    .venv/bin/python -m pytest tests/test_apply_enforced_policy.py tests/test_graph_repair_self_healing.py tests/test_graph_rebase_review.py tests/test_runtime_graph_rebase_shadow_producer.py tests/test_self_learning.py -q

For intent-obligation graph adequacy and local producer-before-consumer dependency work, run:

    .venv/bin/python -m pytest tests/test_request_phase_graph_runtime.py tests/test_response_semantics_runtime.py tests/test_graph_repair_self_healing.py -q

For redraw-scope and duplicate artifact-ref final projection work, run:

    .venv/bin/python -m pytest tests/test_redraw_scope_ladder.py tests/test_response_frames.py::ResponseFrameTests::test_response_frame_canonicalizes_duplicate_artifact_aliases_in_final_projection tests/test_response_frames.py::ResponseFrameTests::test_response_frame_keeps_conflicting_duplicate_artifact_ref_repair_needed -q

For cleanup/archive retention policy, run:

    .venv/bin/python -m pytest tests/test_clean_repo_state_policy.py tests/test_self_learning.py -q

For response runtime lifecycle wiring, also run:

    .venv/bin/python -m pytest tests/test_response_semantics_runtime.py -q

These tests check that accepted learning remains soft orientation, backend runtime evidence owns graph-repair proposal truth, advisory-only surfaces do not synthesize repair work, graph patch lifecycle honors `off`/`shadow`/`stage`/`apply_safe`, `apply_reviewed` requires explicit authorization, `apply_enforced` is policy-gated, graph rebase stays successor-only and authorization-gated, invalid autonomy values stay safe `off` with diagnostics, self-learning sidecar retention stays visible, and live degraded liveness stays advisory unless hard runtime truth proves unavailability.

## Ghost Policy Sources

Use the source files deliberately:

- `OLLMO_FOR_AGENTS.md` is the external-client/operator guide.
- `GHOST.md` is Ollmo-specific runtime policy.
- `docs/GHOST_ROUTER.md` is the human-facing Ghost design and maintenance guide.
- `references/ollmo-contract.md` is the concise reference for ordinary Codex use.

Read `GHOST.md` only when debugging Ghost behavior, asking why Ghost routed a request a certain way, modifying Ollmo Ghost policy, inspecting Ghost's current-turn intent/routing boundaries, or comparing runtime behavior against Ghost policy. Do not copy `GHOST.md` wholesale into ordinary Codex context.

## Architectural Rules

Preserve these principles when using or modifying Ollmo-related code:

- Runtime owns truth.
- Model output is not authority.
- Outputs are the public contract.
- Ghost anchors current-turn intent and routing; runtime, promotions, and closure decide what becomes real work.
- Accepted learning is reviewed soft orientation; current runtime/Closure truth remains the proof source.
- Graph repair is proposal -> validation -> staged/applied additive patch -> late fill -> closure. Runtime owns the transition.
- Graph rebase/redraw occupies the upper rungs of the same redraw scope ladder; its dedicated successor authority is not an additive-repair shortcut or a parallel layer.
- Intent obligations and redraw scope are runtime/Closure readouts; they orient repair/rebase but do not bypass validation.
- Candidate possibilities are not owed work until promoted.
- Old history or old artifacts do not become new intent unless the current turn explicitly refers to them.
- External clients should call Ollmo, not bypass it by talking to raw providers.
- Control-plane endpoints are preferred over raw backend ports.
- Local model calls execute selected phases; Ollmo runtime decides whether obligations are fulfilled.

## Anti-Patterns

Avoid these mistakes:

- Using stale `~/.codex/config.toml` `model_providers` as truth.
- Auto-syncing or mutating Codex config without an explicit user request.
- Starting, stopping, restarting, cleaning, archiving, unloading, removing, or resetting Ollmo from a read-only status/routing request.
- Calling `/api/responses` for a read-only route/capability question.
- Using `ollmoctl --recover-control-plane` as a "read-only" fallback when the user did not ask for control-plane recovery.
- Writing `model_ports.json`, runtime status, response frames, artifacts, logs, or provider config from read-only use.
- Treating cached observer surfaces as live probes, or using `refresh=true` without explicit intent to refresh runtime status.
- Treating computed Ghost preview evidence as pure observer truth or as automatic self-learning input.
- Treating `state/artifact_registry.jsonl` as canonical artifact bytes or as a strict append-only event log.
- Treating response artifact bundles as model-generated outputs, owed response work, or original response closure truth.
- Treating stale bundle registry rows, deleted bundle directories, or missing bundle entrypoints as current openable bundle truth.
- Repairing text artifacts by replaying the root prompt, redesigning unrelated content, or ignoring the bounded target-file closure repair payload.
- Treating `./ollmo clean` or `./ollmo archive --full` as harmless status commands.
- Treating one failed localhost probe as proof that no Ollmo instances are running.
- Treating Codex-localhost sandbox denial as an Ollmo runtime failure when Terminal/API evidence shows Ollmo works.
- Retrying raw backend ports, lifecycle commands, recovery, cleanup, or config edits after `EPERM` or `Operation not permitted`.
- Hard-coding raw backend ports when Ollmo control-plane endpoints can be used.
- Treating Ghost as a general swarm, debate, or orchestration layer.
- Loading `GHOST.md` into ordinary context when Ghost policy/debugging is not the task.
- Letting old history or artifacts become fresh intent without a current-turn reference.
- Inferring artifact creation or task completion from assistant prose instead of `outputs`, `artifacts`, response frames, and runtime truth.
- Starting a new `/api/responses` request when an existing `response_id` is still in `late_fill_pending` or `late_fill_running`.
- Polling full response payloads repeatedly when compact `?view=status` observation is enough.
- Treating compact status as artifact content; compact status is only for observation.
- Treating UI, chat-history, SSE, compact status, stale in-memory lookup state, or raw `output_text` as stronger than canonical `outputs`, response-frame output truth, and saved artifacts.
- Exposing provisional planner/status handoff prose as public output when fulfilled artifact outputs exist.
- Treating chat-only HTML/CSS/code output as a fulfilled file artifact.
- Treating a non-empty TTS artifact or STT transcript as semantic fulfillment without digest-bound producer-source evidence.
- Letting one vision/evidence branch inspect sibling media, or rewriting an invalid structured join into apparent success.
- Treating placeholder or guessed artifact links as final closure when concrete saved artifacts exist but are not linked.
- Ignoring `late_fill.fill_results` branch/path truth when final projection or hydration is missing saved image/audio/text artifacts.
- Letting repair/intermediate dossiers, generated-image text misbindings, or duplicate internal text artifacts become normal public bundle roots.
- Retrying required text artifacts by replaying the root prompt, or auto-retrying outside same-branch bounded retry gates.
- Treating accepted learning as executable authority, graph patch validation, closure truth, or proof of fulfillment.
- Treating run-monitor summaries as the only authority when backend response-frame truth is available.
- Treating advisory degraded liveness, cached/busy/provider-family warnings, or monitor-only evidence as a hard failure, broad provider ban, or graph patch proof.
- Treating explicit local image assets in page/site prompts as late polish instead of current-turn graph adequacy and dependency work.
- Applying `apply_reviewed` graph patches without explicit `graph_patch_authorization` on the concrete proposal review.
- Treating `apply_enforced` as enabled without the current `OLLMO_APPLY_ENFORCED_POLICY` gate and safe-v1 validation requirements.
- Using `OLLMO_GRAPH_REPAIR_AUTONOMY=apply_reviewed` as graph rebase authorization.
- Treating `shadow` or `stage` graph patch lifecycle records as executable graph mutation.
- Mutating a terminal/frozen response frame instead of recording `successor_reopen_requests[]` or `successor_rebase_requests[]`.
- Deleting self-learning-referenced response-frame sidecars without a retention manifest, retained copies, or visible missing-sidecar diagnostics.
