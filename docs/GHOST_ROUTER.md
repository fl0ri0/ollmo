# Ghost Router

This note describes the backend Ghost routing layer behind the frontend `Auto` mode.

Related runtime policy file:

- `GHOST.md` in the repo root is the canonical runtime policy injected into Ghost routing prompts and Ghost-owned user-facing chat turns.
- this `docs/GHOST_ROUTER.md` file is the human-facing routing contract.

## Goal

Keep Ghost's current-turn semantic resolution inside Ollmo's runtime-intelligence boundary, with any separate semantic-router call demoted to an explicit opt-in helper path.

Ghost should help Ollmo understand:

- what the current truthful phase is
- whether the phase is chat, OCR/vision, image generation, speech-to-text, or text-to-speech
- whether downstream document/file persistence is needed as an output/materialization step after a text phase
- whether the request is multi-phase and needs later continuation
- whether the user is referring to a recent artifact such as a generated image, OCR result, transcript, or saved text file

Ghost should decide the next truthful phase, not pretend the whole request is one flat capability choice.

Ghost is not the whole of Ollmo. Ghost is the semantic/current-turn interpretation layer inside Ollmo; Ollmo is the larger runtime/control-plane substrate that records phase truth, executes branches, persists artifacts, and preserves replayable outputs.

The current request rule is: intent is anchored by Ghost, graph state is refined by runtime evidence, and final freeze is allowed only after one bounded graph closure review.

The current contract lifecycle is: possibility -> relevance -> promoted contract -> runtime work -> review -> freeze. Ghost proposes semantic possibility; runtime graph truth, promotion review, output slots, artifacts, and closure review decide what becomes executable work.

## Boundary

Ghost is **not** orchestration.

It should:

- derive or respect the request phase graph when one is available
- propose candidate space for outputs, workload, context, references, evidence, repairs, and continuations without treating those possibilities as owed work
- classify current-phase runtime intent
- link relevant recent artifacts
- choose a capability or running instance for the current phase
- explain the choice briefly

It should **not**:

- pretend the whole request is one flat capability choice
- plan broad orchestration outside Ollmo's bounded request/phase substrate
- spawn subagents
- become a new team/debate layer

Execution-resolution note:

- Ghost stops at phase resolution and routing policy
- the request phase graph, control hints, resolver work, and late fill continue the already-decided request after routing
- that continuation substrate is normal runtime behavior, not a second assistant or orchestration swarm
- every Ghost-owned request should freeze into a request phase graph, even when a plain chat request later ends at phase 1
- old optional plan-refinement or critique reviewers are not the canonical graph closure loop; the closure loop is deterministic runtime review over graph, branch, slot, output, and artifact truth

## Routing Contract

### Inputs

- current user prompt
- attachment/file metadata
- recent conversation turns
- recent generated artifacts in the same conversation
- current ghost payload
- current runtime manifest
- request phase graph or continuation metadata when already available

Fresh turns normally use `current_turn_only` context strategy. Recent conversation turns and generated artifacts are injected only when the current request is clearly referential, continuation-like, or explicitly selects a reference.

### Router Outputs

Router and route-preview calls return strict JSON only:

- `capability`
- optional `instance_id`
- `reuse_last_artifact`
- optional `artifact_path`
- `confidence`
- `reason`

When a request phase graph already exists, that JSON must describe only the current truthful phase.

Ghost-owned user-facing chat is different. When the same `GHOST.md` policy is injected into a visible chat/materialization turn, Ghost must render the requested user-facing answer or artifact payload normally. It must not expose router JSON, request IR, candidate graph JSON, or control-plane schemas unless the user explicitly asks to inspect them.

### Safety

- validate the JSON before execution
- never trust free-form routing prose
- prefer existing phase-graph truth and deterministic runtime facts first
- keep bounded artifact-anchor helpers and embedding tie-breaks, but do not let heuristic route guesses become a competing final authority

## Current Implementation

The backend Ghost router now lives in the canonical `/api/responses` path when requests include `ghost_route=true`.

Current flow:

- the frontend `Auto` workbench sends the prompt plus a compact conversation snapshot
- the backend builds routing context from prompt, attachment metadata, explicit references, selected recent artifacts, Ghost payload, runtime manifest, and request-phase context when present
- the default fresh-turn context strategy is `current_turn_only`; older message windows or compressed history may be available for budget/reference hygiene, but they are not fresh intent
- recent artifact continuity should prefer durable `artifact_ref` identity and artifact-dossier truth over raw copied-path reuse when that data is already available
- the backend injects the repo-local `GHOST.md` runtime policy into both Ghost routing context and Ghost-owned user-facing chat context so current-turn interpretation and visible Ghost replies share the same maintained policy source
- the runtime manifest carries richer provider truth per instance, including package/contract identity, feature and modality summaries, compact session-control summaries, and compact backend metadata/runtime summaries where available
- Ghost derives or respects a request phase graph for Ghost-owned requests before treating the turn like a single flat route choice
- the general candidate layer can describe possible outputs, workload tasks, context, references, evidence, repair paths, and continuations before any of them become executable obligations
- promotion review is the boundary between possibility and owed work; rejected, reserved, or merely possible candidates stay visible but non-executable
- workload tasks surround output obligations with branch-local inputs, dependencies, lifecycle stages, output contract, visibility, and review criteria
- plain chat may end at phase 1, while Ghost-owned image/audio requests now default to `chat -> downstream materialization branches`
- simple chat, graph-resolved prepare phases, bounded explicit direct contracts, and artifact-aware follow-ups now resolve on Ghost's own current-turn path
- if a running helper instance actually exposes embedding support and a supported embedding transport, the backend can pre-rank live capabilities and instances and pass those results into the router prompt as soft `embedding_hints`
- the backend validates bounded helper output and resolves the final instance for the current phase, but the normal path no longer depends on heuristic or secondary semantic route authority
- once `/api/responses` has returned a `response_id`, clients observe that same response instead of reposting the root prompt; while late fill work is open, use `GET /api/responses/<response_id>?view=status` for compact lifecycle/branch state and fetch the default response view only when `state_version` changes or artifact detail handles are needed
- bounded rollout/debug evidence is opt-in via `GET /api/responses/<response_id>?view=debug`; fully hydrated response-frame/runtime truth is opt-in via `view=full`, `view=raw`, or `view=truth`; normal UI rendering and reconciliation must use `view=status`, the default UI projection, artifact entries, and targeted artifact endpoints
- normal POST/default/UI/status/debug results never hydrate sidecars and are byte-budgeted to at most 8 MiB for the final serialized outer envelope; retry/control/status wrappers and each serialized Responses-style SSE event are included in that ceiling
- public compaction is byte-based rather than a fixed character/item cut: ordinary 5,000-character text and 65 tiny records remain intact when they fit, while strings from 256 KiB and collections from 1 MiB may become a preview plus exact count/length, byte size, SHA-256, and an adjacent content-addressed `*_snapshot_ref`
- exact `full`/`raw`/`truth` reads recursively hydrate and validate authoritative CAS refs; missing, malformed, or corrupt refs fail closed with HTTP 409 instead of producing partial canonical truth
- response artifact bundles do not infer a complete artifact set from truncated or emergency wire handles; they hydrate exact CAS truth and fail closed if the complete inputs cannot be validated

The bounded wire is therefore truth by reference, not a semantic summary. Ghost and UI observers can work from lifecycle, handles, previews, counts, and digests, while replay, operator, and bundle paths can recover the exact content without semantic loss.

Current operational guardrails:

- obvious plain-text chat may still fast-path when current-turn truth already describes a terminal chat phase
- plain text-only chat requests can also fast-path when there is no artifact reuse, attachment ambiguity, or competing modality signal
- selected-reference and latest-artifact follow-ups can resolve directly on Ghost when the current turn plus artifact truth is sufficient
- Ghost-owned image/audio requests should normally freeze into a prepare phase on chat rather than jumping straight to the materializer
- bounded route hints and embedding tie-breaks remain helper tools only; they do not compete with Ghost's current-turn meaning

Current provider-truth boundary:

- `model_ports.json` remains the durable registry for stable instance/provider facts
- `state/runtime_status.json` remains the volatile runtime layer for readiness, health, and live backend runtime state
- Ghost consumes the merged control-plane view, so routing can use richer runtime truth without turning the registry into a transient status dump

Current frontend-to-backend fields for auto-routing:

- `ghost_route`
- `conversation_id`
- `ghost_messages` for JSON requests
- `ghost_messages_json` for multipart requests
- optional `batch_prompts` as a client convenience input for repeated image prompts after one Ghost route decision

Batch image note:

- Ghost still returns one route decision for the current phase
- `batch_prompts` does not itself force Ghost to pick `image_generation`
- if that final route resolves to `image_generation`, `/api/responses` can still execute multiple prompts sequentially via `batch_prompts`
- if the resolved capability is anything else, `/api/responses` rejects `batch_prompts` instead of attempting generalized cross-capability batching
- this is a client/frontend convenience path, not Ghost's canonical multi-output runtime model
- canonical branch fan-out now lives in the request phase graph, late fill, work tree, and `outputs`/`output_slots` substrate
- the same architectural rule now applies more generally: Ghost-owned image/audio materialization is prepare-first by default, while explicit direct contracts remain explicit exceptions
- the canonical response returns `results`, flattened `artifacts`, and `batch_count`
- the Responses tab also accepts an explicit JSON array of prompt strings in the prompt box for this first safe version

Current preview flow for model-specific requirements:

- `POST /api/ghost_route_preview`
- returns the resolved Auto target instance plus that instance's truthful `session_controls`
- returns cached runtime truth by default unless the caller explicitly passes `refresh=true`
- semantic helper and embedding evidence are preview-only computed truth: unset or invalid `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS` resolves to `off`; explicit `compute_semantics=true` or active non-UI policy `on` / `auto` enables compute, and `auto` currently behaves like `on`
- explicit `compute_semantics=false` is passive by default because unset or invalid `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE` resolves to `allow`; the UI uses explicit false for automatic Auto-target preview. Setting the override to `deny` lets active `on` / `auto` policy remain in force despite explicit false
- preview metadata discloses the truth boundary with fields such as `truth_mode`, `semantic_compute_requested`, `semantic_compute_performed`, `compute_semantics_source`, `compute_semantics_policy`, and `compute_semantics_false_override`; `semantic_compute_requested` is the effective resolved permission after request/policy handling, while `semantic_compute_performed` says helper work actually ran; `refresh=true` and semantic compute remain independent
- also returns the same post-route resolver/control-hint/runtime metadata that live execution uses, so preview and execution share one truth surface
- lets the UI validate required fields like VoiceDesign `Style / Instruct` before final `/api/responses` execution

Current execution safeguard:

- final `/api/responses` execution still validates required session controls for the resolved Auto target
- `request_ir.decision_contract` is the read-only role boundary for Ghost's richer semantic planning: it tells Ghost what can be proposed, reconsidered, promoted for review, waived for review, repaired, semantically reviewed, or superseded, but does not prove execution or create obligations by itself
- `decision_contract.semantic_planning_contract` is the advisory planning rubric Ghost should follow when proposing branch-local workload detail
- `decision_contract.active_reconsideration_review` turns open, blocked, reserved, waived, superseded, repair, and review signals into advisory review decisions before any state change is promoted
- `decision_contract.semantic_quality_review` turns subjective review criteria into explicit pending review contracts instead of treating output existence as quality proof
- `decision_contract.semantic_review_lens_review` gives Ghost and Closure an internal advisory lens for the task or check from the global `ollmo_g/semantic_roles/` library, such as possibility expander, structural planner, materializer, evidence reasoner, integrator, quality reviewer, risk sentinel, simplifier, repairer, or transition committer; it carries success definition, evidence requirements, and failure modes without becoming public `role` routing
- `decision_contract.semantic_role_orientation_review` compiles semantic role hints into advisory attention orientation only; legacy `ghost_mode` aliases must not change planner timeout, branch topology, payload shape, or closure truth outside the unified contract/lens loop
- `decision_contract.recursive_cycle_review` reports the same prepare/gather/execute/verify/repair-or-freeze cycle for each subtask, so depth is handled as repeated local motion rather than root-only planning
- `decision_contract.aspiration_review` keeps possibility, solution ambition, and non-minimal planning visible as advisory "great faith" when a graph risks collapsing to too little work
- `decision_contract.commitment_review` proposes the right-sized sufficient transition as advisory "great courage" when enough evidence exists and a branch should not drift in pending/review state or collapse to a too-small action
- `decision_contract.semantic_decision_review` translates those surfaces into advisory next-transition proposals with reason, confidence, evidence refs, allowed transitions, and accepted-learning orientation
- `decision_contract.controlled_attention_review` turns those surfaces into scoped model-attention frames; Ghost should answer each bounded question for the named branch, task, candidate, or review target instead of replaying the whole root prompt
- `runtime.graph_closure_review.global_semantic_closure_review` checks the whole turn after local runtime evidence exists; it can request bounded semantic review when outputs exist but their fit to the full intent is still unproven
- a completed global semantic review must normalize into `semantic_review_verdict`; `passed` can support truthful freeze, while `failed`, `uncertain`, or unparseable output remains open repair/manual-review/reconsideration state
- branch-level semantic review is available but demand-gated: use it for a specific fulfilled-looking branch when explicit `semantic_review_criteria` or non-deterministic qualitative criteria require proof; do not turn every branch into a reviewer loop
- Ghost route decisions should include `workload_task_proposals` for multi-task or dependency-bearing work when richer branch-local semantics, input refs, review criteria, output-contract hints, or bounded `execution_contract` candidates clarify existing workload tasks
- the request IR validates those proposals before they become graph truth; rejected proposals are recorded under `workload_proposal_review` and cannot mutate executable phases, dependencies, capabilities, output types, visibility, or required outputs
- `workload_proposal_review.coverage` records whether expected proposal coverage is missing, partial, complete, or not required
- accepted rich workload proposals are projected back into downstream branch records, so late fill can execute the branch-local contract without relying on the root prompt
- workload tasks are recursive branch contracts: each task can prepare, gather dependency evidence, execute, verify against review criteria, then repair or freeze before its dependents consume it
- reserved output/materialization candidates, such as "keep an image as an option" or "maybe later", remain non-executable until a later current turn explicitly promotes them
- selected-candidate requests, such as "generate only the second", promote only the selected candidate and leave sibling candidates reserved or unpromoted
- if required fields are missing, the backend returns structured `missing_session_controls` metadata instead of a vague downstream failure
- if a chat-mode execution path reaches a truthful text completion for a non-chat downstream request before the real artifact arrives, `/api/responses` freezes that valid chat moment with `late_fill.status = pending`
- late fill execution uses the promoted branch contract rather than the full root prompt: capability, output type, dependencies, `execution_contract`, `content_payload`, `artifact_prompt`, `stage_direction`, selected/reference artifacts, and runtime evidence are the branch-local task
- late fill handoff labels such as `Image prompt:`, `Poster prompt:`, and `Bild-Prompt:` are payload boundaries; later review/caption instructions must not become image/audio prompts
- counted TTS work uses explicitly labelled contiguous audio variants as candidate authority: runtime extracts exactly one speakable field per index and excludes wrapper prose, transcript claims, analysis, code, and JSON; duplicate, missing, or ambiguous slots fail as branch-contract repair
- each successful TTS fill binds the exact final backend prompt and SHA-256 in `tts_semantic_source`; a directly dependent STT fill compares its actual transcript only with that producer source and records `tts_stt_semantic_evidence`. Missing, drifted, ambiguous, or mismatching evidence blocks with `DEPENDENCY_CHAIN_REPAIR_REQUIRED` / `repair_dependency_chain`. A wrong WAV may remain preserved artifact evidence, but a non-empty WAV or transcript alone cannot fulfill the semantic contract, and expected source text is never inserted into STT
- the backend keeps the same `response_id`, continues late artifact fill in the background, and later merges the real image/audio artifact back into the current lookup payload while appending a successor frozen frame
- text/file artifacts are fulfilled only by materialized files recorded in output/artifact truth; chat-only code blocks, prompt lists, and control JSON are evidence or preparation, not final files
- linked artifact sets close only after concrete saved files are rebound to concrete saved dependencies with usable links from the referencing file; placeholders, guessed names, stale paths, and wrong relative links stay open as binding repair work
- if late fill fails or partially fails, the failed branch records the structured execution truth (`error`, `attempt`, `recovery_context`) while blocked output slots reference that branch with `error_ref`
- if a failed late fill branch is missing a required generated input artifact, recovery records `suggested_action = repair_dependency_chain` with branch-local dependency truth instead of offering a same-branch retry
- if the required dependency artifact already exists but is not attached to the blocked branch, recovery may record `rebind_dependency_evidence` so Closure can repair the binding instead of regenerating the artifact
- if a fulfilled-looking text branch has review criteria such as `uses_dependency_evidence` but no branch-local dependency evidence, closure review may keep it open with `repair_dependency_chain`
- if a fulfilled-looking branch has semantic criteria that runtime cannot deterministically verify, closure review may mark `semantic_review_required` with advisory authority instead of pretending quality was proven by existence; deterministic `review_criteria` stay runtime checks
- `runtime.graph_closure_review.decision_contract_review` summarizes the active decision contract for Closure; matched decision-contract guidance can flow into checks, repair feedback, and pending repair branches without making unpromoted candidates executable
- `decision_contract.block_resolution_reflex` is the global "resolve the block by its own right-sized verified transition" read-model; inspect it before turning a block into a blind retry, rewritten intent, under-scoped action, or modality-specific special case
- `runtime.graph_closure_review.surface_state` projects that same state language to clients: open, blocked, reconsiderable, waived, superseded, repair pending, semantic review pending, and completed
- semantic decision proposals may flow into Closure checks and repair feedback, but they are not executable commands; Runtime/Contracts/Closure still decide truth and promotion
- controlled attention frames may flow into Closure checks, repair feedback, late fill, and UI surface state, but they remain focus instructions only; they do not promote, execute, waive, supersede, cancel, fulfill, or freeze work by themselves
- semantic review lenses may flow into Closure checks, repair feedback, branch semantic review prompts, and UI surface state, but they remain review posture only; they do not route, promote, execute, waive, supersede, cancel, fulfill, or freeze work by themselves
- `ghost_repair_feedback.repair_loop` exposes bounded repair candidates, next actions, and whether any Closure-promoted contract is currently auto-executable. `auto_execute=true` means Runtime may schedule that concrete repair branch through late fill and keep it open until its bounded automatic repair budget is exhausted; `auto_execute=false` keeps the repair visible without running arbitrary Ghost candidates.
- the UI keeps watching that response and updates the original assistant turn in place when late fill completes or fails
- the UI should read `surface_state` and late fill branch arrays before stale output slots when rendering queued, blocked, review-pending, waived, superseded, or completed branch status
- immediately before final response freeze, `/api/responses` runs a graph closure review over the frozen request phase graph, work tree, output slots, runtime outputs, artifacts, and late fill state
- before that freeze, `/api/responses` may attach a refined `runtime.request_phase_graph` when strong same-turn evidence shows the initial graph missed an obligation; for example, assistant text that explicitly claims `text_to_speech` is pending can create the missing TTS branch instead of allowing a false text-only completion
- the closure review writes `runtime.graph_closure_review`; developer builds may also mirror the same review under diagnostics
- the closure review can allow freeze, mark a branch pending for late fill, record blocked/failed obligations, or explain why a partial result is truthful, but it must not create a new semantic user intent
- closure repair may promote repair candidates grounded in the existing graph/runtime state, but it must not turn stale history or a fresh guess into new owed work

Frontend naming note:

- the UI presents this path as `Auto`
- backend/internal names remain `ghost_route`, `ghost_messages`, and `ghost_route_preview` for compatibility

Current preference boundary:

- `primary_target` / `fallback_target` remain part of the current Ghost preference payload, but they are not blanket downstream execution locks
- in the current UI shape they should be read mainly as Ghost and chat/vision routing preferences
- if the current request strongly resolves to a different capability such as `image_generation` or `text_to_speech`, incompatible chat-only preference targets are ignored during final execution selection
- if an earlier preview payload stayed on chat but the fresh live Ghost/current-turn route strongly resolves to a non-chat capability, live execution ignores that stale preview route and recomputes against current truth

Current response metadata surfaced back to the UI/client:

- `route_source`
- `route_reason`
- `route_confidence`
- `route_reuse_last_artifact`
- `route_artifact_path`
- `context_mode`
- `context_reason`
- `runtime.context_strategy`
- optional `runtime.intake_context`
- optional `runtime.graph_closure_review`
- optional `runtime.route_traits`
- optional `runtime.embedding_audit`
- optional resolver metadata under `runtime.execution_planner`
- optional `runtime.control_hints`
- direct `recent_messages`, `recent_artifacts`, `latest_artifacts`, and selected-reference anchors

## Trait, Context, and Memory Behavior

Ghost can use same-capability trait signals during final instance handoff instead of relying only on readiness/activity.

Ghost should consume the merged live control-plane truth for each instance, not a hard-coded per-capability stereotype:

- `model_ports.json` plus live runtime/session-control metadata are the source of truth for what a model actually supports right now
- route context preserves dynamic control summary details such as visible fields, required fields, field types, option lists, and generic dynamic model traits where available
- when multiple instances share one capability, Ghost should prefer the instance whose truthful controls best match the request instead of only the simplest fallback

Current first-pass trait signals include image-aware chat support, tool/function-calling support, and active or declared context budget.

Routed chat execution records whether Ollmo used `current_turn_only`, `recent_history`, `compressed_history`, or `bounded_file_context`.

The current compression path is deterministic and local: older turns can be collapsed into one compact system summary when the recent history would otherwise be too large for the current context budget. Compression is a context-budget tool, not a way to turn old actions into the next request's intent.

Ghost also carries a shared normalized intent layer used by bounded artifact/follow-up helpers and prompt-to-control extraction:

- prompt text is normalized once with accent folding, whitespace cleanup, and simple typo repair
- capability scores can be raised by indirect and multilingual cues instead of only narrow command words
- compact intent evidence is passed into the bounded Ghost route prompt as soft context

Live Ghost routing no longer carries a separate derived memory chain.

Current live-router boundary:

- the default live Ghost router prompt does not replay thread-history messages on fresh turns and records that shape as `current_turn_only`
- only clearly referential or continuation-like turns carry thread-history messages, and then only as one answered prior user turn plus its direct assistant reply window
- artifact continuity is output-centered: all artifacts from the last relevant assistant artifact message are carried first, then one latest older artifact per type acts as fallback continuity
- selected references still override generic recency when the user explicitly anchors them
- implicit latest-image reuse is only for clear image references or genuine edit cues; abstract acknowledgements, process/coordination turns, or generic image-generation steering must stay off edit reuse
- explanatory runtime/process prompts that contain quoted or hypothetical multimodal examples stay on chat unless the outer request explicitly asks Ollmo to execute the example
- the live route-context object follows the same rule; it does not carry a separate `memory` block
- any remaining `hot_memory` / `warm_memory` / `deep_memory` helpers are archive/diagnostic-only and must not bias live routing
- `stable_user_preferences`, when inspected directly, are explicit-preference-only and are not inferred implicitly from ordinary user turns
- visible assistant text must not claim `saved locally`, `artifact created`, `[artifact: ...]`, or similar success unless runtime truth actually contains that saved output or artifact

These archive/helper names are diagnostic implementation details and are not part of the live Ghost routing contract.

Compiled memory still exists as an archival/runtime-diagnostic surface, but it is not a competing live router memory.

## Route vs Detail Fill

Route selection is only the first step.

A route can be correct while detail filling is still weak or incomplete.

Post-route detail fill is where Ollmo maps natural-language requirements onto the chosen instance's truthful controls, such as:

- language
- style / instruct
- response format
- speaker / voice
- OCR/document mode
- output dimensions or format

The resolver and late fill then continue unresolved downstream phases under the same request/response lineage. They continue only obligations that already exist in the request phase graph or late fill state.

Each downstream branch should carry a focused work contract. The materializer should receive the specific payload or artifact prompt it needs, not the whole original multi-task user prompt. If a later text branch depends on generated media, it should consume the resulting artifact reference, dossier enrichment, vision evidence, transcript evidence, or other dependency result after that media exists. For exact-spoken-text confirmation, transcript evidence is sufficient only when `tts_stt_semantic_evidence` binds it to the exact digest-backed `tts_semantic_source` of its declared TTS producer.

Old single-turn route-rating and learned-policy feedback are retired from the canonical runtime path.

## Graph Closure Loop

The closure loop belongs to the canonical `/api/responses` freeze path, not to optional reviewer experiments.

The loop asks:

- what did the request phase graph require?
- what did runtime truth actually produce?
- which output slots, branches, artifacts, or late fill obligations are fulfilled?
- which obligations are still pending, blocked, failed, or newly clarified?
- which pending obligations may continue inside the same intent?
- which would become a new user intent and must not continue automatically?

Allowed outcomes:

- freeze as complete when graph requirements are fulfilled
- freeze as truthful interim output with `late_fill.status = pending` when existing downstream obligations can continue
- update lookup/frame state after late fill materializes a pending branch
- report blocked, partial, or failed state when runtime truth cannot satisfy an existing obligation
- record dependency-chain repair when an obligation is real but its required source artifact was never materialized
- keep failures diagnostic and branch-local: branch errors explain why execution failed, slots expose compact references, and no automatic retry may create new intent

Disallowed outcomes:

- infer a new modality from stale history alone
- reinterpret the user's request after execution
- retry the same branch when the real defect is an absent dependency artifact or an invalid branch-local task
- execute a `repair_dependency_chain` or `repair_branch_contract` branch as normal materialization before the missing evidence or bounded contract exists
- freeze a dependency-sensitive text branch by text existence alone when its review criteria require generated artifact or evidence input
- use local model critique as the source of fulfillment truth
- let a secondary reviewer replace graph/runtime closure review

Related diagrams:

- [State Substrate Architecture](diagrams/ollmo-state-substrate-architecture.html)

## Compound Execution and Artifact Continuity

Routed non-chat requests can trigger a local resolver (`execution_planner`) before final backend execution.

Current resolver coverage focuses on downstream artifact materialization such as:

- `text_to_speech`
- `image_generation`
- text-first image or audio follow-ups that need a carried text phase before materialization

Text/file artifact materialization follows the same contract. A clear request for a README, markdown file, HTML, CSS, JavaScript, or plain text output can become a text artifact obligation. A deictic request such as "make this an HTML artifact" needs a selected source, current payload, or explicit reference; without one it should clarify rather than saving Ghost's own explanation. If a model returns a structured wrapper such as `output_obligations[].content`, persistence must save the declared content payload and not the router/control wrapper around it.

In active image chains, the same resolver can also rewrite short natural edit requests against the latest cached `image_state`, so Ghost does not need to memorize every wording variant just to preserve subject, scene, and style.

When the resolver succeeds, Ghost runtime metadata exposes compact resolver metadata under `runtime.execution_planner` with:

- whether resolving was attempted or applied
- which local chat-capable resolver instance was used
- the resulting semantic payloads or planned prompt
- which request fields were filled

If no resolver is available, the output is invalid, or resolver work is unnecessary, Ollmo falls back to the prior direct execution path. Local backend model calls remain execution/materialization calls; closure and fulfillment are decided from runtime state.

Current image-artifact continuity behavior:

- successful image-generation results can carry a compact cached `image_state` derived from the actual generated output image
- this state is attached to generated-image response payloads and artifacts, preserved through durable chat history, and reused by Ghost on later image follow-ups
- artifact dossiers keyed by durable `artifact_ref` gather identity, provenance, metadata, enrichments, linked response/message ids, and availability so later branches can reuse existing evidence when it is sufficient
- image-state enrichment prefers a running multimodal chat/vision model for description work and only falls back to OCR-style vision helpers when no better visual describer is available
- the Responses workbench can also pass a one-shot explicit older reference artifact for the next request; Ghost and the resolver treat that selected artifact as the active anchor for that turn, then later turns fall back to the newest output again unless the user selects another older artifact
- selected references are conversation-scoped UI state, so switching to another conversation does not keep the old pinned artifact globally active

Ghost uses cached `image_state` conservatively:

- actual-image anchors help vague style-only follow-ups stay grounded in what was generated
- ordinary subject carry-forward from earlier user prompts still keeps older inline rewrite behavior for prompts like `make it more cinematic` or `make a poster of it`

This keeps image continuity closer to the real artifact without forcing a fresh vision-description pass on every turn.

## Embedding Helper Note

Embedding-capable instances are runtime helpers for Ghost pre-ranking.

Helper eligibility is metadata-driven: primary capability, `provider_capabilities`, `outputs`, and backend-advertised endpoint paths can all contribute.

The same rule applies to mixed MLX/VLM routing. A VLM whose runtime metadata advertises both `chat` and `vision_analysis` is multi-capability and should be selected by the current task contract, while OCR/vision-only VLMs stay vision routes. For Single Chat, a selected MLX/VLM instance with a startup `vision_analysis` label can still rebound to effective `chat` when the request has no upload, file path, or selected-reference file context. If that selected MLX/VLM text transport fails with a provider 5xx/load error, backend runtime may retry once with that instance excluded and a compatible chat route selected from live truth. Per-wave Late Fill candidate snapshots are scheduling evidence only; if they are exhausted or contain only excluded/unusable candidates, the resolver refreshes live runtime truth before returning a no-route failure.

Mixed-capability models can therefore act as helpers when they actually expose embeddings.

Helper execution is transport-aware:

- Ollama helpers use `/api/embed`
- other helpers can participate when their backend contract advertises a compatible embeddings endpoint such as `/v1/embeddings`

They are not normal `/api/chat` or `/api/responses` targets.

If no supported helper is running, or if helper execution fails, Ghost records that fallback state in `runtime.embedding_helper` and behaves exactly like before.

Embedding influence is intentionally narrow:

- embeddings remain soft signals by default
- they can break a generic `chat` fallback only for narrow anchored follow-up classes such as pinned-image edits or artifact-linked read-aloud or transcription continuations when the top score and score gap are both strong

Ghost records compact `runtime.embedding_audit` metadata so later self-observation can compare:

- deterministic or compatibility route
- top embedding suggestion
- final route
- whether an embedding tie-break was actually applied

That audit exists to improve bounded evidence-based behavior, not to create a second hidden router worldview.

## Why This Fits Ollmo

This is still runtime intelligence:

- self-description
- self-linking
- phase-aware routing
- better artifact continuity
- bounded self-healing

It does not cross the boundary into orchestration magic, which should stay with external clients.
