# Core Contracts

This note freezes the current Ollmo core vocabulary and request/artifact contracts. It is the contract companion to [Canonical Stack](CANONICAL_STACK.md), [Architecture Map](ARCHITECTURE_MAP.md), and [Truth Sources](TRUTH_SOURCES.md).

## Core Vocabulary

- `Ollmo`
  The whole product idea and surface: a continuable AI-work state model plus its current local runtime/control-plane embodiment. Ollmo turns intent into possibility space, obligations, runtime truth, closure review, and frozen response frames.
- `Ollmo core`
  The runtime plus control plane. Core owns canonical execution, runtime truth, durable artifacts, history, and response freezing.
- `Ollmo_G`
  Embedded runtime intelligence inside core. Ghost routes from runtime truth plus bounded hints; it is not a public orchestration framework.
- `extension space`
  Optional recipes, integrations, docks, and higher-order composition that sit around core without widening the core request contract.
- `candidate_graph`
  The general possibility layer for output, workload, context, reference, evidence, repair, continuation, and learning candidates before they become owed work.
- `promotion_review`
  The validation boundary that decides whether a candidate is relevant enough in the current turn to become a promoted contract.
- `decision_contract`
  A read-only Ghost decision surface derived from candidate, promotion, workload, obligation, promotion-suggestion, waiver-candidate, repair, supersession, semantic review, and accepted-learning state. It guides Ghost's proposals without becoming runtime truth or promotion authority.

- `semantic_planning_contract`
  A nested advisory rubric inside `decision_contract`. It describes Ghost's planning cycle, proposal requirements, current proposal obligations, and non-authority boundaries. It is not executable topology and does not prove fulfillment, waiver, supersession, or review.

- `block_resolution_reflex`
  A nested advisory read-model inside `decision_contract`. It applies the global rule that the solution to a block is the block's own resolution. It keeps open, blocked, reserved, stale, waived, superseded, repair, and semantic-review signals visible so Ghost/Closure can consider the right-sized verified transition without forcing completion or under-scoping the work.

- `active_reconsideration_review`
  A nested advisory review inside `decision_contract`. It turns reflex signals into reviewable decisions such as promotion relevance review, waiver evidence review, supersession truth review, repair contract review, semantic quality review, or continuation/repair review. It does not execute or change contract state.

- `semantic_quality_review`
  A nested advisory quality contract inside `decision_contract`. It keeps semantic review criteria visible as pending review work. Runtime output existence can satisfy artifact truth; it cannot by itself prove tone, seriousness, visual fit, usefulness, or evidence quality. Deterministic review criteria remain runtime/closure checks first.

- `semantic_review_lens_review`
  A nested advisory posture report inside `decision_contract`. It derives internal review lenses such as planner, worker, materializer, evidence verifier, integrator, quality reviewer, repairer, transition committer, or whole-turn reviewer, then attaches success definitions, evidence requirements, and failure modes. It is not public `role` routing and cannot execute, fulfill, waive, supersede, or freeze work.

- `recursive_cycle_review`
  A nested workload-cycle read-model inside `decision_contract`. It reports every task's branch-local cycle: prepare, gather evidence, execute, verify, repair or freeze. It tracks depth as graph state without imposing a fixed depth cap.

- `aspiration_review`
  A nested advisory orientation surface inside `decision_contract`. It is the technical form of "great faith": keep possibility, solution ambition, and non-minimal planning visible without creating obligations or runtime truth.

- `commitment_review`
  A nested advisory orientation surface inside `decision_contract`. It is the technical form of "great courage": propose the right-sized sufficient transition without force-completing, freezing, waiving, superseding, under-scoping, or executing work.

- `semantic_decision_review`
  A nested advisory next-transition review inside `decision_contract`. It combines active reconsideration, semantic quality, recursive cycle state, evidence refs, confidence, and accepted-learning orientation into proposal-only decisions. It does not promote, execute, waive, supersede, fulfill, or freeze.

- `controlled_attention_review`
  A nested advisory focus review inside `decision_contract`. It converts reconsideration, quality, recursive-cycle, semantic-decision, and learning signals into scoped attention frames with allowed transitions and evidence refs. It focuses model attention between steps but is not execution permission or runtime truth.

- `graph_repair_proposal`
  A proposal-only repair object for additive request-phase-graph changes. It may come from Ghost, Closure, decision contracts, accepted-learning orientation, or the backend runtime evidence bridge, but it is not executable graph truth until `validate_graph_repair_proposal` accepts it. Provider-family trouble is route-health, preference, or cooldown evidence by default, not a graph patch or broad provider ban. The run monitor may summarize graph-repair diagnostics, but it is observer-only.

- `semantic_review_verdict`
  A structured advisory verdict produced by a promoted semantic reviewer. It says whether whole-intent fit or qualitative criteria are `passed`, `failed`, or `uncertain`, lists criterion results, evidence refs, defects, confidence, and the recommended transition. It is evidence for Closure, not freeze authority by itself.

- `branch_semantic_review`
  A Closure-promoted branch-local semantic verifier. It is available for any branch but created only when explicit `semantic_review_criteria` or non-deterministic qualitative criteria require it. It reviews one branch's contract, output, dependencies, and criteria, then returns `semantic_review_verdict`.

- `global_semantic_closure_review`
  A Closure/runtime review for whole-turn semantic fit. It checks whether local branch truth satisfies the current intent together and may promote bounded semantic-review work when fit is unproven.

- `surface_state`
  A Closure/runtime projection for UI and diagnostics. It summarizes open, blocked, reconsiderable, waived, superseded, repair-pending, semantic-review-pending, and completed state from runtime truth. It is not a frontend authority and should not be inferred from prose.

- `repair_rebuild_contract`
  A Closure-promoted contract for bounded repair/rebuild work. It is created from open Closure Review checks, not from Ghost suggestions alone. It may be schedulable through late fill or blocked until dependency evidence, a branch contract, manual review, or semantic review exists.
- `promoted contract`
  Executable owed work derived from current evidence. Unpromoted, omitted, stale, rejected, or reserved candidates remain visible state but are not executable obligations.
- `workload_task`
  The branch-scale task contract derived from a promoted phase or obligation. It carries declared inputs, dependencies, lifecycle stages, output contract, visibility, and review criteria so the runtime can verify the small task as well as the whole request.

Ghost is part of Ollmo, not a synonym for Ollmo. Ghost is the semantic/current-turn interpretation layer inside the larger Ollmo runtime/control-plane substrate.

## Retired Core Concepts

The following are retired from first-party core UI and canonical backend contracts:

- `Expert Mode`
- `pipeline`
- `role`
- generic request `mode`
- public `execution_profile`

Core may still accept those fields at the ingestion edge as deprecated compatibility shims, but they are not canonical request truth and must not reappear in first-party payloads, docs, or routing prompts.

## Canonical Request Hints

Core requests may carry:

- compatibility `ghost_mode` API alias
- `capability_hint`
- `language_hint`
- `developer_flags`

`ghost_mode` is a narrow compatibility hint at the API edge. It does not create a second orchestration surface, does not replace runtime truth, and does not directly control planner timeout, branching, payload shaping, promotion, waiver, supersession, or freeze. When present, it is translated into `semantic_role_profile` and compiled into `decision_contract.semantic_role_orientation_review` as advisory orientation frames. Those frames may bias controlled attention only when they do not conflict with branch-local semantic review lenses, execution contracts, runtime evidence, or Closure truth.

## Routing Contract

Ghost routes from:

- prompt and normalized messages
- explicit `input_artifacts`
- explicit `reference_artifacts`
- live runtime truth
- bounded request hints
- request phase graph when one is already available
- referential-only recent thread context when the current turn clearly points back, explicit-only stable preference statements when present, and output-centered artifact continuity anchors

Fresh turns normally use a `current_turn_only` context strategy. Older history, old tool calls, and prior artifacts may explain references, but they must not become the next turn's intent by recency alone.
Ghost does not treat retired orchestration vocabulary as canonical routing truth.
Live Ghost routing does not carry a separate derived `memory` block. Compiled memory remains archival/runtime-diagnostic state and must not act as competing fresh-intent route authority.
Every Ghost-owned request freezes into a request phase graph. The graph is frozen in intent and fluid in state: Ghost anchors the user's request, while runtime evidence marks graph obligations fulfilled, pending, blocked, waived, superseded, failed, or clarified. Plain chat may end at phase 1; image/audio materialization defaults to prepare-first plus downstream branches unless the caller supplied an explicit low-level direct contract. If an initial graph is too thin, pre-freeze graph refinement may add a missing downstream branch only from strong same-turn evidence such as an explicit pending/queued assistant output claim; this is continuation of an anchored obligation, not new intent creation.

`request_phase_graph.intent_obligations` is the normalized current-turn promise ledger. It decomposes coarse asks into text artifacts, media artifacts, evidence branches, dependency bindings, navigation promises, and other structural checks before they are surfaced back through runtime truth. The ledger is not a separate executor: only promoted graph branches/phases and validated runtime patches create owed work. Strong current-turn dependency obligations such as local generated image assets before HTML consumers may shape branch dependencies before execution; binding-only relations such as shared CSS or page navigation remain visible structural promises unless runtime evidence requires an executable repair.

The general contract lifecycle is: possibility -> relevance -> promoted contract -> runtime work -> review -> freeze. Ghost may propose candidates, promotion suggestions, waiver candidates, repair paths, semantic-review work, and supersession candidates, but runtime validation decides which candidates become output obligations, workload tasks, context promotions, dependency repairs, continuations, waivers, or supersessions. `decision_contract` is the compact read model for this boundary. Reserved/omitted/stale candidates are reconsiderable and may inform planning, but they are preserved as graph truth without becoming late fill work. Superseded obligations are closed by newer runtime truth instead of being retried as missing work.

The block-resolution reflex is the same lifecycle seen through failures and vacancies. It does not wait for a dramatic error: pending work, blocked work, reserved candidates, stale candidates, waivers, supersessions, repair candidates, and semantic-review needs are all reconsideration signals. The safe movement is always the right-sized verified state transition: continue, repair, wait, clarify, waive, supersede, or freeze truthfully at the scope that actually matches the intent and evidence.

Active reconsideration is the operational read-model for that movement. It does not decide stronger semantic truth by itself, but it gives Ghost, Closure, and UI the same list of state transitions that deserve attention now. Quality review and recursive cycle review are part of the same language: subjective success stays pending review until evidence exists, and every subtask should expose the same mini-cycle as the root request.

Aspiration and commitment complete the orientation triad around the existing doubt/review surfaces. Aspiration opens the coherent solution space before the graph collapses to too little work. Doubt verifies evidence through reconsideration, semantic quality, verdicts, and Closure. Commitment proposes the right-sized sufficient movement once enough truth is visible. All three speak the same lifecycle language, and only Runtime/Contracts/Closure can turn proposals into truth.

Scale movement is part of the same contract language, but it has two separate axes. Shallow/deep is semantic depth: visible request, intent, evidence meaning, quality, contradiction, learning, and aspiration/doubt/commitment. Coarse/fine is structural granularity: whole turn, candidate set, branch contract, payload, dependency, artifact evidence, and review criterion.

During discovery, Ollmo may move shallow -> deep when surface truth is insufficient, and coarse -> fine when relevant work needs decomposition. During integration, it may move deep -> shallow when semantic review has produced enough truth for a practical next action, and fine -> coarse when artifact or branch truth must resolve into global closure. These zoom directions are advisory orientation, not authority; Runtime/Contracts/Closure still decide truth.

Semantic decision review is an advisory layer over that movement. It gives Ghost and Closure structured candidate decisions with reason, confidence, evidence refs, and allowed transitions, while remaining subordinate to promotion review, Closure Review, runtime evidence, and artifact truth.

Semantic role orientation is the compatibility bridge for old Ghost modes. `repair`, `worker`, `explorer`, and `improviser` remain accepted request hints, but they are only API aliases. They translate into `semantic_role_profile` and advisory frames inside the same decision contract, where semantic lenses and runtime truth can accept, ignore, or supersede the hint.

Semantic review lenses sharpen that brain loop without becoming a new authority surface. A lens tells a model whether the branch should be judged as planning coverage, worker execution, materialization, evidence verification, dependency integration, quality review, repair, transition commitment, or whole-turn fit. The lens travels with semantic quality contracts, controlled attention frames, Closure checks, branch semantic review prompts, and repair feedback so the model asks the right question at the right scope.

Global semantic closure is the whole-turn review layer. It does not replace structural `intent_graph_adequacy`: structural adequacy asks whether the graph has enough promoted obligations and whether the normalized intent obligation ledger is represented by the graph shape and executable dependencies. Global semantic closure asks whether fulfilled branches fit the current intent together. When semantic fit is unproven after local obligations are complete, Closure can promote a bounded `global_semantic_closure` check and a `semantic_review` branch. The resulting review must normalize into `semantic_review_verdict`. Closure can freeze only when that verdict passes; failed, uncertain, or unparseable verdicts stay visible as repair/manual-review/reconsideration work.

Branch execution is local to the promoted branch contract. Later branches should consume their own `content_payload`, `artifact_prompt`, `stage_direction`, dependencies, input artifacts, reference artifacts, and prior branch outputs instead of treating the full root prompt as the task again.

Late-fill executor handoff uses a bounded `execution_contract`. The contract carries branch identity (`branch_id`, `phase_id`), workload task identity, output obligation identity, output contract, dependency refs, and artifact-control hints into `/api/infer`, and infer results echo that identity back for closure/review. It is not the full workload graph and does not move planning authority into `/api/infer`.

Closure Repair is contract-driven. Closure checks may carry `repair_action` values such as `retry_same_branch`, `retry_excluding_instance`, `start_compatible_instance`, `repair_dependency_chain`, `rebind_dependency_evidence`, `repair_branch_contract`, or `rebuild_from_promoted_obligations`. Repair feedback preserves the branch-local `execution_contract`, workload task refs, output obligation refs, input refs, review criteria, and output contract where available, so repair can start at the failing contract edge instead of replaying the full root prompt.

Blocked repair contracts block blind materialization, not repair work. When a prerequisite is absent, contracts should expose `materialization_blocked=true` and `repair_work_available=true` whenever the dependency chain, branch identity, obligation, or contract source gives Ollmo a bounded local repair path. `needs_external_input=true` is reserved for cases where runtime truth cannot derive the missing prerequisite.

Closure Review reads `decision_contract` as a guidance surface for the same loop. Matched repair candidates may supply a missing repair action for an already-open check; semantic-review candidates can make qualitative review work explicit; supersession candidates remain advisory until Closure confirms replacement truth; reconsiderable candidates stay visible without becoming executable work.

Closure Review also projects `surface_state` for clients. The UI should show a block as a block with its branch, cause, and possible recovery direction instead of turning it into an endless spinner, a hidden omission, or a false completion. This is the surface form of "the solution to a block is the block's own resolution."

When Closure Review itself proves missing or blocked work, it may promote a bounded `repair_rebuild_contract`. The response runtime patches the current request phase graph with the promoted repair branch and then lets late fill decide execution or block state from the contract's `execution_policy`. This is the operational Reconsideration/Rebuild loop, but it remains bounded to runtime-proven open checks.

The same lifecycle repeats at branch scale. A promoted workload task should prepare focused inputs, bind or gather dependency evidence, execute or materialize the branch, verify the result against its output contract, and freeze the branch result into runtime truth. If the branch has explicit semantic criteria that cannot be proven deterministically, Closure can insert `branch_semantic_review` before treating the branch's semantic role as settled. That review is demand-gated, not universal.

Branch scale does not isolate the branch from the larger phrase. A branch may zoom inward for exact evidence or deeper meaning, but its result remains reviewable upward and outward against the whole current intent whenever global semantic closure, reconsideration, waiver, supersession, or repair requires it.

## Response Lifecycle Contract

The canonical execution path is `/api/responses`.
Detailed response, late fill, slot, and recovery payload shapes are specified in `docs/RESPONSES_CONTRACT.md`.

Lifecycle:

1. normalize request and compatibility shims
2. derive or update the request phase graph and candidate graph
3. run promotion review to decide which candidates become promoted contracts
4. preview or resolve the current phase route
5. build/update `working_frame`
6. execute the selected backend path or promoted branch-local task
7. run a bounded pre-freeze closure review against frozen graph/branch/artifact truth
8. continue through resolver transformation, self-heal, dependency-chain repair, or late fill only when existing frozen obligations still need completion
9. freeze `response_frame`
10. persist history, artifacts, provenance, and lookup state

Current frame truth:

- `working_frame` is the mutable live request image
- `response_frame` is the frozen request image
- `planning.artifact_flow.work_tree` is the canonical internal work tree only when it is marked `work_tree_source = "runtime_owned"` and `authoritative = true`
- `work_tree_source = "derived_planning_snapshot"` with `compatibility_derived = true` is a legacy compatibility view, not deepest runtime truth
- `planning.artifact_flow.output_slots` is one projection/view of that tree
- `response_frame.output.outputs` mirrors the canonical external output projection for replay/resume
- `response_frame.current_state` is recovery lookup state captured inside a frame; it does not make the frozen frame mutable
- persisted late fill updates are successor frames with parent linkage, not rewrites of the original frame
- persisted response-frame ledger rows are compact; bulky internal runtime/planning/review snapshots, context candidates, large request inputs, and repeated work-tree projections are preserved by sidecar snapshot refs and hashes instead of being inlined repeatedly in `responses.jsonl`
- successor frame rows inherit unchanged parent snapshot refs and store only new/changed `external_snapshots.items`; recovery/replay expands the effective manifest before exposing a current response view
- sidecar snapshots may recursively split only worthwhile large child structures into CAS child refs; this keeps full diagnostic truth while reducing repeated graph/contract/candidate payloads
- compact `working_frame` sidecars keep logic state inline and move prompt, input, graph, contract, context-candidate, and artifact-flow bodies into child snapshot refs
- `outputs[]` and `output_slots[]` are handle projections over artifact/work truth; they keep refs and status, while dossier/provenance/metadata/path details stay in artifact dossier snapshots and the artifact registry
- top-level response `status` is compatibility-facing; `lifecycle_state` is the canonical status field for continuation, late fill, and branch truth
- lifecycle openness is allowlist-based: `late_fill_pending` and `late_fill_running` are active continuation, while `late_fill_failed`, `late_fill_completed`, `completed`, `failed`, `cancelled`, `waived`, `superseded`, `skipped`, and `frozen` are non-polling terminal/frozen states
- `blocked` is the canonical blocked lifecycle value; `late_fill_blocked` is a compatibility wording for blocked repair/control truth and must not imply active execution

Current request-shape truth:

- every Ghost-owned request first freezes into `request_phase_graph`
- plain chat can complete at the current phase without downstream branches
- image/audio requests normally keep the current phase on `chat` and materialize the final artifact through downstream branches
- text/file artifact requests are output materialization obligations only when the source payload is clear; ambiguous source language such as "this" without a selected source should clarify instead of persisting a guessed artifact
- structured text/file artifact wrappers such as `output_obligations[].content` are payload envelopes; persistence saves the declared `content`, not the router JSON or control-plane metadata around it
- immediately before freeze, Ollmo should run one bounded closure review that can continue only already-anchored obligations; it may refine graph state from strong runtime/output evidence, but it must not reinterpret user intent or invent new capability goals
- that review is surfaced as `runtime.graph_closure_review` and may also appear under developer diagnostics
- local model calls execute selected phases or materialize branches; graph, slot, output, artifact, status, and late fill state decide fulfillment
- branch-local model calls should carry `execution_contract`, `workload_task_ref`, and `output_obligation_ref` through late fill and infer payloads so completion can be matched to the planned branch without relying on model wording
- if a promoted branch cannot run because its required generated input artifact is absent, recovery should record `repair_dependency_chain` rather than offering same-branch retry
- if the branch itself is under-specified, recovery should record `repair_branch_contract`; if the whole graph lacks promoted obligations for the current intent, recovery should record `rebuild_from_promoted_obligations`
- visible file/artifact claims in assistant text must be truth-gated by runtime outputs; if no saved artifact exists, the frozen response must not claim one was created
- explicit low-level direct contracts such as direct `instance_id` execution or explicit `batch_prompts` remain explicit exceptions

Current public response truth:

- top-level `outputs` is the canonical external output surface
- canonical `outputs` originate from promoted output slots / work-tree nodes and carry `compatibility_derived: false`
- `output` and `output_text` remain compatibility projections derived from `outputs`
- fallback `outputs` derived from `output_text`, `content_payload`, or stray artifacts must be marked `compatibility_derived: true`
- `output_slots`, `output_branches`, and `work_tree` remain exposed for branch/slot/tree-aware consumers
- runtime-owned `work_tree` wins over fallback frame planning; derived work trees exist only as compatibility snapshots when no runtime work tree is present
- failed late fill branches keep structured branch truth under `late_fill.failed_branches[].error`, `attempt`, and `recovery_context`; blocked slots and public output projections carry only compact `error_ref` plus display-safe `blocked_reason`

## Artifact Contract

Artifact continuity is durable identity plus provenance, not accidental copied paths.

- `input_artifacts`
  Explicit user uploads or explicit local file submissions for the current turn.
- `reference_artifacts`
  Intentional references to prior artifacts or messages from history.
- `output_artifacts`
  Files/materialized outputs produced by the current response.
- `artifact_bindings`
  Internal execution bindings that resolve a durable `artifact_ref` or registry record to a local file path for a backend call.

Canonical durable fields:

- `artifact_id`
- `artifact_ref`
- `kind`
- `mime_type`
- `origin`
- `path`
- `source_message_id`
- `source_response_id`
- `provenance_id`
- `derived_from`
- `availability`

Rules:

- `path` is metadata attached to durable identity, not the identity itself.
- Executor temp copies and backend-specific materialization stay internal.
- Internal bindings may translate a `reference_artifact` into a temp file, but history and response frames must continue to point at the original durable `artifact_ref`.
- Only true external current-turn files are `input_artifacts`. A prior Ollmo output, selected reference, route-reused artifact, artifact binding, or registry-known path must remain a reference/binding and must not be copied back into the public input surface.
- Provenance is the durable continuity layer for generated outputs and derived artifacts.
- Canonical read-side retrieval is now artifact-centered: `artifact_dossiers` are keyed by `artifact_ref` and gather artifact identity, provenance, metadata, enrichments, and linked response/message IDs together.
- `state/artifact_registry.jsonl` is the durable materialized artifact index for concrete artifacts. It may merge provenance and integrity metadata by artifact identity; final saved artifact files remain the canonical artifact bytes. Text/document, audio, image, OCR/transcript, and other `artifacts[]` outputs are registered with `roles = ["output"]`; true external request files are registered with `roles = ["input"]`. This lets clients resolve artifact identity without mining `responses.jsonl` and without confusing reused outputs with fresh user input. Generated-image provenance remains the richest image-specific provenance and generic output registration must not overwrite it; generated-image helper enrichments append back onto the original artifact record instead of creating new input truth. A registered TTS WAV proves that bytes were produced, not that the audio obligation was fulfilled: source/file-bound `tts_audio_integrity_evidence` must pass effective-signal, duration, silence, and readability checks before promotion, while a failed file remains diagnostic artifact truth.
- `state/response_frames/snapshots/` stores large response-frame sidecars as content-addressed JSON. Multiple ledger refs may share one physical sidecar when the payload is identical; the refs preserve the semantic JSON path, while the SHA-256 path preserves the bytes once. Successor frames inherit unchanged parent snapshot refs and append only changed refs. Large nested runtime, graph, contract, candidate, evidence, and working-frame subtrees should be split into child refs, not lossy summaries. Volatile timestamp keys and path-local nested ref metadata inside graph/contract/context/work-tree diagnostics are excluded from the content hash; timing and semantic paths belong to frame/ref metadata, while the sidecar hash names stable diagnostic content.
- There is no separate generated-image provenance compatibility module anymore; image provenance is read and written through the artifact-registry/dossier system.
- Document/file saving is modeled as an output/materialization concern on top of text, not as a standalone routed capability.

## Compatibility Notes

- `/api/chat` and `/api/infer` remain compatibility surfaces; `/api/responses` is canonical.
- Deprecated request fields may be ingested and normalized internally, but first-party core surfaces must emit only the canonical vocabulary.
- Legacy selected-reference payloads may still be accepted, but canonical durable state uses `reference_artifacts`.
