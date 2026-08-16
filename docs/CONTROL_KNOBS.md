# CONTROL KNOBS

This document defines the primary control points in Ollmo.

Its purpose is practical:

- to understand where behavior comes from
- to debug problems quickly
- to separate concerns cleanly during testing and development

---

## Core Principle

When something feels wrong, the question is no longer:

> "Why is this system weird?"

The question is:

> "Which knob is responsible?"

Hidden hard limits are not valid knobs. Ollmo runs locally, so canonical policy, graph, artifact, and contract sources should stay open unless a bound is technically required for safety, concurrency, storage, transport, or a concrete model/request budget. When a bound is needed, make it explicit, name the knob, document the reason, and prefer graceful budget handling over silent truncation.

---

## The 5 Control Knobs

### 1. Ghost model (intent/graph knob)

This is the main **intelligence knob**.

It controls:

- how requests are interpreted
- how the work structure is formed
- how branches are created
- how rich or shallow the decomposition is
- whether fresh turns stay inside the current-turn intent bracket

Ollmo's routing posture is **Ghost-first**: Ghost owns intent and graph formation. Hardcoded heuristics are now narrow shadow/guardrail signals. They may help validate obvious runtime-truth edges, but they are not a competing primary graph builder.

Ghost structure is projected into a **workload graph** before execution. The workload graph is the general task layer: each phase becomes a task with declared inputs, dependencies, lifecycle stages, output contract, visibility, decomposition level, child tasks, and review criteria. The current v1 graph is derived deterministically from the existing request phase graph.

Ghost is expected to provide `workload_task_proposals` for multi-task or dependency-bearing work when richer branch-local semantics help execution. These proposals are advisory until validated: they may add bounded semantic intent, objectives, input-reference labels, review criteria, promotion suggestions, waiver candidates, and execution-contract detail for existing workload tasks, but they must not create phases, change dependencies, change capability, change output type, or change required outputs.

`request_ir.decision_contract` is the current Ghost-brain read model. It joins `candidate_graph`, `promotion_review`, `workload_graph`, `workload_proposal_review`, output obligations, promotion suggestions, waiver candidates, repair/supersession/reconsideration state, semantic review candidates, controlled attention frames, and accepted-learning orientation into one boundary object. Its `semantic_planning_contract` names Ghost's advisory planning cycle, proposal requirements, current proposal obligations, and non-authority boundaries. Ghost may use it to propose richer candidate sets, promotion review inputs, waiver review inputs, repair paths, semantic review, supersession, reconsideration, or scoped attention targets; it may not use it to assert runtime truth or bypass promotion review.

If something feels:

- misunderstood
- overcomplicated
- under-specified
- collapsed too early into simple chat
- contaminated by old history or previous artifacts

Then this is the first knob to inspect.

Primary posture:

- `ghost_first`: the request phase graph and runtime truth are the decision authority.
- `heuristic_role=shadow_guardrail`: hardcoded cues are compared and audited, but should not silently override Ghost.
- `accepted_learning_authority=soft_hint_only`: reviewed learnings can orient Ghost without mutating Graph, IR, Closure Review, routing, context promotion, output obligations, or graph-patch authority by themselves. Enabling an accepted snapshot does not upgrade this runtime effect.
- Route preview is read-only. Ghost route selection may choose a compatible live target, but it must not start, load, unload, or restart model instances.

Non-UI Ghost preview semantic compute policy:

- `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS` controls the default when `/api/ghost_route_preview` omits `compute_semantics`.
  - `off`, `false`, `0`, `no`, or `n`: omitted preview stays cached/passive.
  - `on`, `true`, `1`, `yes`, or `y`: omitted preview may use embedding or semantic helper compute.
  - `auto`: currently behaves like `on`.
  - unset or invalid: built-in default is `off`.
- Explicit `compute_semantics=true` may opt into computed preview even when the default policy is `off`.
- Explicit `compute_semantics=false` is the standard passive observer escape hatch. It opts out by default because `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE` defaults to `allow`; operators can set it to `deny` when an active `on` or `auto` policy must remain in force despite explicit false. Accepted allow values are `allow`, `allowed`, `true`, `1`, `on`, `yes`, or `y`; accepted deny values are `deny`, `denied`, `false`, `0`, `off`, `no`, or `n`; unset or invalid defaults to `allow`.
- This is not a visible UI knob. Do not add a visible toggle, button, or settings control for it. Automatic UI route preview sends `compute_semantics=false` to stay passive; computed preview remains available through explicit `compute_semantics=true` or by setting `OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS=on` or `auto` for callers that omit `compute_semantics`. Preview responses must disclose `compute_semantics_source`, `compute_semantics_policy`, `compute_semantics_false_override`, `semantic_compute_requested`, and `semantic_compute_performed`. `semantic_compute_requested` is the effective resolved permission after explicit-request/policy handling; `semantic_compute_performed` separately says whether helper work actually ran.

Brain knobs are deliberately split into hard knobs and read-model surfaces:

- Hard knobs:
  - `developer_flags.planner_timeout_ms` is the explicit resolver compatibility budget. The literal key is retained for request compatibility; semantic roles, `ghost_mode`, learning hints, or retry state must not add hidden timeout bonuses.
  - `semantic_role_profile.loop.max_passes` is currently fixed at `1`; `critic_passes` is `1` for normal runtime and `0` for preview. This is a cadence shape, not a user-facing intelligence dial.
  - `semantic_role_profile.runtime_orientation.runtime_effect` is `none`; roles can orient wording/review posture only.
- Read-model surfaces:
  - `candidate_graph` shows possible work. It is not executable until `promotion_review` creates a promoted contract.
  - `promotion_review` decides candidate-to-obligation movement; it is the place to inspect when something should have become owed work or should have stayed reserved.
  - `decision_contract.block_resolution_reflex` and `decision_contract.active_reconsideration_review` show whether the brain is preserving blocked/reserved/waived/superseded state instead of retrying or rewriting intent.
  - `decision_contract.semantic_quality_review`, `semantic_review_lens_review`, `semantic_decision_review`, `controlled_attention_review`, `aspiration_review`, and `commitment_review` are advisory attention/review surfaces. They can recommend repair, waiver, supersession, semantic review, or freeze, but Runtime/Contracts/Closure must still promote or confirm the transition.
  - `runtime.graph_closure_review.global_semantic_closure_review` is the whole-turn review surface. It may request bounded semantic review after local branches are complete, but review completion is not truth unless the verdict passes.

If the "brain" makes the wrong move, first determine which surface was wrong:

- wrong possibility set -> `candidate_graph`
- wrong owed work -> `promotion_review`
- wrong blocked/repair movement -> `block_resolution_reflex` and `active_reconsideration_review`
- wrong quality question -> `semantic_quality_review` or `semantic_review_lens_review`
- wrong model focus or root-prompt replay -> `controlled_attention_review`
- wrong final freeze -> `global_semantic_closure_review` and `surface_state`

---

### 2. Resolver logic (execution knob)

This is the **correctness and mapping knob**.

It controls:

- which model/provider is chosen per branch
- execution order
- `depends_on` sequencing between dependent branches
- fallback behavior
- same-level branch parallelism
- when something is considered ready to run
- whether late fill continues only existing graph obligations
- whether workload tasks are ready according to their declared dependencies

If something feels:

- the wrong model was used
- the wrong modality was triggered
- the wrong branch executed
- execution order seems wrong
- independent sibling branches are serialized unnecessarily
- dependent branches run before their evidence branch has finished
- pending work continues as if it were a new request
- route selection or preview appears to have started a model

Then this is the knob to inspect.

Primary bounded knob:

- `OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS`: caps same-level branch materialization workers. Runtime-module default is `4`; normal local startup currently exports `4` when unset. Values are clamped to a safe bounded range. Per-instance locks still prevent concurrent calls into the same selected local instance.
- `OLLMO_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS`: caps the default total automatic attempts for Closure-promoted auto-executable repair/materialization retry branches, including the first attempt. Runtime default is `6`, and values are clamped to a safe upper bound. Branch or repair-contract metadata such as `auto_executable_repair_max_attempts` may provide a more specific per-contract budget. This is not a UI knob; it is a non-UI policy boundary for "open until exhausted" repair so Ollmo resolves blocks gently without looping forever.

Start-source boundary:

- route selection, Ghost preview, late fill, and backend automatic paths reuse eligible running instances and must not force-start models
- explicit frontend play/start is a user lifecycle action; it may send `start_source=frontend_button` and `force_start=true` only for deliberate duplicate same-model starts
- live process, listening port, and backend runtime truth decide availability; stale degraded, busy, timeout, or cooldown projections are advisory unless that live truth says the instance is unusable
- read-like CLI commands, including `ollmoctl responses get`, must not recover or start the control plane by default; use `--recover-control-plane` only when the caller explicitly wants local control-plane recovery

---

### 3. Provider layer (capability knob)

This is the **output quality and modality knob**.

It controls:

- which models/providers are available
- the quality of generated outputs
- speed vs quality tradeoffs
- provider-specific output style

If something feels:

- low quality
- too slow
- inconsistent in style
- weaker than expected despite correct routing
- truncated despite the model being capable of longer output

Then this is the knob to inspect.

Generation token budgets should be explicit and high enough for the requested work. Normal chat/responses calls should not add an Ollmo-side `max_tokens` cap when the user did not request one; internal resolver and translation helper budgets should avoid small hidden ceilings. The explicit `planner_timeout_ms` developer flag is a named resolver compatibility safety budget, not a small hidden wait cap, and is allowed up to the local long-running job budget of 2 hours.

Qwen3-TTS is a model-specific exception because its generation tokens represent audio time and the MLX-Audio server otherwise defaults to 1200 tokens (about 96 seconds at 12.5 audio tokens/second). Runtime policy `qwen3_tts_adaptive_audio_tokens_v2` derives `max_tokens` from the final branch-local spoken text: it takes the slower estimate of 2 Unicode words/second or a script-aware character rate (12 ordinary non-space characters/second and 4 CJK/Kana/Hangul characters/second), applies a 1.5x duration margin plus 8 fixed seconds, converts at 12.5 audio tokens/second, and clamps to 256..1200. The inference result records the calculation as `tts_generation_budget`. Other TTS families do not inherit this Qwen3-specific budget.

Long Qwen3 Base, VoiceDesign, and CustomVoice WAV requests use `qwen3_tts_sentence_chunks_v1` when estimated speech exceeds 16 seconds. Runtime preserves ordered source spans, targets chunks of at most 10 estimated seconds, applies the adaptive budget separately to each backend call, verifies every returned chunk, and joins only compatible uncompressed PCM WAV frames into one persisted artifact. A single-sequence Qwen3 WAV that lands exactly on its declared codec-token ceiling is blocked as generation-limit exhaustion even if residual energy would otherwise resemble active signal. Base input containing multiple nonempty lines is marked `segmented_sequence` because MLX-Audio internally generates each line as a separate segment; Runtime does not claim an exact single-sequence cap proof for that aggregate. Chunk verification additionally rejects internal inactive gaps longer than 3 seconds. These checks add negative evidence; they do not replace or weaken the existing no-signal, effective-duration, trailing-silence, source-digest, or Closure gates.

Supported Qwen3 Base, VoiceDesign, and CustomVoice single-sequence generation has one narrow in-call recovery for a deterministically proven exact-ceiling result. Policy `qwen3_tts_single_sequence_generation_limit_retry_v2` retries only the exhausted logical sequence/chunk, only once, with the same source, language, voice/style instruction, format, and sampling controls. It changes only `max_tokens`, using `min(1200, max(initial + 128, ceil(initial * 1.5)))`. Eligibility reads the structured generation-limit evidence rather than the primary integrity reason, because a result can be too short, mostly trailing silence, and exactly at the hard cap simultaneously. Both attempts and their budgets, hashes, complete defect evidence, and selection state remain in chunk diagnostics. The retry bytes are usable only if the unchanged integrity policy passes them independently; a failed second attempt remains blocked. Segmented/unverified Qwen output, non-Qwen models, malformed audio, unavailable cap evidence, and Closure behavior do not use this in-call retry.

External downstream execution is one provider mode, not a transfer of runtime
authority. When Ollmo asks ChatGPT/Codex to execute an already-shaped branch,
`[OLLMO_DOWNSTREAM_EXECUTION_V1]` is the first non-whitespace content. The
provider executes only `<ollmo_bounded_task>` under the supplied
`<ollmo_promoted_context>` and must not invoke Ollmo, its companion skill, API,
CLI, or local models recursively. It must not widen the branch or create
follow-up work. Ghost planning itself is a different Ollmo-internal runtime
role: it does not use the companion skill and receives runtime manifest, model,
and capability orientation directly from Ollmo. If Ghost then resolves an
already-shaped branch to an external ChatGPT/Codex target, that target call is
downstream execution and does receive the marker.

The selected external target may remain eligible for later graph-owned
`chat` branches (for example review or text-artifact materialization) when it
is still selectable and is not explicitly excluded by recovery truth. This is
provider continuity, not admission to the generic local candidate pool:
image, audio, vision, speech, and other non-chat branches never use it. Ollmo
still performs every file write, syntax/integrity check, Closure decision, and
publication step; an external `BLOCKED:` result is recorded before any
materialization attempt.

---

### 4. Runtime / substrate (truth knob)

This is the **state and persistence knob**.

It controls:

- branch identity
- output_slots
- outputs
- replay/resume
- persistence and continuity of work
- workload task lifecycle state
- workload proposal validation and rejection records
- normalized candidate graph and promotion review state
- pre-freeze graph closure review
- selective evidence handoff between phases and branches
- Closure Review repair feedback back into Ghost when runtime truth does not cover the graph

If something feels:

- outputs are duplicated or missing
- continuation is wrong
- replay is inconsistent
- branch/output state is drifting from reality
- final text claims an artifact exists when runtime truth does not contain it
- a final HTML/CSS/media artifact points at placeholder, guessed, stale, or duplicated generated dependency paths
- a later branch consumed hidden/stale text instead of declared runtime evidence
- a requested `.txt`, `.md`, `.html`, `.css`, `.js`, or other text-like file appears only as chat text and not as a saved artifact
- a branch receives the whole prior answer when it should receive only a prompt, script, caption, or other bounded evidence segment
- a final text branch after generated media receives only artifact paths instead of declared analysis/transcript evidence

Then this is the knob to inspect.

Text-like output artifacts are runtime-backed only when the current turn explicitly asks for file/artifact/export/save/download behavior, names a concrete text-like filename/extension, or edits a selected text-like source artifact. Detection is scoped to the latest user turn; previous assistant text or artifacts may provide selected source evidence, but must not turn a fresh ambiguous `this` request into an artifact unless that source is explicitly selected and text-like. Ordinary chat text remains inline text and does not create a file artifact by default.

Multiple text-like artifacts may materialize from one response only when the latest turn explicitly asks for multiple file artifacts and the model output exposes clear matching payloads, such as fenced `html` plus `css` blocks. Structured wrappers such as `output_obligations[].content` are payload envelopes: persistence saves the declared `content`, not the surrounding router/control JSON. README is normalized as a markdown text artifact when requested as a file. The legacy `saved_text_path` remains the first artifact for compatibility; `saved_text_artifacts` and canonical `artifacts` carry the full list.

Linked artifact sets close only when concrete saved files point to the concrete saved dependency artifacts they expose. If a surviving generated output path does not resolve to a saved local artifact, or if duplicate substitution reuses one saved path before all owed saved artifacts are bound, closure remains repair-needed/non-complete unless Closure records a waiver or supersession.

Local image asset requirements are graph adequacy, not late presentation polish. A page/site prompt that asks for local generated or embedded images, local image assets, or non-external image paths must promote image-generation branches even without an exact image count. Structural hints such as "two image sections" set the minimum count, and subpage/image-asset language can raise that minimum. HTML/CSS artifact branches that must reference those images should depend on the generated image phases and carry `dependency_contract=local_visual_asset_binding`.

Post-media text follow-ups need evidence branches, not just paths. Requests such as "generate two images, then compare them" should form `image_generation -> vision_analysis -> chat`; requests such as "turn it into audio, then confirm the exact spoken text" should form `text_to_speech -> speech_to_text -> chat`.

Generated-image evidence branches must receive focused analysis prompts for the attached/generated artifact. They may keep the original request as bounded intent context, but must not execute the original multi-step request again.

Generated-media graph joins must preserve artifact-specific dependencies. Image analysis depends on generated image phases; spoken-text checks depend on generated audio phases; final text joins depend on those evidence phases, not on whichever media branch happened to appear immediately before them in prompt order.

Text-to-speech branches must execute on the branch-local speakable text, not on wrapper wording such as "read that reply aloud". When the current turn or branch payload clearly indicates language, the branch must carry `lang_code` through to `/api/infer`; backend defaults must not silently replace declared German, French, Spanish, or other spoken-language truth.

Explicitly labelled audio variants are the candidate authority for counted TTS work. Runtime extracts exactly one speakable field per contiguous variant index and excludes sibling transcript claims, analysis, code, and JSON; duplicate, missing, or ambiguous slots fail as branch-contract repair. A successful TTS fill records the exact final backend prompt plus its SHA-256 as `tts_semantic_source`. A directly dependent STT fill compares only its actual transcript with that bound producer source under the named deterministic fidelity policy and stores `tts_stt_semantic_evidence`. Missing, drifted, ambiguous, or mismatching source evidence blocks the STT branch with `DEPENDENCY_CHAIN_REPAIR_REQUIRED` / `repair_dependency_chain`, so downstream joins and Closure cannot treat a merely non-empty WAV or transcript as fulfillment. A physically generated wrong WAV remains immutable artifact/evidence truth rather than being deleted or reclassified as success. The expected source text is verification truth and must never be injected into the STT request itself.

TTS materialization also has an output-side physical integrity gate. For PCM WAV output, runtime measures fixed-window signal activity, effective active duration, total/leading/trailing silence, and source-relative minimum speech duration, and binds those facts to the exact `tts_semantic_source` digest and artifact SHA-256. `tts_audio_integrity_evidence.status=passed` plus `materialization_eligible=true` is required before the WAV can fulfill an audio output slot. Silence, severe truncation, excessive trailing padding, malformed/unsupported WAV, missing files, or source-digest drift fail closed. The file and artifact registry entry remain available for diagnosis, but output slots and Closure stay blocked/repair-needed. This deterministic gate does not secretly schedule STT; when semantic transcript fidelity is explicitly owed, the existing dependency-bound TTS-to-STT branch remains the stronger semantic proof.

Response-local exclusions remain authoritative for TTS retries. A non-excluded compatible instance is always preferred. An excluded instance may be reused only by an `explicit_retry_endpoint` or the runtime-owned `tts_auto_recovery` same-branch audio contract, and only after current runtime truth confirms that exactly one compatible usable TTS instance exists and it is the same failed/excluded instance. The automatic contract is additionally bound to required audio, policy `tts_bounded_materialization_recovery_v1`, attempt 2 of 2, the exact branch and prior retryable error, and retained integrity evidence when physical integrity caused the retry. This exception is exposed as `selection_policy=excluded_reuse_for_single_tts_recovery`, with the reuse reason, instance id, trigger, policy, and candidate diagnostics. It is not available to initial routing, other capabilities, multiple compatible candidates, or stale snapshot-only selection.

The workload graph does not replace output obligations. It surrounds them with per-task lifecycle and review metadata so the runtime can later verify every small branch using the same recursive prepare → gather evidence → execute/materialize → verify → repair or freeze cadence that applies to the full request. Depth is a graph property, not a fixed small cap.

`workload_proposal_review` is the audit point for Ghost-authored task annotations. Its `coverage` subrecord shows whether semantic proposals were `not_required`, `missing`, `partial`, or `complete` for the executable workload tasks. If a proposal is missing from the final graph, check whether it was never preserved by the router or whether the IR rejected it for structural mismatch.

Ghost may propose richer workload task contracts than deterministic heuristics can derive, including semantic objectives, advisory roles, `semantic_review_lens`, success definitions, failure modes, evidence requirements, promotion suggestions, waiver candidates, reconsideration triggers, semantic review criteria, repair candidates, supersession candidates, learning-hint refs, input refs, review criteria, output-contract hints, and bounded `execution_contract` candidates. Runtime still decides promotion and waiver: identity, capability, output type, dependency, visibility, and required-state mismatches are rejected, and only accepted proposal fields are projected into downstream branch records and output obligations.

`candidate_graph` and `promotion_review` are the general contract layer. They normalize output candidates, output obligations, workload tasks, context/reference/memory candidates, and rejected workload proposals into one visible lifecycle: possibility → relevance → promoted contract → runtime work → review → freeze. A candidate is not owed work. Only a promotion decision can create an obligation, active context/reference, repair contract, evidence branch, or continuation. Reserved, omitted, and stale candidates carry `reconsiderable=true`: they may still inform planning as advisory possibility space, but they remain non-executable until a later promotion. Waivers, rejections, and supersessions stay visible with reasons so they do not look like silent success or hidden failure.

A modality cue that is only reserved, negated, or inferred remains orientation,
not authority. Without a promoted current-turn obligation it cannot create an
executable branch, materialization request, or Closure-repair contract.

`decision_contract.block_resolution_reflex` is the global movement/read-model for the principle "the solution to a block is the block's own resolution." It summarizes open obligations, blocked obligations, reconsiderable candidates, waivers, supersessions, repair candidates, semantic-review candidates, and related workload-task state as advisory signals. It does not execute work; it tells Ghost and Closure where the right-sized verified transition should be considered: continue, repair dependency, repair branch contract, wait, clarify, waive with evidence, supersede with replacement truth, or freeze truthfully.

For marked external downstream execution, provider output whose first content is
`BLOCKED:` is a typed branch block signal. Runtime preserves the reason as
blocked lifecycle and surface truth, but excludes the text from artifact
acceptance, direct materialization, fulfillment, and Late Fill content. The
block may orient a right-sized repair; it is not the requested output.

`decision_contract.active_reconsideration_review` is the active version of that reflex. It converts signals into reviewable advisory decisions such as promotion relevance review, waiver evidence review, supersession truth review, repair contract review, semantic quality review, or continuation/repair review. It should be inspected when a formerly reserved idea should become relevant, when an old obligation may now be waived or superseded, or when a branch is blocked but retry would only repeat the same failure.

`decision_contract.semantic_quality_review` is the quality knob. It makes semantic criteria explicit as pending review contracts instead of letting artifact existence or generated prose imply success. Use it for checks like "voice sounds serious", "layout actually works", "caption matches the image", or "final answer respects the intended tone". Runtime-checkable criteria such as dependency evidence or no root-prompt replay should stay deterministic Closure checks first.

`decision_contract.semantic_review_lens_review` is the review-posture knob. It gives each relevant task, contract, attention frame, or semantic review a non-authoritative lens from `ollmo_g/semantic_roles/`, such as possibility expander, structural planner, materializer, evidence reasoner, quality reviewer, risk sentinel, simplifier, repairer, transition committer, or integrator. The lens states `success_definition`, `evidence_requirements`, and `failure_modes` so the model asks the right question for the branch. It must not be treated as public `role`, routing authority, or proof of fulfillment.

`decision_contract.semantic_role_orientation_review` is the global role-orientation knob. `GHOST.md` remains the constitution; `ollmo_g/semantic_roles/` is the managed role/lens library underneath it. Bounded `ghost_mode` values still exist at the API edge for compatibility, but they are immediately translated into `semantic_role_profile` and advisory orientation frames instead of direct planner or routing behavior. Use this knob when a response needs the right posture: possibility expansion, doubt/evidence, materialization, quality review, risk, simplification, repair, integration, or commitment. The branch-local `semantic_review_lens`, execution contract, runtime evidence, and Closure truth always win.

Branch-level semantic review is available everywhere but demand-gated. A fulfilled-looking branch with explicit `semantic_review_criteria` or non-deterministic qualitative criteria may promote a `branch_semantic_review` check and a schedulable `semantic_review` branch. Plain deterministic `review_criteria` do not trigger that by themselves. The reviewer receives only the branch contract, branch output/evidence, dependencies, criteria, and current intent as bounded context. A passed `semantic_review_verdict` clears the branch's semantic gate; failed, uncertain, or unparseable verdicts keep that exact branch pending/blocked with repair/manual-review guidance. Branch review checks the local note; global semantic closure still checks the whole phrase afterward.

`decision_contract.recursive_cycle_review` is the depth-n knob. It reports every workload task through the same prepare -> gather evidence -> execute -> verify -> repair/freeze cycle without imposing a fixed depth cap. If a subtask skips evidence gathering, restarts the root prompt, or freezes before verification, inspect this review and the matching workload task contract.

Scale movement is the zoom knob across these reviews. It has two axes. Shallow/deep is semantic depth: surface wording, intent, meaning, quality, contradiction, learning, and aspiration/doubt/commitment. Coarse/fine is structural granularity: whole turn, candidate set, branch, payload, dependency, artifact evidence, and review criterion.

Discovery should be allowed to move shallow -> deep when surface truth is insufficient and coarse -> fine when relevance demands decomposition. Review should be allowed to move deep -> shallow when enough truth exists for a practical transition and fine -> coarse when branch evidence must be checked against the whole current intent. If Ollmo gets stuck in a local branch, collapses to chat, overthinks after enough evidence, or never returns to whole-turn closure, inspect `recursive_cycle_review`, `semantic_decision_review`, `controlled_attention_review`, and `global_semantic_closure_review` together before adding a modality-specific rule.

`decision_contract.aspiration_review` is the great-faith knob. It keeps possibility, solution ambition, and non-minimal planning visible when the graph risks becoming too small. It can surface `expand_candidate_space`, `raise_solution_bar`, `review_underplanned_graph`, `preserve_possibility_space`, or `avoid_minimal_collapse`, but it never creates owed work by itself.

`decision_contract.commitment_review` is the great-courage knob. It prevents indefinite pending/review drift by proposing the right-sized sufficient transition, such as `continue_branch_local_work`, `repair_dependency_chain`, `repair_branch_contract`, `clarify`, `waive_with_evidence`, `supersede_with_replacement_truth`, or `truthful_freeze_after_review`. It is not minimalism and not force-completion; Runtime, Contracts, and Closure still decide truth.

`decision_contract.semantic_decision_review` is the brain-loop knob. It turns active reconsideration, semantic quality, recursive cycle state, evidence refs, confidence, and accepted-learning orientation into advisory next-transition proposals. It may recommend actions such as `repair_dependency_chain`, `repair_branch_contract`, `semantic_review`, `waive_with_evidence`, `supersede_with_replacement_truth`, `keep_reserved`, or `truthful_freeze`, but those are proposal-only until Runtime/Contracts/Closure promote or confirm them.

`decision_contract.controlled_attention_review` is the controlled model-attention knob. It turns reconsideration decisions, semantic quality contracts, recursive subtask state, semantic decision proposals, and accepted-learning hints into scoped attention frames. Each frame asks one bounded question, names the branch/task/candidate/review target, lists allowed transitions, carries evidence refs, and repeats the non-authority boundary. Ghost should answer the frame, not replay the root prompt or create unpromoted work. Runtime/Contracts/Closure still decide promotion, execution, waiver, supersession, cancellation, fulfillment, and freeze.

`runtime.request_phase_graph.intent_obligations` is the generic promise/obligation readout for the current turn. It normalizes broad asks into text artifacts, media artifacts, evidence branches, dependency bindings, navigation promises, and structural checks, then projects safe current-turn dependency obligations into the graph before execution when the evidence is strong. Local generated image assets before HTML consumers are executable dependency obligations; shared CSS and navigation promises remain visible binding obligations unless runtime evidence later proves a concrete missing edge. Accepted learning, provider/degraded/cache/liveness diagnostics, and advisory-only movement surfaces do not create entries here by themselves.

`runtime.graph_closure_review.intent_graph_adequacy.intent_lens_review` applies the same attention, commitment, and aspiration movement to current user intent. Attention checks whether material intent has a visible promise ledger or a concrete structural gap. Commitment checks whether promised producer/consumer bindings and dependency order are executable or repairable. Aspiration checks explicit semantic-fit promises, such as text matching generated images and local artifacts forming one coherent set, and promotes whole-turn semantic review evidence instead of direct graph mutation.

`runtime.graph_closure_review.global_semantic_closure_review` is the whole-turn semantic closure knob. It sits beside structural `intent_graph_adequacy`: adequacy asks whether the graph has enough promoted output/capability obligations and whether the intent-obligation ledger is represented by graph shape/dependencies, while global semantic closure asks whether fulfilled local branches actually fit the whole current intent together. If local obligations are still open, it waits. If local truth is complete but subjective fit remains unproven, Closure may promote a bounded `global_semantic_closure` check and schedulable `semantic_review` chat branch. That branch must return a structured `semantic_review_verdict` with `verdict=passed|failed|uncertain`, criterion results, evidence refs, defects, confidence, and a recommended transition. Review completion alone is not truth: Closure may freeze only on a passed verdict. Failed, uncertain, or unparseable verdicts remain pending/blocked and feed repair, manual review, waiver, supersession, or reconsideration.

`runtime.graph_closure_review.surface_state` is the UI/surface projection. It groups runtime truth into categories such as open, blocked, reconsiderable, waived, superseded, repair pending, semantic review pending, advisory movement, and completed. The frontend should render this projection instead of inferring state from assistant prose or stale output slots. Graph repair should treat fulfilled-contract/surface-state reconciliation as actionable only when the surface carries blocked, repair-pending, semantic-review-pending, dependency, artifact, or promoted owed-work evidence; advisory-only pending movement remains visible but non-repair.

`runtime.request_phase_graph.redraw_scope_ladder_review` is the intent-aligned scope classifier. Runtime attaches it after Closure and before graph repair/rebase lifecycle decisions. It orders possible movement as reserved slot/candidate fill, additive repair, binding/dependency repair, duplicate artifact-ref identity repair, partial subtree rebase, then full successor rebase. The selected scope is anchored to the current Intent Contract and current Runtime/Closure evidence. Graph repair proposals may carry it as `redraw_scope_orientation`, and graph rebase proposals may preserve bounded scope fields such as `scope_root_ids` and `preserve_outside_scope`, but the review does not validate or authorize patches/rebases by itself. `apply_enforced` consumes the scope review as one policy gate only. Accepted learning, advisory roles, degraded/provider/cache/liveness signals, frontend state, monitor-only summaries, and UI labels can orient or diagnose only; they cannot become scope authority.

Duplicate artifact refs are both scope evidence and final-output hygiene. Proven aliases may be canonicalized into one final output/artifact projection while preserving `alias_artifact_refs` and `alias_metadata`. Conflicting refs remain `repair_needed` with `final_projection_blocked` instead of being silently projected as success.

`OLLMO_GRAPH_REPAIR_AUTONOMY` is the graph patch rollout knob. When the environment variable is absent, the product default is `apply_enforced`; when `OLLMO_APPLY_ENFORCED_POLICY` is absent, its product default is `safe_v1`. This pairing turns on only the default-deny, allowlisted additive path. `safe_v1` can apply the narrow classes `safe_additive_missing_branch`, `safe_additive_dependency_repair`, `safe_additive_artifact_binding_repair`, and `duplicate_artifact_alias_canonicalization` after validation, safe-additive risk classification for safe-additive classes, current runtime evidence, redraw-scope alignment, idempotency, lineage/audit recording, and forbidden-evidence checks pass. Set `OLLMO_GRAPH_REPAIR_AUTONOMY=off` for the full diagnostics-only rollback, or `OLLMO_APPLY_ENFORCED_POLICY=off` to keep enforced proposals non-applying. Invalid autonomy or enforced-policy values fail closed to `off` and remain visible in developer diagnostics; absent values are reported with `source: product_default`, not as operator configuration.

The other autonomy levels remain available for deliberate operation. `shadow` records validated lifecycle records without mutation. `stage` records staged patches without exposing them as Late Fill work. `apply_safe` applies only validated safe additive classes. `apply_reviewed` requires the concrete graph repair review to carry accepted runtime/operator `graph_patch_authorization`, `allowed_autonomy` containing `apply_reviewed`, and non-empty evidence refs; the environment knob alone is insufficient. Every applied patch records `graph_patch_lifecycle` and `applied_graph_patches` with validation, evidence refs, idempotency keys, graph digests, and outcome. For a pre-freeze application, response runtime reconciles newly applied executable branches into the current closure gap and Late Fill state in the same response turn, then recomputes Closure against the patched graph so an open branch cannot coexist with stale `fulfilled` truth. A branch whose repair policy, block state, or execution contract remains non-executable stays unscheduled and visible in reconciliation diagnostics. Late Fill reads the current `artifact_payload.runtime.request_phase_graph` before any older `route_payload.route_runtime.request_phase_graph`, so a stale route graph cannot erase the applied work.

Terminal/frozen response frames remain immutable. A newly allowed patch there records a blocked parent lifecycle plus a deduplicated `successor_reopen_requests[]` candidate with parent-frame lineage and the would-have-applied graph. The production terminal owner now consumes that bounded continuation: it independently revalidates current graph-repair autonomy, enforced policy when autonomy is `apply_enforced`, exact parent frame, patch application, graph digests, exact owed branch set, depth budget, and the no-root-replay contract; persists one `graph_patch_reopen_successor` frame under the same response id; then schedules only those branches through existing Late Fill after releasing the terminal owner's current response claim. Queue/running/completed/failed truth remains synchronized across Late Fill, graph execution, request, and developer-diagnostic projections, and replay of an already consumed successor key is inert. Explicit `OLLMO_GRAPH_REPAIR_AUTONOMY=off` blocks every transition. Explicit `OLLMO_APPLY_ENFORCED_POLICY=off` blocks the product-default `apply_enforced` transition, but does not override explicitly selected `apply_safe`. Accepted learning remains `soft_hint_only`: it may orient proposal priority and outcome calibration, but cannot validate a current patch or satisfy the enforced-policy gate. Degraded/cache/liveness and backend-family route-health warnings remain preference, cooldown, or retry diagnostics unless hard runtime truth or explicit operator disablement proves otherwise.

`OLLMO_GRAPH_REBASE_AUTONOMY` governs the partial-subtree and full-successor rungs at the upper end of the same redraw scope ladder. The knob is a separate authority boundary, not a separate architecture layer: it prevents the autonomously active, default-deny lower additive repair rungs from silently climbing to broad redraw. Its absent-environment product default is `shadow`; `shadow` is the non-executing mode of those upper rungs, not another rung or a new shadow layer. When the variable is absent, normal startup does not synthesize or export it, so diagnostics retain `source: product_default, configured: false`, while an explicitly inherited operator value remains visible as operator configuration. Runtime may synthesize a proposal only from a concrete backend-built candidate graph after current Closure and the scope ladder are recomputed, active Late Fill has settled, and no smaller reserved/additive/binding/identity scope is eligible. When active Late Fill consumed the earlier bounded comparison context, terminal materialization deterministically derives a fresh candidate from final request/route/runtime truth. Shadow validates the digest, meaningful diff, dependency and graph-wide semantic preservation, derived-topology consistency, candidate-bookkeeping exclusion, and partial-scope containment, then records `shadow_no_mutation`; it creates no branch, obligation, staged graph, successor, or scheduler work. Ghost feedback items, accepted learning, and advisory/degraded/provider surfaces cannot create rebase authority. Set `OLLMO_GRAPH_REBASE_AUTONOMY=off` for the immediate diagnostics-only rollback; explicit or fail-closed `off` blocks both `stage` and `authorize_partial` before trusted transition truth or a successor frame can be written. Evidence-only `adjudicate` remains available because it is not a rebase transition. Invalid values also fail closed to `off`.

`GET /api/graph_rebase/readiness` is the canonical read-only rollout report over hydrated response-frame evidence and trusted operator records. It reports separate `shadow_to_stage`, `partial_stage_to_apply_reviewed`, and full-shadow-only gates; reading the report has `runtime_effect: none` and never flips a knob or grants authority. The product default remains `shadow` while evidence is insufficient and the gates are not green. Promotion uses one explicit per-response endpoint, `POST /api/responses/<response_id>/graph_rebase/operator`, in strict order: `adjudicate -> stage -> authorize_partial`. `adjudicate` records a trusted classification such as `useful_proposal`, `false_positive`, `false_negative`, or `needs_investigation`. `stage` requires the prior accepted adjudication and writes both durable registry and runtime stage truth with `staged_no_executable_mutation`. `authorize_partial` requires that exact review/stage chain, a matching runtime stage, and the green `partial_stage_to_apply_reviewed` gate. Every action is compare-and-swap bound to the finalized response/frame sequence, proposal id, base and candidate graph digests, and `partial_subtree_rebase`; wildcard, stale, request-inline, model-inline, candidate-inline, or full-rebase authorization fails closed.

The current `ollmo-graph-rebase-safe-partial-v1` rollout policy makes those gates concrete. `shadow_to_stage` requires at least 20 settled candidate opportunities across 5 workload families, at least 5 settled `not_proposed` cases, 3 qualifying partial proposals, 3 useful partial adjudications, and 2 deterministic partial replay confirmations. `partial_stage_to_apply_reviewed` requires 3 paired partial stages, 3 useful partial adjudications, 2 replay confirmations, and 3 local execution-contract proofs. Both gates tolerate zero false-positive adjudications, zero false-negative adjudications, zero open investigations, and zero unresolved critical safety findings. Full execution is false. These values describe current policy v1; the endpoint report and `ollmo_services/graph_rebase_rollout.py` remain canonical if a later policy version changes them.

Mutating operator actions are unavailable unless both `OLLMO_GRAPH_REBASE_OPERATOR_TOKEN` (at least 32 characters) and an exact `OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY` were explicitly configured at process start. The request must authenticate with that token through `Authorization: Bearer ...` or `X-Ollmo-Graph-Rebase-Operator-Token` and repeat the configured identity in `X-Ollmo-Graph-Rebase-Operator`. Startup exposes these values only to the control-plane process; canonical and internal alias names are unexported or removed from every child-process environment and are never written to frames or the registry. A useful-proposal adjudication automatically replays deterministic Runtime validation and records `replay_verified` only when the frozen and replayed reviews match; caller-supplied replay claims are not evidence. The operator owner combines the full durable state with the bounded observation projection only when both identify the exact same latest frame; the latter supplies successor-safe inherited root truth. A `false_negative` may bind a settled candidate that produced no proposal by using exact frame/base/candidate CAS plus `expected_proposal_id=no_formal_proposal`; it cannot stage or authorize work. The record remains append-only history. A later exact replay-verified `useful_proposal` adjudication may reference it through `resolves_record_id`; the registry accepts only one same-class resolution, and readiness gates count unresolved rather than historical false negatives. Readiness counts a partial stage only when the exact trusted stage and durable runtime stage form a complete matching pair, and reports either half as an orphan.

Use `./ollmo ctl graph-rebase readiness` and `./ollmo ctl graph-rebase inspect <response_id>` for passive inspection. The mutation commands are `adjudicate`, `stage`, and `authorize-partial`; they derive the exact current response/frame/proposal/base/candidate/class bindings rather than accepting hand-copied digests. They never recover or start the control plane. Credentialed calls are restricted to the exact loopback control plane, bypass environment proxies, reject redirects, require visible single-line credentials, and refuse a returned response identity that differs from the requested one. `stage` and `authorize-partial` fail locally when the current Runtime proof or gate is not eligible; authorization also requires `--execute` because a successful call immediately queues branch-local work. Use `--adjudication false_negative --rebase-class partial_subtree_rebase` only after human inspection of a settled no-proposal candidate; the CLI never makes that judgment automatically.

After `authorize_partial`, Runtime joins authorization only from the trusted registry, independently rebuilds the local execution-contract proof, revalidates proposal/review/lifecycle/digests/scope/current root, and may consume the exact partial `successor_rebase_requests[]` record. It appends a `graph_rebase_partial_successor` frame under the same response id before scheduling only the exact owed branch-local work through existing Late Fill. The frozen parent frame and its graph remain unchanged. Root request, assistant output, root/current-phase prompt, and other broad inherited fallback are forbidden across primary payloads, batch prompts, phase summaries, stage directions, instructions, and review criteria on both phase and downstream records; missing current root truth, a mismatched guard, missing branch-local payload, or missing dependency bindings block execution. Idempotency, exact parent lineage, bounded depth, and durable replay make duplicate delivery inert. `successor_rebase_requests[]` therefore remains audit/lineage truth by default, but is no longer universally non-executable: only this exact trusted partial path can consume one. Staged-only, stale, widened, untrusted, full, or gate-blocked records remain non-executable. Full successor rebase remains shadow/non-executable under safe partial v1, and `apply_enforced` does not bypass the explicit reviewed partial path.

Self-learning retention is a cleanup/archive truth knob. Before `clean` or `archive` removes response-frame ballast, `clean_repo_state.sh` collects sidecar refs reachable from active `state/self_learning/*.json` and `*.jsonl`, writes `state/self_learning/retention_manifest.json`, and copies retained sidecars to `state/self_learning/retained_sidecars/`. Dry-runs print retained and missing counts. Missing sidecars are diagnostics in the manifest/report; they must not disappear silently or promote learning authority.

Late-fill surface state is branch-control aware. A visible queued or running branch can be cancelled from the UI through `/api/responses/<response_id>/late_fill/control`; the surface should then show the terminal branch state instead of an endless spinner. Cancellation is not a claim that the provider stopped instantly; it is runtime truth that queued work is skipped and late-arriving results from that branch are stale.

Promoted obligations close through runtime truth, not prose. Valid terminal/working states include `fulfilled`, `pending`, `blocked`, `waived`, and `superseded`. Use `waived` when owed work was explicitly released; use `superseded` when owed work was replaced or made no longer relevant by newer evidence. A superseded obligation should not trigger retry or late fill execution, but its replacement edge and reason should remain inspectable.

Late-fill execution contracts are the handoff knob between graph authority and executor mechanics. If a branch executes the root prompt again, writes the wrong artifact, blocks another branch, or completes without matching the planned obligation, inspect whether `execution_contract`, `workload_task_ref`, and `output_obligation_ref` survived `late_fill -> effective_data -> infer_payload -> infer_result -> fill_results`.

The semantic execution gate is the branch-start relevance knob. Before a pending late fill branch reaches a backend, Runtime checks whether that branch is still executable or has become `cancelled`, `waived`, or `superseded`. User or runtime branch controls are stored as `late_fill.branch_controls` plus terminal branch records under `late_fill.cancelled_branches`. Queued terminal branches are skipped; results that return after a branch was stopped are treated as stale and are not merged as fulfilled artifacts. This is the execution-scope form of "the solution to a block is the block's own resolution": stop, waive, or supersede the exact branch instead of forcing the rest of the graph through the wrong work.

Reserved materialization candidates are diagnostic possibilities, not executable branches. A prompt such as "sketch a poster idea, but do not generate an image yet" or "keep this image only as a reserved option" may record an image candidate in `candidate_graph`, but it must not appear in executable `downstream_capabilities`, `downstream_phase_ids`, `continuation_required`, or Closure Repair output until a later promotion explicitly asks for it. If the same turn contains other promoted work, such as "do not generate an image yet, but read the idea as audio", only the promoted audio branch is executable. This is not a ban on planning input: reserved candidates can still be considered when forming or repairing the graph.

Artifact-fulfillment negation is different from materialization defer. Wording such as "a prompt is not the image artifact", "a script is not the audio artifact", or "placeholder prose does not count as the HTML file" strengthens the open artifact obligation; it must not reserve or suppress the requested materialization branch. Use reserved candidates only for explicit stop/defer language such as "do not generate it yet", "hold audio for later", or "keep this output as a reserved option".

Late-fill retry semantics must distinguish backend retries from graph/dependency repair. If a branch fails because its required artifact is missing, such as STT without an audio artifact or vision analysis without an image artifact, recovery is `repair_dependency_chain`; retrying the same branch is expected to repeat the failure. If the dependency artifact or evidence already exists but is not bound to the blocked branch, recovery may be `rebind_dependency_evidence`.

Closure Repair now classifies open graph checks with a concrete recovery action. Transient executor failures stay `retry_same_branch` or `retry_excluding_instance`; unavailable capability is `start_compatible_instance`; missing upstream evidence is `repair_dependency_chain`; existing-but-unbound dependency evidence is `rebind_dependency_evidence`; an under-specified branch handoff is `repair_branch_contract`; a graph that planned too little work is `rebuild_from_promoted_obligations`; and qualitative checks can surface as `semantic_review`. Inspect `runtime.graph_closure_review.checks[].repair_action`, `ghost_repair_feedback.items[].repair_action`, and `late_fill.pending_branches[].repair_action` before pressing a retry button or changing Ghost prompts.

Closure-promoted text artifact repairs must carry the defect evidence into the branch prompt. Syntax repair names the target artifact, deterministic syntax issues, and current saved file content. Link rebind names unresolved placeholders or guessed links, concrete runtime artifacts, and current saved target content. HTML/CSS selector-binding repair names the CSS target, linked HTML artifact, HTML classes missing CSS selectors, CSS class selectors unused by HTML, and current saved HTML/CSS content. These payloads are patch/replace authority for the bounded target file only; they are not permission to replay the root prompt, redesign, translate, or rewrite unrelated artifact structure.

Closure Review also reads `decision_contract` as a unifying guidance layer. Matched promotion suggestions, waiver candidates, repair candidates, semantic-review candidates, supersession candidates, evidence requirements, reconsideration triggers, advisory roles, semantic review lenses, success definitions, failure modes, and learning-hint refs can be attached to closure checks and repair feedback. This does not execute candidates by itself: reconsideration stays advisory, promotion suggestions need promotion review, waiver candidates need explicit release evidence, supersession candidates need Closure truth, and repair still starts only from an open promoted check.

Repair feedback is loop-shaped but conservative. `ghost_repair_feedback.repair_loop` records next actions and promoted repair contracts. Closure Review may promote bounded `repair_rebuild_contracts` from its own open checks; Ghost suggestions alone remain advisory. `auto_execute=true` means at least one Closure-promoted repair or required artifact materialization retry branch is schedulable through late fill, not that Ghost can execute arbitrary candidates. Auto-executable repair/materialization retry should stay open while its bounded automatic budget remains; `OLLMO_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS` sets the default budget and per-contract metadata can narrow or extend it within the safe cap. `auto_execute=false` does not mean Ollmo must stop: inspect `repair_work_available`, `materialization_blocked`, `blocked_scope`, `blocked_prerequisite`, and `needs_external_input` to distinguish blocked backend materialization from available contract/dependency repair work.

Required TTS has a stricter materialization-recovery budget than the general repair loop. Any typed retryable same-branch materialization failure—including backend timeout/unavailability and runtime-owned `TTS_AUDIO_INTEGRITY_REPAIR_REQUIRED` evidence—may enter policy `tts_bounded_materialization_recovery_v1` exactly once, so the total TTS materialization attempt count is at most two regardless of `OLLMO_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS`. Dependency, branch-contract, underplanning, external-input, optional-audio, and non-retryable failures do not enter this path. A rejected WAV remains `materialization_blocked` and diagnostic-only while the repair branch runs. Response-local exclusions stay authoritative: a ready non-excluded alternative is selected first and recorded as `nonexcluded_alternative_for_tts_recovery`. Only the exact required-audio recovery contract may reuse an excluded instance, and only after live runtime truth proves it is the sole usable compatible TTS instance; that route records `excluded_reuse_for_single_tts_recovery`, the policy/trigger, prior error, and candidate diagnostics. Branch normalization must retain the retry counter, policy, attempt limit, prior error, complete defect list, and integrity evidence so persisted/resumed recovery cannot silently lose its authority. Every retry goes through the unchanged physical integrity, source binding, semantic evidence, artifact, Closure, and publication gates.

The bounded Reconsideration/Rebuild loop is now operational for Closure-promoted work. A missing or under-covered promoted obligation becomes a `repair_rebuild_contract`, the response runtime patches `runtime.request_phase_graph` with the promoted repair branch, and late fill then either schedules that branch or blocks it at the dependency/branch-contract gate. This is still one bounded repair round, not unbounded autonomous graph recursion.

The late fill scheduler gates non-executable repair actions before backend execution. `repair_dependency_chain` and `rebind_dependency_evidence` branches need dependency evidence first; `repair_branch_contract` branches need a bounded execution contract first. If those prerequisites are absent, the target materialization branch is blocked with structured recovery state instead of calling the backend again, but repair work may remain available. If those prerequisites are present, the block flag is resolved by its own evidence instead of remaining sticky. `rebuild_from_promoted_obligations` may still create executable materialization branches once the missing obligation has been promoted.

Closure Review now also carries workload-task review criteria into output-obligation checks. Deterministic criteria such as `uses_dependency_evidence` and `does_not_restart_root_request` can keep a fulfilled-looking text branch open when the branch lacks late fill dependency evidence. Explicit `semantic_review_criteria` or non-deterministic qualitative criteria may be marked with `semantic_review_required`, `semantic_review_authority=advisory`, and `semantic_review_criteria` instead of being silently treated as proven by artifact existence.

Branch semantic review is the local form of that rule. Inspect `checks[].check_kind=branch_semantic_review`, `branch_semantic_review_status`, and `semantic_review_verdict` when a branch exists but its tone, layout usefulness, declared role, or other semantic fit is not yet proven. It should not run for every branch; it runs when the branch should be reviewed.

Global semantic closure is the answer to "branch fulfilled but the whole feels wrong." Inspect `runtime.graph_closure_review.global_semantic_closure_review`, `global_semantic_closure_review.semantic_review_verdict`, `checks[].check_kind=global_semantic_closure`, and the resulting `ghost_repair_feedback.items[]` before adding modality-specific heuristics. The expected flow is local truth first, then structured whole-intent semantic verdict, then repair/waive/supersede/freeze from runtime evidence.

Useful controls for this layer are: candidate extraction mode, candidate count policy, decomposition-depth reporting, promotion threshold/policy, promotion authority, waiver authority, supersession authority, reconsideration policy, repair round policy, and accepted learning authority. The default authority split remains conservative: Ghost may propose, Runtime/Review promotes, and accepted learning remains `soft_hint_only`.

Remaining 30-second timeout values should be read carefully: in the UI and backend transports they are lower bounds or per-poll HTTP safety timeouts, not maximum work waits. A true maximum wait should be named as a budget knob, documented with a reason, and sized for local long-running work.

---

### 5. UX / surface (perception knob)

This is the **user experience knob**.

It controls:

- wording
- layout
- status messages
- visibility of structure
- overall product feel
- ordered mixed-output presentation

If something feels:

- confusing
- too technical
- cluttered
- immersion-breaking
- inconsistent with what the runtime actually did
- text appears above an image/audio artifact even though the graph says it belongs after that artifact

Then this is the knob to inspect.

For mixed outputs, the surface should prefer canonical `outputs` order when text and artifact outputs are both present. The fallback remains plain message text plus appended artifacts for older or simpler responses.

Interactive HTML preview packages are derived UX state, not response artifacts
or Closure truth. When a fulfilled HTML output lives directly in a shared flat
artifact bucket and no persistent bundle copy exists, `View` may create a
response-bound package under the control-plane process's private temporary
directory. It uses only exact canonical public output paths and explicit
response-owned dependencies, never global artifact-ref hydration or arbitrary
sibling discovery. The package is not written to `artifacts/bundles/`, chat
history, response frames, or the artifact registry.

The non-UI storage controls are:

- `OLLMO_HTML_PREVIEW_TTL_SECONDS`: absolute package lifetime; default `1800`, clamped to `60..86400`. Asset access does not extend it.
- `OLLMO_HTML_PREVIEW_MAX_PACKAGES`: process-local package count; default `24`, clamped to `1..128`.
- `OLLMO_HTML_PREVIEW_MAX_FILES`: maximum files copied into one package; default `64`, clamped to `1..512`.
- `OLLMO_HTML_PREVIEW_MAX_PACKAGE_BYTES`: maximum bytes in one package; default `268435456` (256 MiB), clamped to 1 MiB..2 GiB.
- `OLLMO_HTML_PREVIEW_MAX_TOTAL_BYTES`: maximum bytes across the process-local package cache; default `536870912` (512 MiB), clamped to 1 MiB..4 GiB.

The least-recently-used unleased package is removed when count or total-byte
bounds require space. Expiry is creation-bound rather than sliding, and process
exit removes the temporary root. These are preview resource controls only; they
do not truncate canonical artifacts or persistent bundle truth.

---

## Quick Debug Flow

When testing, use this sequence:

### Step 1
Did Ghost understand the request correctly?

- If no, or if stale history became new intent → inspect the **Ghost knob**

### Step 2
Was the right work created?

- If no → inspect **Ghost structure/graph derivation**

### Step 3
Was the right thing executed?

- If no → inspect the **Resolver knob**

### Step 4
Was the output itself correct?

- If no → inspect the **Provider knob**

### Step 5
Was it stored, continued, or replayed correctly?

- If no → inspect the **Runtime/Substrate knob**

### Step 5b
Did the pre-freeze closure review correctly classify complete, pending, blocked, partial, or failed graph obligations?

- If no → inspect the **Runtime/Substrate knob** first. If the review emits repair feedback, Ghost patches the graph; then inspect the **Resolver knob** if continuation selection or dependency readiness was wrong.

### Step 5c
Did a later branch receive the right bounded input from an earlier branch?

- If no → inspect **Runtime/Substrate** for evidence handoff first, then **Ghost** if the graph failed to declare the dependency.

### Step 6
Did it feel right to the user?

- If no → inspect the **UX/Surface knob**

---

## Why this matters

Ollmo now has clear separation of concerns.

Instead of everything being mixed together, the system is separated into:

- thinking
- executing
- producing
- storing
- showing

Closure Review belongs to the truth layer: it audits runtime evidence against obligations and can send a small repair package back to Ghost. It does not replace Ghost as the graph-building intelligence, and it does not guarantee semantic perfection by itself.

This makes debugging more precise and design decisions cleaner.

---

## Example

### Scenario
User asks:

> "Make 2 images and an explanation"

Observed result:

- 2 images generated correctly
- explanation feels weak or off

### Diagnosis

- Ghost → probably fine (the right branches were created)
- Resolver → probably fine (the right providers were chosen)
- Provider → likely the issue (the text model output was weak)

### Conclusion

Fix the **Provider knob**, not Ghost.

---

## Summary

Ollmo currently has 5 clean control knobs:

1. Ghost
2. Resolver
3. Providers
4. Runtime / Substrate
5. UX / Surface

Every real problem should map primarily to one of these.

That is the operational clarity to keep during testing.
