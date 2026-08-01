# KNOBS CHEATSHEET

Fast mental model during testing.

General posture:
- no hidden hard caps on canonical policy, graph, artifact, or contract sources
- if a limit is technically required, expose it as a named budget/safety knob and document why

---

## 1. Ghost (intent/graph)

Controls:
- interpretation
- structure
- branching
- current-turn intent bracket
- workload graph decomposition

Default posture:
- Ghost-first graph formation
- heuristics as shadow/guardrails only
- accepted learning as `soft_hint_only`; enabling an accepted snapshot does not grant execution or patch authority
- workload graph v1 is derived from existing phases
- `workload_task_proposals` are expected for multi-task/dependency work when Ghost can add useful branch-local semantics, but they remain advisory until runtime validates them
- `decision_contract` is Ghost's read-only brain boundary: it summarizes candidates, promotions, obligations, proposal coverage, reconsideration, promotion suggestions, waiver candidates, supersession, repair, graph-repair proposals, semantic review, and accepted-learning orientation without becoming execution truth
- `decision_contract.semantic_planning_contract` is the advisory brain rubric: planning cycle, proposal requirements, non-authority boundaries, and current proposal obligations

Hard brain knobs found in runtime:
- `developer_flags.planner_timeout_ms`: explicit resolver compatibility budget only; roles/learning/retry state do not change it.
- `semantic_role_profile.loop.max_passes=1`: one semantic loop pass.
- `semantic_role_profile.loop.critic_passes=1`: normal runtime reviewer pass; preview mode sets it to `0`.
- `semantic_role_profile.runtime_orientation.runtime_effect=none`: semantic roles are wording/review orientation only.

Brain read-model surfaces:
- `candidate_graph`: possible work, including reserved materialization candidates.
- `promotion_review`: only promoted candidates become owed work.
- `decision_contract.block_resolution_reflex`: open/blocked/reserved/waived/superseded/repair signals.
- `decision_contract.active_reconsideration_review`: reviewable next-state decisions.
- `decision_contract.semantic_quality_review`: subjective quality as explicit review work.
- `decision_contract.semantic_review_lens_review`: which advisory lens asks the branch question.
- `decision_contract.semantic_decision_review`: advisory next-transition proposals.
- `decision_contract.graph_repair_proposals`: advisory topology repair candidates; inspect these when basic intent was not represented in the graph, but apply nothing until runtime validation accepts a bounded patch.
- `decision_contract.controlled_attention_review`: scoped model-attention frames; use this when branches replay the root prompt or start unpromoted work.
- `runtime.graph_closure_review.global_semantic_closure_review`: whole-turn fit after local branches.
- `runtime.graph_closure_review.surface_state`: frontend-facing truth projection.
- `state/self_learning/report.json.shadow_hints`: offline shadow/eval suggestions with `authority: shadow` and `runtime_effect: none`; useful for diagnosis, never live route/graph/closure authority.
- `state/self_learning/report.json.retention`: retention integrity for learning evidence sidecars; missing counts are diagnostics, not a reason to promote learning authority.
- `state/self_learning/retention_manifest.json` and `state/self_learning/retained_sidecars/`: response-frame sidecars reachable from active self-learning files and copied before clean/archive removes response-frame ballast.
- `state/self_learning/accepted_policy_snapshot.json`: reviewed accepted learnings. When explicitly enabled, `/api/ghost`, route context, and decision contracts see soft runtime hints with concrete hint text and `case_kinds`; they remain orientation only.

Fast diagnosis:
- wrong possible/reserved work -> inspect `candidate_graph`
- wrong executable work -> inspect `promotion_review`
- blind retry or hidden block -> inspect `block_resolution_reflex`
- stale root prompt in a branch -> inspect `controlled_attention_review` and branch `execution_contract`
- quality claimed too early -> inspect `semantic_quality_review` and `semantic_review_verdict`
- response freezes too early -> inspect `global_semantic_closure_review` and `surface_state`
- basic current-turn intent not met -> inspect `intent_graph_adequacy`, `graph_repair_proposals`, `graph_repair_reviews`, and Closure repair feedback
- learning would have warned but must not mutate behavior -> inspect `state/self_learning/report.json.shadow_hints`, `report.json.retention`, accepted snapshot `enabled`/`runtime_effect`, and `accepted_learning_hints.hints[].case_kinds`

Fix here if:
- request is misunderstood
- system collapses to simple chat too early
- old history or artifacts steer a fresh turn
- heuristic cues appear to override Ghost instead of only auditing/guarding it
- work is not split into the needed task-level branches
- semantic task labels/objectives are absent before validation

---

## 2. Resolver (executor)

Controls:
- which model runs
- execution order
- `depends_on` branch sequencing
- fallback
- same-level branch parallelism

Fix here if:
- wrong model used
- wrong branch executed
- dependent branch ran too early
- independent sibling branches did not run as siblings
- sibling outputs overload or underuse available local instances

Knob:
- `OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS` caps branch materialization workers; per-instance locks still serialize calls into the same local instance.
- Normal local startup currently exports this as `4` when unset; explicit operator environment values still win.

---

## 3. Providers (models)

Controls:
- output quality
- speed vs quality

Fix here if:
- output is weak
- style is off
- output appears truncated by an Ollmo-side generation cap

Default posture:
- Do not send `max_tokens` for normal chat/responses unless explicitly requested.
- Keep internal resolver/translation helper budgets high enough for branch plans and long transcripts.
- `planner_timeout_ms` is an explicit resolver compatibility key and may run up to the local long-running job budget; 30s values should be minimums/polling safeguards, not hidden work caps.

---

## 4. Runtime (state)

Controls:
- outputs
- slots
- replay
- continuation
- workload task lifecycle
- workload proposal validation
- candidate graph / promotion review
- graph closure review
- selective evidence handoff between phases/branches
- repair feedback from Closure Review back to Ghost

Fix here if:
- duplication
- missing outputs
- broken continuation
- final response claims work runtime truth did not produce
- branch consumed stale or hidden text instead of declared evidence
- explicit text-file requests produce only inline chat instead of a saved `.txt`, `.md`, `.html`, `.css`, `.js`, or similar artifact
- TTS/image branches consume the whole previous answer instead of the focused script, prompt, caption, or other declared evidence segment
- final text after generated image/audio consumes only artifact paths instead of vision/transcript evidence

Text-file artifacts are explicit-only and current-turn scoped: ordinary chat stays inline, while file/artifact/export/save/download cues, concrete text-like filenames/extensions, or edits to a selected text-like source artifact in the latest user turn should produce runtime-backed `saved_text_path` evidence. Old transcript text alone must not satisfy an ambiguous `this` source. Multiple text artifacts use `saved_text_artifacts` plus canonical `artifacts`; `saved_text_path` stays the first artifact for compatibility.

README is treated as a markdown text artifact when requested alongside other files. Reserved materialization language such as “nur als reservierte Option” stays a candidate, not an executable branch. A missing input artifact failure such as STT without audio is `repair_dependency_chain`, not `retry_same_branch`; late fill now gates that branch before backend execution if the dependency evidence is still absent.

Structured text artifact wrappers are payload envelopes. If a model returns `output_obligations[].content`, save that `content` as the file and not the surrounding router/control JSON.

Closure Review audits runtime evidence against obligations. If the graph is missing work or the basic current-turn intent was not represented, it may return a bounded repair package. Ghost and `decision_contract.graph_repair_proposals` can describe the candidate repair, but response runtime must validate the proposal before applying an additive graph patch. Resolver schedules only the resulting validated/promoted branches, and Runtime stores the evidence.

Generated-media follow-ups should branch through analysis first: image comparisons/captions use Vision evidence before final chat, and exact spoken-text checks use STT evidence before final chat.

Vision evidence branches should analyze the actual attached/generated image using a focused prompt; the original user request is only bounded intent context, not the branch task itself.

Mixed media joins should be dependency-specific: TTS and image generation can be siblings from prepared text, STT depends on audio, Vision depends on image, and final Chat depends on the evidence branches.

The workload graph is the general per-task contract. Runtime should verify each task's declared inputs, output contract, recursive lifecycle, and review criteria instead of relying on the full initial prompt when a later branch needs focused evidence. Each branch repeats prepare -> gather evidence -> execute -> verify -> repair/freeze.

Closure Review carries workload-task review criteria into output-obligation checks. A text branch that exists but lacks required generated evidence can stay open with `repair_dependency_chain` instead of freezing by text existence alone.

If Ghost proposes task annotations, `workload_proposal_review` decides what becomes graph truth. Ghost may propose rich branch-local semantics, advisory roles, evidence requirements, promotion suggestions, waiver candidates, reconsideration triggers, semantic review criteria, repair candidates, supersession candidates, learning hint refs, and bounded `execution_contract` details for existing workload tasks. Structural changes such as new dependencies, changed capability, changed output type, changed visibility, changed required state, or mismatched contract identity are rejected.

`workload_proposal_review.coverage` shows whether semantic task proposal coverage is missing, partial, complete, or not required.

Accepted workload proposal detail should project into downstream branch records so late fill receives the richer promoted branch contract instead of reconstructing intent from the root prompt.

Qualitative review criteria are now visible work. Deterministic failures can become `repair_dependency_chain` or `repair_branch_contract`; subjective checks become `semantic_review_required` with advisory authority until a promoted semantic verifier exists.

Repair feedback is loop-shaped and Closure-promoted when runtime truth proves open work. `ghost_repair_feedback.repair_rebuild_contracts` names bounded promoted repair/rebuild contracts; `repair_loop.auto_execute` means at least one promoted contract is schedulable through late fill, not that Ghost can run arbitrary candidates.

Auto-executable repair is "open until exhausted", not one blind retry. `OLLMO_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS` caps the default total automatic attempts, including the first attempt; runtime default is `6`, values are clamped to a safe upper bound, and branch/repair-contract metadata such as `auto_executable_repair_max_attempts` can set a more specific budget. This is a non-UI policy knob.

The bounded Reconsideration/Rebuild loop validates Closure-promoted repair branches, then patches `runtime.request_phase_graph` additively before late fill scheduling. Dependency-repair and branch-contract-repair branches still block at their gates until evidence or contract shape exists. Patch reviews are visible under graph repair diagnostics such as `graph_repair_reviews` and developer diagnostics.

Closure Repair now uses `decision_contract` as guidance for unity across fluid state. Matched decision-contract repair candidates can fill a missing repair action on an already-open closure check; promotion suggestions, waiver candidates, semantic review, and supersession candidates remain visible guidance; reconsiderable candidates remain non-executable until promoted.

Provider-family failures are route-health or preference/cooldown evidence by default, not graph repair patches or broad provider bans. Only hard runtime evidence or explicit operator disablement should disable a provider or instance.

The backend runtime evidence bridge can synthesize graph-repair proposals into response truth. It maps concrete evidence such as `materialization_contract_unmet`, terminal pending branches, duplicate artifact refs, fake or missing artifact dependencies, and fulfilled-contract/surface-state mismatches into proposal-only repairs only when the surface classifier finds actionable blocked, repair-pending, semantic-review-pending, dependency, artifact, or promoted owed-work evidence. Advisory-only pending movement from `controlled_attention_review`, `aspiration_review`, `commitment_review`, or reconsideration remains visible but must not become `reconcile_surface_state_or_reopen_contract` by itself. Inspect `runtime.request_phase_graph.graph_repair_proposals[]`, `runtime.request_phase_graph.graph_repair_reviews[]`, and `runtime.developer_diagnostics.surface_repair_actionability` first. Monitor `learning_healing.graph_repair` output is an observer summary for Codex-side run watching, not product truth.

Validated graph patches are controlled by `OLLMO_GRAPH_REPAIR_AUTONOMY`. Absent environment means product-default `apply_enforced`; absent `OLLMO_APPLY_ENFORCED_POLICY` means product-default `safe_v1`. That pairing is default-deny and applies only the narrow allowlisted missing-branch, dependency, artifact-binding, and proven alias classes after every validation/evidence/scope/idempotency gate passes. Explicit `OLLMO_GRAPH_REPAIR_AUTONOMY=off` is the full diagnostics-only rollback; explicit enforced-policy `off` disables enforced application; invalid values fail closed to `off`. `shadow` and `stage` never mutate executable work, while `apply_safe` and explicitly authorized `apply_reviewed` retain their narrower operational meanings. A pre-freeze applied patch is reconciled into the same-turn closure gap and Late Fill state, Closure is recomputed against the patched graph, and only branches whose repair/execution policy is now executable are scheduled; unresolved blocks stay unscheduled diagnostics. The current artifact payload graph takes precedence over a stale route graph. A terminal/frozen parent is not mutated: a newly applied safe additive `successor_reopen_requests[]` candidate is revalidated at the terminal sink, appended as one same-response `graph_patch_reopen_successor` frame, and its exact owed branches are scheduled through existing Late Fill without root-prompt replay. Set graph-repair autonomy to `off` for the global stop; setting enforced policy to `off` stops only `apply_enforced`, including the product-default path, and does not override explicit `apply_safe`.

Graph rebase/redraw stays on the upper partial-subtree and full-successor rungs of the same scope ladder under its dedicated authority contract; it is not a separate layer. Lower bounded additive redraw stays autonomously active under graph-repair `safe_v1`. The product-default upper-rung mode is non-executable `shadow`, which is not an additional rung and remains the default while the canonical evidence gates are not green. `GET /api/graph_rebase/readiness` is read-only. Promotion uses `POST /api/responses/<response_id>/graph_rebase/operator` in exact order: `adjudicate -> stage -> authorize_partial`. Stage is durable audit-only. `authorize_partial` requires a trusted registry chain, a green partial gate, exact CAS bindings for response/frame/proposal/base/candidate/class, matching durable full/observation frame truth, the current root digest, and a branch-local execution proof; it appends one same-response successor and schedules only that partial work. Root text in any executable prompt carrier and parent mutation are forbidden. Explicit/fail-closed `OLLMO_GRAPH_REBASE_AUTONOMY=off` blocks stage and authorization before trusted transition truth is written. Full successor rebase stays shadow/non-executable under safe partial v1. `successor_rebase_requests[]` is normally lineage/audit truth; only the exact trusted partial request is consumable. No-op, dangling refs, digest mismatch, lost dependencies/failure visibility, semantic drift, bookkeeping smuggling, scope escape, stale lineage, missing current root truth, or missing local payload fail closed. Ghost feedback, Learning-only, degraded/cache/liveness, provider-family, frontend, advisory, and monitor-only evidence cannot authorize rebase.

`OLLMO_GRAPH_REBASE_OPERATOR_TOKEN` and `OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY` have no permissive defaults. Unless startup explicitly sets a token of at least 32 characters plus one exact identity, and a request presents both matching values, the mutating operator endpoint is unavailable. The control-plane credentials are never persisted or inherited by any child process; canonical and internal alias names are stripped. Runtime produces replay confirmation; callers cannot claim it. No-proposal false negatives use `expected_proposal_id=no_formal_proposal` as diagnostic evidence only. A later same-class replay-verified useful proposal may append one exact `resolves_record_id` link; historical false negatives remain visible while only unresolved ones block promotion. Promotion counts only complete trusted/runtime stage pairs.

The frontend should render current work from late fill branch state before stale slot state. Completed or blocked branches suppress older matching queued slots.

The general candidate contract layer is `candidate_graph` plus `promotion_review`. It keeps possible outputs, tasks, context, references, memory, evidence, repairs, continuations, learning hints, and rejected proposals in one lifecycle. Candidates can stay reserved, omitted, or stale without creating work and should remain `reconsiderable` when later evidence changes. Promoted candidates become contracts. Waived, rejected, or superseded records must carry reasons.

`decision_contract.block_resolution_reflex` is the quick place to inspect whether Ollmo is applying the global block-resolution rule. It should show signals for open, blocked, reserved/stale, waived, superseded, repair, and semantic-review state. If the system retries blindly, rewrites intent, or hides a block behind prose, inspect this reflex plus Closure Review before changing a single modality heuristic.

`decision_contract.active_reconsideration_review` is the next place to inspect. It translates reflex signals into reviewable decisions: promote relevance, preserve reserved state, review waiver evidence, review supersession truth, repair the branch contract, repair the dependency chain, run semantic quality review, or freeze truthfully.

`decision_contract.semantic_quality_review` is the quality-review work queue. If an output exists but the user-facing criterion is qualitative, such as serious audio, usable layout, evidence-grounded comparison, or correct visual tone, this review should remain pending until a semantic verifier or explicit runtime review resolves it.

`decision_contract.semantic_review_lens_review` is the review-lens report. It should show which internal posture applies to a task or check from the global `ollmo_g/semantic_roles/` library: possibility expander, structural planner, materializer, evidence reasoner, integrator, quality reviewer, risk sentinel, simplifier, repairer, or transition committer. Use it when a branch asks the wrong question, such as reviewing audio without STT evidence, comparing only one image, materializing prose instead of a file, or retrying a dependency block. It is advisory only and must not revive public `role` routing.

`decision_contract.semantic_role_orientation_review` is where global semantic role posture lands. `repair`, `worker`, `explorer`, and `improviser` are legacy API aliases only; they should appear as translated semantic roles, not as a separate Deliberation system. Semantic roles must not create planner timeout bonuses, fast-path changes, branching authority, payload authority, or truth. If a response feels pulled by an old role, the right fix is to make the relevant role lens, contract, evidence, or Closure state stronger.

`checks[].check_kind=branch_semantic_review` is the branch-local verifier. It should appear only when a fulfilled-looking branch has qualitative criteria that deserve model attention. Passed `semantic_review_verdict` clears the branch gate; failed, uncertain, or unparseable verdicts keep that branch open/blocked with a transition recommendation.

`decision_contract.recursive_cycle_review` is the mini-cycle report. It should show that every subtask, branch, and repair follows the same movement as the whole request: prepare -> gather evidence -> execute -> verify -> repair/freeze.

`decision_contract.aspiration_review` is the "great faith" report. It should show when the graph may be too small, when the solution bar should be raised, or when possibility should remain visible instead of collapsing to minimal chat. It is advisory only.

`decision_contract.commitment_review` is the "great courage" report. It should show the right-sized sufficient next transition when enough evidence exists and the system risks drifting in pending/review state. It is advisory only and must not force completion or collapse to a too-small action.

`decision_contract.semantic_decision_review` is the advisory brain-loop output. If Ghost sees the right state but still proposes weak repairs, wrong waivers, missing quality review, or shallow follow-up decisions, inspect the proposals, confidence, evidence refs, and learning orientation here.

`decision_contract.controlled_attention_review` is the "space between the tones" surface. It should contain bounded attention frames for the exact candidate, branch, task, quality review, recursive cycle, semantic proposal, or accepted-learning hint that deserves model attention now. If later branches replay the full root prompt, start unpromoted work, or ignore a stop/waiver/supersession signal, inspect these frames and their allowed transitions.

`runtime.graph_closure_review.global_semantic_closure_review` is the whole-turn check. Use it when each local branch looks fulfilled but the final result still fails the bigger request: wrong comparison, wrong evidence use, wrong tone, missing synthesis, or outputs that do not belong together. It should wait while local branches are still open, then promote a bounded `semantic_review` branch when whole-intent fit is the remaining unproven work. The completed review must carry `semantic_review_verdict`: `passed` permits truthful freeze, while `failed`, `uncertain`, or unparseable output keeps Closure open with a recommended transition.

`runtime.graph_closure_review.surface_state` is the frontend-facing projection. UI should follow the same principle: the solution to a visible block is to show and resolve the block, not to keep a stale spinner, hide the branch, or imply success from prose.

Reserved materialization candidates should be visible but non-executable. If a turn says not to generate an image yet, the image may remain in `candidate_graph`, but it should not set executable `downstream_capabilities`, `continuation_required`, or Closure Repair work. It may still inform planning as advisory possibility space. Other promoted branches in the same turn still run.

Accepted learning may orient graph repair when repeated frames show basic intent was not met, but it remains soft evidence. It cannot validate or apply a graph patch without current Closure/runtime truth.

Artifact-fulfillment negation is not a defer. If a turn says a prompt/script/placeholder is not the image, audio, HTML, CSS, or file artifact, keep the artifact obligation executable and require runtime artifact truth. Only "not yet", "for later", "hold/reserve", or equivalent stop language should create reserved non-executable materialization candidates.

Promoted obligations can close as `fulfilled`, `waived`, or `superseded`. Use `superseded` when a newer branch or newer runtime truth replaces older owed work; it should not remain pending or trigger a retry.

Mental trace:
- possibility
- relevance
- promoted contract
- runtime work
- review
- freeze

---

## 5. UX (surface)

Controls:
- wording
- layout
- clarity
- runtime-state visibility
- block/review/repair status display

Fix here if:
- confusing
- too technical
- breaks immersion
- mixed text/image/audio/file outputs render in the wrong order
- a blocked branch looks like endless loading
- waived, superseded, reconsiderable, repair-pending, or semantic-review-pending state is invisible

---

## One Rule

Every issue belongs to one knob first.
