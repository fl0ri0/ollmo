# PATTERNS

## ✅ Good Patterns

### 1. Substrate-first execution

state → structure → decision → execution → state

---

### 2. Outputs as first-class objects

- outputs are actionable
- outputs can be refined
- outputs persist

---

### 3. Branch identity preserved

- multiple outputs remain distinct
- no collapsing into capability buckets

---

### 4. Continuation over time

- replay works
- pending work exists
- outputs evolve

---

### 5. Current-turn intent bracket

- fresh turns use the current user request as the intent anchor
- older history is admitted only as explicit reference or continuation context
- old tool calls and old artifacts do not become new intent by recency alone

---

### 6. Pre-freeze graph closure

- compare request phase graph requirements with runtime truth
- continue only existing open obligations
- freeze complete, pending, blocked, partial, or failed state explicitly
- linked artifact sets close only when surviving generated links resolve to saved local artifacts

---

### 7. External systems call Ollmo

external → Ollmo → outputs

Not:

external → providers directly

Route preview and route selection are not lifecycle actions. They must not start models or feed runtime load back into routing.

---

### 8. Tools as runtime capabilities

- tools are invoked by Ollmo
- results come back into substrate

---

### 9. Model as interchangeable provider

- local or remote
- cheap or premium
- same contract
- provider execution receives the current request plus only context that Ollmo
  promoted as relevant for that turn
- files and artifacts cross the provider boundary only when they are explicit
  current-turn inputs; a remote provider additionally requires the declared
  consent and data scope
- runtime records the concrete input handoff; provider prose is not proof of
  which context or files were supplied
- external execution of an already-shaped branch begins with
  `[OLLMO_DOWNSTREAM_EXECUTION_V1]`; Ollmo remains supervisor while the provider
  executes only `<ollmo_bounded_task>` under the supplied
  `<ollmo_promoted_context>`, without recursively invoking Ollmo or widening the
  task
- Ghost planning itself is a separate Ollmo-internal role: it does not invoke
  the Ollmo companion skill and receives runtime manifest, model, and capability
  orientation directly from Ollmo. When Ghost subsequently resolves an
  already-shaped branch to an external target, that target call is downstream
  execution and does receive the marker

---

### 10. Contract-preserving block resolution

- a blocked obligation stays visible
- runtime evidence explains why it is blocked
- downstream provider output beginning with `BLOCKED:` is branch block truth;
  preserve its reason in runtime state, but never accept that text as artifact
  or materialization content or as fulfillment evidence
- the next transition is the right-sized verified continuation, repair, explicit waiver, supersession, clarification, or truthful freeze
- freeze captures the truthful state, including remaining blockage if needed
- UI renders the block as runtime state, not as a stale spinner or false success
- the same pattern applies to reconsiderable, waived, superseded, repair-pending, and semantic-review-pending state
- if the basic current-turn intent was not represented in the graph, keep that as repairable graph state and validate an additive graph patch instead of pretending the simpler graph was enough

---

### 11. Candidate before obligation

- possible outputs may exist as candidates or reserved vacancies
- candidates do not count as pending obligations
- unpromoted/reserved candidates may inform planning, but remain non-executable until promoted
- a modality cue that is only reserved, negated, or inferred cannot create
  executable work or Closure-repair work without a promoted current-turn
  obligation
- promotion records explain why a candidate became owed work
- waiver records explain why owed work was released
- supersession records explain why owed work was replaced by newer runtime truth

---

### 12. Workload task lifecycle

- every promoted task has declared inputs, dependencies, output contract, visibility, lifecycle, and review criteria
- the same prepare -> gather evidence -> execute/materialize -> verify -> repair or freeze cadence applies inside each branch
- decomposition depth is descriptive and recursive; it should not be treated as a fixed small hard limit
- graph-level truth and branch-level truth should agree before final freeze
- deterministic review criteria such as dependency-evidence use can keep a fulfilled-looking branch open for repair when branch-local evidence is absent
- semantic review criteria are separate demand-gated checks; they are not implied by every `review_criteria` item

---

### 13. Branch-local handoff

- downstream branches consume focused `content_payload`, `artifact_prompt`, `stage_direction`, dependencies, artifact refs, and evidence
- later branches do not replay the full root prompt as their task
- multilingual handoff labels such as `Bild-Prompt:` or `Poster prompt:` should preserve the intended payload instead of mixing in later review text
- Closure-promoted text repairs are bounded target-file work: the repair payload names the target artifact, the concrete defect, the relevant runtime artifact evidence, and the current saved file content needed to patch or replace only that target
- Syntax repair, link rebind, and HTML/CSS selector-binding repair should fix the evidenced connection or local syntax defect; they should not redesign, translate, or regenerate unrelated artifact structure
- Target-bound text repairs are path-bound evidence: when a repair names `target_path`, only that concrete saved path can fulfill the branch or final materialization contract. A new sibling artifact such as `index.html` cannot satisfy a `styles.css` repair and must stay `repair_needed` for learning/healing.

---

### 14. Evidence before post-artifact text

- media-dependent text runs after the media exists
- image review/comparison uses image artifact evidence or vision enrichment
- audio confirmation uses the generated audio plus dependency-bound transcript evidence when exact spoken meaning is owed; neither surface is interchangeable with the other
- every generated PCM WAV records `tts_audio_integrity_evidence` for effective active-signal duration, silence ratio, trailing silence, source binding, and file identity; HTTP 200, a WAV header, and non-zero bytes do not fulfill audio when this evidence fails or is unavailable
- generated TTS confirmation records `tts_semantic_source` and binds the actual STT transcript through `tts_stt_semantic_evidence` to the exact digest-bound text handed to that one TTS producer; a non-empty audio file or unbound transcript alone is not semantic fulfillment. Invalid audio remains diagnostic artifact truth but is not promoted as a public fulfilled output
- final text branches consume dependency evidence, not hypothetical descriptions

---

### 14b. Execution gate before backend work

- queued branches are rechecked before dispatch
- user/runtime branch controls can cancel, waive, or supersede exact late fill branches
- stale results that return after cancellation are not merged as fulfilled work
- UI shows terminal branch state instead of leaving an old spinner active

---

### 14c. Controlled attention between steps

- model attention is focused through scoped attention frames, not by replaying the whole root prompt
- each frame names the candidate, branch, task, review, or learning hint that should be considered now
- each frame lists allowed transitions and evidence refs
- attention is advisory only; Runtime, Contracts, Closure, or the user still decide execution truth
- the same attention pattern applies before work starts, between dependency edges, during repair, and before truthful freeze

---

### 14d. Validated Graph Repair

- Ghost, Closure, decision contracts, and accepted learning may propose graph repair, but proposals are non-executable until reviewed
- accepted learning can orient repair when repeated frames show basic intent was not met, but it cannot validate or apply a patch
- current response-frame, Closure, late fill, and artifact evidence can be mapped by backend runtime into proposal-only repairs for known failure classes before validation; monitor reports are optional observer evidence only
- fulfilled-contract/surface-state reconciliation is repairable only when surface actionability shows blocked, repair-pending, semantic-review-pending, dependency, artifact, or promoted owed-work evidence
- advisory-only pending state from `controlled_attention_review`, `aspiration_review`, `commitment_review`, or reconsideration remains visible but must not create graph repair by itself
- validation rejects broad provider disablement requests, reserved/deferred intent conflicts, missing runtime evidence, and capability/output mismatches
- accepted patches are additive and idempotent; they add missing branches, phases, obligations, or dependencies without replacing the whole graph
- generic current-turn promises are first normalized into `request_phase_graph.intent_obligations`; graph adequacy checks this ledger against branches, phases, and executable dependency edges before Closure can call the graph structurally complete
- local generated/embedded image asset promises belong to graph adequacy before file execution: promote image branches first, then bind HTML/CSS text-artifact branches to those image phase dependencies instead of letting page files materialize against guessed paths
- if the initial graph planned a producer and consumer in parallel or in the wrong order, Runtime may propose and validate a safe additive missing-dependency-edge patch before execution; after frozen/terminal state, use target-bound repair or successor/reopen truth instead of rewriting the parent frame
- `graph_patch_lifecycle` records make staged/applied truth inspectable: `shadow` and `stage` are diagnostic only, while `apply_safe` can apply only safe additive classes and records idempotency/digest/outcome evidence
- terminal/frozen parent frames are not patched in place; newly applied safe additive terminal repairs create `successor_reopen_requests[]` with parent-frame lineage, then the terminal owner revalidates and appends one same-response successor frame whose exact owed branches execute through normal Late Fill
- terminal repair continuation is append-only and branch-local: preserve the frozen parent bytes, bind `graph_patch_reopen_successor` to its exact frame sequence, deduplicate the patch/execution key, release the current response claim before rescheduling, and never replay the root prompt
- review-required classes need explicit `graph_patch_authorization` on the concrete proposal review before `apply_reviewed` may apply them; an autonomy knob alone is not runtime review
- repaired branches still run through normal dependency gates, late fill, review, waiver, supersession, and closure
- self-learning records patch outcomes and terminal successor/reopen outcomes, but accepted learning remains soft orientation and cannot validate a patch by itself
- active self-learning evidence keeps a retention root set; cleanup/archive copies reachable response-frame sidecars into learning-owned retained sidecars or reports missing refs visibly

---

### 14e. Intent-Aligned Redraw Scope Ladder

- Runtime attaches `redraw_scope_ladder_review` after Closure and before graph repair/rebase lifecycle decisions
- the ladder chooses the smallest current-intent-aligned scope: reserved slot/candidate fill, additive repair, binding/dependency repair, duplicate artifact-ref identity repair, partial subtree rebase, then full successor rebase
- the Intent Contract anchors the decision; accepted learning, advisory roles, degraded/provider/cache/liveness evidence, monitor-only summaries, frontend state, and UI labels remain orientation/diagnostics only
- graph repair proposals may carry `redraw_scope_orientation`, but validators still require current runtime evidence and safe patch classes
- graph rebase proposals may carry bounded scope fields such as `scope_root_ids`, `scope_phase_ids`, `scope_artifact_refs`, and `preserve_outside_scope`; Runtime still owns diff, preservation proof, authorization, lifecycle, and successor truth
- duplicate artifact refs are final-output hygiene as well as scope evidence: proven aliases collapse to one projection with alias metadata, while conflicting refs stay `repair_needed` and block final projection
- self-learning records redraw-scope outcomes under `redraw_scope_policy`, but accepted learning remains `soft_hint_only`

---

### 14f. Reviewed Graph Rebase

- the ladder escalates from additive graph repair to the higher-risk partial-subtree and full-successor rebase rungs when a broader candidate graph is needed
- bounded additive redraw remains autonomously available on the lower graph-repair rungs; `shadow` is only the non-mutating rollout mode of the upper rebase rungs, not a new architectural layer
- Ghost, semantic review, or an operator may propose `ollmo.graph_rebase_proposal`, but Runtime must compute the diff and preservation proof instead of trusting model prose
- preservation proof must keep required intent obligations, output obligations, artifact refs, dependencies, review duties, target-bound repair contracts, failure visibility, and frozen parent lineage visible
- the upper rebase rungs use their own authority boundary, `OLLMO_GRAPH_REBASE_AUTONOMY=off|shadow|stage|apply_reviewed|apply_enforced`, so additive repair authority cannot silently climb the ladder; absent configuration defaults to non-executable `shadow`, explicit `off` blocks every rebase transition, and invalid values fail closed
- `GET /api/graph_rebase/readiness` is the canonical read-only evidence report; it does not promote, stage, authorize, or execute anything, and the product default remains `shadow` while its evidence gates are not green
- promotion is explicit and ordered through `POST /api/responses/<response_id>/graph_rebase/operator`: `adjudicate` records a trusted useful/false-positive/false-negative/investigation judgment, `stage` durably records the exact accepted proposal with `staged_no_executable_mutation`, and `authorize_partial` requires that trusted chain plus the green partial-promotion gate
- the mutating operator writer is dormant without both an explicit startup token and configured exact operator identity; requests must match both, the credentials stay control-plane-only and are stripped from every child-process environment, Runtime creates deterministic replay confirmation, and only exact registry/runtime stage pairs count toward promotion
- no-proposal false negatives remain append-only response-bound evidence and never become execution authority; only one later same-class, replay-verified useful-proposal adjudication may reference the old record with `resolves_record_id`, after which history reports both records while gates count only unresolved false negatives
- every operator transition is CAS-bound to the exact response id, finalized frame id/sequence, proposal id, base graph digest, candidate graph digest, and requested rebase class; request/model/candidate-supplied authorization dictionaries are not trusted operator authority
- an authorized partial successor is appended under the same response id, preserves the frozen parent frame, carries its own exact branch-local payload/dependency contract, and schedules only the owed partial branches through normal Late Fill; full-state and bounded-observation truth must bind the same current frame, and inherited root request, assistant output, or root/current-phase text in any executable prompt carrier is forbidden
- `successor_rebase_requests[]` is not a generic queue: untrusted, stale, staged-only, full, or otherwise blocked records remain lineage/audit truth, while only the exact registry-trusted, gate-approved partial request may be consumed by the safe partial v1 successor owner
- full successor rebase remains shadow/non-executable under safe partial v1; accepted learning records outcomes as soft calibration only and cannot satisfy readiness, preservation, registry, authorization, or execution gates

---

### 15. Artifact dossiers as read-side truth

- artifact continuity should prefer durable `artifact_ref` identity and dossier evidence over raw copied paths
- existing enrichments are reusable evidence when they satisfy the current branch contract
- fresh analysis is promoted only when the current task needs evidence that is missing, stale, or insufficient
- dossiers are diagnostic/audit evidence, not the public deliverable list; frontend artifact rendering, response lookup/recovery, and bundle roots should project from fulfilled canonical `outputs` / `artifacts.output`, while raw repair/intermediate records remain available inside dossiers for review and healing
- public web artifacts include the local dependency closure of the final public entrypoint: if a fulfilled `index.html`/CSS/JS file links to saved local images, CSS, scripts, audio, or other bundleable files, those linked files remain public dependencies even when older output refs were missing, duplicated, or alias-poor
- text/document artifacts whose ref is shaped like `artifact:text_generated_image_...` are generated-image repair/misbinding evidence, not normal public HTML output, even if a stale slot says `fulfilled`

---

### 16. Structured artifact payloads

- text/file artifacts persist the payload content, not control-plane wrappers
- preparation-phase output crosses one canonical acceptance boundary before persistence, Closure, streaming completion, or downstream materialization; a planner/control envelope may yield exactly one safe `content_payload`, gets at most one same-phase repair attempt, and otherwise remains repair-needed rather than becoming artifact or TTS content
- `output_obligations[].content` is an artifact payload when the user requested a file
- ambiguous deictic requests such as "make this HTML" need a source before persistence

---

### 17. Explicit lifecycle start source

- backend/API route selection reuses eligible running instances by default
- explicit frontend play/start is a user lifecycle action and can send `start_source=frontend_button`
- `force_start` belongs only to deliberate duplicate starts, not Ghost, route preview, late fill, or backend automatic paths

---

## ❌ Anti-Patterns

### 1. Text as truth

- treating assistant output as canonical
- reconstructing state from text

---

### 2. Capability-first thinking

Instead of:

"this is an image task → call image model"

Use:

"this branch requires image output"

---

### 3. Agents talking to each other

- free-form agent conversations
- no shared substrate

---

### 4. Hidden side effects

- file changes not tracked
- external actions not recorded

---

### 5. Multiple intent owners

- resolver becoming a second brain
- conflicting decisions

---

### 6. Stale history as intent

- previous image/audio/file work forces the next turn into the same modality
- old artifacts are reused without explicit reference
- compressed history becomes hidden instruction

---

### 7. Reviewer as truth

- optional critique/reviewer paths replace graph closure review
- model commentary decides whether an artifact exists
- fulfillment is inferred from prose instead of runtime state
- every branch or `review_criteria` item becomes `semantic_review_required`

---

### 8. External control over providers

- exposing provider selection externally
- config-driven provider injection

---

### 9. Forced completion

- rewriting intent because execution is blocked
- hiding a blocked obligation behind successful prose
- freezing a desired outcome instead of runtime truth

---

### 10. Candidate collapse

- treating every possible graph slot as owed output
- treating candidate/reserved space as a pending failure
- executing unpromoted candidates as if they were obligations

---

### 11. Full-prompt replay into branches

- feeding a later materializer the whole original multi-task request
- letting a final-review instruction become an image/audio prompt
- letting an image prompt become TTS spoken text

---

### 12. Evidence-free post-artifact prose

- writing visual reviews before the image exists
- comparing generated media from intended prompts instead of actual artifacts
- confirming spoken text from artifact existence or an unbound transcript instead of digest-bound producer-source plus dependent STT evidence

---

### 13. Wrapper-as-artifact persistence

- saving Ghost/router JSON as the user-facing file
- persisting `output_obligations` metadata instead of `output_obligations[].content`
- treating a model's artifact claim as proof that a file exists

---

### 14. Blind retry instead of repair

- retrying a branch whose required dependency artifact was never produced
- hiding dependency-chain failure behind a generic backend retry
- repairing by inventing new intent instead of promoting a grounded repair candidate

---

### 15. Hidden hard caps

- silently truncating canonical policy, graph, artifact, or contract sources
- treating an arbitrary limit as a control knob without naming it
- making local/open substrate behavior depend on invisible cutoffs

---

### 16. Route-driven starts

- starting models from Ghost route preview or route selection
- using `force_start` outside explicit user lifecycle actions
- treating load, busy, or cooldown feedback as authority when live process/port/backend truth says the instance is usable

---

### 17. Placeholder linked-artifact closure

- treating a placeholder, guessed path, stale link, or duplicate reused path as a completed linked artifact
- closing HTML/CSS/media output before every surviving generated dependency path resolves to a saved local artifact
