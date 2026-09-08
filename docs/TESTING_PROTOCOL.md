# TESTING PROTOCOL

Use this as the canonical test-routing and debugging guide for the affected
behavior. Start with focused checks, broaden for shared contracts or unresolved
risk, and run release validation when preparing a release. Once relevant checks
pass, repeat or broaden only for new changes, failures or unresolved concerns.
Prose-only edits ordinarily need static contract/link checks, not live inference.

For an explicitly authorized live adversarial sweep of the five intent/truth boundaries, run
`./ollmo self-attack`. See [SELF_ATTACK.md](SELF_ATTACK.md) for deterministic
oracles, discovered profiles, full truth capture, reduction and regression replay.

Do not overthink. Just map the issue.

---

## Quick Debug Flow

1. Did it understand the request?
   → Ghost

2. Did it create the right work?
   → Ghost structure + `candidate_graph` / `promotion_review`

3. Did it execute the right thing?
   → Resolver / branch-local workload task

3a. Did route preview or route selection start or force-start anything?
   → Start-source boundary / runtime liveness guard

3b. Did an external downstream executor receive one bounded task without
    re-entering Ollmo, while Ghost planning remained an internal Ollmo role?
   → External downstream execution boundary

4. Is the output itself good?
   → Provider

5. Is state / continuation correct?
   → Runtime / late fill / graph closure review

5a. Do final linked artifacts resolve to saved local dependency paths?
   → Runtime / artifact registry / terminal rebind / repair-needed closure

5b. Did a known terminal/closure failure produce no graph-repair proposal?
   → Backend runtime evidence bridge / `ollmo_services.graph_repair`

5c. Did `BLOCKED:` provider output become content or an artifact?
   → External-provider block projection / runtime truth gate

6. Does it feel right?
   → UX

---

## Short Mapping

- misunderstood → Ghost
- wrong branch / flow → Ghost, candidate graph, promotion review
- wrong execution → Resolver, branch-local payload, backend fabric
- recursive or widened external execution → downstream execution marker and bounded-task contract
- route-driven start → start-source guard, runtime liveness, model control
- bad output → Provider
- broken state → Runtime, response frame, artifact dossier
- unresolved generated links → artifact registry, terminal rebind, graph closure review
- counted image branch received a whole answer or selected data file → exact image-prompt cohort / branch-local handoff
- explicit image retry repeats `NO_COMPATIBLE_INSTANCE` while excluded providers are currently ready → exhausted-pool retry policy / live candidate truth
- a successful successor under the same response id has no new report → monitor frame identity / reference-export binding
- graph repair proposal missing despite runtime evidence → backend runtime evidence bridge, `graph_repair_proposals`, `graph_repair_reviews`
- graph repair patch staged/applied unexpectedly → `OLLMO_GRAPH_REPAIR_AUTONOMY`, `graph_patch_lifecycle`, `staged_graph_patches`, `applied_graph_patches`
- `BLOCKED:` provider text materialized as output content → external-provider block projection, artifact acceptance, late fill
- modality cue created work without a current-turn obligation → candidate graph, promotion review, Closure-repair authority
- bad feel → UX

## Current Failure Mapping

- possible work executed even though it was only reserved → promotion review
- a reserved, negated, or inferred modality cue creates executable or Closure-repair work without a promoted current-turn obligation → promotion/repair-authority regression
- required work missing from the graph → candidate extraction or graph closure repair
- an external branch-executor call lacks `[OLLMO_DOWNSTREAM_EXECUTION_V1]`, can recursively invoke Ollmo, widens `<ollmo_bounded_task>`, or applies the marker to Ghost planning → downstream execution-boundary regression
- downstream output beginning with `BLOCKED:` becomes artifact/materialization content, fulfillment, or Late Fill work instead of blocked runtime truth → external-provider block-projection regression
- later branch used the whole first answer → branch-local handoff
- two or more requested images run without the exact number of distinct branch-local prompts, an explicitly malformed count is treated as an absent single-image count, or a full website/code response becomes every image prompt → counted image-prompt contract regression
- an explicit image retry clears exclusion history, prefers an excluded provider over a ready non-excluded provider, reuses stale snapshot-only liveness, bypasses a missing prompt contract, or automatically retries again after failure → explicit image exhausted-pool policy regression
- a failed frame prevents a later same-response success frame from receiving its own monitor report, or reference export accepts a report for the wrong frame → frame-scoped observer/export regression
- image/audio follow-up text is hypothetical → missing evidence branch or artifact dossier
- TTS produces a non-empty but wrong recording and any non-empty STT transcript still closes the graph → TTS source-fidelity evidence gate regression
- TTS returns HTTP 200 plus a readable but silent, severely truncated, or mostly padded WAV and the audio slot still fulfills → output-side TTS integrity regression
- labelled/numbered TTS candidate speaks wrapper prose, transcript claims, code, or JSON → branch-local audio candidate extraction regression
- dependent STT runs without a digest-bound direct TTS producer result, or typed mismatch evidence disappears after frame normalization → dependency evidence fail-open/durability regression
- dependency artifact missing → `repair_dependency_chain`, not same-branch retry
- file artifact contains router JSON → text artifact payload extraction
- saved text syntax failure loses the target path/current bytes/issues or repeatedly regenerates the whole artifact → target-bound saved-text syntax recovery regression
- Ghost route preview starts a model → start-source policy regression
- duplicate, placeholder, or template-variable links such as `{{IMG_PATH_1}}` survive in final HTML/CSS/media → linked-artifact closure regression
- a multi-page site closes after generated images were appended as an unstyled detached stack to the first HTML file, or room/item images remain unbound from their semantic records → composed-site image-role/layout closure regression
- a composed-site target repair disappears because its existing file is mistaken for new write evidence, or a bundle leaves a retained-input path unresolved after selecting its newer authoritative file → target-bound handoff/bundle-authority regression
- optional generated-image `image_state_enrichment` disappears without `pending_existing`, `skipped`, or a suppression reason → image-state enrichment transparency regression
- `proposal_count=0` even though there is `materialization_contract_unmet`, terminal pending work, duplicate artifact refs, fake artifact refs, or actionable blocked/repair/semantic-review surface mismatch → graph-repair runtime evidence bridge regression
- `reconcile_surface_state_or_reopen_contract` appears for advisory-only pending `controlled_attention_review`, `aspiration_review`, `commitment_review`, or reconsideration state → graph-repair surface-actionability classifier regression
- `redraw_scope_ladder_review` skips a smaller current-intent scope, treats accepted learning/advisory/degraded/provider/cache/frontend/monitor/UI-label evidence as authority, or allows duplicate artifact refs to project as successful final output when refs conflict → intent-aligned redraw scope regression
- explicit text/media/local-artifact fit promises complete without `intent_lens_review` aspiration evidence or whole-turn semantic-review promotion → current-intent closure regression
- Intent Lens attention adds duplicate `rebuild_from_promoted_obligations` repair when a concrete `intent_graph_adequacy` or branch-contract repair already exists → duplicate repair-promotion regression
- `apply_safe` mutates a graph for review-required/forbidden classes, terminal frozen frames, degraded-only evidence, backend-family route-health diagnostics, accepted-learning-only proof, or advisory-only surfaces → graph patch lifecycle/autonomy regression
- `apply_reviewed` mutates review-required graph work without `graph_patch_authorization.status=accepted`, runtime/operator authority, `allowed_autonomy=["apply_reviewed"]`, and evidence refs → graph patch authorization regression
- `apply_enforced` fails to resolve an absent `OLLMO_APPLY_ENFORCED_POLICY` to product-default `safe_v1`, applies with explicit `off`/`audit`, applies a class outside safe-v1, skips safe-additive risk classification, redraw-scope/current-evidence/idempotency/forbidden-evidence gates, or treats accepted learning/degraded/provider/frontend/monitor-only evidence as authority → enforced policy regression
- invalid `OLLMO_GRAPH_REPAIR_AUTONOMY` silently falls back to `off` without `raw_value` and `invalid_value` diagnostics → graph patch autonomy diagnostics regression
- `shadow` or `stage` creates branches, obligations, dependency edges, or late fill work → graph patch lifecycle regression
- terminal `apply_safe`/allowed `apply_enforced` mutates the frozen parent, stops at an inert `successor_reopen_requests[]` candidate, widens beyond the exact applied branch set, replays the root prompt, loses same-response parent lineage, or schedules the same successor key twice → terminal successor/reopen execution regression
- graph rebase proposal applies without runtime-computed diff and preservation proof, drops required obligations/artifact refs/review duties/lineage, relies on learning-only/provider/degraded/advisory evidence, or mutates a parent graph directly → graph rebase preservation regression
- absent `OLLMO_GRAPH_REBASE_AUTONOMY` is not visible as product-default non-executable `shadow`, `shadow` is treated as a separate layer/rung, `stage` creates executable work, explicit rebase `off` does not block `stage` and `authorize_partial` while retaining evidence-only adjudication, lower bounded additive repair is accidentally disabled, or full successor rebase executes under safe partial v1 → graph rebase lifecycle/autonomy regression
- readiness reads mutate runtime state, insufficient evidence reports a green gate, operator actions work without both the configured token and matching configured identity, credentials reach any child process, durable full/observation projections do not bind the same latest frame, accept caller-authored replay truth, cannot record a response-bound no-proposal false negative, permanently block on a false negative after one exact same-class replay-verified resolution, allow unknown/non-useful/duplicate resolution links, count unpaired registry/runtime stages, skip `adjudicate -> stage -> authorize_partial`, accept stale/wildcard/non-CAS identities or inline authorization, execute a stage record, consume a non-partial/full request, mutate the frozen parent, lose atomic parent CAS, schedule before successor persistence, accept missing/drifted current root truth, replay the root prompt through any phase/downstream prompt carrier even after source relabeling, or duplicate a consumed successor → reviewed partial rebase rollout regression
- `clean`/`archive` preserves `state/self_learning/` while deleting the response-frame sidecars it references, without `retention_manifest.json`, retained copies, or missing-sidecar diagnostics → self-learning retention regression
- public prose rename touches only docs/operator text → glossary review plus active-doc search is enough
- literal compatibility key rename touches request/runtime/history/replay payloads such as `planner_timeout_ms`, `runtime.execution_planner`, or `execution_planner_deferred_follow_up` → compatibility migration regression; require aliases or dual-read/dual-write before changing writers
- Ghost, resolver, router, semantic-role, or injected-policy prompt wording changes → behavior-affecting prompt regression; require targeted route/resolver/Responses tests and live A/B checks when a local runtime is available

## Current Self-Healing Test Slices

For the external downstream execution boundary, run:

    .venv/bin/python -m pytest tests/test_codex_runtime_bridge.py -q

The expected shape is that only an actual external branch-executor call starts
with `[OLLMO_DOWNSTREAM_EXECUTION_V1]`, carries one
`<ollmo_bounded_task>` plus only promoted `<ollmo_promoted_context>`, and forbids
recursive Ollmo use or follow-up work. Ghost planning itself remains unmarked
because it is an Ollmo-internal runtime role, does not invoke the companion
skill, and receives manifest, model, and capability orientation from Ollmo. An
external target selected after that planning still receives the downstream
marker. A result beginning with `BLOCKED:` must project blocked lifecycle,
output, and surface truth, create no artifact, and skip materialization,
Closure, and Late Fill.

For labelled/count TTS extraction, output-side WAV integrity, and TTS-to-STT semantic evidence, run:

    .venv/bin/python -m pytest tests/test_tts_audio_integrity.py -q
    .venv/bin/python -m pytest tests/test_response_semantics_runtime.py -q -k "tts or speech_to_text or audio_variant or semantic_evidence or audio_integrity"
    .venv/bin/python -m pytest tests/test_fake_backend_e2e.py -q -k "tts_stt_and_vision or silent_tts"

The expected current shape is exact branch-local speakable payload selection, contiguous labelled candidate authority, exclusion of transcript/analysis/code/JSON siblings, durable exact final-prompt `tts_semantic_source`, deterministic source/file-bound PCM-WAV signal evidence, direct-producer-only `tts_stt_semantic_evidence`, harmless transcript normalization acceptance, and fail-closed silence/truncation/padding/malformed/missing/digest/binding handling. HTTP 200 or a non-empty WAV must not fulfill audio by itself. Expected text must never enter the STT request, downstream joins must stay unexecuted on mismatch, and a physically materialized wrong WAV remains diagnostic evidence rather than fulfillment.

For counted image handoff, explicit exhausted-pool retry, and frame-scoped recovery evidence, run:

    .venv/bin/python -m pytest tests/test_response_semantics_runtime.py -q -k "image_prompt or incomplete_image"
    .venv/bin/python -m pytest tests/test_responses_api.py -q -k "image and (retry or batch or branch_contract)"

For terminal link-rebind write evidence and branch settlement, run:

    .venv/bin/python -m pytest tests/test_responses_api.py -q -k "terminal_link_rebind or terminal_linked_artifact or terminal_materialization_contract"

Keep both the minimized single-branch case and the four-image inline-CSS Hive
case. Applied link-rebind evidence carries exact branch/phase identity only
when an already-owned saved artifact/result and that branch's bounded repair
independently select the actual written transformation. Same-target pending
branches, unrelated revisions, conflicting identities, and different dependency
selections must not inherit the write. A coalesced physical write may produce
separate exact-owner records only after each owner passes this check. Repeated
rebinding must neither rewrite unchanged bytes nor append evidence. Legacy
identity-free records remain readable diagnostics, never retroactively attributed
write authority. Ledger and UI projections preserve optional branch identity.

For composed multi-page image placement and bounded cohort repair, run:

    .venv/bin/python -m pytest tests/test_response_semantics_runtime.py -q -k "composed_site_image_closure or composed_page"
    .venv/bin/python -m pytest tests/test_responses_api.py -q -k "authoritative_composed_site_image_repair"
    .venv/bin/python -m pytest tests/test_response_artifact_bundles.py -q
    .venv/bin/python -m pytest tests/test_ollmo_run_monitor_projection.py tests/test_reference_run_export.py -q

The expected shape is that `batch_prompt_expected_count >= 2` requires the exact number of non-empty branch-local prompts and a valid slot selection. Shared preparation prose, full HTML/CSS/JSON answers, selected room data, root prompts, and heuristically focused fragments must be removed and exposed as `incomplete_image_prompt_batch` before routing. A user-triggered retry after `NO_COMPATIBLE_INSTANCE` keeps all exclusions, prefers any ready non-excluded alternative, and may reuse one excluded provider only under `explicit_image_excluded_pool_retry_v1` after fresh live truth; the attempt cannot auto-follow up. Missing contracts and refresh failures fail closed. A failed frame and a later successful frame under one response id each receive one append-only report, while export accepts only evidence matching the authoritative latest frame and leaves the source ledger byte-identical.

For accepted-learning and graph-repair changes, run:

    .venv/bin/python -m pytest tests/test_graph_repair_self_healing.py tests/test_self_learning.py -q

Use this when touching `ollmo_services/self_learning.py`, `ollmo_services/graph_repair.py`, `ollmo_services/self_learning_retention.py`, decision-contract learning/repair surfaces, or monitor learning/healing summaries. The expected current shape is that accepted learning remains soft orientation, backend runtime evidence can synthesize proposal-only graph repairs into response truth, advisory-only pending surfaces do not synthesize repair-needed graph work, actionable blocked/repair/semantic-review evidence remains repairable, monitor reviews are paired by `proposal_id` as observer summaries, validation rejects missing evidence, broad provider disablement requests, and accepted-learning-only proof, graph patch lifecycle honors `off`/`shadow`/`stage`/`apply_safe` idempotently, `apply_reviewed` requires explicit per-review `graph_patch_authorization`, duplicate lifecycle learning uses the final/informative record, self-learning reports retention integrity, redraw-scope learning stays `soft_hint_only`, and terminal successor/reopen outcomes remain soft eval evidence.

When touching reviewed graph rebase, include the validator, runtime producer, readiness evaluator, trusted operator registry, control-plane authentication, and partial successor owner. The expected current shape is a concrete backend-built candidate, post-repair Closure/scope precedence, no proposal during active Late Fill plus deterministic terminal candidate re-derivation, runtime-owned meaningful diff and preservation proof, no-op/digest/lost-dependency/same-ID or graph-wide semantic-drift/candidate-bookkeeping/partial-containment rejection, advisory Ghost feedback excluded from authority, and product-default non-executable `shadow` with truthful startup provenance. The canonical readiness report is read-only and evidence-gated. Promotion is exactly `adjudicate -> stage -> authorize_partial`; stage is durable audit-only, authorization is registry-trusted and exact-CAS-bound, explicit environment `off` wins, and only a gate-approved partial subtree may append one same-response branch-local successor before scheduling. False negatives remain historical but may be resolved only by one later exact same-class replay-verified useful proposal. Tests must cover missing and drifted current root truth at replay and apply, matching durable full/observation frame identities, phase/downstream prompt carriers (`phase_summary`, `stage_direction`, instructions, criteria), request preservation across successor frames, and credential stripping for direct backend/utility child-process spawns. Root-prompt fallback, parent mutation, full execution, stale lineage, widened scope, missing local execution contracts, and replay duplication must fail closed.

    .venv/bin/python -m pytest tests/test_graph_rebase_review.py tests/test_runtime_graph_rebase_shadow_producer.py tests/test_graph_rebase_readiness.py tests/test_graph_rebase_operator.py tests/test_graph_rebase_partial_successor.py -q

When touching enforced policy, include:

    .venv/bin/python -m pytest tests/test_apply_enforced_policy.py tests/test_graph_repair_self_healing.py tests/test_graph_rebase_review.py tests/test_self_learning.py -q

The expected current shape is default-deny `OLLMO_APPLY_ENFORCED_POLICY`, visible invalid/off/audit diagnostics, safe-v1 allowance only for narrow additive/identity classes, safe-additive risk classification required for safe-additive classes, duplicate artifact alias canonicalization only when refs are proven aliases, conflicting duplicate refs blocked, placeholder/output-slot/work-tree lineage preserved, learning-only and degraded/provider/frontend/monitor-only evidence rejected, full successor rebase blocked, and direct `apply_enforced` partial rebase audit-only/blocked. The separate exact operator-reviewed partial successor path must not be mistaken for enforced authority.

When touching intent-aligned repair/redraw scope selection, include:

    .venv/bin/python -m pytest tests/test_redraw_scope_ladder.py tests/test_response_frames.py::ResponseFrameTests::test_response_frame_canonicalizes_duplicate_artifact_aliases_in_final_projection tests/test_response_frames.py::ResponseFrameTests::test_response_frame_keeps_conflicting_duplicate_artifact_ref_repair_needed -q

The expected current shape is that Runtime exposes `redraw_scope_ladder_review`, reserved/additive/binding/identity scopes are considered before partial or full rebase, graph repair proposals only consume the scope as orientation, rebase proposals preserve bounded scope fields, duplicate refs are canonicalized only when proven aliases, and conflicting duplicate refs stay repair-needed.

For local artifact path identity, also run:

    .venv/bin/python -m pytest tests/test_responses_api.py -q -k "artifact_path_identity"

Different path spellings under one artifact ref may alias only when all resolve
to the same existing local file. Relative paths use the Ollmo checkout root,
never the caller's working directory or a basename search. Checksums alone do
not establish identity. Preserve original paths and branch/phase/provenance in
alias metadata. Distinct files with equal bytes or basenames, unresolved paths,
and incompatible types remain conflicting. Final lifecycle must honor blocked
artifact outputs, and slot hydration must preserve a frozen conflict; only a
new validated frame can replace that adjudication. Active execution, hard
terminal state, and unrelated open Closure checks retain their authority.

For generic intent-obligation graph adequacy, run:

    .venv/bin/python -m pytest tests/test_request_phase_graph_runtime.py tests/test_response_semantics_runtime.py tests/test_graph_repair_self_healing.py -q

Use this when touching `ollmo_g/intent_obligations.py`, `ollmo_g/request_phase_graph.py`, structural `intent_graph_adequacy`, or graph-repair bridges from adequacy checks. The expected current shape is that `request_phase_graph.intent_obligations` exposes text artifact, media artifact, evidence branch, dependency, and navigation promises; strong producer-before-consumer bindings such as generated local images before HTML consumers become executable dependency edges before work runs; missing executable edges surface as `intent_graph_adequacy_missing_dependency_edge`; and Runtime can validate/apply only a safe additive missing-dependency-edge patch. Advisory/provider/degraded/cache/liveness or accepted-learning-only signals must not create executable obligations or validate patches.

The same slice must prove that reserved, negated, or merely inferred modality
cues remain non-executable and do not create Closure-repair work unless a
current-turn obligation was explicitly promoted.

For cleanup/archive retention policy, run:

    .venv/bin/python -m pytest tests/test_clean_repo_state_policy.py tests/test_self_learning.py -q

Dry-run output should include learning-retained and missing response-frame sidecar counts. `state/self_learning/retention_manifest.json` and `state/self_learning/retained_sidecars/` are the evidence continuity surfaces; missing refs should be visible diagnostics, not silent hydration gaps.

For response runtime lifecycle wiring, also run:

    .venv/bin/python -m pytest tests/test_response_semantics_runtime.py -q

For availability waiting, include `availability_wait` and `unavailable_preparation` tests in `tests/test_responses_api.py`, `tests/test_response_semantics_runtime.py`, and `tests/test_runtime_contract_knobs.py`. Fake-clock tests must show pending status, visible wait metadata, unchanged repair counters/targets, ready-sibling progress, one automatic execution after availability returns, and responsive cancellation. Hard-unavailable and excluded-only candidates must not create wait evidence. Run `tests/test_runtime_liveness.py` to preserve cooldown selectability semantics and `tests/test_response_wire.py` for compact wait projection.

`shadow` and `stage` must not mutate executable graph work and must carry non-executable `runtime_effect` values from lifecycle construction. `apply_safe` may apply only validated safe additive patches. On terminal/frozen parents, allowed safe additive repair must keep the parent blocked and byte-stable, persist an exact same-response successor relation, revalidate current autonomy/policy and patch/graph bindings, schedule only the applied owed branches through Late Fill, and keep repeated preparation idempotent. Request preparation and materialization-spec construction must both reject inherited root/assistant prompt recovery when the exact successor branch has no local payload. Same-key execution truth must progress from queued to running to one immutable terminal result across the complete Late Fill envelope, graph, request, and diagnostic projections; delayed queued/running or conflicting terminal callbacks must not restore pending/active branches, regress the canonical response lifecycle, or replace terminal truth. A fake-backend E2E must prove one branch-local backend execution and no root-prompt replay. Invalid graph repair autonomy values must stay safe `off` while surfacing diagnostics.

## Response-Ledger Lookup And Test Isolation

For causal convergence observations, also run:

    .venv/bin/python -m pytest tests/test_causal_telemetry.py tests/test_self_attack_convergence.py tests/test_self_attack_production.py tests/test_ollmo_run_monitor_projection.py -q

See `docs/CAUSAL_TELEMETRY.md` for event bounds, exact-identity proof gates and
inclusive persistence timings. Projected lenses are not model invocations;
historical or incomplete causality stays unknown. New observations must not
change owner outputs, retry/lock behavior, authority or durability boundaries.

When touching response lookup, index persistence, or `/api/responses/<id>` recovery, run:

    .venv/bin/python -m pytest tests/test_response_frames.py -q
    .venv/bin/python -m pytest tests/test_responses_api.py -q --durations=20

For recursive snapshot preparation, include
`tests/test_response_snapshot_preparation.py`. It checks eliminated duplicate
preparation, byte/provenance equivalence, distinct media-file identities, changed
inputs/files, missing/corrupt repeated sidecars, and post-write verification
failure. Snapshot preparation reuse must never suppress fresh integrity checks.

For epoch/map preparation changes, include
`tests/test_response_frame_epoch_verification.py`. It checks exact versus changed
entry bindings, relocated epochs, physical evidence changes during verification,
downstream map tampering, and rejection of old readiness after a successor or
same-byte file replacement. Private digest reuse must not extend file freshness.

`ResponsesApiTests` must redirect `ollmo_webserver.RESPONSE_FRAMES_DIR` to a per-test temporary root. Tests must never scan or write the checkout's production `state/response_frames/responses.jsonl`. A globally fresh, coverage-verified response map may serve validated historical byte-offset hits and prove a missing response id without a ledger scan. Legacy, stale, incomplete, malformed, or corrupt coverage remains uncertain and must retain the safe full-ledger fallback. Do not weaken product timeout or state-transition limits merely to shorten this suite; first inspect duration output for missing mocks, unintended real subprocess/network work, or protected-state coupling.

For the explicit legacy-index boundary, include the `attest_response_frame_index` regressions in `tests/test_response_frames.py`. Attestation must stream rather than call `Path.read_text()` or `_iter_ledger_frames`, preserve the existing `responses` mapping exactly, reject missing ids/latest-coordinate drift/malformed rows/moving evidence without writing, and use an atomic replace only after exact verification. `scripts/attest_response_frame_index.py --check-only` is the operator preflight; tests use temporary roots or a copied temp index with a symlinked source ledger and must never attest the checkout's production index implicitly.

## Fake-Backend E2E Truth Harness

For PNG/SVG request typing and negative-format scope, run:

    .venv/bin/python -m pytest tests/test_inference_service.py tests/test_visual_materialization_intent.py tests/test_generated_image_artifact_routing.py -q

The generated-image routing regressions trace the exact HTML-IMAGE-SMOKE-01
prompt through candidates, promotion, obligations, phases, branches and Late
Fill eligibility. Their bounded fake-provider integration executes the real
branch preparation/execution and final materialization owners, verifies one
PNG producer followed by separate HTML/CSS files, and rejects filename/prose
as production evidence. It does not test the asynchronous scheduler or frame
persistence. All provider outputs and ledgers are temporary; no live image
generation is part of this suite. Broaden intent/extraction changes with the
Ghost routing, request-phase-graph and affected response-semantics tests.

The older `ResponsesApiTests` slice has an isolation limitation: direct calls to
late-fill completion outside a Flask application context can outlive per-test
patches and reach later tests as closure-repair workers. Its fixture also restores
checkout configuration files, and preview paths retain unmocked port/subprocess
probes. Do not describe this slice as fully offline or isolated merely because it
uses the Flask test client. Revalidate it in a disposable physical checkout with
production state absent and external I/O blocked; report blocked probes or leaked
worker calls separately from the request-typing regression result. Do not weaken
the runtime gates or test assertions to hide these isolation failures.

Run:

    .venv/bin/python -m pytest tests/test_fake_backend_e2e.py -q

Use this before Ghost self-learning changes or larger response-frame, artifact, late fill, or observer refactors. The harness patches `/api/responses` to deterministic test-only fake backends and writes only temp `artifacts/`, `state/response_frames/`, `state/artifact_registry.jsonl`, `state/chat_history/`, and `logs/` roots. Assertions are based on runtime truth fields and saved files, not model prose.

The harness also covers the current learning/healing truth boundary: fake `/api/responses` payloads must expose `runtime.request_phase_graph.intent_obligations`, local producer-before-consumer dependency edges, and structural `intent_graph_adequacy`; accepted-learning hints may surface as soft decision-contract orientation but must not create executable graph repair proposals, graph patch lifecycle truth, staged patches, or applied patches by themselves.

## Naming, Schema, And Prompt-Wording Changes

For docs-only public terminology cleanup, verify the glossary and active-doc search:

    rg -n "Planner|Ghost \\+ Planner|execution planner|follow-up generation" README.md GHOST.md OLLMO_FOR_AGENTS.md docs --glob '!docs/diagrams/**'

Expected: hits only where `docs/CANONICAL_GLOSSARY.md` names wording to avoid or where an exact compatibility identifier is backticked.

For compatibility key migrations involving resolver-named replacements for literal legacy keys, do not rely on prose review. Add or keep tests that prove dual-read and replay compatibility for request keys, runtime payload keys, late fill trigger strings, response frames, lookup payloads, and history hydration. Minimum automated suite:

    .venv/bin/python -m pytest tests/test_ghost_execution_planner.py tests/test_responses_api.py tests/test_response_frames.py tests/test_working_frame.py -q -k "execution_planner or planner_timeout or planner_deferred or late_fill or response_frame or history"

For prompt or injected policy wording changes that reach Ghost routing, the resolver, semantic roles, or `GHOST.md`, run:

    .venv/bin/python -m pytest tests/test_ghost_router.py tests/test_ghost_execution_planner.py tests/test_semantic_roles.py -q
    .venv/bin/python -m pytest tests/test_responses_api.py -q -k "ghost_route or ghost_auto or ghost_route_preview or execution_planner or planner_deferred or late_fill"

When a local chat-capable runtime is available, also compare a small live set: plain chat, write-then-speak, describe-then-image, selected-reference follow-up, latest-artifact edit, and each compatibility `ghost_mode` variant that is still accepted at the API edge.

For bounded handoff/transition observations, additionally run:

    .venv/bin/python -m pytest tests/test_transition_telemetry.py tests/test_runtime_contract_knobs.py -q
    .venv/bin/python -m pytest tests/test_responses_api.py -q -k late_fill_branch_progress_updates_response_lookup_before_wave_terminal

These isolated tests cover exhausted observation budgets, paired reservation,
multiple branches/attempts, gate/result provenance, exceptions/aborts, callback
ordering and retained publication/drain waits. Persistence and hydration use
only temporary frame roots; no live-model or self-attack execution is needed.

For opt-in state-flow observation, first run:

    .venv/bin/python -m pytest tests/test_state_flow.py tests/test_transition_telemetry.py tests/test_causal_telemetry.py tests/test_response_snapshot_preparation.py tests/test_readiness_observation_reuse.py -q

Then include affected frame, lookup, readiness/registry, artifact, response
semantics and disposable-checkout API/E2E coverage. Instrumentation must preserve
canonical bytes, owner outputs/exceptions, existing checks and scheduling.
See [STATE_FLOW_DIAGNOSTICS](STATE_FLOW_DIAGNOSTICS.md) for scope bounds and
partial byte/timing coverage. No live submission is implicit in these tests.
