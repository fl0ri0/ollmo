# Ghost Self-Alignment

Status: architecture note plus runtime-visible offline substrate, enabled accepted-learning soft hints, and runtime-evidence graph-repair proposal bridge.

## Boundary

This is not Reinforcement Learning.

Ghost self-alignment does not replace Ollmo's runtime truth model.

Ollmo's authority remains:

```text
Intake / Context Gate
  -> Request Phase Graph
  -> Candidate Graph / Promotion Review
  -> Request IR
  -> Workload / Branch Runtime Work State
  -> Closure / Context Review
  -> Frozen Response Frame
```

## Frame-Eval Loop

Ollmo's useful Ghost-alignment loop is trace-based and offline:

```text
execution traces
  -> reflection / critique
  -> candidate prompt or policy mutations
  -> eval selection
  -> improved instructions / heuristics
```

The trace material already exists:

- frozen response frames
- `runtime.graph_closure_review`
- `context_contract.context_gate_review`
- request phase graphs
- `candidate_graph` and `promotion_review`
- `request_ir.decision_contract`
- request IR output obligations
- workload task lifecycle and review records
- output candidates and promotions
- context candidates and promotions
- blocked / pending / waived states

The useful loop is:

```text
frozen frames + closure/context reviews
  -> eval cases
  -> offline Ghost self-alignment report
  -> proposed changes to Ghost / Intake / Graph heuristics
  -> reviewed patch, not hidden runtime mutation
```

This keeps optimization auditable. It does not let a model silently rewrite the live runtime contract.

The implemented substrate is intentionally bounded:

- `ollmo_services/self_learning.py` reverse-streams a bounded recent ledger window and extracts proposal-only eval cases from only the newest durable frame per response id. It unions that recent selection with exactly bound cases from the curated graph-rebase shadow corpus, then runs one ordinary extractor and one stable dedupe stream; corpus evidence has no separate strength, weight, priority, or policy tier. It reports inspected physical frames, evaluated responses, and superseded frames separately so transitional Late Fill states do not contradict their fulfilled successors.
- Corpus binding uses `final_debug.summary.id` plus the embedded `response_frame.frame_id` and `frame_sequence` against the newest durable ledger frame. Older runner `last_frame_*` fields remain diagnostics, so a stale helper field such as D4R2's frame-1 value cannot override its exact final-debug frame-2 identity. Planned, dependency-blocked, malformed, missing, or stale records stay visible in coverage and do not invent eval outcomes.
- Bounded `final_debug.summary.graph_records`, `diagnostic_records`, and `redraw_scope` records are merged only into an eval-time copy when their stable identities are absent from canonical frame projections. This exposes corpus-captured graph lifecycle evidence to the existing extractors without rewriting Response Frames, corpus manifests, CAS sidecars, or runtime truth.
- `scripts/build_self_learning_eval_cases.py` reads `state/graph_rebase_shadow_corpus` by default and materializes `state/self_learning/eval_cases.jsonl` and `state/self_learning/report.json`. `--frames` is repeatable so a fresh active Response Ledger epoch and one or more read-only archived epochs enter one ordinary deduplicated eval stream; the first ledger wins only when the same response id appears in multiple inputs, and each selected frame hydrates from the CAS tree beside its own ledger. The library accepts optional additional ledger paths and an optional corpus directory so tests and other callers do not implicitly scan production state. Default execution still replaces the eval-case ledger. The opt-in `--merge-existing` mode instead reads the existing `--output` ledger, unions old and fresh cases by `case_id`, preserves old-only cases, and lets a fresh duplicate replace its old record. `--max-cases` limits fresh extraction only and cannot delete preserved history. Historical graph-rebase cases retained this way remain offline eval evidence, not current runtime, frame-selection, or corpus-binding truth. Counts, improvement candidates, and shadow hints are recomputed over the merged cases; `--no-persist` previews all merge counts and the merge policy before any write. Both outputs are staged before installation and each file uses whole-file atomic replacement with caught-failure rollback, so neither file body can be truncated. The two paths are not one portable cross-file snapshot transaction: a hard stop between replacements may leave adjacent complete generations, and one deterministic rerun converges them. The CLI holds cooperative output locks across read/merge/commit; composed service callers must use `self_learning_output_update_lock()` over the same scope. Merge rejects protected input/state destinations, leaves Response Frames, indexes, CAS sidecars, retained sidecars, and `accepted_policy_snapshot.json` untouched, and cannot promote or enable accepted learning.
- `ollmo_services/self_learning_retention.py` and `scripts/collect_self_learning_retention_roots.py` collect sidecar refs reachable from active self-learning state and write `state/self_learning/retention_manifest.json`; cleanup/archive copies retained sidecars into `state/self_learning/retained_sidecars/`. After the source ledger epoch moves, a retained copy satisfies the root only when the copied file exists and matches the declared Response Frame SHA-256 when present; missing or mismatched source and retained copies remain a visible partial failure.
- `/api/ghost` exposes a bounded offline Ghost self-alignment summary without dumping the full case ledger.
- `state/self_learning/accepted_policy_snapshot.json` is the reviewed bridge for accepted learnings. New or reset snapshots are disabled by default; this checkout can enable the snapshot as readable soft policy input when `enabled` is true.
- `scripts/manage_self_learning_policy.py` can initialize, show, promote, enable, and disable reviewed accepted-learning snapshots.
- The report builder now emits diagnostic `shadow_hints` from extracted eval cases and policy candidates. These hints always carry `authority: shadow` and `runtime_effect: none`; they do not enter live routing, graph construction, context promotion, closure, provider selection, or artifact fulfillment.
- Optional monitor-report ingestion can attach supporting evidence from `state/ollmo_run_monitor/reports.jsonl`, such as HTML syntax issues, broken links, final materialization contract status, failed branch counts, and backend-family chat route-health events. Monitor evidence is supporting context only; frozen response frames remain canonical truth.
- `/api/ghost` and the Ghost router context expose `accepted_learning_hints`, but disabled snapshots return no hints.
- Enabled accepted-learning hints are still soft hints only. They carry concrete reviewed hint text, case kinds, severity counts, evidence ids, and a conflict boundary. They may orient a present turn, but they do not mutate Graph, IR, Closure Review, routing, context promotion, or output obligations by themselves.
- The normal non-destructive eval refresh command is:

  ```bash
  python3 scripts/build_self_learning_eval_cases.py \
    --monitor-reports state/ollmo_run_monitor/reports.jsonl \
    --frame-limit 200 \
    --max-cases 300 \
    --merge-existing
  ```
- Self-healing and accepted learning must not promote weak liveness projections into hard runtime truth. A `degraded` readiness/cache marker is advisory while live process, port, or backend evidence still proves the instance callable; it may lower preference, but it is not provider-ban, offline, graph-repair, or route-mutation evidence by itself.
- `request_ir.decision_contract` may summarize accepted-learning hints for Ghost's current thinking posture, preserving the hint text and case-kind evidence. That summary is orientation only and never promotion, waiver, supersession, repair, or semantic-review proof by itself.
- `decision_contract.semantic_decision_review` may attach accepted-learning orientation to proposal-only decisions. This gives the brain loop a reviewed hint surface without making learning live authority.
- `decision_contract.controlled_attention_review` may turn accepted-learning hints into low-priority attention frames. They can focus model attention between steps, but they cannot promote, execute, waive, supersede, satisfy review, or freeze anything.
- `runtime.graph_closure_review.global_semantic_closure_review` produces learner-visible traces for whole-turn semantic fit: unresolved reviews become proposal-only eval cases, and completed reviews become positive traces. Accepted learning cannot replace review evidence or Closure truth.
- The accepted-learning snapshot carries an `authority` field. The supported value is `soft_hint`; other values do not grant authority. Runtime truth remains the final authority for what exists.
- Basic current-turn intent failures are first-class learning evidence. When `intent_graph_adequacy` or Closure checks show that the graph did not represent the requested outputs or dependencies, self-learning may produce accepted hints that orient graph repair. Those hints remain proposal-only: they can focus Ghost on a bounded repair candidate, but Runtime still requires current Closure/runtime evidence before applying any graph patch.
- `ollmo_services.graph_repair.build_graph_repair_proposals_from_runtime_evidence(...)` maps current response-frame, Closure, late fill, and artifact evidence into graph-repair proposals for known repair classes such as unmet materialization contracts, terminal pending branches, duplicate artifact refs, broken/fake artifact dependencies, and fulfilled-contract/surface-state mismatches. Backend response runtime stores current proposal/review diagnostics in response truth; monitor reports are optional supporting observer evidence, not product truth. Fulfilled-contract/surface reconciliation now requires actionable surface evidence; advisory-only pending state from `controlled_attention_review`, `aspiration_review`, `commitment_review`, or reconsideration stays visible as movement/attention evidence, not repair evidence. These proposals still require validation before Runtime can apply an additive graph patch.
- Validated graph patches record lifecycle truth under `runtime.request_phase_graph.graph_patch_lifecycle`. `shadow` and `stage` are non-executable diagnostics; `apply_safe` may apply only safe additive classes and records `applied_graph_patches` with idempotency and graph digests. If a parent frame is terminal/frozen, `apply_safe` or allowed `apply_enforced` additive repair keeps the parent blocked and records `successor_reopen_requests[]` with parent-frame lineage and the would-have-applied successor graph. The terminal owner revalidates the current policy and exact patch/graph/parent bindings, appends one `graph_patch_reopen_successor` frame under the same response id, and schedules only its exact owed branches through normal Late Fill; the parent and unrelated completed truth stay unchanged, and the root prompt is never replayed. `apply_reviewed` requires explicit per-review `graph_patch_authorization`; the rollout knob alone cannot authorize review-required mutations. `apply_enforced` is default-deny through `OLLMO_APPLY_ENFORCED_POLICY`; only safe-v1 missing-branch, dependency, artifact-binding, and proven duplicate-alias classes can apply after current evidence, redraw-scope, idempotency, and forbidden-evidence gates pass. Self-learning can extract outcomes such as applied-and-closed, applied-but-blocked, rejected-due-to-conflict, solved-missing-obligation, false-work, degraded-signal-ignored, terminal successor/reopen created/solved/blocked, and enforced-policy allowed/blocked cases, preferring the final/informative lifecycle record when duplicates exist, but accepted learning remains a soft hint and cannot validate a future patch by itself.
- Reviewed graph rebase occupies the successor-only upper rungs of the same redraw scope ladder, not a separate layer. Lower bounded additive redraw remains autonomously active under graph-repair `safe_v1`; product-default `shadow` is the non-mutating mode of only the upper rebase rungs. Runtime may retain a distinct deterministic response-time graph as a bounded candidate and re-derive it from final runtime truth after active Late Fill, but only actionable current Closure checks plus post-repair scope truth can promote it to proposal review once smaller scopes are exhausted. Ghost repair-feedback items remain advisory and cannot supply rebase authority. Runtime must compute `ollmo.graph_rebase_diff`, require a meaningful change, reject candidate-owned repair/rebase bookkeeping, and pass dependency and graph-wide semantic `ollmo.graph_rebase_preservation_proof`. `GET /api/graph_rebase/readiness` is read-only evidence truth. Partial promotion is an explicit trusted `adjudicate -> stage -> authorize_partial` sequence through the per-response operator endpoint; stage is durable audit-only, while authorization requires an exact CAS-bound registry chain, matching durable frame/current-root truth, a green partial gate, and a branch-local execution contract. The resulting partial continuation is one append-only same-response successor that preserves the parent and forbids root replay through every executable prompt carrier. Explicit rebase autonomy `off` blocks it. Full successor rebase stays shadow/non-executable under safe partial v1. Learning-only/degraded/provider/advisory evidence cannot authorize rebase, and accepted learning cannot satisfy any preservation, readiness, registry, or execution gate.

## Context Access

Ollmo's context access should be described in Ollmo's own terms:

```text
history / memory / artifact substrate
  -> context candidates
  -> optional promoted history_scan
  -> ranked scan matches with ids and artifact_refs
  -> promoted active context / active reference
  -> frozen context_gate_review
```

This is not merely retrieval. It is a recorded state transition:

```text
possibility
  -> relevance
  -> promoted contract / active reference
  -> runtime truth
  -> closure review
  -> frozen frame
```

The same rule applies to outputs and history:

```text
candidate_output
  -> promoted output_obligation
  -> fulfilled / pending / blocked / waived

context_candidate
  -> promoted active_context / active_reference
  -> used or withheld for this turn
```

## Intake and Closure

Intake is the opening boundary:

- What is the current user intent?
- Does it need old context?
- Does it need broader history scan?
- What is only possible?
- What is promoted?

Closure is the ending boundary:

- Did the graph represent the current intent?
- Did the runtime fulfill the promoted obligations?
- Did context stay non-binding unless promoted?
- Is the frozen frame truthful?

They are not the same component, but they use the same logic.

## Practical Rule

Do not add Ghost self-alignment as hidden live authority.

Use it as an offline evaluation loop over Ollmo's own frames:

```text
Ollmo produces truth traces.
Ghost self-alignment studies truth traces.
Human/reviewed patch updates Ghost/Intake/Graph policy.
Runtime truth remains in Ollmo.
```

The accepted-policy snapshot is the reviewed activation seam:

```text
eval cases
  -> improvement candidates
  -> reviewed accepted learnings
  -> accepted_policy_snapshot enabled by operator/review
  -> bounded runtime hint, default authority: soft_hint
```

Before that explicit enable step is used, accepted learnings are diagnostic state only. In the current enabled checkout, they are readable bounded hints, not hidden live authority.

The learner now also watches semantic decision, controlled-attention, and surface-state traces:

```text
semantic_decision_review / controlled_attention_review / surface_state
  -> eval cases
  -> semantic_decision_policy or controlled_attention_policy improvement candidates
  -> reviewed accepted learning
  -> bounded runtime hint
```

That keeps the new brain loop trainable without letting it silently rewrite the runtime contract.

The learner now also watches whole-turn semantic closure:

```text
global_semantic_closure_review
  -> semantic_review_verdict when a promoted reviewer ran
  -> eval cases
  -> semantic_review_policy / semantic_verdict_policy improvement candidates
  -> reviewed accepted learning
  -> bounded runtime hint
```

That keeps "branch fulfilled but whole request wrong" failures trainable without giving the learner authority to fulfill, waive, supersede, or freeze. A passed verdict is a positive trace; failed, uncertain, or unparseable verdicts are unresolved traces for reviewer/prompt/evidence policy.

The learner also watches branch-level semantic verdicts:

```text
branch_semantic_review
  -> semantic_review_verdict
  -> semantic_verdict_policy eval cases
  -> reviewed accepted learning
  -> bounded runtime hint
```

That keeps "this one branch exists but does not satisfy its role" failures trainable without forcing every branch through review.

The learner also watches graph adequacy and open-obligation traces:

```text
intent_graph_adequacy / open_graph_obligation
  -> eval cases
  -> ghost_intake_graph_policy or closure_review_policy improvement candidates
  -> reviewed accepted learning
  -> graph-repair orientation only
  -> current Closure/runtime validation before any additive patch
```

That keeps "the basic request was not met" failures trainable and healable without letting the learner become executable graph truth.

The graph-repair bridge closes the next diagnostic gap:

```text
response frame / Closure / late fill / monitor report
  -> runtime-evidence graph-repair proposals
  -> validate_graph_repair_proposal
  -> graph_patch_lifecycle under off/shadow/stage/apply_safe/apply_reviewed/apply_enforced
  -> apply_validated_graph_patch only when accepted and allowed by autonomy
  -> successor_reopen_requests when safe additive repair is needed after terminal freeze
  -> terminal-owner revalidation
  -> graph_patch_reopen_successor
  -> exact owed branches through normal Late Fill
```

That means repeated failures can become concrete repair proposals, while backend-family route-health events still default to preference, cooldown, or retry diagnostics rather than broad provider disablement or graph patches. With no override, `apply_enforced` plus `safe_v1` is the default-deny bounded additive path; `apply_reviewed` still needs exact authorization, and neither path grants upper-rung Rebase authority.
