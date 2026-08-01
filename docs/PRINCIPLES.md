# PRINCIPLES

See also `docs/TRUTH_SOURCES.md` for the operational truth-source hierarchy.

## 1. Runtime owns truth

Ollmo does not treat model output as truth.

Truth is what exists in the runtime substrate:

- work_tree
- branches
- output_slots
- outputs
- artifacts
- replayable state

For live model availability, process, port, backend runtime, and control-plane truth win over stale projections. Degraded, busy, timeout, or cooldown markers are advisory unless live process/port/backend truth proves the instance unusable. Backend events, Ghost payloads, frontend labels, and self-learning/self-healing hints must preserve that boundary; a weak `degraded` projection must not become `failed`, `offline`, a provider ban, or graph/route mutation evidence by itself.

---

## 2. Model is not authority

The model:
- proposes structure
- produces results

The runtime decides what is accepted as real.

---

## 3. Planning and execution are separate

- Ghost -> intent, candidate space, and graph structure
- Promotion review -> what becomes owed work
- Resolver/runtime -> branch-local execution and materialization

Ghost anchors and proposes. Runtime review promotes. Resolver acts.

Graph repair follows the same split. Ghost, Closure, decision contracts, and accepted learning may orient or propose a bounded repair, but Runtime applies only validated additive patches to the request phase graph. Advisory movement surfaces such as `controlled_attention_review`, `aspiration_review`, and `commitment_review` can focus attention, but they do not become graph-repair evidence without current runtime/Closure/Monitor actionability.

Every validated graph patch must carry lifecycle truth: proposal id, validation review, repair class, autonomy level, risk level, evidence refs, idempotency key, graph digests, and outcome. Shadow and staged patches are diagnostic truth only; applied safe patches become normal graph state that late fill and Closure must judge. If safe additive repair is needed after a terminal/frozen frame, the old frame remains frozen and the movement is recorded as successor/reopen truth with parent lineage and owed work, not as a silent mutation.

Reviewed graph rebase occupies the higher-risk partial-subtree and full-successor rungs of the same redraw scope ladder; it is not a separate layer. Lower bounded additive redraw remains autonomously active through the default-deny graph-repair `safe_v1` path. A dedicated rebase validator and authority boundary prevent that lower-rung permission from silently authorizing a broader redraw. A rebase/redraw candidate is proposal-only until Runtime computes a meaningful base/candidate diff, verifies the declared candidate digest, and proves preservation of intent obligations, output obligations, artifact refs, dependency edges, same-ID record meaning, partial-scope containment, review duties, target-bound repair contracts, hidden-failure visibility, and frozen parent lineage, then records lifecycle truth. Product-default `shadow` is the non-mutating mode of those same upper rungs, not a new layer; the product stays there while canonical evidence gates are not green. Readiness may combine verified bounded observations from active and archived response-frame epochs through an evidence-only append-only registry, but that durability creates no mutation or operator authority. Explicit `stage` is durable audit truth with no executable effect. Partial execution requires the trusted operator sequence `adjudicate -> stage -> authorize_partial`, an exact CAS binding to the frozen response/frame/proposal/base/candidate identity, a green partial-promotion gate, and a branch-local execution-contract proof. Full replay and the final apply sink must bind the same current root digest recovered from matching durable projections; every executable prompt carrier on phase and downstream records is checked before execution. The resulting continuation is an append-only frame under the same response id; it preserves the parent, schedules only the exact partial successor branches, and never replays the root prompt. Explicit rebase autonomy `off` blocks `stage` and `authorize_partial`; evidence-only adjudication remains available because it is not a rebase transition. Full successor rebase remains shadow/non-executable under safe partial v1.

---

## 4. Intent is anchored, state is fluid

Ghost anchors the current user intent.

The request phase graph is frozen in intent, but runtime evidence can update branch state:

- candidate / reserved
- fulfilled
- pending
- blocked
- failed
- clarified

Candidates are meaningful vacancies in the graph. They may become obligations later, but they are not open obligations until explicitly promoted into the request IR contract.

No old history, old tool call, or old artifact becomes new intent without a current-turn reference.

The same rule applies to context. A remembered fact, old chat turn, or prior artifact may be represented as a context candidate, but it does not become active context or input until current-turn relevance promotes it.

---

## 5. Closure before freeze

Before final response freeze, Ollmo checks graph requirements against runtime truth.

The closure review can continue existing obligations through resolver or late fill, but it cannot invent a new request.

When Closure proves that the basic current-turn intent was not represented in the graph, the remedy is a bounded graph repair proposal and validation review, not prose completion. Accepted learning can make that pattern easier to notice, but it cannot supply executable truth by itself.

---

## 6. Outputs are the contract

Everything externally visible must be:

- stable
- persistent
- referable

`outputs` is the public truth surface.

For linked artifact sets, public output is not closed until surviving generated links resolve to concrete saved local artifacts. Prompts, prose, placeholder paths, guessed filenames, or stale relative links do not satisfy the output contract.

---

## 7. Work, not conversation

Ollmo is not a chat system.

It is a system where work persists, evolves, and continues.

---

## 8. No hidden side effects

All meaningful work must be:

- visible
- attributable
- replayable

No mutation outside the substrate.

---

## 9. One authority

There must be a single place where truth is decided.

Ollmo runtime is that authority.

---

## 10. Resolve blocks by preserving intent

A block is not permission to rewrite the request.

When an obligation cannot be fulfilled yet, Ollmo keeps it visible and resolves the block by the right-sized verified state transition:

- keep the anchored intent
- keep the obligation open
- record the runtime evidence
- continue, block, or explicitly waive only when verified
- freeze the truthful state, not the desired story

The solution to a block is the block's own resolution, handled gently through the substrate, not by forcing a different outcome.

This is a global relevance rule, not only an error handler. The same reflex applies to pending work, reserved or stale candidates, waiver candidates, supersession candidates, repair candidates, and semantic-review work: preserve the state, reconsider current relevance, and choose the right-sized verified transition instead of rewriting intent, under-scoping the work, or pretending completion.

The same rule applies at the surface. UI must not hide a block behind an endless spinner, stale queued state, or successful-looking prose. It should show the truthful runtime state and recovery direction from `surface_state`: open, blocked, reconsiderable, waived, superseded, repair pending, semantic review pending, advisory movement, or completed. A pending advisory surface alone is visible state, not proof that graph repair is needed.

Active reconsideration, quality review, recursive subtask cycles, Closure Repair, Late Fill, and UI are all expressions of the same movement logic. They differ in authority, not in philosophy.

Validated graph repair is the topology form of this rule: preserve the unmet intent as visible state, validate the proposed additive patch against Closure/runtime truth, and then schedule the repaired branch through the normal dependency and late fill gates.

Semantic decision review is the advisory brain-loop form of that logic: it proposes the next transition with reason, confidence, and evidence refs, but Runtime/Contracts/Closure still decide truth.

Global semantic closure is the same principle applied to the whole turn. A branch can be locally fulfilled and still not yet proven as the right note in the whole phrase. Structural graph adequacy checks whether the needed obligations exist; global semantic closure checks whether the fulfilled obligations fit the anchored intent together. When that fit is unproven and cannot be checked deterministically, the system should promote bounded review evidence, not pretend completion.

Intent Lens review is the same principle applied before freeze to current intent itself. Attention verifies that material intent has visible promises, commitment verifies that bindings and dependency order can execute, and aspiration marks explicit text/media/artifact fit promises for whole-turn semantic review. It may surface repairable gaps or promote review evidence, but it must not use advisory language alone to rewrite the graph.

The semantic execution gate is the same principle applied before and during materialization. A branch that was valid a moment ago may become cancelled, waived, or superseded before its backend call starts or before its result returns. Runtime must preserve that new truth, skip queued work, and ignore stale results instead of letting already-started work force the graph to accept the wrong output.

Controlled attention is the same principle applied to model focus. Between visible outputs, Ghost/Reviewer should not receive the whole request as an undifferentiated prompt again. They should receive scoped attention frames: what target is in question, what evidence anchors it, which transitions are allowed, and where authority stops. This is the "space between the tones": filter, reconsider, reserve, promote for review, repair, waive, supersede, stop, or freeze only through the right-sized verified transition.

---

## 11. Possibility promotes into contract

A possible output is not automatically owed.

Ollmo may keep candidate or reserved outputs in the graph as meaningful vacancies. Those candidates become contractual work only through relevance and explicit promotion:

- candidate / reserved possibility
- promoted contract
- pending / fulfilled / deferred / blocked / waived / superseded runtime state

This is the positive counterpart to explicit waiver. Waiver releases an obligation; promotion claims a candidate as real work.
Reserved, omitted, and stale candidates stay reconsiderable but non-executable. Supersession closes an already-promoted obligation when newer runtime truth or a replacement branch makes the old obligation no longer owed.

---

## 12. Context promotes into relevance

Not every memory becomes context.

Not every context candidate becomes a duty.

Not every old artifact becomes input.

Ollmo may preserve History, Memory, and Reference candidates as available orientation, but they become active context only through explicit promotion:

- history / memory / reference candidate
- history scan candidate
- promoted active context, active memory, or active reference
- promoted history scan over existing history, response-frame, and artifact ledgers
- scan matches become ordinary context candidates with existing ids and refs
- runtime route, artifact, or response may use it
- frozen response frame records why it was active

Unpromoted context remains possible, not binding. This keeps continuity available without letting stale history steer the current turn.

---

## 13. Work repeats at branch scale

The request is not the only unit that deserves planning, execution, review, and freeze.

Every promoted workload task or downstream branch should carry the same cadence at its own scale:

- prepare the focused task
- gather or bind required inputs and evidence
- execute or materialize the branch
- verify against the branch contract
- freeze the branch result into runtime truth

Deterministic verification should stay deterministic. Semantic review is added only when the branch contract contains semantic criteria runtime cannot prove.

Later branches must not rerun the full original prompt as if it were their own task. They should consume their promoted branch contract: focused `content_payload`, `artifact_prompt`, `stage_direction`, dependencies, input artifacts, reference artifacts, prior branch results, and evidence.

---

## 14. Evidence travels through artifacts

An artifact is not only a path.

Durable artifact truth includes identity, provenance, metadata, enrichments, linked response/message ids, and availability. `artifact_dossiers` are the read-side shape for that truth.

When artifact evidence already exists, Ollmo should reuse it as evidence before rerunning expensive or noisy analysis. If the evidence is missing, stale, or insufficient for the current branch contract, a new evidence branch may be promoted.

---

## 15. Hidden hard caps are not control

Arbitrary hidden limits are not valid runtime knobs.

If a bound is required for safety or transport, it must be explicit, named, justified, observable, and adjustable where practical. The local substrate should preserve canonical policy, graph, artifact, and contract truth rather than silently truncating it.

---

## 16. Learning is reviewed trace improvement

Ollmo may learn from its frozen response frames, but learning is not hidden live mutation.

The allowed path is:

- frozen response frames and reviews
- extracted eval cases
- policy improvement candidates
- reviewed accepted learnings
- explicit policy snapshot activation

Until a learning is reviewed and explicitly enabled, it is evidence only. It may appear in Ghost diagnostics and offline reports, but it must not silently alter Intake, Graph, Context Gate, Closure Review, routing, or output behavior.

The accepted-policy snapshot is a bridge, not authority. New or reset snapshots are disabled by default; this checkout may enable the snapshot as readable soft policy input after review. The bridge gives reviewed learnings a clear place to land without bypassing runtime truth.

Even when explicitly enabled, accepted learnings become bounded soft hints only. They may focus attention in the present turn, but they do not override current user intent, live capability evidence, Graph, IR, Closure Review, context promotion, output obligations, or artifact truth.

Learning may remember graph patch outcomes such as applied-and-closed, applied-but-blocked, conflict rejection, solved missing obligation, false work, degraded-signal ignored, or terminal successor/reopen created/solved/blocked. Those outcomes can calibrate future proposal orientation, but they still do not validate or apply a current patch without runtime truth.

Learning may also remember graph rebase outcomes such as proposal accepted/rejected, preservation proof failure, staged rebase, partial successor queued/solved/blocked, learning-only rejection, degraded-signal rejection, gate rejection, or full-rebase safe-v1 block. Those cases can orient when a future rebase proposal is worth considering, but they cannot satisfy preservation proof, make a readiness gate green, write a trusted operator record, authorize `apply_reviewed`, or create successor truth.

Learning evidence must remain inspectable. If active self-learning records reference response-frame sidecars, cleanup/archive must either retain learning-owned copies or report missing sidecars visibly through retention diagnostics. A missing sidecar weakens historical evidence; it must not silently disappear or become executable truth.
