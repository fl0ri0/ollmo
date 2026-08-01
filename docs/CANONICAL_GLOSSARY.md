# Canonical Glossary

Status: recommended naming source of truth for docs, UI text, diagrams, onboarding material, operator copy, and code review.

## Freeze Rule

Treat this file as the naming authority for public and human-facing terminology.

When docs, UI copy, diagrams, onboarding text, operator/admin labels, or review comments need a term, this glossary wins unless the surface is deliberately quoting a literal compatibility identifier or a historical migration note.

Do not introduce new public wording that re-centers legacy terms such as `planner`, `execution planner`, `Ghost planning`, `late-fill`, or generic `follow-up generation` when the canonical terms in this glossary apply.

## Purpose

Use this file to keep naming aligned with the actual architecture.

When public prose, UI labels, or review comments need a term, this glossary wins over legacy habit.
When code, payload, or file identifiers still use legacy names, mention the canonical term first and the literal identifier second.

## Canonical Terms

### Ghost

The runtime-intelligence layer inside Ollmo.

Use when referring to the subsystem as a whole.
Do not use `Ghost` as a synonym for the whole Ollmo product.

### Ghost-owned graph derivation

Ghost's structural phase and request-graph responsibility.

Use for:

- request-phase interpretation
- request phase graph derivation
- structural phase decisions
- Ghost-owned routing intent

Preferred public phrasing:

- `Ghost`
- `Ghost-owned graph derivation`
- `Ghost-owned phase decision`

Avoid in public prose:

- `Planner` as a separate layer beside Ghost
- `Ghost + Planner + Resolver` as the architecture model
- `execution planner` when you actually mean the resolver stage
- `Ghost planning` when `Ghost-owned graph derivation` or `Ghost phase decision` is clearer

### Resolver

The compatibility-named execution layer that advances already-planned work.

Use for:

- selecting ready downstream branches
- refining executable payloads over time
- materializing already-planned work
- continuing frozen obligations without re-planning the request

Preferred public phrasing:

- `resolver`
- `execution resolver` when a longer form improves readability

Allowed compatibility form when a literal identifier matters:

- `resolver` (`execution_planner`)

Avoid in public prose:

- `execution planner` as the primary term

### Compatibility identifiers

Some internal code, payload, trigger, and test surfaces still carry legacy planner names for schema and replay compatibility. These are not active architecture terms and should not be introduced into public wording.

Allowed literal identifiers when naming code or payload fields:

- `ollmo_g/execution_planner.py`
- `runtime.execution_planner`
- `execution_planner`
- `planner_timeout_ms`
- `planner_timeout_sec`
- `planner_*`
- `execution_planner_deferred_follow_up`

Writing rule:

- Say `resolver` first, then put the literal compatibility identifier in backticks when the exact key or module matters.
- Do not describe `Planner` as a separate layer beside Ghost.
- Do not rename persisted compatibility keys casually; code/schema migrations require dual-read or compatibility wrappers.
- Python module or function renames may use aliases first, but request keys, persisted response/runtime payload keys, trigger strings, response-frame fields, history fields, and replay/resume data need staged dual-read before any new write shape.
- Prompt or policy wording that reaches Ghost, the resolver, semantic roles, route construction, or injected runtime policy is behavior-affecting. Rename such wording only as a dedicated rollout with targeted router/resolver/Responses tests and, when local runtime is available, side-by-side route/output comparison.
- Do not keep separate naming migration docs as active architecture. This glossary is the current naming authority.

### Late fill

Continuation/materialization after the current phase has already produced a truthful interim result.

Use for:

- pending downstream artifact completion
- continuation after a truthful phase-1 or prepare-phase result
- UI and operator status labels for deferred completion

Preferred public phrasing:

- `late fill`

Allowed code-style forms when referring to literal identifiers:

- `late_fill`
- `late_fill_runtime.py`

Avoid in public prose:

- `late-fill`
- generic `follow-up generation` when `late fill` is the actual runtime concept

## Related Terms

### Request phase graph

The structural graph of phases and dependencies for a request.
Ghost owns its derivation.
The resolver and late fill act within that graph; they do not replace it.
The graph is frozen in intent and fluid in state: Ghost anchors what the user asked for, while runtime evidence updates fulfillment, pending, blocked, failed, or clarified state.

### Candidate graph

The visible possibility layer before work is owed.
Use for possible outputs, workload tasks, context, references, memory, evidence, repairs, continuations, learning hints, and reserved or rejected options.
The pure helper module is `ollmo_g/candidate_contracts.py`.

Literal identifier:

- `candidate_graph`

### Promotion review

The review boundary that turns a candidate into a promoted contract or leaves it reserved, omitted, waived, rejected, or stale.
Use this term when describing why a possibility became executable work, active context, repair work, evidence, or a continuation.
Reserved, omitted, and stale decisions are reconsiderable possibility states; they are not executable work, but later current evidence may promote them.

Literal identifier:

- `promotion_review`

### Promoted contract

Executable owed work created from current evidence.
Use this instead of implying that every possible candidate must run.
Unpromoted, omitted, stale, and reserved candidates stay visible but non-executable.

### Superseded obligation

Owed work that was replaced or made no longer relevant by newer runtime truth.
Use this separately from `waived`: waiver releases owed work by explicit review or policy; supersession closes owed work because another branch, artifact, or contract now represents the relevant work.

Literal identifier:

- `superseded`

### Workload task

The branch-scale task contract derived from a promoted phase or obligation.
Use when describing the focused work a downstream branch must execute: declared inputs, dependencies, lifecycle stages, output contract, visibility, and review criteria.

Literal identifier:

- `workload_task`

### Artifact dossier

The read-side artifact continuity record keyed by durable `artifact_ref`.
Use for artifact identity, provenance, metadata, enrichments, linked response/message ids, and availability.

Literal identifier:

- `artifact_dossiers`

### Text artifact payload envelope

A structured wrapper that contains the actual file payload.
Use when a model returns `output_obligations[].content` or similar metadata around a requested text/file artifact. The artifact payload is the declared `content`, not the wrapper.

### Current-turn-only intake

The default context strategy for fresh turns.
Use when older thread history, old tool calls, or prior artifacts must not be interpreted as the new request's intent unless the current turn explicitly references them.

Literal identifier:

- `current_turn_only`

### Graph closure review

The deterministic pre-freeze review that compares the request phase graph against runtime truth.
Use for fulfillment checks before final response freeze.

It asks what the graph required, what runtime truth produced, what remains pending or blocked, and which existing obligations may continue through resolver or late fill.

Literal identifier:

- `runtime.graph_closure_review`

### Runtime truth

The actual state Ollmo can prove from graph, branch, slot, output, artifact, response frame, late fill, and runtime-status data.
Use this instead of visible assistant prose or model critique when deciding whether work is real.

### Local model execution

Provider/backend calls that produce selected phase or branch outputs.
Use separately from fulfillment review: models execute or materialize, while runtime closure decides what is fulfilled.

### Optional legacy reviewers

Historical or experimental plan-refinement, semantic handoff, or critique/review paths.
Use only when discussing optional experiments or compatibility surfaces.
Do not describe them as the canonical graph closure loop.

### Working frame

The mutable live request image before freeze.
Use for the fluid middle.

### Response frame

The frozen auditable request image after execution/freeze.
Use for replay, audit, and durable truth.

### Control hints

Post-route detail filling that maps user intent onto truthful runtime/session controls.
Use separately from both Ghost-owned graph derivation and the resolver.

## Usage

- In docs and UI, use canonical terms by default.
- If a literal code or payload identifier still uses legacy naming, write the canonical term first and the literal identifier second.
- Keep legacy wording only in migration notes, schema-compatibility notes, or direct code identifiers.
- When editing diagrams or labels, prefer short noun phrases: `Ghost`, `Request phase graph`, `resolver`, `late fill`.
- Do not describe hidden hard caps as architecture. If a bound is technically necessary, name it as an explicit budget or safety knob and document the reason.
