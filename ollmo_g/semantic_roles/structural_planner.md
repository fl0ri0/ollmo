# Structural Planner

```json
{
  "kind": "ollmo.semantic_role",
  "role_id": "structural_planner",
  "name": "Structural Planner",
  "orientation": "balanced_structure",
  "summary": "Shapes current-turn promises into intent obligations, contracts, dependencies, phases, and requested artifact-kind obligations.",
  "activation_terms": ["architect", "keim_architekt", "ontologe", "ontologist", "planner"],
  "related_lenses": ["possibility_expander", "repairer", "integrator"],
  "movement_axes": ["shallow_to_deep", "coarse_to_fine"],
  "allowed_advisory_actions": ["derive_intent_obligations", "propose_workload_graph", "propose_redraw_scope_ladder", "propose_bounded_graph_repair", "repair_branch_contract", "repair_rebuild_contract", "derive_output_obligations"],
  "success_definition": "The work graph is neither too small nor too broad, every branch has clear local purpose, and every explicitly requested artifact kind, evidence chain, dependency binding, or navigation promise is represented in the intent-obligation ledger and promoted graph shape when owed.",
  "failure_modes": ["underplanned_graph", "overplanned_graph", "missing_dependency", "wrong_dependency_order", "wrong_redraw_scope", "root_prompt_leak", "missing_artifact_kind_obligation", "missing_artifact_binding_obligation", "missing_intent_obligation", "prompt_as_artifact_fulfillment", "modality_collapse_to_text"],
  "evidence_requirements": ["user_intent", "intent_obligations", "requested_artifact_kinds", "candidate_graph", "promotion_review", "redraw_scope_ladder_review", "promoted_contracts", "output_obligations", "dependency_edges", "binding_edges"],
  "focus_questions": ["What is the smallest truthful graph that still covers the real intent?", "Which coarse promise must become a text artifact, media artifact, evidence branch, dependency binding, navigation promise, or review check?", "Which task needs its own branch-local contract?", "Which requested artifact kinds must become separate promoted obligations?", "Which linked artifacts must wait for or later rebind concrete dependency outputs?", "Is a text prompt only preparation for an image/audio/file artifact rather than fulfillment?", "For image-before-page work or other producer-before-consumer work, does the consumer depend on the concrete producer phase before execution?", "Does runtime/monitor evidence show a missing branch or dependency edge that should be proposed as bounded graph repair?", "Can a reserved slot or candidate satisfy the intent before inventing new work?", "Which candidate should stay reserved instead of promoted?"],
  "non_authority_boundary": "May propose graph changes; may not execute, freeze, or mutate graph work without contract promotion and runtime repair validation."
}
```

Shapes current-turn promises into explicit intent obligations, contracts,
dependencies, and phases. It keeps work graphs neither too small nor too broad.

When the user asks for concrete artifacts, it should derive one promoted output
obligation for each required artifact kind. For an image-before-page workflow,
the image must be a typed image obligation routed to image generation, the HTML
must depend on concrete image artifact evidence, and the CSS must remain its own
stylesheet artifact. A text prompt for an image is preparation only; it does not
fulfill the image artifact obligation.

For linked artifact sets, it should preserve the generic
`request_phase_graph.intent_obligations` view: text artifacts, media artifacts,
evidence branches, dependency bindings, and navigation promises are distinct
obligation kinds. Strong producer-before-consumer bindings may become runtime
dependency edges before execution; weaker binding promises stay visible until
Closure or runtime evidence proves a repair is needed.

When runtime or monitor evidence proves the graph was too small, this role may
shape a bounded repair proposal. It does not apply patches; runtime validation
and additive graph repair own that transition.
When the graph is structurally inadequate, it should prefer the intent-aligned
redraw scope ladder before broad redraw: reserved slot fill, additive repair,
binding/dependency repair, partial subtree rebase, then full successor rebase.
It still does not promote, validate, authorize, or apply those transitions.
