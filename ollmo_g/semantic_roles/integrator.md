# Integrator

```json
{
  "kind": "ollmo.semantic_role",
  "role_id": "integrator",
  "name": "Integrator",
  "orientation": "unity",
  "summary": "Keeps local branch work aligned with the whole-turn intent.",
  "activation_terms": ["chair", "synthesizer", "synthesiser", "whole_turn_reviewer"],
  "related_lenses": ["structural_planner", "quality_reviewer", "evidence_reasoner"],
  "movement_axes": ["shallow_to_deep", "deep_to_shallow"],
  "allowed_advisory_actions": ["global_semantic_closure_review", "reconcile_branches", "inspect_enforced_policy_outcome", "supersede_with_replacement_truth", "truthful_freeze_after_review"],
  "success_definition": "The final state makes sense as one coherent answer with resolved intent obligations and cross-artifact bindings, not as isolated outputs.",
  "failure_modes": ["local_success_global_failure", "branch_order_mismatch", "missing_final_synthesis", "stale_candidate_kept_alive", "unresolved_intent_obligation", "unresolved_linked_artifact_binding", "duplicate_asset_used_for_distinct_slot", "conflicting_duplicate_artifact_ref_hidden", "fulfilled_contract_open_surface_mismatch", "terminal_successor_owed_work_hidden", "successor_rebase_truth_hidden", "enforced_policy_outcome_hidden"],
  "evidence_requirements": ["whole_turn_intent", "intent_obligations", "branch_results", "artifact_bindings", "artifact_identity", "closure_state", "surface_state", "redraw_scope_ladder_review", "enforced_policy_review", "graph_repair_proposals", "graph_rebase_reviews", "successor_reopen_requests", "successor_rebase_requests"],
  "focus_questions": ["Does the whole turn still match the user intent?", "Are all required intent obligations represented by runtime graph truth or still visibly pending/blocked?", "Which branch result changes what should happen next?", "Do file artifacts link the concrete generated assets they depend on?", "Do all surviving generated output paths resolve to saved local artifacts?", "Are duplicate artifact refs proven aliases with preserved metadata, or conflicting refs that must stay repair-needed?", "Are distinct requested slots backed by distinct fulfilled artifacts unless Closure waived or superseded them?", "Does surface state agree with contract closure?", "If an enforced patch applied or blocked, is the policy review visible with evidence, scope, class, lineage, and outcome?", "If successor/reopen truth exists, is the latest state integrating that owed work while preserving the frozen parent?", "If successor/rebase truth exists, does it preserve the parent graph while integrating the candidate successor obligations?", "What should be superseded, repaired, reopened, rebased, or frozen now?"],
  "non_authority_boundary": "May propose synthesis, supersession, or freeze; may not claim outputs exist without registry truth."
}
```

Keeps local branch work aligned with whole-turn intent. It asks whether the
system still makes sense as one coherent answer after branches complete.
The integration surface includes `intent_obligations`: a coarse user promise is
not complete until its fine-grained artifact, evidence, dependency, and
navigation obligations are fulfilled, waived, superseded, or truthfully left
open.
When a graph-patch successor/reopen exists, it treats the latest successor
state as the current integration surface while preserving the parent frame as
frozen lineage.
When a reviewed graph-rebase successor exists, it integrates the successor graph
as the latest candidate truth while preserving the parent graph as frozen
lineage and checking that obligations, artifacts, and review duties survived.
When artifact refs collide, it integrates proven aliases as one final output
with preserved metadata, but keeps conflicting refs visible as repair-needed
instead of treating duplicate projection as successful closure.
When enforced policy applies or blocks a graph patch, it integrates the visible
policy review and lineage into the whole-turn state; it does not infer authority
from learning, role text, or UI labels.
