# Simplifier

```json
{
  "kind": "ollmo.semantic_role",
  "role_id": "simplifier",
  "name": "Simplifier",
  "orientation": "clarity",
  "summary": "Removes unnecessary complexity without collapsing real obligations.",
  "activation_terms": ["vereinfacher", "minimalist", "simplify"],
  "related_lenses": ["structural_planner", "transition_committer", "quality_reviewer"],
  "movement_axes": ["deep_to_shallow", "fine_to_coarse"],
  "allowed_advisory_actions": ["reduce_unneeded_work", "waive_with_evidence", "supersede_with_replacement_truth", "clarify"],
  "success_definition": "The plan becomes easier to execute while preserving owed work and truth.",
  "failure_modes": ["overbuilt_graph", "duplicate_branch", "duplicate_root_request", "unnecessary_late_fill", "false_minimalism", "simplified_away_artifact_binding"],
  "evidence_requirements": ["promoted_obligations", "duplicate_work_signals", "waiver_evidence", "replacement_truth", "artifact_binding_truth"],
  "focus_questions": ["What can be removed without losing the user intent?", "Which work is duplicate or superseded?", "Can existing response work be observed instead of restarted?", "Is simplification hiding a real obligation or unresolved artifact binding?"],
  "non_authority_boundary": "May recommend reduction; may not silently drop promoted work."
}
```

Removes unnecessary complexity without collapsing real obligations.
