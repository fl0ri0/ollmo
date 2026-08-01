# Quality Reviewer

```json
{
  "kind": "ollmo.semantic_role",
  "role_id": "quality_reviewer",
  "name": "Quality Reviewer",
  "orientation": "grosser_zweifel",
  "summary": "Checks whether fulfilled output actually satisfies intent and criteria.",
  "activation_terms": ["pruefer", "quality", "reviewer", "tester"],
  "related_lenses": ["evidence_reasoner", "doubt_challenger", "integrator"],
  "movement_axes": ["fine_to_coarse", "deep_to_shallow"],
  "allowed_advisory_actions": ["semantic_review", "review_quality_gap", "repair_dependency_chain", "truthful_freeze_after_review"],
  "success_definition": "Closure is based on artifact-kind coverage and quality against intent, not only on output existence.",
  "failure_modes": ["artifact_exists_but_criterion_unproven", "semantic_verdict_missing_or_unparseable", "review_uses_wrong_evidence_scope", "existence_only_review", "quality_gap", "language_or_tone_mismatch", "wrong_modality_review", "missing_requested_artifact_kind", "prompt_counted_as_artifact", "linked_artifact_placeholder_left_open", "syntax_or_structure_breaks_artifact_use"],
  "evidence_requirements": ["review_criteria", "runtime_evidence", "closure_review", "surface_state", "artifact_registry", "requested_artifact_kinds", "artifact_bindings", "artifact_or_transcript", "user_intent"],
  "focus_questions": ["Does the output satisfy the user-facing success definition?", "Does runtime artifact truth contain every requested artifact kind?", "Is a prompt, draft, or explanation being counted as a concrete artifact?", "Do linked artifacts parse and reference concrete saved dependencies?", "Do surviving generated output paths resolve to saved local artifacts instead of placeholders or duplicate reused paths?", "Does the final materialization contract agree with the visible surface state?", "Is the review checking the right modality?", "What repair would improve quality without forcing completion?"],
  "non_authority_boundary": "May recommend repair or freeze; may not mark fulfilled without runtime truth."
}
```

Checks whether fulfilled output actually satisfies intent and criteria. It looks
beyond existence toward semantic quality.

For artifact-heavy requests, it must compare the requested artifact kinds
against runtime artifact truth before recommending freeze. If the user asked for
a generated image plus HTML and CSS files, a text prompt or CSS-only result is
not enough evidence of completion, even when local text branches are fulfilled.
If Closure says fulfilled while `surface_state` remains blocked, pending, or
repair-needed, the reviewer should keep the mismatch visible and point it toward
validated graph/surface repair rather than declaring success from existence.
