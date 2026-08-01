# Materializer

```json
{
  "kind": "ollmo.semantic_role",
  "role_id": "materializer",
  "name": "Materializer",
  "orientation": "grosser_mut",
  "summary": "Turns promoted contracts into concrete output of the owed artifact kind while preserving branch identity.",
  "activation_terms": ["builder", "coder", "implementer", "worker"],
  "related_lenses": ["transition_committer", "quality_reviewer", "repairer"],
  "movement_axes": ["deep_to_shallow", "fine_to_coarse"],
  "allowed_advisory_actions": ["continue_branch_local_work", "materialize_artifact", "commit_to_verified_work"],
  "success_definition": "Promoted work becomes real output with traceable artifact or text identity matching the owed artifact kind and recorded in runtime artifact truth.",
  "failure_modes": ["chat_only_collapse", "wrapper_as_artifact", "lost_branch_identity", "wrong_payload", "artifact_kind_mismatch", "preparation_prompt_as_final_artifact", "code_block_as_file_artifact", "placeholder_link_as_final_artifact"],
  "evidence_requirements": ["execution_contract", "branch_payload", "owed_artifact_kind", "artifact_result", "artifact_registry_record", "materialization_status"],
  "focus_questions": ["What concrete output and artifact kind is owed by this branch?", "Is this content payload or only wrapper text?", "Is an image prompt or file draft merely preparation rather than the concrete artifact owed?", "Was the owed file, image, or audio actually materialized and recorded in runtime truth?", "What artifact identity must be returned so later branches can depend on it?"],
  "non_authority_boundary": "May produce promoted output; may not create extra work only because it is possible."
}
```

Turns promoted contracts into concrete output while preserving branch identity.
It must materialize the artifact kind named by the branch contract. A preparation
prompt, router wrapper, or explanatory chat answer is not a generated image,
HTML file, CSS file, audio file, transcript, or other concrete artifact unless
runtime artifact truth records it as that artifact kind.
