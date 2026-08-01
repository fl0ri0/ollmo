from ollmo_g.decision_contracts import build_ghost_decision_contract


def test_deterministic_review_criteria_do_not_create_semantic_review_candidates() -> None:
    contract = build_ghost_decision_contract(
        workload_graph={
            'tasks': [
                {
                    'task_id': 'task-simple-chat',
                    'phase_id': 'phase-1',
                    'branch_id': 'phase-1',
                    'capability': 'chat',
                    'status': 'fulfilled',
                    'review_criteria': [
                        'output_contract_matches_capability',
                        'runtime_status_reaches_fulfilled_blocked_failed_waived_superseded_or_pending',
                        'runtime_text_exists_when_fulfilled',
                    ],
                }
            ],
        },
    )

    assert contract.get('semantic_review_candidates', []) == []
    assert contract['semantic_quality_review']['status'] == 'not_required'
    assert contract.get('semantic_quality_contracts', []) == []
    assert 'run_semantic_quality_review_before_claiming_quality_truth' not in contract['next_decision_priorities']


def test_decision_contract_surfaces_reconsideration_supersession_repair_and_learning() -> None:
    contract = build_ghost_decision_contract(
        promotion_review={
            'counts': {'reserved': 1, 'promoted': 1},
            'decisions': [
                {
                    'candidate_id': 'candidate-image-option',
                    'candidate_type': 'output',
                    'decision': 'reserved',
                    'reconsiderable': True,
                    'reconsideration_policy': 'review_again_when_context_or_intent_changes',
                    'execution_policy': 'non_executable_until_promoted',
                }
            ],
        },
        output_obligations=[
            {
                'obligation_id': 'obligation-current',
                'phase_id': 'phase-1',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
            },
            {
                'obligation_id': 'obligation-old-image',
                'phase_id': 'phase-2',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'superseded',
                'superseded_by_obligation_id': 'obligation-new-image',
                'supersession_reason': 'newer image branch replaced the old one',
            },
        ],
        workload_graph={
            'tasks': [
                {
                    'task_id': 'task-final',
                    'phase_id': 'phase-3',
                    'capability': 'chat',
                    'status': 'pending',
                    'semantic_intent': 'Compare generated evidence only.',
                    'evidence_requirements': ['generated image evidence'],
                    'semantic_review_criteria': ['judges whether comparison uses both generated artifacts'],
                    'promotion_suggestions': [
                        {
                            'candidate_id': 'candidate-followup-summary',
                            'promotion_reason': 'user requested a final comparison after generated evidence exists',
                        }
                    ],
                    'waiver_candidates': [
                        {
                            'obligation_id': 'obligation-optional-caption',
                            'waiver_reason': 'caption was explicitly optional',
                        }
                    ],
                    'repair_candidates': [
                        {
                            'task_id': 'task-final',
                            'repair_action': 'repair_dependency_chain',
                            'reason': 'missing generated evidence',
                        }
                    ],
                    'supersession_candidates': [
                        {
                            'obligation_id': 'obligation-old-image',
                            'superseded_by_obligation_id': 'obligation-new-image',
                            'supersession_reason': 'replacement branch is newer truth',
                        }
                    ],
                }
            ],
        },
        workload_proposal_review={'coverage': {'status': 'complete'}},
        accepted_learning_hints={
            'status': 'active',
            'enabled': True,
            'authority': 'soft_hint',
            'runtime_effect': 'soft_hints_available',
            'hint_count': 1,
            'hints': [
                {
                    'learning_id': 'accepted-policy-improvement-workload',
                    'candidate_id': 'policy-improvement-workload_decision_policy',
                    'target_area': 'workload_decision_policy',
                    'authority': 'soft_hint',
                    'allowed_use': 'soft_hint_only',
                }
            ],
        },
    )

    assert contract['kind'] == 'ollmo.ghost_decision_contract'
    assert contract['decision_contract_version'] == 11
    assert contract['open_obligation_ids'] == ['obligation-current']
    assert contract['closed_obligation_ids'] == ['obligation-old-image']
    assert contract['reconsideration_candidates'][0]['candidate_id'] == 'candidate-image-option'
    assert contract['supersession_records'][0]['superseded_by_obligation_id'] == 'obligation-new-image'
    assert contract['semantic_review_candidates'][0]['review_criteria'] == [
        'judges whether comparison uses both generated artifacts'
    ]
    assert contract['promotion_suggestions'][0]['promotion_suggestions'][0]['candidate_id'] == 'candidate-followup-summary'
    assert contract['waiver_candidates'][0]['waiver_candidates'][0]['obligation_id'] == 'obligation-optional-caption'
    assert contract['repair_candidates'][0]['repair_candidates'][0]['repair_action'] == 'repair_dependency_chain'
    assert contract['supersession_candidates'][0]['supersession_candidates'][0]['obligation_id'] == 'obligation-old-image'
    reflex = contract['block_resolution_reflex']
    assert reflex['kind'] == 'ollmo.block_resolution_reconsideration_reflex'
    assert reflex['status'] == 'active'
    assert reflex['authority'] == 'advisory_read_model_only'
    assert reflex['open_obligation_count'] == 1
    assert reflex['reconsiderable_candidate_count'] == 1
    assert reflex['category_counts']['open_obligation'] == 1
    assert reflex['category_counts']['superseded_obligation'] == 1
    assert any(
        item['category'] == 'reconsiderable_candidate'
        and item['action'] == 'keep_visible_for_future_relevance_review_without_execution'
        for item in contract['reconsideration_reflex_signals']
    )
    active_reconsideration = contract['active_reconsideration_review']
    assert active_reconsideration['kind'] == 'ollmo.active_reconsideration_review'
    assert active_reconsideration['status'] == 'active'
    assert active_reconsideration['decision_count'] == reflex['signal_count']
    assert any(
        item['source_category'] == 'reconsiderable_candidate'
        and item['review_type'] == 'promotion_relevance_review'
        for item in contract['active_reconsideration_decisions']
    )
    semantic_quality = contract['semantic_quality_review']
    assert semantic_quality['kind'] == 'ollmo.semantic_quality_review'
    assert semantic_quality['status'] == 'required'
    assert contract['semantic_quality_contracts'][0]['status'] == 'pending_semantic_review'
    assert contract['semantic_quality_contracts'][0]['review_policy'] == 'quality_is_not_proven_by_output_existence'
    assert contract['semantic_quality_contracts'][0]['semantic_review_lens'] == 'quality_reviewer'
    assert contract['semantic_quality_contracts'][0]['success_definition']
    assert 'artifact_exists_but_criterion_unproven' in contract['semantic_quality_contracts'][0]['failure_modes']
    lens_review = contract['semantic_review_lens_review']
    assert lens_review['kind'] == 'ollmo.semantic_review_lens_review'
    assert lens_review['status'] == 'active'
    assert lens_review['authority'] == 'advisory_read_model_only'
    assert lens_review['lens_count'] == len(contract['semantic_review_lenses'])
    assert lens_review['lens_counts']['quality_reviewer'] >= 1
    assert any(
        item['lens'] == 'quality_reviewer'
        and item['semantic_role_id'] == 'quality_reviewer'
        and item['authority'] == 'advisory_read_model_only'
        for item in contract['semantic_review_lenses']
    )
    recursive_cycle = contract['recursive_cycle_review']
    assert recursive_cycle['kind'] == 'ollmo.recursive_cycle_review'
    assert recursive_cycle['status'] == 'active'
    assert recursive_cycle['task_count'] == 1
    assert contract['recursive_cycle_tasks'][0]['cycle_policy'] == 'prepare_gather_execute_verify_repair_or_freeze'
    aspiration = contract['aspiration_review']
    assert aspiration['kind'] == 'ollmo.aspiration_review'
    assert aspiration['status'] == 'active'
    assert aspiration['authority'] == 'advisory_read_model_only'
    assert aspiration['frame_count'] == len(contract['aspiration_frames'])
    assert any(
        item['source_kind'] == 'aspiration_frame'
        and item['aspiration_action'] in {'preserve_possibility_space', 'expand_candidate_space', 'raise_solution_bar'}
        and item['non_authority_boundary'] == 'aspiration_only_runtime_contracts_closure_decide_truth'
        for item in contract['aspiration_frames']
    )
    commitment = contract['commitment_review']
    assert commitment['kind'] == 'ollmo.commitment_review'
    assert commitment['status'] == 'active'
    assert commitment['authority'] == 'advisory_read_model_only'
    assert commitment['frame_count'] == len(contract['commitment_frames'])
    assert any(
        item['source_kind'] == 'commitment_frame'
        and item['recommended_transition'] in {'continue_branch_local_work', 'semantic_review', 'keep_reserved_or_promote_after_relevance_review'}
        and item['non_authority_boundary'] == 'commitment_only_runtime_contracts_closure_decide_truth'
        for item in contract['commitment_frames']
    )
    semantic_decision = contract['semantic_decision_review']
    assert semantic_decision['kind'] == 'ollmo.semantic_decision_review'
    assert semantic_decision['status'] == 'active'
    assert semantic_decision['authority'] == 'advisory_read_model_only'
    assert semantic_decision['proposal_count'] == len(contract['semantic_decision_proposals'])
    assert any(
        item['decision_action'] == 'semantic_review'
        and item['source_kind'] == 'semantic_quality_contract'
        for item in contract['semantic_decision_proposals']
    )
    assert any(
        item['decision_action'] == 'keep_reserved_or_promote_after_relevance_review'
        for item in contract['semantic_decision_proposals']
    )
    assert any(
        item['source_kind'] == 'aspiration_frame'
        for item in contract['semantic_decision_proposals']
    )
    assert any(
        item['source_kind'] == 'commitment_frame'
        for item in contract['semantic_decision_proposals']
    )
    attention = contract['controlled_attention_review']
    assert attention['kind'] == 'ollmo.controlled_attention_review'
    assert attention['status'] == 'active'
    assert attention['authority'] == 'advisory_read_model_only'
    assert attention['frame_count'] == len(contract['controlled_attention_frames'])
    assert attention['scope_counts']['candidate_relevance'] >= 1
    assert attention['scope_counts']['semantic_quality'] >= 1
    assert attention['scope_counts']['aspiration_possibility'] >= 1
    assert attention['scope_counts']['commitment_transition'] >= 1
    assert any(
        item['scope'] == 'candidate_relevance'
        and 'promote_after_current_relevance_review' in item['allowed_transitions']
        and item['non_authority_boundary'] == 'attention_only_runtime_contracts_closure_decide_truth'
        for item in contract['controlled_attention_frames']
    )
    assert any(
        item['source_kind'] == 'semantic_quality_contract'
        and item['attention_question'] == 'What semantic evidence is needed before quality can truthfully freeze?'
        and item['semantic_review_lens'] == 'quality_reviewer'
        and item['success_definition']
        for item in contract['controlled_attention_frames']
    )
    assert contract['accepted_learning']['allowed_use'] == 'orientation_only_not_promotion_authority'
    assert contract['accepted_learning']['hint_count'] == 1
    planning = contract['semantic_planning_contract']
    assert planning['kind'] == 'ollmo.ghost_semantic_planning_contract'
    assert planning['authority'] == 'advisory_read_model_only'
    assert 'promotion_suggestions' in planning['task_proposal_fields']
    assert 'waiver_candidates' in planning['task_proposal_fields']
    assert planning['block_resolution_reflex']['signal_count'] == reflex['signal_count']
    assert planning['active_reconsideration_review']['decision_count'] == reflex['signal_count']
    assert planning['semantic_quality_review']['contract_count'] == 1
    assert planning['recursive_cycle_review']['task_count'] == 1
    assert planning['aspiration_review']['frame_count'] == aspiration['frame_count']
    assert planning['commitment_review']['frame_count'] == commitment['frame_count']
    assert planning['semantic_decision_review']['proposal_count'] == semantic_decision['proposal_count']
    assert planning['controlled_attention_review']['frame_count'] == attention['frame_count']
    assert planning['semantic_review_lens_review']['lens_count'] == lens_review['lens_count']
    assert 'review_advisory_promotion_suggestions_before_promoting_work' in planning['current_focus']
    assert 'apply_block_resolution_reconsideration_reflex_between_steps' in planning['current_focus']
    assert 'review_active_reconsideration_decisions_before_state_change' in planning['current_focus']
    assert 'run_semantic_quality_review_before_claiming_quality_truth' in planning['current_focus']
    assert 'apply_recursive_mini_cycle_per_subtask' in planning['current_focus']
    assert 'use_aspiration_review_to_keep_possibility_and_solution_bar_visible' in planning['current_focus']
    assert 'use_commitment_review_to_choose_the_right_sized_sufficient_transition' in planning['current_focus']
    assert 'review_semantic_decision_proposals_before_state_transition' in planning['current_focus']
    assert 'use_controlled_model_attention_between_execution_steps' in planning['current_focus']
    assert 'apply_semantic_review_lenses_to_branch_expectations' in planning['current_focus']
    assert any(
        item['action'] == 'review_advisory_waiver_candidates_against_explicit_release_evidence'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'apply_block_resolution_reconsideration_reflex_between_steps'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'review_active_reconsideration_decisions_before_changing_contract_state'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'treat_semantic_quality_as_pending_review_work'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'apply_prepare_gather_execute_verify_repair_or_freeze_cycle_per_subtask'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'use_aspiration_review_to_keep_possibility_and_solution_bar_visible'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'use_commitment_review_to_choose_the_right_sized_sufficient_transition'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'use_semantic_decision_review_as_advisory_next_transition_input'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'use_controlled_attention_frames_as_scoped_prompt_targets'
        for item in planning['proposal_obligations']
    )
    assert any(
        item['action'] == 'apply_semantic_review_lenses_to_expectation_success_and_evidence_checks'
        for item in planning['proposal_obligations']
    )


def test_decision_contract_reflex_distinguishes_blocked_and_waived_truth() -> None:
    contract = build_ghost_decision_contract(
        promotion_review={
            'decisions': [
                {
                    'candidate_id': 'candidate-old-audio',
                    'candidate_type': 'output',
                    'decision': 'waived',
                    'reason': 'user explicitly released the audio',
                }
            ],
        },
        output_obligations=[
            {
                'obligation_id': 'obligation-audio',
                'phase_id': 'phase-audio',
                'branch_id': 'branch-audio',
                'capability': 'text_to_speech',
                'output_type': 'audio',
                'status': 'blocked',
            },
            {
                'obligation_id': 'obligation-caption',
                'phase_id': 'phase-caption',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'waived',
            },
        ],
    )

    reflex = contract['block_resolution_reflex']
    assert reflex['blocked_obligation_count'] == 1
    assert reflex['category_counts']['blocked_obligation'] == 1
    assert reflex['category_counts']['waived_obligation'] == 1
    assert reflex['category_counts']['waived_candidate'] == 1
    blocked_signal = next(
        item for item in reflex['signals']
        if item.get('category') == 'blocked_obligation'
    )
    assert blocked_signal['action'] == 'resolve_block_from_dependency_contract_waiver_supersession_or_truthful_freeze'
    assert blocked_signal['resolution_policy'] == 'right_sized_verified_state_transition'
    assert blocked_signal['principle'] == 'the_solution_to_a_block_is_the_blocks_own_resolution'
    blocked_decision = next(
        item for item in contract['active_reconsideration_decisions']
        if item.get('source_category') == 'blocked_obligation'
    )
    assert blocked_decision['review_type'] == 'block_resolution_review'
    assert 'repair_dependency_chain' in blocked_decision['allowed_outcomes']
    blocked_proposal = next(
        item for item in contract['semantic_decision_proposals']
        if item.get('obligation_id') == 'obligation-audio'
    )
    assert blocked_proposal['decision_action'] == 'repair_dependency_chain'
    assert blocked_proposal['authority'] == 'advisory_read_model_only'


def test_decision_contract_compiles_semantic_roles_into_advisory_orientation() -> None:
    contract = build_ghost_decision_contract(
        semantic_role_profile={
            'kind': 'ollmo.semantic_role_profile',
            'mode': 'improviser',
            'mode_source': 'request',
            'semantic_role_ids': ['possibility_expander', 'quality_reviewer'],
            'semantic_role_orientation': {
                'kind': 'ollmo.semantic_role_orientation',
                'authority': 'advisory_read_model_only',
                'mode': 'improviser',
                'mode_source': 'request',
                'reason': 'semantic roles may orient attention inside the promoted contract',
                'suggested_semantic_review_lenses': ['possibility_expander', 'quality_reviewer'],
                'attention_biases': ['style_surface', 'solution_bar'],
                'evidence_refs': ['semantic_role_profile', 'request'],
            },
        },
    )

    review = contract['semantic_role_orientation_review']
    assert review['status'] == 'active'
    assert review['authority'] == 'advisory_read_model_only'
    assert review['frame_count'] == 2
    assert contract['semantic_role_orientation_frames'][0]['source_kind'] == 'semantic_role_orientation_frame'
    assert contract['semantic_role_orientation_frames'][0]['authority'] == 'advisory_read_model_only'
    assert any(
        proposal['source_kind'] == 'semantic_role_orientation_frame'
        and proposal['decision_action'] == 'orient_attention_only'
        for proposal in contract['semantic_decision_proposals']
    )
    assert any(
        frame['source_kind'] == 'semantic_role_orientation_frame'
        and frame['scope'] == 'semantic_role_orientation'
        for frame in contract['controlled_attention_frames']
    )
    assert any(
        lens['source_kind'] == 'semantic_role_orientation_frame'
        and lens['authority'] == 'advisory_read_model_only'
        and lens['semantic_role_id']
        for lens in contract['semantic_review_lenses']
    )
    planning = contract['semantic_planning_contract']
    assert any(
        item['action'] == 'use_semantic_roles_as_advisory_orientation_only'
        for item in planning['proposal_obligations']
    )
    assert 'do_not_treat_semantic_roles_as_planner_timeout_branching_or_payload_authority' in planning['non_authority_boundaries']


def test_decision_contract_aspiration_reviews_underplanned_workload_coverage() -> None:
    contract = build_ghost_decision_contract(
        candidate_graph={
            'candidate_count': 1,
            'candidates': [
                {
                    'candidate_id': 'candidate-html',
                    'candidate_type': 'output',
                    'status': 'promoted',
                    'output_type': 'text',
                }
            ],
        },
        promotion_review={'counts': {'promoted': 1}},
        output_obligations=[
            {
                'obligation_id': 'obligation-html',
                'phase_id': 'phase-html',
                'branch_id': 'branch-html',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
            }
        ],
        workload_proposal_review={
            'coverage': {
                'status': 'partial',
                'missing_task_ids': ['task-css'],
            }
        },
    )

    aspiration = contract['aspiration_review']
    assert aspiration['status'] == 'active'
    assert aspiration['frames'][0]['aspiration_action'] == 'review_underplanned_graph'
    assert aspiration['frames'][0]['task_id'] == 'task-css'
    assert 'review_underplanned_graph' in aspiration['frames'][0]['allowed_actions']
    assert any(
        proposal['source_kind'] == 'aspiration_frame'
        and proposal['decision_action'] == 'review_underplanned_graph'
        for proposal in contract['semantic_decision_proposals']
    )
    assert any(
        frame['source_kind'] == 'aspiration_frame'
        and frame['scope'] == 'aspiration_possibility'
        for frame in contract['controlled_attention_frames']
    )
