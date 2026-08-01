import unittest

from ollmo_g.candidate_contracts import (
    build_candidate_graph,
    normalize_candidate,
    review_candidate_promotions,
)


class CandidateContractTests(unittest.TestCase):
    def test_normalize_candidate_keeps_possibility_without_obligation(self):
        candidate = normalize_candidate(
            {
                'candidate_id': 'candidate-image-1',
                'capability': 'image_generation',
                'output_type': 'image',
                'status': 'reserved',
                'promotion_policy': 'requires_user_confirmation',
                'reason': 'possible visual direction',
            },
            source='request_ir.output_candidates',
            intent_ref='intent_anchor',
            candidate_type='output',
        )

        self.assertEqual(candidate['kind'], 'ollmo.candidate')
        self.assertEqual(candidate['candidate_type'], 'output')
        self.assertEqual(candidate['status'], 'reserved')
        self.assertEqual(candidate['promotion_policy'], 'requires_user_confirmation')
        self.assertTrue(candidate['reconsiderable'])
        self.assertEqual(candidate['execution_policy'], 'non_executable_until_promoted')
        self.assertNotIn('contract_ref', candidate)

    def test_promotion_review_turns_only_promoted_candidates_into_contracts(self):
        candidate_graph = build_candidate_graph(
            output_candidates=[
                {
                    'candidate_id': 'candidate-image-1',
                    'phase_id': 'phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'promoted',
                    'promoted_obligation_id': 'obligation-phase-2',
                    'promotion_reason': 'user asked to generate it now',
                },
                {
                    'candidate_id': 'candidate-audio-1',
                    'phase_id': 'phase-3',
                    'capability': 'text_to_speech',
                    'output_type': 'audio',
                    'status': 'reserved',
                },
            ],
            output_obligations=[
                {
                    'obligation_id': 'obligation-phase-2',
                    'promoted_from_candidate_id': 'candidate-image-1',
                    'phase_id': 'phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'pending',
                }
            ],
            intent_ref='intent_anchor',
        )
        review = review_candidate_promotions(candidate_graph)

        self.assertEqual(candidate_graph['candidate_count'], 2)
        self.assertEqual(review['promoted_count'], 1)
        self.assertEqual(review['reserved_count'], 1)
        promoted = next(item for item in review['decisions'] if item['candidate_id'] == 'candidate-image-1')
        reserved = next(item for item in review['decisions'] if item['candidate_id'] == 'candidate-audio-1')
        self.assertEqual(promoted['decision'], 'promoted')
        self.assertEqual(promoted['contract_ref'], 'obligation-phase-2')
        self.assertEqual(promoted['execution_policy'], 'executable_obligation')
        self.assertEqual(reserved['decision'], 'reserved')
        self.assertTrue(reserved['reconsiderable'])
        self.assertEqual(reserved['execution_policy'], 'non_executable_until_promoted')
        self.assertNotIn('contract_ref', reserved)

    def test_superseded_promoted_candidate_closes_contract_without_work(self):
        candidate_graph = build_candidate_graph(
            output_candidates=[
                {
                    'candidate_id': 'candidate-image-1',
                    'phase_id': 'phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'promoted',
                    'promoted_obligation_id': 'obligation-phase-2',
                },
            ],
            output_obligations=[
                {
                    'obligation_id': 'obligation-phase-2',
                    'promoted_from_candidate_id': 'candidate-image-1',
                    'phase_id': 'phase-2',
                    'capability': 'image_generation',
                    'output_type': 'image',
                    'status': 'superseded',
                    'superseded_by_obligation_id': 'obligation-phase-3',
                    'supersession_reason': 'newer visual branch replaced this one',
                },
            ],
            intent_ref='intent_anchor',
        )
        review = review_candidate_promotions(candidate_graph)

        self.assertEqual(candidate_graph['candidate_count'], 1)
        self.assertEqual(candidate_graph['status_counts']['superseded'], 1)
        self.assertEqual(review['superseded_count'], 1)
        self.assertEqual(review['promoted_count'], 0)
        decision = review['decisions'][0]
        self.assertEqual(decision['decision'], 'superseded')
        self.assertTrue(decision['terminal'])
        self.assertEqual(decision['superseded_by_obligation_id'], 'obligation-phase-3')
        self.assertEqual(decision['supersession_policy'], 'closed_by_current_runtime_truth')

    def test_workload_rejections_are_visible_without_creating_work(self):
        candidate_graph = build_candidate_graph(
            workload_tasks=[
                {
                    'task_id': 'task-phase-1',
                    'phase_id': 'phase-1',
                    'capability': 'chat',
                    'status': 'fulfilled',
                }
            ],
            workload_proposal_review={
                'kind': 'ollmo.workload_proposal_review',
                'status': 'rejected',
                'rejections': [
                    {
                        'proposal_id': 'bad-edge',
                        'target': 'task-phase-2',
                        'reason': 'target_task_not_found',
                    }
                ],
            },
        )
        review = review_candidate_promotions(candidate_graph)

        rejected = [
            item for item in review['decisions']
            if item['decision'] == 'rejected'
        ]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['reason'], 'target_task_not_found')
        self.assertEqual(review['rejected_count'], 1)

    def test_workload_task_candidates_preserve_decision_metadata(self):
        candidate_graph = build_candidate_graph(
            workload_tasks=[
                {
                    'task_id': 'task-phase-5',
                    'phase_id': 'phase-5',
                    'capability': 'chat',
                    'status': 'pending',
                    'semantic_intent': 'Join generated evidence.',
                    'advisory_role': 'semantic join planner',
                    'evidence_requirements': ['generated audio evidence', 'generated image evidence'],
                    'semantic_review_criteria': ['uses both generated artifacts'],
                    'promotion_suggestions': [
                        {
                            'candidate_id': 'candidate-final-join',
                            'promotion_reason': 'generated evidence is ready',
                        }
                    ],
                    'waiver_candidates': [
                        {
                            'obligation_id': 'obligation-phase-3',
                            'waiver_reason': 'intermediate text is replaced by final join',
                        }
                    ],
                    'repair_candidates': [
                        {
                            'task_id': 'task-phase-5',
                            'repair_action': 'repair_dependency_chain',
                            'reason': 'missing dependency evidence',
                        }
                    ],
                    'supersession_candidates': [
                        {
                            'obligation_id': 'obligation-phase-3',
                            'superseded_by_obligation_id': 'obligation-phase-5',
                        }
                    ],
                }
            ],
        )

        candidate = candidate_graph['candidates'][0]
        self.assertEqual(candidate['advisory_role'], 'semantic join planner')
        self.assertEqual(candidate['evidence_requirements'], ['generated audio evidence', 'generated image evidence'])
        self.assertEqual(candidate['semantic_review_criteria'], ['uses both generated artifacts'])
        self.assertEqual(candidate['promotion_suggestions'][0]['candidate_id'], 'candidate-final-join')
        self.assertEqual(candidate['waiver_candidates'][0]['obligation_id'], 'obligation-phase-3')
        self.assertEqual(candidate['repair_candidates'][0]['repair_action'], 'repair_dependency_chain')
        self.assertEqual(candidate['supersession_candidates'][0]['obligation_id'], 'obligation-phase-3')

    def test_context_candidate_promotes_to_active_reference_contract(self):
        candidate_graph = build_candidate_graph(
            context_candidates=[
                {
                    'candidate_id': 'ctx-image',
                    'source_kind': 'artifact',
                    'status': 'promoted',
                    'promotion_target': 'active_reference',
                    'artifact_ref': 'artifact:image_previous',
                    'promotion_reason': 'selected by current turn',
                }
            ],
            intent_ref='intent_anchor',
        )
        review = review_candidate_promotions(candidate_graph)

        self.assertEqual(candidate_graph['type_counts']['reference'], 1)
        self.assertEqual(review['promoted_count'], 1)
        self.assertEqual(review['decisions'][0]['contract_ref'], 'active_reference')


if __name__ == '__main__':
    unittest.main()
