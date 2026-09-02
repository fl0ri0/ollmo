from ollmo_services.semantic_review_verdict import (
    semantic_review_verdict_freeze_acceptance,
    semantic_review_verdict_from_text,
)


def test_semantic_review_verdict_parses_strict_pass_json() -> None:
    verdict = semantic_review_verdict_from_text(
        '''
        {
          "kind": "ollmo.semantic_review_verdict",
          "verdict": "passed",
          "overall_status": "fulfilled",
          "whole_intent_fit": "The final text uses the generated image evidence.",
          "criterion_results": [
            {
              "criterion": "whole_turn_output_fits_current_user_intent",
              "status": "passed",
              "reason": "The runtime evidence is referenced.",
              "evidence_refs": ["branch-final-review"]
            }
          ],
          "evidence_refs": ["branch-final-review"],
          "defects": [],
          "confidence": 0.91,
          "recommended_transition": "truthful_freeze"
        }
        ''',
        branch_id='branch-global-semantic-closure-review',
    )

    assert verdict['verdict'] == 'passed'
    assert verdict['status'] == 'fulfilled'
    assert verdict['recommended_transition'] == 'truthful_freeze'
    assert verdict['criterion_results'][0]['status'] == 'passed'
    assert verdict['evidence_refs'] == ['branch-final-review']
    assert verdict['declared_schema']['kind'] == 'ollmo.semantic_review_verdict'
    assert verdict['declared_schema']['defects_is_empty_array'] is True

    acceptance = semantic_review_verdict_freeze_acceptance(
        verdict,
        required_criteria=['whole_turn_output_fits_current_user_intent'],
    )
    assert acceptance['accepted'] is True
    assert acceptance['status'] == 'accepted'

    reparsed = semantic_review_verdict_from_text(verdict)
    reparsed_acceptance = semantic_review_verdict_freeze_acceptance(
        reparsed,
        required_criteria=['whole_turn_output_fits_current_user_intent'],
    )
    assert reparsed_acceptance['accepted'] is True


def test_semantic_review_verdict_parses_fenced_failed_json() -> None:
    verdict = semantic_review_verdict_from_text(
        '''
        Review result:
        ```json
        {
          "verdict": "failed",
          "overall_status": "blocked",
          "whole_intent_fit": "The final text ignores the generated artifact.",
          "defects": ["missing generated artifact comparison"],
          "recommended_transition": "repair_dependency_chain",
          "confidence": 84
        }
        ```
        '''
    )

    assert verdict['verdict'] == 'failed'
    assert verdict['status'] == 'blocked'
    assert verdict['recommended_transition'] == 'repair_dependency_chain'
    assert verdict['confidence'] == 0.84
    assert verdict['defects'] == ['missing generated artifact comparison']


def test_semantic_review_verdict_keeps_legacy_headings_bounded() -> None:
    verdict = semantic_review_verdict_from_text(
        '''
        overall_status: needs_repair
        whole_intent_fit: The response does not compare both generated images.
        evidence_used: branch-image-1
        missing_or_wrong_work: branch-image-2 was ignored
        recommended_transition: repair_dependency_chain
        '''
    )

    assert verdict['verdict'] == 'failed'
    assert verdict['status'] == 'blocked'
    assert verdict['source_format'] == 'legacy_headings'
    assert verdict['recommended_transition'] == 'repair_dependency_chain'


def test_unparseable_semantic_review_verdict_does_not_pass() -> None:
    verdict = semantic_review_verdict_from_text('Looks good to me.')

    assert verdict['verdict'] == 'uncertain'
    assert verdict['status'] == 'pending'
    assert verdict['recommended_transition'] == 'manual_review'
    assert verdict['parse_status'] == 'missing_structured_verdict'


def test_status_only_verdict_remains_parseable_but_cannot_freeze() -> None:
    verdict = semantic_review_verdict_from_text('{"status":"completed"}')

    assert verdict['parse_status'] == 'parsed'
    assert verdict['verdict'] == 'passed'

    acceptance = semantic_review_verdict_freeze_acceptance(
        verdict,
        required_criteria=['whole_turn_output_fits_current_user_intent'],
    )
    assert acceptance['accepted'] is False
    assert acceptance['status'] == 'rejected'
    assert 'kind_not_explicit_or_invalid' in acceptance['rejection_reasons']
    assert 'verdict_not_explicit_passed' in acceptance['rejection_reasons']
    assert 'missing_evidence_refs' in acceptance['rejection_reasons']
    assert acceptance['missing_criteria'] == ['whole_turn_output_fits_current_user_intent']

    reparsed = semantic_review_verdict_from_text(verdict)
    reparsed_acceptance = semantic_review_verdict_freeze_acceptance(
        reparsed,
        required_criteria=['whole_turn_output_fits_current_user_intent'],
    )
    assert reparsed_acceptance['accepted'] is False
    assert 'missing_evidence_refs' in reparsed_acceptance['rejection_reasons']
    assert 'required_criteria_missing' in reparsed_acceptance['rejection_reasons']


def test_model_declared_schema_field_cannot_spoof_freeze_provenance() -> None:
    verdict = semantic_review_verdict_from_text(
        {
            'status': 'completed',
            'criterion_results': [
                {
                    'criterion': 'whole_turn_output_fits_current_user_intent',
                    'status': 'passed',
                    'evidence_refs': ['phase-artifact'],
                },
            ],
            'evidence_refs': ['phase-artifact'],
            'defects': [],
            'declared_schema': {
                'kind': 'ollmo.semantic_review_verdict',
                'verdict': 'passed',
                'recommended_transition': 'truthful_freeze',
                'defects_is_empty_array': True,
                'evidence_refs_is_array': True,
                'criterion_results_is_array': True,
            },
        }
    )

    acceptance = semantic_review_verdict_freeze_acceptance(
        verdict,
        required_criteria=['whole_turn_output_fits_current_user_intent'],
    )

    assert acceptance['accepted'] is False
    assert 'kind_not_explicit_or_invalid' in acceptance['rejection_reasons']
    assert 'verdict_not_explicit_passed' in acceptance['rejection_reasons']
    assert 'transition_not_explicit_truthful_freeze' in acceptance['rejection_reasons']


def test_truthful_freeze_requires_passed_coverage_of_every_required_criterion() -> None:
    verdict = semantic_review_verdict_from_text(
        '''
        {
          "kind": "ollmo.semantic_review_verdict",
          "verdict": "passed",
          "criterion_results": [
            {
              "criterion": "whole_turn_output_fits_current_user_intent",
              "status": "passed",
              "evidence_refs": ["branch-final-review"]
            }
          ],
          "evidence_refs": ["branch-final-review"],
          "defects": [],
          "recommended_transition": "truthful_freeze"
        }
        '''
    )

    acceptance = semantic_review_verdict_freeze_acceptance(
        verdict,
        required_criteria=[
            'whole_turn_output_fits_current_user_intent',
            'local_branch_outputs_are_used_in_their_declared_roles',
        ],
    )

    assert acceptance['accepted'] is False
    assert acceptance['missing_criteria'] == [
        'local_branch_outputs_are_used_in_their_declared_roles'
    ]
    assert 'required_criteria_missing' in acceptance['rejection_reasons']
