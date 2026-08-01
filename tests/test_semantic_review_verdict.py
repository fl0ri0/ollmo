from ollmo_services.semantic_review_verdict import semantic_review_verdict_from_text


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
