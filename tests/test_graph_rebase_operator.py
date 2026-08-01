import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ollmo_services.graph_rebase import (
    apply_validated_graph_rebase,
    build_graph_rebase_lifecycle,
    build_graph_rebase_proposal,
    stable_graph_digest,
    validate_graph_rebase_proposal,
)
from ollmo_services.graph_rebase_operator import (
    GraphRebaseOperatorRegistryError,
    find_trusted_graph_rebase_authorization,
    load_graph_rebase_operator_records,
    record_graph_rebase_operator_action,
    stable_graph_rebase_operator_record_id,
    verify_graph_rebase_operator_record,
)


class GraphRebaseOperatorRegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self._tmpdir.name) / 'operator_reviews.jsonl'

    def tearDown(self):
        self._tmpdir.cleanup()

    def _base_graph(self):
        return {
            'kind': 'ollmo.request_phase_graph',
            'graph_version': 3,
            'response_id': 'resp-operator-review',
            'frame_id': 'frame-graph-base',
            'phases': [
                {
                    'phase_id': 'phase-root',
                    'branch_id': 'branch-root',
                    'obligation_id': 'obligation-root',
                    'capability': 'chat',
                    'output_type': 'text',
                    'status': 'completed',
                }
            ],
            'downstream_branches': [],
            'intent_obligations': [
                {
                    'obligation_id': 'intent-root',
                    'phase_id': 'phase-root',
                    'kind': 'text',
                    'required': True,
                }
            ],
            'output_obligations': [
                {
                    'obligation_id': 'obligation-root',
                    'phase_id': 'phase-root',
                    'capability': 'chat',
                    'output_type': 'text',
                    'required': True,
                }
            ],
        }

    def _candidate_graph(self):
        candidate = copy.deepcopy(self._base_graph())
        candidate['phases'].append(
            {
                'phase_id': 'phase-review',
                'branch_id': 'branch-review',
                'obligation_id': 'obligation-review',
                'capability': 'chat',
                'output_type': 'text',
                'status': 'pending',
                'depends_on': ['phase-root'],
                'content_payload': 'Review the exact saved branch result for structural closure.',
                'content_payload_source': 'runtime_local_contract',
                'execution_contract': {
                    'execution_scope': 'branch_local',
                    'input_refs': [
                        {
                            'kind': 'runtime_evidence',
                            'ref': 'closure:intent_graph_adequacy',
                        }
                    ],
                },
                'lineage': {
                    'parent_phase_id': 'phase-root',
                    'relation': 'split_branch',
                },
            }
        )
        candidate['downstream_branches'].append(
            {
                'phase_id': 'phase-review',
                'branch_id': 'branch-review',
                'capability': 'chat',
                'output_type': 'text',
                'depends_on': ['phase-root'],
                'content_payload': 'Review the exact saved branch result for structural closure.',
                'content_payload_source': 'runtime_local_contract',
                'execution_contract': {
                    'execution_scope': 'branch_local',
                    'input_refs': [
                        {
                            'kind': 'runtime_evidence',
                            'ref': 'closure:intent_graph_adequacy',
                        }
                    ],
                },
                'lineage': {
                    'parent_phase_id': 'phase-root',
                    'relation': 'split_branch',
                },
            }
        )
        candidate['output_obligations'].append(
            {
                'obligation_id': 'obligation-review',
                'phase_id': 'phase-review',
                'capability': 'chat',
                'output_type': 'text',
                'required': True,
                'lineage': {
                    'parent_obligation_id': 'obligation-root',
                    'relation': 'split_branch',
                },
            }
        )
        return candidate

    def _payload(
        self,
        *,
        requested_rebase_class='partial_subtree_rebase',
        staged=False,
        frame_id='resp-operator-review:frame-1',
        frame_sequence=1,
    ):
        graph = self._base_graph()
        graph['redraw_scope_ladder_review'] = {
            'kind': 'ollmo.redraw_scope_ladder_review',
            'status': 'selected',
            'selected_scope': requested_rebase_class,
        }
        proposal_kwargs = {}
        if requested_rebase_class == 'partial_subtree_rebase':
            proposal_kwargs = {
                'scope_root_ids': ['obligation-root', 'obligation-review'],
                'scope_phase_ids': ['phase-root', 'phase-review'],
                'scope_branch_ids': ['phase-root', 'branch-review'],
                'preserve_outside_scope': True,
            }
        proposal = build_graph_rebase_proposal(
            request_phase_graph=graph,
            candidate_graph=self._candidate_graph(),
            target_response_id='resp-operator-review',
            target_frame_id='frame-graph-base',
            source='runtime_closure_review',
            reason='Current Closure requires a structural successor.',
            evidence_refs=['closure:intent_graph_adequacy'],
            requested_rebase_class=requested_rebase_class,
            root_prompt='Create the operator-review graph from the original request.',
            **proposal_kwargs,
        )
        closure_review = {
            'kind': 'ollmo.graph_closure_review',
            'status': 'repair_required',
        }
        review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=closure_review,
            root_prompt='Create the operator-review graph from the original request.',
        )
        self.assertEqual(review['status'], 'accepted', review.get('blocked_reasons'))
        self.assertEqual(review['preservation_proof']['status'], 'passed')
        graph['graph_rebase_proposals'] = [proposal]
        graph['graph_rebase_reviews'] = [review]
        if staged:
            lifecycle = build_graph_rebase_lifecycle(
                request_phase_graph=graph,
                rebase_review=review,
                autonomy_level='stage',
            )
            application = apply_validated_graph_rebase(
                graph,
                lifecycle,
                autonomy_level='stage',
            )
            graph = application['graph']
            self.assertEqual(len(graph['staged_graph_rebases']), 1)
        payload = {
            'id': 'resp-operator-review',
            'response_id': 'resp-operator-review',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'late_fill': {
                'status': 'completed',
                'active_count': 0,
                'pending_count': 0,
            },
            'request': {
                'prompt': 'Create the operator-review graph from the original request.',
            },
            'response_frame': {
                'kind': 'ollmo.response_frame',
                'response_id': 'resp-operator-review',
                'frame_id': frame_id,
                'frame_sequence': frame_sequence,
                'lifecycle_state': 'completed',
            },
            'runtime': {
                'request_phase_graph': graph,
                'graph_closure_review': closure_review,
            },
        }
        return payload

    def _expected(self, payload):
        graph = payload['runtime']['request_phase_graph']
        proposal = graph['graph_rebase_proposals'][0]
        return {
            'expected_response_id': payload['response_id'],
            'expected_frame_id': payload['response_frame']['frame_id'],
            'expected_frame_sequence': payload['response_frame']['frame_sequence'],
            'expected_proposal_id': proposal['proposal_id'],
            'expected_base_graph_digest': stable_graph_digest(graph),
            'expected_candidate_graph_digest': proposal['candidate_graph_digest'],
            'expected_requested_rebase_class': proposal['requested_rebase_class'],
        }

    def _false_negative_payload(
        self,
        *,
        frame_id='resp-operator-review:frame-missed',
        frame_sequence=1,
    ):
        payload = self._payload(frame_id=frame_id, frame_sequence=frame_sequence)
        graph = payload['runtime']['request_phase_graph']
        proposal = graph['graph_rebase_proposals'][0]
        graph.pop('graph_rebase_proposals', None)
        graph.pop('graph_rebase_reviews', None)
        payload['runtime']['developer_diagnostics'] = {
            'runtime_graph_rebase_candidate_review': {
                'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                'status': 'not_proposed',
                'reason': 'current_structural_closure_evidence_missing',
                'base_graph_digest': stable_graph_digest(graph),
                'candidate_graph_digest': proposal['candidate_graph_digest'],
                'runtime_effect': 'none',
            }
        }
        expected = {
            'expected_response_id': payload['response_id'],
            'expected_frame_id': payload['response_frame']['frame_id'],
            'expected_frame_sequence': payload['response_frame']['frame_sequence'],
            'expected_proposal_id': 'no_formal_proposal',
            'expected_base_graph_digest': stable_graph_digest(graph),
            'expected_candidate_graph_digest': proposal['candidate_graph_digest'],
            'expected_requested_rebase_class': 'partial_subtree_rebase',
        }
        return payload, expected

    def _record(
        self,
        payload,
        *,
        action,
        adjudication,
        reason,
        evidence_refs,
        gate=None,
        resolves_record_id='',
    ):
        return record_graph_rebase_operator_action(
            payload,
            action=action,
            adjudication=adjudication,
            reason=reason,
            evidence_refs=evidence_refs,
            resolves_record_id=resolves_record_id,
            trusted_partial_promotion_gate=gate,
            operator_identity='test-operator',
            registry_path=self.registry_path,
            **self._expected(payload),
        )

    def _useful_adjudication(self, payload):
        return self._record(
            payload,
            action='adjudicate',
            adjudication='useful_proposal',
            reason='The structural proposal is useful and preserves current truth.',
            evidence_refs=['operator:useful-proposal-review'],
        )

    def _stage(self, payload):
        return self._record(
            payload,
            action='stage',
            adjudication='accepted',
            reason='Record an audit-only stage before any reviewed execution.',
            evidence_refs=['operator:stage-review'],
        )

    def _promotion_gate(self, **updates):
        gate = {
            'kind': 'ollmo.graph_rebase_promotion_gate',
            'gate_id': 'partial-gate-2026-07-19',
            'gate': 'partial_stage_to_apply_reviewed',
            'status': 'ready',
            'decision': 'promote',
            'evidence_refs': ['readiness:partial-stage-corpus'],
            'policy_digest': 'policy-sha256:1234abcd',
        }
        gate.update(updates)
        return gate

    def test_adjudication_is_content_addressed_append_only_and_idempotent(self):
        payload = self._payload()
        first = self._useful_adjudication(payload)
        second = self._useful_adjudication(payload)

        self.assertEqual(second, first)
        self.assertTrue(verify_graph_rebase_operator_record(first))
        self.assertEqual(
            first['record_id'],
            stable_graph_rebase_operator_record_id(first),
        )
        records = load_graph_rebase_operator_records(
            registry_path=self.registry_path
        )
        self.assertEqual(records, [first])
        self.assertEqual(
            len(self.registry_path.read_text(encoding='utf-8').splitlines()),
            1,
        )
        self.assertTrue(first['replay_verified'])
        self.assertEqual(first['replay_status'], 'matched')
        self.assertEqual(first['operator_identity'], 'test-operator')

    def test_partial_operator_action_requires_current_durable_root_truth(self):
        payload = self._payload()
        payload.pop('request')

        with self.assertRaises(GraphRebaseOperatorRegistryError) as raised:
            self._useful_adjudication(payload)

        self.assertEqual(
            raised.exception.code,
            'partial_rebase_current_root_prompt_truth_unavailable',
        )

    def test_partial_operator_action_rejects_root_guard_drift(self):
        payload = self._payload()
        payload['request']['prompt'] = 'A different current durable root request.'

        with self.assertRaises(GraphRebaseOperatorRegistryError) as raised:
            self._useful_adjudication(payload)

        self.assertEqual(
            raised.exception.code,
            'partial_rebase_root_prompt_guard_mismatch',
        )

    def test_operator_action_requires_explicit_terminal_lifecycle_truth(self):
        payload = self._payload()
        payload.pop('status', None)
        payload.pop('lifecycle_state', None)
        payload['response_frame'].pop('lifecycle_state', None)

        with self.assertRaises(GraphRebaseOperatorRegistryError) as raised:
            self._useful_adjudication(payload)

        self.assertEqual(
            raised.exception.code,
            'nonterminal_response_frame_forbidden',
        )
        self.assertFalse(self.registry_path.exists())

    def test_false_negative_requires_explicit_terminal_lifecycle_truth(self):
        payload, expected = self._false_negative_payload()
        payload.pop('status', None)
        payload.pop('lifecycle_state', None)
        payload['response_frame'].pop('lifecycle_state', None)

        with self.assertRaises(GraphRebaseOperatorRegistryError) as raised:
            record_graph_rebase_operator_action(
                payload,
                action='adjudicate',
                adjudication='false_negative',
                reason='Missing lifecycle truth must never qualify as settled.',
                evidence_refs=['operator:missing-lifecycle'],
                operator_identity='test-operator',
                registry_path=self.registry_path,
                **expected,
            )

        self.assertEqual(
            raised.exception.code,
            'nonterminal_response_frame_forbidden',
        )
        self.assertFalse(self.registry_path.exists())

    def test_false_negative_adjudication_binds_settled_candidate_without_proposal(self):
        payload, expected = self._false_negative_payload()

        record = record_graph_rebase_operator_action(
            payload,
            action='adjudicate',
            adjudication='false_negative',
            reason='This settled candidate should have produced a partial proposal.',
            evidence_refs=['operator:false-negative-review'],
            operator_identity='test-operator',
            registry_path=self.registry_path,
            **expected,
        )

        self.assertEqual(record['adjudication'], 'false_negative')
        self.assertNotIn('proposal_id', record)
        self.assertTrue(record['candidate_observation_id'])
        self.assertNotIn('replay_verified', record)
        self.assertTrue(verify_graph_rebase_operator_record(record))

    def test_later_replay_verified_useful_proposal_resolves_false_negative_append_only(self):
        missed_payload, missed_expected = self._false_negative_payload()
        false_negative = record_graph_rebase_operator_action(
            missed_payload,
            action='adjudicate',
            adjudication='false_negative',
            reason='This settled candidate should have produced a partial proposal.',
            evidence_refs=['operator:false-negative-review'],
            operator_identity='test-operator',
            registry_path=self.registry_path,
            **missed_expected,
        )
        proposal_payload = self._payload(
            frame_id='resp-operator-review:frame-remediated',
            frame_sequence=2,
        )

        resolution = self._record(
            proposal_payload,
            action='adjudicate',
            adjudication='useful_proposal',
            reason='The remediated producer now emits the required partial proposal.',
            evidence_refs=['operator:false-negative-resolution'],
            resolves_record_id=false_negative['record_id'],
        )

        self.assertTrue(resolution['replay_verified'])
        self.assertEqual(
            resolution['resolves_record_id'],
            false_negative['record_id'],
        )
        self.assertEqual(
            resolution['resolved_candidate_observation_id'],
            false_negative['candidate_observation_id'],
        )
        self.assertEqual(
            resolution['resolved_response_id'],
            false_negative['response_id'],
        )
        self.assertEqual(
            load_graph_rebase_operator_records(registry_path=self.registry_path),
            [false_negative, resolution],
        )

    def test_false_negative_resolution_rejects_unknown_nonuseful_and_duplicate_links(self):
        proposal_payload = self._payload(
            frame_id='resp-operator-review:frame-remediated',
            frame_sequence=2,
        )
        with self.assertRaises(GraphRebaseOperatorRegistryError) as unknown:
            self._record(
                proposal_payload,
                action='adjudicate',
                adjudication='useful_proposal',
                reason='Unknown resolution targets are forbidden.',
                evidence_refs=['operator:false-negative-resolution'],
                resolves_record_id='graph-rebase-operator-missing',
            )
        self.assertEqual(
            unknown.exception.code,
            'false_negative_resolution_target_not_found',
        )

        missed_payload, missed_expected = self._false_negative_payload()
        false_negative = record_graph_rebase_operator_action(
            missed_payload,
            action='adjudicate',
            adjudication='false_negative',
            reason='This settled candidate should have produced a partial proposal.',
            evidence_refs=['operator:false-negative-review'],
            operator_identity='test-operator',
            registry_path=self.registry_path,
            **missed_expected,
        )
        with self.assertRaises(GraphRebaseOperatorRegistryError) as nonuseful:
            self._record(
                proposal_payload,
                action='adjudicate',
                adjudication='false_positive',
                reason='A non-useful proposal cannot resolve missed evidence.',
                evidence_refs=['operator:false-negative-resolution'],
                resolves_record_id=false_negative['record_id'],
            )
        self.assertEqual(
            nonuseful.exception.code,
            'false_negative_resolution_requires_useful_proposal_adjudication',
        )

        first_resolution = self._record(
            proposal_payload,
            action='adjudicate',
            adjudication='useful_proposal',
            reason='The producer remediation is replay-verified.',
            evidence_refs=['operator:false-negative-resolution'],
            resolves_record_id=false_negative['record_id'],
        )
        self.assertTrue(first_resolution['replay_verified'])
        with self.assertRaises(GraphRebaseOperatorRegistryError) as duplicate:
            self._record(
                proposal_payload,
                action='adjudicate',
                adjudication='useful_proposal',
                reason='A conflicting second resolution must not replace the first.',
                evidence_refs=['operator:false-negative-resolution'],
                resolves_record_id=false_negative['record_id'],
            )
        self.assertEqual(duplicate.exception.code, 'false_negative_already_resolved')

    def test_useful_adjudication_requires_runtime_replay_match(self):
        payload = self._payload()
        review = payload['runtime']['request_phase_graph']['graph_rebase_reviews'][0]
        review['validation_checks'] = [
            *review.get('validation_checks', []),
            {
                'check': 'tampered_review_projection',
                'status': 'passed',
                'reason': 'not produced by deterministic validation',
            },
        ]

        with self.assertRaises(GraphRebaseOperatorRegistryError) as mismatch:
            self._useful_adjudication(payload)

        self.assertEqual(
            mismatch.exception.code,
            'graph_rebase_review_replay_mismatch',
        )
        self.assertFalse(self.registry_path.exists())

    def test_false_negative_rejects_surviving_formal_review_truth(self):
        payload = self._payload()
        graph = payload['runtime']['request_phase_graph']
        proposal = graph['graph_rebase_proposals'][0]
        graph.pop('graph_rebase_proposals', None)
        payload['runtime']['developer_diagnostics'] = {
            'runtime_graph_rebase_candidate_review': {
                'kind': 'ollmo.runtime_graph_rebase_candidate_review',
                'status': 'not_proposed',
                'reason': 'producer_missed_candidate',
                'base_graph_digest': stable_graph_digest(graph),
                'candidate_graph_digest': proposal['candidate_graph_digest'],
                'runtime_effect': 'none',
            }
        }

        with self.assertRaises(GraphRebaseOperatorRegistryError) as formal_truth:
            record_graph_rebase_operator_action(
                payload,
                action='adjudicate',
                adjudication='false_negative',
                reason='Do not misclassify surviving review truth.',
                evidence_refs=['operator:false-negative-review'],
                expected_response_id=payload['response_id'],
                expected_frame_id=payload['response_frame']['frame_id'],
                expected_frame_sequence=payload['response_frame']['frame_sequence'],
                expected_proposal_id='no_formal_proposal',
                expected_base_graph_digest=stable_graph_digest(graph),
                expected_candidate_graph_digest=proposal['candidate_graph_digest'],
                expected_requested_rebase_class='partial_subtree_rebase',
                operator_identity='test-operator',
                registry_path=self.registry_path,
            )

        self.assertEqual(
            formal_truth.exception.code,
            'false_negative_requires_no_formal_rebase_truth',
        )

    def test_loading_missing_registry_is_observer_only(self):
        missing_path = Path(self._tmpdir.name) / 'missing' / 'reviews.jsonl'

        self.assertEqual(
            load_graph_rebase_operator_records(registry_path=missing_path),
            [],
        )
        self.assertFalse(missing_path.parent.exists())

    def test_operator_targets_reject_noncanonical_frame_sequences(self):
        invalid_sequences = (
            True,
            0,
            -1,
            1.0,
            float('inf'),
            float('nan'),
            '1',
            '01',
            '+1',
            '1.0',
        )
        targets = (
            ('formal_proposal', self._payload(), None),
            ('false_negative', *self._false_negative_payload()),
        )
        for target_kind, payload, false_negative_expected in targets:
            expected = (
                false_negative_expected
                if false_negative_expected is not None
                else self._expected(payload)
            )
            adjudication = (
                'false_negative'
                if target_kind == 'false_negative'
                else 'useful_proposal'
            )
            for source in ('expected', 'canonical'):
                for sequence in invalid_sequences:
                    with self.subTest(
                        target_kind=target_kind,
                        source=source,
                        sequence=sequence,
                    ):
                        current_payload = copy.deepcopy(payload)
                        current_expected = dict(expected)
                        if source == 'expected':
                            current_expected['expected_frame_sequence'] = sequence
                            expected_code = 'expected_frame_sequence_invalid'
                        else:
                            current_payload['response_frame']['frame_sequence'] = sequence
                            expected_code = 'canonical_frame_sequence_invalid'
                        with self.assertRaises(
                            GraphRebaseOperatorRegistryError
                        ) as raised:
                            record_graph_rebase_operator_action(
                                current_payload,
                                action='adjudicate',
                                adjudication=adjudication,
                                reason='Reject a noncanonical frame-sequence CAS.',
                                evidence_refs=['operator:frame-sequence-integrity'],
                                operator_identity='test-operator',
                                registry_path=self.registry_path,
                                **current_expected,
                            )
                        self.assertEqual(raised.exception.code, expected_code)

        self.assertFalse(self.registry_path.exists())

    def test_operator_targets_fail_closed_on_nonfinite_late_fill_counts(self):
        targets = (
            ('formal_proposal', self._payload(), None),
            ('false_negative', *self._false_negative_payload()),
        )
        for target_kind, payload, false_negative_expected in targets:
            expected = (
                false_negative_expected
                if false_negative_expected is not None
                else self._expected(payload)
            )
            adjudication = (
                'false_negative'
                if target_kind == 'false_negative'
                else 'useful_proposal'
            )
            for field in ('active_count', 'pending_count'):
                for count in (float('inf'), float('-inf'), float('nan')):
                    with self.subTest(
                        target_kind=target_kind,
                        field=field,
                        count=count,
                    ):
                        current_payload = copy.deepcopy(payload)
                        current_payload['late_fill'][field] = count
                        with self.assertRaises(
                            GraphRebaseOperatorRegistryError
                        ) as raised:
                            record_graph_rebase_operator_action(
                                current_payload,
                                action='adjudicate',
                                adjudication=adjudication,
                                reason='Reject a non-finite active-work count.',
                                evidence_refs=['operator:late-fill-count-integrity'],
                                operator_identity='test-operator',
                                registry_path=self.registry_path,
                                **expected,
                            )
                        self.assertEqual(
                            raised.exception.code,
                            'active_late_fill_must_settle',
                        )

        self.assertFalse(self.registry_path.exists())

    def test_concurrent_duplicate_submission_appends_one_record(self):
        payload = self._payload()

        with ThreadPoolExecutor(max_workers=8) as executor:
            records = list(
                executor.map(
                    lambda _index: self._useful_adjudication(payload),
                    range(16),
                )
            )

        self.assertEqual(len({record['record_id'] for record in records}), 1)
        self.assertEqual(
            len(load_graph_rebase_operator_records(registry_path=self.registry_path)),
            1,
        )

    def test_stage_requires_prior_useful_adjudication(self):
        payload = self._payload()
        with self.assertRaises(GraphRebaseOperatorRegistryError) as missing:
            self._stage(payload)
        self.assertEqual(
            missing.exception.code,
            'trusted_useful_proposal_adjudication_required',
        )

        self._record(
            payload,
            action='adjudicate',
            adjudication='false_positive',
            reason='This proposal should not be promoted.',
            evidence_refs=['operator:false-positive-review'],
        )
        with self.assertRaises(GraphRebaseOperatorRegistryError) as rejected:
            self._stage(payload)
        self.assertEqual(
            rejected.exception.code,
            'trusted_useful_proposal_adjudication_required',
        )

        review_record = self._useful_adjudication(payload)
        stage_record = self._stage(payload)
        self.assertEqual(stage_record['status'], 'staged')
        self.assertEqual(
            stage_record['runtime_effect'],
            'staged_no_executable_mutation',
        )
        self.assertEqual(stage_record['review_record_id'], review_record['record_id'])
        self.assertNotIn('authorization', stage_record)

    def test_authorize_partial_emits_joinable_exact_authorization(self):
        initial_payload = self._payload()
        review_record = self._useful_adjudication(initial_payload)
        stage_record = self._stage(initial_payload)
        staged_payload = self._payload(
            staged=True,
            frame_id='resp-operator-review:frame-2',
            frame_sequence=2,
        )

        authorization_record = self._record(
            staged_payload,
            action='authorize_partial',
            adjudication='accepted',
            reason='Authorize this exact staged partial successor only.',
            evidence_refs=['operator:partial-authorization'],
            gate=self._promotion_gate(),
        )

        authorization = authorization_record['authorization']
        self.assertEqual(authorization['kind'], 'ollmo.graph_rebase_authorization')
        self.assertEqual(authorization['status'], 'accepted')
        self.assertEqual(authorization['authority'], 'operator_review')
        self.assertEqual(authorization['source'], 'runtime_operator_registry')
        self.assertEqual(authorization['provenance'], 'runtime_operator_registry')
        self.assertEqual(
            authorization['registry_record_id'],
            authorization_record['record_id'],
        )
        self.assertEqual(authorization['review_record_id'], review_record['record_id'])
        self.assertEqual(authorization['stage_record_id'], stage_record['record_id'])
        self.assertEqual(authorization['allowed_autonomy'], ['apply_reviewed'])
        self.assertEqual(
            authorization['requested_rebase_class'],
            'partial_subtree_rebase',
        )
        self.assertEqual(
            authorization_record['promotion_gate'],
            self._promotion_gate(),
        )
        self.assertIn(
            'readiness:partial-stage-corpus',
            authorization['evidence_refs'],
        )
        self.assertTrue(verify_graph_rebase_operator_record(authorization_record))

        expected = self._expected(staged_payload)
        joined = find_trusted_graph_rebase_authorization(
            response_id=expected['expected_response_id'],
            frame_id=expected['expected_frame_id'],
            proposal_id=expected['expected_proposal_id'],
            base_graph_digest=expected['expected_base_graph_digest'],
            candidate_graph_digest=expected['expected_candidate_graph_digest'],
            requested_rebase_class=expected['expected_requested_rebase_class'],
            registry_path=self.registry_path,
        )
        self.assertEqual(joined, authorization)

        graph = staged_payload['runtime']['request_phase_graph']
        proposal = copy.deepcopy(graph['graph_rebase_proposals'][0])
        proposal['graph_rebase_authorization'] = joined
        runtime_review = validate_graph_rebase_proposal(
            proposal,
            request_phase_graph=graph,
            closure_review=staged_payload['runtime']['graph_closure_review'],
            trusted_authorization=joined,
            root_prompt='Create the operator-review graph from the original request.',
        )
        lifecycle = build_graph_rebase_lifecycle(
            request_phase_graph=graph,
            rebase_review=runtime_review,
            autonomy_level='apply_reviewed',
            trusted_authorization=joined,
        )
        self.assertEqual(lifecycle['status'], 'staged')

        self.registry_path.write_text(
            json.dumps(authorization_record, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        self.assertEqual(
            find_trusted_graph_rebase_authorization(
                response_id=expected['expected_response_id'],
                proposal_id=expected['expected_proposal_id'],
                base_graph_digest=expected['expected_base_graph_digest'],
                candidate_graph_digest=expected['expected_candidate_graph_digest'],
                requested_rebase_class=expected['expected_requested_rebase_class'],
                registry_path=self.registry_path,
            ),
            {},
        )

    def test_authorize_partial_requires_registry_stage_runtime_stage_and_green_gate(self):
        initial_payload = self._payload()
        self._useful_adjudication(initial_payload)
        self._stage(initial_payload)

        with self.assertRaises(GraphRebaseOperatorRegistryError) as no_runtime_stage:
            self._record(
                initial_payload,
                action='authorize_partial',
                adjudication='accepted',
                reason='Attempt authorization without a durable runtime stage.',
                evidence_refs=['operator:partial-authorization'],
                gate=self._promotion_gate(),
            )
        self.assertEqual(
            no_runtime_stage.exception.code,
            'exact_runtime_graph_rebase_stage_required',
        )

        staged_payload = self._payload(staged=True)
        with self.assertRaises(GraphRebaseOperatorRegistryError) as red_gate:
            self._record(
                staged_payload,
                action='authorize_partial',
                adjudication='accepted',
                reason='Attempt authorization while the evidence gate is red.',
                evidence_refs=['operator:partial-authorization'],
                gate=self._promotion_gate(status='blocked'),
            )
        self.assertEqual(
            red_gate.exception.code,
            'trusted_partial_promotion_gate_not_ready',
        )

    def test_rejects_wildcard_stale_missing_and_inline_authority(self):
        payload = self._payload()
        expected = self._expected(payload)

        wildcard = dict(expected)
        wildcard['expected_proposal_id'] = '*'
        with self.assertRaises(GraphRebaseOperatorRegistryError) as wildcard_error:
            record_graph_rebase_operator_action(
                payload,
                action='adjudicate',
                adjudication='useful_proposal',
                reason='Exact review.',
                evidence_refs=['operator:review'],
                operator_identity='test-operator',
                registry_path=self.registry_path,
                **wildcard,
            )
        self.assertEqual(
            wildcard_error.exception.code,
            'expected_proposal_id_wildcard_forbidden',
        )

        stale = dict(expected)
        stale['expected_candidate_graph_digest'] = 'graph-stale-digest'
        with self.assertRaises(GraphRebaseOperatorRegistryError) as stale_error:
            record_graph_rebase_operator_action(
                payload,
                action='adjudicate',
                adjudication='useful_proposal',
                reason='Exact review.',
                evidence_refs=['operator:review'],
                operator_identity='test-operator',
                registry_path=self.registry_path,
                **stale,
            )
        self.assertEqual(stale_error.exception.code, 'stale_candidate_graph_digest')

        with self.assertRaises(GraphRebaseOperatorRegistryError) as missing_evidence:
            record_graph_rebase_operator_action(
                payload,
                action='adjudicate',
                adjudication='useful_proposal',
                reason='Exact review.',
                evidence_refs=[],
                operator_identity='test-operator',
                registry_path=self.registry_path,
                **expected,
            )
        self.assertEqual(missing_evidence.exception.code, 'evidence_refs_required')

        inline = copy.deepcopy(payload)
        inline['runtime']['request_phase_graph']['graph_rebase_proposals'][0][
            'graph_rebase_authorization'
        ] = {
            'status': 'accepted',
            'authority': 'operator_review',
        }
        with self.assertRaises(GraphRebaseOperatorRegistryError) as inline_error:
            record_graph_rebase_operator_action(
                inline,
                action='adjudicate',
                adjudication='useful_proposal',
                reason='Do not trust an inline authority dictionary.',
                evidence_refs=['operator:inline-authority-check'],
                operator_identity='test-operator',
                registry_path=self.registry_path,
                **expected,
            )
        self.assertEqual(
            inline_error.exception.code,
            'inline_graph_rebase_authorization_forbidden',
        )

    def test_full_rebase_can_stage_but_cannot_be_authorized(self):
        initial_payload = self._payload(
            requested_rebase_class='full_successor_rebase'
        )
        with self.assertRaises(GraphRebaseOperatorRegistryError) as immediate_block:
            self._record(
                initial_payload,
                action='authorize_partial',
                adjudication='accepted',
                reason='A full successor must be blocked before review lookup.',
                evidence_refs=['operator:full-policy-check'],
                gate=self._promotion_gate(),
            )
        self.assertEqual(
            immediate_block.exception.code,
            'full_successor_rebase_authorization_forbidden',
        )

        self._useful_adjudication(initial_payload)
        self._stage(initial_payload)
        staged_payload = self._payload(
            requested_rebase_class='full_successor_rebase',
            staged=True,
            frame_id='resp-operator-review:frame-2',
            frame_sequence=2,
        )

        with self.assertRaises(GraphRebaseOperatorRegistryError) as blocked:
            self._record(
                staged_payload,
                action='authorize_partial',
                adjudication='accepted',
                reason='A full successor must remain shadow-only.',
                evidence_refs=['operator:full-policy-check'],
                gate=self._promotion_gate(),
            )
        self.assertEqual(
            blocked.exception.code,
            'full_successor_rebase_authorization_forbidden',
        )

    def test_registry_tamper_or_corrupt_json_fails_closed(self):
        payload = self._payload()
        record = self._useful_adjudication(payload)
        tampered = copy.deepcopy(record)
        tampered['reason'] = 'Tampered after append.'
        self.registry_path.write_text(
            json.dumps(tampered, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        with self.assertRaises(GraphRebaseOperatorRegistryError) as tamper_error:
            load_graph_rebase_operator_records(registry_path=self.registry_path)
        self.assertEqual(
            tamper_error.exception.code,
            'operator_registry_record_verification_failed',
        )

        self.registry_path.write_text('{not-json\n', encoding='utf-8')
        with self.assertRaises(GraphRebaseOperatorRegistryError) as json_error:
            load_graph_rebase_operator_records(registry_path=self.registry_path)
        self.assertEqual(
            json_error.exception.code,
            'operator_registry_corrupt_json',
        )


if __name__ == '__main__':
    unittest.main()
