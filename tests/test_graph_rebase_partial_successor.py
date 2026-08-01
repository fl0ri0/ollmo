import copy
import os
import unittest
from unittest.mock import patch

from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_services.graph_rebase import build_graph_rebase_proposal
from tests import test_graph_rebase_review as rebase_test_helpers


class PartialGraphRebaseSuccessorTests(unittest.TestCase):
    def setUp(self):
        self.contract = rebase_test_helpers.GraphRebaseReviewTests()
        self.owner = ResponsesRequestRuntimeOwner(
            hooks={
                'normalize_capability': lambda value: str(value or '').strip().lower(),
                'build_late_fill_state': self._build_late_fill_state,
                'attach_late_fill_state': self._attach_late_fill_state,
            },
            capability_chat='chat',
            capability_embedding='embedding',
            capability_image_generation='image_generation',
            capability_speech_to_text='speech_to_text',
            request_timeout_error=TimeoutError,
            request_exception_error=Exception,
        )

    @staticmethod
    def _build_late_fill_state(gap, *, status, prior_state=None, extra=None):
        return {
            **dict(prior_state or {}),
            **dict(gap or {}),
            **dict(extra or {}),
            'status': status,
        }

    @staticmethod
    def _attach_late_fill_state(payload, state):
        updated = copy.deepcopy(payload)
        updated['late_fill'] = copy.deepcopy(state)
        return updated

    def _partial_base_graph(self):
        graph = self.contract._base_graph()
        for collection in ('phases', 'downstream_branches'):
            for record in graph[collection]:
                if record.get('branch_id') == 'branch-html':
                    record['depends_on'] = [
                        'phase-1',
                        'phase-image-1',
                        'phase-image-2',
                    ]
        return graph

    def _partial_candidate_graph(self):
        candidate = self.contract._executable_partial_candidate_graph()
        for collection in ('phases', 'downstream_branches'):
            for record in candidate[collection]:
                if record.get('branch_id') == 'branch-html':
                    record['depends_on'] = [
                        'phase-1',
                        'phase-image-1',
                        'phase-image-2',
                    ]
        return candidate

    def _partial_proposal(self, *, candidate=None):
        return build_graph_rebase_proposal(
            request_phase_graph=self._partial_base_graph(),
            candidate_graph=candidate or self._partial_candidate_graph(),
            target_response_id='resp-base',
            target_frame_id='frame-base',
            source='runtime_closure_review',
            reason='Review one bounded HTML subtree from current Closure evidence.',
            intent_anchor={'prompt': 'create linked local image site'},
            evidence_refs=['closure:partial_subtree_rebase'],
            requested_rebase_class='partial_subtree_rebase',
            scope_root_ids=['phase-html', 'phase-html-review', 'branch-html-review'],
            scope_phase_ids=['phase-html', 'phase-html-review'],
            scope_branch_ids=['branch-html', 'branch-html-review'],
            preserve_outside_scope=True,
            root_prompt='Create the linked local image site from the original request.',
        )

    def _payload(self):
        graph = self._partial_base_graph()
        graph['redraw_scope_evidence'] = {
            'recommended_scope': 'partial_subtree_rebase',
            'reason': 'Exact branch-local structural successor is required.',
            'evidence_refs': ['closure:partial_subtree_rebase'],
            'scope_root_ids': ['obligation-root'],
        }
        graph['redraw_scope_ladder_review'] = {
            'kind': 'ollmo.redraw_scope_ladder_review',
            'status': 'selected',
            'selected_scope': 'partial_subtree_rebase',
            'selected_candidate': {
                'scope': 'partial_subtree_rebase',
                'eligible': True,
                'runtime_action': 'reviewed_graph_rebase_only',
            },
            'blocked_reasons': [],
        }
        proposal = self._partial_proposal()
        graph['graph_rebase_proposals'] = [copy.deepcopy(proposal)]
        return {
            'id': 'resp-base',
            'response_id': 'resp-base',
            'status': 'completed',
            'lifecycle_state': 'completed',
            'output_text': 'The parent output stays immutable.',
            'request': {
                'prompt': 'Create the linked local image site from the original request.',
            },
            'response_frame': {
                'frame_id': 'frame-base',
                'frame_sequence': 4,
                'frame_relation': {'kind': 'late_fill_successor'},
            },
            'late_fill': {
                'status': 'completed',
                'pending_branches': [],
                'active_branches': [],
                'completed_branches': [],
                'failed_branches': [],
            },
            'runtime': {
                'graph_closure_review': {
                    'status': 'repair_required',
                    'current_evidence_refs': ['closure:partial_subtree_rebase'],
                },
                'request_phase_graph': graph,
            },
        }

    def _authorization(self, payload):
        proposal = payload['runtime']['request_phase_graph']['graph_rebase_proposals'][0]
        return {
            **self.contract._trusted_partial_authorization(proposal),
            'response_id': payload['response_id'],
            'frame_id': payload['response_frame']['frame_id'],
            'base_graph_digest': proposal['base_graph_digest'],
        }

    def test_exact_trusted_partial_rebase_prepares_one_branch_local_successor(self):
        parent = self._payload()
        original_parent = copy.deepcopy(parent)
        proposal = parent['runtime']['request_phase_graph']['graph_rebase_proposals'][0]
        authorization = self._authorization(parent)

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                parent,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=authorization,
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'queued')
        self.assertEqual(parent, original_parent)
        self.assertEqual(result['execution']['scheduled_branch_ids'], ['branch-html-review'])
        successor = result['response_payload']
        self.assertEqual(successor['frame_relation']['kind'], 'graph_rebase_partial_successor')
        self.assertEqual(successor['frame_relation']['parent_frame_id'], 'frame-base')
        self.assertEqual(successor['response_id'], parent['response_id'])
        self.assertFalse(result['artifact_gap']['execution_contract']['root_scoped'])
        branch = result['artifact_gap']['pending_branches'][0]
        self.assertEqual(branch['branch_id'], 'branch-html-review')
        self.assertIn('Review the completed HTML artifact', branch['content_payload'])
        self.assertNotIn('The parent output stays immutable.', branch['content_payload'])

    def test_product_shadow_never_executes_without_explicit_operator_action(self):
        payload = self._payload()
        proposal = payload['runtime']['request_phase_graph']['graph_rebase_proposals'][0]

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=self._authorization(payload),
            )

        self.assertEqual(result['status'], 'not_applicable')
        self.assertEqual(result['reason'], 'graph_rebase_autonomy_not_apply_reviewed')

    def test_explicit_off_blocks_even_explicit_operator_apply(self):
        payload = self._payload()
        proposal = payload['runtime']['request_phase_graph']['graph_rebase_proposals'][0]

        with patch.dict(os.environ, {'OLLMO_GRAPH_REBASE_AUTONOMY': 'off'}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=self._authorization(payload),
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'not_applicable')
        self.assertEqual(result['reason'], 'graph_rebase_autonomy_off')

    def test_inline_authorization_and_missing_registry_join_are_blocked(self):
        payload = self._payload()
        proposal = payload['runtime']['request_phase_graph']['graph_rebase_proposals'][0]
        proposal['graph_rebase_authorization'] = self._authorization(payload)

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=None,
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn('trusted_graph_rebase_authorization_missing', result['blocked_reasons'])

    def test_stale_parent_frame_binding_is_blocked(self):
        payload = self._payload()
        proposal = payload['runtime']['request_phase_graph']['graph_rebase_proposals'][0]
        authorization = self._authorization(payload)
        authorization['frame_id'] = 'frame-stale'

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=authorization,
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn('graph_rebase_authorization_frame_mismatch', result['blocked_reasons'])

    def test_full_rebase_is_never_consumed_by_partial_successor(self):
        payload = self._payload()
        graph = payload['runtime']['request_phase_graph']
        proposal = self.contract._proposal()
        proposal['requested_rebase_class'] = 'full_successor_rebase'
        graph['graph_rebase_proposals'] = [proposal]
        authorization = {
            **self.contract._trusted_partial_authorization(proposal),
            'requested_rebase_class': 'full_successor_rebase',
            'response_id': payload['response_id'],
            'frame_id': payload['response_frame']['frame_id'],
            'base_graph_digest': proposal['base_graph_digest'],
        }

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=authorization,
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn('apply_reviewed_partial_rebase_only', result['blocked_reasons'])

    def test_missing_local_execution_payload_blocks_before_successor_creation(self):
        payload = self._payload()
        graph = payload['runtime']['request_phase_graph']
        candidate = self._partial_candidate_graph()
        for collection in ('phases', 'downstream_branches'):
            for record in candidate[collection]:
                if record.get('branch_id') == 'branch-html-review':
                    record.pop('content_payload', None)
        proposal = self._partial_proposal(candidate=candidate)
        graph['graph_rebase_proposals'] = [proposal]
        authorization = self._authorization(payload)
        authorization['proposal_id'] = proposal['proposal_id']
        authorization['candidate_graph_digest'] = proposal['candidate_graph_digest']

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=authorization,
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn('apply_reviewed_execution_contract_proof_required', result['blocked_reasons'])

    def test_relabelled_root_prompt_is_blocked_before_successor_creation(self):
        payload = self._payload()
        root_prompt = 'Create the entire original multi-artifact site request again.'
        candidate = self._partial_candidate_graph()
        for collection in ('phases', 'downstream_branches'):
            for record in candidate[collection]:
                if record.get('branch_id') == 'branch-html-review':
                    record['content_payload'] = root_prompt.swapcase()
                    record['content_payload_source'] = 'runtime_partial_rebase_review'
        proposal = self._partial_proposal(candidate=candidate)
        payload['request'] = {'prompt': root_prompt}
        payload['runtime']['request_phase_graph']['graph_rebase_proposals'] = [proposal]
        authorization = self._authorization(payload)

        with patch.dict(os.environ, {}, clear=True):
            result = self.owner.prepare_terminal_partial_graph_rebase_successor(
                payload,
                proposal_id=proposal['proposal_id'],
                trusted_authorization=authorization,
                graph_rebase_autonomy='apply_reviewed',
            )

        self.assertEqual(result['status'], 'blocked')
        self.assertIn(
            'partial_rebase_root_prompt_fallback_forbidden',
            result['blocked_reasons'],
        )


if __name__ == '__main__':
    unittest.main()
