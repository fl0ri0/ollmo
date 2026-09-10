"""Late Fill-specific observation assembly over the existing telemetry services.

No scheduling, branch mutation, evidence validation or canonical persistence.
Callers retain the causal anchors and attach returned diagnostic fields themselves.
Legacy inline metadata/log errors deliberately retain their existing propagation;
the underlying causal/transition services retain their own fail-open boundaries.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

from ollmo_services.events import (
    causal_event, exact_target, judgment_summary, select_fields,
    observe_call, observe_request, observe_transition,
    transition_attempt, transition_binding, transition_span,
)

def observe_execution_gate(function):
    """Configure existing wrappers without adding a runtime interception layer."""
    function = observe_call('late_fill.semantic_execution_gate_decision',
                  target=lambda a: exact_target(a['branch']),
                  inputs=lambda a: a['self']._causal_execution_gate_inputs(a['branch'], a['current_payload']),
                  authority=lambda a: {'rule': 'semantic_execution_gate_current_branch_v1'},
                  evidence=lambda a: [], complete=True,
                  required_rule='semantic_execution_gate_current_branch_v1', result=judgment_summary)(function)
    function = observe_transition('late_fill.start_check', target=lambda a: exact_target(
                        a['branch'], 'response_id', 'branch_id', 'phase_id',
                        'attempt_id', 'attempt_count', 'attempt_number',
                        'retry_count', 'auto_executable_repair_retry_count'),
                        result=lambda value: {
                            'gate_scope': 'semantic_execution_gate_current_branch_v1',
                            'gate_check_complete': True,
                            'gate_status': value.get('status'),
                            'gate_action': value.get('action'),
                            'aggregate_start_authority': 'not_established_by_this_gate',
                        })(function)
    return function


def observe_prepared_branch(function):
    """Configure existing wrappers without adding a runtime interception layer."""
    function = observe_call('late_fill.execute_prepared_late_fill_branch',
                  record_kind='materialization_invocation', target=lambda a: exact_target(a['plan']),
                  inputs=lambda a: {'contract': select_fields(a['plan'], ('branch_id', 'phase_id', 'capability', 'depends_on'))},
                  context=lambda a: {
                      source: select_fields(a['plan'].get(source), (
                          'semantic_review_lens', 'semantic_role_id', 'review_type',
                          'semantic_depth', 'structural_granularity', 'scope',
                          'attention_target', 'controlled_attention_frame_id',
                      )) for source in ('branch', 'infer_payload', 'effective_data')
                  }, result=judgment_summary)(function)
    return function


def observe_worker(function):
    """Configure existing wrappers without adding a runtime interception layer."""
    function = observe_transition('late_fill.worker', target=lambda a: {'response_id': a['response_payload'].get('id')})(function)
    function = observe_call('late_fill.complete_response_late_fill',
                  target=lambda a: {'response_id': a['response_payload'].get('id')},
                  inputs=lambda a: {'trigger': a['artifact_gap'].get('trigger')})(function)
    function = observe_request(function)
    return function


def observe_schedule(function):
    """Configure existing wrappers without adding a runtime interception layer."""
    function = observe_transition('late_fill.schedule', target=lambda a: {'response_id': a['response_payload'].get('id')},
                        result=lambda value: {'schedule_return_value': value})(function)
    function = observe_call('late_fill.schedule_response_late_fill',
                  target=lambda a: {'response_id': a['response_payload'].get('id')},
                  result=lambda value: {'scheduled': value})(function)
    return function


class LateFillTrace:
    """Explicit response/callback binding; no mutable runtime owner is retained."""

    def __init__(self, response_id, *, target=None):
        self.response_id = response_id
        self.target = {'response_id': response_id} if target is None else target

    def span(self, stage):
        fields = ({'hydration_scope': 'included_if_existing_lookup_hydrates'}
                  if stage == 'callback.lookup' else {})
        return transition_span('late_fill.' + stage, target=self.target, **fields)

    def callback(self, event):
        return LateFillTrace(self.response_id, target={
            'response_id': self.response_id, **exact_target(event),
        })

    def handoff(self):
        return transition_binding(transition_attempt({'response_id': self.response_id}))

    @staticmethod
    def availability_wait_started(branch, waiting, previous, evidence):
        wait_event = causal_event('late_fill.availability', 'wait_started',
            target=exact_target(branch), predicate='availability_poll_due',
            unresolved_state='compatible_route_recheck_due',
            next_check_epoch=waiting['next_check_epoch'],
            producer_instance_ids=evidence.get('candidate_instance_ids'),
            producer_version=None, gates_complete=False,
            required_rule='late_fill_availability_poll',
            previous_wait_id=(previous.get('causal') or {}).get('wait_id'))
        if wait_event:
            return {
                'wait_id': wait_event['event_id'],
                'next_check_epoch': waiting['next_check_epoch'],
                'process_boot_id': wait_event['process_boot_id'],
                'start_monotonic_ns': wait_event['monotonic_ns'],
            }
        return None

    @staticmethod
    def wait_settled(branch, removed_wait, status):
        if isinstance(removed_wait, Mapping):
            causal_event('late_fill.availability', 'wait_settled',
                target=exact_target(branch), status=status,
                wait_id=(removed_wait.get('causal') or {}).get('wait_id'),
                predicate='availability_poll_due', gates_complete=False)

    @staticmethod
    def branch_settled(branch, normalized_branch, status, attempt):
        causal_event('late_fill.branch_state', 'state_transition',
            target=exact_target(branch), status=status,
            runtime_state_delta={'status_before': branch.get('status'),
                                 'status_after': normalized_branch.get('status')},
            attempt=select_fields(attempt, ('attempt', 'attempt_count', 'instance_id', 'code')),
            outcome_ref=exact_target(normalized_branch))

    @staticmethod
    def availability_wake(branch):
        retained_wait = branch.get('availability_wait') or {}
        wait_identity = retained_wait.get('causal') or {}
        if wait_identity.get('wait_id'):
            causal_event('late_fill.availability', 'wait_wake',
                target=exact_target(branch), wait_id=wait_identity['wait_id'],
                predicate='availability_poll_due', predicate_satisfied=True,
                next_check_epoch=retained_wait.get('next_check_epoch'),
                required_rule='late_fill_availability_poll', gates_complete=False,
                wake_reason='existing_poll_clock_recheck',
                route_eligibility='not_yet_revalidated')

    @staticmethod
    def result_ignored(branch, decision):
        causal_event('late_fill.result_gate', 'result_disposition',
            target=exact_target(branch), stale_result_disposition='ignored',
            judgment=judgment_summary(decision),
            required_rule='semantic_execution_gate_current_branch_v1')

    def repair_requeued(self, sink, branch_id, phase_id, capability, attempt, *, reason):
        messages = {
            'failed_materialization': 'Retryable auto-executable repair branch requeued after failed materialization attempt.',
            'saved_truth_failure': 'Retryable auto-executable text artifact repair requeued after saved-truth failure.',
        }
        sink(category='responses', action='late_fill', status='queued',
             response_id=self.response_id, capability=capability,
             branch_id=branch_id, phase_id=phase_id,
             attempt=select_fields(attempt, ('attempt', 'attempt_count', 'instance_id', 'code')),
             message=messages[reason])

    @staticmethod
    def clock():
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(started_at):
        return round((time.perf_counter() - started_at) * 1000, 3)

    @staticmethod
    def finalizer_timing(payload):
        runtime = payload.get('runtime') if isinstance(payload.get('runtime'), Mapping) else {}
        diagnostics = (runtime.get('developer_diagnostics')
                       if isinstance(runtime.get('developer_diagnostics'), Mapping) else {})
        return (diagnostics.get('response_frame_finalize_timing')
                if isinstance(diagnostics.get('response_frame_finalize_timing'), Mapping) else {})

    @staticmethod
    def checkpoint_timing(elapsed_ms, checkpoint_mode, **counts):
        return {
            'kind': 'ollmo.response_frame_finalize_timing',
            'phase': 'nonterminal_late_fill',
            'skipped': True,
            'reason': 'lightweight_nonterminal_checkpoint',
            'checkpoint_mode': checkpoint_mode,
            'persist_requested': False,
            'total_elapsed_ms': elapsed_ms,
            **counts,
        }

    @staticmethod
    def post_wave_timing(status, terminal, checkpoint_mode, elapsed_ms, finalizer_timing, **counts):
        timing = {
            'kind': 'ollmo.late_fill_post_wave_backend_timing',
            'phase': 'terminal' if terminal else 'nonterminal',
            'status': status,
            'checkpoint_mode': checkpoint_mode,
            'finalize_skipped': not terminal,
            'finalize_elapsed_ms': elapsed_ms,
            **counts,
        }
        if finalizer_timing:
            timing['response_frame_finalize_timing'] = dict(finalizer_timing)
        return timing

    def log_post_wave_timing(self, sink, timing, finalizer_timing, **counts):
        sink(
            category='responses', action='late_fill_post_wave_backend_timing', status='ok',
            response_id=self.response_id, phase=timing['phase'],
            late_fill_status=timing['status'], checkpoint_mode=timing['checkpoint_mode'],
            finalize_skipped=timing['finalize_skipped'],
            finalize_elapsed_ms=timing['finalize_elapsed_ms'],
            touch_response_lookup_elapsed_ms=timing['touch_response_lookup_elapsed_ms'],
            response_frame_finalize_timing=dict(finalizer_timing) if finalizer_timing else None,
            **counts,
            message=('Late-fill post-wave backend timing '
                     f"{timing['phase']} finalize={timing['finalize_elapsed_ms']}ms "
                     f"lookup={timing['touch_response_lookup_elapsed_ms']}ms"),
        )
