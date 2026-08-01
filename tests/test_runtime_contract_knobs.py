import os
import subprocess
import sys
import threading
import time
import unittest

from ollmo_server.multi_materialization_runtime import (
    DEFAULT_MAX_PARALLEL_WORKERS,
    MAX_MAX_PARALLEL_WORKERS,
    MultiMaterializationRuntimeOwner,
    normalize_max_parallel_workers,
)
from ollmo_server.recovery_contract import (
    RECOVERY_ACTION_MANUAL_REVIEW,
    RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
    RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
    RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
    RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
    RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
    RECOVERY_ACTION_RETRY_SAME_BRANCH,
    RECOVERY_ACTION_SEMANTIC_REVIEW,
    RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
    RECOVERY_SUGGESTED_ACTIONS,
    normalize_recovery_suggested_action,
)


class RuntimeContractKnobTests(unittest.TestCase):
    def test_recovery_suggested_actions_are_stable_wire_values(self):
        self.assertEqual(
            RECOVERY_SUGGESTED_ACTIONS,
            {
                'manual_review',
                'start_compatible_instance',
                'retry_excluding_instance',
                'retry_same_branch',
                'rebind_dependency_evidence',
                'repair_dependency_chain',
                'repair_branch_contract',
                'rebuild_from_promoted_obligations',
                'semantic_review',
            },
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE),
            RECOVERY_ACTION_RETRY_EXCLUDING_INSTANCE,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_START_COMPATIBLE_INSTANCE),
            RECOVERY_ACTION_START_COMPATIBLE_INSTANCE,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_RETRY_SAME_BRANCH),
            RECOVERY_ACTION_RETRY_SAME_BRANCH,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE),
            RECOVERY_ACTION_REBIND_DEPENDENCY_EVIDENCE,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN),
            RECOVERY_ACTION_REPAIR_DEPENDENCY_CHAIN,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT),
            RECOVERY_ACTION_REPAIR_BRANCH_CONTRACT,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS),
            RECOVERY_ACTION_REBUILD_FROM_PROMOTED_OBLIGATIONS,
        )
        self.assertEqual(
            normalize_recovery_suggested_action(RECOVERY_ACTION_SEMANTIC_REVIEW),
            RECOVERY_ACTION_SEMANTIC_REVIEW,
        )
        self.assertEqual(
            normalize_recovery_suggested_action('unknown'),
            RECOVERY_ACTION_MANUAL_REVIEW,
        )

    def test_multi_materialization_parallel_worker_knob_is_bounded(self):
        self.assertEqual(normalize_max_parallel_workers(None), DEFAULT_MAX_PARALLEL_WORKERS)
        self.assertEqual(normalize_max_parallel_workers('0'), 1)
        self.assertEqual(normalize_max_parallel_workers('3'), 3)
        self.assertEqual(normalize_max_parallel_workers('999'), MAX_MAX_PARALLEL_WORKERS)
        self.assertEqual(MultiMaterializationRuntimeOwner(max_parallel_workers='999').max_parallel_workers, MAX_MAX_PARALLEL_WORKERS)

    def test_multi_materialization_reuses_instances_after_reservation_wave_exhaustion(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=3)
        available_instance_ids = ['img-1', 'img-2']
        prepare_calls = []

        branches = [
            {
                'branch_id': f'branch-image-{index}',
                'phase_id': f'phase-image-{index}',
                'capability': 'image_generation',
                'prepare_args': {'branch_index': index},
            }
            for index in range(1, 4)
        ]

        def prepare_branch_plan(*, branch_index, excluded_instance_ids=None):
            excluded = list(excluded_instance_ids or [])
            prepare_calls.append(excluded)
            selected_instance_id = next(
                (
                    instance_id
                    for instance_id in available_instance_ids
                    if instance_id not in excluded
                ),
                '',
            )
            if not selected_instance_id:
                raise RuntimeError(
                    "No non-excluded running instance found for capability 'image_generation'."
                )
            return {
                'route_info': {
                    'instance_id': selected_instance_id,
                    'capability': 'image_generation',
                },
                'infer_payload': {
                    'prompt': f'image prompt {branch_index}',
                    'instance_id': selected_instance_id,
                },
            }

        def execute_prepared_branch(plan):
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {
                    'saved_image_path': f"/tmp/{plan['branch_id']}.png",
                },
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        self.assertEqual(len(result['ordered_branch_results']), 3)
        self.assertEqual(
            [
                entry['result']['route_info']['instance_id']
                for entry in result['ordered_branch_results']
            ],
            ['img-1', 'img-2', 'img-1'],
        )
        self.assertEqual(
            prepare_calls,
            [
                [],
                ['img-1'],
                ['img-1', 'img-2'],
                [],
            ],
        )
        policy = result['concurrency_policy']
        self.assertEqual(policy['scheduler'], 'multi_materialization_runtime')
        self.assertEqual(policy['prepared_branch_count'], 3)
        self.assertEqual(policy['distinct_instance_count'], 2)
        self.assertEqual(policy['worker_count'], 2)
        self.assertEqual(policy['gpu_heavy_guard'], 'not_serialized')
        self.assertEqual(policy['distinct_instance_ids'], ['img-1', 'img-2'])
        self.assertEqual(
            policy['same_instance_lock_groups'],
            {'img-1': ['branch-image-1', 'branch-image-3']},
        )
        self.assertEqual(
            [item['branch_id'] for item in policy['prepared_branches']],
            ['branch-image-1', 'branch-image-2', 'branch-image-3'],
        )
        self.assertEqual(
            [item['branch_id'] for item in policy['branch_timings']],
            ['branch-image-1', 'branch-image-2', 'branch-image-3'],
        )
        self.assertEqual(
            [item['branch_id'] for item in policy['prepare_timings']],
            ['branch-image-1', 'branch-image-2', 'branch-image-3'],
        )
        self.assertEqual(
            [item['status'] for item in policy['prepare_timings']],
            ['ok', 'ok', 'ok_after_retry'],
        )
        self.assertTrue(all('prepare_elapsed_ms' in item for item in policy['prepare_timings']))
        self.assertIn('planning_elapsed_ms', policy)
        self.assertTrue(all('elapsed_ms' in item for item in policy['branch_timings']))

    def test_image_and_large_chat_on_distinct_instances_use_worker_budget(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=2)
        branches = [
            {
                'branch_id': 'branch-image-1',
                'phase_id': 'phase-image-1',
                'capability': 'image_generation',
                'prepare_args': {
                    'branch_id': 'branch-image-1',
                    'capability': 'image_generation',
                    'instance_id': 'image-1',
                    'model': 'x/flux2-klein:latest',
                },
            },
            {
                'branch_id': 'branch-chat-helper',
                'phase_id': 'phase-chat-helper',
                'capability': 'chat',
                'prepare_args': {
                    'branch_id': 'branch-chat-helper',
                    'capability': 'chat',
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                },
            },
        ]
        intervals: dict[str, tuple[float, float]] = {}
        interval_lock = threading.Lock()

        def prepare_branch_plan(*, branch_id, capability, instance_id, model, excluded_instance_ids=None):
            return {
                'branch_id': branch_id,
                'phase_id': branch_id.replace('branch-', 'phase-'),
                'capability': capability,
                'route_info': {
                    'capability': capability,
                    'instance_id': instance_id,
                },
                'instance': {
                    'instance_id': instance_id,
                    'model': model,
                    'capability': capability,
                },
            }

        def execute_prepared_branch(plan):
            branch_id = plan['branch_id']
            start = time.perf_counter()
            time.sleep(0.05)
            end = time.perf_counter()
            with interval_lock:
                intervals[branch_id] = (start, end)
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {'ok': True},
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        policy = result['concurrency_policy']
        self.assertEqual(policy['prepared_branch_count'], 2)
        self.assertEqual(policy['distinct_instance_count'], 2)
        self.assertEqual(policy['worker_count'], 2)
        self.assertEqual(policy['gpu_heavy_guard'], 'not_serialized')
        self.assertNotIn('reason', policy)
        image = intervals['branch-image-1']
        chat = intervals['branch-chat-helper']
        self.assertTrue(image[0] < chat[1] and chat[0] < image[1])

    def test_text_io_repair_fast_lane_submits_before_saturated_model_queue(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=4)
        branches = [
            {
                'branch_id': f'branch-image-{index}',
                'phase_id': f'phase-image-{index}',
                'capability': 'image_generation',
                'prepare_args': {
                    'branch_id': f'branch-image-{index}',
                    'capability': 'image_generation',
                    'instance_id': f'image-{index}',
                    'sleep_seconds': 0.06,
                },
            }
            for index in range(1, 5)
        ]
        branches.append(
            {
                'branch_id': 'repair-html-syntax',
                'phase_id': 'repair-html-syntax',
                'capability': 'chat',
                'repair_scope': 'syntax_only',
                'resource_class': 'text_io',
                'dependency_policy': 'target_artifact_snapshot_only',
                'prepare_args': {
                    'branch_id': 'repair-html-syntax',
                    'capability': 'chat',
                    'sleep_seconds': 0.005,
                    'runtime_scheduling_context': {
                        'repair_scope': 'syntax_only',
                        'resource_class': 'text_io',
                        'dependency_policy': 'target_artifact_snapshot_only',
                    },
                },
            }
        )
        started: list[str] = []
        start_lock = threading.Lock()

        def prepare_branch_plan(
            *,
            branch_id,
            capability,
            instance_id='',
            sleep_seconds=0.0,
            runtime_scheduling_context=None,
            excluded_instance_ids=None,
        ):
            route_info = {'capability': capability}
            if instance_id:
                route_info['instance_id'] = instance_id
            return {
                'branch_id': branch_id,
                'phase_id': branch_id,
                'capability': capability,
                'route_info': route_info,
                'effective_data': {
                    'runtime_scheduling_context': dict(runtime_scheduling_context or {}),
                },
                'infer_payload': {
                    'sleep_seconds': sleep_seconds,
                    'runtime_scheduling_context': dict(runtime_scheduling_context or {}),
                },
            }

        def execute_prepared_branch(plan):
            branch_id = plan['branch_id']
            with start_lock:
                started.append(branch_id)
            time.sleep(float((plan.get('infer_payload') or {}).get('sleep_seconds') or 0.0))
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {'ok': True},
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        policy = result['concurrency_policy']
        self.assertEqual(policy['prepared_branch_count'], 5)
        self.assertEqual(policy['distinct_instance_count'], 4)
        self.assertEqual(policy['local_text_io_fast_lane_count'], 1)
        self.assertEqual(policy['scheduling_capacity_units'], 5)
        self.assertEqual(policy['worker_count'], 4)
        self.assertEqual(policy['text_io_fast_lane_branch_ids'], ['repair-html-syntax'])
        self.assertEqual(policy['execution_submission_order'][0], 'repair-html-syntax')
        self.assertEqual(started[0], 'repair-html-syntax')

    def test_no_instance_text_io_repairs_add_local_worker_slots(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=4)
        branches = [
            {
                'branch_id': f'repair-html-syntax-{index}',
                'phase_id': f'repair-html-syntax-{index}',
                'capability': 'chat',
                'repair_scope': 'syntax_only',
                'resource_class': 'text_io',
                'dependency_policy': 'target_artifact_snapshot_only',
                'prepare_args': {
                    'branch_id': f'repair-html-syntax-{index}',
                    'capability': 'chat',
                    'runtime_scheduling_context': {
                        'repair_scope': 'syntax_only',
                        'resource_class': 'text_io',
                        'dependency_policy': 'target_artifact_snapshot_only',
                    },
                },
            }
            for index in range(1, 4)
        ]
        intervals: dict[str, tuple[float, float]] = {}
        interval_lock = threading.Lock()

        def prepare_branch_plan(*, branch_id, capability, runtime_scheduling_context=None, excluded_instance_ids=None):
            return {
                'branch_id': branch_id,
                'phase_id': branch_id,
                'capability': capability,
                'route_info': {'capability': capability},
                'effective_data': {
                    'runtime_scheduling_context': dict(runtime_scheduling_context or {}),
                },
            }

        def execute_prepared_branch(plan):
            branch_id = plan['branch_id']
            start = time.perf_counter()
            time.sleep(0.03)
            end = time.perf_counter()
            with interval_lock:
                intervals[branch_id] = (start, end)
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {'ok': True},
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        policy = result['concurrency_policy']
        self.assertEqual(policy['distinct_instance_count'], 0)
        self.assertEqual(policy['local_text_io_fast_lane_count'], 3)
        self.assertEqual(policy['scheduling_capacity_units'], 3)
        self.assertEqual(policy['worker_count'], 3)
        first = intervals['repair-html-syntax-1']
        second = intervals['repair-html-syntax-2']
        third = intervals['repair-html-syntax-3']
        self.assertTrue(first[0] < second[1] and second[0] < first[1])
        self.assertTrue(first[0] < third[1] and third[0] < first[1])

    def test_multi_materialization_emits_compact_branch_progress_before_return(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=2)
        branches = [
            {
                'branch_id': 'branch-image-1',
                'phase_id': 'phase-image-1',
                'capability': 'image_generation',
                'prepare_args': {'instance_id': 'img-1'},
            },
            {
                'branch_id': 'branch-image-2',
                'phase_id': 'phase-image-2',
                'capability': 'image_generation',
                'prepare_args': {'instance_id': 'img-2', 'should_fail': True},
            },
        ]
        returned = False
        progress_events = []

        def prepare_branch_plan(*, instance_id, should_fail=False, excluded_instance_ids=None):
            return {
                'route_info': {
                    'instance_id': instance_id,
                    'capability': 'image_generation',
                },
                'infer_payload': {
                    'instance_id': instance_id,
                    'should_fail': should_fail,
                },
            }

        def execute_prepared_branch(plan):
            time.sleep(0.01 if plan['branch_id'] == 'branch-image-1' else 0.02)
            if plan.get('infer_payload', {}).get('should_fail'):
                raise RuntimeError('synthetic branch failure')
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {'saved_image_path': f"/tmp/{plan['branch_id']}.png"},
            }

        def on_branch_progress(event):
            progress_events.append({'event': dict(event), 'returned': returned})

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
            on_branch_progress=on_branch_progress,
        )
        returned = True

        self.assertEqual(set(result['branch_results']), {'branch-image-1'})
        self.assertEqual(set(result['branch_errors']), {'branch-image-2'})
        self.assertEqual({item['event']['branch_id'] for item in progress_events}, {'branch-image-1', 'branch-image-2'})
        self.assertTrue(all(item['returned'] is False for item in progress_events))
        events_by_branch = {item['event']['branch_id']: item['event'] for item in progress_events}
        self.assertEqual(events_by_branch['branch-image-1']['status'], 'completed')
        self.assertEqual(events_by_branch['branch-image-1']['instance_id'], 'img-1')
        self.assertEqual(events_by_branch['branch-image-1']['progress_stage'], 'branch_execution')
        self.assertIn('timing', events_by_branch['branch-image-1'])
        self.assertEqual(events_by_branch['branch-image-2']['status'], 'failed')
        self.assertEqual(events_by_branch['branch-image-2']['error']['code'], 'BACKEND_ERROR')
        self.assertEqual(events_by_branch['branch-image-2']['error']['stage'], 'execute_prepared_branch')
        self.assertNotIn('infer_result', events_by_branch['branch-image-1'])
        self.assertNotIn('image_data_url', events_by_branch['branch-image-1'])

    def test_multi_materialization_async_branch_progress_does_not_block_serial_branch_execution(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=4)
        branches = [
            {
                'branch_id': 'branch-image-1',
                'phase_id': 'phase-image-1',
                'capability': 'image_generation',
                'prepare_args': {'instance_id': 'img-1'},
            },
            {
                'branch_id': 'branch-image-2',
                'phase_id': 'phase-image-2',
                'capability': 'image_generation',
                'prepare_args': {'instance_id': 'img-1'},
            },
        ]
        progress_entered = threading.Event()
        release_progress = threading.Event()
        second_branch_started = threading.Event()
        second_progress_finished = threading.Event()
        events_lock = threading.Lock()
        progress_events = []
        result_holder = {}
        error_holder = []

        def prepare_branch_plan(*, instance_id, excluded_instance_ids=None):
            return {
                'route_info': {
                    'instance_id': instance_id,
                    'capability': 'image_generation',
                },
                'infer_payload': {
                    'instance_id': instance_id,
                },
            }

        def execute_prepared_branch(plan):
            branch_id = plan['branch_id']
            if branch_id == 'branch-image-2':
                second_branch_started.set()
            time.sleep(0.01)
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {'saved_image_path': f"/tmp/{branch_id}.png"},
            }

        def on_branch_progress(event):
            branch_id = event['branch_id']
            with events_lock:
                progress_events.append(('start', branch_id))
            if branch_id == 'branch-image-1':
                progress_entered.set()
                release_progress.wait(timeout=2.0)
            with events_lock:
                progress_events.append(('end', branch_id))
            if branch_id == 'branch-image-2':
                second_progress_finished.set()

        def run_materialization():
            try:
                result_holder['result'] = runtime.execute_materialization_branches(
                    branches,
                    prepare_branch_plan=prepare_branch_plan,
                    execute_prepared_branch=execute_prepared_branch,
                    on_branch_progress=on_branch_progress,
                    async_branch_progress=True,
                )
            except Exception as exc:  # noqa: BLE001
                error_holder.append(exc)

        worker = threading.Thread(target=run_materialization)
        worker.start()
        self.assertTrue(progress_entered.wait(timeout=1.0))
        self.assertTrue(second_branch_started.wait(timeout=1.0))
        with events_lock:
            blocked_progress_events = list(progress_events)
        self.assertIn(('start', 'branch-image-1'), blocked_progress_events)
        self.assertNotIn(('end', 'branch-image-1'), blocked_progress_events)
        self.assertTrue(worker.is_alive())

        release_progress.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertFalse(error_holder)
        self.assertTrue(second_progress_finished.is_set())

        result = result_holder['result']
        self.assertEqual(set(result['branch_results']), {'branch-image-1', 'branch-image-2'})
        self.assertEqual(result['branch_errors'], {})
        policy = result['concurrency_policy']
        self.assertEqual(policy['worker_count'], 1)
        self.assertEqual(policy['branch_progress_dispatch'], 'async_ordered')
        self.assertEqual(policy['branch_progress_callback_count'], 2)

    def test_multi_materialization_spread_retry_passes_preferred_instance_order_and_diagnostics(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=3)
        available_instance_ids = ['img-1', 'img-2']
        prepare_calls = []

        branches = [
            {
                'branch_id': f'branch-image-{index}',
                'phase_id': f'phase-image-{index}',
                'capability': 'image_generation',
                'prepare_args': {'branch_index': index, 'artifact_gap': {}},
            }
            for index in range(1, 4)
        ]

        def prepare_branch_plan(*, branch_index, artifact_gap=None, excluded_instance_ids=None):
            excluded = list(excluded_instance_ids or [])
            preferred = list((artifact_gap or {}).get('_spread_retry_preferred_instance_ids') or [])
            prepare_calls.append({'excluded': excluded, 'preferred': preferred})
            selected_instance_id = preferred[0] if preferred else ''
            if not selected_instance_id:
                selected_instance_id = next(
                    (
                        instance_id
                        for instance_id in available_instance_ids
                        if instance_id not in excluded
                    ),
                    '',
                )
            if not selected_instance_id:
                exc = RuntimeError(
                    "No non-excluded running instance found for capability 'image_generation'."
                )
                exc.route_diagnostics = {
                    'candidate_diagnostics': [
                        {'instance_id': instance_id, 'rejection_reasons': ['excluded_instance']}
                        for instance_id in excluded
                    ],
                }
                raise exc
            route_runtime = {}
            if preferred:
                route_runtime = {
                    'selection_policy': 'spread_retry_preferred_instance',
                    'spread_retry_reason': (artifact_gap or {}).get('_spread_retry_reason'),
                    'candidate_diagnostics': [
                        {'instance_id': instance_id, 'selected': instance_id == selected_instance_id}
                        for instance_id in available_instance_ids
                    ],
                }
            return {
                'route_info': {
                    'instance_id': selected_instance_id,
                    'capability': 'image_generation',
                    'route_runtime': route_runtime,
                },
                'infer_payload': {
                    'prompt': f'image prompt {branch_index}',
                    'instance_id': selected_instance_id,
                },
            }

        def execute_prepared_branch(plan):
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {
                    'saved_image_path': f"/tmp/{plan['branch_id']}.png",
                },
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        self.assertEqual(
            [
                entry['result']['route_info']['instance_id']
                for entry in result['ordered_branch_results']
            ],
            ['img-1', 'img-2', 'img-2'],
        )
        self.assertEqual(
            prepare_calls,
            [
                {'excluded': [], 'preferred': []},
                {'excluded': ['img-1'], 'preferred': []},
                {'excluded': ['img-1', 'img-2'], 'preferred': []},
                {'excluded': [], 'preferred': ['img-2', 'img-1']},
            ],
        )
        retry_timing = result['concurrency_policy']['prepare_timings'][-1]
        self.assertEqual(retry_timing['status'], 'ok_after_retry')
        self.assertEqual(retry_timing['initial_error_code'], 'NO_COMPATIBLE_INSTANCE')
        self.assertEqual(retry_timing['selection_policy'], 'spread_retry_preferred_instance')
        self.assertEqual(retry_timing['spread_retry_reason'], 'internal_reservation_exhausted')
        self.assertIn('initial_route_diagnostics', retry_timing)
        self.assertIn('candidate_diagnostics', retry_timing)

    def test_required_text_artifact_large_chat_branches_can_parallelize_on_distinct_instances(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=4)
        branches = [
            {
                'branch_id': 'branch-chat-index',
                'phase_id': 'phase-index',
                'capability': 'chat',
                'output_type': 'text',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': 'html',
                'text_artifact_source_name': 'index',
                'prepare_args': {
                    'branch_id': 'branch-chat-index',
                    'text_artifact_extension': 'html',
                    'text_artifact_source_name': 'index',
                },
            },
            {
                'branch_id': 'branch-chat-styles',
                'phase_id': 'phase-styles',
                'capability': 'chat',
                'output_type': 'text',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': 'css',
                'text_artifact_source_name': 'styles',
                'prepare_args': {
                    'branch_id': 'branch-chat-styles',
                    'text_artifact_extension': 'css',
                    'text_artifact_source_name': 'styles',
                },
            },
        ]
        instance_by_branch = {
            'branch-chat-index': 'chat-1',
            'branch-chat-styles': 'chat-2',
        }
        intervals: dict[str, tuple[float, float]] = {}
        interval_lock = threading.Lock()

        def prepare_branch_plan(
            *,
            branch_id,
            text_artifact_extension,
            text_artifact_source_name,
            excluded_instance_ids=None,
        ):
            instance_id = instance_by_branch[branch_id]
            return {
                'capability': 'chat',
                'branch_id': branch_id,
                'phase_id': branch_id.replace('branch-chat-', 'phase-'),
                'route_info': {
                    'capability': 'chat',
                    'instance_id': instance_id,
                    'server_kind': 'mlx_vlm',
                },
                'instance': {
                    'instance_id': instance_id,
                    'model': 'gemma4:26b',
                    'backend': 'mlx',
                    'capability': 'chat',
                },
                'effective_data': {
                    'capability': 'chat',
                    'output_type': 'text',
                    'requires_artifact': True,
                    'text_artifact_extension': text_artifact_extension,
                    'text_artifact_source_name': text_artifact_source_name,
                    'server_kind': 'mlx_vlm',
                },
                'infer_payload': {
                    'capability': 'chat',
                    'output_type': 'text',
                    'server_kind': 'mlx_vlm',
                },
            }

        def execute_prepared_branch(plan):
            branch_id = plan['branch_id']
            start = time.perf_counter()
            time.sleep(0.08)
            end = time.perf_counter()
            with interval_lock:
                intervals[branch_id] = (start, end)
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {
                    'saved_text_path': f"/tmp/{branch_id}.txt",
                },
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        policy = result['concurrency_policy']
        self.assertEqual(policy['prepared_branch_count'], 2)
        self.assertEqual(policy['distinct_instance_count'], 2)
        self.assertEqual(policy['worker_count'], 2)
        self.assertEqual(policy['gpu_heavy_guard'], 'not_serialized')
        self.assertEqual(policy['distinct_instance_ids'], ['chat-1', 'chat-2'])
        self.assertEqual(policy['same_instance_lock_groups'], {})
        self.assertEqual(
            [item['selected_instance_id'] for item in policy['prepare_timings']],
            ['chat-1', 'chat-2'],
        )
        first = intervals['branch-chat-index']
        second = intervals['branch-chat-styles']
        self.assertTrue(first[0] < second[1] and second[0] < first[1])

    def test_required_text_artifact_large_chat_branches_stay_serial_on_same_instance(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=4)
        branches = [
            {
                'branch_id': 'branch-chat-index',
                'phase_id': 'phase-index',
                'capability': 'chat',
                'output_type': 'text',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': 'html',
                'prepare_args': {'branch_id': 'branch-chat-index'},
            },
            {
                'branch_id': 'branch-chat-styles',
                'phase_id': 'phase-styles',
                'capability': 'chat',
                'output_type': 'text',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': 'css',
                'prepare_args': {'branch_id': 'branch-chat-styles'},
            },
        ]
        intervals: dict[str, tuple[float, float]] = {}
        interval_lock = threading.Lock()

        def prepare_branch_plan(*, branch_id, excluded_instance_ids=None):
            return {
                'capability': 'chat',
                'branch_id': branch_id,
                'phase_id': branch_id.replace('branch-chat-', 'phase-'),
                'route_info': {
                    'capability': 'chat',
                    'instance_id': 'chat-1',
                },
                'instance': {
                    'instance_id': 'chat-1',
                    'model': 'gemma4:26b',
                    'backend': 'mlx',
                    'capability': 'chat',
                },
                'effective_data': {
                    'capability': 'chat',
                    'output_type': 'text',
                    'requires_artifact': True,
                    'text_artifact_extension': 'txt',
                },
            }

        def execute_prepared_branch(plan):
            branch_id = plan['branch_id']
            start = time.perf_counter()
            time.sleep(0.03)
            end = time.perf_counter()
            with interval_lock:
                intervals[branch_id] = (start, end)
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {
                    'saved_text_path': f"/tmp/{branch_id}.txt",
                },
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        policy = result['concurrency_policy']
        self.assertEqual(policy['distinct_instance_count'], 1)
        self.assertEqual(policy['worker_count'], 1)
        self.assertEqual(policy['gpu_heavy_guard'], 'not_serialized')
        self.assertEqual(policy['same_instance_lock_groups'], {'chat-1': ['branch-chat-index', 'branch-chat-styles']})
        first = intervals['branch-chat-index']
        second = intervals['branch-chat-styles']
        self.assertFalse(first[0] < second[1] and second[0] < first[1])

    def test_image_wave_with_required_mlx_text_artifact_uses_worker_budget(self):
        runtime = MultiMaterializationRuntimeOwner(max_parallel_workers=4)
        branches = [
            {
                'branch_id': 'branch-image_generation-1',
                'phase_id': 'phase-2',
                'capability': 'image_generation',
                'prepare_args': {'branch_id': 'branch-image_generation-1', 'instance_id': 'image-1'},
            },
            {
                'branch_id': 'branch-image_generation-2',
                'phase_id': 'phase-3',
                'capability': 'image_generation',
                'prepare_args': {'branch_id': 'branch-image_generation-2', 'instance_id': 'image-2'},
            },
            {
                'branch_id': 'branch-image_generation-3',
                'phase_id': 'phase-4',
                'capability': 'image_generation',
                'prepare_args': {'branch_id': 'branch-image_generation-3', 'instance_id': 'image-2'},
            },
            {
                'branch_id': 'coalesced-text-artifacts-repair-chat-repair-chat-7',
                'phase_id': 'coalesced-text-artifacts-repair-chat-repair-chat-7',
                'capability': 'chat',
                'output_type': 'text',
                'stage_direction': 'materialize_requested_text_artifact',
                'requires_artifact': True,
                'text_artifact_extension': 'html',
                'text_artifact_source_name': 'index',
                'prepare_args': {
                    'branch_id': 'coalesced-text-artifacts-repair-chat-repair-chat-7',
                    'instance_id': 'mlx-text-1',
                    'text_artifact_extension': 'html',
                },
            },
        ]

        def prepare_branch_plan(
            *,
            branch_id,
            instance_id,
            text_artifact_extension='',
            excluded_instance_ids=None,
        ):
            capability = 'chat' if branch_id.startswith('coalesced-text-artifacts-') else 'image_generation'
            is_text = capability == 'chat'
            return {
                'branch_id': branch_id,
                'phase_id': branch_id,
                'capability': capability,
                'route_info': {
                    'capability': capability,
                    'instance_id': instance_id,
                    **({'server_kind': 'mlx_vlm'} if is_text else {}),
                },
                'instance': {
                    'instance_id': instance_id,
                    'model': 'mlx-community/gemma-4-e4b-8bit' if is_text else 'x/flux2-klein:latest',
                    'backend': 'mlx' if is_text else 'ollama',
                    'capability': capability,
                },
                'effective_data': {
                    'capability': capability,
                    **(
                        {
                            'output_type': 'text',
                            'requires_artifact': True,
                            'text_artifact_extension': text_artifact_extension or 'html',
                            'server_kind': 'mlx_vlm',
                        }
                        if is_text
                        else {}
                    ),
                },
                'infer_payload': {
                    'capability': capability,
                    **({'server_kind': 'mlx_vlm'} if is_text else {}),
                },
            }

        def execute_prepared_branch(plan):
            return {
                'route_info': dict(plan.get('route_info') or {}),
                'infer_result': {
                    'saved_text_path': f"/tmp/{plan['branch_id']}.html"
                    if plan['capability'] == 'chat'
                    else None,
                    'saved_image_path': f"/tmp/{plan['branch_id']}.png"
                    if plan['capability'] == 'image_generation'
                    else None,
                },
            }

        result = runtime.execute_materialization_branches(
            branches,
            prepare_branch_plan=prepare_branch_plan,
            execute_prepared_branch=execute_prepared_branch,
        )

        self.assertEqual(result['ordered_branch_errors'], [])
        policy = result['concurrency_policy']
        self.assertEqual(policy['prepared_branch_count'], 4)
        self.assertEqual(policy['distinct_instance_count'], 3)
        self.assertEqual(policy['default_worker_count'], 3)
        self.assertEqual(policy['worker_count'], 3)
        self.assertEqual(policy['gpu_heavy_guard'], 'not_serialized')
        self.assertNotIn('reason', policy)
        self.assertEqual(
            policy['same_instance_lock_groups'],
            {'image-2': ['branch-image_generation-2', 'branch-image_generation-3']},
        )

    def test_webserver_parallel_worker_env_knob_is_read_at_import(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = '.'
        env['OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS'] = '2'
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import ollmo_webserver; print(ollmo_webserver._MULTI_MATERIALIZATION_RUNTIME.max_parallel_workers)',
            ],
            cwd=os.getcwd(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), '2')


if __name__ == '__main__':
    unittest.main()
