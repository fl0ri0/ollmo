"""Shared multi-materialization orchestration for same-level sibling outputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional


DEFAULT_MAX_PARALLEL_WORKERS = 4
MIN_MAX_PARALLEL_WORKERS = 1
MAX_MAX_PARALLEL_WORKERS = 16


def _normalized_token(value: Any) -> str:
    return str(value or '').strip()


def _nested_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _plan_payloads(plan: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            payloads.append(value)

    add(plan)
    for key in ('route_info', 'instance', 'effective_data', 'infer_payload', 'execution_contract'):
        add(plan.get(key))
    branch = _nested_mapping(plan.get('branch'))
    add(branch)
    prepare_args = _nested_mapping(branch.get('prepare_args'))
    add(prepare_args)
    artifact_gap = _nested_mapping(prepare_args.get('artifact_gap'))
    add(artifact_gap)
    for payload in list(payloads):
        for key in ('runtime_scheduling_context', 'execution_contract', 'artifact_request'):
            add(payload.get(key))
    return payloads


def _plan_is_text_io_repair_fast_lane(plan: dict[str, Any]) -> bool:
    payloads = _plan_payloads(plan)
    resource_classes = {
        _normalized_token(payload.get('resource_class')).lower()
        for payload in payloads
        if _normalized_token(payload.get('resource_class'))
    }
    repair_scopes = {
        _normalized_token(payload.get('repair_scope')).lower()
        for payload in payloads
        if _normalized_token(payload.get('repair_scope'))
    }
    dependency_policies = {
        _normalized_token(payload.get('dependency_policy')).lower()
        for payload in payloads
        if _normalized_token(payload.get('dependency_policy'))
    }
    check_kinds = {
        _normalized_token(payload.get('check_kind')).lower()
        for payload in payloads
        if _normalized_token(payload.get('check_kind'))
    }
    roles = {
        _normalized_token(payload.get('role')).lower()
        for payload in payloads
        if _normalized_token(payload.get('role'))
    }
    payload_sources = {
        _normalized_token(payload.get('content_payload_source')).lower()
        for payload in payloads
        if _normalized_token(payload.get('content_payload_source'))
    }
    artifact_sources = {
        _normalized_token(payload.get('text_artifact_source')).lower()
        for payload in payloads
        if _normalized_token(payload.get('text_artifact_source'))
    }
    text_io = bool(resource_classes & {'text_io', 'local_text_io'})
    target_snapshot_only = 'target_artifact_snapshot_only' in dependency_policies
    syntax_only = bool(repair_scopes & {'syntax_only', 'text_artifact_syntax'})
    syntax_repair_marker = (
        'text_artifact_syntax_sanity' in check_kinds
        or 'text_artifact_syntax_repair' in roles
        or 'closure_text_artifact_syntax_sanity' in payload_sources
        or 'closure_syntax_repair' in artifact_sources
    )
    return target_snapshot_only or (text_io and (syntax_only or syntax_repair_marker))


def _plan_instance_id(plan: dict[str, Any]) -> str:
    route_info = _nested_mapping(plan.get('route_info'))
    instance = _nested_mapping(plan.get('instance'))
    return _normalized_token(route_info.get('instance_id') or instance.get('instance_id'))


def _compact_plan_scheduling_summary(plan: dict[str, Any]) -> dict[str, Any]:
    route_info = _nested_mapping(plan.get('route_info'))
    instance = _nested_mapping(plan.get('instance'))
    summary = {
        'branch_id': _normalized_token(plan.get('branch_id')),
        'phase_id': _normalized_token(plan.get('phase_id')),
        'capability': _normalized_token(plan.get('capability') or route_info.get('capability')),
        'instance_id': _plan_instance_id(plan),
        'model': _normalized_token(
            instance.get('model')
            or instance.get('model_name')
            or route_info.get('model')
            or plan.get('model')
        ),
        'backend': _normalized_token(instance.get('backend') or route_info.get('backend') or plan.get('backend')),
    }
    return {key: value for key, value in summary.items() if value not in (None, '', [], {})}


def normalize_max_parallel_workers(
    value: Any,
    *,
    default: int = DEFAULT_MAX_PARALLEL_WORKERS,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(MIN_MAX_PARALLEL_WORKERS, min(parsed, MAX_MAX_PARALLEL_WORKERS))


def _error_code_for_exception(exc: Exception, *, stage: str, status_code: Optional[int]) -> str:
    message = str(exc or '').strip().lower()
    if stage == 'prepare_branch_plan':
        if any(token in message for token in ('no running instance', 'no selectable instance', 'no non-excluded', 'could not resolve')):
            return 'NO_COMPATIBLE_INSTANCE'
        if any(token in message for token in ('unsupported capability', 'capability unsupported')):
            return 'CAPABILITY_UNSUPPORTED'
        if any(token in message for token in ('missing required', 'session control', 'validation')):
            return 'CONTROL_VALIDATION_FAILED'
        return 'PREPARE_FAILED'
    if status_code in {408, 504} or any(token in message for token in ('timeout', 'timed out')):
        return 'BACKEND_TIMEOUT'
    if status_code in {404, 503} or any(token in message for token in ('unavailable', 'connection refused', 'offline')):
        return 'INSTANCE_UNAVAILABLE'
    if status_code in {400, 401, 403, 409, 422}:
        return 'BACKEND_REJECTED'
    if status_code is not None and status_code >= 500:
        return 'BACKEND_ERROR'
    return 'BACKEND_ERROR'


def _error_retryable(code: str, *, stage: str) -> bool:
    if code in {'CAPABILITY_UNSUPPORTED', 'CONTROL_VALIDATION_FAILED'}:
        return False
    if code == 'PREPARE_FAILED':
        return False
    return True


def _exception_error_payload(exc: Exception, *, stage: str) -> dict[str, Any]:
    message = str(exc or '').strip() or exc.__class__.__name__
    payload: dict[str, Any] = {
        'message': message,
        'stage': stage,
    }
    status_code = getattr(exc, 'status_code', None)
    try:
        normalized_status_code = int(status_code)
    except (TypeError, ValueError):
        normalized_status_code = None
    code = _error_code_for_exception(exc, stage=stage, status_code=normalized_status_code)
    payload['code'] = code
    payload['retryable'] = _error_retryable(code, stage=stage)
    exception_type = exc.__class__.__name__
    if exception_type:
        payload['exception_type'] = exception_type
    if normalized_status_code is not None:
        payload['status_code'] = normalized_status_code
    route_diagnostics = getattr(exc, 'route_diagnostics', None)
    if isinstance(route_diagnostics, dict):
        payload['route_diagnostics'] = dict(route_diagnostics)
    return payload


@dataclass
class MultiMaterializationRuntimeOwner:
    max_parallel_workers: int = DEFAULT_MAX_PARALLEL_WORKERS

    def __post_init__(self) -> None:
        self.max_parallel_workers = normalize_max_parallel_workers(self.max_parallel_workers)

    def execute_materialization_branches(
        self,
        branches: list[dict[str, Any]],
        *,
        prepare_branch_plan: Callable[..., dict[str, Any]],
        execute_prepared_branch: Callable[[dict[str, Any]], dict[str, Any]],
        max_workers: Optional[int] = None,
        on_branch_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        async_branch_progress: bool = False,
    ) -> dict[str, Any]:
        normalized_branches: list[dict[str, Any]] = []
        prepared_branch_plans: list[dict[str, Any]] = []
        branch_results: dict[str, dict[str, Any]] = {}
        branch_errors: dict[str, dict[str, Any]] = {}
        reserved_instance_ids_by_group: dict[str, list[str]] = {}
        reserved_instance_counts_by_group: dict[str, dict[str, int]] = {}
        prepare_timings: list[dict[str, Any]] = []
        planning_start = time.perf_counter()

        for raw_branch in branches:
            if not isinstance(raw_branch, dict):
                continue
            branch = dict(raw_branch)
            branch_id = _normalized_token(branch.get('branch_id') or branch.get('phase_id'))
            capability = _normalized_token(branch.get('capability'))
            if not branch_id or not capability:
                continue
            branch['branch_id'] = branch_id
            branch['phase_id'] = _normalized_token(branch.get('phase_id') or branch_id) or branch_id
            branch['capability'] = capability
            normalized_branches.append(branch)

            reservation_group = _normalized_token(branch.get('reservation_group') or capability) or capability
            prepare_args = (
                dict(branch.get('prepare_args') or {})
                if isinstance(branch.get('prepare_args'), dict)
                else {}
            )
            caller_excluded = [
                _normalized_token(item)
                for item in (prepare_args.pop('excluded_instance_ids', []) or [])
                if _normalized_token(item)
            ]
            branch_request_payload = (
                prepare_args.get('branch_request_payload')
                if isinstance(prepare_args.get('branch_request_payload'), dict)
                else {}
            )
            explicit_pin = _normalized_token(prepare_args.get('forced_instance_id'))
            reserved_instance_ids = [
                _normalized_token(item)
                for item in (reserved_instance_ids_by_group.get(reservation_group) or [])
                if _normalized_token(item)
            ] if not explicit_pin or caller_excluded else []

            def build_excluded_instance_ids(
                *,
                include_internal_reservations: bool = True,
            ) -> list[str]:
                excluded_instance_ids: list[str] = []
                active_reserved_ids = reserved_instance_ids if include_internal_reservations else []
                for instance_id in [*caller_excluded, *active_reserved_ids]:
                    if instance_id and instance_id not in excluded_instance_ids:
                        excluded_instance_ids.append(instance_id)
                return excluded_instance_ids

            def record_prepared_plan(plan: dict[str, Any]) -> None:
                normalized_plan = dict(plan or {})
                normalized_plan['branch_id'] = branch_id
                normalized_plan['phase_id'] = branch['phase_id']
                normalized_plan['capability'] = capability
                normalized_plan['branch'] = branch
                prepared_branch_plans.append(normalized_plan)
                chosen_instance_id = _normalized_token(
                    ((normalized_plan.get('route_info') or {}).get('instance_id') or '')
                )
                if chosen_instance_id:
                    group_reservations = reserved_instance_ids_by_group.setdefault(reservation_group, [])
                    if chosen_instance_id not in group_reservations:
                        group_reservations.append(chosen_instance_id)
                    group_counts = reserved_instance_counts_by_group.setdefault(reservation_group, {})
                    group_counts[chosen_instance_id] = int(group_counts.get(chosen_instance_id) or 0) + 1

            def prepare_with_exclusions(
                excluded_instance_ids: list[str],
                *,
                spread_retry_preferred_instance_ids: Optional[list[str]] = None,
                spread_retry_reason: str = '',
            ) -> dict[str, Any]:
                plan_prepare_args = dict(prepare_args)
                plan_prepare_args['excluded_instance_ids'] = list(excluded_instance_ids)
                preferred_ids = [
                    _normalized_token(item)
                    for item in (spread_retry_preferred_instance_ids or [])
                    if _normalized_token(item)
                ]
                if preferred_ids and 'artifact_gap' in plan_prepare_args:
                    gap = (
                        dict(plan_prepare_args.get('artifact_gap') or {})
                        if isinstance(plan_prepare_args.get('artifact_gap'), dict)
                        else {}
                    )
                    gap['_spread_retry_preferred_instance_ids'] = preferred_ids
                    gap['_spread_retry_reason'] = spread_retry_reason or 'internal_reservation_exhausted'
                    plan_prepare_args['artifact_gap'] = gap
                return prepare_branch_plan(**plan_prepare_args)

            excluded_instance_ids = build_excluded_instance_ids()
            prepare_started_at = time.perf_counter()
            prepare_attempt_count = 1
            initial_error_payload: Optional[dict[str, Any]] = None

            def record_prepare_timing(
                *,
                status: str,
                plan: Optional[dict[str, Any]] = None,
                error: Optional[dict[str, Any]] = None,
                excluded: Optional[list[str]] = None,
            ) -> None:
                finished_at = time.perf_counter()
                selected_instance_id = _plan_instance_id(plan or {}) if isinstance(plan, dict) else ''
                timing = {
                    'branch_id': branch_id,
                    'phase_id': branch['phase_id'],
                    'capability': capability,
                    'status': status,
                    'attempt_count': prepare_attempt_count,
                    'queued_elapsed_ms': round((prepare_started_at - planning_start) * 1000, 3),
                    'prepare_elapsed_ms': round(max(0.0, finished_at - prepare_started_at) * 1000, 3),
                    'excluded_instance_ids': list(excluded or []),
                    'selected_instance_id': selected_instance_id or None,
                }
                if error:
                    timing['error_code'] = error.get('code')
                    timing['error_stage'] = error.get('stage')
                    if isinstance(error.get('route_diagnostics'), dict):
                        timing['route_diagnostics'] = dict(error.get('route_diagnostics') or {})
                if initial_error_payload:
                    timing['initial_error_code'] = initial_error_payload.get('code')
                    if isinstance(initial_error_payload.get('route_diagnostics'), dict):
                        timing['initial_route_diagnostics'] = dict(
                            initial_error_payload.get('route_diagnostics') or {}
                        )
                if isinstance(plan, dict):
                    route_runtime = (
                        ((plan.get('route_info') or {}).get('route_runtime') or {})
                        if isinstance(plan.get('route_info'), dict)
                        else {}
                    )
                    if isinstance(route_runtime, dict):
                        if route_runtime.get('selection_policy'):
                            timing['selection_policy'] = route_runtime.get('selection_policy')
                        if route_runtime.get('spread_retry_reason'):
                            timing['spread_retry_reason'] = route_runtime.get('spread_retry_reason')
                        if isinstance(route_runtime.get('candidate_diagnostics'), list):
                            timing['candidate_diagnostics'] = list(route_runtime.get('candidate_diagnostics') or [])
                prepare_timings.append(
                    {
                        key: value
                        for key, value in timing.items()
                        if value not in (None, '', [], {})
                    }
                )

            try:
                plan = prepare_with_exclusions(excluded_instance_ids)
                record_prepared_plan(plan)
                record_prepare_timing(status='ok', plan=plan, excluded=excluded_instance_ids)
            except Exception as exc:  # noqa: BLE001
                error_payload = _exception_error_payload(exc, stage='prepare_branch_plan')
                initial_error_payload = dict(error_payload)
                can_retry_without_internal_reservations = (
                    bool(reserved_instance_ids)
                    and not explicit_pin
                    and error_payload.get('code') == 'NO_COMPATIBLE_INSTANCE'
                )
                if can_retry_without_internal_reservations:
                    reserved_instance_ids_by_group[reservation_group] = []
                    try:
                        prepare_attempt_count += 1
                        retry_excluded_instance_ids = build_excluded_instance_ids(include_internal_reservations=False)
                        group_counts = dict(reserved_instance_counts_by_group.get(reservation_group) or {})
                        reservation_order = list(reserved_instance_ids)
                        order_index = {
                            instance_id: index
                            for index, instance_id in enumerate(reservation_order)
                        }
                        spread_retry_preferred_instance_ids = sorted(
                            reservation_order,
                            key=lambda instance_id: (
                                int(group_counts.get(instance_id) or 0),
                                -int(order_index.get(instance_id) or 0),
                            ),
                        )
                        plan = prepare_with_exclusions(
                            retry_excluded_instance_ids,
                            spread_retry_preferred_instance_ids=spread_retry_preferred_instance_ids,
                            spread_retry_reason='internal_reservation_exhausted',
                        )
                        record_prepared_plan(plan)
                        record_prepare_timing(status='ok_after_retry', plan=plan, excluded=retry_excluded_instance_ids)
                        continue
                    except Exception as retry_exc:  # noqa: BLE001
                        error_payload = _exception_error_payload(
                            retry_exc,
                            stage='prepare_branch_plan',
                        )
                branch_errors[branch_id] = error_payload
                record_prepare_timing(status='error', error=error_payload, excluded=excluded_instance_ids)

        if prepared_branch_plans:
            instance_locks: dict[str, threading.Lock] = {}
            concurrency_policy: dict[str, Any] = {}
            branch_timings: dict[str, dict[str, Any]] = {}
            branch_timing_lock = threading.Lock()
            wave_start = time.perf_counter()
            progress_executor: Optional[ThreadPoolExecutor] = (
                ThreadPoolExecutor(max_workers=1)
                if async_branch_progress and on_branch_progress
                else None
            )
            progress_futures: list[Any] = []

            def dispatch_branch_progress(event: dict[str, Any]) -> None:
                if not on_branch_progress:
                    return
                try:
                    on_branch_progress(event)
                except Exception:  # noqa: BLE001
                    return

            def drain_branch_progress() -> None:
                nonlocal progress_executor
                for future in progress_futures:
                    try:
                        future.result()
                    except Exception:  # noqa: BLE001
                        continue
                if progress_executor is not None:
                    progress_executor.shutdown(wait=True)
                    progress_executor = None

            def compact_branch_error(error: Optional[dict[str, Any]]) -> dict[str, Any]:
                if not isinstance(error, dict):
                    return {}
                compact: dict[str, Any] = {}
                for key in (
                    'code',
                    'message',
                    'stage',
                    'exception_type',
                    'status_code',
                    'retryable',
                ):
                    value = error.get(key)
                    if value not in (None, '', [], {}):
                        compact[key] = value
                return compact

            def emit_branch_progress(
                plan: dict[str, Any],
                *,
                status: str,
                error: Optional[dict[str, Any]] = None,
            ) -> None:
                if not on_branch_progress:
                    return
                branch_id = _normalized_token(plan.get('branch_id'))
                if not branch_id:
                    return
                with branch_timing_lock:
                    timing = dict(branch_timings.get(branch_id) or {})
                event = {
                    'branch_id': branch_id,
                    'phase_id': _normalized_token(plan.get('phase_id') or branch_id) or branch_id,
                    'capability': _normalized_token(plan.get('capability')),
                    'status': status,
                    'progress_stage': 'branch_execution',
                    'instance_id': _plan_instance_id(plan) or None,
                    'timing': timing,
                    'error': compact_branch_error(error),
                }
                compact_event = {
                    key: value
                    for key, value in event.items()
                    if value not in (None, '', [], {})
                }
                if progress_executor is not None:
                    progress_futures.append(progress_executor.submit(dispatch_branch_progress, compact_event))
                    return
                dispatch_branch_progress(compact_event)

            def execute_plan(plan: dict[str, Any]) -> dict[str, Any]:
                branch_id = _normalized_token(plan.get('branch_id'))
                instance_id = _plan_instance_id(plan)
                queued_at = time.perf_counter()
                lock_acquired_at = queued_at
                finished_at = queued_at
                status = 'ok'
                try:
                    if not instance_id:
                        return execute_prepared_branch(dict(plan))
                    lock = instance_locks.setdefault(instance_id, threading.Lock())
                    with lock:
                        lock_acquired_at = time.perf_counter()
                        return execute_prepared_branch(dict(plan))
                except Exception:
                    status = 'error'
                    raise
                finally:
                    finished_at = time.perf_counter()
                    if branch_id:
                        timing = {
                            'branch_id': branch_id,
                            'instance_id': instance_id or None,
                            'status': status,
                            'queued_elapsed_ms': round((queued_at - wave_start) * 1000, 3),
                            'lock_wait_ms': round(max(0.0, lock_acquired_at - queued_at) * 1000, 3),
                            'execution_ms': round(max(0.0, finished_at - lock_acquired_at) * 1000, 3),
                            'elapsed_ms': round(max(0.0, finished_at - queued_at) * 1000, 3),
                        }
                        with branch_timing_lock:
                            branch_timings[branch_id] = {
                                key: value
                                for key, value in timing.items()
                                if value not in (None, '', [], {})
                            }

            distinct_instance_ids = {
                _plan_instance_id(plan)
                for plan in prepared_branch_plans
                if _plan_instance_id(plan)
            }
            instance_branch_groups: dict[str, list[str]] = {}
            for plan in prepared_branch_plans:
                instance_id = _plan_instance_id(plan)
                branch_id = _normalized_token(plan.get('branch_id'))
                if instance_id and branch_id:
                    instance_branch_groups.setdefault(instance_id, []).append(branch_id)
            fast_lane_branch_ids = [
                _normalized_token(plan.get('branch_id'))
                for plan in prepared_branch_plans
                if _plan_is_text_io_repair_fast_lane(plan)
                and _normalized_token(plan.get('branch_id'))
            ]
            local_fast_lane_count = sum(
                1
                for plan in prepared_branch_plans
                if _plan_is_text_io_repair_fast_lane(plan)
                and not _plan_instance_id(plan)
            )
            scheduling_capacity_units = len(distinct_instance_ids) + local_fast_lane_count
            default_workers = max(1, min(len(prepared_branch_plans), scheduling_capacity_units or 1, self.max_parallel_workers))
            worker_count = max(1, int(max_workers or default_workers))
            execution_branch_plans = [
                plan
                for _index, plan in sorted(
                    enumerate(prepared_branch_plans),
                    key=lambda item: (
                        0 if _plan_is_text_io_repair_fast_lane(item[1]) else 1,
                        item[0],
                    ),
                )
            ]
            concurrency_policy = {
                'scheduler': 'multi_materialization_runtime',
                'prepared_branch_count': len(prepared_branch_plans),
                'max_parallel_workers': self.max_parallel_workers,
                'default_worker_count': default_workers,
                'worker_count': worker_count,
                'worker_count_source': 'explicit_override' if max_workers else 'default',
                'scheduling_capacity_units': scheduling_capacity_units,
                'local_text_io_fast_lane_count': local_fast_lane_count,
                'text_io_fast_lane_branch_ids': fast_lane_branch_ids,
                'execution_submission_order': [
                    _normalized_token(plan.get('branch_id'))
                    for plan in execution_branch_plans
                    if _normalized_token(plan.get('branch_id'))
                ],
                'distinct_instance_count': len(distinct_instance_ids),
                'distinct_instance_ids': sorted(distinct_instance_ids),
                'instance_branch_groups': {
                    instance_id: list(branch_ids)
                    for instance_id, branch_ids in sorted(instance_branch_groups.items())
                },
                'same_instance_lock_groups': {
                    instance_id: list(branch_ids)
                    for instance_id, branch_ids in sorted(instance_branch_groups.items())
                    if len(branch_ids) > 1
                },
                'prepared_branches': [
                    _compact_plan_scheduling_summary(plan)
                    for plan in prepared_branch_plans
                ],
                'prepare_timings': [
                    dict(timing)
                    for timing in prepare_timings
                    if timing.get('branch_id') in {
                        _normalized_token(plan.get('branch_id'))
                        for plan in prepared_branch_plans
                    }
                ],
                'planning_elapsed_ms': round(max(0.0, time.perf_counter() - planning_start) * 1000, 3),
                'gpu_heavy_guard': 'not_serialized',
                'branch_progress_dispatch': (
                    'async_ordered'
                    if progress_executor is not None
                    else ('sync' if on_branch_progress else 'none')
                ),
            }
            try:
                if worker_count == 1:
                    for plan in execution_branch_plans:
                        branch_id = _normalized_token(plan.get('branch_id'))
                        try:
                            branch_results[branch_id] = execute_plan(plan)
                            emit_branch_progress(plan, status='completed')
                        except Exception as exc:  # noqa: BLE001
                            error_payload = _exception_error_payload(exc, stage='execute_prepared_branch')
                            branch_errors[branch_id] = error_payload
                            emit_branch_progress(plan, status='failed', error=error_payload)
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        future_map = {
                            executor.submit(execute_plan, dict(plan)): dict(plan)
                            for plan in execution_branch_plans
                        }
                        for future in as_completed(future_map):
                            plan = future_map[future]
                            branch_id = _normalized_token(plan.get('branch_id'))
                            try:
                                branch_results[branch_id] = future.result()
                                emit_branch_progress(plan, status='completed')
                            except Exception as exc:  # noqa: BLE001
                                error_payload = _exception_error_payload(exc, stage='execute_prepared_branch')
                                branch_errors[branch_id] = error_payload
                                emit_branch_progress(plan, status='failed', error=error_payload)
            finally:
                drain_branch_progress()
            concurrency_policy['branch_progress_callback_count'] = len(progress_futures)
            concurrency_policy['branch_timings'] = [
                branch_timings[branch_id]
                for branch_id in [
                    _normalized_token(plan.get('branch_id'))
                    for plan in prepared_branch_plans
                    if _normalized_token(plan.get('branch_id'))
                ]
                if branch_id in branch_timings
            ]
            concurrency_policy['elapsed_ms'] = round(max(0.0, time.perf_counter() - wave_start) * 1000, 3)
        else:
            concurrency_policy = {}

        ordered_branch_results: list[dict[str, Any]] = []
        ordered_branch_errors: list[dict[str, Any]] = []
        for branch in normalized_branches:
            branch_id = _normalized_token(branch.get('branch_id'))
            if branch_id in branch_results:
                ordered_branch_results.append(
                    {
                        'branch': branch,
                        'result': branch_results[branch_id],
                    }
                )
            elif branch_id in branch_errors:
                ordered_branch_errors.append(
                    {
                        'branch': branch,
                        'error': branch_errors[branch_id],
                    }
                )

        return {
            'prepared_branch_plans': prepared_branch_plans,
            'branch_results': branch_results,
            'branch_errors': branch_errors,
            'ordered_branch_results': ordered_branch_results,
            'ordered_branch_errors': ordered_branch_errors,
            'concurrency_policy': concurrency_policy,
        }
