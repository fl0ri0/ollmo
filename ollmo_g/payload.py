"""Self-describing runtime intelligence helpers for Ollmo."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    build_feature_contract,
    infer_capability,
    normalize_backend,
    normalize_capability,
)
from ollmo_g.ghost_mode_compat import (
    DEFAULT_LEGACY_MODE,
    build_legacy_mode_catalog,
)
from ollmo_g.semantic_role_profile import (
    build_self_modification_contract,
)
from ollmo_g.semantic_roles import (
    build_semantic_role_catalog,
)
from ollmo_core.runtime_liveness import (
    runtime_instance_liveness,
    runtime_instance_score,
)
from ollmo_g.memory import (
    build_recent_self_observations,
    build_self_healing_hints,
)
from ollmo_services.self_learning import (
    DEFAULT_ACCEPTED_POLICY_SNAPSHOT,
    DEFAULT_SELF_LEARNING_DIR,
    build_accepted_learning_runtime_hints,
    build_self_learning_report,
    load_accepted_learning_policy_snapshot,
)

GHOST_DISCOVERY_VERSION = 1
DEFAULT_GHOST_GUIDE_PATH = Path('GHOST.md')
DEFAULT_RUNTIME_LOG_PATH = Path('logs/flask_webserver.log')
DEFAULT_EVENT_LOG_PATH = Path('state/events.jsonl')
DEFAULT_RESPONSE_FRAME_LEDGER_PATH = Path('state/response_frames/responses.jsonl')
DEFAULT_ACCEPTED_LEARNING_POLICY_PATH = DEFAULT_SELF_LEARNING_DIR / DEFAULT_ACCEPTED_POLICY_SNAPSHOT

_CAPABILITY_ORDER = (
    CAPABILITY_CHAT,
    CAPABILITY_VISION_ANALYSIS,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
)

_CAPABILITY_LABELS = {
    CAPABILITY_CHAT: 'chat',
    CAPABILITY_VISION_ANALYSIS: 'vision / OCR',
    CAPABILITY_IMAGE_GENERATION: 'image generation',
    CAPABILITY_SPEECH_TO_TEXT: 'speech to text',
    CAPABILITY_TEXT_TO_SPEECH: 'text to speech',
}


def _readiness_rank(value: object) -> int:
    token = str(value or '').strip().lower()
    if token == 'ready':
        return 3
    if token in {'started', 'idle'}:
        return 2
    if token in {'degraded', 'unreachable'}:
        return 1
    return 0


def _activity_rank(value: object) -> int:
    token = str(value or '').strip().lower()
    if token in {'idle', 'ready'}:
        return 2
    if token in {'busy', 'working'}:
        return 1
    return 0


def _pick_recommended_instance(candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not candidates:
        return None

    def _score(item: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, str]:
        return runtime_instance_score(item, capability=item.get('capability'))

    ranked = sorted(candidates, key=_score, reverse=True)
    if ranked and runtime_instance_liveness(ranked[0], capability=ranked[0].get('capability')).get('selectable'):
        return ranked[0]
    return None


def _normalize_instances(instances: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in instances:
        if not isinstance(entry, dict):
            continue
        instance_id = str(entry.get('instance_id') or '').strip()
        if not instance_id:
            continue
        runtime_status = entry.get('runtime_status') if isinstance(entry.get('runtime_status'), dict) else {}
        model_name = str(entry.get('model') or entry.get('modelName') or '').strip()
        backend = normalize_backend(entry.get('backend'))
        capability = normalize_capability(entry.get('capability')) or infer_capability(model_name, backend)
        feature_contract = build_feature_contract(model_name, backend, capability, metadata=entry)
        process_alive = entry.get('process_alive') if entry.get('process_alive') is not None else runtime_status.get('process_alive')
        port_listening = entry.get('port_listening') if entry.get('port_listening') is not None else runtime_status.get('port_listening')
        normalized.append(
            {
                'instance_id': instance_id,
                'model': model_name,
                'backend': backend,
                'capability': capability,
                'port': entry.get('port'),
                'readiness': str(runtime_status.get('readiness') or entry.get('readiness') or '').strip() or None,
                'status': str(runtime_status.get('status') or entry.get('status') or '').strip() or None,
                'activity': str(runtime_status.get('activity') or entry.get('activity') or '').strip() or None,
                'process_alive': process_alive,
                'port_listening': port_listening,
                'last_error': str(runtime_status.get('last_error') or entry.get('last_error') or '').strip() or None,
                'last_error_at': str(runtime_status.get('last_error_at') or entry.get('last_error_at') or '').strip() or None,
                'cooldown_until': str(runtime_status.get('cooldown_until') or entry.get('cooldown_until') or '').strip() or None,
                'failure_cooldown_until': str(runtime_status.get('failure_cooldown_until') or entry.get('failure_cooldown_until') or '').strip() or None,
                'cooldown_capability': normalize_capability(runtime_status.get('cooldown_capability') or entry.get('cooldown_capability')) or None,
                'features': feature_contract.get('features') or {},
                'feature_sources': feature_contract.get('feature_sources') or {},
                'inputs': feature_contract.get('inputs') or [],
                'outputs': feature_contract.get('outputs') or [],
            }
        )
    return normalized


def _build_capability_map(instances: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in instances:
        groups.setdefault(str(item.get('capability') or 'unknown'), []).append(item)

    payload: dict[str, dict[str, Any]] = {}
    for capability in list(_CAPABILITY_ORDER) + [key for key in groups.keys() if key not in _CAPABILITY_ORDER]:
        candidates = groups.get(capability, [])
        recommended = _pick_recommended_instance(candidates)
        default_instance_id = None
        if recommended:
            default_instance_id = str(recommended.get('instance_id') or '').strip() or None
        payload[capability] = {
            'label': _CAPABILITY_LABELS.get(capability, capability.replace('_', ' ')),
            'count': len(candidates),
            'default_instance_id': default_instance_id,
            'candidates': candidates,
        }
    return payload


def _summarize_counts(instances: list[dict[str, Any]]) -> dict[str, Any]:
    readiness_counts = Counter(str(item.get('readiness') or 'unknown') for item in instances)
    backend_counts = Counter(str(item.get('backend') or 'unknown') for item in instances)
    capability_counts = Counter(str(item.get('capability') or 'unknown') for item in instances)
    return {
        'instances': len(instances),
        'ready': readiness_counts.get('ready', 0),
        'failed': readiness_counts.get('failed', 0),
        'degraded': readiness_counts.get('degraded', 0),
        'by_readiness': dict(readiness_counts),
        'by_backend': dict(backend_counts),
        'by_capability': dict(capability_counts),
    }


def _build_recommendations(capabilities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for capability in _CAPABILITY_ORDER:
        entry = capabilities.get(capability) or {}
        default_instance_id = str(entry.get('default_instance_id') or '').strip()
        if not default_instance_id:
            continue
        candidates = entry.get('candidates') if isinstance(entry.get('candidates'), list) else []
        chosen = next((item for item in candidates if str(item.get('instance_id') or '').strip() == default_instance_id), None)
        if not chosen:
            continue
        readiness = str(chosen.get('readiness') or 'unknown').strip()
        reason = f"best current {entry.get('label') or capability} match"
        if readiness:
            reason += f" ({readiness})"
        items.append(
            {
                'slot': capability,
                'label': entry.get('label') or capability,
                'instance_id': default_instance_id,
                'model': chosen.get('model'),
                'backend': chosen.get('backend'),
                'reason': reason,
            }
        )
    return items


def _build_issues(instances: list[dict[str, Any]], capabilities: dict[str, dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if not instances:
        issues.append('No running instances. Start a model before routing prompts through Ollmo.')
        return issues

    for item in instances:
        readiness = str(item.get('readiness') or '').strip().lower()
        liveness = runtime_instance_liveness(item, capability=item.get('capability'))
        if liveness.get('hard_unavailable'):
            reason = readiness or str(item.get('status') or '').strip().lower() or 'unavailable'
            message = f"{item.get('instance_id')} is unavailable ({reason})"
            last_error = str(item.get('last_error') or '').strip()
            if last_error:
                message += f': {last_error}'
            issues.append(message)
        elif liveness.get('fresh_cooldown'):
            cooldown_until = str(liveness.get('cooldown_until') or '').strip()
            message = f"{item.get('instance_id')} is cooling down"
            if cooldown_until:
                message += f" until {cooldown_until}"
            issues.append(message)

    chat_default = capabilities.get(CAPABILITY_CHAT, {}).get('default_instance_id')
    if not chat_default:
        issues.append('No default chat instance is currently available.')
    return issues


def _build_recovery_hints(instances: list[dict[str, Any]], issues: list[str]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = [
        {
            'when': 'control plane is unreachable',
            'reason': 'restore the Ollmo Flask control plane before trying to route model traffic',
            'commands': [
                './ollmo start',
                'python3 scripts/ollmoctl.py doctor runtime --json',
            ],
        },
        {
            'when': '5001 is up but no instances are running',
            'reason': 'the control plane is alive but the runtime is empty',
            'commands': [
                'python3 scripts/ollmoctl.py models list --json',
                'python3 scripts/ollmoctl.py start --model <model> --backend <ollama|mlx> --capability <chat|vision_analysis|image_generation|speech_to_text|text_to_speech> --json',
            ],
        },
    ]

    has_actionable_liveness_issue = False
    for item in instances:
        liveness = runtime_instance_liveness(item, capability=item.get('capability'))
        if liveness.get('hard_unavailable') or liveness.get('fresh_cooldown'):
            has_actionable_liveness_issue = True
            break

    if has_actionable_liveness_issue:
        hints.append(
            {
                'when': 'a running instance is unavailable or cooling down',
                'reason': 'inspect live process, port, and backend truth before stopping, restarting, or choosing a different capability default',
                'commands': [
                    'python3 scripts/ollmoctl.py instances list --json',
                    'python3 scripts/ollmoctl.py doctor runtime --json',
                    'python3 scripts/ollmoctl.py stop <instance_id> --json',
                ],
            }
        )
    if not issues:
        hints.append(
            {
                'when': 'normal operation',
                'reason': 'use the canonical control-plane path rather than raw ports',
                'commands': [
                    'python3 scripts/ollmoctl.py send <instance_id> "<prompt>" --json',
                    'python3 scripts/ollmoctl.py history infer --limit 20 --json',
                ],
            }
        )
    return hints


def _normalize_recent_events(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get('action')
        status = item.get('status')
        readiness = str(item.get('readiness') or '').strip().lower()
        severity = item.get('severity')
        runtime_truth_note = item.get('runtime_truth_note')
        if action == 'status_transition' and readiness == 'degraded' and status == 'failed':
            status = 'warning'
            severity = severity or 'advisory'
            runtime_truth_note = runtime_truth_note or 'legacy_degraded_failed_event_reclassified_as_advisory'
        payload = {
            'timestamp': item.get('timestamp'),
            'category': item.get('category'),
            'action': action,
            'status': status,
            'instance_id': item.get('instance_id'),
            'message': item.get('message'),
        }
        for key in ('model', 'backend', 'capability', 'route_source', 'readiness'):
            value = item.get(key)
            if value not in (None, ''):
                payload[key] = value
        if severity:
            payload['severity'] = severity
        if runtime_truth_note:
            payload['runtime_truth_note'] = runtime_truth_note
        for key in ('previous_instance_id', 'previous_model', 'new_instance_id', 'new_model'):
            value = str(item.get(key) or '').strip()
            if value:
                payload[key] = value
        normalized.append(payload)
        if len(normalized) >= 8:
            break
    return normalized


def _ghost_display_target(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ''
    model = str(item.get('model') or '').strip()
    backend = str(item.get('backend') or '').strip()
    instance_id = str(item.get('instance_id') or '').strip()
    if model and backend:
        return f'{model} [{backend}]'
    if model:
        return model
    return instance_id


def _read_recent_log_lines(
    log_path: Path | str | None,
    *,
    limit: int = 200,
) -> list[str]:
    target = Path(log_path) if log_path else DEFAULT_RUNTIME_LOG_PATH
    if limit <= 0 or not target.exists():
        return []
    try:
        lines = target.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return []
    return [str(line or '').strip() for line in lines[-limit:] if str(line or '').strip()]


def _build_self_learning_payload(response_frame_ledger: Path) -> dict[str, Any]:
    try:
        return build_self_learning_report(
            response_frame_ledger_path=response_frame_ledger,
            frame_limit=120,
            max_cases=24,
            include_cases=False,
        )
    except Exception as exc:
        return {
            'kind': 'ollmo.self_learning_report',
            'status': 'unavailable',
            'mode': 'offline_eval_cases',
            'optimization_policy': 'proposal_only_reviewed_patch_required',
            'error': str(exc),
        }


def _deferred_self_learning_payload(response_frame_ledger: Path) -> dict[str, Any]:
    return {
        'kind': 'ollmo.self_learning_report',
        'status': 'omitted',
        'mode': 'offline_eval_cases',
        'optimization_policy': 'proposal_only_reviewed_patch_required',
        'runtime_effect': 'none',
        'reason': 'offline_self_learning_report_omitted_for_hot_routing_path',
        'source': {
            'response_frame_ledger': str(response_frame_ledger),
            'load_policy': 'diagnostic_endpoint_only',
        },
    }


def _load_accepted_learning_policy(path: Path) -> dict[str, Any]:
    try:
        return load_accepted_learning_policy_snapshot(snapshot_path=path)
    except Exception as exc:
        return {
            'kind': 'ollmo.accepted_learning_policy_snapshot',
            'status': 'unavailable',
            'enabled': False,
            'activation_policy': 'disabled_until_explicit_review',
            'runtime_effect': 'none',
            'error': str(exc),
        }


def render_ghost_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        '# Ollmo Ghost',
        '',
        'Ollmo is running as self-describing local runtime intelligence.',
        '',
        'It should understand and explain the local runtime, but leave orchestration to external clients.',
        '',
        '## Summary',
        '',
    ]

    counts = payload.get('counts') if isinstance(payload.get('counts'), dict) else {}
    lines.extend(
        [
            f"- running instances: {counts.get('instances', 0)}",
            f"- ready instances: {counts.get('ready', 0)}",
        ]
    )

    recommendations = payload.get('recommendations') if isinstance(payload.get('recommendations'), list) else []
    if recommendations:
        lines.extend(['', '## Recommended Defaults', ''])
        for item in recommendations:
            target = _ghost_display_target(item)
            lines.append(
                f"- {item.get('label')}: `{target}` ({item.get('reason')})"
            )

    issues = payload.get('issues') if isinstance(payload.get('issues'), list) else []
    lines.extend(['', '## Current Issues', ''])
    if issues:
        for item in issues:
            lines.append(f'- {item}')
    else:
        lines.append('- no active runtime issues detected')

    lines.extend(['', '## Recovery Hints', ''])
    recovery = payload.get('recovery') if isinstance(payload.get('recovery'), list) else []
    for item in recovery:
        when = str(item.get('when') or '').strip()
        reason = str(item.get('reason') or '').strip()
        commands = item.get('commands') if isinstance(item.get('commands'), list) else []
        lines.append(f'- {when}: {reason}')
        for command in commands:
            lines.append(f'  `{command}`')

    lines.extend(['', '## Paths', ''])
    paths = payload.get('paths') if isinstance(payload.get('paths'), dict) else {}
    for key in ('guide', 'artifacts', 'state', 'logs'):
        value = paths.get(key)
        if value:
            lines.append(f'- {key}: `{value}`')

    events = payload.get('recent_events') if isinstance(payload.get('recent_events'), list) else []
    if events:
        lines.extend(['', '## Recent Signals', ''])
        for item in events[:5]:
            prefix = f"[{item.get('timestamp')}] {item.get('category')}/{item.get('action')} {item.get('status')}"
            target = _ghost_display_target(item)
            if target:
                prefix += f" on {target}"
            message = str(item.get('message') or '').strip()
            if message:
                prefix += f": {message}"
            lines.append(f'- {prefix}')

    self_healing_hints = payload.get('self_healing_hints') if isinstance(payload.get('self_healing_hints'), list) else []
    if self_healing_hints:
        lines.extend(['', '## Self Healing Hints', ''])
        for item in self_healing_hints[:5]:
            reason = str(item.get('reason') or '').strip()
            if reason:
                lines.append(f'- {reason}')

    self_learning = payload.get('self_learning') if isinstance(payload.get('self_learning'), dict) else {}
    if self_learning:
        lines.extend(['', '## Offline Self Learning', ''])
        status = str(self_learning.get('status') or 'unknown').strip()
        mode = str(self_learning.get('mode') or 'offline_eval_cases').strip()
        policy = str(self_learning.get('optimization_policy') or '').strip()
        lines.append(f'- status: `{status}` ({mode})')
        if policy:
            lines.append(f'- policy: `{policy}`')
        lines.append(
            f"- eval cases: {self_learning.get('case_count', 0)} from "
            f"{self_learning.get('frame_count', 0)} frozen frame(s)"
        )
        candidates = (
            self_learning.get('improvement_candidates')
            if isinstance(self_learning.get('improvement_candidates'), list)
            else []
        )
        for item in candidates[:5]:
            target_area = str(item.get('target_area') or '').strip()
            summary = str(item.get('summary') or '').strip()
            if target_area and summary:
                lines.append(f'- `{target_area}`: {summary}')
        accepted_policy = (
            payload.get('accepted_learning_policy')
            if isinstance(payload.get('accepted_learning_policy'), dict)
            else {}
        )
        if accepted_policy:
            accepted_status = str(accepted_policy.get('status') or 'unknown').strip()
            accepted_enabled = 'enabled' if accepted_policy.get('enabled') is True else 'disabled'
            runtime_effect = str(accepted_policy.get('runtime_effect') or 'none').strip()
            lines.append(
                f'- accepted policy snapshot: `{accepted_status}` / `{accepted_enabled}` '
                f'(runtime effect: `{runtime_effect}`)'
            )
        accepted_hints = (
            payload.get('accepted_learning_hints')
            if isinstance(payload.get('accepted_learning_hints'), dict)
            else {}
        )
        if accepted_hints:
            lines.append(
                f"- accepted runtime hints: {accepted_hints.get('hint_count', 0)} "
                f"(effect: `{accepted_hints.get('runtime_effect') or 'none'}`)"
            )

    ghost_modes = payload.get('ghost_modes') if isinstance(payload.get('ghost_modes'), dict) else {}
    supported_modes = ghost_modes.get('supported') if isinstance(ghost_modes.get('supported'), list) else []
    if supported_modes:
        lines.extend(['', '## Ghost Mode Compatibility Aliases', ''])
        lines.append('- API-edge aliases only; runtime thinking uses file-backed semantic roles.')
        for item in supported_modes:
            mode = str(item.get('mode') or '').strip()
            summary = str(item.get('summary') or '').strip()
            autonomy = str(item.get('autonomy') or '').strip()
            prefix = f'- `{mode}`'
            if autonomy:
                prefix += f' [{autonomy}]'
            if summary:
                prefix += f': {summary}'
            lines.append(prefix)
    semantic_roles = payload.get('semantic_roles') if isinstance(payload.get('semantic_roles'), dict) else {}
    role_items = semantic_roles.get('roles') if isinstance(semantic_roles.get('roles'), list) else []
    if role_items:
        lines.extend(['', '## Semantic Roles', ''])
        lines.append('- File truth: every valid `ollmo_g/semantic_roles/*.md` role is advisory read-model orientation.')
        for item in role_items:
            role_id = str(item.get('role_id') or '').strip()
            summary = str(item.get('summary') or '').strip()
            if not role_id:
                continue
            line = f'- `{role_id}`'
            if summary:
                line += f': {summary}'
            lines.append(line)
    self_modification = ghost_modes.get('self_modification') if isinstance(ghost_modes.get('self_modification'), dict) else {}
    if self_modification:
        lines.extend(['', '## Self Modification Boundary', ''])
        summary = str(self_modification.get('summary') or '').strip()
        if summary:
            lines.append(f'- {summary}')
        allowed_surfaces = self_modification.get('allowed_surfaces') if isinstance(self_modification.get('allowed_surfaces'), list) else []
        gated_surfaces = self_modification.get('gated_surfaces') if isinstance(self_modification.get('gated_surfaces'), list) else []
        if allowed_surfaces:
            lines.append(f"- allowed: `{', '.join(str(item) for item in allowed_surfaces)}`")
        if gated_surfaces:
            lines.append(f"- gated: `{', '.join(str(item) for item in gated_surfaces)}`")

    return '\n'.join(lines).strip() + '\n'


def build_ghost_payload(
    instances: Iterable[dict[str, Any]],
    *,
    recent_events: Optional[Iterable[dict[str, Any]]] = None,
    base_url: Optional[str] = None,
    contract_path: Path | str | None = None,
    runtime_log_path: Path | str | None = None,
    response_frame_ledger_path: Path | str | None = None,
    accepted_learning_policy_path: Path | str | None = None,
    include_self_learning_report: bool = True,
) -> dict[str, Any]:
    normalized_instances = _normalize_instances(instances)
    guide_path = Path(contract_path) if contract_path else DEFAULT_GHOST_GUIDE_PATH
    response_frame_ledger = Path(response_frame_ledger_path) if response_frame_ledger_path else DEFAULT_RESPONSE_FRAME_LEDGER_PATH
    accepted_learning_policy_path = (
        Path(accepted_learning_policy_path)
        if accepted_learning_policy_path
        else DEFAULT_ACCEPTED_LEARNING_POLICY_PATH
    )
    counts = _summarize_counts(normalized_instances)
    capabilities = _build_capability_map(normalized_instances)
    recommendations = _build_recommendations(capabilities)
    issues = _build_issues(normalized_instances, capabilities)
    recovery = _build_recovery_hints(normalized_instances, issues)
    normalized_recent_events = _normalize_recent_events(recent_events or [])
    recent_log_lines = _read_recent_log_lines(runtime_log_path)
    self_observations = build_recent_self_observations(
        normalized_recent_events,
        runtime_issues=issues,
        log_lines=recent_log_lines,
    )
    self_healing_hints = build_self_healing_hints(self_observations)
    self_learning = (
        _build_self_learning_payload(response_frame_ledger)
        if include_self_learning_report
        else _deferred_self_learning_payload(response_frame_ledger)
    )
    accepted_learning_policy = _load_accepted_learning_policy(accepted_learning_policy_path)
    accepted_learning_hints = build_accepted_learning_runtime_hints(accepted_learning_policy)
    payload = {
        'identity': {
            'name': 'ollmo-ghost',
            'role': 'self-describing local runtime intelligence',
            'discovery_version': GHOST_DISCOVERY_VERSION,
        },
        'summary': (
            f"Ollmo sees {len(normalized_instances)} running instance(s) and "
            f"{sum(1 for item in normalized_instances if str(item.get('readiness') or '').strip().lower() == 'ready')} ready instance(s)."
        ),
        'counts': counts,
        'capabilities': capabilities,
        'recommendations': recommendations,
        'issues': issues,
        'recovery': recovery,
        'recent_events': normalized_recent_events,
        'self_healing_hints': self_healing_hints,
        'self_learning': self_learning,
        'accepted_learning_policy': accepted_learning_policy,
        'accepted_learning_hints': accepted_learning_hints,
        'paths': {
            'guide': str(guide_path),
            'artifacts': 'artifacts/',
            'state': 'state/',
            'logs': 'logs/',
            'response_frame_ledger': str(response_frame_ledger),
            'self_learning_report': 'state/self_learning/report.json',
            'accepted_learning_policy': str(accepted_learning_policy_path),
        },
        'service': {
            'base_url': str(base_url or '').rstrip('/') or None,
            'routes': {
                'ghost': '/api/ghost',
                'agent_contract': '/api/agent_contract',
                'instances': '/api/running_instances',
                'runtime_manifest': '/api/runtime_manifest',
                'responses': '/api/responses',
            },
        },
        'ghost_modes': {
            'default_mode': DEFAULT_LEGACY_MODE,
            'compatibility_only': True,
            'supported': build_legacy_mode_catalog(),
            'self_modification': build_self_modification_contract(),
        },
        'semantic_roles': {
            'kind': 'ollmo.semantic_role_catalog',
            'authority': 'advisory_read_model_only',
            'roles': build_semantic_role_catalog(),
        },
    }
    payload['markdown'] = render_ghost_markdown(payload)
    return payload
