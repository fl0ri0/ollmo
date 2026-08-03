"""Ghost route-runtime owners for Ollmo."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from ollmo_core.backend_fabric import build_backend_fabric_snapshot
from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    build_feature_contract,
    infer_capability,
    infer_supported_capabilities,
    normalize_backend,
    normalize_capability,
)
from ollmo_g import build_ghost_payload
from ollmo_g.control_hints import infer_tts_speaker_from_prompt
from ollmo_g.semantic_role_profile import build_semantic_role_profile
from ollmo_g.intent import (
    analyze_prompt_intent,
    prompt_has_self_contained_direct_tts_source,
)
from ollmo_g.request_meta import (
    apply_request_meta_to_route_context,
    compact_request_meta,
    effective_developer_flags,
    extract_request_meta,
)
from ollmo_g.request_phase_graph import (
    build_request_phase_graph,
    current_phase_capability,
    current_phase_is_graph_resolved,
    current_phase_reason,
    downstream_phase_capabilities,
    downstream_phase_records,
)
from ollmo_g.router import (
    build_embedding_hints_from_vectors,
    build_embedding_route_audit,
    build_embedding_route_candidates,
    build_failure_recovery_route,
    build_route_hint,
    build_route_context,
    build_route_memory_scope,
    maybe_apply_embedding_route_bias,
    select_embedding_instance,
    validate_route_decision,
)
from ollmo_services.chat_history import read_chat_history
from ollmo_services.context_scan import build_history_scan_context_candidates
from ollmo_services.events import read_events

_ROUTING_DYNAMIC_TRAIT_SKIP_KEYS = {
    'activity',
    'backend',
    'backend_contract',
    'backend_metadata',
    'backend_package',
    'backend_runtime',
    'capability',
    'canonical_responses',
    'direct_responses',
    'feature_sources',
    'features',
    'inputs',
    'instance_id',
    'last_error',
    'model',
    'modelname',
    'output_modality',
    'outputs',
    'path',
    'pid',
    'port',
    'provider_capabilities',
    'readiness',
    'request_model',
    'routing_summary',
    'runtime_status',
    'session_controls',
    'session_controls_summary',
    'supported_capabilities',
    'text_capable',
}

_DOWNSTREAM_CONTRACT_ALLOWED_CAPABILITIES = {
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
}
_SELECTED_REFERENCE_CONTEXT_PREFIX = 'Selected prior message reference for this conversation turn.'
_THREAD_CONTEXT_REFERENCE_RE = re.compile(
    r'\b('
    r'previous|prior|earlier|above|last|recent|before|again|continue|same|'
    r'as discussed|as mentioned|as above|use the previous|use the last|'
    r'vorher(?:ige[nrsm]?)?|vorhin|davor|oben|letzte|letzten|letzter|weiter|nochmal|noch einmal|'
    r'wie gesagt|wie besprochen|verwende das|benutze das|nutze das'
    r')\b',
    re.IGNORECASE,
)
_THREAD_CONTEXT_PRONOUN_ACTION_RE = re.compile(
    r'\b('
    r'use|reuse|edit|change|revise|continue|redo|read|speak|show|turn|convert|'
    r'summarize|translate|explain|compare|fix|make'
    r')\s+(?:it|that|this|them|those|the previous|the last)\b|'
    r'\b('
    r'verwende|benutze|nutze|ändere|aendere|bearbeite|mach|mache|lies|zeige|'
    r'übersetze|uebersetze|erkläre|erklaere|vergleiche|korrigiere'
    r')\b.{0,80}\b(?:das|dies|diese|diesen|dem|ihn|sie|es|vorherige|letzte)\b',
    re.IGNORECASE,
)
_DEEP_HISTORY_SCAN_RE = re.compile(
    r'\b('
    r'entire thread|whole thread|full thread|entire conversation|whole conversation|'
    r'conversation history|chat history|search history|scan history|find.*history|'
    r'somewhere earlier|somewhere above|all previous|all prior|older than recent|'
    r'ganze(?:n|r|s)? verlauf|gesamte(?:n|r|s)? verlauf|chatverlauf|'
    r'ganze(?:n|r|s)? history|gesamte(?:n|r|s)? history|'
    r'durchsuch.*(?:history|verlauf|chat)|such.*(?:history|verlauf|chat)|'
    r'irgendwo (?:oben|früher|frueher|vorher)'
    r')\b',
    re.IGNORECASE,
)
_MATERIALIZATION_OBJECT_TERM_RE = re.compile(
    r'\b(?:artifact|artifacts|artefakt|artefakte|output|outputs|image|images|bild|bilder)\b',
    re.IGNORECASE,
)
_MATERIALIZATION_AUDIT_TERM_RE = re.compile(
    r'\b(?:materiali[sz]ed|materialisiert|existier|exists?|erzeugt|generiert|generated|'
    r'saved|gespeichert|vorhanden|fulfilled|erfüllt|erfuellt|pruef|prüf|check|audit|erkannt)\b',
    re.IGNORECASE,
)


@dataclass
class GhostRouteRuntimeOwner:
    hooks: dict[str, Any]
    wrapper_capability_aliases: dict[str, list[str]]
    max_recent_messages: int

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def _planner_timeout_seconds_for_payload(
        self,
        payload: Any,
        *,
        semantic_role_profile: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        timeout_ms_to_seconds = self._hook('timeout_ms_to_seconds')

        explicit_timeout_sec = timeout_ms_to_seconds(
            effective_developer_flags(payload).get('planner_timeout_ms')
        )
        if explicit_timeout_sec is not None:
            return explicit_timeout_sec
        # Semantic roles are advisory-only. They must not change
        # planner budgets; use explicit developer_flags.planner_timeout_ms.
        return None

    def _effective_request_meta_payload(self, payload: Any) -> dict[str, Any]:
        return compact_request_meta(extract_request_meta(payload))

    def _build_developer_diagnostics_payload(
        self,
        payload: Any,
        *,
        planner_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        timeout_ms_to_seconds = self._hook('timeout_ms_to_seconds')

        developer_flags = effective_developer_flags(payload)
        planner_payload = planner_meta if isinstance(planner_meta, dict) else {}
        planner_timeout_ms = int(planner_payload.get('planner_timeout_ms') or 0) or developer_flags.get('planner_timeout_ms')
        embedding_signals_enabled = bool(developer_flags.get('embedding_signals_enabled', True))
        return {
            'routing_contract': 'ghost_primary',
            'routing_policy': 'ghost_first',
            'heuristic_role': 'shadow_guardrail',
            'embedding_signals_enabled': embedding_signals_enabled,
            'accepted_learning_authority': developer_flags.get('accepted_learning_authority') or 'soft_hint',
            'planner_timeout_ms': planner_timeout_ms,
            'planner_timeout_sec': timeout_ms_to_seconds(planner_timeout_ms),
        }

    def merge_request_meta_runtime_truth(
        self,
        route_runtime: Optional[dict[str, Any]],
        payload: Any,
        *,
        planner_meta: Optional[dict[str, Any]] = None,
        route_payload: Optional[dict[str, Any]] = None,
        response_payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        extract_responses_prompt = self._hook('extract_responses_prompt')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')

        updated = dict(route_runtime or {})
        request_meta = self._effective_request_meta_payload(payload)
        if request_meta:
            updated['request_meta'] = request_meta
        developer_diagnostics = self._build_developer_diagnostics_payload(
            payload,
            planner_meta=planner_meta,
        )
        existing_diagnostics = (
            updated.get('developer_diagnostics')
            if isinstance(updated.get('developer_diagnostics'), dict)
            else {}
        )
        if existing_diagnostics.get('planner_timeout_ms') is not None and developer_diagnostics.get('planner_timeout_ms') is None:
            developer_diagnostics['planner_timeout_ms'] = existing_diagnostics.get('planner_timeout_ms')
        if isinstance(existing_diagnostics.get('route_graph_consistency'), dict):
            developer_diagnostics['route_graph_consistency'] = dict(existing_diagnostics.get('route_graph_consistency') or {})
        updated['developer_diagnostics'] = developer_diagnostics
        normalized_payload = payload if isinstance(payload, dict) else dict(payload or {})
        prompt = extract_responses_prompt(normalized_payload)
        current_turn_prompt = extract_responses_current_turn_prompt(normalized_payload)
        request_phase_graph = build_request_phase_graph(
            prompt,
            intent_prompt=current_turn_prompt,
            request_payload=normalized_payload,
            route_payload=route_payload if isinstance(route_payload, dict) else {},
            response_payload=response_payload if isinstance(response_payload, dict) else {},
        )
        if request_phase_graph:
            updated['request_phase_graph'] = request_phase_graph
        return updated

    def _review_phase_graph_downstream_contract(
        self,
        *,
        prompt: str,
        current_turn_prompt: str,
        payload: Any,
        draft_phase_graph: Optional[dict[str, Any]],
        downstream: Any,
        branch_source: str,
        image_branch_limit: Optional[int] = None,
        empty_reason: str = 'downstream contract was missing',
    ) -> dict[str, Any]:
        review_result = {
            'accepted': False,
            'status': 'invalid_patch',
            'reason': empty_reason,
            'candidate_phase_graph': None,
            'downstream_branches': [],
        }
        if not isinstance(downstream, list) or not downstream:
            return review_result

        expanded_branches: list[dict[str, Any]] = []
        expected_capabilities: list[str] = []
        expected_branch_capabilities: list[str] = []
        phase_index = 2
        for contract_index, raw_item in enumerate(downstream, start=1):
            if not isinstance(raw_item, dict):
                review_result['reason'] = 'downstream contract entry was not an object'
                return review_result
            capability = normalize_capability(raw_item.get('capability'))
            if capability not in _DOWNSTREAM_CONTRACT_ALLOWED_CAPABILITIES:
                review_result['reason'] = f'downstream contract capability {capability or "unknown"} is not allowed'
                return review_result
            try:
                count = int(raw_item.get('count') or 0)
            except (TypeError, ValueError):
                review_result['reason'] = f'downstream contract count for {capability} was not numeric'
                return review_result
            if count < 1:
                review_result['reason'] = f'downstream contract count for {capability} must be positive'
                return review_result
            if capability == CAPABILITY_IMAGE_GENERATION and image_branch_limit is not None and count > image_branch_limit:
                review_result['reason'] = (
                    f'downstream contract image count must stay between 1 and {image_branch_limit}'
                )
                return review_result
            if capability not in expected_capabilities:
                expected_capabilities.append(capability)
            output_type = str(
                raw_item.get('output_type')
                or (
                    'audio'
                    if capability == CAPABILITY_TEXT_TO_SPEECH
                    else 'image'
                    if capability == CAPABILITY_IMAGE_GENERATION
                    else 'text'
                )
            ).strip().lower()
            branch_kind = str(
                raw_item.get('kind')
                or (
                    'materialize'
                    if capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}
                    else 'evidence'
                    if capability in {CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_VISION_ANALYSIS}
                    else 'postprocess'
                )
            ).strip().lower()
            branch_role = str(raw_item.get('role') or f'{capability}_follow_up').strip()
            raw_depends_on = raw_item.get('depends_on')
            depends_on = (
                [
                    str(item or '').strip()
                    for item in raw_depends_on
                    if str(item or '').strip()
                ]
                if isinstance(raw_depends_on, list)
                else ['phase-1']
            ) or ['phase-1']
            base_branch_id = str(raw_item.get('branch_id') or '').strip()
            for occurrence in range(1, count + 1):
                expected_branch_capabilities.append(capability)
                branch_id = (
                    base_branch_id
                    if base_branch_id and count == 1
                    else f'branch-{capability}-{contract_index}-{occurrence}'
                )
                branch = {
                    'branch_id': branch_id,
                    'phase_id': f'phase-{phase_index}',
                    'capability': capability,
                    'output_type': output_type,
                    'kind': branch_kind,
                    'role': branch_role,
                    'depends_on': list(depends_on),
                    'queue_index': occurrence,
                    'source': branch_source,
                }
                for key in (
                    'candidate_id',
                    'contract_state',
                    'contract_status',
                    'obligation_state',
                    'intent_state',
                    'promotion_policy',
                    'promotion_reason',
                    'promotion_source',
                    'promoted_from_candidate_id',
                    'artifact_prompt',
                    'artifact_prompt_source',
                    'content_payload',
                    'content_payload_source',
                    'phase_summary',
                    'stage_direction',
                    'candidate_selection_index',
                    'selection_policy',
                    'semantic_intent',
                    'objective',
                    'deliverable',
                    'rationale',
                    'review_criteria',
                    'input_refs',
                    'execution_contract',
                    'workload_task_ref',
                    'output_obligation_ref',
                    'output_contract',
                    'accepted_proposals',
                    'requires_artifact',
                    'text_artifact_extension',
                    'text_artifact_source_name',
                    'text_artifact_source',
                    'text_artifact_target_path',
                    'artifact_request',
                    'repair_action',
                    'recovery_action',
                    'repair_action_reason',
                    'blocked_by_dependency_input',
                    'blocked_by_branch_contract',
                ):
                    value = raw_item.get(key)
                    if value not in (None, '', [], {}):
                        branch[key] = value
                if isinstance(raw_item.get('batch_prompts'), list):
                    branch['batch_prompts'] = [
                        str(item or '').strip()
                        for item in raw_item.get('batch_prompts') or []
                        if str(item or '').strip()
                    ]
                if isinstance(raw_item.get('promotion'), dict):
                    branch['promotion'] = dict(raw_item.get('promotion') or {})
                expanded_branches.append(branch)
                phase_index += 1

        normalized_payload = payload if isinstance(payload, dict) else dict(payload or {})
        candidate_phase_graph = build_request_phase_graph(
            prompt,
            intent_prompt=current_turn_prompt,
            request_payload=normalized_payload,
            route_payload={'downstream_branches': expanded_branches},
        )
        review_result['candidate_phase_graph'] = candidate_phase_graph
        review_result['downstream_branches'] = expanded_branches

        if normalize_capability(current_phase_capability(candidate_phase_graph)) != CAPABILITY_CHAT:
            review_result['status'] = 'review_rejected'
            review_result['reason'] = 'reviewed candidate did not keep the current phase on Ghost chat'
            return review_result
        if not current_phase_is_graph_resolved(candidate_phase_graph):
            review_result['status'] = 'review_rejected'
            review_result['reason'] = 'reviewed candidate did not resolve into a prepare-first phase graph'
            return review_result
        downstream_branch_capabilities = [
            normalize_capability(item.get('capability'))
            for item in downstream_phase_records(candidate_phase_graph)
        ]
        if downstream_phase_capabilities(candidate_phase_graph) != expected_capabilities:
            review_result['status'] = 'review_rejected'
            review_result['reason'] = 'reviewed candidate downstream capability surface did not match the expected contract'
            return review_result
        if downstream_branch_capabilities != expected_branch_capabilities:
            review_result['status'] = 'review_rejected'
            review_result['reason'] = 'reviewed candidate downstream branch order did not match the expected contract'
            return review_result
        if len(candidate_phase_graph.get('downstream_branch_ids') or []) != len(expanded_branches):
            review_result['status'] = 'review_rejected'
            review_result['reason'] = 'reviewed candidate downstream branch count did not match the expected contract'
            return review_result
        if normalize_capability(current_phase_capability(draft_phase_graph)) != CAPABILITY_CHAT:
            review_result['status'] = 'review_rejected'
            review_result['reason'] = 'draft plan changed away from chat before downstream review completed'
            return review_result

        review_result['accepted'] = True
        review_result['status'] = 'accepted'
        review_result['reason'] = 'downstream contract accepted'
        return review_result

    def _derive_chat_route_downstream_contract(
        self,
        *,
        prompt_analysis: dict[str, Any],
        route_hint: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        downstream: list[dict[str, Any]] = []
        wants_audio = bool(
            prompt_analysis.get('requests_audio_output')
            or prompt_analysis.get('has_audio_follow_up_request')
            or prompt_analysis.get('text_preparation_before_audio_output')
        )
        wants_visual = bool(
            prompt_analysis.get('requests_visual_output')
            or prompt_analysis.get('has_visual_follow_up_request')
            or prompt_analysis.get('text_preparation_before_visual_output')
            or int(prompt_analysis.get('requested_visual_output_count') or 0) > 0
        )
        if bool(prompt_analysis.get('explicit_audio_defer_materialization')):
            wants_audio = False
        if bool(prompt_analysis.get('audio_output_count_exceeds_bound')):
            wants_audio = False
        if bool(prompt_analysis.get('explicit_visual_defer_materialization')):
            wants_visual = False
        if bool(prompt_analysis.get('explicit_defer_materialization')) and not (wants_audio or wants_visual):
            return []
        if wants_audio:
            try:
                requested_audio_count = int(
                    prompt_analysis.get('requested_audio_output_count') or 0
                )
            except (TypeError, ValueError):
                requested_audio_count = 0
            if requested_audio_count < 0 or requested_audio_count > 6:
                requested_audio_count = 0
            downstream.append(
                {
                    'capability': CAPABILITY_TEXT_TO_SPEECH,
                    'count': max(1, requested_audio_count),
                }
            )
        if wants_visual:
            downstream.append(
                {
                    'capability': CAPABILITY_IMAGE_GENERATION,
                    'count': max(1, int(prompt_analysis.get('requested_visual_output_count') or 0)),
                }
            )
        return downstream

    def _maybe_enforce_chat_route_graph_consistency(
        self,
        *,
        payload: Any,
        prompt: str,
        current_turn_prompt: str,
        prompt_analysis: dict[str, Any],
        draft_phase_graph: Optional[dict[str, Any]],
        route_hint: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        result = {
            'phase_graph': draft_phase_graph,
            'phase_graph_source': 'draft',
            'downstream_branches': [],
            'diagnostics': None,
        }
        diagnostics: dict[str, Any] = {
            'eligible': False,
            'final_graph_source': 'draft',
            'route_hint_capability': normalize_capability((route_hint or {}).get('capability')),
            'status': 'skipped',
        }
        if downstream_phase_capabilities(draft_phase_graph):
            diagnostics['reason'] = 'draft_already_has_downstream'
            result['diagnostics'] = diagnostics
            return result
        if normalize_capability(current_phase_capability(draft_phase_graph)) != CAPABILITY_CHAT:
            diagnostics['reason'] = 'draft_not_chat'
            result['diagnostics'] = diagnostics
            return result
        if bool(prompt_analysis.get('explicit_defer_materialization')) and not (
            bool(prompt_analysis.get('requests_audio_output'))
            or bool(prompt_analysis.get('has_audio_follow_up_request'))
            or bool(prompt_analysis.get('text_preparation_before_audio_output'))
            or bool(prompt_analysis.get('requests_visual_output'))
            or bool(prompt_analysis.get('has_visual_follow_up_request'))
            or bool(prompt_analysis.get('text_preparation_before_visual_output'))
            or int(prompt_analysis.get('requested_visual_output_count') or 0) > 0
        ):
            diagnostics['reason'] = 'explicit_defer_materialization'
            result['diagnostics'] = diagnostics
            return result
        if bool(prompt_analysis.get('meta_execution_explanation_request')):
            diagnostics['reason'] = 'meta_execution_explanation_request'
            result['diagnostics'] = diagnostics
            return result

        downstream = self._derive_chat_route_downstream_contract(
            prompt_analysis=prompt_analysis,
            route_hint=route_hint,
        )
        diagnostics['expected_capabilities'] = [
            normalize_capability(item.get('capability'))
            for item in downstream
            if isinstance(item, dict)
        ]
        if not downstream:
            diagnostics['reason'] = 'no_downstream_obligation_signals'
            result['diagnostics'] = diagnostics
            return result

        diagnostics['eligible'] = True
        review_result = self._review_phase_graph_downstream_contract(
            prompt=prompt,
            current_turn_prompt=current_turn_prompt,
            payload=payload,
            draft_phase_graph=draft_phase_graph,
            downstream=downstream,
            branch_source='ghost_route_graph_consistency_v1',
            empty_reason='route/graph consistency review found no downstream obligations to enforce',
        )
        diagnostics['review_status'] = str(review_result.get('status') or '').strip() or None
        diagnostics['review_reason'] = str(review_result.get('reason') or '').strip() or None
        if bool(review_result.get('accepted')):
            result['phase_graph'] = review_result.get('candidate_phase_graph')
            result['phase_graph_source'] = 'consistency_enforced'
            result['downstream_branches'] = list(review_result.get('downstream_branches') or [])
            diagnostics['status'] = 'accepted'
            diagnostics['final_graph_source'] = 'consistency_enforced'
        else:
            diagnostics['status'] = str(review_result.get('status') or 'review_rejected')
        result['diagnostics'] = diagnostics
        return result

    def _prompt_requests_structured_or_tooling(self, prompt: str) -> bool:
        return bool(
            re.search(
                r"\b(json|yaml|schema|structured|function call|function-calling|tool call|tools|valid json)\b",
                str(prompt or ''),
                re.IGNORECASE,
            )
        )

    def _route_context_prefers_multimodal_chat(self, route_context: dict[str, Any]) -> bool:
        attachment = (
            route_context.get('request_attachment')
            if isinstance(route_context.get('request_attachment'), dict)
            else {}
        )
        file_kind = str(attachment.get('file_kind') or '').strip().lower()
        if file_kind == 'image':
            return True
        prompt = str(route_context.get('prompt') or '')
        return bool(re.search(r"\b(image|picture|photo|screenshot|diagram|chart)\b", prompt, re.IGNORECASE))

    def _route_context_prefers_long_context(self, route_context: dict[str, Any]) -> bool:
        estimate_route_context_tokens = self._hook('estimate_route_context_tokens')

        prompt = str(route_context.get('prompt') or '')
        estimated_tokens = estimate_route_context_tokens(
            prompt=prompt,
            messages=route_context.get('recent_messages') if isinstance(route_context.get('recent_messages'), list) else [],
        )
        if estimated_tokens >= 3500:
            return True
        return bool(
            re.search(
                r"\b(long context|large context|huge context|transcript|document|logs?|conversation history|summari[sz]e|all of this|entire thread)\b",
                prompt,
                re.IGNORECASE,
            )
        )

    def _preview_can_carry_simple_chat(
        self,
        *,
        preview_mode: bool,
        prompt_analysis: dict[str, Any],
        selected_reference_artifacts: list[dict[str, Any]],
        upload_filename: str,
        file_path: str,
        phase_graph_current_capability: Optional[str],
        phase_graph_downstream_capabilities: list[str],
    ) -> bool:
        if normalize_capability(phase_graph_current_capability) != CAPABILITY_CHAT:
            return False
        if phase_graph_downstream_capabilities:
            return False
        if selected_reference_artifacts:
            return False
        if str(upload_filename or '').strip() or str(file_path or '').strip():
            return False
        if bool(prompt_analysis.get('requests_audio_output')) or bool(prompt_analysis.get('requests_visual_output')):
            return False
        primary_capability = normalize_capability(prompt_analysis.get('primary_capability'))
        if primary_capability and primary_capability != CAPABILITY_CHAT:
            return False
        return bool(preview_mode)

    def _can_resolve_simple_current_turn_chat(
        self,
        *,
        prompt_analysis: dict[str, Any],
        selected_reference_artifacts: list[dict[str, Any]],
        upload_filename: str,
        file_path: str,
        phase_graph_current_capability: Optional[str],
        phase_graph_downstream_capabilities: list[str],
    ) -> bool:
        return self._preview_can_carry_simple_chat(
            preview_mode=True,
            prompt_analysis=prompt_analysis,
            selected_reference_artifacts=selected_reference_artifacts,
            upload_filename=upload_filename,
            file_path=file_path,
            phase_graph_current_capability=phase_graph_current_capability,
            phase_graph_downstream_capabilities=phase_graph_downstream_capabilities,
        )

    def _direct_current_turn_route_reason(
        self,
        capability: str,
        *,
        route_context: dict[str, Any],
        helper_route: Optional[dict[str, Any]] = None,
    ) -> str:
        helper_reason = str((helper_route or {}).get('reason') or '').strip()
        if helper_reason and helper_reason.lower() != 'default chat fallback':
            return helper_reason

        request_attachment = (
            route_context.get('request_attachment')
            if isinstance(route_context.get('request_attachment'), dict)
            else {}
        )
        explicit_file_kind = str(request_attachment.get('file_kind') or '').strip().lower()
        if explicit_file_kind == 'audio':
            return 'current turn resolved directly from explicit audio input'
        if explicit_file_kind == 'pdf':
            return 'current turn resolved directly from explicit PDF input'
        if explicit_file_kind == 'image':
            if capability == CAPABILITY_IMAGE_GENERATION:
                return 'current turn resolved directly from explicit image input plus image-generation intent'
            return 'current turn resolved directly from explicit image input'
        if explicit_file_kind == 'text':
            if capability == CAPABILITY_TEXT_TO_SPEECH:
                return 'current turn resolved directly from explicit text input plus audio intent'
            return 'current turn resolved directly from explicit text input'

        if capability == CAPABILITY_IMAGE_GENERATION:
            return 'current turn resolved directly as image generation'
        if capability == CAPABILITY_TEXT_TO_SPEECH:
            return 'current turn resolved directly as text to speech'
        if capability == CAPABILITY_SPEECH_TO_TEXT:
            return 'current turn resolved directly as speech to text'
        if capability == CAPABILITY_VISION_ANALYSIS:
            return 'current turn resolved directly as vision analysis'
        return 'current turn resolved directly as chat'

    def _direct_current_turn_route_confidence(
        self,
        capability: str,
        *,
        route_context: dict[str, Any],
        helper_route: Optional[dict[str, Any]] = None,
    ) -> float:
        helper_confidence = float((helper_route or {}).get('confidence') or 0.0)
        request_attachment = (
            route_context.get('request_attachment')
            if isinstance(route_context.get('request_attachment'), dict)
            else {}
        )
        explicit_file_kind = str(request_attachment.get('file_kind') or '').strip().lower()
        if explicit_file_kind == 'audio':
            return max(helper_confidence, 0.98)
        if explicit_file_kind == 'pdf':
            return max(helper_confidence, 0.98)
        if explicit_file_kind == 'image':
            return max(helper_confidence, 0.97)
        if explicit_file_kind == 'text':
            return max(helper_confidence, 0.95)
        if capability in {
            CAPABILITY_IMAGE_GENERATION,
            CAPABILITY_TEXT_TO_SPEECH,
            CAPABILITY_SPEECH_TO_TEXT,
            CAPABILITY_VISION_ANALYSIS,
        }:
            return max(helper_confidence, 0.9)
        return max(helper_confidence, 0.82)

    def _current_turn_artifact_anchor_path(
        self,
        capability: str,
        *,
        route_context: dict[str, Any],
    ) -> Optional[str]:
        request_attachment = (
            route_context.get('request_attachment')
            if isinstance(route_context.get('request_attachment'), dict)
            else {}
        )
        if bool(request_attachment.get('has_explicit_file')):
            return None

        normalized_capability = normalize_capability(capability)
        target_types: set[str]
        if normalized_capability in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_VISION_ANALYSIS}:
            target_types = {'image'}
        elif normalized_capability == CAPABILITY_SPEECH_TO_TEXT:
            target_types = {'audio'}
        elif normalized_capability == CAPABILITY_TEXT_TO_SPEECH:
            target_types = {'text', 'markdown', 'md', 'json', 'csv', 'message'}
        else:
            return None

        candidates: list[dict[str, Any]] = []
        selected_reference_artifact = (
            route_context.get('selected_reference_artifact')
            if isinstance(route_context.get('selected_reference_artifact'), dict)
            else None
        )
        if selected_reference_artifact:
            candidates.append(selected_reference_artifact)
        selected_reference_artifacts = (
            route_context.get('selected_reference_artifacts')
            if isinstance(route_context.get('selected_reference_artifacts'), list)
            else []
        )
        for item in selected_reference_artifacts:
            if isinstance(item, dict):
                candidates.append(item)

        latest_artifacts = route_context.get('latest_artifacts') if isinstance(route_context.get('latest_artifacts'), dict) else {}
        for key in ('image', 'audio', 'text'):
            artifact = latest_artifacts.get(key)
            if isinstance(artifact, dict):
                candidates.append(artifact)

        recent_artifacts = route_context.get('recent_artifacts') if isinstance(route_context.get('recent_artifacts'), list) else []
        for item in recent_artifacts:
            if isinstance(item, dict):
                candidates.append(item)

        seen_paths: set[str] = set()
        for candidate in candidates:
            artifact_type = str(candidate.get('type') or '').strip().lower()
            path = str(candidate.get('path') or '').strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            if artifact_type in target_types:
                return path
        return None

    def _prompt_prefers_artifact_vision_analysis(self, prompt: str) -> bool:
        return bool(
            re.search(
                r"\b(describe this image|describe the image|describe what happened|what happened in this image|what is in this image|what's in this image|analy[sz]e this image|analy[sz]e the image|read the text in this image|ocr|caption this image|explain this image)\b",
                str(prompt or ''),
                re.IGNORECASE,
            )
        )

    def _route_context_prefers_richer_tts(self, route_context: dict[str, Any]) -> bool:
        intent = route_context.get('intent') if isinstance(route_context.get('intent'), dict) else {}
        language_codes = intent.get('language_codes') if isinstance(intent.get('language_codes'), list) else []
        voice_descriptors = intent.get('voice_descriptors') if isinstance(intent.get('voice_descriptors'), list) else []
        prompt = str(route_context.get('prompt') or '').strip()
        if language_codes or voice_descriptors:
            return True
        return bool(re.search(r'\b(custom voice|my voice|voice clone|clone voice|speaker)\b', prompt, re.IGNORECASE))

    def _tts_language_supported(self, instance: dict[str, Any], requested_codes: list[str]) -> bool:
        if not requested_codes:
            return False
        languages = {
            str(item or '').strip().lower()
            for item in (instance.get('tts_languages') or [])
            if str(item or '').strip()
        }
        if not languages:
            return False
        if 'auto' in languages:
            return True
        code_map = {
            'de': 'german',
            'en': 'english',
            'fr': 'french',
            'es': 'spanish',
            'it': 'italian',
            'pt': 'portuguese',
            'ja': 'japanese',
            'ko': 'korean',
            'ru': 'russian',
            'zh': 'chinese',
        }
        for code in requested_codes:
            token = str(code or '').strip().lower()
            if token in languages or code_map.get(token) in languages:
                return True
        return False

    def _pick_tts_trait_aware_instance(
        self,
        instances: list[dict[str, Any]],
        route_context: dict[str, Any],
    ) -> Optional[str]:
        candidates = [
            item for item in instances
            if isinstance(item, dict) and normalize_capability(item.get('capability')) == CAPABILITY_TEXT_TO_SPEECH
        ]
        if len(candidates) < 2 or not self._route_context_prefers_richer_tts(route_context):
            return None

        prompt = str(route_context.get('prompt') or '').strip()
        intent = route_context.get('intent') if isinstance(route_context.get('intent'), dict) else {}
        requested_languages = [
            str(item or '').strip().lower()
            for item in (intent.get('language_codes') or [])
            if str(item or '').strip()
        ]
        requested_voice_descriptors = [
            str(item or '').strip().lower()
            for item in (intent.get('voice_descriptors') or [])
            if str(item or '').strip()
        ]
        wants_named_speaker = False
        wants_custom_voice = bool(re.search(r'\b(custom voice|my voice|voice clone|clone voice|speaker)\b', prompt, re.IGNORECASE))
        wants_voice_design = bool(requested_voice_descriptors or requested_languages)

        scored: list[tuple[int, int, int, int, int, str]] = []
        for item in candidates:
            model_type = str(item.get('tts_model_type') or '').strip().lower()
            speakers = [str(raw or '').strip() for raw in (item.get('tts_speakers') or []) if str(raw or '').strip()]
            matched_speaker = infer_tts_speaker_from_prompt(prompt, speakers)
            if matched_speaker:
                wants_named_speaker = True
            language_rank = 2 if self._tts_language_supported(item, requested_languages) else 0
            if model_type == 'voice_design':
                tts_rank = 4
            elif model_type == 'custom_voice':
                tts_rank = 3
            elif model_type in {'base', ''}:
                tts_rank = 2
            elif model_type == 'kitten_tts':
                tts_rank = 1
            else:
                tts_rank = 0
            readiness = str(item.get('readiness') or '').strip().lower()
            activity = str(item.get('activity') or '').strip().lower()
            readiness_rank = 2 if readiness == 'ready' else 1 if readiness in {'started', 'idle'} else 0
            activity_rank = 1 if activity in {'idle', 'ready'} else 0
            scored.append(
                (
                    3 if matched_speaker else 0,
                    2 if wants_custom_voice and model_type == 'custom_voice' else 0,
                    1 if wants_voice_design else 0,
                    language_rank + tts_rank,
                    readiness_rank * 10 + activity_rank,
                    str(item.get('instance_id') or ''),
                )
            )

        if not scored:
            return None
        if wants_named_speaker and max(item[0] for item in scored) <= 0:
            return None
        best = sorted(scored, reverse=True)[0]
        if best[:4] == (0, 0, 0, 0):
            return None
        return best[5] or None

    def pick_trait_aware_instance(
        self,
        instances: list[dict[str, Any]],
        route_context: dict[str, Any],
    ) -> Optional[str]:
        if len(instances) < 2:
            return None

        tts_preference = self._pick_tts_trait_aware_instance(instances, route_context)
        if tts_preference:
            return tts_preference

        prompt = str(route_context.get('prompt') or '')
        prefers_multimodal = self._route_context_prefers_multimodal_chat(route_context)
        prefers_structured = self._prompt_requests_structured_or_tooling(prompt)
        prefers_long_context = self._route_context_prefers_long_context(route_context)
        if not any((prefers_multimodal, prefers_structured, prefers_long_context)):
            return None

        scored: list[tuple[int, int, int, int, int, int, str]] = []
        for item in instances:
            if not isinstance(item, dict):
                continue
            traits = self.build_instance_trait_summary(item)
            readiness = str(item.get('readiness') or '').strip().lower()
            activity = str(item.get('activity') or '').strip().lower()
            readiness_rank = 2 if readiness == 'ready' else 1 if readiness in {'started', 'idle'} else 0
            activity_rank = 1 if activity in {'idle', 'ready'} else 0
            scored.append(
                (
                    1 if prefers_multimodal and traits.get('supports_vision') else 0,
                    1 if prefers_structured and traits.get('supports_tools') else 0,
                    1 if prefers_structured and traits.get('supports_structured_outputs') else 0,
                    int(traits.get('active_context_window') or 0) if prefers_long_context else 0,
                    int(traits.get('declared_context_window') or 0) if prefers_long_context else 0,
                    readiness_rank * 10 + activity_rank,
                    str(item.get('instance_id') or ''),
                )
            )

        if not scored:
            return None
        best = sorted(scored, reverse=True)[0]
        if best[:5] == (0, 0, 0, 0, 0):
            return None
        return best[6] or None

    def _build_route_trait_reasons(
        self,
        instance: dict[str, Any],
        route_context: dict[str, Any],
    ) -> list[str]:
        prompt = str(route_context.get('prompt') or '')
        traits = self.build_instance_trait_summary(instance)
        reasons: list[str] = []
        if self._route_context_prefers_multimodal_chat(route_context) and traits.get('supports_vision'):
            reasons.append('supports image-aware chat')
        if self._prompt_requests_structured_or_tooling(prompt):
            if traits.get('supports_tools'):
                reasons.append('supports tool/function calling')
            elif traits.get('supports_structured_outputs'):
                reasons.append('supports structured outputs')
        if self._route_context_prefers_long_context(route_context):
            active_context = traits.get('active_context_window')
            declared_context = traits.get('declared_context_window')
            if active_context:
                reasons.append(f'active context budget {active_context} tokens')
            elif declared_context:
                reasons.append(f'declared context budget {declared_context} tokens')
        return reasons[:3]

    def augment_route_reason(
        self,
        base_reason: str,
        instance: dict[str, Any],
        route_context: dict[str, Any],
    ) -> str:
        cleaned_reason = str(base_reason or '').strip() or 'ghost route'
        trait_reasons = self._build_route_trait_reasons(instance, route_context)
        if not trait_reasons:
            return cleaned_reason
        suffix = '; '.join(trait_reasons)
        if suffix.lower() in cleaned_reason.lower():
            return cleaned_reason
        if cleaned_reason == 'ghost route':
            return f'{cleaned_reason}: {suffix}'
        return f'{cleaned_reason} [{suffix}]'

    def _build_compressed_history_message(
        self,
        messages: list[dict[str, Any]],
        *,
        keep_last: int = 4,
    ) -> Optional[dict[str, str]]:
        if len(messages) <= keep_last:
            return None
        older_messages = [
            item
            for item in messages[:-keep_last]
            if (
                isinstance(item, dict)
                and str(item.get('content') or '').strip()
                and not self._is_stale_selected_reference_context_message(item)
            )
        ]
        if not older_messages:
            return None
        sample = older_messages[-6:]
        summary_lines = []
        omitted = len(older_messages) - len(sample)
        if omitted > 0:
            summary_lines.append(f'- {omitted} earlier turns omitted for brevity.')
        for item in sample:
            role = str(item.get('role') or 'user').strip() or 'user'
            content = str(item.get('content') or '').strip()
            if len(content) > 220:
                content = content[:220].rstrip() + '...[truncated]'
            summary_lines.append(f'- {role}: {content}')
        if not summary_lines:
            return None
        return {
            'role': 'system',
            'content': 'Conversation summary prepared by Ollmo to fit the current context budget:\n' + '\n'.join(summary_lines),
        }

    def _is_stale_selected_reference_context_message(self, item: dict[str, Any]) -> bool:
        if not isinstance(item, dict):
            return False
        content = str(item.get('content') or '').strip()
        if not content.startswith(_SELECTED_REFERENCE_CONTEXT_PREFIX):
            return False
        return not bool(item.get('selected_reference'))

    def _prompt_needs_thread_context(self, prompt: str) -> bool:
        text = str(prompt or '').strip()
        if not text:
            return False
        if prompt_has_self_contained_direct_tts_source(text):
            return False
        return bool(
            _THREAD_CONTEXT_REFERENCE_RE.search(text)
            or _THREAD_CONTEXT_PRONOUN_ACTION_RE.search(text)
            or self._prompt_needs_materialization_readback(text)
        )

    def _prompt_needs_deep_history_scan(self, prompt: str) -> bool:
        text = str(prompt or '').strip()
        if not text:
            return False
        return bool(_DEEP_HISTORY_SCAN_RE.search(text))

    def _prompt_needs_materialization_readback(self, prompt: str) -> bool:
        text = str(prompt or '').strip()
        if not text:
            return False
        for line in text.split('\n'):
            if not _MATERIALIZATION_OBJECT_TERM_RE.search(line):
                continue
            if _MATERIALIZATION_AUDIT_TERM_RE.search(line):
                return True
        return False

    def _current_turn_only_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            return []
        latest_user_index: Optional[int] = None
        for index, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get('role') or '').strip().lower()
            content = str(item.get('content') or '').strip()
            if role == 'user' and content:
                latest_user_index = index
        if latest_user_index is None:
            return list(messages)

        kept: list[dict[str, Any]] = []
        kept_indexes: set[int] = set()
        for index, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get('role') or '').strip().lower()
            if self._is_stale_selected_reference_context_message(item):
                continue
            if role == 'system' or bool(item.get('selected_reference')) or index == latest_user_index:
                kept.append(item)
                kept_indexes.add(index)
        if latest_user_index not in kept_indexes:
            kept.append(messages[latest_user_index])
        return kept

    def _context_gate_text(self, value: Any, *, limit: int = 240) -> str:
        text = str(value or '').strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + '...'

    def _context_gate_token(self, value: Any) -> str:
        text = str(value or '').strip()
        if not text:
            return ''
        cleaned = ''.join(ch if ch.isalnum() else '-' for ch in text.lower())
        return '-'.join(part for part in cleaned.split('-') if part)[:96]

    def _context_gate_message_candidate_id(self, item: dict[str, Any], *, index: int) -> str:
        for key in ('message_id', 'messageId', 'id', 'source_message_id', 'sourceMessageId'):
            token = self._context_gate_token(item.get(key))
            if token:
                return f'context-message-{token}'
        return f'context-message-{index}'

    def _context_gate_message_summary(self, item: dict[str, Any]) -> str:
        content = item.get('content')
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get('text') or part.get('type') or '').strip())
                else:
                    parts.append(str(part or '').strip())
            content = ' '.join(part for part in parts if part)
        return self._context_gate_text(content)

    def _context_gate_artifact_evidence(self, item: dict[str, Any]) -> dict[str, Any]:
        refs: list[str] = []
        evidence: list[str] = []

        def append_ref(value: Any) -> None:
            ref = str(value or '').strip()
            if ref and ref not in refs:
                refs.append(ref)

        def append_evidence(value: str) -> None:
            text = self._context_gate_text(value, limit=220)
            if text and text not in evidence:
                evidence.append(text)

        def visit(value: Any, *, surface: str) -> None:
            if isinstance(value, list):
                for child in value:
                    visit(child, surface=surface)
                return
            if not isinstance(value, dict):
                return

            artifact_ref = str(value.get('artifact_ref') or value.get('ref') or '').strip()
            path = str(value.get('path') or value.get('saved_image_path') or value.get('saved_path') or '').strip()
            artifact_type = str(value.get('type') or value.get('kind') or value.get('output_type') or '').strip()
            status = str(value.get('status') or value.get('availability') or '').strip()
            slot_id = str(value.get('slot_id') or '').strip()
            response_id = str(value.get('response_id') or '').strip()
            if artifact_ref:
                append_ref(artifact_ref)
            if artifact_ref or path or status:
                parts = [surface]
                if artifact_type:
                    parts.append(f'type={artifact_type}')
                if slot_id:
                    parts.append(f'slot={slot_id}')
                if status:
                    parts.append(f'status={status}')
                if artifact_ref:
                    parts.append(f'ref={artifact_ref}')
                if path:
                    parts.append(f'path={path}')
                if response_id:
                    parts.append(f'response={response_id}')
                append_evidence(', '.join(parts))

            for key in ('artifact', 'artifacts', 'outputs', 'output_slots', 'output_branches', 'results'):
                if key in value:
                    visit(value.get(key), surface=key)

        if not isinstance(item, dict):
            return {}
        for key in ('artifacts', 'outputs', 'output_slots', 'output_branches', 'results'):
            if key in item:
                visit(item.get(key), surface=key)
        if str(item.get('saved_image_path') or '').strip():
            visit(
                {
                    'type': 'image',
                    'path': item.get('saved_image_path'),
                    'response_id': item.get('response_id'),
                    'availability': 'available',
                },
                surface='saved_image_path',
            )
        return {
            key: value
            for key, value in {
                'artifact_refs': refs[:8],
                'artifact_evidence': evidence[:10],
            }.items()
            if value
        }

    def _recent_durable_artifact_candidates(
        self,
        *,
        conversation_id: Any,
        prompt: str,
        history_dir: Any = None,
    ) -> list[dict[str, Any]]:
        if not self._prompt_needs_materialization_readback(prompt):
            return []
        history_id = str(conversation_id or '').strip()
        if not history_id:
            return []
        try:
            history = read_chat_history(history_id, history_dir=history_dir)
        except (OSError, ValueError, TypeError):
            return []
        messages = history.get('messages') if isinstance(history, dict) else []
        if not isinstance(messages, list):
            return []

        candidates: list[dict[str, Any]] = []
        for reverse_index, item in enumerate(reversed(messages[-24:]), start=1):
            if not isinstance(item, dict):
                continue
            role = str(item.get('role') or '').strip().lower()
            if role != 'assistant':
                continue
            artifact_context = self._context_gate_artifact_evidence(item)
            if not artifact_context:
                continue
            evidence = artifact_context.get('artifact_evidence') or []
            refs = artifact_context.get('artifact_refs') or []
            has_materialized_artifact = bool(refs) or any(
                'type=image' in str(entry)
                or 'type=audio' in str(entry)
                or 'type=document' in str(entry)
                or 'type=artifact' in str(entry)
                or 'saved_image_path' in str(entry)
                for entry in evidence
            )
            if not has_materialized_artifact:
                continue
            message_id = self._context_gate_token(
                item.get('message_id') or item.get('id') or item.get('response_id') or f'recent-{reverse_index}'
            )
            summary = (
                f'recent assistant turn has {len(refs)} artifact ref(s) and '
                f'{len(evidence)} durable output fact(s)'
            )
            response_id = str(item.get('response_id') or '').strip()
            if response_id:
                summary += f' for response {response_id}'
            candidate = {
                'candidate_id': f'durable-artifacts-{message_id}',
                'source_kind': 'artifact',
                'source_surface': 'chat_history',
                'status': 'promoted',
                'role': role,
                'summary': summary,
                'artifact_refs': refs,
                'artifact_evidence': evidence,
                'promotion_source': 'durable_readback',
                'promotion_target': 'active_context',
                'promotion_policy': 'requires_materialization_audit_turn',
                'promotion_reason': 'current turn asks whether durable artifacts or outputs materialized',
            }
            candidates.append({key: value for key, value in candidate.items() if value not in (None, '', [], {})})
            if len(candidates) >= 4:
                break
        return candidates

    def build_context_gate_state(
        self,
        *,
        messages: list[dict[str, Any]],
        strategy: dict[str, Any],
        prompt: str = '',
        conversation_id: Any = None,
        history_dir: Any = None,
        artifact_registry_ledger: Any = None,
    ) -> dict[str, Any]:
        """Build an auditable context candidate view for the chosen history gate."""

        mode = str(strategy.get('mode') or '').strip()
        if not messages or not mode:
            return {}
        deep_scan_needed = self._prompt_needs_deep_history_scan(prompt)
        latest_user_index: Optional[int] = None
        for index, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get('role') or '').strip().lower()
            content = str(item.get('content') or '').strip()
            if role == 'user' and content:
                latest_user_index = index

        candidates: list[dict[str, Any]] = []
        for index, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get('role') or '').strip().lower()
            if role == 'system' or index == latest_user_index:
                continue
            selected_reference = bool(item.get('selected_reference'))
            promoted_by_history = mode == 'recent_history'
            status = 'promoted' if selected_reference or promoted_by_history else 'not_promoted'
            candidate: dict[str, Any] = {
                'candidate_id': self._context_gate_message_candidate_id(item, index=index + 1),
                'source_kind': 'message',
                'status': status,
                'role': role or None,
                'summary': self._context_gate_message_summary(item),
                'promotion_source': 'context_strategy',
                'promotion_policy': 'requires_current_turn_relevance',
            }
            if selected_reference:
                candidate['promotion_target'] = 'active_reference'
                candidate['promotion_reason'] = 'explicit_selected_reference_message'
            elif promoted_by_history:
                candidate['promotion_target'] = 'active_context'
                candidate['promotion_reason'] = 'referential_turn_keeps_recent_history'
            artifact_context = self._context_gate_artifact_evidence(item)
            if artifact_context.get('artifact_refs'):
                candidate['artifact_refs'] = artifact_context['artifact_refs'][:6]
            if artifact_context.get('artifact_evidence'):
                candidate['artifact_evidence'] = artifact_context['artifact_evidence'][:6]
            candidates.append({key: value for key, value in candidate.items() if value not in (None, '', [], {})})
            if len(candidates) >= 12:
                break

        candidates.extend(
            self._recent_durable_artifact_candidates(
                conversation_id=conversation_id,
                prompt=prompt,
                history_dir=history_dir,
            )
        )

        scan_result: dict[str, Any] = {}
        scan_result_candidates: list[dict[str, Any]] = []
        if deep_scan_needed:
            scan_result = build_history_scan_context_candidates(
                prompt=prompt,
                history_dir=history_dir,
                artifact_registry_ledger=artifact_registry_ledger,
            )
            scan_result_candidates = [
                item
                for item in (scan_result.get('context_candidates') or [])
                if isinstance(item, dict)
            ]

        scan_candidate: dict[str, Any] = {
            'candidate_id': 'history-scan-deeper-pool',
            'source_kind': 'history_scan',
            'status': 'promoted' if deep_scan_needed else 'not_promoted',
            'summary': 'larger history, memory, and artifact pool scan',
            'scan_scope': 'history_memory_artifact_pool',
            'scan_status': 'needed' if deep_scan_needed else 'not_needed_for_this_turn',
            'scan_execution_status': str(scan_result.get('status') or '').strip() or None,
            'scan_result_count': scan_result.get('candidate_count') if scan_result else None,
            'scan_policy': 'use_existing_history_ledgers_and_artifact_refs',
            'scan_targets': ['chat_history', 'response_frame_ledger', 'artifact_registry'],
            'promotion_source': 'context_strategy',
            'promotion_policy': 'requires_current_turn_need_for_deeper_context',
        }
        if deep_scan_needed:
            scan_candidate['promotion_target'] = 'history_scan'
            scan_candidate['promotion_reason'] = 'current turn asks for broader history search'
        else:
            scan_candidate['reason'] = 'recent or selected context was sufficient, or old context was not needed'
        candidates.append(scan_candidate)
        candidates.extend(scan_result_candidates)

        if not candidates:
            return {}
        promoted_candidate_ids = [
            str(item.get('candidate_id') or '').strip()
            for item in candidates
            if str(item.get('status') or '').strip() == 'promoted'
        ]
        not_promoted_candidate_ids = [
            str(item.get('candidate_id') or '').strip()
            for item in candidates
            if str(item.get('status') or '').strip() == 'not_promoted'
        ]
        selected_reference_count = sum(
            1
            for item in candidates
            if str(item.get('promotion_reason') or '').strip() == 'explicit_selected_reference_message'
        )
        scan_summary: dict[str, Any] = {
            'decision': 'promoted' if deep_scan_needed else 'not_promoted',
            'executed': bool(deep_scan_needed),
            'reason': (
                'current turn asks for broader history search'
                if deep_scan_needed
                else 'deeper history scan was not needed for this turn'
            ),
            'scan_targets': ['chat_history', 'response_frame_ledger', 'artifact_registry'],
        }
        if scan_result:
            for key in (
                'status',
                'ranking_policy',
                'candidate_count',
                'matched_candidate_count',
                'promoted_candidate_count',
                'omitted_candidate_count',
                'matched',
                'scanned',
            ):
                if scan_result.get(key) not in (None, '', [], {}):
                    scan_summary[key] = scan_result.get(key)
        context_gate_review = {
            'kind': 'ollmo.context_gate_review',
            'status': 'checked',
            'intake_boundary': 'current_turn',
            'mode': mode,
            'strategy_reason': str(strategy.get('reason') or '').strip() or None,
            'history_scan': scan_summary,
            'recent_history_decision': 'promoted' if mode in {'recent_history', 'compressed_history'} else 'not_promoted',
            'selected_reference_count': selected_reference_count,
            'candidate_count': len(candidates),
            'promoted_candidate_count': len(promoted_candidate_ids),
            'not_promoted_candidate_count': len(not_promoted_candidate_ids),
            'promoted_candidate_ids': promoted_candidate_ids,
            'not_promoted_candidate_ids': not_promoted_candidate_ids,
        }
        return {
            'context_candidates': candidates,
            'candidate_count': len(candidates),
            'promoted_candidate_ids': promoted_candidate_ids,
            'context_gate_review': {
                key: value
                for key, value in context_gate_review.items()
                if value not in (None, '', [], {})
            },
        }

    def choose_context_strategy(
        self,
        *,
        instance: Optional[dict[str, Any]],
        messages: list[dict[str, Any]],
        prompt: str,
        has_file_context: bool,
        conversation_id: Any = None,
        history_dir: Any = None,
        artifact_registry_ledger: Any = None,
    ) -> dict[str, Any]:
        estimate_route_context_tokens = self._hook('estimate_route_context_tokens')

        estimated_tokens = estimate_route_context_tokens(prompt=prompt, messages=messages)
        traits = self.build_instance_trait_summary(instance or {}) if isinstance(instance, dict) else {}
        budget_tokens = int(traits.get('active_context_window') or traits.get('declared_context_window') or 0)
        if has_file_context:
            strategy = {
                'mode': 'bounded_file_context',
                'reason': 'request carries explicit file or artifact context',
                'estimated_tokens': estimated_tokens,
                'budget_tokens': budget_tokens or None,
                'applied': True,
            }
        else:
            should_compress = (
                len(messages) > 8
                or estimated_tokens > 6000
                or (budget_tokens and estimated_tokens > max(1500, int(budget_tokens * 0.45)))
            )
            if should_compress and len(messages) > 4:
                strategy = {
                    'mode': 'compressed_history',
                    'reason': 'recent chat history was compressed to fit the current context budget',
                    'estimated_tokens': estimated_tokens,
                    'budget_tokens': budget_tokens or None,
                    'applied': True,
                }
            elif not self._prompt_needs_thread_context(prompt):
                strategy = {
                    'mode': 'current_turn_only',
                    'reason': 'current turn is not referential; stale thread history is withheld from execution context',
                    'estimated_tokens': estimated_tokens,
                    'budget_tokens': budget_tokens or None,
                    'applied': True,
                }
            else:
                strategy = {
                    'mode': 'recent_history',
                    'reason': 'recent chat history fits the current context budget',
                    'estimated_tokens': estimated_tokens,
                    'budget_tokens': budget_tokens or None,
                    'applied': False,
                }
        gate_state = self.build_context_gate_state(
            messages=messages,
            strategy=strategy,
            prompt=prompt,
            conversation_id=conversation_id,
            history_dir=history_dir,
            artifact_registry_ledger=artifact_registry_ledger,
        )
        if gate_state:
            strategy = {**strategy, **gate_state}
        return strategy

    def _promoted_history_scan_context_message(self, strategy: dict[str, Any]) -> Optional[dict[str, str]]:
        candidates = strategy.get('context_candidates') if isinstance(strategy.get('context_candidates'), list) else []
        lines: list[str] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if str(item.get('status') or '').strip() != 'promoted':
                continue
            artifact_evidence = item.get('artifact_evidence') if isinstance(item.get('artifact_evidence'), list) else []
            has_artifact_context = bool(item.get('artifact_ref') or item.get('artifact_refs') or artifact_evidence)
            if str(item.get('promotion_source') or '').strip() != 'history_scan' and not has_artifact_context:
                continue
            source = str(item.get('source_surface') or item.get('source_kind') or 'history').strip()
            summary = self._context_gate_text(item.get('summary'), limit=260)
            if not summary:
                continue
            refs = item.get('artifact_refs') if isinstance(item.get('artifact_refs'), list) else []
            artifact_ref = str(item.get('artifact_ref') or '').strip()
            ref_text = ''
            if artifact_ref:
                ref_text = f' [artifact_ref: {artifact_ref}]'
            elif refs:
                compact_refs = ', '.join(str(ref) for ref in refs[:3] if str(ref).strip())
                if compact_refs:
                    ref_text = f' [artifact_refs: {compact_refs}]'
            evidence_text = ''
            if artifact_evidence:
                compact_evidence = '; '.join(
                    self._context_gate_text(item, limit=180)
                    for item in artifact_evidence[:3]
                    if str(item).strip()
                )
                if compact_evidence:
                    evidence_text = f' Evidence: {compact_evidence}.'
            lines.append(f'- {source}: {summary}{ref_text}.{evidence_text}')
            if len(lines) >= 8:
                break
        if not lines:
            return None
        return {
            'role': 'system',
            'content': (
                'Ollmo promoted these prior-context matches from existing history ledgers for this turn. '
                'Use them only where they answer the current request. Selected durable truth returns as bounded context, not hidden memory:\n'
                + '\n'.join(lines)
            ),
        }

    def _inject_promoted_history_scan_context(
        self,
        messages: list[dict[str, Any]],
        strategy: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(strategy, dict):
            return messages
        scan_message = self._promoted_history_scan_context_message(strategy)
        if not scan_message:
            return messages
        for item in messages:
            if (
                isinstance(item, dict)
                and str(item.get('role') or '').strip() == 'system'
                and str(item.get('content') or '').startswith('Ollmo promoted these prior-context matches')
            ):
                return messages
        insert_at = 0
        for index, item in enumerate(messages):
            if isinstance(item, dict) and str(item.get('role') or '').strip() == 'system':
                insert_at = index + 1
                continue
            break
        return [*messages[:insert_at], scan_message, *messages[insert_at:]]

    def apply_context_strategy(
        self,
        messages: list[dict[str, Any]],
        strategy: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        if not isinstance(strategy, dict):
            return messages
        mode = str(strategy.get('mode') or '').strip()
        if mode == 'current_turn_only':
            return self._inject_promoted_history_scan_context(
                self._current_turn_only_messages(messages),
                strategy,
            )
        if mode != 'compressed_history':
            return self._inject_promoted_history_scan_context(messages, strategy)
        summary_message = self._build_compressed_history_message(messages)
        if not summary_message:
            return self._inject_promoted_history_scan_context(messages, strategy)
        keep_last = 4
        preserved_system = [
            item
            for item in messages[:-keep_last]
            if isinstance(item, dict) and str(item.get('role') or '').strip() == 'system'
        ]
        recent_messages = messages[-keep_last:]
        return self._inject_promoted_history_scan_context(
            preserved_system + [summary_message] + recent_messages,
            strategy,
        )

    def _normalize_chat_content_for_backend(self, content: Any, backend: Optional[str]) -> Any:
        normalized_backend = normalize_backend(backend)
        if not isinstance(content, list):
            return content

        normalized_parts: list[Any] = []
        for raw_part in content:
            if not isinstance(raw_part, dict):
                normalized_parts.append(raw_part)
                continue

            part = dict(raw_part)
            part_type = str(part.get('type') or '').strip().lower()
            if normalized_backend == 'llama_cpp':
                if part_type in {'input_text', 'output_text'}:
                    normalized_parts.append({
                        'type': 'text',
                        'text': str(part.get('text') or ''),
                    })
                    continue
                if part_type == 'input_image':
                    image_value = part.get('image_url')
                    image_url = ''
                    if isinstance(image_value, dict):
                        image_url = str(image_value.get('url') or '').strip()
                    else:
                        image_url = str(image_value or '').strip()
                    if image_url:
                        normalized_parts.append({
                            'type': 'image_url',
                            'image_url': {'url': image_url},
                        })
                        continue
                if part_type == 'image_url' and isinstance(part.get('image_url'), str):
                    normalized_parts.append({
                        'type': 'image_url',
                        'image_url': {'url': str(part.get('image_url') or '').strip()},
                    })
                    continue
            normalized_parts.append(part)
        return normalized_parts

    def normalize_chat_messages_for_backend(
        self,
        messages: list[dict[str, Any]],
        *,
        backend: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        normalized: list[dict[str, Any]] = []
        for raw_item in messages:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            if 'content' in item:
                item['content'] = self._normalize_chat_content_for_backend(item.get('content'), backend)
            normalized.append(item)
        if len(normalized) < 2:
            return normalized
        system_messages = [
            item for item in normalized
            if str(item.get('role') or '').strip().lower() == 'system'
        ]
        if not system_messages:
            return normalized
        non_system_messages = [
            item for item in normalized
            if str(item.get('role') or '').strip().lower() != 'system'
        ]
        return system_messages + non_system_messages

    def _compact_string_list(self, value: Any, *, limit: int = 8) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for raw in value:
            token = str(raw or '').strip()
            if token and token not in items:
                items.append(token)
            if len(items) >= limit:
                break
        return items

    def _compact_dynamic_trait_value(self, value: Any) -> Optional[Any]:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            return token[:80]
        if isinstance(value, list):
            items = self._compact_string_list(value, limit=10)
            return items or None
        return None

    def _summarize_dynamic_model_traits_for_routing(self, entry: Any) -> dict[str, Any]:
        if not isinstance(entry, dict):
            return {}
        traits: dict[str, Any] = {}
        for raw_key, raw_value in entry.items():
            key = str(raw_key or '').strip()
            if not key:
                continue
            lowered = key.lower()
            if lowered in _ROUTING_DYNAMIC_TRAIT_SKIP_KEYS:
                continue
            compact_value = self._compact_dynamic_trait_value(raw_value)
            if compact_value is None:
                continue
            traits[key] = compact_value
            if len(traits) >= 14:
                break
        return traits

    def summarize_session_controls_for_routing(self, schema: Any) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        fields = schema.get('fields') if isinstance(schema.get('fields'), dict) else {}
        visible_fields: list[str] = []
        required_fields: list[str] = []
        labels: list[str] = []
        field_types: dict[str, str] = {}
        field_options: dict[str, list[str]] = {}
        for field_key, field in fields.items():
            if not isinstance(field, dict) or field.get('visible') is False:
                continue
            key = str(field_key or '').strip()
            label = str(field.get('label') or key).strip()
            if key:
                visible_fields.append(key)
            if label and label not in labels:
                labels.append(label)
            if field.get('required') and key:
                required_fields.append(key)
            field_type = str(field.get('type') or '').strip()
            if key and field_type:
                field_types[key] = field_type
            options = [
                str(item or '').strip()
                for item in (field.get('options') or [])
                if str(item or '').strip()
            ]
            if key and options:
                field_options[key] = options[:12]
        hint = str(schema.get('hint') or '').strip() or None
        return {
            'enabled': bool(schema.get('enabled')),
            'hint': hint,
            'visible_fields': visible_fields[:12],
            'required_fields': required_fields[:8],
            'labels': labels[:12],
            'field_types': field_types,
            'field_options': field_options,
        }

    def summarize_backend_metadata_for_routing(self, metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        return {
            'source': str(metadata.get('source') or '').strip() or None,
            'package_label': str(metadata.get('package_label') or '').strip() or None,
            'capabilities': self._compact_string_list(metadata.get('capabilities')),
            'instance_capabilities': self._compact_string_list(metadata.get('instance_capabilities')),
            'package_capabilities': self._compact_string_list(metadata.get('package_capabilities')),
            'runtime_constraints': self._compact_string_list(metadata.get('runtime_constraints')),
            'runtime_knobs': self._compact_string_list(metadata.get('runtime_knobs')),
            'native_endpoint_paths': self._compact_string_list(metadata.get('native_endpoint_paths')),
            'lazy_loads_model': bool(metadata.get('lazy_loads_model')) if 'lazy_loads_model' in metadata else None,
            'single_loaded_model': bool(metadata.get('single_loaded_model')) if 'single_loaded_model' in metadata else None,
            'supports_unload': bool(metadata.get('supports_unload')) if 'supports_unload' in metadata else None,
            'shim_kind': str(metadata.get('shim_kind') or '').strip() or None,
        }

    def summarize_backend_runtime_for_routing(self, runtime: Any) -> dict[str, Any]:
        if not isinstance(runtime, dict):
            return {}
        endpoint_urls = sorted(
            key
            for key, value in runtime.items()
            if key.endswith('_url') and value
        )
        return {
            'source': str(runtime.get('source') or '').strip() or None,
            'native_base_url': str(runtime.get('native_base_url') or '').strip() or None,
            'request_model_strategy': str(runtime.get('request_model_strategy') or '').strip() or None,
            'runtime_knobs': self._compact_string_list(runtime.get('runtime_knobs')),
            'lazy_loads_model': bool(runtime.get('lazy_loads_model')) if 'lazy_loads_model' in runtime else None,
            'single_loaded_model': bool(runtime.get('single_loaded_model')) if 'single_loaded_model' in runtime else None,
            'supports_unload': bool(runtime.get('supports_unload')) if 'supports_unload' in runtime else None,
            'shim_kind': str(runtime.get('shim_kind') or '').strip() or None,
            'endpoint_urls': endpoint_urls[:10],
        }

    def build_instance_trait_summary(self, instance: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(instance, dict):
            return {}
        inputs = {
            str(item or '').strip().lower()
            for item in (instance.get('inputs') or [])
            if str(item or '').strip()
        }
        provider_capabilities = {
            normalize_capability(item)
            for item in (instance.get('provider_capabilities') or [])
            if str(item or '').strip()
        }
        backend_metadata = (
            self.summarize_backend_metadata_for_routing(instance.get('backend_metadata'))
            if isinstance(instance.get('backend_metadata'), dict)
            else {}
        )
        runtime_status = instance.get('runtime_status') if isinstance(instance.get('runtime_status'), dict) else {}
        backend_runtime = self.summarize_backend_runtime_for_routing(
            runtime_status.get('backend_runtime') if isinstance(runtime_status.get('backend_runtime'), dict) else {}
        )
        feature_flags = instance.get('features') if isinstance(instance.get('features'), dict) else {}
        routing_summary = (
            instance.get('routing_summary')
            if isinstance(instance.get('routing_summary'), dict)
            else {}
        )
        session_controls = (
            routing_summary.get('session_controls')
            if isinstance(routing_summary.get('session_controls'), dict)
            else {}
        )
        visible_fields = {
            str(item or '').strip().lower()
            for item in (session_controls.get('visible_fields') or [])
            if str(item or '').strip()
        }
        dynamic_traits = (
            routing_summary.get('dynamic_model_traits')
            if isinstance(routing_summary.get('dynamic_model_traits'), dict)
            else {}
        )
        active_context = dynamic_traits.get('active_context_window') or runtime_status.get('context_length')
        declared_context = (
            dynamic_traits.get('context_length')
            or dynamic_traits.get('declared_context_window')
            or instance.get('context_length')
            or backend_metadata.get('context_length')
        )
        return {
            'backend_package': str(instance.get('backend_package') or '').strip() or None,
            'backend_contract': str(instance.get('backend_contract') or '').strip() or None,
            'supports_vision': bool(
                feature_flags.get('supports_vision')
                or feature_flags.get('supports_images')
                or 'image' in inputs
                or CAPABILITY_VISION_ANALYSIS in provider_capabilities
            ),
            'supports_tools': bool(feature_flags.get('supports_tools')),
            'supports_structured_outputs': bool(
                feature_flags.get('supports_structured_outputs')
                or feature_flags.get('supports_json_mode')
            ),
            'supports_audio_input': bool(feature_flags.get('supports_audio_input') or feature_flags.get('audio_input')),
            'supports_audio_output': bool(feature_flags.get('supports_audio_output') or feature_flags.get('audio_output')),
            'active_context_window': int(active_context) if isinstance(active_context, int) else 0,
            'declared_context_window': int(declared_context) if isinstance(declared_context, int) else 0,
            'request_model_strategy': str(backend_runtime.get('request_model_strategy') or '').strip() or None,
            'visible_session_controls': sorted(visible_fields),
            'dynamic_model_traits': dynamic_traits,
        }

    def build_instance_routing_summary(
        self,
        entry: dict[str, Any],
        runtime_status: dict[str, Any],
    ) -> dict[str, Any]:
        features = entry.get('features') if isinstance(entry.get('features'), dict) else {}
        backend_metadata = entry.get('backend_metadata') if isinstance(entry.get('backend_metadata'), dict) else {}
        backend_runtime = (
            runtime_status.get('backend_runtime')
            if isinstance(runtime_status.get('backend_runtime'), dict)
            else {}
        )
        return {
            'backend_package': str(entry.get('backend_package') or '').strip() or None,
            'backend_contract': str(entry.get('backend_contract') or '').strip() or None,
            'provider_capabilities': self._compact_string_list(entry.get('provider_capabilities')),
            'feature_flags': sorted(key for key, value in features.items() if value),
            'dynamic_model_traits': self._summarize_dynamic_model_traits_for_routing(entry),
            'session_controls': self.summarize_session_controls_for_routing(entry.get('session_controls')),
            'backend_metadata': self.summarize_backend_metadata_for_routing(backend_metadata),
            'backend_runtime': self.summarize_backend_runtime_for_routing(backend_runtime),
        }

    def build_routing_manifest_payload(
        self,
        instances: list[dict[str, Any]],
        *,
        backend_fabric: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        request_base_url = self._hook('request_base_url')
        build_instance_responses_path = self._hook('build_instance_responses_path')
        pick_default_capability_instance = self._hook('pick_default_capability_instance')
        external_targets_provider = self.hooks.get('external_targets')

        base_url = request_base_url()
        canonical_path = '/api/responses'
        canonical_v1_path = '/v1/responses'
        capability_groups: dict[str, list[dict[str, Any]]] = {}
        public_instances: list[dict[str, Any]] = []

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
            feature_contract = build_feature_contract(
                model_name,
                backend,
                capability,
                metadata=entry,
            )
            supported_capabilities = infer_supported_capabilities(
                model_name,
                backend,
                capability,
                metadata=entry,
            )
            text_capable = CAPABILITY_CHAT in supported_capabilities
            readiness = str(runtime_status.get('readiness') or entry.get('readiness') or '').strip() or None
            activity = str(runtime_status.get('activity') or entry.get('activity') or '').strip() or None
            last_error = str(runtime_status.get('last_error') or entry.get('last_error') or '').strip() or None
            backend_metadata = entry.get('backend_metadata') if isinstance(entry.get('backend_metadata'), dict) else {}
            backend_runtime = runtime_status.get('backend_runtime') if isinstance(runtime_status.get('backend_runtime'), dict) else {}
            routing_summary = self.build_instance_routing_summary(entry, runtime_status)
            direct_path = build_instance_responses_path(instance_id)
            instance_payload = {
                'instance_id': instance_id,
                'model': model_name,
                'backend': backend,
                'capability': capability,
                'supported_capabilities': supported_capabilities,
                'text_capable': text_capable,
                'port': entry.get('port'),
                'backend_package': routing_summary.get('backend_package'),
                'backend_contract': routing_summary.get('backend_contract'),
                'provider_capabilities': routing_summary.get('provider_capabilities'),
                'features': feature_contract.get('features') or {},
                'feature_sources': feature_contract.get('feature_sources') or {},
                'inputs': feature_contract.get('inputs') or [],
                'outputs': feature_contract.get('outputs') or [],
                'backend_metadata': backend_metadata,
                'backend_runtime': backend_runtime,
                'session_controls_summary': routing_summary.get('session_controls') or {},
                'dynamic_model_traits': routing_summary.get('dynamic_model_traits') or {},
                'tts_model_type': str(entry.get('tts_model_type') or '').strip() or None,
                'tts_languages': entry.get('tts_languages') if isinstance(entry.get('tts_languages'), list) else [],
                'tts_speakers': entry.get('tts_speakers') if isinstance(entry.get('tts_speakers'), list) else [],
                'routing_summary': routing_summary,
                'readiness': readiness,
                'activity': activity,
                'last_error': last_error,
                'canonical_responses': {
                    'method': 'POST',
                    'path': canonical_path,
                    'url': f'{base_url}{canonical_path}',
                    'requires_instance_id': True,
                },
                'direct_responses': {
                    'method': 'POST',
                    'path': direct_path,
                    'url': f'{base_url}{direct_path}',
                    'requires_instance_id': False,
                },
            }
            public_instances.append(instance_payload)
            grouped_candidate = {
                'instance_id': instance_id,
                'model': model_name,
                'backend': backend,
                'backend_package': routing_summary.get('backend_package'),
                'backend_contract': routing_summary.get('backend_contract'),
                'provider_capabilities': routing_summary.get('provider_capabilities'),
                'supported_capabilities': supported_capabilities,
                'text_capable': text_capable,
                'readiness': readiness,
                'activity': activity,
                'inputs': feature_contract.get('inputs') or [],
                'outputs': feature_contract.get('outputs') or [],
                'features': feature_contract.get('features') or {},
                'session_controls_summary': routing_summary.get('session_controls') or {},
                'dynamic_model_traits': routing_summary.get('dynamic_model_traits') or {},
                'tts_model_type': str(entry.get('tts_model_type') or '').strip() or None,
                'tts_languages': entry.get('tts_languages') if isinstance(entry.get('tts_languages'), list) else [],
                'tts_speakers': entry.get('tts_speakers') if isinstance(entry.get('tts_speakers'), list) else [],
                'routing_summary': routing_summary,
            }
            for supported_capability in supported_capabilities or [capability or 'unknown']:
                capability_groups.setdefault(supported_capability or 'unknown', []).append(grouped_candidate)

        capabilities_payload: dict[str, Any] = {}
        aliases_payload: dict[str, str] = {}
        for capability, aliases in self.wrapper_capability_aliases.items():
            candidates = capability_groups.get(capability, [])
            capabilities_payload[capability] = {
                'aliases': aliases,
                'default_instance_id': pick_default_capability_instance(candidates),
                'count': len(candidates),
                'candidates': candidates,
            }
            for alias in aliases:
                aliases_payload[alias] = capability

        for capability, candidates in capability_groups.items():
            if capability in capabilities_payload:
                continue
            capabilities_payload[capability] = {
                'aliases': [capability],
                'default_instance_id': pick_default_capability_instance(candidates),
                'count': len(candidates),
                'candidates': candidates,
            }
            aliases_payload.setdefault(capability, capability)

        external_targets = (
            external_targets_provider()
            if callable(external_targets_provider)
            else []
        )
        if not isinstance(external_targets, list):
            external_targets = []

        return {
            'service': {
                'name': 'ollmo',
                'discovery_version': 1,
                'canonical_responses': {
                    'method': 'POST',
                    'path': canonical_path,
                    'url': f'{base_url}{canonical_path}',
                    'requires_instance_id': True,
                },
                'canonical_v1_responses': {
                    'method': 'POST',
                    'path': canonical_v1_path,
                    'url': f'{base_url}{canonical_v1_path}',
                    'requires_instance_id': True,
                },
            },
            'aliases': aliases_payload,
            'capabilities': capabilities_payload,
            'instances': public_instances,
            'count': len(public_instances),
            'external_targets': [
                dict(item)
                for item in external_targets
                if isinstance(item, dict)
            ],
            'backend_fabric': backend_fabric or build_backend_fabric_snapshot(instances=instances),
        }

    def apply_selected_reference_artifact_to_route_context(
        self,
        route_context: dict[str, Any],
        selected_reference_artifact: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        extract_artifact_ref = self._hook('extract_artifact_ref')

        if not isinstance(route_context, dict) or not isinstance(selected_reference_artifact, dict):
            return route_context
        artifact_type = str(selected_reference_artifact.get('type') or '').strip()
        if artifact_type == 'message':
            content = str(selected_reference_artifact.get('content') or '').strip()
            if not content:
                return route_context
            message_role = str(selected_reference_artifact.get('message_role') or 'assistant').strip().lower() or 'assistant'
            if message_role not in {'user', 'assistant', 'system'}:
                message_role = 'assistant'
            updated = dict(route_context)
            recent_messages = list(updated.get('recent_messages') or [])
            recent_messages.append(
                {
                    'role': 'system',
                    'content': (
                        'Selected prior message reference for this conversation turn. '
                        'Treat it as bounded reference context only; the current user message remains the live instruction. '
                        'Do not infer new tasks from this reference unless the current turn explicitly asks.\n\n'
                        f'[{message_role}]\n{content}'
                    ),
                    'timestamp': str(selected_reference_artifact.get('timestamp') or '').strip() or None,
                    'response_model': str(selected_reference_artifact.get('response_model') or '').strip() or None,
                    'response_instance_id': str(selected_reference_artifact.get('response_instance_id') or '').strip() or None,
                }
            )
            updated['recent_messages'] = recent_messages[-14:]
            updated['selected_reference_artifact'] = dict(selected_reference_artifact)
            return updated
        artifact_path = str(selected_reference_artifact.get('path') or '').strip()
        artifact_ref = str(
            selected_reference_artifact.get('artifact_ref')
            or selected_reference_artifact.get('ref')
            or extract_artifact_ref(selected_reference_artifact)
            or ''
        ).strip() or None
        if not artifact_type or not artifact_path:
            return route_context
        updated = dict(route_context)
        recent_artifacts = [
            item for item in (updated.get('recent_artifacts') or [])
            if isinstance(item, dict)
            and str(item.get('path') or '').strip() != artifact_path
            and (
                not artifact_ref
                or str(item.get('artifact_ref') or item.get('ref') or extract_artifact_ref(item) or '').strip() != artifact_ref
            )
        ]
        recent_artifacts.insert(0, dict(selected_reference_artifact))
        updated['recent_artifacts'] = recent_artifacts
        latest_artifacts = dict(updated.get('latest_artifacts') or {})
        latest_artifacts[artifact_type] = dict(selected_reference_artifact)
        updated['latest_artifacts'] = latest_artifacts
        recent_messages = list(updated.get('recent_messages') or [])
        recent_messages.append(
            {
                'role': 'assistant',
                'content': 'Selected reference artifact.',
                'timestamp': None,
            }
        )
        updated['recent_messages'] = recent_messages[-self.max_recent_messages:]
        updated['selected_reference_artifact'] = dict(selected_reference_artifact)
        return updated

    def apply_selected_reference_artifacts_to_route_context(
        self,
        route_context: dict[str, Any],
        selected_reference_artifacts: Any,
    ) -> dict[str, Any]:
        sanitize_selected_reference_artifacts = self._hook('sanitize_selected_reference_artifacts')

        selected_references = sanitize_selected_reference_artifacts(selected_reference_artifacts)
        updated = dict(route_context or {})
        for selected_reference in selected_references:
            updated = self.apply_selected_reference_artifact_to_route_context(updated, selected_reference)
        if selected_references:
            updated['reference_artifacts'] = [dict(item) for item in selected_references]
            updated['selected_reference_artifacts'] = [dict(item) for item in selected_references]
            compatibility_reference = next(
                (item for item in selected_references if str(item.get('type') or '').strip().lower() != 'message'),
                selected_references[0],
            )
            updated['selected_reference_artifact'] = dict(compatibility_reference)
        return updated

    def _route_context_reference_artifacts(self, route_context: Any) -> list[dict[str, Any]]:
        if not isinstance(route_context, dict):
            return []
        payload = (
            route_context.get('reference_artifacts')
            if isinstance(route_context.get('reference_artifacts'), list)
            else (
                route_context.get('selected_reference_artifacts')
                if isinstance(route_context.get('selected_reference_artifacts'), list)
                else []
            )
        )
        return [dict(item) for item in payload if isinstance(item, dict)]

    def _find_route_artifact_ref(self, recent_artifacts: Any, artifact_path: Any) -> Optional[str]:
        extract_artifact_ref = self._hook('extract_artifact_ref')

        normalized_path = str(artifact_path or '').strip()
        if not normalized_path:
            return None
        for item in recent_artifacts or []:
            if not isinstance(item, dict):
                continue
            if str(item.get('path') or '').strip() != normalized_path:
                continue
            return str(item.get('artifact_ref') or item.get('ref') or extract_artifact_ref(item)).strip() or None
        return None

    def resolve_route_artifact_ref(
        self,
        route_context: Any,
        *,
        artifact_path: Any,
        preview_payload: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        if isinstance(preview_payload, dict):
            preview_route = preview_payload.get('route') if isinstance(preview_payload.get('route'), dict) else {}
            for source in (preview_payload, preview_route):
                for key in ('route_artifact_ref', 'routeArtifactRef', 'artifact_ref', 'artifactRef'):
                    token = str(source.get(key) or '').strip()
                    if token:
                        return token
        ref = self._find_route_artifact_ref(self._route_context_reference_artifacts(route_context), artifact_path)
        if ref:
            return ref
        recent_artifacts = (
            route_context.get('recent_artifacts')
            if isinstance(route_context, dict) and isinstance(route_context.get('recent_artifacts'), list)
            else []
        )
        return self._find_route_artifact_ref(recent_artifacts, artifact_path)

    def _candidate_matches_ghost_preference(
        self,
        candidate: dict[str, Any],
        preference: Optional[dict[str, Any]],
    ) -> bool:
        instance_supports_capability = self._hook('instance_supports_capability')

        if not isinstance(candidate, dict) or not isinstance(preference, dict):
            return False
        preferred_model = str(preference.get('model') or '').strip()
        preferred_backend = normalize_backend(preference.get('backend'))
        if preferred_model and str(candidate.get('model') or '').strip() != preferred_model:
            return False
        if preferred_backend and normalize_backend(candidate.get('backend')) != preferred_backend:
            return False
        preferred_capability = normalize_capability(preference.get('capability'))
        if preferred_capability and not instance_supports_capability(candidate, preferred_capability):
            return False
        return True

    def _ghost_execution_preference_applies_to_capability(
        self,
        target: Optional[dict[str, Any]],
        capability: Optional[str],
    ) -> bool:
        if not isinstance(target, dict):
            return False
        normalized_capability = normalize_capability(capability)
        if not normalized_capability:
            return False
        target_capability = normalize_capability(target.get('capability'))
        if target_capability:
            return target_capability == normalized_capability
        return normalized_capability == CAPABILITY_CHAT

    def pick_ghost_preference_instance(
        self,
        candidates: list[dict[str, Any]],
        route_context: dict[str, Any],
        *,
        route_selected_instance_id: Optional[str] = None,
        requested_capability: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
        preferences = runtime.get('ghost_preferences') if isinstance(runtime.get('ghost_preferences'), dict) else {}
        if not preferences:
            return None, None
        primary_mode = str(preferences.get('primary_mode') or 'auto').strip().lower()
        if primary_mode not in {'prefer', 'lock'}:
            return None, None

        primary_target = preferences.get('primary_target') if isinstance(preferences.get('primary_target'), dict) else None
        fallback_target = preferences.get('fallback_target') if isinstance(preferences.get('fallback_target'), dict) else None
        if not self._ghost_execution_preference_applies_to_capability(primary_target, requested_capability):
            primary_target = None
        if not self._ghost_execution_preference_applies_to_capability(fallback_target, requested_capability):
            fallback_target = None

        if primary_target:
            primary_match = next(
                (entry for entry in candidates if self._candidate_matches_ghost_preference(entry, primary_target)),
                None,
            )
            if primary_match:
                return str(primary_match.get('instance_id') or '').strip() or None, {
                    'mode': primary_mode,
                    'applied': 'primary_target',
                    'target': primary_target,
                }
        if route_selected_instance_id and primary_mode != 'lock':
            return None, None
        if fallback_target:
            fallback_match = next(
                (entry for entry in candidates if self._candidate_matches_ghost_preference(entry, fallback_target)),
                None,
            )
            if fallback_match:
                return str(fallback_match.get('instance_id') or '').strip() or None, {
                    'mode': primary_mode,
                    'applied': 'fallback_target',
                    'target': fallback_target,
                }
        return None, None

    def should_ignore_preview_route_for_live_route_hint(
        self,
        preview_route: Optional[dict[str, Any]],
        route_hint: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(preview_route, dict) or not isinstance(route_hint, dict):
            return False
        preview_capability = normalize_capability(preview_route.get('capability'))
        route_hint_capability = normalize_capability(route_hint.get('capability'))
        if preview_capability != CAPABILITY_CHAT:
            return False
        if not route_hint_capability or route_hint_capability == CAPABILITY_CHAT:
            return False
        try:
            route_hint_confidence = float(route_hint.get('confidence') or 0.0)
        except (TypeError, ValueError):
            route_hint_confidence = 0.0
        return route_hint_confidence >= 0.9

    def should_ignore_preview_route_for_live_heuristic(
        self,
        preview_route: Optional[dict[str, Any]],
        heuristic_route: Optional[dict[str, Any]],
    ) -> bool:
        return self.should_ignore_preview_route_for_live_route_hint(preview_route, heuristic_route)

    def _embedding_endpoint_url(self, instance: dict[str, Any], target_port: int, transport: str) -> str:
        routing_summary = instance.get('routing_summary') if isinstance(instance.get('routing_summary'), dict) else {}
        backend_metadata = routing_summary.get('backend_metadata') if isinstance(routing_summary.get('backend_metadata'), dict) else {}
        backend_runtime = routing_summary.get('backend_runtime') if isinstance(routing_summary, dict) else {}
        native_base_url = str(backend_runtime.get('native_base_url') or '').strip() or f'http://127.0.0.1:{target_port}'
        native_base_url = native_base_url.rstrip('/')

        if transport == 'ollama_api_embed':
            endpoint_path = '/api/embed'
        elif transport == 'openai_embeddings':
            native_paths = {
                str(item or '').strip()
                for item in (backend_metadata.get('native_endpoint_paths') or [])
                if str(item or '').strip()
            }
            endpoint_path = next(
                (
                    path
                    for path in ('/v1/embeddings', '/embeddings', '/api/embeddings')
                    if path in native_paths
                ),
                '/v1/embeddings',
            )
        else:
            raise ValueError(f"Embedding transport '{transport}' is not supported.")
        return f'{native_base_url}{endpoint_path}'

    def _parse_embedding_backend_response(self, payload: Any, transport: str) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise ValueError('Embedding response was not a JSON object.')
        embeddings = payload.get('embeddings')
        if isinstance(embeddings, list) and embeddings and all(isinstance(item, list) for item in embeddings):
            return embeddings
        single_embedding = payload.get('embedding')
        if isinstance(single_embedding, list):
            return [single_embedding]
        if transport == 'openai_embeddings':
            data = payload.get('data')
            if isinstance(data, list):
                ordered_rows = []
                for index, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue
                    embedding = item.get('embedding')
                    if not isinstance(embedding, list):
                        continue
                    try:
                        sort_index = int(item.get('index'))
                    except (TypeError, ValueError):
                        sort_index = index
                    ordered_rows.append((sort_index, embedding))
                if ordered_rows:
                    return [embedding for _index, embedding in sorted(ordered_rows, key=lambda row: row[0])]
        raise ValueError('Embedding response did not include embeddings.')

    def _execute_embedding_backend_request(
        self,
        *,
        target_port: int,
        model_name: str,
        backend: str,
        inputs: list[str],
        request_model_override: Optional[str] = None,
        embedding_transport: Optional[str] = None,
        instance: Optional[dict[str, Any]] = None,
    ) -> list[list[float]]:
        requests_post = self._hook('requests_post')
        chat_timeout_seconds = self._hook('chat_timeout_seconds')

        normalized_backend = normalize_backend(backend)
        transport = str(embedding_transport or '').strip() or 'ollama_api_embed'
        response = requests_post(
            self._embedding_endpoint_url(instance or {}, target_port, transport),
            json={
                'model': request_model_override or model_name,
                'input': inputs,
            },
            timeout=max(30, chat_timeout_seconds(model_name, normalized_backend, CAPABILITY_CHAT)),
        )
        response.raise_for_status()
        return self._parse_embedding_backend_response(response.json(), transport)

    def attach_embedding_hints_to_route_context(
        self,
        route_context: dict[str, Any],
        *,
        runtime_manifest: dict[str, Any],
        instances: list[dict[str, Any]],
    ) -> None:
        execute_embedding_backend_request = self._hook('execute_embedding_backend_request')

        runtime_payload = route_context.setdefault('runtime', {})
        ghost_preferences = runtime_payload.get('ghost_preferences') if isinstance(runtime_payload.get('ghost_preferences'), dict) else {}
        preferred_embedding_helper = (
            ghost_preferences.get('embedding_helper')
            if isinstance(ghost_preferences.get('embedding_helper'), dict)
            else None
        )
        prompt = str(route_context.get('prompt') or '').strip()
        if not prompt:
            runtime_payload['embedding_helper'] = {
                'available': False,
                'attached': False,
                'reason': 'empty_prompt',
            }
            return
        embedding_instance = select_embedding_instance(instances, preferred_target=preferred_embedding_helper)
        if not embedding_instance:
            runtime_payload['embedding_helper'] = {
                'available': False,
                'attached': False,
                'reason': 'no_supported_embedding_helper',
            }
            return
        candidates = build_embedding_route_candidates(
            runtime_manifest=runtime_manifest,
            instances=instances,
        )
        helper_status = {
            'available': True,
            'attached': False,
            'instance_id': str(embedding_instance.get('instance_id') or '').strip() or None,
            'model': str(embedding_instance.get('model') or '').strip() or None,
            'backend': normalize_backend(embedding_instance.get('backend')),
            'backend_package': str(embedding_instance.get('backend_package') or '').strip() or None,
            'backend_contract': str(embedding_instance.get('backend_contract') or '').strip() or None,
            'transport': str(embedding_instance.get('embedding_transport') or '').strip() or None,
            'candidate_count': len(candidates),
            'preference_target': preferred_embedding_helper,
            'preference_applied': bool(
                preferred_embedding_helper
                and self._candidate_matches_ghost_preference(embedding_instance, preferred_embedding_helper)
            ),
        }
        runtime_payload['embedding_helper'] = helper_status
        if not candidates:
            helper_status['reason'] = 'no_route_candidates'
            return
        try:
            helper_status['semantic_compute_performed'] = True
            vectors = execute_embedding_backend_request(
                target_port=int(embedding_instance['port']),
                model_name=str(embedding_instance.get('model') or ''),
                backend=normalize_backend(embedding_instance.get('backend')),
                inputs=[prompt, *[str(candidate.get('text') or '') for candidate in candidates]],
                request_model_override=str(embedding_instance.get('request_model') or '').strip() or None,
                embedding_transport=str(embedding_instance.get('embedding_transport') or '').strip() or None,
                instance=embedding_instance,
            )
        except Exception as exc:  # noqa: BLE001
            logging.info('Ghost embedding hint fallback: %s', exc)
            helper_status['reason'] = 'helper_execution_failed'
            helper_status['error'] = str(exc)
            return

        if len(vectors) != len(candidates) + 1:
            logging.info(
                'Ghost embedding hint fallback: expected %s vectors, received %s.',
                len(candidates) + 1,
                len(vectors),
            )
            helper_status['reason'] = 'helper_response_mismatch'
            helper_status['received_vectors'] = len(vectors)
            return

        hints = build_embedding_hints_from_vectors(vectors[0], candidates, vectors[1:])
        if not hints:
            helper_status['reason'] = 'no_embedding_hints'
            return

        runtime_payload['embedding_hints'] = {
            **hints,
            'embedding_instance_id': str(embedding_instance.get('instance_id') or '').strip() or None,
            'embedding_model': str(embedding_instance.get('model') or '').strip() or None,
        }
        helper_status['attached'] = True
        helper_status['reason'] = 'attached'

    def resolve_ghost_auto_route(
        self,
        data: Any,
        *,
        upload=None,
        excluded_instance_ids: Optional[list[str]] = None,
        retry_failure: Optional[dict[str, Any]] = None,
        preview_mode: bool = False,
        refresh_runtime_status: bool = False,
        compute_semantics: bool = False,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        normalize_request_payload = self._hook('normalize_request_payload')
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        load_running_instances = self._hook('load_running_instances')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        extract_responses_prompt = self._hook('extract_responses_prompt')
        extract_responses_current_turn_prompt = self._hook('extract_responses_current_turn_prompt')
        extract_selected_reference_artifacts = self._hook('extract_selected_reference_artifacts')
        extract_ghost_preferences = self._hook('extract_ghost_preferences')
        extract_ghost_route_messages = self._hook('extract_ghost_route_messages')
        request_base_url = self._hook('request_base_url')
        ghost_guide_path_getter = self._hook('ghost_guide_path_getter')
        flask_log_path_getter = self._hook('flask_log_path_getter')
        apply_ghost_preferences_to_route_context = self._hook('apply_ghost_preferences_to_route_context')
        extract_ghost_preview_route = self._hook('extract_ghost_preview_route')
        attach_embedding_hints_to_route_context = self._hook('attach_embedding_hints_to_route_context')
        instance_supports_capability = self._hook('instance_supports_capability')
        pick_prompt_preferred_instance = self._hook('pick_prompt_preferred_instance')
        pick_default_capability_instance = self._hook('pick_default_capability_instance')
        build_working_frame = self._hook('build_working_frame')
        event_log_path_getter = self._hook('event_log_path_getter')
        validate_external_target_request = self.hooks.get(
            'validate_external_target_request'
        )

        data = normalize_request_payload(data)
        developer_flags = effective_developer_flags(data)
        embedding_signals_enabled = bool(developer_flags.get('embedding_signals_enabled', True))
        semantic_compute_allowed = bool((not preview_mode) or compute_semantics)
        local_instances = merge_instances_with_runtime_status(
            load_running_instances(),
            path=runtime_status_path_getter(),
            refresh=refresh_runtime_status,
        )
        runtime_manifest = self.build_routing_manifest_payload(local_instances)
        external_targets = (
            runtime_manifest.get('external_targets')
            if isinstance(runtime_manifest.get('external_targets'), list)
            else []
        )
        external_target_request_allowed = True
        if callable(validate_external_target_request):
            validation = validate_external_target_request(
                data,
                upload_present=bool(upload and getattr(upload, 'filename', None)),
            )
            external_target_request_allowed = bool(
                validation[0] if isinstance(validation, tuple) else validation
            )
        instances = [
            *local_instances,
            *[
                dict(item)
                for item in external_targets
                if (
                    isinstance(item, dict)
                    and item.get('selectable') is True
                    and external_target_request_allowed
                )
            ],
        ]
        if not instances:
            return None, 'No running instances. Start a model before routing prompts through Ollmo.'

        normalized_payload = data if isinstance(data, dict) else dict(data)
        prompt = extract_responses_prompt(normalized_payload)
        current_turn_prompt = extract_responses_current_turn_prompt(normalized_payload) or prompt
        prompt_analysis = analyze_prompt_intent(current_turn_prompt)
        upload_filename = str(getattr(upload, 'filename', None) or data.get('upload_filename') or '').strip()
        file_path = str(data.get('file_path') or '').strip()
        selected_reference_artifacts = extract_selected_reference_artifacts(data)
        ghost_preferences = extract_ghost_preferences(data if isinstance(data, dict) else dict(data))
        conversation_id = str(data.get('conversation_id') or '').strip() or None
        ghost_messages = extract_ghost_route_messages(
            data,
            include_selected_reference=True,
        )
        recent_events = read_events(path=event_log_path_getter(), limit=64)
        ghost_payload = build_ghost_payload(
            local_instances,
            recent_events=recent_events,
            base_url=request_base_url(),
            contract_path=ghost_guide_path_getter(),
            runtime_log_path=flask_log_path_getter(),
            include_self_learning_report=False,
        )
        route_context = build_route_context(
            prompt=current_turn_prompt,
            upload_filename=upload_filename,
            file_path=file_path,
            conversation_id=conversation_id,
            messages=ghost_messages,
            runtime_manifest=runtime_manifest,
            ghost_payload=ghost_payload,
            instances=instances,
        )
        route_context = self.apply_selected_reference_artifacts_to_route_context(
            route_context,
            selected_reference_artifacts,
        )
        route_context = apply_ghost_preferences_to_route_context(route_context, ghost_preferences)
        request_meta = extract_request_meta(data)
        request_phase_graph = build_request_phase_graph(
            prompt,
            intent_prompt=current_turn_prompt,
            request_payload=data,
        )
        runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
        runtime['request_phase_graph'] = request_phase_graph
        route_context['runtime'] = runtime
        route_context = apply_request_meta_to_route_context(route_context, request_meta)
        normalized_retry_failure = retry_failure if isinstance(retry_failure, dict) else {}
        if normalized_retry_failure:
            runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
            runtime['retry_failure'] = {
                'failed_capability': normalize_capability(normalized_retry_failure.get('capability')),
                'failed_instance_id': str(normalized_retry_failure.get('failed_instance_id') or '').strip() or None,
                'status_code': int(normalized_retry_failure.get('status_code') or 0) or None,
                'error_message': str(normalized_retry_failure.get('error_message') or '').strip() or None,
            }
            route_context['runtime'] = runtime
        if not embedding_signals_enabled:
            runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
            runtime['embedding_helper'] = {
                'available': False,
                'attached': False,
                'reason': 'disabled_by_request_meta',
            }
            runtime.pop('embedding_hints', None)
            route_context['runtime'] = runtime
        elif preview_mode and not semantic_compute_allowed:
            runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
            runtime['embedding_helper'] = {
                'available': False,
                'attached': False,
                'reason': 'semantic_compute_not_requested',
            }
            runtime.pop('embedding_hints', None)
            route_context['runtime'] = runtime
        route_hint: Optional[dict[str, Any]] = None
        validated_route_hint: Optional[dict[str, Any]] = None
        route_hint_validation_error: Optional[str] = None
        route_hint_resolved = False

        def ensure_route_hint() -> None:
            nonlocal route_hint, validated_route_hint, route_hint_validation_error, route_hint_resolved
            if route_hint_resolved:
                return
            route_hint_resolved = True
            route_hint = build_route_hint(route_context)
            validated_route_hint, route_hint_validation_error = validate_route_decision(
                route_hint,
                instances=instances,
                recent_artifacts=route_context.get('recent_artifacts') or [],
            )

        routing_scope = build_route_memory_scope(route_context)
        runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
        runtime['routing_scope'] = routing_scope
        route_context['runtime'] = runtime
        routing_preferences = (
            routing_scope.get('routing_preferences')
            if isinstance(routing_scope.get('routing_preferences'), dict)
            else {}
        )
        route_prompt_class = str(routing_scope.get('prompt_class') or '').strip() or None
        route_session_class = str(routing_scope.get('session_class') or '').strip() or None
        semantic_role_profile = build_semantic_role_profile(
            route_context,
            request_meta=request_meta,
            preview_mode=preview_mode,
            retry_failure=normalized_retry_failure,
        )
        runtime = route_context.get('runtime') if isinstance(route_context.get('runtime'), dict) else {}
        runtime['semantic_role_profile'] = semantic_role_profile
        route_context['runtime'] = runtime
        ensure_route_hint()
        route_graph_consistency = self._maybe_enforce_chat_route_graph_consistency(
            payload=data,
            prompt=prompt,
            current_turn_prompt=current_turn_prompt,
            prompt_analysis=prompt_analysis,
            draft_phase_graph=request_phase_graph,
            route_hint=route_hint,
        )
        request_phase_graph = (
            route_graph_consistency.get('phase_graph')
            if isinstance(route_graph_consistency.get('phase_graph'), dict)
            else request_phase_graph
        )
        consistency_downstream_branches = (
            list(route_graph_consistency.get('downstream_branches') or [])
            if isinstance(route_graph_consistency.get('downstream_branches'), list)
            else []
        )
        route_graph_consistency_diagnostics = (
            route_graph_consistency.get('diagnostics')
            if isinstance(route_graph_consistency.get('diagnostics'), dict)
            else None
        )
        phase_graph_current_capability = current_phase_capability(request_phase_graph)
        phase_graph_downstream_capabilities = downstream_phase_capabilities(request_phase_graph)
        phase_graph_locked_current_phase = bool(
            phase_graph_current_capability == CAPABILITY_CHAT
            and phase_graph_downstream_capabilities
            and current_phase_is_graph_resolved(request_phase_graph)
        )
        phase_graph_current_reason = (
            current_phase_reason(request_phase_graph)
            or 'current phase was already resolved by the request phase graph'
        )
        phase_graph_simple_chat_preview = self._preview_can_carry_simple_chat(
            preview_mode=preview_mode,
            prompt_analysis=prompt_analysis,
            selected_reference_artifacts=selected_reference_artifacts,
            upload_filename=upload_filename,
            file_path=file_path,
            phase_graph_current_capability=phase_graph_current_capability,
            phase_graph_downstream_capabilities=phase_graph_downstream_capabilities,
        )
        if phase_graph_simple_chat_preview:
            phase_graph_current_reason = 'single-phase text chat was resolved locally for preview'

        route_payload: Optional[dict[str, Any]] = None
        route_source = 'unresolved'
        resolution_status: Optional[str] = None
        resolution_message: Optional[str] = None
        preview_payload = extract_ghost_preview_route(data)
        raw_preview_payload = (
            preview_payload
            if isinstance(preview_payload, dict)
            else data.get('ghost_preview')
            if isinstance(data, dict) and isinstance(data.get('ghost_preview'), dict)
            else None
        )
        preview_route_candidate: Optional[dict[str, Any]] = None
        preview_route_source: Optional[str] = None
        embedding_bias_applied = False
        embedding_signals_attached = False
        route_resolution_error: Optional[str] = None

        if raw_preview_payload:
            validated_preview_route, preview_error = validate_route_decision(
                raw_preview_payload,
                instances=instances,
                recent_artifacts=route_context.get('recent_artifacts') or [],
            )
            if validated_preview_route:
                preview_route_candidate = validated_preview_route
                preview_route_source = str(
                    raw_preview_payload.get('route_source') or raw_preview_payload.get('source') or 'preview'
                ).strip() or 'preview'
            elif preview_error:
                logging.info('Ghost preview route fallback: %s', preview_error)

        if not route_payload and normalized_retry_failure:
            recovery_preview = build_failure_recovery_route(
                route_context,
                failed_capability=normalized_retry_failure.get('capability'),
                failed_error_message=str(normalized_retry_failure.get('error_message') or ''),
            )
            if recovery_preview:
                validated_recovery_route, recovery_error = validate_route_decision(
                    recovery_preview,
                    instances=instances,
                    recent_artifacts=route_context.get('recent_artifacts') or [],
                )
                if validated_recovery_route:
                    route_payload = validated_recovery_route
                    route_source = 'self_heal'
                    logging.info(
                        'Ghost self-heal retry route: capability=%s confidence=%.3f reason=%s',
                        str(validated_recovery_route.get('capability') or '').strip(),
                        float(validated_recovery_route.get('confidence') or 0.0),
                        str(validated_recovery_route.get('reason') or '').strip(),
                    )
                elif recovery_error:
                    logging.info('Ghost self-heal preview fallback: %s', recovery_error)

        def build_ghost_carried_route_payload(
            *,
            follow_up_capabilities: Optional[list[str]] = None,
            follow_up_capability: Optional[str] = None,
            reason_override: Optional[str] = None,
        ) -> dict[str, Any]:
            deferred_capabilities = []
            for candidate in list(follow_up_capabilities or []) + [follow_up_capability]:
                normalized_candidate = normalize_capability(candidate)
                if not normalized_candidate or normalized_candidate in deferred_capabilities:
                    continue
                deferred_capabilities.append(normalized_candidate)
            hinted_capability = normalize_capability(
                (deferred_capabilities[0] if deferred_capabilities else None)
                or prompt_analysis.get('primary_capability')
            )
            failure_label = 'current turn remained on Ghost chat'
            failure_text = str(resolution_message or '').strip()
            if resolution_status == 'phase_resolved':
                failure_label = 'current phase already resolved by request phase graph'
            elif resolution_status == 'current_turn_resolved':
                failure_label = 'current turn was resolved directly by Ghost'
            continuation_label = 'carry on Ghost chat so the resolver and late fill remain available'
            if len(deferred_capabilities) > 1:
                continuation_label = (
                    'carry on Ghost chat so the resolver and late fill can continue toward '
                    + ', '.join(deferred_capabilities)
                )
            elif hinted_capability and hinted_capability != CAPABILITY_CHAT:
                continuation_label = (
                    f'carry on Ghost chat so the resolver and late fill can continue toward {hinted_capability}'
                )
            if failure_text and len(failure_text) > 140:
                failure_text = failure_text[:137].rstrip() + '...'
            reason = str(reason_override or '').strip() or f'{failure_label}; {continuation_label}'
            if failure_text and failure_text.lower() not in reason.lower():
                reason = f'{reason} ({failure_text})'
            return {
                'capability': CAPABILITY_CHAT,
                'confidence': 0.58 if hinted_capability and hinted_capability != CAPABILITY_CHAT else 0.52,
                'reuse_last_artifact': False,
                'artifact_path': None,
                'reason': reason,
            }

        def pick_primary_chat_instance_id() -> str:
            primary_chat_candidates = [
                entry
                for entry in instances
                if isinstance(entry, dict)
                and normalize_capability(entry.get('capability')) == CAPABILITY_CHAT
            ]
            preferred_plain_chat_instance_id = (
                pick_prompt_preferred_instance(primary_chat_candidates, route_context.get('prompt') or '')
                or self.pick_trait_aware_instance(primary_chat_candidates, route_context)
                or pick_default_capability_instance(primary_chat_candidates)
                or ''
            )
            return preferred_plain_chat_instance_id

        def validate_bounded_helper_route(candidate: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
            if not isinstance(candidate, dict):
                return None
            normalized_capability = normalize_capability(candidate.get('capability'))
            if not normalized_capability or normalized_capability == CAPABILITY_CHAT:
                return None
            route_candidate = dict(candidate)
            artifact_path = str(route_candidate.get('artifact_path') or '').strip() or None
            if not bool(route_candidate.get('reuse_last_artifact')) or not artifact_path:
                artifact_path = self._current_turn_artifact_anchor_path(
                    normalized_capability,
                    route_context=route_context,
                )
                if artifact_path:
                    route_candidate['reuse_last_artifact'] = True
                    route_candidate['artifact_path'] = artifact_path
            validated_candidate, helper_error = validate_route_decision(
                route_candidate,
                instances=instances,
                recent_artifacts=route_context.get('recent_artifacts') or [],
            )
            if not validated_candidate and helper_error:
                logging.info('Ghost bounded helper route fallback: %s', helper_error)
            return validated_candidate

        def apply_current_turn_resolution() -> bool:
            nonlocal route_payload, route_source, resolution_status, resolution_message, route_resolution_error
            if route_payload:
                return False

            vision_anchor_path = self._current_turn_artifact_anchor_path(
                CAPABILITY_VISION_ANALYSIS,
                route_context=route_context,
            )
            if vision_anchor_path and self._prompt_prefers_artifact_vision_analysis(route_context.get('prompt') or ''):
                validated_vision_route, vision_validation_error = validate_route_decision(
                    {
                        'capability': CAPABILITY_VISION_ANALYSIS,
                        'instance_id': None,
                        'reuse_last_artifact': True,
                        'artifact_path': vision_anchor_path,
                        'confidence': 0.9,
                        'reason': 'prompt refers to the latest image artifact',
                    },
                    instances=instances,
                    recent_artifacts=route_context.get('recent_artifacts') or [],
                )
                if validated_vision_route:
                    resolution_status = 'current_turn_resolved'
                    resolution_message = str(validated_vision_route.get('reason') or '').strip() or None
                    route_payload = validated_vision_route
                    route_source = 'ghost_carried'
                    return True
                if vision_validation_error:
                    logging.info('Ghost artifact vision-analysis fallback: %s', vision_validation_error)

            ensure_route_hint()
            bounded_artifact_follow_up_route = validate_bounded_helper_route(validated_route_hint or route_hint)
            bounded_helper_capability = normalize_capability(
                (bounded_artifact_follow_up_route or {}).get('capability')
            )
            bounded_helper_anchor_path = (
                self._current_turn_artifact_anchor_path(
                    bounded_helper_capability,
                    route_context=route_context,
                )
                if bounded_helper_capability
                else None
            )
            if bounded_artifact_follow_up_route and (
                bounded_helper_anchor_path
                or route_prompt_class in {
                    'artifact_follow_up',
                    'image_edit_follow_up',
                    'text_to_speech_follow_up',
                    'speech_to_text_follow_up',
                }
            ):
                resolution_status = 'current_turn_resolved'
                resolution_message = str(bounded_artifact_follow_up_route.get('reason') or '').strip() or None
                route_payload = bounded_artifact_follow_up_route
                route_source = 'ghost_carried'
                return True

            direct_chat_reason = 'single-phase text chat was resolved directly from the current user turn'
            if phase_graph_simple_chat_preview or (
                self._can_resolve_simple_current_turn_chat(
                    prompt_analysis=prompt_analysis,
                    selected_reference_artifacts=selected_reference_artifacts,
                    upload_filename=upload_filename,
                    file_path=file_path,
                    phase_graph_current_capability=phase_graph_current_capability,
                    phase_graph_downstream_capabilities=phase_graph_downstream_capabilities,
                )
            ):
                resolution_status = 'current_turn_resolved'
                resolution_message = direct_chat_reason
                route_payload = build_ghost_carried_route_payload(
                    reason_override=direct_chat_reason,
                )
                preferred_plain_chat_instance_id = pick_primary_chat_instance_id()
                if preferred_plain_chat_instance_id:
                    route_payload['instance_id'] = preferred_plain_chat_instance_id
                route_source = 'ghost_carried'
                return True

            if phase_graph_locked_current_phase:
                resolution_status = 'phase_resolved'
                resolution_message = phase_graph_current_reason
                route_payload = build_ghost_carried_route_payload(
                    follow_up_capabilities=phase_graph_downstream_capabilities,
                    reason_override=phase_graph_current_reason,
                )
                preferred_plain_chat_instance_id = pick_primary_chat_instance_id()
                if preferred_plain_chat_instance_id:
                    route_payload['instance_id'] = preferred_plain_chat_instance_id
                route_source = 'ghost_carried'
                return True

            current_turn_capability = normalize_capability(phase_graph_current_capability)
            if (
                not current_turn_capability
                or current_turn_capability == CAPABILITY_CHAT
            ):
                return False

            helper_route = None
            if normalize_capability((validated_route_hint or {}).get('capability')) == current_turn_capability:
                helper_route = validated_route_hint
            elif normalize_capability((route_hint or {}).get('capability')) == current_turn_capability:
                helper_route = route_hint

            artifact_path = str((helper_route or {}).get('artifact_path') or '').strip() or None
            reuse_last_artifact = bool((helper_route or {}).get('reuse_last_artifact')) and bool(artifact_path)
            if not reuse_last_artifact:
                artifact_path = self._current_turn_artifact_anchor_path(
                    current_turn_capability,
                    route_context=route_context,
                )
                reuse_last_artifact = bool(artifact_path)
            direct_reason = self._direct_current_turn_route_reason(
                current_turn_capability,
                route_context=route_context,
                helper_route=helper_route,
            )
            validated_direct_route, direct_validation_error = validate_route_decision(
                {
                    'capability': current_turn_capability,
                    'instance_id': str((helper_route or {}).get('instance_id') or '').strip() or None,
                    'reuse_last_artifact': reuse_last_artifact,
                    'artifact_path': artifact_path if reuse_last_artifact else None,
                    'confidence': self._direct_current_turn_route_confidence(
                        current_turn_capability,
                        route_context=route_context,
                        helper_route=helper_route,
                    ),
                    'reason': direct_reason,
                },
                instances=instances,
                recent_artifacts=route_context.get('recent_artifacts') or [],
            )
            if not validated_direct_route:
                if direct_validation_error:
                    logging.info('Ghost current-turn resolution fallback: %s', direct_validation_error)
                    route_resolution_error = direct_validation_error
                return False

            resolution_status = 'current_turn_resolved'
            resolution_message = direct_reason
            route_payload = validated_direct_route
            route_source = 'ghost_carried'
            return True

        apply_current_turn_resolution()

        if (
            raw_preview_payload
            and (
                not route_payload
                or normalize_capability(route_payload.get('capability')) == CAPABILITY_CHAT
            )
        ):
            ensure_route_hint()
            live_hint = validated_route_hint or route_hint
            live_hint_capability = normalize_capability((live_hint or {}).get('capability'))
            direct_visual_count = int(prompt_analysis.get('requested_visual_output_count') or 0)
            can_promote_live_preview_truth = (
                live_hint_capability
                and live_hint_capability != CAPABILITY_CHAT
                and self.should_ignore_preview_route_for_live_heuristic(
                    preview_route_candidate or raw_preview_payload,
                    live_hint,
                )
                and not bool(prompt_analysis.get('text_preparation_before_visual_output'))
                and not bool(prompt_analysis.get('text_preparation_before_audio_output'))
                and direct_visual_count <= 1
            )
            if can_promote_live_preview_truth:
                direct_preview_route = dict(live_hint or {})
                direct_preview_route.setdefault('capability', live_hint_capability)
                direct_preview_route.setdefault('confidence', 0.82)
                direct_preview_route.setdefault(
                    'reason',
                    'live deterministic route overrode stale chat preview',
                )
                validated_direct_preview_route, direct_preview_error = validate_route_decision(
                    direct_preview_route,
                    instances=instances,
                    recent_artifacts=route_context.get('recent_artifacts') or [],
                )
                if validated_direct_preview_route:
                    resolution_status = 'current_turn_resolved'
                    resolution_message = str(
                        validated_direct_preview_route.get('reason') or ''
                    ).strip() or None
                    route_payload = validated_direct_preview_route
                    route_source = 'ghost_carried'
                elif direct_preview_error:
                    logging.info('Ghost stale-preview live route fallback: %s', direct_preview_error)

        if not route_payload and preview_route_candidate:
            route_payload = preview_route_candidate
            route_source = preview_route_source or 'preview'

        if (
            not route_payload
            and not phase_graph_locked_current_phase
            and embedding_signals_enabled
            and semantic_compute_allowed
        ):
            attach_embedding_hints_to_route_context(
                route_context,
                runtime_manifest=runtime_manifest,
                instances=instances,
            )
            embedding_signals_attached = True

        if not route_payload and embedding_signals_enabled and semantic_compute_allowed:
            ensure_route_hint()
            if not embedding_signals_attached:
                attach_embedding_hints_to_route_context(
                    route_context,
                    runtime_manifest=runtime_manifest,
                    instances=instances,
                )
                embedding_signals_attached = True
            embedding_bias_route = maybe_apply_embedding_route_bias(
                route_context,
                route_hint=validated_route_hint or route_hint or {'capability': CAPABILITY_CHAT},
            )
            if embedding_bias_route:
                validated_embedding_bias_route, embedding_bias_error = validate_route_decision(
                    embedding_bias_route,
                    instances=instances,
                    recent_artifacts=route_context.get('recent_artifacts') or [],
                )
                if validated_embedding_bias_route:
                    route_payload = validated_embedding_bias_route
                    route_source = 'embedding_tiebreak'
                    embedding_bias_applied = True
                    logging.info(
                        'Ghost embedding tie-break route: capability=%s confidence=%.3f reason=%s',
                        str(validated_embedding_bias_route.get('capability') or '').strip(),
                        float(validated_embedding_bias_route.get('confidence') or 0.0),
                        str(validated_embedding_bias_route.get('reason') or '').strip(),
                    )
                elif embedding_bias_error:
                    logging.info('Ghost embedding tie-break fallback: %s', embedding_bias_error)

        if (
            route_payload
            and phase_graph_locked_current_phase
            and normalize_capability(route_payload.get('capability')) != CAPABILITY_CHAT
        ):
            resolution_status = 'phase_resolved'
            resolution_message = phase_graph_current_reason
            route_payload = build_ghost_carried_route_payload(
                follow_up_capabilities=phase_graph_downstream_capabilities,
                reason_override=phase_graph_current_reason,
            )
            route_source = 'ghost_carried'
        if not route_payload and route_resolution_error:
            return None, route_resolution_error
        if not route_payload:
            route_payload = build_ghost_carried_route_payload(
                follow_up_capabilities=phase_graph_downstream_capabilities,
                follow_up_capability=phase_graph_current_capability,
            )
            route_source = 'ghost_carried'

        if route_payload and (raw_preview_payload or not bool(data.get('ghost_route'))):
            ensure_route_hint()
            heuristic_candidate_for_preview = validated_route_hint or route_hint or {}
            heuristic_capability_for_preview = normalize_capability(
                heuristic_candidate_for_preview.get('capability')
                if isinstance(heuristic_candidate_for_preview, dict)
                else None
            )
            preview_capability_for_preview = normalize_capability(
                (preview_route_candidate or raw_preview_payload).get('capability')
                if isinstance(preview_route_candidate or raw_preview_payload, dict)
                else None
            )
            try:
                heuristic_confidence_for_preview = float(
                    (heuristic_candidate_for_preview or {}).get('confidence') or 0.0
                )
            except (TypeError, ValueError):
                heuristic_confidence_for_preview = 0.0
            direct_visual_count = int(prompt_analysis.get('requested_visual_output_count') or 0)
            preview_allows_live_truth = (
                preview_capability_for_preview == CAPABILITY_CHAT
                if raw_preview_payload
                else True
            )
            if (
                normalize_capability(route_payload.get('capability')) == CAPABILITY_CHAT
                and preview_allows_live_truth
                and heuristic_capability_for_preview
                and heuristic_capability_for_preview != CAPABILITY_CHAT
                and heuristic_confidence_for_preview >= 0.9
                and not bool(prompt_analysis.get('text_preparation_before_visual_output'))
                and not bool(prompt_analysis.get('text_preparation_before_audio_output'))
                and direct_visual_count <= 1
            ):
                validated_direct_preview_route, direct_preview_error = validate_route_decision(
                    dict(heuristic_candidate_for_preview),
                    instances=instances,
                    recent_artifacts=route_context.get('recent_artifacts') or [],
                )
                if validated_direct_preview_route:
                    route_payload = validated_direct_preview_route
                    route_source = 'ghost_carried'
                    resolution_status = 'current_turn_resolved'
                    resolution_message = str(
                        validated_direct_preview_route.get('reason') or ''
                    ).strip() or None
                elif direct_preview_error:
                    logging.info('Ghost final stale-preview live route fallback: %s', direct_preview_error)

        if not route_payload:
            return None, 'Ghost could not resolve a valid route.'

        def route_workload_task_proposals() -> list[dict[str, Any]]:
            for candidate in (route_payload, validated_route_hint, route_hint, preview_route_candidate):
                proposals = (
                    candidate.get('workload_task_proposals')
                    if isinstance(candidate, dict) and isinstance(candidate.get('workload_task_proposals'), list)
                    else None
                )
                if proposals:
                    return [dict(item) for item in proposals if isinstance(item, dict)]
            return []

        workload_task_proposals = route_workload_task_proposals()
        if workload_task_proposals and isinstance(route_payload, dict) and not route_payload.get('workload_task_proposals'):
            route_payload['workload_task_proposals'] = workload_task_proposals

        capability = str(route_payload.get('capability') or '').strip()
        candidates = [
            entry
            for entry in instances
            if isinstance(entry, dict)
            and instance_supports_capability(entry, capability)
        ]
        excluded_ids = {
            str(item or '').strip()
            for item in (excluded_instance_ids or [])
            if str(item or '').strip()
        }
        if excluded_ids:
            candidates = [
                entry
                for entry in candidates
                if str(entry.get('instance_id') or '').strip() not in excluded_ids
            ]
        selected_instance_id = str(route_payload.get('instance_id') or '').strip()
        if selected_instance_id and selected_instance_id in excluded_ids:
            selected_instance_id = ''
        preferred_instance_id, preference_meta = self.pick_ghost_preference_instance(
            candidates,
            route_context,
            route_selected_instance_id=selected_instance_id or None,
            requested_capability=capability,
        )
        if preferred_instance_id:
            selected_instance_id = preferred_instance_id
        elif not isinstance(preference_meta, dict):
            preference_meta = None
        if not selected_instance_id:
            selected_instance_id = (
                pick_prompt_preferred_instance(candidates, route_context.get('prompt') or '')
                or self.pick_trait_aware_instance(candidates, route_context)
                or pick_default_capability_instance(candidates)
                or ''
            )
        if not selected_instance_id:
            return None, f"No running instance found for capability '{capability}'."
        selected_instance = next(
            (entry for entry in candidates if str(entry.get('instance_id') or '').strip() == selected_instance_id),
            None,
        )
        if not selected_instance:
            return None, f"Instance '{selected_instance_id}' for capability '{capability}' could not be resolved."

        route_traits = self.build_instance_trait_summary(selected_instance)
        has_preview_file_context = bool(upload_filename or file_path) or bool(
            route_payload.get('reuse_last_artifact') and route_payload.get('artifact_path')
        )
        context_strategy = self.choose_context_strategy(
            instance=selected_instance,
            messages=ghost_messages,
            prompt=prompt,
            has_file_context=has_preview_file_context,
        )

        resolved = {
            'instance_id': selected_instance_id,
            'instance': selected_instance,
            'capability': capability,
            'request_meta': (
                route_context.get('request_meta')
                if isinstance(route_context.get('request_meta'), dict)
                else None
            ),
            'route_source': route_source,
            'route_reason': self.augment_route_reason(
                str(route_payload.get('reason') or '').strip() or 'ghost route',
                selected_instance,
                route_context,
            ),
            'route_confidence': float(route_payload.get('confidence') or 0.0),
            'route_reuse_last_artifact': bool(route_payload.get('reuse_last_artifact')),
            'route_artifact_ref': self.resolve_route_artifact_ref(
                route_context,
                artifact_path=route_payload.get('artifact_path'),
                preview_payload=preview_payload,
            ),
            'route_artifact_path': str(route_payload.get('artifact_path') or '').strip() or None,
            'route_runtime': {
                'embedding_helper': (route_context.get('runtime') or {}).get('embedding_helper'),
                'embedding_hints': (route_context.get('runtime') or {}).get('embedding_hints'),
                'accepted_learning_hints': (route_context.get('runtime') or {}).get('accepted_learning_hints'),
                'ghost_preferences': (route_context.get('runtime') or {}).get('ghost_preferences'),
                'ghost_preference_selection': preference_meta,
                'prompt_class': route_prompt_class,
                'session_class': route_session_class,
                'routing_preferences': routing_preferences,
                'semantic_role_profile': semantic_role_profile,
                'route_traits': route_traits,
                'context_strategy': context_strategy,
            },
        }
        if workload_task_proposals:
            resolved['workload_task_proposals'] = workload_task_proposals
        if consistency_downstream_branches:
            resolved['downstream_branches'] = consistency_downstream_branches
        if route_graph_consistency_diagnostics:
            route_runtime = (
                resolved.get('route_runtime')
                if isinstance(resolved.get('route_runtime'), dict)
                else {}
            )
            developer_diagnostics = (
                route_runtime.get('developer_diagnostics')
                if isinstance(route_runtime.get('developer_diagnostics'), dict)
                else {}
            )
            developer_diagnostics['route_graph_consistency'] = route_graph_consistency_diagnostics
            route_runtime['developer_diagnostics'] = developer_diagnostics
            resolved['route_runtime'] = route_runtime
        route_runtime = self.merge_request_meta_runtime_truth(
            resolved.get('route_runtime') if isinstance(resolved.get('route_runtime'), dict) else {},
            data,
            route_payload=resolved,
        )
        if preview_mode:
            embedding_helper = (
                route_runtime.get('embedding_helper')
                if isinstance(route_runtime.get('embedding_helper'), dict)
                else {}
            )
            semantic_compute_performed = bool(embedding_helper.get('semantic_compute_performed'))
            route_runtime['semantic_compute'] = {
                'requested': bool(compute_semantics),
                'allowed': bool(semantic_compute_allowed),
                'performed': semantic_compute_performed,
                'preview': True,
                'learnable': False,
                'evidence_role': (
                    'preview_computed_non_learnable'
                    if semantic_compute_performed
                    else 'preview_cached_non_learnable'
                ),
            }
        heuristic_candidate = validated_route_hint or route_hint or {}
        heuristic_capability = normalize_capability((heuristic_candidate or {}).get('capability'))
        final_capability = normalize_capability((route_payload or {}).get('capability'))
        heuristic_shadow = None
        if isinstance(heuristic_candidate, dict) and heuristic_candidate:
            heuristic_shadow = {
                'capability': heuristic_capability or None,
                'confidence': round(float(heuristic_candidate.get('confidence') or 0.0), 4),
                'reason': str(heuristic_candidate.get('reason') or '').strip() or None,
                'reuse_last_artifact': bool(heuristic_candidate.get('reuse_last_artifact')),
                'artifact_path': str(heuristic_candidate.get('artifact_path') or '').strip() or None,
                'validation_error': route_hint_validation_error,
            }
        route_runtime['routing_policy'] = {
            'mode': 'ghost_first',
            'decision_authority': 'request_phase_graph_and_runtime_truth',
            'heuristic_role': 'shadow_guardrail',
            'heuristic_shadow': heuristic_shadow,
            'guardrail_effect': (
                'aligned_with_final_route'
                if heuristic_capability and final_capability and heuristic_capability == final_capability
                else 'shadow_only'
            ),
            'accepted_learning_authority': developer_flags.get('accepted_learning_authority') or 'soft_hint',
            'future_authority_levels': ['soft_hint', 'advisory', 'preferred', 'enforced'],
        }
        route_runtime['ghost_resolution'] = {
            'status': resolution_status or ('resolved' if route_payload else None),
            'message': resolution_message,
            'carried': route_source == 'ghost_carried',
            'route_source': route_source,
        }
        route_runtime['embedding_audit'] = build_embedding_route_audit(
            route_context,
            route_hint=validated_route_hint or route_hint or {},
            final_route=route_payload or {},
            bias_applied=embedding_bias_applied,
        )
        if normalized_retry_failure:
            route_runtime['retry_failure'] = {
                'failed_capability': normalize_capability(normalized_retry_failure.get('capability')),
                'failed_instance_id': str(normalized_retry_failure.get('failed_instance_id') or '').strip() or None,
                'status_code': int(normalized_retry_failure.get('status_code') or 0) or None,
                'error_message': str(normalized_retry_failure.get('error_message') or '').strip() or None,
            }
        route_runtime['working_frame'] = build_working_frame(
            request_payload=data,
            route_payload={**resolved, 'route_runtime': route_runtime},
            freeze=False,
        )
        resolved['route_runtime'] = route_runtime
        return resolved, None
