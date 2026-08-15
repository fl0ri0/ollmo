"""Normalize current-turn intent promises into graph obligation records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from helpers.model_capabilities import (
    CAPABILITY_CHAT,
    CAPABILITY_IMAGE_GENERATION,
    CAPABILITY_SPEECH_TO_TEXT,
    CAPABILITY_TEXT_TO_SPEECH,
    CAPABILITY_VISION_ANALYSIS,
    normalize_capability,
)


_LOCAL_VISUAL_BINDING_RE = re.compile(
    r'\b(?:local(?:ly)?|lokal(?:e|en|er|es)?)\b[^.;!?]{0,120}\b'
    r'(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?|assets?)\b'
    r'|'
    r'\b(?:generated|generierte(?:n|s|r|m)?|erzeugte(?:n|s|r|m)?)\b[^.;!?]{0,80}\b'
    r'(?:local(?:ly)?|lokal(?:e|en|er|es)?)\b[^.;!?]{0,80}\b'
    r'(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?|assets?)\b'
    r'|'
    r'\b(?:linked|link|linking|eingebunden|verlinkt|verweisen)\b[^.;!?]{0,120}\b'
    r'(?:local(?:ly)?|lokal(?:e|en|er|es)?)?\s*'
    r'(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?|assets?)\b',
    re.IGNORECASE,
)
_SHARED_CSS_RE = re.compile(
    r'\b(?:shared|common|gemeinsam(?:e|en|er|es)?|same)\b[^.;!?]{0,80}\b(?:css|stylesheet|styles\.css)\b'
    r'|'
    r'\b(?:styles\.css|style\.css|stylesheet)\b',
    re.IGNORECASE,
)
_NAVIGATION_RE = re.compile(
    r'\b(?:navigation|nav|links?|verlinkung|verlinkt|linked)\b[^.;!?]{0,120}\b'
    r'(?:between|among|zwischen|both|beide|pages?|seiten?)\b'
    r'|'
    r'\b(?:between|zwischen)\b[^.;!?]{0,80}\b(?:pages?|seiten?)\b',
    re.IGNORECASE,
)
_NON_EXECUTABLE_OBLIGATION_STATES = {
    'candidate',
    'deferred',
    'draft',
    'not-promoted',
    'not_promoted',
    'optional',
    'possible',
    'reserved',
    'unpromoted',
}


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _slug(value: Any, *, fallback: str = 'item') -> str:
    text = _clean_text(value).lower()
    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return text or fallback


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        text = _clean_text(raw_item)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def _phase_id(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('phase_id') or record.get('branch_id'))


def _branch_id(record: Mapping[str, Any]) -> str:
    return _clean_text(record.get('branch_id') or record.get('phase_id'))


def _text_artifact_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _clean_text(record.get('text_artifact_source_name') or record.get('source_name')).lower(),
        _clean_text(record.get('text_artifact_extension') or record.get('extension')).lower(),
    )


def _text_branch_lookup(branches: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        key = _text_artifact_key(branch)
        if key[0] and key[1] and key not in lookup:
            lookup[key] = branch
    return lookup


def _media_obligation_id(capability: str, queue_index: int) -> str:
    return f'intent-obligation-media-{_slug(capability)}-{queue_index}'


def _text_obligation_id(name: str, extension: str) -> str:
    return f'intent-obligation-text-{_slug(name)}-{_slug(extension)}'


def _dependency_obligation_id(contract: str, target_id: str, source_id: str) -> str:
    return f'intent-obligation-dependency-{_slug(contract)}-{_slug(target_id)}-{_slug(source_id)}'


def _navigation_obligation_id(source_name: str, target_name: str) -> str:
    return f'intent-obligation-navigation-{_slug(source_name)}-to-{_slug(target_name)}'


def _prompt_text(prompt_analysis: Mapping[str, Any], prompt: str | None) -> str:
    return _clean_text(prompt_analysis.get('normalized_prompt') or prompt).lower()


def _local_visual_binding_required(prompt_analysis: Mapping[str, Any], prompt: str | None) -> bool:
    text = _prompt_text(prompt_analysis, prompt)
    return bool(prompt_analysis.get('local_visual_asset_requirement') or _LOCAL_VISUAL_BINDING_RE.search(text))


def _shared_css_required(prompt_analysis: Mapping[str, Any], prompt: str | None) -> bool:
    return bool(_SHARED_CSS_RE.search(_prompt_text(prompt_analysis, prompt)))


def _navigation_required(prompt_analysis: Mapping[str, Any], prompt: str | None) -> bool:
    return bool(_NAVIGATION_RE.search(_prompt_text(prompt_analysis, prompt)))


def required_intent_obligations(
    intent_obligations: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Return only obligations that already represent executable owed work."""

    required: list[Mapping[str, Any]] = []
    for obligation in intent_obligations or []:
        if not isinstance(obligation, Mapping) or obligation.get('required') is False:
            continue
        if _clean_text(obligation.get('kind')).lower() == 'intent_cardinality_guard':
            # This is required clarification truth, not executable owed work.
            # Closure handles it as one blocked branch-contract guard; projecting
            # it as a runnable capability would incorrectly synthesize media work.
            continue
        if _clean_text(obligation.get('promotion_policy')).lower() == 'reserved_only':
            continue
        states = {
            _clean_text(obligation.get(key)).lower()
            for key in (
                'status',
                'candidate_status',
                'contract_state',
                'contract_status',
                'obligation_state',
                'intent_state',
            )
            if _clean_text(obligation.get(key))
        }
        if states & _NON_EXECUTABLE_OBLIGATION_STATES:
            continue
        required.append(obligation)
    return required


def summarize_required_intent_obligations(
    intent_obligations: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Build a compact projection from authoritative executable intent obligations."""

    capability_counts: dict[str, int] = {}
    material_output_counts: dict[str, int] = {}
    capabilities: list[str] = []
    kinds: list[str] = []
    records = required_intent_obligations(intent_obligations)
    for obligation in records:
        try:
            count = int(obligation.get('count') or 1)
        except (TypeError, ValueError):
            count = 1
        count = max(1, count)
        kind = _clean_text(obligation.get('kind')).lower()
        if kind and kind not in kinds:
            kinds.append(kind)
        capability = normalize_capability(obligation.get('capability'))
        if capability:
            capability_counts[capability] = capability_counts.get(capability, 0) + count
            if capability not in capabilities:
                capabilities.append(capability)
        output_type = _clean_text(obligation.get('output_type')).lower()
        if kind == 'media_artifact' and output_type in {'audio', 'image'}:
            material_output_counts[output_type] = material_output_counts.get(output_type, 0) + count
    return {
        'required_count': len(records),
        'kinds': kinds,
        'capabilities': capabilities,
        'capability_counts': capability_counts,
        'material_output_counts': material_output_counts,
    }


def build_intent_obligation_ledger(
    *,
    prompt: str | None = None,
    prompt_analysis: Mapping[str, Any] | None = None,
    text_artifact_requests: Sequence[Mapping[str, Any]] | None = None,
    downstream_branches: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build normalized promise records from current-turn intent and graph branches."""

    analysis = prompt_analysis if isinstance(prompt_analysis, Mapping) else {}
    branches = [item for item in (downstream_branches or []) if isinstance(item, Mapping)]
    text_requests = [item for item in (text_artifact_requests or []) if isinstance(item, Mapping)]
    text_lookup = _text_branch_lookup(branches)
    obligations: list[dict[str, Any]] = []
    text_obligations: list[dict[str, Any]] = []

    for request in text_requests:
        extension = _clean_text(request.get('extension')).lower() or 'txt'
        name = _clean_text(request.get('source_name')) or f'generated-{extension}'
        branch = text_lookup.get((name.lower(), extension), {})
        obligation = {
            'obligation_id': _text_obligation_id(name, extension),
            'kind': 'text_artifact',
            'source': 'current_user_intent',
            'evidence': _clean_text(request.get('source')) or 'text_artifact_request',
            'required': True,
            'output_type': 'text',
            'capability': CAPABILITY_CHAT,
            'count': 1,
            'target_extension': extension,
            'target_name': name,
            'relationship': 'materializes',
            'depends_on_obligation_ids': [],
            'dependency_contract': None,
            'promotion_policy': 'promote_if_current_turn_explicit',
        }
        if branch:
            obligation.update(
                {
                    'phase_id': _phase_id(branch),
                    'branch_id': _branch_id(branch),
                }
            )
        text_obligations.append(obligation)
        obligations.append(obligation)

    media_obligations: list[dict[str, Any]] = []
    media_index = 0
    evidence_source_by_phase: dict[str, str] = {}
    for branch in branches:
        capability = normalize_capability(branch.get('capability'))
        if capability not in {CAPABILITY_IMAGE_GENERATION, CAPABILITY_TEXT_TO_SPEECH}:
            continue
        if _clean_text(branch.get('contract_state')).lower() == 'reserved':
            continue
        media_index += 1
        queue_index = int(branch.get('queue_index') or media_index)
        output_type = 'image' if capability == CAPABILITY_IMAGE_GENERATION else 'audio'
        obligation = {
            'obligation_id': _media_obligation_id(capability, queue_index),
            'kind': 'media_artifact',
            'source': 'current_user_intent',
            'evidence': _clean_text(branch.get('source')) or f'{capability}_branch',
            'required': branch.get('required') if isinstance(branch.get('required'), bool) else True,
            'output_type': output_type,
            'capability': capability,
            'count': 1,
            'queue_index': queue_index,
            'target_extension': output_type,
            'target_name': f'{output_type}-{queue_index}',
            'relationship': 'materializes',
            'depends_on_obligation_ids': [],
            'dependency_contract': None,
            'promotion_policy': 'promote_if_current_turn_explicit',
            'phase_id': _phase_id(branch),
            'branch_id': _branch_id(branch),
        }
        media_obligations.append(obligation)
        obligations.append(obligation)
        phase_id = _phase_id(branch)
        if phase_id:
            evidence_source_by_phase[phase_id] = obligation['obligation_id']

    evidence_index = 0
    for branch in branches:
        capability = normalize_capability(branch.get('capability'))
        if capability not in {CAPABILITY_VISION_ANALYSIS, CAPABILITY_SPEECH_TO_TEXT, CAPABILITY_CHAT}:
            continue
        depends_on = _clean_string_list(branch.get('depends_on'))
        source_obligation_ids = [
            evidence_source_by_phase[item]
            for item in depends_on
            if item in evidence_source_by_phase
        ]
        if not source_obligation_ids and capability != CAPABILITY_CHAT:
            continue
        if capability == CAPABILITY_CHAT:
            source_obligation_ids = [
                evidence_source_by_phase.get(item, '')
                for item in depends_on
                if evidence_source_by_phase.get(item, '')
            ]
            source_obligation_ids.extend(
                item.get('obligation_id')
                for item in obligations
                if item.get('kind') == 'evidence_branch'
                and item.get('phase_id') in depends_on
            )
            source_obligation_ids = [item for item in source_obligation_ids if item]
            if not source_obligation_ids:
                continue
        evidence_index += 1
        obligation_id = f'intent-obligation-evidence-{_slug(capability)}-{evidence_index}'
        obligations.append(
            {
                'obligation_id': obligation_id,
                'kind': 'evidence_branch',
                'source': 'current_user_intent',
                'evidence': _clean_text(branch.get('source')) or f'{capability}_evidence_branch',
                'required': branch.get('required') if isinstance(branch.get('required'), bool) else True,
                'output_type': 'text',
                'capability': capability,
                'count': 1,
                'target_extension': 'text',
                'target_name': f'{capability}-{evidence_index}',
                'relationship': 'evidence_for',
                'depends_on_obligation_ids': list(dict.fromkeys(source_obligation_ids)),
                'dependency_contract': 'media_evidence_binding',
                'promotion_policy': 'promote_if_current_turn_explicit',
                'phase_id': _phase_id(branch),
                'branch_id': _branch_id(branch),
            }
        )
        phase_id = _phase_id(branch)
        if phase_id:
            evidence_source_by_phase[phase_id] = obligation_id

    image_obligations = [
        item for item in media_obligations
        if item.get('capability') == CAPABILITY_IMAGE_GENERATION and item.get('phase_id')
    ]
    html_text_obligations = [
        item for item in text_obligations
        if item.get('target_extension') in {'html', 'htm'} and item.get('phase_id')
    ]
    css_text_obligations = [
        item for item in text_obligations
        if item.get('target_extension') == 'css' and item.get('phase_id')
    ]
    json_text_obligations = [
        item for item in text_obligations
        if item.get('target_extension') == 'json' and item.get('phase_id')
    ]

    local_visual_targets = list(html_text_obligations)
    if analysis.get('local_visual_asset_requirement'):
        local_visual_targets.extend(css_text_obligations)
        local_visual_targets.extend(json_text_obligations)

    if image_obligations and local_visual_targets and _local_visual_binding_required(analysis, prompt):
        image_obligation_ids = [
            item['obligation_id'] for item in image_obligations if item.get('obligation_id')
        ]
        image_phase_ids = [item['phase_id'] for item in image_obligations if item.get('phase_id')]
        for target in local_visual_targets:
            target_phase_id = _clean_text(target.get('phase_id'))
            if not target_phase_id:
                continue
            obligations.append(
                {
                    'obligation_id': _dependency_obligation_id(
                        'local_visual_asset_binding',
                        target_phase_id,
                        '-'.join(image_phase_ids),
                    ),
                    'kind': 'dependency',
                    'source': 'current_user_intent',
                    'evidence': 'local_visual_asset_binding_requirement',
                    'required': True,
                    'output_type': None,
                    'capability': None,
                    'count': len(image_phase_ids),
                    'target_extension': target.get('target_extension'),
                    'target_name': target.get('target_name'),
                    'relationship': 'depends_on',
                    'depends_on_obligation_ids': image_obligation_ids,
                    'dependency_contract': 'local_visual_asset_binding',
                    'promotion_policy': 'promote_if_current_turn_explicit',
                    'execution_dependency_required': True,
                    'target_phase_id': target_phase_id,
                    'target_branch_id': target.get('branch_id'),
                    'source_phase_ids': image_phase_ids,
                    'repair_action': 'rebind_artifact_dependency',
                    'add_dependencies': [
                        {
                            'target_phase_id': target_phase_id,
                            'source_phase_id': source_phase_id,
                            'dependency_contract': 'local_visual_asset_binding',
                        }
                        for source_phase_id in image_phase_ids
                    ],
                }
            )

    if html_text_obligations and css_text_obligations and _shared_css_required(analysis, prompt):
        css = css_text_obligations[0]
        for target in html_text_obligations:
            obligations.append(
                {
                    'obligation_id': _dependency_obligation_id(
                        'shared_css_binding',
                        _clean_text(target.get('phase_id') or target.get('target_name')),
                        _clean_text(css.get('phase_id') or css.get('target_name')),
                    ),
                    'kind': 'dependency',
                    'source': 'current_user_intent',
                    'evidence': 'shared_css_binding_requirement',
                    'required': True,
                    'output_type': None,
                    'capability': None,
                    'count': 1,
                    'target_extension': target.get('target_extension'),
                    'target_name': target.get('target_name'),
                    'relationship': 'references',
                    'depends_on_obligation_ids': [css.get('obligation_id')],
                    'dependency_contract': 'shared_css_binding',
                    'promotion_policy': 'promote_if_current_turn_explicit',
                    'execution_dependency_required': False,
                    'target_phase_id': target.get('phase_id'),
                    'source_phase_ids': [css.get('phase_id')],
                }
            )

    if len(html_text_obligations) >= 2 and _navigation_required(analysis, prompt):
        for source, target in ((html_text_obligations[0], html_text_obligations[1]), (html_text_obligations[1], html_text_obligations[0])):
            obligations.append(
                {
                    'obligation_id': _navigation_obligation_id(
                        _clean_text(source.get('target_name')),
                        _clean_text(target.get('target_name')),
                    ),
                    'kind': 'navigation',
                    'source': 'current_user_intent',
                    'evidence': 'navigation_between_pages_requirement',
                    'required': True,
                    'output_type': None,
                    'capability': None,
                    'count': 1,
                    'target_extension': source.get('target_extension'),
                    'target_name': source.get('target_name'),
                    'from_target_name': source.get('target_name'),
                    'to_target_name': target.get('target_name'),
                    'relationship': 'navigates_to',
                    'depends_on_obligation_ids': [target.get('obligation_id')],
                    'dependency_contract': 'navigation_binding',
                    'promotion_policy': 'promote_if_current_turn_explicit',
                }
            )

    return [dict(item) for item in obligations]


def apply_intent_obligation_dependency_edges(
    downstream_branches: list[dict[str, Any]],
    intent_obligations: Sequence[Mapping[str, Any]],
) -> None:
    """Apply safe current-turn dependency obligations to mutable branch records."""

    branch_lookup: dict[str, dict[str, Any]] = {}
    for branch in downstream_branches:
        if not isinstance(branch, dict):
            continue
        for record_id in (_phase_id(branch), _branch_id(branch)):
            if record_id and record_id not in branch_lookup:
                branch_lookup[record_id] = branch
    for obligation in intent_obligations:
        if not isinstance(obligation, Mapping):
            continue
        if obligation.get('kind') != 'dependency' or not obligation.get('execution_dependency_required'):
            continue
        target_phase_id = _clean_text(obligation.get('target_phase_id') or obligation.get('target_branch_id'))
        source_phase_ids = _clean_string_list(obligation.get('source_phase_ids'))
        target = branch_lookup.get(target_phase_id)
        if not target or not source_phase_ids:
            continue
        current = _clean_string_list(target.get('depends_on'))
        if obligation.get('dependency_contract') == 'local_visual_asset_binding' and current in ([], ['phase-1']):
            merged = list(source_phase_ids)
        else:
            merged = list(current)
            for source_phase_id in source_phase_ids:
                if source_phase_id not in merged:
                    merged.append(source_phase_id)
        target['depends_on'] = merged
        target['dependency_contract'] = _clean_text(obligation.get('dependency_contract'))
        target['intent_obligation_id'] = _clean_text(obligation.get('obligation_id'))
        if obligation.get('dependency_contract') == 'local_visual_asset_binding':
            target['image_asset_binding_required'] = True
            target['required_image_phase_ids'] = source_phase_ids
