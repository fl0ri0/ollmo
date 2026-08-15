"""Request-intake normalization owners for Ollmo."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from helpers.model_capabilities import CAPABILITY_EMBEDDING
from ollmo_g.text_artifact_revision_intent import classify_named_text_revision_intent
from ollmo_services.responses import extract_canonical_predecessor_image_prompts

_TEMPORARY_QUARANTINED_MODEL_MARKERS: tuple[str, ...] = (
    'mlx-community/qwen3.6-35b-a3b-4bit',
    'models--mlx-community--qwen3.6-35b-a3b-4bit',
)
_TEMPORARY_QUARANTINE_REASON = (
    'Temporarily quarantined due to repeated MLX Qwen3.6 system-message-ordering and/or OOM failures.'
)
_CURRENT_PREDECESSOR_IMAGE_MATERIALIZATION_RE = re.compile(
    r'\b(?:create|generate|make|produce|render|'
    r'erstell(?:e|en|t)?|generier(?:e|en|t)?|erzeug(?:e|en|t)?|mach(?:e|en|t)?)\b'
    r'[^.;!?\n]{0,120}\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?)\b|'
    r'\b(?:image(?:s)?|picture(?:s)?|photo(?:s)?|bild(?:er)?|foto(?:s)?)\b'
    r'[^.;!?\n]{0,120}\b(?:create|generate|make|produce|render|'
    r'erstell(?:e|en|t)?|generier(?:e|en|t)?|erzeug(?:e|en|t)?|mach(?:e|en|t)?)\b',
    re.IGNORECASE,
)
_CURRENT_PREDECESSOR_PAGE_BINDING_RE = re.compile(
    r'\b(?:link|bind|embed|insert|include|update|repair|fix|'
    r'verlink(?:e|en|t)?|verkn(?:u|ü)pf(?:e|en|t)?|einbind(?:e|en|et)?|'
    r'einf(?:u|ü)g(?:e|en|t)?|aktualisier(?:e|en|t)?|reparier(?:e|en|t)?)\b'
    r'[^.;!?\n]{0,160}\b(?:site|page|website|webseite|seite|landingpage|landing[\s-]?page|html)\b|'
    r'\b(?:site|page|website|webseite|seite|landingpage|landing[\s-]?page|html)\b'
    r'[^.;!?\n]{0,160}\b(?:link|bind|embed|insert|include|update|repair|fix|'
    r'verlink(?:e|en|t)?|verkn(?:u|ü)pf(?:e|en|t)?|einbind(?:e|en|et)?|'
    r'einf(?:u|ü)g(?:e|en|t)?|aktualisier(?:e|en|t)?|reparier(?:e|en|t)?)\b',
    re.IGNORECASE,
)
_CURRENT_PREDECESSOR_TEXT_ARTIFACT_TYPES = {'text', 'document', 'file'}
_CURRENT_PREDECESSOR_NAMED_TEXT_EDIT_ORIGIN = 'current_predecessor_named_text_edit'


def _explicit_current_turn_named_text_edit_targets(prompt: str) -> list[str]:
    """Return ordered filenames explicitly governed by a current-turn edit action."""
    intent = classify_named_text_revision_intent(prompt)
    return [str(item) for item in intent.get('named_targets') or [] if str(item)]


@dataclass
class RequestIntakeRuntimeOwner:
    hooks: dict[str, Any]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def _resolve_wrapper_capability(self, selector: str) -> Optional[str]:
        normalize_capability = self._hook('normalize_capability')
        wrapper_capability_aliases = self._hook('wrapper_capability_aliases')

        normalized = normalize_capability(selector) or str(selector or '').strip().lower()
        if not normalized:
            return None
        for capability, aliases in wrapper_capability_aliases.items():
            if normalized == capability or normalized in aliases:
                return capability
        return None

    def _instance_is_temporarily_quarantined(self, instance: Any) -> bool:
        normalize_backend = self._hook('normalize_backend')

        if not isinstance(instance, dict):
            return False
        if normalize_backend(instance.get('backend')) != 'mlx':
            return False
        haystacks = [
            str(instance.get('instance_id') or '').strip().lower(),
            str(instance.get('model') or instance.get('modelName') or '').strip().lower(),
            str(instance.get('request_model') or instance.get('requestModel') or '').strip().lower(),
            str(instance.get('model_path') or '').strip().lower(),
        ]
        return any(
            marker in haystack
            for marker in _TEMPORARY_QUARANTINED_MODEL_MARKERS
            for haystack in haystacks
            if haystack
        )

    def _resolve_responses_target_instance(
        self,
        data: Any,
        *,
        forced_instance_id: Optional[str] = None,
        excluded_instance_ids: Optional[list[str]] = None,
    ) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[str]]:
        normalize_external_identifier = self._hook('normalize_external_identifier')
        lookup_instance = self._hook('lookup_instance')
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        load_running_instances = self._hook('load_running_instances')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        instance_supports_capability = self._hook('instance_supports_capability')
        pick_default_capability_instance = self._hook('pick_default_capability_instance')

        excluded = [
            str(item).strip()
            for item in (excluded_instance_ids or [])
            if str(item).strip()
        ]
        explicit_instance_id = str(forced_instance_id or data.get('instance_id') or '').strip()
        if explicit_instance_id:
            try:
                explicit_instance_id = normalize_external_identifier(explicit_instance_id, field_name='instance_id')
            except ValueError as exc:
                return None, None, None, str(exc)
            if explicit_instance_id in excluded:
                return None, None, None, (
                    f"Instance '{explicit_instance_id}' is excluded for this retry."
                )
            instance = lookup_instance(explicit_instance_id)
            if not instance:
                recovered_instance_id, recovered_instance = self._recover_missing_explicit_target_instance(
                    explicit_instance_id,
                    data,
                )
                if recovered_instance_id and recovered_instance:
                    logging.info(
                        "Recovered stale explicit target '%s' as '%s' (model=%s backend=%s capability=%s).",
                        explicit_instance_id,
                        recovered_instance_id,
                        str(recovered_instance.get('model') or recovered_instance.get('modelName') or '').strip() or None,
                        normalize_backend(recovered_instance.get('backend')),
                        normalize_capability(recovered_instance.get('capability')),
                    )
                    return recovered_instance_id, recovered_instance, None, None
                return None, None, None, f"Instance '{explicit_instance_id}' was not found."
            if self._instance_is_temporarily_quarantined(instance):
                recovery_payload = dict(data) if isinstance(data, dict) else dict(data)
                for key in ('instance_id', 'model', 'modelName', 'request_model', 'requestModel'):
                    recovery_payload.pop(key, None)
                recovered_instance_id, recovered_instance = self._recover_missing_explicit_target_instance(
                    '',
                    recovery_payload,
                )
                if recovered_instance_id and recovered_instance:
                    logging.info(
                        "Recovered quarantined explicit target '%s' as '%s' (model=%s backend=%s capability=%s).",
                        explicit_instance_id,
                        recovered_instance_id,
                        str(recovered_instance.get('model') or recovered_instance.get('modelName') or '').strip() or None,
                        normalize_backend(recovered_instance.get('backend')),
                        normalize_capability(recovered_instance.get('capability')),
                    )
                    return recovered_instance_id, recovered_instance, None, None
                return None, None, None, (
                    f"Instance '{explicit_instance_id}' is unavailable. {_TEMPORARY_QUARANTINE_REASON}"
                )
            return explicit_instance_id, instance, None, None

        alias_selector = str(data.get('alias') or data.get('profile') or '').strip()
        requested_capability = normalize_capability(data.get('capability'))
        selector = alias_selector or requested_capability
        if not selector:
            return None, None, None, "Parameter 'instance_id' is missing."

        resolved_capability = self._resolve_wrapper_capability(selector if alias_selector else requested_capability or '')
        if not resolved_capability:
            if alias_selector:
                return None, None, None, f"Unknown alias/profile '{alias_selector}'."
            return None, None, None, f"Unknown capability '{selector}'."

        instances = merge_instances_with_runtime_status(
            load_running_instances(),
            path=runtime_status_path_getter(),
            refresh=True,
        )
        candidates = [
            entry for entry in instances
            if (
                isinstance(entry, dict)
                and instance_supports_capability(entry, resolved_capability)
                and not self._instance_is_temporarily_quarantined(entry)
            )
        ]
        selectable_candidates = [
            entry for entry in candidates
            if str(entry.get('instance_id') or '').strip() not in excluded
        ] if excluded else candidates
        if candidates and excluded and not selectable_candidates:
            excluded_text = ', '.join(excluded)
            return None, None, resolved_capability, (
                f"No non-excluded running instance found for capability '{resolved_capability}'. "
                f"Excluded instance ids: {excluded_text}."
            )
        selected_instance_id = pick_default_capability_instance(selectable_candidates)
        if not selected_instance_id:
            if alias_selector:
                return None, None, resolved_capability, (
                    f"No running instance found for alias/profile '{alias_selector}' "
                    f"(capability '{resolved_capability}')."
                )
            return None, None, resolved_capability, (
                f"No running instance found for capability '{resolved_capability}'."
            )

        selected_instance = next(
            (entry for entry in selectable_candidates if str(entry.get('instance_id') or '').strip() == selected_instance_id),
            None,
        )
        if not selected_instance:
            return None, None, resolved_capability, (
                f"Instance '{selected_instance_id}' for capability '{resolved_capability}' could not be resolved."
            )
        return selected_instance_id, selected_instance, resolved_capability, None

    def _recover_missing_explicit_target_instance(
        self,
        explicit_instance_id: str,
        data: Any,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')
        merge_instances_with_runtime_status = self._hook('merge_instances_with_runtime_status')
        load_running_instances = self._hook('load_running_instances')
        runtime_status_path_getter = self._hook('runtime_status_path_getter')
        instance_supports_capability = self._hook('instance_supports_capability')
        pick_default_capability_instance = self._hook('pick_default_capability_instance')

        payload = data if isinstance(data, dict) else dict(data)
        model_hints: list[str] = []
        for raw_value in (
            payload.get('model'),
            payload.get('modelName'),
            payload.get('request_model'),
            payload.get('requestModel'),
        ):
            token = str(raw_value or '').strip()
            if token and token not in model_hints:
                model_hints.append(token)

        stale_match = re.match(r'^(?P<model>.+)-(?P<ordinal>\d+)$', str(explicit_instance_id or '').strip())
        stale_model_hint = str((stale_match.group('model') if stale_match else '') or '').strip()
        if stale_model_hint and stale_model_hint not in model_hints:
            model_hints.append(stale_model_hint)

        requested_backend = normalize_backend(payload.get('backend'))
        requested_capability = normalize_capability(payload.get('capability'))
        if not model_hints and not requested_backend and not requested_capability:
            return None, None

        instances = merge_instances_with_runtime_status(
            load_running_instances(),
            path=runtime_status_path_getter(),
            refresh=True,
        )
        candidates: list[dict[str, Any]] = []
        for entry in instances:
            if not isinstance(entry, dict):
                continue
            if self._instance_is_temporarily_quarantined(entry):
                continue
            if requested_backend and normalize_backend(entry.get('backend')) != requested_backend:
                continue
            if requested_capability and not instance_supports_capability(entry, requested_capability):
                continue
            candidate_model = str(entry.get('model') or entry.get('modelName') or '').strip()
            if model_hints and candidate_model not in model_hints:
                continue
            candidates.append(entry)

        selected_instance_id = pick_default_capability_instance(candidates)
        if not selected_instance_id:
            return None, None
        selected_instance = next(
            (entry for entry in candidates if str(entry.get('instance_id') or '').strip() == selected_instance_id),
            None,
        )
        if not selected_instance:
            return None, None
        return selected_instance_id, selected_instance

    def _parse_jsonish_field(self, raw_value: Any) -> Any:
        if raw_value is None:
            return None
        if isinstance(raw_value, (list, dict)):
            return raw_value
        text = str(raw_value or '').strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _selected_reference_message_text_from_history_entry(self, entry: dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        role = str(entry.get('role') or '').strip().lower()
        if role == 'user':
            request_snapshot = entry.get('request_snapshot') if isinstance(entry.get('request_snapshot'), dict) else {}
            prompt_text = str(
                request_snapshot.get('prompt_text')
                or request_snapshot.get('promptText')
                or ''
            ).strip()
            if prompt_text:
                return prompt_text
        content = str(entry.get('content') or '').strip()
        return content or None

    def _resolve_selected_reference_message_from_history(
        self,
        message_id: str,
        *,
        payload_source: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        read_chat_history = self._hook('read_chat_history')
        chat_history_dir_getter = self._hook('chat_history_dir_getter')

        normalized_message_id = str(message_id or '').strip()
        if not normalized_message_id or not isinstance(payload_source, dict):
            return None
        conversation_id = str(
            payload_source.get('conversation_id')
            or payload_source.get('conversationId')
            or ''
        ).strip()
        if not conversation_id:
            return None
        history = read_chat_history(conversation_id, history_dir=chat_history_dir_getter())
        for entry in history.get('messages') or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get('message_id') or entry.get('messageId') or '').strip() != normalized_message_id:
                continue
            content = self._selected_reference_message_text_from_history_entry(entry)
            if not content:
                return None
            return {
                'content': content,
                'message_role': str(entry.get('role') or '').strip().lower() or 'assistant',
                'response_model': str(entry.get('response_model') or entry.get('responseModel') or '').strip() or None,
                'response_instance_id': str(
                    entry.get('response_instance_id')
                    or entry.get('responseInstanceId')
                    or ''
                ).strip() or None,
                'timestamp': str(entry.get('timestamp') or '').strip() or None,
            }
        return None

    @staticmethod
    def _canonical_artifact_path(record: dict[str, Any]) -> str:
        return str(record.get('path') or record.get('source_path') or '').strip()

    @staticmethod
    def _canonical_text_artifact_extension(record: dict[str, Any]) -> str:
        request_payload = (
            record.get('artifact_request')
            if isinstance(record.get('artifact_request'), dict)
            else record.get('text_artifact_request')
            if isinstance(record.get('text_artifact_request'), dict)
            else {}
        )
        path = RequestIntakeRuntimeOwner._canonical_artifact_path(record)
        for value in (
            record.get('text_artifact_extension'),
            record.get('extension'),
            request_payload.get('extension'),
            Path(path).suffix.lstrip('.') if path else '',
            Path(str(record.get('name') or '')).suffix.lstrip('.'),
        ):
            extension = str(value or '').strip().lower().lstrip('.')
            if extension:
                return extension
        return ''

    @staticmethod
    def _normalized_text_artifact_source_name(value: Any) -> str:
        token = str(value or '').strip()
        if not token:
            return ''
        name = Path(token).name
        if Path(name).suffix:
            name = Path(name).stem
        return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

    @classmethod
    def _canonical_artifact_matches_named_text_target(
        cls,
        record: dict[str, Any],
        filename: str,
    ) -> bool:
        artifact_kind = str(record.get('type') or record.get('kind') or '').strip().lower()
        if artifact_kind not in _CURRENT_PREDECESSOR_TEXT_ARTIFACT_TYPES:
            return False
        expected_extension = Path(filename).suffix.lstrip('.').lower()
        if not expected_extension or cls._canonical_text_artifact_extension(record) != expected_extension:
            return False
        expected_source_name = cls._normalized_text_artifact_source_name(filename)
        request_payload = (
            record.get('artifact_request')
            if isinstance(record.get('artifact_request'), dict)
            else record.get('text_artifact_request')
            if isinstance(record.get('text_artifact_request'), dict)
            else {}
        )
        explicit_names = {
            cls._normalized_text_artifact_source_name(value)
            for value in (
                record.get('text_artifact_source_name'),
                record.get('source_name'),
                request_payload.get('source_name'),
                record.get('name'),
            )
            if str(value or '').strip()
        }
        if explicit_names:
            return any(
                name == expected_source_name or name.endswith(f'-{expected_source_name}')
                for name in explicit_names
            )
        path_name = cls._normalized_text_artifact_source_name(cls._canonical_artifact_path(record))
        return bool(
            path_name == expected_source_name
            or path_name.endswith(f'-{expected_source_name}')
        )

    def _canonical_named_text_edit_artifacts(
        self,
        canonical_artifacts: list[dict[str, Any]],
        filenames: list[str],
    ) -> Optional[list[tuple[str, dict[str, Any]]]]:
        resolve_saved_path = self.hooks.get('resolve_saved_downloadable_artifact_path')
        concrete_artifacts: list[dict[str, Any]] = []
        for item in canonical_artifacts:
            path = self._canonical_artifact_path(item)
            if not path:
                continue
            resolved_path: Any = path
            if callable(resolve_saved_path):
                try:
                    resolved_path = resolve_saved_path(path)
                except Exception:  # noqa: BLE001 - canonical source resolution fails closed
                    resolved_path = None
            if not resolved_path:
                continue
            record = dict(item)
            record['path'] = str(resolved_path)
            concrete_artifacts.append(record)

        matched: list[tuple[str, dict[str, Any]]] = []
        matched_paths: set[str] = set()
        for filename in filenames:
            candidates_by_path: dict[str, dict[str, Any]] = {}
            for artifact in concrete_artifacts:
                if not self._canonical_artifact_matches_named_text_target(artifact, filename):
                    continue
                path = self._canonical_artifact_path(artifact)
                if path:
                    candidates_by_path.setdefault(path, artifact)
            if len(candidates_by_path) != 1:
                return None
            path, artifact = next(iter(candidates_by_path.items()))
            if path in matched_paths:
                return None
            matched_paths.add(path)
            matched.append((filename, artifact))
        return matched

    def _canonical_carried_predecessor_dependencies(
        self,
        canonical_artifacts: list[dict[str, Any]],
        *,
        excluded_paths: set[str],
        source_response_id: str,
    ) -> list[dict[str, Any]]:
        """Return immutable public predecessor files needed by a follow-up bundle."""

        resolve_saved_path = self.hooks.get(
            'resolve_saved_downloadable_artifact_path'
        )
        normalized_excluded_paths = {
            str(Path(path).expanduser().resolve(strict=False))
            for path in excluded_paths
            if str(path or '').strip()
        }
        carried_reversed: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_named_text_identities: set[tuple[str, str]] = set()
        for item in reversed(canonical_artifacts):
            path = self._canonical_artifact_path(item)
            if not path:
                continue
            resolved_path: Any = path
            if callable(resolve_saved_path):
                try:
                    resolved_path = resolve_saved_path(path)
                except Exception:  # noqa: BLE001 - canonical dependency resolution fails closed
                    resolved_path = None
            if not resolved_path:
                continue
            try:
                normalized_path = str(
                    Path(str(resolved_path)).expanduser().resolve(strict=False)
                )
            except (OSError, RuntimeError, ValueError):
                continue
            if (
                normalized_path in normalized_excluded_paths
                or normalized_path in seen_paths
            ):
                continue
            artifact_type = str(
                item.get('type') or item.get('kind') or ''
            ).strip().lower()
            if artifact_type not in {
                'audio',
                'document',
                'file',
                'image',
                'text',
            }:
                continue
            extension = str(
                item.get('text_artifact_extension')
                or Path(normalized_path).suffix.lstrip('.')
                or ''
            ).strip().lower()
            source_name = str(
                item.get('text_artifact_source_name')
                or item.get('name')
                or Path(normalized_path).stem
                or ''
            ).strip().lower()
            if artifact_type in {'document', 'text'} and extension and source_name:
                identity = (extension, source_name)
                if identity in seen_named_text_identities:
                    continue
                seen_named_text_identities.add(identity)
            seen_paths.add(normalized_path)
            carried_reversed.append(
                {
                    key: value
                    for key, value in {
                        'type': artifact_type,
                        'kind': artifact_type,
                        'artifact_ref': str(
                            item.get('artifact_ref') or item.get('ref') or ''
                        ).strip()
                        or None,
                        'ref': str(
                            item.get('artifact_ref') or item.get('ref') or ''
                        ).strip()
                        or None,
                        'path': normalized_path,
                        'source_path': normalized_path,
                        'name': str(item.get('name') or '').strip() or None,
                        'mime_type': str(item.get('mime_type') or '').strip()
                        or None,
                        'source_response_id': source_response_id,
                        'origin': 'canonical_predecessor_bundle_dependency',
                        'carried_public_dependency': True,
                    }.items()
                    if value not in (None, '')
                }
            )
            if len(carried_reversed) >= 64:
                break
        return list(reversed(carried_reversed))

    def promote_current_predecessor_context(self, data: Any) -> dict[str, Any]:
        """Bind a selected same-conversation predecessor to canonical repair evidence."""

        payload = dict(data) if isinstance(data, dict) else dict(data or {})
        prompt = str(payload.get('prompt') or '').strip()
        named_text_edit_targets = _explicit_current_turn_named_text_edit_targets(prompt)
        legacy_image_site_recovery = bool(
            _CURRENT_PREDECESSOR_IMAGE_MATERIALIZATION_RE.search(prompt)
            and _CURRENT_PREDECESSOR_PAGE_BINDING_RE.search(prompt)
        )
        if not named_text_edit_targets and not legacy_image_site_recovery:
            return payload

        raw_references = self._parse_jsonish_field(
            payload.get('reference_artifacts')
            if payload.get('reference_artifacts') is not None
            else (
                payload.get('selected_reference_artifact')
                if payload.get('selected_reference_artifact') is not None
                else payload.get('selected_reference_artifacts')
            )
        )
        reference_items = raw_references if isinstance(raw_references, list) else [raw_references]
        selected_message = next(
            (
                dict(item)
                for item in reversed(reference_items)
                if isinstance(item, dict)
                and str(item.get('type') or '').strip().lower() == 'message'
            ),
            None,
        )
        message_id = str(
            (selected_message or {}).get('message_id')
            or (selected_message or {}).get('messageId')
            or ''
        ).strip()
        conversation_id = str(
            payload.get('conversation_id')
            or payload.get('conversationId')
            or ''
        ).strip()
        if not message_id or not conversation_id:
            return payload

        read_chat_history = self._hook('read_chat_history')
        chat_history_dir_getter = self._hook('chat_history_dir_getter')
        history = read_chat_history(conversation_id, history_dir=chat_history_dir_getter())
        history_entry = next(
            (
                dict(item)
                for item in history.get('messages') or []
                if isinstance(item, dict)
                and str(item.get('message_id') or item.get('messageId') or '').strip()
                == message_id
                and str(item.get('role') or '').strip().lower() == 'assistant'
            ),
            None,
        )
        source_response_id = str((history_entry or {}).get('response_id') or '').strip()
        get_response_lookup_record = self.hooks.get('get_response_lookup_record')
        if not source_response_id or not callable(get_response_lookup_record):
            return payload
        try:
            predecessor_record = get_response_lookup_record(source_response_id)
        except Exception:  # noqa: BLE001 - canonical predecessor lookup must fail closed
            return payload
        if not isinstance(predecessor_record, dict):
            return payload
        predecessor_payload = (
            predecessor_record.get('response_payload')
            if isinstance(predecessor_record.get('response_payload'), dict)
            else {}
        )
        predecessor_frame = (
            predecessor_payload.get('response_frame')
            if isinstance(predecessor_payload.get('response_frame'), dict)
            else {}
        )
        predecessor_request = (
            predecessor_frame.get('request')
            if isinstance(predecessor_frame.get('request'), dict)
            else {}
        )
        predecessor_conversation_id = str(
            predecessor_request.get('conversation_id')
            or predecessor_request.get('conversationId')
            or ''
        ).strip()
        history_message_id = str(
            (history_entry or {}).get('message_id')
            or (history_entry or {}).get('messageId')
            or ''
        ).strip()
        predecessor_message_id = str(
            predecessor_record.get('message_id')
            or predecessor_payload.get('message_id')
            or history_message_id
            or ''
        ).strip()
        predecessor_lifecycle = str(
            predecessor_record.get('lifecycle_state')
            or predecessor_payload.get('lifecycle_state')
            or predecessor_payload.get('status')
            or ''
        ).strip().lower()
        if (
            str(predecessor_record.get('id') or '').strip() != source_response_id
            or (
                predecessor_conversation_id
                and predecessor_conversation_id != conversation_id
            )
            or not predecessor_message_id
            or predecessor_lifecycle not in {'completed', 'repair_needed'}
        ):
            return payload

        predecessor_conversation_authority = (
            'response_frame_request'
            if predecessor_conversation_id
            else 'current_conversation_history_mapping'
        )
        predecessor_message_authority = (
            'canonical_response_record'
            if str(
                predecessor_record.get('message_id')
                or predecessor_payload.get('message_id')
                or ''
            ).strip()
            else 'current_conversation_history_mapping'
        )

        canonical_artifacts = [
            dict(item)
            for item in predecessor_payload.get('artifacts') or []
            if isinstance(item, dict)
            and str(item.get('path') or item.get('source_path') or '').strip()
        ]
        text_artifacts = [
            item
            for item in canonical_artifacts
            if str(item.get('type') or item.get('kind') or '').strip().lower()
            in _CURRENT_PREDECESSOR_TEXT_ARTIFACT_TYPES
        ]
        predecessor_text = str(predecessor_payload.get('output_text') or '').strip()
        if not predecessor_text:
            return payload

        promotion_mode = 'legacy_image_site_recovery'
        batch_prompts: list[str] = []
        named_text_matches: list[tuple[str, dict[str, Any]]] = []
        artifacts_for_promotion = canonical_artifacts
        if named_text_edit_targets:
            matched = self._canonical_named_text_edit_artifacts(
                canonical_artifacts,
                named_text_edit_targets,
            )
            if not matched:
                return payload
            named_text_matches = matched
            artifacts_for_promotion = [dict(artifact) for _filename, artifact in matched]
            text_artifacts = artifacts_for_promotion
            promotion_mode = 'named_text_edit'
        else:
            batch_prompts = extract_canonical_predecessor_image_prompts(predecessor_payload)
            extract_batch_image_prompts = self.hooks.get('extract_batch_image_prompts')
            if not batch_prompts and callable(extract_batch_image_prompts):
                predecessor_late_fill = (
                    predecessor_payload.get('late_fill')
                    if isinstance(predecessor_payload.get('late_fill'), dict)
                    else {}
                )
                prepared_content = str(
                    predecessor_late_fill.get('content_payload')
                    or predecessor_payload.get('content_payload')
                    or ''
                ).strip()
                if prepared_content:
                    batch_prompts = extract_batch_image_prompts(
                        prepared_content,
                        expected_count=0,
                        allow_plain_alpha_sequence=False,
                    )
            if not batch_prompts or not text_artifacts:
                return payload

        message_reference = {
            'type': 'message',
            'message_role': 'assistant',
            'message_id': predecessor_message_id,
            'source_message_id': predecessor_message_id,
            'source_response_id': source_response_id,
            'artifact_ref': predecessor_message_id,
            'content': predecessor_text,
            'origin': 'conversation_reference',
        }
        artifact_references: list[dict[str, Any]] = []
        matched_text_artifacts: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts_for_promotion, start=1):
            reference = dict(artifact)
            reference['source_message_id'] = predecessor_message_id
            reference['source_response_id'] = source_response_id
            if promotion_mode == 'named_text_edit':
                filename = named_text_matches[index - 1][0]
                extension = Path(filename).suffix.lstrip('.').lower()
                source_name = Path(filename).stem
                reference['type'] = 'text'
                reference['kind'] = 'text'
                reference['name'] = filename
                reference['text_artifact_source_name'] = source_name
                reference['text_artifact_extension'] = extension
                reference['artifact_request'] = {
                    'source_name': source_name,
                    'extension': extension,
                }
                reference['origin'] = _CURRENT_PREDECESSOR_NAMED_TEXT_EDIT_ORIGIN
                reference['batch_index'] = index
                matched_text_artifacts.append(
                    {
                        'filename': filename,
                        'artifact_ref': str(
                            reference.get('artifact_ref')
                            or reference.get('ref')
                            or ''
                        ).strip(),
                        'path': self._canonical_artifact_path(reference),
                    }
                )
            else:
                reference.setdefault('origin', 'conversation_reference')
            artifact_references.append(reference)
        promoted_references = [message_reference, *artifact_references]
        payload['reference_artifacts'] = promoted_references
        payload['selected_reference_artifacts'] = promoted_references
        payload.pop('selected_reference_artifact', None)
        payload.pop('selectedReferenceArtifact', None)

        carrier = {
            'role': 'assistant',
            'message_id': predecessor_message_id,
            'response_id': source_response_id,
            'content': predecessor_text,
            'artifacts': artifact_references,
        }
        ghost_messages = [
            dict(item)
            for item in payload.get('ghost_messages') or []
            if isinstance(item, dict)
            and str(item.get('response_id') or '').strip() != source_response_id
        ]
        ghost_messages.append(carrier)
        payload['ghost_messages'] = ghost_messages
        payload['current_predecessor_context'] = {
            'kind': 'ollmo.current_predecessor_context',
            'status': 'authorized',
            'authorization': 'canonical_same_conversation_predecessor',
            'promotion_mode': promotion_mode,
            'current_prompt_authority': 'payload.prompt',
            'source_response_id': source_response_id,
            'source_message_id': predecessor_message_id,
            'selected_message_id': message_id,
            'predecessor_conversation_authority': predecessor_conversation_authority,
            'predecessor_message_authority': predecessor_message_authority,
            'batch_prompts': batch_prompts,
            'text_artifact_refs': [
                str(item.get('artifact_ref') or item.get('ref') or '').strip()
                for item in text_artifacts
                if str(item.get('artifact_ref') or item.get('ref') or '').strip()
            ],
        }
        if matched_text_artifacts:
            payload['current_predecessor_context']['requested_text_artifact_names'] = list(
                named_text_edit_targets
            )
            payload['current_predecessor_context']['matched_text_artifacts'] = matched_text_artifacts
            carried_dependencies = self._canonical_carried_predecessor_dependencies(
                canonical_artifacts,
                excluded_paths={
                    str(item.get('path') or '').strip()
                    for item in matched_text_artifacts
                    if str(item.get('path') or '').strip()
                },
                source_response_id=source_response_id,
            )
            if carried_dependencies:
                payload['current_predecessor_context'][
                    'carried_public_dependencies'
                ] = carried_dependencies
        return payload

    def _sanitize_selected_reference_artifact(
        self,
        raw_value: Any,
        *,
        payload_source: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        resolve_saved_downloadable_artifact_path = self._hook('resolve_saved_downloadable_artifact_path')
        sanitize_artifact_record = self._hook('sanitize_artifact_record')
        get_cached_generated_image_state = self._hook('get_cached_generated_image_state')

        if not isinstance(raw_value, dict):
            return None
        artifact_type = str(raw_value.get('type') or '').strip().lower()
        if artifact_type == 'message':
            message_id = str(raw_value.get('message_id') or raw_value.get('messageId') or '').strip() or None
            content = str(
                raw_value.get('content')
                or raw_value.get('text')
                or raw_value.get('prompt')
                or ''
            ).strip()
            resolved_message = None
            if not content and message_id:
                resolved_message = self._resolve_selected_reference_message_from_history(
                    message_id,
                    payload_source=payload_source,
                )
                content = str((resolved_message or {}).get('content') or '').strip()
            if not content:
                return None
            role = str(raw_value.get('message_role') or raw_value.get('role') or 'assistant').strip().lower() or 'assistant'
            if resolved_message and not str(raw_value.get('message_role') or raw_value.get('role') or '').strip():
                role = str((resolved_message or {}).get('message_role') or role).strip().lower() or role
            if role not in {'user', 'assistant', 'system'}:
                role = 'assistant'
            payload = {
                'type': 'message',
                'content': content[:12000],
                'message_role': role,
                'source_message_id': message_id,
                'artifact_ref': str(raw_value.get('artifact_ref') or raw_value.get('artifactRef') or raw_value.get('ref') or '').strip() or message_id or None,
                'artifact_id': str(raw_value.get('artifact_id') or raw_value.get('artifactId') or '').strip() or None,
                'origin': 'conversation_reference',
            }
            for source_key, target_key in (
                ('message_id', 'message_id'),
                ('messageId', 'message_id'),
                ('source_response_id', 'source_response_id'),
                ('sourceResponseId', 'source_response_id'),
                ('response_model', 'response_model'),
                ('responseModel', 'response_model'),
                ('response_instance_id', 'response_instance_id'),
                ('responseInstanceId', 'response_instance_id'),
                ('timestamp', 'timestamp'),
            ):
                value = str(raw_value.get(source_key) or '').strip()
                if not value and resolved_message:
                    value = str((resolved_message or {}).get(target_key) or '').strip()
                if value:
                    payload[target_key] = value
            return sanitize_artifact_record(
                payload,
                default_kind='message',
                default_origin='conversation_reference',
                include_content=True,
            )
        raw_path = str(raw_value.get('path') or raw_value.get('source_path') or '').strip()
        if artifact_type == 'file':
            artifact_type = 'text'
        if artifact_type not in {'image', 'audio', 'text', 'document'} or not raw_path:
            return None
        resolved = resolve_saved_downloadable_artifact_path(raw_path)
        if not resolved:
            return None
        payload: dict[str, Any] = {
            'type': artifact_type,
            'path': str(resolved),
            'artifact_ref': str(raw_value.get('artifact_ref') or raw_value.get('artifactRef') or raw_value.get('ref') or '').strip() or str(resolved),
            'artifact_id': str(raw_value.get('artifact_id') or raw_value.get('artifactId') or '').strip() or None,
            'origin': 'conversation_reference',
        }
        message_id = str(raw_value.get('message_id') or raw_value.get('messageId') or '').strip()
        if message_id:
            payload['source_message_id'] = message_id
        source_response_id = str(
            raw_value.get('source_response_id')
            or raw_value.get('sourceResponseId')
            or ''
        ).strip()
        if source_response_id:
            payload['source_response_id'] = source_response_id
        name = str(raw_value.get('name') or '').strip()
        if name:
            payload['name'] = name
        kind = str(raw_value.get('kind') or '').strip().lower()
        if kind:
            payload['kind'] = kind
        origin = str(raw_value.get('origin') or '').strip().lower()
        if origin:
            payload['origin'] = origin
        source_path = str(raw_value.get('source_path') or raw_value.get('sourcePath') or '').strip()
        if source_path:
            payload['source_path'] = source_path
        mime_type = str(raw_value.get('mime_type') or raw_value.get('mimeType') or '').strip()
        if mime_type:
            payload['mime_type'] = mime_type
        prompt = str(raw_value.get('prompt') or '').strip()
        if prompt:
            payload['prompt'] = prompt
        seed = raw_value.get('seed')
        if isinstance(seed, (int, float, str)):
            try:
                parsed_seed = int(str(seed).strip())
            except (TypeError, ValueError):
                parsed_seed = None
            if parsed_seed is not None and parsed_seed >= 0:
                payload['seed'] = parsed_seed
        batch_index = raw_value.get('batch_index')
        if isinstance(batch_index, (int, float, str)):
            try:
                parsed_batch_index = int(str(batch_index).strip())
            except (TypeError, ValueError):
                parsed_batch_index = None
            if parsed_batch_index is not None and parsed_batch_index > 0:
                payload['batch_index'] = parsed_batch_index
        image_state = raw_value.get('image_state')
        if artifact_type == 'image':
            if isinstance(image_state, dict) and image_state:
                payload['image_state'] = image_state
            else:
                cached_image_state = get_cached_generated_image_state(str(resolved))
                if cached_image_state:
                    payload['image_state'] = cached_image_state
        return sanitize_artifact_record(
            payload,
            default_origin='conversation_reference',
        )

    def _sanitize_selected_reference_artifacts(
        self,
        raw_value: Any,
        *,
        payload_source: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        raw_items = raw_value if isinstance(raw_value, list) else [raw_value]
        message_reference: Optional[dict[str, Any]] = None
        artifact_reference: Optional[dict[str, Any]] = None
        bounded_artifact_references: list[dict[str, Any]] = []
        context = (
            payload_source.get('current_predecessor_context')
            if isinstance(payload_source, dict)
            and isinstance(payload_source.get('current_predecessor_context'), dict)
            else {}
        )
        bounded_entries = (
            [dict(item) for item in context.get('matched_text_artifacts') or [] if isinstance(item, dict)]
            if (
                context.get('status') == 'authorized'
                and context.get('authorization') == 'canonical_same_conversation_predecessor'
                and context.get('promotion_mode') == 'named_text_edit'
            )
            else []
        )
        expected_source_response_id = str(context.get('source_response_id') or '').strip()

        def reference_matches_entry(reference: dict[str, Any], entry: dict[str, Any]) -> bool:
            entry_ref = str(entry.get('artifact_ref') or '').strip()
            entry_path = str(entry.get('path') or '').strip()
            reference_ref = str(reference.get('artifact_ref') or reference.get('ref') or '').strip()
            reference_path = str(reference.get('path') or reference.get('source_path') or '').strip()
            return bool(
                (entry_ref and reference_ref == entry_ref)
                or (entry_path and reference_path == entry_path)
            )

        for item in raw_items:
            normalized = self._sanitize_selected_reference_artifact(item, payload_source=payload_source)
            if not normalized:
                continue
            if str(normalized.get('type') or '').strip().lower() == 'message':
                message_reference = normalized
                continue
            if bounded_entries:
                source_response_id = str(normalized.get('source_response_id') or '').strip()
                if (
                    expected_source_response_id
                    and source_response_id != expected_source_response_id
                ):
                    continue
                if any(reference_matches_entry(normalized, entry) for entry in bounded_entries):
                    bounded_artifact_references.append(normalized)
                continue
            if (
                payload_source is None
                and str(normalized.get('origin') or '').strip().lower()
                == _CURRENT_PREDECESSOR_NAMED_TEXT_EDIT_ORIGIN
            ):
                bounded_artifact_references.append(normalized)
                continue
            artifact_reference = normalized

        if bounded_entries:
            ordered_bounded: list[dict[str, Any]] = []
            seen_identities: set[tuple[str, str]] = set()
            for entry in bounded_entries:
                candidates = [
                    reference
                    for reference in bounded_artifact_references
                    if reference_matches_entry(reference, entry)
                ]
                unique_candidates: list[dict[str, Any]] = []
                for candidate in candidates:
                    identity = (
                        str(candidate.get('artifact_ref') or candidate.get('ref') or '').strip(),
                        str(candidate.get('path') or candidate.get('source_path') or '').strip(),
                    )
                    if identity in seen_identities:
                        continue
                    seen_identities.add(identity)
                    unique_candidates.append(candidate)
                if len(unique_candidates) != 1:
                    return [message_reference] if isinstance(message_reference, dict) else []
                ordered_bounded.append(unique_candidates[0])
            return [
                item
                for item in (message_reference, *ordered_bounded)
                if isinstance(item, dict)
            ]

        if payload_source is None and bounded_artifact_references:
            unique_bounded: list[dict[str, Any]] = []
            seen_bounded: set[tuple[str, str]] = set()
            for reference in bounded_artifact_references:
                identity = (
                    str(reference.get('artifact_ref') or reference.get('ref') or '').strip(),
                    str(reference.get('path') or reference.get('source_path') or '').strip(),
                )
                if identity in seen_bounded:
                    continue
                seen_bounded.add(identity)
                unique_bounded.append(reference)
            return [
                item
                for item in (message_reference, *unique_bounded)
                if isinstance(item, dict)
            ]
        return [item for item in (message_reference, artifact_reference) if isinstance(item, dict)]

    def _extract_selected_reference_artifacts(self, data: Any) -> list[dict[str, Any]]:
        payload_source = data if isinstance(data, dict) else dict(data)
        raw_payload = self._parse_jsonish_field(
            payload_source.get('reference_artifacts')
            if payload_source.get('reference_artifacts') is not None
            else (
                payload_source.get('selected_reference_artifact')
                if payload_source.get('selected_reference_artifact') is not None
                else payload_source.get('selected_reference_artifacts')
            )
        )
        return self._sanitize_selected_reference_artifacts(raw_payload, payload_source=payload_source)

    def _extract_selected_reference_artifact(self, data: Any) -> Optional[dict[str, Any]]:
        selected_references = self._extract_selected_reference_artifacts(data)
        return selected_references[0] if selected_references else None

    def _inject_selected_reference_message(
        self,
        messages: list[dict[str, Any]],
        selected_reference_artifact: Any,
    ) -> list[dict[str, Any]]:
        sanitize_ghost_messages = self._hook('sanitize_ghost_messages')

        selected_references = self._sanitize_selected_reference_artifacts(selected_reference_artifact)
        if not selected_references:
            return messages
        injected = list(messages or [])
        for selected_reference in selected_references:
            reference_type = str(selected_reference.get('type') or '').strip().lower()
            if reference_type == 'message':
                content = str(selected_reference.get('content') or '').strip()
                if not content:
                    continue
                message_role = str(selected_reference.get('message_role') or 'assistant').strip().lower() or 'assistant'
                if message_role not in {'user', 'assistant', 'system'}:
                    message_role = 'assistant'
                injected.append(
                    {
                        'role': 'system',
                        'content': (
                            'Selected prior message reference for this conversation turn. '
                            'Treat it as bounded reference context only; the current user message remains the live instruction. '
                            'Do not infer new tasks from this reference unless the current turn explicitly asks.\n\n'
                            f'[{message_role}]\n{content}'
                        ),
                        'timestamp': selected_reference.get('timestamp'),
                        'response_model': selected_reference.get('response_model'),
                        'response_instance_id': selected_reference.get('response_instance_id'),
                        'selected_reference': True,
                    }
                )
                continue
            injected.append(
                {
                    'role': 'assistant',
                    'content': 'Selected reference artifact.',
                    'artifacts': [selected_reference],
                    'selected_reference': True,
                }
            )
        return sanitize_ghost_messages(injected)

    def _extract_ghost_route_messages(self, data: Any, *, include_selected_reference: bool = True) -> list[dict[str, Any]]:
        extract_responses_messages = self._hook('extract_responses_messages')
        sanitize_ghost_messages = self._hook('sanitize_ghost_messages')

        payload_source = data if isinstance(data, dict) else dict(data)
        payload = self._parse_jsonish_field(payload_source.get('ghost_messages'))
        if isinstance(payload, list):
            base_messages = sanitize_ghost_messages(payload)
            if include_selected_reference:
                return self._inject_selected_reference_message(
                    base_messages,
                    self._extract_selected_reference_artifacts(payload_source),
                )
            return base_messages

        payload = self._parse_jsonish_field(payload_source.get('ghost_messages_json'))
        if isinstance(payload, list):
            base_messages = sanitize_ghost_messages(payload)
            if include_selected_reference:
                return self._inject_selected_reference_message(
                    base_messages,
                    self._extract_selected_reference_artifacts(payload_source),
                )
            return base_messages

        input_payload = self._parse_jsonish_field(payload_source.get('input'))
        if isinstance(input_payload, list):
            fallback_messages = extract_responses_messages({'input': input_payload})
            base_messages = sanitize_ghost_messages(fallback_messages)
            if include_selected_reference:
                return self._inject_selected_reference_message(
                    base_messages,
                    self._extract_selected_reference_artifacts(payload_source),
                )
            return base_messages

        fallback_messages = extract_responses_messages(payload_source)
        base_messages = sanitize_ghost_messages(fallback_messages)
        if include_selected_reference:
            return self._inject_selected_reference_message(
                base_messages,
                self._extract_selected_reference_artifacts(payload_source),
            )
        return base_messages

    def _extract_ghost_preview_route(self, data: Any) -> Optional[dict[str, Any]]:
        payload = self._parse_jsonish_field(data.get('ghost_preview'))
        if not isinstance(payload, dict):
            return None
        return payload

    def _normalize_ghost_preference_target(
        self,
        raw_value: Any,
        *,
        capability: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        normalize_backend = self._hook('normalize_backend')
        normalize_capability = self._hook('normalize_capability')

        if not isinstance(raw_value, dict):
            return None
        model = str(raw_value.get('model') or '').strip()
        backend = normalize_backend(raw_value.get('backend'))
        normalized_capability = normalize_capability(raw_value.get('capability') or capability)
        normalized_role = str(raw_value.get('role') or role or '').strip().lower() or None
        if not model and not backend:
            return None
        payload: dict[str, Any] = {}
        if model:
            payload['model'] = model
        if backend:
            payload['backend'] = backend
        if normalized_capability:
            payload['capability'] = normalized_capability
        if normalized_role:
            payload['role'] = normalized_role
        return payload or None

    def _coerce_ghost_preferences_payload(self, raw_value: Any) -> dict[str, Any]:
        parse_bool = self._hook('parse_bool')
        payload = raw_value if isinstance(raw_value, dict) else {}
        primary_target = self._normalize_ghost_preference_target(
            payload.get('primary_target') if 'primary_target' in payload else payload.get('primaryTarget')
        )
        fallback_target = self._normalize_ghost_preference_target(
            payload.get('fallback_target') if 'fallback_target' in payload else payload.get('fallbackTarget')
        )
        embedding_helper = self._normalize_ghost_preference_target(
            payload.get('embedding_helper') if 'embedding_helper' in payload else payload.get('embeddingHelper'),
            capability=CAPABILITY_EMBEDDING,
            role='embedding_helper',
        )
        external_targets_source = (
            payload.get('external_targets')
            if isinstance(payload.get('external_targets'), dict)
            else payload.get('externalTargets')
            if isinstance(payload.get('externalTargets'), dict)
            else {}
        )
        codex_source = (
            external_targets_source.get('codex')
            if isinstance(external_targets_source.get('codex'), dict)
            else {}
        )
        codex_enabled = parse_bool(
            codex_source.get('enabled')
            if 'enabled' in codex_source
            else payload.get('codex_enabled')
            if 'codex_enabled' in payload
            else payload.get('codexEnabled'),
            default=False,
        )
        codex_data_scope = str(
            codex_source.get('data_scope')
            or codex_source.get('dataScope')
            or ''
        ).strip().lower()
        codex_files_enabled = parse_bool(
            codex_source.get('files_enabled')
            if 'files_enabled' in codex_source
            else codex_source.get('filesEnabled'),
            default=False,
        )
        if codex_data_scope == 'selected_files_v1':
            codex_files_enabled = True
        primary_mode = str(payload.get('primary_mode') or payload.get('primaryMode') or '').strip().lower()
        if primary_mode not in {'auto', 'prefer', 'lock'}:
            lock_primary = payload.get('lockPrimary')
            if isinstance(lock_primary, bool) and primary_target:
                primary_mode = 'lock' if lock_primary else 'prefer'
            elif primary_target or fallback_target:
                primary_mode = 'prefer'
            else:
                primary_mode = 'auto'
        if (
            primary_mode == 'auto'
            and not primary_target
            and not fallback_target
            and not embedding_helper
            and not codex_enabled
        ):
            return {}
        normalized = {
            'primary_mode': primary_mode,
            'primary_target': primary_target,
            'fallback_target': fallback_target,
            'embedding_helper': embedding_helper,
        }
        if codex_enabled:
            normalized['external_targets'] = {
                'codex': {
                    'enabled': True,
                    **(
                        {
                            'files_enabled': True,
                            'data_scope': 'selected_files_v1',
                        }
                        if codex_files_enabled
                        else {}
                    ),
                }
            }
        return normalized

    def _extract_ghost_preferences(self, data: Any) -> dict[str, Any]:
        payload = self._parse_jsonish_field(data.get('ghost_preferences'))
        return self._coerce_ghost_preferences_payload(payload)
