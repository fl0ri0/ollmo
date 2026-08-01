"""Infer-support owners for Ollmo."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class InferSupportRuntimeOwner:
    hooks: dict[str, Any]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def input_artifact_type_from_kind(self, file_kind: str) -> str:
        normalized_kind = str(file_kind or '').strip().lower()
        if normalized_kind == 'image':
            return 'image'
        if normalized_kind == 'audio':
            return 'audio'
        if normalized_kind == 'text':
            return 'text'
        if normalized_kind == 'pdf':
            return 'document'
        return 'binary'

    def build_input_artifact_payload(
        self,
        saved_path: str,
        *,
        file_name: str = '',
        file_kind: str = '',
        origin: str = '',
        source_path: str = '',
    ) -> Optional[dict[str, Any]]:
        normalized_saved_path = str(saved_path or '').strip()
        normalized_kind = str(file_kind or '').strip().lower() or 'binary'
        if not normalized_saved_path:
            return None
        payload: dict[str, Any] = {
            'type': self.input_artifact_type_from_kind(normalized_kind),
            'path': normalized_saved_path,
            'name': str(file_name or '').strip() or Path(normalized_saved_path).name,
            'kind': normalized_kind,
        }
        normalized_origin = str(origin or '').strip().lower()
        if normalized_origin:
            payload['origin'] = normalized_origin
        normalized_source_path = str(source_path or '').strip()
        if normalized_source_path:
            payload['source_path'] = normalized_source_path
        mime_type, _ = mimetypes.guess_type(normalized_saved_path)
        if mime_type:
            payload['mime_type'] = mime_type
        return payload

    def persist_request_input_artifacts(
        self,
        *,
        temp_path: Optional[Path] = None,
        file_name: str = '',
        file_kind: str = '',
        upload=None,
        source_path: str = '',
    ) -> list[dict[str, Any]]:
        file_kind_from_name = self._hook('file_kind_from_name')
        persist_input_file_locally = self._hook('persist_input_file_locally')

        if not temp_path or not file_name:
            return []
        normalized_kind = str(file_kind or '').strip().lower() or file_kind_from_name(file_name)
        saved_path = persist_input_file_locally(
            temp_path,
            source_name=file_name,
            file_kind=normalized_kind,
        )
        if not saved_path:
            return []
        origin = 'upload' if upload and getattr(upload, 'filename', None) else 'local_path'
        artifact = self.build_input_artifact_payload(
            saved_path,
            file_name=file_name,
            file_kind=normalized_kind,
            origin=origin,
            source_path=source_path,
        )
        return [artifact] if artifact else []

    def truncate_for_history(self, text: str, max_chars: int = 120_000) -> str:
        truncate_for_history_impl = self._hook('truncate_for_history')
        return truncate_for_history_impl(text, max_chars=max_chars)

    def read_infer_history(self, limit: int = 200) -> list[dict[str, Any]]:
        is_testing = bool(self._hook('app_testing')())
        if is_testing:
            return []
        read_infer_history_impl = self._hook('read_infer_history')
        infer_history_path_getter = self._hook('infer_history_path_getter')
        return read_infer_history_impl(infer_history_path_getter(), limit=limit)

    def append_infer_history(self, entry: dict[str, Any]) -> None:
        append_infer_history_impl = self._hook('append_infer_history')
        infer_history_path_getter = self._hook('infer_history_path_getter')
        append_infer_history_impl(entry, history_path=infer_history_path_getter())

    def find_cached_pdf_insight(
        self,
        *,
        file_sha256: str,
        model_name: str,
        backend: str,
        capability: str,
        prompt: str,
        read_infer_history_fn: Callable[..., list[dict[str, Any]]],
        looks_like_ocr_prompt_echo_fn: Callable[..., bool],
    ) -> Optional[dict[str, Any]]:
        if not file_sha256:
            return None
        history = read_infer_history_fn(limit=1000)
        for entry in history:
            if entry.get('status') != 'ok':
                continue
            if entry.get('file_kind') != 'pdf':
                continue
            if entry.get('file_sha256') != file_sha256:
                continue
            if str(entry.get('model') or '') != model_name:
                continue
            if str(entry.get('backend') or '') != backend:
                continue
            if str(entry.get('capability') or '') != capability:
                continue
            if str(entry.get('prompt') or '') != prompt:
                continue
            content_value = str(entry.get('content') or '').strip()
            if not content_value:
                continue
            if looks_like_ocr_prompt_echo_fn(content_value, user_hint=prompt):
                continue
            if str(entry.get('mode') or '').strip() == 'vision_analysis_pdf_scan':
                if int(entry.get('pdf_processed_pages') or 0) == 0:
                    continue
            return entry
        return None

    def log_pdf_infer_event(
        self,
        *,
        instance_id: str,
        model_name: str,
        backend: str,
        capability: str,
        prompt: str,
        file_name: str,
        file_sha256: str,
        status: str,
        mode: Optional[str] = None,
        content: Optional[str] = None,
        error: Optional[str] = None,
        warnings: Optional[list[str]] = None,
        pdf_source: Optional[str] = None,
        pdf_total_pages: Optional[int] = None,
        pdf_processed_pages: Optional[int] = None,
        artifact_path: Optional[str] = None,
        truncate_for_history_fn: Callable[..., str] | None = None,
        append_infer_history_fn: Callable[[dict[str, Any]], None] | None = None,
        timestamp_factory: Callable[[], str] | None = None,
        entry_id_factory: Callable[[], str] | None = None,
    ) -> None:
        truncate_fn = truncate_for_history_fn or self.truncate_for_history
        append_fn = append_infer_history_fn or self.append_infer_history
        timestamp_fn = timestamp_factory or self._hook('infer_history_timestamp')
        entry_id_fn = entry_id_factory or self._hook('infer_history_entry_id')
        entry = {
            'id': entry_id_fn(),
            'timestamp': timestamp_fn(),
            'instance_id': instance_id,
            'model': model_name,
            'backend': backend,
            'capability': capability,
            'prompt': prompt,
            'file_kind': 'pdf',
            'file_name': file_name,
            'file_sha256': file_sha256,
            'status': status,
            'mode': mode,
            'content': truncate_fn(content or ''),
            'error': error,
            'warnings': warnings or [],
            'pdf_source': pdf_source,
            'pdf_total_pages': pdf_total_pages,
            'pdf_processed_pages': pdf_processed_pages,
            'artifact_path': artifact_path,
        }
        append_fn(entry)
