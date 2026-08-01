"""OCR/PDF runtime owners for Ollmo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class OcrPdfRuntimeOwner:
    hooks: dict[str, Any]
    request_timeout_error: type[Exception]
    request_connection_error: type[Exception]
    request_exception_error: type[Exception]

    def _hook(self, name: str) -> Any:
        return self.hooks[name]

    def extract_pdf_text_content(self, pdf_path: Path, max_chars: Optional[int] = None) -> str:
        return self._hook('extract_pdf_text_content')(pdf_path, max_chars=max_chars)

    def render_pdf_pages_to_base64(
        self,
        pdf_path: Path,
        *,
        max_pages: Optional[int] = None,
        dpi: int = 180,
        max_image_side_px: int = 2400,
    ) -> tuple[list[str], int, list[str]]:
        return self._hook('render_pdf_pages_to_base64')(
            pdf_path,
            max_pages=max_pages,
            dpi=dpi,
            max_image_side_px=max_image_side_px,
        )

    def render_single_pdf_page_to_base64(
        self,
        pdf_path: Path,
        *,
        page_index: int,
        dpi: int,
        max_image_side_px: int = 2400,
        crop_margin_ratio: float = 0.0,
    ) -> Optional[str]:
        return self._hook('render_single_pdf_page_to_base64')(
            pdf_path,
            page_index=page_index,
            dpi=dpi,
            max_image_side_px=max_image_side_px,
            crop_margin_ratio=crop_margin_ratio,
        )

    def looks_like_ocr_prompt_echo(self, content: str, *, user_hint: str) -> bool:
        return self._hook('looks_like_ocr_prompt_echo')(content, user_hint=user_hint)

    def normalize_ocr_line(self, line: str) -> str:
        return self._hook('normalize_ocr_line')(line)

    def strip_ocr_structural_lines(self, text: str) -> str:
        return self._hook('strip_ocr_structural_lines')(text)

    def collapse_repeated_ocr_lines(self, text: str, *, max_repeats: int = 3) -> str:
        return self._hook('collapse_repeated_ocr_lines')(text, max_repeats=max_repeats)

    def line_has_ocr_garbage_pattern(self, line: str) -> bool:
        return self._hook('line_has_ocr_garbage_pattern')(line)

    def sanitize_ocr_noise_lines(self, text: str) -> str:
        return self._hook('sanitize_ocr_noise_lines')(text)

    def detect_low_quality_ocr_reason(self, content: str) -> Optional[str]:
        return self._hook('detect_low_quality_ocr_reason')(content)

    def clean_ocr_output_text(self, raw_text: str) -> str:
        return self._hook('clean_ocr_output_text')(raw_text)

    def ocr_pdf_page_with_ollama(
        self,
        *,
        port: int,
        model_name: str,
        base_prompt: str,
        page_index: int,
        total_pages: int,
        image_b64: str,
        timeout_sec: int,
    ) -> tuple[str, Optional[str]]:
        return self._hook('ocr_pdf_page_with_ollama')(
            port=port,
            model_name=model_name,
            base_prompt=base_prompt,
            page_index=page_index,
            total_pages=total_pages,
            image_b64=image_b64,
            timeout_sec=timeout_sec,
            generate_func=self._hook('generate_func'),
            extract_generate_content_func=self._hook('extract_generate_content_func'),
            request_timeout_error=self.request_timeout_error,
            request_connection_error=self.request_connection_error,
            request_exception_error=self.request_exception_error,
        )

    def is_generic_ocr_instruction_prompt(self, prompt: str) -> bool:
        return self._hook('is_generic_ocr_instruction_prompt')(prompt)

    def ocr_image_with_deepseek(
        self,
        *,
        port: int,
        model_name: str,
        image_b64: str,
        user_prompt: str,
        timeout_sec: int,
    ) -> tuple[str, Optional[str]]:
        return self._hook('ocr_image_with_deepseek')(
            port=port,
            model_name=model_name,
            image_b64=image_b64,
            user_prompt=user_prompt,
            timeout_sec=timeout_sec,
            generate_func=self._hook('generate_func'),
            extract_generate_content_func=self._hook('extract_generate_content_func'),
            request_timeout_error=self.request_timeout_error,
            request_connection_error=self.request_connection_error,
            request_exception_error=self.request_exception_error,
        )
