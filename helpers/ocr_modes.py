"""Model-aware OCR/document mode helpers for OCR-specialized vision models."""

from __future__ import annotations

from typing import Any

GLM_OCR_MODES = ['auto', 'text', 'table', 'formula', 'extract']
DEEPSEEK_OCR2_MODES = ['auto', 'markdown', 'free_ocr', 'extract']

GENERIC_OCR_FALLBACK_PROMPT = 'Analyze this image and extract relevant text and details.'


def normalize_ocr_mode(value: Any) -> str:
    return str(value or '').strip().lower() or 'auto'


def get_ocr_model_family(model_name: str | None) -> str | None:
    model_name_lower = str(model_name or '').strip().lower()
    if not model_name_lower:
        return None
    if 'deepseek-ocr-2' in model_name_lower:
        return 'deepseek_ocr2'
    if 'glm-ocr' in model_name_lower:
        return 'glm_ocr'
    return None


def get_ocr_mode_options(model_name: str | None) -> list[str]:
    family = get_ocr_model_family(model_name)
    if family == 'glm_ocr':
        return list(GLM_OCR_MODES)
    if family == 'deepseek_ocr2':
        return list(DEEPSEEK_OCR2_MODES)
    return []


def get_ocr_mode_copy(model_name: str | None) -> dict[str, str]:
    family = get_ocr_model_family(model_name)
    if family == 'glm_ocr':
        return {
            'hint': 'GLM-OCR document parsing controls for this vision-analysis model.',
            'label': 'GLM OCR Mode',
            'description': 'Use GLM-OCR parsing presets for text, tables, formulas, or custom extraction.',
            'mode_description': 'Choose a GLM-OCR parsing preset or use your typed prompt for extraction.',
        }
    if family == 'deepseek_ocr2':
        return {
            'hint': 'DeepSeek-OCR-2 document conversion controls for this vision-analysis model.',
            'label': 'DeepSeek OCR 2 Mode',
            'description': 'Use DeepSeek-OCR-2 markdown conversion or free OCR presets for document images and PDFs.',
            'mode_description': 'Choose a DeepSeek-OCR-2 OCR preset or use your typed prompt for extraction.',
        }
    return {
        'hint': 'PDF OCR controls for this vision-analysis model.',
        'label': 'OCR / Document Mode',
        'description': 'These controls are used for PDF OCR/document requests with this vision-analysis model.',
        'mode_description': 'Choose how Ollmo should process OCR/document prompts for this model.',
    }


def resolve_ocr_prompt(
    model_name: str | None,
    *,
    user_prompt: str | None = None,
    ocr_mode: str | None = None,
    generic_fallback: str = GENERIC_OCR_FALLBACK_PROMPT,
) -> str:
    family = get_ocr_model_family(model_name)
    prompt = str(user_prompt or '').strip()
    mode = normalize_ocr_mode(ocr_mode)

    if family == 'glm_ocr':
        preset_map = {
            'text': 'Text Recognition:',
            'table': 'Table Recognition:',
            'formula': 'Formula Recognition:',
        }
        if mode in preset_map:
            return preset_map[mode]
        if mode in {'auto', 'extract'}:
            return prompt or preset_map['text']

    if family == 'deepseek_ocr2':
        preset_map = {
            'markdown': '<|grounding|>Convert the document to markdown.',
            'free_ocr': 'Free OCR.',
        }
        if mode in preset_map:
            return preset_map[mode]
        if mode in {'auto', 'extract'}:
            return prompt or preset_map['markdown']

    return prompt or generic_fallback
