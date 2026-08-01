"""Shared PDF/OCR and OCR-quality helpers for Ollmo."""

from __future__ import annotations

import base64
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

OCR_STRUCTURAL_LINE_TOKENS = {
    'text',
    'image',
    'table',
    'title',
    'sub_title',
    'table_caption',
}

PDF_TEXT_EXTRACTION_MAX_CHARS = 2_000_000
PDF_PAGE_OCR_NUM_PREDICT = 8192
IMAGE_OCR_NUM_PREDICT = 4096


def extract_pdf_text_content(pdf_path: Path, max_chars: Optional[int] = PDF_TEXT_EXTRACTION_MAX_CHARS) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return ''

    chunks: list[str] = []
    total_len = 0
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = (page.extract_text() or '').strip()
            if not text:
                continue
            if max_chars is not None:
                remaining = max_chars - total_len
                if remaining <= 0:
                    logging.warning(
                        'PDF text extraction truncated for %s at %s chars.',
                        pdf_path,
                        max_chars,
                    )
                    break
                if len(text) > remaining:
                    chunks.append(text[:remaining])
                    total_len += remaining
                    logging.warning(
                        'PDF text extraction truncated for %s at %s chars.',
                        pdf_path,
                        max_chars,
                    )
                    break
            chunks.append(text)
            total_len += len(text)
    except Exception as exc:  # noqa: BLE001
        logging.warning('PDF text extraction failed for %s: %s', pdf_path, exc)
        return ''

    return '\n\n'.join(chunks).strip()


def render_pdf_pages_to_base64(
    pdf_path: Path,
    *,
    max_pages: Optional[int] = None,
    dpi: int = 180,
    max_image_side_px: int = 2400,
) -> tuple[list[str], int, list[str]]:
    warnings: list[str] = []
    try:
        import fitz  # type: ignore
    except ImportError:
        if sys.platform == 'darwin' and shutil.which('sips'):
            tmp = tempfile.NamedTemporaryFile(prefix='ollmo_pdf_', suffix='.png', delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            try:
                subprocess.run(
                    ['sips', '-s', 'format', 'png', str(pdf_path), '--out', str(tmp_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                encoded = base64.b64encode(tmp_path.read_bytes()).decode('utf-8')
                warnings.append('PyMuPDF is not installed: the PDF was rendered as a first-page-only image via sips.')
                return [encoded], 1, warnings
            except Exception as exc:  # noqa: BLE001
                logging.warning('sips fallback failed for %s: %s', pdf_path, exc)
                warnings.append(
                    "The PDF could not be rendered. Multi-page scanned-PDF "
                    "rendering can use optional 'PyMuPDF'; review its separate "
                    "AGPL-3.0 or commercial upstream terms before installation."
                )
                return [], 0, warnings
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        warnings.append(
            "PDF scan analysis requires optional 'PyMuPDF'. Review its separate "
            "AGPL-3.0 or commercial upstream terms before installation, then "
            "restart the webserver."
        )
        return [], 0, warnings

    encoded_pages: list[str] = []
    page_count = 0
    try:
        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            page_count = total_pages
            limit = total_pages if max_pages is None else min(total_pages, max_pages)
            base_zoom = max(1.0, float(dpi) / 72.0)
            did_downscale = False
            for page_index in range(limit):
                page = doc.load_page(page_index)
                max_page_points = max(float(page.rect.width), float(page.rect.height), 1.0)
                max_zoom_for_side = max(0.75, float(max_image_side_px) / max_page_points)
                effective_zoom = min(base_zoom, max_zoom_for_side)
                matrix = fitz.Matrix(effective_zoom, effective_zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                encoded_pages.append(base64.b64encode(pix.tobytes('png')).decode('utf-8'))
                if effective_zoom + 1e-6 < base_zoom:
                    did_downscale = True
            if did_downscale:
                warnings.append(
                    f'PDF pages were limited to a maximum edge length of {max_image_side_px}px to keep OCR stable.'
                )
            if total_pages > limit:
                warnings.append(f'PDF umfasst {total_pages} Seiten; analysiert wurden die ersten {limit} Seiten.')
    except Exception as exc:  # noqa: BLE001
        logging.warning('PDF rendering failed for %s: %s', pdf_path, exc)
        warnings.append('PDF-Seiten konnten nicht gerendert werden.')
        return [], page_count, warnings

    return encoded_pages, page_count, warnings


def render_single_pdf_page_to_base64(
    pdf_path: Path,
    *,
    page_index: int,
    dpi: int,
    max_image_side_px: int = 2400,
    crop_margin_ratio: float = 0.0,
) -> Optional[str]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return None

    try:
        with fitz.open(str(pdf_path)) as doc:
            if page_index < 0 or page_index >= len(doc):
                return None
            page = doc.load_page(page_index)
            base_zoom = max(1.0, float(dpi) / 72.0)
            max_page_points = max(float(page.rect.width), float(page.rect.height), 1.0)
            max_zoom_for_side = max(0.75, float(max_image_side_px) / max_page_points)
            effective_zoom = min(base_zoom, max_zoom_for_side)
            matrix = fitz.Matrix(effective_zoom, effective_zoom)
            clip_rect = None
            ratio = max(0.0, min(0.2, float(crop_margin_ratio or 0.0)))
            if ratio > 0.0:
                rect = page.rect
                dx = rect.width * ratio
                dy = rect.height * ratio
                candidate = fitz.Rect(rect.x0 + dx, rect.y0 + dy, rect.x1 - dx, rect.y1 - dy)
                if candidate.width > 72 and candidate.height > 72:
                    clip_rect = candidate
            pix = page.get_pixmap(matrix=matrix, alpha=False, clip=clip_rect)
            return base64.b64encode(pix.tobytes('png')).decode('utf-8')
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            'Single-page PDF rendering failed for %s (page=%s, dpi=%s): %s',
            pdf_path,
            page_index + 1,
            dpi,
            exc,
        )
        return None


def is_generic_ocr_instruction_prompt(prompt: str) -> bool:
    text = re.sub(r'\s+', ' ', str(prompt or '').strip().lower())
    if not text:
        return True
    exact_markers = {
        'ocr',
        'free ocr',
        'extract text',
        'read text',
        'scan text',
        'text',
        'ocr these',
        'ocr all',
        'transcribe',
    }
    if text in exact_markers:
        return True
    if text.startswith('ocr ') and len(text) <= 36:
        return True
    if len(text) <= 18 and 'text' in text:
        return True
    return False


def looks_like_ocr_prompt_echo(content: str, *, user_hint: str) -> bool:
    text = str(content or '').strip()
    if not text:
        return True
    lowered = text.lower()
    marker_values = [
        'user request/context:',
        'for each page:',
        'verbatim transcription (preserve line breaks and structure)',
        'mark uncertain readings as [unclear]',
        'open questions / ambiguities',
        'action items',
        '<|grounding|>convert the document to markdown',
        'free ocr.',
    ]
    marker_hits = sum(1 for marker in marker_values if marker in lowered)
    if lowered.startswith('user request/context:'):
        return True
    if lowered.count('user request/context:') >= 2:
        return True
    if marker_hits >= 3:
        return True

    normalized_text = re.sub(r'\s+', ' ', lowered).strip()
    normalized_hint = re.sub(r'\s+', ' ', str(user_hint or '').lower()).strip()
    if normalized_hint and not is_generic_ocr_instruction_prompt(normalized_hint):
        if normalized_text.count(normalized_hint) >= 2:
            return True
        if normalized_hint in normalized_text and len(normalized_text) <= max(1000, len(normalized_hint) * 6):
            return True
    return False


def normalize_ocr_line(line: str) -> str:
    return re.sub(r'\s+', ' ', str(line or '').strip()).lower()


def strip_ocr_structural_lines(text: str) -> str:
    lines = str(text or '').splitlines()
    kept: list[str] = []
    for raw_line in lines:
        if normalize_ocr_line(raw_line) in OCR_STRUCTURAL_LINE_TOKENS:
            continue
        kept.append(raw_line)
    cleaned = '\n'.join(kept)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def collapse_repeated_ocr_lines(text: str, *, max_repeats: int = 3) -> str:
    lines = str(text or '').splitlines()
    if not lines:
        return ''
    collapsed: list[str] = []
    prev_norm = None
    repeat_count = 0
    for raw_line in lines:
        norm = normalize_ocr_line(raw_line)
        if norm and norm == prev_norm:
            repeat_count += 1
            if repeat_count <= max_repeats:
                collapsed.append(raw_line)
            continue
        prev_norm = norm if norm else None
        repeat_count = 1
        collapsed.append(raw_line)
    cleaned = '\n'.join(collapsed)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def line_has_ocr_garbage_pattern(line: str) -> bool:
    normalized = re.sub(r'[^0-9a-zA-Z]+', ' ', str(line or '').lower()).strip()
    if not normalized:
        return False
    tokens = [token for token in normalized.split() if token]
    if len(tokens) < 20:
        return False

    max_run = 1
    run_token = tokens[0]
    current_run = 1
    current_token = tokens[0]
    for token in tokens[1:]:
        if token == current_token:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                run_token = token
        else:
            current_run = 1
            current_token = token
    if max_run >= 18:
        return True

    token_counts = Counter(tokens)
    top_token, top_count = token_counts.most_common(1)[0]
    token_ratio = top_count / max(1, len(tokens))
    if top_token.isdigit() and top_count >= 24 and token_ratio >= 0.45:
        return True
    if len(run_token) >= 2 and max_run >= 12 and (max_run / max(1, len(tokens))) >= 0.45:
        return True

    for n in (6, 5, 4, 3):
        if len(tokens) < n * 6:
            continue
        ngram_counts = Counter(' '.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        _top_phrase, phrase_count = ngram_counts.most_common(1)[0]
        phrase_coverage = (phrase_count * n) / max(1, len(tokens))
        if (phrase_count >= 8 and phrase_coverage >= 0.30) or (phrase_count >= 20 and phrase_coverage >= 0.18):
            return True
    return False


def sanitize_ocr_noise_lines(text: str) -> str:
    lines = str(text or '').splitlines()
    if not lines:
        return ''
    sanitized_lines: list[str] = []
    replaced_lines = 0
    for raw_line in lines:
        if line_has_ocr_garbage_pattern(raw_line):
            replaced_lines += 1
            if not sanitized_lines or normalize_ocr_line(sanitized_lines[-1]) != '[unclear]':
                sanitized_lines.append('[unclear]')
            continue
        sanitized_lines.append(raw_line)
    if replaced_lines:
        logging.info('OCR cleanup replaced %s noisy line(s) with [unclear].', replaced_lines)
    cleaned = '\n'.join(sanitized_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def detect_low_quality_ocr_reason(content: str) -> Optional[str]:
    text = str(content or '').strip()
    if not text:
        return 'empty OCR output'

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 'empty OCR output'

    useful_lines = [line for line in lines if normalize_ocr_line(line) not in OCR_STRUCTURAL_LINE_TOKENS]
    if not useful_lines:
        return 'only structural OCR markers'

    normalized_lines = [normalize_ocr_line(line) for line in useful_lines if normalize_ocr_line(line)]
    if normalized_lines:
        unique_line_count = len(set(normalized_lines))
        if unique_line_count == 1 and len(normalized_lines) >= 3 and len(normalized_lines[0]) >= 4:
            return f'single-line repetition only ({normalized_lines[0][:64]})'
        line_counts = Counter(normalized_lines)
        top_line, top_count = line_counts.most_common(1)[0]
        if top_count >= 6 and (top_count / max(1, len(normalized_lines))) >= 0.7:
            return f'dominant repeated line ({top_count}x: {top_line[:64]})'
        if top_count >= 20 and (top_count / max(1, len(normalized_lines))) >= 0.35:
            return f'repeated line pattern ({top_count}x: {top_line[:64]})'

    tokens = [token.lower() for token in re.findall(r'\w+', ' '.join(useful_lines), flags=re.UNICODE) if token]
    if len(tokens) >= 120:
        max_run = 1
        run_token = tokens[0]
        current_run = 1
        current_token = tokens[0]
        for token in tokens[1:]:
            if token == current_token:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    run_token = token
            else:
                current_run = 1
                current_token = token
        if max_run >= 24:
            return f'consecutive repeated token run ({max_run}x: {run_token})'

        token_counts = Counter(token for token in tokens if token.strip())
        if token_counts:
            top_token, top_token_count = token_counts.most_common(1)[0]
            top_ratio = top_token_count / max(1, len(tokens))
            if (len(top_token) >= 2 or top_token.isdigit()) and top_token_count >= 40 and top_ratio >= 0.28:
                return f'repeated token pattern ({top_token_count}x: {top_token})'

        for n in (4, 3):
            if len(tokens) < n * 10:
                continue
            ngram_counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
            top_ngram, top_ngram_count = ngram_counts.most_common(1)[0]
            ngram_coverage = (top_ngram_count * n) / max(1, len(tokens))
            if top_ngram_count >= 10 and ngram_coverage >= 0.22:
                return f"repeated phrase loop ({top_ngram_count}x: {' '.join(top_ngram[:3])})"

    return None


def clean_ocr_output_text(raw_text: str, *, max_chars: Optional[int] = None) -> str:
    text = str(raw_text or '')
    if max_chars is not None and len(text) > max_chars:
        logging.warning('OCR raw output truncated from %s to %s chars for stability.', len(text), max_chars)
        text = text[:max_chars]
    if not text:
        return ''
    text = re.sub(r'<\|/?ref\|>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|det\|>\s*\[\[[^\]]*\]\]\s*<\|/det\|>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|/?grounding\|>', '', text, flags=re.IGNORECASE)
    text = strip_ocr_structural_lines(text)
    text = collapse_repeated_ocr_lines(text, max_repeats=3)
    text = sanitize_ocr_noise_lines(text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def ocr_pdf_page_with_ollama(
    *,
    port: int,
    model_name: str,
    base_prompt: str,
    page_index: int,
    total_pages: int,
    image_b64: str,
    timeout_sec: int,
    generate_func: Callable[..., dict],
    extract_generate_content_func: Callable[[dict], str],
    request_timeout_error,
    request_connection_error,
    request_exception_error,
) -> tuple[str, Optional[str]]:
    user_hint_raw = str(base_prompt or '').strip()

    logging.info(
        'PDF OCR page request: page=%s/%s model=%s timeout_sec=%s',
        page_index,
        total_pages,
        model_name,
        timeout_sec,
    )
    page_prompt = '<image>\n<|grounding|>Convert the document to markdown.'
    generate_error: Optional[str] = None
    try:
        page_out = generate_func(
            port,
            model_name,
            page_prompt,
            images=[image_b64],
            timeout_sec=timeout_sec,
            options={'num_predict': PDF_PAGE_OCR_NUM_PREDICT},
            max_retries=1,
            allow_port_fallback=False,
        )
        page_content = clean_ocr_output_text(extract_generate_content_func(page_out))
        logging.info('PDF OCR page=%s primary generate chars=%s', page_index, len(page_content))
        if page_content:
            if looks_like_ocr_prompt_echo(page_content, user_hint=user_hint_raw):
                generate_error = f'Page {page_index}: prompt echo returned instead of OCR text via /api/generate.'
            else:
                low_quality_reason = detect_low_quality_ocr_reason(page_content)
                if low_quality_reason:
                    generate_error = f'Page {page_index}: low OCR quality ({low_quality_reason}).'
                else:
                    return page_content, None
    except request_timeout_error:
        generate_error = f'Page {page_index}: timed out after {timeout_sec}s.'
    except request_connection_error as exc:
        generate_error = f'Page {page_index}: connection dropped ({exc}).'
    except request_exception_error as exc:
        status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
        details = ''
        response_obj = getattr(exc, 'response', None)
        if response_obj is not None:
            try:
                payload = response_obj.json()
                details = str(payload.get('error') or payload.get('message') or '').strip() if isinstance(payload, dict) else ''
            except Exception:
                details = str(getattr(response_obj, 'text', '') or '').strip()[:220]
        if status_code:
            generate_error = f'Page {page_index}: upstream HTTP {status_code}.'
            if details:
                generate_error = f'{generate_error} {details}'
        else:
            generate_error = f'Page {page_index}: upstream error ({exc}).'
    except Exception as exc:  # noqa: BLE001
        generate_error = f'Page {page_index}: unexpected OCR error ({exc}).'

    logging.info(
        'PDF OCR page=%s has no usable /api/generate content (%s); trying emergency /api/generate fallback.',
        page_index,
        generate_error or 'empty-response',
    )

    emergency_prompt = '<image>\nFree OCR.'
    emergency_error: Optional[str] = None
    emergency_timeout_sec = max(45, min(timeout_sec, 45))
    try:
        logging.info('PDF OCR page=%s emergency generate fallback timeout_sec=%s', page_index, emergency_timeout_sec)
        emergency_out = generate_func(
            port,
            model_name,
            emergency_prompt,
            images=[image_b64],
            timeout_sec=emergency_timeout_sec,
            options={'num_predict': PDF_PAGE_OCR_NUM_PREDICT},
            max_retries=1,
            allow_port_fallback=False,
        )
        emergency_content = clean_ocr_output_text(extract_generate_content_func(emergency_out))
        logging.info('PDF OCR page=%s emergency generate chars=%s', page_index, len(emergency_content))
        if emergency_content:
            if looks_like_ocr_prompt_echo(emergency_content, user_hint=user_hint_raw):
                emergency_error = f'Page {page_index}: prompt echo returned instead of OCR text in the emergency fallback.'
            else:
                low_quality_reason = detect_low_quality_ocr_reason(emergency_content)
                if low_quality_reason:
                    emergency_error = f'Page {page_index}: low OCR quality in the emergency fallback ({low_quality_reason}).'
                else:
                    return emergency_content, None
    except request_timeout_error:
        emergency_error = f'Page {page_index}: emergency fallback timed out after {emergency_timeout_sec}s.'
    except request_connection_error as exc:
        emergency_error = f'Page {page_index}: emergency fallback connection dropped ({exc}).'
    except request_exception_error as exc:
        status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status_code:
            emergency_error = f'Page {page_index}: emergency fallback HTTP {status_code}.'
        else:
            emergency_error = f'Page {page_index}: emergency fallback error ({exc}).'
    except Exception as exc:  # noqa: BLE001
        emergency_error = f'Page {page_index}: unexpected emergency fallback error ({exc}).'
        logging.info('PDF OCR emergency generate retry failed for page=%s: %s', page_index, exc)

    if generate_error and emergency_error:
        return '', f'{generate_error} {emergency_error}'
    if generate_error:
        return '', generate_error
    if emergency_error:
        return '', emergency_error
    return '', f'Page {page_index}: no OCR text was returned.'


def ocr_image_with_deepseek(
    *,
    port: int,
    model_name: str,
    image_b64: str,
    user_prompt: str,
    timeout_sec: int,
    generate_func: Callable[..., dict],
    extract_generate_content_func: Callable[[dict], str],
    request_timeout_error,
    request_connection_error,
    request_exception_error,
) -> tuple[str, Optional[str]]:
    user_hint_raw = str(user_prompt or '').strip()
    primary_prompt = '<image>\n<|grounding|>Convert the document to markdown.'
    primary_error: Optional[str] = None
    try:
        primary_out = generate_func(
            port,
            model_name,
            primary_prompt,
            images=[image_b64],
            timeout_sec=timeout_sec,
            options={'num_predict': IMAGE_OCR_NUM_PREDICT},
            max_retries=1,
            allow_port_fallback=False,
        )
        primary_content = clean_ocr_output_text(extract_generate_content_func(primary_out))
        logging.info('Image OCR primary generate chars=%s', len(primary_content))
        if primary_content:
            if looks_like_ocr_prompt_echo(primary_content, user_hint=user_hint_raw):
                primary_error = 'Image OCR: prompt echo returned instead of OCR text via /api/generate.'
            else:
                low_quality_reason = detect_low_quality_ocr_reason(primary_content)
                if low_quality_reason:
                    primary_error = f'Image OCR: low OCR quality ({low_quality_reason}).'
                else:
                    return primary_content, None
    except request_timeout_error:
        primary_error = f'Image OCR: timed out after {timeout_sec}s.'
    except request_connection_error as exc:
        primary_error = f'Image OCR: connection dropped ({exc}).'
    except request_exception_error as exc:
        status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status_code:
            primary_error = f'Image OCR: upstream HTTP {status_code}.'
        else:
            primary_error = f'Image OCR: upstream error ({exc}).'
    except Exception as exc:  # noqa: BLE001
        primary_error = f'Image OCR: unexpected error ({exc}).'

    logging.info(
        'Image OCR has no usable /api/generate content (%s); trying emergency fallback.',
        primary_error or 'empty-response',
    )

    emergency_prompt = '<image>\nFree OCR.'
    emergency_error: Optional[str] = None
    try:
        emergency_timeout_sec = max(45, min(timeout_sec, 90))
        emergency_out = generate_func(
            port,
            model_name,
            emergency_prompt,
            images=[image_b64],
            timeout_sec=emergency_timeout_sec,
            options={'num_predict': IMAGE_OCR_NUM_PREDICT},
            max_retries=1,
            allow_port_fallback=False,
        )
        emergency_content = clean_ocr_output_text(extract_generate_content_func(emergency_out))
        logging.info('Image OCR emergency generate chars=%s', len(emergency_content))
        if emergency_content:
            if looks_like_ocr_prompt_echo(emergency_content, user_hint=user_hint_raw):
                emergency_error = 'Image OCR: prompt echo returned instead of OCR text in the emergency fallback.'
            else:
                low_quality_reason = detect_low_quality_ocr_reason(emergency_content)
                if low_quality_reason:
                    emergency_error = f'Image OCR: low OCR quality in the emergency fallback ({low_quality_reason}).'
                else:
                    return emergency_content, None
    except request_timeout_error:
        emergency_error = 'Image OCR: emergency fallback timed out.'
    except request_connection_error as exc:
        emergency_error = f'Image OCR: emergency fallback connection dropped ({exc}).'
    except request_exception_error as exc:
        status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
        if status_code:
            emergency_error = f'Image OCR: emergency fallback HTTP {status_code}.'
        else:
            emergency_error = f'Image OCR: emergency fallback error ({exc}).'
    except Exception as exc:  # noqa: BLE001
        emergency_error = f'Image OCR: unexpected emergency fallback error ({exc}).'

    if primary_error and emergency_error:
        return '', f'{primary_error} {emergency_error}'
    if primary_error:
        return '', primary_error
    if emergency_error:
        return '', emergency_error
    return '', 'Image OCR: no OCR text was returned.'
