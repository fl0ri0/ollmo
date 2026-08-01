"""Shared transport and artifact helpers for Ollmo model backends."""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from helpers.model_capabilities import CAPABILITY_IMAGE_GENERATION

ARTIFACTS_ROOT = Path('artifacts')
ARTIFACT_OUTPUTS_ROOT = ARTIFACTS_ROOT
ARTIFACT_INPUTS_ROOT = ARTIFACTS_ROOT / 'inputs'
ARTIFACT_OUTPUTS_AUDIO_DIR = ARTIFACTS_ROOT / 'audio'
ARTIFACT_OUTPUTS_IMAGES_DIR = ARTIFACTS_ROOT / 'images'
ARTIFACT_OUTPUTS_OCR_DIR = ARTIFACTS_ROOT / 'ocr'
ARTIFACT_OUTPUTS_TRANSCRIPTS_DIR = ARTIFACTS_ROOT / 'transcripts'
ARTIFACT_OUTPUTS_DOCUMENTS_DIR = ARTIFACTS_ROOT / 'documents'
ARTIFACT_OUTPUTS_BENCHMARKS_DIR = ARTIFACTS_ROOT / 'benchmarks'
ARTIFACT_OUTPUTS_MANIFESTS_DIR = ARTIFACTS_ROOT / 'manifests'
ARTIFACT_OUTPUTS_BUNDLES_DIR = ARTIFACTS_ROOT / 'bundles'
ARTIFACT_INPUTS_TEXT_DIR = ARTIFACT_INPUTS_ROOT / 'text'
TEXT_ARTIFACT_EXTENSIONS = {
    'txt',
    'md',
    'markdown',
    'html',
    'htm',
    'css',
    'js',
    'mjs',
    'cjs',
    'ts',
    'tsx',
    'jsx',
    'json',
    'yaml',
    'yml',
    'xml',
    'csv',
    'svg',
    'py',
    'sh',
    'sql',
}
_TEXT_ARTIFACT_OUTER_FENCE_RE = re.compile(
    r'\A\s*```(?P<info>[^\r\n`]*)\r?\n(?P<body>[\s\S]*?)\r?\n?```\s*\Z'
)
_TEXT_ARTIFACT_MATERIALIZER_INSTRUCTION_LINE_RE = re.compile(
    r'^\s*(?:'
    r'Use this fence language:\s*[A-Za-z0-9_.+-]+'
    r'|Materialize only the requested\b.*'
    r'|Write the complete\b.*\bfile payload\b.*'
    r'|Return (?:only )?the complete file payload\b.*'
    r'|Output only the file body\b.*'
    r'|Do not output planner JSON\b.*'
    r'|Original user request for bounded intent context:.*'
    r'|Use the fulfilled dependency evidence below as concrete runtime truth\b.*'
    r'|Dependency evidence(?:\s*\([^)]*\))?:.*'
    r')\s*$',
    re.IGNORECASE,
)


def _safe_fragment(value: str, fallback: str) -> str:
    token = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(value or '')).strip('._-')
    return token or fallback


def _timestamp_prefix() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def expand_repo_relative_roots(*relative_roots: Path | str) -> set[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    bases = {Path.cwd(), repo_root}
    expanded: set[Path] = set()
    for relative_root in relative_roots:
        candidate = Path(relative_root)
        if candidate.is_absolute():
            expanded.add(candidate.resolve())
            continue
        for base in bases:
            expanded.add((base / candidate).resolve())
    return expanded


def _dedupe_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f'{path.stem}_{counter}{path.suffix}')
        if not candidate.exists():
            return candidate
        counter += 1


def extract_text_payload(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            chunk = extract_text_payload(item)
            if chunk:
                parts.append(chunk)
        return '\n'.join(parts).strip()
    if isinstance(value, dict):
        for key in ('text', 'content', 'response', 'output'):
            if key in value:
                chunk = extract_text_payload(value.get(key))
                if chunk:
                    return chunk
    return ''


def _normalize_text_artifact_extension_token(value: Any) -> str:
    token = re.sub(r'[^a-zA-Z0-9]+', '', str(value or '').strip().lower().lstrip('.'))
    aliases = {
        'markdown': 'md',
        'javascript': 'js',
        'typescript': 'ts',
        'bash': 'sh',
        'shell': 'sh',
        'plain': 'txt',
        'text': 'txt',
    }
    return aliases.get(token, token)


def strip_enclosing_text_artifact_fence(content: str, extension: str) -> str:
    """Strip one whole-payload Markdown fence from a persisted text artifact."""

    text = str(content or '').strip()
    if not text:
        return ''
    match = _TEXT_ARTIFACT_OUTER_FENCE_RE.match(text)
    if not match:
        return text
    info = str(match.group('info') or '').strip()
    language = _normalize_text_artifact_extension_token(info.split(None, 1)[0] if info else '')
    normalized_ext = _normalize_text_artifact_extension_token(extension)
    if (
        not language
        or language == normalized_ext
        or normalized_ext in {'', 'txt'}
        or language in TEXT_ARTIFACT_EXTENSIONS
    ):
        return str(match.group('body') or '').strip()
    return text


def text_artifact_content_is_materializer_instruction_echo(content: Any) -> bool:
    """Return true when a text artifact contains only internal materializer instructions."""

    text = str(content or '').strip()
    if not text:
        return False
    text = strip_enclosing_text_artifact_fence(text, '').strip()
    if re.match(r'^\s*Use the fulfilled dependency evidence below as concrete runtime truth\b', text, flags=re.IGNORECASE):
        return 'Dependency evidence' in text
    if re.match(r'^\s*Materialize only the requested\b', text, flags=re.IGNORECASE):
        return bool(
            re.search(r'\bReturn (?:only )?the complete file payload\b', text, flags=re.IGNORECASE)
            or re.search(r'\bDo not output planner JSON\b', text, flags=re.IGNORECASE)
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if all(_TEXT_ARTIFACT_MATERIALIZER_INSTRUCTION_LINE_RE.match(line) for line in lines):
        return True
    if (
        re.search(r'\bWrite the complete\b.*\bfile payload\b', text, flags=re.IGNORECASE)
        and re.search(r'\bOutput only the file body\b', text, flags=re.IGNORECASE)
    ):
        return True
    fence_language_lines = [
        line for line in lines
        if re.match(r'^\s*Use this fence language:', line, flags=re.IGNORECASE)
    ]
    return len(fence_language_lines) >= 2 and len(fence_language_lines) == len(lines)


def ollama_generate(
    port: int,
    model_name: str,
    prompt: str,
    requests_module,
    request_timeout_error,
    request_connection_error,
    request_exception_error,
    *,
    images: Optional[List[str]] = None,
    timeout_sec: int = 900,
    options: Optional[dict[str, Any]] = None,
    max_retries: int = 3,
    allow_port_fallback: bool = True,
) -> dict:
    payload: dict[str, Any] = {'model': model_name, 'prompt': prompt or '', 'stream': False}
    if images:
        payload['images'] = images
    model_name_lower = str(model_name or '').lower()
    if 'deepseek-ocr' in model_name_lower:
        payload['options'] = {'num_ctx': 4096, 'num_keep': 0}
    if options:
        merged = dict(payload.get('options') or {})
        merged.update(options)
        payload['options'] = merged

    request_timeout = max(30, int(timeout_sec))
    candidate_ports = [int(port)]
    if allow_port_fallback and int(port) != 11434:
        candidate_ports.append(11434)
    attempts_per_port = max(1, int(max_retries))

    last_exc: Optional[Exception] = None
    def _raise_non_json_response(response: Any, target_port: int, exc: Exception) -> None:
        status_code = getattr(response, 'status_code', None)
        text = str(getattr(response, 'text', '') or '').strip()
        if not text:
            content = getattr(response, 'content', b'')
            if isinstance(content, bytes):
                text = content[:240].decode('utf-8', errors='replace').strip()
            elif content:
                text = str(content).strip()
        preview = text[:240] if text else '<empty body>'
        raise RuntimeError(
            'Ollama generate returned non-JSON response '
            f'for model {model_name!r} on port {target_port} '
            f'(http_status={status_code or "unknown"}): {preview}'
        ) from exc

    for target_port in candidate_ports:
        for attempt in range(1, attempts_per_port + 1):
            try:
                response = requests_module.post(
                    f'http://127.0.0.1:{target_port}/api/generate',
                    json=payload,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    _raise_non_json_response(response, target_port, exc)
                if not isinstance(data, dict):
                    raise RuntimeError(
                        'Ollama generate returned unexpected JSON payload '
                        f'for model {model_name!r} on port {target_port}: {type(data).__name__}'
                    )
                return data
            except request_timeout_error as exc:
                last_exc = exc
                if attempt >= attempts_per_port:
                    break
                time.sleep(1.2 * attempt)
            except request_connection_error as exc:
                last_exc = exc
                if attempt >= attempts_per_port:
                    break
                time.sleep(1.2 * attempt)
            except request_exception_error as exc:
                raise exc
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break

        logging.warning(
            'Ollama generate failed on port %s for model %s; trying fallback if available.',
            target_port,
            model_name,
        )

    if last_exc:
        raise last_exc
    raise RuntimeError('Ollama generate failed without a captured exception.')


def extract_generate_content(data: dict) -> str:
    if not isinstance(data, dict):
        return ''
    response = extract_text_payload(data.get('response'))
    if response:
        return response
    message = data.get('message')
    if isinstance(message, dict):
        content = extract_text_payload(message.get('content'))
        if content:
            return content
    for key in ('text', 'output'):
        value = extract_text_payload(data.get(key))
        if value:
            return value
    return ''


def _coerce_seed_value(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    token = str(value or '').strip()
    if not token or not re.fullmatch(r'\d+', token):
        return None
    try:
        parsed = int(token)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def extract_generate_seed(data: dict) -> Optional[int]:
    if not isinstance(data, dict):
        return None

    candidate_dicts: list[dict[str, Any]] = [data]
    for key in ('result', 'options', 'metadata', 'response_metadata', 'message'):
        value = data.get(key)
        if isinstance(value, dict):
            candidate_dicts.append(value)

    for candidate in candidate_dicts:
        for key in ('seed', 'random_seed', 'rng_seed'):
            parsed = _coerce_seed_value(candidate.get(key))
            if parsed is not None:
                return parsed

    possible_texts: list[str] = []
    for value in (data.get('response'), data.get('text'), data.get('output')):
        if isinstance(value, str) and value.strip():
            possible_texts.append(value.strip())
    message = data.get('message')
    if isinstance(message, dict):
        content = message.get('content')
        if isinstance(content, str) and content.strip():
            possible_texts.append(content.strip())
    for text in possible_texts:
        match = re.search(r'\bseed\s*[:=]\s*(\d+)\b', text, re.IGNORECASE)
        if match:
            parsed = _coerce_seed_value(match.group(1))
            if parsed is not None:
                return parsed
    return None


def locate_saved_image_file_from_generate_output(data: dict, *, to_base64_func) -> Optional[Path]:
    if not isinstance(data, dict):
        return None

    possible_texts: list[str] = []
    for value in (data.get('response'), data.get('text'), data.get('output')):
        if isinstance(value, str) and value.strip():
            possible_texts.append(value.strip())
    message = data.get('message')
    if isinstance(message, dict):
        content = message.get('content')
        if isinstance(content, str) and content.strip():
            possible_texts.append(content.strip())

    path_pattern = re.compile(
        r'image\s+saved\s+to:\s*([^\n\r]+?\.(?:png|jpe?g|webp|gif|bmp|tiff?))',
        re.IGNORECASE,
    )
    for text in possible_texts:
        match = path_pattern.search(text)
        if not match:
            continue
        raw_path = match.group(1).strip().strip("'\"")
        candidate = Path(raw_path).expanduser()
        candidates: list[Path] = [candidate]
        if not candidate.is_absolute():
            candidates.append((Path.cwd() / candidate).resolve())
            candidates.append((Path.home() / candidate).resolve())
        for image_path in candidates:
            if image_path.exists() and image_path.is_file():
                return image_path.resolve()
        logging.warning(
            'Ollama returned saved-image path but file was not found: %s (candidates=%s)',
            raw_path,
            [str(item) for item in candidates],
        )
    return None


def extract_image_data_url_from_generate_output(data: dict, *, to_base64_func) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    images = data.get('images')
    if isinstance(images, list) and images and isinstance(images[0], str):
        first = images[0].strip()
        if not first:
            return None
        if first.startswith('data:image/'):
            return first
        return f'data:image/png;base64,{first}'

    located = locate_saved_image_file_from_generate_output(data, to_base64_func=to_base64_func)
    if located:
        mime, _ = mimetypes.guess_type(str(located))
        if not mime:
            mime = 'image/png'
        encoded = to_base64_func(located)
        return f'data:{mime};base64,{encoded}'

    return None


def extract_saved_image_path_from_generate_output(data: dict, *, to_base64_func) -> Optional[str]:
    located = locate_saved_image_file_from_generate_output(data, to_base64_func=to_base64_func)
    if not located:
        return None
    return str(located)


def persist_image_data_url_locally(image_data_url: Optional[str], model_name: str) -> Optional[str]:
    if not image_data_url or not isinstance(image_data_url, str):
        return None
    match = re.match(r'^data:(image/[^;]+);base64,(.+)$', image_data_url, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    mime_type = match.group(1).lower()
    encoded = match.group(2).strip()
    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        logging.warning('Could not decode generated image payload: %s', exc)
        return None

    ext = mimetypes.guess_extension(mime_type) or '.png'
    safe_model = _safe_fragment(model_name, 'image')
    ts = _timestamp_prefix()
    out_dir = ARTIFACT_OUTPUTS_IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _dedupe_output_path(out_dir / f'{ts}_image_{safe_model}{ext}')
    try:
        out_path.write_bytes(raw)
        return str(out_path.resolve())
    except OSError as exc:
        logging.warning('Could not persist generated image to disk: %s', exc)
        return None


def persist_audio_bytes_locally(
    audio_bytes: Optional[bytes],
    *,
    model_name: str,
    output_dir: Path,
    response_format: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Optional[str]:
    if not audio_bytes:
        return None
    ext = ''
    token = str(response_format or '').strip().lower().lstrip('.')
    if token in {'mp3', 'mpeg'}:
        ext = '.mp3'
    elif token in {'wav', 'wave', 'x-wav'}:
        ext = '.wav'
    elif token:
        ext = f'.{token}'
    elif content_type:
        normalized_content_type = str(content_type).split(';', 1)[0].strip().lower()
        if normalized_content_type in {'audio/mp3', 'audio/mpeg'}:
            ext = '.mp3'
        elif normalized_content_type in {'audio/wav', 'audio/x-wav', 'audio/wave'}:
            ext = '.wav'
        else:
            ext = mimetypes.guess_extension(normalized_content_type) or ''
    if not ext:
        ext = '.wav'

    safe_model = _safe_fragment(model_name, 'audio')
    ts = _timestamp_prefix()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _dedupe_output_path(output_dir / f'{ts}_audio_{safe_model}{ext}')
    try:
        out_path.write_bytes(audio_bytes)
        return str(out_path.resolve())
    except OSError as exc:
        logging.warning('Could not persist generated audio to disk: %s', exc)
        return None


def persist_text_markdown_locally(
    content: Optional[str],
    *,
    model_name: str,
    source_file_name: str,
    mode: str,
    output_dir: Path,
) -> Optional[str]:
    text = str(content or '').strip()
    if not text:
        return None
    safe_model = _safe_fragment(model_name, 'ocr')
    source_stem = Path(str(source_file_name or 'document')).stem
    safe_source = _safe_fragment(source_stem, 'document')
    safe_mode = _safe_fragment(mode, 'ocr')
    ts = _timestamp_prefix()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _dedupe_output_path(output_dir / f'{ts}_{safe_mode}_{safe_model}_{safe_source}.md')
    try:
        out_path.write_text(text + '\n', encoding='utf-8')
        return str(out_path.resolve())
    except OSError as exc:
        logging.warning('Could not persist OCR markdown output to disk: %s', exc)
        return None


def persist_text_artifact_locally(
    content: Optional[str],
    *,
    model_name: str,
    source_name: str,
    mode: str,
    extension: str,
    output_dir: Path,
    target_path: Optional[str] = None,
) -> Optional[str]:
    normalized_ext = re.sub(r'[^a-zA-Z0-9]+', '', str(extension or '').strip().lower().lstrip('.'))
    if normalized_ext == 'markdown':
        normalized_ext = 'md'
    if normalized_ext not in TEXT_ARTIFACT_EXTENSIONS:
        normalized_ext = 'txt'
    text = strip_enclosing_text_artifact_fence(str(content or ''), normalized_ext).strip()
    if not text:
        return None
    if text_artifact_content_is_materializer_instruction_echo(text):
        logging.warning(
            'Refusing to persist text artifact %s.%s because content is an internal materializer instruction echo',
            source_name,
            normalized_ext,
        )
        return None
    safe_model = _safe_fragment(model_name, 'chat')
    safe_source = _safe_fragment(Path(str(source_name or 'generated-text')).stem, 'generated-text')
    safe_mode = _safe_fragment(mode, 'text_artifact')
    ts = _timestamp_prefix()
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(target_path or '').strip():
        try:
            raw_target = Path(str(target_path or '').strip()).expanduser()
            target_candidates = (
                [raw_target.resolve()]
                if raw_target.is_absolute()
                else [
                    (Path.cwd() / raw_target).resolve(),
                    (Path(__file__).resolve().parent.parent / raw_target).resolve(),
                ]
            )
            allowed_roots = expand_repo_relative_roots(output_dir)
            for target in target_candidates:
                if any(target == root or target.is_relative_to(root) for root in allowed_roots):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text + '\n', encoding='utf-8')
                    return str(target)
        except (OSError, RuntimeError, ValueError) as exc:
            logging.warning('Could not persist text artifact to target path %s: %s', target_path, exc)
    out_path = _dedupe_output_path(output_dir / f'{ts}_{safe_mode}_{safe_model}_{safe_source}.{normalized_ext}')
    try:
        out_path.write_text(text + '\n', encoding='utf-8')
        return str(out_path.resolve())
    except OSError as exc:
        logging.warning('Could not persist generated text artifact to disk: %s', exc)
        return None


def persist_input_file_locally(
    source_path: Path,
    *,
    source_name: str,
    file_kind: str,
    output_root: Path = ARTIFACT_INPUTS_ROOT,
) -> Optional[str]:
    if not source_path or not source_path.exists() or not source_path.is_file():
        return None
    normalized_kind = _safe_fragment(str(file_kind or '').strip().lower(), 'input')
    safe_name = Path(str(source_name or source_path.name or 'input')).name
    source_stem = _safe_fragment(Path(safe_name).stem, 'input')
    suffix = Path(safe_name).suffix or source_path.suffix or ''
    ts = _timestamp_prefix()
    out_dir = output_root / normalized_kind
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _dedupe_output_path(out_dir / f'{ts}_{normalized_kind}_{source_stem}{suffix}')
    try:
        shutil.copy2(source_path, out_path)
        return str(out_path.resolve())
    except OSError as exc:
        logging.warning('Could not persist input artifact to disk: %s', exc)
        return None


def is_path_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_saved_artifact_path(raw_path: str, *, allowed_roots: set[Path]) -> Optional[Path]:
    raw_value = str(raw_path or '').strip()
    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate.resolve())
    else:
        app_root = Path(__file__).resolve().parent.parent
        candidates.append((Path.cwd() / candidate).resolve())
        candidates.append((app_root / candidate).resolve())

    for resolved in candidates:
        if not resolved.exists() or not resolved.is_file():
            continue
        if any(is_path_within(resolved, root) for root in allowed_roots):
            return resolved
    return None


def open_path_in_file_manager(path: Path) -> None:
    target = str(path)
    if os.name == 'nt':
        subprocess.Popen(['explorer', f'/select,{target}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    if sys.platform == 'darwin':
        subprocess.Popen(['open', '-R', target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    opener = shutil.which('xdg-open')
    if opener:
        subprocess.Popen([opener, str(path.parent)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    raise RuntimeError('No supported file-manager launcher was found on this system.')


def ollama_openai_image_generation(
    port: int,
    model_name: str,
    prompt: str,
    requests_module,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[str]:
    payload: dict[str, Any] = {
        'model': model_name,
        'prompt': prompt or '',
        'response_format': 'b64_json',
    }
    if width is not None and height is not None:
        payload['size'] = f'{int(width)}x{int(height)}'
    try:
        response = requests_module.post(
            f'http://127.0.0.1:{port}/v1/images/generations',
            json=payload,
            timeout=600,
        )
    except Exception as exc:  # noqa: BLE001
        logging.info('OpenAI image fallback request failed: %s', exc)
        return None

    if response.status_code == 404:
        return None
    try:
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        logging.info('OpenAI image fallback parsing failed: %s', exc)
        return None

    if not isinstance(data, dict):
        return None
    entries = data.get('data')
    if not isinstance(entries, list) or not entries:
        return None
    first = entries[0]
    if not isinstance(first, dict):
        return None
    b64_value = first.get('b64_json') or first.get('b64')
    if isinstance(b64_value, str) and b64_value.strip():
        token = b64_value.strip()
        if token.startswith('data:image/'):
            return token
        return f'data:image/png;base64,{token}'
    return None


def ollama_chat_with_options(
    *,
    port: int,
    model_name: str,
    messages: list[dict],
    requests_module,
    request_timeout_error,
    request_connection_error,
    request_exception_error,
    timeout_sec: int,
    allow_port_fallback: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> dict:
    payload: dict[str, Any] = {'model': model_name, 'messages': messages, 'stream': False}
    model_name_lower = str(model_name or '').lower()
    if 'deepseek-ocr' in model_name_lower:
        payload['options'] = {'num_ctx': 4096, 'num_keep': 0}
    if temperature is not None:
        payload.setdefault('options', {})['temperature'] = temperature
    if top_p is not None:
        payload.setdefault('options', {})['top_p'] = top_p
    candidate_ports = [int(port)]
    if allow_port_fallback and int(port) != 11434:
        candidate_ports.append(11434)

    last_exc: Optional[Exception] = None
    for target_port in candidate_ports:
        try:
            response = requests_module.post(
                f'http://127.0.0.1:{target_port}/api/chat',
                json=payload,
                timeout=max(30, int(timeout_sec)),
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return {'content': ''}
            message = data.get('message')
            content = extract_text_payload(message.get('content') if isinstance(message, dict) else None)
            if not content:
                content = extract_text_payload(data.get('content'))
            if not content:
                content = extract_text_payload(data.get('response'))
            return {'content': content}
        except request_timeout_error as exc:
            last_exc = exc
            continue
        except request_connection_error as exc:
            last_exc = exc
            continue
        except request_exception_error as exc:
            last_exc = exc
            raise exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            break

    if last_exc:
        raise last_exc
    raise RuntimeError('Ollama chat failed without a captured exception.')


def whisper_transcribe(port: int, audio_path: Path, requests_module, *, task: str = 'transcribe', language: Optional[str] = None) -> dict:
    payload: dict[str, Any] = {'audio_path': str(audio_path), 'task': task}
    if language:
        payload['language'] = language
    response = requests_module.post(
        f'http://127.0.0.1:{port}/v1/audio/transcriptions',
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError('Unerwartete Whisper-Antwort.')
    return data


def mlx_audio_speech(
    port: int,
    model_name: str,
    prompt: str,
    requests_module,
    *,
    instruct: Optional[str] = None,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: float = 1.0,
    pitch: float = 1.0,
    lang_code: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_sec: int = 600,
) -> dict:
    resolved_response_format = str(response_format or '').strip().lower() or 'wav'
    payload: dict[str, Any] = {
        'model': model_name,
        'input': prompt or '',
        'stream': False,
        'speed': speed,
        'pitch': pitch,
        'response_format': resolved_response_format,
    }
    if max_tokens is not None:
        payload['max_tokens'] = int(max_tokens)
    if instruct:
        payload['instruct'] = instruct
    if voice:
        payload['voice'] = voice
    if lang_code:
        payload['lang_code'] = lang_code

    response = requests_module.post(
        f'http://127.0.0.1:{port}/v1/audio/speech',
        json=payload,
        timeout=max(30, int(timeout_sec)),
    )
    response.raise_for_status()

    body = getattr(response, 'content', None)
    if body is None:
        body = getattr(response, '_body', b'')
    audio_bytes = bytes(body or b'')
    if not audio_bytes:
        raise ValueError('Unerwartete MLX Audio-Antwort ohne Audioinhalt.')

    headers = getattr(response, 'headers', {}) or {}
    content_type = headers.get('content-type') or headers.get('Content-Type')
    if not content_type:
        if resolved_response_format == 'mp3':
            content_type = 'audio/mpeg'
        elif resolved_response_format == 'wav':
            content_type = 'audio/wav'
        elif resolved_response_format == 'flac':
            content_type = 'audio/flac'
        else:
            content_type = f'audio/{resolved_response_format}'
    content_disposition = headers.get('content-disposition') or headers.get('Content-Disposition') or ''
    filename = None
    match = re.search(r'filename="?([^";]+)"?', str(content_disposition))
    if match:
        filename = match.group(1).strip()

    return {
        'audio_bytes': audio_bytes,
        'content_type': content_type,
        'filename': filename,
        'result': {
            'content_type': content_type,
            'filename': filename,
            'bytes': len(audio_bytes),
        },
    }


def mlx_chat_completions(
    port: int,
    model_name: str,
    messages: list[dict],
    requests_module,
    *,
    timeout_sec: int = 600,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> dict:
    payload = {
        'model': model_name,
        'messages': messages,
        'stream': False,
        'enable_thinking': False,
    }
    if temperature is not None:
        payload['temperature'] = temperature
    if top_p is not None:
        payload['top_p'] = top_p
    response = requests_module.post(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        json=payload,
        timeout=max(30, int(timeout_sec)),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError('Unerwartete MLX-Antwort.')
    choices = data.get('choices') or []
    if not isinstance(choices, list) or not choices:
        return {'content': ''}
    message = choices[0].get('message') if isinstance(choices[0], dict) else None
    content = extract_text_payload(message.get('content') if isinstance(message, dict) else None)
    if not content and isinstance(message, dict):
        content = extract_text_payload(message.get('reasoning'))
    return {'content': content, 'result': data}
