"""Build portable bundles from existing response artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import mimetypes
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from ollmo_services.artifact_contracts import sanitize_artifact_record


DEFAULT_BUNDLE_ROOT = Path('artifacts/bundles')
TEXT_EXTENSIONS = {'html', 'htm', 'css', 'js', 'mjs', 'cjs', 'json', 'md', 'txt', 'svg', 'xml'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'svg'}
AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'oga', 'opus'}
HTML_EXTENSIONS = {'html', 'htm'}
CSS_EXTENSIONS = {'css'}
JS_EXTENSIONS = {'js', 'mjs', 'cjs'}
SYNTAX_CHECK_EXTENSIONS = {'html', 'htm', 'css', 'json'}
IGNORED_LINK_SCHEMES = ('http:', 'https:', 'data:', 'blob:', 'mailto:', 'tel:', 'javascript:')


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _timestamp_token(value: Any) -> str:
    token = str(value or '').strip()
    if not token:
        return _utc_timestamp()
    compact = re.sub(r'[^0-9TZ]', '', token.upper())
    if re.fullmatch(r'\d{8}T\d{6}Z', compact):
        return compact
    if re.fullmatch(r'\d{14}', compact):
        return f'{compact[:8]}T{compact[8:]}Z'
    return _utc_timestamp()


def _safe_fragment(value: Any, fallback: str) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value or '').strip()).strip('._-')
    return token or fallback


def _clean_text(value: Any) -> str:
    return str(value or '').strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_existing_file(raw_path: Any) -> Optional[Path]:
    token = _clean_text(raw_path)
    if not token:
        return None
    candidate = Path(token).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, _repo_root() / candidate]
    for item in candidates:
        try:
            resolved = item.resolve(strict=False)
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_for_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    extension = path.suffix.lower().lstrip('.')
    if extension:
        return extension
    mime_type = _clean_text(artifact.get('mime_type') or artifact.get('mimeType')).lower()
    guessed = mimetypes.guess_extension(mime_type) if mime_type else ''
    return str(guessed or '').lower().lstrip('.')


def _type_for_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    artifact_type = _clean_text(artifact.get('type') or artifact.get('kind')).lower()
    extension = _extension_for_artifact(path, artifact)
    if artifact_type:
        return artifact_type
    if extension in IMAGE_EXTENSIONS:
        return 'image'
    if extension in AUDIO_EXTENSIONS:
        return 'audio'
    if extension in TEXT_EXTENSIONS:
        return 'text'
    return 'file'


def _iter_artifact_values(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_artifact_values(item)
        return
    if not isinstance(value, Mapping):
        return
    if isinstance(value.get('artifact'), Mapping):
        yield from _iter_artifact_values(value.get('artifact'))
    if value.get('path') or value.get('source_path') or value.get('artifact_ref') or value.get('ref'):
        normalized = sanitize_artifact_record(value, include_content=True)
        if normalized:
            yield normalized
    for key in (
        'artifacts',
        'input_artifacts',
        'reference_artifacts',
        'selected_reference_artifacts',
        'output_artifacts',
        'outputs',
        'output',
        'results',
    ):
        child = value.get(key)
        if isinstance(child, (list, Mapping)):
            yield from _iter_artifact_values(child)
    frame = value.get('response_frame')
    if isinstance(frame, Mapping):
        frame_artifacts = frame.get('artifacts')
        if isinstance(frame_artifacts, Mapping):
            for key in ('output', 'reference', 'input'):
                yield from _iter_artifact_values(frame_artifacts.get(key))
        output = frame.get('output')
        if isinstance(output, Mapping):
            yield from _iter_artifact_values(output.get('outputs'))


def _collect_existing_artifacts(response_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in _iter_artifact_values(response_payload):
        path = _resolve_existing_file(record.get('path') or record.get('source_path'))
        if path is None:
            continue
        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        extension = _extension_for_artifact(path, record)
        artifact_type = _type_for_artifact(path, record)
        artifacts.append(
            {
                **dict(record),
                'source_path': str(path),
                'type': artifact_type,
                'kind': artifact_type,
                'extension': extension,
                'basename': path.name,
            }
        )
    return artifacts


def _status_token(value: Any) -> str:
    return _clean_text(value).lower().replace('-', '_').replace(' ', '_')


def _artifact_public_keys(record: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in ('artifact_ref', 'ref'):
        token = _clean_text(record.get(key))
        if token:
            keys.add(f'ref:{token}')
    for key in ('path', 'source_path'):
        token = _clean_text(record.get(key))
        if token:
            keys.add(f'path:{str(Path(token).expanduser())}')
            try:
                keys.add(f'path:{Path(token).expanduser().resolve(strict=False)}')
            except OSError:
                pass
    return keys


def _artifact_ref_token(record: Mapping[str, Any]) -> str:
    artifact_id = _clean_text(record.get('artifact_id'))
    return _clean_text(
        record.get('artifact_ref')
        or record.get('ref')
        or (f'artifact:{artifact_id}' if artifact_id else '')
    )


def _is_generated_image_text_misbinding(record: Mapping[str, Any]) -> bool:
    artifact_type = _status_token(record.get('type') or record.get('kind'))
    if artifact_type not in {'text', 'document'}:
        return False
    artifact_ref = _artifact_ref_token(record)
    artifact_id = _clean_text(record.get('artifact_id'))
    return (
        artifact_ref.startswith('artifact:text_generated_image_')
        or artifact_id.startswith('text_generated_image_')
    )


def _is_public_output_artifact(record: Mapping[str, Any]) -> bool:
    if _is_generated_image_text_misbinding(record):
        return False
    status = _status_token(record.get('status') or record.get('state'))
    if status in {
        'blocked',
        'cancelled',
        'failed',
        'open',
        'pending',
        'repair_needed',
        'rejected',
        'skipped',
        'superseded',
        'waived',
    }:
        return False
    if bool(record.get('compatibility_derived')):
        return False
    source = _status_token(record.get('source'))
    if source in {'compatibility_derived', 'raw_saved_artifact_fallback'}:
        return False
    if not (record.get('artifact_ref') or record.get('ref') or record.get('path') or record.get('source_path')):
        return False
    return True


def _iter_output_surfaces(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    for key in ('outputs', 'output_artifacts'):
        child = value.get(key)
        if isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    yield item
        elif isinstance(child, Mapping):
            yield child
    output = value.get('output')
    if isinstance(output, Mapping):
        child = output.get('outputs')
        if isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    yield item
    current_state = value.get('current_state')
    if isinstance(current_state, Mapping):
        child = current_state.get('outputs')
        if isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    yield item
    frame = value.get('response_frame')
    if isinstance(frame, Mapping):
        yield from _iter_output_surfaces(frame)
    working_frame = value.get('working_frame')
    if isinstance(working_frame, Mapping):
        yield from _iter_output_surfaces(working_frame)


def _public_output_artifact_filter(response_payload: Mapping[str, Any]) -> tuple[bool, set[str]]:
    has_output_surface = False
    public_keys: set[str] = set()
    for output in _iter_output_surfaces(response_payload):
        has_output_surface = True
        if not _is_public_output_artifact(output):
            continue
        public_keys.update(_artifact_public_keys(output))
        for artifact in _iter_artifact_values(output.get('artifacts')):
            if _is_public_output_artifact(artifact):
                public_keys.update(_artifact_public_keys(artifact))
    return has_output_surface, public_keys


def _filter_public_bundle_artifacts(
    response_payload: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_output_surface, public_keys = _public_output_artifact_filter(response_payload)
    if not has_output_surface:
        return artifacts
    if not public_keys:
        return []
    filtered: list[dict[str, Any]] = []
    for artifact in artifacts:
        if _artifact_public_keys(artifact) & public_keys:
            filtered.append(artifact)
    return filtered


def _is_bundleable_artifact(artifact: Mapping[str, Any]) -> bool:
    extension = _clean_text(artifact.get('extension')).lower()
    artifact_type = _clean_text(artifact.get('type') or artifact.get('kind')).lower()
    return artifact_type in {'text', 'document', 'image', 'audio', 'file'} or bool(extension)


def _select_entrypoint(artifacts: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    html_artifacts = [item for item in artifacts if _clean_text(item.get('extension')).lower() in HTML_EXTENSIONS]
    for item in html_artifacts:
        source = Path(_clean_text(item.get('source_path')))
        name = _clean_text(item.get('name')).lower()
        if source.stem.lower() == 'index' or name == 'index':
            return item
    if html_artifacts:
        return html_artifacts[0]
    for item in artifacts:
        extension = _clean_text(item.get('extension')).lower()
        if (
            _clean_text(item.get('type')).lower() in {'text', 'document'}
            and extension not in CSS_EXTENSIONS | JS_EXTENSIONS
        ):
            return item
    return None


def _subdir_for_artifact(artifact: Mapping[str, Any], *, is_entrypoint: bool) -> Path:
    extension = _clean_text(artifact.get('extension')).lower()
    artifact_type = _clean_text(artifact.get('type') or artifact.get('kind')).lower()
    if is_entrypoint and (
        extension in HTML_EXTENSIONS
        or (artifact_type in {'text', 'document'} and extension not in CSS_EXTENSIONS | JS_EXTENSIONS)
    ):
        return Path()
    if extension in CSS_EXTENSIONS:
        return Path('assets/css')
    if extension in JS_EXTENSIONS:
        return Path('assets/js')
    if artifact_type == 'image' or extension in IMAGE_EXTENSIONS:
        return Path('assets/images')
    if artifact_type == 'audio' or extension in AUDIO_EXTENSIONS:
        return Path('assets/audio')
    return Path('assets/files')


def _dedupe_path(path: Path, used: set[Path]) -> Path:
    candidate = path
    index = 2
    while candidate in used or candidate.exists():
        candidate = path.with_name(f'{path.stem}_{index}{path.suffix}')
        index += 1
    used.add(candidate)
    return candidate


def _safe_filename(source: Path, artifact: Mapping[str, Any], *, is_entrypoint: bool) -> str:
    extension = _clean_text(artifact.get('extension')).lower()
    suffix = f'.{extension}' if extension and not source.name.lower().endswith(f'.{extension}') else source.suffix
    if is_entrypoint and extension in HTML_EXTENSIONS:
        return f'index{suffix or ".html"}'
    name = _clean_text(artifact.get('name'))
    if name:
        candidate = Path(name).name
        if suffix and not Path(candidate).suffix:
            candidate = f'{candidate}{suffix}'
    else:
        candidate = source.name
    stem = _safe_fragment(Path(candidate).stem, 'artifact')
    final_suffix = Path(candidate).suffix or suffix
    return f'{stem}{final_suffix}'


def _is_ignored_reference(value: Any) -> bool:
    token = _clean_text(value)
    if not token or token.startswith('#'):
        return True
    lowered = token.lower()
    return lowered.startswith(IGNORED_LINK_SCHEMES) or lowered.startswith('//')


def _split_ref(value: str) -> tuple[str, str]:
    match = re.match(r'^([^?#]*)(.*)$', value)
    if not match:
        return value, ''
    return match.group(1), match.group(2)


def _normalize_ref_key(value: Any) -> str:
    path_part, _suffix = _split_ref(_clean_text(value).replace('\\', '/'))
    return path_part.strip().lstrip('./').lower()


def _add_alias(alias_map: dict[str, Path], alias: Any, destination: Path) -> None:
    key = _normalize_ref_key(alias)
    if key and key not in alias_map:
        alias_map[key] = destination


def _build_alias_map(copied_artifacts: list[dict[str, Any]]) -> dict[str, Path]:
    alias_map: dict[str, Path] = {}
    for item in copied_artifacts:
        destination = Path(item['path'])
        source = Path(item['source_path'])
        destination_relative = _clean_text(item.get('relative_path'))
        aliases = {
            source.name,
            str(source),
            destination.name,
            destination_relative,
            Path(destination_relative).name if destination_relative else '',
            item.get('name'),
        }
        extension = source.suffix or destination.suffix
        if item.get('name') and extension:
            aliases.add(f'{item.get("name")}{extension}')
        for alias in aliases:
            _add_alias(alias_map, alias, destination)
    return alias_map


def _relative_path_between(from_file: Path, to_file: Path, suffix: str = '') -> str:
    rel = Path(
        re.sub(
            r'\\',
            '/',
            str(Path(to_file).resolve()),
        )
    )
    try:
        import os

        value = os.path.relpath(str(Path(to_file).resolve()), str(Path(from_file).parent.resolve()))
        return value.replace('\\', '/') + suffix
    except ValueError:
        return Path(to_file).name + suffix


def _resolve_reference(value: str, from_file: Path, alias_map: Mapping[str, Path]) -> Optional[tuple[Path, str]]:
    if _is_ignored_reference(value):
        return None
    path_part, suffix = _split_ref(value)
    key = _normalize_ref_key(path_part)
    if not key:
        return None
    if key in alias_map:
        return alias_map[key], suffix
    basename = Path(key).name.lower()
    if basename in alias_map:
        return alias_map[basename], suffix
    direct = (from_file.parent / path_part).resolve(strict=False)
    for destination in alias_map.values():
        try:
            if direct == Path(destination).resolve(strict=False):
                return destination, suffix
        except OSError:
            continue
    return None


def _rewrite_ref(value: str, from_file: Path, alias_map: Mapping[str, Path], rewrites: list[dict[str, Any]], kind: str) -> str:
    resolved = _resolve_reference(value, from_file, alias_map)
    if not resolved:
        return value
    destination, suffix = resolved
    rewritten = _relative_path_between(from_file, destination, suffix)
    if rewritten != value:
        rewrites.append(
            {
                'file': str(from_file),
                'kind': kind,
                'original': value,
                'rewritten': rewritten,
            }
        )
    return rewritten


def _rewrite_srcset(value: str, from_file: Path, alias_map: Mapping[str, Path], rewrites: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    changed = False
    for raw_part in value.split(','):
        token = raw_part.strip()
        if not token:
            continue
        segments = token.split()
        if not segments:
            continue
        original_url = segments[0]
        rewritten_url = _rewrite_ref(original_url, from_file, alias_map, rewrites, 'html_srcset')
        if rewritten_url != original_url:
            changed = True
        parts.append(' '.join([rewritten_url, *segments[1:]]))
    return ', '.join(parts) if changed else value


HTML_ATTR_RE = re.compile(r'\b(?P<attr>src|href|poster|srcset)\s*=\s*(?P<quote>["\'])(?P<value>[^"\']*)(?P=quote)', re.IGNORECASE)
HTML_STYLE_DOUBLE_RE = re.compile(r'\b(?P<attr>style)\s*=\s*"(?P<value>[^"]*)"', re.IGNORECASE)
HTML_STYLE_SINGLE_RE = re.compile(r"\b(?P<attr>style)\s*=\s*'(?P<value>[^']*)'", re.IGNORECASE)
CSS_URL_RE = re.compile(r'url\(\s*(?P<quote>["\']?)(?P<value>[^)"\']+)(?P=quote)\s*\)', re.IGNORECASE)
CSS_IMPORT_RE = re.compile(r'@import\s+(?P<quote>["\'])(?P<value>[^"\']+)(?P=quote)', re.IGNORECASE)
JS_QUOTED_RE = re.compile(r'(?P<quote>["\'])(?P<value>[^"\']+\.(?:png|jpe?g|webp|gif|svg|css|js|mjs|wav|mp3|m4a|ogg|json))(?P=quote)', re.IGNORECASE)


def _iter_html_style_values(text: str) -> Iterable[str]:
    for match in HTML_STYLE_DOUBLE_RE.finditer(text):
        yield match.group('value')
    for match in HTML_STYLE_SINGLE_RE.finditer(text):
        yield match.group('value')


def _rewrite_css_urls(
    value: str,
    from_file: Path,
    alias_map: Mapping[str, Path],
    rewrites: list[dict[str, Any]],
    kind: str,
) -> str:
    def replace_css_url(match: re.Match[str]) -> str:
        rewritten = _rewrite_ref(match.group('value'), from_file, alias_map, rewrites, kind)
        quote = match.group('quote') or ''
        return f'url({quote}{rewritten}{quote})'

    return CSS_URL_RE.sub(replace_css_url, value)


def _rewrite_text_file(path: Path, alias_map: Mapping[str, Path]) -> list[dict[str, Any]]:
    extension = path.suffix.lower().lstrip('.')
    if extension not in HTML_EXTENSIONS | CSS_EXTENSIONS | JS_EXTENSIONS:
        return []
    try:
        original = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        original = path.read_text(encoding='utf-8', errors='replace')
    rewrites: list[dict[str, Any]] = []
    updated = original
    if extension in HTML_EXTENSIONS:
        def replace_html(match: re.Match[str]) -> str:
            value = match.group('value')
            attr = match.group('attr').lower()
            rewritten = (
                _rewrite_srcset(value, path, alias_map, rewrites)
                if attr == 'srcset'
                else _rewrite_ref(value, path, alias_map, rewrites, f'html_{attr}')
            )
            return f'{match.group("attr")}={match.group("quote")}{rewritten}{match.group("quote")}'

        updated = HTML_ATTR_RE.sub(replace_html, updated)

        def replace_style_double(match: re.Match[str]) -> str:
            rewritten = _rewrite_css_urls(match.group('value'), path, alias_map, rewrites, 'html_style_url')
            return f'{match.group("attr")}="{rewritten}"'

        def replace_style_single(match: re.Match[str]) -> str:
            rewritten = _rewrite_css_urls(match.group('value'), path, alias_map, rewrites, 'html_style_url')
            return f"{match.group('attr')}='{rewritten}'"

        updated = HTML_STYLE_DOUBLE_RE.sub(replace_style_double, updated)
        updated = HTML_STYLE_SINGLE_RE.sub(replace_style_single, updated)
    if extension in CSS_EXTENSIONS:
        def replace_css_import(match: re.Match[str]) -> str:
            value = match.group('value')
            rewritten = _rewrite_ref(value, path, alias_map, rewrites, 'css_import')
            return f'@import {match.group("quote")}{rewritten}{match.group("quote")}'

        updated = _rewrite_css_urls(updated, path, alias_map, rewrites, 'css_url')
        updated = CSS_IMPORT_RE.sub(replace_css_import, updated)
    if extension in JS_EXTENSIONS:
        def replace_js(match: re.Match[str]) -> str:
            value = match.group('value')
            resolved = _resolve_reference(value, path, alias_map)
            if not resolved:
                return match.group(0)
            rewritten = _rewrite_ref(value, path, alias_map, rewrites, 'js_asset_string')
            return f'{match.group("quote")}{rewritten}{match.group("quote")}'

        updated = JS_QUOTED_RE.sub(replace_js, updated)
    if updated != original:
        path.write_text(updated, encoding='utf-8')
    return rewrites


def _iter_local_refs(path: Path) -> Iterable[tuple[str, str]]:
    extension = path.suffix.lower().lstrip('.')
    if extension not in HTML_EXTENSIONS | CSS_EXTENSIONS | JS_EXTENSIONS:
        return
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='utf-8', errors='replace')
    if extension in HTML_EXTENSIONS:
        for match in HTML_ATTR_RE.finditer(text):
            attr = match.group('attr').lower()
            value = match.group('value')
            if attr == 'srcset':
                for raw_part in value.split(','):
                    token = raw_part.strip().split()
                    if token:
                        yield 'html_srcset', token[0]
            else:
                yield f'html_{attr}', value
        for value in _iter_html_style_values(text):
            for match in CSS_URL_RE.finditer(value):
                yield 'html_style_url', match.group('value')
    if extension in CSS_EXTENSIONS:
        for match in CSS_URL_RE.finditer(text):
            yield 'css_url', match.group('value')
        for match in CSS_IMPORT_RE.finditer(text):
            yield 'css_import', match.group('value')
    if extension in JS_EXTENSIONS:
        for match in JS_QUOTED_RE.finditer(text):
            yield 'js_asset_string', match.group('value')


def _artifact_for_dependency_path(path: Path, artifacts_by_path: Mapping[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        return None
    existing = artifacts_by_path.get(str(resolved))
    if existing:
        return dict(existing)
    if not resolved.exists() or not resolved.is_file():
        return None
    extension = resolved.suffix.lower().lstrip('.')
    artifact_type = 'file'
    if extension in IMAGE_EXTENSIONS:
        artifact_type = 'image'
    elif extension in AUDIO_EXTENSIONS:
        artifact_type = 'audio'
    elif extension in TEXT_EXTENSIONS:
        artifact_type = 'text'
    mime_type, _encoding = mimetypes.guess_type(str(resolved))
    normalized = sanitize_artifact_record(
        {
            'type': artifact_type,
            'kind': artifact_type,
            'path': str(resolved),
            'source_path': str(resolved),
            'name': resolved.stem,
            'mime_type': mime_type or None,
            'origin': 'linked_public_dependency',
            'source': 'linked_public_dependency',
        },
        include_content=True,
    )
    if not normalized:
        return None
    return {
        **normalized,
        'source_path': str(resolved),
        'type': artifact_type,
        'kind': artifact_type,
        'extension': extension,
        'basename': resolved.name,
    }


def _include_linked_bundle_dependencies(
    selected_artifacts: list[dict[str, Any]],
    all_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selected_artifacts:
        return selected_artifacts
    artifacts_by_path: dict[str, dict[str, Any]] = {}
    for artifact in all_artifacts:
        source_path = _resolve_existing_file(artifact.get('source_path') or artifact.get('path'))
        if source_path is not None:
            artifacts_by_path[str(source_path.resolve(strict=False))] = artifact

    result: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    queue: list[Path] = []
    for artifact in selected_artifacts:
        source_path = _resolve_existing_file(artifact.get('source_path') or artifact.get('path'))
        if source_path is not None:
            key = str(source_path.resolve(strict=False))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            if source_path.suffix.lower().lstrip('.') in HTML_EXTENSIONS | CSS_EXTENSIONS | JS_EXTENSIONS:
                queue.append(source_path)
        result.append(artifact)

    scanned: set[str] = set()
    while queue:
        source = queue.pop(0)
        source_key = str(source.resolve(strict=False))
        if source_key in scanned:
            continue
        scanned.add(source_key)
        for _kind, value in _iter_local_refs(source):
            if _is_ignored_reference(value):
                continue
            path_part, _suffix = _split_ref(value)
            if not path_part:
                continue
            try:
                dependency = (source.parent / path_part).expanduser().resolve(strict=False)
            except OSError:
                continue
            if not dependency.exists() or not dependency.is_file():
                continue
            dependency_key = str(dependency)
            if dependency_key in seen_paths:
                continue
            artifact = _artifact_for_dependency_path(dependency, artifacts_by_path)
            if not artifact:
                continue
            seen_paths.add(dependency_key)
            result.append(artifact)
            if dependency.suffix.lower().lstrip('.') in HTML_EXTENSIONS | CSS_EXTENSIONS | JS_EXTENSIONS:
                queue.append(dependency)
    return result


def _text_file_syntax_sanity_issues(path: Path) -> list[str]:
    extension = path.suffix.lower().lstrip('.')
    if extension not in SYNTAX_CHECK_EXTENSIONS:
        return []
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    try:
        from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner

        return ResponseSemanticsRuntimeOwner.text_artifact_syntax_sanity_issues_for_extension(
            extension,
            text,
        )[:6]
    except Exception:  # noqa: BLE001
        return []


def _link_check(bundle_dir: Path, copied_files: list[Path]) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    for path in copied_files:
        syntax_issues = _text_file_syntax_sanity_issues(path)
        if syntax_issues:
            missing.append(
                {
                    'file': str(path),
                    'kind': 'text_syntax_sanity',
                    'target': path.name,
                    'reason': 'syntax',
                    'issues': syntax_issues,
                }
            )
        for kind, value in _iter_local_refs(path):
            if _is_ignored_reference(value):
                continue
            path_part, _suffix = _split_ref(value)
            if not path_part:
                continue
            target = (path.parent / path_part).resolve(strict=False)
            try:
                target.relative_to(bundle_dir.resolve())
            except ValueError:
                missing.append({'file': str(path), 'kind': kind, 'target': value, 'reason': 'outside_bundle'})
                continue
            if not target.exists():
                missing.append({'file': str(path), 'kind': kind, 'target': value, 'reason': 'missing'})
    return {'status': 'passed' if not missing else 'failed', 'missing': missing}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if item not in (None, '', [], {})}
    if isinstance(value, list):
        return [_json_safe(item) for item in value if item not in (None, '', [], {})]
    if isinstance(value, Path):
        return str(value)
    return value


def _make_bundle_dir(
    response_payload: Mapping[str, Any],
    *,
    target_name: str | None,
    bundle_root: Path | str,
    created_at: str | None,
) -> Path:
    response_id = _safe_fragment(response_payload.get('id') or response_payload.get('response_id'), 'response')
    slug = _safe_fragment(target_name, '') if target_name else ''
    timestamp = _timestamp_token(created_at)
    folder = f'{timestamp}_package_{response_id}'
    if slug:
        folder = f'{folder}_{slug}'
    root = Path(bundle_root)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / folder
    index = 2
    while candidate.exists():
        candidate = root / f'{folder}_{index}'
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def bundle_response_artifacts(
    response_payload: Mapping[str, Any],
    *,
    target_name: str | None = None,
    bundle_root: Path | str = DEFAULT_BUNDLE_ROOT,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Copy saved response artifacts into a portable bundle and verify local links."""
    if not isinstance(response_payload, Mapping):
        raise ValueError('response_payload must be a mapping')
    existing_artifacts = _collect_existing_artifacts(response_payload)
    public_artifacts = _filter_public_bundle_artifacts(response_payload, existing_artifacts)
    artifacts = [
        item
        for item in _include_linked_bundle_dependencies(public_artifacts, existing_artifacts)
        if _is_bundleable_artifact(item)
    ]
    if not artifacts:
        raise ValueError('No existing local response artifacts were available to bundle.')

    entrypoint_artifact = _select_entrypoint(artifacts)
    bundle_dir = _make_bundle_dir(
        response_payload,
        target_name=target_name,
        bundle_root=bundle_root,
        created_at=created_at,
    )
    copied_artifacts: list[dict[str, Any]] = []
    copied_files: list[Path] = []
    used_paths: set[Path] = set()

    for artifact in artifacts:
        source = Path(_clean_text(artifact.get('source_path')))
        is_entrypoint = bool(entrypoint_artifact and artifact.get('source_path') == entrypoint_artifact.get('source_path'))
        subdir = _subdir_for_artifact(artifact, is_entrypoint=is_entrypoint)
        destination = _dedupe_path(bundle_dir / subdir / _safe_filename(source, artifact, is_entrypoint=is_entrypoint), used_paths)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        relative_path = destination.relative_to(bundle_dir).as_posix()
        record = {
            'type': artifact.get('type'),
            'kind': artifact.get('kind'),
            'artifact_ref': artifact.get('artifact_ref') or artifact.get('ref'),
            'name': artifact.get('name'),
            'source_path': str(source),
            'path': str(destination),
            'relative_path': relative_path,
            'extension': destination.suffix.lower().lstrip('.'),
            'sha256': _file_sha256(destination),
            'size_bytes': destination.stat().st_size,
        }
        copied_artifacts.append({key: value for key, value in record.items() if value not in (None, '', [], {})})
        copied_files.append(destination)

    alias_map = _build_alias_map(copied_artifacts)
    rewritten_links: list[dict[str, Any]] = []
    for path in copied_files:
        rewritten_links.extend(_rewrite_text_file(path, alias_map))

    link_check = _link_check(bundle_dir, copied_files)
    entrypoint_path = ''
    if entrypoint_artifact:
        for item in copied_artifacts:
            if item.get('source_path') == entrypoint_artifact.get('source_path'):
                entrypoint_path = _clean_text(item.get('path'))
                break
    source_response_id = _clean_text(response_payload.get('id') or response_payload.get('response_id'))
    source_refs = [
        _clean_text(item.get('artifact_ref') or item.get('ref'))
        for item in artifacts
        if _clean_text(item.get('artifact_ref') or item.get('ref'))
    ]
    digest_payload = json.dumps(
        {
            'source_response_id': source_response_id,
            'bundle_path': str(bundle_dir),
            'copied': [item.get('relative_path') for item in copied_artifacts],
        },
        sort_keys=True,
    )
    bundle_id = f'bundle:{hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()[:24]}'
    manifest_path = bundle_dir / 'manifest.json'
    payload = {
        'status': 'bundled' if link_check.get('status') == 'passed' else 'failed',
        'bundle_id': bundle_id,
        'bundle_path': str(bundle_dir),
        'entrypoint': entrypoint_path or None,
        'entrypoint_relative_path': (
            Path(entrypoint_path).relative_to(bundle_dir).as_posix()
            if entrypoint_path and Path(entrypoint_path).is_relative_to(bundle_dir)
            else None
        ),
        'manifest_path': str(manifest_path),
        'source_response_id': source_response_id or None,
        'source_artifact_refs': list(dict.fromkeys(source_refs)),
        'copied_artifacts': copied_artifacts,
        'rewritten_links': rewritten_links,
        'link_check': link_check,
    }
    manifest = {
        'kind': 'ollmo.response_artifact_bundle_manifest',
        'bundle_version': 1,
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
        **payload,
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return _json_safe(payload)


def build_response_artifact_bundle_registry_record(bundle_payload: Mapping[str, Any]) -> dict[str, Any]:
    bundle_id = _clean_text(bundle_payload.get('bundle_id'))
    artifact = {
        'type': 'bundle',
        'kind': 'response_artifact_bundle',
        'artifact_id': bundle_id.replace('bundle:', 'bundle_') if bundle_id else None,
        'artifact_ref': bundle_id or None,
        'path': bundle_payload.get('bundle_path'),
        'entrypoint': bundle_payload.get('entrypoint'),
        'manifest_path': bundle_payload.get('manifest_path'),
        'source_response_id': bundle_payload.get('source_response_id'),
        'source_artifact_refs': bundle_payload.get('source_artifact_refs'),
        'link_check': bundle_payload.get('link_check'),
    }
    record = {
        'kind': 'ollmo.artifact_registry_record',
        'artifact_registry_version': 1,
        'artifact_ref': bundle_id or None,
        'artifact': artifact,
        'bundle': dict(bundle_payload),
        'source': {
            'kind': 'response_artifact_bundle',
            'response_id': bundle_payload.get('source_response_id'),
        },
    }
    return _json_safe(record)
