#!/usr/bin/env python3
"""Build and verify deterministic Ollmo source release archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = Path('ollmo_core/version.py')
MANIFEST_NAME = 'MANIFEST.sha256'

# Explicit transport-safety bounds. They are intentionally far above the
# current source archive and prevent verify-only from becoming an extraction
# bomb. Changing them is a release-policy decision, not a hidden runtime cap.
MAX_COMPRESSED_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
MAX_RELEASE_FILE_BYTES = 128 * 1024 * 1024
MAX_RELEASE_TOTAL_BYTES = 1024 * 1024 * 1024

REQUIRED_ROOT_FILES = frozenset(
    {
        'CHANGELOG.md',
        'CITATION.cff',
        'CONTRIBUTING.md',
        'GHOST.md',
        'LICENSE',
        'NOTICE',
        'OLLMO_FOR_AGENTS.md',
        'README.md',
        'SECURITY.md',
        'THIRD_PARTY_NOTICES.md',
        'clean_repo_state.sh',
        'ollmo',
        'ollmo_webUI.html',
        'ollmo_webserver.py',
        'pytest.ini',
        'requirements.txt',
        'restart.sh',
        'start_multi_models.sh',
        'stop_multi_models.sh',
    }
)

PUBLIC_DOCS = frozenset(
    {
        'ARCHITECTURE_MAP.md',
        'BACKEND_FABRIC.md',
        'CANONICAL_GLOSSARY.md',
        'CANONICAL_STACK.md',
        'CONTROL_KNOBS.md',
        'CORE_CONTRACTS.md',
        'GHOST_ROUTER.md',
        'GHOST_SELF_ALIGNMENT.md',
        'KNOWN_LIMITATIONS.md',
        'KNOBS_CHEATSHEET.md',
        'PATTERNS.md',
        'PRINCIPLES.md',
        'RELEASE_SCOPE.md',
        'RESPONSES_CONTRACT.md',
        'TESTING_PROTOCOL.md',
        'TRUTH_SOURCES.md',
        'VISION_ALIGNMENT.md',
    }
)
CURRENT_DIAGRAMS = frozenset(
    {
        'ollmo-state-substrate-architecture.html',
    }
)
CURRENT_DIAGRAM_PATHS = frozenset(
    Path('docs/diagrams') / name for name in CURRENT_DIAGRAMS
)
PUBLIC_DOC_PATHS = frozenset(
    {Path('docs') / name for name in PUBLIC_DOCS} | set(CURRENT_DIAGRAM_PATHS)
)
RELEASE_SKILL_FILES = frozenset(
    {
        Path('skills/ollmo/SKILL.md'),
        Path('skills/ollmo/NOTICE'),
        Path('skills/ollmo/agents/openai.yaml'),
        Path('skills/ollmo/references/ollmo-contract.md'),
    }
)
RELEASE_SKILL_DIRECTORIES = frozenset(
    {
        Path('skills'),
        Path('skills/ollmo'),
        Path('skills/ollmo/agents'),
        Path('skills/ollmo/references'),
    }
)
TREE_SUFFIXES = {
    'helpers': frozenset({'.py'}),
    'ollmo_core': frozenset({'.py'}),
    'ollmo_g': frozenset({'.md', '.py'}),
    'ollmo_integrations': frozenset({'.py'}),
    'ollmo_orchestration': frozenset({'.py'}),
    'ollmo_runtime': frozenset({'.py'}),
    'ollmo_server': frozenset({'.py'}),
    'ollmo_services': frozenset({'.py'}),
    'scripts': frozenset({'.py'}),
    'site': frozenset(
        {
            '.css',
            '.html',
            '.js',
            '.md',
            '.txt',
            '.woff',
            '.woff2',
        }
    ),
    'static': frozenset(
        {
            '.avif',
            '.css',
            '.gif',
            '.html',
            '.ico',
            '.jpeg',
            '.jpg',
            '.js',
            '.json',
            '.png',
            '.svg',
            '.txt',
            '.webp',
            '.woff',
            '.woff2',
        }
    ),
    'tests': frozenset({'.json', '.py'}),
}

# Empty substrate only: the builder creates these directories but never copies
# any artifact file from the active checkout.
EMPTY_ARTIFACT_DIRECTORIES = frozenset(
    {
        Path('artifacts'),
        Path('artifacts/audio'),
        Path('artifacts/audits'),
        Path('artifacts/benchmarks'),
        Path('artifacts/bundles'),
        Path('artifacts/documents'),
        Path('artifacts/images'),
        Path('artifacts/inputs'),
        Path('artifacts/inputs/audio'),
        Path('artifacts/inputs/image'),
        Path('artifacts/inputs/pdf'),
        Path('artifacts/inputs/text'),
        Path('artifacts/manifests'),
        Path('artifacts/ocr'),
        Path('artifacts/settings'),
        Path('artifacts/transcripts'),
    }
)

REQUIRED_RELEASE_PATHS = frozenset(
    {
        Path('CHANGELOG.md'),
        Path('CITATION.cff'),
        Path('CONTRIBUTING.md'),
        Path('LICENSE'),
        Path('NOTICE'),
        Path('README.md'),
        Path('SECURITY.md'),
        Path('THIRD_PARTY_NOTICES.md'),
        Path('docs/KNOWN_LIMITATIONS.md'),
        Path('docs/RELEASE_SCOPE.md'),
        Path('docs/VISION_ALIGNMENT.md'),
        Path('ollmo'),
        Path('ollmo_core/version.py'),
        Path('ollmo_webUI.html'),
        Path('ollmo_webserver.py'),
        Path('requirements.txt'),
        Path('scripts/build_release_archive.py'),
        Path('site/index.html'),
        Path('site/landing.css'),
        Path('site/landing.js'),
        Path('site/fonts/Montserrat-Black-latin.woff2'),
        Path('site/fonts/OFL.txt'),
        Path('site/fonts/README.md'),
        Path('start_multi_models.sh'),
        Path('stop_multi_models.sh'),
    }
) | CURRENT_DIAGRAM_PATHS | RELEASE_SKILL_FILES

FORBIDDEN_ROOT_COMPONENTS = frozenset(
    {
        '.git',
        '.ollmo_archiv',
        '.pytest_cache',
        '.venv',
        '__pycache__',
        'artifacts',
        'dist',
        'logs',
        'plans',
        'state',
    }
)
FORBIDDEN_ANY_COMPONENTS = frozenset(
    {
        '.DS_Store',
        '.git',
        '.mypy_cache',
        '.pytest_cache',
        '.ruff_cache',
        '.venv',
        '__pycache__',
    }
)

TEXT_SUFFIXES = frozenset(
    {
        '.css',
        '.cff',
        '.html',
        '.ini',
        '.js',
        '.json',
        '.md',
        '.py',
        '.sh',
        '.toml',
        '.txt',
        '.yaml',
        '.yml',
    }
)
SECRET_PATTERNS = (
    (
        'private_key',
        re.compile(
            rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
            re.IGNORECASE,
        ),
    ),
    ('openai_key', re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}\b')),
    ('github_token', re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{20,}\b')),
    ('aws_access_key', re.compile(rb'\bAKIA[0-9A-Z]{16}\b')),
    (
        'slack_token',
        re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
    ),
)
PERSONAL_PATTERNS = (
    ('personal_home', re.compile(r'/Users/' + r'dev(?:/|\b)')),
    ('saved_build_reference', re.compile(r'ollmo_' + r'saved_builts')),
    ('personal_checkout', re.compile(r'Desktop/' + r'ollmo(?:/|\b)')),
)
VERSION_PATTERN = re.compile(
    r"^__version__\s*=\s*['\"](?P<version>[0-9]+\.[0-9]+\.[0-9]+)['\"]\s*$",
    re.MULTILINE,
)
VERSION_VALUE_PATTERN = re.compile(r'^[0-9]+\.[0-9]+\.[0-9]+$')
MANIFEST_LINE_PATTERN = re.compile(
    r'^(?P<digest>[0-9a-f]{64})  (?P<path>[^\r\n]+)$'
)


class ReleaseArchiveError(RuntimeError):
    """Raised when a release archive cannot be built or verified safely."""


def _ensure_relative_release_path(
    path: Path | PurePosixPath,
    *,
    allow_empty_artifact_directory: bool = False,
) -> None:
    parts = path.parts
    if not parts or path.is_absolute() or any(part in {'', '.', '..'} for part in parts):
        raise ReleaseArchiveError(f'Unsafe release path: {path}')
    if parts[0] in FORBIDDEN_ROOT_COMPONENTS and not (
        allow_empty_artifact_directory
        and Path(*parts) in EMPTY_ARTIFACT_DIRECTORIES
    ):
        raise ReleaseArchiveError(f'Forbidden release root: {path}')
    if any(
        part in FORBIDDEN_ANY_COMPONENTS or part.startswith('.')
        for part in parts
    ):
        raise ReleaseArchiveError(f'Forbidden hidden/cache path: {path}')


def _assert_regular_source_file(path: Path, *, relative_path: Path) -> None:
    if path.is_symlink():
        raise ReleaseArchiveError(f'Symlinks are not allowed: {relative_path}')
    if not path.is_file():
        raise ReleaseArchiveError(f'Required regular file is missing: {relative_path}')
    if path.stat().st_size > MAX_RELEASE_FILE_BYTES:
        raise ReleaseArchiveError(
            f'Release file exceeds the {MAX_RELEASE_FILE_BYTES}-byte safety '
            f'limit: {relative_path}'
        )


def read_source_version(source_root: Path) -> str:
    version_path = source_root / VERSION_FILE
    _assert_regular_source_file(version_path, relative_path=VERSION_FILE)
    try:
        text = version_path.read_text(encoding='utf-8')
    except OSError as exc:
        raise ReleaseArchiveError(f'Cannot read {VERSION_FILE}: {exc}') from exc
    match = VERSION_PATTERN.search(text)
    if not match:
        raise ReleaseArchiveError(
            f'{VERSION_FILE} must contain one semantic __version__ assignment.'
        )
    return match.group('version')


def _iter_tree_files(
    source_root: Path,
    directory: str,
    suffixes: frozenset[str],
) -> Iterable[tuple[Path, Path]]:
    root = source_root / directory
    if not root.is_dir() or root.is_symlink():
        raise ReleaseArchiveError(f'Required source directory is missing: {directory}')
    for source_path in sorted(root.rglob('*'), key=lambda item: item.as_posix()):
        relative_path = source_path.relative_to(source_root)
        if source_path.is_symlink():
            raise ReleaseArchiveError(f'Symlinks are not allowed: {relative_path}')
        if not source_path.is_file():
            continue
        if any(
            part in FORBIDDEN_ANY_COMPONENTS or part.startswith('.')
            for part in relative_path.parts
        ):
            continue
        if source_path.suffix.lower() not in suffixes:
            continue
        if source_path.stat().st_size > MAX_RELEASE_FILE_BYTES:
            raise ReleaseArchiveError(
                f'Release file exceeds the {MAX_RELEASE_FILE_BYTES}-byte '
                f'safety limit: {relative_path}'
            )
        _ensure_relative_release_path(relative_path)
        yield relative_path, source_path


def discover_release_files(source_root: Path) -> dict[Path, Path]:
    """Return the explicit source allowlist as release-relative paths."""

    source_root = source_root.resolve()
    selected: dict[Path, Path] = {}

    for name in sorted(REQUIRED_ROOT_FILES):
        relative_path = Path(name)
        source_path = source_root / relative_path
        _assert_regular_source_file(source_path, relative_path=relative_path)
        selected[relative_path] = source_path

    for name in sorted(PUBLIC_DOCS):
        relative_path = Path('docs') / name
        source_path = source_root / relative_path
        if source_path.exists():
            _assert_regular_source_file(source_path, relative_path=relative_path)
            selected[relative_path] = source_path

    for relative_path in sorted(
        CURRENT_DIAGRAM_PATHS,
        key=lambda item: item.as_posix(),
    ):
        source_path = source_root / relative_path
        _assert_regular_source_file(source_path, relative_path=relative_path)
        selected[relative_path] = source_path

    for relative_path in sorted(
        RELEASE_SKILL_FILES,
        key=lambda item: item.as_posix(),
    ):
        source_path = source_root / relative_path
        _assert_regular_source_file(source_path, relative_path=relative_path)
        _ensure_relative_release_path(relative_path)
        selected[relative_path] = source_path

    for directory, suffixes in TREE_SUFFIXES.items():
        for relative_path, source_path in _iter_tree_files(
            source_root,
            directory,
            suffixes,
        ):
            selected[relative_path] = source_path

    missing = sorted(path.as_posix() for path in REQUIRED_RELEASE_PATHS - selected.keys())
    if missing:
        raise ReleaseArchiveError(
            'Release allowlist is missing required paths: ' + ', '.join(missing)
        )
    total_bytes = sum(path.stat().st_size for path in selected.values())
    if total_bytes > MAX_RELEASE_TOTAL_BYTES:
        raise ReleaseArchiveError(
            f'Allowlisted source exceeds the {MAX_RELEASE_TOTAL_BYTES}-byte '
            'release safety limit.'
        )
    return dict(sorted(selected.items(), key=lambda item: item[0].as_posix()))


def _normalized_mode(source_path: Path) -> int:
    source_mode = stat.S_IMODE(source_path.stat().st_mode)
    return 0o755 if source_mode & stat.S_IXUSR else 0o644


def _write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    os.utime(path, (0, 0))


def stage_release_tree(
    source_root: Path,
    staging_root: Path,
    selected: dict[Path, Path],
) -> None:
    """Copy allowlisted sources into a new isolated staging tree."""

    if staging_root.exists():
        raise ReleaseArchiveError(f'Staging root already exists: {staging_root}')
    staging_root.mkdir(parents=True)

    for relative_path, source_path in selected.items():
        destination = staging_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        destination.chmod(_normalized_mode(source_path))
        os.utime(destination, (0, 0))

    # Never copy the active registry. A release starts with a truthful empty one.
    _write_bytes(staging_root / 'model_ports.json', b'[]\n')
    for relative_directory in sorted(
        EMPTY_ARTIFACT_DIRECTORIES,
        key=lambda item: (len(item.parts), item.as_posix()),
    ):
        (staging_root / relative_directory).mkdir(parents=True, exist_ok=True)

    for directory in sorted(
        (path for path in staging_root.rglob('*') if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o755)
        os.utime(directory, (0, 0))
    staging_root.chmod(0o755)
    os.utime(staging_root, (0, 0))


def _is_text_path(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        'LICENSE',
        'NOTICE',
        'ollmo',
    }


def _scan_release_file(relative_path: Path, data: bytes) -> None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise ReleaseArchiveError(
                f'Potential {label} found in {relative_path.as_posix()}.'
            )

    if not _is_text_path(relative_path):
        return
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        raise ReleaseArchiveError(
            f'Expected UTF-8 public text file is not valid UTF-8: {relative_path}'
        )
    for label, pattern in PERSONAL_PATTERNS:
        if pattern.search(text):
            raise ReleaseArchiveError(
                f'Potential {label} found in public file {relative_path.as_posix()}.'
            )


def _publication_gate_errors(
    release_root: Path,
    *,
    version: str,
    release_target: str | None,
) -> list[str]:
    errors: list[str] = []
    license_path = release_root / 'LICENSE'
    if not license_path.is_file() or not license_path.read_bytes().strip():
        errors.append('a reviewed non-empty LICENSE is required')
    if not str(release_target or '').strip():
        errors.append('an explicit --release-target is required')
    changelog_path = release_root / 'CHANGELOG.md'
    changelog = (
        changelog_path.read_text(encoding='utf-8')
        if changelog_path.is_file()
        else ''
    )
    unreleased_pattern = re.compile(
        rf'^## \[{re.escape(version)}\] - Unreleased\s*$',
        re.MULTILINE,
    )
    if unreleased_pattern.search(changelog):
        errors.append(f'CHANGELOG.md still marks {version} as Unreleased')
    return errors


def validate_release_tree(
    release_root: Path,
    *,
    version: str,
    publish: bool = False,
    release_target: str | None = None,
    expect_manifest: bool = False,
) -> list[Path]:
    """Validate structure, content safety, version, and publication gates."""

    if not release_root.is_dir() or release_root.is_symlink():
        raise ReleaseArchiveError(f'Invalid release root: {release_root}')

    files: list[Path] = []
    for path in sorted(release_root.rglob('*'), key=lambda item: item.as_posix()):
        relative_path = path.relative_to(release_root)
        _ensure_relative_release_path(
            relative_path,
            allow_empty_artifact_directory=(
                path.is_dir() and relative_path in EMPTY_ARTIFACT_DIRECTORIES
            ),
        )
        if path.is_symlink():
            raise ReleaseArchiveError(f'Symlinks are not allowed: {relative_path}')
        if relative_path.parts[0] == 'skills':
            allowed_paths = (
                RELEASE_SKILL_DIRECTORIES if path.is_dir() else RELEASE_SKILL_FILES
            )
            if relative_path not in allowed_paths:
                raise ReleaseArchiveError(
                    'Release tree contains a non-public skill path: '
                    f'{relative_path.as_posix()}'
                )
        if (
            not path.is_dir()
            and relative_path.parts[:2] == ('docs', 'diagrams')
            and relative_path not in CURRENT_DIAGRAM_PATHS
        ):
            raise ReleaseArchiveError(
                'Release tree contains a non-current diagram: '
                f'{relative_path.as_posix()}'
            )
        if (
            not path.is_dir()
            and relative_path.parts[0] == 'docs'
            and relative_path not in PUBLIC_DOC_PATHS
        ):
            raise ReleaseArchiveError(
                'Release tree contains a non-public documentation path: '
                f'{relative_path.as_posix()}'
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReleaseArchiveError(f'Unsupported staged entry: {relative_path}')
        files.append(relative_path)
        _scan_release_file(relative_path, path.read_bytes())

    actual_artifact_directories = {
        path.relative_to(release_root)
        for path in (release_root / 'artifacts').rglob('*')
        if path.is_dir()
    } | {Path('artifacts')}
    if actual_artifact_directories != set(EMPTY_ARTIFACT_DIRECTORIES):
        raise ReleaseArchiveError(
            'Release artifact skeleton does not match the standard empty '
            'bucket contract.'
        )

    required = set(REQUIRED_RELEASE_PATHS) | {Path('model_ports.json')}
    if expect_manifest:
        required.add(Path(MANIFEST_NAME))
    missing = sorted(path.as_posix() for path in required - set(files))
    if missing:
        raise ReleaseArchiveError(
            'Release tree is missing required files: ' + ', '.join(missing)
        )

    registry = release_root / 'model_ports.json'
    if registry.read_bytes() != b'[]\n':
        raise ReleaseArchiveError(
            'model_ports.json must be a generated empty registry, not runtime state.'
        )

    staged_version = read_source_version(release_root)
    if staged_version != version:
        raise ReleaseArchiveError(
            f'Version mismatch: archive={version}, source={staged_version}.'
        )

    if publish:
        gate_errors = _publication_gate_errors(
            release_root,
            version=version,
            release_target=release_target,
        )
        if gate_errors:
            raise ReleaseArchiveError(
                'Publish mode is blocked: ' + '; '.join(gate_errors) + '.'
            )
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(release_root: Path) -> Path:
    manifest_path = release_root / MANIFEST_NAME
    if manifest_path.exists():
        raise ReleaseArchiveError(f'Unexpected existing manifest: {manifest_path}')
    lines = []
    for path in sorted(release_root.rglob('*'), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative_path = path.relative_to(release_root)
        lines.append(f'{sha256_file(path)}  {relative_path.as_posix()}')
    _write_bytes(manifest_path, ('\n'.join(lines) + '\n').encode('utf-8'))
    return manifest_path


def _tar_info(name: str, *, is_dir: bool, mode: int, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ''
    info.gname = ''
    info.mtime = 0
    info.size = 0 if is_dir else size
    info.pax_headers = {}
    return info


def write_deterministic_archive(release_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open('wb') as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(
                fileobj=gz,
                mode='w',
                format=tarfile.PAX_FORMAT,
            ) as archive:
                root_name = release_root.name
                archive.addfile(
                    _tar_info(root_name, is_dir=True, mode=0o755)
                )
                for path in sorted(
                    release_root.rglob('*'),
                    key=lambda item: item.relative_to(release_root).as_posix(),
                ):
                    relative_path = path.relative_to(release_root)
                    archive_name = f'{root_name}/{relative_path.as_posix()}'
                    if path.is_dir():
                        archive.addfile(
                            _tar_info(archive_name, is_dir=True, mode=0o755)
                        )
                        continue
                    data = path.read_bytes()
                    mode = stat.S_IMODE(path.stat().st_mode)
                    info = _tar_info(
                        archive_name,
                        is_dir=False,
                        mode=mode,
                        size=len(data),
                    )
                    archive.addfile(info, fileobj=io.BytesIO(data))


def _safe_archive_relative_path(name: str) -> PurePosixPath:
    if not name or name.startswith('/') or '\\' in name:
        raise ReleaseArchiveError(f'Unsafe archive member name: {name!r}')
    path = PurePosixPath(name)
    if any(part in {'', '.', '..'} for part in path.parts):
        raise ReleaseArchiveError(f'Unsafe archive member name: {name!r}')
    return path


def _extract_verified_members(
    archive: tarfile.TarFile,
    members: Sequence[tarfile.TarInfo],
    destination: Path,
) -> None:
    for member in members:
        member_path = _safe_archive_relative_path(member.name)
        target = destination.joinpath(*member_path.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isreg():
            raise ReleaseArchiveError(
                f'Archive contains a non-regular entry: {member.name}'
            )
        handle = archive.extractfile(member)
        if handle is None:
            raise ReleaseArchiveError(f'Cannot read archive member: {member.name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as output:
            shutil.copyfileobj(handle, output)
        target.chmod(member.mode & 0o777)


def _parse_manifest(manifest_path: Path) -> dict[Path, str]:
    try:
        lines = manifest_path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseArchiveError(f'Cannot read {MANIFEST_NAME}: {exc}') from exc
    if not lines:
        raise ReleaseArchiveError(f'{MANIFEST_NAME} is empty.')
    records: dict[Path, str] = {}
    for line in lines:
        match = MANIFEST_LINE_PATTERN.fullmatch(line)
        if not match:
            raise ReleaseArchiveError(f'Malformed manifest line: {line!r}')
        posix_path = PurePosixPath(match.group('path'))
        _ensure_relative_release_path(posix_path)
        relative_path = Path(*posix_path.parts)
        if relative_path.name == MANIFEST_NAME:
            raise ReleaseArchiveError(f'{MANIFEST_NAME} must not hash itself.')
        if relative_path in records:
            raise ReleaseArchiveError(
                f'Duplicate manifest path: {relative_path.as_posix()}'
            )
        records[relative_path] = match.group('digest')
    if list(records) != sorted(records, key=lambda item: item.as_posix()):
        raise ReleaseArchiveError(f'{MANIFEST_NAME} paths are not sorted.')
    return records


def _verify_manifest(release_root: Path) -> None:
    records = _parse_manifest(release_root / MANIFEST_NAME)
    actual_files = {
        path.relative_to(release_root)
        for path in release_root.rglob('*')
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if set(records) != actual_files:
        missing = sorted(
            path.as_posix() for path in actual_files - set(records)
        )
        extra = sorted(path.as_posix() for path in set(records) - actual_files)
        details = []
        if missing:
            details.append('unlisted=' + ','.join(missing))
        if extra:
            details.append('missing=' + ','.join(extra))
        raise ReleaseArchiveError(
            f'{MANIFEST_NAME} does not exactly cover archive files: '
            + '; '.join(details)
        )
    for relative_path, expected_digest in records.items():
        actual_digest = sha256_file(release_root / relative_path)
        if actual_digest != expected_digest:
            raise ReleaseArchiveError(
                f'Hash mismatch for {relative_path.as_posix()}.'
            )


def _version_from_release_name(root_name: str) -> str:
    prefix = 'ollmo-'
    if not root_name.startswith(prefix):
        raise ReleaseArchiveError(
            f'Archive root must be named ollmo-<version>, got {root_name!r}.'
        )
    version = root_name[len(prefix):]
    if not VERSION_VALUE_PATTERN.fullmatch(version):
        raise ReleaseArchiveError(f'Invalid archive version: {version!r}.')
    return version


def verify_archive(
    archive_path: Path,
    *,
    expected_version: str | None = None,
    publish: bool = False,
    release_target: str | None = None,
) -> dict[str, str | int | bool]:
    """Verify an archive without writing to the active source checkout."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise ReleaseArchiveError(f'Archive does not exist: {archive_path}')
    compressed_size = archive_path.stat().st_size
    if compressed_size > MAX_COMPRESSED_ARCHIVE_BYTES:
        raise ReleaseArchiveError(
            f'Archive exceeds the {MAX_COMPRESSED_ARCHIVE_BYTES}-byte '
            'compressed safety limit.'
        )

    with tempfile.TemporaryDirectory(prefix='ollmo-release-verify-') as temp_dir:
        extraction_root = Path(temp_dir)
        try:
            with tarfile.open(archive_path, mode='r:gz') as archive:
                members = archive.getmembers()
                if not members:
                    raise ReleaseArchiveError('Archive is empty.')
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise ReleaseArchiveError(
                        f'Archive exceeds the {MAX_ARCHIVE_MEMBERS}-member '
                        'safety limit.'
                    )
                total_file_bytes = 0
                for member in members:
                    if member.size < 0:
                        raise ReleaseArchiveError(
                            f'Archive member has a negative size: {member.name}'
                        )
                    if member.isfile() and member.size > MAX_RELEASE_FILE_BYTES:
                        raise ReleaseArchiveError(
                            f'Archive member exceeds the '
                            f'{MAX_RELEASE_FILE_BYTES}-byte safety limit: '
                            f'{member.name}'
                        )
                    if member.isfile():
                        total_file_bytes += member.size
                if total_file_bytes > MAX_RELEASE_TOTAL_BYTES:
                    raise ReleaseArchiveError(
                        f'Archive exceeds the {MAX_RELEASE_TOTAL_BYTES}-byte '
                        'expanded safety limit.'
                    )
                names = [member.name for member in members]
                if len(names) != len(set(names)):
                    raise ReleaseArchiveError('Archive contains duplicate members.')
                parsed_paths = [_safe_archive_relative_path(name) for name in names]
                root_names = {path.parts[0] for path in parsed_paths}
                if len(root_names) != 1:
                    raise ReleaseArchiveError(
                        'Archive must contain exactly one top-level directory.'
                    )
                root_name = next(iter(root_names))
                version = _version_from_release_name(root_name)
                if expected_version and version != expected_version:
                    raise ReleaseArchiveError(
                        f'Expected version {expected_version}, found {version}.'
                    )
                _extract_verified_members(archive, members, extraction_root)
        except (tarfile.TarError, EOFError, OSError) as exc:
            if isinstance(exc, ReleaseArchiveError):
                raise
            raise ReleaseArchiveError(f'Cannot read archive: {exc}') from exc

        release_root = extraction_root / root_name
        validate_release_tree(
            release_root,
            version=version,
            publish=publish,
            release_target=release_target,
            expect_manifest=True,
        )
        _verify_manifest(release_root)
        file_count = sum(1 for path in release_root.rglob('*') if path.is_file())

    return {
        'status': 'verified',
        'version': version,
        'file_count': file_count,
        'archive_sha256': sha256_file(archive_path),
        'publish_mode': publish,
    }


def _validate_output_location(source_root: Path, output_dir: Path) -> None:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source_root or source_root.is_relative_to(output_dir):
        raise ReleaseArchiveError(
            'Output directory cannot be the source root or one of its parents.'
        )


def _remove_exact_generated_target(path: Path, *, parent: Path, name: str) -> None:
    if path.parent != parent or path.name != name:
        raise ReleaseArchiveError(f'Refusing to replace unexpected target: {path}')
    if not path.exists():
        return
    if path.is_symlink():
        raise ReleaseArchiveError(f'Refusing to replace symlink target: {path}')
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()
    else:
        raise ReleaseArchiveError(f'Refusing to replace special target: {path}')


def build_release_archive(
    *,
    source_root: Path,
    output_dir: Path,
    version: str | None = None,
    publish: bool = False,
    release_target: str | None = None,
) -> dict[str, str | int | bool]:
    """Build, verify, and atomically publish one local archive under output_dir."""

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    _validate_output_location(source_root, output_dir)
    source_version = read_source_version(source_root)
    requested_version = version or source_version
    if not VERSION_VALUE_PATTERN.fullmatch(requested_version):
        raise ReleaseArchiveError(f'Invalid requested version: {requested_version!r}.')
    if requested_version != source_version:
        raise ReleaseArchiveError(
            f'Requested version {requested_version} does not match '
            f'{VERSION_FILE} ({source_version}).'
        )

    selected = discover_release_files(source_root)
    release_name = f'ollmo-{requested_version}'
    archive_name = f'{release_name}.tar.gz'
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_container = Path(
        tempfile.mkdtemp(prefix=f'.{release_name}.build-', dir=output_dir)
    )
    try:
        staged_root = temp_container / release_name
        staged_archive = temp_container / archive_name
        stage_release_tree(source_root, staged_root, selected)
        validate_release_tree(
            staged_root,
            version=requested_version,
            publish=publish,
            release_target=release_target,
        )
        write_manifest(staged_root)
        validate_release_tree(
            staged_root,
            version=requested_version,
            publish=publish,
            release_target=release_target,
            expect_manifest=True,
        )
        _verify_manifest(staged_root)
        write_deterministic_archive(staged_root, staged_archive)
        verification = verify_archive(
            staged_archive,
            expected_version=requested_version,
            publish=publish,
            release_target=release_target,
        )

        final_root = output_dir / release_name
        final_archive = output_dir / archive_name
        _remove_exact_generated_target(
            final_root,
            parent=output_dir,
            name=release_name,
        )
        _remove_exact_generated_target(
            final_archive,
            parent=output_dir,
            name=archive_name,
        )
        shutil.move(str(staged_root), str(final_root))
        os.replace(staged_archive, final_archive)
    finally:
        shutil.rmtree(temp_container, ignore_errors=True)

    return {
        **verification,
        'mode': 'publish' if publish else 'release_candidate',
        'archive': str(final_archive),
        'staging_root': str(final_root),
        'publication_ready': publish,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build or verify a deterministic Ollmo source archive.'
    )
    parser.add_argument(
        '--source-root',
        type=Path,
        default=REPO_ROOT,
        help='Ollmo source root (default: the parent of this script).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('dist'),
        help='Build output directory (default: dist).',
    )
    parser.add_argument('--version', help='Expected semantic version.')
    parser.add_argument(
        '--verify-only',
        type=Path,
        metavar='ARCHIVE',
        help='Verify an existing archive without building.',
    )
    parser.add_argument(
        '--publish',
        action='store_true',
        help='Enforce final LICENSE, changelog, and release-target gates.',
    )
    parser.add_argument(
        '--release-target',
        help='Explicit publication destination label required by --publish.',
    )
    return parser


def _print_result(result: dict[str, str | int | bool]) -> None:
    for key in (
        'status',
        'mode',
        'version',
        'file_count',
        'archive',
        'staging_root',
        'archive_sha256',
        'publication_ready',
        'publish_mode',
    ):
        if key in result:
            print(f'{key}={str(result[key]).lower() if isinstance(result[key], bool) else result[key]}')


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.release_target and not args.publish:
        parser.error('--release-target is only valid together with --publish.')
    try:
        if args.verify_only:
            result = verify_archive(
                args.verify_only,
                expected_version=args.version,
                publish=args.publish,
                release_target=args.release_target,
            )
            result['mode'] = 'publish' if args.publish else 'release_candidate'
            result['publication_ready'] = bool(args.publish)
        else:
            result = build_release_archive(
                source_root=args.source_root,
                output_dir=args.output_dir,
                version=args.version,
                publish=args.publish,
                release_target=args.release_target,
            )
    except ReleaseArchiveError as exc:
        parser.exit(1, f'error: {exc}\n')
    _print_result(result)
    if not args.publish:
        print(
            'publication_gate=LICENSE, final changelog date, and explicit '
            'publication target remain required'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
