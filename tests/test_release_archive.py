from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from scripts import build_release_archive as release


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    path.chmod(0o755 if executable else 0o644)


def _make_release_source(tmp_path: Path) -> Path:
    source = tmp_path / 'source'
    source.mkdir()

    root_content = {
        'CHANGELOG.md': '# Changelog\n\n## [0.1.0] - Unreleased\n',
        'CITATION.cff': 'cff-version: 1.2.0\ntitle: Ollmo\n',
        'CONTRIBUTING.md': '# Contributing\n',
        'GHOST.md': '# Ghost\n',
        'LICENSE': 'Fixture Apache License 2.0 text.\n',
        'NOTICE': 'Ollmo\nCopyright 2025-2026 fl0ri0\n',
        'OLLMO_FOR_AGENTS.md': '# Ollmo for agents\n',
        'README.md': '# Ollmo\n\nRelease candidate.\n',
        'SECURITY.md': '# Security\n',
        'THIRD_PARTY_NOTICES.md': '# Third-party components\n',
        'clean_repo_state.sh': '#!/usr/bin/env bash\nexit 0\n',
        'ollmo': '#!/usr/bin/env bash\nexit 0\n',
        'ollmo_webUI.html': '<!doctype html><title>Ollmo</title>\n',
        'ollmo_webserver.py': '"""Fixture webserver."""\n',
        'pytest.ini': '[pytest]\ntestpaths = tests\n',
        'requirements.txt': 'Flask\npytest\n',
        'restart.sh': '#!/usr/bin/env bash\nexit 0\n',
        'start_multi_models.sh': '#!/usr/bin/env bash\nexit 0\n',
        'stop_multi_models.sh': '#!/usr/bin/env bash\nexit 0\n',
    }
    executable_names = {
        'clean_repo_state.sh',
        'ollmo',
        'restart.sh',
        'start_multi_models.sh',
        'stop_multi_models.sh',
    }
    for name, content in root_content.items():
        _write(source / name, content, executable=name in executable_names)

    _write(
        source / 'docs' / 'RELEASE_SCOPE.md',
        '# Release scope\n',
    )
    _write(
        source / 'docs' / 'KNOWN_LIMITATIONS.md',
        '# Known limitations\n',
    )
    _write(
        source / 'docs' / 'BENCHMARK_PLAN.md',
        '# Benchmark plan\n',
    )
    _write(
        source / 'docs' / 'LEGACY_ORCHESTRATION.md',
        '# Internal legacy-orchestration note\n',
    )
    _write(
        source / 'docs' / 'RELEASE_CHECK_0.1.0.md',
        '# Release check\n',
    )
    _write(
        source / 'docs' / 'WEB_FRAMEWORK_DECISION.md',
        '# Internal web-framework decision\n',
    )
    for relative_path in release.CURRENT_DIAGRAM_PATHS:
        _write(source / relative_path, f'Current diagram: {relative_path.name}\n')
    _write(
        source
        / 'docs'
        / 'diagrams'
        / 'ollmo-state-substrate-architecture 2.html',
        'Historical diagram: excluded\n',
    )
    _write(source / 'docs' / 'IDEAS.md', '# Internal ideas: excluded\n')
    _write(
        source / 'docs' / 'PRODUCT_DIRECTION.md',
        '# Internal product direction: excluded\n',
    )
    _write(
        source / 'docs' / 'VISION_ALIGNMENT.md',
        '# Internal vision alignment: excluded\n',
    )
    _write(
        source / 'ollmo_core' / 'version.py',
        '"""Version."""\n\n__version__ = \'0.1.0\'\n',
    )

    for directory in (
        'helpers',
        'ollmo_core',
        'ollmo_g',
        'ollmo_integrations',
        'ollmo_orchestration',
        'ollmo_runtime',
        'ollmo_server',
        'ollmo_services',
    ):
        _write(source / directory / '__init__.py', '"""Fixture package."""\n')

    _write(
        source / 'scripts' / 'build_release_archive.py',
        '"""Fixture copy of the release builder."""\n',
    )
    _write(source / 'scripts' / 'helper.py', 'VALUE = 1\n')
    _write(
        source / 'skills' / 'ollmo' / 'SKILL.md',
        (
            '---\n'
            'name: ollmo\n'
            'description: Use Ollmo runtime truth.\n'
            '---\n\n'
            '# Ollmo\n'
        ),
    )
    _write(
        source / 'skills' / 'ollmo' / 'NOTICE',
        'Ollmo Companion Skill\nCopyright 2026 fl0ri0\n',
    )
    _write(
        source / 'skills' / 'ollmo' / 'agents' / 'openai.yaml',
        (
            'interface:\n'
            '  display_name: "Ollmo"\n'
            '  short_description: "Use Ollmo runtime truth"\n'
            '  default_prompt: "Use $ollmo for local runtime work."\n'
        ),
    )
    _write(
        source / 'skills' / 'ollmo' / 'references' / 'ollmo-contract.md',
        '# Ollmo contract\n',
    )
    _write(
        source / 'skills' / 'ollmo' / 'private-note.md',
        '# Unreviewed skill note: excluded\n',
    )
    _write(
        source / 'skills' / 'ollmo-run-monitor' / 'SKILL.md',
        (
            '---\n'
            'name: ollmo-run-monitor\n'
            'description: Internal monitor skill.\n'
            '---\n'
        ),
    )
    _write(
        source / 'skills' / 'ollmo-run-monitor' / 'agents' / 'openai.yaml',
        'interface:\n  display_name: "Ollmo Run Monitor"\n',
    )
    _write(source / 'static' / 'ui' / 'app.js', 'window.ollmo = true;\n')
    _write(
        source / 'site' / 'index.html',
        (
            '<!doctype html>\n'
            '<link rel="stylesheet" href="./landing.css">\n'
            '<script defer src="./landing.js"></script>\n'
            '<title>Ollmo project page</title>\n'
        ),
    )
    _write(
        source / 'site' / 'landing.css',
        '@font-face { src: url("./fonts/Montserrat-Black-latin.woff2"); }\n',
    )
    _write(
        source / 'site' / 'landing.js',
        'window.ollmoLanding = true;\n',
    )
    _write(
        source / 'site' / 'fonts' / 'Montserrat-Black-latin.woff2',
        'fixture webfont bytes\n',
    )
    _write(
        source / 'site' / 'fonts' / 'OFL.txt',
        'SIL Open Font License fixture\n',
    )
    _write(
        source / 'site' / 'fonts' / 'README.md',
        '# Checkout-only font source note\n',
    )
    _write(source / 'tests' / 'test_smoke.py', 'def test_smoke():\n    assert True\n')
    _write(
        source
        / 'tests'
        / 'testdata'
        / 'ghost_diagnostics'
        / 'tts_capability_hint_summary_read_aloud.json',
        '{"case_id":"tts_capability_hint_summary_read_aloud"}\n',
    )
    _write(
        source / 'tests' / 'test_path_redaction_fixture.py',
        "SYNTHETIC_PATH = '/Users/example/example.txt'\n",
    )

    _write(
        source / 'model_ports.json',
        '[{"instance_id":"private-live-instance","port":12345}]\n',
    )
    _write(source / 'state' / 'runtime_status.json', '{"private": true}\n')
    _write(source / 'logs' / 'runtime.log', 'private log\n')
    _write(source / 'artifacts' / 'images' / 'private.txt', 'private artifact\n')
    _write(source / 'plans' / 'private-plan.md', '# Private plan\n')
    _write(source / '.env', 'SHOULD_NOT_ENTER_ARCHIVE=1\n')
    _write(source / 'Readme_current_building_state.md', '# Private build note\n')
    return source


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_stages_only_allowlisted_clean_release_files(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
        version='0.1.0',
    )

    archive_path = Path(str(result['archive']))
    staged_root = Path(str(result['staging_root']))
    assert archive_path.is_file()
    assert staged_root.is_dir()
    assert (staged_root / 'model_ports.json').read_text(encoding='utf-8') == '[]\n'
    assert not (staged_root / 'state').exists()
    assert not (staged_root / 'logs').exists()
    assert {
        path.relative_to(staged_root)
        for path in (staged_root / 'artifacts').rglob('*')
        if path.is_dir()
    } | {Path('artifacts')} == set(release.EMPTY_ARTIFACT_DIRECTORIES)
    assert not any(path.is_file() for path in (staged_root / 'artifacts').rglob('*'))
    assert not (staged_root / 'plans').exists()
    assert not (staged_root / '.env').exists()
    assert not (staged_root / 'Readme_current_building_state.md').exists()
    assert not (staged_root / 'docs' / 'IDEAS.md').exists()
    assert not (staged_root / 'docs' / 'BENCHMARK_PLAN.md').exists()
    assert not (staged_root / 'docs' / 'LEGACY_ORCHESTRATION.md').exists()
    assert not (staged_root / 'docs' / 'PRODUCT_DIRECTION.md').exists()
    assert not (staged_root / 'docs' / 'RELEASE_CHECK_0.1.0.md').exists()
    assert not (staged_root / 'docs' / 'VISION_ALIGNMENT.md').exists()
    assert not (staged_root / 'docs' / 'WEB_FRAMEWORK_DECISION.md').exists()
    staged_diagrams = {
        path.relative_to(staged_root)
        for path in (staged_root / 'docs' / 'diagrams').rglob('*')
        if path.is_file()
    }
    assert staged_diagrams == set(release.CURRENT_DIAGRAM_PATHS)
    assert not (staged_root / '.github').exists()
    assert (staged_root / 'tests' / 'test_smoke.py').is_file()
    assert not (staged_root / 'ollmo_landing.html').exists()
    assert (staged_root / 'site' / 'index.html').is_file()
    assert (staged_root / 'site' / 'landing.css').is_file()
    assert (staged_root / 'site' / 'landing.js').is_file()
    assert (
        staged_root
        / 'site'
        / 'fonts'
        / 'Montserrat-Black-latin.woff2'
    ).is_file()
    assert (staged_root / 'site' / 'fonts' / 'OFL.txt').is_file()
    assert (staged_root / 'site' / 'fonts' / 'README.md').is_file()
    for legal_path in (
        'CITATION.cff',
        'CONTRIBUTING.md',
        'LICENSE',
        'NOTICE',
        'THIRD_PARTY_NOTICES.md',
    ):
        assert (staged_root / legal_path).is_file()
    assert (
        staged_root
        / 'tests'
        / 'testdata'
        / 'ghost_diagnostics'
        / 'tts_capability_hint_summary_read_aloud.json'
    ).is_file()
    staged_skill_files = {
        path.relative_to(staged_root)
        for path in (staged_root / 'skills').rglob('*')
        if path.is_file()
    }
    assert staged_skill_files == set(release.RELEASE_SKILL_FILES)
    assert not (staged_root / 'skills' / 'ollmo' / 'private-note.md').exists()
    assert not (staged_root / 'skills' / 'ollmo-run-monitor').exists()

    verified = release.verify_archive(
        archive_path,
        expected_version='0.1.0',
    )
    assert verified['status'] == 'verified'
    assert verified['publish_mode'] is False


def test_manifest_exactly_covers_release_files_and_hashes(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))

    records = release._parse_manifest(staged_root / release.MANIFEST_NAME)
    actual = {
        path.relative_to(staged_root)
        for path in staged_root.rglob('*')
        if path.is_file() and path.name != release.MANIFEST_NAME
    }
    assert set(records) == actual
    assert all(
        release.sha256_file(staged_root / relative_path) == digest
        for relative_path, digest in records.items()
    )


@pytest.mark.parametrize(
    'relative_path',
    sorted(release.RELEASE_SKILL_FILES, key=lambda item: item.as_posix()),
    ids=lambda item: item.as_posix(),
)
def test_each_release_ollmo_skill_file_is_required(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    (source / relative_path).unlink()

    with pytest.raises(release.ReleaseArchiveError, match='missing'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


@pytest.mark.parametrize(
    'relative_path',
    (
        Path('site/index.html'),
        Path('site/landing.css'),
        Path('site/landing.js'),
        Path('site/fonts/Montserrat-Black-latin.woff2'),
        Path('site/fonts/OFL.txt'),
    ),
    ids=lambda item: item.as_posix(),
)
def test_each_standalone_landing_dependency_is_required(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    (source / relative_path).unlink()

    with pytest.raises(release.ReleaseArchiveError, match='missing'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


@pytest.mark.parametrize(
    'relative_path',
    sorted(release.CURRENT_DIAGRAM_PATHS, key=lambda item: item.as_posix()),
    ids=lambda item: item.as_posix(),
)
def test_each_current_release_diagram_is_required(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    (source / relative_path).unlink()

    with pytest.raises(release.ReleaseArchiveError, match='missing'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


def test_verify_only_rejects_non_current_diagram(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))
    (staged_root / release.MANIFEST_NAME).unlink()
    _write(
        staged_root
        / 'docs'
        / 'diagrams'
        / 'ollmo-state-substrate-architecture 2.html',
        'Historical diagram: forbidden in release\n',
    )
    release.write_manifest(staged_root)
    archive_path = tmp_path / 'non-current-diagram.tar.gz'
    release.write_deterministic_archive(staged_root, archive_path)

    with pytest.raises(release.ReleaseArchiveError, match='non-current diagram'):
        release.verify_archive(archive_path)


def test_other_hidden_release_metadata_remains_forbidden(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))
    (staged_root / release.MANIFEST_NAME).unlink()
    _write(staged_root / '.github' / 'private.txt', 'not public\n')
    release.write_manifest(staged_root)
    archive_path = tmp_path / 'unexpected-hidden-metadata.tar.gz'
    release.write_deterministic_archive(staged_root, archive_path)

    with pytest.raises(
        release.ReleaseArchiveError,
        match='Forbidden hidden/cache path',
    ):
        release.verify_archive(archive_path)


def test_verify_only_rejects_incomplete_standalone_landing_page(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))
    (staged_root / release.MANIFEST_NAME).unlink()
    (staged_root / 'site' / 'landing.css').unlink()
    release.write_manifest(staged_root)
    archive_path = tmp_path / 'missing-landing-css.tar.gz'
    release.write_deterministic_archive(staged_root, archive_path)

    with pytest.raises(
        release.ReleaseArchiveError,
        match='missing required files',
    ):
        release.verify_archive(archive_path)


def test_verify_only_rejects_non_public_skill_path(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))
    (staged_root / release.MANIFEST_NAME).unlink()
    _write(
        staged_root / 'skills' / 'ollmo-run-monitor' / 'SKILL.md',
        (
            '---\n'
            'name: ollmo-run-monitor\n'
            'description: Internal monitor skill.\n'
            '---\n'
        ),
    )
    release.write_manifest(staged_root)
    archive_path = tmp_path / 'unexpected-skill.tar.gz'
    release.write_deterministic_archive(staged_root, archive_path)

    with pytest.raises(
        release.ReleaseArchiveError,
        match='non-public skill path',
    ):
        release.verify_archive(archive_path)


def test_verify_only_rejects_missing_required_ollmo_skill(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))
    (staged_root / release.MANIFEST_NAME).unlink()
    (staged_root / 'skills' / 'ollmo' / 'SKILL.md').unlink()
    release.write_manifest(staged_root)
    archive_path = tmp_path / 'missing-skill.tar.gz'
    release.write_deterministic_archive(staged_root, archive_path)

    with pytest.raises(
        release.ReleaseArchiveError,
        match='missing required files',
    ):
        release.verify_archive(archive_path)


def test_archive_bytes_are_reproducible_across_output_directories(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    first = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist-a',
    )
    for path in source.rglob('*'):
        if path.is_file():
            path.touch()
    second = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist-b',
    )

    first_archive = Path(str(first['archive']))
    second_archive = Path(str(second['archive']))
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first['archive_sha256'] == second['archive_sha256']


def test_build_does_not_modify_allowlisted_or_private_source_files(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    before = {
        path.relative_to(source): _file_digest(path)
        for path in source.rglob('*')
        if path.is_file()
    }

    release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )

    after = {
        path.relative_to(source): _file_digest(path)
        for path in source.rglob('*')
        if path.is_file()
    }
    assert after == before


def test_verify_only_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist',
    )
    staged_root = Path(str(result['staging_root']))
    (staged_root / 'README.md').write_text('# Tampered\n', encoding='utf-8')
    tampered_archive = tmp_path / 'tampered.tar.gz'
    release.write_deterministic_archive(staged_root, tampered_archive)

    with pytest.raises(release.ReleaseArchiveError, match='Hash mismatch'):
        release.verify_archive(tampered_archive)


def test_verify_only_rejects_path_traversal_without_extracting_it(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(archive_path, mode='w:gz') as archive:
        data = b'escape'
        member = tarfile.TarInfo('ollmo-0.1.0/../../escape.txt')
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    with pytest.raises(release.ReleaseArchiveError, match='Unsafe archive member'):
        release.verify_archive(archive_path)
    assert not (tmp_path / 'escape.txt').exists()


def test_release_candidate_requires_the_reviewed_project_license(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    (source / 'LICENSE').unlink()

    with pytest.raises(release.ReleaseArchiveError, match='Required regular file'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist-rc',
        )


def test_publish_mode_requires_target_and_final_changelog_date(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    with pytest.raises(release.ReleaseArchiveError, match='release-target'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist-no-target',
            publish=True,
        )

    with pytest.raises(release.ReleaseArchiveError, match='Unreleased'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist-unreleased',
            publish=True,
            release_target='fixture-host',
        )

    _write(
        source / 'CHANGELOG.md',
        '# Changelog\n\n## [0.1.0] - 2026-07-30\n',
    )
    result = release.build_release_archive(
        source_root=source,
        output_dir=tmp_path / 'dist-publish',
        publish=True,
        release_target='fixture-host',
    )
    assert result['publication_ready'] is True
    release.verify_archive(
        Path(str(result['archive'])),
        publish=True,
        release_target='fixture-host',
    )


def test_public_personal_path_reference_blocks_build(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    personal_path = '/Users/' + 'dev/private'
    _write(source / 'README.md', f'# Ollmo\n\nPath: {personal_path}\n')

    with pytest.raises(release.ReleaseArchiveError, match='personal_home'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


def test_personal_path_in_other_test_fixture_blocks_build(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    personal_path = '/Users/' + 'dev/private.txt'
    _write(
        source / 'tests' / 'test_path_redaction_fixture.py',
        f'SYNTHETIC_PATH = {personal_path!r}\n',
    )

    with pytest.raises(release.ReleaseArchiveError, match='personal_home'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


def test_personal_path_in_release_skill_blocks_build(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    personal_path = '/Users/' + 'dev/private'
    _write(
        source / 'skills' / 'ollmo' / 'SKILL.md',
        (
            '---\n'
            'name: ollmo\n'
            'description: Use Ollmo runtime truth.\n'
            '---\n\n'
            f'Private checkout: {personal_path}\n'
        ),
    )

    with pytest.raises(release.ReleaseArchiveError, match='personal_home'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


def test_secret_pattern_in_any_included_file_blocks_build(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    fake_key = 'sk-' + ('A' * 24)
    _write(source / 'helpers' / 'leak.py', f"LEAK = '{fake_key}'\n")

    with pytest.raises(release.ReleaseArchiveError, match='openai_key'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
        )


def test_requested_version_must_match_single_source_of_truth(
    tmp_path: Path,
) -> None:
    source = _make_release_source(tmp_path)
    with pytest.raises(release.ReleaseArchiveError, match='does not match'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path / 'dist',
            version='0.2.0',
        )


def test_output_directory_cannot_contain_source_root(tmp_path: Path) -> None:
    source = _make_release_source(tmp_path)
    with pytest.raises(release.ReleaseArchiveError, match='one of its parents'):
        release.build_release_archive(
            source_root=source,
            output_dir=tmp_path,
        )
