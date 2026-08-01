import json
from pathlib import Path

from ollmo_g.reset_learning_state import reset_ghost_learning_state


def test_reset_ghost_learning_state_archives_legacy_files_and_preserves_response_frames(tmp_path: Path):
    state_dir = tmp_path / 'state'
    archive_root = state_dir / 'ghost_learning_archives' / 'test-reset'
    response_frames_dir = state_dir / 'response_frames'
    response_frames_dir.mkdir(parents=True)
    readiness_registry = state_dir / 'graph_rebase' / 'readiness_observations.jsonl'
    readiness_registry.parent.mkdir(parents=True)
    readiness_registry.write_text('{"kind":"ollmo.graph_rebase_readiness_registry_record"}\n', encoding='utf-8')

    event_log = state_dir / 'events.jsonl'
    event_log.write_text('{"kind":"legacy"}\n', encoding='utf-8')

    runtime_log = tmp_path / 'logs' / 'flask_webserver.log'
    runtime_log.parent.mkdir(parents=True)
    runtime_log.write_text('legacy router timeout\n', encoding='utf-8')

    learned_policy = state_dir / 'ghost_learned_policy.json'
    learned_policy.write_text('{"policies":[{"policy_id":"legacy"}]}\n', encoding='utf-8')

    learned_policy_markdown = state_dir / 'ghost_learned_policy.md'
    learned_policy_markdown.write_text('# legacy\n', encoding='utf-8')

    compiled_memory = state_dir / 'ghost_compiled_memory.json'
    compiled_memory.write_text('{"summary":{"recent_learning_count":4}}\n', encoding='utf-8')

    compiled_memory_markdown = state_dir / 'ghost_compiled_memory.md'
    compiled_memory_markdown.write_text('# compiled\n', encoding='utf-8')

    self_learning_dir = state_dir / 'self_learning'
    self_learning_dir.mkdir()
    (self_learning_dir / 'accepted_policy_snapshot.json').write_text('{"enabled": false}\n', encoding='utf-8')

    response_ledger = response_frames_dir / 'responses.jsonl'
    response_ledger.write_text('{"kind":"ollmo.response_frame"}\n', encoding='utf-8')

    payload = reset_ghost_learning_state(
        archive_root=archive_root,
        event_log_path=event_log,
        runtime_log_path=runtime_log,
        compiled_memory_path=compiled_memory,
        compiled_memory_markdown_path=compiled_memory_markdown,
        self_learning_dir_path=self_learning_dir,
        response_frame_ledger_path=response_ledger,
        graph_rebase_readiness_registry_path=readiness_registry,
    )

    assert payload['ok'] is True
    assert Path(payload['archive_dir']) == archive_root
    assert payload['preserved_paths']['response_frame_ledger'] == str(response_ledger)
    assert payload['preserved_paths']['graph_rebase_readiness_registry'] == str(readiness_registry)

    manifest = json.loads((archive_root / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['archived_files']['event_log'].endswith('state/events.jsonl')
    assert manifest['archived_files']['runtime_log'].endswith('logs/flask_webserver.log')
    assert manifest['preserved_paths']['response_frame_ledger'] == str(response_ledger)
    assert manifest['preserved_paths']['graph_rebase_readiness_registry'] == str(readiness_registry)

    assert event_log.read_text(encoding='utf-8') == ''
    assert runtime_log.read_text(encoding='utf-8') == ''
    assert response_ledger.read_text(encoding='utf-8') == '{"kind":"ollmo.response_frame"}\n'
    assert readiness_registry.read_text(encoding='utf-8') == (
        '{"kind":"ollmo.graph_rebase_readiness_registry_record"}\n'
    )

    archived_event_log = archive_root / 'state' / 'events.jsonl'
    assert archived_event_log.read_text(encoding='utf-8') == '{"kind":"legacy"}\n'
    archived_runtime_log = archive_root / 'logs' / 'flask_webserver.log'
    assert archived_runtime_log.read_text(encoding='utf-8') == 'legacy router timeout\n'

    assert not learned_policy.exists()
    assert not learned_policy_markdown.exists()

    assert not compiled_memory.exists()
    assert not compiled_memory_markdown.exists()
    assert not self_learning_dir.exists()
    assert (archive_root / 'state' / 'self_learning' / 'accepted_policy_snapshot.json').exists()
