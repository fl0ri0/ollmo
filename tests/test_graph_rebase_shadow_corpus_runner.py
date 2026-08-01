import hashlib
import json
from email.message import Message
from pathlib import Path
import threading
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request

import pytest

from ollmo_g.router import extract_recent_artifacts
from ollmo_server.late_fill_runtime import LateFillRuntimeOwner
from ollmo_server.request_intake_runtime import RequestIntakeRuntimeOwner
from ollmo_server.responses_request_runtime import ResponsesRequestRuntimeOwner
from ollmo_services.artifact_contracts import sanitize_artifact_record
from scripts.run_graph_rebase_shadow_corpus import (
    CorpusError,
    HttpResult,
    JsonHttpClient,
    MAX_PREDECESSOR_ARTIFACTS,
    MAX_PREDECESSOR_CONTEXT_BYTES,
    ManifestLockedError,
    ShadowCorpusRunner,
    assert_manifest_matches_corpus,
    atomic_write_json,
    build_manifest,
    build_parser,
    build_request_payload,
    classify_compact_status,
    derive_rebase_opportunity_summary,
    deterministic_conversation_id,
    deterministic_response_id,
    load_corpus,
    load_manifest,
    manifest_lock,
    manifest_status_payload,
    plan_payload,
    summarize_debug_payload,
    validate_base_url,
    _RejectRedirectHandler,
)


def _write_corpus(path: Path, cases=None, **overrides):
    payload = {
        'schema_version': 1,
        'corpus_id': 'shadow-corpus-test',
        'title': 'Shadow corpus test',
        'cases': cases
        or [
            {
                'case_id': 'first-case',
                'wave': 1,
                'category': 'clean_chat',
                'prompt': 'Explain a bounded topic in two sentences.',
                'expected_capability_families': ['chat'],
            }
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _readiness_payload(settled=6):
    return {
        'kind': 'ollmo.graph_rebase_rollout_readiness',
        'runtime_effect': 'none',
        'report_digest': f'readiness-{settled}',
        'corpus': {
            'corpus_digest': f'corpus-{settled}',
            'settled_final_response_count': settled,
            'nonterminal_active_late_fill_response_count': 0,
            'unique_workload_family_count': 3,
            'workload_family_counts': {'clean_chat': settled},
        },
        'candidate_opportunities': {
            'settled_final': {
                'total': settled,
                'not_proposed_count': settled,
                'with_formal_proposal_count': 0,
                'by_status': {'not_proposed': settled},
            }
        },
        'formal_evidence': {'proposals': {'total': 0}},
        'qualifying_evidence': {'partial_proposal_count': 0},
        'safety': {
            'unresolved_critical_finding_count': 0,
            'zero_tolerance_satisfied': True,
        },
        'gates': {
            'shadow_to_stage': {
                'ready': False,
                'decision': 'remain_shadow',
                'unmet_requirements': ['minimum_settled_candidate_opportunities'],
            }
        },
        'observer': {
            'hydrated_response_count': settled,
            'selected_graph_rebase_observation_count': settled,
            'load_error_count': 0,
        },
    }


def _terminal_status(response_id, *, repair=False, version='terminal-v1'):
    lifecycle = 'repair_needed' if repair else 'completed'
    return {
        'id': response_id,
        'object': 'response.status',
        'status': 'completed',
        'lifecycle_state': lifecycle,
        'state_version': version,
        'frame_id': f'frame-{response_id}',
        'frame_sequence': 2,
        'status_semantics': {
            'canonical_lifecycle_state': lifecycle,
            'has_open_continuation': False,
            'has_actionable_repair': repair,
            'is_terminal': not repair,
        },
        'late_fill': {
            'status': 'completed' if not repair else 'failed',
            'pending_count': 0,
            'active_count': 0,
            'failed_count': 1 if repair else 0,
        },
        'output_counts': {'output_count': 1, 'artifact_count': 0},
    }


def _open_status(response_id, version='open-v1'):
    return {
        'id': response_id,
        'object': 'response.status',
        'status': 'completed',
        'lifecycle_state': 'late_fill_running',
        'state_version': version,
        'status_semantics': {
            'has_open_continuation': True,
            'has_actionable_repair': False,
            'is_terminal': False,
        },
        'late_fill': {'status': 'running', 'pending_count': 0, 'active_count': 1},
    }


def _debug_payload(response_id):
    return {
        'id': response_id,
        'status': 'completed',
        'lifecycle_state': 'completed',
        'output': [
            {
                'id': f'msg-{response_id}',
                'type': 'message',
                'role': 'assistant',
                'content': [],
            }
        ],
        'outputs': [
            {
                'slot_id': 'slot-text',
                'type': 'text',
                'status': 'fulfilled',
                'artifact_ref': 'artifact:text:1',
                'content': 'x' * 100_000,
            }
        ],
        'artifacts': [
            {
                'artifact_id': 'artifact-text-1',
                'artifact_ref': 'artifact:text:1',
                'type': 'text',
                'path': '/tmp/result.txt',
                'content': 'y' * 100_000,
            }
        ],
        'late_fill': {
            'status': 'completed',
            'completed_branches': [
                {
                    'branch_id': 'branch-text',
                    'status': 'fulfilled',
                    'content_payload': 'z' * 100_000,
                }
            ],
        },
        'runtime': {
            'graph_closure_review': {
                'status': 'completed',
                'intent_graph_adequacy': {
                    'status': 'passed',
                    'adequate': True,
                    'entire_graph': {'body': 'not copied'},
                },
            },
            'request_phase_graph': {
                'redraw_scope_ladder_review': {
                    'review_id': 'scope-1',
                    'selected_scope': 'additive_repair',
                },
                'graph_rebase_proposals': [],
                'graph_rebase_reviews': [],
            },
            'developer_diagnostics': {
                'runtime_graph_rebase_candidate_review': {
                    'status': 'not_proposed',
                    'reason': 'current_structural_closure_evidence_missing',
                    'candidate_graph': {'large': 'not copied'},
                }
            },
        },
        'response_frame': {'frame_id': f'frame-{response_id}', 'frame_sequence': 2},
    }


def _capture_predecessor_context(case, *, text='Exact prior assistant text.', artifacts=None):
    artifacts = list(
        artifacts
        or [
            {
                'artifact_id': 'image-prior-1',
                'artifact_ref': 'artifact:image-prior-1',
                'type': 'image',
                'kind': 'image',
                'path': '/tmp/prior-image.png',
                'file_sha256': 'a' * 64,
                'file_size_bytes': 321,
                'mime_type': 'image/png',
            }
        ]
    )
    response_id = case['response_id']
    frame_id = f'{response_id}:frame-2'
    message_id = f'msg-{case["case_id"]}-final'
    case.update(
        {
            'state': 'settled_terminal',
            'settled_outcome': 'success',
            'dependency_satisfied': True,
            'last_lifecycle_state': 'completed',
            'last_late_fill_status': 'completed',
            'last_frame_id': frame_id,
            'last_frame_sequence': 2,
            'final_debug': {
                'status': 'captured',
                'attempted_at': '2026-07-21T00:00:00Z',
                'finished_at': '2026-07-21T00:00:01Z',
                'summary': {
                    'id': response_id,
                    'lifecycle_state': 'completed',
                    'message_id': message_id,
                    'message_identity': {
                        'status': 'exact',
                        'message_id': message_id,
                    },
                    'response_frame': {
                        'frame_id': frame_id,
                        'frame_sequence': 2,
                        'status': 'completed',
                    },
                    'final_text': {
                        'source': 'output_text',
                        'text': text,
                        'length_chars': len(text),
                        'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                        'truncated': False,
                    },
                    'artifact_count': len(artifacts),
                    'artifacts': artifacts,
                    'outputs': [
                        {
                            'artifact_ref': item['artifact_ref'],
                            'type': item['type'],
                            'status': 'fulfilled',
                        }
                        for item in artifacts
                    ],
                },
            },
        }
    )
    return case


def _opportunity_manifest(tmp_path):
    corpus = load_corpus(
        _write_corpus(
            tmp_path / 'opportunity-context.json',
            cases=[
                {
                    'case_id': 'root',
                    'wave': 1,
                    'category': 'root',
                    'prompt': 'Create bounded predecessor evidence.',
                    'conversation_key': 'shared-opportunity-context',
                    'opportunity_contract': {
                        'sequence_id': 'context-sequence',
                        'motif': 'evidence_determined_fanout',
                        'turn_index': 1,
                        'turn_role': 'root',
                    },
                },
                {
                    'case_id': 'follow',
                    'wave': 2,
                    'category': 'follow',
                    'prompt': 'Use only the immediately preceding evidence.',
                    'conversation_key': 'shared-opportunity-context',
                    'depends_on': ['root'],
                    'opportunity_contract': {
                        'sequence_id': 'context-sequence',
                        'motif': 'evidence_determined_fanout',
                        'turn_index': 2,
                        'turn_role': 'follow_up',
                    },
                },
            ],
        )
    )
    return corpus, build_manifest(corpus, tmp_path / 'run.json')


class FakeClient:
    def __init__(self, *, status_sequences=None, post_results=None, debug_payloads=None):
        self.status_sequences = {
            key: list(values) for key, values in (status_sequences or {}).items()
        }
        self.post_results = list(post_results or [])
        self.debug_payloads = dict(debug_payloads or {})
        self.calls = []
        self.lock = threading.Lock()
        self.readiness_count = 0

    def get(self, path, *, timeout=None):
        with self.lock:
            self.calls.append(('GET', path, None))
            if path == '/api/running_instances':
                return HttpResult(
                    200,
                    [
                        {
                            'instance_id': 'chat-ready',
                            'readiness': 'ready',
                            'capability': 'chat',
                            'provider_capabilities': ['chat', 'vision_analysis'],
                        }
                    ],
                )
            if path == '/api/ghost_preferences':
                return HttpResult(200, {'preferences': {'primary': 'ghost-live'}})
            if path == '/api/graph_rebase/readiness':
                self.readiness_count += 1
                return HttpResult(200, _readiness_payload(5 + self.readiness_count), 800)
            if path.endswith('?view=debug'):
                response_id = path.split('/api/responses/', 1)[1].split('?', 1)[0]
                return HttpResult(
                    200,
                    self.debug_payloads.get(response_id, _debug_payload(response_id)),
                    250_000,
                )
            if path.endswith('?view=status'):
                response_id = path.split('/api/responses/', 1)[1].split('?', 1)[0]
                sequence = self.status_sequences.get(response_id)
                if not sequence:
                    raise AssertionError(f'No fake status left for {response_id}: {path}')
                return sequence.pop(0)
            raise AssertionError(f'Unexpected GET: {path}')

    def post(self, path, payload, *, timeout=None):
        with self.lock:
            self.calls.append(('POST', path, dict(payload)))
            if path != '/api/responses':
                raise AssertionError(f'Unexpected POST: {path}')
            if self.post_results:
                return self.post_results.pop(0)
            return HttpResult(200, {'id': payload['response_id']}, 128)


def _runner(tmp_path, corpus, manifest, client, **overrides):
    manifest_path = tmp_path / 'run.json'
    atomic_write_json(manifest_path, manifest)
    return ShadowCorpusRunner(
        corpus=corpus,
        manifest=manifest,
        manifest_path=manifest_path,
        client=client,
        poll_interval=0,
        sleep_fn=lambda _seconds: None,
        emit=lambda _message: None,
        **overrides,
    )


@pytest.mark.parametrize(
    'base_url',
    [
        'http://example.test:5001',
        'http://127.0.0.1:5999',
        'http://user:pass@127.0.0.1:5001',
        'http://127.0.0.1:5001/nested',
        'http://127.0.0.1:not-a-port',
    ],
)
def test_shadow_corpus_transport_requires_exact_local_control_plane(base_url):
    with pytest.raises(CorpusError):
        validate_base_url(base_url)


def test_shadow_corpus_transport_disables_proxy_and_redirects():
    with patch(
        'scripts.run_graph_rebase_shadow_corpus.build_opener'
    ) as mock_build_opener:
        JsonHttpClient('http://127.0.0.1:5001')
    handlers = mock_build_opener.call_args.args
    proxy_handlers = [
        handler for handler in handlers if isinstance(handler, ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}

    handler = _RejectRedirectHandler()
    request = Request('http://127.0.0.1:5001/api/responses')
    with pytest.raises(HTTPError, match='Redirects are forbidden'):
        handler.http_error_302(request, None, 302, 'Found', Message())


def test_corpus_digest_ids_and_conversations_are_stable_and_prompt_bound(tmp_path):
    path = _write_corpus(
        tmp_path / 'corpus.json',
        cases=[
            {
                'case_id': 'root',
                'wave': 1,
                'category': 'chat',
                'prompt': 'First prompt',
                'conversation_key': 'shared-thread',
            },
            {
                'case_id': 'follow-up',
                'wave': 2,
                'category': 'follow_up',
                'prompt': 'Change only the title.',
                'conversation_key': 'shared-thread',
                'predecessor': 'root',
            },
        ],
    )
    corpus = load_corpus(path)
    corpus_again = load_corpus(path)

    assert corpus['corpus_digest'] == corpus_again['corpus_digest']
    assert deterministic_response_id(corpus, corpus['cases'][0]) == deterministic_response_id(
        corpus_again, corpus_again['cases'][0]
    )
    assert deterministic_conversation_id(corpus, corpus['cases'][0]) == deterministic_conversation_id(
        corpus, corpus['cases'][1]
    )
    assert corpus['cases'][1]['depends_on'] == ['root']

    payload = json.loads(path.read_text())
    payload['cases'][0]['prompt'] = 'Changed prompt'
    path.write_text(json.dumps(payload), encoding='utf-8')
    changed = load_corpus(path)
    assert changed['corpus_digest'] != corpus['corpus_digest']
    assert deterministic_response_id(changed, changed['cases'][0]) != deterministic_response_id(
        corpus, corpus['cases'][0]
    )


def test_corpus_rejects_cycles_duplicates_and_runner_owned_overrides(tmp_path):
    cycle = _write_corpus(
        tmp_path / 'cycle.json',
        cases=[
            {'case_id': 'a', 'category': 'chat', 'prompt': 'A', 'depends_on': ['b']},
            {'case_id': 'b', 'category': 'chat', 'prompt': 'B', 'depends_on': ['a']},
        ],
    )
    with pytest.raises(CorpusError, match='dependency cycle'):
        load_corpus(cycle)

    duplicate = _write_corpus(
        tmp_path / 'duplicate.json',
        cases=[
            {'case_id': 'a', 'category': 'chat', 'prompt': 'A'},
            {'case_id': 'a', 'category': 'chat', 'prompt': 'B'},
        ],
    )
    with pytest.raises(CorpusError, match='Duplicate'):
        load_corpus(duplicate)

    override = _write_corpus(
        tmp_path / 'override.json',
        cases=[
            {
                'case_id': 'a',
                'category': 'chat',
                'prompt': 'A',
                'request_overrides': {'instance_id': 'bypass-ghost'},
            }
        ],
    )
    with pytest.raises(CorpusError, match='runner-owned'):
        load_corpus(override)


def test_unlabeled_opportunity_sequence_is_preserved_but_not_sent_to_runtime(tmp_path):
    path = _write_corpus(
        tmp_path / 'opportunity.json',
        cases=[
            {
                'case_id': 'root',
                'wave': 1,
                'category': 'root',
                'prompt': 'Create one bounded artifact.',
                'conversation_key': 'shared-opportunity',
                'opportunity_contract': {
                    'sequence_id': 'bounded-sequence',
                    'motif': 'evidence_determined_fanout',
                    'turn_index': 1,
                    'turn_role': 'root',
                    'observation_questions': ['Which runtime transition is sufficient?'],
                },
            },
            {
                'case_id': 'follow',
                'wave': 2,
                'category': 'follow',
                'prompt': 'Use the immediately preceding artifact and preserve it.',
                'conversation_key': 'shared-opportunity',
                'depends_on': ['root'],
                'opportunity_contract': {
                    'sequence_id': 'bounded-sequence',
                    'motif': 'evidence_determined_fanout',
                    'turn_index': 2,
                    'turn_role': 'follow_up',
                    'preservation_expectations': ['Keep the prior artifact unchanged.'],
                },
            },
        ],
    )
    corpus = load_corpus(path)
    manifest = build_manifest(corpus, tmp_path / 'run.json')

    assert corpus['cases'][1]['opportunity_contract']['turn_role'] == 'follow_up'
    assert deterministic_conversation_id(corpus, corpus['cases'][0]) == (
        deterministic_conversation_id(corpus, corpus['cases'][1])
    )
    root_payload = build_request_payload(manifest, manifest['cases'][0], {})
    assert 'ghost_messages' not in root_payload
    assert 'reference_artifacts' not in root_payload
    assert 'predecessor_context' not in root_payload['request_meta']
    with pytest.raises(CorpusError, match='not settled'):
        build_request_payload(manifest, manifest['cases'][1], {})

    _capture_predecessor_context(manifest['cases'][0])
    request_payload = build_request_payload(manifest, manifest['cases'][1], {})
    assert 'opportunity_contract' not in request_payload
    assert 'opportunity_contract' not in request_payload['request_meta']
    assert request_payload['ghost_messages'][0]['content'] == (
        'Exact prior assistant text.'
    )
    assert request_payload['reference_artifacts'][0]['type'] == 'message'
    plan = plan_payload(corpus)
    assert plan['cases'][1]['opportunity_contract']['motif'] == (
        'evidence_determined_fanout'
    )

    biased = json.loads(path.read_text())
    biased['cases'][0]['opportunity_contract']['expected_proposal'] = True
    path.write_text(json.dumps(biased), encoding='utf-8')
    with pytest.raises(CorpusError, match='outcome labels or injected authority'):
        load_corpus(path)


def test_opportunity_followup_carries_exact_bounded_predecessor_context(tmp_path):
    _corpus, manifest = _opportunity_manifest(tmp_path)
    artifacts = [
        {
            'artifact_id': 'image-exact',
            'artifact_ref': 'artifact:image-exact',
            'type': 'image',
            'path': '/tmp/exact-image.png',
            'file_sha256': 'a' * 64,
            'file_size_bytes': 101,
        },
        {
            'artifact_id': 'audio-exact',
            'artifact_ref': 'artifact:audio-exact',
            'type': 'audio',
            'path': '/tmp/exact-audio.wav',
            'file_sha256': 'b' * 64,
            'file_size_bytes': 202,
            'mime_type': 'audio/wav',
        },
    ]
    root = _capture_predecessor_context(
        manifest['cases'][0],
        text='The exact immutable predecessor answer.',
        artifacts=artifacts,
    )
    follow = manifest['cases'][1]

    payload = build_request_payload(manifest, follow, {})
    repeated = build_request_payload(manifest, follow, {})

    assert payload == repeated
    assert len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) < (
        MAX_PREDECESSOR_CONTEXT_BYTES + 8_192
    )
    assert 'input_artifacts' not in payload
    assistant = payload['ghost_messages'][0]
    assert assistant['role'] == 'assistant'
    assert assistant['content'] == 'The exact immutable predecessor answer.'
    assert assistant['message_id'] == 'msg-root-final'
    assert assistant['response_id'] == root['response_id']
    assert [item['artifact_ref'] for item in assistant['artifacts']] == [
        'artifact:image-exact',
        'artifact:audio-exact',
    ]
    assert all(
        item['source_response_id'] == root['response_id']
        and item['source_message_id'] == 'msg-root-final'
        for item in assistant['artifacts']
    )
    references = payload['reference_artifacts']
    assert references[0] == {
        'type': 'message',
        'message_role': 'assistant',
        'message_id': 'msg-root-final',
        'source_response_id': root['response_id'],
        'content': 'The exact immutable predecessor answer.',
    }
    assert [item['artifact_ref'] for item in references[1:]] == [
        'artifact:image-exact',
        'artifact:audio-exact',
    ]
    assert all(
        item['source_response_id'] == root['response_id']
        and item['source_message_id'] == 'msg-root-final'
        for item in references[1:]
    )
    audit = payload['request_meta']['predecessor_context']
    assert audit['source'] == 'immutable_final_debug_summary'
    assert audit['response_id'] == root['response_id']
    assert audit['frame_id'] == root['last_frame_id']
    assert audit['frame_sequence'] == root['last_frame_sequence']
    assert audit['message_id'] == 'msg-root-final'
    assert audit['artifact_refs'] == [
        'artifact:image-exact',
        'artifact:audio-exact',
    ]
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        'expected_outcome',
        'graph_rebase_authorization',
        'opportunity_contract',
        'repair_action',
        'selected_scope',
        'synthetic_content',
    ):
        assert forbidden not in encoded


def test_runner_dispatches_opportunity_root_then_one_context_bound_followup(tmp_path):
    corpus, manifest = _opportunity_manifest(tmp_path)
    root_id = manifest['cases'][0]['response_id']
    follow_id = manifest['cases'][1]['response_id']
    root_debug = _debug_payload(root_id)
    root_debug['output_text'] = 'Exact settled root response.'
    client = FakeClient(
        status_sequences={
            root_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(200, _terminal_status(root_id)),
            ],
            follow_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(200, _terminal_status(follow_id)),
            ],
        },
        debug_payloads={root_id: root_debug},
    )
    runner = _runner(tmp_path, corpus, manifest, client, max_in_flight=1)

    assert runner.run(max_cycles=30) == 0
    posts = [call[2] for call in client.calls if call[0] == 'POST']
    assert len(posts) == 2
    assert posts[0]['response_id'] == root_id
    assert 'ghost_messages' not in posts[0]
    assert 'reference_artifacts' not in posts[0]
    assert posts[1]['response_id'] == follow_id
    assert posts[1]['ghost_messages'][0]['response_id'] == root_id
    assert posts[1]['ghost_messages'][0]['message_id'] == f'msg-{root_id}'
    assert posts[1]['reference_artifacts'][1]['artifact_ref'] == 'artifact:text:1'
    assert posts[1]['request_meta']['predecessor_context']['response_id'] == root_id
    assert [payload['response_id'] for payload in posts] == [root_id, follow_id]


def test_all_predecessor_handles_reach_routing_direct_context_and_late_fill(tmp_path):
    _corpus, manifest = _opportunity_manifest(tmp_path)
    artifact_specs = [
        ('text-first', 'text', 'first.json'),
        ('image', 'image', 'image.png'),
        ('text-story', 'text', 'story.json'),
        ('audio', 'audio', 'audio.wav'),
        ('text-transcript', 'text', 'transcript.md'),
        ('text-final', 'text', 'final.json'),
    ]
    artifacts = []
    for identity, artifact_type, filename in artifact_specs:
        path = tmp_path / filename
        path.write_text(identity, encoding='utf-8')
        artifacts.append(
            {
                'artifact_id': identity,
                'artifact_ref': f'artifact:{identity}',
                'type': artifact_type,
                'path': str(path),
            }
        )
    root = _capture_predecessor_context(manifest['cases'][0], artifacts=artifacts)
    payload = build_request_payload(manifest, manifest['cases'][1], {})
    expected_refs = {item['artifact_ref'] for item in artifacts}
    expected_paths = {item['path'] for item in artifacts}

    routed = extract_recent_artifacts(payload['ghost_messages'])
    assert {item['path'] for item in routed} == expected_paths
    assert len(routed) == len(artifacts)

    direct_owner = object.__new__(ResponsesRequestRuntimeOwner)
    direct_owner.hooks = {}
    direct = direct_owner.collect_direct_materialization_context_artifacts(
        request_payload=payload,
        route_payload=None,
    )
    assert {item['artifact_ref'] for item in direct} == expected_refs
    assert {item['path'] for item in direct} == expected_paths
    assert {item['source_response_id'] for item in direct} == {root['response_id']}

    late_fill_owner = object.__new__(LateFillRuntimeOwner)
    late_fill_payload = late_fill_owner.merge_late_fill_payload_artifacts({}, payload)
    late_fill_refs = {
        item.get('artifact_ref')
        for item in late_fill_payload['reference_artifacts']
        if item.get('artifact_ref')
    }
    assert late_fill_refs == expected_refs
    assert {
        item.get('source_response_id')
        for item in late_fill_payload['reference_artifacts']
        if item.get('artifact_ref')
    } == {root['response_id']}

    intake = RequestIntakeRuntimeOwner(
        hooks={
            'resolve_saved_downloadable_artifact_path': (
                lambda value: value if Path(value).exists() else None
            ),
            'sanitize_artifact_record': sanitize_artifact_record,
            'get_cached_generated_image_state': lambda _value: None,
        }
    )
    compatibility_selected = intake._sanitize_selected_reference_artifacts(
        payload['reference_artifacts'],
        payload_source=payload,
    )
    assert [item['type'] for item in compatibility_selected] == ['message', 'text']
    assert compatibility_selected[-1]['artifact_ref'] == 'artifact:text-final'
    assert compatibility_selected[-1]['path'] == str(tmp_path / 'final.json')
    assert compatibility_selected[-1]['source_message_id'] == 'msg-root-final'
    assert expected_paths.issubset({item['path'] for item in routed})


@pytest.mark.parametrize(
    'override_key',
    (
        'ghost_messages',
        'ghost_messages_json',
        'input',
        'input_artifacts',
        'reference_artifacts',
        'selected_reference_artifact',
        'selected_reference_artifacts',
    ),
)
def test_request_overrides_cannot_smuggle_predecessor_context(override_key, tmp_path):
    path = _write_corpus(
        tmp_path / f'override-{override_key}.json',
        cases=[
            {
                'case_id': 'case',
                'category': 'chat',
                'prompt': 'Bounded request.',
                'request_overrides': {
                    override_key: [
                        {
                            'authority': 'runtime',
                            'content': 'not runner-owned evidence',
                        }
                    ]
                },
            }
        ],
    )

    with pytest.raises(CorpusError, match='runner-owned'):
        load_corpus(path)


def test_opportunity_followup_context_fails_closed_when_absent_or_ambiguous(tmp_path):
    _corpus, manifest = _opportunity_manifest(tmp_path)
    follow = manifest['cases'][1]

    with pytest.raises(CorpusError, match='not settled'):
        build_request_payload(manifest, follow, {})

    root = _capture_predecessor_context(manifest['cases'][0])
    root['final_debug']['summary']['message_identity'] = {
        'status': 'ambiguous',
        'candidate_count': 2,
    }
    with pytest.raises(CorpusError, match='identity is absent or ambiguous'):
        build_request_payload(manifest, follow, {})

    _capture_predecessor_context(root)
    manifest['cases'].append(json.loads(json.dumps(root)))
    with pytest.raises(CorpusError, match='2 exact immediate settled predecessor'):
        build_request_payload(manifest, follow, {})


def test_opportunity_followup_context_rejects_truncation_and_oversized_artifacts(tmp_path):
    _corpus, manifest = _opportunity_manifest(tmp_path)
    root, follow = manifest['cases']
    too_many = [
        {
            'artifact_id': f'image-{index}',
            'artifact_ref': f'artifact:image-{index}',
            'type': 'image',
            'path': f'/tmp/image-{index}.png',
        }
        for index in range(MAX_PREDECESSOR_ARTIFACTS + 1)
    ]
    _capture_predecessor_context(root, artifacts=too_many)
    with pytest.raises(CorpusError, match='bounded limit'):
        build_request_payload(manifest, follow, {})

    _capture_predecessor_context(root, text='x' * 8_193)
    with pytest.raises(CorpusError, match='bounded exact text'):
        build_request_payload(manifest, follow, {})


@pytest.mark.parametrize(
    'contamination',
    [
        {
            'contract_extra': {
                'nested_review': {
                    'requestedRebaseClass': 'partial_subtree_rebase',
                }
            }
        },
        {
            'request_overrides': {
                'graph_rebase_authorization': {'status': 'accepted'},
            }
        },
        {
            'expected_evidence': {
                'expected_classification': 'useful_proposal',
            }
        },
    ],
)
def test_opportunity_corpus_rejects_nested_authority_and_outcome_injection(
    tmp_path,
    contamination,
):
    contract = {
        'sequence_id': 'bounded-sequence',
        'motif': 'evidence_determined_fanout',
        'turn_index': 1,
        'turn_role': 'root',
        **contamination.get('contract_extra', {}),
    }
    case = {
        'case_id': 'root',
        'category': 'root',
        'prompt': 'Create one bounded artifact.',
        'opportunity_contract': contract,
    }
    if 'request_overrides' in contamination:
        case['request_overrides'] = contamination['request_overrides']
    if 'expected_evidence' in contamination:
        case['expected_evidence'] = contamination['expected_evidence']
    path = _write_corpus(tmp_path / 'contaminated-opportunity.json', cases=[case])

    with pytest.raises(CorpusError, match='authority|overrides|expected_evidence'):
        load_corpus(path)


def test_opportunity_sequence_requires_shared_conversation_and_immediate_dependency(tmp_path):
    path = _write_corpus(
        tmp_path / 'invalid-opportunity.json',
        cases=[
            {
                'case_id': 'root',
                'category': 'root',
                'prompt': 'Root.',
                'conversation_key': 'one',
                'opportunity_contract': {
                    'sequence_id': 'sequence',
                    'motif': 'branch_split',
                    'turn_index': 1,
                    'turn_role': 'root',
                },
            },
            {
                'case_id': 'follow',
                'category': 'follow',
                'prompt': 'Follow.',
                'conversation_key': 'two',
                'opportunity_contract': {
                    'sequence_id': 'sequence',
                    'motif': 'branch_split',
                    'turn_index': 2,
                    'turn_role': 'follow_up',
                },
            },
        ],
    )

    with pytest.raises(CorpusError, match='must depend on immediate predecessor'):
        load_corpus(path)


def test_plan_and_wave_filter_are_read_only_and_dependency_visible(tmp_path):
    corpus = load_corpus(
        _write_corpus(
            tmp_path / 'corpus.json',
            cases=[
                {'case_id': 'one', 'wave': 1, 'category': 'chat', 'prompt': 'One'},
                {
                    'case_id': 'two',
                    'wave': 2,
                    'category': 'follow_up',
                    'prompt': 'Two',
                    'depends_on': ['one'],
                },
            ],
        )
    )
    payload = plan_payload(corpus, wave='2')
    assert payload['runtime_effect'] == 'none'
    assert payload['case_count'] == 1
    assert payload['cases'][0]['depends_on'] == ['one']
    assert not (tmp_path / 'run.json').exists()


def test_atomic_manifest_round_trip_digest_mismatch_and_lock(tmp_path):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    path = tmp_path / 'run.json'
    manifest = build_manifest(corpus, path)
    atomic_write_json(path, manifest)

    loaded = load_manifest(path)
    assert loaded['corpus_digest'] == corpus['corpus_digest']
    assert not list(tmp_path.glob('.run.json.*.tmp'))

    changed = dict(corpus)
    changed['corpus_digest'] = 'changed'
    with pytest.raises(CorpusError, match='digest changed'):
        assert_manifest_matches_corpus(loaded, changed)

    with manifest_lock(path):
        with pytest.raises(ManifestLockedError):
            with manifest_lock(path):
                pass


def test_compact_status_semantics_keep_open_work_open_and_settle_repairs():
    open_but_compat_completed = _open_status('resp-open')
    assert classify_compact_status(open_but_compat_completed) == 'open'

    inconsistent_terminal = _terminal_status('resp-inconsistent')
    inconsistent_terminal['late_fill']['active_count'] = 1
    assert classify_compact_status(inconsistent_terminal) == 'open'

    repair = _terminal_status('resp-repair', repair=True)
    assert classify_compact_status(repair) == 'settled_repair_needed'


def test_compact_status_classifies_partial_cancelled_as_terminal():
    assert classify_compact_status(
        {
            'status': 'completed',
            'lifecycle_state': 'partial_cancelled',
        }
    ) == 'settled_terminal'


def test_compact_status_ignores_nonfinite_open_counts():
    assert classify_compact_status(
        {
            'lifecycle_state': 'completed',
            'late_fill': {'active_count': float('inf')},
        }
    ) == 'settled_terminal'


@pytest.mark.parametrize(
    'status_semantics',
    (
        {'is_terminal': True},
        {'terminal': True},
        {'has_actionable_repair': True},
    ),
)
def test_compact_status_canonical_open_lifecycle_wins_stale_settled_semantics(
    status_semantics,
):
    assert classify_compact_status(
        {
            'lifecycle_state': 'late_fill_running',
            'status_semantics': status_semantics,
        }
    ) == 'open'


@pytest.mark.parametrize(
    'late_fill_status',
    ('accepted', 'active', 'in_progress', 'pending', 'queued', 'running', 'scheduled'),
)
def test_compact_status_active_late_fill_status_wins_stale_terminal_projection(
    late_fill_status,
):
    assert classify_compact_status(
        {
            'lifecycle_state': 'completed',
            'status_semantics': {
                'has_open_continuation': False,
                'is_terminal': True,
            },
            'late_fill': {
                'status': late_fill_status,
                'pending_count': 0,
                'active_count': 0,
            },
        }
    ) == 'open'


def test_compact_status_canonical_repair_lifecycle_wins_stale_terminal_semantics():
    assert classify_compact_status(
        {
            'lifecycle_state': 'repair_needed',
            'status_semantics': {
                'has_actionable_repair': False,
                'is_terminal': True,
            },
        }
    ) == 'settled_repair_needed'


@pytest.mark.parametrize(
    'late_fill_status',
    ('blocked', 'failed', 'late_fill_failed', 'repair_needed', 'partial_failed'),
)
def test_compact_status_late_fill_repair_wins_stale_canonical_completion(
    late_fill_status,
):
    assert classify_compact_status(
        {
            'lifecycle_state': 'completed',
            'status_semantics': {
                'has_actionable_repair': False,
                'is_terminal': True,
            },
            'late_fill': {
                'status': late_fill_status,
                'pending_count': 0,
                'active_count': 0,
                'failed_count': 0,
            },
        }
    ) == 'settled_repair_needed'


@pytest.mark.parametrize(
    'late_fill_status',
    ('canceled', 'cancelled', 'partial_cancelled', 'superseded'),
)
def test_compact_status_late_fill_cancellation_is_non_success_terminal(
    late_fill_status,
):
    assert classify_compact_status(
        {
            'lifecycle_state': 'completed',
            'status_semantics': {
                'has_open_continuation': False,
                'is_terminal': True,
            },
            'late_fill': {
                'status': late_fill_status,
                'pending_count': 0,
                'active_count': 0,
                'failed_count': 0,
            },
        }
    ) == 'settled_terminal'


def test_compact_status_positive_failed_count_wins_stale_canonical_completion():
    assert classify_compact_status(
        {
            'lifecycle_state': 'completed',
            'status_semantics': {
                'has_actionable_repair': False,
                'is_terminal': True,
            },
            'late_fill': {
                'status': 'completed',
                'pending_count': 0,
                'active_count': 0,
                'failed_count': 1,
            },
        }
    ) == 'settled_repair_needed'


def test_runner_gets_before_single_post_polls_status_and_fetches_debug_once(tmp_path):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    response_id = manifest['cases'][0]['response_id']
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(404, {'error': 'Response not found.'}),
                HttpResult(200, _terminal_status(response_id)),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client)

    assert runner.run(max_cycles=30) == 0
    case = runner.manifest['cases'][0]
    assert case['state'] == 'settled_terminal'
    assert case['final_debug']['status'] == 'captured'
    assert case['final_debug']['summary']['rebase_opportunity']['runtime_effect'] == 'none'
    assert case['dispatch_request']['ghost_route'] is True
    assert case['dispatch_request']['response_id'] == response_id
    assert case['dispatch_request']['conversation_id'] == case['conversation_id']
    assert case['dispatch_request']['ghost_preferences'] == {'primary': 'ghost-live'}

    response_calls = [call for call in client.calls if '/api/responses' in call[1]]
    assert response_calls[0][:2] == ('GET', f'/api/responses/{response_id}?view=status')
    assert [call[0] for call in response_calls].count('POST') == 1
    assert [call[1] for call in response_calls].count(
        f'/api/responses/{response_id}?view=debug'
    ) == 1
    assert all('/graph_rebase/operator' not in call[1] for call in client.calls)
    assert all('/late_fill/' not in call[1] for call in client.calls)
    assert all(call[1] not in {'/api/start', '/api/stop', '/api/restart'} for call in client.calls)
    phases = [item['phase'] for item in runner.manifest['readiness_checkpoints']]
    assert phases == ['baseline', 'after_case', 'final']


@pytest.mark.parametrize(
    ('late_fill_status', 'failed_count'),
    (('blocked', 0), ('failed', 1)),
)
def test_runner_settles_stale_completed_with_repair_late_fill_as_repair(
    tmp_path,
    late_fill_status,
    failed_count,
):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    response_id = manifest['cases'][0]['response_id']
    stale_completed = _terminal_status(response_id)
    stale_completed['late_fill'].update(
        {
            'status': late_fill_status,
            'failed_count': failed_count,
        }
    )
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(404, {'error': 'Response not found.'}),
                HttpResult(200, stale_completed),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client)

    assert runner.run(max_cycles=30) == 0
    case = runner.manifest['cases'][0]
    assert case['last_lifecycle_state'] == 'completed'
    assert case['last_late_fill_status'] == late_fill_status
    assert case['state'] == 'settled_repair_needed'
    assert case['settled_outcome'] == 'repair_needed'
    assert case['dependency_satisfied'] is False


@pytest.mark.parametrize(
    'late_fill_status',
    ('canceled', 'cancelled', 'partial_cancelled', 'superseded'),
)
def test_runner_never_settles_cancelled_late_fill_as_success(
    tmp_path,
    late_fill_status,
):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    response_id = manifest['cases'][0]['response_id']
    stale_completed = _terminal_status(response_id)
    stale_completed['late_fill'].update(
        {
            'status': late_fill_status,
            'failed_count': 0,
        }
    )
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(404, {'error': 'Response not found.'}),
                HttpResult(200, stale_completed),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client)

    assert runner.run(max_cycles=30) == 0
    case = runner.manifest['cases'][0]
    assert case['last_lifecycle_state'] == 'completed'
    assert case['last_late_fill_status'] == late_fill_status
    assert case['state'] == 'settled_terminal'
    assert case['settled_outcome'] == 'non_success_terminal'
    assert case['dependency_satisfied'] is False
    status = manifest_status_payload(runner.manifest)
    assert status['cases'][0]['last_late_fill_status'] == late_fill_status


def test_runner_advances_manifest_cursor_to_newer_final_debug_frame(tmp_path):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    response_id = manifest['cases'][0]['response_id']
    terminal = _terminal_status(response_id)
    terminal['frame_id'] = f'{response_id}:frame-1'
    terminal['frame_sequence'] = 1
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(404, {'error': 'Response not found.'}),
                HttpResult(200, terminal),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client)

    assert runner.run(max_cycles=30) == 0

    case = runner.manifest['cases'][0]
    debug_frame = case['final_debug']['summary']['response_frame']
    assert debug_frame['frame_sequence'] == 2
    assert case['last_frame_id'] == debug_frame['frame_id']
    assert case['last_frame_sequence'] == 2
    status = manifest_status_payload(runner.manifest)
    assert status['cases'][0]['last_frame_id'] == debug_frame['frame_id']
    assert status['cases'][0]['last_frame_sequence'] == 2


def test_runner_frame_cursor_ignores_missing_and_older_observations(tmp_path):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    runner = _runner(tmp_path, corpus, manifest, FakeClient())
    case = runner.manifest['cases'][0]

    assert runner._update_last_frame_observation(
        case,
        frame_id='resp-test:frame-2',
        frame_sequence=2,
    )
    assert not runner._update_last_frame_observation(case)
    assert not runner._update_last_frame_observation(
        case,
        frame_id='resp-test:frame-1',
        frame_sequence=1,
    )

    assert case['last_frame_id'] == 'resp-test:frame-2'
    assert case['last_frame_sequence'] == 2


def test_resumed_submitting_404_becomes_dispatch_unknown_and_never_posts(tmp_path):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    case = manifest['cases'][0]
    case['state'] = 'submitting'
    response_id = case['response_id']
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(404, {'error': 'still missing'}),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client)

    assert runner.run(max_cycles=10) == 2
    assert runner.manifest['cases'][0]['state'] == 'dispatch_unknown'
    assert runner.manifest['cases'][0]['dispatch_unknown_reason'] == (
        'submitting_resume_lookup_not_found'
    )
    assert not [call for call in client.calls if call[0] == 'POST']


def test_resumed_opportunity_followup_keeps_context_digest_and_never_reposts(tmp_path):
    corpus, manifest = _opportunity_manifest(tmp_path)
    _capture_predecessor_context(manifest['cases'][0])
    follow = manifest['cases'][1]
    dispatch_request = build_request_payload(manifest, follow, {})
    dispatch_digest = hashlib.sha256(
        json.dumps(
            dispatch_request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    follow.update(
        {
            'state': 'submitting',
            'dispatch_request': dispatch_request,
            'dispatch_request_digest': dispatch_digest,
        }
    )
    response_id = follow['response_id']
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(404, {'error': 'still missing'}),
            ]
        }
    )
    runner = _runner(
        tmp_path,
        corpus,
        manifest,
        client,
        case_ids=['follow'],
    )

    assert runner.run(max_cycles=10) == 2
    resumed = runner.manifest['cases'][1]
    assert resumed['dispatch_request'] == dispatch_request
    assert resumed['dispatch_request_digest'] == dispatch_digest
    assert not [call for call in client.calls if call[0] == 'POST']


def test_dispatch_unknown_is_get_only_until_existing_response_settles(tmp_path):
    corpus = load_corpus(_write_corpus(tmp_path / 'corpus.json'))
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    case = manifest['cases'][0]
    case['state'] = 'dispatch_unknown'
    response_id = case['response_id']
    client = FakeClient(
        status_sequences={
            response_id: [
                HttpResult(200, _open_status(response_id)),
                HttpResult(200, _terminal_status(response_id, version='terminal-v2')),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client)

    assert runner.run(max_cycles=10) == 0
    assert runner.manifest['cases'][0]['state'] == 'settled_terminal'
    assert not [call for call in client.calls if call[0] == 'POST']
    assert [call[1] for call in client.calls].count(
        f'/api/responses/{response_id}?view=debug'
    ) == 1


def test_dependency_fail_closed_when_predecessor_needs_repair(tmp_path):
    corpus = load_corpus(
        _write_corpus(
            tmp_path / 'corpus.json',
            cases=[
                {
                    'case_id': 'producer',
                    'wave': 1,
                    'category': 'producer',
                    'prompt': 'Produce one artifact.',
                    'expected_capability_families': ['chat'],
                },
                {
                    'case_id': 'follow-up',
                    'wave': 1,
                    'category': 'follow_up',
                    'prompt': 'Change only one bounded part.',
                    'depends_on': ['producer'],
                    'expected_capability_families': ['chat'],
                },
            ],
        )
    )
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    producer_id = manifest['cases'][0]['response_id']
    follow_up_id = manifest['cases'][1]['response_id']
    client = FakeClient(
        status_sequences={
            producer_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(200, _terminal_status(producer_id, repair=True)),
            ],
            follow_up_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(200, _terminal_status(follow_up_id)),
            ],
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client, max_in_flight=1)

    assert runner.run(max_cycles=30) == 0
    assert [case['state'] for case in runner.manifest['cases']] == [
        'settled_repair_needed',
        'dependency_blocked',
    ]
    post_case_ids = [
        call[2]['request_meta']['case_id'] for call in client.calls if call[0] == 'POST'
    ]
    assert post_case_ids == ['producer']


def test_wave_filter_does_not_dispatch_other_wave(tmp_path):
    corpus = load_corpus(
        _write_corpus(
            tmp_path / 'corpus.json',
            cases=[
                {'case_id': 'wave-one', 'wave': 1, 'category': 'chat', 'prompt': 'One'},
                {'case_id': 'wave-two', 'wave': 2, 'category': 'chat', 'prompt': 'Two'},
            ],
        )
    )
    manifest = build_manifest(corpus, tmp_path / 'run.json')
    wave_one_id = manifest['cases'][0]['response_id']
    client = FakeClient(
        status_sequences={
            wave_one_id: [
                HttpResult(404, {'error': 'missing'}),
                HttpResult(200, _terminal_status(wave_one_id)),
            ]
        }
    )
    runner = _runner(tmp_path, corpus, manifest, client, wave='1')

    assert runner.run(max_cycles=20) == 0
    assert runner.manifest['cases'][0]['state'] == 'settled_terminal'
    assert runner.manifest['cases'][1]['state'] == 'planned'
    assert len([call for call in client.calls if call[0] == 'POST']) == 1
    status = manifest_status_payload(runner.manifest, wave='1')
    assert status['case_count'] == 1


def test_final_debug_summary_is_bounded_and_excludes_payload_bodies():
    payload = _debug_payload('resp-debug')
    summary = summarize_debug_payload(payload, byte_count=350_000)
    encoded = json.dumps(summary)

    assert summary['response_bytes'] == 350_000
    assert summary['artifact_count'] == 1
    assert summary['message_identity'] == {
        'status': 'exact',
        'message_id': 'msg-resp-debug',
    }
    assert summary['message_id'] == 'msg-resp-debug'
    assert summary['graph_evidence']['graph_rebase_proposals']['total'] == 0
    assert len(encoded) < 20_000
    assert summary['final_text']['length_chars'] == 100_000
    assert len(summary['final_text']['text']) == 8_192
    assert summary['final_text']['truncated'] is True
    assert 'y' * 200 not in encoded
    assert 'z' * 200 not in encoded
    assert 'content_payload' not in encoded
    assert 'candidate_graph' not in encoded
    assert 'entire_graph' not in encoded


def test_debug_summary_keeps_dependency_and_identity_evidence():
    payload = _debug_payload('resp-dependency')
    closure = payload['runtime']['graph_closure_review']
    closure['intent_graph_adequacy']['reason'] = 'dependencies_fulfilled'
    graph = payload['runtime']['request_phase_graph']
    graph['intent_obligations'] = [
        {
            'obligation_id': 'join',
            'depends_on': ['vision-a'],
            'depends_on_obligation_ids': ['obligation-vision-a'],
            'source_phase_ids': ['phase-vision-a'],
            'target_phase_id': 'phase-join',
            'execution_dependency_required': True,
            'dependency_contract': {'input_refs': ['artifact:image:a']},
        }
    ]
    payload['late_fill']['fill_results'] = [
        {
            'phase_id': 'phase-vision-a',
            'saved_image_path': '/tmp/a.png',
            'artifact_ref': 'artifact:image:a',
            'output_obligation_ref': 'obligation-vision-a',
        }
    ]

    summary = summarize_debug_payload(payload)
    assert summary['closure']['intent_graph_adequacy']['reason'] == (
        'dependencies_fulfilled'
    )
    obligation = summary['request_phase_graph']['intent_obligations'][0]
    assert obligation['source_phase_ids'] == ['phase-vision-a']
    assert obligation['target_phase_id'] == 'phase-join'
    assert obligation['dependency_contract']['input_refs'] == ['artifact:image:a']
    fill = summary['late_fill']['fill_results'][0]
    assert fill['saved_image_path'] == '/tmp/a.png'
    assert fill['output_obligation_ref'] == 'obligation-vision-a'


def test_derived_rebase_opportunity_separates_review_candidate_from_blockers():
    summary = {
        'closure': {
            'status': 'repair_required',
            'checks': [
                {
                    'check_kind': 'intent_graph_adequacy',
                    'status': 'repair_required',
                    'repair_action': 'rebuild_from_promoted_obligations',
                }
            ],
        },
        'late_fill': {'status': 'completed', 'active_count': 0, 'pending_count': 0},
        'redraw_scope': {'selected_scope': 'observe'},
        'rebase_candidate': {
            'review': {
                'status': 'not_proposed',
                'reason': 'current_structural_closure_evidence_missing',
                'diff_summary': {
                    'meaningful_change_count': 3,
                    'operation_counts': {'remove_phase': 1, 'change_graph_semantics': 1},
                    'removed_ids': {'phases': ['phase-old']},
                },
            }
        },
        'graph_evidence': {'graph_rebase_proposals': {'total': 0}},
        'diagnostic_evidence': {'runtime_graph_rebase_proposals': {'total': 0}},
    }
    contract = {
        'sequence_id': 'sequence',
        'motif': 'branch_split',
        'turn_index': 2,
        'turn_role': 'follow_up',
    }

    opportunity = derive_rebase_opportunity_summary(
        summary,
        opportunity_contract=contract,
    )
    assert opportunity['disposition'] == 'unproposed_structural_opportunity'
    assert opportunity['eligible_for_operator_inspection'] is True
    assert opportunity['false_negative_review'] == (
        'candidate_requires_operator_judgment'
    )
    assert opportunity['runtime_effect'] == 'none'
    assert opportunity['authority'] == 'diagnostic_only_operator_judgment_required'

    lower_scope = json.loads(json.dumps(summary))
    lower_scope['redraw_scope']['selected_scope'] = 'repair_binding_dependency'
    blocked = derive_rebase_opportunity_summary(lower_scope)
    assert blocked['disposition'] == 'smaller_scope_precedes_rebase'
    assert blocked['eligible_for_operator_inspection'] is False
    assert blocked['blockers'] == [
        'smaller_scope_selected:repair_binding_dependency'
    ]

    healthy = json.loads(json.dumps(summary))
    healthy['closure'] = {'status': 'fulfilled', 'checks': []}
    no_structural_evidence = derive_rebase_opportunity_summary(healthy)
    assert no_structural_evidence['disposition'] == (
        'no_current_structural_closure_evidence'
    )
    assert no_structural_evidence['false_negative_review'] == 'not_applicable'
    assert no_structural_evidence['blockers'] == [
        'no_current_structural_closure_action'
    ]


def test_parser_defaults_to_one_in_flight():
    args = build_parser().parse_args(['run', '--corpus', 'corpus.json'])
    assert args.max_in_flight == 1


def test_case_filter_is_repeatable_and_preserves_evaluation_contract(tmp_path):
    corpus = load_corpus(
        _write_corpus(
            tmp_path / 'corpus.json',
            cases=[
                {
                    'case_id': 'one',
                    'wave': 1,
                    'category': 'chat',
                    'prompt': 'One',
                    'expected_evidence': {'expected_classification': 'clean'},
                    'stop_signals': ['unexpected_proposal'],
                },
                {'case_id': 'two', 'wave': 1, 'category': 'chat', 'prompt': 'Two'},
            ],
        )
    )

    assert corpus['cases'][0]['expected_evidence'] == {
        'expected_classification': 'clean'
    }
    assert corpus['cases'][0]['stop_signals'] == ['unexpected_proposal']
    payload = plan_payload(corpus, wave='1', case_ids=['one'])
    assert [case['case_id'] for case in payload['cases']] == ['one']
    args = build_parser().parse_args(
        ['run', '--corpus', 'corpus.json', '--case-id', 'one', '--case-id', 'two']
    )
    assert args.case_id == ['one', 'two']
