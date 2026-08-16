from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import ollmo_services.self_learning as self_learning_module
from ollmo_g.payload import build_ghost_payload
from ollmo_g.router import build_route_context
from ollmo_g.decision_contracts import build_ghost_decision_contract
from ollmo_services.self_learning import (
    CHAT_ROUTE_HEALTH_CASE_KIND,
    build_accepted_learning_runtime_hints,
    build_eval_cases_from_response_frame,
    build_policy_improvement_candidates,
    build_shadow_learning_hints,
    build_self_learning_report,
    enrich_eval_cases_from_monitor_reports,
    load_accepted_learning_policy_snapshot,
    promote_policy_improvement_candidate,
    persist_eval_cases,
    persist_accepted_learning_policy_snapshot,
    persist_self_learning_outputs,
    set_accepted_learning_policy_enabled,
)
from ollmo_services.self_learning_retention import (
    collect_self_learning_retention_roots,
    copy_retained_sidecars,
    retention_summary,
    write_retention_manifest,
)


def _problem_frame() -> dict:
    return {
        'response_id': 'resp_problem',
        'status': 'failed',
        'request': {'prompt': 'describe two scenes, then make images for both'},
        'target': {
            'backend': 'ollama',
            'capability': 'image_generation',
            'instance_id': 'image-1',
            'model': 'x/flux2-klein:latest',
        },
        'runtime': {
            'graph_closure_review': {
                'status': 'pending',
                'counts': {'pending': 1, 'fulfilled': 0},
                'intent_graph_adequacy': {
                    'status': 'pending',
                    'expected_output_counts': {'image': 2},
                    'graph_output_counts': {'image': 1},
                    'checks': [
                        {
                            'check_kind': 'intent_graph_adequacy',
                            'status': 'pending',
                            'output_type': 'image',
                            'evidence': 'current user intent asked for two image outputs',
                        }
                    ],
                },
                'checks': [
                    {
                        'check_kind': 'output_obligation',
                        'status': 'pending',
                        'evidence': 'image output slot still pending',
                    }
                ],
            }
        },
        'planning': {
            'artifact_flow': {
                'review': {
                    'status': 'blocked',
                    'pending_output_slot_ids': ['output-1'],
                    'blocked_output_slot_ids': ['output-2'],
                }
            },
            'context_contract': {
                'context_gate_review': {
                    'history_scan': {
                        'executed': True,
                        'matched_candidate_count': 0,
                        'promoted_candidate_count': 0,
                    }
                }
            },
        },
    }


def _positive_frame() -> dict:
    return {
        'response_id': 'resp_positive',
        'status': 'completed',
        'request': {'prompt': 'use the prior image and make it brighter'},
        'runtime': {
            'graph_closure_review': {
                'status': 'fulfilled',
                'counts': {'fulfilled': 1, 'pending': 0},
                'contract_source': 'request_ir',
                'checks': [
                    {
                        'check_kind': 'output_obligation',
                        'status': 'fulfilled',
                        'evidence': 'output slot fulfilled',
                    }
                ],
            }
        },
        'planning': {
            'context_contract': {
                'context_gate_review': {
                    'history_scan': {
                        'executed': True,
                        'matched_candidate_count': 2,
                        'promoted_candidate_count': 1,
                    }
                }
            }
        },
    }


def test_reverse_jsonl_reader_is_bounded_and_preserves_recent_record_semantics(tmp_path, monkeypatch):
    ledger = tmp_path / 'records.jsonl'
    middle_payload = {'id': 'middle', 'text': 'x' * 4096}
    ledger.write_bytes(
        (
            json.dumps({'id': 'old'})
            + '\n\n'
            + json.dumps(middle_payload)
            + '\n{malformed\n'
            + json.dumps(['valid', 'but', 'not', 'a', 'mapping'])
            + '\n'
            + json.dumps({'id': 'new', 'text': 'Grüezi'}, ensure_ascii=False)
        ).encode('utf-8')
    )
    monkeypatch.setattr('ollmo_services.self_learning._JSONL_REVERSE_READ_CHUNK_BYTES', 17)

    def fail_read_text(*_args, **_kwargs):
        raise AssertionError('bounded JSONL reads must not call Path.read_text')

    monkeypatch.setattr(Path, 'read_text', fail_read_text)

    records = list(self_learning_module._iter_jsonl(ledger, limit=4))
    compatibility_records = self_learning_module._read_jsonl(ledger, limit=4)

    assert [record['id'] for record in records] == ['new', 'middle']
    assert compatibility_records == records
    assert records[1]['text'] == middle_payload['text']


def test_reverse_jsonl_reader_preserves_malformed_and_non_mapping_limit_behavior(tmp_path):
    malformed_tail = tmp_path / 'malformed-tail.jsonl'
    malformed_tail.write_text('{"id":"older"}\n{malformed', encoding='utf-8')
    non_mapping_tail = tmp_path / 'non-mapping-tail.jsonl'
    non_mapping_tail.write_text('{"id":"older"}\n[1,2,3]', encoding='utf-8')

    assert [record['id'] for record in self_learning_module._iter_jsonl(malformed_tail, limit=1)] == ['older']
    assert list(self_learning_module._iter_jsonl(non_mapping_tail, limit=1)) == []
    assert [record['id'] for record in self_learning_module._iter_jsonl(malformed_tail, limit=0)] == ['older']


def test_self_learning_report_streams_response_frames_without_read_jsonl_list(tmp_path, monkeypatch):
    frames_path = tmp_path / 'responses.jsonl'
    frames = []
    for index in range(3):
        frame = _positive_frame()
        frame['response_id'] = f'resp_streamed_{index}'
        frames.append(frame)
    frames_path.write_text(
        ''.join(json.dumps(frame) + '\n' for frame in frames),
        encoding='utf-8',
    )

    def fail_list_reader(*_args, **_kwargs):
        raise AssertionError('response frames must stream through _iter_jsonl')

    monkeypatch.setattr('ollmo_services.self_learning._read_jsonl', fail_list_reader)
    report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=tmp_path / 'self_learning',
        frame_limit=3,
        max_cases=1,
    )

    assert report['frame_count'] == 3
    assert report['case_count'] == 1


def test_self_learning_report_evaluates_only_latest_frame_per_response(tmp_path):
    frames_path = tmp_path / 'responses.jsonl'
    response_id = 'resp_transition_resolved_by_successor'
    pending_frame = _problem_frame()
    pending_frame['response_id'] = response_id
    fulfilled_successor = _positive_frame()
    fulfilled_successor['response_id'] = response_id
    _write_jsonl(frames_path, [pending_frame, fulfilled_successor])

    report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=tmp_path / 'self_learning',
        frame_limit=20,
        max_cases=20,
    )

    case_kinds = {case['case_kind'] for case in report['eval_cases']}
    assert report['frame_count'] == 2
    assert report['evaluated_response_count'] == 1
    assert report['superseded_frame_count'] == 1
    assert 'fulfilled_graph_contract' in case_kinds
    assert 'open_output_slots' not in case_kinds
    assert 'open_graph_obligation' not in case_kinds
    assert report['improvement_candidate_count'] == 0


def _semantic_decision_frame(status: str = 'pending') -> dict:
    return {
        'response_id': f'resp_semantic_decision_{status}',
        'status': 'completed' if status == 'fulfilled' else 'partial',
        'request': {'prompt': 'generate an image, analyze it, then write a grounded assessment'},
        'runtime': {
            'graph_closure_review': {
                'status': status,
                'counts': {'pending': 1 if status != 'fulfilled' else 0, 'fulfilled': 1 if status == 'fulfilled' else 0},
                'decision_contract_review': {
                    'semantic_decision_review': {
                        'kind': 'ollmo.semantic_decision_review',
                        'status': 'active',
                        'proposal_count': 1,
                        'proposals': [
                            {
                                'proposal_id': 'semantic-decision-final-review',
                                'decision_action': 'semantic_review',
                                'reason': 'final assessment needs generated-image evidence',
                            }
                        ],
                    }
                },
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'status': status,
                    'category_counts': {
                        'open': 1 if status != 'fulfilled' else 0,
                        'semantic_review_pending': 1 if status != 'fulfilled' else 0,
                    },
                },
                'checks': [
                    {
                        'check_kind': 'output_obligation',
                        'status': 'pending' if status != 'fulfilled' else 'fulfilled',
                        'branch_id': 'branch-final-review',
                        'evidence': 'semantic_decision_review',
                    }
                ],
            }
        },
    }


def _controlled_attention_frame(status: str = 'pending') -> dict:
    return {
        'response_id': f'resp_controlled_attention_{status}',
        'status': 'completed' if status == 'fulfilled' else 'partial',
        'request': {'prompt': 'generate evidence, review it, and then decide whether to continue'},
        'runtime': {
            'graph_closure_review': {
                'status': status,
                'counts': {'pending': 1 if status != 'fulfilled' else 0, 'fulfilled': 1 if status == 'fulfilled' else 0},
                'decision_contract_review': {
                    'controlled_attention_review': {
                        'kind': 'ollmo.controlled_attention_review',
                        'status': 'active',
                        'frame_count': 1,
                        'frames': [
                            {
                                'frame_id': 'controlled-attention-branch-review',
                                'scope': 'block_resolution',
                                'priority': 'high',
                                'attention_question': 'What is the right-sized truthful transition for this block?',
                            }
                        ],
                    }
                },
                'checks': [
                    {
                        'check_kind': 'output_obligation',
                        'status': 'pending' if status != 'fulfilled' else 'fulfilled',
                        'branch_id': 'branch-review',
                        'evidence': 'controlled_attention_review',
                    }
                ],
            }
        },
    }


def _orientation_frame(review_key: str, status: str = 'pending') -> dict:
    layer = 'aspiration' if review_key == 'aspiration_review' else 'commitment'
    frame_key = 'aspiration_frame_count' if layer == 'aspiration' else 'commitment_frame_count'
    return {
        'response_id': f'resp_{layer}_{status}',
        'status': 'completed' if status == 'fulfilled' else 'partial',
        'request': {'prompt': 'plan a multi-step artifact workflow and resolve it truthfully'},
        'runtime': {
            'graph_closure_review': {
                'status': status,
                'counts': {'pending': 1 if status != 'fulfilled' else 0, 'fulfilled': 1 if status == 'fulfilled' else 0},
                'decision_contract_review': {
                    frame_key: 1,
                    review_key: {
                        'kind': f'ollmo.{review_key}',
                        'status': 'active',
                        'authority': 'advisory_read_model_only',
                        'frame_count': 1,
                        'frames': [
                            {
                                'frame_id': f'{layer}-frame',
                                f'{layer}_action': 'review_underplanned_graph' if layer == 'aspiration' else 'commit_to_right_sized_sufficient_transition',
                                'reason': 'orientation frame for eval extraction',
                            }
                        ],
                    },
                },
                'checks': [
                    {
                        'check_kind': 'output_obligation',
                        'status': 'pending' if status != 'fulfilled' else 'fulfilled',
                        'branch_id': f'branch-{layer}',
                        'evidence': review_key,
                    }
                ],
            }
        },
    }


def _global_semantic_closure_frame(status: str = 'pending') -> dict:
    return {
        'response_id': f'resp_global_semantic_closure_{status}',
        'status': 'completed' if status == 'fulfilled' else 'partial',
        'request': {'prompt': 'generate an image, analyze it, then write a grounded assessment'},
        'runtime': {
            'graph_closure_review': {
                'status': status,
                'counts': {'pending': 1 if status != 'fulfilled' else 0, 'fulfilled': 1 if status == 'fulfilled' else 0},
                'global_semantic_closure_review': {
                    'kind': 'ollmo.global_semantic_closure_review',
                    'status': status,
                    'proposal_count': 1 if status != 'fulfilled' else 0,
                    'reason': 'whole-turn semantic closure requires review before truthful freeze',
                    'proposals': [
                        {
                            'proposal_id': 'global-semantic-closure-whole-intent-fit',
                            'decision_action': 'semantic_review',
                        }
                    ] if status != 'fulfilled' else [],
                },
                'checks': [
                    {
                        'check_kind': 'global_semantic_closure',
                        'status': 'pending' if status != 'fulfilled' else 'fulfilled',
                        'branch_id': 'branch-global-semantic-closure-review',
                        'evidence': 'global_semantic_closure_review',
                    }
                ],
            }
        },
    }


def _semantic_verdict_frame(verdict: str = 'failed') -> dict:
    status = 'fulfilled' if verdict == 'passed' else ('blocked' if verdict == 'failed' else 'pending')
    return {
        'response_id': f'resp_semantic_verdict_{verdict}',
        'status': 'completed' if verdict == 'passed' else 'partial',
        'request': {'prompt': 'generate an image, analyze it, then write a grounded assessment'},
        'runtime': {
            'graph_closure_review': {
                'status': status,
                'global_semantic_closure_review': {
                    'kind': 'ollmo.global_semantic_closure_review',
                    'status': status,
                    'semantic_review_verdict': {
                        'kind': 'ollmo.semantic_review_verdict',
                        'verdict': verdict,
                        'status': status,
                        'recommended_transition': 'truthful_freeze' if verdict == 'passed' else 'repair_dependency_chain',
                        'evidence_refs': ['branch-final-review'],
                    },
                },
            }
        },
    }


def _branch_semantic_verdict_frame(verdict: str = 'failed') -> dict:
    status = 'fulfilled' if verdict == 'passed' else ('blocked' if verdict == 'failed' else 'pending')
    return {
        'response_id': f'resp_branch_semantic_verdict_{verdict}',
        'status': 'completed' if verdict == 'passed' else 'partial',
        'request': {'prompt': 'generate an image, then write a grounded assessment'},
        'runtime': {
            'graph_closure_review': {
                'status': status,
                'checks': [
                    {
                        'check_kind': 'branch_semantic_review',
                        'status': status,
                        'branch_id': 'branch-semantic-review-final',
                        'branch_semantic_review': {
                            'kind': 'ollmo.branch_semantic_review',
                            'status': status,
                            'source_branch_id': 'branch-final-review',
                        },
                        'semantic_review_verdict': {
                            'kind': 'ollmo.semantic_review_verdict',
                            'verdict': verdict,
                            'status': status,
                            'recommended_transition': 'truthful_freeze' if verdict == 'passed' else 'repair_dependency_chain',
                            'evidence_refs': ['branch-final-review'],
                        },
                    }
                ],
            }
        },
    }


def _artifact_collapse_frame() -> dict:
    return {
        'response_id': 'resp_artifact_collapse',
        'status': 'completed',
        'request': {
            'prompt': (
                'Create exactly three generated image artifacts plus index.html and styles.css. '
                'Do not return only code in chat; save real local artifacts.'
            )
        },
        'output': {
            'text': (
                '```json\n'
                '{"request_ir":{"output_obligations":[{"kind":"image_generation","count":3},'
                '{"kind":"file","name":"index.html"},{"kind":"file","name":"styles.css"}]}}\n'
                '```'
            ),
            'outputs': [{'type': 'message', 'text': 'request_ir'}],
        },
        'current_state': {
            'lifecycle_state': 'completed',
            'output_text': '{"request_ir":{"output_obligations":[]}}',
            'outputs': [{'type': 'message', 'text': 'request_ir'}],
        },
        'planning': {
            'artifact_flow': {
                'review': {
                    'status': 'pending',
                    'pending_output_slot_ids': ['slot-image-1', 'slot-html'],
                    'blocked_output_slot_ids': [],
                }
            }
        },
    }


def _materialization_failure_frame() -> dict:
    return {
        'response_id': 'resp_materialization_failure',
        'status': 'completed',
        'request': {'prompt': 'Create index.html, styles.css, and exactly three generated images.'},
        'current_state': {'lifecycle_state': 'repair_needed'},
        'artifacts': {
            'output': [
                {'kind': 'text', 'mime_type': 'text/html', 'path': '/tmp/index.html', 'artifact_ref': 'artifact:text_index'},
                {'kind': 'text', 'mime_type': 'text/css', 'path': '/tmp/styles.css', 'artifact_ref': 'artifact:text_styles'},
                {'kind': 'image', 'path': '/tmp/image-1.png', 'artifact_ref': 'artifact:image_1'},
            ]
        },
        'late_fill': {
            'status': 'partial_failed',
            'failed_branch_count': 2,
            'final_materialization_contract_status': 'unmet',
            'materialization_contract_unmet': True,
            'error': (
                'branch-text_artifact-1: Model request failed: 500 Server Error: Internal Server Error '
                'for url: http://127.0.0.1:11503/v1/chat/completions'
            ),
            'materialization_contract_open_checks': [
                {
                    'check_kind': 'text_artifact_syntax_sanity',
                    'status': 'pending',
                    'evidence': 'text_artifact_syntax_issue',
                    'reason': 'HTML has unsupported href element <can> at line 20; use <a> for navigation links',
                    'repair_action': 'retry_same_branch',
                },
                {
                    'check_kind': 'linked_artifact_rebind',
                    'status': 'pending',
                    'evidence': 'missing image link does not resolve',
                    'repair_action': 'rebind_dependency_evidence',
                },
            ],
            'failed_branches': [
                {
                    'branch_id': 'repair-chat',
                    'capability': 'chat',
                    'instance_id': 'mlx-community__gemma-4-e4b-8bit-mlx-11503',
                    'content_payload': 'HTML has unsupported href element <can> at line 20; use <a>.',
                    'error': {
                        'message': (
                            'Model request failed: 500 Server Error: Internal Server Error for url: '
                            'http://127.0.0.1:11503/v1/chat/completions'
                        )
                    },
                }
            ],
        },
    }


def _fulfilled_with_nonterminal_error_frame() -> dict:
    return {
        'response_id': 'resp_fulfilled_with_nonterminal_error',
        'status': 'completed',
        'request': {'prompt': 'Create a fulfilled page with images and files.'},
        'current_state': {'lifecycle_state': 'completed'},
        'artifacts': {
            'output': [
                {'kind': 'text', 'path': '/tmp/index.html', 'artifact_ref': 'artifact:text_index'},
                {'kind': 'image', 'path': '/tmp/image.png', 'artifact_ref': 'artifact:image_1'},
            ]
        },
        'late_fill': {
            'status': 'completed',
            'failed_branch_count': 0,
            'final_materialization_contract_status': 'fulfilled',
            'materialization_contract_unmet': False,
            'error_summary': {
                'preview': (
                    'coalesced-text branch: Model request failed: 500 Server Error for url: '
                    'http://127.0.0.1:11503/v1/chat/completions'
                )
            },
            'completed_branches': [
                {'branch_id': 'branch-image_generation-1', 'status': 'fulfilled', 'capability': 'image_generation'}
            ],
        },
    }


def _graph_repair_outcome_frame() -> dict:
    return {
        'response_id': 'resp_graph_repair_outcomes',
        'status': 'completed',
        'request': {'prompt': 'Create a fulfilled artifact page and repair only runtime-backed gaps.'},
        'runtime': {
            'graph_closure_review': {
                'status': 'fulfilled',
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'pending',
                    'category_counts': {
                        'controlled_attention_advisory': 2,
                        'aspiration_advisory': 1,
                        'commitment_advisory': 1,
                    },
                    'items': [
                        {'category': 'controlled_attention_advisory', 'status': 'active'},
                        {'category': 'aspiration_advisory', 'status': 'active'},
                        {'category': 'commitment_advisory', 'status': 'active'},
                    ],
                },
            },
            'request_phase_graph': {
                'graph_repair_proposals': [
                    {
                        'proposal_id': 'proposal-accepted',
                        'repair_type': 'repair_missing_materialization_contract',
                        'source': 'monitor_evidence',
                        'evidence_refs': ['monitor:materialization_contract_unmet'],
                    },
                    {
                        'proposal_id': 'proposal-rejected',
                        'repair_type': 'reconcile_surface_state_or_reopen_contract',
                        'source': 'monitor_evidence',
                        'evidence_refs': ['surface_state:pending'],
                    },
                ],
                'graph_repair_reviews': [
                    {
                        'review_id': 'review-accepted',
                        'proposal_id': 'proposal-accepted',
                        'status': 'accepted',
                        'reasons': [],
                    },
                    {
                        'review_id': 'review-rejected',
                        'proposal_id': 'proposal-rejected',
                        'status': 'rejected',
                        'reasons': ['deferred_or_reserved_intent_conflict'],
                    },
                    {
                        'review_id': 'review-unmatched',
                        'proposal_id': 'proposal-missing',
                        'status': 'rejected',
                        'reasons': ['runtime_evidence_missing'],
                    },
                ],
            },
        },
    }


def _graph_repair_missing_despite_evidence_frame() -> dict:
    return {
        'response_id': 'resp_graph_repair_missing_despite_evidence',
        'status': 'completed',
        'request': {'prompt': 'Create a grounded final review after branch semantic review.'},
        'runtime': {
            'graph_closure_review': {
                'status': 'fulfilled',
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'review_pending',
                    'category_counts': {'semantic_review_pending': 1},
                    'items': [
                        {
                            'category': 'semantic_review_pending',
                            'status': 'pending',
                            'check_kind': 'branch_semantic_review',
                            'branch_id': 'branch-final-review',
                        }
                    ],
                },
            },
            'request_phase_graph': {
                'graph_repair_proposals': [],
                'graph_repair_reviews': [],
            },
        },
    }


def _graph_patch_lifecycle_outcome_frame(closure_status: str = 'fulfilled') -> dict:
    return {
        'response_id': f'resp_graph_patch_lifecycle_{closure_status}',
        'status': 'completed' if closure_status == 'fulfilled' else 'repair_needed',
        'request': {'prompt': 'Create artifact work and repair only runtime-backed graph gaps.'},
        'runtime': {
            'graph_closure_review': {
                'status': closure_status,
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'blocked' if closure_status != 'fulfilled' else 'completed',
                    'category_counts': {'blocked': 1} if closure_status != 'fulfilled' else {'completed': 1},
                },
            },
            'developer_diagnostics': {
                'graph_patch_lifecycle': [
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-materialization',
                        'proposal_id': 'proposal-materialization',
                        'repair_class': 'missing_materialization_branch',
                        'status': 'applied',
                        'autonomy_level': 'apply_safe',
                        'risk_level': 'safe_additive',
                        'source_evidence_refs': ['late_fill:materialization_contract_unmet'],
                        'idempotency_key': 'idem-materialization',
                        'outcome': {'status': 'applied'},
                    },
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-conflict',
                        'proposal_id': 'proposal-conflict',
                        'repair_class': 'branch_identity_split',
                        'status': 'blocked',
                        'autonomy_level': 'apply_safe',
                        'risk_level': 'review_required',
                        'blocked_reasons': ['deferred_or_reserved_intent_conflict'],
                        'idempotency_key': 'idem-conflict',
                    },
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-degraded',
                        'proposal_id': 'proposal-degraded',
                        'repair_class': 'degraded_liveness_only',
                        'status': 'blocked',
                        'autonomy_level': 'apply_safe',
                        'risk_level': 'forbidden',
                        'source_evidence_refs': ['route_health:degraded'],
                        'blocked_reasons': ['degraded_liveness_advisory_not_graph_repair_evidence'],
                        'idempotency_key': 'idem-degraded',
                    },
                ]
            },
            'request_phase_graph': {
                'graph_patch_lifecycle': [
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-materialization',
                        'proposal_id': 'proposal-materialization',
                        'repair_class': 'missing_materialization_branch',
                        'status': 'applied',
                        'autonomy_level': 'apply_safe',
                        'risk_level': 'safe_additive',
                        'source_evidence_refs': ['late_fill:materialization_contract_unmet'],
                        'idempotency_key': 'idem-materialization',
                        'outcome': {'status': 'applied'},
                    }
                ]
            },
        },
    }


def _graph_patch_duplicate_lifecycle_frame() -> dict:
    return {
        'response_id': 'resp_graph_patch_duplicate_lifecycle',
        'status': 'repair_needed',
        'request': {'prompt': 'Keep lifecycle learning on final patch truth.'},
        'runtime': {
            'graph_closure_review': {
                'status': 'blocked',
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'blocked',
                    'category_counts': {'blocked': 1},
                },
            },
            'request_phase_graph': {
                'graph_patch_lifecycle': [
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-conflict-final',
                        'proposal_id': 'proposal-conflict-final',
                        'status': 'staged',
                        'idempotency_key': 'idem-conflict-final',
                    }
                ]
            },
            'developer_diagnostics': {
                'graph_patch_lifecycle_results': [
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-conflict-final',
                        'proposal_id': 'proposal-conflict-final',
                        'repair_class': 'branch_identity_split',
                        'status': 'blocked',
                        'autonomy_level': 'apply_reviewed',
                        'risk_level': 'review_required',
                        'blocked_reasons': ['apply_reviewed_requires_explicit_review_authorization'],
                        'source_evidence_refs': ['runtime_review:missing_authorization'],
                        'idempotency_key': 'idem-conflict-final',
                        'outcome': {
                            'status': 'blocked',
                            'runtime_effect': 'none',
                        },
                    }
                ]
            },
        },
    }


def _apply_enforced_patch_lifecycle_frame(closure_status: str = 'fulfilled') -> dict:
    return {
        'response_id': 'resp_apply_enforced_patch_lifecycle',
        'status': 'completed' if closure_status == 'fulfilled' else 'repair_needed',
        'request': {'prompt': 'Repair only runtime-backed graph gaps under enforced policy.'},
        'runtime': {
            'graph_closure_review': {
                'status': closure_status,
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'completed' if closure_status == 'fulfilled' else 'blocked',
                    'category_counts': {'completed': 1} if closure_status == 'fulfilled' else {'blocked': 1},
                },
            },
            'developer_diagnostics': {
                'graph_patch_lifecycle': [
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-enforced-materialization',
                        'proposal_id': 'proposal-enforced-materialization',
                        'repair_class': 'missing_materialization_branch',
                        'enforced_class': 'safe_additive_missing_branch',
                        'status': 'applied',
                        'autonomy_level': 'apply_enforced',
                        'authority': 'runtime_enforced_policy',
                        'risk_level': 'safe_additive',
                        'source_evidence_refs': ['closure:intent_graph_adequacy'],
                        'idempotency_key': 'idem-enforced-materialization',
                        'enforced_policy_review': {
                            'kind': 'ollmo.enforced_policy_review',
                            'policy_id': 'ollmo-enforced-policy-v1',
                            'status': 'allowed',
                            'allowed': True,
                            'policy_mode': 'safe_v1',
                            'enforced_class': 'safe_additive_missing_branch',
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                            'blocked_reasons': [],
                        },
                        'outcome': {'status': 'applied', 'runtime_effect': 'graph_mutated'},
                    },
                    {
                        'kind': 'ollmo.graph_patch_lifecycle',
                        'patch_id': 'patch-enforced-default-deny',
                        'proposal_id': 'proposal-enforced-default-deny',
                        'repair_class': 'missing_materialization_branch',
                        'enforced_class': 'safe_additive_missing_branch',
                        'status': 'blocked',
                        'autonomy_level': 'apply_enforced',
                        'authority': 'runtime_enforced_policy_denied',
                        'risk_level': 'safe_additive',
                        'source_evidence_refs': ['closure:intent_graph_adequacy'],
                        'blocked_reasons': ['enforced_policy_off'],
                        'idempotency_key': 'idem-enforced-default-deny',
                        'enforced_policy_review': {
                            'kind': 'ollmo.enforced_policy_review',
                            'policy_id': 'ollmo-enforced-policy-v1',
                            'status': 'blocked',
                            'allowed': False,
                            'policy_mode': 'off',
                            'enforced_class': 'safe_additive_missing_branch',
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                            'blocked_reasons': ['enforced_policy_off'],
                        },
                        'outcome': {'status': 'blocked', 'runtime_effect': 'none'},
                    },
                ]
            },
        },
    }


def _graph_rebase_outcome_frame(closure_status: str = 'blocked', successor_status: str = 'candidate') -> dict:
    return {
        'response_id': f'resp_graph_rebase_{closure_status}_{successor_status}',
        'status': 'completed' if closure_status == 'fulfilled' else 'repair_needed',
        'request': {'prompt': 'Create a linked local image site and preserve all generated assets.'},
        'runtime': {
            'graph_closure_review': {
                'status': closure_status,
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'blocked' if closure_status != 'fulfilled' else 'completed',
                    'category_counts': {'blocked': 1} if closure_status != 'fulfilled' else {'completed': 1},
                },
            },
            'request_phase_graph': {
                'graph_rebase_proposals': [
                    {
                        'kind': 'ollmo.graph_rebase_proposal',
                        'proposal_id': 'rebase-proposal-accepted',
                        'source': 'runtime_closure_review',
                        'evidence_refs': ['closure:intent_graph_adequacy'],
                    },
                    {
                        'kind': 'ollmo.graph_rebase_proposal',
                        'proposal_id': 'rebase-proposal-learning',
                        'source': 'accepted_learning',
                        'evidence_refs': ['accepted_learning:case-1'],
                    },
                    {
                        'kind': 'ollmo.graph_rebase_proposal',
                        'proposal_id': 'rebase-proposal-degraded',
                        'source': 'runtime_closure_review',
                        'evidence_refs': ['route_health:degraded_liveness_only'],
                    },
                ],
                'graph_rebase_reviews': [
                    {
                        'kind': 'ollmo.graph_rebase_review',
                        'review_id': 'rebase-review-accepted',
                        'proposal_id': 'rebase-proposal-accepted',
                        'status': 'accepted',
                        'blocked_reasons': [],
                    },
                    {
                        'kind': 'ollmo.graph_rebase_review',
                        'review_id': 'rebase-review-learning',
                        'proposal_id': 'rebase-proposal-learning',
                        'status': 'rejected',
                        'blocked_reasons': ['accepted_learning_not_rebase_authority'],
                    },
                    {
                        'kind': 'ollmo.graph_rebase_review',
                        'review_id': 'rebase-review-degraded',
                        'proposal_id': 'rebase-proposal-degraded',
                        'status': 'rejected',
                        'blocked_reasons': ['backend_route_health_signal_is_not_rebase_authority'],
                    },
                    {
                        'kind': 'ollmo.graph_rebase_review',
                        'review_id': 'rebase-review-preservation',
                        'proposal_id': 'rebase-proposal-preservation',
                        'status': 'blocked',
                        'blocked_reasons': ['lost_artifact_ref'],
                    },
                ],
                'graph_rebase_lifecycle': [
                    {
                        'kind': 'ollmo.graph_rebase_lifecycle',
                        'rebase_id': 'rebase-stage',
                        'proposal_id': 'rebase-proposal-accepted',
                        'status': 'staged',
                        'autonomy_level': 'stage',
                        'risk_level': 'reviewed_rebase',
                        'idempotency_key': 'idem-rebase-stage',
                        'outcome': {'status': 'staged', 'runtime_effect': 'staged_no_executable_mutation'},
                    },
                    {
                        'kind': 'ollmo.graph_rebase_lifecycle',
                        'rebase_id': 'rebase-successor',
                        'proposal_id': 'rebase-proposal-accepted',
                        'status': 'applied',
                        'autonomy_level': 'apply_reviewed',
                        'risk_level': 'reviewed_rebase',
                        'idempotency_key': 'idem-rebase-successor',
                        'outcome': {'status': 'applied', 'runtime_effect': 'successor_rebase_created'},
                    },
                    {
                        'kind': 'ollmo.graph_rebase_lifecycle',
                        'rebase_id': 'rebase-enforced',
                        'proposal_id': 'rebase-proposal-enforced',
                        'status': 'blocked',
                        'autonomy_level': 'apply_enforced',
                        'risk_level': 'reviewed_rebase',
                        'enforced_class': 'full_successor_rebase',
                        'blocked_reasons': ['full_successor_rebase_not_enforced_v1'],
                        'idempotency_key': 'idem-rebase-enforced',
                        'enforced_policy_review': {
                            'kind': 'ollmo.enforced_policy_review',
                            'policy_id': 'ollmo-enforced-policy-v1',
                            'status': 'blocked',
                            'allowed': False,
                            'policy_mode': 'safe_v1',
                            'enforced_class': 'full_successor_rebase',
                            'blocked_reasons': ['full_successor_rebase_not_enforced_v1'],
                            'current_evidence_refs': ['closure:intent_graph_adequacy'],
                        },
                        'outcome': {'status': 'blocked', 'runtime_effect': 'none'},
                    },
                ],
                'successor_rebase_requests': [
                    {
                        'kind': 'ollmo.graph_rebase_successor_request',
                        'rebase_id': 'rebase-successor',
                        'proposal_id': 'rebase-proposal-accepted',
                        'status': successor_status,
                        'runtime_effect': 'successor_rebase_created',
                        'parent_response_id': 'resp-parent',
                        'parent_frame_id': 'frame-parent',
                        'idempotency_key': 'idem-rebase-successor',
                    }
                ],
            },
        },
    }


def _partial_graph_rebase_execution_frame(
    *,
    execution_status: str = 'queued',
    closure_status: str = 'blocked',
    late_fill_status: str = 'pending',
    blocked_reasons: list[str] | None = None,
    root_prompt_replay: bool = False,
) -> dict:
    execution = {
        'kind': 'ollmo.graph_rebase_partial_successor_execution',
        'status': execution_status,
        'successor_key': 'partial-successor-key-1',
        'execution_key': 'partial-execution-key-1',
        'response_id': 'resp-partial-rebase-execution',
        'parent_frame_id': 'frame-partial-parent',
        'parent_frame_sequence': 4,
        'proposal_id': 'proposal-partial-reviewed',
        'review_id': 'review-partial-reviewed',
        'rebase_id': 'rebase-partial-reviewed',
        'authorization_record_id': 'operator-authorization-partial',
        'partial_rebase_depth': 1,
        'scheduled_branch_ids': ['branch-partial-image', 'branch-partial-page'],
        'root_prompt_replay': root_prompt_replay,
        'blocked_reasons': list(blocked_reasons or []),
    }
    return {
        'response_id': execution['response_id'],
        'status': 'completed' if closure_status == 'fulfilled' else 'repair_needed',
        'request': {'prompt': 'Complete only the reviewed missing local image subtree.'},
        'late_fill': {
            'status': late_fill_status,
            'partial_rebase_execution': dict(execution),
        },
        'runtime': {
            'graph_closure_review': {'status': closure_status},
            'request_phase_graph': {
                'successor_rebase_executions': [dict(execution)],
            },
            'developer_diagnostics': {
                'graph_rebase_partial_successor_execution': dict(execution),
            },
        },
    }


def _redraw_scope_ladder_frame() -> dict:
    partial_review = {
        'kind': 'ollmo.redraw_scope_ladder_review',
        'review_id': 'redraw-scope-partial',
        'status': 'selected',
        'selected_scope': 'partial_subtree_rebase',
        'selected_candidate': {
            'scope': 'partial_subtree_rebase',
            'scope_root_ids': ['phase-image', 'phase-html'],
            'runtime_action': 'reviewed_graph_rebase_only',
        },
        'blocked_reasons': [],
        'learning_orientation': {
            'used': True,
            'authority': 'soft_hint_only',
            'used_as_authority': False,
            'hint_refs': ['learning-redraw-scope'],
        },
        'artifact_identity': {'canonicalization_required': False},
    }
    blocked_review = {
        'kind': 'ollmo.redraw_scope_ladder_review',
        'review_id': 'redraw-scope-blocked',
        'status': 'blocked',
        'selected_scope': 'observe',
        'selected_candidate': {'scope': 'observe'},
        'blocked_reasons': [
            'current_runtime_evidence_required',
            'degraded_or_provider_signal_not_scope_authority',
            'conflicting_duplicate_artifact_ref',
        ],
        'learning_orientation': {
            'used': True,
            'authority': 'soft_hint_only',
            'used_as_authority': False,
        },
        'artifact_identity': {
            'canonicalization_required': True,
            'final_projection_blocked': True,
            'duplicate_refs': ['artifact:hero'],
            'conflicts': [{'artifact_ref': 'artifact:hero'}],
        },
    }
    return {
        'response_id': 'resp_redraw_scope_ladder',
        'status': 'repair_needed',
        'request': {'prompt': 'Create a linked local image site and repair only the needed scope.'},
        'runtime': {
            'graph_closure_review': {
                'status': 'repair_required',
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'blocked',
                    'category_counts': {'blocked': 1},
                },
            },
            'request_phase_graph': {
                'redraw_scope_ladder_review': partial_review,
                'redraw_scope_ladder_reviews': [blocked_review],
            },
            'developer_diagnostics': {
                'redraw_scope_ladder_review': partial_review,
                'redraw_scope_ladder_reviews': [blocked_review],
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows),
        encoding='utf-8',
    )


def test_eval_case_extraction_covers_intake_closure_context_and_artifacts() -> None:
    cases = build_eval_cases_from_response_frame(_problem_frame())
    kinds = {case['case_kind'] for case in cases}

    assert 'intent_graph_inadequacy' in kinds
    assert 'open_graph_obligation' in kinds
    assert 'history_scan_no_matches' in kinds
    assert 'open_output_slots' in kinds
    assert all(case['optimization_policy'] == 'proposal_only_reviewed_patch_required' for case in cases)


def test_eval_case_extraction_names_artifact_collapse_and_control_json_leak() -> None:
    cases = build_eval_cases_from_response_frame(_artifact_collapse_frame())
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['artifact_request_collapsed_to_plain_chat']['target_area'] == 'ghost_intake_graph_policy'
    assert by_kind['artifact_control_json_leaked_to_user']['target_area'] == 'ghost_decision_contract_policy'
    assert by_kind['open_output_slots_after_terminal_state']['target_area'] == 'artifact_fulfillment_policy'
    assert by_kind['artifact_request_collapsed_to_plain_chat']['metadata']['saved_artifact_count'] == 0


def test_eval_case_extraction_names_materialization_route_health_html_and_link_failures() -> None:
    cases = build_eval_cases_from_response_frame(_materialization_failure_frame())
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['materialization_contract_unmet']['target_area'] == 'closure_review_policy'
    assert by_kind[CHAT_ROUTE_HEALTH_CASE_KIND]['target_area'] == 'workload_decision_policy'
    assert by_kind['html_navigation_tag_typo']['target_area'] == 'artifact_fulfillment_policy'
    assert by_kind['broken_artifact_dependency_link']['target_area'] == 'artifact_fulfillment_policy'
    assert by_kind[CHAT_ROUTE_HEALTH_CASE_KIND]['metadata']['backend_family_hint'] == 'mlx'


def test_eval_case_extraction_names_target_text_artifact_binding_violation() -> None:
    frame = _materialization_failure_frame()
    frame['response_id'] = 'resp_target_text_artifact_binding_violation'
    frame['late_fill']['error'] = (
        'TEXT_ARTIFACT_TARGET_BINDING_VIOLATION: Target-bound repairs must update the requested target file; '
        'expected /tmp/styles.css but actual saved_text_path was /tmp/index.html. '
        'evidence=text_artifact_target_path_mismatch'
    )
    frame['late_fill']['failed_branches'] = [
        {
            'branch_id': 'branch-text_artifact-2',
            'capability': 'chat',
            'error': {
                'code': 'TEXT_ARTIFACT_TARGET_BINDING_VIOLATION',
                'expected_target_path': '/tmp/styles.css',
                'actual_saved_text_path': '/tmp/index.html',
            },
        }
    ]

    cases = build_eval_cases_from_response_frame(frame)
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['target_text_artifact_binding_violation']['severity'] == 'high'
    assert by_kind['target_text_artifact_binding_violation']['target_area'] == 'artifact_fulfillment_policy'
    assert 'ollmo_server/late_fill_runtime.py' in by_kind['target_text_artifact_binding_violation']['target_surfaces']


def test_eval_case_extraction_names_fulfilled_contract_with_nonterminal_failure() -> None:
    cases = build_eval_cases_from_response_frame(_fulfilled_with_nonterminal_error_frame())
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['nonterminal_failed_branch_with_fulfilled_contract']['severity'] == 'low'
    assert by_kind['nonterminal_failed_branch_with_fulfilled_contract']['target_area'] == 'closure_review_policy'
    assert by_kind[CHAT_ROUTE_HEALTH_CASE_KIND]['metadata']['backend_family_hint'] == 'chat_backend_family'


def test_eval_case_extraction_names_graph_repair_outcomes() -> None:
    cases = build_eval_cases_from_response_frame(_graph_repair_outcome_frame())
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['graph_repair_proposal_accepted']['severity'] == 'positive'
    assert by_kind['graph_repair_proposal_rejected']['target_area'] == 'graph_repair_policy'
    assert by_kind['graph_repair_proposal_rejected']['metadata']['reasons'] == ['deferred_or_reserved_intent_conflict']
    assert by_kind['graph_repair_review_unmatched']['metadata']['proposal_id'] == 'proposal-missing'
    assert by_kind['graph_repair_false_positive_advisory_surface']['target_area'] == 'graph_repair_policy'
    assert by_kind['graph_repair_false_positive_advisory_surface']['metadata']['surface_actionability'] == 'advisory'


def test_eval_case_extraction_names_graph_repair_missing_despite_actionable_evidence() -> None:
    cases = build_eval_cases_from_response_frame(_graph_repair_missing_despite_evidence_frame())
    missing_case = next(
        case for case in cases
        if case['case_kind'] == 'graph_repair_missing_despite_evidence'
    )

    assert missing_case['target_area'] == 'graph_repair_policy'
    assert missing_case['metadata']['surface_actionability'] == 'actionable'


def test_eval_case_extraction_does_not_call_active_late_fill_open_work_missing_repair() -> None:
    frame = {
        'response_id': 'resp-active-late-fill-open-work',
        'status': 'completed',
        'late_fill': {
            'status': 'pending',
            'pending_branch_count': 1,
            'materialization_contract_unmet': True,
            'final_materialization_contract_status': 'unmet',
        },
        'runtime': {
            'graph_closure_review': {
                'status': 'pending',
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'pending',
                    'active_categories': ['open', 'late_fill_pending'],
                    'items': [
                        {
                            'category': 'open',
                            'status': 'pending',
                            'obligation_id': 'obligation-image-1',
                            'branch_id': 'branch-image-1',
                        }
                    ],
                },
            },
            'request_phase_graph': {
                'graph_repair_proposals': [],
                'graph_repair_reviews': [],
            },
        },
    }

    cases = build_eval_cases_from_response_frame(frame)

    assert 'graph_repair_missing_despite_evidence' not in {
        case['case_kind'] for case in cases
    }


def test_eval_case_extraction_does_not_call_missing_source_truth_guard_missing_repair() -> None:
    frame = {
        'response_id': 'resp-missing-source-truth-guard',
        'status': 'repair_needed',
        'runtime': {
            'graph_closure_review': {
                'status': 'repair_needed',
                'surface_state': {
                    'kind': 'ollmo.surface_state',
                    'state': 'repair_needed',
                    'reason': 'artifact request used a demonstrative reference without a current or selected source',
                },
                'checks': [
                    {
                        'check_kind': 'truth_guard',
                        'status': 'pending',
                        'repair_action': 'request_or_select_source',
                    }
                ],
            },
            'request_phase_graph': {
                'graph_repair_proposals': [],
                'graph_repair_reviews': [],
            },
        },
    }

    cases = build_eval_cases_from_response_frame(frame)

    assert 'graph_repair_missing_despite_evidence' not in {
        case['case_kind'] for case in cases
    }


def test_eval_case_extraction_names_graph_patch_lifecycle_outcomes() -> None:
    cases = build_eval_cases_from_response_frame(_graph_patch_lifecycle_outcome_frame('fulfilled'))
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['graph_patch_applied_and_closed']['severity'] == 'positive'
    assert by_kind['graph_patch_solved_missing_obligation']['metadata']['repair_class'] == 'missing_materialization_branch'
    assert by_kind['graph_patch_rejected_due_to_conflict']['metadata']['blocked_reasons'] == ['deferred_or_reserved_intent_conflict']
    assert by_kind['graph_patch_degraded_signal_ignored']['target_area'] == 'graph_repair_policy'


def test_eval_case_extraction_names_apply_enforced_policy_outcomes() -> None:
    cases = build_eval_cases_from_response_frame(_apply_enforced_patch_lifecycle_frame('fulfilled'))
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['apply_enforced_safe_additive_solved']['target_area'] == 'apply_enforced_policy'
    assert by_kind['apply_enforced_safe_additive_solved']['metadata']['enforced_class'] == 'safe_additive_missing_branch'
    assert by_kind['apply_enforced_policy_blocked']['metadata']['blocked_reasons'] == ['enforced_policy_off']
    assert by_kind['apply_enforced_policy_blocked']['metadata']['enforced_policy_review']['allowed'] is False


def test_eval_case_extraction_names_graph_patch_applied_but_blocked() -> None:
    cases = build_eval_cases_from_response_frame(_graph_patch_lifecycle_outcome_frame('blocked'))
    blocked_case = next(
        case for case in cases
        if case['case_kind'] == 'graph_patch_applied_but_blocked'
    )

    assert blocked_case['severity'] == 'high'
    assert blocked_case['metadata']['closure_status'] == 'blocked'


def test_eval_case_extraction_prefers_final_duplicate_graph_patch_lifecycle_record() -> None:
    cases = build_eval_cases_from_response_frame(_graph_patch_duplicate_lifecycle_frame())
    blocked_case = next(
        case for case in cases
        if case['case_kind'] == 'graph_patch_rejected_due_to_conflict'
    )

    assert blocked_case['metadata']['patch_status'] == 'blocked'
    assert blocked_case['metadata']['repair_class'] == 'branch_identity_split'
    assert blocked_case['metadata']['risk_level'] == 'review_required'
    assert blocked_case['metadata']['blocked_reasons'] == ['apply_reviewed_requires_explicit_review_authorization']
    assert blocked_case['metadata']['source_evidence_refs'] == ['runtime_review:missing_authorization']


def test_eval_case_extraction_names_graph_rebase_outcomes() -> None:
    cases = build_eval_cases_from_response_frame(_graph_rebase_outcome_frame())
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['graph_rebase_proposal_accepted']['severity'] == 'positive'
    assert by_kind['graph_rebase_proposal_rejected']['target_area'] == 'graph_rebase_policy'
    assert by_kind['graph_rebase_preservation_failed']['metadata']['blocked_reasons'] == ['lost_artifact_ref']
    assert by_kind['graph_rebase_staged']['metadata']['runtime_effect'] == 'staged_no_executable_mutation'
    assert by_kind['graph_rebase_successor_created']['metadata']['successor_status'] == 'candidate'
    assert by_kind['graph_rebase_rejected_learning_only']['metadata']['blocked_reasons'] == [
        'accepted_learning_not_rebase_authority'
    ]
    assert by_kind['graph_rebase_rejected_degraded_signal']['target_area'] == 'graph_rebase_policy'
    assert by_kind['graph_rebase_apply_enforced_blocked']['metadata']['autonomy_level'] == 'apply_enforced'
    assert by_kind['apply_enforced_full_rebase_blocked']['target_area'] == 'apply_enforced_policy'


def test_eval_case_extraction_names_graph_rebase_successor_solved() -> None:
    cases = build_eval_cases_from_response_frame(_graph_rebase_outcome_frame('fulfilled', 'solved'))
    solved_case = next(
        case for case in cases
        if case['case_kind'] == 'graph_rebase_successor_solved'
    )

    assert solved_case['severity'] == 'positive'
    assert solved_case['metadata']['closure_status'] == 'fulfilled'


def test_eval_case_extraction_observes_partial_rebase_successor_execution_once() -> None:
    cases = build_eval_cases_from_response_frame(
        _partial_graph_rebase_execution_frame()
    )
    execution_cases = [
        case
        for case in cases
        if case['case_kind'].startswith(
            'graph_rebase_partial_successor_execution_'
        )
    ]

    assert len(execution_cases) == 1
    created = execution_cases[0]
    assert created['case_kind'] == 'graph_rebase_partial_successor_execution_created'
    assert created['severity'] == 'medium'
    assert created['metadata']['execution_status'] == 'queued'
    assert created['metadata']['scheduled_branch_count'] == 2
    assert created['metadata']['authority'] == 'non_authoritative_observer'
    assert created['metadata']['observer_runtime_effect'] == 'none'
    assert created['metadata']['successor_rebase_execution']['kind'] == (
        'ollmo.graph_rebase_partial_successor_execution'
    )


def test_eval_case_extraction_observes_partial_rebase_successor_terminal_outcomes() -> None:
    solved_cases = build_eval_cases_from_response_frame(
        _partial_graph_rebase_execution_frame(
            closure_status='fulfilled',
            late_fill_status='completed',
        )
    )
    blocked_cases = build_eval_cases_from_response_frame(
        _partial_graph_rebase_execution_frame(
            execution_status='failed',
            closure_status='blocked',
            late_fill_status='partial_failed',
            blocked_reasons=['partial_rebase_branch_failed'],
        )
    )

    solved = next(
        case
        for case in solved_cases
        if case['case_kind'] == 'graph_rebase_partial_successor_execution_solved'
    )
    blocked = next(
        case
        for case in blocked_cases
        if case['case_kind'] == 'graph_rebase_partial_successor_execution_blocked'
    )

    assert solved['severity'] == 'positive'
    assert solved['metadata']['closure_status'] == 'fulfilled'
    assert solved['metadata']['late_fill_status'] == 'completed'
    assert blocked['severity'] == 'high'
    assert blocked['metadata']['blocked_reasons'] == [
        'partial_rebase_branch_failed'
    ]


def test_partial_rebase_execution_learning_remains_soft_and_non_authoritative() -> None:
    cases = build_eval_cases_from_response_frame(
        _partial_graph_rebase_execution_frame()
    )
    candidate = next(
        candidate
        for candidate in build_policy_improvement_candidates(cases)
        if candidate['target_area'] == 'graph_rebase_policy'
    )
    snapshot = promote_policy_improvement_candidate(
        None,
        candidate,
        reviewer='test',
        accepted_at='2026-07-19T17:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='partial rebase execution observer boundary test',
        changed_at='2026-07-19T17:01:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    assert hints['hint_count'] == 1
    assert hints['hints'][0]['allowed_use'] == 'soft_hint_only'
    assert 'cannot promote a rollout gate' in hints['hints'][0]['hint']
    assert 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth' in (
        hints['hints'][0]['forbidden_use']
    )


def test_eval_case_extraction_names_redraw_scope_ladder_outcomes() -> None:
    cases = build_eval_cases_from_response_frame(_redraw_scope_ladder_frame())
    by_kind = {case['case_kind']: case for case in cases}

    assert by_kind['redraw_scope_selected']['metadata']['selected_scope'] == 'partial_subtree_rebase'
    assert by_kind['redraw_scope_partial_subtree_rebase_selected']['target_area'] == 'redraw_scope_policy'
    assert by_kind['redraw_scope_learning_orientation_soft_hint']['metadata']['learning_orientation']['used_as_authority'] is False
    assert by_kind['redraw_scope_non_authoritative_evidence_ignored']['severity'] == 'positive'
    assert by_kind['redraw_scope_duplicate_artifact_ref_conflict']['metadata']['artifact_identity']['final_projection_blocked'] is True


def test_accepted_redraw_scope_learning_remains_soft_hint_only() -> None:
    cases = build_eval_cases_from_response_frame(_redraw_scope_ladder_frame())
    candidates = build_policy_improvement_candidates(cases)
    candidate = next(candidate for candidate in candidates if candidate['target_area'] == 'redraw_scope_policy')
    snapshot = promote_policy_improvement_candidate(
        None,
        candidate,
        reviewer='test',
        accepted_at='2026-06-07T17:48:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='redraw scope orientation soft hint test',
        changed_at='2026-06-07T17:49:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    assert hints['hint_count'] == 1
    assert hints['hints'][0]['target_area'] == 'redraw_scope_policy'
    assert hints['hints'][0]['allowed_use'] == 'soft_hint_only'
    assert 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth' in hints['hints'][0]['forbidden_use']


def test_accepted_graph_repair_learning_remains_soft_hint_only() -> None:
    cases = build_eval_cases_from_response_frame(_graph_repair_outcome_frame())
    candidates = build_policy_improvement_candidates(cases)
    candidate = next(candidate for candidate in candidates if candidate['target_area'] == 'graph_repair_policy')
    snapshot = promote_policy_improvement_candidate(
        None,
        candidate,
        reviewer='test',
        accepted_at='2026-06-06T19:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='graph repair repair-outcome soft hint test',
        changed_at='2026-06-06T19:01:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    assert hints['hint_count'] == 1
    assert hints['hints'][0]['target_area'] == 'graph_repair_policy'
    assert hints['hints'][0]['allowed_use'] == 'soft_hint_only'
    assert 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth' in hints['hints'][0]['forbidden_use']


def _accepted_learning_snapshot_with_targets(target_areas: list[str], *, enabled: bool = True) -> dict:
    return {
        'kind': 'ollmo.accepted_learning_policy_snapshot',
        'status': 'enabled' if enabled else 'disabled',
        'enabled': enabled,
        'authority': 'soft_hint',
        'accepted_learning_count': len(target_areas),
        'accepted_learnings': [
            {
                'learning_id': f'accepted-policy-improvement-{target_area}',
                'candidate_id': f'policy-improvement-{target_area}',
                'target_area': target_area,
                'bounded_hint': f'Keep {target_area} as reviewed soft orientation only.',
                'case_kinds': {f'{target_area}_case': index + 1},
                'severity_counts': {'medium': index + 1},
                'evidence_case_ids': [f'eval-{index + 1}'],
            }
            for index, target_area in enumerate(target_areas)
        ],
    }


def test_accepted_learning_runtime_hints_include_all_valid_learnings_by_default() -> None:
    snapshot = _accepted_learning_snapshot_with_targets(
        [
            'closure_review_policy',
            'artifact_fulfillment_policy',
            'ghost_decision_contract_policy',
            'semantic_decision_policy',
            'workload_decision_policy',
            'graph_repair_policy',
        ]
    )

    hints = build_accepted_learning_runtime_hints(snapshot)
    target_areas = [hint['target_area'] for hint in hints['hints']]

    assert hints['hint_count'] == 6
    assert len(hints['hints']) == 6
    assert 'graph_repair_policy' in target_areas
    assert all(hint['allowed_use'] == 'soft_hint_only' for hint in hints['hints'])
    assert all(
        'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth' in hint['forbidden_use']
        for hint in hints['hints']
    )


def test_accepted_learning_runtime_hints_respect_explicit_limit() -> None:
    snapshot = _accepted_learning_snapshot_with_targets(
        [
            'closure_review_policy',
            'artifact_fulfillment_policy',
            'ghost_decision_contract_policy',
            'semantic_decision_policy',
            'workload_decision_policy',
            'graph_repair_policy',
        ]
    )

    hints = build_accepted_learning_runtime_hints(snapshot, limit=5)
    target_areas = [hint['target_area'] for hint in hints['hints']]

    assert hints['hint_count'] == 5
    assert len(hints['hints']) == 5
    assert 'graph_repair_policy' not in target_areas


def test_accepted_learning_runtime_hints_keep_disabled_invalid_and_soft_hint_safety() -> None:
    target_areas = [
        'closure_review_policy',
        'invalid_policy_area',
        'graph_repair_policy',
    ]
    disabled = _accepted_learning_snapshot_with_targets(target_areas, enabled=False)
    enabled = _accepted_learning_snapshot_with_targets(target_areas, enabled=True)

    disabled_hints = build_accepted_learning_runtime_hints(disabled)
    enabled_hints = build_accepted_learning_runtime_hints(enabled)
    target_areas = [hint['target_area'] for hint in enabled_hints['hints']]

    assert disabled_hints['hint_count'] == 0
    assert disabled_hints['runtime_effect'] == 'none'
    assert enabled_hints['hint_count'] == 2
    assert target_areas == ['closure_review_policy', 'graph_repair_policy']
    assert all(hint['allowed_use'] == 'soft_hint_only' for hint in enabled_hints['hints'])
    assert all(
        hint['conflict_boundary']
        == 'Ignored whenever current user intent, live capability evidence, Graph, IR, Closure Review, output obligations, artifact evidence, or runtime truth conflict.'
        for hint in enabled_hints['hints']
    )


def test_policy_candidates_preserve_specific_target_areas_for_named_failures() -> None:
    cases = []
    cases.extend(build_eval_cases_from_response_frame(_artifact_collapse_frame()))
    cases.extend(build_eval_cases_from_response_frame(_materialization_failure_frame()))
    cases.extend(build_eval_cases_from_response_frame(_fulfilled_with_nonterminal_error_frame()))

    candidates = build_policy_improvement_candidates(cases)
    target_areas = {candidate['target_area'] for candidate in candidates}

    assert 'ghost_intake_graph_policy' in target_areas
    assert 'ghost_decision_contract_policy' in target_areas
    assert 'closure_review_policy' in target_areas
    assert 'workload_decision_policy' in target_areas
    assert 'artifact_fulfillment_policy' in target_areas


def test_shadow_hints_are_visible_and_inert() -> None:
    cases = build_eval_cases_from_response_frame(_artifact_collapse_frame())
    candidates = build_policy_improvement_candidates(cases)
    hints = build_shadow_learning_hints({'improvement_candidates': candidates})

    assert hints
    assert all(hint['authority'] == 'shadow' for hint in hints)
    assert all(hint['runtime_effect'] == 'none' for hint in hints)
    assert all('runtime truth' in hint['conflict_boundary'] for hint in hints)


def test_monitor_report_enrichment_adds_supporting_evidence_without_replacing_frame_truth() -> None:
    frame_cases = build_eval_cases_from_response_frame(_positive_frame())
    monitor_report = {
        'response_id': 'resp_monitor_support',
        'status': 'completed',
        'lifecycle_state': 'repair_needed',
        'late_fill_status': 'partial_failed',
        'final_materialization_contract_status': 'unmet',
        'materialization_contract_unmet': True,
        'failed_branch_count': 1,
        'verdict': 'needs_attention',
        'notes': [
            'Model request failed: 500 Server Error: Internal Server Error for url: http://127.0.0.1:11503/v1/chat/completions'
        ],
        'artifacts': {
            'html_issues': ['HTML has unsupported href element <can> at line 20; use <a> for navigation links'],
            'html_image_links': [{'src': 'missing.png', 'exists': False}],
        },
        'timing_diagnostics': {
            'branches': [
                {'branch_id': 'coalesced-text', 'status': 'error', 'instance_id': 'mlx-community__gemma'}
            ]
        },
    }

    enriched = enrich_eval_cases_from_monitor_reports(frame_cases, [monitor_report])
    by_kind = {case['case_kind']: case for case in enriched}

    assert 'fulfilled_graph_contract' in by_kind
    assert by_kind['materialization_contract_unmet']['metadata']['truth_source'] == 'monitor_supporting_evidence'
    assert by_kind['html_navigation_tag_typo']['metadata']['truth_source'] == 'monitor_supporting_evidence'
    assert by_kind['broken_artifact_dependency_link']['metadata']['truth_source'] == 'monitor_supporting_evidence'


def test_eval_case_extraction_covers_semantic_decision_review() -> None:
    cases = build_eval_cases_from_response_frame(_semantic_decision_frame('pending'))
    semantic_case = next(
        case for case in cases
        if case['case_kind'] == 'semantic_decision_proposals_unresolved'
    )

    assert semantic_case['layer'] == 'semantic_decision'
    assert semantic_case['target_area'] == 'semantic_decision_policy'
    assert semantic_case['severity'] == 'medium'
    assert semantic_case['metadata']['semantic_decision_review']['proposal_count'] == 1


def test_positive_semantic_decision_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_semantic_decision_frame('fulfilled'))
    kinds = {case['case_kind'] for case in cases}
    assert 'semantic_decision_review_resolved' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_eval_case_extraction_covers_controlled_attention_review() -> None:
    cases = build_eval_cases_from_response_frame(_controlled_attention_frame('pending'))
    attention_case = next(
        case for case in cases
        if case['case_kind'] == 'controlled_attention_unresolved'
    )

    assert attention_case['layer'] == 'controlled_attention'
    assert attention_case['target_area'] == 'controlled_attention_policy'
    assert attention_case['metadata']['controlled_attention_review']['frame_count'] == 1


def test_positive_controlled_attention_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_controlled_attention_frame('fulfilled'))
    kinds = {case['case_kind'] for case in cases}
    assert 'controlled_attention_resolved' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_eval_case_extraction_covers_aspiration_review() -> None:
    cases = build_eval_cases_from_response_frame(_orientation_frame('aspiration_review', 'pending'))
    aspiration_case = next(
        case for case in cases
        if case['case_kind'] == 'aspiration_review_unresolved'
    )

    assert aspiration_case['layer'] == 'aspiration'
    assert aspiration_case['target_area'] == 'aspiration_policy'
    assert aspiration_case['metadata']['aspiration_review']['frame_count'] == 1


def test_positive_aspiration_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_orientation_frame('aspiration_review', 'fulfilled'))
    kinds = {case['case_kind'] for case in cases}
    assert 'aspiration_review_resolved' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_eval_case_extraction_covers_commitment_review() -> None:
    cases = build_eval_cases_from_response_frame(_orientation_frame('commitment_review', 'pending'))
    commitment_case = next(
        case for case in cases
        if case['case_kind'] == 'commitment_review_unresolved'
    )

    assert commitment_case['layer'] == 'commitment'
    assert commitment_case['target_area'] == 'commitment_policy'
    assert commitment_case['metadata']['commitment_review']['frame_count'] == 1


def test_positive_commitment_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_orientation_frame('commitment_review', 'fulfilled'))
    kinds = {case['case_kind'] for case in cases}
    assert 'commitment_review_resolved' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_eval_case_extraction_covers_global_semantic_closure_review() -> None:
    cases = build_eval_cases_from_response_frame(_global_semantic_closure_frame('pending'))
    semantic_case = next(
        case for case in cases
        if case['case_kind'] == 'global_semantic_closure_unresolved'
    )

    assert semantic_case['layer'] == 'semantic_closure'
    assert semantic_case['target_area'] == 'semantic_review_policy'
    assert semantic_case['metadata']['global_semantic_closure_review']['proposal_count'] == 1


def test_eval_case_extraction_covers_failed_semantic_review_verdict() -> None:
    cases = build_eval_cases_from_response_frame(_semantic_verdict_frame('failed'))
    verdict_case = next(
        case for case in cases
        if case['case_kind'] == 'semantic_review_verdict_failed'
    )

    assert verdict_case['layer'] == 'semantic_verdict'
    assert verdict_case['target_area'] == 'semantic_verdict_policy'
    assert verdict_case['severity'] == 'high'
    assert verdict_case['metadata']['semantic_review_verdict']['recommended_transition'] == 'repair_dependency_chain'


def test_positive_semantic_review_verdict_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_semantic_verdict_frame('passed'))
    kinds = {case['case_kind'] for case in cases}
    assert 'semantic_review_verdict_passed' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_eval_case_extraction_covers_failed_branch_semantic_review_verdict() -> None:
    cases = build_eval_cases_from_response_frame(_branch_semantic_verdict_frame('failed'))
    verdict_case = next(
        case for case in cases
        if case['case_kind'] == 'branch_semantic_review_verdict_failed'
    )

    assert verdict_case['layer'] == 'semantic_verdict'
    assert verdict_case['target_area'] == 'semantic_verdict_policy'
    assert verdict_case['severity'] == 'high'
    assert verdict_case['metadata']['semantic_review_verdict']['recommended_transition'] == 'repair_dependency_chain'


def test_positive_branch_semantic_review_verdict_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_branch_semantic_verdict_frame('passed'))
    kinds = {case['case_kind'] for case in cases}
    assert 'branch_semantic_review_verdict_passed' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_positive_global_semantic_closure_trace_is_not_clustered_into_improvement_candidate() -> None:
    cases = build_eval_cases_from_response_frame(_global_semantic_closure_frame('fulfilled'))
    kinds = {case['case_kind'] for case in cases}
    assert 'global_semantic_closure_resolved' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_positive_traces_are_kept_but_not_clustered_into_improvement_candidates() -> None:
    cases = build_eval_cases_from_response_frame(_positive_frame())
    kinds = {case['case_kind'] for case in cases}
    assert 'fulfilled_graph_contract' in kinds
    assert 'history_scan_promoted_matches' in kinds

    candidates = build_policy_improvement_candidates(cases)
    assert candidates == []


def test_self_learning_report_reads_frame_ledger_and_can_omit_case_dump(tmp_path: Path) -> None:
    ledger = tmp_path / 'responses.jsonl'
    _write_jsonl(ledger, [_positive_frame(), _problem_frame()])

    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )

    assert report['status'] == 'completed'
    assert report['frame_count'] == 2
    assert report['case_count'] >= 4
    assert report['improvement_candidate_count'] >= 1
    assert 'eval_cases' not in report


def test_eval_cases_can_be_persisted_as_jsonl(tmp_path: Path) -> None:
    cases = build_eval_cases_from_response_frame(_problem_frame())
    output = persist_eval_cases(cases, output_path=tmp_path / 'eval_cases.jsonl')

    rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(cases)
    assert rows[0]['kind'] == 'ollmo.self_learning_eval_case'


def test_accepted_learning_policy_snapshot_defaults_to_disabled(tmp_path: Path) -> None:
    snapshot = load_accepted_learning_policy_snapshot(snapshot_path=tmp_path / 'accepted_policy_snapshot.json')

    assert snapshot['kind'] == 'ollmo.accepted_learning_policy_snapshot'
    assert snapshot['enabled'] is False
    assert snapshot['runtime_effect'] == 'none'
    assert snapshot['activation_policy'] == 'disabled_until_explicit_review'


def test_accepted_learning_policy_snapshot_can_be_initialized_disabled(tmp_path: Path) -> None:
    path = persist_accepted_learning_policy_snapshot(output_path=tmp_path / 'accepted_policy_snapshot.json')
    snapshot = load_accepted_learning_policy_snapshot(snapshot_path=path)

    assert snapshot['status'] == 'not_configured'
    assert snapshot['enabled'] is False
    assert snapshot['accepted_learning_count'] == 0


def test_promoted_learning_stays_inert_until_snapshot_is_enabled(tmp_path: Path) -> None:
    ledger = tmp_path / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])
    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )
    candidate = report['improvement_candidates'][0]

    snapshot = promote_policy_improvement_candidate(
        None,
        candidate,
        reviewer='test',
        review_note='accepted for bounded hint test',
        accepted_at='2026-05-03T10:00:00Z',
    )
    disabled_hints = build_accepted_learning_runtime_hints(snapshot)
    enabled_snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='bounded hint activation test',
        changed_at='2026-05-03T10:01:00Z',
    )
    enabled_hints = build_accepted_learning_runtime_hints(enabled_snapshot)

    assert snapshot['enabled'] is False
    assert snapshot['runtime_effect'] == 'none'
    assert snapshot['accepted_learning_count'] == 1
    assert disabled_hints['hint_count'] == 0
    assert disabled_hints['runtime_effect'] == 'none'
    assert enabled_snapshot['enabled'] is True
    assert enabled_hints['hint_count'] == 1
    assert enabled_hints['hints'][0]['allowed_use'] == 'soft_hint_only'
    assert 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth' in enabled_hints['hints'][0]['forbidden_use']


def test_accepted_learning_policy_preserves_future_authority_knob(tmp_path: Path) -> None:
    ledger = tmp_path / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])
    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )
    snapshot = promote_policy_improvement_candidate(
        {'authority': 'enforced'},
        report['improvement_candidates'][0],
        reviewer='test',
        accepted_at='2026-05-03T10:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='future authority schema test',
        changed_at='2026-05-03T10:01:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    assert hints['authority'] == 'enforced'
    assert hints['hints'][0]['authority'] == 'enforced'
    assert hints['hints'][0]['allowed_use'] == 'enforced_accepted_learning_hint'
    assert 'runtime_truth' in hints['hints'][0]['forbidden_use']


def test_accepted_learning_runtime_hint_keeps_actionable_case_kind_guidance() -> None:
    snapshot = promote_policy_improvement_candidate(
        None,
        {
            'candidate_id': 'policy-improvement-workload_decision_policy',
            'target_area': 'workload_decision_policy',
            'summary': 'Repeated chat repair failures happened on a provider route.',
            'case_kinds': {CHAT_ROUTE_HEALTH_CASE_KIND: 3},
            'severity_counts': {'medium': 3},
            'evidence_case_ids': ['eval-workload-1'],
        },
        reviewer='test',
        accepted_at='2026-05-09T10:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='actionable hint test',
        changed_at='2026-05-09T10:01:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    hint = hints['hints'][0]
    assert hints['runtime_effect'] == 'soft_hints_available'
    assert 'route-health preference evidence' in hint['hint']
    assert 'current runtime truth permits it' in hint['hint']
    assert 'provider bans' in hint['hint']
    assert 'offline state' in hint['hint']
    assert 'hard degraded truth' in hint['hint']
    assert 'graph repair proof' in hint['hint']
    assert hint['case_kinds'] == {CHAT_ROUTE_HEALTH_CASE_KIND: 3}
    assert hint['severity_counts'] == {'medium': 3}
    assert 'runtime truth conflict' in hint['conflict_boundary']


def test_accepted_learning_runtime_hint_sanitizes_legacy_route_health_snapshot() -> None:
    legacy_case_kind = 'fra' + 'gile_chat_' + 'provider_' + 'family'
    legacy_hint = (
        'Prefer robust chat/text-repair routes when recent evidence shows a provider ' + 'family is '
        + 'fra' + 'gile for chat completions; do not treat this as a global modality '
        'ban.'
    )
    snapshot = {
        'enabled': True,
        'authority': 'soft_hint',
        'status': 'enabled',
        'accepted_learnings': [
            {
                'learning_id': 'accepted-policy-improvement-workload_decision_policy',
                'candidate_id': 'policy-improvement-workload_decision_policy',
                'target_area': 'workload_decision_policy',
                'bounded_hint': legacy_hint,
                'case_kinds': {legacy_case_kind: 9},
                'severity_counts': {'medium': 9},
                'evidence_case_ids': ['eval-workload-legacy'],
            }
        ],
    }

    hints = build_accepted_learning_runtime_hints(snapshot)
    hint_text = hints['hints'][0]['hint']
    legacy_status_word = 'fra' + 'gile'
    legacy_provider_phrase = 'provider ' + 'family is ' + legacy_status_word
    legacy_ban_phrase = 'global modality ' + 'ban'

    assert legacy_provider_phrase not in hint_text
    assert legacy_ban_phrase not in hint_text
    assert hints['hints'][0]['case_kinds'] == {CHAT_ROUTE_HEALTH_CASE_KIND: 9}
    assert 'route-health preference evidence' in hint_text
    assert 'current runtime truth permits it' in hint_text
    assert 'provider bans' in hint_text
    assert 'offline state' in hint_text
    assert 'hard degraded truth' in hint_text
    assert 'graph repair proof' in hint_text
    assert hints['hints'][0]['allowed_use'] == 'soft_hint_only'
    assert 'do_not_mutate_graph_ir_closure_or_routing_without_runtime_truth' in hints['hints'][0]['forbidden_use']


def test_accepted_learning_runtime_hint_names_basic_intent_graph_repair_boundary() -> None:
    snapshot = promote_policy_improvement_candidate(
        None,
        {
            'candidate_id': 'policy-improvement-ghost_intake_graph_policy',
            'target_area': 'ghost_intake_graph_policy',
            'summary': 'Basic current-turn intent was underrepresented in the graph.',
            'case_kinds': {'intent_graph_inadequacy': 2, 'open_graph_obligation': 1},
            'severity_counts': {'high': 2, 'medium': 1},
            'evidence_case_ids': ['eval-intent-1'],
        },
        reviewer='test',
        accepted_at='2026-06-06T10:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='basic intent graph repair hint test',
        changed_at='2026-06-06T10:01:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    hint = hints['hints'][0]
    assert 'basic current-turn intent' in hint['hint']
    assert 'bounded graph repair' in hint['hint']
    assert 'executable truth' in hint['hint']
    assert hint['case_kinds']['intent_graph_inadequacy'] == 2
    assert 'runtime truth conflict' in hint['conflict_boundary']


def test_decision_contract_preserves_accepted_learning_hint_details() -> None:
    accepted_learning_hints = {
        'kind': 'ollmo.accepted_learning_runtime_hints',
        'enabled': True,
        'authority': 'soft_hint',
        'status': 'active',
        'runtime_effect': 'soft_hints_available',
        'hint_count': 1,
        'hints': [
            {
                'kind': 'ollmo.accepted_learning_runtime_hint',
                'learning_id': 'accepted-policy-improvement-artifact_fulfillment_policy',
                'candidate_id': 'policy-improvement-artifact_fulfillment_policy',
                'target_area': 'artifact_fulfillment_policy',
                'hint': 'Run deterministic linked-artifact rebind before closure.',
                'case_kinds': {'broken_artifact_dependency_link': 2},
                'authority': 'soft_hint',
                'allowed_use': 'soft_hint_only',
            }
        ],
    }

    contract = build_ghost_decision_contract(
        candidate_graph={},
        promotion_review={},
        workload_graph={},
        workload_proposal_review={},
        output_obligations=[],
        accepted_learning_hints=accepted_learning_hints,
    )

    summary_hint = contract['accepted_learning']['hints'][0]
    attention_frame = next(
        frame
        for frame in contract['controlled_attention_frames']
        if frame['source_kind'] == 'accepted_learning_hint'
    )
    orientation_hint = attention_frame['learning_orientation']['hints'][0]

    assert summary_hint['hint'] == 'Run deterministic linked-artifact rebind before closure.'
    assert summary_hint['case_kinds'] == {'broken_artifact_dependency_link': 2}
    assert orientation_hint['hint'] == 'Run deterministic linked-artifact rebind before closure.'
    assert orientation_hint['case_kinds'] == {'broken_artifact_dependency_link': 2}
    assert attention_frame['priority'] == 'low'
    assert attention_frame['non_authority_boundary'] == 'attention_only_runtime_contracts_closure_decide_truth'


def test_accepted_learning_allows_ghost_decision_contract_target_area() -> None:
    snapshot = promote_policy_improvement_candidate(
        None,
        {
            'candidate_id': 'policy-improvement-ghost-decision-contract',
            'target_area': 'ghost_decision_contract_policy',
            'summary': 'Keep reconsideration and supersession visible to Ghost as soft orientation.',
            'evidence_case_ids': ['eval-1'],
        },
        reviewer='test',
        accepted_at='2026-05-09T10:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='decision contract hint area test',
        changed_at='2026-05-09T10:01:00Z',
    )
    hints = build_accepted_learning_runtime_hints(snapshot)

    assert hints['hint_count'] == 1
    assert hints['hints'][0]['target_area'] == 'ghost_decision_contract_policy'
    assert hints['hints'][0]['allowed_use'] == 'soft_hint_only'


def test_accepted_learning_allows_aspiration_and_commitment_target_areas() -> None:
    for target_area in ('aspiration_policy', 'commitment_policy'):
        snapshot = promote_policy_improvement_candidate(
            None,
            {
                'candidate_id': f'policy-improvement-{target_area}',
                'target_area': target_area,
                'summary': f'Keep {target_area} as soft orientation only.',
                'evidence_case_ids': ['eval-1'],
            },
            reviewer='test',
            accepted_at='2026-05-09T10:00:00Z',
        )
        snapshot = set_accepted_learning_policy_enabled(
            snapshot,
            enabled=True,
            reviewer='test',
            reason=f'{target_area} hint area test',
            changed_at='2026-05-09T10:01:00Z',
        )
        hints = build_accepted_learning_runtime_hints(snapshot)

        assert hints['hint_count'] == 1
        assert hints['hints'][0]['target_area'] == target_area
        assert hints['hints'][0]['allowed_use'] == 'soft_hint_only'


def test_ghost_payload_exposes_compact_offline_self_learning_summary(tmp_path: Path) -> None:
    ledger = tmp_path / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])

    payload = build_ghost_payload(
        [],
        recent_events=[],
        runtime_log_path=tmp_path / 'missing.log',
        response_frame_ledger_path=ledger,
        accepted_learning_policy_path=tmp_path / 'accepted_policy_snapshot.json',
    )

    assert payload['self_learning']['status'] == 'completed'
    assert payload['self_learning']['case_count'] >= 4
    assert 'eval_cases' not in payload['self_learning']
    assert payload['accepted_learning_policy']['enabled'] is False
    assert payload['accepted_learning_hints']['hint_count'] == 0
    assert payload['accepted_learning_policy']['runtime_effect'] == 'none'
    assert 'Offline Self Learning' in payload['markdown']


def test_ghost_payload_and_route_context_surface_enabled_hints_as_soft_runtime_context(tmp_path: Path) -> None:
    ledger = tmp_path / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])
    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )
    snapshot = promote_policy_improvement_candidate(
        None,
        report['improvement_candidates'][0],
        reviewer='test',
        accepted_at='2026-05-03T10:00:00Z',
    )
    snapshot = set_accepted_learning_policy_enabled(
        snapshot,
        enabled=True,
        reviewer='test',
        reason='surface bounded live read point',
        changed_at='2026-05-03T10:01:00Z',
    )
    snapshot_path = tmp_path / 'accepted_policy_snapshot.json'
    persist_accepted_learning_policy_snapshot(snapshot, output_path=snapshot_path)

    payload = build_ghost_payload(
        [],
        recent_events=[],
        runtime_log_path=tmp_path / 'missing.log',
        response_frame_ledger_path=ledger,
        accepted_learning_policy_path=snapshot_path,
    )
    context = build_route_context(
        prompt='make two images',
        upload_filename='',
        file_path='',
        conversation_id=None,
        messages=[],
        runtime_manifest={'capabilities': {}, 'instances': []},
        ghost_payload=payload,
        instances=[],
    )

    assert payload['accepted_learning_hints']['hint_count'] == 1
    assert payload['accepted_learning_hints']['runtime_effect'] == 'soft_hints_available'
    assert context['runtime']['accepted_learning_hints']['hint_count'] == 1
    assert context['runtime']['accepted_learning_hints']['hints'][0]['allowed_use'] == 'soft_hint_only'


def test_manage_self_learning_policy_cli_promotes_without_opening_gate(tmp_path: Path) -> None:
    ledger = tmp_path / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])
    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    candidate_id = report['improvement_candidates'][0]['candidate_id']
    snapshot_path = tmp_path / 'accepted_policy_snapshot.json'
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'manage_self_learning_policy.py'

    subprocess.run(
        [sys.executable, str(script_path), '--snapshot', str(snapshot_path), 'init'],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            '--snapshot',
            str(snapshot_path),
            'promote',
            '--report',
            str(report_path),
            '--candidate-id',
            candidate_id,
            '--reviewer',
            'test',
            '--note',
            'cli promote test',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    promoted = load_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)

    assert promoted['enabled'] is False
    assert promoted['runtime_effect'] == 'none'
    assert promoted['accepted_learning_count'] == 1
    assert build_accepted_learning_runtime_hints(promoted)['hint_count'] == 0

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            '--snapshot',
            str(snapshot_path),
            'enable',
            '--reviewer',
            'test',
            '--reason',
            'cli enable test',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    enabled = load_accepted_learning_policy_snapshot(snapshot_path=snapshot_path)

    assert enabled['enabled'] is True
    assert build_accepted_learning_runtime_hints(enabled)['hint_count'] == 1


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    path.write_text(text, encoding='utf-8')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def test_self_learning_retention_collector_finds_nested_sidecar_refs_and_missing_refs(tmp_path: Path) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    learning_dir = tmp_path / 'state' / 'self_learning'
    runtime_sha = _write_json(frames_dir / 'snapshots' / 'runtime.json', {'runtime': {'status': 'blocked'}})
    child_sha = _write_json(frames_dir / 'snapshots' / 'child.json', {'child': True})
    learning_dir.mkdir(parents=True)
    _write_jsonl(
        learning_dir / 'eval_cases.jsonl',
        [
            {
                'case_id': 'eval-retention-sidecar',
                'metadata': {
                    'runtime_snapshot_ref': {
                        'path': 'snapshots/runtime.json',
                        'sha256': runtime_sha,
                    },
                    'sidecar_manifest': {
                        'items': {
                            'child': {
                                'path': 'snapshots/child.json',
                                'sha256': child_sha,
                            },
                            'missing': {
                                'path': 'snapshots/missing.json',
                                'sha256': 'missing-sha',
                            },
                            'unsafe_absolute': {
                                'path': str(tmp_path.parent / 'outside-response-frame.json'),
                                'sha256': 'unsafe-sha',
                            },
                        }
                    },
                },
            }
        ],
    )

    manifest = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=frames_dir,
    )

    retained_paths = {item['path'] for item in manifest['retained_response_frame_sidecars']}
    missing_paths = {item['path'] for item in manifest['missing_response_frame_sidecars']}
    unsafe_paths = {item['path'] for item in manifest['external_or_unsafe_refs']}
    assert manifest['kind'] == 'ollmo.self_learning_retention_manifest'
    assert manifest['status'] == 'partial'
    assert retained_paths == {'snapshots/runtime.json', 'snapshots/child.json'}
    assert missing_paths == {'snapshots/missing.json'}
    assert str(tmp_path.parent / 'outside-response-frame.json') in unsafe_paths
    assert manifest['retention_root_count'] == 2


def test_self_learning_report_includes_retention_integrity_diagnostics(tmp_path: Path) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    learning_dir = tmp_path / 'state' / 'self_learning'
    frames_dir.mkdir(parents=True)
    learning_dir.mkdir(parents=True)
    ledger = frames_dir / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])
    _write_jsonl(
        learning_dir / 'eval_cases.jsonl',
        [
            {
                'case_id': 'eval-retention-report',
                'metadata': {
                    'runtime_snapshot_ref': {
                        'path': 'snapshots/missing-runtime.json',
                        'sha256': 'missing-runtime-sha',
                    }
                },
            }
        ],
    )

    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        self_learning_dir=learning_dir,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )

    assert report['retention']['status'] == 'partial'
    assert report['retention']['retained_sidecar_count'] == 0
    assert report['retention']['missing_sidecar_count'] == 1
    assert report['retention']['manifest_path'].endswith('state/self_learning/retention_manifest.json')
    assert report['retention']['missing_response_frame_sidecars'][0]['path'] == 'snapshots/missing-runtime.json'


def test_self_learning_report_does_not_claim_complete_retention_without_manifest_or_copies(tmp_path: Path) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    learning_dir = tmp_path / 'state' / 'self_learning'
    frames_dir.mkdir(parents=True)
    learning_dir.mkdir(parents=True)
    ledger = frames_dir / 'responses.jsonl'
    _write_jsonl(ledger, [_problem_frame()])
    runtime_sha = _write_json(frames_dir / 'snapshots' / 'runtime.json', {'runtime': {'status': 'blocked'}})
    _write_jsonl(
        learning_dir / 'eval_cases.jsonl',
        [
            {
                'case_id': 'eval-retention-report-complete-source-only',
                'metadata': {
                    'runtime_snapshot_ref': {
                        'path': 'snapshots/runtime.json',
                        'sha256': runtime_sha,
                    }
                },
            }
        ],
    )

    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        self_learning_dir=learning_dir,
        frame_limit=20,
        max_cases=20,
        include_cases=False,
    )

    assert report['retention']['source_status'] == 'complete'
    assert report['retention']['status'] == 'partial'
    assert report['retention']['storage_status'] == 'missing_manifest'
    assert report['retention']['manifest_exists'] is False
    assert report['retention']['retained_sidecar_count'] == 1
    assert report['retention']['retained_copy_missing_count'] == 1
    assert report['retention']['missing_retained_copies'][0]['path'] == 'snapshots/runtime.json'


def test_retention_summary_reports_complete_after_manifest_and_retained_copy_exist(tmp_path: Path) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    learning_dir = tmp_path / 'state' / 'self_learning'
    frames_dir.mkdir(parents=True)
    learning_dir.mkdir(parents=True)
    runtime_sha = _write_json(frames_dir / 'snapshots' / 'runtime.json', {'runtime': {'status': 'blocked'}})
    _write_jsonl(
        learning_dir / 'eval_cases.jsonl',
        [
            {
                'case_id': 'eval-retention-report-complete-storage',
                'metadata': {
                    'runtime_snapshot_ref': {
                        'path': 'snapshots/runtime.json',
                        'sha256': runtime_sha,
                    }
                },
            }
        ],
    )
    manifest_path = learning_dir / 'retention_manifest.json'
    manifest = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=frames_dir,
    )
    manifest = copy_retained_sidecars(
        manifest,
        response_frames_dir=frames_dir,
        retained_sidecars_dir=learning_dir / 'retained_sidecars',
    )
    write_retention_manifest(manifest, manifest_path)

    summary = retention_summary(manifest, manifest_path=manifest_path)

    assert summary['source_status'] == 'complete'
    assert summary['status'] == 'complete'
    assert summary['storage_status'] == 'complete'
    assert summary['manifest_exists'] is True
    assert summary['retained_copy_missing_count'] == 0


def test_retention_collector_uses_verified_retained_copy_after_source_epoch_moves(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    learning_dir = tmp_path / 'state' / 'self_learning'
    retained_dir = learning_dir / 'retained_sidecars'
    snapshot_path = frames_dir / 'snapshots' / 'runtime.json'
    runtime_sha = _write_json(snapshot_path, {'runtime': {'status': 'blocked'}})
    _write_jsonl(
        learning_dir / 'eval_cases.jsonl',
        [
            {
                'case_id': 'eval-retention-post-epoch',
                'metadata': {
                    'runtime_snapshot_ref': {
                        'path': 'snapshots/runtime.json',
                        'sha256': runtime_sha,
                    }
                },
            }
        ],
    )
    manifest = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=frames_dir,
        retained_sidecars_dir=retained_dir,
    )
    manifest = copy_retained_sidecars(
        manifest,
        response_frames_dir=frames_dir,
        retained_sidecars_dir=retained_dir,
    )
    write_retention_manifest(manifest, learning_dir / 'retention_manifest.json')
    snapshot_path.unlink()

    after_epoch_move = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=frames_dir,
        retained_sidecars_dir=retained_dir,
    )

    assert after_epoch_move['status'] == 'complete'
    assert after_epoch_move['missing_sidecar_count'] == 0
    assert after_epoch_move['retained_sidecar_count'] == 1
    assert after_epoch_move['source_sidecar_count'] == 0
    assert after_epoch_move['retained_copy_sidecar_count'] == 1
    retained_record = after_epoch_move['retained_response_frame_sidecars'][0]
    assert retained_record['storage_source'] == 'retained_copy'
    assert retained_record['actual_sha256'] == runtime_sha

    (retained_dir / 'snapshots' / 'runtime.json').write_text('{"corrupt":true}', encoding='utf-8')
    after_corruption = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=frames_dir,
        retained_sidecars_dir=retained_dir,
    )

    assert after_corruption['status'] == 'partial'
    assert after_corruption['missing_sidecar_count'] == 1
    assert after_corruption['retained_copy_sha256_mismatch_count'] == 1


def test_retention_hash_matches_cas_payload_with_one_terminal_newline(tmp_path: Path) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    learning_dir = tmp_path / 'state' / 'self_learning'
    snapshot_path = frames_dir / 'snapshots' / 'runtime.json'
    snapshot_path.parent.mkdir(parents=True)
    payload = json.dumps({'runtime': {'status': 'blocked'}}, ensure_ascii=False, sort_keys=True).encode('utf-8')
    expected_sha = hashlib.sha256(payload).hexdigest()
    snapshot_path.write_bytes(payload + b'\n')
    _write_jsonl(
        learning_dir / 'eval_cases.jsonl',
        [
            {
                'case_id': 'eval-retention-terminal-newline',
                'metadata': {
                    'runtime_snapshot_ref': {
                        'path': 'snapshots/runtime.json',
                        'sha256': expected_sha,
                    }
                },
            }
        ],
    )
    manifest_path = learning_dir / 'retention_manifest.json'

    manifest = collect_self_learning_retention_roots(
        self_learning_dir=learning_dir,
        response_frames_dir=frames_dir,
    )

    assert manifest['status'] == 'complete'
    assert manifest['retained_sidecar_count'] == 1
    assert manifest['retained_response_frame_sidecars'][0]['actual_sha256'] == expected_sha
    assert 'sha256_mismatch' not in manifest['retained_response_frame_sidecars'][0]

    manifest = copy_retained_sidecars(
        manifest,
        response_frames_dir=frames_dir,
        retained_sidecars_dir=learning_dir / 'retained_sidecars',
    )
    write_retention_manifest(manifest, manifest_path)
    summary = retention_summary(manifest, manifest_path=manifest_path)

    assert summary['status'] == 'complete'
    assert summary['retained_copy_sha256_mismatch_count'] == 0


def _terminal_successor_reopen_frame(status: str = 'candidate') -> dict:
    outcome_status = 'applied' if status in {'applied_to_successor', 'solved'} else 'blocked'
    closure_status = 'fulfilled' if status == 'solved' else 'repair_needed'
    return {
        'response_id': f'resp-terminal-successor-{status}',
        'status': 'completed',
        'request': {'prompt': 'make the missing image too'},
        'runtime': {
            'graph_closure_review': {'status': closure_status},
            'request_phase_graph': {
                'successor_reopen_requests': [
                    {
                        'kind': 'ollmo.graph_patch_successor_reopen_request',
                        'status': status,
                        'parent_response_id': 'resp-terminal-parent',
                        'parent_frame_id': 'resp-terminal-parent:frame-1',
                        'parent_frame_sequence': 1,
                        'proposal_id': 'graph-repair-terminal-successor',
                        'patch_id': 'patch-terminal-successor',
                        'repair_class': 'missing_materialization_branch',
                        'autonomy_level': 'apply_safe',
                        'runtime_effect': 'successor_reopen_required',
                        'blocked_reasons': [] if status != 'blocked' else ['late_fill_provider_unavailable'],
                    }
                ],
                'graph_patch_lifecycle': [
                    {
                        'patch_id': 'patch-terminal-successor',
                        'proposal_id': 'graph-repair-terminal-successor',
                        'repair_class': 'missing_materialization_branch',
                        'status': 'applied' if status in {'applied_to_successor', 'solved'} else 'blocked',
                        'outcome': {'status': outcome_status},
                    }
                ],
            },
        },
    }


def test_self_learning_extracts_terminal_successor_reopen_cases() -> None:
    created_cases = build_eval_cases_from_response_frame(_terminal_successor_reopen_frame('candidate'))
    solved_cases = build_eval_cases_from_response_frame(_terminal_successor_reopen_frame('solved'))
    blocked_cases = build_eval_cases_from_response_frame(_terminal_successor_reopen_frame('blocked'))

    assert 'graph_patch_terminal_successor_reopen_created' in {
        case['case_kind'] for case in created_cases
    }
    assert 'graph_patch_terminal_successor_reopen_solved' in {
        case['case_kind'] for case in solved_cases
    }
    assert 'graph_patch_terminal_successor_reopen_blocked' in {
        case['case_kind'] for case in blocked_cases
    }


def _corpus_settled_case(
    case_id: str,
    response_id: str,
    frame_id: str,
    frame_sequence: int,
    *,
    state: str = 'settled_terminal',
    declared_frame_id: str | None = None,
    declared_frame_sequence: int | None = None,
    graph_records: dict | None = None,
    diagnostic_records: dict | None = None,
    redraw_scope: dict | None = None,
) -> dict:
    return {
        'case_id': case_id,
        'state': state,
        'response_id': response_id,
        'last_frame_id': declared_frame_id or frame_id,
        'last_frame_sequence': declared_frame_sequence if declared_frame_sequence is not None else frame_sequence,
        'settled_outcome': 'success' if state == 'settled_terminal' else 'repair_needed',
        'category': 'test_graph_rebase_corpus',
        'workload_family': 'test_graph_rebase',
        'final_debug': {
            'status': 'captured',
            'summary': {
                'id': response_id,
                'response_frame': {
                    'frame_id': frame_id,
                    'frame_sequence': frame_sequence,
                },
                'graph_records': graph_records or {},
                'diagnostic_records': diagnostic_records or {},
                'redraw_scope': redraw_scope or {},
            },
        },
    }


def _write_graph_rebase_corpus_manifest(path: Path, cases: list[object], *, schema_version: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'kind': 'ollmo.graph_rebase_shadow_corpus_manifest',
                'schema_version': schema_version,
                'corpus_id': 'test-corpus',
                'corpus_digest': 'test-corpus-digest',
                'cases': cases,
            },
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )


def test_graph_rebase_corpus_includes_outside_window_and_overlays_eval_only_graph_evidence(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    ledger = frames_dir / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    corpus_dir = tmp_path / 'state' / 'graph_rebase_shadow_corpus'
    accepted_path = learning_dir / 'accepted_policy_snapshot.json'
    learning_dir.mkdir(parents=True)
    accepted_bytes = b'{"enabled":true,"accepted_learnings":[{"learning_id":"keep"}]}\n'
    accepted_path.write_bytes(accepted_bytes)

    corpus_frame = _positive_frame()
    corpus_frame.update(
        {
            'response_id': 'resp-corpus-outside-window',
            'frame_id': 'resp-corpus-outside-window:frame-1',
            'frame_sequence': 1,
        }
    )
    recent_frame = _positive_frame()
    recent_frame.update(
        {
            'response_id': 'resp-recent-only',
            'frame_id': 'resp-recent-only:frame-1',
            'frame_sequence': 1,
        }
    )
    _write_jsonl(ledger, [corpus_frame, recent_frame])

    patch_record = {
        'kind': 'ollmo.graph_patch_lifecycle',
        'patch_id': 'patch-from-corpus-debug',
        'proposal_id': 'proposal-from-corpus-debug',
        'repair_class': 'missing_dependency_edge',
        'status': 'applied',
        'outcome': {'status': 'applied'},
    }
    redraw_scope = {
        'kind': 'ollmo.redraw_scope_ladder_review',
        'review_id': 'redraw-from-corpus-debug',
        'status': 'selected',
        'selected_scope': 'repair_binding_dependency',
        'selected_candidate': {'scope': 'repair_binding_dependency', 'eligible': True},
    }
    _write_graph_rebase_corpus_manifest(
        corpus_dir / 'valid.json',
        [
            _corpus_settled_case(
                'outside-window',
                'resp-corpus-outside-window',
                'resp-corpus-outside-window:frame-1',
                1,
                graph_records={'graph_patch_lifecycle': [patch_record]},
                diagnostic_records={'graph_patch_lifecycle': [patch_record]},
                redraw_scope=redraw_scope,
            )
        ],
    )

    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        graph_rebase_corpus_dir=corpus_dir,
        self_learning_dir=learning_dir,
        frame_limit=1,
        max_cases=100,
    )

    corpus_cases = [
        case for case in report['eval_cases'] if case['response_id'] == 'resp-corpus-outside-window'
    ]
    case_kinds = {case['case_kind'] for case in corpus_cases}
    assert report['frame_count'] == 1
    assert report['evaluated_response_count'] == 2
    assert report['graph_rebase_corpus']['exact_bound_case_count'] == 1
    assert report['graph_rebase_corpus']['corpus_linked_response_count'] == 1
    assert 'graph_patch_applied_and_closed' in case_kinds
    assert 'redraw_scope_selected' in case_kinds
    assert all(case['kind'] == 'ollmo.self_learning_eval_case' for case in corpus_cases)
    assert all(case['optimization_policy'] == 'proposal_only_reviewed_patch_required' for case in corpus_cases)
    assert all('graph_rebase_corpus' in case['metadata'] for case in corpus_cases)
    assert accepted_path.read_bytes() == accepted_bytes
    assert 'request_phase_graph' not in corpus_frame['runtime']


def test_graph_rebase_corpus_final_debug_binding_wins_and_recent_overlap_is_deduped(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    corpus_dir = tmp_path / 'state' / 'graph_rebase_shadow_corpus'
    response_id = 'resp-corpus-d4r2-shape'
    frame_one = _problem_frame()
    frame_one.update(
        {
            'response_id': response_id,
            'frame_id': f'{response_id}:frame-1',
            'frame_sequence': 1,
        }
    )
    frame_two = _positive_frame()
    frame_two.update(
        {
            'response_id': response_id,
            'frame_id': f'{response_id}:frame-2',
            'frame_sequence': 2,
        }
    )
    _write_jsonl(ledger, [frame_one, frame_two])
    _write_graph_rebase_corpus_manifest(
        corpus_dir / 'd4r2.json',
        [
            _corpus_settled_case(
                'D4R2',
                response_id,
                f'{response_id}:frame-2',
                2,
                declared_frame_id=f'{response_id}:frame-1',
                declared_frame_sequence=1,
            )
        ],
    )

    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        graph_rebase_corpus_dir=corpus_dir,
        self_learning_dir=tmp_path / 'state' / 'self_learning',
        frame_limit=20,
        max_cases=100,
    )

    coverage = report['graph_rebase_corpus']
    response_cases = [case for case in report['eval_cases'] if case['response_id'] == response_id]
    dedupe_keys = {
        (case['response_id'], case['case_kind'], case['evidence'])
        for case in response_cases
    }
    assert coverage['exact_bound_case_count'] == 1
    assert coverage['stale_frame_case_count'] == 0
    assert coverage['declared_last_frame_mismatch_count'] == 1
    assert report['evaluated_response_count'] == 1
    assert len(dedupe_keys) == len(response_cases)
    assert {case['case_kind'] for case in response_cases} == {
        'fulfilled_graph_contract',
        'history_scan_promoted_matches',
    }


def test_self_learning_report_combines_ledger_epochs_and_hydrates_each_cas_root(
    tmp_path: Path,
) -> None:
    active_frames_dir = tmp_path / 'active' / 'state' / 'response_frames'
    archive_frames_dir = tmp_path / 'archive' / 'state' / 'response_frames'
    active_ledger = active_frames_dir / 'responses.jsonl'
    archive_ledger = archive_frames_dir / 'responses.jsonl'
    corpus_dir = tmp_path / 'active' / 'state' / 'graph_rebase_shadow_corpus'
    learning_dir = tmp_path / 'active' / 'state' / 'self_learning'

    active_runtime = _positive_frame()['runtime']
    archive_runtime = _problem_frame()['runtime']
    _write_json(active_frames_dir / 'snapshots' / 'active-runtime.json', active_runtime)
    _write_json(archive_frames_dir / 'snapshots' / 'archive-runtime.json', archive_runtime)
    active_frame = {
        'response_id': 'resp-active-epoch',
        'frame_id': 'resp-active-epoch:frame-1',
        'frame_sequence': 1,
        'runtime_snapshot_ref': {'path': 'snapshots/active-runtime.json'},
    }
    archive_frame = {
        'response_id': 'resp-archive-epoch',
        'frame_id': 'resp-archive-epoch:frame-1',
        'frame_sequence': 1,
        'runtime_snapshot_ref': {'path': 'snapshots/archive-runtime.json'},
    }
    _write_jsonl(active_ledger, [active_frame])
    _write_jsonl(archive_ledger, [archive_frame])
    _write_graph_rebase_corpus_manifest(
        corpus_dir / 'archive.json',
        [
            _corpus_settled_case(
                'archive-epoch',
                'resp-archive-epoch',
                'resp-archive-epoch:frame-1',
                1,
                state='settled_repair_needed',
            )
        ],
    )

    report = build_self_learning_report(
        response_frame_ledger_path=active_ledger,
        additional_response_frame_ledger_paths=[archive_ledger],
        graph_rebase_corpus_dir=corpus_dir,
        self_learning_dir=learning_dir,
        frame_limit=20,
        max_cases=100,
    )

    response_ids = {case['response_id'] for case in report['eval_cases']}
    archive_case_kinds = {
        case['case_kind']
        for case in report['eval_cases']
        if case['response_id'] == 'resp-archive-epoch'
    }
    assert report['frame_count'] == 2
    assert report['evaluated_response_count'] == 2
    assert report['source']['response_frame_ledgers'] == [str(active_ledger), str(archive_ledger)]
    assert response_ids == {'resp-active-epoch', 'resp-archive-epoch'}
    assert 'open_graph_obligation' in archive_case_kinds
    assert report['graph_rebase_corpus']['exact_bound_case_count'] == 1
    assert report['graph_rebase_corpus']['missing_response_case_count'] == 0


def test_graph_rebase_corpus_reports_malformed_planned_dependency_missing_and_stale(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    corpus_dir = tmp_path / 'state' / 'graph_rebase_shadow_corpus'
    stale_response_id = 'resp-corpus-stale'
    stale_latest = _positive_frame()
    stale_latest.update(
        {
            'response_id': stale_response_id,
            'frame_id': f'{stale_response_id}:frame-2',
            'frame_sequence': 2,
        }
    )
    _write_jsonl(ledger, [stale_latest])
    uncaptured = _corpus_settled_case(
        'uncaptured-final-debug',
        'resp-corpus-uncaptured',
        'resp-corpus-uncaptured:frame-1',
        1,
    )
    uncaptured['final_debug']['status'] = 'failed'
    mismatched_response = _corpus_settled_case(
        'mismatched-response',
        'resp-corpus-declared',
        'resp-corpus-declared:frame-1',
        1,
    )
    mismatched_response['final_debug']['summary']['id'] = 'resp-corpus-summary-other'
    cases: list[object] = [
        _corpus_settled_case(
            'missing-response',
            'resp-corpus-missing',
            'resp-corpus-missing:frame-1',
            1,
        ),
        _corpus_settled_case(
            'stale-frame',
            stale_response_id,
            f'{stale_response_id}:frame-1',
            1,
        ),
        {'case_id': 'planned', 'state': 'planned'},
        {'case_id': 'dependency', 'state': 'dependency_blocked'},
        {'case_id': 'missing-state'},
        'not-a-case-object',
        uncaptured,
        mismatched_response,
    ]
    _write_graph_rebase_corpus_manifest(corpus_dir / 'coverage.json', cases)
    _write_graph_rebase_corpus_manifest(corpus_dir / 'unsupported.json', [], schema_version=2)
    (corpus_dir / 'malformed.json').write_text('{broken', encoding='utf-8')

    report = build_self_learning_report(
        response_frame_ledger_path=ledger,
        graph_rebase_corpus_dir=corpus_dir,
        self_learning_dir=tmp_path / 'state' / 'self_learning',
        frame_limit=1,
        max_cases=100,
    )

    coverage = report['graph_rebase_corpus']
    assert coverage['status'] == 'partial'
    assert coverage['manifest_file_count'] == 3
    assert coverage['manifest_count'] == 1
    assert coverage['malformed_manifest_count'] == 1
    assert coverage['unsupported_manifest_count'] == 1
    assert coverage['malformed_case_count'] == 4
    assert coverage['malformed_binding_case_count'] == 2
    assert coverage['missing_response_case_count'] == 1
    assert coverage['stale_frame_case_count'] == 1
    assert coverage['planned_case_count'] == 1
    assert coverage['dependency_blocked_case_count'] == 1
    assert coverage['declared_response_id_mismatch_count'] == 1
    assert coverage['exact_bound_case_count'] == 0
    assert all(case['response_id'] != 'resp-corpus-missing' for case in report['eval_cases'])


def test_corpus_overlay_preserves_canonical_redraw_and_distinct_review_identities() -> None:
    canonical_review = {
        'review_id': 'canonical-redraw',
        'selected_scope': 'observe',
    }
    frame = {
        'response_id': 'resp-overlay-canonical',
        'runtime': {
            'request_phase_graph': {
                'redraw_scope_ladder_review': canonical_review,
            }
        },
    }
    overlaid = self_learning_module._overlay_corpus_eval_evidence(
        frame,
        [
            {
                '_graph_records': {
                    'graph_repair_reviews': [
                        {'review_id': 'review-a', 'proposal_id': 'shared-proposal'},
                        {'review_id': 'review-b', 'proposal_id': 'shared-proposal'},
                    ]
                },
                '_redraw_scope': {
                    'review_id': 'corpus-redraw',
                    'selected_scope': 'repair_binding_dependency',
                },
            }
        ],
    )
    graph = overlaid['runtime']['request_phase_graph']

    assert graph['redraw_scope_ladder_review'] == canonical_review
    assert {item['review_id'] for item in graph['redraw_scope_ladder_reviews']} == {
        'canonical-redraw',
        'corpus-redraw',
    }
    assert {item['review_id'] for item in graph['graph_repair_reviews']} == {
        'review-a',
        'review-b',
    }


def test_self_learning_cli_accepts_explicit_graph_rebase_corpus_path(tmp_path: Path) -> None:
    ledger = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    corpus_dir = tmp_path / 'state' / 'graph_rebase_shadow_corpus'
    frame = _positive_frame()
    frame.update(
        {
            'response_id': 'resp-cli-corpus',
            'frame_id': 'resp-cli-corpus:frame-1',
            'frame_sequence': 1,
        }
    )
    _write_jsonl(ledger, [frame])
    _write_graph_rebase_corpus_manifest(
        corpus_dir / 'cli.json',
        [_corpus_settled_case('cli', 'resp-cli-corpus', 'resp-cli-corpus:frame-1', 1)],
    )

    result = subprocess.run(
        [
            sys.executable,
            'scripts/build_self_learning_eval_cases.py',
            '--frames',
            str(ledger),
            '--graph-rebase-corpus-dir',
            str(corpus_dir),
            '--self-learning-dir',
            str(tmp_path / 'state' / 'self_learning'),
            '--frame-limit',
            '1',
            '--max-cases',
            '100',
            '--no-persist',
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary['graph_rebase_corpus']['exact_bound_case_count'] == 1
    assert summary['graph_rebase_corpus']['selected_eval_case_count'] == 2
    assert summary['eval_case_output'] is None
    assert summary['report_output'] is None


def test_self_learning_cli_accepts_repeated_frame_ledgers(tmp_path: Path) -> None:
    active_ledger = tmp_path / 'active' / 'state' / 'response_frames' / 'responses.jsonl'
    archive_ledger = tmp_path / 'archive' / 'state' / 'response_frames' / 'responses.jsonl'
    active_frame = _positive_frame()
    active_frame['response_id'] = 'resp-cli-active-epoch'
    archive_frame = _problem_frame()
    archive_frame['response_id'] = 'resp-cli-archive-epoch'
    _write_jsonl(active_ledger, [active_frame])
    _write_jsonl(archive_ledger, [archive_frame])

    result = subprocess.run(
        [
            sys.executable,
            'scripts/build_self_learning_eval_cases.py',
            '--frames',
            str(active_ledger),
            '--frames',
            str(archive_ledger),
            '--graph-rebase-corpus-dir',
            str(tmp_path / 'state' / 'missing-corpus'),
            '--self-learning-dir',
            str(tmp_path / 'state' / 'self_learning'),
            '--frame-limit',
            '20',
            '--max-cases',
            '100',
            '--no-persist',
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary['frame_count'] == 2
    assert summary['evaluated_response_count'] == 2


def _historical_eval_case(
    case_id: str,
    *,
    response_id: str | None = None,
    case_kind: str = 'graph_rebase_proposal_rejected',
    severity: str = 'medium',
) -> dict:
    resolved_response_id = response_id or f'resp-{case_id}'
    return {
        'kind': 'ollmo.self_learning_eval_case',
        'case_version': 1,
        'case_id': case_id,
        'response_id': resolved_response_id,
        'frame_status': 'completed',
        'layer': 'graph_rebase',
        'case_kind': case_kind,
        'severity': severity,
        'summary': f'Historical eval evidence for {case_id}.',
        'evidence': f'historical.eval_case:{case_id}',
        'target_area': 'graph_rebase_policy',
        'target_surfaces': ['ollmo_services/graph_rebase.py'],
        'suggested_action': 'Keep as reviewed offline evidence only.',
        'metadata': {'historical_fixture': True},
        'optimization_policy': 'proposal_only_reviewed_patch_required',
    }


def test_merge_existing_preserves_old_only_cases_beyond_new_case_limit(tmp_path: Path) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    existing_path = learning_dir / 'eval_cases.jsonl'
    _write_jsonl(frames_path, [_problem_frame()])
    historical_cases = [
        _historical_eval_case('eval-historical-z'),
        _historical_eval_case('eval-historical-a'),
        _historical_eval_case('eval-historical-m'),
    ]
    historical_cases[0]['empty_fields_must_survive'] = {
        'empty_text': '',
        'empty_list': [],
        'empty_mapping': {},
        'null_value': None,
    }
    _write_jsonl(existing_path, historical_cases)

    report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        existing_eval_case_ledger_path=existing_path,
        merge_existing=True,
        frame_limit=20,
        max_cases=1,
    )

    merged_ids = {case['case_id'] for case in report['eval_cases']}
    historical_ids = {case['case_id'] for case in historical_cases}
    assert historical_ids <= merged_ids
    assert report['previous_case_count'] == 3
    assert report['new_case_count'] == 1
    assert report['preserved_case_count'] == 3
    assert report['replaced_case_count'] == 0
    assert report['removed_case_count'] == 0
    assert report['case_count'] == 4
    assert next(
        case for case in report['eval_cases'] if case['case_id'] == 'eval-historical-z'
    ) == historical_cases[0]
    assert report['merge_policy']['mode'] == 'merge_existing'
    assert report['merge_policy']['max_cases_scope'] == 'newly_generated_cases_only'


def test_merge_existing_prefers_fresh_case_for_duplicate_case_id(tmp_path: Path) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    existing_path = learning_dir / 'eval_cases.jsonl'
    _write_jsonl(frames_path, [_problem_frame()])
    fresh_report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        frame_limit=20,
        max_cases=20,
    )
    fresh_case = fresh_report['eval_cases'][0]
    stale_case = {
        **fresh_case,
        'summary': 'Stale historical content that must be replaced.',
        'historical_only_marker': True,
    }
    _write_jsonl(existing_path, [stale_case])

    merged_report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        existing_eval_case_ledger_path=existing_path,
        merge_existing=True,
        frame_limit=20,
        max_cases=20,
    )

    matching_cases = [
        case
        for case in merged_report['eval_cases']
        if case['case_id'] == fresh_case['case_id']
    ]
    assert matching_cases == [fresh_case]
    assert 'historical_only_marker' not in matching_cases[0]
    assert merged_report['previous_case_count'] == 1
    assert merged_report['new_case_count'] == fresh_report['case_count']
    assert merged_report['preserved_case_count'] == 0
    assert merged_report['replaced_case_count'] == 1
    assert merged_report['removed_case_count'] == 0


def test_merge_existing_preserves_historical_graph_rebase_case_without_current_truth(
    tmp_path: Path,
) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    existing_path = learning_dir / 'eval_cases.jsonl'
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path.write_text('', encoding='utf-8')
    historical_case = _historical_eval_case(
        'eval-historical-graph-rebase-successor',
        response_id='resp-synthetic-frame-no-longer-present',
        case_kind='graph_rebase_partial_successor_execution_solved',
    )
    _write_jsonl(existing_path, [historical_case])

    report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        existing_eval_case_ledger_path=existing_path,
        merge_existing=True,
        frame_limit=1,
        max_cases=1,
    )

    assert report['status'] == 'no_frames'
    assert report['frame_count'] == 0
    assert report['recent_evaluated_response_count'] == 0
    assert report['evaluated_response_count'] == 0
    assert report['new_case_count'] == 0
    assert report['previous_case_count'] == 1
    assert report['preserved_case_count'] == 1
    assert report['case_count'] == 1
    assert report['eval_cases'] == [historical_case]
    assert report['graph_rebase_corpus']['status'] == 'not_configured'
    assert report['counts_by_kind'] == {
        'graph_rebase_partial_successor_execution_solved': 1,
    }
    assert report['improvement_candidate_count'] == 1
    assert report['improvement_candidates'][0]['evidence_case_ids'] == [historical_case['case_id']]
    assert report['shadow_hint_count'] == 1
    assert 'historical' in report['merge_policy']['preserved_case_truth_role']


def test_merge_existing_orders_union_canonically_by_case_id(tmp_path: Path) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    existing_path = learning_dir / 'eval_cases.jsonl'
    _write_jsonl(frames_path, [_positive_frame()])
    _write_jsonl(
        existing_path,
        [
            _historical_eval_case('eval-z-last'),
            _historical_eval_case('eval-a-first'),
            _historical_eval_case('eval-m-middle'),
        ],
    )

    first_report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        existing_eval_case_ledger_path=existing_path,
        merge_existing=True,
        frame_limit=20,
        max_cases=20,
    )
    second_report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        existing_eval_case_ledger_path=existing_path,
        merge_existing=True,
        frame_limit=20,
        max_cases=20,
    )

    first_ids = [case['case_id'] for case in first_report['eval_cases']]
    second_ids = [case['case_id'] for case in second_report['eval_cases']]
    assert first_ids == sorted(first_ids)
    assert second_ids == first_ids
    assert second_report['eval_cases'] == first_report['eval_cases']


def test_self_learning_report_keeps_default_replacement_order_and_content(tmp_path: Path) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    existing_path = learning_dir / 'eval_cases.jsonl'
    _write_jsonl(frames_path, [_problem_frame()])
    expected_report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        frame_limit=20,
        max_cases=2,
    )
    historical_case = _historical_eval_case('eval-must-not-survive-default-replacement')
    _write_jsonl(existing_path, [historical_case])

    replacement_report = build_self_learning_report(
        response_frame_ledger_path=frames_path,
        self_learning_dir=learning_dir,
        existing_eval_case_ledger_path=existing_path,
        frame_limit=20,
        max_cases=2,
    )

    assert replacement_report['eval_cases'] == expected_report['eval_cases']
    assert historical_case['case_id'] not in {
        case['case_id'] for case in replacement_report['eval_cases']
    }
    assert replacement_report['merge_policy']['mode'] == 'replace_existing'


@pytest.mark.parametrize(
    'existing_payload',
    [
        '{not-json\n',
        json.dumps(['not', 'an', 'eval', 'case']) + '\n',
        json.dumps({'kind': 'ollmo.self_learning_eval_case', 'summary': 'missing case id'}) + '\n',
        (
            json.dumps(_historical_eval_case('eval-duplicate'))
            + '\n'
            + json.dumps(_historical_eval_case('eval-duplicate'))
            + '\n'
        ),
    ],
    ids=['malformed-json', 'non-object', 'missing-case-id', 'duplicate-case-id'],
)
def test_merge_existing_rejects_invalid_old_ledger_without_rewriting_it(
    tmp_path: Path,
    existing_payload: str,
) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    existing_path = learning_dir / 'eval_cases.jsonl'
    _write_jsonl(frames_path, [_positive_frame()])
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text(existing_payload, encoding='utf-8')
    original_bytes = existing_path.read_bytes()

    with pytest.raises(ValueError):
        build_self_learning_report(
            response_frame_ledger_path=frames_path,
            self_learning_dir=learning_dir,
            existing_eval_case_ledger_path=existing_path,
            merge_existing=True,
            frame_limit=20,
            max_cases=20,
        )

    assert existing_path.read_bytes() == original_bytes


def test_self_learning_cli_merge_preview_reports_counts_without_writing(tmp_path: Path) -> None:
    frames_path = tmp_path / 'state' / 'response_frames' / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    eval_path = learning_dir / 'eval_cases.jsonl'
    report_path = learning_dir / 'report.json'
    accepted_path = learning_dir / 'accepted_policy_snapshot.json'
    _write_jsonl(frames_path, [_positive_frame()])
    _write_jsonl(eval_path, [_historical_eval_case('eval-preview-historical')])
    report_path.write_bytes(b'{"sentinel":"old-report"}\n')
    accepted_path.write_bytes(b'{"enabled":true,"sentinel":"accepted-policy"}\n')
    original_eval = eval_path.read_bytes()
    original_report = report_path.read_bytes()
    original_accepted = accepted_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            'scripts/build_self_learning_eval_cases.py',
            '--frames',
            str(frames_path),
            '--output',
            str(eval_path),
            '--report',
            str(report_path),
            '--accepted-policy',
            str(accepted_path),
            '--graph-rebase-corpus-dir',
            str(tmp_path / 'state' / 'missing-corpus'),
            '--self-learning-dir',
            str(learning_dir),
            '--frame-limit',
            '20',
            '--max-cases',
            '1',
            '--merge-existing',
            '--no-persist',
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary['merge_preview'] is True
    assert summary['previous_case_count'] == 1
    assert summary['new_case_count'] == 1
    assert summary['preserved_case_count'] == 1
    assert summary['replaced_case_count'] == 0
    assert summary['removed_case_count'] == 0
    assert summary['case_count'] == 2
    assert summary['merge_policy']['mode'] == 'merge_existing'
    assert summary['merge_policy']['max_cases_scope'] == 'newly_generated_cases_only'
    assert summary['eval_case_output'] is None
    assert summary['report_output'] is None
    assert eval_path.read_bytes() == original_eval
    assert report_path.read_bytes() == original_report
    assert accepted_path.read_bytes() == original_accepted


def test_self_learning_cli_merge_persists_union_without_mutating_protected_state(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    frames_path = frames_dir / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    eval_path = learning_dir / 'eval_cases.jsonl'
    report_path = learning_dir / 'report.json'
    accepted_path = learning_dir / 'accepted_policy_snapshot.json'
    retained_path = learning_dir / 'retained_sidecars' / 'keep.json'
    cas_path = frames_dir / 'snapshots' / 'content' / 'keep.json'
    _write_jsonl(frames_path, [_positive_frame()])
    _write_jsonl(eval_path, [_historical_eval_case('eval-persist-historical')])
    accepted_path.write_bytes(b'{"enabled":true,"sentinel":"accepted-policy"}\n')
    retained_path.parent.mkdir(parents=True, exist_ok=True)
    retained_path.write_bytes(b'{"sentinel":"retained-sidecar"}\n')
    cas_path.parent.mkdir(parents=True, exist_ok=True)
    cas_path.write_bytes(b'{"sentinel":"response-cas"}\n')
    original_frames = frames_path.read_bytes()
    original_accepted = accepted_path.read_bytes()
    original_retained = retained_path.read_bytes()
    original_cas = cas_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            'scripts/build_self_learning_eval_cases.py',
            '--frames',
            str(frames_path),
            '--output',
            str(eval_path),
            '--report',
            str(report_path),
            '--accepted-policy',
            str(accepted_path),
            '--graph-rebase-corpus-dir',
            str(tmp_path / 'state' / 'missing-corpus'),
            '--self-learning-dir',
            str(learning_dir),
            '--frame-limit',
            '20',
            '--max-cases',
            '1',
            '--merge-existing',
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    rows = [json.loads(line) for line in eval_path.read_text(encoding='utf-8').splitlines()]
    persisted_report = json.loads(report_path.read_text(encoding='utf-8'))

    assert summary['merge_preview'] is False
    assert summary['previous_case_count'] == 1
    assert summary['new_case_count'] == 1
    assert summary['preserved_case_count'] == 1
    assert summary['replaced_case_count'] == 0
    assert summary['removed_case_count'] == 0
    assert [case['case_id'] for case in rows] == sorted(case['case_id'] for case in rows)
    assert persisted_report['eval_cases'] == rows
    assert persisted_report['case_count'] == len(rows) == 2
    assert persisted_report['merge_policy']['mode'] == 'merge_existing'
    assert frames_path.read_bytes() == original_frames
    assert accepted_path.read_bytes() == original_accepted
    assert retained_path.read_bytes() == original_retained
    assert cas_path.read_bytes() == original_cas


def test_persist_self_learning_outputs_preserves_complete_case_payloads(tmp_path: Path) -> None:
    eval_path = tmp_path / 'state' / 'self_learning' / 'eval_cases.jsonl'
    report_path = tmp_path / 'state' / 'self_learning' / 'report.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text('{"case_id":"old"}\n', encoding='utf-8')
    report_path.write_text('{"case_count":1}\n', encoding='utf-8')
    eval_path.chmod(0o640)
    report_path.chmod(0o600)
    historical_case = {
        **_historical_eval_case('eval-complete-payload'),
        'empty_text': '',
        'empty_list': [],
        'empty_mapping': {},
        'null_value': None,
    }
    report = {
        'kind': 'ollmo.self_learning_report',
        'case_count': 1,
        'eval_cases': [historical_case],
    }

    written_eval, written_report = persist_self_learning_outputs(
        [historical_case],
        report,
        eval_case_output_path=eval_path,
        report_output_path=report_path,
    )

    eval_rows = [json.loads(line) for line in written_eval.read_text(encoding='utf-8').splitlines()]
    persisted_report = json.loads(written_report.read_text(encoding='utf-8'))
    assert eval_rows == [historical_case]
    assert persisted_report['eval_cases'] == [historical_case]
    assert int(written_eval.stat().st_mode) & 0o777 == 0o640
    assert int(written_report.stat().st_mode) & 0o777 == 0o600


def test_persist_self_learning_outputs_uses_normal_creation_mode_for_new_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / 'state' / 'self_learning'
    output_dir.mkdir(parents=True)
    mode_control = output_dir / 'ordinary-write.json'
    mode_control.write_bytes(b'{}\n')
    expected_mode = int(mode_control.stat().st_mode) & 0o777
    eval_path = output_dir / 'eval_cases.jsonl'
    report_path = output_dir / 'report.json'

    persist_self_learning_outputs(
        [_historical_eval_case('eval-new-file-mode')],
        {'kind': 'ollmo.self_learning_report', 'case_count': 1},
        eval_case_output_path=eval_path,
        report_output_path=report_path,
    )

    assert int(eval_path.stat().st_mode) & 0o777 == expected_mode
    assert int(report_path.stat().st_mode) & 0o777 == expected_mode


@pytest.mark.parametrize(
    'protected_kind',
    ['accepted-policy', 'response-frame', 'response-index', 'response-cas', 'retained-sidecar'],
)
def test_self_learning_cli_merge_rejects_protected_output_collisions(
    tmp_path: Path,
    protected_kind: str,
) -> None:
    frames_dir = tmp_path / 'state' / 'response_frames'
    frames_path = frames_dir / 'responses.jsonl'
    learning_dir = tmp_path / 'state' / 'self_learning'
    eval_path = learning_dir / 'eval_cases.jsonl'
    accepted_path = learning_dir / 'accepted_policy_snapshot.json'
    response_index_path = frames_dir / 'current_index.json'
    cas_path = frames_dir / 'snapshots' / 'content' / 'keep.json'
    retained_path = learning_dir / 'retained_sidecars' / 'keep.json'
    _write_jsonl(frames_path, [_positive_frame()])
    _write_jsonl(eval_path, [_historical_eval_case('eval-protected-path')])
    accepted_path.write_bytes(b'{"sentinel":"accepted"}\n')
    response_index_path.write_bytes(b'{"sentinel":"response-index"}\n')
    cas_path.parent.mkdir(parents=True, exist_ok=True)
    cas_path.write_bytes(b'{"sentinel":"cas"}\n')
    retained_path.parent.mkdir(parents=True, exist_ok=True)
    retained_path.write_bytes(b'{"sentinel":"retained"}\n')
    protected_paths = {
        'accepted-policy': accepted_path,
        'response-frame': frames_path,
        'response-index': response_index_path,
        'response-cas': cas_path,
        'retained-sidecar': retained_path,
    }
    original_bytes = {path: path.read_bytes() for path in protected_paths.values()}
    original_eval = eval_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            'scripts/build_self_learning_eval_cases.py',
            '--frames',
            str(frames_path),
            '--output',
            str(eval_path),
            '--report',
            str(protected_paths[protected_kind]),
            '--accepted-policy',
            str(accepted_path),
            '--graph-rebase-corpus-dir',
            str(tmp_path / 'state' / 'graph_rebase_shadow_corpus'),
            '--self-learning-dir',
            str(learning_dir),
            '--merge-existing',
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert 'protected' in result.stderr
    assert eval_path.read_bytes() == original_eval
    for path, payload in original_bytes.items():
        assert path.read_bytes() == payload


@pytest.mark.parametrize(
    'protected_report_path',
    [
        Path('state/response_frames/responses.jsonl'),
        Path('state/response_frames/current_index.json'),
        Path('state/response_frames/snapshots/content_sha256/keep.json'),
        Path('state/self_learning/retained_sidecars/keep.json'),
        Path('state/self_learning/accepted_policy_snapshot.json'),
    ],
)
def test_self_learning_service_persistence_rejects_default_protected_targets(
    tmp_path: Path,
    monkeypatch,
    protected_report_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    protected_report_path.parent.mkdir(parents=True, exist_ok=True)
    protected_report_path.write_bytes(b'{"sentinel":"protected-service-state"}\n')
    original_payload = protected_report_path.read_bytes()

    with pytest.raises(ValueError, match='protected state'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-service-protected-target')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=Path('state/self_learning/eval_cases.jsonl'),
            report_output_path=protected_report_path,
        )

    assert protected_report_path.read_bytes() == original_payload
    assert not Path('state/self_learning/eval_cases.jsonl').exists()


def test_self_learning_service_protects_repository_state_outside_repository_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_repo_root = tmp_path / 'repository'
    protected_report_path = fake_repo_root / 'state' / 'response_frames' / 'responses.jsonl'
    protected_report_path.parent.mkdir(parents=True)
    protected_report_path.write_bytes(b'{"sentinel":"repository-frame-ledger"}\n')
    original_payload = protected_report_path.read_bytes()
    outside_cwd = tmp_path / 'outside-cwd'
    outside_cwd.mkdir()
    from scripts import build_self_learning_eval_cases as self_learning_cli_module

    monkeypatch.setattr(self_learning_cli_module, 'REPO_ROOT', fake_repo_root)
    monkeypatch.setattr(self_learning_module, '_REPOSITORY_ROOT', fake_repo_root)
    monkeypatch.chdir(outside_cwd)

    with pytest.raises(ValueError, match='protected'):
        self_learning_cli_module._validate_merge_output_paths(
            output_path=outside_cwd / 'eval_cases.jsonl',
            report_path=protected_report_path,
            accepted_policy_path=outside_cwd / 'accepted_policy_snapshot.json',
            self_learning_dir=outside_cwd / 'self_learning',
            frame_paths=[outside_cwd / 'response_frames' / 'responses.jsonl'],
            monitor_report_path=outside_cwd / 'monitor' / 'reports.jsonl',
            graph_rebase_corpus_dir=outside_cwd / 'graph_rebase_shadow_corpus',
        )

    with pytest.raises(ValueError, match='protected state'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-outside-cwd-protection')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=outside_cwd / 'eval_cases.jsonl',
            report_output_path=protected_report_path,
        )

    assert protected_report_path.read_bytes() == original_payload
    assert not (outside_cwd / 'eval_cases.jsonl').exists()


@pytest.mark.parametrize('link_kind', ['live-target', 'dangling-target', 'parent-component'])
def test_self_learning_persistence_rejects_symlink_output_paths_without_mutation(
    tmp_path: Path,
    link_kind: str,
) -> None:
    output_dir = tmp_path / 'outputs'
    output_dir.mkdir()
    report_path = output_dir / 'report.json'
    report_path.write_bytes(b'{"sentinel":"old-report"}\n')
    original_report = report_path.read_bytes()

    if link_kind == 'parent-component':
        real_dir = tmp_path / 'real-output-dir'
        real_dir.mkdir()
        linked_dir = tmp_path / 'linked-output-dir'
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        eval_path = linked_dir / 'eval_cases.jsonl'
        protected_path = linked_dir
    else:
        eval_path = output_dir / 'eval_cases.jsonl'
        target = output_dir / 'existing-ledger.jsonl'
        if link_kind == 'live-target':
            target.write_bytes(b'{"case_id":"symlink-referent"}\n')
        eval_path.symlink_to(target.name)
        protected_path = eval_path
    original_link = protected_path.readlink()

    with pytest.raises(ValueError, match='cannot contain symlinks'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-symlink-rejection')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=eval_path,
            report_output_path=report_path,
        )

    assert protected_path.is_symlink()
    assert protected_path.readlink() == original_link
    assert report_path.read_bytes() == original_report


def test_atomic_self_learning_pair_rolls_back_when_report_install_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_path = tmp_path / 'state' / 'self_learning' / 'eval_cases.jsonl'
    report_path = tmp_path / 'state' / 'self_learning' / 'report.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_bytes(b'{"case_id":"old-case"}\n')
    report_path.write_bytes(b'{"case_count":1,"sentinel":"old-report"}\n')
    original_eval = eval_path.read_bytes()
    original_report = report_path.read_bytes()
    real_replace = self_learning_module.os.replace
    replace_count = 0

    def fail_second_replace(source, target):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError('injected report install failure')
        return real_replace(source, target)

    monkeypatch.setattr(self_learning_module.os, 'replace', fail_second_replace)

    with pytest.raises(OSError, match='injected report install failure'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-new-case')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=eval_path,
            report_output_path=report_path,
        )

    assert replace_count == 3
    assert eval_path.read_bytes() == original_eval
    assert report_path.read_bytes() == original_report
    assert json.loads(eval_path.read_text(encoding='utf-8'))['case_id'] == 'old-case'
    assert json.loads(report_path.read_text(encoding='utf-8'))['sentinel'] == 'old-report'
    assert list(eval_path.parent.glob('.*.tmp')) == []
    assert list(eval_path.parent.glob('.*.rollback')) == []


def test_atomic_self_learning_pair_staging_failure_leaves_prior_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_path = tmp_path / 'state' / 'self_learning' / 'eval_cases.jsonl'
    report_path = tmp_path / 'state' / 'self_learning' / 'report.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_bytes(b'{"case_id":"old-stage-case"}\n')
    report_path.write_bytes(b'{"case_count":1,"sentinel":"old-stage-report"}\n')
    original_eval = eval_path.read_bytes()
    original_report = report_path.read_bytes()
    real_stage = self_learning_module._stage_atomic_file_bytes
    stage_count = 0

    def fail_second_stage(target, payload):
        nonlocal stage_count
        stage_count += 1
        if stage_count == 2:
            raise OSError('injected report staging failure')
        return real_stage(target, payload)

    monkeypatch.setattr(self_learning_module, '_stage_atomic_file_bytes', fail_second_stage)

    with pytest.raises(OSError, match='injected report staging failure'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-stage-new-case')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=eval_path,
            report_output_path=report_path,
        )

    assert stage_count == 2
    assert eval_path.read_bytes() == original_eval
    assert report_path.read_bytes() == original_report
    assert list(eval_path.parent.glob('.*.tmp')) == []
    assert list(eval_path.parent.glob('.*.rollback')) == []


def test_atomic_self_learning_pair_first_install_failure_leaves_prior_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_path = tmp_path / 'state' / 'self_learning' / 'eval_cases.jsonl'
    report_path = tmp_path / 'state' / 'self_learning' / 'report.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_bytes(b'{"case_id":"old-install-case"}\n')
    report_path.write_bytes(b'{"case_count":1,"sentinel":"old-install-report"}\n')
    original_eval = eval_path.read_bytes()
    original_report = report_path.read_bytes()
    replace_count = 0

    def fail_first_replace(_source, _target):
        nonlocal replace_count
        replace_count += 1
        raise OSError('injected eval install failure')

    monkeypatch.setattr(self_learning_module.os, 'replace', fail_first_replace)

    with pytest.raises(OSError, match='injected eval install failure'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-install-new-case')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=eval_path,
            report_output_path=report_path,
        )

    assert replace_count == 1
    assert eval_path.read_bytes() == original_eval
    assert report_path.read_bytes() == original_report
    assert list(eval_path.parent.glob('.*.tmp')) == []
    assert list(eval_path.parent.glob('.*.rollback')) == []


def test_atomic_self_learning_pair_retains_recovery_files_when_rollback_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_path = tmp_path / 'state' / 'self_learning' / 'eval_cases.jsonl'
    report_path = tmp_path / 'state' / 'self_learning' / 'report.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_bytes(b'{"case_id":"old-recovery-case"}\n')
    report_path.write_bytes(b'{"case_count":1,"sentinel":"old-recovery-report"}\n')
    original_eval = eval_path.read_bytes()
    real_replace = self_learning_module.os.replace
    replace_count = 0

    def fail_report_and_rollback(source, target):
        nonlocal replace_count
        replace_count += 1
        if replace_count in {2, 3}:
            raise OSError(f'injected replace failure {replace_count}')
        return real_replace(source, target)

    monkeypatch.setattr(self_learning_module.os, 'replace', fail_report_and_rollback)

    with pytest.raises(RuntimeError, match='recovery files retained at'):
        persist_self_learning_outputs(
            [_historical_eval_case('eval-recovery-new-case')],
            {'kind': 'ollmo.self_learning_report', 'case_count': 1},
            eval_case_output_path=eval_path,
            report_output_path=report_path,
        )

    rollback_paths = list(eval_path.parent.glob('.*.rollback'))
    staged_paths = list(eval_path.parent.glob('.*.tmp'))
    assert replace_count == 3
    assert rollback_paths
    assert staged_paths
    assert any(path.read_bytes() == original_eval for path in rollback_paths)
