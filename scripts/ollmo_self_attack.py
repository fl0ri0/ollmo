#!/usr/bin/env python3
"""One-command adversarial conformance, composed with the shadow corpus runner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import shutil
import sys
import threading
import time
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_graph_rebase_shadow_corpus import (
    CorpusError, JsonHttpClient, ShadowCorpusRunner, atomic_write_json,
    assert_manifest_matches_corpus, build_manifest, load_corpus, load_manifest,
    clean_identifier, manifest_lock, stable_digest, utc_now,
    classify_compact_status,
)
from scripts.self_attack_checks import (
    audit_history, audit_truth, audit_provider_bindings, compare_profiles, finding, records,
)
from scripts.self_attack_knobs import build_profiles, discover_knobs, validate_saved_profile, diverse_live_profiles

PRIORITIES = [
    ('commitment_closure', 'commitment ↔ closure'),
    ('aspiration_promotion', 'aspiration ↔ promotion'),
    ('repair_rebase_intent', 'repair/rebase ↔ anchored intent'),
    ('late_fill_binding', 'Late Fill ↔ exact evidence/source binding'),
    ('lenses_attention_scope', 'lenses/attention ↔ graph scope'),
]
OWNER_TESTS = {
    'commitment_closure': (['tests/test_response_semantics_runtime.py', 'tests/test_ghost_decision_contracts.py'],
                           'commitment or global_semantic or truthful_freeze or branch_semantic_review'),
    'aspiration_promotion': (['tests/test_candidate_contracts.py', 'tests/test_ghost_decision_contracts.py',
                             'tests/test_request_phase_graph_runtime.py'], 'aspiration or promot or reserved or defer'),
    'repair_rebase_intent': (['tests/test_graph_rebase_review.py', 'tests/test_graph_rebase_partial_successor.py',
                             'tests/test_runtime_graph_rebase_shadow_producer.py', 'tests/test_graph_repair_self_healing.py'], ''),
    'late_fill_binding': (['tests/test_response_semantics_runtime.py', 'tests/test_fake_backend_e2e.py'],
                          'tts_stt or source_digest or producer_pair or dependency_evidence or terminal_safe_graph_patch or restart_recovery or empty_image_source'),
    'lenses_attention_scope': (['tests/test_semantic_roles.py', 'tests/test_ghost_decision_contracts.py',
                               'tests/test_response_semantics_runtime.py'], 'attention or lens or advisory or role'),
}


class LiveBudgetExpired(BaseException):
    """Hard observer stop; transport error handlers must not swallow it."""


class LiveBudget:
    """Durable limits for this command, never a policy applied to Ollmo."""
    def __init__(self, output, *, main_seconds, confirmation_seconds, attempt_cap, sequence_seconds=1200, replay_seconds=300):
        self.path = Path(output) / 'live-budget.json'
        limits = dict(main_seconds=main_seconds, confirmation_seconds=confirmation_seconds,
                      attempt_cap=attempt_cap, sequence_seconds=sequence_seconds, replay_seconds=replay_seconds)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['limits'] != limits:
                raise CorpusError('Live budget differs from the persisted run; it cannot be replenished on resume.')
        else:
            now = time.time()
            self.state = dict(limits=limits, started_at=utc_now(), main_deadline=now + main_seconds,
                              whole_deadline=now + main_seconds + confirmation_seconds,
                              stage='main', attempts=[], exhausted=[])
            self.save()
        self.monotonic_deadlines = {
            key: time.monotonic() + max(0, self.state[key] - time.time())
            for key in ('main_deadline', 'whole_deadline', 'confirmation_deadline') if key in self.state}

    def save(self):
        atomic_write_json(self.path, self.state)

    def begin_confirmation(self):
        if 'confirmation_deadline' not in self.state:
            self.state['confirmation_deadline'] = min(
                time.time() + self.state['limits']['confirmation_seconds'], self.state['whole_deadline'])
            self.monotonic_deadlines['confirmation_deadline'] = min(
                time.monotonic() + self.state['limits']['confirmation_seconds'], self.monotonic_deadlines['whole_deadline'])
        self.state['stage'] = 'confirmation'
        self.save()

    def remaining(self):
        key = 'confirmation_deadline' if self.state['stage'] == 'confirmation' else 'main_deadline'
        return max(0, min(min(self.state[key], self.state['whole_deadline']) - time.time(),
                          min(self.monotonic_deadlines[key], self.monotonic_deadlines['whole_deadline']) - time.monotonic()))

    def expire(self, reason):
        if reason not in self.state['exhausted']:
            self.state['exhausted'].append(reason)
            self.save()

    def reserve_attempt(self, key):
        # Reserve durably BEFORE execution. An interrupted attempt is consumed.
        if key in self.state['attempts']:
            return 'live_attempt_already_started'
        if self.remaining() <= 0:
            self.expire('live_confirmation_budget_exhausted')
            return 'live_confirmation_budget_exhausted'
        if len(self.state['attempts']) >= self.state['limits']['attempt_cap']:
            self.expire('live_attempt_cap_exhausted')
            return 'live_attempt_cap_exhausted'
        self.state['attempts'].append(key)
        self.save()
        return None

    @contextmanager
    def deadline(self, *, max_seconds=None, reason=None):
        """Interrupt blocked preflight/analysis as well as isolated workers."""
        if threading.current_thread() is not threading.main_thread():
            raise CorpusError('Live conformance requires the main thread for its hard deadline.')
        seconds = self.remaining()
        reason = (f'live_{self.state["stage"]}_budget_exhausted'
                  if max_seconds is None or seconds <= max_seconds else reason or 'live_replay_budget_exhausted')
        if max_seconds is not None:
            seconds = min(seconds, max_seconds)
        if seconds <= 0:
            self.expire(reason)
            raise LiveBudgetExpired(reason)
        previous_handler = signal.getsignal(signal.SIGALRM)
        def elapsed(signum, frame):
            raise LiveBudgetExpired(reason)
        signal.signal(signal.SIGALRM, elapsed)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
        started = time.monotonic()
        try:
            yield
        except LiveBudgetExpired:
            self.expire(reason)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])


class CaptureClient:
    """Observe the same response; never dispatch another request to poll it."""
    def __init__(self, inner, root, *, deadline=None):
        self.inner, self.root = inner, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.versions = {}
        self.deadline = deadline

    def timeout(self, requested):
        if self.deadline is None:
            return requested
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise LiveBudgetExpired('live_sequence_budget_exhausted')
        return min(requested, remaining)

    def capture(self, payload, source):
        if not isinstance(payload, dict) or not payload.get('id') or 'runtime' not in payload:
            return False
        self.timeout(1)  # A late transport return cannot extend observation.
        with self.lock:
            identity = clean_identifier(payload['id'], field='captured response id')
            if identity in {'.', '..'}:
                raise CorpusError('Captured response identity cannot name a parent directory.')
            folder = self.root / identity
            folder.mkdir(exist_ok=True)
            evidence = {}
            for artifact in records(payload.get('artifacts')):
                ref, name = artifact.get('artifact_ref'), artifact.get('path')
                if not ref or not name:
                    continue
                path = Path(name)
                if not path.is_absolute():
                    path = ROOT / path
                entry = {'path': str(path), 'exists': path.is_file()}
                if entry['exists']:
                    digest = hashlib.sha256()
                    retained = self.root / 'artifacts'
                    retained.mkdir(exist_ok=True)
                    before = path.stat()
                    with path.open('rb') as stream, tempfile.NamedTemporaryFile(dir=retained, delete=False) as copied:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                            copied.write(block)
                    after = path.stat()
                    entry.update(sha256=digest.hexdigest(), bytes=before.st_size,
                                 stable_during_capture=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
                    saved = retained / digest.hexdigest()
                    os.replace(copied.name, saved)
                    entry['retained_path'] = str(saved)
                evidence[ref] = entry
            value = dict(payload=payload, source=source, artifact_evidence=evidence)
            path = folder / f'{stable_digest(value)}.json'
            self.timeout(1)
            if not path.exists():
                atomic_write_json(path, dict(value, captured_at=utc_now(), observed_ns=time.time_ns()))
            return True

    def capture_truth(self, payload):
        """Label freshly fetched full truth; never promote saved observations."""
        if not isinstance(payload, dict):
            return False
        settled = classify_compact_status(payload) in {'settled_terminal', 'settled_repair_needed'}
        return self.capture(payload, 'settled' if settled else 'observation')

    def fetch_companion_truth(self, path, *, timeout, trigger):
        """One bounded retry; a debug summary never substitutes for full truth."""
        for attempt in (1, 2):
            started = time.monotonic()
            attempted_at = utc_now()
            effective_timeout = None
            try:
                effective_timeout = self.timeout(timeout)
                result = self.inner.get(path, timeout=effective_timeout)
            except LiveBudgetExpired as exc:
                self.record_truth_fetch_failure(path, trigger, attempt, attempted_at, started,
                    effective_timeout, 0, f'{type(exc).__name__}: {exc}')
                raise
            except OSError as exc:
                self.record_truth_fetch_failure(path, trigger, attempt, attempted_at, started,
                    effective_timeout, 0, f'{type(exc).__name__}: {exc}')
                if attempt == 2:
                    raise
                continue
            if result.ok:
                return result
            self.record_truth_fetch_failure(path, trigger, attempt, attempted_at, started,
                effective_timeout, result.status_code, result.error or 'HTTP request failed')
            if result.status_code not in {0, 408, 429} and result.status_code < 500:
                break
        return result

    def record_truth_fetch_failure(self, path, trigger, attempt, attempted_at, started,
                                  timeout, status, error):
        folder = self.root / 'truth-fetch-failures'
        folder.mkdir(exist_ok=True)
        atomic_write_json(folder / f'{time.time_ns()}-{uuid.uuid4().hex}.json', dict(
            endpoint=path, trigger=trigger, attempt=attempt, max_attempts=2,
            attempted_at=attempted_at, recorded_at=utc_now(), elapsed_seconds=time.monotonic() - started,
            timeout_seconds=timeout, http_status=status, error=error,
            remaining_sequence_seconds=max(0, self.deadline - time.monotonic()) if self.deadline is not None else None,
            authority='diagnostic_only', source='companion_truth_fetch'))

    def get(self, path, *, timeout):
        result = self.inner.get(path, timeout=self.timeout(timeout))
        if result.ok and path in {'/api/running_instances', '/api/ghost_preferences', '/api/graph_rebase/readiness'}:
            atomic_write_json(self.root / f'observer-{stable_digest(path)}.json',
                              {'endpoint': path, 'payload': result.payload, 'observed_at': utc_now()})
        if result.ok and '/api/responses/' in path:
            if 'view=status' in path:
                version = stable_digest(result.payload)
                if self.versions.get(path) != version:
                    # Compact observations are durable but are never graph truth.
                    atomic_write_json(self.root / f'status-{stable_digest([path, version])}.json',
                                      {'path': path, 'payload': result.payload, 'observed_at': utc_now()})
                    full = self.fetch_companion_truth(path.replace('view=status', 'view=truth'),
                                                     timeout=timeout, trigger=path)
                    if full.ok and self.capture_truth(full.payload):
                        # Failed companion reads remain retryable even when the
                        # compact state is unchanged on the next observation.
                        self.versions[path] = version
            else:
                # A truth read already has the canonical payload. Re-fetching it
                # wastes the shared sequence budget and can lose this evidence.
                full = result if 'view=truth' in path else self.fetch_companion_truth(
                    path.split('?')[0] + '?view=truth', timeout=timeout, trigger=path)
                if full.ok:
                    self.capture_truth(full.payload)
        return result

    def post(self, path, payload, *, timeout):
        result = self.inner.post(path, payload, timeout=self.timeout(timeout))
        if result.ok:
            self.capture(result.payload, 'submission')
        return result


def materialize_corpus(raw, namespace, path):
    value = deepcopy(raw)
    value['corpus_id'] = f'self-{namespace}'
    for case in value['cases']:
        fragments = (case.get('metadata') or {}).get('attack_fragments') or []
        case['prompt'] = ' '.join([case['prompt'], *fragments])
    atomic_write_json(path, value)
    return load_corpus(path)


def check_expectations(payload, case):
    meta = case.get('metadata') or {}
    graph = (payload.get('runtime') or {}).get('request_phase_graph') or {}
    obligations = records(graph.get('output_obligations'))
    capabilities = {o.get('capability') for o in obligations
                    if o.get('status') not in {'reserved', 'candidate', 'rejected'}
                    and o.get('contract_state') not in {'reserved', 'candidate'}}
    findings = []
    for capability in meta.get('forbid_capabilities', []):
        if capability in capabilities:
            findings.append(finding('unrequested_promotion', '/runtime/request_phase_graph/output_obligations',
                                    f'Reserved {capability} became an obligation.'))
    for capability in meta.get('require_capabilities', []):
        if capability not in capabilities:
            findings.append(finding('anchored_obligation_missing', '/runtime/request_phase_graph/output_obligations',
                                    f'Explicit {capability} obligation disappeared.'))
    forbidden = set(meta.get('forbid_artifact_extensions', []))
    for obligation in obligations:
        extension = obligation.get('text_artifact_extension') or (obligation.get('artifact_request') or {}).get('extension')
        if extension in forbidden and obligation.get('status') not in {'reserved', 'candidate', 'rejected'}:
            findings.append(finding('unrequested_artifact_promotion', '/runtime/request_phase_graph/output_obligations',
                                    f'Reserved {extension} artifact became an obligation.'))
    return findings


def evaluate_capture(manifest, capture_root):
    cases = []
    for case in manifest['cases']:
        folder = capture_root / case['response_id']
        snapshots, unreadable = [], []
        for path in folder.glob('*.json'):
            try:
                snapshot = json.loads(path.read_text())
                if not isinstance(snapshot, dict) or 'observed_ns' not in snapshot:
                    raise ValueError('Missing capture timestamp')
                snapshots.append(snapshot)
            except (OSError, ValueError) as exc:
                unreadable.append(f'capture_unreadable:{path.name}:{exc}')
        snapshots.sort(key=lambda v: v['observed_ns'])
        settled = [s for s in snapshots if s['source'] == 'settled']
        item = dict(case_id=case['case_id'], category=case['category'], state=case['state'],
                    response_id=case['response_id'], findings=[], missing=[])
        if not settled or case['state'] not in {'settled_terminal', 'settled_repair_needed'}:
            item['missing'].append('settled_full_truth')
        else:
            final = settled[-1]
            try:
                audit = audit_truth(final['payload'], artifact_evidence=final['artifact_evidence'])
                item.update(audit)
                item['findings'] += check_expectations(final['payload'], case)
                item['findings'] += audit_history([s['payload'] for s in snapshots if s['source'] != 'submission'])
            except (KeyError, TypeError, ValueError) as exc:
                item['missing'].append(f'oracle_input_error:{type(exc).__name__}:{exc}')
            item['payload'] = final['payload']
            item['capture_path'] = str(folder)
            for check in (case.get('metadata') or {}).get('require_exercised', []):
                if check not in item.get('exercised', []):
                    item['missing'].append(f'scenario_not_exercised:{check}')
        item['missing'] += unreadable
        cases.append(item)
    provider_path = capture_root.parent / 'backend_calls.json'
    if provider_path.exists():
        for binding in audit_provider_bindings(json.loads(provider_path.read_text()).get('records', [])):
            case = next((c for c in cases if c['response_id'] == binding['response_id']), None)
            if case:
                case['findings'] += binding['findings']
                if binding['missing']:
                    case['missing'].append(binding['missing'])
                else:
                    case.setdefault('exercised', []).append('exact_provider_artifact_binding')
    return cases


def explain_incomplete(cases, output, *, error=None):
    """Classify absent evidence without guessing that a timeout proves a bug."""
    progress_path = Path(output) / 'execution-progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    by_id = {case['case_id']: case for case in cases}
    manifest_path = Path(output) / 'manifest.json'
    originals = {c['case_id']: c for c in load_manifest(manifest_path)['cases']} if manifest_path.exists() else {}
    for case in cases:
        observations = []
        dependencies = originals.get(case['case_id'], {}).get('depends_on', [])
        for reason in case['missing']:
            waiting = [key for key in dependencies if by_id.get(key, {}).get('missing')]
            scenario = (reason.startswith('scenario_not_exercised:') or case['state'] == 'unobserved'
                        or (case['state'] == 'planned' and waiting))
            observations.append(dict(
                classification='missing_scenario_coverage' if scenario else 'missing_observability',
                code='predecessor_not_settled' if scenario and waiting else reason,
                detail=(f'Not dispatched because predecessors lack settled truth: {waiting}.' if scenario and waiting
                        else f'{reason}; state={case["state"]}; worker={error or "returned"}; last owner stage={progress.get("stage", "unobserved")}.'),
                evidence_path=str(manifest_path), execution_progress=progress))
        case['incomplete_observations'] = observations
    return cases


def run_profile(raw, profile, output, *, mode, base_url, cycles, namespace, deadline_at=None):
    output.mkdir(parents=True, exist_ok=True)
    corpus = materialize_corpus(raw, namespace, output / 'corpus.json')
    manifest_path = output / 'manifest.json'
    with manifest_lock(manifest_path):
        manifest = load_manifest(manifest_path) if manifest_path.exists() else build_manifest(corpus, manifest_path)
        assert_manifest_matches_corpus(manifest, corpus)
        def transform(payload, case):
            payload.update(deepcopy(profile['request']))
            if mode == 'fake':
                # Test-provider workload, independent of expected verdicts. Live
                # Ghost must derive its own route and source from the same prompt.
                fixture = (case.get('metadata') or {}).get('fake_request') or {}
                if set(fixture) - {'capability_hint', 'content_payload'}:
                    raise CorpusError('Unsupported fake-provider fixture field.')
                payload.update(deepcopy(fixture))
            # Explicit current-turn references receive only the exact predecessor's
            # saved truth. Other history never becomes fresh work by recency.
            predecessors = [c for c in manifest['cases'] if c['case_id'] in case.get('depends_on', [])]
            if predecessors:
                messages = []
                refs = []
                for previous in predecessors:
                    summary = (previous.get('final_debug') or {}).get('summary') or {}
                    messages.append({'role': 'user', 'content': previous['prompt']})
                    messages.append({'role': 'assistant', 'content': json.dumps({
                        'response_id': previous['response_id'], 'lifecycle_state': previous.get('last_lifecycle_state'),
                        'outputs': summary.get('outputs'), 'artifacts': summary.get('artifacts')})})
                    refs.extend(records(summary.get('artifacts')))
                payload['ghost_messages'] = messages
                if refs:
                    payload['reference_artifacts'] = refs
            return payload
        if mode == 'fake':
            from tests.fake_backends.self_attack import SelfAttackBackend
            context = SelfAttackBackend(root=output / 'runtime')
        else:
            context = nullcontext(JsonHttpClient(base_url))
        # Restore settings even on failures. The fake harness joins/drains its work
        # before this context closes; live processes never receive env overrides.
        with patch.dict(os.environ, profile['environment'], clear=False), context as inner:
            probes = {}
            if mode == 'fake':
                from tests.fake_backends.self_attack_probes import probe_boundaries
                probes = probe_boundaries(profile['request'])
                atomic_write_json(output / 'owner_probes.json', probes)
            deadline = time.monotonic() + max(0, deadline_at - time.time()) if deadline_at is not None else None
            client = CaptureClient(inner, output / 'captures', deadline=deadline)
            runner = ShadowCorpusRunner(corpus=corpus, manifest=manifest, manifest_path=manifest_path,
                                        client=client, poll_interval=.01 if mode == 'fake' else 2,
                                        request_transform=transform, emit=lambda _: None)
            status = runner.run(max_cycles=cycles)
            # Do not let in-flight fake calls outlive patched roots or controls.
            if mode == 'fake':
                for thread in list(runner._dispatch_threads.values()):
                    thread.join()
                runner._drain_dispatch_results()
                if status and not inner.scheduled:
                    status = runner.run(max_cycles=3)
            # Reobserve settled parents after the sequence, including after a
            # later user turn. A continuation must not rewrite a frozen frame.
            for case in manifest['cases']:
                if case['state'] in {'settled_terminal', 'settled_repair_needed'}:
                    client.get(f'/api/responses/{case["response_id"]}?view=truth', timeout=30)
            cases = explain_incomplete(evaluate_capture(manifest, output / 'captures'), output)
            if mode == 'fake':
                atomic_write_json(output / 'backend_calls.json', {'calls': dict(inner.calls), 'records': inner.call_records})
        return dict(profile=profile, runner_status=status, cases=cases, probes=probes, output=str(output))


def unfinished_group(raw, profile, output, error):
    """Retain observer evidence without dispatching or inventing a verdict."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.json'
    if manifest.exists():
        cases = evaluate_capture(load_manifest(manifest), output / 'captures')
    else:
        cases = [dict(case_id=c['case_id'], category=c['category'], state='unobserved',
                      findings=[], missing=['worker_completion']) for c in raw['cases']]
    for case in cases:
        # A later stalled turn does not invalidate a previously settled capture.
        if case['missing']:
            case['missing'].append(error or 'worker_result_missing')
    explain_incomplete(cases, output, error=error)
    probes_path = output / 'owner_probes.json'
    probes = json.loads(probes_path.read_text()) if probes_path.exists() else {}
    return dict(profile=profile, runner_status=2, cases=cases, probes=probes,
                output=str(output), error=error)


def isolated_group(raw, profile, output, *, mode, base_url, cycles, namespace, timeout=120):
    """A process owns all provider threads, controls and diagnostic storage."""
    output.mkdir(parents=True, exist_ok=True)
    deadline_at = None
    if mode == 'live':
        budget_path = output / 'observation-budget.json'
        if budget_path.exists():
            deadline_at = json.loads(budget_path.read_text())['deadline_at']
        else:
            deadline_at = time.time() + max(0, timeout)
            atomic_write_json(budget_path, dict(deadline_at=deadline_at, seconds=timeout))
        timeout = min(timeout, deadline_at - time.time())
        if timeout <= 0:
            return unfinished_group(raw, profile, output, 'live_sequence_budget_exhausted')
    config = dict(raw=raw, profile=profile, output=str(output), mode=mode,
                  base_url=base_url, cycles=cycles, namespace=namespace, deadline_at=deadline_at)
    job = output / 'worker.json'
    atomic_write_json(job, config)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('OLLMO_GRAPH_REBASE_OPERATOR_')
                   and (mode != 'fake' or not key.startswith('OLLMO_'))}
    if mode == 'fake':
        # Startup-bound owners must see these before importing the webserver.
        environment.update(profile['environment'])
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(job)]
    error = None
    with (output / 'worker.log').open('a') as log:
        try:
            if deadline_at is not None:
                timeout = min(timeout, deadline_at - time.time())
                if timeout <= 0:
                    return unfinished_group(raw, profile, output, 'live_sequence_budget_exhausted')
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout)
            if completed.returncode:
                error = f'worker_exit_{completed.returncode}'
        except subprocess.TimeoutExpired:
            error = f'{"live_sequence" if mode == "live" else "profile"}_budget_exhausted_{timeout}s'
    if not error and (output / 'worker-result.json').exists():
        return json.loads((output / 'worker-result.json').read_text())
    return unfinished_group(raw, profile, output, error)


def sequence_corpora(raw):
    # Merge shared conversations and dependency-connected cases before partitioning.
    groups = [{c['case_id']} for c in raw['cases']]
    by_id = {c['case_id']: c for c in raw['cases']}
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(groups):
            for j in range(i + 1, len(groups)):
                b = groups[j]
                connected = any(set(by_id[k].get('depends_on', [])) & b for k in a)
                connected |= any(set(by_id[k].get('depends_on', [])) & a for k in b)
                connected |= bool({by_id[k].get('conversation_key', k) for k in a}
                                  & {by_id[k].get('conversation_key', k) for k in b})
                if connected:
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [dict(raw, cases=[c for c in raw['cases'] if c['case_id'] in group]) for group in groups]


def combine_groups(profile, output, runs):
    return dict(profile=profile, runner_status=max(r['runner_status'] for r in runs),
                cases=[c for r in runs for c in r['cases']],
                probes=next((r['probes'] for r in runs if r.get('probes')), {}),
                output=str(output), groups=[dict(output=r['output'], error=r.get('error')) for r in runs])


def unfinished_profile(raw, profile, output, error):
    return combine_groups(profile, output, [
        unfinished_group(corpus, profile, output / f'sequence-{index + 1}', error)
        for index, corpus in enumerate(sequence_corpora(raw))])


def isolated_profile(raw, profile, output, *, mode, base_url, cycles, namespace, timeout=120,
                     live_budget=None):
    groups = sequence_corpora(raw)
    runs = []
    for index, corpus in enumerate(groups):
        # Live uses the explicit sequence ceiling; the shared stage budget owns
        # total duration. Smaller reducer corpora never acquire longer windows.
        seconds = timeout / len(groups) if live_budget is None else min(
            live_budget.state['limits']['sequence_seconds'], live_budget.remaining())
        run = isolated_group(corpus, profile, output / f'sequence-{index + 1}', mode=mode,
                             base_url=base_url, cycles=cycles, namespace=f'{namespace}-{index + 1}',
                             timeout=seconds)
        runs.append(run)
    result = combine_groups(profile, output, runs)
    if mode == 'live':
        atomic_write_json(output / 'profile-result.json', result)
    return result


def signatures(result):
    values = {(case['case_id'], f['code'], f.get('signature', '')) for case in result['cases'] for f in case['findings']}
    values |= {(f'@probe:{check["name"]}', f['code'], f.get('signature', ''))
               for check in result.get('probes', {}).get('checks', []) for f in check['findings']}
    return values


def matches_signature(target, values):
    return any(tuple(target) == v[:len(target)] for v in values)


def minimize_failure(raw, profile, target, replay, *, budget=24, confirmations=2):
    """Bounded deletion reduction, preserving the same case/invariant signature.

    Immutable intent clauses and expectations remain intact. Only adversarial
    fragments, irrelevant turns and profile deltas are reducible.
    """
    attempts = []
    stop_reason = None
    def reproduces(corpus, controls):
        nonlocal stop_reason
        if stop_reason or len(attempts) + confirmations > budget:
            return False
        for _ in range(confirmations):
            result = replay(corpus, controls, len(attempts))
            if result.get('replay_stop_reason'):
                stop_reason = result['replay_stop_reason']
                return False
            incomplete = bool(result.get('runner_status') or any(c.get('missing') for c in result['cases']))
            matched = not incomplete and matches_signature(target, signatures(result))
            attempts.append({'corpus_digest': stable_digest(corpus), 'profile': controls['id'],
                             'matched': matched, 'incomplete': incomplete, 'output': result.get('output')})
            if incomplete:
                stop_reason = 'confirmation_observation_incomplete'
            if not matched:
                return False
        return True
    reduced, controls = deepcopy(raw), deepcopy(profile)
    if not reproduces(reduced, controls):
        return dict(status='unconfirmed', corpus=reduced, profile=controls, attempts=attempts, stop_reason=stop_reason)
    # Remove dependent leaves first; never leave a dangling predecessor.
    changed = True
    while changed and len(attempts) + confirmations <= budget:
        changed = False
        for case in reversed(reduced['cases']):
            if len(reduced['cases']) <= 1:
                break
            if case['case_id'] == target[0] or any(case['case_id'] in c.get('depends_on', []) for c in reduced['cases']):
                continue
            candidate = deepcopy(reduced)
            candidate['cases'] = [c for c in candidate['cases'] if c['case_id'] != case['case_id']]
            if reproduces(candidate, controls):
                reduced, changed = candidate, True
                break
    for case in list(reduced['cases']):
        for fragment in list((case.get('metadata') or {}).get('attack_fragments') or []):
            candidate = deepcopy(reduced)
            entry = next(c for c in candidate['cases'] if c['case_id'] == case['case_id'])
            entry['metadata']['attack_fragments'].remove(fragment)
            if reproduces(candidate, controls):
                reduced = candidate
    for key in list(controls['settings']):
        candidate = deepcopy(controls)
        candidate['settings'].pop(key)
        if key in candidate['environment']:
            candidate['environment'].pop(key)
        elif key.startswith('developer_flags.'):
            candidate['request']['developer_flags'].pop(key.split('.', 1)[1])
        else:
            candidate['request'].pop(key, None)
        candidate['id'] = stable_digest(candidate['settings'])[:12]
        if reproduces(reduced, candidate):
            controls = candidate
    return dict(status='confirmed', corpus=reduced, profile=controls, attempts=attempts,
                budget_exhausted=bool(stop_reason) or len(attempts) + confirmations > budget, stop_reason=stop_reason,
                reduction='bounded deletion; minimality not claimed')


def run_owner_tests(output, *, timeout=600):
    results = {}
    for key, label in PRIORITIES:
        paths, selection = OWNER_TESTS[key]
        junit = output / f'{key}.xml'
        command = [sys.executable, '-m', 'pytest', *paths, '-q', f'--junitxml={junit}']
        if selection:
            command += ['-k', selection]
        print(f'Checking {label} runtime owners…', flush=True)
        with (output / f'{key}.log').open('w') as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                           env={k: v for k, v in os.environ.items() if not k.startswith('OLLMO_GRAPH_REBASE_OPERATOR_')})
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        cases = []
        if junit.exists():
            for case in ET.parse(junit).getroot().iter('testcase'):
                cases.append(dict(name=case.attrib['name'], classname=case.attrib.get('classname'),
                                  status='failed' if case.find('failure') is not None or case.find('error') is not None
                                  else 'skipped' if case.find('skipped') is not None else 'passed'))
        results[key] = dict(exit_code=exit_code, cases=cases,
                            status='passed' if exit_code == 0 and cases else 'incomplete' if exit_code == 124 else 'failed')
    return results


def verdict_scopes(result):
    """Report selected coverage separately; never promote the legacy verdict."""
    scopes = dict(deterministic_fake='incomplete', representative_live_gate='incomplete',
                  full_live_conformance='incomplete')
    if result['mode'] == 'fake':
        scopes['deterministic_fake'] = result['verdict']
        live = result.get('live') or {}
        scopes['representative_live_gate'] = (live.get('scoped_verdicts') or {}).get(
            'representative_live_gate', 'incomplete')
        if live.get('status') in {'passed', 'failed', 'incomplete'}:
            scopes['full_live_conformance'] = live['status']
        return scopes

    scopes['deterministic_fake'] = result.get('fake_conformance_verdict', 'incomplete')
    scopes['full_live_conformance'] = result['verdict']
    selected = result.get('selected_live_cases') or result['coverage'].get('case_selection') or {}
    sweep = result['runs'][:result['coverage']['profiles_selected']]
    observed = result['runs'] + result.get('confirmation_runs', [])
    failed = (result['verdict'] == 'failed' or any(signatures(run) for run in observed)
              or any(check.get('status') == 'failed' for run in observed
                     for check in run.get('probes', {}).get('checks', []))
              or any(owner.get('status') == 'failed' for owner in result['owner_tests'].values()))
    complete = (bool(selected) and len(selected) == result['coverage']['profiles_selected'] == len(sweep)
                and len({run['profile']['id'] for run in sweep}) == len(selected)
                and {run['profile']['id'] for run in sweep} == set(selected)
                and all(result['owner_tests'].get(key, {}).get('status') == 'passed' for key, _ in PRIORITIES)
                and not result.get('confirmation_stop_reason'))
    for run in sweep:
        expected = selected.get(run['profile']['id'], [])
        actual = [case['case_id'] for case in run['cases']]
        complete &= (bool(expected) and len(actual) == len(set(actual)) == len(expected)
                     and set(actual) == set(expected))
    for run in observed:
        complete &= (run.get('runner_status') == 0 and all(
            bool(case.get('capture_path')) and not case.get('missing', ['unobserved'])
            and case.get('state') in {'settled_terminal', 'settled_repair_needed'} for case in run['cases']))
    scopes['representative_live_gate'] = 'failed' if failed else 'passed' if complete else 'incomplete'
    return scopes


def refresh_report(source):
    """Reformat completed evidence only; no oracle, provider or sweep execution."""
    source = Path(source).resolve()
    with manifest_lock(source / 'run.json'):
        result = json.loads((source / 'results.json').read_text())
        config = json.loads((source / 'run.json').read_text())
        if result['mode'] == 'live':
            result['selected_live_cases'] = config.get('case_selection') or {
                profile['id']: [case['case_id'] for case in config['corpus']['cases']]
                for profile in config['profiles']}
            proof_path = result.get('fake_evidence')
            if proof_path:
                proof = json.loads(Path(proof_path).read_text())
                if proof.get('mode') == 'fake':
                    result['fake_conformance_verdict'] = proof['verdict']
        elif (result.get('live') or {}).get('results'):
            live = json.loads(Path(result['live']['results']).read_text())
            result['live']['scoped_verdicts'] = verdict_scopes(live)
        result['scoped_verdicts'] = verdict_scopes(result)
        backup = source / 'reporting-backups' / str(time.time_ns())
        backup.mkdir(parents=True)
        for name in ('results.json', 'report.md'):
            if (source / name).exists():
                shutil.copy2(source / name, backup / name)
        result['reporting_refresh'] = dict(updated_at=utc_now(), execution_performed=False,
            oracle_rechecked=False, previous_report=str(backup), capture_source_digest=config['source_digest'])
        report = render_report(result)
        previous_report = (backup / 'report.md').read_text() if (backup / 'report.md').exists() else ''
        # Keep the operator's continuation accounting alongside the new headings.
        marker = '\n\n## Continuation accounting\n'
        if marker in previous_report:
            report += marker + previous_report.split(marker, 1)[1]
        atomic_write_json(source / 'results.json', result)
        (source / 'report.md').write_text(report, encoding='utf-8')
        print(f'Report refreshed from completed evidence: {source / "report.md"}', flush=True)
        return 0


def render_report(result):
    sweep = result['runs'][:result['coverage']['profiles_selected']]
    observed = list({r.get('output', str(i)): r for i, r in enumerate(
        result['runs'] + result.get('confirmation_runs', []))}.values())
    lines = ['# Ollmo self-attack conformance', '', '**Knobs may change strategy, never truth.**', '',
             f"Verdict: **{result['verdict']}** · mode: {result['mode']} · profiles: {len(sweep)}", '',
             'Pass/fail comes from deterministic runtime assertions. LLM prose has no vote.', '',
             (f"Offline recheck of `{result['recheck']['evidence_origin']}`; no requests were executed."
              if result.get('recheck') else ''), '',
             '| Priority | Boundary | Full captures / cases | Findings | Incomplete cases | Profile probes | Owner tests |',
             '|---|---|---:|---:|---:|---:|---|']
    labels = dict(passed='PASS', failed='FAIL', incomplete='INCOMPLETE')
    scopes = verdict_scopes(result)
    lines[6:6] = [
        f"Deterministic/fake conformance: **{labels[scopes['deterministic_fake']]}**",
        f"Representative live gate: **{labels[scopes['representative_live_gate']]}**",
        f"Full live conformance: **{labels[scopes['full_live_conformance']]}**", '',
        'Representative PASS covers only the selected profiles/cases. Full live PASS requires the complete intended live scope.',
        'A scope without sufficient recorded evidence remains INCOMPLETE.', '',
    ]
    if result.get('live'):
        lines[6:6] = [f"Live Ghost: **{result['live']['status']}** (separate observation; fake invariants remain authoritative).",
                      f"Combined observation verdict: **{result.get('overall_verdict', 'incomplete')}**.", '']
    for i, (key, label) in enumerate(PRIORITIES, 1):
        cases = [c for run in sweep for c in run['cases'] if c['category'] == key]
        owner = result['owner_tests'].get(key) or {}
        probes = [p for run in sweep for p in run.get('probes', {}).get('checks', []) if p['category'] == key]
        lines.append(f"| {i} | {label} | {sum(bool(c.get('capture_path')) and not c['missing'] for c in cases)}/{len(cases)} | {sum(len(c['findings']) for c in cases)} | "
                     f"{sum(bool(c['missing']) for c in cases)} | {sum(p['status'] == 'passed' for p in probes)}/{len(probes)} | {owner.get('status', 'not run')} |")
    lines += ['', '## Controls and coverage', '',
              f"Planned design: {result['coverage']['design']}. Profiles selected: {result['coverage']['profiles_selected']}/{result['coverage']['profiles_available']}.",
              f"Profiles with complete captures: {sum(not r['runner_status'] and not any(c['missing'] for c in r['cases']) for r in sweep)}/{result['coverage']['profiles_available']}. Regression replays are reported separately.",
              'Finite controls are derived from runtime catalogs and bounds; unobserved profiles are not counted as tested.',
              f"Inventory-only controls: {result['coverage']['inventory_only']}. See controls.json for exact sources and exclusions.",
              'Live runs change request controls only. Environment controls require the isolated fake-provider run.',
              'Fake runs exercise runtime owners with deterministic model/routing substitutes; they do not certify live Ghost interpretation.',
              '', '## Findings', '']
    if result['coverage'].get('case_selection'):
        selection = result['coverage']['case_selection']
        insert = lines.index('## Findings')
        lines[insert:insert] = [
            f"High-risk subset: {sum(map(len, selection.values()))} planned case executions across six profiles; all five boundaries are represented across the gate, not on every profile.",
            'Omitted profile/case combinations remain unobserved. Cross-profile comparisons apply only where the same case has captured baseline truth.',
            *[f"- `{profile}`: {', '.join(cases)}" for profile, cases in selection.items()], '',
        ]
    for run in observed:
        for check in run.get('probes', {}).get('checks', []):
            for f in check['findings']:
                lines.append(f"- `{run['profile']['id']}` / `{check['name']}`: **{f['code']}**")
        for case in run['cases']:
            for f in case['findings']:
                lines.append(f"- `{case['case_id']}` / `{run['profile']['id']}`: **{f['code']}** — {f['detail']}")
    if not any(signatures(r) for r in observed):
        lines.append('No deterministic invariant failures found in observed cases.')
    lines += ['', '## Incomplete observations', '']
    incomplete_cases = [c for r in observed for c in r['cases'] if c['missing']]
    for run in observed:
        for case in run['cases']:
            for item in case.get('incomplete_observations', []):
                lines.append(f"- `{run['profile']['id']}` / `{case['case_id']}`: **{item['classification']}** — {item['detail']}")
    if not incomplete_cases:
        lines.append('None in executed cases.')
    if result['coverage']['profiles_selected'] < result['coverage']['profiles_available']:
        lines.append('Missing scenario coverage: the selected profile limit excluded part of the planned sweep.')
    if result['coverage'].get('case_selection'):
        lines.append('Missing scenario coverage: the high-risk selection omits other profile/case combinations; it cannot establish full live conformance.')
    for key, label in PRIORITIES:
        if result['owner_tests'].get(key, {}).get('status') != 'passed':
            lines.append(f'Missing observability: {label} owner checks did not complete successfully.')
    lines += ['', '## Regressions', '']
    for regression in result['regressions']:
        lines.append(f"- {regression['status']}: {regression.get('path', regression.get('target'))}")
    if not result['regressions']:
        lines.append('No new reproducible failures persisted.')
    for replay in result.get('regression_replays', []):
        lines.append(f"- Existing regression **{replay['status']}**: `{replay['path']}`")
    if result.get('live_budget'):
        budget = result['live_budget']
        limits = budget['limits']
        lines += ['', '## Hard live limits', '',
                  f"Sequence observation ceiling: {limits['sequence_seconds']:g}s. Main sweep including preflight: {limits['main_seconds']:g}s. All confirmation work: {limits['confirmation_seconds']:g}s.",
                  f"Each additional profile replay: at most {limits['replay_seconds']:g}s. Total live execution: at most {limits['main_seconds'] + limits['confirmation_seconds']:g}s plus evidence/report finalization.",
                  f"Additional profile attempts consumed: {len(budget['attempts'])}/{limits['attempt_cap']} (saved regressions and paired baseline replays included).",
                  f"Exhausted limits: {', '.join(budget['exhausted']) or 'none'}.",
                  'Deadlines and attempts persist across resume. An exhausted limit cannot schedule more live work. Final report writing occurs with dispatch disabled; already submitted Ollmo responses may continue.', '']
    if result.get('manual_review'):
        lines += ['', '## Manual review required', '',
                  'These records preserve unresolved observations or unconfirmed findings. They are not automatically replayed.']
        for item in result['manual_review']:
            lines.append(f"- `{item['profile']}` / `{item.get('case_id') or item['kind']}`: {item['reason']} — [{Path(item['path']).name}]({item['path']})")
    if result.get('capture_validations'):
        lines += ['', '## Supplemental capture validation', '',
                  'These are newly fetched, read-only canonical captures of existing responses after the original run. '
                  'They supplement selected-case evidence; they do not relabel historical snapshots, rerun model work, '
                  'or claim that the original sequence captured the missing evidence within its deadline.']
        for validation in result['capture_validations']:
            lines.append(f"- `{validation['profile']}` / `{validation['case_id']}`: **{validation['status']}**; "
                         f"captured {validation['finished_at']}; unchanged oracle checked the affected case and its retained history. "
                         f"[Validation evidence]({validation['path']}).")
    lines += ['', '## Evidence', '',
              'results.json contains verdicts, settings, findings, reproduction records and coverage. Each profile directory contains the original shadow manifest, full captures, retained artifact bytes and (fake mode) the isolated runtime ledger/sidecars.',
              'Missing or unfinished observations make the result incomplete. A blocked runtime obligation can be truthful and is not by itself a conformance failure.', '']
    return '\n'.join(lines)


def recheck_captures(source: Path, output: Path, inventory: dict) -> int:
    """Reapply the deterministic oracle to retained evidence, without dispatch."""
    prior_path = source / 'results.json'
    if not prior_path.exists():
        prior_path = source / 'progress.json'
    prior = json.loads(prior_path.read_text())
    config = json.loads((source / 'run.json').read_text())
    result = dict(schema_version=1, mode=prior['mode'], started_at=utc_now(), runs=[],
                  regressions=[], regression_replays=[], owner_tests=prior['owner_tests'], coverage=prior['coverage'],
                  recheck=dict(evidence_origin=str(source), execution_performed=False,
                               capture_source_digest=config['source_digest'], analysis_source_digest=inventory['source_digest']))
    baseline = {}
    for previous in prior['runs']:
        cases = []
        for group in previous.get('groups', [{'output': previous['output']}]):
            directory = Path(group['output'])
            manifest = directory / 'manifest.json'
            if manifest.exists():
                cases.extend(evaluate_capture(load_manifest(manifest), directory / 'captures'))
        if not cases:
            cases = [dict(c, findings=[], missing=['retained_capture_unavailable']) for c in previous['cases']]
        run = dict(previous, cases=cases)
        for case in cases:
            if case.get('payload') and not case['missing']:
                if run['profile']['id'] == 'baseline':
                    baseline[case['case_id']] = case['payload']
                elif case['case_id'] in baseline:
                    case['findings'] += compare_profiles(baseline[case['case_id']], case['payload'])
            case.pop('payload', None)
        result['runs'].append(run)
    failures = any(signatures(r) for r in result['runs']) or any(v['status'] == 'failed' for v in result['owner_tests'].values())
    incomplete = (not result['owner_tests'] or len(result['runs']) < result['coverage']['profiles_available']
                  or bool(result['coverage'].get('case_selection'))
                  or any(v['status'] != 'passed' for v in result['owner_tests'].values())
                  or any(r['runner_status'] or any(c['missing'] for c in r['cases']) for r in result['runs']))
    result['verdict'] = 'failed' if failures else 'incomplete' if incomplete else 'passed'
    result['finished_at'] = utc_now()
    if result['mode'] == 'live':
        result['selected_live_cases'] = config.get('case_selection') or {
            p['id']: [c['case_id'] for c in config['corpus']['cases']] for p in config['profiles']}
        result['fake_conformance_verdict'] = prior.get('fake_conformance_verdict', 'incomplete')
    result['scoped_verdicts'] = verdict_scopes(result)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'controls.json', inventory)
    atomic_write_json(output / 'results.json', result)
    (output / 'report.md').write_text(render_report(result), encoding='utf-8')
    print(f"{result['verdict'].upper()}: {output / 'report.md'}", flush=True)
    return {'passed': 0, 'failed': 1, 'incomplete': 2}[result['verdict']]


def high_risk_live_cases(raw, profiles):
    """Dependency-closed subsets of the already fake-validated corpus."""
    if len(profiles) != 6:
        raise CorpusError('The high-risk live subset requires six representative profiles.')
    categories = [
        ('aspiration_promotion', 2), ('aspiration_promotion', 2),
        ('repair_rebase_intent', 2), ('late_fill_binding', 1),
        ('lenses_attention_scope', 2), ('commitment_closure', 1),
    ]
    by_id = {c['case_id']: c for c in raw['cases']}
    selected = {}
    for profile, (category, count) in zip(profiles, categories):
        choices = [c['case_id'] for c in raw['cases'] if c['category'] == category][:count]
        if len(choices) != count:
            raise CorpusError(f'High-risk live subset lacks {category} cases.')
        pending = list(choices)
        while pending:
            for predecessor in by_id[pending.pop()].get('depends_on', []):
                if predecessor not in choices:
                    choices.append(predecessor)
                    pending.append(predecessor)
        selected[profile['id']] = [c['case_id'] for c in raw['cases'] if c['case_id'] in choices]
    return selected


def sweep_corpus(raw, profile, result):
    selected = result['coverage'].get('case_selection', {}).get(profile['id'])
    return dict(raw, cases=[c for c in raw['cases'] if c['case_id'] in selected]) if selected is not None else raw


def execute_sweep(args, raw, profiles, output, run_id, result, live_budget):
    live_error = None
    if args.mode == 'live':
        observations = []
        client = JsonHttpClient(args.base_url)
        for endpoint in ('runtime_manifest', 'running_instances', 'backend_fabric', 'ghost', 'runtime_status', 'ghost_preferences'):
            response = client.get(f'/api/{endpoint}', timeout=min(5, live_budget.remaining()))
            observations.append(dict(endpoint=endpoint, status=response.status_code, payload=response.payload, error=response.error))
            atomic_write_json(output / 'live-preflight.json', observations)
            if not response.ok:
                live_error = f'live_preflight_unavailable:{endpoint}:{response.error or response.status_code}'
                result['live_preflight_error'] = live_error
                break
        result['live_preflight_passed'] = not live_error and len(observations) == 6
    def execute(profile):
        corpus = sweep_corpus(raw, profile, result)
        if live_error:
            cases = [dict(case_id=c['case_id'], category=c['category'], state='unobserved', findings=[], missing=[live_error]) for c in corpus['cases']]
            return dict(profile=profile, corpus=corpus, runner_status=2, cases=explain_incomplete(cases, output, error=live_error), probes={}, output=str(output))
        path = output / profile['id']
        if live_budget and (path / 'profile-result.json').exists():
            run = json.loads((path / 'profile-result.json').read_text())
            if corpus != raw:
                run['corpus'] = corpus
            return run
        if live_budget and args.replay:
            reason = live_budget.reserve_attempt(str(path.relative_to(output)))
            if reason:
                return unfinished_profile(raw, profile, path, reason)
        run = isolated_profile(corpus, profile, path, mode=args.mode,
            base_url=args.base_url, cycles=args.max_cycles, namespace=f'{run_id}-{profile["id"]}',
            timeout=args.profile_timeout, live_budget=live_budget)
        if corpus != raw:
            run['corpus'] = corpus
        return run
    baseline = {}
    # A live deadline must interrupt the same thread that owns subprocess.run,
    # so its exception kills/reaps the observer and cannot leave queued work.
    with ThreadPoolExecutor(max_workers=args.jobs) if args.mode == 'fake' else nullcontext() as pool:
        futures = [pool.submit(execute, profile) for profile in profiles] if pool else []
        try:
            result['runs'] = []
            for index, profile in enumerate(profiles, 1):
                print(f'Profile {index}/{len(profiles)}: {profile["id"]}', flush=True)
                run = futures[index - 1].result() if pool else execute(profile)
                for case in run['cases']:
                    if case.get('payload') and not case['missing']:
                        if profile['id'] == 'baseline':
                            baseline[case['case_id']] = case['payload']
                        elif case['case_id'] in baseline:
                            case['findings'] += compare_profiles(baseline[case['case_id']], case['payload'])
                    case.pop('payload', None)
                result['runs'].append(run)
                if live_budget:
                    result['live_budget'] = deepcopy(live_budget.state)
                atomic_write_json(output / 'progress.json', result)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def compare_reference(run, reference):
    previous = {c['case_id']: c for c in reference['cases']}
    for case in run['cases']:
        other = previous.get(case['case_id'], {})
        if case.get('payload') and not case['missing'] and other.get('payload') and not other['missing']:
            case['findings'] += compare_profiles(other['payload'], case['payload'])
        else:
            case['missing'].append('comparison_reference_truth_unavailable')
    run['runner_status'] = max(run['runner_status'], reference['runner_status'])


def confirm_failures(args, raw, profiles, output, run_id, result, inventory, live_budget):
    remaining = args.minimize_budget
    result.setdefault('confirmation_runs', [])
    def extra(corpus, profile, path, namespace):
        pending = {c['case_id'] for r in result['runs'] + result['confirmation_runs'] for c in r['cases']
                   if c.get('response_id') and c['state'] not in {'planned', 'unobserved', 'settled_terminal', 'settled_repair_needed'}}
        reason = ('live_pending_response_not_replayed' if live_budget and pending & {c['case_id'] for c in corpus['cases']}
                  else live_budget.reserve_attempt(str(path.relative_to(output))) if live_budget else None)
        if reason == 'live_attempt_already_started' and (path / 'profile-result.json').exists():
            run = json.loads((path / 'profile-result.json').read_text())
        elif reason:
            run = unfinished_profile(corpus, profile, path, reason)
            run['replay_stop_reason'] = reason
        else:
            try:
                with live_budget.deadline(max_seconds=args.live_replay_timeout, reason='live_replay_budget_exhausted') if live_budget else nullcontext():
                    run = isolated_profile(corpus, profile, path, mode=args.mode,
                        base_url=args.base_url, cycles=args.max_cycles, namespace=namespace,
                        timeout=args.profile_timeout, live_budget=live_budget)
            except LiveBudgetExpired as exc:
                live_budget.expire(str(exc))
                run = unfinished_profile(corpus, profile, path, str(exc))
                run['replay_stop_reason'] = str(exc)
                result['confirmation_stop_reason'] = str(exc)
        run['corpus'] = corpus
        result['confirmation_runs'].append(run)
        atomic_write_json(output / 'progress.json', result)
        return run
    # Previously persisted regression inputs automatically rejoin future runs.
    result['regression_replays'] = []
    if not args.replay:
        for path in sorted(args.regressions.glob('*.json')):
            saved = json.loads(path.read_text())
            if saved.get('mode') != args.mode:
                result['regression_replays'].append(dict(path=str(path), status='different_mode'))
                continue
            saved['profile'] = validate_saved_profile(saved['profile'], inventory, live=args.mode == 'live')
            if saved['target'][1] == 'forbidden_semantic_divergence':
                saved['reference_profile'] = validate_saved_profile(saved['reference_profile'], inventory, live=args.mode == 'live')
            replay_run = extra(saved['corpus'], saved['profile'], output / 'regression-replays' / path.stem,
                               f'{run_id}-regression-{path.stem[:12]}')
            if saved['target'][1] == 'forbidden_semantic_divergence':
                reference = extra(saved['corpus'], saved['reference_profile'],
                                  output / 'regression-replays' / f'{path.stem}-baseline',
                                  f'{run_id}-reference-{path.stem[:12]}')
                compare_reference(replay_run, reference)
            replay_status = ('incomplete' if replay_run['runner_status'] or any(c['missing'] for c in replay_run['cases']) else
                             'reproduced' if matches_signature(saved['target'], signatures(replay_run)) else 'resolved')
            result['regression_replays'].append(dict(path=str(path), status=replay_status, output=replay_run['output']))
            result['runs'].append(replay_run)
    handled = set()
    for run in result['runs']:
        for target in sorted(signatures(run)):
            if target in handled:
                continue
            handled.add(target)
            if remaining < 2:
                result['regressions'].append(dict(status='unconfirmed_budget_exhausted', target=target,
                                                  corpus=raw, profile=run['profile']))
                continue
            reduction_root = output / 'reductions' / stable_digest([run['profile']['id'], target])[:12]
            def replay(corpus, profile, attempt):
                replayed = extra(corpus, profile, reduction_root / str(attempt),
                                 f'{run_id}-reduce-{reduction_root.name}-{attempt}')
                if target[1] == 'forbidden_semantic_divergence' and not replayed.get('replay_stop_reason'):
                    reference = extra(corpus, profiles[0], reduction_root / f'{attempt}-baseline',
                                      f'{run_id}-base-{reduction_root.name}-{attempt}')
                    compare_reference(replayed, reference)
                    if reference.get('replay_stop_reason'):
                        replayed['replay_stop_reason'] = reference['replay_stop_reason']
                return replayed
            minimized = minimize_failure(run.get('corpus', raw), run['profile'], target, replay, budget=remaining)
            remaining -= len(minimized['attempts'])
            minimized['target'] = target
            if minimized['status'] == 'confirmed':
                identity = stable_digest({k: minimized[k] for k in ('corpus', 'profile', 'target')})
                path = args.regressions / f'{identity}.json'
                atomic_write_json(path, dict(minimized, schema_version=1, source_digest=inventory['source_digest'],
                                             mode=args.mode, origin=str(output), reference_profile=profiles[0]))
                minimized['path'] = str(path)
            result['regressions'].append(minimized)


def persist_manual_review(result, raw, output, source_digest):
    """Unresolved observations are durable, but never auto-replay regressions."""
    entries = []
    def retain(value):
        identity = stable_digest(value)
        path = output / 'manual-review' / f'{identity}.json'
        envelope = dict(value, schema_version=1, mode=result['mode'], source_digest=source_digest,
                        origin=str(output), automatic_retry=False, recorded_at=utc_now())
        atomic_write_json(path, envelope)
        entries.append(dict(path=str(path), kind=value['kind'], profile=value['profile']['id'],
                            case_id=value.get('case', {}).get('case_id'), reason=value['reason']))
    seen = set()
    runs = result['runs'] + result.get('confirmation_runs', [])
    for run in runs:
        for case in run['cases']:
            key = (run['output'], case['case_id'])
            if not case['missing'] or key in seen:
                continue
            seen.add(key)
            retained_case = {k: v for k, v in case.items() if k != 'payload'}
            retain(dict(kind='incomplete_observation', status='incomplete', case=retained_case,
                        corpus=run.get('corpus', raw), profile=run['profile'], evidence_path=run['output'],
                        reason='; '.join(case['missing']), live_budget=result.get('live_budget')))
    confirmed = {tuple(r['target']) for r in result['regressions'] if r['status'] == 'confirmed'}
    targets = set()
    for run in runs:
        for target in sorted(signatures(run)):
            if target in confirmed or target in targets:
                continue
            targets.add(target)
            retain(dict(kind='unconfirmed_failure', status='unconfirmed', target=target,
                        corpus=run.get('corpus', raw), profile=run['profile'], evidence_path=run['output'],
                        reason=result.get('confirmation_stop_reason', 'Confirmation incomplete or attempt budget exhausted.'),
                        live_budget=result.get('live_budget')))
    result['manual_review'] = entries


def import_live_evidence(source, output, raw, profiles, seed):
    """Explicit handoff preserves shadow IDs and ambiguous dispatch state."""
    source = source.resolve()
    if source == output:
        raise CorpusError('Use the original --output to resume in place; handoff needs a distinct destination.')
    with manifest_lock(source / 'run.json'):
        config = json.loads((source / 'run.json').read_text())
        if (config.get('mode') != 'live' or config.get('corpus') != raw
                or config.get('profiles') != profiles or config.get('seed') != seed):
            raise CorpusError('Live handoff must preserve the exact corpus, profiles and seed.')
        marker = output / 'live-handoff.json'
        expected = dict(source=str(source), source_digest=config['source_digest'], run_id=config['run_id'])
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise CorpusError('Live handoff source differs from the persisted handoff.')
            return config['run_id']
        for directory in [p['id'] for p in profiles] + ['regression-replays', 'reductions']:
            old, new = source / directory, output / directory
            if old.exists():
                if new.exists():
                    raise CorpusError('Partial handoff destination exists; preserve it for manual recovery.')
                shutil.copytree(old, new)
                for manifest_path in new.glob('sequence-*/manifest.json'):
                    manifest = load_manifest(manifest_path)
                    manifest['manifest_path'] = str(manifest_path)
                    manifest['corpus_path'] = str(manifest_path.parent / 'corpus.json')
                    atomic_write_json(manifest_path, manifest)
        # New budgets may replace only the old, unbounded implementation.
        # A hardened run's counters/deadlines never reset merely by detaching.
        for name in ('live-budget.json', 'progress.json', 'results.json', 'report.md'):
            old = source / name
            if old.exists():
                shutil.copy2(old, output / name)
        atomic_write_json(marker, expected)
        return config['run_id']


def launch_detached(argv, output):
    """Detach this same command, with durable handles and no inherited terminal."""
    with manifest_lock(output / 'process.json'):
        process_path = output / 'process.json'
        if process_path.exists():
            previous = json.loads(process_path.read_text())
            if (output / 'completion.json').exists():
                print(f'Detached run already finished: {process_path}; evidence was preserved.', flush=True)
                return 0
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                print(f'Detached controller already running (PID {previous["pid"]}): {process_path}', flush=True)
                return 0
        command_args = [value for value in argv if value != '--detach'] + ['--output', str(output)]
        job = output / 'detached-job.json'
        atomic_write_json(job, dict(argv=command_args, completion_path=str(output / 'completion.json')))
        log_path = output / 'stdout-stderr.log'
        with log_path.open('ab', buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--detached-job', str(job)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        metadata = dict(pid=process.pid, session_id=process.pid, detached=True, started_at=utc_now(),
                        stdout_path=str(log_path), stderr_path=str(log_path), argv=command_args,
                        run_manifest_path=str(output / 'run.json'), live_manifest_path=str(output / 'live/run.json'),
                        final_report_path=str(output / 'report.md'), final_live_report_path=str(output / 'live/report.md'),
                        results_path=str(output / 'results.json'), completion_path=str(output / 'completion.json'))
        atomic_write_json(process_path, metadata)
        print(f'Detached PID {process.pid}: {process_path}', flush=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fake', 'live'], default='fake')
    parser.add_argument('--detach', action='store_true', help='Run independently of the calling terminal/session; persist PID and log/report paths.')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'config/self_attack_corpus.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-url', default=os.environ.get('OLLMO_WEB_BASE', 'http://127.0.0.1:5001'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--profile-limit', type=int)
    parser.add_argument('--max-cycles', type=int, default=300)
    parser.add_argument('--minimize-budget', type=int, default=24)
    parser.add_argument('--profile-timeout', type=float, default=600,
                        help='Fake-mode seconds per isolated profile or reduction replay.')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=2,
                        help='Bounded isolated profile processes in parallel (results remain in seeded order).')
    parser.add_argument('--owner-timeout', type=float, default=600)
    parser.add_argument('--regressions', type=Path, default=ROOT / 'state/self_attack/regressions')
    parser.add_argument('--skip-owner-tests', action='store_true')
    parser.add_argument('--live-after-fake', action='store_true', help='Run bounded live Ghost observations only after a complete passing fake sweep.')
    parser.add_argument('--live-profile-limit', type=int, default=2)
    parser.add_argument('--live-high-risk', action='store_true',
                        help='Six diverse live profiles with ten targeted case executions across all five boundaries; fake coverage stays full.')
    parser.add_argument('--live-sequence-timeout', type=float, default=1200,
                        help='Live observation seconds per dependency sequence (default: 1200; also bounded by the main budget).')
    parser.add_argument('--live-main-budget', type=float, default=3600,
                        help='Hard seconds for all live preflight and main sweep work (default: 3600).')
    parser.add_argument('--live-confirmation-budget', type=float, default=900,
                        help='Hard total seconds for all live regression/confirmation work (default: 900).')
    parser.add_argument('--live-replay-timeout', type=float, default=300,
                        help='Hard seconds for one entire live profile replay (default: 300).')
    parser.add_argument('--live-attempt-cap', type=int, default=4,
                        help='Shared cap on extra live profile executions, including reference replays (default: 4).')
    parser.add_argument('--fake-evidence', type=Path, help='Passing fake results.json required before live execution.')
    parser.add_argument('--resume-live-from', type=Path,
                        help='Explicit live handoff retaining response IDs and dispatch state; also forwarded by --live-after-fake.')
    parser.add_argument('--replay', type=Path, help='Replay one saved regression envelope')
    parser.add_argument('--recheck', type=Path, help='Recheck retained captures without any runtime execution')
    parser.add_argument('--audit-convergence', type=Path,
                        help='Audit retained self-attack convergence evidence offline; never execute workloads.')
    parser.add_argument('--audit-production', type=Path,
                        help='Audit production response frames and retained state offline; never execute workloads.')
    parser.add_argument('--refresh-report', type=Path,
                        help='Update completed JSON/Markdown reporting only; do not execute or recheck the suite.')
    args = parser.parse_args(argv)
    if args.audit_production:
        if args.audit_convergence or args.live_after_fake or args.mode != 'fake' or args.replay or args.recheck or args.refresh_report:
            parser.error('--audit-production cannot be combined with other audit/execution modes.')
        output = (args.output or ROOT / 'state/self_attack' / f'production-convergence-{time.time_ns()}').resolve()
        if output == args.audit_production.resolve() or output.is_relative_to(args.audit_production.resolve()):
            parser.error('Production audit output must be outside the response-frame evidence directory.')
        output.mkdir(parents=True, exist_ok=True)
        if args.detach:
            return launch_detached(list(sys.argv[1:] if argv is None else argv), output)
        from scripts.self_attack_production import run_production_audit
        with manifest_lock(output / 'run.json'):
            return run_production_audit(args.audit_production, output)
    if args.audit_convergence:
        if args.live_after_fake or args.mode != 'fake' or args.replay or args.recheck or args.refresh_report:
            parser.error('--audit-convergence cannot be combined with execution/recheck/report modes.')
        output = (args.output or args.audit_convergence / f'convergence-audit-{time.time_ns()}').resolve()
        if output == args.audit_convergence.resolve():
            parser.error('Audit output must be separate from its evidence root.')
        output.mkdir(parents=True, exist_ok=True)
        if args.detach:
            return launch_detached(list(sys.argv[1:] if argv is None else argv), output)
        from scripts.self_attack_convergence import run_audit
        with manifest_lock(output / 'run.json'):
            return run_audit(args.audit_convergence, output)
    if args.refresh_report:
        return refresh_report(args.refresh_report)
    if args.live_after_fake and args.mode != 'fake':
        parser.error('--live-after-fake starts in fake mode.')
    if args.mode == 'live' and not args.fake_evidence:
        parser.error('Live execution requires --fake-evidence from a complete passing fake sweep.')
    if args.recheck:
        return recheck_captures(args.recheck.resolve(),
                               (args.output or args.recheck / 'rechecks' / str(time.time_ns())).resolve(),
                               discover_knobs(ROOT))
    seconds = (args.profile_timeout, args.owner_timeout, args.live_sequence_timeout,
               args.live_main_budget, args.live_confirmation_budget, args.live_replay_timeout)
    if (args.max_cycles < 1 or args.minimize_budget < 0 or args.live_attempt_cap < 0
            or any(not math.isfinite(v) or v <= 0 for v in seconds)
            or args.live_profile_limit < 1 or (args.profile_limit is not None and args.profile_limit < 1)):
        parser.error('Budgets must be finite and positive (attempt/minimize budgets may be zero).')
    if args.live_high_risk and (args.profile_limit if args.mode == 'live' else args.live_profile_limit) != 6:
        parser.error('--live-high-risk requires six live profiles.')
    output = (args.output or ROOT / 'state/self_attack' / f'run-{time.time_ns()}').resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        return launch_detached(list(sys.argv[1:] if argv is None else argv), output)
    with manifest_lock(output / 'run.json'):
        inventory = discover_knobs(ROOT)
        if args.mode == 'live':
            proof = json.loads(args.fake_evidence.read_text())
            proof_config = json.loads((args.fake_evidence.parent / 'run.json').read_text())
            if (proof.get('mode') != 'fake' or proof.get('verdict') != 'passed'
                    or proof_config.get('source_digest') != inventory['source_digest']):
                raise CorpusError('Live execution requires passing fake evidence for the current source.')
        atomic_write_json(output / 'controls.json', inventory)
        raw = json.loads(args.corpus.read_text())
        if args.mode == 'live' and raw != proof_config.get('corpus'):
            raise CorpusError('Live corpus differs from the passing fake corpus.')
        profiles = build_profiles(inventory, live=args.mode == 'live', seed=args.seed, pairwise=args.pairwise)
        total_profiles = len(profiles)
        if args.profile_limit:
            profiles = (diverse_live_profiles(profiles, args.profile_limit)
                        if args.mode == 'live' and args.live_high_risk else profiles[:args.profile_limit])
        if args.replay:
            saved = json.loads(args.replay.read_text())
            if saved.get('mode') != args.mode:
                raise CorpusError('Regression mode differs; select the original --mode explicitly.')
            raw, profiles = saved['corpus'], [validate_saved_profile(saved['profile'], inventory, live=args.mode == 'live')]
            if saved['target'][1] == 'forbidden_semantic_divergence':
                profiles.insert(0, validate_saved_profile(saved['reference_profile'], inventory, live=args.mode == 'live'))
            total_profiles = len(profiles)
        if args.mode == 'live' and raw != proof_config.get('corpus'):
            raise CorpusError('Live replay corpus differs from the passing fake corpus.')
        config = dict(corpus=raw, profiles=profiles, mode=args.mode, seed=args.seed,
                      budgets=dict(profile_timeout=args.profile_timeout, cycles=args.max_cycles, jobs=args.jobs,
                                   minimize_budget=args.minimize_budget, live_sequence_timeout=args.live_sequence_timeout,
                                   live_main_budget=args.live_main_budget, live_confirmation_budget=args.live_confirmation_budget,
                                   live_attempt_cap=args.live_attempt_cap, live_replay_timeout=args.live_replay_timeout),
                      source_digest=inventory['source_digest'])
        if args.mode == 'live' and args.live_high_risk:
            config['case_selection'] = high_risk_live_cases(raw, profiles)
        config_path = output / 'run.json'
        if config_path.exists():
            saved_config = json.loads(config_path.read_text())
            run_id = saved_config.pop('run_id', None)
            if saved_config != config or not run_id:
                raise CorpusError('Output directory belongs to a different source/corpus/profile run; choose a new --output.')
        else:
            run_id = (import_live_evidence(args.resume_live_from, output, raw, profiles, args.seed)
                      if args.mode == 'live' and args.resume_live_from else uuid.uuid4().hex)
        config['run_id'] = run_id
        atomic_write_json(config_path, config)
        atomic_write_json(output / 'controller.json', dict(pid=os.getpid(), session_id=os.getsid(0),
                          hangup_ignored=signal.getsignal(signal.SIGHUP) == signal.SIG_IGN,
                          mode=args.mode, manifest_path=str(config_path), report_path=str(output / 'report.md')))
        live_budget = None
        if args.mode == 'live':
            live_budget = LiveBudget(output, main_seconds=args.live_main_budget,
                                     confirmation_seconds=args.live_confirmation_budget,
                                     attempt_cap=args.live_attempt_cap, sequence_seconds=args.live_sequence_timeout,
                                     replay_seconds=args.live_replay_timeout)
            if (output / 'results.json').exists():
                previous = json.loads((output / 'results.json').read_text())
                print(f"Retained {previous['verdict'].upper()}: {output / 'report.md'}; no live work restarted.", flush=True)
                return {'passed': 0, 'failed': 1, 'incomplete': 2}[previous['verdict']]
        result = dict(schema_version=1, mode=args.mode, started_at=utc_now(), runs=[], regressions=[], owner_tests={},
                      authority='deterministic_fake_runtime' if args.mode == 'fake' else 'live_observation_only',
                      coverage=dict(design='pairwise plus boundaries' if args.pairwise else 'every finite value plus four seeded mixed profiles',
                                    inventory_only=sum(k['disposition'] != 'sweep' for k in inventory['controls']),
                                    profiles_available=total_profiles, profiles_selected=len(profiles)))
        if config.get('case_selection'):
            result['coverage'].update(case_selection=config['case_selection'],
                design='six representative profiles; maximum control-value coverage, then interactions; dependency-closed high-risk case subsets',
                case_executions_selected=sum(map(len, config['case_selection'].values())),
                case_executions_available=len(raw['cases']) * len(profiles))
        if args.mode == 'live':
            result['fake_evidence'] = str(args.fake_evidence.resolve())
            result['owner_tests'] = proof['owner_tests']
        elif not args.skip_owner_tests:
            result['owner_tests'] = run_owner_tests(output, timeout=args.owner_timeout)
        if live_budget and (output / 'progress.json').exists():
            result = json.loads((output / 'progress.json').read_text())
        if args.mode == 'live':
            result['fake_conformance_verdict'] = proof['verdict']
            result['selected_live_cases'] = config.get('case_selection') or {
                p['id']: [c['case_id'] for c in raw['cases']] for p in profiles}
        atomic_write_json(output / 'progress.json', result)
        try:
            if not live_budget or live_budget.state['stage'] == 'main':
                with live_budget.deadline() if live_budget else nullcontext():
                    execute_sweep(args, raw, profiles, output, run_id, result, live_budget)
        except LiveBudgetExpired as exc:
            for profile in profiles[len(result['runs']):]:
                corpus = sweep_corpus(raw, profile, result)
                run = unfinished_profile(corpus, profile, output / profile['id'], str(exc))
                if corpus != raw:
                    run['corpus'] = corpus
                result['runs'].append(run)
        atomic_write_json(output / 'progress.json', result)
        if live_budget:
            live_budget.begin_confirmation()
        try:
            if not live_budget or result.get('live_preflight_passed'):
                with live_budget.deadline() if live_budget else nullcontext():
                    confirm_failures(args, raw, profiles, output, run_id, result, inventory, live_budget)
            else:
                result['confirmation_stop_reason'] = 'live_preflight_not_complete'
        except LiveBudgetExpired as exc:
            result['confirmation_stop_reason'] = str(exc)
        if live_budget:
            result['live_budget'] = deepcopy(live_budget.state)
            persist_manual_review(result, raw, output, inventory['source_digest'])
        failures = any(signatures(r) or any(c['status'] == 'failed' for c in r.get('probes', {}).get('checks', []))
                       for r in result['runs'] + result.get('confirmation_runs', [])) or any(
            r['status'] == 'failed' for r in result['owner_tests'].values())
        incomplete = ((args.skip_owner_tests and args.mode == 'fake') or len(profiles) < total_profiles or
                      bool(result['coverage'].get('case_selection')) or
                      any(r['status'] == 'incomplete' for r in result['owner_tests'].values()) or
                      bool(result.get('confirmation_stop_reason')) or
                      any(r['runner_status'] or any(c['missing'] for c in r['cases'])
                          for r in result['runs'] + result.get('confirmation_runs', [])))
        result['verdict'] = 'failed' if failures else 'incomplete' if incomplete else 'passed'
        sweep = result['runs'][:len(profiles)]
        result['coverage']['profiles_completed'] = sum(not r['runner_status'] and not any(c['missing'] for c in r['cases']) for r in sweep)
        result['coverage']['boundaries'] = {
            key: dict(expected=sum(c['category'] == key for p in profiles for c in sweep_corpus(raw, p, result)['cases']),
                      captured=sum(bool(c.get('capture_path')) and not c['missing'] for r in sweep for c in r['cases'] if c['category'] == key),
                      incomplete=sum(bool(c['missing']) for r in sweep for c in r['cases'] if c['category'] == key),
                      failures=sum(len(c['findings']) for r in sweep for c in r['cases'] if c['category'] == key))
            for key, _ in PRIORITIES}
        result['finished_at'] = utc_now()
        result['scoped_verdicts'] = verdict_scopes(result)
        # Full payloads already live in captures; keep the machine report manageable.
        for run in result['runs'] + result.get('confirmation_runs', []):
            for case in run['cases']:
                case.pop('payload', None)
        atomic_write_json(output / 'results.json', result)
        (output / 'report.md').write_text(render_report(result), encoding='utf-8')
        print(f"{result['verdict'].upper()}: {output / 'report.md'}", flush=True)
        if args.live_after_fake:
            if result['verdict'] != 'passed':
                result['live'] = dict(status='not_run', reason='Fake coverage/invariants are not complete and passing.')
            else:
                command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'live',
                           '--fake-evidence', str(output / 'results.json'), '--corpus', str(args.corpus),
                           '--output', str(output / 'live'), '--base-url', args.base_url,
                           '--profile-limit', str(args.live_profile_limit), '--seed', str(args.seed),
                           '--profile-timeout', str(args.profile_timeout), '--max-cycles', str(args.max_cycles),
                           '--live-sequence-timeout', str(args.live_sequence_timeout),
                           '--live-main-budget', str(args.live_main_budget),
                           '--live-confirmation-budget', str(args.live_confirmation_budget),
                           '--live-replay-timeout', str(args.live_replay_timeout),
                           '--live-attempt-cap', str(args.live_attempt_cap),
                           '--minimize-budget', str(args.minimize_budget), '--regressions', str(args.regressions)]
                if args.resume_live_from:
                    command += ['--resume-live-from', str(args.resume_live_from.resolve())]
                if args.live_high_risk:
                    command += ['--live-high-risk']
                completed = subprocess.run(command, cwd=ROOT)
                live_path = output / 'live/results.json'
                live_result = json.loads(live_path.read_text()) if live_path.exists() else {}
                result['live'] = dict(status=live_result.get('verdict', 'incomplete'), exit_code=completed.returncode,
                                      results=str(live_path), authority='live_observation_only',
                                      scoped_verdicts=live_result.get('scoped_verdicts', {}))
            result['overall_verdict'] = (result['verdict'] if result['verdict'] != 'passed' else
                                         result['live']['status'] if result['live']['status'] in {'passed', 'failed'} else 'incomplete')
            result['scoped_verdicts'] = verdict_scopes(result)
            atomic_write_json(output / 'results.json', result)
            (output / 'report.md').write_text(render_report(result), encoding='utf-8')
        return {'passed': 0, 'failed': 1, 'incomplete': 2}[result.get('overall_verdict', result['verdict'])]


if __name__ == '__main__':
    try:
        if len(sys.argv) == 3 and sys.argv[1] == '--detached-job':
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            job = json.loads(Path(sys.argv[2]).read_text())
            atomic_write_json(Path(job['completion_path']).with_name('controller.json'), dict(
                pid=os.getpid(), parent_pid=os.getppid(), session_id=os.getsid(0),
                process_group_id=os.getpgid(0), is_session_leader=os.getsid(0) == os.getpid(),
                is_process_group_leader=os.getpgid(0) == os.getpid(),
                hangup_ignored=signal.getsignal(signal.SIGHUP) == signal.SIG_IGN,
                stdin_is_terminal=sys.stdin.isatty(), observed_at=utc_now()))
            exit_code = 2
            error = None
            try:
                exit_code = main(job['argv'])
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                atomic_write_json(Path(job['completion_path']), dict(exit_code=exit_code, error=error, finished_at=utc_now()))
            raise SystemExit(exit_code)
        if len(sys.argv) == 3 and sys.argv[1] == '--worker':
            job = json.loads(Path(sys.argv[2]).read_text())
            job['output'] = Path(job['output'])
            value = run_profile(**job)
            atomic_write_json(job['output'] / 'worker-result.json', value)
            raise SystemExit(0)
        raise SystemExit(main())
    except (CorpusError, ValueError, OSError, LiveBudgetExpired) as exc:
        print(f'Self-attack incomplete: {exc}', file=sys.stderr)
        raise SystemExit(2)
