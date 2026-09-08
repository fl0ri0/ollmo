"""Source-backed control discovery for diagnostics, never runtime authority."""
from __future__ import annotations

import ast
from itertools import combinations, product
from pathlib import Path
import random

from ollmo_g import request_meta
from ollmo_g.ghost_mode_compat import SUPPORTED_LEGACY_MODES
from ollmo_g.semantic_roles import build_semantic_role_catalog
from ollmo_services import enforced_policy, graph_rebase, graph_repair
from ollmo_server import multi_materialization_runtime as materialization
from scripts.run_graph_rebase_shadow_corpus import CorpusError, stable_digest


def discover_knobs(root: Path) -> dict:
    """Inventory every literal control in active code; bind known domains to owners.

    Unbound controls are visible exclusions, never guessed setters. Credentials
    and paths are inventoried by name only; environment values are never read.
    """
    controls = []

    def add(name, scope, values, source, default=None):
        controls.append(dict(name=name, scope=scope, values=list(values),
                             default=default, source=source, disposition='sweep'))

    add('ghost_mode', 'request', SUPPORTED_LEGACY_MODES,
        'ollmo_g/ghost_mode_compat.py:SUPPORTED_LEGACY_MODES')
    for name, default in request_meta.DEFAULT_DEVELOPER_FLAGS.items():
        if isinstance(default, bool):
            values = [False, True]
        elif name == 'planner_timeout_ms':
            values = [request_meta.MIN_PLANNER_TIMEOUT_MS, request_meta.MAX_PLANNER_TIMEOUT_MS]
        elif name == 'accepted_learning_authority':
            values = sorted(request_meta.ACCEPTED_LEARNING_AUTHORITY_LEVELS)
        else:
            controls.append(dict(name=f'developer_flags.{name}', scope='request',
                                 disposition='unbound', source='ollmo_g/request_meta.py'))
            continue
        add(f'developer_flags.{name}', 'request', values, 'ollmo_g/request_meta.py', default)
    for module, prefix in ((graph_repair, 'GRAPH_REPAIR'), (graph_rebase, 'GRAPH_REBASE')):
        add(getattr(module, f'{prefix}_AUTONOMY_ENV'), 'environment',
            sorted(getattr(module, f'_{prefix}_AUTONOMY_LEVELS')) + ['invalid_self_attack_value'],
            f'{module.__name__.replace(".", "/")}.py',
            getattr(module, f'{prefix}_AUTONOMY_PRODUCT_DEFAULT'))
    add('OLLMO_APPLY_ENFORCED_POLICY', 'environment',
        sorted(enforced_policy._ENFORCED_POLICY_MODES) + ['invalid_self_attack_value'],
        'ollmo_services/enforced_policy.py')
    add('OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS', 'environment',
        [materialization.MIN_MAX_PARALLEL_WORKERS, materialization.DEFAULT_MAX_PARALLEL_WORKERS,
         materialization.MAX_MAX_PARALLEL_WORKERS], 'ollmo_server/multi_materialization_runtime.py')
    known = {item['name'] for item in controls}
    literals = {}
    paths = [root / 'ollmo_webserver.py'] + [p
        for directory in ('ollmo_g', 'ollmo_core', 'ollmo_services', 'ollmo_server', 'ollmo_runtime', 'helpers')
        for p in sorted((root / directory).rglob('*.py'))]
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
                if name.startswith('OLLMO_') and name.replace('_', '').isalnum() and len(name) < 120:
                    literals.setdefault(name, []).append(f'{path.relative_to(root)}:{node.lineno}')
    for name, sources in sorted(literals.items()):
        if name not in known:
            controls.append(dict(name=name, scope='environment', source=sources,
                                 disposition='inventory_only',
                                 reason='No reviewed finite diagnostic setter; not changed.'))
    diagnostic_sources = [root / 'scripts' / name for name in
                          ('ollmo_self_attack.py', 'self_attack_checks.py', 'self_attack_knobs.py', 'run_graph_rebase_shadow_corpus.py')]
    diagnostic_sources += list((root / 'tests/fake_backends').glob('*.py'))
    files = {str(p.relative_to(root)): stable_digest(p.read_text(encoding='utf-8')) for p in paths + diagnostic_sources}
    return dict(controls=controls, semantic_roles=build_semantic_role_catalog(),
                advisory_surfaces=['controlled_attention_review', 'aspiration_review',
                                   'commitment_review', 'semantic_decision_review'],
                source_digest=stable_digest(files), source_files=files)


def build_profiles(inventory: dict, *, live=False, seed=0, pairwise=False) -> list[dict]:
    knobs = [k for k in inventory['controls'] if k['disposition'] == 'sweep'
             and (not live or k['scope'] == 'request')]
    settings = [{}]
    # Every discovered finite value is swept; mixed coverage is an explicit option.
    settings.extend({k['name']: v} for k in knobs for v in k['values'])
    if pairwise:
        uncovered = {(a['name'], str(va), b['name'], str(vb))
                     for a, b in combinations(knobs, 2)
                     for va, vb in product(a['values'], b['values'])}
        rng = random.Random(seed)
        while uncovered:
            candidates = [{k['name']: rng.choice(k['values']) for k in knobs} for _ in range(80)]
            first = min(uncovered)
            forced = dict(candidates[0])
            for name, value in ((first[0], first[1]), (first[2], first[3])):
                forced[name] = next(v for k in knobs if k['name'] == name for v in k['values'] if str(v) == value)
            candidates.append(forced)
            def pairs(s):
                return {(a['name'], str(s[a['name']]), b['name'], str(s[b['name']]))
                        for a, b in combinations(knobs, 2)}
            chosen = max(candidates, key=lambda s: len(pairs(s) & uncovered))
            settings.append(chosen)
            uncovered -= pairs(chosen)
    else:
        # Include deterministic interactions, without claiming pairwise completeness.
        rng = random.Random(seed)
        settings.extend({k['name']: rng.choice(k['values']) for k in knobs} for _ in range(4))
    profiles = []
    seen = set()
    scopes = {k['name']: k['scope'] for k in knobs}
    for setting in settings:
        digest = stable_digest(setting)
        if digest in seen:
            continue
        seen.add(digest)
        request, environment = {}, {}
        for key, value in setting.items():
            if scopes[key] == 'environment':
                environment[key] = str(value)
            elif key.startswith('developer_flags.'):
                request.setdefault('developer_flags', {})[key.split('.', 1)[1]] = value
            else:
                request[key] = value
        profiles.append(dict(id='baseline' if not setting else digest[:12], settings=setting,
                             request=request, environment=environment))
    return profiles


def validate_saved_profile(profile, inventory, *, live=False):
    """Rebind persisted diagnostic settings; stored request bodies are not authority."""
    settings = profile.get('settings')
    if not isinstance(settings, dict):
        raise CorpusError('Saved profile has no settings object.')
    known = {k['name']: k for k in inventory['controls'] if k['disposition'] == 'sweep'}
    request, environment = {}, {}
    for name, value in settings.items():
        knob = known.get(name)
        if not knob or not any(type(v) is type(value) and v == value for v in knob['values']):
            raise CorpusError(f'Saved control is outside the current finite domain: {name}')
        if knob['scope'] == 'environment':
            if live:
                raise CorpusError('An environment profile cannot be replayed against a live control plane.')
            environment[name] = str(value)
        elif name.startswith('developer_flags.'):
            request.setdefault('developer_flags', {})[name.split('.', 1)[1]] = value
        else:
            request[name] = value
    return dict(id='baseline' if not settings else stable_digest(settings)[:12],
                settings=settings, request=request, environment=environment)


def diverse_live_profiles(profiles, limit):
    """Select from the discovered matrix, preserving baseline and repair.

    Maximize covered control values, then interactions; catalog order breaks
    ties reproducibly. The representative gate has six slots, not a new matrix.
    """
    if limit >= len(profiles):
        return profiles
    required = profiles[:2]
    if limit <= 2:
        return required[:limit]
    if limit != 6:
        raise CorpusError('The representative live gate requires exactly six profiles.')
    def score(selected):
        values, pairs = set(), set()
        for profile in selected:
            entries = sorted((k, repr(v)) for k, v in profile['settings'].items())
            values.update(entries)
            pairs.update(combinations(entries, 2))
        return len(values), len(pairs)
    chosen = max(combinations(profiles[2:], limit - len(required)),
                 key=lambda remaining: score([*required, *remaining]))
    return [*required, *chosen]
