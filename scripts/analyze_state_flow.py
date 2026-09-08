"""Analyze only captured state-flow summaries; never read or mutate runtime truth."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

CONTAINERS = {'responses_request.handle_responses_request', 'late_fill.complete_response_late_fill',
              'late_fill.worker', 'multi_materialization.execute_materialization_branches',
              'multi_materialization.worker', 'multi_materialization.execution'}


def union(intervals):
    merged = []
    for start, end in sorted((a, b) for a, b in intervals if b >= a):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def analyze(records, response_id):
    scopes = {r['scope_id'] for r in records if r.get('response_id') == response_id}
    population = [r for r in records if r.get('scope_id') in scopes]
    seen = {}; conflicts = []
    for record in population:
        if record.get('record_kind') != 'transition': continue
        key = record['transition_id']
        if key in seen and seen[key] != record: conflicts.append(key)
        seen[key] = record
    transitions = sorted(seen.values(), key=lambda r: (r.get('process_boot_id', ''), r['start_monotonic_ns']))
    counts = Counter(r['owner'] for r in transitions)
    summaries = {}
    for r in population:
        if r.get('record_kind') == 'scope_summary':
            if r['scope_id'] not in summaries or r['monotonic_ns'] > summaries[r['scope_id']]['monotonic_ns']:
                summaries[r['scope_id']] = r
    boots = defaultdict(list)
    for r in transitions: boots[r['process_boot_id']].append(r)
    timing = {}; rows = []; top_blocks = []; versions = defaultdict(Counter)
    for boot, items in boots.items():
        start = min(r['start_monotonic_ns'] for r in items)
        end = max(r['end_monotonic_ns'] for r in items)
        work = [r for r in items if r['owner'] not in CONTAINERS]
        covered = duration((r['start_monotonic_ns'], r['end_monotonic_ns']) for r in work)
        finalizers = [r for r in items if r['owner'] == 'response_frame.finalize']
        timing[boot] = {'start_monotonic_ns': start, 'end_monotonic_ns': end,
                       'observed_window_ns': end-start, 'measured_owner_union_ns': covered,
                       'unknown_rest_ns': end-start-covered,
                       'elapsed_coverage_fraction': covered/(end-start) if end > start else None,
                       'critical_path_coverage': None,
                       'finalizer_inclusive_sum_ns': sum(r['inclusive_ns'] for r in finalizers),
                       'finalizer_interval_union_ns': duration((r['start_monotonic_ns'], r['end_monotonic_ns']) for r in finalizers)}
        boundaries = sorted({start, end, *(r['start_monotonic_ns'] for r in work), *(r['end_monotonic_ns'] for r in work)})
        blocks = []
        for a, b in zip(boundaries, boundaries[1:]):
            active = [r for r in work if r['start_monotonic_ns'] <= a and r['end_monotonic_ns'] >= b]
            # Keep outer measured owners; simultaneous work on other threads is
            # explicitly overlap, never added as separate elapsed blocks.
            outer = [r for r in active if not any(
                q['transition_id'] != r['transition_id'] and q['thread_id'] == r['thread_id']
                and q['start_monotonic_ns'] <= r['start_monotonic_ns']
                and q['end_monotonic_ns'] >= r['end_monotonic_ns']
                and q['inclusive_ns'] > r['inclusive_ns'] for q in active)]
            ids = tuple(sorted(r['transition_id'] for r in outer))
            owners = sorted({r['owner'] for r in outer}) or ['UNKNOWN']
            if blocks and blocks[-1]['transition_ids'] == ids:
                blocks[-1]['end_monotonic_ns'] = b
                blocks[-1]['duration_ns'] += b-a
            else:
                blocks.append(dict(process_boot_id=boot, start_monotonic_ns=a, end_monotonic_ns=b,
                                   duration_ns=b-a, owners=owners, transition_ids=ids,
                                   simultaneous_threads=len({r['thread_id'] for r in outer})))
        top_blocks.extend(blocks)
        for r in items:
            children = [(max(r['start_monotonic_ns'], c['start_monotonic_ns']), min(r['end_monotonic_ns'], c['end_monotonic_ns']))
                        for c in items if c.get('parent_transition_id') == r['transition_id'] and c['thread_id'] == r['thread_id']]
            before, after = r.get('source') or {}, r.get('target') or {}
            row = dict(r, ordinal=len(rows)+1, relative_start_s=(r['start_monotonic_ns']-start)/1e9,
                       relative_end_s=(r['end_monotonic_ns']-start)/1e9,
                       same_thread_child_union_ns=duration(children),
                       exclusive_or_unattributed_rest_ns=max(0,r['inclusive_ns']-duration(children)))
            rows.append(row)
            identity = (after.get('frame_id') or before.get('frame_id'), after.get('frame_sequence') or before.get('frame_sequence'))
            if identity[0]:
                key = f'{identity[0]} / sequence={identity[1]}'
                versions[key][r['owner']] += 1
    work_totals = Counter()
    for r in transitions: work_totals.update(r.get('work') or {})
    scope_cas = set(); identity_complete = True
    for r in summaries.values():
        scope_cas.update((r.get('identities') or {}).get('produced_cas') or [])
        identity_complete &= not bool(r.get('identity_drops'))
    return {'schema': 'ollmo.state_flow_analysis.v1', 'response_id': response_id,
            'transitions': rows, 'owner_counts': dict(counts), 'selected_work_counters': dict(work_totals),
            'turn_summary': {'full_finalizers_completed': counts['response_frame.finalize'],
              'canonical_load_invocations': counts['response_frame.canonical_load'],
              'full_reconstruction_invocations': counts['response_frame.canonical_reconstruction'],
              'bounded_observation_loads': counts['response_frame.bounded_observation_load'],
              'response_frame_constructions': counts['response_frame.build'],
              'working_frame_constructions': counts['working_frame.build'],
              'compactions': counts['response_frame.compact'],
              'snapshot_roots': work_totals.get('snapshot_root_calls'),
              'unique_produced_cas_identities': len(scope_cas) if summaries and identity_complete else None,
              'actual_cas_writes': work_totals.get('cas_actual_writes'),
              'index_full_map_digest_passes': work_totals.get('index_full_map_digest_passes'),
              'index_full_map_rebuilds': None,
              'index_writes': work_totals.get('index_writes'),
              'readiness_passes': counts['readiness.pass']},
            'frame_bound_owner_counts': {k: dict(v) for k,v in versions.items()},
            'timing_by_process_boot': timing,
            'top_10_nonoverlapping_outer_blocks': sorted(top_blocks,key=lambda r:r['duration_ns'],reverse=True)[:10],
            'coverage': {'scope_summaries': list(summaries.values()),'conflicting_transition_ids': conflicts,
                         'dropped_records': sum(s.get('dropped_count',0) for s in summaries.values()) if summaries else None,
                         'delivery_failures': sum(s.get('delivery_failures',0) for s in summaries.values()) if summaries else None,
                         'byte_coverage': 'partial_selected_existing_operations',
                         'unfinished_spans': 'UNKNOWN_without_worker_and_request_end_evidence',
                         'version_coverage': 'frame_binding_only_mutable_live_versions_not_proven'},
            'amplification': {'total_processed_over_final_canonical_bytes': None,
                              'reconstructed_over_unique_authoritative_bytes': None,
                              'reason': 'complete byte coverage and unique authoritative byte set not measured'},
            'authority_analysis_rule': 'Equal identities never establish an unnecessary check. Reconstruction necessity remains UNKNOWN without an owner/authority equivalence proof.'}


def markdown(data):
    lines = ['# State-flow analysis', '', f"Response: `{data['response_id']}`", '',
             'Times are inclusive monotonic intervals; UNKNOWN is not zero. Byte totals cover selected operations only.', '',
             '| Ordinal | Owner | Source → Target | Start s | End s | Duration s | Size in/out | Evidence | Authority | Full reconstruction | Persist / reload |',
             '|---|---|---|---:|---:|---:|---|---|---|---|---|']
    def val(v): return 'UNKNOWN' if v is None else str(v)
    for r in data['transitions']:
        source,target=r.get('source') or {},r.get('target') or {}
        lines.append(f"| {r['ordinal']} | {r['owner']} | {r['source_representation']} → {r['target_representation']} | {r['relative_start_s']:.6f} | {r['relative_end_s']:.6f} | {r['inclusive_ns']/1e9:.6f} | {val(source.get('serialized_bytes'))} / {val(target.get('serialized_bytes'))} | {val(r.get('new_evidence'))} | {val(r.get('new_authority_boundary'))} | {val(r.get('full_reconstruction'))} | {val(r.get('disk_write'))} / {val(r.get('full_hydration'))} |")
    lines += ['', 'Machine report includes source/target frame bindings, parent IDs, selected byte/work counts, per-frame owner counts, interval unions and non-overlapping outer blocks.', '',
              'Review necessity and reconstruction necessity are separate questions. This report makes no optimization or avoidability claim.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--response-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    paths=sorted(args.input.glob('*.jsonl')) if args.input.is_dir() else [args.input]
    records=[json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    result=analyze(records,args.response_id)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'state-flow-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'state-flow-table.md').write_text(markdown(result))


if __name__=='__main__':main()
