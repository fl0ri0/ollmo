# Causal observations for convergence analysis

Knobs may change strategy, never truth. Every semantic movement should be
explainable by the change that made it newly relevant. Missing causal evidence
is `unknown`, never an inferred model call, unnecessary retry or false wait.

## Existing owners and new observations

`scripts/ollmo_run_monitor.py` is the canonical monitor. A development checkout
may also retain `state/ollmo_run_monitor/monitor_once.py` as a compatibility
entry point; the mutable `state/` tree is not shipped or required. The monitor is an observer, not execution
authority. There is no new profiler, scheduler, log store or dependency.

`ollmo_services/events.py` extends the existing unified event path. A scoped
response request or Late Fill worker records `action=causal_observation` in the
existing event log. Its `causal_event.schema` is `ollmo.causal_event.v1`.
The outer unified event ID identifies the transport record; the inner
`event_id` identifies the causal witness. Joins use the latter. A completed
invocation retains its `invocation_id`, `start_event_id`, parent invocation,
trigger event, exact target, monotonic interval, process ID and process-boot
identity. Scope IDs distinguish separate executions. Process-boot identity
combines the process PID and an import-time random token; it is not a machine
OS boot identifier and cannot establish timing order across processes.

The request's response ID may not exist during routing. Bounded observations
are buffered until the existing finalizer supplies it. An early failure that
never receives a response ID remains unbound. No timestamp-proximity join is
performed. Existing worker/thread submission points carry observation context;
this does not change worker counts, scheduling or lock policy.

| Owner | Additions | Meaning / limitations |
| --- | --- | --- |
| Responses request handler, Ghost route owner | actual request/routing spans and lineage | A routing invocation does not establish a model call. |
| Decision-contract builders | rebuild spans for aspiration, commitment, attention, doubt/quality, decisions; per-target frame and lens selection witnesses | `read_model_invocation` is a deterministic rebuild. It is never counted as model execution. Selected lens/transition and exact candidate/obligation/branch/phase references come from the owner. |
| Candidate promotion review | selected candidate/contract/policy input digests and exact per-candidate cause, outcome, contract reference and authority source | Records the existing promotion review. Aspiration and lenses grant no authority. |
| Backend chat transport | actual non-streaming model-call span and input-field digests | Provider arguments are hashed, not copied. Hidden timeout/normalization inputs mean coverage is partial. |
| Materialization batch and branch owners | batch/branch invocation IDs, preparation-ready, submission/dequeue, exact phase, same-instance wait and release linkage | Submission time is immediately before the existing submit call. This is not proof that all semantic gates have passed. |
| Late Fill | exact defensive execution-gate inputs/rule; availability-poll identities; branch settlement deltas; stale-result disposition; branch/phase/attempt on two automatic repair requeue events | Availability poll expiry is not route readiness. Stale disposition records do not yet bind every provider result ID. |
| Repair/rebase validators and structural/global Closure | actual validator/review calls and bounded judgment references | Full changed-input provenance and application/consumption lineage remain partial. Existing authoritative proposal/review/graph identities remain canonical. |
| Semantic verdict freeze gate | relevant verdict/schema/criteria/evidence identity and mandatory validation rule | A repeat under the same exact target and inputs may be classified as defensive. No waiver of this gate is implied. |
| Finalizer/frame persistence | existing inclusive step timers enriched with invocation/process identity and nested operation totals | These timers observe the current owner path; they do not control work or durability. |

## Late Fill instrumentation ownership

`ollmo_server/late_fill_runtime.py` retains semantic decisions and the exact
start-check, worker/submission, lookup, callback, publication, branch-settlement,
retry and availability anchors. `ollmo_services/late_fill_telemetry.py` supplies
an explicitly bound `LateFillTrace` for event metadata, wait identities,
callback targets and post-wave timing schemas. It returns diagnostic values;
Late Fill still attaches them and owns every runtime mutation and publication.
The import-time observation adapters configure the same existing wrappers in
the same order; they add no interception, scheduler, store or authority.

IDs, parent links, process-boot identity, clocks, budgets/reservations, redaction,
drop accounting and persistence remain in the existing events/state-flow services.
The exact execution-gate input selector stays beside its semantic owner to retain
control precedence and reader evaluation. State-flow hooks and the sibling
executor's preparation/lock/result/callback-drain anchors are unchanged.

Failure behavior retains its existing boundaries: causal/transition/state-flow
sinks fail open; inline metadata preparation and the legacy post-wave timing log
are not universally guarded. In particular, a legacy timing-log exception still
enters the existing Late Fill worker error handler. This structural extraction
does not silently broaden fail-open guarantees or change runtime errors.

## Identity, input coverage and bounds

`effective_input_hash` hashes an owner's explicitly selected input-field
digests. `input_field_hashes` and `changed_relevant_inputs` expose the changed
dimensions without duplicating prompts, graph bodies or artifacts. They do not
hash a response snapshot or reuse an output hash as input identity.
`judgment_hash` is separately a bounded result/judgment identity.

Only the exact execution gate and structured freeze validator currently claim
complete selected-input coverage. Lens selection excludes unrelated projection
fields but remains partial because role-catalog changes are not completely
bound. Other owner selections are deliberately partial or absent. Parent call
entry establishes call-stack lineage; it does not by itself explain why a
semantic change became newly relevant. `relevance_cause=unknown` preserves that
distinction. Concurrent calls do not acquire a fabricated previous-call order.

There are at most 1,024 ordinary causal records per scope, with a 32-record
reserve for finalizer start/completion witnesses. The captured diagnostic tail
contains at most 24 records. Each record is bounded to 32 KiB; transport limits
on strings and collections are explicit through `coverage_incomplete`, which
invalidates complete-input claims. Overflow reports dropped counts and a
coverage-gap event. These are observation limits, not execution limits.
A tail is never a complete interval. Failure of an observation or event sink
does not fail an owner operation or suppress its original exception.

`runtime.developer_diagnostics.causal_telemetry` carries that optional bounded
tail and coverage metadata. Frame persistence gives it a separate existing SHA
sidecar. The canonical sidecar writer, normalization, parent-manifest inheritance,
ledger append/fsync, index protocol and authorized hydration remain in use.
No historical records are rewritten. Start records are immutable: later timer
accumulation appears in the separate completion witness.

An event completing persistence cannot be in the frame whose write it measures
without another write. Completion is therefore retained in the unified event
log and returned diagnostics. The frozen frame is not rewritten for telemetry.

## Persistence timing interpretation

`response_frame_finalize_timing.steps` retains its existing meaning. Optional
`operations` adds counts, elapsed nanoseconds, failures, and a role for actual
implementation operations:

- parent index lookup, indexed row read and fallback ledger scan;
- frame identity serialization and hashing;
- snapshot media/timestamp normalization, recursive split, serialization,
  content addressing, existing-blob integrity/dedupe and canonical writes;
- ledger serialization and append/flush/fsync;
- recovery-index construction and atomic write;
- file flush/fsync, atomic replacement and directory fsync, annotated with the
  enclosing canonical or derived role;
- readiness projection, epoch verification, selection, hydration, durable
  projection and evidence-registry append;
- response-frame construction and output hoisting/projection.

These are **inclusive** nested timers. Recursive snapshot splitting, blob
verification and atomic writes overlap their enclosing compaction/finalizer
timers. Do not sum them as elapsed response time or subtract all of them from
the parent. No exclusive CPU attribution or recoverable-time estimate is made.
Invocation duration includes nested observation overhead; it is not a CPU
profile. Canonical SHA sidecars are durable runtime truth, not secondary
non-authoritative copies. The recovery index is derived acceleration;
readiness retention is secondary evidence, not primary frame-append authority.
Readiness still runs synchronously after successful CAS/Ledger/Index persistence
and before finalizer return. It is outside the canonical durable-completion
boundary and its failure does not roll back the frame. A finalizer invocation,
a durable append and response delivery are separate observations; see
[Finalization](RESPONSES_CONTRACT.md#finalization-and-durable-completion).
Private preparation reuse can reduce transformation counts while all current
authority/integrity gates remain active; neither repeated checks nor unchanged
output hashes prove redundancy.

## Consumers and conservative classification

The existing capture format remains valid. Full/debug payload captures retain
the optional diagnostics; the existing conformance debug summary exposes
coverage metadata without copying the event tail. The canonical monitor reads
the same optional observations and reports model calls separately from rebuilds.

`scripts/self_attack_convergence.py` adds `causal_analysis`; the production
analyzer also joins exact response-bound unified events. Historical timing
witnesses and the new causal population remain separate to avoid counting one
call twice. Exact IDs deduplicate copied captures; conflicting IDs cannot prove
causality. Optional operation decomposition is exposed in the machine report.

The existing nine classifications remain: `necessary_repeat`,
`defensive_repeat`, `redundant_repeat`, `necessary_wait`, `false_wait`,
`missed_wakeup`, `over_trigger`, `avoidable_serialization`, and `unknown`.
Complete owner inputs and a recorded relevant change may justify a necessary
repeat. Unchanged complete inputs plus an explicit mandatory owner rule may
justify a defensive repeat. Redundancy still requires complete interval,
authority and negative downstream-role proof; this instrumentation does not
manufacture those facts. A necessary mutex wait requires the exact wait,
different holder, release boundary, same instance/process/scope and ordered
monotonic witnesses. Its classification covers only that mutex predicate.

Still unresolved: complete semantic relevance triggers and catalog/config
versions; all dependency predicates and producer versions; result-to-consumer
and repair/rebase application causality; complete interval proof for negative
findings; streaming chat and all non-chat provider dispatches; cross-process
restart/stream worker causality; exact physical lock-release instant; exclusive
CPU/serialization attribution. These remain explicit unknowns. Historical
identity-free and partially instrumented evidence remains readable.

## Deterministic validation and fresh evidence

`tests/test_causal_telemetry.py` covers actual calls versus projections, exact
input selection, defensive repeats, SHA/lineage recovery, queue/mutex handoff,
availability identity, model-call context, overflow, observation failure and
unchanged canonical persistence bytes. Run it with the affected monitor,
convergence, frame, semantic, API and fake-backend E2E suites. Tests use temporary
roots and must not write or scan the production ledger.

The running webserver uses its loaded code; checkout instrumentation changes
require an authorized restart to take effect. Dated implementation evidence is
not proof of the code or observations in a current process.

After explicit authorization, three small turns can provide initial evidence:

1. A short plain-text turn, to capture routing, actual chat execution and ordinary
   finalization.
2. One small existing dependency case (for example one generated image, one
   bound evidence branch and one final text result), to capture branch/phase,
   evidence, attention and Late Fill movement.
3. One bounded edit of an existing small local artifact that exercises a normal
   repair/Closure path, to measure changed evidence, defensive validation,
   persistence and the final truthful state.

Use the current runtime's already-running capabilities. If a selected case has
no contention or repair, that phenomenon remains unobserved; do not force waits,
failures, timeout changes or rebase authorization to manufacture evidence.
Reviewed rebase/operator transitions need their own explicit authorization and
are not part of this three-turn proposal.

## Bounded branch handoff witnesses

`CAUSAL_TRANSITION_RESERVE = 256` reserves at most 256 additional compact
`transition_boundary` records per existing causal scope. It is independent of
both the 1,024 ordinary records and the 32 finalizer overflow records. It uses
`_CausalScope.retain`, the existing unified event sink and the existing 24-record
diagnostic tail; there is no additional snapshot, ledger or runtime scheduler.
A transition span reserves both enter/exit slots before entering its original
body. Point markers reserve one slot. No available slots means explicit
`transition_dropped_count`, not delayed or suppressed execution. Reservations,
recorded transitions and drops are exposed in `causal_snapshot`; scope coverage
gaps include transition drops. Existing byte/collection bounds still apply.

Recorded boundaries:

- `late_fill.schedule`, `late_fill.worker_submission`, `late_fill.worker` and
  `late_fill.initial_lookup`: actual scheduling-owner entry, Thread.start,
  continuation-owner entry and its existing initial lookup. The submission
  context carries the same observation-only handoff identity into the worker.
  A true scheduler return is not a claim of execution (the TESTING path still
  returns without starting a thread).
- `late_fill.start_check`: the actual existing complete
  `semantic_execution_gate_current_branch_v1` check, with its returned action,
  status and monotonic end. No second gate or selector evaluation is introduced
  by this reserved witness. This gate is not an aggregate proof of every
  dependency, repair, prepared-input and transport precondition.
- `multi_materialization.prepare`, `.queue`, `.worker`, `.instance_lock`,
  `.execution` and `.result`: existing preparation attempts, submission,
  worker-body entry, mutex wait/acquisition, execute_prepared_branch and its
  returned result. `available` means the result exists in the invoking owner;
  it does not mean canonical acceptance, persistence or downstream permission.
- `multi_materialization.callback` and `.callback_drain`: the original ordered
  callback and original future drain/shutdown. Exceptions swallowed by the
  existing callback handler still appear as a raised callback span.
- `late_fill.callback.progress_record`, `.lookup`, `.state_build`, `.publication`:
  the actual callback's progress metadata, lookup, state construction and lookup
  publication. `response_frame.callback_manifest_expansion` and
  `.callback_payload_hydration` appear only if those existing operations are
  actually invoked inside the selected callback/initial lookup. No extra read,
  verification, hydration or publication is done for measurement.
- `late_fill.running_state_build`: the actual existing construction of the
  running-state payload. This is a state-construction boundary, not a fabricated
  branch/thread start.

Each actual dispatch has a unique **observation-only** `handoff_attempt_id`;
`wave_id` joins its callbacks to the wave's drain handoff identity. Available
canonical attempt fields are copied separately, never invented or incremented.
Branch/phase IDs are retained independently. Missing canonical attempts have
`canonical_attempt_status=unknown`. The latest retained start check is linked
by exact response/branch/phase target; matching recorded attempt fields are
labelled `matching_recorded_attempt`, otherwise the attempt binding is unknown.
Neither equality of retry counters nor that link claims current execution
permission. `start_authority_status` remains `unknown` for aggregate permission.

A returned branch result receives the corresponding availability event's
observation result ID; its callback carries that `predecessor_result_id`.
Cross-phase producer IDs that are absent from the owner are not reconstructed
from dependency declarations, filenames, similar outputs or timestamps.
`predecessor_result_status=unknown` survives normal empty/null-field pruning.
No observer identity is added to executable branch plans or provider inputs.

Enter/exit records retain monotonic timestamps, process/boot identity, the
start event ID and parent transition event ID. Exit outcome is `returned`,
`raised`, or `aborted`, with exception class only. Enter records require an exit;
an absent exit after process death or delivery failure is incomplete evidence.
Dropped, missing, truncated and failed observations never become zero time.
A point lock-acquisition marker without its wait marker likewise cannot prove
wait duration. Inclusive callback/hydration/drain and parallel intervals must
be combined by interval union, not by summing their elapsed times.

These markers split the previously opaque pre-Late-Fill region into scheduling,
worker entry, initial lookup and later state construction. They do not identify
every owner before scheduling, nor all preparation between lookup and running
state. The old saved started_at value is not backdated or equated to worker
entry. Existing historical responses remain unchanged. Telemetry adds bounded
observation overhead; it does not add model work, gates, hydration or retries.

## State-flow investigation carrier

For response representation transitions that must survive ordinary event and
handoff-budget exhaustion, see [STATE_FLOW_DIAGNOSTICS](STATE_FLOW_DIAGNOSTICS.md).
This opt-in observer aggregates existing recursive work separately and leaves
all execution, authority, integrity and semantic owners unchanged.
