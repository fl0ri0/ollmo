# Bounded state-flow diagnostics

`ollmo_services/state_flow.py` observes existing owners when the server process
has `OLLMO_STATE_FLOW_DIAGNOSTICS_DIR` set to a diagnostic output directory.
Unset means no carrier, payload inspection or diagnostic I/O. It is an observer
switch, with no routing, scheduling, cache, authority or persistence decisions.
The existing request scope and `traced_thread_target` carry context into existing
workers. No additional worker, hydration, model call or canonical write is added.

The carrier is separate from the 1,024 causal events and both existing reserves.
Each scope has a maximum of 2,048 transition summaries and 8 MiB of output, with
256 KiB reserved for coverage summaries. Each identity family retains at most
2,048 digests; each transition aggregates at most 128 operation names. Explicit
drop and delivery-failure counters prevent an exhausted budget becoming zero
work. Summaries after request return can precede worker completion; a worker's
own completion summary is required. Abrupt death can lose an unfinished
transition: missing ends and absent worker completion remain UNKNOWN.

Only allowlisted shallow identities/status/counts are captured. No recursive
payload walk, added JSON serialization of runtime state, prompt, artifact body,
Base64, credential, full graph or response payload enters the carrier. Existing
byte buffers supply byte counts; existing CAS digests supply identities. The
carrier is diagnostic evidence, never an input to runtime decisions or a new
state/authority store. It is not inserted into canonical frames or sidecars.

## Boundaries and interpretation

Observed transitions include request normalization; Ghost routing and candidate
promotion; phase-graph attachment; live record registration/update; Late Fill
handoffs, dependency bindings, branch results and state construction; semantic
and graph Closure; terminal link rebind, reconcile and materialization; working
and response-frame construction; runtime/current-state copies; canonical
lookup/comparison/recovery; manifest expansion and payload reconstruction;
compaction, frame append and index write; epoch validation; bounded observation,
readiness projection, registry validation/reuse/append; and public UI projection.
Frequent decision/attention/aspiration/commitment/doubt read models remain
aggregated operation timers, independent of ordinary causal-event exhaustion.
They are never classified as redundant reviews.

A full finalizer is an actual `response_frame.finalize` invocation. A lightweight
nonterminal checkpoint is not a full finalizer. Lifecycle roles are inferred
from the entry Late Fill state and existing frame identity; ambiguous repair
versus ordinary successor roles require call-site/canonical lineage evidence.
A canonical-load invocation may fail, retry for stability, or return absence;
full reconstruction is recorded separately at the actual reconstruction owner.
The call-site and enclosing transition explain comparison versus recovery.

CAS root calls, child split operations, serializations, produced identities,
existing matches and actual writes are distinct counts. `recursive_split_operations`
counts selected child splits, not every scalar visit of the recursive walker.
Unique source subtree identity is UNKNOWN: object addresses or equal output
hashes do not prove identical authoritative inputs. Produced and consumed CAS
identity counts are independent, with explicit bounded-identity coverage.

Byte counters cover selected existing frame/snapshot/index/ledger serialization,
CAS read/hash/JSON decode and observation read/decode boundaries. They exclude
unmetered normalizers, object copies, some map/index reads, registry I/O and
other serializers. Missing fields mean UNKNOWN, not zero. Do not describe the
sum as total processed bytes or derive complete amplification ratios from it.

Transition intervals are inclusive wall times in a process-boot monotonic clock.
Operation times are inclusive and recursive; never add them to parent times.
Subtract only the union of enclosed, same-thread transition intervals to obtain
unattributed rest, which includes uninstrumented work and observer overhead.
Different threads can overlap; their union measures covered elapsed time, not
CPU time or proof of the actual dependency critical path. No time order is
inferred across process boots. Compare frame/sequence/epoch identities, but
never infer that equal identities make an authority/freshness check unnecessary.

Validate with `tests/test_state_flow.py`, the existing causal/transition tests,
and affected frame/snapshot/lookup/readiness/registry/semantics suites. API/E2E
validation must use a disposable checkout without production state, as described
in [TESTING_PROTOCOL](TESTING_PROTOCOL.md). Enabling diagnostics and restarting
or submitting live work each require applicable task authority.
