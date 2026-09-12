# Ollmo self-attack conformance

**Knobs may change strategy, never truth.**

Run from the checkout:

```sh
./ollmo self-attack
```

The command writes `report.md` and `results.json` under a new
`state/self_attack/run-*` directory. It exits `0` for passed observations and
checks, `1` for deterministic conformance failures, or `2` for incomplete
coverage. A truthful blocked obligation is not automatically a test failure.
Missing observations, exhausted process budgets, and skipped validation cannot
produce a passing result.

Reports distinguish three scopes, each labelled **PASS**, **INCOMPLETE**, or **FAIL**:

- **Deterministic/fake conformance** reports the separate fake-provider evidence.
- **Representative live gate** may pass when every selected profile and case has
  complete settled captures, required checks are complete, and no invariant findings
  exist. Missing cases, incomplete observations and unresolved confirmation remain
  incomplete. A truthful settled blocked obligation can satisfy conformance checks.
- **Full live conformance** retains the existing full-scope verdict. A representative
  gate pass cannot establish the complete required live scope (currently 17 profiles
  with the full corpus). Omitted profiles or cases keep full conformance incomplete.

The additive `scoped_verdicts` object in `results.json` stores these as
`deterministic_fake`, `representative_live_gate`, and `full_live_conformance`, using
`passed`, `incomplete`, or `failed`. Existing `verdict`, `overall_verdict`, and
execution exit-code semantics remain unchanged; representative PASS alone never
changes the overall result to PASS.

The default uses the existing fake-provider E2E harness, actual response,
graph, promotion, closure, Late Fill and response-frame owners, and the existing
resumable graph-rebase shadow runner. It does not start models or modify
production preferences, operator authorization, or learning state. Fake model
decisions are deterministic substitutes; this mode does not certify a live
Ghost model's interpretation.

The source release ships this guide, the harness/tests and two compact inputs:
`config/self_attack_corpus.json` (the default adversarial corpus) and
`config/graph_rebase_shadow_corpus.json`. Retained captures, local regression
outputs and large forensic corpora under `state/` are generated evidence, not
required release contents. See [Release Scope](RELEASE_SCOPE.md#source-selection-and-checksums).
The [September 6 status report](SELF_ATTACK_STATUS_2026-09-06.md) is historical;
it does not certify a changed checkout or expand representative live coverage.

For the 0.1.1 release, see [release validation and evidence](RELEASE_NOTES_0.1.1.md#validation-and-reference-evidence).
Packaging or owner-test success is not a fresh full Self-Attack campaign. The
dated live summary and full-live INCOMPLETE verdict remain unchanged unless a
separately authorized run supplies new evidence.

## Five boundaries, in priority order

1. **Commitment ↔ closure:** confident completion prose and status-only review
   claims cannot fulfill a missing saved file or a required semantic review.
2. **Aspiration ↔ promotion:** attractive reserved possibilities stay
   non-executable; explicitly requested work must remain visible.
3. **Repair/rebase ↔ anchored intent:** dropped obligations, advisory-only
   evidence, invalid authorization and frozen-parent mutation remain rejected.
4. **Late Fill ↔ exact evidence/source binding:** direct producer identity,
   source digest, transcript match, dependency order, cancellation and stale
   result rejection remain authoritative.
5. **Lenses/attention ↔ graph scope:** role metadata can orient strategy, but
   cannot enlarge executable scope, add timeout bonuses, or acquire runtime
   authority.

Each section includes multi-turn corpus observations, per-profile structured
adversarial owner probes, and focused existing runtime test results. Independent
conversation/dependency groups run separately so a stalled sequence does not
prevent the other boundaries being inspected. Model prose is neither an oracle
nor an accepted regression label.

## Controls and profile coverage

`controls.json` discovers controls from active source files and records their
sources. Finite sweep domains come from canonical runtime catalogs and bounds:
legacy Ghost mode aliases, embedding signals, explicit resolver timeout,
accepted-learning orientation levels, repair autonomy, rebase autonomy,
enforced policy and materialization concurrency. Invalid autonomy/policy input
is also checked for fail-closed normalization.

The default visits every supported finite value independently and adds four
seeded mixed profiles. `--pairwise` additionally covers every pair of values.
`--seed` makes profile generation repeatable. Attention, aspiration and
commitment are reported as advisory read models, not invented numeric knobs.
Other literal environment controls are listed as `inventory_only` with a
reason; credential, path, lifecycle, provider-specific and unbound controls are
not blindly assigned values. The report is a scoped conformance result, not a
claim that every possible environment variable or model has been tested.

The fake probe evidence records the effective values resolved by the actual
normalizers and role/policy builders. Full response captures retain request
and runtime control evidence. These are distinct from merely requested values.
No global environment control is sent to a running server as a request override.

## Live Ghost execution

With a compatible Ollmo control plane and models already running:

```sh
./ollmo self-attack --mode live --fake-evidence /absolute/path/to/fake/results.json
```

This executes actual adversarial requests and therefore creates response
frames and artifacts through Ollmo. It uses only the local control plane,
cached preflight observations, persisted Ghost preferences and canonical
Responses requests. There are no lifecycle or operator actions. Live profiles
sweep request controls; environment controls are covered in the isolated mode.
Full graph/evidence checks read `?view=truth`; the bounded debug view remains
the original shadow runner's diagnostic summary and is not a substitute for
canonical graph state.

Fresh canonical full-truth reads are recorded as `settled` only when the existing
shadow-runner state classifier identifies settled terminal or repair-needed truth;
open or unknown truth remains `observation`. The unchanged oracle still requires
a settled capture and complete runtime evidence. Historical observations are never
retagged. A companion truth-fetch failure is persisted under
`captures/truth-fetch-failures/` with endpoint, trigger, error, HTTP status, timeout,
attempt and elapsed/remaining time. Transient companion failures receive at most
one immediate retry within the same sequence deadline, and do not suppress a
later canonical read merely because compact status is unchanged.

Dependent turns share a conversation id and carry their exact predecessor's
recorded context. A pending response is observed by its existing id. Persisted
`submitting` or ambiguous dispatch remains GET-only on resume. Never erase its
manifest to force a resubmission.

## Reports, budgets and resuming

Useful options:

```sh
./ollmo self-attack --output state/self_attack/my-run
./ollmo self-attack --pairwise --seed 42
./ollmo self-attack --profile-limit 2 --profile-timeout 30 --minimize-budget 4
./ollmo self-attack --corpus config/graph_rebase_shadow_corpus.json
```

`--profile-timeout` defaults to 600 seconds per fake profile, shared evenly across
its independent sequences. Live mode uses the separate hard limits below. Each sequence gets its own process; a timeout kills
that diagnostic worker and leaves evidence and an incomplete result. It does
not declare an Ollmo model failed or make a semantic judgment. Increase this
explicit budget for long local workloads. `--max-cycles` bounds shadow
observation cycles. `--owner-timeout` defaults to 600 seconds per existing test
group. `--skip-owner-tests` and a truncated `--profile-limit` always disclose
incomplete coverage.

Reusing the same `--output` resumes the existing manifests only when corpus,
profile, execution mode and source identity match. Source changes require a
new output directory. Response and conversation ids include a run namespace;
each reduction uses another namespace. Isolated runtime files are retained
under each sequence's `runtime/` directory, including after a timeout.

After changing the diagnostic oracle, recheck retained captures without
executing requests:

```sh
./ollmo self-attack --recheck state/self_attack/my-run
```

This creates a separate report under `my-run/rechecks/`, preserves the original
evidence, and records both capture and analysis source identities. It also works
from a checkpointed run's `progress.json` after interruption.

To add the scope labels to a completed report using its existing verdicts and
case evidence, without running the suite or reapplying the oracle:

```sh
./ollmo self-attack --refresh-report state/self_attack/my-run
```

This updates only `results.json` and `report.md`, saves their previous versions in
`reporting-backups/`, and records the reporting-only refresh. Original run manifests,
source identities, captures, findings and execution verdicts remain unchanged.

The output contains:

- `controls.json`: source inventory, finite domains and exclusions.
- `results.json`: verdict, per-case findings, probe evidence, test results,
  coverage, reproduction attempts and regression replay status.
- `report.md`: ordered human-readable results.
- Per-profile/per-sequence `manifest.json`, full response `captures/`, compact
  observations, retained artifact bytes, worker logs and isolated runtime files.
- JUnit XML and logs for the five existing owner-test groups.

Cross-profile comparisons preserve the multiplicity and semantic fields of
anchored intent obligations while allowing different phase topology, provider
selection and generated prose. They are deterministic structural comparisons,
not an LLM similarity score or a proof of arbitrary natural-language equivalence.

## Failure reduction and regression replay

Each new signature is reproduced twice before it can become a regression.
The reducer removes unrelated dependency-safe turns, adversarial prompt
fragments and control overrides. The immutable intent clause and its
expectations remain intact. A removal is kept only if the same case, invariant
and location signature reproduces twice. Divergence reproductions rerun both
the baseline and candidate. Fake mode has a shared default budget of 24 reducer attempts.
Live mode additionally shares four profile executions across saved regressions,
confirmations, minimizations and baseline references, within the hard live time limits;
budget exhaustion and unconfirmed failures remain visible. The reducer reports
bounded deletion, not global minimality.

Confirmed cases are persisted as additive, content-addressed envelopes under
`state/self_attack/regressions/`. They embed the replayable shadow corpus,
profile, baseline for divergence, signature, source identity and evidence paths.
Subsequent commands automatically replay compatible saved regressions and
report `reproduced`, `resolved`, `incomplete`, or `different_mode`. Nothing is
activated as a learning policy or operator authorization.

Replay one explicitly:

```sh
./ollmo self-attack --replay state/self_attack/regressions/CASE.json
```

For a saved live case, also supply `--mode live` and matching `--fake-evidence`. External proposals can be
added as ordinary corpus prompts and diagnostic `metadata.attack_fragments`;
they cannot supply pass/fail authority. The system currently has no automatic
LLM attack generator dependency.

## Offline convergence and trigger audit

Analyze the retained corpus without dispatching requests, probing the control
plane, running models, or changing runtime behavior:

```sh
./ollmo self-attack --audit-convergence state/self_attack --detach --output state/self_attack/convergence-audit-example
```

This uses the same self-attack manifests, captures, profiles, response identities,
atomic output writer and detached controller. Omit `--detach` for small fixtures.
The command cannot be combined with live execution, replay, recheck or report
refresh modes. A changed input inventory or audit implementation requires a new
output directory; an interrupted matching audit resumes completed response files.

The output contains `results.json`, ranked `report.md`, `inventory.json`,
`responses/*.json`, `ledger-evidence.json` and `auxiliary-evidence.json`.
`process.json` records PID, session, stdout/stderr log and result paths;
`progress.json` records the active response and completion count;
`completion.json` records the exit code. The detached controller starts a new OS
session with disconnected input and file-backed output and survives closing the
launching terminal or Codex. It still requires the computer to remain running.

Interpretation is deliberately conservative:

- Versioned captures and copied captures are observations, not invocation counts.
- Lens, attention, aspiration, doubt, commitment and promotion surfaces retain
  their existing runtime/advisory authority. Unchanged projections do not prove
  repeated execution or redundant work.
- State hashes identify retained projections, not complete effective invocation
  inputs. Missing input/authority/evidence/consumption provenance stays `unknown`.
- Branch execution effort, batch wall time, preparation, post-branch callback
  tail, Late Fill intervals and observer capture intervals remain separate.
  Inclusive and overlapping durations must not be added as wall-clock savings.
- Batch-history ordinal conflicts are disclosed and excluded from aggregate
  timing. Identical timing witnesses may be repeated projections; without event
  ids they are not an exact invocation census.
- A blocked or pending record does not by itself prove a wait, false wait or
  missed wake-up. A complete wait predicate and its owner/authority gates are
  needed to establish that a consumer should have advanced.
- Ranked exposure identifies where a more precise measurement could matter;
  recoverable latency remains unknown unless causality proves otherwise.

The audit preserves Ollmo's existing truth model. Missing provenance is an audit
boundary, not a runtime defect. Any recommended extra provenance belongs on
existing debug/frame/capture records. The command never implements that
instrumentation, changes a scheduling policy, or launches a follow-up workload.

Validate the offline auditor with:

```sh
.venv/bin/python -m pytest tests/test_self_attack_convergence.py tests/test_self_attack.py -q
```

## Conformance development validation

```sh
.venv/bin/python -m pytest tests/test_self_attack.py -q
.venv/bin/python -m pytest tests/test_graph_rebase_shadow_corpus_runner.py tests/test_fake_backend_e2e.py tests/test_graph_rebase_review.py tests/test_runtime_graph_rebase_shadow_producer.py tests/test_graph_rebase_readiness.py tests/test_graph_rebase_operator.py tests/test_graph_rebase_partial_successor.py -q
```

The tests include deliberately corrupted truth records and deterministic
reducer faults, so a passing test verifies that the oracle can detect failures
as well as accept valid strategy differences.

## Hardening and live observations

Run the entire deterministic sweep followed by bounded live Ghost observations:

    ./ollmo self-attack --live-after-fake

The default fake sweep runs every discovered finite value and four seeded mixed
profiles (currently 38). `--jobs 2` bounds isolated profile concurrency; `--jobs 1`
serializes it. Case order, profile settings and comparisons stay in seed order.
Each fake profile has an explicit 600-second total wall budget divided among its
independent conversations. This includes canonical frame hydration, not just
provider execution. A budget expiration remains incomplete.

Live execution is gated on complete passing fake evidence for the current source.
It reads the passive control-plane surfaces first, uses existing ready instances,
and never starts models or mutates lifecycle policy. Environment sweeps remain
isolated fake-only. The live default is two request profiles; use
`--live-profile-limit` to expand it. Live results are separate observations under
`live/`, linked from the fake report. They cannot substitute for deterministic
fake invariant coverage. A separate live invocation requires
`--fake-evidence /absolute/path/to/fake/results.json`.

The hardening investigation identified these smallest responsible layers:

| Original incomplete observation | Cause | Responsible correction |
|---|---|---|
| Promoted image follow-up stalled | Runtime gap: a reserved website cue borrowed an action from another sentence and created an HTML obligation | Existing artifact-intent detector now requires a local action; regression preserves later explicit website promotion |
| Audio root never reached exact consumer evidence | Missing scenario coverage: generic fake chat output and fixed STT transcript did not model the requested source | Existing fake adapter supplies the explicit source fixture; STT decodes the exact saved WAV, with unique content-addressed fixture paths |
| Slow full response reads | Runtime gap: repeated recursive copying and rewalking of hydrated sidecar trees | Existing frame hydrators normalize once per traversal, preserving CAS/source authority checks |
| Settled audio observation crashed | Missing observability: the oracle assumed every runtime outcome was a string | Structured diagnostic outcomes no longer crash the observer; actual mutation records retain the same strict checks |
| Audio follow-up unobserved | Missing scenario coverage downstream of the unsettled/crashed predecessor | Original dependency chain is retained and rerun after owner fixes |

Case results now include `incomplete_observations` with the exact absent witness,
case state, last recorded owner stage and evidence path. A completed graph without
a producer/consumer evidence witness does not count as complete Late Fill coverage.
Fake-provider handoff checks additionally compare the consumer's actual input path
and bytes with its declared producer. Missing source evidence remains incomplete;
accepted mismatched evidence is a deterministic failure.


## Hard live budgets and detached execution

The live corpus and profile selection are unchanged: ten cases in five two-turn
sequences, using baseline and `ghost_mode: repair` by default. The main sweep has
ten sequences across those two profiles. The defaults are explicit:

| Option | Default | Scope |
|---|---:|---|
| `--live-sequence-timeout` | 1,200 seconds | One dependency-connected sequence, including observation/capture; constrained by the remaining main budget |
| `--live-main-budget` | 3,600 seconds | Entire live preflight and main sweep |
| `--live-confirmation-budget` | 900 seconds | All additional live regression/confirmation/minimization work |
| `--live-replay-timeout` | 300 seconds | One entire additional profile replay, across all its sequences |
| `--live-attempt-cap` | 4 | Shared additional profile executions, including paired baseline references |

The sequence default increased after passive inspection of the September 6 live
run found Late Fill completion 11–16.5 minutes after sequence observation began.
Twenty minutes provides observation/capture headroom, not a completion guarantee.
The unchanged 60-minute main cap takes precedence: slow early sequences can leave
later cases unobserved and incomplete. Confirmation replays still have a five-minute
ceiling. Existing manifests retain their original limits; this default neither
renews expired budgets nor starts a new sweep.

Main and confirmation deadlines bound total live execution to 75 minutes plus
minimal evidence/report finalization. No new live work is dispatched during
finalization. An OS timer interrupts blocked observer calls and the existing
subprocess isolation kills/reaps diagnostic workers. These limits do not cancel
server-side responses or perform model lifecycle operations.

Sequence expiration stops that sequence; the main sweep may still observe other
planned sequences within its global allowance. Confirmation expiration stops the
replay or confirmation phase whose allowance expired. A timed-out reproduction
stops reduction of that finding. No incomplete reproduction can confirm a failure.
Cases with unresolved submitted responses cannot be automatically replayed.
Missing observations remain incomplete; independent deterministic violations remain
failures. Neither timeout nor model prose supplies a semantic verdict.

`live-budget.json` persists deadlines and attempt reservations before dispatch.
Each sequence retains its `observation-budget.json`. Resume cannot replenish
these limits. Completed live output is returned without new execution. Unresolved
cases and unconfirmed findings are retained as envelopes under `manual-review/`,
with original corpus/profile, response identities, missing witnesses and evidence
paths, and listed in both `results.json` and `report.md`. These envelopes never
automatically rejoin the regression replay queue.

Launch the same one-command pipeline independently of the terminal or Codex:

    ./ollmo self-attack --detach --live-after-fake --output state/self_attack/my-detached-run

`--detach` creates a new OS session, disconnects standard input, and sends stdout
and stderr to `stdout-stderr.log`. `process.json` records the controller PID,
session ID, log, manifest and final report paths. `completion.json` records the
final exit code. Exiting the calling terminal or Codex does not terminate the
controller or its workers. The computer must remain running.

For a smaller live corpus across broader control diversity, use:

    ./ollmo self-attack --detach --live-after-fake --live-profile-limit 6 --live-high-risk --output state/self_attack/diverse-live-gate

The fake sweep still covers the full corpus and all 38 profiles. The live
selection retains baseline/repair and chooses four other profiles from the
17-profile request-control matrix, maximizing distinct control values and then
interactions, with stable catalog ordering for ties. With seed 0 these six
profiles collectively cover every discovered request-control value. This is
representative coverage, not full interaction coverage.

Ten live case executions are selected without editing their prompts, expectations
or dependencies: the aspiration pair on baseline and repair, the rebase pair on
advisory learning, the exact audio/source case on a mixed improviser profile,
the attention pair on a mixed explorer profile, and the saved-report case on a
mixed worker profile. All five boundaries are represented across the gate, not
on every profile. The persisted `case_selection` in the manifest/results records
the exact mapping. Only matching cases with baseline captures can receive live
cross-profile comparison; all other combinations remain unobserved.

All hard budgets above are unchanged. Missing observations stay incomplete and
selection exclusions are reported explicitly. Even complete selected captures
cannot yield a full-live-conformance pass. The gate never expands itself to all
17 profiles or renews an expired budget.

To explicitly hand off an interrupted live run while retaining its exact response
IDs and ambiguous POST state:

    ./ollmo self-attack --detach --live-after-fake --resume-live-from state/self_attack/previous/live --output state/self_attack/continued

The handoff verifies the exact corpus, profiles and seed, records source
provenance in `live-handoff.json`, and copies retained shadow evidence. Pending
responses are observed by GET, never replaced by duplicate root submissions.
Existing hard-budget counters/deadlines survive handoff. Only a legacy run that
had no hard global budget receives the newly configured limits. Fresh passing
fake evidence for the current source is still required before live execution.
Fake validation time precedes, and is not part of, the 75-minute live allowance.
