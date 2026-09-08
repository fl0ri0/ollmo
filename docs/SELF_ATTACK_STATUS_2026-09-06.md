# Self-Attack status — 2026-09-06

This is the compact public evidence summary for the current Self-Attack harness.
It deliberately excludes retained captures, response ledgers, worker runtimes,
logs, and other local forensic material.

## Verdicts

- Deterministic/fake conformance: **PASS**.
- Representative live gate: **PASS**.
- Full live conformance: **INCOMPLETE**.

The representative gate covered six of seventeen profiles and ten selected case
executions across all five documented authority boundaries. Every selected case
had a complete settled capture, owner tests passed, and no deterministic
invariant finding was recorded. The omitted profiles and case combinations remain
unobserved, so this evidence does not claim full live conformance.

## Boundary coverage

1. Commitment and closure: 1 of 1 selected case complete.
2. Aspiration and promotion: 4 of 4 selected cases complete.
3. Repair/rebase and anchored intent: 2 of 2 selected cases complete.
4. Late Fill and exact evidence/source binding: 1 of 1 selected case complete.
5. Lenses/attention and graph scope: 2 of 2 selected cases complete.

No new reproducible failure was promoted by this representative run. Existing
regression envelopes from another execution mode were reported as
`different_mode`, not counted as passing or failing live evidence.

## Additional regression evidence

The linked-artifact terminal evidence correction was validated by 57 focused
tests plus 11 subtests and by 732 broader conformance tests plus 459 subtests.
The combined completed evidence was 789 tests and 470 subtests, with syntax
checks passing. A separate broader Responses attempt remained incomplete after
an unrelated pre-existing syntax-repair retry fixture failed; it is not presented
as passing evidence and no assertion or product guard was weakened.

## Reproduction boundary

The executable harness, invariants, compact corpora and tests are published.
Run `./ollmo self-attack` from a checkout to create fresh evidence locally. Live
mode may use an already running control plane but performs no lifecycle action.
See [SELF_ATTACK.md](SELF_ATTACK.md) for commands, limits, resume semantics and
verdict definitions.

The large content-addressed ledgers, raw captures, local forensic corpora,
temporary live-check data and production runtime state are intentionally not
part of the public source selection.
