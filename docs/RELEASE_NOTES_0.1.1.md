# Ollmo 0.1.1 — stabilization and evidence hardening

Release date: September 12, 2026. Tag: `v0.1.1`. These notes compare the
released source with the August 1, 2026 `v0.1.0` public tag. They describe the
accumulated changes, not a new product architecture or a promise of production
stability. The release remains a solo-maintained experimental `0.x` project.

## What changed

### Requested work stays attached to its evidence

Requested file identities and counts survive reduced detector results and graph
rebuilds. Category-scoped deferral keeps “do not create SVG” from suppressing
unrelated requested text files. Deferred work stays visible; it does not become
silently waived or fulfilled. Saved-file consumers receive the actual authorized
saved bytes with producer, response, branch, phase, path and digest bindings.
Changed or missing evidence prevents unsupported completion.
Explicit image reservations also survive comma/sentence boundaries and scoped
pronoun/ordinal references while remaining non-executable until promoted.

Artifact registration and public outputs follow the accepted canonical frame.
Equal bytes in different files do not erase distinct obligations. HTML/CSS/image
composition, counted image prompts, linked-artifact rebind, exact repair targets
and exported bundles have stronger checks against the actual saved outputs.

### Audio, images and continuation are more reliable

TTS generation has stronger source, duration/completeness and physical integrity
checks, with bounded evidence-driven recovery for supported cases. Direct
TTS-to-STT work binds its transcript to its own generated source. Image preparation
and retry paths preserve exact per-branch prompts and current provider evidence.
Semantic review, deferred branch continuation, retained context and public output
projection have additional regression coverage. Explicit MLX reasoning controls
and audio transport handling are also improved.

These changes strengthen supported workflows; they do not guarantee model quality,
add automatic authority to model prose, or remove truthful BLOCKED outcomes.

### Durable state and diagnosis have clearer boundaries

Frame storage, Index publication/verification, Epoch binding and Readiness
retention have been hardened. Private snapshot serialization/size preparation and
single-use observation reuse avoid repeated transformations while retaining fresh
source, identity, integrity and authority checks. No benchmark percentage is a
release guarantee.

Causal/transition telemetry and bounded state-flow diagnostics make convergence
and persistence costs easier to inspect. Late Fill telemetry has a distinct
observer owner. Readiness epoch-mismatch diagnostics report bounded physical
file changes without attributing an unproven writer or rereading payloads solely
for diagnostics. Observers remain separate from runtime execution authority.

### Reproducibility and public guidance

The package now includes Self-Attack's deterministic/fake harness, explicit live
coverage scopes, resumable evidence capture, two compact corpora and five curated
reference packages. The documentation audit reconciled response truth, repeated
Closure, artifact identity, lookup/hydration and finalization ordering. Accepted
learning remains available as explicitly enabled advisory hints; it cannot replace
current evidence or independently authorize work.

## Upgrading from 0.1.0

The supported platform remains macOS on Apple Silicon with Python 3.11 or newer.
Model weights and backend packages are installed separately. The canonical
Responses endpoints remain `/api/responses` and `/v1/responses`; their limited
compatibility surface is documented in [Responses Contract](RESPONSES_CONTRACT.md).
This release does not establish a stable 1.0 API or a blanket downgrade guarantee.

1. Retain the old installation and back up its configuration, populated model
   registry, saved artifacts and durable state together before changing code.
   Preserve the Ledger and referenced sidecars as a set.
2. Extract the new archive into a separate directory and install its requirements
   using the [installation instructions](../README.md#install-and-start).
   Its generated empty `model_ports.json` is for a fresh installation; do not copy
   it over an existing populated registry.
3. Stop the old control plane explicitly before switching the running installation.
   Do not mix old/new processes or replace active state during execution. Preserve
   existing state rather than running clean/reset commands as an upgrade step.
4. Check existing response/artifact lookup after switching. Stricter evidence
   gates can expose previously hidden missing artifacts, invalid references or
   unresolved obligations as blocked/repair-needed states. Report those states;
   do not delete evidence or relax gates to make them appear complete.

Legacy/derived Index recovery remains supported by its current validation paths;
this release preparation does not run an automatic production-state migration.
Explicit maintenance/attestation is separate operator work. For rollback, retain
the matching old code and pre-upgrade data backup; newer writes are not assumed
to be understood by older code. See [Truth Sources](TRUTH_SOURCES.md).

## Validation and reference evidence

### Release validation on September 12, 2026

The final full public-source suite passed: **3,394 tests**, with **one expected
skip** and **942 passing subtests**, in 11 minutes 12 seconds. It ran in a frozen,
disposable copy with network access blocked and writes confined to temporary
test storage. The skipped check concerns a development-only monitor shim that
public packages intentionally exclude; the packaged monitor entrypoint passed.

An initial run exposed nine failures that also reproduced against the latest
published source. Preparation corrected reserved-image candidate tracking,
updated stale/incomplete test fixtures and moved scheduling-test artifacts into
isolated storage. The affected checks passed (208 tests and 18 subtests, plus the
same development-only skip) before the final full run. Runtime limits and safety
assertions were preserved. Release, landing-page and reference-export checks
also passed: **70 tests**, included again in the final full suite.

All five reference packages passed their existing checksum, file-set and
applicable bundle checks. No new live campaign or benchmark was performed.
This is release validation evidence for `v0.1.1`, dated September 12, 2026.

### How to read the evidence

Evidence must retain its scope:

| Evidence | What it establishes | Limit |
| --- | --- | --- |
| Current release metadata, package and regression checks | Tested behavior and public-file integrity for the prepared source | Not proof of every live model/workflow |
| Deterministic/fake Self-Attack and its owner tests | Runtime authority and evidence invariants with controlled providers | Not live Ghost interpretation or provider quality |
| [September 6 Self-Attack summary](SELF_ATTACK_STATUS_2026-09-06.md) | Historical fake PASS and representative live-gate PASS: six of seventeen profiles, ten selected cases | Full live conformance remains INCOMPLETE; no new live campaign is implied |
| [Five curated reference packages](../examples/README.md) | Historical completed examples with saved outputs, identity-bound evidence and checksum manifests | Publication-copy verification is not a fresh execution on 0.1.1 |
| Performance investigations | Measurements for their recorded source, workload and environment | No universal speedup or relaxation of integrity gates |

The references cover a linked website with images/audio, continuous narration,
a TTS/STT roundtrip, three independently inspected images, and saved JSON read
by an HTML consumer. Their original dates, response/frame IDs, media and manifests
are preserved. Use the package verifier to check a downloaded example:

```sh
python3 scripts/export_reference_run.py --verify examples/reference-runs/2026-09-07-saved-json-read-html-verified
```

For new local deterministic evidence, follow [Self-Attack](SELF_ATTACK.md).
Raw captures, production Ledgers, local benchmark corpora, credentials and model
files are excluded. Do not relabel a historical run as current release validation.

## Known limits and release artifacts

The [known limitations](KNOWN_LIMITATIONS.md) and [release scope](RELEASE_SCOPE.md)
remain applicable. In particular, ordinary finalizer persistence errors can be
logged without becoming request errors. HTTP success or an in-memory completed
frame alone is not a durable-commit receipt. Verify matching durable state and
saved-artifact evidence; Readiness retention does not repair this limitation.

The source archive contains an empty registry and a `MANIFEST.sha256` covering
its staged files except the manifest itself. The archive's own SHA-256 is separate.
A mutable public-main checkout does not carry that archive manifest. Version,
release date and tag/commit should be recorded with any published result. Use
[CITATION.cff](../CITATION.cff) for software attribution and the release date.
