# Changelog

Notable public releases of Ollmo are recorded here.

## [0.1.1] - 2026-09-12

Stabilization and evidence hardening since the August 1, 2026 public release.
This patch release retains the existing local-first product and experimental
`0.x` support scope. See [release notes](docs/RELEASE_NOTES_0.1.1.md) for upgrade
considerations, references and validation limits.

### Fixed and hardened

- Preserve exact requested file identities, counts and producer/consumer
  dependencies through graph rebuilds, repair and category-scoped deferral.
- Preserve explicitly reserved image candidates across punctuation and scoped
  pronoun/ordinal references without promoting neighboring artifact categories.
- Bind saved-file consumers to the actual saved bytes, paths, identities and
  digests; keep missing or changed evidence visible as incomplete work.
- Tighten artifact authority, multi-file HTML/CSS/image binding, target-specific
  repair, canonical outputs and openable bundle validation.
- Improve counted image prompts and retries, TTS source/completeness checks,
  bounded generation recovery and direct TTS-to-STT dependency evidence.
- Harden durable frame/Index/Epoch handling and Readiness retention; reduce
  repeated preparation/hydration while retaining current integrity checks.
- Improve semantic-review publication, late-fill continuation, retained context,
  MLX reasoning controls, audio transport and accepted-learning eval refresh.

### Added

- Deterministic/fake Self-Attack conformance, explicit live coverage scopes,
  resumable captures and two compact reproducible corpora.
- Causal, transition and state-flow diagnostics; extracted Late Fill telemetry
  ownership and bounded epoch-mismatch diagnostics.
- Five curated, self-contained reference-run packages, with saved artifacts,
  sanitized response/monitor evidence, verified bundles where applicable and
  checksum manifests.

### Documentation and distribution

- Audited current response, artifact, Closure, durability, lookup and authority
  contracts; preserved dated evidence and corrected misleading completion claims.
- Updated public guidance and companion-skill contracts; documented output
  rendering boundaries and the experimental support scope.
- Extended the source allowlist for public diagnostics, Self-Attack and references;
  exclude Finder metadata and keep SHA manifests specific to built archives.

## [0.1.0] - 2026-08-01

Initial public release of Ollmo.

- Local-first AI runtime substrate and control plane for Ollama, MLX, and
  llama.cpp, with an optional explicitly enabled ChatGPT provider.
- Canonical Responses, Ghost routing, work graphs, materialized artifacts,
  continuable runtime state, and evidence-gated closure.
- Local model and media workbench, Codex companion skill, static project site,
  reproducible release archive, and Apache-2.0 project licensing.
