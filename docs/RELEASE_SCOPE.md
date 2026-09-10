# Ollmo 0.1.0 Release Scope

Ollmo 0.1.0 is the first public release candidate of Ollmo's local-first AI
runtime and control plane. It is experimental software: useful enough to run
and inspect, but still allowed to evolve within the `0.x` series.

## What 0.1.0 Provides

- A local Flask control plane and browser interface.
- A standalone static repository landing page, packaged with all of its local
  CSS, JavaScript, font, and font-license dependencies. The live local root
  remains the dashboard; `/site/` provides the canonical local preview of the
  self-contained `site/` publication directory. Its contents can be placed at
  the root of a dedicated Pages branch and activated manually; the release
  includes no publication workflow.
- Local model discovery and lifecycle support for the backends already
  implemented by Ollmo, including Ollama, MLX, and llama.cpp paths.
- The canonical `/api/responses` and `/v1/responses` execution surfaces.
- Ghost routing, durable response frames, runtime status, history, and
  materialized artifact tracking.
- Truthful response state: `outputs`, artifacts, response frames, lifecycle,
  closure, and late-fill state are authoritative; model prose is not proof of
  success.
- A bundled, Codex-specific Ollmo skill for inspecting local runtime truth,
  choosing safe request paths, and using canonical responses and artifacts.
  Skill installation is manual and does not start Ollmo or enable cloud use.
- An optional, explicitly enabled ChatGPT route through Codex that accepts a
  prompt, bounded context promoted for the current turn, and local files or
  Ollmo artifacts explicitly selected for that turn, then returns text. Ollmo reuses a login owned by the ChatGPT app
  or a separately installed Codex CLI without reading, copying, or storing that
  login. The dashboard exposes ChatGPT as an external cloud target and Ghost
  preference, never as a local running instance.
- A reproducible, allowlist-based source archive with a SHA-256 manifest.
- Five explicitly curated, self-contained reference-run packages with copied
  public artifacts, sanitized final-response and monitor evidence, and package
  checksums. Production response ledgers, monitor ledgers, runtime state, and
  unrelated generated artifacts remain excluded from the source release.
- Apache License 2.0 coverage for Ollmo's own code and project documentation,
  with separate notices for third-party components, machine-readable citation
  metadata, and research-contact paths maintained by `@fl0ri0`. Public issues,
  pull requests, support, and review are not currently promised.

Existing capabilities remain available unless they would make installation,
security, or response truth materially unsafe. Optional backends and
modalities remain experimental.

## Source selection and checksums

`scripts/build_release_archive.py` owns the current source allowlist and verifier.
`PUBLIC_DOCS` includes the normative contracts, `SELF_ATTACK.md`, its dated
September 6 status summary, causal telemetry and state-flow diagnostics.
`PUBLIC_CONFIG_PATHS` includes exactly `config/self_attack_corpus.json` and
`config/graph_rebase_shadow_corpus.json` as compact reproducible inputs. Harness
and test code are included through the normal source-tree rules. Raw captures,
production ledgers, local forensic corpora and generated `state/` are excluded;
running conformance creates new local evidence.

The four exact companion-skill files are `skills/ollmo/SKILL.md`,
`skills/ollmo/NOTICE`, `skills/ollmo/agents/openai.yaml` and
`skills/ollmo/references/ollmo-contract.md`. Other skills are excluded. Some skill
engineering/monitoring references describe development-checkout resources:
`AGENTS.md`, `plans/` and `skills/ollmo-run-monitor/` are not shipped. Public
operation uses the bundled contracts and `OLLMO_FOR_AGENTS.md`; missing optional
development resources are not a reason to infer runtime authority.

Development-only benchmark/idea/direction notes, legacy/framework decisions,
implementation reports and the release acceptance journal are excluded. The
allowlist, rather than an old journal's file count, determines current packaging.
The canonical `scripts/ollmo_run_monitor.py` ships; any compatibility wrapper
under `state/` does not.

`MANIFEST.sha256` covers every staged regular release file except itself,
including the generated empty `model_ports.json`. The verifier checks the exact
file set and every digest, as well as archive-path/type/size and public-scope
constraints. The manifest cannot contain the final archive's own digest; the
builder reports that SHA-256 separately. Deterministic staging/archive metadata
makes unchanged selected inputs reproducible. This is content-integrity and
packaging evidence, not a signed release or proof of runtime conformance. The
builder performs no upload; publication gates and external publication remain
separate actions.

## Supported Environment

The primary tested environment is a recent macOS release on Apple Silicon
with Python 3.11 or newer. Individual local capabilities additionally require
their own backend and model packages. Other operating systems and processor
architectures are not part of the 0.1.0 support promise.

Ollmo is intended for local, single-user use and binds its web control plane
to `127.0.0.1` by default. Remote and multi-user deployment are outside this
release scope.

## Local-First and Optional Cloud Use

Local execution is the default product posture. ChatGPT is a separate optional
cloud path:

- it must be explicitly enabled before Ollmo can route a prompt or selected
  files to it;
- only the prompt, context promoted by Ollmo for the current turn, and files or
  Ollmo artifacts explicitly selected for that turn are included; unrelated
  files and conversation artifacts are not added automatically;
- recognized images use Codex's native image-input path, while other selected
  regular files are copied into a temporary working directory configured
  read-only for that Codex run;
- one request accepts at most 5 files, up to 100 MiB per file and 250 MiB in
  total; URLs, folders, and symbolic links are rejected;
- the route returns text, and acceptance of a regular file does not guarantee
  semantic interpretation of every format;
- the ChatGPT app's bundled Codex executable is preferred, with a separately
  installed `codex` executable as fallback;
- Ollmo does not select a fixed cloud model and does not promise to mirror the
  model selected in an open ChatGPT conversation;
- the exact GPT variant is not exposed to Ollmo, and model self-description is
  not runtime identity proof;
- the prompt, promoted context, and selected file bytes sent through this route
  leave the machine and are processed by OpenAI under the user's existing
  account and applicable terms;
- direct dashboard turns are ephemeral and independent; Ollmo can promote
  bounded relevant context for a referential turn, but no provider session is
  silently resumed.

Direct provider API-key management and other external providers are not part of
0.1.0.

## Compatibility

Saved and historical builds that were never publicly released do not create a
compatibility obligation. Public contracts introduced in the `0.x` series
will be changed deliberately and documented, but may still evolve.
