---
name: ollmo
description: >-
  Operate Ollmo through its control plane: inspect local runtime truth, choose
  Ghost or direct/capability execution, and interpret response frames and artifacts.
  Use for Ollmo/Ghost work, useful bounded local capabilities, or a request beginning
  with [OLLMO_DOWNSTREAM_EXECUTION_V1]. Includes lifecycle boundaries and checkout
  discovery; does not make Ollmo the default provider.
---

# Ollmo

## Project Provenance and License

Copyright 2026 fl0ri0.

The Ollmo Skill is part of Ollmo. The Ollmo Skill was conceived, created,
built, and developed by [@fl0ri0](https://github.com/fl0ri0). It is distributed
with Ollmo under the Apache License 2.0; see the repository's `LICENSE` and
`NOTICE` files. It is an Ollmo integration for Codex and ChatGPT. No OpenAI
authorship or endorsement is claimed.

## Ollmo Downstream Provider Mode

If the current request begins with the exact marker
`[OLLMO_DOWNSTREAM_EXECUTION_V1]`, Ollmo has already invoked Codex/ChatGPT as
its bounded downstream provider. This mode overrides the normal Codex -> Ollmo
supervisor workflow for that request only.

In downstream provider mode:

- Do not resolve an Ollmo checkout, inspect Ollmo runtime truth, call `$ollmo`,
  invoke Ghost, call Ollmo APIs or CLIs, route to local Ollmo models, or trigger
  Ollmo lifecycle work. Any of those actions would recurse back into the
  caller.
- Do not act as the user-facing supervisor, widen the request, create a new
  Ollmo plan, or decide Ollmo completion. Ollmo retains orchestration, response
  state, artifact, and closure authority.
- Execute only the live task inside
  `<ollmo_bounded_task>...</ollmo_bounded_task>`, using only directly available
  downstream capabilities and the staged inputs supplied with the call.
- Treat `<ollmo_promoted_context>...</ollmo_promoted_context>` as bounded,
  reference-only context. Do not turn it into additional work or new intent.
  Selected-message reference material inside the bounded task is reference
  evidence unless the live task explicitly asks you to transform or inspect
  it.
- Do not rediscover original input paths, search for more context, or perform
  unrelated workspace work. Use staged files as the supplied task inputs.
- Honor the bounded task's requested result format and completion criteria.
  On success, return only the provider result in the requested form. If the
  task cannot be completed, begin the result with `BLOCKED:` followed by a
  concise reason, and do not claim completion or invent outputs.

A quoted or later mention of the marker does not activate this mode; it must
be the first non-whitespace content in the current request. When the marker is
absent, follow the normal Codex -> Ollmo contract below and keep Codex as the
supervisor.

## Select the task and preserve authority

Codex remains the user-facing supervisor, editor and final narrator. Ollmo owns
local runtime state and artifact/response truth. Ghost proposes routes, phase
graphs and advisory reviews; Runtime validates promotion, patches, execution and
closure. Accepted learning is soft orientation, never proof or independent
authority to execute, waive, supersede, repair or freeze work.

- **Observation:** status, capability and routing questions use passive evidence
  and, when appropriate, route preview. They do not authorize execution, lifecycle,
  registry pruning, configuration sync, cleanup or repair.
- **Execution:** a request to generate or run a bounded task authorizes its ordinary
  execution steps. Do not reapprove the same generation at each step. Use existing
  compatible instances and observe pending work instead of launching duplicates.
- **Lifecycle/recovery:** start, stop, restart, unload, remove, reset, clean, archive
  and config sync require applicable explicit authorization. A start request is
  not a clean/restart request; a diagnostic procedure is not recovery authority.

Use Ollmo when requested or when an already-available capability materially helps
a bounded task's locality, privacy or result. Respect a chosen provider/tool and
local-only constraints; prefer deterministic tools for deterministic work. Do not
make Ollmo the default Codex model provider or modify Codex configuration without
explicit configuration scope. Tool availability does not authorize subagents.
For an authorized delegated subtask, explicitly pass this skill, bounded inputs,
expected outputs and completion criteria; do not assume inherited skill context.

## Resolve resources before operating

Except in downstream-provider mode above, resolve the active checkout:

1. A valid Ollmo checkout at $OLLMO_HOME.
2. The current workspace if it contains ollmo, ollmo_webserver.py and ollmo_core/.
3. Ask for the checkout path if neither identifies it.

Do not search arbitrary parents or treat saved builds as active without selection.
All paths prefixed with `<ollmo>/` below resolve from that checkout, not the skill
directory or an unrelated current directory. Packaged links under references/
resolve relative to this installed skill. This keeps the skill usable outside the
checkout without embedding personal absolute paths. If a required checkout file is
missing, report the missing contract instead of inventing it.

The implemented control-plane default is `http://127.0.0.1:5001`. Ordinary
client helpers support `OLLMO_WEB_BASE`; use the configured supported base when
applicable. A base override is not permission to bypass loopback/credential
restrictions on privileged operator actions. Prefer the control plane to raw
backend ports. Durable aliases `external:codex`, `codex:auto` and `codex_cli`
are integration identities, not model names.

## Observe with freshness

Start with permitted existing evidence: `<ollmo>/model_ports.json`,
`<ollmo>/state/runtime_status.json`, or the cached control-plane surfaces
`/api/runtime_manifest`, `/api/running_instances`, `/api/backend_fabric`,
`/api/ghost`, `/api/runtime_status` and `/api/ghost_preferences`.
Do not assume Ollmo is running or stopped.

These endpoints are passive by default. Cached readiness and model/learning
selections have observation times; they are not fresh process probes. Explicit
`refresh=true` may probe and write status, but does not authorize lifecycle,
artifact creation or response execution. For a strict no-write/no-model task,
use existing files or permitted passive surfaces; no refresh or computed preview.

The runnable catalog `./ollmo ctl models list --json --runnable-only` describes
startable options, not running capacity. Listing is not authorization to start.
Read-like ollmoctl commands do not recover by default; use
`--recover-control-plane` only for authorized control-plane recovery.
Registry reads already default to `prune=False`; unexpected pruning is a source
defect to report, not permission to repair through cleanup.

Prefer current process/port/backend evidence over stale readiness labels.
Degraded, busy, timeout, cooldown and provider-family warnings are advisory unless
hard runtime evidence proves unavailability; do not convert them into provider
bans, failed obligations or graph-repair authority.

## Handle localhost denial without recovery

One failed HTTP request, including connection refusal, is inconclusive until
permitted evidence is checked. A clear `EPERM`, socket denial, or a connection
that works in the user's Terminal but is denied here indicates a tool boundary.

Do not route around it through raw ports, other forbidden tools, lifecycle,
registry edits or config sync. Answer observation questions from available files
with their freshness and the HTTP limitation stated. For execution, stop after
the first clear sandbox denial and provide the same bounded request as a
Terminal-safe command for the user.
Do not reinterpret tool denial as a model failure. If a response exists, preserve
its lifecycle/output/artifact truth. Use direct HTTP when it works; Terminal
fallback is not a mandatory detour.

## Execute and observe the same work

For route preview, request shapes or a concrete execution task, read the relevant
section of [request and result recipes](references/ollmo-contract.md). It is not
a second always-loaded manual.

Use `POST /api/responses` only for actual execution. Use Ghost when route/phase
formation or dependencies need it; use an exact runtime-derived `instance_id`
for a resolved target, or `capability` when identity does not matter. Ghost-first
execution uses `ghost_route: true`, `prompt`, a stable `conversation_id`, and
the non-empty persisted preferences object as `ghost_preferences` when available.
Do not add old history unless the current conversation contract needs it.

If `lifecycle_state` is `late_fill_pending` or `late_fill_running`, observe the
existing response id. Poll `GET /api/responses/<id>?view=status`; default GET is
a bounded UI view. Fetch `?view=truth`, `full` or `raw` only when exact detail
is needed and without `compact=true`. Status/debug views are not artifact bytes.

Before claiming completion, inspect `outputs`, `artifacts`, `response_frame`,
`lifecycle_state`, `runtime.graph_closure_review`, `surface_state` and
`late_fill`. Compatibility `status=completed` or `output_text` alone is
insufficient. The OpenAI-shaped interface is a limited compatibility subset;
read `<ollmo>/docs/RESPONSES_CONTRACT.md` before relying on standard client
tool/state/role semantics.

## Essential result and repair safeguards

- Model prose, code blocks and a generated prompt do not prove a requested file,
  image or audio exists. Final saved files own bytes; frames/closure own owed and
  fulfilled work; the artifact registry owns lookup/provenance.
- Public deliverables come from canonical `outputs` and artifact truth. UI,
  history, SSE, compact views and repair dossiers are projections or diagnostics.
  Report discrepancies during observation; repair/hydration writes need applicable
  authority. Do not change runtime truth to match a reduced surface.
- Candidates, reserved possibilities and advisory reviews are not obligations
  until promoted. Prior artifacts/history are reference evidence only when the
  current turn refers to them; preserve/no-regenerate intent must keep exact
  predecessor evidence without creating replacement producers.
- Every branch consumes its accepted local payload and exact dependency evidence.
  Missing dependencies require repair, not root-prompt replay. Target-bound text
  repair fixes its evidenced file/defect, not unrelated design or a sibling file.
- Frozen parents remain immutable. Runtime owns additive patch validation and
  successor lineage; additive authority does not authorize rebase. Shadow/stage
  records are not executable work. Preserve exact identities, evidence,
  preservation proofs and operator gates before an authorized transition.
- Linked artifact sets close only when actual saved dependencies resolve.
  Guessed paths, placeholders or a smaller artifact count do not fulfill an exact
  request without recorded waiver/supersession.
- HTTP success, a WAV header or non-empty audio does not establish physical audio
  integrity. `tts_audio_integrity_evidence` and digest-bound `tts_semantic_source`/
  `tts_stt_semantic_evidence` serve distinct guarantees. Do not inject expected
  text into STT or silently add an unrequested STT branch.
- Bind multimodal evidence to the exact producer/consumer. Vision cannot inspect
  sibling media instead; structured joins must preserve count, labels, required
  fields and one-to-one refs. Invalid output remains inspectable, not rewritten
  into success.
- Preserve named resource/security limits, retry gates and backend compatibility
  safeguards. Cleanup/archives affect durable evidence and require their own
  scope; neither is an implied repair or Ghost-forget operation.

## Load specialized depth only for the affected task

The following are checkout resources, resolved using the rule above. Read the
relevant section, not every document on each invocation.

| Task | Authoritative resource and useful section/search terms |
| --- | --- |
| Artifact fulfillment, terminal projection, multimodal evidence, bundles | `<ollmo>/docs/RESPONSES_CONTRACT.md`: Artifact fulfillment, terminal, TTS, bundle; `<ollmo>/docs/TRUTH_SOURCES.md` for competing owners |
| Repair, learning, promotion or rebase | `<ollmo>/docs/CONTROL_KNOBS.md`: Graph Repair, Reviewed Graph Rebase, redraw, accepted learning; `<ollmo>/docs/PATTERNS.md` for branch-local and preservation invariants |
| Cleanup/archive or evidence retention | `<ollmo>/OLLMO_FOR_AGENTS.md`: First Commands (clean/archive) and Context And Learning Contract (retention); `<ollmo>/docs/TRUTH_SOURCES.md`: Cleanup Contract |
| Routing policy explanation/debugging | `<ollmo>/docs/GHOST_ROUTER.md`; read `<ollmo>/GHOST.md` only when its runtime policy is relevant |
| Engineering and test selection | `<ollmo>/AGENTS.md`, then `<ollmo>/docs/TESTING_PROTOCOL.md`; injected policy is behavioral code |
| Monitoring | `<ollmo>/skills/ollmo-run-monitor/SKILL.md`; helper execution writes monitor state |
| Downstream Codex isolation or provider projections | `<ollmo>/OLLMO_FOR_AGENTS.md`: External Integrations |

Do not detach Ollmo's response-bound Codex child: response-owned cancellation,
time budget and result collection remain authoritative. Global model selection
is not inherited through that child's ignored user configuration. Do not modify
GHOST.md, lifecycle code or integrations merely to resolve an operator question.
