# Ollmo

Ollmo is a local-first AI runtime substrate that turns requests into truthful,
continuable work state instead of treating model prose as proof of success.

**Status:** `0.1.0` release candidate — experimental, usable, and still
evolving within the `0.x` series.

**Conceived, designed, created, built, and developed by
[@fl0ri0](https://github.com/fl0ri0).**

**Independent project.** Ollmo is built and maintained by one person, not a
company, research lab, or funded team. It is human-led and AI-enhanced — a
vibe-coded project in the literal, accountable sense. Product direction and
release decisions remain human; AI tools helped explore, implement, test, and
document the system. It has grown for more than a year from an early prototype
whose small changes were still copied by hand into the current runtime
substrate. This release does not imply institutional backing, a formal security
audit, or a support organization.

Ollmo combines a local browser interface and control plane with local model
backends, Ghost routing, durable response frames, history, and materialized
artifacts. Its core loop is:

```text
Intent -> possibility space -> relevance -> promoted contracts -> runtime work -> review -> frozen state
```

A response is a frozen truthful moment of work: what was requested, what
became owed work, what runtime evidence exists, what remains pending or
blocked, and which outputs or artifacts can be referenced later.

## Project, Citation, and Collaboration

If Ollmo, its state-substrate architecture, possibility space and selective
promotion into obligations, work-graph model, recorded pre-freeze working
state, intent-preserving block-resolution principle, cross-instance Late Fill
and branch-local output binding within one canonical response, evidence-gated
closure review, durable response-frame design, or evaluation artifacts
contribute to research or a published system, cite the software and repository using
[`CITATION.cff`](CITATION.cff).

The intent-preserving block-resolution principle means that a block is not
permission to rewrite the request. Its solution is its own verified
resolution, handled gently through the substrate: keep intent anchored, keep
the obligation visible, record the runtime evidence, choose the right-sized
state transition, and freeze truthful state rather than the desired story.
See [Principle 10](docs/PRINCIPLES.md#10-resolve-blocks-by-preserving-intent).

Within one canonical response, Ollmo can route different promoted branches
independently to compatible model instances and capabilities, materialize
their distinct outputs through Late Fill, and bind each evidenced result back
into the same response state. This is capability- and availability-dependent;
it does not imply that every response uses more than one instance.

For research collaboration, replications, talks, panels, or work around Ollmo,
use the public contact methods on
[@fl0ri0's GitHub profile](https://github.com/fl0ri0), with no guaranteed
response time. Ollmo 0.1.0 is published for inspection, use, research, and
independent forks. Public issues and pull requests are not currently solicited
or promised review; [`CONTRIBUTING.md`](CONTRIBUTING.md) records that capacity
boundary.

Useful public context:

- [State Substrate Architecture](docs/diagrams/ollmo-state-substrate-architecture.html)
- [Project Participation and Capacity](CONTRIBUTING.md)

## Supported Environment

The primary tested environment is a recent macOS release on Apple Silicon with
Python 3.11 or newer. Local capabilities require their corresponding Ollama,
MLX, or llama.cpp backend and model. Windows, Linux, Intel Mac, remote hosting,
and multi-user operation are outside the 0.1.0 support promise.

## Install and Start

Extract the release archive, enter its root, and run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./ollmo start
```

`./ollmo start` can create the virtual environment and install requirements
itself, but the explicit steps above make installation failures easier to
diagnose.

Open the local interface:

```bash
./ollmo dashboard
```

The dashboard is served directly at `http://127.0.0.1:5001/`; `/dashboard`
remains a compatibility alias. The separate static project page is available
at `http://127.0.0.1:5001/site/` while Ollmo is running, or by opening
`site/index.html` directly from the checkout. The self-contained `site/`
directory is also the GitHub Pages publication artifact. It uses only
relative, bundled assets and is not the live control surface. Its contents can
be placed at the root of a dedicated Pages branch and activated manually; no
publication workflow is included.

Useful lifecycle commands:

```bash
./ollmo status
./ollmo shutdown
```

To inspect locally runnable models and start one chat model explicitly:

```bash
./ollmo ctl models list --json --runnable-only
./ollmo ctl start --model "<model-id>" --backend "<backend>" --capability chat --json
```

Use an identifier and backend reported by the first command. Ollmo does not
bundle model weights.

## Optional: Use Ollmo from Codex

The release includes a dedicated Codex skill that teaches Codex how to inspect
Ollmo runtime truth, use Ghost routing safely, and work with canonical
responses and artifacts. During broader Codex work, the skill also makes
already-running local text, image, vision/OCR, speech, embedding, and
multimodal capabilities available as optional bounded tools while Codex
remains the supervisor.

From the extracted Ollmo directory:

```bash
OLLMO_CODEX_SKILLS_DIR="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$OLLMO_CODEX_SKILLS_DIR"

if [ -e "$OLLMO_CODEX_SKILLS_DIR/ollmo" ]; then
  printf 'Ollmo skill already exists at %s\n' "$OLLMO_CODEX_SKILLS_DIR/ollmo"
else
  cp -R skills/ollmo "$OLLMO_CODEX_SKILLS_DIR/ollmo"
fi
```

Restart Codex, open the extracted Ollmo directory as the active workspace, and
then mention Ollmo normally or invoke `$ollmo` explicitly. When working from a
different workspace, set `OLLMO_HOME` to the extracted Ollmo root or provide
the checkout path when the skill asks.

Installing the skill does not start Ollmo, change Codex model-provider
settings, or enable cloud routing.

## Optional ChatGPT Cloud Input

Ollmo runs locally by default. It can optionally use ChatGPT as an external
target through Codex after the user reviews the cloud disclosure and explicitly
enables the connection. Ollmo first looks for the Codex executable bundled with
the ChatGPT app and then falls back to a separately installed `codex`
executable. The integration reuses the login owned by that executable; Ollmo
does not read, copy, or store the credentials and does not set a fixed model.

Once enabled, ChatGPT appears as a clearly marked cloud tab, as an explicit
Ollmo target, and as a primary or fallback preference. It can be turned off
again at any time. Only the prompt, context Ollmo promotes for the current
turn, and local files or Ollmo artifacts explicitly selected for that turn are
handed to ChatGPT. That text and those selected bytes leave the machine and
are processed by OpenAI.

Recognized images use Codex's native image-input path. Other selected regular
files are copied into a temporary working directory configured read-only for
the Codex run. One request can include at most 5 files, no more than 100 MiB per
file and 250 MiB in total. URLs, folders, and symbolic links are rejected.
ChatGPT returns text through this route; Ollmo does not guarantee that every
regular-file format can be interpreted semantically by the selected model and
its available tools.

The ChatGPT target never appears as a running local instance and has no Start,
Stop, Pull, or Delete actions. Direct tab turns are ephemeral and independent;
Ollmo can promote bounded relevant context for a referential turn, but does not
create a persistent provider session. Model selection is automatic,
and the exact GPT variant is not exposed to Ollmo; model self-descriptions are
not runtime proof. Direct API-key management and other external providers are
outside the 0.1.0 contract.

## Release and Safety Notes

- [Release Scope](docs/RELEASE_SCOPE.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Third-Party Components](THIRD_PARTY_NOTICES.md)

Ollmo's own source code and project documentation are licensed under the
[Apache License 2.0](LICENSE), unless a file says otherwise. Bundled and
optional third-party components retain their own licenses; see
[`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The public repository is <https://github.com/fl0ri0/ollmo>. Release tooling
builds and verifies local archives; it does not publish or upload them.

Canonical artifact storage lives under `artifacts/`: generated files are
written to `artifacts/`, saved request inputs to `artifacts/inputs/`, and
audit/report-style outputs to `artifacts/audits/`.

## Runtime Architecture

Ollmo separates model proposals from runtime authority. Ghost anchors current
intent and derives candidate work. Promotion turns selected candidates into
explicit obligations. The resolver and Late Fill materialize those obligations
through compatible runtimes, and Closure Review compares the result with
runtime and artifact evidence before the state can freeze.

The mutable `working_frame` records the fluid pre-freeze state, including the
possibility space, candidate graph, promotion review, branch-local work, and
closure state. The frozen `response_frame` preserves the resulting request,
route, runtime evidence, outputs, artifacts, and unresolved work. Canonical
frames are appended to `state/response_frames/responses.jsonl`; normal bounded
wire views expose handles and content-addressed references, while explicit
truth views restore the exact referenced state.

Current code ownership is split across:

- `ollmo_webserver.py` for the Flask API/UI composition root;
- `ollmo_server/` for request, routing, execution, transport, and Late Fill
  runtime owners;
- `ollmo_services/` for response frames, artifacts, history, inference, and
  related product services;
- `ollmo_g/` for Ghost's current-turn semantic and graph layer;
- `ollmo_orchestration/` for mutable pre-freeze working state;
- `ollmo_runtime/` and `ollmo_core/` for backend lifecycle and shared runtime
  truth;
- `ollmo_integrations/` for external-client adapter infrastructure and the
  current Codex integration boundary; and
- `static/ui/` plus `ollmo_webUI.html` for the browser control surface.

Reference docs:

- [Core Contracts](docs/CORE_CONTRACTS.md)
- [Canonical Glossary](docs/CANONICAL_GLOSSARY.md)
- [Canonical Stack](docs/CANONICAL_STACK.md)
- [Principles](docs/PRINCIPLES.md)
- [Patterns](docs/PATTERNS.md)
- [Architecture Map](docs/ARCHITECTURE_MAP.md)
- [Ghost Runtime Policy](GHOST.md)
- [Ghost Self-Alignment](docs/GHOST_SELF_ALIGNMENT.md)
- [State Substrate Architecture](docs/diagrams/ollmo-state-substrate-architecture.html)

## `ollmoctl`

`ollmoctl` is the common CLI adapter layer over the current Ollmo Flask control plane.

Use it when you want external or user-built wrappers, shells, and scripts to
call one stable command surface instead of reaching into startup scripts or
raw ports. The bundled Codex skill is the validated 0.1 agent integration.

Entry point:

- `python3 scripts/ollmoctl.py ...`

Base URL resolution:

- `--base-url http://127.0.0.1:5001`
- or environment variable `OLLMOCTL_BASE`
- fallback: `OLLMO_WEB_BASE`
- default: `http://127.0.0.1:5001`

Local runtime recovery order:

- `ollmoctl` talks to the current Ollmo Flask control plane on `127.0.0.1:5001`.
- First dependency: if local runtime commands hit `connection refused` on `5001`, the control plane is down.
- Read-like `ollmoctl` commands such as `instances list`, `models list`, `ghost`, and `responses get` do not recover/start the control plane by default. Pass `--recover-control-plane` only when that recovery behavior is intentional.
- Mutating or lifecycle-oriented `ollmoctl` commands may auto-attempt a one-shot local recovery by starting `ollmo_webserver.py` and retrying once for the default local base URL.
- If `5001` is up but `instances list --json` is empty, start the required model with `ollmoctl start ...`.
- `doctor runtime` stays diagnostic and reports the broken state instead of auto-starting anything.
- If the one-shot recovery still fails, run `./ollmo start` for the full local stack and inspect `logs/flask_webserver_auto.log`.

Core examples:

```bash
python3 scripts/ollmoctl.py models list --json
python3 scripts/ollmoctl.py instances list --json
python3 scripts/ollmoctl.py ghost --json
python3 scripts/ollmoctl.py doctor runtime --json
python3 scripts/ollmoctl.py start --model mlx-community/Qwen3.5-27B-4bit --backend mlx --capability chat --json
python3 scripts/ollmoctl.py send mlx-community__Qwen3.5-27B-4bit-mlx-11502 "hello"
python3 scripts/ollmoctl.py send mlx-community__whisper-large-v3-mlx-mlx-11501 --file "/absolute/path/to/audio.mp3" --json
python3 scripts/ollmoctl.py send mlx-community__Qwen3.5-27B-4bit-mlx-11502 "hello" --truth-json
python3 scripts/ollmoctl.py responses get resp_abc123
python3 scripts/ollmoctl.py responses get resp_abc123 --json
python3 scripts/ollmoctl.py responses get resp_abc123 --truth-json
python3 scripts/ollmoctl.py events list --category infer --status started --json
python3 scripts/ollmoctl.py events tail --category infer --instance-id mlx-community__whisper-large-v3-mlx-mlx-11506
python3 scripts/ollmoctl.py wait mlx-community__whisper-large-v3-mlx-mlx-11506 --json
python3 scripts/ollmoctl.py history infer --limit 20 --json
```

Most important command:

- `send`
  - targets the canonical `/api/responses` endpoint
  - lets Ollmo decide internally whether the request becomes chat or infer/capability dispatch
  - defaults to a long timeout suitable for STT/TTS and other long-running infer requests
  - `--json` performs one bounded POST and prints its public wire envelope
  - `--truth-json` performs that POST, fetches the same id through exact `view=truth`, and prints a normalized canonical summary
- `responses get`
  - reads canonical `/api/responses/<id>?view=truth` lookup state
  - default output summarizes `lifecycle_state`, `status_semantics`, frame identity, output provenance, work-tree authority, and durability
  - `--json` preserves the raw canonical truth-view payload, while `--truth-json` emits only the normalized response-truth summary for scripts
  - `--json` and `--truth-json` are mutually exclusive on both `send` and `responses get`
- `wait`
  - polls Ollmo event history until the latest matching request for an instance reaches a terminal state
- `events tail`
  - polls and prints new matching event log entries as they appear
- `ghost`
  - reads Ollmo's runtime-intelligence summary with recommended defaults, current issues, and recovery hints

## `ollmo_g`

`ollmo_g` is the package boundary for the self-describing runtime-intelligence layer inside Ollmo. Its runtime identity remains `ollmo-ghost`.

It does not orchestrate agents. It explains and uses the runtime:

- what is running
- which instance is the current best default for each capability
- what problems are active
- which recovery commands should be tried next
- which live capability/instance best fits an Auto request according to merged runtime truth rather than a hardcoded provider preference

Current Ghost/Auto behavior:

- `Auto` routes from merged control-plane truth, not a fixed provider table
- Ghost, resolver transformation, control-hint filling, and late fill are separate responsibilities:
  - Ghost chooses the next truthful phase and compatible runtime target
  - Ollmo can then fill truthful request fields such as language, speaker, format, style, or image aspect controls for that chosen target
  - downstream phases can continue through the resolver + late fill under the same request/response identity
  - local backend model calls materialize selected phases or branches; runtime code decides fulfillment from graph, slot, output, artifact, and status truth
- the request phase graph is now the shared substrate for Ghost-owned requests:
  - every Ghost-owned request freezes into a request phase graph
  - `candidate_graph` records possible outputs, workload tasks, context, references, evidence, repairs, continuations, and learning hints
  - `promotion_review` decides what becomes executable owed work and what remains reserved, waived, rejected, stale, or merely possible state
  - branch-local workload tasks carry focused payloads, prompts, dependencies, artifact refs, evidence, output contracts, visibility, review criteria, and freeze state
  - deterministic `review_criteria` stay runtime/closure checks; only explicit `semantic_review_criteria` or non-deterministic qualitative criteria require semantic review
  - the graph is frozen in intent and fluid in state
  - plain chat may end at phase 1
  - image/audio materialization defaults to `chat -> downstream branches`
  - downstream phases remain explicit instead of being flattened into one capability
- later branches should consume branch-local task truth rather than replaying the full root prompt, and missing generated input artifacts should become `repair_dependency_chain` instead of blind same-branch retry
- canonical `/api/responses` now runs one bounded pre-freeze closure review against the frozen graph/work-tree/artifact truth:
  - the review is exposed as `runtime.graph_closure_review` plus developer diagnostics where applicable
  - it should continue resolver/late fill work for still-missing frozen obligations
  - it must not reinterpret the user request or invent new semantic goals
- live Ghost route context is intentionally short and direct:
  - current turn stays dominant
  - fresh turns normally use `current_turn_only` context strategy
  - recent thread context is injected only for clearly referential or continuation-like turns
  - when that thread context is needed, Ghost sees one answered prior user turn plus its direct assistant reply window
  - artifact continuity is output-centered: all artifacts from the last relevant assistant artifact message, then latest-by-type fallback
  - old tool calls and old artifacts do not become new intent just because they happened earlier
  - explicit stable preferences are explicit-only; live routing does not carry a separate derived `memory` block
- Auto preview and final execution share the same resolver/control-hint/runtime path so the UI sees the same required-field truth the backend will execute
- Auto/Ghost preview and route selection are read-only planning surfaces. They must not start models, load providers, or turn route choice into runtime load feedback. Model lifecycle starts require an explicit lifecycle source such as a frontend play/start action.
- embedding helpers remain internal Ghost tools; they can attach routing-context hints and narrow tie-break signals for anchored follow-ups, and those audits remain visible in runtime metadata
- the old single-turn route-rating and learned-policy system is retired from the live runtime path
- explicit low-level direct contracts such as direct `instance_id` requests or `batch_prompts` remain explicit exceptions; they are not the normal Ghost path
- Ghost now exposes file-backed semantic roles as advisory decision lenses. Any valid `ollmo_g/semantic_roles/*.md` role definition becomes part of the role catalog.
- `ghost_mode` remains only a bounded API-edge compatibility hint (`repair`, `worker`, `explorer`, `improviser`) and is immediately projected into `semantic_role_profile`; it is not the internal thinking model.
- `semantic_role_profile` can travel through request meta, route preview, working frames, and canonical Responses runtime payloads as advisory orientation only.
- `GHOST.md` is the primary runtime-policy anchor Ghost uses for current-turn interpretation; the backend now injects it into both Ghost routing decisions and Ghost-owned user-facing chat turns, while `OLLMO_FOR_AGENTS.md` remains the operator/client guide
- visible assistant text must not claim `saved locally`, `artifact created`, or similar file/artifact success unless runtime truth actually contains that saved output

Surfaces:

- `GET /api/ghost`
- `GET /api/agent_contract`
- `python3 scripts/ollmoctl.py ghost`
- [GHOST.md](GHOST.md)
- [OLLMO_FOR_AGENTS.md](OLLMO_FOR_AGENTS.md)

Frontend note:

- the Responses workbench can now use `Auto` to route prompts and file-backed requests by capability instead of forcing one fixed target instance
- the UI label is `Auto`, while the backend currently keeps the internal `ghost_route` / `ghost_preview` names for this routing path
- Ghost `primary_target` / `fallback_target` preferences remain Ghost and chat-routing hints; they do not override non-chat execution when the request resolves to image, TTS, or another incompatible capability
- Explicit frontend play/start is the user lifecycle path for starting an additional instance. The frontend sends `start_source: "frontend_button"` and sends `force_start: true` only when deliberately starting another same-model instance; Ghost, route preview, late fill, and backend automatic paths must not use `force_start`.

## Canonical Responses API

Ollmo now exposes one canonical execution endpoint on top of the existing route split:

- `POST /api/responses`
- `POST /v1/responses`

Use this when you want one stable request/response language for execution calls. The older specialized routes still exist:

- `/api/chat`
- `/api/infer`

but the canonical path is now `responses`, and first-party UI flows should stay on it.

This is the official external contract for wrappers, agents, CLIs, editors, and new integrations.

Prefer `responses` even for file-backed, OCR, image, STT, and TTS work. `/api/responses` now executes shared internal helpers directly; external callers should not need to pick `/api/chat` or `/api/infer` unless they intentionally want the older compatibility surface.

Reviewed redraw remains part of the same intent-aligned scope ladder. The lower bounded additive repair rungs stay autonomously active under the default-deny graph-repair policy; partial-subtree and full-successor rebase are the higher-risk rungs, with non-mutating `shadow` as their product default rather than a separate shadow layer. Read their canonical evidence gates at `GET /api/graph_rebase/readiness`. The report is observer-only and does not promote work.

An operator advances one exact proposal through `POST /api/responses/<response_id>/graph_rebase/operator` in strict order: `adjudicate`, then durable audit-only `stage`, then gate-controlled `authorize_partial`. The last action is compare-and-swap bound to the finalized response/frame/proposal/base/candidate/class identities and joins authority only from Ollmo's trusted operator registry. Full durable state and bounded observation truth must identify the same latest frame, and current root truth is rechecked at replay, apply, persistence, and Late Fill. A successful partial authorization appends one branch-local successor frame under the same response id before scheduling its exact Late Fill work; it never rewrites the parent or replays the root prompt through another prompt carrier. Explicit `OLLMO_GRAPH_REBASE_AUTONOMY=off` blocks the `stage` and `authorize_partial` transitions; evidence-only `adjudicate` remains available because it grants no rebase authority. The product stays in `shadow` while evidence gates are not green, and full successor rebase remains non-executable under safe partial v1.

The mutating endpoint is dormant unless startup explicitly supplies both a 32-or-more-character `OLLMO_GRAPH_REBASE_OPERATOR_TOKEN` and an exact `OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY`. Calls must present the token as a Bearer token or `X-Ollmo-Graph-Rebase-Operator-Token` and the configured identity as `X-Ollmo-Graph-Rebase-Operator`; neither value is persisted or inherited by model subprocesses. Runtime itself produces replay confirmation by deterministically revalidating the frozen proposal. Trusted-only or runtime-only stage halves do not count toward promotion. A no-proposal `false_negative` adjudication is evidence-only and uses the exact sentinel `expected_proposal_id=no_formal_proposal`; it cannot become stage or execution authority. It remains historical truth, but a later exact replay-verified `useful_proposal` adjudication may append `resolves_record_id=<false-negative-record-id>` so only unresolved false negatives block promotion.

Current scope:

- supports direct targeting with a running `instance_id`
- also supports routed/default targeting layers such as Auto/`ghost_route`, wrapper alias/profile, or canonical `capability` when the caller or backend chooses the target
- accepts one normalized request body with `input`, optional `instructions`, and optional file context (`file_path` or multipart `file`)
- keeps explicit `input_artifacts` separate from explicit `reference_artifacts`
- supports `stream=true` with Responses-style `text/event-stream` events
- returns one normalized response envelope with:
  - `outputs`
  - `status`
  - `mode`
  - `instance_id`
  - `model`
  - `output_text`
  - `artifacts`
  - `warnings`

For successful non-streaming calls, that envelope is serialized from the compact indexed Response Frame after canonical persistence. Normal POST/default/UI/status/debug results are byte-budgeted to at most 8 MiB for the complete serialized outer envelope and never hydrate sidecars. Compaction is based on bytes rather than a fixed item limit: ordinary 5,000-character text and 65 tiny records remain inline when they fit, while strings from 256 KiB and collections from 1 MiB may become a bounded preview plus exact count/length, byte size, SHA-256, and an adjacent content-addressed `*_snapshot_ref`. Public lifecycle, frame identity, outputs/slots, artifact handles, and effective content-SHA refs remain on the wire; hydrated Runtime, Working Frame, and duplicate full-frame bodies do not.

`GET /api/responses/<id>?view=debug` adds bounded Closure/rebase observation evidence. `view=full`, `view=raw`, and `view=truth` recursively restore exact CAS-backed canonical truth and can be larger than 8 MiB. Canonical hydration validates every authoritative ref recursively and fails with HTTP 409 when a ref is missing, malformed, or corrupt. This keeps full truth by reference without semantic loss. Artifact-bundle operations likewise resolve exact CAS truth and fail closed rather than exporting an incomplete preview.

Example:

```json
POST /api/responses
{
  "instance_id": "x/flux2-klein:latest-1",
  "input": "seven frogs posing like a family portrait on the forest floor"
}
```

Streaming note:

- canonical streaming currently uses Responses-style SSE over the finalized normalized response payload; the final serialized SSE event, including its outer wrapper, obeys the same 8 MiB ceiling
- this is already useful for one client language and UI compatibility
- for chat-capable Ollama/MLX runtimes, canonical streaming now forwards real backend deltas
- non-chat capabilities still use completed-response style payloads/synthetic SSE where appropriate
- the main UI chat flow now uses canonical responses streaming for ordinary chat turns

## Token Limits

Ollmo should not invent token limits on behalf of models.

- app-side guessed token limits have been removed from the active model-manager/UI path
- available model payloads should only expose token limits when a runtime or provider actually reports them
- downstream sync layers should omit unknown `contextWindow` / `maxTokens` instead of filling in guessed defaults

## Startup Policy

Use `./ollmo start` as the operator-facing entrypoint. It delegates to `./start_multi_models.sh` and starts the canonical runtime/UI stack.

- Starts model management and the current Flask UI/API.
- Starts MLX support when available.
- Leaves external-client provider projections untouched; projection is an explicit operator action.

The active startup flow is the runtime/UI stack only. External provider projection is an explicit operator action, not part of start/stop/restart.

## Clean Repo State

When you want a local green-field reset without touching anything outside this repo, use:

```bash
./ollmo clean
```

This is a repo-local maintenance helper implemented by `./clean_repo_state.sh`. It does not edit `~/.codex` or any other external config.

Default cleanup removes repo-local generated/runtime ballast:

- contents under `artifacts/*` buckets while preserving the standard empty artifact bucket directories, including `artifacts/bundles/`; missing standard directories are created only as a fallback
- `logs/*`
- `state/chat_history/*`
- `state/events.jsonl`
- `state/infer_history.jsonl`
- `state/artifact_registry.jsonl`
  - current artifact-centered backing ledger used by artifact dossiers, provenance, metadata, and enrichments
- `state/generated_image_provenance.jsonl`
  - legacy/generated-image-specific provenance ledger when present
- `state/response_frames/*`
  - compact response ledger, `current_index.json`, and sidecar snapshots under `snapshots/`
- `state/runtime_status.json`
- Python/test cache files and directories such as `__pycache__/`, `.pytest_cache/`, `*.pyc`, `*.pyo`, `.coverage`, `htmlcov/`, `.mypy_cache/`, and `.ruff_cache/`

Default cleanup preserves:

- `model_ports.json`
- `state/llama_cpp_catalog.json`
- `state/ghost_preferences.json`
- `state/ghost_compiled_memory.json`
- `state/ghost_compiled_memory.md`
- `state/self_learning/`

If you want a true empty repo-local state, combine:

```bash
./ollmo clean --forget-ghost --reset-registry --reset-llama-catalog
```

The shorter equivalent is:

```bash
./ollmo clean --full
```

For the current recommended archive-first rotation into a clean live state, use:

```bash
./ollmo archive --full
```

`archive --full` snapshots protected Ghost state when present, but preserves the active checkout copies. Protected Ghost state includes `state/ghost_preferences.json`, `state/ghost_compiled_memory.json`, `state/ghost_compiled_memory.md`, `state/self_learning/`, and the accepted-learning policy snapshot under `state/self_learning/accepted_policy_snapshot.json`.

Preview it first with:

```bash
./ollmo archive --full --dry-run
```

Optional deeper reset flags:

- `./ollmo clean --full`
  - equivalent to `--forget-ghost --reset-registry --reset-llama-catalog`
- `./ollmo archive --full`
  - archive-first cleanup plus registry/catalog reset, while preserving protected Ghost state in the active checkout
- `./ollmo clean --reset-registry`
  - rewrite `model_ports.json` to `[]`
- `./ollmo clean --forget-ghost`
  - remove Ghost preferences and any retired Ghost archive residue

If you want to reset Ghost's local event/log state without wiping the durable frame ledger, use:

```bash
python3 scripts/ollmoctl.py ghost --reset-learning-state --json
```

This archives the prior Ghost event/log state plus `logs/flask_webserver.log` under `state/ghost_learning_archives/<timestamp>/`, recreates fresh `state/events.jsonl`, removes any retired learned-policy or compiled-memory residue if it still exists, and intentionally preserves `state/response_frames/responses.jsonl`.

Ghost learning reset is deliberately separate from `archive --full` and `clean --forget-ghost`. If the reset-learning command removes `state/self_learning/`, it archives that directory first.

- `./ollmo clean --reset-llama-catalog`
  - remove `state/llama_cpp_catalog.json`
- `./ollmo clean --dry-run`
  - preview the cleanup without deleting anything

Use this when the repo has accumulated logs, artifacts, history, or cache ballast and you want a predictable local reset before starting again.

macOS/Chrome note: `clean` and `archive` preserve the standard `artifacts/` bucket structure. Chrome can keep or lose local folder permissions across that rotation. If a local `artifacts/bundles/.../index.html` opens but bundle CSS or images fail with `net::ERR_ACCESS_DENIED`, quit Chrome and re-grant Chrome access under macOS Privacy & Security (`Files and Folders` / `Desktop Folder`, or `Full Disk Access`). If needed, reset the Desktop Folder permission and approve it again:

```bash
tccutil reset SystemPolicyDesktopFolder com.google.Chrome
```

To make macOS ask again on the next Chrome `file://` bundle access, include the opt-in cleanup flag:

```bash
./ollmo archive --full --reset-chrome-file-access-prompt
```

Live generated/operator file buckets now belong under `artifacts/`, especially:

- `artifacts/inputs/` for saved request inputs
- `artifacts/audits/` for audit/report-style outputs as part of the basic artifact skeleton recreated by the reset helper
- `artifacts/bundles/` for post-response, local-openable response artifact bundles
- `artifacts/settings/` for explicitly promoted reusable settings artifacts from response-frame control snapshots

If you want the same live cleanup, but do not want to lose the cleaned runtime/generated data immediately, use:

```bash
./ollmo archiv
```

`archiv` and `archive` both point to the same archive-first cleanup mode. It copies the `artifacts/` tree into the archive, archives the other useful cleaned ballast, and stores it under:

- `.ollmo_archiv/<timestamp>/`

and then applies the same live cleanup to `logs/` and the volatile `state/` paths. For `artifacts/`, live cleanup removes generated contents while preserving the standard bucket directories, including `artifacts/bundles/`. Missing standard directories are created only as a fallback if they were manually deleted or do not exist yet.

This is intentionally only a hidden repo-local data store, not a live runtime path.

Important distinction:

- `./ollmo clean`
  - hard cleanup, no archive
- `./ollmo archiv`
  - archive-first cleanup for temporary retention
- `./ollmo archive`
  - archive-first cleanup for temporary retention

Notes:

- `archiv` / `archive` can also be combined with `--forget-ghost`, `--reset-registry`, or `--reset-llama-catalog`; if explicit removal/reset flags are used, the affected files are archived or snapshotted first and only then cleaned/reset.
- `archiv` / `archive --full` is the concise archive-first rotation for runtime/generated ballast plus registry/catalog reset. It is not a Ghost-forget or self-learning reset shortcut.

## Ollama Lifecycle Ownership

Use exactly one owner for `ollama serve` at a time.

- If Ollmo manages your model lifecycle, stop the Homebrew background service first:
  - `brew services stop ollama`
- Then start Ollmo normally and let Ollmo launch the `ollama serve` processes it needs.
- Do not keep both Homebrew `ollama` service and Ollmo-managed `ollama serve` processes active in parallel.
- `./start_multi_models.sh` now checks this and stops early with a clear fix message unless you explicitly override it via `OLLMO_ALLOW_BREW_OLLAMA_SERVICE=1`.

Why this matters:

- parallel `ollama serve` owners can leave stale processes running after code changes,
- restart/stop logs become misleading,
- environment-dependent fixes can appear to "not work" because requests still hit an older process.

Homebrew path note:

- `/opt/homebrew/bin/ollama` and `/opt/homebrew/opt/ollama/bin/ollama` are normal Homebrew paths to the same installed binary,
- seeing both in `ps` output does not mean Ollama was installed in the wrong place,
- the real problem is duplicate process ownership, not the path spelling.

## Environment

The canonical project environment is:

- `.venv/`

Install Python dependencies from `requirements.txt`.

Local test runner:

- `python -m pytest`

MLX runtime note:

- `MLX_PYTHON` controls the shared MLX LM/VLM/Whisper interpreter path.
- `MLX_AUDIO_PYTHON` can point to a separate interpreter for `mlx_audio.server`.
- If unset, Ollmo also checks `/opt/mlx-audio/venv/bin/python` automatically for TTS.
- Use a separate TTS venv when `mlx-audio` pins package versions that conflict with the LM/VLM stack in `/opt/mlx/venv`.
- Whisper discovery safely preflights its Numba dependency without importing MLX or initializing Metal. An incompatible NumPy/Numba pair is reported as degraded and is not startable.
- Run `python scripts/ollmoctl.py doctor runtime --json` to distinguish a package/dependency incompatibility from a stale degraded instance that only needs a bounded restart.

## Model Capability Routing

`/api/start_model` is capability-aware and validates startup requests before launching a model instance.

Supported capabilities:
- `chat`: text/chat models (default fallback).
- `vision_analysis`: OCR/vision analysis models.
- `image_generation`: image-generation models (for example Flux family).
- `speech_to_text`: speech transcription models (for example Whisper on MLX).
- `text_to_speech`: speech synthesis models (for example Qwen3-TTS on MLX Audio).

Capability-specific runtime behavior:
- `chat` (MLX LM): starts `mlx_lm.server`; registry metadata now records `backend_package=mlx_lm`, `backend_contract=mlx_lm.server`, and the LM-specific runtime knobs surfaced by Ollmo (`prefill_step_size`, `prompt_cache_size`, `prompt_cache_bytes`).
- `image_generation` (Ollama): starts a dedicated Ollama instance without chat preload.
- `speech_to_text` (MLX): starts the local `mlx_whisper` HTTP shim (`backend_package=mlx_whisper_shim`), not native `mlx-audio`; the shim exposes `/healthz`, `/v1/audio/transcriptions`, and `/api/transcribe`.
- `text_to_speech` (MLX): starts `mlx_audio.server` and serves OpenAI-style speech synthesis through `/v1/audio/speech`; Ollmo records that the upstream package is broader than the current instance contract and also covers STT/STS upstream.
- `vision_analysis` (MLX VLM): starts `mlx_vlm.server` and uses its OpenAI-style multimodal routes; Ollmo records the richer upstream contract including `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/health`, and `/unload`, plus the one-model-at-a-time / lazy-load semantics. This is also the generic path for MLX multimodal snapshots whose metadata resolves to `image-to-text` or `image-text-to-text`.

Unsupported backend/capability combinations return `400` with a clear message instead of generic `500`.

Available-model discovery is intentionally broader than startup eligibility:

- `/api/available_models` and `ollmoctl models list` can show cached source entries that are present locally but not yet startable through the current backend contract.
- Those entries carry `discovery_state`, `runnable`, `runnable_checks`, and `disabled_reason` so callers can distinguish "seen in cache/catalog" from "can be launched now."
- Interactive startup surfaces should only offer runnable entries as start choices.

## Compatibility Capability-Aware Inference (`/api/infer`)

Keep `/api/infer` only for lower-level debugging, narrow compatibility cases, or older direct clients that have not moved yet. It is now a compatibility wrapper around the same shared infer execution used by `/api/responses`. New integrations and first-party UI flows should prefer `/api/responses`.

- `chat`:
  - JSON prompt only or multipart with text/image/text-file attachments.
- `vision_analysis`:
  - multipart image upload, or PDF upload (text-layer extraction + scan fallback via rendered pages).
- `image_generation`:
  - prompt required; optional image attachment for image-to-image flows.
- `speech_to_text`:
  - multipart audio upload required.
- `text_to_speech`:
  - text prompt required; backend persists generated audio to `artifacts/audio/` and returns the saved path.
  - Session Controls for TTS now support speaker/style tuning via `voice`, `instruct`, `speed`, and `pitch`.

Current direct UI/internal behavior:
- Direct backend chat route `/api/chat` still exists only as a compatibility wrapper for specialized or older callers.
- First-party UI execution flows, voice input, `ollmoctl send`, and the Responses workbench use `/api/responses` as the canonical execution surface.
- `/api/responses` now rejects plain chat text that claims image/audio generation for a non-chat routed turn when no real image/audio artifact was produced.
- Arena mode stays chat-only.
- Voice input button (UI):
  - Requires one running `speech_to_text` instance (for example Whisper).
  - Records microphone audio, auto-stops on silence (with a max-duration safety cap), transcribes via `/api/responses`, and inserts text into the prompt box.
  - If transcript ends with `send` (or `senden`), UI treats it as a voice command and triggers `Run` automatically.
  - Works as prompt input for chat, vision, and image-generation tabs.
- Speech output handling:
  - UI renders an inline audio player plus `Open Folder` and `Download Audio` actions for `text_to_speech` responses.
  - Backend persists synthesized audio under `artifacts/audio/`.
  - In a canonical multi-phase `text_to_speech -> speech_to_text` response, artifact existence is not enough to prove spoken meaning. The TTS fill records the exact final backend prompt and SHA-256 as `tts_semantic_source`; its directly dependent STT fill records `tts_stt_semantic_evidence` against only that producer source. Missing, drifted, ambiguous, or mismatching evidence blocks the branch with `DEPENDENCY_CHAIN_REPAIR_REQUIRED` / `repair_dependency_chain`, even when a WAV and transcript physically exist.
  - For counted audio variants, explicitly labelled contiguous variants are the candidate authority. Runtime selects one speakable field per index and excludes sibling transcript claims, analysis, code, and JSON; duplicate, missing, or ambiguous slots fail closed instead of being spoken.
  - For Qwen3-TTS model families:
    - `CustomVoice` variants are the right fit when you want one stable named speaker plus optional emotion/style instructions.
    - `VoiceDesign` variants are the right fit when you want the voice itself described in natural language.
- Image output handling:
  - UI renders `Open Folder` (local file manager) and `Download` buttons below generated images.
  - Backend persists generated images to `artifacts/images/` when needed and returns the saved path in API payload for local open actions.
- PDF document handling:
  - PDF attachments are accepted in `/api/responses` and still supported by `/api/infer` compatibility calls.
  - For `vision_analysis` models (for example DeepSeek-OCR), PDFs are treated image-first: pages are rendered and OCR runs page-by-page.
  - DeepSeek-OCR page extraction follows the official prompt pipeline from the upstream docs:
    - primary: `<image>` + `<|grounding|>Convert the document to markdown.`
    - fallback: `<image>` + `Free OCR.`
  - For searchable/text PDFs, backend extracts text and sends it as document context.
  - For scanned PDFs, backend renders pages as images and runs page-wise OCR/context analysis.
  - OCR page execution uses `/api/generate` first and automatically retries with an emergency OCR prompt when Ollama returns empty content or upstream generate errors.
  - DeepSeek-OCR requests are sent with conservative runner options (`num_ctx=4096`, `num_keep=0`) to avoid sequence errors on some Ollama builds.
  - Prompt-echo responses (where the model repeats user instructions instead of OCR text) are treated as invalid page OCR output and trigger fallback automatically.
  - Optional request params:
    - `pdf_max_pages` (blank by default; only set when you want to cap how many pages Ollmo renders and OCRs for this request, max `500`)
    - `pdf_dpi` (default `260` for `vision_analysis`, max `600`)
    - `pdf_page_timeout_sec` (default `240`; timeout per OCR page call)
    - `pdf_page_retry_dpi` (default auto; retries failed pages with lower DPI)
    - `pdf_max_image_side` (default `2400`; constrains rendered page size for stability)
    - `pdf_prefer_text` (default `false`; for vision OCR path we render PDF pages first)
    - `infer_timeout_sec` (default `1200`, max `7200`)
    - `pdf_synthesize` (default `false`; set `true` for one-pass global merge)
  - If individual pages fail, the backend returns partial page results plus warnings instead of dropping the full document response.
  - Inline PDF output is a convenience view and may be shortened for UI stability; the saved markdown artifact is the completeness source of truth.
  - Text-layer PDF runs now persist both the extracted source text artifact and the model-produced result artifact when available.
  - PDF inference history is persisted to `state/infer_history.jsonl` for later reuse/retrieval.
  - Cache reuse is enabled by default for PDFs (`reuse_cached=true`) based on file hash + model + capability + prompt.
  - Query stored insights via `GET /api/infer_history?file_kind=pdf&limit=50`.
  - `pypdf` is installed by default for text-layer extraction.
  - Multi-page scanned-PDF rendering can optionally use `PyMuPDF`. It is not a
    default dependency because its upstream license is AGPL-3.0 or commercial.
    Review the applicable upstream terms before installing it separately with
    `python -m pip install PyMuPDF`. Without it, Ollmo retains a limited macOS
    first-page rendering fallback when available.

## Registry Schema

Running model records in `model_ports.json` now include:
- `modelName`: canonical model identifier (same semantic value as `model`).
- `backend`: normalized backend (`ollama` or `mlx`).
- `capability`: normalized capability used by routing.
- `features`: canonical feature flags such as `vision_input`, `audio_input`, `image_output`, `audio_output`, `tool_calling`, `function_calling`, `computer_use`, and `structured_outputs`.
- `feature_sources`: per-feature provenance labels such as `explicit_metadata`, `local_template`, `curated_override`, `capability_contract`, or `conservative_default`.
- `inputs`: normalized input modality list.
- `outputs`: normalized output modality list.
- `backend_metadata` for Ollama-backed entries: a doc-aligned `/api/show` summary carrying fields such as `capabilities`, `details`, `parameters`, and derived `context_length`.
- `backend_metadata` for MLX-backed entries: package-contract metadata describing which surface is in play (`mlx_lm`, `mlx_vlm`, `mlx_audio`, or `mlx_whisper_shim`), the upstream/native endpoint set, package capabilities, and key constraints such as lazy loading or one-model-at-a-time model binding.
- `backend_package` / `backend_contract`: additive top-level provenance fields so callers can distinguish generic `backend = mlx` from the actual MLX package contract.
- `discovery_state`, `runnable`, `runnable_checks`, `disabled_reason`: available-model fields that now separate cache/catalog discovery from locally runnable backend truth.

The legacy `model` field remains for backward compatibility.

Dynamic runtime state now lives beside the stable registry in `state/runtime_status.json`.
This status registry tracks live metadata such as readiness, activity, port/process health, and recent success/error timing without overloading `model_ports.json`.
For Ollama-backed entries it also carries `backend_runtime`, a doc-aligned `/api/ps` summary with live loaded-model facts such as `expires_at`, `size_vram`, `context_length`, `digest`, and backend `details`.
For MLX-backed entries the same nested field now carries package-aware live runtime facts such as the native base URL, package-specific health or infer routes, unload/model-list URLs where applicable, and the current model-binding strategy (`model_bound_at_launch` vs `model_selected_per_request`).
It is now fed by canonical start/stop plus successful or failed work through chat and `/api/responses` flows, with `/api/infer` and `/api/chat` retained only as compatibility plumbing where still needed.
That split is intentional:
- `model_ports.json` is the stable registry plus doc-backed `/api/show` facts.
- `state/runtime_status.json` is the live status layer plus backend-specific runtime facts (`/api/ps` for Ollama, native-route/runtime binding facts for MLX).
- `ollmoctl instances list --json` and `GET /api/running_instances` merge both views so callers do not have to guess from model names alone.

For external wrappers, Ollmo now exposes a discovery manifest:
- `GET /api/runtime_manifest`
- alias: `GET /api/routing_table`

The manifest advertises:
- canonical responses endpoints (`/api/responses`, `/v1/responses`)
- direct per-instance responses URLs (`/api/local_provider/<instance_id>/v1/responses`)
- running instance metadata (`instance_id`, `model`, `backend`, `capability`, `features`, `feature_sources`, `inputs`, `outputs`, `backend_metadata`, `backend_runtime`, readiness/activity)
- capability aliases such as `chat`, `image`, `ocr`, `stt`, `tts`

This is intended for external wrappers that need to self-dock against Ollmo
without hard-coding ports or internal route knowledge.

Canonical `/api/responses` can now be targeted in three ways:
- explicit `instance_id`
- wrapper alias / profile such as `chat`, `image`, `ocr`, `stt`, `tts`
- canonical `capability` when only one default running instance for that capability should be chosen

That means wrapper code can first read the manifest and then either:
- call `/api/local_provider/<instance_id>/v1/responses` directly for a fixed model, or
- call `/api/responses` with `alias`, `profile`, or `capability` instead of hard-coding `instance_id`

Wrapper code should not need `/api/infer` or `/api/chat` unless it is intentionally targeting the lower-level compatibility surface.

Feature-contract note:
- `capability` is the coarse top-level mode.
- `features` / `inputs` / `outputs` are the finer-grained routing contract for downstream clients.
- Tool/function/computer-use flags stay conservative unless Ollmo has explicit evidence, while modality flags and known local template evidence are projected directly when available.

Active UI chat history persists under `state/chat_history/`, including the synthetic Responses workbench conversation. This is the canonical backend chat-history store for the active Ollmo chat UI.

Frontend/history shape note:

- the page shell owns root state/startup plus timeline glue, while active frontend behavior lives in `static/ui/messages.js`, `static/ui/conversations.js`, `static/ui/settings-history.js`, `static/ui/models.js`, `static/ui/message-state.js`, and `static/ui/request-lifecycle.js`
- durable chat history persists both request metadata and reusable artifacts
- exact canonical Responses truth includes `working_frame` for the final orchestration state plus `response_frame`, a frozen local snapshot of request, target, route, runtime, non-default control snapshots, artifacts, error, output metadata, and the frozen working frame; bounded normal wire views expose refs and handles instead of hydrating these bodies
- reusable settings artifacts can be promoted explicitly from response-frame controls through `/api/settings_artifacts`; they store replay `request_overrides` under `artifacts/settings/`
- `message.artifacts[]` is the canonical reusable-file ledger for both user inputs and assistant outputs
- `request_snapshot` remains the durable request-context record, with `request_snapshot.input_artifacts[]` kept only for explicit user-side inputs instead of silently mirroring assistant outputs back into user turns
- one-shot selected reference artifacts/messages are conversation-scoped UI state for the next turn, not a global workbench carry-over
- lineage/rotation state for chat tabs is tracked through conversation metadata plus `slot_history_ids`
- active workspaces center on the current or selected draft/conversation
- saved chats live in the durable backend-driven `History` archive surface

The model maintenance panel is backend-aware:
- `Ollama` pulls/removes through the existing Ollama CLI path.
- `HF / MLX` pulls/removes through the Hugging Face cache using the MLX venv `hf` CLI.

## Add New Models Safely

1. Pull/install the model locally first.
2. Confirm the exact model ID:
   - Ollama: `ollama list`
   - MLX / llama.cpp catalog-backed entries: check `./ollmo ctl models list --json` or `/api/available_models`
3. Pick the correct capability:
   - chat -> `chat`
   - OCR/vision -> `vision_analysis`
   - Flux/image -> `image_generation`
   - Whisper/STT -> `speech_to_text`
   - Qwen3-TTS / speech synthesis -> `text_to_speech`
4. Start via API/UI using matching backend + capability.
5. If startup fails:
   - Check the API error text for model-name suggestions or unsupported capability/backend guidance.
   - Re-run `./ollmo ctl models list --json` and inspect `runnable` plus `disabled_reason` for the model before retrying.
   - Verify model naming (for example `x/flux2-klein` vs `flux2-klein:latest`).

## API Start Example

```json
POST /api/start_model
{
  "model": "x/flux2-klein",
  "backend": "ollama",
  "capability": "image_generation"
}
```

## Whisper Transcription Example

```json
POST /v1/audio/transcriptions
{
  "audio_path": "/absolute/path/to/audio.wav",
  "task": "transcribe",
  "language": "de"
}
```

## Canonical TTS Speech Example

```json
POST /api/responses
{
  "instance_id": "mlx-community__Qwen3-TTS-12Hz-0.6B-Base-bf16-mlx-11504",
  "input": "Guten Tag aus Ollmo.",
  "voice": "Chelsie",
  "instruct": "Warm, calm, elegant German narration with a slight smile.",
  "speed": 0.95,
  "pitch": 1.1,
  "lang_code": "de",
  "response_format": "wav"
}
```

This single-phase example materializes one audio artifact. When a request also asks Ollmo to confirm the exact spoken text, use the canonical TTS-to-STT dependency chain and read its Closure plus `tts_semantic_source` / `tts_stt_semantic_evidence`; a non-empty WAV or transcript alone is not semantic fulfillment. The expected source text is verification truth and is never inserted into the STT request.
