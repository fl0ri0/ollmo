# Ollmo request and result recipes

Read only the section needed for the current operation. The skill entrypoint
owns checkout resolution, scope and essential safeguards. All `<ollmo>/` paths
here use that resolved checkout, even when Codex starts elsewhere. Endpoint paths
use its supported control-plane base (default `http://127.0.0.1:5001`, or the
applicable `OLLMO_WEB_BASE` client override).

## Observation and route preview

For a status/capability question, inspect permitted existing runtime records or
the passive manifest/instances/fabric/Ghost/status/preferences surfaces named in
the entrypoint. Read `<ollmo>/docs/TRUTH_SOURCES.md` when sources disagree.
The registry preserves identity; cached status does not prove current liveness.
Do not prune, synchronize, recover or repair while observing.

For a route question, `POST /api/ghost_route_preview` accepts:

    {
      "prompt": "<current request>",
      "compute_semantics": false
    }

Preview is distinct from response execution. It must not start/stop/unload models,
materialize branches or artifacts, freeze frames, or call `/api/responses`.
However, semantic preview can use helper backends: non-UI
`OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS=off|on|auto` defaults to off when absent
or invalid; auto currently behaves like on. Explicit true opts into compute.
Explicit false normally keeps preview passive, but
`OLLMO_GHOST_PREVIEW_COMPUTE_SEMANTICS_FALSE_OVERRIDE=deny` can preserve an
active on/auto policy. For a strict no-model/no-write task, confirm a permitted
passive path or use existing files rather than assuming false overrides policy.
Computed preview is not fresh execution evidence or automatic learning input.

If preview is unavailable, report the best supported capability answer from
existing evidence. Do not execute a response to discover its route or recover the
control plane as a fallback. Once a clear sandbox denial is established, follow
the entrypoint's failure path instead of trying raw ports.

## Execution request selection

Use `POST /api/responses` for an authorized generation/run. Select one path:

**Ghost-first** when Ollmo should form the route, phases or dependencies:

    {
      "ghost_route": true,
      "prompt": "<current request>",
      "conversation_id": "<stable conversation identity>"
    }

Read `GET /api/ghost_preferences` when available. Pass its non-empty
`preferences` object as `ghost_preferences`; omit it if empty. Include
`ghost_messages`, `ghost_preview` or `request_meta` only when preserving the
actual active conversation/UI contract. Do not add old context merely because it
exists. Ghost planning can return pending downstream branches after initial text.

**Direct instance** when the requested/resolved running target matters:

    {
      "instance_id": "<exact identity from runtime truth>",
      "prompt": "<bounded request>"
    }

Do not guess the identity from a display name or repeat Ghost routing for a
resolved single-capability task. Resolve dynamic identities from runtime truth;
preserve the user's selected provider and local-only constraints.

**Capability** when any compatible running target is acceptable:

    {
      "capability": "chat",
      "prompt": "<bounded request>"
    }

Common capabilities include chat, image_generation, vision_analysis,
speech_to_text and text_to_speech; use the manifest for current compatibility.
A runnable-but-stopped catalog entry is not a running target. Model start remains
a separate lifecycle action.

Shape the request around the exact outcome, inputs/artifact refs, output format,
count and material constraints (language, dimensions, voice, preservation or
no-inference requirements), plus what counts as incomplete. For linked files,
require final relative links to actual saved dependencies. For prepare-then-speak,
the TTS branch consumes the accepted answer payload, not a restatement of the
root request. Selected/reference artifacts do not become fresh generation intent.
Do not loosen named budgets or retry gates to make a request fit.

The `/v1/responses` alias exposes Ollmo's limited OpenAI-shaped compatibility
surface. Read `<ollmo>/docs/RESPONSES_CONTRACT.md` under “OpenAI-shaped
compatibility boundary” before using standard message/tool/conversation fields.
The current role-normalization, tool-item and usage limitations are documented
there; do not invent full support or silently change integration behavior.

## Follow an existing response

Retain the returned response id. While Late Fill is pending/running, inspect:

    GET /api/responses/<id>?view=status

Default GET returns a bounded UI projection. Status/compact/observer and debug
views are bounded; they are not copyable artifact content. When exact response
details are required, use `?view=truth`, `?view=full` or `?view=raw` without
`compact=true`. Fetch only when needed, rather than full-payload polling.
Do not resubmit the original prompt to ask whether the existing work is complete.

Judge completion from canonical outputs, artifacts, frames, lifecycle, closure,
surface state and Late Fill. Compatibility `status=completed` may describe an
initial message while work remains. Active Late Fill outranks stale completed/
repair projections; durable hard-terminal truth outranks legacy active residue.
Use `status_semantics`/`status_lookup` when present and preserve failure detail.

For missing or conflicting public output, inspect exact `late_fill.fill_results`
branch/path evidence and the latest frame. Do not substitute generic same-type
artifacts for bound refs. A read-only client reports disagreement; projection
repair or registry writes require applicable engineering/operator scope.

## Artifact and multimodal results

Use final saved files for bytes, frame/closure truth for obligations and the
artifact registry/dossiers for identities and provenance. Registry merge/rewrite
semantics are not an append-only event log or permission to mutate artifact bytes.
Prefer canonical public `outputs` over raw dossier harvesting and provisional
planner text. Explicit non-artifact chat stays public only when represented as
its own canonical output. Exact counts remain owed absent waiver/supersession.

Surface confirmed paths and artifact refs. In clients that support local previews,
use absolute filesystem paths for media/file links. Code, a prompt or a claimed
path alone does not establish a materialized file.

Read the “Artifact fulfillment”, terminal-projection, multimodal and TTS sections
of `<ollmo>/docs/RESPONSES_CONTRACT.md` for the affected task. Preserve these
decision-point checks:

- Linked HTML/CSS/JS/media outputs need actual saved dependency closure; placeholders
  and guessed paths do not close the request. A target-file repair cannot be
  fulfilled by writing a different sibling file.
- Explicit preserve/no-regenerate/no-reanalyze intent carries exact predecessor
  evidence and fails closed if that evidence is missing or conflicting.
- Evidence consumers bind to their own producer. A vision branch cannot use a
  sibling's image; invalid structured joins remain inspectable with repair-needed
  closure rather than fabricated count/label/ref agreement.
- Counted TTS uses explicitly labelled contiguous variants with one speakable
  field per index. Physical WAV integrity is separate from semantic fidelity.
  The actual TTS source and digest bind dependent STT evidence; never supply the
  expected text as STT input or add STT when no such work is owed.
- Required text artifacts may reuse validated canonical saved evidence; a failed
  validation follows existing same-branch retry/dependency gates, never root-prompt
  replay. Detailed deterministic repair rules remain in the contract and PATTERNS.

## Bundles and cleanup

For an authorized bundle/export, read the bundle sections of
`<ollmo>/docs/RESPONSES_CONTRACT.md`. Bundles are derived export surfaces under
artifacts/bundles/, not original model outputs or closure authority. They must not
run models, change routing/Late Fill, mutate original files or rewrite response
closure. Resolve full canonical truth; a truncated wire projection is insufficient.
Bundle manifests own copied files, rewrites, checksums and link status. Public
roots include the saved dependency closure of the final entrypoint, not every
diagnostic repair artifact. Historical bundle records are current only when they
identify the latest existing/openable directory; stale records are not a valid
bundle or a reason to claim one exists.

Before authorized clean/archive/forget work, read “First Commands” (clean/archive)
and “Context And Learning Contract” (retention) in `<ollmo>/OLLMO_FOR_AGENTS.md`, plus
the cleanup contracts in
`<ollmo>/docs/TRUTH_SOURCES.md`. Clean is a development reset, not a harmless
diagnostic. Archive is not Ghost-forget. Preserve protected preferences, compiled
memory, active learning and referenced sidecars; missing retained evidence is a
diagnostic, never permission to silently replace it with empty content.
`clean --forget-ghost` and `ghost --reset-learning-state` are distinct explicit
operations. None is implied by start, stop, route selection or a failed request.

## Repair, development and integration depth

For repair/rebase, use `<ollmo>/docs/CONTROL_KNOBS.md` and
`<ollmo>/docs/PATTERNS.md`: proposals remain advisory until runtime validation;
shadow/stage records are non-executable; exact branch/target/parent and operator
identities remain binding. A model statement, learning hint, monitor summary or
advisory degraded state cannot satisfy those gates.

For engineering and test selection, use `<ollmo>/AGENTS.md` and
`<ollmo>/docs/TESTING_PROTOCOL.md`. There is one affected-test matrix, not a
second broader skill mandate. Isolate tests from production ledgers/artifacts;
do not invoke live workloads merely to validate instructions. GHOST.md is runtime
prompt input and is read/edited only for relevant authorized policy work.

For downstream Codex and opt-in provider projections, use “External Integrations”
in `<ollmo>/OLLMO_FOR_AGENTS.md`. Preserve isolated user-config/rules handling,
bounded tasks and the response-owned child lifecycle. Do not infer a downstream
model from the supervising Codex session or turn old provider projections into
runtime truth.
