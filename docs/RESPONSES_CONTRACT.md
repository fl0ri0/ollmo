# Responses Contract

This document names the response-state fields that are part of Ollmo's runtime truth contract. It is intentionally narrower than general product docs: it describes what callers and UI code may rely on.

## Response Frame

`response_frame` is the frozen response snapshot. It is immutable once written. Late fill and recovery do not edit an old frame; they produce successor lookup payloads and, outside tests, successor frame ledger entries.

Important surfaces:

- `response_frame.request`: the request snapshot used for replay/audit.
- `response_frame.planning.artifact_flow.work_tree`: internal work truth.
- `response_frame.planning.artifact_flow.output_slots`: slot projection from the work tree.
- `response_frame.output.outputs`: canonical public output projection.

Frame ledger rules:

- The first persisted frame for a response id is `frame_relation.kind = initial`.
- A later persisted state for the same response id is a typed successor frame with `parent_response_id`, `parent_frame_id`, and `parent_frame_sequence`. Ordinary continuation normally uses `frame_relation.kind = late_fill_successor`. Bounded graph relations are not interchangeable: `graph_patch_reopen_successor` is executable only after exact terminal-owner revalidation; `graph_patch_terminal_review` is audit-only; `graph_rebase_stage_successor` is durable audit-only; and `graph_rebase_partial_successor` is executable only for the exact registry-trusted, gate-approved branch-local continuation. Full, staged-only, stale, widened, untrusted, or gate-blocked rebase records remain non-executable.
- Every appended frame for the same response id must have a unique monotonic `frame_sequence` and matching `frame_id` such as `resp_x:frame-1`, `resp_x:frame-2`, `resp_x:frame-3`. If a live payload still carries stale successor metadata, persistence rewrites the append metadata so the new ledger fact links to the latest prior frame instead of reusing an old frame id.
- Successor frames are new audit facts. They do not mutate the original frame and must be distinguishable during replay.
- Late Fill producer/consumer result truth is also typed. A TTS producer records the exact final backend prompt and SHA-256 as `tts_semantic_source`, and every produced PCM WAV receives deterministic `tts_audio_integrity_evidence` bound to that source and file. HTTP 200, a readable WAV header, or non-zero file size is transport/artifact-existence truth only. Audio fulfillment additionally requires sufficient effective active-signal duration for the source and acceptable silence/trailing-padding ratios. Failed or unavailable integrity keeps the physical file and registry record as diagnostic evidence, marks it non-materializable, blocks the audio obligation, and requests branch repair. Its directly dependent STT consumer records `tts_stt_semantic_evidence` against that one bound source. Missing, drifted, ambiguous, or mismatching evidence blocks fulfillment with dependency-chain repair even if an audio artifact and transcript exist; expected source text is never supplied to STT.
- `response_frame.current_state` is the compact current lookup snapshot captured inside that frame. It is recovery data for `/api/responses/<id>`, not permission to rewrite the frozen frame.
- Persisted ledger rows are compact audit facts. Large internal snapshots such as full `runtime`, `working_frame`, request-phase graphs, context candidates, planner diagnostics, bulky semantic-review state, work trees, large request inputs, and oversized planning contracts may be moved into sidecar JSON files under `state/response_frames/snapshots/`. Sidecars are content-addressed by SHA-256, so separate semantic refs such as `runtime` and `current_state.runtime`, `planning.request_phase_graph` and `planning.artifact_flow.request_phase_graph`, or repeated work-tree projections may point at the same physical file when their payloads are byte-identical. Large nested runtime/working-frame subtrees are split into their own `*_snapshot_ref` entries rather than summarized; the parent snapshot keeps the ref structure, and the child sidecar keeps the full raw subtree. The ledger keeps machine-readable `*_snapshot_ref` / `external_snapshots` entries with path, SHA-256 digest, byte size, and JSON path. This preserves truth without duplicating multi-megabyte internal state in every successor row.
- Successor ledger rows delta-log external snapshots against their parent frame. `external_snapshots.items` on a persisted successor contains only new or changed refs for that row; unchanged parent refs are listed under `external_snapshots.inheritance` and omitted from the row-local `items`. Recovery/replay merges the parent manifest before returning the current response view. When a recovered frame exposes both effective and row-local truth, `external_snapshots.items` is the effective merged manifest and `external_snapshots.delta_items` is the successor row's own diff.
- Sidecar snapshots may themselves be compact CAS manifests. Large worthwhile child fields such as request-phase graph nodes/edges, context candidates, decision/semantic contracts, graph-closure reviews, dependency evidence, branch/result collections, and work-tree structures can be replaced inside the sidecar by child `*_snapshot_ref` entries when they exceed the split threshold. This is ref-splitting, not summarization: child sidecars keep the full content, and snapshot readers expand those refs for replay/recovery.
- `working_frame` ledger sidecars are logic-only vessels. They keep small orchestration state such as `status`, pending ids, closure/loop state, and compact route/request summaries inline, while graph, contract, prompt, input, context-candidate, and artifact-flow bodies are represented by child `*_snapshot_ref` entries. The child sidecars remain full diagnostic truth.
- `outputs[]` and `output_slots[]` are handle projections. They keep `artifact_ref`, slot/status/type/order, and compact recovery fields, but must not inline artifact dossiers, prompts, provenance, metadata, `image_state`, paths, or nested `artifacts[]` once an artifact ref exists. Full artifact identity, provenance, metadata, and enrichments live in the artifact dossier snapshot and registry.
- Diagnostic snapshot hashes exclude volatile timestamp keys such as `created_at`, `updated_at`, `started_at`, and `completed_at` for graph/contract/context/work-tree sidecars. Those timestamps are bookkeeping metadata, not diagnostic content identity; frame/index metadata remains the operational time source, while the snapshot hash represents stable diagnostic content.
- Persisted frames are indexed by `state/response_frames/current_index.json` for current-state recovery. The index is an optimization only; the compact JSONL ledger remains durable recovery truth. A current verified index binds the complete response map to physical ledger EOF with its byte size, entry count, and stable map digest. Entry-local ledger sizes are historical append-time facts: an older byte offset remains usable after unrelated append-only rows are added, provided global freshness holds and the decoded row still validates the requested response and frame identity. A verified complete map can also prove an absent response id without scanning the ledger. Legacy, incomplete, stale, malformed, or corrupt indexes cannot prove absence and fall back to the ledger; a failed direct read does the same. Recovery then returns the latest valid frame for the requested response id, or truthful not-found/corrupt-ledger state rather than unrelated or older state.
- A complete legacy v1 map can cross that boundary only through the explicit operator command `.venv/bin/python scripts/attest_response_frame_index.py`. Use `--check-only` first when inspecting an existing ledger. Attestation streams the physical ledger one line at a time, requires the exact response-id set and exact latest frame id/sequence/byte offset/line length for every entry, preserves all response entries and effective snapshot manifests, and atomically adds only the v2 coverage fields. Malformed rows, mismatches, or moving ledger/index evidence reject without writing. This is an index attestation, not a ledger rewrite or read-path migration.

Wire projection and canonical truth are deliberately separate:

- Successful non-streaming `POST` responses and `GET` views `default`, `ui`, `status`, and `debug` are byte-budgeted public projections. The final serialized outer envelope is at most 8 MiB, including retry/control/status wrappers; each serialized Responses-style SSE event is held to the same ceiling. These paths never hydrate sidecars.
- Public compaction is byte-based, not a fixed character or item-count cut. Ordinary 5,000-character text and collections of 65 tiny records remain intact when they fit the budget. Bulky strings at or above 256 KiB and collections at or above 1 MiB may instead expose a bounded preview plus exact length/count, byte size, SHA-256, and an adjacent content-addressed `*_snapshot_ref` for the complete value.
- `view=full`, `view=raw`, and `view=truth` are exact canonical reads. They may exceed the public wire budget because they recursively restore the complete content-addressed sidecar graph. Every authoritative ref is recursively validated for its declared CAS identity and content before it is exposed; a missing, malformed, or corrupt ref fails closed with HTTP 409 instead of returning partial canonical truth.

This is truth by reference, not semantic summarization. The public wire may carry handles, previews, counts, and digests, while exact bytes remain recoverable from the referenced CAS graph without semantic loss.

`GET /api/responses/<id>` returns the current UI response view by default. It may come from live lookup state or, after lookup TTL expiry/restart, from the latest valid persisted frame for that response id, but the default projection must stay frontend-safe: lifecycle/status truth, output handles, artifact cards, display text, and compact late fill state are included; raw `response_frame`, `runtime`, `working_frame`, work-tree, graph diagnostics, and large repair/debug payloads are not. Recovered default views still carry compact recovery provenance through status fields such as `status_lookup`, `frame_id`, `frame_sequence`, and `state_version`. If the ledger is missing or corrupt, the API must return a truthful not-found or corrupt-ledger error; it must not synthesize a successful response.

Non-streaming `POST /api/responses` and `POST /v1/responses` persist canonical frame truth before serializing the successful response. Their normal wire result is then projected from the current indexed compact ledger row: public lifecycle/status, output handles, output slots/branches, artifact handles, display text, frame identity, and the effective content-addressed snapshot-ref manifest remain available, while hydrated `runtime`, `working_frame`, and repeated full-frame bodies are not copied onto the wire. This projection is read-only and performs no sidecar hydration. If the durable row is not yet available, a bounded in-memory fallback keeps small compatibility payloads inline and replaces large internal bodies with clearly audit-only digest identities; those digest-only identities do not claim replay authority. Explicit canonical reads remain available through the truth views below.

`GET /api/responses/<id>?view=status` returns the compact observer view for an existing response. It is for polling and UI state updates, not artifact copying. It exposes canonical lifecycle truth, open late fill branch state, compact surface/recovery state, and a `state_version` that changes when the observable response state changes. Clients should poll this compact view while work is open, fetch the default UI view only when `state_version` changes or artifact/detail handles are needed, and never start a duplicate `/api/responses` request just to check whether the original work finished. File contents for copy/open/show-all actions should be resolved from artifact entries and targeted artifact endpoints, not by pulling raw response-frame debug truth.

`GET /api/responses/<id>?view=debug` returns a bounded developer observation. It combines the indexed compact frame and all effective snapshot refs with only the selected graph-rebase, Closure, scope-ladder, and candidate-review evidence needed by rollout diagnostics. It does not hydrate unrelated work trees, artifact bodies, graph nodes, or recursive diagnostic children. `?view=full`, `?view=raw`, and `?view=truth` remain the explicit canonical diagnostic lookup and recursively hydrate expanded response frames, runtime graphs, working frames, closure reviews, repair feedback, and other large truth payloads. UI code must not use any of these developer/operator views for normal rendering or reconciliation; exact CAS/replay/operator clients must use `truth` or `raw`, not bounded `debug`.

When present, `status_lookup` is a compact status companion inside the response lookup payload. It should carry the same lifecycle/status semantics used by the compact observer so clients can render state without inferring from assistant prose, stale slots, or compatibility `status`.

`ollmoctl` makes the wire/truth split explicit. `send ... --json` performs one bounded `POST /api/responses` and prints that public wire result. `send ... --truth-json` performs the bounded POST, then reads the same response id through exact `view=truth` and emits a normalized canonical summary. `responses get <response_id> --json` returns the raw canonical truth-view payload, while `--truth-json` emits its normalized summary. `--json` and `--truth-json` are mutually exclusive for each command.

Artifact registry split:

- Response frames carry output refs and compact artifact dossiers for replay/recovery.
- `state/artifact_registry.jsonl` is the durable lookup surface for concrete artifacts across modalities. New output artifacts from `artifacts[]`, `saved_text_path`, `saved_text_artifacts`, `saved_audio_path`, and `saved_image_path` are persisted with `roles = ["output"]`, `artifact_ref`, path, type, provenance, metadata, and linked response ids. True external user inputs are persisted with `roles = ["input"]`.
- Reused Ollmo artifacts are references, not new inputs. Route reuse, selected reference artifacts, artifact bindings, and registry-known paths must carry stable refs/bindings and must not be re-materialized as fresh `input_artifacts`.
- Modality-specific provenance wins over generic output provenance. For example, generated-image provenance remains attached to the image artifact and generic output registration may add lookup metadata or linked responses without downgrading that provenance.

Response artifact bundle operations fail closed on incomplete wire projections. A previewed, truncated, or emergency handle set is not enough to choose bundle contents: bundle creation resolves and recursively hydrates the exact CAS-backed response truth first. Missing, malformed, or corrupt refs return HTTP 409 instead of producing a silently incomplete bundle.

Artifact fulfillment rules:

- Model prose, Markdown code blocks, prompt lists, or chat text are evidence, not fulfilled file artifacts. They fulfill a requested file only when runtime materializes the file and records it in output slots, `outputs`, `artifacts`, response-frame truth, or artifact registry truth.
- A prompt for an image is not the image artifact. A script for audio is not the audio artifact. HTML/CSS prose is not a saved page unless materialized as files.
- Linked artifact sets close only when concrete saved files point to their concrete saved dependencies. Placeholder names, guessed paths, stale paths, wrong relative links, or missing CSS/image references are open artifact-binding problems.
- Template-style asset variables such as `IMG_PATH_1`, `IMAGE_URL_2`, or `ASSET_REF_3` are linked-artifact placeholders. When concrete artifacts already exist, terminal rebind should replace those variables with correct relative artifact links before closure instead of regenerating duplicate assets.
- When concrete artifacts exist but are not linked, recovery should prefer deterministic rebind or bounded repair against existing artifacts before declaring failure or creating duplicate page files.
- If the user requested an exact artifact count, a smaller count remains incomplete unless Closure records an explicit waiver or supersession.

Generated-image `image_state_enrichment` is optional artifact evidence, not the image artifact itself and not proof of artifact closure. `status=pending_existing` means another background worker already owns the same image path; `status=skipped` should carry a reason such as `downstream_vision_analysis` or `required_artifact_closure_priority` when enrichment was intentionally suppressed to avoid duplicate or lower-priority analysis.

Status semantics:

- Top-level `status` is compatibility-facing OpenAI-style state. It may remain `completed` once the first answer is returned.
- `lifecycle_state` is canonical for continuation, late fill, and branch truth.
- First-party runtime and UI code must not treat `status = completed` as terminal when `lifecycle_state` indicates active continuation. Active continuation is explicit, not prefix-based: `late_fill_pending` and `late_fill_running` keep polling/execution active.
- Actionable repair/control states such as `blocked`, `repair_needed`, `late_fill_repair_needed`, `repair_branch_contract`, `repair_dependency_chain`, and `rebuild_from_promoted_obligations` are diagnostic/control states. They must not show running/queued spinners unless a separate active execution state is present.
- Terminal or frozen states such as `late_fill_failed`, `late_fill_completed`, `completed`, `failed`, `cancelled`, `waived`, `superseded`, `skipped`, and `frozen` are non-polling. Failed late fill may still expose recovery candidates or failed branch diagnostics, but it is not an open continuation.
- Responses expose `canonical_status_field = "lifecycle_state"` and a machine-readable `status_semantics` object. When compatibility status and canonical lifecycle split, `status_compatibility = true`.

Example compatibility split:

    {
      "status": "completed",
      "lifecycle_state": "late_fill_running",
      "canonical_status_field": "lifecycle_state",
      "status_compatibility": true,
      "status_semantics": {
        "compatibility_status": "completed",
        "canonical_lifecycle_state": "late_fill_running",
        "canonical_status_field": "lifecycle_state",
        "status_compatibility": true,
        "has_open_continuation": true,
        "has_actionable_repair": false,
        "is_terminal": false,
        "terminal": false
      }
    }

Blocked naming note: runtime uses plain `blocked` as the canonical lifecycle value for blocked late fill/current-state lookup. Older docs or compatibility payloads may say `late_fill_blocked`; clients should treat that as a compatibility alias for blocked repair/control truth, not as active late fill execution.

## Request Phase Graph

`runtime.request_phase_graph` is the request obligation graph. Its user intent is anchored by Ghost and the current turn, but its state may be refined before freeze from runtime evidence.

Graph refinement is allowed only when it continues the same request intent. It must not invent a new task because an older turn or generic history mentioned an artifact.

Additive refinement diagnostics:

    {
      "graph_refinements": [
        {
          "source": "assistant_output_claim",
          "capability": "text_to_speech",
          "reason": "assistant_output_explicitly_claimed_pending_materialization"
        }
      ],
      "prompt_intent": {
        "refined_from_output_claim": true
      }
    }

When this happens, synthesized branch records may carry:

    {
      "branch_id": "branch-text_to_speech-1",
      "capability": "text_to_speech",
      "source": "assistant_output_claim_refinement",
      "refinement_source": "assistant_output_claim"
    }

This is a closure-safety mechanism: if model text explicitly says a downstream phase is pending or queued but the original graph was too thin, Ollmo can materialize the missing branch before the response is treated as final.

## Candidate Graph And Promotion Review

`candidate_graph` is the visible possibility layer. It may contain outputs, workload tasks, context/reference/memory candidates, evidence candidates, repair candidates, continuations, learning hints, and rejected proposals.

`promotion_review` is the boundary that turns a candidate into a promoted contract or leaves it reserved, omitted, waived, rejected, or stale. Only promoted contracts are owed work. Reserved or candidate-only items are state, not pending failures.

The canonical lifecycle is:

    possibility -> relevance -> promoted contract -> runtime work -> review -> freeze

When a candidate is promoted into executable work, the runtime should also carry the branch-scale workload task truth where available:

    {
      "task_id": "task-phase-2",
      "branch_id": "branch-image_generation-1",
      "capability": "image_generation",
      "input_refs": [{"kind": "phase_output", "ref": "phase-1"}],
      "output_contract": {
        "output_type": "image",
        "required": true,
        "status": "planned"
      },
      "lifecycle": {
        "policy": "prepare_execute_verify_freeze",
        "recursive": true,
        "scope": "branch_local",
        "cycle": [
          "prepare",
          "gather_evidence",
          "execute",
          "verify",
          "repair_or_freeze"
        ],
        "stages": [
          {"stage": "prepare", "status": "pending"},
          {"stage": "execute", "status": "pending"},
          {"stage": "verify", "status": "pending"},
          {"stage": "freeze", "status": "pending"}
        ]
      }
    }

The `stages` array remains compatibility state. The `cycle` field is the general branch-scale workload contract: every subtask can prepare focused inputs, gather dependency evidence, execute, verify against its output contract and review criteria, then either repair or freeze. `decomposition_level` is descriptive graph depth and is not a fixed small execution cap.

## Late Fill

`late_fill` records continuation of obligations already present in the request phase graph. It must not create a new user intent.

Late fill executes branch-local contracts. The focused prompt or payload for a branch can come from `content_payload`, `artifact_prompt`, `stage_direction`, dependency artifacts, artifact dossiers, prior branch outputs, vision evidence, transcript evidence, or other promoted input refs. It should not fall back to the full original multi-step prompt when a branch-local task is available.

Core fields:

- `status`: `pending`, `running`, `completed`, `partial_failed`, `failed`, or `skipped`.
- `pending_branches[]`: branch obligations that can still be materialized.
- `completed_branches[]`: branch obligations with runtime evidence.
- `failed_branches[]`: branch obligations that failed with structured diagnostic truth.
- `fill_results[]`: successful materialization attempts.
- `error`: display-safe aggregate error text when the overall late fill failed or partially failed.
- `skip_kind`, `skip_reason`, `skip_source`: present when `status == "skipped"` so "no work needed" is not confused with a failed or suppressed continuation.
- `recovery_candidates[]`: branch-local recovery options discovered by failure analysis. Candidates are inert unless explicitly promoted.
- `auto_recovery_enabled`: currently `false`; recovery state is visible, but no hidden repair loop may execute it.
- `repair_action` / `repair_actions`: optional Closure Repair classification for the active gap or pending repair branches.

UI surfaces should treat `pending_branches`, `active_branches`, `completed_branches`, `failed_branches`, and `fill_results` as branch-state truth before older slot projections. Slots are still useful for layout and artifact identity, but a stale pending slot must not keep showing queued when the matching branch is completed or blocked.

Failed branch shape:

    {
      "branch_id": "branch-image_generation-2",
      "phase_id": "phase-3",
      "capability": "image_generation",
      "output_type": "image",
      "status": "failed",
      "error": {
        "code": "INSTANCE_UNAVAILABLE",
        "message": "image backend unavailable",
        "stage": "execute_prepared_branch",
        "retryable": true,
        "exception_type": "RuntimeError"
      },
      "attempt": {
        "stage": "execute_prepared_branch",
        "capability": "image_generation",
        "instance_id": "flux-2",
        "backend": "ollama",
        "model": "x/flux2-klein:latest"
      },
      "recovery_context": {
        "can_retry": true,
        "retry_scope": "same_branch",
        "suggested_action": "retry_excluding_instance",
        "preserve_intent": true,
        "exclude_instance_ids": ["flux-2"]
      },
      "recovery_state": {
        "kind": "ollmo.late_fill_recovery_state",
        "status": "candidate",
        "trigger": "late_fill_failure",
        "branch_id": "branch-image_generation-2",
        "capability": "image_generation",
        "promotion_required": true,
        "auto_execute": false,
        "preserve_intent": true,
        "retry_scope": "same_branch",
        "suggested_action": "retry_excluding_instance",
        "failed_instance_id": "flux-2",
        "exclude_instance_ids": ["flux-2"]
      }
    }

Allowed `recovery_context.suggested_action` values are:

- `manual_review`
- `start_compatible_instance`
- `retry_excluding_instance`
- `retry_same_branch`
- `repair_dependency_chain`
- `rebind_dependency_evidence`
- `repair_branch_contract`
- `rebuild_from_promoted_obligations`
- `semantic_review`

`repair_dependency_chain` means the failed branch is missing an input artifact from an upstream phase, so repeating the same branch would repeat the graph error. The next recovery step must repair or promote the dependency chain before the branch is executable again.

`rebind_dependency_evidence` means the dependency artifact or evidence already exists in runtime truth but is not bound to the blocked branch. The next recovery step should bind the existing evidence to the branch-local contract instead of regenerating or asking for the same input again.

`repair_branch_contract` means the branch is under-specified or failed contract validation before it can safely execute. `rebuild_from_promoted_obligations` means closure review found that the graph planned too little work for the current intent, so the workload/obligation layer needs repair before execution. Repair-created pending branches should preserve `execution_contract`, `workload_task_ref`, `output_obligation_ref`, `input_refs`, `review_criteria`, and `output_contract` where available.

`semantic_review` means the output exists but qualitative review criteria remain unverified by deterministic runtime truth. Today this is advisory review work unless promotion policy explicitly turns it into an executable verifier branch.

Closure repair feedback may include:

    {
      "repair_loop": {
        "kind": "ollmo.repair_loop",
        "status": "candidate",
        "authority": "runtime_review_promoted",
        "auto_execute": false,
        "repair_work_available": true,
        "round": 1,
        "max_rounds_policy": "bounded_runtime_repair_policy",
        "next_actions": ["repair_dependency_chain"],
        "requires_promotion": true
      }
    }

This candidate loop frame is not an autonomous retry. It tells clients and repair policy where the next bounded repair round would start. When Closure promotes a concrete executable repair contract, `repair_loop.auto_execute` may become `true`; that means the promoted repair branch is schedulable through late fill and may remain open until its bounded automatic repair budget is exhausted. The default budget is controlled by non-UI policy `OLLMO_AUTO_EXECUTABLE_REPAIR_MAX_ATTEMPTS`, and branch/repair-contract metadata such as `auto_executable_repair_max_attempts` can provide a more specific budget within the safe cap.

Late fill gates repair branches before backend execution. A `repair_dependency_chain` or `rebind_dependency_evidence` branch without dependency artifacts or prior branch evidence becomes a blocked failed branch with `DEPENDENCY_CHAIN_REPAIR_REQUIRED`. A `repair_branch_contract` branch without a bounded `execution_contract` becomes a blocked failed branch with `BRANCH_CONTRACT_REPAIR_REQUIRED`. These blocks apply to target materialization, not to the whole repair loop. Response frames should surface `materialization_blocked`, `blocked_scope`, `blocked_prerequisite`, `repair_work_available`, `repair_work_policy`, and `needs_external_input` so clients know whether Ollmo can still repair the missing prerequisite locally. When the required dependency evidence or execution contract is present, the corresponding block flag is resolved and the bounded branch can run. These are repair states, not backend retries.

## Output Slots And Outputs

Slots are presentation truth tied to work truth. They should stay compact and reference branch-local diagnostic truth instead of duplicating it.

Canonical output rules:

- Work-tree output nodes are the internal source of output slots when a runtime-owned `planning.artifact_flow.work_tree` is present.
- Slot/work-tree/promoted-obligation outputs are canonical public truth and should carry `source = promoted_output_slot` plus `compatibility_derived = false`.
- Fallback outputs assembled from `output_text`, `content_payload`, or stray artifacts are compatibility projections only. They should carry `source = compatibility_derived` plus `compatibility_derived = true`.
- Runtime consumers must not treat compatibility-derived outputs as proof that a promoted obligation was fulfilled. Use output slots, work-tree nodes, branch state, artifact refs, and closure/recovery state for that decision.

Work-tree ownership rules:

- Runtime-owned work trees are authoritative when present and response frames snapshot them. They are marked with `work_tree_source = "runtime_owned"`, `authoritative = true`, and `compatibility_derived = false`.
- `output_slots` are derived from that work tree; they should not silently replace it with prose, legacy `outputs`, or artifact fallbacks.
- For legacy payloads without a runtime-owned work tree, the frame planner may still build a `derived_planning_snapshot` for compatibility. That fallback is marked `authoritative = false` and `compatibility_derived = true`; it is not deeper truth than an explicit runtime work tree.

Blocked slot shape:

    {
      "slot_id": "output-phase-3",
      "branch_id": "branch-image_generation-2",
      "phase_id": "phase-3",
      "type": "image",
      "status": "blocked",
      "blocked_reason": "image backend unavailable",
      "error_ref": {
        "branch_id": "branch-image_generation-2",
        "code": "INSTANCE_UNAVAILABLE",
        "stage": "execute_prepared_branch"
      },
      "recovery_context": {
        "can_retry": true,
        "retry_scope": "same_branch",
        "suggested_action": "retry_excluding_instance",
        "preserve_intent": true,
        "exclude_instance_ids": ["flux-2"]
      },
      "recovery_state": {
        "kind": "ollmo.late_fill_recovery_state",
        "status": "candidate",
        "trigger": "late_fill_failure",
        "branch_id": "branch-image_generation-2",
        "capability": "image_generation",
        "promotion_required": true,
        "auto_execute": false,
        "preserve_intent": true,
        "retry_scope": "same_branch",
        "suggested_action": "retry_excluding_instance",
        "failed_instance_id": "flux-2",
        "exclude_instance_ids": ["flux-2"]
      }
    }

Top-level `outputs[]`, `output_slots[]`, and `output_branches[]` should preserve `blocked_reason`, `error_ref`, compact `recovery_context`, and compact `recovery_state` so UI and API consumers can render status and offer explicit recovery.

## Text Artifact Payloads

Text/file artifacts are materialized outputs. If a model returns a structured envelope such as:

    {
      "output_obligations": [
        {
          "type": "artifact",
          "name": "README.md",
          "mime_type": "text/markdown",
          "content": "# Local Canvas\n\n..."
        }
      ]
    }

then the persisted artifact payload is `output_obligations[].content`, not the surrounding route/control JSON. A wrapper without payload content is not proof that a file exists.

Ambiguous deictic requests such as "make this HTML" need a selected source, current payload, or explicit reference before persistence.

## Artifact Dossiers And Evidence

Artifact continuity is centered on durable identity, not copied paths. `artifact_dossiers` are keyed by `artifact_ref` and gather identity, provenance, metadata, enrichments, linked response/message ids, and availability.

Late branches should use existing dossier evidence when it satisfies the promoted branch contract. Examples:

- a visual review can use generated-image enrichment or promoted vision-analysis evidence
- an audio confirmation can use generated-audio identity plus transcript evidence
- a source edit can use the selected text/document artifact as active reference truth

If the dossier evidence is missing, stale, or insufficient, the runtime may promote a new evidence branch. It should not silently pretend the evidence already exists.

## Explicit Recovery

Recovery is user-triggered and branch-scoped. It preserves the original request intent and reopens only an existing failed branch.

The failure side exposes `recovery_state.status == "candidate"`. The retry endpoint promotes exactly one candidate into `recovery_state.status == "attempting"` and records a `recovery_attempt`. This is preparation for operational recovery, not general automatic recovery.

Endpoint:

    POST /api/responses/<response_id>/late_fill/retry

Request:

    {
      "branch_id": "branch-image_generation-2",
      "exclude_instance_ids": ["flux-2"]
    }

Retry state:

    {
      "recovery_state": {
        "kind": "ollmo.late_fill_recovery_state",
        "status": "attempting",
        "trigger": "explicit_retry_endpoint",
        "branch_id": "branch-image_generation-2",
        "capability": "image_generation",
        "promotion_required": false,
        "auto_execute": false
      },
      "recovery_attempt": {
        "kind": "ollmo.late_fill_recovery_attempt",
        "trigger": "explicit_retry_endpoint",
        "branch_id": "branch-image_generation-2",
        "capability": "image_generation",
        "preserve_intent": true,
        "auto_execute": false,
        "failed_instance_id": "flux-2",
        "excluded_instance_ids": ["flux-2"]
      }
    }

Behavior:

- rejects missing, unknown, or non-retryable branches
- moves the selected failed branch back to `pending_branches[]`
- keeps completed branches and artifacts intact
- carries `failed_instance_id` and `excluded_instance_ids` into late fill route resolution
- returns the current canonical lookup payload and lets the existing lookup poller observe completion or failure

## Control Gating

Required session controls are hard requirements only after Ollmo has applied safe defaults and prompt-derived hints.

Safe defaults:

- `tts_voice` may use `default_first_option` when a backend exposes a speaker list.
- generic `tts_instruct` may default to `Use a natural, conversational voice.` for text-to-speech when no stricter voice description is present.

Hard blocks:

- explicit custom voice or speaker requirements without a valid default remain `missing_session_controls`.
- controls without a safe default or prompt-derived value remain blocking.
