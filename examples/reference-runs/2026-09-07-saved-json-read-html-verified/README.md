# Saved JSON → actual read → derived HTML

Functional live reference, inspected on 2026-09-07: **PASS**.

Response: `resp_1788803596168_832ad3bf8fc31`.
Observed current frame: `resp_1788803596168_832ad3bf8fc31:frame-2` (sequence 2).

## Request

```text
Save read-check.json with values [7,11]. Read the actually saved read-check.json again. Create read-check.html from the read data with a table containing both values and their sum. Use self-contained HTML with embedded CSS and no external resources. Return exactly these two files.
```

## Result and evidence

- Exactly two deliverables: JSON containing values `[7,11]`, and
  HTML with table values 7, 11 and sum 18. CSS is embedded;
  no external resources or separate CSS artifact.
- The JSON producer is `branch-text_artifact-2`, phase-3. Its actual saved read
  binds captured UTF-8 bytes, digest and artifact identity to HTML consumer
  `branch-text_artifact-1`, phase-2. Execution order follows the dependency,
  not the numeric phase labels.
- Stored consumption evidence below records the read bytes, exact
  producer/consumer identities, bound input digest and saved output identity.
  The existing `saved_file_consumption_artifact_issue` verifier returned no issue
  during passive inspection against the canonical artifacts. Content agreement
  alone was not the acceptance criterion.
- Lifecycle and Late Fill are `completed`; materialization and Closure are
  `fulfilled`. The consumer closure check is `saved_file_consumption_verified`.
- Both copied deliverables matched the original artifact bytes and bundle hashes.
  Package checksums cover this documentation copy.

## Scope

The independent monitor recorded `clean` for this exact frame on
2026-09-07T18:01:24.146200Z. The standard exporter independently checks exact
frame, prompt, monitor and artifact identity before admitting this package.

This request uses English `Save`, an explicit named saved-file `Read`, and
`Create ... from the read data` for the consumer. It does not live-test `Create`
as the producer verb or arbitrary language formulations. This is a functional
reference, not a performance comparison. Successor lineage, recovery and
missing/corrupt evidence scenarios are not claimed as exercised.

## Captured dependency evidence

The following sanitized inspection copy preserves the runtime read/consumption
binding. Canonical paths are provenance labels, not executable paths. The
original runtime frame remains authoritative.

```json
{
  "contract": {
    "consumer_branch_id": "branch-text_artifact-1",
    "consumer_instruction": "Create read-check.html from the read data with a table containing both values and their sum. Use self-contained HTML with embedded CSS and no external resources. Return exactly these two files.",
    "consumer_phase_id": "phase-2",
    "consumer_request": {
      "extension": "html",
      "source_name": "read-check"
    },
    "kind": "ollmo.saved_file_dependency",
    "producer_branch_id": "branch-text_artifact-2",
    "producer_phase_id": "phase-3",
    "producer_request": {
      "extension": "json",
      "source_name": "read-check"
    },
    "version": 1
  },
  "input_sha256": "6bbf95ca3d99fb2b6548bd6887ac2e66ff880b7fdf18dbb46ee2f99baf45c93c",
  "output": {
    "path": "canonical-artifact:20260907T175521Z_chat_text_artifact_ornith-1.5_9b_read-check.html",
    "sha256": "9851fbc10d9a4ed572ac35728fa1ef3c31c87db7d538d8be34596482b912e9bb",
    "size_bytes": 1071
  },
  "read": {
    "artifact_id": "text_0c14ec7b99ca1fd285efe3f1",
    "artifact_ref": "artifact:text_0c14ec7b99ca1fd285efe3f1",
    "branch_id": "branch-text_artifact-2",
    "consumer_branch_id": "branch-text_artifact-1",
    "consumer_phase_id": "phase-2",
    "encoding": "utf-8",
    "path": "canonical-artifact:20260907T175454Z_chat_text_artifact_ornith-1.5_9b_read-check.json",
    "phase_id": "phase-3",
    "producer_branch_id": "branch-text_artifact-2",
    "producer_phase_id": "phase-3",
    "response_id": "resp_1788803596168_832ad3bf8fc31",
    "sha256": "b1ba85eba8552c657bda618fd13c7e72557a0f3c55e7a1d56fc90d7d7cf23839",
    "size_bytes": 36,
    "source_response_id": "resp_1788803596168_832ad3bf8fc31",
    "utf8_base64": "ewogICJ2YWx1ZXMiOiBbCiAgICA3LAogICAgMTEKICBdCn0K"
  },
  "status": "consumed"
}
```

## Publication Package

This directory is a sanitized, immutable publication copy of the reviewed run. It does not replace the original response frame or runtime artifacts.

- [Exact reviewed prompt](prompt.txt)
- [Sanitized final response truth](response.json)
- [Sanitized independent monitor snapshot](monitor-report.json)
- [Package checksums and provenance](manifest.json)
