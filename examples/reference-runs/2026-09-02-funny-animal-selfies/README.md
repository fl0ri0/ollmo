# Funny Animal Selfies

**Status:** verified reference run  
**Verified:** 2026-09-02  
**Response:** `resp_1788366165828_153c2a8423dff`  
**Final frame:** `resp_1788366165828_153c2a8423dff:frame-2`

## Request

```text
Create a playful series of exactly three funny animal selfies. Choose the animals, settings, poses, and visual style yourself, but make the three images clearly distinct. Then inspect each generated image separately and briefly describe only what is actually visible, including any accidental readable text or obvious visual defect.
```

This is intentionally an open, natural prompt. It names no animal, setting,
image model, vision model, routing rule, output dimensions, or Ollmo-specific
recovery instruction.

## Result

Ollmo selected and materialized three clearly different scenes:

1. a capybara relaxing in a steaming hot spring with cucumber slices over its eyes;
2. a lemur taking a wide-angle selfie in rainbow sunglasses at a neon-lit party;
3. a bulldog in a red polka-dot birthday hat with frosting on its snout.

- [Open the capybara image](artifacts/images/image-01.png)
- [Open the lemur image](artifacts/images/image-02.png)
- [Open the bulldog image](artifacts/images/image-03.png)

Verified publication copies are included in this package; the original runtime artifacts remain canonical.

## Runtime Truth

| Evidence | Verified state |
| --- | --- |
| Response lifecycle | `completed`, terminal |
| Late Fill | `completed`; 7 of 7 branches completed, 0 failed, 0 cancelled |
| Final materialization contract | `fulfilled` |
| Graph Closure | `fulfilled`; all checked obligations fulfilled, waived, or superseded |
| Actionable repair | none |
| Public image outputs | exactly 3 distinct image artifacts |
| Evidence topology | phase 2/3/4 images → phase 5/6/7 one-to-one vision analyses → phase 8 terminal chat join |
| Terminal report | one combined three-part inspection |
| Consumed evidence projection | branch-local vision text retained in Late Fill but not repeated as separate public descriptions |
| Artifact identity | canonical final projection not blocked |
| Saved files | all three exist; 4,356,171 bytes total |

The root text output also retains the three generation prompts chosen by the
model. That preparation text is distinct from the inspection report; the three
consumed vision-analysis results themselves are not duplicated on the public
surface.

## Artifact Evidence

| Image | Dimensions | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Capybara | 1024 × 1024 | 1,384,818 | `078160af9e4850f1…1f98bbeb303c943` |
| Lemur | 1280 × 720 | 1,780,940 | `974250f22425878f…2442b4d3e4db67a` |
| Bulldog | 768 × 1024 | 1,190,413 | `38510729f797b08c…94dbe7e139786` |

Each image has its own saved path, producer phase, artifact identity, and direct
vision-analysis consumer. Image generation used one Z-Image branch and two Flux
branches. Vision analysis used one local MLX Gemma branch and two Ollama Gemma
branches; the final local MLX chat branch consumed exactly those three analyses.

The bulldog registry record preserves an older artifact alias alongside the
canonical output ref. Runtime artifact-identity truth reports no required
canonicalization and no blocked final projection, so the alias is durable
provenance rather than a duplicate public image.

## Inspection Evidence

The final joined report describes all three saved artifacts in order. It records
no readable text and no obvious visual defect in any image. Independent visual
review confirms that the three compositions are distinct and that no baked-in
words are visible.

The first description calls the cucumber slices attached to the sides of the
animal's head; visually they cover the eye area. This is a minor wording choice,
not an artifact, identity, count, or Closure defect.

## Monitor Report Snapshot

The Ollmo run monitor recorded verdict `clean` at
`2026-09-02T16:44:47.162777Z`.

- Response start: `16:22:45Z`
- Initial Gemma chat and graph completion: `16:24:24Z`, about 1 minute 39 seconds
- Image wave: `16:25:30Z` to `16:28:34Z`, about 3 minutes 4 seconds
- Terminal output: `16:33:03Z`, about 10 minutes 18 seconds after start
- Post-response hygiene complete: `16:33:57Z`, about 11 minutes 12 seconds after start
- Capability work: 1 chat branch, 3 image-generation branches, and 3
  vision-analysis branches
- Image execution: two Flux2 Klein branches and one Z-Image Turbo branch
- Vision execution: two local Gemma4 branches and one MLX Gemma4 branch
- Terminal join: one local MLX Gemma4 branch
- Saved artifact size: 4,356,171 bytes across 3 PNG files
- Registry parsing, file existence, checksums, artifact identity, and final
  projection checks passed
- No failed or pending branch, unmet contract, missing file, SHA mismatch,
  duplicate output, repair/requeue event, or weak-freeze condition was observed

The monitor also recorded accepted-learning and reconsideration material as
advisory only. Graph Closure and surface state remained `fulfilled`; no repair
authority was exercised and no successor work was created.

The complete observer records remain searchable by response ID in:

- [Published monitor snapshot](monitor-report.json)
- [Published monitor snapshot](monitor-report.json)

The monitor supplements but does not replace the authoritative final response
frame, Late Fill state, Closure state, artifact registry, and saved artifact
bytes.

## Qualitative Sign-off

The user confirmed that the rerun completed successfully. Direct inspection
found three coherent, playful images: the quiet spa portrait, saturated party
selfie, and warm birthday portrait are immediately distinguishable while still
forming a recognizable series.

## Why This Is a Reference

This run demonstrates Ollmo's open multimodal planning rather than execution of
three supplied image prompts. One short natural request leaves the subjects and
visual worlds open; the graph authors three concrete prompts, routes generation
across two local image-model families, binds three independent vision consumers
one-to-one to the saved artifacts, and joins their observations only after all
dependencies finish. Runtime evidence stays durable while the public result
avoids repeating each consumed inspection beside the final combined report.

## Publication Package

This directory is a sanitized, immutable publication copy of the reviewed run. It does not replace the original response frame or runtime artifacts.

- [Exact reviewed prompt](prompt.txt)
- [Sanitized final response truth](response.json)
- [Sanitized independent monitor snapshot](monitor-report.json)
- [Package checksums and provenance](manifest.json)
