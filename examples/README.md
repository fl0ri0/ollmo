# Ollmo Reference Runs

This directory records selected live Ollmo runs whose success is established by
runtime evidence, not by model prose or visual plausibility alone. These records
are human-facing examples, distinct from test fixtures, evaluation corpora, and
self-learning state.

A reference run is admitted only when its final response frame, Closure state,
materialization contract, saved artifacts, and bundle checks agree. Media-specific
integrity evidence and clearly labelled user observations may supplement that
runtime truth.

## Evidence included with 0.1.1

The five packages below retain their original August/September execution dates,
response/frame identities, artifacts and manifests. Inclusion in 0.1.1 does not
make them fresh 0.1.1 live runs. Release preparation rechecks the publication
copies and their checksums; it does not regenerate the media or replay the prompts.
See the [release notes](../docs/RELEASE_NOTES_0.1.1.md#validation-and-reference-evidence)
for the distinction between current checks, historical live coverage and benchmarks.

## Curated Examples

- [Echoes of the Pass](reference-runs/2026-08-31-echoes-of-the-pass/README.md) — a
  single-response local website with two generated images, one generated English
  WAV narration, HTML, CSS, working relative media links, and a verified bundle.
- [Evening Rain](reference-runs/2026-08-31-evening-rain/README.md) — a two-phase
  text-and-audio response whose complete three-sentence reflection is spoken in
  one VoiceDesign sequence with verified source binding and continuous voice color.
- [Lighthouse Audio Roundtrip](reference-runs/2026-08-31-lighthouse-audio-roundtrip/README.md)
  — a three-phase text-to-speech-to-transcription response with an exact generated
  source dependency, a valid local WAV, and deterministic roundtrip fidelity evidence.
- [Funny Animal Selfies](reference-runs/2026-09-02-funny-animal-selfies/README.md) — an
  open-ended three-image request in which Ollmo chooses three distinct scenes,
  generates each image, inspects each artifact independently, and publishes one
  combined evidence-grounded report.

- [Saved JSON → actual read → derived HTML](reference-runs/2026-09-07-saved-json-read-html-verified/README.md)
  — an explicit Save/Read/Create-consumer request with exactly two deliverables,
  captured JSON bytes bound to the HTML consumer, fulfilled Closure, and a clean
  independent monitor report. Includes a verified local bundle.

Each listed directory is a self-contained publication package with the reviewed prompt,
copied public artifacts, a sanitized final-response projection, an independent
monitor snapshot, and a checksum manifest. Echoes of the Pass also includes its
openable local website bundle. The canonical response frame and original saved
artifacts remain authoritative; these publication copies do not replace runtime
truth.
