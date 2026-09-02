# Lighthouse Audio Roundtrip

**Status:** verified reference run  
**Verified:** 2026-09-01  
**Response:** `resp_1788207207762_eb84aee2876de8`  
**Final frame:** `resp_1788207207762_eb84aee2876de8:frame-2`

## Request

```text
Write a short two-sentence English field note about fog surrounding a coastal lighthouse. Create one local WAV audio artifact that reads the complete note in a calm, natural voice. Then transcribe that generated audio back into text. Return the original note, the audio, and the transcription together.
```

This is a natural three-phase request. It does not provide the text to be spoken,
name a model, prescribe routing, or include Ollmo-specific recovery instructions.

## Result

Ollmo first wrote the field note:

> Thick, grey fog rolls heavily against the base of the coastal lighthouse, obscuring the lantern room from view. The rhythmic pulse of the light struggles to pierce through the dense, damp atmosphere.

It then spoke that exact text as one VoiceDesign sequence and passed the resulting
WAV—not the expected text—to Whisper for transcription.

Whisper returned:

> Thick, gray fog rolls heavily against the base of the coastal lighthouse, obscuring the lantern room from view. The rhythmic pulse of the light struggles to pierce through the dense, damp atmosphere.

- [Play or inspect the generated WAV](artifacts/audio/narration.wav)
- [Inspect the saved transcription](artifacts/transcripts/transcript.md)

Verified publication copies are included in this package; the original runtime artifacts remain canonical.

## Runtime Truth

| Evidence | Verified state |
| --- | --- |
| Response lifecycle | `completed`, terminal |
| Late Fill | `completed`; 2 branches completed, 0 failed |
| Final materialization contract | `fulfilled` |
| Graph Closure | `fulfilled`; 3 of 3 obligations fulfilled |
| Actionable repair | none |
| Surface state | `fulfilled`; 0 open, blocked, or repair-pending items |
| Public outputs | original note, 1 WAV artifact, and 1 transcription artifact |
| Dependency chain | phase 1 text → phase 2 TTS → phase 3 STT |
| Audio artifact | exists; 522,284 bytes; SHA-256 `152656608fbf…bae832ccbfe` |
| Transcript artifact | exists; 200 bytes; SHA-256 `df5d2d403c8c…a5c7fd3f586` |
| Image topology | no image branch |

The WAV exists at 522,284 bytes with SHA-256
`152656608fbf…bae832ccbfe`. The transcription exists at 200 bytes with SHA-256
`df5d2d403c8c…a5c7fd3f586`. The saved files match the recorded artifact evidence.

## Monitor Report Snapshot

The Ollmo run monitor recorded verdict `clean` at
`2026-08-31T20:18:44.976541Z`.

- Response start: `20:13:27Z`
- Initial Gemma text and graph completion: `20:15:02Z`, about 1 minute 35 seconds
- TTS execution: one Qwen3-TTS VoiceDesign branch, about 5.9 seconds
- STT execution: one Whisper Large V3 branch, about 3.6 seconds
- Terminal output: `20:16:22Z`, about 2 minutes 55 seconds after start
- Post-response hygiene complete: `20:16:51Z`, about 3 minutes 24 seconds after start
- Both artifact files exist and their recorded checksums match
- No failed branch, unmet contract, missing file, SHA mismatch, repair/requeue
  event, weak freeze, or actionable repair was observed

Accepted learning, reconsideration, and graph-repair proposals remained advisory
or rejected and did not create executable work or alter the fulfilled result.

The complete monitor records remain searchable by response ID in:

- [Published monitor snapshot](monitor-report.json)
- [Published monitor snapshot](monitor-report.json)

## Audio Evidence

The runtime recorded the final backend prompt as `tts_semantic_source`. Its digest
matches both the original phase output and the source digest used for audio
integrity verification.

- Model type: VoiceDesign
- Generation scope: `single_sequence`
- Adaptive limit: `max_tokens=400`
- Nominal duration: 10.88 seconds
- Effective active signal: 9.7 seconds
- Total silence: 1.18 seconds
- Longest internal silence: 0.5 seconds
- Trailing silence: 0.18 seconds
- Inferred audio tokens: 136 of 400
- Generation limit reached: no
- Integrity result: `TTS_AUDIO_INTEGRITY_PASSED`

The audio is materialization-eligible and its saved SHA-256 matches runtime truth.

## Roundtrip Fidelity

The STT branch declares the TTS phase as its direct dependency. Runtime evidence
binds the transcription to `branch-text_to_speech-1`, its exact source digest,
and the concrete generated WAV.

Whisper returned the complete two-sentence note. Its only lexical difference is
the equally valid American spelling `gray` in place of British `grey`.

- Fidelity status: `matched`
- Reason: `TTS_STT_SEMANTIC_MATCH`
- Sequence ratio: `0.994845`
- Token precision, recall, and F1: `0.96875`
- Overlap: 31 of 32 normalized source tokens
- Negation consistency: passed

The expected text was verification evidence; it was not injected into the STT
request as a substitute for transcription.

## Diagnostic Note

The Late Fill snapshot retains the transitional field
`missing_artifact_type="text"` even though the final materialization contract is
fulfilled and no artifact is missing. Canonical lifecycle, Closure, artifact,
surface, and monitor truth all agree that this value is non-actionable residue.

## Why This Is a Reference

This run demonstrates a complete local roundtrip rather than simple file
existence: a chat model authors new text, TTS binds and speaks that exact text,
STT consumes the concrete generated WAV, and deterministic runtime evidence
compares the actual transcript with the bound source. The response freezes only
after audio integrity, semantic fidelity, artifact checks, Closure, and the run
monitor agree.

## Publication Package

This directory is a sanitized, immutable publication copy of the reviewed run. It does not replace the original response frame or runtime artifacts.

- [Exact reviewed prompt](prompt.txt)
- [Sanitized final response truth](response.json)
- [Sanitized independent monitor snapshot](monitor-report.json)
- [Package checksums and provenance](manifest.json)
