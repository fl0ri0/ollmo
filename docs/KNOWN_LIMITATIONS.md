# Known Limitations in Ollmo 0.1.0

This list records current, non-blocking limits of the first release candidate.
It is not a roadmap or a general backlog.

## Platform and Packaging

- The primary tested platform is macOS on Apple Silicon. Windows, Linux, and
  Intel Mac behavior is not guaranteed for 0.1.0.
- Distribution is a source archive. It is not a signed or notarized macOS app,
  a container image, or a package-manager release.
- Backend packages and model weights are not bundled. A capability can be
  listed but remain unavailable until its local backend and model are
  installed.
- Multi-page scanned-PDF rendering can use PyMuPDF, but PyMuPDF is not part of
  the default installation because its upstream terms are AGPL-3.0 or a
  commercial Artifex license. Without it, text-layer extraction still uses
  `pypdf`, and macOS may provide a limited first-page rendering fallback.
- The bundled Ollmo skill is Codex-specific in 0.1.0 and must be copied into
  the user's Codex skills directory manually; marketplace/plugin distribution
  and other agent integrations are outside this release.
- The dashboard currently requests Google Fonts, Axios, and Font Awesome from
  external CDNs when opened with network access. The architecture diagram also
  requests JetBrains Mono from Google Fonts. The standalone `site/` landing
  page uses bundled local assets and does not make those requests.

## Runtime

- Ollmo is intended for local, single-user operation. Remote exposure and
  multi-user isolation are not supported.
- Flask remains the current control-plane implementation.
- Optional models and modalities can be unavailable or degraded. This is
  acceptable only when status and recovery evidence remain truthful.
- Model output quality and latency are provider-dependent and are not
  deterministic.

## Optional ChatGPT Route

- The ChatGPT execution route uses Codex, is optional and cloud-based, and must
  be explicitly enabled. It accepts a prompt plus files or Ollmo artifacts
  explicitly selected for the current turn. Referential turns may also include
  context promoted by Ollmo's context gate, but the provider response is text
  only.
- Ollmo can reuse authentication owned by the ChatGPT app or Codex CLI, but it
  cannot guarantee that the app's internal executable location will remain
  unchanged. Discovery fails closed and then tries the documented fallback.
- Automatic model selection means the invoked Codex executable chooses its
  current default. Ollmo neither receives the exact GPT variant nor mirrors
  the model selected in an already open ChatGPT conversation. Model prose is
  not authoritative identity evidence.
- The dashboard's ChatGPT tab is an external conversation target. It has no
  local lifecycle controls and is intentionally excluded from Start, Stop,
  Pull, Delete, and Arena.
- Direct ChatGPT tab turns are ephemeral and independent. Displayed history is
  retained for inspection. For referential turns, Ollmo may promote bounded
  relevant context into the current request, but it does not resume a hidden
  provider session or resend the entire conversation automatically.
- Recognized images are attached through Codex's native image-input path.
  Other selected regular files are only made available inside a temporary
  working directory configured read-only for the Codex run. Acceptance does
  not guarantee that every document, audio, binary, or other file format can be
  interpreted semantically by the selected model and its available tools.
- The fixed request limits are 5 files, 100 MiB per file, and 250 MiB in total.
  URLs, folders, and symbolic links are not accepted.
- Direct API-key management and other external providers remain outside this
  optional route's 0.1.0 contract.

## Compatibility

- No compatibility is promised for private, saved, or historical builds that
  predate 0.1.0.
- Public `0.x` interfaces may evolve as the runtime contracts mature.

See [Release Scope](RELEASE_SCOPE.md) for the supported 0.1.0 release boundary.
