# Third-Party Components

Ollmo's own source code and project documentation are licensed under the
Apache License 2.0. Third-party components keep their own licenses.

## Bundled with the Source Distribution

### Montserrat Black

The standalone project page includes a local Latin webfont subset of
Montserrat Black at weight 900.

- Copyright 2024 The Montserrat.Git Project Authors
- Source: <https://github.com/JulietaUla/Montserrat.git>
- License: SIL Open Font License 1.1
- License text: [`site/fonts/OFL.txt`](site/fonts/OFL.txt)
- Provenance and file hash: [`site/fonts/README.md`](site/fonts/README.md)

Montserrat is not relicensed under Apache-2.0.

## Installed Separately

Python packages installed through `requirements.txt`, local AI runtimes,
model weights, and models are not bundled with Ollmo. They remain subject to
their respective upstream licenses and terms.

### Optional scanned-PDF rendering

Ollmo can use PyMuPDF for multi-page scanned-PDF rendering, but PyMuPDF is not
part of the default Ollmo installation. PyMuPDF is offered upstream under the
GNU Affero General Public License 3.0 or a commercial Artifex license. Review
and accept the applicable upstream terms before installing it separately.

Without PyMuPDF, Ollmo retains text-layer PDF extraction through `pypdf` and a
limited macOS first-page rendering fallback when available.

## Loaded at Runtime from External CDNs

The current dashboard requests Google Fonts, Axios, and Font Awesome from
their upstream CDNs when the dashboard is opened with network access. The
architecture diagram requests JetBrains Mono from Google Fonts. These
resources are not bundled in the Ollmo source archive and remain under their
respective upstream licenses and terms.

The standalone repository landing page under `site/` does not make those
requests; its display font is bundled locally with the license and provenance
described above.
