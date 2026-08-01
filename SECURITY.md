# Security Policy

Ollmo 0.1.0 is experimental, local-first software intended for one trusted
user on one machine.

## Local Boundary

The web control plane binds to `127.0.0.1` by default. Do not expose it through
a public interface, reverse proxy, tunnel, or shared host without a separate
security review and an authentication/authorization layer. Remote and
multi-user deployment are not supported in 0.1.0.

Runtime state, prompts, history, logs, and generated artifacts can contain
sensitive information. Keep the Ollmo directory private and review files
before sharing them.

## ChatGPT Through Codex and Cloud Processing

Ollmo stays local by default. The optional ChatGPT route is disabled until the
user explicitly enables the external connection. When ChatGPT is selected for
a turn, the prompt, context Ollmo promotes for that turn, and only the local
files or Ollmo artifacts explicitly selected for that turn leave the machine
and are processed by OpenAI.

Ollmo:

- discovers a Codex executable and asks that executable for login status;
- does not read, copy, or persist provider credentials, OAuth tokens,
  keychain entries, or Codex authentication files;
- does not require a separate provider API key for this route;
- hands recognized images to Codex through its native image-input path;
- copies other selected regular files into a temporary per-request working
  directory configured read-only for the Codex run;
- accepts at most 5 selected files, up to 100 MiB each and 250 MiB in total,
  and rejects URLs, folders, and symbolic links;
- returns text output through this route;
- runs Codex ephemerally while ignoring project rules and user provider
  projections;
- does not set a fixed model.

The read-only setting describes the temporary working directory used for the
request; it is not a general remote- or multi-user-isolation guarantee. A
regular file being accepted also does not guarantee that the selected model
and its available tools can interpret that format semantically.

Review every selected file or artifact before sending it, and do not include
secrets in prompts or selected files. Disable the ChatGPT connection when cloud
processing is not intended. Normal request teardown removes the temporary
working copy; it does not delete the original Ollmo artifact or make any claim
about deletion by the external service.

## Release Archives

Build archives only with `scripts/build_release_archive.py`. The builder uses
an explicit allowlist and excludes runtime state, logs, artifact contents,
caches, plans, hidden files, and the active model registry. It generates a
fresh empty registry, the standard empty artifact bucket structure, and
`MANIFEST.sha256`, then verifies the archive.

Every release candidate includes the reviewed Apache-2.0 `LICENSE`, `NOTICE`,
machine-readable citation metadata, project-participation policy, and
third-party notices. Publish mode
also fails closed until the changelog date is finalized and an explicit
release target is present. The builder never uploads an archive.
Verify-only also enforces named member-count, per-file, compressed-size, and
expanded-size safety bounds before extraction.

Never add credentials, private keys, environment files, personal runtime
state, or generated user artifacts to the source tree in order to make them
part of a release.

## Reporting a Vulnerability

Do not disclose a suspected vulnerability in a public issue. If the repository
offers GitHub private vulnerability reporting, use **Security > Report a
vulnerability**. Otherwise, preserve a minimal reproduction without user data
and use the public contact methods on
[@fl0ri0's GitHub profile](https://github.com/fl0ri0) only to request a private
channel. Do not include vulnerability details in that initial public contact.
