# Project Participation

Ollmo 0.1.0 is published so people can inspect it, use it, study it, reproduce
results, and make independent forks. It is not currently operated as a managed
community project.

The project is maintained by one independent person with limited capacity.
Public issues and pull requests are therefore not solicited, and
submission does not imply that a report or change will be reviewed, discussed,
merged, or supported. There is no response-time or support commitment.

## Authorship and Maintenance

Ollmo was conceived, designed, created, built, and developed by
[@fl0ri0](https://github.com/fl0ri0). fl0ri0 currently maintains Ollmo and
decides release scope, project direction, and whether any external change is
considered. This boundary may change only when real maintainer capacity exists.

Ollmo is human-led and AI-enhanced. AI tools have helped explore, implement,
test, and document the project; product direction and release decisions remain
with fl0ri0. The project has no company, research-lab, or funded-team
backing, and publication does not claim a formal security audit or support
organization.

The public runtime principles and patterns are documented in
[`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) and
[`docs/PATTERNS.md`](docs/PATTERNS.md). The current test and evidence boundary
is documented in [`docs/TESTING_PROTOCOL.md`](docs/TESTING_PROTOCOL.md).

## Independent Work and Research

Forks, independent experiments, benchmark work, and research replications are
welcome under the repository license. Preserve enough environment, model,
prompt, configuration, and artifact information for others to understand what
was tested, and cite Ollmo using [`CITATION.cff`](CITATION.cff) when its ideas,
software, or evaluation artifacts contribute to published work. Do not publish
private prompts, credentials, or user data.

## Runtime Truth Is the Acceptance Boundary

Ollmo does not accept model prose as proof that work happened. Changes in a
fork that touch responses, artifacts, routing, or lifecycle state should
preserve the authoritative runtime surfaces: outputs, artifacts, response
frames, lifecycle state, closure review, and late-fill state.

Validation should be proportionate to the change. Run the narrowest relevant
tests first and include broader integration evidence when a change crosses
runtime or public-contract boundaries.

## Security Reports

Do not put suspected vulnerabilities or sensitive reproduction data in a
public issue. Follow [`SECURITY.md`](SECURITY.md).

## Unsolicited Contributions and License

The project is licensed under the Apache License 2.0. If someone nevertheless
submits an intentional contribution for possible inclusion, they retain
copyright in that contribution. Unless explicitly stated otherwise, the
submission is provided under the Apache License 2.0, consistent with Section 5
of that license. Submission still creates no review or merge commitment. Ollmo
does not currently require copyright assignment, a Contributor License
Agreement, or a Developer Certificate of Origin.

If a contribution cannot be provided under those terms, say so clearly before
submitting it for inclusion.

## Research, Talks, and Collaboration

For research collaboration, talks, panels, or work around Ollmo, use the
public contact methods on [@fl0ri0's GitHub profile](https://github.com/fl0ri0).
Contact is an invitation to make a respectful approach, not a promise of a
response, technical support, or project management.
