# VISION ALIGNMENT

Ollmo is a local-first AI runtime substrate for turning model output into
durable, inspectable work.

Models can propose, reason, and generate. Ollmo records what the runtime can
prove happened: which work became required, which outputs exist, which work
remains open, and why.

This is execution truth, not a guarantee that every model statement is
factually correct. The quality of generated content still depends on the
models and capabilities involved. Ollmo provides the state, evidence, and
review structure around that work.

## Frozen Moments, Fluid State

A normal chat often reduces a turn to prompt in, answer out.

Ollmo treats a response as a frozen moment inside a larger, fluid work state.

Between two visible replies, the runtime may contain:

- possible work that has not become required
- promoted obligations
- branches and dependencies
- open artifact slots
- materialized outputs
- pending or deferred work
- blocks and repair paths
- continuation state

Fluid does not mean unrecorded. Frozen does not necessarily mean finished.

A response can truthfully preserve an open state. Work may continue later
without pretending that it was already complete.

The central movement is:

```text
intent
  -> possibility space
  -> relevance and promotion
  -> obligations
  -> runtime work and evidence
  -> closure review
  -> response frame
```

## From Possibility To Obligation

Intent opens a possibility space.

A possibility can describe an output, a branch, a reference, a continuation,
or another useful direction. It may remain visible without becoming
executable work.

Only promotion turns a possibility into an obligation.

```text
possibility
  -> relevance
  -> promoted obligation
  -> runtime work
  -> review
  -> freeze
```

This distinction matters.

Not everything the system can imagine is work it owes. A reserved possibility
may remain available for later reconsideration, but it must not silently
become a task, consume a model, or be reported as unfinished work.

Once promoted, an obligation becomes part of the runtime contract. It must be
resolved truthfully as fulfilled, open, blocked, failed, repair-needed,
deferred, waived, or superseded.

## Context Follows The Same Rule

History is a resource, not a cage.

A prior message, artifact, or remembered fact may be available as a context
candidate. It does not automatically become part of the current request.

```text
possible continuity
  -> current relevance
  -> active context
  -> runtime use
  -> recorded state
```

Explicit references and clear follow-ups can promote prior context. Irrelevant
history can remain checked but not bound.

This keeps continuity available without allowing old conversations, artifacts,
or tool calls to become new intent on their own.

## Work Becomes A Graph

A response can contain more than one stream of prose.

Promoted obligations can become branches with their own dependencies,
evidence, and output contracts. A document may depend on images. A
transcription may depend on audio. A comparison may depend on several
completed results.

Each branch should receive the work it actually owes, not a replay of the
entire root request.

```text
branch possibility
  -> promoted branch obligation
  -> focused inputs and dependencies
  -> execution or materialization
  -> branch evidence
  -> recorded branch state
```

Different branches may be handled by different compatible runtime
capabilities or model instances. Their outputs still belong to one connected
response state.

A branch may also remain pending or be materialized later. Continuation does
not erase the earlier state; it carries the same work forward with visible
lineage.

## Move Between Scales

Ollmo should not inspect branches in isolation and forget the request that
gave them meaning.

Discovery may move from the surface request toward deeper intent, and from the
whole request toward the individual branch, dependency, artifact, or review
criterion. Integration must also move in the other direction: from evidence
back toward a practical action, and from locally fulfilled branches back
toward the whole current intent.

Three movements keep that process balanced:

- aspiration preserves the possibility of a coherent solution instead of
  collapsing immediately to the smallest visible action
- doubt asks what the evidence really proves
- commitment chooses the right-sized truthful transition once enough evidence
  exists

These movements guide attention. They do not create obligations or runtime
truth by themselves.

The balanced cadence is aspire, inquire, commit.

## Resolve Blocks By Preserving Intent

A block is not permission to rewrite the request.

It is also not permission to declare success, hide the missing work, or retry
the same broken action without new evidence.

The solution to a block is the block's own resolution.

Depending on runtime truth, that may mean:

- repairing a missing dependency
- repairing a bounded branch contract
- continuing an existing obligation
- requesting external input
- preserving the block for later work
- explicitly waiving the obligation
- superseding it with a justified replacement
- freezing the state honestly as blocked

Ollmo should resolve blocks gently, not violently. It should preserve the
user's intent and change only the smallest scope that the evidence requires.

The goal is not to force every branch into success. The goal is to make the
next truthful transition visible.

## Outputs Have To Exist

Model prose is useful evidence, but it is not automatically a materialized
result.

If the request requires a file, image, audio artifact, or another concrete
output, completion requires concrete runtime evidence.

For example:

- an image prompt is not an image
- a code block is not a saved file
- a placeholder path is not a resolved artifact
- a saved webpage is not complete if its required local links are broken
- producing fewer artifacts than the promoted contract requires is not full
  fulfillment

Ollmo records concrete outputs and their relationships so they can be
inspected, referenced, continued, and recovered.

## Closure Before Freeze

Closure Review compares the current work state with the obligations created
from the user's intent.

It asks:

- Which obligations were fulfilled?
- Which outputs and dependencies actually exist?
- Which work is still pending, deferred, blocked, failed, or repair-needed?
- Which obligations were explicitly waived or superseded?
- Do required links and artifact relationships resolve?
- Where needed, does the finished set still fit the current intent?

Closure Review is not a second free-form interpretation of the request. It is
a runtime review of contracts, evidence, outputs, dependencies, and current
state.

If proof is missing, the response remains open, blocked, or repair-needed. It
must not report fulfillment merely because a model says it is done.

Closure does not establish universal factual truth. It establishes what the
Ollmo runtime can prove about the execution and materialization of the work.

## The Response Frame

The response frame is the frozen instant.

It can preserve:

- the current intent
- the visible possibility space
- the obligations created through promotion
- the state of each relevant branch
- fulfilled outputs
- pending or deferred work
- blocks and repair state
- waivers and supersessions
- materialized artifacts and their evidence
- continuation and lineage

A later continuation can produce a successor state without silently rewriting
the evidence preserved at the earlier boundary.

The reply summarizes the moment. The response preserves the work.

## Authority Boundaries

Ollmo keeps proposal and authority separate.

- Models propose structure and produce candidate results.
- Planning opens and shapes the possibility space.
- Promotion determines what becomes owed work.
- Runtime execution and materialization produce evidence.
- Closure Review compares that evidence with the obligations.
- The runtime owns the recorded state.

A model can help decide what to try next, but model prose cannot prove that a
file exists, that a dependency resolved, or that an obligation was fulfilled.

Likewise, prior learning or earlier successful paths may help orient later
work, but they remain guidance. Current runtime evidence remains the authority
for the current response.

## North Star

Ollmo should not become a pile of unrelated special cases.

It should keep sharpening one repeated logic:

```text
possibility
  -> relevance
  -> promoted obligation
  -> runtime evidence
  -> review
  -> freeze
```

For continuity, the same logic is:

```text
possible context
  -> current relevance
  -> active context
  -> runtime evidence
  -> review
  -> freeze
```

For repair, it remains:

```text
visible block
  -> bounded resolution
  -> new runtime evidence
  -> review
  -> truthful next state
```

This is the concrete form of the original music image.

The model provides movement. Ollmo provides the recording surface, the state,
and the boundary at which one moment can be preserved.

The music continues between frozen moments. A response frame is the truthful
still captured at one point in that movement.
