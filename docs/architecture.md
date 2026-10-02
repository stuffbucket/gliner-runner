# Architecture

## Boundaries

The system has four deliberate boundaries:

1. **HTTP adapter** validates a versioned request, assigns a request UUID, and
   returns the OpenAPI-defined envelope.
2. **Router** resolves an explicit model/backend/device/precision combination.
   It does not silently substitute another combination.
3. **Worker supervisor** maintains queues and one inference worker for each
   loaded `(accelerator, model, backend, precision)` ownership key.
4. **Python backend** owns preprocessing, tensor execution, decoding, and
   backend capability reporting. PyTorch is the semantic reference.

The TypeScript package is only a transport client. It does not duplicate
tokenization, decoding, or model logic.

## Worker and batching model

Each model/device tuple has exactly one owning worker. This avoids duplicate
weights and concurrent mutation of accelerator state. Requests enter a bounded
queue and are grouped only when their ownership key and operation are
compatible. A batch closes at the earliest of a configurable item limit or a short latency
deadline. Compatibility includes operation, schema, and options.

Backpressure is explicit: a full queue produces a retryable overload response.
Cancellation removes queued work when possible; cancellation after dispatch
does not imply that device execution can be interrupted. Per-request results retain their original order and request IDs. Requests are
never transparently replayed. Run one server process per device; multiple
processes would each own a separate model copy.

## Model lifecycle

The operator installs a manifest entry before serving:

`logical ID -> source revision -> files with size and SHA-256 -> license`.

Acquisition writes to a staging location, verifies all metadata, then
atomically publishes an immutable directory keyed by a manifest digest.
`models install` performs acquisition. Workers receive resolved local paths,
verify file size and SHA-256 at load, and never resolve mutable branch names or
fetch from the network.

## Failure semantics and observability

Routing errors distinguish unknown model, unsupported capability, unavailable
device, and failed integrity checks. Responses expose the selected backend and
precision. Metrics cover queue depth, batch size, completions, cancellation,
rejection, and failures without input-derived labels.

See [protocol](protocol.md) and [operations](operations.md).
