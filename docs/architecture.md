# Architecture

## Boundaries

The system has four deliberate boundaries:

1. **HTTP adapter** validates a versioned request, assigns a request UUID, and
   returns the OpenAPI-defined envelope.
2. **Router** resolves an explicit model/backend/device/precision combination.
   It does not silently substitute another combination.
3. **Worker supervisor** maintains queues and exactly one resident inference
   worker/profile. It drains and closes that owner before switching identities.
4. **Backend adapter** owns preprocessing, tensor execution, decoding, and
   backend capability reporting. The initial adapters run in Python because
   PyTorch and future native MLX execution are Python-native.

The boundaries are language-neutral: HTTP/OpenAPI is the process boundary,
JSON-compatible Pydantic models are the contract authority, and model
manifests are TOML. The TypeScript package is a Zod-validated transport client;
it does not duplicate tokenization, decoding, or model logic.

## Control-plane language decision

Rust, Go, Zig, and Zod are valid implementation choices. The initial release keeps
HTTP, scheduling, and model ownership in one Python process because that is the
lowest-complexity and highest-reliability design for the current workload:

- device inference dominates HTTP and queue bookkeeping costs;
- in-process batching avoids copying text and results across IPC twice;
- cancellation and model lifetime share one ownership domain;
- wheels plus Docker cover the current distribution targets;
- one runtime avoids shipping and supervising a native daemon alongside the
  required Python/PyTorch environment on Windows, macOS, and Linux.

A Rust or Go daemon becomes materially better when measurements show control
plane saturation, hard multi-process resource isolation is required, or a
single native executable must supervise several independently crashing model
workers. That migration does not change callers: a daemon can implement the
checked-in OpenAPI contract and supervise Python backend workers over a private,
versioned IPC protocol. Rust is preferred for a single-binary, low-overhead
daemon; Go is preferred when simpler operations and cross-compilation outweigh
resident-memory and FFI concerns. Neither language should own model
preprocessing or decoding unless parity with the Python oracle is demonstrated.

The same threshold applies to Zig. A Zig HTTP/control plane could reduce
startup size or queue-management overhead, but it would still call the
PyTorch/Metal/CUDA runtime that owns model deserialization, device transfer,
and tensor kernels. The measured workload is dominated by those operations, so
rewriting the current control plane in Zig is not expected to materially
improve model load or steady-state throughput. Zig would need a maintained
LibTorch/C++ bridge or a separately validated native backend; either adds
cross-platform ABI and parity risk without removing the model runtime.

The checked-in [`openapi.json`](../openapi/openapi.json) is generated from the
Python server and verified in CI. This prevents a future daemon or client
generator from defining a second public contract.

## Worker and batching model

Each process has exactly one owning model worker. This avoids duplicate weights
and concurrent mutation of accelerator state. Requests enter a bounded
queue and are grouped only when their ownership key and operation are
compatible. A batch closes at the earliest of a configurable item limit or a short latency
deadline. Compatibility includes operation, schema, and options.

Backpressure is explicit: a full queue produces a retryable overload response.
Cancellation removes queued work when possible; cancellation after dispatch
does not imply that device execution can be interrupted. Per-request results retain their original order and request IDs. Requests are
never transparently replayed. Run one server process per device; multiple
processes would each own a separate model copy.

## Model lifecycle

The operator may install a manifest entry before serving:

`logical ID -> source revision -> files with size and SHA-256 -> license`.

Acquisition writes to a staging location, verifies all metadata, then
atomically publishes an immutable directory keyed by a manifest digest.
`models install` performs acquisition. Workers receive resolved local paths,
verify file size and SHA-256 at load, and never resolve mutable branch names or
fetch from the network.

The public management API additionally exposes a curated official Fastino
catalog. Refresh only caches remote metadata. A missing-model event directs the
client to a separate asynchronous download endpoint that requires explicit
approval and the configured destination. Completed provider snapshots record
every file size and SHA-256 digest and are re-verified on first use. Neither
refresh nor inference changes the pinned active revision.

## Failure semantics and observability

Routing errors distinguish unknown model, unsupported capability, unavailable
device, and failed integrity checks. Responses expose the selected backend and
precision. Metrics cover queue depth, batch size, completions, cancellation,
rejection, and failures without input-derived labels.

See [protocol](protocol.md) and [operations](operations.md).
