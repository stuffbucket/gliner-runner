# Operations

## Installation and model acquisition

Use `mise install` for contributor tools and a release artifact or container
for deployment. Models are installed separately. An operator-approved manifest
must pin the upstream revision and list each file's size, SHA-256 digest, and
license metadata. Download into staging, verify, then publish to a
content-addressed, read-only cache. Offline operation should be possible after
installation.

Never mount a shared cache writable by an untrusted tenant. Never place tokens
in manifest URLs. Mirror artifacts when upstream availability is operationally
important, subject to their licenses.

Package registry configuration and container secret mounts are defined in the
[registry policy](registries.md). Registry credentials are build-time inputs,
not application configuration, image metadata, or runtime environment.

## Configuration

The server reads these environment variables when corresponding CLI options are
not supplied:

| Variable | Default |
| --- | --- |
| `GLINER_RUNNER_HOST` | `127.0.0.1` |
| `GLINER_RUNNER_PORT` | `8090` |
| `GLINER_RUNNER_DEVICE` | `cpu` |
| `GLINER_RUNNER_QUEUE_CAPACITY` | `256` |
| `GLINER_RUNNER_MAX_BATCH_SIZE` | `16` |
| `GLINER_RUNNER_BATCH_WINDOW_MS` | `4.0` |
| `GLINER_RUNNER_MANIFEST_DIRECTORY` | platform user config/models |
| `GLINER_RUNNER_MODEL_STORE` | platform user cache/models |

CLI options are the deployment interface for `serve`; environment loading is
used by the importable ASGI app. Unsupported values fail at startup or
capability negotiation.

Supported PyTorch profiles are CPU/FP32, CUDA/FP32, CUDA/FP16, CUDA/BF16, and
the parity-gated Apple Silicon MPS/FP16 profile. Device selection is
process-wide and explicit. An unsupported precision for the selected device is
rejected; the runtime never falls back to CPU or changes precision.

## Health and lifecycle

`GET /healthz` reports process admission health. Models load lazily on their
first request after all files are re-verified. On shutdown, admission stops,
queued work drains, and model ownership is released. Orchestrators should use
their own termination grace period.

A model update creates a new manifest digest alongside the old one. Install the
new generation before changing the logical manifest entry and restarting the
service. Do not mutate files beneath a loaded worker.

## Containers

The `Dockerfile` is a CPU-safe packaging baseline and assumes the Python
project surface supplies a buildable package and a `gliner-runner` console
entry point. Accelerator deployments need runtime-specific base images,
drivers, and scheduling. They must retain the non-root process, immutable
application layer, external model volume, explicit device selection, and
health behavior. The image contains no model weights. Its container command
explicitly listens on all container interfaces; publish it only to loopback
unless an authenticated, encrypted proxy protects the service.

## Observability and privacy

Log structured metadata, selected execution identity, timings, and error codes.
Do not log input text, schemas containing user data, outputs, authorization
headers, or signed model URLs by default. Bound metric cardinality. Record the
model digest, backend, precision, and runtime versions in diagnostic output.

Capacity planning uses queue wait, batch fill, and tail latency. Benchmark
results are meaningful only with
hardware, software, digest, precision, workload, warm-up, concurrency, and
batch policy recorded.

## Integration and benchmark safety

Default CI and `mise run check` are offline. `mise run integration` and
`mise run benchmark:real` are explicit opt-ins. Integration requires a
separately installed model and `GLINER_RUNNER_INTEGRATION_REQUEST`. The offline
`mise run benchmark` command uses a fake backend and downloads nothing. The
real-model command requires `GLINER_RUNNER_BENCHMARK_MODEL` to identify a local
pinned snapshot; see [real-model characterization](benchmarks.md). CI jobs
running integration must be manually or schedule-triggered, use
least-privilege credentials, verify artifacts, and publish no sensitive input
or model file.
