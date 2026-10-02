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
| `GLINER_RUNNER_MODEL_PROVIDER_DIRECTORY` | model store/providers/huggingface |
| `GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS` | `0` (disabled; keep warm) |
| `GLINER_RUNNER_MEMORY_LIMIT_BYTES` | 75% of physical memory |

CLI options are the deployment interface for `serve`; environment loading is
used by the importable ASGI app. Unsupported values fail at startup or
capability negotiation.

Supported PyTorch profiles are discovered from the host. CPU/FP32 is always
eligible. Available Apple Silicon exposes parity-gated MPS/FP16 and MPS/FP32.
Available NVIDIA devices expose CUDA/FP32, CUDA/FP16 when their compute
capability is at least 5.3, and CUDA/BF16 only when PyTorch reports BF16
support. Device selection is process-wide and explicit. An unsupported
precision for the selected device is rejected; the runtime never falls back to
CPU or changes precision.

## Health and lifecycle

`GET /healthz` reports process admission health. Models load lazily on their
first request after all files are re-verified. On shutdown, admission stops,
queued work drains, and model ownership is released. Orchestrators should use
their own termination grace period.

Exactly one model/backend/precision/device identity may be resident in a
process. Loading a different identity stops admission to the resident worker,
drains in-flight work, closes it, and only then loads the replacement. The
default idle TTL is zero, so the resident model stays warm until replacement or
shutdown. A positive TTL evicts only after the worker has no active requests.
The one-shot `infer` CLI still closes its runtime after every invocation and is
not appropriate for repeated latency-sensitive calls.

The default memory ceiling is 75% of physical memory. The runtime rejects a
known model/profile before load when its parameter-based minimum exceeds the
ceiling. It also measures after load and unloads before returning
`model_memory_limit` if the ceiling is exceeded. CPU uses process RSS. Apple
unified-memory enforcement uses the larger of process RSS and MPS driver
allocation, never their sum; CUDA uses the larger of RSS and reserved device
memory. Inventory figures are planning floors, not peak guarantees.

## Curated inventory and downloads

`GET /v1/models` includes all three official Fastino Decide models even when
not installed. It distinguishes the repository's pinned revision, the latest
revision observed by an explicit refresh, locally complete revisions, and the
one currently loaded profile. Precision changes do not duplicate checkpoint
files: FP16/FP32 are explicit load profiles over the same immutable snapshot.

`POST /v1/models/refresh` performs the only remote catalog lookup. It updates a
small local metadata cache but neither downloads nor replaces weights.
Official Fastino repositories matching the Decide family but not yet in the
curated runnable set are returned as `discovered_models`; they are not
automatically trusted, downloaded, or made runnable.
Inference against a missing curated model returns a
`model_download_required` event. An application must obtain user/operator
approval and submit `approved: true` plus the exact destination reported by
inventory to `POST /v1/models/{model_id}/downloads`. Poll
`GET /v1/model-downloads/{job_id}` until completion. A different destination or
unapproved request is rejected. The download is revision-pinned and a complete
SHA-256 file inventory is written before the snapshot is considered available.
Job state is retained for the lifetime of the server process.

## Electron sidecar lifecycle

`@gliner-runner/client/sidecar` exposes a Node-only `GlinerSidecar` supervisor
for Electron's main process. It uses `spawn` with `shell: false`, waits for the
loopback `/healthz` endpoint, retains only a bounded tail of child diagnostics,
reports early exit and startup timeout explicitly, and terminates the child
with a configurable grace period. The inference transport remains the typed
HTTP/OpenAPI protocol; Electron IPC should proxy application requests from a
sandboxed renderer to the main process rather than exposing the local server or
child-process APIs directly.

The one-shot `gliner-runner infer` command accepts one JSON request from
`--request` or stdin, emits exactly one JSON response on stdout, and exits.
Operational logs and validation failures use stderr so they cannot corrupt a
successful machine-readable response.

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
separately installed model and `GLINER_RUNNER_INTEGRATION_REQUEST`;
`GLINER_RUNNER_INTEGRATION_DEVICE` selects `cpu`, `mps`, or `cuda` and defaults
to CPU. The smoke test micro-batches two different-length requests and verifies
distinct encoded input usage plus classification `outputTokens=0`. The offline
`mise run benchmark` command uses a fake backend and downloads nothing. The
real-model command requires `GLINER_RUNNER_BENCHMARK_MODEL` to identify a local
pinned snapshot; see [real-model characterization](benchmarks.md). CI jobs
running integration must be manually or schedule-triggered, use least-privilege
credentials, verify artifacts, and publish no sensitive input or model file.
