# GLiNER Runner

GLiNER Runner is a local, backend-neutral Python runtime and HTTP service for
explicit, reproducible GLiNER inference. The initial release implements
constrained classification through Fastino GLiNER2 2.0's public PyTorch
`Classifier.batch_classify` API. It supports the three published Decide model
repositories when supplied through operator-owned, revision-pinned manifests.

## Guarantees

- no silent backend, precision, device, or model fallback;
- exactly one loaded model/profile owner per process, with safe draining before
  replacement and optional idle eviction;
- bounded queues, cancellation-aware admission, compatible-schema dynamic
  micro-batching, and explicit HTTP 429 backpressure;
- strict Pydantic request/result/schema contracts and reproducible OpenAPI;
- exact per-request `{inputTokens, outputTokens}` usage from the encoder batch,
  with classification output tokens fixed at zero;
- external models with exact revisions, sizes, SHA-256 checksums, licenses,
  and content-addressed storage;
- offline unit/parity-fixture tests; real-model integration is opt-in.

PyTorch is the correctness oracle. MLX and ONNX appear in the stable backend
enumeration for plugin compatibility, but are not registered or advertised.
See [backend policy](docs/backends.md).

## Install and develop

The supported development entry point uses
[mise](https://mise.jdx.dev/) and [pnpm](https://pnpm.io/). Node 24 is the
supported JavaScript runtime; `.nvmrc` selects the major release while
`mise.toml` pins the exact toolchain:

```sh
mise install
mise run setup
mise run check
```

For a Python deployment:

```sh
python -m pip install 'gliner-runner[pytorch]'
```

The PyTorch extra pins `gliner2[local]==2.0.0`; model artifacts are never part
of this package.

## Models and explicit downloads

`GET /v1/models` reports the curated official Fastino GLiNER2.5 Decide family,
host-supported CPU/GPU precision profiles, pinned and newly discovered
revisions, local availability, the currently loaded profile, estimated memory
floors, and measured score-drift guidance. `POST /v1/models/refresh` refreshes
only the cached provider metadata; it never installs or replaces a model.

Inference never downloads. A missing curated model returns
`model_download_required` with its exact revision, configured destination, and
download endpoint. Start a download only after user/operator approval:

```sh
curl http://127.0.0.1:8090/v1/models/decide-340m/downloads \
  -H 'content-type: application/json' \
  -d '{"approved":true,"destination":"<value from GET /v1/models>"}'
```

The `202` response contains a job ID. Poll
`GET /v1/model-downloads/{job_id}`. Downloads use an exact 40-character
revision, remain outside Git, and produce a local file/size/SHA-256 inventory
before becoming available. Provider credentials use Hugging Face's standard
environment/cache configuration and are never returned by the API.

Operator-owned mirrors remain supported through TOML manifests. Create a
manifest in the configured manifest directory (by default
`~/.config/gliner-runner/models`). Every URL must address the pinned revision,
not a moving branch:

```toml
format_version = 1
model_id = "decide-340m"
source = "https://huggingface.co/fastino/GLiNER2.5-Decide"
revision = "5a7adf72a23b4d311abae6ce050d7f0012bb3416"
license = "Apache-2.0"
backend = "pytorch"
precisions = ["fp32", "fp16", "bf16"]
operations = ["classify"]

[provenance]
upstream_model = "fastino/GLiNER2.5-Decide"
upstream_runtime = "gliner2==2.0.0"

[[artifacts]]
path = "config.json"
url = "https://huggingface.co/fastino/GLiNER2.5-Decide/resolve/5a7adf72a23b4d311abae6ce050d7f0012bb3416/config.json"
sha256 = "<64 lowercase hex characters>"
size = 1234
```

List every file required by the pinned local snapshot, then install explicitly:

```sh
gliner-runner models verify ~/.config/gliner-runner/models/decide-340m.toml
gliner-runner models install ~/.config/gliner-runner/models/decide-340m.toml
```

Inference verifies the installed files and never downloads a missing model.
Manifests are operator data because upstream artifact hashes and file sets can
change; this repository does not publish guessed checksums.

## Run

```sh
gliner-runner serve --device cpu
```

The default memory ceiling is 75% of physical RAM. Set
`GLINER_RUNNER_MEMORY_LIMIT_BYTES` or `--memory-limit-bytes`; known profiles
whose estimated residency exceeds the limit are rejected before loading, and
loads exceeding it are unloaded with a structured error. Models stay warm by
default. Set `GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS` or
`--model-idle-ttl-seconds` to evict the single resident model after an idle
period.

The validated Apple Silicon profiles are explicit MPS/FP16 and MPS/FP32:

```sh
gliner-runner serve --device mps
```

Requests sent to that process must specify `"precision": "fp16"` or
`"precision": "fp32"`. Capabilities are host-aware: unavailable accelerators
and unsupported architecture/precision pairs fail without falling back to CPU.

The default origin is `http://127.0.0.1:8090`; OpenAPI is at `/openapi.json`.

```sh
curl http://127.0.0.1:8090/v1/infer \
  -H 'content-type: application/json' \
  -d '{
    "model": "decide-340m",
    "backend": "pytorch",
    "precision": "fp32",
    "operation": "classify",
    "text": "The invoice needs urgent review.",
    "schema": {
      "kind": "classification",
      "tasks": {
        "priority": {"labels": ["urgent", "routine"]}
      }
    }
  }'
```

The process device is explicit. A request that asks for an unsupported
precision or operation fails; it is never rewritten.

## TypeScript client

[`@gliner-runner/client`](packages/client) is a Zod-validated transport client
for Node/Electron. It validates requests before transport and successful
responses before they cross an application boundary. Python remains the model
execution implementation, not a requirement for application integration.

```ts
import { GlinerClient } from "@gliner-runner/client";

const client = new GlinerClient();
const result = await client.infer({
  model: "decide-340m",
  backend: "pytorch",
  precision: "fp32",
  operation: "classify",
  text: "The invoice needs urgent review.",
  schema: {
    kind: "classification",
    tasks: {
      priority: {
        labels: ["urgent", "routine"],
        label_descriptions: {
          urgent: "Requires immediate handling",
          routine: "Can follow the normal review queue",
        },
      },
    },
  },
});
```

Electron main processes can own the runner as a supervised HTTP sidecar without
shell invocation:

```ts
import { GlinerSidecar } from "@gliner-runner/client/sidecar";

const sidecar = new GlinerSidecar({ port: 8090, device: "mps" });
await sidecar.start();
const models = await sidecar.client.models();
// On application shutdown:
await sidecar.stop();
```

The supervisor waits for `/healthz`, rejects early child exit and startup
timeouts with bounded diagnostics, and escalates shutdown only after a grace
period. Use it from Electron's main process, not a sandboxed renderer.

## Validation

```sh
mise run check        # lint, strict types, offline tests, client, docs
mise run integration  # requires GLINER_RUNNER_INTEGRATION_REQUEST
mise run test:mutation # slower mutation suite for critical Python paths
mise run benchmark    # scheduler benchmark; no model download
mise run benchmark:real  # requires GLINER_RUNNER_BENCHMARK_MODEL
```

See [architecture](docs/architecture.md), [protocol](docs/protocol.md),
[operations](docs/operations.md), [package registry policy](docs/registries.md),
[real-model benchmarks](docs/benchmarks.md), [contributing](CONTRIBUTING.md),
and [security](SECURITY.md).

Apache-2.0. External models retain their own licenses.
