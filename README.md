# GLiNER Runner

GLiNER Runner is a local, backend-neutral Python runtime and HTTP service for
explicit, reproducible GLiNER inference. The initial release implements
constrained classification through Fastino GLiNER2 2.0's public PyTorch
`Classifier.batch_classify` API. It supports the three published Decide model
repositories when supplied through operator-owned, revision-pinned manifests.

## Guarantees

- no silent backend, precision, device, or model fallback;
- one loaded model owner per `(backend, model digest, precision, device)`;
- bounded queues, cancellation-aware admission, compatible-schema dynamic
  micro-batching, and explicit HTTP 429 backpressure;
- strict Pydantic request/result/schema contracts and reproducible OpenAPI;
- external models with exact revisions, sizes, SHA-256 checksums, licenses,
  and content-addressed storage;
- offline unit/parity-fixture tests; real-model integration is opt-in.

PyTorch is the correctness oracle. MLX and ONNX appear in the stable backend
enumeration for plugin compatibility, but are not registered or advertised.
See [backend policy](docs/backends.md).

## Install and develop

The supported development entry point uses
[mise](https://mise.jdx.dev/) and [pnpm](https://pnpm.io/):

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

## Install a model

Create a TOML manifest in the configured manifest directory (by default
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
    tasks: { priority: { labels: ["urgent", "routine"] } },
  },
});
```

## Validation

```sh
mise run check        # lint, strict types, offline tests, client, docs
mise run integration  # requires GLINER_RUNNER_INTEGRATION_REQUEST
mise run benchmark    # scheduler benchmark; no model download
mise run benchmark:real  # requires GLINER_RUNNER_BENCHMARK_MODEL
```

See [architecture](docs/architecture.md), [protocol](docs/protocol.md),
[operations](docs/operations.md), [package registry policy](docs/registries.md),
[real-model benchmarks](docs/benchmarks.md), [contributing](CONTRIBUTING.md),
and [security](SECURITY.md).

Apache-2.0. External models retain their own licenses.
