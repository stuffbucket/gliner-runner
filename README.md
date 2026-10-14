# GLiNER Runner

GLiNER Runner sorts a piece of text into labels you choose: give it a message and a list of
labels such as `billing, support, shipping`, and it tells you which one fits, with a confidence
score. It runs on your own computer with no Python, and your text never leaves your machine.

## Install

With [Homebrew](https://brew.sh) (macOS on Apple Silicon, and Linux):

```sh
brew tap stuffbucket/tap
brew trust stuffbucket/tap
brew install gliner-runner
```

`brew trust` tells Homebrew 7 and later that it may load formulas from this tap; skip it if your
Homebrew says the command does not exist.

To update later:

```sh
brew update && brew upgrade gliner-runner
```

To test a release candidate instead, run `brew uninstall gliner-runner` and then
`brew install gliner-runner-rc`. The release candidate conflicts with the stable package, so
install only one of them.

Without Homebrew, this installs the same program into `~/.local`:

```sh
curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh
```

Check that it works:

```sh
gliner-runner --version
```

If an older Python `gliner-runner` (from `pip`) is also on your `PATH`, it can shadow this one.
`which gliner-runner` shows which program runs; the version output above starts with
`gliner-runner-onnx` for the right one.

## Download a model

The program does not include a model. Download one once (this asks you to confirm, then
downloads from Hugging Face and prepares it):

```sh
gliner-runner models install decide-340m
```

| Model | Download | Use it for |
| --- | --- | --- |
| `decide-340m` | about 1.8 GiB | English text; the default |
| `decide-multilingual` | about 1.1 GiB | Spanish, French, German and other languages |
| `decide-1b` | about 4.4 GiB | English text, larger model |

Models are stored in your user cache (`~/Library/Caches/gliner-runner/models` on macOS,
`~/.cache/gliner-runner/models` on Linux). Add `--yes` to skip the confirmation question.

## Try it

Run a few built-in examples:

```sh
gliner-runner demo
```

```text
gliner-runner demo (model: decide-340m)

1. Support ticket routing
   Text:    I was charged twice for my last order and I want one of the payments refunded.
   Labels:  billing, support, shipping
   Result:  billing (confidence 0.866)

2. Support ticket routing
   Text:    My package has been in transit for two weeks and the tracking page has not changed.
   Labels:  billing, support, shipping
   Result:  shipping (confidence 0.921)

3. Several topics in one message (multi-label)
   Text:    The battery is weak and the delivery was slow.
   Labels:  battery, screen, price, delivery  (several allowed)
   Result:  battery (0.948), delivery (0.919)

4. Labels with descriptions
   Text:    Can I swap the blue jacket I bought for the same one in a larger size?
   Labels:  refund, exchange, question
            refund: The customer wants their money back
            exchange: The customer wants a different item or size
            question: The customer asks for information only
   Result:  exchange (confidence 0.833)

5. Sentiment
   Text:    The checkout was quick and the staff were friendly and helpful.
   Labels:  positive, neutral, negative
   Result:  positive (confidence 0.944)

6. Priority
   Text:    Our payment system is down and customers cannot check out. We need help right now.
   Labels:  urgent, normal, low
   Result:  urgent (confidence 0.855)

Classified 6 texts in 1.5 s.
Try your own: gliner-runner classify --labels billing,support,shipping "your text"
```

Then classify your own text. Pick the labels yourself.

**Route a support ticket** (one label):

```sh
gliner-runner classify --labels billing,support,shipping "My package never arrived and the tracking page has not updated in two weeks."
```

```text
shipping (confidence 0.942)
```

**Allow several labels** with `--multi`:

```sh
gliner-runner classify --labels battery,screen,price,delivery --multi "The battery is weak and the delivery was slow."
```

```text
battery (0.948), delivery (0.919)
```

**Describe the labels** when their names are not enough, with `--describe label=text`:

```sh
gliner-runner classify --labels refund,exchange,question \
  --describe "refund=The customer wants their money back" \
  --describe "exchange=The customer wants a different item or size" \
  --describe "question=The customer only asks for information" \
  "Can I swap the blue jacket for a larger size?"
```

```text
exchange (confidence 0.826)
```

**Other languages** need the multilingual model (`gliner-runner models install decide-multilingual`):

```sh
gliner-runner classify --model decide-multilingual --labels billing,support,shipping "Me cobraron dos veces el mismo pedido."
```

```text
billing (confidence 0.779)
```

```sh
gliner-runner classify --model decide-multilingual --labels billing,support,shipping "Mon colis n'est toujours pas arrivé après deux semaines."
```

```text
shipping (confidence 0.514)
```

Text can also come from a pipe, and `--json` prints the full result for scripts:

```sh
echo "The checkout was quick and the staff were friendly." | gliner-runner classify --labels positive,neutral,negative
```

```text
positive (confidence 0.935)
```

```sh
gliner-runner classify --labels urgent,normal,low --json "Our payment system is down and customers cannot check out."
```

```json
{"category":{"value":"urgent","confidence":0.6183409935537204,"probabilities":{"urgent":0.6183409935537204,"normal":0.18731622346263635,"low":0.19434278298364327}},"_meta":{"feasible":true,"decoder":"independent","exact":true,"objective":0.13862642645835876,"violations":[]}}
```

Confidence scores depend on the model and the wording; they are not percentages of accuracy.
If a model is not installed, the command prints the exact `gliner-runner models install` command to
run. `gliner-runner --help` lists every command.

## Run it as a local service

Applications can call the runner over HTTP. Start it and leave it running:

```sh
gliner-runner serve
```

It listens on `http://127.0.0.1:8090`. In another terminal:

```sh
curl -s http://127.0.0.1:8090/v1/infer -H 'content-type: application/json' -d '{
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "operation": "classify",
  "text": "I was charged twice for my last order.",
  "schema": {
    "kind": "classification",
    "tasks": {"category": {"labels": ["billing", "support", "shipping"]}}
  }
}'
```

The answer is JSON; the result is under `output.category.value` (here `"billing"`). Use
`--port` to change the port; see [ONNX runtime](docs/onnx-runtime.md) for every option and the
[protocol](docs/protocol.md) for the request and response shapes.

## Check for updates

With Homebrew, `brew update && brew upgrade gliner-runner` installs the newest release. `gliner-runner update check`
tells you whether a newer stable release exists and which command installs it; it never installs
anything.

More: [project site](https://stuffbucket.github.io/gliner-runner/),
[ONNX runtime](docs/onnx-runtime.md), [development](docs/development.md),
[releases](docs/releases.md). Technical details follow.

## Overview

GLiNER Runner is a local, backend-neutral runtime and HTTP service for
explicit, reproducible GLiNER inference. It implements constrained
classification with the three published Fastino GLiNER2.5 Decide models, each
pinned to an exact revision. There are two runners behind one HTTP protocol:
the Python-free ONNX runner that Homebrew and the installer provide (the
`gliner-runner` command used above; the original name `gliner-runner-onnx` also
works), and a Python server that uses Fastino GLiNER2 2.1's public PyTorch
`Classifier.batch_classify` API as the correctness oracle.

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

PyTorch is the correctness oracle. The Python server registers only PyTorch.
MLX appears in the stable backend enumeration for plugin compatibility but is
not registered or advertised. ONNX is a production backend served by a separate,
Python-free Node runner (`gliner-runner`, also installed as `gliner-runner-onnx`) for constrained classification on
CPU/FP32 with the three curated revisions, backed by end-to-end parity evidence
from an Apple M-series host; see [ONNX runtime](docs/onnx-runtime.md). See
[backend policy](docs/backends.md).

## Install the Python server

You do not need this to use the Homebrew or installer build described above.

```sh
python -m pip install 'gliner-runner[pytorch]'
```

The PyTorch extra pins `gliner2[local]==2.1.0`; model artifacts are never part
of this package. The package also installs a `gliner-runner` command, which
collides with the Homebrew/installer command of the same name; the Python
examples below therefore start the Python server with
`python -m gliner_runner.cli`. To work on the repository, see
[development](docs/development.md) (`mise install`, `mise run setup`,
`mise run check`).

## Models and explicit downloads

`GET /v1/models` reports the curated official Fastino GLiNER2.5 Decide family,
host-supported CPU/GPU precision profiles, pinned and newly discovered
revisions, local availability, the currently loaded profile, estimated memory
floors, and measured score-drift guidance. `POST /v1/models/refresh` refreshes
only the cached provider metadata; it never installs or replaces a model. The
Python-free ONNX runner implements the same endpoints and flow; see
[ONNX runtime](docs/onnx-runtime.md#installing-models).

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
revision = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90"
license = "Apache-2.0"
backend = "pytorch"
precisions = ["fp32", "fp16", "bf16"]
operations = ["classify"]

[provenance]
upstream_model = "fastino/GLiNER2.5-Decide"
upstream_runtime = "gliner2==2.1.0"

[[artifacts]]
path = "config.json"
url = "https://huggingface.co/fastino/GLiNER2.5-Decide/resolve/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/config.json"
sha256 = "<64 lowercase hex characters>"
size = 1234
```

List every file required by the pinned local snapshot, then install explicitly:

```sh
python -m gliner_runner.cli models verify ~/.config/gliner-runner/models/decide-340m.toml
python -m gliner_runner.cli models install ~/.config/gliner-runner/models/decide-340m.toml
```

Inference verifies the installed files and never downloads a missing model.
Manifests are operator data because upstream artifact hashes and file sets can
change; this repository does not publish guessed checksums.

## Run

```sh
python -m gliner_runner.cli serve --device cpu
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
python -m gliner_runner.cli serve --device mps
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

## Supported operations and models

Two runners implement the same HTTP protocol. The Python server runs PyTorch;
the Python-free Node runner (`gliner-runner`, also installed as `gliner-runner-onnx`) runs ONNX Runtime. Each
advertises only what it implements in its own `/v1/capabilities`. See
[backend policy](docs/backends.md) and [ONNX runtime](docs/onnx-runtime.md).

### Operations

| Operation | PyTorch (Python server) | ONNX (Python-free runner) |
| --- | --- | --- |
| `classify` | Supported | Supported |
| `extract_entities`, `extract_relations`, `extract_structured` | Protocol enum values only; not advertised, `409 unsupported_configuration` | Protocol enum values only; not advertised, `409 unsupported_configuration` |

### `classify` feature coverage

"Exact" means identical labels, ordering, cardinality, empty results, decoder
metadata, structured errors, and usage between the ONNX runner and PyTorch
CPU/FP32; only floating-point confidence and probability values differ, by less
than the `1e-4` gate. The evidence is the
[production parity report](benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md)
(`english-v1` and `multilingual-v1`, 14 cases each).

| `classify` feature | PyTorch | ONNX | ONNX parity evidence (corpus cases) |
| --- | --- | --- | --- |
| Plain and described labels | Yes | Yes | `multi-label`, `ordered-label`, `described-single`, `mixed-length-long`; exact |
| Single-label and multi-label results | Yes | Yes | `described-single`, `softmax-activation`; `multi-label`, `default-temperature-multi-label`; exact |
| Multiple tasks in one schema | Yes | Yes | `multi-task-described-instruction`, `feasible-constraint`; exact |
| Task threshold and threshold-driven empty results | Yes | Yes | `high-threshold-empty`; exact |
| Minimum and maximum label constraints | Yes | Yes | `multi-label` (0 to 3), `softmax-activation` (1 to 1); `invalid-min-exceeds-max` 422; exact |
| Ordered results | Yes | Yes | `ordered-label`; exact |
| Activation and temperature | Yes | Yes | `softmax-activation`, `default-temperature-multi-label`; exact |
| Defaults and instructions | Yes | Yes | `multi-task-described-instruction`, `default-temperature-multi-label`; exact |
| Feasible constraints; infeasible constraints returned as `422` | Yes | Yes | `feasible-constraint`; `infeasible-constraint` 422; exact |
| Confidence, probabilities, and `_meta` decoder metadata | Yes | Yes | every successful case; labels and metadata exact, numbers within `1e-4` |
| Invalid-schema structured `422` errors | Yes | Yes | `invalid-reserved-marker`, `invalid-min-exceeds-max`, `invalid-unknown-constraint-label`; code, message, and retryable flag exact |
| Usage `inputTokens` exact, `outputTokens` `0` | Yes | Yes | every successful case; exact |
| Singleton, repeated, `/v1/batch`, and dynamic batching | Yes | Yes | singleton, repeated, batch, concurrent-group, and concurrent-all modes; exact |

### Models and execution profiles

All three are the curated, immutable Fastino revisions. Every cell lists a
profile with reviewed evidence on one Apple M5 Max (macOS arm64); see
[benchmarks](docs/benchmarks.md) for each cell's report. CUDA is advertised by
the Python server when available but was not evaluated.

| Model and exact revision | PyTorch CPU/FP32 | PyTorch MPS/FP16 | PyTorch MPS/FP32 | ONNX CPU/FP32 |
| --- | --- | --- | --- | --- |
| Decide 340M `fastino/GLiNER2.5-Decide@9d1bfb84...` (`decide-340m`) | Supported | Supported | Supported | Supported |
| Multilingual Decide `fastino/GLiNER2.5-multi-Decide@e173bca1...` (`decide-multilingual`) | Supported | Supported | Supported | Supported |
| Decide 1B `fastino/GLiNER2.5-Decide-1B@67af949e...` (`decide-1b`) | Supported | Supported | Supported | Supported |
| GLiNER2.5 Small | Not curated | Not curated | Not curated | Not curated (research export control only) |

The full 40-character revisions are in [backend policy](docs/backends.md).
ONNX support is limited to CPU/FP32 `classify` on these revisions; MPS, CUDA,
FP16, BF16, and quantization are not supported on the ONNX runner.

### Runtime differences

| | Python server (PyTorch) | Python-free runner (ONNX) |
| --- | --- | --- |
| Python required at run time | Yes (`gliner-runner[pytorch]`) | No (bundled Node.js 24 and ONNX Runtime) |
| Devices | CPU, plus MPS or CUDA when available | CPU only |
| Precisions | FP32; FP16 and BF16 where validated or supported | FP32 only |
| Quantization | None | None |
| Model acquisition | Approved `POST /v1/models/{id}/downloads` or operator manifests | No Python: approved `POST /v1/models/{id}/downloads` (or `gliner-runner models install <id> --yes`, or opt-in `--auto-download`) assembles the artifact in Node from the pinned Hugging Face files and a bundled per-revision template, byte-identical to the parity-verified artifact; a verified local snapshot is used first and no network is needed |
| Template generation | Not applicable | Build time only, with Python: `python -m gliner_runner.cli prepare-onnx --template-dir` for new models or revisions |
| Integrity | File sizes and SHA-256 verified before loading | Manifest, file set, sizes, and SHA-256 verified before loading |
| Fallback | None | None; `pytorch`/`mlx` requests and other precisions or devices return `409` |

### Raw logit deltas

The production artifact manifests record raw logit differences against PyTorch
at preparation time on three batches (19 rows, 69 valid logits per model; the
artifact is rejected above the `1e-4` absolute tolerance). "Wrapper" is the
project's PyTorch classification wrapper; "official" is Fastino's public
`Classifier.batch_score`. Values are maximum absolute differences, exactly as
recorded in the
[promoted manifests](benchmark-results/2026-10-14-onnx-runtime-parity-v1/):

| Model | ONNX vs wrapper | Wrapper vs official | ONNX vs official | Batch vs singleton (ONNX) | Selected labels matched | Valid logits compared | Nonzero deltas shown as synthetic zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decide-340m` | 5.007e-6 | 4.768e-7 | 5.245e-6 | 2.384e-6 | 19/19 | 69 | 67 of 69 |
| `decide-multilingual` | 9.298e-6 | 2.384e-7 | 9.179e-6 | 7.629e-6 | 19/19 | 69 | 69 of 69 |
| `decide-1b` | 1.931e-5 | 9.537e-7 | 1.931e-5 | 1.878e-6 | 19/19 | 69 | 69 of 69 |

End-to-end, the Node runner was replayed against PyTorch CPU/FP32 goldens in
every execution mode. Maximum raw numeric drift in confidences and
probabilities, with exact matching semantics in all 62 requests per cell:

| Model | `english-v1` | `multilingual-v1` |
| --- | ---: | ---: |
| `decide-340m` | 7.153e-6 | 4.292e-6 |
| `decide-multilingual` | 1.550e-5 | 2.189e-5 |
| `decide-1b` | 2.205e-6 | 3.979e-6 |

The acceptance tolerance is `1e-4` absolute. The synthetic-zero view (a nonzero
raw delta strictly below `1e-4` shown as zero) is an optional, non-gating
comparison aid: raw values, raw differences, label selection, and the
acceptance decision are always retained and authoritative. Cross-language cells
(English corpus on the multilingual model and the reverse) are parity probes,
not language-quality claims.

## Using the Python-free ONNX runner

1. **Install the runtime** with Homebrew or the one-line installer in [Install](#install),
   or download `gliner-runner-onnx-<version>-<platform>-<arch>.tar.gz` and its
   `.sha256` from the GitHub release, verify the digest, and extract it. The
   archive bundles Node.js, ONNX Runtime, and a small graph template per
   curated revision, but no Python and no model weights. Its `bin/` holds the
   same launcher as `gliner-runner` and `gliner-runner-onnx`.

2. **Install a model** (once per revision; no Python needed):

   ```sh
   RUNNER=./gliner-runner-onnx-<version>-<platform>-<arch>/bin/gliner-runner
   $RUNNER models install decide-340m --models-dir /path/to/models --yes
   ```

   This uses a verified local copy of the pinned snapshot when one is found
   (for example in the Hugging Face cache; no network), otherwise downloads
   the pinned files from Hugging Face (about 1.8 GiB for `decide-340m`),
   verifies their SHA-256, and assembles the artifact. `--yes` approves the
   download; on a terminal it prompts instead. Inside an application, a missing
   model returns `409 model_download_required`; after user approval, post it to
   `POST /v1/models/{id}/downloads` and poll the job, or call the TypeScript
   client's `ensureModel(modelId, { approve })`. `serve --auto-download` (off by
   default) lets inference install a missing model itself. Build-time Python
   (`python -m gliner_runner.cli prepare-onnx --template-dir`) is needed only to generate
   templates for new models or revisions.

3. **Run it**:

   ```sh
   $RUNNER serve --host 127.0.0.1 --port 8080 --models-dir /path/to/models
   ```

4. **Classify**:

   ```sh
   curl http://127.0.0.1:8080/v1/infer \
     -H 'content-type: application/json' \
     -d '{
       "model": "decide-340m",
       "backend": "onnx",
       "precision": "fp32",
       "operation": "classify",
       "text": "The invoice needs urgent review.",
       "schema": {
         "kind": "classification",
         "tasks": {"priority": {"labels": ["urgent", "routine"]}}
       }
     }'
   ```

   ```json
   {
     "request_id": "7636b0d0-7836-4a05-8924-80a6589079a4",
     "model": "decide-340m",
     "backend": "onnx",
     "precision": "fp32",
     "output": {
       "priority": {
         "value": "urgent",
         "confidence": 0.9987061594988828,
         "probabilities": {"urgent": 0.9987061594988828, "routine": 0.0012938405011171596}
       },
       "_meta": {"feasible": true, "decoder": "independent", "exact": true,
                 "objective": 3.248636484146118, "violations": []}
     },
     "error": null,
     "timing": {"queue_ms": 3.74, "inference_ms": 31.63},
     "usage": {"inputTokens": 17, "outputTokens": 0}
   }
   ```

   Timing values vary per run. A `pytorch` request, any precision other than
   `fp32`, or an operation other than `classify` returns `409`, not a fallback.

5. **Or supervise it from Electron or Node** with the TypeScript client, using
   the absolute launcher path:

   ```ts
   import { GlinerSidecar } from "@gliner-runner/client/sidecar";

   const sidecar = new GlinerSidecar({
     command: "/abs/path/to/gliner-runner-onnx-<version>-<platform>-<arch>/bin/gliner-runner",
     port: 8080,
     runnerArguments: ["--models-dir", "/abs/path/to/models"],
   });
   await sidecar.start();
   const result = await sidecar.client.infer({
     model: "decide-340m",
     backend: "onnx",
     precision: "fp32",
     operation: "classify",
     text: "The invoice needs urgent review.",
     schema: { kind: "classification", tasks: { priority: { labels: ["urgent", "routine"] } } },
   });
   await sidecar.stop();
   ```

Settings, model installation (interactive, command line, offline snapshot
discovery, processing records, and Hugging Face configuration), artifact layout,
integrity rules, limitations, and reproduction are in
[ONNX runtime](docs/onnx-runtime.md).

## Validation samples

- **Request corpora:** `tests/fixtures/onnx-parity/corpus-english-v1.json` and
  `corpus-multilingual-v1.json` (14 cases each, synthetic text), generated by
  `scripts/generate_onnx_parity_corpus.py` from the behavior cases in
  `benchmarks/benchmark_real.py` plus additional multi-task, default,
  temperature, repeated, and structured-error cases
  (`python scripts/generate_onnx_parity_corpus.py --check`).
- **Artifact-preparation parity fixtures:** `src/gliner_runner/onnx_prepare.py`
  defines the deterministic fixture batches (`_FIXTURE_BATCHES`, `_FIXTURE_TEXTS`,
  and the `_FULL_SCHEMA`, `_TWO_TASK_SCHEMA`, `_DESCRIBED_SCHEMA`, and
  `_SMALL_SCHEMA` schemas through `fixture_batches()`); their digest is recorded
  as `fixture_sha256` in each manifest.
- **GLiNER2 TypeScript port goldens:** `packages/onnx-runner/test/fixtures/gliner2/`
  (`decode.json`, `e2e-small-model.json`, `input-ids.json`, `schema-errors.json`,
  `tokenizer-ids.json`) produced from the Python reference and replayed by the
  runner's tests.
- **Promoted end-to-end evidence:**
  [`benchmark-results/2026-10-14-onnx-runtime-parity-v1/`](benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md)
  holds the PyTorch goldens, the 12 Node-versus-PyTorch reports, and the three
  artifact manifests (no weights).
- **Reproduce:** generate goldens with `mise run onnx:reference`, then run
  `GLINER_RUNNER_ONNX_MODELS_DIR=/path/to/models node --test packages/onnx-runner/dist/test/onnx/parity.test.js`;
  see [reproducing parity evidence](docs/onnx-runtime.md#reproducing-parity-evidence).

## TypeScript client

[`@gliner-runner/client`](packages/client) is a Zod-validated transport client
for Node/Electron. It validates requests before transport and successful
responses before they cross an application boundary. Python (or the
Python-free ONNX runner) is the model execution implementation, not a requirement
for application integration.

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
mise run check        # everything CI runs; offline, no models
mise run integration  # requires GLINER_RUNNER_INTEGRATION_REQUEST
mise run test:mutation # slower critical-path suite with a non-regression gate
mise run benchmark    # scheduler benchmark; no model download
mise run benchmark:real  # requires GLINER_RUNNER_BENCHMARK_MODEL
```

See [architecture](docs/architecture.md), [protocol](docs/protocol.md),
[operations](docs/operations.md), [package registry policy](docs/registries.md),
[releases and provider packaging](docs/releases.md),
[ONNX runtime](docs/onnx-runtime.md), [development](docs/development.md),
[real-model benchmarks](docs/benchmarks.md), [contributing](CONTRIBUTING.md),
and [security](SECURITY.md).

Apache-2.0. External models retain their own licenses.
