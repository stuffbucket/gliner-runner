# Python-free ONNX runtime

`gliner-runner-onnx` is a production backend: a Node.js 24 TypeScript server
(package `@gliner-runner/onnx-runner`) that implements the full
[HTTP protocol](protocol.md) and runs classification through ONNX Runtime. It
needs no Python, PyTorch, Transformers, or GLiNER2 at run time. Tokenization,
schema and prompt compilation, label-position construction, constrained
decoding, result formatting, and usage accounting are implemented in
TypeScript and are checked against the PyTorch reference; only the encoder and
classification head execute inside ONNX Runtime.

Models are acquired without Python too. The runner bundles a small graph
template for each curated revision and assembles the production artifact in
Node.js from the pinned Hugging Face safetensors, producing files byte-identical
to the parity-verified artifacts built in CI. Python is needed only at build
time, to (re)generate templates when a model or revision is added
(`gliner-runner prepare-onnx --template-dir`, optionally checked with
`assemble-onnx`) and to produce reference outputs for parity checks
(`gliner-runner onnx-reference`). See [Installing models](#installing-models)
and [Build-time templates](#build-time-templates-python).

## Supported scope

Support is exactly this, and nothing else may be inferred from it:

| Dimension | Supported | Evidence |
| --- | --- | --- |
| Backend / provider | `onnx` on ONNX Runtime `CPUExecutionProvider` (`onnxruntime-node` 1.30.0) | [parity report](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md) |
| Precision | FP32, no quantization | same |
| Operation | `classify` | same |
| Models | `decide-340m` `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`, `decide-multilingual` `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`, `decide-1b` `67af949e17bebd20f05fe020a971079889e7a462` | same |
| Host evidence | Apple M5 Max, macOS arm64 (`darwin-arm64`) | same |

Not supported, and rejected rather than approximated: MPS, CUDA, FP16, BF16,
dynamic or static quantization, any other operation, any other model or
revision, and any host without its own evidence. The packaged archive is built
per platform; `darwin-arm64` is the only platform with parity evidence.

The Python server does **not** register an ONNX backend. ONNX is served only
by this Node runner. The shared `onnx` enum value is the same wire contract in
both servers; check `/v1/capabilities` of the process you are talking to.

## Installing models

The runner never needs Python to obtain a model. Everything below runs inside
the packaged Node.js runtime, offline unless a download is explicitly approved.

### How installation works

The package bundles one template per curated revision under
`templates/<owner>--<name>/<revision>/` (`app/templates/` in the runtime
archive): `template.json`, `graph.onnx` (2.4 to 4.8 MiB depending on the model),
`gliner2.json`, and the final `manifest.json`. `template.json` records, for the
pinned revision:

- the source files the preparation reads (`model.safetensors` and the tokenizer
  files) with their exact sizes and SHA-256 digests;
- the layout of `model.onnx.data`: for each large graph initializer, the
  safetensors tensor it comes from, whether it is used unchanged (`identity`) or
  as a 2-D `transpose`, and its 64-byte-aligned offset and length;
- the size and SHA-256 of every output file of the artifact.

Installing a model finds or downloads the pinned source files, then assembles
the artifact in Node.js:

1. verify the template files, and the source files by size and SHA-256;
2. check that every safetensors tensor the template reads is F32 with the
   recorded shape;
3. in a staging directory next to the destination, copy `graph.onnx` as
   `model.onnx`, stream `model.onnx.data` (identity tensors in 8 MiB blocks,
   transposes in bounded 32 MiB bands, so memory use does not scale with the
   model), and copy the tokenizer files and `gliner2.json`;
4. re-read every output and compare its size and SHA-256 with the template,
   then write `manifest.json` last and rename the staging directory into
   `<models-dir>/onnx/<owner>--<name>/<revision>/`.

The result is byte-identical to the artifact that `prepare-onnx` built and
parity-verified at build time, because every output digest is fixed by the
template; a mismatch fails the install and publishes nothing. An existing
artifact directory is never overwritten (a valid one is reused, an invalid one
is an error until you remove it), staging is removed on any failure, and the
host must be little-endian. Free disk space for the outputs plus a 64 MiB margin
is checked first; a download needs space for the source files as well, because
they are fetched into a temporary directory under
`<models-dir>/onnx/<owner>--<name>/` that is deleted afterwards.

Only the pinned revision of each curated model has a template. Any other
revision, and any other model, cannot be installed.

| Model | Download (source files) | Artifact on disk |
| --- | --- | --- |
| `decide-340m` | 1.82 GiB | 1.64 GiB |
| `decide-multilingual` | 1.09 GiB | 1.06 GiB |
| `decide-1b` | 4.43 GiB | 3.86 GiB |

### Interactive flow (HTTP)

Inference never downloads unless auto-download is enabled
([below](#auto-download)). With the default settings the flow is:

1. A request for a model that is not installed returns
   `409 model_download_required`. The error carries an `event`
   (`type`, `model`, exact `revision`, `destination`, `download_endpoint`); it is
   a notification, not approval.
2. The application obtains user or operator approval and posts it:

   ```sh
   curl http://127.0.0.1:8090/v1/models/decide-340m/downloads \
     -H 'content-type: application/json' \
     -d '{"approved":true,"destination":"<download_destination from GET /v1/models>"}'
   ```

   `approved` must be `true` (otherwise `422 invalid_download_request`), the
   model must be a curated one, and `revision`, if given, is a 40-character
   commit. The response is `202` with a job `{job_id, model_id, revision,
   destination, state, error}`. A second request for a model revision that
   already has an active job is rejected with `422`.
3. Poll `GET /v1/model-downloads/{job_id}` until `state` is `succeeded` or
   `failed` (`queued` and `running` are in progress). `destination` must equal
   the server's configured models directory; a different value does not fail the
   request but fails the job. On failure `error` carries the reason (no
   template for the revision, digest mismatch, not enough disk space, network
   failure after retries, and so on). When a job succeeds the artifact is
   already visible in `GET /v1/models` and usable for inference. Job state lives
   in memory for the life of the server process.

The job uses a verified local snapshot when one exists (see
[Offline snapshot discovery](#offline-snapshot-discovery)) and only otherwise
downloads, so approving a download can complete without touching the network.

The TypeScript client wraps this flow:

```ts
const model = await client.ensureModel("decide-340m", {
  approve: (item) => askTheUser(item), // asked once, only when the model is missing
  pollIntervalMs: 1000,                // optional, default 1000
  signal,                              // optional AbortSignal
});
```

`ensureModel` reads `GET /v1/models`, returns immediately when the pinned
revision is installed (`"decide-340m@<revision>"` checks that exact revision
instead), otherwise calls `approve` with the inventory entry, posts the approved
job to the entry's `download_destination`, polls it, and returns the refreshed
inventory entry. It throws `ModelDownloadDeniedError` when `approve` returns
false and `ModelDownloadFailedError` (carrying the job) when the job fails or
the model is still missing afterwards. Nothing is downloaded without approval.

### Command line

`gliner-runner-onnx models <command>` installs and inspects models without a
running server. It honors `--models-dir` (default `GLINER_RUNNER_MODELS_DIR`,
else the [user cache directory](#artifact-layout)), `--templates-dir`, and
`--revision` (default and only installable value: the pinned revision). Progress
goes to stderr. Exit codes: `0` success, `1` failure or download not approved,
`2` usage error.

| Command | Behavior |
| --- | --- |
| `install <model-id>... [--all] [--yes]` | Reuses a valid installed artifact; else prepares from a verified local snapshot (no network); else downloads the pinned source files and prepares. Prints one JSON line `{"model_id", "path", "revision"}` per model on stdout. Several ids, or `--all` (every curated model), install one after another; see below. |
| `prepare <model-id> [--snapshot <dir>]` | Offline only. Builds the artifact from `--snapshot`, or from the first verified snapshot in the standard locations; never downloads. Fails if the artifact directory already exists. |
| `scan [--model <id>] [--json]` | Lists every snapshot candidate found for the curated models, whether it verified or why it was rejected, and the processing status. Writes nothing into snapshot locations. |
| `list` | Prints JSON: per model the pinned revision, installed revisions, template availability and download size, and processing state. |

Approval applies only to downloading. `install` asks for it when the artifact
is missing and no verified local snapshot exists: with a terminal on stdin it
prints the model, size, source, and destination and prompts `[y/N]`; without a
terminal it refuses unless `--yes` (`-y`) is given.

With several ids or `--all`, `install` first prints a plan to stderr, one line
per model, then a total:

```text
decide-340m: already installed
decide-multilingual: found on this machine at /path/to/snapshot — no download needed
decide-1b: will download 4.8 GB (needs about 8.9 GB disk)
Total: will download 4.8 GB (needs about 8.9 GB disk)
```

Sizes come from the templates. It then refuses (exit `1`) if the free space in
the models directory is not enough; asks `Download 4.8 GB now? [y/N]` on a
terminal (only when something must be downloaded); and without a terminal
refuses unless `--yes` is given. Installed models are skipped
(`decide-340m: already installed (<path>)`).

```sh
gliner-runner-onnx models install decide-340m --yes          # non-interactive
gliner-runner-onnx models install --all --yes                # every curated model
gliner-runner-onnx models scan --model decide-340m           # what is on this machine?
gliner-runner-onnx models prepare decide-340m --snapshot /path/to/snapshot
gliner-runner-onnx models list
```

### Auto-download

`serve --auto-download` (or `GLINER_RUNNER_AUTO_DOWNLOAD=1`; accepted values
`1/0`, `true/false`, `yes/no`, `on/off`) lets an inference request for a missing
curated model start the install itself and wait for it. It is off by default,
so inference otherwise never downloads implicitly. Concurrent requests for the
same model revision, including the items of a batch, share one job. A client
that disconnects stops waiting; the job continues for the others. If the model
cannot be installed (unknown revision, no template) the request returns the
ordinary `409 model_download_required`; if the job fails, the same `409` is
returned with `automatic installation failed: <reason>`. Shutdown cancels
running jobs.

### CORS

The server answers a browser only for origins on its allow-list; pages from
other origins are blocked by the browser. A request whose `Origin` header
exactly equals an allowed origin (`scheme://host[:port]`, no `*`, no path) gets
`Access-Control-Allow-Origin` (the origin) and `Vary: Origin`, and an `OPTIONS`
preflight is answered `204` with `Access-Control-Allow-Methods: GET, POST,
OPTIONS`, `Access-Control-Allow-Headers: content-type` and
`Access-Control-Max-Age: 600`. Other origins get no CORS headers. `serve` logs
the allowed origins at startup.

The allow-list is saved in `$XDG_CONFIG_HOME/gliner-runner/settings.json`
(default `~/.config/gliner-runner/settings.json`, also on macOS;
`%APPDATA%\gliner-runner\settings.json` on Windows) under `cors_origins`. Until
that key exists the defaults apply: the usual local development servers
`http://localhost:3000`, `:4173` (Vite preview), `:4200`, `:4321` (Astro),
`:5173` (Vite) and `:8080`. Manage the list with `gliner-runner cors`; origins
are numbered from 1, like `ufw status numbered`:

```sh
gliner-runner cors list                               # numbered list and the file path (--json for JSON)
gliner-runner cors add https://stuffbucket.github.io  # allow the hosted playground
gliner-runner cors show 7                             # print origin number 7
gliner-runner cors set 7 https://app.example.com      # replace number 7
gliner-runner cors delete 7                           # remove by number (or by exact origin)
gliner-runner cors reset                              # back to the defaults
```

The first change writes the file with the defaults plus the change; other keys
in the file are kept. An empty list stays empty (no browser access). Changes
apply the next time `serve` starts.

`serve --cors-origin http://localhost:5173` (repeatable) or
`GLINER_RUNNER_CORS_ORIGINS=http://localhost:5173,https://app.example.com`
replaces the saved list for that run; the flag wins over the variable.

### Browser playground

`gliner-runner playground [--port 8090] [--model decide-340m] [--yes] [--no-open] [--site <url>]`
connects the [hosted playground](https://stuffbucket.github.io/gliner-runner/playground/)
to this computer in one step:

1. it adds the playground's origin to the saved CORS list (as `cors add` does);
2. it reuses a runner already serving on `--port` that allows that origin (a
   `GET /healthz` with an `Origin` header), or starts one in this process: on
   `--port`, or on the next free port when that port is taken by another
   program or by a runner started before the origin was allowed;
3. if the model is not installed, it asks (or takes `--yes`) and installs it
   through that runner's `POST /v1/models/{id}/downloads`, so a reused runner's
   own models directory is used;
4. it prints `<site>?mode=local&runner=http://127.0.0.1:<port>` on stdout and
   opens it with `open`, `xdg-open` or `start` (`--no-open` only prints it).

A runner started this way serves until Ctrl+C; when an existing one was reused
the command exits. Without a terminal and without `--yes` it refuses to download
and exits 1. `install.sh --playground [options]` runs it after installing, or
right away with a `gliner-runner` already on `PATH` that has the command.

The playground can also run the runner inside the page: the browser build
(`packages/onnx-runner-web`) hosts the same request handler, catalog, installer
and ONNX backend in a Web Worker, with the origin private file system for
storage and onnxruntime-web (WebGPU, else WASM) for inference. It offers
`decide-340m` only and downloads its pinned files from Hugging Face with
redirects followed by the browser (no token).

### Offline snapshot discovery

Before downloading, `install`, download jobs, and auto-download look for a local
copy of the pinned snapshot. Locations are searched in this precedence order:

1. `--snapshot <dir>` (`models prepare` only; when given, only this directory is
   considered);
2. each entry of `GLINER_RUNNER_SNAPSHOT_DIRS` (a path list, `:` separated on
   POSIX and `;` on Windows). An entry may be the snapshot directory itself, a
   provider-store root, or a Hugging Face cache root;
3. `HF_HUB_CACHE` (or `HUGGINGFACE_HUB_CACHE`), a Hugging Face hub cache;
4. `$HF_HOME/hub`;
5. `${XDG_CACHE_HOME:-~/.cache}/huggingface/hub`;
6. the Python runner's provider store,
   `<user cache>/gliner-runner/models/providers/huggingface`;
7. the models directory's own `providers/huggingface` and, last, the models
   directory itself, as provider stores.

Hugging Face caches are read as `models--<owner>--<name>/snapshots/<revision>/`
and provider stores as `<owner>--<name>/<revision>/`. A candidate qualifies
only when every source file named by the template exists and matches the pinned
size and SHA-256 (sizes of all files are checked before any hashing, so a wrong
snapshot is rejected cheaply); the first qualifying candidate is used. Symbolic
links, as in a Hugging Face cache's `blobs/`, are followed for reading, and
nothing is ever written to any snapshot location. Directories holding another
revision of the repository are reported as rejected rather than used.
`models scan` checks every candidate and reports each as verified or rejected
with the reason; `install` stops at the first verified one.

### Processing record

Every installation attempt leaves a record of what was done to a model
revision, as a sibling of the artifact directory (never inside it, so the
artifact's file set stays exactly what its manifest lists):

```text
<models-dir>/onnx/<owner>--<name>/<revision>.gliner-runner.json
```

| Field | Meaning |
| --- | --- |
| `format_version`, `kind` | `1` and `gliner-runner-onnx-processing-record` |
| `model_id`, `repository`, `revision`, `gliner_runner_version`, `updated_at` | Identity, runner version, ISO timestamp of the last update |
| `source.location` | The snapshot directory used, `huggingface:<endpoint>` for a download, or `null` when no source was found |
| `source.files[]` | Each pinned source file's `path`, `size`, `sha256` (from the template) and whether it `verified` |
| `state`, `processed` | `not_processed`, `verified_source`, `prepared`, or `failed`; `processed` is true only for `prepared` |
| `steps[]` | Ordered steps applied: `discovered_local` or `downloaded`, `verified_source_sha256`, `assembled_onnx_from_template:v1 (98 transpose, 296 identity of 394 tensors)`, `verified_outputs_sha256`, `published` |
| `template` | SHA-256 of the `template.json` used and its format version |
| `artifact` | Only when `prepared`: the artifact `path`, the `manifest_sha256`, and each output file's size and SHA-256 |
| `error` | Only when `failed`: the reason |
| `rejected_candidates[]` | Snapshot candidates that were looked at and refused, with `location` and `reason` (most recent attempt, at most 20) |

States: `not_processed` (a scan found nothing usable), `verified_source` (a
verified source exists, or assembly is in progress), `prepared` (artifact
published), `failed` (the attempt failed; `error` says why).

A record is written by `models install`, `models prepare`, download jobs, and
auto-download: `verified_source` before assembly, then `prepared` or `failed`;
a failed download or a missing or non-matching snapshot is recorded as `failed`
with no steps. `models scan` records `not_processed` or `verified_source`, but
leaves a `prepared` record alone while the artifact is installed. Nothing is
written when the artifact already exists (an existing valid artifact is reused
and `prepare` refuses before recording), when the model or revision has no
template, or by `models list`, `GET /v1/models`, inference, or server startup.
Writing is best effort: the file is written atomically (temporary file, then
rename), and a failure to write it is logged and never fails the installation.
A missing, unreadable, or foreign record counts as absent. The record is
informational; whether a model is runnable is decided only by the artifact's
manifest and digests.

Example after a download (the artifact path is a placeholder; digests are
abbreviated and `artifact.files[]` is shortened to two of its six entries):

```json
{
  "format_version": 1,
  "kind": "gliner-runner-onnx-processing-record",
  "model_id": "decide-340m",
  "repository": "fastino/GLiNER2.5-Decide",
  "revision": "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
  "gliner_runner_version": "0.2.1",
  "updated_at": "2026-10-14T18:03:21.482Z",
  "source": {
    "location": "huggingface:https://huggingface.co",
    "files": [
      { "path": "model.safetensors", "size": 1945828140, "sha256": "40a5a23f...6dc997", "verified": true },
      { "path": "tokenizer.json", "size": 8333952, "sha256": "3ad87d9f...67bcc8", "verified": true },
      { "path": "tokenizer_config.json", "size": 3356, "sha256": "323199a4...d21309", "verified": true },
      { "path": "special_tokens_map.json", "size": 2414, "sha256": "84ea7014...e83f4f", "verified": true }
    ]
  },
  "processed": true,
  "state": "prepared",
  "steps": [
    "downloaded",
    "verified_source_sha256",
    "assembled_onnx_from_template:v1 (98 transpose, 296 identity of 394 tensors)",
    "verified_outputs_sha256",
    "published"
  ],
  "template": { "sha256": "fd1620a0...b8a6c2", "format_version": 1 },
  "artifact": {
    "path": "<models-dir>/onnx/fastino--GLiNER2.5-Decide/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
    "manifest_sha256": "2912a88c...033d86",
    "files": [
      { "path": "model.onnx", "size": 4990967, "sha256": "782fa725...a26e89" },
      { "path": "model.onnx.data", "size": 1744089152, "sha256": "2a753374...388262" }
    ]
  }
}
```

### Hugging Face access, environment, and flags

| Setting | Applies to | Meaning |
| --- | --- | --- |
| `--models-dir`, `GLINER_RUNNER_MODELS_DIR` | `serve`, `models` | Models root. Default: user cache directory plus `gliner-runner/models` |
| `--templates-dir`, `GLINER_RUNNER_ONNX_TEMPLATES_DIR` | `serve`, `models` | Template directory; the flag wins, then the variable, then the bundled templates (`app/templates/` in the archive) |
| `--auto-download`, `GLINER_RUNNER_AUTO_DOWNLOAD` | `serve` | Enable [auto-download](#auto-download); default off |
| `--cors-origin`, `GLINER_RUNNER_CORS_ORIGINS` | `serve` | Browser origins allowed to call the server, replacing the list saved with `gliner-runner cors`; see [CORS](#cors) |
| `XDG_CONFIG_HOME` (`APPDATA` on Windows) | `serve`, `cors` | Directory holding `gliner-runner/settings.json`; default `~/.config` |
| `HF_ENDPOINT` | downloads, refresh | Hub base URL, default `https://huggingface.co`. Files are fetched from `{endpoint}/{repository}/resolve/{revision}/{file}` |
| `HF_TOKEN` | downloads, refresh | Optional bearer token for gated or private repositories. Redirects are followed manually; the token is sent only to the endpoint's own origin and dropped when a redirect (for example to a CDN) leaves it. It is never logged or returned by the API |
| `GLINER_RUNNER_SNAPSHOT_DIRS` | install, prepare, scan, jobs | Extra snapshot search roots (see [discovery](#offline-snapshot-discovery)) |
| `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME` | same | Where Hugging Face caches are looked for |

Each downloaded file is streamed to a `.partial` file while its size and
SHA-256 are checked against the template; an oversize body, a mismatch, or a
digest failure is fatal and nothing is published. Transient failures (HTTP 408,
425, 429, 5xx, a transfer that delivers no bytes for 60 seconds) are retried up
to five attempts per file with exponential backoff from 0.5 to 8 seconds. A
`401` or `403` suggests setting `HF_TOKEN`.

`POST /v1/models/refresh` is the only other network operation. It looks up the
current revision of each curated repository and lists the publisher's
`GLiNER2.5` Decide repositories on the Hub (`HF_ENDPOINT`, `HF_TOKEN`), and
records the result in `<models-dir>/catalog-cache.json` (same format as the
Python server: per model `discovered_revision` and `refreshed_at`, plus
`_discovered_models`). It only updates `discovered_revision` and
`discovered_models` in `GET /v1/models`; it never installs a model or changes a
pin, and a Hub failure returns `502 catalog_refresh_failed`.

## Build-time templates (Python)

Templates are generated once per model revision at build time, never at run time
and never on server start. This is the only place Python is involved. It needs
the Python build environment and the pinned snapshot:

```sh
python -m pip install -e '.[onnx-export]'

gliner-runner prepare-onnx --model decide-340m \
  --snapshot /absolute/path/to/GLiNER2.5-Decide/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90 \
  --output-dir /absolute/path/to/models \
  --template-dir packages/onnx-runner/templates \
  --atol 0.0001
```

`mise run prepare:onnx` wraps the same command
(`GLINER_RUNNER_ONNX_PREPARE_MODEL`, `GLINER_RUNNER_ONNX_PREPARE_SNAPSHOT`,
`GLINER_RUNNER_ONNX_PREPARE_OUTPUT`, default `dist/onnx-models`;
`GLINER_RUNNER_ONNX_TEMPLATE_DIR`, default `packages/onnx-runner/templates`;
`GLINER_RUNNER_ONNX_ATOL` overrides the gate, at most `1e-3`). The snapshot must
verify against the pinned repository and revision before conversion. The command
fails, and publishes nothing, if the exported graph does not reproduce the
PyTorch logits within the tolerance, if batch invariance fails, or if the
GLiNER2, Torch, Transformers, ONNX, ONNX Runtime, or ONNX Script versions differ
from the qualified set (GLiNER2 2.1.0, Torch 2.14.0, Transformers 5.19.0, ONNX
1.23.2, ONNX Runtime 1.31.0, ONNX Script 0.7.2). The 1B model needs
substantially more memory and disk than the others.

After export, `prepare-onnx` canonicalizes the graph: every initializer whose
bytes equal a `model.safetensors` tensor (unchanged, or a 2-D transpose) becomes
external data in `model.onnx.data`, in graph order, 64-byte aligned, with zero
padding; unmatched initializers of 64 KiB or more fail the build. All parity
gates run against these canonical bytes. `--template-dir <dir>` also writes
`<dir>/<owner>--<name>/<revision>/{template.json,graph.onnx,gliner2.json,manifest.json}`,
the package the Node runner bundles. Adding a model or revision therefore means
producing its template here (and a curated pin in the runner), and committing it
under `packages/onnx-runner/templates/`; the packaging script fails if no
template is present. `prepare-onnx` always checks that the pure-Python reference
assembler reproduces the artifact byte for byte; operators with Python can run
that assembler directly, without conversion:

```sh
gliner-runner assemble-onnx --template <template-dir>/<owner>--<name>/<revision> \
  --snapshot <snapshot> --output-dir <models-dir>
```

The format is specified in `dist/port-spec/onnx-template-contract.md` and
implemented by `gliner_runner.onnx_template` (Python) and
`packages/onnx-runner/src/onnx/{templates,assemble}.ts` (Node).

## Artifact layout

```text
<models-dir>/onnx/<owner>--<name>/<revision>/
  manifest.json            identity, provenance, graph contract, parity, file hashes
  model.onnx               graph (opset 20, dynamic batch/sequence/label axes)
  model.onnx.data          canonical external-data sidecar (every safetensors-derived weight)
  tokenizer.json           copied verbatim from the snapshot
  tokenizer_config.json    copied verbatim from the snapshot
  special_tokens_map.json  copied verbatim when the snapshot ships one
  gliner2.json             preprocessing facts a Python-free port needs
```

For example `<models-dir>/onnx/fastino--GLiNER2.5-Decide/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/`.
The default models directory is the user cache directory plus
`gliner-runner/models` (`~/Library/Caches/gliner-runner/models` on macOS,
`$XDG_CACHE_HOME` or `~/.cache` on Linux, `%LOCALAPPDATA%` on Windows).
Next to each artifact directory the runner keeps an informational
`<revision>.gliner-runner.json` [processing record](#processing-record), and the
models directory holds `catalog-cache.json` once a refresh has run; neither is
part of the artifact.

`manifest.json` records, and the runtime enforces:

- `format_version`, `kind` (`gliner-runner-onnx-classifier`), `model_id`,
  `repository`, and the immutable 40-character `revision`;
- `source_snapshot_sha256` and `source_snapshot_bytes` of the verified source
  snapshot, plus parameter count and license;
- `operations` (`classify`), `precision` (`fp32`), `quantization` (`none`), and
  `provider` (`CPUExecutionProvider`);
- the graph path, external-data sidecars, input and output names, types, and
  dynamic axes, and opset imports;
- exporter and dependency versions;
- `parity`: the absolute tolerance, maximum raw logit differences, selected-label
  matches, batch-invariance difference, and the fixture suite digest used at
  build time;
- `files`: the size and SHA-256 of every file in the directory except the
  manifest itself.

The manifests of the three evidenced artifacts are published, without weights,
in the [parity evidence](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/).

## Running the packaged runtime

`mise run package:onnx-runtime` builds a self-contained, model-free archive for
the host platform:

```text
dist/release/onnx-runtime/gliner-runner-onnx-<version>-<platform>-<arch>.tar.gz
dist/release/onnx-runtime/gliner-runner-onnx-<version>-<platform>-<arch>.tar.gz.sha256
```

The archive contains `bin/gliner-runner-onnx` (a launcher that runs the bundled
Node.js), `runtime/node/`, `app/` (compiled server plus production
`node_modules`, with only this platform's ONNX Runtime native library, and
`app/templates/`, the per-revision graph templates that `models install`
assembles artifacts from), `licenses/`, `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md`, and a `release-manifest.json` listing every file with
its size and SHA-256. It
contains no Python and no model files. It is the `onnx-runtime` asset of each
[GitHub release](releases.md).

`mise run verify:onnx-runtime` extracts the archive outside the repository,
checks every file against its manifest, confirms no Python or ML-framework
files are present, starts the launcher with a sanitized environment (minimal
`PATH`, a temporary `HOME`, no repository `node_modules`), checks that
`--version --json` reports the checked-in evaluation receipt SHA, runner identity
and run id (see [Evaluation receipt](#evaluation-receipt)), checks `/healthz` and
`/v1/capabilities`, performs one `decide-340m` inference when artifacts are
available (`GLINER_RUNNER_ONNX_MODELS_DIR`, default `dist/onnx-models`), and
requires a clean exit on `SIGTERM`.

Install a model and run it:

```sh
tar -xzf gliner-runner-onnx-<version>-<platform>-<arch>.tar.gz
RUNNER=./gliner-runner-onnx-<version>-<platform>-<arch>/bin/gliner-runner-onnx
$RUNNER models install decide-340m --models-dir /absolute/path/to/models --yes
$RUNNER serve --models-dir /absolute/path/to/models
```

`models install` needs no Python and no pre-prepared artifacts; see
[Installing models](#installing-models) for the offline, interactive, and
auto-download variants.

The server listens on `127.0.0.1:8090` by default and logs its listening
address to stderr. `SIGTERM` or `SIGINT` stops admission, drains queued work,
and exits.

### Settings

Each setting is a `serve` option or a `GLINER_RUNNER_*` environment variable;
the command-line flag wins, then the environment, then the default.

| Option | Variable | Default |
| --- | --- | --- |
| `--host` | `GLINER_RUNNER_HOST` | `127.0.0.1` |
| `--port` | `GLINER_RUNNER_PORT` | `8090` |
| `--device` | `GLINER_RUNNER_DEVICE` | `cpu` (the only supported value) |
| `--models-dir` | `GLINER_RUNNER_MODELS_DIR` | user cache `gliner-runner/models` |
| `--queue-capacity` | `GLINER_RUNNER_QUEUE_CAPACITY` | `256` pending requests per model (the intake queue and the backlog of requests waiting for a compatible batch are bounded together) |
| `--max-batch-size` | `GLINER_RUNNER_MAX_BATCH_SIZE` | `16` |
| `--batch-window-ms` | `GLINER_RUNNER_BATCH_WINDOW_MS` | `4` |
| `--model-idle-ttl-seconds` | `GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS` | `0` (keep warm) |
| `--memory-limit-bytes` | `GLINER_RUNNER_MEMORY_LIMIT_BYTES` | 75% of physical memory (minimum 512 MiB) |
| `--shutdown-timeout-ms` | `GLINER_RUNNER_SHUTDOWN_TIMEOUT_MS` | `4500` |
| `--max-body-bytes` | `GLINER_RUNNER_MAX_BODY_BYTES` | `16777216` (16 MiB); larger request bodies get `413 request_too_large` |
| `--auto-download` | `GLINER_RUNNER_AUTO_DOWNLOAD` | off; `1`, `true`, `yes`, or `on` lets inference install a missing curated model (see [auto-download](#auto-download)) |
| `--templates-dir` | `GLINER_RUNNER_ONNX_TEMPLATES_DIR` | the bundled templates |
| `--cors-origin` (repeatable) | `GLINER_RUNNER_CORS_ORIGINS` (comma-separated) | the list in `settings.json`, else common localhost development origins (see [CORS](#cors)) |
| n/a | `GLINER_RUNNER_ONNX_INTRA_OP_THREADS` | ONNX Runtime's physical-core default |

Scheduling, schema and model grouping, bounded queue with `429` backpressure,
cancellation, one resident model, idle eviction, and the memory ceiling behave
as described in [operations](operations.md); they use the same contract as the
Python server.

## Sidecar use from the TypeScript client

`GlinerSidecar` launches any executable that accepts
`serve --host --port --device`. Point `command` at the absolute launcher path
and pass the models directory through `runnerArguments`:

```ts
import { GlinerSidecar } from "@gliner-runner/client/sidecar";

const sidecar = new GlinerSidecar({
  command: "/absolute/path/to/gliner-runner-onnx-<version>-<platform>-<arch>/bin/gliner-runner-onnx",
  port: 8090,
  device: "cpu",
  runnerArguments: ["--models-dir", "/absolute/path/to/models"],
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

Requests must set `"backend": "onnx"` and `"precision": "fp32"`; the client
validates and the server enforces them.

To install a missing model from the application, call
`sidecar.client.ensureModel("decide-340m", { approve })` before the first
request (see [Interactive flow](#interactive-flow-http)); it asks `approve`
once, then starts and polls the download job. Alternatively pass
`"--auto-download"` in `runnerArguments` so inference installs a missing model
itself, with no approval prompt.

## Integrity, no fallback, and limitations

- **Integrity.** Loading a model re-reads `manifest.json`, rejects a manifest whose
  identity, revision, provider, precision, quantization, operations, graph
  contract, or opset differs from what the runner expects, rejects missing or
  unlisted files, and checks every file's size and SHA-256 before the graph is
  created. Artifact files are inspected with `lstat` and opened without following
  symbolic links, so a symlinked directory, manifest, or listed file is refused
  rather than followed. Verification is cached only while each file's path,
  size, and modification time are unchanged. A bad artifact returns a
  structured integrity error and is never loaded.
- **No fallback.** A request for `pytorch` or `mlx`, any precision other than
  `fp32`, any device other than CPU, or any operation other than `classify`
  returns `409 unsupported_configuration`. The runner never converts, retries on
  another backend, or changes precision. A model that is not installed returns
  `409 model_download_required`.
- **Downloads are explicit.** Inference never downloads unless the operator
  started the server with `--auto-download`. Network requests happen only for an
  approved download (`POST /v1/models/{id}/downloads`, `models install --yes`, or
  auto-download), which fetches the template's pinned files and verifies their
  size and SHA-256, and for `POST /v1/models/refresh`, which looks up revisions
  on the Hub and writes `catalog-cache.json`. `GET /v1/models` reports the three
  curated models, their installed revisions, `download_required`, and the
  `download_destination` (the models directory).
- **Only the pinned revisions are runnable.** A manifest whose revision is not
  the curated pinned revision for its model is rejected as invalid.
- **Bounded resources.** Request bodies above `--max-body-bytes` are rejected
  with `413 request_too_large`; a full pending queue (including requests set
  aside for a compatible batch) returns `429` with `Retry-After`; and the memory
  ceiling is enforced as described in [operations](operations.md).
- **CPU/FP32 only.** There is no MPS, CUDA, FP16, BF16, or quantization path.
  Performance on CPU is not claimed to match accelerated PyTorch profiles.
- **Numerics.** Outputs are exact for labels, ordering, cardinality, empty
  results, metadata, errors, and usage. Raw confidences differ from PyTorch by
  at most `2.2e-5` in the evidence; they are not bitwise equal.

## Parity evidence

The [evidence report](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md)
replays two 14-case corpora through the real server for each curated model in
singleton, repeated, batch, concurrent-group, and concurrent-all modes: 372/372
requests match the PyTorch goldens exactly, with raw numeric drift at most
`2.189e-5` and no value over the `1e-4` gate. A diagnostic "synthetic zero" view
counts nonzero deltas below `1e-4` as zero; it never replaces the raw values or
the acceptance decision. The report lists the artifact digests, runtime versions,
and host.

### Reproducing parity evidence

Install artifacts as above (`models install`), then generate PyTorch goldens (build environment
with `.[onnx-export]`) for each model and corpus. The goldens use the checked-in
corpora in `tests/fixtures/onnx-parity/`, which
`python scripts/generate_onnx_parity_corpus.py --check` keeps current:

```sh
export GLINER_RUNNER_ONNX_PREPARE_MODEL=decide-340m
export GLINER_RUNNER_ONNX_PREPARE_SNAPSHOT=/absolute/path/to/snapshot
export GLINER_RUNNER_ONNX_REFERENCE_CORPUS=tests/fixtures/onnx-parity/corpus-english-v1.json
export GLINER_RUNNER_ONNX_REFERENCE_OUTPUT=dist/onnx-parity/decide-340m-english-v1.json
mise run onnx:reference
```

The golden file name must be `<model_id>-<corpus>.json`. The checked-in goldens live in
`tests/fixtures/onnx-parity/goldens/` (copy regenerated goldens there). Then run
the guarded test, which starts the real server, replays every golden in all
modes, and writes `node-vs-pytorch-<model_id>-<corpus>.{json,md}` reports to a
separate output directory, never into the goldens:

```sh
pnpm --filter @gliner-runner/onnx-runner run build
GLINER_RUNNER_ONNX_MODELS_DIR=/absolute/path/to/models \
  node --test packages/onnx-runner/dist/test/onnx/parity.test.js
```

Optional inputs: `GLINER_RUNNER_ONNX_PARITY_DIR` (golden directory, default
`tests/fixtures/onnx-parity/goldens`), `GLINER_RUNNER_ONNX_PARITY_OUT_DIR` (report
directory, default `dist/onnx-parity-run`, git-ignored), `GLINER_RUNNER_ONNX_CORPUS_DIR`, and
`GLINER_RUNNER_ONNX_PARITY_MODELS` (comma-separated model ids; the default is
every installed model). Without `GLINER_RUNNER_ONNX_MODELS_DIR` the test is
skipped, so `mise run check` stays offline and model-free. To publish new
evidence, review the reports for local paths and scrub them as the existing
evidence directory did.

## Evaluation receipt

CI never runs models. The machine that prepared a change runs the real
evaluation and checks in a *receipt*; CI only asserts that the receipt is
coherent with the code and data it claims to cover, and that it is not a replay.

Checked-in files:

| File | Role |
| --- | --- |
| `release/onnx-eval-receipt.json` | The canonical receipt: runner version and identity, model pins with `template.json`/`manifest.json` digests, corpus and golden digests, per model x corpus results (`requests`, `exact_semantic_matches`, `max_raw_abs_diff`, `synthetic_zero_diffs`, `compared_values`), tolerance `1e-4`, host, `run_id`, `created_at`, `previous_receipt_sha256`. Receipt SHA = sha256 of this file's bytes. |
| `release/onnx-eval-receipts.jsonl` | Append-only ledger, one line per receipt: `receipt_sha256`, `identity_sha256`, `run_id`, `created_at`, `previous_receipt_sha256`. |
| `packages/onnx-runner/src/build-info.ts` | Generated; embeds the identity SHA, receipt SHA and run id so the binary reports them. |
| `tests/fixtures/onnx-parity/{corpus-*.json,goldens/}` | Request corpora and the PyTorch golden outputs the evaluation compares against. |

The runner identity digest (`node scripts/onnx-eval-receipt.mjs identity --list`)
is a sha256 over sorted `(path, sha256)` entries for `packages/onnx-runner/src/**`
(minus `build-info.ts`), `package.json` (without its release-managed `version`),
`tsconfig.json`, `templates/**`, the production dependency closure of
`pnpm-lock.yaml` (`name@version@integrity`), `tests/fixtures/onnx-parity/**`, and the
curated catalog pins. Line endings are normalized for text files and files git
ignores are skipped, so the digest is the same locally and in CI.
`gliner-runner-onnx --version` (also `-v`, `--v`; `--json` for machine output)
prints the version plus these SHAs and works without models, including in the
packaged archive.

Developer workflow:

1. Change runner code, templates, corpora, goldens, dependencies, or catalog pins.
2. With prepared artifacts for all three curated models, run
   `GLINER_RUNNER_ONNX_MODELS_DIR=/path/to/models mise run eval:onnx-receipt`. It
   builds, runs the guarded parity test for every model x both corpora, writes
   nothing unless everything passes, then writes the receipt and `build-info.ts`
   and appends one ledger line.
3. Commit the receipt, the ledger, and `build-info.ts` with the change. Do not edit
   them by hand and never rewrite earlier ledger lines.

`mise run check:onnx-receipt` (part of `mise run check` and run again before
release assets are built) checks, without models or network: the files exist and
are schema-valid with every model x corpus passed within `1e-4`; the recomputed
identity equals the receipt's and `build-info.ts`'s (otherwise the receipt is
stale and the message says to rerun `eval:onnx-receipt`); sha256 of the receipt
equals the SHA embedded in the runner and reported by the built CLI; the
template, manifest, corpus, and golden digests match the tree; and anti-replay:
receipt SHA and `run_id` are unique across the ledger, the last ledger line is the
current receipt, the ledger chain (`previous_receipt_sha256`) is unbroken, and
when `GLINER_RUNNER_RECEIPT_BASE_REF` names a base revision (CI sets it to the PR
base or the previous push tip) the base ledger is an exact prefix of the head
ledger, an unchanged receipt must keep the same identity, and a new receipt must
have a new SHA and `run_id` and chain to the base receipt (several local runs on
one branch each chain to the previous one).

What it deliberately does not check: that the evaluation really happened on the
claimed host or with the claimed models (the receipt is an honest-developer
attestation, not a signature, and a determined author could forge one); anything
about files outside the identity inputs (tests, docs, CI, Python code); and
model behavior on other platforms. Reviewers can re-run the evaluation to
reproduce the receipt.

## Research exporter

`benchmarks/export_classification_onnx.py` and its `benchmark:onnx-export*`
tasks remain research tooling for raw-logit feasibility; see
[benchmarks](benchmarks.md#research-only-onnx-feasibility-export). The
production path is `prepare-onnx` plus the Node runner described here.
