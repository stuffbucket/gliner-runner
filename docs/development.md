# Development

Host tools: [mise](https://mise.jdx.dev/) (pins Node 24, pnpm, Python). Everything below runs from the repository root.

## Tasks

| Task | What it does |
| --- | --- |
| `mise run setup` | Install dependencies (`pnpm install --frozen-lockfile`, plus the Python dev extras). |
| `mise run check` | Everything CI runs. Offline; no models, no network. |
| `mise run dev` | Build the ONNX runner and serve it on `127.0.0.1:8080` with `dist/onnx-models`. Override with `GLINER_RUNNER_DEV_PORT` and `GLINER_RUNNER_ONNX_MODELS_DIR`. |
| `mise run models` | Install the three curated models into `dist/onnx-models` with `gliner-runner-onnx models install <id> --yes`. **Downloads about 7 GB** unless a verified local snapshot (for example the Hugging Face cache) is found first. |
| `mise run receipt` | Alias of `eval:onnx-receipt`. Runs real parity against the models in `dist/onnx-models` and writes the receipt, ledger line and `build-info.ts`. |
| `mise run release:local` | Build and verify this host's release assets into `dist/release`. See [releases](releases.md). |
| `mise run lint:shell` | `shellcheck` on `install.sh` (not part of `check`; needs `shellcheck`). |

## (a) Normal development

```sh
mise install
mise run setup
mise run check        # before every push
mise run dev          # optional: run the runner locally
```

`mise run models` is needed only to exercise real inference through `dev`; `check` never needs models.
Change `install.sh` and run `mise run lint:shell`; its tests run inside `check`.

## (b) Producing a receipt CI will accept

Needed after any change to runner source, templates, corpora, goldens, dependencies or catalog pins
(CI fails with "receipt is stale" otherwise). See [evaluation receipt](onnx-runtime.md#evaluation-receipt).

```sh
mise run models                    # once; fills dist/onnx-models (about 7 GB)
mise run receipt                   # builds, runs parity for 3 models x 2 corpora, writes the receipt
mise run check                     # confirms the receipt matches the tree
git add release/ packages/onnx-runner/src/build-info.ts
git commit
```

Commit `release/onnx-eval-receipt.json`, `release/onnx-eval-receipts.jsonl` and
`packages/onnx-runner/src/build-info.ts` together with the change. Do not edit them by hand or rewrite earlier
ledger lines. Re-run `mise run receipt` after any further change to the identity inputs; each run appends a ledger line
that chains to the previous one. To reproduce CI's base comparison locally:

```sh
GLINER_RUNNER_RECEIPT_BASE_REF=origin/main mise run check:onnx-receipt
```

## (c) Releases

Both kinds are described in [releases](releases.md); the commands to run are:

```sh
# Check the host build first (no models, no network)
mise run check
mise run release:local

# Release candidate: X.Y.Z-rc.N, a GitHub prerelease, never "latest"
GLINER_RUNNER_RELEASE_VERSION=X.Y.Z-rc.N mise run release:stamp   # working tree only; never commit it
GLINER_RUNNER_RELEASE_VERSION=X.Y.Z-rc.N mise run release:build

# Stable release: merge the Release Please pull request, then approve the `release` environment job
```

Users install an RC with `GLINER_RUNNER_VERSION=X.Y.Z-rc.N` (see the README); `gliner-runner-onnx update check`
only ever offers stable releases.
