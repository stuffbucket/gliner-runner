# Releases and provider packaging

## Release flow

There are two channels. Both use the same build: a per-platform matrix
(`darwin-arm64` on `macos-15`, `linux-x64` on `ubuntu-24.04`, `linux-arm64` on
`ubuntu-24.04-arm`) that runs `mise run setup`, `check:onnx-receipt`, packages the
ONNX runtime archive, and smoke-verifies it with `verify-onnx-runtime.mjs`
(sanitized environment: `/healthz`, `/v1/capabilities`, `SIGTERM`, `--version`
against the checked-in receipt), followed by one assemble job on Ubuntu that
builds the wheel, sdist, client tarball and OpenAPI document, collects the
platform archives, and runs `scripts/assemble_release.py`. The shared build lives
in `.github/workflows/_release-build.yml`. No job downloads or runs models, so
inference is skipped in CI; pull requests run the same package and smoke step on
all three platforms (`package-smoke` in `ci.yml`).

```
stable   release-please -> approve (environment: release) -> build (3 platforms -> assemble)
                        -> publish (gh release upload) -> notify-tap -> brew-smoke
rc       dispatch -> prepare (choose rc number) -> build (3 platforms -> assemble)
                  -> publish (tag + prerelease) -> notify-tap -> brew-smoke
```

### Stable

Release Please owns the repository version, changelog, release pull request, tag
and GitHub release. Conventional commits merged to `main` update one release
pull request; merging it creates `vX.Y.Z` and a non-prerelease GitHub release,
which GitHub serves as `/releases/latest`. The `approve` job targets the GitHub
`release` environment (configure required reviewers): nothing from the tag is
built, and no write permission is granted, until a reviewer approves. Repository
settings must also permit Actions to create pull requests.

Assets attached to a stable release:

- `gliner_runner-X.Y.Z-py3-none-any.whl`, the sdist, and the
  `@gliner-runner/client` npm tarball;
- `openapi.json`;
- `gliner-runner-onnx-X.Y.Z-<platform>-<arch>.tar.gz` and its `.sha256` for
  `darwin-arm64`, `linux-x64`, and `linux-arm64`;
- `gliner-runner.rb`, the rendered Homebrew formula (see [Homebrew](#homebrew));
- `install.sh` from the repository root (skipped with a warning if absent);
- `release-manifest.json` and `SHA256SUMS` (covering every asset except
  `latest.json` and itself);
- `latest.json`, the upgrade feed.

### Release candidate

Run the `Release candidate` workflow (`workflow_dispatch`) with `version`
`X.Y.Z`. It picks the next `rc` number from existing `vX.Y.Z-rc.N` tags, builds
the dispatched commit with the package versions stamped to `X.Y.Z-rc.N` in the
build workspace only (`mise run release:stamp`; nothing is committed; the Python
and Docker markers use the PEP 440 spelling `X.Y.ZrcN`), then creates tag
`vX.Y.Z-rc.N` and a GitHub **prerelease** with the same assets as a stable
release except `latest.json`, and `gliner-runner-rc.rb` in place of
`gliner-runner.rb`. Prereleases never become `/releases/latest`, so an RC cannot
trigger an upgrade notification; Homebrew sees it only as the separate
`gliner-runner-rc` formula.
The manifest records `"channel": "rc"` and `"prerelease": true`. Tag, release
and assets are created only after every platform builds and passes smoke.

Release Please has not been verified against `-rc.N` tags; confirm that its next
release pull request still computes the right base version and changelog range
after the first RC.

The next stable release is `0.3.0` (`v0.2.1` is the current latest). Release
Please is pinned to it with `"release-as": "0.3.0"` on the root package in
`release-please-config.json`; **remove that line once `v0.3.0` is released**, or
every later release pull request is forced to 0.3.0 again. Dispatch the
`Release candidate` workflow with `0.3.0` to get `v0.3.0-rc.1`, `v0.3.0-rc.2`, ...

### Manifest and upgrade feed

`release-manifest.json` (`formatVersion` 2) records `version`, `channel`,
`prerelease`, every artifact with size and SHA-256, and `onnxRuntime.archives[]`:
one entry per platform archive with `sha256`, `executable`, `smokeTested`,
`parityEvidence`, and `status` (`parity-evidence` or `smoke-tested`).
`onnxRuntime.receipt` gives the receipt's path, SHA-256, run id, runner identity
and the platform the evidence was produced on (`darwin-arm64`). Only that
platform's archive is marked `parity-evidence`; the Linux archives are
smoke-tested without parity evidence. A platform archive is a packaging
artifact; it is *supported* only where parity evidence exists (see
[ONNX runtime](onnx-runtime.md)).

`latest.json` (stable only) carries `version`, `channel`, `releaseUrl`,
`notesUrl`, and the manifest, checksums, installer and per-platform archive URLs
with SHA-256. Clients check
`https://github.com/stuffbucket/gliner-runner/releases/latest/download/latest.json`
and compare `version` with the installed one; verify downloads against the
digests it lists.

### Homebrew

Two formulas, `gliner-runner` (stable) and `gliner-runner-rc` (release candidate),
are templates in `packaging/homebrew/`. Both install the per-platform ONNX archive
(`darwin-arm64`, `linux-x64`, `linux-arm64`) into `libexec` and link
`gliner-runner` and `gliner-runner-onnx` into `bin`; they `conflicts_with` each
other because both provide `gliner-runner`. The formula version comes from the
archive URL (`vX.Y.Z` or `vX.Y.Z-rc.N`); an explicit `version` is rejected by
`brew audit --strict` as redundant. The bundled Node binary and native `.node`
add-ons keep their code signatures when installed by Homebrew.

During assembly `scripts/render-homebrew-formula.mjs` fills the template from the
archive digests (`--channel stable|rc`, defaulting from the version: stable needs
`X.Y.Z`, rc needs `X.Y.Z-rc.N`), and `assemble_release.py --homebrew-formula`
attaches it as `gliner-runner.rb` (stable) or `gliner-runner-rc.rb` (rc), lists it
in `release-manifest.json` (role `homebrew-formula`) and `SHA256SUMS`, and checks
that it names the release version and every archive digest.

**Tap sync.** No secret is needed. `packaging/homebrew/tap/` holds files to copy
into `stuffbucket/homebrew-tap`: `.github/workflows/sync-gliner-runner.yml` (every
30 minutes, on `workflow_dispatch`, and on `repository_dispatch` type
`gliner-runner-release`) reads this repository's public releases with the tap's
`GITHUB_TOKEN`, downloads `gliner-runner.rb` from `/releases/latest` and
`gliner-runner-rc.rb` from the newest prerelease, verifies each against that
release's `SHA256SUMS`, writes `Formula/`, and commits only on change. Also:
`install/gliner-runner`, `apps/gliner-runner.md`, and `README-row.md` (README
snippets). The tap's `Fix Formulas` action (`homebrew-transform`) only rewrites
GoReleaser-generated files (it skips files without the GoReleaser marker) and only
nested `install`/`test`/`caveats` methods, so these formulas are unchanged by it.
Pushes made with `GITHUB_TOKEN` do not trigger the tap's other workflows.

The `notify-tap` job in `release.yml` and `release-candidate.yml` sends the
`repository_dispatch` when `HOMEBREW_TAP_TOKEN` is set, which only shortens the
wait for the schedule; it is skipped otherwise.

**Smoke test.** `.github/workflows/brew-smoke.yml` (`workflow_dispatch` with
`channel` `stable|rc` and optional `version`, and called by both release
workflows) taps `stuffbucket/tap` explicitly, polls until the tap serves the
expected version (up to 40 minutes), runs `brew install`, checks
`gliner-runner --version`, starts `gliner-runner serve --port 18090`, probes
`/healthz` and `/v1/capabilities`, and stops. It runs on `macos-15`,
`ubuntu-24.04` and `ubuntu-24.04-arm` and downloads no models. A failure marks
the release run failed after the release is already published; re-run it from the
Actions tab once the tap sync is fixed.

Users install with `brew tap stuffbucket/tap && brew install
stuffbucket/tap/gliner-runner` (or `-rc`). Recent Homebrew refuses to load
formulae from a third-party tap that was never tapped, so tap explicitly.
`brew update && brew outdated` reports new versions once the tap sync has copied
them.

### Secrets and settings

- `HOMEBREW_TAP_TOKEN` (optional): a token that can create `repository_dispatch`
  events on `stuffbucket/homebrew-tap` (contents write). Without it the tap
  syncs on its 30-minute schedule.
- `release` environment with required reviewers (stable approval gate).
- Actions may create pull requests (Release Please).
- In the tap: Actions enabled, workflow permissions allow `contents: write`, and
  the default branch accepts pushes from `github-actions[bot]`.

### Local equivalents

- `mise run release:local` checks the receipt, then builds and verifies the host
  platform's assets into `dist/release` at the current version (includes
  `latest.json` and `install.sh` when present). It needs no models; if
  `dist/onnx-models` exists the smoke test also runs inference.
- `mise run release:build` does the same for `GLINER_RUNNER_RELEASE_VERSION`;
  `release:packages` and `release:onnx-archive` are its two halves.
- For an RC dry run, stamp a throwaway checkout:
  `export GLINER_RUNNER_RELEASE_VERSION=X.Y.Z-rc.N; mise run release:stamp && mise run release:build`.

Assembly requires at least one `gliner-runner-onnx` archive, at most one per
platform, each carrying the release version and a `.sha256` that matches. Release
assembly fails if any release-managed version differs from the release version.
Release Please synchronizes `pyproject.toml`, the Python runtime version, the
three Node package versions and the Docker default.

Before approving a stable release, on the evidenced platform with installed
artifacts for the curated models (`gliner-runner-onnx models install <id> --yes`
or `gliner-runner prepare-onnx`):

```sh
mise run package:onnx-runtime
GLINER_RUNNER_ONNX_MODELS_DIR=/path/to/models mise run verify:onnx-runtime
GLINER_RUNNER_ONNX_MODELS_DIR=/path/to/models \
  node --test packages/onnx-runner/dist/test/onnx/parity.test.js
```

The archive verification extracts the archive outside the repository, checks
every file against the archive's own `release-manifest.json`, confirms no Python
or ML-framework files are bundled, and starts the launcher in a sanitized
environment. Model artifacts are never release assets (the archive carries only
the small per-revision templates under `app/templates/` that let
`gliner-runner-onnx models install` assemble them without Python).

### ONNX evaluation receipt

The ONNX runner ships with a checked-in evaluation receipt
(`release/onnx-eval-receipt.json`, ledger `release/onnx-eval-receipts.jsonl`, and
the generated `packages/onnx-runner/src/build-info.ts`). CI and the release job
never run models; `mise run check:onnx-receipt` only verifies that the receipt
covers the exact runner identity being released, that its SHA is the one embedded
in the binary and reported by `gliner-runner-onnx --version`, and that
the (identity, receipt) pair is not a replay. The platform build repeats this
through `verify-onnx-runtime.mjs`, which asserts the packaged binary's
`--version --json` receipt SHA, identity, and run id equal the checked-in receipt.
Each platform build job also runs `check:onnx-receipt` before packaging. Release
Please bumps `packages/onnx-runner/package.json` `version`, which is deliberately
excluded from the identity, so a version bump never invalidates the receipt. After
changing runner code or data, regenerate the receipt locally with
`mise run eval:onnx-receipt` and commit it; see
[ONNX runtime](onnx-runtime.md#evaluation-receipt).

GitHub Releases are the initial distribution authority. The workflow does not
publish to PyPI, npm, or GHCR and therefore needs no registry credentials.
Registry publication can be added later as a separate approved job without
changing the artifact contract.

## Provider integration

The runner and the provider have separate responsibilities:

- this repository owns the inference process, HTTP/OpenAPI protocol, model
  lifecycle, and release artifact integrity;
- a provider owns installation policy, selecting a compatible host profile,
  approval UI for model downloads, Electron IPC, and process supervision;
- model weights remain external and are never release assets.

The provider should download `release-manifest.json` and `SHA256SUMS`, verify
the selected artifacts, install the wheel into an isolated Python environment,
and pass the resulting absolute `gliner-runner` executable path to
`GlinerSidecar`. To use ONNX instead, extract the `onnx-runtime` archive after
verifying its digest, pass the absolute `bin/gliner-runner-onnx` path as
`command` with `runnerArguments: ["--models-dir", "<dir>"]` for a models
directory the provider owns, and install models either with
`gliner-runner-onnx models install <id> --yes` or through the approved download
flow (`client.ensureModel(id, { approve })`). No Python is involved. The npm tarball can be installed directly by a provider that
does not consume the client from a registry. The sidecar transport remains
loopback HTTP/JSON; Electron renderer processes should communicate with their
main process over application-owned IPC.

For a GitHub wheel installation, install the PyTorch extra so Fastino GLiNER2
and its platform dependencies are resolved:

```sh
python -m venv /path/to/provider/runtime
/path/to/provider/runtime/bin/python -m pip install \
  './gliner_runner-X.Y.Z-py3-none-any.whl[pytorch]'
```

On Windows, use the virtual environment's `Scripts\python.exe` and
`Scripts\gliner-runner.exe`. Providers that require offline installation
should construct and verify a platform-specific wheelhouse rather than
allowing pip to resolve dependencies during application startup.

## Why there is no frozen Python executable

The ONNX runtime ships as a platform archive because it contains Node.js and
ONNX Runtime and no Python. That does not change the Python server: a frozen
binary would include PyTorch, Fastino GLiNER2, tokenizers, and native
accelerator libraries. It would be large and must vary by operating system,
architecture, CUDA runtime, and potentially Apple/NVIDIA precision support.
Treating one such binary as universal would undermine the runner's explicit
capability and resource guarantees.

The wheel plus machine-readable release manifest is the stable first boundary
for the Python server. The ONNX runtime archive is the first platform bundle and
the model for later ones: platform bundles are additional artifact roles built on
native runners, smoke-tested on the matching device class, and model-free.
The existing `command` and `commandArguments` sidecar options mean adding
those bundles does not require a protocol or client redesign.
