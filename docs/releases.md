# Releases and provider packaging

## Release flow

Release Please owns the repository version, changelog, release pull request,
tag, and GitHub release. Conventional commits merged to `main` update a single
release pull request. Merging that pull request creates a `vX.Y.Z` tag and
GitHub release.

Release asset publication is a separate job in the same workflow and targets
the GitHub `release` environment. Configure that environment with required
reviewers. The job does not start, receive write permission, or execute the
tagged source until an authorized reviewer approves it. Repository settings
must also permit GitHub Actions to create pull requests.

The release job builds from the immutable release tag and attaches:

- the pure-Python `gliner_runner-X.Y.Z-py3-none-any.whl`;
- the Python source distribution;
- the `@gliner-runner/client` npm tarball;
- the checked-in OpenAPI document;
- `release-manifest.json`, containing artifact roles, sizes, SHA-256 digests,
  and the provider launch contract;
- `SHA256SUMS`, covering every asset except itself.

`mise run release:build` produces the same files locally when
`GLINER_RUNNER_RELEASE_VERSION` matches every release-managed version.
Release Please synchronizes `pyproject.toml`, the Python runtime version, both
Node package versions, and the Docker build default.

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
`GlinerSidecar`. The npm tarball can be installed directly by a provider that
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

## Why there is no frozen executable yet

A frozen binary would include PyTorch, Fastino GLiNER2, tokenizers, and native
accelerator libraries. It would be large and must vary by operating system,
architecture, CUDA runtime, and potentially Apple/NVIDIA precision support.
Treating one such binary as universal would undermine the runner's explicit
capability and resource guarantees.

The wheel plus machine-readable release manifest is the stable first boundary.
After the provider's installer and target matrix are known, platform bundles
can be added as additional artifact roles. They must be built on native
runners, smoke-tested on the matching device class, and remain model-free.
The existing `command` and `commandArguments` sidecar options mean adding
those bundles does not require a protocol or client redesign.
