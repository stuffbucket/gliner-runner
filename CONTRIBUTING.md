# Contributing

Thank you for improving GLiNER Runner.

## Development contract

1. Install the versions in `mise.toml` with `mise install`.
2. Run `mise run setup`.
3. Make a focused change.
4. Run `mise run check`.
5. Opt into `mise run integration` only when the required external artifacts
   and hardware are available. Run `mise run benchmark` separately.

mise installs the exact pnpm version declared by both `mise.toml` and the root
`packageManager` field. pnpm uses the contributor's canonical `~/.npmrc`
directly. Do not enable a second package-manager shim, add a repository
`.npmrc`, use a token-bearing URL, or generate ecosystem credentials. Follow
the [package registry policy](docs/registries.md) for Python and future native
toolchains.

Python changes must pass Ruff, strict mypy, and pytest through
`mise run check:python`. TypeScript changes use the `check:client` workspace
script. Keep these shared task names stable because CI and contributors use
them as the cross-surface contract.

Rust and Go are permitted where benchmarks or packaging requirements justify
another runtime. New control-plane implementations must consume the checked-in
OpenAPI contract, preserve scheduling and error semantics, and include
cross-platform distribution tests. Do not move preprocessing or decoding out
of Python merely to reduce the language count: oracle parity is the gate.

## Backend changes

Python remains the inference authority. A new backend or precision must:

- state its supported operations, languages, devices, and numeric formats;
- reject unsupported combinations rather than changing backend implicitly;
- pass deterministic shape, decoding, error, and representative parity tests
  against the official Fastino PyTorch reference;
- document tolerances and the dataset used to set them without committing
  restricted data;
- preserve request/response compatibility or introduce a versioned protocol;
- keep model acquisition pinned, checksum-verified, and outside the repository.

MLX starts with English span extraction in BF16 and INT8. INT4 is explicit
opt-in and needs its own quality gate. ONNX remains classification-only until
exports produced and controlled by this project are validated. Do not present
roadmap backends as available.

## Tests and benchmarks

Unit tests must be offline and deterministic. Integration tests must be
selected explicitly, identify required model digests, and skip with a useful
reason when prerequisites are absent. Benchmarks are opt-in, record hardware,
software versions, model digest, precision, workload, warm-up, and batch
settings, and must not be used as correctness evidence.

Never add model weights, credentials, private datasets, or local benchmark
results to commits. A reproducible public characterization may be committed
only when it contains no private paths, hostnames, usernames, inputs, or
credentials and documents its model digest and methodology. Keep pull requests
small, explain user-visible behavior, and update the relevant documentation
and OpenAPI contract.

Contributions are accepted under the Apache License 2.0 as described in
`LICENSE`.

## Releases

Release Please maintains the release pull request and synchronized versions.
Merging that pull request creates the tag and GitHub release; publication of
the built assets then requires approval through the GitHub `release`
environment. Do not edit the release manifest version or changelog manually
outside a release repair. See [releases and provider packaging](docs/releases.md)
for the artifact contract and local build command.
