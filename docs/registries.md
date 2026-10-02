# Package registry and credential policy

## Canonical npm policy

`~/.npmrc` is the canonical local registry, scope, proxy, certificate, and
authentication policy for npm-compatible tooling. pnpm reads it natively; the
repository does not translate it, set a competing registry, or commit a
project-level `.npmrc`.

Normal local setup therefore needs no registry flags:

```sh
mise run setup
```

Do not pass tokens on a command line or enable pnpm debug/HTTP logging in CI.
The lockfile records integrity hashes but must not contain URL userinfo,
credential query parameters, or auth fields.

## Docker and BuildKit

Docker may receive npm policy only through the BuildKit `npmrc` secret. The
secret exists for the `RUN` instruction that executes pnpm and is absent from
the resulting layer:

```sh
docker build \
  --secret id=npmrc,src="$HOME/.npmrc" \
  -t gliner-runner:local .
```

The npm secret is required because the image builds and tests the TypeScript
client. Never replace the secret mount with `COPY`, `ADD`, `ARG`, `ENV`, or a
token-bearing registry URL.

Python registry policy is separate because pip does not parse npmrc. To use a
private Python index, create a native pip configuration outside the repository
with owner-only permissions and pass it as a second secret:

```sh
chmod 600 "$HOME/.config/pip/pip.conf"
docker build \
  --secret id=npmrc,src="$HOME/.npmrc" \
  --secret id=pip_config,src="$HOME/.config/pip/pip.conf" \
  -t gliner-runner:local .
```

The `pip_config` secret is optional for public PyPI builds. The runtime stage
installs only from wheels produced by the builder and performs no registry
access.

## Native ecosystem adapters

Each ecosystem keeps its native configuration. Registry endpoints may be
mapped from the same organization policy, but credentials must come from the
tool's credential store or a secret manager rather than generated files in the
repository.

| Tool | Non-secret policy | Credential adapter | Container mapping |
| --- | --- | --- | --- |
| pip/uv | User `pip.conf`/`pip.ini`, or an organization-managed index setting | Keyring/plugin or owner-only native config | BuildKit secret `pip_config` mounted at `/etc/pip.conf` |
| Go | `GOPROXY`, `GOPRIVATE`, `GONOPROXY`, `GONOSUMDB` | Credential helper or owner-only `.netrc`; keep credentials out of `GOPROXY` | Mount `.netrc` as a BuildKit secret only for the download step |
| Cargo | User `~/.cargo/config.toml` for registry/index policy | Cargo credential provider or owner-only `credentials.toml` | Separate BuildKit secrets for config and credentials |
| Zig | Organization proxy/mirror selected by the build wrapper | Secret-manager-backed fetch wrapper; never put auth in `build.zig.zon` URLs | Mount only the wrapper's native credential input for fetch |

For local Python development, prefer an owner-only native config or keyring.
Environment variables such as `PIP_INDEX_URL` and Cargo registry tokens are
acceptable only when injected by a trusted secret manager; do not put them in
shell history, `mise.toml`, task output, or committed environment files.

Future Go, Rust, or Zig build stages must follow the same pattern: copy only
non-secret manifests first, mount credentials on the single dependency-fetch
instruction, and keep subsequent build/package stages offline.

## Enforcement

`mise run check:registry` rejects:

- credential/config files that must remain outside the repository;
- credential-bearing or sensitive-query URLs in lock/generated configuration;
- common token/password assignments in generated configuration;
- Docker `COPY`, `ADD`, `ARG`, or `ENV` patterns for registry credentials;
- removal of the approved npm and pip BuildKit adapters.

This check is defense in depth. It does not make verbose package-manager logs
safe and does not replace secret scanning in the hosting platform.
