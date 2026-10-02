# Security policy

## Supported versions

Security fixes are applied to the latest released minor version. Until the
first release, only the default branch is supported.

## Reporting a vulnerability

Please use GitHub's **Report a vulnerability** private-reporting form for this
repository. Do not open a public issue. Include affected versions, impact,
reproduction steps, and any suggested mitigation. Maintainers will acknowledge
receipt when the report is reviewed and will coordinate disclosure after a fix
is available. No response-time guarantee is made.

## Deployment boundaries

GLiNER Runner is local-first, not an authorization boundary. Loopback binding
is the default design. Exposing it to another host requires an authenticated,
rate-limiting reverse proxy, transport encryption, request-size limits, and
network policy. Treat input and model output as sensitive.

Model files are executable-adjacent supply-chain inputs. Only load artifacts
declared by an operator-controlled manifest with an exact source revision,
expected size, cryptographic digest, and license metadata. Verify before
publishing into the content-addressed store. Do not deserialize untrusted
pickle-based artifacts. Inference must not trigger implicit downloads.

Never include access tokens in model URLs, logs, exceptions, traces, or issue
reports. Keep caches non-world-writable and run containers as a non-root user.

Package registry credentials remain outside the repository. pnpm reads the
user's `~/.npmrc`; Docker receives it only as a BuildKit secret. Python and
future native toolchains use their own credential providers or owner-only
configuration mounted as secrets. See the
[registry policy](docs/registries.md).
