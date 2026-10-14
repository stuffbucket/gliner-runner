---
title: Releases
description: Where to get GLiNER Runner releases and how stable and release-candidate builds differ.
---

All builds are published on the
[GitHub releases page](https://github.com/stuffbucket/gliner-runner/releases). Each release attaches
the Python wheel and sdist, the TypeScript client, the OpenAPI document, the ONNX runner archive,
`release-manifest.json`, and `SHA256SUMS`.

**Stable** releases (`X.Y.Z`) are the default: the installer picks the latest stable release, and
only stable releases trigger upgrade notifications. **Release candidates** (`X.Y.Z-rc.N`) are for
testing a build before it is promoted; they are never selected implicitly and do not trigger
upgrade notifications.

To pin a specific build, set `GLINER_RUNNER_VERSION` when running the installer:

```sh
curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh \
  | GLINER_RUNNER_VERSION=X.Y.Z-rc.N sh
```

How releases are produced and packaged is documented in [Release engineering](/gliner-runner/docs/releases/).
