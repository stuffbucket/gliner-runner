# gliner-runner

Local GLiNER2 inference runner: a loopback HTTP server for entity extraction and classification, shipped as a self-contained bundle (its own Node.js and ONNX Runtime, no Python).

<table style="width: 100%; border: none; background-color: #f6f8fa; margin: 20px 0;">
  <tr>
    <td style="padding: 12px 16px; border: none; text-align: center; font-weight: 500; color: #57606a;">
      Platforms: macOS arm64, Linux x64, Linux arm64
    </td>
  </tr>
</table>

## Prerequisites

**Homebrew required.** If you don't have it installed, see [HOMEBREW.md](../HOMEBREW.md).

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/stuffbucket/homebrew-tap/main/install/gliner-runner | bash && eval "$(brew shellenv)"
```

Or directly with brew:

```bash
brew tap stuffbucket/tap
brew install stuffbucket/tap/gliner-runner
```

## Quick Start

Models are not bundled. Install one (it is downloaded after you confirm), then start the server:

```bash
gliner-runner models install <model-id> --yes
gliner-runner serve --port 8090
curl http://127.0.0.1:8090/healthz
```

`gliner-runner --version` prints the installed version. `gliner-runner-onnx` is the same program under its original name.

## Release candidates

Release candidates are separate and conflict with the stable formula (both install `gliner-runner`):

```bash
brew uninstall gliner-runner   # if installed
brew install stuffbucket/tap/gliner-runner-rc
```

Go back with `brew uninstall gliner-runner-rc && brew install stuffbucket/tap/gliner-runner`.

## Updates

The formulas are copied from the project's GitHub releases about every 30 minutes. Pick up new versions with:

```bash
brew update && brew outdated
brew upgrade gliner-runner      # or gliner-runner-rc
```

## Links

- **Repository:** [stuffbucket/gliner-runner](https://github.com/stuffbucket/gliner-runner)
- **Issues:** [Report a bug](https://github.com/stuffbucket/gliner-runner/issues)
- **License:** Apache-2.0
---

<p align="center">
  <a href="https://github.com/stuffbucket/gliner-runner">Repository</a> •
  <a href="https://github.com/stuffbucket/gliner-runner/issues">Report a Bug</a> •
  <a href="https://github.com/stuffbucket/gliner-runner/blob/main/LICENSE">License: Apache-2.0</a>
</p>
