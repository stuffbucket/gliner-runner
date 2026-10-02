from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE_ENGINE = ">=24 <25"
NODE_IMAGE_SUFFIX = "-bookworm-slim"


def main() -> None:
    errors: list[str] = []
    mise = tomllib.loads((ROOT / "mise.toml").read_text(encoding="utf-8"))
    tools = mise["tools"]
    node_version = tools["node"]
    pnpm_version = tools["npm:pnpm"]
    node_major = node_version.partition(".")[0]

    nvm_major = (ROOT / ".nvmrc").read_text(encoding="utf-8").strip()
    if nvm_major != node_major:
        errors.append(f".nvmrc {nvm_major!r} does not match mise Node {node_version!r}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    expected_image = f"FROM node:{node_version}{NODE_IMAGE_SUFFIX} AS client-builder"
    if expected_image not in dockerfile:
        errors.append(f"Dockerfile must use {expected_image!r}")

    root_package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    client_package = json.loads(
        (ROOT / "packages/client/package.json").read_text(encoding="utf-8")
    )
    packages = (
        ("package.json", root_package),
        ("packages/client/package.json", client_package),
    )
    for name, package in packages:
        if package.get("engines", {}).get("node") != NODE_ENGINE:
            errors.append(f"{name} must declare Node engine {NODE_ENGINE!r}")

    expected_package_manager = f"pnpm@{pnpm_version}"
    if root_package.get("packageManager") != expected_package_manager:
        errors.append(
            "package.json packageManager must match mise's npm:pnpm version "
            f"{expected_package_manager!r}"
        )
    if re.search(r"(?i)\bcorepack\b", (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")):
        errors.append("CONTRIBUTING.md must not instruct contributors to use Corepack")

    if errors:
        print("\n".join(f"tool version policy: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("tool version policy: ok")


if __name__ == "__main__":
    main()
