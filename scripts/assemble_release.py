from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble machine-readable GitHub release assets."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_versions() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as source:
        python_version = tomllib.load(source)["project"]["version"]
    root_package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    client_package = json.loads(
        (ROOT / "packages/client/package.json").read_text(encoding="utf-8")
    )
    init_match = re.search(
        r'^__version__ = "([^"]+)"',
        (ROOT / "src/gliner_runner/__init__.py").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    docker_match = re.search(
        r"^ARG GLINER_RUNNER_VERSION=([^\s]+)",
        (ROOT / "Dockerfile").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if init_match is None or docker_match is None:
        raise ValueError("release-managed version marker is missing")
    return {
        "python": str(python_version),
        "rootPackage": str(root_package["version"]),
        "clientPackage": str(client_package["version"]),
        "runtime": init_match.group(1),
        "docker": docker_match.group(1),
    }


def artifact_role(path: Path) -> str:
    if path.suffix == ".whl":
        return "python-wheel"
    if path.name.endswith(".tar.gz"):
        return "python-sdist"
    if path.suffix == ".tgz":
        return "typescript-client"
    if path.name == "openapi.json":
        return "openapi"
    raise ValueError(f"unexpected release artifact: {path.name}")


def media_type(path: Path) -> str:
    if path.suffix == ".whl":
        return "application/zip"
    if path.name.endswith((".tar.gz", ".tgz")):
        return "application/gzip"
    if path.suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def assemble(output_dir: Path, version: str) -> None:
    versions = repository_versions()
    mismatches = {name: value for name, value in versions.items() if value != version}
    if mismatches:
        details = ", ".join(f"{name}={value}" for name, value in sorted(mismatches.items()))
        raise ValueError(f"release version {version} is not synchronized: {details}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "openapi/openapi.json", output_dir / "openapi.json")
    package_assets = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name not in {"release-manifest.json", "SHA256SUMS"}
    )
    artifacts = [
        {
            "name": path.name,
            "role": artifact_role(path),
            "mediaType": media_type(path),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in package_assets
    ]
    roles = {artifact["role"] for artifact in artifacts}
    required_roles = {"python-wheel", "python-sdist", "typescript-client", "openapi"}
    if roles != required_roles or len(artifacts) != len(required_roles):
        raise ValueError(
            f"release assets must contain exactly one of each required role: {sorted(roles)}"
        )

    manifest = {
        "formatVersion": 1,
        "name": "gliner-runner",
        "version": version,
        "artifacts": artifacts,
        "modelsIncluded": False,
        "providerContract": {
            "transport": "http-json",
            "openapiArtifact": "openapi.json",
            "executable": "gliner-runner",
            "healthPath": "/healthz",
        },
    }
    manifest_path = output_dir / "release-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksummed = [*package_assets, manifest_path]
    checksums = "".join(
        f"{file_sha256(path)}  {path.name}\n" for path in sorted(checksummed)
    )
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def main() -> None:
    args = parse_args()
    assemble(args.output_dir.resolve(), args.version)


if __name__ == "__main__":
    main()
