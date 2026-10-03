from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.assemble_release import assemble, file_sha256, repository_versions


def test_repository_release_versions_are_synchronized() -> None:
    assert len(set(repository_versions().values())) == 1


def test_release_manifest_and_checksums(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    assets = {
        f"gliner_runner-{version}-py3-none-any.whl": b"wheel",
        f"gliner_runner-{version}.tar.gz": b"sdist",
        f"gliner-runner-client-{version}.tgz": b"client",
    }
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)

    assemble(tmp_path, version)

    manifest_path = tmp_path / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formatVersion"] == 1
    assert manifest["version"] == version
    assert manifest["modelsIncluded"] is False
    assert {artifact["role"] for artifact in manifest["artifacts"]} == {
        "python-wheel",
        "python-sdist",
        "typescript-client",
        "openapi",
    }
    assert manifest["providerContract"] == {
        "transport": "http-json",
        "openapiArtifact": "openapi.json",
        "executable": "gliner-runner",
        "healthPath": "/healthz",
    }
    checksums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"{file_sha256(manifest_path)}  release-manifest.json\n" in checksums
    assert "SHA256SUMS" not in checksums


def test_release_assembly_rejects_version_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not synchronized"):
        assemble(tmp_path, "0.0.0-invalid")
