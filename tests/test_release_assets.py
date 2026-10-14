from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from scripts import assemble_release
from scripts.assemble_release import (
    assemble,
    file_sha256,
    pep440_version,
    repository_versions,
    stamp_versions,
)


@contextmanager
def stamped_repository() -> Iterator[None]:
    """Run a test that stamps versions, then restore every release-managed file."""
    files = [
        assemble_release.ROOT / path
        for path in (
            "pyproject.toml",
            "package.json",
            "packages/client/package.json",
            "packages/onnx-runner/package.json",
            "src/gliner_runner/__init__.py",
            "Dockerfile",
        )
    ]
    originals = {path: path.read_bytes() for path in files}
    try:
        yield
    finally:
        for path, content in originals.items():
            path.write_bytes(content)


def test_repository_release_versions_are_synchronized() -> None:
    assert len(set(repository_versions().values())) == 1


def write_onnx_runtime(directory: Path, version: str, platform: str = "darwin-arm64") -> str:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"gliner-runner-onnx-{version}-{platform}.tar.gz"
    archive = directory / name
    archive.write_bytes(b"onnx runtime archive")
    (directory / f"{name}.sha256").write_text(f"{file_sha256(archive)}  {name}\n", encoding="utf-8")
    return name


def write_package_assets(directory: Path, version: str) -> None:
    assets = {
        f"gliner_runner-{version}-py3-none-any.whl": b"wheel",
        f"gliner_runner-{version}.tar.gz": b"sdist",
        f"gliner-runner-client-{version}.tgz": b"client",
    }
    for name, content in assets.items():
        (directory / name).write_bytes(content)


def test_release_manifest_and_checksums(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, version)
    onnx_name = write_onnx_runtime(tmp_path / "onnx", version)

    assemble(release, version, [tmp_path / "onnx"])

    manifest_path = release / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formatVersion"] == 2
    assert manifest["version"] == version
    assert manifest["modelsIncluded"] is False
    assert len(manifest["artifacts"]) == 6
    assert {artifact["role"] for artifact in manifest["artifacts"]} == {
        "python-wheel",
        "python-sdist",
        "typescript-client",
        "openapi",
        "onnx-runtime",
        "onnx-runtime-checksum",
    }
    onnx_artifact = next(
        artifact for artifact in manifest["artifacts"] if artifact["role"] == "onnx-runtime"
    )
    assert onnx_artifact["name"] == onnx_name
    assert onnx_artifact["platform"] == "darwin"
    assert onnx_artifact["arch"] == "arm64"
    assert onnx_artifact["sha256"] == file_sha256(release / onnx_name)
    assert manifest["channel"] == "stable"
    assert manifest["prerelease"] is False
    runtime = manifest["onnxRuntime"]
    assert runtime["archives"] == [
        {
            "platform": "darwin",
            "arch": "arm64",
            "archive": onnx_name,
            "checksum": f"{onnx_name}.sha256",
            "sha256": file_sha256(release / onnx_name),
            "size": (release / onnx_name).stat().st_size,
            "executable": "bin/gliner-runner-onnx",
            "smokeTested": True,
            "parityEvidence": True,
            "status": "parity-evidence",
        }
    ]
    assert {key: runtime[key] for key in ("backend", "device", "precision", "operations")} == {
        "backend": "onnx",
        "device": "cpu",
        "precision": "fp32",
        "operations": ["classify"],
    }
    assert runtime["pythonRequired"] is False and runtime["modelsIncluded"] is False
    receipt = runtime["receipt"]
    receipt_file = assemble_release.ROOT / "release/onnx-eval-receipt.json"
    assert receipt["sha256"] == file_sha256(receipt_file)
    assert (receipt["evidencePlatform"], receipt["evidenceArch"]) == ("darwin", "arm64")
    assert manifest["providerContract"] == {
        "transport": "http-json",
        "openapiArtifact": "openapi.json",
        "executable": "gliner-runner",
        "healthPath": "/healthz",
    }
    checksums = (release / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"{file_sha256(manifest_path)}  release-manifest.json\n" in checksums
    assert f"{file_sha256(release / onnx_name)}  {onnx_name}\n" in checksums
    assert f"{file_sha256(release / f'{onnx_name}.sha256')}  {onnx_name}.sha256\n" in checksums
    assert "SHA256SUMS" not in checksums


def test_release_assembly_rejects_version_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not synchronized"):
        assemble(tmp_path, "0.0.0", [])


def test_release_assembly_requires_onnx_runtime(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    write_package_assets(tmp_path, version)

    with pytest.raises(ValueError, match="at least one gliner-runner-onnx archive"):
        assemble(tmp_path, version)


def test_release_assembly_rejects_onnx_runtime_for_another_version(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    write_package_assets(tmp_path, version)
    write_onnx_runtime(tmp_path, "0.0.0-invalid")

    with pytest.raises(ValueError, match="does not carry release version"):
        assemble(tmp_path, version)


def test_release_assembly_rejects_stale_onnx_runtime_checksum(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    write_package_assets(tmp_path, version)
    name = write_onnx_runtime(tmp_path, version)
    (tmp_path / f"{name}.sha256").write_text(f"{'0' * 64}  {name}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        assemble(tmp_path, version)


def test_release_assembly_accepts_multiple_platform_archives(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, version)
    names = [
        write_onnx_runtime(tmp_path / "darwin", version, "darwin-arm64"),
        write_onnx_runtime(tmp_path / "linux", version, "linux-x64"),
        write_onnx_runtime(tmp_path / "linux", version, "linux-arm64"),
    ]

    assemble(release, version, [tmp_path / "darwin", tmp_path / "linux"])

    manifest = json.loads((release / "release-manifest.json").read_text(encoding="utf-8"))
    archives = manifest["onnxRuntime"]["archives"]
    assert [entry["archive"] for entry in archives] == [names[0], names[2], names[1]]
    assert [entry["status"] for entry in archives] == [
        "parity-evidence",
        "smoke-tested",
        "smoke-tested",
    ]
    assert [entry["parityEvidence"] for entry in archives] == [True, False, False]
    sums = (release / "SHA256SUMS").read_text(encoding="utf-8")
    for name in names:
        assert f"{file_sha256(release / name)}  {name}\n" in sums
    latest = json.loads((release / "latest.json").read_text(encoding="utf-8"))
    assert sorted(latest["platforms"]) == ["darwin-arm64", "linux-arm64", "linux-x64"]
    assert "latest.json" not in sums


def test_stable_release_writes_latest_json_and_installer(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, version)
    name = write_onnx_runtime(tmp_path / "onnx", version)
    installer = tmp_path / "install.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")

    assemble(release, version, [tmp_path / "onnx"], installer=installer)

    latest = json.loads((release / "latest.json").read_text(encoding="utf-8"))
    base = f"https://github.com/stuffbucket/gliner-runner/releases/download/v{version}"
    assert latest["version"] == version
    assert latest["channel"] == "stable"
    assert latest["releaseUrl"].endswith(f"/releases/tag/v{version}")
    assert latest["platforms"]["darwin-arm64"] == {
        "name": name,
        "url": f"{base}/{name}",
        "sha256": file_sha256(release / name),
        "size": (release / name).stat().st_size,
        "parityEvidence": True,
    }
    assert latest["manifest"]["sha256"] == file_sha256(release / "release-manifest.json")
    assert latest["installer"]["sha256"] == file_sha256(installer)
    sums = (release / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"{file_sha256(installer)}  install.sh\n" in sums


def test_release_candidate_has_no_latest_json(tmp_path: Path) -> None:
    rc = "9.8.7-rc.2"
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, "9.8.7rc2")
    write_onnx_runtime(tmp_path / "onnx", rc, "linux-x64")
    with stamped_repository():
        stamp_versions(rc)
        assert set(repository_versions().values()) == {rc, "9.8.7rc2"}
        assemble(release, rc, [tmp_path / "onnx"])

    manifest = json.loads((release / "release-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == rc
    assert manifest["channel"] == "rc"
    assert manifest["prerelease"] is True
    assert manifest["onnxRuntime"]["archives"][0]["status"] == "smoke-tested"
    assert not (release / "latest.json").exists()
    assert "latest.json" not in (release / "SHA256SUMS").read_text(encoding="utf-8")


def write_formula(directory: Path, name: str, version: str, archive: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    formula = directory / name
    formula.write_text(
        f'url ".../download/v{version}/x.tar.gz"\nsha256 "{file_sha256(archive)}"\n',
        encoding="utf-8",
    )
    return formula


def test_stable_release_attaches_checksummed_homebrew_formula(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, version)
    onnx_name = write_onnx_runtime(tmp_path / "onnx", version)
    formula = write_formula(
        tmp_path / "formula", "gliner-runner.rb", version, tmp_path / "onnx" / onnx_name
    )

    assemble(release, version, [tmp_path / "onnx"], homebrew_formula=formula)

    assert (release / "gliner-runner.rb").read_bytes() == formula.read_bytes()
    manifest = json.loads((release / "release-manifest.json").read_text(encoding="utf-8"))
    roles = {artifact["name"]: artifact["role"] for artifact in manifest["artifacts"]}
    assert roles["gliner-runner.rb"] == "homebrew-formula"
    sums = (release / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"{file_sha256(formula)}  gliner-runner.rb\n" in sums


def test_release_candidate_attaches_rc_formula_only(tmp_path: Path) -> None:
    rc = "9.8.7-rc.2"
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, "9.8.7rc2")
    onnx_name = write_onnx_runtime(tmp_path / "onnx", rc, "linux-x64")
    archive = tmp_path / "onnx" / onnx_name
    stable_formula = write_formula(tmp_path / "formula", "gliner-runner.rb", rc, archive)
    rc_formula = write_formula(tmp_path / "formula", "gliner-runner-rc.rb", rc, archive)
    with stamped_repository():
        stamp_versions(rc)
        with pytest.raises(ValueError, match=r"attaches gliner-runner-rc\.rb"):
            assemble(release, rc, [tmp_path / "onnx"], homebrew_formula=stable_formula)
        assemble(release, rc, [tmp_path / "onnx"], homebrew_formula=rc_formula)

    assert (release / "gliner-runner-rc.rb").is_file()
    assert not (release / "gliner-runner.rb").exists()
    assert f"{file_sha256(rc_formula)}  gliner-runner-rc.rb\n" in (
        release / "SHA256SUMS"
    ).read_text(encoding="utf-8")


def test_homebrew_formula_must_match_version_and_archives(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, version)
    onnx_name = write_onnx_runtime(tmp_path / "onnx", version)
    archive = tmp_path / "onnx" / onnx_name
    stale = write_formula(tmp_path / "stale", "gliner-runner.rb", "0.0.1", archive)
    with pytest.raises(ValueError, match="does not carry release version"):
        assemble(release, version, [tmp_path / "onnx"], homebrew_formula=stale)
    wrong_digest = tmp_path / "wrong" / "gliner-runner.rb"
    wrong_digest.parent.mkdir()
    wrong_digest.write_text(
        f'url ".../download/v{version}/x.tar.gz"\nsha256 "{"0" * 64}"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lacks the digest"):
        assemble(release, version, [tmp_path / "onnx"], homebrew_formula=wrong_digest)


def test_prerelease_version_must_be_stamped_everywhere(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    write_onnx_runtime(tmp_path / "onnx", "9.8.7-rc.2")
    with pytest.raises(ValueError, match="is not synchronized"):
        assemble(release, "9.8.7-rc.2", [tmp_path / "onnx"])


def test_channel_must_match_version(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    with pytest.raises(ValueError, match="not a rc release version"):
        assemble(tmp_path, version, [], channel="rc")
    with pytest.raises(ValueError, match="unsupported release version"):
        pep440_version("1.2")


def test_pep440_spelling_of_semver() -> None:
    assert pep440_version("1.2.3") == "1.2.3"
    assert pep440_version("1.2.3-rc.4") == "1.2.3rc4"


def test_release_assembly_requires_onnx_runtime_checksum_file(tmp_path: Path) -> None:
    version = next(iter(repository_versions().values()))
    release = tmp_path / "release"
    release.mkdir()
    write_package_assets(release, version)
    name = write_onnx_runtime(tmp_path / "onnx", version)
    (tmp_path / "onnx" / f"{name}.sha256").unlink()

    with pytest.raises(ValueError, match="missing checksum file"):
        assemble(release, version, [tmp_path / "onnx"])
