from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "stuffbucket/gliner-runner"
RECEIPT_PATH = "release/onnx-eval-receipt.json"
SEMVER = re.compile(r"^(?P<core>\d+\.\d+\.\d+)(?:-(?P<kind>alpha|beta|rc)\.(?P<number>\d+))?$")
PEP440_KIND = {"alpha": "a", "beta": "b", "rc": "rc"}


def parse_semver(version: str) -> re.Match[str]:
    match = SEMVER.fullmatch(version)
    if match is None:
        raise ValueError(f"unsupported release version {version!r}; expected X.Y.Z or X.Y.Z-rc.N")
    return match


def pep440_version(version: str) -> str:
    """The PEP 440 spelling of a semver release version (1.2.3-rc.1 -> 1.2.3rc1)."""
    match = parse_semver(version)
    if match.group("kind") is None:
        return match.group("core")
    return f"{match.group('core')}{PEP440_KIND[match.group('kind')]}{match.group('number')}"


def default_channel(version: str) -> str:
    return "stable" if parse_semver(version).group("kind") is None else "rc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble machine-readable GitHub release assets.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version", required=True, help="semver, e.g. 1.2.3 or 1.2.3-rc.1")
    parser.add_argument(
        "--onnx-runtime-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "directory holding gliner-runner-onnx archives and their .sha256 files "
            "(repeatable); every archive is copied into the output directory"
        ),
    )
    parser.add_argument(
        "--channel",
        choices=("stable", "rc"),
        help="release channel; defaults to rc for prerelease versions, stable otherwise",
    )
    parser.add_argument("--installer", type=Path, help="install.sh to attach and checksum")
    parser.add_argument(
        "--homebrew-formula",
        type=Path,
        help=(
            "rendered Homebrew formula to attach and checksum: gliner-runner.rb for stable, "
            "gliner-runner-rc.rb for release candidates"
        ),
    )
    parser.add_argument("--repository", default=REPOSITORY, help="owner/name of the GitHub repo")
    parser.add_argument(
        "--stamp-only",
        action="store_true",
        help="rewrite the release-managed versions in the working tree and exit",
    )
    args = parser.parse_args()
    if not args.stamp_only and args.output_dir is None:
        parser.error("--output-dir is required")
    return args


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
    client_package = json.loads((ROOT / "packages/client/package.json").read_text(encoding="utf-8"))
    onnx_runner_package = json.loads(
        (ROOT / "packages/onnx-runner/package.json").read_text(encoding="utf-8")
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
        "onnxRunnerPackage": str(onnx_runner_package["version"]),
        "runtime": init_match.group(1),
        "docker": docker_match.group(1),
    }


ONNX_RUNTIME_ARCHIVE = re.compile(
    r"^gliner-runner-onnx-(?P<version>.+)-(?P<platform>[a-z0-9]+)-(?P<arch>[a-z0-9_]+)\.tar\.gz$"
)
ONNX_RUNTIME_CHECKSUM_SUFFIX = ".sha256"
HOMEBREW_FORMULAS = {"stable": "gliner-runner.rb", "rc": "gliner-runner-rc.rb"}


def artifact_role(path: Path) -> str:
    if ONNX_RUNTIME_ARCHIVE.fullmatch(path.name):
        return "onnx-runtime"
    if path.name.startswith("gliner-runner-onnx-") and path.name.endswith(
        f".tar.gz{ONNX_RUNTIME_CHECKSUM_SUFFIX}"
    ):
        return "onnx-runtime-checksum"
    if path.suffix == ".whl":
        return "python-wheel"
    if path.name.endswith(".tar.gz"):
        return "python-sdist"
    if path.suffix == ".tgz":
        return "typescript-client"
    if path.name == "openapi.json":
        return "openapi"
    if path.name == "install.sh":
        return "installer"
    if path.name in HOMEBREW_FORMULAS.values():
        return "homebrew-formula"
    raise ValueError(f"unexpected release artifact: {path.name}")


def media_type(path: Path) -> str:
    if path.name.endswith(ONNX_RUNTIME_CHECKSUM_SUFFIX):
        return "text/plain"
    if path.suffix == ".whl":
        return "application/zip"
    if path.name.endswith((".tar.gz", ".tgz")):
        return "application/gzip"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".sh":
        return "text/x-shellscript"
    if path.suffix == ".rb":
        return "text/x-ruby"
    return "application/octet-stream"


def onnx_archives(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and ONNX_RUNTIME_ARCHIVE.fullmatch(path.name)
    )


def stage_onnx_runtime(source_dirs: list[Path], output_dir: Path) -> None:
    """Copy every platform ONNX runtime archive and its checksum file."""
    archives = [archive for source in source_dirs for archive in onnx_archives(source)]
    if not archives:
        raise ValueError("expected at least one gliner-runner-onnx archive, found none")
    for archive in archives:
        checksum = archive.with_name(f"{archive.name}{ONNX_RUNTIME_CHECKSUM_SUFFIX}")
        if not checksum.is_file():
            raise ValueError(f"missing checksum file {checksum.name}")
        for path in (archive, checksum):
            shutil.copyfile(path, output_dir / path.name)


def verify_onnx_runtime(output_dir: Path, version: str) -> None:
    archives = onnx_archives(output_dir)
    if not archives:
        raise ValueError("release assets must contain at least one gliner-runner-onnx archive")
    seen: set[tuple[str, str]] = set()
    for archive in archives:
        match = ONNX_RUNTIME_ARCHIVE.fullmatch(archive.name)
        if match is None or match.group("version") != version:
            raise ValueError(f"{archive.name} does not carry release version {version}")
        target = (match.group("platform"), match.group("arch"))
        if target in seen:
            raise ValueError(f"more than one gliner-runner-onnx archive for {'-'.join(target)}")
        seen.add(target)
        checksum = output_dir / f"{archive.name}{ONNX_RUNTIME_CHECKSUM_SUFFIX}"
        if not checksum.is_file():
            raise ValueError(f"missing checksum file {checksum.name}")
        expected = f"{file_sha256(archive)}  {archive.name}\n"
        if checksum.read_text(encoding="utf-8") != expected:
            raise ValueError(f"{checksum.name} does not match {archive.name}")


def describe_artifact(path: Path) -> dict[str, object]:
    artifact: dict[str, object] = {
        "name": path.name,
        "role": artifact_role(path),
        "mediaType": media_type(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    match = ONNX_RUNTIME_ARCHIVE.fullmatch(path.name)
    if match is not None:
        artifact["platform"] = match.group("platform")
        artifact["arch"] = match.group("arch")
    return artifact


def expected_versions(version: str) -> dict[str, str]:
    """The version each release-managed marker must carry for a semver release version."""
    python = pep440_version(version)
    return {
        "python": python,
        "rootPackage": version,
        "clientPackage": version,
        "onnxRunnerPackage": version,
        "runtime": python,
        "docker": python,
    }


def stamp_versions(version: str) -> None:
    """Rewrite release-managed versions in the working tree (used for release candidates)."""
    python = pep440_version(version)

    def sub(path: str, pattern: str, replacement: str) -> None:
        file = ROOT / path
        text = file.read_text(encoding="utf-8")
        updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"version marker not found in {path}")
        file.write_text(updated, encoding="utf-8")

    sub("pyproject.toml", r'^version = "[^"]+"', f'version = "{python}"')
    for package in (
        "package.json",
        "packages/client/package.json",
        "packages/onnx-runner/package.json",
    ):
        sub(package, r'^(\s*"version": )"[^"]+"', rf'\g<1>"{version}"')
    sub("src/gliner_runner/__init__.py", r'^__version__ = "[^"]+"', f'__version__ = "{python}"')
    sub("Dockerfile", r"^(ARG GLINER_RUNNER_VERSION=)\S+", rf"\g<1>{python}")


def receipt_summary() -> dict[str, object]:
    raw = (ROOT / RECEIPT_PATH).read_bytes()
    receipt = json.loads(raw)
    return {
        "path": RECEIPT_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "runId": receipt["run_id"],
        "identitySha256": receipt["runner"]["identity_sha256"],
        "evidencePlatform": receipt["host"]["platform"],
        "evidenceArch": receipt["host"]["arch"],
    }


def release_url(repository: str, version: str, name: str | None = None) -> str:
    base = f"https://github.com/{repository}/releases"
    if name is None:
        return f"{base}/tag/v{version}"
    return f"{base}/download/v{version}/{name}"


def assemble(
    output_dir: Path,
    version: str,
    onnx_runtime_dirs: list[Path] | None = None,
    channel: str | None = None,
    installer: Path | None = None,
    repository: str = REPOSITORY,
    homebrew_formula: Path | None = None,
) -> None:
    channel = channel or default_channel(version)
    if channel != default_channel(version):
        raise ValueError(f"version {version} is not a {channel} release version")
    expected = expected_versions(version)
    versions = repository_versions()
    mismatches = {name: value for name, value in versions.items() if value != expected[name]}
    if mismatches:
        details = ", ".join(f"{name}={value}" for name, value in sorted(mismatches.items()))
        raise ValueError(f"release version {version} is not synchronized: {details}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if onnx_runtime_dirs:
        stage_onnx_runtime(onnx_runtime_dirs, output_dir)
    verify_onnx_runtime(output_dir, version)
    shutil.copyfile(ROOT / "openapi/openapi.json", output_dir / "openapi.json")
    if installer is not None:
        shutil.copyfile(installer, output_dir / "install.sh")
    if homebrew_formula is not None:
        if homebrew_formula.name != HOMEBREW_FORMULAS[channel]:
            raise ValueError(
                f"a {channel} release attaches {HOMEBREW_FORMULAS[channel]}, "
                f"not {homebrew_formula.name}"
            )
        formula_text = homebrew_formula.read_text(encoding="utf-8")
        if f"/download/v{version}/" not in formula_text:
            raise ValueError(f"{homebrew_formula.name} does not carry release version {version}")
        for archive in onnx_archives(output_dir):
            if file_sha256(archive) not in formula_text:
                raise ValueError(f"{homebrew_formula.name} lacks the digest of {archive.name}")
        if homebrew_formula.parent != output_dir:
            shutil.copyfile(homebrew_formula, output_dir / homebrew_formula.name)
    generated = {"release-manifest.json", "SHA256SUMS", "latest.json"}
    package_assets = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name not in generated
    )
    artifacts = [describe_artifact(path) for path in package_assets]
    roles = [str(artifact["role"]) for artifact in artifacts]
    single_roles = {"python-wheel", "python-sdist", "typescript-client", "openapi"}
    optional_roles = {"installer", "homebrew-formula"}
    per_platform_roles = {"onnx-runtime", "onnx-runtime-checksum"}
    counts = {role: roles.count(role) for role in set(roles)}
    if (
        not single_roles <= set(counts)
        or not per_platform_roles <= set(counts)
        or not set(counts) <= single_roles | optional_roles | per_platform_roles
        or any(counts[role] != 1 for role in single_roles | (optional_roles & set(counts)))
        or counts["onnx-runtime"] != counts["onnx-runtime-checksum"]
    ):
        raise ValueError(f"release assets do not have the required roles: {sorted(counts.items())}")

    receipt = receipt_summary()
    platforms = []
    for artifact in artifacts:
        if artifact["role"] != "onnx-runtime":
            continue
        name = str(artifact["name"])
        platform, arch = str(artifact["platform"]), str(artifact["arch"])
        evidenced = (platform, arch) == (receipt["evidencePlatform"], receipt["evidenceArch"])
        platforms.append(
            {
                "platform": platform,
                "arch": arch,
                "archive": name,
                "checksum": f"{name}{ONNX_RUNTIME_CHECKSUM_SUFFIX}",
                "sha256": artifact["sha256"],
                "size": artifact["size"],
                "executable": f"bin/gliner-runner-onnx{'.cmd' if platform == 'win32' else ''}",
                "smokeTested": True,
                "parityEvidence": evidenced,
                "status": "parity-evidence" if evidenced else "smoke-tested",
            }
        )
    platforms.sort(key=lambda entry: (str(entry["platform"]), str(entry["arch"])))

    manifest = {
        "formatVersion": 2,
        "name": "gliner-runner",
        "version": version,
        "channel": channel,
        "prerelease": channel != "stable",
        "artifacts": artifacts,
        "modelsIncluded": False,
        "providerContract": {
            "transport": "http-json",
            "openapiArtifact": "openapi.json",
            "executable": "gliner-runner",
            "healthPath": "/healthz",
        },
        "onnxRuntime": {
            "archives": platforms,
            "healthPath": "/healthz",
            "backend": "onnx",
            "device": "cpu",
            "precision": "fp32",
            "operations": ["classify"],
            "pythonRequired": False,
            "modelsIncluded": False,
            "receipt": receipt,
        },
    }
    manifest_path = output_dir / "release-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksummed = [*package_assets, manifest_path]
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in sorted(checksummed)),
        encoding="utf-8",
    )

    latest_path = output_dir / "latest.json"
    latest_path.unlink(missing_ok=True)
    if channel == "stable":
        # The upgrade feed. Prereleases never get one, so they never become an upgrade target.
        def reference(name: str, sha256: str) -> dict[str, str]:
            return {"name": name, "url": release_url(repository, version, name), "sha256": sha256}

        latest = {
            "formatVersion": 1,
            "name": "gliner-runner",
            "version": version,
            "channel": channel,
            "releaseUrl": release_url(repository, version),
            "notesUrl": release_url(repository, version),
            "manifest": reference("release-manifest.json", file_sha256(manifest_path)),
            "checksums": reference("SHA256SUMS", file_sha256(sums_path)),
            "platforms": {
                f"{entry['platform']}-{entry['arch']}": {
                    **reference(str(entry["archive"]), str(entry["sha256"])),
                    "size": entry["size"],
                    "parityEvidence": entry["parityEvidence"],
                }
                for entry in platforms
            },
        }
        if installer is not None:
            latest["installer"] = reference("install.sh", file_sha256(output_dir / "install.sh"))
        latest_path.write_text(
            json.dumps(latest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    parse_semver(args.version)
    if args.stamp_only:
        stamp_versions(args.version)
        return
    assemble(
        args.output_dir.resolve(),
        args.version,
        [path.resolve() for path in args.onnx_runtime_dir],
        args.channel,
        None if args.installer is None else args.installer.resolve(),
        args.repository,
        None if args.homebrew_formula is None else args.homebrew_formula.resolve(),
    )


if __name__ == "__main__":
    main()
