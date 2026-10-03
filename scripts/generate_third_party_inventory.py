from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from pathlib import Path

LICENSE_FILE_PREFIXES = ("license", "licence", "copying", "notice", "authors")


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def license_files(distribution: importlib.metadata.Distribution) -> tuple[str, ...]:
    paths: list[str] = []
    for file in distribution.files or ():
        name = Path(str(file)).name.lower()
        if not name.startswith(LICENSE_FILE_PREFIXES):
            continue
        paths.append(str(file))
    return tuple(sorted(paths))


def license_label(distribution: importlib.metadata.Distribution, files: tuple[str, ...]) -> str:
    metadata = distribution.metadata
    expression_values = metadata.get_all("License-Expression") or []
    expression_value = expression_values[0] if expression_values else ""
    expression = expression_value.strip() if expression_value else ""
    if expression:
        return expression

    legacy_values = metadata.get_all("License") or []
    legacy_value = legacy_values[0] if legacy_values else ""
    legacy = legacy_value.strip() if legacy_value else ""
    if legacy and "\n" not in legacy and len(legacy) <= 120:
        return legacy

    classifiers = metadata.get_all("Classifier", [])
    licenses = [
        classifier.removeprefix("License :: ").strip()
        for classifier in classifiers
        if classifier.startswith("License :: ")
    ]
    if licenses:
        return "; ".join(licenses)
    if files:
        return "See packaged license files"
    return "UNKNOWN"


def parse_requirement(value: str) -> tuple[str, str | None]:
    name, separator, version = value.partition("==")
    if not name or (separator and not version):
        raise ValueError(f"invalid requirement: {value}")
    return normalized_name(name), version or None


def parse_license_overrides(values: tuple[str, ...]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, license_name = value.partition("=")
        if not separator or not name or not license_name:
            raise ValueError(f"invalid license override: {value}")
        normalized = normalized_name(name)
        if normalized in overrides:
            raise ValueError(f"duplicate license override: {name}")
        overrides[normalized] = license_name
    return overrides


def generate_inventory(
    distributions: list[importlib.metadata.Distribution],
    *,
    excluded: frozenset[str],
    license_overrides: dict[str, str],
    required: tuple[str, ...],
) -> str:
    rows: list[tuple[str, str, str, tuple[str, ...]]] = []
    installed: dict[str, str] = {}
    missing_licenses: list[str] = []

    for distribution in distributions:
        name = distribution.metadata["Name"]
        if not name:
            continue
        normalized = normalized_name(name)
        version = distribution.version
        installed[normalized] = version
        if normalized in excluded:
            continue

        files = license_files(distribution)
        license_name = license_overrides.get(normalized) or license_label(distribution, files)
        if license_name == "UNKNOWN":
            missing_licenses.append(f"{name}=={version}")
        rows.append((name, version, license_name, files))

    requirement_errors: list[str] = []
    for requirement in required:
        name, required_version = parse_requirement(requirement)
        actual = installed.get(name)
        if actual is None:
            requirement_errors.append(f"required distribution is missing: {name}")
        elif required_version is not None and actual != required_version:
            requirement_errors.append(
                f"required {name}=={required_version}, found {actual}"
            )

    errors = requirement_errors + [
        f"distribution has no license metadata or packaged license file: {item}"
        for item in missing_licenses
    ]
    if errors:
        raise ValueError("\n".join(errors))

    lines = [
        "# Exact installed Python distribution inventory",
        "",
        "Generated from the installed distributions in this release image.",
        "This file is not a substitute for the packaged license files named below.",
        "",
        "| Distribution | Version | Declared license | Packaged license/notice files |",
        "| --- | --- | --- | --- |",
    ]
    for name, version, license_name, files in sorted(rows, key=lambda row: row[0].lower()):
        escaped_license = license_name.replace("|", "\\|")
        file_list = "<br>".join(f"`{path}`" for path in files) or "None"
        lines.append(f"| `{name}` | `{version}` | {escaped_license} | {file_list} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--license-override", action="append", default=[])
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()

    try:
        license_overrides = parse_license_overrides(tuple(args.license_override))
        content = generate_inventory(
            list(importlib.metadata.distributions()),
            excluded=frozenset(normalized_name(name) for name in args.exclude),
            license_overrides=license_overrides,
            required=tuple(args.require),
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
