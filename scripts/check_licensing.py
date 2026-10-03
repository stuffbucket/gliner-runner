from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICE_FILES = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


def dependency_name(specifier: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", specifier)
    if match is None:
        raise ValueError(f"cannot parse dependency: {specifier}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def check() -> list[str]:
    errors: list[str] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    client = json.loads((ROOT / "packages/client/package.json").read_text())
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    project_notice = (ROOT / "NOTICE").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    model_inventory = (ROOT / "src/gliner_runner/model_inventory.py").read_text()
    lockfile = (ROOT / "pnpm-lock.yaml").read_text()

    python_specs = [
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["pytorch"],
    ]
    for specifier in python_specs:
        name = dependency_name(specifier)
        if f"| `{name}` |" not in notices and f"| `{name}` (" not in notices:
            errors.append(f"THIRD_PARTY_NOTICES.md is missing Python dependency {name}")

    for name, version in client["dependencies"].items():
        if f"| `{name}` | `{version}` |" not in notices:
            errors.append(f"THIRD_PARTY_NOTICES.md is missing client dependency {name}@{version}")
        lock_entry = f"  {name}@{version}:"
        if lock_entry not in lockfile:
            errors.append(
                f"pnpm-lock.yaml does not contain exact client dependency {name}@{version}"
            )

    curated_models = (
        ("fastino/GLiNER2.5-Decide", "5a7adf72a23b4d311abae6ce050d7f0012bb3416"),
        (
            "fastino/GLiNER2.5-multi-Decide",
            "a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f",
        ),
        (
            "fastino/GLiNER2.5-Decide-1B",
            "688cd7ba8917a0855ad3ce929cba5a9998932e79",
        ),
    )
    for repository, revision in curated_models:
        if f"| `{repository}` | `{revision}` | `apache-2.0` |" not in notices:
            errors.append(
                "THIRD_PARTY_NOTICES.md is missing curated model metadata: "
                f"{repository}@{revision}"
            )
        if repository not in model_inventory or revision not in model_inventory:
            errors.append(
                f"curated model inventory no longer matches notice: {repository}@{revision}"
            )

    required_model_statements = (
        "license: apache-2.0",
        "no root `LICENSE` or `NOTICE`",
        "does not redistribute the curated model weights",
    )
    for text in required_model_statements:
        if text not in notices:
            errors.append(f"THIRD_PARTY_NOTICES.md is missing required model statement: {text}")

    required_attributions = (
        "Requests\nCopyright 2019 Kenneth Reitz",
        "Asyncio support for files\nCopyright 2016 Tin Tvrtkovic",
    )
    for attribution in required_attributions:
        if attribution not in project_notice:
            errors.append(f"NOTICE is missing required attribution: {attribution!r}")

    license_files = pyproject["project"].get("license-files", [])
    sdist_files = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    client_files = client["files"]
    for name in NOTICE_FILES:
        if name not in license_files:
            errors.append(f"Python wheel license-files is missing {name}")
        if f"/{name}" not in sdist_files:
            errors.append(f"Python sdist is missing {name}")
        if name not in client_files:
            errors.append(f"client package is missing {name}")

    docker_requirements = (
        "THIRD_PARTY_NOTICES.md",
        "generate_third_party_inventory.py",
        'cuda-toolkit=NVIDIA CUDA Toolkit EULA (metadata override)',
        "--require gliner2==2.0.0",
        "--require torch",
        "PYTHON_PACKAGES.md",
    )
    for text in docker_requirements:
        if text not in dockerfile:
            errors.append(f"Docker release inventory gate is missing {text}")

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Licensing declarations and package surfaces are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
