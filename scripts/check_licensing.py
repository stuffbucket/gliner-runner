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


def _unquote(value: str) -> str:
    return value.strip().strip("'\"")


def lockfile_production_closure(lockfile: str, importer: str) -> set[tuple[str, str]]:
    """Return the exact (name, version) production closure of a pnpm importer."""
    lines = lockfile.splitlines()

    def section(header: str) -> list[str]:
        start = lines.index(header) + 1
        end = start
        while end < len(lines) and (not lines[end] or lines[end].startswith(" ")):
            end += 1
        return lines[start:end]

    # importer -> direct production dependencies
    direct: dict[str, str] = {}
    block = section("importers:")
    in_importer = in_dependencies = False
    current = ""
    for line in block:
        if re.match(r"^  \S", line):
            in_importer = _unquote(line.strip().rstrip(":")) == importer
            in_dependencies = False
        elif in_importer and re.match(r"^    \S", line):
            in_dependencies = line.strip() == "dependencies:"
        elif in_importer and in_dependencies:
            if re.match(r"^      \S", line):
                current = _unquote(line.strip().rstrip(":"))
            elif line.strip().startswith("version:"):
                direct[current] = _unquote(line.split(":", 1)[1])

    # snapshots -> dependency edges (optional dependencies are bundled only if present)
    edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
    owner: tuple[str, str] | None = None
    in_edges = False
    for line in section("snapshots:"):
        if re.match(r"^  \S", line):
            key = _unquote(line.strip().rstrip(":").removesuffix(" {}").rstrip(":"))
            key = key.split("(", 1)[0]
            name, _, version = key.rpartition("@")
            owner = (name, version)
            edges.setdefault(owner, [])
            in_edges = False
        elif re.match(r"^    \S", line):
            in_edges = line.strip() == "dependencies:"
        elif in_edges and owner is not None and re.match(r"^      \S", line):
            name, _, version = line.strip().partition(":")
            edges[owner].append((_unquote(name), _unquote(version).split("(", 1)[0]))

    closure: set[tuple[str, str]] = set()
    pending = list(direct.items())
    while pending:
        item = pending.pop()
        if item in closure:
            continue
        closure.add(item)
        pending.extend(edges.get(item, []))
    return closure


def check_onnx_runtime(
    errors: list[str],
    *,
    notices: str,
    project_notice: str,
    lockfile: str,
    mise: dict[str, object],
) -> None:
    package = json.loads((ROOT / "packages/onnx-runner/package.json").read_text())
    dependencies: dict[str, str] = package["dependencies"]
    closure = lockfile_production_closure(lockfile, "packages/onnx-runner")
    direct = set(dependencies.items())
    if not direct <= closure:
        errors.append("pnpm-lock.yaml does not match onnx-runner's exact production dependencies")
    for name, version in sorted(dependencies.items()):
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            errors.append(f"onnx-runner dependency {name} must be pinned exactly, found {version}")
    for name, version in sorted(closure):
        if not re.search(
            rf"^\| `{re.escape(name)}` \| `{re.escape(version)}` \| \S", notices, re.M
        ):
            errors.append(
                f"THIRD_PARTY_NOTICES.md is missing ONNX runtime package {name}@{version}"
            )
    forbidden = sorted(
        name for name, _ in closure if re.match(r"(torch|transformers|gliner2?|onnx)$", name)
    )
    if forbidden:
        errors.append(f"ONNX runtime closure contains Python-era inference packages: {forbidden}")

    node_version = str(mise["tools"]["node"])  # type: ignore[index]
    node_row = (
        f"| Node.js runtime (unmodified official build) | `{node_version}` "
        "| MIT and bundled third-party licenses |"
    )
    if node_row not in notices:
        errors.append(f"THIRD_PARTY_NOTICES.md is missing the bundled Node.js {node_version} row")
    ort_version = dependencies.get("onnxruntime-node", "")
    ort_row = f"| ONNX Runtime native library (inside `onnxruntime-node`) | `{ort_version}` | MIT |"
    if ort_row not in notices:
        errors.append("THIRD_PARTY_NOTICES.md is missing the ONNX Runtime native library row")
    if "licenses/node-LICENSE" not in notices:
        errors.append("THIRD_PARTY_NOTICES.md must record the shipped Node.js license file")
    for text in ("Node.js", "ONNX Runtime", "@huggingface/tokenizers"):
        if text not in project_notice:
            errors.append(f"NOTICE is missing the ONNX runtime attribution for {text}")

    packaging = (ROOT / "scripts/package-onnx-runtime.mjs").read_text()
    for text in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "release-manifest.json"):
        if text not in packaging:
            errors.append(f"ONNX runtime packaging script does not ship {text}")
    for task in ("package:onnx-runtime", "verify:onnx-runtime"):
        if task not in mise.get("tasks", {}):  # type: ignore[operator]
            errors.append(f"mise.toml is missing the {task} task")


def check() -> list[str]:
    errors: list[str] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    client = json.loads((ROOT / "packages/client/package.json").read_text())
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    project_notice = (ROOT / "NOTICE").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    model_inventory = (ROOT / "src/gliner_runner/model_inventory.py").read_text()
    lockfile = (ROOT / "pnpm-lock.yaml").read_text()
    mise = tomllib.loads((ROOT / "mise.toml").read_text())
    check_onnx_runtime(
        errors,
        notices=notices,
        project_notice=project_notice,
        lockfile=lockfile,
        mise=mise,
    )

    onnx_export_specs = [
        "gliner2[local]==2.1.0",
        "torch==2.14.0",
        "transformers==5.19.0",
        "onnx==1.23.2",
        "onnxruntime==1.31.0",
        "onnxscript==0.7.2",
    ]
    declared_onnx_export = pyproject["project"]["optional-dependencies"].get("onnx-export", [])
    if declared_onnx_export != onnx_export_specs:
        errors.append("onnx-export dependencies no longer match the qualified exact versions")

    python_specs = [
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["pytorch"],
        *declared_onnx_export,
    ]
    for specifier in python_specs:
        name = dependency_name(specifier)
        if f"| `{name}` |" not in notices and f"| `{name}` (" not in notices:
            errors.append(f"THIRD_PARTY_NOTICES.md is missing Python dependency {name}")
    for specifier in onnx_export_specs:
        name = dependency_name(specifier)
        version = specifier.rsplit("==", maxsplit=1)[1]
        expected_row = f"| `{name}` (optional ONNX export extra) | `=={version}` |"
        if expected_row not in notices:
            errors.append(
                f"THIRD_PARTY_NOTICES.md is missing exact ONNX export dependency {name}=={version}"
            )

    for name, version in client["dependencies"].items():
        if f"| `{name}` | `{version}` |" not in notices:
            errors.append(f"THIRD_PARTY_NOTICES.md is missing client dependency {name}@{version}")
        lock_entry = f"  {name}@{version}:"
        if lock_entry not in lockfile:
            errors.append(
                f"pnpm-lock.yaml does not contain exact client dependency {name}@{version}"
            )

    curated_models = (
        ("fastino/GLiNER2.5-Decide", "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90"),
        (
            "fastino/GLiNER2.5-multi-Decide",
            "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d",
        ),
        (
            "fastino/GLiNER2.5-Decide-1B",
            "67af949e17bebd20f05fe020a971079889e7a462",
        ),
    )
    for repository, revision in curated_models:
        if f"| `{repository}` | `{revision}` | `apache-2.0` |" not in notices:
            errors.append(
                f"THIRD_PARTY_NOTICES.md is missing curated model metadata: {repository}@{revision}"
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
        "cuda-toolkit=NVIDIA CUDA Toolkit EULA (metadata override)",
        "--require gliner2==2.1.0",
        "--require protobuf",
        "--require sentencepiece",
        "--require torch",
        "--require transformers",
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
