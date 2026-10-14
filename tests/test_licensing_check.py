from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.check_licensing import (
    ROOT,
    check,
    check_onnx_runtime,
    lockfile_production_closure,
)

ONNX_IMPORTER = "packages/onnx-runner"


def read(name: str) -> str:
    return (Path(ROOT) / name).read_text(encoding="utf-8")


def run_onnx_check(*, notices: str, lockfile: str | None = None) -> list[str]:
    errors: list[str] = []
    check_onnx_runtime(
        errors,
        notices=notices,
        project_notice=read("NOTICE"),
        lockfile=lockfile or read("pnpm-lock.yaml"),
        mise=tomllib.loads(read("mise.toml")),
    )
    return errors


def test_repository_licensing_declarations_are_consistent() -> None:
    assert check() == []


def test_onnx_closure_is_the_exact_production_set_without_dev_tooling() -> None:
    closure = lockfile_production_closure(read("pnpm-lock.yaml"), ONNX_IMPORTER)
    names = {name for name, _ in closure}

    assert {"onnxruntime-node", "onnxruntime-common", "@huggingface/tokenizers", "zod"} <= names
    assert ("onnxruntime-node", "1.30.0") in closure
    assert not names & {"typescript", "@types/node", "undici-types"}
    assert not names & {"torch", "transformers", "gliner2", "onnx"}


def test_missing_onnx_runtime_notice_row_is_reported() -> None:
    notices = read("THIRD_PARTY_NOTICES.md")
    assert run_onnx_check(notices=notices) == []

    edited = notices.replace("| `global-agent` | `4.1.3` | BSD-3-Clause |\n", "")
    assert edited != notices
    assert any("global-agent@4.1.3" in error for error in run_onnx_check(notices=edited))


def test_changed_node_runtime_pin_must_update_notices() -> None:
    notices = read("THIRD_PARTY_NOTICES.md").replace(
        "| Node.js runtime (unmodified official build) | `24.21.0` |",
        "| Node.js runtime (unmodified official build) | `24.20.0` |",
    )

    assert any("bundled Node.js" in error for error in run_onnx_check(notices=notices))


def test_dependency_drift_in_lockfile_is_reported() -> None:
    original = read("pnpm-lock.yaml")
    lockfile = original.replace(
        "      onnxruntime-common: 1.30.0", "      onnxruntime-common: 1.30.1"
    )
    assert lockfile != original

    errors = run_onnx_check(notices=read("THIRD_PARTY_NOTICES.md"), lockfile=lockfile)

    assert any("onnxruntime-common@1.30.1" in error for error in errors)
