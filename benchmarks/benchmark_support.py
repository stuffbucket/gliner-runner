from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from gliner_runner.snapshot import snapshot_tree_digest, validate_model_snapshot

__all__ = [
    "prepare_output_directory",
    "snapshot_tree_digest",
    "stable_sha256",
    "validate_gliner2_version",
    "validate_model_snapshot",
    "write_new_text",
]


def validate_gliner2_version(expected_version: str) -> str:
    """Require an explicit expected version and return the installed version."""
    if not expected_version or expected_version.strip() != expected_version:
        raise ValueError("expected GLiNER2 version must be a nonempty exact version")
    installed_version = importlib.metadata.version("gliner2")
    if installed_version != expected_version:
        raise RuntimeError(
            "installed GLiNER2 version does not match the benchmark expectation: "
            f"expected {expected_version}, found {installed_version}"
        )
    return installed_version


def prepare_output_directory(path: Path) -> Path:
    """Create a new output directory without reusing prior run state."""
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"benchmark output path already exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def write_new_text(path: Path, content: str) -> Path:
    """Write a new evidence file without replacing an existing path."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return resolved


def stable_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with a deterministic encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
