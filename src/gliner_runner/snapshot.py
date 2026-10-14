"""Immutable model-snapshot identity and content validation shared by build tools."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_EXACT_REVISION = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_MARKER_NAME = ".gliner-runner.json"
_ADMINISTRATIVE_PREFIX = ".cache/huggingface/"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_inventory_marker(marker_path: Path) -> dict[str, Any]:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"model snapshot marker is invalid: {marker_path}") from error
    if not isinstance(marker, dict):
        raise ValueError(f"model snapshot marker must contain a JSON object: {marker_path}")
    return marker


def _inventory_entries(root: Path, marker: dict[str, Any]) -> list[tuple[str, Path, int, str]]:
    files = marker.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"model snapshot marker has no file inventory: {root / _MARKER_NAME}")
    entries: list[tuple[str, Path, int, str]] = []
    seen: set[str] = set()
    for artifact in files:
        if not isinstance(artifact, dict):
            raise ValueError(
                f"model snapshot marker has an invalid file entry: {root / _MARKER_NAME}"
            )
        relative = artifact.get("path")
        size = artifact.get("size")
        sha256 = artifact.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or relative == _MARKER_NAME
            or relative.startswith(_ADMINISTRATIVE_PREFIX)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ValueError(
                f"model snapshot marker has an invalid file entry: {root / _MARKER_NAME}"
            )
        seen.add(relative)
        artifact_path = (root / relative).resolve()
        if not artifact_path.is_relative_to(root) or artifact_path == root:
            raise ValueError(f"model artifact path escapes the snapshot: {relative}")
        entries.append((relative, artifact_path, size, sha256))
    return entries


def _is_administrative(relative: str) -> bool:
    return relative == _MARKER_NAME or relative.startswith(_ADMINISTRATIVE_PREFIX)


def _validate_inventory_marker(
    root: Path,
    marker_path: Path,
    repository: str,
    revision: str,
) -> list[tuple[str, Path, int, str]]:
    marker = _read_inventory_marker(marker_path)
    if marker.get("repository") != repository or marker.get("revision") != revision:
        raise ValueError(
            "model snapshot marker identity does not match the requested repository and revision"
        )

    entries = _inventory_entries(root, marker)
    inventoried = {relative for relative, _path, _size, _sha256 in entries}
    for _relative, artifact_path, size, sha256 in entries:
        if not artifact_path.is_file() or artifact_path.stat().st_size != size:
            raise ValueError(f"model artifact size mismatch: {artifact_path}")
        if file_sha256(artifact_path) != sha256:
            raise ValueError(f"model artifact checksum mismatch: {artifact_path}")

    active_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not _is_administrative(path.relative_to(root).as_posix())
    }
    unlisted = sorted(active_files - inventoried)
    if unlisted:
        raise ValueError(f"model snapshot contains unlisted active file: {unlisted[0]}")
    return entries


def validate_model_snapshot(
    model_path: Path,
    repository: str,
    revision: str,
) -> Path:
    """Validate that a local snapshot has the requested immutable identity."""
    if _REPOSITORY.fullmatch(repository) is None:
        raise ValueError("model repository must be an explicit owner/name repository")
    if _EXACT_REVISION.fullmatch(revision) is None:
        raise ValueError("model revision must be an exact 40-character lowercase commit SHA")

    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    marker_path = resolved / _MARKER_NAME
    if marker_path.exists():
        if not marker_path.is_file():
            raise ValueError(f"model snapshot marker is not a file: {marker_path}")
        _validate_inventory_marker(resolved, marker_path, repository, revision)
        return resolved

    expected_repository_directory = f"models--{repository.replace('/', '--')}"
    if len(resolved.parts) >= 3:
        repository_directory, snapshots_directory, snapshot_directory = resolved.parts[-3:]
        if (
            snapshot_directory == revision
            and snapshots_directory == "snapshots"
            and repository_directory == expected_repository_directory
        ):
            return resolved

    raise ValueError(
        "model path identity does not match the requested repository and revision; "
        "provide a matching .gliner-runner.json marker or use "
        f".../{expected_repository_directory}/snapshots/{revision}"
    )


def snapshot_tree_digest(path: Path) -> tuple[str, int, int]:
    """Hash snapshot identity files by relative path and content."""
    root = path.expanduser().resolve()
    marker_path = root / _MARKER_NAME
    if marker_path.is_file():
        marker = _read_inventory_marker(marker_path)
        files = [
            (relative, artifact_path)
            for relative, artifact_path, _size, _sha256 in _inventory_entries(root, marker)
        ]
    else:
        files = [
            (file_path.relative_to(root).as_posix(), file_path)
            for file_path in root.rglob("*")
            if file_path.is_file()
        ]

    digest = hashlib.sha256()
    total_bytes = 0
    for relative, file_path in sorted(files):
        file_digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return digest.hexdigest(), total_bytes, len(files)
