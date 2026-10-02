from __future__ import annotations

from pathlib import Path

import pytest

from gliner_runner.errors import ModelIntegrityError
from gliner_runner.model_manifest import Artifact, ModelManifest, ModelStore, _safe_target


def test_manifest_content_address_is_stable() -> None:
    manifest = ModelManifest(
        model_id="example/model",
        source="https://huggingface.co/example/model",
        revision="a" * 40,
        license="Apache-2.0",
        backend="pytorch",
        precisions=["fp32"],
        operations=["classify"],
        artifacts=[
            Artifact(
                path="model.safetensors",
                url="https://example.invalid/model.safetensors",
                sha256="b" * 64,
                size=42,
            )
        ],
    )

    assert len(manifest.content_id) == 64
    assert manifest.content_id == manifest.model_copy().content_id


def test_artifact_path_cannot_escape_store(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError, match="escapes"):
        _safe_target(tmp_path, "../outside")


async def test_model_store_never_downloads_during_verify(tmp_path: Path) -> None:
    manifest = ModelManifest(
        model_id="missing/model",
        source="https://example.invalid",
        revision="a" * 40,
        license="Apache-2.0",
        backend="pytorch",
        precisions=["fp32"],
        operations=["classify"],
        artifacts=[
            Artifact(
                path="model.safetensors",
                url="https://example.invalid/model.safetensors",
                sha256="b" * 64,
                size=42,
            )
        ],
    )

    with pytest.raises(ModelIntegrityError, match="models install"):
        await ModelStore(tmp_path).verify(manifest)
