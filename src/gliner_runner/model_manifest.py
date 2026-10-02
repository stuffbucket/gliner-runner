from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Annotated

import aiofiles
import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter

from gliner_runner.contracts import BackendName, InferenceOperation, Precision
from gliner_runner.errors import ModelIntegrityError

Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    url: HttpUrl
    sha256: Sha256
    size: int = Field(gt=0)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int = Field(default=1, ge=1, le=1)
    model_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    license: str = Field(min_length=1)
    provenance: dict[str, str] = Field(default_factory=dict)
    backend: BackendName
    precisions: frozenset[Precision]
    operations: frozenset[InferenceOperation]
    artifacts: tuple[Artifact, ...] = Field(min_length=1)

    @property
    def content_id(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_toml(cls, path: Path) -> ModelManifest:
        with path.open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))


class ModelStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def materialize(self, manifest: ModelManifest) -> Path:
        destination = self.root / manifest.content_id
        destination.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
            for artifact in manifest.artifacts:
                target = _safe_target(destination, artifact.path)
                if target.exists() and _sha256(target) == artifact.sha256:
                    continue
                await self._download(client, artifact, target)
        return destination

    async def verify(self, manifest: ModelManifest) -> Path:
        destination = self.root / manifest.content_id
        for artifact in manifest.artifacts:
            target = _safe_target(destination, artifact.path)
            if not target.is_file():
                raise ModelIntegrityError(
                    f"model artifact is not installed: {artifact.path}; "
                    "run 'gliner-runner models install' first"
                )
            digest = await asyncio.to_thread(_sha256, target)
            stat = await asyncio.to_thread(target.stat)
            size = stat.st_size
            if digest != artifact.sha256 or size != artifact.size:
                raise ModelIntegrityError(f"checksum or size mismatch for {artifact.path}")
        return destination

    async def _download(self, client: httpx.AsyncClient, artifact: Artifact, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            size = 0
            async with client.stream("GET", str(artifact.url)) as response:
                response.raise_for_status()
                async with aiofiles.open(temporary, "wb") as stream:
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        digest.update(chunk)
                        await stream.write(chunk)
            if size != artifact.size or digest.hexdigest() != artifact.sha256:
                raise ModelIntegrityError(f"checksum or size mismatch for {artifact.path}")
            await asyncio.to_thread(temporary.replace, target)
        finally:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ModelIntegrityError(f"artifact path escapes model directory: {relative}")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_file(path: Path) -> ModelManifest:
    return TypeAdapter(ModelManifest).validate_python(ModelManifest.from_toml(path))


class ModelCatalog:
    def __init__(self, directory: Path) -> None:
        self._manifests: dict[str, ModelManifest] = {}
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.toml")):
            manifest = ModelManifest.from_toml(path)
            for key in (manifest.model_id, manifest.content_id):
                if key in self._manifests:
                    raise ValueError(f"duplicate model manifest key {key!r}")
                self._manifests[key] = manifest

    def resolve(self, model: str) -> ModelManifest:
        try:
            return self._manifests[model]
        except KeyError as error:
            raise ModelIntegrityError(
                f"model {model!r} has no pinned manifest in the configured catalog"
            ) from error
