from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from gliner_runner.contracts import Precision, PrecisionProfile
from gliner_runner.errors import ModelDownloadRequiredError, ModelIntegrityError


class InventoryModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    repository: str
    pinned_revision: str
    parameter_count: int
    description: str


KNOWN_MODELS = (
    InventoryModelSpec(
        model_id="decide-340m",
        repository="fastino/GLiNER2.5-Decide",
        pinned_revision="5a7adf72a23b4d311abae6ce050d7f0012bb3416",
        parameter_count=486_444_053,
        description="English GLiNER2.5 classification model",
    ),
    InventoryModelSpec(
        model_id="decide-multilingual",
        repository="fastino/GLiNER2.5-multi-Decide",
        pinned_revision="a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f",
        parameter_count=287_355_159,
        description="Multilingual GLiNER2.5 classification model",
    ),
    InventoryModelSpec(
        model_id="decide-1b",
        repository="fastino/GLiNER2.5-Decide-1B",
        pinned_revision="688cd7ba8917a0855ad3ce929cba5a9998932e79",
        parameter_count=1_188_796_693,
        description="One-billion-parameter English GLiNER2.5 classification model",
    ),
)


class DownloadState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    destination: str = Field(min_length=1)
    revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")


class DownloadJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    model_id: str
    revision: str
    destination: str
    state: DownloadState
    error: str | None = None


class RefreshResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refreshed_at: float
    models_checked: int
    changed_models: tuple[str, ...]
    discovered_models: tuple[str, ...]


class ProfileGuidance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimated_minimum_memory_bytes: int
    memory_basis: str
    baseline_profile: str
    observed_max_abs_score_drift: float | None
    drift_order_of_magnitude: str | None
    characterization_scope: str


class ModelInventoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    repository: str
    family: str = "GLiNER2.5-Decide"
    description: str
    pinned_revision: str
    discovered_revision: str
    parameter_count: int
    local_revisions: tuple[str, ...]
    available_locally: bool
    loaded: bool
    loaded_profile: PrecisionProfile | None
    supported_profiles: tuple[PrecisionProfile, ...]
    default_profile: PrecisionProfile
    profile_guidance: dict[str, ProfileGuidance]
    download_destination: str
    download_required: bool


class ModelInventoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: str
    device: str
    idle_ttl_seconds: float
    memory_limit_bytes: int
    discovered_models: tuple[str, ...]
    models: tuple[ModelInventoryItem, ...]


@dataclass(frozen=True)
class ResolvedModel:
    spec: InventoryModelSpec
    revision: str
    path: Path


class ModelInventory:
    def __init__(self, provider_root: Path) -> None:
        self.provider_root = provider_root.expanduser().resolve()
        self._specs = {spec.model_id: spec for spec in KNOWN_MODELS}
        self._specs.update({spec.repository: spec for spec in KNOWN_MODELS})
        self._cache_path = self.provider_root / "catalog-cache.json"
        self._cache = self._load_cache()
        self._lock = asyncio.Lock()
        self._verified: set[tuple[Path, int]] = set()

    def spec(self, model: str) -> InventoryModelSpec | None:
        model_id, _separator, _revision = model.partition("@")
        return self._specs.get(model_id)

    def revision(self, model: str, spec: InventoryModelSpec) -> str:
        _model_id, separator, revision = model.partition("@")
        return revision if separator else spec.pinned_revision

    def expected_path(self, spec: InventoryModelSpec, revision: str) -> Path:
        slug = spec.repository.replace("/", "--")
        return self.provider_root / slug / revision

    def resolve(self, model: str) -> ResolvedModel:
        spec = self.spec(model)
        if spec is None:
            raise ModelIntegrityError(f"model {model!r} is not in the GLiNER2.5 catalog")
        revision = self.revision(model, spec)
        path = self.expected_path(spec, revision)
        if not self._is_complete(path, spec, revision):
            raise ModelDownloadRequiredError(model, revision, str(self.provider_root))
        return ResolvedModel(spec=spec, revision=revision, path=path)

    async def resolve_verified(self, model: str) -> ResolvedModel:
        resolved = self.resolve(model)
        marker = resolved.path / ".gliner-runner.json"
        cache_key = (resolved.path, marker.stat().st_mtime_ns)
        if cache_key not in self._verified:
            await asyncio.to_thread(_verify_snapshot, resolved.path)
            self._verified.add(cache_key)
        return resolved

    async def refresh(self) -> RefreshResult:
        huggingface_hub = importlib.import_module("huggingface_hub")
        api = huggingface_hub.HfApi()
        changed: list[str] = []
        refreshed_at = time.time()
        async with self._lock:
            for spec in KNOWN_MODELS:
                info = await asyncio.to_thread(api.model_info, spec.repository)
                previous = self._cache.get(spec.model_id, {}).get("discovered_revision")
                if previous != info.sha:
                    changed.append(spec.model_id)
                self._cache[spec.model_id] = {
                    "discovered_revision": info.sha,
                    "refreshed_at": refreshed_at,
                }
            provider_models = await asyncio.to_thread(
                lambda: list(
                    api.list_models(author="fastino", search="GLiNER2.5", full=True)
                )
            )
            known_repositories = {spec.repository for spec in KNOWN_MODELS}
            discovered = sorted(
                info.id
                for info in provider_models
                if "decide" in info.id.casefold() and info.id not in known_repositories
            )
            self._cache["_discovered_models"] = {
                "repositories": discovered,
                "refreshed_at": refreshed_at,
            }
            await asyncio.to_thread(self._write_cache)
        return RefreshResult(
            refreshed_at=refreshed_at,
            models_checked=len(KNOWN_MODELS),
            changed_models=tuple(changed),
            discovered_models=tuple(discovered),
        )

    async def download(self, model_id: str, revision: str, destination: str) -> Path:
        requested_destination = await asyncio.to_thread(_resolved_path, destination)
        if requested_destination != self.provider_root:
            raise ValueError(
                f"destination must equal the configured provider location {self.provider_root}"
            )
        spec = self.spec(model_id)
        if spec is None:
            raise ValueError(f"unknown GLiNER2.5 model {model_id!r}")
        target = self.expected_path(spec, revision)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            importlib.import_module("huggingface_hub").snapshot_download,
            repo_id=spec.repository,
            revision=revision,
            local_dir=target,
        )
        files = await asyncio.to_thread(_snapshot_files, target)
        marker = {
            "model_id": spec.model_id,
            "repository": spec.repository,
            "revision": revision,
            "files": files,
        }
        await asyncio.to_thread(
            (target / ".gliner-runner.json").write_text,
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            "utf-8",
        )
        return target

    def discovered_revision(self, spec: InventoryModelSpec) -> str:
        value = self._cache.get(spec.model_id, {}).get("discovered_revision")
        return value if isinstance(value, str) else spec.pinned_revision

    def discovered_models(self) -> tuple[str, ...]:
        value = self._cache.get("_discovered_models", {}).get("repositories")
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str))

    def local_revisions(self, spec: InventoryModelSpec) -> tuple[str, ...]:
        parent = self.expected_path(spec, spec.pinned_revision).parent
        if not parent.is_dir():
            return ()
        return tuple(
            sorted(
                path.name
                for path in parent.iterdir()
                if path.is_dir() and self._is_complete(path, spec, path.name)
            )
        )

    def _is_complete(self, path: Path, spec: InventoryModelSpec, revision: str) -> bool:
        marker = path / ".gliner-runner.json"
        if not marker.is_file():
            return False
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        identity_matches = (
            value.get("repository") == spec.repository
            and value.get("revision") == revision
            and bool(value.get("files"))
        )
        if not identity_matches:
            return False
        try:
            return all(
                _artifact_path(path, artifact["path"]).is_file()
                and _artifact_path(path, artifact["path"]).stat().st_size == artifact["size"]
                for artifact in value["files"]
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self._cache_path.is_file():
            return {}
        try:
            value = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_cache(self) -> None:
        self.provider_root.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._cache, indent=2, sort_keys=True) + "\n")
        temporary.replace(self._cache_path)


def profile_guidance(
    spec: InventoryModelSpec, profile: PrecisionProfile
) -> ProfileGuidance:
    profile_name = f"{profile.device}/{profile.precision}"
    drift: dict[str, tuple[float, str]] = {
        "cpu/fp32": (3.6e-7, "1e-7"),
        "mps/fp32": (2.9802322387695312e-6, "1e-6"),
        "mps/fp16": (0.0018429756164550781, "1e-3"),
    }
    observed = drift.get(profile_name) if spec.model_id == "decide-340m" else None
    return ProfileGuidance(
        estimated_minimum_memory_bytes=minimum_memory_bytes(
            spec.parameter_count, profile.precision
        ),
        memory_basis=(
            "Estimated model residency floor; post-load enforcement uses the larger "
            "of process RSS and accelerator allocation, never their sum."
        ),
        baseline_profile="cpu/fp32 direct official PyTorch oracle",
        observed_max_abs_score_drift=observed[0] if observed else None,
        drift_order_of_magnitude=observed[1] if observed else None,
        characterization_scope=(
            "Four-example Apple M5 Max characterization; not a universal accuracy guarantee."
            if observed
            else "No published numeric characterization for this model/profile."
        ),
    )


class DownloadJobManager:
    def __init__(self, inventory: ModelInventory) -> None:
        self._inventory = inventory
        self._jobs: dict[UUID, DownloadJob] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def create(self, model_id: str, request: DownloadRequest) -> DownloadJob:
        if not request.approved:
            raise ValueError("approved must be true; downloads never start implicitly")
        spec = self._inventory.spec(model_id)
        if spec is None:
            raise ValueError(f"unknown GLiNER2.5 model {model_id!r}")
        revision = request.revision or spec.pinned_revision
        if any(
            existing.model_id == spec.model_id
            and existing.revision == revision
            and existing.state in {DownloadState.QUEUED, DownloadState.RUNNING}
            for existing in self._jobs.values()
        ):
            raise ValueError(f"a download for {spec.model_id}@{revision} is already running")
        job = DownloadJob(
            job_id=uuid4(),
            model_id=spec.model_id,
            revision=revision,
            destination=request.destination,
            state=DownloadState.QUEUED,
        )
        self._jobs[job.job_id] = job
        task = asyncio.create_task(self._run(job), name=f"model-download-{job.job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def get(self, job_id: UUID) -> DownloadJob:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise KeyError(f"unknown download job {job_id}") from error

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, job: DownloadJob) -> None:
        self._jobs[job.job_id] = job.model_copy(update={"state": DownloadState.RUNNING})
        try:
            await self._inventory.download(job.model_id, job.revision, job.destination)
        except Exception as error:
            self._jobs[job.job_id] = job.model_copy(
                update={"state": DownloadState.FAILED, "error": str(error)}
            )
        else:
            self._jobs[job.job_id] = job.model_copy(
                update={"state": DownloadState.SUCCEEDED}
            )


def minimum_memory_bytes(parameter_count: int, precision: Precision) -> int:
    bytes_per_parameter = 4 if precision is Precision.FP32 else 2
    return int(parameter_count * bytes_per_parameter * 1.6 + 512 * 1024**2)


def _snapshot_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache" or path.name == ".gliner-runner.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise ModelIntegrityError("downloaded snapshot contains no model files")
    return files


def _verify_snapshot(root: Path) -> None:
    marker = root / ".gliner-runner.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        files = value["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelIntegrityError(f"invalid snapshot marker at {marker}") from error
    for artifact in files:
        try:
            path = _artifact_path(root, artifact["path"])
        except (KeyError, TypeError, ValueError) as error:
            raise ModelIntegrityError(f"invalid model artifact path in {marker}") from error
        if not path.is_file() or path.stat().st_size != artifact["size"]:
            raise ModelIntegrityError(f"model artifact size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact["sha256"]:
            raise ModelIntegrityError(f"model artifact checksum mismatch: {path}")


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _artifact_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError("artifact path must be a string")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise ValueError("artifact path escapes the snapshot")
    return resolved
