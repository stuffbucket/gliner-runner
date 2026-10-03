from __future__ import annotations

import asyncio
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from gliner_runner.backends.pytorch import PyTorchBackend
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    InferenceRequest,
    InferenceResponse,
    Precision,
)
from gliner_runner.errors import (
    ModelIntegrityError,
    ModelMemoryLimitError,
    RunnerClosedError,
    UnsupportedCapabilityError,
)
from gliner_runner.metrics import Metrics
from gliner_runner.model_inventory import ModelInventory, minimum_memory_bytes
from gliner_runner.model_manifest import ModelCatalog, ModelStore
from gliner_runner.resources import physical_memory_bytes, process_rss_bytes
from gliner_runner.scheduler import ModelWorker

WorkerKey: TypeAlias = tuple[BackendName, str, Precision, str]


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    queue_capacity: int = 256
    max_batch_size: int = 16
    batch_window_ms: float = 4.0
    manifest_directory: Path | None = None
    model_store: Path | None = None
    model_provider_directory: Path | None = None
    model_idle_ttl_seconds: float = 0
    memory_limit_bytes: int | None = None


class Runtime:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        registry: BackendRegistry | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.metrics = metrics or Metrics()
        self.registry = registry or _default_registry()
        self._catalog = (
            ModelCatalog(self.config.manifest_directory)
            if self.config.manifest_directory is not None
            else None
        )
        self._model_store = (
            ModelStore(self.config.model_store) if self.config.model_store is not None else None
        )
        self.inventory = (
            ModelInventory(self.config.model_provider_directory)
            if self.config.model_provider_directory is not None
            else None
        )
        self._worker_instance: ModelWorker | None = None
        self._worker_key: WorkerKey | None = None
        self._loaded_model: str | None = None
        self._loaded_revision: str | None = None
        self._active_requests = 0
        self._switching = False
        self._generation = 0
        self._last_activity = time.monotonic()
        self._idle_task: asyncio.Task[None] | None = None
        self._condition = asyncio.Condition()
        self._closed = False

    @property
    def accepting(self) -> bool:
        return not self._closed

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        worker = await self._acquire_worker(request)
        try:
            return await worker.submit(request)
        finally:
            await self._release_worker()

    def capabilities(self) -> list[BackendCapabilities]:
        return self.registry.capabilities()

    @property
    def memory_limit_bytes(self) -> int:
        return self.config.memory_limit_bytes or int(physical_memory_bytes() * 0.75)

    def loaded_state(self) -> dict[str, object] | None:
        if self._worker_key is None or self._loaded_model is None:
            return None
        backend, _key, precision, device = self._worker_key
        return {
            "model": self._loaded_model,
            "revision": self._loaded_revision,
            "backend": backend,
            "profile": {"device": device, "precision": precision},
            "active_requests": self._active_requests,
            "last_activity_monotonic": self._last_activity,
        }

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._switching = True
            self._condition.notify_all()
            while self._active_requests:
                await self._condition.wait()
            worker = self._clear_worker_locked()
            self._switching = False
            self._condition.notify_all()
        if worker is not None:
            await worker.close()

    async def _acquire_worker(self, request: InferenceRequest) -> ModelWorker:
        model_location, model_key, revision, parameter_count = await self._resolve_model(request)
        key = (request.backend, model_key, request.precision, self.config.device)
        async with self._condition:
            while self._switching and not self._closed:
                await self._condition.wait()
            if self._closed:
                raise RunnerClosedError("runtime is closed")
            if self._worker_key == key and self._worker_instance is not None:
                self._active_requests += 1
                self._cancel_idle_locked()
                return self._worker_instance
            self._switching = True
            while self._active_requests:
                await self._condition.wait()
            previous = self._clear_worker_locked()
        if previous is not None:
            try:
                await previous.close()
            except Exception:
                await self._load_failed()
                raise
        profile = f"{self.config.device}/{request.precision}"
        required = (
            minimum_memory_bytes(parameter_count, request.precision)
            if parameter_count is not None
            else 0
        )
        if required > self.memory_limit_bytes:
            await self._load_failed()
            raise ModelMemoryLimitError(
                model=request.model,
                profile=profile,
                limit_bytes=self.memory_limit_bytes,
                required_bytes=required,
            )
        worker: ModelWorker | None = None
        try:
            backend = self.registry.create(
                request,
                self.config.device,
                model_location=model_location,
            )
            worker = ModelWorker(
                backend,
                queue_capacity=self.config.queue_capacity,
                max_batch_size=self.config.max_batch_size,
                batch_window_ms=self.config.batch_window_ms,
                metrics=self.metrics,
            )
            await worker.start()
            observed = _memory_pressure_bytes(self.config.device)
            if observed > self.memory_limit_bytes:
                raise ModelMemoryLimitError(
                    model=request.model,
                    profile=profile,
                    limit_bytes=self.memory_limit_bytes,
                    required_bytes=required,
                    observed_bytes=observed,
                )
        except Exception:
            if worker is not None:
                await worker.close()
            await self._load_failed()
            raise
        async with self._condition:
            if self._closed:
                self._switching = False
                self._condition.notify_all()
                close_loaded = True
            else:
                self._worker_instance = worker
                self._worker_key = key
                self._loaded_model = request.model
                self._loaded_revision = revision
                self._active_requests = 1
                self._last_activity = time.monotonic()
                self._generation += 1
                self._switching = False
                self._condition.notify_all()
                close_loaded = False
        if close_loaded:
            await worker.close()
            raise RunnerClosedError("runtime is closed")
        return worker

    async def _resolve_model(
        self, request: InferenceRequest
    ) -> tuple[str, str, str | None, int | None]:
        model_location = request.model
        model_key = request.model
        revision: str | None = None
        parameter_count: int | None = None
        inventory_spec = self.inventory.spec(request.model) if self.inventory is not None else None
        if inventory_spec is not None and self.inventory is not None:
            resolved = await self.inventory.resolve_verified(request.model)
            model_location = str(resolved.path)
            model_key = f"{resolved.spec.model_id}@{resolved.revision}"
            revision = resolved.revision
            parameter_count = resolved.spec.parameter_count
        elif self._catalog is not None:
            manifest = self._catalog.resolve(request.model)
            if (
                manifest.backend != request.backend
                or request.precision not in manifest.precisions
                or request.operation not in manifest.operations
            ):
                raise UnsupportedCapabilityError(
                    "request is incompatible with the pinned model manifest; "
                    "no fallback was attempted"
                )
            if self._model_store is None:
                raise ModelIntegrityError("a model store is required with a manifest catalog")
            model_location = str(await self._model_store.verify(manifest))
            model_key = manifest.content_id
            revision = manifest.revision
        return model_location, model_key, revision, parameter_count

    async def _release_worker(self) -> None:
        async with self._condition:
            self._active_requests -= 1
            self._last_activity = time.monotonic()
            if self._active_requests == 0:
                self._condition.notify_all()
                self._schedule_idle_locked()

    async def _load_failed(self) -> None:
        async with self._condition:
            self._switching = False
            self._condition.notify_all()

    def _clear_worker_locked(self) -> ModelWorker | None:
        self._cancel_idle_locked()
        worker = self._worker_instance
        self._worker_instance = None
        self._worker_key = None
        self._loaded_model = None
        self._loaded_revision = None
        self._generation += 1
        return worker

    def _cancel_idle_locked(self) -> None:
        if self._idle_task is not None:
            if self._idle_task is not asyncio.current_task():
                self._idle_task.cancel()
            self._idle_task = None

    def _schedule_idle_locked(self) -> None:
        ttl = self.config.model_idle_ttl_seconds
        if ttl <= 0 or self._worker_instance is None or self._closed:
            return
        self._cancel_idle_locked()
        generation = self._generation
        self._idle_task = asyncio.create_task(
            self._evict_after_idle(ttl, generation),
            name="gliner-model-idle-eviction",
        )

    async def _evict_after_idle(self, ttl: float, generation: int) -> None:
        try:
            await asyncio.sleep(ttl)
            async with self._condition:
                if (
                    self._closed
                    or self._active_requests
                    or self._generation != generation
                    or self._worker_instance is None
                ):
                    return
                self._switching = True
                worker = self._clear_worker_locked()
            try:
                if worker is not None:
                    await worker.close()
            finally:
                async with self._condition:
                    self._switching = False
                    self._condition.notify_all()
        except asyncio.CancelledError:
            return


def _default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendName.PYTORCH,
        lambda model, precision, device: PyTorchBackend(model, precision, device),
    )
    return registry


def _memory_pressure_bytes(device: str) -> int:
    process_rss = process_rss_bytes()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return process_rss
    accelerator_bytes = 0
    if device == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "driver_allocated_memory"):
            accelerator_bytes = int(mps.driver_allocated_memory())
    elif device.startswith("cuda"):
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "memory_reserved"):
            accelerator_bytes = int(cuda.memory_reserved(device))
    return max(process_rss, accelerator_bytes)
