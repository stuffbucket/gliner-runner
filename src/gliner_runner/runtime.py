from __future__ import annotations

import asyncio
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
from gliner_runner.errors import ModelIntegrityError, UnsupportedCapabilityError
from gliner_runner.metrics import Metrics
from gliner_runner.model_manifest import ModelCatalog, ModelStore
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
        self._workers: dict[WorkerKey, ModelWorker] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def accepting(self) -> bool:
        return not self._closed

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        worker = await self._worker(request)
        return await worker.submit(request)

    def capabilities(self) -> list[BackendCapabilities]:
        return self.registry.capabilities()

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        await asyncio.gather(*(worker.close() for worker in workers))

    async def _worker(self, request: InferenceRequest) -> ModelWorker:
        if self._closed:
            raise RuntimeError("runtime is closed")
        model_location = request.model
        model_key = request.model
        if self._catalog is not None:
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
        key = (request.backend, model_key, request.precision, self.config.device)
        async with self._lock:
            existing = self._workers.get(key)
            if existing is not None:
                return existing
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
            self._workers[key] = worker
            return worker


def _default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendName.PYTORCH,
        lambda model, precision, device: PyTorchBackend(model, precision, device),
    )
    return registry
