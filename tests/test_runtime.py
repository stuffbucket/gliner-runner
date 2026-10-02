from __future__ import annotations

import asyncio

from conftest import RecordingBackend
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import BackendName
from gliner_runner.runtime import Runtime, RuntimeConfig


async def test_runtime_has_one_model_owner_per_device(request_factory: object) -> None:
    created: list[RecordingBackend] = []
    registry = BackendRegistry()

    def factory(_model: str, _precision: object, _device: str) -> RecordingBackend:
        backend = RecordingBackend()
        created.append(backend)
        return backend

    registry.register(BackendName.PYTORCH, factory)
    runtime = Runtime(RuntimeConfig(batch_window_ms=5), registry=registry)

    await asyncio.gather(
        runtime.infer(request_factory(text="one")),  # type: ignore[operator]
        runtime.infer(request_factory(text="two")),  # type: ignore[operator]
    )
    await runtime.close()

    assert len(created) == 1
    assert len(created[0].batches) == 1
