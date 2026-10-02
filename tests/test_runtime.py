from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from conftest import RecordingBackend
from gliner_runner import runtime as runtime_module
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import BackendName, Precision
from gliner_runner.errors import ModelMemoryLimitError
from gliner_runner.model_inventory import KNOWN_MODELS
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


async def test_runtime_replaces_the_single_resident_model(
    request_factory: object, tmp_path: Path
) -> None:
    created: list[RecordingBackend] = []
    registry = BackendRegistry()

    def factory(_model: str, _precision: object, _device: str) -> RecordingBackend:
        backend = RecordingBackend()
        created.append(backend)
        return backend

    registry.register(BackendName.PYTORCH, factory)
    for spec in KNOWN_MODELS[:2]:
        _install_marker(tmp_path, spec.repository, spec.pinned_revision, spec.model_id)
    runtime = Runtime(
        RuntimeConfig(model_provider_directory=tmp_path, batch_window_ms=0),
        registry=registry,
    )

    await runtime.infer(request_factory(model="decide-340m"))  # type: ignore[operator]
    await runtime.infer(request_factory(model="decide-multilingual"))  # type: ignore[operator]

    assert len(created) == 2
    assert created[0].closed
    assert not created[1].closed
    assert runtime.loaded_state()["model"] == "decide-multilingual"  # type: ignore[index]
    await runtime.close()


async def test_runtime_evicts_model_after_idle_ttl(request_factory: object) -> None:
    backend = RecordingBackend()
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: backend)
    runtime = Runtime(
        RuntimeConfig(batch_window_ms=0, model_idle_ttl_seconds=0.01),
        registry=registry,
    )

    await runtime.infer(request_factory())  # type: ignore[operator]
    await asyncio.sleep(0.03)

    assert backend.closed
    assert runtime.loaded_state() is None
    await runtime.close()


async def test_runtime_rejects_known_model_below_memory_floor(
    request_factory: object, tmp_path: Path
) -> None:
    spec = KNOWN_MODELS[0]
    _install_marker(tmp_path, spec.repository, spec.pinned_revision, spec.model_id)
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: RecordingBackend())
    runtime = Runtime(
        RuntimeConfig(
            model_provider_directory=tmp_path,
            memory_limit_bytes=512 * 1024**2,
        ),
        registry=registry,
    )

    with pytest.raises(ModelMemoryLimitError, match="estimated minimum"):
        await runtime.infer(  # type: ignore[operator]
            request_factory(model=spec.model_id, precision=Precision.FP32)
        )

    assert runtime.loaded_state() is None
    await runtime.close()


async def test_runtime_unloads_model_when_observed_memory_exceeds_limit(
    request_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = RecordingBackend()
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: backend)
    limit = 512 * 1024**2
    monkeypatch.setattr(runtime_module, "_memory_pressure_bytes", lambda _device: limit + 1)
    runtime = Runtime(
        RuntimeConfig(batch_window_ms=0, memory_limit_bytes=limit),
        registry=registry,
    )

    with pytest.raises(ModelMemoryLimitError) as captured:
        await runtime.infer(request_factory())  # type: ignore[operator]

    assert captured.value.observed_bytes == limit + 1
    assert backend.closed
    assert runtime.loaded_state() is None
    await runtime.close()


def _install_marker(root: Path, repository: str, revision: str, model_id: str) -> None:
    target = root / repository.replace("/", "--") / revision
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    (target / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "repository": repository,
                "revision": revision,
                "files": [
                    {
                        "path": "config.json",
                        "size": 2,
                        "sha256": digest,
                    }
                ],
            }
        )
    )
