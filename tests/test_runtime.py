from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from conftest import RecordingBackend, bounded
from gliner_runner import runtime as runtime_module
from gliner_runner.backends.pytorch import PyTorchBackend
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import BackendName, BackendResult, InferenceRequest, Precision
from gliner_runner.errors import ModelMemoryLimitError, RunnerClosedError
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

    await bounded(
        asyncio.gather(
            runtime.infer(request_factory(text="one")),  # type: ignore[operator]
            runtime.infer(request_factory(text="two")),  # type: ignore[operator]
        )
    )
    state = runtime.loaded_state()

    assert state is not None
    assert state["model"] == "fastino/decide"
    assert state["revision"] is None
    assert state["backend"] is BackendName.PYTORCH
    assert state["profile"] == {"device": "cpu", "precision": Precision.FP32}
    assert state["active_requests"] == 0
    assert isinstance(state["last_activity_monotonic"], float)
    await bounded(runtime.close())

    assert len(created) == 1
    assert len(created[0].batches) == 1
    assert runtime.loaded_state() is None
    assert runtime.accepting is False


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

    await bounded(
        runtime.infer(request_factory(model="decide-340m"))  # type: ignore[operator]
    )
    await bounded(
        runtime.infer(request_factory(model="decide-multilingual"))  # type: ignore[operator]
    )

    assert len(created) == 2
    assert created[0].closed
    assert not created[1].closed
    assert runtime.loaded_state()["model"] == "decide-multilingual"  # type: ignore[index]
    await bounded(runtime.close())


async def test_runtime_passes_resolved_model_location_to_backend(
    request_factory: object, tmp_path: Path
) -> None:
    spec = KNOWN_MODELS[0]
    _install_marker(tmp_path, spec.repository, spec.pinned_revision, spec.model_id)
    received: list[str] = []
    registry = BackendRegistry()

    def factory(model: str, _precision: object, _device: str) -> RecordingBackend:
        received.append(model)
        return RecordingBackend()

    registry.register(BackendName.PYTORCH, factory)
    runtime = Runtime(
        RuntimeConfig(model_provider_directory=tmp_path, batch_window_ms=0),
        registry=registry,
    )

    await bounded(
        runtime.infer(request_factory(model=spec.model_id))  # type: ignore[operator]
    )

    expected = tmp_path / spec.repository.replace("/", "--") / spec.pinned_revision
    assert received == [str(expected)]
    assert runtime.loaded_state()["revision"] == spec.pinned_revision  # type: ignore[index]
    await bounded(runtime.close())


async def test_runtime_evicts_model_after_idle_ttl(request_factory: object) -> None:
    backend = RecordingBackend()
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: backend)
    runtime = Runtime(
        RuntimeConfig(batch_window_ms=0, model_idle_ttl_seconds=0.01),
        registry=registry,
    )

    await bounded(runtime.infer(request_factory()))  # type: ignore[operator]
    await asyncio.sleep(0.03)

    assert backend.closed
    assert runtime.loaded_state() is None
    await bounded(runtime.close())


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

    with pytest.raises(ModelMemoryLimitError, match="estimated minimum") as captured:
        await bounded(
            runtime.infer(  # type: ignore[operator]
                request_factory(model=spec.model_id, precision=Precision.FP32)
            )
        )

    assert captured.value.model == spec.model_id
    assert captured.value.profile == "cpu/fp32"
    assert captured.value.limit_bytes == 512 * 1024**2
    assert captured.value.required_bytes > captured.value.limit_bytes
    assert captured.value.observed_bytes is None
    assert runtime.loaded_state() is None
    await bounded(runtime.close())


async def test_runtime_unloads_model_when_observed_memory_exceeds_limit(
    request_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = RecordingBackend()
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: backend)
    limit = 512 * 1024**2

    def memory_pressure(device: str) -> int:
        assert device == "cpu"
        return limit + 1

    monkeypatch.setattr(runtime_module, "_memory_pressure_bytes", memory_pressure)
    runtime = Runtime(
        RuntimeConfig(batch_window_ms=0, memory_limit_bytes=limit),
        registry=registry,
    )

    with pytest.raises(ModelMemoryLimitError) as captured:
        await bounded(runtime.infer(request_factory()))  # type: ignore[operator]

    assert captured.value.observed_bytes == limit + 1
    assert captured.value.model == "fastino/decide"
    assert captured.value.profile == "cpu/fp32"
    assert captured.value.limit_bytes == limit
    assert captured.value.required_bytes == 0
    assert backend.closed
    assert runtime.loaded_state() is None
    await bounded(runtime.close())


async def test_memory_limit_is_inclusive_at_exact_boundary(
    request_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RecordingBackend()
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: backend)
    limit = 512 * 1024**2

    def memory_pressure(device: str) -> int:
        assert device == "cpu"
        return limit

    monkeypatch.setattr(runtime_module, "_memory_pressure_bytes", memory_pressure)
    runtime = Runtime(
        RuntimeConfig(batch_window_ms=0, memory_limit_bytes=limit),
        registry=registry,
    )

    response = await bounded(runtime.infer(request_factory()))  # type: ignore[operator]

    assert response.output == {"label": "useful"}
    assert not backend.closed
    await bounded(runtime.close())


class LoadFailureBackend(RecordingBackend):
    async def load(self) -> None:
        raise RuntimeError("load failed")


async def test_runtime_recovers_after_load_failure(request_factory: object) -> None:
    created: list[RecordingBackend] = []
    registry = BackendRegistry()

    def factory(_model: str, _precision: object, _device: str) -> RecordingBackend:
        backend = LoadFailureBackend() if not created else RecordingBackend()
        created.append(backend)
        return backend

    registry.register(BackendName.PYTORCH, factory)
    runtime = Runtime(RuntimeConfig(batch_window_ms=0), registry=registry)

    with pytest.raises(RuntimeError, match="load failed"):
        await bounded(runtime.infer(request_factory()))  # type: ignore[operator]
    response = await bounded(
        runtime.infer(request_factory(text="recovered"))  # type: ignore[operator]
    )

    assert response.output == {"label": "useful"}
    assert len(created) == 2
    assert runtime.loaded_state() is not None
    await bounded(runtime.close())


class BlockingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        self.started.set()
        await self.release.wait()
        return await super().infer_batch(requests)


async def test_runtime_close_waits_for_active_request(request_factory: object) -> None:
    backend = BlockingBackend()
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: backend)
    runtime = Runtime(RuntimeConfig(batch_window_ms=0), registry=registry)
    inference = asyncio.create_task(runtime.infer(request_factory()))  # type: ignore[operator]
    await bounded(backend.started.wait())

    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not closing.done()
    backend.release.set()
    await bounded(inference)
    await bounded(closing)

    assert backend.closed
    assert runtime.loaded_state() is None
    with pytest.raises(RunnerClosedError, match="runtime is closed"):
        await bounded(runtime.infer(request_factory()))  # type: ignore[operator]


async def test_concurrent_model_switches_are_serialized(
    request_factory: object,
) -> None:
    created: list[RecordingBackend] = []
    registry = BackendRegistry()

    def factory(_model: str, _precision: object, _device: str) -> RecordingBackend:
        backend = RecordingBackend()
        created.append(backend)
        return backend

    registry.register(BackendName.PYTORCH, factory)
    runtime = Runtime(RuntimeConfig(batch_window_ms=0), registry=registry)

    await bounded(
        asyncio.gather(
            runtime.infer(request_factory(model="first/model")),  # type: ignore[operator]
            runtime.infer(request_factory(model="second/model")),  # type: ignore[operator]
        )
    )

    assert len(created) == 2
    assert created[0].closed
    assert runtime.loaded_state() is not None
    assert runtime.loaded_state()["model"] in {"first/model", "second/model"}  # type: ignore[index]
    await bounded(runtime.close())


def test_runtime_uses_explicit_or_default_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "physical_memory_bytes", lambda: 1_000)

    assert Runtime(RuntimeConfig(memory_limit_bytes=123)).memory_limit_bytes == 123
    assert Runtime().memory_limit_bytes == 750


def test_runtime_preserves_explicit_dependencies() -> None:
    registry = BackendRegistry()
    metrics = runtime_module.Metrics()
    config = RuntimeConfig(batch_window_ms=9)

    runtime = Runtime(config, registry=registry, metrics=metrics)

    assert runtime.config is config
    assert runtime.registry is registry
    assert runtime.metrics is metrics


def test_runtime_initializes_configured_resources(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    models = tmp_path / "models"
    providers = tmp_path / "providers"
    runtime = Runtime(
        RuntimeConfig(
            manifest_directory=manifests,
            model_store=models,
            model_provider_directory=providers,
        )
    )

    assert runtime._catalog is not None
    assert runtime._model_store is not None
    assert runtime._model_store.root == models
    assert runtime.inventory is not None
    assert runtime.inventory.provider_root == providers.resolve()
    assert runtime._worker_instance is None
    assert runtime._worker_key is None
    assert runtime._loaded_model is None
    assert runtime._loaded_revision is None
    assert runtime._active_requests == 0
    assert runtime._switching is False
    assert runtime._generation == 0
    assert isinstance(runtime._last_activity, float)
    assert runtime._idle_task is None
    assert runtime._closed is False


def test_default_registry_builds_requested_pytorch_backend(request_factory: Any) -> None:
    request = request_factory(model="model/path", precision=Precision.FP32)

    backend = runtime_module._default_registry().create(request, "cpu")

    assert isinstance(backend, PyTorchBackend)
    assert backend._model_id == "model/path"
    assert backend._precision is Precision.FP32
    assert backend._device == "cpu"


def test_memory_pressure_uses_process_rss_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "process_rss_bytes", lambda: 111)

    def missing_torch(name: str) -> ModuleType:
        assert name == "torch"
        raise ImportError

    monkeypatch.setattr(runtime_module.importlib, "import_module", missing_torch)

    assert runtime_module._memory_pressure_bytes("cpu") == 111


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_memory_pressure_handles_missing_accelerator_api(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setattr(runtime_module, "process_rss_bytes", lambda: 0)
    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    assert runtime_module._memory_pressure_bytes(device) == 0


@pytest.mark.parametrize(
    ("device", "torch", "expected"),
    [
        (
            "mps",
            SimpleNamespace(
                mps=SimpleNamespace(driver_allocated_memory=lambda: 222),
            ),
            222,
        ),
        (
            "cuda:1",
            SimpleNamespace(
                cuda=SimpleNamespace(
                    memory_reserved=lambda device: 333 if device == "cuda:1" else 0
                ),
            ),
            333,
        ),
        ("cpu", SimpleNamespace(), 111),
    ],
)
def test_memory_pressure_uses_larger_accelerator_value(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    torch: Any,
    expected: int,
) -> None:
    monkeypatch.setattr(runtime_module, "process_rss_bytes", lambda: 111)
    monkeypatch.setattr(runtime_module.importlib, "import_module", lambda _name: torch)

    assert runtime_module._memory_pressure_bytes(device) == expected


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
