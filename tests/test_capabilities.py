from __future__ import annotations

import pytest

from conftest import RecordingBackend
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import BackendName, Precision
from gliner_runner.errors import BackendUnavailableError, UnsupportedCapabilityError


def test_registry_selects_only_explicit_backend(request_factory: object) -> None:
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda _model, _precision, _device: RecordingBackend())
    request = request_factory()  # type: ignore[operator]

    backend = registry.create(request, "cpu")

    assert backend.capabilities.backend is BackendName.PYTORCH


def test_registry_never_falls_back_when_backend_missing(request_factory: object) -> None:
    registry = BackendRegistry()
    request = request_factory().model_copy(  # type: ignore[operator]
        update={"backend": BackendName.MLX}
    )

    with pytest.raises(BackendUnavailableError, match="no fallback was attempted"):
        registry.create(request, "cpu")


def test_registry_rejects_unadvertised_precision(request_factory: object) -> None:
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda _model, _precision, _device: RecordingBackend())
    request = request_factory(precision=Precision.INT8)  # type: ignore[operator]

    with pytest.raises(UnsupportedCapabilityError, match="no fallback was attempted"):
        registry.create(request, "cpu")
