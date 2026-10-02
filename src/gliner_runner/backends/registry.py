from __future__ import annotations

from collections.abc import Callable

from gliner_runner.backends.base import InferenceBackend
from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    InferenceRequest,
    Precision,
)
from gliner_runner.errors import BackendUnavailableError, UnsupportedCapabilityError

BackendFactory = Callable[[str, Precision, str], InferenceBackend]


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[BackendName, BackendFactory] = {}

    def register(self, name: BackendName, factory: BackendFactory) -> None:
        if name in self._factories:
            raise ValueError(f"backend already registered: {name}")
        self._factories[name] = factory

    def create(
        self,
        request: InferenceRequest,
        device: str,
        *,
        model_location: str | None = None,
    ) -> InferenceBackend:
        factory = self._factories.get(request.backend)
        if factory is None:
            raise BackendUnavailableError(
                f"backend {request.backend} is not installed; no fallback was attempted"
            )
        backend = factory(model_location or request.model, request.precision, device)
        if not backend.capabilities.supports(request, device):
            raise UnsupportedCapabilityError(
                f"backend {request.backend} does not support model={request.model}, "
                f"operation={request.operation}, precision={request.precision}, device={device}; "
                "no fallback was attempted"
            )
        return backend

    def capabilities(self) -> list[BackendCapabilities]:
        return [
            factory("", Precision.FP32, "cpu").capabilities for factory in self._factories.values()
        ]
