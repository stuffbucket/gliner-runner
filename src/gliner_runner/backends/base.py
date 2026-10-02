from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from gliner_runner.contracts import BackendCapabilities, InferenceRequest, JsonValue


class InferenceBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    async def load(self) -> None: ...

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[JsonValue]: ...

    async def close(self) -> None: ...
