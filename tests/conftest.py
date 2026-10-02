from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    ClassificationSchema,
    ClassificationTask,
    InferenceOperation,
    InferenceRequest,
    JsonValue,
    Precision,
)


@pytest.fixture
def request_factory() -> Any:
    def factory(
        text: str = "A useful message",
        *,
        labels: tuple[str, ...] = ("useful", "spam"),
        model: str = "fastino/decide",
        precision: Precision = Precision.FP32,
    ) -> InferenceRequest:
        return InferenceRequest(
            model=model,
            backend=BackendName.PYTORCH,
            precision=precision,
            operation=InferenceOperation.CLASSIFY,
            text=text,
            schema=ClassificationSchema(tasks={"label": ClassificationTask(labels=labels)}),
        )

    return factory


class RecordingBackend:
    def __init__(self) -> None:
        self.batches: list[list[InferenceRequest]] = []
        self.loaded = False
        self.closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=BackendName.PYTORCH,
            operations=frozenset({InferenceOperation.CLASSIFY}),
            precisions=frozenset({Precision.FP32}),
            devices=frozenset({"cpu"}),
            dynamic_batching=True,
        )

    async def load(self) -> None:
        self.loaded = True

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[JsonValue]:
        self.batches.append(list(requests))
        outputs: list[JsonValue] = []
        for request in requests:
            assert isinstance(request.schema_, ClassificationSchema)
            task = next(iter(request.schema_.tasks.values()))
            outputs.append({"label": task.labels[0]})
        return outputs

    async def close(self) -> None:
        self.closed = True
