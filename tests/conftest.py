from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    BackendResult,
    ClassificationSchema,
    ClassificationTask,
    InferenceOperation,
    InferenceRequest,
    InferenceUsage,
    Precision,
)


@pytest.fixture
def request_factory() -> Any:
    def factory(
        text: str = "A useful message",
        *,
        labels: tuple[str, ...] = ("useful", "spam"),
        label_descriptions: dict[str, str | None] | None = None,
        model: str = "fastino/decide",
        precision: Precision = Precision.FP32,
    ) -> InferenceRequest:
        return InferenceRequest(
            model=model,
            backend=BackendName.PYTORCH,
            precision=precision,
            operation=InferenceOperation.CLASSIFY,
            text=text,
            schema=ClassificationSchema(
                tasks={
                    "label": ClassificationTask(
                        labels=labels,
                        label_descriptions=label_descriptions,
                    )
                }
            ),
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

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        self.batches.append(list(requests))
        outputs: list[BackendResult] = []
        for request in requests:
            assert isinstance(request.schema_, ClassificationSchema)
            task = next(iter(request.schema_.tasks.values()))
            outputs.append(
                BackendResult(
                    output={"label": task.labels[0]},
                    usage=InferenceUsage(
                        input_tokens=len(request.text.split()),
                        output_tokens=0,
                    ),
                )
            )
        return outputs

    async def close(self) -> None:
        self.closed = True
