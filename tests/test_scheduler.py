from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from conftest import RecordingBackend
from gliner_runner.contracts import InferenceRequest, JsonValue
from gliner_runner.errors import QueueFullError
from gliner_runner.metrics import Metrics
from gliner_runner.scheduler import ModelWorker


async def test_compatible_requests_are_micro_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await worker.start()
    requests = [request_factory(text=f"text {index}") for index in range(3)]  # type: ignore[operator]

    responses = await asyncio.gather(*(worker.submit(request) for request in requests))
    await worker.close()

    assert [len(batch) for batch in backend.batches] == [3]
    assert [response.request_id for response in responses] == [
        request.request_id for request in requests
    ]
    assert backend.loaded and backend.closed


async def test_incompatible_schemas_are_never_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await worker.start()

    await asyncio.gather(
        worker.submit(request_factory(labels=("positive", "negative"))),  # type: ignore[operator]
        worker.submit(request_factory(labels=("urgent", "routine"))),  # type: ignore[operator]
    )
    await worker.close()

    assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_interleaved_compatible_backlog_is_rebatched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await worker.start()

    await asyncio.gather(
        worker.submit(request_factory(text="a1", labels=("a", "other"))),  # type: ignore[operator]
        worker.submit(request_factory(text="b1", labels=("b", "other"))),  # type: ignore[operator]
        worker.submit(request_factory(text="a2", labels=("a", "other"))),  # type: ignore[operator]
        worker.submit(request_factory(text="b2", labels=("b", "other"))),  # type: ignore[operator]
    )
    await worker.close()

    assert [len(batch) for batch in backend.batches] == [2, 2]


async def test_close_drains_incompatible_accepted_work(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, batch_window_ms=10)
    await worker.start()
    tasks = [
        asyncio.create_task(
            worker.submit(request_factory(text="a", labels=("a", "other")))  # type: ignore[operator]
        ),
        asyncio.create_task(
            worker.submit(request_factory(text="b", labels=("b", "other")))  # type: ignore[operator]
        ),
    ]
    await asyncio.sleep(0)

    await worker.close()
    responses = await asyncio.gather(*tasks)

    assert len(responses) == 2
    assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_cancellation_is_removed_before_inference(request_factory: object) -> None:
    backend = RecordingBackend()
    metrics = Metrics()
    worker = ModelWorker(backend, batch_window_ms=25, metrics=metrics)
    await worker.start()
    task = asyncio.create_task(worker.submit(request_factory()))  # type: ignore[operator]
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await worker.close()

    assert backend.batches == []
    assert metrics.snapshot()["requests_cancelled_total"] == 1


class BlockingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[JsonValue]:
        self.started.set()
        await self.release.wait()
        return await super().infer_batch(requests)


async def test_full_queue_applies_backpressure(request_factory: object) -> None:
    backend = BlockingBackend()
    worker = ModelWorker(backend, queue_capacity=1, batch_window_ms=0)
    await worker.start()
    first = asyncio.create_task(worker.submit(request_factory(text="first")))  # type: ignore[operator]
    await backend.started.wait()
    second = asyncio.create_task(worker.submit(request_factory(text="second")))  # type: ignore[operator]
    await asyncio.sleep(0)

    with pytest.raises(QueueFullError):
        await worker.submit(request_factory(text="third"))  # type: ignore[operator]

    backend.release.set()
    await asyncio.gather(first, second)
    await worker.close()
