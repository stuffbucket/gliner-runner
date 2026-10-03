from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence

import pytest

from conftest import RecordingBackend
from gliner_runner.contracts import (
    BackendResult,
    InferenceOptions,
    InferenceRequest,
    InferenceUsage,
)
from gliner_runner.errors import QueueFullError, RunnerClosedError
from gliner_runner.metrics import Metrics
from gliner_runner.scheduler import ModelWorker, _Work


def test_scheduler_rejects_invalid_configuration() -> None:
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="invalid scheduler configuration"):
        ModelWorker(backend, queue_capacity=0)
    with pytest.raises(ValueError, match="invalid scheduler configuration"):
        ModelWorker(backend, max_batch_size=0)
    with pytest.raises(ValueError, match="invalid scheduler configuration"):
        ModelWorker(backend, batch_window_ms=-0.1)


def test_scheduler_defaults_and_minimum_batch_size() -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend)

    assert worker._queue.maxsize == 256
    assert worker._max_batch_size == 16
    assert worker._batch_window == 0.004
    assert worker._accepting is False
    assert worker._task is None
    assert worker._backlog == deque()
    assert isinstance(worker._metrics, Metrics)

    minimum = ModelWorker(backend, max_batch_size=1)
    assert minimum._max_batch_size == 1


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
    assert [response.usage.input_tokens for response in responses] == [2, 2, 2]
    assert [response.usage.output_tokens for response in responses] == [0, 0, 0]
    encoded = responses[0].model_dump(mode="json", by_alias=True)
    assert encoded["usage"] == {"inputTokens": 2, "outputTokens": 0}
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


async def test_incompatible_options_are_never_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await worker.start()
    first = request_factory(text="first")  # type: ignore[operator]
    second = request_factory(text="second").model_copy(  # type: ignore[operator]
        update={"options": InferenceOptions(threshold=0.9)}
    )

    await asyncio.gather(worker.submit(first), worker.submit(second))
    await worker.close()

    assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_equivalent_mapping_order_is_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await worker.start()

    await asyncio.gather(
        worker.submit(
            request_factory(  # type: ignore[operator]
                text="first",
                label_descriptions={"useful": "keep", "spam": "discard"},
            )
        ),
        worker.submit(
            request_factory(  # type: ignore[operator]
                text="second",
                label_descriptions={"spam": "discard", "useful": "keep"},
            )
        ),
    )
    await worker.close()

    assert [len(batch) for batch in backend.batches] == [2]


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
    assert metrics.snapshot() == {
        "requests_submitted_total": 1,
        "requests_cancelled_total": 1,
        "queue_depth": 0.0,
    }


class BlockingBackend(RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        self.started.set()
        await self.release.wait()
        return await super().infer_batch(requests)


async def test_full_queue_applies_backpressure(request_factory: object) -> None:
    backend = BlockingBackend()
    metrics = Metrics()
    worker = ModelWorker(
        backend,
        queue_capacity=1,
        batch_window_ms=0,
        metrics=metrics,
    )
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

    assert metrics.snapshot() == {
        "requests_submitted_total": 2,
        "requests_rejected_total": 1,
        "batches_total": 2,
        "last_batch_size": 1.0,
        "requests_completed_total": 2,
        "queue_depth": 0.0,
    }


async def test_start_is_idempotent_and_closed_worker_rejects_work(
    request_factory: object,
) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, batch_window_ms=0)

    with pytest.raises(RunnerClosedError, match="not accepting"):
        await worker.submit(request_factory())  # type: ignore[operator]
    await worker.start()
    task = worker._task
    await worker.start()
    assert worker._task is task
    await worker.close()
    assert worker.accepting is False
    with pytest.raises(RunnerClosedError, match="not accepting"):
        await worker.submit(request_factory())  # type: ignore[operator]


async def test_close_before_start_does_not_load_or_close_backend() -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend)

    await worker.close()

    assert worker.accepting is False
    assert backend.loaded is False
    assert backend.closed is False


class WrongCountBackend(RecordingBackend):
    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        return []


class FailingBackend(RecordingBackend):
    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        raise RuntimeError("inference failed")


@pytest.mark.parametrize("backend", [WrongCountBackend(), FailingBackend()])
async def test_backend_failures_reach_submitter_and_metrics(
    backend: RecordingBackend,
    request_factory: object,
) -> None:
    metrics = Metrics()
    worker = ModelWorker(backend, batch_window_ms=0, metrics=metrics)
    await worker.start()

    with pytest.raises(
        RuntimeError,
        match=(
            "backend returned the wrong number of results"
            if isinstance(backend, WrongCountBackend)
            else "inference failed"
        ),
    ):
        await worker.submit(request_factory())  # type: ignore[operator]
    await worker.close()

    assert metrics.snapshot() == {
        "requests_submitted_total": 1,
        "batches_total": 1,
        "last_batch_size": 1.0,
        "batch_failures_total": 1,
        "queue_depth": 0.0,
    }


async def test_response_preserves_identity_usage_and_exact_timings(
    request_factory: object,
) -> None:
    worker = ModelWorker(RecordingBackend())
    request = request_factory()  # type: ignore[operator]
    work = _Work(
        request=request,
        future=asyncio.get_running_loop().create_future(),
        enqueued_at=10.0,
    )
    result = BackendResult(
        output={"label": "useful"},
        usage=InferenceUsage(input_tokens=3, output_tokens=0),
    )

    response = worker._response(work, result, started=10.25, finished=10.75)

    assert response.request_id == request.request_id
    assert response.model == request.model
    assert response.backend == request.backend
    assert response.precision == request.precision
    assert response.output == {"label": "useful"}
    assert response.usage == result.usage
    assert response.timing.queue_ms == 250.0
    assert response.timing.inference_ms == 500.0
