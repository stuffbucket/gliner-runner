from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence

import pytest

from conftest import RecordingBackend, bounded
from gliner_runner.contracts import (
    BackendResult,
    InferenceOptions,
    InferenceRequest,
    InferenceResponse,
    InferenceUsage,
)
from gliner_runner.errors import QueueFullError, RunnerClosedError
from gliner_runner.metrics import Metrics
from gliner_runner.scheduler import _STOP, ModelWorker, _Work


def test_scheduler_rejects_invalid_configuration() -> None:
    backend = RecordingBackend()

    with pytest.raises(ValueError) as captured:
        ModelWorker(backend, queue_capacity=0)
    assert str(captured.value) == "invalid scheduler configuration"
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
    metrics = Metrics()
    worker = ModelWorker(
        backend,
        max_batch_size=8,
        batch_window_ms=10,
        metrics=metrics,
    )
    await bounded(worker.start())
    assert worker._task is not None
    assert worker._task.get_name() == "gliner-model-worker"
    requests = [request_factory(text=f"text {index}") for index in range(3)]  # type: ignore[operator]

    responses = await bounded(
        asyncio.gather(*(worker.submit(request) for request in requests))
    )
    await bounded(worker.close())

    assert [len(batch) for batch in backend.batches] == [3]
    assert [response.request_id for response in responses] == [
        request.request_id for request in requests
    ]
    assert [response.usage.input_tokens for response in responses] == [2, 2, 2]
    assert [response.usage.output_tokens for response in responses] == [0, 0, 0]
    encoded = responses[0].model_dump(mode="json", by_alias=True)
    assert encoded["usage"] == {"inputTokens": 2, "outputTokens": 0}
    assert backend.loaded and backend.closed
    assert worker._task is None
    assert metrics.snapshot()["requests_completed_total"] == 3


async def test_incompatible_schemas_are_never_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await bounded(worker.start())

    await bounded(
        asyncio.gather(
            worker.submit(request_factory(labels=("positive", "negative"))),  # type: ignore[operator]
            worker.submit(request_factory(labels=("urgent", "routine"))),  # type: ignore[operator]
        )
    )
    await bounded(worker.close())

    assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_incompatible_options_are_never_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await bounded(worker.start())
    first = request_factory(text="first")  # type: ignore[operator]
    second = request_factory(text="second").model_copy(  # type: ignore[operator]
        update={"options": InferenceOptions(threshold=0.9)}
    )

    await bounded(asyncio.gather(worker.submit(first), worker.submit(second)))
    await bounded(worker.close())

    assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_equivalent_mapping_order_is_batched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await bounded(worker.start())

    await bounded(
        asyncio.gather(
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
    )
    await bounded(worker.close())

    assert [len(batch) for batch in backend.batches] == [2]


async def test_interleaved_compatible_backlog_is_rebatched(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, max_batch_size=8, batch_window_ms=10)
    await bounded(worker.start())

    await bounded(
        asyncio.gather(
            worker.submit(request_factory(text="a1", labels=("a", "other"))),  # type: ignore[operator]
            worker.submit(request_factory(text="b1", labels=("b", "other"))),  # type: ignore[operator]
            worker.submit(request_factory(text="a2", labels=("a", "other"))),  # type: ignore[operator]
            worker.submit(request_factory(text="b2", labels=("b", "other"))),  # type: ignore[operator]
        )
    )
    await bounded(worker.close())

    assert [len(batch) for batch in backend.batches] == [2, 2]


async def test_close_drains_incompatible_accepted_work(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, batch_window_ms=10)
    await bounded(worker.start())
    tasks = [
        asyncio.create_task(
            worker.submit(request_factory(text="a", labels=("a", "other")))  # type: ignore[operator]
        ),
        asyncio.create_task(
            worker.submit(request_factory(text="b", labels=("b", "other")))  # type: ignore[operator]
        ),
    ]
    await asyncio.sleep(0)

    await bounded(worker.close())
    responses = await bounded(asyncio.gather(*tasks))

    assert len(responses) == 2
    assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_stop_barrier_preserves_later_backlog_work(request_factory: object) -> None:
    worker = ModelWorker(RecordingBackend(), max_batch_size=8, batch_window_ms=0)
    loop = asyncio.get_running_loop()
    first = _Work(
        request=request_factory(text="first"),  # type: ignore[operator]
        future=loop.create_future(),
        enqueued_at=0,
    )
    later = _Work(
        request=request_factory(text="later"),  # type: ignore[operator]
        future=loop.create_future(),
        enqueued_at=0,
    )
    worker._backlog.extend([_STOP, later])

    batch = await bounded(worker._collect(first))

    assert batch == [first]
    assert list(worker._backlog) == [later, _STOP]


async def test_worker_exit_fails_pending_backlog(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, batch_window_ms=0)
    future: asyncio.Future[InferenceResponse] = (
        asyncio.get_running_loop().create_future()
    )
    pending = _Work(
        request=request_factory(),  # type: ignore[operator]
        future=future,
        enqueued_at=0,
    )
    worker._backlog.extend([_STOP, pending])

    await bounded(worker.start())

    with pytest.raises(RunnerClosedError) as captured:
        await bounded(future)
    assert str(captured.value) == "model worker closed"
    await bounded(worker.close())
    assert backend.closed


async def test_cancellation_is_removed_before_inference(request_factory: object) -> None:
    backend = RecordingBackend()
    metrics = Metrics()
    worker = ModelWorker(backend, batch_window_ms=25, metrics=metrics)
    await bounded(worker.start())
    task = asyncio.create_task(worker.submit(request_factory()))  # type: ignore[operator]
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await bounded(task)
    await bounded(worker.close())

    assert backend.batches == []
    assert metrics.snapshot() == {
        "requests_submitted_total": 1,
        "requests_cancelled_total": 1,
        "queue_depth": 0.0,
    }


async def test_cancelled_batch_does_not_stop_worker(request_factory: object) -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend, batch_window_ms=25)
    await bounded(worker.start())
    cancelled = asyncio.create_task(worker.submit(request_factory(text="cancel")))  # type: ignore[operator]
    await asyncio.sleep(0)
    cancelled.cancel()

    with pytest.raises(asyncio.CancelledError):
        await bounded(cancelled)
    await asyncio.sleep(0.05)
    response = await bounded(worker.submit(request_factory(text="next")))  # type: ignore[operator]

    assert response.output == {"label": "useful"}
    await bounded(worker.close())


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
    await bounded(worker.start())
    first = asyncio.create_task(worker.submit(request_factory(text="first")))  # type: ignore[operator]
    await bounded(backend.started.wait())
    second = asyncio.create_task(worker.submit(request_factory(text="second")))  # type: ignore[operator]
    await asyncio.sleep(0)

    with pytest.raises(QueueFullError) as captured:
        await bounded(worker.submit(request_factory(text="third")))  # type: ignore[operator]
    assert str(captured.value) == "model worker queue is full"

    backend.release.set()
    await bounded(asyncio.gather(first, second))
    await bounded(worker.close())

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

    with pytest.raises(RunnerClosedError) as before_start:
        await bounded(worker.submit(request_factory()))  # type: ignore[operator]
    assert str(before_start.value) == "model worker is not accepting requests"
    await bounded(worker.start())
    task = worker._task
    await bounded(worker.start())
    assert worker._task is task
    await bounded(worker.close())
    assert worker.accepting is False
    with pytest.raises(RunnerClosedError) as after_close:
        await bounded(worker.submit(request_factory()))  # type: ignore[operator]
    assert str(after_close.value) == "model worker is not accepting requests"


async def test_close_before_start_does_not_load_or_close_backend() -> None:
    backend = RecordingBackend()
    worker = ModelWorker(backend)

    await bounded(worker.close())

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
    await bounded(worker.start())

    with pytest.raises(RuntimeError) as captured:
        await bounded(worker.submit(request_factory()))  # type: ignore[operator]
    expected = (
        "backend returned the wrong number of results"
        if isinstance(backend, WrongCountBackend)
        else "inference failed"
    )
    assert str(captured.value) == expected
    await bounded(worker.close())

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

    clamped = worker._response(work, result, started=9.0, finished=8.0)
    assert clamped.timing.queue_ms == 0
    assert clamped.timing.inference_ms == 0


async def test_collect_enforces_batch_limit_for_backlog_and_queue(
    request_factory: object,
) -> None:
    worker = ModelWorker(RecordingBackend(), max_batch_size=1, batch_window_ms=0)
    loop = asyncio.get_running_loop()

    def work(text: str) -> _Work:
        return _Work(
            request=request_factory(text=text),  # type: ignore[operator]
            future=loop.create_future(),
            enqueued_at=0,
        )

    first = work("first")
    backlog = work("backlog")
    queued = work("queued")
    worker._backlog.append(backlog)
    worker._queue.put_nowait(queued)

    assert await bounded(worker._collect(first)) == [first]
    assert list(worker._backlog) == [backlog]
    assert worker._queue.get_nowait() is queued

    queue_worker = ModelWorker(
        RecordingBackend(),
        max_batch_size=2,
        batch_window_ms=0,
    )
    queue_worker._queue.put_nowait(queued)
    assert await bounded(queue_worker._collect(first)) == [first, queued]


async def test_fail_pending_drains_queue(request_factory: object) -> None:
    worker = ModelWorker(RecordingBackend())
    future: asyncio.Future[InferenceResponse] = (
        asyncio.get_running_loop().create_future()
    )
    worker._queue.put_nowait(
        _Work(
            request=request_factory(),  # type: ignore[operator]
            future=future,
            enqueued_at=0,
        )
    )

    worker._fail_pending(RunnerClosedError("exact"))

    assert worker._queue.empty()
    assert isinstance(future.exception(), RunnerClosedError)
    assert str(future.exception()) == "exact"


async def test_queue_depth_includes_backlog(request_factory: object) -> None:
    metrics = Metrics()
    worker = ModelWorker(RecordingBackend(), metrics=metrics)
    loop = asyncio.get_running_loop()

    def work() -> _Work:
        return _Work(
            request=request_factory(),  # type: ignore[operator]
            future=loop.create_future(),
            enqueued_at=0,
        )

    worker._queue.put_nowait(work())
    worker._backlog.append(work())
    worker._update_depth()

    assert metrics.snapshot()["queue_depth"] == 2.0
