from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeAlias

from gliner_runner.backends.base import InferenceBackend
from gliner_runner.contracts import BackendResult, InferenceRequest, InferenceResponse, Timing
from gliner_runner.errors import QueueFullError, RunnerClosedError
from gliner_runner.metrics import Metrics

BatchKey: TypeAlias = tuple[str, str, str]
_STOP: Final = object()


@dataclass
class _Work:
    request: InferenceRequest
    future: asyncio.Future[InferenceResponse]
    enqueued_at: float


class ModelWorker:
    """Owns exactly one loaded model instance for a model/backend/precision/device tuple."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        queue_capacity: int = 256,
        max_batch_size: int = 16,
        batch_window_ms: float = 4.0,
        metrics: Metrics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if queue_capacity < 1 or max_batch_size < 1 or batch_window_ms < 0:
            raise ValueError("invalid scheduler configuration")
        self._backend = backend
        self._queue: asyncio.Queue[_Work | object] = asyncio.Queue(queue_capacity)
        self._backlog: deque[_Work | object] = deque()
        self._max_batch_size = max_batch_size
        self._batch_window = batch_window_ms / 1000
        self._metrics = metrics or Metrics()
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._accepting = False

    @property
    def accepting(self) -> bool:
        return self._accepting

    async def start(self) -> None:
        if self._task is not None:
            return
        await self._backend.load()
        self._accepting = True
        self._task = asyncio.create_task(self._run(), name="gliner-model-worker")

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        if not self._accepting:
            raise RunnerClosedError("model worker is not accepting requests")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResponse] = loop.create_future()
        work = _Work(request=request, future=future, enqueued_at=self._clock())
        try:
            self._queue.put_nowait(work)
        except asyncio.QueueFull as error:
            self._metrics.increment("requests_rejected_total")
            raise QueueFullError("model worker queue is full") from error
        self._metrics.increment("requests_submitted_total")
        self._update_depth()
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            self._metrics.increment("requests_cancelled_total")
            raise

    async def close(self) -> None:
        self._accepting = False
        if self._task is None:
            return
        await self._queue.put(_STOP)
        await self._task
        self._task = None
        await self._backend.close()

    async def _run(self) -> None:
        while True:
            first = await self._next()
            if first is _STOP:
                break
            assert isinstance(first, _Work)
            batch = await self._collect(first)
            active = [work for work in batch if not work.future.cancelled()]
            if not active:
                continue
            started = self._clock()
            self._metrics.increment("batches_total")
            self._metrics.gauge("last_batch_size", float(len(active)))
            try:
                outputs = await self._backend.infer_batch([work.request for work in active])
                if len(outputs) != len(active):
                    raise RuntimeError("backend returned the wrong number of results")
            except Exception as error:
                self._metrics.increment("batch_failures_total")
                for work in active:
                    if not work.future.done():
                        work.future.set_exception(error)
            else:
                finished = self._clock()
                for work, result in zip(active, outputs, strict=True):
                    if not work.future.done():
                        work.future.set_result(self._response(work, result, started, finished))
                self._metrics.increment("requests_completed_total", len(active))
            finally:
                self._update_depth()
        self._fail_pending(RunnerClosedError("model worker closed"))

    async def _next(self) -> _Work | object:
        if self._backlog:
            return self._backlog.popleft()
        item = await self._queue.get()
        self._update_depth()
        return item

    async def _collect(self, first: _Work) -> list[_Work]:
        batch = [first]
        key = _batch_key(first.request)
        if self._batch_window:
            await asyncio.sleep(self._batch_window)
        backlog_size = len(self._backlog)
        for _ in range(backlog_size):
            item = self._backlog.popleft()
            if item is _STOP:
                self._backlog.append(item)
                continue
            assert isinstance(item, _Work)
            if len(batch) < self._max_batch_size and _batch_key(item.request) == key:
                batch.append(item)
            else:
                self._backlog.append(item)
        while len(batch) < self._max_batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is _STOP:
                self._backlog.append(_STOP)
                break
            assert isinstance(item, _Work)
            if _batch_key(item.request) == key:
                batch.append(item)
            else:
                self._backlog.append(item)
        return batch

    def _response(
        self,
        work: _Work,
        result: BackendResult,
        started: float,
        finished: float,
    ) -> InferenceResponse:
        request = work.request
        return InferenceResponse(
            request_id=request.request_id,
            model=request.model,
            backend=request.backend,
            precision=request.precision,
            output=result.output,
            timing=Timing(
                queue_ms=max(0.0, (started - work.enqueued_at) * 1000),
                inference_ms=max(0.0, (finished - started) * 1000),
            ),
            usage=result.usage,
        )

    def _fail_pending(self, error: Exception) -> None:
        pending: list[_Work] = []
        for item in self._backlog:
            if item is not _STOP:
                assert isinstance(item, _Work)
                pending.append(item)
        self._backlog.clear()
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not _STOP:
                assert isinstance(item, _Work)
                pending.append(item)
        for work in pending:
            if not work.future.done():
                work.future.set_exception(error)
        self._update_depth()

    def _update_depth(self) -> None:
        self._metrics.gauge("queue_depth", float(self._queue.qsize() + len(self._backlog)))


def _batch_key(request: InferenceRequest) -> BatchKey:
    schema = json.dumps(request.schema_.model_dump(mode="json"), sort_keys=True)
    options = json.dumps(request.options.model_dump(mode="json"), sort_keys=True)
    return request.operation, schema, options
