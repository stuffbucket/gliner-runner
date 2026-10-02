from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence

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
from gliner_runner.scheduler import ModelWorker


class BenchmarkBackend:
    def __init__(self) -> None:
        self.batches = 0

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
        return None

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[JsonValue]:
        self.batches += 1
        return [{"label": "benchmark"} for _ in requests]

    async def close(self) -> None:
        return None


def make_request(index: int) -> InferenceRequest:
    return InferenceRequest(
        model="benchmark/model",
        backend=BackendName.PYTORCH,
        precision=Precision.FP32,
        operation=InferenceOperation.CLASSIFY,
        text=f"benchmark {index}",
        schema=ClassificationSchema(
            tasks={
                "label": ClassificationTask(labels=("benchmark", "other")),
            }
        ),
    )


async def benchmark(concurrency: int, requests: int) -> None:
    backend = BenchmarkBackend()
    worker = ModelWorker(backend, max_batch_size=concurrency)
    await worker.start()
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int) -> None:
        async with semaphore:
            await worker.submit(make_request(index))

    started = time.perf_counter()
    await asyncio.gather(*(run(index) for index in range(requests)))
    elapsed = time.perf_counter() - started
    await worker.close()
    print(f"{requests / elapsed:.1f} requests/s; {backend.batches} batches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(benchmark(args.concurrency, args.requests))
