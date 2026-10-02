from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "5a7adf72a23b4d311abae6ce050d7f0012bb3416"
LOGICAL_MODEL = f"{MODEL_ID}@{MODEL_REVISION}"
SUPPORTED_PROFILES = {
    ("cpu", "fp32"),
    ("cuda", "fp32"),
    ("cuda", "fp16"),
    ("cuda", "bf16"),
    ("mps", "fp16"),
    ("mps", "fp32"),
}
SCHEMA_LABELS = ("billing", "technical_support", "account_access", "shipping")
PARITY_CASES = (
    ("My credit card was charged twice for the same invoice.", "billing"),
    ("The desktop application crashes whenever I export a report.", "technical_support"),
    ("I cannot sign in after resetting my password.", "account_access"),
    ("Where is the package that was supposed to arrive yesterday?", "shipping"),
)
BASE_WORDS = (
    "The",
    "customer",
    "reported",
    "an",
    "account",
    "issue",
    "and",
    "requested",
    "a",
    "careful",
    "review",
    "of",
    "the",
    "invoice,",
    "delivery",
    "status,",
    "application",
    "behavior,",
    "and",
    "access",
    "history.",
)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def schema(task_name: str = "intent") -> dict[str, Any]:
    return {
        "kind": "classification",
        "tasks": {
            task_name: {
                "labels": list(SCHEMA_LABELS),
                "min_labels": 1,
                "max_labels": 1,
            }
        },
    }


def request(
    text: str,
    *,
    precision: str,
    model: str = LOGICAL_MODEL,
    task_name: str = "intent",
) -> dict[str, Any]:
    return {
        "model": model,
        "backend": "pytorch",
        "precision": precision,
        "operation": "classify",
        "text": text,
        "schema": schema(task_name),
        "options": {
            "threshold": 0.5,
            "output_format": "native",
            "include_confidence": True,
        },
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def worker_oracle(model_path: Path, event_path: Path) -> None:
    import psutil

    process = psutil.Process()
    append_event(
        event_path,
        {
            "event": "started",
            "rss_bytes": process.memory_info().rss,
            "monotonic": time.monotonic(),
        },
    )

    import torch
    from gliner2.classification import (
        ClassificationConfig,
        ClassificationSchema,
        Classifier,
    )

    load_started = time.perf_counter()
    classifier = Classifier.from_pretrained(
        str(model_path),
        device="cpu",
        dtype="float32",
    ).eval()
    load_ms = (time.perf_counter() - load_started) * 1000
    loaded_rss = process.memory_info().rss
    official_schema = ClassificationSchema().single("intent", list(SCHEMA_LABELS))
    config = ClassificationConfig(batch_size=len(PARITY_CASES), include_confidence=True)
    inference_started = time.perf_counter()
    results = classifier.batch_classify(
        [text for text, _ in PARITY_CASES],
        official_schema,
        config=config,
    )
    inference_ms = (time.perf_counter() - inference_started) * 1000
    payload = {
        "event": "finished",
        "load_ms": load_ms,
        "batch_inference_ms": inference_ms,
        "loaded_rss_bytes": loaded_rss,
        "steady_rss_bytes": process.memory_info().rss,
        "outputs": [result.to_dict(include_confidence=True) for result in results],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gliner2": importlib.metadata.version("gliner2"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "torch_threads": torch.get_num_threads(),
    }
    append_event(event_path, payload)


def accelerator_memory(device: str) -> dict[str, int] | None:
    if device != "mps":
        return None
    import torch

    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "recommended_max_bytes": torch.mps.recommended_max_memory(),
    }


class RecordedBackend:
    def __init__(
        self,
        backend: Any,
        logical_model: str,
        device: str,
        event_path: Path,
        event_lock: threading.Lock,
    ) -> None:
        self._backend = backend
        self._logical_model = logical_model
        self._device = device
        self._event_path = event_path
        self._event_lock = event_lock

    @property
    def capabilities(self) -> Any:
        return self._backend.capabilities

    async def load(self) -> None:
        started = time.perf_counter()
        await self._backend.load()
        self._record(
            {
                "event": "load",
                "logical_model": self._logical_model,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "accelerator_memory": accelerator_memory(self._device),
            }
        )

    async def infer_batch(self, requests: Sequence[Any]) -> list[Any]:
        task_names = sorted(
            {
                task_name
                for item in requests
                for task_name in getattr(item.schema_, "tasks", {})
            }
        )
        started = time.perf_counter()
        outputs = await self._backend.infer_batch(requests)
        self._record(
            {
                "event": "batch",
                "logical_model": self._logical_model,
                "size": len(requests),
                "task_names": task_names,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "accelerator_memory": accelerator_memory(self._device),
            }
        )
        return outputs

    async def close(self) -> None:
        await self._backend.close()

    def _record(self, event: dict[str, Any]) -> None:
        with self._event_lock:
            append_event(self._event_path, event)


def worker_server(
    model_path: Path,
    event_path: Path,
    port: int,
    queue_capacity: int,
    max_batch_size: int,
    batch_window_ms: float,
    device: str,
) -> None:
    import psutil
    import uvicorn

    from gliner_runner.backends.pytorch import PyTorchBackend
    from gliner_runner.backends.registry import BackendRegistry
    from gliner_runner.contracts import BackendName
    from gliner_runner.runtime import Runtime, RuntimeConfig
    from gliner_runner.server import create_app

    lock = threading.Lock()
    registry = BackendRegistry()
    registry.register(
        BackendName.PYTORCH,
        lambda requested_model, precision, device: RecordedBackend(
            PyTorchBackend(str(model_path), precision, device),
            requested_model,
            device,
            event_path,
            lock,
        ),
    )
    runtime = Runtime(
        RuntimeConfig(
            device=device,
            queue_capacity=queue_capacity,
            max_batch_size=max_batch_size,
            batch_window_ms=batch_window_ms,
        ),
        registry=registry,
    )
    append_event(
        event_path,
        {
            "event": "server_starting",
            "rss_bytes": psutil.Process().memory_info().rss,
        },
    )
    uvicorn.run(
        create_app(runtime=runtime),
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ProcessRssSampler:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        import psutil

        self._process = psutil.Process(process.pid)
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak(self) -> int:
        return max(self._samples, default=0)

    def _run(self) -> None:
        import psutil

        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                for child in self._process.children(recursive=True):
                    with suppress(psutil.Error):
                        rss += child.memory_info().rss
                self._samples.append(rss)
            except psutil.Error:
                return
            self._stop.wait(0.02)


def launch_worker(arguments: list[str]) -> subprocess.Popen[bytes]:
    event_index = arguments.index("--events") + 1
    error_path = Path(arguments[event_index]).with_suffix(".stderr.log")
    with error_path.open("wb") as error_stream:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            stdout=subprocess.DEVNULL,
            stderr=error_stream,
        )


def worker_error(
    process: subprocess.Popen[bytes],
    event_path: Path,
    label: str,
) -> RuntimeError:
    error_path = event_path.with_suffix(".stderr.log")
    details = error_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    message = f"{label} exited with status {process.returncode}"
    if details.strip():
        message = f"{message}:\n{details}"
    return RuntimeError(message)


def wait_for_event(
    path: Path,
    event_name: str,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in read_events(path):
            if event.get("event") == event_name:
                return event
        if process.poll() is not None:
            raise worker_error(process, path, "worker")
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {event_name}")


def terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(
    base_url: str,
    process: subprocess.Popen[bytes],
    event_path: Path,
    timeout: float,
) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise worker_error(process, event_path, "server")
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=0.5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError("server did not become healthy")


def tree_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        file_digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_digest.digest())
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def host_metadata() -> dict[str, Any]:
    import psutil

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            cpu = result.stdout.strip()
    return {
        "cpu": cpu,
        "architecture": platform.machine(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.mac_ver()[0] if sys.platform == "darwin" else platform.version(),
    }


def direct_oracle_measurement(
    model_path: Path,
    temporary: Path,
) -> tuple[dict[str, Any], int]:
    event_path = temporary / "oracle.jsonl"
    spawn_started = time.monotonic()
    process = launch_worker(
        ["_oracle", "--model-path", str(model_path), "--events", str(event_path)]
    )
    sampler = ProcessRssSampler(process)
    sampler.start()
    try:
        started = wait_for_event(event_path, "started", process, timeout=30)
        process_start_ms = (time.monotonic() - spawn_started) * 1000
        process.wait(timeout=300)
        if process.returncode != 0:
            raise worker_error(process, event_path, "oracle worker")
    finally:
        sampler.stop()
        terminate(process)
    finished = next(event for event in read_events(event_path) if event["event"] == "finished")
    finished["process_start_ms"] = process_start_ms
    finished["starting_rss_bytes"] = started["rss_bytes"]
    finished["peak_rss_bytes"] = sampler.peak
    return finished, sampler.peak


def make_text(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    words: list[str] = []
    index = 0
    while True:
        words.append(BASE_WORDS[index % len(BASE_WORDS)])
        index += 1
        text = " ".join(words)
        count = len(tokenizer.encode(text, add_special_tokens=True))
        if count >= target_tokens:
            return text, count


def response_payload(response: Any) -> dict[str, Any]:
    response.raise_for_status()
    body = response.json()
    if body.get("error") is not None:
        raise RuntimeError(f"inference failed: {body['error']['code']}")
    usage = body.get("usage")
    if not isinstance(usage, dict) or set(usage) != {"inputTokens", "outputTokens"}:
        raise RuntimeError("inference response is missing the exact usage contract")
    if usage["outputTokens"] != 0:
        raise RuntimeError("classification unexpectedly reported output tokens")
    return body


def response_output(response: Any) -> dict[str, Any]:
    body = response_payload(response)
    return body["output"]


def selected_label(output: dict[str, Any], task_name: str = "intent") -> str:
    value = output[task_name]
    selected = value["value"] if isinstance(value, dict) else value
    if not isinstance(selected, str):
        raise TypeError("expected one selected label")
    return selected


def numeric_values(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, bool):
        return output
    if isinstance(value, int | float):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            output.update(numeric_values(child, f"{prefix}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(numeric_values(child, f"{prefix}/{index}"))
    return output


def parity_summary(
    oracle_outputs: list[dict[str, Any]],
    runner_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    exact = 0
    labels = 0
    maximum_difference = 0.0
    for oracle, runner in zip(oracle_outputs, runner_outputs, strict=True):
        if oracle == runner:
            exact += 1
        if selected_label(oracle) == selected_label(runner):
            labels += 1
        oracle_numbers = numeric_values(oracle)
        runner_numbers = numeric_values(runner)
        for key in oracle_numbers.keys() & runner_numbers.keys():
            maximum_difference = max(
                maximum_difference,
                abs(oracle_numbers[key] - runner_numbers[key]),
            )
    count = len(oracle_outputs)
    return {
        "cases": count,
        "exact_output_matches": exact,
        "exact_output_rate": exact / count,
        "selected_label_matches": labels,
        "selected_label_rate": labels / count,
        "max_abs_numeric_difference": maximum_difference,
    }


async def concurrent_infer(
    base_url: str,
    requests: list[dict[str, Any]],
    delays: list[float] | None = None,
) -> list[int]:
    import httpx

    delays = delays or [0.0] * len(requests)
    async with httpx.AsyncClient(timeout=300) as client:
        async def send(item: dict[str, Any], delay: float) -> int:
            await asyncio.sleep(delay)
            response = await client.post(f"{base_url}/v1/infer", json=item)
            return response.status_code

        return await asyncio.gather(
            *(send(item, delay) for item, delay in zip(requests, delays, strict=True))
        )


def batch_events_since(path: Path, offset: int) -> list[dict[str, Any]]:
    return [
        event
        for event in read_events(path)[offset:]
        if event.get("event") == "batch"
    ]


def run_normal_server(
    model_path: Path,
    temporary: Path,
    oracle: dict[str, Any],
    repetitions: int,
    token_targets: list[int],
    device: str,
    precision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    import httpx
    import psutil
    from transformers import AutoTokenizer

    event_path = temporary / "server.jsonl"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    spawn_started = time.monotonic()
    process = launch_worker(
        [
            "_server",
            "--model-path",
            str(model_path),
            "--events",
            str(event_path),
            "--port",
            str(port),
            "--queue-capacity",
            "128",
            "--max-batch-size",
            "16",
            "--batch-window-ms",
            "4",
            "--device",
            device,
        ]
    )
    sampler = ProcessRssSampler(process)
    sampler.start()
    try:
        wait_for_server(base_url, process, event_path, timeout=30)
        server_ready_ms = (time.monotonic() - spawn_started) * 1000
        starting_rss = psutil.Process(process.pid).memory_info().rss
        with httpx.Client(timeout=300) as client:
            cold_started = time.perf_counter()
            first_response = response_payload(
                client.post(
                    f"{base_url}/v1/infer",
                    json=request(PARITY_CASES[0][0], precision=precision),
                )
            )
            cold_http_ms = (time.perf_counter() - cold_started) * 1000
            parity_outputs = [first_response["output"]]
            parity_usage = [first_response["usage"]]
            for text, _ in PARITY_CASES[1:]:
                parity_response = response_payload(
                    client.post(
                        f"{base_url}/v1/infer",
                        json=request(text, precision=precision),
                    )
                )
                parity_outputs.append(parity_response["output"])
                parity_usage.append(parity_response["usage"])
            steady_rss = psutil.Process(process.pid).memory_info().rss

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            texts = {
                target: make_text(tokenizer, target)
                for target in token_targets
            }
            rows: list[dict[str, Any]] = []
            for target in token_targets:
                text, actual_tokens = texts[target]
                for batch_size in (1, 2, 4, 8):
                    latencies: list[float] = []
                    queue_values: list[float] = []
                    inference_values: list[float] = []
                    usage_input_values: list[int] = []
                    for _ in range(repetitions):
                        payload = {
                            "requests": [
                                request(text, precision=precision, task_name="latency")
                                for _index in range(batch_size)
                            ]
                        }
                        started = time.perf_counter()
                        response = client.post(f"{base_url}/v1/batch", json=payload)
                        elapsed = time.perf_counter() - started
                        response.raise_for_status()
                        responses = response.json()["responses"]
                        for item in responses:
                            usage = item.get("usage")
                            if (
                                not isinstance(usage, dict)
                                or set(usage) != {"inputTokens", "outputTokens"}
                                or usage["outputTokens"] != 0
                            ):
                                raise RuntimeError(
                                    "batch response violated the classification usage contract"
                                )
                            usage_input_values.append(int(usage["inputTokens"]))
                        latencies.append(elapsed * 1000)
                        queue_values.extend(item["timing"]["queue_ms"] for item in responses)
                        inference_values.extend(
                            item["timing"]["inference_ms"] for item in responses
                        )
                    total_requests = batch_size * repetitions
                    total_seconds = sum(latencies) / 1000
                    unique_usage_counts = set(usage_input_values)
                    if len(unique_usage_counts) != 1:
                        raise RuntimeError(
                            "identical benchmark requests reported different token usage"
                        )
                    rows.append(
                        {
                            "target_tokens": target,
                            "actual_tokens": actual_tokens,
                            "runner_input_tokens": unique_usage_counts.pop(),
                            "output_tokens": 0,
                            "batch_size": batch_size,
                            "batch_samples": repetitions,
                            "total_requests": total_requests,
                            "wall_p50_ms": percentile(latencies, 0.50),
                            "wall_p95_ms": percentile(latencies, 0.95),
                            "throughput_requests_per_s": total_requests / total_seconds,
                            "queue_p50_ms": percentile(queue_values, 0.50),
                            "inference_p50_ms": percentile(inference_values, 0.50),
                        }
                    )

            fill_results: list[dict[str, Any]] = []
            scenarios = (
                ("simultaneous", [0.0] * 8),
                ("inside_window", [index * 0.0004 for index in range(8)]),
                ("outside_window", [index * 0.012 for index in range(4)]),
            )
            fill_text = texts[token_targets[0]][0]
            for name, delays in scenarios:
                offset = len(read_events(event_path))
                statuses = asyncio.run(
                    concurrent_infer(
                        base_url,
                        [
                            request(
                                fill_text,
                                precision=precision,
                                task_name=f"fill_{name}",
                            )
                            for _ in delays
                        ],
                        delays,
                    )
                )
                events = batch_events_since(event_path, offset)
                fill_results.append(
                    {
                        "scenario": name,
                        "arrival_offsets_ms": [delay * 1000 for delay in delays],
                        "http_statuses": statuses,
                        "observed_batch_sizes": [event["size"] for event in events],
                    }
                )

            grouping_offset = len(read_events(event_path))
            grouping_statuses = asyncio.run(
                concurrent_infer(
                    base_url,
                    [
                        request(
                            fill_text,
                            precision=precision,
                            task_name="group_a" if index % 2 == 0 else "group_b",
                        )
                        for index in range(8)
                    ],
                )
            )
            schema_group_events = batch_events_since(event_path, grouping_offset)

            model_grouping_offset = len(read_events(event_path))
            model_statuses = asyncio.run(
                concurrent_infer(
                    base_url,
                    [
                        request(fill_text, precision=precision, model=LOGICAL_MODEL),
                        request(
                            fill_text,
                            precision=precision,
                            model=f"{LOGICAL_MODEL}#alias",
                        ),
                    ],
                )
            )
            model_group_events = batch_events_since(event_path, model_grouping_offset)
            post_grouping_rss = psutil.Process(process.pid).memory_info().rss
            metrics_text = client.get(f"{base_url}/metrics").text

        load_events = [
            event for event in read_events(event_path) if event.get("event") == "load"
        ]
        accelerator_values = [
            event["accelerator_memory"]["driver_allocated_bytes"]
            for event in read_events(event_path)
            if event.get("accelerator_memory") is not None
        ]
        result = {
            "server_ready_ms": server_ready_ms,
            "starting_rss_bytes": starting_rss,
            "steady_single_model_rss_bytes": steady_rss,
            "post_two_model_grouping_rss_bytes": post_grouping_rss,
            "cold_first_http_ms": cold_http_ms,
            "backend_load_events": load_events,
            "accelerator_peak_bytes": max(accelerator_values, default=None),
            "parity": parity_summary(oracle["outputs"], parity_outputs),
            "parity_usage": parity_usage,
            "expected_label_accuracy": {
                "cases": len(PARITY_CASES),
                "correct": sum(
                    selected_label(output) == expected
                    for output, (_text, expected) in zip(
                        parity_outputs, PARITY_CASES, strict=True
                    )
                ),
            },
            "fill_window": fill_results,
            "schema_grouping": {
                "http_statuses": grouping_statuses,
                "batches": schema_group_events,
                "mixed_schema_batch_observed": any(
                    len(event["task_names"]) > 1 for event in schema_group_events
                ),
            },
            "model_grouping": {
                "http_statuses": model_statuses,
                "batches": model_group_events,
                "logical_models_observed": sorted(
                    {event["logical_model"] for event in model_group_events}
                ),
            },
            "metrics_excerpt": [
                line
                for line in metrics_text.splitlines()
                if "requests_cancelled_total" in line
                or "requests_rejected_total" in line
                or "last_batch_size" in line
            ],
        }
        return result, rows, sampler.peak
    finally:
        sampler.stop()
        terminate(process)


async def overload_requests(
    base_url: str,
    long_text: str,
    precision: str,
) -> dict[str, Any]:
    import httpx

    timeout = httpx.Timeout(300)
    async with httpx.AsyncClient(timeout=timeout) as client:
        first = asyncio.create_task(
            client.post(
                f"{base_url}/v1/infer",
                json=request(long_text, precision=precision, task_name="overload"),
            )
        )
        await asyncio.sleep(0.01)
        cancellable = asyncio.create_task(
            client.post(
                f"{base_url}/v1/infer",
                json=request(long_text, precision=precision, task_name="cancelled"),
            )
        )
        await asyncio.sleep(0.001)
        cancelled_by_client = cancellable.cancel()
        with suppress(asyncio.CancelledError):
            await cancellable
        burst = await asyncio.gather(
            *(
                client.post(
                    f"{base_url}/v1/infer",
                    json=request(long_text, precision=precision, task_name="overload"),
                )
                for _ in range(10)
            )
        )
        first_response = await first
        await asyncio.sleep(0.1)
        metrics = await client.get(f"{base_url}/metrics")
        return {
            "first_status": first_response.status_code,
            "burst_statuses": [response.status_code for response in burst],
            "accepted": sum(response.status_code == 200 for response in burst),
            "rejected_429": sum(response.status_code == 429 for response in burst),
            "client_task_cancelled": cancelled_by_client,
            "metrics": metrics.text,
        }


def metric_value(metrics: str, name: str) -> float:
    marker = f'gliner_runner_internal{{metric="{name}"}} '
    for line in metrics.splitlines():
        if line.startswith(marker):
            return float(line.removeprefix(marker))
    return 0.0


def run_overload_server(
    model_path: Path,
    temporary: Path,
    long_text: str,
    device: str,
    precision: str,
) -> tuple[dict[str, Any], int]:
    import httpx

    event_path = temporary / "overload.jsonl"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = launch_worker(
        [
            "_server",
            "--model-path",
            str(model_path),
            "--events",
            str(event_path),
            "--port",
            str(port),
            "--queue-capacity",
            "2",
            "--max-batch-size",
            "1",
            "--batch-window-ms",
            "50",
            "--device",
            device,
        ]
    )
    sampler = ProcessRssSampler(process)
    sampler.start()
    try:
        wait_for_server(base_url, process, event_path, timeout=30)
        with httpx.Client(timeout=300) as client:
            response_output(
                client.post(
                    f"{base_url}/v1/infer",
                    json=request(
                        PARITY_CASES[0][0],
                        precision=precision,
                        task_name="warmup",
                    ),
                )
            )
        result = asyncio.run(overload_requests(base_url, long_text, precision))
        result["server_cancelled_metric"] = metric_value(
            result.pop("metrics"),
            "requests_cancelled_total",
        )
        result["server_rejected_metric"] = metric_value(
            httpx.get(f"{base_url}/metrics", timeout=30).text,
            "requests_rejected_total",
        )
        result["observed_batches"] = batch_events_since(event_path, 0)
        return result, sampler.peak
    finally:
        sampler.stop()
        terminate(process)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gibibytes(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def profile_name(device: str, precision: str) -> str:
    return f"{device.upper()} / {precision.upper()}"


def comparison_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_metadata = baseline["metadata"]
    candidate_metadata = candidate["metadata"]
    if baseline_metadata["model_content_sha256"] != candidate_metadata["model_content_sha256"]:
        raise ValueError("baseline model content digest does not match the candidate")
    if baseline_metadata["host"]["cpu"] != candidate_metadata["host"]["cpu"]:
        raise ValueError("baseline CPU does not match the candidate host")

    baseline_rows = {
        (row["target_tokens"], row["batch_size"]): row
        for row in baseline["latency_rows"]
    }
    matrix: list[dict[str, Any]] = []
    for candidate_row in candidate["latency_rows"]:
        key = (candidate_row["target_tokens"], candidate_row["batch_size"])
        baseline_row = baseline_rows.get(key)
        if baseline_row is None:
            raise ValueError(f"baseline is missing latency cell {key}")
        matrix.append(
            {
                "target_tokens": key[0],
                "batch_size": key[1],
                "baseline_wall_p50_ms": baseline_row["wall_p50_ms"],
                "candidate_wall_p50_ms": candidate_row["wall_p50_ms"],
                "wall_p50_speedup": (
                    baseline_row["wall_p50_ms"] / candidate_row["wall_p50_ms"]
                ),
                "baseline_throughput_requests_per_s": baseline_row[
                    "throughput_requests_per_s"
                ],
                "candidate_throughput_requests_per_s": candidate_row[
                    "throughput_requests_per_s"
                ],
                "throughput_ratio": (
                    candidate_row["throughput_requests_per_s"]
                    / baseline_row["throughput_requests_per_s"]
                ),
            }
        )
    baseline_load = baseline["http_runner"]["backend_load_events"][0]["duration_ms"]
    candidate_load = candidate["http_runner"]["backend_load_events"][0]["duration_ms"]
    return {
        "baseline_profile": {
            "device": baseline_metadata["device"],
            "precision": baseline_metadata["precision"],
        },
        "candidate_profile": {
            "device": candidate_metadata["device"],
            "precision": candidate_metadata["precision"],
        },
        "cold": {
            "baseline_backend_load_ms": baseline_load,
            "candidate_backend_load_ms": candidate_load,
            "backend_load_ratio": baseline_load / candidate_load,
            "baseline_first_http_ms": baseline["http_runner"]["cold_first_http_ms"],
            "candidate_first_http_ms": candidate["http_runner"]["cold_first_http_ms"],
            "first_http_ratio": (
                baseline["http_runner"]["cold_first_http_ms"]
                / candidate["http_runner"]["cold_first_http_ms"]
            ),
        },
        "memory": {
            "baseline_steady_rss_bytes": baseline["http_runner"][
                "steady_single_model_rss_bytes"
            ],
            "candidate_steady_rss_bytes": candidate["http_runner"][
                "steady_single_model_rss_bytes"
            ],
            "baseline_peak_rss_bytes": baseline["memory"]["http_peak_rss_bytes"],
            "candidate_peak_rss_bytes": candidate["memory"]["http_peak_rss_bytes"],
            "candidate_accelerator_peak_bytes": candidate["memory"].get(
                "accelerator_peak_bytes"
            ),
        },
        "matrix": matrix,
    }


def accelerator_peak(
    server: dict[str, Any],
    overload: dict[str, Any],
) -> int | None:
    events = overload["observed_batches"]
    values = [server["accelerator_peak_bytes"]]
    values.extend(
        event["accelerator_memory"]["driver_allocated_bytes"]
        for event in events
        if event.get("accelerator_memory") is not None
    )
    return max((value for value in values if value is not None), default=None)


def generate_report(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metadata = result["metadata"]
    cold = result["cold_oracle"]
    server = result["http_runner"]
    overload = result["overload"]
    fastest = max(rows, key=lambda row: row["throughput_requests_per_s"])
    slowest_tail = max(rows, key=lambda row: row["wall_p95_ms"])
    accuracy = server["expected_label_accuracy"]
    fill_rows = "\n".join(
        "| "
        f"{item['scenario']} | "
        f"{', '.join(f'{offset:g}' for offset in item['arrival_offsets_ms'])} | "
        f"{', '.join(map(str, item['observed_batch_sizes']))} |"
        for item in server["fill_window"]
    )
    schema_batch_sizes = [
        event["size"] for event in server["schema_grouping"]["batches"]
    ]
    cancellation_observation = (
        "The client task was cancelled, but the server recorded no scheduler "
        "cancellation; this loopback HTTP disconnect did not propagate to the "
        "in-flight route task."
        if overload["client_task_cancelled"]
        and overload["server_cancelled_metric"] == 0
        else "Client and server cancellation counters are recorded in the JSON artifact."
    )
    accelerator_memory_row = ""
    if result["memory"].get("accelerator_peak_bytes") is not None:
        accelerator_memory_row = (
            "| Peak MPS driver-allocated memory | "
            f"{gibibytes(result['memory']['accelerator_peak_bytes'])} |\n"
        )
    comparison_section = ""
    comparison = result.get("comparison")
    if comparison is not None:
        comparison_rows = "\n".join(
            "| "
            f"{row['target_tokens']} | {row['batch_size']} | "
            f"{row['baseline_wall_p50_ms']:.1f} | "
            f"{row['candidate_wall_p50_ms']:.1f} | "
            f"{row['wall_p50_speedup']:.2f}x | "
            f"{row['throughput_ratio']:.2f}x |"
            for row in comparison["matrix"]
        )
        baseline_profile = comparison["baseline_profile"]
        candidate_profile = comparison["candidate_profile"]
        baseline_name = profile_name(
            baseline_profile["device"],
            baseline_profile["precision"],
        )
        candidate_name = profile_name(
            candidate_profile["device"],
            candidate_profile["precision"],
        )
        cold_comparison = comparison["cold"]
        memory_comparison = comparison["memory"]
        comparison_section = f"""
## {baseline_name} versus {candidate_name}

Baseline: {baseline_name}.
Candidate: {candidate_name}.
Both measurements use the same model content digest and machine.

- Backend load: {cold_comparison["baseline_backend_load_ms"]:.1f} ms baseline
  versus {cold_comparison["candidate_backend_load_ms"]:.1f} ms candidate
  ({cold_comparison["backend_load_ratio"]:.2f}x baseline/candidate ratio).
- Cold first HTTP request: {cold_comparison["baseline_first_http_ms"]:.1f} ms
  baseline versus {cold_comparison["candidate_first_http_ms"]:.1f} ms candidate
  ({cold_comparison["first_http_ratio"]:.2f}x baseline/candidate ratio).
- Steady RSS: {gibibytes(memory_comparison["baseline_steady_rss_bytes"])} baseline
  versus {gibibytes(memory_comparison["candidate_steady_rss_bytes"])} candidate.
- Peak process RSS: {gibibytes(memory_comparison["baseline_peak_rss_bytes"])} baseline
  versus {gibibytes(memory_comparison["candidate_peak_rss_bytes"])} candidate.

| Tokens | Batch | Baseline p50 ms | Candidate p50 ms | Latency speedup | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
{comparison_rows}
"""
    title_profile = f"{metadata['device'].upper()} {metadata['precision'].upper()}"
    return f"""# Real-model characterization: GLiNER2.5-Decide 340M ({title_profile})

Generated from `characterization.json` by `benchmarks/benchmark_real.py`.

## Scope and caution

These are one-machine measurements, not general hardware or deployment
recommendations. The workload is a small deterministic characterization set,
the p95 values have only {metadata["repetitions"]} batch samples per matrix
cell, and synthetic length probes are not representative of every production
distribution.

## Environment

- Hardware: {metadata["host"]["cpu"]}, {metadata["host"]["physical_cores"]} physical /
  {metadata["host"]["logical_cores"]} logical cores, {gibibytes(metadata["host"]["memory_bytes"])}
- OS: {metadata["host"]["os"]} {metadata["host"]["os_version"]} ({metadata["host"]["architecture"]})
- Python {metadata["versions"]["python"]}; PyTorch {metadata["versions"]["torch"]};
  GLiNER2 {metadata["versions"]["gliner2"]}; Transformers {metadata["versions"]["transformers"]}
- Candidate device/precision: {profile_name(metadata["device"], metadata["precision"])}
- Correctness oracle: CPU / FP32
- Model: `{metadata["model_id"]}` at `{metadata["model_revision"]}`
- Snapshot content SHA-256: `{metadata["model_content_sha256"]}`
- Snapshot: {metadata["model_files"]} files, {gibibytes(metadata["model_bytes"])}

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | {cold["process_start_ms"]:.1f} ms |
| Direct official model load | {cold["load_ms"]:.1f} ms |
| First direct batch ({len(PARITY_CASES)} items) | {cold["batch_inference_ms"]:.1f} ms |
| Oracle starting RSS | {gibibytes(cold["starting_rss_bytes"])} |
| Oracle steady RSS | {gibibytes(cold["steady_rss_bytes"])} |
| Oracle peak RSS | {gibibytes(cold["peak_rss_bytes"])} |
| HTTP server ready before model load | {server["server_ready_ms"]:.1f} ms |
| Candidate backend load | {server["backend_load_events"][0]["duration_ms"]:.1f} ms |
| Cold first HTTP request including lazy load | {server["cold_first_http_ms"]:.1f} ms |
| HTTP process starting RSS | {gibibytes(server["starting_rss_bytes"])} |
| HTTP steady RSS, one loaded model | {gibibytes(server["steady_single_model_rss_bytes"])} |
| HTTP peak RSS, including two-model probe | {gibibytes(result["memory"]["http_peak_rss_bytes"])} |
{accelerator_memory_row}

## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets {", ".join(map(str, metadata["token_targets"]))}.

- Highest observed throughput: {fastest["throughput_requests_per_s"]:.2f} requests/s
  at batch {fastest["batch_size"]}, {fastest["runner_input_tokens"]} runner input tokens
  ({fastest["actual_tokens"]} raw-text tokenizer tokens).
- Largest observed p95 batch wall time: {slowest_tail["wall_p95_ms"]:.1f} ms
  at batch {slowest_tail["batch_size"]},
  {slowest_tail["runner_input_tokens"]} runner input tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  {server["parity"]["exact_output_matches"]}/{server["parity"]["cases"]}
  ({server["parity"]["exact_output_rate"]:.1%}).
- Selected-label parity:
  {server["parity"]["selected_label_matches"]}/{server["parity"]["cases"]}
  ({server["parity"]["selected_label_rate"]:.1%}).
- Maximum numeric difference: {server["parity"]["max_abs_numeric_difference"]:.3g}.
- Expected-label accuracy on the four transparent examples:
  {accuracy["correct"]}/{accuracy["cases"]}.
- Mixed-schema batch observed: {server["schema_grouping"]["mixed_schema_batch_observed"]}.
- Interleaved schema batch sizes: {schema_batch_sizes}.
- Distinct logical models observed:
  {len(server["model_grouping"]["logical_models_observed"])}.
- Overload probe: {overload["rejected_429"]} HTTP 429 responses and
  {int(overload["server_cancelled_metric"])} server-observed cancellations.
  {cancellation_observation}

### Fill-window observations

| Scenario | Arrival offsets (ms) | Observed batch sizes |
| --- | --- | --- |
{fill_rows}
{comparison_section}

## Methodology

The direct official `Classifier.batch_classify` oracle ran in a fresh process.
The runner ran through a real loopback Uvicorn/FastAPI server and the public
JSON endpoints. RSS was sampled every 20 ms from each worker process. Latency
uses wall-clock `perf_counter`; throughput is total successful requests divided
by summed batch wall time. On Apple unified memory, process RSS and MPS
driver-allocated memory are separate accounting views and must not be added
together. Runner input usage is counted from the attention mask produced by
Fastino's compiled-schema processor for the actual encoder batch;
classification output usage is zero. The model snapshot stayed outside Git.
Inputs are deterministic synthetic length probes plus the four examples visible
in the benchmark source.
"""


def orchestrate(arguments: argparse.Namespace) -> None:
    model_path = arguments.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    digest, model_bytes, model_files = tree_digest(model_path)
    with tempfile.TemporaryDirectory(prefix="gliner-runner-benchmark-") as temporary_name:
        temporary = Path(temporary_name)
        oracle, oracle_peak = direct_oracle_measurement(model_path, temporary)
        server, rows, server_peak = run_normal_server(
            model_path,
            temporary,
            oracle,
            arguments.repetitions,
            arguments.token_targets,
            arguments.device,
            arguments.precision,
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        overload_text, overload_tokens = make_text(tokenizer, max(arguments.token_targets))
        overload, overload_peak = run_overload_server(
            model_path,
            temporary,
            overload_text,
            arguments.device,
            arguments.precision,
        )
        del tokenizer
        gc.collect()

    metadata = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host_metadata(),
        "versions": oracle["versions"],
        "torch_threads": oracle["torch_threads"],
        "device": arguments.device,
        "precision": arguments.precision,
        "oracle_device": "cpu",
        "oracle_precision": "fp32",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_content_sha256": digest,
        "model_bytes": model_bytes,
        "model_files": model_files,
        "batch_sizes": [1, 2, 4, 8],
        "token_targets": arguments.token_targets,
        "repetitions": arguments.repetitions,
        "overload_tokens": overload_tokens,
        "batch_window_ms": 4.0,
        "rss_sampling_interval_ms": 20,
    }
    result: dict[str, Any] = {
        "format_version": 2,
        "metadata": metadata,
        "cold_oracle": oracle,
        "http_runner": server,
        "overload": overload,
        "memory": {
            "oracle_peak_rss_bytes": oracle_peak,
            "http_peak_rss_bytes": server_peak,
            "overload_peak_rss_bytes": overload_peak,
            "peak_rss_bytes": max(oracle_peak, server_peak, overload_peak),
            "accelerator_peak_bytes": accelerator_peak(server, overload),
        },
        "latency_rows": rows,
    }
    if arguments.baseline_result is not None:
        baseline = json.loads(arguments.baseline_result.read_text(encoding="utf-8"))
        result["comparison"] = comparison_summary(baseline, result)
    (output_dir / "characterization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "latency.csv", rows)
    (output_dir / "REPORT.md").write_text(
        generate_report(result, rows),
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")

    oracle = subparsers.add_parser("_oracle")
    oracle.add_argument("--model-path", type=Path, required=True)
    oracle.add_argument("--events", type=Path, required=True)

    server = subparsers.add_parser("_server")
    server.add_argument("--model-path", type=Path, required=True)
    server.add_argument("--events", type=Path, required=True)
    server.add_argument("--port", type=int, required=True)
    server.add_argument("--queue-capacity", type=int, required=True)
    server.add_argument("--max-batch-size", type=int, required=True)
    server.add_argument("--batch-window-ms", type=float, required=True)
    server.add_argument("--device", required=True)

    root.add_argument("--model-path", type=Path)
    root.add_argument("--output-dir", type=Path, default=Path("benchmark-results/local"))
    root.add_argument("--repetitions", type=int, default=3)
    root.add_argument("--token-targets", type=int, nargs="+", default=[16, 64, 256])
    root.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    root.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    root.add_argument("--baseline-result", type=Path)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "_oracle":
        worker_oracle(arguments.model_path, arguments.events)
    elif arguments.command == "_server":
        worker_server(
            arguments.model_path,
            arguments.events,
            arguments.port,
            arguments.queue_capacity,
            arguments.max_batch_size,
            arguments.batch_window_ms,
            arguments.device,
        )
    else:
        if arguments.model_path is None:
            raise SystemExit("--model-path is required")
        if arguments.repetitions < 2:
            raise SystemExit("--repetitions must be at least 2")
        profile = (arguments.device, arguments.precision)
        if profile not in SUPPORTED_PROFILES:
            raise SystemExit(
                f"unsupported benchmark profile {profile}; no fallback was attempted"
            )
        if arguments.device == "mps":
            import torch

            if not torch.backends.mps.is_available():
                raise SystemExit("MPS is not available; no fallback was attempted")
        orchestrate(arguments)


if __name__ == "__main__":
    main()
