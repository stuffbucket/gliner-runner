"""Golden end-to-end outputs from the official PyTorch path for ONNX parity checks.

The reference runs a request corpus through :class:`PyTorchBackend` (Fastino GLiNER2 2.1.0
``Classifier.batch_classify``) at CPU/FP32 and records exactly what the runner returns for
each request: the output and usage, or the structured error the HTTP layer would emit. A
Node ONNX server is later compared against these files case by case.

Importing this module needs neither PyTorch nor GLiNER2; the backend imports them lazily.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gliner_runner.backends.pytorch import PyTorchBackend
from gliner_runner.contracts import (
    BackendName,
    BackendResult,
    InferenceOperation,
    InferenceRequest,
    Precision,
)
from gliner_runner.errors import (
    BackendUnavailableError,
    InvalidSchemaError,
    ModelIntegrityError,
    UnsupportedCapabilityError,
)
from gliner_runner.onnx_prepare import resolve_curated_model
from gliner_runner.snapshot import validate_model_snapshot

CORPUS_FORMAT_VERSION = 1
REFERENCE_FORMAT_VERSION = 1
REFERENCE_KIND = "gliner-runner-onnx-reference-outputs"
DEFAULT_REPEATS = 2


class ReferenceBackend(Protocol):
    async def load(self) -> None: ...

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    categories: tuple[str, ...]
    batch_group: str | None
    language: str | None
    request: InferenceRequest


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    cases: tuple[CorpusCase, ...]
    sha256: str


def load_corpus(path: Path) -> Corpus:
    """Parse and validate a request corpus file."""
    raw = path.expanduser().read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"corpus is not valid JSON: {path}") from error
    if not isinstance(document, dict) or document.get("format_version") != CORPUS_FORMAT_VERSION:
        raise ValueError(f"corpus format_version must be {CORPUS_FORMAT_VERSION}")
    corpus_id = document.get("corpus_id")
    cases_value = document.get("cases")
    if not isinstance(corpus_id, str) or not corpus_id:
        raise ValueError("corpus_id must be a nonempty string")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("corpus must contain a nonempty cases list")
    cases: list[CorpusCase] = []
    seen: set[str] = set()
    for item in cases_value:
        if not isinstance(item, dict):
            raise ValueError("corpus cases must be objects")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("corpus case_id values must be unique nonempty strings")
        seen.add(case_id)
        categories = item.get("categories", [])
        batch_group = item.get("batch_group")
        language = item.get("language")
        if (
            not isinstance(categories, list)
            or not all(isinstance(value, str) for value in categories)
            or not (batch_group is None or isinstance(batch_group, str))
            or not (language is None or isinstance(language, str))
        ):
            raise ValueError(f"corpus case {case_id!r} has invalid metadata")
        request = InferenceRequest.model_validate(item.get("request"))
        if request.operation is not InferenceOperation.CLASSIFY:
            raise ValueError(f"corpus case {case_id!r} must be a classify request")
        cases.append(
            CorpusCase(
                case_id=case_id,
                categories=tuple(categories),
                batch_group=batch_group,
                language=language,
                request=request,
            )
        )
    return Corpus(corpus_id=corpus_id, cases=tuple(cases), sha256=hashlib.sha256(raw).hexdigest())


def structured_error(error: BaseException) -> tuple[int, dict[str, Any]]:
    """Map a backend failure to the HTTP status and error body the runner server emits."""
    if isinstance(error, InvalidSchemaError):
        return 422, {"code": "invalid_schema", "message": str(error), "retryable": False}
    if isinstance(error, BackendUnavailableError | UnsupportedCapabilityError):
        return 409, {
            "code": "unsupported_configuration",
            "message": str(error),
            "retryable": False,
        }
    if isinstance(error, ModelIntegrityError):
        return 422, {"code": "model_integrity", "message": str(error), "retryable": False}
    return 500, {
        "code": "internal_error",
        "message": str(error),
        "retryable": False,
        "exception_type": type(error).__name__,
    }


def _success(result: BackendResult) -> dict[str, Any]:
    return {
        "status": "success",
        "http_status": 200,
        "output": result.output,
        "usage": result.usage.model_dump(mode="json", by_alias=True),
    }


def _failure(error: BaseException) -> dict[str, Any]:
    http_status, body = structured_error(error)
    return {"status": "error", "http_status": http_status, "error": body}


async def _run_batch(
    backend: ReferenceBackend, requests: Sequence[InferenceRequest]
) -> list[dict[str, Any]]:
    try:
        results = await backend.infer_batch(requests)
        if len(results) != len(requests):
            raise RuntimeError("backend returned the wrong number of results")
    except Exception as error:
        return [_failure(error) for _ in requests]
    return [_success(result) for result in results]


def compare_json(left: Any, right: Any) -> tuple[bool, float]:
    """Return ``(same_structure, max_abs_float_difference)`` for two JSON values."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right, 0.0
    if isinstance(left, int | float) and isinstance(right, int | float):
        if left == right:
            return True, 0.0
        difference = abs(float(left) - float(right))
        return True, difference if math.isfinite(difference) else math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if list(left) != list(right):
            return False, 0.0
        maximum, same = 0.0, True
        for key in left:
            child_same, child = compare_json(left[key], right[key])
            same, maximum = same and child_same, max(maximum, child)
        return same, maximum
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        if len(left) != len(right):
            return False, 0.0
        maximum, same = 0.0, True
        for left_item, right_item in zip(left, right, strict=True):
            child_same, child = compare_json(left_item, right_item)
            same, maximum = same and child_same, max(maximum, child)
        return same, maximum
    return left == right and type(left) is type(right), 0.0


def _reference_request(case: CorpusCase, model_id: str) -> InferenceRequest:
    return case.request.model_copy(
        update={
            "model": model_id,
            "backend": BackendName.PYTORCH,
            "precision": Precision.FP32,
        }
    )


async def collect_reference(
    backend: ReferenceBackend,
    *,
    model_id: str,
    corpus: Corpus,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Run every case singly (repeated) and every batch group together."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    requests = {case.case_id: _reference_request(case, model_id) for case in corpus.cases}
    singleton: dict[str, dict[str, Any]] = {}
    for case in corpus.cases:
        runs = [await _run_batch(backend, [requests[case.case_id]]) for _ in range(repeats)]
        first = runs[0][0]
        singleton[case.case_id] = {
            "case_id": case.case_id,
            "categories": list(case.categories),
            "language": case.language,
            "batch_group": case.batch_group,
            **first,
            "repeats": repeats,
            "repeat_identical": all(run[0] == first for run in runs[1:]),
        }
    groups: dict[str, list[CorpusCase]] = {}
    for case in corpus.cases:
        if case.batch_group is not None:
            groups.setdefault(case.batch_group, []).append(case)
    batches: list[dict[str, Any]] = []
    for group, members in groups.items():
        outcomes = await _run_batch(backend, [requests[member.case_id] for member in members])
        entries: list[dict[str, Any]] = []
        for member, outcome in zip(members, outcomes, strict=True):
            alone = {key: singleton[member.case_id][key] for key in outcome}
            same, difference = compare_json(outcome, alone)
            entries.append(
                {
                    "case_id": member.case_id,
                    **outcome,
                    "matches_singleton_structure": same,
                    "matches_singleton_exactly": outcome == alone,
                    "max_abs_float_difference_vs_singleton": difference,
                }
            )
        batches.append(
            {
                "batch_group": group,
                "case_ids": [member.case_id for member in members],
                "results": entries,
            }
        )
    return {
        "singleton": [singleton[case.case_id] for case in corpus.cases],
        "batches": batches,
    }


def reference_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("gliner2", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _default_backend(snapshot: Path) -> ReferenceBackend:
    return PyTorchBackend(str(snapshot), Precision.FP32, "cpu")


def run_reference(
    model: str,
    *,
    snapshot: Path,
    corpus_path: Path,
    output: Path,
    revision: str | None = None,
    repeats: int = DEFAULT_REPEATS,
    backend_factory: Callable[[Path], ReferenceBackend] = _default_backend,
) -> Path:
    """Write golden PyTorch outputs for a corpus to a new JSON file."""
    spec, pinned = resolve_curated_model(model, revision)
    resolved_snapshot = validate_model_snapshot(snapshot, spec.repository, pinned)
    corpus = load_corpus(corpus_path)
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"reference output already exists: {destination}")

    async def execute() -> dict[str, Any]:
        backend = backend_factory(resolved_snapshot)
        await backend.load()
        try:
            return await collect_reference(
                backend, model_id=spec.model_id, corpus=corpus, repeats=repeats
            )
        finally:
            await backend.close()

    collected = asyncio.run(execute())
    document = {
        "format_version": REFERENCE_FORMAT_VERSION,
        "kind": REFERENCE_KIND,
        "model_id": spec.model_id,
        "repository": spec.repository,
        "revision": pinned,
        "corpus_id": corpus.corpus_id,
        "corpus_sha256": corpus.sha256,
        "reference": {
            "backend": "pytorch",
            "device": "cpu",
            "precision": "fp32",
            "path": "gliner_runner.backends.pytorch.PyTorchBackend",
            "versions": reference_versions(),
        },
        **collected,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return destination
