from __future__ import annotations

import asyncio
import importlib
from collections.abc import Sequence
from typing import Any

from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    BackendResult,
    ClassificationSchema,
    ClassificationTask,
    InferenceOperation,
    InferenceRequest,
    InferenceUsage,
    JsonValue,
    Precision,
    PrecisionProfile,
)
from gliner_runner.errors import BackendUnavailableError, UnsupportedCapabilityError


class PyTorchBackend:
    """Adapter around Fastino's public GLiNER2 API."""

    def __init__(self, model: str, precision: Precision, device: str) -> None:
        self._model_id = model
        self._precision = precision
        self._device = device
        self._model: Any | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        profiles = _available_precision_profiles()
        return BackendCapabilities(
            backend=BackendName.PYTORCH,
            operations=frozenset({InferenceOperation.CLASSIFY}),
            precisions=frozenset(profile.precision for profile in profiles),
            devices=frozenset(profile.device for profile in profiles),
            dynamic_batching=True,
            precision_profiles=profiles,
        )

    async def load(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            module = importlib.import_module("gliner2.classification")
        except ImportError as error:
            raise BackendUnavailableError(
                "PyTorch backend requires the 'pytorch' extra: pip install 'gliner-runner[pytorch]'"
            ) from error
        model_class = getattr(module, "Classifier", None)
        if model_class is None:
            raise BackendUnavailableError(
                "installed gliner2 package does not export classification.Classifier"
            )
        dtype: str = "float32"
        if self._precision is Precision.FP16:
            dtype = "float16"
        elif self._precision is Precision.BF16:
            dtype = "bfloat16"
        self._model = model_class.from_pretrained(
            self._model_id,
            device=self._device,
            dtype=dtype,
        ).to(device=self._device, dtype=dtype).eval()

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        if not requests:
            return []
        if self._model is None:
            raise RuntimeError("backend must be loaded before inference")
        return await asyncio.to_thread(self._infer_sync, requests)

    def _infer_sync(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        first = requests[0]
        if any(not _compatible(first, request) for request in requests[1:]):
            raise ValueError("infer_batch received incompatible requests")

        texts = [request.text for request in requests]
        schema = first.schema_
        if not isinstance(schema, ClassificationSchema):
            raise UnsupportedCapabilityError(f"unsupported schema: {type(schema).__name__}")
        model = self._model
        if model is None:
            raise RuntimeError("backend must be loaded before inference")
        module = importlib.import_module("gliner2.classification")
        oracle_schema = module.ClassificationSchema.from_dict(
            {
                "version": 3,
                "tasks": {
                    name: _oracle_task(task)
                    for name, task in schema.tasks.items()
                },
                "constraints": list(schema.constraints),
            }
        )
        config = module.ClassificationConfig(
            candidate_threshold=first.options.threshold,
            batch_size=len(texts),
            include_confidence=first.options.include_confidence,
        )
        compiled_schema = model.compile_schema(oracle_schema)
        results, input_token_counts = _batch_classify_with_usage(
            model,
            texts,
            compiled_schema,
            config,
        )
        raw = [
            result.to_dict(include_confidence=first.options.include_confidence)
            for result in results
        ]
        if (
            not isinstance(raw, list)
            or len(raw) != len(requests)
            or len(input_token_counts) != len(requests)
        ):
            raise RuntimeError("GLiNER2 backend returned an invalid batch result")
        return [
            BackendResult(
                output=_json_value(item),
                usage=InferenceUsage(input_tokens=input_tokens, output_tokens=0),
            )
            for item, input_tokens in zip(raw, input_token_counts, strict=True)
        ]

    async def close(self) -> None:
        self._model = None


def _compatible(left: InferenceRequest, right: InferenceRequest) -> bool:
    return (
        left.model == right.model
        and left.backend == right.backend
        and left.precision == right.precision
        and left.operation == right.operation
        and left.schema_ == right.schema_
        and left.options == right.options
    )


def _oracle_task(task: ClassificationTask) -> dict[str, Any]:
    payload = task.model_dump(mode="json", exclude_none=True)
    descriptions = payload.pop("label_descriptions", None)
    if descriptions is not None:
        payload["labels"] = {
            label: descriptions.get(label)
            for label in task.labels
        }
    return payload


def _batch_classify_with_usage(
    model: Any,
    texts: list[str],
    compiled_schema: Any,
    config: Any,
) -> tuple[list[Any], list[int]]:
    processor = model.scorer.processor
    original_collate = processor.collate_fn_inference
    input_token_counts: list[int] = []

    def collate_with_usage(rows: Any, *args: Any, **kwargs: Any) -> Any:
        batch = original_collate(rows, *args, **kwargs)
        counts = batch.attention_mask.sum(dim=1).tolist()
        input_token_counts.extend(int(count) for count in counts)
        return batch

    processor.collate_fn_inference = collate_with_usage
    try:
        results = model.batch_classify(texts, compiled_schema, config=config)
    finally:
        processor.collate_fn_inference = original_collate
    if len(input_token_counts) != len(texts):
        raise RuntimeError(
            "GLiNER2 preprocessing did not expose one encoded token count per request"
        )
    return list(results), input_token_counts


def _available_precision_profiles() -> frozenset[PrecisionProfile]:
    profiles = {PrecisionProfile(device="cpu", precision=Precision.FP32)}
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return frozenset(profiles)

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        profiles.add(PrecisionProfile(device="cuda", precision=Precision.FP32))
        if cuda.get_device_capability() >= (5, 3):
            profiles.add(PrecisionProfile(device="cuda", precision=Precision.FP16))
        if cuda.is_bf16_supported():
            profiles.add(PrecisionProfile(device="cuda", precision=Precision.BF16))

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        profiles.add(PrecisionProfile(device="mps", precision=Precision.FP32))
        profiles.add(PrecisionProfile(device="mps", precision=Precision.FP16))
    return frozenset(profiles)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    raise TypeError(f"backend result is not JSON serializable: {type(value).__name__}")
