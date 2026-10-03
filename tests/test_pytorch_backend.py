from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gliner_runner.backends.pytorch import (
    PyTorchBackend,
    _available_precision_profiles,
    _compatible,
)
from gliner_runner.contracts import (
    BackendName,
    InferenceOperation,
    InferenceOptions,
    Precision,
    PrecisionProfile,
)


class FakeOracleSchema:
    received: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> object:
        assert set(value) == {"version", "tasks", "constraints"}
        assert value["version"] == 3
        assert value["constraints"] == []
        assert set(value["tasks"]) == {"label"}
        cls.received = value
        return object()


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self, *, include_confidence: bool) -> dict[str, object]:
        return {
            "label": {
                "value": "useful",
                "confidence": 0.9 if include_confidence else None,
            }
        }


class FakeClassifier:
    instance: FakeClassifier | None = None
    loads = 0

    def __init__(self) -> None:
        self.call: tuple[list[str], object, object] | None = None
        self.load_kwargs: dict[str, object] = {}
        self.to_kwargs: dict[str, object] = {}
        self.scorer = SimpleNamespace(processor=FakeProcessor())

    @classmethod
    def from_pretrained(cls, _path: str, **kwargs: object) -> FakeClassifier:
        cls.loads += 1
        cls.instance = cls()
        cls.instance.load_kwargs = kwargs
        return cls.instance

    def to(self, **kwargs: object) -> FakeClassifier:
        self.to_kwargs = kwargs
        return self

    def eval(self) -> FakeClassifier:
        return self

    def batch_classify(
        self, texts: list[str], schema: object, *, config: object
    ) -> list[FakeResult]:
        self.call = (texts, schema, config)
        self.scorer.processor.collate_fn_inference(
            [(text, schema.build()) for text in texts]
        )
        return [FakeResult(text) for text in texts]

    def compile_schema(self, _schema: object) -> FakeCompiledSchema:
        return FakeCompiledSchema()


class FakeConfig:
    def __init__(self, **kwargs: object) -> None:
        assert set(kwargs) == {
            "candidate_threshold",
            "batch_size",
            "include_confidence",
        }
        assert kwargs["candidate_threshold"] == 0.5
        assert kwargs["include_confidence"] is True
        self.kwargs = kwargs


class FakeCompiledSchema:
    def build(self) -> dict[str, object]:
        return {"compiled": True}


class FakeVector:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


class FakeAttentionMask:
    def __init__(self, counts: list[int]) -> None:
        self._counts = counts

    def sum(self, *, dim: int) -> FakeVector:
        assert dim == 1
        return FakeVector(self._counts)


class FakeProcessor:
    def collate_fn_inference(
        self, rows: list[tuple[str, dict[str, object]]], **_kwargs: object
    ) -> SimpleNamespace:
        assert all(schema == {"compiled": True} for _text, schema in rows)
        return SimpleNamespace(
            attention_mask=FakeAttentionMask([len(text) + 4 for text, _schema in rows])
        )


async def test_adapter_uses_fastino_public_batch_classify(
    monkeypatch: Any, request_factory: Any
) -> None:
    FakeClassifier.loads = 0
    module = ModuleType("gliner2.classification")
    module.Classifier = FakeClassifier  # type: ignore[attr-defined]
    module.ClassificationSchema = FakeOracleSchema  # type: ignore[attr-defined]
    module.ClassificationConfig = FakeConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2.classification", module)
    backend = PyTorchBackend("/models/pinned", Precision.FP32, "cpu")
    await backend.load()
    descriptions = {
        "useful": "Actionable content that should be retained",
    }
    requests = [
        request_factory(text="one", label_descriptions=descriptions),
        request_factory(text="longer", label_descriptions=descriptions),
    ]

    outputs = await backend.infer_batch(requests)
    await backend.load()

    assert len(outputs) == 2
    assert FakeClassifier.loads == 1
    assert [output.usage.input_tokens for output in outputs] == [7, 10]
    assert [output.usage.output_tokens for output in outputs] == [0, 0]
    assert [output.output["label"]["value"] for output in outputs] == [  # type: ignore[index]
        "useful",
        "useful",
    ]
    assert outputs[0].output == {
        "label": {"value": "useful", "confidence": 0.9}
    }
    assert FakeClassifier.instance is not None
    assert FakeClassifier.instance.call is not None
    assert FakeClassifier.instance.call[0] == ["one", "longer"]
    assert FakeClassifier.instance.load_kwargs == {"device": "cpu", "dtype": "float32"}
    assert FakeClassifier.instance.to_kwargs == {"device": "cpu", "dtype": "float32"}
    assert FakeOracleSchema.received is not None
    assert FakeOracleSchema.received["tasks"]["label"]["labels"] == {
        "useful": descriptions["useful"],
        "spam": None,
    }
    config = FakeClassifier.instance.call[2]
    assert isinstance(config, FakeConfig)
    assert config.kwargs["batch_size"] == 2
    processor = FakeClassifier.instance.scorer.processor
    assert callable(processor.collate_fn_inference)
    restored_batch = processor.collate_fn_inference(
        [("restored", {"compiled": True})]
    )
    assert restored_batch.attention_mask.sum(dim=1).tolist() == [12]

    labels_only = await backend.infer_batch([request_factory(text="compatible")])

    assert labels_only[0].usage.input_tokens == 14
    assert FakeOracleSchema.received is not None
    assert FakeOracleSchema.received["tasks"]["label"] == {
        "labels": ["useful", "spam"],
        "min_labels": 1,
        "max_labels": 1,
        "ordered": False,
        "threshold": 0.5,
        "activation": "auto",
        "temperature": 1.0,
    }

    incompatible = request_factory(model="other/model")
    with pytest.raises(ValueError, match="incompatible requests"):
        await backend.infer_batch([requests[0], incompatible])
    await backend.close()
    with pytest.raises(RuntimeError, match="loaded before inference"):
        await backend.infer_batch([requests[0]])


async def test_empty_batch_does_not_require_loaded_model() -> None:
    backend = PyTorchBackend("/models/pinned", Precision.FP32, "cpu")

    assert await backend.infer_batch([]) == []


@pytest.mark.parametrize(
    "update",
    [
        {"model": "other/model"},
        {"backend": BackendName.MLX},
        {"precision": Precision.FP16},
        {"operation": InferenceOperation.EXTRACT_ENTITIES},
        {"schema_": None},
        {"options": InferenceOptions(threshold=0.9)},
    ],
)
def test_batch_compatibility_requires_every_grouping_field(
    request_factory: Any,
    update: dict[str, object],
) -> None:
    request = request_factory()

    assert _compatible(request, request)
    assert not _compatible(request, request.model_copy(update=update))


def test_capabilities_adapt_to_available_mps_profiles(
    monkeypatch: Any,
    request_factory: Any,
) -> None:
    monkeypatch.setattr(
        "gliner_runner.backends.pytorch._available_precision_profiles",
        lambda: frozenset(
            {
                PrecisionProfile(device="cpu", precision=Precision.FP32),
                PrecisionProfile(device="mps", precision=Precision.FP16),
                PrecisionProfile(device="mps", precision=Precision.FP32),
            }
        ),
    )
    capabilities = PyTorchBackend("/models/pinned", Precision.FP16, "mps").capabilities

    assert capabilities.supports(request_factory(precision=Precision.FP16), "mps")
    assert capabilities.supports(request_factory(precision=Precision.FP32), "mps")
    assert not capabilities.supports(request_factory(precision=Precision.BF16), "mps")


def test_available_profiles_follow_cuda_architecture(monkeypatch: Any) -> None:
    torch = ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability() -> tuple[int, int]:
            return (8, 0)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.cuda = FakeCuda()  # type: ignore[attr-defined]
    torch.backends = type("FakeBackends", (), {"mps": FakeMps()})()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    profiles = _available_precision_profiles()

    assert profiles == frozenset(
        {
            PrecisionProfile(device="cpu", precision=Precision.FP32),
            PrecisionProfile(device="cuda", precision=Precision.FP32),
            PrecisionProfile(device="cuda", precision=Precision.FP16),
            PrecisionProfile(device="cuda", precision=Precision.BF16),
        }
    )

    torch.cuda.get_device_capability = lambda: (5, 2)  # type: ignore[attr-defined,method-assign]
    torch.cuda.is_bf16_supported = lambda: False  # type: ignore[attr-defined,method-assign]
    limited_profiles = _available_precision_profiles()

    assert limited_profiles == frozenset(
        {
            PrecisionProfile(device="cpu", precision=Precision.FP32),
            PrecisionProfile(device="cuda", precision=Precision.FP32),
        }
    )


def test_available_profiles_without_torch_are_cpu_only(monkeypatch: Any) -> None:
    def missing_torch(_name: str) -> ModuleType:
        raise ImportError

    monkeypatch.setattr(
        "gliner_runner.backends.pytorch.importlib.import_module",
        missing_torch,
    )

    assert _available_precision_profiles() == frozenset(
        {PrecisionProfile(device="cpu", precision=Precision.FP32)}
    )


def test_available_profiles_include_both_mps_precisions(monkeypatch: Any) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _available_precision_profiles() == frozenset(
        {
            PrecisionProfile(device="cpu", precision=Precision.FP32),
            PrecisionProfile(device="mps", precision=Precision.FP32),
            PrecisionProfile(device="mps", precision=Precision.FP16),
        }
    )


def test_available_profiles_with_unavailable_accelerators_are_cpu_only(
    monkeypatch: Any,
) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _available_precision_profiles() == frozenset(
        {PrecisionProfile(device="cpu", precision=Precision.FP32)}
    )
