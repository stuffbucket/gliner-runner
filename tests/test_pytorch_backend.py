from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from gliner_runner.backends.pytorch import PyTorchBackend
from gliner_runner.contracts import Precision


class FakeOracleSchema:
    received: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> object:
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

    def __init__(self) -> None:
        self.call: tuple[list[str], object, object] | None = None
        self.load_kwargs: dict[str, object] = {}
        self.to_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, _path: str, **kwargs: object) -> FakeClassifier:
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
        return [FakeResult(text) for text in texts]


class FakeConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


async def test_adapter_uses_fastino_public_batch_classify(
    monkeypatch: Any, request_factory: Any
) -> None:
    module = ModuleType("gliner2.classification")
    module.Classifier = FakeClassifier  # type: ignore[attr-defined]
    module.ClassificationSchema = FakeOracleSchema  # type: ignore[attr-defined]
    module.ClassificationConfig = FakeConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2.classification", module)
    backend = PyTorchBackend("/models/pinned", Precision.FP32, "cpu")
    await backend.load()
    requests = [request_factory(text="one"), request_factory(text="two")]

    outputs = await backend.infer_batch(requests)

    assert len(outputs) == 2
    assert FakeClassifier.instance is not None
    assert FakeClassifier.instance.call is not None
    assert FakeClassifier.instance.call[0] == ["one", "two"]
    assert FakeClassifier.instance.load_kwargs == {"device": "cpu", "dtype": "float32"}
    assert FakeClassifier.instance.to_kwargs == {"device": "cpu", "dtype": "float32"}
    assert FakeOracleSchema.received is not None
    assert FakeOracleSchema.received["tasks"]["label"]["labels"] == ["useful", "spam"]
    config = FakeClassifier.instance.call[2]
    assert isinstance(config, FakeConfig)
    assert config.kwargs["batch_size"] == 2


def test_capabilities_advertise_only_validated_mps_fp16(request_factory: Any) -> None:
    capabilities = PyTorchBackend("/models/pinned", Precision.FP16, "mps").capabilities

    assert capabilities.supports(request_factory(precision=Precision.FP16), "mps")
    assert not capabilities.supports(request_factory(precision=Precision.FP32), "mps")
