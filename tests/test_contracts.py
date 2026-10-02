from __future__ import annotations

import pytest
from pydantic import ValidationError

from gliner_runner.contracts import ClassificationTask, InferenceRequest


def test_request_round_trip_uses_public_schema_name(request_factory: object) -> None:
    request = request_factory()  # type: ignore[operator]
    encoded = request.model_dump_json(by_alias=True)

    assert '"schema"' in encoded
    assert '"schema_"' not in encoded
    assert InferenceRequest.model_validate_json(encoded) == request


def test_request_rejects_operation_schema_mismatch(request_factory: object) -> None:
    request = request_factory()  # type: ignore[operator]
    body = request.model_dump(mode="json", by_alias=True)
    body["operation"] = "extract_entities"

    with pytest.raises(ValidationError, match="requires a entities schema"):
        InferenceRequest.model_validate(body)


def test_contract_rejects_unknown_fields(request_factory: object) -> None:
    request = request_factory()  # type: ignore[operator]
    body = request.model_dump(mode="json", by_alias=True)
    body["allow_fallback"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InferenceRequest.model_validate(body)


def test_classification_task_accepts_partial_descriptions_and_null() -> None:
    task = ClassificationTask(
        labels=("urgent", "routine", "unknown"),
        label_descriptions={
            "urgent": "Requires immediate handling",
            "unknown": None,
        },
    )

    assert task.label_descriptions == {
        "urgent": "Requires immediate handling",
        "unknown": None,
    }
    assert "label_descriptions" in task.model_dump(mode="json")


def test_classification_task_rejects_undeclared_description_keys() -> None:
    with pytest.raises(ValidationError, match="must be declared labels: missing"):
        ClassificationTask(
            labels=("urgent", "routine"),
            label_descriptions={"missing": "Not declared"},
        )


def test_classification_task_rejects_blank_descriptions() -> None:
    with pytest.raises(ValidationError, match="non-blank strings or null: urgent"):
        ClassificationTask(
            labels=("urgent", "routine"),
            label_descriptions={"urgent": "  "},
        )
