from __future__ import annotations

import pytest
from pydantic import ValidationError

from gliner_runner.contracts import InferenceRequest


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
