from __future__ import annotations

import os

import pytest

from gliner_runner.contracts import InferenceRequest
from gliner_runner.runtime import Runtime


@pytest.mark.integration
@pytest.mark.skipif(
    "GLINER_RUNNER_INTEGRATION_REQUEST" not in os.environ,
    reason="set GLINER_RUNNER_INTEGRATION_REQUEST to an explicit JSON request",
)
async def test_real_model_request() -> None:
    request = InferenceRequest.model_validate_json(os.environ["GLINER_RUNNER_INTEGRATION_REQUEST"])
    runtime = Runtime()
    try:
        response = await runtime.infer(request)
    finally:
        await runtime.close()

    assert response.error is None
    assert response.output is not None
