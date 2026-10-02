from __future__ import annotations

import asyncio
import os

import pytest

from gliner_runner.contracts import InferenceRequest
from gliner_runner.runtime import Runtime, RuntimeConfig


@pytest.mark.integration
@pytest.mark.skipif(
    "GLINER_RUNNER_INTEGRATION_REQUEST" not in os.environ,
    reason="set GLINER_RUNNER_INTEGRATION_REQUEST to an explicit JSON request",
)
async def test_real_model_request() -> None:
    request = InferenceRequest.model_validate_json(os.environ["GLINER_RUNNER_INTEGRATION_REQUEST"])
    longer_request = request.model_copy(update={"text": f"{request.text} with additional context"})
    runtime = Runtime(
        RuntimeConfig(
            device=os.getenv("GLINER_RUNNER_INTEGRATION_DEVICE", "cpu"),
            batch_window_ms=20,
        )
    )
    try:
        responses = await asyncio.gather(
            runtime.infer(request),
            runtime.infer(longer_request),
        )
    finally:
        await runtime.close()

    assert all(response.error is None for response in responses)
    assert all(response.output is not None for response in responses)
    assert all(response.usage.output_tokens == 0 for response in responses)
    assert responses[0].usage.input_tokens < responses[1].usage.input_tokens
    assert all(
        set(response.model_dump(mode="json", by_alias=True)["usage"])
        == {"inputTokens", "outputTokens"}
        for response in responses
    )
