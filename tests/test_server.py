from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import RecordingBackend
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import BackendName
from gliner_runner.runtime import Runtime, RuntimeConfig
from gliner_runner.server import create_app


def test_http_contract_and_openapi(request_factory: object) -> None:
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda _model, _precision, _device: RecordingBackend())
    runtime = Runtime(RuntimeConfig(batch_window_ms=0), registry=registry)

    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/infer",
            json=request_factory().model_dump(mode="json", by_alias=True),  # type: ignore[operator]
        )
        openapi = client.get("/openapi.json")
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["output"] == {"label": "useful"}
    assert "/v1/infer" in openapi.json()["paths"]
    assert "requests_completed_total" in metrics.text
