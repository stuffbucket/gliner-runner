from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from conftest import RecordingBackend
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import BackendName
from gliner_runner.runtime import Runtime, RuntimeConfig
from gliner_runner.server import create_app
from gliner_runner.settings import Settings


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


def test_model_inventory_and_download_required_event(
    request_factory: object, tmp_path: Path
) -> None:
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda *_args: RecordingBackend())
    runtime = Runtime(
        RuntimeConfig(model_provider_directory=tmp_path),
        registry=registry,
    )
    settings = Settings(model_provider_directory=tmp_path)

    with TestClient(create_app(runtime=runtime, settings=settings)) as client:
        models = client.get("/v1/models")
        missing = client.post(
            "/v1/infer",
            json=request_factory(model="decide-340m").model_dump(  # type: ignore[operator]
                mode="json", by_alias=True
            ),
        )
        unapproved = client.post(
            "/v1/models/decide-340m/downloads",
            json={"approved": False, "destination": str(tmp_path)},
        )

    assert models.status_code == 200
    assert models.json()["default_model"] == "decide-340m"
    assert models.json()["models"][0]["download_required"]
    assert missing.status_code == 409
    assert missing.json()["detail"]["code"] == "model_download_required"
    assert missing.json()["detail"]["event"]["destination"] == str(tmp_path.resolve())
    assert unapproved.status_code == 422
