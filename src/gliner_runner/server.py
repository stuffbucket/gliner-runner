from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from gliner_runner import __version__
from gliner_runner.contracts import (
    BackendCapabilities,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
)
from gliner_runner.errors import (
    BackendUnavailableError,
    ModelIntegrityError,
    QueueFullError,
    RunnerClosedError,
    UnsupportedCapabilityError,
)
from gliner_runner.runtime import Runtime, RuntimeConfig
from gliner_runner.settings import Settings


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[InferenceRequest] = Field(min_length=1, max_length=256)


class BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responses: list[InferenceResponse]


def create_app(
    *,
    settings: Settings | None = None,
    runtime: Runtime | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    owned_runtime = runtime or Runtime(
        RuntimeConfig(
            device=configured.device,
            queue_capacity=configured.queue_capacity,
            max_batch_size=configured.max_batch_size,
            batch_window_ms=configured.batch_window_ms,
            manifest_directory=configured.manifest_directory,
            model_store=configured.model_store,
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = owned_runtime
        yield
        await owned_runtime.close()

    app = FastAPI(
        title="GLiNER Runner",
        summary="Local, explicitly configured GLiNER inference",
        version=__version__,
        lifespan=lifespan,
    )

    def get_runtime(request: Request) -> Runtime:
        value = getattr(request.app.state, "runtime", owned_runtime)
        if not isinstance(value, Runtime):
            raise RuntimeError("runtime has not been configured")
        return value

    @app.get("/healthz", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        accepting = owned_runtime.accepting
        return HealthResponse(
            status="ok" if accepting else "degraded",
            accepting_requests=accepting,
        )

    @app.get(
        "/v1/capabilities",
        response_model=list[BackendCapabilities],
        tags=["inference"],
    )
    async def capabilities(request: Request) -> list[BackendCapabilities]:
        return get_runtime(request).capabilities()

    @app.post(
        "/v1/infer",
        response_model=InferenceResponse,
        responses={
            409: {"description": "Unsupported explicit configuration"},
            429: {"description": "Queue capacity exhausted"},
        },
        tags=["inference"],
    )
    async def infer(
        body: InferenceRequest,
        request: Request,
    ) -> InferenceResponse:
        return await _infer(get_runtime(request), body)

    @app.post(
        "/v1/batch",
        response_model=BatchResponse,
        responses={
            409: {"description": "Unsupported explicit configuration"},
            429: {"description": "Queue capacity exhausted"},
        },
        tags=["inference"],
    )
    async def batch(
        body: BatchRequest,
        request: Request,
    ) -> BatchResponse:
        engine = get_runtime(request)
        responses = await asyncio.gather(*(_infer(engine, request) for request in body.requests))
        return BatchResponse(responses=list(responses))

    @app.get("/metrics", include_in_schema=False, tags=["operations"])
    async def metrics(request: Request) -> Response:
        return Response(
            get_runtime(request).metrics.prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    return app


async def _infer(runtime: Runtime, request: InferenceRequest) -> InferenceResponse:
    try:
        return await runtime.infer(request)
    except QueueFullError as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "queue_full", "message": str(error), "retryable": True},
            headers={"Retry-After": "1"},
        ) from error
    except (BackendUnavailableError, UnsupportedCapabilityError) as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "unsupported_configuration",
                "message": str(error),
                "retryable": False,
            },
        ) from error
    except ModelIntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "model_integrity", "message": str(error), "retryable": False},
        ) from error
    except RunnerClosedError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "runtime_closed", "message": str(error), "retryable": True},
        ) from error


app = create_app()
