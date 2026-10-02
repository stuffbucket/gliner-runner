from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from gliner_runner import __version__
from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    Precision,
    PrecisionProfile,
)
from gliner_runner.errors import (
    BackendUnavailableError,
    ModelDownloadRequiredError,
    ModelIntegrityError,
    ModelMemoryLimitError,
    QueueFullError,
    RunnerClosedError,
    UnsupportedCapabilityError,
)
from gliner_runner.model_inventory import (
    KNOWN_MODELS,
    DownloadJob,
    DownloadJobManager,
    DownloadRequest,
    ModelInventory,
    ModelInventoryItem,
    ModelInventoryResponse,
    RefreshResult,
    profile_guidance,
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
            model_provider_directory=configured.model_provider_directory,
            model_idle_ttl_seconds=configured.model_idle_ttl_seconds,
            memory_limit_bytes=configured.memory_limit_bytes,
        )
    )
    inventory = owned_runtime.inventory or ModelInventory(configured.model_provider_directory)
    downloads = DownloadJobManager(inventory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = owned_runtime
        yield
        await downloads.close()
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

    @app.get("/v1/models", response_model=ModelInventoryResponse, tags=["models"])
    async def models(request: Request) -> ModelInventoryResponse:
        return _models_response(
            get_runtime(request),
            inventory,
            configured.model_idle_ttl_seconds,
        )

    @app.post(
        "/v1/models/refresh",
        response_model=RefreshResult,
        tags=["models"],
    )
    async def refresh_models() -> RefreshResult:
        try:
            return await inventory.refresh()
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "code": "catalog_refresh_failed",
                    "message": str(error),
                    "retryable": True,
                },
            ) from error

    @app.post(
        "/v1/models/{model_id}/downloads",
        response_model=DownloadJob,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["models"],
    )
    async def create_download(model_id: str, body: DownloadRequest) -> DownloadJob:
        try:
            return downloads.create(model_id, body)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "invalid_download_request",
                    "message": str(error),
                    "retryable": False,
                },
            ) from error

    @app.get(
        "/v1/model-downloads/{job_id}",
        response_model=DownloadJob,
        tags=["models"],
    )
    async def download_status(job_id: UUID) -> DownloadJob:
        try:
            return downloads.get(job_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "download_job_not_found",
                    "message": str(error),
                    "retryable": False,
                },
            ) from error

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
    except ModelDownloadRequiredError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "model_download_required",
                "message": str(error),
                "retryable": False,
                "event": {
                    "type": "model_download_required",
                    "model": error.model,
                    "revision": error.revision,
                    "destination": error.destination,
                    "download_endpoint": f"/v1/models/{error.model}/downloads",
                },
            },
        ) from error
    except ModelMemoryLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "model_memory_limit",
                "message": str(error),
                "retryable": False,
                "model": error.model,
                "profile": error.profile,
                "limit_bytes": error.limit_bytes,
                "required_bytes": error.required_bytes,
                "observed_bytes": error.observed_bytes,
                "attempted_load_unloaded": error.observed_bytes is not None,
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


def _models_response(
    runtime: Runtime,
    inventory: ModelInventory,
    idle_ttl_seconds: float,
) -> ModelInventoryResponse:
    profiles = tuple(
        sorted(
            (
                profile
                for capability in runtime.capabilities()
                if capability.backend is BackendName.PYTORCH
                for profile in capability.precision_profiles
            ),
            key=lambda profile: (profile.device, profile.precision),
        )
    )
    if not profiles:
        profiles = (PrecisionProfile(device="cpu", precision=Precision.FP32),)
    preferred = next(
        (
            profile
            for preferred_precision in (
                Precision.FP16,
                Precision.BF16,
                Precision.FP32,
            )
            for profile in profiles
            if profile.device == runtime.config.device
            and profile.precision is preferred_precision
        ),
        profiles[0],
    )
    loaded = runtime.loaded_state()
    items: list[ModelInventoryItem] = []
    for spec in KNOWN_MODELS:
        local_revisions = inventory.local_revisions(spec)
        loaded_spec = inventory.spec(str(loaded["model"])) if loaded else None
        is_loaded = loaded_spec is not None and loaded_spec.model_id == spec.model_id
        loaded_profile = (
            PrecisionProfile.model_validate(loaded["profile"]) if is_loaded and loaded else None
        )
        items.append(
            ModelInventoryItem(
                model_id=spec.model_id,
                repository=spec.repository,
                description=spec.description,
                pinned_revision=spec.pinned_revision,
                discovered_revision=inventory.discovered_revision(spec),
                parameter_count=spec.parameter_count,
                local_revisions=local_revisions,
                available_locally=spec.pinned_revision in local_revisions,
                loaded=is_loaded,
                loaded_profile=loaded_profile,
                supported_profiles=profiles,
                default_profile=preferred,
                profile_guidance={
                    f"{profile.device}/{profile.precision}": profile_guidance(spec, profile)
                    for profile in profiles
                },
                download_destination=str(inventory.provider_root),
                download_required=spec.pinned_revision not in local_revisions,
            )
        )
    return ModelInventoryResponse(
        default_model="decide-340m",
        device=runtime.config.device,
        idle_ttl_seconds=idle_ttl_seconds,
        memory_limit_bytes=runtime.memory_limit_bytes,
        discovered_models=inventory.discovered_models(),
        models=tuple(items),
    )


app = create_app()
