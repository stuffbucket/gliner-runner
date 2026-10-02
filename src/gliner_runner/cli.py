from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from platformdirs import user_cache_path, user_config_path

from gliner_runner.contracts import InferenceRequest
from gliner_runner.model_manifest import ModelManifest, ModelStore
from gliner_runner.runtime import Runtime, RuntimeConfig
from gliner_runner.server import create_app
from gliner_runner.settings import Settings

app = typer.Typer(
    no_args_is_help=True,
    help="Run explicit, local GLiNER inference.",
)
models_app = typer.Typer(help="Inspect and validate external model manifests.")
app.add_typer(models_app, name="models")

DEFAULT_MANIFEST_DIRECTORY = user_config_path("gliner-runner") / "models"
DEFAULT_MODEL_STORE = user_cache_path("gliner-runner") / "models"
DEFAULT_PROVIDER_DIRECTORY = DEFAULT_MODEL_STORE / "providers" / "huggingface"


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Bind port.")] = 8090,
    device: Annotated[str, typer.Option(help="Explicit torch device.")] = "cpu",
    queue_capacity: Annotated[int, typer.Option(min=1)] = 256,
    max_batch_size: Annotated[int, typer.Option(min=1)] = 16,
    batch_window_ms: Annotated[float, typer.Option(min=0)] = 4.0,
    manifest_directory: Annotated[
        Path, typer.Option(help="Directory of pinned TOML model manifests.")
    ] = DEFAULT_MANIFEST_DIRECTORY,
    model_store: Annotated[
        Path, typer.Option(help="Content-addressed model cache.")
    ] = DEFAULT_MODEL_STORE,
    model_provider_directory: Annotated[
        Path, typer.Option(help="Approved Hugging Face snapshot location.")
    ] = DEFAULT_PROVIDER_DIRECTORY,
    model_idle_ttl_seconds: Annotated[
        float, typer.Option(min=0, help="Idle eviction delay; zero keeps the model warm.")
    ] = 0,
    memory_limit_bytes: Annotated[
        int | None,
        typer.Option(min=512 * 1024**2, help="Hard model-process memory ceiling."),
    ] = None,
) -> None:
    """Start the local JSON/OpenAPI server."""
    settings = Settings(
        host=host,
        port=port,
        device=device,
        queue_capacity=queue_capacity,
        max_batch_size=max_batch_size,
        batch_window_ms=batch_window_ms,
        manifest_directory=manifest_directory,
        model_store=model_store,
        model_provider_directory=model_provider_directory,
        model_idle_ttl_seconds=model_idle_ttl_seconds,
        memory_limit_bytes=(
            memory_limit_bytes
            if memory_limit_bytes is not None
            else Settings.memory_limit_bytes
        ),
    )
    uvicorn.run(create_app(settings=settings), host=host, port=port)


@app.command()
def infer(
    request_file: Annotated[
        Path | None,
        typer.Option(
            "--request",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON request file; omit to read stdin.",
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="Explicit torch device.")] = "cpu",
    manifest_directory: Annotated[
        Path, typer.Option(help="Directory of pinned TOML model manifests.")
    ] = DEFAULT_MANIFEST_DIRECTORY,
    model_store: Annotated[
        Path, typer.Option(help="Content-addressed model cache.")
    ] = DEFAULT_MODEL_STORE,
    model_provider_directory: Annotated[
        Path, typer.Option(help="Approved Hugging Face snapshot location.")
    ] = DEFAULT_PROVIDER_DIRECTORY,
) -> None:
    """Run one JSON request and print one JSON response."""
    raw = request_file.read_text() if request_file else sys.stdin.read()
    request = InferenceRequest.model_validate_json(raw)
    runtime = Runtime(
        RuntimeConfig(
            device=device,
            manifest_directory=manifest_directory,
            model_store=model_store,
            model_provider_directory=model_provider_directory,
        )
    )

    async def execute() -> str:
        try:
            response = await runtime.infer(request)
            return response.model_dump_json(by_alias=True)
        finally:
            await runtime.close()

    typer.echo(asyncio.run(execute()))


@models_app.command("verify")
def verify_manifest(
    manifest_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Validate a pinned model manifest and print its content address."""
    manifest = ModelManifest.from_toml(manifest_file)
    typer.echo(
        json.dumps(
            {"model_id": manifest.model_id, "content_id": manifest.content_id},
            sort_keys=True,
        )
    )


@models_app.command("install")
def install_model(
    manifest_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    model_store: Annotated[
        Path, typer.Option(help="Content-addressed model cache.")
    ] = DEFAULT_MODEL_STORE,
) -> None:
    """Download and verify every artifact in a pinned manifest."""
    manifest = ModelManifest.from_toml(manifest_file)
    destination = asyncio.run(ModelStore(model_store).materialize(manifest))
    typer.echo(
        json.dumps(
            {
                "model_id": manifest.model_id,
                "content_id": manifest.content_id,
                "path": str(destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
