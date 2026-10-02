from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_path, user_config_path


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8090
    device: str = "cpu"
    queue_capacity: int = 256
    max_batch_size: int = 16
    batch_window_ms: float = 4.0
    manifest_directory: Path = user_config_path("gliner-runner") / "models"
    model_store: Path = user_cache_path("gliner-runner") / "models"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            host=os.getenv("GLINER_RUNNER_HOST", cls.host),
            port=_integer("GLINER_RUNNER_PORT", cls.port, minimum=1, maximum=65535),
            device=os.getenv("GLINER_RUNNER_DEVICE", cls.device),
            queue_capacity=_integer("GLINER_RUNNER_QUEUE_CAPACITY", cls.queue_capacity, minimum=1),
            max_batch_size=_integer("GLINER_RUNNER_MAX_BATCH_SIZE", cls.max_batch_size, minimum=1),
            batch_window_ms=_floating(
                "GLINER_RUNNER_BATCH_WINDOW_MS", cls.batch_window_ms, minimum=0
            ),
            manifest_directory=Path(
                os.getenv(
                    "GLINER_RUNNER_MANIFEST_DIRECTORY",
                    str(cls.manifest_directory),
                )
            ),
            model_store=Path(os.getenv("GLINER_RUNNER_MODEL_STORE", str(cls.model_store))),
        )


def _integer(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return value


def _floating(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value
