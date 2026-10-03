"""Local, backend-neutral GLiNER inference runtime."""

from gliner_runner.contracts import (
    BackendCapabilities,
    BackendName,
    ClassificationSchema,
    ClassificationTask,
    EntitySchema,
    InferenceError,
    InferenceOperation,
    InferenceOptions,
    InferenceRequest,
    InferenceResponse,
    InferenceUsage,
    Precision,
    PrecisionProfile,
    RelationSchema,
    StructuredSchema,
)

__all__ = [
    "BackendCapabilities",
    "BackendName",
    "ClassificationSchema",
    "ClassificationTask",
    "EntitySchema",
    "InferenceError",
    "InferenceOperation",
    "InferenceOptions",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceUsage",
    "Precision",
    "PrecisionProfile",
    "RelationSchema",
    "StructuredSchema",
]

__version__ = "0.2.1"  # x-release-please-version
