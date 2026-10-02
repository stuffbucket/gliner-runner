from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import JsonValue as PydanticJsonValue

JsonValue: TypeAlias = PydanticJsonValue


class BackendName(StrEnum):
    PYTORCH = "pytorch"
    MLX = "mlx"
    ONNX = "onnx"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class InferenceOperation(StrEnum):
    CLASSIFY = "classify"
    EXTRACT_ENTITIES = "extract_entities"
    EXTRACT_RELATIONS = "extract_relations"
    EXTRACT_STRUCTURED = "extract_structured"


class OutputFormat(StrEnum):
    NATIVE = "native"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClassificationTask(StrictModel):
    labels: tuple[str, ...] = Field(min_length=1)
    label_descriptions: dict[str, str | None] | None = None
    min_labels: int = Field(default=1, ge=0)
    max_labels: int | None = Field(default=1, ge=0)
    ordered: bool = False
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    activation: Literal["auto", "sigmoid", "softmax"] = "auto"
    temperature: float = Field(default=1.0, gt=0)
    default: str | None = None
    instruction: str | None = None

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(labels)) != len(labels):
            raise ValueError("classification labels must be unique")
        return labels

    @model_validator(mode="after")
    def descriptions_match_labels(self) -> Self:
        if self.label_descriptions is None:
            return self
        unknown = set(self.label_descriptions) - set(self.labels)
        if unknown:
            raise ValueError(
                "label description keys must be declared labels: "
                + ", ".join(sorted(unknown))
            )
        blank = sorted(
            label
            for label, description in self.label_descriptions.items()
            if description is not None and not description.strip()
        )
        if blank:
            raise ValueError(
                "label descriptions must be non-blank strings or null: "
                + ", ".join(blank)
            )
        return self


class ClassificationSchema(StrictModel):
    kind: Literal["classification"] = "classification"
    tasks: dict[str, ClassificationTask] = Field(min_length=1)
    constraints: tuple[dict[str, JsonValue], ...] = ()


class EntitySchema(StrictModel):
    kind: Literal["entities"] = "entities"
    labels: tuple[str, ...] = Field(min_length=1)

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(labels)) != len(labels):
            raise ValueError("entity labels must be unique")
        return labels


class RelationSchema(StrictModel):
    kind: Literal["relations"] = "relations"
    relation_types: tuple[str, ...] = Field(min_length=1)


class StructuredSchema(StrictModel):
    kind: Literal["structured"] = "structured"
    definition: dict[str, JsonValue] = Field(alias="schema", min_length=1)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )


InferenceSchema: TypeAlias = Annotated[
    ClassificationSchema | EntitySchema | RelationSchema | StructuredSchema,
    Field(discriminator="kind"),
]


class InferenceOptions(StrictModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    output_format: OutputFormat = OutputFormat.NATIVE
    include_confidence: bool = True


class InferenceRequest(StrictModel):
    request_id: UUID = Field(default_factory=uuid4)
    model: str = Field(min_length=1)
    backend: BackendName
    precision: Precision
    operation: InferenceOperation
    text: str = Field(min_length=1)
    schema_: InferenceSchema = Field(alias="schema")
    options: InferenceOptions = Field(default_factory=InferenceOptions)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @field_validator("model", "text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("schema_")
    @classmethod
    def operation_matches_schema(cls, schema: InferenceSchema, info: Any) -> InferenceSchema:
        operation = info.data.get("operation")
        expected = {
            InferenceOperation.CLASSIFY: "classification",
            InferenceOperation.EXTRACT_ENTITIES: "entities",
            InferenceOperation.EXTRACT_RELATIONS: "relations",
            InferenceOperation.EXTRACT_STRUCTURED: "structured",
        }.get(operation)
        if expected is not None and schema.kind != expected:
            raise ValueError(f"{operation} requires a {expected} schema")
        return schema


class PrecisionProfile(StrictModel):
    device: str
    precision: Precision


class BackendCapabilities(StrictModel):
    backend: BackendName
    operations: frozenset[InferenceOperation]
    precisions: frozenset[Precision]
    devices: frozenset[str]
    dynamic_batching: bool
    precision_profiles: frozenset[PrecisionProfile] = frozenset()
    validated_models: frozenset[str] = frozenset()

    def supports(self, request: InferenceRequest, device: str) -> bool:
        return (
            request.operation in self.operations
            and request.precision in self.precisions
            and device in self.devices
            and (
                not self.precision_profiles
                or PrecisionProfile(device=device, precision=request.precision)
                in self.precision_profiles
            )
            and (not self.validated_models or request.model in self.validated_models)
        )


class InferenceError(StrictModel):
    code: str
    message: str
    retryable: bool = False


class Timing(StrictModel):
    queue_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)


class InferenceUsage(StrictModel):
    input_tokens: int = Field(alias="inputTokens", ge=0)
    output_tokens: Literal[0] = Field(alias="outputTokens")

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class BackendResult(StrictModel):
    output: JsonValue
    usage: InferenceUsage


class InferenceResponse(StrictModel):
    request_id: UUID
    model: str
    backend: BackendName
    precision: Precision
    output: JsonValue | None = None
    error: InferenceError | None = None
    timing: Timing
    usage: InferenceUsage

    @field_validator("error")
    @classmethod
    def exactly_one_result(cls, error: InferenceError | None, info: Any) -> InferenceError | None:
        if (info.data.get("output") is None) == (error is None):
            raise ValueError("response must contain exactly one of output or error")
        return error


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    accepting_requests: bool
