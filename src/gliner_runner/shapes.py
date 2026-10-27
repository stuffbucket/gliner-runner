"""Request shapes of other decision APIs, translated into ordinary classification requests.

- ``POST /v1/questions`` takes the ``state`` and typed ``questions`` of TypeSafe's jev
  (https://docs.typesafe.ai/api/system-one.md, OpenAPI 0.2.0) and Microsoft-Decision-1
  (https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision,
  2026-10-09). Each question becomes one ``/v1/infer`` request with one task, so questions cannot
  affect each other.
- ``POST /v1/classify_text`` takes the arguments of GLiNER2's ``classify_text`` and
  ``batch_classify_text`` (fastino-ai/GLiNER2 v2.1.0, gliner2/inference/runtime.py). All tasks go
  into one ``/v1/infer`` request per text, as GLiNER2 scores them in one pass.

The answers are computed only from the probabilities the classification returns, so a client that
sends the translated requests itself gets the same numbers.
packages/onnx-runner/src/shapes.ts and the shape schemas in packages/onnx-runner/src/contracts.ts
are the TypeScript twin of this module and must stay in step with it: tests/fixtures/api-shapes
holds cases that both test suites run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self, TypeGuard
from uuid import UUID, uuid4

from pydantic import Field, RootModel, field_validator, model_validator

from gliner_runner.contracts import (
    BackendName,
    InferenceResponse,
    JsonValue,
    Precision,
    StrictModel,
)

# Validation messages; packages/onnx-runner/src/contracts.ts (SHAPE_MESSAGES) uses the same text.
QUESTION_TYPE = "unknown question type; use noul, choice, score or multi_select"
BOX = (
    "box questions find things in images, and the GLiNER2.5 Decide models read text only; "
    "use an image model for box questions"
)
ASSETS = (
    "images are not supported: the GLiNER2.5 Decide models read text only; send the text as state"
)
STATE = (
    "state must be one string here: the GLiNER2.5 Decide models read plain text; "
    'write structured state out as text, for example "subject: ...\\nmessage: ..."'
)
BLANK_STATE = "state must not be blank"
INSTRUCTIONS = (
    "instructions must be one non-blank string or null; write structured instructions out as text"
)
THRESHOLD = "threshold applies only to multi_select questions"
NOUL_CRITERIA = (
    "noul criteria must be an object with only 'true' and 'false', each a non-blank string or null"
)
CHOICE_CRITERIA = "criteria must map each option name to a non-blank description string or null"
CHOICE_COUNT = "a choice needs at least two options"
MULTI_COUNT = "a multi_select needs at least one option"
SCORE_CRITERIA = (
    "score criteria must be a list of at least two level descriptions, each a non-blank string"
)
SCORE_UNIQUE = "score levels must be unique"
TEXT_OR_TEXTS = "send exactly one of text or texts"
GLINER2_TASK = (
    "a task must be a list of label names, or an object with labels and any of multi_label, "
    "cls_threshold, top_k, class_act and prompt"
)
GLINER2_LABELS = (
    "labels must be a non-empty list of unique names, "
    "or an object mapping each name to a description or null"
)
GLINER2_EXAMPLES = (
    "few-shot examples are not part of this runner's protocol; leave examples out "
    "and describe the labels instead"
)
GLINER2_THRESHOLD = (
    "cls_threshold must be greater than 0 and less than 1 here; "
    "to keep the best labels whatever their score, use top_k"
)
GLINER2_TOP_K = "top_k must be an integer of at least 1"
GLINER2_ACTIVATION = "class_act must be auto, sigmoid or softmax"
GLINER2_SOFTMAX_MULTI = (
    "multi_label with class_act softmax picks labels differently in this runner; "
    "use class_act auto or sigmoid"
)
GLINER2_MULTI_LABEL = "multi_label must be true or false"
GLINER2_PROMPT = "prompt must be a non-blank string"

QUESTION_TYPES = ("noul", "choice", "score", "multi_select")
GLINER2_TASK_KEYS = ("labels", "multi_label", "cls_threshold", "top_k", "class_act", "prompt")
# The labels a noul (yes/no) question is asked with. Fastino's model card asks yes/no questions
# the same way.
NOUL_LABELS = ("yes", "no")


def _non_blank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _description(value: object) -> bool:
    return value is None or _non_blank(value)


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def state_problem(state: object) -> str | None:
    """The reason ``state`` cannot be translated, or ``None``."""
    if not isinstance(state, str):
        return STATE
    return None if state.strip() else BLANK_STATE


def question_problem(
    question_type: str,
    instructions: object,
    criteria: object,
    threshold: float | None,
) -> str | None:
    """The reason a jev or Decision-1 question cannot be translated, or ``None``."""
    if question_type == "box":
        return BOX
    if question_type not in QUESTION_TYPES:
        return QUESTION_TYPE
    if instructions is not None and not _non_blank(instructions):
        return INSTRUCTIONS
    if threshold is not None and question_type != "multi_select":
        return THRESHOLD
    if question_type == "noul":
        if criteria is None:
            return None
        if not isinstance(criteria, dict):
            return NOUL_CRITERIA
        valid = all(
            key in ("true", "false") and _description(value) for key, value in criteria.items()
        )
        return None if valid else NOUL_CRITERIA
    if question_type == "score":
        if (
            not isinstance(criteria, list)
            or len(criteria) < 2
            or not all(_non_blank(level) for level in criteria)
        ):
            return SCORE_CRITERIA
        return None if len(set(criteria)) == len(criteria) else SCORE_UNIQUE
    if not isinstance(criteria, dict) or not all(_description(v) for v in criteria.values()):
        return CHOICE_CRITERIA
    if question_type == "choice" and len(criteria) < 2:
        return CHOICE_COUNT
    if question_type == "multi_select" and len(criteria) < 1:
        return MULTI_COUNT
    return None


def _label_list(labels: list[Any]) -> bool:
    return (
        len(labels) > 0
        and all(isinstance(label, str) for label in labels)
        and len(set(labels)) == len(labels)
    )


def gliner2_task_problem(task: object) -> str | None:
    """The reason a GLiNER2 ``classify_text`` task cannot be translated, or ``None``."""
    if isinstance(task, list):
        return None if _label_list(task) else GLINER2_LABELS
    if not isinstance(task, dict) or "labels" not in task:
        return GLINER2_TASK
    if "examples" in task:
        return GLINER2_EXAMPLES
    if not all(key in GLINER2_TASK_KEYS for key in task):
        return GLINER2_TASK
    labels = task["labels"]
    if isinstance(labels, list):
        labels_valid = _label_list(labels)
    else:
        labels_valid = (
            isinstance(labels, dict)
            and len(labels) > 0
            and all(_description(value) for value in labels.values())
        )
    if not labels_valid:
        return GLINER2_LABELS
    multi = task.get("multi_label")
    if multi is None:
        multi = False
    if not isinstance(multi, bool):
        return GLINER2_MULTI_LABEL
    if "cls_threshold" in task:
        threshold = task["cls_threshold"]
        if not (_number(threshold) and 0 < threshold < 1):
            return GLINER2_THRESHOLD
    top_k = task.get("top_k")
    if top_k is not None and not (_number(top_k) and float(top_k).is_integer() and top_k >= 1):
        return GLINER2_TOP_K
    activation = task.get("class_act")
    if activation is None:
        activation = "auto"
    if activation not in ("auto", "sigmoid", "softmax"):
        return GLINER2_ACTIVATION
    if multi and activation == "softmax":
        return GLINER2_SOFTMAX_MULTI
    prompt = task.get("prompt")
    if prompt is not None and not _non_blank(prompt):
        return GLINER2_PROMPT
    return None


# --- request models ---------------------------------------------------------------------------


class Question(StrictModel):
    """One jev or Microsoft-Decision-1 question."""

    type: str
    instructions: JsonValue = None
    criteria: JsonValue = None
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def translatable(self) -> Self:
        problem = question_problem(self.type, self.instructions, self.criteria, self.threshold)
        if problem is not None:
            raise ValueError(problem)
        return self


class QuestionsRequest(StrictModel):
    """``POST /v1/questions``: jev and Decision-1 ``state`` and ``questions``, plus the runner's
    explicit model, backend and precision."""

    request_id: UUID = Field(default_factory=uuid4)
    model: str = Field(min_length=1)
    backend: BackendName
    precision: Precision
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1)
    assets: JsonValue = None

    @field_validator("model")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("state")
    @classmethod
    def state_is_text(cls, value: JsonValue) -> JsonValue:
        problem = state_problem(value)
        if problem is not None:
            raise ValueError(problem)
        return value

    @field_validator("assets")
    @classmethod
    def no_assets(cls, value: JsonValue) -> JsonValue:
        raise ValueError(ASSETS)


class Gliner2Task(RootModel[JsonValue]):
    """One GLiNER2 ``classify_text`` task: a label list, or an object with ``labels``."""

    @model_validator(mode="after")
    def translatable(self) -> Self:
        problem = gliner2_task_problem(self.root)
        if problem is not None:
            raise ValueError(problem)
        return self


class ClassifyTextRequest(StrictModel):
    """``POST /v1/classify_text``: the arguments of GLiNER2's ``classify_text`` (``text``) and
    ``batch_classify_text`` (``texts``), plus the runner's explicit model, backend and precision."""

    request_id: UUID = Field(default_factory=uuid4)
    model: str = Field(min_length=1)
    backend: BackendName
    precision: Precision
    text: str | None = Field(default=None, min_length=1)
    texts: list[str] | None = Field(default=None, min_length=1, max_length=256)
    tasks: dict[str, Gliner2Task] = Field(min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_confidence: bool = False

    @field_validator("model", "text")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("texts")
    @classmethod
    def reject_blank_texts(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not all(text.strip() for text in value):
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def text_or_texts(self) -> Self:
        if (self.text is None) == (self.texts is None):
            raise ValueError(TEXT_OR_TEXTS)
        return self


class QuestionsUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: Literal[0]


class QuestionsResponse(StrictModel):
    request_id: UUID
    model: str
    backend: BackendName
    precision: Precision
    answers: dict[str, JsonValue]
    usage: QuestionsUsage


class ClassifyTextResponse(StrictModel):
    """Has ``result`` for ``text`` and ``results`` for ``texts``, never both."""

    request_id: UUID
    model: str
    backend: BackendName
    precision: Precision
    result: dict[str, JsonValue] | None = None
    results: list[dict[str, JsonValue]] | None = None
    usage: dict[str, int]


# --- jev and Microsoft-Decision-1 questions ---------------------------------------------------


@dataclass(frozen=True)
class PlannedQuestion:
    """One question and the ``/v1/infer`` request (wire form) that answers it."""

    id: str
    type: str
    # Labels in the order they were sent: option names, or score level descriptions.
    labels: tuple[str, ...]
    request: dict[str, Any]


def _described(criteria: dict[str, Any], labels: tuple[str, ...]) -> dict[str, str] | None:
    described = {label: criteria[label] for label in labels if isinstance(criteria.get(label), str)}
    return described or None


def plan_questions(request: QuestionsRequest) -> list[PlannedQuestion]:
    """Plans one classification request per question, in the order the questions were sent."""
    planned: list[PlannedQuestion] = []
    for question_id, question in request.questions.items():
        criteria = question.criteria
        task: dict[str, Any] = {}
        options: dict[str, Any] | None = None
        if question.type == "noul":
            labels: tuple[str, ...] = NOUL_LABELS
            task["labels"] = list(labels)
            if isinstance(criteria, dict):
                descriptions = _described(
                    {"yes": criteria.get("true"), "no": criteria.get("false")}, labels
                )
                if descriptions is not None:
                    task["label_descriptions"] = descriptions
        elif question.type == "score":
            assert isinstance(criteria, list)
            labels = tuple(str(level) for level in criteria)
            task["labels"] = list(labels)
            task["ordered"] = True
        else:
            assert isinstance(criteria, dict)
            labels = tuple(criteria)
            task["labels"] = list(labels)
            descriptions = _described(criteria, labels)
            if descriptions is not None:
                task["label_descriptions"] = descriptions
            if question.type == "multi_select":
                task["min_labels"] = 0
                task["max_labels"] = None
                if question.threshold is not None:
                    task["threshold"] = question.threshold
                    # The request threshold is the floor for keeping a label as a candidate; it
                    # must not drop a label the task threshold would select.
                    options = {"threshold": question.threshold}
        if isinstance(question.instructions, str):
            task["instruction"] = question.instructions
        wire: dict[str, Any] = {
            "model": request.model,
            "backend": request.backend.value,
            "precision": request.precision.value,
            "operation": "classify",
            "text": request.state,
            "schema": {"kind": "classification", "tasks": {question_id: task}},
        }
        if options is not None:
            wire["options"] = options
        planned.append(PlannedQuestion(question_id, question.type, labels, wire))
    return planned


def choice_confidence(probabilities: list[float]) -> float:
    """jev's Choice confidence (https://docs.typesafe.ai/confidence.md, "Choice"), floored at 0."""
    n = len(probabilities)
    top = probabilities[0]
    for p in probabilities:
        if p > top:
            top = p
    return max(0.0, (top - 1 / n) / (1 - 1 / n))


def score_value(probabilities: list[float]) -> float:
    """jev's Score: each level index times its probability, added up in level order.

    The loop is explicit because ``sum()`` of floats uses compensated summation from Python 3.12
    and would differ from the TypeScript runner in the last bits.
    """
    total = 0.0
    for index, p in enumerate(probabilities):
        total += index * p
    return total


def score_confidence(probabilities: list[float]) -> float:
    """jev's Score confidence (https://docs.typesafe.ai/confidence.md, "Score"), floored at 0."""
    n = len(probabilities)
    peak = 0
    for index, p in enumerate(probabilities):
        if p > probabilities[peak]:
            peak = index
    spread = 0.0
    for index, p in enumerate(probabilities):
        spread += p * abs(index - peak)
    even = 0.0
    for index in range(n):
        even += abs(index - (n - 1) / 2)
    even /= n
    return max(0.0, 1 - spread / even)


def _task_output(
    response: InferenceResponse, task: str, labels: tuple[str, ...]
) -> tuple[Any, list[float]]:
    output = response.output
    entry = output.get(task) if isinstance(output, dict) else None
    probabilities = entry.get("probabilities") if isinstance(entry, dict) else None
    values = (
        [probabilities.get(label) for label in labels] if isinstance(probabilities, dict) else []
    )
    if len(values) != len(labels) or not all(_number(p) for p in values):
        raise RuntimeError(f"classification output has no probabilities for task {task!r}")
    assert isinstance(entry, dict)
    return entry.get("value"), [float(p) for p in values]  # type: ignore[arg-type]


def question_answer(planned: PlannedQuestion, response: InferenceResponse) -> dict[str, JsonValue]:
    """One jev-shaped answer, computed only from the classification output of its task."""
    value, probabilities = _task_output(response, planned.id, planned.labels)
    by_label: dict[str, JsonValue] = dict(zip(planned.labels, probabilities, strict=True))
    if planned.type == "noul":
        return {"type": "noul", "noul": probabilities[0]}
    if planned.type == "choice":
        return {
            "type": "choice",
            "choice": value,
            "confidence": choice_confidence(probabilities),
            "probabilities": by_label,
        }
    if planned.type == "multi_select":
        return {"type": "multi_select", "selected": list(value), "probabilities": by_label}
    return {
        "type": "score",
        "score": score_value(probabilities),
        "confidence": score_confidence(probabilities),
        "legend": {str(index): label for index, label in enumerate(planned.labels)},
        "probabilities": {str(index): p for index, p in enumerate(probabilities)},
    }


def questions_response(
    request: QuestionsRequest,
    planned: list[PlannedQuestion],
    responses: list[InferenceResponse],
) -> dict[str, JsonValue]:
    """The ``POST /v1/questions`` response body."""
    input_tokens = 0
    for response in responses:
        input_tokens += response.usage.input_tokens
    return {
        "request_id": str(request.request_id),
        "model": request.model,
        "backend": request.backend.value,
        "precision": request.precision.value,
        "answers": {
            question.id: question_answer(question, response)
            for question, response in zip(planned, responses, strict=True)
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


# --- GLiNER2 classify_text --------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedTask:
    name: str
    multi: bool
    # GLiNER2 sorts multi-label results by probability when top_k is set.
    top_k: int | None


@dataclass(frozen=True)
class ClassifyTextPlan:
    tasks: tuple[PlannedTask, ...]
    # One /v1/infer request (wire form) per text, in order.
    requests: tuple[dict[str, Any], ...]


def plan_classify_text(request: ClassifyTextRequest) -> ClassifyTextPlan:
    """Plans GLiNER2 ``classify_text`` (``text``) or ``batch_classify_text`` (``texts``).

    Every task goes into one classification request per text:

    - a list of labels, or ``labels`` as ``{name: description}``, become ``labels`` and
      ``label_descriptions``;
    - ``prompt`` becomes ``instruction``, and ``class_act`` becomes ``activation``;
    - ``multi_label: true`` becomes ``min_labels: 1, max_labels: top_k or unbounded`` with
      ``threshold: cls_threshold``: GLiNER2 keeps every label at or above ``cls_threshold`` and
      returns the best label alone when none is;
    - a single-label task keeps the runner defaults (exactly one label, the most probable). GLiNER2
      ignores ``cls_threshold`` and ``top_k`` there too;
    - the top-level ``threshold`` is GLiNER2's span threshold and has no effect on classification
      there either.
    """
    planned: list[PlannedTask] = []
    tasks: dict[str, dict[str, Any]] = {}
    floor = 0.5
    for name, wrapped in request.tasks.items():
        source = wrapped.root
        config: dict[str, Any] = {"labels": source} if isinstance(source, list) else dict(source)  # type: ignore[arg-type]
        raw_labels = config["labels"]
        task: dict[str, Any] = {}
        if isinstance(raw_labels, list):
            task["labels"] = list(raw_labels)
        else:
            labels = tuple(raw_labels)
            task["labels"] = list(labels)
            descriptions = _described(raw_labels, labels)
            if descriptions is not None:
                task["label_descriptions"] = descriptions
        multi = config.get("multi_label") is True
        top_k_raw = config.get("top_k")
        top_k = None if top_k_raw is None else int(top_k_raw)
        if multi:
            task["min_labels"] = 1
            task["max_labels"] = None if top_k is None else min(top_k, len(task["labels"]))
            threshold = config.get("cls_threshold")
            if _number(threshold):
                task["threshold"] = threshold
                floor = min(floor, float(threshold))
        activation = config.get("class_act")
        if activation in ("sigmoid", "softmax"):
            task["activation"] = activation
        prompt = config.get("prompt")
        if isinstance(prompt, str):
            task["instruction"] = prompt
        planned.append(PlannedTask(name, multi, top_k if multi else None))
        tasks[name] = task
    texts = request.texts if request.texts is not None else [request.text or ""]
    requests: list[dict[str, Any]] = []
    for text in texts:
        wire: dict[str, Any] = {
            "model": request.model,
            "backend": request.backend.value,
            "precision": request.precision.value,
            "operation": "classify",
            "text": text,
            "schema": {"kind": "classification", "tasks": tasks},
        }
        if floor < 0.5:
            # The request threshold is the floor for keeping a label as a candidate; it must not
            # drop a label that a task's cls_threshold would select.
            wire["options"] = {"threshold": floor}
        requests.append(wire)
    return ClassifyTextPlan(tuple(planned), tuple(requests))


def classify_text_result(
    plan: ClassifyTextPlan,
    response: InferenceResponse,
    include_confidence: bool,
) -> dict[str, JsonValue]:
    """One GLiNER2-shaped result (``format_results=True``), computed from the classification
    output."""
    output = response.output
    result: dict[str, JsonValue] = {}
    for task in plan.tasks:
        entry = output.get(task.name) if isinstance(output, dict) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("probabilities"), dict):
            raise RuntimeError(f"classification output has no probabilities for task {task.name!r}")
        probabilities = entry["probabilities"]
        assert isinstance(probabilities, dict)
        value = entry.get("value")
        if not task.multi:
            label = str(value)
            result[task.name] = (
                {"label": label, "confidence": probabilities[label]}
                if include_confidence
                else label
            )
            continue
        assert isinstance(value, list)
        labels = [str(label) for label in value]
        if task.top_k is not None:
            # sorted() is stable: equal probabilities keep the declared order.
            labels = sorted(labels, key=lambda label: -float(probabilities[label]))  # type: ignore[arg-type]
        result[task.name] = (
            [{"label": label, "confidence": probabilities[label]} for label in labels]
            if include_confidence
            else list(labels)
        )
    return result


def classify_text_response(
    request: ClassifyTextRequest,
    plan: ClassifyTextPlan,
    responses: list[InferenceResponse],
) -> dict[str, JsonValue]:
    """The ``POST /v1/classify_text`` response body: ``result`` for ``text``, ``results`` for
    ``texts``."""
    input_tokens = 0
    for response in responses:
        input_tokens += response.usage.input_tokens
    results = [
        classify_text_result(plan, response, request.include_confidence) for response in responses
    ]
    body: dict[str, JsonValue] = {
        "request_id": str(request.request_id),
        "model": request.model,
        "backend": request.backend.value,
        "precision": request.precision.value,
    }
    if request.texts is None:
        body["result"] = results[0]
    else:
        body["results"] = results  # type: ignore[assignment]
    body["usage"] = {"inputTokens": input_tokens, "outputTokens": 0}
    return body
