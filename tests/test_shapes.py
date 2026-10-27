from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from conftest import RecordingBackend
from gliner_runner import shapes
from gliner_runner.backends.registry import BackendRegistry
from gliner_runner.contracts import (
    BackendName,
    BackendResult,
    ClassificationSchema,
    InferenceRequest,
    InferenceResponse,
    InferenceUsage,
)
from gliner_runner.runtime import Runtime, RuntimeConfig
from gliner_runner.server import create_app
from gliner_runner.shapes import (
    ClassifyTextRequest,
    QuestionsRequest,
    choice_confidence,
    classify_text_response,
    plan_classify_text,
    plan_questions,
    questions_response,
    score_confidence,
    score_value,
)

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "api-shapes" / "cases.json").read_text(encoding="utf-8")
)


def _response(output: dict[str, Any], input_tokens: int) -> InferenceResponse:
    return InferenceResponse.model_validate(
        {
            "request_id": "955b971b-4295-4e39-9e45-9b404e32bb6e",
            "model": "decide-340m",
            "backend": "onnx",
            "precision": "fp32",
            "output": output,
            "error": None,
            "timing": {"queue_ms": 0, "inference_ms": 0},
            "usage": {"inputTokens": input_tokens, "outputTokens": 0},
        }
    )


@pytest.mark.parametrize("case", CASES["questions"], ids=lambda case: case["name"])
def test_questions_translate_like_the_shared_fixture(case: dict[str, Any]) -> None:
    request = QuestionsRequest.model_validate(case["request"])
    planned = plan_questions(request)
    assert [item.request for item in planned] == case["requests"]
    for item in planned:
        InferenceRequest.model_validate(item.request)
    responses = [_response(item["output"], item["inputTokens"]) for item in case["outputs"]]
    body = questions_response(request, planned, responses)
    # Key order is part of the wire shape (question ids and option names keep the order sent).
    assert json.dumps(body) == json.dumps(case["response"])


@pytest.mark.parametrize("case", CASES["classify_text"], ids=lambda case: case["name"])
def test_classify_text_translates_like_the_shared_fixture(case: dict[str, Any]) -> None:
    request = ClassifyTextRequest.model_validate(case["request"])
    plan = plan_classify_text(request)
    assert list(plan.requests) == case["requests"]
    for item in plan.requests:
        InferenceRequest.model_validate(item)
    responses = [_response(item["output"], item["inputTokens"]) for item in case["outputs"]]
    body = classify_text_response(request, plan, responses)
    assert json.dumps(body) == json.dumps(case["response"])


@pytest.mark.parametrize("case", CASES["invalid"], ids=lambda case: case["msg"][:60])
def test_untranslatable_requests_say_what_to_do(case: dict[str, Any]) -> None:
    model = QuestionsRequest if case["endpoint"] == "questions" else ClassifyTextRequest
    with pytest.raises(ValidationError) as caught:
        model.model_validate(case["body"])
    found = [(["body", *error["loc"]], error["msg"]) for error in caught.value.errors()]
    assert (case["loc"], case["msg"]) in found


def test_confidence_formulas_match_the_jev_documentation() -> None:
    # https://docs.typesafe.ai/confidence.md: (0.6, 0.3, 0.1) and (0.6, 0.2, 0.2) both have
    # Choice confidence 0.4; with three levels, (0, 0.5, 0.5) has Score confidence 0.25 and
    # (0.5, 0, 0.5) has 0; the bug_severity answer (0, 0.57, 0.43) has confidence about 0.35.
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    assert choice_confidence([0.5, 0.5]) == 0.0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert score_confidence([0.5, 0.0, 0.5]) == 0.0
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(0.355)
    assert score_confidence([0.0, 0.0, 1.0]) == 1.0
    # https://docs.typesafe.ai/primitives/score.md: 0 x 0.0 + 1 x 0.57 + 2 x 0.43 = 1.43.
    assert score_value([0.0, 0.57, 0.43]) == pytest.approx(1.43)


def test_missing_noul_criteria_and_null_descriptions_send_no_descriptions() -> None:
    request = QuestionsRequest.model_validate(
        {
            "model": "m",
            "backend": "pytorch",
            "precision": "fp32",
            "state": "text",
            "questions": {
                "a": {"type": "noul", "criteria": {"true": None, "false": None}},
                "b": {"type": "choice", "criteria": {"x": None, "y": None}},
            },
        }
    )
    tasks = [
        next(iter(item.request["schema"]["tasks"].values())) for item in plan_questions(request)
    ]
    assert tasks == [{"labels": ["yes", "no"]}, {"labels": ["x", "y"]}]


class ScriptedBackend(RecordingBackend):
    """Returns probabilities that depend on the text and each task's labels, so a wrong
    translation shows up as different numbers."""

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        self.batches.append(list(requests))
        results: list[BackendResult] = []
        for request in requests:
            schema = request.schema_
            assert isinstance(schema, ClassificationSchema)
            output: dict[str, Any] = {}
            for name, task in schema.tasks.items():
                raw = [
                    ((index * 37 + len(request.text) + len(name)) % 10 + 0.5) / 10.5
                    for index, _ in enumerate(task.labels)
                ]
                exclusive = task.min_labels == 1 and task.max_labels == 1
                if exclusive:
                    total = 0.0
                    for p in raw:
                        total += p
                    raw = [p / total for p in raw]
                probabilities = dict(zip(task.labels, raw, strict=True))
                best = max(task.labels, key=lambda label: probabilities[label])
                value: Any
                if exclusive:
                    value = best
                else:
                    value = [
                        label for label in task.labels if probabilities[label] > task.threshold
                    ]
                    if len(value) < task.min_labels:
                        value = [best]
                    if task.max_labels is not None:
                        value = sorted(value, key=lambda label: -probabilities[label])
                        value = [
                            label for label in task.labels if label in value[: task.max_labels]
                        ]
                output[name] = {"value": value, "confidence": 0.5, "probabilities": probabilities}
            results.append(
                BackendResult(
                    output=output,
                    usage=InferenceUsage(input_tokens=len(request.text), output_tokens=0),
                )
            )
        return results


def _client() -> TestClient:
    registry = BackendRegistry()
    registry.register(BackendName.PYTORCH, lambda _model, _precision, _device: ScriptedBackend())
    runtime = Runtime(RuntimeConfig(batch_window_ms=0), registry=registry)
    return TestClient(create_app(runtime=runtime))


QUESTIONS = {
    "model": "fastino/decide",
    "backend": "pytorch",
    "precision": "fp32",
    "state": "I have asked three times now. Can I please just talk to a real person?",
    "questions": {
        "wants_human": {"type": "noul", "instructions": "Is the customer asking for a person?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"returns": "Exchanges", "shipping": None, "billing": "Charges"},
        },
        "severity": {"type": "score", "criteria": ["Calm", "Frustrated", "Very angry"]},
        "topics": {
            "type": "multi_select",
            "criteria": {"refund": None, "person": None, "delivery": None},
            "threshold": 0.3,
        },
    },
}


def test_questions_endpoint_answers_exactly_like_the_requests_sent_by_hand() -> None:
    with _client() as client:
        response = client.post("/v1/questions", json=QUESTIONS)
        assert response.status_code == 200
        body = response.json()
        request = QuestionsRequest.model_validate(QUESTIONS)
        planned = plan_questions(request)
        by_hand = []
        for item in planned:
            reply = client.post("/v1/infer", json=item.request)
            assert reply.status_code == 200
            by_hand.append(InferenceResponse.model_validate(reply.json()))
    expected = questions_response(request, planned, by_hand)
    assert body["answers"] == expected["answers"]
    assert list(body["answers"]) == ["wants_human", "team", "severity", "topics"]
    assert body["usage"] == expected["usage"]
    assert body["usage"]["input_tokens"] == 4 * len(QUESTIONS["state"])
    first: Any = by_hand[0].output
    assert body["answers"]["wants_human"]["noul"] == first["wants_human"]["probabilities"]["yes"]


def test_classify_text_endpoint_answers_exactly_like_the_request_sent_by_hand() -> None:
    payload = {
        "model": "fastino/decide",
        "backend": "pytorch",
        "precision": "fp32",
        "texts": ["The battery is weak.", "The delivery was slow and the price too high."],
        "tasks": {
            "sentiment": ["positive", "negative", "neutral"],
            "aspects": {
                "labels": ["battery", "price", "delivery"],
                "multi_label": True,
                "cls_threshold": 0.4,
                "top_k": 2,
            },
        },
        "include_confidence": True,
    }
    with _client() as client:
        response = client.post("/v1/classify_text", json=payload)
        assert response.status_code == 200
        request = ClassifyTextRequest.model_validate(payload)
        plan = plan_classify_text(request)
        by_hand = [
            InferenceResponse.model_validate(client.post("/v1/infer", json=item).json())
            for item in plan.requests
        ]
    body = response.json()
    expected = classify_text_response(request, plan, by_hand)
    assert body["results"] == expected["results"]
    assert "result" not in body
    tokens = sum(len(text) for text in payload["texts"])
    assert body["usage"] == {"inputTokens": tokens, "outputTokens": 0}


def test_untranslatable_questions_return_422_with_the_reason() -> None:
    with _client() as client:
        response = client.post(
            "/v1/questions",
            json={**QUESTIONS, "questions": {"mug": {"type": "box", "instructions": "the mug"}}},
        )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "questions", "mug"]
    assert detail[0]["msg"] == "Value error, " + shapes.BOX


def test_openapi_lists_the_translated_shapes() -> None:
    with _client() as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/questions" in paths
    assert "/v1/classify_text" in paths
