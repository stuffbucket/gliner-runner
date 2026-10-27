"""Writes tests/fixtures/api-shapes/cases.json from hand-written inputs and the Python translation.

Run from the repository root:

    python scripts/api_shapes_fixture.py tests/fixtures/api-shapes/cases.json

with the gliner_runner package from this checkout importable (PYTHONPATH=src). The TypeScript
tests then check that their translation produces the same requests and the same answers,
number for number.
"""

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from gliner_runner.contracts import InferenceRequest, InferenceResponse
from gliner_runner.shapes import (
    ClassifyTextRequest,
    QuestionsRequest,
    classify_text_response,
    plan_classify_text,
    plan_questions,
    questions_response,
)

REQUEST_ID = "955b971b-4295-4e39-9e45-9b404e32bb6e"
BASE = {"request_id": REQUEST_ID, "model": "decide-340m", "backend": "onnx", "precision": "fp32"}


def native(tasks: dict) -> dict:
    """A native classification output: {task: {value, confidence, probabilities}} plus _meta."""
    out = {}
    for name, (value, probabilities) in tasks.items():
        out[name] = {"value": value, "confidence": 0.5, "probabilities": probabilities}
    out["_meta"] = {
        "feasible": True,
        "decoder": "independent",
        "exact": True,
        "objective": 0.0,
        "violations": [],
    }
    return out


def response(output: dict, tokens: int) -> InferenceResponse:
    return InferenceResponse.model_validate(
        {
            "request_id": REQUEST_ID,
            "model": "decide-340m",
            "backend": "onnx",
            "precision": "fp32",
            "output": output,
            "error": None,
            "timing": {"queue_ms": 0, "inference_ms": 0},
            "usage": {"inputTokens": tokens, "outputTokens": 0},
        }
    )


QUESTION_CASES = [
    {
        "name": "every question type, with integer-like ids and option names",
        "request": {
            **BASE,
            "state": "I have asked three times now. Can I please just talk to a real person?",
            "questions": {
                "is_human_escalation": {
                    "type": "noul",
                    "instructions": "Is the customer asking for a human agent?",
                },
                "is_repeat_contact": {
                    "type": "noul",
                    "instructions": "Has the customer contacted support about this before?",
                    "criteria": {"true": "Mentions a prior attempt or ticket", "false": None},
                },
                "10": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "returns": "Exchanges, wrong or damaged items",
                        "shipping": None,
                        "billing": "Charges",
                    },
                },
                "2": {
                    "type": "choice",
                    "criteria": {"10": "Ten", "2": "Two", "x": None},
                },
                "bug_severity": {
                    "type": "score",
                    "instructions": "How severe is the reported issue?",
                    "criteria": [
                        "Cosmetic; no impact to functionality",
                        "Broken or degraded feature, but workaround exists",
                        "Blocking issue; no workaround exists",
                    ],
                },
                "topics": {
                    "type": "multi_select",
                    "instructions": "Which topics does the message raise?",
                    "criteria": {"refund": None, "human agent": "Wants a person", "delivery": None},
                    "threshold": 0.3,
                },
                "tags": {"type": "multi_select", "criteria": {"a": None, "b": None}},
            },
        },
        "outputs": [
            ({"is_human_escalation": ("yes", {"yes": 0.99, "no": 0.010000000000000009})}, 30),
            ({"is_repeat_contact": ("yes", {"yes": 0.93, "no": 0.06999999999999995})}, 41),
            ({"10": ("returns", {"returns": 0.61, "shipping": 0.04, "billing": 0.35})}, 52),
            ({"2": ("10", {"10": 0.5, "2": 0.25, "x": 0.25})}, 23),
            (
                {
                    "bug_severity": (
                        "Broken or degraded feature, but workaround exists",
                        {
                            "Cosmetic; no impact to functionality": 0.0,
                            "Broken or degraded feature, but workaround exists": 0.57,
                            "Blocking issue; no workaround exists": 0.43,
                        },
                    )
                },
                60,
            ),
            (
                {
                    "topics": (
                        ["human agent", "delivery"],
                        {"refund": 0.1, "human agent": 0.9, "delivery": 0.31},
                    )
                },
                44,
            ),
            ({"tags": ([], {"a": 0.2, "b": 0.4})}, 12),
        ],
    },
    {
        "name": "a score split between opposite ends has zero confidence",
        "request": {
            **BASE,
            "state": "Some text",
            "questions": {
                "level": {"type": "score", "criteria": ["low", "middle", "high"]},
            },
        },
        "outputs": [
            ({"level": ("low", {"low": 0.5, "middle": 0.0, "high": 0.5})}, 9),
        ],
    },
]

CLASSIFY_TEXT_CASES = [
    {
        "name": "single, multi with top_k and threshold, prompt and activation, with confidence",
        "request": {
            **BASE,
            "text": "The battery is weak and the delivery was slow.",
            "tasks": {
                "sentiment": ["positive", "negative", "neutral"],
                "aspects": {
                    "labels": {
                        "battery": "Battery life",
                        "screen": None,
                        "price": None,
                        "delivery": "Shipping speed",
                    },
                    "multi_label": True,
                    "cls_threshold": 0.3,
                    "top_k": 2,
                },
                "1": {
                    "labels": ["yes", "no"],
                    "prompt": "Is the customer unhappy?",
                    "class_act": "sigmoid",
                    "cls_threshold": 0.9,
                },
            },
            "include_confidence": True,
        },
        "outputs": [
            (
                {
                    "sentiment": (
                        "negative",
                        {"positive": 0.02, "negative": 0.95, "neutral": 0.03},
                    ),
                    "aspects": (
                        ["battery", "delivery"],
                        {"battery": 0.8, "screen": 0.35, "price": 0.1, "delivery": 0.97},
                    ),
                    "1": ("yes", {"yes": 0.88, "no": 0.2}),
                },
                70,
            ),
        ],
    },
    {
        "name": "batch of texts without confidence, multi-label without top_k",
        "request": {
            **BASE,
            "texts": ["First text", "Second text"],
            "tasks": {
                "topics": {"labels": ["weather", "cooking", "money"], "multi_label": True},
                "intent": ["ask", "tell"],
            },
            "threshold": 0.4,
        },
        "outputs": [
            (
                {
                    "topics": (
                        ["cooking", "money"],
                        {"weather": 0.1, "cooking": 0.7, "money": 0.6},
                    ),
                    "intent": ("tell", {"ask": 0.3, "tell": 0.7}),
                },
                20,
            ),
            (
                {
                    "topics": (["weather"], {"weather": 0.45, "cooking": 0.2, "money": 0.3}),
                    "intent": ("ask", {"ask": 0.6, "tell": 0.4}),
                },
                21,
            ),
        ],
    },
]

INVALID_CASES = [
    (
        "questions",
        {
            **BASE,
            "state": {"subject": "Refund"},
            "questions": {"q": {"type": "noul", "instructions": "Refund?"}},
        },
        ["state"],
        "STATE",
    ),
    (
        "questions",
        {**BASE, "state": "   ", "questions": {"q": {"type": "noul"}}},
        ["state"],
        "BLANK_STATE",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "box", "instructions": "the mug"}}},
        ["questions", "q"],
        "BOX",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "rank"}}},
        ["questions", "q"],
        "QUESTION_TYPE",
    ),
    (
        "questions",
        {
            **BASE,
            "state": "text",
            "assets": {"photo": {"type": "image"}},
            "questions": {"q": {"type": "noul"}},
        },
        ["assets"],
        "ASSETS",
    ),
    (
        "questions",
        {
            **BASE,
            "state": "text",
            "questions": {"q": {"type": "noul", "instructions": {"question": "Refund?"}}},
        },
        ["questions", "q"],
        "INSTRUCTIONS",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "noul", "threshold": 0.3}}},
        ["questions", "q"],
        "THRESHOLD",
    ),
    (
        "questions",
        {
            **BASE,
            "state": "text",
            "questions": {"q": {"type": "noul", "criteria": {"yes": "Refund"}}},
        },
        ["questions", "q"],
        "NOUL_CRITERIA",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
        ["questions", "q"],
        "CHOICE_CRITERIA",
    ),
    (
        "questions",
        {
            **BASE,
            "state": "text",
            "questions": {"q": {"type": "choice", "criteria": {"a": {"what": "A"}, "b": None}}},
        },
        ["questions", "q"],
        "CHOICE_CRITERIA",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "choice", "criteria": {"a": None}}}},
        ["questions", "q"],
        "CHOICE_COUNT",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "multi_select", "criteria": {}}}},
        ["questions", "q"],
        "MULTI_COUNT",
    ),
    (
        "questions",
        {**BASE, "state": "text", "questions": {"q": {"type": "score", "criteria": ["only one"]}}},
        ["questions", "q"],
        "SCORE_CRITERIA",
    ),
    (
        "questions",
        {
            **BASE,
            "state": "text",
            "questions": {"q": {"type": "score", "criteria": ["same", "same"]}},
        },
        ["questions", "q"],
        "SCORE_UNIQUE",
    ),
    ("classify_text", {**BASE, "tasks": {"t": ["a", "b"]}}, [], "TEXT_OR_TEXTS"),
    (
        "classify_text",
        {**BASE, "text": "x", "texts": ["y"], "tasks": {"t": ["a", "b"]}},
        [],
        "TEXT_OR_TEXTS",
    ),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": {"labels": ["a", "b"], "examples": [["text", "a"]]}}},
        ["tasks", "t"],
        "GLINER2_EXAMPLES",
    ),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": {"labels": ["a", "b"], "max_len": 10}}},
        ["tasks", "t"],
        "GLINER2_TASK",
    ),
    ("classify_text", {**BASE, "text": "x", "tasks": {"t": "a"}}, ["tasks", "t"], "GLINER2_TASK"),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": ["a", "a"]}},
        ["tasks", "t"],
        "GLINER2_LABELS",
    ),
    (
        "classify_text",
        {
            **BASE,
            "text": "x",
            "tasks": {"t": {"labels": ["a", "b"], "multi_label": True, "cls_threshold": 0}},
        },
        ["tasks", "t"],
        "GLINER2_THRESHOLD",
    ),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": {"labels": ["a", "b"], "top_k": 0}}},
        ["tasks", "t"],
        "GLINER2_TOP_K",
    ),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": {"labels": ["a", "b"], "class_act": "relu"}}},
        ["tasks", "t"],
        "GLINER2_ACTIVATION",
    ),
    (
        "classify_text",
        {
            **BASE,
            "text": "x",
            "tasks": {"t": {"labels": ["a", "b"], "multi_label": True, "class_act": "softmax"}},
        },
        ["tasks", "t"],
        "GLINER2_SOFTMAX_MULTI",
    ),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": {"labels": ["a", "b"], "multi_label": "yes"}}},
        ["tasks", "t"],
        "GLINER2_MULTI_LABEL",
    ),
    (
        "classify_text",
        {**BASE, "text": "x", "tasks": {"t": {"labels": ["a", "b"], "prompt": " "}}},
        ["tasks", "t"],
        "GLINER2_PROMPT",
    ),
]


def main() -> None:
    import gliner_runner.shapes as shapes

    cases = {
        "_comment": (
            "Written by scripts/api_shapes_fixture.py from the Python translation; "
            "the Python and TypeScript tests both check against it."
        ),
        "questions": [],
        "classify_text": [],
        "invalid": [],
    }
    for case in QUESTION_CASES:
        request = QuestionsRequest.model_validate(case["request"])
        planned = plan_questions(request)
        for item in planned:
            InferenceRequest.model_validate(item.request)
        responses = [response(native(out), tokens) for out, tokens in case["outputs"]]
        cases["questions"].append(
            {
                "name": case["name"],
                "request": case["request"],
                "requests": [item.request for item in planned],
                "outputs": [
                    {"output": r.output, "inputTokens": r.usage.input_tokens} for r in responses
                ],
                "response": questions_response(request, planned, responses),
            }
        )
    for case in CLASSIFY_TEXT_CASES:
        request = ClassifyTextRequest.model_validate(case["request"])
        plan = plan_classify_text(request)
        for item in plan.requests:
            InferenceRequest.model_validate(item)
        responses = [response(native(out), tokens) for out, tokens in case["outputs"]]
        cases["classify_text"].append(
            {
                "name": case["name"],
                "request": case["request"],
                "requests": list(plan.requests),
                "outputs": [
                    {"output": r.output, "inputTokens": r.usage.input_tokens} for r in responses
                ],
                "response": classify_text_response(request, plan, responses),
            }
        )
    for endpoint, body, loc, constant in INVALID_CASES:
        model = QuestionsRequest if endpoint == "questions" else ClassifyTextRequest
        try:
            model.model_validate(body)
        except ValidationError as error:
            errors = [e for e in error.errors() if list(e["loc"]) == loc]
            expected = "Value error, " + getattr(shapes, constant)
            assert any(e["msg"] == expected for e in errors), (constant, error.errors())
        else:
            raise AssertionError(f"{constant} was accepted")
        cases["invalid"].append(
            {
                "endpoint": endpoint,
                "body": body,
                "loc": ["body", *loc],
                "msg": "Value error, " + getattr(shapes, constant),
            }
        )
    Path(sys.argv[1]).write_text(
        json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


main()
