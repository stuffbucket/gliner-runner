"""Generate the checked-in ONNX parity request corpora.

The corpora are built from the preserved compatibility cases in
``benchmarks/benchmark_real.py`` (workloads ``english-v1`` and ``multilingual-v1``) plus
additional multi-task, default/temperature, repeated, and structured-error cases. Requests
use the runner ``InferenceRequest`` JSON shape; consumers substitute ``model`` (and
``backend``) for the implementation under test.

    python scripts/generate_onnx_parity_corpus.py          # rewrite the files
    python scripts/generate_onnx_parity_corpus.py --check  # verify they are current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_real  # noqa: E402

OUTPUT_DIRECTORY = ROOT / "tests" / "fixtures" / "onnx-parity"
CORPORA = {
    "english-v1": ("corpus-english-v1.json", "decide-340m"),
    "multilingual-v1": ("corpus-multilingual-v1.json", "decide-multilingual"),
}
EXTRA_TEXTS = {
    "english-v1": {
        "multi-task": "The app crashes and my card was charged twice.",
        "default": "Hello there.",
        "softmax": "Please refund the duplicate invoice charge.",
        "invalid": "Answer yes or no.",
    },
    "multilingual-v1": {
        "multi-task": "L'application plante et ma carte a été débitée deux fois.",
        "default": "こんにちは。",
        "softmax": "Por favor, reembolse el cargo duplicado de la factura.",
        "invalid": "Répondez oui ou non.",
    },
}


def _classification(tasks: dict[str, Any], constraints: list[Any] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"kind": "classification", "tasks": tasks}
    if constraints:
        schema["constraints"] = constraints
    return schema


def _case(
    case_id: str,
    categories: list[str],
    text: str,
    schema: dict[str, Any],
    threshold: float,
    model: str,
    workload: Any,
    batch_group: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    case: dict[str, Any] = {"case_id": case_id, "categories": categories}
    if language is not None:
        case["language"] = language
    if batch_group is not None:
        case["batch_group"] = batch_group
    case["request"] = benchmark_real.request(
        text,
        precision="fp32",
        model=model,
        workload=workload,
        schema_payload=schema,
        threshold=threshold,
    )
    return case


def _extra_cases(workload: Any, model: str) -> list[dict[str, Any]]:
    texts = EXTRA_TEXTS[workload.workload_id]
    descriptions = dict(workload.label_descriptions)
    labels = list(workload.labels)
    specs: list[tuple[str, list[str], str, dict[str, Any]]] = [
        (
            "multi-task-described-instruction",
            ["multi_task", "described_label", "instruction", "multi_label", "token_accounting"],
            texts["multi-task"],
            _classification(
                {
                    "department": {
                        "labels": labels,
                        "label_descriptions": descriptions,
                        "min_labels": 1,
                        "max_labels": 1,
                        "instruction": "Choose the department that should handle the message",
                    },
                    "topics": {
                        "labels": labels,
                        "min_labels": 0,
                        "max_labels": 3,
                        "activation": "sigmoid",
                        "threshold": 0.3,
                    },
                }
            ),
        ),
        (
            "default-temperature-multi-label",
            ["default", "temperature", "multi_label", "token_accounting"],
            texts["default"],
            _classification(
                {
                    "topics": {
                        "labels": ["billing", "shipping", "other"],
                        "min_labels": 0,
                        "max_labels": 2,
                        "activation": "sigmoid",
                        "temperature": 2.0,
                        "default": "other",
                    }
                }
            ),
        ),
        (
            "softmax-activation",
            ["activation", "single_label", "token_accounting"],
            texts["softmax"],
            _classification(
                {
                    "intent": {
                        "labels": ["refund", "question", "complaint"],
                        "min_labels": 1,
                        "max_labels": 1,
                        "activation": "softmax",
                    }
                }
            ),
        ),
        (
            "invalid-reserved-marker",
            ["invalid_schema", "structured_error"],
            texts["invalid"],
            _classification({"intent": {"labels": ["yes", "[P]no"], "min_labels": 1}}),
        ),
        (
            "invalid-min-exceeds-max",
            ["invalid_schema", "structured_error"],
            texts["invalid"],
            _classification(
                {"intent": {"labels": ["yes", "no"], "min_labels": 2, "max_labels": 1}}
            ),
        ),
        (
            "invalid-unknown-constraint-label",
            ["invalid_schema", "structured_error", "constraint"],
            texts["invalid"],
            _classification(
                {"decision": {"labels": ["yes", "no"], "min_labels": 1, "max_labels": 1}},
                [{"type": "LabelRef", "task": "decision", "label": "maybe"}],
            ),
        ),
    ]
    return [
        _case(case_id, categories, text, schema, 0.5, model, workload)
        for case_id, categories, text, schema in specs
    ]


def build_corpus(workload_id: str) -> dict[str, Any]:
    workload = benchmark_real.workload_spec(workload_id)
    _file, model = CORPORA[workload_id]
    cases: list[dict[str, Any]] = []
    for source in benchmark_real.behavior_cases(workload):
        cases.append(
            _case(
                source["case_id"],
                list(source["categories"]),
                source["text"],
                source["schema"],
                source["threshold"],
                model,
                workload,
                source.get("batch_group"),
                source.get("language"),
            )
        )
    described = next(case for case in cases if case["case_id"] == "described-single")
    repeated = json.loads(json.dumps(described))
    repeated["case_id"] = "described-single-repeat"
    repeated["categories"] = ["single_label", "described_label", "repeated_run", "batching"]
    repeated["batch_group"] = "mixed-length"
    cases.append(repeated)
    cases.extend(_extra_cases(workload, model))
    return {
        "format_version": 1,
        "corpus_id": workload_id,
        "description": (
            f"Runner classification requests for the {workload_id} compatibility workload. "
            "The first seven cases are the benchmark_real behavior cases; the rest add "
            "multi-task, default/temperature, repeated, and structured-error coverage. "
            "Consumers replace request.model (and backend) with the implementation under test."
        ),
        "languages": list(workload.languages),
        "cases": cases,
    }


def render(workload_id: str) -> str:
    return json.dumps(build_corpus(workload_id), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if files are not current")
    arguments = parser.parse_args()
    stale: list[str] = []
    for workload_id, (filename, _model) in CORPORA.items():
        path = OUTPUT_DIRECTORY / filename
        content = render(workload_id)
        if arguments.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if stale:
        print("stale ONNX parity corpus: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
