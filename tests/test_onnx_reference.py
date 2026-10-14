from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from scripts.generate_onnx_parity_corpus import CORPORA, OUTPUT_DIRECTORY, render

from gliner_runner.contracts import (
    BackendName,
    BackendResult,
    InferenceRequest,
    InferenceUsage,
    Precision,
)
from gliner_runner.errors import InvalidSchemaError, UnsupportedCapabilityError
from gliner_runner.model_inventory import KNOWN_MODELS
from gliner_runner.onnx_reference import (
    CORPUS_FORMAT_VERSION,
    collect_reference,
    compare_json,
    load_corpus,
    run_reference,
    structured_error,
)

CORE_CASES = [
    "described-single",
    "multi-label",
    "high-threshold-empty",
    "ordered-label",
    "feasible-constraint",
    "infeasible-constraint",
    "mixed-length-long",
]


class FakeBackend:
    """Deterministic stand-in for the PyTorch backend."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.loaded = False
        self.closed = False
        self.calls = 0

    async def load(self) -> None:
        self.loaded = True

    async def close(self) -> None:
        self.closed = True

    async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
        self.batches.append([request.text for request in requests])
        if any(request.text == "boom" for request in requests):
            raise InvalidSchemaError("bad schema")
        if any(request.text == "explode" for request in requests):
            raise RuntimeError("kaboom")
        self.calls += 1
        return [
            BackendResult(
                output={"label": request.text, "score": 0.5 + len(requests) * 1e-9},
                usage=InferenceUsage(input_tokens=len(request.text), output_tokens=0),
            )
            for request in requests
        ]


def write_corpus(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.write_text(
        json.dumps({"format_version": CORPUS_FORMAT_VERSION, "corpus_id": "tiny", "cases": cases}),
        encoding="utf-8",
    )
    return path


def case(case_id: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "categories": ["single_label"],
        **extra,
        "request": {
            "model": "decide-340m",
            "backend": "onnx",
            "precision": "fp16",
            "operation": "classify",
            "text": text,
            "schema": {
                "kind": "classification",
                "tasks": {"intent": {"labels": ["a", "b"]}},
            },
        },
    }


# --- committed corpora ------------------------------------------------------------------


@pytest.mark.parametrize("workload_id", CORPORA)
def test_committed_corpora_are_current_and_valid(workload_id: str) -> None:
    filename, model = CORPORA[workload_id]
    path = OUTPUT_DIRECTORY / filename

    assert path.read_text(encoding="utf-8") == render(workload_id)
    corpus = load_corpus(path)

    assert corpus.corpus_id == workload_id
    assert [item.case_id for item in corpus.cases[:7]] == CORE_CASES
    assert len({item.case_id for item in corpus.cases}) == len(corpus.cases)
    assert all(item.request.model == model for item in corpus.cases)
    assert all(item.request.precision is Precision.FP32 for item in corpus.cases)
    categories = {category for item in corpus.cases for category in item.categories}
    assert {
        "multi_task",
        "described_label",
        "instruction",
        "multi_label",
        "ordered",
        "constraint",
        "infeasible",
        "threshold",
        "empty_result",
        "invalid_schema",
        "token_accounting",
    } <= categories
    groups = {item.batch_group for item in corpus.cases if item.batch_group}
    assert groups == {"mixed-length"}


def test_multilingual_corpus_covers_every_language() -> None:
    corpus = load_corpus(OUTPUT_DIRECTORY / "corpus-multilingual-v1.json")

    assert {item.language for item in corpus.cases[:7]} == {"es", "fr", "ja", "de"}


# --- corpus loading ---------------------------------------------------------------------


def test_load_corpus_rejects_malformed_documents(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    for document, message in (
        ("not json", "not valid JSON"),
        ("[]", "format_version"),
        (json.dumps({"format_version": 2, "corpus_id": "x", "cases": []}), "format_version"),
        (json.dumps({"format_version": 1, "corpus_id": "", "cases": [1]}), "corpus_id"),
        (json.dumps({"format_version": 1, "corpus_id": "x", "cases": []}), "nonempty cases"),
        (json.dumps({"format_version": 1, "corpus_id": "x", "cases": [1]}), "objects"),
    ):
        path.write_text(document, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_corpus(path)

    with pytest.raises(ValueError, match="unique"):
        load_corpus(write_corpus(path, [case("a", "x"), case("a", "y")]))
    broken = case("a", "x")
    broken["request"]["text"] = " "
    with pytest.raises(ValueError, match="must not be blank"):
        load_corpus(write_corpus(path, [broken]))
    entity = case("a", "x")
    entity["request"]["operation"] = "extract_entities"
    entity["request"]["schema"] = {"kind": "entities", "labels": ["x"]}
    with pytest.raises(ValueError, match="classify"):
        load_corpus(write_corpus(path, [entity]))
    with pytest.raises(ValueError, match="invalid metadata"):
        load_corpus(write_corpus(path, [case("a", "x", batch_group=3)]))


def test_corpus_hash_covers_the_exact_bytes(tmp_path: Path) -> None:
    first = load_corpus(write_corpus(tmp_path / "a.json", [case("a", "x")]))
    second = load_corpus(write_corpus(tmp_path / "b.json", [case("a", "y")]))

    assert len(first.sha256) == 64
    assert first.sha256 != second.sha256


# --- errors and comparison --------------------------------------------------------------


def test_structured_errors_follow_the_server_mapping() -> None:
    assert structured_error(InvalidSchemaError("nope")) == (
        422,
        {"code": "invalid_schema", "message": "nope", "retryable": False},
    )
    status, body = structured_error(UnsupportedCapabilityError("unsupported"))
    assert (status, body["code"]) == (409, "unsupported_configuration")
    status, body = structured_error(KeyError("x"))
    assert (status, body["code"], body["exception_type"]) == (500, "internal_error", "KeyError")


def test_compare_json_reports_structure_and_float_differences() -> None:
    assert compare_json({"a": [1.0, {"b": 2}]}, {"a": [1.0, {"b": 2}]}) == (True, 0.0)
    same, difference = compare_json({"a": [1.0, 0.5]}, {"a": [1.0, 0.25]})
    assert same
    assert difference == 0.25
    assert compare_json({"a": 1}, {"b": 1})[0] is False
    assert compare_json([1], [1, 2])[0] is False
    assert compare_json(True, 1)[0] is False
    assert compare_json("a", "b")[0] is False
    assert compare_json(None, None) == (True, 0.0)


# --- collection -------------------------------------------------------------------------


async def test_collect_reference_runs_singletons_repeats_and_batch_groups(tmp_path: Path) -> None:
    corpus = load_corpus(
        write_corpus(
            tmp_path / "corpus.json",
            [
                case("one", "alpha", batch_group="g"),
                case("two", "beta", batch_group="g", language="en"),
                case("solo", "gamma"),
                case("bad", "boom"),
                case("crash", "explode"),
            ],
        )
    )
    backend = FakeBackend()

    result = await collect_reference(backend, model_id="decide-1b", corpus=corpus, repeats=3)

    singleton = {item["case_id"]: item for item in result["singleton"]}
    assert list(singleton) == ["one", "two", "solo", "bad", "crash"]
    assert singleton["one"]["status"] == "success"
    assert singleton["one"]["output"] == {"label": "alpha", "score": 0.5 + 1e-9}
    assert singleton["one"]["usage"] == {"inputTokens": 5, "outputTokens": 0}
    assert singleton["one"]["repeats"] == 3
    assert singleton["one"]["repeat_identical"] is True
    assert singleton["two"]["language"] == "en"
    assert singleton["bad"]["http_status"] == 422
    assert singleton["bad"]["error"]["code"] == "invalid_schema"
    assert singleton["crash"]["http_status"] == 500
    assert backend.batches.count(["alpha"]) == 3
    assert backend.batches[-1] == ["alpha", "beta"]

    [batch] = result["batches"]
    assert batch["batch_group"] == "g"
    assert batch["case_ids"] == ["one", "two"]
    first = batch["results"][0]
    assert first["status"] == "success"
    assert first["matches_singleton_structure"] is True
    assert first["matches_singleton_exactly"] is False
    assert first["max_abs_float_difference_vs_singleton"] == pytest.approx(1e-9, abs=1e-12)


async def test_collect_reference_substitutes_the_model_under_test(tmp_path: Path) -> None:
    corpus = load_corpus(write_corpus(tmp_path / "corpus.json", [case("one", "alpha")]))
    seen: list[InferenceRequest] = []

    class Recording(FakeBackend):
        async def infer_batch(self, requests: Sequence[InferenceRequest]) -> list[BackendResult]:
            seen.extend(requests)
            return await super().infer_batch(requests)

    await collect_reference(Recording(), model_id="decide-1b", corpus=corpus, repeats=1)

    assert [(item.model, item.backend, item.precision) for item in seen] == [
        ("decide-1b", BackendName.PYTORCH, Precision.FP32)
    ]


async def test_collect_reference_rejects_nonpositive_repeats(tmp_path: Path) -> None:
    corpus = load_corpus(write_corpus(tmp_path / "corpus.json", [case("one", "alpha")]))
    with pytest.raises(ValueError, match="repeats"):
        await collect_reference(FakeBackend(), model_id="decide-1b", corpus=corpus, repeats=0)


# --- end-to-end writer ------------------------------------------------------------------


def hf_snapshot(root: Path, repository: str, revision: str) -> Path:
    path = root / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def test_run_reference_writes_a_new_golden_file(tmp_path: Path) -> None:
    spec = KNOWN_MODELS[2]
    snapshot = hf_snapshot(tmp_path, spec.repository, spec.pinned_revision)
    corpus_path = write_corpus(tmp_path / "corpus.json", [case("one", "alpha")])
    output = tmp_path / "out" / "golden.json"
    backend = FakeBackend()

    written = run_reference(
        "decide-1b",
        snapshot=snapshot,
        corpus_path=corpus_path,
        output=output,
        repeats=1,
        backend_factory=lambda path: backend if path == snapshot.resolve() else FakeBackend(),
    )

    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["kind"] == "gliner-runner-onnx-reference-outputs"
    assert document["model_id"] == "decide-1b"
    assert document["repository"] == spec.repository
    assert document["revision"] == spec.pinned_revision
    assert document["corpus_id"] == "tiny"
    assert document["corpus_sha256"] == load_corpus(corpus_path).sha256
    assert document["reference"]["backend"] == "pytorch"
    assert document["reference"]["precision"] == "fp32"
    assert document["singleton"][0]["output"]["label"] == "alpha"
    assert backend.loaded and backend.closed

    with pytest.raises(FileExistsError):
        run_reference(
            "decide-1b",
            snapshot=snapshot,
            corpus_path=corpus_path,
            output=output,
            backend_factory=lambda _path: backend,
        )


def test_run_reference_rejects_uncurated_models_and_wrong_snapshots(tmp_path: Path) -> None:
    spec = KNOWN_MODELS[0]
    snapshot = hf_snapshot(tmp_path, spec.repository, spec.pinned_revision)
    corpus_path = write_corpus(tmp_path / "corpus.json", [case("one", "alpha")])

    with pytest.raises(ValueError, match="not curated"):
        run_reference(
            "fastino/other", snapshot=snapshot, corpus_path=corpus_path, output=tmp_path / "o"
        )
    with pytest.raises(ValueError, match="identity does not match"):
        run_reference(
            "decide-1b", snapshot=snapshot, corpus_path=corpus_path, output=tmp_path / "o"
        )
    assert not (tmp_path / "o").exists()
