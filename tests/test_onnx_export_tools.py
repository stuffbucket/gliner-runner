from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.export_classification_onnx import (
    DECIDE_1B_REPOSITORY,
    DECIDE_1B_REVISION,
    DECIDE_REPOSITORY,
    DECIDE_REVISION,
    INPUT_NAMES,
    MAX_ATOL,
    MAX_SYNTHETIC_ZERO_THRESHOLD,
    MULTILINGUAL_REPOSITORY,
    MULTILINGUAL_REVISION,
    REPOSITORY_ROOT,
    SMALL_REPOSITORY,
    SMALL_REVISION,
    artifact_inventory,
    compare_logits,
    export_target_spec,
    fixture_suite_for_target,
    generate_report,
    parser,
    run,
    validate_atol,
    validate_comparison_synthetic_zero_threshold,
    validate_export_target,
    validate_external_data_files,
    validate_graph_contract,
    validate_masked_logits,
    validate_opset_imports,
    validate_output_location,
)


def dynamic_graph_contract() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inputs = [
        {"name": "input_ids", "type": "tensor(int64)", "shape": ["B", "S"]},
        {"name": "attention_mask", "type": "tensor(int64)", "shape": ["B", "S"]},
        {"name": "label_positions", "type": "tensor(int64)", "shape": ["B", "L"]},
        {"name": "label_mask", "type": "tensor(bool)", "shape": ["B", "L"]},
    ]
    outputs = [{"name": "logits", "type": "tensor(float)", "shape": ["B", "L"]}]
    return inputs, outputs


def report_fixture() -> dict[str, object]:
    parity = {
        "valid_logits_compared": 5,
        "selected_index_matches": 2,
        "rows": [{}, {}],
        "max_abs_logit_difference": 1e-6,
    }
    return {
        "model_repository": SMALL_REPOSITORY,
        "model_revision": SMALL_REVISION,
        "model_content_sha256": "f" * 64,
        "versions": {
            "gliner2": "2.1.0",
            "torch": "2.14.0",
            "onnx": "1.23.2",
            "onnxruntime": "1.31.0",
        },
        "provider": "CPUExecutionProvider",
        "opset": 20,
        "opset_imports": [{"domain": "ai.onnx", "version": 20}],
        "absolute_tolerance": 1e-4,
        "quantization": "none",
        "fixture_suite": "classification-logits-english-v1",
        "fixture_languages": ["en"],
        "graph": {
            "inputs": [{"name": name} for name in INPUT_NAMES],
            "outputs": [{"name": "logits"}],
        },
        "parity": parity,
        "official_path": "gliner2.classification.Classifier.batch_score",
        "official_parity": parity,
        "fixture_shapes": ["B=1/S=12/K=3", "B=2/S=24/K=4"],
        "artifacts": [{"path": "classifier.onnx", "size": 42, "sha256": "a" * 64}],
    }


@pytest.mark.parametrize(
    ("repository", "revision", "target_kind", "fixture_suite"),
    [
        (
            SMALL_REPOSITORY,
            SMALL_REVISION,
            "small",
            "classification-logits-english-v1",
        ),
        (
            DECIDE_REPOSITORY,
            DECIDE_REVISION,
            "decide",
            "classification-logits-english-v1",
        ),
        (
            MULTILINGUAL_REPOSITORY,
            MULTILINGUAL_REVISION,
            "decide-multilingual",
            "classification-logits-multilingual-v1",
        ),
        (
            DECIDE_1B_REPOSITORY,
            DECIDE_1B_REVISION,
            "decide-1b",
            "classification-logits-english-v1",
        ),
    ],
)
def test_export_targets_are_pinned_to_reviewed_revisions(
    repository: str,
    revision: str,
    target_kind: str,
    fixture_suite: str,
) -> None:
    assert validate_export_target(repository, revision) == target_kind
    spec = export_target_spec(repository, revision)
    assert spec.fixture_suite == fixture_suite

    with pytest.raises(ValueError, match="export requires"):
        validate_export_target(repository, "a" * 40)


def test_unknown_export_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported ONNX research target"):
        validate_export_target("other/model", "a" * 40)


def test_multilingual_export_fixture_is_explicitly_non_english() -> None:
    target = export_target_spec(MULTILINGUAL_REPOSITORY, MULTILINGUAL_REVISION)
    fixture = fixture_suite_for_target(target)

    assert fixture.suite_id == "classification-logits-multilingual-v1"
    assert fixture.languages == ("es", "fr", "de")
    assert any(not text.isascii() for rows in fixture.texts for text in rows)
    assert tuple(map(len, fixture.texts)) == (1, 2)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), MAX_ATOL * 2])
def test_absolute_tolerance_must_be_finite_positive_and_reviewed(value: float) -> None:
    with pytest.raises(ValueError, match="absolute tolerance"):
        validate_atol(value)


def test_absolute_tolerance_accepts_reviewed_boundary() -> None:
    assert validate_atol(MAX_ATOL) == MAX_ATOL


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), MAX_SYNTHETIC_ZERO_THRESHOLD * 2],
)
def test_synthetic_zero_threshold_must_be_finite_positive_and_reviewed(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="synthetic-zero threshold"):
        validate_comparison_synthetic_zero_threshold(value)


def test_synthetic_zero_threshold_accepts_none_and_reviewed_boundary() -> None:
    assert validate_comparison_synthetic_zero_threshold(None) is None
    assert (
        validate_comparison_synthetic_zero_threshold(MAX_SYNTHETIC_ZERO_THRESHOLD)
        == MAX_SYNTHETIC_ZERO_THRESHOLD
    )


def test_synthetic_zero_cli_is_explicitly_opt_in(tmp_path: Path) -> None:
    required = [
        "--model-path",
        str(tmp_path / "model"),
        "--model-repository",
        SMALL_REPOSITORY,
        "--model-revision",
        SMALL_REVISION,
        "--expected-gliner2-version",
        "2.1.0",
        "--output-dir",
        str(tmp_path / "output"),
    ]

    assert parser().parse_args(required).comparison_synthetic_zero_threshold is None
    configured = parser().parse_args([*required, "--comparison-synthetic-zero-threshold", "0.0001"])
    assert configured.comparison_synthetic_zero_threshold == 1e-4


def test_import_forces_offline_model_loading_in_isolated_subprocess(tmp_path: Path) -> None:
    code = f"""
import json, os, sys
sys.path.insert(0, {str(REPOSITORY_ROOT)!r})
import benchmarks.export_classification_onnx
keys = (
    'HF_HUB_OFFLINE',
    'TRANSFORMERS_OFFLINE',
    'HF_HUB_DISABLE_TELEMETRY',
    'DO_NOT_TRACK',
)
print(json.dumps({{key: os.environ[key] for key in keys}}, sort_keys=True))
"""
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_HUB_DISABLE_TELEMETRY": "0",
        "DO_NOT_TRACK": "0",
    }

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_compare_logits_ignores_padding_and_checks_selected_labels() -> None:
    result = compare_logits(
        [[0.2, 0.9, -3.4e38], [0.7, 0.6, 0.1]],
        [[0.20001, 0.90001, -3.3e38], [0.69999, 0.60001, 0.10001]],
        [[True, True, False], [True, True, True]],
    )

    assert set(result) == {
        "rows",
        "valid_logits_compared",
        "selected_index_matches",
        "max_abs_logit_difference",
    }
    assert result["valid_logits_compared"] == 5
    assert result["selected_index_matches"] == 2
    assert result["max_abs_logit_difference"] == pytest.approx(1e-5)
    assert all("synthetic_zero_delta" not in row for row in result["rows"])


def test_compare_logits_reports_optional_synthetic_zero_delta_without_hiding_raw_drift() -> None:
    result = compare_logits(
        [[0.0, 0.0, 0.0, -3.4e38]],
        [[5e-5, 1e-4, 2e-4, -3.3e38]],
        [[True, True, True, False]],
        synthetic_zero_threshold=1e-4,
    )

    assert result["max_abs_logit_difference"] == pytest.approx(2e-4)
    assert result["selected_index_matches"] == 0
    normalized = result["synthetic_zero_delta"]
    assert normalized == {
        "threshold": 1e-4,
        "boundary": "strict-less-than",
        "valid_deltas_compared": 3,
        "differences_below_threshold": 1,
        "nonzero_differences_zeroed": 1,
        "max_abs_normalized_difference": pytest.approx(2e-4),
    }
    row = result["rows"][0]
    assert row["selected_index_match"] is False
    assert row["synthetic_zero_delta"] == normalized


def test_synthetic_zero_delta_preserves_exact_zero_and_strict_boundary() -> None:
    result = compare_logits(
        [[0.0, 0.0, 0.0]],
        [[0.0, 5e-5, 1e-4]],
        [[True, True, True]],
        synthetic_zero_threshold=1e-4,
    )

    normalized = result["synthetic_zero_delta"]
    assert normalized["differences_below_threshold"] == 2
    assert normalized["nonzero_differences_zeroed"] == 1
    assert normalized["max_abs_normalized_difference"] == pytest.approx(1e-4)
    assert result["max_abs_logit_difference"] == pytest.approx(1e-4)
    assert result["selected_index_matches"] == 0


def test_synthetic_zero_delta_does_not_add_a_zero_selection_candidate() -> None:
    result = compare_logits(
        [[-0.2, -0.1]],
        [[-0.20005, -0.10005]],
        [[True, True]],
        synthetic_zero_threshold=1e-4,
    )

    assert result["selected_index_matches"] == 1
    assert result["rows"][0]["pytorch_selected_index"] == 1
    assert result["rows"][0]["onnx_selected_index"] == 1
    assert result["synthetic_zero_delta"]["max_abs_normalized_difference"] == 0.0


def test_compare_logits_rejects_bad_shapes_empty_routes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="shapes do not match"):
        compare_logits([[1.0]], [[1.0, 2.0]], [[True]])
    with pytest.raises(ValueError, match="at least one valid label"):
        compare_logits([[1.0]], [[1.0]], [[False]])
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="must all be finite"):
            compare_logits([[value]], [[1.0]], [[True]])
        with pytest.raises(ValueError, match="must all be finite"):
            compare_logits([[1.0]], [[value]], [[True]])


def test_masked_logits_must_be_finite_and_strongly_negative() -> None:
    validate_masked_logits([-3.4e38, -1e30])
    for value in (float("nan"), float("inf"), -float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and strongly negative"):
            validate_masked_logits([value])


def test_compare_logits_reports_selection_mismatch() -> None:
    result = compare_logits([[0.6, 0.5]], [[0.4, 0.7]], [[True, True]])

    assert result["selected_index_matches"] == 0
    assert result["rows"][0]["selected_index_match"] is False


def test_graph_contract_requires_shared_nonempty_dynamic_symbols() -> None:
    inputs, outputs = dynamic_graph_contract()

    result = validate_graph_contract(inputs, outputs)

    assert tuple(item["name"] for item in result["inputs"]) == INPUT_NAMES
    assert result["outputs"][0]["name"] == "logits"

    fixed_inputs = [dict(item) for item in inputs]
    fixed_inputs[0] = {**fixed_inputs[0], "shape": [2, "S"]}
    with pytest.raises(ValueError, match="nonempty dynamic"):
        validate_graph_contract(fixed_inputs, outputs)

    empty_symbol = [dict(item) for item in inputs]
    empty_symbol[0] = {**empty_symbol[0], "shape": ["", "S"]}
    with pytest.raises(ValueError, match="nonempty dynamic"):
        validate_graph_contract(empty_symbol, outputs)

    mismatched = [dict(item) for item in inputs]
    mismatched[1] = {**mismatched[1], "shape": ["other", "S"]}
    with pytest.raises(ValueError, match="share one dynamic batch"):
        validate_graph_contract(mismatched, outputs)

    wrong_type = [dict(item) for item in inputs]
    wrong_type[3] = {**wrong_type[3], "type": "tensor(int64)"}
    with pytest.raises(ValueError, match="label_mask has type"):
        validate_graph_contract(wrong_type, outputs)

    with pytest.raises(ValueError, match="one logits output"):
        validate_graph_contract(inputs, [])


def test_opset_contract_requires_standard_opset_20_and_records_all_imports() -> None:
    assert validate_opset_imports(
        [
            {"domain": "com.microsoft", "version": 1},
            {"domain": "", "version": 20},
        ]
    ) == [
        {"domain": "ai.onnx", "version": 20},
        {"domain": "com.microsoft", "version": 1},
    ]

    with pytest.raises(ValueError, match="standard opset 20"):
        validate_opset_imports([{"domain": "ai.onnx", "version": 19}])


def test_output_location_rejects_snapshot_and_unignored_repository_paths(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="inside the model snapshot"):
        validate_output_location(snapshot / "onnx", snapshot)
    with pytest.raises(ValueError, match="repository-local"):
        validate_output_location(REPOSITORY_ROOT / "onnx-output", snapshot)

    allowed = REPOSITORY_ROOT / "benchmark-results" / "local" / "onnx-test"
    assert validate_output_location(allowed, snapshot) == allowed.resolve()
    assert (
        validate_output_location(tmp_path / "external-output", snapshot)
        == (tmp_path / "external-output").resolve()
    )


def test_artifact_inventory_hashes_graph_and_external_sidecar_exactly(tmp_path: Path) -> None:
    graph = tmp_path / "classifier.onnx"
    sidecar = tmp_path / "classifier.onnx.data"
    graph.write_bytes(b"graph")
    sidecar.write_bytes(b"weights")
    nested = tmp_path / "export-report"
    nested.mkdir()
    (nested / "a.md").write_text("report", encoding="utf-8")
    (tmp_path / "reproducibility.json").write_text("{}", encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("report", encoding="utf-8")

    result = artifact_inventory(tmp_path)

    assert [item["path"] for item in result] == [
        "classifier.onnx",
        "classifier.onnx.data",
        "export-report/a.md",
    ]
    assert result[0]["sha256"] == hashlib.sha256(b"graph").hexdigest()
    assert result[1]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert validate_external_data_files(tmp_path, ["classifier.onnx.data"]) == [
        "classifier.onnx.data"
    ]
    with pytest.raises(ValueError, match="escapes"):
        validate_external_data_files(tmp_path, ["../outside"])


def test_generated_report_records_reviewed_contract_and_official_oracle() -> None:
    report = generate_report(report_fixture())

    assert "research evidence" in report
    assert "does not advertise or register a production backend" in report
    assert "Tokenization" in report
    assert "Selected-label parity: 2/2" in report
    assert "Absolute tolerance: 0.0001" in report
    assert "Classifier.batch_score" in report
    assert "classification-logits-english-v1" in report
    assert "Fixture languages: en" in report
    assert '"version": 20' in report
    assert "Synthetic-zero delta view: disabled" in report


def test_generated_report_records_optional_synthetic_zero_diagnostics() -> None:
    fixture = report_fixture()
    parity = dict(fixture["parity"])
    parity["synthetic_zero_delta"] = {
        "threshold": 1e-4,
        "boundary": "strict-less-than",
        "valid_deltas_compared": 5,
        "differences_below_threshold": 5,
        "nonzero_differences_zeroed": 4,
        "max_abs_normalized_difference": 0.0,
    }
    fixture["parity"] = parity

    report = generate_report(fixture)

    assert "Synthetic-zero delta view: enabled, diagnostic only" in report
    assert "Synthetic-zero threshold: 0.0001" in report
    assert "normalized maximum absolute difference: 0" in report
    assert "Positive differences converted to synthetic zero: 4/5" in report


def test_onnx_export_dependencies_and_notices_match_tested_versions() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    notices = (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    expected = [
        "gliner2[local]==2.1.0",
        "torch==2.14.0",
        "transformers==5.19.0",
        "onnx==1.23.2",
        "onnxruntime==1.31.0",
        "onnxscript==0.7.2",
    ]

    assert pyproject["project"]["optional-dependencies"]["onnx-export"] == expected
    for name, version in (
        ("gliner2", "2.1.0"),
        ("torch", "2.14.0"),
        ("transformers", "5.19.0"),
        ("onnx", "1.23.2"),
        ("onnxruntime", "1.31.0"),
        ("onnxscript", "0.7.2"),
    ):
        assert f"| `{name}` (optional ONNX export extra) | `=={version}` |" in notices


def assert_real_export_evidence(
    evidence: dict[str, object],
    *,
    repository: str,
    revision: str,
    target_kind: str,
    fixture_suite: str,
    require_external_data: bool,
) -> None:
    assert evidence["format_version"] == 3
    assert evidence["kind"] == "classification-logits-research"
    assert evidence["model_repository"] == repository
    assert evidence["model_revision"] == revision
    assert evidence["target_kind"] == target_kind
    assert evidence["fixture_suite"] == fixture_suite
    assert evidence["official_path"] == "gliner2.classification.Classifier.batch_score"
    official = evidence["official_parity"]
    parity = evidence["parity"]
    assert isinstance(official, dict)
    assert isinstance(parity, dict)
    assert official["selected_index_matches"] == len(official["rows"])
    assert parity["selected_index_matches"] == len(parity["rows"])
    assert official["max_abs_logit_difference"] <= evidence["absolute_tolerance"]
    assert parity["max_abs_logit_difference"] <= evidence["absolute_tolerance"]
    assert "synthetic_zero_delta" not in official
    normalized = parity["synthetic_zero_delta"]
    assert isinstance(normalized, dict)
    assert normalized["threshold"] == 1e-4
    assert normalized["boundary"] == "strict-less-than"
    assert normalized["valid_deltas_compared"] == parity["valid_logits_compared"]
    assert normalized["max_abs_normalized_difference"] == 0.0
    assert evidence["comparison_delta_policy"] == {
        "mode": "synthetic-zero-below-threshold",
        "threshold": 1e-4,
        "boundary": "strict-less-than",
        "scope": "pytorch-wrapper-vs-onnx-valid-label-logits-only",
        "affects_acceptance": False,
    }
    assert evidence["opset_imports"] == [{"domain": "ai.onnx", "version": 20}]
    assert evidence["absolute_tolerance"] == 1e-4
    graph = evidence["graph"]
    assert isinstance(graph, dict)
    assert tuple(item["name"] for item in graph["inputs"]) == INPUT_NAMES
    assert [item["name"] for item in graph["outputs"]] == ["logits"]
    external_data = evidence["external_data_files"]
    artifacts = evidence["artifacts"]
    assert isinstance(external_data, list)
    assert isinstance(artifacts, list)
    artifact_paths = {item["path"] for item in artifacts}
    assert set(external_data) <= artifact_paths
    if require_external_data:
        assert external_data


@pytest.mark.integration
@pytest.mark.benchmark
@pytest.mark.parametrize(
    (
        "environment_variable",
        "repository",
        "revision",
        "target_kind",
        "fixture_suite",
        "require_external_data",
    ),
    [
        (
            "GLINER_RUNNER_ONNX_SMALL_MODEL",
            SMALL_REPOSITORY,
            SMALL_REVISION,
            "small",
            "classification-logits-english-v1",
            False,
        ),
        (
            "GLINER_RUNNER_ONNX_DECIDE_MODEL",
            DECIDE_REPOSITORY,
            DECIDE_REVISION,
            "decide",
            "classification-logits-english-v1",
            False,
        ),
        (
            "GLINER_RUNNER_ONNX_MULTILINGUAL_MODEL",
            MULTILINGUAL_REPOSITORY,
            MULTILINGUAL_REVISION,
            "decide-multilingual",
            "classification-logits-multilingual-v1",
            False,
        ),
        (
            "GLINER_RUNNER_ONNX_1B_MODEL",
            DECIDE_1B_REPOSITORY,
            DECIDE_1B_REVISION,
            "decide-1b",
            "classification-logits-english-v1",
            True,
        ),
    ],
)
def test_real_export_when_explicitly_configured(
    tmp_path: Path,
    environment_variable: str,
    repository: str,
    revision: str,
    target_kind: str,
    fixture_suite: str,
    require_external_data: bool,
) -> None:
    model_path_value = os.environ.get(environment_variable)
    if not model_path_value:
        pytest.skip(f"set {environment_variable} to the reviewed local snapshot")

    output = tmp_path / target_kind
    run(
        SimpleNamespace(
            model_path=Path(model_path_value),
            model_repository=repository,
            model_revision=revision,
            expected_gliner2_version="2.1.0",
            output_dir=output,
            atol=1e-4,
            comparison_synthetic_zero_threshold=1e-4,
        )
    )

    evidence = json.loads((output / "reproducibility.json").read_text(encoding="utf-8"))
    assert_real_export_evidence(
        evidence,
        repository=repository,
        revision=revision,
        target_kind=target_kind,
        fixture_suite=fixture_suite,
        require_external_data=require_external_data,
    )
