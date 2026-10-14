from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.benchmark_real import (
    WORKLOADS,
    behavior_cases,
    comparison_summary,
    metric_value,
    parser,
)
from benchmarks.benchmark_support import (
    prepare_output_directory,
    snapshot_tree_digest,
    stable_sha256,
    validate_gliner2_version,
    validate_model_snapshot,
    write_new_text,
)
from benchmarks.compare_behavior import _mode_consistent, compare_behavior_artifacts

REVISION = "a" * 40


def write_inventory_marker(
    snapshot: Path,
    *,
    repository: str = "owner/model",
    revision: str = REVISION,
) -> None:
    content = b"model"
    (snapshot / "model.safetensors").write_bytes(content)
    (snapshot / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "repository": repository,
                "revision": revision,
                "files": [
                    {
                        "path": "model.safetensors",
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def real_artifact(
    *,
    repository: str,
    revision: str,
    version: str,
    confidence: float,
    workload: str = "workload",
    digest: str = "f" * 64,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "metadata": {
            "model_repository": repository,
            "model_revision": revision,
            "model_content_sha256": digest,
            "versions": {"gliner2": version},
            "device": "cpu",
            "precision": "fp32",
            "behavior_workload_sha256": workload,
            "behavior_case_ids": ["one", "two"],
        },
        "http_runner": {
            "behavior_cases": [
                {
                    "case_id": "one",
                    "categories": ["single_label", "token_accounting"],
                    "singleton": {
                        "status": "success",
                        "http_status": 200,
                        "usage": {"inputTokens": 10, "outputTokens": 0},
                        "output": {
                            "intent": {
                                "value": "billing",
                                "confidence": confidence,
                                "probabilities": {"billing": confidence},
                            },
                            "_meta": {
                                "decoder": "independent",
                                "exact": True,
                                "feasible": True,
                                "objective": confidence,
                                "violations": [],
                            },
                        },
                    },
                },
                {
                    "case_id": "two",
                    "categories": ["single_label", "token_accounting"],
                    "singleton": {
                        "status": "success",
                        "http_status": 200,
                        "usage": {"inputTokens": 11, "outputTokens": 0},
                        "output": {"intent": "shipping"},
                    },
                },
            ]
        },
    }
    cases = artifact["http_runner"]["behavior_cases"]  # type: ignore[index]
    for case in cases:
        case["repeated"] = json.loads(json.dumps(case["singleton"]))
        case["batch"] = json.loads(json.dumps(case["singleton"]))
    return artifact


def test_snapshot_identity_requires_repository_and_exact_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--owner--model" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)

    assert validate_model_snapshot(snapshot, "owner/model", REVISION) == snapshot.resolve()

    with pytest.raises(ValueError, match="repository and revision"):
        validate_model_snapshot(snapshot, "other/model", REVISION)
    with pytest.raises(ValueError, match="exact 40-character"):
        validate_model_snapshot(snapshot, "owner/model", "main")


def test_snapshot_identity_accepts_matching_inventory_marker(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    write_inventory_marker(snapshot)

    assert validate_model_snapshot(snapshot, "owner/model", REVISION) == snapshot.resolve()


def test_snapshot_identity_rejects_mismatching_inventory_marker(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    write_inventory_marker(snapshot, repository="other/model")

    with pytest.raises(ValueError, match="marker identity"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_snapshot_identity_rejects_malformed_inventory_marker(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / ".gliner-runner.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="marker is invalid"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_snapshot_identity_rejects_tampered_inventory_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    write_inventory_marker(snapshot)
    (snapshot / "model.safetensors").write_bytes(b"other")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_snapshot_identity_rejects_unlisted_active_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    write_inventory_marker(snapshot)
    (snapshot / "unlisted.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unlisted active file: unlisted\.json"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_marker_snapshot_ignores_downloader_admin_and_digests_only_inventory(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    write_inventory_marker(snapshot)
    admin = snapshot / ".cache" / "huggingface"
    admin.mkdir(parents=True)
    (admin / "download-state.json").write_text("changing metadata", encoding="utf-8")

    assert validate_model_snapshot(snapshot, "owner/model", REVISION) == snapshot.resolve()
    digest, size, count = snapshot_tree_digest(snapshot)
    expected = hashlib.sha256()
    expected.update(b"model.safetensors\0")
    expected.update(hashlib.sha256(b"model").digest())

    assert digest == expected.hexdigest()
    assert size == len(b"model")
    assert count == 1

    (admin / "download-state.json").write_text("changed again", encoding="utf-8")
    assert snapshot_tree_digest(snapshot) == (digest, size, count)


def test_snapshot_identity_rejects_admin_file_as_inventory_entry(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    admin = snapshot / ".cache" / "huggingface"
    admin.mkdir(parents=True)
    content = b"metadata"
    (admin / "state").write_bytes(content)
    (snapshot / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "repository": "owner/model",
                "revision": REVISION,
                "files": [
                    {
                        "path": ".cache/huggingface/state",
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid file entry"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_snapshot_identity_rejects_escaping_inventory_path(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)
    outside = snapshot.parent / "outside"
    outside.write_bytes(b"model")
    (snapshot / ".gliner-runner.json").write_text(
        json.dumps(
            {
                "repository": "owner/model",
                "revision": REVISION,
                "files": [
                    {
                        "path": "../outside",
                        "size": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the snapshot"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_snapshot_identity_rejects_unidentified_directory(tmp_path: Path) -> None:
    snapshot = tmp_path / "owner--model" / REVISION
    snapshot.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"matching \.gliner-runner\.json marker"):
        validate_model_snapshot(snapshot, "owner/model", REVISION)


def test_behavior_corpus_covers_compatibility_dimensions() -> None:
    cases = behavior_cases()
    categories = {category for case in cases for category in case["categories"]}

    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {
        "single_label",
        "described_label",
        "multi_label",
        "selected_order",
        "threshold",
        "empty_result",
        "ordered",
        "constraint",
        "feasible",
        "infeasible",
        "singleton_batch",
        "repeated_run",
        "mixed_length",
        "token_accounting",
    } <= categories


def test_workload_suites_are_pinned_and_language_specific() -> None:
    assert tuple(WORKLOADS) == ("english-v1", "multilingual-v1")
    assert WORKLOADS["english-v1"].languages == ("en",)

    multilingual = WORKLOADS["multilingual-v1"]
    assert multilingual.languages == ("de", "es", "fr", "ja")
    assert {language for _text, _expected, language in multilingual.parity_cases} == set(
        multilingual.languages
    )
    assert {case["language"] for case in behavior_cases(multilingual)} == set(
        multilingual.languages
    )


def test_english_behavior_workload_identity_remains_compatible() -> None:
    assert stable_sha256(behavior_cases(WORKLOADS["english-v1"])) == (
        "096ab29800d9d141345b65d5a888262e2a5b25172671529ed3401ef2a66b9c51"
    )


def test_workload_suite_parser_wires_parent_and_oracle_commands() -> None:
    root_arguments = parser().parse_args(["--workload-suite", "multilingual-v1"])
    oracle_arguments = parser().parse_args(
        [
            "_oracle",
            "--model-path",
            "/tmp/model",
            "--events",
            "/tmp/events.jsonl",
            "--workload-suite",
            "multilingual-v1",
        ]
    )

    assert root_arguments.workload_suite == "multilingual-v1"
    assert oracle_arguments.workload_suite == "multilingual-v1"


def test_performance_comparison_rejects_different_workload_suites() -> None:
    shared_metadata = {
        "model_content_sha256": "f" * 64,
        "host": {"cpu": "test cpu"},
        "schema_label_descriptions": {"billing": "Billing"},
        "behavior_workload_sha256": "same-legacy-digest",
    }
    baseline = {"metadata": {**shared_metadata, "workload_id": "english-v1"}}
    candidate = {"metadata": {**shared_metadata, "workload_id": "multilingual-v1"}}

    with pytest.raises(ValueError, match="workload suite"):
        comparison_summary(baseline, candidate)


def test_performance_comparison_rejects_changed_complete_workload() -> None:
    shared_metadata = {
        "model_content_sha256": "f" * 64,
        "host": {"cpu": "test cpu"},
        "schema_label_descriptions": {"billing": "Billing"},
        "behavior_workload_sha256": "same-behavior-digest",
        "workload_id": "english-v1",
    }
    baseline = {"metadata": {**shared_metadata, "workload_sha256": "a" * 64}}
    candidate = {"metadata": {**shared_metadata, "workload_sha256": "b" * 64}}

    with pytest.raises(ValueError, match="complete workload"):
        comparison_summary(baseline, candidate)


def test_performance_comparison_rejects_incomplete_workload_identity() -> None:
    shared_metadata = {
        "model_content_sha256": "f" * 64,
        "host": {"cpu": "test cpu"},
        "schema_label_descriptions": {"billing": "Billing"},
        "behavior_workload_sha256": "same-behavior-digest",
    }
    baseline = {"metadata": shared_metadata}
    candidate = {
        "metadata": {
            **shared_metadata,
            "workload_id": "english-v1",
            "workload_sha256": "a" * 64,
        }
    }

    with pytest.raises(ValueError, match="suite identity is incomplete"):
        comparison_summary(baseline, candidate)


def test_performance_comparison_requires_same_host_identity() -> None:
    shared_metadata = {
        "model_content_sha256": "f" * 64,
        "schema_label_descriptions": {"billing": "Billing"},
        "behavior_workload_sha256": "same-behavior-digest",
        "workload_id": "english-v1",
        "workload_sha256": "a" * 64,
    }
    baseline = {"metadata": {**shared_metadata, "host": {"cpu": "test cpu", "memory": 1}}}
    candidate = {"metadata": {**shared_metadata, "host": {"cpu": "test cpu", "memory": 2}}}

    with pytest.raises(ValueError, match="host identity"):
        comparison_summary(baseline, candidate)


def test_metric_value_requires_finite_present_measurement() -> None:
    assert metric_value('gliner_runner_internal{metric="measured"} 2.0\n', "measured") == 2.0
    with pytest.raises(ValueError, match="missing"):
        metric_value("", "missing")
    with pytest.raises(ValueError, match="not finite"):
        metric_value('gliner_runner_internal{metric="broken"} NaN\n', "broken")


def test_gliner2_version_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda package: "2.1.0")

    assert validate_gliner2_version("2.1.0") == "2.1.0"
    with pytest.raises(RuntimeError, match=r"expected 2\.0\.0, found 2\.1\.0"):
        validate_gliner2_version("2.0.0")


def test_output_directory_must_be_new(tmp_path: Path) -> None:
    output = tmp_path / "new" / "results"
    assert prepare_output_directory(output) == output.resolve()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_output_directory(output)


def test_evidence_file_must_be_new(tmp_path: Path) -> None:
    output = tmp_path / "new" / "comparison.json"
    assert write_new_text(output, "first") == output.resolve()

    with pytest.raises(FileExistsError):
        write_new_text(output, "replacement")
    assert output.read_text(encoding="utf-8") == "first"


def test_mode_consistency_includes_matching_errors() -> None:
    error = {
        "status": "error",
        "http_status": 422,
        "error": {
            "detail": {
                "code": "invalid_schema",
                "message": "constraint set is infeasible",
                "retryable": False,
            }
        },
    }

    assert _mode_consistent({"singleton": error, "repeated": error})
    assert not _mode_consistent(
        {
            "singleton": error,
            "repeated": {
                **error,
                "error": {"detail": {**error["error"]["detail"], "code": "other"}},  # type: ignore[index]
            },
        }
    )


def test_behavior_comparison_distinguishes_different_errors() -> None:
    baseline = real_artifact(
        repository="owner/model",
        revision=REVISION,
        version="2.1.0",
        confidence=0.8,
    )
    candidate = real_artifact(
        repository="owner/model",
        revision=REVISION,
        version="2.1.0",
        confidence=0.8,
    )
    baseline_case = baseline["http_runner"]["behavior_cases"][0]  # type: ignore[index]
    candidate_case = candidate["http_runner"]["behavior_cases"][0]  # type: ignore[index]
    for mode in ("singleton", "repeated", "batch"):
        baseline_case[mode] = {
            "status": "error",
            "http_status": 422,
            "error": {"code": "a"},
        }
        candidate_case[mode] = {
            "status": "error",
            "http_status": 422,
            "error": {"code": "b"},
        }

    comparison = compare_behavior_artifacts(baseline, candidate)

    assert comparison["cases"][0]["exact_output_match"] is False
    assert comparison["cases"][0]["behavioral_output_match"] is False
    assert comparison["cases"][0]["numeric_path_sets_match"] is None


def test_behavior_comparison_allows_runtime_and_revision_changes() -> None:
    baseline = real_artifact(
        repository="owner/model",
        revision="a" * 40,
        version="2.0.0",
        confidence=0.8,
        digest="a" * 64,
    )
    candidate = real_artifact(
        repository="owner/model",
        revision="b" * 40,
        version="2.1.0",
        confidence=0.7,
        digest="b" * 64,
    )

    comparison = compare_behavior_artifacts(
        baseline,
        candidate,
        allow_revision_change=True,
    )

    assert comparison["baseline"]["model_revision"] == "a" * 40
    assert comparison["candidate"]["gliner2_version"] == "2.1.0"
    assert comparison["summary"] == {
        "cases": 2,
        "exact_output_matches": 1,
        "behavioral_output_matches": 2,
        "token_usage_measured": 2,
        "token_usage_matches": 2,
        "decoder_metadata_measured": 1,
        "decoder_metadata_matches": 1,
        "baseline_mode_consistency_measured": 2,
        "baseline_mode_consistent": 2,
        "candidate_mode_consistency_measured": 2,
        "candidate_mode_consistent": 2,
        "numeric_path_sets_measured": 1,
        "numeric_path_set_matches": 1,
        "max_abs_numeric_difference": pytest.approx(0.1),
    }
    assert [item["case_id"] for item in comparison["cases"]] == ["one", "two"]


def test_behavior_comparison_reports_token_usage_changes() -> None:
    baseline = real_artifact(
        repository="owner/model",
        revision=REVISION,
        version="2.1.0",
        confidence=0.8,
    )
    candidate = real_artifact(
        repository="owner/model",
        revision=REVISION,
        version="2.1.0",
        confidence=0.8,
    )
    candidate["http_runner"]["behavior_cases"][0]["singleton"]["usage"]["inputTokens"] = 12  # type: ignore[index]

    comparison = compare_behavior_artifacts(baseline, candidate)

    assert comparison["summary"]["behavioral_output_matches"] == 2
    assert comparison["summary"]["token_usage_matches"] == 1


def test_behavior_comparison_requires_revision_change_opt_in() -> None:
    baseline = real_artifact(
        repository="owner/model",
        revision="a" * 40,
        version="2.0.0",
        confidence=0.8,
        digest="a" * 64,
    )
    candidate = real_artifact(
        repository="owner/model",
        revision="b" * 40,
        version="2.1.0",
        confidence=0.8,
        digest="b" * 64,
    )

    with pytest.raises(ValueError, match="without explicit opt-in"):
        compare_behavior_artifacts(baseline, candidate)


def test_behavior_comparison_rejects_different_workloads() -> None:
    baseline = real_artifact(
        repository="owner/model",
        revision="a" * 40,
        version="2.0.0",
        confidence=0.8,
    )
    candidate = real_artifact(
        repository="owner/model",
        revision="b" * 40,
        version="2.0.0",
        confidence=0.8,
        workload="other-workload",
    )

    with pytest.raises(ValueError, match="workload identities"):
        compare_behavior_artifacts(baseline, candidate, allow_revision_change=True)


def test_model_card_comparison_selects_explicit_profiles() -> None:
    baseline = {
        "model_repository": "owner/model",
        "model_revision": "a" * 40,
        "model_content_sha256": "a" * 64,
        "versions": {"gliner2": "2.0.0"},
        "behavior_workload_sha256": "card-workload",
        "profiles": [
            {
                "device": "cpu",
                "precision": "fp32",
                "examples": [
                    {
                        "name": "example",
                        "actual_output": {
                            "singleton": {"status": "success", "value": "a"},
                            "batch": {"status": "success", "value": "b"},
                        },
                    }
                ],
            }
        ],
    }
    candidate = {
        "model_repository": "owner/model",
        "model_revision": "b" * 40,
        "model_content_sha256": "b" * 64,
        "versions": {"gliner2": "2.1.0"},
        "behavior_workload_sha256": "card-workload",
        "profiles": [
            {
                "device": "mps",
                "precision": "fp16",
                "examples": [
                    {
                        "name": "example",
                        "actual_output": {
                            "singleton": {"status": "success", "value": "a"},
                            "batch": {"status": "success", "value": "b"},
                        },
                    }
                ],
            }
        ],
    }

    comparison = compare_behavior_artifacts(
        baseline,
        candidate,
        baseline_profile="cpu/fp32",
        candidate_profile="mps/fp16",
        allow_revision_change=True,
    )

    assert comparison["summary"]["exact_output_matches"] == 1
    assert comparison["summary"]["token_usage_measured"] == 0
    assert comparison["summary"]["decoder_metadata_measured"] == 0
    assert comparison["summary"]["baseline_mode_consistency_measured"] == 0
    assert comparison["summary"]["candidate_mode_consistency_measured"] == 0
    assert comparison["candidate"]["profile"] == "mps/fp16"


def test_model_card_comparison_requires_profile_when_ambiguous() -> None:
    artifact = {
        "model_repository": "owner/model",
        "model_revision": REVISION,
        "model_content_sha256": "f" * 64,
        "versions": {"gliner2": "2.1.0"},
        "behavior_workload_sha256": "card-workload",
        "profiles": [
            {
                "device": device,
                "precision": precision,
                "examples": [
                    {
                        "name": "example",
                        "actual_output": {
                            "singleton": {"status": "success", "value": "a"},
                            "batch": {"status": "success", "value": "b"},
                        },
                    }
                ],
            }
            for device, precision in (("cpu", "fp32"), ("mps", "fp16"))
        ],
    }

    with pytest.raises(ValueError, match="explicit profile"):
        compare_behavior_artifacts(artifact, artifact)
