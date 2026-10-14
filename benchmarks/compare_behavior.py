from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .benchmark_support import write_new_text
else:
    from benchmark_support import write_new_text  # type: ignore[import-not-found, no-redef]


def _numeric_values(value: Any, path: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(value, bool):
        return values
    if isinstance(value, int | float):
        values[path or "/"] = float(value)
    elif isinstance(value, Mapping):
        for key in sorted(value):
            values.update(_numeric_values(value[key], f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            values.update(_numeric_values(item, f"{path}/{index}"))
    return values


def _behavioral_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "value" in value and ("confidence" in value or "probabilities" in value):
            return _behavioral_value(value["value"])
        return {
            key: _behavioral_value(item)
            for key, item in sorted(value.items())
            if not (key == "objective" and "decoder" in value)
        }
    if isinstance(value, list):
        return [_behavioral_value(item) for item in value]
    return value


MODES = ("singleton", "repeated", "batch")


@dataclass(frozen=True)
class _RawOutput:
    value: Any


def _mode_records(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, _RawOutput):
        return {"singleton": {"status": "success", "output": value.value}}
    if isinstance(value, Mapping):
        present_modes = [mode for mode in MODES if mode in value]
        if present_modes and all(
            isinstance(value[mode], Mapping) and value[mode].get("status") in {"success", "error"}
            for mode in present_modes
        ):
            return {mode: value[mode] for mode in present_modes}
    raise ValueError("behavior case has no valid execution-mode records")


def _semantic_case(value: Any) -> dict[str, Any]:
    return {
        mode: {key: _behavioral_value(item) for key, item in record.items() if key != "usage"}
        for mode, record in _mode_records(value).items()
    }


def _usage_case(value: Any) -> dict[str, Any]:
    return {
        mode: record.get("usage")
        for mode, record in _mode_records(value).items()
        if "usage" in record
    }


def _decoder_case(value: Any) -> dict[str, Any]:
    decoder: dict[str, Any] = {}
    for mode, record in _mode_records(value).items():
        output = record.get("output")
        if isinstance(output, Mapping) and isinstance(output.get("_meta"), Mapping):
            decoder[mode] = _behavioral_value(output["_meta"])
    return decoder


def _exact_case(value: Any) -> dict[str, Any]:
    return {
        mode: {key: item for key, item in record.items() if key != "usage"}
        for mode, record in _mode_records(value).items()
    }


def _numeric_case(value: Any) -> dict[str, Any]:
    return {
        mode: record["output"]
        for mode, record in _mode_records(value).items()
        if "output" in record
    }


def _mode_consistent(value: Any) -> bool | None:
    records = list(_mode_records(value).values())
    if len(records) < 2:
        return None
    semantic_records = [
        _behavioral_value({key: item for key, item in record.items() if key != "usage"})
        for record in records
    ]
    return all(record == semantic_records[0] for record in semantic_records[1:])


def _optional_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool | None:
    if not left and not right:
        return None
    return left == right


def _profile(profiles: Sequence[Mapping[str, Any]], requested: str | None) -> Mapping[str, Any]:
    if requested is None:
        if not profiles:
            raise ValueError("model-card artifact has no profiles")
        if len(profiles) != 1:
            raise ValueError("select an explicit profile from the model-card artifact")
        return profiles[0]
    for profile in profiles:
        if f"{profile.get('device')}/{profile.get('precision')}" == requested:
            return profile
    raise ValueError(f"model-card artifact has no profile {requested}")


def _identity(artifact: Mapping[str, Any]) -> dict[str, str]:
    metadata = artifact.get("metadata", artifact)
    versions = metadata.get("versions", artifact.get("versions", {}))
    identity = {
        "model_repository": metadata.get("model_repository", metadata.get("model_id")),
        "model_revision": metadata.get("model_revision"),
        "model_content_sha256": metadata.get("model_content_sha256"),
        "gliner2_version": versions.get("gliner2"),
    }
    if not isinstance(identity["model_repository"], str) or "/" not in identity["model_repository"]:
        raise ValueError("behavior artifact has no explicit model repository")
    if (
        not isinstance(identity["model_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", identity["model_revision"]) is None
    ):
        raise ValueError("behavior artifact has no exact model revision")
    if (
        not isinstance(identity["model_content_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["model_content_sha256"]) is None
    ):
        raise ValueError("behavior artifact has no snapshot content digest")
    if not isinstance(identity["gliner2_version"], str) or not identity["gliner2_version"]:
        raise ValueError("behavior artifact has no GLiNER2 version")
    return identity


def _extract_cases(
    artifact: Mapping[str, Any],
    requested_profile: str | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    metadata = artifact.get("metadata", {})
    workload_sha256 = metadata.get("behavior_workload_sha256")
    if "http_runner" in artifact:
        if not isinstance(workload_sha256, str):
            raise ValueError("real-model artifact has no behavior workload identity")
        records = artifact["http_runner"].get("behavior_cases")
        if isinstance(records, list):
            cases: dict[str, Any] = {}
            for record in records:
                if not isinstance(record, Mapping) or not isinstance(record.get("case_id"), str):
                    raise ValueError("real-model artifact has an invalid behavior case")
                case_id = record["case_id"]
                if case_id in cases:
                    raise ValueError(f"duplicate real-model behavior case: {case_id}")
                _mode_records(record)
                cases[case_id] = record
            if list(cases) != metadata.get("behavior_case_ids"):
                raise ValueError("real-model behavior case identities do not match metadata")
        else:
            outputs = artifact["http_runner"].get("behavior_outputs", [])
            case_ids = metadata.get("behavior_case_ids", [])
            if len(outputs) != len(case_ids) or len(set(case_ids)) != len(case_ids):
                raise ValueError("real-model behavior case identities do not match its outputs")
            cases = {
                case_id: _RawOutput(output)
                for case_id, output in zip(case_ids, outputs, strict=True)
            }
        if not cases:
            raise ValueError("real-model artifact has no runner behavior cases")
        return (
            workload_sha256,
            cases,
            {
                "kind": "real-model-runner",
                "profile": f"{metadata.get('device')}/{metadata.get('precision')}",
            },
        )

    profiles = artifact.get("profiles")
    if isinstance(profiles, list):
        workload_sha256 = artifact.get("behavior_workload_sha256")
        if not isinstance(workload_sha256, str):
            raise ValueError("model-card artifact has no behavior workload identity")
        selected = _profile(profiles, requested_profile)
        card_cases: dict[str, Any] = {}
        for example in selected.get("examples", []):
            if "actual_output" not in example:
                raise ValueError(
                    f"model-card example did not produce output: {example.get('name')}"
                )
            name = example["name"]
            if name in card_cases:
                raise ValueError(f"duplicate model-card example name: {name}")
            card_cases[name] = _RawOutput(example["actual_output"])
        return (
            workload_sha256,
            card_cases,
            {
                "kind": "model-card",
                "profile": f"{selected.get('device')}/{selected.get('precision')}",
            },
        )

    raise ValueError("unrecognized behavior artifact")


def compare_behavior_artifacts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_profile: str | None = None,
    candidate_profile: str | None = None,
    allow_revision_change: bool = False,
) -> dict[str, Any]:
    """Compare behavior across runtimes, with explicit opt-in for revision changes."""
    baseline_identity = _identity(baseline)
    candidate_identity = _identity(candidate)
    if baseline_identity["model_repository"] != candidate_identity["model_repository"]:
        raise ValueError("behavior artifact model repositories do not match")
    if not allow_revision_change and (
        baseline_identity["model_revision"] != candidate_identity["model_revision"]
        or baseline_identity["model_content_sha256"] != candidate_identity["model_content_sha256"]
    ):
        raise ValueError("model revision or content changed without explicit opt-in")

    baseline_workload, baseline_cases, baseline_source = _extract_cases(baseline, baseline_profile)
    candidate_workload, candidate_cases, candidate_source = _extract_cases(
        candidate, candidate_profile
    )
    if baseline_workload != candidate_workload:
        raise ValueError("behavior workload identities do not match")
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("behavior case identities do not match")

    cases: list[dict[str, Any]] = []
    for case_id in baseline_cases:
        baseline_output = baseline_cases[case_id]
        candidate_output = candidate_cases[case_id]
        baseline_numbers = _numeric_values(_numeric_case(baseline_output))
        candidate_numbers = _numeric_values(_numeric_case(candidate_output))
        common_paths = baseline_numbers.keys() & candidate_numbers.keys()
        has_numeric_evidence = bool(baseline_numbers or candidate_numbers)
        maximum_difference = (
            max(
                (abs(baseline_numbers[path] - candidate_numbers[path]) for path in common_paths),
                default=0.0,
            )
            if has_numeric_evidence
            else None
        )
        baseline_usage = _usage_case(baseline_output)
        candidate_usage = _usage_case(candidate_output)
        baseline_decoder = _decoder_case(baseline_output)
        candidate_decoder = _decoder_case(candidate_output)
        cases.append(
            {
                "case_id": case_id,
                "exact_output_match": (
                    _exact_case(baseline_output) == _exact_case(candidate_output)
                ),
                "behavioral_output_match": (
                    _semantic_case(baseline_output) == _semantic_case(candidate_output)
                ),
                "token_usage_match": _optional_match(baseline_usage, candidate_usage),
                "decoder_metadata_match": _optional_match(baseline_decoder, candidate_decoder),
                "baseline_mode_consistent": _mode_consistent(baseline_output),
                "candidate_mode_consistent": _mode_consistent(candidate_output),
                "numeric_paths_compared": len(common_paths),
                "numeric_path_sets_match": (
                    baseline_numbers.keys() == candidate_numbers.keys()
                    if has_numeric_evidence
                    else None
                ),
                "max_abs_numeric_difference": maximum_difference,
            }
        )

    def measured_count(name: str) -> int:
        return sum(item[name] is not None for item in cases)

    def match_count(name: str) -> int:
        return sum(item[name] is True for item in cases)

    numeric_differences = [
        item["max_abs_numeric_difference"]
        for item in cases
        if item["max_abs_numeric_difference"] is not None
    ]
    return {
        "format_version": 2,
        "workload_sha256": baseline_workload,
        "baseline": {**baseline_identity, **baseline_source},
        "candidate": {**candidate_identity, **candidate_source},
        "summary": {
            "cases": len(cases),
            "exact_output_matches": match_count("exact_output_match"),
            "behavioral_output_matches": match_count("behavioral_output_match"),
            "token_usage_measured": measured_count("token_usage_match"),
            "token_usage_matches": match_count("token_usage_match"),
            "decoder_metadata_measured": measured_count("decoder_metadata_match"),
            "decoder_metadata_matches": match_count("decoder_metadata_match"),
            "baseline_mode_consistency_measured": measured_count("baseline_mode_consistent"),
            "baseline_mode_consistent": match_count("baseline_mode_consistent"),
            "candidate_mode_consistency_measured": measured_count("candidate_mode_consistent"),
            "candidate_mode_consistent": match_count("candidate_mode_consistent"),
            "numeric_path_sets_measured": measured_count("numeric_path_sets_match"),
            "numeric_path_set_matches": match_count("numeric_path_sets_match"),
            "max_abs_numeric_difference": (
                max(numeric_differences) if numeric_differences else None
            ),
        },
        "cases": cases,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Compare benchmark behavior across runtime or model revisions."
    )
    root.add_argument("--baseline", type=Path, required=True)
    root.add_argument("--candidate", type=Path, required=True)
    root.add_argument("--baseline-profile", help="Model-card device/precision profile.")
    root.add_argument("--candidate-profile", help="Model-card device/precision profile.")
    root.add_argument(
        "--allow-revision-change",
        action="store_true",
        help="Permit different revisions and snapshot content digests.",
    )
    root.add_argument("--output", type=Path, help="Write JSON here instead of stdout.")
    return root


def main() -> None:
    arguments = parser().parse_args()
    baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(arguments.candidate.read_text(encoding="utf-8"))
    result = compare_behavior_artifacts(
        baseline,
        candidate,
        baseline_profile=arguments.baseline_profile,
        candidate_profile=arguments.candidate_profile,
        allow_revision_change=arguments.allow_revision_change,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        write_new_text(arguments.output, rendered)


if __name__ == "__main__":
    main()
