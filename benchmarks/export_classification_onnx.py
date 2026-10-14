from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from gliner_runner.onnx_prepare import (  # noqa: F401
    DEFAULT_ATOL,
    INPUT_NAMES,
    MAX_ATOL,
    MAX_SYNTHETIC_ZERO_THRESHOLD,
    OPSET,
    OUTPUT_NAME,
    _make_batch,
    build_classification_wrapper,
    compare_logits,
    session_contract,
    validate_atol,
    validate_comparison_synthetic_zero_threshold,
    validate_external_data_files,
    validate_graph_contract,
    validate_masked_logits,
    validate_opset_imports,
)

for _name, _value in {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_name] = _value

if __package__:
    from .benchmark_support import (
        prepare_output_directory,
        snapshot_tree_digest,
        stable_sha256,
        validate_gliner2_version,
        validate_model_snapshot,
    )
else:
    from benchmark_support import (  # type: ignore[import-not-found, no-redef]
        prepare_output_directory,
        snapshot_tree_digest,
        stable_sha256,
        validate_gliner2_version,
        validate_model_snapshot,
    )

SMALL_REPOSITORY = "fastino/gliner2.5-small-v1"
SMALL_REVISION = "df5910e44bc4ffdb0d95399a83b0ca4516349aa5"
DECIDE_REPOSITORY = "fastino/GLiNER2.5-Decide"
DECIDE_REVISION = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90"
MULTILINGUAL_REPOSITORY = "fastino/GLiNER2.5-multi-Decide"
MULTILINGUAL_REVISION = "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"
DECIDE_1B_REPOSITORY = "fastino/GLiNER2.5-Decide-1B"
DECIDE_1B_REVISION = "67af949e17bebd20f05fe020a971079889e7a462"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExportTargetSpec:
    repository: str
    revision: str
    target_kind: str
    display_name: str
    fixture_suite: str


@dataclass(frozen=True)
class FixtureSuite:
    suite_id: str
    languages: tuple[str, ...]
    texts: tuple[tuple[str, ...], ...]


ENGLISH_FIXTURES = FixtureSuite(
    suite_id="classification-logits-english-v1",
    languages=("en",),
    texts=(
        ("The invoice contains a duplicate charge.",),
        (
            "The application crashes.",
            "The package is late and I cannot access my account after changing my password.",
        ),
    ),
)
MULTILINGUAL_FIXTURES = FixtureSuite(
    suite_id="classification-logits-multilingual-v1",
    languages=("es", "fr", "de"),
    texts=(
        ("La factura contiene un cargo duplicado.",),
        (
            "L'application plante.",
            (
                "Das Paket ist verspätet und ich kann nach der Passwortänderung "
                "nicht auf mein Konto zugreifen."
            ),
        ),
    ),
)
FIXTURE_SUITES: Mapping[str, FixtureSuite] = MappingProxyType(
    {
        ENGLISH_FIXTURES.suite_id: ENGLISH_FIXTURES,
        MULTILINGUAL_FIXTURES.suite_id: MULTILINGUAL_FIXTURES,
    }
)
EXPORT_TARGETS: Mapping[str, ExportTargetSpec] = MappingProxyType(
    {
        SMALL_REPOSITORY: ExportTargetSpec(
            repository=SMALL_REPOSITORY,
            revision=SMALL_REVISION,
            target_kind="small",
            display_name="Small",
            fixture_suite=ENGLISH_FIXTURES.suite_id,
        ),
        DECIDE_REPOSITORY: ExportTargetSpec(
            repository=DECIDE_REPOSITORY,
            revision=DECIDE_REVISION,
            target_kind="decide",
            display_name="English Decide",
            fixture_suite=ENGLISH_FIXTURES.suite_id,
        ),
        MULTILINGUAL_REPOSITORY: ExportTargetSpec(
            repository=MULTILINGUAL_REPOSITORY,
            revision=MULTILINGUAL_REVISION,
            target_kind="decide-multilingual",
            display_name="Multilingual Decide",
            fixture_suite=MULTILINGUAL_FIXTURES.suite_id,
        ),
        DECIDE_1B_REPOSITORY: ExportTargetSpec(
            repository=DECIDE_1B_REPOSITORY,
            revision=DECIDE_1B_REVISION,
            target_kind="decide-1b",
            display_name="Decide 1B",
            fixture_suite=ENGLISH_FIXTURES.suite_id,
        ),
    }
)


def export_target_spec(repository: str, revision: str) -> ExportTargetSpec:
    """Return one exact, reviewed research target specification."""
    spec = EXPORT_TARGETS.get(repository)
    if spec is None:
        supported = ", ".join(sorted(EXPORT_TARGETS))
        raise ValueError(f"unsupported ONNX research target; use one of: {supported}")
    if revision != spec.revision:
        raise ValueError(f"{spec.display_name} export requires {spec.repository}@{spec.revision}")
    return spec


def validate_export_target(repository: str, revision: str) -> str:
    """Return the target kind for one exact, reviewed research target."""
    return export_target_spec(repository, revision).target_kind


def fixture_suite_for_target(target: ExportTargetSpec) -> FixtureSuite:
    """Return the immutable fixture suite selected by a target specification."""
    return FIXTURE_SUITES[target.fixture_suite]


def validate_output_location(output: Path, snapshot: Path) -> Path:
    """Keep repository-local graphs in ignored research directories."""
    resolved = output.expanduser().resolve()
    model_root = snapshot.expanduser().resolve()
    if resolved == model_root or resolved.is_relative_to(model_root):
        raise ValueError("ONNX output must not be inside the model snapshot")
    if resolved.is_relative_to(REPOSITORY_ROOT):
        allowed_roots = (
            REPOSITORY_ROOT / "benchmark-results" / "local",
            REPOSITORY_ROOT / "dist",
        )
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise ValueError(
                "repository-local ONNX output must be under benchmark-results/local or dist"
            )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    """Record deterministic hashes for generated files except final reports."""
    artifacts: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"reproducibility.json", "REPORT.md"}:
            continue
        artifacts.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return artifacts


def generate_report(result: Mapping[str, Any]) -> str:
    parity = result["parity"]
    official_parity = result["official_parity"]
    graph = result["graph"]
    official_selected = official_parity["selected_index_matches"]
    official_rows = len(official_parity["rows"])
    official_difference = official_parity["max_abs_logit_difference"]
    synthetic_zero = parity.get("synthetic_zero_delta")
    if isinstance(synthetic_zero, Mapping):
        synthetic_zero_summary = (
            "- Synthetic-zero delta view: enabled, diagnostic only (does not affect "
            "logits, selection, or acceptance)\n"
            f"- Synthetic-zero threshold: {synthetic_zero['threshold']:.8g} "
            "(`strict-less-than`, valid labels only)\n"
            "- Synthetic-zero normalized maximum absolute difference: "
            f"{synthetic_zero['max_abs_normalized_difference']:.8g}\n"
            "- Positive differences converted to synthetic zero: "
            f"{synthetic_zero['nonzero_differences_zeroed']}/"
            f"{synthetic_zero['valid_deltas_compared']}"
        )
    else:
        synthetic_zero_summary = "- Synthetic-zero delta view: disabled"
    artifacts = "\n".join(
        f"| `{item['path']}` | {item['size']} | `{item['sha256']}` |"
        for item in result["artifacts"]
    )
    return f"""# ONNX classification feasibility

This is research evidence for a raw classification-logit graph. Tokenization,
schema compilation, constrained decoding, formatting, and usage accounting
remain outside ONNX. It does not advertise or register a production backend.

## Source and runtime

- Model: `{result["model_repository"]}@{result["model_revision"]}`
- Snapshot SHA-256: `{result["model_content_sha256"]}`
- GLiNER2: {result["versions"]["gliner2"]}
- PyTorch: {result["versions"]["torch"]}
- ONNX: {result["versions"]["onnx"]}
- ONNX Runtime: {result["versions"]["onnxruntime"]}
- Provider: `{result["provider"]}`
- Opset imports: {json.dumps(result["opset_imports"], sort_keys=True)}
- Absolute tolerance: {result["absolute_tolerance"]:.8g}
- Quantization: `{result["quantization"]}`
- Fixture suite: `{result["fixture_suite"]}`
- Fixture languages: {", ".join(result["fixture_languages"])}

## Contract and parity

- Inputs: {", ".join(item["name"] for item in graph["inputs"])}
- Output: `{graph["outputs"][0]["name"]}`
- Valid logits compared: {parity["valid_logits_compared"]}
- Selected-label parity: {parity["selected_index_matches"]}/{len(parity["rows"])}
- Raw maximum absolute logit difference: {parity["max_abs_logit_difference"]:.8g}
{synthetic_zero_summary}
- Official path: `{result["official_path"]}`
- Wrapper/official selected-label parity: {official_selected}/{official_rows}
- Wrapper/official maximum absolute logit difference: {official_difference:.8g}
- Dynamic fixture shapes: {", ".join(result["fixture_shapes"])}

## Generated artifacts

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
{artifacts}
"""


def _fixture_schemas(module: Any) -> list[Any]:
    return [
        module.ClassificationSchema.from_dict(
            {
                "version": 3,
                "tasks": {
                    "intent": {
                        "labels": {
                            "billing": "Invoices and duplicate charges",
                            "support": "Application failures and troubleshooting",
                            "shipping": "Delivery and package tracking",
                        },
                        "min_labels": 1,
                        "max_labels": 1,
                    }
                },
                "constraints": [],
            }
        ),
        module.ClassificationSchema.from_dict(
            {
                "version": 3,
                "tasks": {
                    "intent": {
                        "labels": ["billing", "support"],
                        "min_labels": 1,
                        "max_labels": 1,
                    }
                },
                "constraints": [],
            }
        ),
        module.ClassificationSchema.from_dict(
            {
                "version": 3,
                "tasks": {
                    "intent": {
                        "labels": ["billing", "support", "shipping", "account"],
                        "min_labels": 1,
                        "max_labels": 1,
                    }
                },
                "constraints": [],
            }
        ),
    ]


def _official_logits(
    classifier: Any,
    rows: Sequence[tuple[str, Any]],
    module: Any,
) -> list[list[float]]:
    scores = classifier.batch_score(
        [text for text, _schema in rows],
        [schema for _text, schema in rows],
        config=module.ClassificationConfig(batch_size=len(rows)),
    )
    logits: list[list[float]] = []
    for score in scores:
        task_scores = score.tasks.get("intent")
        if not isinstance(task_scores, Mapping) or not task_scores:
            raise RuntimeError("official GLiNER2 scorer did not return intent logits")
        logits.append([float(value) for value in task_scores.values()])
    return logits


def run(arguments: argparse.Namespace) -> None:
    if arguments.expected_gliner2_version != "2.1.0":
        raise ValueError("the research exporter is pinned to GLiNER2 2.1.0 internals")
    target = export_target_spec(arguments.model_repository, arguments.model_revision)
    target_kind = target.target_kind
    fixture_suite = fixture_suite_for_target(target)
    model_path = validate_model_snapshot(
        arguments.model_path,
        arguments.model_repository,
        arguments.model_revision,
    )
    output_location = validate_output_location(arguments.output_dir, model_path)
    atol = validate_atol(arguments.atol)
    synthetic_zero_threshold = validate_comparison_synthetic_zero_threshold(
        getattr(arguments, "comparison_synthetic_zero_threshold", None)
    )
    validate_gliner2_version(arguments.expected_gliner2_version)
    if output_location.exists():
        raise FileExistsError(f"benchmark output path already exists: {output_location}")

    import onnx  # type: ignore[import-not-found]
    import onnxruntime as ort  # type: ignore[import-not-found]
    import torch
    from gliner2 import classification as classification_module
    from gliner2.classification import Classifier

    output_dir = prepare_output_directory(output_location)
    reports_dir = output_dir / "export-report"
    reports_dir.mkdir()
    model_sha256, model_bytes, model_files = snapshot_tree_digest(model_path)
    classifier = Classifier.from_pretrained(
        str(model_path),
        attention_backend="standard",
        device="cpu",
        dtype="float32",
    ).eval()
    wrapper = build_classification_wrapper(torch, classifier.model)
    schemas = _fixture_schemas(classification_module)
    fixture_schema_indices = ((0,), (1, 2))
    fixtures = [
        [
            (text, schemas[schema_index])
            for text, schema_index in zip(texts, schema_indices, strict=True)
        ]
        for texts, schema_indices in zip(
            fixture_suite.texts,
            fixture_schema_indices,
            strict=True,
        )
    ]
    batches = [_make_batch(classifier, rows) for rows in fixtures]
    example_batch = batches[1]
    example_args = (
        example_batch.input_ids,
        example_batch.attention_mask,
        example_batch.cls_marker_indices,
        example_batch.cls_marker_mask,
    )
    batch_axis = torch.export.Dim("batch", min=1)
    sequence_axis = torch.export.Dim("sequence", min=2)
    label_axis = torch.export.Dim("labels", min=1)
    dynamic_shapes = {
        "input_ids": {0: batch_axis, 1: sequence_axis},
        "attention_mask": {0: batch_axis, 1: sequence_axis},
        "label_positions": {0: batch_axis, 1: label_axis},
        "label_mask": {0: batch_axis, 1: label_axis},
    }
    graph_path = output_dir / "classifier.onnx"
    program = torch.onnx.export(
        wrapper,
        example_args,
        dynamo=True,
        opset_version=OPSET,
        input_names=list(INPUT_NAMES),
        output_names=[OUTPUT_NAME],
        dynamic_shapes=dynamic_shapes,
        optimize=True,
        verify=True,
        report=True,
        artifacts_dir=str(reports_dir),
    )
    program.save(str(graph_path), external_data=False)
    onnx.checker.check_model(str(graph_path))
    onnx_model = onnx.load(str(graph_path), load_external_data=False)
    opset_imports = validate_opset_imports(
        [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in onnx_model.opset_import
        ]
    )
    external_data_files = validate_external_data_files(
        output_dir,
        sorted(
            {
                entry.value
                for initializer in onnx_model.graph.initializer
                for entry in initializer.external_data
                if entry.key == "location"
            }
        ),
    )
    if target_kind == "decide-1b" and not external_data_files:
        raise RuntimeError("Decide 1B export must use validated ONNX external data")

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    graph = session_contract(session)
    fixture_results: list[dict[str, Any]] = []
    fixture_shapes: list[str] = []
    all_expected: list[list[float]] = []
    all_actual: list[list[float]] = []
    all_official: list[list[float]] = []
    all_masks: list[list[bool]] = []
    for fixture_rows, batch in zip(fixtures, batches, strict=True):
        inputs = (
            batch.input_ids,
            batch.attention_mask,
            batch.cls_marker_indices,
            batch.cls_marker_mask,
        )
        with torch.inference_mode():
            expected = wrapper(*inputs).detach().cpu().numpy()
        ort_inputs = {
            name: value.detach().cpu().numpy()
            for name, value in zip(INPUT_NAMES, inputs, strict=True)
        }
        actual = session.run([OUTPUT_NAME], ort_inputs)[0]
        mask = batch.cls_marker_mask.detach().cpu().numpy().astype(bool)
        if actual.shape != expected.shape or actual.shape != mask.shape:
            raise RuntimeError("ONNX output shape does not match PyTorch logits and mask")
        validate_masked_logits(expected[~mask].tolist())
        validate_masked_logits(actual[~mask].tolist())
        expected_rows = expected.tolist()
        actual_rows = actual.tolist()
        mask_rows = mask.tolist()
        official_rows = _official_logits(classifier, fixture_rows, classification_module)
        padded_official_rows: list[list[float]] = []
        for official_row, mask_row in zip(official_rows, mask_rows, strict=True):
            valid_count = sum(mask_row)
            if len(official_row) != valid_count:
                raise RuntimeError("official scorer label count does not match exported routes")
            padded_official_rows.append(official_row + [0.0] * (len(mask_row) - valid_count))
        comparison = compare_logits(
            expected_rows,
            actual_rows,
            mask_rows,
            synthetic_zero_threshold=synthetic_zero_threshold,
        )
        official_comparison = compare_logits(expected_rows, padded_official_rows, mask_rows)
        fixture_results.append(
            {
                "batch": int(actual.shape[0]),
                "sequence": int(batch.input_ids.shape[1]),
                "labels": int(actual.shape[1]),
                "comparison": comparison,
                "official_comparison": official_comparison,
            }
        )
        fixture_shapes.append(
            f"B={actual.shape[0]}/S={batch.input_ids.shape[1]}/K={actual.shape[1]}"
        )
        all_expected.extend(expected_rows)
        all_actual.extend(actual_rows)
        all_official.extend(padded_official_rows)
        all_masks.extend(mask_rows)

    parity = compare_logits(
        all_expected,
        all_actual,
        all_masks,
        synthetic_zero_threshold=synthetic_zero_threshold,
    )
    official_parity = compare_logits(all_expected, all_official, all_masks)
    if parity["selected_index_matches"] != len(parity["rows"]):
        raise RuntimeError("ONNX selected-label parity failed")
    if official_parity["selected_index_matches"] != len(official_parity["rows"]):
        raise RuntimeError("wrapper selected labels do not match the official GLiNER2 scorer")
    if official_parity["max_abs_logit_difference"] > atol:
        raise RuntimeError(
            "wrapper logits differ from the official GLiNER2 scorer beyond tolerance: "
            f"{official_parity['max_abs_logit_difference']} > {atol}"
        )
    if parity["max_abs_logit_difference"] > atol:
        raise RuntimeError(
            f"ONNX logit parity exceeded tolerance: {parity['max_abs_logit_difference']} > {atol}"
        )

    del session, wrapper, classifier
    gc.collect()
    artifacts = artifact_inventory(output_dir)
    artifact_paths = {item["path"] for item in artifacts}
    missing_sidecars = sorted(set(external_data_files) - artifact_paths)
    if missing_sidecars:
        raise RuntimeError(
            "ONNX external-data files are missing from the artifact inventory: "
            f"{', '.join(missing_sidecars)}"
        )
    result = {
        "format_version": 3,
        "kind": "classification-logits-research",
        "target_kind": target_kind,
        "model_repository": arguments.model_repository,
        "model_revision": arguments.model_revision,
        "model_content_sha256": model_sha256,
        "model_bytes": model_bytes,
        "model_files": model_files,
        "fixture_suite": fixture_suite.suite_id,
        "fixture_languages": list(fixture_suite.languages),
        "fixture_sha256": stable_sha256(
            {
                "suite": fixture_suite.suite_id,
                "languages": fixture_suite.languages,
                "rows": [
                    {"text": text, "schema": schema.to_dict()}
                    for rows in fixtures
                    for text, schema in rows
                ],
            }
        ),
        "versions": {
            "python": platform.python_version(),
            "gliner2": importlib.metadata.version("gliner2"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "onnx": importlib.metadata.version("onnx"),
            "onnxscript": importlib.metadata.version("onnxscript"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
        },
        "opset": OPSET,
        "opset_imports": opset_imports,
        "external_data_files": external_data_files,
        "absolute_tolerance": atol,
        "comparison_delta_policy": {
            "mode": (
                "raw" if synthetic_zero_threshold is None else "synthetic-zero-below-threshold"
            ),
            "threshold": synthetic_zero_threshold,
            "boundary": (None if synthetic_zero_threshold is None else "strict-less-than"),
            "scope": "pytorch-wrapper-vs-onnx-valid-label-logits-only",
            "affects_acceptance": False,
        },
        "provider": "CPUExecutionProvider",
        "quantization": "none",
        "graph": graph,
        "fixture_shapes": fixture_shapes,
        "fixtures": fixture_results,
        "parity": parity,
        "official_path": "gliner2.classification.Classifier.batch_score",
        "official_parity": official_parity,
        "artifacts": artifacts,
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(generate_report(result), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Export and verify the research-only GLiNER2 classification-logit graph."
    )
    root.add_argument("--model-path", type=Path, required=True)
    root.add_argument("--model-repository", required=True)
    root.add_argument("--model-revision", required=True)
    root.add_argument("--expected-gliner2-version", required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    root.add_argument(
        "--comparison-synthetic-zero-threshold",
        type=float,
        help=(
            "Report PyTorch/ONNX valid-logit deltas below this value as synthetic "
            "zero without changing raw parity gates."
        ),
    )
    return root


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
