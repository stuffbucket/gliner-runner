"""Build-time preparation of verified ONNX classification artifacts.

This module is the only place where a pinned Hugging Face snapshot is converted to the
Python-free runtime format. It runs during build, release, installation, or explicit model
preparation, never per request. Importing it needs neither PyTorch nor ONNX; the heavy
libraries are imported inside the functions that need them.

Artifact layout (``<output-dir>/onnx/<owner>--<name>/<revision>/``)::

    model.onnx            ONNX graph (opset 20, dynamic batch/sequence/label axes)
    model.onnx.data       canonical external-data sidecar holding every safetensors-derived weight
    tokenizer.json        copied verbatim from the snapshot
    tokenizer_config.json copied verbatim from the snapshot
    special_tokens_map.json  copied verbatim when the snapshot ships one
    gliner2.json          preprocessing facts a Python-free port needs beyond the tokenizer
    manifest.json         identity, provenance, graph contract, parity evidence, file hashes

The exported graph is canonicalized before it is verified: every initializer whose bytes come
from ``model.safetensors`` (unchanged, or as a 2-D transpose) becomes external data in
``model.onnx.data`` with a deterministic layout, so the Python-free runner can rebuild the
identical artifact from a tracked *template* package (``--template-dir``) and a pinned snapshot
without any conversion. See ``gliner_runner.onnx_template`` and
``dist/port-spec/onnx-template-contract.md``.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import mmap
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gliner_runner.model_inventory import KNOWN_MODELS, InventoryModelSpec
from gliner_runner.onnx_template import (
    DATA_ALIGNMENT,
    DATA_OUTPUT,
    GRAPH_OUTPUT,
    LARGE_TENSOR_BYTES,
    MANIFEST_OUTPUT,
    PREPROCESSING_OUTPUT,
    TEMPLATE_FILE,
    TEMPLATE_FORMAT_VERSION,
    TEMPLATE_GRAPH_FILE,
    TEMPLATE_KIND,
    WEIGHTS_FILE,
    SafetensorsIndex,
    assemble_from_template,
    plan_layout,
    read_safetensors_index,
    transposed_shape,
    validate_template,
    validate_template_manifest,
)
from gliner_runner.snapshot import file_sha256, snapshot_tree_digest, validate_model_snapshot

FORMAT_VERSION = 1
MANIFEST_KIND = "gliner-runner-onnx-classifier"
PREPROCESSING_KIND = "gliner-runner-gliner2-preprocessing"
EXPECTED_GLINER2_VERSION = "2.1.0"
OPSET = 20
PROVIDER = "CPUExecutionProvider"
PRECISION = "fp32"
QUANTIZATION = "none"
OPERATIONS = ("classify",)
DEFAULT_ATOL = 1e-4
MAX_ATOL = 1e-3
SYNTHETIC_ZERO_THRESHOLD = 1e-4
MAX_SYNTHETIC_ZERO_THRESHOLD = 1e-4
INPUT_NAMES = ("input_ids", "attention_mask", "label_positions", "label_mask")
OUTPUT_NAME = "logits"
GRAPH_FILE = "model.onnx"
MANIFEST_FILE = "manifest.json"
PREPROCESSING_FILE = "gliner2.json"
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILES = ("special_tokens_map.json",)
SPECIAL_TOKENS = (
    "[SEP_STRUCT]",
    "[SEP_TEXT]",
    "[P]",
    "[C]",
    "[E]",
    "[R]",
    "[L]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[DESCRIPTION]",
)
RESERVED_STRINGS = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "(",
    ")",
)
EXPORTER_PACKAGES = (
    ("gliner2", "gliner2"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("onnx", "onnx"),
    ("onnxscript", "onnxscript"),
    ("onnxruntime", "onnxruntime"),
)
FLOAT32_ITEMSIZE = 4
_OUTPUT_ORDER = {
    GRAPH_OUTPUT: 0,
    DATA_OUTPUT: 1,
    PREPROCESSING_OUTPUT: 2,
    "tokenizer.json": 3,
    "tokenizer_config.json": 4,
    "special_tokens_map.json": 5,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_OFFLINE_ENVIRONMENT = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}


# --------------------------------------------------------------------------------------
# Identity and argument validation
# --------------------------------------------------------------------------------------


def resolve_curated_model(
    model: str, revision: str | None = None
) -> tuple[InventoryModelSpec, str]:
    """Return the curated model spec and its pinned revision, rejecting anything else."""
    if not isinstance(model, str) or not model or model != model.strip():
        raise ValueError("model must be a curated logical id or repository")
    if "@" in model:
        raise ValueError("pass the revision with --revision, not as part of the model")
    matches = [spec for spec in KNOWN_MODELS if model in (spec.model_id, spec.repository)]
    if not matches:
        supported = ", ".join(f"{spec.model_id} ({spec.repository})" for spec in KNOWN_MODELS)
        raise ValueError(
            f"model {model!r} is not curated for ONNX preparation; use one of: {supported}"
        )
    spec = matches[0]
    if revision is not None:
        if _REVISION.fullmatch(revision) is None:
            raise ValueError("revision must be an exact 40-character lowercase commit SHA")
        if revision != spec.pinned_revision:
            raise ValueError(
                f"{spec.model_id} is supported only at its pinned revision "
                f"{spec.pinned_revision}; refusing {revision}"
            )
    return spec, spec.pinned_revision


def artifact_directory(output_dir: Path, spec: InventoryModelSpec, revision: str) -> Path:
    """Return ``<output-dir>/onnx/<owner>--<name>/<revision>`` for a curated model."""
    return (
        output_dir.expanduser().resolve() / "onnx" / spec.repository.replace("/", "--") / revision
    )


def validate_atol(value: float) -> float:
    """Require a finite positive tolerance within the reviewed bound."""
    if not math.isfinite(value) or value <= 0 or value > MAX_ATOL:
        raise ValueError(f"absolute tolerance must be finite and in (0, {MAX_ATOL}]")
    return value


def validate_comparison_synthetic_zero_threshold(value: float | None) -> float | None:
    """Validate an optional diagnostic threshold independently of acceptance."""
    if value is None:
        return None
    if not math.isfinite(value) or value <= 0 or value > MAX_SYNTHETIC_ZERO_THRESHOLD:
        raise ValueError(
            "comparison synthetic-zero threshold must be finite and in "
            f"(0, {MAX_SYNTHETIC_ZERO_THRESHOLD}]"
        )
    return value


def validate_output_location(output_dir: Path, snapshot: Path, target: Path) -> Path:
    """Keep artifacts out of the snapshot and refuse to reuse an existing artifact path."""
    root = output_dir.expanduser().resolve()
    model_root = snapshot.expanduser().resolve()
    if root == model_root or root.is_relative_to(model_root) or model_root.is_relative_to(root):
        raise ValueError("ONNX output directory and model snapshot must not contain each other")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"ONNX artifact path already exists: {target}")
    return target


# --------------------------------------------------------------------------------------
# Pure comparison and graph-contract helpers (shared with the research exporter)
# --------------------------------------------------------------------------------------


def compare_logits(
    pytorch_logits: Sequence[Sequence[float]],
    onnx_logits: Sequence[Sequence[float]],
    label_mask: Sequence[Sequence[bool]],
    *,
    synthetic_zero_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare valid logits and selected indices without optional dependencies."""
    synthetic_zero_threshold = validate_comparison_synthetic_zero_threshold(
        synthetic_zero_threshold
    )
    if not (
        len(pytorch_logits) == len(onnx_logits) == len(label_mask)
        and all(
            len(expected) == len(actual) == len(mask)
            for expected, actual, mask in zip(pytorch_logits, onnx_logits, label_mask, strict=True)
        )
    ):
        raise ValueError("logit and label-mask shapes do not match")

    maximum = 0.0
    normalized_maximum = 0.0
    valid_values = 0
    below_threshold = 0
    nonzero_zeroed = 0
    selected_matches = 0
    rows: list[dict[str, Any]] = []
    for expected, actual, mask in zip(pytorch_logits, onnx_logits, label_mask, strict=True):
        valid = [index for index, enabled in enumerate(mask) if enabled]
        if not valid:
            raise ValueError("each fixture row must contain at least one valid label")
        expected_values = [float(expected[index]) for index in valid]
        actual_values = [float(actual[index]) for index in valid]
        if not all(math.isfinite(value) for value in (*expected_values, *actual_values)):
            raise ValueError("valid logits must all be finite")
        differences = [
            abs(expected_value - actual_value)
            for expected_value, actual_value in zip(expected_values, actual_values, strict=True)
        ]
        if not all(math.isfinite(value) for value in differences):
            raise ValueError("logit differences must all be finite")
        maximum = max(maximum, *differences)
        valid_values += len(valid)
        expected_index = max(valid, key=lambda index: float(expected[index]))
        actual_index = max(valid, key=lambda index: float(actual[index]))
        selected_matches += expected_index == actual_index
        row: dict[str, Any] = {
            "valid_labels": len(valid),
            "pytorch_selected_index": expected_index,
            "onnx_selected_index": actual_index,
            "selected_index_match": expected_index == actual_index,
            "max_abs_logit_difference": max(differences),
        }
        if synthetic_zero_threshold is not None:
            normalized = [
                0.0 if difference < synthetic_zero_threshold else difference
                for difference in differences
            ]
            row_below_threshold = sum(
                difference < synthetic_zero_threshold for difference in differences
            )
            row_nonzero_zeroed = sum(
                0.0 < difference < synthetic_zero_threshold for difference in differences
            )
            row_normalized_maximum = max(normalized)
            normalized_maximum = max(normalized_maximum, row_normalized_maximum)
            below_threshold += row_below_threshold
            nonzero_zeroed += row_nonzero_zeroed
            row["synthetic_zero_delta"] = {
                "threshold": synthetic_zero_threshold,
                "boundary": "strict-less-than",
                "valid_deltas_compared": len(differences),
                "differences_below_threshold": row_below_threshold,
                "nonzero_differences_zeroed": row_nonzero_zeroed,
                "max_abs_normalized_difference": row_normalized_maximum,
            }
        rows.append(row)
    result: dict[str, Any] = {
        "rows": rows,
        "valid_logits_compared": valid_values,
        "selected_index_matches": selected_matches,
        "max_abs_logit_difference": maximum,
    }
    if synthetic_zero_threshold is not None:
        result["synthetic_zero_delta"] = {
            "threshold": synthetic_zero_threshold,
            "boundary": "strict-less-than",
            "valid_deltas_compared": valid_values,
            "differences_below_threshold": below_threshold,
            "nonzero_differences_zeroed": nonzero_zeroed,
            "max_abs_normalized_difference": normalized_maximum,
        }
    return result


def compare_task_logits(
    expected: Sequence[Sequence[float]],
    actual: Sequence[Sequence[float]],
    label_mask: Sequence[Sequence[bool]],
    task_sizes: Sequence[Sequence[int]],
    *,
    synthetic_zero_threshold: float = SYNTHETIC_ZERO_THRESHOLD,
) -> dict[str, Any]:
    """Compare valid logits and the per-task argmax of every (row, task) selection.

    ``task_sizes[row]`` lists each task's label count in schema order; the valid labels of a
    row must be exactly their concatenation followed by padding.
    """
    if not (len(expected) == len(actual) == len(label_mask) == len(task_sizes)):
        raise ValueError("logit, label-mask, and task-size row counts do not match")
    selections: list[dict[str, Any]] = []
    maximum = 0.0
    compared = 0
    zeroed = 0
    below = 0
    matches = 0
    for row, (expected_row, actual_row, mask, sizes) in enumerate(
        zip(expected, actual, label_mask, task_sizes, strict=True)
    ):
        if not (len(expected_row) == len(actual_row) == len(mask)):
            raise ValueError("logit and label-mask shapes do not match")
        valid_count = sum(1 for enabled in mask if enabled)
        if (
            not sizes
            or any(size < 1 for size in sizes)
            or sum(sizes) != valid_count
            or not all(mask[:valid_count])
        ):
            raise ValueError("label mask must be a prefix matching the task label counts")
        offset = 0
        for task, size in enumerate(sizes):
            window = range(offset, offset + size)
            offset += size
            expected_values = [float(expected_row[index]) for index in window]
            actual_values = [float(actual_row[index]) for index in window]
            if not all(math.isfinite(value) for value in (*expected_values, *actual_values)):
                raise ValueError("valid logits must all be finite")
            differences = [
                abs(left - right)
                for left, right in zip(expected_values, actual_values, strict=True)
            ]
            expected_index = expected_values.index(max(expected_values))
            actual_index = actual_values.index(max(actual_values))
            matches += expected_index == actual_index
            compared += size
            maximum = max(maximum, *differences)
            zeroed += sum(1 for value in differences if 0.0 < value < synthetic_zero_threshold)
            below += sum(1 for value in differences if value < synthetic_zero_threshold)
            selections.append(
                {
                    "row": row,
                    "task": task,
                    "labels": size,
                    "expected_selected_index": expected_index,
                    "actual_selected_index": actual_index,
                    "selected_index_match": expected_index == actual_index,
                    "max_abs_logit_difference": max(differences),
                }
            )
    return {
        "selections": selections,
        "valid_logits_compared": compared,
        "selected_index_matches": matches,
        "max_abs_logit_difference": maximum,
        "nonzero_differences_zeroed": zeroed,
        "differences_below_threshold": below,
    }


def validate_masked_logits(values: Sequence[float]) -> None:
    """Require finite, strongly negative logits at every padded label position."""
    if any(not math.isfinite(float(value)) or float(value) > -1e20 for value in values):
        raise ValueError("padded label logits must be finite and strongly negative")


def validate_graph_contract(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the public graph names, element types, ranks, and dynamic axes."""
    by_name = {item.get("name"): item for item in inputs}
    if set(by_name) != set(INPUT_NAMES):
        raise ValueError(f"unexpected ONNX inputs: {sorted(str(name) for name in by_name)}")
    expected_types = {
        "input_ids": "tensor(int64)",
        "attention_mask": "tensor(int64)",
        "label_positions": "tensor(int64)",
        "label_mask": "tensor(bool)",
    }
    shapes: dict[str, list[Any]] = {}
    for name, expected_type in expected_types.items():
        item = by_name[name]
        if item.get("type") != expected_type:
            raise ValueError(f"{name} has type {item.get('type')}, expected {expected_type}")
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"{name} must be rank 2")
        if not all(isinstance(dimension, str) and dimension for dimension in shape):
            raise ValueError(f"{name} must expose nonempty dynamic batch and width symbols")
        shapes[name] = shape
    if len(outputs) != 1 or outputs[0].get("name") != OUTPUT_NAME:
        raise ValueError("ONNX graph must expose one logits output")
    output = outputs[0]
    if output.get("type") != "tensor(float)":
        raise ValueError("logits output must be float32")
    output_shape = output.get("shape")
    if not isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError("logits output must be rank 2")
    if not all(isinstance(dimension, str) and dimension for dimension in output_shape):
        raise ValueError("logits must expose nonempty dynamic batch and label symbols")

    batch_symbol = shapes["input_ids"][0]
    sequence_symbol = shapes["input_ids"][1]
    label_symbol = shapes["label_positions"][1]
    if any(shapes[name][0] != batch_symbol for name in INPUT_NAMES):
        raise ValueError("all inputs must share one dynamic batch symbol")
    if shapes["attention_mask"][1] != sequence_symbol:
        raise ValueError("input_ids and attention_mask must share one sequence symbol")
    if shapes["label_mask"][1] != label_symbol:
        raise ValueError("label_positions and label_mask must share one label symbol")
    if output_shape != [batch_symbol, label_symbol]:
        raise ValueError("logits must share the input batch and label symbols")
    return {
        "inputs": [dict(by_name[name]) for name in INPUT_NAMES],
        "outputs": [dict(output)],
    }


def validate_opset_imports(imports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Require the emitted graph to import the reviewed standard ONNX opset."""
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in imports:
        domain = item.get("domain") or "ai.onnx"
        version = item.get("version")
        if not isinstance(domain, str) or not domain or domain in seen:
            raise ValueError("ONNX graph contains an invalid or duplicate opset domain")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise ValueError(f"ONNX graph has an invalid opset version for {domain}")
        seen.add(domain)
        normalized.append({"domain": domain, "version": version})
    standard = next((item for item in normalized if item["domain"] == "ai.onnx"), None)
    if standard is None or standard["version"] != OPSET:
        raise ValueError(f"ONNX graph must import standard opset {OPSET}")
    return sorted(normalized, key=lambda item: str(item["domain"]))


def validate_external_data_files(root: Path, locations: Sequence[str]) -> list[str]:
    """Require every graph sidecar to resolve to a regular output file."""
    resolved_root = root.expanduser().resolve()
    unique: set[str] = set()
    for location in locations:
        if not location or location in unique:
            raise ValueError("ONNX graph contains an invalid or duplicate external-data location")
        sidecar = (resolved_root / location).resolve()
        if not sidecar.is_relative_to(resolved_root) or sidecar == resolved_root:
            raise ValueError(f"ONNX external-data path escapes the output directory: {location}")
        if not sidecar.is_file():
            raise ValueError(f"ONNX external-data file is missing: {location}")
        unique.add(location)
    return sorted(unique)


def validate_external_data_inventory(
    root: Path,
    entries: Iterable[tuple[str, int, int]],
    graph_file: str = GRAPH_FILE,
) -> list[str]:
    """Verify every external tensor range lies inside its sidecar and no sidecar is unused.

    ``entries`` holds ``(location, offset, length)`` for every external tensor in the graph.
    Returns the sorted sidecar locations. Files other than the graph and its sidecars are
    ignored here; the file inventory accounts for them separately.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    for location, offset, length in entries:
        if offset < 0 or length < 0:
            raise ValueError(f"ONNX external tensor has an invalid range in {location}")
        ranges.setdefault(location, []).append((offset, length))
    locations = validate_external_data_files(root, sorted(ranges))
    resolved_root = root.expanduser().resolve()
    for location in locations:
        size = (resolved_root / location).stat().st_size
        if any(offset + length > size for offset, length in ranges[location]):
            raise ValueError(f"ONNX external-data file is truncated: {location}")
    expected_prefix = f"{graph_file}."
    unreferenced = sorted(
        path.name
        for path in resolved_root.iterdir()
        if path.is_file() and path.name.startswith(expected_prefix) and path.name not in locations
    )
    if unreferenced:
        raise ValueError(f"unreferenced ONNX sidecar file present: {unreferenced[0]}")
    return locations


# --------------------------------------------------------------------------------------
# Fixtures: multi-task schemas with padding in both the token and label dimensions
# --------------------------------------------------------------------------------------

_DEPARTMENT_LABELS = {
    "billing": "Invoices, charges, refunds, and payment problems",
    "technical_support": "Application failures, defects, and technical troubleshooting",
    "account_access": "Authentication, passwords, and account sign-in problems",
    "shipping": "Package delivery, tracking, and shipment status",
}
_TOPIC_LABELS = ["billing", "technical_support", "account_access", "shipping"]
_FULL_SCHEMA: dict[str, Any] = {
    "version": 3,
    "tasks": {
        "department": {
            "labels": _DEPARTMENT_LABELS,
            "min_labels": 1,
            "max_labels": 1,
            "instruction": "Choose the department that should handle the customer message",
        },
        "topics": {
            "labels": _TOPIC_LABELS,
            "min_labels": 0,
            "max_labels": 3,
            "activation": "sigmoid",
            "threshold": 0.3,
        },
        "urgency": {
            "labels": ["low", "medium", "high"],
            "min_labels": 1,
            "max_labels": 1,
            "ordered": True,
        },
    },
    "constraints": [],
}
_TWO_TASK_SCHEMA: dict[str, Any] = {
    "version": 3,
    "tasks": {
        "sentiment": {
            "labels": {
                "positive": "The customer is satisfied",
                "negative": "The customer is unhappy",
                "neutral": "No clear emotion",
            },
            "min_labels": 1,
            "max_labels": 1,
        },
        "topics": {
            "labels": _TOPIC_LABELS,
            "min_labels": 0,
            "max_labels": 2,
            "activation": "sigmoid",
        },
    },
    "constraints": [],
}
_DESCRIBED_SCHEMA: dict[str, Any] = {
    "version": 3,
    "tasks": {
        "intent": {
            "labels": _DEPARTMENT_LABELS,
            "min_labels": 1,
            "max_labels": 1,
        }
    },
    "constraints": [],
}
_SMALL_SCHEMA: dict[str, Any] = {
    "version": 3,
    "tasks": {"intent": {"labels": ["billing", "support"], "min_labels": 1, "max_labels": 1}},
    "constraints": [],
}
_SCHEMAS: Mapping[str, dict[str, Any]] = {
    "full": _FULL_SCHEMA,
    "two-task": _TWO_TASK_SCHEMA,
    "described": _DESCRIBED_SCHEMA,
    "small": _SMALL_SCHEMA,
}
# Each batch lists (text index, schema key). Batches mix lengths, schemas, and label counts.
_FIXTURE_BATCHES: tuple[tuple[tuple[int, str], ...], ...] = (
    ((0, "full"),),
    ((1, "small"), (4, "full"), (5, "two-task")),
    ((5, "full"), (1, "full")),
    ((2, "described"), (5, "described"), (3, "described"), (0, "described")),
)
_FIXTURE_TEXTS: Mapping[str, tuple[str, ...]] = {
    "english": (
        "The invoice contains a duplicate charge.",
        "The application crashes.",
        "My credit card was charged twice for the same invoice.",
        "Where is the package that was supposed to arrive yesterday?",
        "I cannot sign in after resetting my password and I need help right away.",
        (
            "The customer reported that the desktop application crashes whenever a report is "
            "exported, and after the update their account sign-in also stopped working, so "
            "they would like the delivery of the replacement package to be tracked as well."
        ),
    ),
    "multilingual": (
        "La factura contiene un cargo duplicado.",
        "L'application plante.",
        "Mi tarjeta fue cobrada dos veces por la misma factura.",
        "昨日届くはずだった荷物はどこにありますか?",
        "Ich kann mich nach dem Zurücksetzen des Passworts nicht anmelden.",
        (
            "Das Paket ist verspätet und ich kann nach der Passwortänderung nicht auf mein "
            "Konto zugreifen, außerdem wurde meine Karte zweimal belastet und die application "
            "plante bei jedem Export."
        ),
    ),
}
_MULTILINGUAL_MODEL_IDS = frozenset({"decide-multilingual"})


@dataclass(frozen=True)
class FixtureRow:
    text: str
    schema: dict[str, Any]

    @property
    def task_sizes(self) -> list[int]:
        return [len(task["labels"]) for task in self.schema["tasks"].values()]


def fixture_suite_id(spec: InventoryModelSpec) -> str:
    language = "multilingual" if spec.model_id in _MULTILINGUAL_MODEL_IDS else "english"
    return f"classification-logits-multitask-{language}-v1"


def fixture_batches(spec: InventoryModelSpec) -> list[list[FixtureRow]]:
    """Return the deterministic parity fixture batches for a curated model."""
    language = "multilingual" if spec.model_id in _MULTILINGUAL_MODEL_IDS else "english"
    texts = _FIXTURE_TEXTS[language]
    return [
        [FixtureRow(texts[text_index], _SCHEMAS[schema_key]) for text_index, schema_key in batch]
        for batch in _FIXTURE_BATCHES
    ]


def stable_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with a deterministic encoding."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def fixture_sha256(spec: InventoryModelSpec) -> str:
    return stable_sha256(
        {
            "suite": fixture_suite_id(spec),
            "batches": [
                [{"text": row.text, "schema": row.schema} for row in batch]
                for batch in fixture_batches(spec)
            ],
        }
    )


# --------------------------------------------------------------------------------------
# File inventory, preprocessing config, manifest
# --------------------------------------------------------------------------------------


def file_inventory(
    root: Path, *, exclude: Iterable[str] = (MANIFEST_FILE,)
) -> list[dict[str, Any]]:
    """Return deterministic ``{path, size, sha256}`` entries for every regular file."""
    excluded = set(exclude)
    resolved = root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = path.relative_to(resolved).as_posix()
        if relative in excluded:
            continue
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)})
    return entries


def special_token_ids_from_tokenizer_json(tokenizer: Mapping[str, Any]) -> dict[str, int]:
    """Read the ten GLiNER2 marker ids from a ``tokenizer.json`` added-token table."""
    added = tokenizer.get("added_tokens")
    if not isinstance(added, list):
        raise ValueError("tokenizer.json has no added_tokens table")
    by_content: dict[str, int] = {}
    for item in added:
        if not isinstance(item, dict):
            continue
        content, identifier = item.get("content"), item.get("id")
        if (
            isinstance(content, str)
            and isinstance(identifier, int)
            and not isinstance(identifier, bool)
        ):
            by_content[content] = identifier
    missing = [token for token in SPECIAL_TOKENS if token not in by_content]
    if missing:
        raise ValueError(f"tokenizer.json is missing GLiNER2 marker tokens: {', '.join(missing)}")
    ids = {token: by_content[token] for token in SPECIAL_TOKENS}
    if len(set(ids.values())) != len(ids) or any(value < 0 for value in ids.values()):
        raise ValueError("GLiNER2 marker token ids must be distinct non-negative integers")
    return ids


def build_preprocessing_config(
    *,
    spec: InventoryModelSpec,
    revision: str,
    special_token_ids: Mapping[str, int],
    unknown_token: tuple[str, int] | None,
    model_config: Mapping[str, Any],
    tokenizer_config: Mapping[str, Any],
    has_special_tokens_map: bool,
) -> dict[str, Any]:
    """Describe the GLiNER2 2.1.0 classification preprocessing a Python-free port must apply."""
    ids = {token: special_token_ids[token] for token in SPECIAL_TOKENS}
    architecture = model_config.get("architecture")
    pooling = model_config.get("token_pooling")
    if pooling != "first":
        raise ValueError(f"unsupported token pooling {pooling!r}; the port implements 'first'")
    max_len = model_config.get("max_len")
    return {
        "format_version": FORMAT_VERSION,
        "kind": PREPROCESSING_KIND,
        "gliner2_version": EXPECTED_GLINER2_VERSION,
        "model": {
            "model_id": spec.model_id,
            "repository": spec.repository,
            "revision": revision,
            "architecture": architecture,
            "encoder": model_config.get("model_name"),
            "token_pooling": pooling,
            "snapshot_config_max_len": max_len,
        },
        "tokenizer": {
            "tokenizer_json": "tokenizer.json",
            "tokenizer_config_json": "tokenizer_config.json",
            "special_tokens_map_json": "special_tokens_map.json"
            if has_special_tokens_map
            else None,
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "encode_each_word_independently": True,
            "add_special_tokens": False,
            "unknown_token": (
                {"token": unknown_token[0], "id": unknown_token[1]} if unknown_token else None
            ),
        },
        "special_tokens": ids,
        "markers": {
            "structure_separator": "[SEP_STRUCT]",
            "text_separator": "[SEP_TEXT]",
            "prompt": "[P]",
            "label": "[L]",
            "description": "[DESCRIPTION]",
            "example": "[EXAMPLE]",
            "output": "[OUTPUT]",
        },
        "reserved_strings": list(RESERVED_STRINGS),
        "text": {
            "normalization": {
                "empty_text": ".",
                "terminal_characters": [".", "!", "?"],
                "append_when_missing": ".",
            },
            "word_splitter": "whitespace",
            "lowercase_words": True,
            "lowercase_scope": "text-words-only",
            "max_words": None,
        },
        "prompt": {
            "task_words": ["(", "[P]", "<prompt>", "(", "[L]", "<label>", "...", ")", ")"],
            "prompt_text": "<task>[: <instruction>]",
            "description_suffix": " [DESCRIPTION] <label>: <description>",
            "example_suffix": " [EXAMPLE] <input> [OUTPUT] <label>",
            "task_order": "schema declaration order",
        },
        "batching": {
            "padding_side": "right",
            "input_ids_pad_value": 0,
            "attention_mask_pad_value": 0,
            "label_positions_pad_value": 0,
            "label_mask_pad_value": False,
        },
        "label_positions": {
            "marker": "[L]",
            "scope": "all tasks concatenated in schema order, labels in declaration order",
            "position": "index of the marker subword in input_ids",
        },
        "usage": {"input_tokens": "sum of attention_mask for the request row"},
    }


def build_manifest(
    *,
    spec: InventoryModelSpec,
    revision: str,
    source_snapshot_sha256: str,
    source_snapshot_bytes: int,
    parameter_count: int,
    graph: Mapping[str, Any],
    opset_imports: Sequence[Mapping[str, Any]],
    external_data_files: Sequence[str],
    exporter: Mapping[str, str],
    parity: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
    license_id: str | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": MANIFEST_KIND,
        "model_id": spec.model_id,
        "repository": spec.repository,
        "revision": revision,
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_snapshot_bytes": source_snapshot_bytes,
        "parameter_count": parameter_count,
        "license": license_id,
        "operations": list(OPERATIONS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "provider": PROVIDER,
        "opset": OPSET,
        "opset_imports": [dict(item) for item in opset_imports],
        "preprocessing": PREPROCESSING_FILE,
        "graph": {
            "path": GRAPH_FILE,
            "external_data": list(external_data_files),
            "inputs": [dict(item) for item in graph["inputs"]],
            "outputs": [dict(item) for item in graph["outputs"]],
        },
        "exporter": dict(exporter),
        "parity": dict(parity),
        "files": [dict(item) for item in files],
    }
    validate_manifest(manifest)
    return manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid ONNX artifact manifest: {message}")


def _dict(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid ONNX artifact manifest: {what} must be an object")
    return value


def _list(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"invalid ONNX artifact manifest: {what} must be a list")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless a manifest is complete, consistent, and within its own gates."""
    constants = {
        "format_version": FORMAT_VERSION,
        "kind": MANIFEST_KIND,
        "operations": list(OPERATIONS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "provider": PROVIDER,
        "opset": OPSET,
        "preprocessing": PREPROCESSING_FILE,
    }
    for key, expected in constants.items():
        _require(manifest.get(key) == expected, f"{key} must be {expected!r}")
    spec = next((item for item in KNOWN_MODELS if item.model_id == manifest.get("model_id")), None)
    if spec is None:
        raise ValueError("invalid ONNX artifact manifest: model_id is not curated")
    _require(manifest.get("repository") == spec.repository, "repository does not match model_id")
    _require(
        manifest.get("revision") == spec.pinned_revision, "revision is not the pinned revision"
    )
    _require(
        isinstance(manifest.get("source_snapshot_sha256"), str)
        and _SHA256.fullmatch(manifest["source_snapshot_sha256"]) is not None,
        "source_snapshot_sha256 must be a SHA-256 digest",
    )
    _require(
        _is_int(manifest.get("source_snapshot_bytes")) and manifest["source_snapshot_bytes"] > 0,
        "source_snapshot_bytes must be positive",
    )
    _require(
        manifest.get("parameter_count") == spec.parameter_count,
        "parameter_count does not match the curated model",
    )
    license_id = manifest.get("license")
    _require(license_id is None or isinstance(license_id, str), "license must be a string or null")

    validate_opset_imports(_list(manifest.get("opset_imports"), "opset_imports"))

    graph = _dict(manifest.get("graph"), "graph")
    _require(graph.get("path") == GRAPH_FILE, "graph.path mismatch")
    validate_graph_contract(
        _list(graph.get("inputs"), "graph.inputs"), _list(graph.get("outputs"), "graph.outputs")
    )
    sidecars = _list(graph.get("external_data"), "graph.external_data")
    _require(
        all(isinstance(item, str) and item for item in sidecars)
        and len(set(sidecars)) == len(sidecars),
        "graph.external_data must list unique sidecar paths",
    )

    exporter = _dict(manifest.get("exporter"), "exporter")
    expected_exporter = {"gliner_runner", *(name for name, _package in EXPORTER_PACKAGES)}
    _require(
        set(exporter) == expected_exporter,
        "exporter must record gliner_runner and every exporter dependency version",
    )
    _require(
        all(isinstance(value, str) and value for value in exporter.values()),
        "exporter versions must be nonempty strings",
    )
    _require(exporter["gliner2"] == EXPECTED_GLINER2_VERSION, "unexpected GLiNER2 version")

    _validate_parity(manifest.get("parity"))

    files = _list(manifest.get("files"), "files")
    _require(bool(files), "files must be a nonempty list")
    paths: list[str] = []
    for entry in files:
        item = _dict(entry, "file entry")
        _require(set(item) == {"path", "size", "sha256"}, "file entries need path, size, sha256")
        path = item["path"]
        _require(
            isinstance(path, str)
            and bool(path)
            and not path.startswith("/")
            and ".." not in Path(path).parts
            and path != MANIFEST_FILE,
            f"invalid file path {path!r}",
        )
        _require(_is_int(item["size"]) and item["size"] >= 0, f"invalid size for {path}")
        _require(
            isinstance(item["sha256"], str) and _SHA256.fullmatch(item["sha256"]) is not None,
            f"invalid digest for {path}",
        )
        paths.append(path)
    _require(len(set(paths)) == len(paths), "file paths must be unique")
    required = {GRAPH_FILE, PREPROCESSING_FILE, *REQUIRED_TOKENIZER_FILES, *sidecars}
    _require(required <= set(paths), "files must inventory the graph, sidecars, and assets")
    allowed = required | set(OPTIONAL_TOKENIZER_FILES)
    _require(set(paths) <= allowed, "files lists an unexpected artifact file")


def _validate_parity(value: object) -> None:
    parity = _dict(value, "parity")
    required = (
        "absolute_tolerance",
        "max_abs_logit_difference",
        "official_max_abs_logit_difference",
        "onnx_official_max_abs_logit_difference",
        "batch_invariance_max_abs_logit_difference",
        "selected_index_matches",
        "rows",
        "synthetic_zero_threshold",
        "nonzero_differences_zeroed",
    )
    for key in required:
        _require(key in parity, f"parity.{key} is required")
    atol = parity["absolute_tolerance"]
    _require(_is_finite_number(atol) and 0 < atol <= MAX_ATOL, "parity tolerance is out of bounds")
    for key in required[1:5]:
        _require(
            _is_finite_number(parity[key]) and 0 <= parity[key] <= atol,
            f"parity.{key} exceeds the absolute tolerance",
        )
    _require(
        _is_int(parity["rows"]) and parity["rows"] > 0, "parity.rows must be a positive integer"
    )
    _require(
        parity["selected_index_matches"] == parity["rows"],
        "every per-task selection must match",
    )
    _require(
        parity["synthetic_zero_threshold"] == SYNTHETIC_ZERO_THRESHOLD,
        "parity.synthetic_zero_threshold must be 1e-4",
    )
    _require(
        _is_int(parity["nonzero_differences_zeroed"]) and parity["nonzero_differences_zeroed"] >= 0,
        "parity.nonzero_differences_zeroed must be a non-negative integer",
    )


def verify_artifact_directory(root: Path) -> dict[str, Any]:
    """Re-hash an artifact directory against its manifest and return the manifest.

    Rejects a missing, altered, truncated, extra, or symlinked file.
    """
    resolved = root.expanduser().resolve()
    manifest_path = resolved / MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ONNX artifact manifest is unreadable: {manifest_path}") from error
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    validate_manifest(manifest)
    listed = {item["path"]: item for item in manifest["files"]}
    actual = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() or path.is_symlink()
    } - {MANIFEST_FILE}
    if actual != set(listed):
        difference = sorted(actual ^ set(listed))[0]
        raise ValueError(f"ONNX artifact file inventory mismatch: {difference}")
    for relative, item in listed.items():
        path = resolved / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"ONNX artifact file size mismatch: {relative}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"ONNX artifact file checksum mismatch: {relative}")
    return manifest  # type: ignore[no-any-return]


def snapshot_license(snapshot: Path) -> str | None:
    """Return the ``license:`` value from the snapshot README front matter, if any."""
    readme = snapshot / "README.md"
    try:
        text = readme.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if not text.startswith("---"):
        return None
    front_matter = text.split("---", 2)
    if len(front_matter) < 3:
        return None
    match = re.search(r"^license:\s*([A-Za-z0-9._+-]+)\s*$", front_matter[1], re.MULTILINE)
    return match.group(1) if match else None


# --------------------------------------------------------------------------------------
# Heavy export path (torch, onnx, onnxruntime, gliner2 imported lazily)
# --------------------------------------------------------------------------------------


def build_classification_wrapper(torch: Any, extractor: Any) -> Any:
    """Wrap the encoder and classifier head into the exported label-logit module."""

    class RawClassificationOnnx(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = extractor.encoder
            self.classifier = extractor.classifier

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            label_positions: Any,
            label_mask: Any,
        ) -> Any:
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
            safe_positions = label_positions.clamp(0, hidden.shape[1] - 1)
            selected = hidden.gather(
                1,
                safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
            )
            logits = self.classifier(selected).squeeze(-1)
            return logits.masked_fill(~label_mask, torch.finfo(logits.dtype).min)

    return RawClassificationOnnx().eval()


def session_contract(session: Any) -> dict[str, Any]:
    inputs = [
        {"name": item.name, "type": item.type, "shape": list(item.shape)}
        for item in session.get_inputs()
    ]
    outputs = [
        {"name": item.name, "type": item.type, "shape": list(item.shape)}
        for item in session.get_outputs()
    ]
    return validate_graph_contract(inputs, outputs)


def _external_tensor_entries(onnx_model: Any) -> list[tuple[str, int, int]]:
    """Collect ``(location, offset, length)`` for every external tensor in a graph proto."""
    entries: list[tuple[str, int, int]] = []

    def visit_tensor(tensor: Any) -> None:
        if not tensor.external_data:
            return
        values = {entry.key: entry.value for entry in tensor.external_data}
        location = values.get("location")
        if not location:
            raise ValueError("ONNX external tensor has no location")
        entries.append((location, int(values.get("offset", 0)), int(values["length"])))

    def visit_graph(graph: Any) -> None:
        for tensor in graph.initializer:
            visit_tensor(tensor)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    visit_tensor(attribute.t)
                for tensor in attribute.tensors:
                    visit_tensor(tensor)
                if attribute.HasField("g"):
                    visit_graph(attribute.g)
                for subgraph in attribute.graphs:
                    visit_graph(subgraph)

    visit_graph(onnx_model.graph)
    for function in onnx_model.functions:
        for node in function.node:
            for attribute in node.attribute:
                if attribute.HasField("t") and attribute.t.external_data:
                    raise ValueError("ONNX function attributes must not use external data")
    return entries


# --------------------------------------------------------------------------------------
# Canonicalization: exported graph -> template graph + model.onnx.data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalData:
    """Result of canonicalizing an exported graph into the contract layout."""

    tensors: list[dict[str, Any]]
    size: int
    sha256: str


def _numpy_u8(array: Any) -> Any:
    return array.reshape(-1).view("uint8")


def _initializer_payload(
    np: Any, onnx: Any, tensor: Any, root: Path, mapped: dict[str, Any]
) -> Any | None:
    """Return a tensor's raw little-endian bytes as a uint8 array, or ``None`` if unavailable."""
    if tensor.external_data:
        values = {entry.key: entry.value for entry in tensor.external_data}
        location = values.get("location")
        if not location or "length" not in values:
            raise ValueError(
                f"exported tensor {tensor.name!r} has an incomplete external reference"
            )
        if location not in mapped:
            with (root / location).open("rb") as stream:
                mapped[location] = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        return np.frombuffer(
            mapped[location],
            dtype=np.uint8,
            count=int(values["length"]),
            offset=int(values.get("offset", 0)),
        )
    if tensor.raw_data:
        return np.frombuffer(tensor.raw_data, dtype=np.uint8)
    if tensor.data_type == onnx.TensorProto.FLOAT and tensor.float_data:
        return _numpy_u8(np.asarray(tensor.float_data, dtype="<f4"))
    return None


def _initializer_nbytes(np: Any, onnx: Any, tensor: Any, payload: Any | None) -> int:
    if payload is not None:
        return int(payload.size)
    count = 1
    for dimension in tensor.dims:
        count *= int(dimension)
    try:
        itemsize = int(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize)
    except (TypeError, KeyError, ValueError):
        itemsize = 1
    return count * itemsize


def _reject_large_nested_tensors(onnx: Any, graph: Any) -> None:
    """Fail on large tensors outside the top-level initializers; they could not be templated."""

    def visit(candidate: Any, what: str) -> None:
        large = candidate.external_data or len(candidate.raw_data) >= LARGE_TENSOR_BYTES
        if large:
            raise ValueError(f"ONNX {what} tensor {candidate.name!r} is too large to template")

    def walk(subgraph: Any) -> None:
        for node in subgraph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    visit(attribute.t, "node attribute")
                for tensor in attribute.tensors:
                    visit(tensor, "node attribute")
                if attribute.HasField("g"):
                    walk(attribute.g)
                for nested in attribute.graphs:
                    walk(nested)

    if len(graph.sparse_initializer):
        raise ValueError("ONNX sparse initializers are not supported by the template format")
    walk(graph)


def canonicalize_graph(
    *,
    exported_dir: Path,
    weights: Path,
    output_dir: Path,
    log: Any = None,
) -> CanonicalData:
    """Rewrite an exported graph so safetensors-derived weights live in ``model.onnx.data``.

    Every float32 initializer is matched to a ``model.safetensors`` tensor by exact bytes
    (``identity``) or as a 2-D transpose. Matched initializers become external data in graph
    initializer order with 64-byte aligned offsets; unmatched constants stay inline, and any
    unmatched initializer of 64 KiB or more is a hard failure. Writes ``model.onnx`` and
    ``model.onnx.data`` into ``output_dir``.
    """
    emit = log or (lambda _message: None)
    np = importlib.import_module("numpy")
    onnx = importlib.import_module("onnx")
    exported_graph = exported_dir / GRAPH_FILE
    model = onnx.load(str(exported_graph), load_external_data=False)
    _reject_large_nested_tensors(onnx, model.graph)

    index: SafetensorsIndex = read_safetensors_index(weights)
    names = sorted(name for name, item in index.tensors.items() if item.dtype == "F32")
    with weights.open("rb") as stream:
        weights_map = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

    def source_view(name: str) -> Any:
        item = index.tensors[name]
        flat = np.frombuffer(weights_map, dtype="<f4", count=item.element_count, offset=item.start)
        return flat.reshape(item.shape)

    emit(f"indexing {len(names)} safetensors tensors")
    by_hash: dict[str, list[str]] = {}
    by_shape: dict[tuple[int, ...], list[str]] = {}
    for name in names:
        view = source_view(name)
        by_hash.setdefault(hashlib.sha256(_numpy_u8(view)).hexdigest(), []).append(name)
        by_shape.setdefault(tuple(index.tensors[name].shape), []).append(name)
    transposed_hash: dict[str, str] = {}

    def transposed_digest(name: str) -> str:
        if name not in transposed_hash:
            flipped = np.ascontiguousarray(source_view(name).T)
            transposed_hash[name] = hashlib.sha256(_numpy_u8(flipped)).hexdigest()
        return transposed_hash[name]

    mapped: dict[str, Any] = {}
    matches: list[tuple[Any, str, str, Any]] = []
    unmatched_large: list[str] = []
    inlined = 0
    for tensor in model.graph.initializer:
        payload = _initializer_payload(np, onnx, tensor, exported_dir, mapped)
        nbytes = _initializer_nbytes(np, onnx, tensor, payload)
        dims = tuple(int(item) for item in tensor.dims)
        found: tuple[str, str] | None = None
        if (
            payload is not None
            and tensor.data_type == onnx.TensorProto.FLOAT
            and dims
            and nbytes == FLOAT32_ITEMSIZE * math.prod(dims)
        ):
            digest = hashlib.sha256(payload).hexdigest()
            for name in by_hash.get(digest, ()):
                if tuple(index.tensors[name].shape) == dims and np.array_equal(
                    _numpy_u8(source_view(name)), payload
                ):
                    found = (name, "identity")
                    break
            if found is None and len(dims) == 2:
                for name in by_shape.get((dims[1], dims[0]), ()):
                    if transposed_digest(name) == digest and np.array_equal(
                        _numpy_u8(np.ascontiguousarray(source_view(name).T)), payload
                    ):
                        found = (name, "transpose")
                        break
        if found is None:
            if nbytes >= LARGE_TENSOR_BYTES:
                unmatched_large.append(f"{tensor.name} ({nbytes} bytes)")
            continue
        matches.append((tensor, found[0], found[1], payload))
    if unmatched_large:
        shown = ", ".join(unmatched_large[:8])
        raise ValueError(
            f"{len(unmatched_large)} exported initializers of 64 KiB or more do not match "
            f"a model.safetensors tensor (identity or 2-D transpose): {shown}"
        )

    lengths = [int(payload.size) for _tensor, _name, _kind, payload in matches]
    offsets, total = plan_layout(lengths, DATA_ALIGNMENT)
    emit(f"matched {len(matches)} initializers, {total} bytes of external data")

    output_dir.mkdir(parents=True, exist_ok=True)
    digest_all = hashlib.sha256()
    position = 0
    entries: list[dict[str, Any]] = []
    with (output_dir / DATA_OUTPUT).open("wb") as out:
        for (tensor, name, kind, payload), offset in zip(matches, offsets, strict=True):
            padding = offset - position
            if padding:
                out.write(bytes(padding))
                digest_all.update(bytes(padding))
            if kind == "transpose":
                chunk = _numpy_u8(np.ascontiguousarray(source_view(name).T))
            else:
                chunk = payload
            out.write(chunk)
            digest_all.update(chunk)
            position = offset + int(chunk.size)
            source_shape = list(index.tensors[name].shape)
            entries.append(
                {
                    "name": tensor.name,
                    "source": name,
                    "transform": kind,
                    "dtype": "float32",
                    "source_shape": source_shape,
                    "shape": source_shape if kind == "identity" else transposed_shape(source_shape),
                    "offset": offset,
                    "length": int(chunk.size),
                }
            )
    if position != total:
        raise RuntimeError("canonical data size does not match the planned layout")

    matched_names = {item[0].name for item in matches}
    for tensor in model.graph.initializer:
        if tensor.name in matched_names:
            continue
        if tensor.external_data:  # a small constant the exporter externalized: keep it inline
            payload = _initializer_payload(np, onnx, tensor, exported_dir, mapped)
            raw = bytes(payload) if payload is not None else b""
            del tensor.external_data[:]
            tensor.ClearField("data_location")
            tensor.raw_data = raw
            inlined += 1
    for (tensor, _name, _kind, payload), offset in zip(matches, offsets, strict=True):
        length = int(payload.size)
        for field in ("raw_data", "float_data"):
            tensor.ClearField(field)
        del tensor.external_data[:]
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for key, value in (("location", DATA_OUTPUT), ("offset", offset), ("length", length)):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = str(value)
    if inlined:
        emit(f"inlined {inlined} small exporter-externalized constants")
    (output_dir / GRAPH_FILE).write_bytes(model.SerializeToString(deterministic=True))
    return CanonicalData(tensors=entries, size=total, sha256=digest_all.hexdigest())


def build_template(
    *,
    spec: InventoryModelSpec,
    revision: str,
    snapshot: Path,
    artifact: Path,
    canonical: CanonicalData,
    stage: Path,
) -> dict[str, Any]:
    """Write the template package for a finished artifact into ``stage`` and return it."""
    stage.mkdir(parents=True, exist_ok=True)
    source_names = [
        WEIGHTS_FILE,
        *REQUIRED_TOKENIZER_FILES,
        *(name for name in OPTIONAL_TOKENIZER_FILES if (snapshot / name).is_file()),
    ]
    source_files = []
    for name in source_names:
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"model snapshot is missing {name}")
        source_files.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "role": "weights" if name == WEIGHTS_FILE else "tokenizer",
            }
        )

    template_files = []
    for staged_name, output in (
        (TEMPLATE_GRAPH_FILE, GRAPH_OUTPUT),
        (PREPROCESSING_OUTPUT, PREPROCESSING_OUTPUT),
        (MANIFEST_OUTPUT, MANIFEST_OUTPUT),
    ):
        shutil.copyfile(artifact / output, stage / staged_name)
        template_files.append(
            {
                "path": staged_name,
                "size": (stage / staged_name).stat().st_size,
                "sha256": file_sha256(stage / staged_name),
                "output": output,
            }
        )
    data_path = artifact / DATA_OUTPUT
    if data_path.stat().st_size != canonical.size or file_sha256(data_path) != canonical.sha256:
        raise RuntimeError("canonical data file changed after canonicalization")
    outputs = [
        {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
        for item in file_inventory(artifact)
    ]
    outputs.sort(key=lambda item: _OUTPUT_ORDER.get(item["path"], len(_OUTPUT_ORDER)))
    template: dict[str, Any] = {
        "format_version": TEMPLATE_FORMAT_VERSION,
        "kind": TEMPLATE_KIND,
        "model_id": spec.model_id,
        "repository": spec.repository,
        "revision": revision,
        "source_files": source_files,
        "template_files": template_files,
        "data": {
            "path": DATA_OUTPUT,
            "alignment": DATA_ALIGNMENT,
            "size": canonical.size,
            "sha256": canonical.sha256,
            "tensors": canonical.tensors,
        },
        "outputs": outputs,
    }
    validate_template(template)
    manifest = json.loads((artifact / MANIFEST_FILE).read_text(encoding="utf-8"))
    validate_template_manifest(template, manifest)
    (stage / TEMPLATE_FILE).write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return template


def check_assembler_equivalence(
    *, template_dir: Path, snapshot: Path, artifact: Path, scratch_parent: Path
) -> dict[str, Any]:
    """Assemble from the template with the pure-Python reference and compare every byte."""
    scratch = Path(tempfile.mkdtemp(prefix=".assemble-check-", dir=scratch_parent))
    try:
        rebuilt = assemble_from_template(template_dir, snapshot, scratch / "artifact")
        expected = file_inventory(artifact, exclude=())
        actual = file_inventory(rebuilt, exclude=())
        if actual != expected:
            raise RuntimeError("template assembler output differs from the canonical artifact")
        return {"files": len(actual), "verified": [item["path"] for item in actual]}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def exporter_versions() -> dict[str, str]:
    versions = {"gliner_runner": _gliner_runner_version()}
    versions.update({key: _version(package) for key, package in EXPORTER_PACKAGES})
    return versions


def _gliner_runner_version() -> str:
    try:
        return _version("gliner-runner")
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module("gliner_runner")
        return str(module.__version__)


def _configure_offline_environment() -> None:
    for name, value in _OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


def _require_gliner2_version() -> None:
    try:
        installed = _version("gliner2")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "ONNX preparation requires the 'onnx-export' extra: "
            "pip install 'gliner-runner[onnx-export]'"
        ) from error
    if installed != EXPECTED_GLINER2_VERSION:
        raise RuntimeError(
            f"installed GLiNER2 {installed} does not match the pinned {EXPECTED_GLINER2_VERSION}"
        )


def _make_batch(classifier: Any, rows: Sequence[tuple[str, Any]]) -> Any:
    compiled = [classifier.compile_schema(schema) for _text, schema in rows]
    processor_rows = [
        (text, compiled_schema.build())
        for (text, _schema), compiled_schema in zip(rows, compiled, strict=True)
    ]
    return classifier.model.processor.collate_fn_inference(
        processor_rows,
        architecture=classifier.model.architecture,
        error_policy="raise",
    )


def _official_logits(
    classifier: Any,
    module: Any,
    rows: Sequence[tuple[str, Any]],
    fixture_rows: Sequence[FixtureRow],
) -> list[list[float]]:
    scores = classifier.batch_score(
        [text for text, _schema in rows],
        [schema for _text, schema in rows],
        config=module.ClassificationConfig(batch_size=len(rows)),
    )
    flat: list[list[float]] = []
    for score, fixture in zip(scores, fixture_rows, strict=True):
        row: list[float] = []
        for task_name, task in fixture.schema["tasks"].items():
            task_scores = score.tasks.get(task_name)
            labels = list(task["labels"])
            if not isinstance(task_scores, Mapping) or set(task_scores) != set(labels):
                raise RuntimeError(f"official GLiNER2 scorer did not return logits for {task_name}")
            row.extend(float(task_scores[label]) for label in labels)
        flat.append(row)
    return flat


def _pad_rows(
    rows: Sequence[Sequence[float]], masks: Sequence[Sequence[bool]]
) -> list[list[float]]:
    return [
        list(row) + [0.0] * (len(mask) - len(row)) for row, mask in zip(rows, masks, strict=True)
    ]


def _marker_ids(classifier: Any) -> dict[str, int]:
    tokenizer = classifier.model.processor.tokenizer
    return {token: int(tokenizer.convert_tokens_to_ids(token)) for token in SPECIAL_TOKENS}


def _export_graph(
    *,
    torch: Any,
    wrapper: Any,
    example_batch: Any,
    graph_path: Path,
    report_dir: Path,
) -> None:
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
        artifacts_dir=str(report_dir),
    )
    program.save(str(graph_path))


def _batch_inputs(batch: Any) -> tuple[Any, Any, Any, Any]:
    return (
        batch.input_ids,
        batch.attention_mask,
        batch.cls_marker_indices,
        batch.cls_marker_mask,
    )


def _verify_parity(
    *,
    torch: Any,
    session: Any,
    wrapper: Any,
    classifier: Any,
    classification_module: Any,
    spec: InventoryModelSpec,
    atol: float,
) -> dict[str, Any]:
    """Compare ONNX Runtime, the PyTorch wrapper, and the official scorer; fail closed."""
    ort_vs_wrapper: list[dict[str, Any]] = []
    wrapper_vs_official: list[dict[str, Any]] = []
    ort_vs_official: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []
    invariance_maximum = 0.0
    padded_tokens = False
    padded_labels = False
    multi_task = False
    for fixture_rows in fixture_batches(spec):
        rows = [
            (row.text, classification_module.ClassificationSchema.from_dict(row.schema))
            for row in fixture_rows
        ]
        batch = _make_batch(classifier, rows)
        inputs = _batch_inputs(batch)
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
        task_sizes = [row.task_sizes for row in fixture_rows]
        mask_rows = mask.tolist()
        expected_rows = expected.tolist()
        actual_rows = actual.tolist()
        official_rows = _pad_rows(
            _official_logits(classifier, classification_module, rows, fixture_rows), mask_rows
        )
        ort_vs_wrapper.append(
            compare_task_logits(expected_rows, actual_rows, mask_rows, task_sizes)
        )
        wrapper_vs_official.append(
            compare_task_logits(expected_rows, official_rows, mask_rows, task_sizes)
        )
        ort_vs_official.append(
            compare_task_logits(actual_rows, official_rows, mask_rows, task_sizes)
        )

        attention = batch.attention_mask.detach().cpu().numpy()
        padded_tokens = padded_tokens or bool((attention == 0).any())
        padded_labels = padded_labels or bool((~mask).any())
        multi_task = multi_task or any(len(sizes) > 1 for sizes in task_sizes)
        shapes.append(
            {
                "batch": int(actual.shape[0]),
                "sequence": int(attention.shape[1]),
                "labels": int(actual.shape[1]),
                "input_tokens": [int(value) for value in attention.sum(axis=1)],
            }
        )

        if len(rows) > 1:
            for index, single in enumerate(rows):
                single_batch = _make_batch(classifier, [single])
                single_inputs = {
                    name: value.detach().cpu().numpy()
                    for name, value in zip(INPUT_NAMES, _batch_inputs(single_batch), strict=True)
                }
                single_logits = session.run([OUTPUT_NAME], single_inputs)[0][0].tolist()
                count = sum(task_sizes[index])
                invariance_maximum = max(
                    invariance_maximum,
                    *(
                        abs(left - right)
                        for left, right in zip(
                            single_logits[:count], actual_rows[index][:count], strict=True
                        )
                    ),
                )

    if not (padded_tokens and padded_labels and multi_task):
        raise RuntimeError("parity fixtures must exercise token padding, label padding, and tasks")

    def combine(items: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
        return {
            "rows": sum(len(item["selections"]) for item in items),
            "selected_index_matches": sum(item["selected_index_matches"] for item in items),
            "valid_logits_compared": sum(item["valid_logits_compared"] for item in items),
            "max_abs_logit_difference": max(item["max_abs_logit_difference"] for item in items),
            "nonzero_differences_zeroed": sum(item["nonzero_differences_zeroed"] for item in items),
        }

    onnx_wrapper = combine(ort_vs_wrapper)
    official = combine(wrapper_vs_official)
    onnx_official = combine(ort_vs_official)
    for name, summary in (
        ("ONNX/PyTorch-wrapper", onnx_wrapper),
        ("wrapper/official", official),
        ("ONNX/official", onnx_official),
    ):
        if summary["selected_index_matches"] != summary["rows"]:
            raise RuntimeError(f"{name} per-task selected labels do not all match")
        if summary["max_abs_logit_difference"] > atol:
            raise RuntimeError(
                f"{name} logit parity exceeded tolerance: "
                f"{summary['max_abs_logit_difference']} > {atol}"
            )
    if invariance_maximum > atol:
        raise RuntimeError(
            f"ONNX batch/singleton consistency exceeded tolerance: {invariance_maximum} > {atol}"
        )
    parity: dict[str, Any] = {
        "absolute_tolerance": atol,
        "max_abs_logit_difference": onnx_wrapper["max_abs_logit_difference"],
        "official_max_abs_logit_difference": official["max_abs_logit_difference"],
        "onnx_official_max_abs_logit_difference": onnx_official["max_abs_logit_difference"],
        "batch_invariance_max_abs_logit_difference": invariance_maximum,
        "selected_index_matches": onnx_wrapper["selected_index_matches"],
        "rows": onnx_wrapper["rows"],
        "valid_logits_compared": onnx_wrapper["valid_logits_compared"],
        "synthetic_zero_threshold": SYNTHETIC_ZERO_THRESHOLD,
        "nonzero_differences_zeroed": onnx_wrapper["nonzero_differences_zeroed"],
        "fixture_suite": fixture_suite_id(spec),
        "fixture_sha256": fixture_sha256(spec),
        "fixture_shapes": shapes,
    }
    return parity


def _build_artifact(
    *,
    work: Path,
    spec: InventoryModelSpec,
    revision: str,
    snapshot: Path,
    atol: float,
    log: Any,
    template_stage: Path | None = None,
) -> dict[str, Any]:
    onnx = importlib.import_module("onnx")
    ort = importlib.import_module("onnxruntime")
    torch = importlib.import_module("torch")
    classification_module = importlib.import_module("gliner2.classification")

    log("hashing source snapshot")
    snapshot_sha256, snapshot_bytes, _file_count = snapshot_tree_digest(snapshot)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_json = json.loads((snapshot / "tokenizer.json").read_text(encoding="utf-8"))
    marker_ids = special_token_ids_from_tokenizer_json(tokenizer_json)

    log("loading official GLiNER2 classifier")
    classifier = classification_module.Classifier.from_pretrained(
        str(snapshot),
        attention_backend="standard",
        device="cpu",
        dtype="float32",
    ).eval()
    parameter_count = sum(int(item.numel()) for item in classifier.model.parameters())
    if parameter_count != spec.parameter_count:
        raise RuntimeError(
            f"parameter count {parameter_count} does not match curated {spec.parameter_count}"
        )
    if _marker_ids(classifier) != marker_ids:
        raise RuntimeError("tokenizer.json marker ids do not match the loaded GLiNER2 tokenizer")
    hf_tokenizer = classifier.model.processor.tokenizer
    unknown_id = getattr(hf_tokenizer, "unk_token_id", None)
    unknown_token = getattr(hf_tokenizer, "unk_token", None)

    wrapper = build_classification_wrapper(torch, classifier.model)
    example_rows = [
        (row.text, classification_module.ClassificationSchema.from_dict(row.schema))
        for row in fixture_batches(spec)[1]
    ]
    example_batch = _make_batch(classifier, example_rows)
    export_dir = work / ".export"
    export_dir.mkdir()
    graph_path = work / GRAPH_FILE
    log("exporting ONNX graph (Dynamo, opset 20)")
    report_dir = export_dir / "report"
    report_dir.mkdir()
    _export_graph(
        torch=torch,
        wrapper=wrapper,
        example_batch=example_batch,
        graph_path=export_dir / GRAPH_FILE,
        report_dir=report_dir,
    )
    shutil.rmtree(report_dir)
    log("checking exported graph")
    onnx.checker.check_model(str(export_dir / GRAPH_FILE))

    log("canonicalizing graph against model.safetensors")
    canonical = canonicalize_graph(
        exported_dir=export_dir, weights=snapshot / WEIGHTS_FILE, output_dir=work, log=log
    )
    shutil.rmtree(export_dir)

    log("checking canonical graph")
    onnx.checker.check_model(str(graph_path))
    onnx_model = onnx.load(str(graph_path), load_external_data=False)
    opset_imports = validate_opset_imports(
        [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in onnx_model.opset_import
        ]
    )
    external_data_files = validate_external_data_inventory(
        work, _external_tensor_entries(onnx_model)
    )
    if spec.parameter_count * 4 > 2 * 1024**3 and not external_data_files:
        raise RuntimeError("graphs over 2 GiB must use validated ONNX external data")
    del onnx_model

    log("verifying parity with ONNX Runtime on the canonical artifact")
    session = ort.InferenceSession(str(graph_path), providers=[PROVIDER])
    graph = session_contract(session)
    parity = _verify_parity(
        torch=torch,
        session=session,
        wrapper=wrapper,
        classifier=classifier,
        classification_module=classification_module,
        spec=spec,
        atol=atol,
    )
    del session, wrapper, classifier
    gc.collect()

    log("writing assets")
    for name in (*REQUIRED_TOKENIZER_FILES, *OPTIONAL_TOKENIZER_FILES):
        source = snapshot / name
        if source.is_file():
            shutil.copyfile(source, work / name)
        elif name in REQUIRED_TOKENIZER_FILES:
            raise ValueError(f"model snapshot is missing {name}")
    preprocessing = build_preprocessing_config(
        spec=spec,
        revision=revision,
        special_token_ids=marker_ids,
        unknown_token=(
            (str(unknown_token), int(unknown_id))
            if unknown_token is not None and unknown_id is not None
            else None
        ),
        model_config=config,
        tokenizer_config=tokenizer_config,
        has_special_tokens_map=(work / "special_tokens_map.json").is_file(),
    )
    (work / PREPROCESSING_FILE).write_text(
        json.dumps(preprocessing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    log("inventorying files")
    manifest = build_manifest(
        spec=spec,
        revision=revision,
        source_snapshot_sha256=snapshot_sha256,
        source_snapshot_bytes=snapshot_bytes,
        parameter_count=parameter_count,
        graph=graph,
        opset_imports=opset_imports,
        external_data_files=external_data_files,
        exporter=exporter_versions(),
        parity=parity,
        files=file_inventory(work),
        license_id=snapshot_license(snapshot),
    )
    (work / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_artifact_directory(work)

    if template_stage is not None:
        log("building template package")
        build_template(
            spec=spec,
            revision=revision,
            snapshot=snapshot,
            artifact=work,
            canonical=canonical,
            stage=template_stage,
        )
        log("verifying the pure-Python assembler reproduces the artifact")
        check_assembler_equivalence(
            template_dir=template_stage,
            snapshot=snapshot,
            artifact=work,
            scratch_parent=work.parent,
        )
    return manifest


def prepare_onnx(
    model: str,
    *,
    snapshot: Path,
    output_dir: Path,
    revision: str | None = None,
    atol: float = DEFAULT_ATOL,
    template_dir: Path | None = None,
    log: Any = None,
) -> Path:
    """Convert one curated, pinned snapshot into a verified ONNX artifact directory.

    The artifact is built in a sibling temporary directory and atomically renamed into place
    only after every verification passes. An existing artifact is never overwritten.

    The artifact is canonical (see ``gliner_runner.onnx_template``): a template package is
    always built and the pure-Python reference assembler must reproduce the artifact byte for
    byte. With ``template_dir`` the package is also published to
    ``<template-dir>/<owner>--<name>/<revision>/``.
    """
    emit = log or (lambda message: print(f"[prepare-onnx] {message}", file=sys.stderr, flush=True))
    spec, pinned = resolve_curated_model(model, revision)
    atol = validate_atol(atol)
    resolved_snapshot = validate_model_snapshot(snapshot, spec.repository, pinned)
    target = validate_output_location(
        output_dir, resolved_snapshot, artifact_directory(output_dir, spec, pinned)
    )
    template_target: Path | None = None
    if template_dir is not None:
        template_target = (
            template_dir.expanduser().resolve() / spec.repository.replace("/", "--") / pinned
        )
        if template_target.exists() or template_target.is_symlink():
            raise FileExistsError(f"ONNX template path already exists: {template_target}")
    _configure_offline_environment()
    _require_gliner2_version()

    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{pinned[:12]}.tmp-", dir=target.parent))
    stage = Path(tempfile.mkdtemp(prefix=f".{pinned[:12]}.template-", dir=target.parent))
    published_template: Path | None = None
    try:
        _build_artifact(
            work=work,
            spec=spec,
            revision=pinned,
            snapshot=resolved_snapshot,
            atol=atol,
            log=emit,
            template_stage=stage / "template",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"ONNX artifact path already exists: {target}")
        if template_target is not None:
            if not (stage / "template").is_dir():
                raise RuntimeError("template package was not built")
            if template_target.exists() or template_target.is_symlink():
                raise FileExistsError(f"ONNX template path already exists: {template_target}")
            template_target.parent.mkdir(parents=True, exist_ok=True)
            published_template = template_target
            shutil.move(str(stage / "template"), str(template_target))
        work.chmod(0o755)
        work.rename(target)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if published_template is not None:
            shutil.rmtree(published_template, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    emit(f"prepared {target}")
    return target
